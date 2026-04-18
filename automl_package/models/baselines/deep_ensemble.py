"""Deep Ensembles baseline (Lakshminarayanan et al., 2017, NeurIPS).

Ensemble of M independently-initialized NNs, each predicting (μ, σ²),
combined as an equal-weighted Gaussian mixture.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from automl_package.models.baselines.base import BaselineModel
from automl_package.utils.distributions import MixtureOfGaussiansDistribution


class DeepEnsemble(BaselineModel):
    """Deep Ensemble wrapper over any model with predict + predict_uncertainty.

    Fits M copies of the base model with different random seeds and
    combines predictions as an equal-weighted mixture.

    Args:
        base_model: A fitted-or-unfitted model instance to clone M times.
            Must support predict() and predict_uncertainty().
        n_members: Number of ensemble members (default 5).
        base_seed: Starting seed; member i uses base_seed + i.
    """

    def __init__(
        self,
        base_model: Any,
        n_members: int = 5,
        base_seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.base_model = base_model
        self.n_members = n_members
        self.base_seed = base_seed
        self.members_: list[Any] = []

    @property
    def name(self) -> str:
        base_name = getattr(self.base_model, "name", self.base_model.__class__.__name__)
        return f"DeepEnsemble(M={self.n_members}, {base_name})"

    def _fit_single(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        forced_iterations: int | None = None,
    ) -> tuple[int, list[float]]:
        self.members_ = []
        for i in range(self.n_members):
            member = self.base_model._clone()
            if hasattr(member, "random_seed"):
                member.random_seed = self.base_seed + i
            member._fit_single(x_train, y_train, x_val, y_val, forced_iterations)
            self.members_.append(member)
        return self.n_members, []

    def _get_member_predictions(self, x: np.ndarray) -> np.ndarray:
        """Get predictions from all members, shape (M, N)."""
        preds = [np.asarray(m.predict(x, filter_data=False)).ravel() for m in self.members_]
        return np.stack(preds, axis=0)

    def predict(self, x: np.ndarray | pd.DataFrame, filter_data: bool = True) -> np.ndarray:
        """Ensemble mean: average of member means."""
        if filter_data:
            x = self._filter_predict_data(x)
        return np.mean(self._get_member_predictions(x), axis=0)

    def predict_uncertainty(self, x: np.ndarray | pd.DataFrame, filter_data: bool = True) -> np.ndarray:
        """Ensemble std via law of total variance across members."""
        if filter_data:
            x = self._filter_predict_data(x)
        means = self._get_member_predictions(x)  # (M, N)

        # If members provide uncertainty, use law of total variance
        if hasattr(self.members_[0], "predict_uncertainty"):
            try:
                stds = np.stack([np.asarray(m.predict_uncertainty(x, filter_data=False)).ravel() for m in self.members_], axis=0)
                mean_of_var = np.mean(stds**2, axis=0)
                var_of_mean = np.var(means, axis=0)
                return np.sqrt(mean_of_var + var_of_mean)
            except (NotImplementedError, RuntimeError):
                pass

        # Fallback: just use variance of means
        return np.std(means, axis=0)

    def predict_distribution(self, x: np.ndarray | pd.DataFrame, filter_data: bool = True) -> MixtureOfGaussiansDistribution:
        """Returns an equal-weighted mixture of Gaussians from ensemble members."""
        if filter_data:
            x = self._filter_predict_data(x)
        means_list = [np.asarray(m.predict(x, filter_data=False)).ravel() for m in self.members_]
        means = np.stack(means_list, axis=1)  # (N, M)
        weights = np.full_like(means, 1.0 / self.n_members)

        try:
            stds_list = [m.predict_uncertainty(x, filter_data=False) for m in self.members_]
            stds = np.stack(stds_list, axis=1)  # (N, M)
        except (NotImplementedError, RuntimeError):
            # Use a small constant if members don't provide uncertainty
            stds = np.full_like(means, np.std(means, axis=1, keepdims=True).clip(min=1e-3))

        return MixtureOfGaussiansDistribution(weights, means, stds)

    def get_num_parameters(self) -> int:
        if not self.members_:
            return 0
        return sum(m.get_num_parameters() for m in self.members_)
