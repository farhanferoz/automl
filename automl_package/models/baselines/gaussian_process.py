"""Gaussian Process baseline (sklearn GaussianProcessRegressor).

Textbook Bayesian nonparametric baseline. Only practical for small
datasets (O(n^3) training complexity). Include on UCI sets where n <= 5000.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from automl_package.models.baselines.base import BaselineModel


class GaussianProcessModel(BaselineModel):
    """Gaussian Process regression wrapper.

    Args:
        kernel_nu: Smoothness parameter for the Matérn kernel (0.5, 1.5, or 2.5).
        n_restarts_optimizer: Number of restarts for kernel hyperparameter optimization.
        alpha: Noise level added to diagonal (regularization).
    """

    def __init__(
        self,
        kernel_nu: float = 2.5,
        n_restarts_optimizer: int = 5,
        alpha: float = 1e-10,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.kernel_nu = kernel_nu
        self.n_restarts_optimizer = n_restarts_optimizer
        self.alpha = alpha
        self.model_: GaussianProcessRegressor | None = None
        self._scaler: StandardScaler | None = None

    @property
    def name(self) -> str:
        return f"GP(Matern-{self.kernel_nu})"

    def _fit_single(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        forced_iterations: int | None = None,
    ) -> tuple[int, list[float]]:
        # GP benefits from standardized inputs
        self._scaler = StandardScaler()
        x_scaled = self._scaler.fit_transform(x_train)

        kernel = Matern(nu=self.kernel_nu, length_scale_bounds=(1e-3, 1e3)) + WhiteKernel(noise_level_bounds=(1e-5, 1e1))
        self.model_ = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=self.n_restarts_optimizer,
            alpha=self.alpha,
            normalize_y=True,
        )
        self.model_.fit(x_scaled, y_train.ravel())
        return 1, []

    def predict(self, x: np.ndarray | pd.DataFrame, filter_data: bool = True) -> np.ndarray:
        if filter_data:
            x = self._filter_predict_data(x)
        x_scaled = self._scaler.transform(np.asarray(x))
        return self.model_.predict(x_scaled)

    def predict_uncertainty(self, x: np.ndarray | pd.DataFrame, filter_data: bool = True) -> np.ndarray:
        if filter_data:
            x = self._filter_predict_data(x)
        x_scaled = self._scaler.transform(np.asarray(x))
        _, std = self.model_.predict(x_scaled, return_std=True)
        return std

    def get_hyperparameter_search_space(self) -> dict[str, Any]:
        return {
            "kernel_nu": ("categorical", [0.5, 1.5, 2.5]),
            "alpha": ("float", 1e-12, 1e-2, True),
        }
