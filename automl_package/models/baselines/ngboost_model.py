"""NGBoost baseline wrapper (Duan et al., 2020, ICML).

Natural gradient boosting producing parametric predictive distributions.
Install: pip install ngboost
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from automl_package.models.baselines.base import BaselineModel


class NGBoostModel(BaselineModel):
    """NGBoost wrapper for probabilistic regression.

    Args:
        n_estimators: Number of boosting stages.
        learning_rate: Shrinkage parameter.
        minibatch_frac: Fraction of data used per boosting iteration.
        natural_gradient: Whether to use natural gradient (default True).
    """

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.01,
        minibatch_frac: float = 1.0,
        natural_gradient: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.minibatch_frac = minibatch_frac
        self.natural_gradient = natural_gradient
        self.model_: Any = None

    @property
    def name(self) -> str:
        return "NGBoost"

    def _fit_single(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        forced_iterations: int | None = None,
    ) -> tuple[int, list[float]]:
        from ngboost import NGBRegressor
        from ngboost.distns import Normal

        n_est = forced_iterations or self.n_estimators
        self.model_ = NGBRegressor(
            n_estimators=n_est,
            learning_rate=self.learning_rate,
            minibatch_frac=self.minibatch_frac,
            natural_gradient=self.natural_gradient,
            Dist=Normal,
            verbose=False,
        )

        fit_kwargs: dict[str, Any] = {}
        if x_val is not None and y_val is not None:
            fit_kwargs["X_val"] = x_val
            fit_kwargs["Y_val"] = y_val.ravel()
            fit_kwargs["early_stopping_rounds"] = self.early_stopping_rounds or 50

        self.model_.fit(x_train, y_train.ravel(), **fit_kwargs)
        return n_est, []

    def predict(self, x: np.ndarray | pd.DataFrame, filter_data: bool = True) -> np.ndarray:
        if filter_data:
            x = self._filter_predict_data(x)
        x = np.asarray(x)
        return self.model_.predict(x)

    def predict_uncertainty(self, x: np.ndarray | pd.DataFrame, filter_data: bool = True) -> np.ndarray:
        if filter_data:
            x = self._filter_predict_data(x)
        x = np.asarray(x)
        dist = self.model_.pred_dist(x)
        return dist.params["scale"]

    def get_hyperparameter_search_space(self) -> dict[str, Any]:
        return {
            "n_estimators": ("int", 100, 2000),
            "learning_rate": ("float", 1e-3, 0.3, True),
            "minibatch_frac": ("float", 0.5, 1.0),
        }

    def get_num_parameters(self) -> int:
        if self.model_ is None:
            return 0
        return self.n_estimators
