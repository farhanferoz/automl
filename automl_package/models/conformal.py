"""Conformal prediction wrapper for regression models.

Provides distribution-free prediction intervals via split conformal prediction.
"""

from typing import Any

import numpy as np


class ConformalWrapper:
    """Split conformal prediction wrapper for any regression model.

    Wraps a fitted regression model and produces prediction intervals with
    coverage guarantee >= 1-α on exchangeable data, regardless of the underlying
    model's assumptions.

    Usage:
        model.fit(x_train, y_train)
        cw = ConformalWrapper(model)
        cw.calibrate(x_cal, y_cal, alpha=0.1)  # target 90% coverage
        lower, upper = cw.predict_interval(x_test)
    """

    def __init__(self, model: Any) -> None:
        """Wraps a fitted regression model.

        Args:
            model: A fitted regression model with a `predict(x)` method.
        """
        self.model = model
        self.quantile: float | None = None
        self.alpha: float | None = None

    def calibrate(self, x_cal: np.ndarray, y_cal: np.ndarray, alpha: float = 0.1) -> None:
        """Compute the conformity quantile from a held-out calibration set.

        Args:
            x_cal: Calibration features.
            y_cal: Calibration targets.
            alpha: Miscoverage rate. Coverage target is 1-α.
        """
        y_pred = self.model.predict(x_cal)
        y_cal_flat = y_cal.ravel() if y_cal.ndim > 1 else y_cal
        y_pred_flat = y_pred.ravel() if y_pred.ndim > 1 else y_pred
        residuals = np.abs(y_cal_flat - y_pred_flat)

        # Finite-sample correction: use ceil((n+1)(1-α)) / n quantile
        n = len(residuals)
        q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        self.quantile = float(np.quantile(residuals, q_level))
        self.alpha = alpha

    def predict_interval(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (lower, upper) prediction interval for each row in x."""
        if self.quantile is None:
            raise RuntimeError("ConformalWrapper.calibrate() must be called before predict_interval().")
        y_pred = self.model.predict(x)
        y_pred_flat = y_pred.ravel() if y_pred.ndim > 1 else y_pred
        return y_pred_flat - self.quantile, y_pred_flat + self.quantile
