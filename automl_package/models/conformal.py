"""Conformal prediction wrappers for regression models.

Provides distribution-free prediction intervals via split conformal prediction:
  - ConformalWrapper: marginal coverage with constant interval width.
  - LocallyAdaptiveConformalWrapper: conditional coverage by normalizing
    residuals by predicted σ (Lei & Wasserman 2014).
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

    def predict_interval(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (lower, upper) prediction interval for each row in x."""
        if self.quantile is None:
            raise RuntimeError("ConformalWrapper.calibrate() must be called before predict_interval().")
        y_pred = self.model.predict(x)
        y_pred_flat = y_pred.ravel() if y_pred.ndim > 1 else y_pred
        return y_pred_flat - self.quantile, y_pred_flat + self.quantile


class LocallyAdaptiveConformalWrapper:
    """Locally-adaptive conformal prediction (Lei & Wasserman 2014).

    Normalizes residuals by predicted σ to produce conditionally valid
    intervals. On heteroscedastic data, this gives narrower intervals
    in low-noise regions and wider intervals in high-noise regions.

    Requires a model that supports both predict(x) and predict_uncertainty(x).

    Usage:
        model.fit(x_train, y_train)
        lacw = LocallyAdaptiveConformalWrapper(model)
        lacw.calibrate(x_cal, y_cal, alpha=0.1)
        lower, upper = lacw.predict_interval(x_test)
    """

    def __init__(self, model: Any) -> None:
        """Wraps a fitted model with predict() and predict_uncertainty().

        Args:
            model: A fitted regression model with predict(x) and predict_uncertainty(x).
        """
        self.model = model
        self.quantile: float | None = None

    @staticmethod
    def _safe_sigma(sigma: np.ndarray) -> np.ndarray:
        return np.maximum(np.asarray(sigma).ravel(), 1e-9)

    def calibrate(self, x_cal: np.ndarray, y_cal: np.ndarray, alpha: float = 0.1) -> None:
        """Compute the conformity quantile from normalized residuals.

        Nonconformity score: |y - ŷ| / σ̂(x), where σ̂ is the predicted std.
        """
        y_pred = self.model.predict(x_cal)
        sigma = self.model.predict_uncertainty(x_cal)
        y_cal_flat = y_cal.ravel() if y_cal.ndim > 1 else y_cal
        y_pred_flat = y_pred.ravel() if y_pred.ndim > 1 else y_pred

        normalized_residuals = np.abs(y_cal_flat - y_pred_flat) / self._safe_sigma(sigma)

        n = len(normalized_residuals)
        q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        self.quantile = float(np.quantile(normalized_residuals, q_level))

    def predict_interval(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns locally-adaptive (lower, upper) prediction intervals."""
        if self.quantile is None:
            raise RuntimeError("LocallyAdaptiveConformalWrapper.calibrate() must be called before predict_interval().")
        y_pred = self.model.predict(x)
        sigma = self.model.predict_uncertainty(x)
        y_pred_flat = y_pred.ravel() if y_pred.ndim > 1 else y_pred

        half_width = self.quantile * self._safe_sigma(sigma)
        return y_pred_flat - half_width, y_pred_flat + half_width
