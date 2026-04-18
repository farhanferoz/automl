"""Conformalized Quantile Regression (Romano et al., 2019, NeurIPS).

Combines quantile regression with conformal prediction for
distribution-free locally-adaptive intervals.
"""

from __future__ import annotations

import numpy as np


class CQRWrapper:
    """Conformalized Quantile Regression wrapper.

    Wraps a fitted quantile regressor and produces locally-adaptive
    prediction intervals with distribution-free coverage guarantees.

    Usage:
        qr_model.fit(x_train, y_train)
        cqr = CQRWrapper(qr_model, alpha=0.1)
        cqr.calibrate(x_cal, y_cal)
        lower, upper = cqr.predict_interval(x_test)

    Args:
        model: A fitted model with a predict_quantiles(x) method returning
            shape (N, Q) or a predict method that can produce lower/upper quantiles.
        alpha: Miscoverage rate (default 0.1 for 90% coverage).
        quantile_low: Lower quantile level (default alpha/2).
        quantile_high: Upper quantile level (default 1 - alpha/2).
    """

    def __init__(
        self,
        model: object,
        alpha: float = 0.1,
        quantile_low: float | None = None,
        quantile_high: float | None = None,
    ) -> None:
        self.model = model
        self.alpha = alpha
        self.quantile_low = quantile_low or alpha / 2
        self.quantile_high = quantile_high or 1 - alpha / 2
        self.q_correction: float | None = None

    def _get_bounds(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Get lower and upper quantile predictions from the model."""
        if hasattr(self.model, "predict_quantiles"):
            q_vals = self.model.predict_quantiles(x, filter_data=False)
            quantiles = self.model.quantiles
            # Interpolate to get the desired quantile levels
            n = q_vals.shape[0]
            lower = np.array([np.interp(self.quantile_low, quantiles, q_vals[i]) for i in range(n)])
            upper = np.array([np.interp(self.quantile_high, quantiles, q_vals[i]) for i in range(n)])
        else:
            raise TypeError("Model must have a predict_quantiles method or be a QuantileRegressionNN.")
        return lower, upper

    def calibrate(self, x_cal: np.ndarray, y_cal: np.ndarray) -> None:
        """Compute the conformal correction from a held-out calibration set.

        The nonconformity score is: E_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i))
        The correction Q is the ceil((n+1)(1-alpha))/n quantile of {E_i}.
        """
        y_cal = np.asarray(y_cal).ravel()
        lower, upper = self._get_bounds(x_cal)
        scores = np.maximum(lower - y_cal, y_cal - upper)

        n = len(scores)
        q_level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.q_correction = float(np.quantile(scores, q_level))

    def predict_interval(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (lower, upper) conformalized prediction intervals."""
        if self.q_correction is None:
            raise RuntimeError("CQRWrapper.calibrate() must be called before predict_interval().")
        lower, upper = self._get_bounds(x)
        return lower - self.q_correction, upper + self.q_correction
