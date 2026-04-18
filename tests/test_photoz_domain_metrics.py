"""Sanity tests for photo-z domain metrics and LocallyAdaptiveConformalWrapper."""

from __future__ import annotations

import numpy as np
import pytest

from automl_package.models.conformal import ConformalWrapper, LocallyAdaptiveConformalWrapper
from automl_package.utils.distributions import GaussianDistribution
from automl_package.utils.domain_metrics import photo_z_metrics


class _ToyModel:
    """Model with configurable mean and (optional) heteroscedastic sigma."""

    def __init__(self, sigma: np.ndarray | float) -> None:
        self.sigma = sigma

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x.ravel()  # identity

    def predict_uncertainty(self, x: np.ndarray) -> np.ndarray:
        if np.ndim(self.sigma) == 0:
            return np.full(x.shape[0], float(self.sigma))
        return np.asarray(self.sigma).ravel()


class TestPhotoZMetrics:
    def test_perfect_predictions_give_zero_stats(self):
        z = np.linspace(0.1, 2.0, 100)
        m = photo_z_metrics(z, z)
        assert m["sigma_mad"] == pytest.approx(0.0, abs=1e-9)
        assert m["bias"] == pytest.approx(0.0, abs=1e-9)
        assert m["outlier_fraction"] == 0.0

    def test_bias_is_mean_normalized_residual(self):
        z = np.full(200, 1.0)
        z_pred = z + 0.1  # constant bias of 0.1 / (1+1) = 0.05
        m = photo_z_metrics(z, z_pred)
        assert m["bias"] == pytest.approx(0.05, abs=1e-9)

    def test_outlier_fraction_counts_above_threshold(self):
        rng = np.random.default_rng(0)
        z = rng.uniform(0.1, 1.0, 500)
        z_pred = z + rng.choice([0.0, 1.0], size=500, p=[0.9, 0.1])  # ~10% outliers
        m = photo_z_metrics(z, z_pred, outlier_threshold=0.15)
        assert 0.05 < m["outlier_fraction"] < 0.2

    def test_cde_loss_uses_distribution_log_prob(self):
        z = np.linspace(0.1, 1.0, 100)
        z_pred = z  # perfect mean
        dist = GaussianDistribution(z, np.full_like(z, 0.05))
        m = photo_z_metrics(z, z_pred, dist=dist)
        assert "cde_loss" in m
        # Analytical: -log N(0; 0, sigma) = 0.5*log(2 pi sigma^2). For sigma=0.05, -log ~ -log(1/(sqrt(2pi)*0.05))
        expected = 0.5 * np.log(2 * np.pi * 0.05 ** 2)
        assert m["cde_loss"] == pytest.approx(expected, abs=1e-4)


class TestLocallyAdaptiveConformal:
    def _heteroscedastic(self, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        x = rng.uniform(-3.0, 3.0, 1200).reshape(-1, 1)
        sigma = 0.1 + 0.5 * np.abs(x.ravel())
        y = x.ravel() + rng.normal(0.0, sigma)
        return x.astype(np.float32), y.astype(np.float32), sigma.astype(np.float32)

    def test_marginal_coverage_close_to_target(self):
        x, y, sigma = self._heteroscedastic()
        x_cal, x_te = x[:600], x[600:]
        y_cal, y_te = y[:600], y[600:]
        sig_cal, sig_te = sigma[:600], sigma[600:]
        model_cal = _ToyModel(sig_cal)
        wrapper = LocallyAdaptiveConformalWrapper(model_cal)
        wrapper.calibrate(x_cal, y_cal, alpha=0.1)
        model_te = _ToyModel(sig_te)
        wrapper.model = model_te
        low, high = wrapper.predict_interval(x_te)
        coverage = float(np.mean((y_te >= low) & (y_te <= high)))
        # Coverage should be within 0.85 - 0.95 band
        assert 0.82 < coverage < 0.98, f"marginal coverage {coverage:.3f} out of band"

    def test_locally_adaptive_narrower_in_low_noise_region(self):
        """Interval in low-noise region should be narrower than in high-noise region."""
        x, y, sigma = self._heteroscedastic()
        model = _ToyModel(sigma)
        wrapper = LocallyAdaptiveConformalWrapper(model)
        wrapper.calibrate(x, y, alpha=0.1)
        x_low = np.array([[0.0], [0.1], [-0.1]], dtype=np.float32)
        x_high = np.array([[3.0], [-3.0], [2.8]], dtype=np.float32)
        model_low = _ToyModel(np.full(3, 0.1))
        wrapper.model = model_low
        low_lo, low_hi = wrapper.predict_interval(x_low)
        model_high = _ToyModel(np.full(3, 1.6))
        wrapper.model = model_high
        hi_lo, hi_hi = wrapper.predict_interval(x_high)
        low_width = (low_hi - low_lo).mean()
        high_width = (hi_hi - hi_lo).mean()
        assert high_width > low_width * 2, (
            f"Locally adaptive widths should scale with sigma: low={low_width:.3f} high={high_width:.3f}"
        )

    def test_marginal_conformal_also_covers(self):
        x, y, sigma = self._heteroscedastic()
        x_cal, x_te = x[:600], x[600:]
        y_cal, y_te = y[:600], y[600:]
        model = _ToyModel(0.0)  # prediction = x
        wrapper = ConformalWrapper(model)
        wrapper.calibrate(x_cal, y_cal, alpha=0.1)
        low, high = wrapper.predict_interval(x_te)
        coverage = float(np.mean((y_te >= low) & (y_te <= high)))
        assert 0.82 < coverage < 0.98
