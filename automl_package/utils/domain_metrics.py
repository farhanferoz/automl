"""Domain-specific metrics for specialized applications.

Each function group is gated by domain. Currently supported:
  - photo_z: photometric redshift estimation metrics

These metrics are NOT general-purpose and should only be used when
the prediction task matches the domain semantics.
"""

from __future__ import annotations

import numpy as np

from automl_package.utils.distributions import PredictiveDistribution


# --- Photometric Redshift (photo-z) metrics ---


def photo_z_metrics(
    z_true: np.ndarray,
    z_pred: np.ndarray,
    dist: PredictiveDistribution | None = None,
    outlier_threshold: float = 0.15,
) -> dict[str, float]:
    """Standard photometric redshift evaluation metrics.

    All point metrics use the normalized residual Δz/(1+z_spec),
    which is the convention across LSST DESC, DES, KiDS, and SDSS.

    Args:
        z_true: Spectroscopic redshifts (ground truth), shape (N,).
        z_pred: Photometric redshifts (predictions), shape (N,).
        dist: Optional predictive distribution for CDE loss.
        outlier_threshold: Threshold for outlier fraction (default 0.15).

    Returns:
        Dict with sigma_mad, bias, outlier_fraction, and optionally cde_loss.
    """
    z_true = np.asarray(z_true, dtype=np.float64).ravel()
    z_pred = np.asarray(z_pred, dtype=np.float64).ravel()

    dz_norm = _normalized_residuals(z_true, z_pred)

    result: dict[str, float] = {
        "sigma_mad": _sigma_mad(dz_norm),
        "bias": float(np.mean(dz_norm)),
        "outlier_fraction": float(np.mean(np.abs(dz_norm) > outlier_threshold)),
    }

    if dist is not None:
        result["cde_loss"] = _cde_loss(z_true, dist)

    return result


def _normalized_residuals(z_true: np.ndarray, z_pred: np.ndarray) -> np.ndarray:
    """Δz / (1 + z_spec), the standard photo-z normalization."""
    return (z_pred - z_true) / (1 + z_true)


def _sigma_mad(dz_norm: np.ndarray) -> float:
    """Normalized Median Absolute Deviation.

    σ_MAD = 1.4826 × median(|Δz/(1+z)|)

    The 1.4826 factor normalizes MAD to equal σ for Gaussian distributions.
    """
    return float(1.4826 * np.median(np.abs(dz_norm)))


def _cde_loss(z_true: np.ndarray, dist: PredictiveDistribution) -> float:
    """Conditional Density Estimation loss (Izbicki & Lee 2017).

    CDE loss = -mean(log p̂(z_true | x))

    Evaluates the predicted PDF at the true redshift, averaged over samples.
    This is essentially the negative log-likelihood under the full predictive
    distribution (not just a Gaussian approximation).
    """
    log_probs = dist.log_prob(z_true)
    return float(-np.mean(log_probs))
