"""Proper scoring rules for probabilistic regression.

Implements CRPS (closed-form Gaussian and general), CRPS decomposition
(Hersbach 2000), Winkler interval score, and pinball (quantile) loss.

All functions accept either raw arrays (mean, std for Gaussian) or
PredictiveDistribution objects for generality.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from automl_package.utils.distributions import GaussianDistribution, PredictiveDistribution


def calculate_crps_gaussian(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    """CRPS for Gaussian predictive distributions (Gneiting & Raftery 2007).

    Closed-form: CRPS = σ [z(2Φ(z) - 1) + 2φ(z) - 1/√π], where z = (y - μ)/σ.

    Args:
        y_true: True target values, shape (N,).
        mean: Predicted means, shape (N,).
        std: Predicted standard deviations, shape (N,).

    Returns:
        Mean CRPS across all samples (lower is better).
    """
    y_true = np.asarray(y_true).ravel()
    mean = np.asarray(mean).ravel()
    std = np.maximum(np.asarray(std).ravel(), 1e-9)

    z = (y_true - mean) / std
    crps_per_sample = std * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    return float(np.mean(crps_per_sample))


def calculate_crps(y_true: np.ndarray, dist: PredictiveDistribution, n_quadrature: int = 200) -> float:
    """CRPS for an arbitrary predictive distribution via numerical integration.

    CRPS = integral_{-inf}^{inf} [F(y) - 1(y >= y_true)]^2 dy

    For Gaussian distributions, prefer calculate_crps_gaussian for efficiency.

    Args:
        y_true: True target values, shape (N,).
        dist: A PredictiveDistribution instance.
        n_quadrature: Number of quadrature points for integration.

    Returns:
        Mean CRPS across all samples.
    """
    y_true = np.asarray(y_true).ravel()
    n = len(y_true)

    # Determine integration range from the distribution
    mu = dist.mean
    sd = np.sqrt(np.maximum(dist.variance, 1e-18))
    lo = np.min(np.minimum(mu - 6 * sd, y_true - sd))
    hi = np.max(np.maximum(mu + 6 * sd, y_true + sd))

    grid = np.linspace(lo, hi, n_quadrature)
    dy = grid[1] - grid[0]

    crps_sum = 0.0
    for y_q in grid:
        f_y = dist.cdf(np.full(n, y_q))
        indicator = (y_q >= y_true).astype(float)
        crps_sum += np.mean((f_y - indicator) ** 2) * dy

    return float(crps_sum)


def calculate_crps_decomposition(
    y_true: np.ndarray, dist: PredictiveDistribution, n_bins: int = 20
) -> dict[str, float]:
    """CRPS decomposed into reliability and sharpness (Hersbach 2000).

    Reliability measures calibration; sharpness measures interval width.
    CRPS = reliability - sharpness + uncertainty (uncertainty is data-dependent constant).

    Args:
        y_true: True target values, shape (N,).
        dist: A PredictiveDistribution instance.
        n_bins: Number of bins for the PIT histogram.

    Returns:
        Dict with keys: crps, reliability, sharpness.
    """
    y_true = np.asarray(y_true).ravel()
    n = len(y_true)

    # PIT values
    pit = dist.cdf(y_true)

    # Bin PIT values
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_counts = np.histogram(pit, bins=bin_edges)[0]
    bin_freqs = bin_counts / n

    # Reliability: sum_k (o_k - bar_o_k)^2 * n_k, where o_k = empirical freq, bar_o_k = expected
    target_freqs = np.full(n_bins, 1.0 / n_bins)
    reliability = float(np.sum(n * (bin_freqs - target_freqs) ** 2) / n)

    # Sharpness: mean predictive variance (as a proxy)
    sharpness = float(np.mean(dist.variance))

    # Total CRPS
    if isinstance(dist, GaussianDistribution):
        crps = calculate_crps_gaussian(y_true, dist.mean, np.sqrt(dist.variance))
    else:
        crps = calculate_crps(y_true, dist)

    return {"crps": crps, "reliability": reliability, "sharpness": sharpness}


def calculate_winkler(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float = 0.05
) -> float:
    """Winkler interval score (Winkler 1972).

    W_α(L, U, y) = (U - L) + (2/α)(L - y)1[y < L] + (2/α)(y - U)1[y > U]

    Penalizes both interval width and miscoverage.

    Args:
        y_true: True values, shape (N,).
        lower: Lower interval bounds, shape (N,).
        upper: Upper interval bounds, shape (N,).
        alpha: Nominal miscoverage rate (e.g., 0.05 for 95% intervals).

    Returns:
        Mean Winkler score (lower is better).
    """
    y_true = np.asarray(y_true).ravel()
    lower = np.asarray(lower).ravel()
    upper = np.asarray(upper).ravel()

    width = upper - lower
    penalty_low = (2.0 / alpha) * np.maximum(lower - y_true, 0)
    penalty_high = (2.0 / alpha) * np.maximum(y_true - upper, 0)

    return float(np.mean(width + penalty_low + penalty_high))


def calculate_winkler_from_gaussian(
    y_true: np.ndarray, mean: np.ndarray, std: np.ndarray, alpha: float = 0.05
) -> float:
    """Winkler score using Gaussian intervals at coverage 1-alpha."""
    z = norm.ppf(1 - alpha / 2)
    lower = mean - z * std
    upper = mean + z * std
    return calculate_winkler(y_true, lower, upper, alpha)


def calculate_pinball_loss(y_true: np.ndarray, q_pred: np.ndarray, tau: float) -> float:
    """Pinball (quantile) loss at quantile level tau.

    L_τ(y, q̂) = max(τ(y - q̂), (τ-1)(y - q̂))

    At τ=0.5 this equals MAE.

    Args:
        y_true: True values, shape (N,).
        q_pred: Predicted quantile values, shape (N,).
        tau: Quantile level in (0, 1).

    Returns:
        Mean pinball loss.
    """
    y_true = np.asarray(y_true).ravel()
    q_pred = np.asarray(q_pred).ravel()
    diff = y_true - q_pred
    return float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))
