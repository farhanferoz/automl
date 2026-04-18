"""Calibration metrics and recalibration for probabilistic regression.

Implements PIT-based calibration curve (Kuleshov et al. 2018), ECE,
miscalibration area, sharpness, isotonic recalibration, and PIT
uniformity tests (KS, CvM, AD).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import anderson, cramervonmises, kstest, norm

from automl_package.utils.distributions import GaussianDistribution, PredictiveDistribution


def pit_values(y_true: np.ndarray, dist: PredictiveDistribution) -> np.ndarray:
    """Compute Probability Integral Transform values.

    For a calibrated model, PIT values should be Uniform(0, 1).

    Args:
        y_true: True target values, shape (N,).
        dist: Predictive distribution.

    Returns:
        PIT values, shape (N,).
    """
    return dist.cdf(np.asarray(y_true).ravel())


def pit_values_gaussian(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """PIT values for Gaussian predictions (convenience shortcut)."""
    std = np.maximum(np.asarray(std).ravel(), 1e-9)
    return norm.cdf(np.asarray(y_true).ravel(), loc=np.asarray(mean).ravel(), scale=std)


def calibration_curve(
    y_true: np.ndarray, dist: PredictiveDistribution, n_bins: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """Calibration curve for regression (Kuleshov et al. 2018).

    For each target quantile p, computes the empirical fraction of test
    points with CDF(y_true) <= p. A perfectly calibrated model produces
    (target, empirical) on the diagonal.

    Args:
        y_true: True values, shape (N,).
        dist: Predictive distribution.
        n_bins: Number of evaluation points.

    Returns:
        (target_quantiles, empirical_quantiles), each shape (n_bins+1,).
    """
    pit = pit_values(y_true, dist)
    target = np.linspace(0, 1, n_bins + 1)
    empirical = np.mean(pit[:, np.newaxis] <= target[np.newaxis, :], axis=0)
    return target, empirical


def calibration_curve_gaussian(
    y_true: np.ndarray, mean: np.ndarray, std: np.ndarray, n_bins: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """Calibration curve for Gaussian predictions (convenience shortcut)."""
    dist = GaussianDistribution(mean, std)
    return calibration_curve(y_true, dist, n_bins)


def ece_regression(y_true: np.ndarray, dist: PredictiveDistribution, n_bins: int = 20) -> float:
    """Expected Calibration Error for regression (mean absolute deviation).

    Args:
        y_true: True values, shape (N,).
        dist: Predictive distribution.
        n_bins: Number of evaluation points.

    Returns:
        ECE (lower is better).
    """
    target, empirical = calibration_curve(y_true, dist, n_bins)
    return float(np.mean(np.abs(target - empirical)))


def miscalibration_area(y_true: np.ndarray, dist: PredictiveDistribution, n_bins: int = 100) -> float:
    """Miscalibration area: trapezoidal integral of |empirical - target|.

    More robust than binwise-mean ECE. Values in [0, 0.5].

    Args:
        y_true: True values, shape (N,).
        dist: Predictive distribution.
        n_bins: Number of evaluation points (higher = more precise).

    Returns:
        Miscalibration area (lower is better).
    """
    target, empirical = calibration_curve(y_true, dist, n_bins)
    return float(np.trapezoid(np.abs(empirical - target), target))


def calculate_sharpness(dist: PredictiveDistribution) -> float:
    """Sharpness: mean predictive standard deviation.

    A perfectly calibrated model that always predicts very wide intervals
    is useless; sharpness quantifies how narrow the intervals are.

    Args:
        dist: Predictive distribution.

    Returns:
        Mean predictive standard deviation (lower is sharper).
    """
    return float(np.mean(np.sqrt(np.maximum(dist.variance, 0))))


def calculate_sharpness_from_std(std: np.ndarray) -> float:
    """Sharpness from raw standard deviations."""
    return float(np.mean(np.maximum(np.asarray(std).ravel(), 0)))


def isotonic_recalibrate(pit_cal: np.ndarray, pit_test: np.ndarray) -> np.ndarray:
    """Post-hoc recalibration via isotonic regression (Kuleshov et al. 2018).

    Fits isotonic regression from predicted CDF values to observed
    frequencies on a calibration set, then applies to test PIT values.

    Args:
        pit_cal: PIT values on calibration set, shape (N_cal,).
        pit_test: PIT values on test set, shape (N_test,).

    Returns:
        Recalibrated PIT values for test set, shape (N_test,).
    """
    from sklearn.isotonic import IsotonicRegression

    pit_cal = np.asarray(pit_cal).ravel()
    pit_test = np.asarray(pit_test).ravel()

    # Sort calibration PIT values and compute empirical CDF
    sorted_pit = np.sort(pit_cal)
    empirical_cdf = np.arange(1, len(sorted_pit) + 1) / len(sorted_pit)

    # Fit isotonic regression: maps predicted CDF → observed CDF
    iso_reg = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso_reg.fit(sorted_pit, empirical_cdf)

    return iso_reg.predict(pit_test)


# --- PIT uniformity tests ---


def pit_ks_test(pit: np.ndarray) -> dict[str, float]:
    """Kolmogorov-Smirnov test for PIT uniformity.

    Returns:
        Dict with 'statistic' and 'p_value'. High p_value = good calibration.
    """
    pit = np.asarray(pit).ravel()
    stat, p_value = kstest(pit, "uniform")
    return {"statistic": float(stat), "p_value": float(p_value)}


def pit_cvm_test(pit: np.ndarray) -> dict[str, float]:
    """Cramér-von Mises test for PIT uniformity.

    More sensitive to deviations in the tails than KS.

    Returns:
        Dict with 'statistic' and 'p_value'.
    """
    pit = np.asarray(pit).ravel()
    result = cramervonmises(pit, "uniform")
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}


def pit_ad_test(pit: np.ndarray) -> dict[str, float]:
    """Anderson-Darling test for PIT uniformity.

    Transforms PIT ~ U(0,1) to Z ~ N(0,1) via Φ^{-1}, then applies the
    AD test for normality. This is equivalent to testing PIT uniformity
    and is the standard workaround since scipy.anderson does not support
    the uniform distribution directly.

    Returns:
        Dict with 'statistic' and 'critical_values' (dict of significance_level: critical_value).
    """
    pit = np.asarray(pit).ravel()
    # Transform U(0,1) → N(0,1) via inverse normal CDF; clip to avoid ±inf
    z = norm.ppf(np.clip(pit, 1e-10, 1 - 1e-10))
    try:
        # scipy >= 1.17 with method parameter
        result = anderson(z, dist="norm", method="interpolate")
        return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}
    except TypeError:
        # scipy < 1.17 without method parameter
        result = anderson(z, dist="norm")
        critical_values = {f"{sl}%": float(cv) for sl, cv in zip(result.significance_level, result.critical_values, strict=False)}
        return {"statistic": float(result.statistic), "critical_values": critical_values}


def calculate_picp_at_alphas(
    y_true: np.ndarray, mean: np.ndarray, std: np.ndarray, alphas: tuple[float, ...] = (0.05, 0.1, 0.2, 0.32)
) -> dict[str, float]:
    """Prediction Interval Coverage Probability at multiple coverage levels.

    Args:
        y_true: True values.
        mean: Predicted means.
        std: Predicted standard deviations.
        alphas: Miscoverage rates (e.g., 0.05 → 95% interval).

    Returns:
        Dict mapping 'picp@{coverage}' to empirical coverage fraction.
    """
    y_true = np.asarray(y_true).ravel()
    mean = np.asarray(mean).ravel()
    std = np.maximum(np.asarray(std).ravel(), 1e-9)

    result = {}
    for alpha in alphas:
        z = norm.ppf(1 - alpha / 2)
        lower = mean - z * std
        upper = mean + z * std
        coverage = float(np.mean((y_true >= lower) & (y_true <= upper)))
        result[f"picp@{int((1 - alpha) * 100)}"] = coverage
    return result


def calculate_mpiw(std: np.ndarray, alpha: float = 0.05) -> float:
    """Mean Prediction Interval Width at coverage 1-alpha.

    Args:
        std: Predicted standard deviations.
        alpha: Miscoverage rate.

    Returns:
        Mean interval width.
    """
    std = np.maximum(np.asarray(std).ravel(), 1e-9)
    z = norm.ppf(1 - alpha / 2)
    return float(np.mean(2 * z * std))
