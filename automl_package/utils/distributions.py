"""Predictive distribution abstractions for probabilistic regression evaluation.

Provides a Protocol defining the interface that all predictive distributions must
satisfy, plus concrete implementations for the most common cases: Gaussian,
mixture-of-Gaussians, empirical (sample-based), and quantile.
"""

from __future__ import annotations

import warnings
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


@runtime_checkable
class PredictiveDistribution(Protocol):
    """Interface for predictive distributions used by scoring/calibration modules.

    Every method operates element-wise over a batch of N predictions.
    Array shapes: mean/variance are (N,), cdf/ppf/log_prob accept/return (N,).
    """

    def cdf(self, y: np.ndarray) -> np.ndarray:
        """Cumulative distribution function evaluated at y."""
        ...

    def ppf(self, q: np.ndarray) -> np.ndarray:
        """Percent-point (quantile) function evaluated at probability q."""
        ...

    def log_prob(self, y: np.ndarray) -> np.ndarray:
        """Log probability density evaluated at y."""
        ...

    @property
    def mean(self) -> np.ndarray:
        """Predictive mean, shape (N,)."""
        ...

    @property
    def variance(self) -> np.ndarray:
        """Predictive variance, shape (N,)."""
        ...


class GaussianDistribution:
    """Univariate Gaussian N(mu, sigma^2) per sample."""

    def __init__(self, mu: np.ndarray, sigma: np.ndarray) -> None:
        self._mu = np.asarray(mu, dtype=np.float64).ravel()
        self._sigma = np.maximum(np.asarray(sigma, dtype=np.float64).ravel(), 1e-9)

    def cdf(self, y: np.ndarray) -> np.ndarray:
        return norm.cdf(y, loc=self._mu, scale=self._sigma)

    def ppf(self, q: np.ndarray) -> np.ndarray:
        return norm.ppf(q, loc=self._mu, scale=self._sigma)

    def log_prob(self, y: np.ndarray) -> np.ndarray:
        return norm.logpdf(y, loc=self._mu, scale=self._sigma)

    @property
    def mean(self) -> np.ndarray:
        return self._mu

    @property
    def variance(self) -> np.ndarray:
        return self._sigma**2


class MixtureOfGaussiansDistribution:
    """Mixture of K Gaussians per sample: p(y|x) = sum_k w_k N(y; mu_k, sigma_k^2).

    Args:
        weights: (N, K) mixture weights summing to 1 along axis=1.
        means: (N, K) component means.
        stds: (N, K) component standard deviations.
    """

    def __init__(self, weights: np.ndarray, means: np.ndarray, stds: np.ndarray) -> None:
        self._weights = np.asarray(weights, dtype=np.float64)
        self._means = np.asarray(means, dtype=np.float64)
        self._stds = np.maximum(np.asarray(stds, dtype=np.float64), 1e-9)
        if self._weights.ndim == 1:
            # Single-sample case: reshape to (1, K)
            self._weights = self._weights[np.newaxis, :]
            self._means = self._means[np.newaxis, :]
            self._stds = self._stds[np.newaxis, :]

    def cdf(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y).ravel()
        # (N, K) component CDFs
        component_cdfs = norm.cdf(y[:, np.newaxis], loc=self._means, scale=self._stds)
        return np.sum(self._weights * component_cdfs, axis=1)

    def ppf(self, q: np.ndarray) -> np.ndarray:
        """Numerical inversion of CDF via Brent's method, per sample."""
        q = np.asarray(q).ravel()
        n = len(q)
        result = np.empty(n)

        # Bracket: use component means ± wide range
        lo = np.min(self._means - 6 * self._stds, axis=1)
        hi = np.max(self._means + 6 * self._stds, axis=1)

        for i in range(n):
            w_i, m_i, s_i = self._weights[i], self._means[i], self._stds[i]

            def _cdf_minus_q(y: float) -> float:
                return float(np.sum(w_i * norm.cdf(y, loc=m_i, scale=s_i)) - q[i])

            try:
                result[i] = brentq(_cdf_minus_q, lo[i], hi[i], xtol=1e-8)
            except ValueError:
                # Fallback: return weighted mean if bracket fails
                result[i] = float(np.sum(w_i * m_i))
        return result

    def log_prob(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y).ravel()
        # (N, K) component log-pdfs
        component_logpdfs = norm.logpdf(y[:, np.newaxis], loc=self._means, scale=self._stds)
        # log-sum-exp with weights for numerical stability
        log_weights = np.log(np.maximum(self._weights, 1e-30))
        return _logsumexp(log_weights + component_logpdfs, axis=1)

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @property
    def means(self) -> np.ndarray:
        return self._means

    @property
    def stds(self) -> np.ndarray:
        return self._stds

    @property
    def mean(self) -> np.ndarray:
        return np.sum(self._weights * self._means, axis=1)

    @property
    def variance(self) -> np.ndarray:
        # Law of total variance: Var = E[Var_k] + Var[E_k]
        mean_of_var = np.sum(self._weights * self._stds**2, axis=1)
        var_of_mean = np.sum(self._weights * (self._means - self.mean[:, np.newaxis]) ** 2, axis=1)
        return mean_of_var + var_of_mean


class EmpiricalDistribution:
    """Sample-based predictive distribution (e.g., from MC Dropout or ensembles).

    Args:
        samples: (N, S) array where S is the number of Monte Carlo samples per input.
    """

    def __init__(self, samples: np.ndarray) -> None:
        self._samples = np.asarray(samples, dtype=np.float64)
        if self._samples.ndim == 1:
            self._samples = self._samples[:, np.newaxis]
        # Pre-sort for efficient quantile computation
        self._sorted = np.sort(self._samples, axis=1)

    def cdf(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y).ravel()
        # Fraction of samples <= y, per input
        return np.mean(self._sorted <= y[:, np.newaxis], axis=1)

    def ppf(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q).ravel()
        n_samples = self._sorted.shape[1]
        indices = np.clip((q * n_samples).astype(int), 0, n_samples - 1)
        return self._sorted[np.arange(len(q)), indices]

    def log_prob(self, y: np.ndarray) -> np.ndarray:
        warnings.warn("EmpiricalDistribution.log_prob uses KDE approximation.", stacklevel=2)
        y = np.asarray(y).ravel()
        # Silverman bandwidth per sample
        stds = np.std(self._samples, axis=1)
        n_s = self._samples.shape[1]
        bw = 1.06 * np.maximum(stds, 1e-9) * n_s ** (-0.2)
        # KDE log-density
        diff = (y[:, np.newaxis] - self._samples) / bw[:, np.newaxis]
        log_components = -0.5 * diff**2 - np.log(bw[:, np.newaxis]) - 0.5 * np.log(2 * np.pi)
        return _logsumexp(log_components, axis=1) - np.log(n_s)

    @property
    def mean(self) -> np.ndarray:
        return np.mean(self._samples, axis=1)

    @property
    def variance(self) -> np.ndarray:
        return np.var(self._samples, axis=1)


class QuantileDistribution:
    """Distribution defined by predicted quantiles (e.g., from quantile regression).

    Uses linear interpolation between predicted quantiles for CDF/PPF.

    Args:
        quantile_levels: (Q,) sorted quantile levels, e.g., [0.05, 0.25, 0.5, 0.75, 0.95].
        quantile_values: (N, Q) predicted quantile values.
    """

    def __init__(self, quantile_levels: np.ndarray, quantile_values: np.ndarray) -> None:
        self._levels = np.asarray(quantile_levels, dtype=np.float64)
        self._values = np.asarray(quantile_values, dtype=np.float64)
        if self._values.ndim == 1:
            self._values = self._values[np.newaxis, :]

    def cdf(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y).ravel()
        n = len(y)
        result = np.empty(n)
        for i in range(n):
            result[i] = np.interp(y[i], self._values[i], self._levels, left=0.0, right=1.0)
        return result

    def ppf(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q).ravel()
        n = len(q)
        result = np.empty(n)
        for i in range(n):
            result[i] = np.interp(q[i], self._levels, self._values[i])
        return result

    def log_prob(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError("QuantileDistribution does not support log_prob. Use CDF-based metrics (CRPS, calibration) instead.")

    @property
    def mean(self) -> np.ndarray:
        # Use median (0.5 quantile) as point estimate
        return np.array([np.interp(0.5, self._levels, self._values[i]) for i in range(self._values.shape[0])])

    @property
    def variance(self) -> np.ndarray:
        # Approximate from IQR: Var ≈ (IQR / 1.35)^2
        q25 = np.array([np.interp(0.25, self._levels, self._values[i]) for i in range(self._values.shape[0])])
        q75 = np.array([np.interp(0.75, self._levels, self._values[i]) for i in range(self._values.shape[0])])
        iqr = q75 - q25
        return (iqr / 1.35) ** 2


def _logsumexp(a: np.ndarray, axis: int = 1) -> np.ndarray:
    """Numerically stable log-sum-exp."""
    a_max = np.max(a, axis=axis, keepdims=True)
    return np.log(np.sum(np.exp(a - a_max), axis=axis)) + a_max.squeeze(axis=axis)
