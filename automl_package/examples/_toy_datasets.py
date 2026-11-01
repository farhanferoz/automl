"""Shared toy datasets for sweep scripts."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import log_ndtr


def make_datasets() -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Four canonical toy datasets used across ProbReg and ClassReg sweeps."""
    rng = np.random.default_rng(42)

    x = rng.uniform(-5, 5, 800).reshape(-1, 1).astype(np.float32)
    y = (np.sin(x.ravel()) * 2 + 0.5 * x.ravel() + rng.normal(0.0, 0.1 + 0.4 * np.abs(x.ravel()))).astype(np.float32)
    datasets = [("heteroscedastic", x, y)]

    x = rng.uniform(-3, 3, 800).reshape(-1, 1).astype(np.float32)
    sign = rng.choice([-1.0, 1.0], size=800).reshape(-1, 1)
    y = (x + sign * 1.5 + rng.normal(0, 0.1, (800, 1))).ravel().astype(np.float32)
    datasets.append(("bimodal", x, y))

    x = rng.uniform(-5.0, 5.0, 800).reshape(-1, 1).astype(np.float32)
    y_true = np.where(x.ravel() < 0, 0.5 * x.ravel(), 0.5 * x.ravel() + np.sin(4 * np.pi * x.ravel()))
    y = (y_true + rng.normal(0, 0.2, 800)).astype(np.float32)
    datasets.append(("piecewise", x, y))

    x = rng.uniform(-3, 3, 600).reshape(-1, 1).astype(np.float32)
    y = (np.exp(x.ravel()) + rng.normal(0.0, 0.5, 600)).astype(np.float32)
    datasets.append(("exponential", x, y))

    return datasets


def make_toy_a(n: int = 800, sigma: float = 0.5, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Toy A: smooth unimodal with controlled homoscedastic noise.

    y = sin(2π x) + σ · ε,   x ∈ [0, 1],   ε ~ N(0, 1)

    The true mean is sin(2π x) and the noise variance is σ², constant in x.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n).reshape(-1, 1).astype(np.float32)
    y = (np.sin(2 * np.pi * x.ravel()) + sigma * rng.normal(0.0, 1.0, n)).astype(np.float32)
    return x, y


def make_toy_b(
    n: int = 800,
    k_true: int = 3,
    separation: float = 4.0,
    sigma: float = 0.3,
    seed: int = 42,
    baseline: str = "sin",
) -> tuple[np.ndarray, np.ndarray]:
    """Toy B: conditional Gaussian mixture with KNOWN intrinsic k.

    For each x, y is drawn from a uniform mixture of k_true Gaussians whose
    means are evenly spaced and shifted by f(x):

        μ_j(x) = base(x) + (j - (k_true - 1)/2) · separation · σ
        y | x ~ (1 / k_true) Σ_j N(μ_j(x), σ²)

    Components are equiseparated by `separation·σ`, so `separation` is the
    resolvability dial (mode spacing in noise-widths): at separation ≥ 4 the
    modes are clearly distinguishable; below ~1 they merge into one blob.

    The baseline controls whether the mixture moves with x:
      - "sin"  : base(x) = sin(2π x)  — conditional mixture (the mixture location
        sweeps with x; exercises the shared network / per-input regime).
      - "zero" : base(x) = 0          — the mixture sits still, so the data are
        i.i.d. draws from one fixed mixture. This is the grounding (Basis A)
        regime where the α₀ pruning theory applies; use it to isolate mode-count
        from range-tiling.

    Args:
        n: sample count.
        k_true: ground-truth number of mixture components.
        separation: per-component spacing in units of σ (the resolvability dial).
        sigma: per-component noise σ.
        seed: random seed.
        baseline: "sin" (moving) or "zero" (fixed location).

    Returns:
        (x, y) with shapes (n, 1) and (n,), both float32.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n).reshape(-1, 1).astype(np.float32)
    base = np.sin(2 * np.pi * x.ravel()) if baseline == "sin" else np.zeros(n, dtype=np.float64)
    component_idx = rng.integers(0, k_true, size=n)
    offsets = (component_idx - (k_true - 1) / 2.0) * separation * sigma
    y = (base + offsets + sigma * rng.normal(0.0, 1.0, n)).astype(np.float32)
    return x, y


def make_broad_unimodal(
    n: int = 800,
    separation: float = 4.0,
    sigma: float = 0.3,
    seed: int = 42,
    baseline: str = "zero",
) -> tuple[np.ndarray, np.ndarray]:
    """Single Gaussian matched in mean AND variance to the k_true=2 bimodal toy.

    The two-mode toy ``make_toy_b(k_true=2, separation, sigma)`` has modes at
    base ± 0.5·separation·σ with within-mode variance σ², so its total variance
    is σ²·(1 + separation²/4) and its mean is base. This generator draws a single
    Gaussian with that SAME mean and variance, so it differs from the bimodal toy
    only in shape (one broad bell vs two peaks). The blend-of-summaries objective
    cannot tell them apart; the genuine mixture objective can (note §9, Check 3).

    Args:
        n: sample count.
        separation: matches the bimodal toy's separation (sets the variance).
        sigma: matches the bimodal toy's per-mode σ.
        seed: random seed.
        baseline: "sin" or "zero" (must match the bimodal partner).

    Returns:
        (x, y) with shapes (n, 1) and (n,), both float32.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n).reshape(-1, 1).astype(np.float32)
    base = np.sin(2 * np.pi * x.ravel()) if baseline == "sin" else np.zeros(n, dtype=np.float64)
    sigma_broad = sigma * np.sqrt(1.0 + separation**2 / 4.0)
    y = (base + sigma_broad * rng.normal(0.0, 1.0, n)).astype(np.float32)
    return x, y


def sep_schedule(x: np.ndarray, sep_min: float = 0.3, sep_max: float = 4.0) -> np.ndarray:
    """Per-input mode spacing (in noise-widths) that grows linearly across ``x ∈ [0, 1]``.

    Equal-weight, equal-variance two-Gaussian mixtures are genuinely bimodal iff their
    means are more than ``2σ`` apart, so ``sep(x) = 2`` is the KNOWN ground-truth boundary
    between "one effective mode" and "two": below it the density is a single blurry blob,
    above it two resolved peaks. With the defaults the crossing sits at
    ``x* = (2 - sep_min) / (sep_max - sep_min)``.

    Args:
        x: input locations, any shape; values expected in ``[0, 1]``.
        sep_min: spacing at ``x = 0`` (merged / unimodal end).
        sep_max: spacing at ``x = 1`` (resolved / bimodal end).

    Returns:
        Per-input spacing array, same shape as ``x`` raveled.
    """
    xr = np.asarray(x, dtype=np.float64).ravel()
    return sep_min + (sep_max - sep_min) * xr


def sep_hump(x: np.ndarray, sep_min: float = 0.3, sep_max: float = 4.0) -> np.ndarray:
    """Per-input mode spacing (noise-widths) that PEAKS mid-range — a non-monotone twin of :func:`sep_schedule`.

    Triangle wave ``sep(x) = sep_min + (sep_max - sep_min)·(1 - |2x - 1|)``: ``sep_min`` at the ends
    ``x ∈ {0, 1}`` and ``sep_max`` at ``x = 0.5``. It therefore crosses the ``2σ`` bimodality boundary
    TWICE — rising near ``x ≈ 0.23`` and falling near ``x ≈ 0.77`` with the defaults — so the
    ground-truth count is non-monotone in ``x``: ``1 → 2 → 1``. The monotone :func:`sep_schedule`
    confounds "tracks the input" with "tracks the structure" (both rise together); this schedule
    separates them, since here the count must come back DOWN while ``x`` keeps rising.

    Args:
        x: input locations, any shape; values expected in ``[0, 1]``.
        sep_min: spacing at the ends ``x ∈ {0, 1}`` (merged / unimodal).
        sep_max: spacing at the centre ``x = 0.5`` (resolved / bimodal).

    Returns:
        Per-input spacing array, same shape as ``x`` raveled.
    """
    xr = np.asarray(x, dtype=np.float64).ravel()
    return sep_min + (sep_max - sep_min) * (1.0 - np.abs(2.0 * xr - 1.0))


def _bimodal_targets(sep: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Draw one equal-weight two-Gaussian sample per element of ``sep`` (mode spacing in noise-widths)."""
    comp = rng.integers(0, 2, size=sep.size)
    offset = (comp - 0.5) * sep * sigma
    return (offset + sigma * rng.normal(0.0, 1.0, sep.size)).astype(np.float32)


def _broad_targets(sep: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Draw one single-Gaussian sample per element of ``sep``, variance-matched to :func:`_bimodal_targets`."""
    sigma_broad = sigma * np.sqrt(1.0 + sep**2 / 4.0)
    return (sigma_broad * rng.normal(0.0, 1.0, sep.size)).astype(np.float32)


def _toy_c_y(x: np.ndarray, sigma: float, sep_min: float, sep_max: float, rng: np.random.Generator) -> np.ndarray:
    """Draw one bimodal target per input ``x`` with spacing ``sep_schedule(x)``."""
    return _bimodal_targets(sep_schedule(x, sep_min, sep_max), sigma, rng)


def _toy_c_broad_y(x: np.ndarray, sigma: float, sep_min: float, sep_max: float, rng: np.random.Generator) -> np.ndarray:
    """Draw one single-mode target per input ``x``, variance matched to :func:`_toy_c_y`."""
    return _broad_targets(sep_schedule(x, sep_min, sep_max), sigma, rng)


def sample_toy_c_given_x(x: np.ndarray, sigma: float = 0.3, sep_min: float = 0.3, sep_max: float = 4.0, seed: int = 42) -> np.ndarray:
    """Draw bimodal ``y`` for an EXPLICIT ``x`` array (used for the gold-standard arbiter).

    Lets a caller resample ``p(y | x)`` many times at a fixed ``x`` (pass a repeated value)
    to estimate the population per-input held-out NLL that a single draw cannot give.
    """
    return _toy_c_y(np.asarray(x), sigma, sep_min, sep_max, np.random.default_rng(seed))


def sample_toy_c_broad_given_x(x: np.ndarray, sigma: float = 0.3, sep_min: float = 0.3, sep_max: float = 4.0, seed: int = 42) -> np.ndarray:
    """Draw matched-variance single-mode ``y`` for an EXPLICIT ``x`` array (gold-standard arbiter)."""
    return _toy_c_broad_y(np.asarray(x), sigma, sep_min, sep_max, np.random.default_rng(seed))


def make_toy_c(n: int = 1500, sigma: float = 0.3, sep_min: float = 0.3, sep_max: float = 4.0, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Toy C: zero-mean target whose noise level grows with ``x``.

    The true mean ``E[y | x]`` is 0 everywhere, so a regressor's best squared error is the noise
    variance ``σ²·(1 + sep_schedule(x)²/4)`` (see :func:`sep_schedule`), which rises with ``x``.

    Pair with :func:`make_toy_c_broad` (same mean and variance at every ``x``, drawn from one
    Gaussian) as the control.

    Args:
        n: sample count.
        sigma: per-component noise σ.
        sep_min: mode spacing at ``x = 0`` (merged end).
        sep_max: mode spacing at ``x = 1`` (resolved end).
        seed: random seed.

    Returns:
        (x, y) with shapes (n, 1) and (n,), both float32.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n).reshape(-1, 1).astype(np.float32)
    y = _toy_c_y(x.ravel(), sigma, sep_min, sep_max, rng)
    return x, y


def make_toy_c_broad(n: int = 1500, sigma: float = 0.3, sep_min: float = 0.3, sep_max: float = 4.0, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian twin of :func:`make_toy_c`, variance-matched at every input.

    At each ``x`` this draws ONE Gaussian whose variance equals that of :func:`make_toy_c`
    at that ``x`` (``σ²·(1 + sep_schedule(x)²/4)``), with the same zero mean. The true mean
    and the noise level are therefore identical to :func:`make_toy_c`.

    Args:
        n: sample count.
        sigma: per-component σ of the bimodal partner (sets the matched variance).
        sep_min: spacing at ``x = 0`` of the partner.
        sep_max: spacing at ``x = 1`` of the partner.
        seed: random seed.

    Returns:
        (x, y) with shapes (n, 1) and (n,), both float32.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n).reshape(-1, 1).astype(np.float32)
    y = _toy_c_broad_y(x.ravel(), sigma, sep_min, sep_max, rng)
    return x, y


def sample_toy_e_given_x(x: np.ndarray, sigma: float = 0.3, sep_min: float = 0.3, sep_max: float = 4.0, seed: int = 42) -> np.ndarray:
    """Draw bimodal ``y`` for an EXPLICIT ``x`` array under the humped schedule (gold-standard arbiter)."""
    return _bimodal_targets(sep_hump(np.asarray(x), sep_min, sep_max), sigma, np.random.default_rng(seed))


def sample_toy_e_broad_given_x(x: np.ndarray, sigma: float = 0.3, sep_min: float = 0.3, sep_max: float = 4.0, seed: int = 42) -> np.ndarray:
    """Draw matched-variance single-mode ``y`` for an EXPLICIT ``x`` array under the humped schedule."""
    return _broad_targets(sep_hump(np.asarray(x), sep_min, sep_max), sigma, np.random.default_rng(seed))


def make_toy_e(n: int = 1500, sigma: float = 0.3, sep_min: float = 0.3, sep_max: float = 4.0, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Toy E: zero-mean target whose noise level rises and then falls across ``x``.

    Like :func:`make_toy_c` (true mean 0 everywhere) but the noise variance ``σ²·(1 + sep_hump(x)²/4)``
    follows :func:`sep_hump` instead of the monotone :func:`sep_schedule`: low at both ends of ``x``
    and highest in the middle.

    Pair with :func:`make_toy_e_broad` (same mean and variance, drawn from one Gaussian) as the control.

    Args:
        n: sample count.
        sigma: per-component noise σ.
        sep_min: spacing at the ends (merged).
        sep_max: spacing at the centre (resolved).
        seed: random seed.

    Returns:
        (x, y) with shapes (n, 1) and (n,), both float32.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n).reshape(-1, 1).astype(np.float32)
    y = _bimodal_targets(sep_hump(x.ravel(), sep_min, sep_max), sigma, rng)
    return x, y


def make_toy_e_broad(n: int = 1500, sigma: float = 0.3, sep_min: float = 0.3, sep_max: float = 4.0, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian twin of :func:`make_toy_e`: same humped variance, same zero mean.

    At each ``x`` this draws ONE Gaussian whose variance equals that of :func:`make_toy_e`
    at that ``x`` (``σ²·(1 + sep_hump(x)²/4)``).

    Args:
        n: sample count.
        sigma: per-component σ of the bimodal partner (sets the matched variance).
        sep_min: spacing at the ends of the partner.
        sep_max: spacing at the centre of the partner.
        seed: random seed.

    Returns:
        (x, y) with shapes (n, 1) and (n,), both float32.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n).reshape(-1, 1).astype(np.float32)
    y = _broad_targets(sep_hump(x.ravel(), sep_min, sep_max), sigma, rng)
    return x, y


V_TOY0_W = np.array([1.5, -2.0, 0.5, 3.0, -1.0], dtype=np.float64)  # fixed, known coefficients (p=5)


def make_v_toy0(n: int = 200, sigma: float = 1.0, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """V-toy0: linear-Gaussian, well-specified LINEAR model — the WS3 variance-estimation anchor.

    y = wᵀx + σ·ε,   x ~ N(0, I_p),   ε ~ N(0, 1),   w = :data:`V_TOY0_W` (fixed across seeds;
    only the x/noise draw varies with ``seed``).

    Ordinary least squares is exactly well-specified here, so the classical in-sample MLE
    variance bias ``(N−p)/N`` and its closed-form correction ``RSS/(N−p)`` are analytic — this
    is the twin that anchors the WS3 variance framework where ground truth is exact, contrasted
    with the nonlinear, heteroscedastic :func:`make_v_toy1`.

    Args:
        n: sample count.
        sigma: noise σ (homoscedastic).
        seed: random seed.

    Returns:
        (x, y) with shapes (n, p) and (n,), both float32, p = ``V_TOY0_W.size``.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, (n, V_TOY0_W.size)).astype(np.float32)
    y = (x @ V_TOY0_W + sigma * rng.normal(0.0, 1.0, n)).astype(np.float32)
    return x, y


def v_toy1_f(x: np.ndarray) -> np.ndarray:
    """V-toy1 known smooth mean: f(x) = sin(2π x), x ∈ [0, 1]."""
    xr = np.asarray(x, dtype=np.float64).ravel()
    return np.sin(2 * np.pi * xr)


def v_toy1_sigma(x: np.ndarray) -> np.ndarray:
    """V-toy1 known heteroscedastic noise scale: σ(x) = 0.1 + 0.3·sigmoid(4x), x ∈ [0, 1]."""
    xr = np.asarray(x, dtype=np.float64).ravel()
    return 0.1 + 0.3 / (1.0 + np.exp(-4.0 * xr))


def make_v_toy1(n: int = 200, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """V-toy1: 1-D heteroscedastic regression with KNOWN smooth f(x) and σ(x) — the WS3 disease toy.

    y = f(x) + σ(x)·ε,   x ~ U(0, 1),   ε ~ N(0, 1),   f = :func:`v_toy1_f`, σ = :func:`v_toy1_sigma`.

    f and σ are exposed as standalone functions (not baked into the sampler) so a fitted model's
    σ̂(x) can be compared directly against ground truth: the in-sample σ̂/σ_true ratio that WS3/V0
    tracks per epoch, and the SSR-gap tell (train vs held-out standardized squared residuals).

    Args:
        n: sample count.
        seed: random seed.

    Returns:
        (x, y) with shapes (n, 1) and (n,), both float32.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n).reshape(-1, 1).astype(np.float32)
    y = (v_toy1_f(x.ravel()) + v_toy1_sigma(x.ravel()) * rng.normal(0.0, 1.0, n)).astype(np.float32)
    return x, y


V_TOY1H_SIGMA = 0.3  # constant (homoscedastic) noise scale for make_v_toy1h


def make_v_toy1h(n: int = 200, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """V-toy1h: HOMOSCEDASTIC twin of :func:`make_v_toy1` -- same mean, CONSTANT σ.

    y = f(x) + σ·ε,   x ~ U(0, 1),   ε ~ N(0, 1),   f = :func:`v_toy1_f`, σ = :data:`V_TOY1H_SIGMA`.

    V-toy1 plants a RISING σ(x) (:func:`v_toy1_sigma`), so any variance-estimation mechanism that
    assumes a single global noise level is already misspecified on the NOISE model there, which
    confounds WS3/V1's mean-model-misspecification arm (fit a linear model to the curved
    f(x)=sin(2πx)) with a variance-model-misspecification arm it isn't meant to exercise. This twin
    holds σ constant so V1 can isolate mean-model (mis)specification while the noise stays exactly
    global-homoscedastic underneath -- the well-specified-noise anchor V-toy0 provides for the
    LINEAR mean, extended to a curved mean.

    Args:
        n: sample count.
        seed: random seed.

    Returns:
        (x, y) with shapes (n, 1) and (n,), both float32.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n).reshape(-1, 1).astype(np.float32)
    y = (v_toy1_f(x.ravel()) + V_TOY1H_SIGMA * rng.normal(0.0, 1.0, n)).astype(np.float32)
    return x, y


def _staircase_k(x: np.ndarray) -> np.ndarray:
    """Ground-truth component count per input for Toy D: 1 on [0,1/3), 2 on [1/3,2/3), 3 on [2/3,1]."""
    xr = np.asarray(x, dtype=np.float64).ravel()
    return np.where(xr < 1.0 / 3.0, 1, np.where(xr < 2.0 / 3.0, 2, 3)).astype(int)


def _staircase_targets(x: np.ndarray, sigma: float, separation: float, rng: np.random.Generator) -> np.ndarray:
    """Draw one sample per input from a ``k(x)``-component equal-weight mixture (means evenly spaced by ``separation·σ``, centred at 0)."""
    k = _staircase_k(x)
    comp = np.minimum((rng.uniform(0.0, 1.0, k.size) * k).astype(int), k - 1)  # 0..k-1 per input
    offset = (comp - (k - 1) / 2.0) * separation * sigma
    return (offset + sigma * rng.normal(0.0, 1.0, k.size)).astype(np.float32)


def sample_toy_d_given_x(x: np.ndarray, sigma: float = 0.3, separation: float = 4.0, seed: int = 42) -> np.ndarray:
    """Draw staircase ``y`` for an EXPLICIT ``x`` array (gold-standard arbiter)."""
    return _staircase_targets(np.asarray(x), sigma, separation, np.random.default_rng(seed))


def make_toy_d(n: int = 1800, sigma: float = 0.3, separation: float = 4.0, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Toy D: zero-mean target whose noise level steps up across ``x`` in thirds.

    The true mean ``E[y | x]`` is 0 everywhere. The noise variance is ``σ²·(1 + separation²·(c² − 1)/12)``
    with ``c = 1`` on ``[0, 1/3)``, ``c = 2`` on ``[1/3, 2/3)`` and ``c = 3`` on ``[2/3, 1]``, so it steps
    up by thirds (the draw takes ``c`` equally likely values spaced ``separation·σ`` apart).

    Args:
        n: sample count (split roughly equally across the three thirds).
        sigma: per-component noise σ.
        separation: adjacent-mode spacing in units of σ (4 = cleanly resolved).
        seed: random seed.

    Returns:
        (x, y) with shapes (n, 1) and (n,), both float32.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n).reshape(-1, 1).astype(np.float32)
    y = _staircase_targets(x.ravel(), sigma, separation, rng)
    return x, y


# ---------------------------------------------------------------------------
# Toy D, multi-dimensional (task T2, per-input selector program): the port de-risk.
# ---------------------------------------------------------------------------


def _toy_d_ndim_direction(dim: int, seed: int = 20260710) -> np.ndarray:
    """Fixed, known unit vector used as the "rotated" projection axis for `make_toy_d_ndim`.

    Deterministic in ``dim`` alone (NOT `make_toy_d_ndim`'s own ``seed`` argument, which only
    controls the noise/x draw) so every seed/config at a given ``dim`` shares the IDENTICAL
    rotation direction — an isotropic standard-normal draw normalized to unit L2 norm, fixed by
    a constant internal seed offset by ``dim`` (so different ``dim`` values get different, but
    each individually fixed, directions).
    """
    rng = np.random.default_rng(seed + dim)
    u = rng.normal(size=dim)
    return (u / np.linalg.norm(u)).astype(np.float64)


def toy_d_ndim_s(x: np.ndarray, rotated: bool = False) -> np.ndarray:
    """The scalar staircase-driving coordinate ``s`` for `make_toy_d_ndim`.

    ``s = x[:, 0]`` (axis-aligned) or ``s = (u . x) / sqrt(dim)`` with ``u`` the fixed direction
    from :func:`_toy_d_ndim_direction` (rotated) — EXECUTION_PLAN.md T2's literal formula. With
    ``u`` an isotropic (mixed-sign, in general) unit vector this does NOT reproduce the exact
    ``[0, 1]`` range or the uniform SHAPE of the axis-aligned case (a Cauchy-Schwarz bound gives
    ``|s| <= 1`` always, but the realized marginal concentrates more tightly around its mean than
    a genuine ``Uniform[0, 1]`` draw, a mild CLT effect that grows with ``dim``) — reused verbatim
    against `_staircase_k`'s hardcoded ``1/3, 2/3`` cutoffs regardless, per the plan's literal
    "staircase k*(s) by thirds of s" text; the resulting region-size imbalance at ``dim=5`` is a
    KNOWN, documented caveat for interpreting T2's rotated-vs-axis bar (iii) (see
    `capacity_ladder_results/T2/PREREGISTRATION.md`), not silently corrected here.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(-1, 1)
    if not rotated:
        return x_arr[:, 0].copy()
    dim = x_arr.shape[1]
    u = _toy_d_ndim_direction(dim)
    return (x_arr @ u) / math.sqrt(dim)


def toy_d_ndim_k_star(x: np.ndarray, rotated: bool = False) -> np.ndarray:
    """Analytic ground-truth component count ``k*(s) in {1, 2, 3}`` for `make_toy_d_ndim` (gold read only — never selector-visible)."""
    return _staircase_k(toy_d_ndim_s(x, rotated))


def make_toy_d_ndim(n: int = 2500, dim: int = 2, rotated: bool = False, sigma: float = 0.3, separation: float = 4.0, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Multi-dimensional generalization of Toy D (task T2, the per-input-selector port de-risk).

    ``x ~ U[0, 1]^dim``; the staircase-driving coordinate is ``s = toy_d_ndim_s(x, rotated)``
    (axis-aligned ``x[:, 0]``, or a fixed rotated projection mixing every coordinate — see
    :func:`toy_d_ndim_s`). The remaining coordinates (all of ``x`` when axis-aligned; none when
    rotated, since the rotated projection mixes all of them) are NUISANCE — drawn, never used to
    generate ``y`` beyond their contribution to ``s``. SAME component geometry as :func:`make_toy_d`
    (separation 4σ, σ=0.3, means centred at 0): ground truth ``k*(s) in {1, 2, 3}`` steps at thirds
    of ``s``, via the identical `_staircase_k`/`_staircase_targets` machinery — at ``dim=1,
    rotated=False`` this reduces to exactly :func:`make_toy_d`'s generative process.

    Args:
        n: sample count.
        dim: input dimension (``dim=1`` reproduces :func:`make_toy_d`'s process exactly).
        rotated: if True, drive the staircase off the rotated projection instead of ``x[:, 0]``.
        sigma: per-component noise σ.
        separation: adjacent-mode spacing in units of σ.
        seed: random seed.

    Returns:
        ``(x, y)`` with shapes ``(n, dim)`` and ``(n,)``, both float32.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=(n, dim)).astype(np.float32)
    s = toy_d_ndim_s(x, rotated)
    y = _staircase_targets(s, sigma, separation, rng)
    return x, y


def sample_toy_d_ndim_given_x(x: np.ndarray, rotated: bool = False, sigma: float = 0.3, separation: float = 4.0, seed: int = 42) -> np.ndarray:
    """Draws staircase ``y`` for an EXPLICIT ``(n, dim)`` ``x`` array (gold-standard arbiter), multi-D toy D."""
    s = toy_d_ndim_s(np.asarray(x), rotated)
    return _staircase_targets(s, sigma, separation, np.random.default_rng(seed))


def _toy_d_ndim_total_var(k: np.ndarray, sigma: float, separation: float) -> np.ndarray:
    """Closed-form total marginal variance of one staircase-mixture draw at component count ``k``.

    For ``k`` equally-likely components spaced ``separation·σ`` apart and centred at 0, the
    component index is uniform over ``{0, ..., k-1}``, whose variance is ``(k^2 - 1) / 12``
    (standard discrete-uniform identity); ``Var(offset) = separation^2 · σ^2 · (k^2 - 1) / 12``.
    Total variance adds the independent per-component noise ``σ^2``. Used by
    `make_toy_d_ndim_broad`'s variance-matched single-mode twin (mirrors `_broad_targets`'s role
    for the continuous-spacing toys C/E, here for the discrete staircase count).
    """
    k_arr = np.asarray(k, dtype=np.float64)
    return sigma**2 * (1.0 + separation**2 * (k_arr**2 - 1.0) / 12.0)


def make_toy_d_ndim_broad(n: int = 2500, dim: int = 2, sigma: float = 0.3, separation: float = 4.0, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Single-mode, variance-matched twin of `make_toy_d_ndim` (axis-aligned): ``k*(s) = 1`` everywhere.

    At each ``x`` draws ONE Gaussian ``N(0, total_var(k*(s)))`` — the SAME total marginal variance
    the genuine staircase mixture has at that region (:func:`_toy_d_ndim_total_var`), but never
    develops a second/third mode. The multi-D over-chopping trap, the same role
    :func:`make_toy_c_broad`/:func:`make_toy_e_broad` play for toys C/E: a selector that merely
    tracks the variance envelope (which genuinely does step up with ``s``, matching the bimodal/
    trimodal envelope) must not be credited for "detecting structure" here.

    Args:
        n: sample count.
        dim: input dimension (nuisance coordinates beyond ``x[:, 0]``, matching the axis-aligned
            staircase's own nuisance dims — no rotated broad twin is defined, per the T2 matrix).
        sigma: per-component σ of the staircase partner.
        separation: adjacent-mode spacing of the staircase partner.
        seed: random seed.

    Returns:
        ``(x, y)`` with shapes ``(n, dim)`` and ``(n,)``, both float32.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=(n, dim)).astype(np.float32)
    s = toy_d_ndim_s(x, rotated=False)
    k = _staircase_k(s)
    var = _toy_d_ndim_total_var(k, sigma, separation)
    y = (np.sqrt(var) * rng.normal(0.0, 1.0, n)).astype(np.float32)
    return x, y


# ---------------------------------------------------------------------------
# Exact conditional laws (the true conditional mean and variance) and the review's new toys
# (docs/probreg/review/review.md Appendix C: T-N1 switch, T-N2 Student-t, T-N3 skewed, T-N4 edge).
# Every toy above is restated as a `GaussianMixtureLaw` so its exact conditional mean and variance
# and repeated draws at fixed x all come from one object. The generators
# above are left untouched (their random streams reproduce every stored result).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GaussianMixtureLaw:
    """The conditional law ``y | x`` of a Gaussian-mixture toy at ``n`` inputs.

    Attributes:
        weights: ``(n, m)`` component weights, rows summing to 1 (zeros allowed for absent components).
        means: ``(n, m)`` component means.
        sds: ``(n, m)`` component standard deviations.
    """

    weights: np.ndarray
    means: np.ndarray
    sds: np.ndarray

    def log_density(self, y: np.ndarray) -> np.ndarray:
        """Per-point exact ``log p(y_i | x_i)`` in nats, shape ``(n,)``."""
        yr = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        with np.errstate(divide="ignore"):
            log_w = np.log(self.weights)
        log_comp = -0.5 * (math.log(2 * math.pi) + 2.0 * np.log(self.sds) + ((yr - self.means) / self.sds) ** 2)
        a = log_w + log_comp
        a_max = np.max(a, axis=1, keepdims=True)
        return (a_max + np.log(np.sum(np.exp(a - a_max), axis=1, keepdims=True))).ravel()

    def mean(self) -> np.ndarray:
        """``E[y | x]`` per input, shape ``(n,)``."""
        return np.sum(self.weights * self.means, axis=1)

    def variance(self) -> np.ndarray:
        """``Var(y | x)`` per input, shape ``(n,)`` (its mean over x is the squared-error Bayes risk)."""
        return np.sum(self.weights * (self.sds**2 + self.means**2), axis=1) - self.mean() ** 2

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """One draw of ``y`` per input, float32 ``(n,)``."""
        n = self.weights.shape[0]
        cum = np.cumsum(self.weights, axis=1)
        comp = np.minimum((rng.uniform(0.0, 1.0, (n, 1)) > cum).sum(axis=1), self.weights.shape[1] - 1)
        rows = np.arange(n)
        return (self.means[rows, comp] + self.sds[rows, comp] * rng.normal(0.0, 1.0, n)).astype(np.float32)


def _law(weights: list[np.ndarray], means: list[np.ndarray], sds: list[np.ndarray]) -> GaussianMixtureLaw:
    """Stacks per-component ``(n,)`` arrays into a :class:`GaussianMixtureLaw`."""
    return GaussianMixtureLaw(np.stack(weights, axis=1), np.stack(means, axis=1), np.stack(sds, axis=1))


def _baseline(xr: np.ndarray, baseline: str) -> np.ndarray:
    """The ``"sin"``/``"zero"`` location baseline shared by toys A, B and the broad twin."""
    if baseline == "sin":
        return np.sin(2 * np.pi * xr)
    if baseline == "zero":
        return np.zeros_like(xr)
    raise ValueError(f"baseline must be 'sin' or 'zero', got {baseline!r}")


def toy_a_law(x: np.ndarray, sigma: float = 0.5) -> GaussianMixtureLaw:
    """Exact law of :func:`make_toy_a`: ``N(sin 2πx, σ²)``."""
    xr = np.asarray(x, dtype=np.float64).ravel()
    return _law([np.ones_like(xr)], [np.sin(2 * np.pi * xr)], [np.full_like(xr, sigma)])


def toy_b_law(x: np.ndarray, k_true: int = 3, separation: float = 4.0, sigma: float = 0.3, baseline: str = "sin") -> GaussianMixtureLaw:
    """Exact law of :func:`make_toy_b`: ``k_true`` equal-weight components spaced ``separation·σ`` about the baseline."""
    xr = np.asarray(x, dtype=np.float64).ravel()
    base = _baseline(xr, baseline)
    offsets = [(j - (k_true - 1) / 2.0) * separation * sigma for j in range(k_true)]
    return _law([np.full_like(xr, 1.0 / k_true)] * k_true, [base + o for o in offsets], [np.full_like(xr, sigma)] * k_true)


def broad_unimodal_law(x: np.ndarray, separation: float = 4.0, sigma: float = 0.3, baseline: str = "zero") -> GaussianMixtureLaw:
    """Exact law of :func:`make_broad_unimodal`: one Gaussian moment-matched to the two-mode toy B."""
    xr = np.asarray(x, dtype=np.float64).ravel()
    return _law([np.ones_like(xr)], [_baseline(xr, baseline)], [np.full_like(xr, sigma * math.sqrt(1.0 + separation**2 / 4.0))])


def _two_mode_law(sep: np.ndarray, sigma: float) -> GaussianMixtureLaw:
    """Equal-weight pair at ``±sep·σ/2`` (the law :func:`_bimodal_targets` draws from)."""
    half = 0.5 * sep * sigma
    return _law([np.full_like(sep, 0.5)] * 2, [-half, half], [np.full_like(sep, sigma)] * 2)


def _broad_law(sep: np.ndarray, sigma: float) -> GaussianMixtureLaw:
    """One Gaussian variance-matched to :func:`_two_mode_law` (the law :func:`_broad_targets` draws from)."""
    return _law([np.ones_like(sep)], [np.zeros_like(sep)], [sigma * np.sqrt(1.0 + sep**2 / 4.0)])


def toy_c_law(x: np.ndarray, sigma: float = 0.3, sep_min: float = 0.3, sep_max: float = 4.0) -> GaussianMixtureLaw:
    """Exact law of :func:`make_toy_c`."""
    return _two_mode_law(sep_schedule(x, sep_min, sep_max), sigma)


def toy_c_broad_law(x: np.ndarray, sigma: float = 0.3, sep_min: float = 0.3, sep_max: float = 4.0) -> GaussianMixtureLaw:
    """Exact law of :func:`make_toy_c_broad`."""
    return _broad_law(sep_schedule(x, sep_min, sep_max), sigma)


def toy_e_law(x: np.ndarray, sigma: float = 0.3, sep_min: float = 0.3, sep_max: float = 4.0) -> GaussianMixtureLaw:
    """Exact law of :func:`make_toy_e`."""
    return _two_mode_law(sep_hump(x, sep_min, sep_max), sigma)


def toy_d_law(x: np.ndarray, sigma: float = 0.3, separation: float = 4.0) -> GaussianMixtureLaw:
    """Exact law of :func:`make_toy_d`: three component slots, unused ones at weight 0."""
    xr = np.asarray(x, dtype=np.float64).ravel()
    k = _staircase_k(xr)
    weights, means = [], []
    for j in range(3):
        present = j < k
        weights.append(np.where(present, 1.0 / k, 0.0))
        means.append(np.where(present, (j - (k - 1) / 2.0) * separation * sigma, 0.0))
    return _law(weights, means, [np.full_like(xr, sigma)] * 3)


def switch_weight(x: np.ndarray, x_lo: float = 0.3, x_peak: float = 0.5, x_hi: float = 0.7, w_peak: float = 0.5) -> np.ndarray:
    """T-N1's second-component weight: 0 outside ``(x_lo, x_hi)``, rising linearly to ``w_peak`` at ``x_peak``."""
    xr = np.asarray(x, dtype=np.float64).ravel()
    rise = w_peak * (xr - x_lo) / (x_peak - x_lo)
    fall = w_peak * (x_hi - xr) / (x_hi - x_peak)
    return np.clip(np.where(xr <= x_peak, rise, fall), 0.0, w_peak)


def switch_bimodal_law(x: np.ndarray, a: float = 0.6, sigma: float = 0.3) -> GaussianMixtureLaw:
    """T-N1, fixed-location varying-weight bimodal: ``(1 − w(x)) N(−a, σ²) + w(x) N(a, σ²)``.

    The true mean is ``−a + 2a·w(x)`` and the noise variance is ``σ² + 4a²·w(x)(1 − w(x))``.
    """
    w = switch_weight(x)
    return _law([1.0 - w, w], [np.full_like(w, -a), np.full_like(w, a)], [np.full_like(w, sigma)] * 2)


def edge_component_law(x: np.ndarray, w_e: float, a: float = 0.6, edge_loc: float = 1.8, sigma: float = 0.3) -> GaussianMixtureLaw:
    """T-N4: the zero-baseline toy B plus a third group of points at ``edge_loc``, drawn with probability ``w_e``."""
    xr = np.asarray(x, dtype=np.float64).ravel()
    w_main = np.full_like(xr, (1.0 - w_e) / 2.0)
    return _law([w_main, w_main, np.full_like(xr, w_e)], [np.full_like(xr, -a), np.full_like(xr, a), np.full_like(xr, edge_loc)], [np.full_like(xr, sigma)] * 3)


def _draw_x(n: int, rng: np.random.Generator) -> np.ndarray:
    """``x ~ U[0, 1]`` as the ``(n, 1)`` float32 column every generator returns."""
    return rng.uniform(0.0, 1.0, n).reshape(-1, 1).astype(np.float32)


def make_switch_bimodal(n: int = 2000, a: float = 0.6, sigma: float = 0.3, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """T-N1 sample (see :func:`switch_bimodal_law`)."""
    rng = np.random.default_rng(seed)
    x = _draw_x(n, rng)
    return x, switch_bimodal_law(x, a, sigma).sample(rng)


def make_edge_component(n: int, w_e: float, a: float = 0.6, edge_loc: float = 1.8, sigma: float = 0.3, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """T-N4 sample (see :func:`edge_component_law`)."""
    rng = np.random.default_rng(seed)
    x = _draw_x(n, rng)
    return x, edge_component_law(x, w_e, a, edge_loc, sigma).sample(rng)


def student_t_log_density(y: np.ndarray, x: np.ndarray, nu: float = 3.0, scale: float = 0.3, baseline: str = "zero") -> np.ndarray:
    """T-N2's exact per-point log-density: ``y = m(x) + scale·T_ν`` with ``m`` the ``"zero"``/``"sin"`` baseline."""
    z = (np.asarray(y, dtype=np.float64).ravel() - _baseline(np.asarray(x, dtype=np.float64).ravel(), baseline)) / scale
    const = math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(nu * math.pi) - math.log(scale)
    return const - 0.5 * (nu + 1) * np.log1p(z**2 / nu)


def sample_student_t_given_x(x: np.ndarray, nu: float = 3.0, scale: float = 0.3, baseline: str = "zero", seed: int = 42) -> np.ndarray:
    """T-N2 draws at an explicit ``x`` (repeated-draw helper)."""
    xr = np.asarray(x, dtype=np.float64).ravel()
    rng = np.random.default_rng(seed)
    return (_baseline(xr, baseline) + scale * rng.standard_t(nu, xr.size)).astype(np.float32)


def make_student_t_unimodal(n: int = 2000, nu: float = 3.0, scale: float = 0.3, baseline: str = "zero", seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """T-N2 sample: true mean is the baseline (0 by default), heavy-tailed Student-t noise with ``nu`` degrees of freedom."""
    rng = np.random.default_rng(seed)
    x = _draw_x(n, rng)
    return x, (_baseline(x.ravel().astype(np.float64), baseline) + scale * rng.standard_t(nu, n)).astype(np.float32)


def _skew_normal_constants(shape: float, sd: float) -> tuple[float, float]:
    """``(s, E[Z])`` for T-N3: ``y = s(Z − E[Z])``, ``Z ~ skew-normal(shape)``, ``s`` set so ``sd(y) = sd``."""
    delta = shape / math.sqrt(1.0 + shape**2)
    mean_z = delta * math.sqrt(2.0 / math.pi)
    return sd / math.sqrt(1.0 - mean_z**2), mean_z


def skewed_unimodal_log_density(y: np.ndarray, shape: float = 5.0, sd: float = 0.6) -> np.ndarray:
    """T-N3's exact per-point log-density (no x dependence): ``log 2 − log s + log φ(z) + log Φ(shape·z)``."""
    s, mean_z = _skew_normal_constants(shape, sd)
    z = np.asarray(y, dtype=np.float64).ravel() / s + mean_z
    return math.log(2.0) - math.log(s) - 0.5 * (math.log(2 * math.pi) + z**2) + log_ndtr(shape * z)


def sample_skewed_unimodal_given_x(x: np.ndarray, shape: float = 5.0, sd: float = 0.6, seed: int = 42) -> np.ndarray:
    """T-N3 draws at an explicit ``x`` (the law does not depend on ``x``)."""
    rng = np.random.default_rng(seed)
    return _skewed_draws(np.asarray(x).ravel().size, shape, sd, rng)


def _skewed_draws(n: int, shape: float, sd: float, rng: np.random.Generator) -> np.ndarray:
    """Skew-normal by the ``δ|U₀| + √(1−δ²) U₁`` construction, centred and scaled to ``sd``."""
    s, mean_z = _skew_normal_constants(shape, sd)
    delta = shape / math.sqrt(1.0 + shape**2)
    z = delta * np.abs(rng.normal(0.0, 1.0, n)) + math.sqrt(1.0 - delta**2) * rng.normal(0.0, 1.0, n)
    return (s * (z - mean_z)).astype(np.float32)


def make_skewed_unimodal(n: int = 2000, shape: float = 5.0, sd: float = 0.6, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """T-N3 sample: true mean 0, skewed noise."""
    rng = np.random.default_rng(seed)
    x = _draw_x(n, rng)
    return x, _skewed_draws(n, shape, sd, rng)


@dataclass(frozen=True)
class StudentTLaw:
    """T-N2 at ``n`` inputs, with the same interface as :class:`GaussianMixtureLaw` (log-density, draws, mean, variance)."""

    x: np.ndarray
    nu: float = 3.0
    scale: float = 0.3
    baseline: str = "zero"

    def log_density(self, y: np.ndarray) -> np.ndarray:
        """Per-point exact log-density."""
        return student_t_log_density(y, self.x, self.nu, self.scale, self.baseline)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """One draw of ``y`` per input, float32 ``(n,)``."""
        xr = np.asarray(self.x, dtype=np.float64).ravel()
        return (_baseline(xr, self.baseline) + self.scale * rng.standard_t(self.nu, xr.size)).astype(np.float32)

    def mean(self) -> np.ndarray:
        """``E[y | x]`` (finite for ``nu > 1``)."""
        return _baseline(np.asarray(self.x, dtype=np.float64).ravel(), self.baseline)

    def variance(self) -> np.ndarray:
        """``Var(y | x) = scale² ν / (ν − 2)`` (finite for ``nu > 2``)."""
        return np.full(np.asarray(self.x).ravel().size, self.scale**2 * self.nu / (self.nu - 2.0))


def level_staircase_mean(x: np.ndarray, strips: int = 4, resolutions: tuple[int, ...] = (1, 2, 4), gap: float = 1.0) -> np.ndarray:
    """``E[y | x]`` of the level staircase (docs/probreg/shared/toy_design_spec.md §3; review Appendix C T-M3).

    ``[0, 1]`` is cut into ``len(resolutions)`` equal zones; zone ``j`` carries ``R_j`` levels spaced ``gap`` apart
    and centred on 0, assigned by ``r = 1 + floor(R_j · frac(strips · u))`` with ``u`` the within-zone coordinate,
    so every level of a zone has probability ``1/R_j`` and ``strips`` sets how finely the levels alternate.
    """
    xr = np.clip(np.asarray(x, dtype=np.float64).ravel(), 0.0, np.nextafter(1.0, 0.0))
    n_zones = len(resolutions)
    zone = np.floor(xr * n_zones).astype(int)
    u = xr * n_zones - zone
    res = np.asarray(resolutions)[zone]
    level = 1 + np.floor(res * np.mod(strips * u, 1.0)).astype(int)
    return gap * (level - (res + 1) / 2.0)


def level_staircase_law(x: np.ndarray, sigma: float = 0.3, strips: int = 4, resolutions: tuple[int, ...] = (1, 2, 4), gap: float = 1.0) -> GaussianMixtureLaw:
    """T-M3: ``y = μ_{r(x)} + σ ε`` with constant ``σ``; the only squared-error floor is ``σ²`` (review F13e)."""
    mean = level_staircase_mean(x, strips, resolutions, gap)
    return _law([np.ones_like(mean)], [mean], [np.full_like(mean, sigma)])
