"""Shared toy datasets for sweep scripts."""

from __future__ import annotations

import math

import numpy as np


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

    No intrinsic mixture structure: p(y|x) is a single Gaussian. Used as the
    "no intrinsic k" baseline in the k-selection diagnostic experiments.
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
    """Toy C: per-input mode structure — the Basis B toy.

    The conditional ``p(y | x)`` is a two-component Gaussian mixture whose spacing grows
    with ``x`` (see :func:`sep_schedule`): at low ``x`` the two modes are merged into one
    blurry blob (one effective mode), at high ``x`` they are cleanly resolved (two modes).
    The mean is held at zero so ALL the ``x`` dependence lives in the SHAPE, not the
    location — that is what no existing toy provides, and what an input-adaptive
    resolution dial has to track. Ground truth per input: ``k*(x) = 2`` iff
    ``sep_schedule(x) > 2`` else ``1``.

    Pair with :func:`make_toy_c_broad` (same mean & variance at every ``x``, but always one
    mode) as the over-chopping control.

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
    """Single-mode twin of :func:`make_toy_c`, variance-matched at every input (over-chopping trap).

    At each ``x`` this draws ONE Gaussian whose variance equals the marginal variance of the
    bimodal :func:`make_toy_c` at that ``x`` (``σ²·(1 + sep_schedule(x)²/4)``), with the same
    zero mean. So it widens with ``x`` exactly like the bimodal envelope but never develops a
    second peak: ``k*(x) = 1`` everywhere. A tiling model is tempted to raise its bucket count
    here as the spread grows; the honest held-out arbiter must give those buckets no credit.

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
    """Toy E: non-monotone per-input structure — the x-confound breaker.

    Identical in spirit to :func:`make_toy_c` but the two-component spacing follows :func:`sep_hump`
    instead of the monotone :func:`sep_schedule`: the modes are merged at both ends of ``x`` and
    cleanly resolved only in the middle. Ground truth per input: ``k*(x) = 2`` iff ``sep_hump(x) > 2``
    else ``1``, i.e. ``1 → 2 → 1`` across ``x``. An input-adaptive resolution dial must therefore
    raise its count and then LOWER it again while ``x`` increases monotonically — which a selector
    that merely tracks ``x`` (or the marginal variance, which also humps) cannot fake.

    Pair with :func:`make_toy_e_broad` (same humped mean & variance, but always one mode) as the
    over-chopping / variance-tracking control.

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
    """Single-mode twin of :func:`make_toy_e`: same humped marginal variance, never a second peak.

    At each ``x`` this draws ONE Gaussian whose variance equals the marginal variance of
    :func:`make_toy_e` at that ``x`` (``σ²·(1 + sep_hump(x)²/4)``), zero mean. So the spread humps
    mid-range exactly like the bimodal envelope but ``k*(x) = 1`` everywhere. If the effective count
    humps here too, the selector is tracking the variance hump, not genuine bimodality.

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
    """Toy D: a staircase in component count — the count-beyond-binary / ceiling test.

    The conditional ``p(y | x)`` is an equal-weight Gaussian mixture whose component COUNT steps up
    in ``x``: 1 component on ``[0, 1/3)``, 2 on ``[1/3, 2/3)``, 3 on ``[2/3, 1]``, all centred at 0
    with adjacent modes ``separation·σ`` apart (cleanly resolved at the default 4σ). The mean is held
    at 0 everywhere, so the only thing that changes across the staircase is HOW MANY modes the density
    has. Ground truth ``k*(x) ∈ {1, 2, 3}`` by third. Tests whether per-input structure reading
    (a) generalises past the 1-vs-2 case to 3, and (b) STOPS at 3 rather than tiling out to ``k_max``.

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
