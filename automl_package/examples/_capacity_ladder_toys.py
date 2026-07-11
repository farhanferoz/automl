"""Toys with INPUT-VARYING required capacity (F1, capacity-ladder execution plan
docs/plans/capacity_ladder_2026-07-09/EXECUTION_PLAN.md §3 WS2).

Three generators, all 1-D, x in [-1, 1], following the conventions of
``_toy_datasets.py`` (plain functions, keyword defaults, float32 arrays, a
standalone ground-truth function per toy so callers/tests can grade against it
without re-deriving it):

  * Toy G — the region x < 0 is purely linear (trivial, depth-1-sufficient);
    the region x >= 0 is a sine modulated by a linear envelope (compositional,
    needs more depth to resolve). The compositional amplitude is solved
    (:func:`toy_g_amplitude`, a deterministic fine-grid quadrature, not RNG)
    so the two regions' TOTAL marginal variance (signal + noise) matches
    exactly — :func:`toy_g_region_variance` gives the shared target. Ground
    truth: required capacity is higher on x >= 0.

  * Toy G-flat — the negative control (the bypass-confound lesson: smooth
    data everywhere is the negative control). Reuses Toy G's linear formula
    across the WHOLE domain instead of switching to the compositional form at
    x = 0. Because Var(a*x) is the same (a^2/12) on both unit-width halves
    [-1, 0) and [0, 1], using the SAME slope `a` and noise `sigma` as Toy G
    reproduces Toy G's per-region variance target exactly while having
    uniformly trivial structure — no region should read as needing more
    capacity than the other.

  * Toy H — the SNR dial (k-as-resolution-dial thesis, WS1 memory
    project_classification_as_regulariser). ONE fixed function
    (:func:`toy_h_f`) is used on both sides of x = 0 (no structural change);
    only the noise scale changes (:func:`toy_h_sigma`, low on x < 0 = high
    SNR, high on x >= 0 = low SNR), reusing Toy G's x = 0 region boundary
    (:func:`toy_g_region`) so the same per-half analysis code applies to all
    three toys. Capacity need should vary through SNR here, not through
    structure.
"""

from __future__ import annotations

import numpy as np


def toy_g_region(x: np.ndarray) -> np.ndarray:
    """Ground-truth region label for Toy G (and, by convention, Toy H): 0 = linear (x < 0), 1 = compositional (x >= 0)."""
    xr = np.asarray(x, dtype=np.float64).ravel()
    return (xr >= 0.0).astype(int)


def toy_g_region_variance(a: float = 1.5, sigma: float = 0.25) -> float:
    """Shared per-region TOTAL marginal variance (signal + noise) that Toy G is engineered to hit on both halves.

    x ~ Uniform(-1, 0) and x ~ Uniform(0, 1) are both unit-width, so Var(a*x) = a^2/12 on
    either half; the linear region's total variance is therefore a^2/12 + sigma^2, and
    :func:`toy_g_amplitude` solves the compositional region's amplitude so its signal
    variance matches a^2/12 too, making this the target for BOTH regions.
    """
    return a**2 / 12.0 + sigma**2


def toy_g_amplitude(a: float = 1.5, omega: float = 4 * np.pi, n_grid: int = 200_001) -> float:
    """Solves the compositional-region amplitude so its signal variance matches the linear region's.

    Uses a deterministic fine grid over x in [0, 1] (not RNG sampling) to estimate
    Var(sin(omega*x)*(1+x)), then scales so Var(amplitude*sin(omega*x)*(1+x)) == a^2/12 —
    the linear region's exact population variance. Deterministic and reproducible: callers
    get the same amplitude every time regardless of seed.
    """
    xg = np.linspace(0.0, 1.0, n_grid)
    raw = np.sin(omega * xg) * (1.0 + xg)
    raw_var = raw.var()
    target_var = a**2 / 12.0
    return float(np.sqrt(target_var / raw_var))


def toy_g_signal(x: np.ndarray, a: float = 1.5, omega: float = 4 * np.pi, amplitude: float | None = None) -> np.ndarray:
    """Toy G's noise-free signal: a*x for x < 0, amplitude*sin(omega*x)*(1+x) for x >= 0.

    Both halves evaluate to 0 at x = 0, so the signal is continuous at the region boundary
    (no artificial kink at the split point). `amplitude` defaults to :func:`toy_g_amplitude`.
    """
    xr = np.asarray(x, dtype=np.float64).ravel()
    if amplitude is None:
        amplitude = toy_g_amplitude(a, omega)
    linear = a * xr
    compositional = amplitude * np.sin(omega * xr) * (1.0 + xr)
    return np.where(xr < 0.0, linear, compositional)


def make_toy_g(n: int = 1600, a: float = 1.5, omega: float = 4 * np.pi, sigma: float = 0.25, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Toy G: varying required capacity — linear region (x < 0) vs compositional region (x >= 0).

    y = a*x + eps for x < 0 (linear, depth-1-sufficient); y = amplitude*sin(omega*x)*(1+x) + eps
    for x >= 0 (a sine modulated by a linear envelope — compositional, needs more depth), with
    the amplitude solved (:func:`toy_g_amplitude`) so both regions share the same TOTAL marginal
    variance (:func:`toy_g_region_variance`). Ground-truth region label: :func:`toy_g_region`.

    Args:
        n: sample count.
        a: linear-region slope.
        omega: compositional-region angular frequency (default = 2 full cycles over x in [0, 1]).
        sigma: shared noise std (same in both regions, so only the SIGNAL shape differs).
        seed: random seed.

    Returns:
        (x, y) with shapes (n, 1) and (n,), both float32.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n).reshape(-1, 1).astype(np.float32)
    amplitude = toy_g_amplitude(a, omega)
    signal = toy_g_signal(x.ravel(), a, omega, amplitude)
    y = (signal + sigma * rng.normal(0.0, 1.0, n)).astype(np.float32)
    return x, y


def make_toy_g_flat(n: int = 1600, a: float = 1.5, sigma: float = 0.25, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Toy G-flat: negative control for Toy G — same slope and noise, uniform (linear) complexity everywhere.

    y = a*x + eps for ALL x in [-1, 1] (Toy G's linear-region formula extended across the whole
    domain, instead of switching to the compositional form at x = 0). With the SAME `a` and
    `sigma` as :func:`make_toy_g`, this reproduces Toy G's per-region variance target
    (:func:`toy_g_region_variance`) exactly on both halves, while having no structure that could
    make one half legitimately need more capacity than the other — the bypass-confound negative
    control (smooth data everywhere = no per-input capacity signal should emerge).

    Args:
        n: sample count.
        a: slope (must match the `a` passed to :func:`make_toy_g` for the variance match to hold).
        sigma: noise std (must match the `sigma` passed to :func:`make_toy_g`).
        seed: random seed.

    Returns:
        (x, y) with shapes (n, 1) and (n,), both float32.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n).reshape(-1, 1).astype(np.float32)
    y = (a * x.ravel() + sigma * rng.normal(0.0, 1.0, n)).astype(np.float32)
    return x, y


def toy_h_f(x: np.ndarray, omega: float = 2 * np.pi, amplitude: float = 1.0) -> np.ndarray:
    """Toy H's FIXED mean function: amplitude*sin(omega*x). Identical formula on both sides of x = 0 — only the noise scale (:func:`toy_h_sigma`) differs by region."""
    xr = np.asarray(x, dtype=np.float64).ravel()
    return amplitude * np.sin(omega * xr)


def toy_h_sigma(x: np.ndarray, sigma_low: float = 0.1, sigma_high: float = 0.8) -> np.ndarray:
    """Toy H's per-input noise std: `sigma_low` (high SNR) for x < 0, `sigma_high` (low SNR) for x >= 0.

    Reuses Toy G's region boundary (:func:`toy_g_region`) so the same x = 0 split and per-half
    analysis code used for Toy G/G-flat applies unchanged to Toy H.
    """
    xr = np.asarray(x, dtype=np.float64).ravel()
    return np.where(toy_g_region(xr) == 0, sigma_low, sigma_high)


def make_toy_h(
    n: int = 1600,
    omega: float = 2 * np.pi,
    amplitude: float = 1.0,
    sigma_low: float = 0.1,
    sigma_high: float = 0.8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Toy H: the SNR dial — a FIXED function, noise sigma varying with x (k-as-resolution-dial thesis).

    y = toy_h_f(x) + toy_h_sigma(x)*eps, x ~ Uniform(-1, 1). The mean function is IDENTICAL on
    both sides of x = 0 (no structural change); only the noise scale changes. Capacity need
    should therefore vary with SNR, not with structure — the resolution-dial reading that WS1's
    k-selection work motivates (project_classification_as_regulariser: k is an SNR/difficulty-
    adaptive resolution dial, not an intrinsic component count).

    Args:
        n: sample count.
        omega: angular frequency of the fixed mean function (default = 1 full cycle over x in [-1, 1]).
        amplitude: amplitude of the fixed mean function.
        sigma_low: noise std for x < 0 (high SNR).
        sigma_high: noise std for x >= 0 (low SNR).
        seed: random seed.

    Returns:
        (x, y) with shapes (n, 1) and (n,), both float32.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n).reshape(-1, 1).astype(np.float32)
    f = toy_h_f(x.ravel(), omega, amplitude)
    sig = toy_h_sigma(x.ravel(), sigma_low, sigma_high)
    y = (f + sig * rng.normal(0.0, 1.0, n)).astype(np.float32)
    return x, y


# ---------------------------------------------------------------------------
# Toy T1 — the provably-deep-required toy (per-input selector program, WS-B task T1,
# docs/plans/perinput_selector_2026-07-10/EXECUTION_PLAN.md). Unlike Toy G, whose
# compositional region turned out to carry an incidentally tiny per-input depth signal,
# T1's region B is a Telgarsky-style composed tent map: iterating tent(z) = 1 - |2z - 1|
# n_iter times doubles the piece count each time (2**n_iter linear pieces at n_iter=5 ->
# 32), so at the F2-style trunk width used by the T1 driver (8) a depth-1 net (which
# realizes at most ~hidden_size linear kinks) PROVABLY cannot fit region B, while a
# depth >= 3 net can. x is on [0, 1] here (not [-1, 1] like G/G-flat/H above) so the
# 0.5 region boundary sits at the domain midpoint.
# ---------------------------------------------------------------------------


def toy_t1_region(x: np.ndarray) -> np.ndarray:
    """Ground-truth region label for Toy T1: 0 = linear (x < 0.5), 1 = composed-tent-map (x >= 0.5)."""
    xr = np.asarray(x, dtype=np.float64).ravel()
    return (xr >= 0.5).astype(int)


def _tent_map(z: np.ndarray) -> np.ndarray:
    """One application of the tent map on [0, 1]: tent(z) = 1 - |2z - 1| (tent(0)=tent(1)=0, tent(0.5)=1)."""
    return 1.0 - np.abs(2.0 * z - 1.0)


def toy_t1_tent_iterated(z: np.ndarray, n_iter: int = 5) -> np.ndarray:
    """`n_iter`-fold iterate of the tent map (:func:`_tent_map`) on [0, 1].

    Each application doubles the number of linear pieces (2 -> 4 -> 8 -> 16 -> 32 at the
    default `n_iter=5`) — the standard Telgarsky (2016) depth-separation construction: a
    network needs depth growing with `log2` of the piece count to represent it exactly,
    so a shallow net cannot fit this region no matter how wide, while a deep-enough net can.
    """
    out = np.asarray(z, dtype=np.float64).copy()
    for _ in range(n_iter):
        out = _tent_map(out)
    return out


def toy_t1_signal(x: np.ndarray, a: float = 1.5, n_iter: int = 5) -> np.ndarray:
    """Toy T1's noise-free signal: linear for x < 0.5, a composed tent map for x >= 0.5.

    y = a*x for x < 0.5 (linear, depth-1-sufficient); y = tent^(n_iter)(2*(x-0.5)) for x >= 0.5
    (the composed tent map, :func:`toy_t1_tent_iterated` — 2**n_iter linear pieces). Ground-truth
    region label: :func:`toy_t1_region`.
    """
    xr = np.asarray(x, dtype=np.float64).ravel()
    linear = a * xr
    compositional = toy_t1_tent_iterated(2.0 * (xr - 0.5), n_iter)
    return np.where(xr < 0.5, linear, compositional)


def make_toy_t1(n: int = 1600, a: float = 1.5, n_iter: int = 5, sigma: float = 0.1, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Toy T1: the provably-deep-required toy (region B needs depth by construction, not incidentally).

    y = a*x + eps for x < 0.5 (linear, depth-1-sufficient); y = tent^(n_iter)(2*(x-0.5)) + eps for
    x >= 0.5 (n_iter-fold composed tent map -> 2**n_iter linear pieces; PROVABLY needs depth >= 3 at
    trunk width 8 — a depth-1 net realizes at most ~hidden_size linear kinks, far short of 32), with
    eps ~ N(0, sigma**2) shared across both regions (only the SIGNAL's required capacity differs,
    matching Toy G/H's convention of isolating one axis of variation per toy).

    Args:
        n: sample count.
        a: linear-region slope.
        n_iter: tent-map iteration count (default 5 -> 32 pieces in region B).
        sigma: noise std (same in both regions).
        seed: random seed.

    Returns:
        (x, y) with shapes (n, 1) and (n,), both float32.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n).reshape(-1, 1).astype(np.float32)
    signal = toy_t1_signal(x.ravel(), a, n_iter)
    y = (signal + sigma * rng.normal(0.0, 1.0, n)).astype(np.float32)
    return x, y


def sample_toy_t1_given_x(x: np.ndarray, sigma: float = 0.1, seed: int = 0, a: float = 1.5, n_iter: int = 5) -> np.ndarray:
    """Gold conditional sampler for Toy T1: fresh y | x draws at caller-supplied x (for gold reads).

    Args:
        x: input coordinates to condition on (any shape; raveled internally).
        sigma: noise std (must match :func:`make_toy_t1`'s `sigma` for a like-for-like gold read).
        seed: random seed for the noise draw.
        a: linear-region slope (must match :func:`make_toy_t1`'s `a`).
        n_iter: tent-map iteration count (must match :func:`make_toy_t1`'s `n_iter`).

    Returns:
        (n,) float32 y draws, one per input row.
    """
    rng = np.random.default_rng(seed)
    xr = np.asarray(x, dtype=np.float64).ravel()
    signal = toy_t1_signal(xr, a, n_iter)
    return (signal + sigma * rng.normal(0.0, 1.0, xr.shape[0])).astype(np.float32)
