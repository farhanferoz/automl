"""D1 — the depth-programme positive control: a depth-hungry-but-GD-learnable regression toy.

(`docs/plans/capacity_programme/depth.md` task D1.) The old depth lane died for lack of one: its
tent-map target was representable at depth but GD-UNLEARNABLE at every depth (packed self-similar
frequency content hits the same spectral-bias wall `nested_width_net.py`'s `make_hetero` docstring
records for width) — so every depth-architecture result downstream of it was a null. This module
constructs a fresh candidate, tries three constructions IN ORDER, and probes each against four
PRE-REGISTERED bars (P1-P4 below) before any architecture work is allowed to start.

**Region convention** (mirrors `nested_width_net.py`'s `make_hetero`): every candidate has an EASY
region (width/depth-flat: a plain straight line) and a HARD region (the candidate's depth
construction). Both carry i.i.d. Gaussian noise, std `NOISE_SIGMA=0.05`, so the analytic noise
floor is `NOISE_SIGMA**2` in BOTH regions — held-out MSE is measured in ORIGINAL y-units against
this floor, exactly the `sinc_width_experiment.py::_fit_bar_mse` convention (train/eval standardize
x and y using TRAIN-set statistics only, then un-standardize predictions before scoring).

**Candidates** (`Candidate` enum; `--candidate` selects one; tried in this fixed order per the
plan, first pass wins):

  1. `GENTLE_COMPOSITION` — hard region `f_4(x) = g^{o4}(x)`, `g(u) = sin(1.5*u)`, domain scaled
     to `GENTLE_R` so `g`'s own frequency stays O(1) (`nested_width_net.py::make_hetero`'s
     "spread the periods" lesson). `g(0) = 0` is a fixed point of the whole composition, so the
     hard region is 0 at the `x=0` seam, matching the easy line's value there — continuous by
     construction, no explicit blending needed.
  2. `HIERARCHICAL_SPLINE` — hard region is a TWO-LEVEL composition `s(s(u))` of the smooth
     3-segment map `s(u) = sin(1.5*pi*u)**2` on `[0,1]` (0 at both `u=0` and its own composition,
     so segment count grows 3x3=9 across the two levels while each level stays low-frequency,
     avoiding the tent map's exploding-slope self-similarity). `s(0)=0` gives the same free seam
     continuity as candidate 1.
  3. `MULTIPLICATIVE_2D` — 2-D input; hard region `f(x1,x2) = sin(a*x1)*sin(a*x2)` (a PRODUCT of
     univariate ridge functions, which a single hidden layer of ridge functions cannot compute
     exactly and must approximate at a width cost scaling with the number of quadrature points
     needed for the product identity — the textbook depth-separation example); easy region is the
     ADDITIVE `sin(a*x1)+sin(a*x2)` (representable by a single hidden layer with NO product needed).
     Because the two formulas disagree at any shared boundary (`sin(a*x1)+sin(a*x2) != 0` in
     general even where the product vanishes), continuity is enforced with an explicit smoothstep
     BLEND over `x1 in [-MULT_BAND, +MULT_BAND]`; that band is labelled `REGION_TRANSITION` and
     EXCLUDED from every probe bar (only the pure-additive and pure-multiplicative sides count).

**Candidate 4 — fixed-width / vary-depth** (`FIXEDWIDTH_HETERO`; the follow-up after candidates 1-3
hit the kill criterion — see `docs/depth_capacity/depth_toy_negative_note.md`). Candidates 1-3 all
tried to INVENT a function that is depth-hungry in an absolute sense; two of them died on the
spectral-bias wall (composing a sine saturates / seed-unstable optimization) — i.e. the target was
not cleanly GD-learnable at ALL. Candidate 4 inverts the strategy: it reuses the WIDTH toy's
`nested_width_net.make_hetero` target verbatim — a flat-easy line spliced to `0.5*sin(x)` over
`[0, 4*pi]` (2 native-frequency periods), which is KNOWN GD-learnable by a wide shallow net (that is
the whole width program) and so clears the spectral-bias wall by construction. The depth question is
then not "is it learnable" but "does DEPTH substitute for WIDTH at a FIXED small width `w`": pick `w`
small enough that a depth-1 net at width `w` fits the flat region but STALLS on the sine, and ask
whether adding depth AT THE SAME WIDTH `w` unlocks the sine. Because the target is the same one a
wide-shallow net demonstrably fits, a depth-1-narrow stall is a genuine depth/width tradeoff, not
un-learnability — a distinction the pre-registered CONTROL bar makes explicit.

**Candidate 4 probe bars differ from P1-P4 above** (`_check_fixedwidth_bars`; the STALL and CONTROL
roles are SWAPPED relative to candidates 1-3, because here the wide-shallow net is the POSITIVE
control, not the thing that must stall):
  - P1 learnable-deep: depth-`FIXEDWIDTH_DEPTHS[-1]` net at the FIXED small width `FIXEDWIDTH_W`
    reaches `M_hard <= FIT_PASS_MULTIPLE * FLOOR` (depth compensates for the missing width).
  - P2 depth-hungry: the depth-1 net AT THE SAME width `FIXEDWIDTH_W` STALLS at
    `M_hard >= HUNGRY_STALL_MULTIPLE * FLOOR` (shallow-narrow cannot fit the sine).
  - P3 graded: `M_hard` orders monotonically (within slack) as depth increases 1 -> deepest at `w`.
  - P4 easy-region-flat: the depth-1 net at `w` fits the EASY region to `M_easy <= EASY_FLAT_MULTIPLE
    * FLOOR` (the flat line is width/depth-flat even at the small width).
  - CONTROL wide-shallow-fits: a WIDE shallow net (width `FIXEDWIDTH_CONTROL_W >> w`, depth 1) reaches
    `M_hard <= FIT_PASS_MULTIPLE * FLOOR` — proves the target IS learnable, so P2's stall is a
    depth-for-width substitution, not the candidate-1/2 un-learnability failure. `FIXEDWIDTH_W` is
    LOCKED to the seed-0 pilot (`pilot_fixedwidth`) result and recorded in `depth.md`.

**Probe bars** (candidates 1-3; `run_probe`; PRE-REGISTERED, `depth.md` task D1 spec, 2 seeds each):
  - P1 learnable-deep: the depth-`PROBE_DEPTHS[-1]` (=4) NARROW net (width `NARROW_WIDTH`=8 per
    layer) reaches `M_hard <= FIT_PASS_MULTIPLE * NOISE_SIGMA**2` at TRUSTWORTHY convergence
    (`convergence.py::ConvergenceTracker`, full-trajectory rule — a `hit_cap` or still-improving
    run may NOT be used to conclude a pass or a fail).
  - P2 depth-hungry: a param-matched WIDE-SHALLOW net (1 hidden layer, width chosen so its
    parameter count is within `PARAM_MATCH_TOL`=5% of the depth-4 net's) STALLS at
    `M_hard >= HUNGRY_STALL_MULTIPLE * NOISE_SIGMA**2` at trustworthy convergence.
  - P3 graded: `M_hard` at the intermediate probe depths (2, 3) orders monotonically (within
    `P3_MONOTONIC_SLACK` noise tolerance) between the wide-shallow (P2) level and the depth-4
    (P1) level.
  - P4 easy-region-flat: the depth-1 narrow net fits the EASY region to
    `M_easy <= EASY_FLAT_MULTIPLE * NOISE_SIGMA**2`.

A candidate PASSES only if all four hold on BOTH seeds in `SEEDS`. Kill criterion (all three
candidates exhaust probes on 2 seeds with no pass): STOP, do not build any depth architecture on a
non-hungry toy, write the negative note + escalate — see `depth.md` task D1 for the full protocol.

**Non-goals** (this file only): no architecture/3-arm work (task D2+); MSE-only, no NLL; no
depth > 6; the tent map is not reused here (refuted per `docs/plans/width_mse_2026-07-16/
EXECUTION_PLAN.md` §9 roadmap item 1).

Usage:
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/depth_toy.py --selftest
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/depth_toy.py \
        --candidate gentle_composition --seed 0
"""

from __future__ import annotations

import argparse
import enum
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root, so `import automl_package` works

import convergence as cvg  # noqa: E402 — `ConvergenceTracker`/`fit_to_convergence`, the trajectory-discipline gate
from nested_width_net import HETERO_NOISE_SIGMA, make_hetero  # noqa: E402 — candidate 4 reuses the WIDTH toy's known-learnable flat+sine target

# ---------------------------------------------------------------------------
# Region labels — closed set, NAMED (not magic ints), same convention as
# `sinc_width_experiment.py:336` (`REGION_EASY, REGION_HARD, REGION_NOISY_EASY = 0, 1, 2`).
# ---------------------------------------------------------------------------
REGION_EASY = 0
REGION_HARD = 1
REGION_TRANSITION = 2  # candidate 3 only: the smoothstep blend band, excluded from every bar

NOISE_SIGMA = 0.05
FLOOR = NOISE_SIGMA**2

NARROW_WIDTH = 8  # "width fixed at 8/layer" (depth.md P1 spec)
PROBE_DEPTHS = (1, 2, 3, 4)  # uniform probe ladder across all 3 candidates (P1 reads depth=4)
PARAM_MATCH_TOL = 0.05  # P2 spec: wide-shallow width matched within 5% of the depth-4 param count

FIT_PASS_MULTIPLE = 1.25  # P1
HUNGRY_STALL_MULTIPLE = 2.0  # P2
EASY_FLAT_MULTIPLE = 1.3  # P4
P3_MONOTONIC_SLACK = 1.10  # each successive term in the P3 ladder may exceed the previous by at most 10%

SEEDS = (0, 1)
N_TRAIN = 1000
N_VAL = 500  # fresh held-out draw (different RNG stream, not a split of the train set)

LR = 1e-2
MAX_EPOCHS = 30000
CHECK_EVERY = 250
PATIENCE = 8
MIN_DELTA = 1e-3  # recalibrated from an initial 3e-5 (D1 pilot, `hierarchical_spline` seed0): full-batch Adam's
# own step-to-step val-loss jitter on the wide-shallow arm is O(1e-4..1e-3) in standardized-y units even once
# genuinely plateaued, so 3e-5 never let `converged` fire (patience kept resetting on noise) though the
# trajectory was visibly flat over 30+ checkpoints. 1e-3 is far below any genuine early-training improvement
# rate and comfortably above the observed noise band -- one-time recalibration, frozen after this point
# (mirrors MASTER Decision 9 / the D3 Step-1 recalibration allowance, applied here at toy-construction time).

GENTLE_G_COEF = 1.5
GENTLE_D = 4
GENTLE_R = 3.0  # keeps g's own frequency O(1) over the domain (extrema~2, |max deriv|~5, verified numerically)

SPLINE_FREQ = 1.5
SPLINE_R = 4.0
SPLINE_AMPLITUDE = 0.5

MULT_A = 1.0  # sin(a*x_i) keeps ~1 period over the domain (O(1) input-space frequency)
MULT_R = 3.0
MULT_BAND = 0.5  # half-width of the smoothstep continuity blend in x1

# Candidate 4 (fixed-width / vary-depth). The target is `nested_width_net.make_hetero` verbatim
# (flat line + `0.5*sin(x)` over [0, 4*pi]); its noise std is `HETERO_NOISE_SIGMA`, asserted equal to
# `NOISE_SIGMA` in the selftest so `FLOOR` stays the shared analytic floor. `FIXEDWIDTH_W` is a
# PLACEHOLDER pending the seed-0 pilot (`pilot_fixedwidth`): the locked value is the smallest width at
# which depth-1 STALLS (>= HUNGRY_STALL_MULTIPLE*FLOOR) while depth-`FIXEDWIDTH_DEPTHS[-1]` FITS
# (<= FIT_PASS_MULTIPLE*FLOOR); record the locked value + pilot evidence in `depth.md`.
FIXEDWIDTH_W = 4  # small fixed width (depth-1 must stall, deepest must fit) — PLACEHOLDER, lock after the pilot.
FIXEDWIDTH_CONTROL_W = 32  # wide-shallow CONTROL width (>> FIXEDWIDTH_W); must FIT the sine (proves learnability, not depth).
FIXEDWIDTH_DEPTHS = (1, 2, 3, 4)  # depth ladder at the FIXED small width; P1 reads the deepest, P2 reads depth 1.

_CONTINUITY_TOL = 1e-4  # selftest seam-continuity tolerance (noise-free signal, not the noisy sample)
_SEAM_WINDOW = 0.1  # candidate-4 selftest: |x| < this is the near-seam band for the noise-free continuity read
_SEAM_SIGNAL_TOL = 0.1  # candidate-4 selftest: noise-free |y| must stay below this within the seam band (both branches -> 0 at x=0)
_STD_FLOOR = 1e-8  # below this, treat a column's std as degenerate (constant column) and use scale 1.0 instead
_1D_INPUT_NDIM = 2  # `x.ndim==2` (an `(n, d)` array) marks a multi-column input; `d=1` toys are stored `(n,)`
_SELFTEST_DEPTH4_PARAMS = 241  # known-answer: build_narrow(depth=4, width=8, in_dim=1) param count
_SELFTEST_MATCH_WIDTH = 80  # known-answer: match_wide_width(241, in_dim=1) -> exact match at width=80
_SELFTEST_MATCH_DIFF_TOL = 1e-9  # the width-80 match above should be an EXACT param-count match

DEFAULT_OUT_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "D_TOY_PROBES")


class Candidate(enum.StrEnum):
    """Which depth-toy construction a probe run uses (closed set; tried in THIS order, first pass wins)."""

    GENTLE_COMPOSITION = "gentle_composition"
    HIERARCHICAL_SPLINE = "hierarchical_spline"
    MULTIPLICATIVE_2D = "multiplicative_2d"
    FIXEDWIDTH_HETERO = "fixedwidth"  # candidate 4 (follow-up): fixed small width, vary DEPTH, over make_hetero's flat+sine


# Candidates tried IN ORDER for the original kill-set (first pass wins). Candidate 4 is a SEPARATE
# follow-up (dispatched after 1-3 hit the kill criterion), NOT part of this ordered sequence — it has
# its own probe driver (`run_fixedwidth_probe`) with swapped stall/control bar semantics.
CANDIDATE_ORDER = (Candidate.GENTLE_COMPOSITION, Candidate.HIERARCHICAL_SPLINE, Candidate.MULTIPLICATIVE_2D)


# ---------------------------------------------------------------------------
# Signal generators (noise-free) — exposed separately so the selftest can probe seam continuity
# directly, without re-deriving the formula from the noisy generator.
# ---------------------------------------------------------------------------


def _gentle_g(u: np.ndarray, coef: float = GENTLE_G_COEF) -> np.ndarray:
    """`g(u) = sin(coef*u)`; `g(0) = 0` is the fixed point that keeps the composition seam-continuous."""
    return np.sin(coef * u)


def _gentle_signal(x: np.ndarray, depth: int = GENTLE_D, r: float = GENTLE_R) -> np.ndarray:
    """Candidate 1's noise-free `y(x)`: easy line for `x<0`, `g` composed `depth` times for `x>=0`.

    `g(0)=0` at every composition level, so the hard branch is exactly 0 at `x=0`, matching the easy
    branch's value there (`(0.5/r)*0=0`) — the seam is continuous by construction.
    """
    hard = x.copy()
    for _ in range(depth):
        hard = _gentle_g(hard)
    return np.where(x < 0.0, (0.5 / r) * x, hard)


def make_gentle_composition(n: int, seed: int, depth: int = GENTLE_D, r: float = GENTLE_R, sigma: float = NOISE_SIGMA) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Candidate 1 — gentle composition: `x ~ Uniform(-r, r)`; region 0 = easy line, region 1 = `g^{oD}(x)`.

    Args:
        n: number of points.
        seed: RNG seed (`np.random.default_rng`).
        depth: number of times `g` is composed in the hard region (default `GENTLE_D=4`).
        r: domain half-width (default `GENTLE_R`; keeps `g`'s frequency O(1), see module docstring).
        sigma: Gaussian noise std, common-mode across both regions (default `NOISE_SIGMA`).

    Returns:
        `(x, y, region)`, each shape `(n,)`; `x`/`y` float32, `region` int in `{REGION_EASY, REGION_HARD}`.
        `x` is NOT standardized (callers standardize on TRAIN-set stats, `nested_width_net.py::make_hetero`
        convention).
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-r, r, n)
    y_signal = _gentle_signal(x, depth=depth, r=r)
    y = (y_signal + rng.normal(0.0, sigma, n)).astype(np.float32)
    region = (x >= 0.0).astype(int)
    return x.astype(np.float32), y, region


def _spline_s(u: np.ndarray, freq: float = SPLINE_FREQ) -> np.ndarray:
    """3-segment smooth map on `[0,1]`: `sin(freq*pi*u)**2`; `s(0)=0` is its own fixed point."""
    return np.sin(freq * math.pi * u) ** 2


def _spline_signal(x: np.ndarray, r: float = SPLINE_R, amplitude: float = SPLINE_AMPLITUDE) -> np.ndarray:
    """Candidate 2's noise-free `y(x)`: easy line for `x<0`; TWO-LEVEL composition `s(s(u))` for `x>=0`.

    `u = clip(x/r, 0, 1)` is the hard-region local coordinate; `s(0)=0` at both composition levels
    keeps the hard branch at exactly 0 when `x=0`, matching the easy branch there.
    """
    u = np.clip(x / r, 0.0, 1.0)
    s1 = _spline_s(u)
    s2 = _spline_s(s1)
    return np.where(x < 0.0, (0.5 / r) * x, amplitude * s2)


def make_hierarchical_spline(n: int, seed: int, r: float = SPLINE_R, amplitude: float = SPLINE_AMPLITUDE, sigma: float = NOISE_SIGMA) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Candidate 2 — hierarchical piecewise-smooth: `x ~ Uniform(-r, r)`; hard region is `s(s(u))`.

    Segment count grows multiplicatively with composition depth (3 segments -> 9 across the two
    levels) while each level stays low-frequency (bounded slope), avoiding the tent map's exploding
    self-similar slope (refuted: GD-unlearnable at every depth).

    Args:
        n: number of points.
        seed: RNG seed.
        r: domain half-width (hard region's local coordinate is `x/r`, clipped to `[0,1]`).
        amplitude: hard-region signal scale (default `SPLINE_AMPLITUDE`, matches `make_hetero`'s 0.5).
        sigma: Gaussian noise std, common-mode (default `NOISE_SIGMA`).

    Returns:
        `(x, y, region)` as `make_gentle_composition`.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-r, r, n)
    y_signal = _spline_signal(x, r=r, amplitude=amplitude)
    y = (y_signal + rng.normal(0.0, sigma, n)).astype(np.float32)
    region = (x >= 0.0).astype(int)
    return x.astype(np.float32), y, region


def _smoothstep(t: np.ndarray) -> np.ndarray:
    """Standard cubic smoothstep, clipped to `[0,1]` outside its domain: `3t^2 - 2t^3`."""
    tc = np.clip(t, 0.0, 1.0)
    return tc * tc * (3.0 - 2.0 * tc)


def _mult_blend(x1: np.ndarray, band: float = MULT_BAND) -> np.ndarray:
    """0 for `x1<=-band`, 1 for `x1>=+band`, smoothstep in between — the additive/multiplicative crossfade."""
    return _smoothstep((x1 + band) / (2.0 * band))


def _mult_signal(x1: np.ndarray, x2: np.ndarray, a: float = MULT_A, band: float = MULT_BAND) -> np.ndarray:
    """Candidate 3's noise-free `y(x1,x2)`: smoothstep-blended additive (`x1<=-band`) <-> multiplicative (`x1>=+band`)."""
    additive = np.sin(a * x1) + np.sin(a * x2)
    multiplicative = np.sin(a * x1) * np.sin(a * x2)
    blend = _mult_blend(x1, band=band)
    return (1.0 - blend) * additive + blend * multiplicative


def make_multiplicative_2d(
    n: int, seed: int, r: float = MULT_R, a: float = MULT_A, band: float = MULT_BAND, sigma: float = NOISE_SIGMA
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Candidate 3 — 2-D multiplicative: `f(x1,x2)=sin(a*x1)*sin(a*x2)`.

    Needs effective depth>=2 under tanh MLPs — a single hidden layer of ridge functions cannot compute
    a product of two independent univariate factors exactly; easy region is additive
    `sin(a*x1)+sin(a*x2)` (a plain sum of ridge functions, exactly what one hidden layer IS).

    Region is 0 (easy, `x1<=-band`), 1 (hard, `x1>=+band`), or `REGION_TRANSITION=2` (the smoothstep
    blend band, `|x1|<band`) — the transition band is excluded from every probe bar; it exists only to
    keep the target continuous (the two formulas disagree at any shared boundary point in general).

    Args:
        n: number of points.
        seed: RNG seed.
        r: domain half-width per axis (`x1, x2 ~ Uniform(-r, r)` independently).
        a: frequency inside each `sin` (default `MULT_A`; keeps each factor's own frequency O(1)).
        band: half-width of the smoothstep transition in `x1` (default `MULT_BAND`).
        sigma: Gaussian noise std, common-mode (default `NOISE_SIGMA`).

    Returns:
        `(x, y, region)`: `x` shape `(n, 2)` float32 (columns `x1, x2`); `y` shape `(n,)` float32;
        `region` int `(n,)` in `{REGION_EASY, REGION_HARD, REGION_TRANSITION}`.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-r, r, n)
    x2 = rng.uniform(-r, r, n)
    y_signal = _mult_signal(x1, x2, a=a, band=band)
    y = (y_signal + rng.normal(0.0, sigma, n)).astype(np.float32)
    region = np.select([x1 <= -band, x1 >= band], [REGION_EASY, REGION_HARD], default=REGION_TRANSITION)
    x = np.stack([x1, x2], axis=1).astype(np.float32)
    return x, y, region.astype(int)


def make_fixedwidth_hetero(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Candidate 4 target — the WIDTH toy's flat-line + 2-period-sine (`nested_width_net.make_hetero`).

    Reused VERBATIM (default `r=4*pi`, `sigma=HETERO_NOISE_SIGMA`, asserted `== NOISE_SIGMA` so the
    analytic floor stays `FLOOR = NOISE_SIGMA**2`): region 0 = easy flat line `(0.5/r)*x` (width- AND
    depth-flat — 1 node fits it), region 1 = hard `0.5*sin(x)` over `[0, 4*pi]` (2 native-frequency
    periods; KNOWN GD-learnable by a WIDE shallow net — that is the entire width program, so it clears
    the spectral-bias wall candidates 1/2 hit). The fixed-width/vary-depth probe
    (`run_fixedwidth_probe`) asks whether DEPTH substitutes for WIDTH on exactly this target.
    """
    return make_hetero(n, seed)


_GENERATORS = {
    Candidate.GENTLE_COMPOSITION: make_gentle_composition,
    Candidate.HIERARCHICAL_SPLINE: make_hierarchical_spline,
    Candidate.MULTIPLICATIVE_2D: make_multiplicative_2d,
    Candidate.FIXEDWIDTH_HETERO: make_fixedwidth_hetero,
}


def generate(candidate: Candidate, n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dispatches to the generator for `candidate` (`_GENERATORS`, the closed set above)."""
    return _GENERATORS[candidate](n, seed)


# ---------------------------------------------------------------------------
# Net builders — a generic depth-d "narrow" MLP family and a width-matched "wide-shallow" twin.
# ---------------------------------------------------------------------------


def build_narrow(depth: int, width: int = NARROW_WIDTH, in_dim: int = 1) -> nn.Sequential:
    """`Linear(in_dim, width) -> Tanh -> [Linear(width, width) -> Tanh] * (depth-1) -> Linear(width, 1)`."""
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")
    layers: list[nn.Module] = [nn.Linear(in_dim, width), nn.Tanh()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), nn.Tanh()]
    layers += [nn.Linear(width, 1)]
    return nn.Sequential(*layers)


def build_wide_shallow(width: int, in_dim: int = 1) -> nn.Sequential:
    """`Linear(in_dim, width) -> Tanh -> Linear(width, 1)` — the P2 param-matched positive control."""
    return nn.Sequential(nn.Linear(in_dim, width), nn.Tanh(), nn.Linear(width, 1))


def count_params(module: nn.Module) -> int:
    """Total learnable scalar parameter count (weights + biases)."""
    return sum(p.numel() for p in module.parameters())


def match_wide_width(target_params: int, in_dim: int, tol: float = PARAM_MATCH_TOL) -> tuple[int, float]:
    """Smallest-relative-error 1-hidden-layer width whose param count is closest to `target_params`.

    `params(w) = w*(in_dim+2) + 1` (`Linear(in_dim,w)`: `w*in_dim+w`; `Linear(w,1)`: `w+1`), linear
    in `w`, so a direct bounded search suffices (no need for anything fancier).

    Returns:
        `(width, relative_diff)`. Caller (`run_probe`) records `relative_diff` in the probe JSON even
        when it exceeds `tol`, so a failed match is visible rather than silently accepted.
    """
    best_w, best_diff = 1, math.inf
    for w in range(1, 5000):
        p = w * (in_dim + 2) + 1
        diff = abs(p - target_params) / target_params
        if diff < best_diff:
            best_w, best_diff = w, diff
        if p > target_params and diff > best_diff:
            break  # params(w) is monotonically increasing in w; past the minimum, stop searching
    if best_diff > tol:
        print(f"[depth_toy] WARNING: match_wide_width could not match within tol={tol}: width={best_w} relative_diff={best_diff:.4f}")
    return best_w, best_diff


# ---------------------------------------------------------------------------
# Standardization (TRAIN-set stats only, applied to held-out data too) + convergence-gated MSE training.
# ---------------------------------------------------------------------------


def _fit_norm(x: np.ndarray, y: np.ndarray) -> dict:
    """Per-column x stats + scalar y stats from the TRAIN set (`nested_width_net.py::make_hetero` convention)."""
    x2 = x.reshape(x.shape[0], -1)
    mx = x2.mean(axis=0)
    sx = x2.std(axis=0)
    sx = np.where(sx < _STD_FLOOR, 1.0, sx)
    my, sy = float(y.mean()), float(y.std())
    sy = sy if sy > _STD_FLOOR else 1.0
    return {"mx": mx, "sx": sx, "my": my, "sy": sy}


def _standardize_x(x: np.ndarray, norm: dict) -> np.ndarray:
    x2 = x.reshape(x.shape[0], -1)
    return (x2 - norm["mx"]) / norm["sx"]


def train_mse(
    net: nn.Module,
    x_tr_n: np.ndarray,
    y_tr_n: np.ndarray,
    x_val_n: np.ndarray,
    y_val_n: np.ndarray,
    device: str = "cpu",
    lr: float = LR,
    max_epochs: int = MAX_EPOCHS,
    check_every: int = CHECK_EVERY,
    patience: int = PATIENCE,
    min_delta: float = MIN_DELTA,
) -> cvg.ConvergenceResult:
    """Trains `net` (in place) by full-batch Adam on standardized `(x, y)`, gated by `convergence.fit_to_convergence`.

    Loss and held-out metric are both plain MSE in STANDARDIZED-y units (no NLL, no learned variance —
    MASTER Decision 2, MSE-only). Caller un-standardizes predictions before comparing to `FLOOR`.
    """
    net.to(device)
    x_tr_t = torch.as_tensor(x_tr_n, dtype=torch.float32, device=device)
    y_tr_t = torch.as_tensor(y_tr_n, dtype=torch.float32, device=device).reshape(-1)
    x_val_t = torch.as_tensor(x_val_n, dtype=torch.float32, device=device)
    y_val_t = torch.as_tensor(y_val_n, dtype=torch.float32, device=device).reshape(-1)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    def step_fn() -> None:
        opt.zero_grad()
        pred = net(x_tr_t).squeeze(-1)
        loss = torch.mean((pred - y_tr_t) ** 2)
        loss.backward()
        opt.step()

    def val_fn() -> float:
        net.eval()
        with torch.no_grad():
            pred = net(x_val_t).squeeze(-1)
            v = torch.mean((pred - y_val_t) ** 2).item()
        net.train()
        return v

    result = cvg.fit_to_convergence(net, step_fn, val_fn, max_epochs=max_epochs, check_every=check_every, patience=patience, min_delta=min_delta)
    net.eval()
    return result


def _region_mse_orig(net: nn.Module, norm: dict, x: np.ndarray, y: np.ndarray, region: np.ndarray, region_id: int, device: str = "cpu") -> float | None:
    """Held-out MSE in ORIGINAL y-units, restricted to `region==region_id`; `None` if the mask is empty."""
    mask = region == region_id
    if not mask.any():
        return None
    x_n = _standardize_x(x, norm)
    x_t = torch.as_tensor(x_n, dtype=torch.float32, device=device)
    net.eval()
    with torch.no_grad():
        pred_n = net(x_t).squeeze(-1).cpu().numpy()
    pred_orig = pred_n * norm["sy"] + norm["my"]
    err2 = (pred_orig[mask] - y[mask]) ** 2
    return float(err2.mean())


# ---------------------------------------------------------------------------
# Probe driver — trains the full depth ladder + the param-matched wide-shallow net for ONE
# (candidate, seed), evaluates region-wise held-out MSE, and checks bars P1-P4.
# ---------------------------------------------------------------------------


def run_probe(candidate: Candidate, seed: int, device: str = "cpu") -> dict:
    """Runs one full (candidate, seed) probe.

    Trains `PROBE_DEPTHS` narrow nets + 1 wide-shallow net, evaluates held-out region MSE for each, and
    checks all four pre-registered bars. Returns a JSON-able dict (see module docstring for the bar
    definitions); every net's convergence result is included in full so a non-`trustworthy` net is
    visible rather than silently swallowed.
    """
    x_tr, y_tr, _region_tr = generate(candidate, N_TRAIN, seed)
    x_val, y_val, region_val = generate(candidate, N_VAL, seed + 10_000)  # independent held-out draw
    in_dim = x_tr.shape[1] if x_tr.ndim == _1D_INPUT_NDIM else 1

    norm = _fit_norm(x_tr, y_tr)
    x_tr_n = _standardize_x(x_tr, norm)
    y_tr_n = (y_tr - norm["my"]) / norm["sy"]
    x_val_n = _standardize_x(x_val, norm)
    y_val_n = (y_val - norm["my"]) / norm["sy"]

    depth_max = PROBE_DEPTHS[-1]
    torch.manual_seed(1000 * seed + depth_max)
    depth_max_params = count_params(build_narrow(depth_max, NARROW_WIDTH, in_dim))
    wide_width, wide_diff = match_wide_width(depth_max_params, in_dim)

    narrow_results: dict[int, dict] = {}
    trustworthy_flags: list[bool] = []
    for d in PROBE_DEPTHS:
        torch.manual_seed(1000 * seed + d)
        net = build_narrow(d, NARROW_WIDTH, in_dim)
        result = train_mse(net, x_tr_n, y_tr_n, x_val_n, y_val_n, device=device)
        m_hard = _region_mse_orig(net, norm, x_val, y_val, region_val, REGION_HARD, device=device)
        m_easy = _region_mse_orig(net, norm, x_val, y_val, region_val, REGION_EASY, device=device)
        trustworthy_flags.append(bool(result.trustworthy))
        narrow_results[d] = {
            "params": count_params(net),
            "m_hard": m_hard,
            "m_easy": m_easy,
            "convergence": result.summary(),
        }

    torch.manual_seed(1000 * seed + 999)
    wide_net = build_wide_shallow(wide_width, in_dim)
    wide_result = train_mse(wide_net, x_tr_n, y_tr_n, x_val_n, y_val_n, device=device)
    wide_m_hard = _region_mse_orig(wide_net, norm, x_val, y_val, region_val, REGION_HARD, device=device)
    wide_m_easy = _region_mse_orig(wide_net, norm, x_val, y_val, region_val, REGION_EASY, device=device)
    wide_out = {
        "width": wide_width,
        "params": count_params(wide_net),
        "param_diff_frac": wide_diff,
        "m_hard": wide_m_hard,
        "m_easy": wide_m_easy,
        "convergence": wide_result.summary(),
    }

    bars = _check_bars(narrow_results, wide_out, depth_max)

    return {
        "candidate": candidate.value,
        "seed": seed,
        "noise_sigma": NOISE_SIGMA,
        "floor": FLOOR,
        "n_train": N_TRAIN,
        "n_val": N_VAL,
        "probe_depths": list(PROBE_DEPTHS),
        "narrow_width": NARROW_WIDTH,
        "narrow": {str(d): v for d, v in narrow_results.items()},
        "wide_shallow": wide_out,
        "bars": bars,
        "all_pass": all(b["pass"] for b in bars.values()),
        "all_trustworthy": bool(wide_result.trustworthy) and all(trustworthy_flags),
    }


def _check_bars(narrow_results: dict[int, dict], wide_out: dict, depth_max: int) -> dict:
    """P1-P4 verdicts from already-trained probe results (`run_probe`'s per-net dicts)."""
    m_hard_max = narrow_results[depth_max]["m_hard"]
    ratio_p1 = m_hard_max / FLOOR
    p1 = {"depth": depth_max, "m_hard": m_hard_max, "ratio_to_floor": ratio_p1, "pass": bool(ratio_p1 <= FIT_PASS_MULTIPLE)}

    m_hard_wide = wide_out["m_hard"]
    ratio_p2 = m_hard_wide / FLOOR
    p2 = {"width": wide_out["width"], "m_hard": m_hard_wide, "ratio_to_floor": ratio_p2, "pass": bool(ratio_p2 >= HUNGRY_STALL_MULTIPLE)}

    ladder = [wide_out["m_hard"]] + [narrow_results[d]["m_hard"] for d in PROBE_DEPTHS]
    monotonic = all(ladder[i + 1] <= ladder[i] * P3_MONOTONIC_SLACK for i in range(len(ladder) - 1))
    p3 = {
        "ladder_m_hard": {"wide_shallow": wide_out["m_hard"], **{str(d): narrow_results[d]["m_hard"] for d in PROBE_DEPTHS}},
        "slack_multiple": P3_MONOTONIC_SLACK,
        "pass": bool(monotonic),
    }

    m_easy_d1 = narrow_results[1]["m_easy"]
    ratio_p4 = m_easy_d1 / FLOOR
    p4 = {"depth": 1, "m_easy": m_easy_d1, "ratio_to_floor": ratio_p4, "pass": bool(ratio_p4 <= EASY_FLAT_MULTIPLE)}

    return {"p1_learnable_deep": p1, "p2_depth_hungry": p2, "p3_graded": p3, "p4_easy_region_flat": p4}


def _result_path(out_dir: str, candidate: Candidate, seed: int) -> str:
    return os.path.join(out_dir, f"depth_toy_probe_{candidate.value}_seed{seed}.json")


def run_and_save_probe(candidate: Candidate, seed: int, out_dir: str = DEFAULT_OUT_DIR, device: str = "cpu") -> dict:
    """`run_probe` + immediate JSON land (standing clause: land to disk the moment a result exists)."""
    os.makedirs(out_dir, exist_ok=True)
    probe = run_probe(candidate, seed, device=device)
    path = _result_path(out_dir, candidate, seed)
    with open(path, "w") as f:
        json.dump(probe, f, indent=2)
    return probe


# ---------------------------------------------------------------------------
# Candidate 4 — fixed-width / vary-depth probe. Separate driver from `run_probe`: same net builders and
# training, but the depth ladder runs at a FIXED small width and the bar semantics differ (the STALL
# bar is depth-1-narrow, and the wide-shallow net is the positive CONTROL, not the thing that stalls).
# ---------------------------------------------------------------------------


def _ratio_or_none(m: float | None) -> float | None:
    """`m / FLOOR`, or `None` when the region MSE is `None` (an empty region mask)."""
    return None if m is None else m / FLOOR


def _train_region_probe(
    x_tr_n: np.ndarray,
    y_tr_n: np.ndarray,
    x_val_n: np.ndarray,
    y_val_n: np.ndarray,
    norm: dict,
    x_val: np.ndarray,
    y_val: np.ndarray,
    region_val: np.ndarray,
    net: nn.Module,
    device: str,
) -> dict:
    """Train `net` to convergence, then read held-out per-region MSE in ORIGINAL y-units (shared helper)."""
    result = train_mse(net, x_tr_n, y_tr_n, x_val_n, y_val_n, device=device)
    m_hard = _region_mse_orig(net, norm, x_val, y_val, region_val, REGION_HARD, device=device)
    m_easy = _region_mse_orig(net, norm, x_val, y_val, region_val, REGION_EASY, device=device)
    return {"params": count_params(net), "m_hard": m_hard, "m_easy": m_easy, "convergence": result.summary(), "trustworthy": bool(result.trustworthy)}


def _fixedwidth_data(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """`make_hetero` train + independent held-out draw, standardized on TRAIN stats.

    Returns the standardized train/val arrays plus the RAW `(x_val, y_val, region_val)` (region MSE is
    scored in original y-units) and the `norm` stats dict.
    """
    x_tr, y_tr, _region_tr = make_fixedwidth_hetero(N_TRAIN, seed)
    x_val, y_val, region_val = make_fixedwidth_hetero(N_VAL, seed + 10_000)  # independent held-out draw (different RNG stream)
    norm = _fit_norm(x_tr, y_tr)
    x_tr_n = _standardize_x(x_tr, norm)
    y_tr_n = (y_tr - norm["my"]) / norm["sy"]
    x_val_n = _standardize_x(x_val, norm)
    y_val_n = (y_val - norm["my"]) / norm["sy"]
    return x_tr_n, y_tr_n, x_val_n, y_val_n, x_val, y_val, region_val, norm


def run_fixedwidth_probe(
    seed: int,
    fixed_width: int = FIXEDWIDTH_W,
    control_width: int = FIXEDWIDTH_CONTROL_W,
    depths: tuple[int, ...] = FIXEDWIDTH_DEPTHS,
    device: str = "cpu",
) -> dict:
    """One full candidate-4 probe: depth ladder at FIXED `fixed_width` + a wide-shallow CONTROL at `control_width`.

    Trains `build_narrow(d, fixed_width)` for each `d in depths` and one `build_wide_shallow(control_width)`,
    each convergence-gated, and checks the swapped-semantics bars (`_check_fixedwidth_bars`). Every net's
    full convergence summary is retained so a non-`trustworthy` read is visible, not silently used.
    """
    in_dim = 1  # make_hetero is 1-D
    x_tr_n, y_tr_n, x_val_n, y_val_n, x_val, y_val, region_val, norm = _fixedwidth_data(seed)

    narrow_results: dict[int, dict] = {}
    for d in depths:
        torch.manual_seed(1000 * seed + d)
        net = build_narrow(d, fixed_width, in_dim)
        row = _train_region_probe(x_tr_n, y_tr_n, x_val_n, y_val_n, norm, x_val, y_val, region_val, net, device)
        row["width"] = fixed_width
        row["depth"] = d
        narrow_results[d] = row

    torch.manual_seed(1000 * seed + 777)
    control_net = build_wide_shallow(control_width, in_dim)
    control_out = _train_region_probe(x_tr_n, y_tr_n, x_val_n, y_val_n, norm, x_val, y_val, region_val, control_net, device)
    control_out["width"] = control_width

    bars = _check_fixedwidth_bars(narrow_results, control_out, depths)
    trustworthy = control_out["trustworthy"] and all(narrow_results[d]["trustworthy"] for d in depths)
    return {
        "candidate": Candidate.FIXEDWIDTH_HETERO.value,
        "seed": seed,
        "fixed_width": fixed_width,
        "control_width": control_width,
        "noise_sigma": NOISE_SIGMA,
        "floor": FLOOR,
        "n_train": N_TRAIN,
        "n_val": N_VAL,
        "probe_depths": list(depths),
        "narrow": {str(d): v for d, v in narrow_results.items()},
        "wide_control": control_out,
        "bars": bars,
        "all_pass": all(b["pass"] for b in bars.values()),
        "all_trustworthy": trustworthy,
    }


def _check_fixedwidth_bars(narrow_results: dict[int, dict], control_out: dict, depths: tuple[int, ...]) -> dict:
    """Candidate-4 verdicts (swapped stall/control roles vs `_check_bars`) from already-trained probe rows."""
    depth_max = depths[-1]
    m_hard_deep = narrow_results[depth_max]["m_hard"]
    ratio_p1 = m_hard_deep / FLOOR
    p1 = {"depth": depth_max, "width": narrow_results[depth_max]["width"], "m_hard": m_hard_deep, "ratio_to_floor": ratio_p1, "pass": bool(ratio_p1 <= FIT_PASS_MULTIPLE)}

    m_hard_d1 = narrow_results[1]["m_hard"]
    ratio_p2 = m_hard_d1 / FLOOR
    p2 = {"depth": 1, "width": narrow_results[1]["width"], "m_hard": m_hard_d1, "ratio_to_floor": ratio_p2, "pass": bool(ratio_p2 >= HUNGRY_STALL_MULTIPLE)}

    # P3 ladder is depth 1 -> deepest at FIXED width (no wide-shallow term): each deeper net's M_hard must
    # not exceed the previous by more than the slack (i.e. error is non-increasing in depth, within noise).
    ladder = [narrow_results[d]["m_hard"] for d in depths]
    monotonic = all(ladder[i + 1] <= ladder[i] * P3_MONOTONIC_SLACK for i in range(len(ladder) - 1))
    p3 = {"ladder_m_hard": {str(d): narrow_results[d]["m_hard"] for d in depths}, "slack_multiple": P3_MONOTONIC_SLACK, "pass": bool(monotonic)}

    m_easy_d1 = narrow_results[1]["m_easy"]
    ratio_p4 = m_easy_d1 / FLOOR
    p4 = {"depth": 1, "width": narrow_results[1]["width"], "m_easy": m_easy_d1, "ratio_to_floor": ratio_p4, "pass": bool(ratio_p4 <= EASY_FLAT_MULTIPLE)}

    m_hard_ctrl = control_out["m_hard"]
    ratio_ctrl = m_hard_ctrl / FLOOR
    control = {"width": control_out["width"], "m_hard": m_hard_ctrl, "ratio_to_floor": ratio_ctrl, "pass": bool(ratio_ctrl <= FIT_PASS_MULTIPLE)}

    return {"p1_learnable_deep": p1, "p2_depth_hungry": p2, "p3_graded": p3, "p4_easy_region_flat": p4, "control_wide_shallow_fits": control}


def pilot_fixedwidth(seed: int, widths: tuple[int, ...], max_depth: int = FIXEDWIDTH_DEPTHS[-1], control_width: int = FIXEDWIDTH_CONTROL_W, device: str = "cpu") -> dict:
    """CHEAP width-locator for candidate 4 (depth-1 + deepest only, control once).

    Per `w`, trains ONLY depth-1 and depth-`max_depth` narrow nets; trains the wide-shallow control
    ONCE. Lets the orchestrator lock `FIXEDWIDTH_W` to the smallest `w` where depth-1 STALLS
    (>= HUNGRY_STALL_MULTIPLE*FLOOR) AND depth-`max_depth` FITS (<= FIT_PASS_MULTIPLE*FLOOR) before
    paying for the full graded ladder.
    """
    in_dim = 1
    x_tr_n, y_tr_n, x_val_n, y_val_n, x_val, y_val, region_val, norm = _fixedwidth_data(seed)
    rows: list[dict] = []
    for w in widths:
        row: dict = {"width": w}
        for d in (1, max_depth):
            torch.manual_seed(1000 * seed + d)
            net = build_narrow(d, w, in_dim)
            probe = _train_region_probe(x_tr_n, y_tr_n, x_val_n, y_val_n, norm, x_val, y_val, region_val, net, device)
            row[f"depth{d}"] = {
                "m_hard": probe["m_hard"], "ratio_hard": _ratio_or_none(probe["m_hard"]),
                "m_easy": probe["m_easy"], "ratio_easy": _ratio_or_none(probe["m_easy"]),
                "trustworthy": probe["trustworthy"],
            }
        rows.append(row)
    torch.manual_seed(1000 * seed + 777)
    cnet = build_wide_shallow(control_width, in_dim)
    cprobe = _train_region_probe(x_tr_n, y_tr_n, x_val_n, y_val_n, norm, x_val, y_val, region_val, cnet, device)
    control = {"width": control_width, "m_hard": cprobe["m_hard"], "ratio_hard": _ratio_or_none(cprobe["m_hard"]), "trustworthy": cprobe["trustworthy"]}
    return {
        "seed": seed,
        "max_depth": max_depth,
        "floor": FLOOR,
        "fit_pass_multiple": FIT_PASS_MULTIPLE,
        "hungry_stall_multiple": HUNGRY_STALL_MULTIPLE,
        "rows": rows,
        "control": control,
    }


def run_and_save_fixedwidth_probe(
    seed: int,
    fixed_width: int = FIXEDWIDTH_W,
    control_width: int = FIXEDWIDTH_CONTROL_W,
    depths: tuple[int, ...] = FIXEDWIDTH_DEPTHS,
    out_dir: str = DEFAULT_OUT_DIR,
    device: str = "cpu",
) -> dict:
    """`run_fixedwidth_probe` + immediate JSON land to `depth_toy_probe_fixedwidth_seed{seed}.json` (`_result_path`)."""
    os.makedirs(out_dir, exist_ok=True)
    probe = run_fixedwidth_probe(seed, fixed_width=fixed_width, control_width=control_width, depths=depths, device=device)
    path = _result_path(out_dir, Candidate.FIXEDWIDTH_HETERO, seed)
    with open(path, "w") as f:
        json.dump(probe, f, indent=2)
    return probe


def run_and_save_pilot(
    seed: int,
    widths: tuple[int, ...],
    max_depth: int = FIXEDWIDTH_DEPTHS[-1],
    control_width: int = FIXEDWIDTH_CONTROL_W,
    out_dir: str = DEFAULT_OUT_DIR,
    device: str = "cpu",
) -> dict:
    """`pilot_fixedwidth` + JSON land to `depth_toy_pilot_fixedwidth_seed{seed}.json`."""
    os.makedirs(out_dir, exist_ok=True)
    pilot = pilot_fixedwidth(seed, widths, max_depth=max_depth, control_width=control_width, device=device)
    path = os.path.join(out_dir, f"depth_toy_pilot_fixedwidth_seed{seed}.json")
    with open(path, "w") as f:
        json.dump(pilot, f, indent=2)
    return pilot


# ---------------------------------------------------------------------------
# Selftest — no training. Shape/finite + seam-continuity checks for all 3 generators.
# ---------------------------------------------------------------------------


def _check_shape_finite(name: str, x: np.ndarray, y: np.ndarray, region: np.ndarray, n: int) -> bool:
    ok = x.shape[0] == n and y.shape == (n,) and region.shape == (n,)
    ok = ok and bool(np.isfinite(x).all()) and bool(np.isfinite(y).all())
    print(f"[depth_toy selftest] ({name}) shape/finite: x.shape={x.shape} y.shape={y.shape} region.shape={region.shape}  {'PASS' if ok else 'FAIL'}")
    return ok


def _check_gentle_continuity() -> bool:
    eps = 1e-6
    left = _gentle_signal(np.array([-eps]))[0]
    right = _gentle_signal(np.array([eps]))[0]
    err = abs(left - right)
    ok = err < _CONTINUITY_TOL
    print(f"[depth_toy selftest] (gentle_composition) seam continuity at x=0: left={left:.6f} right={right:.6f} err={err:.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


def _check_spline_continuity() -> bool:
    eps = 1e-6
    left = _spline_signal(np.array([-eps]))[0]
    right = _spline_signal(np.array([eps]))[0]
    err = abs(left - right)
    ok = err < _CONTINUITY_TOL
    print(f"[depth_toy selftest] (hierarchical_spline) seam continuity at x=0: left={left:.6f} right={right:.6f} err={err:.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


def _check_mult_continuity() -> bool:
    x2_probe = np.array([-1.7, 0.3, 2.1])
    max_err = 0.0
    for edge in (-MULT_BAND, MULT_BAND):
        eps = 1e-6
        left = _mult_signal(np.full(3, edge - eps), x2_probe)
        right = _mult_signal(np.full(3, edge + eps), x2_probe)
        max_err = max(max_err, float(np.abs(left - right).max()))
    ok = max_err < _CONTINUITY_TOL
    print(f"[depth_toy selftest] (multiplicative_2d) blend-edge continuity at x1=+-{MULT_BAND}: max_err={max_err:.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


def _check_region_labels(name: str, x: np.ndarray, region: np.ndarray) -> bool:
    """Every candidate's region label must match its own boundary rule (not just be in-range)."""
    if name == Candidate.MULTIPLICATIVE_2D.value:
        x1 = x[:, 0]
        expected = np.select([x1 <= -MULT_BAND, x1 >= MULT_BAND], [REGION_EASY, REGION_HARD], default=REGION_TRANSITION)
    else:
        expected = (x >= 0.0).astype(int)
    ok = bool(np.array_equal(region, expected))
    print(f"[depth_toy selftest] ({name}) region labels match boundary rule  {'PASS' if ok else 'FAIL'}")
    return ok


def run_selftest() -> bool:
    """Structural checks only (no training): shape/finite, region-label correctness, seam continuity."""
    ok = True

    x, y, region = make_gentle_composition(500, seed=0)
    ok = _check_shape_finite("gentle_composition", x, y, region, 500) and ok
    ok = _check_region_labels(Candidate.GENTLE_COMPOSITION.value, x, region) and ok
    ok = _check_gentle_continuity() and ok

    x, y, region = make_hierarchical_spline(500, seed=0)
    ok = _check_shape_finite("hierarchical_spline", x, y, region, 500) and ok
    ok = _check_region_labels(Candidate.HIERARCHICAL_SPLINE.value, x, region) and ok
    ok = _check_spline_continuity() and ok

    x, y, region = make_multiplicative_2d(500, seed=0)
    ok_shape = x.shape == (500, 2) and y.shape == (500,) and region.shape == (500,)
    ok_finite = bool(np.isfinite(x).all()) and bool(np.isfinite(y).all())
    verdict = "PASS" if (ok_shape and ok_finite) else "FAIL"
    print(f"[depth_toy selftest] (multiplicative_2d) shape/finite: x.shape={x.shape} y.shape={y.shape} region.shape={region.shape}  {verdict}")
    ok = ok_shape and ok_finite and ok
    ok = _check_region_labels(Candidate.MULTIPLICATIVE_2D.value, x, region) and ok
    ok = _check_mult_continuity() and ok

    # candidate 4 (fixedwidth) — reuses make_hetero: (i) FLOOR consistency (its noise std must equal
    # NOISE_SIGMA or the shared FLOOR is wrong for the bars); (ii) shape/finite + region labels ((x>=0)
    # rule); (iii) seam continuity read from a NOISE-FREE draw (sigma=0), reusing make_hetero directly
    # rather than re-deriving its formula (make_hetero is the source of truth for the target).
    ok_floor = HETERO_NOISE_SIGMA == NOISE_SIGMA
    print(f"[depth_toy selftest] (fixedwidth) FLOOR consistency: HETERO_NOISE_SIGMA={HETERO_NOISE_SIGMA} == NOISE_SIGMA={NOISE_SIGMA}  {'PASS' if ok_floor else 'FAIL'}")
    ok = ok_floor and ok
    x, y, region = make_fixedwidth_hetero(500, seed=0)
    ok = _check_shape_finite("fixedwidth", x, y, region, 500) and ok
    ok = _check_region_labels("fixedwidth", x, region) and ok
    x_free, y_free, _region_free = make_hetero(4000, 0, sigma=0.0)
    seam = np.abs(x_free) < _SEAM_WINDOW
    max_seam_abs = float(np.abs(y_free[seam]).max()) if seam.any() else 0.0
    ok_seam = max_seam_abs < _SEAM_SIGNAL_TOL  # both branches -> 0 at x=0; near the seam the noise-free signal stays small
    print(f"[depth_toy selftest] (fixedwidth) seam continuity near x=0 (noise-free max|y| within |x|<{_SEAM_WINDOW}): {max_seam_abs:.3e}  {'PASS' if ok_seam else 'FAIL'}")
    ok = ok_seam and ok

    # net builders / param matching: a quick known-answer check (in_dim=1, depth=4, width=8 -> 241 params;
    # the matching width search should land on an EXACT match, 0 relative error).
    depth4 = build_narrow(4, NARROW_WIDTH, in_dim=1)
    p4 = count_params(depth4)
    ok_p4 = p4 == _SELFTEST_DEPTH4_PARAMS
    print(f"[depth_toy selftest] build_narrow(depth=4, width=8, in_dim=1) params={p4} (expect {_SELFTEST_DEPTH4_PARAMS})  {'PASS' if ok_p4 else 'FAIL'}")
    ok = ok_p4 and ok

    w, diff = match_wide_width(p4, in_dim=1)
    ok_match = w == _SELFTEST_MATCH_WIDTH and diff < _SELFTEST_MATCH_DIFF_TOL
    print(f"[depth_toy selftest] match_wide_width({p4}, in_dim=1) -> width={w} diff={diff:.2e} (expect width={_SELFTEST_MATCH_WIDTH}, diff=0)  {'PASS' if ok_match else 'FAIL'}")
    ok = ok_match and ok

    print(f"[depth_toy selftest] {'PASS' if ok else 'FAIL'}")
    return ok


def _print_bars(probe: dict, path: str) -> None:
    """Shared post-probe stdout: destination path + per-bar pass flags + the two roll-up flags."""
    print(f"[depth_toy] wrote {path}")
    for bar_name, bar in probe["bars"].items():
        ratio_key = "ratio_to_floor" if "ratio_to_floor" in bar else None
        ratio = f" ratio={bar[ratio_key]:.3f}" if ratio_key and bar[ratio_key] is not None else ""
        print(f"  {bar_name}: pass={bar['pass']}{ratio}")
    print(f"  all_pass={probe['all_pass']} all_trustworthy={probe['all_trustworthy']}")


def main() -> None:
    """Parses args and runs `--selftest`, a candidate-1-3 probe, or the candidate-4 pilot/probe; else help."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="No-training structural checks (shape/finite/continuity/region-labels).")
    parser.add_argument("--candidate", type=str, choices=[c.value for c in Candidate], help="Which candidate to probe (real training).")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the probe run (default 0).")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR, help="Directory for probe JSON output.")
    parser.add_argument("--pilot-fixedwidth", type=str, default=None, help="Comma-separated widths (e.g. '3,4,5,6'): cheap candidate-4 width-locator -> pilot JSON.")
    parser.add_argument("--fixed-width", type=int, default=FIXEDWIDTH_W, help=f"Candidate-4 fixed small width for the depth ladder (default {FIXEDWIDTH_W}; lock after the pilot).")
    parser.add_argument("--control-width", type=int, default=FIXEDWIDTH_CONTROL_W, help=f"Candidate-4 wide-shallow CONTROL width (default {FIXEDWIDTH_CONTROL_W}).")
    parser.add_argument("--max-depth", type=int, default=FIXEDWIDTH_DEPTHS[-1], help=f"Candidate-4 deepest probe depth (default {FIXEDWIDTH_DEPTHS[-1]}; ladder is 1..max-depth).")
    args = parser.parse_args()

    if args.selftest:
        sys.exit(0 if run_selftest() else 1)

    device = os.environ.get("AUTOML_DEVICE", "cpu")

    if args.pilot_fixedwidth:
        widths = tuple(int(w) for w in args.pilot_fixedwidth.split(","))
        pilot = run_and_save_pilot(args.seed, widths, max_depth=args.max_depth, control_width=args.control_width, out_dir=args.out_dir, device=device)
        ctrl_r = pilot["control"]["ratio_hard"]
        print(f"[depth_toy] pilot seed={args.seed} floor={FLOOR:.5f} fit<={FIT_PASS_MULTIPLE}x stall>={HUNGRY_STALL_MULTIPLE}x ctrl(w={args.control_width})={ctrl_r}")
        for row in pilot["rows"]:
            d1, dmax = row["depth1"], row[f"depth{args.max_depth}"]
            print(f"  w={row['width']:>3}: d1 ratio_hard={d1['ratio_hard']}  d{args.max_depth} ratio_hard={dmax['ratio_hard']}  (d1 ratio_easy={d1['ratio_easy']})")
        sys.exit(0)

    if args.candidate:
        candidate = Candidate(args.candidate)
        if candidate is Candidate.FIXEDWIDTH_HETERO:
            depths = tuple(range(1, args.max_depth + 1))
            probe = run_and_save_fixedwidth_probe(
                args.seed, fixed_width=args.fixed_width, control_width=args.control_width, depths=depths, out_dir=args.out_dir, device=device
            )
        else:
            probe = run_and_save_probe(candidate, args.seed, out_dir=args.out_dir, device=device)
        _print_bars(probe, _result_path(args.out_dir, candidate, args.seed))
        sys.exit(0)

    parser.print_help()


if __name__ == "__main__":
    main()
