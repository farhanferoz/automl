"""W1 — per-input WIDTH dial on the ramped-sinc positive control.

(docs/plans/width_dial_2026-07-11/EXECUTION_PLAN.md §2b/2c/2d/2e; pre-registration in
capacity_ladder_results/W1/PREREGISTRATION.md)

Builds the nested-width network (`nested_width_net.NestedWidthNet`, single hidden layer, WIDTH is
the only capacity axis), scores it at every width on held-out data, distills a per-input width
SELECTOR from the frozen net, and checks three pre-registered bars: does held-out log-likelihood
climb with width in the oscillating centre and stay flat in the flat tails (CONSTRUCTION); does the
distilled selector read a strictly larger expected width for the centre than the tails
(RECOVERY); does the selector's blended prediction match-or-beat the best single global width and
land within 0.02 nat of the per-input oracle (DEPLOY). Strictly probabilistic throughout: every
score is a per-example Gaussian log-likelihood (`nested_width_net.gaussian_log_likelihood`), no
MSE-only bar, no penalty/lambda, no tuned regularizer.

Two-stage, no leak: phase 1 trains the nested net (`nested_width_net.train_nested_width`) on one
half of the training set (index-parity split); phase 2 FREEZES that net and trains only the
selector on the OTHER half — the "held-out-within-train" split `capacity_ladder_h1.py`'s
two-phase arm (b) uses (`h1.py::run_case`'s `p1_idx`/`p2_idx`), so the selector's distillation
targets are never read off data the net memorized. The selector itself is
`capacity_ladder_k6.py::_RouterMLP` trained with `capacity_ladder_k6.py::_train_router`'s SOFT
objective (`_soft_targets`'s per-tercile EM-stacked responsibilities) — reused verbatim, not
reimplemented, mirroring `capacity_ladder_h1.py`'s own reuse of the same two functions for its
phase-2 gate distillation.

Sinc data (`ramped_sinc`/`make_data`/`region`) is reused verbatim from the Step-0 probe
(`scratchpad/sinc_width_probe.py`) that established the per-input width gradient exists and is
learnable. Because the probe's own fixed-width sweep only converged to a clean per-region MSE
read after STANDARDIZING x and y (raw x spans +-5*pi, saturating tanh at typical init scales),
this driver does the same for training stability — but always converts the net's (mean, log_var)
back to the ORIGINAL y-scale (via the standardization's own affine Jacobian, `logvar_orig =
logvar_norm + 2*log(sy)`, EXACT for a linear rescale, unlike symlog's linearized approximation)
before computing any Gaussian log-likelihood, so every reported LL is directly comparable to the
plan's oracle constant (`-log(0.05) - 0.5*log(2*pi) - 0.5`, the true sinc noise level).

Config (EXECUTION_PLAN.md §2d): W_max=16, 3 seeds (0,1,2), single hidden layer, tanh, 1500 train /
500 test, Adam lr=1e-2, phase-1 epochs tuned to convergence (~2500 real, `SMOKE_EPOCHS` `--smoke`;
see that constant's comment for why it is 600, not the plan's literal smoke figure of 300).
K_LO=1, K_MID=w_max//2 is this driver's construction-bar read point (not pinned by the plan
numerically; a judgment call documented in PREREGISTRATION.md).

Usage:
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/sinc_width_experiment.py --selftest
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/sinc_width_experiment.py --smoke
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/sinc_width_experiment.py --config 0
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/sinc_width_experiment.py   # all seeds
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as nnf

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root, so `import automl_package` works

import _capacity_ladder as cl  # noqa: E402
import capacity_ladder_k6 as ck6  # noqa: E402 — reuse _RouterMLP/_train_router/_soft_targets verbatim
import nested_width_net as nwn  # noqa: E402

from automl_package.utils.pytorch_utils import get_device  # noqa: E402

RESULTS_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "W1")

SEEDS = (0, 1, 2)
W_MAX = 16
N_TRAIN = 1500
N_TEST = 500
LR = 1e-2
N_EPOCHS_PHASE1 = 2500
SMOKE_EPOCHS = 600  # see the `--smoke` branch in main() for why this is not the plan's literal "300 ep"
NOISE_SIGMA = 0.05
ORACLE_LL = -math.log(NOISE_SIGMA) - 0.5 * math.log(2.0 * math.pi) - 0.5
CENTRE_BOUNDARY = 2.0 * math.pi
NOISE_FLOOR_MULTIPLE = 1.5  # "within ~1.5x noise floor" (PREREGISTRATION.md bar i)
NOISE_FLOOR_GAP_NAT = 0.5 * math.log(NOISE_FLOOR_MULTIPLE)  # LL gap equivalent of an Nx-noise-floor MSE read
DEPLOY_ORACLE_TOL_NAT = 0.02
_BOOT_N = 1000
_BOOT_SEED = 0
_SINC_ZERO_TOL = 1e-9  # |x| below this is treated as x==0 for sinc(x)'s removable singularity


@dataclass
class RunConfig:
    """One battery configuration (full or `--smoke`'s tiny stand-in)."""

    w_max: int
    seeds: list[int]
    n_train: int
    n_test: int
    n_epochs_phase1: int
    results_dir: str


# ---------------------------------------------------------------------------
# Sinc data generator + region split (REUSED VERBATIM from scratchpad/sinc_width_probe.py).
# ---------------------------------------------------------------------------


def ramped_sinc(x: np.ndarray) -> np.ndarray:
    """`sinc(x) + 0.04*x` — the ramped-sinc target function, `x` in `[-5*pi, 5*pi]`."""
    s = np.where(np.abs(x) < _SINC_ZERO_TOL, 1.0, np.sin(x) / np.where(x == 0, 1.0, x))
    return s + 0.04 * x


def make_data(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """`(x, y)`, `x ~ Uniform(-5*pi, 5*pi)`, `y = ramped_sinc(x) + Normal(0, 0.05)`."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-5 * np.pi, 5 * np.pi, n)
    y = ramped_sinc(x) + rng.normal(0.0, 0.05, n)
    return x.astype(np.float32), y.astype(np.float32)


def region(x: np.ndarray) -> np.ndarray:
    """`1` = centre/hard (oscillating, `|x| <= 2*pi`), `0` = tail/easy (near-flat ramp)."""
    return (np.abs(x) <= 2 * np.pi).astype(int)


# ---------------------------------------------------------------------------
# Phase 1: fit the nested-width net (standardized x/y for training stability).
# ---------------------------------------------------------------------------


def _fit_phase1(x_tr: np.ndarray, y_tr: np.ndarray, w_max: int, seed: int, n_epochs: int, lr: float, device: str) -> tuple[nwn.NestedWidthNet, dict]:
    """Trains one `NestedWidthNet` with the NESTED-width schedule, standardizing x/y first.

    Returns the frozen (eval-mode) net plus the standardization constants needed to convert its
    (mean, log_var) back to the ORIGINAL y-scale (`_score_all_widths`, `_score_fixed_width`).
    """
    mx, sx = float(x_tr.mean()), float(x_tr.std())
    my, sy = float(y_tr.mean()), float(y_tr.std())
    norm = {"mx": mx, "sx": sx, "my": my, "sy": sy}

    x_n = (x_tr - mx) / sx
    y_n = (y_tr - my) / sy
    x_t = torch.as_tensor(x_n, dtype=torch.float32).reshape(-1, 1)
    y_t = torch.as_tensor(y_n, dtype=torch.float32)

    torch.manual_seed(seed)
    net = nwn.NestedWidthNet(w_max=w_max)
    nwn.train_nested_width(net, x_t, y_t, n_epochs=n_epochs, lr=lr, seed=seed, device=device)
    return net, norm


def _score_all_widths(net: nwn.NestedWidthNet, norm: dict, x: np.ndarray, y: np.ndarray, device: str) -> np.ndarray:
    """`(N, w_max)` held-out Gaussian LL table, ORIGINAL y-scale, at every width in one pass.

    The affine un-standardization is EXACT (not an approximation like symlog's linearized
    Jacobian, `utils/transforms.py`'s docstring): `Y = sy*Y_n + my` with `Y_n ~ N(mean_n,
    exp(logvar_n))` implies `Y ~ N(sy*mean_n + my, sy^2 * exp(logvar_n))`, i.e. `logvar_orig =
    logvar_n + 2*log(sy)`.
    """
    x_n = (x - norm["mx"]) / norm["sx"]
    x_t = torch.as_tensor(x_n, dtype=torch.float32, device=device).reshape(-1, 1)
    net.eval()
    with torch.no_grad():
        mean_n, logvar_n = net.all_widths_forward(x_t)  # (N, w_max), normalized-y space
    sy, my = norm["sy"], norm["my"]
    mean_orig = mean_n * sy + my
    logvar_orig = logvar_n + 2.0 * math.log(sy)
    y_t = torch.as_tensor(y, dtype=torch.float32, device=device).reshape(-1, 1)
    ll = nwn.gaussian_log_likelihood(mean_orig, logvar_orig, y_t)  # broadcasts y_t over the w_max columns
    return ll.cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# 2b — CONSTRUCTION bar: per-region read of the fixed-width sweep (mirrors
# `capacity_ladder_t1._construction_bar`/`_region_gain`, over WIDTH instead of DEPTH).
# ---------------------------------------------------------------------------


def _plain_boot_se(vec: np.ndarray, n_boot: int = _BOOT_N, seed: int = _BOOT_SEED) -> float:
    """Plain i.i.d. bootstrap SE of a 1-D vector's mean (`_capacity_ladder._bootstrap_col_means`, F2/T1 convention)."""
    rng = np.random.default_rng(seed)
    boot = cl._bootstrap_col_means(vec.reshape(-1, 1), n_boot, None, rng)
    return float(boot[:, 0].std(ddof=1))


def _two_sample_boot_se(a: np.ndarray, b: np.ndarray, n_boot: int = _BOOT_N, seed: int = _BOOT_SEED) -> float:
    """Bootstrap SE of `mean(a) - mean(b)` for two INDEPENDENT (unpaired) samples."""
    rng = np.random.default_rng(seed)
    n_a, n_b = len(a), len(b)
    diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        diffs[i] = a[rng.integers(0, n_a, n_a)].mean() - b[rng.integers(0, n_b, n_b)].mean()
    return float(diffs.std(ddof=1))


def _region_gain(fixed_width_ll: dict[int, np.ndarray], mask: np.ndarray, k_lo: int, k_hi: int) -> dict:
    """One region's held-out-LL gain from width `k_lo` to `k_hi` (paired plain bootstrap SE, F2/T1 convention)."""
    delta = fixed_width_ll[k_hi][mask] - fixed_width_ll[k_lo][mask]
    mean_delta = float(delta.mean())
    se = _plain_boot_se(delta)
    return {"mean_delta": mean_delta, "se": se, "beats_2se": bool(mean_delta > 2.0 * se)}


def _construction_bar(fixed_width_ll: dict[int, np.ndarray], region_labels: np.ndarray, k_lo: int, k_mid: int, w_max: int) -> dict:
    """CONSTRUCTION bar (i): centre climbs `k_lo -> k_mid` (>2*SE) and nears the oracle; tail flat.

    Centre (region==1, `|x|<=2*pi`): held-out LL must improve `k_lo -> k_mid` by > 2*SE AND the LL
    at `w_max` must land within `NOISE_FLOOR_GAP_NAT` of `ORACLE_LL` (the "within ~1.5x noise
    floor" MSE-multiple read translated to nats: an effective variance `N` times the true noise
    floor gives an LL gap of exactly `0.5*log(N)` from the oracle -- see module docstring).
    Tail (region==0, `|x|>2*pi`): comparatively flat -- no width past `k_lo` may beat it by > 2*SE.
    """
    mask_tail = region_labels == 0
    mask_centre = region_labels == 1

    centre_gain = _region_gain(fixed_width_ll, mask_centre, k_lo, k_mid)
    centre_climbs = centre_gain["beats_2se"]
    centre_best_ll = float(fixed_width_ll[w_max][mask_centre].mean())
    centre_gap_to_oracle = ORACLE_LL - centre_best_ll
    centre_near_floor = bool(centre_gap_to_oracle <= NOISE_FLOOR_GAP_NAT)
    centre_pass = bool(centre_climbs and centre_near_floor)

    tail_gains = {f"k{k_lo}_to_k{k}": _region_gain(fixed_width_ll, mask_tail, k_lo, k) for k in range(1, w_max + 1) if k != k_lo}
    tail_flat = not any(g["beats_2se"] for g in tail_gains.values())

    return {
        "n_centre": int(mask_centre.sum()),
        "n_tail": int(mask_tail.sum()),
        "k_lo": k_lo,
        "k_mid": k_mid,
        "centre_gain_klo_to_kmid": centre_gain,
        "centre_climbs": bool(centre_climbs),
        "centre_best_ll_at_wmax": centre_best_ll,
        "centre_gap_to_oracle_nat": centre_gap_to_oracle,
        "centre_near_noise_floor": centre_near_floor,
        "centre_pass": centre_pass,
        "tail_gains": tail_gains,
        "tail_flat": bool(tail_flat),
        "construction_pass": bool(centre_pass and tail_flat),
    }


# ---------------------------------------------------------------------------
# 2c — distilled per-input width selector: `capacity_ladder_k6._RouterMLP` + the SOFT objective,
# reused verbatim, trained on the phase-2 (held-out-within-train) half.
# ---------------------------------------------------------------------------


def _fit_selector(score_p2: np.ndarray, x_p2: np.ndarray, w_max: int, seed: int, device: str) -> torch.nn.Module:
    """Distills the SOFT-target router off the frozen phase-1 net's phase-2-half score table."""
    q_soft = ck6._soft_targets(score_p2, x_p2)  # (n_p2, w_max), c_grid = [1..w_max] ascending
    return ck6._train_router(x_p2, n_cols=w_max, device=device, soft_targets=q_soft, seed=seed, n_epochs=ck6.N_EPOCHS, lr=ck6.LR)


def _selector_eval(router: torch.nn.Module, score: np.ndarray, x: np.ndarray, device: str) -> tuple[float, np.ndarray, np.ndarray]:
    """Selector's blended held-out NLL, its full marginal width distribution, and per-input expected width.

    Blended NLL mirrors `capacity_ladder_h1.py::_blended_nll`: `-mean_i logsumexp_k(log w_k(x_i) +
    log p_k(y_i|x_i))`. Expected width per input is `sum_k k * P(k|x_i)` (a continuous summary of
    the router's OWN full probability vector, not a collapsed argmax — used only for the RECOVERY
    bar's region-separation read; the marginal_p full vector is still reported, G6 convention).
    """
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device).reshape(-1, 1)
    score_t = torch.as_tensor(score, dtype=torch.float32, device=device)
    router.eval()
    with torch.no_grad():
        logits = router(x_t)
        log_w = nnf.log_softmax(logits, dim=1)
        probs = nnf.softmax(logits, dim=1)
        log_blend = torch.logsumexp(log_w + score_t, dim=1)
        nll = float(-log_blend.mean().item())
        marginal_p = probs.mean(dim=0).cpu().numpy().astype(np.float64)
        k_grid = np.arange(1, score.shape[1] + 1, dtype=np.float64)
        expected_width = (probs.cpu().numpy().astype(np.float64) * k_grid[None, :]).sum(axis=1)
    return nll, marginal_p, expected_width


# ---------------------------------------------------------------------------
# Bars (ii) RECOVERY and (iii) DEPLOY.
# ---------------------------------------------------------------------------


def _recovery_bar(expected_width: np.ndarray, region_labels: np.ndarray) -> dict:
    """RECOVERY bar (ii): the selector assigns a strictly larger expected width to centre than tail."""
    centre = expected_width[region_labels == 1]
    tail = expected_width[region_labels == 0]
    mean_centre, mean_tail = float(centre.mean()), float(tail.mean())
    diff = mean_centre - mean_tail
    se = _two_sample_boot_se(centre, tail)
    return {
        "mean_expected_width_centre": mean_centre,
        "mean_expected_width_tail": mean_tail,
        "diff": diff,
        "se": se,
        "separation_beats_2se": bool(diff > 2.0 * se),
    }


def _deploy_bar(selector_nll: float, per_k_nll: dict[int, float], oracle_nll: float) -> dict:
    """DEPLOY bar (iii): selector matches-or-beats the best global width AND is within 0.02 nat of oracle."""
    best_global_k = min(per_k_nll, key=per_k_nll.get)
    best_global_nll = per_k_nll[best_global_k]
    matches_or_beats_global = bool(selector_nll <= best_global_nll)
    gap_to_oracle = selector_nll - oracle_nll
    within_oracle = bool(gap_to_oracle <= DEPLOY_ORACLE_TOL_NAT)
    return {
        "selector_nll": selector_nll,
        "best_global_k": best_global_k,
        "best_global_nll": best_global_nll,
        "matches_or_beats_global": matches_or_beats_global,
        "oracle_nll": oracle_nll,
        "gap_to_oracle_nat": gap_to_oracle,
        "within_oracle_bar": within_oracle,
        "deploy_pass": bool(matches_or_beats_global and within_oracle),
    }


# ---------------------------------------------------------------------------
# MSE-only bars + selector (width-MSE program, docs/plans/width_mse_2026-07-16/EXECUTION_PLAN.md §5)
# — a second set of readouts for `--loss mse` runs, parallel to the LL-based CONSTRUCTION/RECOVERY/
# DEPLOY bars above (those functions are UNCHANGED). `_recovery_bar` above IS reused verbatim for the
# MSE dial bar below — its "difference between two groups' means, > 2*SE" check is metric-agnostic,
# so no new function is needed for that one (search-before-write: it already does the job).
# ---------------------------------------------------------------------------

FIT_BAR_PASS_MULTIPLE = 1.25  # M_hard(w_max) <= 1.25 * floor_hard -> pass (§5 bar 1)
FIT_BAR_STRONG_MULTIPLE = 1.10  # ... <= 1.10 * floor_hard -> strong pass
CURVE_GATE_HARD_DROP_MULTIPLE = 0.5  # M_hard(k_mid) <= 0.5 * M_hard(1) (§5 bar 2)
CURVE_GATE_HARD_PLATEAU_MULTIPLE = 1.2  # M_hard(w_max) <= 1.2 * min_k M_hard(k)
CURVE_GATE_EASY_FLAT_MULTIPLE = 1.3  # M_easy(k_easy_lo) <= 1.3 * M_easy(w_max)
DELTA_TIE = 0.25  # cheapest-within-tolerance selector-target tolerance (§5 selector targets)
NOISY_EASY_STAY_NARROW_SLACK = 1.0  # width(noisy-easy) <= width(easy) + this -> "stays narrow" (§5 bar 4, WP-3)
# Region labels (make_hetero / make_hetero3): closed set, NAMED so §5.4's noisy-easy read is not a bare magic 2.
REGION_EASY, REGION_HARD, REGION_NOISY_EASY = 0, 1, 2


def _score_all_widths_mse(net: nwn.NestedWidthNet | nwn.IndependentWidthNet, norm: dict, x: np.ndarray, y: np.ndarray, device: str) -> np.ndarray:
    """`(N, w_max)` held-out squared-error table, ORIGINAL y-units, at every width in one pass.

    MSE twin of `_score_all_widths`: reads only the MEAN readout — `logvar_head` is untouched/unused
    under `--loss mse` (the plan's charter: no variance anywhere in this program).
    """
    x_n = (x - norm["mx"]) / norm["sx"]
    x_t = torch.as_tensor(x_n, dtype=torch.float32, device=device).reshape(-1, 1)
    net.eval()
    with torch.no_grad():
        mean_n, _logvar_n = net.all_widths_forward(x_t)  # (N, w_max), normalized-y space
    mean_orig = mean_n * norm["sy"] + norm["my"]
    y_t = torch.as_tensor(y, dtype=torch.float32, device=device).reshape(-1, 1)
    err2 = (y_t - mean_orig) ** 2  # broadcasts y_t over the w_max columns
    return err2.cpu().numpy().astype(np.float64)


def _region_mse(err2_by_width: dict[int, np.ndarray], mask: np.ndarray, k: int) -> float:
    """`M_region(k)` (§5 notation): mean squared error at width `k`, restricted to `mask`."""
    return float(err2_by_width[k][mask].mean())


def _fit_bar_mse(err2_by_width: dict[int, np.ndarray], region_labels: np.ndarray, w_max: int, floor_hard: float) -> dict:
    """§5 bar 1 — FIT: hard-region MSE at `w_max` vs the analytic noise floor `floor_hard = sigma_hard**2`."""
    mask_hard = region_labels == 1
    m_hard_wmax = _region_mse(err2_by_width, mask_hard, w_max)
    ratio = m_hard_wmax / floor_hard
    return {
        "m_hard_wmax": m_hard_wmax,
        "floor_hard": floor_hard,
        "ratio_to_floor": ratio,
        "pass": bool(ratio <= FIT_BAR_PASS_MULTIPLE),
        "strong_pass": bool(ratio <= FIT_BAR_STRONG_MULTIPLE),
    }


def _curve_shape_gate_mse(err2_by_width: dict[int, np.ndarray], region_labels: np.ndarray, w_max: int) -> dict:
    """§5 bar 2 — per-seed CURVE-SHAPE gate, read BEFORE the dial bar. Fail -> quarantine the seed.

    Generalizes the plan's literal w_max=12 read points to any `w_max`, the same convention
    `_construction_bar` already uses for its `k_mid`: `k_mid = max(2, w_max // 2)` (== 6 at
    w_max=12); `k_easy_lo = min(2, w_max)` (== 2 at w_max=12).
    """
    mask_hard = region_labels == 1
    mask_easy = region_labels == 0
    k_mid = max(2, w_max // 2)
    k_easy_lo = min(2, w_max)

    m_hard_1 = _region_mse(err2_by_width, mask_hard, 1)
    m_hard_mid = _region_mse(err2_by_width, mask_hard, k_mid)
    m_hard_wmax = _region_mse(err2_by_width, mask_hard, w_max)
    m_hard_min = min(_region_mse(err2_by_width, mask_hard, k) for k in range(1, w_max + 1))
    hard_drops_to_mid = bool(m_hard_mid <= CURVE_GATE_HARD_DROP_MULTIPLE * m_hard_1)
    hard_plateaus_at_wmax = bool(m_hard_wmax <= CURVE_GATE_HARD_PLATEAU_MULTIPLE * m_hard_min)

    m_easy_lo = _region_mse(err2_by_width, mask_easy, k_easy_lo)
    m_easy_wmax = _region_mse(err2_by_width, mask_easy, w_max)
    easy_flat = bool(m_easy_lo <= CURVE_GATE_EASY_FLAT_MULTIPLE * m_easy_wmax)

    return {
        "k_mid": k_mid,
        "k_easy_lo": k_easy_lo,
        "m_hard_1": m_hard_1,
        "m_hard_mid": m_hard_mid,
        "m_hard_wmax": m_hard_wmax,
        "m_hard_min": m_hard_min,
        "hard_drops_to_mid": hard_drops_to_mid,
        "hard_plateaus_at_wmax": hard_plateaus_at_wmax,
        "m_easy_lo": m_easy_lo,
        "m_easy_wmax": m_easy_wmax,
        "easy_flat": easy_flat,
        "curve_gate_pass": bool(hard_drops_to_mid and hard_plateaus_at_wmax and easy_flat),
    }


def _cheapest_within_tolerance_labels(err2: np.ndarray, delta_tie: float = DELTA_TIE) -> np.ndarray:
    """§5 selector targets, PRIMARY (pre-registered): smallest k with `err2[i,k] <= (1+delta_tie)*min_j err2[i,j]`.

    Returns `(N,)` 0-based column index (`k-1`) per row, for `capacity_ladder_k6._train_router`'s
    `hard_labels`. `argmax` on a boolean row returns the FIRST True — the smallest qualifying k,
    since columns are k=1..w_max ascending.
    """
    min_err2 = err2.min(axis=1, keepdims=True)
    within_tol = err2 <= (1.0 + delta_tie) * min_err2
    return within_tol.argmax(axis=1)


def _soft_targets_mse(err2: np.ndarray) -> np.ndarray:
    """§5 selector targets, SENSITIVITY ARM (report only): `softmax_k(-err2[i,k] / (2*s_B**2))`.

    `s_B**2 = median_i min_k err2[i,k]` — ONE global scale computed from slice B, not tuned.
    """
    s2_b = float(np.median(err2.min(axis=1)))
    logits = -err2 / (2.0 * s2_b)
    logits = logits - logits.max(axis=1, keepdims=True)
    w = np.exp(logits)
    return w / w.sum(axis=1, keepdims=True)


def _fit_selector_mse(
    err2_p2: np.ndarray, x_p2: np.ndarray, w_max: int, seed: int, device: str, *, delta_tie: float = DELTA_TIE, hidden: tuple[int, ...] = ck6.HIDDEN
) -> torch.nn.Module:
    """Distills the PRIMARY hard-label cheapest-within-tolerance router off slice-B's MSE table.

    `hidden` (default `capacity_ladder_k6.HIDDEN`) is width-cert W6's router-capacity sensitivity
    knob — passed straight through to `ck6._train_router`; unchanged callers get the identical
    default router MLP.
    """
    labels = _cheapest_within_tolerance_labels(err2_p2, delta_tie)
    return ck6._train_router(x_p2, n_cols=w_max, device=device, hard_labels=labels, seed=seed, n_epochs=ck6.N_EPOCHS, lr=ck6.LR, hidden=hidden)


def _fit_selector_mse_soft(err2_p2: np.ndarray, x_p2: np.ndarray, w_max: int, seed: int, device: str) -> torch.nn.Module:
    """Sensitivity-arm router (report only): soft targets off the same MSE table."""
    q_soft = _soft_targets_mse(err2_p2)
    return ck6._train_router(x_p2, n_cols=w_max, device=device, soft_targets=q_soft, seed=seed, n_epochs=ck6.N_EPOCHS, lr=ck6.LR)


def _selector_eval_mse(router: torch.nn.Module, err2: np.ndarray, x: np.ndarray, device: str) -> tuple[float, np.ndarray]:
    """Soft-blend MSE + per-input expected width for a router (works for hard- or soft-trained routers).

    MSE twin of `_selector_eval`'s blended-NLL/expected-width shape. The blended MSE here EXECUTES
    ALL widths (probability-weighted) — a labeled secondary, not the deploy claim (see
    `_route_hardpick_mse`/`_deploy_bar_mse` for the hard-pick primary the compute claim may cite).
    """
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device).reshape(-1, 1)
    router.eval()
    with torch.no_grad():
        probs = nnf.softmax(router(x_t), dim=1).cpu().numpy().astype(np.float64)
    blended_mse = float((probs * err2).sum(axis=1).mean())
    k_grid = np.arange(1, err2.shape[1] + 1, dtype=np.float64)
    expected_width = (probs * k_grid[None, :]).sum(axis=1)
    return blended_mse, expected_width


def _route_hardpick_mse(router: torch.nn.Module, err2: np.ndarray, x: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Hard-pick per-input routing (deploy primary): argmax width, only that prefix executed.

    Reuses `capacity_ladder_k6._route` verbatim (argmax over router logits). Returns
    `(err2_hardpick, executed_width)`, each length N; `executed_width` is the routed width VALUE
    (`col_idx + 1`), not a probability-weighted mean — the number the compute claim may cite.
    """
    col_idx = ck6._route(router, x, device)
    rows = np.arange(err2.shape[0])
    return err2[rows, col_idx], (col_idx + 1).astype(np.float64)


def _deploy_bar_mse(err2_hardpick: np.ndarray, executed_width: np.ndarray, err2_by_width: dict[int, np.ndarray]) -> dict:
    """§5 bar 5 — DEPLOY: hard-pick preserves accuracy (paired bootstrap SE) AND saves compute."""
    per_k_mse = {k: float(v.mean()) for k, v in err2_by_width.items()}
    best_fixed_k = min(per_k_mse, key=per_k_mse.get)
    mse_best_fixed = per_k_mse[best_fixed_k]
    mse_hardpick = float(err2_hardpick.mean())
    mean_executed_width = float(executed_width.mean())

    paired_delta = err2_hardpick - err2_by_width[best_fixed_k]
    se_paired = _plain_boot_se(paired_delta)
    accuracy_preserved = bool(mse_hardpick <= mse_best_fixed + 2.0 * se_paired)
    compute_saved = bool(mean_executed_width < best_fixed_k)
    return {
        "mse_hardpick": mse_hardpick,
        "mean_executed_width": mean_executed_width,
        "mse_best_fixed": mse_best_fixed,
        "best_fixed_k": best_fixed_k,
        "se_paired": se_paired,
        "accuracy_preserved": accuracy_preserved,
        "compute_saved": compute_saved,
        "deploy_pass": bool(accuracy_preserved and compute_saved),
    }


def _deploy_bar_mse_valselected(err2_p2_by_width: dict[int, np.ndarray], err2_te_by_width: dict[int, np.ndarray]) -> dict:
    """Width-cert W7 — DEPLOY baseline, VAL-selected (beside, not replacing, `_deploy_bar_mse`).

    `_deploy_bar_mse`'s `best_fixed_k`/`mse_best_fixed` are picked using the TEST set's own MSE
    table — a hindsight choice (verdict-doc §6 caveat: the baseline gets to look at the data it is
    then scored on). This picks `best_fixed_k` using only SLICE-B (`err2_p2_by_width`, the
    router-train half; TEST is never touched for selection), then reports THAT k's TEST MSE — the
    non-hindsight, actually-deployable fixed-width comparator.
    """
    per_k_mse_p2 = {k: float(v.mean()) for k, v in err2_p2_by_width.items()}
    best_fixed_k_valselected = min(per_k_mse_p2, key=per_k_mse_p2.get)
    mse_best_fixed_valselected_test = float(err2_te_by_width[best_fixed_k_valselected].mean())
    return {
        "best_fixed_k_valselected": best_fixed_k_valselected,
        "mse_best_fixed_valselected_test": mse_best_fixed_valselected_test,
    }


def _noisy_easy_bar_mse(expected_width: np.ndarray, region_labels: np.ndarray) -> dict:
    """§5 bar 4 (WP-3 only) — the noisy-easy NEGATIVE control: the dial reads capacity-hunger, not raw error.

    Region labels (`make_hetero3`): `REGION_EASY=0`, `REGION_HARD=1`, `REGION_NOISY_EASY=2`. Two clauses,
    both must hold (`docs/plans/width_mse_2026-07-16/EXECUTION_PLAN.md` §5 bar 4):
      (a) STAY NARROW — `width(noisy-easy) <= width(easy) + NOISY_EASY_STAY_NARROW_SLACK`. Noise is
          common-mode across widths at a fixed input, so no width fits the noisy-easy region down; the
          honest capacity verdict is "stay narrow". A dial keyed on raw error magnitude would over-feed
          it (that is the failure this control catches).
      (b) HARD STILL WINS — `width(hard) - width(noisy-easy) > 2*SE` (two-sample bootstrap SE, same
          estimator as the dial bar): the genuinely capacity-hungry region is still fed more than the
          merely-noisy one.
    """
    easy = expected_width[region_labels == REGION_EASY]
    hard = expected_width[region_labels == REGION_HARD]
    noisy = expected_width[region_labels == REGION_NOISY_EASY]
    w_easy, w_hard, w_noisy = float(easy.mean()), float(hard.mean()), float(noisy.mean())
    stays_narrow = bool(w_noisy <= w_easy + NOISY_EASY_STAY_NARROW_SLACK)
    se_hard_noisy = _two_sample_boot_se(hard, noisy)
    hard_beats_noisy = bool((w_hard - w_noisy) > 2.0 * se_hard_noisy)
    return {
        "mean_width_easy": w_easy,
        "mean_width_hard": w_hard,
        "mean_width_noisy_easy": w_noisy,
        "stays_narrow": stays_narrow,
        "se_hard_vs_noisy": se_hard_noisy,
        "hard_beats_noisy_2se": hard_beats_noisy,
        "noisy_easy_pass": bool(stays_narrow and hard_beats_noisy),
    }


def _jsonable(obj: object) -> object:
    """Recursively converts numpy/torch scalars and arrays to plain Python/JSON types (F2/K4 convention)."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


# ---------------------------------------------------------------------------
# One (W1, seed) unit: phase 1 + construction bar + phase 2 + recovery/deploy bars.
# ---------------------------------------------------------------------------


def run_case(seed: int, cfg: RunConfig, device: str) -> dict:
    """Runs phase 1 (nested net), the construction bar, phase 2 (selector), and the recovery/deploy bars."""
    x_tr, y_tr = make_data(cfg.n_train, seed)
    x_te, y_te = make_data(cfg.n_test, seed + 500)

    # Two-stage, no leak: index-parity split of the TRAIN set (H1's p1/p2 convention). Phase 1
    # trains ONLY on p1; phase 2's targets/selector are built ONLY on p2, data the net never saw.
    p1_idx = np.arange(0, cfg.n_train, 2)
    p2_idx = np.arange(1, cfg.n_train, 2)
    x_p1, y_p1 = x_tr[p1_idx], y_tr[p1_idx]
    x_p2, y_p2 = x_tr[p2_idx], y_tr[p2_idx]

    net, norm = _fit_phase1(x_p1, y_p1, cfg.w_max, seed, cfg.n_epochs_phase1, LR, device)

    # --- 2b: CONSTRUCTION bar, on the untouched TEST set ---
    score_te = _score_all_widths(net, norm, x_te, y_te, device)  # (n_test, w_max)
    fixed_width_ll = {k: score_te[:, k - 1] for k in range(1, cfg.w_max + 1)}
    region_te = region(x_te)
    k_mid = max(2, cfg.w_max // 2)
    construction = _construction_bar(fixed_width_ll, region_te, k_lo=1, k_mid=k_mid, w_max=cfg.w_max)

    # --- 2c: phase-2 selector, distilled on the p2 held-out-within-train half ---
    score_p2 = _score_all_widths(net, norm, x_p2, y_p2, device)
    router = _fit_selector(score_p2, x_p2, cfg.w_max, seed, device)

    # --- evaluate the selector on the TEST set: bars (ii)/(iii) ---
    selector_nll, marginal_p, expected_width_te = _selector_eval(router, score_te, x_te, device)
    recovery = _recovery_bar(expected_width_te, region_te)
    per_k_nll = {k: float(-fixed_width_ll[k].mean()) for k in range(1, cfg.w_max + 1)}
    oracle_nll = float(-score_te.max(axis=1).mean())
    deploy = _deploy_bar(selector_nll, per_k_nll, oracle_nll)

    return {
        "seed": seed,
        "n_test": int(score_te.shape[0]),
        "construction": construction,
        "recovery": recovery,
        "deploy": deploy,
        "marginal_p": marginal_p.tolist(),
        "per_k_nll": per_k_nll,
    }


# ---------------------------------------------------------------------------
# Selftest -- 2a/2c WIRING check, no phase-1 training of the big net.
# ---------------------------------------------------------------------------


def _synthetic_width_table(w_max: int, seed: int, n_per: int = 1000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Planted known-answer `(N, w_max)` width-LL table: centre climbs to near-oracle, tail prefers LOW width.

    Region assignment uses the real `region()` split on x drawn from the real sinc x-range, so the
    CONSTRUCTION bar's region masks are exercised exactly as the real read uses them; only the
    per-width LL VALUES are planted (no net, no training). Tail is given a MILD, genuinely
    OPPOSING width preference (declining, not flat) rather than true-flat noise -- a true-flat tail
    trivially makes the single best GLOBAL width (== the centre's optimum) already ideal for both
    regions at once, which cannot exercise the RECOVERY/DEPLOY bars' per-input trade-off (mirrors
    `capacity_ladder_t1.py`'s selftest part (b): "a positive control needs GENUINELY CONFLICTING
    regions, not one flat and one dominant"). `_construction_bar`'s tail_flat check only fails on a
    SIGNIFICANT POSITIVE gain past k_lo, so a declining tail still reads "flat" by that bar's own
    definition. Noise is a single value SHARED across all w_max columns of a row (the same held-out
    y residual scored under every width, matching how the real net's per-width table is built --
    NOT independent per-column noise, which would inflate the per-input oracle read unrealistically).
    """
    rng = np.random.default_rng(seed)
    x_tail_pos = rng.uniform(2 * np.pi + 0.5, 5 * np.pi, n_per // 2)
    x_tail = np.concatenate([x_tail_pos, -x_tail_pos])
    x_centre = rng.uniform(-2 * np.pi + 0.5, 2 * np.pi - 0.5, n_per)
    x = np.concatenate([x_tail, x_centre]).astype(np.float64)
    reg = region(x)

    k = np.arange(1, w_max + 1, dtype=np.float64)
    means_tail = 1.0 - 0.15 * (k - 1)  # mildly prefers LOW width (declining, not flat)
    means_centre = ORACLE_LL - 3.0 * np.exp(-(k - 1) / 2.0)  # climbs from ORACLE_LL-3 to within ~0.09 nat of ORACLE_LL

    n = len(x)
    noise_sd = 0.3
    row_noise = rng.normal(0, noise_sd, size=n)  # one shared value per row, applied to every column
    score = np.empty((n, w_max), dtype=np.float64)
    tail_mask = reg == 0
    score[tail_mask] = means_tail[None, :] + row_noise[tail_mask, None]
    score[~tail_mask] = means_centre[None, :] + row_noise[~tail_mask, None]
    return score, x, reg


def run_selftest() -> bool:
    """No-train wiring check.

    (0) the phase-1 scoring pipeline on an UNTRAINED net; (a) the CONSTRUCTION bar on a planted
    centre-climbs/tail-flat table; (b)/(c) selector distillation + the RECOVERY/DEPLOY bars on the
    SAME planted table (a positive control on both sides).
    """
    device = "cpu"
    ok = True

    print("[w1 selftest] (0) nested_width_net integration wiring (random-init net, no phase-1 training)")
    torch.manual_seed(0)
    net = nwn.NestedWidthNet(w_max=6)
    x_fake = np.array([0.0, 1.0, -1.0, 10.0, -6.0], dtype=np.float32)
    y_fake = np.array([0.1, 0.2, -0.1, 0.5, -0.3], dtype=np.float32)
    norm = {"mx": 0.0, "sx": 1.0, "my": 0.0, "sy": 1.0}
    score0 = _score_all_widths(net, norm, x_fake, y_fake, device)
    ok0 = score0.shape == (5, 6) and bool(np.isfinite(score0).all())
    print(f"  score table shape={score0.shape} finite={bool(np.isfinite(score0).all())}  {'PASS' if ok0 else 'FAIL'}")
    ok = ok and ok0

    w_max = 8
    print("[w1 selftest] (a) construction bar wiring: planted centre-climbs-to-oracle / tail-flat table")
    score_syn, x_syn, region_syn = _synthetic_width_table(w_max=w_max, seed=0)
    fixed_width_ll = {k: score_syn[:, k - 1] for k in range(1, w_max + 1)}
    construction = _construction_bar(fixed_width_ll, region_syn, k_lo=1, k_mid=w_max // 2, w_max=w_max)
    ok_a = construction["construction_pass"]
    print(
        f"  centre_climbs={construction['centre_climbs']} centre_near_noise_floor={construction['centre_near_noise_floor']} "
        f"tail_flat={construction['tail_flat']} construction_pass={construction['construction_pass']}  {'PASS' if ok_a else 'FAIL'}"
    )
    ok = ok and ok_a

    print("[w1 selftest] (b)/(c) selector distillation + recovery/deploy bar wiring (same planted table)")
    q_soft = ck6._soft_targets(score_syn, x_syn)
    router = ck6._train_router(x_syn, n_cols=w_max, device=device, soft_targets=q_soft, seed=0, n_epochs=ck6.N_EPOCHS, lr=ck6.LR)
    selector_nll, _marginal_p, expected_width = _selector_eval(router, score_syn, x_syn, device)
    recovery = _recovery_bar(expected_width, region_syn)
    per_k_nll = {k: float(-fixed_width_ll[k].mean()) for k in range(1, w_max + 1)}
    oracle_nll = float(-score_syn.max(axis=1).mean())
    deploy = _deploy_bar(selector_nll, per_k_nll, oracle_nll)
    ok_b = recovery["separation_beats_2se"]
    # Wiring check for DEPLOY: the selector must beat the best single GLOBAL width -- the core
    # per-input-vs-global claim. The literal "within 0.02 nat of oracle" THRESHOLD (real-run bar,
    # `deploy["deploy_pass"]`) is NOT gated here: `_soft_targets`'s tercile-quantile blending has
    # a genuine, non-bug smoothing cost right at a sharp region boundary (reported below, not
    # asserted) -- reused verbatim per the plan, not something this selftest should paper over by
    # constructing an easier synthetic table.
    ok_c = deploy["matches_or_beats_global"]
    print(
        f"  recovery: expected_width centre={recovery['mean_expected_width_centre']:.3f} tail={recovery['mean_expected_width_tail']:.3f} "
        f"diff={recovery['diff']:.3f} se={recovery['se']:.4f}  {'PASS' if ok_b else 'FAIL'}"
    )
    print(
        f"  deploy: selector_nll={deploy['selector_nll']:.4f} best_global_nll={deploy['best_global_nll']:.4f} "
        f"oracle_nll={deploy['oracle_nll']:.4f} gap_to_oracle={deploy['gap_to_oracle_nat']:.4f} "
        f"(matches_or_beats_global gated here; within-0.02-nat-of-oracle is a real-run bar, reported not asserted)  {'PASS' if ok_c else 'FAIL'}"
    )
    ok = ok and ok_b and ok_c

    print(f"[w1 selftest] {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Real reader.
# ---------------------------------------------------------------------------


def main() -> None:
    """Runs the W1 battery (or `--smoke`'s tiny stand-in / `--selftest`'s wiring check)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="No-train wiring check (2a/2c), then exit.")
    parser.add_argument("--smoke", action="store_true", help="Tiny config (w_max=6, SMOKE_EPOCHS phase-1 epochs, 1 seed); no save.")
    parser.add_argument("--config", choices=[str(s) for s in SEEDS], default=None, help="Run only this one seed (sharded parallel launching); default = all seeds.")
    args = parser.parse_args()

    if args.selftest:
        sys.exit(0 if run_selftest() else 1)

    device = str(get_device())

    if args.smoke:
        # 300 epochs (the plan's literal smoke figure) is too little to converge past training
        # noise at w_max=6 -- the centre-region LL-by-width curve is non-monotonic there (measured:
        # worst width is k=2, not k=1). SMOKE_EPOCHS=600 is the smallest epoch count found to
        # reliably reproduce the width gradient (k=1 clearly worst, k=2..6 clearly and consistently
        # better) while still running in ~1s; still far below the real run's 2500.
        cfg = RunConfig(w_max=6, seeds=[0], n_train=N_TRAIN, n_test=N_TEST, n_epochs_phase1=SMOKE_EPOCHS, results_dir=RESULTS_DIR)
        save = False
    else:
        seeds = list(SEEDS) if args.config is None else [int(args.config)]
        cfg = RunConfig(w_max=W_MAX, seeds=seeds, n_train=N_TRAIN, n_test=N_TEST, n_epochs_phase1=N_EPOCHS_PHASE1, results_dir=RESULTS_DIR)
        save = True

    print(f"[w1] device={device} config={cfg}")
    os.makedirs(cfg.results_dir, exist_ok=True)

    per_case = []
    for seed in cfg.seeds:
        print(f"=== W1 seed={seed} ===", flush=True)
        case = run_case(seed, cfg, device)
        per_case.append(case)
        c, r, d = case["construction"], case["recovery"], case["deploy"]
        print(
            f"  construction: centre_climbs={c['centre_climbs']} centre_near_floor={c['centre_near_noise_floor']} "
            f"tail_flat={c['tail_flat']} pass={c['construction_pass']}"
        )
        print(f"  recovery: centre_width={r['mean_expected_width_centre']:.3f} tail_width={r['mean_expected_width_tail']:.3f} sep_pass={r['separation_beats_2se']}")
        print(f"  deploy: selector_nll={d['selector_nll']:.4f} best_global_nll={d['best_global_nll']:.4f} oracle_nll={d['oracle_nll']:.4f} pass={d['deploy_pass']}")

    n_seeds = len(per_case)
    n_construction_pass = sum(1 for c in per_case if c["construction"]["construction_pass"])
    n_recovery_pass = sum(1 for c in per_case if c["recovery"]["separation_beats_2se"])
    n_deploy_pass = sum(1 for c in per_case if c["deploy"]["deploy_pass"])
    bar_i = {"n_pass": n_construction_pass, "n_seeds": n_seeds, "pass": bool(n_seeds) and n_construction_pass >= math.ceil(2 * n_seeds / 3)}
    bar_ii = {"n_pass": n_recovery_pass, "n_seeds": n_seeds, "pass": bool(n_seeds) and n_recovery_pass >= math.ceil(2 * n_seeds / 3)}
    bar_iii = {"n_pass": n_deploy_pass, "n_seeds": n_seeds, "pass": bool(n_seeds) and n_deploy_pass >= math.ceil(2 * n_seeds / 3)}

    if bar_i["pass"] and bar_ii["pass"] and bar_iii["pass"]:
        verdict = "FOUND_LEARNABLE_WIDTH_CONTROL"
    elif not bar_i["pass"]:
        verdict = f"FAIL_i_CONSTRUCTION: {n_construction_pass}/{n_seeds} seeds passed (need >=2/3)."
    elif not bar_ii["pass"]:
        verdict = f"FAIL_ii_RECOVERY: {n_recovery_pass}/{n_seeds} seeds passed (need >=2/3)."
    else:
        verdict = f"FAIL_iii_DEPLOY: {n_deploy_pass}/{n_seeds} seeds passed (need >=2/3)."

    print(f"\nbar (i) CONSTRUCTION: {bar_i}")
    print(f"bar (ii) RECOVERY: {bar_ii}")
    print(f"bar (iii) DEPLOY: {bar_iii}")
    print(f"VERDICT: {verdict}")

    summary = {
        "config": {
            "w_max": cfg.w_max, "seeds": cfg.seeds, "n_train": cfg.n_train, "n_test": cfg.n_test,
            "n_epochs_phase1": cfg.n_epochs_phase1, "lr": LR, "phase2_epochs": ck6.N_EPOCHS, "phase2_lr": ck6.LR,
            "oracle_ll": ORACLE_LL, "noise_sigma": NOISE_SIGMA, "centre_boundary": CENTRE_BOUNDARY,
            "noise_floor_multiple": NOISE_FLOOR_MULTIPLE, "deploy_oracle_tol_nat": DEPLOY_ORACLE_TOL_NAT,
        },
        "per_case": per_case,
        "bar_i_construction": bar_i,
        "bar_ii_recovery": bar_ii,
        "bar_iii_deploy": bar_iii,
        "verdict": verdict,
    }

    if save:
        summary_path = os.path.join(cfg.results_dir, "w1_summary.json")
        with open(summary_path, "w") as f:
            json.dump(_jsonable(summary), f, indent=2)
        print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
