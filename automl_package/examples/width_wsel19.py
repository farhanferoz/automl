"""WSEL-19 D1 -- the router-backend bake-off, 1-D slice (`docs/plans/capacity_programme/width.md` WSEL-19 ~1146-1219, D1 execution spec ~1220-1268).

**Question.** WSEL-7's sign-off left three rulings undecided by fiat: the router must be
input-size-RELATIVE (ruling 2), regularisation is a first-class requirement (ruling 3, concretised
by ruling 6 as an internal validation split + early stopping + mild weight decay, dropout
excluded), and whether ruling 2 is met by a sizing rule or by a tree backend is decided
EMPIRICALLY (ruling 5) -- this driver IS that decision procedure.

**Four backends, one cell each (ratified, WSEL-19 header):**
  1. `frozen_mlp`   -- `DistilledCapacityRouter` exactly as shipped (`routing.py:77-80`), the
     baseline every other arm is measured against.
  2. `rule_mlp`     -- the SAME `_CapacityRouterMLP` architecture, hidden sizes set by an
     input-dimensionality-relative rule this driver defines and records (`_rule_sized_hidden`
     below), trained with an internal validation split, early stopping and mild AdamW weight
     decay, dropout excluded (ruling 6). The training loop is the ONE piece this arm re-implements
     -- `routing.py` is out of this task's write set and its own `_fit_from_targets` has no
     early-stopping/weight-decay hooks to extend (plain full-batch Adam, `routing.py:296-311`).
  3. `xgboost`      -- `xgb.XGBClassifier` on the SAME hard labels the other arms train against,
     native early stopping on the SAME internal split as `rule_mlp` (ratified: "a tree backend
     satisfies [ruling 6] natively -- early stopping + depth limits + shrinkage + subsampling").
  4. `constant`     -- always the single globally-best capacity from the training-side error
     table (`_global_cheapest_within_tolerance` below) -- the control that asks whether per-input
     routing pays for itself at all.

**Two evaluation modes, no extra training (ratified: "blending is TESTED, never hand-waved").**
HARD argmax-routes; BLEND weights every width's prediction by the router's own class
probabilities. Training is IDENTICAL between the two -- `run_cell` fits the requested backend
once and reads off both the argmax and the probability-weighted number from the SAME fit. Ruling
4 pins the readout: routed error on the underlying model's own training metric (squared error on
this MSE-only toy, `width.md` SS3.7), no size penalty, smallness only through the declared
`DEFAULT_TOLERANCE` tie-band -- so BLEND's "quality" is the probability-weighted EXPECTATION of
the same per-width squared-error table HARD reads by argmax, not a likelihood
(`DistilledCapacityRouter.blend_scores`/`blend_nll` port the log-likelihood blend `capacity_
ladder_t2` needs elsewhere in this programme; they do not apply to a plain squared-error metric,
so this driver computes the direct linear-space expectation instead, `_score_blend` below).

**REUSE FIRST (`width.md` SS3.9; verified against source, not recalled).**
  - `automl_package.models.flexnn.routing`: `DistilledCapacityRouter`/`DEFAULT_HIDDEN`/
    `DEFAULT_N_EPOCHS`/`DEFAULT_LR`/`DEFAULT_TOLERANCE` (backend 1, unmodified). `_CapacityRouterMLP`
    and `_cheapest_within_tolerance_labels` are imported too, leading underscore notwithstanding --
    `cascade_width_net.py:76` already imports `_linear_macs` straight from the package's
    `capacity_accounting.py` the same way (package -> examples is the sanctioned dependency
    direction; `routing.py`'s own "copy, not import" choice for these exact two names was about the
    OPPOSITE direction, package code depending on `examples/`, which does not apply here). Backend 2
    reuses the router's own network shape; backend 2/3's shared hard-label rule is the identical
    formula `DistilledCapacityRouter.fit()` uses internally, imported once rather than re-derived
    per arm.
  - `width_wsel6.py`: `_TIER_CONFIG`/`Tier`/`_build_split`/`_get_or_train`/`_load_cached_model`/
    `_sweep_cache_paths`/`W_MAX`/`SEEDS` -- the 12 dedicated per-width `FlexibleWidthNN` sweep nets
    WSEL-6 already trained for tier-1 hetero, seeds 0/1/2, under the WSEL-4-vetted ported-arm
    protocol (Tanh, full-batch, `width_wsel4.py`'s convergence gate) are the SAME per-width models
    this task's toy-design confound C3 requires be trained ONCE per seed and shared by every
    backend -- `_get_or_build_sweep_cache` below loads them straight off
    `capacity_ladder_results/WSEL6/_cache/sweep_tier1_seed{0,1,2}_w{1..12}.{pt,json}` (confirmed on
    disk at authoring time, `objective="mse"`, `trustworthy=True`, `hit_cap=False`) instead of
    retraining; a cell that needs a `(n_train, n_test)` WSEL-6 never ran (only `--selftest`'s tiny
    toy) falls back to training via `w6._get_or_train` into THIS task's own results dir, so the
    fallback path never writes into `WSEL6/`.
  - `automl_package.utils.capacity_accounting.executed_flops` -- the per-width FLOPs table shared by
    every backend's deployed-compute readout (a property of the underlying model at a given width,
    not of whichever backend chose that width).
  - `xgboost` (already a project dependency) -- import verified (`3.3.0`, `XGBClassifier`,
    constructor `early_stopping_rounds` + `.fit(eval_set=...)`) before any code assuming its API was
    written.

**Data design -- three DISJOINT `make_hetero` draws per seed, all at tier-1's canonical
`sigma=0.05` (`width.md` SS3.8's tier-1 row):**
  1. **Training draw** (`p1` of `make_hetero(1500, seed)`) -- WSEL-6's own draw, already baked into
     the cached per-width `state_dict`s; this driver never re-derives it (loaded nets only).
  2. **Selection pool** (`make_hetero(1200, seed + 1000)`) -- a THIRD disjoint draw, needed because
     this task's selection sizes (75, 300, 1200) run past `p2`'s own 50%-of-`n_train` capacity (750
     at tier 1), which is what every other selection-fraction driver in this strand (`width_
     wsel6.py`) tops out at. Drawn ONCE at the largest swept size and shuffled with a seeded
     permutation (`np.random.default_rng(seed)`); every smaller `n_sel` is the shuffled draw's
     PREFIX, so the three sizes are nested (adapting `width_wsel6._selection_subsample`'s nesting
     principle -- SAME idea, different pool-construction shape,
     since the source is a fresh draw, not a fractional carve of `p2`) and a size-vs-size
     comparison reflects added selection data, never independently-resampled noise.
  3. **Report draw** (`make_hetero(500, seed + 500)`) -- the SAME `seed + 500` convention every
     sibling driver in this strand uses for its disjoint held-out test set
     (`width_wsel4.py`/`width_wsel6.py`/`width_wsel7.py`/`width_wsel11.py`, `converged_width_
     experiment.py`, all identical). Scores every backend/mode cell; never touched by training,
     selection-pool labelling, or router fitting.

**Caching (the load-bearing efficiency requirement, mirrors `width_wsel7._get_or_build_cache`).**
`_get_or_build_sweep_cache` builds the 12-net error tables on the selection pool and the report
draw ONCE per seed, caches them to `WSEL19/cache_seed{seed}.json`, and every `(backend, mode,
n_sel)` cell for that seed reads a PREFIX of the cached tables -- no repeat `predict()` sweep.
Router/backend fitting itself is cheap (<=1200 points, a small MLP or a depth-3 tree) and is
NOT cached across `--mode` invocations, mirroring `width_wsel7.py`'s own choice to refit the
(cheap) router fresh per cell while caching only the (expensive) net training.

**Arm 2's sizing rule (a chosen default, ruling 2's deliverable, flagged for review like every
other chosen default in this strand).** `hidden_width(d) = round(32 * sqrt(d))` -- calibrated so
`d=1` recovers the frozen recipe's own hidden width (32) EXACTLY, satisfying ruling 2's
requirement that the frozen `(32, 32)` be the rule's d=1 instance. Depth stays fixed at 2 (the
frozen recipe's own depth): ruling 2 is about the router's WIDTH failing to track input size, not
its depth, and WSEL-7's own depth sweep already found depth within the frozen default's plateau
band (user ruling 1: no dimension's default changes). Square-root (rather than linear) growth is
the reason `rule_mlp` is a genuine SECOND regularisation lever alongside early stopping/weight
decay: linear-in-`d` growth would multiply router parameters by 32x at `d=32` (the toy-design
spec's own starvation cell) and make the "does regularising a bigger router help" question moot by
re-introducing the same over-parameterisation ruling 3 exists to fix. Defined for arbitrary `d`
(`width.md` SS3.10) even though the 1-D slice only ever calls it at `d=1`.

**Non-goals (authoring contract):** no edits to `routing.py`/`width_wsel6.py`/`width_wsel7.py`/any
plan doc; no grid beyond the one real smoke cell (the root runs the grid, backgrounded); no
tolerance sweep (MASTER Decision 18).

**Multi-feature extension (GO rendered, `shared/wsel19-toy-design.md` amended spec).** `--dim
{1,2,8,32} --geometry {axis,oblique}` (required together; both omitted => the 1-D slice above,
byte-identical) route a cell through `width_wsel19_toys` (`wt` below) instead of the 1-D sweep
cache. The per-width error table comes from `w_max` dedicated `FlexibleWidthNN(widths=(k,))` nets
-- the SAME certified PORTED-arm protocol `width_wsel6.py`/`width_wsel4.py` use -- trained fresh
per `(d, geometry, seed[, n_train])` on `wt.make_hetero_multifeature`'s data (`_get_or_build_mf_
models`). No package change was needed for this: `FlexibleWidthNN`/`FlexibleWidthNNModule` already
infer `input_size` from the training data's own column count at `_fit_single` time
(`base_pytorch.py:124-125`, confirmed at authoring time, not assumed) -- `automl_package/models/
flexnn/width/architectures.py`'s separate `in_dim` generalization (already landed) is the record
of the same §2b requirement for the raw research classes and is not what this driver's per-width
sweep calls.

The §5 validity checks run IN ORDER and are recorded in every multi-feature cell JSON's
`validity_checks` block:
  1. **Identity (§5.1):** `wt.identity_holds` at a fixed check size -- asserts, does not gate
     (holds by construction; a `False` here means the construction itself is broken, so it raises
     rather than being recorded as a soft status).
  2. **d=1 calibration cell (§5.2):** `--calibrate` runs the SAME per-width sweep at `d=1,
     geometry=axis` (the canonical anchor -- a flagged choice, see `run_calibration`) across every
     canonical seed, asserts the canonical regime structure (different best widths in the easy vs
     width-hungry region) survives the probability-integral-transform lift, and writes the gating
     artifact (`wsel19_calibration_d1.json`). Every multi-feature cell (`_load_calibration_or_
     refuse`) REFUSES (raises) unless this artifact exists and `passed=True` -- a hard gate, unlike
     checks 1/3/4/5 below, which are recorded, not refused on.
  3. **Fit gate (§5.3/F6):** the cell's best-fixed-width held-out MSE, as a multiple of the noise
     floor (`HETERO_NOISE_SIGMA**2`), must not exceed the calibration artifact's own worst-seed
     ratio (anchor-relative, no new constant) -- `fit_status: ok` or `void_for_fit`. `--n-train-
     fallback` reruns the SAME cell at the §3b pre-authorized `n_train=4000` instead of the
     canonical 1500 (an explicit flag, never automatic).
  4. **Hidden-ness falsifier (§5.4/F4):** the true-error oracle (`(pred_w(x) - h(t))**2`, `h`
     `nested_width_net.make_hetero`'s own noise-free signal, evaluated on the untouched `t` --
     never a noisy single draw) must beat the best FIXED width by >= 10% of the latter's true
     error on the report set -- `routing_status: ok` or `void_for_routing`.
  5. **Regime visibility (§5.5):** the per-width TRUE-error table must prefer different widths in
     the easy (`region=0`) vs width-hungry (`region=1`) report subsets -- `regime_visible`.

Backends/modes/readouts are UNCHANGED from the 1-D slice (`_fit_backend`/`_score_hard`/
`_score_blend`, already `in_dim`-parametrized) -- the multi-feature extension only had to fix each
backend's internal `x.reshape(-1, 1)` (hardcoded to a scalar column) to `routing._as_capacity_
input` (already `in_dim`-generic, imported rather than re-derived) and add the data-provisioning
and §5-check layer above. Backend 2's sizing rule now genuinely varies with `d` (`_rule_sized_
hidden`), recorded in `sizing_rule` as before.

Non-goals (multi-feature authoring contract): no edits to `architectures.py`/`width_wsel19_toys.
py`/any plan doc; no real multi-feature grid beyond one smoke cell (the root runs the grid,
backgrounded); no `--summarize` change (multi-feature cell filenames are disjoint from the 1-D
glob it reads, by construction -- aggregating the multi-feature grid is the root's job).

Driver CLI contract (root-run grid; this file is never run over the full grid by its author):
  `--backend {frozen_mlp,rule_mlp,xgboost,constant} --mode {hard,blend} --n-sel {75,300,1200}
  --seed <int>` runs ONE 1-D cell, writing its per-cell JSON immediately (and, on a cache miss for
  that seed's sweep, the shared per-width error tables).
  `--dim {1,2,8,32} --geometry {axis,oblique}` alongside the four args above runs ONE multi-feature
  cell instead (`run_cell_multifeature`); `--n-train-fallback` optionally applies the §3b F6
  fallback. REFUSES unless `--calibrate` has already been run (see above).
  `--calibrate` runs the d=1 calibration cell, writes the gating artifact, and exits.
  `--summarize` aggregates every 1-D per-cell JSON on disk into `WSEL19/frozen.json`.
  `--selftest` runs a tiny end-to-end check in a temp dir (small `w_max`, tiny selection pool, all
  four backends x both modes, PLUS the multi-feature identity/calibration-gate/cell wiring) and
  asserts the sweep cache is built once and reused -- no real cell is ever run here, and nothing is
  written under this task's real results directory.
  `--tag` appends a suffix to a cell's own JSON filename (never to any shared cache) -- intended
  for a throwaway verification cell that gets deleted afterward without touching any shared
  artifact a later real cell would want to reuse.

Usage:
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel19.py --selftest
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel19.py --backend frozen_mlp --mode hard --n-sel 300 --seed 0
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel19.py --backend xgboost --mode blend --n-sel 75 --seed 1
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/width_wsel19.py --summarize
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel19.py --calibrate
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel19.py \
        --backend rule_mlp --mode hard --n-sel 300 --seed 0 --dim 8 --geometry oblique
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel19.py \
        --backend rule_mlp --mode hard --n-sel 300 --seed 0 --dim 32 --geometry axis --n-train-fallback
"""

from __future__ import annotations

import argparse
import enum
import json
import math
import os
import shutil
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as nnf
import xgboost as xgb

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root

import converged_width_experiment as cwe  # noqa: E402 -- VAL_EVERY, the internal-validation-split convention every driver in this strand shares
import nested_width_net as nwn  # noqa: E402 -- make_hetero, the canonical tier-1 toy
import width_wsel4 as w4  # noqa: E402 -- PORTED_* ported-arm protocol constants, reused verbatim (see module docstring)
import width_wsel6 as w6  # noqa: E402 -- Tier/_TIER_CONFIG/_build_split/_get_or_train/_load_cached_model/_sweep_cache_paths, reused verbatim
import width_wsel19_toys as wt  # noqa: E402 -- multi-feature construction: Geometry, TRAIN_N/REPORT_N, make_*_split, identity_holds

from automl_package.enums import ActivationFunction, CapacitySelection, TaskType  # noqa: E402
from automl_package.models.flexnn.routing import (  # noqa: E402
    DEFAULT_HIDDEN,
    DEFAULT_LR,
    DEFAULT_N_EPOCHS,
    DEFAULT_TOLERANCE,
    DistilledCapacityRouter,
    _as_capacity_input,
    _CapacityRouterMLP,
    _cheapest_within_tolerance_labels,
)
from automl_package.models.flexnn.width.model import FlexibleWidthNN  # noqa: E402
from automl_package.utils.capacity_accounting import executed_flops  # noqa: E402
from automl_package.utils.run_provenance import run_provenance  # noqa: E402

RESULTS_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "WSEL19")
_WSEL6_RESULTS_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "WSEL6")

W_MAX = w6.W_MAX  # 12
SEEDS = w6.SEEDS  # (0, 1, 2)
N_SEL_VALUES = (75, 300, 1200)  # toy-design spec SS3: 75 = starved regime, 1200 = unstarved contrast.

_SELECTION_POOL_SEED_OFFSET = 1000  # the third disjoint make_hetero draw this task needs -- see module docstring.
_REPORT_SEED_OFFSET = 500  # this strand's universal disjoint-test-draw convention (module docstring).

# ---------------------------------------------------------------------------
# Multi-feature extension constants (toy-design spec §2b/§3b/§5, amended -- GO rendered).
# ---------------------------------------------------------------------------

_MF_DIM_CHOICES = (1, *wt.D_GRID)  # 1 (the calibration/anchor dimension) + the toy-design's own d-grid (2, 8, 32).
_MF_CACHE_DIRNAME = "_mf_cache"
_MF_N_TRAIN_FALLBACK = 4000  # §3b F6: the pre-authorized fallback n_train (a §3.8 ladder value), an EXPLICIT flag only.
_IDENTITY_CHECK_N = 500  # §5.1: any n proves the by-construction identity; a fixed, modest size keeps the check cheap.
_CALIBRATION_D = 1  # §5.2's anchor dimension.
_CALIBRATION_GEOMETRY = wt.Geometry.AXIS  # the canonical d=1 geometry -- v=e_1 is the closest thing to a no-op reparametrization; flagged choice, see run_calibration.
_CALIBRATION_FILENAME = "wsel19_calibration_d1.json"
_HIDDENNESS_MIN_RELATIVE_GAIN = 0.10  # §5.4/F4: the oracle must beat the best FIXED width by >= 10% of the latter's true error.
_HETERO_EASY_REGION = 0  # nwn.make_hetero's region id for x < 0 (the flat-easy branch).
_HETERO_HARD_REGION = 1  # nwn.make_hetero's region id for x >= 0 (the width-hungry sine branch).

# Arm 2 (rule_mlp): sizing rule + regularised-training hyperparameters (chosen defaults, flagged
# for review like every other chosen default in this strand -- see module docstring).
_ARM2_SIZING_BASE = DEFAULT_HIDDEN[0]  # 32 -- calibrated so d=1 recovers the frozen default exactly.
_ARM2_DEPTH = len(DEFAULT_HIDDEN)  # 2, held fixed -- ruling 2 is about WIDTH tracking input size, not depth.
_ARM2_MAX_EPOCHS = 2000
_ARM2_PATIENCE = 30
_ARM2_MIN_DELTA = 1e-4
_ARM2_WEIGHT_DECAY = 1e-4  # "mild" per ruling 6; AdamW (decoupled decay), matching width_wsel11.py's own regularisation-study convention.

# Arms 2 and 3 share ONE internal train/val carve of the selection pool (D1 spec: "the same
# internal split"). VAL_EVERY reused verbatim from converged_width_experiment.py, not re-derived.
_INTERNAL_VAL_EVERY = cwe.VAL_EVERY

# Arm 3 (xgboost): native early stopping + shrinkage + subsampling (ruling 6's own justification
# for testing a tree backend at all). Chosen defaults, not tuned per cell.
_XGB_N_ESTIMATORS = 300
_XGB_MAX_DEPTH = 3
_XGB_LEARNING_RATE = 0.1
_XGB_SUBSAMPLE = 0.8
_XGB_COLSAMPLE_BYTREE = 0.8
_XGB_EARLY_STOPPING_ROUNDS = 20
_MIN_CLASSES_FOR_CLASSIFIER = 2  # below this, every internal-train label is the same capacity -- nothing to fit.


class Backend(enum.Enum):
    """The four router backends this bake-off compares (WSEL-19 header, ratified)."""

    FROZEN_MLP = "frozen_mlp"  # DistilledCapacityRouter at its shipped defaults, unmodified.
    RULE_MLP = "rule_mlp"  # same MLP shape, input-relative hidden size, early stop + weight decay.
    XGBOOST = "xgboost"  # gradient-boosted trees on the same hard labels, native regularisation.
    CONSTANT = "constant"  # the single globally-best capacity -- the does-routing-pay-off control.


class Mode(enum.Enum):
    """The two evaluation modes every backend runs under (ratified: blending is tested, not hand-waved)."""

    HARD = "hard"  # argmax route: one width per input.
    BLEND = "blend"  # probability-weighted: every width contributes, cost reported next to quality.


class FitStatus(enum.Enum):
    """§5.3/F6 fit-gate verdict for one multi-feature `(d, geometry, seed)` cell -- closed set, recorded not refused on."""

    OK = "ok"
    VOID_FOR_FIT = "void_for_fit"


class RoutingStatus(enum.Enum):
    """§5.4/F4 hidden-ness-falsifier verdict -- closed set, recorded not refused on."""

    OK = "ok"
    VOID_FOR_ROUTING = "void_for_routing"


def _rule_sized_hidden(in_dim: int) -> tuple[int, int]:
    """Arm 2's input-dimensionality-relative hidden-width rule (see module docstring for the choice).

    `hidden_width(d) = round(32 * sqrt(d))`, both of the frozen recipe's two layers. `d=1` gives
    exactly 32 -- the frozen default is this rule's own `d=1` instance, satisfying ruling 2.
    """
    width = max(1, round(_ARM2_SIZING_BASE * math.sqrt(in_dim)))
    return (width, width)


def _global_cheapest_within_tolerance(error_table: np.ndarray, tolerance: float) -> int:
    """The constant arm's "one capacity for everyone" pick, SAME tie-band as every other arm.

    Applies `DistilledCapacityRouter`'s own per-row `_cheapest_within_tolerance_labels` FORMULA to
    the table's column MEANS instead of one row -- the whole-selection-set analogue of the
    per-input rule, at `DEFAULT_TOLERANCE`. Deliberately NOT `automl_package.utils.capacity_
    selection.cheapest_within_tolerance`: that helper uses a DIFFERENT, bootstrap-SE-calibrated
    margin for a different purpose (MASTER Decision 18, that module's own docstring: "both exist;
    neither replaces the other"). Ruling 4 pins `DEFAULT_TOLERANCE` as this bake-off's tie-band of
    record, so the constant arm reads the same number the other three arms train against.

    Args:
        error_table: `(n_samples, n_capacities)` held-out per-sample error, cheapest-first columns.
        tolerance: relative tolerance above the table's best column mean.

    Returns:
        0-based column index of the smallest capacity within `tolerance` of the best mean error.
    """
    mean_error = error_table.mean(axis=0)
    best = mean_error.min()
    within_tolerance = mean_error <= (1.0 + tolerance) * best
    return int(np.argmax(within_tolerance))  # first True == cheapest capacity within tolerance.


def _jsonable(obj: object) -> object:
    """Minimal JSON-safe coercion (numpy/torch scalars, non-str dict keys) -- local twin of every sibling WSEL driver's helper."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj


# ---------------------------------------------------------------------------
# The expensive part -- 12 dedicated per-width nets, trained/loaded ONCE per seed, cached to disk.
# ---------------------------------------------------------------------------


@dataclass
class _SweepCache:
    """Everything a router backend needs from one seed's per-width sweep -- no net object required."""

    x_pool: np.ndarray  # (max n_sel,) raw x, shuffled once -- nested n_sel subsamples are prefixes.
    y_pool: np.ndarray
    error_table_pool: np.ndarray  # (max n_sel, w_max) plain squared error per width -- labels/training features.
    x_report: np.ndarray  # (n_report,) raw x, disjoint from the pool and from training.
    y_report: np.ndarray
    error_table_report: np.ndarray  # (n_report, w_max) plain squared error per width -- scores every backend.
    flops_by_width: list[float]  # len w_max, executed_flops(net, k) -- deployed-compute readouts.
    hit_cap: bool  # any of the 12 per-width nets hit its epoch cap.
    trustworthy: bool  # all 12 per-width nets converged trustworthily.
    reused_from_wsel6: bool  # True iff every one of the 12 nets was LOADED from WSEL6's cache, not retrained.
    n_train: int
    n_test: int
    n_report: int


def _sweep_cache_json_path(results_dir: str, seed: int) -> str:
    return os.path.join(results_dir, f"cache_seed{seed}.json")


def _get_or_build_sweep_cache(
    seed: int,
    *,
    w_max: int = W_MAX,
    n_train: int | None = None,
    n_test: int | None = None,
    n_sel_values: tuple[int, ...] = N_SEL_VALUES,
    results_dir: str = RESULTS_DIR,
) -> _SweepCache:
    """Loads (or builds) one seed's 12-net error tables, caching them to `results_dir/cache_seed{seed}.json`.

    Reuse only fires when `(n_train, n_test)` match the canonical tier-1 cell exactly (the default,
    `n_train=None`/`n_test=None` resolves to WSEL-6's own `1500`/`500`): `--selftest`'s tiny toy
    overrides both, which correctly skips the reuse check (WSEL-6's on-disk cache has no entry at a
    tiny toy's size, and this task's own naming convention does not encode `n_train` -- checking
    `(n_train, n_test)` explicitly, rather than trusting a filename match, is what keeps a
    tiny-toy selftest from silently loading a mismatched real-sized net).
    """
    cache_path = _sweep_cache_json_path(results_dir, seed)
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            d = json.load(f)
        return _SweepCache(
            x_pool=np.array(d["x_pool"], dtype=np.float32),
            y_pool=np.array(d["y_pool"], dtype=np.float32),
            error_table_pool=np.array(d["error_table_pool"]),
            x_report=np.array(d["x_report"], dtype=np.float32),
            y_report=np.array(d["y_report"], dtype=np.float32),
            error_table_report=np.array(d["error_table_report"]),
            flops_by_width=d["flops_by_width"],
            hit_cap=d["hit_cap"],
            trustworthy=d["trustworthy"],
            reused_from_wsel6=d["reused_from_wsel6"],
            n_train=d["n_train"],
            n_test=d["n_test"],
            n_report=d["n_report"],
        )

    cfg = w6._TIER_CONFIG[w6.Tier.ONE]  # canonical tier-1 hetero cell (width.md SS3.8): n_train=1500, n_test=500, sigma=0.05.
    n_train = cfg.n_train if n_train is None else n_train
    n_test = cfg.n_test if n_test is None else n_test
    can_reuse_wsel6 = n_train == cfg.n_train and n_test == cfg.n_test

    models: dict[int, Any] = {}
    per_width_meta: dict[int, dict[str, Any]] = {}
    reused_all = True
    split: dict[str, Any] | None = None
    for width in range(1, w_max + 1):
        state_path, meta_path = w6._sweep_cache_paths(_WSEL6_RESULTS_DIR, w6.Tier.ONE, seed, width)
        if can_reuse_wsel6 and os.path.exists(state_path) and os.path.exists(meta_path):
            model = w6._load_cached_model((width,), seed, state_path, max_epochs=w4.PORTED_N_EPOCHS_CAP, patience=w4.PORTED_PATIENCE, lr=w4.PORTED_LR_DEFAULT)
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            reused_all = False
            if split is None:
                split = w6._build_split(w6.Tier.ONE, seed, n_train=n_train, n_test=n_test)
            my_state_path, my_meta_path = w6._sweep_cache_paths(results_dir, w6.Tier.ONE, seed, width)
            os.makedirs(os.path.dirname(my_state_path), exist_ok=True)
            model, meta, _cache_hit = w6._get_or_train(
                (width,),
                w6.Tier.ONE,
                seed,
                split,
                my_state_path,
                my_meta_path,
                max_epochs=w4.PORTED_N_EPOCHS_CAP,
                patience=w4.PORTED_PATIENCE,
                min_delta=w4.PORTED_MIN_DELTA,
                lr=w4.PORTED_LR_DEFAULT,
            )
        models[width] = model
        per_width_meta[width] = meta

    # Three disjoint draws (module docstring): training is baked into the loaded/trained nets
    # above; the selection pool and the report set are fresh, never touched by training.
    x_report, y_report, _region_report = nwn.make_hetero(n_test, seed + _REPORT_SEED_OFFSET, sigma=cfg.sigma)
    n_pool = max(n_sel_values)
    x_pool_raw, y_pool_raw, _region_pool = nwn.make_hetero(n_pool, seed + _SELECTION_POOL_SEED_OFFSET, sigma=cfg.sigma)
    perm = np.random.default_rng(seed).permutation(n_pool)
    x_pool, y_pool = x_pool_raw[perm], y_pool_raw[perm]

    x_report_in = x_report.reshape(-1, 1).astype(np.float32)
    x_pool_in = x_pool.reshape(-1, 1).astype(np.float32)
    error_table_report = np.stack([(models[w].predict(x_report_in, filter_data=False, width=w) - y_report) ** 2 for w in range(1, w_max + 1)], axis=1)
    error_table_pool = np.stack([(models[w].predict(x_pool_in, filter_data=False, width=w) - y_pool) ** 2 for w in range(1, w_max + 1)], axis=1)
    flops_by_width = [float(executed_flops(models[w].model, w)) for w in range(1, w_max + 1)]

    cache = _SweepCache(
        x_pool=x_pool,
        y_pool=y_pool,
        error_table_pool=error_table_pool,
        x_report=x_report,
        y_report=y_report,
        error_table_report=error_table_report,
        flops_by_width=flops_by_width,
        hit_cap=any(per_width_meta[w]["hit_cap"] for w in range(1, w_max + 1)),
        trustworthy=all(per_width_meta[w]["trustworthy"] for w in range(1, w_max + 1)),
        reused_from_wsel6=reused_all,
        n_train=n_train,
        n_test=n_test,
        n_report=len(x_report),
    )
    os.makedirs(results_dir, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(
            _jsonable(
                {
                    "x_pool": cache.x_pool,
                    "y_pool": cache.y_pool,
                    "error_table_pool": cache.error_table_pool,
                    "x_report": cache.x_report,
                    "y_report": cache.y_report,
                    "error_table_report": cache.error_table_report,
                    "flops_by_width": cache.flops_by_width,
                    "hit_cap": cache.hit_cap,
                    "trustworthy": cache.trustworthy,
                    "reused_from_wsel6": cache.reused_from_wsel6,
                    "n_train": cache.n_train,
                    "n_test": cache.n_test,
                    "n_report": cache.n_report,
                }
            ),
            f,
        )
    return cache


# ---------------------------------------------------------------------------
# The four backends -- one fit function each, returning a common (route_index, route_proba) shape.
# ---------------------------------------------------------------------------


@dataclass
class _FittedBackend:
    """A trained router backend, reduced to what `_score_hard`/`_score_blend` need to read it."""

    route_index_fn: Callable[[np.ndarray], np.ndarray]  # (N, 1) raw x -> (N,) 0-based capacity index.
    route_proba_fn: Callable[[np.ndarray], np.ndarray]  # (N, 1) raw x -> (N, w_max) class probabilities.
    config: dict[str, Any]  # the fitted backend's own hyperparameters/architecture, for provenance.
    sizing_rule: dict[str, Any] | None  # arm 2 only -- the input-relative rule this cell used.


def _fit_frozen_mlp(x_sel: np.ndarray, y_sel: np.ndarray, error_table_sel: np.ndarray, flops_by_width: list[float], seed: int, w_max: int) -> _FittedBackend:
    """Backend 1: `DistilledCapacityRouter` at its shipped defaults, unmodified (`routing.py:77-80`)."""
    capacity_grid = [(k,) for k in range(1, w_max + 1)]

    def eval_fn(x: np.ndarray, capacity: tuple[int, ...]) -> np.ndarray:
        del x  # ignored -- .fit() always calls this with x_sel, whose per-width error is already cached.
        return error_table_sel[:, capacity[0] - 1]

    def cost_fn(capacity: tuple[int, ...]) -> float:
        return flops_by_width[capacity[0] - 1]

    router = DistilledCapacityRouter(hidden=DEFAULT_HIDDEN, n_epochs=DEFAULT_N_EPOCHS, lr=DEFAULT_LR, seed=seed, device="cpu")
    router.fit(eval_fn=eval_fn, x_val=x_sel, y_val=y_sel, capacity_grid=capacity_grid, tolerance=DEFAULT_TOLERANCE, cost_fn=cost_fn)

    def route_proba(x: np.ndarray) -> np.ndarray:
        x_arr = _as_capacity_input(x)
        with torch.no_grad():
            logits = router.router_(torch.as_tensor(x_arr, dtype=torch.float32, device=router.device))
        return nnf.softmax(logits, dim=1).cpu().numpy()

    return _FittedBackend(
        route_index_fn=router.route_index,
        route_proba_fn=route_proba,
        config={"hidden": list(DEFAULT_HIDDEN), "depth": len(DEFAULT_HIDDEN), "epochs": DEFAULT_N_EPOCHS, "lr": DEFAULT_LR, "weight_decay": 0.0, "early_stopping": False},
        sizing_rule=None,
    )


def _fit_rule_mlp(x_sel: np.ndarray, error_table_sel: np.ndarray, seed: int, w_max: int, *, in_dim: int = 1) -> _FittedBackend:
    """Backend 2: input-relative hidden size, early stopping + mild weight decay, no dropout (ruling 6)."""
    hidden = _rule_sized_hidden(in_dim)
    labels = _cheapest_within_tolerance_labels(error_table_sel, tolerance=DEFAULT_TOLERANCE)
    x_arr = _as_capacity_input(x_sel)
    n = len(x_arr)
    val_mask = (np.arange(n) % _INTERNAL_VAL_EVERY) == 0

    torch.manual_seed(seed)
    net = _CapacityRouterMLP(in_dim, w_max, hidden=hidden)
    opt = torch.optim.AdamW(net.parameters(), lr=DEFAULT_LR, weight_decay=_ARM2_WEIGHT_DECAY)
    x_t = torch.as_tensor(x_arr, dtype=torch.float32)
    y_t = torch.as_tensor(labels, dtype=torch.long)
    x_tr_t, y_tr_t = x_t[~val_mask], y_t[~val_mask]
    x_val_t, y_val_t = x_t[val_mask], y_t[val_mask]

    best_val, patience_counter, best_state = float("inf"), 0, None
    actual_epochs = 0
    for epoch in range(_ARM2_MAX_EPOCHS):
        net.train()
        opt.zero_grad()
        loss = nnf.cross_entropy(net(x_tr_t), y_tr_t)
        loss.backward()
        opt.step()

        net.eval()
        with torch.no_grad():
            val_loss = float(nnf.cross_entropy(net(x_val_t), y_val_t).item())
        actual_epochs = epoch + 1
        if val_loss < best_val - _ARM2_MIN_DELTA:
            best_val, patience_counter = val_loss, 0
            best_state = {name: t.detach().clone() for name, t in net.state_dict().items()}
        else:
            patience_counter += 1
        if patience_counter >= _ARM2_PATIENCE:
            break

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()

    def route_index(x: np.ndarray) -> np.ndarray:
        x_arr2 = _as_capacity_input(x)
        with torch.no_grad():
            logits = net(torch.as_tensor(x_arr2, dtype=torch.float32))
        return logits.argmax(dim=1).cpu().numpy()

    def route_proba(x: np.ndarray) -> np.ndarray:
        x_arr2 = _as_capacity_input(x)
        with torch.no_grad():
            logits = net(torch.as_tensor(x_arr2, dtype=torch.float32))
        return nnf.softmax(logits, dim=1).cpu().numpy()

    return _FittedBackend(
        route_index_fn=route_index,
        route_proba_fn=route_proba,
        config={
            "hidden": list(hidden),
            "depth": _ARM2_DEPTH,
            "lr": DEFAULT_LR,
            "weight_decay": _ARM2_WEIGHT_DECAY,
            "early_stopping": True,
            "patience": _ARM2_PATIENCE,
            "min_delta": _ARM2_MIN_DELTA,
            "max_epochs": _ARM2_MAX_EPOCHS,
            "actual_epochs": actual_epochs,
            "hit_cap": actual_epochs >= _ARM2_MAX_EPOCHS,
            "n_train_internal": int((~val_mask).sum()),
            "n_val_internal": int(val_mask.sum()),
        },
        sizing_rule={
            "kind": "sqrt_input_dim",
            "hidden_width_formula": "round(32 * sqrt(in_dim))",
            "in_dim": in_dim,
            "hidden": list(hidden),
            "depth": _ARM2_DEPTH,
            "base_at_d1": _ARM2_SIZING_BASE,
        },
    )


def _fit_xgboost(x_sel: np.ndarray, error_table_sel: np.ndarray, seed: int, w_max: int) -> _FittedBackend:
    """Backend 3: gradient-boosted trees on the same hard labels, native early stopping (ruling 6).

    XGBoost's sklearn API requires the FIT call's own `y` to be a DENSE `0..k-1` range -- confirmed
    by probing (`ValueError: Invalid classes inferred from unique values of y. Expected: [0, 1],
    got [0, 2]`) rather than assumed from the docs. A capacity label absent from the internal TRAIN
    split is routine at small `n_sel` (the toy-design spec's own starvation arithmetic anticipates
    exactly this), so labels are remapped to a dense space keyed off the TRAIN split's own observed
    classes; any validation-split label outside that space is dropped from `eval_set` (the model has
    no training signal for a class it never saw). If only one class survives in TRAIN, there is
    nothing for a classifier to learn -- route everyone to it directly rather than calling xgboost
    on a single-class fit (which XGBoost itself rejects).
    """
    labels = _cheapest_within_tolerance_labels(error_table_sel, tolerance=DEFAULT_TOLERANCE)
    x_arr = _as_capacity_input(x_sel)
    n = len(x_arr)
    val_mask = (np.arange(n) % _INTERNAL_VAL_EVERY) == 0  # SAME internal split as rule_mlp (D1 spec).
    x_tr, y_tr_raw = x_arr[~val_mask], labels[~val_mask]
    x_val_raw, y_val_raw = x_arr[val_mask], labels[val_mask]

    train_classes = np.unique(y_tr_raw)  # sorted ascending -- the dense label space this fit trains in.
    class_to_dense = {int(c): i for i, c in enumerate(train_classes)}
    y_tr = np.array([class_to_dense[int(c)] for c in y_tr_raw], dtype=int)
    in_vocab = np.isin(y_val_raw, train_classes)
    x_val, y_val = x_val_raw[in_vocab], np.array([class_to_dense[int(c)] for c in y_val_raw[in_vocab]], dtype=int)
    use_early_stopping = len(x_val) > 0

    if len(train_classes) < _MIN_CLASSES_FOR_CLASSIFIER:
        c_star = int(train_classes[0])

        def route_index_constant(x: np.ndarray) -> np.ndarray:
            return np.full(len(x), c_star, dtype=int)

        def route_proba_constant(x: np.ndarray) -> np.ndarray:
            proba = np.zeros((len(x), w_max), dtype=np.float64)
            proba[:, c_star] = 1.0
            return proba

        return _FittedBackend(
            route_index_fn=route_index_constant,
            route_proba_fn=route_proba_constant,
            config={
                "degenerate_single_class": True,
                "selected_capacity_index": c_star,
                "n_train_internal": int((~val_mask).sum()),
                "n_val_internal": int(val_mask.sum()),
            },
            sizing_rule=None,
        )

    clf = xgb.XGBClassifier(
        n_estimators=_XGB_N_ESTIMATORS,
        max_depth=_XGB_MAX_DEPTH,
        learning_rate=_XGB_LEARNING_RATE,
        subsample=_XGB_SUBSAMPLE,
        colsample_bytree=_XGB_COLSAMPLE_BYTREE,
        early_stopping_rounds=_XGB_EARLY_STOPPING_ROUNDS if use_early_stopping else None,
        eval_metric="mlogloss",
        random_state=seed,
        verbosity=0,
    )
    if use_early_stopping:
        clf.fit(x_tr, y_tr, eval_set=[(x_val, y_val)], verbose=False)
    else:
        clf.fit(x_tr, y_tr)

    def route_index(x: np.ndarray) -> np.ndarray:
        dense_idx = clf.predict(_as_capacity_input(x))
        return train_classes[dense_idx]

    def route_proba(x: np.ndarray) -> np.ndarray:
        proba_dense = clf.predict_proba(_as_capacity_input(x))  # (N, len(train_classes)), column i == dense class i.
        proba_full = np.zeros((len(x), w_max), dtype=np.float64)
        proba_full[:, train_classes] = proba_dense  # dense class i IS train_classes[i] -- direct placement.
        return proba_full

    return _FittedBackend(
        route_index_fn=route_index,
        route_proba_fn=route_proba,
        config={
            "n_estimators": _XGB_N_ESTIMATORS,
            "max_depth": _XGB_MAX_DEPTH,
            "learning_rate": _XGB_LEARNING_RATE,
            "subsample": _XGB_SUBSAMPLE,
            "colsample_bytree": _XGB_COLSAMPLE_BYTREE,
            "early_stopping_rounds": _XGB_EARLY_STOPPING_ROUNDS if use_early_stopping else None,
            "early_stopping_active": use_early_stopping,
            "best_iteration": int(clf.best_iteration) if use_early_stopping else _XGB_N_ESTIMATORS,
            "n_classes_observed": len(train_classes),
            "n_train_internal": int((~val_mask).sum()),
            "n_val_internal": len(x_val),
        },
        sizing_rule=None,
    )


def _fit_constant(error_table_sel: np.ndarray, w_max: int) -> _FittedBackend:
    """Backend 4: always the single globally-best capacity -- the does-routing-pay-off control."""
    c_star = _global_cheapest_within_tolerance(error_table_sel, DEFAULT_TOLERANCE)

    def route_index(x: np.ndarray) -> np.ndarray:
        return np.full(len(x), c_star, dtype=int)

    def route_proba(x: np.ndarray) -> np.ndarray:
        proba = np.zeros((len(x), w_max), dtype=np.float64)
        proba[:, c_star] = 1.0  # a one-hot distribution -- HARD and BLEND coincide for this arm.
        return proba

    return _FittedBackend(
        route_index_fn=route_index,
        route_proba_fn=route_proba,
        config={"selected_capacity_index": c_star, "selected_width": c_star + 1, "tolerance": DEFAULT_TOLERANCE},
        sizing_rule=None,
    )


def _fit_backend(
    backend: Backend, x_sel: np.ndarray, y_sel: np.ndarray, error_table_sel: np.ndarray, flops_by_width: list[float], seed: int, w_max: int, in_dim: int
) -> _FittedBackend:
    """Dispatches to the requested backend's fit function -- each arm's own signature differs, so this is a plain branch, not a uniform dict of lambdas."""
    if backend is Backend.FROZEN_MLP:
        return _fit_frozen_mlp(x_sel, y_sel, error_table_sel, flops_by_width, seed, w_max)
    if backend is Backend.RULE_MLP:
        return _fit_rule_mlp(x_sel, error_table_sel, seed, w_max, in_dim=in_dim)
    if backend is Backend.XGBOOST:
        return _fit_xgboost(x_sel, error_table_sel, seed, w_max)
    return _fit_constant(error_table_sel, w_max)


# ---------------------------------------------------------------------------
# Readouts -- HARD (argmax) and BLEND (probability-weighted expectation), same fitted backend.
# ---------------------------------------------------------------------------


def _score_hard(fitted: _FittedBackend, x_report: np.ndarray, error_table_report: np.ndarray, flops_by_width: list[float], w_max: int) -> dict[str, Any]:
    """HARD mode: argmax route, routed held-out quality, oracle agreement, mean deployed FLOPs."""
    idx = np.asarray(fitted.route_index_fn(x_report), dtype=int)
    rows = np.arange(len(x_report))
    flops_arr = np.asarray(flops_by_width)
    oracle_idx = _cheapest_within_tolerance_labels(error_table_report, tolerance=DEFAULT_TOLERANCE)
    routed_widths = (idx + 1).tolist()

    return {
        "routed_held_out_quality": float(error_table_report[rows, idx].mean()),
        "mean_deployed_flops": float(flops_arr[idx].mean()),
        "oracle_agreement": float(np.mean(idx == oracle_idx)),
        "mean_routed_width": float(np.mean(routed_widths)),
        "width_distribution": {str(k): routed_widths.count(k) for k in range(1, w_max + 1)},
    }


def _score_blend(fitted: _FittedBackend, x_report: np.ndarray, error_table_report: np.ndarray, flops_by_width: list[float], w_max: int) -> dict[str, Any]:
    """BLEND mode: probability-weighted expected quality and expected deployed FLOPs, no single pick."""
    del w_max  # unused -- proba's own column count fixes it; kept for signature symmetry with _score_hard.
    proba = fitted.route_proba_fn(x_report)  # (n_report, w_max)
    flops_arr = np.asarray(flops_by_width)
    widths_arr = np.arange(1, proba.shape[1] + 1)

    return {
        "routed_held_out_quality": float((proba * error_table_report).sum(axis=1).mean()),
        "mean_deployed_flops": float((proba * flops_arr[None, :]).sum(axis=1).mean()),
        "oracle_agreement": None,  # HARD-only readout (D1 spec) -- BLEND has no single per-row pick to agree/disagree.
        "mean_routed_width": float((proba * widths_arr[None, :]).sum(axis=1).mean()),
        "width_distribution": None,
    }


_SCORERS: dict[Mode, Callable[[_FittedBackend, np.ndarray, np.ndarray, list[float], int], dict[str, Any]]] = {
    Mode.HARD: _score_hard,
    Mode.BLEND: _score_blend,
}


# ---------------------------------------------------------------------------
# One cell.
# ---------------------------------------------------------------------------


def run_cell(
    backend: Backend,
    mode: Mode,
    n_sel: int,
    seed: int,
    *,
    w_max: int = W_MAX,
    n_train: int | None = None,
    n_test: int | None = None,
    n_sel_values: tuple[int, ...] = N_SEL_VALUES,
    results_dir: str = RESULTS_DIR,
) -> dict[str, Any]:
    """Runs one (backend, mode, n_sel, seed) cell: get-or-build the sweep cache, fit, score. Returns the JSON-able case."""
    cache = _get_or_build_sweep_cache(seed, w_max=w_max, n_train=n_train, n_test=n_test, n_sel_values=n_sel_values, results_dir=results_dir)

    x_sel = cache.x_pool[:n_sel]
    y_sel = cache.y_pool[:n_sel]
    error_table_sel = cache.error_table_pool[:n_sel]
    in_dim = 1 if x_sel.ndim == 1 else x_sel.shape[1]

    fitted = _fit_backend(backend, x_sel, y_sel, error_table_sel, cache.flops_by_width, seed, w_max, in_dim)
    readout = _SCORERS[mode](fitted, cache.x_report, cache.error_table_report, cache.flops_by_width, w_max)

    case = {
        "backend": backend.value,
        "mode": mode.value,
        "n_sel": n_sel,
        "seed": seed,
        "toy": nwn.Toy.HETERO.value,
        "n_train": cache.n_train,
        "n_test": cache.n_test,
        "w_max": w_max,
        "n_report": cache.n_report,
        "backend_config": fitted.config,
        "sizing_rule": fitted.sizing_rule,
        "table_provenance": {
            "reused_from_wsel6": cache.reused_from_wsel6,
            "wsel6_cache_dir": _WSEL6_RESULTS_DIR if cache.reused_from_wsel6 else None,
            "sweep_hit_cap": cache.hit_cap,
            "sweep_trustworthy": cache.trustworthy,
        },
        "provenance": run_provenance(),
    }
    case.update(readout)
    return case


def _cell_json_path(results_dir: str, backend: Backend, mode: Mode, n_sel: int, seed: int, tag: str | None = None) -> str:
    suffix = f"_{tag}" if tag else ""
    return os.path.join(results_dir, f"wsel19_{backend.value}_{mode.value}_nsel{n_sel}_seed{seed}{suffix}.json")


# ---------------------------------------------------------------------------
# Multi-feature extension (toy-design spec §2b/§3b/§5, amended -- GO rendered, see module
# docstring). The expensive part -- w_max dedicated per-width nets, trained fresh per (d,
# geometry, seed[, n_train]) and cached to disk -- mirrors `_SweepCache`/`_get_or_build_sweep_
# cache` above in shape, but built on `FlexibleWidthNN` directly (the SAME PORTED-arm protocol
# width_wsel6.py uses) rather than reusing WSEL6's 1-D-only cache.
# ---------------------------------------------------------------------------


def _mf_new_model(in_dim: int, width: int, seed: int, *, max_epochs: int, patience: int, lr: float) -> FlexibleWidthNN:
    """One dedicated single-width `FlexibleWidthNN`, `width_wsel6._new_model`'s PORTED-arm protocol verbatim, generalized to `in_dim`.

    No package change is needed for this: `FlexibleWidthNN`/`FlexibleWidthNNModule` already infer
    `input_size` from the training data's own column count at `_fit_single` time (`automl_package/
    models/base_pytorch.py:124-125`, confirmed by direct read and a live check at authoring time,
    not assumed) -- `input_size=in_dim` here is only what an unfitted instance reports before that
    inference runs.
    """
    return FlexibleWidthNN(
        input_size=in_dim,
        output_size=1,
        task_type=TaskType.REGRESSION,
        widths=(width,),
        learning_rate=lr,
        n_epochs=max_epochs,
        early_stopping_rounds=patience,
        batch_size=w4.PORTED_BATCH_SIZE,
        random_seed=seed,
        calculate_feature_importance=False,
        capacity_selection=CapacitySelection.FIXED,
        activation=ActivationFunction.TANH,  # width_wsel4.py's confound-doctrine finding, reused verbatim (width_wsel6.py module docstring).
    )


def _mf_train_one_width(
    in_dim: int, width: int, seed: int, x_tr: np.ndarray, y_tr: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, *, max_epochs: int, patience: int, lr: float
) -> tuple[FlexibleWidthNN, list[float]]:
    """Trains one dedicated width-`width` `FlexibleWidthNN` via the `_fit_single` bypass.

    `width_wsel6._train_tier1`'s exact pattern (tier-1 hetero's built-in MSE criterion IS SS3.7's
    fixed-sigma likelihood on this toy's single constant sigma; no hetero3/weighted-loss cells here
    per the toy-design spec's own non-goals).
    """
    model = _mf_new_model(in_dim, width, seed, max_epochs=max_epochs, patience=patience, lr=lr)
    _best_epoch, val_loss_history = model._fit_single(x_tr, y_tr, x_val=x_val, y_val=y_val)
    return model, val_loss_history


def _mf_model_cache_paths(results_dir: str, d: int, geometry: wt.Geometry, seed: int, n_train: int, width: int) -> tuple[str, str]:
    base = os.path.join(results_dir, _MF_CACHE_DIRNAME)
    os.makedirs(base, exist_ok=True)
    tag = f"d{d}_{geometry.value}_seed{seed}_ntrain{n_train}_w{width}"
    return os.path.join(base, f"{tag}.pt"), os.path.join(base, f"{tag}_meta.json")


def _mf_load_cached_model(in_dim: int, width: int, seed: int, state_path: str, *, max_epochs: int, patience: int, lr: float) -> FlexibleWidthNN:
    """Rebuilds a multi-feature `FlexibleWidthNN`'s module and loads a previously-cached `state_dict` back into it."""
    model = _mf_new_model(in_dim, width, seed, max_epochs=max_epochs, patience=patience, lr=lr)
    model.input_size = in_dim
    model.build_model()
    model.model.load_state_dict(torch.load(state_path, map_location=model.device))
    model.model.eval()
    return model


def _mf_get_or_train_one_width(
    in_dim: int,
    width: int,
    seed: int,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    state_path: str,
    meta_path: str,
    *,
    max_epochs: int,
    patience: int,
    min_delta: float,
    lr: float,
) -> tuple[FlexibleWidthNN, dict[str, Any], bool]:
    """Get-or-train ONE multi-feature width-`width` net, cached at `state_path`/`meta_path` (mirrors `width_wsel6._get_or_train`)."""
    if os.path.exists(state_path) and os.path.exists(meta_path):
        model = _mf_load_cached_model(in_dim, width, seed, state_path, max_epochs=max_epochs, patience=patience, lr=lr)
        with open(meta_path) as f:
            meta = json.load(f)
        return model, meta, True

    model, val_loss_history = _mf_train_one_width(in_dim, width, seed, x_tr, y_tr, x_val, y_val, max_epochs=max_epochs, patience=patience, lr=lr)
    replay = w4._replay(val_loss_history, patience, min_delta)
    hit_cap = bool(len(val_loss_history) >= max_epochs)
    trustworthy = bool(replay.trustworthy and not hit_cap)
    meta = {
        "trajectory": replay.summary()["trajectory"],
        "actual_epochs": len(val_loss_history),
        "n_train_used": len(x_tr),
        "trustworthy": trustworthy,
        "hit_cap": hit_cap,
        "objective": "mse",
    }
    torch.save(model.model.state_dict(), state_path)
    with open(meta_path, "w") as f:
        json.dump(_jsonable(meta), f)
    return model, meta, False


def _get_or_build_mf_models(
    seed: int,
    d: int,
    geometry: wt.Geometry,
    *,
    w_max: int = W_MAX,
    n_train: int | None = None,
    results_dir: str = RESULTS_DIR,
    max_epochs: int = w4.PORTED_N_EPOCHS_CAP,
    patience: int = w4.PORTED_PATIENCE,
    min_delta: float = w4.PORTED_MIN_DELTA,
    lr: float = w4.PORTED_LR_DEFAULT,
) -> tuple[dict[int, FlexibleWidthNN], dict[int, dict[str, Any]], bool, int]:
    """The §2b/§3b multi-feature twin of `width_wsel6._get_or_train_sweep_all`.

    `w_max` dedicated single-width nets, trained ONCE per (d, geometry, seed, n_train) on the FULL
    independent training draw (`wt.make_hetero_multifeature` at the cell's own base seed -- no
    p1/p2 carve; §3b/C8: selection is never a carve of training), cached to disk under
    `results_dir/_mf_cache/`.

    Returns:
        `(models, metas, all_cache_hit, n_train)` -- `n_train` is the ACTUAL value used (the §3b
        canonical 1500 unless the caller passed an explicit override, e.g. the F6 fallback).
    """
    n_train = wt.TRAIN_N if n_train is None else n_train
    x_tr_full, y_tr_full, _region_tr, _t_tr = wt.make_hetero_multifeature(n_train, seed, d, geometry)
    val_mask = (np.arange(n_train) % _INTERNAL_VAL_EVERY) == 0
    x_tr, y_tr = x_tr_full[~val_mask], y_tr_full[~val_mask]
    x_val, y_val = x_tr_full[val_mask], y_tr_full[val_mask]

    models: dict[int, FlexibleWidthNN] = {}
    metas: dict[int, dict[str, Any]] = {}
    all_cache_hit = True
    for width in range(1, w_max + 1):
        state_path, meta_path = _mf_model_cache_paths(results_dir, d, geometry, seed, n_train, width)
        model, meta, cache_hit = _mf_get_or_train_one_width(
            d, width, seed, x_tr, y_tr, x_val, y_val, state_path, meta_path, max_epochs=max_epochs, patience=patience, min_delta=min_delta, lr=lr
        )
        models[width], metas[width] = model, meta
        all_cache_hit = all_cache_hit and cache_hit
    return models, metas, all_cache_hit, n_train


def _mf_error_table(models: dict[int, FlexibleWidthNN], x: np.ndarray, y: np.ndarray, w_max: int) -> np.ndarray:
    """`(n, w_max)` per-sample squared error against `y`, one column per width -- the held-out (noisy-target) table every backend trains/scores against."""
    return np.stack([(models[w].predict(x, filter_data=False, width=w) - y) ** 2 for w in range(1, w_max + 1)], axis=1)


def _hetero_h(t: np.ndarray, r: float = nwn.HETERO_R_DEFAULT) -> np.ndarray:
    """The generator-TRUE, noise-free hetero signal `h(t)`.

    `nested_width_net.make_hetero`'s own `y_signal` formula (`nested_width_net.py:175`), verbatim,
    evaluated on the untouched `t` (never re-derived from a noisy `y`). §5.4/F4's oracle-optimism
    fix needs exactly this: the oracle must be graded against the TRUE signal, never a single noisy
    draw (whose row-minima reward selection-on-noise, F4's own finding).
    """
    return np.where(t < 0, (0.5 / r) * t, 0.5 * np.sin(t)).astype(np.float32)


def _regime_prefers_different_widths(true_error_table: np.ndarray, region: np.ndarray) -> tuple[bool, int, int]:
    """§5.2/§5.5: does the per-width TRUE-error table prefer a different width in the easy vs width-hungry region?

    Args:
        true_error_table: `(n, w_max)` per-sample `(pred_w(x) - h(t))**2`.
        region: `(n,)` int array, `nwn.make_hetero`'s own region id (0 = easy, 1 = hard), untouched by the §2 lift.

    Returns:
        `(regime_visible, best_width_easy, best_width_hard)` -- widths are 1-based.
    """
    best_easy = int(np.argmin(true_error_table[region == _HETERO_EASY_REGION].mean(axis=0))) + 1
    best_hard = int(np.argmin(true_error_table[region == _HETERO_HARD_REGION].mean(axis=0))) + 1
    return best_easy != best_hard, best_easy, best_hard


def _calibration_path(results_dir: str) -> str:
    return os.path.join(results_dir, _CALIBRATION_FILENAME)


def _load_calibration_or_refuse(results_dir: str) -> dict[str, Any]:
    """§5.2 gate: every multi-feature cell REFUSES (raises) unless the calibration artifact exists and passed."""
    path = _calibration_path(results_dir)
    if not os.path.exists(path):
        raise RuntimeError(f"multi-feature cell refused (§5.2 gate): no calibration artifact at {path} -- run --calibrate first.")
    with open(path) as f:
        calibration = json.load(f)
    if not calibration.get("passed", False):
        raise RuntimeError(
            f"multi-feature cell refused (§5.2 gate): calibration artifact at {path} did not pass -- "
            "the d=1 calibration cell must reproduce the canonical regime structure before any multi-feature cell may run."
        )
    return calibration


def run_calibration(
    *, w_max: int = W_MAX, seeds: tuple[int, ...] = SEEDS, n_train: int | None = None, results_dir: str = RESULTS_DIR
) -> dict[str, Any]:
    """§5.2: the d=1 calibration cell, gating every multi-feature cell.

    Runs the SAME multi-feature per-width sweep at `d=1, geometry=axis` (the canonical anchor:
    `v=e_1` is the closest thing to a no-op reparametrization of the 1-D construction, a flagged
    choice like every other chosen default in this strand), for every canonical seed. Asserts the
    canonical qualitative regime structure (different best widths in the easy vs width-hungry
    region) survives the probability-integral-transform lift BEFORE any multi-feature (d>1) cell
    may run, and writes the anchor every such cell's fit gate (§5.3) reads
    (`_load_calibration_or_refuse`).

    The anchor (`anchor_ratio_to_noise_floor`) is the MAXIMUM best-fixed-width-held-out-MSE /
    noise-floor ratio across the calibration seeds -- the worst-case d=1 fit quality, not a mean or
    a single seed, so a later multi-feature cell is graded against the loosest defensible 1-D
    reference rather than amplifying calibration seed noise into a spurious fit-gate failure. No
    new constant is introduced (F6: "anchor-relative, no new arbitrary constant") -- the ratio
    itself IS the bar, with no added tolerance multiplier.
    """
    noise_floor = float(nwn.HETERO_NOISE_SIGMA**2)
    per_seed: dict[str, Any] = {}
    ratios: list[float] = []
    regime_flags: list[bool] = []
    trustworthy_flags: list[bool] = []

    for seed in seeds:
        models, metas, _cache_hit, n_train_used = _get_or_build_mf_models(seed, _CALIBRATION_D, _CALIBRATION_GEOMETRY, w_max=w_max, n_train=n_train, results_dir=results_dir)
        x_report, y_report, region_report, t_report = wt.make_report_split(seed, _CALIBRATION_D, _CALIBRATION_GEOMETRY)
        error_table_report = _mf_error_table(models, x_report, y_report, w_max)
        h_report = _hetero_h(t_report)
        true_error_table = np.stack([(models[w].predict(x_report, filter_data=False, width=w) - h_report) ** 2 for w in range(1, w_max + 1)], axis=1)

        best_fixed_mse = float(error_table_report.mean(axis=0).min())
        ratio = best_fixed_mse / noise_floor
        regime_visible, best_easy, best_hard = _regime_prefers_different_widths(true_error_table, region_report)
        seed_trustworthy = all(metas[w]["trustworthy"] for w in range(1, w_max + 1))

        ratios.append(ratio)
        regime_flags.append(regime_visible)
        trustworthy_flags.append(seed_trustworthy)
        per_seed[str(seed)] = {
            "best_fixed_held_out_mse": best_fixed_mse,
            "ratio_to_noise_floor": ratio,
            "regime_visible": regime_visible,
            "best_width_easy": best_easy,
            "best_width_hard": best_hard,
            "trustworthy": seed_trustworthy,
            "n_train": n_train_used,
        }

    anchor_ratio = float(max(ratios))
    regime_visible_all = bool(all(regime_flags))
    all_trustworthy = bool(all(trustworthy_flags))
    passed = regime_visible_all and all_trustworthy

    out = {
        "d": _CALIBRATION_D,
        "geometry": _CALIBRATION_GEOMETRY.value,
        "seeds": list(seeds),
        "per_seed": per_seed,
        "anchor_ratio_to_noise_floor": anchor_ratio,
        "regime_visible_all_seeds": regime_visible_all,
        "all_trustworthy": all_trustworthy,
        "passed": passed,
        "noise_floor": noise_floor,
        "w_max": w_max,
        "provenance": run_provenance(),
    }
    os.makedirs(results_dir, exist_ok=True)
    path = _calibration_path(results_dir)
    with open(path, "w") as f:
        json.dump(_jsonable(out), f, indent=2)
    print(f"wrote {path}  passed={passed}  anchor_ratio_to_noise_floor={anchor_ratio:.4g}")
    return out


def run_cell_multifeature(
    backend: Backend,
    mode: Mode,
    n_sel: int,
    seed: int,
    d: int,
    geometry: wt.Geometry,
    *,
    w_max: int = W_MAX,
    n_train: int | None = None,
    n_train_fallback: bool = False,
    fallback_n_train: int = _MF_N_TRAIN_FALLBACK,
    results_dir: str = RESULTS_DIR,
) -> dict[str, Any]:
    """Runs one multi-feature (backend, mode, n_sel, seed, d, geometry) cell.

    The §5 validity checks run IN ORDER and are recorded in `validity_checks`, BEFORE the backend
    verdict below (§5's own ordering requirement) -- see module docstring for what each one means.

    `n_train` is a direct size override for `--selftest`'s tiny toys only (mirrors `run_cell`'s own
    `n_train`/`n_test` params -- no CLI flag, since the toy-design spec exposes only the binary F6
    fallback as a first-class knob). `n_train_fallback=True` uses `fallback_n_train` (the §3b F6
    pre-authorized 4000 by default -- overridable so `--selftest` can exercise the flag's wiring at
    a tiny size instead) and takes precedence over an explicit `n_train` if both are somehow given.
    """
    calibration = _load_calibration_or_refuse(results_dir)  # §5.2 -- REFUSES (raises) if missing/failed.

    identity_ok = wt.identity_holds(_IDENTITY_CHECK_N, seed, d, geometry)  # §5.1 -- holds by construction; a False here is a real defect.
    if not identity_ok:
        raise RuntimeError(f"§5.1 identity check FAILED at d={d} geometry={geometry.value} seed={seed} -- the multi-feature construction itself is broken.")

    n_train_arg = fallback_n_train if n_train_fallback else n_train
    models, metas, cache_hit, n_train_used = _get_or_build_mf_models(seed, d, geometry, w_max=w_max, n_train=n_train_arg, results_dir=results_dir)

    x_report, y_report, region_report, t_report = wt.make_report_split(seed, d, geometry)
    x_sel, y_sel, _region_sel, _t_sel = wt.make_selection_split(seed, d, geometry, n_sel)

    error_table_report = _mf_error_table(models, x_report, y_report, w_max)
    error_table_sel = _mf_error_table(models, x_sel, y_sel, w_max)
    flops_by_width = [float(executed_flops(models[w].model, w)) for w in range(1, w_max + 1)]

    # §5.3/F6 fit gate -- anchor-relative to the d=1 calibration cell, no new constant.
    noise_floor = float(nwn.HETERO_NOISE_SIGMA**2)
    best_fixed_held_out_mse = float(error_table_report.mean(axis=0).min())
    ratio_to_noise_floor = best_fixed_held_out_mse / noise_floor
    fit_status = FitStatus.OK if ratio_to_noise_floor <= calibration["anchor_ratio_to_noise_floor"] else FitStatus.VOID_FOR_FIT

    # §5.4/F4 hidden-ness falsifier -- true-error oracle vs best FIXED width, both against h(t), never noisy y.
    h_report = _hetero_h(t_report)
    true_error_table_report = np.stack([(models[w].predict(x_report, filter_data=False, width=w) - h_report) ** 2 for w in range(1, w_max + 1)], axis=1)
    oracle_true_mean = float(true_error_table_report.min(axis=1).mean())
    best_fixed_true_mean = float(true_error_table_report.mean(axis=0).min())
    relative_gain = (best_fixed_true_mean - oracle_true_mean) / best_fixed_true_mean if best_fixed_true_mean > 0 else 0.0
    routing_status = RoutingStatus.OK if relative_gain >= _HIDDENNESS_MIN_RELATIVE_GAIN else RoutingStatus.VOID_FOR_ROUTING

    # §5.5 regime visibility, off the SAME true-error table.
    regime_visible, best_width_easy, best_width_hard = _regime_prefers_different_widths(true_error_table_report, region_report)

    fitted = _fit_backend(backend, x_sel, y_sel, error_table_sel, flops_by_width, seed, w_max, d)
    readout = _SCORERS[mode](fitted, x_report, error_table_report, flops_by_width, w_max)

    case = {
        "backend": backend.value,
        "mode": mode.value,
        "n_sel": n_sel,
        "seed": seed,
        "d": d,
        "geometry": geometry.value,
        "toy": nwn.Toy.HETERO.value,
        "n_train": n_train_used,
        "n_train_fallback_applied": bool(n_train_fallback),
        "w_max": w_max,
        "n_report": len(x_report),
        "backend_config": fitted.config,
        "sizing_rule": fitted.sizing_rule,
        "table_provenance": {
            "reused_from_wsel6": False,
            "sweep_hit_cap": any(metas[w]["hit_cap"] for w in range(1, w_max + 1)),
            "sweep_trustworthy": all(metas[w]["trustworthy"] for w in range(1, w_max + 1)),
            "cache_hit": bool(cache_hit),
        },
        "validity_checks": {
            "identity_holds": identity_ok,
            "fit_status": fit_status.value,
            "ratio_to_noise_floor": ratio_to_noise_floor,
            "calibration_anchor_ratio_to_noise_floor": calibration["anchor_ratio_to_noise_floor"],
            "routing_status": routing_status.value,
            "hiddenness_relative_gain": relative_gain,
            "regime_visible": regime_visible,
            "best_width_easy": best_width_easy,
            "best_width_hard": best_width_hard,
        },
        "provenance": run_provenance(),
    }
    case.update(readout)
    return case


def _mf_cell_json_path(results_dir: str, d: int, geometry: wt.Geometry, backend: Backend, mode: Mode, n_sel: int, seed: int, tag: str | None = None) -> str:
    suffix = f"_{tag}" if tag else ""
    return os.path.join(results_dir, f"wsel19_mf_d{d}_{geometry.value}_{backend.value}_{mode.value}_nsel{n_sel}_seed{seed}{suffix}.json")


# ---------------------------------------------------------------------------
# --summarize -- aggregates every per-cell JSON on disk into WSEL19/frozen.json. No verdict is
# invented here: the D1 spec asks for an aggregate, not a plateau/invariance read (unlike WSEL-6/7,
# which specify one) -- mean +/- SE per (backend, mode, n_sel) across present seeds, transparently.
# ---------------------------------------------------------------------------


def _mean_se(values: list[float]) -> dict[str, float]:
    """Mean and standard error of `values` (SE is 0.0 for fewer than 2 values, not NaN)."""
    arr = np.asarray(values, dtype=np.float64)
    se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return {"mean": float(arr.mean()), "se": se, "n": len(arr)}


def summarize(results_dir: str = RESULTS_DIR) -> None:
    """Aggregates every per-cell JSON on disk into `WSEL19/frozen.json`. Does not train or fit anything."""
    per_group: dict[str, dict[str, Any]] = {}
    n_present = 0
    for backend in Backend:
        for mode in Mode:
            for n_sel in N_SEL_VALUES:
                quality, flops, agreement, width = [], [], [], []
                for seed in SEEDS:
                    path = _cell_json_path(results_dir, backend, mode, n_sel, seed)
                    if not os.path.exists(path):
                        continue
                    with open(path) as f:
                        cell = json.load(f)
                    n_present += 1
                    quality.append(cell["routed_held_out_quality"])
                    flops.append(cell["mean_deployed_flops"])
                    width.append(cell["mean_routed_width"])
                    if cell["oracle_agreement"] is not None:
                        agreement.append(cell["oracle_agreement"])
                if not quality:
                    continue
                key = f"{backend.value}:{mode.value}:{n_sel}"
                per_group[key] = {
                    "routed_held_out_quality": _mean_se(quality),
                    "mean_deployed_flops": _mean_se(flops),
                    "mean_routed_width": _mean_se(width),
                    "oracle_agreement": _mean_se(agreement) if agreement else None,
                }

    n_expected = len(Backend) * len(Mode) * len(N_SEL_VALUES) * len(SEEDS)
    out = {
        "per_group": per_group,
        "backends": [b.value for b in Backend],
        "modes": [m.value for m in Mode],
        "n_sel_values": list(N_SEL_VALUES),
        "seeds": list(SEEDS),
        "n_cells_present": n_present,
        "n_cells_expected": n_expected,
        "provenance": run_provenance(),
    }
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "frozen.json")
    with open(path, "w") as f:
        json.dump(_jsonable(out), f, indent=2)
    print(f"wrote {path}")
    print(f"n_cells_present={n_present}/{n_expected}  n_groups={len(per_group)}/{len(Backend) * len(Mode) * len(N_SEL_VALUES)}")


# ---------------------------------------------------------------------------
# --selftest
# ---------------------------------------------------------------------------

_SELFTEST_KW = {"w_max": 3, "n_train": 60, "n_test": 30}
_SELFTEST_MF_D = 2  # the multi-feature selftest's own tiny d (both geometries) -- distinct from the d in {1, 8} identity-only checks.

_SELFTEST_REQUIRED_KEYS = (
    "backend",
    "mode",
    "n_sel",
    "seed",
    "toy",
    "n_train",
    "n_test",
    "w_max",
    "n_report",
    "backend_config",
    "sizing_rule",
    "table_provenance",
    "routed_held_out_quality",
    "mean_deployed_flops",
    "oracle_agreement",
    "mean_routed_width",
    "width_distribution",
    "provenance",
)


def run_selftest() -> bool:
    """Tiny wiring check (<60s), entirely in a temp dir: all 4 backends x both modes, cache reuse, --summarize."""
    ok = True
    tmp_dir = tempfile.mkdtemp(prefix="wsel19_selftest_")
    try:
        seed = SEEDS[0]
        n_sel = N_SEL_VALUES[0]  # real closed-set values throughout -- only the 12-net sweep's OWN
        # toy size shrinks (_SELFTEST_KW); the selection pool is a cheap fresh draw regardless of
        # size, and --summarize's own iteration below expects these exact real identifiers.
        cache_path = _sweep_cache_json_path(tmp_dir, seed)

        cases: dict[tuple[Backend, Mode], dict[str, Any]] = {}
        for backend in Backend:
            for mode in Mode:
                case = run_cell(backend, mode, n_sel, seed, results_dir=tmp_dir, **_SELFTEST_KW)
                cases[(backend, mode)] = case
                with open(_cell_json_path(tmp_dir, backend, mode, n_sel, seed), "w") as f:
                    json.dump(_jsonable(case), f, indent=2)

                keys_ok = all(k in case for k in _SELFTEST_REQUIRED_KEYS)
                roundtrip = json.loads(json.dumps(_jsonable(case)))
                roundtrip_ok = all(k in roundtrip for k in _SELFTEST_REQUIRED_KEYS)
                mode_shape_ok = (case["oracle_agreement"] is None) == (mode is Mode.BLEND) and (case["width_distribution"] is None) == (mode is Mode.BLEND)
                sizing_rule_ok = (case["sizing_rule"] is not None) == (backend is Backend.RULE_MLP)
                finite_ok = math.isfinite(case["routed_held_out_quality"]) and math.isfinite(case["mean_deployed_flops"]) and math.isfinite(case["mean_routed_width"])
                cell_ok = keys_ok and roundtrip_ok and mode_shape_ok and sizing_rule_ok and finite_ok
                ok = ok and cell_ok
                print(
                    f"[wsel19 selftest] backend={backend.value} mode={mode.value} "
                    f"quality={case['routed_held_out_quality']:.4g} flops={case['mean_deployed_flops']:.3g}  {'PASS' if cell_ok else 'FAIL'}"
                )

        # Sweep cache built once, shared by every backend/mode above (not retrained per fit).
        mtime_after_all = os.path.getmtime(cache_path)
        n_sel_2 = N_SEL_VALUES[1]
        case_again = run_cell(Backend.FROZEN_MLP, Mode.HARD, n_sel_2, seed, results_dir=tmp_dir, **_SELFTEST_KW)
        mtime_after_repeat = os.path.getmtime(cache_path)
        cache_reused = mtime_after_all == mtime_after_repeat
        print(f"[wsel19 selftest] sweep cache built once, reused across backends/modes/n_sel: {cache_reused}  {'PASS' if cache_reused else 'FAIL'}")
        ok = ok and cache_reused

        # A tiny-toy selftest cannot reuse WSEL6's real (n_train=1500) cache -- confirms the guard fires.
        fresh_toy_ok = not case_again["table_provenance"]["reused_from_wsel6"]
        print(f"[wsel19 selftest] tiny-toy cell correctly falls back to local training (never touches WSEL6/): {fresh_toy_ok}  {'PASS' if fresh_toy_ok else 'FAIL'}")
        ok = ok and fresh_toy_ok

        with open(_cell_json_path(tmp_dir, Backend.FROZEN_MLP, Mode.HARD, n_sel_2, seed), "w") as f:
            json.dump(_jsonable(case_again), f, indent=2)

        summarize(results_dir=tmp_dir)
        with open(os.path.join(tmp_dir, "frozen.json")) as f:
            frozen = json.load(f)
        summarize_ok = "per_group" in frozen and frozen["n_cells_present"] >= len(Backend) * len(Mode)
        print(f"[wsel19 selftest] summarize: n_cells_present={frozen.get('n_cells_present')}  {'PASS' if summarize_ok else 'FAIL'}")
        ok = ok and summarize_ok

        # --- Multi-feature extension: identity at d in {1, 8} (both geometries), the calibration
        # gate REFUSING before the artifact exists, a tiny --calibrate run, and a tiny end-to-end
        # multi-feature cell at d=2, both geometries (reduced sizes throughout). ---
        for d_check in (1, 8):
            for geometry in wt.Geometry:
                identity_ok = wt.identity_holds(200, seed, d_check, geometry)
                ok = ok and identity_ok
                print(f"[wsel19 selftest] mf identity d={d_check} geometry={geometry.value}  {'PASS' if identity_ok else 'FAIL'}")

        try:
            _load_calibration_or_refuse(tmp_dir)
            gate_refused = False
        except RuntimeError:
            gate_refused = True
        print(f"[wsel19 selftest] mf calibration gate refuses before artifact exists: {gate_refused}  {'PASS' if gate_refused else 'FAIL'}")
        ok = ok and gate_refused

        calibration = run_calibration(w_max=3, seeds=(seed,), n_train=40, results_dir=tmp_dir)
        calibration_ok = "passed" in calibration and "anchor_ratio_to_noise_floor" in calibration and math.isfinite(calibration["anchor_ratio_to_noise_floor"])
        anchor = calibration.get("anchor_ratio_to_noise_floor")
        print(f"[wsel19 selftest] mf calibration: passed={calibration.get('passed')} anchor={anchor:.4g}  {'PASS' if calibration_ok else 'FAIL'}")
        ok = ok and calibration_ok

        # A w_max=3/n_train=40 toy is too tiny to reliably reproduce real regime structure (that is
        # a SCIENCE property of the real d=1 grid, checked at real scale by the root, not a wiring
        # property) -- overwrite with a synthetic PASSING fixture so the downstream gate-consumption
        # wiring (_load_calibration_or_refuse -> run_cell_multifeature) is exercised deterministically.
        with open(_calibration_path(tmp_dir), "w") as f:
            json.dump({**calibration, "passed": True, "regime_visible_all_seeds": True, "all_trustworthy": True}, f)

        mf_d = _SELFTEST_MF_D
        mf_n_train = 40  # tiny -- keeps the wiring check fast; the real 1500/4000 sizes are the root's grid, not this check.
        for geometry in wt.Geometry:
            mf_case = run_cell_multifeature(Backend.FROZEN_MLP, Mode.HARD, n_sel=20, seed=seed, d=mf_d, geometry=geometry, w_max=3, n_train=mf_n_train, results_dir=tmp_dir)
            mf_keys_ok = "validity_checks" in mf_case and mf_case["d"] == mf_d and mf_case["geometry"] == geometry.value and mf_case["n_train"] == mf_n_train
            mf_finite_ok = math.isfinite(mf_case["routed_held_out_quality"])
            mf_ok = mf_keys_ok and mf_finite_ok
            ok = ok and mf_ok
            print(
                f"[wsel19 selftest] mf cell d={mf_d} geometry={geometry.value} quality={mf_case['routed_held_out_quality']:.4g} "
                f"fit_status={mf_case['validity_checks']['fit_status']} routing_status={mf_case['validity_checks']['routing_status']}  {'PASS' if mf_ok else 'FAIL'}"
            )

        # n_train_fallback threads through and is recorded -- `fallback_n_train` overrides the real
        # §3b F6 value (4000, exercised at real scale by the root) with a tiny size for this check.
        mf_fallback_case = run_cell_multifeature(
            Backend.CONSTANT, Mode.HARD, n_sel=20, seed=seed, d=mf_d, geometry=wt.Geometry.AXIS, w_max=3, results_dir=tmp_dir, n_train_fallback=True, fallback_n_train=mf_n_train
        )
        fallback_ok = mf_fallback_case["n_train_fallback_applied"] is True and mf_fallback_case["n_train"] == mf_n_train
        print(f"[wsel19 selftest] mf n-train-fallback recorded: n_train={mf_fallback_case['n_train']}  {'PASS' if fallback_ok else 'FAIL'}")
        ok = ok and fallback_ok
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"[wsel19 selftest] {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """Parses args and dispatches to a mode flag or one real 1-D or multi-feature cell.

    `--selftest` / `--summarize` / `--calibrate`, or one real 1-D or multi-feature (`--dim`/
    `--geometry`) cell.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true", help="Tiny wiring check in a temp dir, then exit.")
    parser.add_argument("--summarize", action="store_true", help="Aggregate every 1-D per-cell JSON on disk into WSEL19/frozen.json.")
    parser.add_argument("--calibrate", action="store_true", help="Run the §5.2 d=1 calibration cell, write the gating artifact, and exit.")
    parser.add_argument("--backend", choices=[b.value for b in Backend], default=None)
    parser.add_argument("--mode", choices=[m.value for m in Mode], default=None)
    parser.add_argument("--n-sel", type=int, choices=list(N_SEL_VALUES), default=None)
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for this cell (canonical suite: 0, 1, 2).")
    parser.add_argument(
        "--dim", type=int, choices=list(_MF_DIM_CHOICES), default=None, help="Multi-feature input dimension. Requires --geometry; both omitted => the 1-D slice."
    )
    parser.add_argument("--geometry", choices=[g.value for g in wt.Geometry], default=None, help="Multi-feature geometry of v. Requires --dim.")
    parser.add_argument(
        "--n-train-fallback",
        action="store_true",
        help="§3b F6 pre-authorized fallback: raise this cell's n_train to 4000 (only meaningful after a VOID_FOR_FIT verdict at the canonical 1500). Requires --dim/--geometry.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Suffix for this cell's OWN JSON filename only (never any shared cache) -- for a throwaway verification cell.",
    )
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR)
    args = parser.parse_args()

    if sum([args.selftest, args.summarize, args.calibrate]) > 1:
        parser.error("--selftest, --summarize and --calibrate are mutually exclusive.")
    if args.selftest:
        sys.exit(0 if run_selftest() else 1)
    if args.summarize:
        summarize(results_dir=args.results_dir)
        return
    if args.calibrate:
        run_calibration(results_dir=args.results_dir)
        return

    if (args.dim is None) != (args.geometry is None):
        parser.error("--dim and --geometry must be given together (or both omitted for the existing 1-D slice).")
    if args.n_train_fallback and args.dim is None:
        parser.error("--n-train-fallback only applies to a multi-feature cell (--dim/--geometry).")
    if args.backend is None or args.mode is None or args.n_sel is None or args.seed is None:
        parser.error("--backend, --mode, --n-sel and --seed are all required for a real cell (or pass --selftest / --summarize / --calibrate).")

    backend = Backend(args.backend)
    mode = Mode(args.mode)
    os.makedirs(args.results_dir, exist_ok=True)

    if args.dim is not None:
        geometry = wt.Geometry(args.geometry)
        print(f"[wsel19 mf] backend={backend.value} mode={mode.value} n_sel={args.n_sel} seed={args.seed} d={args.dim} geometry={geometry.value}", flush=True)
        case = run_cell_multifeature(backend, mode, args.n_sel, args.seed, args.dim, geometry, results_dir=args.results_dir, n_train_fallback=args.n_train_fallback)
        cell_path = _mf_cell_json_path(args.results_dir, args.dim, geometry, backend, mode, args.n_sel, args.seed, tag=args.tag)
    else:
        print(f"[wsel19] backend={backend.value} mode={mode.value} n_sel={args.n_sel} seed={args.seed}", flush=True)
        case = run_cell(backend, mode, args.n_sel, args.seed, results_dir=args.results_dir)
        cell_path = _cell_json_path(args.results_dir, backend, mode, args.n_sel, args.seed, tag=args.tag)

    if not case["table_provenance"]["sweep_trustworthy"]:
        print(
            f"*** DO-NOT-CONCLUDE GUARD: backend={backend.value} mode={mode.value} n_sel={args.n_sel} seed={args.seed} "
            f"one or more per-width sweep nets did NOT converge trustworthily (sweep_hit_cap="
            f"{case['table_provenance']['sweep_hit_cap']}). ***"
        )

    with open(cell_path, "w") as f:
        json.dump(_jsonable(case), f, indent=2)
    print(
        f"wrote {cell_path}  routed_held_out_quality={case['routed_held_out_quality']:.5f}  "
        f"mean_deployed_flops={case['mean_deployed_flops']:.3g}  oracle_agreement={case['oracle_agreement']}"
    )


if __name__ == "__main__":
    main()
