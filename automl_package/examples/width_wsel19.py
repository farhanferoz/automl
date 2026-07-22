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
tolerance sweep (MASTER Decision 18); no multi-feature cells (gated on the toy-design spec's GO,
`shared/wsel19-toy-design.md`).

Driver CLI contract (root-run grid; this file is never run over the full grid by its author):
  `--backend {frozen_mlp,rule_mlp,xgboost,constant} --mode {hard,blend} --n-sel {75,300,1200}
  --seed <int>` runs ONE cell, writing its per-cell JSON immediately (and, on a cache miss for that
  seed's sweep, the shared per-width error tables).
  `--summarize` aggregates every per-cell JSON on disk into `WSEL19/frozen.json`.
  `--selftest` runs a tiny end-to-end check in a temp dir (small `w_max`, tiny selection pool, all
  four backends x both modes) and asserts the sweep cache is built once and reused -- no real cell
  is ever run here, and nothing is written under this task's real results directory.
  `--tag` appends a suffix to a cell's own JSON filename (never to the shared sweep cache) --
  intended for a throwaway verification cell that gets deleted afterward without touching any
  shared artifact a later real cell would want to reuse.

Usage:
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel19.py --selftest
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel19.py --backend frozen_mlp --mode hard --n-sel 300 --seed 0
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel19.py --backend xgboost --mode blend --n-sel 75 --seed 1
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/width_wsel19.py --summarize
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

from automl_package.models.flexnn.routing import (  # noqa: E402
    DEFAULT_HIDDEN,
    DEFAULT_LR,
    DEFAULT_N_EPOCHS,
    DEFAULT_TOLERANCE,
    DistilledCapacityRouter,
    _CapacityRouterMLP,
    _cheapest_within_tolerance_labels,
)
from automl_package.utils.capacity_accounting import executed_flops  # noqa: E402
from automl_package.utils.run_provenance import run_provenance  # noqa: E402

RESULTS_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "WSEL19")
_WSEL6_RESULTS_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "WSEL6")

W_MAX = w6.W_MAX  # 12
SEEDS = w6.SEEDS  # (0, 1, 2)
N_SEL_VALUES = (75, 300, 1200)  # toy-design spec SS3: 75 = starved regime, 1200 = unstarved contrast.

_SELECTION_POOL_SEED_OFFSET = 1000  # the third disjoint make_hetero draw this task needs -- see module docstring.
_REPORT_SEED_OFFSET = 500  # this strand's universal disjoint-test-draw convention (module docstring).

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
        x_arr = x.reshape(-1, 1).astype(np.float32)
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
    x_arr = x_sel.reshape(-1, 1).astype(np.float32)
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
        x_arr2 = x.reshape(-1, 1).astype(np.float32)
        with torch.no_grad():
            logits = net(torch.as_tensor(x_arr2, dtype=torch.float32))
        return logits.argmax(dim=1).cpu().numpy()

    def route_proba(x: np.ndarray) -> np.ndarray:
        x_arr2 = x.reshape(-1, 1).astype(np.float32)
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
    x_arr = x_sel.reshape(-1, 1).astype(np.float32)
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
        dense_idx = clf.predict(x.reshape(-1, 1).astype(np.float32))
        return train_classes[dense_idx]

    def route_proba(x: np.ndarray) -> np.ndarray:
        proba_dense = clf.predict_proba(x.reshape(-1, 1).astype(np.float32))  # (N, len(train_classes)), column i == dense class i.
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
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"[wsel19 selftest] {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """Parses args and dispatches to `--selftest` / `--summarize` / one real `--backend`/`--mode`/`--n-sel`/`--seed` cell."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true", help="Tiny wiring check in a temp dir, then exit.")
    parser.add_argument("--summarize", action="store_true", help="Aggregate every per-cell JSON on disk into WSEL19/frozen.json.")
    parser.add_argument("--backend", choices=[b.value for b in Backend], default=None)
    parser.add_argument("--mode", choices=[m.value for m in Mode], default=None)
    parser.add_argument("--n-sel", type=int, choices=list(N_SEL_VALUES), default=None)
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for this cell (canonical suite: 0, 1, 2).")
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Suffix for this cell's OWN JSON filename only (never the shared sweep cache) -- for a throwaway verification cell.",
    )
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR)
    args = parser.parse_args()

    if args.selftest and args.summarize:
        parser.error("--selftest and --summarize are mutually exclusive.")
    if args.selftest:
        sys.exit(0 if run_selftest() else 1)
    if args.summarize:
        summarize(results_dir=args.results_dir)
        return
    if args.backend is None or args.mode is None or args.n_sel is None or args.seed is None:
        parser.error("--backend, --mode, --n-sel and --seed are all required for a real cell (or pass --selftest / --summarize).")

    backend = Backend(args.backend)
    mode = Mode(args.mode)
    os.makedirs(args.results_dir, exist_ok=True)
    print(f"[wsel19] backend={backend.value} mode={mode.value} n_sel={args.n_sel} seed={args.seed}", flush=True)
    case = run_cell(backend, mode, args.n_sel, args.seed, results_dir=args.results_dir)

    if not case["table_provenance"]["sweep_trustworthy"]:
        print(
            f"*** DO-NOT-CONCLUDE GUARD: backend={backend.value} mode={mode.value} n_sel={args.n_sel} seed={args.seed} "
            f"one or more per-width sweep nets did NOT converge trustworthily (sweep_hit_cap="
            f"{case['table_provenance']['sweep_hit_cap']}). ***"
        )

    cell_path = _cell_json_path(args.results_dir, backend, mode, args.n_sel, args.seed, tag=args.tag)
    with open(cell_path, "w") as f:
        json.dump(_jsonable(case), f, indent=2)
    print(
        f"wrote {cell_path}  routed_held_out_quality={case['routed_held_out_quality']:.5f}  "
        f"mean_deployed_flops={case['mean_deployed_flops']:.3g}  oracle_agreement={case['oracle_agreement']}"
    )


if __name__ == "__main__":
    main()
