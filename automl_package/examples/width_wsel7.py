"""WSEL-7 — is the router's architecture right for width? (`docs/plans/capacity_programme/width.md` ~914-947).

The `DistilledCapacityRouter` (`automl_package/models/flexnn/routing.py`) is used FROZEN and untuned
everywhere else in this strand: two hidden layers of 32 units (`DEFAULT_HIDDEN = (32, 32)`), 300
full-batch Adam epochs (`DEFAULT_N_EPOCHS`), lr 1e-2 (`DEFAULT_LR`) -- `routing.py:57-60`. The only
existing evidence for that choice is one dimension, two settings, one toy, 3 seeds
(`width-cert.md:234,237`). This driver extends it to an OFAT search over router hidden width,
depth (layer count), epoch count and learning rate -- one dimension swept at a time, everything else
held at the frozen default -- to establish whether width's routing conclusions are invariant to the
router's own architecture, and if not, what the router needs.

**Struck, 2026-07-21 (MASTER Decision 18):** the labelling tolerance (`DEFAULT_TOLERANCE = 0.25`) is
NOT swept here -- its sensitivity sweep is ruled "not scheduled" and this task's own `frozen.json`
does not own that constant. Only router hidden/depth/epochs/lr are swept and owned.

**Root-assigned suite** (`width.md` SS3.8's root-assignment rule; this task's own row is not listed
in SS3.8's table, so the root assigned it directly): TIER 1 (`hetero`, `n_train=1500`, `n_test=500`,
`sigma=0.05`) AND TIER 2 (`hetero3`, `n_train=2250`, `n_test=750`, the fixed-sigma WEIGHTED objective
for any net training) -- seeds 0/1/2, `w_max=12`, the certified multi-head `SharedTrunkPerWidthHeadNet`
ONLY (no other architecture enters this task). Tier 2 mirrors `width_wsel13.py`'s own tier-2 pattern
exactly (imported, not reimplemented): `WidthSchedule.ALL` (every width, every step) trained on
`width_candidates.weighted_squared_error` via `width_wsel13._train_all_schedule_weighted_to_convergence`,
while tier 1 keeps the certified SANDWICH schedule + plain MSE through
`kdropout_converged_width_experiment._train_kdropout_to_convergence` (SS3.7: sigma is FIXED at the
generator's true value on every tier, never learned; the two tiers only differ in HOW that is done).

**Caching design (the load-bearing efficiency requirement).** Training the multi-head net and scoring
its per-width held-out squared error is the expensive part of a cell, and it depends on `(tier, seed)`
ONLY -- never on the router variant under test. `_get_or_build_cache` trains and scores it ONCE per
`(tier, seed)`, then caches the trained `state_dict` (`.pt`) and BOTH per-width held-out error tables
(FIT split for router labelling, REPORT split for scoring routed quality; `.json`, alongside `x`/`y`
for each split, FLOPs-by-width, and the training provenance) to disk under this task's results dir.
Every `(dimension, value)` cell for that `(tier, seed)` then fits its router purely against the cached
tables -- no forward pass through the net, let alone a re-train. A cache HIT is a plain
`os.path.exists` check; the grid never retrains the same `(tier, seed)` twice.

**Router API note (a REUSE finding, not a gap needing a patch).** `FlexibleWidthNN.fit_router`
(`automl_package/models/flexnn/width/model.py:358-403`) hardcodes `DistilledCapacityRouter(device=
self.device)` with no hidden/epochs/lr passthrough -- it cannot express this task's swept dimensions.
The class it delegates to, `DistilledCapacityRouter` itself, DOES expose `hidden`/`n_epochs`/`lr` as
constructor arguments (`routing.py:149-166`), so this driver constructs `DistilledCapacityRouter`
directly with the swept CONSTRUCTION arguments and calls its unmodified `.fit()`, mirroring
`fit_router`'s own internal `eval_fn`/`cost_fn` wiring (squared error per width, `executed_flops` per
width) exactly as `width_wsel16._distilled_router_selected_width` already does for the same reason.
Nothing in `routing.py` or `distilled_router.py`-equivalent modules is edited -- varying construction
arguments of the shared class it delegates to is reuse, not a reimplementation, and this task's write
set excludes those files regardless (`width.md` SS3.5's WSEL-7 branch: "NEVER write
`automl_package/models/common/distilled_router.py` from this strand"). The layers dimension (1 vs 3)
needs no special-casing either: `hidden` is already an arbitrary-length tuple, so depth is just
`len(hidden)`.

**Per-cell selection criterion (a chosen default).** Every router variant is scored by the SAME metric
a caller of `route()` would see: mean plain squared error on the REPORT split at each sample's ROUTED
width (never the training objective -- `width_wsel16._distilled_router_selected_width` and
`_global_selected_width` both use plain squared error as the selection/routing criterion even at tier
2, and this driver follows that same established convention so the two tasks' numbers sit on the same
footing). `quality_ratio_to_default = variant_quality / frozen_default_quality`, computed inside EVERY
cell against a same-seed default-config router fit on the identical cached tables, is the dimensionless
number `--summarize`'s invariance/plateau read is built on (SS3.5's WSEL-7 branch).

**Invariance / plateau read (`--summarize`, a chosen algorithm -- retune `_PLATEAU_REL_TOL` if the
user wants a tighter or looser bar).** Per dimension, `ratio_to_default_by_value[v]` is the mean
`quality_ratio_to_default` over every present `(tier, seed)` cell at that value (scale-free, so tier
1/tier 2 and different-seed cells pool directly; the default's own value contributes ratio 1.0 by
construction). The dimension is INVARIANT iff the frozen default's own ratio is within
`_PLATEAU_REL_TOL` (5%, a chosen default) of the best ratio achieved anywhere in the sweep -- i.e. no
other swept value beats the default by more than the tolerance. If NOT invariant, `plateau_value` is
the smallest value (ascending capacity order for hidden/layers/epochs; best-ratio order for lr, which
has no capacity ordering) whose ratio is within tolerance of the best -- SS3.5's "smallest
configuration that reaches the plateau." The strand-level `invariant` flag is the AND over all four
dimensions; `new_default` (non-null iff `invariant` is false) combines each non-invariant dimension's
own `plateau_value` with the frozen default's value on every dimension that stayed invariant.

**Non-goals:** no per-dataset tuning of the router, ever; no change to the labelling rule or its
tolerance; no edits to `routing.py` / any shared router module; no single-head (running-sum prefix)
architecture; no re-derivation of the multi-head training loop already owned by
`kdropout_converged_width_experiment.py` / `width_wsel13.py`.

Driver CLI contract (root-run grid; this file is never run over the full grid by its author):
  `--tier {1,2} --seed <int> --dimension {hidden,layers,epochs,lr} --value <num>` runs ONE cell,
  writing its per-cell JSON immediately (and, on a cache miss for that `(tier, seed)`, the shared
  multi-head cache files).
  `--summarize` aggregates every per-cell JSON on disk into `WSEL7/frozen.json`.
  `--selftest` runs a tiny end-to-end check in a temp dir (small `w_max`, tiny table, a couple of
  router variants) and asserts the cache is built once and reused -- no real cell is ever run here,
  and nothing is written under this task's real results directory.

Usage:
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel7.py --selftest
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel7.py --tier 1 --seed 0 --dimension hidden --value 64
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel7.py --tier 2 --seed 0 --dimension lr --value 0.03
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel7.py --summarize
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
import time
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import train_test_split

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root

import converged_width_experiment as cwe  # noqa: E402
import kdropout_converged_width_experiment as kce  # noqa: E402
import nested_width_net as nwn  # noqa: E402
import width_wsel13 as w13  # noqa: E402

from automl_package.models.flexnn.routing import DEFAULT_HIDDEN, DEFAULT_LR, DEFAULT_N_EPOCHS, DEFAULT_TOLERANCE, DistilledCapacityRouter  # noqa: E402
from automl_package.utils.capacity_accounting import executed_flops  # noqa: E402
from automl_package.utils.run_provenance import run_provenance  # noqa: E402

RESULTS_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "WSEL7")

# Certified configuration, reused verbatim (this task does not retune the multi-head net itself).
W_MAX = cwe.W_MAX  # 12
SEEDS = cwe.SEEDS  # (0, 1, 2)
DEFAULT_MAX_EPOCHS = kce.DEFAULT_MAX_EPOCHS  # multi-head net's convergence-gate cap, NOT a router hyperparameter.
DEFAULT_CHECK_EVERY = w13.DEFAULT_CHECK_EVERY
DEFAULT_PATIENCE = w13.DEFAULT_PATIENCE
DEFAULT_MIN_DELTA = w13.DEFAULT_MIN_DELTA

_TIER_CONFIG = w13._TIER_CONFIG  # identical SS3.8 tier config w13/w16 already use; not re-derived.

# Half of the driver's own held-out test set fits the router (and builds its labelling error table);
# the other half REPORTS routed quality and width distribution, untouched by anything upstream of it.
_ROUTER_REPORT_FRACTION = 0.5

# Chosen default (retune for a tighter/looser bar): a swept value "reaches the plateau" once its
# ratio-to-best is within this fraction.
_PLATEAU_REL_TOL = 0.05


class Tier(enum.IntEnum):
    """SS3.8's canonical toy suite tiers this task runs (root-assigned: both)."""

    ONE = 1  # the reference cell -- hetero, sandwich schedule, plain MSE.
    TWO = 2  # the noisy-easy control -- hetero3, ALL schedule, fixed-sigma weighted squared error.


class Dimension(enum.Enum):
    """The four router-architecture axes this task owns (SS3.6). The labelling tolerance is struck."""

    HIDDEN = "hidden"  # per-layer width, depth held at the frozen default (2 layers).
    LAYERS = "layers"  # layer count, per-layer width held at the frozen default (32).
    EPOCHS = "epochs"  # router training epochs.
    LR = "lr"  # router Adam learning rate.


# OFAT sweep values -- each dimension varies alone, everything else at DEFAULT_HIDDEN/DEFAULT_N_EPOCHS/
# DEFAULT_LR (routing.py:57-60). Every tuple includes its own dimension's frozen-default value, so the
# center point of the design is itself one of the swept cells (a built-in self-consistency check: see
# `run_selftest`'s default-agrees-with-itself assertion).
_HIDDEN_VALUES: tuple[int, ...] = (16, 32, 64, 128)
_LAYERS_VALUES: tuple[int, ...] = (1, 2, 3)
_EPOCHS_VALUES: tuple[int, ...] = (150, 300, 600)
_LR_VALUES: tuple[float, ...] = (3e-3, 1e-2, 3e-2)

_DIMENSION_VALUES: dict[Dimension, tuple] = {
    Dimension.HIDDEN: _HIDDEN_VALUES,
    Dimension.LAYERS: _LAYERS_VALUES,
    Dimension.EPOCHS: _EPOCHS_VALUES,
    Dimension.LR: _LR_VALUES,
}


def _frozen_default_value_for(dimension: Dimension) -> float:
    """The current frozen default's OWN value along `dimension` (routing.py:57-60)."""
    return {
        Dimension.HIDDEN: DEFAULT_HIDDEN[0],
        Dimension.LAYERS: len(DEFAULT_HIDDEN),
        Dimension.EPOCHS: DEFAULT_N_EPOCHS,
        Dimension.LR: DEFAULT_LR,
    }[dimension]


def _router_config_for(dimension: Dimension, value: float) -> tuple[tuple[int, ...], int, float]:
    """OFAT construction args for one `(dimension, value)` cell: `dimension` varies, the rest stay frozen."""
    hidden, n_epochs, lr = DEFAULT_HIDDEN, DEFAULT_N_EPOCHS, DEFAULT_LR
    if dimension is Dimension.HIDDEN:
        hidden = (int(value),) * len(DEFAULT_HIDDEN)
    elif dimension is Dimension.LAYERS:
        hidden = (DEFAULT_HIDDEN[0],) * int(value)
    elif dimension is Dimension.EPOCHS:
        n_epochs = int(value)
    else:
        lr = float(value)
    return hidden, n_epochs, lr


def _jsonable(obj: object) -> object:
    """Minimal JSON-safe coercion (numpy/torch scalars, non-str dict keys) -- local twin of `width_wsel13._jsonable`."""
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
# The expensive part -- trained ONCE per (tier, seed), cached to disk, read by every router variant.
# ---------------------------------------------------------------------------


@dataclass
class _MultiHeadCache:
    """Everything a router variant needs from one `(tier, seed)`'s trained multi-head net -- no net object required."""

    error_table_fit: np.ndarray  # (n_fit, w_max) plain squared error per width -- the router's labelling table.
    x_fit: np.ndarray  # (n_fit,) standardized inputs, the router's own training features.
    y_fit: np.ndarray  # (n_fit,) standardized targets -- only used by DistilledCapacityRouter.fit for a length check.
    error_table_report: np.ndarray  # (n_report, w_max) plain squared error per width -- scores routed quality.
    x_report: np.ndarray  # (n_report,) standardized inputs the router actually routes for reporting.
    y_report: np.ndarray  # (n_report,) standardized targets, kept for provenance/round-trip completeness.
    flops_by_width: list[float]  # len w_max, executed_flops(net, k) -- the router's cost_fn table.
    hit_cap: bool  # any width's convergence tracker hit the epoch cap (per-cell "trustworthy" guard).
    convergence: dict  # {width: ConvergenceResult.summary()} for the multi-head net's own training.
    steps_to_converge: int
    train_wall_clock_s: float
    training_schedule: str
    objective: str


def _cache_meta_path(results_dir: str, tier: Tier, seed: int, w_max: int) -> str:
    return os.path.join(results_dir, f"cache_tier{tier.value}_seed{seed}_wmax{w_max}.json")


def _cache_state_path(results_dir: str, tier: Tier, seed: int, w_max: int) -> str:
    return os.path.join(results_dir, f"cache_tier{tier.value}_seed{seed}_wmax{w_max}.pt")


def _get_or_build_cache(
    results_dir: str,
    tier: Tier,
    seed: int,
    *,
    w_max: int = W_MAX,
    n_train: int | None = None,
    n_test: int | None = None,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    check_every: int = DEFAULT_CHECK_EVERY,
    patience: int = DEFAULT_PATIENCE,
    min_delta: float = DEFAULT_MIN_DELTA,
) -> _MultiHeadCache:
    """Loads `(tier, seed)`'s cached multi-head net + error tables, training and caching them on a miss.

    Data prep mirrors `width_wsel13.run_cell` verbatim (same phase-1 train/val carve, same
    standardization) -- reimplemented here rather than calling it because this task needs the trained
    net object directly to build its own FIT/REPORT error tables, which `w13.run_cell` does not return.
    """
    meta_path = _cache_meta_path(results_dir, tier, seed, w_max)
    state_path = _cache_state_path(results_dir, tier, seed, w_max)
    if os.path.exists(meta_path) and os.path.exists(state_path):
        with open(meta_path) as f:
            meta = json.load(f)
        return _MultiHeadCache(
            error_table_fit=np.array(meta["error_table_fit"]),
            x_fit=np.array(meta["x_fit"]),
            y_fit=np.array(meta["y_fit"]),
            error_table_report=np.array(meta["error_table_report"]),
            x_report=np.array(meta["x_report"]),
            y_report=np.array(meta["y_report"]),
            flops_by_width=meta["flops_by_width"],
            hit_cap=meta["hit_cap"],
            convergence=meta["convergence"],
            steps_to_converge=meta["steps_to_converge"],
            train_wall_clock_s=meta["train_wall_clock_s"],
            training_schedule=meta["training_schedule"],
            objective=meta["objective"],
        )

    cfg = _TIER_CONFIG[tier]
    n_tr = n_train if n_train is not None else cfg.n_train
    n_te = n_test if n_test is not None else cfg.n_test

    if cfg.toy is nwn.Toy.HETERO3:
        x_tr, y_tr, region_tr = nwn.make_hetero3(n_tr, seed)
        x_te, y_te, _reg_te = nwn.make_hetero3(n_te, seed + 500)
    else:
        x_tr, y_tr, region_tr = nwn.make_hetero(n_tr, seed, sigma=cfg.sigma)
        x_te, y_te, _reg_te = nwn.make_hetero(n_te, seed + 500, sigma=cfg.sigma)

    p1_idx = np.arange(0, n_tr, 2)
    x_p1, y_p1, region_p1 = x_tr[p1_idx], y_tr[p1_idx], region_tr[p1_idx]
    val_mask = (np.arange(len(x_p1)) % cwe.VAL_EVERY) == 0
    norm = cwe._standardize_fit(x_p1[~val_mask], y_p1[~val_mask])
    x_tr_t, y_tr_t = cwe._to_std_tensors(x_p1[~val_mask], y_p1[~val_mask], norm)
    x_val_t, y_val_t = cwe._to_std_tensors(x_p1[val_mask], y_p1[val_mask], norm)

    torch.manual_seed(seed)
    net = nwn.SharedTrunkPerWidthHeadNet(w_max=w_max)

    t0 = time.perf_counter()
    if tier is Tier.TWO:
        training_schedule = nwn.WidthSchedule.ALL.value
        objective = "weighted_squared_error"
        sigma_tr = w13._sigma_true_tensor(cfg.toy, region_p1[~val_mask], cfg.sigma, norm)
        sigma_val = w13._sigma_true_tensor(cfg.toy, region_p1[val_mask], cfg.sigma, norm)
        conv = w13._train_all_schedule_weighted_to_convergence(
            net, x_tr_t, y_tr_t, x_val_t, y_val_t, sigma_tr, sigma_val, w_max=w_max, max_epochs=max_epochs, check_every=check_every, patience=patience, min_delta=min_delta
        )
    else:
        training_schedule = nwn.WidthSchedule.SANDWICH.value
        objective = kce.LossType.MSE.value
        conv, _best_epoch = kce._train_kdropout_to_convergence(
            net,
            x_tr_t,
            y_tr_t,
            x_val_t,
            y_val_t,
            arch=kce.Arch.SHARED_TRUNK,
            loss=kce.LossType.MSE,
            max_epochs=max_epochs,
            check_every=check_every,
            patience=patience,
            min_delta=min_delta,
            seed=seed,
            schedule=nwn.WidthSchedule.SANDWICH,
        )
    train_wall_clock_s = time.perf_counter() - t0
    hit_cap = any(r.hit_cap for r in conv.values())
    steps_to_converge = max(r.stop_epoch for r in conv.values())

    # FIT / REPORT split of the driver's OWN held-out test set (never touched by gradient training) --
    # FIT builds the router's labelling table and supplies its training features, REPORT scores the
    # routed system untouched by either training or router-fitting.
    x_fit_raw, x_report_raw, y_fit_raw, y_report_raw = train_test_split(x_te, y_te, test_size=_ROUTER_REPORT_FRACTION, random_state=seed)
    x_fit_t, y_fit_t = cwe._to_std_tensors(x_fit_raw, y_fit_raw, norm)
    x_report_t, y_report_t = cwe._to_std_tensors(x_report_raw, y_report_raw, norm)

    net.eval()
    with torch.no_grad():
        error_table_fit = np.stack([((net.forward_width(x_fit_t, k)[0].squeeze(1) - y_fit_t) ** 2).numpy() for k in range(1, w_max + 1)], axis=1)
        error_table_report = np.stack([((net.forward_width(x_report_t, k)[0].squeeze(1) - y_report_t) ** 2).numpy() for k in range(1, w_max + 1)], axis=1)
    flops_by_width = [float(executed_flops(net, k)) for k in range(1, w_max + 1)]

    cache = _MultiHeadCache(
        error_table_fit=error_table_fit,
        x_fit=x_fit_t.numpy(),
        y_fit=y_fit_t.numpy(),
        error_table_report=error_table_report,
        x_report=x_report_t.numpy(),
        y_report=y_report_t.numpy(),
        flops_by_width=flops_by_width,
        hit_cap=hit_cap,
        convergence={k: r.summary() for k, r in conv.items()},
        steps_to_converge=steps_to_converge,
        train_wall_clock_s=train_wall_clock_s,
        training_schedule=training_schedule,
        objective=objective,
    )

    os.makedirs(results_dir, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(
            _jsonable(
                {
                    "error_table_fit": cache.error_table_fit,
                    "x_fit": cache.x_fit,
                    "y_fit": cache.y_fit,
                    "error_table_report": cache.error_table_report,
                    "x_report": cache.x_report,
                    "y_report": cache.y_report,
                    "flops_by_width": cache.flops_by_width,
                    "hit_cap": cache.hit_cap,
                    "convergence": cache.convergence,
                    "steps_to_converge": cache.steps_to_converge,
                    "train_wall_clock_s": cache.train_wall_clock_s,
                    "training_schedule": cache.training_schedule,
                    "objective": cache.objective,
                }
            ),
            f,
        )
    torch.save(net.state_dict(), state_path)
    return cache


# ---------------------------------------------------------------------------
# One router-variant cell.
# ---------------------------------------------------------------------------


def run_cell(
    tier: Tier,
    seed: int,
    dimension: Dimension,
    value: float,
    *,
    w_max: int = W_MAX,
    n_train: int | None = None,
    n_test: int | None = None,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    check_every: int = DEFAULT_CHECK_EVERY,
    patience: int = DEFAULT_PATIENCE,
    min_delta: float = DEFAULT_MIN_DELTA,
    results_dir: str = RESULTS_DIR,
) -> dict:
    """Fits ONE router variant against `(tier, seed)`'s cached multi-head tables. Returns the JSON-able case.

    `eval_fn`/`cost_fn` below are table lookups, not net forward passes -- the cache IS the expensive
    part already paid for by whichever cell (this one, or an earlier one at a different `dimension`/
    `value`) first hit `_get_or_build_cache` for this `(tier, seed)`.
    """
    cache = _get_or_build_cache(
        results_dir, tier, seed, w_max=w_max, n_train=n_train, n_test=n_test, max_epochs=max_epochs, check_every=check_every, patience=patience, min_delta=min_delta
    )
    capacity_grid = [(k,) for k in range(1, w_max + 1)]

    def eval_fn(x: np.ndarray, capacity: tuple[int, ...]) -> np.ndarray:
        del x  # ignored -- .fit() always calls this with cache.x_fit, whose per-width error is already cached.
        return cache.error_table_fit[:, capacity[0] - 1]

    def cost_fn(capacity: tuple[int, ...]) -> float:
        return cache.flops_by_width[capacity[0] - 1]

    variant_hidden, variant_epochs, variant_lr = _router_config_for(dimension, value)
    variant_router = DistilledCapacityRouter(hidden=variant_hidden, n_epochs=variant_epochs, lr=variant_lr, seed=seed, device="cpu")
    variant_router.fit(eval_fn=eval_fn, x_val=cache.x_fit, y_val=cache.y_fit, capacity_grid=capacity_grid, tolerance=DEFAULT_TOLERANCE, cost_fn=cost_fn)

    default_router = DistilledCapacityRouter(hidden=DEFAULT_HIDDEN, n_epochs=DEFAULT_N_EPOCHS, lr=DEFAULT_LR, seed=seed, device="cpu")
    default_router.fit(eval_fn=eval_fn, x_val=cache.x_fit, y_val=cache.y_fit, capacity_grid=capacity_grid, tolerance=DEFAULT_TOLERANCE, cost_fn=cost_fn)

    variant_routed_idx = variant_router.route_index(cache.x_report)
    default_routed_idx = default_router.route_index(cache.x_report)
    n_report = len(cache.x_report)
    report_rows = np.arange(n_report)

    variant_quality = float(cache.error_table_report[report_rows, variant_routed_idx].mean())
    default_quality = float(cache.error_table_report[report_rows, default_routed_idx].mean())
    quality_ratio = variant_quality / default_quality if default_quality else math.inf
    agreement = float(np.mean(variant_routed_idx == default_routed_idx))

    routed_widths = (variant_routed_idx + 1).tolist()  # capacity_grid[i] = (i+1,), 1-indexed width.
    width_distribution = {k: routed_widths.count(k) for k in range(1, w_max + 1)}

    return {
        "tier": tier.value,
        "seed": seed,
        "toy": _TIER_CONFIG[tier].toy.value,
        "n_train": n_train if n_train is not None else _TIER_CONFIG[tier].n_train,
        "n_test": n_test if n_test is not None else _TIER_CONFIG[tier].n_test,
        "w_max": w_max,
        "dimension": dimension.value,
        "value": value,
        "router_config": {"hidden": list(variant_hidden), "depth": len(variant_hidden), "epochs": variant_epochs, "lr": variant_lr, "tolerance": DEFAULT_TOLERANCE},
        "frozen_default_router_config": {"hidden": list(DEFAULT_HIDDEN), "depth": len(DEFAULT_HIDDEN), "epochs": DEFAULT_N_EPOCHS, "lr": DEFAULT_LR},
        "routed_held_out_quality": variant_quality,
        "frozen_default_routed_held_out_quality": default_quality,
        "quality_ratio_to_default": quality_ratio,
        "width_distribution": width_distribution,
        "mean_routed_width": float(np.mean(routed_widths)),
        "agreement_with_frozen_default": agreement,
        "n_fit": len(cache.x_fit),
        "n_report": n_report,
        "hit_cap": cache.hit_cap,
        "training_schedule": cache.training_schedule,
        "objective": cache.objective,
        "multihead_steps_to_converge": cache.steps_to_converge,
        "multihead_train_wall_clock_s": cache.train_wall_clock_s,
        "provenance": run_provenance(),
    }


def _value_tag(dimension: Dimension, value: float) -> str:
    if dimension is Dimension.LR:
        return f"{value:g}".replace(".", "p")
    return str(int(value))


def _cell_json_path(results_dir: str, tier: Tier, seed: int, dimension: Dimension, value: float) -> str:
    return os.path.join(results_dir, f"wsel7_tier{tier.value}_seed{seed}_{dimension.value}_{_value_tag(dimension, value)}.json")


# ---------------------------------------------------------------------------
# --summarize: per-dimension invariance/plateau read + the SS3.6 frozen-constants artifact.
# ---------------------------------------------------------------------------


def _dimension_verdict(dimension: Dimension, per_value_cells: dict[float, dict]) -> dict:
    """Invariance + plateau read for one swept dimension (`width.md` SS3.5's WSEL-7 branch; module docstring)."""
    values = _DIMENSION_VALUES[dimension]
    ratio_by_value: dict[float, float] = {}
    n_cells_by_value: dict[float, int] = {}
    for v in values:
        cells = per_value_cells.get(v, {})
        n_cells_by_value[v] = len(cells)
        if cells:
            ratio_by_value[v] = float(np.mean([c["quality_ratio_to_default"] for c in cells.values()]))

    if not ratio_by_value:
        return {"values": list(values), "n_cells_by_value": n_cells_by_value, "status": "no cells on disk yet"}

    best_ratio = min(ratio_by_value.values())
    default_value = _frozen_default_value_for(dimension)
    default_ratio = ratio_by_value.get(default_value, math.inf)
    invariant = default_ratio <= best_ratio * (1.0 + _PLATEAU_REL_TOL)

    if dimension is Dimension.LR:
        plateau_value = min(ratio_by_value, key=ratio_by_value.get)  # no capacity ordering -- take the best.
    else:
        plateau_value = next(v for v in sorted(ratio_by_value) if ratio_by_value[v] <= best_ratio * (1.0 + _PLATEAU_REL_TOL))

    return {
        "values": list(values),
        "n_cells_by_value": n_cells_by_value,
        "ratio_to_default_by_value": ratio_by_value,
        "frozen_default_value": default_value,
        "default_ratio": default_ratio,
        "best_ratio": best_ratio,
        "plateau_rel_tol": _PLATEAU_REL_TOL,
        "invariant": invariant,
        "plateau_value": plateau_value,
    }


_NEW_DEFAULT_KEY_FOR_DIMENSION = {Dimension.HIDDEN: "hidden", Dimension.LAYERS: "depth", Dimension.EPOCHS: "epochs", Dimension.LR: "lr"}


def summarize(results_dir: str = RESULTS_DIR) -> None:
    """Aggregates every per-cell JSON on disk into `WSEL7/frozen.json`. Does not train or fit anything."""
    per_dimension_summary: dict[str, dict] = {}
    new_default = {"hidden": DEFAULT_HIDDEN[0], "depth": len(DEFAULT_HIDDEN), "epochs": DEFAULT_N_EPOCHS, "lr": DEFAULT_LR}
    any_noninvariant = False
    any_pending = False

    for dimension in Dimension:
        per_value_cells: dict[float, dict] = {}
        for v in _DIMENSION_VALUES[dimension]:
            cells = {}
            for tier in Tier:
                for seed in SEEDS:
                    path = _cell_json_path(results_dir, tier, seed, dimension, v)
                    if os.path.exists(path):
                        with open(path) as f:
                            cells[(tier.value, seed)] = json.load(f)
            per_value_cells[v] = cells
        verdict = _dimension_verdict(dimension, per_value_cells)
        per_dimension_summary[dimension.value] = verdict
        if verdict.get("status") == "no cells on disk yet":
            any_pending = True
            continue
        if not verdict["invariant"]:
            any_noninvariant = True
            new_default[_NEW_DEFAULT_KEY_FOR_DIMENSION[dimension]] = verdict["plateau_value"]

    invariant = not any_noninvariant
    out = {
        "hidden": DEFAULT_HIDDEN[0] if invariant else new_default["hidden"],
        "depth": len(DEFAULT_HIDDEN) if invariant else new_default["depth"],
        "epochs": DEFAULT_N_EPOCHS if invariant else new_default["epochs"],
        "lr": DEFAULT_LR if invariant else new_default["lr"],
        "invariant": invariant,
        "new_default": None if invariant else new_default,
        "any_dimension_pending": any_pending,
        "per_dimension": per_dimension_summary,
        "config": {
            "w_max": W_MAX,
            "seeds": list(SEEDS),
            "router_tolerance": DEFAULT_TOLERANCE,
            "plateau_rel_tol": _PLATEAU_REL_TOL,
            "router_report_fraction": _ROUTER_REPORT_FRACTION,
            "frozen_default_at_authoring_time": {"hidden": DEFAULT_HIDDEN[0], "depth": len(DEFAULT_HIDDEN), "epochs": DEFAULT_N_EPOCHS, "lr": DEFAULT_LR},
            "tier1": {
                "toy": _TIER_CONFIG[Tier.ONE].toy.value,
                "n_train": _TIER_CONFIG[Tier.ONE].n_train,
                "n_test": _TIER_CONFIG[Tier.ONE].n_test,
                "sigma": _TIER_CONFIG[Tier.ONE].sigma,
            },
            "tier2": {"toy": _TIER_CONFIG[Tier.TWO].toy.value, "n_train": _TIER_CONFIG[Tier.TWO].n_train, "n_test": _TIER_CONFIG[Tier.TWO].n_test},
        },
        "provenance": run_provenance(),
    }
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "frozen.json")
    with open(path, "w") as f:
        json.dump(_jsonable(out), f, indent=2)
    print(f"wrote {path}")
    print(f"invariant={invariant}  new_default={new_default if not invariant else None}  any_dimension_pending={any_pending}")


# ---------------------------------------------------------------------------
# --selftest
# ---------------------------------------------------------------------------

_SELFTEST_KW = {"w_max": 3, "n_train": 60, "n_test": 40, "max_epochs": 400, "check_every": 50, "patience": 2, "min_delta": 5e-2}
_SELFTEST_SELF_CONSISTENCY_TOL = 1e-9  # a default-value cell vs itself: exact float equality modulo router-training nondeterminism slop.

_SELFTEST_REQUIRED_KEYS = (
    "router_config",
    "frozen_default_router_config",
    "routed_held_out_quality",
    "frozen_default_routed_held_out_quality",
    "quality_ratio_to_default",
    "width_distribution",
    "mean_routed_width",
    "agreement_with_frozen_default",
    "hit_cap",
    "training_schedule",
    "objective",
    "multihead_steps_to_converge",
    "multihead_train_wall_clock_s",
)


def run_selftest() -> bool:
    """Tiny wiring check (<60s), entirely in a temp dir: cache-reuse, JSON round-trip, tier-2 wiring, `--summarize`."""
    ok = True
    tmp_dir = tempfile.mkdtemp(prefix="wsel7_selftest_")
    try:
        tier, seed = Tier.ONE, 0
        state_path = _cache_state_path(tmp_dir, tier, seed, _SELFTEST_KW["w_max"])

        case1 = run_cell(tier, seed, Dimension.HIDDEN, 16, results_dir=tmp_dir, **_SELFTEST_KW)
        mtime_after_first = os.path.getmtime(state_path)
        case2 = run_cell(tier, seed, Dimension.EPOCHS, 150, results_dir=tmp_dir, **_SELFTEST_KW)
        mtime_after_second = os.path.getmtime(state_path)
        cache_reused = mtime_after_first == mtime_after_second and case1["multihead_train_wall_clock_s"] == case2["multihead_train_wall_clock_s"]
        print(f"[wsel7 selftest] multi-head cache built once, reused across two router variants: {cache_reused}  {'PASS' if cache_reused else 'FAIL'}")
        ok = ok and cache_reused

        for case in (case1, case2):
            keys_ok = all(k in case for k in _SELFTEST_REQUIRED_KEYS)
            roundtrip = json.loads(json.dumps(_jsonable(case)))
            roundtrip_ok = all(k in roundtrip for k in _SELFTEST_REQUIRED_KEYS)
            combo_ok = keys_ok and roundtrip_ok
            print(f"[wsel7 selftest] dimension={case['dimension']} value={case['value']} keys_ok={keys_ok} json_roundtrip_ok={roundtrip_ok}  {'PASS' if combo_ok else 'FAIL'}")
            ok = ok and combo_ok

        # A variant at the frozen default's own value must agree perfectly with itself.
        case_default = run_cell(tier, seed, Dimension.HIDDEN, DEFAULT_HIDDEN[0], results_dir=tmp_dir, **_SELFTEST_KW)
        default_self_consistent = case_default["agreement_with_frozen_default"] == 1.0 and abs(case_default["quality_ratio_to_default"] - 1.0) < _SELFTEST_SELF_CONSISTENCY_TOL
        print(f"[wsel7 selftest] default-value cell agrees with itself: {default_self_consistent}  {'PASS' if default_self_consistent else 'FAIL'}")
        ok = ok and default_self_consistent

        # Tier 2 wiring: ALL schedule, fixed-sigma weighted objective (mirrors width_wsel13.py's tier 2).
        case_tier2 = run_cell(Tier.TWO, 0, Dimension.LR, 3e-3, results_dir=tmp_dir, **_SELFTEST_KW)
        tier2_ok = case_tier2["objective"] == "weighted_squared_error" and case_tier2["training_schedule"] == nwn.WidthSchedule.ALL.value
        print(f"[wsel7 selftest] tier-2 ALL-schedule / weighted-objective wiring: {tier2_ok}  {'PASS' if tier2_ok else 'FAIL'}")
        ok = ok and tier2_ok

        # --summarize: the SS3.6 frozen-constants artifact's required keys.
        summarize(results_dir=tmp_dir)
        with open(os.path.join(tmp_dir, "frozen.json")) as f:
            frozen = json.load(f)
        frozen_keys_ok = {"hidden", "depth", "epochs", "lr"} <= frozen.keys() and "invariant" in frozen
        print(f"[wsel7 selftest] frozen.json required keys present: {frozen_keys_ok}  {'PASS' if frozen_keys_ok else 'FAIL'}")
        ok = ok and frozen_keys_ok
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"[wsel7 selftest] {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """Parses args and dispatches to `--selftest` / `--summarize` / one real `--tier`/`--seed`/`--dimension`/`--value` cell."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="Tiny wiring check in a temp dir, then exit.")
    parser.add_argument("--summarize", action="store_true", help="Aggregate every per-cell JSON on disk into WSEL7/frozen.json.")
    parser.add_argument("--tier", type=int, choices=[t.value for t in Tier], default=None, help="SS3.8 tier (1 = reference, 2 = noisy-easy control).")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for this cell (canonical suite: 0, 1, 2).")
    parser.add_argument("--dimension", type=str, choices=[d.value for d in Dimension], default=None, help="Which router-architecture axis this cell varies.")
    parser.add_argument("--value", type=str, default=None, help="Swept value for --dimension (int for hidden/layers/epochs, float for lr).")
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS, help="Multi-head net's convergence-gate cap (not a router hyperparameter).")
    parser.add_argument("--check-every", type=int, default=DEFAULT_CHECK_EVERY)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA)
    parser.add_argument("--w-max", type=int, default=None)
    args = parser.parse_args()

    if args.selftest:
        sys.exit(0 if run_selftest() else 1)
    if args.summarize:
        summarize()
        return
    if args.tier is None or args.seed is None or args.dimension is None or args.value is None:
        parser.error("--tier, --seed, --dimension and --value are all required for a real cell (or pass --selftest / --summarize).")

    tier = Tier(args.tier)
    dimension = Dimension(args.dimension)
    value = float(args.value) if dimension is Dimension.LR else int(float(args.value))
    w_max = args.w_max if args.w_max is not None else W_MAX

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"[wsel7] tier={tier.value} seed={args.seed} dimension={dimension.value} value={value} w_max={w_max}", flush=True)
    case = run_cell(tier, args.seed, dimension, value, w_max=w_max, max_epochs=args.max_epochs, check_every=args.check_every, patience=args.patience, min_delta=args.min_delta)

    cell_path = _cell_json_path(RESULTS_DIR, tier, args.seed, dimension, value)
    with open(cell_path, "w") as f:
        json.dump(_jsonable(case), f, indent=2)
    print(f"wrote {cell_path}")
    print(
        f"routed_held_out_quality={case['routed_held_out_quality']:.5f}  "
        f"quality_ratio_to_default={case['quality_ratio_to_default']:.4f}  "
        f"agreement_with_frozen_default={case['agreement_with_frozen_default']:.3f}"
    )


if __name__ == "__main__":
    main()
