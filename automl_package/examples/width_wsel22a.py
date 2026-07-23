"""WSEL-22(a) -- the labelling-tolerance sensitivity sweep, plus the promoted WSEL-9 uncapped LightGBM re-run.

`docs/plans/capacity_programme/width.md` WSEL-22 (part a) and WSEL-10's real-data-inversion note.

**Why.** `routing.py`'s `DEFAULT_TOLERANCE = 0.25` (`automl_package/models/flexnn/routing.py:77`) is an
inherited constant, never measured (copied from `sinc_width_experiment.py`'s tie threshold). The
2026-07-23 joint review leaned on it ("arbitrary & frankly very large threshold"), firing width.md
SS1's pre-registered trigger. This driver turns it from inherited into measured: it relabels the
CACHED per-width error tables of the two DECIDED WSEL-19 router-backend bake-off levels (the 1-D
slice, unconditionally decided, and the d=2 multi-feature slice's verdict-weighted triples only --
`d2_oblique_seed0` is VOID and excluded, read live from `WSEL19/frozen_mf.json`) at
tolerance in {0.05, 0.10, 0.25, 0.50}, refitting routers per cell with `routing.py`'s OWN machinery
(`DistilledCapacityRouter.fit`/`fit_soft`, `_cheapest_within_tolerance_labels`) -- never
reimplemented, only ever imported. NO per-width net is ever retrained: every cell reads the SAME
cached per-width error tables/state dicts `width_wsel19.py` already built (`WSEL19/cache_seed*.json`,
`WSEL19/_mf_cache/*.pt`), verified present on disk before use and never written to (this task's
write set excludes `WSEL19/` entirely).

**Reuse first (verified against source, not recalled).** `width_wsel19.py`'s four backend-fit
functions (`_fit_frozen_mlp`/`_fit_rule_mlp`/`_fit_xgboost`/`_fit_constant`) each hardcode
`tolerance=DEFAULT_TOLERANCE` inline and are not parameterized -- and `width_wsel19.py` is outside
this task's write set (WSEL-22's non-goals: no edits to `routing.py`/any sibling driver). Per this
codebase's own established convention for exactly this situation (`routing.py`'s module docstring:
"COPIED, not imported, with provenance cited" when the source cannot be depended on or edited), the
four fit functions below are that same kind of copy: the OUTER backend-selection wrapper is
reproduced from `width_wsel19.py:552-774` with tolerance parameterized, one line per function; every
constant they use (hidden sizes, xgboost hyperparameters, the rule-sizing formula) and every actual
labelling/training primitive (`DistilledCapacityRouter`, `_cheapest_within_tolerance_labels`,
`_CapacityRouterMLP`, `_rule_sized_hidden`, `_global_cheapest_within_tolerance`) is IMPORTED from
`routing.py`/`width_wsel19.py`, never restated. `_score_hard`/`_score_blend`/`_FittedBackend`/
`_jsonable`/`Backend`/`Mode` carry no tolerance dependency and are imported and used verbatim.

**Part (a) -- the sensitivity sweep.** Per (level, tolerance): verdict stability (do the recorded
bake-off findings -- constant-wins-quality, the starved-cell inversion, blend-dominated-by-hard,
xgboost's geometry sensitivity, low oracle agreement -- still hold as ORDERING relationships, not
exact numbers), label churn vs the 0.25 labels (selection-side, on the cached error table), and the
routed-error / deployed-compute shift per tolerance (report-side, via the recomputed `per_group`
table). Frozen to `capacity_ladder_results/WSEL22A/frozen.json`.

**WSEL-9's yacht per-input cells -- explicitly in scope (user review, 2026-07-23), with a DEEPER
data-availability limit than first appeared, recorded here rather than hidden.** The real
`W-PERINPUT` router trains on the JOINTLY-TRAINED dial net's OWN per-width predictions
(`_run_dial_cell`'s `model.fit_router(x_sel, y_sel_std)`, `model` the dial net; internally
`FlexibleWidthNN.fit_router`'s `eval_fn` calls `self._per_sample_error_at_width` on THAT SAME
instance, `automl_package/models/flexnn/width/model.py:330,393`) -- and neither that network's
trained weights nor its own SELECT-split error table were ever cached (`capacity_ladder_results/
WSEL9/yacht_{seed}_dial.json` records only the AGGREGATE `w_perinput` fields -- `mean_routed_width`,
`width_distribution`, `held_out_mse` as a scalar -- and no `.pt`/state-dict file exists under
`WSEL9/`). So the TRUE per-input router cannot be relabeled or refit at all: this driver would need
either the dial net's weights (absent) or its SELECT-split error table (never persisted), and
neither exists on disk.

What IS on disk, and is NOT the same thing: `yacht_{seed}_w_sweep_{k}.json` (k=1..12) each cache a
`select_squared_error` array -- but from `_run_w_sweep_cell`'s INDEPENDENTLY-TRAINED dedicated net
at width k, a DIFFERENT model family than the dial net (WSEL-8's own finding: the dial net is
2.6-7.2x worse than dedicated nets at matched middle widths -- the two are not interchangeable
without introducing exactly the single-difference confound MASTER Decision 15 forbids). Using it
as W-PERINPUT's error table would silently answer a different question ("how would a per-input
router trained against the EXPENSIVE dedicated-net family respond to tolerance", not "does the
CHEAP jointly-trained W-PERINPUT's own win survive"). This driver therefore computes it anyway but
labels every field `_proxy` and states in every cell what it is NOT: a reconstruction of the actual
W-PERINPUT mechanism. What the proxy DOES answer honestly: how tolerance-sensitive the cheapest-
within-tolerance LABELLING RULE itself is on yacht's real held-out error landscape (a property of
the rule and the dataset, not of which model produced the curve) -- the selection-split `x`/`y`
inputs are DETERMINISTIC given `(dataset, seed, fraction_pct, test_fraction, max_train)`
(`width_wsel9._build_split`, pure `numpy` RNG, no model involved) -- re-deriving them is data
replay, not retraining. Neither the proxy's routing behaviour nor its accuracy should be read as
"the yacht win" surviving or not; the honest verdict is that the real per-input router's tolerance
sensitivity is UNANSWERABLE from cached artifacts, and this proxy is reported as the closest
available evidence, clearly flagged.

**Part 2 -- the promoted uncapped-LightGBM re-run (width.md WSEL-10's real-data-inversion note):**
the 5 `hit_cap: true` LightGBM cells under `WSEL9/` (identified by scanning every `*_lightgbm.json`
on disk, never from prose) -- `california_2`, `energy_2`, `kin8nm_{0,1,2}` -- re-run with IDENTICAL
config except `n_estimators` raised well past the original 200-tree cap. Landed beside the originals
under NEW filenames (`<dataset>_<seed>_lightgbm_uncapped.json`); the 5 capped originals are gate
evidence and are never overwritten, edited, or deleted.

**Non-goals:** no per-width retraining anywhere (every relabel reads a cached error table or a
cached/deterministically-replayed split); no change to `routing.py`'s or `width_wsel19.py`'s
defaults (FP-5.b binds; this task measures); no d ∈ {8, 32} cells (WSEL-21's scope, and both are
OPEN/protocol-blocked per `WSEL19/frozen_mf.json`, never verdict-weighted); no adoption decision
((b)/(b') own that, spec-gated, not this task).

Usage:
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel22a.py --selftest
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel22a.py --run-1d
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel22a.py --run-d2
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel22a.py --run-yacht
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel22a.py --run-uncapped-lightgbm
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/width_wsel22a.py --summarize
"""

from __future__ import annotations

import argparse
import enum
import json
import math
import os
import re
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
import torch
import torch.nn.functional as nnf
import xgboost as xgb

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root, so `import automl_package` works

import width_wsel9 as w9  # noqa: E402 -- Dataset/_build_split/_read_constants/_flexwidth_module_for_cost/LightGBMModel, reused verbatim
import width_wsel19 as w19  # noqa: E402 -- Backend/Mode/_FittedBackend/_score_hard/_score_blend/_jsonable/caches, reused verbatim
import width_wsel19_toys as wt  # noqa: E402 -- Geometry, make_selection_split/make_report_split

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
from automl_package.utils.capacity_accounting import executed_flops  # noqa: E402
from automl_package.utils.run_provenance import run_provenance  # noqa: E402

RESULTS_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "WSEL22A")
_WSEL19_RESULTS_DIR = w19.RESULTS_DIR
_WSEL9_RESULTS_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "WSEL9")
_FROZEN_MF_PATH = os.path.join(_WSEL19_RESULTS_DIR, "frozen_mf.json")

TOLERANCES: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50)  # the pre-registered sweep, width.md WSEL-22(a).
_BASELINE_TOLERANCE = DEFAULT_TOLERANCE  # 0.25 -- label churn is measured against this tolerance's labels.

YACHT_D = w9.Dataset.YACHT
_YACHT_ROUTER_SEED = 0  # `FlexibleWidthNN.fit_router` never overrides `DistilledCapacityRouter`'s own seed=0 default -- matched, not a bug introduced here.

# The 5 `hit_cap: true` LightGBM baseline cells (width.md WSEL-10's promoted follow-up), identified
# by scanning every `*_lightgbm.json` under WSEL9/ for `hit_cap: true` at authoring time -- recorded
# here as the closed set this task reruns, not re-derived from prose.
PROMOTED_UNCAPPED_CELLS: tuple[tuple[w9.Dataset, int], ...] = (
    (w9.Dataset.CALIFORNIA, 2),
    (w9.Dataset.ENERGY, 2),
    (w9.Dataset.KIN8NM, 0),
    (w9.Dataset.KIN8NM, 1),
    (w9.Dataset.KIN8NM, 2),
)
UNCAPPED_N_ESTIMATORS = 3000  # >>15x the original 200-tree cap; early stopping (rounds=15, unchanged) is expected to halt training well before this.

_MF_D_TARGET = 2  # this sweep's decided multi-feature level -- d=8/d=32 are OPEN/protocol-blocked (WSEL-21's scope, not this task's).


class Slice(enum.Enum):
    """Closed set: the two decided WSEL-19 bake-off levels this sweep relabels."""

    ONE_D = "1d"
    D2 = "d2"


# ---------------------------------------------------------------------------
# Backend fits, parameterized by tolerance (copies width_wsel19.py:552-774's outer wrapper only;
# every constant/primitive is imported, see module docstring).
# ---------------------------------------------------------------------------


def _fit_frozen_mlp_at(
    x_sel: np.ndarray, y_sel: np.ndarray, error_table_sel: np.ndarray, flops_by_width: list[float], seed: int, w_max: int, tolerance: float
) -> w19._FittedBackend:
    """Backend 1 at an explicit tolerance -- `width_wsel19._fit_frozen_mlp`, `tolerance` no longer hardcoded to `DEFAULT_TOLERANCE`."""
    capacity_grid = [(k,) for k in range(1, w_max + 1)]

    def eval_fn(x: np.ndarray, capacity: tuple[int, ...]) -> np.ndarray:
        del x
        return error_table_sel[:, capacity[0] - 1]

    def cost_fn(capacity: tuple[int, ...]) -> float:
        return flops_by_width[capacity[0] - 1]

    router = DistilledCapacityRouter(hidden=DEFAULT_HIDDEN, n_epochs=DEFAULT_N_EPOCHS, lr=DEFAULT_LR, seed=seed, device="cpu")
    router.fit(eval_fn=eval_fn, x_val=x_sel, y_val=y_sel, capacity_grid=capacity_grid, tolerance=tolerance, cost_fn=cost_fn)

    def route_proba(x: np.ndarray) -> np.ndarray:
        x_arr = _as_capacity_input(x)
        with torch.no_grad():
            logits = router.router_(torch.as_tensor(x_arr, dtype=torch.float32, device=router.device))
        return nnf.softmax(logits, dim=1).cpu().numpy()

    frozen_config = {
        "hidden": list(DEFAULT_HIDDEN),
        "depth": len(DEFAULT_HIDDEN),
        "epochs": DEFAULT_N_EPOCHS,
        "lr": DEFAULT_LR,
        "weight_decay": 0.0,
        "early_stopping": False,
        "tolerance": tolerance,
    }
    return w19._FittedBackend(route_index_fn=router.route_index, route_proba_fn=route_proba, config=frozen_config, sizing_rule=None)


def _fit_rule_mlp_at(x_sel: np.ndarray, error_table_sel: np.ndarray, seed: int, w_max: int, tolerance: float, *, in_dim: int = 1) -> w19._FittedBackend:
    """Backend 2 at an explicit tolerance -- `width_wsel19._fit_rule_mlp`, `tolerance` no longer hardcoded to `DEFAULT_TOLERANCE`."""
    hidden = w19._rule_sized_hidden(in_dim)
    labels = _cheapest_within_tolerance_labels(error_table_sel, tolerance=tolerance)
    x_arr = _as_capacity_input(x_sel)
    n = len(x_arr)
    val_mask = (np.arange(n) % w19._INTERNAL_VAL_EVERY) == 0

    torch.manual_seed(seed)
    net = _CapacityRouterMLP(in_dim, w_max, hidden=hidden)
    opt = torch.optim.AdamW(net.parameters(), lr=DEFAULT_LR, weight_decay=w19._ARM2_WEIGHT_DECAY)
    x_t = torch.as_tensor(x_arr, dtype=torch.float32)
    y_t = torch.as_tensor(labels, dtype=torch.long)
    x_tr_t, y_tr_t = x_t[~val_mask], y_t[~val_mask]
    x_val_t, y_val_t = x_t[val_mask], y_t[val_mask]

    best_val, patience_counter, best_state = float("inf"), 0, None
    actual_epochs = 0
    for epoch in range(w19._ARM2_MAX_EPOCHS):
        net.train()
        opt.zero_grad()
        loss = nnf.cross_entropy(net(x_tr_t), y_tr_t)
        loss.backward()
        opt.step()

        net.eval()
        with torch.no_grad():
            val_loss = float(nnf.cross_entropy(net(x_val_t), y_val_t).item())
        actual_epochs = epoch + 1
        if val_loss < best_val - w19._ARM2_MIN_DELTA:
            best_val, patience_counter = val_loss, 0
            best_state = {name: t.detach().clone() for name, t in net.state_dict().items()}
        else:
            patience_counter += 1
        if patience_counter >= w19._ARM2_PATIENCE:
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

    rule_mlp_config = {
        "hidden": list(hidden),
        "depth": w19._ARM2_DEPTH,
        "lr": DEFAULT_LR,
        "weight_decay": w19._ARM2_WEIGHT_DECAY,
        "early_stopping": True,
        "patience": w19._ARM2_PATIENCE,
        "min_delta": w19._ARM2_MIN_DELTA,
        "max_epochs": w19._ARM2_MAX_EPOCHS,
        "actual_epochs": actual_epochs,
        "hit_cap": actual_epochs >= w19._ARM2_MAX_EPOCHS,
        "n_train_internal": int((~val_mask).sum()),
        "n_val_internal": int(val_mask.sum()),
        "tolerance": tolerance,
    }
    sizing_rule = {
        "kind": "sqrt_input_dim",
        "hidden_width_formula": "round(32 * sqrt(in_dim))",
        "in_dim": in_dim,
        "hidden": list(hidden),
        "depth": w19._ARM2_DEPTH,
        "base_at_d1": w19._ARM2_SIZING_BASE,
    }
    return w19._FittedBackend(route_index_fn=route_index, route_proba_fn=route_proba, config=rule_mlp_config, sizing_rule=sizing_rule)


def _fit_xgboost_at(x_sel: np.ndarray, error_table_sel: np.ndarray, seed: int, w_max: int, tolerance: float) -> w19._FittedBackend:
    """Backend 3 at an explicit tolerance -- `width_wsel19._fit_xgboost`, `tolerance` no longer hardcoded to `DEFAULT_TOLERANCE`."""
    labels = _cheapest_within_tolerance_labels(error_table_sel, tolerance=tolerance)
    x_arr = _as_capacity_input(x_sel)
    n = len(x_arr)
    val_mask = (np.arange(n) % w19._INTERNAL_VAL_EVERY) == 0
    x_tr, y_tr_raw = x_arr[~val_mask], labels[~val_mask]
    x_val_raw, y_val_raw = x_arr[val_mask], labels[val_mask]

    train_classes = np.unique(y_tr_raw)
    class_to_dense = {int(c): i for i, c in enumerate(train_classes)}
    y_tr = np.array([class_to_dense[int(c)] for c in y_tr_raw], dtype=int)
    in_vocab = np.isin(y_val_raw, train_classes)
    x_val, y_val = x_val_raw[in_vocab], np.array([class_to_dense[int(c)] for c in y_val_raw[in_vocab]], dtype=int)
    use_early_stopping = len(x_val) > 0

    if len(train_classes) < w19._MIN_CLASSES_FOR_CLASSIFIER:
        c_star = int(train_classes[0])

        def route_index_constant(x: np.ndarray) -> np.ndarray:
            return np.full(len(x), c_star, dtype=int)

        def route_proba_constant(x: np.ndarray) -> np.ndarray:
            proba = np.zeros((len(x), w_max), dtype=np.float64)
            proba[:, c_star] = 1.0
            return proba

        degenerate_config = {
            "degenerate_single_class": True,
            "selected_capacity_index": c_star,
            "n_train_internal": int((~val_mask).sum()),
            "n_val_internal": int(val_mask.sum()),
            "tolerance": tolerance,
        }
        return w19._FittedBackend(route_index_fn=route_index_constant, route_proba_fn=route_proba_constant, config=degenerate_config, sizing_rule=None)

    clf = xgb.XGBClassifier(
        n_estimators=w19._XGB_N_ESTIMATORS,
        max_depth=w19._XGB_MAX_DEPTH,
        learning_rate=w19._XGB_LEARNING_RATE,
        subsample=w19._XGB_SUBSAMPLE,
        colsample_bytree=w19._XGB_COLSAMPLE_BYTREE,
        early_stopping_rounds=w19._XGB_EARLY_STOPPING_ROUNDS if use_early_stopping else None,
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
        proba_dense = clf.predict_proba(_as_capacity_input(x))
        proba_full = np.zeros((len(x), w_max), dtype=np.float64)
        proba_full[:, train_classes] = proba_dense
        return proba_full

    return w19._FittedBackend(
        route_index_fn=route_index,
        route_proba_fn=route_proba,
        config={
            "n_estimators": w19._XGB_N_ESTIMATORS,
            "max_depth": w19._XGB_MAX_DEPTH,
            "learning_rate": w19._XGB_LEARNING_RATE,
            "subsample": w19._XGB_SUBSAMPLE,
            "colsample_bytree": w19._XGB_COLSAMPLE_BYTREE,
            "early_stopping_rounds": w19._XGB_EARLY_STOPPING_ROUNDS if use_early_stopping else None,
            "early_stopping_active": use_early_stopping,
            "best_iteration": int(clf.best_iteration) if use_early_stopping else w19._XGB_N_ESTIMATORS,
            "n_classes_observed": len(train_classes),
            "n_train_internal": int((~val_mask).sum()),
            "n_val_internal": len(x_val),
            "tolerance": tolerance,
        },
        sizing_rule=None,
    )


def _fit_constant_at(error_table_sel: np.ndarray, w_max: int, tolerance: float) -> w19._FittedBackend:
    """Backend 4 at an explicit tolerance -- `width_wsel19._fit_constant`, `tolerance` no longer hardcoded to `DEFAULT_TOLERANCE`."""
    c_star = w19._global_cheapest_within_tolerance(error_table_sel, tolerance)

    def route_index(x: np.ndarray) -> np.ndarray:
        return np.full(len(x), c_star, dtype=int)

    def route_proba(x: np.ndarray) -> np.ndarray:
        proba = np.zeros((len(x), w_max), dtype=np.float64)
        proba[:, c_star] = 1.0
        return proba

    return w19._FittedBackend(
        route_index_fn=route_index, route_proba_fn=route_proba, config={"selected_capacity_index": c_star, "selected_width": c_star + 1, "tolerance": tolerance}, sizing_rule=None
    )


def _fit_backend_at(
    backend: w19.Backend, x_sel: np.ndarray, y_sel: np.ndarray, error_table_sel: np.ndarray, flops_by_width: list[float], seed: int, w_max: int, in_dim: int, tolerance: float
) -> w19._FittedBackend:
    """Dispatches to the requested backend's tolerance-parameterized fit function."""
    if backend is w19.Backend.FROZEN_MLP:
        return _fit_frozen_mlp_at(x_sel, y_sel, error_table_sel, flops_by_width, seed, w_max, tolerance)
    if backend is w19.Backend.RULE_MLP:
        return _fit_rule_mlp_at(x_sel, error_table_sel, seed, w_max, tolerance, in_dim=in_dim)
    if backend is w19.Backend.XGBOOST:
        return _fit_xgboost_at(x_sel, error_table_sel, seed, w_max, tolerance)
    return _fit_constant_at(error_table_sel, w_max, tolerance)


def _label_churn(error_table: np.ndarray, tolerance: float, baseline_tolerance: float = _BASELINE_TOLERANCE) -> dict[str, Any]:
    """Fraction of rows whose cheapest-within-tolerance label differs from the `baseline_tolerance` labelling, same error table."""
    labels_tau = _cheapest_within_tolerance_labels(error_table, tolerance=tolerance)
    labels_baseline = _cheapest_within_tolerance_labels(error_table, tolerance=baseline_tolerance)
    churned = labels_tau != labels_baseline
    return {
        "tolerance": tolerance,
        "baseline_tolerance": baseline_tolerance,
        "n_rows": len(labels_tau),
        "n_churned": int(churned.sum()),
        "churn_fraction": float(churned.mean()),
        "mean_label_shift": float(np.mean(labels_tau.astype(np.int64) - labels_baseline.astype(np.int64))),
    }


# ---------------------------------------------------------------------------
# 1-D slice -- reads WSEL19/cache_seed{seed}.json (verified present, never (re)written by this task).
# ---------------------------------------------------------------------------


def _verify_1d_cache_exists(seed: int) -> str:
    """Asserts the 1-D bake-off's cache file for `seed` exists under WSEL19/ -- never written by this task."""
    path = w19._sweep_cache_json_path(_WSEL19_RESULTS_DIR, seed)
    if not os.path.exists(path):
        raise SystemExit(f"{path} is missing -- WSEL-22(a) reads the WSEL-19 1-D bake-off's cache, it never builds one. Run WSEL-19's 1-D slice first.")
    return path


def run_1d_cell(seed: int, n_sel: int, backend: w19.Backend, tolerance: float, *, results_dir: str = RESULTS_DIR) -> tuple[dict[str, Any], dict[str, Any]]:
    """Relabels + refits ONE (backend, n_sel, seed) 1-D cell at `tolerance`; returns (hard_case, blend_case).

    Fits ONCE, scores both modes -- no extra training, matches `width_wsel19.py`'s own convention.
    """
    _verify_1d_cache_exists(seed)
    cache = w19._get_or_build_sweep_cache(seed, results_dir=_WSEL19_RESULTS_DIR)
    w_max = cache.error_table_pool.shape[1]  # the cache's own column count -- w19.W_MAX for the real grid, whatever `--selftest`'s tiny fixture used otherwise.

    x_sel = cache.x_pool[:n_sel]
    y_sel = cache.y_pool[:n_sel]
    error_table_sel = cache.error_table_pool[:n_sel]
    in_dim = 1 if x_sel.ndim == 1 else x_sel.shape[1]

    fitted = _fit_backend_at(backend, x_sel, y_sel, error_table_sel, cache.flops_by_width, seed, w_max, in_dim, tolerance)
    churn = _label_churn(error_table_sel, tolerance)

    cases = []
    for mode in w19.Mode:
        readout = w19._SCORERS[mode](fitted, cache.x_report, cache.error_table_report, cache.flops_by_width, w_max)
        if mode is w19.Mode.HARD:
            idx = np.asarray(fitted.route_index_fn(cache.x_report), dtype=int)
            oracle_idx_tau = _cheapest_within_tolerance_labels(cache.error_table_report, tolerance=tolerance)
            readout = {**readout, "oracle_agreement_at_tolerance": float(np.mean(idx == oracle_idx_tau))}
        case = {
            "slice": Slice.ONE_D.value,
            "backend": backend.value,
            "mode": mode.value,
            "n_sel": n_sel,
            "seed": seed,
            "tolerance": tolerance,
            "backend_config": fitted.config,
            "sizing_rule": fitted.sizing_rule,
            "label_churn": churn,
            "provenance": run_provenance(),
        }
        case.update(readout)
        cases.append(case)
        path = os.path.join(results_dir, f"wsel22a_1d_{backend.value}_{mode.value}_nsel{n_sel}_seed{seed}_tol{tolerance}.json")
        os.makedirs(results_dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump(w19._jsonable(case), f, indent=2)
    return cases[0], cases[1]


def run_1d_sweep(tolerances: tuple[float, ...] = TOLERANCES, results_dir: str = RESULTS_DIR) -> None:
    """Relabels + refits every (backend, n_sel, seed) 1-D bake-off cell at every tolerance in `tolerances`."""
    for seed in w19.SEEDS:
        for n_sel in w19.N_SEL_VALUES:
            for backend in w19.Backend:
                for tolerance in tolerances:
                    hard_case, _blend_case = run_1d_cell(seed, n_sel, backend, tolerance, results_dir=results_dir)
                    print(
                        f"[wsel22a 1d] seed={seed} n_sel={n_sel} backend={backend.value} tol={tolerance} "
                        f"quality={hard_case['routed_held_out_quality']:.4g} churn={hard_case['label_churn']['churn_fraction']:.3f}"
                    )


# ---------------------------------------------------------------------------
# d=2 slice -- reads WSEL19/_mf_cache/*.pt (verified present, never (re)written by this task).
# ---------------------------------------------------------------------------


def decided_d2_triples(frozen_mf_path: str = _FROZEN_MF_PATH) -> list[tuple[wt.Geometry, int, int]]:
    """Reads `WSEL19/frozen_mf.json`'s `triples` for every VERDICT-WEIGHTED d=2 (geometry, seed, n_train).

    d=2 oblique seed 0 is VOID (`verdict_weight: false`, fit-gate failure) and is excluded here by
    construction, read from disk rather than hardcoded (width.md's own binding: the decided scope is
    whatever the ledger currently marks verdict-weighted, not a restated list).
    """
    with open(frozen_mf_path) as f:
        frozen_mf = json.load(f)
    triples: list[tuple[wt.Geometry, int, int]] = []
    for key, value in frozen_mf["triples"].items():
        m = re.match(r"^d(\d+)_(axis|oblique)_seed(\d+)$", key)
        if m is None:
            continue
        d, geometry_str, seed = int(m.group(1)), m.group(2), int(m.group(3))
        if d == _MF_D_TARGET and value.get("verdict_weight") is True:
            triples.append((wt.Geometry(geometry_str), seed, int(value["n_train"])))
    return sorted(triples, key=lambda t: (t[0].value, t[1]))


def _verify_mf_cache_complete(d: int, geometry: wt.Geometry, seed: int, n_train: int, w_max: int, results_dir: str = _WSEL19_RESULTS_DIR) -> None:
    """Asserts every one of the `w_max` per-width state dicts is already cached -- refuses rather than silently retraining."""
    for width in range(1, w_max + 1):
        state_path, meta_path = w19._mf_model_cache_paths(results_dir, d, geometry, seed, n_train, width)
        if not (os.path.exists(state_path) and os.path.exists(meta_path)):
            raise SystemExit(
                f"{state_path} (or its meta) is missing -- WSEL-22(a) only reads WSEL-19's cached multi-feature nets, it never trains one. "
                "Run WSEL-19's d=2 multi-feature slice first."
            )


def run_d2_cell(
    geometry: wt.Geometry,
    seed: int,
    n_train: int,
    n_sel: int,
    backend: w19.Backend,
    tolerance: float,
    *,
    d: int = _MF_D_TARGET,
    w_max: int = w19.W_MAX,
    results_dir: str = RESULTS_DIR,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Relabels + refits ONE (geometry, seed, n_sel, backend) d=2 cell at `tolerance`; returns (hard_case, blend_case)."""
    _verify_mf_cache_complete(d, geometry, seed, n_train, w_max)
    models, _metas, cache_hit, n_train_used = w19._get_or_build_mf_models(seed, d, geometry, w_max=w_max, n_train=n_train, results_dir=_WSEL19_RESULTS_DIR)
    if not cache_hit:
        raise SystemExit(f"d=2 {geometry.value} seed={seed} n_train={n_train}: not every width was a cache hit -- refusing (this task never trains a per-width net).")

    x_report, y_report, _region_report, _t_report = wt.make_report_split(seed, d, geometry)
    x_sel, y_sel, _region_sel, _t_sel = wt.make_selection_split(seed, d, geometry, n_sel)
    error_table_report = w19._mf_error_table(models, x_report, y_report, w_max)
    error_table_sel = w19._mf_error_table(models, x_sel, y_sel, w_max)
    flops_by_width = [float(executed_flops(models[w].model, w)) for w in range(1, w_max + 1)]

    fitted = _fit_backend_at(backend, x_sel, y_sel, error_table_sel, flops_by_width, seed, w_max, d, tolerance)
    churn = _label_churn(error_table_sel, tolerance)

    cases = []
    for mode in w19.Mode:
        readout = w19._SCORERS[mode](fitted, x_report, error_table_report, flops_by_width, w_max)
        if mode is w19.Mode.HARD:
            idx = np.asarray(fitted.route_index_fn(x_report), dtype=int)
            oracle_idx_tau = _cheapest_within_tolerance_labels(error_table_report, tolerance=tolerance)
            readout = {**readout, "oracle_agreement_at_tolerance": float(np.mean(idx == oracle_idx_tau))}
        case = {
            "slice": Slice.D2.value,
            "d": d,
            "geometry": geometry.value,
            "backend": backend.value,
            "mode": mode.value,
            "n_sel": n_sel,
            "seed": seed,
            "n_train": n_train_used,
            "tolerance": tolerance,
            "backend_config": fitted.config,
            "sizing_rule": fitted.sizing_rule,
            "label_churn": churn,
            "provenance": run_provenance(),
        }
        case.update(readout)
        cases.append(case)
        path = os.path.join(results_dir, f"wsel22a_d2_{geometry.value}_{backend.value}_{mode.value}_nsel{n_sel}_seed{seed}_tol{tolerance}.json")
        os.makedirs(results_dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump(w19._jsonable(case), f, indent=2)
    return cases[0], cases[1]


def run_d2_sweep(tolerances: tuple[float, ...] = TOLERANCES, results_dir: str = RESULTS_DIR) -> None:
    """Relabels + refits every VERDICT-WEIGHTED d=2 (geometry, seed, n_sel, backend) cell at every tolerance in `tolerances`."""
    triples = decided_d2_triples()
    for geometry, seed, n_train in triples:
        for n_sel in w19.N_SEL_VALUES:
            for backend in w19.Backend:
                for tolerance in tolerances:
                    hard_case, _blend_case = run_d2_cell(geometry, seed, n_train, n_sel, backend, tolerance, results_dir=results_dir)
                    print(
                        f"[wsel22a d2] geometry={geometry.value} seed={seed} n_train={n_train} n_sel={n_sel} backend={backend.value} tol={tolerance} "
                        f"quality={hard_case['routed_held_out_quality']:.4g} churn={hard_case['label_churn']['churn_fraction']:.3f}"
                    )


# ---------------------------------------------------------------------------
# WSEL-9 yacht per-input sensitivity -- a PROXY only (see `_YACHT_MECHANISM_CAVEAT` and the module
# docstring): the actual W-PERINPUT router's error table (the dial net's own per-width predictions)
# was never cached, so this section relabels the W-SWEEP dedicated nets' cached `select_squared_error`
# arrays instead -- a different, better-performing model family. x/y are replayed deterministically
# via `width_wsel9._build_split` (pure numpy RNG, no model involved -- data replay, not retraining).
# ---------------------------------------------------------------------------


_YACHT_MECHANISM_CAVEAT = (
    "the real W-PERINPUT router trains on the JOINTLY-TRAINED dial net's OWN per-width predictions "
    "(_run_dial_cell's model.fit_router, model the dial net), and neither that net's weights nor its "
    "SELECT-split error table were ever cached -- so the true per-input router cannot be relabeled or "
    "refit at all. This cell's error table comes from yacht_{seed}_w_sweep_{k}.json's select_squared_error "
    "instead -- the INDEPENDENTLY-TRAINED dedicated-net family (WSEL-8: 2.6-7.2x BETTER than the dial net "
    "at matched middle widths), a different model. Every field below is therefore a PROXY answering 'how "
    "tolerance-sensitive is the cheapest-within-tolerance rule on yacht's real held-out error landscape', "
    "NOT 'does the actual W-PERINPUT win survive this tolerance' -- that question is unanswerable from "
    "cached artifacts. held_out_mse is additionally None because no per-sample TEST-split prediction was "
    "ever cached for the dial net at any width (only the aggregate scalar in yacht_{seed}_dial.json)."
)


def _load_yacht_proxy_context(seed: int) -> dict[str, Any]:
    """Reconstructs `(x_sel, y_sel, error_table_sel, x_test, d, flops_by_width)` for yacht `seed` from the cached W-SWEEP dedicated nets.

    A proxy, not the actual dial-network error table -- see `_YACHT_MECHANISM_CAVEAT`.
    """
    constants = w9._read_constants(w9.WSEL6_FROZEN_PATH, w9.WSEL7_FROZEN_PATH, w9.WSEL8_DIR)
    fraction_pct = constants["selection_fraction"]["fraction_pct"]
    w_max = constants["width_ladder"]["w_max"]
    split = w9._build_split(w9.Dataset.YACHT, seed, fraction_pct=fraction_pct, test_fraction=w9.TEST_FRACTION, max_train=None)

    error_cols = []
    for width in range(1, w_max + 1):
        cell_path = os.path.join(_WSEL9_RESULTS_DIR, f"yacht_{seed}_w_sweep_{width}.json")
        if not os.path.exists(cell_path):
            raise SystemExit(f"{cell_path} is missing -- yacht's per-input sensitivity proxy needs every cached w_sweep width cell.")
        with open(cell_path) as f:
            cell = json.load(f)
        if cell["n_selection_used"] != split["n_selection_used"]:
            raise SystemExit(
                f"{cell_path}: n_selection_used {cell['n_selection_used']} != replayed split's {split['n_selection_used']} -- "
                "split reconstruction mismatch, refusing to relabel on stale data."
            )
        error_cols.append(cell["select_squared_error"])
    error_table_sel = np.array(error_cols, dtype=np.float64).T  # (n_selection_used, w_max), cheapest-first -- W-SWEEP's dedicated nets, NOT the dial net.

    d = split["d"]
    flops_by_width = [float(executed_flops(w9._flexwidth_module_for_cost(d, seed, width), width)) for width in range(1, w_max + 1)]
    return {
        "x_sel": split["x_select"],
        "y_sel": split["y_select_std"],
        "error_table_sel": error_table_sel,
        "x_test": split["x_test"],
        "d": d,
        "w_max": w_max,
        "flops_by_width": flops_by_width,
        "n_selection_used": split["n_selection_used"],
    }


def run_yacht_cell(seed: int, tolerance: float, *, results_dir: str = RESULTS_DIR) -> dict[str, Any]:
    """Relabels yacht's W-SWEEP-proxy selection-side error table at `tolerance`, refits a frozen-MLP router on it.

    NOT the actual W-PERINPUT mechanism -- see `_YACHT_MECHANISM_CAVEAT`.
    """
    ctx = _load_yacht_proxy_context(seed)
    churn = _label_churn(ctx["error_table_sel"], tolerance)

    fitted = _fit_frozen_mlp_at(ctx["x_sel"], ctx["y_sel"], ctx["error_table_sel"], ctx["flops_by_width"], _YACHT_ROUTER_SEED, ctx["w_max"], tolerance)
    routed_idx = np.asarray(fitted.route_index_fn(ctx["x_test"]), dtype=int)
    routed_widths = (routed_idx + 1).tolist()
    flops_arr = np.asarray(ctx["flops_by_width"])

    case = {
        "slice": "wsel9_yacht_perinput_proxy",
        "seed": seed,
        "tolerance": tolerance,
        "error_table_source": "w_sweep_dedicated_nets_proxy",
        "is_actual_w_perinput_mechanism": False,
        "label_churn": churn,
        "proxy_mean_routed_width": float(np.mean(routed_widths)),
        "proxy_width_distribution": {str(k): routed_widths.count(k) for k in range(1, ctx["w_max"] + 1)},
        "proxy_mean_deployed_flops": float(flops_arr[routed_idx].mean()),
        "held_out_mse": None,
        "accuracy_recomputable": False,
        "mechanism_caveat": _YACHT_MECHANISM_CAVEAT,
        "router_config": fitted.config,
        "n_selection_used": ctx["n_selection_used"],
        "provenance": run_provenance(),
    }
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"wsel22a_yacht_perinput_seed{seed}_tol{tolerance}.json")
    with open(path, "w") as f:
        json.dump(w19._jsonable(case), f, indent=2)
    return case


def run_yacht_sweep(tolerances: tuple[float, ...] = TOLERANCES, results_dir: str = RESULTS_DIR) -> None:
    """Runs `run_yacht_cell` for every yacht seed x every tolerance."""
    for seed in w19.SEEDS:
        for tolerance in tolerances:
            case = run_yacht_cell(seed, tolerance, results_dir=results_dir)
            print(
                f"[wsel22a yacht PROXY, not actual W-PERINPUT] seed={seed} tol={tolerance} "
                f"mean_routed_width={case['proxy_mean_routed_width']:.3f} churn={case['label_churn']['churn_fraction']:.3f}"
            )


# ---------------------------------------------------------------------------
# Part 2 -- the promoted uncapped-LightGBM re-run.
# ---------------------------------------------------------------------------


def _run_lightgbm_cell_uncapped(dataset: w9.Dataset, seed: int, split: dict[str, Any], constants: dict[str, Any], n_estimators: int, original_path: str) -> dict[str, Any]:
    """Reruns `width_wsel9._run_lightgbm_cell` with `n_estimators` raised past the original cap; records what it supersedes."""
    model = w9.LightGBMModel(n_estimators=n_estimators, early_stopping_rounds=w9.LIGHTGBM_EARLY_STOPPING_ROUNDS, random_seed=seed, verbose=-1, calculate_feature_importance=False)
    best_iteration, loss_history = model._fit_single(split["x_fit"], split["y_fit_raw"], x_val=split["x_stop"], y_val=split["y_stop_raw"])
    replay = w9.w4._replay(loss_history, w9.LIGHTGBM_EARLY_STOPPING_ROUNDS, w9.LIGHTGBM_REPLAY_MIN_DELTA)
    hit_cap = bool(best_iteration >= n_estimators)
    trustworthy = bool(replay.trustworthy and not hit_cap)

    pred = np.asarray(model.predict(split["x_test"], filter_data=False)).reshape(-1)
    held_out_mse = float(np.mean((pred - split["y_test_raw"]) ** 2))

    with open(original_path) as f:
        original = json.load(f)

    return {
        "dataset": dataset.value,
        "seed": seed,
        "arm": "lightgbm_uncapped",
        "width": None,
        "held_out_mse": held_out_mse,
        "held_out_trajectory": replay.summary()["trajectory"],
        "trustworthy": trustworthy,
        "hit_cap": hit_cap,
        "converged": replay.converged,
        "trajectory_applicable": True,
        "selection_cost": None,
        "config": {"n_estimators": n_estimators, "early_stopping_rounds": w9.LIGHTGBM_EARLY_STOPPING_ROUNDS, "best_iteration": int(best_iteration), **split["config"]},
        "constants": constants,
        "supersedes": {
            "original_path": os.path.abspath(original_path),
            "original_held_out_mse": original["held_out_mse"],
            "original_hit_cap": original["hit_cap"],
            "original_n_estimators": original["config"]["n_estimators"],
            "config_delta": {"n_estimators": {"before": original["config"]["n_estimators"], "after": n_estimators}},
        },
        "provenance": run_provenance(),
    }


def run_uncapped_lightgbm(results_dir: str = _WSEL9_RESULTS_DIR, n_estimators: int = UNCAPPED_N_ESTIMATORS) -> None:
    """Reruns every `PROMOTED_UNCAPPED_CELLS` entry uncapped, landing each beside its original under a new filename."""
    constants = w9._read_constants(w9.WSEL6_FROZEN_PATH, w9.WSEL7_FROZEN_PATH, w9.WSEL8_DIR)
    for dataset, seed in PROMOTED_UNCAPPED_CELLS:
        original_path = os.path.join(_WSEL9_RESULTS_DIR, f"{dataset.value}_{seed}_lightgbm.json")
        if not os.path.exists(original_path):
            raise SystemExit(f"{original_path} is missing -- cannot supersede a cell that isn't on disk.")
        split = w9._build_split(dataset, seed, fraction_pct=constants["selection_fraction"]["fraction_pct"], test_fraction=w9.TEST_FRACTION, max_train=None)
        cell = _run_lightgbm_cell_uncapped(dataset, seed, split, constants, n_estimators, original_path)
        out_path = os.path.join(results_dir, f"{dataset.value}_{seed}_lightgbm_uncapped.json")
        with open(out_path, "w") as f:
            json.dump(w19._jsonable(cell), f, indent=2)
        print(
            f"[wsel22a uncapped-lightgbm] {dataset.value} seed={seed}: "
            f"capped mse={cell['supersedes']['original_held_out_mse']:.5g} hit_cap={cell['supersedes']['original_hit_cap']} "
            f"-> uncapped mse={cell['held_out_mse']:.5g} best_iteration={cell['config']['best_iteration']} hit_cap={cell['hit_cap']}"
        )


# ---------------------------------------------------------------------------
# --summarize -- verdict-stability checks, label churn, routed-error/deployed-compute shift, all
# per tolerance; frozen to WSEL22A/frozen.json.
# ---------------------------------------------------------------------------


def _load_cells(results_dir: str, prefix: str) -> list[dict[str, Any]]:
    """Loads every `<prefix>*.json` cell under `results_dir`."""
    cells = []
    for name in sorted(os.listdir(results_dir)):
        if name.startswith(prefix) and name.endswith(".json"):
            with open(os.path.join(results_dir, name)) as f:
                cells.append(json.load(f))
    return cells


def _group_key(cell: dict[str, Any]) -> tuple[Any, ...]:
    """Groups a cell by everything except its seed, so `_mean_se` can average across seeds.

    `cell["d"]` is deliberately OMITTED from the d=2 key (every d2 cell has `d == _MF_D_TARGET`; the
    "d2" slice tag already disambiguates from the 1-D slice) -- this keeps the label format
    `slice:geometry:backend:mode:n_sel:tolerance` identical to what `_verdict_stability_table`'s
    `prefix = f"d2:{geometry.value}"` lookups construct.
    """
    if cell["slice"] == Slice.D2.value:
        return (cell["slice"], cell["geometry"], cell["backend"], cell["mode"], cell["n_sel"], cell["tolerance"])
    return (cell["slice"], cell["backend"], cell["mode"], cell["n_sel"], cell["tolerance"])


def _aggregate_per_group(cells: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Mean +/- SE of `routed_held_out_quality`/`mean_deployed_flops`/`label_churn.churn_fraction` per group, across seeds -- mirrors `width_wsel19.summarize`'s own shape."""
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for cell in cells:
        grouped.setdefault(_group_key(cell), []).append(cell)
    out = {}
    for key, group in grouped.items():
        label = ":".join(str(k) for k in key)
        out[label] = {
            "n_seeds": len(group),
            "routed_held_out_quality": w19._mean_se([c["routed_held_out_quality"] for c in group]),
            "mean_deployed_flops": w19._mean_se([c["mean_deployed_flops"] for c in group]),
            "churn_fraction": w19._mean_se([c["label_churn"]["churn_fraction"] for c in group]),
        }
    return out


def _finding_constant_wins_quality(per_group: dict[str, dict[str, Any]], slice_prefix: str, n_sel: int, tolerance: float) -> bool | None:
    """F1/G1: does `constant` still have the lowest routed quality AND the highest deployed FLOPs among the 4 backends, at (n_sel, tolerance), HARD mode?"""
    rows = {}
    for backend in w19.Backend:
        key = f"{slice_prefix}:{backend.value}:hard:{n_sel}:{tolerance}"
        if key not in per_group:
            return None
        rows[backend] = per_group[key]
    constant_quality = rows[w19.Backend.CONSTANT]["routed_held_out_quality"]["mean"]
    constant_flops = rows[w19.Backend.CONSTANT]["mean_deployed_flops"]["mean"]
    other_qualities = [rows[b]["routed_held_out_quality"]["mean"] for b in w19.Backend if b is not w19.Backend.CONSTANT]
    other_flops = [rows[b]["mean_deployed_flops"]["mean"] for b in w19.Backend if b is not w19.Backend.CONSTANT]
    return bool(constant_quality <= min(other_qualities) and constant_flops >= max(other_flops))


def _finding_blend_dominated_by_hard(per_group: dict[str, dict[str, Any]], slice_prefix: str, backend: w19.Backend, n_sel: int, tolerance: float) -> bool | None:
    """F3/G3: is BLEND worse quality AND more deployed compute than HARD, for `backend` at (n_sel, tolerance)?"""
    hard_key = f"{slice_prefix}:{backend.value}:hard:{n_sel}:{tolerance}"
    blend_key = f"{slice_prefix}:{backend.value}:blend:{n_sel}:{tolerance}"
    if hard_key not in per_group or blend_key not in per_group:
        return None
    hard, blend = per_group[hard_key], per_group[blend_key]
    return bool(blend["routed_held_out_quality"]["mean"] > hard["routed_held_out_quality"]["mean"] and blend["mean_deployed_flops"]["mean"] > hard["mean_deployed_flops"]["mean"])


def _finding_frozen_more_robust_than_rule_mlp(per_group: dict[str, dict[str, Any]], slice_prefix: str, n_sel: int, tolerance: float) -> bool | None:
    """F2/G2 (axis leg): at the starved n_sel, is frozen_mlp's routed quality better (lower) than rule_mlp's, HARD mode?"""
    frozen_key = f"{slice_prefix}:frozen_mlp:hard:{n_sel}:{tolerance}"
    rule_key = f"{slice_prefix}:rule_mlp:hard:{n_sel}:{tolerance}"
    if frozen_key not in per_group or rule_key not in per_group:
        return None
    return bool(per_group[frozen_key]["routed_held_out_quality"]["mean"] < per_group[rule_key]["routed_held_out_quality"]["mean"])


# CONSTANT's route_proba is one-hot by construction (`w19._fit_constant`'s own docstring: "HARD and
# BLEND coincide for this arm") -- blend-vs-hard is not a meaningful comparison for it, at any tolerance.
_BLEND_COMPARABLE_BACKENDS = tuple(b for b in w19.Backend if b is not w19.Backend.CONSTANT)


def _verdict_stability_table(per_group_1d: dict[str, dict[str, Any]], per_group_d2: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Recomputes the bake-off's recorded ORDERING findings at every tolerance.

    `True` == the finding still holds, `False` == it flipped, `None` == not computable (missing cells).
    """
    table: dict[str, Any] = {}
    for tolerance in TOLERANCES:
        entry = {}
        for n_sel in w19.N_SEL_VALUES:
            entry[f"1d_constant_wins_quality_nsel{n_sel}"] = _finding_constant_wins_quality(per_group_1d, "1d", n_sel, tolerance)
            entry[f"1d_frozen_more_robust_than_rule_mlp_nsel{n_sel}"] = _finding_frozen_more_robust_than_rule_mlp(per_group_1d, "1d", n_sel, tolerance)
            for backend in _BLEND_COMPARABLE_BACKENDS:
                entry[f"1d_blend_dominated_by_hard_{backend.value}_nsel{n_sel}"] = _finding_blend_dominated_by_hard(per_group_1d, "1d", backend, n_sel, tolerance)
        for geometry in wt.Geometry:
            prefix = f"d2:{geometry.value}"
            for n_sel in w19.N_SEL_VALUES:
                entry[f"d2_{geometry.value}_constant_wins_quality_nsel{n_sel}"] = _finding_constant_wins_quality(per_group_d2, prefix, n_sel, tolerance)
                entry[f"d2_{geometry.value}_frozen_more_robust_than_rule_mlp_nsel{n_sel}"] = _finding_frozen_more_robust_than_rule_mlp(per_group_d2, prefix, n_sel, tolerance)
                for backend in _BLEND_COMPARABLE_BACKENDS:
                    key = f"d2_{geometry.value}_blend_dominated_by_hard_{backend.value}_nsel{n_sel}"
                    entry[key] = _finding_blend_dominated_by_hard(per_group_d2, prefix, backend, n_sel, tolerance)
        table[str(tolerance)] = entry
    return table


def _yacht_survival(results_dir: str) -> dict[str, Any]:
    """Per-tolerance yacht label-churn + PROXY routing-behaviour summary; `is_actual_w_perinput_mechanism` is `False` throughout (see `_YACHT_MECHANISM_CAVEAT`)."""
    cells = _load_cells(results_dir, "wsel22a_yacht_perinput_seed")
    grouped: dict[float, list[dict[str, Any]]] = {}
    for cell in cells:
        grouped.setdefault(cell["tolerance"], []).append(cell)
    out: dict[str, Any] = {"mechanism_caveat": _YACHT_MECHANISM_CAVEAT, "is_actual_w_perinput_mechanism": False}
    for tolerance, group in grouped.items():
        out[str(tolerance)] = {
            "n_seeds": len(group),
            "proxy_mean_routed_width": w19._mean_se([c["proxy_mean_routed_width"] for c in group]),
            "churn_fraction": w19._mean_se([c["label_churn"]["churn_fraction"] for c in group]),
            "proxy_mean_deployed_flops": w19._mean_se([c["proxy_mean_deployed_flops"] for c in group]),
            "accuracy_recomputable": False,
        }
    return out


def summarize(results_dir: str = RESULTS_DIR) -> None:
    """Aggregates every landed cell under `results_dir` into `frozen.json`.

    Verdict stability, label churn, routed-error/deployed-compute shift, yacht survival, uncapped-LightGBM deltas.
    """
    cells_1d = _load_cells(results_dir, "wsel22a_1d_")
    cells_d2 = _load_cells(results_dir, "wsel22a_d2_")
    per_group_1d = _aggregate_per_group(cells_1d)
    per_group_d2 = _aggregate_per_group(cells_d2)

    uncapped_cells = []
    for dataset, seed in PROMOTED_UNCAPPED_CELLS:
        path = os.path.join(_WSEL9_RESULTS_DIR, f"{dataset.value}_{seed}_lightgbm_uncapped.json")
        if os.path.exists(path):
            with open(path) as f:
                uncapped_cells.append(json.load(f))

    frozen = {
        "tolerances_swept": list(TOLERANCES),
        "baseline_tolerance": _BASELINE_TOLERANCE,
        "n_cells_1d": len(cells_1d),
        "n_cells_d2": len(cells_d2),
        "decided_d2_triples": [{"geometry": g.value, "seed": s, "n_train": n} for g, s, n in decided_d2_triples()],
        "per_group_1d": per_group_1d,
        "per_group_d2": per_group_d2,
        "verdict_stability": _verdict_stability_table(per_group_1d, per_group_d2),
        "yacht_survival": _yacht_survival(results_dir),
        "uncapped_lightgbm": [
            {
                "dataset": c["dataset"],
                "seed": c["seed"],
                "original_held_out_mse": c["supersedes"]["original_held_out_mse"],
                "original_hit_cap": c["supersedes"]["original_hit_cap"],
                "uncapped_held_out_mse": c["held_out_mse"],
                "uncapped_hit_cap": c["hit_cap"],
                "best_iteration": c["config"]["best_iteration"],
            }
            for c in uncapped_cells
        ],
        "provenance": run_provenance(),
    }
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "frozen.json"), "w") as f:
        json.dump(w19._jsonable(frozen), f, indent=2)
    print(
        f"[wsel22a summarize] wrote {os.path.join(results_dir, 'frozen.json')}: "
        f"{len(cells_1d)} 1-D cells, {len(cells_d2)} d=2 cells, {len(uncapped_cells)} uncapped-lightgbm cells."
    )


# ---------------------------------------------------------------------------
# --selftest
# ---------------------------------------------------------------------------


def run_selftest() -> bool:
    """Known-answer relabelling check (incl. a tolerance-driven label flip) + a tiny end-to-end fit/score/churn/summarize wiring check, entirely in a temp dir."""
    ok = True

    # Known-answer table: 3 rows, 2 capacities (cheapest-first). Row 0 is built to FLIP its label
    # between the low tolerances (0.05, 0.10) and the high ones (0.25, 0.50): col0 error 1.20 is
    # within tolerance of col1's 1.00 (the row min) iff 1.20 <= (1+tau)*1.00, i.e. tau >= 0.20.
    error_table = np.array(
        [
            [1.20, 1.00],  # flips: label=1 at tau in {0.05, 0.10}, label=0 at tau in {0.25, 0.50}.
            [1.00, 5.00],  # never flips: col0 is both cheapest AND best at every tau.
            [3.00, 3.00],  # never flips: tie -- cheapest (col0) wins by construction (argmax of an all-True row).
        ]
    )
    expected_labels = {0.05: [1, 0, 0], 0.10: [1, 0, 0], 0.25: [0, 0, 0], 0.50: [0, 0, 0]}
    for tau, expected in expected_labels.items():
        labels = _cheapest_within_tolerance_labels(error_table, tolerance=tau).tolist()
        cell_ok = labels == expected
        ok = ok and cell_ok
        print(f"[wsel22a selftest] labels at tau={tau}: {labels} (expected {expected})  {'PASS' if cell_ok else 'FAIL'}")

    churn_low = _label_churn(error_table, 0.05, baseline_tolerance=0.25)
    churn_ok = churn_low["n_churned"] == 1 and math.isclose(churn_low["churn_fraction"], 1.0 / 3.0)
    ok = ok and churn_ok
    print(f"[wsel22a selftest] label churn tau=0.05 vs 0.25: {churn_low}  {'PASS' if churn_ok else 'FAIL'}")

    churn_none = _label_churn(error_table, 0.25, baseline_tolerance=0.25)
    churn_none_ok = churn_none["n_churned"] == 0
    ok = ok and churn_none_ok
    print(f"[wsel22a selftest] label churn tau=0.25 vs itself is zero: {churn_none_ok}")

    # Tiny end-to-end: a synthetic (x, y) pool with 2 capacities, fit each backend at two
    # tolerances, score both modes, and check summarize()'s round-trip -- entirely in a temp dir.
    tmp_dir = tempfile.mkdtemp(prefix="wsel22a_selftest_")
    try:
        rng = np.random.default_rng(0)
        n = 40
        x_sel = rng.uniform(-1, 1, size=n).astype(np.float32)
        y_sel = rng.normal(size=n).astype(np.float32)
        # A synthetic error table shaped like the toy: capacity 0 (cheap) noisier than capacity 1 (expensive).
        synth_error_table = np.stack([np.abs(x_sel) + 0.3, np.abs(x_sel) * 0.1], axis=1).astype(np.float64)
        flops_by_width = [1.0, 4.0]
        x_report = rng.uniform(-1, 1, size=20).astype(np.float32)
        report_error_table = np.stack([np.abs(x_report) + 0.3, np.abs(x_report) * 0.1], axis=1).astype(np.float64)

        for tolerance in (0.05, 0.50):
            for backend in w19.Backend:
                fitted = _fit_backend_at(backend, x_sel, y_sel, synth_error_table, flops_by_width, seed=0, w_max=2, in_dim=1, tolerance=tolerance)
                for mode in w19.Mode:
                    readout = w19._SCORERS[mode](fitted, x_report, report_error_table, flops_by_width, 2)
                    finite_ok = math.isfinite(readout["routed_held_out_quality"]) and math.isfinite(readout["mean_deployed_flops"])
                    ok = ok and finite_ok
                    if not finite_ok:
                        print(f"[wsel22a selftest] FAIL: backend={backend.value} mode={mode.value} tol={tolerance} produced a non-finite readout: {readout}")

        # Full cell wiring + summarize, using the 1-D cell path against a monkey-patched tiny cache.
        seed, n_sel = 0, 10
        cache_path = w19._sweep_cache_json_path(tmp_dir, seed)
        with open(cache_path, "w") as f:
            json.dump(
                w19._jsonable(
                    {
                        "x_pool": x_sel,
                        "y_pool": y_sel,
                        "error_table_pool": synth_error_table,
                        "x_report": x_report,
                        "y_report": rng.normal(size=20).astype(np.float32),
                        "error_table_report": report_error_table,
                        "flops_by_width": flops_by_width,
                        "hit_cap": False,
                        "trustworthy": True,
                        "reused_from_wsel6": False,
                        "n_train": 100,
                        "n_test": 20,
                        "n_report": 20,
                    }
                ),
                f,
            )
        global _WSEL19_RESULTS_DIR  # noqa: PLW0603 -- selftest-only redirect so run_1d_cell reads the tiny fixture instead of the real WSEL19 cache; restored in `finally`.
        real_wsel19_dir = _WSEL19_RESULTS_DIR
        _WSEL19_RESULTS_DIR = tmp_dir
        try:
            for tolerance in (0.05, 0.25):
                run_1d_cell(seed, n_sel, w19.Backend.FROZEN_MLP, tolerance, results_dir=tmp_dir)
        finally:
            _WSEL19_RESULTS_DIR = real_wsel19_dir

        summarize(results_dir=tmp_dir)
        with open(os.path.join(tmp_dir, "frozen.json")) as f:
            frozen = json.load(f)
        _expected_selftest_cells = 4  # the two `run_1d_cell` calls above (tau in {0.05, 0.25}), each landing BOTH a HARD and a BLEND cell.
        summarize_ok = "verdict_stability" in frozen and "0.05" in frozen["verdict_stability"] and frozen["n_cells_1d"] == _expected_selftest_cells
        ok = ok and summarize_ok
        print(f"[wsel22a selftest] summarize round-trip: n_cells_1d={frozen.get('n_cells_1d')}  {'PASS' if summarize_ok else 'FAIL'}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"[wsel22a selftest] {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """Dispatches to one of `--selftest`/`--run-1d`/`--run-d2`/`--run-yacht`/`--run-uncapped-lightgbm`/`--summarize`."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--run-1d", action="store_true")
    parser.add_argument("--run-d2", action="store_true")
    parser.add_argument("--run-yacht", action="store_true")
    parser.add_argument("--run-uncapped-lightgbm", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR)
    args = parser.parse_args()

    modes = (args.selftest, args.run_1d, args.run_d2, args.run_yacht, args.run_uncapped_lightgbm, args.summarize)
    if sum(bool(m) for m in modes) != 1:
        parser.error("exactly one of --selftest/--run-1d/--run-d2/--run-yacht/--run-uncapped-lightgbm/--summarize is required.")

    if args.selftest:
        sys.exit(0 if run_selftest() else 1)
    if args.run_1d:
        run_1d_sweep(results_dir=args.results_dir)
        return
    if args.run_d2:
        run_d2_sweep(results_dir=args.results_dir)
        return
    if args.run_yacht:
        run_yacht_sweep(results_dir=args.results_dir)
        return
    if args.run_uncapped_lightgbm:
        run_uncapped_lightgbm()
        return
    summarize(results_dir=args.results_dir)


if __name__ == "__main__":
    main()
