r"""WSEL-22(b) -- the sigma-anchored per-input labelling-band evaluation driver.

`docs/plans/capacity_programme/width.md` WSEL-22 (part b); implements
`docs/plans/capacity_programme/shared/wsel22b-sigma-anchored-band.md` (the ratified, adversarially-read
spec -- read it in full before touching this file, especially SS4/SS5.2/SS5.3/SS5.5's root amendments) SS9's
driver contract.

**What this driver measures.** `_cheapest_within_tolerance_labels`'s flat rule (`routing.py:95-113`, budget
`tau * e_i(w*_i)`, currently `DEFAULT_TOLERANCE=0.25` everywhere) is replaced, per row, with a candidate
whose acceptability budget is anchored on the toy's own KNOWN noise level instead of a free constant: width
`w` is acceptable for row `i` at confidence `c` iff `excess_i(w) := e_i(w) - e_i(w*_i) <= sigma_i**2 *
chi2_{1,c}` (spec SS2.3), where `chi2_{1,c}` is the one-sided chi-square(1) quantile -- this is EXACTLY a
two-sided z-test on the row's own residual at the same confidence (spec SS2.4). `sigma_i` is a LOOKUP, never
an estimate (`HETERO_NOISE_SIGMA=0.05`, `nested_width_net.py:93`) -- every decided cell this driver runs on
is `make_hetero`-family (spec SS1/SS5.4), so `sigma_i` is the SAME constant for every row; see `_sigma_true`.

**Adoption gate (spec SS4, root-amended): WIN-ONLY.** The sigma-anchored band is adopted as the toy-domain
replacement candidate iff it WINS the generator-true oracle-agreement comparison (SS5) at `c=0.95` -- a tie
retains the flat-0.25 incumbent (a difference is not an improvement; (a)'s own sensitivity sweep already
showed the incumbent's headline findings are band-insensitive). The draft's alternative bar -- "does
swapping the labels flip a decided verdict" -- is DEMOTED to a reported, non-gating impact assessment
(`_verdict_stability_impact`), never adoption evidence.

**Reuse first (verified against source, not recalled).** `_cheapest_within_tolerance_labels`,
`DistilledCapacityRouter` (incl. its private `_fit_from_targets` training step, called directly here to
bypass `fit()`'s own internal flat-tolerance labelling -- see `_fit_frozen_mlp_at_labels`), `_as_capacity_input`
and `_CapacityRouterMLP` are IMPORTED verbatim from `automl_package.models.flexnn.routing` (never edited --
SS11 forbids it). `width_wsel19.py`'s cache loaders (`_get_or_build_sweep_cache`, `_get_or_build_mf_models`),
scorers (`_SCORERS`/`_score_hard`/`_score_blend`), the generator-true machinery (`_hetero_h`, `_mf_error_table`),
`_mean_se`, `_jsonable`, `Backend`/`Mode`/`_FittedBackend`, and every backend hyperparameter constant are used
by qualified reference (`w19.X`), never restated. `width_wsel22a.py`'s `decided_d2_triples`,
`_verify_1d_cache_exists`/`_verify_mf_cache_complete`, `_aggregate_per_group`/`_group_key`/`_load_cells`, and
the three verdict-stability finding functions (`_finding_constant_wins_quality`,
`_finding_frozen_more_robust_than_rule_mlp`, `_finding_blend_dominated_by_hard`) are likewise reused by
qualified reference (`w22a.X`) -- the finding functions format their last argument into an f-string GROUP KEY
only (never operate on it numerically), so passing a sigma confidence in the same position they format a
flat tolerance into reuses them UNCHANGED (see `_verdict_stability_impact`).

**What could NOT be imported, and is COPIED with provenance instead (this codebase's own established
convention for exactly this situation -- `routing.py`'s module docstring; `width_wsel22a.py`'s own docstring
repeats it for its tolerance-parameterized copies of `width_wsel19.py`'s backend-fit functions).**
`width_wsel19.py`/`width_wsel22a.py`'s four backend-fit functions each hard-code a call to
`_cheapest_within_tolerance_labels` (or, for the CONSTANT arm, `width_wsel19._global_cheapest_within_tolerance`)
internally and are not parameterized to accept a precomputed label array, and neither module may be edited
from this task (write-set exclusion). `_fit_frozen_mlp_at_labels`/`_fit_rule_mlp_at_labels`/
`_fit_xgboost_at_labels`/`_fit_constant_at_labels` below are those same wrappers, COPIED, with their one
internal labelling call replaced by a `labels`/`rule` argument the caller supplies -- spec SS9's own words:
"swaps the labelling function called inside those wrappers, not the wrappers themselves". Every
hyperparameter/architecture/primitive inside them is otherwise identical to (and reused directly from,
where it is a bare constant or class) `width_wsel19.py`/`width_wsel22a.py`.

**The new function, one home (spec SS9).** `_sigma_anchored_acceptable`/`_sigma_anchored_labels` (SS2.3) and
the `LabelRule` dispatcher that lets the SAME backend-fit-at-labels functions serve the flat rule too (needed
for the SS5.2 M1/M2 oracle protocol's `flat_0.25` leg, and for a byte-identical regression guard against
`_cheapest_within_tolerance_labels`'s own flat path -- selftest assertion 4).

**The oracle-agreement protocol (spec SS5.2, root-amended to the symmetric M1/M2 two-metric form) needs NO
backend refit at all** -- `_oracle_agreement` compares raw rule-applied LABELS on the noisy vs generator-true
REPORT-split error tables (M1: a rule's noisy-table label vs its own true-table label; M2: a rule's
noisy-table label vs the true table's strict per-row argmin, `width_wsel19.py:1143-1151`'s own machinery,
scored against the noise-free target). This is cheap (one extra forward pass per width, spec SS5.1) and is
computed ONCE per decided cell (`run_1d_oracle_cell`/`run_d2_oracle_cell`), never per (backend, mode, n_sel)
sub-cell -- those three axes are irrelevant to a pure label-agreement comparison. The SEPARATE, non-gating
verdict-stability impact assessment (spec SS4(i)/SS6 step 4) DOES need one router refit per (backend, n_sel)
at the sigma-anchored labelling, ONLY at the pre-registered default `c=0.95` (spec SS4(i)'s own text: "at
c=0.95") -- `run_1d_impact_cell`/`run_d2_impact_cell`. The flat-tolerance comparison points for that
assessment are NEVER recomputed (spec SS6 step 2: "already on disk, reused not recomputed") -- `summarize()`
reads WSEL22A's own frozen `verdict_stability["0.25"]` entry directly.

**Two flagged textual ambiguities in the spec, resolved here (smallest faithful reading, not asserted as the
only possible one):**
1. SS3's prose says the evaluation "sweeps all four listed `c` values" immediately below a table that lists
   FIVE candidate confidence levels (0.68, 0.6827, 0.90, 0.95, 0.99). Rather than arbitrarily dropping one,
   `CONFIDENCE_LEVELS` below sweeps all five -- strictly more informative, and nothing in SS4/SS5's adoption
   gate depends on which four the prose meant (adoption keys off `c=0.95` alone).
2. SS9's "Per-cell CLI" flag list (`--geometry`/`--seed`/`--n-train`/`--backend`/`--mode`/`--confidence`/
   `--n-sel`) matches `width_wsel19.py`'s own true one-cell-per-process CLI shape, which this driver's
   `main()` follows literally. Within that shape, `--mode` is accepted but does not gate which router gets
   fit: mirroring `width_wsel22a.py`'s own convention (its `run_1d_cell`/`run_d2_cell` fit once and land BOTH
   modes), `run_1d_impact_cell`/`run_d2_impact_cell` fit once and always write both the HARD and BLEND JSON
   for a cell -- `--mode` only selects which case's summary line `main()` prints, halving the router-training
   compute the ROOT's grid run pays for no loss of coverage (both files land either way).

**Non-goals (spec SS11, restated -- binding regardless of outcome):** no edit to `routing.py` (this driver's
new function lives here, not there); no per-width retraining anywhere (every cell reads a cached per-width
net); no estimated-sigma variant (spec SS7: toys-only, `sigma_true` is a lookup); no change to the blend's
evaluation path (spec SS8: HARD routing's labels only); no new cells beyond the 1-D slice + WSEL22A's 5
decided d=2 triples; no yacht per-input coverage (spec SS5.4: not merely uncached, not DEFINABLE -- tracked
at WSEL-6-R); no resolution of WSEL-22(c) (dormant, separate task). This driver's author never runs the
evaluation grid -- the ROOT runs every cell, backgrounded, and re-verifies against disk before freezing.

Usage:
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m automl_package.examples.width_wsel22b --selftest
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m automl_package.examples.width_wsel22b \\
        --oracle --seed 0                                        # 1-D oracle cell (M1/M2 + A1 coverage)
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m automl_package.examples.width_wsel22b \\
        --oracle --geometry axis --seed 0 --n-train 4000          # d=2 oracle cell
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m automl_package.examples.width_wsel22b \\
        --backend frozen_mlp --n-sel 300 --seed 0 --confidence 0.95   # 1-D impact cell (both modes landed)
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m automl_package.examples.width_wsel22b \\
        --backend rule_mlp --n-sel 300 --seed 1 --geometry oblique --n-train 1500 --confidence 0.95
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m automl_package.examples.width_wsel22b --summarize
"""

from __future__ import annotations

import argparse
import enum
import itertools
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as nnf
import xgboost as xgb
from scipy.stats import chi2 as _chi2_dist

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root, so `import automl_package` works

import nested_width_net as nwn  # noqa: E402 -- HETERO_NOISE_SIGMA, the known noise level every sigma-anchored cell reads.
import width_wsel4 as w4  # noqa: E402 -- PORTED_* ported-arm protocol constants, reused verbatim (see module docstring).
import width_wsel6 as w6  # noqa: E402 -- Tier/_sweep_cache_paths/_load_cached_model, reused verbatim to reload 1-D per-width nets.
import width_wsel19 as w19  # noqa: E402 -- Backend/Mode/_FittedBackend/_SCORERS/caches/_hetero_h/_mf_error_table/_mean_se/_jsonable, reused verbatim.
import width_wsel19_toys as wt  # noqa: E402 -- Geometry, make_selection_split/make_report_split.
import width_wsel22a as w22a  # noqa: E402 -- Slice/decided_d2_triples/_verify_*_cache*/_aggregate_per_group/_load_cells/finding functions, reused verbatim.

from automl_package.models.flexnn.routing import (  # noqa: E402
    DEFAULT_HIDDEN,
    DEFAULT_LR,
    DEFAULT_N_EPOCHS,
    DistilledCapacityRouter,
    _as_capacity_input,
    _CapacityRouterMLP,
    _cheapest_within_tolerance_labels,
)
from automl_package.utils.capacity_accounting import executed_flops  # noqa: E402
from automl_package.utils.run_provenance import run_provenance  # noqa: E402

RESULTS_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "WSEL22B")

# Spec SS3's candidate table -- all 5 rows swept (see module docstring's ambiguity note 1).
CONFIDENCE_LEVELS: tuple[float, ...] = (0.68, 0.6827, 0.90, 0.95, 0.99)
_DEFAULT_CONFIDENCE = 0.95  # spec SS3's pre-registered default -- SS4's adoption gate and the verdict-stability impact assessment key off this value alone.
_SE_MARGIN = 2.0  # Decision 33(i)'s noise-aware bar, reused per spec SS5.3's root amendment (no new, unsourced threshold).


class RuleKind(enum.Enum):
    """Closed set: which acceptability test a `LabelRule` applies -- the incumbent flat rule (spec SS2.2) or this spec's sigma-anchored rule (SS2.3)."""

    FLAT_TOLERANCE = "flat"
    SIGMA_ANCHORED = "sigma"


@dataclass(frozen=True)
class LabelRule:
    """One labelling-rule instance: `FLAT_TOLERANCE` at a tolerance `tau`, or `SIGMA_ANCHORED` at a confidence `c`."""

    kind: RuleKind
    param: float

    def labels(self, error_table: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Per-row labels under this rule -- `_cheapest_within_tolerance_labels` (flat) or `_sigma_anchored_labels` (sigma, spec SS2.3). `sigma` is unused for the flat branch."""
        if self.kind is RuleKind.FLAT_TOLERANCE:
            return _cheapest_within_tolerance_labels(error_table, tolerance=self.param)
        return _sigma_anchored_labels(error_table, sigma, confidence=self.param)

    @property
    def tag(self) -> str:
        """A unique, human-readable identifier for this rule instance (e.g. `sigma_0.95`, `flat_0.25`)."""
        return f"{self.kind.value}_{self.param}"


def _chi2_quantile(confidence: float) -> float:
    """`chi2_{1,c}` -- the `c`-quantile of the chi-square(1) distribution (spec SS2.3/SS2.4)."""
    return float(_chi2_dist.ppf(confidence, df=1))


def _sigma_true(n: int) -> np.ndarray:
    """`(n,)` per-row KNOWN noise std -- the constant `HETERO_NOISE_SIGMA` for every row, every decided cell.

    Spec SS1/SS5.4: every decided cell this driver runs on is `make_hetero`-family (never `make_hetero3`'s
    region-dependent noise), so `sigma_true` is the SAME constant for every row here -- a lookup, never an
    estimate (SS3.7). No region-indexed branch is implemented (this task's non-goals: no cells beyond the
    1-D slice + WSEL22A's decided d=2 triples, all `make_hetero`-family).
    """
    return np.full(n, nwn.HETERO_NOISE_SIGMA, dtype=np.float64)


def _sigma_anchored_acceptable(error_table: np.ndarray, sigma: np.ndarray, confidence: float) -> np.ndarray:
    """Boolean acceptability matrix for spec SS2.3's rule: `excess_i(w) <= sigma_i**2 * chi2_{1,c}` per row/column.

    Args:
        error_table: `(N, n_capacities)` per-sample squared error at each capacity, cheapest-first columns.
        sigma: `(N,)` per-row known noise std (`_sigma_true`).
        confidence: the chi-square one-sided confidence level `c` (spec SS3's candidate table).

    Returns:
        `(N, n_capacities)` boolean acceptability matrix.
    """
    min_error = error_table.min(axis=1, keepdims=True)
    excess = error_table - min_error  # >= 0 by construction -- w*_i is the row minimum (spec SS2.3).
    band = (np.asarray(sigma, dtype=np.float64) ** 2 * _chi2_quantile(confidence))[:, None]
    return excess <= band


def _sigma_anchored_labels(error_table: np.ndarray, sigma: np.ndarray, confidence: float) -> np.ndarray:
    """Cheapest capacity within the sigma-anchored band of the row-best (spec SS2.3).

    `argmax` on the boolean row returns the first True -- the cheapest acceptable capacity, matching
    `_cheapest_within_tolerance_labels`'s own convention (capacity_grid columns are cheapest-first). The
    row-best itself always satisfies this trivially (`excess=0`), so this reduces to the SAME "first True in
    a cheapest-first boolean row" mechanism as the flat rule -- only the acceptability test differs.
    """
    return _sigma_anchored_acceptable(error_table, sigma, confidence).argmax(axis=1)


def _global_capacity_for_rule(error_table: np.ndarray, sigma: np.ndarray, rule: LabelRule) -> int:
    """The CONSTANT backend's "one capacity for everyone" pick at `rule`'s labelling.

    Generalizes `width_wsel19._global_cheapest_within_tolerance` to an arbitrary `LabelRule` by applying the
    SAME per-row acceptability test to the table's column-mean row (the whole-selection-set analogue of one
    row). Identical to `width_wsel19._global_cheapest_within_tolerance`'s own result when `rule` is
    `FLAT_TOLERANCE` -- exercised by `--selftest`'s byte-identical regression check.
    """
    mean_error = error_table.mean(axis=0, keepdims=True)  # (1, n_capacities).
    mean_sigma = np.array([float(np.mean(sigma))])
    return int(rule.labels(mean_error, mean_sigma)[0])


# ---------------------------------------------------------------------------
# Backend fits at a PRECOMPUTED label array -- COPIED from `width_wsel19.py`/`width_wsel22a.py` with
# provenance (see module docstring): every hyperparameter/primitive is identical, only the internal
# labelling call is replaced by an externally supplied `labels`/`rule` argument.
# ---------------------------------------------------------------------------


def _fit_frozen_mlp_at_labels(x_sel: np.ndarray, flops_by_width: list[float], seed: int, w_max: int, labels: np.ndarray, rule_config: dict[str, Any]) -> w19._FittedBackend:
    """Backend 1 (frozen_mlp) trained on a PRECOMPUTED label array.

    Calls `DistilledCapacityRouter._fit_from_targets` directly with `hard_labels=labels` -- the SAME shared
    training step `fit()` itself calls after building its own flat-tolerance labels internally
    (`routing.py:213-218`) -- bypassing that internal labelling so an arbitrary rule can be trained against
    (spec SS9: "swaps the labelling function called inside those wrappers, not the wrappers themselves").
    Architecture/hyperparameters are otherwise identical to `width_wsel22a._fit_frozen_mlp_at`.
    """
    capacity_grid = [(k,) for k in range(1, w_max + 1)]

    def cost_fn(capacity: tuple[int, ...]) -> float:
        return flops_by_width[capacity[0] - 1]

    router = DistilledCapacityRouter(hidden=DEFAULT_HIDDEN, n_epochs=DEFAULT_N_EPOCHS, lr=DEFAULT_LR, seed=seed, device="cpu")
    x_arr = _as_capacity_input(x_sel)
    router._fit_from_targets(x_arr, capacity_grid, cost_fn, hard_labels=labels)

    def route_proba(x: np.ndarray) -> np.ndarray:
        x_arr2 = _as_capacity_input(x)
        with torch.no_grad():
            logits = router.router_(torch.as_tensor(x_arr2, dtype=torch.float32, device=router.device))
        return nnf.softmax(logits, dim=1).cpu().numpy()

    frozen_config = {
        "hidden": list(DEFAULT_HIDDEN),
        "depth": len(DEFAULT_HIDDEN),
        "epochs": DEFAULT_N_EPOCHS,
        "lr": DEFAULT_LR,
        "weight_decay": 0.0,
        "early_stopping": False,
        **rule_config,
    }
    return w19._FittedBackend(route_index_fn=router.route_index, route_proba_fn=route_proba, config=frozen_config, sizing_rule=None)


def _fit_rule_mlp_at_labels(x_sel: np.ndarray, labels: np.ndarray, seed: int, w_max: int, rule_config: dict[str, Any], *, in_dim: int = 1) -> w19._FittedBackend:
    """Backend 2 (rule_mlp) at a PRECOMPUTED label array.

    `width_wsel22a._fit_rule_mlp_at`'s body, COPIED with provenance (its own
    `_cheapest_within_tolerance_labels` call replaced by `labels`, supplied by the caller).
    """
    hidden = w19._rule_sized_hidden(in_dim)
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
        **rule_config,
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


def _fit_xgboost_at_labels(x_sel: np.ndarray, labels: np.ndarray, seed: int, w_max: int, rule_config: dict[str, Any]) -> w19._FittedBackend:
    """Backend 3 (xgboost) at a PRECOMPUTED label array.

    `width_wsel22a._fit_xgboost_at`'s body, COPIED with provenance (its own
    `_cheapest_within_tolerance_labels` call replaced by `labels`).
    """
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
            **rule_config,
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
            **rule_config,
        },
        sizing_rule=None,
    )


def _fit_constant_at_labels(error_table_sel: np.ndarray, sigma_sel: np.ndarray, w_max: int, rule: LabelRule, rule_config: dict[str, Any]) -> w19._FittedBackend:
    """Backend 4 (constant) at `rule`'s labelling.

    `width_wsel22a._fit_constant_at`'s body, generalized via `_global_capacity_for_rule` (COPIED with provenance).
    """
    c_star = _global_capacity_for_rule(error_table_sel, sigma_sel, rule)

    def route_index(x: np.ndarray) -> np.ndarray:
        return np.full(len(x), c_star, dtype=int)

    def route_proba(x: np.ndarray) -> np.ndarray:
        proba = np.zeros((len(x), w_max), dtype=np.float64)
        proba[:, c_star] = 1.0
        return proba

    return w19._FittedBackend(
        route_index_fn=route_index, route_proba_fn=route_proba, config={"selected_capacity_index": c_star, "selected_width": c_star + 1, **rule_config}, sizing_rule=None
    )


def _fit_backend_at_rule(
    backend: w19.Backend, x_sel: np.ndarray, error_table_sel: np.ndarray, sigma_sel: np.ndarray, flops_by_width: list[float], seed: int, w_max: int, in_dim: int, rule: LabelRule
) -> w19._FittedBackend:
    """Dispatches to the requested backend's fit-at-labels function under `rule`'s labelling."""
    rule_config = {"rule": rule.kind.value, "rule_param": rule.param}
    if backend is w19.Backend.CONSTANT:
        return _fit_constant_at_labels(error_table_sel, sigma_sel, w_max, rule, rule_config)
    labels = rule.labels(error_table_sel, sigma_sel)
    if backend is w19.Backend.FROZEN_MLP:
        return _fit_frozen_mlp_at_labels(x_sel, flops_by_width, seed, w_max, labels, rule_config)
    if backend is w19.Backend.RULE_MLP:
        return _fit_rule_mlp_at_labels(x_sel, labels, seed, w_max, rule_config, in_dim=in_dim)
    return _fit_xgboost_at_labels(x_sel, labels, seed, w_max, rule_config)


def _label_churn_for_rule(error_table: np.ndarray, rule: LabelRule, sigma: np.ndarray, baseline_tolerance: float = w22a._BASELINE_TOLERANCE) -> dict[str, Any]:
    """Fraction of rows whose `rule`-labelled capacity differs from the flat `baseline_tolerance` labelling.

    Same error table -- generalizes `width_wsel22a._label_churn` to an arbitrary `LabelRule`.
    """
    labels_rule = rule.labels(error_table, sigma)
    labels_baseline = _cheapest_within_tolerance_labels(error_table, tolerance=baseline_tolerance)
    churned = labels_rule != labels_baseline
    return {
        "rule": rule.kind.value,
        "rule_param": rule.param,
        "baseline_tolerance": baseline_tolerance,
        "n_rows": len(labels_rule),
        "n_churned": int(churned.sum()),
        "churn_fraction": float(churned.mean()),
        "mean_label_shift": float(np.mean(labels_rule.astype(np.int64) - labels_baseline.astype(np.int64))),
    }


# ---------------------------------------------------------------------------
# Impact cells (spec SS4(i)/SS6 steps 1/3/4) -- one router refit per (backend, n_sel, seed/triple) at the
# sigma-anchored labelling, ONLY at `_DEFAULT_CONFIDENCE` (spec SS4(i): "at c=0.95"). Fits ONCE, lands BOTH
# modes (mirrors `width_wsel22a.run_1d_cell`/`run_d2_cell` -- see module docstring's ambiguity note 2).
# ---------------------------------------------------------------------------


def run_1d_impact_cell(seed: int, n_sel: int, backend: w19.Backend, confidence: float, *, results_dir: str = RESULTS_DIR) -> tuple[dict[str, Any], dict[str, Any]]:
    """Refits ONE (backend, n_sel, seed) 1-D cell at the sigma-anchored labelling `confidence`; returns (hard_case, blend_case).

    The flat-tolerance comparison points are NEVER recomputed here (spec SS6 step 2) -- `summarize()` reads
    WSEL22A's own frozen ledger for the flat-0.25 baseline.
    """
    w22a._verify_1d_cache_exists(seed)
    cache = w19._get_or_build_sweep_cache(seed, results_dir=w22a._WSEL19_RESULTS_DIR)
    w_max = cache.error_table_pool.shape[1]

    x_sel = cache.x_pool[:n_sel]
    error_table_sel = cache.error_table_pool[:n_sel]
    in_dim = 1 if x_sel.ndim == 1 else x_sel.shape[1]
    sigma_sel = _sigma_true(n_sel)
    sigma_report = _sigma_true(len(cache.x_report))
    rule = LabelRule(RuleKind.SIGMA_ANCHORED, confidence)

    fitted = _fit_backend_at_rule(backend, x_sel, error_table_sel, sigma_sel, cache.flops_by_width, seed, w_max, in_dim, rule)
    churn = _label_churn_for_rule(error_table_sel, rule, sigma_sel)

    cases = []
    for mode in w19.Mode:
        readout = w19._SCORERS[mode](fitted, cache.x_report, cache.error_table_report, cache.flops_by_width, w_max)
        if mode is w19.Mode.HARD:
            idx = np.asarray(fitted.route_index_fn(cache.x_report), dtype=int)
            rule_labels_report = rule.labels(cache.error_table_report, sigma_report)
            readout = {**readout, "rule_agreement": float(np.mean(idx == rule_labels_report))}
        case = {
            "slice": w22a.Slice.ONE_D.value,
            "backend": backend.value,
            "mode": mode.value,
            "n_sel": n_sel,
            "seed": seed,
            "tolerance": confidence,  # reuses width_wsel22a's group-key/finding-function machinery verbatim (see module docstring) -- a sigma confidence here, never a flat tau.
            "rule": rule.kind.value,
            "backend_config": fitted.config,
            "sizing_rule": fitted.sizing_rule,
            "label_churn": churn,
            "provenance": run_provenance(),
        }
        case.update(readout)
        cases.append(case)
        path = os.path.join(results_dir, f"wsel22b_impact_1d_{backend.value}_{mode.value}_nsel{n_sel}_seed{seed}_c{confidence}.json")
        os.makedirs(results_dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump(w19._jsonable(case), f, indent=2)
    return cases[0], cases[1]


def run_d2_impact_cell(
    geometry: wt.Geometry,
    seed: int,
    n_train: int,
    n_sel: int,
    backend: w19.Backend,
    confidence: float,
    *,
    d: int = w22a._MF_D_TARGET,
    w_max: int = w19.W_MAX,
    results_dir: str = RESULTS_DIR,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Refits ONE (geometry, seed, n_train, n_sel, backend) d=2 cell at the sigma-anchored labelling `confidence`; returns (hard_case, blend_case)."""
    w22a._verify_mf_cache_complete(d, geometry, seed, n_train, w_max)
    models, _metas, cache_hit, n_train_used = w19._get_or_build_mf_models(seed, d, geometry, w_max=w_max, n_train=n_train, results_dir=w22a._WSEL19_RESULTS_DIR)
    if not cache_hit:
        raise SystemExit(f"d=2 {geometry.value} seed={seed} n_train={n_train}: not every width was a cache hit -- refusing (this task never trains a per-width net).")

    x_report, y_report, _region_report, _t_report = wt.make_report_split(seed, d, geometry)
    x_sel, y_sel, _region_sel, _t_sel = wt.make_selection_split(seed, d, geometry, n_sel)
    error_table_report = w19._mf_error_table(models, x_report, y_report, w_max)
    error_table_sel = w19._mf_error_table(models, x_sel, y_sel, w_max)
    flops_by_width = [float(executed_flops(models[w].model, w)) for w in range(1, w_max + 1)]

    sigma_sel = _sigma_true(n_sel)
    sigma_report = _sigma_true(len(x_report))
    rule = LabelRule(RuleKind.SIGMA_ANCHORED, confidence)

    fitted = _fit_backend_at_rule(backend, x_sel, error_table_sel, sigma_sel, flops_by_width, seed, w_max, d, rule)
    churn = _label_churn_for_rule(error_table_sel, rule, sigma_sel)

    cases = []
    for mode in w19.Mode:
        readout = w19._SCORERS[mode](fitted, x_report, error_table_report, flops_by_width, w_max)
        if mode is w19.Mode.HARD:
            idx = np.asarray(fitted.route_index_fn(x_report), dtype=int)
            rule_labels_report = rule.labels(error_table_report, sigma_report)
            readout = {**readout, "rule_agreement": float(np.mean(idx == rule_labels_report))}
        case = {
            "slice": w22a.Slice.D2.value,
            "d": d,
            "geometry": geometry.value,
            "backend": backend.value,
            "mode": mode.value,
            "n_sel": n_sel,
            "seed": seed,
            "n_train": n_train_used,
            "tolerance": confidence,
            "rule": rule.kind.value,
            "backend_config": fitted.config,
            "sizing_rule": fitted.sizing_rule,
            "label_churn": churn,
            "provenance": run_provenance(),
        }
        case.update(readout)
        cases.append(case)
        path = os.path.join(results_dir, f"wsel22b_impact_d2_{geometry.value}_{backend.value}_{mode.value}_nsel{n_sel}_seed{seed}_c{confidence}.json")
        os.makedirs(results_dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump(w19._jsonable(case), f, indent=2)
    return cases[0], cases[1]


# ---------------------------------------------------------------------------
# Oracle cells (spec SS5, root-amended M1/M2 protocol + SS5.5's A1 coverage diagnostic) -- NO backend refit;
# a pure function of the (noisy, generator-true) REPORT-split error tables and the known sigma. One cell per
# decided (seed) [1-D] or (geometry, seed, n_train) [d=2] -- never per (backend, mode, n_sel).
# ---------------------------------------------------------------------------


def _row_best_chi2_ratio(error_table: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """`e_i(w*_i) / sigma_i**2` per row -- the SS5.5 A1 diagnostic's test statistic (row-best relative to its claimed chi2_1 null)."""
    row_best_error = error_table.min(axis=1)
    return row_best_error / (np.asarray(sigma, dtype=np.float64) ** 2)


def _a1_coverage_diagnostic(error_table: np.ndarray, sigma: np.ndarray, confidences: tuple[float, ...] = CONFIDENCE_LEVELS) -> list[dict[str, float]]:
    """Spec SS5.5: `(nominal_c, empirical_coverage)` pairs.

    Under A1 (the row-best-is-unbiased approximation), `e_i(w*_i)/sigma_i**2 ~ chi2_1`, so the empirical
    fraction of rows at/under each `c`'s chi2 quantile should match `c` itself; systematic under-coverage
    flags row-best residual bias at this cell (spec SS2.5(A1)). NON-GATING (informational diagnostic only).
    """
    ratio = _row_best_chi2_ratio(error_table, sigma)
    return [{"nominal_c": c, "empirical_coverage": float(np.mean(ratio <= _chi2_quantile(c)))} for c in confidences]


def _oracle_agreement(noisy_table: np.ndarray, true_table: np.ndarray, rule: LabelRule, sigma: np.ndarray) -> dict[str, float]:
    """M1 (noise-robustness, self-referenced) and M2 (truth-tracking, rule-free) for one labelling `rule` -- spec SS5.2 as amended."""
    labels_noisy = rule.labels(noisy_table, sigma)
    labels_true_self = rule.labels(true_table, sigma)
    oracle_argmin = true_table.argmin(axis=1)
    return {
        "m1_noise_robustness": float(np.mean(labels_noisy == labels_true_self)),
        "m2_truth_tracking": float(np.mean(labels_noisy == oracle_argmin)),
    }


def _agreement_at_every_rule(noisy_table: np.ndarray, true_table: np.ndarray, sigma: np.ndarray) -> dict[str, dict[str, float]]:
    """M1/M2 at the flat-0.25 baseline plus every sigma-anchored confidence in `CONFIDENCE_LEVELS` (spec SS5.3)."""
    agreement = {"flat_0.25": _oracle_agreement(noisy_table, true_table, LabelRule(RuleKind.FLAT_TOLERANCE, w22a._BASELINE_TOLERANCE), sigma)}
    for c in CONFIDENCE_LEVELS:
        agreement[f"sigma_{c}"] = _oracle_agreement(noisy_table, true_table, LabelRule(RuleKind.SIGMA_ANCHORED, c), sigma)
    return agreement


def _load_1d_models(seed: int, w_max: int = w19.W_MAX) -> dict[int, Any]:
    """Reloads the `w_max` already-cached tier-1 per-width nets for `seed` from WSEL6's cache.

    The SAME state dicts `width_wsel19._get_or_build_sweep_cache` reads, exposed here as raw model objects
    (needed to score the generator-true target `_hetero_h`, which that cache's own cached `error_table_report`
    never computes). Verified on disk by the spec (SS5.4): 12 state dicts x 3 seeds, all present -- refuses
    rather than trains if any are missing.
    """
    models: dict[int, Any] = {}
    for width in range(1, w_max + 1):
        state_path, meta_path = w6._sweep_cache_paths(w19._WSEL6_RESULTS_DIR, w6.Tier.ONE, seed, width)
        if not (os.path.exists(state_path) and os.path.exists(meta_path)):
            raise SystemExit(f"{state_path} (or its meta) is missing -- WSEL-22(b)'s oracle cell only reads WSEL-6's cached tier-1 nets, it never trains one.")
        models[width] = w6._load_cached_model((width,), seed, state_path, max_epochs=w4.PORTED_N_EPOCHS_CAP, patience=w4.PORTED_PATIENCE, lr=w4.PORTED_LR_DEFAULT)
    return models


def run_1d_oracle_cell(seed: int, *, results_dir: str = RESULTS_DIR) -> dict[str, Any]:
    """Spec SS5 oracle-agreement (M1/M2) + SS5.5 A1-coverage for one 1-D decided seed, REPORT split only -- NO backend refit.

    Spec SS5.1: for the 1-D toy, `t` IS `x` directly -- `_hetero_h` is evaluated on `cache.x_report` unmodified.
    """
    w22a._verify_1d_cache_exists(seed)
    cache = w19._get_or_build_sweep_cache(seed, results_dir=w22a._WSEL19_RESULTS_DIR)
    w_max = cache.error_table_report.shape[1]
    models = _load_1d_models(seed, w_max=w_max)
    x_report_in = cache.x_report.reshape(-1, 1).astype(np.float32)
    true_error_table_report = w19._mf_error_table(models, x_report_in, w19._hetero_h(cache.x_report), w_max)
    sigma_report = _sigma_true(len(cache.x_report))

    case = {
        "slice": w22a.Slice.ONE_D.value,
        "seed": seed,
        "n_report": len(cache.x_report),
        "agreement": _agreement_at_every_rule(cache.error_table_report, true_error_table_report, sigma_report),
        "a1_coverage": _a1_coverage_diagnostic(cache.error_table_report, sigma_report),
        "provenance": run_provenance(),
    }
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"wsel22b_oracle_1d_seed{seed}.json")
    with open(path, "w") as f:
        json.dump(w19._jsonable(case), f, indent=2)
    return case


def run_d2_oracle_cell(geometry: wt.Geometry, seed: int, n_train: int, *, d: int = w22a._MF_D_TARGET, w_max: int = w19.W_MAX, results_dir: str = RESULTS_DIR) -> dict[str, Any]:
    """Spec SS5 oracle-agreement (M1/M2) + SS5.5 A1-coverage for one d=2 decided triple, REPORT split only -- NO backend refit."""
    w22a._verify_mf_cache_complete(d, geometry, seed, n_train, w_max)
    models, _metas, cache_hit, n_train_used = w19._get_or_build_mf_models(seed, d, geometry, w_max=w_max, n_train=n_train, results_dir=w22a._WSEL19_RESULTS_DIR)
    if not cache_hit:
        raise SystemExit(f"d=2 {geometry.value} seed={seed} n_train={n_train}: not every width was a cache hit -- refusing (this task never trains a per-width net).")

    x_report, y_report, _region_report, t_report = wt.make_report_split(seed, d, geometry)
    noisy_error_table_report = w19._mf_error_table(models, x_report, y_report, w_max)
    true_error_table_report = w19._mf_error_table(models, x_report, w19._hetero_h(t_report), w_max)
    sigma_report = _sigma_true(len(x_report))

    case = {
        "slice": w22a.Slice.D2.value,
        "d": d,
        "geometry": geometry.value,
        "seed": seed,
        "n_train": n_train_used,
        "n_report": len(x_report),
        "agreement": _agreement_at_every_rule(noisy_error_table_report, true_error_table_report, sigma_report),
        "a1_coverage": _a1_coverage_diagnostic(noisy_error_table_report, sigma_report),
        "provenance": run_provenance(),
    }
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"wsel22b_oracle_d2_{geometry.value}_seed{seed}.json")
    with open(path, "w") as f:
        json.dump(w19._jsonable(case), f, indent=2)
    return case


# ---------------------------------------------------------------------------
# --summarize -- the SS4 win-only adoption verdict per confidence level, the SS5.5 A1-coverage table, and the
# SS4(i)/SS6-step-4 non-gating verdict-stability impact assessment at the default confidence. Frozen to a NEW
# ledger (WSEL22A/frozen.json stays untouched, SS6 step 6).
# ---------------------------------------------------------------------------


def _win_verdict(oracle_cells: list[dict[str, Any]], confidence: float) -> dict[str, Any]:
    """Spec SS5.3-as-amended win criterion at one confidence `c`: pooled paired-difference 2*SE win/no-worse, plus the no-group-loss clause.

    Groups are `"1d"` (all 1-D seeds) and `"d2:<geometry>"` (each d=2 geometry's decided triples) -- the SAME
    grouping WSEL22A's own `per_group_1d`/`per_group_d2` tables use. "Aggregate over decided groups" (SS5.3)
    is read as the POOLED per-seed paired difference across every decided cell; the no-group-loss clause is
    checked per GROUP separately. Condition 2 ("not worse... on the OTHER metric") is implemented as "neither
    metric is worse by more than 2*SE" -- a strictly conservative superset of "the other metric" when only one
    of M1/M2 clears condition 1, and equivalent to it otherwise; flagged here as the resolution of a genuine
    textual ambiguity, not asserted as the only possible reading.
    """
    sigma_key = f"sigma_{confidence}"
    groups: dict[str, list[dict[str, float]]] = {}
    for cell in oracle_cells:
        group = cell["slice"] if cell["slice"] == w22a.Slice.ONE_D.value else f"{cell['slice']}:{cell['geometry']}"
        flat = cell["agreement"]["flat_0.25"]
        sigma = cell["agreement"][sigma_key]
        diff = {"m1": sigma["m1_noise_robustness"] - flat["m1_noise_robustness"], "m2": sigma["m2_truth_tracking"] - flat["m2_truth_tracking"]}
        groups.setdefault(group, []).append(diff)

    per_group = {group: {"m1": w19._mean_se([d["m1"] for d in diffs]), "m2": w19._mean_se([d["m2"] for d in diffs]), "n": len(diffs)} for group, diffs in groups.items()}
    pooled_m1 = w19._mean_se([d["m1"] for diffs in groups.values() for d in diffs])
    pooled_m2 = w19._mean_se([d["m2"] for diffs in groups.values() for d in diffs])

    m1_better = pooled_m1["mean"] > _SE_MARGIN * pooled_m1["se"]
    m2_better = pooled_m2["mean"] > _SE_MARGIN * pooled_m2["se"]
    m1_worse = pooled_m1["mean"] < -_SE_MARGIN * pooled_m1["se"]
    m2_worse = pooled_m2["mean"] < -_SE_MARGIN * pooled_m2["se"]
    condition_1_better_on_one_metric = m1_better or m2_better
    condition_2_not_worse_other_metric = not (m1_worse or m2_worse)

    group_losses = {
        group: {"m1_loss": bool(stats["m1"]["mean"] < -_SE_MARGIN * stats["m1"]["se"]), "m2_loss": bool(stats["m2"]["mean"] < -_SE_MARGIN * stats["m2"]["se"])}
        for group, stats in per_group.items()
    }
    condition_3_no_group_loss = not any(v["m1_loss"] or v["m2_loss"] for v in group_losses.values())

    wins = bool(condition_1_better_on_one_metric and condition_2_not_worse_other_metric and condition_3_no_group_loss)
    return {
        "confidence": confidence,
        "pooled_m1_diff": pooled_m1,
        "pooled_m2_diff": pooled_m2,
        "per_group_diff": per_group,
        "group_losses": group_losses,
        "condition_1_better_on_one_metric": condition_1_better_on_one_metric,
        "condition_2_not_worse_other_metric": condition_2_not_worse_other_metric,
        "condition_3_no_group_loss": condition_3_no_group_loss,
        "wins": wins,
    }


def _verdict_stability_impact(per_group_1d: dict[str, Any], per_group_d2: dict[str, Any], confidence: float, flat_baseline: dict[str, Any]) -> dict[str, Any]:
    """Spec SS4(i)/SS6 step 4: recomputes (a)'s 20 verdict-stability findings under sigma-anchored labels at `confidence`.

    Diffed against WSEL22A's OWN frozen flat-0.25 findings (`flat_baseline`, read never recomputed -- SS6 step
    2). NON-GATING (SS4 as amended): a reported impact assessment, never adoption evidence.
    """
    entry: dict[str, Any] = {}
    for n_sel in w19.N_SEL_VALUES:
        entry[f"1d_constant_wins_quality_nsel{n_sel}"] = w22a._finding_constant_wins_quality(per_group_1d, "1d", n_sel, confidence)
        entry[f"1d_frozen_more_robust_than_rule_mlp_nsel{n_sel}"] = w22a._finding_frozen_more_robust_than_rule_mlp(per_group_1d, "1d", n_sel, confidence)
        for backend in w22a._BLEND_COMPARABLE_BACKENDS:
            entry[f"1d_blend_dominated_by_hard_{backend.value}_nsel{n_sel}"] = w22a._finding_blend_dominated_by_hard(per_group_1d, "1d", backend, n_sel, confidence)
    for geometry in wt.Geometry:
        prefix = f"d2:{geometry.value}"
        for n_sel in w19.N_SEL_VALUES:
            entry[f"d2_{geometry.value}_constant_wins_quality_nsel{n_sel}"] = w22a._finding_constant_wins_quality(per_group_d2, prefix, n_sel, confidence)
            entry[f"d2_{geometry.value}_frozen_more_robust_than_rule_mlp_nsel{n_sel}"] = w22a._finding_frozen_more_robust_than_rule_mlp(per_group_d2, prefix, n_sel, confidence)
            for backend in w22a._BLEND_COMPARABLE_BACKENDS:
                key = f"d2_{geometry.value}_blend_dominated_by_hard_{backend.value}_nsel{n_sel}"
                entry[key] = w22a._finding_blend_dominated_by_hard(per_group_d2, prefix, backend, n_sel, confidence)

    flipped = {key: entry[key] != flat_baseline[key] for key in entry if key in flat_baseline and entry[key] is not None and flat_baseline[key] is not None}
    return {"sigma_findings": entry, "flat_0.25_baseline": flat_baseline, "flipped": flipped, "any_flip": any(flipped.values())}


def summarize(results_dir: str = RESULTS_DIR) -> None:
    """Aggregates every landed oracle + impact cell under `results_dir` into `frozen.json`.

    Spec SS4 win-only adoption verdict (per confidence level), SS5.5 A1-coverage per decided cell, and the
    SS4(i)/SS6-step-4 verdict-stability impact assessment at `_DEFAULT_CONFIDENCE` (non-gating; diffed
    against WSEL22A's own frozen flat-0.25 findings, read never recomputed).
    """
    oracle_cells = w22a._load_cells(results_dir, "wsel22b_oracle_1d_seed") + w22a._load_cells(results_dir, "wsel22b_oracle_d2_")
    win_by_confidence = {str(c): _win_verdict(oracle_cells, c) for c in CONFIDENCE_LEVELS}

    impact_cells_1d = w22a._load_cells(results_dir, "wsel22b_impact_1d_")
    impact_cells_d2 = w22a._load_cells(results_dir, "wsel22b_impact_d2_")
    per_group_1d = w22a._aggregate_per_group(impact_cells_1d)
    per_group_d2 = w22a._aggregate_per_group(impact_cells_d2)

    with open(os.path.join(w22a.RESULTS_DIR, "frozen.json")) as f:
        wsel22a_frozen = json.load(f)
    flat_baseline = wsel22a_frozen["verdict_stability"][str(w22a._BASELINE_TOLERANCE)]
    verdict_stability_impact = _verdict_stability_impact(per_group_1d, per_group_d2, _DEFAULT_CONFIDENCE, flat_baseline)

    a1_coverage_by_cell = {(str(c["seed"]) if c["slice"] == w22a.Slice.ONE_D.value else f"{c['geometry']}_seed{c['seed']}"): c["a1_coverage"] for c in oracle_cells}
    adopted_at_default_confidence = win_by_confidence[str(_DEFAULT_CONFIDENCE)]["wins"]

    frozen = {
        "confidence_levels_swept": list(CONFIDENCE_LEVELS),
        "default_confidence": _DEFAULT_CONFIDENCE,
        "flat_reference_tolerance": w22a._BASELINE_TOLERANCE,
        "n_oracle_cells": len(oracle_cells),
        "n_impact_cells_1d": len(impact_cells_1d),
        "n_impact_cells_d2": len(impact_cells_d2),
        "decided_d2_triples": [{"geometry": g.value, "seed": s, "n_train": n} for g, s, n in w22a.decided_d2_triples()],
        "oracle_agreement_win_by_confidence": win_by_confidence,
        "adopted": adopted_at_default_confidence,
        "adoption_gate": "spec SS4 as amended: win-only on generator-true oracle agreement (SS5.3) at c=0.95, tie retains the flat-0.25 incumbent.",
        "a1_coverage_by_cell": a1_coverage_by_cell,
        "per_group_1d": per_group_1d,
        "per_group_d2": per_group_d2,
        "verdict_stability_impact_at_default_confidence": verdict_stability_impact,
        "provenance": run_provenance(),
    }
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "frozen.json"), "w") as f:
        json.dump(w19._jsonable(frozen), f, indent=2)
    print(
        f"[wsel22b summarize] wrote {os.path.join(results_dir, 'frozen.json')}: {len(oracle_cells)} oracle cells, "
        f"{len(impact_cells_1d)} 1-D impact cells, {len(impact_cells_d2)} d=2 impact cells. adopted_at_c={_DEFAULT_CONFIDENCE}={adopted_at_default_confidence}"
    )


# ---------------------------------------------------------------------------
# --selftest -- spec SS9's four pre-registered assertions + a tiny synthetic M1/M2 fixture, plus an A1-coverage
# sanity check and a backend-fit wiring smoke check. No real cells, no cache reads (SS9) -- every table below
# is synthetic, built in this function.
# ---------------------------------------------------------------------------


def run_selftest() -> bool:
    """Exercises the sigma-anchored labelling function and the M1/M2 oracle-agreement machinery against synthetic fixtures only."""
    ok = True

    # (1) The row-best always passes its own band trivially (excess=0 <= any non-negative band), at every confidence.
    error_table = np.array([[1.20, 1.00, 4.00], [1.00, 5.00, 1.00], [3.00, 3.00, 3.10]])
    sigma = np.array([0.05, 0.10, 0.05])
    row_best_ok = True
    for confidence in CONFIDENCE_LEVELS:
        acceptable = _sigma_anchored_acceptable(error_table, sigma, confidence)
        argmin_col = error_table.argmin(axis=1)
        row_best_ok = row_best_ok and bool(np.all(acceptable[np.arange(len(error_table)), argmin_col]))
    ok = ok and row_best_ok
    print(f"[wsel22b selftest] (1) row-best trivially passes its own band at every confidence: {'PASS' if row_best_ok else 'FAIL'}")

    # (2) Monotonicity in c: a larger c never excludes a width the same row accepted at a smaller c.
    monotone_ok = True
    for c_lo, c_hi in itertools.pairwise(sorted(CONFIDENCE_LEVELS)):
        acc_lo = _sigma_anchored_acceptable(error_table, sigma, c_lo)
        acc_hi = _sigma_anchored_acceptable(error_table, sigma, c_hi)
        monotone_ok = monotone_ok and bool(np.all(~acc_lo | acc_hi))
    ok = ok and monotone_ok
    print(f"[wsel22b selftest] (2) monotone in confidence: {'PASS' if monotone_ok else 'FAIL'}")

    # (3) sigma -> 0 degenerate limit: only the strict row-best is acceptable, at any confidence.
    zero_sigma = np.zeros(len(error_table))
    degenerate_labels = _sigma_anchored_labels(error_table, zero_sigma, _DEFAULT_CONFIDENCE)
    degenerate_ok = bool(np.array_equal(degenerate_labels, error_table.argmin(axis=1)))
    ok = ok and degenerate_ok
    print(f"[wsel22b selftest] (3) sigma->0 degenerate limit == strict argmin: {'PASS' if degenerate_ok else 'FAIL'}")

    # (4) Byte-identical agreement with `_cheapest_within_tolerance_labels`'s existing flat-tolerance path.
    flat_rule = LabelRule(RuleKind.FLAT_TOLERANCE, w22a._BASELINE_TOLERANCE)
    dispatch_labels = flat_rule.labels(error_table, sigma)
    direct_labels = _cheapest_within_tolerance_labels(error_table, tolerance=w22a._BASELINE_TOLERANCE)
    dispatch_ok = bool(np.array_equal(dispatch_labels, direct_labels))
    ok = ok and dispatch_ok
    print(f"[wsel22b selftest] (4a) LabelRule(FLAT_TOLERANCE) byte-identical to _cheapest_within_tolerance_labels: {'PASS' if dispatch_ok else 'FAIL'}")

    global_flat = _global_capacity_for_rule(error_table, sigma, flat_rule)
    global_direct = w19._global_cheapest_within_tolerance(error_table, w22a._BASELINE_TOLERANCE)
    global_ok = global_flat == global_direct
    ok = ok and global_ok
    print(f"[wsel22b selftest] (4b) _global_capacity_for_rule(FLAT_TOLERANCE) byte-identical to _global_cheapest_within_tolerance: {'PASS' if global_ok else 'FAIL'}")

    # (5) A tiny synthetic M1/M2 fixture, cross-checked against an independent brute-force per-row recompute.
    noisy_table = np.array([[1.0, 1.2, 5.0], [2.0, 2.0, 2.0], [5.0, 1.0, 1.1], [3.0, 3.1, 0.5]])
    true_table = np.array([[1.0, 1.05, 5.0], [2.0, 2.0, 2.0], [1.0, 1.0, 1.1], [3.0, 3.1, 0.5]])
    sigma_fixture = np.full(4, 1.0)

    def _brute_force_flat_label(row: np.ndarray, tolerance: float) -> int:
        best = row.min()
        for j, v in enumerate(row):
            if v <= (1.0 + tolerance) * best:
                return j
        raise AssertionError("unreachable -- the row minimum always satisfies its own tolerance band.")

    expected_noisy = np.array([_brute_force_flat_label(row, w22a._BASELINE_TOLERANCE) for row in noisy_table])
    expected_true_self = np.array([_brute_force_flat_label(row, w22a._BASELINE_TOLERANCE) for row in true_table])
    expected_true_argmin = true_table.argmin(axis=1)
    expected_m1 = float(np.mean(expected_noisy == expected_true_self))
    expected_m2 = float(np.mean(expected_noisy == expected_true_argmin))

    got = _oracle_agreement(noisy_table, true_table, flat_rule, sigma_fixture)
    fixture_ok = math.isclose(got["m1_noise_robustness"], expected_m1) and math.isclose(got["m2_truth_tracking"], expected_m2)
    ok = ok and fixture_ok
    print(f"[wsel22b selftest] (5) M1/M2 fixture: got={got} expected=(m1={expected_m1}, m2={expected_m2})  {'PASS' if fixture_ok else 'FAIL'}")

    # (6) A1-coverage sanity: as sigma -> 0 (with every row error > 0), coverage collapses to 0 at every c < 1.
    a1 = _a1_coverage_diagnostic(noisy_table, np.full(4, 1e-6))
    a1_ok = all(pair["empirical_coverage"] == 0.0 for pair in a1)
    ok = ok and a1_ok
    print(f"[wsel22b selftest] (6) A1 coverage collapses to 0 as sigma->0 (row errors > 0): {'PASS' if a1_ok else 'FAIL'}")

    # (7) Bonus wiring smoke check: every backend fits and scores finitely on a tiny synthetic pool, at a sigma-anchored rule.
    rng = np.random.default_rng(0)
    n = 40
    x_sel = rng.uniform(-1, 1, size=n).astype(np.float32)
    synth_error_table = np.stack([np.abs(x_sel) + 0.3, np.abs(x_sel) * 0.1], axis=1).astype(np.float64)
    flops_by_width = [1.0, 4.0]
    x_report = rng.uniform(-1, 1, size=20).astype(np.float32)
    report_error_table = np.stack([np.abs(x_report) + 0.3, np.abs(x_report) * 0.1], axis=1).astype(np.float64)
    sigma_sel = _sigma_true(n)
    sigma_rule = LabelRule(RuleKind.SIGMA_ANCHORED, _DEFAULT_CONFIDENCE)

    wiring_ok = True
    for backend in w19.Backend:
        fitted = _fit_backend_at_rule(backend, x_sel, synth_error_table, sigma_sel, flops_by_width, seed=0, w_max=2, in_dim=1, rule=sigma_rule)
        for mode in w19.Mode:
            readout = w19._SCORERS[mode](fitted, x_report, report_error_table, flops_by_width, 2)
            finite_ok = math.isfinite(readout["routed_held_out_quality"]) and math.isfinite(readout["mean_deployed_flops"])
            wiring_ok = wiring_ok and finite_ok
            if not finite_ok:
                print(f"[wsel22b selftest] FAIL: backend={backend.value} mode={mode.value} produced a non-finite readout: {readout}")
    ok = ok and wiring_ok
    print(f"[wsel22b selftest] (7) every backend fits+scores finitely at a sigma-anchored rule: {'PASS' if wiring_ok else 'FAIL'}")

    print(f"[wsel22b selftest] {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# main -- a true one-cell-per-process CLI (spec SS9's flag list; see module docstring's ambiguity note 2).
# ---------------------------------------------------------------------------


def main() -> None:
    """Dispatches to `--selftest`/`--summarize`, one SS5 oracle cell (`--oracle`), or one SS4(i)/SS6 impact cell."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument(
        "--oracle", action="store_true", help="SS5 M1/M2 oracle-agreement + SS5.5 A1-coverage cell for one decided cell (ignores --backend/--mode/--n-sel/--confidence)."
    )
    parser.add_argument("--backend", choices=[b.value for b in w19.Backend], default=None)
    parser.add_argument(
        "--mode",
        choices=[m.value for m in w19.Mode],
        default=None,
        help="Impact cells always fit once and land BOTH hard and blend JSONs (width_wsel22a.py's convention); this only selects which case's summary line is printed.",
    )
    parser.add_argument("--n-sel", type=int, choices=list(w19.N_SEL_VALUES), default=None)
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for this cell (canonical suite: 0, 1, 2).")
    parser.add_argument("--geometry", choices=[g.value for g in wt.Geometry], default=None, help="d=2 slice (requires --n-train). Omit for the 1-D slice.")
    parser.add_argument("--n-train", type=int, default=None, help="Required with --geometry -- must match one of WSEL22A's decided (geometry, seed, n_train) triples.")
    parser.add_argument("--confidence", type=float, default=_DEFAULT_CONFIDENCE, help=f"Sigma-anchored confidence level for an impact cell -- one of {CONFIDENCE_LEVELS}.")
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR)
    args = parser.parse_args()

    if sum([args.selftest, args.summarize]) > 1:
        parser.error("--selftest and --summarize are mutually exclusive.")
    if args.selftest:
        sys.exit(0 if run_selftest() else 1)
    if args.summarize:
        summarize(results_dir=args.results_dir)
        return

    if (args.geometry is None) != (args.n_train is None):
        parser.error("--geometry and --n-train must be given together (or both omitted for the 1-D slice).")
    if args.seed is None:
        parser.error("--seed is required for a real cell (or pass --selftest / --summarize).")
    os.makedirs(args.results_dir, exist_ok=True)

    if args.oracle:
        if args.geometry is not None:
            case = run_d2_oracle_cell(wt.Geometry(args.geometry), args.seed, args.n_train, results_dir=args.results_dir)
            print(
                f"[wsel22b oracle d2] geometry={args.geometry} seed={args.seed} n_train={case['n_train']} "
                f"m1(0.95)={case['agreement']['sigma_0.95']['m1_noise_robustness']:.4g} m2(0.95)={case['agreement']['sigma_0.95']['m2_truth_tracking']:.4g}"
            )
        else:
            case = run_1d_oracle_cell(args.seed, results_dir=args.results_dir)
            print(
                f"[wsel22b oracle 1d] seed={args.seed} "
                f"m1(0.95)={case['agreement']['sigma_0.95']['m1_noise_robustness']:.4g} m2(0.95)={case['agreement']['sigma_0.95']['m2_truth_tracking']:.4g}"
            )
        return

    if args.backend is None or args.n_sel is None:
        parser.error("--backend and --n-sel are required for an impact cell (or pass --selftest / --summarize / --oracle).")
    if args.confidence not in CONFIDENCE_LEVELS:
        parser.error(f"--confidence must be one of {CONFIDENCE_LEVELS}, got {args.confidence}.")

    backend = w19.Backend(args.backend)
    if args.geometry is not None:
        hard_case, blend_case = run_d2_impact_cell(wt.Geometry(args.geometry), args.seed, args.n_train, args.n_sel, backend, args.confidence, results_dir=args.results_dir)
    else:
        hard_case, blend_case = run_1d_impact_cell(args.seed, args.n_sel, backend, args.confidence, results_dir=args.results_dir)
    selected = blend_case if args.mode == w19.Mode.BLEND.value else hard_case
    print(
        f"[wsel22b impact] backend={backend.value} n_sel={args.n_sel} seed={args.seed} confidence={args.confidence} "
        f"quality={selected['routed_held_out_quality']:.4g} churn={selected['label_churn']['churn_fraction']:.3f}"
    )


if __name__ == "__main__":
    main()
