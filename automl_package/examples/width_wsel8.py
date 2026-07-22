"""WSEL-8 -- the W-SHARED ~= W-SWEEP claim, both halves, on toys (`docs/plans/capacity_programme/width.md` ~1052-1081).

`width.md` SS1 draws a line between two DIFFERENT claims about the cheap global read:
  * **(a) same quality** -- read at a given width, does the jointly-trained multi-head dial network
    match a network DEDICATED to that width? (partially covered by the certification's fit-at-floor
    evidence, never against the converged per-width sweep).
  * **(b) same choice** -- does the cheap read pick the width the sweep would pick? Never tested
    anywhere before this task.
This driver measures both, on the ONE toy/seed/width ladder WSEL-4 already certified as a usable
W-SWEEP reference: `hetero`, seeds `(0, 1, 2)`, widths `1..12` -- read straight off
`automl_package/examples/capacity_ladder_results/WSEL4/`'s landed per-cell JSONs (top-level files
only; the `relu_confounded_run/`/`tanh_minibatch_run/` subdirectories are QUARANTINED confounded
runs, per that task's own result block, and are never read by anything here -- `glob.glob` without
`**` already skips them, no explicit exclusion needed). No cell is invented.

Two arms per (toy, seed), reusing existing code rather than rewriting it:
  * `--arm w_sweep` -- ONE dedicated `FlexibleWidthNN(widths=(width,))` net, trained under WSEL-4's
    OWN ported protocol (Tanh, full-batch, `width_wsel4.PORTED_LR_DEFAULT`/`PORTED_PATIENCE`/
    `PORTED_MIN_DELTA`/`PORTED_N_EPOCHS_CAP`, raw x/y) -- via `width_wsel6._get_or_train`, which
    already trains and caches exactly this net (tier 1 = `hetero` = plain MSE = WSEL-4's ported
    protocol verbatim; `width_wsel6.py`'s own module docstring). Nothing about the protocol is
    re-derived here.
  * `--arm w_shared` -- the certified multi-head `FlexibleWidthNN(widths=1..w_max)`, trained ONCE
    per (toy, seed) by a LOCAL joint-training loop (see below for why this cannot reuse
    `width_wsel6._train_tier1`), then read two ways: `fit_global_selector` (W-SHARED's own chosen
    width) and, from the ALREADY-CACHED W-SWEEP nets this same driver trained, the sweep's held-out
    error curve fed to the SAME `cheapest_within_tolerance` selector (`width_wsel6._run_w_sweep`'s
    exact wiring, never re-derived) -- so both arms' chosen widths come from one rule, per SS1: "the
    same rule applies to W-SWEEP's curve, or W-SHARED and W-SWEEP are not answering the same
    question."

**MASTER Decision 14 hard gate, enforced MECHANICALLY, not by prose.** The known-good arm (W-SWEEP,
reproducing WSEL-4's reference) must run first, alone, and pass before any W-SHARED number is read.
`--wsweep-control` compares every landed `w_sweep` cell's `held_out_mse` against WSEL-4's own landed
reference cell for the same (toy, seed, width) and writes `WSEL8/wsweep_control.json` with a
per-cell `relative_error` and a top-level `reproduces` bool, at the SAME 2% bar WSEL-4 used on
itself. `--arm w_shared` REFUSES to run (`SystemExit`, exit code 1) unless that file exists on disk
with `reproduces: true` (`_assert_wsweep_control_passed`) -- see SS3.5's WSEL-4 branch: "the ported
sweep does not reproduce -> halt WSEL-8."

**Anchor discipline (SS3.6's anchor warning).** The quality-at-matched-width table below is built
from THIS task's own freshly-trained, trajectory-verified W-SWEEP nets -- never from the
certification's own historical numbers, and never from the WSEL-4 reference cells its own result
block flagged `untrustworthy_seeds`. Re-deriving an anchor with the same code would conscript the
worker into confirming its own bug; pairing against an INDEPENDENTLY-trained-here dedicated net is
the whole point of running W-SWEEP inside this task rather than reading WSEL-4's stored MSEs
directly for the comparison (WSEL-4's numbers are used ONLY for the `--wsweep-control` positive
control, never for the quality comparison itself).

**The corrected joint-training stop rule (SS3.5 point 5 -- why `width_wsel6._train_tier1` is NOT
reused for the shared net).** `FlexibleWidthNN._fit_single` (the generic package bypass
`width_wsel6._train_tier1` calls) tracks convergence on ONE aggregate scalar -- the SUMMED loss
across every configured width. Under joint training, one width's held-out loss can still be
drifting while another's cancels it out in the sum, so a flat AGGREGATE does not certify every width
individually flat -- exactly the failure `width_wsel13._train_all_schedule_weighted_to_convergence`
diagnosed and fixed for its own (different-architecture, tier-2-weighted) net: a naive
`all(tracker.done for ...)` LATCHES per width, so an early-latched width can keep drifting while the
shared trunk keeps moving to serve widths still training. `_train_shared_to_convergence` below
mirrors that fix on `FlexibleWidthNNModule` instead: ONE `ConvergenceTracker` per width, held-out
loss checked every epoch (matching the ported protocol's own per-epoch granularity, no
`check_every`), stop only once EVERY width's tracker is simultaneously `trustworthy` or `diverged`
AT THE SAME checkpoint -- never merely once each has EVER latched `done`. Unweighted MSE, not
`width_candidates.weighted_squared_error`: every cell here is tier 1 (`hetero`, one constant sigma),
where plain MSE already IS the SS3.7 fixed-sigma likelihood up to a constant scale.

**The three-way split (SS3.5's closing line: "the reported numbers come from a split not used for
stopping or selection").** Reused verbatim from `width_wsel6._build_split` (tier 1 = `hetero`) --
the SAME split object for BOTH arms (MASTER Decision 15's single-difference rule):
  * FIT   -- `split["x_train"]`/`y_train` (phase-1, even indices): trains every net.
  * STOP  -- `split["x_p1_val"]`/`y_p1_val` (every `VAL_EVERY`-th FIT point): convergence monitoring
    ONLY, for both the per-width sweep nets and the shared net's per-width trackers.
  * SELECT -- a WSEL-6-frozen-fraction subsample of `split["x_p2"]`/`y_p2` (phase-2, odd indices,
    shuffled once, seeded): `fit_global_selector`'s read AND W-SWEEP's `cheapest_within_tolerance`
    error table. Both arms read the SAME subsample -- SS1's "same seeds throughout" requirement.
  * REPORT -- `split["x_test"]`/`y_test` (a disjoint held-out draw, `seed + 500`): quality-at-
    matched-width and the two arms' own reported `held_out_mse`. Touched by neither training nor
    selection.

**The frozen selection fraction is READ, never hardcoded** (SS3.6 feed-forward rule): from
`automl_package/examples/capacity_ladder_results/WSEL6/frozen.json`'s `fraction`/`fraction_pct`
keys, with the source path and value recorded under the output cell's `selection.fraction_source`.

Per-cell CLI (one cell per invocation -- the root runs the grid, never this driver):
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel8.py \
        --toy hetero --seed 0 --width 5 --arm w_sweep
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel8.py \
        --wsweep-control
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel8.py \
        --toy hetero --seed 0 --arm w_shared

Aggregate, after every (toy, seed) cell has landed:
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/width_wsel8.py --summarize

Selftest / lint:
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel8.py --selftest

Non-goals (task brief): no real data or baselines (WSEL-9's budget, parked); no re-tuning of the
selector; no changes to `width_wsel4.py`/`width_wsel6.py`/`width_wsel7.py`/`kdropout_converged_width_
experiment.py` or any package module -- this file only IMPORTS and mirrors their protocols.
"""

from __future__ import annotations

import argparse
import enum
import glob
import json
import math
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root, so `import automl_package` works

import converged_width_experiment as cwe  # noqa: E402 -- W_MAX/N_TRAIN/N_TEST/SEEDS, the canonical hetero ladder
import convergence as cvg  # noqa: E402
import nested_width_net as nwn  # noqa: E402 -- Toy
import width_wsel4 as w4  # noqa: E402 -- the ported protocol's constants (PORTED_*), reused verbatim
import width_wsel6 as w6  # noqa: E402 -- _build_split/_selection_subsample/_get_or_train(_sweep_all)/_sweep_cache_paths, reused verbatim

from automl_package.enums import ActivationFunction, CapacitySelection, TaskType  # noqa: E402
from automl_package.models.flexnn.width.model import FlexibleWidthNN  # noqa: E402
from automl_package.utils.capacity_accounting import global_cheap_cost, held_out_read_cost, sweep_cost  # noqa: E402
from automl_package.utils.capacity_selection import DEFAULT_N_BOOT, cheapest_within_tolerance  # noqa: E402
from automl_package.utils.run_provenance import run_provenance  # noqa: E402

RESULTS_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "WSEL8")
WSEL4_RESULTS_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "WSEL4")
WSEL6_FROZEN_PATH = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "WSEL6", "frozen.json")

_DRIVER_NAME = "automl_package/examples/width_wsel8.py"
TOY = nwn.Toy.HETERO  # the only toy with a landed WSEL-4 reference (WSEL-4's own scope restriction)
SEEDS = cwe.SEEDS  # (0, 1, 2)
W_MAX = cwe.W_MAX  # 12 -- verified against WSEL4/hetero_0_1_control.json's own config.w_max
N_TRAIN = cwe.N_TRAIN  # 1500
N_TEST = cwe.N_TEST  # 500

REPRODUCTION_BAR = 0.02  # mirrors WSEL-4's own reproduction bar (width_wsel4.REPRODUCTION_BAR)
_REL_ERR_EPS = 1e-12  # floors the relative-error denominator so a near-zero reference cannot divide-by-zero


class Arm(enum.Enum):
    """Closed set: `width.md` SS1's two untested-together arms -- WSEL-8's whole comparison."""

    W_SWEEP = "w_sweep"  # one dedicated FlexibleWidthNN(widths=(k,)) per width -- the expensive reference.
    W_SHARED = "w_shared"  # the certified multi-head dial network, read via fit_global_selector.


# ---------------------------------------------------------------------------
# W-SWEEP -- one dedicated per-width net, reusing width_wsel6's own get-or-train/cache machinery
# (tier 1 = hetero = plain MSE = the wsel4 ported protocol verbatim). No convergence-rule mirroring
# is needed here: each dedicated net trains ONE width alone, so there is no "joint" series to
# reconcile -- `width_wsel6._get_or_train`'s existing single-tracker replay is already correct for it.
# ---------------------------------------------------------------------------


def _run_w_sweep_cell(
    seed: int,
    width: int,
    w_max: int,
    *,
    results_dir: str,
    n_train: int,
    n_test: int,
    max_epochs: int,
    patience: int = w4.PORTED_PATIENCE,
    min_delta: float = w4.PORTED_MIN_DELTA,
    lr: float = w4.PORTED_LR_DEFAULT,
) -> dict[str, Any]:
    """Trains (or loads, cache HIT) ONE dedicated `FlexibleWidthNN(widths=(width,))`, scores it on REPORT.

    Reuses `width_wsel6._build_split`/`_sweep_cache_paths`/`_get_or_train` verbatim -- this IS the
    W-SWEEP dedicated net for `width`, trained under WSEL-4's own ported protocol, with no code here
    re-deriving any of it.
    """
    split = w6._build_split(w6.Tier.ONE, seed, n_train=n_train, n_test=n_test)
    state_path, meta_path = w6._sweep_cache_paths(results_dir, w6.Tier.ONE, seed, width)
    model, meta, cache_hit = w6._get_or_train(
        (width,), w6.Tier.ONE, seed, split, state_path, meta_path, max_epochs=max_epochs, patience=patience, min_delta=min_delta, lr=lr
    )

    x_test_in = split["x_test"].reshape(-1, 1).astype(np.float32)
    pred = model.predict(x_test_in, filter_data=False, width=width)
    held_out_mse = float(np.mean((pred - split["y_test"]) ** 2))

    return {
        "toy": TOY.value,
        "seed": seed,
        "width": width,
        "w_max": w_max,
        "arm": Arm.W_SWEEP.value,
        "held_out_mse": held_out_mse,
        "held_out_trajectory": meta["trajectory"],
        "actual_epochs": meta["actual_epochs"],
        "n_train_used": meta["n_train_used"],
        "training_macs": meta["training_macs"],
        "trustworthy": meta["trustworthy"],
        "hit_cap": meta["hit_cap"],
        "objective": meta["objective"],
        "cache_hit": bool(cache_hit),
        "config": {"n_train": n_train, "n_test": n_test, "lr": lr, "patience": patience, "min_delta": min_delta, "max_epochs_cap": max_epochs},
        "provenance": {"driver": _DRIVER_NAME, "reused_from": "width_wsel6._get_or_train (tier 1, wsel4 ported protocol)", **run_provenance()},
    }


# ---------------------------------------------------------------------------
# W-SHARED -- the certified multi-head net, trained by a LOCAL joint-training loop implementing the
# corrected per-width SIMULTANEOUS-flat stop rule (see module docstring; mirrors
# `width_wsel13._train_all_schedule_weighted_to_convergence`'s fix, adapted to `FlexibleWidthNNModule`
# and to tier 1's unweighted MSE).
# ---------------------------------------------------------------------------


def _train_shared_to_convergence(
    widths: tuple[int, ...],
    seed: int,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    max_epochs: int,
    patience: int,
    min_delta: float,
    lr: float,
) -> tuple[FlexibleWidthNN, dict[int, cvg.ConvergenceResult]]:
    """Joint multi-width training for W-SHARED, gated by PER-WIDTH simultaneous-flat convergence.

    `FlexibleWidthNNModule.forward` already sums every configured width every step (the class's own
    ALL-schedule shape -- see that module's docstring), so the only thing missing from the generic
    `_fit_single` bypass is a stop rule that certifies every width's held-out trajectory individually
    rather than one aggregate scalar. This reproduces `width_wsel13._train_all_schedule_weighted_to_
    convergence`'s corrected rule: stop only when every width's tracker is SIMULTANEOUSLY trustworthy
    (patience-flat, not still creeping, not hit-cap) or diverged AT THE SAME checkpoint -- never
    merely once each has EVER latched `done` (that latched rule lets an early-flat width's held-out
    loss keep drifting while the shared trunk keeps moving to serve widths still training). Held-out
    loss is checked every epoch, matching the ported protocol's own per-epoch granularity (no
    `check_every` -- `width_wsel4`/`width_wsel6`'s ported-arm trainers validate every epoch too).

    Returns:
        `(model, {width -> ConvergenceResult})`, best (lowest mean-per-width-val) weights restored.
    """
    model = FlexibleWidthNN(
        input_size=1,
        output_size=1,
        task_type=TaskType.REGRESSION,
        widths=widths,
        learning_rate=lr,
        n_epochs=max_epochs,
        early_stopping_rounds=patience,
        batch_size=w4.PORTED_BATCH_SIZE,
        random_seed=seed,
        calculate_feature_importance=False,
        capacity_selection=CapacitySelection.FIXED,
        # Tanh, not the class's ReLU default -- width_wsel4.py's confound-doctrine finding, reused
        # verbatim (the control classes this whole protocol reproduces are Tanh).
        activation=ActivationFunction.TANH,
    )
    model.input_size = 1
    torch.manual_seed(seed)
    model.build_model()
    opt = torch.optim.Adam(model.model.parameters(), lr=lr)

    x_tr_t = torch.tensor(x_tr, dtype=torch.float32, device=model.device).reshape(-1, 1)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=model.device)
    x_val_t = torch.tensor(x_val, dtype=torch.float32, device=model.device).reshape(-1, 1)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=model.device)

    trackers = {w: cvg.ConvergenceTracker(patience=patience, min_delta=min_delta) for w in widths}
    best_mean_val = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    final_epoch = max_epochs

    model.model.train()
    for epoch in range(1, max_epochs + 1):
        opt.zero_grad()
        stacked = model.model(x_tr_t)  # (len(widths), N, 1) -- every width, every step (the class's ALL schedule).
        total = stacked.new_zeros(())
        for i in range(stacked.shape[0]):
            total = total + ((stacked[i].squeeze(1) - y_tr_t) ** 2).mean()
        total.backward()
        opt.step()

        model.model.eval()
        with torch.no_grad():
            stacked_val = model.model(x_val_t)
            per_width_val = {w: float(((stacked_val[i].squeeze(1) - y_val_t) ** 2).mean().item()) for i, w in enumerate(widths)}
        for w, v in per_width_val.items():
            trackers[w].update(epoch, v)
        mean_val = sum(per_width_val.values()) / len(widths)
        if mean_val < best_mean_val:
            best_mean_val = mean_val
            best_state = {name: t.detach().clone() for name, t in model.model.state_dict().items()}
        model.model.train()

        # STOP RULE -- corrected: simultaneously flat-or-diverged NOW, not merely ever-latched `done`
        # (see module docstring / width_wsel13._train_all_schedule_weighted_to_convergence).
        results_now = {w: trackers[w].result(final_epoch=epoch) for w in widths}
        if all(r.trustworthy or r.diverged for r in results_now.values()):
            final_epoch = epoch
            break

    if best_state is not None:
        model.model.load_state_dict(best_state)
    model.model.eval()
    return model, {w: trackers[w].result(final_epoch=final_epoch) for w in widths}


def _read_frozen_fraction(wsel6_frozen_path: str = WSEL6_FROZEN_PATH) -> tuple[float, int, dict[str, Any]]:
    """Reads WSEL-6's frozen selection fraction, with provenance -- never hardcoded (SS3.6 feed-forward rule)."""
    if not os.path.exists(wsel6_frozen_path):
        raise RuntimeError(
            f"WSEL-6 frozen fraction not found at {wsel6_frozen_path!r}; width.md SS3.6's feed-forward rule requires "
            "WSEL-8 to read the selection-set fraction from that artifact -- run WSEL-6 (or point --results-dir "
            "wiring at a build that has) before running --arm w_shared."
        )
    with open(wsel6_frozen_path) as f:
        wsel6_frozen = json.load(f)
    fraction = float(wsel6_frozen["fraction"])
    fraction_pct = int(wsel6_frozen["fraction_pct"]) if "fraction_pct" in wsel6_frozen else round(fraction * 100)
    provenance = {"source": os.path.abspath(wsel6_frozen_path), "fraction": fraction, "fraction_pct": fraction_pct}
    return fraction, fraction_pct, provenance


def _run_w_shared_cell(
    seed: int,
    w_max: int,
    *,
    results_dir: str,
    n_train: int,
    n_test: int,
    max_epochs: int,
    patience: int = w4.PORTED_PATIENCE,
    min_delta: float = w4.PORTED_MIN_DELTA,
    lr: float = w4.PORTED_LR_DEFAULT,
    wsel6_frozen_path: str = WSEL6_FROZEN_PATH,
) -> dict[str, Any]:
    """Builds the FINAL per-(toy, seed) cell: trains W-SHARED, reloads the cached W-SWEEP nets, compares both.

    Caller's contract: every width's `--arm w_sweep` cell must already have landed (and cached its
    state_dict) for this `(toy, seed)` -- enforced by `all_cache_hit` below, not silently retrained
    (retraining here would defeat the "one dedicated net per width" cost accounting W-SWEEP prices).
    """
    widths = tuple(range(1, w_max + 1))
    split = w6._build_split(w6.Tier.ONE, seed, n_train=n_train, n_test=n_test)

    sweep_models, sweep_metas, all_cache_hit = w6._get_or_train_sweep_all(
        w6.Tier.ONE, seed, split, w_max, results_dir=results_dir, max_epochs=max_epochs, patience=patience, min_delta=min_delta, lr=lr
    )
    if not all_cache_hit:
        raise RuntimeError(
            f"missing cached W-SWEEP net(s) for (toy={TOY.value}, seed={seed}) under {results_dir!r}; "
            f"run --arm w_sweep --width W for every width 1..{w_max} first."
        )

    shared_model, per_width_results = _train_shared_to_convergence(
        widths, seed, split["x_train"], split["y_train"], split["x_p1_val"], split["y_p1_val"], max_epochs=max_epochs, patience=patience, min_delta=min_delta, lr=lr
    )
    hit_cap_shared = any(r.hit_cap for r in per_width_results.values())
    trustworthy_shared = all(r.trustworthy for r in per_width_results.values())
    trajectory_shared = {str(w): [[int(e), float(v)] for e, v in r.trajectory] for w, r in per_width_results.items()}
    actual_epochs_shared = max(r.stop_epoch for r in per_width_results.values())
    n_train_used = len(split["x_train"])

    # REPORT split -- untouched by stopping or selection (SS3.5's closing line). Scored while the
    # shared net is still CapacitySelection.FIXED, so an explicit `width=` is legal on both arms.
    x_test_in = split["x_test"].reshape(-1, 1).astype(np.float32)
    y_test = split["y_test"]
    quality_at_matched_width = []
    for w in widths:
        shared_mse = float(np.mean((shared_model.predict(x_test_in, filter_data=False, width=w) - y_test) ** 2))
        sweep_mse = float(np.mean((sweep_models[w].predict(x_test_in, filter_data=False, width=w) - y_test) ** 2))
        quality_at_matched_width.append(
            {
                "width": w,
                "w_shared_report_mse": shared_mse,
                "w_sweep_report_mse": sweep_mse,
                "ratio_shared_over_sweep": (shared_mse / sweep_mse) if sweep_mse else None,
            }
        )

    # SELECT split -- WSEL-6's frozen fraction of the SAME p2 pool both arms read (SS1's "same seeds
    # throughout" requirement); feeds BOTH selectors.
    fraction, fraction_pct, fraction_provenance = _read_frozen_fraction(wsel6_frozen_path)
    x_sel, y_sel, _region_sel = w6._selection_subsample(split, fraction_pct)
    x_sel_in = x_sel.reshape(-1, 1).astype(np.float32)

    w_shared_width = shared_model.fit_global_selector(x_sel_in, y_sel, seed=seed)
    shared_model.capacity_selection = CapacitySelection.GLOBAL_CHEAP
    w_shared_held_out_mse = float(np.mean((shared_model.predict(x_test_in, filter_data=False) - y_test) ** 2))

    # W-SWEEP's held-out error curve, the SAME selector (`width_wsel6._run_w_sweep`'s exact wiring).
    error_table = np.stack([(sweep_models[w].predict(x_sel_in, filter_data=False, width=w) - y_sel) ** 2 for w in widths], axis=1)
    sweep_idx = cheapest_within_tolerance(error_table, n_boot=DEFAULT_N_BOOT, seed=seed)
    w_sweep_width = widths[sweep_idx]
    w_sweep_held_out_mse = float(np.mean((sweep_models[w_sweep_width].predict(x_test_in, filter_data=False, width=w_sweep_width) - y_test) ** 2))

    # Cost accounting (WSEL-5 primitives; width_wsel6's own wiring, never re-derived).
    shared_training_macs = sweep_cost(shared_model.model, list(widths), n_train_used, actual_epochs_shared)
    shared_cost = global_cheap_cost(training_macs=shared_training_macs, net=shared_model.model, capacity_grid=list(widths), n_samples=len(x_sel))
    sweep_training_macs = sum(sweep_metas[w]["training_macs"] for w in widths)
    sweep_selection_macs = sum(held_out_read_cost(sweep_models[w].model, [w], len(x_sel)) for w in widths)

    hit_cap = bool(hit_cap_shared or any(sweep_metas[w]["hit_cap"] for w in widths))
    trustworthy = bool(trustworthy_shared and all(sweep_metas[w]["trustworthy"] for w in widths))

    return {
        "toy": TOY.value,
        "seed": seed,
        "w_max": w_max,
        "w_shared_width": int(w_shared_width),
        "w_sweep_width": int(w_sweep_width),
        "agreement": {
            "widths_match": bool(w_shared_width == w_sweep_width),
            "width_diff": int(w_shared_width - w_sweep_width),
            "abs_width_diff": int(abs(w_shared_width - w_sweep_width)),
        },
        "quality": {"w_shared_held_out_mse": w_shared_held_out_mse, "w_sweep_held_out_mse": w_sweep_held_out_mse},
        "quality_at_matched_width": quality_at_matched_width,
        "ceiling_bound": {"w_shared": bool(w_shared_width == w_max), "w_sweep": bool(w_sweep_width == w_max)},
        "held_out_trajectory": {"w_shared": trajectory_shared, "w_sweep": {str(w): sweep_metas[w]["trajectory"] for w in widths}},
        "hit_cap": hit_cap,
        "trustworthy": trustworthy,
        "selection_cost": {
            "w_shared": {"training_macs": shared_cost.training_macs, "selection_macs": shared_cost.selection_macs, "total_macs": shared_cost.total_macs},
            "w_sweep": {"training_macs": sweep_training_macs, "selection_macs": sweep_selection_macs, "total_macs": sweep_training_macs + sweep_selection_macs},
        },
        "selection": {"fraction": fraction, "fraction_pct": fraction_pct, "n_selection_used": len(x_sel), "fraction_source": fraction_provenance},
        "global_selector_curve": shared_model.global_selector_curve_,
        "sweep_all_cache_hit": bool(all_cache_hit),
        "provenance": run_provenance(),
    }


# ---------------------------------------------------------------------------
# --wsweep-control -- MASTER Decision 14's positive-control gate. Compares THIS task's freshly-
# trained W-SWEEP cells against WSEL-4's own landed reference cells at the SAME 2% bar WSEL-4 used.
# ---------------------------------------------------------------------------


def _load_w_sweep_cells(results_dir: str, toy_value: str, seeds: Sequence[int]) -> dict[tuple[int, int], dict[str, Any]]:
    cells: dict[tuple[int, int], dict[str, Any]] = {}
    for path in sorted(glob.glob(os.path.join(results_dir, f"{toy_value}_*_w_sweep.json"))):
        with open(path) as f:
            cell = json.load(f)
        if int(cell["seed"]) in seeds:
            cells[(int(cell["seed"]), int(cell["width"]))] = cell
    return cells


def _load_wsel4_reference_cells(reference_dir: str, toy_value: str, seeds: Sequence[int]) -> dict[tuple[int, int], dict[str, Any]]:
    """Reads WSEL-4's own landed PORTED cells (top-level files only -- quarantined subdirs never glob-match)."""
    cells: dict[tuple[int, int], dict[str, Any]] = {}
    for path in sorted(glob.glob(os.path.join(reference_dir, f"{toy_value}_*.json"))):
        name = os.path.basename(path)
        if name == "reproduction.json" or name.endswith("_control.json"):
            continue
        with open(path) as f:
            cell = json.load(f)
        if cell.get("arm") != w4.Arm.PORTED.value:
            continue
        if int(cell["seed"]) in seeds:
            cells[(int(cell["seed"]), int(cell["width"]))] = cell
    return cells


def run_wsweep_control(
    results_dir: str = RESULTS_DIR,
    reference_dir: str = WSEL4_RESULTS_DIR,
    *,
    toy_value: str = TOY.value,
    seeds: Sequence[int] = SEEDS,
    w_max: int = W_MAX,
) -> dict[str, Any]:
    """Builds `wsweep_control.json`: per-cell relative error + the top-level `reproduces` gate."""
    ours = _load_w_sweep_cells(results_dir, toy_value, seeds)
    reference = _load_wsel4_reference_cells(reference_dir, toy_value, seeds)
    expected = {(seed, width) for seed in seeds for width in range(1, w_max + 1)}

    cells = []
    for seed, width in sorted(expected & set(ours) & set(reference)):
        c, r = ours[(seed, width)], reference[(seed, width)]
        rel_err = abs(c["held_out_mse"] - r["held_out_mse"]) / (abs(r["held_out_mse"]) + _REL_ERR_EPS)
        cells.append(
            {
                "toy": toy_value,
                "seed": seed,
                "width": width,
                "w_sweep_held_out_mse": c["held_out_mse"],
                "wsel4_reference_held_out_mse": r["held_out_mse"],
                "relative_error": rel_err,
                "within_bar": bool(rel_err <= REPRODUCTION_BAR),
            }
        )

    missing = sorted(expected - set(ours))
    complete = not missing
    all_within_bar = complete and bool(cells) and all(cell["within_bar"] for cell in cells)
    reproduces = bool(complete and all_within_bar)

    summary = {
        "bar": REPRODUCTION_BAR,
        "n_expected": len(expected),
        "n_landed": len(ours),
        "complete": complete,
        "missing_cells": [{"seed": s, "width": w} for s, w in missing],
        "all_within_bar": all_within_bar,
        "reproduces": reproduces,
        "cells": cells,
        "reference_dir": os.path.abspath(reference_dir),
        "provenance": run_provenance(),
    }
    if not reproduces:
        print("!" * 78)
        print("WSWEEP CONTROL DID NOT REPRODUCE (MASTER Decision 14 / width.md WSEL-8 hard gate):")
        print(f"  complete={complete} (missing={len(missing)}), all_within_bar={all_within_bar}")
        print("  --arm w_shared will refuse to run until this passes. Report the discrepancy (SS3.5).")
        print("!" * 78)

    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "wsweep_control.json")
    with open(out_path, "w") as f:
        json.dump(_jsonable(summary), f, indent=2)
    return summary


def _assert_wsweep_control_passed(results_dir: str) -> None:
    """Mechanical hard gate (SS3.6's anchor warning + MASTER Decision 14): refuse, don't warn."""
    path = os.path.join(results_dir, "wsweep_control.json")
    if not os.path.exists(path):
        raise SystemExit(
            f"{path} is missing. width.md's WSEL-8 hard gate (MASTER Decision 14): run --wsweep-control after "
            "every W-SWEEP dedicated net has landed, BEFORE any --arm w_shared cell."
        )
    with open(path) as f:
        control = json.load(f)
    if control.get("reproduces") is not True:
        raise SystemExit(
            f"{path} has reproduces={control.get('reproduces')!r}. --arm w_shared refuses to run until the W-SWEEP "
            "positive control reproduces WSEL-4's reference within its bar (width.md SS3.5: report the discrepancy and halt)."
        )


# ---------------------------------------------------------------------------
# _jsonable -- local twin of every sibling WSEL driver's helper.
# ---------------------------------------------------------------------------


def _jsonable(obj: object) -> object:
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
# --summarize -- WSEL8/frozen.json: the quality-at-matched-width table (mean over seeds) + the
# agreement table (both halves of the SS1 claim, per the task brief).
# ---------------------------------------------------------------------------


def summarize(results_dir: str = RESULTS_DIR, *, toy_value: str = TOY.value) -> None:
    """Aggregates every landed (toy, seed) cell into `WSEL8/frozen.json` (quality table + agreement table)."""
    cells = []
    for path in sorted(glob.glob(os.path.join(results_dir, f"{toy_value}_*.json"))):
        name = os.path.basename(path)
        if name.endswith("_w_sweep.json") or name in ("wsweep_control.json", "frozen.json"):
            continue
        with open(path) as f:
            cells.append(json.load(f))

    by_width: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        for row in cell["quality_at_matched_width"]:
            by_width[row["width"]].append(row)

    quality_table = []
    for width in sorted(by_width):
        rows = by_width[width]
        ratios = [r["ratio_shared_over_sweep"] for r in rows if r["ratio_shared_over_sweep"] is not None]
        quality_table.append(
            {
                "width": width,
                "n_seeds": len(rows),
                "mean_w_shared_report_mse": float(np.mean([r["w_shared_report_mse"] for r in rows])),
                "mean_w_sweep_report_mse": float(np.mean([r["w_sweep_report_mse"] for r in rows])),
                "mean_ratio_shared_over_sweep": float(np.mean(ratios)) if ratios else None,
            }
        )

    agreement_table = [
        {"seed": cell["seed"], "w_shared_width": cell["w_shared_width"], "w_sweep_width": cell["w_sweep_width"], **cell["agreement"]} for cell in cells
    ]
    n_agree = sum(1 for cell in cells if cell["agreement"]["widths_match"])

    frozen = {
        "toy": toy_value,
        "seeds_present": sorted(cell["seed"] for cell in cells),
        "n_cells_present": len(cells),
        "quality_at_matched_width": quality_table,
        "agreement": agreement_table,
        "agreement_rate": (n_agree / len(cells)) if cells else None,
        "any_untrustworthy": any(not cell["trustworthy"] for cell in cells),
        "any_hit_cap": any(cell["hit_cap"] for cell in cells),
        "provenance": run_provenance(),
    }
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "frozen.json")
    with open(out_path, "w") as f:
        json.dump(_jsonable(frozen), f, indent=2)
    print(f"[width_wsel8] wrote {out_path} ({frozen['n_cells_present']} cells, agreement_rate={frozen['agreement_rate']})")


# ---------------------------------------------------------------------------
# Selftest -- tiny w_max=3 toy, one seed, every mode, in a throwaway tmp dir. Never a real cell.
# The wsweep-control comparison uses a SYNTHETIC reference (an exact copy of our own freshly-trained
# numbers, written into its own tmp subdir) rather than the REAL WSEL4 dir, so the KNOWN answer
# (`reproduces: True`) is checkable without depending on production artifacts landing first.
# ---------------------------------------------------------------------------


def run_selftest() -> bool:
    """Tiny end-to-end pass: every mode, the hard gate (both refuse and pass), a known-answer reproduction check."""
    tmp_dir = tempfile.mkdtemp(prefix="width_wsel8_selftest_")
    try:
        w_max = 3
        seed = 0
        n_train, n_test = 60, 30
        max_epochs, patience, min_delta, lr = 40, 3, 1e-2, 0.05

        synthetic_ref_dir = os.path.join(tmp_dir, "wsel4_ref")
        os.makedirs(synthetic_ref_dir, exist_ok=True)

        ok = True
        for width in range(1, w_max + 1):
            cell = _run_w_sweep_cell(seed, width, w_max, results_dir=tmp_dir, n_train=n_train, n_test=n_test, max_epochs=max_epochs, patience=patience, min_delta=min_delta, lr=lr)
            with open(os.path.join(tmp_dir, f"{TOY.value}_{seed}_{width}_w_sweep.json"), "w") as f:
                json.dump(_jsonable(cell), f, indent=2)
            with open(os.path.join(synthetic_ref_dir, f"{TOY.value}_{seed}_{width}.json"), "w") as f:
                json.dump({"toy": TOY.value, "seed": seed, "width": width, "arm": w4.Arm.PORTED.value, "held_out_mse": cell["held_out_mse"]}, f, indent=2)

            cell_ok = math.isfinite(cell["held_out_mse"]) and isinstance(cell["trustworthy"], bool) and isinstance(cell["hit_cap"], bool)
            ok = ok and cell_ok
            print(f"[wsel8 selftest] w_sweep width={width} mse={cell['held_out_mse']:.4g} cache_hit={cell['cache_hit']}  {'PASS' if cell_ok else 'FAIL'}")

        control = run_wsweep_control(results_dir=tmp_dir, reference_dir=synthetic_ref_dir, toy_value=TOY.value, seeds=(seed,), w_max=w_max)
        control_ok = control["reproduces"] is True and control["complete"] and len(control["cells"]) == w_max
        ok = ok and control_ok
        print(f"[wsel8 selftest] wsweep_control (known-answer, self-copied reference) reproduces={control['reproduces']}  {'PASS' if control_ok else 'FAIL'}")

        # Known-answer gate check: MUST refuse before a control has landed in a fresh dir.
        empty_dir = os.path.join(tmp_dir, "no_control_yet")
        os.makedirs(empty_dir, exist_ok=True)
        gate_refused = False
        try:
            _assert_wsweep_control_passed(empty_dir)
        except SystemExit:
            gate_refused = True
        ok = ok and gate_refused
        print(f"[wsel8 selftest] hard gate refuses without wsweep_control.json: {'PASS' if gate_refused else 'FAIL'}")

        _assert_wsweep_control_passed(tmp_dir)  # must NOT raise -- the control landed above with reproduces=True.

        wsel6_frozen_path = os.path.join(tmp_dir, "wsel6_frozen.json")
        with open(wsel6_frozen_path, "w") as f:
            json.dump({"fraction": 0.4, "fraction_pct": 40}, f)

        shared_cell = _run_w_shared_cell(
            seed,
            w_max,
            results_dir=tmp_dir,
            n_train=n_train,
            n_test=n_test,
            max_epochs=max_epochs,
            patience=patience,
            min_delta=min_delta,
            lr=lr,
            wsel6_frozen_path=wsel6_frozen_path,
        )
        with open(os.path.join(tmp_dir, f"{TOY.value}_{seed}.json"), "w") as f:
            json.dump(_jsonable(shared_cell), f, indent=2)

        shared_ok = (
            1 <= shared_cell["w_shared_width"] <= w_max
            and 1 <= shared_cell["w_sweep_width"] <= w_max
            and isinstance(shared_cell["hit_cap"], bool)
            and isinstance(shared_cell["trustworthy"], bool)
            and len(shared_cell["quality_at_matched_width"]) == w_max
            and shared_cell["selection_cost"]["w_shared"]["total_macs"] > 0
            and shared_cell["selection_cost"]["w_sweep"]["total_macs"] > 0
        )
        ok = ok and shared_ok
        print(f"[wsel8 selftest] w_shared cell: w_shared_width={shared_cell['w_shared_width']} w_sweep_width={shared_cell['w_sweep_width']}  {'PASS' if shared_ok else 'FAIL'}")

        summarize(results_dir=tmp_dir, toy_value=TOY.value)
        with open(os.path.join(tmp_dir, "frozen.json")) as f:
            frozen = json.load(f)
        summarize_ok = frozen["n_cells_present"] == 1 and len(frozen["quality_at_matched_width"]) == w_max and "agreement_rate" in frozen
        ok = ok and summarize_ok
        print(f"[wsel8 selftest] summarize: n_cells={frozen['n_cells_present']}  {'PASS' if summarize_ok else 'FAIL'}")

        print(f"[wsel8 selftest] {'PASS' if ok else 'FAIL'}")
        return ok
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Runs one per-cell training (`--arm w_sweep`/`w_shared`), `--wsweep-control`, `--summarize`, or `--selftest`."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true", help="Tiny end-to-end wiring check (both arms + the hard gate), then exit.")
    parser.add_argument("--summarize", action="store_true", help="Aggregate landed (toy, seed) cells into WSEL8/frozen.json, then exit.")
    parser.add_argument("--wsweep-control", action="store_true", help="MASTER Decision 14 positive control: writes WSEL8/wsweep_control.json, then exit.")

    parser.add_argument("--toy", choices=[TOY.value], default=TOY.value, help="Only hetero has a landed WSEL-4 reference (WSEL-8's scope).")
    parser.add_argument("--seed", type=int, default=None, help="Required outside --selftest/--summarize/--wsweep-control.")
    parser.add_argument("--width", type=int, default=None, help="Required for --arm w_sweep; must not be given for --arm w_shared.")
    parser.add_argument("--arm", choices=[a.value for a in Arm], default=None, help="Required outside --selftest/--summarize/--wsweep-control.")
    parser.add_argument("--w-max", type=int, default=W_MAX)
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--n-test", type=int, default=N_TEST)
    parser.add_argument("--epoch-cap", type=int, default=w4.PORTED_N_EPOCHS_CAP, help="Safety cap on training epochs for whichever net(s) this cell trains.")
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR)

    args = parser.parse_args()

    modes = (args.selftest, args.summarize, args.wsweep_control)
    if sum(bool(m) for m in modes) > 1:
        parser.error("--selftest, --summarize and --wsweep-control are mutually exclusive.")
    if args.selftest:
        sys.exit(0 if run_selftest() else 1)
    if args.summarize:
        summarize(results_dir=args.results_dir)
        return
    if args.wsweep_control:
        control = run_wsweep_control(results_dir=args.results_dir)
        print(
            f"[width_wsel8] wrote wsweep_control.json (reproduces={control['reproduces']}, "
            f"{control['n_landed']}/{control['n_expected']} landed, all_within_bar={control['all_within_bar']})"
        )
        return

    if args.seed is None or args.arm is None:
        parser.error("--seed and --arm are required for a real cell (or pass --selftest / --summarize / --wsweep-control).")
    arm = Arm(args.arm)
    os.makedirs(args.results_dir, exist_ok=True)

    if arm is Arm.W_SWEEP:
        if args.width is None:
            parser.error("--width is required for --arm w_sweep.")
        cell = _run_w_sweep_cell(args.seed, args.width, args.w_max, results_dir=args.results_dir, n_train=args.n_train, n_test=args.n_test, max_epochs=args.epoch_cap)
        if not cell["trustworthy"]:
            print(f"*** DO-NOT-CONCLUDE GUARD: w_sweep seed={args.seed} width={args.width} did NOT converge trustworthily (hit_cap={cell['hit_cap']}). ***")
        out_path = os.path.join(args.results_dir, f"{args.toy}_{args.seed}_{args.width}_w_sweep.json")
        with open(out_path, "w") as f:
            json.dump(_jsonable(cell), f, indent=2)
        print(f"[width_wsel8] wrote {out_path} (held_out_mse={cell['held_out_mse']:.6f} trustworthy={cell['trustworthy']} hit_cap={cell['hit_cap']})")
    else:
        if args.width is not None:
            parser.error("--width does not apply to --arm w_shared (the multi-head net spans the whole ladder).")
        _assert_wsweep_control_passed(args.results_dir)
        cell = _run_w_shared_cell(args.seed, args.w_max, results_dir=args.results_dir, n_train=args.n_train, n_test=args.n_test, max_epochs=args.epoch_cap)
        if not cell["trustworthy"]:
            print(f"*** DO-NOT-CONCLUDE GUARD: w_shared seed={args.seed} did NOT converge trustworthily (hit_cap={cell['hit_cap']}). ***")
        out_path = os.path.join(args.results_dir, f"{args.toy}_{args.seed}.json")
        with open(out_path, "w") as f:
            json.dump(_jsonable(cell), f, indent=2)
        print(
            f"[width_wsel8] wrote {out_path} (w_shared_width={cell['w_shared_width']} w_sweep_width={cell['w_sweep_width']} "
            f"trustworthy={cell['trustworthy']} hit_cap={cell['hit_cap']})"
        )


if __name__ == "__main__":
    main()
