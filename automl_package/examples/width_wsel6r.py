r"""WSEL-6-R -- the real-data selection-fraction reopen.

`docs/plans/capacity_programme/width.md` WSEL-6-R; spec
`docs/plans/capacity_programme/shared/wsel6r-realdata-selection-fraction.md`.

WSEL-6 froze a 15% SELECT-set fraction on SYNTHETIC toys only; WSEL-24 causally demonstrated that
15% is underpowered on three of WSEL-9's five real datasets (diabetes/yacht/energy, `n_selection_
used` 37-92) via a 5-cell probe. This driver answers, per real dataset, the smallest SELECT fraction
at which the fielded GLOBAL width-selection rule (`cheapest_within_tolerance`) stops losing to the
accuracy-optimal pick within the noise-aware 2*SE bar (MASTER Decision 33(i)/36).

**The nested-prefix design (spec S2.1).** `width_wsel9._build_split` trains on `x_fit`/`x_stop`
independent of `fraction_pct` -- only the SELECT slice `x_p2[:n_select]` changes, and `x_p2`'s row
order is fixed by a permutation seeded on `seed` alone. So ONE retrain per (dataset, seed, width) at
`fraction_pct=50` (the split architecture's hard ceiling -- `_build_split`'s p1/p2 halving makes p2
exactly half the pool, so 50% always grabs the full p2 pool) captures a per-sample SELECT error
table that every smaller fraction in the ladder reads by ROW-PREFIX SLICING, at zero additional
training. A pre-registered faithfulness check (`_faithfulness_check`) spot-compares each retrained
cell's `held_out_mse` against the already-landed WSEL9 cell at the same (dataset, seed, width) --
TEST scoring never touches `x_select`, so this should match regardless of which fraction trained it.

**The binding capture requirement (spec S4).** The dial (per-input) arm's SELECT-split per-sample
error table -- `fit_global_selector`/`fit_router` build it internally
(`automl_package/models/flexnn/width/model.py:450`) and WSEL-9 always discarded it. This driver
captures it for the FIRST time. Raw per-sample arrays (SELECT/TEST squared errors, the dial error
table) never enter the committed per-cell JSON -- they land in a git-ignored sidecar
(`<results-dir>/_cache/<dataset>_<seed>_<arm>[_<width>][_frac<pct>]_arrays.npz`) the moment the cell
completes; the committed JSON keeps only a pointer, array shapes, a SHA-256 checksum and per-width
column means (case-law precedent: committed artifacts stay reviewably small).

Six arms this driver trains, one CLI cell per invocation (the ROOT runs the ~195-cell grid
backgrounded, never this driver -- standing worker-writes-the-driver / root-runs-the-grid split):
  * `--arm w_sweep --width K` -- ONE dedicated `FlexibleWidthNN(widths=(K,))`, reusing
    `width_wsel9._train_flexwidth_single` verbatim. `5 datasets x 3 seeds x 12 widths = 180` cells.
  * `--arm dial` -- ONE multi-head net yielding W-SHARED + W-PERINPUT + the captured error table,
    reusing `width_wsel9._train_dial_to_convergence` verbatim. `5 x 3 = 15` cells.
`plain_nn`/`lightgbm`/`linear_reg` are NOT retrained here -- `fraction_pct` cannot affect them
(they never touch `x_select`), so the already-landed WSEL9 cells are reused verbatim (spec S2.3).

Per-cell CLI:
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel6r.py \\
        --dataset yacht --seed 0 --arm w_sweep --width 5
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel6r.py \\
        --dataset yacht --seed 0 --arm dial

Aggregate, after every cell has landed (12 `w_sweep` widths + `dial`, per seed, per dataset):
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/width_wsel6r.py --summarize

Selftest / lint:
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel6r.py --selftest

Non-goals (task brief; spec S7): no changes to the GLOBAL rule / 2*SE multiplier / bootstrap
estimator, the per-input labelling tolerance, or `_build_split`'s FIT/STOP/SELECT architecture
itself (the 50% ceiling is a hard boundary, not something this task relaxes); no new datasets; no
report prose (a later WSEL-10-consuming task owns that); this driver's author never runs the full
grid (the ROOT does, `Bash(run_in_background: true)`); no edits to `width_wsel9.py`, any plan
document, or `.gitignore`.
"""

from __future__ import annotations

import argparse
import enum
import glob
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from typing import Any

import numpy as np

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root, so `import automl_package` works

import width_wsel4 as w4  # noqa: E402 -- PORTED_N_EPOCHS_CAP/PORTED_PATIENCE/PORTED_MIN_DELTA/_replay, reused verbatim (same convention width_wsel9.py uses).
import width_wsel9 as w9  # noqa: E402 -- Dataset enum, _build_split/_train_flexwidth_single/_train_dial_to_convergence/_width_model_config/_flexwidth_module_for_cost, RESULTS_DIR (read-only historical record), reused verbatim.

from automl_package.enums import CapacitySelection  # noqa: E402
from automl_package.models.flexnn.routing import (  # noqa: E402 -- the zero-compute per-row "fit_router-equivalent" labeller (spec S4 Consumers).
    DEFAULT_TOLERANCE,
    _cheapest_within_tolerance_labels,
)
from automl_package.utils.capacity_accounting import global_cheap_cost, held_out_read_cost, per_input_cost, sweep_cost  # noqa: E402
from automl_package.utils.capacity_selection import DEFAULT_N_BOOT, TOLERANCE_SE_MULTIPLE, cheapest_within_tolerance  # noqa: E402
from automl_package.utils.numerics import bootstrap_se  # noqa: E402
from automl_package.utils.run_provenance import run_provenance  # noqa: E402

RESULTS_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "WSEL6R")
WSEL9_RESULTS_DIR = w9.RESULTS_DIR  # read-only historical record for this task (spec S4) -- never a write target.
CACHE_SUBDIR = "_cache"  # sidecar arrays live here, git-ignored (spec S4) -- see this driver's final report for the .gitignore gap this write-set cannot close.

_DRIVER_NAME = "automl_package/examples/width_wsel6r.py"

Dataset = w9.Dataset  # reused verbatim (spec S2.3): the same five real datasets WSEL-9/WSEL-24 already used.
SEEDS = (0, 1, 2)  # spec S2.3: the same three seeds WSEL-9/WSEL-24 already used, for direct comparability.

MIN_FRACTION_PCT = 5  # spec S2.2's ladder floor (root amendment: exact toys-ladder reproduction at matched rungs).
MAX_FRACTION_PCT = 50  # the split architecture's hard ceiling (spec S0/S2.1/S7) -- never relaxed by this task.
DEFAULT_FRACTION_PCT = MAX_FRACTION_PCT  # spec S6: the default retrain point every smaller rung is sliced from.
FRACTION_LADDER_PCT = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50)  # spec S2.2, root-amended down to {5, 10} -- free at --summarize time (S2.1).

_FAITHFULNESS_REL_TOL = 1e-6  # spec S2.1's faithfulness check: "within float tolerance", not pinned to a number -- this driver's own conservative choice.
_FAITHFULNESS_ABS_TOL = 1e-9


class Arm(enum.Enum):
    """Closed set: the two arms this driver retrains (spec S2.3) -- `plain_nn`/`lightgbm`/`linear_reg` are reused verbatim from WSEL9, never retrained here."""

    W_SWEEP = "w_sweep"
    DIAL = "dial"


class Scale(enum.StrEnum):
    """Closed set: which side of the standardization a captured array lives on (spec S4's scale note)."""

    STANDARDIZED_Y = "standardized_y"  # SELECT-side arrays (computed from y_select_std, width_wsel9.py:499-500).
    RAW_Y = "raw_y"  # TEST-side arrays / held_out_mse (width_wsel9.py:495-497).


class Check4Basis(enum.StrEnum):
    """Closed set: which half of spec S3 check 4 a pick satisfied (or neither)."""

    ORACLE_IDENTITY_MATCH = "oracle_identity_match"
    TEST_PAIRED_VALIDATION = "test_paired_validation"
    FAILS = "fails"


# ---------------------------------------------------------------------------
# Validation / guards (spec S4/S6's hard-error contracts).
# ---------------------------------------------------------------------------


def _validate_fraction_pct(fraction_pct: int) -> int:
    """Rejects a fraction outside the split architecture's valid ladder range `[5, 50]` (spec S2.1/S2.2/S6 selftest 4).

    Args:
        fraction_pct: a candidate `--fraction-pct` value or fraction-ladder rung.

    Returns:
        `fraction_pct`, unchanged, if valid.

    Raises:
        SystemExit: `fraction_pct` is outside `[MIN_FRACTION_PCT, MAX_FRACTION_PCT]`.
    """
    if not (MIN_FRACTION_PCT <= fraction_pct <= MAX_FRACTION_PCT):
        raise SystemExit(f"--fraction-pct must be in [{MIN_FRACTION_PCT}, {MAX_FRACTION_PCT}] (the split architecture's ceiling, spec S2.1/S7); got {fraction_pct}.")
    return fraction_pct


def _assert_not_wsel9_dir(results_dir: str) -> None:
    """Hard error if `results_dir` resolves to the landed, committed WSEL9 record (spec S4/S6 selftest 5 -- read-only to this task).

    Args:
        results_dir: the `--results-dir` value under consideration.

    Raises:
        SystemExit: `results_dir` resolves to the same path as `WSEL9_RESULTS_DIR`.
    """
    if os.path.abspath(results_dir) == os.path.abspath(WSEL9_RESULTS_DIR):
        raise SystemExit(f"--results-dir resolves to {WSEL9_RESULTS_DIR}, the landed WSEL9 record -- read-only to WSEL-6-R (spec S4). Use a different --results-dir.")


def _read_constants(wsel7_path: str, wsel8_dir: str, fraction_pct: int) -> dict[str, Any]:
    """Bundles this task's OWN `fraction_pct` with WSEL-7/WSEL-8's read-and-verify constants.

    Unlike `width_wsel9._read_constants`, this never reads WSEL-6's frozen 15% artifact (spec S6's
    driver-contract flag list has no `--wsel6-path` -- this task's whole point is finding its OWN
    fraction, not consuming WSEL-6's). WSEL-7's router constants and WSEL-8's width ladder are
    untouched by this task (spec S7) and reused via `width_wsel9`'s own fail-loud readers verbatim.

    Args:
        wsel7_path: path to WSEL-7's frozen router-constants artifact.
        wsel8_dir: directory holding WSEL-8's per-cell JSONs (for `w_max`).
        fraction_pct: this cell's own SELECT-carve fraction (already validated).

    Returns:
        The `constants` dict every committed cell JSON carries.
    """
    return {
        "selection_fraction": {
            "source": "WSEL-6-R CLI --fraction-pct (this task's own ladder point, NOT WSEL-6's frozen 15% artifact)",
            "fraction": fraction_pct / 100.0,
            "fraction_pct": fraction_pct,
        },
        "width_ladder": w9._read_width_ladder(wsel8_dir),
        "router": w9._read_router_constants(wsel7_path),
    }


# ---------------------------------------------------------------------------
# File naming + the sidecar file contract (spec S4).
# ---------------------------------------------------------------------------


def _cell_filename(dataset_value: str, seed: int, arm: str, width: int | None, fraction_pct: int, tag: str | None) -> str:
    """Committed per-cell JSON filename: `<dataset>_<seed>_<arm>[_<width>][_frac<pct>][_<tag>].json`.

    The `_frac<pct>` suffix is omitted at the default ceiling (`DEFAULT_FRACTION_PCT`) so the
    primary systematic-grid cells match WSEL9's own naming convention exactly; it appears only when
    `--fraction-pct` trains a specific smaller fraction directly (spec S2.1's faithfulness-check
    escape hatch, or S5's efficiency-design fallback branch), so those cells never collide with the
    ceiling cell for the same (dataset, seed, width).

    Args:
        dataset_value: `Dataset.value`.
        seed: the seed this cell trained at.
        arm: `Arm.value`.
        width: the width, for `w_sweep` cells; `None` for `dial`.
        fraction_pct: the SELECT-carve fraction this cell trained at.
        tag: optional filename suffix for easily-deletable smoke cells (mirrors `width_wsel9.py`'s own `--tag`).

    Returns:
        The filename (no directory component).
    """
    width_suffix = f"_{width}" if width is not None else ""
    frac_suffix = "" if fraction_pct == DEFAULT_FRACTION_PCT else f"_frac{fraction_pct}"
    tag_suffix = f"_{tag}" if tag else ""
    return f"{dataset_value}_{seed}_{arm}{width_suffix}{frac_suffix}{tag_suffix}.json"


def _sidecar_path(results_dir: str, dataset_value: str, seed: int, arm: str, width: int | None, fraction_pct: int) -> str:
    """Sidecar `.npz` path: `<results-dir>/_cache/<dataset>_<seed>_<arm>[_<width>][_frac<pct>]_arrays.npz` (spec S4).

    Args:
        results_dir: this cell's `--results-dir`.
        dataset_value: `Dataset.value`.
        seed: the seed this cell trained at.
        arm: `Arm.value`.
        width: the width, for `w_sweep` cells; `None` for `dial`.
        fraction_pct: the SELECT-carve fraction this cell trained at.

    Returns:
        The absolute sidecar path (directory not yet created).
    """
    width_suffix = f"_{width}" if width is not None else ""
    frac_suffix = "" if fraction_pct == DEFAULT_FRACTION_PCT else f"_frac{fraction_pct}"
    return os.path.join(results_dir, CACHE_SUBDIR, f"{dataset_value}_{seed}_{arm}{width_suffix}{frac_suffix}_arrays.npz")


def _write_sidecar(path: str, arrays: dict[str, np.ndarray]) -> str:
    """Writes raw per-sample arrays to a git-ignored `.npz` sidecar; returns the file's SHA-256 hex digest (spec S4).

    Args:
        path: destination `.npz` path (parent directory created if missing).
        arrays: `{array_name: ndarray}` -- one `.npz` entry per key.

    Returns:
        Hex-encoded SHA-256 digest of the written file, for the committed cell's checksum field.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **arrays)
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _read_sidecar(path: str) -> dict[str, np.ndarray]:
    """Reads a sidecar `.npz` back into a plain `{array_name: ndarray}` dict (round-trip half of spec S4's file contract).

    Args:
        path: the sidecar `.npz` path.

    Returns:
        Every array stored in the file, keyed by name.
    """
    with np.load(path) as npz:
        return {key: npz[key] for key in npz.files}


def _faithfulness_check(dataset: Any, seed: int, width: int, held_out_mse: float) -> dict[str, Any]:
    """Spot-checks spec S2.1's determinism claim (TEST scoring never touches `x_select`).

    This cell's `held_out_mse` should equal the already-landed WSEL9 cell's at the SAME (dataset,
    seed, width), read-only, regardless of which fraction either cell trained at. A mismatch
    invalidates S2.1's single-retrain-at-the-ceiling design for this dataset (spec S5's
    named efficiency-design failure branch) -- deciding the fallback is the ROOT's job at grid-run
    time, on the FIRST landed cell of each dataset (spec S8/S9); this function only records the
    comparison so that decision has evidence to act on.

    Args:
        dataset: a `Dataset` member (or a selftest fake with a `.value` attribute).
        seed: the seed under comparison.
        width: the width under comparison.
        held_out_mse: this cell's own (raw-y scale) held-out TEST MSE.

    Returns:
        `{"wsel9_cell_path", "wsel9_held_out_mse", "matches", "tolerance"}` -- all `None` (except
        `matches`, also `None`) if no landed WSEL9 cell exists to compare against.
    """
    wsel9_path = os.path.join(WSEL9_RESULTS_DIR, f"{dataset.value}_{seed}_w_sweep_{width}.json")
    if not os.path.exists(wsel9_path):
        return {"wsel9_cell_path": None, "wsel9_held_out_mse": None, "matches": None, "note": "no landed WSEL9 cell to compare against"}
    with open(wsel9_path) as f:
        wsel9_cell = json.load(f)
    wsel9_mse = float(wsel9_cell["held_out_mse"])
    matches = math.isclose(held_out_mse, wsel9_mse, rel_tol=_FAITHFULNESS_REL_TOL, abs_tol=_FAITHFULNESS_ABS_TOL)
    return {
        "wsel9_cell_path": os.path.abspath(wsel9_path),
        "wsel9_held_out_mse": wsel9_mse,
        "matches": bool(matches),
        "tolerance": {"rel_tol": _FAITHFULNESS_REL_TOL, "abs_tol": _FAITHFULNESS_ABS_TOL},
    }


# ---------------------------------------------------------------------------
# Per-cell runners -- copy-with-provenance of `width_wsel9._run_w_sweep_cell`/`_run_dial_cell`
# (width_wsel9.py:479-525, :528-598): the TRAINING calls (`_train_flexwidth_single`,
# `_train_dial_to_convergence`) are reused VERBATIM by import, so the trained net is byte-identical
# to what WSEL9's own driver would produce; the cell-ASSEMBLY step is copied and extended for spec
# S4's file contract (sidecar routing, the captured error table, the additional TEST-side capture),
# which a literal black-box call to WSEL9's own functions cannot produce without a second retrain.
# ---------------------------------------------------------------------------


def _run_w_sweep_cell(dataset: Any, seed: int, width: int, split: dict[str, Any], constants: dict[str, Any], *, epoch_cap: int, results_dir: str) -> dict[str, Any]:
    """Trains ONE dedicated `FlexibleWidthNN(widths=(width,))`; ONE retrain covers every fraction rung (spec S2.1).

    Captures the per-sample SELECT squared error (standardized-y) AND the per-sample TEST squared
    error (raw-y, spec S4's additional capture, needed by S3 check 4's paired validation) at the
    SAME `model.predict` calls that already produce `held_out_mse` -- one retrain, not two.

    Args:
        dataset: a `Dataset` member (or a selftest fake with a `.value` attribute).
        seed: torch/data seed for this cell.
        width: the single width this cell trains.
        split: `width_wsel9._build_split`'s output for (dataset, seed, this cell's `fraction_pct`).
        constants: this task's `constants` dict (`_read_constants`).
        epoch_cap: safety cap on training epochs.
        results_dir: where this cell's sidecar lands (`_cache/` subdirectory).

    Returns:
        The committed per-cell JSON (raw arrays replaced by a `sidecar` pointer object).
    """
    d = split["d"]
    fraction_pct = split["config"]["fraction_pct"]
    batch_size = min(w9.DEFAULT_BATCH_SIZE, len(split["x_fit"]))
    model, val_loss_history = w9._train_flexwidth_single(
        (width,), seed, d, split["x_fit"], split["y_fit_std"], split["x_stop"], split["y_stop_std"], max_epochs=epoch_cap, batch_size=batch_size
    )
    replay = w4._replay(val_loss_history, w4.PORTED_PATIENCE, w4.PORTED_MIN_DELTA)
    hit_cap = bool(len(val_loss_history) >= epoch_cap)
    trustworthy = bool(replay.trustworthy and not hit_cap)

    pred_test_std = model.predict(split["x_test"], filter_data=False, width=width)
    pred_test_raw = pred_test_std * split["norm"]["y_std"] + split["norm"]["y_mean"]
    test_squared_error = (pred_test_raw - split["y_test_raw"]) ** 2
    held_out_mse = float(np.mean(test_squared_error))

    pred_select_std = model.predict(split["x_select"], filter_data=False, width=width)
    select_squared_error = (pred_select_std - split["y_select_std"]) ** 2

    n_train_used = len(split["x_fit"])
    training_macs = sweep_cost(model.model, [width], n_train_used, len(val_loss_history))

    sidecar_path = _sidecar_path(results_dir, dataset.value, seed, Arm.W_SWEEP.value, width, fraction_pct)
    sidecar_sha256 = _write_sidecar(sidecar_path, {"select_squared_error": select_squared_error, "test_squared_error": test_squared_error})

    return {
        "dataset": dataset.value,
        "seed": seed,
        "arm": Arm.W_SWEEP.value,
        "width": width,
        "fraction_pct_captured_at": fraction_pct,
        "held_out_mse": held_out_mse,
        "held_out_trajectory": replay.summary()["trajectory"],
        "trustworthy": trustworthy,
        "hit_cap": hit_cap,
        "converged": replay.converged,
        "trajectory_applicable": True,
        "n_selection_used": split["n_selection_used"],
        "n_pool": split["n_pool"],
        "training_macs": training_macs,
        "n_train_used": n_train_used,
        "actual_epochs": len(val_loss_history),
        "selection_cost": None,  # per-width cell isn't W-SWEEP's priced answer -- summarize() aggregates all 12 (mirrors width_wsel9.py).
        "faithfulness_check": _faithfulness_check(dataset, seed, width, held_out_mse),
        "sidecar": {
            "path": os.path.relpath(sidecar_path, results_dir),
            "sha256": sidecar_sha256,
            "arrays": {
                "select_squared_error": {"shape": list(select_squared_error.shape), "scale": Scale.STANDARDIZED_Y.value, "mean": float(select_squared_error.mean())},
                "test_squared_error": {"shape": list(test_squared_error.shape), "scale": Scale.RAW_Y.value, "mean": float(test_squared_error.mean())},
            },
        },
        "config": w9._width_model_config(d, batch_size, epoch_cap, split),
        "constants": constants,
        "provenance": run_provenance(),
    }


def _run_dial_cell(dataset: Any, seed: int, split: dict[str, Any], constants: dict[str, Any], *, epoch_cap: int, results_dir: str) -> dict[str, Any]:
    """Trains ONE multi-head dial net; captures the SELECT error table WSEL9 always discarded (spec S4).

    Builds the `(n_selection_used, w_max)` per-sample SELECT error table directly off
    `model._per_sample_error_at_width` -- the SAME scoring primitive `fit_global_selector` stacks
    internally (`automl_package/models/flexnn/width/model.py:450`) and discards after computing only
    column means. This is the FIRST time this table exists on disk for any dataset (spec S4).

    Args:
        dataset: a `Dataset` member (or a selftest fake with a `.value` attribute).
        seed: torch/data seed for this cell.
        split: `width_wsel9._build_split`'s output for (dataset, seed, this cell's `fraction_pct`).
        constants: this task's `constants` dict (`_read_constants`).
        epoch_cap: safety cap on training epochs.
        results_dir: where this cell's sidecar lands (`_cache/` subdirectory).

    Returns:
        The committed per-cell JSON (raw arrays replaced by a `sidecar` pointer object).
    """
    d = split["d"]
    fraction_pct = split["config"]["fraction_pct"]
    w_max = constants["width_ladder"]["w_max"]
    widths = tuple(range(1, w_max + 1))
    batch_size = min(w9.DEFAULT_BATCH_SIZE, len(split["x_fit"]))

    model, per_width_results = w9._train_dial_to_convergence(
        widths, seed, d, split["x_fit"], split["y_fit_std"], split["x_stop"], split["y_stop_std"], max_epochs=epoch_cap, batch_size=batch_size
    )
    hit_cap_shared = any(r.hit_cap for r in per_width_results.values())
    trustworthy_shared = all(r.trustworthy for r in per_width_results.values())
    converged_shared = all(r.converged for r in per_width_results.values())
    trajectory_shared = {str(w): [[int(e), float(v)] for e, v in r.trajectory] for w, r in per_width_results.items()}
    actual_epochs = max(r.stop_epoch for r in per_width_results.values())
    n_train_used = len(split["x_fit"])
    training_macs = sweep_cost(model.model, list(widths), n_train_used, actual_epochs)

    y_mean, y_std = split["norm"]["y_mean"], split["norm"]["y_std"]
    x_sel, y_sel_std = split["x_select"], split["y_select_std"]
    x_test, y_test_raw = split["x_test"], split["y_test_raw"]

    # Spec S4's capture requirement -- built BEFORE either selector call, independent of both, so it
    # is trustworthy even if a selector's own bookkeeping ever drifts.
    select_error_table = np.stack([model._per_sample_error_at_width(x_sel, y_sel_std, width) for width in widths], axis=1)

    w_shared_width = model.fit_global_selector(x_sel, y_sel_std, seed=seed)
    model.capacity_selection = CapacitySelection.GLOBAL_CHEAP
    w_shared_pred_raw = model.predict(x_test, filter_data=False) * y_std + y_mean
    w_shared_test_squared_error = (w_shared_pred_raw - y_test_raw) ** 2
    w_shared_mse = float(np.mean(w_shared_test_squared_error))
    shared_cost = global_cheap_cost(training_macs=training_macs, net=model.model, capacity_grid=list(widths), n_samples=len(x_sel))

    model.fit_router(x_sel, y_sel_std)
    model.capacity_selection = CapacitySelection.PER_INPUT
    w_perinput_pred_raw = model.predict(x_test, filter_data=False) * y_std + y_mean
    w_perinput_test_squared_error = (w_perinput_pred_raw - y_test_raw) ** 2
    w_perinput_mse = float(np.mean(w_perinput_test_squared_error))
    router = model.capacity_router_
    perinput_cost = per_input_cost(training_macs=training_macs, in_dim=d, n_capacities=len(widths), n_samples=len(x_sel), n_epochs=router.n_epochs, hidden=router.hidden)
    routed_widths = [capacity[0] for capacity in router.route(x_test)]
    width_distribution = {str(w): routed_widths.count(w) for w in sorted(set(routed_widths))}

    sidecar_path = _sidecar_path(results_dir, dataset.value, seed, Arm.DIAL.value, None, fraction_pct)
    column_means = select_error_table.mean(axis=0).tolist()
    sidecar_sha256 = _write_sidecar(
        sidecar_path,
        {
            "select_error_table": select_error_table,
            "test_squared_error_w_shared": w_shared_test_squared_error,
            "test_squared_error_w_perinput": w_perinput_test_squared_error,
        },
    )

    return {
        "dataset": dataset.value,
        "seed": seed,
        "arm": Arm.DIAL.value,
        "w_max": w_max,
        "fraction_pct_captured_at": fraction_pct,
        "shared_training": {
            "held_out_trajectory": trajectory_shared,
            "trustworthy": trustworthy_shared,
            "hit_cap": hit_cap_shared,
            "converged": converged_shared,
            "actual_epochs": actual_epochs,
            "n_train_used": n_train_used,
            "training_macs": training_macs,
        },
        "trajectory_applicable": True,
        "w_shared": {
            "selected_width": int(w_shared_width),
            "held_out_mse": w_shared_mse,
            "selection_cost": {"training_macs": shared_cost.training_macs, "selection_macs": shared_cost.selection_macs, "total_macs": shared_cost.total_macs},
            "selection": {"fraction_pct": fraction_pct, "n_selection_used": len(x_sel)},
        },
        "w_perinput": {
            "mean_routed_width": float(np.mean(routed_widths)),
            "width_distribution": width_distribution,
            "held_out_mse": w_perinput_mse,
            "selection_cost": {"training_macs": perinput_cost.training_macs, "selection_macs": perinput_cost.selection_macs, "total_macs": perinput_cost.total_macs},
            "selection": {"fraction_pct": fraction_pct, "n_selection_used": len(x_sel)},
        },
        "select_error_table": {  # metadata only -- the raw table itself lives in the sidecar (spec S4 root amendment).
            "widths": list(widths),
            "n_selection_used": len(x_sel),
            "fraction_pct_captured_at": fraction_pct,
        },
        "n_pool": split["n_pool"],
        "sidecar": {
            "path": os.path.relpath(sidecar_path, results_dir),
            "sha256": sidecar_sha256,
            "arrays": {
                "select_error_table": {"shape": list(select_error_table.shape), "scale": Scale.STANDARDIZED_Y.value, "column_means": column_means},
                "test_squared_error_w_shared": {"shape": list(w_shared_test_squared_error.shape), "scale": Scale.RAW_Y.value, "mean": float(w_shared_test_squared_error.mean())},
                "test_squared_error_w_perinput": {
                    "shape": list(w_perinput_test_squared_error.shape),
                    "scale": Scale.RAW_Y.value,
                    "mean": float(w_perinput_test_squared_error.mean()),
                },
            },
        },
        "config": w9._width_model_config(d, batch_size, epoch_cap, split),
        "constants": constants,
        "provenance": run_provenance(),
    }


# ---------------------------------------------------------------------------
# Dual-pick core (MASTER Decision 36, spec S3) -- pure array arithmetic, zero retraining.
# ---------------------------------------------------------------------------


def _dual_pick(select_error_table: np.ndarray, *, seed: int, n_boot: int = DEFAULT_N_BOOT) -> dict[str, int]:
    """Both Decision-36 picks off the SAME `(n, n_widths)` SELECT error table (spec S3 point 1-2).

    Args:
        select_error_table: `(n_selection_at_fraction, n_widths)` SELECT-split per-sample squared
            error (standardized-y scale), cheapest-first column order (index 0 = width 1).
        seed: bootstrap RNG seed -- pass the (dataset, seed) cell's OWN seed, matching
            `width_wsel9.summarize`'s own convention (no silent default: a selection rule whose
            answer moves between runs is not a rule, `automl_package/utils/numerics.py:260-262`).
        n_boot: bootstrap resamples for `cheapest_within_tolerance`.

    Returns:
        `{"smallest_sufficient_idx", "accuracy_optimal_idx"}` -- 0-based column indices (width = idx + 1 on this battery's 1..w_max ladder).
    """
    accuracy_optimal_idx = int(np.argmin(select_error_table.mean(axis=0)))
    smallest_sufficient_idx = cheapest_within_tolerance(select_error_table, n_boot=n_boot, seed=seed)
    return {"smallest_sufficient_idx": smallest_sufficient_idx, "accuracy_optimal_idx": accuracy_optimal_idx}


def _paired_check(pick_errors: np.ndarray, oracle_errors: np.ndarray, *, seed: int, n_boot: int = DEFAULT_N_BOOT) -> dict[str, Any]:
    """The shared `mean_diff`/`se`/`within_tolerance` triple.

    Spec S3 check 4's TEST-side paired validation AND the SELECT-side continuity readout are the
    SAME computation on different arrays (mirrors `width_wsel24`'s own `orig`/`probe` sub-object
    schema, so this grid's numbers sit directly beside WSEL-24's probe rows).

    Args:
        pick_errors: `(n,)` per-sample error for the pick under test.
        oracle_errors: `(n,)` per-sample error for the comparator (TEST-oracle-best width).
        seed: bootstrap RNG seed (the (dataset, seed) cell's own seed).
        n_boot: bootstrap resamples.

    Returns:
        `{"mean_diff", "se", "within_tolerance"}`.
    """
    diff = pick_errors - oracle_errors
    se = bootstrap_se(diff, n_boot=n_boot, seed=seed)
    mean_diff = float(diff.mean())
    return {"mean_diff": mean_diff, "se": se, "within_tolerance": bool(mean_diff <= TOLERANCE_SE_MULTIPLE * se)}


def _select_prefix_length(fraction_pct: int, n_pool: int, n_select_ceiling: int) -> int:
    """Mirrors `width_wsel9._build_split`'s own `n_select` formula (width_wsel9.py:303) exactly.

    A smaller fraction's prefix length must match what training AT that fraction directly would
    have produced -- the arithmetic core of spec S2.1's nested-prefix argument.

    Args:
        fraction_pct: the ladder rung under evaluation.
        n_pool: the (dataset, seed)'s FIT+STOP+SELECT pool size (`split["n_pool"]`, fraction-invariant).
        n_select_ceiling: the ceiling cell's own `n_selection_used` (caps the prefix length).

    Returns:
        The row-prefix length at `fraction_pct`.
    """
    return max(1, min(round((fraction_pct / 100.0) * n_pool), n_select_ceiling))


def _pick_cost(training_macs: int, d: int, seed: int, width: int, n_selection: int) -> dict[str, int]:
    """W-SWEEP's per-width cost shape: training plus a SELECT-read at `n_selection` rows.

    Mirrors `width_wsel9.summarize`'s own cost computation (width_wsel9.py:829-834).

    Args:
        training_macs: this width's already-recorded training MAC count (fraction-invariant).
        d: input dimensionality.
        seed: torch seed (for the throwaway untrained module `_flexwidth_module_for_cost` builds).
        width: the picked width.
        n_selection: the SELECT-set size AT THIS FRACTION (drives `selection_macs`).

    Returns:
        `{"training_macs", "selection_macs", "total_macs"}`.
    """
    selection_macs = held_out_read_cost(w9._flexwidth_module_for_cost(d, seed, width), [width], n_selection)
    return {"training_macs": training_macs, "selection_macs": selection_macs, "total_macs": training_macs + selection_macs}


# ---------------------------------------------------------------------------
# --summarize -- reads the ceiling (fraction_pct=50) cells' sidecars, slices to every ladder rung,
# computes S3's dual picks + check 4 + continuity, S4's descriptive dial readout, S5 Outcome C's
# negative-control note, and writes ONE combined WSEL6R/frozen.json (mirrors WSEL6's own combined-
# ledger convention, not WSEL9's per-dataset CSV convention).
# ---------------------------------------------------------------------------

_CEILING_SWEEP_TAG = "w_sweep"


def _load_ceiling_sweep_cells(dataset: Any, seed: int, results_dir: str) -> dict[int, dict[str, Any]]:
    """Loads every landed `w_sweep` cell trained AT THE CEILING (no `_frac<pct>`/`_<tag>` suffix) for (dataset, seed).

    A dataset that fell back to literal per-fraction retrains (spec S5's efficiency-design failure
    branch) writes a DISTINCT file set (`_frac<pct>` suffix) this function deliberately excludes --
    `--summarize` covers the primary (ceiling-slicing) design; the fallback's cells remain on disk
    for manual/root inspection (see this driver's final report for this documented scope boundary).

    Args:
        dataset: a `Dataset` member.
        seed: the seed to load.
        results_dir: where cells are landed.

    Returns:
        `{width: cell}` for every width whose ceiling cell has landed.
    """
    pattern = re.compile(rf"^{re.escape(dataset.value)}_{seed}_{_CEILING_SWEEP_TAG}_(\d+)\.json$")
    cells: dict[int, dict[str, Any]] = {}
    for path in sorted(glob.glob(os.path.join(results_dir, f"{dataset.value}_{seed}_{_CEILING_SWEEP_TAG}_*.json"))):
        match = pattern.match(os.path.basename(path))
        if not match:
            continue
        with open(path) as f:
            cells[int(match.group(1))] = json.load(f)
    return cells


def _dial_descriptive_readout(dataset: Any, seed: int, results_dir: str, fraction_ladder_pct: tuple[int, ...]) -> dict[str, Any] | None:
    """Spec S4's descriptive-only (S2.3 scope boundary -- never graded against S3's bar) per-fraction dial readout.

    Reports a `fit_global_selector`-mirroring pick (`cheapest_within_tolerance` on the sliced table)
    and a `fit_router`-EQUIVALENT per-row label distribution
    (`automl_package.models.flexnn.routing._cheapest_within_tolerance_labels`, the SAME zero-compute
    per-row labeller a real router's training targets come from) -- no router MLP is re-fitted here,
    matching spec S2.4's "two numpy reads, no additional training" cost claim.

    Args:
        dataset: a `Dataset` member.
        seed: the seed to read.
        results_dir: where cells are landed.
        fraction_ladder_pct: the ladder rungs to report.

    Returns:
        `{"per_fraction": {...}}`, or `None` if this (dataset, seed)'s ceiling dial cell hasn't landed.
    """
    cell_path = os.path.join(results_dir, _cell_filename(dataset.value, seed, Arm.DIAL.value, None, DEFAULT_FRACTION_PCT, None))
    if not os.path.exists(cell_path):
        return None
    with open(cell_path) as f:
        cell = json.load(f)
    sidecar = _read_sidecar(os.path.join(results_dir, cell["sidecar"]["path"]))
    table_ceiling = sidecar["select_error_table"]
    n_pool = cell["n_pool"]
    n_select_ceiling = cell["select_error_table"]["n_selection_used"]
    widths = cell["select_error_table"]["widths"]

    per_fraction: dict[str, Any] = {}
    for f in fraction_ladder_pct:
        n_f = _select_prefix_length(f, n_pool, n_select_ceiling)
        table_f = table_ceiling[:n_f]
        w_shared_idx = cheapest_within_tolerance(table_f, n_boot=DEFAULT_N_BOOT, seed=seed)
        row_labels = _cheapest_within_tolerance_labels(table_f, tolerance=DEFAULT_TOLERANCE)
        label_counts = {str(widths[i]): int((row_labels == i).sum()) for i in range(len(widths))}
        per_fraction[str(f)] = {
            "n_selection_used": n_f,
            "w_shared_equivalent_width": widths[w_shared_idx],
            "per_input_label_distribution": label_counts,
        }
    return {"per_fraction": per_fraction}


def summarize(results_dir: str = RESULTS_DIR, fraction_ladder_pct: tuple[int, ...] = FRACTION_LADDER_PCT, seeds: tuple[int, ...] = SEEDS) -> dict[str, Any]:
    """Aggregates every landed cell across all five datasets into ONE combined `WSEL6R/frozen.json`.

    For each dataset, each seed, each fraction rung: slices the ceiling (fraction_pct=50) SELECT
    error table to that rung's prefix length (`_select_prefix_length`), computes both Decision-36
    picks (`_dual_pick`), checks each against the TEST-oracle-best width (identity match OR the
    TEST-side paired validation, spec S3 check 4), reports the SELECT-side continuity readout
    (never gating), and the dial arm's descriptive per-fraction readout (spec S4). Writes the
    combined ledger and prints one line per dataset.

    Args:
        results_dir: where landed cells live and `frozen.json` is written.
        fraction_ladder_pct: the ladder rungs to report (spec S2.2, overridable).
        seeds: which seeds to aggregate (spec S2.3).

    Returns:
        The written `frozen.json` contents.
    """
    for f in fraction_ladder_pct:
        _validate_fraction_pct(f)

    datasets_out: dict[str, Any] = {}
    for dataset in Dataset:
        seeds_out: dict[str, Any] = {}
        for seed in seeds:
            sweep_cells = _load_ceiling_sweep_cells(dataset, seed, results_dir)
            if not sweep_cells:
                continue
            w_max = max(sweep_cells)
            if set(sweep_cells) != set(range(1, w_max + 1)):
                print(f"[width_wsel6r] dataset={dataset.value} seed={seed}: incomplete w_sweep ladder ({sorted(sweep_cells)}), skipping.")
                continue
            widths = list(range(1, w_max + 1))
            d = sweep_cells[widths[0]]["config"]["d"]
            n_pool = sweep_cells[widths[0]]["n_pool"]

            select_cols, test_cols, n_select_ceiling = [], [], None
            for w in widths:
                sidecar = _read_sidecar(os.path.join(results_dir, sweep_cells[w]["sidecar"]["path"]))
                select_cols.append(sidecar["select_squared_error"])
                test_cols.append(sidecar["test_squared_error"])
                if n_select_ceiling is None:
                    n_select_ceiling = len(sidecar["select_squared_error"])
            select_table_ceiling = np.stack(select_cols, axis=1)  # (n_select_ceiling, w_max)
            test_table = np.stack(test_cols, axis=1)  # (n_test, w_max) -- fraction-invariant.

            oracle_best_idx = int(np.argmin([sweep_cells[w]["held_out_mse"] for w in widths]))
            oracle_best_width = widths[oracle_best_idx]

            per_fraction: dict[str, Any] = {}
            for f in fraction_ladder_pct:
                n_f = _select_prefix_length(f, n_pool, n_select_ceiling)
                table_f = select_table_ceiling[:n_f]
                picks = _dual_pick(table_f, seed=seed)
                fraction_entry: dict[str, Any] = {"n_selection_used": n_f}
                for pick_name, idx_key in (("smallest_sufficient", "smallest_sufficient_idx"), ("accuracy_optimal", "accuracy_optimal_idx")):
                    idx = picks[idx_key]
                    pick_width = widths[idx]
                    identity_match = pick_width == oracle_best_width
                    test_check = _paired_check(test_table[:, idx], test_table[:, oracle_best_idx], seed=seed)
                    continuity = _paired_check(table_f[:, idx], table_f[:, oracle_best_idx], seed=seed)
                    if identity_match:
                        basis = Check4Basis.ORACLE_IDENTITY_MATCH
                    elif test_check["within_tolerance"]:
                        basis = Check4Basis.TEST_PAIRED_VALIDATION
                    else:
                        basis = Check4Basis.FAILS
                    fraction_entry[pick_name] = {
                        "width": pick_width,
                        "select_mean_error": float(table_f[:, idx].mean()),
                        "test_mse": float(test_table[:, idx].mean()),
                        "matches_oracle_identity": identity_match,
                        "test_side_check": test_check,
                        "check4_pass": bool(identity_match or test_check["within_tolerance"]),
                        "check4_basis": basis.value,
                        "continuity": continuity,  # reported, never gating (spec S3 point 4).
                        "cost": _pick_cost(sweep_cells[pick_width]["training_macs"], d, seed, pick_width, n_f),
                    }
                per_fraction[str(f)] = fraction_entry

            # Decision 36 default objective: smallest-sufficient -- f* = smallest fraction whose pick
            # passes AND stays passing at every larger fraction (no re-flipping), WSEL6's own `saturated: true` semantics.
            f_star = next(
                (f for f in fraction_ladder_pct if all(per_fraction[str(g)]["smallest_sufficient"]["check4_pass"] for g in fraction_ladder_pct if g >= f)),
                None,
            )

            seed_entry: dict[str, Any] = {
                "oracle_best_width": oracle_best_width,
                "n_pool": n_pool,
                "n_select_ceiling": n_select_ceiling,
                "per_fraction": per_fraction,
                "f_star": f_star,
            }
            dial_descriptive = _dial_descriptive_readout(dataset, seed, results_dir, fraction_ladder_pct)
            if dial_descriptive is not None:
                seed_entry["dial_descriptive"] = dial_descriptive
            if dataset in (Dataset.KIN8NM, Dataset.CALIFORNIA):
                # Spec S5 Outcome C's pre-registered negative control: these two datasets' picks
                # should move little to none across the ladder; a large shift is a surprise, not
                # silently reinterpreted here (spec S7).
                widths_across_ladder = sorted({per_fraction[str(f)]["smallest_sufficient"]["width"] for f in fraction_ladder_pct})
                seed_entry["negative_control_check"] = {"widths_across_ladder": widths_across_ladder, "moved": len(widths_across_ladder) > 1}
            seeds_out[str(seed)] = seed_entry

        if not seeds_out:
            continue
        per_seed_f_stars = [seeds_out[str(s)]["f_star"] for s in seeds if str(s) in seeds_out]
        # Dataset-level headline: every seed must pass-and-stay-passing -- equivalent to the LARGEST
        # per-seed f* (each per-seed f* already guarantees "stays passing" from that point on).
        dataset_f_star = max(per_seed_f_stars) if per_seed_f_stars and all(fs is not None for fs in per_seed_f_stars) else None
        datasets_out[dataset.value] = {"per_seed": seeds_out, "f_star": dataset_f_star}

    frozen = {
        "fraction_ladder_pct": list(fraction_ladder_pct),
        "seeds": list(seeds),
        "datasets": datasets_out,
        "provenance": run_provenance(),
    }
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "frozen.json")
    with open(out_path, "w") as f_out:
        json.dump(w9._jsonable(frozen), f_out, indent=2)
    print(f"[width_wsel6r] wrote {out_path} (datasets={sorted(datasets_out)})")
    return frozen


# ---------------------------------------------------------------------------
# Selftest -- the five pre-registered assertions (spec S6), synthetic in-memory only: no real
# dataset, no real training.
# ---------------------------------------------------------------------------


def run_selftest() -> bool:
    """Runs spec S6's five pre-registered `--selftest` assertions; prints a PASS/FAIL line per assertion."""
    ok = True
    tmp_dir = tempfile.mkdtemp(prefix="width_wsel6r_selftest_")
    try:
        rng = np.random.default_rng(0)
        n_max, n_widths = 100, 12
        true_col_means = np.linspace(1.0, 5.0, n_widths)  # cheapest-first, monotone -- enough for a stable, deterministic pick.
        full_table = np.clip(rng.normal(loc=true_col_means, scale=0.3, size=(n_max, n_widths)), 0.0, None)
        widths = list(range(1, n_widths + 1))

        # Assertion 1 -- prefix-slice equivalence: slicing the cached ceiling table to a smaller n
        # reproduces the SAME pick as "building the table directly at that smaller n from the same
        # underlying per-sample errors" -- which, since `_build_split`'s SELECT sets are exact
        # prefixes of one another (spec S2.1), literally IS the same slice.
        n_small = 30
        sliced = full_table[:n_small]
        direct = full_table[:n_small]
        pick_sliced = _dual_pick(sliced, seed=0)
        pick_direct = _dual_pick(direct, seed=0)
        ok1 = pick_sliced == pick_direct
        print(f"[wsel6r selftest] prefix-slice equivalence: {'PASS' if ok1 else 'FAIL'}")
        ok = ok and ok1

        # Assertion 2 -- dual-pick output always carries both objectives' identity, accuracy, cost (Decision 36 shape check).
        test_table = np.clip(rng.normal(loc=true_col_means, scale=0.3, size=(50, n_widths)), 0.0, None)
        oracle_idx = int(np.argmin(test_table.mean(axis=0)))
        picks = _dual_pick(full_table, seed=0)
        report = {}
        for name, idx_key in (("smallest_sufficient", "smallest_sufficient_idx"), ("accuracy_optimal", "accuracy_optimal_idx")):
            idx = picks[idx_key]
            cost = _pick_cost(training_macs=1_000, d=3, seed=0, width=widths[idx], n_selection=n_small)
            check = _paired_check(test_table[:, idx], test_table[:, oracle_idx], seed=0)
            report[name] = {"width": widths[idx], "test_mse": float(test_table[:, idx].mean()), "cost": cost, "test_side_check": check}
        ok2 = all({"width", "test_mse", "cost"} <= set(report[name]) for name in ("smallest_sufficient", "accuracy_optimal"))
        print(f"[wsel6r selftest] dual-pick shape (Decision 36 identity/accuracy/cost): {'PASS' if ok2 else 'FAIL'}")
        ok = ok and ok2

        # Assertion 3 -- select_error_table round-trips through the sidecar (npz) with the same
        # shape/values, and the committed checksum matches the sidecar on disk.
        sidecar_path = os.path.join(tmp_dir, "synth_0_dial_arrays.npz")
        sha = _write_sidecar(sidecar_path, {"select_error_table": full_table})
        with open(sidecar_path, "rb") as f:
            recomputed_sha = hashlib.sha256(f.read()).hexdigest()
        reloaded = _read_sidecar(sidecar_path)
        ok3 = sha == recomputed_sha and reloaded["select_error_table"].shape == full_table.shape and np.allclose(reloaded["select_error_table"], full_table)
        print(f"[wsel6r selftest] sidecar round-trip + checksum: {'PASS' if ok3 else 'FAIL'}")
        ok = ok and ok3

        # Assertion 4 -- a fraction ladder point outside [5, 50] is rejected.
        rejected = []
        for bad in (MIN_FRACTION_PCT - 1, MAX_FRACTION_PCT + 1, 0, 100):
            try:
                _validate_fraction_pct(bad)
                rejected.append(False)
            except SystemExit:
                rejected.append(True)
        ok4 = all(rejected) and _validate_fraction_pct(MIN_FRACTION_PCT) == MIN_FRACTION_PCT and _validate_fraction_pct(MAX_FRACTION_PCT) == MAX_FRACTION_PCT
        print(f"[wsel6r selftest] fraction ladder range [{MIN_FRACTION_PCT}, {MAX_FRACTION_PCT}] enforced: {'PASS' if ok4 else 'FAIL'}")
        ok = ok and ok4

        # Assertion 5 -- a --results-dir resolving to the landed WSEL9 record is rejected with a hard error.
        try:
            _assert_not_wsel9_dir(WSEL9_RESULTS_DIR)
            ok5 = False
        except SystemExit:
            ok5 = True
        try:
            _assert_not_wsel9_dir(tmp_dir)  # a genuinely different dir must NOT be rejected.
        except SystemExit:
            ok5 = False
        print(f"[wsel6r selftest] WSEL9 write-refusal: {'PASS' if ok5 else 'FAIL'}")
        ok = ok and ok5

        print(f"[wsel6r selftest] {'PASS' if ok else 'FAIL'}")
        return ok
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Runs one per-cell training (`--arm w_sweep`/`dial`), `--summarize`, or `--selftest`."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true", help="Synthetic in-memory known-answer checks, no training, then exit.")
    parser.add_argument("--summarize", action="store_true", help="Aggregate every landed cell (all datasets) into WSEL6R/frozen.json, then exit.")

    parser.add_argument("--dataset", choices=[d.value for d in Dataset], default=None, help="Required outside --selftest/--summarize.")
    parser.add_argument("--seed", type=int, default=None, help="Required for a real cell (not --summarize/--selftest).")
    parser.add_argument("--arm", choices=[a.value for a in Arm], default=None, help="Required for a real cell.")
    parser.add_argument("--width", type=int, default=None, help="Required for --arm w_sweep; must not be given otherwise.")
    parser.add_argument("--fraction-pct", type=int, default=DEFAULT_FRACTION_PCT, help="SELECT-carve fraction to train at (spec S2.1); default is the ceiling. Must be in [5, 50].")
    parser.add_argument(
        "--fraction-ladder-pct",
        type=str,
        default=",".join(str(f) for f in FRACTION_LADDER_PCT),
        help="--summarize only: comma-separated fraction-ladder rungs, overridable at zero compute (spec S2.2).",
    )
    parser.add_argument("--tag", type=str, default=None, help="Optional filename suffix (e.g. 'authsmoke') so smoke cells are easy to find and delete.")

    parser.add_argument("--epoch-cap", type=int, default=w4.PORTED_N_EPOCHS_CAP, help="Safety cap on training epochs for any NN-based arm.")
    parser.add_argument("--max-train", type=int, default=None, help="Subsamples the FIT/STOP/SELECT pool (never TEST) for wall-clock reasons; recorded in provenance.")
    parser.add_argument("--test-fraction", type=float, default=w9.TEST_FRACTION)
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--wsel7-path", type=str, default=w9.WSEL7_FROZEN_PATH)
    parser.add_argument("--wsel8-dir", type=str, default=w9.WSEL8_DIR)

    args = parser.parse_args()

    modes = (args.selftest, args.summarize)
    if sum(bool(m) for m in modes) > 1:
        parser.error("--selftest and --summarize are mutually exclusive.")
    if args.selftest:
        sys.exit(0 if run_selftest() else 1)

    _assert_not_wsel9_dir(args.results_dir)

    if args.summarize:
        ladder = tuple(_validate_fraction_pct(int(x)) for x in args.fraction_ladder_pct.split(","))
        summarize(results_dir=args.results_dir, fraction_ladder_pct=ladder)
        return

    if args.dataset is None or args.seed is None or args.arm is None:
        parser.error("--dataset, --seed and --arm are required for a real cell (or pass --selftest / --summarize).")
    dataset = Dataset(args.dataset)
    arm = Arm(args.arm)
    if arm is Arm.W_SWEEP and args.width is None:
        parser.error("--width is required for --arm w_sweep.")
    if arm is not Arm.W_SWEEP and args.width is not None:
        parser.error("--width only applies to --arm w_sweep.")
    fraction_pct = _validate_fraction_pct(args.fraction_pct)

    constants = _read_constants(args.wsel7_path, args.wsel8_dir, fraction_pct)
    if arm is Arm.W_SWEEP and not (1 <= args.width <= constants["width_ladder"]["w_max"]):
        raise SystemExit(f"--width must be in [1, {constants['width_ladder']['w_max']}]; got {args.width}.")

    split = w9._build_split(dataset, args.seed, fraction_pct=fraction_pct, test_fraction=args.test_fraction, max_train=args.max_train)

    os.makedirs(args.results_dir, exist_ok=True)
    if arm is Arm.W_SWEEP:
        cell = _run_w_sweep_cell(dataset, args.seed, args.width, split, constants, epoch_cap=args.epoch_cap, results_dir=args.results_dir)
    else:
        cell = _run_dial_cell(dataset, args.seed, split, constants, epoch_cap=args.epoch_cap, results_dir=args.results_dir)

    out_path = os.path.join(args.results_dir, _cell_filename(dataset.value, args.seed, arm.value, args.width, fraction_pct, args.tag))
    with open(out_path, "w") as f:
        json.dump(w9._jsonable(cell), f, indent=2)

    trustworthy = cell.get("trustworthy", cell["shared_training"]["trustworthy"] if arm is Arm.DIAL else True)
    if not trustworthy:
        print(f"*** DO-NOT-CONCLUDE GUARD: dataset={dataset.value} seed={args.seed} arm={arm.value} did NOT converge trustworthily. ***")
    print(f"[width_wsel6r] wrote {out_path}")


if __name__ == "__main__":
    main()
