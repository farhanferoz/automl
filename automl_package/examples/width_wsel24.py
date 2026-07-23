r"""WSEL-24 -- the selection post-mortem (`docs/plans/capacity_programme/width.md` WSEL-24).

WSEL-9's real-data battery found W-SWEEP loses to the plain fixed-width control on all five
datasets. This module attributes WHY, in the block's stated order, reading WSEL-9's already-landed
per-cell JSONs (`automl_package/examples/capacity_ladder_results/WSEL9/`) rather than retraining:

  * **Q1 (zero compute, decides the branch)** -- per (dataset, seed), does the best width in
    W-SWEEP's OWN recorded table (`held_out_mse` on TEST, across its 12 dedicated `w_sweep_<width>`
    cells) beat the plain control? `_attribute_seed` / `attribute_dataset`.
  * **Q2 (zero compute)** -- replay selection counterfactually from the recorded SELECT-split
    curves (`select_squared_error`): the SELECT-argmin pick, the shipped rule's actual pick
    (`cheapest_within_tolerance`), and the pick under OTHER bootstrap-SE multiples
    (`_replay_pick_at_multiple` -- a read-only mirror of that function's algorithm shape with the
    multiple exposed as a parameter; `capacity_selection.py` hardcodes `TOLERANCE_SE_MULTIPLE=2.0`
    as a module constant on purpose, and this module never changes it).
  * **Q3 (cheap)** -- `n_selection_used` per dataset (read off every cell); WSEL-9 never caches
    trained weights (`width_wsel9.py`'s own `_flexwidth_module_for_cost` docstring), so "re-score
    on larger selection carves" means RETRAINING under an enlarged `--wsel6-path` fraction override
    -- `run_selection_set_size_probe` does exactly that, reusing `width_wsel9`'s own
    `_build_split`/`_run_w_sweep_cell` verbatim. Capped at 6 training cells total for this task; 5
    were spent on two probes (diabetes seed 1 widths 1/3; yacht seed 0 widths 2/8) whose results
    are embedded in `attribution.json`'s `selection_set_size_probe` key for those two datasets.
  * **Q4 (recipe branch, only if Q1 = NO)** -- dedicated width-`w_max` vs the plain control at
    identical capacity; checked automatically inside `attribute_dataset` (`DominantCause.RECIPE`).

Also checked, zero-compute: **data-carve consistency** (`_check_data_carve_consistency`) -- every
arm in a (dataset, seed) cell must share `test_fraction`/`max_train`/`fraction_pct`
(`docs/width_benchmark/benchmark_spec.md` SS2/SS3's "one split, shared verbatim across every arm"
rule) -- a mismatch would be `DominantCause.DATA_CARVE`.

Non-goals (task brief): no retraining beyond the 6-cell probe cap; no selector/band CHANGES
(WSEL-22 owns those -- this module only replays counterfactuals); no new datasets; no report prose
(WSEL-10 consumes this ledger); no changes to `routing.py`/`capacity_accounting.py`/`width_wsel9.py`
or any plan document.

CLI:
    # Build the ledger from WSEL-9's landed disk artifacts (zero compute):
    ~/dev/.venv/bin/python automl_package/examples/width_wsel24.py --build

    # Re-run one of the two selection-set-size probes (TRAINS 2 cells -- counts against the cap):
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel24.py \\
        --run-probe --dataset diabetes --seed 1 --width-lo 1 --width-hi 3 --output /tmp/probe_diabetes.json

    # Selftest (synthetic in-memory tables, no real dataset, no training):
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/width_wsel24.py --selftest
"""

from __future__ import annotations

import argparse
import enum
import glob
import json
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

import width_wsel4 as w4  # noqa: E402 -- PORTED_N_EPOCHS_CAP, reused verbatim for the compute-probe branch only.
import width_wsel9 as w9  # noqa: E402 -- Dataset enum, RESULTS_DIR, _read_constants/_build_split/_run_w_sweep_cell (reused, not reimplemented).

from automl_package.utils.capacity_selection import DEFAULT_N_BOOT, TOLERANCE_SE_MULTIPLE, cheapest_within_tolerance  # noqa: E402
from automl_package.utils.numerics import bootstrap_se  # noqa: E402

RESULTS_DIR = w9.RESULTS_DIR
LEDGER_PATH = os.path.join(RESULTS_DIR, "attribution.json")
SEEDS = (0, 1, 2)
_SWEEP_RE = re.compile(r"_w_sweep_(\d+)(?:_\w+)?\.json$")

# Q2's "narrower/wider band" replay -- 2.0 is the shipped default (`TOLERANCE_SE_MULTIPLE`); the
# rest are the counterfactual multiples this task asks for.
BAND_MULTIPLES_TESTED = (0.0, 1.0, TOLERANCE_SE_MULTIPLE, 3.0, 4.0, 6.0, 10.0)

# This battery's n_selection_used clusters at {37, 53, 92} (diabetes/yacht/energy, all sub-100)
# vs {983, 2477} (kin8nm/california) -- a ~10x gap, so any threshold placed inside it (100..900)
# sorts this battery identically. Not a general-purpose constant for other datasets.
_SMALL_N_SELECTION_THRESHOLD = 300

# "No winning width" (Q1 = NO) needs a tie tolerance, not exact float equality -- the three cells
# where oracle-best is w_max reproduce plain_nn's own architecture almost exactly (same hidden
# size/activation/recipe) but not bit-for-bit (different model wrapper).
_RECIPE_TIE_REL_TOL = 1e-3

# `run_selection_set_size_probe`'s enlarged SELECT fraction: `_build_split`'s p1/p2 carve makes p2
# exactly half of the pool (even/odd row split), so `fraction_pct=50` always grabs the FULL p2 pool
# regardless of dataset size -- verified (module docstring's Q3 probe).
PROBE_FRACTION_PCT = 50


class DominantCause(enum.StrEnum):
    """Closed taxonomy for the per-dataset verdict (`width.md` WSEL-24 block's own six-way split)."""

    BAND = "band"
    SE_ESTIMATE = "se_estimate"
    SELECTION_SET_SIZE = "selection_set_size"
    RULE_OBJECTIVE_MISMATCH = "rule_objective_mismatch"
    RECIPE = "recipe"
    DATA_CARVE = "data_carve"


# ---------------------------------------------------------------------------
# Disk reads -- zero compute.
# ---------------------------------------------------------------------------


def _load_cell(dataset: Any, seed: int, arm: str, results_dir: str, width: int | None = None) -> dict[str, Any]:
    """Loads one landed WSEL-9 per-cell JSON (`<dataset>_<seed>_<arm>[_<width>].json`)."""
    suffix = f"_{width}" if width is not None else ""
    path = os.path.join(results_dir, f"{dataset.value}_{seed}_{arm}{suffix}.json")
    with open(path) as f:
        return json.load(f)


def _load_sweep_table(dataset: Any, seed: int, results_dir: str) -> dict[int, dict[str, Any]]:
    """Loads every landed `w_sweep_<width>` cell for (dataset, seed) -- `{width: cell}`, any tag suffix ignored."""
    cells: dict[int, dict[str, Any]] = {}
    pattern = os.path.join(results_dir, f"{dataset.value}_{seed}_w_sweep_*.json")
    for path in sorted(glob.glob(pattern)):
        match = _SWEEP_RE.search(os.path.basename(path))
        if not match:
            continue
        width = int(match.group(1))
        with open(path) as f:
            cells[width] = json.load(f)
    return cells


def _error_table(cells: dict[int, dict[str, Any]]) -> tuple[list[int], np.ndarray]:
    """Sorted widths + the `(n_selection_used, n_widths)` SELECT squared-error table, cheapest-first."""
    widths = sorted(cells)
    table = np.array([cells[w]["select_squared_error"] for w in widths], dtype=np.float64).T
    return widths, table


def _replay_pick_at_multiple(widths: list[int], table: np.ndarray, seed: int, multiple: float) -> int:
    """Mirrors `cheapest_within_tolerance`'s exact algorithm shape with an explicit SE multiple.

    `capacity_selection.cheapest_within_tolerance` hardcodes `TOLERANCE_SE_MULTIPLE=2.0` as a module
    constant, not a parameter -- correct for the shipped rule, but Q2 needs the pick under OTHER
    multiples too. This is a read-only counterfactual replay; it is never wired back into
    `capacity_selection.py` (WSEL-22(c) owns any real change to the shipped multiplier).

    Args:
        widths: sorted width ladder, cheapest-first (column order of `table`).
        table: `(n_selection_used, n_widths)` SELECT squared-error table.
        seed: bootstrap RNG seed (matches `cheapest_within_tolerance`'s own `seed` usage).
        multiple: the SE multiple to test acceptance against (2.0 reproduces the shipped rule).

    Returns:
        The picked width (not a column index).
    """
    column_means = table.mean(axis=0)
    best_idx = int(np.argmin(column_means))
    best_col = table[:, best_idx]
    for idx in range(table.shape[1]):
        diff = table[:, idx] - best_col
        se = bootstrap_se(diff, n_boot=DEFAULT_N_BOOT, seed=seed)
        if float(diff.mean()) <= multiple * se:
            return widths[idx]
    return widths[best_idx]  # unreachable, mirrors cheapest_within_tolerance's own fallback.


def _check_data_carve_consistency(dataset: Any, results_dir: str) -> bool:
    """Zero-compute confirmation of benchmark_spec.md SS2/SS3's "one split, shared verbatim across every arm" rule.

    Every arm in a (dataset, seed) cell must record the identical
    `test_fraction`/`max_train`/`fraction_pct` -- a mismatch would mean an arm silently trained on
    a different data carve than its siblings (`DominantCause.DATA_CARVE`).
    """
    keys = ("test_fraction", "max_train", "fraction_pct")
    for seed in SEEDS:
        configs: list[dict[str, Any]] = [cell["config"] for cell in _load_sweep_table(dataset, seed, results_dir).values()]
        for arm in ("plain_nn", "lightgbm", "linear_reg", "dial"):
            configs.append(_load_cell(dataset, seed, arm, results_dir)["config"])
        if not configs:
            continue
        reference = {k: configs[0][k] for k in keys}
        if any(config.get(k) != reference[k] for config in configs for k in keys):
            return False
    return True


def _attribute_seed(dataset: Any, seed: int, results_dir: str) -> dict[str, Any]:
    """Q1/Q2 for one (dataset, seed): oracle-best width, SELECT-argmin, the shipped rule's pick, and the band-multiple replay."""
    sweep_cells = _load_sweep_table(dataset, seed, results_dir)
    if not sweep_cells:
        raise FileNotFoundError(f"No w_sweep_<width> cells found for dataset={dataset.value} seed={seed} under {results_dir}.")

    test_mse = {w: cell["held_out_mse"] for w, cell in sweep_cells.items()}
    oracle_best_w = min(test_mse, key=test_mse.get)

    widths, table = _error_table(sweep_cells)
    select_argmin_w = widths[int(np.argmin(table.mean(axis=0)))]
    actual_pick_w = widths[cheapest_within_tolerance(table, n_boot=DEFAULT_N_BOOT, seed=seed)]
    band_picks = {multiple: _replay_pick_at_multiple(widths, table, seed, multiple) for multiple in BAND_MULTIPLES_TESTED}

    plain_mse = _load_cell(dataset, seed, "plain_nn", results_dir)["held_out_mse"]

    return {
        "oracle_best_width": oracle_best_w,
        "oracle_best_test_mse": test_mse[oracle_best_w],
        "select_argmin_width": select_argmin_w,
        "select_argmin_test_mse": test_mse[select_argmin_w],
        "actual_pick_width": actual_pick_w,
        "actual_pick_test_mse": test_mse[actual_pick_w],
        "plain_nn_test_mse": plain_mse,
        "band_picks_by_se_multiple": {str(multiple): picked for multiple, picked in band_picks.items()},
        "n_selection_used": sweep_cells[widths[0]]["n_selection_used"],
    }


def attribute_dataset(dataset: Any, results_dir: str = RESULTS_DIR, probe: dict[str, Any] | None = None) -> dict[str, Any]:
    """Full per-dataset attribution: Q1/Q2 aggregated over `SEEDS`, Q3's n_selection_used, the dominant-cause verdict.

    Args:
        dataset: a `width_wsel9.Dataset` member (or a selftest fake with a `.value` attribute).
        results_dir: directory holding the landed WSEL-9 per-cell JSONs.
        probe: an optional `run_selection_set_size_probe` result (Q3's compute-probe evidence),
            embedded verbatim under the `selection_set_size_probe` key when given.

    Returns:
        The per-dataset ledger entry -- `winning_width_exists`, `selection_lost_it`,
        `dominant_cause`, per-cause MSE magnitudes, per-seed leaves, and provenance.
    """
    per_seed = {seed: _attribute_seed(dataset, seed, results_dir) for seed in SEEDS}

    mean_oracle = float(np.mean([v["oracle_best_test_mse"] for v in per_seed.values()]))
    mean_argmin = float(np.mean([v["select_argmin_test_mse"] for v in per_seed.values()]))
    mean_actual = float(np.mean([v["actual_pick_test_mse"] for v in per_seed.values()]))
    mean_plain = float(np.mean([v["plain_nn_test_mse"] for v in per_seed.values()]))
    n_selection_used = next(iter(per_seed.values()))["n_selection_used"]

    winning_width_exists = mean_oracle < mean_plain * (1.0 - _RECIPE_TIE_REL_TOL)
    selection_lost_it = winning_width_exists and mean_actual >= mean_plain

    data_carve_consistent = _check_data_carve_consistency(dataset, results_dir)

    if not data_carve_consistent:
        dominant_cause = DominantCause.DATA_CARVE
    elif not winning_width_exists:
        dominant_cause = DominantCause.RECIPE
    elif n_selection_used < _SMALL_N_SELECTION_THRESHOLD:
        dominant_cause = DominantCause.SELECTION_SET_SIZE
    else:
        dominant_cause = DominantCause.RULE_OBJECTIVE_MISMATCH

    per_seed_relative_gap_pct = [100.0 * (v["actual_pick_test_mse"] - v["plain_nn_test_mse"]) / v["plain_nn_test_mse"] for v in per_seed.values()]

    result = {
        "winning_width_exists": winning_width_exists,
        "selection_lost_it": selection_lost_it,
        "dominant_cause": dominant_cause.value,
        "n_selection_used": n_selection_used,
        "data_carve_consistent": data_carve_consistent,
        "aggregate_mean_test_mse": {
            "oracle_best": mean_oracle,
            "select_argmin": mean_argmin,
            "actual_pick": mean_actual,
            "plain_nn": mean_plain,
        },
        "magnitudes": {
            "selection_noise_mse_forfeited": mean_argmin - mean_oracle,
            "band_mse_forfeited": mean_actual - mean_argmin,
            "total_mse_forfeited": mean_actual - mean_oracle,
            "gap_vs_plain_mse": mean_actual - mean_plain,
            "gap_vs_plain_relative_pct": 100.0 * (mean_actual - mean_plain) / mean_plain,
            "per_seed_relative_gap_pct": per_seed_relative_gap_pct,
            "per_seed_relative_gap_std_pct": float(np.std(per_seed_relative_gap_pct)),
        },
        "per_seed": {str(seed): v for seed, v in per_seed.items()},
        "provenance": {
            "sweep_cells": f"{results_dir}/{dataset.value}_<seed>_w_sweep_<width>.json (seeds={list(SEEDS)})",
            "plain_nn_cells": f"{results_dir}/{dataset.value}_<seed>_plain_nn.json",
            "selector": "automl_package.utils.capacity_selection.cheapest_within_tolerance",
        },
    }
    if probe is not None:
        result["selection_set_size_probe"] = probe
    return result


# ---------------------------------------------------------------------------
# Q3's compute probe -- TRAINS. Counts against the task's 6-cell cap. Never called by --build or
# --selftest; opt-in only via --run-probe.
# ---------------------------------------------------------------------------


def _probe_from_cells(
    dataset: Any,
    seed: int,
    w_lo: int,
    w_hi: int,
    orig_lo: dict[str, Any],
    orig_hi: dict[str, Any],
    probe_lo: dict[str, Any],
    probe_hi: dict[str, Any],
    probe_fraction_pct: int,
) -> dict[str, Any]:
    """Pure computation half of the selection-set-size probe -- no training, given already-loaded cells.

    Compares the paired SELECT difference (`w_lo` minus `w_hi`) and its bootstrap SE at the
    original `n_selection_used` against the SAME comparison at the enlarged (`probe_fraction_pct`)
    carve -- `w_lo` is the actual (cheaper) pick under test, `w_hi` the oracle-best/wider
    comparator it lost to on TEST.
    """
    e_lo_orig = np.asarray(orig_lo["select_squared_error"])
    e_hi_orig = np.asarray(orig_hi["select_squared_error"])
    e_lo_probe = np.asarray(probe_lo["select_squared_error"])
    e_hi_probe = np.asarray(probe_hi["select_squared_error"])

    n_orig = len(e_lo_orig)
    prefix_consistent = bool(np.allclose(e_lo_probe[:n_orig], e_lo_orig) and np.allclose(e_hi_probe[:n_orig], e_hi_orig))

    diff_orig = e_lo_orig - e_hi_orig
    diff_probe = e_lo_probe - e_hi_probe
    se_orig = bootstrap_se(diff_orig, n_boot=DEFAULT_N_BOOT, seed=seed)
    se_probe = bootstrap_se(diff_probe, n_boot=DEFAULT_N_BOOT, seed=seed)

    return {
        "dataset": dataset.value,
        "seed": seed,
        "width_lo_actual_pick": w_lo,
        "width_hi_comparator": w_hi,
        "probe_fraction_pct": probe_fraction_pct,
        "n_selection_orig": n_orig,
        "n_selection_probe": len(e_lo_probe),
        "prefix_consistent": prefix_consistent,
        "test_mse_lo": orig_lo["held_out_mse"],
        "test_mse_hi": orig_hi["held_out_mse"],
        "orig": {
            "mean_diff": float(diff_orig.mean()),
            "se": se_orig,
            "within_tolerance": bool(diff_orig.mean() <= TOLERANCE_SE_MULTIPLE * se_orig),
        },
        "probe": {
            "mean_diff": float(diff_probe.mean()),
            "se": se_probe,
            "within_tolerance": bool(diff_probe.mean() <= TOLERANCE_SE_MULTIPLE * se_probe),
        },
    }


def run_selection_set_size_probe(
    dataset: Any, seed: int, w_lo: int, w_hi: int, results_dir: str = RESULTS_DIR, probe_fraction_pct: int = PROBE_FRACTION_PCT
) -> dict[str, Any]:
    """TRAINS `w_lo` and `w_hi` under an ENLARGED SELECT carve; compares against the landed originals.

    WSEL-9 never caches trained weights, so "re-score on a larger selection carve" means retraining
    -- identical in every respect to the original cell (same FIT/STOP split, same seed, same
    hyperparameters) except the fraction fed to `_build_split`, which only changes how large a
    prefix of the SAME seeded shuffle of the p2 pool is scored (`_build_split`'s `x_select =
    x_p2[:n_select]` is a strict prefix, so the original SELECT rows are a subset of the enlarged
    set -- `prefix_consistent` in the return value verifies this). Reuses `width_wsel9`'s own
    `_read_constants`/`_build_split`/`_run_w_sweep_cell` verbatim; trains no new code path.

    Counts as 2 training cells against WSEL-24's 6-cell compute-probe cap.

    Args:
        dataset: the `width_wsel9.Dataset` member to probe.
        seed: the (dataset, seed) cell to re-run.
        w_lo: the actual (cheaper) pick under test.
        w_hi: the oracle-best/wider comparator width.
        results_dir: where the ORIGINAL landed cells live (read, never written).
        probe_fraction_pct: the enlarged selection-set fraction (50 grabs the full p2 pool).

    Returns:
        See `_probe_from_cells`.
    """
    orig_lo = _load_cell(dataset, seed, "w_sweep", results_dir, width=w_lo)
    orig_hi = _load_cell(dataset, seed, "w_sweep", results_dir, width=w_hi)

    constants = w9._read_constants(w9.WSEL6_FROZEN_PATH, w9.WSEL7_FROZEN_PATH, w9.WSEL8_DIR)
    probe_constants = dict(
        constants,
        selection_fraction={
            "source": "WSEL-24 compute probe (in-memory override, not WSEL-6's frozen artifact)",
            "fraction": probe_fraction_pct / 100.0,
            "fraction_pct": probe_fraction_pct,
        },
    )
    split = w9._build_split(dataset, seed, fraction_pct=probe_fraction_pct)
    probe_lo = w9._run_w_sweep_cell(dataset, seed, w_lo, split, probe_constants, epoch_cap=w4.PORTED_N_EPOCHS_CAP)
    probe_hi = w9._run_w_sweep_cell(dataset, seed, w_hi, split, probe_constants, epoch_cap=w4.PORTED_N_EPOCHS_CAP)

    return _probe_from_cells(dataset, seed, w_lo, w_hi, orig_lo, orig_hi, probe_lo, probe_hi, probe_fraction_pct)


# ---------------------------------------------------------------------------
# --build -- assembles attribution.json, one dataset at a time, writing after each (volatile
# context -- never accumulate all five in memory before the first write).
# ---------------------------------------------------------------------------


def _write_ledger_entry(dataset_key: str, entry: dict[str, Any], ledger_path: str = LEDGER_PATH) -> None:
    ledger: dict[str, Any] = {}
    if os.path.exists(ledger_path):
        with open(ledger_path) as f:
            ledger = json.load(f)
    ledger[dataset_key] = entry
    os.makedirs(os.path.dirname(ledger_path), exist_ok=True)
    with open(ledger_path, "w") as f:
        json.dump(ledger, f, indent=2)


def build_ledger(results_dir: str = RESULTS_DIR, ledger_path: str = LEDGER_PATH, probe_results: dict[str, dict[str, Any]] | None = None) -> None:
    """Runs `attribute_dataset` for every `width_wsel9.Dataset`, writing `ledger_path` incrementally.

    Args:
        results_dir: directory holding WSEL-9's landed per-cell JSONs.
        ledger_path: output path for the attribution ledger.
        probe_results: optional `{dataset_value: run_selection_set_size_probe(...)-shaped dict}` --
            embedded under each matching dataset's `selection_set_size_probe` key.
    """
    probe_results = probe_results or {}
    for dataset in w9.Dataset:
        entry = attribute_dataset(dataset, results_dir=results_dir, probe=probe_results.get(dataset.value))
        _write_ledger_entry(dataset.value, entry, ledger_path=ledger_path)
        print(f"[width_wsel24] {dataset.value}: winning_width_exists={entry['winning_width_exists']} "
              f"selection_lost_it={entry['selection_lost_it']} dominant_cause={entry['dominant_cause']}")


# ---------------------------------------------------------------------------
# Selftest -- synthetic in-memory tables, no real dataset, no training.
# ---------------------------------------------------------------------------


class _FakeDataset:
    """Mimics `width_wsel9.Dataset`'s `.value` attribute for a selftest scenario written under its own name."""

    def __init__(self, value: str) -> None:
        self.value = value


def _write_synthetic_sweep_cell(results_dir: str, dataset_value: str, seed: int, width: int, held_out_mse: float, select_errors: np.ndarray, config: dict[str, Any]) -> None:
    cell = {
        "dataset": dataset_value,
        "seed": seed,
        "arm": "w_sweep",
        "width": width,
        "held_out_mse": held_out_mse,
        "select_squared_error": select_errors.tolist(),
        "n_selection_used": len(select_errors),
        "config": config,
    }
    with open(os.path.join(results_dir, f"{dataset_value}_{seed}_w_sweep_{width}.json"), "w") as f:
        json.dump(cell, f)


def _write_synthetic_other_arm(results_dir: str, dataset_value: str, seed: int, arm: str, held_out_mse: float, config: dict[str, Any]) -> None:
    cell = {"dataset": dataset_value, "seed": seed, "arm": arm, "held_out_mse": held_out_mse, "config": config}
    with open(os.path.join(results_dir, f"{dataset_value}_{seed}_{arm}.json"), "w") as f:
        json.dump(cell, f)


def _build_synthetic_dataset(
    results_dir: str,
    dataset_value: str,
    *,
    n_widths: int,
    n_selection: int,
    winner_width: int,
    plain_mse: float,
    oracle_mse: float,
    worst_mse: float,
    noise_scale: float,
    rng_seed: int,
    fraction_pct: int = 15,
    break_data_carve_on_seed: int | None = None,
) -> None:
    """Writes a synthetic (dataset, seed) family of `w_sweep_<width>`/`plain_nn`/etc. cells for one known-answer scenario.

    TEST mse per width is a deterministic V-shape bottoming at `winner_width` (`oracle_mse` there,
    rising to `worst_mse` at the ladder's ends); the SELECT curve is that SAME shape plus
    `noise_scale`-sized per-row noise over `n_selection` rows -- shrinking `n_selection` (holding
    `noise_scale` fixed) is exactly WSEL-24's `selection_set_size` mechanism.
    """
    rng = np.random.default_rng(rng_seed)
    widths = list(range(1, n_widths + 1))
    for seed in SEEDS:
        config = {"test_fraction": 0.2, "max_train": None, "fraction_pct": fraction_pct}
        for width in widths:
            distance = abs(width - winner_width) / max(winner_width - 1, n_widths - winner_width, 1)
            true_mse = oracle_mse + distance * (worst_mse - oracle_mse)
            select_errors = np.clip(rng.normal(loc=true_mse, scale=noise_scale, size=n_selection), a_min=0.0, a_max=None)
            width_config = dict(config)
            if break_data_carve_on_seed == seed and width == widths[-1]:
                width_config["fraction_pct"] = fraction_pct + 1  # deliberately inconsistent, for the data_carve-detection scenario.
            _write_synthetic_sweep_cell(results_dir, dataset_value, seed, width, true_mse, select_errors, width_config)
        for arm in ("plain_nn", "lightgbm", "linear_reg", "dial"):
            _write_synthetic_other_arm(results_dir, dataset_value, seed, arm, plain_mse, dict(config))


def run_selftest() -> bool:
    """Known-answer checks: synthetic in-memory tables through the REAL `attribute_dataset` pipeline."""
    ok = True
    tmp_dir = tempfile.mkdtemp(prefix="width_wsel24_selftest_")
    try:
        # `_replay_pick_at_multiple` at the shipped multiple must agree with `cheapest_within_tolerance` itself.
        rng = np.random.default_rng(0)
        synthetic_table = rng.normal(loc=np.linspace(5.0, 1.0, 8), scale=0.3, size=(60, 8))
        synthetic_table = np.clip(synthetic_table, 0.0, None)
        widths8 = list(range(1, 9))
        shipped_pick = widths8[cheapest_within_tolerance(synthetic_table, n_boot=DEFAULT_N_BOOT, seed=0)]
        replay_pick = _replay_pick_at_multiple(widths8, synthetic_table, 0, TOLERANCE_SE_MULTIPLE)
        ok_replay_matches = shipped_pick == replay_pick
        print(f"[wsel24 selftest] _replay_pick_at_multiple agrees with cheapest_within_tolerance at multiple={TOLERANCE_SE_MULTIPLE}: {'PASS' if ok_replay_matches else 'FAIL'}")
        ok = ok and ok_replay_matches

        # Band-multiple monotonicity: a bigger multiple can only accept a cheaper-or-equal width (never a MORE expensive one).
        picks_by_multiple = [_replay_pick_at_multiple(widths8, synthetic_table, 0, m) for m in (0.0, 1.0, 2.0, 4.0, 10.0)]
        ok_monotone = all(picks_by_multiple[i] >= picks_by_multiple[i + 1] for i in range(len(picks_by_multiple) - 1))
        print(f"[wsel24 selftest] band replay monotonicity ({picks_by_multiple}): {'PASS' if ok_monotone else 'FAIL'}")
        ok = ok and ok_monotone

        # Scenario A -- Q1 = NO (RECIPE): oracle-best TIES plain_nn (no winning width anywhere).
        _build_synthetic_dataset(
            tmp_dir, "synthA", n_widths=12, n_selection=200, winner_width=12, plain_mse=10.0, oracle_mse=10.0, worst_mse=15.0, noise_scale=0.5, rng_seed=1
        )
        entry_a = attribute_dataset(_FakeDataset("synthA"), results_dir=tmp_dir)
        ok_a = entry_a["winning_width_exists"] is False and entry_a["selection_lost_it"] is False and entry_a["dominant_cause"] == DominantCause.RECIPE.value
        print(f"[wsel24 selftest] scenario A (Q1=NO -> RECIPE): {entry_a['dominant_cause']} {'PASS' if ok_a else 'FAIL'}")
        ok = ok and ok_a

        # Scenario B -- Q1 = YES, SMALL n_selection_used (SELECTION_SET_SIZE): true winner at a middle width, tiny noisy SELECT sample.
        _build_synthetic_dataset(
            tmp_dir, "synthB", n_widths=12, n_selection=40, winner_width=8, plain_mse=6.0, oracle_mse=1.0, worst_mse=9.0, noise_scale=4.0, rng_seed=2
        )
        entry_b = attribute_dataset(_FakeDataset("synthB"), results_dir=tmp_dir)
        ok_b = (
            entry_b["winning_width_exists"] is True
            and entry_b["n_selection_used"] < _SMALL_N_SELECTION_THRESHOLD
            and entry_b["dominant_cause"] == DominantCause.SELECTION_SET_SIZE.value
        )
        print(f"[wsel24 selftest] scenario B (small n -> SELECTION_SET_SIZE): {entry_b['dominant_cause']} n_sel={entry_b['n_selection_used']} {'PASS' if ok_b else 'FAIL'}")
        ok = ok and ok_b

        # Scenario C -- Q1 = YES, LARGE n_selection_used (RULE_OBJECTIVE_MISMATCH): same shape, large SELECT sample.
        _build_synthetic_dataset(
            tmp_dir, "synthC", n_widths=12, n_selection=2000, winner_width=8, plain_mse=6.0, oracle_mse=1.0, worst_mse=9.0, noise_scale=4.0, rng_seed=3
        )
        entry_c = attribute_dataset(_FakeDataset("synthC"), results_dir=tmp_dir)
        ok_c = (
            entry_c["winning_width_exists"] is True
            and entry_c["n_selection_used"] >= _SMALL_N_SELECTION_THRESHOLD
            and entry_c["dominant_cause"] == DominantCause.RULE_OBJECTIVE_MISMATCH.value
        )
        print(f"[wsel24 selftest] scenario C (large n -> RULE_OBJECTIVE_MISMATCH): {entry_c['dominant_cause']} n_sel={entry_c['n_selection_used']} {'PASS' if ok_c else 'FAIL'}")
        ok = ok and ok_c

        # Scenario D -- a data-carve mismatch (one arm silently trained on a different fraction_pct) must be caught.
        _build_synthetic_dataset(
            tmp_dir, "synthD", n_widths=12, n_selection=200, winner_width=8, plain_mse=6.0, oracle_mse=1.0, worst_mse=9.0, noise_scale=0.5, rng_seed=4, break_data_carve_on_seed=0
        )
        entry_d = attribute_dataset(_FakeDataset("synthD"), results_dir=tmp_dir)
        ok_d = entry_d["data_carve_consistent"] is False and entry_d["dominant_cause"] == DominantCause.DATA_CARVE.value
        print(f"[wsel24 selftest] scenario D (data-carve mismatch detected): consistent={entry_d['data_carve_consistent']} {'PASS' if ok_d else 'FAIL'}")
        ok = ok and ok_d

        # _probe_from_cells (pure computation half of Q3's probe) on synthetic before/after cells.
        orig_lo = {"held_out_mse": 9.0, "select_squared_error": [1.0, 1.1, 0.9, 1.0] * 10}  # n=40, mean~1.0
        orig_hi = {"held_out_mse": 1.0, "select_squared_error": [0.8, 0.9, 1.0, 0.8] * 10}  # n=40, mean~0.875 -- close enough to look "within tolerance" at n=40
        probe_lo = {"held_out_mse": 9.0, "select_squared_error": ([1.0, 1.1, 0.9, 1.0] * 10) + ([1.3, 1.4, 1.2] * 20)}  # more rows, same low-n prefix, shifted mean
        probe_hi = {"held_out_mse": 1.0, "select_squared_error": ([0.8, 0.9, 1.0, 0.8] * 10) + ([0.5, 0.6, 0.4] * 20)}
        probe_result = _probe_from_cells(_FakeDataset("synthB"), 0, 1, 8, orig_lo, orig_hi, probe_lo, probe_hi, PROBE_FRACTION_PCT)
        ok_probe = probe_result["prefix_consistent"] is True and probe_result["n_selection_probe"] > probe_result["n_selection_orig"]
        print(f"[wsel24 selftest] _probe_from_cells wiring: {'PASS' if ok_probe else 'FAIL'}")
        ok = ok and ok_probe

        print(f"[wsel24 selftest] {'PASS' if ok else 'FAIL'}")
        return ok
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Runs `--build` (disk-read attribution), `--run-probe` (Q3's opt-in training probe), or `--selftest`."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true", help="Synthetic in-memory known-answer checks, no training, then exit.")
    parser.add_argument("--build", action="store_true", help="Build attribution.json from WSEL-9's landed disk artifacts (zero compute), then exit.")
    parser.add_argument("--run-probe", action="store_true", help="Q3's compute probe: retrains --width-lo/--width-hi under an enlarged SELECT carve (2 training cells).")

    parser.add_argument("--dataset", choices=[d.value for d in w9.Dataset], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--width-lo", type=int, default=None, help="--run-probe only: the actual (cheaper) pick under test.")
    parser.add_argument("--width-hi", type=int, default=None, help="--run-probe only: the oracle-best/wider comparator.")
    parser.add_argument("--probe-fraction-pct", type=int, default=PROBE_FRACTION_PCT)
    parser.add_argument("--output", type=str, default=None, help="--run-probe only: optional path to also write the probe result JSON (never inside this task's write set).")
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--ledger-path", type=str, default=LEDGER_PATH)
    parser.add_argument("--probe-result", action="append", default=[], help="--build only: path(s) to a previously-produced --run-probe JSON, merged in under its dataset's key.")

    args = parser.parse_args()

    modes = (args.selftest, args.build, args.run_probe)
    if sum(bool(m) for m in modes) != 1:
        parser.error("Exactly one of --selftest / --build / --run-probe is required.")

    if args.selftest:
        sys.exit(0 if run_selftest() else 1)

    if args.build:
        probe_results = {}
        for path in args.probe_result:
            with open(path) as f:
                probe = json.load(f)
            probe_results[probe["dataset"]] = probe
        build_ledger(results_dir=args.results_dir, ledger_path=args.ledger_path, probe_results=probe_results)
        return

    if args.dataset is None or args.seed is None or args.width_lo is None or args.width_hi is None:
        parser.error("--run-probe requires --dataset, --seed, --width-lo and --width-hi.")
    result = run_selection_set_size_probe(
        w9.Dataset(args.dataset), args.seed, args.width_lo, args.width_hi, results_dir=args.results_dir, probe_fraction_pct=args.probe_fraction_pct
    )
    print(json.dumps(result, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[width_wsel24] wrote {args.output}")


if __name__ == "__main__":
    main()
