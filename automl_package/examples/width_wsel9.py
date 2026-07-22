"""WSEL-9 -- real data + baselines (`docs/plans/capacity_programme/width.md` WSEL-9; spec `docs/width_benchmark/benchmark_spec.md`).

Compares the three §1 width-choosing models -- W-SHARED, W-PERINPUT, W-SWEEP -- against the
baseline set the task names -- LightGBM (no architecture-vs-input-size problem, native
regularization), a plain single-output NN (the key control: same capacity as the dial net's
widest head, none of its selection machinery), and unregularized linear regression (the floor) --
on five real regression datasets frozen in the spec (`docs/width_benchmark/benchmark_spec.md`
SS1). The spec is the single source of truth for the split protocol, standardization, the
variance/objective choice, per-model configuration and the output schema; this module implements
exactly what it freezes and does not restate its rationale.

**Constants read from their artifacts, fail loudly if missing (spec SS5).** Every real cell reads
WSEL-6's selection-set fraction, WSEL-8's width ladder, and WSEL-7's router-constants artifact
(read-and-verify only -- the router is never overridden, see `_read_router_constants`'s docstring)
at startup, before any model trains, and records what it read under every output JSON's
`constants` key. Missing or drifted -> `SystemExit`, no silent default.

Six arms, one CLI cell per invocation (the root runs the grid, never this driver):
  * `--arm dial`       -- ONE multi-head `FlexibleWidthNN(widths=1..w_max)`, yielding BOTH
                          W-SHARED (`fit_global_selector`) and W-PERINPUT (`fit_router`) in one
                          JSON -- they are read off the SAME trained net (spec SS6's single-
                          difference rule: training is not a variable between them).
  * `--arm w_sweep --width K` -- ONE dedicated `FlexibleWidthNN(widths=(K,))`; W-SWEEP's own
                          dataset-level chosen width is derived at `--summarize` time once all
                          `1..w_max` cells for a (dataset, seed) have landed (each cell records its
                          own SELECT-split per-sample squared error for exactly this purpose --
                          no model reloading needed, `cheapest_within_tolerance` runs directly off
                          the stacked per-cell arrays).
  * `--arm lightgbm` / `--arm plain_nn` / `--arm linear_reg` -- the three baselines (spec SS7).

Per-cell CLI:
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel9.py \
        --dataset yacht --seed 0 --arm w_sweep --width 5
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel9.py \
        --dataset yacht --seed 0 --arm dial
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel9.py \
        --dataset yacht --seed 0 --arm lightgbm
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel9.py \
        --dataset yacht --seed 0 --arm plain_nn
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel9.py \
        --dataset yacht --seed 0 --arm linear_reg

Aggregate, after every arm has landed for a dataset (12 `w_sweep` widths + `dial` + the 3 baselines,
per seed):
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/width_wsel9.py --summarize --dataset yacht

Selftest / lint:
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel9.py --selftest

Non-goals (task brief): no real-grid execution (the root's wave-E job); no HPO/Optuna search for
any arm; no report writing (WSEL-10); no changes to `routing.py`, `capacity_accounting.py`,
`width_wsel4.py`, `width_wsel6.py`, `width_wsel8.py`, or any plan document.
"""

from __future__ import annotations

import argparse
import csv
import enum
import glob
import json
import math
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from sklearn.datasets import fetch_california_housing, fetch_openml, load_diabetes

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root, so `import automl_package` works

import converged_width_experiment as cwe  # noqa: E402 -- VAL_EVERY, the phase-1 train/val carve convention
import convergence as cvg  # noqa: E402
import width_wsel4 as w4  # noqa: E402 -- PORTED_* protocol constants + `_replay`, reused verbatim (spec SS6)

from automl_package.enums import ActivationFunction, CapacitySelection, TaskType, UncertaintyMethod  # noqa: E402
from automl_package.models.flexnn.routing import DEFAULT_HIDDEN, DEFAULT_LR, DEFAULT_N_EPOCHS  # noqa: E402 -- drift guard only, never overridden (spec SS5)
from automl_package.models.flexnn.width.model import FlexibleWidthNN  # noqa: E402
from automl_package.models.lightgbm_model import LightGBMModel  # noqa: E402
from automl_package.models.neural_network import PyTorchNeuralNetwork  # noqa: E402
from automl_package.models.normal_equation_linear_regression import NormalEquationLinearRegression  # noqa: E402
from automl_package.utils.capacity_accounting import global_cheap_cost, held_out_read_cost, per_input_cost, sweep_cost  # noqa: E402
from automl_package.utils.capacity_selection import DEFAULT_N_BOOT, cheapest_within_tolerance  # noqa: E402
from automl_package.utils.run_provenance import run_provenance  # noqa: E402

RESULTS_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "WSEL9")
WSEL6_FROZEN_PATH = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "WSEL6", "frozen.json")
WSEL7_FROZEN_PATH = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "WSEL7", "frozen.json")
WSEL8_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "WSEL8")

_DRIVER_NAME = "automl_package/examples/width_wsel9.py"

TEST_FRACTION = 0.2  # held-out REPORT split (spec SS2 step 2) -- this driver's own frozen value, no upstream artifact owns a real-data test fraction.
DEFAULT_BATCH_SIZE = 256  # mini-batch cap for every NN-based arm (spec SS6); actual batch = min(this, n_fit).
_REL_ERR_EPS = 1e-12  # floors a near-zero denominator; mirrors every sibling WSEL driver's convention.

LIGHTGBM_N_ESTIMATORS = 200  # full_benchmark.py's common_tree/model-factory convention, reused verbatim (spec SS7).
LIGHTGBM_EARLY_STOPPING_ROUNDS = 15  # same convention.
LIGHTGBM_REPLAY_MIN_DELTA = 0.0  # mirrors LightGBM's own "any improvement counts" early-stopping rule exactly.

_SELFTEST_FRACTION_PCT = 40  # arbitrary but fixed synthetic selection-set fraction, used only by run_selftest()'s known-answer check.


class Dataset(enum.Enum):
    """Closed set: the five real datasets frozen by `docs/width_benchmark/benchmark_spec.md` SS1."""

    DIABETES = "diabetes"
    YACHT = "yacht"
    ENERGY = "energy"
    KIN8NM = "kin8nm"
    CALIFORNIA = "california"


class Arm(enum.Enum):
    """Closed set: the six arms this driver trains (spec SS6/SS7); `dial` carries two model rows (w_shared, w_perinput)."""

    DIAL = "dial"  # FlexibleWidthNN(widths=1..w_max) -- yields BOTH W-SHARED and W-PERINPUT.
    W_SWEEP = "w_sweep"  # ONE dedicated FlexibleWidthNN(widths=(k,)); needs --width.
    LIGHTGBM = "lightgbm"
    PLAIN_NN = "plain_nn"
    LINEAR_REG = "linear_reg"


# ---------------------------------------------------------------------------
# Dataset loaders (spec SS1) -- four of five are `full_benchmark.py`'s own UCI loaders, reused
# verbatim (not re-derived); `diabetes` is new (bundled, no network, added for a robust selftest-
# adjacent smoke path).
# ---------------------------------------------------------------------------


def _load_diabetes() -> tuple[np.ndarray, np.ndarray]:
    d = load_diabetes()
    return d.data.astype(np.float32), d.target.astype(np.float32)


def _load_yacht() -> tuple[np.ndarray, np.ndarray]:
    """UCI yacht_hydrodynamics -- `full_benchmark.py::_uci_yacht`, reused verbatim."""
    d = fetch_openml(name="yacht_hydrodynamics", version=1, as_frame=False, parser="auto")
    return d.data.astype(np.float32), d.target.astype(np.float32)


def _load_energy() -> tuple[np.ndarray, np.ndarray]:
    """UCI energy-efficiency -- `full_benchmark.py::_uci_energy`, reused verbatim.

    This OpenML mirror folds the SECOND target (y2, cooling load) into `data` as a 9th column
    (verified: `d.feature_names[-1] == "y2"`) -- dropped here, never used as a feature.
    """
    d = fetch_openml(name="energy-efficiency", version=1, as_frame=False, parser="auto")
    x = d.data[:, :8].astype(np.float32)
    y = d.target.astype(np.float32) if d.target.ndim == 1 else d.target[:, 0].astype(np.float32)
    return x, y


def _load_kin8nm() -> tuple[np.ndarray, np.ndarray]:
    """UCI kin8nm -- `full_benchmark.py::_uci_kin8nm`, reused verbatim."""
    d = fetch_openml(name="kin8nm", version=1, as_frame=False, parser="auto")
    return d.data.astype(np.float32), d.target.astype(np.float32)


def _load_california() -> tuple[np.ndarray, np.ndarray]:
    """California housing -- `full_benchmark.py::_uci_california`, reused verbatim."""
    d = fetch_california_housing()
    return d.data.astype(np.float32), d.target.astype(np.float32)


_DATASET_LOADERS = {
    Dataset.DIABETES: _load_diabetes,
    Dataset.YACHT: _load_yacht,
    Dataset.ENERGY: _load_energy,
    Dataset.KIN8NM: _load_kin8nm,
    Dataset.CALIFORNIA: _load_california,
}


# ---------------------------------------------------------------------------
# Constants read from their artifacts (spec SS5) -- fail loudly, never a silent default.
# ---------------------------------------------------------------------------


def _read_selection_fraction(path: str) -> dict[str, Any]:
    """Reads WSEL-6's frozen selection-set fraction. `SystemExit` if the artifact is missing."""
    if not os.path.exists(path):
        raise SystemExit(
            f"{path} is missing. width.md SS3.6 requires WSEL-9 to read the selection-set fraction "
            "from WSEL-6's frozen artifact -- run WSEL-6 (or point --wsel6-path at a build that has) first."
        )
    with open(path) as f:
        frozen = json.load(f)
    fraction = float(frozen["fraction"])
    fraction_pct = int(frozen["fraction_pct"]) if "fraction_pct" in frozen else round(fraction * 100)
    return {"source": os.path.abspath(path), "fraction": fraction, "fraction_pct": fraction_pct}


def _read_width_ladder(dir_: str) -> dict[str, Any]:
    """Reads WSEL-8's frozen width ladder (`w_max`, off any of its per-cell JSONs). `SystemExit` if none exist."""
    matches = sorted(glob.glob(os.path.join(dir_, "hetero_*.json")))
    if not matches:
        raise SystemExit(
            f"No per-cell JSON found under {dir_}. width.md SS3.6 requires WSEL-9 to read the width "
            "ladder (w_max) from WSEL-8's artifact -- run WSEL-8 (or point --wsel8-dir at a build that has) first."
        )
    with open(matches[0]) as f:
        cell = json.load(f)
    if "w_max" not in cell:
        raise SystemExit(f"{matches[0]} has no 'w_max' field -- not a valid WSEL-8 per-cell JSON.")
    return {"source": os.path.abspath(matches[0]), "w_max": int(cell["w_max"])}


def _read_router_constants(path: str) -> dict[str, Any]:
    """Reads + verifies WSEL-7's router-constants artifact. `SystemExit` if missing or drifted.

    `FlexibleWidthNN.fit_router()` never takes hidden/epochs/lr arguments -- it always builds a
    `DistilledCapacityRouter` at THAT class's own constructor defaults (`routing.py:57-60`), so
    calling it plainly already IS "the frozen router defaults as-is" per width.md WSEL-7's
    2026-07-22 sign-off ruling 1 ("the frozen default STAYS ... `new_default` is NOT adopted").
    This function does not choose a config -- there is none to choose -- it verifies, at startup,
    that WSEL-7's own artifact still records that shipped default and that `routing.py`'s ACTUAL
    shipped constants have not silently drifted from it since (a fail-loud consistency check).
    """
    if not os.path.exists(path):
        raise SystemExit(
            f"{path} is missing. width.md SS3.6 requires WSEL-9's router-consuming arm (W-PERINPUT, "
            "inside --arm dial) to verify the router-constants artifact WSEL-7 owns -- run WSEL-7 "
            "(or point --wsel7-path at a build that has) first."
        )
    with open(path) as f:
        frozen = json.load(f)
    if "invariant" not in frozen or "config" not in frozen or "frozen_default_at_authoring_time" not in frozen["config"]:
        raise SystemExit(f"{path} is missing the expected WSEL-7 fields (invariant / config.frozen_default_at_authoring_time).")
    shipped = frozen["config"]["frozen_default_at_authoring_time"]
    # WSEL-7's own schema records a UNIFORM per-layer width ("hidden") + a layer count ("depth"),
    # not a list -- e.g. {"hidden": 32, "depth": 2} means a (32, 32) hidden stack. Reconstructed as
    # a tuple here so it compares directly against `DEFAULT_HIDDEN`'s own shape.
    shipped_hidden = (int(shipped["hidden"]),) * int(shipped["depth"])
    recorded = {"hidden": shipped_hidden, "depth": int(shipped["depth"]), "epochs": int(shipped["epochs"]), "lr": float(shipped["lr"])}
    actual = {"hidden": tuple(DEFAULT_HIDDEN), "depth": len(DEFAULT_HIDDEN), "epochs": DEFAULT_N_EPOCHS, "lr": DEFAULT_LR}
    if recorded != actual:
        raise SystemExit(
            f"routing.py's shipped router defaults {actual} no longer match WSEL-7's recorded "
            f"frozen_default_at_authoring_time {recorded} -- width.md WSEL-7 ruling 1 ('the frozen "
            "default STAYS') no longer describes the code as it stands; this needs a fresh ruling, "
            "not a silent re-read."
        )
    return {
        "source": os.path.abspath(path),
        "invariant": bool(frozen["invariant"]),
        "ratified_default": {"hidden": int(shipped["hidden"]), "depth": int(shipped["depth"]), "epochs": int(shipped["epochs"]), "lr": float(shipped["lr"])},
        "ruling": "width.md WSEL-7 2026-07-22 sign-off ruling 1: frozen default STAYS, new_default NOT adopted",
    }


def _read_constants(wsel6_path: str, wsel7_path: str, wsel8_dir: str) -> dict[str, Any]:
    """Bundles the three SS5 constants into the `constants` key every output JSON carries."""
    return {
        "selection_fraction": _read_selection_fraction(wsel6_path),
        "width_ladder": _read_width_ladder(wsel8_dir),
        "router": _read_router_constants(wsel7_path),
    }


# ---------------------------------------------------------------------------
# Split (spec SS2) + standardization (spec SS3) -- ONE builder for every arm, so a quality
# difference between arms reflects the model, not a data-exposure confound (MASTER Decision 15).
# ---------------------------------------------------------------------------


def _build_split(dataset: Dataset, seed: int, *, fraction_pct: int, test_fraction: float = TEST_FRACTION, max_train: int | None = None) -> dict[str, Any]:
    """Builds the TEST/FIT/STOP/SELECT split + standardization norm for one (dataset, seed).

    Deterministic in `seed` (and in `fraction_pct`/`test_fraction`/`max_train`, which must be held
    fixed between the cells that trained a model and any later `--summarize` call reconstructing
    the SAME split -- both default to the production values, so an unflagged run reproduces).
    """
    x_full, y_full = _DATASET_LOADERS[dataset]()
    n = len(x_full)
    perm = np.random.default_rng(seed).permutation(n)
    x_full, y_full = x_full[perm], y_full[perm]

    n_test = max(1, round(test_fraction * n))
    x_test, y_test_raw = x_full[:n_test], y_full[:n_test]
    x_pool, y_pool = x_full[n_test:], y_full[n_test:]
    if max_train is not None and len(x_pool) > max_train:
        x_pool, y_pool = x_pool[:max_train], y_pool[:max_train]
    n_pool = len(x_pool)

    p1_idx = np.arange(0, n_pool, 2)
    p2_idx = np.arange(1, n_pool, 2)
    x_p1, y_p1 = x_pool[p1_idx], y_pool[p1_idx]
    x_p2, y_p2 = x_pool[p2_idx], y_pool[p2_idx]

    val_mask = (np.arange(len(x_p1)) % cwe.VAL_EVERY) == 0
    x_fit, y_fit_raw = x_p1[~val_mask], y_p1[~val_mask]
    x_stop, y_stop_raw = x_p1[val_mask], y_p1[val_mask]

    perm2 = np.random.default_rng(seed).permutation(len(x_p2))
    x_p2, y_p2 = x_p2[perm2], y_p2[perm2]
    n_select = max(1, min(round((fraction_pct / 100.0) * n_pool), len(x_p2)))
    x_select, y_select_raw = x_p2[:n_select], y_p2[:n_select]

    x_mean, x_std = x_fit.mean(axis=0), x_fit.std(axis=0)
    x_std_safe = np.where(x_std > 0, x_std, 1.0)
    y_mean, y_std = float(y_fit_raw.mean()), float(y_fit_raw.std())
    y_std_safe = y_std if y_std > 0 else 1.0

    def _std_x(x: np.ndarray) -> np.ndarray:
        return ((x - x_mean) / x_std_safe).astype(np.float32)

    def _std_y(y: np.ndarray) -> np.ndarray:
        return ((y - y_mean) / y_std_safe).astype(np.float32)

    return {
        "d": int(x_full.shape[1]),
        "n_total": n,
        "n_test": n_test,
        "n_pool": n_pool,
        "n_selection_used": n_select,
        "x_fit": _std_x(x_fit),
        "y_fit_raw": y_fit_raw.astype(np.float32),
        "y_fit_std": _std_y(y_fit_raw),
        "x_stop": _std_x(x_stop),
        "y_stop_raw": y_stop_raw.astype(np.float32),
        "y_stop_std": _std_y(y_stop_raw),
        "x_select": _std_x(x_select),
        "y_select_raw": y_select_raw.astype(np.float32),
        "y_select_std": _std_y(y_select_raw),
        "x_test": _std_x(x_test),
        "y_test_raw": y_test_raw.astype(np.float32),
        "norm": {"x_mean": x_mean.tolist(), "x_std": x_std_safe.tolist(), "y_mean": y_mean, "y_std": y_std_safe},
        "config": {"test_fraction": test_fraction, "max_train": max_train, "fraction_pct": fraction_pct},
    }


# ---------------------------------------------------------------------------
# Width-model training (spec SS6) -- the established `_fit_single` bypass (`width_wsel4._train_
# ported_default`'s pattern) for W-SWEEP's single-width nets, and a per-width simultaneous-
# convergence loop (this driver's generalization of `width_wsel8._train_shared_to_convergence` to
# arbitrary `input_size=d`) for the dial net.
# ---------------------------------------------------------------------------


def _train_flexwidth_single(
    widths: tuple[int, ...], seed: int, d: int, x_fit: np.ndarray, y_fit: np.ndarray, x_stop: np.ndarray, y_stop: np.ndarray, *, max_epochs: int, batch_size: int
) -> tuple[FlexibleWidthNN, list[float]]:
    """ONE `FlexibleWidthNN(widths=widths)`, `width_wsel4._train_ported_default`'s bypass, `input_size=d`."""
    model = FlexibleWidthNN(
        input_size=d,
        output_size=1,
        task_type=TaskType.REGRESSION,
        widths=widths,
        learning_rate=w4.PORTED_LR_DEFAULT,
        n_epochs=max_epochs,
        early_stopping_rounds=w4.PORTED_PATIENCE,
        batch_size=batch_size,
        random_seed=seed,
        calculate_feature_importance=False,
        capacity_selection=CapacitySelection.FIXED,
        activation=ActivationFunction.TANH,  # spec SS6: this strand's established width-net convention.
    )
    _best_epoch, val_loss_history = model._fit_single(x_fit, y_fit, x_val=x_stop, y_val=y_stop)
    return model, val_loss_history


def _train_dial_to_convergence(
    widths: tuple[int, ...], seed: int, d: int, x_fit: np.ndarray, y_fit: np.ndarray, x_stop: np.ndarray, y_stop: np.ndarray, *, max_epochs: int, batch_size: int
) -> tuple[FlexibleWidthNN, dict[int, cvg.ConvergenceResult]]:
    """Joint multi-width training for the dial net (W-SHARED and W-PERINPUT share this ONE net).

    Generalizes `width_wsel8._train_shared_to_convergence` to arbitrary input dimensionality `d`
    (that function hardcodes `input_size=1` for its scalar toy input). The per-width stop rule
    itself -- ONE `ConvergenceTracker` per width, stop only once EVERY width is simultaneously
    trustworthy-or-diverged AT THE SAME checkpoint, never merely once each has EVER latched `done`
    -- is copied verbatim: `FlexibleWidthNNModule.forward` sums every configured width's loss every
    step regardless of `d`, so the same aggregate-scalar blind spot WSEL-8 diagnosed applies here
    unchanged.
    """
    model = FlexibleWidthNN(
        input_size=d,
        output_size=1,
        task_type=TaskType.REGRESSION,
        widths=widths,
        learning_rate=w4.PORTED_LR_DEFAULT,
        n_epochs=max_epochs,
        early_stopping_rounds=w4.PORTED_PATIENCE,
        batch_size=batch_size,
        random_seed=seed,
        calculate_feature_importance=False,
        capacity_selection=CapacitySelection.FIXED,
        activation=ActivationFunction.TANH,
    )
    model.input_size = d
    torch.manual_seed(seed)
    model.build_model()
    opt = torch.optim.Adam(model.model.parameters(), lr=w4.PORTED_LR_DEFAULT)

    x_fit_t = torch.tensor(x_fit, dtype=torch.float32, device=model.device)
    y_fit_t = torch.tensor(y_fit, dtype=torch.float32, device=model.device)
    x_stop_t = torch.tensor(x_stop, dtype=torch.float32, device=model.device)
    y_stop_t = torch.tensor(y_stop, dtype=torch.float32, device=model.device)
    dataset_t = torch.utils.data.TensorDataset(x_fit_t, y_fit_t)

    trackers = {w: cvg.ConvergenceTracker(patience=w4.PORTED_PATIENCE, min_delta=w4.PORTED_MIN_DELTA) for w in widths}
    best_mean_val = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    final_epoch = max_epochs

    for epoch in range(1, max_epochs + 1):
        model.model.train()
        loader = torch.utils.data.DataLoader(dataset_t, batch_size=batch_size, shuffle=True)
        for batch_x, batch_y in loader:
            opt.zero_grad()
            stacked = model.model(batch_x)  # (len(widths), N, 1) -- every configured width, every step.
            total = stacked.new_zeros(())
            for i in range(stacked.shape[0]):
                total = total + ((stacked[i].squeeze(1) - batch_y) ** 2).mean()
            total.backward()
            opt.step()

        model.model.eval()
        with torch.no_grad():
            stacked_val = model.model(x_stop_t)
            per_width_val = {w: float(((stacked_val[i].squeeze(1) - y_stop_t) ** 2).mean().item()) for i, w in enumerate(widths)}
        for w, v in per_width_val.items():
            trackers[w].update(epoch, v)
        mean_val = sum(per_width_val.values()) / len(widths)
        if mean_val < best_mean_val:
            best_mean_val = mean_val
            best_state = {name: t.detach().clone() for name, t in model.model.state_dict().items()}

        results_now = {w: trackers[w].result(final_epoch=epoch) for w in widths}
        if all(r.trustworthy or r.diverged for r in results_now.values()):
            final_epoch = epoch
            break

    if best_state is not None:
        model.model.load_state_dict(best_state)
    model.model.eval()
    return model, {w: trackers[w].result(final_epoch=final_epoch) for w in widths}


def _flexwidth_module_for_cost(d: int, seed: int, width: int) -> torch.nn.Module:
    """A throwaway, UNTRAINED `FlexibleWidthNN(widths=(width,))`, built only for its architecture SHAPE.

    `executed_flops`/`param_count` are analytic MAC/param counts of a module's shape (layer sizes),
    never its trained weight values -- so a freshly-built net of the same `(input_size, widths)`
    configuration prices identically to the one actually trained at that cell. Used by
    `summarize()` to re-derive W-SWEEP's per-width SELECT-read cost without reloading any saved
    state (SS9 never caches trained weights -- see the module docstring's W-SWEEP arm note).
    """
    model = FlexibleWidthNN(input_size=d, output_size=1, task_type=TaskType.REGRESSION, widths=(width,), random_seed=seed, calculate_feature_importance=False)
    model.input_size = d
    model.build_model()
    return model.model


def _width_model_config(d: int, batch_size: int, epoch_cap: int, split: dict[str, Any]) -> dict[str, Any]:
    """The `config` sub-dict every width-net arm (`w_sweep`, `dial`) records -- one shared shape, not restated per caller."""
    return {
        "d": d,
        "lr": w4.PORTED_LR_DEFAULT,
        "patience": w4.PORTED_PATIENCE,
        "min_delta": w4.PORTED_MIN_DELTA,
        "batch_size": batch_size,
        "epoch_cap": epoch_cap,
        **split["config"],
    }


# ---------------------------------------------------------------------------
# Per-cell runners
# ---------------------------------------------------------------------------


def _run_w_sweep_cell(dataset: Dataset, seed: int, width: int, split: dict[str, Any], constants: dict[str, Any], *, epoch_cap: int) -> dict[str, Any]:
    """Trains ONE dedicated `FlexibleWidthNN(widths=(width,))`; scores TEST + records SELECT per-sample error.

    The SELECT-split per-sample squared error (standardized-y scale) is stored so `summarize()` can
    assemble the (n_selection_used, w_max) error table across all 12 widths' JSONs and run
    `cheapest_within_tolerance` WITHOUT reloading any trained model.
    """
    d = split["d"]
    batch_size = min(DEFAULT_BATCH_SIZE, len(split["x_fit"]))
    model, val_loss_history = _train_flexwidth_single(
        (width,), seed, d, split["x_fit"], split["y_fit_std"], split["x_stop"], split["y_stop_std"], max_epochs=epoch_cap, batch_size=batch_size
    )
    replay = w4._replay(val_loss_history, w4.PORTED_PATIENCE, w4.PORTED_MIN_DELTA)
    hit_cap = bool(len(val_loss_history) >= epoch_cap)
    trustworthy = bool(replay.trustworthy and not hit_cap)

    pred_test_std = model.predict(split["x_test"], filter_data=False, width=width)
    pred_test_raw = pred_test_std * split["norm"]["y_std"] + split["norm"]["y_mean"]
    held_out_mse = float(np.mean((pred_test_raw - split["y_test_raw"]) ** 2))

    pred_select_std = model.predict(split["x_select"], filter_data=False, width=width)
    select_squared_error = ((pred_select_std - split["y_select_std"]) ** 2).tolist()

    n_train_used = len(split["x_fit"])
    training_macs = sweep_cost(model.model, [width], n_train_used, len(val_loss_history))

    return {
        "dataset": dataset.value,
        "seed": seed,
        "arm": Arm.W_SWEEP.value,
        "width": width,
        "held_out_mse": held_out_mse,
        "held_out_trajectory": replay.summary()["trajectory"],
        "trustworthy": trustworthy,
        "hit_cap": hit_cap,
        "converged": replay.converged,
        "trajectory_applicable": True,
        "select_squared_error": select_squared_error,
        "n_selection_used": split["n_selection_used"],
        "training_macs": training_macs,
        "n_train_used": n_train_used,
        "actual_epochs": len(val_loss_history),
        "selection_cost": None,  # per-width cell isn't W-SWEEP's priced answer -- summarize() aggregates all 12 (spec SS8).
        "config": _width_model_config(d, batch_size, epoch_cap, split),
        "constants": constants,
        "provenance": run_provenance(),
    }


def _run_dial_cell(dataset: Dataset, seed: int, split: dict[str, Any], constants: dict[str, Any], *, epoch_cap: int) -> dict[str, Any]:
    """Trains ONE multi-head dial net; derives BOTH W-SHARED and W-PERINPUT off it (spec SS6)."""
    d = split["d"]
    w_max = constants["width_ladder"]["w_max"]
    widths = tuple(range(1, w_max + 1))
    batch_size = min(DEFAULT_BATCH_SIZE, len(split["x_fit"]))

    model, per_width_results = _train_dial_to_convergence(
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

    # W-SHARED -- fit_global_selector on SELECT, scored on TEST.
    w_shared_width = model.fit_global_selector(x_sel, y_sel_std, seed=seed)
    model.capacity_selection = CapacitySelection.GLOBAL_CHEAP
    w_shared_pred_raw = model.predict(x_test, filter_data=False) * y_std + y_mean
    w_shared_mse = float(np.mean((w_shared_pred_raw - y_test_raw) ** 2))
    shared_cost = global_cheap_cost(training_macs=training_macs, net=model.model, capacity_grid=list(widths), n_samples=len(x_sel))

    # W-PERINPUT -- fit_router on the SAME SELECT set, read off the SAME net (no hyperparameter override, SS5).
    model.fit_router(x_sel, y_sel_std)
    model.capacity_selection = CapacitySelection.PER_INPUT
    w_perinput_pred_raw = model.predict(x_test, filter_data=False) * y_std + y_mean
    w_perinput_mse = float(np.mean((w_perinput_pred_raw - y_test_raw) ** 2))
    router = model.capacity_router_
    perinput_cost = per_input_cost(training_macs=training_macs, in_dim=d, n_capacities=len(widths), n_samples=len(x_sel), n_epochs=router.n_epochs, hidden=router.hidden)
    routed_widths = [capacity[0] for capacity in router.route(x_test)]
    width_distribution = {str(w): routed_widths.count(w) for w in sorted(set(routed_widths))}

    return {
        "dataset": dataset.value,
        "seed": seed,
        "arm": Arm.DIAL.value,
        "w_max": w_max,
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
            "selection": {"fraction_pct": constants["selection_fraction"]["fraction_pct"], "n_selection_used": len(x_sel)},
        },
        "w_perinput": {
            "mean_routed_width": float(np.mean(routed_widths)),
            "width_distribution": width_distribution,
            "held_out_mse": w_perinput_mse,
            "selection_cost": {"training_macs": perinput_cost.training_macs, "selection_macs": perinput_cost.selection_macs, "total_macs": perinput_cost.total_macs},
            "selection": {"fraction_pct": constants["selection_fraction"]["fraction_pct"], "n_selection_used": len(x_sel)},
        },
        "config": _width_model_config(d, batch_size, epoch_cap, split),
        "constants": constants,
        "provenance": run_provenance(),
    }


def _run_lightgbm_cell(dataset: Dataset, seed: int, split: dict[str, Any], constants: dict[str, Any]) -> dict[str, Any]:
    """LightGBM baseline -- `full_benchmark.py`'s established config, reused verbatim (spec SS7)."""
    model = LightGBMModel(
        n_estimators=LIGHTGBM_N_ESTIMATORS, early_stopping_rounds=LIGHTGBM_EARLY_STOPPING_ROUNDS, random_seed=seed, verbose=-1, calculate_feature_importance=False
    )
    best_iteration, loss_history = model._fit_single(split["x_fit"], split["y_fit_raw"], x_val=split["x_stop"], y_val=split["y_stop_raw"])
    replay = w4._replay(loss_history, LIGHTGBM_EARLY_STOPPING_ROUNDS, LIGHTGBM_REPLAY_MIN_DELTA)
    hit_cap = bool(best_iteration >= LIGHTGBM_N_ESTIMATORS)
    trustworthy = bool(replay.trustworthy and not hit_cap)

    pred = np.asarray(model.predict(split["x_test"], filter_data=False)).reshape(-1)
    held_out_mse = float(np.mean((pred - split["y_test_raw"]) ** 2))

    return {
        "dataset": dataset.value,
        "seed": seed,
        "arm": Arm.LIGHTGBM.value,
        "width": None,
        "held_out_mse": held_out_mse,
        "held_out_trajectory": replay.summary()["trajectory"],
        "trustworthy": trustworthy,
        "hit_cap": hit_cap,
        "converged": replay.converged,
        "trajectory_applicable": True,
        "selection_cost": None,
        "config": {"n_estimators": LIGHTGBM_N_ESTIMATORS, "early_stopping_rounds": LIGHTGBM_EARLY_STOPPING_ROUNDS, "best_iteration": int(best_iteration), **split["config"]},
        "constants": constants,
        "provenance": run_provenance(),
    }


def _run_plain_nn_cell(dataset: Dataset, seed: int, split: dict[str, Any], constants: dict[str, Any], *, epoch_cap: int) -> dict[str, Any]:
    """Plain single-output NN baseline -- same protocol as the width nets, no dial machinery (spec SS7)."""
    d = split["d"]
    w_max = constants["width_ladder"]["w_max"]
    batch_size = min(DEFAULT_BATCH_SIZE, len(split["x_fit"]))
    model = PyTorchNeuralNetwork(
        input_size=d,
        output_size=1,
        task_type=TaskType.REGRESSION,
        hidden_layers=1,
        hidden_size=w_max,  # "the dial network at fixed width ~= this" -- same capacity as its widest configured head.
        activation=ActivationFunction.TANH,
        uncertainty_method=UncertaintyMethod.CONSTANT,  # no variance head anywhere in this battery (spec SS4).
        learning_rate=w4.PORTED_LR_DEFAULT,
        n_epochs=epoch_cap,
        early_stopping_rounds=w4.PORTED_PATIENCE,
        batch_size=batch_size,
        random_seed=seed,
        calculate_feature_importance=False,
    )
    _best_epoch, val_loss_history = model._fit_single(split["x_fit"], split["y_fit_std"], x_val=split["x_stop"], y_val=split["y_stop_std"])
    replay = w4._replay(val_loss_history, w4.PORTED_PATIENCE, w4.PORTED_MIN_DELTA)
    hit_cap = bool(len(val_loss_history) >= epoch_cap)
    trustworthy = bool(replay.trustworthy and not hit_cap)

    pred_raw = np.asarray(model.predict(split["x_test"], filter_data=False)).reshape(-1) * split["norm"]["y_std"] + split["norm"]["y_mean"]
    held_out_mse = float(np.mean((pred_raw - split["y_test_raw"]) ** 2))

    return {
        "dataset": dataset.value,
        "seed": seed,
        "arm": Arm.PLAIN_NN.value,
        "width": None,
        "held_out_mse": held_out_mse,
        "held_out_trajectory": replay.summary()["trajectory"],
        "trustworthy": trustworthy,
        "hit_cap": hit_cap,
        "converged": replay.converged,
        "trajectory_applicable": True,
        "selection_cost": None,  # no capacity-selection mechanism to price (spec SS8) -- a fixed architecture, not a system that chooses.
        "config": {
            "hidden_layers": 1,
            "hidden_size": w_max,
            "lr": w4.PORTED_LR_DEFAULT,
            "patience": w4.PORTED_PATIENCE,
            "batch_size": batch_size,
            "epoch_cap": epoch_cap,
            **split["config"],
        },
        "constants": constants,
        "provenance": run_provenance(),
    }


def _run_linear_reg_cell(dataset: Dataset, seed: int, split: dict[str, Any], constants: dict[str, Any]) -> dict[str, Any]:
    """Linear regression floor -- unregularized OLS via the normal equation (spec SS7)."""
    model = NormalEquationLinearRegression(l2_lambda=0.0, calculate_feature_importance=False)
    _n_iter, loss_history = model._fit_single(split["x_fit"], split["y_fit_raw"])
    if loss_history:
        # NormalEquationLinearRegression._fit_single is a direct closed-form solve and returns (1, [])
        # by construction (normal_equation_linear_regression.py:51-86) -- a non-empty history here
        # would mean that contract changed underneath this driver, so fail loudly rather than
        # silently treat a real trajectory as "not applicable".
        raise RuntimeError(f"NormalEquationLinearRegression._fit_single returned a non-empty loss_history ({loss_history!r}); spec SS7's 'no trajectory' claim no longer holds.")

    pred = np.asarray(model.predict(split["x_test"], filter_data=False)).reshape(-1)
    held_out_mse = float(np.mean((pred - split["y_test_raw"]) ** 2))

    return {
        "dataset": dataset.value,
        "seed": seed,
        "arm": Arm.LINEAR_REG.value,
        "width": None,
        "held_out_mse": held_out_mse,
        "held_out_trajectory": [],
        "trustworthy": True,
        "hit_cap": False,
        "converged": True,
        "trajectory_applicable": False,
        "note": "Closed-form OLS solve (NormalEquationLinearRegression._fit_single) -- no iterative trajectory exists to converge "
        "on; trustworthy trivially, not because it was checked.",
        "selection_cost": None,
        "config": {"l2_lambda": 0.0, **split["config"]},
        "constants": constants,
        "provenance": run_provenance(),
    }


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
# --summarize -- per-dataset CSV (spec SS9). W-SWEEP's dataset-level chosen width is derived HERE,
# once all 1..w_max cells have landed, by stacking their stored `select_squared_error` arrays --
# no model reloading needed.
# ---------------------------------------------------------------------------

_CSV_FIELDS = ["dataset", "seed", "arm", "width", "held_out_mse", "training_macs", "selection_macs", "total_macs", "trustworthy", "hit_cap", "converged", "n_selection_used"]


def _csv_row(
    dataset: Dataset,
    seed: int,
    arm: str,
    width: int | None,
    held_out_mse: float,
    cost: dict[str, int] | None,
    trustworthy: bool,
    hit_cap: bool,
    converged: bool,
    n_sel: int | None,
) -> dict[str, Any]:
    return {
        "dataset": dataset.value,
        "seed": seed,
        "arm": arm,
        "width": width if width is not None else "",
        "held_out_mse": held_out_mse,
        "training_macs": cost["training_macs"] if cost else "",
        "selection_macs": cost["selection_macs"] if cost else "",
        "total_macs": cost["total_macs"] if cost else "",
        "trustworthy": trustworthy,
        "hit_cap": hit_cap,
        "converged": converged,
        "n_selection_used": n_sel if n_sel is not None else "",
    }


def summarize(dataset: Dataset, results_dir: str = RESULTS_DIR) -> str:
    """Aggregates every landed cell for `dataset` into `WSEL9/<dataset>_summary.csv`."""
    prefix = dataset.value
    dial_cells: dict[int, dict[str, Any]] = {}
    sweep_cells: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    other_cells: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)

    for path in sorted(glob.glob(os.path.join(results_dir, f"{prefix}_*.json"))):
        if os.path.basename(path).endswith("_summary.csv"):
            continue
        with open(path) as f:
            cell = json.load(f)
        seed = int(cell["seed"])
        arm = cell["arm"]
        if arm == Arm.DIAL.value:
            dial_cells[seed] = cell
        elif arm == Arm.W_SWEEP.value:
            sweep_cells[seed][int(cell["width"])] = cell
        else:
            other_cells[seed][arm] = cell

    seeds = sorted(set(dial_cells) | set(sweep_cells) | set(other_cells))
    rows: list[dict[str, Any]] = []

    for seed in seeds:
        if seed in dial_cells:
            dc = dial_cells[seed]
            ws, wp = dc["w_shared"], dc["w_perinput"]
            st = dc["shared_training"]
            shared_flags = (st["trustworthy"], st["hit_cap"], st["converged"])  # the dial net is ONE training run -- W-SHARED and W-PERINPUT share its verdict.
            rows.append(_csv_row(dataset, seed, "w_shared", ws["selected_width"], ws["held_out_mse"], ws["selection_cost"], *shared_flags, ws["selection"]["n_selection_used"]))
            rows.append(_csv_row(dataset, seed, "w_perinput", None, wp["held_out_mse"], wp["selection_cost"], *shared_flags, wp["selection"]["n_selection_used"]))

        widths_present = sweep_cells.get(seed, {})
        w_max = max(widths_present) if widths_present else None
        if widths_present and w_max is not None and len(widths_present) == w_max and set(widths_present) == set(range(1, w_max + 1)):
            error_cols = []
            n_sel_values = set()
            for w in range(1, w_max + 1):
                cell = widths_present[w]
                n_sel_values.add(len(cell["select_squared_error"]))
                error_cols.append(cell["select_squared_error"])
            if len(n_sel_values) != 1:
                print(
                    f"[width_wsel9] WARNING: dataset={prefix} seed={seed} w_sweep cells disagree on n_selection_used "
                    f"({sorted(n_sel_values)}) -- skipping the w_sweep row (rebuild with consistent --max-train/--test-fraction)."
                )
            else:
                error_table = np.array(error_cols, dtype=np.float64).T  # (n_selection_used, w_max), cheapest-first.
                idx = cheapest_within_tolerance(error_table, n_boot=DEFAULT_N_BOOT, seed=seed)
                chosen_width = idx + 1
                chosen_cell = widths_present[chosen_width]
                d = chosen_cell["config"]["d"]  # recorded on every w_sweep cell at write time (see _run_w_sweep_cell).
                n_sel_used = len(chosen_cell["select_squared_error"])

                training_macs = sum(widths_present[w]["training_macs"] for w in range(1, w_max + 1))
                selection_macs = sum(held_out_read_cost(_flexwidth_module_for_cost(d, seed, w), [w], n_sel_used) for w in range(1, w_max + 1))
                trustworthy_all = all(widths_present[w]["trustworthy"] for w in range(1, w_max + 1))
                hit_cap_any = any(widths_present[w]["hit_cap"] for w in range(1, w_max + 1))
                converged_all = all(widths_present[w]["converged"] for w in range(1, w_max + 1))
                cost = {"training_macs": training_macs, "selection_macs": selection_macs, "total_macs": training_macs + selection_macs}
                rows.append(_csv_row(dataset, seed, "w_sweep", chosen_width, chosen_cell["held_out_mse"], cost, trustworthy_all, hit_cap_any, converged_all, n_sel_used))
        elif widths_present:
            print(f"[width_wsel9] dataset={prefix} seed={seed}: {len(widths_present)} w_sweep width(s) landed, incomplete (need every 1..w_max) -- skipping for now.")

        for arm_name, cell in other_cells.get(seed, {}).items():
            rows.append(
                _csv_row(dataset, seed, arm_name, cell.get("width"), cell["held_out_mse"], cell["selection_cost"], cell["trustworthy"], cell["hit_cap"], cell["converged"], None)
            )

    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"{prefix}_summary.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[width_wsel9] wrote {out_path} ({len(rows)} rows, seeds={seeds})")
    return out_path


# ---------------------------------------------------------------------------
# Selftest -- tiny synthetic array (NOT a real dataset; fast, deterministic, offline), every arm,
# a synthetic copy of the three SS5 artifacts (known-answer + missing-artifact refusal), --summarize.
# ---------------------------------------------------------------------------


def _write_synthetic_constants(tmp_dir: str, *, w_max: int) -> tuple[str, str, str]:
    """Writes tiny synthetic WSEL6/WSEL7/WSEL8 artifacts into `tmp_dir`, mirroring `width_wsel8.run_selftest`'s pattern."""
    wsel6_path = os.path.join(tmp_dir, "wsel6_frozen.json")
    with open(wsel6_path, "w") as f:
        json.dump({"fraction": _SELFTEST_FRACTION_PCT / 100.0, "fraction_pct": _SELFTEST_FRACTION_PCT}, f)

    wsel7_path = os.path.join(tmp_dir, "wsel7_frozen.json")
    with open(wsel7_path, "w") as f:
        # Real schema (verified against the landed WSEL7/frozen.json): "hidden" is the UNIFORM
        # per-layer width, "depth" the layer count -- e.g. {"hidden": 32, "depth": 2} for (32, 32).
        json.dump(
            {
                "invariant": False,
                "config": {"frozen_default_at_authoring_time": {"hidden": DEFAULT_HIDDEN[0], "depth": len(DEFAULT_HIDDEN), "epochs": DEFAULT_N_EPOCHS, "lr": DEFAULT_LR}},
            },
            f,
        )

    wsel8_dir = os.path.join(tmp_dir, "wsel8")
    os.makedirs(wsel8_dir, exist_ok=True)
    with open(os.path.join(wsel8_dir, "hetero_0.json"), "w") as f:
        json.dump({"w_max": w_max}, f)

    return wsel6_path, wsel7_path, wsel8_dir


def run_selftest() -> bool:
    """Tiny end-to-end pass: every arm on a synthetic array, the constants fail-loud gate (both ways), --summarize."""
    tmp_dir = tempfile.mkdtemp(prefix="width_wsel9_selftest_")
    try:
        rng = np.random.default_rng(0)
        n, d, w_max, seed = 80, 3, 3, 0
        x_synth = rng.normal(size=(n, d)).astype(np.float32)
        true_w = rng.normal(size=d).astype(np.float32)
        y_synth = x_synth @ true_w + 0.1 * rng.normal(size=n).astype(np.float32)

        class _FakeDataset:
            value = "synthtest"

        _DATASET_LOADERS[_FakeDataset] = lambda: (x_synth, y_synth)  # type: ignore[index]

        wsel6_path, wsel7_path, wsel8_dir = _write_synthetic_constants(tmp_dir, w_max=w_max)

        # Missing-artifact refusal MUST fire before anything trains.
        bad_paths = (
            {"wsel6_path": os.path.join(tmp_dir, "nope.json"), "wsel7_path": wsel7_path, "wsel8_dir": wsel8_dir},
            {"wsel6_path": wsel6_path, "wsel7_path": os.path.join(tmp_dir, "nope.json"), "wsel8_dir": wsel8_dir},
            {"wsel6_path": wsel6_path, "wsel7_path": wsel7_path, "wsel8_dir": os.path.join(tmp_dir, "nope")},
        )
        gate_refused = []
        for bad_kwargs in bad_paths:
            try:
                _read_constants(**bad_kwargs)
                gate_refused.append(False)
            except SystemExit:
                gate_refused.append(True)
        ok_gate = all(gate_refused)
        print(f"[wsel9 selftest] missing-artifact refusal (3/3 paths): {'PASS' if ok_gate else 'FAIL'}")

        constants = _read_constants(wsel6_path, wsel7_path, wsel8_dir)
        ok_constants = (
            constants["width_ladder"]["w_max"] == w_max
            and constants["selection_fraction"]["fraction_pct"] == _SELFTEST_FRACTION_PCT
            and constants["router"]["invariant"] is False
        )
        print(f"[wsel9 selftest] constants read (known-answer): {'PASS' if ok_constants else 'FAIL'}")

        split = _build_split(_FakeDataset, seed, fraction_pct=constants["selection_fraction"]["fraction_pct"], max_train=None)  # type: ignore[arg-type]
        ok_split = split["d"] == d and len(split["x_fit"]) > 0 and len(split["x_stop"]) > 0 and len(split["x_select"]) > 0 and len(split["x_test"]) > 0
        split_sizes = f"fit={len(split['x_fit'])} stop={len(split['x_stop'])} select={len(split['x_select'])} test={len(split['x_test'])}"
        print(f"[wsel9 selftest] split (d={split['d']} {split_sizes}): {'PASS' if ok_split else 'FAIL'}")

        epoch_cap = 30
        results_dir = os.path.join(tmp_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        ok_cells = True

        for width in range(1, w_max + 1):
            cell = _run_w_sweep_cell(_FakeDataset, seed, width, split, constants, epoch_cap=epoch_cap)  # type: ignore[arg-type]
            with open(os.path.join(results_dir, f"synthtest_{seed}_w_sweep_{width}.json"), "w") as f:
                json.dump(_jsonable(cell), f, indent=2)
            cell_ok = math.isfinite(cell["held_out_mse"]) and isinstance(cell["trustworthy"], bool) and len(cell["select_squared_error"]) == split["n_selection_used"]
            ok_cells = ok_cells and cell_ok
        print(f"[wsel9 selftest] w_sweep (1..{w_max}): {'PASS' if ok_cells else 'FAIL'}")

        dial_cell = _run_dial_cell(_FakeDataset, seed, split, constants, epoch_cap=epoch_cap)  # type: ignore[arg-type]
        with open(os.path.join(results_dir, f"synthtest_{seed}_dial.json"), "w") as f:
            json.dump(_jsonable(dial_cell), f, indent=2)
        ok_dial = (
            1 <= dial_cell["w_shared"]["selected_width"] <= w_max
            and math.isfinite(dial_cell["w_shared"]["held_out_mse"])
            and math.isfinite(dial_cell["w_perinput"]["held_out_mse"])
            and dial_cell["w_shared"]["selection_cost"]["total_macs"] > 0
            and dial_cell["w_perinput"]["selection_cost"]["total_macs"] > 0
        )
        print(f"[wsel9 selftest] dial (w_shared={dial_cell['w_shared']['selected_width']}): {'PASS' if ok_dial else 'FAIL'}")

        lgbm_cell = _run_lightgbm_cell(_FakeDataset, seed, split, constants)  # type: ignore[arg-type]
        with open(os.path.join(results_dir, f"synthtest_{seed}_lightgbm.json"), "w") as f:
            json.dump(_jsonable(lgbm_cell), f, indent=2)
        ok_lgbm = math.isfinite(lgbm_cell["held_out_mse"]) and lgbm_cell["selection_cost"] is None
        print(f"[wsel9 selftest] lightgbm: {'PASS' if ok_lgbm else 'FAIL'}")

        nn_cell = _run_plain_nn_cell(_FakeDataset, seed, split, constants, epoch_cap=epoch_cap)  # type: ignore[arg-type]
        with open(os.path.join(results_dir, f"synthtest_{seed}_plain_nn.json"), "w") as f:
            json.dump(_jsonable(nn_cell), f, indent=2)
        ok_nn = math.isfinite(nn_cell["held_out_mse"]) and isinstance(nn_cell["trustworthy"], bool)
        print(f"[wsel9 selftest] plain_nn: {'PASS' if ok_nn else 'FAIL'}")

        lin_cell = _run_linear_reg_cell(_FakeDataset, seed, split, constants)  # type: ignore[arg-type]
        with open(os.path.join(results_dir, f"synthtest_{seed}_linear_reg.json"), "w") as f:
            json.dump(_jsonable(lin_cell), f, indent=2)
        ok_lin = math.isfinite(lin_cell["held_out_mse"]) and lin_cell["held_out_trajectory"] == [] and lin_cell["trajectory_applicable"] is False
        print(f"[wsel9 selftest] linear_reg: {'PASS' if ok_lin else 'FAIL'}")

        csv_path = summarize(_FakeDataset, results_dir=results_dir)  # type: ignore[arg-type]
        with open(csv_path) as f:
            csv_rows = list(csv.DictReader(f))
        arms_present = {row["arm"] for row in csv_rows}
        ok_summarize = {"w_shared", "w_perinput", "w_sweep", "lightgbm", "plain_nn", "linear_reg"} <= arms_present
        print(f"[wsel9 selftest] summarize (arms={sorted(arms_present)}): {'PASS' if ok_summarize else 'FAIL'}")

        ok = ok_gate and ok_constants and ok_split and ok_cells and ok_dial and ok_lgbm and ok_nn and ok_lin and ok_summarize
        print(f"[wsel9 selftest] {'PASS' if ok else 'FAIL'}")
        return ok
    finally:
        _DATASET_LOADERS.pop(_FakeDataset, None) if "_FakeDataset" in dir() else None
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Runs one per-cell training (`--arm dial`/`w_sweep`/`lightgbm`/`plain_nn`/`linear_reg`), `--summarize`, or `--selftest`."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true", help="Tiny end-to-end wiring check on a synthetic array, then exit.")
    parser.add_argument("--summarize", action="store_true", help="Aggregate landed cells for --dataset into <dataset>_summary.csv, then exit.")

    parser.add_argument("--dataset", choices=[d.value for d in Dataset], default=None, help="Required outside --selftest.")
    parser.add_argument("--seed", type=int, default=None, help="Required for a real cell (not --summarize/--selftest).")
    parser.add_argument("--arm", choices=[a.value for a in Arm], default=None, help="Required for a real cell.")
    parser.add_argument("--width", type=int, default=None, help="Required for --arm w_sweep; must not be given otherwise.")
    parser.add_argument("--tag", type=str, default=None, help="Optional filename suffix (e.g. 'authsmoke') so smoke cells are easy to find and delete.")

    parser.add_argument("--epoch-cap", type=int, default=w4.PORTED_N_EPOCHS_CAP, help="Safety cap on training epochs for any NN-based arm.")
    parser.add_argument("--max-train", type=int, default=None, help="Subsamples the FIT/STOP/SELECT pool (never TEST) for wall-clock reasons; recorded in provenance.")
    parser.add_argument("--test-fraction", type=float, default=TEST_FRACTION)
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--wsel6-path", type=str, default=WSEL6_FROZEN_PATH)
    parser.add_argument("--wsel7-path", type=str, default=WSEL7_FROZEN_PATH)
    parser.add_argument("--wsel8-dir", type=str, default=WSEL8_DIR)

    args = parser.parse_args()

    modes = (args.selftest, args.summarize)
    if sum(bool(m) for m in modes) > 1:
        parser.error("--selftest and --summarize are mutually exclusive.")
    if args.selftest:
        sys.exit(0 if run_selftest() else 1)
    if args.summarize:
        if args.dataset is None:
            parser.error("--dataset is required with --summarize.")
        summarize(Dataset(args.dataset), results_dir=args.results_dir)
        return

    if args.dataset is None or args.seed is None or args.arm is None:
        parser.error("--dataset, --seed and --arm are required for a real cell (or pass --selftest / --summarize).")
    dataset = Dataset(args.dataset)
    arm = Arm(args.arm)
    if arm is Arm.W_SWEEP and args.width is None:
        parser.error("--width is required for --arm w_sweep.")
    if arm is not Arm.W_SWEEP and args.width is not None:
        parser.error("--width only applies to --arm w_sweep.")

    constants = _read_constants(args.wsel6_path, args.wsel7_path, args.wsel8_dir)
    split = _build_split(dataset, args.seed, fraction_pct=constants["selection_fraction"]["fraction_pct"], test_fraction=args.test_fraction, max_train=args.max_train)

    if arm is Arm.W_SWEEP:
        cell = _run_w_sweep_cell(dataset, args.seed, args.width, split, constants, epoch_cap=args.epoch_cap)
    elif arm is Arm.DIAL:
        cell = _run_dial_cell(dataset, args.seed, split, constants, epoch_cap=args.epoch_cap)
    elif arm is Arm.LIGHTGBM:
        cell = _run_lightgbm_cell(dataset, args.seed, split, constants)
    elif arm is Arm.PLAIN_NN:
        cell = _run_plain_nn_cell(dataset, args.seed, split, constants, epoch_cap=args.epoch_cap)
    else:
        cell = _run_linear_reg_cell(dataset, args.seed, split, constants)

    os.makedirs(args.results_dir, exist_ok=True)
    width_suffix = f"_{args.width}" if arm is Arm.W_SWEEP else ""
    tag_suffix = f"_{args.tag}" if args.tag else ""
    out_path = os.path.join(args.results_dir, f"{dataset.value}_{args.seed}_{arm.value}{width_suffix}{tag_suffix}.json")
    with open(out_path, "w") as f:
        json.dump(_jsonable(cell), f, indent=2)

    trustworthy = cell.get("trustworthy", cell.get("w_shared", {}).get("held_out_mse") is not None and cell["shared_training"]["trustworthy"] if arm is Arm.DIAL else True)
    if not trustworthy:
        print(f"*** DO-NOT-CONCLUDE GUARD: dataset={dataset.value} seed={args.seed} arm={arm.value} did NOT converge trustworthily. ***")
    print(f"[width_wsel9] wrote {out_path}")


if __name__ == "__main__":
    main()
