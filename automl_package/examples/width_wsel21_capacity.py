"""WSEL-21 -- the d>=8 CAPACITY+DATA diagnostic (re-scope of `docs/flexnn/width/plan.md` WSEL-21, 2026-07-24 user review).

**What this reopens.** The WSEL-21 optimizer-knob escalation ladder (`docs/flexnn/width/specs/wsel21-escalation.md`,
driver `width_wsel19.py`'s `EscalationRung`/`run_escalation_rung`) recorded d>=8 UNREACHABLE under
FOUR protocol rungs (full-batch -> mini-batch {128, 32}; constant LR -> cosine decay; patience
60 -> 180) on the designated calibration cell (d=8, geometry=AXIS, seed=2, n_train=1500): every rung
trained cleanly (`all_widths_trustworthy=True` throughout) but fit quality never moved off the
17.8-19.0x noise-floor band, against a 1.128x graduation bar.
<!-- source: automl_package/examples/capacity_ladder_results/WSEL19/wsel21_escalation_ledger.json -->
The user's 2026-07-24 re-scope narrowed that verdict to the OPTIMIZER-KNOB family only -- it is NOT
a capacity/architecture verdict -- and reopened exactly this question: "can we recover the width when
the signal is buried among many irrelevant features?" (`docs/flexnn/width/plan.md`, WSEL-21 block, `flip_attempt`).
The re-scope names the smallest change that would flip the verdict: sweep per-width net CAPACITY and
training-set SIZE, never more optimizer knobs.

**Why every net in this cell's history is a min-over-12-widths quantity, verified by direct read (not
assumed).** `width_wsel19.py`'s own per-cell fit-gate arithmetic (`run_cell_multifeature`, and
identically `run_escalation_rung`) is:
```
best_fixed_held_out_mse = float(error_table_report.mean(axis=0).min())   # width_wsel19.py:1424
ratio_to_noise_floor = best_fixed_held_out_mse / noise_floor              # width_wsel19.py:1425
```
`.min(axis=0)` is a minimum over the TWELVE dedicated per-width nets' held-out MSE columns -- every
`ratio_to_noise_floor` this strand has ever recorded at this cell, including the escalation ladder's
17.8-19.0x band and the `frozen_mf.json` baseline of 17.835, is a best-of-twelve, never a single net's
own number. **This driver's first cut (superseded by this version) trained ONE net per cell and
compared it directly to that minimum -- an apples-to-oranges comparison that a 3-seed root-run
calibration slice caught: single-net BASELINE landed at 30.711 +/- 2.381 (seed 0/1/2: 28.270 /
28.391 / 35.473), a 1.7x gap from 17.835.** The fix, not a tuning knob: every cell now trains the
FULL 12-width sweep (dial values k=1..12, exactly as `width_wsel19.py` does) and reports the SAME
min-over-12-widths quantity, computed the identical way, so it is directly comparable to the
escalation ladder's four rungs and to `frozen_mf.json`'s own per-seed baselines.

**How capacity is decoupled from the width-dial value without a package change (unchanged from the
first cut, verified by direct read).** Every net the optimizer ladder (and the underlying WSEL-19
multi-feature sweep) ever trained at this cell is a `FlexibleWidthNN(widths=(k,))` for width-dial
value k in 1..12 -- and that architecture couples "how much capacity does this net have" to "what
width-dial value is it" EXACTLY: `FlexibleWidthNNModule` builds ONE shared
`trunk_linear = nn.Linear(input_size, w_max)` (`w_max = max(widths)`) followed by one per-width
`Linear(w_max, output_size)` head, so a width-k dedicated net is, by construction, ONE hidden layer of
EXACTLY k units -- never more, never deeper.
<!-- source: automl_package/models/flexnn/width/model.py:104-130 (FlexibleWidthNN.__init__, FlexibleWidthNNModule.__init__) -->
This driver's 12-width sweep therefore does NOT reuse `FlexibleWidthNN` -- it uses
`automl_package.models.neural_network.PyTorchNeuralNetwork` instead (an EXISTING package model with
`hidden_size`/`hidden_layers` as first-class, independent constructor kwargs, built on the SAME
`PyTorchModelBase._fit_single` bypass `width_wsel19.py`'s own `_mf_train_one_width` already uses).
Per net at width-dial value k, `hidden_size = capacity_level's width_multiplier * k`, so `BASELINE`
(multiplier 1) reproduces `FlexibleWidthNN`'s own per-width hidden size EXACTLY (k units, 1 layer),
while `WIDE_2X` trains a 2k-unit net at the SAME dial position k, etc. -- capacity is now a genuine
axis independent of which dial value is being evaluated.

**Acceptance gate (added after the calibration-slice finding above; see `_baseline_anchor_check`).**
`BASELINE` at `n_train=1500` is required to reproduce `width_wsel19.py`'s own full-batch
`FlexibleWidthNN` sweep's per-seed `ratio_to_noise_floor` (read from `frozen_mf.json`'s
`triples.d8_axis_seed{0,1,2}`, NEVER retyped) within `_BASELINE_ACCEPTANCE_RELATIVE_TOLERANCE`. If it
does not, the model-class swap (`PyTorchNeuralNetwork` vs `FlexibleWidthNN`) is itself a confound and
NO capacity-level cell in this ledger is interpretable -- `--summarize` surfaces this as an explicit
`baseline_anchor_check` field and a printed warning, never silently reports capacity numbers against a
broken baseline.

**Training-set-size levels.** `wt.TRAIN_N` (1500, the grid's own canonical provisioning) and
`width_wsel19._MF_N_TRAIN_FALLBACK` (4000, the pre-authorized §3b F6 data lever) -- both ALREADY
sanctioned constants elsewhere in this strand. The F6 probe pulled a sibling d=8 cell
(`d8_oblique_seed2`) from the 17.8x class to 2.553 across exactly this jump, in one step.
<!-- source: automl_package/examples/capacity_ladder_results/WSEL19/frozen_mf.json triples.d8_oblique_seed2 -->

**Capacity levels (an enum, not magic ints -- see `CapacityLevel` below).** `BASELINE` = width
multiplier 1, depth 1 -- reproduces `FlexibleWidthNN`'s own per-width net exactly (see acceptance gate
above). Every other level changes exactly one of {width multiplier, depth} from `BASELINE` (this
strand's own single-difference discipline, `wsel21-escalation.md` §4's rung table), except the last,
which combines both single-differences -- mirroring the router sweep's own "half/double width, 1 vs 3
layers" combination (`docs/flexnn/width/plan.md:1178-1182`).

**Cache collision guard (escalation spec §4 hazard 2, re-applied across model CLASSES here, not just
protocols).** This driver's cached net weights use a DEDICATED subdirectory (`_capacity_cache/`,
distinct from `width_wsel19.py`'s own `_mf_cache/`) and a tag prefix (`capdiag_`) that
`_capacity_cache_tag` refuses to build if it would ever start with
`width_wsel19._MF_CACHE_CONSTRUCTION_TAG` ("box") -- enforced in code, because a silently mismatched
`state_dict` load (a `PyTorchNeuralNetwork` checkpoint loaded into, or overwriting, a `FlexibleWidthNN`
checkpoint slot) would fail loudly at best and corrupt silently at worst. Each of the 12 per-width nets
within a cell gets its OWN cache-path (tag includes the width dial value).

**Cost (root-run, backgrounded).** 12 nets/cell x 30 cells (5 capacity levels x 2 n_train levels x 3
seeds) = 360 nets. At the ~13s/net the calibration slice measured at this cell's real scale, that is
roughly 80 minutes total.

**Non-goals (authoring contract; this file is never run over the full grid by its author).** No new
optimizer knobs (batch size, LR schedule, patience are held at the PORTED-arm default throughout).
No edits to `width_wsel19.py`/`width_wsel19_toys.py`/`models/flexnn/`/any plan doc. No new toy
construction. No real grid beyond `--selftest`'s tiny synthetic-data wiring checks and exactly ONE
real-scale smoke cell (the root runs the grid, backgrounded).

Driver CLI contract (root-run grid; one cell per invocation, UNCHANGED from the first cut):
    `--capacity-level {baseline,wide_2x,wide_4x,deep_2l,wide_2x_deep_2l} --n-train {1500,4000}
    --seed {0,1,2}` runs ONE cell at the designated (d=8, geometry=AXIS) calibration cell -- now a
    12-width sweep internally -- writing its per-cell JSON immediately.
    `--summarize` aggregates every per-cell JSON on disk into `WSEL19/wsel21_capacity_ledger.json`,
    including the `baseline_anchor_check` acceptance-gate surfacing and a read-only reference to the
    optimizer ladder's own ledger -- never recomputing either source.
    `--selftest` runs `width_wsel19.run_selftest()` in-process, the cache-tag collision guard, a
    ledger round-trip, the real on-disk calibration artifact's anchor read (top-level, not the regime
    block's), tiny synthetic-data wiring checks (min-over-widths formula correctness, depth-2 wiring,
    cache reuse), and exactly ONE real-scale cell (BASELINE, d=8 AXIS, n_train=1500, seed=2, the full
    12-width sweep) used to exercise the acceptance gate against the real `frozen_mf.json` reference.

Usage:
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python -m automl_package.examples.width_wsel21_capacity --selftest
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python -m automl_package.examples.width_wsel21_capacity \
        --capacity-level baseline --n-train 1500 --seed 0
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python -m automl_package.examples.width_wsel21_capacity \
        --capacity-level wide_2x_deep_2l --n-train 4000 --seed 2
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m automl_package.examples.width_wsel21_capacity --summarize
"""

from __future__ import annotations

import argparse
import enum
import json
import os
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
import torch

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root

import nested_width_net as nwn  # noqa: E402 -- HETERO_NOISE_SIGMA, the canonical noise floor.
import width_wsel4 as w4  # noqa: E402 -- PORTED_* protocol constants and `_replay`, reused verbatim (module docstring).
import width_wsel19 as w19  # noqa: E402 -- the calibration gate, W_MAX/SEEDS, cache-tag/mean_se/jsonable helpers, and `run_selftest` (see module docstring).
import width_wsel19_toys as wt  # noqa: E402 -- the d=8 AXIS toy construction (make_hetero_multifeature/make_report_split), Geometry, TRAIN_N.

from automl_package.enums import ActivationFunction, TaskType  # noqa: E402
from automl_package.models.neural_network import PyTorchNeuralNetwork  # noqa: E402
from automl_package.utils.run_provenance import run_provenance  # noqa: E402

RESULTS_DIR = w19.RESULTS_DIR  # capacity_ladder_results/WSEL19 -- the same results dir every WSEL-19/21 artifact lives in.

_DIAGNOSTIC_D = 8  # the WSEL-21 re-scope's designated dimension.
_DIAGNOSTIC_GEOMETRY = wt.Geometry.AXIS  # H = I_d exactly at AXIS -- same calibration-cell choice as `wsel21-escalation.md` §3 (no rotation arithmetic confound).

_N_TRAIN_LEVELS: tuple[int, ...] = (wt.TRAIN_N, w19._MF_N_TRAIN_FALLBACK)  # (1500, 4000) -- both already-sanctioned values in this strand (module docstring), not new constants.

_CAPACITY_CACHE_DIRNAME = "_capacity_cache"  # dedicated cache subdir, disjoint from width_wsel19.py's own `_mf_cache/` (different model CLASS -- never safe to share a namespace).
_CAPACITY_TAG_PREFIX = "capdiag"  # guaranteed distinct from `width_wsel19._MF_CACHE_CONSTRUCTION_TAG` ("box") -- enforced in `_capacity_cache_tag`, not left as a convention.
_CAPACITY_LEDGER_FILENAME = "wsel21_capacity_ledger.json"
_FROZEN_MF_FILENAME = "frozen_mf.json"  # width_wsel19.py's own multi-feature aggregate -- the acceptance gate's reference source (never a magic literal float in this file).

# Judgment call, fixed BEFORE any real BASELINE cell was scored under this restructuring: roughly 3x
# the escalation ladder's own same-seed protocol-induced spread at this exact cell (17.835 vs its
# worst rung 18.99, ~6.5%) -- enough headroom for a genuine model-CLASS swap while still catching a
# real confound (module docstring).
_BASELINE_ACCEPTANCE_RELATIVE_TOLERANCE = 0.15


class CapacityLevel(enum.Enum):
    """Closed set of `(width_multiplier, hidden_layers)` capacity levels for the d>=8 AXIS diagnostic (WSEL-21 re-scope, module docstring).

    Per net at width-dial value k (1..`width_wsel19.W_MAX`), `hidden_size = width_multiplier * k`.
    `BASELINE` (multiplier 1, depth 1) reproduces `FlexibleWidthNN`'s own per-width net exactly (a
    width-k dedicated net there is, by construction, k hidden units in 1 layer -- see module
    docstring) -- this is the acceptance gate's own reference point. Every other level changes
    exactly one of {width multiplier, depth} from `BASELINE`, except `WIDE_2X_DEEP_2L`, which
    combines both single-differences. See `_CAPACITY_ARCH` for the concrete mapping.
    """

    BASELINE = "baseline"  # multiplier=1, depth=1 -- k units, 1 layer (== FlexibleWidthNN's own per-width net).
    WIDE_2X = "wide_2x"  # multiplier=2, depth=1 -- 2k units, 1 layer.
    WIDE_4X = "wide_4x"  # multiplier=4, depth=1 -- 4k units, 1 layer.
    DEEP_2L = "deep_2l"  # multiplier=1, depth=2 -- k units, 2 layers.
    WIDE_2X_DEEP_2L = "wide_2x_deep_2l"  # multiplier=2, depth=2 -- 2k units, 2 layers.


_CAPACITY_ARCH: dict[CapacityLevel, tuple[int, int]] = {
    CapacityLevel.BASELINE: (1, 1),
    CapacityLevel.WIDE_2X: (2, 1),
    CapacityLevel.WIDE_4X: (4, 1),
    CapacityLevel.DEEP_2L: (1, 2),
    CapacityLevel.WIDE_2X_DEEP_2L: (2, 2),
}

_SELFTEST_MIN_LEDGER_CELLS = 1  # the acceptance-gate smoke run lands exactly one real cell -- --summarize must see it.

_SELFTEST_REQUIRED_KEYS = (
    "capacity_level",
    "width_multiplier",
    "hidden_layers",
    "n_train",
    "seed",
    "d",
    "geometry",
    "w_max",
    "ratio_to_noise_floor",
    "best_width",
    "per_width_ratio_to_noise_floor",
    "anchor_ratio_to_noise_floor",
    "all_widths_trustworthy",
    "hit_cap",
    "cache_hit",
    "graduated",
    "provenance",
)


def _capacity_cache_tag(d: int, geometry: wt.Geometry, seed: int, n_train: int, level: CapacityLevel, width: int, *, prefix: str = _CAPACITY_TAG_PREFIX) -> str:
    """Builds this diagnostic's own cache tag for ONE per-width net, refusing (raising) if it would collide with `width_wsel19`'s per-width cache namespace.

    `prefix` is exposed only so `run_selftest` can exercise the guard by deliberately passing
    `width_wsel19._MF_CACHE_CONSTRUCTION_TAG` ("box") and observing the raise -- no real caller ever
    overrides it.

    Args:
        d: input dimension (the designated cell's own 8, or a selftest override).
        geometry: `wt.Geometry.AXIS` or `wt.Geometry.OBLIQUE`.
        seed: RNG seed.
        n_train: training-set size for this cell.
        level: the capacity level this cell trains at.
        width: the width-dial value (1..w_max) this specific net within the cell's sweep is trained at.
        prefix: the tag prefix; defaults to `_CAPACITY_TAG_PREFIX`.

    Returns:
        The tag string (no extension) this net's cache files are named from.

    Raises:
        RuntimeError: if `prefix` would produce a tag colliding with `width_wsel19`'s own "box"-tagged
            per-width cache namespace.
    """
    if prefix.startswith(w19._MF_CACHE_CONSTRUCTION_TAG):
        raise RuntimeError(
            f"capacity cache tag prefix {prefix!r} collides with width_wsel19's own {w19._MF_CACHE_CONSTRUCTION_TAG!r} "
            "construction tag -- refusing (a PyTorchNeuralNetwork checkpoint must never be loaded into, or overwrite, a "
            "FlexibleWidthNN checkpoint slot)."
        )
    return f"{prefix}_d{d}_{geometry.value}_seed{seed}_ntrain{n_train}_{level.value}_w{width}"


def _capacity_model_cache_paths(results_dir: str, d: int, geometry: wt.Geometry, seed: int, n_train: int, level: CapacityLevel, width: int) -> tuple[str, str]:
    """Cache paths for ONE per-width net within a cell's 12-width sweep, under this driver's own dedicated `_capacity_cache/` subdir."""
    base = os.path.join(results_dir, _CAPACITY_CACHE_DIRNAME)
    os.makedirs(base, exist_ok=True)
    tag = _capacity_cache_tag(d, geometry, seed, n_train, level, width)
    return os.path.join(base, f"{tag}.pt"), os.path.join(base, f"{tag}_meta.json")


def _new_capacity_model(in_dim: int, hidden_size: int, hidden_layers: int, seed: int) -> PyTorchNeuralNetwork:
    """Builds one unfitted `PyTorchNeuralNetwork` at an explicit `(hidden_size, hidden_layers)`, the PORTED-arm protocol otherwise.

    Tanh activation, full-batch, lr=0.01, patience=60, max_epochs=6000 -- byte-identical to every
    other dedicated net this strand trains (`width_wsel4.PORTED_*`), varying only capacity.
    """
    return PyTorchNeuralNetwork(
        input_size=in_dim,
        output_size=1,
        task_type=TaskType.REGRESSION,
        hidden_size=hidden_size,
        hidden_layers=hidden_layers,
        activation=ActivationFunction.TANH,
        learning_rate=w4.PORTED_LR_DEFAULT,
        n_epochs=w4.PORTED_N_EPOCHS_CAP,
        early_stopping_rounds=w4.PORTED_PATIENCE,
        batch_size=w4.PORTED_BATCH_SIZE,
        random_seed=seed,
        calculate_feature_importance=False,
    )


def _load_cached_capacity_model(in_dim: int, hidden_size: int, hidden_layers: int, seed: int, state_path: str) -> PyTorchNeuralNetwork:
    """Rebuilds a capacity-diagnostic net's module and loads a previously-cached `state_dict` back into it."""
    model = _new_capacity_model(in_dim, hidden_size, hidden_layers, seed)
    model.build_model()
    model.model.load_state_dict(torch.load(state_path, map_location=model.device))
    model.model.eval()
    return model


def _get_or_train_one_width_model(
    in_dim: int, hidden_size: int, hidden_layers: int, seed: int, x_tr: np.ndarray, y_tr: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, state_path: str, meta_path: str
) -> tuple[PyTorchNeuralNetwork, dict[str, Any], bool]:
    """Get-or-train ONE per-width net within a cell's 12-width sweep, cached at `state_path`/`meta_path` (mirrors `width_wsel19._mf_get_or_train_one_width`)."""
    if os.path.exists(state_path) and os.path.exists(meta_path):
        model = _load_cached_capacity_model(in_dim, hidden_size, hidden_layers, seed, state_path)
        with open(meta_path) as f:
            meta = json.load(f)
        return model, meta, True

    model = _new_capacity_model(in_dim, hidden_size, hidden_layers, seed)
    _best_epoch, val_loss_history = model._fit_single(x_tr, y_tr, x_val=x_val, y_val=y_val)
    replay = w4._replay(val_loss_history, w4.PORTED_PATIENCE, w4.PORTED_MIN_DELTA)
    hit_cap = bool(len(val_loss_history) >= w4.PORTED_N_EPOCHS_CAP)
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
        json.dump(w19._jsonable(meta), f)
    return model, meta, False


def run_cell(
    level: CapacityLevel,
    n_train: int,
    seed: int,
    *,
    results_dir: str = RESULTS_DIR,
    d: int = _DIAGNOSTIC_D,
    geometry: wt.Geometry = _DIAGNOSTIC_GEOMETRY,
    w_max: int = w19.W_MAX,
) -> dict[str, Any]:
    """Runs one (capacity_level, n_train, seed) cell at the designated d=8 AXIS calibration cell (or a selftest override of `d`/`geometry`/`w_max`).

    Trains the FULL `w_max`-width sweep (dial values 1..`w_max`), one dedicated
    `PyTorchNeuralNetwork(hidden_size=level's width_multiplier * dial_value, hidden_layers=level's
    depth)` per dial value -- exactly mirroring `width_wsel19.py`'s own per-width sweep shape, so
    `ratio_to_noise_floor` (the MINIMUM held-out-MSE/noise_floor across the sweep, computed the
    identical way as `width_wsel19.py:1424-1425`) is directly comparable to the escalation ladder's
    four rungs and to `frozen_mf.json`'s own per-seed baselines (module docstring).

    Reads the SAME pinned anchor `width_wsel19`'s own §5.3 fit gate and the optimizer ladder both use
    (top-level `anchor_ratio_to_noise_floor` of `wsel19_calibration_d1.json`, via
    `width_wsel19._load_calibration_or_refuse` -- never re-read or recomputed), and scores every net
    on the SAME disjoint report split `width_wsel19_toys.make_report_split` produces for every other
    cell in this strand.

    Args:
        level: the capacity level (`CapacityLevel`) this cell trains at.
        n_train: training-set size (one of `_N_TRAIN_LEVELS` for a real cell; a selftest may pass
            any positive int).
        seed: RNG seed (one of `width_wsel19.SEEDS` for a real cell).
        results_dir: where the calibration gate is read from and this cell's cache is written.
        d: input dimension; defaults to the designated 8, overridable only for `--selftest`'s tiny toy.
        geometry: defaults to the designated AXIS, overridable only for `--selftest`'s tiny toy.
        w_max: largest width-dial value swept; defaults to `width_wsel19.W_MAX` (12), overridable only
            for `--selftest`'s tiny toy.

    Returns:
        The JSON-able per-cell case dict.
    """
    calibration = w19._load_calibration_or_refuse(results_dir)  # §5.2 gate -- REFUSES (raises) if missing/failed/stale, same as every other WSEL-19/21 cell.
    anchor = calibration["anchor_ratio_to_noise_floor"]

    x_tr_full, y_tr_full, _region_tr, _t_tr = wt.make_hetero_multifeature(n_train, seed, d, geometry)
    val_mask = (np.arange(n_train) % w19._INTERNAL_VAL_EVERY) == 0
    x_tr, y_tr = x_tr_full[~val_mask], y_tr_full[~val_mask]
    x_val, y_val = x_tr_full[val_mask], y_tr_full[val_mask]
    x_report, y_report, _region_report, _t_report = wt.make_report_split(seed, d, geometry)

    width_multiplier, hidden_layers = _CAPACITY_ARCH[level]
    noise_floor = float(nwn.HETERO_NOISE_SIGMA**2)

    per_width_ratio: list[float] = []
    per_width_trustworthy: list[bool] = []
    per_width_hit_cap: list[bool] = []
    all_cache_hit = True
    for width in range(1, w_max + 1):
        hidden_size = width_multiplier * width
        state_path, meta_path = _capacity_model_cache_paths(results_dir, d, geometry, seed, n_train, level, width)
        model, meta, cache_hit = _get_or_train_one_width_model(d, hidden_size, hidden_layers, seed, x_tr, y_tr, x_val, y_val, state_path, meta_path)
        all_cache_hit = all_cache_hit and cache_hit

        y_pred = model.predict(x_report, filter_data=False)
        held_out_mse = float(np.mean((y_pred - y_report) ** 2))
        per_width_ratio.append(held_out_mse / noise_floor)
        per_width_trustworthy.append(meta["trustworthy"])
        per_width_hit_cap.append(meta["hit_cap"])

    ratio_to_noise_floor = float(min(per_width_ratio))
    best_width = int(np.argmin(per_width_ratio)) + 1
    all_widths_trustworthy = bool(all(per_width_trustworthy))
    graduated = bool(all_widths_trustworthy and ratio_to_noise_floor <= anchor)

    return {
        "capacity_level": level.value,
        "width_multiplier": width_multiplier,
        "hidden_layers": hidden_layers,
        "n_train": n_train,
        "seed": seed,
        "d": d,
        "geometry": geometry.value,
        "w_max": w_max,
        "ratio_to_noise_floor": ratio_to_noise_floor,
        "best_width": best_width,
        "per_width_ratio_to_noise_floor": per_width_ratio,
        "anchor_ratio_to_noise_floor": anchor,
        "all_widths_trustworthy": all_widths_trustworthy,
        "hit_cap": bool(any(per_width_hit_cap)),
        "cache_hit": bool(all_cache_hit),
        "graduated": graduated,
        "provenance": run_provenance(),
    }


def _cell_json_path(results_dir: str, level: CapacityLevel, n_train: int, seed: int, *, d: int = _DIAGNOSTIC_D, geometry: wt.Geometry = _DIAGNOSTIC_GEOMETRY) -> str:
    return os.path.join(results_dir, f"wsel21_capacity_d{d}_{geometry.value}_{level.value}_ntrain{n_train}_seed{seed}.json")


# ---------------------------------------------------------------------------
# --summarize -- aggregates every per-cell JSON on disk into WSEL19/wsel21_capacity_ledger.json,
# surfaces the BASELINE acceptance gate (module docstring), and includes a read-only (never
# recomputed) reference to the optimizer ladder's own ledger, for a direct side-by-side against its
# four rungs.
# ---------------------------------------------------------------------------


def _read_full_batch_reference_by_seed(results_dir: str, d: int, geometry: wt.Geometry, n_train: int) -> dict[int, float]:
    """Reads `width_wsel19.py`'s OWN full-batch `FlexibleWidthNN` sweep's per-seed `ratio_to_noise_floor` (min-over-widths) from `frozen_mf.json`, never recomputed or retyped.

    This is the acceptance gate's apples-to-apples reference: identical `(d, geometry, seed, n_train)`
    cell, identical width-dial architecture shape (per-width dedicated nets, hidden_size == width, 1
    hidden layer -- `BASELINE`'s own definition), the SAME min-over-widths quantity -- only the model
    CLASS differs (`PyTorchNeuralNetwork` here, `FlexibleWidthNN` there).

    Returns:
        `{seed: ratio_to_noise_floor}` for every canonical seed whose `frozen_mf.json` triple exists,
        was recorded at exactly `n_train` (never a fallback triple), and carries the field. Empty if
        `frozen_mf.json` is absent.
    """
    path = os.path.join(results_dir, _FROZEN_MF_FILENAME)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        frozen_mf = json.load(f)
    out: dict[int, float] = {}
    for seed in w19.SEEDS:
        triple = frozen_mf.get("triples", {}).get(f"d{d}_{geometry.value}_seed{seed}")
        if triple is not None and triple.get("n_train") == n_train and "ratio_to_noise_floor" in triple:
            out[seed] = triple["ratio_to_noise_floor"]
    return out


def _baseline_anchor_check(cells: list[dict[str, Any]], results_dir: str) -> dict[str, Any] | None:
    """The acceptance gate (module docstring): landed BASELINE/n_train=1500 cells vs `width_wsel19`'s own matched-seed full-batch reference.

    Returns `None` if no BASELINE/n_train=1500 cell has landed yet (nothing to check) or
    `frozen_mf.json` is absent. Otherwise returns a per-seed comparison plus an overall verdict --
    `all_within_tolerance=False` means every OTHER capacity-level cell in this ledger is NOT
    interpretable against the recorded 17.8-19.0x wall: the model-class swap itself would be the
    confound, not capacity or training-set size.
    """
    reference = _read_full_batch_reference_by_seed(results_dir, _DIAGNOSTIC_D, _DIAGNOSTIC_GEOMETRY, wt.TRAIN_N)
    baseline_cells = {c["seed"]: c for c in cells if c["capacity_level"] == CapacityLevel.BASELINE.value and c["n_train"] == wt.TRAIN_N}
    if not baseline_cells or not reference:
        return None

    per_seed: dict[str, Any] = {}
    for seed, cell in baseline_cells.items():
        ref = reference.get(seed)
        if ref is None:
            continue
        observed = cell["ratio_to_noise_floor"]
        relative_delta = abs(observed - ref) / ref
        within = relative_delta <= _BASELINE_ACCEPTANCE_RELATIVE_TOLERANCE
        per_seed[str(seed)] = {"observed": observed, "reference": ref, "relative_delta": relative_delta, "within_tolerance": within}

    all_within_tolerance = bool(per_seed) and all(v["within_tolerance"] for v in per_seed.values())
    return {
        "reference_source": _FROZEN_MF_FILENAME,
        "tolerance": _BASELINE_ACCEPTANCE_RELATIVE_TOLERANCE,
        "per_seed": per_seed,
        "n_seeds_compared": len(per_seed),
        "all_within_tolerance": all_within_tolerance,
        "verdict": (
            "OK -- capacity cells are interpretable against the recorded wall"
            if all_within_tolerance
            else "MISMATCH -- the model-class swap is confounding this ledger; capacity cells are NOT interpretable until resolved"
        ),
    }


def summarize(results_dir: str = RESULTS_DIR) -> None:
    """Aggregates every per-cell JSON on disk into `WSEL19/wsel21_capacity_ledger.json`. Does not train or fit anything."""
    cells: list[dict[str, Any]] = []
    for level in CapacityLevel:
        for n_train in _N_TRAIN_LEVELS:
            for seed in w19.SEEDS:
                path = _cell_json_path(results_dir, level, n_train, seed)
                if not os.path.exists(path):
                    continue
                with open(path) as f:
                    cells.append(json.load(f))

    anchor = None
    calibration_path = w19._calibration_path(results_dir)
    if os.path.exists(calibration_path):
        with open(calibration_path) as f:
            anchor = json.load(f)["anchor_ratio_to_noise_floor"]

    escalation_reference = None
    escalation_ledger_path = w19._escalation_ledger_path(results_dir)
    if os.path.exists(escalation_ledger_path):
        with open(escalation_ledger_path) as f:
            esc = json.load(f)
        escalation_reference = {
            "verdict": esc.get("verdict"),
            "graduated_rung": esc.get("graduated_rung"),
            "rung_ratios": {r["name"]: r["ratio_to_noise_floor"] for r in esc.get("rungs", [])},
        }

    baseline_anchor_check = _baseline_anchor_check(cells, results_dir)

    per_group: dict[str, dict[str, Any]] = {}
    for level in CapacityLevel:
        for n_train in _N_TRAIN_LEVELS:
            ratios = [c["ratio_to_noise_floor"] for c in cells if c["capacity_level"] == level.value and c["n_train"] == n_train]
            if not ratios:
                continue
            per_group[f"{level.value}:n_train{n_train}"] = w19._mean_se(ratios)

    n_expected = len(CapacityLevel) * len(_N_TRAIN_LEVELS) * len(w19.SEEDS)
    out = {
        "calibration_cell": {"d": _DIAGNOSTIC_D, "geometry": _DIAGNOSTIC_GEOMETRY.value},
        "capacity_levels": {level.value: {"width_multiplier": _CAPACITY_ARCH[level][0], "hidden_layers": _CAPACITY_ARCH[level][1]} for level in CapacityLevel},
        "n_train_levels": list(_N_TRAIN_LEVELS),
        "seeds": list(w19.SEEDS),
        "anchor_ratio_to_noise_floor": anchor,
        "anchor_source": calibration_path,
        "baseline_anchor_check": baseline_anchor_check,
        "escalation_ladder_reference": escalation_reference,
        "escalation_ladder_source": escalation_ledger_path,
        "cells": cells,
        "per_group": per_group,
        "n_cells_present": len(cells),
        "n_cells_expected": n_expected,
        "provenance": run_provenance(),
    }
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, _CAPACITY_LEDGER_FILENAME)
    with open(path, "w") as f:
        json.dump(w19._jsonable(out), f, indent=2)
    print(f"wrote {path}")
    print(f"n_cells_present={len(cells)}/{n_expected}  n_groups={len(per_group)}/{len(CapacityLevel) * len(_N_TRAIN_LEVELS)}")
    if baseline_anchor_check is None:
        print("baseline_anchor_check: no BASELINE/n_train=1500 cell landed yet (or frozen_mf.json missing) -- acceptance gate not yet evaluable.")
    elif baseline_anchor_check["all_within_tolerance"]:
        print(f"baseline_anchor_check: PASS ({baseline_anchor_check['n_seeds_compared']} seed(s) compared, tol={_BASELINE_ACCEPTANCE_RELATIVE_TOLERANCE:.0%})")
    else:
        print(f"*** baseline_anchor_check: {baseline_anchor_check['verdict']} *** -- {baseline_anchor_check['per_seed']}")


# ---------------------------------------------------------------------------
# --selftest
# ---------------------------------------------------------------------------


def run_selftest() -> bool:
    """Wiring checks on tiny synthetic data, plus exactly ONE real-scale cell for the acceptance gate. See module docstring for what each check proves."""
    ok = True

    # (a) Default/unswept-path byte-identity: this driver only ever IMPORTS width_wsel19 (never
    # edits or monkeypatches it), so the strongest available check is that width_wsel19's OWN
    # selftest still passes when run from a process that has also imported this module.
    w19_ok = w19.run_selftest()
    print(f"[wsel21 capacity selftest] width_wsel19.run_selftest() unaffected by this driver's imports: {'PASS' if w19_ok else 'FAIL'}")
    ok = ok and w19_ok

    # (b) Cache-tag guard actually fires on a collision (not just structurally avoided).
    guard_fired = False
    try:
        _capacity_cache_tag(_DIAGNOSTIC_D, _DIAGNOSTIC_GEOMETRY, 0, wt.TRAIN_N, CapacityLevel.BASELINE, 1, prefix=w19._MF_CACHE_CONSTRUCTION_TAG)
    except RuntimeError:
        guard_fired = True
    print(f"[wsel21 capacity selftest] cache-tag guard fires on a 'box'-colliding prefix: {'PASS' if guard_fired else 'FAIL'}")
    ok = ok and guard_fired

    # Recorded capacity/n_train levels are tied to real numbers already on disk, not invented here.
    levels_ok = _CAPACITY_ARCH[CapacityLevel.BASELINE] == (1, 1) and _N_TRAIN_LEVELS == (wt.TRAIN_N, w19._MF_N_TRAIN_FALLBACK) == (1500, 4000)
    print(f"[wsel21 capacity selftest] BASELINE==multiplier1/1-layer and n_train levels==(1500,4000): {'PASS' if levels_ok else 'FAIL'}")
    ok = ok and levels_ok

    # (d) The anchor is read from the top-level field of the REAL on-disk calibration artifact --
    # never the regime block's (a different provisioning, the exact wrong-pin hazard the escalation
    # spec's own adversarial-read addendum named, `wsel21-escalation.md` §9 finding 1).
    real_calibration_path = w19._calibration_path(RESULTS_DIR)
    if os.path.exists(real_calibration_path):
        with open(real_calibration_path) as f:
            raw_calibration = json.load(f)
        anchor_from_gate = w19._load_calibration_or_refuse(RESULTS_DIR)["anchor_ratio_to_noise_floor"]
        anchor_ok = (
            anchor_from_gate == raw_calibration["anchor_ratio_to_noise_floor"]
            and anchor_from_gate == raw_calibration["anchor_block"]["anchor_ratio_to_noise_floor"]
            and anchor_from_gate != raw_calibration["regime_block"]["anchor_ratio_to_noise_floor"]
        )
    else:
        anchor_ok = False
    print(f"[wsel21 capacity selftest] anchor read from top-level (== anchor_block, != regime_block) of the real calibration artifact: {'PASS' if anchor_ok else 'FAIL'}")
    ok = ok and anchor_ok

    # --- Tiny synthetic-data wiring checks (isolated temp dir, small w_max/n_train/d -- NOT the
    # real cell; this is plumbing verification, not the acceptance gate). ---
    tiny_dir = tempfile.mkdtemp(prefix="wsel21_capacity_selftest_tiny_")
    try:
        tiny_seed, tiny_d, tiny_geometry, tiny_w_max, tiny_n_train = 0, 2, wt.Geometry.AXIS, 3, 30

        try:
            w19._load_calibration_or_refuse(tiny_dir)
            gate_refused = False
        except RuntimeError:
            gate_refused = True
        print(f"[wsel21 capacity selftest] gate refuses before a calibration artifact exists: {'PASS' if gate_refused else 'FAIL'}")
        ok = ok and gate_refused

        # A tiny, unpassing-by-construction calibration would legitimately refuse every cell (§5.2
        # gate) -- seed a synthetic PASSING v2 fixture, exactly width_wsel19.py's own selftest
        # recipe (reused, not reinvented).
        tiny_calibration = w19.run_calibration(w_max=3, seeds=(tiny_seed,), n_train_regime=40, n_train_anchor=60, results_dir=tiny_dir)
        with open(w19._calibration_path(tiny_dir), "w") as f:
            json.dump({**tiny_calibration, "passed": True}, f)

        case = run_cell(CapacityLevel.BASELINE, tiny_n_train, tiny_seed, results_dir=tiny_dir, d=tiny_d, geometry=tiny_geometry, w_max=tiny_w_max)
        keys_ok = all(k in case for k in _SELFTEST_REQUIRED_KEYS)
        roundtrip = json.loads(json.dumps(w19._jsonable(case)))
        roundtrip_ok = all(k in roundtrip for k in _SELFTEST_REQUIRED_KEYS)
        finite_ok = np.isfinite(case["ratio_to_noise_floor"]) and all(np.isfinite(r) for r in case["per_width_ratio_to_noise_floor"])
        shape_ok = len(case["per_width_ratio_to_noise_floor"]) == tiny_w_max and 1 <= case["best_width"] <= tiny_w_max
        # The hard, mechanical check that motivated this whole restructuring: ratio_to_noise_floor
        # MUST be the min over the 12 (here 3) per-width ratios, computed independently here --
        # never trusted from the driver's own arithmetic.
        min_over_widths_ok = case["ratio_to_noise_floor"] == min(case["per_width_ratio_to_noise_floor"])
        graduated_consistent = case["graduated"] == bool(case["all_widths_trustworthy"] and case["ratio_to_noise_floor"] <= case["anchor_ratio_to_noise_floor"])
        cell_ok = keys_ok and roundtrip_ok and finite_ok and shape_ok and min_over_widths_ok and graduated_consistent
        print(
            f"[wsel21 capacity selftest] tiny cell level=baseline w_max={tiny_w_max} ratio={case['ratio_to_noise_floor']:.4g} "
            f"best_width={case['best_width']} min-over-widths formula correct={min_over_widths_ok}  {'PASS' if cell_ok else 'FAIL'}"
        )
        ok = ok and cell_ok

        # Cache reuse: repeat the SAME tiny cell, confirm every per-width net cache-hits.
        state_path_w1, _meta_path_w1 = _capacity_model_cache_paths(tiny_dir, tiny_d, tiny_geometry, tiny_seed, tiny_n_train, CapacityLevel.BASELINE, 1)
        mtime_first = os.path.getmtime(state_path_w1)
        case_again = run_cell(CapacityLevel.BASELINE, tiny_n_train, tiny_seed, results_dir=tiny_dir, d=tiny_d, geometry=tiny_geometry, w_max=tiny_w_max)
        mtime_second = os.path.getmtime(state_path_w1)
        cache_reused = case_again["cache_hit"] is True and mtime_first == mtime_second
        print(f"[wsel21 capacity selftest] tiny cell's per-width caches built once, reused on a repeat call: {'PASS' if cache_reused else 'FAIL'}")
        ok = ok and cache_reused

        # A second capacity level (deeper) at the SAME tiny cell, to exercise hidden_layers=2's wiring.
        case_deep = run_cell(CapacityLevel.DEEP_2L, tiny_n_train, tiny_seed, results_dir=tiny_dir, d=tiny_d, geometry=tiny_geometry, w_max=tiny_w_max)
        expected_multiplier, expected_hidden_layers = _CAPACITY_ARCH[CapacityLevel.DEEP_2L]
        deep_arch_ok = case_deep["hidden_layers"] == expected_hidden_layers and case_deep["width_multiplier"] == expected_multiplier
        deep_ok = deep_arch_ok and np.isfinite(case_deep["ratio_to_noise_floor"])
        print(f"[wsel21 capacity selftest] tiny cell level=deep_2l hidden_layers=2 wiring: {'PASS' if deep_ok else 'FAIL'}")
        ok = ok and deep_ok
    finally:
        shutil.rmtree(tiny_dir, ignore_errors=True)

    # --- Exactly ONE real-scale cell: BASELINE, the designated d=8 AXIS calibration cell,
    # n_train=1500, seed=2 (the escalation ladder's own designated screening seed -- directly
    # comparable to the literal 17.835 figure). Isolated temp dir seeded with COPIES (never edits)
    # of the real on-disk calibration artifact and frozen_mf.json, so this run never touches the
    # shared results dir or its cache. This is the acceptance gate this whole restructuring exists
    # to check (module docstring). ---
    acceptance_dir = tempfile.mkdtemp(prefix="wsel21_capacity_selftest_acceptance_")
    try:
        real_frozen_mf_path = os.path.join(RESULTS_DIR, _FROZEN_MF_FILENAME)
        if os.path.exists(real_calibration_path) and os.path.exists(real_frozen_mf_path):
            shutil.copy(real_calibration_path, w19._calibration_path(acceptance_dir))
            shutil.copy(real_frozen_mf_path, os.path.join(acceptance_dir, _FROZEN_MF_FILENAME))

            acceptance_seed = w19.SEEDS[2]  # seed=2 -- the escalation ladder's own designated cell.
            acceptance_case = run_cell(CapacityLevel.BASELINE, wt.TRAIN_N, acceptance_seed, results_dir=acceptance_dir)
            with open(_cell_json_path(acceptance_dir, CapacityLevel.BASELINE, wt.TRAIN_N, acceptance_seed), "w") as f:
                json.dump(w19._jsonable(acceptance_case), f, indent=2)

            summarize(results_dir=acceptance_dir)
            with open(os.path.join(acceptance_dir, _CAPACITY_LEDGER_FILENAME)) as f:
                ledger = json.load(f)
            check = ledger.get("baseline_anchor_check")
            ledger_ok = ledger["n_cells_present"] >= _SELFTEST_MIN_LEDGER_CELLS and check is not None and str(acceptance_seed) in check["per_seed"]
            print(f"[wsel21 capacity selftest] real acceptance-gate cell summarize round-trip: n_cells_present={ledger.get('n_cells_present')}  {'PASS' if ledger_ok else 'FAIL'}")
            ok = ok and ledger_ok

            if check is not None and str(acceptance_seed) in check["per_seed"]:
                seed_check = check["per_seed"][str(acceptance_seed)]
                print(
                    f"[wsel21 capacity selftest] ACCEPTANCE GATE: BASELINE d={_DIAGNOSTIC_D} {_DIAGNOSTIC_GEOMETRY.value} seed={acceptance_seed} n_train={wt.TRAIN_N} "
                    f"observed={seed_check['observed']:.4g} vs width_wsel19's own full-batch reference={seed_check['reference']:.4g} "
                    f"(rel_delta={seed_check['relative_delta']:.1%}, tol={_BASELINE_ACCEPTANCE_RELATIVE_TOLERANCE:.0%})  "
                    f"{'PASS' if seed_check['within_tolerance'] else 'FAIL -- capacity cells NOT interpretable against this baseline'}"
                )
                ok = ok and seed_check["within_tolerance"]
            else:
                print("[wsel21 capacity selftest] ACCEPTANCE GATE: could not be evaluated (baseline_anchor_check missing the acceptance seed)  FAIL")
                ok = False
        else:
            print(
                "[wsel21 capacity selftest] ACCEPTANCE GATE: SKIPPED -- real calibration artifact or frozen_mf.json not found on disk "
                f"({real_calibration_path}, {real_frozen_mf_path})"
            )
    finally:
        shutil.rmtree(acceptance_dir, ignore_errors=True)

    print(f"[wsel21 capacity selftest] {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """Parses args and dispatches to `--selftest` / `--summarize`, or one real cell."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true", help="Wiring checks in temp dirs (plus width_wsel19.run_selftest() and one real acceptance-gate cell), then exit.")
    parser.add_argument("--summarize", action="store_true", help="Aggregate every per-cell JSON on disk into WSEL19/wsel21_capacity_ledger.json.")
    parser.add_argument("--capacity-level", choices=[level.value for level in CapacityLevel], default=None, help="Per-width net capacity level (width multiplier, hidden_layers).")
    parser.add_argument("--n-train", type=int, choices=list(_N_TRAIN_LEVELS), default=None, help="Training-set size for this cell.")
    parser.add_argument("--seed", type=int, choices=list(w19.SEEDS), default=None, help="RNG seed for this cell (canonical suite: 0, 1, 2).")
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR)
    args = parser.parse_args()

    if sum([args.selftest, args.summarize]) > 1:
        parser.error("--selftest and --summarize are mutually exclusive.")
    if args.selftest:
        sys.exit(0 if run_selftest() else 1)
    if args.summarize:
        summarize(results_dir=args.results_dir)
        return
    if args.capacity_level is None or args.n_train is None or args.seed is None:
        parser.error("--capacity-level, --n-train and --seed are all required for a real cell (or pass --selftest / --summarize).")

    level = CapacityLevel(args.capacity_level)
    os.makedirs(args.results_dir, exist_ok=True)
    print(f"[wsel21 capacity] level={level.value} n_train={args.n_train} seed={args.seed} d={_DIAGNOSTIC_D} geometry={_DIAGNOSTIC_GEOMETRY.value} (12-width sweep)", flush=True)
    case = run_cell(level, args.n_train, args.seed, results_dir=args.results_dir)
    cell_path = _cell_json_path(args.results_dir, level, args.n_train, args.seed)
    with open(cell_path, "w") as f:
        json.dump(w19._jsonable(case), f, indent=2)
    print(
        f"wrote {cell_path}  ratio_to_noise_floor={case['ratio_to_noise_floor']:.4g} (anchor={case['anchor_ratio_to_noise_floor']:.4g}, best_width={case['best_width']})  "
        f"all_widths_trustworthy={case['all_widths_trustworthy']}  graduated={case['graduated']}"
    )


if __name__ == "__main__":
    main()
