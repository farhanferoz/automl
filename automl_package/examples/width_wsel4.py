"""WSEL-4 — make W-SWEEP a usable reference (`docs/plans/capacity_programme/width.md` WSEL-4).

W-SWEEP (§1) is the expensive width-selection reference: a SEPARATE model dedicated to each width,
trained ORDINARILY at that one fixed width (no width-dropout, no multi-width schedule), scored held
out. `converged_width_experiment.py` already runs this protocol, but against the research module's
own `IndependentWidthNet`, not the package's `FlexibleWidthNN` -- so it cannot serve as a like-for-
like reference for W-SHARED/W-PERINPUT, which are both built on `FlexibleWidthNN`
(`automl_package/models/flexnn/width/model.py`, the certified `SharedTrunkPerWidthHeadNet` port).
This driver PORTS the sweep onto that class, one cell (toy, seed, width, arm) per invocation.

Two arms per cell, reusing existing code rather than rewriting it (WSEL-4 non-goal: "a port that
changes the protocol is not a port"):
  * `--arm control` -- `nested_width_net.IndependentWidthNet`, trained EXACTLY as
    `converged_width_experiment.py::_train_widths_to_convergence` trains it (same Gaussian NLL over
    a LEARNED per-width `logvar_head`, same Adam/LR, same convergence gate). Only ONE width's
    disjoint sub-net is actually trained per invocation -- see `_train_control`'s docstring for why
    this reproduces the original all-widths-in-one-process run bit-for-bit at 1/w_max the compute.
    This is MASTER Decision 14's known-good arm: it must reproduce `W_CONVERGED`'s stored per-width
    numbers before any ported number is read (`--summarize`'s `reproduction.json`).
  * `--arm ported` -- `FlexibleWidthNN` configured to `widths=(width,)`, a single-element tuple that
    makes the class's own sum-over-configured-widths training objective degenerate to ordinary
    single-width training (no code path is added for this -- it falls out of the existing class
    unmodified). Trained via the established package-model pattern (`moe_flexnn_comparison.py`'s
    `_fit_single` bypass of `.fit()`'s Optuna wrapper), UNLESS an escalation-ladder flag is set
    (MASTER Decision 16), in which case `_train_ported_escalated` takes over -- see its docstring.

CAVEAT surfaced per the dispatching brief ("match whatever the existing sweep's certified objective
was"): `W_CONVERGED`'s summary JSON never recorded a raw MSE -- only per-width Gaussian
log-likelihood (`per_k_nll`, itself computed under a LEARNED variance, not §3.7's fixed-sigma rule,
because this reproduction target predates that ruling and changing it would no longer be a
reproduction). `reproduction.json`'s `relative_error` therefore compares FRESHLY-COMPUTED per-width
NLL against that stored NLL (the only genuine, not-computed-by-this-driver anchor, per §3.6's anchor
warning) -- NOT MSE. `held_out_mse` is a new field this driver introduces, computed identically for
both arms (`sinc_width_experiment._score_all_widths_mse`, reused), so the ported-vs-control quality
comparison has a common yardstick even though no historical MSE anchor exists to gate it.

Per-cell CLI (one cell per invocation -- the root runs the grid, never this driver):
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel4.py \
        --toy hetero --seed 0 --width 5 --arm control
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel4.py \
        --toy hetero --seed 0 --width 5 --arm ported

Aggregate + reproduction check, after every cell has landed:
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/width_wsel4.py --summarize

Selftest / lint:
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel4.py --selftest
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
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root, so `import automl_package` works

import converged_width_experiment as cwe  # noqa: E402 — reused verbatim: LR, VAL_EVERY, _standardize_fit, _to_std_tensors, _width_nll
import convergence as cvg  # noqa: E402
import nested_width_net as nwn  # noqa: E402
import sinc_width_experiment as sw  # noqa: E402 — reused verbatim: _score_all_widths (NLL), _score_all_widths_mse, _jsonable

from automl_package.enums import ActivationFunction, CapacitySelection, TaskType  # noqa: E402
from automl_package.models.flexnn.width.model import FlexibleWidthNN  # noqa: E402 — imported at its HEAD location, not the WSEL-3-owned shim
from automl_package.utils.pytorch_utils import get_device  # noqa: E402

RESULTS_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "WSEL4")
W_CONVERGED_PATH = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "W_CONVERGED", "w_converged_summary.json")

_DRIVER_NAME = "automl_package/examples/width_wsel4.py"
REPRODUCTION_BAR = 0.02  # WSEL-4's chosen reproduction criterion: relative NLL error <= 2%, per (toy, seed, width)
_REL_ERR_EPS = 1e-12  # floors the relative-error denominator so a near-zero reference NLL cannot divide-by-zero

# Control arm: no protocol knobs beyond what converged_width_experiment.py already uses (its own
# argparse default; not re-exported as a module constant there, so cited here instead of re-derived).
CONTROL_MAX_EPOCHS_DEFAULT = 40000

# Ported arm: FlexibleWidthNN's already-vetted calibration on THIS EXACT toy (`nwn.make_hetero`,
# n_train=1500/n_test=500/w_max=12 -- the same scale `converged_width_experiment.py` uses), lifted
# from `moe_flexnn_comparison.py:161-166`'s `TOY_CONFIGS[ToyKey.WIDTH]` rather than re-derived —
# that file already established `FlexibleWidthNN` trains stably on raw (unstandardized) x/y at
# these values, so WSEL-4 reuses them as ITS protocol default rather than guessing new ones.
PORTED_LR_DEFAULT = 0.01
# Cap and batch regime RE-PINNED 2026-07-22 (root, Decision-16 escalation, one documented rung each):
# the inherited calibration (cap 1500, mini-batch 64) under-fit the TRAIN set at widths >= 6 —
# mini-batch gradient noise floored the fit at ~2.4x the control's level (a like-for-like confound:
# the control protocol is FULL-batch), and full batch needs a larger epoch budget (probe at cap 1500
# hit the cap mid-descent; cap 6000 converged to 1.18x control with train at the noise floor).
# Probe artifacts: scratchpad probe_fullbatch_cap{1500,6000}.json; superseded mini-batch cells
# retained under WSEL4/tanh_minibatch_run/.
PORTED_N_EPOCHS_CAP = 6000
PORTED_PATIENCE = 60
PORTED_MIN_DELTA = 1e-4
PORTED_BATCH_SIZE = 1_000_000_000  # >= any n_train => ONE batch per epoch (full-batch, the control's regime)

# Selftest-only: the escalation-ladder values exercised once to prove the knobs are wired (§ run_selftest).
_SELFTEST_GRAD_CLIP_NORM = 1.0
_SELFTEST_WARMUP_EPOCHS = 3

DEVICE = str(get_device())


class Arm(enum.Enum):
    """Closed set: which W-SWEEP contender a cell trains (§1's W-SWEEP row; WSEL-4's whole job)."""

    CONTROL = "control"  # nested_width_net.IndependentWidthNet, converged_width_experiment.py's own protocol, unmodified
    PORTED = "ported"  # FlexibleWidthNN(widths=(width,)) — the certified SharedTrunkPerWidthHeadNet architecture


class InitScheme(enum.Enum):
    """Closed set of escalation-ladder init overrides (MASTER Decision 16); DEFAULT = no escalation."""

    DEFAULT = "default"  # PyTorch's own nn.Linear init — the protocol default
    KAIMING = "kaiming"
    XAVIER = "xavier"


# Only `nwn.Toy.HETERO` has a landed W_CONVERGED reference to reproduce against; `HETERO3` is a real
# member of the reused enum but out of WSEL-4's scope until a hetero3 sweep+reference exists, so the
# CLI does not offer it (an honest restriction, not a rewrite of `nwn.Toy` — no changes to that file).
_TOY_BUILDERS: dict[nwn.Toy, Callable[[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
    nwn.Toy.HETERO: nwn.make_hetero,
}


# ---------------------------------------------------------------------------
# Shared data — ONE split function for both arms, so a quality difference between them reflects the
# architecture, not a data-exposure confound (MASTER Decision 15's single-difference rule).
# ---------------------------------------------------------------------------


def _build_split(toy: nwn.Toy, seed: int, n_train: int, n_test: int) -> dict[str, np.ndarray]:
    """Reproduces `converged_width_experiment.run_case`'s data + phase-1/val carve verbatim.

    Only phase-1 (`converged_width_experiment.py:96-105`) is used for training/validation; phase-2
    (that script's selector-fit half) is out of WSEL-4's scope (§1 -- W-SWEEP is a fixed-width
    reference, not the distilled router W-PERINPUT owns).
    """
    builder = _TOY_BUILDERS[toy]
    x_tr, y_tr, _region_tr = builder(n_train, seed)
    x_te, y_te, region_te = builder(n_test, seed + 500)
    p1_idx = np.arange(0, n_train, 2)
    x_p1, y_p1 = x_tr[p1_idx], y_tr[p1_idx]
    val_mask = (np.arange(len(x_p1)) % cwe.VAL_EVERY) == 0
    return {
        "x_train": x_p1[~val_mask],
        "y_train": y_p1[~val_mask],
        "x_val": x_p1[val_mask],
        "y_val": y_p1[val_mask],
        "x_test": x_te,
        "y_test": y_te,
        "region_test": region_te,
    }


# ---------------------------------------------------------------------------
# Control arm — reuses converged_width_experiment.py's own per-width primitives, retrained one
# width at a time (the driver contract's "one cell per invocation").
# ---------------------------------------------------------------------------


def _train_control(
    seed: int, w_max: int, width: int, split: dict[str, np.ndarray], max_epochs: int, check_every: int, patience: int, min_delta: float
) -> tuple[nwn.IndependentWidthNet, dict, cvg.ConvergenceResult]:
    """Trains ONLY width `width`'s disjoint sub-net, reusing `converged_width_experiment.py`'s own per-width training primitives verbatim.

    Reuses `_standardize_fit`/`_to_std_tensors`/`_width_nll`/`LR` unchanged. Constructs the FULL `w_max`-wide net (never a `w_max=width` net) before training, so `width`'s
    random init draws from the exact same point in the RNG stream as the historical all-widths run:
    `IndependentWidthNet.__init__` builds every subnet in one `nn.ModuleList` comprehension,
    consuming RNG in order at CONSTRUCTION time, before any training happens. Because subnets share
    no parameters (`_width_nll(net, k, ...)` reads only `net.subnets[k-1]`), training just ONE of
    them afterward reproduces the original per-width result bit-for-bit while spending 1/`w_max` of
    the compute — this is what makes a genuine "one cell per invocation" CLI possible without
    changing the training schedule itself (the WSEL-4 non-goal).
    """
    if not (1 <= width <= w_max):
        raise ValueError(f"width={width} out of range [1, {w_max}]")
    norm = cwe._standardize_fit(split["x_train"], split["y_train"])
    x_tr_t, y_tr_t = cwe._to_std_tensors(split["x_train"], split["y_train"], norm)
    x_val_t, y_val_t = cwe._to_std_tensors(split["x_val"], split["y_val"], norm)

    torch.manual_seed(seed)
    net = nwn.IndependentWidthNet(w_max=w_max)
    sub = net.subnets[width - 1]
    opt = torch.optim.Adam(sub.parameters(), lr=cwe.LR)

    def step() -> None:
        opt.zero_grad()
        loss = cwe._width_nll(net, width, x_tr_t, y_tr_t)
        loss.backward()
        opt.step()

    def val() -> float:
        with torch.no_grad():
            return float(cwe._width_nll(net, width, x_val_t, y_val_t).item())

    conv = cvg.fit_to_convergence(sub, step, val, max_epochs=max_epochs, check_every=check_every, patience=patience, min_delta=min_delta)
    return net, norm, conv


def _run_control_cell(args: argparse.Namespace) -> dict[str, Any]:
    """Builds one CONTROL per-cell result: trains width `args.width`, scores it held out and on train."""
    split = _build_split(args.toy, args.seed, args.n_train, args.n_test)
    net, norm, conv = _train_control(args.seed, args.w_max, args.width, split, args.max_epochs, args.check_every, args.patience, args.min_delta)

    ll_te = sw._score_all_widths(net, norm, split["x_test"], split["y_test"], DEVICE)[:, args.width - 1]
    nll_te = float(-ll_te.mean())
    mse_te = float(sw._score_all_widths_mse(net, norm, split["x_test"], split["y_test"], DEVICE)[:, args.width - 1].mean())
    mse_tr = float(sw._score_all_widths_mse(net, norm, split["x_train"], split["y_train"], DEVICE)[:, args.width - 1].mean())

    summary = conv.summary()
    return {
        "toy": args.toy.value,
        "seed": args.seed,
        "width": args.width,
        "arm": Arm.CONTROL.value,
        "held_out_mse": mse_te,
        "train_mse": mse_tr,
        "held_out_nll": nll_te,
        "held_out_trajectory": summary["trajectory"],
        "trustworthy": summary["trustworthy"],
        "hit_cap": summary["hit_cap"],
        "config": {
            "w_max": args.w_max,
            "n_train": args.n_train,
            "n_test": args.n_test,
            "lr": cwe.LR,
            "val_every": cwe.VAL_EVERY,
            "max_epochs_cap": args.max_epochs,
            "check_every": args.check_every,
            "patience": args.patience,
            "min_delta": args.min_delta,
        },
        "provenance": {
            "driver": _DRIVER_NAME,
            "reused_from": "automl_package/examples/converged_width_experiment.py (IndependentWidthNet, unmodified protocol)",
        },
    }


# ---------------------------------------------------------------------------
# Ported arm — FlexibleWidthNN at a single fixed width, plus MASTER Decision 16's escalation ladder.
# ---------------------------------------------------------------------------


def _apply_init_scheme(module: FlexibleWidthNN.FlexibleWidthNNModule, scheme: InitScheme) -> None:
    """Escalation-ladder init-scheme override (Decision 16); a no-op at the protocol default."""
    if scheme is InitScheme.DEFAULT:
        return
    for lin in (module.trunk_linear, *module.heads.values()):
        if scheme is InitScheme.KAIMING:
            nn.init.kaiming_uniform_(lin.weight, nonlinearity="relu")
        else:
            nn.init.xavier_uniform_(lin.weight)
        nn.init.zeros_(lin.bias)


def _replay(val_loss_history: list[float], patience: int, min_delta: float) -> cvg.ConvergenceResult:
    """Replays a per-epoch val-loss history through `ConvergenceTracker` for the trustworthy verdict.

    Same pattern as `moe_flexnn_comparison.py::_replay` (the established capacity-programme
    convention for package-model training that bypasses `.fit()`'s Optuna wrapper) — reused, not
    reinvented.
    """
    tracker = cvg.ConvergenceTracker(patience=patience, min_delta=min_delta)
    for epoch, val in enumerate(val_loss_history, start=1):
        tracker.update(epoch, val)
    return tracker.result(final_epoch=len(val_loss_history))


def _train_ported_default(model: FlexibleWidthNN, split: dict[str, np.ndarray]) -> list[float]:
    """Protocol-default path: the established `_fit_single` bypass, unmodified.

    Same package-model training pattern `moe_flexnn_comparison.py` uses. Taken whenever every escalation flag is at its default.
    """
    x_tr = split["x_train"].reshape(-1, 1).astype(np.float32)
    x_val = split["x_val"].reshape(-1, 1).astype(np.float32)
    _best_epoch, val_loss_history = model._fit_single(x_tr, split["y_train"], x_val=x_val, y_val=split["y_val"])
    return val_loss_history


def _train_ported_escalated(
    model: FlexibleWidthNN,
    split: dict[str, np.ndarray],
    *,
    n_epochs: int,
    patience: int,
    grad_clip_norm: float,
    warmup_epochs: int,
    init_scheme: InitScheme,
    batch_size: int,
    lr: float,
    seed: int,
) -> list[float]:
    """Escalation-ladder training loop (MASTER Decision 16: LR sweep -> clipping -> warmup -> init scheme -> normalization).

    `_fit_single` has no hook point between `build_model()` and training to inject an init-scheme
    override, and no hook inside its step loop for gradient clipping or LR warmup, so those three
    knobs need a standalone loop. It mirrors `_fit_single`'s own shape exactly otherwise (mini-batch
    Adam, per-epoch validation, best-weights early stopping) so an escalated run stays comparable to
    the protocol-default run it is meant to rescue. Only entered when `_run_ported_cell` sees at
    least one ladder flag off its default — never used for the primary numbers.
    """
    model.input_size = 1
    torch.manual_seed(seed)
    model.build_model()
    _apply_init_scheme(model.model, init_scheme)
    opt = torch.optim.Adam(model.model.parameters(), lr=lr)

    x_tr = torch.tensor(split["x_train"], dtype=torch.float32, device=model.device).reshape(-1, 1)
    y_tr = torch.tensor(split["y_train"], dtype=torch.float32, device=model.device).reshape(-1, 1)
    x_val = torch.tensor(split["x_val"], dtype=torch.float32, device=model.device).reshape(-1, 1)
    y_val = torch.tensor(split["y_val"], dtype=torch.float32, device=model.device).reshape(-1, 1)
    dataset = torch.utils.data.TensorDataset(x_tr, y_tr)

    best_val, patience_counter, best_state = float("inf"), 0, None
    val_loss_history: list[float] = []
    for epoch in range(n_epochs):
        if warmup_epochs > 0:
            warmup_lr = lr * min(1.0, (epoch + 1) / warmup_epochs)
            for group in opt.param_groups:
                group["lr"] = warmup_lr
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model.model.train()
        for batch_x, batch_y in loader:
            opt.zero_grad()
            loss = model.criterion(model.model(batch_x), batch_y)
            loss.backward()
            if grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.model.parameters(), grad_clip_norm)
            opt.step()

        model.model.eval()
        with torch.no_grad():
            val_loss = float(model.criterion(model.model(x_val), y_val).item())
        val_loss_history.append(val_loss)
        if val_loss < best_val:
            best_val, patience_counter = val_loss, 0
            best_state = {k: t.detach().clone() for k, t in model.model.state_dict().items()}
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    if best_state is not None:
        model.model.load_state_dict(best_state)
    return val_loss_history


def _ported_score(model: FlexibleWidthNN, x_raw: np.ndarray, y_raw: np.ndarray, width: int, norm: dict | None) -> float:
    """Held-out/train MSE in ORIGINAL y-units — same units as the control arm's `_score_all_widths_mse`."""
    x_in = ((x_raw - norm["mx"]) / norm["sx"]) if norm else x_raw
    pred = model.predict(x_in.reshape(-1, 1).astype(np.float32), filter_data=False, width=width)
    if norm:
        pred = pred * norm["sy"] + norm["my"]
    return float(np.mean((pred - y_raw) ** 2))


def _run_ported_cell(args: argparse.Namespace) -> dict[str, Any]:
    """Builds one PORTED per-cell result: `FlexibleWidthNN(widths=(width,))`, trained ordinarily."""
    split = _build_split(args.toy, args.seed, args.n_train, args.n_test)

    norm = cwe._standardize_fit(split["x_train"], split["y_train"]) if args.normalize_inputs else None
    train_split = split
    if norm is not None:
        train_split = dict(split)
        train_split["x_train"] = (split["x_train"] - norm["mx"]) / norm["sx"]
        train_split["y_train"] = (split["y_train"] - norm["my"]) / norm["sy"]
        train_split["x_val"] = (split["x_val"] - norm["mx"]) / norm["sx"]
        train_split["y_val"] = (split["y_val"] - norm["my"]) / norm["sy"]

    model = FlexibleWidthNN(
        input_size=1,
        output_size=1,
        task_type=TaskType.REGRESSION,
        widths=(args.width,),
        learning_rate=args.lr,
        n_epochs=args.max_epochs,
        early_stopping_rounds=args.ported_patience,
        batch_size=args.batch_size,
        random_seed=args.seed,
        calculate_feature_importance=False,
        capacity_selection=CapacitySelection.FIXED,
        # Tanh, NOT the class's ReLU default: the control arm's width classes are Tanh
        # (`automl_package/models/flexnn/width/architectures.py:48`), and `width.md` §1's confound
        # doctrine forbids an arm differing from its comparator in more than one respect. The first
        # ported run (ReLU, retained under `WSEL4/relu_confounded_run/`) under-fit the TRAIN set at
        # every width >= 4 (ratios up to 24x vs control) — activation was the second difference.
        activation=ActivationFunction.TANH,
    )

    escalated = args.grad_clip_norm > 0 or args.warmup_epochs > 0 or args.init_scheme is not InitScheme.DEFAULT
    if escalated:
        val_loss_history = _train_ported_escalated(
            model,
            train_split,
            n_epochs=args.max_epochs,
            patience=args.ported_patience,
            grad_clip_norm=args.grad_clip_norm,
            warmup_epochs=args.warmup_epochs,
            init_scheme=args.init_scheme,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
        )
    else:
        val_loss_history = _train_ported_default(model, train_split)

    replay = _replay(val_loss_history, args.ported_patience, args.ported_min_delta)
    hit_cap = bool(len(val_loss_history) >= args.max_epochs)

    mse_te = _ported_score(model, split["x_test"], split["y_test"], args.width, norm)
    mse_tr = _ported_score(model, split["x_train"], split["y_train"], args.width, norm)

    return {
        "toy": args.toy.value,
        "seed": args.seed,
        "width": args.width,
        "arm": Arm.PORTED.value,
        "held_out_mse": mse_te,
        "train_mse": mse_tr,
        "held_out_trajectory": replay.summary()["trajectory"],
        "trustworthy": bool(replay.trustworthy and not hit_cap),
        "hit_cap": hit_cap,
        "config": {
            "widths": [args.width],
            "n_train": args.n_train,
            "n_test": args.n_test,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "max_epochs_cap": args.max_epochs,
            "patience": args.ported_patience,
            "min_delta": args.ported_min_delta,
            "escalation": {
                "grad_clip_norm": args.grad_clip_norm,
                "warmup_epochs": args.warmup_epochs,
                "init_scheme": args.init_scheme.value,
                "normalize_inputs": args.normalize_inputs,
            },
        },
        "provenance": {
            "driver": _DRIVER_NAME,
            "class": "FlexibleWidthNN (automl_package/models/flexnn/width/model.py, SharedTrunkPerWidthHeadNet port)",
        },
    }


# ---------------------------------------------------------------------------
# --summarize — reproduction.json (MASTER Decision 14's positive-control check) + the ported-vs-
# control comparison table.
# ---------------------------------------------------------------------------


def _load_reference() -> dict:
    with open(W_CONVERGED_PATH) as f:
        return json.load(f)


def _reference_nll(reference: dict, seed: int, width: int) -> float:
    for case in reference["per_case"]:
        if case["seed"] == seed:
            return float(case["per_k_nll"][str(width)])
    available = [c["seed"] for c in reference["per_case"]]
    raise ValueError(f"seed={seed} not found in W_CONVERGED reference (available seeds: {available})")


def _load_cells(results_dir: str, toy: nwn.Toy) -> tuple[dict[tuple[int, int], dict], dict[tuple[int, int], dict]]:
    """Scans `results_dir` for landed per-cell JSONs, grouped by arm.

    Keyed off each JSON's OWN `seed`/`width` fields, not the filename, so a rename can't silently desync the two.
    """
    control: dict[tuple[int, int], dict] = {}
    ported: dict[tuple[int, int], dict] = {}
    for path in sorted(glob.glob(os.path.join(results_dir, f"{toy.value}_*.json"))):
        if os.path.basename(path) == "reproduction.json":
            continue
        with open(path) as f:
            cell = json.load(f)
        key = (int(cell["seed"]), int(cell["width"]))
        if cell["arm"] == Arm.CONTROL.value:
            control[key] = cell
        elif cell["arm"] == Arm.PORTED.value:
            ported[key] = cell
    return control, ported


def _summarize(results_dir: str, toy: nwn.Toy) -> dict[str, Any]:
    """Builds `reproduction.json`'s content.

    Per-control-cell relative error vs `W_CONVERGED`, the positive-control gate (MASTER Decision 14), and the ported-vs-control comparison table.
    """
    reference = _load_reference()
    ref_w_max = int(reference["config"]["w_max"])
    ref_seeds = [int(c["seed"]) for c in reference["per_case"]]
    expected = {(seed, width) for seed in ref_seeds for width in range(1, ref_w_max + 1)}

    control, ported = _load_cells(results_dir, toy)

    cells = []
    for seed, width in sorted(control):
        cell = control[(seed, width)]
        ref_nll = _reference_nll(reference, seed, width)
        rel_err = abs(cell["held_out_nll"] - ref_nll) / (abs(ref_nll) + _REL_ERR_EPS)
        cells.append(
            {
                "toy": toy.value,
                "seed": seed,
                "width": width,
                "control_nll": cell["held_out_nll"],
                "reference_nll": ref_nll,
                "relative_error": rel_err,
                "within_bar": bool(rel_err <= REPRODUCTION_BAR),
            }
        )

    missing_control = sorted(expected - set(control))
    control_complete = not missing_control
    control_all_within_bar = all(c["within_bar"] for c in cells) if cells else False
    ported_present = bool(ported)
    positive_control_violation = ported_present and (not control_complete or not control_all_within_bar)

    comparison = []
    for seed, width in sorted(set(control) & set(ported)):
        c, p = control[(seed, width)], ported[(seed, width)]
        comparison.append(
            {
                "toy": toy.value,
                "seed": seed,
                "width": width,
                "control_held_out_mse": c["held_out_mse"],
                "ported_held_out_mse": p["held_out_mse"],
                "control_trustworthy": c["trustworthy"],
                "ported_trustworthy": p["trustworthy"],
                "mse_ratio_ported_over_control": (p["held_out_mse"] / c["held_out_mse"]) if c["held_out_mse"] else None,
            }
        )

    summary = {
        "bar": REPRODUCTION_BAR,
        "n_expected": len(expected),
        "n_control_landed": len(control),
        "n_ported_landed": len(ported),
        "control_complete": control_complete,
        "missing_control_cells": [{"seed": s, "width": w} for s, w in missing_control],
        "control_all_within_bar": control_all_within_bar,
        "positive_control_violation": positive_control_violation,
        "cells": cells,
        "ported_vs_control": comparison,
    }

    if positive_control_violation:
        print("!" * 78)
        print("POSITIVE CONTROL VIOLATION (MASTER Decision 14): ported cells are on disk while the")
        print(f"control has NOT cleanly reproduced W_CONVERGED (missing={len(missing_control)}, ")
        print(f"all_within_{REPRODUCTION_BAR:.0%}_bar={control_all_within_bar}). Per width.md's WSEL-4")
        print("branch table (§3.5): report the discrepancy and HALT WSEL-8.")
        print("!" * 78)

    return summary


# ---------------------------------------------------------------------------
# Selftest — one CONTROL cell, two PORTED cells (protocol-default + every escalation knob exercised
# once), then --summarize over just those, all inside a throwaway tmp dir.
# ---------------------------------------------------------------------------


def run_selftest() -> bool:
    """Tiny-budget end-to-end pass of both arms on one cell plus a synthetic-scale summarize.

    Writes only into a `tempfile.mkdtemp()` directory (never `capacity_ladder_results/WSEL4/`), so
    the real results tree is never polluted by selftest output (the driver contract's own rule).
    """
    tmp_dir = tempfile.mkdtemp(prefix="width_wsel4_selftest_")
    try:
        toy, seed, width, w_max = nwn.Toy.HETERO, 0, 2, 3
        n_train, n_test = 60, 30

        control_args = argparse.Namespace(
            toy=toy,
            seed=seed,
            width=width,
            w_max=w_max,
            n_train=n_train,
            n_test=n_test,
            max_epochs=200,
            check_every=10,
            patience=3,
            min_delta=1e-3,
        )
        control_cell = _run_control_cell(control_args)
        ok_control = (
            len(control_cell["held_out_trajectory"]) >= 1
            and isinstance(control_cell["trustworthy"], bool)
            and isinstance(control_cell["hit_cap"], bool)
            and math.isfinite(control_cell["held_out_mse"])
        )
        with open(os.path.join(tmp_dir, f"{toy.value}_{seed}_{width}_control.json"), "w") as f:
            json.dump(sw._jsonable(control_cell), f, indent=2)

        ported_default_args = argparse.Namespace(
            toy=toy,
            seed=seed,
            width=width,
            n_train=n_train,
            n_test=n_test,
            max_epochs=15,
            lr=PORTED_LR_DEFAULT,
            batch_size=16,
            ported_patience=3,
            ported_min_delta=1e-3,
            grad_clip_norm=0.0,
            warmup_epochs=0,
            init_scheme=InitScheme.DEFAULT,
            normalize_inputs=False,
        )
        ported_default_cell = _run_ported_cell(ported_default_args)
        ok_ported_default = (
            len(ported_default_cell["held_out_trajectory"]) >= 1
            and isinstance(ported_default_cell["trustworthy"], bool)
            and isinstance(ported_default_cell["hit_cap"], bool)
            and math.isfinite(ported_default_cell["held_out_mse"])
        )
        with open(os.path.join(tmp_dir, f"{toy.value}_{seed}_{width}.json"), "w") as f:
            json.dump(sw._jsonable(ported_default_cell), f, indent=2)

        escalated_width = width + 1 if width + 1 <= w_max else width
        ported_escalated_args = argparse.Namespace(
            toy=toy,
            seed=seed,
            width=escalated_width,
            n_train=n_train,
            n_test=n_test,
            max_epochs=15,
            lr=PORTED_LR_DEFAULT,
            batch_size=16,
            ported_patience=3,
            ported_min_delta=1e-3,
            grad_clip_norm=_SELFTEST_GRAD_CLIP_NORM,
            warmup_epochs=_SELFTEST_WARMUP_EPOCHS,
            init_scheme=InitScheme.KAIMING,
            normalize_inputs=True,
        )
        ported_escalated_cell = _run_ported_cell(ported_escalated_args)
        esc = ported_escalated_cell["config"]["escalation"]
        ok_ported_escalated = (
            len(ported_escalated_cell["held_out_trajectory"]) >= 1
            and math.isfinite(ported_escalated_cell["held_out_mse"])
            and esc["grad_clip_norm"] == _SELFTEST_GRAD_CLIP_NORM
            and esc["warmup_epochs"] == _SELFTEST_WARMUP_EPOCHS
            and esc["init_scheme"] == InitScheme.KAIMING.value
            and esc["normalize_inputs"] is True
        )

        summary = _summarize(tmp_dir, toy)
        with open(os.path.join(tmp_dir, "reproduction.json"), "w") as f:
            json.dump(sw._jsonable(summary), f, indent=2)
        ok_summarize = (
            summary["n_control_landed"] == 1
            and summary["n_ported_landed"] == 1  # only the DEFAULT ported cell shares (seed, width) with the control cell
            and len(summary["cells"]) == 1
            and "relative_error" in summary["cells"][0]
            and len(summary["ported_vs_control"]) == 1
        )

        ok = ok_control and ok_ported_default and ok_ported_escalated and ok_summarize
        print(
            f"[width_wsel4 selftest] control={ok_control} ported_default={ok_ported_default} "
            f"ported_escalated={ok_ported_escalated} summarize={ok_summarize}  {'PASS' if ok else 'FAIL'}"
        )
        return ok
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Runs one per-cell training (`--arm control`/`ported`), `--summarize`, or `--selftest`."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true", help="Tiny end-to-end wiring check (both arms + escalation ladder), then exit.")
    parser.add_argument("--summarize", action="store_true", help="Aggregate landed cells into reproduction.json, then exit.")

    parser.add_argument("--toy", choices=[nwn.Toy.HETERO.value], default=nwn.Toy.HETERO.value, help="Only hetero has a landed W_CONVERGED reference (WSEL-4's scope).")
    parser.add_argument("--seed", type=int, default=None, help="Required outside --selftest/--summarize.")
    parser.add_argument("--width", type=int, default=None, help="Required outside --selftest/--summarize.")
    parser.add_argument("--arm", choices=[a.value for a in Arm], default=None, help="Required outside --selftest/--summarize.")
    parser.add_argument("--w-max", type=int, default=cwe.W_MAX, help="Control-arm ladder ceiling (default: converged_width_experiment.W_MAX).")
    parser.add_argument("--n-train", type=int, default=cwe.N_TRAIN, help="Default: converged_width_experiment.N_TRAIN.")
    parser.add_argument("--n-test", type=int, default=cwe.N_TEST, help="Default: converged_width_experiment.N_TEST.")
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR)

    parser.add_argument("--max-epochs", type=int, default=None, help="Safety cap. Default: 40000 (control) / 1500 (ported) — each arm's own protocol default.")
    parser.add_argument("--check-every", type=int, default=cvg.DEFAULT_CHECK_EVERY, help="Control arm only: epochs between held-out checkpoints.")
    parser.add_argument("--patience", type=int, default=cvg.DEFAULT_PATIENCE, help="Control arm only: checkpoints with no improvement before declaring convergence.")
    parser.add_argument("--min-delta", type=float, default=cvg.DEFAULT_MIN_DELTA, help="Control arm only: held-out-loss decrease counted as improvement.")

    parser.add_argument("--lr", type=float, default=PORTED_LR_DEFAULT, help="Ported arm; escalation ladder step 1 (LR sweep).")
    parser.add_argument("--batch-size", type=int, default=PORTED_BATCH_SIZE, help="Ported arm only.")
    parser.add_argument("--ported-patience", type=int, default=PORTED_PATIENCE, help="Ported arm only (epochs, not checkpoints — _fit_single validates every epoch).")
    parser.add_argument("--ported-min-delta", type=float, default=PORTED_MIN_DELTA, help="Ported arm only.")
    parser.add_argument("--grad-clip-norm", type=float, default=0.0, help="Ported arm; escalation ladder step 2. 0 = protocol default (off).")
    parser.add_argument("--warmup-epochs", type=int, default=0, help="Ported arm; escalation ladder step 3. 0 = protocol default (off).")
    parser.add_argument("--init-scheme", choices=[s.value for s in InitScheme], default=InitScheme.DEFAULT.value, help="Ported arm; escalation ladder step 4.")
    parser.add_argument("--normalize-inputs", action="store_true", help="Ported arm; escalation ladder step 5 (standardize x/y before training).")

    args = parser.parse_args()
    args.toy = nwn.Toy(args.toy)
    args.init_scheme = InitScheme(args.init_scheme)

    if args.selftest and args.summarize:
        parser.error("--selftest and --summarize are mutually exclusive.")
    if args.selftest:
        sys.exit(0 if run_selftest() else 1)
    if args.summarize:
        summary = _summarize(args.results_dir, args.toy)
        os.makedirs(args.results_dir, exist_ok=True)
        out_path = os.path.join(args.results_dir, "reproduction.json")
        with open(out_path, "w") as f:
            json.dump(sw._jsonable(summary), f, indent=2)
        print(
            f"[width_wsel4] wrote {out_path} (control {summary['n_control_landed']}/{summary['n_expected']} landed, "
            f"complete={summary['control_complete']}, all_within_bar={summary['control_all_within_bar']}; "
            f"ported {summary['n_ported_landed']}/{summary['n_expected']} landed)"
        )
        return

    if args.seed is None or args.width is None or args.arm is None:
        parser.error("--seed, --width and --arm are required outside --selftest/--summarize.")
    args.arm = Arm(args.arm)

    escalation_set = args.grad_clip_norm > 0 or args.warmup_epochs > 0 or args.init_scheme is not InitScheme.DEFAULT or args.normalize_inputs
    if args.arm is Arm.CONTROL and escalation_set:
        parser.error(
            "Escalation-ladder flags (--grad-clip-norm/--warmup-epochs/--init-scheme/--normalize-inputs) apply to "
            "--arm ported only — the control's protocol is frozen (WSEL-4 non-goal: 'a port that changes the protocol is not a port')."
        )
    if args.max_epochs is None:
        args.max_epochs = CONTROL_MAX_EPOCHS_DEFAULT if args.arm is Arm.CONTROL else PORTED_N_EPOCHS_CAP

    os.makedirs(args.results_dir, exist_ok=True)
    cell = _run_control_cell(args) if args.arm is Arm.CONTROL else _run_ported_cell(args)
    suffix = "_control" if args.arm is Arm.CONTROL else ""
    out_path = os.path.join(args.results_dir, f"{args.toy.value}_{args.seed}_{args.width}{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(sw._jsonable(cell), f, indent=2)
    print(f"[width_wsel4] wrote {out_path} (held_out_mse={cell['held_out_mse']:.6f} trustworthy={cell['trustworthy']} hit_cap={cell['hit_cap']})")


if __name__ == "__main__":
    main()
