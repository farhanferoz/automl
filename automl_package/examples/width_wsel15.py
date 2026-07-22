"""WSEL-15 -- does the nested width design survive a normalisation layer?

`docs/plans/capacity_programme/width.md` ~1416-1515, the transformer-port repairs measured.
`shared/width_transformer_port.md` SS5 lists four repairs for the one obstacle standing between the
certified nested-width design and a transformer port: a normalisation layer computes its statistics
over the WHOLE vector, so truncating to a prefix changes the divisor for every surviving unit. Three
of the four repairs are testable on today's toy (the fourth, a rung-independent normaliser, is
explicitly OUT -- its literature is unverified). This driver measures repair 2 (cumulative prefix
statistics) and repair 3 (per-rung normalisation parameters) against the unnormalised reference, on the
SAME toy, seeds and bars as the rest of the width strand, so the answer is comparable rather than a
side-experiment.

**Arms (all on the certified `SharedTrunkPerWidthHeadNet` shape, MSE-only per SS3.7).** Defined once,
in `automl_package/examples/width_candidates.py` (SS3.9's ONE home for candidate width architectures --
this driver builds nets, it does not define them):

  - A -- no normalisation. The certified net, unchanged. The reference.
  - B -- prefix normalisation via running totals (`PrefixNormMode.RUNNING_TOTALS`): one `cumsum(h**2)`
    pass covers every width's RMS normaliser at once. NO affine parameters.
  - C -- B plus a per-width SCALAR `gamma_k`/`beta_k` (`PrefixNormMode.AFFINE`); tests whether the
    per-width head already absorbs the rung-dependent divisor (prediction on record: it does).

Arm D (the naive, looped, correctness-oracle computation `PrefixNormMode.NAIVE` is checked against) is
**not a grid arm** -- it is exercised ONLY by `tests/test_prefix_norm_equivalence.py` (Step 1), never
trained here. `check_prefix_norm_exact` below is Step 1's load-bearing check; that test file imports it
directly so the pytest gate and `frozen.json`'s `prefix_norm_exact` field can never silently drift
apart from each other.

**Non-goals** (SS4): no transformer, no attention, no real data, no multi-layer net, no variance
fitting, no change to the toy/schedule/selection rule, no mean-centred normalisation (a different
question -- it needs a second cumulative sum and interacts with the head's bias term, moving two things
at once), and no promotion of any variant into the package (WSEL-17's job, gated on this task passing).

Driver CLI contract (root-run grid; this file is never run over the full grid by its author):
  `--arm {a,b,c} --seed <int>` runs ONE cell, writing its per-cell JSON immediately.
  `--summarize` aggregates every per-cell JSON on disk into `WSEL15/frozen.json`.
  `--selftest` runs a tiny w_max=4 config for every grid arm plus `check_prefix_norm_exact` at a tiny
  size -- no real cell is ever run here.

Usage:
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel15.py --selftest
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel15.py --arm a --seed 0
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel15.py --summarize
"""

from __future__ import annotations

import argparse
import copy
import enum
import json
import os
import sys
import time

import numpy as np
import torch

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root

import converged_width_experiment as cwe  # noqa: E402
import kdropout_converged_width_experiment as kce  # noqa: E402
import nested_width_net as nwn  # noqa: E402
import width_candidates as wc  # noqa: E402

from automl_package.utils.capacity_accounting import executed_flops, param_count  # noqa: E402
from automl_package.utils.run_provenance import run_provenance  # noqa: E402

RESULTS_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "WSEL15")

# Canonical cell, reused verbatim so this task's numbers sit in the same table as WSEL-13/14
# (`width.md`'s canonical cell: `--arch shared_trunk --loss mse --toy hetero`, n_train=1500, sigma=0.05,
# w_max=12). WSEL-12 must be merged first so arm A's cost numbers are measured on the fixed training
# loop (this task's stated dependency).
SEEDS = cwe.SEEDS  # (0, 1, 2)
W_MAX = cwe.W_MAX  # 12
N_TRAIN = cwe.N_TRAIN  # 1500
N_TEST = cwe.N_TEST  # 500
SIGMA = nwn.HETERO_NOISE_SIGMA  # 0.05
TOY = nwn.Toy.HETERO
ARCH = kce.Arch.SHARED_TRUNK  # checkpointing-branch selector only; arms B/C are still whole-net-checkpointed like SHARED_TRUNK.
LOSS = kce.LossType.MSE  # MSE-only, no variance fit (SS3.7).
SCHEDULE = nwn.WidthSchedule.SANDWICH  # DEFAULT: the certified schedule -- byte-identical to the landed grid.
# WSEL-15 FOLLOW-UP (width.md, 2026-07-22): `--schedule all` re-runs the arms with NO width starved, to
# separate "normalisation helps the mids" from "longer training feeds the mids the sandwich starves".
# Results are segregated into WSEL15_ALLSCHED/ so the landed WSEL15/ grid and its frozen.json can never
# be clobbered, nor aggregated with mixed-schedule cells.
_ALLSCHED_RESULTS_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "WSEL15_ALLSCHED")


def _apply_schedule(value: str) -> None:
    """Points the module schedule + results dir at the requested follow-up arm (default: unchanged)."""
    global SCHEDULE, RESULTS_DIR  # noqa: PLW0603 -- this script's config IS module-level constants (existing idiom); one mutation point, before any work runs.
    SCHEDULE = nwn.WidthSchedule(value)
    if SCHEDULE is not nwn.WidthSchedule.SANDWICH:
        RESULTS_DIR = _ALLSCHED_RESULTS_DIR

DEFAULT_MAX_EPOCHS = kce.DEFAULT_MAX_EPOCHS
DEFAULT_CHECK_EVERY = kce.cvg.DEFAULT_CHECK_EVERY
DEFAULT_PATIENCE = kce.cvg.DEFAULT_PATIENCE
DEFAULT_MIN_DELTA = kce.cvg.DEFAULT_MIN_DELTA

# Step 1 (`check_prefix_norm_exact`) bar and default toy size -- `width.md` ~1466-1471, no discretion.
_PREFIX_NORM_TOL = 1e-5
_PREFIX_NORM_CHECK_N = 64
_PREFIX_NORM_CHECK_TRAIN_STEPS = 10
_PREFIX_NORM_CHECK_LR = 1e-2

# Pre-registered bars (`width.md` ~1487-1500), fixed BEFORE any run.
_ACCURACY_REL_TOL = 0.10  # arm B (and, for the affine question, arm C) vs arm A/B, per width.
_COST_WALLCLOCK_RATIO_BAR = 1.3  # arm B's per-step wall-clock must stay within 1.3x arm A's.
_AFFINE_NEEDED_REL_GAIN = 0.10  # per_width_affine_needed iff arm C beats arm B by > 10% relative MSE at any width.


class Arm(enum.Enum):
    """WSEL-15's three GRID arms (closed set).

    Arm D (`PrefixNormMode.NAIVE`) is not a grid arm -- see module docstring; it exists only inside
    `width_candidates.py` as Step 1's correctness oracle.
    """

    A = "a"  # no normalisation -- the certified SharedTrunkPerWidthHeadNet, unchanged. The reference.
    B = "b"  # prefix RMS normalisation via running totals (PrefixNormMode.RUNNING_TOTALS), no affine.
    C = "c"  # B plus a per-width scalar affine (PrefixNormMode.AFFINE).


_ARM_MODE: dict[Arm, wc.PrefixNormMode | None] = {
    Arm.A: None,  # None => plain SharedTrunkPerWidthHeadNet, no wrapper.
    Arm.B: wc.PrefixNormMode.RUNNING_TOTALS,
    Arm.C: wc.PrefixNormMode.AFFINE,
}


def _build_net(arm: Arm, w_max: int) -> nwn.SharedTrunkPerWidthHeadNet | wc.PrefixNormWidthNet:
    """Arm A is the certified net directly; arms B/C are `PrefixNormWidthNet` wrapping a fresh one."""
    mode = _ARM_MODE[arm]
    if mode is None:
        return nwn.SharedTrunkPerWidthHeadNet(w_max=w_max)
    return wc.PrefixNormWidthNet(w_max=w_max, mode=mode)


def _params_effective_triangle(w_max: int) -> int:
    """The `1+2+...+w_max` triangle (`shared/width_transformer_port.md` SS3).

    Head `k` reads a vector whose tail is zeroed, so only the first `k` of its `w_max` input columns
    can ever influence width-k's output -- summed over every width, the per-width heads' EFFECTIVE
    (not allocated) input-weight count is `w_max*(w_max+1)/2`, "roughly half the allocation" (SS3's
    own words for `w_max=12`: 78 vs
    the 12*13=156 columns actually allocated across the 12 heads). This is a structural property of the
    shared-trunk-plus-per-width-head design shared by arms A/B/C (same trunk, same heads; normalisation
    only changes what those heads read, not which trunk columns feed which head), so it depends on
    `w_max` alone, not on the arm -- unlike `params_allocated`, which DOES grow with arm C's 2 extra
    scalars per width.
    """
    return w_max * (w_max + 1) // 2


def check_prefix_norm_exact(
    *, w_max: int = W_MAX, n: int = _PREFIX_NORM_CHECK_N, seed: int = 0, n_train_steps: int = _PREFIX_NORM_CHECK_TRAIN_STEPS,
    lr: float = _PREFIX_NORM_CHECK_LR, tol: float = _PREFIX_NORM_TOL,
) -> tuple[bool, float]:
    """Step 1's load-bearing check: does arm B (running-totals cumsum) equal arm D (naive per-k slice)?

    Builds ONE `PrefixNormWidthNet(mode=RUNNING_TOTALS)` under a fixed seed, `deepcopy`s it (so arm D
    starts from BIT-IDENTICAL weights without needing to replay the RNG stream) and flips the copy's
    `mode` to `NAIVE` -- `mode` is a plain attribute, not a parameter, so this is the only difference
    between the two nets. Compares every width's output on a fixed `(n, 1)` input at initialisation AND
    after `n_train_steps` full-batch Adam steps (same input/target, same lr, applied identically to
    both nets) -- exactly Step 1's spec (`width.md` ~1466-1471).

    `tests/test_prefix_norm_equivalence.py` calls this SAME function (not a re-derived comparison), so
    the pytest gate and `frozen.json`'s `prefix_norm_exact` field can never silently disagree.

    Returns:
        `(exact, max_abs_err)` -- `exact = max_abs_err < tol`, checked across every width and both
        checkpoints (init, post-training).
    """
    torch.manual_seed(seed)
    net_b = wc.PrefixNormWidthNet(w_max=w_max, mode=wc.PrefixNormMode.RUNNING_TOTALS)
    net_d = copy.deepcopy(net_b)
    net_d.mode = wc.PrefixNormMode.NAIVE
    x = torch.randn(n, 1)
    y = torch.randn(n)

    def _max_abs_err() -> float:
        net_b.eval()
        net_d.eval()
        max_err = 0.0
        with torch.no_grad():
            for k in range(1, w_max + 1):
                mean_b, _ = net_b.forward_width(x, k)
                mean_d, _ = net_d.forward_width(x, k)
                max_err = max(max_err, (mean_b - mean_d).abs().max().item())
        return max_err

    max_err = _max_abs_err()  # at initialisation

    for net in (net_b, net_d):
        net.train()
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        for _ in range(n_train_steps):
            opt.zero_grad()
            total = torch.zeros(())
            for k in range(1, net.w_max + 1):
                mean, _log_var = net.forward_width(x, k)
                total = total + ((mean.squeeze(1) - y) ** 2).mean()
            total.backward()
            opt.step()

    max_err = max(max_err, _max_abs_err())  # after n_train_steps
    return max_err < tol, max_err


def run_cell(
    arm: Arm,
    seed: int,
    *,
    w_max: int = W_MAX,
    n_train: int = N_TRAIN,
    n_test: int = N_TEST,
    sigma: float = SIGMA,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    check_every: int = DEFAULT_CHECK_EVERY,
    patience: int = DEFAULT_PATIENCE,
    min_delta: float = DEFAULT_MIN_DELTA,
) -> tuple[dict, dict]:
    """Trains one (arm, seed) cell to per-width convergence, then records the Step-3 cost/accuracy fields.

    Data prep / phase-1 train-val carve mirrors `kdropout_converged_width_experiment.run_case` verbatim
    for the canonical `toy=hetero, arch=shared_trunk, loss=mse` cell -- reimplemented here (not called)
    because this task needs the trained net object directly (for `param_count`/`executed_flops`,
    neither of which `run_case` returns), same reasoning as `width_wsel13.py`'s `run_cell`. The TRAINING
    LOOP itself is never reimplemented: `_train_kdropout_to_convergence` is imported and called as-is.

    Returns:
        `(case, state_dict)` -- `case` is the JSON-able per-cell record; `state_dict` is the trained
        net's own (already best-checkpoint-restored) parameters.
    """
    x_tr, y_tr, _reg_tr = nwn.make_hetero(n_train, seed, sigma=sigma)
    x_te, y_te, _reg_te = nwn.make_hetero(n_test, seed + 500, sigma=sigma)

    p1_idx = np.arange(0, n_train, 2)
    x_p1, y_p1 = x_tr[p1_idx], y_tr[p1_idx]
    val_mask = (np.arange(len(x_p1)) % cwe.VAL_EVERY) == 0
    norm = cwe._standardize_fit(x_p1[~val_mask], y_p1[~val_mask])
    x_tr_t, y_tr_t = cwe._to_std_tensors(x_p1[~val_mask], y_p1[~val_mask], norm)
    x_val_t, y_val_t = cwe._to_std_tensors(x_p1[val_mask], y_p1[val_mask], norm)
    x_te_t, y_te_t = cwe._to_std_tensors(x_te, y_te, norm)

    torch.manual_seed(seed)
    net = _build_net(arm, w_max)

    _train_t0 = time.perf_counter()
    conv, _best_mean_val_epoch = kce._train_kdropout_to_convergence(
        net, x_tr_t, y_tr_t, x_val_t, y_val_t,
        arch=ARCH, loss=LOSS, max_epochs=max_epochs, check_every=check_every,
        patience=patience, min_delta=min_delta, seed=seed, schedule=SCHEDULE,
    )
    train_wall_clock_s = time.perf_counter() - _train_t0
    # The joint training loop stops the instant every width's own tracker has converged (or the cap is
    # hit) -- that epoch is `max` over every width's own `stop_epoch` (a hit-cap width's `stop_epoch` is
    # already set to the loop's `final_epoch`, per `ConvergenceTracker.result`), so this needs no new
    # return value from `_train_kdropout_to_convergence` (out of this task's write set).
    steps_to_converge = max(r.stop_epoch for r in conv.values())

    n_trustworthy = sum(1 for r in conv.values() if r.trustworthy)
    all_trustworthy = n_trustworthy == w_max
    hit_cap = any(r.hit_cap for r in conv.values())

    net.eval()
    with torch.no_grad():
        train_mse_by_width = {k: float(nwn._width_mse(net, k, x_tr_t, y_tr_t).item()) for k in range(1, w_max + 1)}
        heldout_mse_by_width = {k: float(nwn._width_mse(net, k, x_te_t, y_te_t).item()) for k in range(1, w_max + 1)}

    executed_flops_by_width = {k: executed_flops(net, k) for k in range(1, w_max + 1)}
    params_allocated = param_count(net)
    params_effective = _params_effective_triangle(w_max)

    case = {
        "arm": arm.value,
        "seed": seed,
        "toy": TOY.value,
        "n_train": n_train,
        "n_test": n_test,
        "sigma": sigma,
        "w_max": w_max,
        "convergence": {k: r.summary() for k, r in conv.items()},
        "n_widths_trustworthy": n_trustworthy,
        "all_widths_trustworthy": all_trustworthy,
        "hit_cap": hit_cap,
        # Step 3 cost instrumentation -- same fields as WSEL-14 (`width.md` ~1475-1478).
        "train_wall_clock_s": train_wall_clock_s,
        "steps_to_converge": steps_to_converge,
        "params_allocated": params_allocated,
        "params_effective": params_effective,
        "executed_flops_by_width": {str(k): v for k, v in executed_flops_by_width.items()},
        "train_mse_by_width": {str(k): v for k, v in train_mse_by_width.items()},
        "heldout_mse_by_width": {str(k): v for k, v in heldout_mse_by_width.items()},
        "training_schedule": SCHEDULE.value,
        "provenance": run_provenance(),
    }
    return case, {name: t.detach().clone() for name, t in net.state_dict().items()}


def _cell_json_path(arm: Arm, seed: int) -> str:
    return os.path.join(RESULTS_DIR, f"wsel15_{arm.value}_seed{seed}.json")


def _state_path(arm: Arm, seed: int) -> str:
    return os.path.join(RESULTS_DIR, f"state_{arm.value}_seed{seed}.pt")


def _jsonable(obj: object) -> object:
    """Minimal JSON-safe coercion (numpy/torch scalars, dict int-keys) -- local twin of `width_wsel13._jsonable`."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj


def run_selftest() -> bool:
    """Tiny wiring check: `check_prefix_norm_exact` at a tiny size, then a tiny w_max=4 cell per grid arm."""
    ok = True

    exact, err = check_prefix_norm_exact(w_max=4, n=32, n_train_steps=5)
    print(f"[wsel15 selftest] check_prefix_norm_exact (arm B vs arm D, tiny): max_abs_err={err:.3e} (tol={_PREFIX_NORM_TOL:.0e})  {'PASS' if exact else 'FAIL'}")
    ok = ok and exact

    required_keys = (
        "train_wall_clock_s", "steps_to_converge", "params_allocated", "params_effective",
        "executed_flops_by_width", "train_mse_by_width", "heldout_mse_by_width", "hit_cap",
    )
    for arm in Arm:
        case, _state = run_cell(arm, seed=0, w_max=4, n_train=200, n_test=100, max_epochs=1500, check_every=100, patience=3, min_delta=2e-3)
        keys_ok = all(key in case for key in required_keys)
        conv_ok = all(len(case["convergence"][k]["trajectory"]) >= 1 for k in range(1, 5)) and all(isinstance(case["convergence"][k]["converged"], bool) for k in range(1, 5))
        combo_ok = keys_ok and conv_ok
        print(f"[wsel15 selftest] arm={arm.value} cost_fields_present={keys_ok} convergence_trajectories={conv_ok}  {'PASS' if combo_ok else 'FAIL'}")
        ok = ok and combo_ok

    print(f"[wsel15 selftest] {'PASS' if ok else 'FAIL'}")
    return ok


def summarize() -> None:
    """Aggregates every per-cell JSON on disk into `WSEL15/frozen.json`. Does not train anything."""
    per_arm: dict[Arm, dict[int, dict]] = {arm: {} for arm in Arm}
    for arm in Arm:
        for seed in SEEDS:
            path = _cell_json_path(arm, seed)
            if os.path.exists(path):
                with open(path) as f:
                    per_arm[arm][seed] = json.load(f)

    def _mean_over_seeds(cells: dict[int, dict], field: str) -> float | None:
        vals = [c[field] for c in cells.values()]
        return float(np.mean(vals)) if vals else None

    def _mean_by_width(cells: dict[int, dict], field: str, w_max: int) -> dict[str, float] | None:
        if not cells:
            return None
        return {str(k): float(np.mean([c[field][str(k)] for c in cells.values()])) for k in range(1, w_max + 1)}

    w_max = W_MAX
    per_arm_heldout_mse = {arm.value: _mean_by_width(per_arm[arm], "heldout_mse_by_width", w_max) for arm in Arm}
    per_arm_train_mse = {arm.value: _mean_by_width(per_arm[arm], "train_mse_by_width", w_max) for arm in Arm}
    per_arm_flops = {arm.value: _mean_by_width(per_arm[arm], "executed_flops_by_width", w_max) for arm in Arm}
    per_arm_cost = {
        arm.value: {
            "train_wall_clock_s": _mean_over_seeds(per_arm[arm], "train_wall_clock_s"),
            "steps_to_converge": _mean_over_seeds(per_arm[arm], "steps_to_converge"),
            "params_allocated": per_arm[arm][next(iter(per_arm[arm]))]["params_allocated"] if per_arm[arm] else None,
            "params_effective": per_arm[arm][next(iter(per_arm[arm]))]["params_effective"] if per_arm[arm] else None,
        }
        for arm in Arm
    }

    # --- Does-it-work: Step 1's equivalence check (self-contained; see `check_prefix_norm_exact` docstring).
    prefix_norm_exact, prefix_norm_max_err = check_prefix_norm_exact()

    # --- Accuracy bar: arm B's per-width held-out MSE within 10% relative of arm A's, at every width.
    mse_a, mse_b, mse_c = per_arm_heldout_mse["a"], per_arm_heldout_mse["b"], per_arm_heldout_mse["c"]
    accuracy_bar = None
    if mse_a is not None and mse_b is not None:
        rel_gap_b_vs_a = {k: abs(mse_b[k] - mse_a[k]) / mse_a[k] for k in mse_a}
        accuracy_bar = {
            "threshold": _ACCURACY_REL_TOL,
            "rel_gap_b_vs_a_by_width": rel_gap_b_vs_a,
            "pass": all(gap <= _ACCURACY_REL_TOL for gap in rel_gap_b_vs_a.values()),
        }

    # --- Cost bar: arm B's wall-clock PER STEP within 1.3x arm A's.
    cost_bar = None
    cost_a, cost_b = per_arm_cost["a"], per_arm_cost["b"]
    if cost_a["train_wall_clock_s"] is not None and cost_a["steps_to_converge"] not in (None, 0) and cost_b["train_wall_clock_s"] is not None and cost_b["steps_to_converge"]:
        per_step_a = cost_a["train_wall_clock_s"] / cost_a["steps_to_converge"]
        per_step_b = cost_b["train_wall_clock_s"] / cost_b["steps_to_converge"]
        ratio = per_step_b / per_step_a if per_step_a else None
        cost_bar = {
            "threshold": _COST_WALLCLOCK_RATIO_BAR,
            "wall_clock_per_step_a": per_step_a,
            "wall_clock_per_step_b": per_step_b,
            "ratio_b_over_a": ratio,
            "pass": ratio is not None and ratio <= _COST_WALLCLOCK_RATIO_BAR,
        }

    # --- Is the per-width affine needed: True iff arm C beats arm B by > 10% relative held-out MSE at ANY width.
    per_width_affine_needed = None
    affine_detail = None
    if mse_b is not None and mse_c is not None:
        rel_gain_c_over_b = {k: (mse_b[k] - mse_c[k]) / mse_b[k] for k in mse_b}
        per_width_affine_needed = any(gain > _AFFINE_NEEDED_REL_GAIN for gain in rel_gain_c_over_b.values())
        affine_detail = {"threshold": _AFFINE_NEEDED_REL_GAIN, "rel_gain_c_over_b_by_width": rel_gain_c_over_b, "prediction_on_record": False}

    frozen = {
        "prefix_norm_exact": prefix_norm_exact,
        "prefix_norm_max_abs_err": prefix_norm_max_err,
        "per_arm_heldout_mse_by_width": per_arm_heldout_mse,
        "per_arm_train_mse_by_width": per_arm_train_mse,
        "per_arm_executed_flops_by_width": per_arm_flops,
        "per_arm_cost": per_arm_cost,
        "accuracy_bar": accuracy_bar,
        "cost_bar": cost_bar,
        "per_width_affine_needed": per_width_affine_needed,
        "affine_detail": affine_detail,
        "config": {
            "arch": ARCH.value,
            "loss": LOSS.value,
            "schedule": SCHEDULE.value,
            "toy": TOY.value,
            "w_max": w_max,
            "n_train": N_TRAIN,
            "n_test": N_TEST,
            "sigma": SIGMA,
            "seeds": list(SEEDS),
            "max_epochs_cap": DEFAULT_MAX_EPOCHS,
            "check_every": DEFAULT_CHECK_EVERY,
            "patience": DEFAULT_PATIENCE,
            "min_delta": DEFAULT_MIN_DELTA,
        },
        "n_cells_present": {arm.value: len(per_arm[arm]) for arm in Arm},
        "provenance": run_provenance(),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "frozen.json")
    with open(path, "w") as f:
        json.dump(_jsonable(frozen), f, indent=2)
    print(f"wrote {path}")
    print(f"prefix_norm_exact={prefix_norm_exact} (max_abs_err={prefix_norm_max_err:.3e})")
    if accuracy_bar is not None:
        print(f"accuracy_bar pass={accuracy_bar['pass']}")
    if cost_bar is not None:
        print(f"cost_bar pass={cost_bar['pass']} ratio_b_over_a={cost_bar['ratio_b_over_a']}")
    print(f"per_width_affine_needed={per_width_affine_needed}")


def main() -> None:
    """Parses args and dispatches to `--selftest` / `--summarize` / one real `--arm`/`--seed` cell."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="Tiny wiring check, then exit.")
    parser.add_argument("--summarize", action="store_true", help="Aggregate every per-cell JSON on disk into WSEL15/frozen.json.")
    parser.add_argument("--arm", type=str, choices=[a.value for a in Arm], default=None, help="Which of the 3 grid arms this cell trains (a=no-norm, b=running-totals, c=+affine).")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for this cell (canonical suite: 0, 1, 2).")
    parser.add_argument(
        "--schedule",
        type=str,
        choices=[nwn.WidthSchedule.SANDWICH.value, nwn.WidthSchedule.ALL.value],
        default=nwn.WidthSchedule.SANDWICH.value,
        help="Training schedule. Default 'sandwich' = the landed grid, byte-identical; 'all' = the WSEL-15 FOLLOW-UP confound check (results go to WSEL15_ALLSCHED/).",
    )
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--check-every", type=int, default=DEFAULT_CHECK_EVERY)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA)
    args = parser.parse_args()
    _apply_schedule(args.schedule)

    if args.selftest:
        sys.exit(0 if run_selftest() else 1)
    if args.summarize:
        summarize()
        return
    if args.arm is None or args.seed is None:
        parser.error("--arm and --seed are both required for a real cell (or pass --selftest / --summarize).")

    arm = Arm(args.arm)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"[wsel15] arm={arm.value} toy={TOY.value} seed={args.seed} w_max={W_MAX} arch={ARCH.value} loss={LOSS.value}", flush=True)
    case, state_dict = run_cell(arm, args.seed, max_epochs=args.max_epochs, check_every=args.check_every, patience=args.patience, min_delta=args.min_delta)

    if not case["all_widths_trustworthy"]:
        print(f"*** DO-NOT-CONCLUDE GUARD: arm={arm.value} seed={args.seed} has widths that did NOT converge trustworthily. Raise --max-epochs. ***")

    cell_path = _cell_json_path(arm, args.seed)
    with open(cell_path, "w") as f:
        json.dump(_jsonable(case), f, indent=2)
    print(f"wrote {cell_path}")

    state_path = _state_path(arm, args.seed)
    torch.save(state_dict, state_path)
    print(f"wrote {state_path}")

    print(
        f"train_wall_clock_s={case['train_wall_clock_s']:.3f}  steps_to_converge={case['steps_to_converge']}  "
        f"params_allocated={case['params_allocated']}  params_effective={case['params_effective']}"
    )


if __name__ == "__main__":
    main()
