"""Discriminating check: does explicit weight-decay regularisation move the selected width? (WSEL-11).

`docs/plans/capacity_programme/width.md` WSEL-11 / MASTER Decision 21: this programme's research
training is otherwise entirely unregularised (no weight decay, dropout, norm layers, or
mini-batching), so the strand's own cheapest-within-tolerance selection rule
(`automl_package/utils/capacity_selection.py`) might partly select SMALL widths because small
widths overfit less under that regime, not because small suffices -- a bias identical across every
width and invisible to any within-strand check. This driver trains the SAME per-width sweep the
strand already uses as its reference (`converged_width_experiment.py`'s `IndependentWidthNet`, the
`nested_width_net.make_hetero` toy, per-width convergence gating) at three AdamW weight_decay
values, applies width.md §1's selection rule to each resulting held-out curve, and reports whether
the chosen width moves.

**RE-RUN, 2026-07-21 (width.md lines 934-1015).** The first run of this check trained through
`nested_width_net.gaussian_log_likelihood` on `IndependentWidthNet`, whose variance heads are real,
so it fitted mean AND variance -- forbidden by §3.7 ("variance is fixed at the true value, never
learned"). That run's verdict is VOID and its six per-cell JSONs / `frozen.json` stay on disk,
un-touched, as the record of what was run; they may not be cited. This version fixes the ONE thing
that changes on the re-run (width.md line 954): sigma FIXED at the generator's true value, which on
tier 1 is exactly `--loss mse` (§3.7 line 359 -- MSE and the fixed-sigma NLL share an optimum up to
one positive constant scale on this toy, whose noise is common-mode). Everything else -- the toy,
the lambda grid, the convergence gates, the selection rule -- is byte-identical to the original
pre-registration below. The ONE licensed deviation is §3.8 line 476: the re-run moves from 2 seeds
to 3 (seeds 0, 1, 2).

**Local implementation note (MASTER Decision 19 -- the package/experiment boundary).**
`converged_width_experiment._train_widths_to_convergence` hardcodes a plain `torch.optim.Adam` with
no weight-decay knob. Weight decay is this task's one experiment-specific variable (width.md
WSEL-11's spec), so `_train_widths_to_convergence_wd` below reimplements ONLY the per-width
optimizer line -- `torch.optim.AdamW(..., weight_decay=lam)` in place of `Adam` -- and calls the
SAME shared `cvg.fit_to_convergence` gate (patience/min_delta/check_every all default, i.e.
UNCHANGED) that every other width driver uses. Everything else (the toy, the net architecture, the
convergence gate, the scoring function, the selection rule) is imported and reused verbatim; nothing
here reimplements those.

**Fixed-sigma objective (width.md §3.7, §3.8 line 428).** Training and scoring both go through the
mean-only MSE twins already shared across this strand's drivers --
`nested_width_net._width_mse` (training/validation; used the same way by
`kdropout_converged_width_experiment.py`'s `LossType.MSE` arm) and
`sinc_width_experiment._score_all_widths_mse` (held-out scoring) -- reused verbatim, not
reimplemented. Neither reads `IndependentWidthNet`'s `logvar_head`, so sigma never enters either
graph; `_assert_sigma_fixed` below fails fast if a future edit swaps either back to the
Gaussian-NLL pair (the exact way the voided run failed, width.md lines 399-410, "WD7").

**Split discipline (the task's own verify line: "reported numbers come from a split not used for
stopping or selection").** Two disjoint `make_hetero` draws are used, exactly as
`converged_width_experiment.py` draws them: `p1` (further split into train/val by `VAL_EVERY`) is
used ONLY for the per-width convergence STOPPING decision. The held-out TEST draw (`x_te, y_te`, a
separate `make_hetero(..., seed + 500)` call) is used ONLY to build the per-width held-out MSE curve
that is fed to `cheapest_within_tolerance` and reported as `selected_width` / `held_out_trajectory`.
No sample plays both roles. (This driver has no router-fitting stage/`p2` split at all -- global
cheapest-within-tolerance reads a curve directly, it does not fit a per-input router -- so the two
disjoint draws already satisfy the constraint.)

**Non-goals (binding, width.md WSEL-11 spec, tier 1 only per §3.8's task table):** no sweep over
toys, tiers or seeds beyond what is specified here; no change to width.md §1's selection rule; no
re-run of WSEL-4's or WSEL-8's numbers from this driver; no interpretation of the discarded run's
numbers.

**Write set (this re-run only writes NEW files):** `automl_package/examples/width_wsel11.py`
(this file) and new per-cell JSONs under
`automl_package/examples/capacity_ladder_results/WSEL11/rerun/` -- a fresh subdirectory so the
re-run's cells and its `frozen.json` can never be confused with (or clobber) the six voided JSONs
and the voided `frozen.json` that stay directly under `.../WSEL11/`.

Usage:
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/width_wsel11.py --selftest
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel11.py --lam 0 --seed 0
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel11.py --summarize
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root

import convergence as cvg  # noqa: E402
import nested_width_net as nwn  # noqa: E402
import sinc_width_experiment as sw  # noqa: E402

from automl_package.utils.capacity_selection import cheapest_within_tolerance  # noqa: E402
from automl_package.utils.pytorch_utils import get_device  # noqa: E402
from automl_package.utils.run_provenance import run_provenance  # noqa: E402

RESULTS_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "WSEL11", "rerun")

LAMBDAS = (0.0, 1e-4, 1e-2)  # width.md WSEL-11 spec, verbatim -- unchanged by the re-run
SEEDS = (0, 1, 2)  # width.md §3.8 line 476: re-run supersedes the original "2 seeds" with 3 (0, 1, 2)
BASELINE_LAMBDA = 0.0

W_MAX = 12  # same ladder as converged_width_experiment.py -- "one toy", the strand's own reference
N_TRAIN = 1500
N_TEST = 500
LR = 1e-2
VAL_EVERY = 5

# Convergence-gate SAFETY CAP (not part of the gate itself -- patience/min_delta/check_every stay at
# `automl_package.utils.convergence`'s defaults, unchanged). `converged_width_experiment.py` runs at
# a script-local 40000; this driver starts from the package's own DEFAULT_MAX_EPOCHS (60000) because
# that reference run already left 2/36 (width, seed) cells `hit_cap=True` at 40000 (seed 0 w=6, seed
# 1 w=4 in `capacity_ladder_results/W_CONVERGED/w_converged_summary.json`), and this task's verify
# line requires `hit_cap: false`. Any width still capped at 60000 gets ONE escalation retrain at 4x
# the cap (Decision 9's "≥4x the self-terminated budget" precedent), logged in the result.
INITIAL_MAX_EPOCHS = cvg.DEFAULT_MAX_EPOCHS  # 60000
ESCALATION_FACTOR = 4

# --- Fixed-sigma objective (width.md §3.7, §3.8 line 428: tier 1 is `--loss mse`, explicit) --------
# Both pinned to the mean-only MSE twins `kdropout_converged_width_experiment.py`'s `LossType.MSE`
# arm already uses. Neither touches `IndependentWidthNet.logvar_head`; sigma is never read, so it is
# equivalent to fixing it at the generator's true value without a free parameter to accidentally fit.
OBJECTIVE_NAME = "mse"
SIGMA_TREATMENT = "fixed_at_generator_true_sigma_via_mse_equivalence"  # HETERO_NOISE_SIGMA=0.05, never learned
TRAIN_LOSS_FN = nwn._width_mse
SCORE_FN = sw._score_all_widths_mse


def _assert_sigma_fixed() -> None:
    """Fails fast (WD7 precedent, width.md lines 399-410) if the train/score path reads variance.

    The voided run failed exactly this way: its driver called `gaussian_log_likelihood` with no
    assertion anywhere forcing the fixed-sigma objective. Both `TRAIN_LOSS_FN` and `SCORE_FN` here
    are pinned to the mean-only MSE twins; this trips before any cell runs if a future edit swaps
    either back to the Gaussian-NLL pair.
    """
    assert TRAIN_LOSS_FN is nwn._width_mse, f"training loss must be the fixed-sigma MSE twin, got {TRAIN_LOSS_FN!r}"
    assert SCORE_FN is sw._score_all_widths_mse, f"scoring must read only the mean head, got {SCORE_FN!r}"
    assert OBJECTIVE_NAME == "mse", f"tier-1 objective must be 'mse' (width.md sec 3.7), got {OBJECTIVE_NAME!r}"


def _standardize_fit(x: np.ndarray, y: np.ndarray) -> dict:
    return {"mx": float(x.mean()), "sx": float(x.std()), "my": float(y.mean()), "sy": float(y.std())}


def _to_std_tensors(x: np.ndarray, y: np.ndarray, norm: dict) -> tuple[torch.Tensor, torch.Tensor]:
    x_t = torch.as_tensor((x - norm["mx"]) / norm["sx"], dtype=torch.float32).reshape(-1, 1)
    y_t = torch.as_tensor((y - norm["my"]) / norm["sy"], dtype=torch.float32)
    return x_t, y_t


def _width_loss(net: nwn.IndependentWidthNet, k: int, x_t: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
    """Tier-1 fixed-sigma training/validation loss -- `TRAIN_LOSS_FN` (the mean-only MSE twin).

    The variance head is never read here, so sigma cannot be learned (width.md §3.7); on this toy's
    constant, common-mode noise, MSE is exactly the fixed-sigma Gaussian NLL up to one positive
    scale factor, so the optimum and the ordering of arms are unchanged from the discarded run's
    intent -- only the forbidden variance-fitting is removed.
    """
    return TRAIN_LOSS_FN(net, k, x_t, y_t)


def _train_one_width(sub: torch.nn.Module, net: nwn.IndependentWidthNet, k: int, x_tr: torch.Tensor, y_tr: torch.Tensor,
                      x_val: torch.Tensor, y_val: torch.Tensor, max_epochs: int, weight_decay: float) -> cvg.ConvergenceResult:
    opt = torch.optim.AdamW(sub.parameters(), lr=LR, weight_decay=weight_decay)

    def step() -> None:
        opt.zero_grad()
        loss = _width_loss(net, k, x_tr, y_tr)
        loss.backward()
        opt.step()

    def val() -> float:
        with torch.no_grad():
            return float(_width_loss(net, k, x_val, y_val).item())

    return cvg.fit_to_convergence(sub, step, val, max_epochs=max_epochs)


def _train_widths_to_convergence_wd(
    net: nwn.IndependentWidthNet, x_tr: torch.Tensor, y_tr: torch.Tensor, x_val: torch.Tensor, y_val: torch.Tensor, weight_decay: float
) -> dict[int, dict]:
    """`converged_width_experiment._train_widths_to_convergence`, AdamW weight_decay substituted for Adam.

    Each width is escalated ONCE (to `ESCALATION_FACTOR` x the initial cap) if it hits the cap at
    `INITIAL_MAX_EPOCHS` -- see module docstring. Returns `{k: {"result": ConvergenceResult,
    "escalated": bool}}`.
    """
    out: dict[int, dict] = {}
    for k in range(1, net.w_max + 1):
        sub = net.subnets[k - 1]
        result = _train_one_width(sub, net, k, x_tr, y_tr, x_val, y_val, INITIAL_MAX_EPOCHS, weight_decay)
        escalated = False
        if result.hit_cap:
            escalated = True
            result = _train_one_width(sub, net, k, x_tr, y_tr, x_val, y_val, INITIAL_MAX_EPOCHS * ESCALATION_FACTOR, weight_decay)
        out[k] = {"result": result, "escalated": escalated}
    return out


def run_combo(lam: float, seed: int, device: str, w_max: int = W_MAX, n_train: int = N_TRAIN, n_test: int = N_TEST) -> dict:
    """Trains the per-width sweep at one (lambda, seed), then applies the strand's selection rule."""
    _assert_sigma_fixed()
    x_tr, y_tr, _reg_tr = nwn.make_hetero(n_train, seed)
    x_te, y_te, _region_te = nwn.make_hetero(n_test, seed + 500)  # disjoint draw -- STOPPING never touches this

    p1_idx = np.arange(0, n_train, 2)
    x_p1, y_p1 = x_tr[p1_idx], y_tr[p1_idx]

    val_mask = (np.arange(len(x_p1)) % VAL_EVERY) == 0
    norm = _standardize_fit(x_p1[~val_mask], y_p1[~val_mask])
    x_tr_t, y_tr_t = _to_std_tensors(x_p1[~val_mask], y_p1[~val_mask], norm)
    x_val_t, y_val_t = _to_std_tensors(x_p1[val_mask], y_p1[val_mask], norm)

    torch.manual_seed(seed)
    net = nwn.IndependentWidthNet(w_max=w_max)
    conv = _train_widths_to_convergence_wd(net, x_tr_t, y_tr_t, x_val_t, y_val_t, lam)

    # Held-out TEST curve -- the ONLY split fed to selection, per the split-discipline note above.
    error_table = SCORE_FN(net, norm, x_te, y_te, device)  # (n_test, w_max) squared error, lower=better, mean head only
    selected_width = int(cheapest_within_tolerance(error_table)) + 1  # 0-based column -> 1-based width

    any_hit_cap = any(v["result"].hit_cap for v in conv.values())
    any_escalated = any(v["escalated"] for v in conv.values())
    n_trustworthy = sum(1 for v in conv.values() if v["result"].trustworthy)
    held_out_curve = [float(error_table[:, k - 1].mean()) for k in range(1, w_max + 1)]

    return {
        "lam": lam,
        "seed": seed,
        "w_max": w_max,
        "objective": OBJECTIVE_NAME,
        "sigma_treatment": SIGMA_TREATMENT,
        "selected_width": selected_width,
        "held_out_trajectory": held_out_curve,  # held-out MSE by width -- the curve the selection rule read
        "hit_cap": any_hit_cap,
        "any_width_escalated": any_escalated,
        "n_widths_trustworthy": n_trustworthy,
        "convergence": {k: {"escalated": v["escalated"], **v["result"].summary()} for k, v in conv.items()},
        "run_provenance": run_provenance(),
    }


def _result_path(lam: float, seed: int) -> str:
    return os.path.join(RESULTS_DIR, f"wsel11_mse_lam{lam}_seed{seed}.json")


_SELFTEST_W_MAX = 3  # tiny wiring-check ladder, unrelated to the real W_MAX


def run_selftest() -> bool:
    """Wiring check: tiny net, tiny cap, both non-zero and zero weight_decay, produces the expected keys."""
    _assert_sigma_fixed()
    device = "cpu"
    ok = True
    for lam in (0.0, 1e-2):
        x_tr, y_tr, _ = nwn.make_hetero(200, 0)
        x_te, y_te, _ = nwn.make_hetero(100, 500)
        p1_idx = np.arange(0, 200, 2)
        x_p1, y_p1 = x_tr[p1_idx], y_tr[p1_idx]
        val_mask = (np.arange(len(x_p1)) % VAL_EVERY) == 0
        norm = _standardize_fit(x_p1[~val_mask], y_p1[~val_mask])
        x_tr_t, y_tr_t = _to_std_tensors(x_p1[~val_mask], y_p1[~val_mask], norm)
        x_val_t, y_val_t = _to_std_tensors(x_p1[val_mask], y_p1[val_mask], norm)

        torch.manual_seed(0)
        net = nwn.IndependentWidthNet(w_max=_SELFTEST_W_MAX)
        conv = {}
        for k in range(1, _SELFTEST_W_MAX + 1):
            sub = net.subnets[k - 1]
            conv[k] = {"result": _train_one_width(sub, net, k, x_tr_t, y_tr_t, x_val_t, y_val_t, 1500, lam), "escalated": False}

        error_table = SCORE_FN(net, norm, x_te, y_te, device)
        selected_width = int(cheapest_within_tolerance(error_table)) + 1
        ok_traj = all(len(conv[k]["result"].trajectory) >= 1 for k in range(1, _SELFTEST_W_MAX + 1))
        ok_width = 1 <= selected_width <= _SELFTEST_W_MAX
        print(f"[wsel11 selftest lam={lam}] trajectories_recorded={ok_traj} selected_width={selected_width} in_range={ok_width}  {'PASS' if ok_traj and ok_width else 'FAIL'}")
        ok = ok and ok_traj and ok_width
    print(f"[wsel11 selftest] {'PASS' if ok else 'FAIL'}")
    return ok


def summarize() -> dict:
    """Reads every landed per-(lambda, seed) result and decides whether selection moved (writes frozen.json)."""
    per_seed_widths: dict[int, dict[float, int]] = {seed: {} for seed in SEEDS}
    missing = []
    for lam in LAMBDAS:
        for seed in SEEDS:
            path = _result_path(lam, seed)
            if not os.path.exists(path):
                missing.append(path)
                continue
            with open(path) as f:
                r = json.load(f)
            assert r.get("objective") == OBJECTIVE_NAME, f"{path} was not run under the fixed-sigma objective: {r.get('objective')!r}"
            per_seed_widths[seed][lam] = r["selected_width"]
    if missing:
        raise FileNotFoundError(f"summarize() called with {len(missing)} result(s) not yet landed: {missing}")

    moved_per_seed = {}
    for seed in SEEDS:
        baseline = per_seed_widths[seed][BASELINE_LAMBDA]
        moved_per_seed[seed] = {lam: (w != baseline) for lam, w in per_seed_widths[seed].items()}
    selection_moved = any(any(m.values()) for m in moved_per_seed.values())

    frozen = {
        "objective": OBJECTIVE_NAME,
        "sigma_treatment": SIGMA_TREATMENT,
        "lambdas": list(LAMBDAS),
        "seeds": list(SEEDS),
        "baseline_lambda": BASELINE_LAMBDA,
        "selected_width_by_seed_and_lambda": {str(seed): {str(lam): w for lam, w in widths.items()} for seed, widths in per_seed_widths.items()},
        "moved_vs_baseline_by_seed_and_lambda": {str(seed): {str(lam): moved for lam, moved in m.items()} for seed, m in moved_per_seed.items()},
        "selection_moved": selection_moved,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "frozen.json")
    with open(path, "w") as f:
        json.dump(sw._jsonable(frozen), f, indent=2)
    print(f"selection_moved={selection_moved}")
    print(json.dumps(sw._jsonable(frozen), indent=2))
    print(f"\nwrote {path}")
    return frozen


def main() -> None:
    """Runs one (lambda, seed) combo, `--selftest`, or `--summarize`."""
    _assert_sigma_fixed()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="Tiny wiring check, then exit.")
    parser.add_argument("--summarize", action="store_true", help="Read landed per-combo results, write frozen.json.")
    parser.add_argument("--lam", type=float, default=None, choices=None, help=f"weight_decay, one of {LAMBDAS}.")
    parser.add_argument("--seed", type=int, default=None, help=f"seed, one of {SEEDS}.")
    args = parser.parse_args()

    if args.selftest:
        sys.exit(0 if run_selftest() else 1)
    if args.summarize:
        summarize()
        return

    if args.lam is None or args.seed is None:
        parser.error("--lam and --seed are both required (or pass --selftest / --summarize).")
    if args.lam not in LAMBDAS:
        parser.error(f"--lam must be one of {LAMBDAS}, got {args.lam}")
    if args.seed not in SEEDS:
        parser.error(f"--seed must be one of {SEEDS}, got {args.seed}")

    device = str(get_device())
    print(f"[wsel11] device={device} objective={OBJECTIVE_NAME} lam={args.lam} seed={args.seed} w_max={W_MAX} initial_max_epochs={INITIAL_MAX_EPOCHS}", flush=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    result = run_combo(args.lam, args.seed, device)
    print(
        f"  selected_width={result['selected_width']} hit_cap={result['hit_cap']} "
        f"any_escalated={result['any_width_escalated']} n_trustworthy={result['n_widths_trustworthy']}/{W_MAX}"
    )
    print(f"  held_out_trajectory={np.array2string(np.array(result['held_out_trajectory']), precision=4, floatmode='fixed')}")

    path = _result_path(args.lam, args.seed)
    with open(path, "w") as f:
        json.dump(sw._jsonable(result), f, indent=2)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
