"""Cascade vs Matryoshka: the width-dial arm-of-record decision (both trained to convergence).

(`docs/plans/width_dial_2026-07-11/cascade_execution_plan_2026-07-11.md` §4.3; pre-registration
`capacity_ladder_results/W_CASCADE/PREREGISTRATION.md`)

Runs BOTH arms on the identical data/split/seeds as `converged_width_experiment.py` (reused
verbatim via `import converged_width_experiment as cwe`, mirroring
`kdropout_converged_width_experiment.py`'s own reuse pattern) so every summary is directly
comparable:

- `--arm cascade`: `cascade_width_net.ResidualCascadeNet` + `cascade_width_net.train_cascade` --
  frozen residual blocks grown one at a time, guaranteed non-increasing held-out NLL by
  construction (plan §2.3 Lemma 2).
- `--arm matryoshka`: `matryoshka_width_net.MatryoshkaWidthNet` + `matryoshka_width_net.
  train_matryoshka` -- shared trunk, per-rung dedicated heads, jointly trained (the fallback arm,
  plan §2.7).
- default (no `--arm`): both, sequentially, one summary JSON each.

Frozen-net scoring, the three bars (CONSTRUCTION/RECOVERY/DEPLOY), and the selector are
`sinc_width_experiment` reused VERBATIM -- both arms expose the same `w_max`/`forward_width`/
`all_widths_forward` interface as `nested_width_net.NestedWidthNet`, so
`sw._score_all_widths`/`_construction_bar`/`_fit_selector`/`_selector_eval`/`_recovery_bar`/
`_deploy_bar`/`_jsonable` are drop-in, identical call shapes to `converged_width_experiment.run_case`.

Beyond `converged_width_experiment`'s case fields, this driver additionally records (plan §4.3
step 4): `accepted_rungs` (cascade only -- which rungs the acceptance rule kept vs reset to inert);
`delta_ll_per_rung` (held-out TEST delta-LL, rung k vs rung k-1, overall + per region, rung 0 =
the standardized N(0,1) marginal converted to original-y-scale via the same exact affine Jacobian
`sw._score_all_widths` uses); `anchor` (this arm's rung-12 test NLL minus the dedicated w12 test
NLL from `capacity_ladder_results/W_CONVERGED/w_converged_summary.json`, same seed -- see
`_load_anchor_nll`'s docstring for the verified on-disk location, which is PER-SEED/NESTED, not a
top-level key as an earlier draft of the plan assumed); `beta_nll_used`.

Usage:
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/cascade_width_experiment.py --selftest
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/cascade_width_experiment.py --smoke
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/cascade_width_experiment.py --arm cascade --config 0
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/cascade_width_experiment.py            # both arms, all seeds
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root

import cascade_width_net as cwn  # noqa: E402
import converged_width_experiment as cwe  # noqa: E402
import convergence as cvg  # noqa: E402
import matryoshka_width_net as mwn  # noqa: E402
import nested_width_net as nwn  # noqa: E402
import sinc_width_experiment as sw  # noqa: E402

from automl_package.utils.pytorch_utils import get_device  # noqa: E402

RESULTS_DIR_CASCADE = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "W_CASCADE")
RESULTS_DIR_MATRYOSHKA = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "W_MRL")
_ANCHOR_PATH = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "W_CONVERGED", "w_converged_summary.json")
_ANCHOR_K = 12  # plan §4.3: the anchor is always read at rung/width 12, regardless of this run's w_max.

# Regime reused verbatim from converged_width_experiment.py so every summary is directly comparable.
SEEDS = cwe.SEEDS
W_MAX = cwe.W_MAX
N_TRAIN = cwe.N_TRAIN
N_TEST = cwe.N_TEST
LR = cwe.LR
VAL_EVERY = cwe.VAL_EVERY

ARMS = ("cascade", "matryoshka")
DEFAULT_MAX_EPOCHS = 40000  # safety cap; convergence decides the real stop (both arms train every rung every step).
DEFAULT_RESTARTS = cwn.RESTARTS_DEFAULT
_SELFTEST_W_MAX = 3  # tiny end-to-end wiring config (plan §4.3 driver selftest).


def _rung0_ll(y: np.ndarray, norm: dict) -> np.ndarray:
    """Rung-0 baseline LL: the standardized N(0,1) marginal, converted to ORIGINAL y-scale.

    Same exact affine convention `sw._score_all_widths` uses (`Y = sy*Y_n + my` =>
    `logvar_orig = logvar_n + 2*log(sy)`), applied to the fixed `(mean_n, logvar_n) = (0, 0)` that
    rung 0 always is (plan §2.1) -- so `delta_ll_per_rung[1]` measures rung 1's gain over the SAME
    baseline every other width driver's `per_k_nll` is implicitly relative to.
    """
    mean_orig = norm["my"]
    logvar_orig = 2.0 * math.log(norm["sy"])
    y_t = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)
    mean_t = torch.full_like(y_t, mean_orig)
    logvar_t = torch.full_like(y_t, logvar_orig)
    ll = nwn.gaussian_log_likelihood(mean_t, logvar_t, y_t)
    return ll.squeeze(1).cpu().numpy().astype(np.float64)


def _delta_ll_per_rung(fixed_width_ll: dict[int, np.ndarray], rung0_ll: np.ndarray, region_labels: np.ndarray, w_max: int) -> dict[int, dict]:
    """Held-out TEST delta-LL per rung (rung k vs rung k-1; rung 0 = `_rung0_ll`), overall + by region."""
    mask_easy = region_labels == 0
    mask_hard = region_labels == 1
    out: dict[int, dict] = {}
    prev_ll = rung0_ll
    for k in range(1, w_max + 1):
        cur_ll = fixed_width_ll[k]
        delta = cur_ll - prev_ll
        out[k] = {"overall": float(delta.mean()), "easy": float(delta[mask_easy].mean()), "hard": float(delta[mask_hard].mean())}
        prev_ll = cur_ll
    return out


def _load_anchor_nll(seed: int) -> float | None:
    """Dedicated-net w12 test NLL for `seed`, from `W_CONVERGED/w_converged_summary.json`.

    CORRECTION vs an earlier draft of the plan: `per_k_nll` is NOT a top-level key of the summary --
    it is PER-SEED, nested inside `per_case` (a list of case dicts, each with its own `seed` and
    `per_k_nll`). Verified on disk (`json.load(...)['per_case'][i]['per_k_nll']['12']`). If the file
    is missing, OR no case in `per_case` matches `seed`, OR that case's `per_k_nll` lacks the "12"
    key -- return None and print a warning; NEVER fail the run (graceful degradation, plan §4.3).
    """
    if not os.path.exists(_ANCHOR_PATH):
        print(f"[cascade-width WARNING] anchor file not found: {_ANCHOR_PATH} -- anchor=null")
        return None
    with open(_ANCHOR_PATH) as f:
        summary = json.load(f)
    for case in summary.get("per_case", []):
        if case.get("seed") == seed:
            val = case.get("per_k_nll", {}).get(str(_ANCHOR_K))
            if val is None:
                print(f"[cascade-width WARNING] anchor 'per_k_nll[{_ANCHOR_K}]' missing for seed={seed} in {_ANCHOR_PATH} -- anchor=null")
                return None
            return float(val)
    print(f"[cascade-width WARNING] no case with seed={seed} in {_ANCHOR_PATH}'s per_case -- anchor=null")
    return None


def _result_of(conv: dict, arm: str, k: int) -> cvg.ConvergenceResult:
    """Unwraps the per-rung `ConvergenceResult` regardless of arm (cascade nests it under `"conv"`)."""
    return conv[k]["conv"] if arm == "cascade" else conv[k]


def run_case(
    seed: int,
    arm: str,
    w_max: int,
    n_train: int,
    n_test: int,
    max_epochs: int,
    device: str,
    *,
    check_every: int,
    patience: int,
    min_delta: float,
    restarts: int,
    beta_nll: bool,
) -> dict:
    """Trains one arm to (rung-)convergence, then scores the frozen net for the 3 bars (reused verbatim)."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm!r}")

    x_tr, y_tr, _reg_tr = nwn.make_hetero(n_train, seed)
    x_te, y_te, region_te = nwn.make_hetero(n_test, seed + 500)

    p1_idx = np.arange(0, n_train, 2)
    p2_idx = np.arange(1, n_train, 2)
    x_p1, y_p1 = x_tr[p1_idx], y_tr[p1_idx]
    x_p2, y_p2 = x_tr[p2_idx], y_tr[p2_idx]

    # Same phase-1 train/val carve as converged_width_experiment.py (rest = train, every 5th = val).
    val_mask = (np.arange(len(x_p1)) % VAL_EVERY) == 0
    norm = cwe._standardize_fit(x_p1[~val_mask], y_p1[~val_mask])
    x_tr_t, y_tr_t = cwe._to_std_tensors(x_p1[~val_mask], y_p1[~val_mask], norm)
    x_val_t, y_val_t = cwe._to_std_tensors(x_p1[val_mask], y_p1[val_mask], norm)

    torch.manual_seed(seed)
    accepted_rungs: list[bool] | None = None
    beta_nll_used = False
    if arm == "cascade":
        net = cwn.ResidualCascadeNet(w_max=w_max)
        conv = cwn.train_cascade(
            net, x_tr_t, y_tr_t, x_val_t, y_val_t,
            seed=seed, restarts=restarts, max_epochs=max_epochs, check_every=check_every, patience=patience, min_delta=min_delta, lr=LR, beta_nll=beta_nll,
        )
        accepted_rungs = [conv[k]["accepted"] for k in range(1, w_max + 1)]
        beta_nll_used = beta_nll
        conv_summary = {}
        for k in range(1, w_max + 1):
            s = conv[k]["conv"].summary()
            s["accepted"] = conv[k]["accepted"]
            s["val_nll_prev"] = conv[k]["val_nll_prev"]
            conv_summary[k] = s
    else:
        net = mwn.MatryoshkaWidthNet(w_max=w_max)
        conv = mwn.train_matryoshka(net, x_tr_t, y_tr_t, x_val_t, y_val_t, max_epochs=max_epochs, check_every=check_every, patience=patience, min_delta=min_delta, lr=LR)
        conv_summary = {k: conv[k].summary() for k in range(1, w_max + 1)}

    # Frozen-net scoring on TEST -- the 3 bars, reused verbatim from sinc_width_experiment.
    score_te = sw._score_all_widths(net, norm, x_te, y_te, device)
    fixed_width_ll = {k: score_te[:, k - 1] for k in range(1, w_max + 1)}
    construction = sw._construction_bar(fixed_width_ll, region_te, k_lo=1, k_mid=max(2, w_max // 2), w_max=w_max)

    score_p2 = sw._score_all_widths(net, norm, x_p2, y_p2, device)
    router = sw._fit_selector(score_p2, x_p2, w_max, seed, device)
    selector_nll, marginal_p, expected_width = sw._selector_eval(router, score_te, x_te, device)
    recovery = sw._recovery_bar(expected_width, region_te)
    per_k_nll = {k: float(-fixed_width_ll[k].mean()) for k in range(1, w_max + 1)}
    oracle_nll = float(-score_te.max(axis=1).mean())
    deploy = sw._deploy_bar(selector_nll, per_k_nll, oracle_nll)

    n_trustworthy = sum(1 for k in range(1, w_max + 1) if _result_of(conv, arm, k).trustworthy)
    all_trustworthy = n_trustworthy == w_max

    rung0_ll = _rung0_ll(y_te, norm)
    delta_ll_per_rung = _delta_ll_per_rung(fixed_width_ll, rung0_ll, region_te, w_max)

    anchor = None
    if w_max >= _ANCHOR_K:
        anchor_dedicated = _load_anchor_nll(seed)
        if anchor_dedicated is not None:
            anchor = per_k_nll[_ANCHOR_K] - anchor_dedicated

    return {
        "seed": seed,
        "arm": arm,
        "convergence": conv_summary,
        "n_widths_trustworthy": n_trustworthy,
        "all_widths_trustworthy": all_trustworthy,
        "hard_curve": [float(score_te[region_te == 1][:, k - 1].mean()) for k in range(1, w_max + 1)],
        "easy_curve": [float(score_te[region_te == 0][:, k - 1].mean()) for k in range(1, w_max + 1)],
        "construction": construction,
        "recovery": recovery,
        "deploy": deploy,
        "marginal_p": marginal_p.tolist(),
        "per_k_nll": per_k_nll,
        "accepted_rungs": accepted_rungs,
        "n_inert_rungs": (sum(1 for a in accepted_rungs if not a) if accepted_rungs is not None else None),
        "delta_ll_per_rung": delta_ll_per_rung,
        "anchor": anchor,
        "beta_nll_used": beta_nll_used,
    }


def _print_case(case: dict, w_max: int) -> None:
    print(f"  arm={case['arm']}")
    print("  per-rung convergence (val NLL, lower=better):")
    for k in range(1, w_max + 1):
        s = case["convergence"][k]
        flag = "OK " if s["trustworthy"] else ("CAP" if s["hit_cap"] else "crp")
        extra = f" accepted={s['accepted']}" if "accepted" in s else ""
        print(f"    k{k:2d} [{flag}] stop@{s['stop_epoch']:6d} best_val={s['best_val']:.4f} recent_impr={s['recent_improvement']:.4f}{extra}")
    print(f"  ALL RUNGS CONVERGED (trustworthy): {case['all_widths_trustworthy']}  ({case['n_widths_trustworthy']}/{w_max})")
    if case["accepted_rungs"] is not None:
        print(f"  accepted_rungs={case['accepted_rungs']}  n_inert={case['n_inert_rungs']}")
    print(f"  hard LL by rung: {np.array2string(np.array(case['hard_curve']), precision=3, floatmode='fixed')}")
    print(f"  easy LL by rung: {np.array2string(np.array(case['easy_curve']), precision=3, floatmode='fixed')}")
    c, r, d = case["construction"], case["recovery"], case["deploy"]
    print(f"  construction: hard_climbs={c['centre_climbs']} hard_near_floor={c['centre_near_noise_floor']} easy_flat={c['tail_flat']} pass={c['construction_pass']}")
    print(f"  recovery: hard_width={r['mean_expected_width_centre']:.3f} easy_width={r['mean_expected_width_tail']:.3f} sep_pass={r['separation_beats_2se']}")
    print(
        f"  deploy: selector_nll={d['selector_nll']:.4f} best_global={d['best_global_nll']:.4f} "
        f"oracle={d['oracle_nll']:.4f} beats_global={d['matches_or_beats_global']} pass={d['deploy_pass']}"
    )
    anchor = case["anchor"]
    anchor_str = f"{anchor:+.4f}" if anchor is not None else "null"
    print(f"  anchor (rung{_ANCHOR_K} NLL - dedicated w{_ANCHOR_K} NLL): {anchor_str}  beta_nll_used={case['beta_nll_used']}", flush=True)


def run_selftest() -> bool:
    """Wiring check per arm: tiny net trains to a tiny cap, produces trajectories + the extra §4.3 fields."""
    ok = True
    for arm in ARMS:
        print(f"[cascade-width selftest] arm={arm}")
        case = run_case(
            seed=0, arm=arm, w_max=_SELFTEST_W_MAX, n_train=200, n_test=100, max_epochs=1500, device="cpu",
            check_every=100, patience=3, min_delta=2e-3, restarts=2, beta_nll=False,
        )
        conv = case["convergence"]
        ok_traj = all(len(conv[k]["trajectory"]) >= 1 for k in range(1, _SELFTEST_W_MAX + 1))
        ok_flags = all(isinstance(conv[k]["converged"], bool) for k in range(1, _SELFTEST_W_MAX + 1))
        ok_accepted = (case["accepted_rungs"] is not None) if arm == "cascade" else True
        ok_bars = all(key in case for key in ("construction", "recovery", "deploy"))
        ok_delta = len(case["delta_ll_per_rung"]) == _SELFTEST_W_MAX
        arm_ok = ok_traj and ok_flags and ok_accepted and ok_bars and ok_delta
        print(
            f"  trajectories_recorded={ok_traj} flags_present={ok_flags} accepted_rungs_present={ok_accepted} "
            f"bars_present={ok_bars} delta_ll_entries={len(case['delta_ll_per_rung'])}  {'PASS' if arm_ok else 'FAIL'}"
        )
        ok = ok and arm_ok
    print(f"[cascade-width selftest] {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    """Runs the cascade/matryoshka width battery (or `--smoke` / `--selftest`)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="Tiny wiring check per arm, then exit.")
    parser.add_argument("--smoke", action="store_true", help="Tiny config (w_max=4, small cap, 1 seed, both arms); no save.")
    parser.add_argument("--arm", choices=ARMS, default=None, help="Run only this arm; default = both, sequentially.")
    parser.add_argument("--config", choices=[str(s) for s in SEEDS], default=None, help="One seed only; default = all.")
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS, help="Safety cap (convergence decides the real stop).")
    parser.add_argument("--check-every", type=int, default=cvg.DEFAULT_CHECK_EVERY, help="Epochs between held-out checkpoints.")
    parser.add_argument("--patience", type=int, default=cvg.DEFAULT_PATIENCE, help="Flat checkpoints that declare convergence.")
    parser.add_argument("--min-delta", type=float, default=cvg.DEFAULT_MIN_DELTA, help="Held-out-loss decrease (nats) counted as improvement.")
    parser.add_argument("--restarts", type=int, default=DEFAULT_RESTARTS, help="Cascade-arm random restarts per stage (ignored by matryoshka).")
    parser.add_argument("--beta-nll", action="store_true", help="Cascade-arm: beta-NLL (beta=0.5) stage training loss (plan §2.5 escalation).")
    args = parser.parse_args()

    if args.selftest:
        sys.exit(0 if run_selftest() else 1)

    device = str(get_device())
    arms = list(ARMS) if args.arm is None else [args.arm]

    if args.smoke:
        seeds, w_max, n_train, n_test, max_epochs, save = [0], 4, N_TRAIN, N_TEST, 4000, False
        check_every, patience, min_delta, restarts = 200, cvg.DEFAULT_PATIENCE, cvg.DEFAULT_MIN_DELTA, 2
    else:
        seeds = list(SEEDS) if args.config is None else [int(args.config)]
        w_max, n_train, n_test, max_epochs, save = W_MAX, N_TRAIN, N_TEST, args.max_epochs, True
        check_every, patience, min_delta, restarts = args.check_every, args.patience, args.min_delta, args.restarts

    print(f"[cascade-width] device={device} arms={arms} seeds={seeds} w_max={w_max} max_epochs_cap={max_epochs} beta_nll={args.beta_nll}")

    for arm in arms:
        results_dir = RESULTS_DIR_CASCADE if arm == "cascade" else RESULTS_DIR_MATRYOSHKA
        os.makedirs(results_dir, exist_ok=True)

        per_case = []
        for seed in seeds:
            print(f"=== {arm.upper()} seed={seed} ===", flush=True)
            case = run_case(
                seed, arm, w_max, n_train, n_test, max_epochs, device,
                check_every=check_every, patience=patience, min_delta=min_delta, restarts=restarts, beta_nll=args.beta_nll,
            )
            per_case.append(case)
            _print_case(case, w_max)

        any_untrustworthy = [c["seed"] for c in per_case if not c["all_widths_trustworthy"]]
        if any_untrustworthy:
            print(f"\n*** DO-NOT-CONCLUDE GUARD ({arm}): seeds {any_untrustworthy} have rungs that did NOT converge (hit cap / still creeping). ***")
            print("*** Raise --max-epochs for those before drawing any conclusion from their curves. ***")
        else:
            print(f"\nAll rungs converged on all seeds ({arm}) -- curves are trustworthy to interpret.")

        if save:
            path = os.path.join(results_dir, f"w_{'cascade' if arm == 'cascade' else 'mrl'}_summary.json")
            summary = {
                "config": {
                    "arm": arm,
                    "w_max": w_max,
                    "seeds": seeds,
                    "max_epochs_cap": max_epochs,
                    "lr": LR,
                    "val_every": VAL_EVERY,
                    "check_every": check_every,
                    "patience": patience,
                    "min_delta": min_delta,
                    "restarts": restarts if arm == "cascade" else None,
                    "beta_nll": args.beta_nll if arm == "cascade" else None,
                    "best_restore": "joint_sum" if arm == "matryoshka" else None,
                },
                "per_case": per_case,
                "untrustworthy_seeds": any_untrustworthy,
            }
            with open(path, "w") as f:
                json.dump(sw._jsonable(summary), f, indent=2)
            print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
