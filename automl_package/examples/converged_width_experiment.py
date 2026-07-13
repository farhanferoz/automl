"""Convergence-GATED per-input WIDTH dial: each width trains to its OWN convergence, no fixed epochs.

This replaces the fixed-epoch width runs (`independent_width_experiment.py` and the scratchpad probes)
with the standing convergence rule (`convergence.py`; agent-memory
`feedback_check_loss_trajectory_before_concluding`): every width is trained until its HELD-OUT loss
flattens (patience-based early stopping, `convergence.fit_to_convergence`), the full loss trajectory is
recorded, and NO conclusion is drawn from a width whose result is not `trustworthy` (converged, not
capped, not still creeping). Different widths converge at very different rates (w12 in a few thousand
epochs; w6 needs ~20k) so a single global epoch count is inherently wrong — hence PER-WIDTH gating.

Architecture = `IndependentWidthNet` (disjoint per-width weights — the design that removed the
shared-trunk obstruction). Each width k is its own sub-net, trained independently to convergence on a
train/val split carved from the phase-1 half; scoring/selector/bars reuse `sinc_width_experiment`
verbatim on the frozen net, exactly as the other width drivers do.

Usage:
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/converged_width_experiment.py --selftest
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/converged_width_experiment.py --smoke
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/converged_width_experiment.py --config 0
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/converged_width_experiment.py            # all seeds
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

from automl_package.utils.pytorch_utils import get_device  # noqa: E402

RESULTS_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "W_CONVERGED")

SEEDS = (0, 1, 2)
W_MAX = 12
N_TRAIN = 1500
N_TEST = 500
LR = 1e-2
VAL_EVERY = 5  # every 5th phase-1 point is the convergence-monitoring validation split (rest = train)


def _standardize_fit(x: np.ndarray, y: np.ndarray) -> dict:
    return {"mx": float(x.mean()), "sx": float(x.std()), "my": float(y.mean()), "sy": float(y.std())}


def _to_std_tensors(x: np.ndarray, y: np.ndarray, norm: dict) -> tuple[torch.Tensor, torch.Tensor]:
    x_t = torch.as_tensor((x - norm["mx"]) / norm["sx"], dtype=torch.float32).reshape(-1, 1)
    y_t = torch.as_tensor((y - norm["my"]) / norm["sy"], dtype=torch.float32)
    return x_t, y_t


def _width_nll(net: nwn.IndependentWidthNet, k: int, x_t: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
    mean, log_var = net.forward_width(x_t, k)
    return -nwn.gaussian_log_likelihood(mean.squeeze(1), log_var.squeeze(1), y_t).mean()


def _train_widths_to_convergence(
    net: nwn.IndependentWidthNet, x_tr: torch.Tensor, y_tr: torch.Tensor, x_val: torch.Tensor, y_val: torch.Tensor, max_epochs: int
) -> dict[int, cvg.ConvergenceResult]:
    """Trains each width's disjoint sub-net independently until its held-out (val) loss flattens."""
    results: dict[int, cvg.ConvergenceResult] = {}
    for k in range(1, net.w_max + 1):
        sub = net.subnets[k - 1]
        opt = torch.optim.Adam(sub.parameters(), lr=LR)

        def step(k: int = k, opt: torch.optim.Optimizer = opt) -> None:
            opt.zero_grad()
            loss = _width_nll(net, k, x_tr, y_tr)
            loss.backward()
            opt.step()

        def val(k: int = k) -> float:
            with torch.no_grad():
                return float(_width_nll(net, k, x_val, y_val).item())

        results[k] = cvg.fit_to_convergence(sub, step, val, max_epochs=max_epochs)
    return results


def run_case(seed: int, w_max: int, n_train: int, n_test: int, max_epochs: int, device: str) -> dict:
    """Trains every width to convergence, then scores the frozen net for the 3 bars (reused verbatim)."""
    x_tr, y_tr, _reg_tr = nwn.make_hetero(n_train, seed)
    x_te, y_te, region_te = nwn.make_hetero(n_test, seed + 500)

    p1_idx = np.arange(0, n_train, 2)
    p2_idx = np.arange(1, n_train, 2)
    x_p1, y_p1 = x_tr[p1_idx], y_tr[p1_idx]
    x_p2, y_p2 = x_tr[p2_idx], y_tr[p2_idx]

    # Carve a convergence-monitoring VAL split out of p1 (kept out of training); rest = train.
    val_mask = (np.arange(len(x_p1)) % VAL_EVERY) == 0
    norm = _standardize_fit(x_p1[~val_mask], y_p1[~val_mask])
    x_tr_t, y_tr_t = _to_std_tensors(x_p1[~val_mask], y_p1[~val_mask], norm)
    x_val_t, y_val_t = _to_std_tensors(x_p1[val_mask], y_p1[val_mask], norm)

    torch.manual_seed(seed)
    net = nwn.IndependentWidthNet(w_max=w_max)
    conv = _train_widths_to_convergence(net, x_tr_t, y_tr_t, x_val_t, y_val_t, max_epochs)

    # Frozen-net scoring on TEST — the 3 bars, reused verbatim.
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

    n_trustworthy = sum(1 for r in conv.values() if r.trustworthy)
    all_trustworthy = n_trustworthy == w_max

    return {
        "seed": seed,
        "convergence": {k: r.summary() for k, r in conv.items()},
        "n_widths_trustworthy": n_trustworthy,
        "all_widths_trustworthy": all_trustworthy,
        "hard_curve": [float(score_te[region_te == 1][:, k - 1].mean()) for k in range(1, w_max + 1)],
        "easy_curve": [float(score_te[region_te == 0][:, k - 1].mean()) for k in range(1, w_max + 1)],
        "construction": construction,
        "recovery": recovery,
        "deploy": deploy,
        "marginal_p": marginal_p.tolist(),
        "per_k_nll": per_k_nll,
    }


def _print_case(case: dict, w_max: int) -> None:
    print("  per-width convergence (val NLL, lower=better):")
    for k in range(1, w_max + 1):
        s = case["convergence"][k]
        flag = "OK " if s["trustworthy"] else ("CAP" if s["hit_cap"] else "crp")
        print(f"    w{k:2d} [{flag}] stop@{s['stop_epoch']:6d} best_val={s['best_val']:.4f} recent_impr={s['recent_improvement']:.4f}")
    print(f"  ALL WIDTHS CONVERGED (trustworthy): {case['all_widths_trustworthy']}  ({case['n_widths_trustworthy']}/{w_max})")
    print(f"  hard LL by width: {np.array2string(np.array(case['hard_curve']), precision=3, floatmode='fixed')}")
    print(f"  easy LL by width: {np.array2string(np.array(case['easy_curve']), precision=3, floatmode='fixed')}")
    c, r, d = case["construction"], case["recovery"], case["deploy"]
    print(f"  construction: hard_climbs={c['centre_climbs']} hard_near_floor={c['centre_near_noise_floor']} easy_flat={c['tail_flat']} pass={c['construction_pass']}")
    print(f"  recovery: hard_width={r['mean_expected_width_centre']:.3f} easy_width={r['mean_expected_width_tail']:.3f} sep_pass={r['separation_beats_2se']}")
    print(
        f"  deploy: selector_nll={d['selector_nll']:.4f} best_global={d['best_global_nll']:.4f} "
        f"oracle={d['oracle_nll']:.4f} beats_global={d['matches_or_beats_global']} pass={d['deploy_pass']}",
        flush=True,
    )


def run_selftest() -> bool:
    """Wiring check: tiny net trains each width for a tiny cap, produces trajectories + a trustworthy flag."""
    device = "cpu"
    case = run_case(seed=0, w_max=3, n_train=200, n_test=100, max_epochs=1500, device=device)
    conv = case["convergence"]
    ok_traj = all(len(conv[k]["trajectory"]) >= 1 for k in range(1, 4))
    ok_flags = all(isinstance(conv[k]["converged"], bool) for k in range(1, 4))
    ok = ok_traj and ok_flags and isinstance(case["all_widths_trustworthy"], bool)
    print(f"[converged-width selftest] trajectories_recorded={ok_traj} flags_present={ok_flags} all_trustworthy_bool=ok  {'PASS' if ok else 'FAIL'}")
    print(f"  (per-width stop epochs: {[conv[k]['stop_epoch'] for k in range(1, 4)]})")
    return ok


def main() -> None:
    """Runs the convergence-gated width battery (or `--smoke` / `--selftest`)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="Tiny wiring check, then exit.")
    parser.add_argument("--smoke", action="store_true", help="Tiny config (w_max=4, small cap, 1 seed); no save.")
    parser.add_argument("--config", choices=[str(s) for s in SEEDS], default=None, help="One seed only; default = all.")
    parser.add_argument("--max-epochs", type=int, default=40000, help="Safety cap per width (convergence decides the real stop).")
    args = parser.parse_args()

    if args.selftest:
        sys.exit(0 if run_selftest() else 1)

    device = str(get_device())
    if args.smoke:
        seeds, w_max, n_train, n_test, max_epochs, save = [0], 4, N_TRAIN, N_TEST, 4000, False
    else:
        seeds = list(SEEDS) if args.config is None else [int(args.config)]
        w_max, n_train, n_test, max_epochs, save = W_MAX, N_TRAIN, N_TEST, args.max_epochs, True

    print(f"[converged-width] device={device} seeds={seeds} w_max={w_max} max_epochs_cap={max_epochs}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    per_case = []
    for seed in seeds:
        print(f"=== W_CONVERGED seed={seed} ===", flush=True)
        case = run_case(seed, w_max, n_train, n_test, max_epochs, device)
        per_case.append(case)
        _print_case(case, w_max)

    any_untrustworthy = [c["seed"] for c in per_case if not c["all_widths_trustworthy"]]
    if any_untrustworthy:
        print(f"\n*** DO-NOT-CONCLUDE GUARD: seeds {any_untrustworthy} have widths that did NOT converge (hit cap / still creeping). ***")
        print("*** Raise --max-epochs for those before drawing any conclusion from their curves. ***")
    else:
        print("\nAll widths converged on all seeds — curves are trustworthy to interpret.")

    if save:
        path = os.path.join(RESULTS_DIR, "w_converged_summary.json")
        summary = {
            "config": {"w_max": w_max, "seeds": seeds, "max_epochs_cap": max_epochs, "lr": LR, "val_every": VAL_EVERY},
            "per_case": per_case,
            "untrustworthy_seeds": any_untrustworthy,
        }
        with open(path, "w") as f:
            json.dump(sw._jsonable(summary), f, indent=2)
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
