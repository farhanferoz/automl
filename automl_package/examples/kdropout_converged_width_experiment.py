"""Convergence-gated per-input WIDTH dial trained with K-DROPOUT (not per-width separate training).

`converged_width_experiment.py` trained each width as its OWN separate network (its own optimizer, its
own convergence) — a clean upper bound, but NOT the deployable model: it costs one training run PER
width. The model the user actually proposed is ONE joint training pass with k-dropout — every step
trains a sampled subset of widths (the SANDWICH schedule: always width=1 and width=w_max, plus 2 random
middle widths) with a single continuous optimizer. This driver runs exactly that, but gated by the same
per-width convergence rule (`convergence.py`; agent-memory
`feedback_check_loss_trajectory_before_concluding`): training continues until EVERY width's held-out
loss has flattened (per-width `ConvergenceTracker`) or the safety cap is hit, each width keeps its OWN
best-not-last weights, and NO conclusion is drawn from a width whose result is not `trustworthy`.

Why this is not subsumed by the separate-training battery: with independent weights the two schemes
reach the same networks at convergence, but (a) the whole efficiency case for k-dropout is ONE pass vs
twelve, and (b) under sandwich each middle width is trained only ~1/5 of the steps AND a single shared
Adam optimizer applies its bias-correction bookkeeping globally, so k-dropout can converge differently
(and slower on the mids) than dedicated training — which is exactly what the per-width gate measures.

Architecture / data / split / scoring / bars are `IndependentWidthNet` +
`converged_width_experiment` + `sinc_width_experiment` reused verbatim, so the summary is directly
comparable to `capacity_ladder_results/W_CONVERGED/w_converged_summary.json`. The ONLY thing that
changes is the training scheme (k-dropout sandwich vs per-width separate).

Usage:
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/kdropout_converged_width_experiment.py --selftest
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/kdropout_converged_width_experiment.py --smoke
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/kdropout_converged_width_experiment.py --config 0
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/kdropout_converged_width_experiment.py            # all seeds
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

import converged_width_experiment as cwe  # noqa: E402
import convergence as cvg  # noqa: E402
import nested_width_net as nwn  # noqa: E402
import sinc_width_experiment as sw  # noqa: E402

from automl_package.utils.pytorch_utils import get_device  # noqa: E402

RESULTS_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "W_KDROPOUT_CONVERGED")

# Regime reused verbatim from the separate-training battery so the two summaries are directly comparable.
SEEDS = cwe.SEEDS
W_MAX = cwe.W_MAX
N_TRAIN = cwe.N_TRAIN
N_TEST = cwe.N_TEST
# Mids are trained only ~1/5 of steps under sandwich, so the honest per-width cap is well above the
# separate-training battery's 40k (where each width trained every step); convergence still decides the
# real stop, the cap is only the safety net.
DEFAULT_MAX_EPOCHS = 200000


def _train_kdropout_to_convergence(
    net: nwn.IndependentWidthNet,
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    max_epochs: int,
    check_every: int,
    patience: int,
    min_delta: float,
    seed: int,
) -> dict[int, cvg.ConvergenceResult]:
    """One joint SANDWICH k-dropout run, gated by PER-WIDTH held-out convergence.

    A single continuous Adam optimizer over the whole net (the sandwich step is replicated inline from
    `nested_width_net.train_nested_width` rather than called, because chunking that function would rebuild
    the optimizer each chunk and reset Adam's momentum — unfaithful to one continuous run). Every
    `check_every` epochs each width's held-out loss is checkpointed into its own `ConvergenceTracker`;
    the loop stops when ALL widths have flattened (or the cap is hit). Each width's own best-not-last
    weights are restored at the end (subnets are disjoint, so restoring one never disturbs another).

    Args:
        net: the `IndependentWidthNet` to train in place.
        x_tr: standardized training inputs, shape `(N, 1)`.
        y_tr: standardized training targets, shape `(N,)`.
        x_val: standardized held-out inputs used only for convergence monitoring.
        y_val: standardized held-out targets used only for convergence monitoring.
        max_epochs: safety cap on optimizer steps (== epochs; full-batch).
        check_every: epochs between per-width held-out checkpoints.
        patience: consecutive flat checkpoints that declare one width converged.
        min_delta: held-out-loss decrease (nats) counted as a real improvement.
        seed: RNG seed for the per-step middle-width draw.

    Returns:
        `{width -> ConvergenceResult}`, best weights already restored per width.
    """
    w_max = net.w_max
    opt = torch.optim.Adam(net.parameters(), lr=cwe.LR)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    mid_candidates = list(range(2, w_max))  # {2..w_max-1}
    n_mid_draw = min(2, len(mid_candidates))

    trackers = {k: cvg.ConvergenceTracker(patience=patience, min_delta=min_delta) for k in range(1, w_max + 1)}
    best_states: dict[int, dict | None] = dict.fromkeys(range(1, w_max + 1))

    net.train()
    final_epoch = max_epochs
    for epoch in range(1, max_epochs + 1):
        opt.zero_grad()
        widths = [1, w_max]
        if n_mid_draw:
            perm = torch.randperm(len(mid_candidates), generator=gen)[:n_mid_draw]
            widths += [mid_candidates[i] for i in perm.tolist()]
        total_loss = torch.zeros(())
        for k in widths:
            total_loss = total_loss + cwe._width_nll(net, k, x_tr, y_tr)
        total_loss.backward()
        opt.step()

        if epoch % check_every == 0:
            net.eval()
            with torch.no_grad():
                for k in range(1, w_max + 1):
                    v = float(cwe._width_nll(net, k, x_val, y_val).item())
                    if trackers[k].update(epoch, v):
                        best_states[k] = {n: t.detach().clone() for n, t in net.subnets[k - 1].state_dict().items()}
            net.train()
            if all(t.done for t in trackers.values()):
                final_epoch = epoch
                break

    for k in range(1, w_max + 1):
        if best_states[k] is not None:
            net.subnets[k - 1].load_state_dict(best_states[k])
    net.eval()
    return {k: trackers[k].result(final_epoch=final_epoch) for k in range(1, w_max + 1)}


def run_case(
    seed: int, w_max: int, n_train: int, n_test: int, max_epochs: int, device: str, *, check_every: int, patience: int, min_delta: float
) -> dict:
    """Trains the k-dropout net to per-width convergence, then scores the frozen net for the 3 bars (reused verbatim)."""
    x_tr, y_tr, _reg_tr = nwn.make_hetero(n_train, seed)
    x_te, y_te, region_te = nwn.make_hetero(n_test, seed + 500)

    p1_idx = np.arange(0, n_train, 2)
    p2_idx = np.arange(1, n_train, 2)
    x_p1, y_p1 = x_tr[p1_idx], y_tr[p1_idx]
    x_p2, y_p2 = x_tr[p2_idx], y_tr[p2_idx]

    # Same phase-1 train/val carve as the separate-training battery (rest = train, every 5th = val).
    val_mask = (np.arange(len(x_p1)) % cwe.VAL_EVERY) == 0
    norm = cwe._standardize_fit(x_p1[~val_mask], y_p1[~val_mask])
    x_tr_t, y_tr_t = cwe._to_std_tensors(x_p1[~val_mask], y_p1[~val_mask], norm)
    x_val_t, y_val_t = cwe._to_std_tensors(x_p1[val_mask], y_p1[val_mask], norm)

    torch.manual_seed(seed)
    net = nwn.IndependentWidthNet(w_max=w_max)
    conv = _train_kdropout_to_convergence(
        net, x_tr_t, y_tr_t, x_val_t, y_val_t, max_epochs=max_epochs, check_every=check_every, patience=patience, min_delta=min_delta, seed=seed
    )

    # Frozen-net scoring on TEST — the 3 bars, reused verbatim from sinc_width_experiment.
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
    return {
        "seed": seed,
        "convergence": {k: r.summary() for k, r in conv.items()},
        "n_widths_trustworthy": n_trustworthy,
        "all_widths_trustworthy": n_trustworthy == w_max,
        "hard_curve": [float(score_te[region_te == 1][:, k - 1].mean()) for k in range(1, w_max + 1)],
        "easy_curve": [float(score_te[region_te == 0][:, k - 1].mean()) for k in range(1, w_max + 1)],
        "construction": construction,
        "recovery": recovery,
        "deploy": deploy,
        "marginal_p": marginal_p.tolist(),
        "per_k_nll": per_k_nll,
    }


def run_selftest() -> bool:
    """Wiring check: tiny k-dropout net trains to a tiny cap, produces per-width trajectories + flags."""
    case = run_case(seed=0, w_max=3, n_train=200, n_test=100, max_epochs=1500, device="cpu", check_every=100, patience=3, min_delta=2e-3)
    conv = case["convergence"]
    ok_traj = all(len(conv[k]["trajectory"]) >= 1 for k in range(1, 4))
    ok_flags = all(isinstance(conv[k]["converged"], bool) for k in range(1, 4))
    ok = ok_traj and ok_flags and isinstance(case["all_widths_trustworthy"], bool)
    print(f"[kdropout-converged selftest] trajectories_recorded={ok_traj} flags_present={ok_flags}  {'PASS' if ok else 'FAIL'}")
    print(f"  (per-width stop epochs: {[conv[k]['stop_epoch'] for k in range(1, 4)]})")
    return ok


def main() -> None:
    """Runs the k-dropout convergence-gated width battery (or `--smoke` / `--selftest`)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="Tiny wiring check, then exit.")
    parser.add_argument("--smoke", action="store_true", help="Tiny config (w_max=4, small cap, 1 seed); no save.")
    parser.add_argument("--config", choices=[str(s) for s in SEEDS], default=None, help="One seed only; default = all.")
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS, help="Safety cap (convergence decides the real stop).")
    parser.add_argument("--check-every", type=int, default=cvg.DEFAULT_CHECK_EVERY, help="Epochs between per-width held-out checkpoints.")
    parser.add_argument("--patience", type=int, default=cvg.DEFAULT_PATIENCE, help="Flat checkpoints that declare a width converged.")
    parser.add_argument("--min-delta", type=float, default=cvg.DEFAULT_MIN_DELTA, help="Held-out-loss decrease (nats) counted as improvement.")
    args = parser.parse_args()

    if args.selftest:
        sys.exit(0 if run_selftest() else 1)

    device = str(get_device())
    if args.smoke:
        seeds, w_max, n_train, n_test, max_epochs, save = [0], 4, N_TRAIN, N_TEST, 8000, False
    else:
        seeds = list(SEEDS) if args.config is None else [int(args.config)]
        w_max, n_train, n_test, max_epochs, save = W_MAX, N_TRAIN, N_TEST, args.max_epochs, True

    print(f"[kdropout-converged] device={device} seeds={seeds} w_max={w_max} max_epochs_cap={max_epochs} check_every={args.check_every}", flush=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    per_case = []
    for seed in seeds:
        print(f"=== W_KDROPOUT_CONVERGED seed={seed} ===", flush=True)
        case = run_case(seed, w_max, n_train, n_test, max_epochs, device, check_every=args.check_every, patience=args.patience, min_delta=args.min_delta)
        per_case.append(case)
        cwe._print_case(case, w_max)

    any_untrustworthy = [c["seed"] for c in per_case if not c["all_widths_trustworthy"]]
    if any_untrustworthy:
        print(f"\n*** DO-NOT-CONCLUDE GUARD: seeds {any_untrustworthy} have widths that did NOT converge (hit cap / still creeping). ***")
        print("*** Raise --max-epochs for those before drawing any conclusion from their curves. ***")
    else:
        print("\nAll widths converged on all seeds — curves are trustworthy to interpret.")

    if save:
        path = os.path.join(RESULTS_DIR, "w_kdropout_converged_summary.json")
        summary = {
            "config": {
                "schedule": "kdropout_sandwich",
                "w_max": w_max,
                "seeds": seeds,
                "max_epochs_cap": max_epochs,
                "lr": cwe.LR,
                "val_every": cwe.VAL_EVERY,
                "check_every": args.check_every,
                "patience": args.patience,
                "min_delta": args.min_delta,
            },
            "per_case": per_case,
            "untrustworthy_seeds": any_untrustworthy,
        }
        with open(path, "w") as f:
            json.dump(sw._jsonable(summary), f, indent=2)
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
