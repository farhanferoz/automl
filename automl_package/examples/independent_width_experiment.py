"""W-INDEP — per-input WIDTH dial with INDEPENDENT per-width weights (no sharing) + k-dropout.

(docs/plans/width_dial_2026-07-11/EXECUTION_PLAN.md §6 follow-up; the single-variable control the
user proposed: keep the k-dropout selection + one module + the selector, change ONLY shared -> disjoint
per-width weights.)

W1/W2 (`sinc_width_experiment.py` / `hetero_width_experiment.py`) fail bar (i) because the SHARED
trunk forces node-0 to double as a good small-width predictor, starving the full-width fit — verified
by a dedicated-vs-shared control (a dedicated w12 net reaches the noise floor; both shared schedules
plateau ~1.95 nat short). This driver removes exactly that one variable: `nested_width_net.
IndependentWidthNet` gives every width its OWN disjoint sub-net (the width twin of
`automl_package/models/independent_weights_flexible_neural_network.py::IndependentWeightsFlexibleNN`,
which does the same per DEPTH). Training is the SAME `WidthSchedule.SANDWICH` k-dropout the shared W2
used, so the ONLY difference from W2 is shared-vs-independent weights.

Everything else mirrors `hetero_width_experiment.py` verbatim: same `make_hetero` toy, same
two-stage/no-leak p1/p2 split, same SOFT selector, the SAME 3 bar functions
(`sinc_width_experiment._construction_bar`/`_recovery_bar`/`_deploy_bar`), same 3 seeds. Region
convention for the bar functions: `region==1` = "centre" slot = HARD sine region; `region==0` =
"tail" slot = EASY line region.

Usage:
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/independent_width_experiment.py --selftest
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/independent_width_experiment.py --smoke
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/independent_width_experiment.py            # all seeds
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
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root, so `import automl_package` works

import nested_width_net as nwn  # noqa: E402
import sinc_width_experiment as sw  # noqa: E402 — reuse RunConfig + the 3 bar functions + score/selector helpers verbatim

from automl_package.utils.pytorch_utils import get_device  # noqa: E402

RESULTS_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "W_INDEP")

SEEDS = (0, 1, 2)
W_MAX = 12  # same as W2 (hetero toy).
N_TRAIN = 1500
N_TEST = 500
LR = 1e-2
N_EPOCHS_PHASE1 = 2500  # same budget as the W2 shared run — the only intended difference is the net's weight-sharing.
SMOKE_EPOCHS = 600

_WIRING_TOL = 1e-6  # selftest tolerance for the exact-arithmetic wiring checks (independence + all-vs-per-width).


def _fit_phase1(x_tr: np.ndarray, y_tr: np.ndarray, w_max: int, seed: int, n_epochs: int, lr: float, device: str) -> tuple[nwn.IndependentWidthNet, dict]:
    """Trains one `IndependentWidthNet` with the SANDWICH k-dropout schedule, standardizing x/y first.

    Identical to `hetero_width_experiment._fit_phase1` EXCEPT the net class: `IndependentWidthNet`
    (disjoint per-width weights) instead of `NestedWidthNet` (shared nested trunk). `train_nested_
    width`'s SANDWICH branch touches only `net.w_max`/`net.forward_width`, so it drives either net
    identically — the sole controlled variable is shared-vs-independent weights.
    """
    mx, sx = float(x_tr.mean()), float(x_tr.std())
    my, sy = float(y_tr.mean()), float(y_tr.std())
    norm = {"mx": mx, "sx": sx, "my": my, "sy": sy}

    x_n = (x_tr - mx) / sx
    y_n = (y_tr - my) / sy
    x_t = torch.as_tensor(x_n, dtype=torch.float32).reshape(-1, 1)
    y_t = torch.as_tensor(y_n, dtype=torch.float32)

    torch.manual_seed(seed)
    net = nwn.IndependentWidthNet(w_max=w_max)
    nwn.train_nested_width(net, x_t, y_t, n_epochs=n_epochs, lr=lr, seed=seed, device=device, schedule=nwn.WidthSchedule.SANDWICH)
    return net, norm


def run_case(seed: int, cfg: sw.RunConfig, device: str) -> dict:
    """Phase 1 (independent-weights net, SANDWICH), construction bar, phase 2 selector, recovery/deploy bars."""
    x_tr, y_tr, _region_tr = nwn.make_hetero(cfg.n_train, seed)
    x_te, y_te, region_te = nwn.make_hetero(cfg.n_test, seed + 500)

    p1_idx = np.arange(0, cfg.n_train, 2)
    p2_idx = np.arange(1, cfg.n_train, 2)
    x_p1, y_p1 = x_tr[p1_idx], y_tr[p1_idx]
    x_p2, y_p2 = x_tr[p2_idx], y_tr[p2_idx]

    net, norm = _fit_phase1(x_p1, y_p1, cfg.w_max, seed, cfg.n_epochs_phase1, LR, device)

    score_te = sw._score_all_widths(net, norm, x_te, y_te, device)  # (n_test, w_max)
    fixed_width_ll = {k: score_te[:, k - 1] for k in range(1, cfg.w_max + 1)}
    k_mid = max(2, cfg.w_max // 2)
    construction = sw._construction_bar(fixed_width_ll, region_te, k_lo=1, k_mid=k_mid, w_max=cfg.w_max)

    score_p2 = sw._score_all_widths(net, norm, x_p2, y_p2, device)
    router = sw._fit_selector(score_p2, x_p2, cfg.w_max, seed, device)

    selector_nll, marginal_p, expected_width_te = sw._selector_eval(router, score_te, x_te, device)
    recovery = sw._recovery_bar(expected_width_te, region_te)
    per_k_nll = {k: float(-fixed_width_ll[k].mean()) for k in range(1, cfg.w_max + 1)}
    oracle_nll = float(-score_te.max(axis=1).mean())
    deploy = sw._deploy_bar(selector_nll, per_k_nll, oracle_nll)

    return {
        "seed": seed,
        "n_test": int(score_te.shape[0]),
        "hard_curve": [float(score_te[region_te == 1][:, k - 1].mean()) for k in range(1, cfg.w_max + 1)],
        "easy_curve": [float(score_te[region_te == 0][:, k - 1].mean()) for k in range(1, cfg.w_max + 1)],
        "construction": construction,
        "recovery": recovery,
        "deploy": deploy,
        "marginal_p": marginal_p.tolist(),
        "per_k_nll": per_k_nll,
    }


def _assert_independence(net: nwn.IndependentWidthNet, x: torch.Tensor) -> tuple[bool, float]:
    """Perturbing sub-net j's weights must NOT change any width-k!=j output (disjoint weights)."""
    ok_all = True
    max_leak = 0.0
    for j in range(1, net.w_max + 1):
        with torch.no_grad():
            base = {k: net.forward_width(x, k)[0].clone() for k in range(1, net.w_max + 1)}
            w_orig = net.subnets[j - 1]["mean_head"].weight.detach().clone()
            net.subnets[j - 1]["mean_head"].weight += torch.randn_like(net.subnets[j - 1]["mean_head"].weight) * 5.0
            for k in range(1, net.w_max + 1):
                if k == j:
                    continue
                leak = (net.forward_width(x, k)[0] - base[k]).abs().max().item()
                max_leak = max(max_leak, leak)
                ok_all = ok_all and (leak < _WIRING_TOL)
            net.subnets[j - 1]["mean_head"].weight.copy_(w_orig)
    return ok_all, max_leak


def run_selftest() -> bool:
    """No-train wiring check: (0) IndependentWidthNet independence + all-widths consistency; (a)/(b)/(c) bars wiring."""
    device = "cpu"
    ok = True

    print("[w-indep selftest] (0a) IndependentWidthNet independence (perturbing width-j leaves width-k!=j unchanged)")
    torch.manual_seed(0)
    net = nwn.IndependentWidthNet(w_max=6)
    net.eval()
    x = torch.randn(23, 1)
    ok0a, leak = _assert_independence(net, x)
    print(f"  max cross-width leak={leak:.3e} (tol=1e-6)  {'PASS' if ok0a else 'FAIL'}")
    ok = ok and ok0a

    print("[w-indep selftest] (0b) all_widths_forward agrees with per-width forward_width, finite shapes")
    with torch.no_grad():
        mean_all, logvar_all = net.all_widths_forward(x)
    ok_shape = tuple(mean_all.shape) == (23, 6) and tuple(logvar_all.shape) == (23, 6) and bool(torch.isfinite(mean_all).all())
    max_err = 0.0
    for k in range(1, net.w_max + 1):
        with torch.no_grad():
            mk, lk = net.forward_width(x, k)
        max_err = max(max_err, (mean_all[:, k - 1 : k] - mk).abs().max().item(), (logvar_all[:, k - 1 : k] - lk).abs().max().item())
    ok0b = ok_shape and max_err < _WIRING_TOL
    print(f"  shapes ok={ok_shape} all-vs-per-width max_err={max_err:.3e}  {'PASS' if ok0b else 'FAIL'}")
    ok = ok and ok0b

    print("[w-indep selftest] (0c) score pipeline on an untrained net (sw._score_all_widths integration)")
    x_fake = np.array([0.0, 1.0, -1.0, 10.0, -6.0], dtype=np.float32)
    y_fake = np.array([0.1, 0.2, -0.1, 0.5, -0.3], dtype=np.float32)
    score0 = sw._score_all_widths(net, {"mx": 0.0, "sx": 1.0, "my": 0.0, "sy": 1.0}, x_fake, y_fake, device)
    ok0c = score0.shape == (5, 6) and bool(np.isfinite(score0).all())
    print(f"  score table shape={score0.shape} finite={bool(np.isfinite(score0).all())}  {'PASS' if ok0c else 'FAIL'}")
    ok = ok and ok0c

    # (a)/(b)/(c): reuse the hetero driver's own planted-table bar wiring check verbatim.
    print("[w-indep selftest] (a)/(b)/(c) bars wiring: reuse hetero planted-table check")
    import hetero_width_experiment as hw  # noqa: PLC0415 — selftest-only reuse of the planted-table bar wiring

    w_max = 8
    score_syn, x_syn, region_syn = hw._synthetic_width_table_hetero(w_max=w_max, seed=0)
    fixed_width_ll = {k: score_syn[:, k - 1] for k in range(1, w_max + 1)}
    construction = sw._construction_bar(fixed_width_ll, region_syn, k_lo=1, k_mid=w_max // 2, w_max=w_max)
    router = sw._fit_selector(score_syn, x_syn, w_max, seed=0, device=device)
    _snll, _mp, expected_width = sw._selector_eval(router, score_syn, x_syn, device)
    recovery = sw._recovery_bar(expected_width, region_syn)
    ok_bars = construction["construction_pass"] and recovery["separation_beats_2se"]
    print(f"  construction_pass={construction['construction_pass']} recovery_sep={recovery['separation_beats_2se']}  {'PASS' if ok_bars else 'FAIL'}")
    ok = ok and ok_bars

    print(f"[w-indep selftest] {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    """Runs the W-INDEP battery (or `--smoke`'s tiny stand-in / `--selftest`'s wiring check)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="No-train wiring check, then exit.")
    parser.add_argument("--smoke", action="store_true", help="Tiny config (w_max=6, SMOKE_EPOCHS epochs, 1 seed); no save.")
    parser.add_argument("--config", choices=[str(s) for s in SEEDS], default=None, help="Run only this one seed; default = all seeds.")
    args = parser.parse_args()

    if args.selftest:
        sys.exit(0 if run_selftest() else 1)

    device = str(get_device())

    if args.smoke:
        cfg = sw.RunConfig(w_max=6, seeds=[0], n_train=N_TRAIN, n_test=N_TEST, n_epochs_phase1=SMOKE_EPOCHS, results_dir=RESULTS_DIR)
        save = False
    else:
        seeds = list(SEEDS) if args.config is None else [int(args.config)]
        cfg = sw.RunConfig(w_max=W_MAX, seeds=seeds, n_train=N_TRAIN, n_test=N_TEST, n_epochs_phase1=N_EPOCHS_PHASE1, results_dir=RESULTS_DIR)
        save = True

    print(f"[w-indep] device={device} config={cfg}")
    os.makedirs(cfg.results_dir, exist_ok=True)

    per_case = []
    for seed in cfg.seeds:
        print(f"=== W-INDEP seed={seed} ===", flush=True)
        case = run_case(seed, cfg, device)
        per_case.append(case)
        c, r, d = case["construction"], case["recovery"], case["deploy"]
        print(f"  hard LL by width: {np.array2string(np.array(case['hard_curve']), precision=3, floatmode='fixed')}")
        print(f"  easy LL by width: {np.array2string(np.array(case['easy_curve']), precision=3, floatmode='fixed')}")
        print(
            f"  construction: hard_climbs={c['centre_climbs']} hard_near_floor={c['centre_near_noise_floor']} "
            f"easy_flat={c['tail_flat']} pass={c['construction_pass']}"
        )
        print(f"  recovery: hard_width={r['mean_expected_width_centre']:.3f} easy_width={r['mean_expected_width_tail']:.3f} sep_pass={r['separation_beats_2se']}")
        print(
            f"  deploy: selector_nll={d['selector_nll']:.4f} best_global_nll={d['best_global_nll']:.4f} "
            f"oracle_nll={d['oracle_nll']:.4f} matches_or_beats={d['matches_or_beats_global']} pass={d['deploy_pass']}",
            flush=True,
        )

    n_seeds = len(per_case)
    n_construction = sum(1 for c in per_case if c["construction"]["construction_pass"])
    n_recovery = sum(1 for c in per_case if c["recovery"]["separation_beats_2se"])
    n_deploy = sum(1 for c in per_case if c["deploy"]["deploy_pass"])
    thr = math.ceil(2 * n_seeds / 3)
    bar_i = {"n_pass": n_construction, "n_seeds": n_seeds, "pass": bool(n_seeds) and n_construction >= thr}
    bar_ii = {"n_pass": n_recovery, "n_seeds": n_seeds, "pass": bool(n_seeds) and n_recovery >= thr}
    bar_iii = {"n_pass": n_deploy, "n_seeds": n_seeds, "pass": bool(n_seeds) and n_deploy >= thr}

    if bar_i["pass"] and bar_ii["pass"] and bar_iii["pass"]:
        verdict = "FOUND_LEARNABLE_WIDTH_CONTROL_INDEPENDENT"
    elif not bar_i["pass"]:
        verdict = f"FAIL_i_CONSTRUCTION: {n_construction}/{n_seeds} (need >={thr})."
    elif not bar_ii["pass"]:
        verdict = f"FAIL_ii_RECOVERY: {n_recovery}/{n_seeds} (need >={thr})."
    else:
        verdict = f"FAIL_iii_DEPLOY: {n_deploy}/{n_seeds} (need >={thr})."

    print(f"\nbar (i) CONSTRUCTION: {bar_i}")
    print(f"bar (ii) RECOVERY: {bar_ii}")
    print(f"bar (iii) DEPLOY: {bar_iii}")
    print(f"VERDICT: {verdict}")

    summary = {
        "config": {
            "w_max": cfg.w_max, "seeds": cfg.seeds, "n_train": cfg.n_train, "n_test": cfg.n_test,
            "n_epochs_phase1": cfg.n_epochs_phase1, "lr": LR, "oracle_ll": sw.ORACLE_LL, "noise_sigma": sw.NOISE_SIGMA,
            "hetero_r": nwn.HETERO_R_DEFAULT, "schedule": nwn.WidthSchedule.SANDWICH.value, "weights": "independent_per_width",
        },
        "per_case": per_case,
        "bar_i_construction": bar_i,
        "bar_ii_recovery": bar_ii,
        "bar_iii_deploy": bar_iii,
        "verdict": verdict,
    }

    if save:
        summary_path = os.path.join(cfg.results_dir, "w_indep_summary.json")
        with open(summary_path, "w") as f:
            json.dump(sw._jsonable(summary), f, indent=2)
        print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
