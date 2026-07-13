"""W2 — per-input WIDTH dial on the heterogeneous (line + native-frequency-sine) toy.

(docs/plans/width_dial_2026-07-11/EXECUTION_PLAN.md §6; pre-registration in
capacity_ladder_results/W2/PREREGISTRATION.md)

W1 (`sinc_width_experiment.py`) FAILED bar (i) CONSTRUCTION on the ramped-sinc toy for two reasons
(EXECUTION_PLAN.md §6): (1) the sinc has no genuinely width-FLAT region, so the "tail" gains nearly
as much LL from width as the "centre" does — little per-input contrast; (2) the NESTED per-sample
uniform width draw trains width=w_max only `1/w_max` of the time, under-fitting the hard region.

W2 fixes both: `nested_width_net.make_hetero` splices a straight LINE (easy, `x<0`, probed FLAT at
every width with a dedicated net) to a native-frequency SINE (hard, `x>=0`, probed width-hungry but
learnable), and phase-1 training uses `nested_width_net.WidthSchedule.SANDWICH` — every step ALWAYS
scores width=1 and width=w_max (plus 2 random intermediate widths), guaranteeing w_max is trained
on every step, not `1/w_max` of them.

Otherwise this driver is a straight mirror of `sinc_width_experiment.py`: same two-stage/no-leak
split, same SOFT-recipe selector (`capacity_ladder_k6._RouterMLP`/`_train_router`/`_soft_targets`),
and the SAME 3 bar functions (`sinc_width_experiment._construction_bar`/`_recovery_bar`/
`_deploy_bar`), reused verbatim — not reimplemented — via `import sinc_width_experiment as sw`.
Region convention for those bar functions: `region==1` is the function's "centre" slot (here: the
HARD sine region) and `region==0` is its "tail" slot (here: the EASY line region); the printed/JSON
field names keep the `centre`/`tail` spelling from W1, mapped to hard/easy in the docstrings and
prints below (EXECUTION_PLAN.md's own §6 wording: "hard climbs with width, easy stays flat" is the
CONSTRUCTION expectation flip vs sinc).

Strictly probabilistic throughout: every score is `nested_width_net.gaussian_log_likelihood`, no
MSE-only bar, no penalty/lambda, no tuned regularizer.

Usage:
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/hetero_width_experiment.py --selftest
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/hetero_width_experiment.py --smoke
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/hetero_width_experiment.py --config 0
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/hetero_width_experiment.py   # all seeds
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

RESULTS_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "W2")

SEEDS = (0, 1, 2)
W_MAX = 12  # EXECUTION_PLAN.md §6 W2 toy: "Use W_max=12" (not W1's 16).
N_TRAIN = 1500
N_TEST = 500
LR = 1e-2
N_EPOCHS_PHASE1 = 2500  # matches the Step-0 W2 probe's own convergence budget (hetero_toy_probe_v2.py) and W1's real run.
SMOKE_EPOCHS = 600  # smallest epoch count found (scratchpad tuning) that reliably shows hard-climbs/easy-flat-ish at w_max=6.


# ---------------------------------------------------------------------------
# Phase 1: fit the nested-width net under the SANDWICH schedule (W2's fix over W1's NESTED draw).
# ---------------------------------------------------------------------------


def _fit_phase1(x_tr: np.ndarray, y_tr: np.ndarray, w_max: int, seed: int, n_epochs: int, lr: float, device: str) -> tuple[nwn.NestedWidthNet, dict]:
    """Trains one `NestedWidthNet` with the SANDWICH schedule, standardizing x/y first.

    Identical shape to `sinc_width_experiment._fit_phase1` (standardize on TRAIN stats, return the
    `norm` dict `sw._score_all_widths` expects) except the schedule: `WidthSchedule.SANDWICH`
    instead of W1's default `WidthSchedule.NESTED` — `sw._fit_phase1` cannot be reused directly
    because it has no schedule parameter (kept exactly as W1 needs it, per the plan's guard against
    touching the sinc path).
    """
    mx, sx = float(x_tr.mean()), float(x_tr.std())
    my, sy = float(y_tr.mean()), float(y_tr.std())
    norm = {"mx": mx, "sx": sx, "my": my, "sy": sy}

    x_n = (x_tr - mx) / sx
    y_n = (y_tr - my) / sy
    x_t = torch.as_tensor(x_n, dtype=torch.float32).reshape(-1, 1)
    y_t = torch.as_tensor(y_n, dtype=torch.float32)

    torch.manual_seed(seed)
    net = nwn.NestedWidthNet(w_max=w_max)
    nwn.train_nested_width(net, x_t, y_t, n_epochs=n_epochs, lr=lr, seed=seed, device=device, schedule=nwn.WidthSchedule.SANDWICH)
    return net, norm


# ---------------------------------------------------------------------------
# One (W2, seed) unit: phase 1 + construction bar + phase 2 + recovery/deploy bars.
# ---------------------------------------------------------------------------


def run_case(seed: int, cfg: sw.RunConfig, device: str) -> dict:
    """Runs phase 1 (SANDWICH-trained nested net), the construction bar, phase 2 (selector), and the recovery/deploy bars."""
    x_tr, y_tr, _region_tr = nwn.make_hetero(cfg.n_train, seed)
    x_te, y_te, region_te = nwn.make_hetero(cfg.n_test, seed + 500)

    # Two-stage, no leak: index-parity split of the TRAIN set (H1/W1 p1/p2 convention). Phase 1
    # trains ONLY on p1; phase 2's targets/selector are built ONLY on p2, data the net never saw.
    p1_idx = np.arange(0, cfg.n_train, 2)
    p2_idx = np.arange(1, cfg.n_train, 2)
    x_p1, y_p1 = x_tr[p1_idx], y_tr[p1_idx]
    x_p2, y_p2 = x_tr[p2_idx], y_tr[p2_idx]

    net, norm = _fit_phase1(x_p1, y_p1, cfg.w_max, seed, cfg.n_epochs_phase1, LR, device)

    # --- construction bar, on the untouched TEST set (sw._score_all_widths/_construction_bar reused verbatim) ---
    score_te = sw._score_all_widths(net, norm, x_te, y_te, device)  # (n_test, w_max)
    fixed_width_ll = {k: score_te[:, k - 1] for k in range(1, cfg.w_max + 1)}
    k_mid = max(2, cfg.w_max // 2)
    # region_te: 1 = hard (x>=0, sine), 0 = easy (x<0, line). Maps onto sw._construction_bar's
    # centre(==1)/tail(==0) slots as hard/easy respectively (module docstring above).
    construction = sw._construction_bar(fixed_width_ll, region_te, k_lo=1, k_mid=k_mid, w_max=cfg.w_max)

    # --- phase-2 selector, distilled on the p2 held-out-within-train half (sw._fit_selector reused verbatim) ---
    score_p2 = sw._score_all_widths(net, norm, x_p2, y_p2, device)
    router = sw._fit_selector(score_p2, x_p2, cfg.w_max, seed, device)

    # --- evaluate the selector on the TEST set: bars (ii)/(iii) (sw._recovery_bar/_deploy_bar reused verbatim) ---
    selector_nll, marginal_p, expected_width_te = sw._selector_eval(router, score_te, x_te, device)
    recovery = sw._recovery_bar(expected_width_te, region_te)
    per_k_nll = {k: float(-fixed_width_ll[k].mean()) for k in range(1, cfg.w_max + 1)}
    oracle_nll = float(-score_te.max(axis=1).mean())
    deploy = sw._deploy_bar(selector_nll, per_k_nll, oracle_nll)

    return {
        "seed": seed,
        "n_test": int(score_te.shape[0]),
        "construction": construction,
        "recovery": recovery,
        "deploy": deploy,
        "marginal_p": marginal_p.tolist(),
        "per_k_nll": per_k_nll,
    }


# ---------------------------------------------------------------------------
# Selftest -- 2a/2c WIRING check, no phase-1 training of the big net.
# ---------------------------------------------------------------------------


def _synthetic_width_table_hetero(w_max: int, seed: int, n_per: int = 1000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Planted known-answer `(N, w_max)` width-LL table for the hetero region split.

    Hard climbs to near-oracle, easy mildly prefers LOW width. Mirrors `sinc_width_experiment.
    _synthetic_width_table` (same reasoning: a genuinely-declining, not flat, easy side is needed so
    RECOVERY/DEPLOY exercise a real per-input trade-off) but drawn from the hetero toy's own
    x-range/region convention (sign-of-x split, `x` in `+-nested_width_net.HETERO_R_DEFAULT`, not
    sinc's `|x|<=2*pi` split).
    """
    rng = np.random.default_rng(seed)
    r = nwn.HETERO_R_DEFAULT
    x_easy = rng.uniform(-r, -0.5, n_per)
    x_hard = rng.uniform(0.5, r, n_per)
    x = np.concatenate([x_easy, x_hard]).astype(np.float64)
    reg = (x >= 0).astype(int)

    k = np.arange(1, w_max + 1, dtype=np.float64)
    means_easy = 1.0 - 0.15 * (k - 1)  # mildly prefers LOW width (declining, not flat)
    means_hard = sw.ORACLE_LL - 3.0 * np.exp(-(k - 1) / 2.0)  # climbs from ORACLE_LL-3 to within ~0.09 nat of ORACLE_LL

    n = len(x)
    noise_sd = 0.3
    row_noise = rng.normal(0, noise_sd, size=n)  # one shared value per row, applied to every column
    score = np.empty((n, w_max), dtype=np.float64)
    easy_mask = reg == 0
    score[easy_mask] = means_easy[None, :] + row_noise[easy_mask, None]
    score[~easy_mask] = means_hard[None, :] + row_noise[~easy_mask, None]
    return score, x, reg


def run_selftest() -> bool:
    """No-train wiring check.

    (0) `make_hetero` shapes/finiteness + the phase-1 scoring pipeline on an UNTRAINED net;
    (a) the CONSTRUCTION bar on a planted hard-climbs/easy-declining table; (b)/(c) selector
    distillation + the RECOVERY/DEPLOY bars on the SAME planted table.
    """
    device = "cpu"
    ok = True

    print("[w2 selftest] (0a) make_hetero wiring: shapes/finite/region values")
    x_h, y_h, reg_h = nwn.make_hetero(200, seed=0)
    ok0a = x_h.shape == (200,) and y_h.shape == (200,) and reg_h.shape == (200,) and bool(np.isfinite(y_h).all()) and set(np.unique(reg_h).tolist()) <= {0, 1}
    print(f"  x/y/region shapes={x_h.shape}/{y_h.shape}/{reg_h.shape} finite_y={bool(np.isfinite(y_h).all())}  {'PASS' if ok0a else 'FAIL'}")
    ok = ok and ok0a

    print("[w2 selftest] (0b) nested_width_net integration wiring (random-init net, no phase-1 training)")
    torch.manual_seed(0)
    net = nwn.NestedWidthNet(w_max=6)
    x_fake = np.array([0.0, 1.0, -1.0, 10.0, -6.0], dtype=np.float32)
    y_fake = np.array([0.1, 0.2, -0.1, 0.5, -0.3], dtype=np.float32)
    norm = {"mx": 0.0, "sx": 1.0, "my": 0.0, "sy": 1.0}
    score0 = sw._score_all_widths(net, norm, x_fake, y_fake, device)
    ok0b = score0.shape == (5, 6) and bool(np.isfinite(score0).all())
    print(f"  score table shape={score0.shape} finite={bool(np.isfinite(score0).all())}  {'PASS' if ok0b else 'FAIL'}")
    ok = ok and ok0b

    w_max = 8
    print("[w2 selftest] (a) construction bar wiring: planted hard-climbs-to-oracle / easy-declining table")
    score_syn, x_syn, region_syn = _synthetic_width_table_hetero(w_max=w_max, seed=0)
    fixed_width_ll = {k: score_syn[:, k - 1] for k in range(1, w_max + 1)}
    construction = sw._construction_bar(fixed_width_ll, region_syn, k_lo=1, k_mid=w_max // 2, w_max=w_max)
    ok_a = construction["construction_pass"]
    print(
        f"  hard(=centre)_climbs={construction['centre_climbs']} hard_near_noise_floor={construction['centre_near_noise_floor']} "
        f"easy(=tail)_flat={construction['tail_flat']} construction_pass={construction['construction_pass']}  {'PASS' if ok_a else 'FAIL'}"
    )
    ok = ok and ok_a

    print("[w2 selftest] (b)/(c) selector distillation + recovery/deploy bar wiring (same planted table)")
    router = sw._fit_selector(score_syn, x_syn, w_max, seed=0, device=device)
    selector_nll, _marginal_p, expected_width = sw._selector_eval(router, score_syn, x_syn, device)
    recovery = sw._recovery_bar(expected_width, region_syn)
    per_k_nll = {k: float(-fixed_width_ll[k].mean()) for k in range(1, w_max + 1)}
    oracle_nll = float(-score_syn.max(axis=1).mean())
    deploy = sw._deploy_bar(selector_nll, per_k_nll, oracle_nll)
    ok_b = recovery["separation_beats_2se"]
    # Wiring check for DEPLOY: gate on "beats the best single GLOBAL width" only, exactly like
    # sw.run_selftest's own (b)/(c) -- the literal "within 0.02 nat of oracle" threshold is a
    # real-run bar, reported not asserted here (same _soft_targets tercile-smoothing caveat as W1).
    ok_c = deploy["matches_or_beats_global"]
    print(
        f"  recovery: expected_width hard(=centre)={recovery['mean_expected_width_centre']:.3f} easy(=tail)={recovery['mean_expected_width_tail']:.3f} "
        f"diff={recovery['diff']:.3f} se={recovery['se']:.4f}  {'PASS' if ok_b else 'FAIL'}"
    )
    print(
        f"  deploy: selector_nll={deploy['selector_nll']:.4f} best_global_nll={deploy['best_global_nll']:.4f} "
        f"oracle_nll={deploy['oracle_nll']:.4f} gap_to_oracle={deploy['gap_to_oracle_nat']:.4f} "
        f"(matches_or_beats_global gated here; within-0.02-nat-of-oracle is a real-run bar, reported not asserted)  {'PASS' if ok_c else 'FAIL'}"
    )
    ok = ok and ok_b and ok_c

    print(f"[w2 selftest] {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Real reader.
# ---------------------------------------------------------------------------


def main() -> None:
    """Runs the W2 battery (or `--smoke`'s tiny stand-in / `--selftest`'s wiring check)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="No-train wiring check, then exit.")
    parser.add_argument("--smoke", action="store_true", help="Tiny config (w_max=6, SMOKE_EPOCHS phase-1 epochs, 1 seed); no save.")
    parser.add_argument("--config", choices=[str(s) for s in SEEDS], default=None, help="Run only this one seed (sharded parallel launching); default = all seeds.")
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

    print(f"[w2] device={device} config={cfg}")
    os.makedirs(cfg.results_dir, exist_ok=True)

    per_case = []
    for seed in cfg.seeds:
        print(f"=== W2 seed={seed} ===", flush=True)
        case = run_case(seed, cfg, device)
        per_case.append(case)
        c, r, d = case["construction"], case["recovery"], case["deploy"]
        print(
            f"  construction: hard(=centre)_climbs={c['centre_climbs']} hard_near_floor={c['centre_near_noise_floor']} "
            f"easy(=tail)_flat={c['tail_flat']} pass={c['construction_pass']}"
        )
        print(f"  recovery: hard_width={r['mean_expected_width_centre']:.3f} easy_width={r['mean_expected_width_tail']:.3f} sep_pass={r['separation_beats_2se']}")
        print(f"  deploy: selector_nll={d['selector_nll']:.4f} best_global_nll={d['best_global_nll']:.4f} oracle_nll={d['oracle_nll']:.4f} pass={d['deploy_pass']}")

    n_seeds = len(per_case)
    n_construction_pass = sum(1 for c in per_case if c["construction"]["construction_pass"])
    n_recovery_pass = sum(1 for c in per_case if c["recovery"]["separation_beats_2se"])
    n_deploy_pass = sum(1 for c in per_case if c["deploy"]["deploy_pass"])
    bar_i = {"n_pass": n_construction_pass, "n_seeds": n_seeds, "pass": bool(n_seeds) and n_construction_pass >= math.ceil(2 * n_seeds / 3)}
    bar_ii = {"n_pass": n_recovery_pass, "n_seeds": n_seeds, "pass": bool(n_seeds) and n_recovery_pass >= math.ceil(2 * n_seeds / 3)}
    bar_iii = {"n_pass": n_deploy_pass, "n_seeds": n_seeds, "pass": bool(n_seeds) and n_deploy_pass >= math.ceil(2 * n_seeds / 3)}

    if bar_i["pass"] and bar_ii["pass"] and bar_iii["pass"]:
        verdict = "FOUND_LEARNABLE_WIDTH_CONTROL"
    elif not bar_i["pass"]:
        verdict = f"FAIL_i_CONSTRUCTION: {n_construction_pass}/{n_seeds} seeds passed (need >=2/3)."
    elif not bar_ii["pass"]:
        verdict = f"FAIL_ii_RECOVERY: {n_recovery_pass}/{n_seeds} seeds passed (need >=2/3)."
    else:
        verdict = f"FAIL_iii_DEPLOY: {n_deploy_pass}/{n_seeds} seeds passed (need >=2/3)."

    print(f"\nbar (i) CONSTRUCTION: {bar_i}")
    print(f"bar (ii) RECOVERY: {bar_ii}")
    print(f"bar (iii) DEPLOY: {bar_iii}")
    print(f"VERDICT: {verdict}")

    summary = {
        "config": {
            "w_max": cfg.w_max, "seeds": cfg.seeds, "n_train": cfg.n_train, "n_test": cfg.n_test,
            "n_epochs_phase1": cfg.n_epochs_phase1, "lr": LR, "phase2_epochs": sw.ck6.N_EPOCHS, "phase2_lr": sw.ck6.LR,
            "oracle_ll": sw.ORACLE_LL, "noise_sigma": sw.NOISE_SIGMA, "hetero_r": nwn.HETERO_R_DEFAULT,
            "noise_floor_multiple": sw.NOISE_FLOOR_MULTIPLE, "deploy_oracle_tol_nat": sw.DEPLOY_ORACLE_TOL_NAT,
            "schedule": nwn.WidthSchedule.SANDWICH.value,
        },
        "per_case": per_case,
        "bar_i_construction": bar_i,
        "bar_ii_recovery": bar_ii,
        "bar_iii_deploy": bar_iii,
        "verdict": verdict,
    }

    if save:
        summary_path = os.path.join(cfg.results_dir, "w2_summary.json")
        with open(summary_path, "w") as f:
            json.dump(sw._jsonable(summary), f, indent=2)
        print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
