"""X4b: E-lane non-nested arm under MULTI-RESTART — does s1's training collapse flip? (X4 follow-up).

(pre-registration in capacity_ladder_results/X4b/PREREGISTRATION.md; parent X4 in
 capacity_ladder_x4.py; EXECUTION_PLAN §8.5 X4 follow-up / RESULTS.md "## X-queue follow-ups".)

X4 ran the June NON-nested per-input arbiter (mixture-vs-best-single-Gaussian held-out advantage) on
the identical K4 toy-E data across 3 seeds and read model-capture 2/3: the gold-standard Δ*(x)
middle-tercile mean humped on s0 (+0.025) and s2 (+0.023) but went NEGATIVE on s1 (−0.065) — a
mixture TRAINING COLLAPSE (the fitted usage network put ~all mass on one component, degenerating the
mixture to the single-Gaussian baseline). X4's "instrument-general" verdict was therefore HEDGED.

X4b tests whether s1's collapse is an avoidable optimisation artifact by re-fitting each seed with
MULTI-RESTART: R=8 independent inits of the pinned primitive `train_aggregate_sparsity` (deterministic
restart seeds `seed*100 + r`), keeping the fit with the LOWEST TRAINING MAP objective
`model.loss(x_tr, y_tr)` — the exact quantity Adam minimises, using NO test information (standard
non-convex multi-restart, not test-set selection). Everything else is byte-for-byte X4: same data
(N_TR=1000 / N_TE=2500, identical seeds), same instrument config, same scoring, and X4's own helpers
reused verbatim. The single-Gaussian baseline is NOT restarted (it is not what collapses) — only the
mixture, to keep the change minimal and the contrast honest.

Verdict (PREREGISTRATION.md):
  * PRIMARY — model-capture flip on s1: gold_mid(s1) > 0 → model-capture 3/3, collapse was avoidable
    → swings toward W8 (nesting-specific); gold_mid(s1) ≤ 0 → collapse is objective-intrinsic → clean
    W8 refutation stands (E-fragility instrument-general).
  * HARD GUARD — X4b-no-false-positive: E_broad must STILL stay flat/≤0 (arbiter and gold) on 3/3, or
    keep-best is cherry-picking noise and the read is void.
  * SOFT — s0/s2 keep gold_mid > 0 (keep-best must not degrade an already-good fit).

Strictly probabilistic: the MAP objective's Dirichlet-usage prior is the model's own term (coeff 1,
no tuned λ) — unchanged.

Usage:
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=2 ~/dev/.venv/bin/python automl_package/examples/capacity_ladder_x4b.py --selftest
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=2 ~/dev/.venv/bin/python automl_package/examples/capacity_ladder_x4b.py --smoke
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=2 ~/dev/.venv/bin/python automl_package/examples/capacity_ladder_x4b.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable

import numpy as np
import torch

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root, so `import automl_package` works

import _variational_em_perinput as vemp  # noqa: E402
import capacity_ladder_x4 as x4  # noqa: E402  — reuse the parent task's helpers verbatim
import probreg_variational_em_step2_perinput_arbiter as p2  # noqa: E402
import probreg_variational_em_step3_perinput_model as p3  # noqa: E402
import probreg_variational_em_toy_e_hump as hump  # noqa: E402

OUT_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "X4b")
X4_SUMMARY_PATH = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "X4", "x4_summary.json")

# The ONLY new knob relative to X4: number of independent restarts, keep-best-by-training-objective.
R_RESTARTS = 8


def _to_xy_tensors(x: np.ndarray, y: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Builds the float32 (x, y) tensors `train_aggregate_sparsity`/`model.loss` expect."""
    x_arr = np.asarray(x, dtype=np.float32)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(-1, 1)
    y_arr = np.asarray(y, dtype=np.float32).ravel()
    return torch.as_tensor(x_arr), torch.as_tensor(y_arr)


def train_best_of_r(
    x_tr: np.ndarray, y_tr: np.ndarray, seed: int, n_epochs: int, r_restarts: int,
) -> tuple[vemp.AggregateSparsityKSelector, dict[str, object]]:
    """Fits the mixture `r_restarts` times and keeps the lowest-training-MAP-objective fit.

    The keep-best criterion is `model.loss(x_tr, y_tr)` — the deterministic MAP objective Adam
    minimises (closed-form mixture log-likelihood minus the coeff-1 aggregate-usage log-prior),
    evaluated on the TRAINING set only. No held-out / gold information enters model selection.

    Args:
        x_tr: training features, shape ``(N, D)`` or ``(N,)``.
        y_tr: training targets, shape ``(N,)``.
        seed: base seed; restart ``r`` uses ``seed * 100 + r``.
        n_epochs: gradient iterations per restart (X4-matched).
        r_restarts: number of independent restarts.

    Returns:
        The kept (best) model and a dict of ``restart_losses`` (per-restart training objective),
        ``best_restart`` index, and ``best_loss``.
    """
    x_t, y_t = _to_xy_tensors(x_tr, y_tr)
    best_model: vemp.AggregateSparsityKSelector | None = None
    best_loss = float("inf")
    best_restart = -1
    restart_losses: list[float] = []
    for r in range(r_restarts):
        model = vemp.train_aggregate_sparsity(
            x_tr, y_tr, k_max=hump.K_MAX, alpha0=hump.ALPHA0, n_epochs=n_epochs, lr=1e-2, seed=seed * 100 + r,
        )
        with torch.no_grad():
            tr_loss = float(model.loss(x_t, y_t))
        restart_losses.append(tr_loss)
        if tr_loss < best_loss:
            best_loss, best_model, best_restart = tr_loss, model, r
    assert best_model is not None
    return best_model, {"restart_losses": restart_losses, "best_restart": best_restart, "best_loss": best_loss}


def fit_and_score_seed_mr(
    make_fn: Callable[..., tuple[np.ndarray, np.ndarray]],
    given_fn: Callable[..., np.ndarray],
    seed: int,
    n_tr: int,
    n_te: int,
    n_epochs: int,
    m_gold: int,
    n_grid: int,
    r_restarts: int,
) -> dict[str, object]:
    """X4's `fit_and_score_seed` with the mixture trained best-of-`r_restarts` instead of once.

    Identical to `capacity_ladder_x4.fit_and_score_seed` except the single
    `vemp.train_aggregate_sparsity(...)` call is replaced by `train_best_of_r`. The plain-Gaussian
    baseline (`p2.train_cond_gaussian`) and every scoring primitive are the SAME as X4.

    Returns:
        X4's per-seed dict (``x``, ``delta``, ``eff``, ``grid``, ``gold``) plus ``restart_info``.
    """
    x_tr, y_tr = make_fn(n=n_tr, seed=seed)
    x_te, y_te = make_fn(n=n_te, seed=seed + 500)
    model, restart_info = train_best_of_r(x_tr, y_tr, seed=seed, n_epochs=n_epochs, r_restarts=r_restarts)
    pure = p2.train_cond_gaussian(x_tr, y_tr, seed=seed)

    delta = p2.pure_nll_per_point(pure, x_te, y_te) - p3.bucket_nll_per_point(model, x_te, y_te)
    eff = p3.effective_count(model.weights(p2._to_xy(x_te, y_te)[0]).cpu().numpy())
    x_flat = np.asarray(x_te, dtype=np.float64).ravel()

    grid = np.linspace(0.0, 1.0, n_grid).astype(np.float32)
    gold = np.empty(n_grid)
    for j, xg in enumerate(grid):
        xs = np.full(m_gold, xg, dtype=np.float32)
        ys = given_fn(xs, sigma=hump.SIGMA, sep_min=hump.SEP_MIN, sep_max=hump.SEP_MAX, seed=seed * 100_003 + j)
        gold[j] = float(
            (p2.pure_nll_per_point(pure, xs.reshape(-1, 1), ys) - p3.bucket_nll_per_point(model, xs.reshape(-1, 1), ys)).mean()
        )

    return {"x": x_flat, "delta": delta, "eff": eff, "grid": grid.astype(np.float64), "gold": gold, "restart_info": restart_info}


def run_toy_mr(toy: str, seeds: tuple[int, ...], n_tr: int, n_te: int, n_epochs: int, m_gold: int, n_grid: int, r_restarts: int) -> list[dict[str, object]]:
    """`capacity_ladder_x4.run_toy` with multi-restart fits; same tercile verdict + gold + eff reads."""
    make_fn, given_fn = x4.TOY_SPECS[toy]
    rows = []
    for seed in seeds:
        t0 = time.time()
        res = fit_and_score_seed_mr(make_fn, given_fn, seed, n_tr, n_te, n_epochs, m_gold, n_grid, r_restarts)
        verdict = x4.tercile_verdict(res["x"], res["delta"])
        gold_tercile = x4._tercile_means(res["grid"], res["gold"])
        eff_tercile = x4._tercile_means(res["x"], res["eff"])
        wall = time.time() - t0
        ri = res["restart_info"]
        print(
            f"[x4b] {toy} s{seed}: Δ̂ edge/mid/edge = "
            f"{verdict['tercile']['edge_lo']['mean']:+.3f}/{verdict['tercile']['middle']['mean']:+.3f}/{verdict['tercile']['edge_hi']['mean']:+.3f}"
            f"  gold_mid={gold_tercile['middle']:+.3f}  recovered={verdict['recovered']}"
            f"  best_restart={ri['best_restart']}/{r_restarts}  ({wall:.0f}s)"
        )
        rows.append(
            {
                "toy": toy,
                "seed": seed,
                **verdict,
                "gold_tercile": gold_tercile,
                "eff_tercile": eff_tercile,
                "restart_info": ri,
                "wall_time_sec": round(wall, 1),
            }
        )
    return rows


def run_selftest_mr() -> bool:
    """Multi-restart version of X4's gold-standard oracle: E humps, E_broad flat (small N/epochs, R=2)."""
    n_tr, n_te, n_epochs, m_gold, n_grid, r = 500, 600, 300, 600, 24, 2
    seed = 0

    print(f"[x4b selftest] fitting toy E   (n_tr={n_tr} n_te={n_te} epochs={n_epochs} R={r})...")
    make_e, given_e = x4.TOY_SPECS["E"]
    res_e = fit_and_score_seed_mr(make_e, given_e, seed, n_tr, n_te, n_epochs, m_gold, n_grid, r)
    gold_e = x4._tercile_means(res_e["grid"], res_e["gold"])

    print(f"[x4b selftest] fitting toy E_broad (n_tr={n_tr} n_te={n_te} epochs={n_epochs} R={r})...")
    make_b, given_b = x4.TOY_SPECS["E_broad"]
    res_b = fit_and_score_seed_mr(make_b, given_b, seed, n_tr, n_te, n_epochs, m_gold, n_grid, r)
    gold_b = x4._tercile_means(res_b["grid"], res_b["gold"])

    print(f"[x4b selftest] gold Δ*  E:       edge_lo={gold_e['edge_lo']:+.3f}  middle={gold_e['middle']:+.3f}  edge_hi={gold_e['edge_hi']:+.3f}")
    print(f"[x4b selftest] gold Δ*  E_broad: edge_lo={gold_b['edge_lo']:+.3f}  middle={gold_b['middle']:+.3f}  edge_hi={gold_b['edge_hi']:+.3f}")

    ok_e = gold_e["middle"] > 0.0 and gold_e["middle"] > gold_e["edge_lo"] and gold_e["middle"] > gold_e["edge_hi"]
    ok_broad = gold_b["middle"] <= 0.05
    # keep-best must actually rank: the two restart losses must be finite and ordered choice valid.
    ok_restart = res_e["restart_info"]["best_restart"] in range(r) and np.isfinite(res_e["restart_info"]["best_loss"])
    ok = ok_e and ok_broad and ok_restart
    print(f"[x4b selftest] E humps={ok_e}  E_broad flat/<=0={ok_broad}  restart_ok={ok_restart}  {'PASS' if ok else 'FAIL'}")
    return ok


def _load_x4_baseline() -> dict[str, dict[str, object]] | None:
    """Loads X4's per-seed E gold_mid + recovered for the X4 → X4b side-by-side contrast."""
    if not os.path.exists(X4_SUMMARY_PATH):
        return None
    with open(X4_SUMMARY_PATH) as f:
        x4s = json.load(f)
    out: dict[str, dict[str, object]] = {}
    for r in x4s.get("E", []):
        out[str(r["seed"])] = {"gold_mid": r["gold_tercile"]["middle"], "recovered": r["recovered"], "delta_mid": r["tercile"]["middle"]["mean"]}
    return out


def _run(smoke: bool) -> None:
    """Runs the X4b battery (real read, or a reduced smoke check) and writes the summary + verdict."""
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()

    if smoke:
        seeds, n_tr, n_te, n_epochs, m_gold, n_grid, r = (0, 1), 300, 400, 100, 400, 24, 3
    else:
        seeds, n_tr, n_te, n_epochs, m_gold, n_grid, r = x4.SEEDS, x4.N_TR, x4.N_TE, x4.N_EPOCHS, x4.M_GOLD, x4.N_GRID, R_RESTARTS
    print(f"[x4b] smoke={smoke} seeds={seeds} n_tr={n_tr} n_te={n_te} epochs={n_epochs} R={r} m_gold={m_gold} n_grid={n_grid}")

    e_rows = run_toy_mr("E", seeds, n_tr, n_te, n_epochs, m_gold, n_grid, r)
    broad_rows = run_toy_mr("E_broad", seeds, n_tr, n_te, n_epochs, m_gold, n_grid, r)

    # --- Verdict machinery (PREREGISTRATION.md). ---
    def _gold_mid(rows: list[dict[str, object]], seed: int) -> float | None:
        for row in rows:
            if row["seed"] == seed:
                return float(row["gold_tercile"]["middle"])
        return None

    model_capture = {int(row["seed"]): bool(row["gold_tercile"]["middle"] > 0.0) for row in e_rows}
    model_capture_count = int(sum(model_capture.values()))
    n_seeds_recovered = int(sum(r_["recovered"] for r_ in e_rows))

    # Hard guard: E_broad must stay flat (arbiter not recovered AND gold_mid <= tolerance) on all seeds.
    broad_flat_tol = 0.05
    broad_stays_flat = bool(all((not row["recovered"]) and (row["gold_tercile"]["middle"] <= broad_flat_tol) for row in broad_rows))

    x4_baseline = _load_x4_baseline()
    s1_gold_mid_x4 = float(x4_baseline["1"]["gold_mid"]) if x4_baseline and "1" in x4_baseline else None
    s1_gold_mid_x4b = _gold_mid(e_rows, 1)
    s1_flipped = bool(s1_gold_mid_x4b is not None and s1_gold_mid_x4b > 0.0)

    if not broad_stays_flat:
        primary = "VOID_false_positive_on_E_broad"
    elif s1_flipped:
        primary = "s1_flipped_model_capture_3of3_swings_toward_W8_nesting_specific"
    else:
        primary = "s1_still_collapses_clean_W8_refutation_instrument_general"

    verdict = {
        "primary": primary,
        "s1_gold_mid_x4": s1_gold_mid_x4,
        "s1_gold_mid_x4b": s1_gold_mid_x4b,
        "s1_flipped": s1_flipped,
        "model_capture_per_seed": {str(k): v for k, v in model_capture.items()},
        "model_capture_count": model_capture_count,
        "n_seeds_recovered": n_seeds_recovered,
        "broad_stays_flat": broad_stays_flat,
    }

    summary = {
        "task": "X4b",
        "config": {
            "n_tr": n_tr, "n_te": n_te, "n_epochs": n_epochs, "r_restarts": r,
            "m_gold": m_gold, "n_grid": n_grid, "seeds": list(seeds),
            "k_max": hump.K_MAX, "alpha0": hump.ALPHA0, "sigma": hump.SIGMA,
            "sep": [hump.SEP_MIN, hump.SEP_MAX], "keep_best_criterion": "train_MAP_objective", "smoke": smoke,
        },
        "E": e_rows,
        "E_broad": broad_rows,
        "x4_baseline_E": x4_baseline,
        "verdict": verdict,
        "wall_time_sec": round(time.time() - t_start, 1),
    }

    name = "x4b_summary_smoke.json" if smoke else "x4b_summary.json"
    out_path = os.path.join(OUT_DIR, name)
    with open(out_path, "w") as f:
        json.dump(x4._jsonable(summary), f, indent=2)
    print(f"\n[x4b] wrote {out_path}  ({summary['wall_time_sec']:.0f}s)")
    print("[x4b] VERDICT:", json.dumps(x4._jsonable(verdict), indent=2))


def main() -> None:
    """Parses args and dispatches to the selftest, the smoke run, or the real 3-seed multi-restart reader."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="Gold-standard known-answer oracle under multi-restart (fast); exits with a status code.")
    parser.add_argument("--smoke", action="store_true", help="2 seeds, reduced N/epochs, R=3 — proves the pipeline runs end-to-end; not the real read.")
    args = parser.parse_args()

    torch.set_num_threads(x4.TORCH_THREADS)
    if args.selftest:
        sys.exit(0 if run_selftest_mr() else 1)

    _run(smoke=args.smoke)


if __name__ == "__main__":
    main()
