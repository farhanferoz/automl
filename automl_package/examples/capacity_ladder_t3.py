"""T3: moving-mode power curve -- the count-lane analog of X10 (WS-B, task T3).

(docs/plans/perinput_selector_2026-07-10/EXECUTION_PLAN.md §2 T3; pre-registration in
 capacity_ladder_results/T3/PREREGISTRATION.md)

X4/X4b closed the moving-mode question at a single N (N_TR=N_TE=1000): the multi-restart
non-nested arbiter (`capacity_ladder_x4b.py`) reads model-capture 2/3 or 3/3 depending on seed and
X4b's own collapse-vs-genuine-limit split, but never asks whether N=1000 is simply UNDER-POWERED
for the hump. T3 turns "absent at N=1000" into a power statement by re-running the identical
instrument at N_train=N_test in {1000, 4000, 16000}: does the gold-standard hump emerge at some
larger N, or does it stay absent up to 16k with the unimodal control staying flat throughout?

Instrument (reused VERBATIM, not reimplemented): `capacity_ladder_x4b.py`'s E-lane non-nested
arm under R=8 multi-restart, keep-best-by-train-MAP-objective (leak-free -- model selection never
sees held-out or gold information). T3 calls X4b's own fitting/scoring primitive
(`fit_and_score_seed_mr`) and X4's own tercile-reduction primitives (`tercile_verdict`,
`_tercile_means`, `_tercile_masks`) unchanged; the ONLY new read is a bootstrap SE on the gold
Δ*(x) middle-tercile mean (X4/X4b report a bare mean for the gold curve, with no dispersion --
T3's pre-registered bars need `gold_mid > 0` judged against `2*SE`). That SE is computed with the
same G1 plain i.i.d. bootstrap utility (`_capacity_ladder._bootstrap_col_means`) every other
capacity-ladder script uses (F2/F3/X1/K1K2K3) -- not an invented statistic, just the missing
dispersion on an existing read.

Design: for each N in {1000, 4000, 16000}, fit toy E and its E_broad control across 3 seeds
(0, 1, 2), R=8 restarts each, everything else (N_EPOCHS, M_GOLD, N_GRID, K_MAX, ALPHA0, SIGMA,
SEP) held at X4/X4b's own instrument config -- unchanged across the N-sweep, since those govern
training/evaluation resolution, not the identifiability question T3 is asking.

Pre-registered outcome readings (locked, EXECUTION_PLAN §2 T3):
  * Hump emerges (gold Δ*_mid > 0 by 2·SE on >= 2/3 seeds) at some N -> "recoverable at N=<N>"
    (report the crossing N, i.e. the smallest N in the sweep at which this holds).
  * Stays <= 0 at 16k with the control flat throughout -> "moving-mode per-input count effectively
    absent up to N=16000" (report the measured bound).
  * Control (E_broad) must stay flat 3/3 at EVERY N (arbiter not-recovered AND gold middle <= the
    X4b flat tolerance) -- if it does not, the instrument is invalid at that N and the whole read
    escalates to a fresh-context adjudicator; it is not folded into either outcome above.

Strictly probabilistic: unchanged from X4/X4b -- the MAP objective's Dirichlet-usage prior is the
model's own term (coefficient 1, no tuned lambda); no penalty is added by T3.

Usage:
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/capacity_ladder_t3.py --selftest
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/capacity_ladder_t3.py --smoke
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/capacity_ladder_t3.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root, so `import automl_package` works

import _capacity_ladder as cl  # noqa: E402
import capacity_ladder_x4 as x4  # noqa: E402 — reuse TOY_SPECS + tercile/verdict machinery verbatim
import capacity_ladder_x4b as x4b  # noqa: E402 — reuse fit_and_score_seed_mr (R=8 multi-restart) verbatim

OUT_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "T3")

# The ONLY new knob relative to X4b: the N-sweep. Everything else is X4b's own instrument config.
N_SWEEP = (1000, 4000, 16000)
SEEDS = x4.SEEDS  # (0, 1, 2)
R_RESTARTS = x4b.R_RESTARTS  # 8
N_EPOCHS = x4.N_EPOCHS
M_GOLD, N_GRID = x4.M_GOLD, x4.N_GRID
TORCH_THREADS = x4.TORCH_THREADS
BROAD_FLAT_TOL = 0.05  # X4b's own hard-guard tolerance, reused unchanged.

_BOOT_N = 1000


# ---------------------------------------------------------------------------
# The one new read: bootstrap SE on the gold Δ*(x) middle-tercile mean.
# ---------------------------------------------------------------------------


def _boot_se(vec: np.ndarray, seed: int, n_boot: int = _BOOT_N) -> float:
    """Plain i.i.d. bootstrap SE of a 1-D vector's mean.

    G1; reuses `_capacity_ladder._bootstrap_col_means`, same convention as
    `capacity_ladder_f2.py`/`f3.py`/`x1.py`/`k1k2k3.py`'s `_boot_se`/`_plain_boot_se`.
    """
    rng = np.random.default_rng(seed)
    boot = cl._bootstrap_col_means(vec.reshape(-1, 1), n_boot, None, rng)
    return float(boot[:, 0].std(ddof=1))


def gold_mid_with_se(grid: np.ndarray, gold: np.ndarray, boot_seed: int) -> dict[str, float]:
    """Gold Δ*_mid (mean of the gold curve's middle x-tercile) plus its bootstrap SE.

    `capacity_ladder_x4._tercile_means` reduces the gold curve to a bare mean with no dispersion;
    T3's pre-registered bar needs `gold_mid > 0` judged against `2*SE`, so this adds ONLY the
    missing SE via the shared bootstrap utility -- it does not re-derive the mean or touch X4/X4b.

    Args:
        grid: gold-standard grid points, shape ``(n_grid,)``.
        gold: gold-standard Δ*(x) at each grid point, shape ``(n_grid,)``.
        boot_seed: RNG seed for the bootstrap (reproducibility).

    Returns:
        Dict with ``mean``, ``se``, ``n`` (grid points in the middle tercile), and
        ``significant_positive`` (``mean - 2*se > 0``).
    """
    mid_mask = x4._tercile_masks(grid)[1]
    mid_vals = gold[mid_mask]
    mean = float(mid_vals.mean())
    se = _boot_se(mid_vals, seed=boot_seed)
    return {"mean": mean, "se": se, "n": int(mid_vals.size), "significant_positive": bool(mean - 2.0 * se > 0.0)}


# ---------------------------------------------------------------------------
# Per-N, per-toy driver: wraps X4b's fitting primitive, reuses X4's tercile readers verbatim.
# ---------------------------------------------------------------------------


def run_toy_n(
    toy: str, seeds: tuple[int, ...], n: int, n_epochs: int, m_gold: int, n_grid: int, r_restarts: int, boot_seed_base: int,
) -> list[dict[str, object]]:
    """Runs X4b's `fit_and_score_seed_mr` at a single N_train=N_test=`n`, across all seeds.

    Thin wrapper around `capacity_ladder_x4b.fit_and_score_seed_mr` (the pinned multi-restart
    primitive, called unchanged) -- adds only the `gold_mid_with_se` bootstrap-SE read on top of
    X4's own `tercile_verdict`/`_tercile_means` reductions.
    """
    make_fn, given_fn = x4.TOY_SPECS[toy]
    rows = []
    for seed in seeds:
        t0 = time.time()
        res = x4b.fit_and_score_seed_mr(make_fn, given_fn, seed, n, n, n_epochs, m_gold, n_grid, r_restarts)
        verdict = x4.tercile_verdict(res["x"], res["delta"])
        gold_tercile = x4._tercile_means(res["grid"], res["gold"])
        gold_mid = gold_mid_with_se(res["grid"], res["gold"], boot_seed=boot_seed_base + seed)
        eff_tercile = x4._tercile_means(res["x"], res["eff"])
        wall = time.time() - t0
        ri = res["restart_info"]
        print(
            f"[t3] N={n} {toy} s{seed}: gold_mid={gold_mid['mean']:+.3f}±{gold_mid['se']:.3f} "
            f"(sig_pos={gold_mid['significant_positive']})  arbiter_recovered={verdict['recovered']}  "
            f"best_restart={ri['best_restart']}/{r_restarts}  ({wall:.0f}s)"
        )
        rows.append(
            {
                "toy": toy,
                "seed": seed,
                "n": n,
                **verdict,
                "gold_tercile": gold_tercile,
                "gold_mid": gold_mid,
                "eff_tercile": eff_tercile,
                "restart_info": ri,
                "wall_time_sec": round(wall, 1),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Selftest -- X4b's own instrument oracle, unchanged, plus a measure-one N=16000 smoke-fit.
# ---------------------------------------------------------------------------


def run_selftest() -> bool:
    """Re-runs X4b's own selftest UNCHANGED, then measures one real-config N=16000 smoke-fit.

    The instrument check is `capacity_ladder_x4b.run_selftest_mr` called verbatim -- T3 does not
    re-derive or duplicate it. The smoke-fit is the "measure ONE unit before the matrix" datum
    (§0b): a single N=16000, R=1, 1-seed fit at the REAL instrument config (N_EPOCHS/M_GOLD/N_GRID
    unchanged), so its wall-time is a faithful per-restart estimate for costing the real N=16000
    row of the sweep (which needs R=8 restarts × 3 seeds × 2 toys at that N).
    """
    print("[t3 selftest] re-running x4b's own instrument selftest (unchanged)...")
    ok_instrument = x4b.run_selftest_mr()
    print(f"[t3 selftest] x4b instrument selftest: {'PASS' if ok_instrument else 'FAIL'}")

    n = 16000
    print(f"[t3 selftest] N={n} smoke-fit (R=1, 1 seed, real N_EPOCHS={N_EPOCHS}/M_GOLD={M_GOLD}/N_GRID={N_GRID}) — measuring wall-time...")
    make_e, given_e = x4.TOY_SPECS["E"]
    t0 = time.time()
    res = x4b.fit_and_score_seed_mr(make_e, given_e, 0, n, n, N_EPOCHS, M_GOLD, N_GRID, 1)
    wall = time.time() - t0
    verdict = x4.tercile_verdict(res["x"], res["delta"])
    gold_mid = gold_mid_with_se(res["grid"], res["gold"], boot_seed=0)

    finite_ok = bool(
        np.isfinite(gold_mid["mean"])
        and np.isfinite(gold_mid["se"])
        and np.isfinite(verdict["tercile"]["middle"]["mean"])
        and np.all(np.isfinite(res["gold"]))
        and np.all(np.isfinite(res["delta"]))
    )
    print(
        f"[t3 selftest] N={n} smoke-fit wall-time: {wall:.1f}s  "
        f"gold_mid={gold_mid['mean']:+.3f}±{gold_mid['se']:.3f}  arbiter_mid={verdict['tercile']['middle']['mean']:+.3f}  "
        f"finite_ok={finite_ok}"
    )

    ok = ok_instrument and finite_ok
    print(f"[t3 selftest] {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Real reader -- the N-sweep, both toys, 3 seeds, R=8 restarts per cell.
# ---------------------------------------------------------------------------


def _run(smoke: bool) -> None:
    """Runs the T3 N-sweep (real read, or a reduced smoke check) and writes the summary + verdict."""
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()

    if smoke:
        n_sweep, seeds, n_epochs, m_gold, n_grid, r = (300, 800), (0, 1), 100, 400, 24, 2
    else:
        n_sweep, seeds, n_epochs, m_gold, n_grid, r = N_SWEEP, SEEDS, N_EPOCHS, M_GOLD, N_GRID, R_RESTARTS
    print(f"[t3] smoke={smoke} n_sweep={n_sweep} seeds={seeds} epochs={n_epochs} R={r} m_gold={m_gold} n_grid={n_grid}")

    by_n: dict[int, dict[str, object]] = {}
    control_flat_all_n = True
    crossing_n: int | None = None

    for n in n_sweep:
        print(f"[t3] ===== N_train=N_test={n} =====")
        e_rows = run_toy_n("E", seeds, n, n_epochs, m_gold, n_grid, r, boot_seed_base=n)
        broad_rows = run_toy_n("E_broad", seeds, n, n_epochs, m_gold, n_grid, r, boot_seed_base=n + 1_000_000)

        n_seeds_hump = int(sum(row["gold_mid"]["significant_positive"] for row in e_rows))
        n_seeds_arbiter_recovered = int(sum(row["recovered"] for row in e_rows))
        broad_stays_flat = bool(all((not row["recovered"]) and (row["gold_tercile"]["middle"] <= BROAD_FLAT_TOL) for row in broad_rows))
        control_flat_all_n = control_flat_all_n and broad_stays_flat

        if crossing_n is None and n_seeds_hump >= 2:
            crossing_n = n

        by_n[n] = {
            "E": e_rows,
            "E_broad": broad_rows,
            "n_seeds_hump_significant": n_seeds_hump,
            "n_seeds_arbiter_recovered": n_seeds_arbiter_recovered,
            "broad_stays_flat": broad_stays_flat,
        }
        print(
            f"[t3] N={n}: hump on >=2/3 seeds={n_seeds_hump >= 2} ({n_seeds_hump}/{len(seeds)})  "
            f"arbiter_recovered={n_seeds_arbiter_recovered}/{len(seeds)}  broad_flat={broad_stays_flat}"
        )

    if not control_flat_all_n:
        outcome = "instrument_invalid_control_not_flat_at_some_N"
    elif crossing_n is not None:
        outcome = f"recoverable_at_N={crossing_n}"
    else:
        outcome = f"moving_mode_per_input_count_effectively_absent_up_to_N={n_sweep[-1]}"

    verdict = {
        "outcome": outcome,
        "crossing_n": crossing_n,
        "control_flat_all_n": control_flat_all_n,
        "n_sweep": list(n_sweep),
    }

    summary = {
        "task": "T3",
        "config": {
            "n_sweep": list(n_sweep),
            "seeds": list(seeds),
            "n_epochs": n_epochs,
            "r_restarts": r,
            "m_gold": m_gold,
            "n_grid": n_grid,
            "k_max": x4.hump.K_MAX,
            "broad_flat_tol": BROAD_FLAT_TOL,
            "keep_best_criterion": "train_MAP_objective",
            "smoke": smoke,
        },
        "by_n": {str(n): v for n, v in by_n.items()},
        "verdict": verdict,
        "wall_time_sec": round(time.time() - t_start, 1),
    }

    name = "t3_summary_smoke.json" if smoke else "t3_summary.json"
    out_path = os.path.join(OUT_DIR, name)
    with open(out_path, "w") as f:
        json.dump(x4._jsonable(summary), f, indent=2)
    print(f"\n[t3] wrote {out_path}  ({summary['wall_time_sec']:.0f}s)")
    print("[t3] VERDICT:", json.dumps(x4._jsonable(verdict), indent=2))


def main() -> None:
    """Parses args and dispatches to the selftest, the smoke run, or the real N-sweep reader."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="x4b's instrument oracle (unchanged) + an N=16000 R=1 wall-time smoke-fit; exits with a status code.")
    parser.add_argument("--smoke", action="store_true", help="2 seeds, 2 small N, reduced epochs/R — proves the N-sweep pipeline runs end-to-end; not the real read.")
    args = parser.parse_args()

    torch.set_num_threads(TORCH_THREADS)
    if args.selftest:
        sys.exit(0 if run_selftest() else 1)

    _run(smoke=args.smoke)


if __name__ == "__main__":
    main()
