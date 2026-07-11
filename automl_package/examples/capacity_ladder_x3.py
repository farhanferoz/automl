"""X3: repeated cross-fit re-issue of the F3 per-bin verdicts (WS2 capacity-ladder follow-up, task X3).

(docs/plans/capacity_ladder_2026-07-09/EXECUTION_PLAN.md §8.5 X3 / §8.6; pre-registration in
 capacity_ladder_results/X3/PREREGISTRATION.md)

Fixes the W4 review finding (§8.2): the F3 toy-G per-input depth advantage recorded as "+2.67 SE"
is SPLIT-FRAGILE — it survives only 3/9 re-randomized fit/score partitions, and R3's robustness
check varied only the bootstrap seed, not the fit/score split. X3 averages the per-bin advantage
over S random 50/50 fit/score splits and reports a SPLIT-AWARE pooled SE (Nadeau & Bengio 2003
corrected resampled estimator), then re-issues the G / G-flat / H per-bin verdicts.

Pure post-hoc: no training. Reuses F3's validated `_perbin_vs_global` reader VERBATIM
(`capacity_ladder_f3.py`), only sweeping its `split_seed`. F2's nested-depth ladder tables
(`capacity_ladder_results/F2/nested_toy{G,G_flat,H}_seed{0,1,2}.pt`) are the input, exactly as F3.

Estimator (locked in PREREGISTRATION.md):
  per (toy, seed): S=50 splits; per split s -> mean_diff_s (tercile per-bin advantage over global,
  held-out log-score) + per-split paired-bootstrap se_s + beats_s = mean_diff_s > 2*se_s.
    point estimate  mu_bar   = mean_s mean_diff_s
    split-aware SE  se_nb    = sqrt( (1/S + n_score/n_fit) * Var_s(mean_diff_s) )   [50/50 -> ~= SD_s]
    corrected t     t        = mu_bar / se_nb
    pass fraction   k/S      = mean_s(beats_s)
  Naive SD_s/sqrt(S) is anti-conservative for overlapping splits and is NOT used.

Usage:
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/capacity_ladder_x3.py --selftest
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/capacity_ladder_x3.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root, so `import automl_package` works

import _capacity_ladder as cl  # noqa: E402
from capacity_ladder_f3 import (  # noqa: E402
    C_GRID,
    SEEDS,
    TOYS,
    _jsonable,
    _perbin_vs_global,
    load_f2_table,
)

OUT_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "X3")

N_SPLITS = 50  # random 50/50 fit/score splits per (toy, seed); split_seed = 0..N_SPLITS-1


# ---------------------------------------------------------------------------
# Aggregation over repeated cross-fit splits.
# ---------------------------------------------------------------------------


def run_repeated_crossfit(score: np.ndarray, x: np.ndarray, n_splits: int = N_SPLITS, n_bins: int = 3) -> dict:
    """Sweeps `_perbin_vs_global` over `n_splits` random fit/score splits and aggregates.

    Args:
        score: `(N, C)` held-out score table for one (toy, seed) F2 case.
        x: `(N,)` inputs.
        n_splits: number of random 50/50 fit/score partitions (`split_seed = 0..n_splits-1`).
        n_bins: quantile bins for the per-bin stack (3 = terciles, F3 convention).

    Returns:
        Aggregated dict: point estimate, Nadeau-Bengio split-aware SE, corrected t, split-pass
        fraction, the raw per-split `mean_diff`/`se`/`beats`, and (for H) the SNR-trend fractions.
    """
    per_split = [_perbin_vs_global(score, x, n_bins, split_seed=s) for s in range(n_splits)]

    mean_diffs = np.array([r["mean_diff"] for r in per_split], dtype=np.float64)
    se_within = np.array([r["se_diff"] for r in per_split], dtype=np.float64)
    beats = np.array([bool(r["beats_global_2se"]) for r in per_split])

    # Nadeau-Bengio (2003) corrected resampled SE of the mean of overlapping-split metrics:
    #   Var_corr = (1/S + n_test/n_train) * sample_var(metric_s).  For 50/50, n_test/n_train ~= 1.
    n_fit = per_split[0]["n_fit"]
    n_score = per_split[0]["n_score"]
    rho = n_score / n_fit
    var_s = float(np.var(mean_diffs, ddof=1))
    mu_bar = float(mean_diffs.mean())
    se_nb = float(np.sqrt(max(0.0, (1.0 / n_splits + rho) * var_s)))
    t_corr = mu_bar / se_nb if se_nb > 0 else float("inf")
    se_naive = float(np.sqrt(var_s / n_splits))  # reported ONLY to show how much smaller (rejected) it is

    # H-specific: per-split tercile argmax capacity in the high-SNR (lowest-x) vs low-SNR (highest-x) bin.
    trend_down = []
    varies = []
    for r in per_split:
        pi_bins = r["pi_bins"]
        bin_ids = sorted(int(b) for b in pi_bins)
        if len(bin_ids) < 2:
            continue
        argmax_lo = C_GRID[int(np.argmax(pi_bins[str(bin_ids[0])]))]   # lowest-x tercile = high SNR (sigma_low)
        argmax_hi = C_GRID[int(np.argmax(pi_bins[str(bin_ids[-1])]))]  # highest-x tercile = low SNR (sigma_high)
        trend_down.append(argmax_lo > argmax_hi)
        varies.append(argmax_lo != argmax_hi)

    return {
        "n_splits": n_splits,
        "n_fit": int(n_fit),
        "n_score": int(n_score),
        "rho_test_over_train": rho,
        "mu_bar": mu_bar,
        "sd_across_splits": float(np.sqrt(var_s)),
        "se_nadeau_bengio": se_nb,
        "se_naive_rejected": se_naive,
        "t_corrected": t_corr,
        "split_pass_fraction": float(beats.mean()),
        "mean_diff_min": float(mean_diffs.min()),
        "mean_diff_max": float(mean_diffs.max()),
        "frac_negative": float((mean_diffs < 0).mean()),
        "per_split_mean_diff": mean_diffs.tolist(),
        "per_split_se": se_within.tolist(),
        "h_trend_down_fraction": (float(np.mean(trend_down)) if trend_down else None),
        "h_varies_fraction": (float(np.mean(varies)) if varies else None),
    }


# ---------------------------------------------------------------------------
# Selftest — synthetic known-answer discrimination (real per-bin structure vs none).
# ---------------------------------------------------------------------------


def _synthetic_table(peaks: tuple[int, int, int], rng: np.random.Generator, n_per: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Three equal x-regions whose score tables peak at `peaks[0..2]` (reuses `cl._peaked_table`)."""
    x = np.concatenate([
        rng.uniform(-1.0, -1.0 / 3.0, n_per),
        rng.uniform(-1.0 / 3.0, 1.0 / 3.0, n_per),
        rng.uniform(1.0 / 3.0, 1.0, n_per),
    ])
    score = np.concatenate([cl._peaked_table(n_per, peak_c=pc, gap=4.0, noise_sd=0.3, rng=rng) for pc in peaks], axis=0)
    return score, x


def run_selftest() -> bool:
    """Known-answer check that the repeated-crossfit machinery DISCRIMINATES per-bin structure from none.

    Positive table: capacities 1/3/6 peak in the three x-regions -> the per-bin stack should beat
    global on (almost) every split (pass-fraction ~ 1, corrected t >> 2). Negative table: the SAME
    capacity peaks everywhere (no per-bin structure) -> per-bin must NOT beat global (pass-fraction
    near 0, corrected t <= 2). This validates the X3 verdict logic before any real F2 table is read.
    """
    n_splits = 20
    rng = np.random.default_rng(0)
    pos_score, pos_x = _synthetic_table((1, 3, 6), rng)
    neg_score, neg_x = _synthetic_table((3, 3, 3), rng)

    pos = run_repeated_crossfit(pos_score, pos_x, n_splits=n_splits)
    neg = run_repeated_crossfit(neg_score, neg_x, n_splits=n_splits)

    print(f"[x3 selftest] POSITIVE (per-region 1/3/6): mu_bar={pos['mu_bar']:+.4f} t={pos['t_corrected']:.2f} "
          f"pass={pos['split_pass_fraction']:.2f}")
    print(f"[x3 selftest] NEGATIVE (flat 3/3/3):        mu_bar={neg['mu_bar']:+.4f} t={neg['t_corrected']:.2f} "
          f"pass={neg['split_pass_fraction']:.2f}")

    ok_pos = pos["split_pass_fraction"] > 0.8 and pos["t_corrected"] > 3.0 and pos["mu_bar"] > 0
    ok_neg = neg["split_pass_fraction"] < 0.2 and neg["t_corrected"] <= 2.0
    ok = ok_pos and ok_neg
    print(f"[x3 selftest] positive-detected={ok_pos}  negative-abstains={ok_neg}  {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Real reader — repeated cross-fit on whatever F2 tables exist.
# ---------------------------------------------------------------------------


def _pooled_across_seeds(seed_rows: list[dict]) -> dict:
    """Pools the per-seed point estimates: between-seed mean +/- SD of mu_bar and a 3-seed t (2 df)."""
    mus = np.array([r["agg"]["mu_bar"] for r in seed_rows], dtype=np.float64)
    n = len(mus)
    mean_mu = float(mus.mean())
    sd_mu = float(np.std(mus, ddof=1)) if n > 1 else float("nan")
    se_between = sd_mu / np.sqrt(n) if n > 1 else float("nan")
    t_between = mean_mu / se_between if (n > 1 and se_between > 0) else float("nan")
    return {
        "n_seeds": n,
        "mu_bar_by_seed": mus.tolist(),
        "mean_mu_bar": mean_mu,
        "sd_mu_bar_between_seeds": sd_mu,
        "t_between_seeds_2df": t_between,
        "t_corrected_by_seed": [r["agg"]["t_corrected"] for r in seed_rows],
        "split_pass_fraction_by_seed": [r["agg"]["split_pass_fraction"] for r in seed_rows],
        "n_seeds_t_gt_2": int(sum(r["agg"]["t_corrected"] > 2.0 for r in seed_rows)),
    }


def _run_reader(n_splits: int = N_SPLITS) -> None:
    """Runs repeated cross-fit on every available F2 table and writes `x3_summary.json`."""
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()

    print("loading F2 tables...")
    tables: dict[tuple[str, int], dict] = {}
    missing: list[tuple[str, int]] = []
    for toy in TOYS:
        for seed in SEEDS:
            t = load_f2_table(toy, seed)
            if t is None:
                missing.append((toy, seed))
                continue
            tables[(toy, seed)] = t
    if missing:
        print(f"  missing F2 tables (skipped): {missing}")
    if not tables:
        print("no F2 tables available -- run capacity_ladder_f2.py first.")
        return

    per_case = []
    for (toy, seed), t in sorted(tables.items()):
        agg = run_repeated_crossfit(t["score"], t["x"], n_splits=n_splits)
        per_case.append({"toy": toy, "seed": seed, "n": int(t["score"].shape[0]), "agg": agg})
        extra = "" if agg["h_trend_down_fraction"] is None else f" h_trend_down={agg['h_trend_down_fraction']:.2f}"
        print(f"[x3] {toy} seed{seed}: mu_bar={agg['mu_bar']:+.4f}  t_corr={agg['t_corrected']:+.2f}  "
              f"pass={agg['split_pass_fraction']:.2f}  (SD_split={agg['sd_across_splits']:.4f}, "
              f"naive_se={agg['se_naive_rejected']:.4f}){extra}")

    pooled = {}
    for toy in TOYS:
        rows = [r for r in per_case if r["toy"] == toy]
        if rows:
            pooled[toy] = _pooled_across_seeds(rows)

    # Re-issued verdicts.
    verdicts = {}
    g = pooled.get("G")
    gf = pooled.get("G_flat")
    if g and gf:
        g_pass_fracs = np.array(g["split_pass_fraction_by_seed"])
        gf_pass_fracs = np.array(gf["split_pass_fraction_by_seed"])
        verdicts["G_signal"] = {
            "n_seeds_t_gt_2": g["n_seeds_t_gt_2"],
            "mean_pass_fraction": float(g_pass_fracs.mean()),
            "gflat_mean_pass_fraction": float(gf_pass_fracs.mean()),
            "separates_from_control": bool(g_pass_fracs.mean() - gf_pass_fracs.mean() > 0.2 and g["n_seeds_t_gt_2"] >= 2),
        }
        verdicts["G_flat_no_false_positive"] = {
            "n_seeds_t_gt_2": gf["n_seeds_t_gt_2"],
            "mean_pass_fraction": float(gf_pass_fracs.mean()),
            "holds": bool(gf["n_seeds_t_gt_2"] == 0 and gf_pass_fracs.mean() < 0.2),
        }
    h = pooled.get("H")
    if h:
        h_rows = [r for r in per_case if r["toy"] == "H"]
        td = np.array([r["agg"]["h_trend_down_fraction"] for r in h_rows if r["agg"]["h_trend_down_fraction"] is not None])
        verdicts["H_snr_dial"] = {
            "trend_down_fraction_by_seed": td.tolist(),
            "mean_trend_down_fraction": float(td.mean()) if td.size else None,
            "stable_majority": bool(td.size and (td > 0.5).all()),
        }

    wall = time.time() - t_start
    summary = {
        "task": "X3",
        "n_splits": n_splits,
        "se_estimator": "Nadeau-Bengio 2003 corrected resampled (1/S + n_test/n_train)*Var_s",
        "missing": _jsonable(missing),
        "per_case": per_case,
        "pooled_across_seeds": pooled,
        "verdicts": verdicts,
        "wall_time_sec": wall,
    }
    out_path = os.path.join(OUT_DIR, "x3_summary.json")
    with open(out_path, "w") as f:
        json.dump(_jsonable(summary), f, indent=2)
    print()
    print(f"wrote {out_path}  ({wall:.1f}s)")
    print("VERDICTS:", json.dumps(_jsonable(verdicts), indent=2))


def main() -> None:
    """Parses args and either runs the synthetic selftest or the repeated cross-fit reader."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="Run the synthetic known-answer discrimination selftest and exit.")
    parser.add_argument("--n-splits", type=int, default=N_SPLITS, help="Number of random fit/score splits (default 50).")
    args = parser.parse_args()

    if args.selftest:
        sys.exit(0 if run_selftest() else 1)

    _run_reader(n_splits=args.n_splits)


if __name__ == "__main__":
    main()
