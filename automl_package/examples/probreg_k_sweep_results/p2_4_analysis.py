"""P2.4 analysis — runs off the completed P2.3 + #34 CSVs.

Answers the two P2.4 questions:
  Q1: Does dynamic-k match/beat best fixed-k per dataset?
  Q2: Does Cell C + dynamic-k close the exponential gap to Cell B?

Also reports middle-k emptiness trends, ClassReg k-sweep bests, and
cross-cell rankings for §7.7 update.

Run from repo root:
    ~/dev/.venv/bin/python automl_package/examples/probreg_k_sweep_results/p2_4_analysis.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent
PROBREG = pd.read_csv(RESULTS_DIR / "results.csv")
SUMMARY = pd.read_csv(RESULTS_DIR / "summary.csv")
CLASSREG_DIR = RESULTS_DIR.parent / "classreg_k_sweep_results"
CLASSREG = pd.read_csv(CLASSREG_DIR / "k_sweep_summary.csv")

DATASETS = ["heteroscedastic", "bimodal", "piecewise", "exponential"]


def q1_dynamic_vs_fixed() -> None:
    """Does dynamic-k match/beat best fixed-k per dataset, per cell?"""
    print("\n" + "=" * 78)
    print("Q1: Does dynamic-k match/beat best fixed-k per dataset?")
    print("=" * 78)
    for ds in DATASETS:
        print(f"\n--- {ds} ---")
        sub = SUMMARY[(SUMMARY["dataset"] == ds) & SUMMARY["mse_mean"].notna()].copy()
        for cell in ("B", "C"):
            cs = sub[sub["cell"] == cell].copy()
            if cs.empty:
                print(f"  Cell {cell}: no valid rows")
                continue
            fixed = cs[cs["dynamic"] == "NONE"].sort_values("mse_mean")
            dyn = cs[cs["dynamic"] != "NONE"].sort_values("mse_mean")
            best_fixed = fixed.iloc[0] if not fixed.empty else None
            best_dyn = dyn.iloc[0] if not dyn.empty else None
            if best_fixed is None or best_dyn is None:
                continue
            delta = best_dyn["mse_mean"] - best_fixed["mse_mean"]
            flag = "dyn WINS" if delta < -0.005 else ("tied" if abs(delta) <= 0.02 else "fixed wins")
            print(f"  Cell {cell}: fixed best = k={int(best_fixed['k_max'])} "
                  f"MSE={best_fixed['mse_mean']:.4f} | "
                  f"dyn best = k_max={int(best_dyn['k_max'])}/{best_dyn['dynamic']}/{best_dyn['k_reg']} "
                  f"MSE={best_dyn['mse_mean']:.4f} eff_k={best_dyn['effective_k_mean']:.2f}  "
                  f"Δ={delta:+.4f} → {flag}")


def q2_cell_c_exponential_gap() -> None:
    """Does Cell C + dynamic-k close the exponential gap to Cell B?"""
    print("\n" + "=" * 78)
    print("Q2: Does Cell C + dynamic-k close the exponential gap to Cell B?")
    print("=" * 78)
    exp = SUMMARY[(SUMMARY["dataset"] == "exponential") & SUMMARY["mse_mean"].notna()].copy()
    best_b = exp[exp["cell"] == "B"].sort_values("mse_mean").iloc[0]
    best_c_fixed = exp[(exp["cell"] == "C") & (exp["dynamic"] == "NONE")].sort_values("mse_mean").iloc[0]
    c_dyn = exp[(exp["cell"] == "C") & (exp["dynamic"] != "NONE")]
    best_c_dyn = c_dyn.sort_values("mse_mean").iloc[0] if not c_dyn.empty else None

    print(f"\n  Cell B best:         MSE={best_b['mse_mean']:.4f} "
          f"({best_b['dynamic']}|{best_c_fixed['k_reg']}|k={int(best_b['k_max'])})")
    print(f"  Cell C fixed-k best: MSE={best_c_fixed['mse_mean']:.4f} "
          f"(k={int(best_c_fixed['k_max'])})")
    if best_c_dyn is not None:
        pre_gap = best_c_fixed["mse_mean"] - best_b["mse_mean"]
        new_gap = best_c_dyn["mse_mean"] - best_b["mse_mean"]
        print(f"  Cell C dynamic best: MSE={best_c_dyn['mse_mean']:.4f} "
              f"(k_max={int(best_c_dyn['k_max'])}|{best_c_dyn['dynamic']}|"
              f"{best_c_dyn['k_reg']}|eff_k={best_c_dyn['effective_k_mean']:.2f})")
        print(f"  Pre-gap (C_fixed - B_best)  = {pre_gap:+.4f}")
        print(f"  New-gap (C_dyn   - B_best)  = {new_gap:+.4f}")
        closed_pct = (1 - new_gap / pre_gap) * 100 if pre_gap > 0 else float("nan")
        print(f"  Gap closed: {closed_pct:.0f}%")


def middle_k_emptiness() -> None:
    """Does middle-k emptiness (max_p_mid → low) rise with k_max in B?"""
    print("\n" + "=" * 78)
    print("Middle-k emptiness (lower max_p_mid = more empty middle bins)")
    print("=" * 78)
    for ds in DATASETS:
        sub = SUMMARY[(SUMMARY["dataset"] == ds) & (SUMMARY["cell"] == "B")
                      & (SUMMARY["dynamic"] == "NONE") & SUMMARY["max_p_mid_mean"].notna()].sort_values("k_max")
        if sub.empty:
            continue
        vals = "  ".join(f"k={int(r['k_max'])}:{r['max_p_mid_mean']:.3f}" for _, r in sub.iterrows())
        print(f"  {ds:16s} {vals}")


def effective_k_contraction() -> None:
    """Does dynamic selection shrink effective_k vs k_max?"""
    print("\n" + "=" * 78)
    print("Effective-k contraction under SOFT_GATING (cell B, most reliable)")
    print("=" * 78)
    for ds in DATASETS:
        print(f"\n  {ds}:")
        for k_max in sorted(SUMMARY[SUMMARY["dataset"] == ds]["k_max"].unique()):
            sub = SUMMARY[(SUMMARY["dataset"] == ds) & (SUMMARY["cell"] == "B")
                          & (SUMMARY["k_max"] == k_max) & (SUMMARY["dynamic"] == "SOFT_GATING")
                          & SUMMARY["effective_k_mean"].notna()].copy()
            if sub.empty:
                continue
            parts = [f"k_max={int(k_max)}"]
            for _, r in sub.iterrows():
                parts.append(f"{r['k_reg']}:eff_k={r['effective_k_mean']:.2f}")
            print("    " + "  ".join(parts))


def classreg_cross_reference() -> None:
    """ClassReg k-sweep: best k per dataset + DirectNN anchor."""
    print("\n" + "=" * 78)
    print("ClassReg k-sweep (task #34) cross-reference")
    print("=" * 78)
    for ds in DATASETS:
        cr = CLASSREG[(CLASSREG["dataset"] == ds) & (CLASSREG["model"] == "ClassReg")].sort_values("mse_mean")
        direct = CLASSREG[(CLASSREG["dataset"] == ds) & (CLASSREG["model"] == "DirectNN")]
        if cr.empty or direct.empty:
            continue
        best_cr = cr.iloc[0]
        dnn = direct.iloc[0]
        print(f"  {ds:16s} ClassReg best: k={int(best_cr['k'])} MSE={best_cr['mse_mean']:.4f} "
              f"NLL={best_cr['nll_gaussian_mean']:.3f} | "
              f"DirectNN: MSE={dnn['mse_mean']:.4f} NLL={dnn['nll_gaussian_mean']:.3f}")


def cross_cell_top3() -> None:
    """Top-3 configs per dataset across all cells, on MSE."""
    print("\n" + "=" * 78)
    print("Top-3 (dataset, cell, k_max, dynamic, k_reg) by MSE")
    print("=" * 78)
    for ds in DATASETS:
        sub = SUMMARY[(SUMMARY["dataset"] == ds) & SUMMARY["mse_mean"].notna()].sort_values("mse_mean").head(3)
        print(f"\n  {ds}:")
        for _, r in sub.iterrows():
            print(f"    cell={r['cell']} k_max={int(r['k_max'])} "
                  f"{r['dynamic']}/{r['k_reg']}  "
                  f"MSE={r['mse_mean']:.4f}±{r['mse_std']:.4f}  "
                  f"NLL={r['nll_gaussian_mean']:.3f}  "
                  f"eff_k={r['effective_k_mean']:.2f}  "
                  f"max_p_mid={r['max_p_mid_mean']:.3f}")


def cell_c_completion_check() -> None:
    """Sanity: no more NaN rows after re-run."""
    print("\n" + "=" * 78)
    print("Completion sanity check")
    print("=" * 78)
    total = len(PROBREG)
    valid = PROBREG["mse"].notna().sum()
    cell_c_dyn = PROBREG[(PROBREG["cell"] == "C") & (PROBREG["dynamic"] != "NONE")]
    print(f"  Total rows: {total}/840")
    print(f"  Valid (mse not NaN): {valid}")
    print(f"  Cell C + non-NONE dynamic: {len(cell_c_dyn)} rows, "
          f"{cell_c_dyn['mse'].notna().sum()} valid")
    if cell_c_dyn["mse"].isna().sum() > 0:
        failing = cell_c_dyn[cell_c_dyn["mse"].isna()][["dataset", "k_max", "dynamic", "k_reg", "seed"]]
        print(f"  ⚠ Still failing ({len(failing)}):")
        print(failing.to_string(index=False))


if __name__ == "__main__":
    cell_c_completion_check()
    q1_dynamic_vs_fixed()
    q2_cell_c_exponential_gap()
    middle_k_emptiness()
    effective_k_contraction()
    classreg_cross_reference()
    cross_cell_top3()
