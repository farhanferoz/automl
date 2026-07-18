"""Emit the report (a) results tables as Markdown from the battery cells.

Reads the 150 cells and prints five tables to stdout, ready to paste into the
report: bin count per target, and — for each of the two target groups — an
accuracy table (squared error, log-likelihood, distribution score) and a
calibration table (coverage, width, interval score, calibration error). Tables
are split by metric family so no single table is too wide to typeset. Best value
per column is bolded (nearest 0.90 for coverage; lowest otherwise; width is never
bolded, since a narrow interval is only good if it covers).

Run AFTER the metric re-run has landed crps/winkler in every cell.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[3] / "automl_package" / "examples" / "report_a_results"
HOME = [("heteroscedastic", "Heteroscedastic"), ("multimodal", "Multimodal"), ("hetero3", "Three-region")]
STD = [("piecewise", "Piecewise"), ("exponential", "Exponential")]
MODELS = [("probreg_fixed_k", "Binning (fixed)"), ("probreg_dynamic_k", "Binning (adaptive)"),
          ("xgboost", "XGBoost"), ("lightgbm", "LightGBM"), ("catboost", "CatBoost"), ("nn_constant", "Plain network")]
SEEDS = range(5)
COMPACT_ABS_MAX = 100.0  # values below this print with 3 decimals; larger ones with 1 (keeps cells narrow)


def mean(toy: str, model: str, key: str) -> float:
    """Mean of one metric over the five seeds for a (target, model) cell."""
    vals = []
    for s in SEEDS:
        d = json.loads((RESULTS / f"{toy}__{model}__seed{s}.json").read_text())
        v = d.get(key)
        if v is not None:
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def fmt(v: float) -> str:
    """Format a value: 3 decimals, 1 decimal for large magnitudes, '--' for NaN."""
    return "--" if v != v else (f"{v:.3f}" if abs(v) < COMPACT_ABS_MAX else f"{v:.1f}")


def emit(title: str, toys: list[tuple[str, str]], cols: list[tuple[str, str, str]]) -> None:
    """Print one Markdown table: models grouped by target, best per column bolded.

    cols: (metric_key, header, mode) with mode in {"min", "target90", "none"}.
    """
    print(f"\n**{title}**\n")
    print("| Target | Model | " + " | ".join(h for _, h, _ in cols) + " |")
    print("|:---|:---|" + "---:|" * len(cols))
    for tkey, tlabel in toys:
        # best per column across models for this toy
        best = {}
        for key, _, mode in cols:
            vals = {m: mean(tkey, m, key) for m, _ in MODELS}
            vals = {m: v for m, v in vals.items() if v == v}
            if not vals:
                best[key] = None
            elif mode == "min":
                best[key] = min(vals, key=vals.get)
            elif mode == "target90":
                best[key] = min(vals, key=lambda m: abs(vals[m] - 0.90))
            else:
                best[key] = None
        for i, (mkey, mlabel) in enumerate(MODELS):
            cells = [tlabel if i == 0 else "", mlabel]
            for key, _, _ in cols:
                v = mean(tkey, mkey, key)
                s = fmt(v)
                if best.get(key) == mkey and s != "--":
                    s = f"**{s}**"
                cells.append(s)
            print("| " + " | ".join(cells) + " |")


# Table 1: bin count per target
print("**Table 1. Bin count selected per target (on a separate held-out draw).**\n")
print("| Target | Bins used |")
print("|:---|---:|")
for tkey, tlabel in HOME + STD:
    ki = json.loads((RESULTS / f"k_selection__{tkey}.json").read_text())
    print(f"| {tlabel} | {ki['selected_k']} |")

ACC = [("mse", "Sq. error", "min"), ("nll", "Neg. log-lik.", "min"), ("crps", "Distrib. score", "min")]
CAL = [("picp@90", "Coverage@90", "target90"), ("mpiw@90", "Width@90", "none"),
       ("winkler@90", "Interval score", "min"), ("pit_ece", "Calib. err", "min")]

emit("Table 2. Home-turf targets — accuracy (lower is better).", HOME, ACC)
emit("Table 3. Home-turf targets — calibration (coverage nearest 0.90; interval score and calib. err lower).", HOME, CAL)
emit("Table 4. Standard targets — accuracy (lower is better).", STD, ACC)
emit("Table 5. Standard targets — calibration (coverage nearest 0.90; interval score and calib. err lower).", STD, CAL)
