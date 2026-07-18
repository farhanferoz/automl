"""Generate the report (a) figures from the report battery cells.

Reads `automl_package/examples/report_a_results/{toy}__{model}__seed{s}.json`
(150 cells) and writes three PNGs into `figures/`:

  probreg-coverage.png    grouped bars: 90% interval coverage per model per toy (target 0.90)
  probreg-reliability.png reliability diagram: observed vs stated coverage (home-turf average)
  probreg-tradeoff.png    scatter: coverage vs interval width, per (model, toy)

Run AFTER the metric re-run has landed crps/winkler/reliability_curve in every cell.
Colours: Okabe-Ito colour-blind-safe palette; the two probabilistic-regression
variants are highlighted, the four baselines muted.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parents[3] / "automl_package" / "examples" / "report_a_results"
FIGDIR = Path(__file__).resolve().parent / "figures"
FIGDIR.mkdir(exist_ok=True)

TOYS = ["heteroscedastic", "multimodal", "hetero3", "piecewise", "exponential"]
TOY_LABEL = {"heteroscedastic": "Heterosced.", "multimodal": "Multimodal", "hetero3": "Three-region",
             "piecewise": "Piecewise", "exponential": "Exponential"}
HOME = ["heteroscedastic", "multimodal", "hetero3"]
MODELS = ["probreg_fixed_k", "probreg_dynamic_k", "xgboost", "lightgbm", "catboost", "nn_constant"]
LABEL = {"probreg_fixed_k": "Binning (fixed)", "probreg_dynamic_k": "Binning (adaptive)",
         "xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost", "nn_constant": "Plain network"}
COLOR = {"probreg_fixed_k": "#0072B2", "probreg_dynamic_k": "#D55E00", "xgboost": "#009E73",
         "lightgbm": "#56B4E9", "catboost": "#CC79A7", "nn_constant": "#999999"}
SEEDS = range(5)

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def cell(toy: str, model: str, seed: int) -> dict:
    return json.loads((RESULTS / f"{toy}__{model}__seed{seed}.json").read_text())


def mean_metric(toy: str, model: str, key: str) -> float:
    vals = [cell(toy, model, s).get(key) for s in SEEDS]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else np.nan


def fig_coverage() -> None:
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    x = np.arange(len(TOYS))
    w = 0.13
    for i, model in enumerate(MODELS):
        vals = [mean_metric(t, model, "picp@90") for t in TOYS]
        ax.bar(x + (i - 2.5) * w, vals, w, label=LABEL[model], color=COLOR[model])
    ax.axhline(0.90, color="black", lw=1.2, ls="--")
    ax.text(len(TOYS) - 0.5, 0.905, "target 0.90", ha="right", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([TOY_LABEL[t] for t in TOYS])
    ax.set_ylabel("90% interval coverage")
    ax.set_ylim(0.65, 1.02)
    ax.legend(ncol=3, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.42), frameon=False)
    fig.savefig(FIGDIR / "probreg-coverage.png")
    plt.close(fig)


def fig_reliability() -> None:
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    target = np.array(cell("heteroscedastic", "probreg_fixed_k", 0)["reliability_curve"]["target"])
    for model in MODELS:
        curves = []
        for t in HOME:
            for s in SEEDS:
                rc = cell(t, model, s).get("reliability_curve")
                if rc:
                    curves.append(rc["empirical"])
        if curves:
            ax.plot(target, np.mean(curves, axis=0), color=COLOR[model], label=LABEL[model], lw=1.8)
    ax.plot([0, 1], [0, 1], color="black", lw=1.0, ls="--")
    ax.text(0.62, 0.55, "perfect calibration", rotation=38, fontsize=8, color="black")
    ax.set_xlabel("stated probability")
    ax.set_ylabel("observed frequency")
    ax.set_title("Reliability diagram (home-turf average)", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    fig.savefig(FIGDIR / "probreg-reliability.png")
    plt.close(fig)


def fig_tradeoff() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for model in MODELS:
        xs = [mean_metric(t, model, "mpiw@90") for t in TOYS]
        ys = [mean_metric(t, model, "picp@90") for t in TOYS]
        ax.scatter(xs, ys, color=COLOR[model], label=LABEL[model], s=46, edgecolor="white", lw=0.6, zorder=3)
    ax.axhline(0.90, color="black", lw=1.1, ls="--")
    ax.text(0.02, 0.905, "coverage target 0.90", transform=ax.get_yaxis_transform(), fontsize=8, va="bottom")
    ax.set_xlabel("90% interval width  (← narrower)")
    ax.set_ylabel("90% interval coverage")
    ax.set_title("Below the line = intervals too narrow to cover", fontsize=10)
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    fig.savefig(FIGDIR / "probreg-tradeoff.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_coverage()
    fig_reliability()
    fig_tradeoff()
    print(f"wrote figures to {FIGDIR}")
