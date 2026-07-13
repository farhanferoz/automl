"""Figure for the k-selection note: the held-out readout recovers the per-input count, and the
prefer-few prior earns nothing on a like-for-like trial.

Reads the two result files this session produced and renders a two-panel summary figure into
this directory (`comparison_ablation.png`), so the note can show what Sections 14-15 report.

Left panel  (comparison, Section 14): the held-out advantage of the flexible mixture over a
single bell-curve, at probe inputs of known true count, across the three families. Bars are
coloured by true count; the reading is near zero at a true count of one and clearly positive
above it.

Right panel (prior ablation, Section 15): the held-out advantage on the hump family (true count
1, 2, 1) for the prior on/off in the moving-class and fixed-class regimes, and the plain mixture.
The moving-class readings hump together; the prior under fixed tiles fails at the middle.

Run with any matplotlib+numpy interpreter:
    python3 docs/kselection_variational_em_2026-06-13/plot_comparison_ablation.py
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
_CMP = os.path.join(_REPO, "automl_package/examples/variational_em_comparison_results/results.json")
_ABL = os.path.join(_REPO, "automl_package/examples/prior_ablation_results/results.json")
_OUT = os.path.join(_HERE, "comparison_ablation.png")

TOY_LABEL = {"C": "rising spacing", "D": "staircase", "E": "hump"}
COUNT_COLOR = {1: "#9e9e9e", 2: "#1f77b4", 3: "#08306b"}


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def panel_comparison(ax: plt.Axes, cmp_data: dict) -> None:
    """Bars of the held-out readout at each probe input, grouped by family, coloured by true count."""
    toys = [t for t in cmp_data["toys"] if t["toy"] in TOY_LABEL]
    x, bars, colors, ticklabels, counts, spans = [], [], [], [], [], []
    pos = 0.0
    for t in toys:
        start = pos
        for p in t["probes"]:
            x.append(pos)
            bars.append(p["our_readout"])
            colors.append(COUNT_COLOR.get(p["true_count"], "#1f77b4"))
            ticklabels.append(f"{p['input']}")
            counts.append(p["true_count"])
            pos += 1.0
        spans.append((start, pos - 1.0, TOY_LABEL[t["toy"]]))
        pos += 1.0  # gap between families
    ax.axhline(0.0, color="k", lw=0.8)
    ax.bar(x, bars, width=0.82, color=colors, edgecolor="black", linewidth=0.4)
    for xi, b, c in zip(x, bars, counts, strict=True):
        ax.annotate(str(c), (xi, b), textcoords="offset points",
                    xytext=(0, 3 if b >= 0 else -11), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(ticklabels, fontsize=8)
    # family names below the tick numbers, with light separators in the gaps between groups
    for i, (s, e, name) in enumerate(spans):
        ax.text((s + e) / 2.0, -0.14, name, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9, fontweight="bold")
        if i < len(spans) - 1:
            ax.axvline(e + 1.0, color="0.8", lw=0.8)
    ax.set_ylabel("held-out advantage of the mixture (nats)")
    ax.set_title("Section 14: the readout recovers the per-input count\n(number above each bar = true count)", fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COUNT_COLOR[k]) for k in (1, 2, 3)]
    ax.legend(handles, ["true count 1", "true count 2", "true count 3"], fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.25)


def panel_ablation(ax: plt.Axes, abl_data: dict) -> None:
    """Grouped bars of the held-out readout on the hump family for prior on/off, both regimes, and plain mixture."""
    hump = next(t for t in abl_data["toys"] if t["toy"] == "E")
    probes = hump["probes"]
    xt = [f"{p['input']}\n(count {p['true_count']})" for p in probes]
    series = [
        ("agg_adaptive_prior", "prior on (moving)", "#1f77b4"),
        ("agg_adaptive_noprior", "prior off (moving)", "#7fb3e0"),
        ("mdn_kmax", "plain mixture", "#2ca02c"),
        ("agg_fixed_prior", "prior on (fixed)", "#d62728"),
        ("agg_fixed_noprior", "prior off (fixed)", "#f4a6a6"),
    ]
    n = len(series)
    idx = np.arange(len(probes))
    width = 0.84 / n
    ax.axhline(0.0, color="k", lw=0.8)
    for j, (key, label, color) in enumerate(series):
        vals = [p["readout"][key] for p in probes]
        ax.bar(idx + (j - (n - 1) / 2.0) * width, vals, width=width, label=label,
               color=color, edgecolor="black", linewidth=0.3)
    ax.set_xticks(idx)
    ax.set_xticklabels(xt, fontsize=9)
    ax.set_xlabel("input")
    ax.set_ylabel("held-out advantage of the mixture (nats)")
    ax.set_title("Section 15: on the hump family, the prior earns nothing (moving), fails (fixed)", fontsize=11)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(axis="y", alpha=0.25)


def main() -> None:
    """Render the two-panel summary figure."""
    cmp_data, abl_data = _load(_CMP), _load(_ABL)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    panel_comparison(axes[0], cmp_data)
    panel_ablation(axes[1], abl_data)
    fig.tight_layout()
    fig.savefig(_OUT, dpi=130, bbox_inches="tight")
    print(f"saved -> {_OUT}")


if __name__ == "__main__":
    main()
