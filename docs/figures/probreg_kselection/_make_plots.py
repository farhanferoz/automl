"""Generate all figures for docs/probreg_kselection_findings.md.

Run:
    ~/dev/.venv/bin/python /home/ff235/dev/MLResearch/automl/docs/figures/probreg_kselection/_make_plots.py
"""

from __future__ import annotations

import ast
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.use("Agg")
mpl.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 130,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

EXAMPLES = Path("/home/ff235/dev/MLResearch/automl/automl_package/examples")
EXP = EXAMPLES / "probreg_kselection_experiments_results"
ELBO = EXAMPLES / "probreg_elbo_prior_check_results"
DIAG = EXAMPLES / "probreg_kselection_diagnostic_results"
OUT = Path("/home/ff235/dev/MLResearch/automl/docs/figures/probreg_kselection")
OUT.mkdir(parents=True, exist_ok=True)


def fig1_cap_tracking() -> None:
    df = pd.read_csv(EXP / "sweep1_kmax.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    k_grid = np.array([5, 10, 20, 40])
    uniform_E = (k_grid + 2) / 2
    uniform_perp = k_grid - 1

    palette = {"toy_a": "#2E7D9A", "toy_b_kT3": "#C04B3B"}
    label = {"toy_a": "Toy A (no intrinsic k)", "toy_b_kT3": "Toy B (K_true=3)"}

    for ax_idx, metric, ref in [(0, "effective_k_nobypass", uniform_E), (1, "selection_perplexity", uniform_perp)]:
        ax = axes[ax_idx]
        for ds in ["toy_a", "toy_b_kT3"]:
            sub = df[df["dataset"] == ds]
            g = sub.groupby("k_max")[metric].agg(["mean", "std"]).reset_index()
            ax.errorbar(g["k_max"], g["mean"], yerr=g["std"], marker="o", linewidth=1.7,
                        capsize=3, label=label[ds], color=palette[ds])
        ax.plot(k_grid, ref, ":", color="#666", label="uniform reference")
        if metric == "effective_k_nobypass":
            ax.axhline(3, color="#888", ls="--", lw=1, label="K_true=3")
            ax.set_ylabel("effective_k_nobypass = E[k]")
            ax.set_title("Cap tracking: E[k|nb] scales with k_max on both toys")
        else:
            ax.set_ylabel("selection_perplexity = exp(H)")
            ax.set_title("Perplexity also grows with k_max (but stays below uniform)")
        ax.set_xlabel("k_max")
        ax.set_xticks(k_grid)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "fig1_cap_tracking.png", bbox_inches="tight")
    plt.close()


def fig2_noise_mechanism() -> None:
    df = pd.read_csv(EXP / "sweep2_noise.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    g = df.groupby("sigma")[["bypass_fraction", "mean_max_p", "selection_perplexity"]].agg(["mean", "std"]).reset_index()
    g.columns = ["sigma"] + [f"{a}_{b}" for a, b in g.columns[1:]]

    ax = axes[0]
    ax.errorbar(g["sigma"], g["bypass_fraction_mean"], yerr=g["bypass_fraction_std"],
                marker="o", color="#C04B3B", capsize=3, label="bypass_fraction")
    ax.errorbar(g["sigma"], g["mean_max_p_mean"], yerr=g["mean_max_p_std"],
                marker="s", color="#2E7D9A", capsize=3, label="mean_max_p")
    ax.set_xscale("log")
    ax.set_xlabel("noise σ")
    ax.set_ylabel("fraction / probability")
    ax.set_title("Bypass-handoff confirmed; CE-confidence rejected")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    ax = axes[1]
    ax.errorbar(g["sigma"], g["selection_perplexity_mean"], yerr=g["selection_perplexity_std"],
                marker="o", color="#7B4FA1", capsize=3)
    ax.set_xscale("log")
    ax.set_xlabel("noise σ")
    ax.set_ylabel("selection_perplexity")
    ax.set_title("k-mode distribution shape ~invariant to noise")
    ax.set_ylim(0, 9)
    ax.axhline(9, color="#888", ls=":", lw=1, label="uniform=9")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(OUT / "fig2_noise.png", bbox_inches="tight")
    plt.close()


def fig3_regulariser_compare() -> None:
    """Bar chart comparing E[k|nb] across {NONE, K_PENALTY, ELBO uniform, ELBO geom λ=0.5}."""
    s3 = pd.read_csv(EXP / "sweep3_kreg.csv")
    s_elbo = pd.read_csv(ELBO / "results.csv")

    rows = []
    for k_reg in ["NONE", "K_PENALTY"]:
        sub = s3[s3["k_reg"] == k_reg]
        rows.append({
            "label": k_reg, "color": "#7B4FA1" if k_reg == "K_PENALTY" else "#888",
            "E_k_mean": sub["effective_k_nobypass"].mean(),
            "E_k_std": sub["effective_k_nobypass"].std(),
            "perp_mean": sub["selection_perplexity"].mean(),
            "perp_std": sub["selection_perplexity"].std(),
            "max_p_mean": sub["mean_max_p"].mean(),
            "bypass_mean": sub["bypass_fraction"].mean(),
        })
    for prior_type, lam in [("uniform", 0.0), ("geometric", 0.5)]:
        sub = s_elbo[(s_elbo["prior_type"] == prior_type) & (s_elbo["lambda"] == lam)]
        rows.append({
            "label": f"ELBO\n{prior_type}" + (f" λ={lam}" if lam > 0 else ""),
            "color": "#C04B3B" if prior_type == "geometric" else "#666",
            "E_k_mean": sub["effective_k_nobypass"].mean(),
            "E_k_std": sub["effective_k_nobypass"].std(),
            "perp_mean": sub["selection_perplexity"].mean(),
            "perp_std": sub["selection_perplexity"].std(),
            "max_p_mean": sub["mean_max_p"].mean(),
            "bypass_mean": sub["bypass_fraction"].mean(),
        })
    df = pd.DataFrame(rows)
    df = df.iloc[[0, 3, 1, 2]].reset_index(drop=True)  # NONE, ELBO uniform, K_PENALTY, ELBO geom

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(df))
    ax = axes[0]
    ax.bar(x, df["E_k_mean"], yerr=df["E_k_std"], color=df["color"], capsize=4, alpha=0.85)
    ax.axhline(3, color="green", ls="--", lw=1.5, label="K_true=3 (ideal)")
    ax.axhline(11, color="#888", ls=":", lw=1, label="uniform mean=11")
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], fontsize=8)
    ax.set_ylabel("E[k|nb]")
    ax.set_title("E[k|nb] across regularisers — Toy B(K_true=3), k_max=20")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    ax.bar(x, df["perp_mean"], yerr=df["perp_std"], color=df["color"], capsize=4, alpha=0.85)
    ax.axhline(1, color="green", ls="--", lw=1.5, label="ideal perp ≈ 1")
    ax.axhline(19, color="#888", ls=":", lw=1, label="uniform perp = 19")
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], fontsize=8)
    ax.set_ylabel("selection_perplexity")
    ax.set_title("Perplexity comparison")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(OUT / "fig3_regulariser_compare.png", bbox_inches="tight")
    plt.close()


def fig4_marginal_distributions() -> None:
    """Marginal_p shape under each regulariser."""
    s3 = pd.read_csv(EXP / "sweep3_kreg.csv")
    s_elbo = pd.read_csv(ELBO / "results.csv")

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    k_modes = np.arange(2, 21)

    panels = [
        ("NONE",          s3[s3["k_reg"] == "NONE"]["marginal_p"].iloc[0], "#888"),
        ("ELBO uniform",  s_elbo[(s_elbo["prior_type"] == "uniform")]["marginal_p"].iloc[0], "#666"),
        ("K_PENALTY",     s3[s3["k_reg"] == "K_PENALTY"]["marginal_p"].iloc[0], "#7B4FA1"),
        ("ELBO geom λ=0.5", s_elbo[(s_elbo["prior_type"] == "geometric") & (s_elbo["lambda"] == 0.5)]["marginal_p"].iloc[0], "#C04B3B"),
    ]
    for ax, (title, marg_str, color) in zip(axes, panels):
        marg = np.array(ast.literal_eval(marg_str))
        ax.bar(k_modes, marg, color=color, alpha=0.85)
        ax.axvline(3, color="green", ls="--", lw=1.5, alpha=0.6, label="K_true=3")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("k mode")
        ax.set_xticks([2, 5, 10, 15, 20])
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("marginal probability")
    fig.suptitle("Distribution shape over modes — Toy B(K_true=3), k_max=20, seed=42",
                 y=1.02, fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT / "fig4_marginal_p.png", bbox_inches="tight")
    plt.close()


def fig5_elbo_prior_sweep() -> None:
    df = pd.read_csv(ELBO / "results.csv")
    df["config"] = df.apply(lambda r: "uniform" if r["prior_type"] == "uniform" else f"geom λ={r['lambda']}", axis=1)
    g = df.groupby("config")[["effective_k_nobypass", "selection_perplexity", "mean_max_p", "dead_mode_count"]].agg(["mean", "std"]).reset_index()
    order = ["uniform", "geom λ=0.05", "geom λ=0.2", "geom λ=0.5"]
    g = g.set_index("config").reindex(order).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(order))
    ax = axes[0]
    ax.errorbar(x, g[("effective_k_nobypass", "mean")], yerr=g[("effective_k_nobypass", "std")],
                marker="o", color="#C04B3B", capsize=4, linewidth=1.7, label="E[k|nb]")
    ax.errorbar(x, g[("selection_perplexity", "mean")], yerr=g[("selection_perplexity", "std")],
                marker="s", color="#2E7D9A", capsize=4, linewidth=1.7, label="perplexity")
    ax.axhline(3, color="green", ls="--", lw=1, label="K_true=3")
    ax.axhline(19, color="#888", ls=":", lw=1, label="uniform perp")
    ax.set_xticks(x)
    ax.set_xticklabels(order, fontsize=9)
    ax.set_ylabel("metric value")
    ax.set_title("Tuning ELBO with k_prior_type='geometric'")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    ax.errorbar(x, g[("mean_max_p", "mean")], yerr=g[("mean_max_p", "std")],
                marker="o", color="#7B4FA1", capsize=4, linewidth=1.7, label="mean_max_p")
    dead_mean = g[("dead_mode_count", "mean")] / 19  # normalise to fraction
    ax.errorbar(x, dead_mean, marker="s", color="#888", capsize=4, linewidth=1.7,
                label="dead-mode fraction (of 19)")
    ax.set_xticks(x)
    ax.set_xticklabels(order, fontsize=9)
    ax.set_ylabel("fraction")
    ax.set_ylim(0, 1)
    ax.set_title("Concentration intensity")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "fig5_elbo_prior.png", bbox_inches="tight")
    plt.close()


def fig6_trajectory() -> None:
    """Toy A trajectory from the diagnostic run."""
    df = pd.read_csv(DIAG / "trajectory.csv")
    # Phase 1 only (with val_loss): the 'find optimal n_epochs' run
    p1 = df[~df["val_loss"].isna()].copy().reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    ax = axes[0]
    ax.plot(p1.index, p1["selection_perplexity"], color="#2E7D9A", linewidth=1.7, label="perplexity")
    ax.plot(p1.index, p1["effective_k_nobypass"], color="#C04B3B", linewidth=1.7, label="E[k|nb]")
    ax.axhline(9, color="#888", ls=":", lw=1, label="uniform perp/E[k]")
    ax.axhline(6, color="#888", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("metric value")
    ax.set_title("Toy A (no intrinsic k) — perp & E[k|nb] over training (k_max=10)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax2 = ax.twinx()
    l1 = ax.plot(p1.index, p1["bypass_fraction"], color="#C04B3B", linewidth=1.7, label="bypass_fraction")
    l2 = ax.plot(p1.index, p1["mean_max_p"], color="#7B4FA1", linewidth=1.7, label="mean_max_p")
    l3 = ax2.plot(p1.index, p1["dead_mode_count"], color="#444", linewidth=1.5, linestyle="--", label="dead modes")
    ax.set_xlabel("epoch")
    ax.set_ylabel("fraction / probability", color="#444")
    ax.set_ylim(0, 1)
    ax2.set_ylabel("dead mode count", color="#444")
    ax2.set_ylim(0, 9)
    lines = l1 + l2 + l3
    ax.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=8)
    ax.set_title("Bypass capture, then progressive starvation")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "fig6_trajectory.png", bbox_inches="tight")
    plt.close()


def fig7_toyb_marginals() -> None:
    """Final marginal_p for Toy B(K_true ∈ {2, 3, 5}) without regulariser."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    k_modes = np.arange(2, 11)
    for ax, kt in zip(axes, [2, 3, 5]):
        marg = np.load(DIAG.parent / "probreg_kselection_experiments_results" / f"marginal_p_toyB_kT{kt}.npy")
        final = marg[-1]
        ax.bar(k_modes, final, color="#888", alpha=0.85)
        ax.axvline(kt, color="green", ls="--", lw=1.7, label=f"K_true={kt}")
        ax.set_title(f"K_true={kt}")
        ax.set_xlabel("k mode")
        ax.set_xticks(k_modes)
        ax.set_ylim(0, 0.5)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(loc="upper left", fontsize=9)
    axes[0].set_ylabel("marginal probability")
    fig.suptitle("Toy B trajectory diagnostic: NO intrinsic-k discovery (k=9 dominates regardless of K_true)",
                 y=1.02, fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT / "fig7_toyb_kT_marginals.png", bbox_inches="tight")
    plt.close()


def main() -> None:
    fig1_cap_tracking();         print("fig1_cap_tracking.png")
    fig2_noise_mechanism();      print("fig2_noise.png")
    fig3_regulariser_compare();  print("fig3_regulariser_compare.png")
    fig4_marginal_distributions();print("fig4_marginal_p.png")
    fig5_elbo_prior_sweep();     print("fig5_elbo_prior.png")
    fig6_trajectory();           print("fig6_trajectory.png")
    fig7_toyb_marginals();       print("fig7_toyb_kT_marginals.png")


if __name__ == "__main__":
    main()
