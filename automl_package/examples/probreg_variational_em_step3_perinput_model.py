"""Step-3: the per-input model (Basis B) on the input-varying toy — the full money plot.

Step-2 validated the per-input held-out arbiter against a gold standard, but against the
GLOBAL-weight Basis A model, so the bucket count could not vary with x. Here we run the
actual per-input model — :class:`AggregateSparsityKSelector`, whose per-input softmax weights
are driven by the conditional-mixture likelihood with a SINGLE sparsity prior on the
dataset-average bucket usage (the principled repair of the per-input-Dirichlet collapse,
see ``_variational_em_perinput.py``) — and overlay, per input x:

  * the effective bucket count  eff#(x)  = exp(entropy of the per-input weights)  — how many
    buckets the model engages at x (right axis);
  * the held-out arbiter  Δ̂(x) / Δ*(x)  — whether those buckets actually help on fresh data,
    reusing the Step-2 machinery (left axis).

Expected reading (the whole thesis at per-input resolution):
  * make_toy_c (genuine): eff#(x) rises from ~1 to ~2 as the modes resolve, and the arbiter
    turns positive at the KNOWN two-peaks boundary (sep=2) — the engaged buckets are earned.
  * make_toy_c_broad (trap): the variance grows identically, yet eff#(x) stays ~1 and the
    arbiter stays flat — no over-chopping, no unearned credit.

Run with any torch+numpy interpreter:
    python3 automl_package/examples/probreg_variational_em_step3_perinput_model.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _toy_datasets as td
import _variational_em_perinput as vemp
import probreg_variational_em_step2_perinput_arbiter as p2

K_MAX = 6
ALPHA0 = 0.1
SIGMA = 0.3
SEP_MIN, SEP_MAX = 0.3, 4.0
N_TR, N_TE = 1500, 4000
N_EPOCHS = 1000
SEEDS = [0, 1, 2]
M_GOLD, N_GRID, N_BINS = 1500, 40, 24
X_STAR = p2.X_STAR
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "variational_em_step3_results")


def effective_count(w: np.ndarray) -> np.ndarray:
    """Per-row effective bucket count ``exp(-Σ_c w_c ln w_c)`` for a weight matrix ``(N, K)``."""
    eps = 1e-12
    return np.exp(-(w * np.log(np.clip(w, eps, None))).sum(axis=1))


@torch.no_grad()
def bucket_nll_per_point(model: vemp.AggregateSparsityKSelector, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-point held-out NLL under the per-input mixture."""
    x_t, y_t = p2._to_xy(x, y)
    return model.mixture_nll_per_point(x_t, y_t).cpu().numpy()


def run_condition(make_fn, given_fn, label: str) -> dict:
    """Fit the per-input model + the plain baseline per seed; return arbiter and eff#(x) curves."""
    grid = np.linspace(0.0, 1.0, N_GRID).astype(np.float32)
    bin_delta, gold_delta, bin_eff = [], [], []
    for seed in SEEDS:
        x_tr, y_tr = make_fn(n=N_TR, seed=seed)
        x_te, y_te = make_fn(n=N_TE, seed=seed + 500)
        model = vemp.train_aggregate_sparsity(x_tr, y_tr, k_max=K_MAX, alpha0=ALPHA0, n_epochs=N_EPOCHS, lr=1e-2, seed=seed)
        pure = p2.train_cond_gaussian(x_tr, y_tr, seed=seed)

        delta = p2.pure_nll_per_point(pure, x_te, y_te) - bucket_nll_per_point(model, x_te, y_te)
        bin_delta.append(p2.binned_mean(x_te.ravel(), delta, N_BINS))
        eff = effective_count(model.weights(p2._to_xy(x_te, y_te)[0]).cpu().numpy())
        bin_eff.append(p2.binned_mean(x_te.ravel(), eff, N_BINS))

        gold = np.empty(N_GRID)
        for j, xg in enumerate(grid):
            xs = np.full(M_GOLD, xg, dtype=np.float32)
            ys = given_fn(xs, sigma=SIGMA, sep_min=SEP_MIN, sep_max=SEP_MAX, seed=seed * 100_003 + j)
            gold[j] = float((p2.pure_nll_per_point(pure, xs.reshape(-1, 1), ys) - bucket_nll_per_point(model, xs.reshape(-1, 1), ys)).mean())
        gold_delta.append(gold)

    bd, gd, be = np.array(bin_delta), np.array(gold_delta), np.array(bin_eff)
    bin_centers = (np.linspace(0.0, 1.0, N_BINS + 1)[:-1] + np.linspace(0.0, 1.0, N_BINS + 1)[1:]) / 2.0
    print(f"  [{label}] eff#(x): x=0 -> {np.nanmean(be[:, :3]):.2f}   x=1 -> {np.nanmean(be[:, -3:]):.2f}")
    print(f"  [{label}] gold Δ* at x<x*: {np.nanmean(gd[:, grid < X_STAR]):+.3f} nats   at x>x*: {np.nanmean(gd[:, grid > X_STAR]):+.3f} nats")
    return {
        "label": label,
        "grid": grid.tolist(),
        "bin_centers": bin_centers.tolist(),
        "binned_delta_mean": np.nanmean(bd, axis=0).tolist(),
        "binned_delta_std": np.nanstd(bd, axis=0).tolist(),
        "gold_delta_mean": np.mean(gd, axis=0).tolist(),
        "gold_delta_std": np.std(gd, axis=0).tolist(),
        "eff_count_mean": np.nanmean(be, axis=0).tolist(),
        "eff_count_std": np.nanstd(be, axis=0).tolist(),
    }


def _plot(results: list[dict]) -> None:
    """Money plot: per-input effective bucket count vs the held-out arbiter, per condition."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(results), figsize=(6.4 * len(results), 4.8), sharey=True)
    if len(results) == 1:
        axes = [axes]
    for ax, r in zip(axes, results, strict=False):
        bc = np.array(r["bin_centers"])
        ax.errorbar(bc, r["binned_delta_mean"], yerr=r["binned_delta_std"], fmt="o", ms=4, capsize=2,
                    color="#1f77b4", label="held-out advantage, one estimate per input")
        ax.plot(r["grid"], r["gold_delta_mean"], "-", color="#d62728", lw=2, label="held-out advantage, exact (gold standard)")
        ax.axhline(0.0, color="k", lw=0.8, ls=":")
        ax.axvline(X_STAR, color="gray", lw=1.2, ls="--", label=r"two-peaks boundary (centres $2\sigma$ apart)")
        ax.set_xlabel("input x  (low → high mode spacing)")
        ax.set_title(r["label"])
        ax.grid(alpha=0.25)
        ax2 = ax.twinx()
        ax2.plot(bc, r["eff_count_mean"], "-s", ms=3, color="#2ca02c", label="classes engaged (right axis)")
        ax2.fill_between(bc, np.array(r["eff_count_mean"]) - np.array(r["eff_count_std"]),
                         np.array(r["eff_count_mean"]) + np.array(r["eff_count_std"]), color="#2ca02c", alpha=0.12)
        ax2.set_ylim(0.8, 2.8)
        ax2.set_ylabel("effective number of classes", color="#2ca02c")
        ax2.tick_params(axis="y", labelcolor="#2ca02c")
        lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
        labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
        ax.legend(lines, labels, fontsize=7.5, loc="upper left")
    axes[0].set_ylabel("held-out advantage of the mixture (nats)")
    fig.suptitle("Per-input model: classes engaged (green) vs detail earned on held-out data (blue/red)", fontsize=12)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "perinput_model.png")
    fig.savefig(path, dpi=130)
    print(f"  saved plot -> {path}")


def main() -> None:
    """Run both conditions with the per-input model, save JSON + the money plot."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("== two-peak data (structure varies with x) ==")
    c = run_condition(td.make_toy_c, td.sample_toy_c_given_x, "two-peak data")
    print("== single-peak twin (variance-matched: over-chopping trap) ==")
    b = run_condition(td.make_toy_c_broad, td.sample_toy_c_broad_given_x, "single-peak twin (variance-matched)")
    results = [c, b]
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump({"config": {"k_max": K_MAX, "alpha0": ALPHA0, "sigma": SIGMA, "sep": [SEP_MIN, SEP_MAX],
                              "n_tr": N_TR, "n_te": N_TE, "epochs": N_EPOCHS, "seeds": SEEDS, "x_star": X_STAR},
                   "conditions": results}, f, indent=2)
    try:
        _plot(results)
    except Exception as exc:
        print(f"  (plot skipped: {exc})")
    print(f"\nSaved results to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
