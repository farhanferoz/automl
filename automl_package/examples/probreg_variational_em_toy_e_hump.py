"""Toy E (the hump): the x-confound breaker for per-input count selection.

Step-3 (``probreg_variational_em_step3_perinput_model.py``) showed the per-input count rising
with x on ``make_toy_c``. But there the mode spacing rises monotonically with x, so "the count
tracks the structure" and "the count tracks the input x" make the SAME prediction. ``make_toy_e``
uses the non-monotone :func:`_toy_datasets.sep_hump` instead: the two modes are merged at BOTH
ends of x and resolved only in the middle band ``[X_LO, X_HI]`` (where ``sep_hump(x) > 2``). The
ground-truth count is therefore ``1 -> 2 -> 1`` — it must come back DOWN while x keeps rising,
which a selector that merely tracks x cannot fake.

This reuses the validated per-input machinery — the :class:`AggregateSparsityKSelector` model and
the Step-2 held-out arbiter (one estimate per input, validated against a resample gold standard)
— and overlays, per input x:
  * effective count  eff#(x) = exp(entropy of the per-input weights)  — classes engaged (right axis);
  * held-out arbiter Δ̂(x)/Δ*(x)  — whether those classes earn their keep on fresh data (left axis).

Expected reading:
  * make_toy_e (genuine): eff#(x) is a top-hat high across [X_LO, X_HI], and the arbiter is
    positive there and ~0 outside (it does NOT credit the unresolved tails even if the count
    creeps above 1 there).
  * make_toy_e_broad (trap): variance humps identically yet eff#(x) stays ~1 and the arbiter is
    flat — no over-chopping, no unearned credit.

Run with any torch+numpy interpreter:
    python3 automl_package/examples/probreg_variational_em_toy_e_hump.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _toy_datasets as td
import _variational_em_perinput as vemp
import probreg_variational_em_step2_perinput_arbiter as p2
import probreg_variational_em_step3_perinput_model as p3

K_MAX = 6
ALPHA0 = 0.1
SIGMA = 0.3
SEP_MIN, SEP_MAX = 0.3, 4.0
N_TR, N_TE = 1500, 4000
N_EPOCHS = 1000
SEEDS = [0, 1, 2]
M_GOLD, N_GRID, N_BINS = 1500, 40, 24
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "variational_em_toy_e_results")

# The two x at which sep_hump(x) crosses the 2σ bimodality boundary (k* = 2 strictly between them).
_T = 1.0 - (2.0 - SEP_MIN) / (SEP_MAX - SEP_MIN)
X_LO, X_HI = (1.0 - _T) / 2.0, (1.0 + _T) / 2.0


def run_condition(make_fn, given_fn, label: str) -> dict:
    """Fit the per-input model + the plain baseline per seed; return arbiter and eff#(x) curves."""
    grid = np.linspace(0.0, 1.0, N_GRID).astype(np.float32)
    bin_delta, gold_delta, bin_eff = [], [], []
    for seed in SEEDS:
        x_tr, y_tr = make_fn(n=N_TR, seed=seed)
        x_te, y_te = make_fn(n=N_TE, seed=seed + 500)
        model = vemp.train_aggregate_sparsity(x_tr, y_tr, k_max=K_MAX, alpha0=ALPHA0, n_epochs=N_EPOCHS, lr=1e-2, seed=seed)
        pure = p2.train_cond_gaussian(x_tr, y_tr, seed=seed)

        delta = p2.pure_nll_per_point(pure, x_te, y_te) - p3.bucket_nll_per_point(model, x_te, y_te)
        bin_delta.append(p2.binned_mean(x_te.ravel(), delta, N_BINS))
        eff = p3.effective_count(model.weights(p2._to_xy(x_te, y_te)[0]).cpu().numpy())
        bin_eff.append(p2.binned_mean(x_te.ravel(), eff, N_BINS))

        gold = np.empty(N_GRID)
        for j, xg in enumerate(grid):
            xs = np.full(M_GOLD, xg, dtype=np.float32)
            ys = given_fn(xs, sigma=SIGMA, sep_min=SEP_MIN, sep_max=SEP_MAX, seed=seed * 100_003 + j)
            gold[j] = float((p2.pure_nll_per_point(pure, xs.reshape(-1, 1), ys) - p3.bucket_nll_per_point(model, xs.reshape(-1, 1), ys)).mean())
        gold_delta.append(gold)

    bd, gd, be = np.array(bin_delta), np.array(gold_delta), np.array(bin_eff)
    bin_centers = (np.linspace(0.0, 1.0, N_BINS + 1)[:-1] + np.linspace(0.0, 1.0, N_BINS + 1)[1:]) / 2.0
    inside = (grid > X_LO) & (grid < X_HI)
    print(f"  [{label}] eff#(x): ends -> {np.nanmean(be[:, [0, 1, -2, -1]]):.2f}   band -> {np.nanmean(be[:, (bin_centers > X_LO) & (bin_centers < X_HI)]):.2f}")
    print(f"  [{label}] gold Δ* inside band [{X_LO:.2f},{X_HI:.2f}]: {np.nanmean(gd[:, inside]):+.3f} nats   outside: {np.nanmean(gd[:, ~inside]):+.3f} nats")
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
    """Money plot: per-input effective count vs the held-out arbiter, with the two-sided band shaded."""
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
        ax.axvspan(X_LO, X_HI, color="gray", alpha=0.12, label="two modes resolved (centres > 2σ apart)")
        ax.set_xlabel("input x  (mode spacing: low → high → low)")
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
    fig.suptitle("Toy E (non-monotone): count must rise THEN fall while x rises — tracking structure, not x", fontsize=12)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "toy_e_hump.png")
    fig.savefig(path, dpi=130)
    print(f"  saved plot -> {path}")


def main() -> None:
    """Run both conditions with the per-input model on the humped toy, save JSON + the money plot."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("== two-peak data, spacing HUMPED in x (genuine, non-monotone count) ==")
    c = run_condition(td.make_toy_e, td.sample_toy_e_given_x, "two-peak data (humped spacing)")
    print("== single-peak twin, variance HUMPED identically (over-chopping / variance-tracking trap) ==")
    b = run_condition(td.make_toy_e_broad, td.sample_toy_e_broad_given_x, "single-peak twin (variance-matched)")
    results = [c, b]
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump({"config": {"k_max": K_MAX, "alpha0": ALPHA0, "sigma": SIGMA, "sep": [SEP_MIN, SEP_MAX],
                              "n_tr": N_TR, "n_te": N_TE, "epochs": N_EPOCHS, "seeds": SEEDS,
                              "x_lo": X_LO, "x_hi": X_HI},
                   "conditions": results}, f, indent=2)
    try:
        _plot(results)
    except Exception as exc:
        print(f"  (plot skipped: {exc})")
    print(f"\nSaved results to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
