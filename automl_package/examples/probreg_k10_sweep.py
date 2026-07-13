"""ProbReg k_max=10 focused sweep — follow-up to probreg_k_sweep.py.

Grid: Cell B & C × k_max=10 × 3 (dynamic, k-reg) pairs × 5 seeds × 4 datasets = 120 runs.

Pruned using previous-sweep findings:
  - GUMBEL_SOFTMAX dropped (dominated by SOFT_GATING everywhere).
  - K_PENALTY dropped (redundant with SOFT_GATING + NONE and less principled than ELBO).
  - k_max fixed at 10; bypass mode ≡ "k=11" (direct regression).

Valid (dynamic, k-reg) pairs (3):
  NONE        × NONE   (fixed k=10 baseline)
  SOFT_GATING × NONE
  SOFT_GATING × ELBO   (uses the split bypass/k-uniform prior — see
                         probabilistic_regression.py ELBO block and
                         bypass_prior_prob / k_prior_type kwargs)

Outputs (probreg_k10_sweep_results/):
  results.csv   — per-run row incl. bypass_fraction, effective_k_nobypass
  summary.csv   — mean ± std across seeds
  results_{dataset}.pdf — metrics table + effective-k histogram
  results_all.pdf — merged

Questions this sweep answers:
  1. At k_max=10, does dynamic-k route mass to bypass on exponential?
  2. Does ELBO-with-uniform-prior pick a data-intrinsic k instead of
     tracking k_max, once the buggy linspace prior is removed?
  3. Does B vs C tie survive at k_max=10, or does one pull ahead?
"""

from __future__ import annotations

import gc
import logging
import math
import subprocess
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split

from automl_package.enums import (
    NClassesRegularization,
    NClassesSelectionMethod,
    ProbabilisticRegressionOptimizationStrategy,
    ProbRegLossType,
    RegressionStrategy,
    UncertaintyMethod,
)
from automl_package.examples._kselection_metrics import compute_kselection_metrics
from automl_package.examples._toy_datasets import make_datasets
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel

mpl.use("Agg")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

OUT_DIR = Path(__file__).parent / "probreg_k10_sweep_results"
OUT_DIR.mkdir(exist_ok=True)

SEEDS = (42, 123, 7, 2026, 31)
K_VALUES = (10,)
N_EPOCHS = 80
LR = 0.01
EARLY_STOP = 15
VAL_FRAC = 0.2
BATCH_SIZE = 64

_NAN_METRICS: dict = {
    "mse": float("nan"), "nll_gaussian": float("nan"), "nll_mdn": float("nan"),
    "effective_k": float("nan"), "bypass_fraction": float("nan"),
    "effective_k_nobypass": float("nan"),
    "selection_perplexity": float("nan"), "dead_mode_count": -1,
    "mean_max_p": float("nan"),
    "max_p_mid": float("nan"), "sum_p_mid": float("nan"),
}


def _release_memory() -> None:
    """Free GPU/XPU cache + Python refs. Called after every training run — without
    this, allocations accumulate across 840 runs and trigger OS-level OOM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()

# Cell B: Gaussian-LTV + REGRESSION_ONLY (ordering auto-resolves to 1.0)
# Cell C: Gaussian-LTV + CE_STOP_GRAD   (ordering auto-resolves to 0.0)
CELLS = {
    "B": ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY,
    "C": ProbabilisticRegressionOptimizationStrategy.CE_STOP_GRAD,
}

# Valid (dynamic, k-reg) pairs — pruned for k_max=10 follow-up.
# Gumbel and K_PENALTY dropped per previous-sweep findings; ELBO uses the
# corrected split bypass/k-uniform prior.
VALID_PAIRS: list[tuple[NClassesSelectionMethod, NClassesRegularization]] = [
    (NClassesSelectionMethod.NONE, NClassesRegularization.NONE),
    (NClassesSelectionMethod.SOFT_GATING, NClassesRegularization.NONE),
    (NClassesSelectionMethod.SOFT_GATING, NClassesRegularization.ELBO),
]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _gaussian_nll(y: np.ndarray, mean: np.ndarray, log_var: np.ndarray) -> float:
    var = np.exp(log_var)
    return float(np.mean(0.5 * (math.log(2 * math.pi) + log_var + (y - mean) ** 2 / var)))


def _mdn_nll(y: np.ndarray, probs: np.ndarray, mus: np.ndarray, log_vars: np.ndarray) -> float:
    y = y.reshape(-1, 1)
    log_comp = -0.5 * (math.log(2 * math.pi) + log_vars + (y - mus) ** 2 / np.exp(log_vars))
    log_weights = np.log(np.clip(probs, 1e-8, None))
    return float(-np.mean(np.logaddexp.reduce(log_weights + log_comp, axis=1)))


def _compute_middle_bin_stats(model: ProbabilisticRegressionModel, x_te: np.ndarray, k: int) -> tuple[float, float]:
    """Middle-third-of-bins probability summary.

    For k bins indexed 0..k-1, the middle third is indices [k//3 : k - k//3]:
    k=3 → {1}, k=5 → {1,2,3}, k=7 → {2,3,4}.

    Returns:
        (max_p_mid, sum_p_mid): each is mean-across-val-set of the per-sample
            max (resp. sum) of class probabilities over the middle-third slice.
            `max_p_mid` matches the handover spec (docs/phase2_handover.md) and is
            the metric P2.4 consumes. `sum_p_mid` is a companion measuring total
            middle-region mass per sample (less concentration-biased than max).
    """
    x_t = torch.tensor(x_te, dtype=torch.float32).to(model.device)
    model.model.eval()
    try:
        with torch.no_grad():
            raw = model.model.classifier_layers(x_t)
            masked = torch.full_like(raw, float("-inf"))
            masked[:, :k] = raw[:, :k]
            probs = torch.softmax(masked, dim=1)[:, :k]
            mid_slice = probs[:, k // 3: k - k // 3]
            max_p_mid = float(mid_slice.max(dim=1).values.mean().cpu().item())
            sum_p_mid = float(mid_slice.sum(dim=1).mean().cpu().item())
        return max_p_mid, sum_p_mid
    except Exception as e:
        logger.warning("middle-bin stats failed (k=%d): %s", k, e)
        return float("nan"), float("nan")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _train_model(
    x_tr: np.ndarray, y_tr: np.ndarray, seed: int,
    cell: str, k_max: int,
    dynamic: NClassesSelectionMethod, k_reg: NClassesRegularization,
) -> ProbabilisticRegressionModel:
    opt_strat = CELLS[cell]

    # For dynamic-k, n_classes is a hint (ignored at build time);
    # max_n_classes_for_probabilistic_path controls the actual mode count.
    model = ProbabilisticRegressionModel(
        input_size=1,
        n_classes=k_max,
        max_n_classes_for_probabilistic_path=k_max,
        regression_strategy=RegressionStrategy.SEPARATE_HEADS,
        uncertainty_method=UncertaintyMethod.PROBABILISTIC,
        optimization_strategy=opt_strat,
        prob_reg_loss_type=ProbRegLossType.GAUSSIAN_LTV,
        n_classes_selection_method=dynamic,
        n_classes_regularization=k_reg,
        use_anchored_heads=False,
        constrain_middle_class=True,
        use_monotonic_constraints=False,
        n_epochs=N_EPOCHS,
        learning_rate=LR,
        early_stopping_rounds=EARLY_STOP,
        validation_fraction=VAL_FRAC,
        batch_size=BATCH_SIZE,
        random_seed=seed,
        calculate_feature_importance=False,
    )

    # Guardrail: auto-resolution must produce the expected weights.
    if cell == "C":
        assert model.ordering_constraint_weight == 0.0, (
            f"DRIFT: Cell C ordering_constraint_weight={model.ordering_constraint_weight}, expected 0.0"
        )
    elif cell == "B":
        assert model.ordering_constraint_weight == 1.0, (
            f"DRIFT: Cell B (dynamic={dynamic.name}) ordering_constraint_weight={model.ordering_constraint_weight}, expected 1.0"
        )

    model.fit(x_tr, y_tr)
    return model


# ---------------------------------------------------------------------------
# Metrics for one run
# ---------------------------------------------------------------------------

def _compute_metrics(
    model: ProbabilisticRegressionModel,
    x_te: np.ndarray, y_te: np.ndarray,
    k_max: int, dynamic: NClassesSelectionMethod,
) -> dict:
    preds = model.predict(x_te)
    if preds.ndim > 1:
        preds = preds.ravel()
    mse = float(np.mean((preds - y_te) ** 2))

    try:
        std = model.predict_uncertainty(x_te)
        log_var = np.log(np.clip(std ** 2, 1e-8, None))
        nll_gaussian = _gaussian_nll(y_te, preds, log_var)
    except Exception:
        nll_gaussian = float("nan")

    nll_mdn = float("nan")
    if hasattr(model, "predict_distribution"):
        try:
            dist = model.predict_distribution(x_te)
            nll_mdn = _mdn_nll(y_te, dist.weights, dist.means, np.log(np.clip(dist.stds ** 2, 1e-8, None)))
        except NotImplementedError:
            pass  # dynamic-k and symlog configs legitimately don't support predict_distribution
        except Exception as e:
            logger.warning("MDN NLL failed: %s", e)

    metrics = compute_kselection_metrics(model, x_te, k_max)

    max_p_mid, sum_p_mid = _compute_middle_bin_stats(model, x_te, k_max)

    return {
        "mse": round(mse, 6),
        "nll_gaussian": round(nll_gaussian, 6),
        "nll_mdn": round(nll_mdn, 6),
        "effective_k": round(metrics["effective_k"], 4),
        "bypass_fraction": round(metrics["bypass_fraction"], 4),
        "effective_k_nobypass": round(metrics["effective_k_nobypass"], 4),
        "selection_perplexity": round(metrics["selection_perplexity"], 4),
        "dead_mode_count": int(metrics["dead_mode_count"]),
        "mean_max_p": round(metrics["mean_max_p"], 4),
        "max_p_mid": round(max_p_mid, 4),
        "sum_p_mid": round(sum_p_mid, 4),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_metrics_table(ax: plt.Axes, ds_name: str, summary_df: pd.DataFrame) -> None:
    ax.axis("off")
    ax.set_title(f"{ds_name} — mean ± std across {len(SEEDS)} seeds", fontsize=9)
    ds = summary_df[summary_df["dataset"] == ds_name].copy()
    if ds.empty:
        return
    ds = ds.sort_values(["k_max", "cell", "dynamic", "k_reg"])
    cols = ["k_max", "cell", "dynamic", "k_reg", "mse_mean", "mse_std", "nll_gaussian_mean", "effective_k_mean", "max_p_mid_mean"]
    present = [c for c in cols if c in ds.columns]
    tbl = ax.table(
        cellText=ds[present].values.tolist(),
        colLabels=present,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6)
    tbl.scale(1, 1.3)


def _plot_effective_k_hist(ax: plt.Axes, ds_name: str, k_max: int, results_df: pd.DataFrame) -> None:
    """Effective-k distribution for dynamic runs at this (dataset, k_max)."""
    sub = results_df[
        (results_df["dataset"] == ds_name) &
        (results_df["k_max"] == k_max) &
        (results_df["dynamic"] != "NONE")
    ]
    if sub.empty:
        ax.set_visible(False)
        return

    ax.set_title(f"{ds_name} k_max={k_max} — effective_k by config", fontsize=8)
    configs = sub.groupby(["cell", "dynamic", "k_reg"])["effective_k"].agg(["mean", "std"]).reset_index()
    labels = [f"{r.cell}|{r.dynamic[:4]}|{r.k_reg[:4]}" for _, r in configs.iterrows()]
    ax.bar(range(len(configs)), configs["mean"], yerr=configs["std"], capsize=3, color="steelblue", alpha=0.7)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("effective_k", fontsize=7)
    ax.axhline(k_max, color="red", lw=0.8, ls="--", alpha=0.7, label=f"k_max={k_max}")
    ax.axhline(2, color="gray", lw=0.8, ls=":", alpha=0.7, label="k_min=2")
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(labelsize=6)


def _plot_mse_by_config(ax: plt.Axes, ds_name: str, k_max: int, summary_df: pd.DataFrame) -> None:
    """Bar chart: MSE by (cell, dynamic, k_reg) for this (dataset, k_max)."""
    sub = summary_df[(summary_df["dataset"] == ds_name) & (summary_df["k_max"] == k_max)].copy()
    if sub.empty:
        ax.set_visible(False)
        return
    sub = sub.sort_values(["cell", "dynamic", "k_reg"])
    labels = [f"{r.cell}|{r.dynamic[:4]}|{r.k_reg[:4]}" for _, r in sub.iterrows()]
    colors = ["C0" if r.cell == "B" else "C1" for _, r in sub.iterrows()]
    ax.bar(range(len(sub)), sub["mse_mean"], yerr=sub["mse_std"], capsize=3, color=colors, alpha=0.7)
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("MSE", fontsize=7)
    ax.set_title(f"{ds_name} k_max={k_max} — MSE (B=blue, C=orange)", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(labelsize=6)


def _make_pdf(ds_name: str, summary_df: pd.DataFrame, results_df: pd.DataFrame) -> None:
    pdf_path = OUT_DIR / f"results_{ds_name}.pdf"
    with PdfPages(pdf_path) as pdf:
        # Page 1: full metrics table
        fig, ax = plt.subplots(figsize=(18, 10))
        _plot_metrics_table(ax, ds_name, summary_df)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Pages 2–4: MSE bar charts + effective_k histograms per k_max
        for k_max in K_VALUES:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f"{ds_name} k_max={k_max}", fontsize=10)
            _plot_mse_by_config(axes[0], ds_name, k_max, summary_df)
            _plot_effective_k_hist(axes[1], ds_name, k_max, results_df)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the 840-run Phase 2 k-sweep and generate result CSVs and PDFs."""
    datasets = make_datasets()
    results_path = OUT_DIR / "results.csv"

    # Resume: skip any (dataset, cell, k_max, dynamic, k_reg, seed) already in results.csv.
    done_keys: set[tuple] = set()
    if results_path.exists():
        prior = pd.read_csv(results_path)
        done_keys = set(zip(prior["dataset"], prior["cell"], prior["k_max"].astype(int), prior["dynamic"], prior["k_reg"], prior["seed"].astype(int)))
        if done_keys:
            print(f"Resuming: {len(done_keys)}/{len(datasets) * len(K_VALUES) * len(VALID_PAIRS) * 2 * len(SEEDS)} runs already done.")

    write_header = not results_path.exists()
    total = len(datasets) * len(K_VALUES) * len(VALID_PAIRS) * 2 * len(SEEDS)
    run_count = 0

    for ds_name, x, y in datasets:
        print(f"\n=== Dataset: {ds_name} ===")
        # Split once per (dataset, seed) — reused across all (k_max, cell, dynamic, k_reg).
        splits: dict[int, tuple] = {
            seed: train_test_split(x, y, test_size=0.2, random_state=seed)
            for seed in SEEDS
        }
        for k_max in K_VALUES:
            for cell in ("B", "C"):
                for dynamic, k_reg in VALID_PAIRS:
                    for seed in SEEDS:
                        run_count += 1
                        key = (ds_name, cell, k_max, dynamic.name, k_reg.name, seed)
                        if key in done_keys:
                            continue
                        label = f"{cell}|k={k_max}|{dynamic.name[:4]}|{k_reg.name[:4]}|s{seed}"
                        print(f"  [{run_count:3d}/{total}] {ds_name} {label} ...", end="", flush=True)

                        x_tr, x_te, y_tr, y_te = splits[seed]
                        t0 = time.time()
                        model = None
                        try:
                            model = _train_model(x_tr, y_tr, seed, cell, k_max, dynamic, k_reg)
                            metrics = _compute_metrics(model, x_te, y_te, k_max, dynamic)
                            wall_time = round(time.time() - t0, 1)
                            row = {
                                "dataset": ds_name,
                                "cell": cell,
                                "k_max": k_max,
                                "dynamic": dynamic.name,
                                "k_reg": k_reg.name,
                                "seed": seed,
                                "wall_time_s": wall_time,
                                **metrics,
                            }
                            print(f" MSE={metrics['mse']:.4f} eff_k={metrics['effective_k']:.2f} ({wall_time:.0f}s)")
                        except Exception as e:
                            wall_time = round(time.time() - t0, 1)
                            print(f" ERROR: {e}")
                            row = {
                                "dataset": ds_name, "cell": cell, "k_max": k_max,
                                "dynamic": dynamic.name, "k_reg": k_reg.name,
                                "seed": seed, "wall_time_s": wall_time,
                                **_NAN_METRICS,
                            }
                        finally:
                            del model
                            _release_memory()
                        # Append immediately — survives a crash at any point.
                        pd.DataFrame([row]).to_csv(results_path, mode="a", header=write_header, index=False)
                        write_header = False

    results_df = pd.read_csv(results_path)

    numeric_cols = [
        "mse", "nll_gaussian", "nll_mdn",
        "effective_k", "bypass_fraction", "effective_k_nobypass",
        "selection_perplexity", "dead_mode_count", "mean_max_p",
        "max_p_mid", "sum_p_mid", "wall_time_s",
    ]
    agg = {c: ["mean", "std"] for c in numeric_cols if c in results_df.columns}
    summary = results_df.groupby(["dataset", "cell", "k_max", "dynamic", "k_reg"]).agg(agg)
    summary.columns = ["_".join(col) for col in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(OUT_DIR / "summary.csv", index=False)

    print("\n\n=== SUMMARY (mean MSE across seeds) ===")
    for ds_name, _, _ in datasets:
        ds_sum = summary[summary["dataset"] == ds_name].sort_values("mse_mean")
        print(f"\n{ds_name}:")
        display_cols = ["cell", "k_max", "dynamic", "k_reg", "mse_mean", "nll_gaussian_mean", "effective_k_mean"]
        print(ds_sum[[c for c in display_cols if c in ds_sum.columns]].head(10).to_string(index=False))

    print("\nGenerating PDFs...")
    for ds_name, _, _ in datasets:
        _make_pdf(ds_name, summary, results_df)

    # Merge PDFs
    try:
        subprocess.run(
            ["pdfunite"] + [str(OUT_DIR / f"results_{ds}.pdf") for ds, _, _ in datasets] + [str(OUT_DIR / "results_all.pdf")],
            check=True, capture_output=True,
        )
        print(f"Merged PDF: {OUT_DIR}/results_all.pdf")
    except Exception as e:
        print(f"pdfunite failed (non-fatal): {e}")

    print(f"\nDone. Results in {OUT_DIR}/")


if __name__ == "__main__":
    main()
