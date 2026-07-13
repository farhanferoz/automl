"""ClassReg k-sweep across all 4 toy datasets (task #34).

Sweeps k ∈ {2, 3, 5, 7, 10, 15} plus a k=∞ baseline (PyTorchNeuralNetwork direct regression).
5 seeds each. Metrics: MSE, NLL (Gaussian via law of total variance), PICP95, sharpness.
CSV schema matches probreg_k_sweep.py for cross-join in P2.4.

Output: classreg_k_sweep_results/
"""

from __future__ import annotations

import gc
import logging
import math
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split

from automl_package.enums import MapperType, UncertaintyMethod
from automl_package.examples._toy_datasets import make_datasets
from automl_package.models.classifier_regression import ClassifierRegressionModel
from automl_package.models.neural_network import PyTorchNeuralNetwork

mpl.use("Agg")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

OUT_DIR = Path(__file__).parent / "classreg_k_sweep_results"
OUT_DIR.mkdir(exist_ok=True)

SEEDS = (42, 123, 7, 2026, 31)
K_VALUES = [2, 3, 5, 7, 10, 15]
N_EPOCHS = 80
LR = 0.01
EARLY_STOP = 15
VAL_FRAC = 0.2
_Z95 = 1.96  # z-score for 95% symmetric prediction interval
_NAN_METRICS: dict = {
    "mse": float("nan"), "nll_gaussian": float("nan"),
    "picp95": float("nan"), "sharpness": float("nan"),
}


def _release_memory() -> None:
    """Free GPU/XPU cache + Python refs. Prevents OOM when sweeping many models."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _bin_stats(y_tr: np.ndarray, boundaries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin centroid and within-bin variance from training data."""
    y_flat = y_tr.ravel()
    bin_indices = np.digitize(y_flat, boundaries[1:-1])  # boundaries include outer edges
    # First/last edges are ±inf (create_bins(min_value=-inf, max_value=inf)); fall back
    # to the training-data range for any empty edge bins so centroids stay finite.
    fallback_lo = float(y_flat.min())
    fallback_hi = float(y_flat.max())
    centroids = np.zeros(k)
    variances = np.zeros(k)
    for i in range(k):
        mask = bin_indices == i
        if mask.sum() > 0:
            centroids[i] = float(np.mean(y_flat[mask]))
            variances[i] = float(np.var(y_flat[mask])) if mask.sum() > 1 else 1e-4
        else:
            lo = boundaries[i] if np.isfinite(boundaries[i]) else fallback_lo
            hi = boundaries[i + 1] if np.isfinite(boundaries[i + 1]) else fallback_hi
            centroids[i] = 0.5 * (lo + hi)
            variances[i] = 1e-4
    return centroids, variances


def _classreg_nll_picp(
    probs: np.ndarray, y_tr: np.ndarray,
    boundaries: np.ndarray, y_te: np.ndarray,
    k: int,
) -> tuple[float, float, float]:
    """Gaussian NLL + PICP95 + sharpness via law of total variance over bins."""
    try:
        centroids, variances = _bin_stats(y_tr, boundaries, k)
        # Law of total variance: var = E[var_i] + Var[mean_i]
        pred_mean = (probs * centroids).sum(axis=1)
        epistemic = (probs * (centroids - pred_mean[:, None]) ** 2).sum(axis=1)
        aleatoric = (probs * variances).sum(axis=1)
        total_var = np.clip(epistemic + aleatoric, 1e-8, None)
        total_std = np.sqrt(total_var)

        log_var = np.log(total_var)
        nll = float(np.mean(0.5 * (math.log(2 * math.pi) + log_var + (y_te - pred_mean) ** 2 / total_var)))
        lo = pred_mean - _Z95 * total_std
        hi = pred_mean + _Z95 * total_std
        picp95 = float(np.mean((y_te >= lo) & (y_te <= hi)))
        sharpness = float(np.mean(2 * _Z95 * total_std))
        return nll, picp95, sharpness
    except Exception as e:
        logger.warning("NLL/PICP computation failed: %s", e)
        return float("nan"), float("nan"), float("nan")


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _run_classreg(
    ds_name: str, x_tr: np.ndarray, y_tr: np.ndarray,
    x_te: np.ndarray, y_te: np.ndarray,
    k: int, seed: int,
) -> dict:
    t0 = time.time()
    model = ClassifierRegressionModel(
        base_classifier_class=PyTorchNeuralNetwork,
        n_classes=k,
        mapper_type=MapperType.LOOKUP_MEDIAN,
        uncertainty_method=UncertaintyMethod.CONSTANT,
        n_epochs=N_EPOCHS,
        learning_rate=LR,
        early_stopping_rounds=EARLY_STOP,
        validation_fraction=VAL_FRAC,
        random_seed=seed,
        calculate_feature_importance=False,
    )
    model.fit(x_tr, y_tr)

    y_pred = model.predict(x_te)
    mse = float(np.mean((y_pred - y_te) ** 2))
    probs = model.predict_proba(x_te, filter_data=False)[:, :k]
    nll, picp95, sharpness = _classreg_nll_picp(probs, y_tr, model.class_boundaries, y_te, k)
    wall_time = round(time.time() - t0, 1)
    return {
        "dataset": ds_name, "model": "ClassReg",
        "k": k, "seed": seed, "wall_time_s": wall_time,
        "mse": round(mse, 6), "nll_gaussian": round(nll, 6),
        "picp95": round(picp95, 4), "sharpness": round(sharpness, 4),
    }


def _run_direct_nn(
    ds_name: str, x_tr: np.ndarray, y_tr: np.ndarray,
    x_te: np.ndarray, y_te: np.ndarray,
    seed: int,
) -> dict:
    """k=∞ baseline: plain PyTorchNeuralNetwork direct regression."""
    t0 = time.time()
    model = PyTorchNeuralNetwork(
        n_epochs=N_EPOCHS,
        learning_rate=LR,
        early_stopping_rounds=EARLY_STOP,
        validation_fraction=VAL_FRAC,
        random_seed=seed,
        calculate_feature_importance=False,
    )
    model.fit(x_tr, y_tr)
    y_pred = model.predict(x_te)
    if y_pred.ndim > 1:
        y_pred = y_pred.ravel()
    y_pred_tr = model.predict(x_tr).ravel()
    mse = float(np.mean((y_pred - y_te) ** 2))
    try:
        std = float(np.std(y_tr - y_pred_tr))
        if std < 1e-8:
            std = 1e-8
        log_var_val = math.log(std ** 2)
        nll = float(np.mean(0.5 * (math.log(2 * math.pi) + log_var_val + (y_te - y_pred) ** 2 / std ** 2)))
    except Exception:
        nll = float("nan")
    wall_time = round(time.time() - t0, 1)
    return {
        "dataset": ds_name, "model": "DirectNN",
        "k": -1, "seed": seed, "wall_time_s": wall_time,
        "mse": round(mse, 6), "nll_gaussian": round(nll, 6),
        "picp95": float("nan"), "sharpness": float("nan"),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_metrics_vs_k(df: pd.DataFrame, summary: pd.DataFrame, pdf: PdfPages) -> None:
    datasets = df["dataset"].unique()
    for ds in datasets:
        sub_cr = summary[(summary["dataset"] == ds) & (summary["model"] == "ClassReg")].sort_values("k")
        sub_nn = summary[(summary["dataset"] == ds) & (summary["model"] == "DirectNN")]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"ClassReg k-sweep — {ds}", fontsize=11)

        for ax, metric, label in zip(axes, ["mse_mean", "nll_gaussian_mean", "picp95_mean"], ["MSE", "NLL (Gaussian)", "PICP95"], strict=False):
            ax.errorbar(sub_cr["k"], sub_cr[metric], yerr=sub_cr.get(metric.replace("_mean", "_std"), None),
                        fmt="o-", color="C0", label="ClassReg", capsize=3)
            if not sub_nn.empty and metric in sub_nn.columns:
                nn_val = float(sub_nn[metric].iloc[0])
                if np.isfinite(nn_val):
                    ax.axhline(nn_val, color="C1", ls="--", lw=1.5, label="DirectNN (k=∞)")
            ax.set_xlabel("k (n_classes)")
            ax.set_ylabel(label)
            ax.set_xticks(K_VALUES)
            ax.grid(True, alpha=0.3)
            # PICP95 nominal target is 0.95; MSE/NLL use plain minimum.
            best_idx = (sub_cr[metric] - 0.95).abs().idxmin() if "picp" in metric else sub_cr[metric].idxmin()
            if pd.notna(best_idx):
                best_k = int(sub_cr.loc[best_idx, "k"])
                ax.axvline(best_k, color="C2", ls=":", alpha=0.7, label=f"best k={best_k}")
            ax.legend(fontsize=7)
            ax.tick_params(labelsize=7)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def _plot_summary_table(ds: str, summary: pd.DataFrame, pdf: PdfPages) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")
    ax.set_title(f"{ds} — ClassReg k-sweep summary (mean ± std, {len(SEEDS)} seeds)", fontsize=9)
    sub = summary[summary["dataset"] == ds].sort_values(["model", "k"])
    cols = ["model", "k", "mse_mean", "mse_std", "nll_gaussian_mean", "nll_gaussian_std", "picp95_mean", "sharpness_mean"]
    present = [c for c in cols if c in sub.columns]
    tbl = ax.table(cellText=sub[present].round(4).values.tolist(), colLabels=present, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 1.4)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run ClassReg k-sweep with 5 seeds and generate CSV + PDF reports."""
    datasets = make_datasets()
    results_path = OUT_DIR / "k_sweep.csv"

    # Resume: skip any (dataset, model, k, seed) already in k_sweep.csv.
    done_keys: set[tuple] = set()
    if results_path.exists():
        prior = pd.read_csv(results_path)
        done_keys = set(zip(prior["dataset"], prior["model"], prior["k"].astype(int), prior["seed"].astype(int)))
        if done_keys:
            print(f"Resuming: {len(done_keys)}/{len(datasets) * (len(K_VALUES) + 1) * len(SEEDS)} runs already done.")

    write_header = not results_path.exists()
    total = len(datasets) * (len(K_VALUES) + 1) * len(SEEDS)
    run_count = 0

    for ds_name, x, y in datasets:
        print(f"\n=== Dataset: {ds_name} ===")
        for seed in SEEDS:
            x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=seed)

            # Direct NN baseline (k=∞)
            run_count += 1
            if (ds_name, "DirectNN", -1, seed) not in done_keys:
                t0 = time.time()
                print(f"  [{run_count:3d}/{total}] DirectNN s={seed} ...", end="", flush=True)
                try:
                    row = _run_direct_nn(ds_name, x_tr, y_tr, x_te, y_te, seed)
                    print(f" MSE={row['mse']:.4f} ({row['wall_time_s']:.0f}s)")
                except Exception as e:
                    print(f" ERROR: {e}")
                    row = {"dataset": ds_name, "model": "DirectNN", "k": -1, "seed": seed,
                           "wall_time_s": round(time.time() - t0, 1), **_NAN_METRICS}
                _release_memory()
                pd.DataFrame([row]).to_csv(results_path, mode="a", header=write_header, index=False)
                write_header = False

            for k in K_VALUES:
                run_count += 1
                if (ds_name, "ClassReg", k, seed) in done_keys:
                    continue
                t0 = time.time()
                print(f"  [{run_count:3d}/{total}] ClassReg k={k:2d} s={seed} ...", end="", flush=True)
                try:
                    row = _run_classreg(ds_name, x_tr, y_tr, x_te, y_te, k, seed)
                    print(f" MSE={row['mse']:.4f} NLL={row['nll_gaussian']:.4f} PICP={row['picp95']:.3f} ({row['wall_time_s']:.0f}s)")
                except Exception as e:
                    print(f" ERROR: {e}")
                    row = {"dataset": ds_name, "model": "ClassReg", "k": k, "seed": seed,
                           "wall_time_s": round(time.time() - t0, 1), **_NAN_METRICS}
                _release_memory()
                pd.DataFrame([row]).to_csv(results_path, mode="a", header=write_header, index=False)
                write_header = False

    df = pd.read_csv(results_path)

    numeric_cols = ["mse", "nll_gaussian", "picp95", "sharpness", "wall_time_s"]
    agg = {c: ["mean", "std"] for c in numeric_cols if c in df.columns}
    summary = df.groupby(["dataset", "model", "k"]).agg(agg)
    summary.columns = ["_".join(col) for col in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(OUT_DIR / "k_sweep_summary.csv", index=False)

    print("\n\n=== SUMMARY (mean MSE across seeds) ===")
    for ds_name, _, _ in datasets:
        ds_sum = summary[summary["dataset"] == ds_name].sort_values("mse_mean")
        print(f"\n{ds_name}:")
        print(ds_sum[["model", "k", "mse_mean", "nll_gaussian_mean", "picp95_mean"]].to_string(index=False))

    pdf_path = OUT_DIR / "k_sweep.pdf"
    with PdfPages(pdf_path) as pdf:
        _plot_metrics_vs_k(df, summary, pdf)
        for ds_name, _, _ in datasets:
            _plot_summary_table(ds_name, summary, pdf)

    print(f"\nDone. Results in {OUT_DIR}/")


if __name__ == "__main__":
    main()
