"""ProbReg SNR sweep on exponential — does bypass fraction scale with noise?

Q2 from the Phase-2 follow-up: "does ProbReg fire bypass mode on high-SNR /
simple problems?" We test by fixing the data function (y = exp(x) + ε) and
varying the noise level. If bypass_fraction rises as noise drops, the model
is responding to SNR. If bypass stays high across noise levels, bypass is
driven by heavy-tail structure, not SNR.

Grid: Cell B × k_max=10 × 2 (dynamic, k-reg) pairs × 3 noise levels × 5 seeds = 30 runs.

Noise levels for exponential (default is 0.5):
  0.1  → high SNR  (clean function)
  0.5  → baseline
  2.0  → low SNR   (heavy noise)
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
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel

mpl.use("Agg")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

OUT_DIR = Path(__file__).parent / "probreg_snr_sweep_results"
OUT_DIR.mkdir(exist_ok=True)

SEEDS = (42, 123, 7, 2026, 31)
NOISE_LEVELS = (0.1, 0.5, 2.0)
K_MAX = 10
N_EPOCHS = 80
LR = 0.01
EARLY_STOP = 15
VAL_FRAC = 0.2
BATCH_SIZE = 64

VALID_PAIRS = [
    (NClassesSelectionMethod.NONE, NClassesRegularization.NONE),
    (NClassesSelectionMethod.SOFT_GATING, NClassesRegularization.NONE),
]


def _release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()


def _make_exponential(noise_std: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42 + seed)
    x = rng.uniform(-3, 3, 600).reshape(-1, 1).astype(np.float32)
    y = (np.exp(x.ravel()) + rng.normal(0.0, noise_std, 600)).astype(np.float32)
    return x, y


def _gaussian_nll(y: np.ndarray, mean: np.ndarray, log_var: np.ndarray) -> float:
    var = np.exp(log_var)
    return float(np.mean(0.5 * (math.log(2 * math.pi) + log_var + (y - mean) ** 2 / var)))


def _train_and_eval(x_tr, y_tr, x_te, y_te, seed, dynamic, k_reg):
    model = ProbabilisticRegressionModel(
        input_size=1,
        n_classes=K_MAX,
        max_n_classes_for_probabilistic_path=K_MAX,
        regression_strategy=RegressionStrategy.SEPARATE_HEADS,
        uncertainty_method=UncertaintyMethod.PROBABILISTIC,
        optimization_strategy=ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY,
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
    model.fit(x_tr, y_tr)
    preds = model.predict(x_te)
    if preds.ndim > 1:
        preds = preds.ravel()
    mse = float(np.mean((preds - y_te) ** 2))
    try:
        std = model.predict_uncertainty(x_te)
        log_var = np.log(np.clip(std ** 2, 1e-8, None))
        nll = _gaussian_nll(y_te, preds, log_var)
    except Exception:
        nll = float("nan")
    metrics = compute_kselection_metrics(model, x_te, K_MAX)
    return model, mse, nll, metrics


def main() -> None:
    results_path = OUT_DIR / "results.csv"
    rows: list[dict] = []
    total = len(NOISE_LEVELS) * len(VALID_PAIRS) * len(SEEDS)
    run_count = 0
    write_header = not results_path.exists()

    for noise_std in NOISE_LEVELS:
        print(f"\n=== noise_std = {noise_std} ===")
        for dynamic, k_reg in VALID_PAIRS:
            for seed in SEEDS:
                run_count += 1
                label = f"σ={noise_std}|{dynamic.name[:4]}|s{seed}"
                print(f"  [{run_count:2d}/{total}] {label} ...", end="", flush=True)
                x, y = _make_exponential(noise_std, seed)
                x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=seed)
                t0 = time.time()
                model = None
                try:
                    model, mse, nll, metrics = _train_and_eval(
                        x_tr, y_tr, x_te, y_te, seed, dynamic, k_reg,
                    )
                    wall = round(time.time() - t0, 1)
                    row = {
                        "noise_std": noise_std,
                        "dynamic": dynamic.name,
                        "k_reg": k_reg.name,
                        "seed": seed,
                        "mse": round(mse, 6),
                        "nll_gaussian": round(nll, 6),
                        "effective_k": round(metrics["effective_k"], 4),
                        "bypass_fraction": round(metrics["bypass_fraction"], 4),
                        "effective_k_nobypass": round(metrics["effective_k_nobypass"], 4),
                        "selection_perplexity": round(metrics["selection_perplexity"], 4),
                        "dead_mode_count": int(metrics["dead_mode_count"]),
                        "mean_max_p": round(metrics["mean_max_p"], 4),
                        "wall_time_s": wall,
                    }
                    print(f" MSE={mse:.4f} bypass={metrics['bypass_fraction']:.3f} ({wall:.0f}s)")
                except Exception as e:
                    wall = round(time.time() - t0, 1)
                    print(f" ERROR: {e}")
                    row = {
                        "noise_std": noise_std, "dynamic": dynamic.name, "k_reg": k_reg.name,
                        "seed": seed, "mse": float("nan"), "nll_gaussian": float("nan"),
                        "effective_k": float("nan"), "bypass_fraction": float("nan"),
                        "effective_k_nobypass": float("nan"),
                        "selection_perplexity": float("nan"), "dead_mode_count": -1,
                        "mean_max_p": float("nan"),
                        "wall_time_s": wall,
                    }
                finally:
                    del model
                    _release_memory()
                rows.append(row)
                pd.DataFrame([row]).to_csv(results_path, mode="a", header=write_header, index=False)
                write_header = False

    df = pd.read_csv(results_path)
    numeric_cols = [
        "mse", "nll_gaussian",
        "effective_k", "bypass_fraction", "effective_k_nobypass",
        "selection_perplexity", "dead_mode_count", "mean_max_p",
        "wall_time_s",
    ]
    agg = {c: ["mean", "std"] for c in numeric_cols if c in df.columns}
    summary = df.groupby(["noise_std", "dynamic", "k_reg"]).agg(agg)
    summary.columns = ["_".join(col) for col in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(OUT_DIR / "summary.csv", index=False)

    print("\n=== SNR-sweep summary (bypass fraction vs noise) ===")
    print(summary[["noise_std", "dynamic", "mse_mean", "bypass_fraction_mean", "effective_k_nobypass_mean"]].to_string(index=False))

    # Plot bypass vs noise_std
    dyn = summary[summary["dynamic"] == "SOFT_GATING"].sort_values("noise_std")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].errorbar(dyn["noise_std"], dyn["bypass_fraction_mean"], yerr=dyn["bypass_fraction_std"], marker="o")
    axes[0].set_xlabel("noise σ"); axes[0].set_ylabel("bypass_fraction"); axes[0].set_xscale("log")
    axes[0].set_title("Bypass fraction vs noise σ (SOFT_GATING, exp data)")
    axes[0].axhline(0.5, color="gray", ls="--", alpha=0.5, label="prior=0.5 (ELBO would pin here)")
    axes[0].legend()
    axes[1].errorbar(dyn["noise_std"], dyn["effective_k_nobypass_mean"], yerr=dyn["effective_k_nobypass_std"], marker="s", color="orange")
    axes[1].set_xlabel("noise σ"); axes[1].set_ylabel("effective_k_nobypass"); axes[1].set_xscale("log")
    axes[1].set_title("Effective k (non-bypass) vs noise σ")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "snr_curves.pdf")
    plt.close(fig)

    print(f"\nDone. Results in {OUT_DIR}/")


if __name__ == "__main__":
    main()
