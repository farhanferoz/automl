"""Ordering-constraint weight ablation for Cell B.

Q3 from Phase-2 follow-up: is the ordering penalty what actually makes Cell B
better than Cell C, and is weight=1.0 well-calibrated?

Grid: Cell B × SOFT_GATING+NONE × k_max=10 × ordering_weight ∈ {0.0, 0.3, 1.0, 3.0}
      × 4 datasets × 5 seeds = 80 runs.

Cell B fixed: GAUSSIAN_LTV + REGRESSION_ONLY + SEPARATE_HEADS.
We override ordering_constraint_weight explicitly (no auto-resolution).

If weight=0 MSE matches weight=1 MSE, the penalty is doing nothing.
If weight=1 MSE is notably better than weight=0, the penalty helps.
Sweep across {0.3, 1.0, 3.0} tells us if weight=1 is near-optimal or off.
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
import torch.nn.functional as F  # noqa: N812
from sklearn.model_selection import train_test_split

from automl_package.enums import (
    NClassesRegularization,
    NClassesSelectionMethod,
    ProbabilisticRegressionOptimizationStrategy,
    ProbRegLossType,
    RegressionStrategy,
    UncertaintyMethod,
)
from automl_package.examples._toy_datasets import make_datasets
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel

mpl.use("Agg")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

OUT_DIR = Path(__file__).parent / "probreg_ordering_ablation_results"
OUT_DIR.mkdir(exist_ok=True)

SEEDS = (42, 123, 7, 2026, 31)
ORDERING_WEIGHTS = (0.0, 0.3, 1.0, 3.0)
K_MAX = 10
N_EPOCHS = 80
LR = 0.01
EARLY_STOP = 15
VAL_FRAC = 0.2
BATCH_SIZE = 64


def _release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()


def _gaussian_nll(y, mean, log_var):
    var = np.exp(log_var)
    return float(np.mean(0.5 * (math.log(2 * math.pi) + log_var + (y - mean) ** 2 / var)))


def _compute_effective_k(model, x_te, k_max):
    if model.model.n_classes_predictor is None:
        return float(k_max), 0.0, float(k_max)
    x_t = torch.tensor(x_te, dtype=torch.float32).to(model.device)
    model.model.eval()
    with torch.no_grad():
        logits = model.model.n_classes_predictor(x_t)
        probs = F.softmax(logits, dim=-1)
        n_modes = probs.size(1)
        k_vals = torch.tensor([i + 2 for i in range(n_modes - 1)] + [k_max], dtype=torch.float32).to(model.device)
        eff_k = float((probs * k_vals).sum(-1).mean().cpu().item())
        bypass_mass = probs[:, -1]
        bypass = float(bypass_mass.mean().cpu().item())
        prob_probs = probs[:, :-1]
        denom = prob_probs.sum(-1, keepdim=True).clamp_min(1e-8)
        prob_k_vals = k_vals[:-1]
        expected_k_prob = (prob_probs / denom * prob_k_vals).sum(-1)
        mask = (1.0 - bypass_mass) > 1e-6
        eff_k_nb = float(expected_k_prob[mask].mean().cpu().item()) if mask.any() else float("nan")
    return eff_k, bypass, eff_k_nb


def _train_and_eval(x_tr, y_tr, x_te, y_te, seed, ordering_weight):
    model = ProbabilisticRegressionModel(
        input_size=1,
        n_classes=K_MAX,
        max_n_classes_for_probabilistic_path=K_MAX,
        regression_strategy=RegressionStrategy.SEPARATE_HEADS,
        uncertainty_method=UncertaintyMethod.PROBABILISTIC,
        optimization_strategy=ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY,
        prob_reg_loss_type=ProbRegLossType.GAUSSIAN_LTV,
        n_classes_selection_method=NClassesSelectionMethod.SOFT_GATING,
        n_classes_regularization=NClassesRegularization.NONE,
        ordering_constraint_weight=ordering_weight,  # explicit override
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
    # Guard: explicit weight must have stuck.
    assert model.ordering_constraint_weight == ordering_weight, (
        f"ordering_constraint_weight drift: requested {ordering_weight}, got {model.ordering_constraint_weight}"
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
    eff_k, bypass, eff_k_nb = _compute_effective_k(model, x_te, K_MAX)
    return model, mse, nll, eff_k, bypass, eff_k_nb


def main() -> None:
    datasets = make_datasets()
    results_path = OUT_DIR / "results.csv"
    total = len(datasets) * len(ORDERING_WEIGHTS) * len(SEEDS)
    run_count = 0
    write_header = not results_path.exists()

    for ds_name, x, y in datasets:
        print(f"\n=== Dataset: {ds_name} ===")
        splits = {seed: train_test_split(x, y, test_size=0.2, random_state=seed) for seed in SEEDS}
        for w in ORDERING_WEIGHTS:
            for seed in SEEDS:
                run_count += 1
                label = f"w={w}|s{seed}"
                print(f"  [{run_count:3d}/{total}] {ds_name} {label} ...", end="", flush=True)
                x_tr, x_te, y_tr, y_te = splits[seed]
                t0 = time.time()
                model = None
                try:
                    model, mse, nll, eff_k, bypass, eff_k_nb = _train_and_eval(
                        x_tr, y_tr, x_te, y_te, seed, w,
                    )
                    wall = round(time.time() - t0, 1)
                    row = {
                        "dataset": ds_name, "ordering_weight": w, "seed": seed,
                        "mse": round(mse, 6), "nll_gaussian": round(nll, 6),
                        "effective_k": round(eff_k, 4), "bypass_fraction": round(bypass, 4),
                        "effective_k_nobypass": round(eff_k_nb, 4), "wall_time_s": wall,
                    }
                    print(f" MSE={mse:.4f} ({wall:.0f}s)")
                except Exception as e:
                    wall = round(time.time() - t0, 1)
                    print(f" ERROR: {e}")
                    row = {
                        "dataset": ds_name, "ordering_weight": w, "seed": seed,
                        "mse": float("nan"), "nll_gaussian": float("nan"),
                        "effective_k": float("nan"), "bypass_fraction": float("nan"),
                        "effective_k_nobypass": float("nan"), "wall_time_s": wall,
                    }
                finally:
                    del model
                    _release_memory()
                pd.DataFrame([row]).to_csv(results_path, mode="a", header=write_header, index=False)
                write_header = False

    df = pd.read_csv(results_path)
    numeric_cols = ["mse", "nll_gaussian", "effective_k", "bypass_fraction", "effective_k_nobypass", "wall_time_s"]
    agg = {c: ["mean", "std"] for c in numeric_cols if c in df.columns}
    summary = df.groupby(["dataset", "ordering_weight"]).agg(agg)
    summary.columns = ["_".join(col) for col in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(OUT_DIR / "summary.csv", index=False)

    print("\n=== Ordering-weight ablation summary ===")
    print(summary[["dataset", "ordering_weight", "mse_mean", "mse_std", "nll_gaussian_mean"]].to_string(index=False))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, ds in zip(axes.ravel(), ["heteroscedastic", "bimodal", "piecewise", "exponential"]):
        sub = summary[summary["dataset"] == ds].sort_values("ordering_weight")
        ax.errorbar(sub["ordering_weight"], sub["mse_mean"], yerr=sub["mse_std"], marker="o")
        ax.set_title(ds)
        ax.set_xlabel("ordering_constraint_weight")
        ax.set_ylabel("MSE")
    fig.suptitle("Cell B — MSE vs ordering_constraint_weight (SOFT_GATING+NONE, k_max=10)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ordering_weight_curves.pdf")
    plt.close(fig)

    print(f"\nDone. Results in {OUT_DIR}/")


if __name__ == "__main__":
    main()
