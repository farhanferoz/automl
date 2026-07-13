"""ELBO prior diagnosis on Toy B (K_true=3) at k_max=20.

Compares ELBO with default uniform k-prior (the configuration that produced
perp≈19 in the main sweep) against ELBO with the geometric prior at three
values of λ ∈ {0.05, 0.2, 0.5}. Tests whether the geometric prior recovers
intrinsic-k concentration.
"""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path

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
from automl_package.examples._toy_datasets import make_toy_b
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel

OUT_DIR = Path(__file__).resolve().parent / "probreg_elbo_prior_check_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
logging.getLogger("automl_package").setLevel(logging.WARNING)

K_MAX = 20
N_EPOCHS = 100
SEEDS = [42, 123, 7]


def _release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()


def run_one(prior_type: str, lam: float | None, seed: int) -> dict:
    x, y = make_toy_b(n=800, k_true=3, separation=4.0, sigma=0.3, seed=seed)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=seed)
    kwargs = {
        "input_size": 1,
        "n_classes": K_MAX,
        "max_n_classes_for_probabilistic_path": K_MAX,
        "regression_strategy": RegressionStrategy.SEPARATE_HEADS,
        "uncertainty_method": UncertaintyMethod.PROBABILISTIC,
        "optimization_strategy": ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY,
        "prob_reg_loss_type": ProbRegLossType.GAUSSIAN_LTV,
        "n_classes_selection_method": NClassesSelectionMethod.SOFT_GATING,
        "n_classes_regularization": NClassesRegularization.ELBO,
        "k_prior_type": prior_type,
        "use_anchored_heads": False,
        "constrain_middle_class": True,
        "use_monotonic_constraints": False,
        "n_epochs": N_EPOCHS,
        "learning_rate": 0.01,
        "early_stopping_rounds": 15,
        "validation_fraction": 0.2,
        "batch_size": 64,
        "random_seed": seed,
        "calculate_feature_importance": False,
    }
    if lam is not None:
        kwargs["k_prior_geometric_lambda"] = lam
    model = ProbabilisticRegressionModel(**kwargs)
    t0 = time.time()
    model.fit(x_tr, y_tr)
    wall = time.time() - t0
    preds = model.predict(x_te)
    if preds.ndim > 1:
        preds = preds.ravel()
    mse = float(np.mean((preds - y_te) ** 2))
    m = compute_kselection_metrics(model, x_te, K_MAX)
    out = {
        "prior_type": prior_type,
        "lambda": lam if lam is not None else 0.0,
        "seed": seed,
        "mse": mse,
        "wall_time_s": wall,
        "effective_k_nobypass": m["effective_k_nobypass"],
        "selection_perplexity": m["selection_perplexity"],
        "bypass_fraction": m["bypass_fraction"],
        "dead_mode_count": int(m["dead_mode_count"]),
        "mean_max_p": m["mean_max_p"],
        "marginal_p": m["marginal_p"].tolist(),
    }
    del model
    _release_memory()
    return out


def main() -> None:
    configs = [
        ("uniform", None),
        ("geometric", 0.05),
        ("geometric", 0.2),
        ("geometric", 0.5),
    ]
    rows = []
    out_path = OUT_DIR / "results.csv"
    write_header = True
    total = len(configs) * len(SEEDS)
    count = 0
    for prior_type, lam in configs:
        for seed in SEEDS:
            count += 1
            print(f"[{count:2d}/{total}] prior={prior_type} λ={lam} seed={seed} ...", end="", flush=True)
            out = run_one(prior_type, lam, seed)
            print(f" perp={out['selection_perplexity']:.2f} E[k|nb]={out['effective_k_nobypass']:.2f} "
                  f"bypass={out['bypass_fraction']:.2f} dead={out['dead_mode_count']:2d} "
                  f"max_p={out['mean_max_p']:.3f} ({out['wall_time_s']:.0f}s)")
            rows.append(out)
            pd.DataFrame([out]).to_csv(out_path, mode="a", header=write_header, index=False)
            write_header = False

    df = pd.DataFrame(rows)
    print("\n=== SUMMARY (k_max=20, K_true=3; ideal: E[k|nb]=3, perp=1) ===")
    print(df.groupby(["prior_type", "lambda"])[
        ["effective_k_nobypass", "selection_perplexity", "bypass_fraction", "dead_mode_count", "mean_max_p"]
    ].agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()
