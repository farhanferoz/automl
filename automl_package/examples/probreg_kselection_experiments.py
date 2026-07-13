"""ProbReg k-selection experimental program.

Runs all sweeps needed to characterise k-selection behaviour:

  Sweep 1 (k_max):   Toy A + Toy B(K_true=3) × k_max ∈ {5, 10, 20, 40}
                     Tests cap tracking on no-intrinsic-k vs known-intrinsic-k.
  Sweep 2 (noise):   Toy A × σ ∈ {0.1, 0.5, 1.0, 2.0}
                     Tests CE-confidence vs bypass-handoff hypothesis.
  Sweep 3 (k_reg):   Toy B(K_true=3) at k_max=20 × {NONE, K_PENALTY, ELBO}
                     Tests whether any regulariser concentrates on K_true.
  Diagnostic B:      Toy B(K_true ∈ {2, 3, 5}) at k_max=10, per-epoch logging.
                     Tests whether trajectories find K_true at any point.

Outputs (probreg_kselection_experiments_results/):
  sweep1_kmax.csv
  sweep2_noise.csv
  sweep3_kreg.csv
  trajectory_toyB_kT{K}.csv   (one per K_true)
  marginal_p_toyB_kT{K}.npy   (one per K_true)
"""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import Any

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
from automl_package.examples._toy_datasets import make_toy_a, make_toy_b
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel

OUT_DIR = Path(__file__).resolve().parent / "probreg_kselection_experiments_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.getLogger("automl_package").setLevel(logging.WARNING)

N_EPOCHS = 100
LR = 0.01
EARLY_STOP = 15
VAL_FRAC = 0.2
BATCH_SIZE = 64
SEEDS = [42, 123, 7]  # 3 seeds; add more if variance is high

DIAGNOSTIC_N_EPOCHS = 200
DIAGNOSTIC_EARLY_STOP = 50


def _release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()


def _make_model(
    k_max: int,
    seed: int,
    selection: NClassesSelectionMethod = NClassesSelectionMethod.SOFT_GATING,
    k_reg: NClassesRegularization = NClassesRegularization.NONE,
    n_epochs: int = N_EPOCHS,
    early_stop: int = EARLY_STOP,
) -> ProbabilisticRegressionModel:
    return ProbabilisticRegressionModel(
        input_size=1,
        n_classes=k_max,
        max_n_classes_for_probabilistic_path=k_max,
        regression_strategy=RegressionStrategy.SEPARATE_HEADS,
        uncertainty_method=UncertaintyMethod.PROBABILISTIC,
        optimization_strategy=ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY,
        prob_reg_loss_type=ProbRegLossType.GAUSSIAN_LTV,
        n_classes_selection_method=selection,
        n_classes_regularization=k_reg,
        use_anchored_heads=False,
        constrain_middle_class=True,
        use_monotonic_constraints=False,
        n_epochs=n_epochs,
        learning_rate=LR,
        early_stopping_rounds=early_stop,
        validation_fraction=VAL_FRAC,
        batch_size=BATCH_SIZE,
        random_seed=seed,
        calculate_feature_importance=False,
    )


def _train_eval_one(
    x: np.ndarray, y: np.ndarray, k_max: int, seed: int,
    selection: NClassesSelectionMethod = NClassesSelectionMethod.SOFT_GATING,
    k_reg: NClassesRegularization = NClassesRegularization.NONE,
) -> dict[str, Any]:
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=seed)
    model = _make_model(k_max, seed, selection, k_reg)
    t0 = time.time()
    model.fit(x_tr, y_tr)
    wall = time.time() - t0
    preds = model.predict(x_te)
    if preds.ndim > 1:
        preds = preds.ravel()
    mse = float(np.mean((preds - y_te) ** 2))
    metrics = compute_kselection_metrics(model, x_te, k_max)
    out = {
        "mse": mse,
        "wall_time_s": wall,
        "effective_k": metrics["effective_k"],
        "effective_k_nobypass": metrics["effective_k_nobypass"],
        "bypass_fraction": metrics["bypass_fraction"],
        "selection_perplexity": metrics["selection_perplexity"],
        "dead_mode_count": metrics["dead_mode_count"],
        "mean_max_p": metrics["mean_max_p"],
        "marginal_p": metrics["marginal_p"].tolist(),
    }
    del model
    _release_memory()
    return out


# ---------------------------------------------------------------------------
# Sweep 1: k_max sweep on Toy A and Toy B (K_true=3)
# ---------------------------------------------------------------------------

def sweep1_kmax() -> None:
    print("\n" + "=" * 60)
    print("SWEEP 1: k_max sweep")
    print("=" * 60)
    rows = []
    out_path = OUT_DIR / "sweep1_kmax.csv"
    write_header = True
    K_MAX_VALUES = [5, 10, 20, 40]

    configs = [
        ("toy_a", lambda s: make_toy_a(n=800, sigma=0.5, seed=s)),
        ("toy_b_kT3", lambda s: make_toy_b(n=800, k_true=3, separation=4.0, sigma=0.3, seed=s)),
    ]

    total = len(configs) * len(K_MAX_VALUES) * len(SEEDS)
    count = 0
    for ds_name, gen in configs:
        for k_max in K_MAX_VALUES:
            for seed in SEEDS:
                count += 1
                x, y = gen(seed)
                t0 = time.time()
                try:
                    out = _train_eval_one(x, y, k_max, seed)
                    print(f"  [{count:2d}/{total}] {ds_name:11s} k_max={k_max:2d} seed={seed:3d}  "
                          f"MSE={out['mse']:.3f}  perp={out['selection_perplexity']:.2f}  "
                          f"E[k|nb]={out['effective_k_nobypass']:.2f}  bypass={out['bypass_fraction']:.2f}  "
                          f"({out['wall_time_s']:.0f}s)")
                    row = {"dataset": ds_name, "k_max": k_max, "seed": seed, **out}
                except Exception as e:
                    wall = time.time() - t0
                    print(f"  [{count:2d}/{total}] {ds_name:11s} k_max={k_max:2d} seed={seed:3d}  ERROR: {e}")
                    row = {"dataset": ds_name, "k_max": k_max, "seed": seed, "mse": float("nan"),
                           "wall_time_s": wall, "effective_k": float("nan"), "effective_k_nobypass": float("nan"),
                           "bypass_fraction": float("nan"), "selection_perplexity": float("nan"),
                           "dead_mode_count": -1, "mean_max_p": float("nan"), "marginal_p": []}
                rows.append(row)
                pd.DataFrame([row]).to_csv(out_path, mode="a", header=write_header, index=False)
                write_header = False
    print(f"  -> {out_path}")


# ---------------------------------------------------------------------------
# Sweep 2: noise sweep on Toy A (homoscedastic, controlled σ)
# ---------------------------------------------------------------------------

def sweep2_noise() -> None:
    print("\n" + "=" * 60)
    print("SWEEP 2: noise sweep on Toy A")
    print("=" * 60)
    rows = []
    out_path = OUT_DIR / "sweep2_noise.csv"
    write_header = True
    SIGMA_VALUES = [0.1, 0.5, 1.0, 2.0]
    K_MAX = 10

    total = len(SIGMA_VALUES) * len(SEEDS)
    count = 0
    for sigma in SIGMA_VALUES:
        for seed in SEEDS:
            count += 1
            x, y = make_toy_a(n=800, sigma=sigma, seed=seed)
            t0 = time.time()
            try:
                out = _train_eval_one(x, y, K_MAX, seed)
                print(f"  [{count:2d}/{total}] σ={sigma:.2f} seed={seed:3d}  "
                      f"MSE={out['mse']:.3f}  perp={out['selection_perplexity']:.2f}  "
                      f"E[k|nb]={out['effective_k_nobypass']:.2f}  max_p={out['mean_max_p']:.3f}  "
                      f"bypass={out['bypass_fraction']:.2f}  ({out['wall_time_s']:.0f}s)")
                row = {"sigma": sigma, "seed": seed, **out}
            except Exception as e:
                wall = time.time() - t0
                print(f"  [{count:2d}/{total}] σ={sigma:.2f} seed={seed:3d}  ERROR: {e}")
                row = {"sigma": sigma, "seed": seed, "mse": float("nan"), "wall_time_s": wall,
                       "effective_k": float("nan"), "effective_k_nobypass": float("nan"),
                       "bypass_fraction": float("nan"), "selection_perplexity": float("nan"),
                       "dead_mode_count": -1, "mean_max_p": float("nan"), "marginal_p": []}
            rows.append(row)
            pd.DataFrame([row]).to_csv(out_path, mode="a", header=write_header, index=False)
            write_header = False
    print(f"  -> {out_path}")


# ---------------------------------------------------------------------------
# Sweep 3: k_reg efficacy on Toy B (K_true=3)
# ---------------------------------------------------------------------------

def sweep3_kreg() -> None:
    print("\n" + "=" * 60)
    print("SWEEP 3: k_reg efficacy on Toy B (K_true=3)")
    print("=" * 60)
    rows = []
    out_path = OUT_DIR / "sweep3_kreg.csv"
    write_header = True
    KREG_VALUES = [NClassesRegularization.NONE, NClassesRegularization.K_PENALTY, NClassesRegularization.ELBO]
    K_MAX = 20

    total = len(KREG_VALUES) * len(SEEDS)
    count = 0
    for k_reg in KREG_VALUES:
        for seed in SEEDS:
            count += 1
            x, y = make_toy_b(n=800, k_true=3, separation=4.0, sigma=0.3, seed=seed)
            t0 = time.time()
            try:
                out = _train_eval_one(x, y, K_MAX, seed, k_reg=k_reg)
                print(f"  [{count:2d}/{total}] k_reg={k_reg.name:10s} seed={seed:3d}  "
                      f"MSE={out['mse']:.3f}  perp={out['selection_perplexity']:.2f}  "
                      f"E[k|nb]={out['effective_k_nobypass']:.2f}  bypass={out['bypass_fraction']:.2f}  "
                      f"({out['wall_time_s']:.0f}s)")
                row = {"k_reg": k_reg.name, "seed": seed, **out}
            except Exception as e:
                wall = time.time() - t0
                print(f"  [{count:2d}/{total}] k_reg={k_reg.name:10s} seed={seed:3d}  ERROR: {e}")
                row = {"k_reg": k_reg.name, "seed": seed, "mse": float("nan"), "wall_time_s": wall,
                       "effective_k": float("nan"), "effective_k_nobypass": float("nan"),
                       "bypass_fraction": float("nan"), "selection_perplexity": float("nan"),
                       "dead_mode_count": -1, "mean_max_p": float("nan"), "marginal_p": []}
            rows.append(row)
            pd.DataFrame([row]).to_csv(out_path, mode="a", header=write_header, index=False)
            write_header = False
    print(f"  -> {out_path}")


# ---------------------------------------------------------------------------
# Diagnostic: Toy B per-epoch logging for K_true ∈ {2, 3, 5}
# ---------------------------------------------------------------------------

def diagnostic_toy_b() -> None:
    print("\n" + "=" * 60)
    print("DIAGNOSTIC: Toy B trajectory per-epoch for K_true ∈ {2, 3, 5}")
    print("=" * 60)
    K_MAX = 10
    K_TRUE_VALUES = [2, 3, 5]
    SEED = 42

    for k_true in K_TRUE_VALUES:
        x, y = make_toy_b(n=800, k_true=k_true, separation=4.0, sigma=0.3, seed=SEED)
        x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=SEED)
        model = _make_model(K_MAX, SEED, n_epochs=DIAGNOSTIC_N_EPOCHS, early_stop=DIAGNOSTIC_EARLY_STOP)

        trajectory: list[dict] = []
        marginal_history: list[np.ndarray] = []

        def on_epoch_end(epoch: int, model: Any, val_loss: float | None) -> None:
            m = compute_kselection_metrics(model, x_te, K_MAX)
            trajectory.append({
                "epoch": epoch, "val_loss": val_loss,
                "effective_k": m["effective_k"], "effective_k_nobypass": m["effective_k_nobypass"],
                "bypass_fraction": m["bypass_fraction"],
                "selection_perplexity": m["selection_perplexity"],
                "dead_mode_count": m["dead_mode_count"], "mean_max_p": m["mean_max_p"],
            })
            marginal_history.append(m["marginal_p"])

        model.epoch_callback = on_epoch_end
        t0 = time.time()
        model.fit(x_tr, y_tr)
        wall = time.time() - t0

        traj_df = pd.DataFrame(trajectory)
        traj_df.to_csv(OUT_DIR / f"trajectory_toyB_kT{k_true}.csv", index=False)
        marginals = np.stack(marginal_history) if marginal_history else np.zeros((0, 0))
        np.save(OUT_DIR / f"marginal_p_toyB_kT{k_true}.npy", marginals)

        final = traj_df.iloc[-1]
        print(f"  K_true={k_true}: {len(traj_df)} epochs ({wall:.0f}s)  final: "
              f"perp={final['selection_perplexity']:.2f} E[k|nb]={final['effective_k_nobypass']:.2f} "
              f"dead={int(final['dead_mode_count'])} bypass={final['bypass_fraction']:.2f} "
              f"max_p={final['mean_max_p']:.3f}")
        if marginals.size > 0:
            print(f"  final marginal_p: {np.round(marginals[-1], 3)}")
        del model
        _release_memory()


def main() -> None:
    t0 = time.time()
    sweep1_kmax()
    sweep2_noise()
    sweep3_kreg()
    diagnostic_toy_b()
    print(f"\n=== ALL EXPERIMENTS DONE in {(time.time() - t0) / 60:.1f} min ===")


if __name__ == "__main__":
    main()
