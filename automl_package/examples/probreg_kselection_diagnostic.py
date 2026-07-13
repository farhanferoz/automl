"""ProbReg k-selection diagnostic — single config, per-epoch logging.

Trains one ProbabilisticRegressionModel on Toy A (smooth unimodal) at k_max=10
with SOFT_GATING + k_reg=NONE, capturing the new k-selection metrics
(perplexity, dead-mode count, mean-max-p, marginal_p) at every epoch.

Outputs (probreg_kselection_diagnostic_results/):
  trajectory.csv     — one row per epoch with all scalar metrics + val_loss
  marginal_p.npy     — (n_epochs, n_k_modes) array of per-mode marginal probs
  summary.txt        — terminal-printed summary

Answers Q2 (distribution shape) and Q6 (training trajectory) for one config.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from automl_package.enums import (
    NClassesRegularization,
    NClassesSelectionMethod,
    ProbabilisticRegressionOptimizationStrategy,
    ProbRegLossType,
    RegressionStrategy,
    UncertaintyMethod,
)
from automl_package.examples._kselection_metrics import compute_kselection_metrics
from automl_package.examples._toy_datasets import make_toy_a
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel

OUT_DIR = Path(__file__).resolve().parent / "probreg_kselection_diagnostic_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_MAX = 10
N_EPOCHS = 200
LR = 0.01
EARLY_STOP = 50
VAL_FRAC = 0.2
BATCH_SIZE = 64
SIGMA = 0.5
SEED = 42


def main() -> None:
    x, y = make_toy_a(n=800, sigma=SIGMA, seed=SEED)
    n_te = int(len(x) * VAL_FRAC)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(x))
    te_idx, tr_idx = perm[:n_te], perm[n_te:]
    x_tr, y_tr = x[tr_idx], y[tr_idx]
    x_te, y_te = x[te_idx], y[te_idx]

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
        use_anchored_heads=False,
        constrain_middle_class=True,
        use_monotonic_constraints=False,
        n_epochs=N_EPOCHS,
        learning_rate=LR,
        early_stopping_rounds=EARLY_STOP,
        validation_fraction=VAL_FRAC,
        batch_size=BATCH_SIZE,
        random_seed=SEED,
        calculate_feature_importance=False,
    )

    trajectory: list[dict] = []
    marginal_history: list[np.ndarray] = []

    def on_epoch_end(epoch: int, model: ProbabilisticRegressionModel, val_loss: float | None) -> None:
        m = compute_kselection_metrics(model, x_te, K_MAX)
        trajectory.append({
            "epoch": epoch,
            "val_loss": val_loss,
            "effective_k": m["effective_k"],
            "effective_k_nobypass": m["effective_k_nobypass"],
            "bypass_fraction": m["bypass_fraction"],
            "selection_perplexity": m["selection_perplexity"],
            "dead_mode_count": m["dead_mode_count"],
            "mean_max_p": m["mean_max_p"],
        })
        marginal_history.append(m["marginal_p"])

    model.epoch_callback = on_epoch_end
    model.fit(x_tr, y_tr)

    traj_df = pd.DataFrame(trajectory)
    traj_df.to_csv(OUT_DIR / "trajectory.csv", index=False)

    marginals = np.stack(marginal_history) if marginal_history else np.zeros((0, 0))
    np.save(OUT_DIR / "marginal_p.npy", marginals)

    n_k = K_MAX - 1  # k ∈ {2..k_max}
    uniform_perplexity = float(n_k)
    uniform_E_k = (2.0 + K_MAX) / 2.0

    print("\n=== diagnostic summary ===")
    print(f"Config: Toy A σ={SIGMA}, k_max={K_MAX}, SOFT_GATING + NONE, seed={SEED}")
    print(f"Epochs trained: {len(traj_df)}")
    print(f"Reference values: uniform perplexity={uniform_perplexity:.2f}, uniform E[k]={uniform_E_k:.2f}")
    print()
    print("Final epoch metrics:")
    print(traj_df.tail(1).to_string(index=False))
    print()
    print("Trajectory snapshots (every ~25% of training):")
    n = len(traj_df)
    if n >= 4:
        snapshot_idxs = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    else:
        snapshot_idxs = list(range(n))
    print(traj_df.iloc[snapshot_idxs][[
        "epoch", "val_loss", "effective_k_nobypass", "selection_perplexity",
        "dead_mode_count", "mean_max_p", "bypass_fraction",
    ]].to_string(index=False))
    print()
    if marginals.size > 0:
        print(f"Final-epoch marginal_p (length {n_k}, sums to 1):")
        print(np.round(marginals[-1], 3))


if __name__ == "__main__":
    main()
