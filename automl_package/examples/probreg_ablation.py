"""Paper A primary table: ProbabilisticRegression ablation sweep.

Sweep:
- regression_strategy in {SEPARATE_HEADS, SINGLE_HEAD_N_OUTPUTS, SINGLE_HEAD_FINAL_OUTPUT}
- loss_type in {NLL, BETA_NLL @ beta in {0.0, 0.5, 1.0}}
- target_transform in {None, "symlog"}
- optimization_strategy in {REGRESSION_ONLY, GRADIENT_STOP} (jointly vs stopgrad)
- dynamic-k selection in {NONE, SOFT_GATING, GUMBEL_SOFTMAX, STE}
- n_classes_regularization in {NONE, K_PENALTY, ELBO}

Runs on a shrunk synthetic set (heteroscedastic + multimodal + B1) to keep
wall-clock tractable. Writes per-run CSV + best-config summary.
"""

from __future__ import annotations

import json
import logging
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from automl_package.enums import (
    NClassesRegularization,
    NClassesSelectionMethod,
    ProbabilisticRegressionOptimizationStrategy,
    RegressionStrategy,
    UncertaintyMethod,
)
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel
from automl_package.models.selection_strategies.base_selection_strategy import DIRECT_REGRESSION_K_SENTINEL
from automl_package.utils.distributions import GaussianDistribution
from automl_package.utils.scoring import calculate_crps
from automl_package.utils.synthetic_datasets import load_fixture

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

OUT_DIR = Path(__file__).parent / "probreg_ablation_results"
OUT_DIR.mkdir(exist_ok=True)


def _make_hetero(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-5, 5, 800).reshape(-1, 1).astype(np.float32)
    y = (np.sin(x.ravel()) * 2 + 0.5 * x.ravel() +
         rng.normal(0.0, 0.1 + 0.4 * np.abs(x.ravel()))).astype(np.float32)
    return x, y


def _make_bimodal(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, 800).reshape(-1, 1).astype(np.float32)
    sign = rng.choice([-1.0, 1.0], size=800).reshape(-1, 1)
    y = (x + sign * 1.5 + rng.normal(0, 0.1, (800, 1))).ravel().astype(np.float32)
    return x, y


def _gauss_nll(y, mu, sigma) -> float:
    sigma = np.maximum(sigma, 1e-9)
    return float(np.mean(0.5 * (np.log(2 * np.pi * sigma**2) + ((y - mu) / sigma) ** 2)))


def run_one(ds_name: str, x_tr, y_tr, x_te, y_te,
            strategy: RegressionStrategy,
            selection: NClassesSelectionMethod,
            regularization: NClassesRegularization,
            loss_type: str, beta: float,
            target_transform: str | None,
            opt_strategy: ProbabilisticRegressionOptimizationStrategy,
            n_classes: int = 3, max_k: int = 7, n_epochs: int = 40, seed: int = 42) -> dict:
    t0 = time.perf_counter()
    m = ProbabilisticRegressionModel(
        input_size=x_tr.shape[1], n_classes=n_classes, max_n_classes_for_probabilistic_path=max_k,
        uncertainty_method=UncertaintyMethod.PROBABILISTIC,
        n_classes_selection_method=selection,
        n_classes_regularization=regularization,
        regression_strategy=strategy,
        loss_type=loss_type, beta=beta,
        target_transform=target_transform,
        optimization_strategy=opt_strategy,
        base_classifier_params={"hidden_layers": 1, "hidden_size": 64},
        regression_head_params={"hidden_layers": 0, "hidden_size": 32},
        n_epochs=n_epochs, learning_rate=0.01, early_stopping_rounds=15,
        validation_fraction=0.2, random_seed=seed,
        calculate_feature_importance=False,
    )
    m.fit(x_tr, y_tr)
    y_pred = m.predict(x_te)
    y_std = m.predict_uncertainty(x_te)
    mse = float(np.mean((y_te - y_pred) ** 2))
    nll = _gauss_nll(y_te, y_pred, y_std)
    dist = GaussianDistribution(y_pred, y_std)
    crps = calculate_crps(y_te, dist)
    # Mixture log-prob if available
    nll_mix: float | None = None
    if selection == NClassesSelectionMethod.NONE and strategy != RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT and target_transform is None:
        try:
            mix = m.predict_distribution(x_te)
            nll_mix = float(-np.mean(mix.log_prob(y_te)))
        except Exception:  # noqa: BLE001
            nll_mix = None
    mean_k: float | None = None
    if selection != NClassesSelectionMethod.NONE:
        try:
            x_t = torch.tensor(x_te, dtype=torch.float32).to(m.device)
            m.model.eval()
            with torch.no_grad():
                _, _, k_actual, _, _ = m.model(x_t)
            k_valid = k_actual[k_actual < DIRECT_REGRESSION_K_SENTINEL]
            mean_k = float(k_valid.float().mean().item()) if k_valid.numel() > 0 else None
        except Exception:  # noqa: BLE001
            mean_k = None
    return {
        "dataset": ds_name, "strategy": strategy.value, "selection": selection.value,
        "regularization": regularization.value, "loss": loss_type, "beta": beta,
        "target_transform": target_transform or "none",
        "opt_strategy": opt_strategy.value,
        "mse": mse, "nll": nll, "crps": crps, "nll_mixture": nll_mix,
        "mean_k": mean_k, "seconds": time.perf_counter() - t0,
    }


def main(subsample: bool = True) -> None:
    rows: list[dict] = []
    datasets: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("heteroscedastic", *_make_hetero()),
        ("bimodal", *_make_bimodal()),
    ]
    try:
        b1 = load_fixture("b1_gravitational")
        # subsample B1 to 1500 for speed
        sel = np.random.RandomState(0).choice(len(b1.y), size=min(1500, len(b1.y)), replace=False)
        datasets.append(("b1_gravitational", b1.x[sel].astype(np.float32), b1.y[sel]))
    except FileNotFoundError:
        pass

    strategy_grid = (
        RegressionStrategy.SEPARATE_HEADS,
        RegressionStrategy.SINGLE_HEAD_N_OUTPUTS,
        RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT,
    )
    selection_grid = (
        NClassesSelectionMethod.NONE,
        NClassesSelectionMethod.SOFT_GATING,
        NClassesSelectionMethod.GUMBEL_SOFTMAX,
        NClassesSelectionMethod.STE,
        NClassesSelectionMethod.REINFORCE,
    )
    regularization_grid = (
        NClassesRegularization.NONE,
        NClassesRegularization.K_PENALTY,
        NClassesRegularization.ELBO,
    )
    loss_grid = (("nll", 0.5), ("beta_nll", 0.0), ("beta_nll", 0.5), ("beta_nll", 1.0))
    transform_grid = (None, "symlog")
    opt_grid = (
        ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY,
        ProbabilisticRegressionOptimizationStrategy.GRADIENT_STOP,
    )

    if subsample:
        # Collapse the full product to headline cells. Skip SINGLE_HEAD_N_OUTPUTS
        # (dominated by SEP_HEADS on prior benchmarks) and beta_nll. These cells
        # are recovered in the full (subsample=False) run.
        strategy_grid = (
            RegressionStrategy.SEPARATE_HEADS, RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT,
        )
        selection_grid = (
            NClassesSelectionMethod.NONE, NClassesSelectionMethod.SOFT_GATING,
            NClassesSelectionMethod.STE, NClassesSelectionMethod.REINFORCE,
        )
        regularization_grid = (NClassesRegularization.NONE, NClassesRegularization.ELBO)
        loss_grid = (("nll", 0.5),)
        transform_grid = (None,)
        opt_grid = (ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY,)

    grid = list(product(strategy_grid, selection_grid, regularization_grid, loss_grid, transform_grid, opt_grid))
    logger.info(f"Ablation grid size = {len(grid)} x {len(datasets)} datasets = {len(grid) * len(datasets)} runs")

    for ds_name, x, y in datasets:
        x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.3, random_state=0)
        for strategy, selection, regularization, (loss_type, beta), transform, opt in grid:
            # Skip incompatible combos
            if selection == NClassesSelectionMethod.NONE and regularization != NClassesRegularization.NONE:
                continue  # regularization only makes sense with dynamic-k
            try:
                row = run_one(ds_name, x_tr, y_tr, x_te, y_te,
                              strategy, selection, regularization,
                              loss_type, beta, transform, opt)
                rows.append(row)
                logger.info(
                    f"{ds_name} {strategy.value[:10]:<10} sel={selection.value[:9]:<9} "
                    f"reg={regularization.value:<10} loss={loss_type}@beta={beta}  "
                    f"mse={row['mse']:.4f} nll={row['nll']:.3f} crps={row['crps']:.3f}"
                )
            except Exception as e:  # noqa: BLE001
                logger.exception(f"config crashed: {e}")
                rows.append({"dataset": ds_name, "strategy": strategy.value, "selection": selection.value,
                              "regularization": regularization.value, "loss": loss_type, "beta": beta,
                              "target_transform": transform or "none", "error": str(e)})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "probreg_ablation.csv", index=False)
    (OUT_DIR / "probreg_ablation.json").write_text(json.dumps(rows, indent=2, default=str))
    if "nll" in df.columns:
        best = df.dropna(subset=["nll"]).sort_values("nll").groupby(["dataset"]).head(3)
        best.to_csv(OUT_DIR / "top3_per_dataset.csv", index=False)
    logger.info(f"Wrote {len(rows)} rows.")


if __name__ == "__main__":
    main()
