"""Multi-seed averaging infrastructure.

Runs a short list of models on fixed datasets across multiple seeds and
reports mean +/- std MSE/NLL. Can plug in any callable that returns a
(MSE, NLL) tuple.

Default sweep: heteroscedastic toy, n=500, 5 seeds, three representative
models — ProbReg(SEP, k=3), FlexNN(ELBO), XGBoost.

Intended for autonomous execution; robust against per-model crashes.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from automl_package.enums import (
    DepthRegularization,
    LayerSelectionMethod,
    NClassesSelectionMethod,
    RegressionStrategy,
    UncertaintyMethod,
)
from automl_package.models.flexible_neural_network import FlexibleHiddenLayersNN
from automl_package.models.lightgbm_model import LightGBMModel
from automl_package.models.neural_network import PyTorchNeuralNetwork
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel
from automl_package.models.xgboost_model import XGBoostModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

OUT_DIR = Path(__file__).parent / "multi_seed_results"
OUT_DIR.mkdir(exist_ok=True)

SEEDS = (0, 1, 2, 3, 4)
DATASET_N = 600


def _hetero_data(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-5, 5, DATASET_N).reshape(-1, 1).astype(np.float32)
    y = (np.sin(x.ravel()) * 2 + 0.5 * x.ravel() + rng.normal(0, 0.1 + 0.4 * np.abs(x.ravel()))).astype(np.float32)
    return x, y


def _gauss_nll(y, mu, sigma) -> float:
    sigma = np.maximum(sigma, 1e-9)
    return float(np.mean(0.5 * (np.log(2 * np.pi * sigma**2) + ((y - mu) / sigma) ** 2)))


def build_probreg(seed: int) -> ProbabilisticRegressionModel:
    return ProbabilisticRegressionModel(
        input_size=1, n_classes=3, max_n_classes_for_probabilistic_path=5,
        uncertainty_method=UncertaintyMethod.PROBABILISTIC,
        n_classes_selection_method=NClassesSelectionMethod.NONE,
        regression_strategy=RegressionStrategy.SEPARATE_HEADS,
        base_classifier_params={"hidden_layers": 1, "hidden_size": 64},
        regression_head_params={"hidden_layers": 0, "hidden_size": 32},
        n_epochs=60, learning_rate=0.01, early_stopping_rounds=15, validation_fraction=0.2,
        random_seed=seed, calculate_feature_importance=False,
    )


def build_flex(seed: int) -> FlexibleHiddenLayersNN:
    # SOFT_GATING is RETIRED under the nested ladder (MASTER Decision 29); the reported row is
    # labelled "FlexNN(ELBO)" and the label itself commits to ELBO being active, so this stays a
    # labelled comparison arm via the explicit opt-out rather than swapping to a survivor that
    # would silently make the label describe a config that no longer ran. n_predictor_layers=1 is
    # now explicit too -- FP-12's default-construction fix changed the class default to 0
    # (required by the NONE/NESTED survivors), which this call relied on implicitly before.
    return FlexibleHiddenLayersNN(
        input_size=1, max_hidden_layers=5, hidden_size=32, n_predictor_layers=1,
        layer_selection_method=LayerSelectionMethod.SOFT_GATING,
        depth_regularization=DepthRegularization.ELBO,
        uncertainty_method=UncertaintyMethod.PROBABILISTIC,
        n_epochs=60, learning_rate=0.01, early_stopping_rounds=15, validation_fraction=0.2,
        random_seed=seed, calculate_feature_importance=False,
        allow_retired_capacity_selection=True,
    )


def build_xgb(seed: int) -> XGBoostModel:
    return XGBoostModel(n_estimators=300, learning_rate=0.05, early_stopping_rounds=15, random_seed=seed, calculate_feature_importance=False)


def build_lgbm(seed: int) -> LightGBMModel:
    return LightGBMModel(n_estimators=300, learning_rate=0.05, early_stopping_rounds=15, random_seed=seed, calculate_feature_importance=False)


def build_nn(seed: int) -> PyTorchNeuralNetwork:
    return PyTorchNeuralNetwork(
        input_size=1, output_size=1, hidden_layers=2, hidden_size=64,
        uncertainty_method=UncertaintyMethod.PROBABILISTIC,
        n_epochs=60, learning_rate=0.01, early_stopping_rounds=15, validation_fraction=0.2,
        random_seed=seed, calculate_feature_importance=False,
    )


def run_seed(name: str, builder, seed: int) -> dict:
    x, y = _hetero_data(seed=42)  # fixed data; seeds affect only model init and dropout
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.3, random_state=0)
    t0 = time.perf_counter()
    m = builder(seed)
    m.fit(x_tr, y_tr)
    y_pred = np.asarray(m.predict(x_te)).ravel()
    mse = float(np.mean((y_te - y_pred) ** 2))
    try:
        y_std = np.asarray(m.predict_uncertainty(x_te)).ravel()
        nll = _gauss_nll(y_te, y_pred, y_std)
    except Exception:  # noqa: BLE001
        nll = None
    return {"model": name, "seed": seed, "mse": mse, "nll": nll, "seconds": time.perf_counter() - t0}


def main(seeds: tuple[int, ...] = SEEDS) -> None:
    rows: list[dict] = []
    builders = {
        "ProbReg(k=3)": build_probreg,
        "FlexNN(ELBO)": build_flex,
        "XGBoost": build_xgb,
        "LightGBM": build_lgbm,
        "PyTorchNN": build_nn,
    }
    for name, builder in builders.items():
        for seed in seeds:
            try:
                row = run_seed(name, builder, seed)
                rows.append(row)
                logger.info(f"{name:<15} seed={seed} mse={row['mse']:.4f} nll={row['nll']!s:<10} t={row['seconds']:.1f}s")
            except Exception as e:  # noqa: BLE001
                logger.exception(f"{name} seed={seed} crashed: {e}")
                rows.append({"model": name, "seed": seed, "error": str(e)})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "per_seed.csv", index=False)

    numeric = df.dropna(subset=["mse"])
    summary = numeric.groupby("model").agg(
        n_seeds=("seed", "count"),
        mse_mean=("mse", "mean"), mse_std=("mse", "std"),
        nll_mean=("nll", "mean"), nll_std=("nll", "std"),
    ).reset_index()
    summary.to_csv(OUT_DIR / "summary_mean_std.csv", index=False)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary.to_dict(orient="records"), indent=2, default=str))
    logger.info(f"\n{summary}")


if __name__ == "__main__":
    main()
