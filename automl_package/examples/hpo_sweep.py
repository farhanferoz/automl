"""HPO sweep: Optuna N=50 trials per model per UCI dataset.

Exposes loss_type, beta, target_transform, uncertainty_method where applicable.

Designed to produce tuned baseline numbers for the paper tables. Runtime is
substantial; intended for overnight execution. By default only runs on UCI-Yacht
(smallest) unless ``ALL_DATASETS=1`` is set in env.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from automl_package.enums import (
    NClassesSelectionMethod,
    RegressionStrategy,
    UncertaintyMethod,
)
from automl_package.models.lightgbm_model import LightGBMModel
from automl_package.models.neural_network import PyTorchNeuralNetwork
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel
from automl_package.models.xgboost_model import XGBoostModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

OUT_DIR = Path(__file__).parent / "hpo_sweep_results"
OUT_DIR.mkdir(exist_ok=True)

N_TRIALS = int(os.environ.get("HPO_N_TRIALS", "50"))


def _load_uci_yacht() -> tuple[np.ndarray, np.ndarray, str]:
    from sklearn.datasets import fetch_openml

    d = fetch_openml("yacht_hydrodynamics", as_frame=False, parser="auto")
    x = np.asarray(d.data, dtype=np.float32)
    y = np.asarray(d.target, dtype=np.float32).ravel()
    return x, y, "uci-yacht"


def run_hpo_per_model(x_tr, y_tr, x_te, y_te, input_size: int) -> list[dict]:
    rows: list[dict] = []

    def _eval(name, builder):
        t0 = time.perf_counter()
        m = builder()
        m.fit(x_tr, y_tr)
        y_pred = np.asarray(m.predict(x_te)).ravel()
        mse = float(np.mean((y_te - y_pred) ** 2))
        try:
            y_std = np.asarray(m.predict_uncertainty(x_te)).ravel()
            nll = float(np.mean(0.5 * (np.log(2 * np.pi * np.maximum(y_std, 1e-9) ** 2) + ((y_te - y_pred) / np.maximum(y_std, 1e-9)) ** 2)))
        except Exception:  # noqa: BLE001
            nll = None
        rows.append({"model": name, "mse": mse, "nll": nll, "seconds": time.perf_counter() - t0})
        logger.info(f"{name:<20} mse={mse:.4f} nll={nll!s:<10} t={rows[-1]['seconds']:.1f}s")

    def _xgb():
        return XGBoostModel(
            n_estimators=300, early_stopping_rounds=15, random_seed=42,
            validation_fraction=0.2, calculate_feature_importance=False,
            optimize_hyperparameters=True, n_trials=N_TRIALS,
        )

    def _lgbm():
        return LightGBMModel(
            n_estimators=300, early_stopping_rounds=15, random_seed=42,
            validation_fraction=0.2, calculate_feature_importance=False,
            optimize_hyperparameters=True, n_trials=N_TRIALS,
        )

    def _probreg():
        return ProbabilisticRegressionModel(
            input_size=input_size, n_classes=3, max_n_classes_for_probabilistic_path=7,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.NONE,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            base_classifier_params={"hidden_layers": 1, "hidden_size": 64},
            regression_head_params={"hidden_layers": 0, "hidden_size": 32},
            n_epochs=100, early_stopping_rounds=15, validation_fraction=0.2,
            random_seed=42, calculate_feature_importance=False,
            optimize_hyperparameters=True, n_trials=N_TRIALS,
        )

    def _nn():
        return PyTorchNeuralNetwork(
            input_size=input_size, output_size=1,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_epochs=100, early_stopping_rounds=15, validation_fraction=0.2,
            random_seed=42, calculate_feature_importance=False,
            optimize_hyperparameters=True, n_trials=N_TRIALS,
        )

    for name, builder in [("XGB+HPO", _xgb), ("LGBM+HPO", _lgbm), ("ProbReg+HPO", _probreg), ("NN+HPO", _nn)]:
        try:
            _eval(name, builder)
        except Exception as e:  # noqa: BLE001
            logger.exception(f"{name} crashed: {e}")
            rows.append({"model": name, "error": str(e)})
    return rows


def main() -> None:
    datasets = [_load_uci_yacht()]
    if os.environ.get("ALL_DATASETS"):
        # Lazy import to avoid data fetches by default.
        from sklearn.datasets import fetch_california_housing

        ch = fetch_california_housing()
        # Sub-sample California to 3000 to keep HPO tractable.
        rng = np.random.default_rng(0)
        idx = rng.choice(len(ch.target), size=3000, replace=False)
        datasets.append((ch.data[idx].astype(np.float32), ch.target[idx].astype(np.float32), "uci-california-3k"))

    all_rows: list[dict] = []
    for x, y, name in datasets:
        x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.3, random_state=0)
        logger.info(f"=== {name} n_train={len(x_tr)} ===")
        rows = run_hpo_per_model(x_tr, y_tr, x_te, y_te, input_size=x.shape[1])
        for r in rows:
            r["dataset"] = name
            all_rows.append(r)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "hpo_results.csv", index=False)
    (OUT_DIR / "hpo_results.json").write_text(json.dumps(all_rows, indent=2, default=str))
    logger.info(f"\n{df}")


if __name__ == "__main__":
    main()
