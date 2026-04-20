"""I4: Noise-robustness benchmark.

Validates the original motivation of the classification-bottleneck approach:
on noisy regression, discretising the target should outperform plain regression
because the bottleneck acts as a quantising regulariser. This benchmark sweeps
ClassReg n=2..20 across noise levels sigma in {0.01, 0.1, 0.5, 1.0} and checks
that (a) increasing sigma favours larger n, and (b) ProbReg with dynamic-k
auto-picks a sensible mean-k at each noise level.

Baselines: XGBoost, LightGBM, CatBoost (all with BinnedResidualStd uncertainty),
and a plain PyTorch NN with CONSTANT uncertainty.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from automl_package.enums import (
    MapperType,
    NClassesRegularization,
    NClassesSelectionMethod,
    RegressionStrategy,
    UncertaintyMethod,
)
from automl_package.models.catboost_model import CatBoostModel
from automl_package.models.classifier_regression import ClassifierRegressionModel
from automl_package.models.lightgbm_model import LightGBMModel
from automl_package.models.neural_network import PyTorchNeuralNetwork
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel
from automl_package.models.selection_strategies.base_selection_strategy import DIRECT_REGRESSION_K_SENTINEL
from automl_package.models.xgboost_model import XGBoostModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

OUT_DIR = Path(__file__).parent / "noise_robustness_results"
OUT_DIR.mkdir(exist_ok=True)

N_SAMPLES = 600
NOISE_LEVELS = (0.05, 0.3, 1.0)
CLASSREG_K_VALUES = (2, 3, 5, 10, 15)


def make_data(noise_std: float, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Smooth noisy target: y = sin(x) + 0.3 x + Normal(0, noise_std)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-5.0, 5.0, N_SAMPLES).reshape(-1, 1).astype(np.float32)
    y_true = np.sin(x.ravel()) * 2 + 0.3 * x.ravel()
    y = (y_true + rng.normal(0.0, noise_std, N_SAMPLES)).astype(np.float32)
    return x, y.astype(np.float32)


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true.ravel() - y_pred.ravel()) ** 2))


def _fit_predict_mse(model, x_tr, y_tr, x_te, y_te) -> tuple[float, float]:
    t0 = time.perf_counter()
    model.fit(x_tr, y_tr)
    y_pred = model.predict(x_te)
    return _mse(y_te, y_pred), time.perf_counter() - t0


def build_classreg(k: int, input_size: int) -> ClassifierRegressionModel:
    return ClassifierRegressionModel(
        base_classifier_class=PyTorchNeuralNetwork, n_classes=k,
        mapper_type=MapperType.LOOKUP_MEDIAN,
        base_classifier_params={"input_size": input_size, "hidden_layers": 2, "hidden_size": 64, "learning_rate": 0.01, "n_epochs": 80},
        early_stopping_rounds=15, uncertainty_method=UncertaintyMethod.BINNED_RESIDUAL_STD,
        validation_fraction=0.2, random_seed=42, calculate_feature_importance=False,
    )


def build_probreg_dyn_k(max_k: int, input_size: int) -> ProbabilisticRegressionModel:
    return ProbabilisticRegressionModel(
        input_size=input_size, n_classes=3, max_n_classes_for_probabilistic_path=max_k,
        uncertainty_method=UncertaintyMethod.PROBABILISTIC,
        n_classes_selection_method=NClassesSelectionMethod.SOFT_GATING,
        n_classes_regularization=NClassesRegularization.ELBO,
        regression_strategy=RegressionStrategy.SEPARATE_HEADS,
        base_classifier_params={"hidden_layers": 1, "hidden_size": 64},
        regression_head_params={"hidden_layers": 0, "hidden_size": 32},
        n_epochs=80, learning_rate=0.01, early_stopping_rounds=15,
        validation_fraction=0.2, random_seed=42, calculate_feature_importance=False,
    )


def build_nn(input_size: int) -> PyTorchNeuralNetwork:
    return PyTorchNeuralNetwork(
        input_size=input_size, output_size=1,
        uncertainty_method=UncertaintyMethod.CONSTANT,
        hidden_layers=2, hidden_size=64, learning_rate=0.01,
        n_epochs=80, early_stopping_rounds=15, validation_fraction=0.2,
        random_seed=42, calculate_feature_importance=False,
    )


def build_xgb() -> XGBoostModel:
    return XGBoostModel(n_estimators=300, learning_rate=0.05, early_stopping_rounds=15, random_seed=42, calculate_feature_importance=False)


def build_lgbm() -> LightGBMModel:
    return LightGBMModel(n_estimators=300, learning_rate=0.05, early_stopping_rounds=15, random_seed=42, calculate_feature_importance=False)


def build_catboost() -> CatBoostModel:
    return CatBoostModel(iterations=300, learning_rate=0.05, early_stopping_rounds=15, verbose=False, random_seed=42, calculate_feature_importance=False)


def dyn_k_mean(model: ProbabilisticRegressionModel, x_te: np.ndarray) -> float:
    """Extract mean selected k from a fitted dynamic-k ProbReg."""
    x_t = torch.tensor(x_te, dtype=torch.float32).to(model.device)
    model.model.eval()
    with torch.no_grad():
        _, _, k_actual, _, _ = model.model(x_t)
    k_valid = k_actual[k_actual < DIRECT_REGRESSION_K_SENTINEL]
    return float(k_valid.float().mean().item()) if k_valid.numel() > 0 else float("nan")


def main() -> None:
    rows: list[dict] = []
    dyn_k_rows: list[dict] = []

    for sigma in NOISE_LEVELS:
        x, y = make_data(sigma)
        x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.3, random_state=0)
        input_size = x.shape[1]

        # Baselines (CatBoost dropped: too slow for this sweep)
        for name, ctor in [
            ("XGBoost", build_xgb),
            ("LightGBM", build_lgbm),
            ("NN", lambda: build_nn(input_size)),
        ]:
            try:
                mse, dt = _fit_predict_mse(ctor(), x_tr, y_tr, x_te, y_te)
                rows.append({"sigma": sigma, "model": name, "k": None, "mse": mse, "seconds": dt})
                logger.info(f"sigma={sigma}  {name:<10}  MSE={mse:.5f}  t={dt:.1f}s")
            except Exception as e:  # noqa: BLE001
                logger.exception(f"{name} crashed at sigma={sigma}: {e}")
                rows.append({"sigma": sigma, "model": name, "k": None, "mse": float("nan"), "seconds": float("nan")})

        # ClassReg k sweep
        for k in CLASSREG_K_VALUES:
            try:
                mse, dt = _fit_predict_mse(build_classreg(k, input_size), x_tr, y_tr, x_te, y_te)
                rows.append({"sigma": sigma, "model": "ClassReg", "k": k, "mse": mse, "seconds": dt})
                logger.info(f"sigma={sigma}  ClassReg k={k:<3}  MSE={mse:.5f}  t={dt:.1f}s")
            except Exception as e:  # noqa: BLE001
                logger.exception(f"ClassReg k={k} crashed at sigma={sigma}: {e}")
                rows.append({"sigma": sigma, "model": "ClassReg", "k": k, "mse": float("nan"), "seconds": float("nan")})

        # ProbReg dynamic-k: both with ELBO (regularised) and without (free).
        for max_k, reg_name, reg in (
            (10, "elbo", NClassesRegularization.ELBO),
            (10, "none", NClassesRegularization.NONE),
        ):
            try:
                model = ProbabilisticRegressionModel(
                    input_size=input_size, n_classes=3, max_n_classes_for_probabilistic_path=max_k,
                    uncertainty_method=UncertaintyMethod.PROBABILISTIC,
                    n_classes_selection_method=NClassesSelectionMethod.SOFT_GATING,
                    n_classes_regularization=reg,
                    regression_strategy=RegressionStrategy.SEPARATE_HEADS,
                    base_classifier_params={"hidden_layers": 1, "hidden_size": 64},
                    regression_head_params={"hidden_layers": 0, "hidden_size": 32},
                    n_epochs=80, learning_rate=0.01, early_stopping_rounds=15,
                    validation_fraction=0.2, random_seed=42, calculate_feature_importance=False,
                )
                mse, dt = _fit_predict_mse(model, x_tr, y_tr, x_te, y_te)
                mean_k = dyn_k_mean(model, x_te)
                label = f"ProbReg_dyn_k(max={max_k},{reg_name})"
                rows.append({"sigma": sigma, "model": label, "k": mean_k, "mse": mse, "seconds": dt})
                dyn_k_rows.append({"sigma": sigma, "max_k": max_k, "reg": reg_name, "mean_k": mean_k, "mse": mse})
                logger.info(f"sigma={sigma}  {label}  mean_k={mean_k:.2f}  MSE={mse:.5f}  t={dt:.1f}s")
            except Exception as e:  # noqa: BLE001
                logger.exception(f"ProbReg dyn_k max={max_k} reg={reg_name} crashed at sigma={sigma}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "results.csv", index=False)

    dyn = pd.DataFrame(dyn_k_rows)
    dyn.to_csv(OUT_DIR / "dynamic_k.csv", index=False)

    # ClassReg k vs sigma summary
    cr = df[df["model"] == "ClassReg"].pivot_table(index="sigma", columns="k", values="mse")
    cr.to_csv(OUT_DIR / "classreg_mse_by_k.csv")

    # Which k minimises MSE at each sigma?
    best_k = cr.idxmin(axis=1)
    best_k_summary = pd.DataFrame({"sigma": best_k.index, "best_k": best_k.values})
    best_k_summary.to_csv(OUT_DIR / "best_k_by_sigma.csv", index=False)

    summary = {
        "best_k_by_sigma": best_k_summary.to_dict(orient="records"),
        "dyn_k_by_sigma": dyn.to_dict(orient="records") if not dyn.empty else [],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    logger.info("== Summary ==")
    logger.info(f"ClassReg best k vs sigma:\n{best_k_summary}")
    if not dyn.empty:
        logger.info(f"ProbReg dyn-k mean-k vs sigma:\n{dyn}")


if __name__ == "__main__":
    main()
