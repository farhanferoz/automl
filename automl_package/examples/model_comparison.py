"""Model Comparison: runs all supported models on toy datasets and compares MSE, NLL, and calibration.

Datasets:
  1. Heteroscedastic sine — noise grows with |x|, ideal for probabilistic regression
  2. Multimodal (bimodal) — y = x ± 1.5, tests classification bottleneck
  3. Simple linear — y = 2x + 1 + noise, sanity check

Models:
  - JAXLinearRegression (baseline)
  - XGBoost, LightGBM (tree baselines)
  - PyTorchNeuralNetwork (constant variance baseline)
  - ClassifierRegressionModel (spline mapper + NN mapper)
  - ProbabilisticRegressionModel (SEPARATE_HEADS, fixed k sweep)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from automl_package.enums import (
    MapperType,
    NClassesSelectionMethod,
    RegressionStrategy,
    UncertaintyMethod,
)
from automl_package.models.classifier_regression import ClassifierRegressionModel
from automl_package.models.lightgbm_model import LightGBMModel
from automl_package.models.linear_regression import LinearRegressionModel
from automl_package.models.neural_network import PyTorchNeuralNetwork
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel
from automl_package.models.xgboost_model import XGBoostModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

OUTPUT_DIR = Path(__file__).parent / "model_comparison_results"


# ---------------------------------------------------------------------------
# Dataset generators
# ---------------------------------------------------------------------------

def generate_heteroscedastic_data(n_samples: int = 1000, random_seed: int = 42) -> tuple:
    """Generates data where noise variance increases with |x|."""
    np.random.seed(random_seed)
    x = np.random.uniform(-5, 5, n_samples).reshape(-1, 1)
    y_true = np.sin(x) * 2 + 0.5 * x
    noise_std = 0.1 + 0.4 * np.abs(x)
    noise = np.random.normal(0, noise_std)
    y = y_true + noise
    return x, y.ravel(), y_true.ravel(), noise_std.ravel()


def generate_multimodal_data(n_samples: int = 1000, random_seed: int = 42) -> tuple:
    """Generates bimodal data: y = x ± 1.5."""
    np.random.seed(random_seed)
    x = np.random.uniform(-3, 3, n_samples).reshape(-1, 1)
    sign = np.random.choice([-1, 1], size=n_samples).reshape(-1, 1)
    y = x + sign * 1.5 + np.random.normal(0, 0.1, (n_samples, 1))
    return x, y.ravel()


def generate_simple_linear(n_samples: int = 500, random_seed: int = 42) -> tuple:
    """Simple y = 2x + 1 + noise."""
    np.random.seed(random_seed)
    x = np.random.uniform(-5, 5, n_samples).reshape(-1, 1)
    y = (2 * x + 1 + np.random.normal(0, 0.3, (n_samples, 1))).ravel()
    return x, y


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean((y_true - y_pred) ** 2))


def compute_gaussian_nll(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray) -> float:
    """Mean Gaussian NLL."""
    variance = np.maximum(y_std ** 2, 1e-8)
    nll = 0.5 * (np.log(2 * np.pi * variance) + ((y_true - y_pred) ** 2) / variance)
    return float(np.mean(nll))


def compute_calibration(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray) -> float:
    """Fraction of test points within ±1 sigma (should be ~68%)."""
    within = np.abs(y_true - y_pred) <= y_std
    return float(np.mean(within))


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_linear_regression() -> LinearRegressionModel:
    """JAX linear regression baseline."""
    return LinearRegressionModel(learning_rate=0.01, n_iterations=1000, early_stopping_rounds=50, validation_fraction=0.2)


def build_xgboost() -> XGBoostModel:
    """XGBoost baseline."""
    return XGBoostModel(n_estimators=200, early_stopping_rounds=15, validation_fraction=0.2, random_seed=42)


def build_lightgbm() -> LightGBMModel:
    """LightGBM baseline."""
    return LightGBMModel(n_estimators=200, early_stopping_rounds=15, validation_fraction=0.2, random_seed=42)


def build_nn_constant(input_size: int = 1) -> PyTorchNeuralNetwork:
    """Constant-variance neural network baseline."""
    return PyTorchNeuralNetwork(
        input_size=input_size, output_size=1, hidden_layers=2, hidden_size=64,
        learning_rate=0.01, n_epochs=100, early_stopping_rounds=15,
        validation_fraction=0.2, uncertainty_method=UncertaintyMethod.CONSTANT, random_seed=42,
    )


def build_classifier_regression_spline(input_size: int = 1, n_classes: int = 7) -> ClassifierRegressionModel:
    """ClassifierRegression with spline mapper (two-stage)."""
    return ClassifierRegressionModel(
        base_classifier_class=PyTorchNeuralNetwork, n_classes=n_classes, mapper_type=MapperType.SPLINE,
        base_classifier_params={"input_size": input_size, "hidden_layers": 2, "hidden_size": 64, "learning_rate": 0.01, "n_epochs": 100},
        early_stopping_rounds=15, uncertainty_method=UncertaintyMethod.BINNED_RESIDUAL_STD,
        validation_fraction=0.2, random_seed=42,
    )


def build_classifier_regression_lookup(input_size: int = 1, n_classes: int = 7) -> ClassifierRegressionModel:
    """ClassifierRegression with lookup median mapper (percentile bins + median)."""
    return ClassifierRegressionModel(
        base_classifier_class=PyTorchNeuralNetwork, n_classes=n_classes, mapper_type=MapperType.LOOKUP_MEDIAN,
        base_classifier_params={"input_size": input_size, "hidden_layers": 2, "hidden_size": 64, "learning_rate": 0.01, "n_epochs": 100},
        early_stopping_rounds=15, uncertainty_method=UncertaintyMethod.BINNED_RESIDUAL_STD,
        validation_fraction=0.2, random_seed=42,
    )


def build_classifier_regression_nn(input_size: int = 1, n_classes: int = 7) -> ClassifierRegressionModel:
    """ClassifierRegression with NN mapper (two-stage: classifier frozen, then NN mapper trained on probabilities)."""
    return ClassifierRegressionModel(
        base_classifier_class=PyTorchNeuralNetwork, n_classes=n_classes, mapper_type=MapperType.NN_SEPARATE_HEADS,
        base_classifier_params={"input_size": input_size, "hidden_layers": 2, "hidden_size": 64, "learning_rate": 0.01, "n_epochs": 100},
        nn_mapper_params={
            "epochs": 100, "learning_rate": 0.01,
            "regression_head_params": {"hidden_layers": 1, "hidden_size": 32},
        },
        regression_strategy=RegressionStrategy.SEPARATE_HEADS,
        early_stopping_rounds=15, uncertainty_method=UncertaintyMethod.BINNED_RESIDUAL_STD,
        validation_fraction=0.2, random_seed=42,
    )


def build_prob_regression(input_size: int = 1, k: int = 5) -> ProbabilisticRegressionModel:
    """ProbabilisticRegression with SEPARATE_HEADS, fixed k."""
    return ProbabilisticRegressionModel(
        input_size=input_size, n_classes=k, uncertainty_method=UncertaintyMethod.PROBABILISTIC,
        n_classes_selection_method=NClassesSelectionMethod.NONE, regression_strategy=RegressionStrategy.SEPARATE_HEADS,
        base_classifier_params={"hidden_layers": 1, "hidden_size": 64},
        regression_head_params={"hidden_layers": 0, "hidden_size": 32},
        n_epochs=100, learning_rate=0.01, early_stopping_rounds=15, validation_fraction=0.2, random_seed=42,
    )


# ---------------------------------------------------------------------------
# Run comparison
# ---------------------------------------------------------------------------

def evaluate_model(model: object, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, has_uncertainty: bool = True) -> dict:
    """Train a model and compute metrics."""
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse = compute_mse(y_test, y_pred)

    result = {"MSE": mse}
    if has_uncertainty:
        y_std = model.predict_uncertainty(x_test)
        result["NLL"] = compute_gaussian_nll(y_test, y_pred, y_std)
        result["Calibration_1sigma"] = compute_calibration(y_test, y_pred, y_std)
    return result


def run_dataset_comparison(dataset_name: str, x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Run all models on a single dataset and return results DataFrame."""
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    input_size = x.shape[1]

    rows = []

    # 1. Linear regression (JAX)
    logger.info(f"[{dataset_name}] Training: LinearRegression")
    result = evaluate_model(build_linear_regression(), x_train, y_train, x_test, y_test)
    rows.append({"Model": "LinearReg", **result})

    # 2. Tree-based models
    logger.info(f"[{dataset_name}] Training: XGBoost")
    result = evaluate_model(build_xgboost(), x_train, y_train, x_test, y_test)
    rows.append({"Model": "XGBoost", **result})

    logger.info(f"[{dataset_name}] Training: LightGBM")
    result = evaluate_model(build_lightgbm(), x_train, y_train, x_test, y_test)
    rows.append({"Model": "LightGBM", **result})

    # 3. NN baseline (constant variance)
    logger.info(f"[{dataset_name}] Training: NN (constant variance)")
    result = evaluate_model(build_nn_constant(input_size), x_train, y_train, x_test, y_test)
    rows.append({"Model": "NN_Constant", **result})

    # 4. ClassifierRegression (lookup median — k sweep)
    for k in [3, 5, 7, 10]:
        logger.info(f"[{dataset_name}] Training: ClassifierRegression (lookup_median, k={k})")
        result = evaluate_model(build_classifier_regression_lookup(input_size, n_classes=k), x_train, y_train, x_test, y_test)
        rows.append({"Model": f"ClassReg_median_k{k}", **result})

    # 5. ClassifierRegression (lookup median with XGBoost classifier)
    logger.info(f"[{dataset_name}] Training: ClassifierRegression (XGB+median, k=7)")
    xgb_classreg = ClassifierRegressionModel(
        base_classifier_class=XGBoostModel, n_classes=7, mapper_type=MapperType.LOOKUP_MEDIAN,
        base_classifier_params={"n_estimators": 200},
        early_stopping_rounds=15, uncertainty_method=UncertaintyMethod.BINNED_RESIDUAL_STD,
        validation_fraction=0.2, random_seed=42,
    )
    result = evaluate_model(xgb_classreg, x_train, y_train, x_test, y_test)
    rows.append({"Model": "ClassReg_XGB_median_k7", **result})

    # 6. ClassifierRegression (spline)
    logger.info(f"[{dataset_name}] Training: ClassifierRegression (spline, k=7)")
    result = evaluate_model(build_classifier_regression_spline(input_size, n_classes=7), x_train, y_train, x_test, y_test)
    rows.append({"Model": "ClassReg_spline_k7", **result})

    # 7. ClassifierRegression (NN mapper — params fix applied)
    logger.info(f"[{dataset_name}] Training: ClassifierRegression (NN mapper, k=7)")
    result = evaluate_model(build_classifier_regression_nn(input_size, n_classes=7), x_train, y_train, x_test, y_test)
    rows.append({"Model": "ClassReg_NN_k7", **result})

    # 8. ProbabilisticRegression (k sweep)
    for k in [3, 5, 7, 10]:
        logger.info(f"[{dataset_name}] Training: ProbabilisticRegression (k={k})")
        result = evaluate_model(build_prob_regression(input_size, k=k), x_train, y_train, x_test, y_test)
        rows.append({"Model": f"ProbReg_k{k}", **result})

    return pd.DataFrame(rows).set_index("Model")


def main() -> None:
    """Run model comparison on all datasets."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {
        "heteroscedastic": generate_heteroscedastic_data(),
        "multimodal": generate_multimodal_data(),
        "linear": generate_simple_linear(),
    }

    all_results = {}
    for name, data in datasets.items():
        x, y = data[0], data[1]
        logger.info(f"\n{'='*60}\n  Dataset: {name} (n={len(x)})\n{'='*60}")
        df = run_dataset_comparison(name, x, y)
        all_results[name] = df

        print(f"\n--- {name} ---")
        print(df.to_string(float_format="%.4f"))
        df.to_csv(OUTPUT_DIR / f"{name}_results.csv")

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for name, df in all_results.items():
        print(f"\n[{name}]")
        print(df.to_string(float_format="%.4f"))


if __name__ == "__main__":
    main()
