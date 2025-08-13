"""Debugging script for AutoML package."""

import json
import logging

import numpy as np
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from automl_package.automl import AutoML
from automl_package.enums import ModelName, TaskType

logger = logging.getLogger("automl_package")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

logger.info("===== Starting AutoML Debugging Session =====")


# --- Regression Example (Simplified for Debugging) ---
logger.info("\n--- Running Regression Example with Reduced Settings ---")
X_reg, y_reg = make_regression(n_samples=100, n_features=5, noise=5.0, random_state=42)
y_reg = y_reg + abs(y_reg.min()) + 1  # Ensure y is positive for certain percentile calcs

# Split data for initial training (for AutoML's CV) and final test evaluation
X_train_initial_reg, X_test_full_reg, y_train_initial_reg, y_test_full_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Instantiate scalers
X_scaler_reg = StandardScaler()
y_scaler_reg = StandardScaler()

# Configure AutoML for quick debugging
automl_reg = AutoML(task_type=TaskType.REGRESSION, metric="rmse", n_trials=2, n_splits=2, random_state=42, feature_scaler=X_scaler_reg, target_scaler=y_scaler_reg, use_wandb=True)

# --- Select ONE model to debug at a time by uncommenting it ---
models_for_reg = [
    # ModelName.JAX_LINEAR_REGRESSION,
    # ModelName.PYTORCH_NEURAL_NETWORK,
    ModelName.FLEXIBLE_NEURAL_NETWORK,
    # ModelName.XGBOOST,
    # ModelName.LIGHTGBM,
    # ModelName.CATBOOST,
    # ModelName.CLASSIFIER_REGRESSION,
    # ModelName.PROBABILISTIC_REGRESSION,
]

automl_reg = AutoML(task_type=TaskType.REGRESSION, metric="rmse", n_trials=1, n_splits=2, random_state=42, feature_scaler=X_scaler_reg, target_scaler=y_scaler_reg, use_wandb=False)
automl_reg.train(X_train_initial_reg, y_train_initial_reg, models_to_consider=models_for_reg)

if automl_reg.best_model_name:
    logger.info(f"\n--- Making Predictions with Best Regression Model ({automl_reg.best_model_name.value}) ---")
    y_pred_test = automl_reg.predict(X_test_full_reg)
    test_rmse = np.sqrt(mean_squared_error(y_test_full_reg, y_pred_test))
    logger.info(f"Best model test RMSE (original scale): {test_rmse:.4f}")
    logger.info(f"Sample predictions (first 5, original scale): {y_pred_test[:5].round(2)}")

    logger.info(f"\n--- Predicting Uncertainty with Best Regression Model ({automl_reg.best_model_name.value}) ---")
    try:
        uncertainty_values = automl_reg.predict_uncertainty(X_test_full_reg)
        logger.info(f"Mean uncertainty estimate (original scale): {np.mean(uncertainty_values):.4f}")
        logger.info(f"Uncertainty estimates (first 5, original scale): {uncertainty_values[:5].round(2)}")
    except ValueError as e:
        logger.info(f"Could not get uncertainty estimates for model {automl_reg.best_model_name.value}: {e}")
    except NotImplementedError as e:
        logger.info(f"Uncertainty prediction not implemented for {automl_reg.best_model_name.value}: {e}")

    logger.info(f"\n--- Getting Feature Importance for Best Regression Model ({automl_reg.best_model_name.value}) ---")
    # Feature names are generic for synthetic data
    feature_names_reg = [f"feature_{i}" for i in range(X_reg.shape[1])]
    feature_importance_summary = automl_reg.get_feature_importance(X_test_full_reg, feature_names=feature_names_reg)
    if "error" not in feature_importance_summary:
        # Convert float32 values to standard floats for JSON serialization
        serializable_importance = {k: float(v) for k, v in feature_importance_summary.items()}
        logger.info(f"Top 5 Features by SHAP Importance:\n{json.dumps(dict(list(serializable_importance.items())[:5]), indent=2)}")

    logger.info("\n--- Retraining with Selected Features (Regression) ---")
    retrained_results_reg = automl_reg.retrain_with_selected_features(
        X_full_train=X_reg, y_full_train=y_reg, X_full_test=X_test_full_reg, y_full_test=y_test_full_reg, feature_names=feature_names_reg, shap_threshold=0.95
    )
    logger.info(f"Retrained model test RMSE (with selected features, original scale): {retrained_results_reg['retrained_metric_value']:.4f}")
    logger.info(f"Selected features: {retrained_results_reg['selected_feature_names']}")

logger.info("\n===== AutoML Debugging Session Complete =====")
