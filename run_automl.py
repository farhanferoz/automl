import numpy as np
import logging
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler

from automl_package.automl import AutoML
from automl_package.enums import TaskType, ModelName
from automl_package.logger import logger

if __name__ == "__main__":

    # Configure the root logger for example output
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Ensure you have the required libraries installed:
    # pip install numpy scikit-learn jax jaxlib optuna torch xgboost lightgbm catboost flax optax scipy shap

    # --- Regression Example with JAXProbabilisticRegressionModel and Uncertainty ---
    logger.info("--- Running Regression Example with Uncertainty ---")

    X_reg, y_reg = make_regression(n_samples=500, n_features=10, noise=10.0, random_state=42)
    y_reg = y_reg + abs(y_reg.min()) + 1  # Ensure y is positive for percentile calc robustness

    # Split data once for initial training and final retraining
    X_full, X_test_full, y_full, y_test_full = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    # For initial AutoML training (cross-validation on X_full, y_full)
    X_train_initial, y_train_initial = X_full, y_full  # Use full training data for main train() call

    # Generate feature names for better SHAP output
    feature_names_reg = [f"feature_{i}" for i in range(X_reg.shape[1])]

    # Instantiate scalers
    X_scaler_reg = StandardScaler()
    y_scaler_reg = StandardScaler()

    automl_reg = AutoML(
        task_type=TaskType.REGRESSION, metric="rmse", n_trials=3, n_splits=2, random_state=42, feature_scaler=X_scaler_reg, target_scaler=y_scaler_reg
    )  # Pass scalers

    # Specify which models to consider for this run
    models_for_reg = [
        ModelName.JAX_LINEAR_REGRESSION,
        ModelName.PYTORCH_NEURAL_NETWORK,
        ModelName.FLEXIBLE_NEURAL_NETWORK,  # <--- FLEXIBLE_NEURAL_NETWORK INCLUDED
        ModelName.XGBOOST,
        ModelName.LIGHTGBM,
        ModelName.CATBOOST,
        ModelName.CLASSIFIER_REGRESSION,  # Only for regression where classification informs regression
        ModelName.PROBABILISTIC_REGRESSION,
        ModelName.JAX_PROBABILISTIC_REGRESSION,
    ]

    automl_reg.train(X_train_initial, y_train_initial, models_to_consider=models_for_reg)

    best_model_info_reg = automl_reg.get_best_model_info()
    logger.info("\nRegression Best Model Info:")
    for key, value in best_model_info_reg.items():
        if key == "instance":
            logger.info(f"  {key}: <{value.__class__.__name__} object>")
        else:
            logger.info(f"  {key}: {value}")

    if automl_reg.best_model_name:
        # Evaluate initial best model on the full test set (predictions are now denormalized)
        y_pred_initial_test = automl_reg.predict(X_test_full)
        initial_test_rmse = np.sqrt(mean_squared_error(y_test_full, y_pred_initial_test))
        logger.info(f"Initial best model ({automl_reg.best_model_name.value}) test RMSE (original scale): {initial_test_rmse:.4f}")

        # Get uncertainty estimates for the initial best model (now denormalized)
        try:
            uncertainty_reg = automl_reg.predict_uncertainty(X_test_full)
            logger.info(f"Mean uncertainty estimate of initial best regression model (original scale): {np.mean(uncertainty_reg):.4f}")
            logger.info(f"Uncertainty estimates (first 5, original scale): {uncertainty_reg[:5].round(2)}")
        except ValueError as e:
            logger.error(f"Could not get uncertainty estimates for initial model: {e}")
        except RuntimeError as e:
            logger.error(f"Error getting uncertainty estimates for initial model: {e}")

        # Perform feature selection and retrain
        retrained_results_reg = automl_reg.retrain_with_selected_features(
            X_full_train=X_full,  # Use the full training data for retraining (original scale)
            y_full_train=y_full,  # (original scale)
            X_full_test=X_test_full,  # (original scale)
            y_full_test=y_test_full,  # (original scale)
            feature_names=feature_names_reg,
            shap_threshold=0.95,
        )
        logger.info(f"Selected features for regression: {retrained_results_reg['selected_feature_names']}")
        logger.info(f"Retrained model test RMSE (with selected features, original scale): {retrained_results_reg['retrained_metric_value']:.4f}")
        # Optionally, get uncertainty for the retrained model
        try:
            uncertainty_retrained_reg = retrained_results_reg["retrained_model_instance"].predict_uncertainty(retrained_results_reg["X_test_filtered"])
            logger.info(f"Mean uncertainty estimate of retrained regression model (original scale): {np.mean(uncertainty_retrained_reg):.4f}")
        except Exception as e:
            logger.error(f"Error getting uncertainty for retrained regression model: {e}")

        # Export the retrained model
        model_export_path_reg = "best_automl_reg_model.pkl"
        automl_reg.export_model(retrained_results_reg["retrained_model_instance"], model_export_path_reg)

        # Demonstrate loading the model and making a prediction
        try:
            loaded_model_reg = AutoML.load_model(model_export_path_reg)
            # Make a prediction with the loaded model. Pass original scale input, AutoML will handle scaling.
            sample_input_reg_original = X_test_full[0:1]  # Get first sample from original test data
            # The loaded model's predict method expects data in the *scaled* format it was trained on.
            # The automl_reg instance itself holds the fitted scalers.
            if automl_reg._fitted_feature_scaler:
                sample_input_reg_scaled = automl_reg._fitted_feature_scaler.transform(sample_input_reg_original)
            else:
                sample_input_reg_scaled = sample_input_reg_original

            loaded_prediction_reg_scaled = loaded_model_reg.predict(sample_input_reg_scaled)

            # Then denormalize the prediction if a target scaler was used during training.
            if automl_reg._fitted_target_scaler:
                loaded_prediction_reg_original_scale = automl_reg._fitted_target_scaler.inverse_transform(loaded_prediction_reg_scaled.reshape(-1, 1)).flatten()
            else:
                loaded_prediction_reg_original_scale = loaded_prediction_reg_scaled

            logger.info(
                f"Prediction from loaded regression model for sample {sample_input_reg_original[0].round(2)} (original scale): {loaded_prediction_reg_original_scale[0]:.4f}"
            )
        except Exception as e:
            logger.error(f"Error demonstrating model load and predict for regression: {e}")

    # Save the leaderboard after all operations
    automl_reg.save_leaderboard("regression_leaderboard.json")

    # --- Classification Example ---
    logger.info("\n--- Running Classification Example ---")
    X_clf, y_clf = make_classification(n_samples=500, n_features=10, n_classes=2, random_state=42)

    # Split data once for initial training and final retraining
    X_full_clf, X_test_full_clf, y_full_clf, y_test_full_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
    X_train_initial_clf, y_train_initial_clf = X_full_clf, y_full_clf  # Use full training data for main train() call

    feature_names_clf = [f"feature_{i}" for i in range(X_clf.shape[1])]

    # Instantiate scaler for classification features
    X_scaler_clf = StandardScaler()

    automl_clf = AutoML(task_type=TaskType.CLASSIFICATION, metric="accuracy", n_trials=3, n_splits=2, random_state=42, feature_scaler=X_scaler_clf)  # Pass feature scaler

    # Specify which models to consider for classification
    models_for_clf = [
        ModelName.SKLEARN_LOGISTIC_REGRESSION,
        ModelName.PYTORCH_NEURAL_NETWORK,
        ModelName.FLEXIBLE_NEURAL_NETWORK,  # <--- FLEXIBLE_NEURAL_NETWORK INCLUDED
        ModelName.XGBOOST,
        ModelName.LIGHTGBM,
        ModelName.CATBOOST,
    ]

    automl_clf.train(X_train_initial_clf, y_train_initial_clf, models_to_consider=models_for_clf)

    best_model_info_clf = automl_clf.get_best_model_info()
    logger.info("\nClassification Best Model Info:")
    for key, value in best_model_info_clf.items():
        if key == "instance":
            logger.info(f"  {key}: <{value.__class__.__name__} object>")
        else:
            logger.info(f"  {key}: {value}")

    if automl_clf.best_model_name:
        # Evaluate initial best model on the full test set (predictions are not denormalized for classification)
        y_pred_initial_test_clf = automl_clf.predict(X_test_full_clf)
        initial_test_accuracy_clf = accuracy_score(y_test_full_clf, y_pred_initial_test_clf)
        logger.info(f"Initial best model ({automl_clf.best_model_name.value}) test accuracy: {initial_test_accuracy_clf:.4f}")

        # Attempt to get uncertainty for initial classification model (should raise ValueError)
        try:
            uncertainty_clf = automl_clf.predict_uncertainty(X_test_full_clf)
            logger.info(f"Mean uncertainty estimate of initial classification model: {np.mean(uncertainty_clf):.4f}")
            logger.info(f"Uncertainty estimates (first 5): {uncertainty_clf[:5].round(2)}")
        except ValueError as e:
            logger.error(f"As expected, cannot get uncertainty for initial classification model: {e}")
        except RuntimeError as e:
            logger.error(f"Error getting uncertainty estimates for initial classification model: {e}")

        # Perform feature selection and retrain
        retrained_results_clf = automl_clf.retrain_with_selected_features(
            X_full_train=X_full_clf, y_full_train=y_full_clf, X_full_test=X_test_full_clf, y_full_test=y_test_full_clf, feature_names=feature_names_clf, shap_threshold=0.95
        )
        logger.info(f"Selected features for classification: {retrained_results_clf['selected_feature_names']}")
        logger.info(f"Retrained model test accuracy (with selected features): {retrained_results_clf['retrained_metric_value']:.4f}")
        # Optionally, check uncertainty for the retrained model (should also raise error for classification)
        try:
            uncertainty_retrained_clf = retrained_results_clf["retrained_model_instance"].predict_uncertainty(retrained_results_clf["X_test_filtered"])
            logger.info(f"Mean uncertainty estimate of retrained classification model: {np.mean(uncertainty_retrained_clf):.4f}")
        except Exception as e:
            logger.error(f"Error getting uncertainty for retrained classification model: {e}")

        # Export the retrained model
        model_export_path_clf = "best_automl_clf_model.pkl"
        automl_clf.export_model(retrained_results_clf["retrained_model_instance"], model_export_path_clf)

        # Demonstrate loading the model and making a prediction
        try:
            loaded_model_clf = AutoML.load_model(model_export_path_clf)
            # Make a prediction with the loaded model (using a sample from filtered test data)
            sample_input_clf_original = X_test_full_clf[0:1]
            # The loaded model's predict method expects data in the *scaled* format it was trained on.
            # So, we need to transform the sample input using the fitted feature scaler *from the AutoML instance*.
            if automl_clf._fitted_feature_scaler:
                sample_input_clf_scaled = automl_clf._fitted_feature_scaler.transform(sample_input_clf_original)
            else:
                sample_input_clf_scaled = sample_input_clf_original

            loaded_prediction_clf = loaded_model_clf.predict(sample_input_clf_scaled)
            logger.info(f"Prediction from loaded classification model for sample {sample_input_clf_original[0].round(2)}: {loaded_prediction_clf[0]}")
        except Exception as e:
            logger.error(f"Error demonstrating model load and predict for classification: {e}")

    # Save the leaderboard after all operations
    automl_clf.save_leaderboard("classification_leaderboard.json")
