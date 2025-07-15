from typing import Dict, List, Union
import numpy as np
import shap
import torch
from sklearn.pipeline import Pipeline  # Import Pipeline for type hinting

from ..enums import ModelName, TaskType
from ..logger import logger
from ..models.base import BaseModel
from ..models.neural_network import PyTorchNeuralNetwork  # Specific import for PyTorchNN DeepExplainer setup
from ..models.probabilistic_regression import ProbabilisticRegressionModel  # Specific import for its internal model


class FeatureExplainer:
    """
    A class to explain model predictions using SHAP (SHapley Additive exPlanations).
    It adapts to different model types (Tree-based, Neural Networks, other scikit-learn models)
    and handles scikit-learn pipelines with an assumed 'scaler' and 'model' step.
    """

    def __init__(
        self, model_instance: Union[BaseModel, Pipeline], X_background: np.ndarray, feature_names: List[str] = None, device: torch.device = None, max_data_points: int = 50000
    ):
        """
        Initializes the SHAP explainer based on the model type.

        Args:
            model_instance (Union[BaseModel, Pipeline]): The trained model instance from the AutoML pipeline.
                                                          Can be a BaseModel subclass or a scikit-learn Pipeline.
            X_background (np.ndarray): A background dataset (e.g., a subset of training data) for explainer initialization.
                                       This data should be in the *scaled* format if the model expects scaled input.
            feature_names (List[str], optional): Names of the features. If None, uses generic names.
        """
        self.original_model = model_instance  # Store the original object (can be pipeline)
        self.X_background = X_background
        self.feature_names = feature_names if feature_names is not None else [f"Feature_{i}" for i in range(X_background.shape[1])]
        self.explainer = None  # SHAP explainer object
        self.pipeline = None  # To store the sklearn Pipeline if present
        self.scaler = None  # To store the StandardScaler if present within a pipeline
        self.model_to_explain_directly = None  # The actual BaseModel or raw model extracted from pipeline

        self.device = device
        self.max_data_points = max_data_points
        self._initialize_explainer()

    def _initialize_explainer(self):
        """
        Initializes the appropriate SHAP explainer based on the model type.
        Handles extraction of model and scaler from scikit-learn Pipelines.
        """
        # Check if the model_instance is a scikit-learn Pipeline
        if isinstance(self.original_model, Pipeline):
            logger.info("Pipeline detected for SHAP explanation. Extracting model and scaler steps.")
            self.pipeline = self.original_model
            # Attempt to get the 'model' step from the pipeline
            self.model_to_explain_directly = self.pipeline.named_steps.get("model")
            if self.model_to_explain_directly is None:
                raise ValueError("Pipeline must contain a step named 'model' to be explainable by FeatureExplainer.")

            # Attempt to get the 'scaler' step from the pipeline (optional, handled by FeatureExplainer's X_background)
            self.scaler = self.pipeline.named_steps.get("scaler")

            # Get the name from the extracted BaseModel
            if isinstance(self.model_to_explain_directly, BaseModel):
                model_name_str = self.model_to_explain_directly.name
            else:  # Fallback for non-BaseModel sklearn estimators
                model_name_str = type(self.model_to_explain_directly).__name__

        # If it's a BaseModel (not wrapped in a pipeline or it's the extracted one from pipeline)
        elif isinstance(self.original_model, BaseModel):
            self.model_to_explain_directly = self.original_model
            model_name_str = self.model_to_explain_directly.name
        else:
            # For raw scikit-learn or other models not wrapped in BaseModel
            self.model_to_explain_directly = self.original_model
            model_name_str = type(self.original_model).__name__

        # Initialize the specific SHAP explainer based on model type
        if model_name_str in [ModelName.XGBOOST.value, ModelName.LIGHTGBM.value, ModelName.CATBOOST.value]:
            logger.info(f"Using shap.TreeExplainer for {model_name_str} model.")
            # SHAP TreeExplainer handles CatBoost, XGBoost, LightGBM directly
            self.explainer = shap.TreeExplainer(self.model_to_explain_directly.get_internal_model())

        elif model_name_str in [ModelName.PYTORCH_NEURAL_NETWORK.value, ModelName.FLEXIBLE_NEURAL_NETWORK.value]:
            logger.info(f"Using shap.DeepExplainer for {model_name_str} model.")
            # For DeepExplainer, we need the raw PyTorch nn.Module and a background dataset
            background_tensor = torch.tensor(self.X_background, dtype=torch.float32).to(self.device)

            self.explainer = shap.DeepExplainer(self.model_to_explain_directly.get_internal_model(), background_tensor)

        elif model_name_str == ModelName.PROBABILISTIC_REGRESSION.value:
            logger.info(f"Using shap.DeepExplainer for ProbabilisticRegression model.")
            # ProbabilisticRegressionModel exposes its combined_model (which is an nn.Module) as the internal model
            background_tensor = torch.tensor(self.X_background, dtype=torch.float32).to(self.device)

            self.explainer = shap.DeepExplainer(self.model_to_explain_directly.get_internal_model(), background_tensor)

        elif model_name_str == ModelName.SKLEARN_LOGISTIC_REGRESSION.value:
            logger.info(f"Using shap.LinearExplainer for SKLearnLogisticRegression model.")
            # LinearExplainer is efficient for linear models
            self.explainer = shap.LinearExplainer(self.model_to_explain_directly.get_internal_model(), self.X_background)

        elif model_name_str == ModelName.JAX_LINEAR_REGRESSION.value:
            logger.info(f"Using shap.KernelExplainer for JAXLinearRegression model (as it's custom JAX and not tree/deep).")

            # For custom JAX models not suitable for DeepExplainer, KernelExplainer is a general fallback
            def jax_linear_predict_wrapper(x):
                return self.model_to_explain_directly.predict(x)

            self.explainer = shap.KernelExplainer(jax_linear_predict_wrapper, self.X_background)

        else:
            logger.warning(f"Model type '{model_name_str}' not specifically optimized for SHAP. Falling back to shap.KernelExplainer.")
            # For other models, KernelExplainer is a universal fallback. It takes a prediction function.
            # We use the BaseModel's predict method.
            self.explainer = shap.KernelExplainer(self.model_to_explain_directly.predict, self.X_background)

    def explain(self, X_to_explain: np.ndarray):
        """
        Computes SHAP values for the given data.

        Args:
            X_to_explain (np.ndarray): The dataset for which to compute SHAP values.
                                       This data should be in the *scaled* format if the model expects scaled input.

        Returns:
            shap.Explanation: A SHAP Explanation object.
        """
        if self.explainer is None:
            raise RuntimeError("SHAP explainer not initialized.")

        # If the number of data points exceeds max_data_points, randomly sample
        if X_to_explain.shape[0] > self.max_data_points:
            logger.info(f"Sampling {self.max_data_points} data points for SHAP explanation from {X_to_explain.shape[0]} available.")
            np.random.seed(42)  # for reproducibility
            sample_indices = np.random.choice(X_to_explain.shape[0], self.max_data_points, replace=False)
            X_to_explain = X_to_explain[sample_indices]

        # Ensure data is in the correct format for the explainer (e.g., PyTorch tensor for DeepExplainer)
        data_for_shap = X_to_explain
        if isinstance(self.model_to_explain_directly, (PyTorchNeuralNetwork, ProbabilisticRegressionModel)):
            data_for_shap = torch.tensor(X_to_explain, dtype=torch.float32).to(self.model_to_explain_directly.device)

        shap_values_obj = self.explainer.shap_values(data_for_shap)

        # shap.DeepExplainer for multi-output models returns a list of arrays.
        # For single output regression, it's usually just an array.
        if isinstance(shap_values_obj, list):
            # For multi-class classification, `shap_values_obj` is a list where each element
            # is an array of SHAP values for that class.
            # For binary classification (where output_size=1), DeepExplainer might return a list of 2 arrays
            # (one for class 0, one for class 1). We typically explain the positive class (index 1).
            if hasattr(self.model_to_explain_directly, "task_type") and self.model_to_explain_directly.task_type == TaskType.CLASSIFICATION:
                if len(shap_values_obj) == 2:  # Binary classification
                    return np.array(shap_values_obj[1])  # SHAP values for the positive class
                else:  # Multi-class
                    # For multi-class, often useful to get the SHAP values for the predicted class,
                    # or average their absolute values. For a general summary, mean absolute over classes.
                    logger.warning("Multi-class classification SHAP values: Returning mean of absolute SHAP values across classes for summary.")
                    return np.mean(np.abs(np.array(shap_values_obj)), axis=0)
            else:  # Multi-output regression (if any model has this and explainer supports)
                # Average across outputs or handle specifically
                logger.warning("Multi-output regression SHAP values: Returning mean of absolute SHAP values across outputs for summary.")
                return np.mean(np.array(shap_values_obj), axis=0)
        return np.array(shap_values_obj)

    def get_feature_importance_summary(self, shap_values_object: Union[np.ndarray, List[np.ndarray]]) -> Dict[str, float]:
        """
        Calculates global feature importance based on mean absolute SHAP values.

        Args:
            shap_values_object (Union[np.ndarray, List[np.ndarray]]): The SHAP values, which can be
                                                                        a single array (regression, binary classification)
                                                                        or a list of arrays (multi-class classification).

        Returns:
            Dict[str, float]: A dictionary of feature names and their mean absolute SHAP values, sorted.
        """
        if isinstance(shap_values_object, list):  # Multi-class classification from TreeExplainer/DeepExplainer
            # Convert list of arrays to a single 3D array (num_classes, num_samples, num_features)
            # Then take mean of absolute values across num_classes and num_samples
            abs_shap_values = np.mean(np.abs(np.array(shap_values_object)), axis=0)
        else:  # Regression or binary classification (single array: num_samples, num_features)
            abs_shap_values = np.abs(shap_values_object)

        mean_abs_shap = np.mean(abs_shap_values, axis=0)

        if len(mean_abs_shap) != len(self.feature_names):
            logger.warning(
                f"Mismatch between number of SHAP values ({len(mean_abs_shap)}) and feature names ({len(self.feature_names)}). Check model input consistency or provide correct feature_names."
            )
            # Adjust feature names if mismatch, to prevent errors
            self.feature_names = [f"Feature_{i}" for i in range(len(mean_abs_shap))]

        feature_importance = dict(zip(self.feature_names, mean_abs_shap))

        # Sort in descending order of importance
        sorted_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))
        return sorted_importance
