from typing import Dict, List, Union
import jax.numpy as jnp
import numpy as np
import shap
import torch

from ..enums import ModelName, TaskType, UncertaintyMethod  # Import enums
from ..logger import logger
from ..models.base import BaseModel


class FeatureExplainer:
    """
    A class to explain model predictions using SHAP, adapting to different model types.
    """

    def __init__(self, model_instance: BaseModel, X_background: np.ndarray, feature_names: List[str] = None):
        """
        Initializes the SHAP explainer based on the model type.

        Args:
            model_instance (BaseModel): The trained model instance from the AutoML pipeline.
            X_background (np.ndarray): A background dataset (e.g., training data) for explainer initialization.
            feature_names (List[str], optional): Names of the features. If None, uses generic names.
        """
        self.model_instance = model_instance
        self.X_background = X_background
        self.feature_names = feature_names if feature_names is not None else [f"Feature_{i}" for i in range(X_background.shape[1])]
        self.explainer = None
        self._initialize_explainer()

    def _initialize_explainer(self):
        """
        Initializes the appropriate SHAP explainer based on the model type.
        """
        model_name = self.model_instance.name  # Get the string name from the BaseModel

        if model_name in [ModelName.XGBOOST.value, ModelName.LIGHTGBM.value, ModelName.CATBOOST.value]:
            logger.info(f"Using shap.TreeExplainer for {model_name} model.")
            # SHAP TreeExplainer handles CatBoost, XGBoost, LightGBM directly
            self.explainer = shap.TreeExplainer(self.model_instance.get_internal_model())

        elif model_name == ModelName.PYTORCH_NEURAL_NETWORK.value:
            logger.info(f"Using shap.DeepExplainer for PyTorchNeuralNetwork model.")
            # For DeepExplainer, we need the raw PyTorch nn.Module and a background dataset
            # Convert background data to PyTorch tensor
            background_tensor = torch.tensor(self.X_background, dtype=torch.float32).to(self.model_instance.device)

            # The model's prediction function for DeepExplainer
            def pytorch_predict_wrapper(x):
                # Ensure input is on the correct device and is a tensor
                x_tensor = torch.tensor(x, dtype=torch.float32).to(self.model_instance.device)
                self.model_instance.get_internal_model().eval()  # Ensure model is in eval mode
                with torch.no_grad():
                    outputs = self.model_instance.get_internal_model()(x_tensor)
                    if self.model_instance._is_regression_model and self.model_instance.uncertainty_method == UncertaintyMethod.PROBABILISTIC:
                        return outputs[:, 0]  # Explain only the mean for probabilistic regression
                    elif self.model_instance.task_type == TaskType.CLASSIFICATION:
                        return outputs  # For classification, DeepExplainer expects raw logits or probabilities.
                    else:
                        return outputs.squeeze(-1)  # Ensure 1D output for standard regression

            # Use PyTorch's internal model directly
            self.explainer = shap.DeepExplainer(self.model_instance.get_internal_model(), background_tensor)

        elif model_name == ModelName.PROBABILISTIC_REGRESSION.value:
            logger.info(f"Using shap.DeepExplainer for ProbabilisticRegression model.")
            # ProbabilisticRegressionModel exposes its combined_model as the internal model
            background_tensor = torch.tensor(self.X_background, dtype=torch.float32).to(self.model_instance.device)

            def pytorch_combined_predict_wrapper(x):
                x_tensor = torch.tensor(x, dtype=torch.float32).to(self.model_instance.device)
                self.model_instance.get_internal_model().eval()
                with torch.no_grad():
                    return self.model_instance.get_internal_model()(x_tensor)

            self.explainer = shap.DeepExplainer(self.model_instance.get_internal_model(), background_tensor)

        elif model_name == ModelName.JAX_PROBABILISTIC_REGRESSION.value:
            logger.info(f"Using shap.DeepExplainer for JAXProbabilisticRegression model.")
            # For JAX/Flax, need the model definition, params, and a predict function.
            jax_model_info = self.model_instance.get_internal_model()
            model_def = jax_model_info["model_def"]
            params = jax_model_info["params"]
            batch_stats = jax_model_info["batch_stats"]

            background_jax = jnp.array(self.X_background, dtype=jnp.float32)

            def jax_predict_wrapper(x):
                x_jax = jnp.array(x, dtype=jnp.float32)
                variables = model_def.apply({"params": params, "batch_stats": batch_stats}, x_jax, train=False, mutable=False)
                predictions_output = variables["output"]

                if self.model_instance.uncertainty_method == UncertaintyMethod.PROBABILISTIC:
                    return predictions_output[:, 0]
                return predictions_output.squeeze(-1)

            self.explainer = shap.DeepExplainer(
                (lambda x, params_and_state: model_def.apply({"params": params_and_state["params"], "batch_stats": params_and_state["batch_stats"]}, x, train=False)["output"]),
                {"params": params, "batch_stats": batch_stats},
                data=background_jax,
            )

        elif model_name == ModelName.SKLEARN_LOGISTIC_REGRESSION.value:
            logger.info(f"Using shap.LinearExplainer for SKLearnLogisticRegression model.")
            self.explainer = shap.LinearExplainer(self.model_instance.get_internal_model(), self.X_background)

        elif model_name == ModelName.JAX_LINEAR_REGRESSION.value:
            logger.info(f"Using shap.KernelExplainer for JAXLinearRegression model (as it's custom JAX).")

            def jax_linear_predict_wrapper(x):
                return self.model_instance.predict(x)

            self.explainer = shap.KernelExplainer(jax_linear_predict_wrapper, self.X_background)

        else:
            logger.warning(f"Model type '{model_name}' not specifically optimized for SHAP. Falling back to shap.KernelExplainer.")
            self.explainer = shap.KernelExplainer(self.model_instance.predict, self.X_background)

    def explain(self, X_to_explain: np.ndarray):
        """
        Computes SHAP values for the given data.

        Args:
            X_to_explain (np.ndarray): The dataset for which to compute SHAP values.

        Returns:
            shap.Explanation: A SHAP Explanation object.
        """
        if self.explainer is None:
            raise RuntimeError("SHAP explainer not initialized.")

        if self.model_instance.name == ModelName.PYTORCH_NEURAL_NETWORK.value and self.model_instance.task_type == TaskType.CLASSIFICATION and self.model_instance.output_size > 1:
            shap_values = self.explainer.shap_values(torch.tensor(X_to_explain, dtype=torch.float32).to(self.model_instance.device))
            if isinstance(shap_values, list):
                return [np.array(s) for s in shap_values]
            return np.array(shap_values)

        elif self.model_instance.name == ModelName.JAX_PROBABILISTIC_REGRESSION.value and self.model_instance.uncertainty_method == UncertaintyMethod.PROBABILISTIC:
            pass

        shap_values = self.explainer.shap_values(X_to_explain)
        return np.array(shap_values)

    def get_feature_importance_summary(self, shap_values_object: Union[np.ndarray, List[np.ndarray]]) -> Dict[str, float]:
        """
        Computes mean absolute SHAP values for each feature from a SHAP Explanation object.

        Args:
            shap_values_object (Union[np.ndarray, List[np.ndarray]]): The SHAP values, which can be
                                                                        a single array (regression, binary classification)
                                                                        or a list of arrays (multi-class classification).

        Returns:
            Dict[str, float]: A dictionary of feature names and their mean absolute SHAP values, sorted.
        """
        if isinstance(shap_values_object, list):  # Multi-class classification
            abs_shap_values = np.mean(np.abs(np.array(shap_values_object)), axis=0)
        else:  # Regression or binary classification
            abs_shap_values = np.abs(shap_values_object)

        mean_abs_shap = np.mean(abs_shap_values, axis=0)

        if len(mean_abs_shap) != len(self.feature_names):
            logger.warning(f"Mismatch between number of SHAP values ({len(mean_abs_shap)}) and feature names ({len(self.feature_names)}). Check model input consistency.")
            self.feature_names = [f"Feature_{i}" for i in range(len(mean_abs_shap))]

        feature_importance = dict(zip(self.feature_names, mean_abs_shap))

        # Sort in descending order of importance
        sorted_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))
        return sorted_importance
