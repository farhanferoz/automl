"""Module for explaining model predictions using SHAP."""

import contextlib
import os
import sys
from collections.abc import Generator
from copy import deepcopy

import numpy as np
import pandas as pd
import shap
import torch
from sklearn.pipeline import Pipeline

from automl_package.enums import ExplainerType, TaskType, UncertaintyMethod
from automl_package.logger import logger
from automl_package.models.base import BaseModel
from automl_package.models.catboost_model import CatBoostModel


class FeatureExplainer:
    """A class to explain model predictions using SHAP (SHapley Additive exPlanations).

    It adapts to different model types (Tree-based, Neural Networks, other scikit-learn models)
    and handles scikit-learn pipelines with an assumed 'scaler' and 'model' step.
    """

    def __init__(
        self,
        model_instance: BaseModel | Pipeline,
        x_background: np.ndarray,
        feature_names: list[str] | None = None,
        device: torch.device | None = None,
        max_data_points: int = 50000,
        random_state: int | None = None,
    ) -> None:
        """Initializes the SHAP explainer based on the model type.

        Args:
            model_instance (Union[BaseModel, Pipeline]): The trained model instance from the AutoML pipeline. Can be a BaseModel subclass or a scikit-learn Pipeline.
            x_background (np.ndarray): A background dataset (e.g., a subset of training data) for explainer initialization. This data should be in the *scaled* format if the
            model expects scaled input.
            feature_names (List[str], optional): Names of the features. If None, uses generic names.
            device (torch.device, optional): The device (CPU or GPU) to use for PyTorch models.
            max_data_points (int): Maximum number of data points to use for SHAP explanation.
            random_state (int, optional): Random seed for reproducibility. Defaults to None.
        """
        self.original_model = model_instance  # Store the original object (can be pipeline)
        self.x_background = x_background
        self.feature_names = feature_names if feature_names is not None else [f"Feature_{i}" for i in range(x_background.shape[1])]
        self.explainer = None  # SHAP explainer object
        self.pipeline = None  # To store the sklearn Pipeline if present
        self.scaler = None  # To store the StandardScaler if present within a pipeline
        self.model_to_explain_directly = None  # The actual BaseModel or raw model extracted from pipeline

        self.device = device
        self.max_data_points = max_data_points
        self.random_state = random_state
        self._initialize_explainer()

    def _get_model_name(self) -> str:
        if isinstance(self.original_model, Pipeline):
            logger.info("Pipeline detected for SHAP explanation. Extracting model and scaler steps.")
            self.pipeline = self.original_model
            self.model_to_explain_directly = self.pipeline.named_steps.get("model")
            if self.model_to_explain_directly is None:
                raise ValueError("Pipeline must contain a step named 'model' to be explainable by FeatureExplainer.")
            self.scaler = self.pipeline.named_steps.get("scaler")
            return self.model_to_explain_directly.name if isinstance(self.model_to_explain_directly, BaseModel) else type(self.model_to_explain_directly).__name__
        if isinstance(self.original_model, BaseModel):
            self.model_to_explain_directly = self.original_model
            return self.model_to_explain_directly.name
        self.model_to_explain_directly = self.original_model
        return type(self.original_model).__name__

    def _summarize_background_data(self, x_background: np.ndarray, n_samples: int = 100) -> np.ndarray:
        if x_background.shape[0] > n_samples * 2:
            logger.info(f"Summarizing background data from {x_background.shape[0]} to {n_samples} samples using shap.kmeans.")
            return shap.kmeans(x_background, n_samples)
        return x_background

    @contextlib.contextmanager
    def _suppress_output(self) -> Generator[None, None, None]:
        with open(os.devnull, "w") as devnull:
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            sys.stdout = devnull
            sys.stderr = devnull
            try:
                yield
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

    def _create_explainer(self) -> None:
        model_instance = self.model_to_explain_directly
        explainer_info = model_instance.get_shap_explainer_info()
        explainer_type = explainer_info["explainer_type"]
        model_to_explain = explainer_info["model"]

        if explainer_type == ExplainerType.TREE:
            logger.info(f"Using shap.TreeExplainer for {model_instance.name} model.")
            self.explainer = shap.TreeExplainer(model_to_explain)

        elif explainer_type == ExplainerType.DEEP:
            logger.info(f"Using shap.DeepExplainer for {model_instance.name} model.")
            x_background_values = self.x_background.values if isinstance(self.x_background, pd.DataFrame) else self.x_background
            background_data = self._summarize_background_data(x_background_values)

            if hasattr(background_data, "data"):
                background_data = background_data.data

            if isinstance(background_data, memoryview):
                background_data = np.asarray(background_data)

            background_tensor = torch.tensor(background_data, dtype=torch.float32).to(self.device)
            model_on_device = model_to_explain.to(self.device)

            with self._suppress_output():
                self.explainer = shap.DeepExplainer(model_on_device, background_tensor)

        elif explainer_type == ExplainerType.LINEAR:
            logger.info(f"Using shap.LinearExplainer for {model_instance.name} model.")
            with self._suppress_output():
                self.explainer = shap.LinearExplainer(model_to_explain, self.x_background)

        elif explainer_type == ExplainerType.KERNEL:
            logger.warning(f"Using shap.KernelExplainer for {model_instance.name} model. This may be slow.")
            background_data = self._summarize_background_data(self.x_background)
            with self._suppress_output():
                self.explainer = shap.KernelExplainer(model_to_explain, background_data)

        elif explainer_type == ExplainerType.CATBOOST_PROBABILISTIC_PROXY:
            logger.warning("Using a proxy model for SHAP explanation of probabilistic CatBoost model. This provides an approximation of feature importances.")
            mean_predictions = model_to_explain.predict(self.x_background)

            # Create a clean set of parameters for the proxy model
            proxy_params = deepcopy(model_to_explain.get_params())
            proxy_params["loss_function"] = "RMSE"
            proxy_params["eval_metric"] = "RMSE"
            proxy_params["uncertainty_method"] = UncertaintyMethod.CONSTANT

            proxy_model = CatBoostModel(**proxy_params)
            proxy_model.fit(self.x_background, mean_predictions)

            self.explainer = shap.TreeExplainer(proxy_model.get_internal_model())

        else:
            raise ValueError(f"Unknown explainer type: {explainer_type}")

    def _initialize_explainer(self) -> None:
        self._get_model_name()
        self._create_explainer()

    def explain(self, x_to_explain: np.ndarray) -> np.ndarray:
        """Computes SHAP values for the given data.

        Args:
            x_to_explain (np.ndarray): The dataset for which to compute SHAP values. This data should be in the *scaled* format if the model expects scaled input.

        Returns:
            shap.Explanation: A SHAP Explanation object.
        """
        if self.explainer is None:
            raise RuntimeError("SHAP explainer not initialized.")

        # If the number of data points exceeds max_data_points, randomly sample
        if x_to_explain.shape[0] > self.max_data_points:
            logger.info(f"Sampling {self.max_data_points} data points for SHAP explanation from {x_to_explain.shape[0]} available.")
            rng = np.random.default_rng(self.random_state)
            sample_indices = rng.choice(x_to_explain.shape[0], self.max_data_points, replace=False)
            x_to_explain = x_to_explain[sample_indices]

        if isinstance(self.explainer, shap.DeepExplainer):
            x_to_explain_values = x_to_explain.values if isinstance(x_to_explain, pd.DataFrame) else x_to_explain
            data_for_shap = torch.tensor(x_to_explain_values, dtype=torch.float32).to(self.device)
            shap_values_obj = self.explainer.shap_values(data_for_shap, check_additivity=False)
        else:
            data_for_shap = x_to_explain
            shap_values_obj = self.explainer.shap_values(data_for_shap)

        if isinstance(shap_values_obj, list):
            if len(shap_values_obj) == 1:
                return np.array(shap_values_obj[0])
            # For multi-class classification, `shap_values_obj` is a list where each element
            # is an array of SHAP values for that class.
            # For binary classification (where output_size=1), DeepExplainer might return a list of 2 arrays
            # (one for class 0, one for class 1). We typically explain the positive class (index 1).
            if hasattr(self.model_to_explain_directly, "task_type") and self.model_to_explain_directly.task_type == TaskType.CLASSIFICATION:
                if len(shap_values_obj) == 2:  # Binary classification
                    return np.array(shap_values_obj[1])  # SHAP values for the positive class
                # Multi-class
                # For multi-class, often useful to get the SHAP values for the predicted class,
                # or average their absolute values. For a general summary, mean absolute over classes.
                logger.warning("Multi-class classification SHAP values: Returning mean of absolute SHAP values across classes for summary.")
                return np.mean(np.abs(np.array(shap_values_obj)), axis=0)
            # Multi-output regression (if any model has this and explainer supports)
            # Average across outputs or handle specifically
            logger.warning("Multi-output regression SHAP values: Returning mean of absolute SHAP values across outputs for summary.")
            return np.mean(np.array(shap_values_obj), axis=0)
        return np.array(shap_values_obj)

    def get_feature_importance_summary(self, shap_values_object: np.ndarray, normalize: bool = True) -> dict[str, float]:
        """Calculates global feature importance based on mean absolute SHAP values.

        Args:
            shap_values_object (np.ndarray): The SHAP values, which can be a single array (regression, binary classification) or a list of arrays
            (multi-class classification).
            normalize (bool): Whether to normalize the feature importances to sum to 1. Defaults to True.

        Returns:
            Dict[str, float]: A dictionary of feature names and their mean absolute SHAP values, sorted.
        """
        shap_values = np.abs(shap_values_object)

        if shap_values.ndim == 3:
            shap_values = np.squeeze(shap_values, axis=2) if shap_values.shape[2] == 1 else np.mean(shap_values, axis=2)

        mean_abs_shap = np.mean(shap_values, axis=0)

        if len(mean_abs_shap) != len(self.feature_names):
            logger.warning(
                f"Mismatch between number of SHAP values ({len(mean_abs_shap)}) and feature names ({len(self.feature_names)}). "
                f"Check model input consistency or provide correct feature_names."
            )
            # Adjust feature names if mismatch, to prevent errors
            self.feature_names = [f"Feature_{i}" for i in range(len(mean_abs_shap))]

        feature_importance = dict(zip(self.feature_names, mean_abs_shap.tolist(), strict=False))

        if normalize:
            total_importance = sum(feature_importance.values())
            if total_importance > 0:
                feature_importance = {feature: importance / total_importance for feature, importance in feature_importance.items()}

        # Sort in descending order of importance
        return dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))
