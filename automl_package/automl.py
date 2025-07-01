from enum import Enum
import numpy as np
import pandas as pd  # Import pandas for feature names in SHAP output
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, accuracy_score, log_loss
from typing import Dict, Any, Type, Union, List, Tuple, Optional
import optuna  # Import optuna for type hinting in objective function
import json  # New import
from datetime import datetime  # New import
import joblib  # New import for model export/import
import torch.nn as nn  # Import for activation functions
import matplotlib.pyplot as plt  # New import for plotting
import seaborn as sns  # New import for plotting

from .models.base import BaseModel
from .models.linear_regression import JAXLinearRegression
from .models.neural_network import PyTorchNeuralNetwork, FlexibleHiddenLayersNN  # <--- ADD FlexibleHiddenLayersNN
from .models.xgboost_lgbm import XGBoostModel, LightGBMModel
from .models.sklearn_logistic_regression import SKLearnLogisticRegression
from .models.catboost_model import CatBoostModel
from .models.probabilistic_regression import ProbabilisticRegressionModel
from .models.classifier_regression import ClassifierRegressionModel
from .optimizers.optuna_optimizer import OptunaOptimizer
from .enums import UncertaintyMethod, RegressionStrategy, MapperType, TaskType, ModelName
from .logger import logger
from .utils.metrics import Metrics
from .explainers.feature_explainer import FeatureExplainer


class AutoML:
    """
    An Automated Machine Learning (AutoML) orchestrator.
    Manages model selection, hyperparameter optimization, and training.
    """

    def __init__(
        self,
        task_type: TaskType = TaskType.REGRESSION,  # Use enum
        metric: str = "rmse",  # e.g., 'rmse', 'accuracy', 'log_loss'
        n_trials: int = 20,
        n_splits: int = 5,
        random_state: int = 42,
        feature_scaler: Any = None,  # New: Optional feature scaler
        target_scaler: Any = None,
    ):  # New: Optional target scaler
        """
        Initializes the AutoML pipeline.

        Args:
            task_type (TaskType): The type of machine learning task (TaskType.REGRESSION or TaskType.CLASSIFICATION).
            metric (str): The evaluation metric to optimize.
                          For regression: 'rmse', 'mse'.
                          For classification: 'accuracy', 'log_loss'.
            n_trials (int): Number of Optuna trials for hyperparameter optimization per model.
            n_splits (int): Number of folds for K-Fold cross-validation during optimization.
            random_state (int): Random seed for reproducibility.
            feature_scaler (Any, optional): An unfitted scikit-learn compatible scaler for features (e.g., StandardScaler()).
                                            If None, no feature scaling is applied by AutoML.
            target_scaler (Any, optional): An unfitted scikit-learn compatible scaler for targets (e.g., StandardScaler()).
                                            Only applicable for regression tasks. If None, no target scaling is applied.
        """
        self.task_type = task_type
        self.metric = metric
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.random_state = random_state

        # Initialize scalers
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler if self.task_type == TaskType.REGRESSION else None

        # Internal storage for fitted scalers
        self._fitted_feature_scaler = None
        self._fitted_target_scaler = None

        # Use ModelName enum for registry keys
        self.models_registry: Dict[ModelName, Type[BaseModel]] = {
            ModelName.JAX_LINEAR_REGRESSION: JAXLinearRegression,
            ModelName.PYTORCH_NEURAL_NETWORK: PyTorchNeuralNetwork,
            # --- REGISTER THE NEW MODEL HERE ---
            ModelName.FLEXIBLE_NEURAL_NETWORK: FlexibleHiddenLayersNN,
            # -----------------------------------
            ModelName.XGBOOST: XGBoostModel,
            ModelName.LIGHTGBM: LightGBMModel,
            ModelName.SKLEARN_LOGISTIC_REGRESSION: SKLearnLogisticRegression,
            ModelName.CATBOOST: CatBoostModel,
            ModelName.CLASSIFIER_REGRESSION: ClassifierRegressionModel,
            ModelName.PROBABILISTIC_REGRESSION: ProbabilisticRegressionModel,
        }
        # Use ModelName enum for best_model_name storage
        self.trained_models: Dict[ModelName, Tuple[BaseModel, Dict[str, Any], float]] = {}  # Stores best_model_instance, best_params, best_score
        self.best_model_name: Optional[ModelName] = None
        self.best_overall_metric: float = float("inf") if self._is_minimize_metric() else -float("inf")

        self.optuna_optimizer = OptunaOptimizer(direction="minimize" if self._is_minimize_metric() else "maximize", n_trials=n_trials, seed=random_state)

        self.X_train_for_shap: Optional[np.ndarray] = None  # For SHAP background data
        self.leaderboard: List[Dict[str, Any]] = []  # New: Initialize leaderboard

        # Ensure regression-only models are only used for regression tasks
        if self.task_type == TaskType.CLASSIFICATION:
            # Removed models that are strictly regression and not suitable for classification output
            self.models_registry = {
                k: v
                for k, v in self.models_registry.items()
                if k
                not in [
                    ModelName.CLASSIFIER_REGRESSION,  # ClassifierRegression output is continuous, for regression tasks
                    ModelName.PROBABILISTIC_REGRESSION,
                    ModelName.JAX_LINEAR_REGRESSION,
                ]
            }
            logger.info(f"Removed regression-only models from consideration for {self.task_type.value} task.")

    def _is_minimize_metric(self) -> bool:
        """Determines if the metric should be minimized."""
        return self.metric in ["rmse", "mse", "log_loss"]

    def _evaluate_metric(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> float:
        """Evaluates the chosen metric."""
        if self.task_type == TaskType.REGRESSION:
            if self.metric == "rmse":
                return np.sqrt(mean_squared_error(y_true, y_pred))
            elif self.metric == "mse":
                return mean_squared_error(y_true, y_pred)
            else:
                raise ValueError(f"Unsupported regression metric: {self.metric}")
        elif self.task_type == TaskType.CLASSIFICATION:
            if self.metric == "accuracy":
                # y_pred for accuracy should be class labels (0 or 1)
                return accuracy_score(y_true, y_pred)
            elif self.metric == "log_loss":
                # y_proba for log_loss should be probabilities for the positive class (binary) or (N, num_classes) for multi-class
                if y_proba is None:
                    raise ValueError("y_proba is required for 'log_loss' metric.")

                # If y_proba is 1D (binary probabilities), reshape to (N, 2) for log_loss
                if y_proba.ndim == 1:
                    y_proba_for_logloss = np.vstack((1 - y_proba, y_proba)).T
                else:
                    y_proba_for_logloss = y_proba  # Already (N, num_classes)

                return log_loss(y_true, y_proba_for_logloss)
            else:
                raise ValueError(f"Unsupported classification metric: {self.metric}")
        else:
            raise ValueError("Task type must be 'regression' or 'classification'.")

    def _instantiate_model(self, model_name: ModelName, params: Dict[str, Any], input_features: int, num_classes: int = None) -> BaseModel:
        """Instantiates a model from the registry with given parameters."""
        if model_name == ModelName.CLASSIFIER_REGRESSION:
            # ClassifierRegressionModel needs special handling for its nested parameters
            base_classifier_name_str = params.pop("base_classifier_name")
            base_model_enum = ModelName(base_classifier_name_str)  # Convert string back to Enum
            base_classifier_class = self.models_registry[base_model_enum]

            mapper_type_str = params.pop("mapper_type", MapperType.SPLINE.value)
            mapper_type = MapperType(mapper_type_str)

            # Reconstruct base_classifier_params
            base_classifier_params_reconstructed = {k.replace("base__", ""): v for k, v in params.items() if k.startswith("base__")}
            for k in list(params.keys()):  # Remove consumed params from top-level `params`
                if k.startswith("base__"):
                    del params[k]

            # Convert enum strings back to Enum objects within reconstructed params
            if "uncertainty_method" in base_classifier_params_reconstructed and isinstance(base_classifier_params_reconstructed["uncertainty_method"], str):
                base_classifier_params_reconstructed["uncertainty_method"] = UncertaintyMethod(base_classifier_params_reconstructed["uncertainty_method"])
            if "activation" in base_classifier_params_reconstructed and isinstance(base_classifier_params_reconstructed["activation"], str):
                base_classifier_params_reconstructed["activation"] = nn.ReLU if base_classifier_params_reconstructed["activation"] == "ReLU" else nn.Tanh

            # Reconstruct mapper_params
            mapper_params_reconstructed = {}
            if mapper_type in [MapperType.LOOKUP_MEAN, MapperType.LOOKUP_MEDIAN]:
                mapper_params_reconstructed["n_partitions_min"] = params.pop("n_partitions_min_lookup", 5)
                mapper_params_reconstructed["n_partitions_max"] = params.pop("n_partitions_max_lookup", np.inf)
            elif mapper_type == MapperType.SPLINE:
                mapper_params_reconstructed["spline_k"] = params.pop("spline_k", 3)
                mapper_params_reconstructed["spline_s"] = params.pop("spline_s", None)

            # Remove 'n_classes' from params before passing to ClassifierRegressionModel's init
            n_classes = params.pop("n_classes")

            # Handle specific parameters for internal models when passed through ClassifierRegression
            if base_model_enum == ModelName.PYTORCH_NEURAL_NETWORK:
                base_classifier_params_reconstructed["input_size"] = input_features
                base_classifier_params_reconstructed["output_size"] = 1 if n_classes == 2 else n_classes  # n_classes from outer model
                base_classifier_params_reconstructed["task_type"] = TaskType.CLASSIFICATION
                # Convert string activation to Type[nn.Module]
                if "activation" in base_classifier_params_reconstructed and isinstance(base_classifier_params_reconstructed["activation"], str):
                    base_classifier_params_reconstructed["activation"] = nn.ReLU if base_classifier_params_reconstructed["activation"] == "ReLU" else nn.Tanh
                # Ensure dropout_rate and n_mc_dropout_samples are set if uncertainty_method is MC_DROPOUT
                if base_classifier_params_reconstructed.get("uncertainty_method") == UncertaintyMethod.MC_DROPOUT.value:
                    base_classifier_params_reconstructed.setdefault("dropout_rate", 0.1)
                    base_classifier_params_reconstructed.setdefault("n_mc_dropout_samples", 100)
                else:  # If not MC_DROPOUT, ensure dropout is off
                    base_classifier_params_reconstructed["dropout_rate"] = 0.0
            elif base_model_enum == ModelName.FLEXIBLE_NEURAL_NETWORK:
                base_classifier_params_reconstructed["input_size"] = input_features
                base_classifier_params_reconstructed["output_size"] = 1 if n_classes == 2 else n_classes
                base_classifier_params_reconstructed["task_type"] = TaskType.CLASSIFICATION
                if "activation" in base_classifier_params_reconstructed and isinstance(base_classifier_params_reconstructed["activation"], str):
                    base_classifier_params_reconstructed["activation"] = nn.ReLU if base_classifier_params_reconstructed["activation"] == "ReLU" else nn.Tanh
                if base_classifier_params_reconstructed.get("uncertainty_method") == UncertaintyMethod.MC_DROPOUT.value:
                    base_classifier_params_reconstructed.setdefault("dropout_rate", 0.1)
                    base_classifier_params_reconstructed.setdefault("n_mc_dropout_samples", 100)
                else:
                    base_classifier_params_reconstructed["dropout_rate"] = 0.0
            elif base_model_enum == ModelName.XGBOOST:
                base_classifier_params_reconstructed["objective"] = "binary:logistic" if n_classes == 2 else "multi:softmax"
                base_classifier_params_reconstructed["eval_metric"] = "logloss"
            elif base_model_enum == ModelName.LIGHTGBM:
                base_classifier_params_reconstructed["objective"] = "binary" if n_classes == 2 else "multiclass"
                base_classifier_params_reconstructed["metric"] = "binary_logloss"
            elif base_model_enum == ModelName.CATBOOST:
                base_classifier_params_reconstructed["task_type"] = TaskType.CLASSIFICATION
                base_classifier_params_reconstructed["loss_function"] = "Logloss" if n_classes == 2 else "MultiClass"

            return ClassifierRegressionModel(
                n_classes=n_classes,
                base_classifier_class=base_classifier_class,
                base_classifier_params=base_classifier_params_reconstructed,
                mapper_type=mapper_type,
                mapper_params=mapper_params_reconstructed,
                **params,  # Pass any remaining top-level params for ClassifierRegressionModel
            )
        elif model_name == ModelName.PYTORCH_NEURAL_NETWORK:
            # PyTorchNeuralNetwork parameters
            hidden_sizes_list = [params.pop("hidden_size")] * params.pop("hidden_layers")
            activation_str = params.pop("activation", "ReLU")
            activation_fn = nn.ReLU if activation_str == "ReLU" else nn.Tanh
            uncertainty_method_enum = UncertaintyMethod(params.pop("uncertainty_method"))

            return self.models_registry[model_name](
                input_size=input_features,
                hidden_sizes=hidden_sizes_list,
                output_size=1 if self.task_type == TaskType.REGRESSION else (1 if num_classes == 2 else num_classes),
                learning_rate=params.pop("learning_rate"),
                n_epochs=params.pop("n_epochs"),
                batch_size=params.pop("batch_size"),
                task_type=self.task_type,
                use_batch_norm=params.pop("use_batch_norm"),
                uncertainty_method=uncertainty_method_enum,
                n_mc_dropout_samples=params.pop("n_mc_dropout_samples", 100),
                dropout_rate=params.pop("dropout_rate", 0.1),
                l1_lambda=params.pop("l1_lambda", 0.0),
                l2_lambda=params.pop("l2_lambda", 0.0),
                activation=activation_fn,
                **params,  # Pass any remaining
            )
        # --- INSTANTIATE THE NEW FLEXIBLE_NEURAL_NETWORK MODEL HERE ---
        elif model_name == ModelName.FLEXIBLE_NEURAL_NETWORK:
            activation_str = params.pop("activation", "ReLU")
            activation_fn = nn.ReLU if activation_str == "ReLU" else nn.Tanh
            uncertainty_method_enum = UncertaintyMethod(params.pop("uncertainty_method"))

            return self.models_registry[model_name](
                input_size=input_features,
                max_hidden_layers=params.pop("max_hidden_layers"),
                hidden_size=params.pop("hidden_size"),
                output_size=1 if self.task_type == TaskType.REGRESSION else (1 if num_classes == 2 else num_classes),
                learning_rate=params.pop("learning_rate"),
                n_epochs=params.pop("n_epochs"),
                batch_size=params.pop("batch_size"),
                task_type=self.task_type,
                use_batch_norm=params.pop("use_batch_norm"),
                uncertainty_method=uncertainty_method_enum,
                n_mc_dropout_samples=params.pop("n_mc_dropout_samples", 100),
                dropout_rate=params.pop("dropout_rate", 0.1),
                l1_lambda=params.pop("l1_lambda", 0.0),
                l2_lambda=params.pop("l2_lambda", 0.0),
                activation=activation_fn,
                **params,
            )
        # -----------------------------------------------------------
        elif model_name == ModelName.PROBABILISTIC_REGRESSION:
            n_classes = params.pop("n_classes")
            regression_strategy_enum = RegressionStrategy(params.pop("regression_strategy"))
            uncertainty_method_enum = UncertaintyMethod(params.pop("uncertainty_method"))

            # Reconstruct base_classifier_params
            base_classifier_params_reconstructed = {k.replace("base_classifier_params__", ""): v for k, v in params.items() if k.startswith("base_classifier_params__")}
            for k in list(params.keys()):  # Remove consumed params
                if k.startswith("base_classifier_params__"):
                    del params[k]
            if "activation" in base_classifier_params_reconstructed and isinstance(base_classifier_params_reconstructed["activation"], str):
                base_classifier_params_reconstructed["activation"] = nn.ReLU if base_classifier_params_reconstructed["activation"] == "ReLU" else nn.Tanh

            # Reconstruct regression_head_params
            regression_head_params_reconstructed = {k.replace("regression_head_params__", ""): v for k, v in params.items() if k.startswith("regression_head_params__")}
            for k in list(params.keys()):  # Remove consumed params
                if k.startswith("regression_head_params__"):
                    del params[k]
            if "activation" in regression_head_params_reconstructed and isinstance(regression_head_params_reconstructed["activation"], str):
                regression_head_params_reconstructed["activation"] = nn.ReLU if regression_head_params_reconstructed["activation"] == "ReLU" else nn.Tanh

            return self.models_registry[model_name](
                input_size=input_features,
                n_classes=n_classes,
                base_classifier_params=base_classifier_params_reconstructed,
                regression_head_params=regression_head_params_reconstructed,
                regression_strategy=regression_strategy_enum,
                learning_rate=params.pop("learning_rate"),
                n_epochs=params.pop("n_epochs"),
                batch_size=params.pop("batch_size"),
                uncertainty_method=uncertainty_method_enum,
                n_mc_dropout_samples=params.pop("n_mc_dropout_samples", 100),
                dropout_rate=params.pop("dropout_rate", 0.1),
                **params,
            )
        elif model_name == ModelName.XGBOOST:
            if self.task_type == TaskType.CLASSIFICATION:
                objective = "binary:logistic" if num_classes == 2 else "multi:softmax"
                eval_metric = "logloss" if num_classes == 2 else "multi:mlogloss"
                return self.models_registry[model_name](objective=objective, eval_metric=eval_metric, **params)
            return self.models_registry[model_name](objective="reg:squarederror", eval_metric="rmse", **params)
        elif model_name == ModelName.LIGHTGBM:
            if self.task_type == TaskType.CLASSIFICATION:
                objective = "binary" if num_classes == 2 else "multiclass"
                metric = "binary_logloss" if num_classes == 2 else "multi_logloss"
                return self.models_registry[model_name](objective=objective, metric=metric, **params)
            return self.models_registry[model_name](objective="regression", metric="rmse", **params)
        elif model_name == ModelName.CATBOOST:
            return self.models_registry[model_name](task_type=self.task_type, **params)
        elif model_name == ModelName.SKLEARN_LOGISTIC_REGRESSION:
            if params.get("penalty") == "elasticnet" and "l1_ratio" in params:  # Ensure l1_ratio only if elasticnet
                return self.models_registry[model_name](**params)
            else:
                params_copy = params.copy()
                if "l1_ratio" in params_copy:
                    del params_copy["l1_ratio"]
                return self.models_registry[model_name](**params_copy)
        elif model_name == ModelName.JAX_LINEAR_REGRESSION:
            return self.models_registry[model_name](**params)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def _sample_params_for_trial(self, trial: optuna.Trial, model_name: ModelName) -> Tuple[Dict[str, Any], Union[str, None]]:
        """Samples hyperparameters for a given model from its search space."""
        model_class = self.models_registry[model_name]
        params = {}
        if model_name == ModelName.CLASSIFIER_REGRESSION:
            params["n_classes"] = trial.suggest_int("n_classes", 2, 5)  # For ClassifierRegression's discretization

            base_classifier_choices_str = [
                ModelName.PYTORCH_NEURAL_NETWORK.value,
                ModelName.XGBOOST.value,
                ModelName.LIGHTGBM.value,
                ModelName.SKLEARN_LOGISTIC_REGRESSION.value,
                ModelName.CATBOOST.value,
            ]
            base_classifier_name_str = trial.suggest_categorical("base_classifier_name", base_classifier_choices_str)
            base_classifier_class = self.models_registry[ModelName(base_classifier_name_str)]
            param_key_prefix = "base__"  # This prefix is specific to base classifier params

            # Get search space for the chosen base classifier
            # Get search space for the chosen base classifier
            if ModelName(base_classifier_name_str) == ModelName.PYTORCH_NEURAL_NETWORK:
                # For temporary instantiation, a generic output_size (e.g., 1 for binary) is sufficient.
                # The actual output_size will be correctly set during the _instantiate_model call.
                temp_model_instance = base_classifier_class(task_type=TaskType.CLASSIFICATION, input_size=10, output_size=1)
            elif ModelName(base_classifier_name_str) == ModelName.FLEXIBLE_NEURAL_NETWORK:
                # Same logic for FlexibleNeuralNetwork
                temp_model_instance = base_classifier_class(task_type=TaskType.CLASSIFICATION, input_size=10, output_size=1)
            elif ModelName(base_classifier_name_str) == ModelName.CATBOOST:
                temp_model_instance = base_classifier_class(task_type=TaskType.CLASSIFICATION)
            elif ModelName(base_classifier_name_str) in [ModelName.XGBOOST, ModelName.LIGHTGBM, ModelName.SKLEARN_LOGISTIC_REGRESSION]:
                temp_model_instance = base_classifier_class(is_classification=True)
            else:
                temp_model_instance = base_classifier_class()  # Fallback for other models

            base_search_space = temp_model_instance.get_hyperparameter_search_space()

            for param_name, config in base_search_space.items():
                param_key = f"base__{param_name}"
                # Conditional sampling for PyTorchNN uncertainty methods within base classifier params
                if param_name == "uncertainty_method" and ModelName(base_classifier_name_str) in [ModelName.PYTORCH_NEURAL_NETWORK, ModelName.FLEXIBLE_NEURAL_NETWORK]:
                    sampled_uncertainty_method_val = trial.suggest_categorical(param_key, [e.value for e in UncertaintyMethod])
                    params[param_key] = sampled_uncertainty_method_val
                    if sampled_uncertainty_method_val == UncertaintyMethod.MC_DROPOUT.value:
                        params[f"{param_key}_n_mc_dropout_samples"] = trial.suggest_int(f"{param_key}_n_mc_dropout_samples", 50, 200, step=50)
                        params[f"{param_key}_dropout_rate"] = trial.suggest_float(f"{param_key}_dropout_rate", 0.1, 0.5, step=0.1)
                    continue
                # Handle l1_ratio for SKLearnLogisticRegression
                if param_name == "l1_ratio" and ModelName(base_classifier_name_str) == ModelName.SKLEARN_LOGISTIC_REGRESSION:
                    if params.get(f"{param_key_prefix}penalty") == "elasticnet":  # Check if penalty is elasticnet (should be sampled already)
                        params[param_key] = trial.suggest_float(param_key, config["low"], config["high"], step=config["step"])
                    continue

                if config["type"] == "int":
                    params[param_key] = trial.suggest_int(param_key, config["low"], config["high"], step=config.get("step", 1))
                elif config["type"] == "float":
                    params[param_key] = trial.suggest_float(param_key, config["low"], config["high"], log=config.get("log", False))
                elif config["type"] == "categorical":
                    params[param_key] = trial.suggest_categorical(param_key, config["choices"])

            # Parameters for the probability mapper
            mapper_type_val = trial.suggest_categorical("mapper_type", [e.value for e in MapperType])
            params["mapper_type"] = mapper_type_val

            # Conditional mapper params
            if MapperType(mapper_type_val) in [MapperType.LOOKUP_MEAN, MapperType.LOOKUP_MEDIAN]:
                params["n_partitions_min_lookup"] = trial.suggest_int("n_partitions_min_lookup", 5, 10)
                params["n_partitions_max_lookup"] = trial.suggest_int("n_partitions_max_lookup", 10, 50)
            elif MapperType(mapper_type_val) == MapperType.SPLINE:
                params["spline_k"] = trial.suggest_int("spline_k", 1, 3)
                params["spline_s"] = trial.suggest_float("spline_s", 0.01, 10.0, log=True)

            return params, base_classifier_name_str

        elif model_name == ModelName.PROBABILISTIC_REGRESSION:
            # Sample common parameters for ProbabilisticRegressionModel
            params["n_classes"] = trial.suggest_int("n_classes", 2, 5)
            params["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
            params["n_epochs"] = trial.suggest_int("n_epochs", 10, 50, step=10)
            params["batch_size"] = trial.suggest_categorical("batch_size", [16, 32, 64])
            params["regression_strategy"] = trial.suggest_categorical("regression_strategy", [e.value for e in RegressionStrategy])
            params["uncertainty_method"] = trial.suggest_categorical("uncertainty_method", [e.value for e in UncertaintyMethod])

            # Conditional top-level dropout params for MC_DROPOUT
            if params["uncertainty_method"] == UncertaintyMethod.MC_DROPOUT.value:
                params["n_mc_dropout_samples"] = trial.suggest_int("n_mc_dropout_samples", 50, 200, step=50)
                params["dropout_rate"] = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)
            else:  # Set dropout to 0 if not MC_DROPOUT
                params["dropout_rate"] = 0.0  # This is the top-level dropout rate
            # Nested params for base classifier (internal PyTorchNN/JAX MLP)
            params["base_classifier_params__hidden_layers"] = trial.suggest_int("base_classifier_params__hidden_layers", 1, 2)
            params["base_classifier_params__hidden_size"] = trial.suggest_int("base_classifier_params__hidden_size", 32, 64, step=32)
            params["base_classifier_params__use_batch_norm"] = trial.suggest_categorical("base_classifier_params__use_batch_norm", [True, False])

            # Nested params for regression heads (internal PyTorchNN/JAX MLP)
            params["regression_head_params__hidden_layers"] = trial.suggest_int("regression_head_params__hidden_layers", 0, 1)
            params["regression_head_params__hidden_size"] = trial.suggest_int("regression_head_params__hidden_size", 16, 32, step=16)
            params["regression_head_params__use_batch_norm"] = trial.suggest_categorical("regression_head_params__use_batch_norm", [True, False])

            return params, None

        else:  # For other standard models including PyTorchNeuralNetwork, FlexibleNeuralNetwork, XGBoost, LightGBM, CatBoost, SKLearnLogisticRegression, JAXLinearRegression
            search_space = model_class().get_hyperparameter_search_space()
            for param_name, config in search_space.items():
                # Conditional sampling for uncertainty method related parameters in PyTorchNNs
                if param_name == "n_mc_dropout_samples" or param_name == "dropout_rate":
                    # Only suggest these if 'uncertainty_method' is MC_DROPOUT
                    if params.get("uncertainty_method") == UncertaintyMethod.MC_DROPOUT.value:
                        if config["type"] == "int":
                            params[param_name] = trial.suggest_int(param_name, config["low"], config["high"], step=config["step"])
                        elif config["type"] == "float":
                            params[param_name] = trial.suggest_float(param_name, config["low"], config["high"], step=config["step"])
                    elif param_name == "dropout_rate":  # If not MC_DROPOUT, set dropout rate to 0.0
                        params[param_name] = 0.0
                    continue  # Skip to next param after handling

                # Handle l1_ratio for SKLearnLogisticRegression's elasticnet
                if param_name == "l1_ratio" and model_name == ModelName.SKLEARN_LOGISTIC_REGRESSION:
                    if params.get("penalty") == "elasticnet":
                        params[param_name] = trial.suggest_float(param_name, config["low"], config["high"], step=config["step"])
                    continue  # Skip to next param

                if config["type"] == "int":
                    params[param_name] = trial.suggest_int(param_name, config["low"], config["high"], step=config.get("step", 1))
                elif config["type"] == "float":
                    params[param_name] = trial.suggest_float(param_name, config["low"], config["high"], log=config.get("log", False))
                elif config["type"] == "categorical":
                    params[param_name] = trial.suggest_categorical(param_name, config["choices"])

            return params, None

    def train(self, X: np.ndarray, y: np.ndarray, models_to_consider: List[ModelName] = None, save_metrics: bool = False):
        """
        Trains and optimizes selected models using cross-validation and Optuna.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
            models_to_consider (List[ModelName], optional): List of model names (enums) to train.
                                                            If None, all registered models are considered.
            save_metrics (bool): If True, save numerical and visual metrics for the final model.
        """
        logger.info(f"Starting AutoML training for {self.task_type.value} task with metric '{self.metric}'.")
        logger.info(
            f"Feature scaling: {'Enabled' if self.feature_scaler else 'Disabled'}, Target scaling (regression): {'Enabled' if self.target_scaler and self.task_type == TaskType.REGRESSION else 'Disabled'}"
        )

        # --- Feature Scaling ---
        if self.feature_scaler:
            X_scaled = self.feature_scaler.fit_transform(X)
            self._fitted_feature_scaler = self.feature_scaler
            logger.info("Features scaled using provided scaler.")
        else:
            X_scaled = X
            logger.info("No feature scaler provided. Features will not be scaled.")

        # Store a subset of X_scaled for SHAP background data
        sample_indices = np.random.choice(X.shape[0], min(200, X.shape[0]), replace=False)
        self.X_train_for_shap = X_scaled[sample_indices]

        # --- Target Scaling (for Regression only) ---
        y_for_training = y
        if self.task_type == TaskType.REGRESSION and self.target_scaler:
            y_reshaped = y.reshape(-1, 1) if y.ndim == 1 else y
            y_scaled = self.target_scaler.fit_transform(y_reshaped).flatten()  # Flatten back to 1D after scaling
            self._fitted_target_scaler = self.target_scaler
            y_for_training = y_scaled
            logger.info("Target variable scaled for regression.")
        elif self.task_type == TaskType.REGRESSION and not self.target_scaler:
            logger.info("No target scaler provided for regression. Target will not be scaled.")

        # Determine num_classes for classification tasks (needed for model instantiation)
        num_classes_for_instantiation = None
        if self.task_type == TaskType.CLASSIFICATION:
            # np.unique(y) returns sorted unique elements. Its shape gives the count.
            num_classes_for_instantiation = np.unique(y).shape[0]
            if num_classes_for_instantiation < 2:
                raise ValueError("Classification task requires at least 2 unique classes in y.")

        # Use only specified models or all registered models
        if models_to_consider is None:
            models_to_consider = list(self.models_registry.keys())

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        self.leaderboard = []  # Reset leaderboard for new training run

        for model_name in models_to_consider:  # model_name is now an Enum
            if model_name not in self.models_registry:
                logger.warning(f"Model '{model_name.value}' not found in registry. Skipping.")
                continue

            # Skip regression-only models if task type is classification (already filtered in __init__ but good to double check)
            if self.task_type == TaskType.CLASSIFICATION and model_name in [
                ModelName.JAX_LINEAR_REGRESSION,
                ModelName.PROBABILISTIC_REGRESSION,
                ModelName.CLASSIFIER_REGRESSION,
            ]:
                logger.info(f"Skipping {model_name.value} as it is designed for regression tasks for {self.task_type.value} task.")
                continue

            logger.info(f"\n--- Optimizing {model_name.value} ---")

            # Define the objective function for Optuna
            def objective(trial: optuna.Trial) -> float:
                current_model_params, base_classifier_name_str = self._sample_params_for_trial(trial, model_name)

                fold_metrics = []
                for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled, y_for_training)):  # Use scaled X and y_for_training
                    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                    y_train, y_val = y_for_training[train_idx], y_for_training[val_idx]

                    try:
                        # Instantiate the model with sampled parameters
                        model_instance = self._instantiate_model(model_name, current_model_params.copy(), X_train.shape[1], num_classes=num_classes_for_instantiation)

                        model_instance.fit(X_train, y_train)  # Fit on scaled/transformed data
                        predictions_scaled = model_instance.predict(X_val)

                        # --- Denormalize predictions for metric evaluation (only for regression) ---
                        if self.task_type == TaskType.REGRESSION and self.target_scaler:
                            predictions_original_scale = self.target_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
                            y_val_original_scale = y[val_idx]  # Compare against original unscaled y
                        else:
                            predictions_original_scale = predictions_scaled
                            y_val_original_scale = y_val  # For classification, y_val is already unscaled or discrete

                        y_proba = None
                        if self.task_type == TaskType.CLASSIFICATION and self.metric == "log_loss":
                            if hasattr(model_instance, "predict_proba"):
                                y_proba = model_instance.predict_proba(X_val)
                                # If predict_proba returns (N, 2) for binary, take the positive class ([:,1]) or handle correctly
                                if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                                    y_proba = y_proba[:, 1]
                            else:
                                logger.warning(f"Model {model_name.value} does not support predict_proba for 'log_loss'. Returning a bad score for trial.")
                                return float("nan")  # Signal failure to Optuna

                        metric_value = self._evaluate_metric(y_val_original_scale, predictions_original_scale, y_proba)

                        if np.isfinite(metric_value):
                            fold_metrics.append(metric_value)
                        else:
                            logger.warning(f"  Trial {trial.number}, Fold {fold}: Metric value is not finite ({metric_value}). Skipping fold.")
                            return float("nan")  # Signal failure to Optuna

                    except Exception as e:
                        logger.error(f"  Trial {trial.number}, Fold {fold}: Error training model {model_name.value} with params {current_model_params}: {e}")
                        return float("nan")  # Return NaN to indicate a failed trial

                if not fold_metrics:
                    logger.warning(f"  Trial {trial.number}: No finite metrics recorded for {model_name.value}. Returning NaN.")
                    return float("nan")

                return np.mean(fold_metrics)

            # Run Optuna optimization for the current model
            study_start_time = datetime.now()
            study = self.optuna_optimizer.optimize(objective_fn=objective)
            study_end_time = datetime.now()
            train_duration_sec = (study_end_time - study_start_time).total_seconds()

            logger.info(f"Finished optimization for {model_name.value}.")
            logger.info(f"  Best trial: {study.best_trial.value:.4f} with params: {study.best_params}")

            # --- Train the final model with best hyperparameters on the full (scaled) dataset ---
            logger.info(f"Training final {model_name.value} model on full dataset with best params.")

            # Reconstruct params for final instantiation (study.best_params is immutable)
            final_params = study.best_params.copy()
            final_model_instance = self._instantiate_model(model_name, final_params, X_scaled.shape[1], num_classes=num_classes_for_instantiation)

            final_model_instance.fit(X_scaled, y_for_training)  # Fit on scaled/transformed data

            # Evaluate final model on the full (scaled) training data for leaderboard entry
            y_pred_final_train_scaled = final_model_instance.predict(X_scaled)

            # Denormalize predictions for metric calculation
            if self.task_type == TaskType.REGRESSION and self.target_scaler:
                y_pred_final_train = self.target_scaler.inverse_transform(y_pred_final_train_scaled.reshape(-1, 1)).flatten()
                y_train_original = y  # Compare against original y
            else:
                y_pred_final_train = y_pred_final_train_scaled
                y_train_original = y_for_training  # For classification, y_for_training is the original scale

            y_proba_final_train = None
            if self.task_type == TaskType.CLASSIFICATION and self.metric == "log_loss":
                if hasattr(final_model_instance, "predict_proba"):
                    y_proba_final_train = final_model_instance.predict_proba(X_scaled)
                    if y_proba_final_train.ndim == 2 and y_proba_final_train.shape[1] == 2:
                        y_proba_final_train = y_proba_final_train[:, 1]

            train_metric_for_entry = self._evaluate_metric(y_train_original, y_pred_final_train, y_proba_final_train)

            # Store the best trained model and its information
            self.trained_models[model_name] = (final_model_instance, study.best_params, study.best_trial.value)

            # Update overall best model
            if (self._is_minimize_metric() and study.best_trial.value < self.best_overall_metric) or (
                not self._is_minimize_metric() and study.best_trial.value > self.best_overall_metric
            ):
                self.best_overall_metric = study.best_trial.value
                self.best_model_name = model_name

            # Add results to leaderboard
            self.leaderboard.append(
                {
                    "model_name": model_name.value,
                    "hyperparameters": json.dumps(study.best_params),  # Store as JSON string
                    "train_metric": train_metric_for_entry,
                    "validation_metric": study.best_trial.value,  # Validation score from Optuna CV
                    "train_time_sec": train_duration_sec,
                }
            )

        # Sort leaderboard
        self.leaderboard.sort(key=lambda x: x["validation_metric"], reverse=not self._is_minimize_metric())
        logger.info("\n--- AutoML Training Complete ---")
        if self.best_model_name:
            logger.info(f"Overall Best Model: {self.best_model_name.value} with {self.metric}: {self.best_overall_metric:.4f}")
            if save_metrics:
                self.evaluate(X, y, save_path=f"{self.best_model_name.value}_training_metrics")
        else:
            logger.info("No models were successfully trained or considered.")
        logger.info("\nLeaderboard:\n" + pd.DataFrame(self.leaderboard).to_string())

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions using the overall best trained model.
        Input X is expected in original scale, output predictions are in original scale.

        Args:
            X (np.ndarray): Feature matrix for prediction (original scale).

        Returns:
            np.ndarray: Predicted values (original scale).
        """
        if self.best_model_name is None:
            raise RuntimeError("No model has been trained yet. Call .train() first.")

        # Scale input features for prediction
        if self._fitted_feature_scaler:
            X_transformed = self._fitted_feature_scaler.transform(X)
        else:
            X_transformed = X

        best_model_instance, _, _ = self.trained_models[self.best_model_name]

        predictions_scaled = best_model_instance.predict(X_transformed)

        # Denormalize predictions if regression task
        if self.task_type == TaskType.REGRESSION and self._fitted_target_scaler:
            predictions_reshaped = predictions_scaled.reshape(-1, 1) if predictions_scaled.ndim == 1 else predictions_scaled
            predictions_original_scale = self._fitted_target_scaler.inverse_transform(predictions_reshaped).flatten()
            return predictions_original_scale
        else:
            return predictions_scaled  # Return as is for classification or no target scaler

    def evaluate(self, X: np.ndarray, y: np.ndarray, save_path: str = "metrics"):
        """
        Evaluates the best model on a given dataset and saves the metrics.

        Args:
            X (np.ndarray): Feature matrix for evaluation (original scale).
            y (np.ndarray): True labels for evaluation (original scale).
            save_path (str): Directory to save the metrics files.
        """
        y_pred = self.predict(X)
        y_proba = None

        # Check if the best model is a composite regression model with an internal classifier
        best_model_instance, _, _ = self.trained_models[self.best_model_name]

        if self.task_type == TaskType.CLASSIFICATION:
            if hasattr(best_model_instance, "predict_proba"):
                if self._fitted_feature_scaler:
                    X_transformed = self._fitted_feature_scaler.transform(X)
                else:
                    X_transformed = X
                y_proba = best_model_instance.predict_proba(X_transformed)

            metrics_calculator = Metrics(self.task_type, self.best_model_name.value, y, y_pred, y_proba)
            metrics_calculator.save_metrics(save_path)
        elif self.task_type == TaskType.REGRESSION and hasattr(best_model_instance, "is_composite_regression_model") and best_model_instance.is_composite_regression_model:
            # For composite regression models, also evaluate the internal classifier's performance
            # Need to pass original X and y to get the internal classifier's view
            y_pred_internal_clf, y_proba_internal_clf, y_true_discretized = best_model_instance.get_classifier_predictions(X, y)

            logger.info(f"Evaluating internal classifier of {self.best_model_name.value} for classification metrics.")
            internal_metrics_calculator = Metrics(
                TaskType.CLASSIFICATION, f"{self.best_model_name.value}_InternalClassifier", y_true_discretized, y_pred_internal_clf, y_proba_internal_clf
            )
            internal_metrics_calculator.save_metrics(f"{save_path}_internal_classifier")

            # Plot probability mappers if it's a ClassifierRegressionModel
            if isinstance(best_model_instance, ClassifierRegressionModel):
                best_model_instance.plot_probability_mappers(plot_path=f"{save_path}_probability_mappers.png")

            # Also save the main regression metrics
            metrics_calculator = Metrics(self.task_type, self.best_model_name.value, y, y_pred, y_proba)
            metrics_calculator.save_metrics(save_path)
        else:
            metrics_calculator = Metrics(self.task_type, self.best_model_name.value, y, y_pred, y_proba)
            metrics_calculator.save_metrics(save_path)

    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """
        Estimates uncertainty for predictions using the best trained regression model.
        Input X is expected in original scale, output uncertainty is in original scale.

        Args:
            X (np.ndarray): Feature matrix for uncertainty estimation (original scale).

        Returns:
            np.ndarray: Uncertainty estimates (e.g., standard deviation) for each prediction (original scale).
        """
        if self.best_model_name is None:
            raise RuntimeError("No model has been trained yet. Call .train() first.")

        if self.task_type == TaskType.CLASSIFICATION:
            raise ValueError("Uncertainty prediction is not supported for classification tasks directly by AutoML.")

        best_model_instance, _, _ = self.trained_models[self.best_model_name]

        # Ensure the model is a regression model and supports predict_uncertainty
        if not hasattr(best_model_instance, "predict_uncertainty") or not best_model_instance.is_regression_model:
            raise NotImplementedError(f"Uncertainty prediction not implemented or not a regression model for {type(best_model_instance).__name__}.")

        # Scale input features
        if self._fitted_feature_scaler:
            X_transformed = self._fitted_feature_scaler.transform(X)
        else:
            X_transformed = X

        # Get raw uncertainty from the model (in scaled target space)
        uncertainty_scaled = best_model_instance.predict_uncertainty(X_transformed)

        # Denormalize uncertainty if target scaler was used
        if self._fitted_target_scaler:
            # Uncertainty (std dev) is scaled by multiplying by the scale_ factor of the scaler
            scale_factor = self._fitted_target_scaler.scale_[0] if self._fitted_target_scaler.scale_.ndim > 1 else self._fitted_target_scaler.scale_
            uncertainty_original_scale = uncertainty_scaled * scale_factor
            return uncertainty_original_scale
        else:
            return uncertainty_scaled  # Return as is if no target scaler

    def get_feature_importance(self, X_test: np.ndarray, feature_names: List[str] = None) -> Union[Dict[str, float], Dict[str, str]]:
        """
        Calculates feature importance using SHAP for the best trained model.

        Args:
            X_test (np.ndarray): The dataset for which to compute feature importances (original scale).
                                 Typically a validation or test set.
            feature_names (List[str], optional): Names of the features. If None, uses generic names.

        Returns:
            Dict[str, float]: A dictionary of feature names and their mean absolute SHAP values, sorted.
                              Returns an error dictionary if SHAP calculation fails.
        """
        if self.best_model_name is None:
            raise RuntimeError("No model has been trained yet. Call .train() first.")
        # Prepare background data for SHAP Explainer (use a subset of training data)
        # It's important that X_train_for_shap is already scaled if _fitted_feature_scaler exists
        if self.X_train_for_shap is None:
            logger.warning("X_train was not stored. SHAP background data will be limited. Ensure X_train is passed to AutoML.train().")
            # Fallback to using a subset of test data for background, but transform it if a scaler is present
            X_background_subset_original_scale = X_test[: min(200, len(X_test))]
        else:
            X_background_subset_original_scale = self.X_train_for_shap[: min(200, len(self.X_train_for_shap))]

        # Scale X_background_subset and X_test_original for SHAP explanation
        # The FeatureExplainer expects scaled data if a scaler was used during training.
        if self._fitted_feature_scaler:
            X_background_scaled = self._fitted_feature_scaler.transform(X_background_subset_original_scale)
            X_test_scaled = self._fitted_feature_scaler.transform(X_test)
        else:
            X_background_scaled = X_background_subset_original_scale
            X_test_scaled = X_test

        best_model_instance, _, _ = self.trained_models[self.best_model_name]

        try:
            # Pass the device to FeatureExplainer if the model has one (e.g., PyTorch models)
            explainer_device = getattr(best_model_instance, "device", None)
            explainer = FeatureExplainer(model_instance=best_model_instance, X_background=X_background_scaled, feature_names=feature_names, device=explainer_device)

            shap_values = explainer.explain(X_test_scaled).values
            feature_importance_summary = explainer.get_feature_importance_summary(shap_values)

            # Normalize importances
            total_importance = sum(feature_importance_summary.values())
            if total_importance == 0:
                logger.warning("Total feature importance is zero. Cannot normalize.")
                normalized_importance = feature_importance_summary
            else:
                normalized_importance = {k: v / total_importance for k, v in feature_importance_summary.items()}

            # Sort in decreasing order
            sorted_normalized_importance = dict(sorted(normalized_importance.items(), key=lambda item: item[1], reverse=True))

            logger.info(f"\nFeature Importances (SHAP) for {self.best_model_name.value} (Normalized and Sorted):")
            for feature, importance in sorted_normalized_importance.items():
                logger.info(f"  {feature}: {importance:.4f}")

            return sorted_normalized_importance

        except Exception as e:
            logger.error(f"Error calculating SHAP feature importance for {self.best_model_name.value}: {e}")
            return {"error": f"Failed to calculate SHAP: {e}"}

    def save_feature_importance_to_csv(self, feature_importances: Dict[str, float], file_path: str = "feature_importance.csv"):
        """
        Saves the normalized and sorted feature importances to a CSV file.

        Args:
            feature_importances (Dict[str, float]): A dictionary of feature names and their
                                                    normalized importance scores.
            file_path (str): The path to the CSV file where the importances will be saved.
        """
        logger.info(f"\n--- Saving Feature Importances to {file_path} ---")
        try:
            df_importance = pd.DataFrame(list(feature_importances.items()), columns=["Feature", "Importance"])
            df_importance.to_csv(file_path, index=False)
            logger.info("Feature importances saved successfully to CSV.")
        except Exception as e:
            logger.error(f"Failed to save feature importances to CSV: {e}")

    def plot_feature_importance(self, feature_importances: Dict[str, float], plot_path: str = "feature_importance.png"):
        """
        Generates and saves a horizontal bar plot of the normalized and sorted feature importances.

        Args:
            feature_importances (Dict[str, float]): A dictionary of feature names and their
                                                    normalized importance scores.
            plot_path (str): The path to the file where the plot will be saved (e.g., 'feature_importance.png').
        """
        logger.info(f"\n--- Plotting Feature Importances to {plot_path} ---")
        try:
            df_importance = pd.DataFrame(list(feature_importances.items()), columns=["Feature", "Importance"])

            plt.figure(figsize=(10, max(6, len(df_importance) * 0.4)))  # Adjust figure size dynamically
            sns.barplot(x="Importance", y="Feature", data=df_importance, palette="viridis")
            plt.title(f"Feature Importance for {self.best_model_name.value} (Normalized SHAP Values)")
            plt.xlabel("Normalized Importance (Sum to 1)")
            plt.ylabel("Feature")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            logger.info("Feature importance plot saved successfully.")
        except Exception as e:
            logger.error(f"Failed to plot feature importances: {e}")

    def select_features_by_cumulative_importance(self, X_test: np.ndarray, threshold: float = 0.9, feature_names: List[str] = None) -> List[str]:
        """
        Selects top features based on their cumulative normalized SHAP importance.

        Args:
            X_test (np.ndarray): The test dataset (original scale, used for SHAP calculation).
            threshold (float): A float between 0 and 1. Features are selected until
                               their cumulative normalized importance is less than this threshold.
            feature_names (List[str], optional): Original names of the features.
                                                If None, generic names 'Feature_X' will be assumed.

        Returns:
            List[str]: A list of selected feature names.
        """
        if not (0.0 < threshold <= 1.0):
            raise ValueError("Threshold must be between 0 (exclusive) and 1 (inclusive).")

        logger.info(f"\n--- Performing Feature Selection by Cumulative SHAP Importance (Threshold: {threshold:.2f}) ---")

        # 1. Get normalized and sorted feature importances
        normalized_importances = self.get_feature_importance(X_test, feature_names)

        if "error" in normalized_importances:
            logger.error(f"Could not perform feature selection due to SHAP calculation error: {normalized_importances['error']}")
            return []

        if not normalized_importances:
            logger.warning("No feature importances found to perform selection.")
            return []

        # 2. Select features based on cumulative importance
        selected_features = []
        cumulative_importance = 0.0

        for feature, norm_importance in normalized_importances.items():
            # Add features as long as their addition does not exceed the threshold
            if cumulative_importance + norm_importance <= threshold:
                selected_features.append(feature)
                cumulative_importance += norm_importance
            else:
                break  # Stop when adding next feature would exceed the threshold

        logger.info(f"Selected {len(selected_features)} features with cumulative importance {cumulative_importance:.4f}:")
        for f in selected_features:
            logger.info(f"  - {f}")

        return selected_features

    def retrain_with_selected_features(
        self,
        X_full_train: np.ndarray,
        y_full_train: np.ndarray,
        X_full_test: np.ndarray,
        y_full_test: np.ndarray,
        feature_names: List[str] = None,
        shap_threshold: float = 0.95,
        save_feature_importance_metrics: bool = False,
    ):
        """
        Retrains the best model found by AutoML.train() using a subset of features
        selected by cumulative SHAP importance.

        Args:
            X_full_train (np.ndarray): The complete training feature set (original scale).
            y_full_train (np.ndarray): The complete training target set (original scale).
            X_full_test (np.ndarray): The complete test feature set (original scale).
            y_full_test (np.ndarray): The complete test target set (original scale).
            feature_names (List[str], optional): Original names of the features.
            shap_threshold (float): Cumulative SHAP importance threshold for feature selection.
            save_feature_importance_metrics (bool): Whther to save the feature importance metrics

        Returns:
            dict: A dictionary containing the retrained model instance, filtered and scaled test data,
                  new test metric, and names of selected features.
        """
        if self.best_model_name is None:
            logger.error("No best model found. Please run AutoML.train() first.")
            raise RuntimeError("No best model found. Please run AutoML.train() first.")

        logger.info("\n--- Retraining Best Model with Selected Features ---")

        selected_feature_names = self.select_features_by_cumulative_importance(X_full_test, threshold=shap_threshold, feature_names=feature_names)

        # Determine num_classes for classification tasks for instantiation during retraining
        num_classes_for_instantiation = None
        if self.task_type == TaskType.CLASSIFICATION:
            num_classes_for_instantiation = np.unique(y_full_train).shape[0]
            if num_classes_for_instantiation < 2:
                raise ValueError("Classification task requires at least 2 unique classes in y_full_train.")

        # Prepare filtered datasets based on selected features
        if not selected_feature_names:
            logger.warning("No features selected by cumulative importance. Retraining with all features.")
            selected_indices = list(range(X_full_train.shape[1]))
            filtered_feature_names = feature_names if feature_names else [f"Feature_{i}" for i in selected_indices]
            X_train_filtered_original_scale = X_full_train
            X_test_filtered_original_scale = X_full_test
        else:
            if feature_names is None:
                logger.warning("Feature names not provided for retraining. Assuming original column order for selection. It is highly recommended to provide feature_names.")
                original_feature_map = {f"Feature_{i}": i for i in range(X_full_train.shape[1])}
            else:
                original_feature_map = {name: i for i, name in enumerate(feature_names)}

            # Get indices corresponding to selected feature names
            selected_indices = [original_feature_map[name] for name in selected_feature_names]
            filtered_feature_names = [feature_names[i] for i in selected_indices]

            X_train_filtered_original_scale = X_full_train[:, selected_indices]
            X_test_filtered_original_scale = X_full_test[:, selected_indices]

        logger.info(f"Retraining {self.best_model_name.value} with {len(filtered_feature_names)} selected features: {filtered_feature_names}")

        # Apply feature scaling to the filtered datasets for model training
        if self._fitted_feature_scaler:
            # Use the *already fitted* scaler from the initial train() call
            X_train_filtered = self._fitted_feature_scaler.transform(X_train_filtered_original_scale)
            X_test_filtered = self._fitted_feature_scaler.transform(X_test_filtered_original_scale)
        else:
            X_train_filtered = X_train_filtered_original_scale
            X_test_filtered = X_test_filtered_original_scale

        # Apply target scaling for regression tasks
        y_full_train_scaled = y_full_train
        if self.task_type == TaskType.REGRESSION and self._fitted_target_scaler:
            y_full_train_scaled = self._fitted_target_scaler.transform(y_full_train.reshape(-1, 1)).flatten()

        # Retrieve best model class and previously learned hyperparameters
        best_model_class_type = self.models_registry[self.best_model_name]
        best_hyperparameters = self.trained_models[self.best_model_name][1].copy()  # Get a copy to modify

        # Adjust input_size in hyperparameters for neural network based models
        # This is crucial as the number of features changes after selection
        if self.best_model_name in [
            ModelName.PYTORCH_NEURAL_NETWORK,
            ModelName.FLEXIBLE_NEURAL_NETWORK,
            ModelName.PROBABILISTIC_REGRESSION,
        ]:
            best_hyperparameters["input_size"] = X_train_filtered.shape[1]
            # For nested models, also update their internal input_size if applicable
            if "base_classifier_params" in best_hyperparameters:
                best_hyperparameters["base_classifier_params"]["input_size"] = X_train_filtered.shape[1]

        # Instantiate the final model with best hyperparameters and adjusted input features
        # This part requires careful reconstruction of parameters for composite models
        if self.best_model_name == ModelName.CLASSIFIER_REGRESSION:
            final_n_classes = best_hyperparameters["n_classes"]
            final_base_classifier_name = best_hyperparameters["base_classifier_name"]
            final_base_classifier_class = self.models_registry[ModelName(final_base_classifier_name)]
            final_mapper_type = MapperType(best_hyperparameters["mapper_type"])

            # Reconstruct base_classifier_params by filtering from best_hyperparameters
            final_base_classifier_params = {k.replace("base__", ""): v for k, v in best_hyperparameters.items() if k.startswith("base__")}
            # Convert enum strings back to Enum values
            if "uncertainty_method" in final_base_classifier_params and isinstance(final_base_classifier_params["uncertainty_method"], str):
                final_base_classifier_params["uncertainty_method"] = UncertaintyMethod(final_base_classifier_params["uncertainty_method"])

            # Reconstruct mapper_params
            final_mapper_params = {}
            if final_mapper_type in [MapperType.LOOKUP_MEAN, MapperType.LOOKUP_MEDIAN]:
                final_mapper_params["n_partitions_min"] = best_hyperparameters.get("n_partitions_min_lookup")
                final_mapper_params["n_partitions_max"] = best_hyperparameters.get("n_partitions_max_lookup")
            elif final_mapper_type == MapperType.SPLINE:
                final_mapper_params["spline_k"] = best_hyperparameters.get("spline_k")
                final_mapper_params["spline_s"] = best_hyperparameters.get("spline_s")

            # Adjust parameters for internal classifier if it's PyTorchNN
            if ModelName(final_base_classifier_name) == ModelName.PYTORCH_NEURAL_NETWORK:
                final_base_classifier_params["input_size"] = X_train_filtered.shape[1]  # Crucial adjustment
                final_base_classifier_params["output_size"] = 1 if final_n_classes == 2 else final_n_classes
                final_base_classifier_params["task_type"] = TaskType.CLASSIFICATION
                if "activation" in final_base_classifier_params and isinstance(final_base_classifier_params["activation"], str):
                    final_base_classifier_params["activation"] = nn.ReLU if final_base_classifier_params["activation"] == "ReLU" else nn.Tanh

            # Adjust parameters for internal boosting classifiers if it's the base
            elif ModelName(final_base_classifier_name) == ModelName.XGBOOST:
                final_base_classifier_params["objective"] = "binary:logistic" if final_n_classes == 2 else "multi:softmax"
                final_base_classifier_params["eval_metric"] = "logloss"
            elif ModelName(final_base_classifier_name) == ModelName.LIGHTGBM:
                final_base_classifier_params["objective"] = "binary" if final_n_classes == 2 else "multiclass"
                final_base_classifier_params["metric"] = "binary_logloss"
            elif ModelName(final_base_classifier_name) == ModelName.CATBOOST:
                final_base_classifier_params["task_type"] = TaskType.CLASSIFICATION
                final_base_classifier_params["loss_function"] = "Logloss" if final_n_classes == 2 else "MultiClass"

            final_model_instance = best_model_class_type(
                n_classes=final_n_classes,
                base_classifier_class=final_base_classifier_class,
                base_classifier_params=final_base_classifier_params,
                mapper_type=final_mapper_type,
                mapper_params=final_mapper_params,
            )
        elif self.best_model_name == ModelName.PROBABILISTIC_REGRESSION:
            # Extract and reconstruct parameters for these composite models
            final_n_classes = best_hyperparameters["n_classes"]
            final_regression_strategy = RegressionStrategy(best_hyperparameters["regression_strategy"])
            final_uncertainty_method = UncertaintyMethod(best_hyperparameters["uncertainty_method"])

            # Reconstruct nested base_classifier_params
            final_base_classifier_params = {k.replace("base_classifier_params__", ""): v for k, v in best_hyperparameters.items() if k.startswith("base_classifier_params__")}
            if "activation" in final_base_classifier_params and isinstance(final_base_classifier_params["activation"], str):
                final_base_classifier_params["activation"] = nn.ReLU if final_base_classifier_params["activation"] == "ReLU" else nn.Tanh

            # Reconstruct nested regression_head_params
            final_regression_head_params = {k.replace("regression_head_params__", ""): v for k, v in best_hyperparameters.items() if k.startswith("regression_head_params__")}
            if "activation" in final_regression_head_params and isinstance(final_regression_head_params["activation"], str):
                final_regression_head_params["activation"] = nn.ReLU if final_regression_head_params["activation"] == "ReLU" else nn.Tanh

            final_model_instance = best_model_class_type(
                input_size=X_train_filtered.shape[1],  # Adjust input_size
                n_classes=final_n_classes,
                base_classifier_params=final_base_classifier_params,
                regression_head_params=final_regression_head_params,
                regression_strategy=final_regression_strategy,
                learning_rate=best_hyperparameters["learning_rate"],
                n_epochs=best_hyperparameters["n_epochs"],
                batch_size=best_hyperparameters["batch_size"],
                uncertainty_method=final_uncertainty_method,
                n_mc_dropout_samples=best_hyperparameters.get("n_mc_dropout_samples", 100),
                dropout_rate=best_hyperparameters.get("dropout_rate", 0.0),
                random_seed=best_hyperparameters.get("random_seed", 0),  # For JAX
            )
        elif self.best_model_name == ModelName.PYTORCH_NEURAL_NETWORK:
            # Reconstruct activation and uncertainty enum values from strings
            best_hyperparameters["activation"] = nn.ReLU if best_hyperparameters["activation"] == "ReLU" else nn.Tanh
            best_hyperparameters["uncertainty_method"] = UncertaintyMethod(best_hyperparameters["uncertainty_method"])
            final_model_instance = best_model_class_type(
                input_size=X_train_filtered.shape[1], task_type=self.task_type, num_classes=num_classes_for_instantiation, **best_hyperparameters
            )
        elif self.best_model_name == ModelName.FLEXIBLE_NEURAL_NETWORK:
            best_hyperparameters["activation"] = nn.ReLU if best_hyperparameters["activation"] == "ReLU" else nn.Tanh
            best_hyperparameters["uncertainty_method"] = UncertaintyMethod(best_hyperparameters["uncertainty_method"])
            final_model_instance = best_model_class_type(
                input_size=X_train_filtered.shape[1], task_type=self.task_type, num_classes=num_classes_for_instantiation, **best_hyperparameters
            )
        elif self.best_model_name == ModelName.XGBOOST:
            objective_str = "reg:squarederror"
            eval_metric_str = "rmse"
            if self.task_type == TaskType.CLASSIFICATION:
                objective_str = "binary:logistic" if y_full_train.ndim == 1 and np.unique(y_full_train).shape[0] == 2 else "multi:softmax"
                eval_metric_str = "logloss"
            final_model_instance = best_model_class_type(objective=objective_str, eval_metric=eval_metric_str, num_classes=num_classes_for_instantiation, **best_hyperparameters)
        elif self.best_model_name == ModelName.LIGHTGBM:
            objective_str = "regression"
            eval_metric_str = "rmse"
            if self.task_type == TaskType.CLASSIFICATION:
                objective_str = "binary" if y_full_train.ndim == 1 and np.unique(y_full_train).shape[0] == 2 else "multiclass"
                eval_metric_str = "binary_logloss" if y_full_train.ndim == 1 and np.unique(y_full_train).shape[0] == 2 else "multi_logloss"
            final_model_instance = best_model_class_type(objective=objective_str, eval_metric=eval_metric_str, num_classes=num_classes_for_instantiation, **best_hyperparameters)
        elif self.best_model_name == ModelName.CATBOOST:
            final_model_instance = best_model_class_type(task_type=self.task_type, num_classes=num_classes_for_instantiation, **best_hyperparameters)
        elif self.best_model_name == ModelName.SKLEARN_LOGISTIC_REGRESSION:
            if best_hyperparameters.get("penalty") == "elasticnet" and "l1_ratio" in best_hyperparameters:
                final_model_instance = best_model_class_type(num_classes=num_classes_for_instantiation, **best_hyperparameters)
            else:  # Remove l1_ratio if not elasticnet to avoid warnings
                params_copy = best_hyperparameters.copy()
                if "l1_ratio" in params_copy:
                    del params_copy["l1_ratio"]
                final_model_instance = best_model_class_type(num_classes=num_classes_for_instantiation, **params_copy)
        elif self.best_model_name == ModelName.JAX_LINEAR_REGRESSION:
            final_model_instance = best_model_class_type(
                num_classes=num_classes_for_instantiation, **best_hyperparameters
            )  # Added num_classes, though not used by JAXLinearRegression directly
        else:
            raise ValueError(f"Unsupported model for retraining: {self.best_model_name.value}")

        # 8. Retrain the model on the filtered and scaled training data
        final_model_instance.fit(X_train_filtered, y_full_train_scaled)

        # 9. Evaluate performance on filtered and scaled test set
        y_pred_retrained_scaled = final_model_instance.predict(X_test_filtered)

        # Denormalize predictions for metric evaluation
        if self.task_type == TaskType.REGRESSION and self._fitted_target_scaler:
            y_pred_retrained_reshaped = y_pred_retrained_scaled.reshape(-1, 1) if y_pred_retrained_scaled.ndim == 1 else y_pred_retrained_scaled
            y_pred_retrained = self._fitted_target_scaler.inverse_transform(y_pred_retrained_reshaped).flatten()
        else:
            y_pred_retrained = y_pred_retrained_scaled

        # Re-evaluate metric using original scale test targets and denormalized predictions
        retrained_metric_value = self._evaluate_metric(y_full_test, y_pred_retrained)

        logger.info(f"Retraining complete for {self.best_model_name.value} with selected features.")
        logger.info(f"New test {self.metric}: {retrained_metric_value:.4f}")

        if save_feature_importance_metrics:
            logger.info("Saving and plotting feature importances for the retrained model.")
            # Recalculate feature importance for the retrained model
            retrained_feature_importances = self.get_feature_importance(X_full_test[:, selected_indices], feature_names=filtered_feature_names)
            if "error" not in retrained_feature_importances:
                self.save_feature_importance_to_csv(retrained_feature_importances, file_path=f"{self.best_model_name.value}_retrained_feature_importance.csv")
                self.plot_feature_importance(retrained_feature_importances, plot_path=f"{self.best_model_name.value}_retrained_feature_importance.png")
            else:
                logger.error(f"Could not save/plot feature importances for retrained model due to: {retrained_feature_importances['error']}")

        return {
            "retrained_model_instance": final_model_instance,
            "X_test_filtered": X_test_filtered,  # This is the SCALED X_test_filtered (for direct use with retrained_model_instance)
            "y_pred_retrained": y_pred_retrained,  # This is the DENORMALIZED prediction
            "retrained_metric_value": retrained_metric_value,
            "selected_feature_names": filtered_feature_names,
        }

    def get_best_model_info(self) -> Dict[str, Any]:
        """
        Returns information about the best performing model.

        Returns:
            Dict[str, Any]: A dictionary containing the best model's name, instance,
                            best hyperparameters, and best metric score.
        """
        if self.best_model_name is None:
            return {"message": "No model has been trained yet."}

        model_instance, params, metric = self.trained_models[self.best_model_name]
        return {"name": self.best_model_name.value, "instance": model_instance, "hyperparameters": params, "metric_score": metric, "metric_name": self.metric}

    def save_leaderboard(self, file_path: str = "automl_leaderboard.json"):
        """
        Saves the AutoML leaderboard to a JSON file.

        Args:
            file_path (str): The path to the JSON file where the leaderboard will be saved.
        """
        logger.info(f"\n--- Saving AutoML Leaderboard to {file_path} ---")

        serializable_leaderboard = []
        for entry in self.leaderboard:
            serializable_entry = entry.copy()

            # Convert ModelName enum to string
            if isinstance(serializable_entry["model_name"], Enum):
                serializable_entry["model_name"] = serializable_entry["model_name"].value

            # Convert any enum values within hyperparameters to strings
            serializable_hyperparameters = {}
            for k, v in serializable_entry["hyperparameters"].items():
                if isinstance(v, Enum):
                    serializable_hyperparameters[k] = v.value
                elif isinstance(v, dict):  # Handle nested dictionaries (e.g., base_classifier_params)
                    nested_dict = {}
                    for nk, nv in v.items():
                        if isinstance(nv, Enum):
                            nested_dict[nk] = nv.value
                        elif isinstance(nv, type) and issubclass(nv, BaseModel):  # Handle ModelName class directly if stored
                            nested_dict[nk] = nv.__name__  # Store class name if a model class
                        else:
                            nested_dict[nk] = nv
                    serializable_hyperparameters[k] = nested_dict
                elif isinstance(v, type) and issubclass(v, BaseModel):  # Handle ModelName class directly if stored
                    serializable_hyperparameters[k] = v.__name__  # Store class name if a model class
                else:
                    serializable_hyperparameters[k] = v
            serializable_entry["hyperparameters"] = serializable_hyperparameters
            serializable_leaderboard.append(serializable_entry)

        try:
            with open(file_path, "w") as f:
                json.dump(serializable_leaderboard, f, indent=4)
            logger.info("Leaderboard saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save leaderboard: {e}")

    def export_model(self, model_instance: BaseModel, file_path: str):
        """
        Exports a trained model to a file using joblib.

        Args:
            model_instance (BaseModel): The trained model instance to export.
            file_path (str): The path where the model will be saved.
        """
        logger.info(f"\n--- Exporting model to {file_path} ---")
        try:
            # When exporting the AutoML object itself, its scalers are implicitly handled.
            # If exporting a sub-model (like PyTorchNeuralNetwork, etc.)
            # it might not carry the scalers with it.
            # For this context, model_instance will be the retrained BaseModel subclass.
            # We assume it has the necessary internal state.
            joblib.dump(model_instance, file_path)
            logger.info(f"Model '{model_instance.name}' successfully exported.")
        except Exception as e:
            logger.error(f"Failed to export model: {e}")
            raise

    @staticmethod
    def load_model(file_path: str) -> BaseModel:
        """
        Loads a trained model from a file.

        Args:
            file_path (str): The path to the saved model file.

        Returns:
            BaseModel: The loaded model instance.
        """
        logger.info(f"\n--- Loading model from {file_path} ---")
        try:
            model = joblib.load(file_path)
            if not isinstance(model, BaseModel):
                logger.warning("Loaded object is not an instance of BaseModel. Ensure it was saved correctly.")
            logger.info("Model loaded successfully.")
            return model
        except Exception as e:
            logger.error(f"Failed to load model from {file_path}: {e}")
            raise
