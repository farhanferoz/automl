import numpy as np
from enums import Enum
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, accuracy_score, log_loss
from typing import Dict, Any, Type, Union, List, Tuple
import optuna  # Import optuna for type hinting in objective function
import json  # New import
from datetime import datetime  # New import
import joblib  # New import for model export/import

from .models.base import BaseModel
from .models.linear_regression import JAXLinearRegression
from .models.neural_network import PyTorchNeuralNetwork
from .models.xgboost_lgbm import XGBoostModel
from .models.sklearn_logistic_regression import SKLearnLogisticRegression
from .models.catboost_model import CatBoostModel
from .models.probabilistic_regression import ProbabilisticRegressionModel
from .models.jax_probabilistic_regression import JAXProbabilisticRegressionModel
from .models.classifier_regression import ClassifierRegressionModel
from .optimizers.optuna_optimizer import OptunaOptimizer
from .utils.probability_mapper import ClassProbabilityMapper
from .enums import UncertaintyMethod, RegressionStrategy, MapperType, TaskType, ModelName  # Import enums
from .logger import logger  # Import logger
from .explainers.feature_explainer import FeatureExplainer  # New import


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
    ):
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
        """
        self.task_type = task_type
        self.metric = metric
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.random_state = random_state

        # Use ModelName enum for registry keys
        self.models_registry: Dict[ModelName, Type[BaseModel]] = {
            ModelName.JAX_LINEAR_REGRESSION: JAXLinearRegression,
            ModelName.PYTORCH_NEURAL_NETWORK: PyTorchNeuralNetwork,
            ModelName.XGBOOST: XGBoostModel,
            # ModelName.LIGHTGBM: LightGBMModel,
            ModelName.SKLEARN_LOGISTIC_REGRESSION: SKLearnLogisticRegression,
            ModelName.CATBOOST: CatBoostModel,
            ModelName.CLASSIFIER_REGRESSION: ClassifierRegressionModel,
            ModelName.PROBABILISTIC_REGRESSION: ProbabilisticRegressionModel,
            ModelName.JAX_PROBABILISTIC_REGRESSION: JAXProbabilisticRegressionModel,
        }
        # Use ModelName enum for best_model_name storage
        self.trained_models: Dict[ModelName, Tuple[BaseModel, Dict[str, Any], float]] = {}
        self.best_model_name: ModelName = None
        self.best_overall_metric: float = float("inf") if self._is_minimize_metric() else -float("inf")

        # This will store the X_train data for SHAP background data when train is called.
        self.X_train_for_shap: np.ndarray = None
        self.leaderboard: List[Dict[str, Any]] = []  # New: Initialize leaderboard

        # Ensure regression-only models are only used for regression tasks
        if self.task_type == TaskType.CLASSIFICATION:
            if ModelName.CLASSIFIER_REGRESSION in self.models_registry:
                del self.models_registry[ModelName.CLASSIFIER_REGRESSION]
                logger.info(f"{ModelName.CLASSIFIER_REGRESSION.value} model is only supported for regression tasks and has been removed from consideration.")
            if ModelName.PROBABILISTIC_REGRESSION in self.models_registry:
                del self.models_registry[ModelName.PROBABILISTIC_REGRESSION]
                logger.info(f"{ModelName.PROBABILISTIC_REGRESSION.value} model is only supported for regression tasks and has been removed from consideration.")
            if ModelName.JAX_PROBABILISTIC_REGRESSION in self.models_registry:
                del self.models_registry[ModelName.JAX_PROBABILISTIC_REGRESSION]
                logger.info(f"{ModelName.JAX_PROBABILISTIC_REGRESSION.value} model is only supported for regression tasks and has been removed from consideration.")

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
                return accuracy_score(y_true, y_pred)
            elif self.metric == "log_loss":
                if y_proba is None:
                    raise ValueError("y_proba is required for 'log_loss' metric.")
                return log_loss(y_true, y_proba)
            else:
                raise ValueError(f"Unsupported classification metric: {self.metric}")
        else:
            raise ValueError("Task type must be 'regression' or 'classification'.")

    def train(self, X: np.ndarray, y: np.ndarray, models_to_consider: List[ModelName] = None):  # Use enum
        """
        Trains and optimizes selected models using cross-validation and Optuna.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
            models_to_consider (List[ModelName], optional): List of model names (enums) to train.
                                                      If None, all registered models are considered.
        """
        # Store X_train for later use by SHAP explainers
        self.X_train_for_shap = X

        if models_to_consider is None:
            models_to_consider = list(self.models_registry.keys())

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for model_name in models_to_consider:  # model_name is now an Enum
            if model_name not in self.models_registry:
                logger.warning(f"Model '{model_name.value}' not found in registry. Skipping.")
                continue

            # Skip regression-only models if task type is classification
            if self.task_type == TaskType.CLASSIFICATION and model_name in [
                ModelName.CLASSIFIER_REGRESSION,
                ModelName.PROBABILISTIC_REGRESSION,
                ModelName.JAX_PROBABILISTIC_REGRESSION,
            ]:
                logger.info(f"Skipping {model_name.value} as it is only for regression tasks.")
                continue

            logger.info(f"\n--- Optimizing {model_name.value} ---")
            model_class = self.models_registry[model_name]

            # Define the objective function for Optuna
            # This objective function dynamically samples hyperparameters based on the model_name
            def objective(trial: optuna.Trial) -> float:
                current_model_params = {}
                base_classifier_params = {}
                regression_head_params = {}
                base_classifier_name_str = None  # Store as string for Optuna suggestions
                base_classifier_class = None

                # --- Dynamic Hyperparameter Sampling based on model_name ---
                if model_name == ModelName.CLASSIFIER_REGRESSION:
                    # Parameters specific to ClassifierRegressionModel
                    n_classes = trial.suggest_int("n_classes", 2, 5)

                    # Decide which base classifier to use (Optuna samples strings)
                    base_classifier_choices_str = [
                        ModelName.PYTORCH_NEURAL_NETWORK.value,
                        ModelName.XGBOOST.value,
                        # ModelName.LIGHTGBM.value,
                        ModelName.SKLEARN_LOGISTIC_REGRESSION.value,
                        ModelName.CATBOOST.value,
                    ]
                    base_classifier_name_str = trial.suggest_categorical("base_classifier_name", base_classifier_choices_str)
                    base_classifier_class = self.models_registry[ModelName(base_classifier_name_str)]  # Convert back to Enum for class lookup

                    # Dynamically sample hyperparameters for the chosen base classifier
                    model_instance_for_space = (
                        base_classifier_class(task_type=TaskType.CLASSIFICATION)
                        if ModelName(base_classifier_name_str) in [ModelName.CATBOOST, ModelName.PYTORCH_NEURAL_NETWORK]
                        else base_classifier_class()
                    )
                    base_search_space = model_instance_for_space.get_hyperparameter_search_space()
                    for param_name, config in base_search_space.items():
                        param_key = f"base__{param_name}"
                        param_type = config["type"]
                        if param_type == "int":
                            if "step" in config:
                                base_classifier_params[param_name] = trial.suggest_int(param_key, config["low"], config["high"], step=config["step"])
                            else:
                                base_classifier_params[param_name] = trial.suggest_int(param_key, config["low"], config["high"])
                        elif param_type == "float":
                            if "log" in config and config["log"]:
                                base_classifier_params[param_name] = trial.suggest_float(param_key, config["low"], config["high"], log=True)
                            else:
                                base_classifier_params[param_name] = trial.suggest_float(param_key, config["low"], config["high"])
                        elif param_type == "categorical":
                            # For categorical parameters, ensure we get the actual enum member if applicable
                            if param_name == "uncertainty_method" and ModelName(base_classifier_name_str) == ModelName.PYTORCH_NEURAL_NETWORK:
                                base_classifier_params[param_name] = UncertaintyMethod(trial.suggest_categorical(param_key, [e.value for e in UncertaintyMethod]))
                            else:
                                base_classifier_params[param_name] = trial.suggest_categorical(param_key, config["choices"])

                        # Handle conditional params for PyTorchNeuralNetwork's uncertainty method
                        if ModelName(base_classifier_name_str) == ModelName.PYTORCH_NEURAL_NETWORK and param_name == "uncertainty_method":
                            if base_classifier_params[param_name] == UncertaintyMethod.MC_DROPOUT:
                                base_classifier_params["n_mc_dropout_samples"] = trial.suggest_int(f"{param_key}_n_mc_dropout_samples", 50, 200, step=50)
                                base_classifier_params["dropout_rate"] = trial.suggest_float(f"{param_key}_dropout_rate", 0.1, 0.5, step=0.1)

                    # Parameters for the probability mapper
                    mapper_type_val = trial.suggest_categorical("mapper_type", [e.value for e in MapperType])
                    mapper_type = MapperType(mapper_type_val)  # Convert back to enum
                    mapper_params = {}
                    if mapper_type in [MapperType.LOOKUP_MEAN, MapperType.LOOKUP_MEDIAN]:
                        mapper_params["n_partitions_min"] = trial.suggest_int("n_partitions_min_lookup", 5, 10)
                        mapper_params["n_partitions_max"] = trial.suggest_int("n_partitions_max_lookup", 10, 50)
                    elif mapper_type == MapperType.SPLINE:
                        mapper_params["spline_k"] = trial.suggest_int("spline_k", 1, 3)
                        mapper_params["spline_s"] = trial.suggest_float("spline_s", 0.01, 10.0, log=True)

                    current_model_params = {
                        "n_classes": n_classes,
                        "base_classifier_class": base_classifier_class,
                        "base_classifier_params": base_classifier_params,
                        "mapper_type": mapper_type,
                        "mapper_params": mapper_params,
                    }
                elif model_name in [ModelName.PROBABILISTIC_REGRESSION, ModelName.JAX_PROBABILISTIC_REGRESSION]:
                    n_classes = trial.suggest_int("n_classes", 2, 5)
                    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
                    n_epochs = trial.suggest_int("n_epochs", 10, 50, step=10)
                    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

                    regression_strategy_val = trial.suggest_categorical("regression_strategy", [e.value for e in RegressionStrategy])
                    regression_strategy = RegressionStrategy(regression_strategy_val)

                    # Base classifier params (for internal PyTorch/JAX NN)
                    base_classifier_params = {
                        "hidden_layers": trial.suggest_int("base_classifier_params__hidden_layers", 1, 2),
                        "hidden_size": trial.suggest_int("base_classifier_params__hidden_size", 32, 64, step=32),
                        "use_batch_norm": trial.suggest_categorical("base_classifier_params__use_batch_norm", [True, False]),
                    }
                    # Regression head params (for internal PyTorch/JAX NN heads)
                    regression_head_params = {
                        "hidden_layers": trial.suggest_int("regression_head_params__hidden_layers", 0, 1),
                        "hidden_size": trial.suggest_int("regression_head_params__hidden_size", 16, 32, step=16),
                        "use_batch_norm": trial.suggest_categorical("regression_head_params__use_batch_norm", [True, False]),
                    }

                    uncertainty_method_val = trial.suggest_categorical("uncertainty_method", [e.value for e in UncertaintyMethod])
                    uncertainty_method = UncertaintyMethod(uncertainty_method_val)

                    n_mc_dropout_samples = None
                    dropout_rate = None
                    if uncertainty_method == UncertaintyMethod.MC_DROPOUT:
                        n_mc_dropout_samples = trial.suggest_int("n_mc_dropout_samples", 50, 200, step=50)
                        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)

                    current_model_params = {
                        "n_classes": n_classes,
                        "base_classifier_class": PyTorchNeuralNetwork,  # PyTorch is fixed for ProbabilisticRegression
                        "base_classifier_params": base_classifier_params,
                        "regression_head_params": regression_head_params,
                        "regression_strategy": regression_strategy,
                        "learning_rate": learning_rate,
                        "n_epochs": n_epochs,
                        "batch_size": batch_size,
                        "uncertainty_method": uncertainty_method,
                        "n_mc_dropout_samples": n_mc_dropout_samples,
                        "dropout_rate": dropout_rate,
                    }
                    if model_name == ModelName.JAX_PROBABILISTIC_REGRESSION:
                        current_model_params["random_seed"] = trial.suggest_int("random_seed", 0, 100)
                        # JAX MLP also has dropout_rate for base classifier and regression head params.
                        # This needs to be sampled if MC_DROPOUT is chosen for the JAX model.
                        if uncertainty_method == UncertaintyMethod.MC_DROPOUT:
                            base_classifier_params["dropout_rate"] = trial.suggest_float("base_classifier_params__dropout_rate", 0.1, 0.5, step=0.1)
                            regression_head_params["dropout_rate"] = trial.suggest_float("regression_head_params__dropout_rate", 0.1, 0.5, step=0.1)
                        else:  # Set dropout to 0 if not MC_DROPOUT
                            base_classifier_params["dropout_rate"] = 0.0
                            regression_head_params["dropout_rate"] = 0.0

                else:
                    # For other standard models
                    model_instance_for_space = None
                    if model_name in [ModelName.CATBOOST, ModelName.PYTORCH_NEURAL_NETWORK]:
                        model_instance_for_space = model_class(task_type=self.task_type)
                    else:
                        model_instance_for_space = model_class()

                    search_space = model_instance_for_space.get_hyperparameter_search_space()

                    if model_name == ModelName.PYTORCH_NEURAL_NETWORK:  # Direct PyTorchNN model
                        uncertainty_method_val = trial.suggest_categorical("uncertainty_method", [e.value for e in UncertaintyMethod])
                        current_model_params["uncertainty_method"] = UncertaintyMethod(uncertainty_method_val)
                        if current_model_params["uncertainty_method"] == UncertaintyMethod.MC_DROPOUT:
                            current_model_params["n_mc_dropout_samples"] = trial.suggest_int("n_mc_dropout_samples", 50, 200, step=50)
                            current_model_params["dropout_rate"] = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)
                        # Add general params sampled from its search_space
                        for param_name, config in search_space.items():
                            if param_name not in ["uncertainty_method", "n_mc_dropout_samples", "dropout_rate"]:
                                param_type = config["type"]
                                if param_type == "int":
                                    if "step" in config:
                                        current_model_params[param_name] = trial.suggest_int(param_name, config["low"], config["high"], step=config["step"])
                                    else:
                                        current_model_params[param_name] = trial.suggest_int(param_name, config["low"], config["high"])
                                elif param_type == "float":
                                    if "log" in config and config["log"]:
                                        current_model_params[param_name] = trial.suggest_float(param_name, config["low"], config["high"], log=True)
                                    else:
                                        current_model_params[param_name] = trial.suggest_float(param_name, config["low"], config["high"])
                                elif param_type == "categorical":
                                    current_model_params[param_name] = trial.suggest_categorical(param_name, config["choices"])
                    else:  # For other models that don't have this complex uncertainty sampling logic
                        for param_name, config in search_space.items():
                            param_type = config["type"]
                            if param_type == "int":
                                if "step" in config:
                                    current_model_params[param_name] = trial.suggest_int(param_name, config["low"], config["high"], step=config["step"])
                                else:
                                    current_model_params[param_name] = trial.suggest_int(param_name, config["low"], config["high"])
                            elif param_type == "float":
                                if "log" in config and config["log"]:
                                    current_model_params[param_name] = trial.suggest_float(param_name, config["low"], config["high"], log=True)
                                else:
                                    current_model_params[param_name] = trial.suggest_float(param_name, config["low"], config["high"])
                            elif param_type == "categorical":
                                current_model_params[param_name] = trial.suggest_categorical(param_name, config["choices"])
                # --- End of Dynamic Hyperparameter Sampling ---

                fold_metrics = []
                for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]

                    model_instance = None
                    try:
                        # Instantiate models using enum values where appropriate
                        if model_name == ModelName.PYTORCH_NEURAL_NETWORK:
                            model_instance = model_class(input_size=X_train.shape[1], task_type=self.task_type, **current_model_params)
                        elif model_name == ModelName.XGBOOST and self.task_type == TaskType.CLASSIFICATION:
                            model_instance = model_class(
                                objective="binary:logistic" if y.ndim == 1 and np.unique(y).shape[0] == 2 else "multi:softmax", eval_metric="logloss", **current_model_params
                            )
                        # elif model_name == ModelName.LIGHTGBM and self.task_type == TaskType.CLASSIFICATION:
                        #     model_instance = model_class(
                        #         objective="binary" if y.ndim == 1 and np.unique(y).shape[0] == 2 else "multiclass",
                        #         metric="binary_logloss" if y.ndim == 1 and np.unique(y).shape[0] == 2 else "multi_logloss",
                        #         **current_model_params,
                        #     )
                        elif model_name == ModelName.CATBOOST:
                            model_instance = model_class(task_type=self.task_type, **current_model_params)
                        elif model_name == ModelName.CLASSIFIER_REGRESSION:
                            model_instance = model_class(
                                n_classes=current_model_params["n_classes"],
                                base_classifier_class=current_model_params["base_classifier_class"],
                                base_classifier_params=current_model_params["base_classifier_params"],
                                mapper_type=current_model_params["mapper_type"],
                                mapper_params=current_model_params["mapper_params"],
                            )
                        elif model_name == ModelName.PROBABILISTIC_REGRESSION:
                            model_instance = model_class(
                                input_size=X_train.shape[1],
                                n_classes=current_model_params["n_classes"],
                                base_classifier_class=current_model_params["base_classifier_class"],
                                base_classifier_params=current_model_params["base_classifier_params"],
                                regression_head_params=current_model_params["regression_head_params"],
                                regression_strategy=current_model_params["regression_strategy"],
                                learning_rate=current_model_params["learning_rate"],
                                n_epochs=current_model_params["n_epochs"],
                                batch_size=current_model_params["batch_size"],
                                uncertainty_method=current_model_params["uncertainty_method"],
                                n_mc_dropout_samples=current_model_params["n_mc_dropout_samples"],
                                dropout_rate=current_model_params["dropout_rate"],
                            )
                        elif model_name == ModelName.JAX_PROBABILISTIC_REGRESSION:
                            model_instance = model_class(
                                input_size=X_train.shape[1],
                                n_classes=current_model_params["n_classes"],
                                base_classifier_params=current_model_params["base_classifier_params"],
                                regression_head_params=current_model_params["regression_head_params"],
                                regression_strategy=current_model_params["regression_strategy"],
                                learning_rate=current_model_params["learning_rate"],
                                n_epochs=current_model_params["n_epochs"],
                                batch_size=current_model_params["batch_size"],
                                random_seed=current_model_params["random_seed"],
                                uncertainty_method=current_model_params["uncertainty_method"],
                                n_mc_dropout_samples=current_model_params["n_mc_dropout_samples"],
                                dropout_rate=current_model_params["dropout_rate"],
                            )
                        else:  # JAXLinearRegression, SKLearnLogisticRegression
                            model_instance = model_class(**current_model_params)

                        model_instance.fit(X_train, y_train)
                        y_pred = model_instance.predict(X_val)
                        y_proba = None

                        if self.task_type == TaskType.CLASSIFICATION and self.metric == "log_loss":
                            if hasattr(model_instance, "predict_proba"):
                                y_proba = model_instance.predict_proba(X_val)
                            else:
                                logger.warning(f"Model {model_name.value} does not support predict_proba for 'log_loss'. Returning a bad score.")
                                return float("inf")

                        metric_value = self._evaluate_metric(y_val, y_pred, y_proba)
                        fold_metrics.append(metric_value)
                    except Exception as e:
                        logger.error(f"Error during {model_name.value} training/prediction in fold {fold}: {e}")
                        return float("inf") if self._is_minimize_metric() else -float("inf")

                if not fold_metrics:
                    return float("inf") if self._is_minimize_metric() else -float("inf")
                return np.mean(fold_metrics)

            optimizer_direction = "minimize" if self._is_minimize_metric() else "maximize"
            optuna_optimizer = OptunaOptimizer(direction=optimizer_direction, n_trials=self.n_trials, seed=self.random_state)

            study = optuna_optimizer.optimize(objective_fn=objective)

            best_params = study.best_params
            best_metric_for_model = study.best_value

            logger.info(f"Best parameters for {model_name.value}: {best_params}")
            logger.info(f"Best {self.metric} for {model_name.value}: {best_metric_for_model:.4f}")

            # --- Train the final model with best hyperparameters on the full dataset ---
            # Instantiate the final model with the best hyperparameters
            final_model_instance = None
            if model_name == ModelName.CLASSIFIER_REGRESSION:
                final_n_classes = best_params["n_classes"]
                final_base_classifier_name = best_params["base_classifier_name"]
                final_base_classifier_class = self.models_registry[ModelName(final_base_classifier_name)]
                final_mapper_type = MapperType(best_params["mapper_type"])

                final_base_classifier_params = {k.replace("base__", ""): v for k, v in best_params.items() if k.startswith("base__")}
                if "uncertainty_method" in final_base_classifier_params and isinstance(final_base_classifier_params["uncertainty_method"], str):
                    final_base_classifier_params["uncertainty_method"] = UncertaintyMethod(final_base_classifier_params["uncertainty_method"])

                final_mapper_params = {}
                if final_mapper_type in [MapperType.LOOKUP_MEAN, MapperType.LOOKUP_MEDIAN]:
                    final_mapper_params["n_partitions_min"] = best_params.get("n_partitions_min_lookup")
                    final_mapper_params["n_partitions_max"] = best_params.get("n_partitions_max_lookup")
                elif final_mapper_type == MapperType.SPLINE:
                    final_mapper_params["spline_k"] = best_params.get("spline_k")
                    final_mapper_params["spline_s"] = best_params.get("spline_s")

                if final_base_classifier_name == ModelName.PYTORCH_NEURAL_NETWORK.value:
                    final_base_classifier_params["input_size"] = X.shape[1]
                    final_base_classifier_params["output_size"] = 1 if final_n_classes == 2 else final_n_classes
                    final_base_classifier_params["task_type"] = TaskType.CLASSIFICATION
                elif final_base_classifier_name == ModelName.XGBOOST.value:
                    final_base_classifier_params["objective"] = "binary:logistic" if final_n_classes == 2 else "multi:softmax"
                    final_base_classifier_params["eval_metric"] = "logloss"
                # elif final_base_classifier_name == ModelName.LIGHTGBM.value:
                #     final_base_classifier_params["objective"] = "binary" if final_n_classes == 2 else "multiclass"
                #     final_base_classifier_params["metric"] = "binary_logloss" if final_n_classes == 2 else "multi_logloss"
                elif final_base_classifier_name == ModelName.CATBOOST.value:
                    final_base_classifier_params["task_type"] = TaskType.CLASSIFICATION
                    if final_n_classes > 2:
                        final_base_classifier_params.setdefault("loss_function", "MultiClass")
                        final_base_classifier_params.setdefault("eval_metric", "MultiClass")
                    else:
                        final_base_classifier_params.setdefault("loss_function", "Logloss")
                        final_base_classifier_params.setdefault("eval_metric", "Logloss")

                final_model_instance = model_class(
                    n_classes=final_n_classes,
                    base_classifier_class=final_base_classifier_class,
                    base_classifier_params=final_base_classifier_params,
                    mapper_type=final_mapper_type,
                    mapper_params=final_mapper_params,
                )
            elif model_name == ModelName.PROBABILISTIC_REGRESSION:
                final_n_classes = best_params["n_classes"]
                final_learning_rate = best_params["learning_rate"]
                final_n_epochs = best_params["n_epochs"]
                final_batch_size = best_params["batch_size"]
                final_regression_strategy = RegressionStrategy(best_params["regression_strategy"])

                final_uncertainty_method = UncertaintyMethod(best_params.get("uncertainty_method", "constant"))
                final_n_mc_dropout_samples = best_params.get("n_mc_dropout_samples", 100)
                final_dropout_rate = best_params.get("dropout_rate", 0.1)

                final_base_classifier_params = {
                    "hidden_layers": best_params["base_classifier_params__hidden_layers"],
                    "hidden_size": best_params["base_classifier_params__hidden_size"],
                    "use_batch_norm": best_params["base_classifier_params__use_batch_norm"],
                }
                final_regression_head_params = {
                    "hidden_layers": best_params["regression_head_params__hidden_layers"],
                    "hidden_size": best_params["regression_head_params__hidden_size"],
                    "use_batch_norm": best_params["regression_head_params__use_batch_norm"],
                }

                final_model_instance = model_class(
                    input_size=X.shape[1],
                    n_classes=final_n_classes,
                    base_classifier_class=PyTorchNeuralNetwork,
                    base_classifier_params=final_base_classifier_params,
                    regression_head_params=final_regression_head_params,
                    regression_strategy=final_regression_strategy,
                    learning_rate=final_learning_rate,
                    n_epochs=final_n_epochs,
                    batch_size=final_batch_size,
                    uncertainty_method=final_uncertainty_method,
                    n_mc_dropout_samples=final_n_mc_dropout_samples,
                    dropout_rate=final_dropout_rate,
                )
            elif model_name == ModelName.JAX_PROBABILISTIC_REGRESSION:
                final_n_classes = best_params["n_classes"]
                final_learning_rate = best_params["learning_rate"]
                final_n_epochs = best_params["n_epochs"]
                final_batch_size = best_params["batch_size"]
                final_regression_strategy = RegressionStrategy(best_params["regression_strategy"])
                final_random_seed = best_params["random_seed"]

                final_uncertainty_method = UncertaintyMethod(best_params.get("uncertainty_method", "constant"))
                final_n_mc_dropout_samples = best_params.get("n_mc_dropout_samples", 100)
                final_dropout_rate = best_params.get("dropout_rate", 0.1)
                if final_uncertainty_method == UncertaintyMethod.MC_DROPOUT:
                    final_base_classifier_params = {
                        "hidden_layers": best_params["base_classifier_params__hidden_layers"],
                        "hidden_size": best_params["base_classifier_params__hidden_size"],
                        "use_batch_norm": best_params["base_classifier_params__use_batch_norm"],
                        "dropout_rate": best_params["base_classifier_params__dropout_rate"],
                    }
                    final_regression_head_params = {
                        "hidden_layers": best_params["regression_head_params__hidden_layers"],
                        "hidden_size": best_params["regression_head_params__hidden_size"],
                        "use_batch_norm": best_params["regression_head_params__use_batch_norm"],
                        "dropout_rate": best_params["regression_head_params__dropout_rate"],
                    }
                else:
                    final_base_classifier_params = {
                        "hidden_layers": best_params["base_classifier_params__hidden_layers"],
                        "hidden_size": best_params["base_classifier_params__hidden_size"],
                        "use_batch_norm": best_params["base_classifier_params__use_batch_norm"],
                        "dropout_rate": 0.0,
                    }
                    final_regression_head_params = {
                        "hidden_layers": best_params["regression_head_params__hidden_layers"],
                        "hidden_size": best_params["regression_head_params__hidden_size"],
                        "use_batch_norm": best_params["regression_head_params__use_batch_norm"],
                        "dropout_rate": 0.0,
                    }

                final_model_instance = model_class(
                    input_size=X.shape[1],
                    n_classes=final_n_classes,
                    base_classifier_params=final_base_classifier_params,
                    regression_head_params=final_regression_head_params,
                    regression_strategy=final_regression_strategy,
                    learning_rate=final_learning_rate,
                    n_epochs=final_n_epochs,
                    batch_size=final_batch_size,
                    random_seed=final_random_seed,
                    uncertainty_method=final_uncertainty_method,
                    n_mc_dropout_samples=final_n_mc_dropout_samples,
                    dropout_rate=final_dropout_rate,
                )
            elif model_name == ModelName.PYTORCH_NEURAL_NETWORK:
                best_params["uncertainty_method"] = UncertaintyMethod(best_params["uncertainty_method"])
                final_model_instance = model_class(input_size=X.shape[1], task_type=self.task_type, **best_params)
            elif model_name == ModelName.XGBOOST:
                objective_str = "reg:squarederror"
                eval_metric_str = "rmse"
                if self.task_type == TaskType.CLASSIFICATION:
                    objective_str = "binary:logistic" if y.ndim == 1 and np.unique(y).shape[0] == 2 else "multi:softmax"
                    eval_metric_str = "logloss"
                final_model_instance = model_class(objective=objective_str, eval_metric=eval_metric_str, **best_params)
            # elif model_name == ModelName.LIGHTGBM:
            #     objective_str = "regression"
            #     metric_str = "rmse"
            #     if self.task_type == TaskType.CLASSIFICATION:
            #         objective_str = "binary" if y.ndim == 1 and np.unique(y).shape[0] == 2 else "multiclass"
            #         metric_str = "binary_logloss" if y.ndim == 1 and np.unique(y).shape[0] == 2 else "multi_logloss"
            #     final_model_instance = model_class(objective=objective_str, metric=metric_str, **best_params)
            elif model_name == ModelName.CATBOOST:
                final_model_instance = model_class(task_type=self.task_type, **best_params)
            else:  # JAXLinearRegression, SKLearnLogisticRegression
                final_model_instance = model_class(**best_params)

            final_model_instance.fit(X, y)  # Train the final model on the full input data

            # Get the train metric for the final model instance
            y_pred_final_train = final_model_instance.predict(X)  # X is the full training data for this AutoML instance
            train_metric_for_entry = self._evaluate_metric(y, y_pred_final_train)

            # Add entry to leaderboard
            self.leaderboard.append(
                {
                    "model_name": model_name,  # Store ModelName enum directly
                    "hyperparameters": best_params,  # Store the best hyperparameters from Optuna
                    "train_metric": train_metric_for_entry,
                    "validation_metric": best_metric_for_model,  # This is the cross-validation score from Optuna
                    "timestamp": datetime.now().isoformat(),
                }
            )

            self.trained_models[model_name] = (final_model_instance, best_params, best_metric_for_model)

            if (self._is_minimize_metric() and best_metric_for_model < self.best_overall_metric) or (
                not self._is_minimize_metric() and best_metric_for_model > self.best_overall_metric
            ):
                self.best_overall_metric = best_metric_for_model
                self.best_model_name = model_name

        logger.info(f"\n--- AutoML Training Complete ---")
        if self.best_model_name:
            logger.info(f"Best overall model: {self.best_model_name.value} with {self.metric}: {self.best_overall_metric:.4f}")
        else:
            logger.info("No models were successfully trained or considered.")

    def get_feature_importance(self, X_test: np.ndarray, feature_names: List[str] = None) -> Dict[str, float]:
        """
        Calculates feature importance using SHAP for the best trained model.

        Args:
            X_test (np.ndarray): The dataset for which to compute feature importances.
            feature_names (List[str], optional): Names of the features. If None, uses generic names.

        Returns:
            Dict[str, float]: A dictionary of feature names and their mean absolute SHAP values, sorted.
        """
        if self.best_model_name is None:
            raise RuntimeError("No model has been trained yet. Call .train() first.")
        if self.X_train_for_shap is None:
            logger.warning("X_train was not stored. SHAP background data will be limited. Ensure X_train is passed to AutoML.train().")
            X_background_subset = X_test[: min(100, len(X_test))]  # Fallback to using test data for background
        else:
            X_background_subset = self.X_train_for_shap[: min(100, len(self.X_train_for_shap))]

        best_model_instance, _, _ = self.trained_models[self.best_model_name]

        try:
            explainer = FeatureExplainer(model_instance=best_model_instance, X_background=X_background_subset, feature_names=feature_names)

            shap_values = explainer.explain(X_test)
            feature_importance_summary = explainer.get_feature_importance_summary(shap_values)

            logger.info(f"\nFeature Importances (SHAP) for {self.best_model_name.value}:")
            for feature, importance in feature_importance_summary.items():
                logger.info(f"  {feature}: {importance:.4f}")

            return feature_importance_summary

        except Exception as e:
            logger.error(f"Error calculating SHAP feature importance for {self.best_model_name.value}: {e}")
            return {"error": f"Failed to calculate SHAP: {e}"}

    def select_features_by_cumulative_importance(self, X_test: np.ndarray, threshold: float = 0.9, feature_names: List[str] = None) -> List[str]:
        """
        Selects top features based on their cumulative normalized SHAP importance.

        Args:
            X_test (np.ndarray): The test dataset (used for SHAP calculation).
            threshold (float): A float between 0 and 1. Features are selected until
                               their cumulative normalized importance is less than this threshold.
            feature_names (List[str], optional): Original names of the features.

        Returns:
            List[str]: A list of selected feature names.
        """
        if not (0.0 < threshold <= 1.0):
            raise ValueError("Threshold must be between 0 (exclusive) and 1 (inclusive).")

        logger.info(f"\n--- Performing Feature Selection by Cumulative SHAP Importance (Threshold: {threshold:.2f}) ---")

        # 1. Get sorted feature importances (mean absolute SHAP values)
        raw_feature_importances = self.get_feature_importance(X_test, feature_names)

        if "error" in raw_feature_importances:
            logger.error(f"Could not perform feature selection due to SHAP calculation error: {raw_feature_importances['error']}")
            return []

        if not raw_feature_importances:
            logger.warning("No feature importances found to perform selection.")
            return []

        # Convert to a list of (feature_name, importance_value) tuples
        sorted_features_and_importances = list(raw_feature_importances.items())

        # Extract just the importance values and sum them
        importance_values = np.array([imp for _, imp in sorted_features_and_importances])
        total_importance = np.sum(importance_values)

        if total_importance == 0:
            logger.warning("Total feature importance is zero. Cannot normalize or select features.")
            return []

        # 2. Normalize importances
        normalized_importances = {name: imp / total_importance for name, imp in raw_feature_importances.items()}

        # Re-sort after normalization to ensure order for cumulative sum
        sorted_normalized_importances = dict(sorted(normalized_importances.items(), key=lambda item: item[1], reverse=True))

        # 3. Select features based on cumulative importance
        selected_features = []
        cumulative_importance = 0.0

        for feature, norm_importance in sorted_normalized_importances.items():
            # Only add if the current feature's importance, when added, does not exceed the threshold
            if cumulative_importance + norm_importance <= threshold:
                selected_features.append(feature)
                cumulative_importance += norm_importance
            else:
                break

        logger.info(f"Selected {len(selected_features)} features with cumulative importance {cumulative_importance:.4f}:")
        for f in selected_features:
            logger.info(f"  - {f}")

        return selected_features

    def retrain_with_selected_features(
        self, X_full_train: np.ndarray, y_full_train: np.ndarray, X_full_test: np.ndarray, y_full_test: np.ndarray, feature_names: List[str] = None, shap_threshold: float = 0.95
    ):
        """
        Retrains the best model found by AutoML.train() using a subset of features
        selected by cumulative SHAP importance.

        Args:
            X_full_train (np.ndarray): The complete training feature set.
            y_full_train (np.ndarray): The complete training target set.
            X_full_test (np.ndarray): The complete test feature set.
            y_full_test (np.ndarray): The complete test target set.
            feature_names (List[str], optional): Original names of the features.
            shap_threshold (float): Cumulative SHAP importance threshold for feature selection.

        Returns:
            dict: A dictionary containing the retrained model instance, filtered test data,
                  new test metric, and names of selected features.
        """
        if self.best_model_name is None:
            logger.error("No best model found. Please run AutoML.train() first.")
            raise RuntimeError("No best model found. Please run AutoML.train() first.")

        logger.info("\n--- Retraining Best Model with Selected Features ---")

        # 1. Get selected feature names based on cumulative SHAP importance
        selected_feature_names = self.select_features_by_cumulative_importance(X_full_test, threshold=shap_threshold, feature_names=feature_names)

        if not selected_feature_names:
            logger.warning("No features selected by cumulative importance. Retraining with all features.")
            # If no features are selected, default to using all features
            selected_indices = list(range(X_full_train.shape[1]))
            filtered_feature_names = feature_names if feature_names else [f"Feature_{i}" for i in selected_indices]
            X_train_filtered = X_full_train
            X_test_filtered = X_full_test
        else:
            # 2. Identify column indices of selected features
            if feature_names is None:
                # If original feature names not provided, infer from test data shape
                logger.warning("Feature names not provided for retraining. Assuming original column order for selection. It is highly recommended to provide feature_names.")
                original_feature_map = {f"Feature_{i}": i for i in range(X_full_train.shape[1])}
            else:
                original_feature_map = {name: i for i, name in enumerate(feature_names)}

            selected_indices = [original_feature_map[name] for name in selected_feature_names]
            filtered_feature_names = [feature_names[i] for i in selected_indices]

            # 3. Filter datasets
            X_train_filtered = X_full_train[:, selected_indices]
            X_test_filtered = X_full_test[:, selected_indices]

        logger.info(f"Retraining {self.best_model_name.value} with {len(selected_feature_names)} selected features: {filtered_feature_names}")

        # 4. Retrieve best model class and previously learned hyperparameters
        best_model_class_type = self.models_registry[self.best_model_name]
        best_hyperparameters = self.trained_models[self.best_model_name][1].copy()  # Get a copy to modify

        # 5. Adjust input_size in hyperparameters for neural network based models
        # This is crucial as the number of features changes after selection
        if self.best_model_name in [ModelName.PYTORCH_NEURAL_NETWORK, ModelName.PROBABILISTIC_REGRESSION, ModelName.JAX_PROBABILISTIC_REGRESSION]:
            best_hyperparameters["input_size"] = X_train_filtered.shape[1]

            # For ProbabilisticRegression and JAXProbabilisticRegression, also update internal classifier's input_size
            if "base_classifier_params" in best_hyperparameters:
                best_hyperparameters["base_classifier_params"]["input_size"] = X_train_filtered.shape[1]

        # 6. Create a new model instance with adjusted parameters
        final_model_instance = None
        if self.best_model_name == ModelName.CATBOOST:
            final_model_instance = best_model_class_type(task_type=self.task_type, **best_hyperparameters)
        elif self.best_model_name == ModelName.XGBOOST:
            objective_str = "reg:squarederror"
            eval_metric_str = "rmse"
            if self.task_type == TaskType.CLASSIFICATION:
                objective_str = "binary:logistic" if y_full_train.ndim == 1 and np.unique(y_full_train).shape[0] == 2 else "multi:softmax"
                eval_metric_str = "logloss"
            final_model_instance = best_model_class_type(objective=objective_str, eval_metric=eval_metric_str, **best_hyperparameters)
        # elif self.best_model_name == ModelName.LIGHTGBM:
        #     objective_str = "regression"
        #     metric_str = "rmse"
        #     if self.task_type == TaskType.CLASSIFICATION:
        #         objective_str = "binary" if y_full_train.ndim == 1 and np.unique(y_full_train).shape[0] == 2 else "multiclass"
        #         metric_str = "binary_logloss" if y_full_train.ndim == 1 and np.unique(y_full_train).shape[0] == 2 else "multi_logloss"
        #     final_model_instance = best_model_class_type(objective=objective_str, metric=metric_str, **best_hyperparameters)
        else:  # Generic instantiation for PyTorchNN, JAXNN, ClassifierRegression, JAXLinearRegression, SKLearnLogisticRegression
            final_model_instance = best_model_class_type(**best_hyperparameters)

        # 7. Retrain the model on the filtered training data
        final_model_instance.fit(X_train_filtered, y_full_train)

        # 8. Evaluate performance on filtered test set
        y_pred_retrained = final_model_instance.predict(X_test_filtered)
        retrained_metric_value = self._evaluate_metric(y_full_test, y_pred_retrained)

        logger.info(f"Retraining complete for {self.best_model_name.value} with selected features.")
        logger.info(f"New test {self.metric}: {retrained_metric_value:.4f}")

        return {
            "retrained_model_instance": final_model_instance,
            "X_test_filtered": X_test_filtered,
            "y_pred_retrained": y_pred_retrained,
            "retrained_metric_value": retrained_metric_value,
            "selected_feature_names": filtered_feature_names,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions using the best trained model.

        Args:
            X (np.ndarray): Feature matrix for prediction.

        Returns:
            np.ndarray: Predicted values from the best model.
        """
        if self.best_model_name is None:
            raise RuntimeError("No model has been trained yet. Call .train() first.")

        best_model_instance, _, _ = self.trained_models[self.best_model_name]
        return best_model_instance.predict(X)

    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """
        Estimates uncertainty for predictions using the best trained regression model.

        Args:
            X (np.ndarray): Feature matrix for uncertainty estimation.

        Returns:
            np.ndarray: Uncertainty estimates (e.g., standard deviation) for each prediction.
        """
        if self.best_model_name is None:
            raise RuntimeError("No model has been trained yet. Call .train() first.")

        best_model_instance, _, _ = self.trained_models[self.best_model_name]

        if not hasattr(best_model_instance, "_is_regression_model") or not best_model_instance._is_regression_model:
            raise ValueError(f"Model '{self.best_model_name.value}' is not a regression model or does not support uncertainty estimates.")

        return best_model_instance.predict_uncertainty(X)

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
