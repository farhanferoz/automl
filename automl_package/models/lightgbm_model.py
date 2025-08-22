"""LightGBM model wrapper for AutoML."""

from typing import Any, NoReturn

import lightgbm as lgb
import numpy as np

from automl_package.enums import TaskType
from automl_package.models.base import BaseModel
from automl_package.utils.data_handler import create_train_val_split
from automl_package.utils.metrics import Metrics


class LightGBMModel(BaseModel):
    """LightGBM model wrapper."""

    def __init__(self, task_type: TaskType = TaskType.REGRESSION, random_seed: int | None = None, **kwargs: Any) -> None:
        """Initializes the LightGBMModel.

        Args:
            task_type (TaskType): The type of machine learning task (regression or classification).
            random_seed (int, optional): Random seed for reproducibility.
            **kwargs: Additional keyword arguments for the LightGBM model.
        """
        super().__init__(**kwargs)
        self.task_type = task_type
        self.random_seed = random_seed
        self.model = None
        self.is_regression_model = task_type == TaskType.REGRESSION
        self._train_residual_std = 0.0  # For regression uncertainty
        self.num_iterations_used = 0

        self.params.setdefault("verbose", -1)  # Suppress verbose output during training
        if self.task_type == TaskType.REGRESSION:
            self.objective = "regression"
            self.metric = "rmse"
        elif self.task_type == TaskType.CLASSIFICATION:
            self.objective = "binary"  # Default for binary classification
            self.metric = "binary_logloss"
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    @property
    def name(self) -> str:
        """Returns the name of the model."""
        return "LightGBMModel"

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fits the LightGBM model to the training data.

        Args:
            x (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
        """
        # Split data for early stopping if enabled
        eval_set = None
        if self.early_stopping_rounds is not None and self.validation_fraction > 0:
            self.train_indices, self.val_indices = create_train_val_split(x, y, self.validation_fraction, self.random_seed)
            x_train, x_val = x[self.train_indices], x[self.val_indices]
            y_train, y_val = y[self.train_indices], y[self.val_indices]
            eval_set = [(x_val, y_val)]
        else:
            x_train, y_train = x, y
            self.train_indices = np.arange(x.shape[0])
            self.val_indices = None

        if self.task_type == TaskType.CLASSIFICATION:
            # For multi-class classification, adjust objective and metric
            if np.unique(y_train).shape[0] > 2:
                self.objective = "multiclass"
                self.metric = "multi_logloss"
            else:
                self.objective = "binary"
                self.metric = "binary_logloss"
            self.model = lgb.LGBMClassifier(objective=self.objective, metric=self.metric, **self.params)
        else:
            # For regression, use LGBMRegressor
            self.model = lgb.LGBMRegressor(objective=self.objective, metric=self.metric, **self.params)

        fit_params = {}
        if eval_set is not None:
            fit_params["eval_set"] = eval_set
            fit_params["callbacks"] = [lgb.early_stopping(self.early_stopping_rounds, verbose=False)]  # Suppress verbose output

        self.model.fit(x_train, y_train, **fit_params)

        if self.is_regression_model:
            y_pred_train = self.predict(x)
            _train_residual_std = np.std(y - y_pred_train)
            if np.isnan(_train_residual_std):
                self._train_residual_std = 0.0
            else:
                self._train_residual_std = _train_residual_std

        self.num_iterations_used = self.model._best_iteration if eval_set is not None and self.model._best_iteration is not None else self.model.n_estimators

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Makes predictions on new data.

        Args:
            x (np.ndarray): Feature matrix for prediction.

        Returns:
            np.ndarray: Predicted values.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if self.task_type == TaskType.CLASSIFICATION:
            # For binary classification, predict_proba returns probabilities of classes [:, 1] for positive class
            return self.model.predict_proba(x)[:, 1]
        return self.model.predict(x)

    def predict_uncertainty(self, x: np.ndarray) -> np.ndarray:
        """Estimates uncertainty for predictions.

        Args:
            x (np.ndarray): Feature matrix for uncertainty estimation.

        Returns:
            np.ndarray: Uncertainty estimates (e.g., standard deviation).
        """
        if not self.is_regression_model:
            raise ValueError("predict_uncertainty is only available for regression models.")
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        # For simplicity, return a constant uncertainty based on training residuals
        return np.full(x.shape[0], self._train_residual_std)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predicts class probabilities for classification tasks.

        Args:
            x (np.ndarray): Features for probability prediction.

        Returns:
            np.ndarray: Predicted probabilities.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if not hasattr(self.model, "predict_proba"):
            raise ValueError("predict_proba is not available for the current LightGBM configuration (likely regression).")
        return self.model.predict_proba(x)

    def get_hyperparameter_search_space(self) -> dict[str, Any]:
        """Defines the hyperparameter search space for LightGBM.

        Returns:
            Dict[str, Any]: A dictionary defining the hyperparameter search space.
        """
        return {
            "n_estimators": {"type": "int", "low": 50, "high": 200, "step": 50},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "num_leaves": {"type": "int", "low": 20, "high": 60, "step": 10},
            "max_depth": {"type": "int", "low": 5, "high": 15, "step": 2},
            "min_child_samples": {"type": "int", "low": 20, "high": 40, "step": 5},
            "subsample": {"type": "float", "low": 0.6, "high": 1.0, "step": 0.1},
            "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0, "step": 0.1},
            "reg_alpha": {"type": "float", "low": 1e-6, "high": 1.0, "log": True},  # L1 regularization
            "reg_lambda": {"type": "float", "low": 1e-6, "high": 1.0, "log": True},  # L2 regularization
        }

    def get_internal_model(self) -> Any:
        """Returns the raw underlying LightGBM model."""
        return self.model

    def get_num_parameters(self) -> int:
        """Returns the number of estimators in the LightGBM model.

        Returns:
            int: The number of estimators.
        """
        if self.model is None:
            return 0
        return self.num_iterations_used + 1

    def get_classifier_predictions(self, x: np.ndarray, y_true_original: np.ndarray) -> NoReturn:
        """Not implemented for LightGBMModel.

        Raises:
            NotImplementedError: LightGBMModel is not a composite model.
        """
        raise NotImplementedError("LightGBMModel is not a composite model and does not have an internal classifier for separate prediction.")

    def evaluate(self, x: np.ndarray, y: np.ndarray, save_path: str = "metrics") -> np.ndarray:
        """Evaluates the model on a given dataset and saves the metrics.

        Args:
            x (np.ndarray): Feature matrix for evaluation.
            y (np.ndarray): True labels for evaluation.
            save_path (str): Directory to save the metrics files.

        Returns:
            np.ndarray: The predictions made by the model.
        """
        y_pred = self.predict(x)
        y_proba = None
        task_type = "regression" if self.is_regression_model else "classification"
        if task_type == "classification":
            y_proba = self.predict_proba(x)
        metrics_calculator = Metrics(task_type=task_type, model_name=self.name, x_data=x, y_true=y, y_pred=y_pred, y_proba=y_proba)
        metrics_calculator.save_metrics(save_path)
        return y_pred
