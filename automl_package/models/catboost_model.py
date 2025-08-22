"""CatBoost model wrapper for AutoML."""

from typing import Any, Never

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

from automl_package.enums import TaskType  # Import TaskType enum
from automl_package.logger import logger  # Import logger
from automl_package.models.base import BaseModel
from automl_package.utils.data_handler import create_train_val_split
from automl_package.utils.metrics import Metrics


class CatBoostModel(BaseModel):
    """CatBoost model wrapper."""

    def __init__(self, task_type: TaskType = TaskType.REGRESSION, random_seed: int | None = None, **kwargs: Any) -> None:
        """Initializes the CatBoostModel.

        Args:
            task_type (TaskType): The type of machine learning task (regression or classification).
            random_seed (int, optional): Random seed for reproducibility.
            **kwargs: Additional keyword arguments for the CatBoost model.
        """  # Use enum
        super().__init__(**kwargs)
        self.task_type = task_type
        self.random_seed = random_seed
        self.model: CatBoostRegressor | CatBoostClassifier | None = None
        self.is_regression_model = task_type == TaskType.REGRESSION
        self._train_residual_std = 0.0  # For regression uncertainty
        self.num_iterations_used = 0

        # Default parameters to ensure basic functionality without warnings
        self.params.setdefault("verbose", 0)  # Suppress verbose output during training
        if self.random_seed is not None:
            self.params.setdefault("random_seed", self.random_seed)  # Ensure reproducibility
        if self.task_type == TaskType.REGRESSION:
            self.params.setdefault("loss_function", "RMSE")
            self.params.setdefault("eval_metric", "RMSE")
        elif self.task_type == TaskType.CLASSIFICATION:
            self.params.setdefault("loss_function", "Logloss")
            self.params.setdefault("eval_metric", "Logloss")

    @property
    def name(self) -> str:
        """Returns the name of the model."""
        return "CatBoostModel"

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fits the CatBoost model to the training data.

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
            eval_set = Pool(x_val, y_val)
        else:
            x_train, y_train = x, y
            self.train_indices = np.arange(x.shape[0])
            self.val_indices = None

        if self.task_type == TaskType.REGRESSION:
            self.model = CatBoostRegressor(**self.params)
        elif self.task_type == TaskType.CLASSIFICATION:
            # Handle binary vs multi-class for CatBoost
            if np.unique(y_train).shape[0] > 2:
                self.params["loss_function"] = "MultiClass"
                self.params["eval_metric"] = "MultiClass"
            else:
                self.params["loss_function"] = "Logloss"
                self.params["eval_metric"] = "Logloss"
            self.model = CatBoostClassifier(**self.params)
        else:
            logger.error(f"task_type must be '{TaskType.REGRESSION.value}' or '{TaskType.CLASSIFICATION.value}'.")
            raise ValueError(f"task_type must be '{TaskType.REGRESSION.value}' or '{TaskType.CLASSIFICATION.value}'.")

        fit_params: dict[str, Any] = {}
        if eval_set is not None:
            fit_params["eval_set"] = eval_set
            fit_params["early_stopping_rounds"] = self.early_stopping_rounds

        self.model.fit(x_train, y_train, **fit_params)

        if self.is_regression_model:
            y_pred_train = self.predict(x)
            _train_residual_std = np.std(y - y_pred_train)
            if np.isnan(_train_residual_std):
                self._train_residual_std = 0.0
            else:
                self._train_residual_std = _train_residual_std

        self.num_iterations_used = self.model.tree_count_ if eval_set is not None and self.model.tree_count_ is not None else self.params.get("iterations", 100)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Makes predictions on new data.

        Args:
            x (np.ndarray): Feature matrix for prediction.

        Returns:
            np.ndarray: Predicted values.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        predictions = self.model.predict(x)
        if self.task_type == TaskType.CLASSIFICATION and predictions.ndim > 1:
            # For multi-class classification, predict returns probabilities if predict_type='RawFormulaVal' or 'Probability'
            # But for default predict, it returns class labels. Ensure it's 1D.
            return predictions.flatten()
        return predictions

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
        if self.task_type == TaskType.REGRESSION:
            raise ValueError("predict_proba is not available for regression tasks.")
        return self.model.predict_proba(x)

    def get_hyperparameter_search_space(self) -> dict[str, Any]:
        """Defines the hyperparameter search space for CatBoost.

        Returns:
            Dict[str, Any]: A dictionary defining the hyperparameter search space.
        """
        # CatBoost hyperparameters
        return {
            "iterations": {"type": "int", "low": 50, "high": 200, "step": 50},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "depth": {"type": "int", "low": 3, "high": 8},
            "l2_leaf_reg": {"type": "float", "low": 1e-2, "high": 10.0, "log": True},  # L2 regularization
            "border_count": {"type": "int", "low": 32, "high": 255},  # For numerical features
            "early_stopping_rounds": {"type": "int", "low": 5, "high": 15, "step": 5},
        }

    def get_internal_model(self) -> CatBoostRegressor | CatBoostClassifier | None:
        """Returns the raw underlying CatBoost model."""
        return self.model

    def get_num_parameters(self) -> int:
        """Returns the number of trees in the CatBoost model.

        Returns:
            int: The number of trees.
        """
        if self.model is None:
            return 0
        # For CatBoost, tree_count_ attribute gives the number of trees (estimators)
        # which is a good proxy for the number of parameters in tree-based models.
        return self.model.tree_count_

    def get_classifier_predictions(self, x: np.ndarray, y_true_original: np.ndarray) -> Never:
        """Not implemented for CatBoostModel.

        Raises:
            NotImplementedError: CatBoostModel is not a composite model.
        """
        raise NotImplementedError("CatBoostModel is not a composite model and does not have an internal classifier for separate prediction.")

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
        if self.task_type == TaskType.CLASSIFICATION:
            y_proba = self.predict_proba(x)
        metrics_calculator = Metrics(task_type=self.task_type.value, model_name=self.name, x_data=x, y_true=y, y_pred=y_pred, y_proba=y_proba)
        metrics_calculator.save_metrics(save_path)
        return y_pred
