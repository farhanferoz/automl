"""LightGBM model wrapper for AutoML."""

from typing import Any

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split

from ..utils.metrics import Metrics
from .base import BaseModel


class LightGBMModel(BaseModel):
    """LightGBM model wrapper."""

    def __init__(self, objective: str = "regression", metric: str = "rmse", **kwargs):
        """Initializes the LightGBMModel.

        Args:
            objective (str): The learning objective function.
            metric (str): The evaluation metric.
            **kwargs: Additional keyword arguments for the LightGBM model.
        """
        super().__init__(**kwargs)
        self.objective = objective
        self.metric = metric
        self.model = None
        # Determine if it's a regression model based on objective
        self.is_regression_model = self.objective in ["regression", "regression_l1", "huber", "fair", "poisson", "quantile", "mape", "gamma", "tweedie"]
        self._train_residual_std = 0.0  # For regression uncertainty
        self.num_iterations_used = 0

    @property
    def name(self) -> str:
        """Returns the name of the model."""
        return "LightGBM"

    def fit(self, X: np.ndarray, y: np.ndarray) -> int:
        """Fits the LightGBM model to the training data.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.

        Returns:
            int: Number of iterations used for training.
        """
        # Split data for early stopping if enabled
        eval_set = None
        if self.early_stopping_rounds is not None and self.validation_fraction > 0:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.validation_fraction, random_state=42)
            eval_set = [(X_val, y_val)]
        else:
            X_train, y_train = X, y

        # Handle classification objectives for LightGBM based on objective string
        if self.objective in ["binary", "multiclass", "multiclassova"]:
            # For classification, use LGBMClassifier
            self.model = lgb.LGBMClassifier(objective=self.objective, metric=self.metric, **self.params)
        else:
            # For regression, use LGBMRegressor
            self.model = lgb.LGBMRegressor(objective=self.objective, metric=self.metric, **self.params)

        fit_params = {}
        if eval_set is not None:
            fit_params["eval_set"] = eval_set
            fit_params["callbacks"] = [lgb.early_stopping(self.early_stopping_rounds, verbose=False)]  # Suppress verbose output

        self.model.fit(X_train, y_train, **fit_params)

        if self.is_regression_model:
            y_pred_train = self.predict(X)
            _train_residual_std = np.std(y - y_pred_train)
            if np.isnan(_train_residual_std):
                self._train_residual_std = 0.0
            else:
                self._train_residual_std = _train_residual_std

        self.num_iterations_used = self.model._best_iteration if eval_set is not None and self.model._best_iteration is not None else self.model.n_estimators
        return self.num_iterations_used

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Makes predictions on new data.

        Args:
            X (np.ndarray): Feature matrix for prediction.

        Returns:
            np.ndarray: Predicted values.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if self.objective in ["binary", "multiclass", "multiclassova"]:
            # For binary classification, predict_proba returns probabilities of classes [:, 1] for positive class
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)

    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """Estimates uncertainty for predictions.

        Args:
            X (np.ndarray): Feature matrix for uncertainty estimation.

        Returns:
            np.ndarray: Uncertainty estimates (e.g., standard deviation).
        """
        if not self.is_regression_model:
            raise ValueError("predict_uncertainty is only available for regression models.")
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        # For simplicity, return a constant uncertainty based on training residuals
        return np.full(X.shape[0], self._train_residual_std)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts class probabilities for classification tasks.

        Args:
            X (np.ndarray): Features for probability prediction.

        Returns:
            np.ndarray: Predicted probabilities.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if not hasattr(self.model, "predict_proba"):
            raise ValueError("predict_proba is not available for the current LightGBM configuration (likely regression).")
        return self.model.predict_proba(X)

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

    def get_internal_model(self):
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

    def get_classifier_predictions(self, X: np.ndarray, y_true_original: np.ndarray):
        """Not implemented for LightGBMModel.

        Raises:
            NotImplementedError: LightGBMModel is not a composite model.
        """
        raise NotImplementedError("LightGBMModel is not a composite model and does not have an internal classifier for separate prediction.")

    def evaluate(self, X: np.ndarray, y: np.ndarray, save_path: str = "metrics") -> np.ndarray:
        """Evaluates the model on a given dataset and saves the metrics.

        Args:
            X (np.ndarray): Feature matrix for evaluation.
            y (np.ndarray): True labels for evaluation.
            save_path (str): Directory to save the metrics files.

        Returns:
            np.ndarray: The predictions made by the model.
        """
        y_pred = self.predict(X)
        y_proba = None
        task_type = "regression" if self.is_regression_model else "classification"
        if task_type == "classification":
            y_proba = self.predict_proba(X)
        metrics_calculator = Metrics(task_type, self.name, y, y_pred, y_proba)
        metrics_calculator.save_metrics(save_path)
        return y_pred
