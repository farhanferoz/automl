"""XGBoost model wrapper for AutoML."""

from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

from ..utils.metrics import Metrics
from .base import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost model wrapper.

    Note: Methods in this class are intentionally named identically to those in BaseModel
    as they implement the BaseModel interface. This is not a redeclaration error.
    """

    def __init__(self, objective: str = "reg:squarederror", eval_metric: str = "rmse", **kwargs):
        """Initializes the XGBoostModel.

        Args:
            objective (str): The learning objective function.
            eval_metric (str): The evaluation metric.
            **kwargs: Additional keyword arguments for the XGBoost model.
        """
        super().__init__(**kwargs)
        self.objective = objective
        self.eval_metric = eval_metric
        self.model = None
        # Determine if it's a regression model based on objective
        self.is_regression_model = self.objective in ["reg:squarederror", "reg:logistic", "count:poisson", "reg:gamma", "reg:tweedie", "reg:quantile"]
        self._train_residual_std = 0.0  # For regression uncertainty
        self.num_iterations_used = 0

    @property
    def name(self) -> str:
        """Returns the name of the model."""
        return "XGBoost"

    def fit(self, X: np.ndarray, y: np.ndarray) -> int:
        """Fits the XGBoost model to the training data.

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
            self.params["early_stopping_rounds"] = self.early_stopping_rounds
        else:
            X_train, y_train = X, y

        # Handle classification objectives for XGBoost based on objective string
        if self.objective in ["binary:logistic", "multi:softmax", "multi:softprob"]:
            # For classification, use XGBClassifier
            self.model = xgb.XGBClassifier(objective=self.objective, eval_metric=self.eval_metric, **self.params)
        else:
            # For regression, use XGBRegressor
            self.model = xgb.XGBRegressor(objective=self.objective, eval_metric=self.eval_metric, **self.params)

        fit_params = {}
        if eval_set is not None:
            fit_params["eval_set"] = eval_set

        self.model.fit(X_train, y_train, **fit_params)

        if self.is_regression_model:
            y_pred_train = self.predict(X)
            _train_residual_std = np.std(y - y_pred_train)
            if np.isnan(_train_residual_std):
                self._train_residual_std = 0.0
            else:
                self._train_residual_std = _train_residual_std

        self.num_iterations_used = self.model.best_iteration if eval_set is not None and self.model.best_iteration is not None else self.model.n_estimators
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
        if self.objective in ["binary:logistic", "multi:softmax", "multi:softprob"]:
            # For binary classification, predict_proba returns probabilities [:, 1] for positive class
            # For multi-class, predict_proba returns (N, num_classes) array of probabilities
            if self.objective == "binary:logistic":
                return self.model.predict_proba(X)[:, 1]
            # For 'multi:softmax' or 'multi:softprob'
            # .predict() for multi:softmax returns class labels, predict_proba gives probabilities
            if self.objective == "multi:softmax":
                return self.model.predict(X)
            # multi:softprob
            return self.model.predict_proba(X)
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
            raise ValueError("predict_proba is not available for the current XGBoost configuration (likely regression).")
        return self.model.predict_proba(X)

    def get_hyperparameter_search_space(self) -> dict[str, Any]:
        """Defines the hyperparameter search space for XGBoost.

        Returns:
            Dict[str, Any]: A dictionary defining the hyperparameter search space.
        """
        return {
            "n_estimators": {"type": "int", "low": 50, "high": 200, "step": 50},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "max_depth": {"type": "int", "low": 3, "high": 9, "step": 2},
            "subsample": {"type": "float", "low": 0.6, "high": 1.0, "step": 0.1},
            "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0, "step": 0.1},
            "gamma": {"type": "float", "low": 0.0, "high": 0.2, "step": 0.05},
            "reg_alpha": {"type": "float", "low": 1e-6, "high": 1.0, "log": True},  # L1 regularization
            "reg_lambda": {"type": "float", "low": 1e-6, "high": 1.0, "log": True},  # L2 regularization
        }

    def get_internal_model(self):
        """Returns the raw underlying XGBoost model."""
        return self.model

    def get_num_parameters(self) -> int:
        """Returns the number of estimators in the XGBoost model.

        Returns:
            int: The number of estimators.
        """
        if self.model is None:
            return 0
        # For tree-based models, n_estimators (number of trees) is a reasonable proxy for complexity.
        return self.num_iterations_used + 1

    def get_classifier_predictions(self, X: np.ndarray, y_true_original: np.ndarray):
        """Not implemented for XGBoostModel.

        Raises:
            NotImplementedError: XGBoostModel is not a composite model.
        """
        raise NotImplementedError("XGBoostModel is not a composite model and does not have an internal classifier for separate prediction.")

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
