import numpy as np
from catboost import CatBoostRegressor, CatBoostClassifier
from .base import BaseModel
from typing import Dict, Any
from ..enums import TaskType  # Import TaskType enum
from ..logger import logger  # Import logger


class CatBoostModel(BaseModel):
    """CatBoost model wrapper."""

    def __init__(self, task_type: TaskType = TaskType.REGRESSION, **kwargs):  # Use enum
        super().__init__(**kwargs)
        self.task_type = task_type
        self.model = None
        self._is_regression_model = task_type == TaskType.REGRESSION
        self._train_residual_std = 0.0  # For regression uncertainty

        # Default parameters to ensure basic functionality without warnings
        if self.task_type == TaskType.REGRESSION:
            self.params.setdefault("loss_function", "RMSE")
            self.params.setdefault("eval_metric", "RMSE")
        elif self.task_type == TaskType.CLASSIFICATION:
            self.params.setdefault("loss_function", "Logloss")
            self.params.setdefault("eval_metric", "Logloss")
            self.params.setdefault("verbose", 0)  # Suppress verbose output during training
            self.params.setdefault("random_seed", 42)  # Ensure reproducibility

    @property
    def name(self) -> str:
        return "CatBoost"

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.task_type == TaskType.REGRESSION:
            self.model = CatBoostRegressor(**self.params)
        elif self.task_type == TaskType.CLASSIFICATION:
            # Handle binary vs multi-class for CatBoost
            if np.unique(y).shape[0] > 2:
                self.params["loss_function"] = "MultiClass"
                self.params["eval_metric"] = "MultiClass"
            else:
                self.params["loss_function"] = "Logloss"
                self.params["eval_metric"] = "Logloss"
            self.model = CatBoostClassifier(**self.params)
        else:
            logger.error(f"task_type must be '{TaskType.REGRESSION.value}' or '{TaskType.CLASSIFICATION.value}'.")
            raise ValueError(f"task_type must be '{TaskType.REGRESSION.value}' or '{TaskType.CLASSIFICATION.value}'.")

        self.model.fit(X, y, verbose=False)  # Suppress verbose output during fit

        if self._is_regression_model:
            y_pred_train = self.predict(X)
            _train_residual_std = np.std(y - y_pred_train)
            if np.isnan(_train_residual_std):
                self._train_residual_std = 0.0
            else:
                self._train_residual_std = _train_residual_std

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        predictions = self.model.predict(X)
        if self.task_type == TaskType.CLASSIFICATION and predictions.ndim > 1:
            # For multi-class classification, predict returns probabilities if predict_type='RawFormulaVal' or 'Probability'
            # But for default predict, it returns class labels. Ensure it's 1D.
            return predictions.flatten()
        return predictions

    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        if not self._is_regression_model:
            raise ValueError("predict_uncertainty is only available for regression models.")
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        # For simplicity, return a constant uncertainty based on training residuals
        return np.full(X.shape[0], self._train_residual_std)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if self.task_type == TaskType.REGRESSION:
            raise ValueError("predict_proba is not available for regression tasks.")
        return self.model.predict_proba(X)

    def get_hyperparameter_search_space(self) -> Dict[str, Any]:
        # CatBoost hyperparameters
        return {
            "iterations": {"type": "int", "low": 50, "high": 200, "step": 50},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "depth": {"type": "int", "low": 3, "high": 8},
            "l2_leaf_reg": {"type": "float", "low": 1e-2, "high": 10.0, "log": True},  # L2 regularization
            "border_count": {"type": "int", "low": 32, "high": 255},  # For numerical features
        }

    def get_internal_model(self):
        """
        Returns the raw underlying CatBoost model.
        """
        return self.model

    def get_classifier_predictions(self, X: np.ndarray, y_true_original: np.ndarray):
        raise NotImplementedError("CatBoostModel is not a composite model and does not have an internal classifier for separate prediction.")
