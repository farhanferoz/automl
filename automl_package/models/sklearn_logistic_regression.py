from sklearn.linear_model import LogisticRegression
import numpy as np
from .base import BaseModel
from typing import Dict, Any


class SKLearnLogisticRegression(BaseModel):
    """
    Logistic Regression model using scikit-learn.
    Primarily for classification, but used internally by ClassifierRegression.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None
        self._is_regression_model = False  # Not a regression model

    @property
    def name(self) -> str:
        return "SKLearnLogisticRegression"

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model = LogisticRegression(**self.params)
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self.model.predict(X)

    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        if not self._is_regression_model:
            raise ValueError("predict_uncertainty is not available for classification models.")
        # This will never be reached because _is_regression_model is False
        return np.array([])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self.model.predict_proba(X)

    def get_hyperparameter_search_space(self) -> Dict[str, Any]:
        return {
            "C": {"type": "float", "low": 0.1, "high": 10.0, "log": True},
            "solver": {"type": "categorical", "choices": ["lbfgs", "liblinear"]},
            "max_iter": {"type": "int", "low": 100, "high": 500, "step": 100},
        }

    def get_internal_model(self):
        """
        Returns the raw underlying scikit-learn model.
        """
        return self.model
