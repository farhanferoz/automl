from sklearn.linear_model import LogisticRegression
import numpy as np
from .base import BaseModel
from typing import Dict, Any


class SKLearnLogisticRegression(BaseModel):
    """
    Logistic Regression model using scikit-learn.
    Primarily for classification, but used internally by ClassifierRegression.
    Supports L1, L2, and ElasticNet regularization.
    """

    def __init__(self, penalty: str = "l2", C: float = 1.0, l1_ratio: float = None, **kwargs):
        super().__init__(**kwargs)
        self.penalty = penalty
        self.C = C  # Inverse of regularization strength
        self.l1_ratio = l1_ratio  # For elasticnet
        self.model = None
        self._is_regression_model = False  # Not a regression model

    @property
    def name(self) -> str:
        return "SKLearnLogisticRegression"

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.penalty == "elasticnet" and self.l1_ratio is None:
            raise ValueError("l1_ratio must be specified when penalty is 'elasticnet'.")

        # scikit-learn LogisticRegression has different solvers for different penalties
        solver = "lbfgs"  # Default solver, supports L2
        if self.penalty == "l1":
            solver = "liblinear"
        elif self.penalty == "elasticnet":
            solver = "saga"  # Supports ElasticNet

        # Pass regularization parameters appropriately
        if self.penalty == "elasticnet":
            self.model = LogisticRegression(penalty=self.penalty, C=self.C, solver=solver, l1_ratio=self.l1_ratio, **self.params)
        else:
            self.model = LogisticRegression(penalty=self.penalty, C=self.C, solver=solver, **self.params)

        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self.model.predict(X)

    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        if not self._is_regression_model:  # This condition is always True for this class
            # For classification, uncertainty is typically 1 - confidence (max probability)
            if self.model is None:
                raise RuntimeError("Model has not been fitted yet.")
            probabilities = self.predict_proba(X)
            # Find the max probability for each sample (confidence)
            max_probs = np.max(probabilities, axis=1)
            # Uncertainty is higher when confidence is lower (closer to 0.5 for binary)
            # Normalize to be between 0 and 1, where 1 is max uncertainty (prob=0.5)
            # 1 - 2 * abs(prob - 0.5) if proba is single value for positive class
            # Or simpler: 1 - max_prob for general classification
            return 1.0 - max_probs
        # This part of the code would technically not be reached given _is_regression_model = False
        # but it's good practice to ensure all abstract methods are fully implemented conceptually.
        return np.array([])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self.model.predict_proba(X)

    def get_hyperparameter_search_space(self) -> Dict[str, Any]:
        space = {
            "penalty": {"type": "categorical", "choices": ["l1", "l2", "elasticnet"]},
            "C": {"type": "float", "low": 1e-4, "high": 10.0, "log": True},  # Inverse of regularization strength
            "max_iter": {"type": "int", "low": 100, "high": 1000, "step": 100},
        }
        # l1_ratio is only relevant for 'elasticnet' penalty
        space["l1_ratio"] = {"type": "float", "low": 0.0, "high": 1.0, "step": 0.1}  # Conditional parameter
        return space

    def get_internal_model(self):
        """
        Returns the raw underlying scikit-learn model.
        """
        return self.model
