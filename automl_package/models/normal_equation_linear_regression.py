import numpy as np
from typing import Dict, Any

from .base import BaseModel
from ..utils.metrics import Metrics


class NormalEquationLinearRegression(BaseModel):
    """
    Linear Regression model implemented using the Normal Equation (Ridge Regression).
    """

    def __init__(self, l2_lambda: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        if self.early_stopping_rounds is not None:
            raise ValueError("Early stopping is not applicable to NormalEquationLinearRegression as it is a direct solution method.")
        if "l1_lambda" in kwargs and kwargs["l1_lambda"] > 0:
            raise ValueError("L1 regularization (Lasso) is not supported by NormalEquationLinearRegression due to non-differentiability at zero. Use iterative solvers for L1 e.g. JAXLinearRegression.")
        self.l2_lambda = l2_lambda
        self.weights = None
        self.bias = None
        self.is_regression_model = True
        self._train_residual_std = 0.0

    @property
    def name(self) -> str:
        return "NormalEquationLinearRegression"

    def fit(self, X: np.ndarray, y: np.ndarray) -> int:
        # Add a bias (intercept) term to X
        X_augmented = np.hstack([X, np.ones((X.shape[0], 1))])

        # Calculate (X.T @ X + lambda * I)
        identity_matrix = np.identity(X_augmented.shape[1])
        # Don't regularize the bias term
        identity_matrix[-1, -1] = 0 
        
        A = X_augmented.T @ X_augmented + self.l2_lambda * identity_matrix
        b = X_augmented.T @ y

        # Solve for beta (weights and bias)
        beta = np.linalg.solve(A, b)

        self.weights = beta[:-1]
        self.bias = beta[-1]

        # Calculate residual standard deviation for uncertainty estimation
        y_pred_train = self.predict(X)
        self._train_residual_std = np.std(y - y_pred_train)
        if np.isnan(self._train_residual_std):
            self._train_residual_std = 0.0

        return 1  # Non-iterative model, so 1 iteration

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None or self.bias is None:
            raise RuntimeError("Model has not been fitted yet.")
        return X @ self.weights + self.bias

    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        if not self.is_regression_model:
            raise ValueError("predict_uncertainty is only available for regression models.")
        if self.weights is None or self.bias is None:
            raise RuntimeError("Model has not been fitted yet.")
        return np.full(X.shape[0], self._train_residual_std)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("NormalEquationLinearRegression is a regression model and does not support predict_proba.")

    def get_hyperparameter_search_space(self) -> Dict[str, Any]:
        return {
            "l2_lambda": {"type": "float", "low": 1e-6, "high": 1.0, "log": True},  # L2 regularization
        }

    def get_num_parameters(self) -> int:
        if self.weights is None:
            return 0
        return self.weights.size + 1  # weights + bias

    def get_classifier_predictions(self, X: np.ndarray, y_true_original: np.ndarray):
        raise NotImplementedError("NormalEquationLinearRegression is not a composite model and does not have an internal classifier for separate prediction.")

    def evaluate(self, X: np.ndarray, y: np.ndarray, save_path: str = "metrics") -> np.ndarray:
        y_pred = self.predict(X)
        metrics_calculator = Metrics("regression", self.name, y, y_pred)
        metrics_calculator.save_metrics(save_path)
        return y_pred
