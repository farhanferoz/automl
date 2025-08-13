"""Linear Regression model using the Normal Equation."""

from typing import Any, Never

import numpy as np

from automl_package.utils.metrics import Metrics

from .base import BaseModel


class NormalEquationLinearRegression(BaseModel):
    """Linear Regression model implemented using the Normal Equation (Ridge Regression)."""

    def __init__(self, l2_lambda: float = 0.0, **kwargs: Any) -> None:
        """Initializes the NormalEquationLinearRegression model.

        Args:
            l2_lambda (float): L2 regularization strength.
            **kwargs: Additional keyword arguments for the BaseModel.
        """
        super().__init__(**kwargs)
        if self.early_stopping_rounds is not None:
            raise ValueError("Early stopping is not applicable to NormalEquationLinearRegression as it is a direct solution method.")
        if "l1_lambda" in kwargs and kwargs["l1_lambda"] > 0:
            raise ValueError(
                "L1 regularization (Lasso) is not supported by NormalEquationLinearRegression due to non-differentiability at zero. "
                "Use iterative solvers for L1 e.g. JAXLinearRegression."
            )
        self.l2_lambda = l2_lambda
        self.weights: np.ndarray | None = None
        self.bias: np.ndarray | None = None
        self.is_regression_model = True
        self._train_residual_std = 0.0

    @property
    def name(self) -> str:
        """Returns the name of the model."""
        return "NormalEquationLinearRegression"

    def fit(self, x: np.ndarray, y: np.ndarray) -> int:
        """Fits the NormalEquationLinearRegression model to the training data.

        Args:
            x (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.

        Returns:
            int: Number of iterations used for training (always 1 for this model).
        """
        # Calculate weights
        # X.T @ X + lambda * I
        identity_matrix = np.identity(x.shape[1])
        a = x.T @ x + self.l2_lambda * identity_matrix
        b = x.T @ y

        self.weights = np.linalg.solve(a, b)

        # Calculate bias (intercept)
        # The bias should be the mean of the target since X is centered
        self.bias = np.mean(y)

        # Calculate residual standard deviation for uncertainty estimation
        y_pred_train = self.predict(x)
        self._train_residual_std = np.std(y - y_pred_train)
        if np.isnan(self._train_residual_std):
            self._train_residual_std = 0.0

        return 1  # Non-iterative model, so 1 iteration

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Makes predictions on new data.

        Args:
            x (np.ndarray): Feature matrix for prediction.

        Returns:
            np.ndarray: Predicted values.
        """
        if self.weights is None or self.bias is None:
            raise RuntimeError("Model has not been fitted yet.")
        return x @ self.weights + self.bias

    def predict_uncertainty(self, x: np.ndarray) -> np.ndarray:
        """Estimates uncertainty for predictions.

        Args:
            x (np.ndarray): Feature matrix for uncertainty estimation.

        Returns:
            np.ndarray: Uncertainty estimates (e.g., standard deviation).
        """
        if not self.is_regression_model:
            raise ValueError("predict_uncertainty is only available for regression models.")
        if self.weights is None or self.bias is None:
            raise RuntimeError("Model has not been fitted yet.")
        return np.full(x.shape[0], self._train_residual_std)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Not implemented for NormalEquationLinearRegression.

        Raises:
            NotImplementedError: NormalEquationLinearRegression is a regression model.
        """
        raise NotImplementedError("NormalEquationLinearRegression is a regression model and does not support predict_proba.")

    def get_hyperparameter_search_space(self) -> dict[str, Any]:
        """Defines the hyperparameter search space for NormalEquationLinearRegression.

        Returns:
            Dict[str, Any]: A dictionary defining the hyperparameter search space.
        """
        return {
            "l2_lambda": {"type": "float", "low": 1e-6, "high": 1.0, "log": True},  # L2 regularization
        }

    def get_num_parameters(self) -> int:
        """Returns the total number of trainable parameters in the model.

        Returns:
            int: The total number of parameters.
        """
        if self.weights is None:
            return 0
        return self.weights.size + 1  # weights + bias

    def get_classifier_predictions(self, x: np.ndarray, y_true_original: np.ndarray) -> Never:
        """Not implemented for NormalEquationLinearRegression.

        Raises:
            NotImplementedError: NormalEquationLinearRegression is not a composite model.
        """
        raise NotImplementedError("NormalEquationLinearRegression is not a composite model and does not have an internal classifier for separate prediction.")

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
        metrics_calculator = Metrics("regression", self.name, y, y_pred)
        metrics_calculator.save_metrics(save_path)
        return y_pred
