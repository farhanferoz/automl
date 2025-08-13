"""Linear Regression model implemented using JAX."""

from typing import Any, Never

import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from automl_package.utils.metrics import Metrics

from .base import BaseModel


class JAXLinearRegression(BaseModel):
    """Linear Regression model implemented using JAX.

    Supports basic linear regression with gradient descent and residual-based uncertainty.
    Includes L1 and L2 regularization.
    """

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000, l1_lambda: float = 0.0, l2_lambda: float = 0.0, **kwargs: Any) -> None:
        """Initializes the JAXLinearRegression model.

        Args:
            learning_rate (float): The learning rate for gradient descent.
            n_iterations (int): The number of training iterations.
            l1_lambda (float): L1 regularization strength.
            l2_lambda (float): L2 regularization strength.
            **kwargs: Additional keyword arguments for the BaseModel.
        """
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.weights: jnp.ndarray | None = None
        self.bias: jnp.ndarray | None = None
        self.key = jax.random.PRNGKey(0)
        self.is_regression_model = True  # Set regression flag
        self._train_residual_std = 0.0  # Stores standard deviation of residuals on training data
        self.n_features = 0  # Initialize n_features

    @property
    def name(self) -> str:
        """Returns the name of the model."""
        return "JAXLinearRegression"

    def _loss_fn(self, weights: jnp.ndarray, bias: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Mean Squared Error loss function with L1 and L2 regularization."""
        predictions = jnp.dot(x, weights) + bias
        mse_loss = jnp.mean((predictions - y) ** 2)

        l1_penalty = self.l1_lambda * jnp.sum(jnp.abs(weights))
        l2_penalty = self.l2_lambda * jnp.sum(weights**2)

        return mse_loss + l1_penalty + l2_penalty

    def fit(self, x: np.ndarray, y: np.ndarray) -> int:
        """Fits the JAXLinearRegression model to the training data.

        Args:
            x (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.

        Returns:
            int: Number of iterations used for training.
        """
        x_train, y_train = x, y

        self.n_features = x_train.shape[1]  # Store n_features

        x_train_jax = jnp.array(x_train, dtype=jnp.float32)
        y_train_jax = jnp.array(y_train, dtype=jnp.float32)

        self.key, subkey_w = jax.random.split(self.key)
        self.weights = jax.random.normal(subkey_w, (self.n_features,), dtype=jnp.float32) * 0.01
        self.bias = jnp.array(0.0, dtype=jnp.float32)

        loss_grad = jit(grad(self._loss_fn, argnums=(0, 1)))

        for i in range(self.n_iterations):
            grads_w, grads_b = loss_grad(self.weights, self.bias, x_train_jax, y_train_jax)
            self.weights = self.weights - self.learning_rate * grads_w
            self.bias = self.bias - self.learning_rate * grads_b

        # Calculate residual standard deviation for uncertainty estimation on the full training data
        y_pred_train = self.predict(x)
        self._train_residual_std = np.std(y - y_pred_train)
        if np.isnan(self._train_residual_std):  # Handle cases of perfect fit or single point
            self._train_residual_std = 0.0

        return self.n_iterations  # Return the number of iterations actually used

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Makes predictions on new data.

        Args:
            x (np.ndarray): Feature matrix for prediction.

        Returns:
            np.ndarray: Predicted values.
        """
        if self.weights is None or self.bias is None:
            raise RuntimeError("Model has not been fitted yet.")
        x_jax = jnp.array(x, dtype=jnp.float32)
        predictions = jnp.dot(x_jax, self.weights) + self.bias
        return np.array(predictions)

    def get_num_parameters(self) -> int:
        """Returns the total number of trainable parameters in the model.

        Returns:
            int: The total number of parameters.
        """
        if self.weights is None:
            return 0  # Or raise an error if model not fitted
        return self.weights.size + 1  # weights + bias

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
        # For simplicity, return a constant uncertainty based on training residuals
        return np.full(x.shape[0], self._train_residual_std)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Not implemented for JAXLinearRegression.

        Raises:
            NotImplementedError: JAXLinearRegression is a regression model.
        """
        raise NotImplementedError("JAXLinearRegression is a regression model and does not support predict_proba.")

    def get_hyperparameter_search_space(self) -> dict[str, Any]:
        """Defines the hyperparameter search space for JAXLinearRegression.

        Returns:
            Dict[str, Any]: A dictionary defining the hyperparameter search space.
        """
        return {
            "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-1, "log": True},
            "n_iterations": {"type": "int", "low": 100, "high": 2000, "step": 100},
            "l1_lambda": {"type": "float", "low": 1e-6, "high": 1.0, "log": True},  # L1 regularization
            "l2_lambda": {"type": "float", "low": 1e-6, "high": 1.0, "log": True},  # L2 regularization
        }

    def get_classifier_predictions(self, x: np.ndarray, y_true_original: np.ndarray) -> Never:
        """Not implemented for JAXLinearRegression.

        Raises:
            NotImplementedError: JAXLinearRegression is not a composite model.
        """
        raise NotImplementedError("JAXLinearRegression is not a composite model and does not have an internal classifier for separate prediction.")

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
