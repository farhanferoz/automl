import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax._src.prng import PRNGKeyArray
import numpy as np
from .base import BaseModel
from typing import Dict, Any


class JAXLinearRegression(BaseModel):
    """
    Linear Regression model implemented using JAX.
    Supports basic linear regression with gradient descent and residual-based uncertainty.
    """

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000, **kwargs):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.key = PRNGKeyArray(jax.random.PRNGKey(0).key)
        self._is_regression_model = True  # Set regression flag
        self._train_residual_std = 0.0  # Stores standard deviation of residuals on training data

    @property
    def name(self) -> str:
        return "JAXLinearRegression"

    def _loss_fn(self, weights: jnp.ndarray, bias: jnp.ndarray, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Mean Squared Error loss function."""
        predictions = jnp.dot(X, weights) + bias
        return jnp.mean((predictions - y) ** 2)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_jax = jnp.array(X, dtype=jnp.float32)
        y_jax = jnp.array(y, dtype=jnp.float32)

        n_features = X.shape[1]

        self.key, subkey_w = jax.random.split(self.key)
        self.weights = jax.random.normal(subkey_w, (n_features,), dtype=jnp.float32) * 0.01
        self.bias = jnp.array(0.0, dtype=jnp.float32)

        loss_grad = jit(grad(self._loss_fn, argnums=(0, 1)))

        for i in range(self.n_iterations):
            grads_w, grads_b = loss_grad(self.weights, self.bias, X_jax, y_jax)
            self.weights = self.weights - self.learning_rate * grads_w
            self.bias = self.bias - self.learning_rate * grads_b

        # Calculate residual standard deviation for uncertainty estimation
        y_pred_train = self.predict(X)
        self._train_residual_std = np.std(y - y_pred_train)
        if np.isnan(self._train_residual_std):  # Handle cases of perfect fit or single point
            self._train_residual_std = 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None or self.bias is None:
            raise RuntimeError("Model has not been fitted yet.")
        X_jax = jnp.array(X, dtype=jnp.float32)
        predictions = jnp.dot(X_jax, self.weights) + self.bias
        return np.array(predictions)

    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        if not self._is_regression_model:
            raise ValueError("predict_uncertainty is only available for regression models.")
        if self.weights is None or self.bias is None:
            raise RuntimeError("Model has not been fitted yet.")
        # For simplicity, return a constant uncertainty based on training residuals
        return np.full(X.shape[0], self._train_residual_std)

    def get_hyperparameter_search_space(self) -> Dict[str, Any]:
        return {
            "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-1, "log": True},
            "n_iterations": {"type": "int", "low": 100, "high": 2000, "step": 100},
        }
