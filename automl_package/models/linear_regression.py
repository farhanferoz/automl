"""Linear Regression model implemented using JAX."""

from typing import Any, ClassVar, Never

import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit
from sklearn.metrics import mean_squared_error

from automl_package.enums import Metric, Penalty
from automl_package.logger import logger
from automl_package.models.base import BaseModel


class LinearRegressionModel(BaseModel):
    """Linear Regression model implemented using JAX."""

    _defaults: ClassVar[dict[str, Any]] = {
        "learning_rate": 0.01,
        "n_iterations": 1000,
        "penalty": None,
        "regularization_strength": 0.0,
    }

    def __init__(self, **kwargs: Any) -> None:
        """Initializes the LinearRegressionModel model."""
        for key, value in self._defaults.items():
            kwargs.setdefault(key, value)

        super().__init__(**kwargs)
        assert self.is_regression_model
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.weights: jnp.ndarray | None = None
        self.bias: jnp.ndarray | None = None
        self.key = jax.random.PRNGKey(0)
        self._train_residual_std = 0.0
        self.n_features = 0

    @property
    def name(self) -> str:
        """Returns the name of the model."""
        return "JAXLinearRegression"

    def _get_optimization_metric(self) -> Metric:
        """Gets the optimization metric for the model."""
        return Metric.RMSE

    def _loss_fn(
        self, weights: jnp.ndarray, bias: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray
    ) -> jnp.ndarray:
        """Mean Squared Error loss function with L1 and L2 regularization."""
        predictions = jnp.dot(x, weights) + bias
        mse_loss = jnp.mean((predictions - y) ** 2)

        if self.penalty == Penalty.L1:
            penalty_term = self.regularization_strength * jnp.sum(jnp.abs(weights))
        elif self.penalty == Penalty.L2:
            penalty_term = self.regularization_strength * jnp.sum(weights**2)
        elif self.penalty == Penalty.ELASTICNET:
            l1_penalty = self.l1_ratio * jnp.sum(jnp.abs(weights))
            l2_penalty = (1 - self.l1_ratio) * jnp.sum(weights**2)
            penalty_term = self.regularization_strength * (l1_penalty + l2_penalty)
        else:
            penalty_term = 0.0

        return mse_loss + penalty_term

    def _fit_single(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        forced_iterations: int | None = None,
    ) -> tuple[int, list[float]]:
        """Fits a single model instance.

        Args:
            x_train (np.ndarray): The training features.
            y_train (np.ndarray): The training targets.
            x_val (np.ndarray | None): The validation features.
            y_val (np.ndarray | None): The validation targets.
            forced_iterations (int | None): If provided, train for this many iterations, ignoring early stopping.

        Returns:
            tuple[int, list[float]]: A tuple containing:
                - The number of iterations the model was trained for.
                - A list of the validation loss values for each epoch.
        """
        self.n_features = x_train.shape[1]

        x_train_jax = jnp.array(x_train, dtype=jnp.float32)
        y_train_jax = jnp.array(y_train, dtype=jnp.float32)

        self.key, subkey_w = jax.random.split(self.key)
        self.weights = (
            jax.random.normal(subkey_w, (self.n_features,), dtype=jnp.float32) * 0.01
        )
        self.bias = jnp.array(0.0, dtype=jnp.float32)

        loss_grad = jit(grad(self._loss_fn, argnums=(0, 1)))

        use_early_stopping = (
            self.early_stopping_rounds is not None
            and forced_iterations is None
            and x_val is not None
            and y_val is not None
        )

        if use_early_stopping:
            best_val_loss = float("inf")
            patience_counter = 0
            best_weights = self.weights
            best_bias = self.bias
            best_iter = 0
            val_loss_history = []

            x_val_jax = jnp.array(x_val, dtype=jnp.float32)
            y_val_jax = jnp.array(y_val, dtype=jnp.float32)

            n_iterations = self.n_iterations
            for i in range(n_iterations):
                grads_w, grads_b = loss_grad(
                    self.weights, self.bias, x_train_jax, y_train_jax
                )
                self.weights = self.weights - self.learning_rate * grads_w
                self.bias = self.bias - self.learning_rate * grads_b

                val_loss = self._loss_fn(self.weights, self.bias, x_val_jax, y_val_jax)
                val_loss_history.append(float(val_loss))

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_weights = self.weights
                    best_bias = self.bias
                    best_iter = i + 1
                else:
                    patience_counter += 1

                if patience_counter >= self.early_stopping_rounds:
                    logger.info(f"Early stopping at iteration {best_iter}")
                    break

            self.weights = best_weights
            self.bias = best_bias
            iterations_to_return = best_iter
        else:
            iterations = forced_iterations or self.n_iterations
            for _i in range(iterations):
                grads_w, grads_b = loss_grad(
                    self.weights, self.bias, x_train_jax, y_train_jax
                )
                self.weights = self.weights - self.learning_rate * grads_w
                self.bias = self.bias - self.learning_rate * grads_b
            iterations_to_return = iterations
            val_loss_history = []

        y_pred_train = self.predict(x_train, filter_data=False)
        self._train_residual_std = np.std(y_train - y_pred_train)
        if np.isnan(self._train_residual_std):
            self._train_residual_std = 0.0

        return iterations_to_return, val_loss_history

    def _evaluate_trial(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Evaluates a trial for hyperparameter optimization."""
        return self._calculate_metric(y_true, y_pred, Metric.RMSE)

    def _calculate_metric(
        self, y_true: np.ndarray, y_pred: np.ndarray, metric: Metric
    ) -> float:
        """Calculates a metric."""
        if metric == Metric.RMSE:
            return np.sqrt(mean_squared_error(y_true, y_pred))
        raise ValueError(f"Unknown metric: {metric}")

    def _clone(self) -> "LinearRegressionModel":
        """Creates a new instance of the model with the same parameters."""
        return LinearRegressionModel(**self.get_params())

    def get_params(self) -> dict[str, Any]:
        """Gets parameters for this estimator.

        Returns:
            dict: Parameter names mapped to their values.
        """
        params = super().get_params()
        params.update(
            {
                "learning_rate": self.learning_rate,
                "n_iterations": self.n_iterations,
                "penalty": self.penalty,
                "regularization_strength": self.regularization_strength,
            }
        )
        if self.penalty == Penalty.ELASTICNET:
            params["l1_ratio"] = self.l1_ratio
        return params

    def predict(self, x: np.ndarray, filter_data: bool = True) -> np.ndarray:
        """Makes predictions on new data.

        Args:
            x (np.ndarray): Feature matrix for prediction.
            filter_data (bool): If True, filter the input data using the feature selection mask.

        Returns:
            np.ndarray: Predicted values.
        """
        if self.weights is None or self.bias is None:
            raise RuntimeError("Model has not been fitted yet.")
        if filter_data:
            x = self._filter_predict_data(x)
        x_jax = jnp.array(x, dtype=jnp.float32)
        predictions = jnp.dot(x_jax, self.weights) + self.bias
        return np.array(predictions)

    def get_num_parameters(self) -> int:
        """Returns the total number of trainable parameters in the model.

        Returns:
            int: The total number of parameters.
        """
        return 0 if self.weights is None else self.weights.size + 1

    def predict_uncertainty(
        self, x: np.ndarray, filter_data: bool = True
    ) -> np.ndarray:
        """Estimates uncertainty for predictions.

        Args:
            x (np.ndarray): Feature matrix for uncertainty estimation.
            filter_data (bool): If True, filter the input data using the feature selection mask.

        Returns:
            np.ndarray: Uncertainty estimates (e.g., standard deviation).
        """
        if not self.is_regression_model:
            raise ValueError(
                "predict_uncertainty is only available for regression models."
            )
        if self.weights is None or self.bias is None:
            raise RuntimeError("Model has not been fitted yet.")
        if filter_data:
            x = self._filter_predict_data(x)
        # For simplicity, return a constant uncertainty based on training residuals
        return np.full(x.shape[0], self._train_residual_std)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Not implemented for JAXLinearRegression.

        Raises:
            NotImplementedError: JAXLinearRegression is a regression model.
        """
        raise NotImplementedError(
            "JAXLinearRegression is a regression model and does not support predict_proba."
        )

    def get_hyperparameter_search_space(self) -> dict[str, Any]:
        """Defines the hyperparameter search space for JAXLinearRegression.

        Returns:
            Dict[str, Any]: A dictionary defining the hyperparameter search space.
        """
        space = {
            "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-1, "log": True},
            "n_iterations": {"type": "int", "low": 100, "high": 2000, "step": 100},
            "penalty": {
                "type": "categorical",
                "choices": [Penalty.L1, Penalty.L2, Penalty.ELASTICNET, None],
            },
            "regularization_strength": {
                "type": "float",
                "low": 1e-6,
                "high": 1.0,
                "log": True,
            },
            "l1_ratio": {"type": "float", "low": 0.0, "high": 1.0},
        }
        if self.search_space_override:
            space.update(self.search_space_override)
        return space

    def get_classifier_predictions(
        self, x: np.ndarray, y_true_original: np.ndarray
    ) -> Never:
        """Not implemented for JAXLinearRegression.

        Raises:
            NotImplementedError: JAXLinearRegression is not a composite model.
        """
        raise NotImplementedError(
            "JAXLinearRegression is not a composite model and does not have an internal classifier for separate prediction."
        )

    def get_internal_model(self) -> Any:
        """Returns the internal model."""

        class ShapModel:
            def __init__(self, coef: np.ndarray, intercept: np.ndarray) -> None:
                self.coef_ = coef
                self.intercept_ = intercept

        return ShapModel(self.weights, self.bias)

    def cross_validate(self, x: np.ndarray, y: np.ndarray, cv: int) -> dict[str, Any]:
        """Performs cross-validation and returns the scores."""
        self.cv_folds = cv
        self.fit(x, y)
        return {"test_score": self.cv_score_mean_}
