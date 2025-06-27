import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
from flax.training import train_state
from jax._src.prng import PRNGKeyArray
from typing import Dict, Any, Sequence, Callable, Tuple

from .base import BaseModel
from ..enums import UncertaintyMethod, RegressionStrategy, TaskType  # Import enums
from ..logger import logger  # Import logger


# Helper function to create a simple MLP for classifier or regression head
class MLP(nn.Module):
    features: Sequence[int]
    output_size: int
    use_batch_norm: bool = False
    dropout_rate: float = 0.0  # New parameter for dropout

    @nn.compact
    def __call__(self, x, train: bool):
        for i, dim in enumerate(self.features):
            x = nn.Dense(dim)(x)
            if self.use_batch_norm:
                x = nn.BatchNorm(use_running_average=not train, name=f"bn_{i}")(x)
            x = nn.relu(x)
            if self.dropout_rate > 0:  # Apply dropout if rate > 0
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        x = nn.Dense(self.output_size)(x)
        return x


class JAXProbabilisticRegressionModel(BaseModel):
    """
    A regression model implemented fully in JAX/Flax.
    It first performs classification, then feeds probabilities into regression heads,
    and combines their outputs. The entire model is trained end-to-end.
    Supports different regression strategies for the heads.
    Supports constant, MC-Dropout, and probabilistic layer uncertainty estimation.
    """

    def __init__(
        self,
        input_size: int = None,
        n_classes: int = 5,
        base_classifier_params: Dict[str, Any] = None,  # Params for the internal JAX NN classifier
        regression_head_params: Dict[str, Any] = None,  # Params for each internal JAX NN regression head
        regression_strategy: RegressionStrategy = RegressionStrategy.SEPARATE_HEADS,
        learning_rate: float = 0.001,
        n_epochs: int = 10,
        batch_size: int = 32,
        random_seed: int = 0,
        uncertainty_method: UncertaintyMethod = UncertaintyMethod.CONSTANT,
        n_mc_dropout_samples: int = 100,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.n_classes = n_classes
        self.base_classifier_params = base_classifier_params if base_classifier_params is not None else {}
        self.regression_head_params = regression_head_params if regression_head_params is not None else {}
        self.regression_strategy = regression_strategy
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.uncertainty_method = uncertainty_method
        self.n_mc_dropout_samples = n_mc_dropout_samples
        self.dropout_rate = dropout_rate

        self.key = PRNGKeyArray(jax.random.PRNGKey(self.random_seed).key)

        if self.regression_strategy not in [RegressionStrategy.SEPARATE_HEADS, RegressionStrategy.SINGLE_HEAD_N_OUTPUTS, RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT]:
            raise ValueError(f"Unsupported regression_strategy: {self.regression_strategy.value}. Choose from enum values.")
        if self.uncertainty_method not in [UncertaintyMethod.CONSTANT, UncertaintyMethod.MC_DROPOUT, UncertaintyMethod.PROBABILISTIC]:
            raise ValueError(f"Unsupported uncertainty_method: {self.uncertainty_method.value}. Choose from enum values.")

        self.model_def = None
        self.params = None
        self.batch_stats = None
        self._is_regression_model = True
        self._train_residual_std = 0.0  # For 'constant' uncertainty method

    @property
    def name(self) -> str:
        return "JAXProbabilisticRegression"

    class _CombinedJAXModel(nn.Module):
        """
        Internal Flax Module combining the classifier and regression heads.
        """

        classifier_features: Sequence[int]
        classifier_use_batch_norm: bool
        classifier_dropout_rate: float  # Added
        n_classes: int
        regression_strategy: RegressionStrategy  # Use enum
        regression_head_features: Sequence[int]
        regression_head_use_batch_norm: bool
        regression_head_dropout_rate: float  # Added
        uncertainty_method: UncertaintyMethod  # Added

        @nn.compact
        def __call__(self, x_input: jnp.ndarray, train: bool):
            # Classifier part
            classifier_output_size = self.n_classes

            classifier_logits = MLP(
                features=self.classifier_features,
                output_size=classifier_output_size,
                use_batch_norm=self.classifier_use_batch_norm,
                dropout_rate=self.classifier_dropout_rate,  # Pass dropout rate
                name="classifier_mlp",
            )(x_input, train=train)

            # Apply softmax to get probabilities
            if self.n_classes == 2:
                proba_raw = nn.sigmoid(classifier_logits)
                probabilities = jnp.concatenate([1 - proba_raw, proba_raw], axis=-1)
            else:
                probabilities = nn.softmax(classifier_logits, axis=-1)

            # Determine regression head output size
            regression_output_size = 1
            if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC:
                regression_output_size = 2  # Mean and Log-Variance

            if self.regression_strategy == RegressionStrategy.SEPARATE_HEADS:
                final_predictions = jnp.zeros((x_input.shape[0], regression_output_size))
                for i in range(self.n_classes):
                    p_i = probabilities[:, i : i + 1]  # Shape (batch_size, 1)

                    y_i_expected_or_dist_params = MLP(
                        features=self.regression_head_features,
                        output_size=regression_output_size,
                        use_batch_norm=self.regression_head_use_batch_norm,
                        dropout_rate=self.regression_head_dropout_rate,  # Pass dropout rate
                        name=f"reg_head_{i}",  # Unique name for each head
                    )(p_i, train=train)

                    if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC:
                        # Store mean and log_var for this class
                        final_predictions += p_i * y_i_expected_or_dist_params
                    else:
                        # For point estimates, directly sum the products
                        final_predictions += p_i * y_i_expected_or_dist_params
                return {"output": final_predictions.squeeze(-1)}  # Squeeze to 1D if not probabilistic
                # If probabilistic, it will be (batch_size, 2)

            elif self.regression_strategy == RegressionStrategy.SINGLE_HEAD_N_OUTPUTS:
                if self.n_classes > 1:
                    head_input_probas = probabilities[:, :-1]
                else:
                    head_input_probas = jnp.empty((x_input.shape[0], 0))

                y_expected_all_classes = MLP(
                    features=self.regression_head_features,
                    output_size=self.n_classes * regression_output_size,  # n classes * (mean, log_var)
                    use_batch_norm=self.regression_head_use_batch_norm,
                    dropout_rate=self.regression_head_dropout_rate,  # Pass dropout rate
                    name="single_reg_head_n_outputs",
                )(head_input_probas, train=train)

                # Reshape to (batch_size, n_classes, regression_output_size)
                y_expected_all_classes = y_expected_all_classes.reshape(x_input.shape[0], self.n_classes, regression_output_size)

                # Expand probabilities for broadcasting: (batch_size, n_classes, 1)
                probabilities_expanded = probabilities[:, :, jnp.newaxis]

                # Final output is sum of (P_i * Y_i_expected)
                final_predictions_summed = (probabilities_expanded * y_expected_all_classes).sum(axis=1)  # Sum over n_classes dimension

                return {"output": final_predictions_summed.squeeze(-1)}  # Squeeze if regression_output_size is 1

            elif self.regression_strategy == RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT:
                if self.n_classes > 1:
                    head_input_probas = probabilities[:, :-1]
                else:
                    head_input_probas = jnp.empty((x_input.shape[0], 0))

                final_predictions = MLP(
                    features=self.regression_head_features,
                    output_size=regression_output_size,  # Single final regression output or (mean, log_var)
                    use_batch_norm=self.regression_head_use_batch_norm,
                    dropout_rate=self.regression_head_dropout_rate,  # Pass dropout rate
                    name="single_reg_head_final_output",
                )(head_input_probas, train=train)

                return {"output": final_predictions.squeeze(-1)}  # Squeeze to 1D if not probabilistic
            else:
                logger.error(f"Invalid regression_strategy: {self.regression_strategy.value}")
                raise ValueError("Invalid regression_strategy")

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_jax = jnp.array(X, dtype=jnp.float32)
        y_jax = jnp.array(y, dtype=jnp.float32)

        # Extract parameters for MLP features and batch norm/dropout
        base_classifier_features = [self.base_classifier_params.get("hidden_size", 64)] * self.base_classifier_params.get("hidden_layers", 1)
        base_classifier_use_batch_norm = self.base_classifier_params.get("use_batch_norm", False)
        base_classifier_dropout_rate = self.base_classifier_params.get("dropout_rate", 0.0)  # From HPO

        regression_head_features = [self.regression_head_params.get("hidden_size", 32)] * self.regression_head_params.get("hidden_layers", 0)
        regression_head_use_batch_norm = self.regression_head_params.get("use_batch_norm", False)
        regression_head_dropout_rate = self.regression_head_params.get("dropout_rate", 0.0)  # From HPO

        self.model_def = self._CombinedJAXModel(
            classifier_features=base_classifier_features,
            classifier_use_batch_norm=base_classifier_use_batch_norm,
            classifier_dropout_rate=base_classifier_dropout_rate,
            n_classes=self.n_classes,
            regression_strategy=self.regression_strategy,
            regression_head_features=regression_head_features,
            regression_head_use_batch_norm=regression_head_use_batch_norm,
            regression_head_dropout_rate=regression_head_dropout_rate,
            uncertainty_method=self.uncertainty_method,  # Pass uncertainty method
        )

        rng_key, init_key = jax.random.split(self.key)
        variables = self.model_def.init({"params": init_key, "dropout": init_key, "batch_stats": init_key}, X_jax[:1], train=True)
        self.params = variables["params"]
        self.batch_stats = variables["batch_stats"]

        optimizer = optax.adam(self.learning_rate)
        opt_state = optimizer.init(self.params)

        @jax.jit
        def loss_fn(params, batch_stats, rng, x, y, train):
            variables = self.model_def.apply({"params": params, "dropout": rng, "batch_stats": batch_stats}, x, train=train, mutable=["batch_stats"])
            predictions_output = variables["output"]  # Can be (batch_size,) or (batch_size, 2)

            if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC:
                mean = predictions_output[:, 0]
                log_var = predictions_output[:, 1]
                loss = 0.5 * jnp.mean(jnp.exp(-log_var) * (y - mean) ** 2 + log_var)
            else:
                loss = jnp.mean(jnp.square(predictions_output - y))
            return loss, variables["batch_stats"]

        @jax.jit
        def train_step(params, batch_stats, opt_state, rng, x, y):
            (loss, new_batch_stats), grads = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(params, batch_stats, rng, x, y, train=True)
            updates, opt_state = optimizer.update(grads["params"], opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_batch_stats, opt_state, loss

        num_batches = int(np.ceil(len(X_jax) / self.batch_size))
        for epoch in range(self.n_epochs):
            self.key, shuffle_key = jax.random.split(self.key)
            indices = jax.random.permutation(shuffle_key, len(X_jax))
            X_shuffled = X_jax[indices]
            y_shuffled = y_jax[indices]

            for i in range(num_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, len(X_jax))
                batch_X = X_shuffled[start_idx:end_idx]
                batch_y = y_shuffled[start_idx:end_idx]

                self.key, dropout_key = jax.random.split(self.key)  # Key for dropout

                self.params, self.batch_stats, opt_state, loss_value = train_step(self.params, self.batch_stats, opt_state, dropout_key, batch_X, batch_y)

        if self.uncertainty_method == UncertaintyMethod.CONSTANT:
            y_pred_train = self.predict(X)
            self._train_residual_std = np.std(y - y_pred_train)
            if np.isnan(self._train_residual_std):
                self._train_residual_std = 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.params is None:
            raise RuntimeError("Model has not been fitted yet.")
        X_jax = jnp.array(X, dtype=jnp.float32)

        @jax.jit
        def predict_fn(params, batch_stats, rng, x, train_mode):
            variables = self.model_def.apply({"params": params, "dropout": rng, "batch_stats": batch_stats}, x, train=train_mode, mutable=False)
            return variables["output"]

        if self.uncertainty_method == UncertaintyMethod.MC_DROPOUT:
            mc_predictions = []
            for _ in range(self.n_mc_dropout_samples):
                self.key, dropout_key = jax.random.split(self.key)
                # Need to pass train=True to enable dropout during prediction for MC-Dropout
                outputs = predict_fn(self.params, self.batch_stats, dropout_key, X_jax, True)
                mc_predictions.append(np.array(outputs).flatten())
            return np.mean(mc_predictions, axis=0)

        # For 'constant' and 'probabilistic' uncertainty, run in eval mode (dropout off)
        self.key, dropout_key = jax.random.split(self.key)
        outputs = predict_fn(self.params, self.batch_stats, dropout_key, X_jax, False)

        if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC:
            predictions = outputs[:, 0]  # Return only the mean
        else:
            predictions = outputs
        return np.array(predictions)

    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        if not self._is_regression_model:
            raise ValueError("predict_uncertainty is only available for regression models.")
        if self.params is None:
            raise RuntimeError("Model has not been fitted yet.")
        X_jax = jnp.array(X, dtype=jnp.float32)

        @jax.jit
        def predict_fn_for_uncertainty(params, batch_stats, rng, x, train_mode):
            variables = self.model_def.apply({"params": params, "dropout": rng, "batch_stats": batch_stats}, x, train=train_mode, mutable=False)
            return variables["output"]

        if self.uncertainty_method == UncertaintyMethod.CONSTANT:
            return np.full(X.shape[0], self._train_residual_std)
        elif self.uncertainty_method == UncertaintyMethod.PROBABILISTIC:
            self.key, dropout_key = jax.random.split(self.key)
            outputs = predict_fn_for_uncertainty(self.params, self.batch_stats, dropout_key, X_jax, False)  # Eval mode
            log_var = outputs[:, 1]
            uncertainty = jnp.exp(0.5 * log_var)
            return np.array(uncertainty)
        elif self.uncertainty_method == UncertaintyMethod.MC_DROPOUT:
            mc_predictions = []
            for _ in range(self.n_mc_dropout_samples):
                self.key, dropout_key = jax.random.split(self.key)
                # Enable dropout for MC sampling
                outputs = predict_fn_for_uncertainty(self.params, self.batch_stats, dropout_key, X_jax, True)
                mc_predictions.append(np.array(outputs).flatten())
            return np.std(mc_predictions, axis=0)
        else:
            logger.error(f"Unknown uncertainty_method: {self.uncertainty_method.value}")
            raise ValueError(f"Unknown uncertainty_method: {self.uncertainty_method.value}")

    def get_hyperparameter_search_space(self) -> Dict[str, Any]:
        return {
            "n_classes": {"type": "int", "low": 2, "high": 5},
            "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
            "n_epochs": {"type": "int", "low": 10, "high": 50, "step": 10},
            "batch_size": {"type": "categorical", "choices": [16, 32, 64]},
            "random_seed": {"type": "int", "low": 0, "high": 100},
            "regression_strategy": {"type": "categorical", "choices": [e.value for e in RegressionStrategy]},
            # Nested parameters for the internal classifier (JAX MLP)
            "base_classifier_params__hidden_layers": {"type": "int", "low": 1, "high": 2},
            "base_classifier_params__hidden_size": {"type": "int", "low": 32, "high": 64, "step": 32},
            "base_classifier_params__use_batch_norm": {"type": "categorical", "choices": [True, False]},
            "base_classifier_params__dropout_rate": {"type": "float", "low": 0.0, "high": 0.5, "step": 0.1},  # New dropout for JAX MLP
            # Nested parameters for the internal regression heads (JAX MLP)
            "regression_head_params__hidden_layers": {"type": "int", "low": 0, "high": 1},
            "regression_head_params__hidden_size": {"type": "int", "low": 16, "high": 32, "step": 16},
            "regression_head_params__use_batch_norm": {"type": "categorical", "choices": [True, False]},
            "regression_head_params__dropout_rate": {"type": "float", "low": 0.0, "high": 0.5, "step": 0.1},  # New dropout for JAX MLP
            "uncertainty_method": {"type": "categorical", "choices": [e.value for e in UncertaintyMethod]},
            "n_mc_dropout_samples": {"type": "int", "low": 50, "high": 200, "step": 50},
            "dropout_rate": {"type": "float", "low": 0.1, "high": 0.5, "step": 0.1},
        }

    def get_internal_model(self):
        """
        Returns the raw underlying Flax model definition and its parameters/batch_stats.
        """
        if self.model_def is None or self.params is None:
            raise RuntimeError("Model has not been fitted yet.")
        return {"model_def": self.model_def, "params": self.params, "batch_stats": self.batch_stats}
