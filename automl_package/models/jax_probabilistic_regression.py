from typing import Dict, Any, Sequence
import flax.linen as nn  # Using nn for Flax to avoid collision with torch.nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax._src.prng import PRNGKeyArray

from .base import BaseModel
from ..enums import UncertaintyMethod, RegressionStrategy, TaskType  # Import enums
from ..logger import logger  # Import logger


# Helper function to create a simple MLP for classifier or regression head
class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron for Flax/JAX.
    Used as the core network in JAXProbabilisticRegressionModel.
    """

    features: Sequence[int]  # List of hidden layer sizes
    output_size: int  # Size of the final output layer (e.g., 1 for mean, 1 for log_var)
    use_batch_norm: bool = False
    dropout_rate: float = 0.0  # New parameter for dropout
    l1_lambda: float = 0.0  # New parameter for L1 regularization
    l2_lambda: float = 0.0  # New parameter for L2 regularization

    @nn.compact
    def __call__(self, x, train: bool):
        """
        Forward pass for the MLP.
        Args:
            x (jnp.ndarray): Input data.
            train (bool): Flag indicating if the model is in training mode (affects dropout/batch norm).
        Returns:
            jnp.ndarray: Output of the MLP.
        """
        kernel_init = nn.initializers.lecun_normal()  # Good default for ReLU
        bias_init = nn.initializers.zeros

        for i, dim in enumerate(self.features):
            x = nn.Dense(dim, kernel_init=kernel_init, bias_init=bias_init)(x)
            if self.use_batch_norm:
                # use_running_average=not train: update batch norm stats during train, use fixed during eval
                x = nn.BatchNorm(use_running_average=not train, name=f"bn_{i}")(x)
            x = nn.relu(x)
            if self.dropout_rate > 0:  # Apply dropout if rate > 0
                # deterministic=not train: apply dropout during train, disable during eval
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        x = nn.Dense(self.output_size, kernel_init=kernel_init, bias_init=bias_init)(x)
        return x


class JAXProbabilisticRegressionModel(BaseModel):
    """
    A regression model implemented fully in JAX/Flax.
    It first performs classification, then feeds probabilities into regression heads,
    and combines their outputs. The entire model is trained end-to-end.
    Supports different regression strategies for the heads.
    Supports constant, MC-Dropout, and probabilistic layer uncertainty estimation.
    Includes L1 and L2 regularization for its MLP components.
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
        dropout_rate: float = 0.1,  # Top-level dropout rate for MC_DROPOUT
        task_type: TaskType = TaskType.REGRESSION,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.task_type = task_type
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
        self.dropout_rate = dropout_rate  # This dropout_rate parameter for MC_DROPOUT in the higher level

        # Base PRNG key for reproducibility in JAX
        self.key = PRNGKeyArray(jax.random.PRNGKey(self.random_seed).key)

        # Validate regression strategy and uncertainty method
        if self.regression_strategy not in [RegressionStrategy.SEPARATE_HEADS, RegressionStrategy.SINGLE_HEAD_N_OUTPUTS, RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT]:
            raise ValueError(f"Unsupported regression_strategy: {self.regression_strategy.value}. Choose from enum values.")
        if self.uncertainty_method not in [UncertaintyMethod.CONSTANT, UncertaintyMethod.MC_DROPOUT, UncertaintyMethod.PROBABILISTIC]:
            raise ValueError(f"Unsupported uncertainty_method: {self.uncertainty_method.value}. Choose from enum values.")

        self.model_def = None  # The Flax MLP module definition
        self.params = None  # Model parameters (Flax's immutable dict)
        self.batch_stats = None  # For BatchNorm
        self.optimizer = None  # Optax optimizer state
        self.state = None  # Flax TrainState

        self._is_regression_model = True
        self._train_residual_std = 0.0  # For 'constant' uncertainty method

    @property
    def name(self) -> str:
        return "JAXProbabilisticRegression"

    class _CombinedJAXModel(nn.Module):
        """
        Internal Flax Module combining the classifier and regression heads.
        This is the actual neural network structure for JAXProbabilisticRegressionModel.
        """

        classifier_features: Sequence[int]
        classifier_use_batch_norm: bool
        classifier_dropout_rate: float  # Dropout rate for classifier MLP
        n_classes: int
        regression_strategy: RegressionStrategy  # Use enum
        regression_head_features: Sequence[int]
        regression_head_use_batch_norm: bool
        regression_head_dropout_rate: float  # Dropout rate for regression head MLP
        uncertainty_method: UncertaintyMethod  # Use enum
        # Regularization parameters for the MLPs (passed from outer class params)
        classifier_l1_lambda: float
        classifier_l2_lambda: float
        regression_head_l1_lambda: float
        regression_head_l2_lambda: float

        @nn.compact
        def __call__(self, x_input: jnp.ndarray, train: bool):
            # Classifier part: Outputs logits for n_classes
            classifier_output_size = self.n_classes
            classifier_logits = MLP(
                features=self.classifier_features,
                output_size=classifier_output_size,
                use_batch_norm=self.classifier_use_batch_norm,
                dropout_rate=self.classifier_dropout_rate,  # Dropout applied here
                l1_lambda=self.classifier_l1_lambda,  # Pass regularization
                l2_lambda=self.classifier_l2_lambda,  # Pass regularization
                name="classifier_mlp",  # Unique name for this MLP
            )(x_input, train=train)

            # Apply softmax to get probabilities. For binary, sigmoid is better but softmax works too.
            if self.n_classes == 2:
                # For binary classification (n_classes=2), usually a single logit is output,
                # then apply sigmoid to get prob of class 1, and 1-prob for class 0.
                # Assuming classifier_logits is (batch_size, 1) for binary from the MLP.
                proba_positive = nn.sigmoid(classifier_logits)
                probabilities = jnp.concatenate([1 - proba_positive, proba_positive], axis=-1)  # Shape (batch_size, 2)
            else:
                probabilities = nn.softmax(classifier_logits, axis=-1)  # Shape (batch_size, n_classes)

            # Determine regression head output size (1 for mean, 2 for mean+log_var)
            regression_output_size = 1
            if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC:
                regression_output_size = 2

            # Pass probabilities through regression heads based on strategy
            if self.regression_strategy == RegressionStrategy.SEPARATE_HEADS:
                final_predictions = jnp.zeros((x_input.shape[0], regression_output_size))  # Initialize sum
                for i in range(self.n_classes):
                    p_i = probabilities[:, i : i + 1]  # Probability for class i, shape (batch_size, 1)

                    # Each regression head is an MLP that takes a single probability as input
                    y_i_expected_or_dist_params = MLP(
                        features=self.regression_head_features,
                        output_size=regression_output_size,
                        use_batch_norm=self.regression_head_use_batch_norm,
                        dropout_rate=self.regression_head_dropout_rate,  # Dropout applied here
                        l1_lambda=self.regression_head_l1_lambda,  # Pass regularization
                        l2_lambda=self.regression_head_l2_lambda,  # Pass regularization
                        name=f"reg_head_{i}",  # Unique name for each head
                    )(p_i, train=train)

                    # Sum (P_i * Y_i_expected) across all classes
                    final_predictions += p_i * y_i_expected_or_dist_params

                return {"output": final_predictions.squeeze(-1)}  # Squeeze to 1D if regression_output_size is 1, otherwise (batch_size, 2)

            elif self.regression_strategy == RegressionStrategy.SINGLE_HEAD_N_OUTPUTS:
                # Input to this head is all but the last probability (or all if n_classes=1, though unlikely for this model)
                # Ensure input is 2D
                head_input_probas = probabilities if self.n_classes == 1 else probabilities[:, :-1]

                # This MLP outputs `n_classes * regression_output_size` values
                y_expected_all_classes = MLP(
                    features=self.regression_head_features,
                    output_size=self.n_classes * regression_output_size,  # Total outputs from head
                    use_batch_norm=self.regression_head_use_batch_norm,
                    dropout_rate=self.regression_head_dropout_rate,
                    l1_lambda=self.regression_head_l1_lambda,  # Pass regularization
                    l2_lambda=self.regression_head_l2_lambda,  # Pass regularization
                    name="single_reg_head_n_outputs",
                )(head_input_probas, train=train)

                # Reshape to (batch_size, n_classes, regression_output_size) for element-wise multiplication
                y_expected_all_classes = y_expected_all_classes.reshape(x_input.shape[0], self.n_classes, regression_output_size)

                # Expand probabilities for broadcasting: (batch_size, n_classes, 1)
                probabilities_expanded = probabilities.unsqueeze(-1)

                # Final output is sum of (P_i * Y_i_expected)
                final_predictions_summed = (probabilities_expanded * y_expected_all_classes).sum(axis=1)  # Sum over n_classes dimension

                return {"output": final_predictions_summed.squeeze(-1)}

            elif self.regression_strategy == RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT:
                # Input to head is all but the last probability (if n_classes > 1)
                head_input_probas = probabilities if self.n_classes == 1 else probabilities[:, :-1]

                final_predictions = MLP(
                    features=self.regression_head_features,
                    output_size=regression_output_size,  # Single final regression output or (mean, log_var)
                    use_batch_norm=self.regression_head_use_batch_norm,
                    dropout_rate=self.regression_head_dropout_rate,
                    l1_lambda=self.regression_head_l1_lambda,  # Pass regularization
                    l2_lambda=self.regression_head_l2_lambda,  # Pass regularization
                    name="single_reg_head_final_output",
                )(head_input_probas, train=train)

                return {"output": final_predictions.squeeze(-1)}
            else:
                logger.error(f"Invalid regression_strategy: {self.regression_strategy.value}")
                raise ValueError("Invalid regression_strategy")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits the JAX probabilistic regression model.
        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training targets.
        """
        if self.input_size is None:
            self.input_size = X.shape[1]

        # Convert NumPy arrays to JAX arrays and reshape target for consistency
        X_jax, y_jax = jnp.array(X, dtype=jnp.float32), jnp.array(y, dtype=jnp.float32).reshape(-1, 1)

        # Extract parameters for MLP features and batch norm/dropout/regularization
        # These are nested in `base_classifier_params` and `regression_head_params`
        base_classifier_features = [self.base_classifier_params.get("hidden_size", 64)] * self.base_classifier_params.get("hidden_layers", 1)
        base_classifier_use_batch_norm = self.base_classifier_params.get("use_batch_norm", False)
        base_classifier_dropout_rate = self.base_classifier_params.get("dropout_rate", 0.0)
        base_classifier_l1_lambda = self.base_classifier_params.get("l1_lambda", 0.0)
        base_classifier_l2_lambda = self.base_classifier_params.get("l2_lambda", 0.0)

        regression_head_features = [self.regression_head_params.get("hidden_size", 32)] * self.regression_head_params.get("hidden_layers", 0)
        regression_head_use_batch_norm = self.regression_head_params.get("use_batch_norm", False)
        regression_head_dropout_rate = self.regression_head_params.get("dropout_rate", 0.0)
        regression_head_l1_lambda = self.regression_head_params.get("l1_lambda", 0.0)
        regression_head_l2_lambda = self.regression_head_params.get("l2_lambda", 0.0)

        self.model_def = self._CombinedJAXModel(
            classifier_features=base_classifier_features,
            classifier_use_batch_norm=base_classifier_use_batch_norm,
            classifier_dropout_rate=base_classifier_dropout_rate,
            n_classes=self.n_classes,
            regression_strategy=self.regression_strategy,
            regression_head_features=regression_head_features,
            regression_head_use_batch_norm=regression_head_use_batch_norm,
            regression_head_dropout_rate=regression_head_dropout_rate,
            uncertainty_method=self.uncertainty_method,
            classifier_l1_lambda=base_classifier_l1_lambda,
            classifier_l2_lambda=base_classifier_l2_lambda,
            regression_head_l1_lambda=regression_head_l1_lambda,
            regression_head_l2_lambda=regression_head_l2_lambda,
        )

        # Initialize the model's parameters and batch_stats
        rng_key, init_key = jax.random.split(self.key)
        # Pass `train=True` and a dropout key for proper initialization if dropout is used
        variables = self.model_def.init({"params": init_key, "dropout": init_key, "batch_stats": init_key}, X_jax[:1], train=True)
        self.params = variables["params"]
        self.batch_stats = variables["batch_stats"]  # Store initial batch stats

        # Initialize Optax optimizer
        optimizer = optax.adam(self.learning_rate)
        opt_state = optimizer.init(self.params)

        @jax.jit
        def loss_fn(params, batch_stats, rng, x, y, train):
            # Apply model to get predictions and potentially new batch stats
            variables_out = self.model_def.apply({"params": params, "dropout": rng, "batch_stats": batch_stats}, x, train=train, mutable=["batch_stats"])
            predictions_output = variables_out["output"]  # Output from the combined model
            new_batch_stats = variables_out.get("batch_stats", batch_stats)  # Update batch stats if mutable

            loss = None
            if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC:
                # Assuming predictions_output is (mean, log_var)
                mean = predictions_output[:, 0]
                log_var = predictions_output[:, 1]
                loss = 0.5 * jnp.mean(jnp.exp(-log_var) * (y - mean) ** 2 + log_var)
            else:
                loss = jnp.mean(jnp.square(predictions_output - y))

            # Add L1 and L2 regularization to the total loss
            l1_reg = 0.0
            l2_reg = 0.0
            # Iterate through parameters to apply regularization
            for name, param_collection in params.items():
                for sub_name, param_tree in param_collection.items():
                    if "kernel" in param_tree:  # Regularize only kernel weights
                        if "classifier_mlp" in name:  # Classifier part
                            l1_reg += self.base_classifier_params.get("l1_lambda", 0.0) * jnp.sum(jnp.abs(param_tree["kernel"]))
                            l2_reg += self.base_classifier_params.get("l2_lambda", 0.0) * jnp.sum(param_tree["kernel"] ** 2)
                        elif "reg_head" in name or "single_reg_head" in name:  # Regression head part
                            l1_reg += self.regression_head_params.get("l1_lambda", 0.0) * jnp.sum(jnp.abs(param_tree["kernel"]))
                            l2_reg += self.regression_head_params.get("l2_lambda", 0.0) * jnp.sum(param_tree["kernel"] ** 2)

            total_loss = loss + l1_reg + l2_reg
            return total_loss, new_batch_stats

        @jax.jit
        def train_step(params, batch_stats, opt_state, rng, x, y):
            (loss, new_batch_stats), grads = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(params, batch_stats, rng, x, y, train=True)
            updates, opt_state = optimizer.update(grads["params"], opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_batch_stats, opt_state, loss

        num_batches = int(np.ceil(len(X_jax) / self.batch_size))
        for epoch in range(self.n_epochs):
            self.key, shuffle_key = jax.random.split(self.key)
            # Shuffle data for each epoch
            permutation = jax.random.permutation(shuffle_key, len(X_jax))
            shuffled_X = X_jax[permutation]
            shuffled_y = y_jax[permutation]

            for i in range(num_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, len(X_jax))
                batch_X = shuffled_X[start_idx:end_idx]
                batch_y = shuffled_y[start_idx:end_idx]

                self.key, dropout_key_batch = jax.random.split(self.key)  # New dropout key for each batch

                self.params, self.batch_stats, opt_state, loss_value = train_step(self.params, self.batch_stats, opt_state, dropout_key_batch, batch_X, batch_y)
            # logger.info(f"JAX Probabilistic Model Epoch {epoch+1}/{self.n_epochs}, Batch Loss: {loss_value:.4f}") # For debugging

        # Calculate residual standard deviation for 'constant' uncertainty (if applicable)
        if self.uncertainty_method == UncertaintyMethod.CONSTANT:
            y_pred_train = self.predict(X)
            self._train_residual_std = np.std(y - y_pred_train)
            if np.isnan(self._train_residual_std):
                self._train_residual_std = 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions (mean only) from the JAX probabilistic model.
        Args:
            X (np.ndarray): Features for prediction.
        Returns:
            np.ndarray: Predicted mean values.
        """
        if self.params is None:
            raise RuntimeError("Model has not been fitted yet.")
        X_jax = jnp.array(X, dtype=jnp.float32)

        @jax.jit
        def predict_fn(params, batch_stats, rng, x, train_mode):
            # Apply model; `mutable=False` ensures batch_stats are not updated during prediction
            variables_out = self.model_def.apply({"params": params, "dropout": rng, "batch_stats": batch_stats}, x, train=train_mode, mutable=False)
            return variables_out["output"]

        if self.uncertainty_method == UncertaintyMethod.MC_DROPOUT:
            mc_predictions = []
            for _ in range(self.n_mc_dropout_samples):
                self.key, dropout_key_mc = jax.random.split(self.key)
                # Need to pass train=True to enable dropout during prediction for MC-Dropout
                outputs = predict_fn(self.params, self.batch_stats, dropout_key_mc, X_jax, True)  # train=True for MC-Dropout
                mc_predictions.append(np.array(outputs).flatten())
            return np.mean(mc_predictions, axis=0)

        # For 'constant' and 'probabilistic' uncertainty, run in eval mode (dropout off)
        self.key, dropout_key = jax.random.split(self.key)  # Get a new key for deterministic prediction if needed
        outputs = predict_fn(self.params, self.batch_stats, dropout_key, X_jax, False)  # train=False for deterministic

        if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC:
            predictions = outputs[:, 0]  # Return only the mean (first output)
        else:
            predictions = outputs  # For non-probabilistic, direct output
        return np.array(predictions).flatten()

    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """
        Estimates uncertainty (standard deviation) from the JAX probabilistic model's learned variance
        or using MC Dropout.
        Args:
            X (np.ndarray): Features for uncertainty estimation.
        Returns:
            np.ndarray: Predicted standard deviation values.
        """
        if not self._is_regression_model:
            raise ValueError("predict_uncertainty is only available for regression models.")
        if self.params is None:
            raise RuntimeError("Model has not been fitted yet.")
        X_jax = jnp.array(X, dtype=jnp.float32)

        @jax.jit
        def predict_fn_for_uncertainty(params, batch_stats, rng, x, train_mode):
            variables_out = self.model_def.apply({"params": params, "dropout": rng, "batch_stats": batch_stats}, x, train=train_mode, mutable=False)
            return variables_out["output"]

        if self.uncertainty_method == UncertaintyMethod.CONSTANT:
            return np.full(X.shape[0], self._train_residual_std)
        elif self.uncertainty_method == UncertaintyMethod.PROBABILISTIC:
            self.key, dropout_key_predict = jax.random.split(self.key)
            outputs = predict_fn_for_uncertainty(self.params, self.batch_stats, dropout_key_predict, X_jax, False)  # Eval mode (dropout off)
            log_var = outputs[:, 1]  # Second output is log_variance
            uncertainty = jnp.exp(0.5 * log_var)  # Convert log_variance to standard deviation
            return np.array(uncertainty).flatten()
        elif self.uncertainty_method == UncertaintyMethod.MC_DROPOUT:
            mc_predictions = []
            for _ in range(self.n_mc_dropout_samples):
                self.key, dropout_key_mc = jax.random.split(self.key)
                # Enable dropout for MC sampling (pass train=True)
                outputs = predict_fn_for_uncertainty(self.params, self.batch_stats, dropout_key_mc, X_jax, True)
                mc_predictions.append(np.array(outputs).flatten())  # Collect mean predictions
            return np.std(mc_predictions, axis=0)  # Return std dev of MC samples
        else:
            logger.error(f"Unknown uncertainty_method: {self.uncertainty_method.value}")
            raise ValueError(f"Unknown uncertainty_method: {self.uncertainty_method.value}")

    def get_hyperparameter_search_space(self) -> Dict[str, Any]:
        """
        Defines the hyperparameter search space for JAXProbabilisticRegressionModel.
        Returns:
            Dict[str, Any]: Search space configuration for Optuna.
        """
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
            "base_classifier_params__dropout_rate": {"type": "float", "low": 0.0, "high": 0.5, "step": 0.1},  # Dropout for base classifier
            # Nested parameters for the internal regression heads (JAX MLP)
            "regression_head_params__hidden_layers": {"type": "int", "low": 0, "high": 1},
            "regression_head_params__hidden_size": {"type": "int", "low": 16, "high": 32, "step": 16},
            "regression_head_params__use_batch_norm": {"type": "categorical", "choices": [True, False]},
            "regression_head_params__dropout_rate": {"type": "float", "low": 0.0, "high": 0.5, "step": 0.1},  # Dropout for regression heads
            "uncertainty_method": {"type": "categorical", "choices": [e.value for e in UncertaintyMethod]},
            "n_mc_dropout_samples": {"type": "int", "low": 50, "high": 200, "step": 50},  # Conditional sampling
            "dropout_rate": {"type": "float", "low": 0.1, "high": 0.5, "step": 0.1},  # Top-level dropout rate
        }

    def get_internal_model(self):
        """
        Returns the raw underlying Flax model definition and its parameters/batch_stats.
        """
        if self.model_def is None or self.params is None:
            raise RuntimeError("Model has not been fitted yet.")
        return {"model_def": self.model_def, "params": self.params, "batch_stats": self.batch_stats}
