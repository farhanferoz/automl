"""This module defines the IndependentWeightsFlexibleNN model."""

import math
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn as nn

from automl_package.enums import (
    ActivationFunction,
    LayerSelectionMethod,
    Metric,
    TaskType,
    UncertaintyMethod,
)
from automl_package.logger import logger
from automl_package.models.base_pytorch import PyTorchModelBase
from automl_package.models.selection_strategies.independent_weights_strategies import (
    IndependentWeightsGumbelSoftmaxStrategy,
    IndependentWeightsNoneStrategy,
    IndependentWeightsReinforceStrategy,
    IndependentWeightsSoftGatingStrategy,
    IndependentWeightsSteStrategy,
)
from automl_package.utils.pytorch_utils import get_activation_function_map


class IndependentWeightsFlexibleNN(PyTorchModelBase):
    """A PyTorch Neural Network with a dynamic number of active hidden layers.

    where each possible number of layers has its own independent set of weights.

    It includes an internal feedforward network that predicts 'n' (1 to max_hidden_layers),
    where 'n' determines which of the independently trained networks (with 'n' layers) is used.
    Supports optional Batch Normalization and L1/L2 regularization.
    Supports constant, MC-Dropout, and probabilistic uncertainty estimation for regression.
    """

    _defaults: ClassVar[dict[str, Any]] = {
        "max_hidden_layers": 3,
        "hidden_size": 64,
        "activation": ActivationFunction.RELU,
        "gumbel_tau": 0.5,
        "n_predictor_layers": 1,
        "feature_scaler": None,
        "gumbel_tau_anneal_rate": 0.99,
        "n_predictor_learning_rate": 0.001,
        "layer_selection_method": LayerSelectionMethod.GUMBEL_SOFTMAX,
    }

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        """Initializes the IndependentWeightsFlexibleNN."""
        # Apply this class's defaults and then pass to the parent constructor
        for key, value in IndependentWeightsFlexibleNN._defaults.items():
            kwargs.setdefault(key, value)
        super().__init__(**kwargs)

        # Validation logic
        if self.layer_selection_method == LayerSelectionMethod.NONE and self.n_predictor_layers != 0:
            raise ValueError("n_predictor_layers must be 0 when layer_selection_method is NONE.")
        if (
            self.layer_selection_method
            in [
                LayerSelectionMethod.GUMBEL_SOFTMAX,
                LayerSelectionMethod.STE,
                LayerSelectionMethod.SOFT_GATING,
                LayerSelectionMethod.REINFORCE,
            ]
            and self.n_predictor_layers <= 0
        ):
            raise ValueError("n_predictor_layers must be > 0 for GUMBEL_SOFTMAX, STE, SOFT_GATING or REINFORCE methods.")

        strategy_map = {
            LayerSelectionMethod.NONE: IndependentWeightsNoneStrategy,
            LayerSelectionMethod.GUMBEL_SOFTMAX: IndependentWeightsGumbelSoftmaxStrategy,
            LayerSelectionMethod.SOFT_GATING: IndependentWeightsSoftGatingStrategy,
            LayerSelectionMethod.STE: IndependentWeightsSteStrategy,
            LayerSelectionMethod.REINFORCE: IndependentWeightsReinforceStrategy,
        }
        self.strategy = strategy_map[self.layer_selection_method](self)

    def _get_optimization_metric(self) -> Metric:
        """Gets the optimization metric for the model."""
        return Metric.RMSE if self.is_regression_model else Metric.ACCURACY

    @property
    def name(self) -> str:
        """Returns the name of the model."""
        return "IndependentWeightsFlexibleNN"

    class IndependentWeightsFlexibleNNModule(nn.Module):
        """Internal PyTorch nn.Module for IndependentWeightsFlexibleNN."""

        def __init__(self, outer_instance: Any) -> None:
            """Initializes the _IndependentWeightsFlexibleNN module."""
            super().__init__()
            self.outer = outer_instance

            # Output size for the main network's output layer
            final_output_neurons = self.outer.output_size
            if self.outer.task_type == TaskType.REGRESSION and self.outer.uncertainty_method == UncertaintyMethod.PROBABILISTIC:
                final_output_neurons = 2  # Mean and Log-Variance

            # n-predictor: Takes full input features and outputs logits for n (1 to max_hidden_layers)
            if self.outer.n_predictor_layers > 0:
                predictor_layers = []
                predictor_hidden_size = max(128, self.outer.hidden_size)
                in_features = self.outer.input_size
                predictor_layers.append(nn.Linear(in_features, predictor_hidden_size))
                predictor_layers.append(self.outer.activation())
                for _ in range(self.outer.n_predictor_layers - 1):
                    predictor_layers.append(nn.Linear(predictor_hidden_size, predictor_hidden_size))
                    predictor_layers.append(self.outer.activation())
                output_layer_predictor = nn.Linear(predictor_hidden_size, self.outer.max_hidden_layers)
                nn.init.normal_(output_layer_predictor.bias, mean=0.0, std=0.1)
                self.n_predictor = nn.Sequential(*predictor_layers, output_layer_predictor)
            else:
                self.n_predictor = None

            # Create independent networks for each possible number of hidden layers
            self.independent_networks = nn.ModuleList()
            for num_layers in range(1, self.outer.max_hidden_layers + 1):
                network_layers = []
                # Input layer
                network_layers.append(nn.Linear(self.outer.input_size, self.outer.hidden_size))
                if self.outer.use_batch_norm:
                    network_layers.append(nn.BatchNorm1d(self.outer.hidden_size))
                network_layers.append(self.outer.activation())
                if self.outer.task_type == TaskType.REGRESSION and self.outer.uncertainty_method == UncertaintyMethod.MC_DROPOUT and self.outer.dropout_rate > 0:
                    network_layers.append(nn.Dropout(self.outer.dropout_rate))

                # Hidden layers (num_layers - 1 because the first layer is already added)
                for _ in range(num_layers - 1):
                    network_layers.append(nn.Linear(self.outer.hidden_size, self.outer.hidden_size))
                    if self.outer.use_batch_norm:
                        network_layers.append(nn.BatchNorm1d(self.outer.hidden_size))
                    network_layers.append(self.outer.activation())
                    if self.outer.task_type == TaskType.REGRESSION and self.outer.uncertainty_method == UncertaintyMethod.MC_DROPOUT and self.outer.dropout_rate > 0:
                        network_layers.append(nn.Dropout(self.outer.dropout_rate))

                # Output layer
                network_layers.append(nn.Linear(self.outer.hidden_size, final_output_neurons))
                self.independent_networks.append(nn.Sequential(*network_layers))

        def forward(self, x_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """Forward pass for the independent weights flexible neural network."""
            n_logits = self.n_predictor(x_input) if self.n_predictor else None

            # The strategy now only returns selection info, not processed x
            # The strategy's forward method is designed to return x_input, n_actual, n_probs, n_logits_out, log_prob
            # where x_input is the original input, as the strategy does not apply layers.
            (
                x_input_from_strategy,
                n_actual_tensor,
                n_probs_tensor,
                n_logits_tensor_from_strategy,
                log_prob,
            ) = self.outer.strategy.forward(x_input, n_logits)

            # --- Soft Selection (Weighted Sum) ---
            # Collect outputs from all independent networks
            all_network_outputs = []
            for network in self.independent_networks:
                all_network_outputs.append(network(x_input))  # Output shape: (batch_size, output_dim)

            # Stack them to form (batch_size, num_networks, output_dim)
            all_network_outputs_stacked = torch.stack(all_network_outputs, dim=1)

            # Expand n_probs_tensor to match output_dim for broadcasting
            # n_probs_tensor shape: (batch_size, num_networks)
            # Desired shape for multiplication: (batch_size, num_networks, 1)
            n_probs_expanded = n_probs_tensor.unsqueeze(-1)

            # Perform weighted sum: (batch_size, num_networks, output_dim) * (batch_size, num_networks, 1)

            weighted_outputs = all_network_outputs_stacked * n_probs_expanded

            # Sum over the num_networks dimension to get final output: (batch_size, output_dim)
            final_output = torch.sum(weighted_outputs, dim=1)
            # --- End Soft Selection ---

            return (
                final_output,
                n_actual_tensor,
                n_probs_tensor,
                n_logits_tensor_from_strategy,
                log_prob,
            )

    def build_model(self) -> None:
        """Builds the internal PyTorch nn.Module for the IndependentWeightsFlexibleNN."""
        activation_function_map = get_activation_function_map()
        self.activation = activation_function_map.get(self.activation)
        if self.activation is None:
            raise ValueError(f"Unsupported activation function: {self.activation}")

        self.model = self.IndependentWeightsFlexibleNNModule(self).to(self.device)
        self.model = self.IndependentWeightsFlexibleNNModule(self).to(self.device)

        if self.task_type == TaskType.REGRESSION:
            if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC:

                def nll_loss(final_output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
                    mean = final_output[:, 0]
                    log_var = final_output[:, 1]
                    targets = targets.squeeze(-1) if targets.ndim > 1 else targets
                    per_sample_nll = 0.5 * (torch.log(torch.tensor(2 * math.pi)) + log_var + (targets - mean) ** 2 / torch.exp(log_var))
                    return torch.mean(per_sample_nll)

                self.criterion = nll_loss
            else:
                self.criterion = nn.MSELoss()
        elif self.task_type == TaskType.CLASSIFICATION:
            if self.output_size == 1:
                self.criterion = nn.BCEWithLogitsLoss()
            else:
                self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError("task_type must be 'regression' or 'classification'")

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
        if self.input_size is None:
            self.input_size = 1 if x_train.ndim == 1 else x_train.shape[1]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.learn_regularization_lambdas:
            if self.using_l1_regularization:
                self.l1_log_lambda = nn.Parameter(torch.tensor(np.log(1e-4), dtype=torch.float32))
            if self.using_l2_regularization:
                self.l2_log_lambda = nn.Parameter(torch.tensor(np.log(1e-4), dtype=torch.float32))

        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_seed)

        self.build_model()
        self._setup_optimizers(self.model)

        x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        if self.task_type == TaskType.CLASSIFICATION:
            y_train_tensor = y_train_tensor.long()
        if (self.task_type == TaskType.REGRESSION) or (self.task_type == TaskType.CLASSIFICATION and self.output_size == 1):
            y_train_tensor = y_train_tensor.unsqueeze(1)

        train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        x_val_tensor, y_val_tensor = None, None
        if x_val is not None:
            x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(self.device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(self.device)
            if self.task_type == TaskType.CLASSIFICATION:
                y_val_tensor = y_val_tensor.long()
            if (self.task_type == TaskType.REGRESSION) or (self.task_type == TaskType.CLASSIFICATION and self.output_size == 1):
                y_val_tensor = y_val_tensor.unsqueeze(1)

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None
        best_epoch = 0
        val_loss_history = []

        n_epochs = forced_iterations if forced_iterations is not None else self.n_epochs

        for epoch in range(int(n_epochs)):
            self.model.train()
            epoch_log_probs = []
            for _batch_x, batch_y in train_dataloader:
                self.optimizer.zero_grad()
                if self.strategy.policy_optimizer:
                    self.strategy.policy_optimizer.zero_grad()
                if self.lambda_optimizer:
                    self.lambda_optimizer.zero_grad()

                final_output, _, _, _, log_prob = self.model(_batch_x)

                # Calculate main loss
                main_loss = self.criterion(final_output, batch_y)
                main_loss = self._calculate_regularization_loss(main_loss, self.model)

                if self.layer_selection_method == LayerSelectionMethod.REINFORCE and log_prob is not None:
                    # Calculate policy loss (REINFORCE)
                    # The reward signal is typically the negative of the validation loss,
                    # but for batch-wise updates, we can use the negative of the current batch's main_loss
                    # or a baseline. For simplicity, let's use negative main_loss as a reward.
                    # We need to ensure the reward is detached to prevent gradients flowing back through it
                    # to the main network parameters.
                    reward = -main_loss.detach()
                    policy_loss = -log_prob * reward  # Multiply by reward

                    # Combine losses
                    # The policy loss is typically added to the main loss.
                    # The weighting of policy_loss can be a hyperparameter.
                    # For now, let's add it directly.
                    loss = main_loss + policy_loss.mean()  # Use .mean() if log_prob is per-sample
                else:
                    loss = main_loss

                loss.backward()

                self.optimizer.step()
                if self.strategy.policy_optimizer:
                    self.strategy.policy_optimizer.step()
                if self.lambda_optimizer:
                    self.lambda_optimizer.step()

            if x_val_tensor is not None and y_val_tensor is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs, _, _, _, _ = self.model(x_val_tensor)
                    val_loss = self.criterion(val_outputs, y_val_tensor).item()
                val_loss_history.append(val_loss)

                # Pass collected epoch_log_probs
                self.strategy.on_epoch_end(validation_loss=val_loss, epoch_log_probs=epoch_log_probs)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict()
                    best_epoch = epoch
                else:
                    patience_counter += 1

                if self.early_stopping_rounds and patience_counter >= self.early_stopping_rounds:
                    logger.info(f"Early stopping at epoch {best_epoch + 1}")
                    break
            else:
                best_epoch = epoch

        if best_model_state and x_val_tensor is not None:
            self.model.load_state_dict(best_model_state)

        if self.is_regression_model and self.uncertainty_method == UncertaintyMethod.CONSTANT:
            y_pred_train = self.predict(x_train)
            self._train_residual_std = np.std(y_train - y_pred_train)
            if np.isnan(self._train_residual_std):
                self._train_residual_std = 0.0

        if self.learn_regularization_lambdas:
            if self.l1_log_lambda is not None:
                logger.info(f"Learned L1 Lambda: {torch.exp(self.l1_log_lambda).item():.6f}")
            if self.l2_log_lambda is not None:
                logger.info(f"Learned L2 Lambda: {torch.exp(self.l2_log_lambda).item():.6f}")

        return best_epoch + 1, val_loss_history

    def get_params(self) -> dict[str, Any]:
        """Gets the parameters of the model."""
        params = super().get_params()
        for key in self._defaults:
            params[key] = getattr(self, key)
        return params

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Makes predictions with the IndependentWeightsFlexibleNN."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        if self.is_regression_model and self.uncertainty_method == UncertaintyMethod.MC_DROPOUT:
            self.model.train()
            mc_predictions = []
            with torch.no_grad():
                final_output_list = []
                # Need to get n_actual for each sample for MC Dropout
                n_logits = self.model.n_predictor(x_tensor) if self.model.n_predictor else None
                _, n_actual_tensor, _, _, _ = self.strategy.forward(x_tensor, n_logits)
                selected_network_indices = (n_actual_tensor - 1).long()

                for _ in range(self.n_mc_dropout_samples):
                    current_mc_output_list = []
                    for i in range(x_tensor.shape[0]):
                        idx = selected_network_indices[i].item()
                        selected_network = self.model.independent_networks[idx]
                        current_mc_output_list.append(selected_network(x_tensor[i].unsqueeze(0)))
                    mc_predictions.append(torch.cat(current_mc_output_list, dim=0).cpu().numpy().flatten())
            self.model.eval()  # Set back to eval mode
            return np.mean(mc_predictions, axis=0)

        self.model.eval()
        with torch.no_grad():
            # Need to get n_actual for each sample
            n_logits = self.model.n_predictor(x_tensor) if self.model.n_predictor else None
            _, n_actual_tensor, _, _, _ = self.strategy.forward(x_tensor, n_logits)
            selected_network_indices = (n_actual_tensor - 1).long()

            final_output_list = []
            for i in range(x_tensor.shape[0]):
                idx = selected_network_indices[i].item()
                selected_network = self.model.independent_networks[idx]
                final_output_list.append(selected_network(x_tensor[i].unsqueeze(0)))

            final_output = torch.cat(final_output_list, dim=0)

            if self.task_type == TaskType.CLASSIFICATION:
                predictions = (torch.sigmoid(final_output) > 0.5).cpu().numpy().astype(int) if self.output_size == 1 else torch.argmax(final_output, dim=1).cpu().numpy()
            else:
                predictions = final_output[:, 0].cpu().numpy() if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC else final_output.cpu().numpy()
        return predictions.flatten()

    def predict_uncertainty(self, x: np.ndarray) -> np.ndarray:
        """Estimates uncertainty for regression."""
        if not self.is_regression_model:
            raise ValueError("predict_uncertainty is only available for regression models.")
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        if self.uncertainty_method == UncertaintyMethod.CONSTANT:
            return np.full(x.shape[0], self._train_residual_std)
        if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC:
            self.model.eval()  # Use eval mode for prediction
            with torch.no_grad():
                # Need to get n_actual for each sample
                n_logits = self.model.n_predictor(x_tensor) if self.model.n_predictor else None
                _, n_actual_tensor, _, _, _ = self.strategy.forward(x_tensor, n_logits)
                selected_network_indices = (n_actual_tensor - 1).long()

                final_output_list = []
                for i in range(x_tensor.shape[0]):
                    idx = selected_network_indices[i].item()
                    selected_network = self.model.independent_networks[idx]
                    final_output_list.append(selected_network(x_tensor[i].unsqueeze(0)))
                final_output = torch.cat(final_output_list, dim=0)

                log_var = final_output[:, 1]
                uncertainty = torch.exp(0.5 * log_var).cpu().numpy()
            return uncertainty.flatten()
        if self.uncertainty_method == UncertaintyMethod.MC_DROPOUT:
            self.model.train()  # Enable dropout during inference for MC Dropout
            mc_predictions = []
            with torch.no_grad():
                # Need to get n_actual for each sample
                n_logits = self.model.n_predictor(x_tensor) if self.model.n_predictor else None
                _, n_actual_tensor, _, _, _ = self.strategy.forward(x_tensor, n_logits)
                selected_network_indices = (n_actual_tensor - 1).long()

                for _ in range(self.n_mc_dropout_samples):
                    current_mc_output_list = []
                    for i in range(x_tensor.shape[0]):
                        idx = selected_network_indices[i].item()
                        selected_network = self.model.independent_networks[idx]
                        current_mc_output_list.append(selected_network(x_tensor[i].unsqueeze(0)))
                    mc_predictions.append(torch.cat(current_mc_output_list, dim=0).cpu().numpy().flatten())
            self.model.eval()  # Set back to eval mode
            return np.std(mc_predictions, axis=0)  # Return std dev of MC samples
        raise ValueError(f"Unknown uncertainty_method: {self.uncertainty_method.value}")

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predicts class probabilities for classification tasks."""
        if self.task_type != TaskType.CLASSIFICATION:
            raise ValueError("predict_proba is only available for classification tasks.")
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")

        self.model.eval()
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            # Need to get n_actual for each sample
            n_logits = self.model.n_predictor(x_tensor) if self.model.n_predictor else None
            _, n_actual_tensor, _, _, _ = self.strategy.forward(x_tensor, n_logits)
            selected_network_indices = (n_actual_tensor - 1).long()

            final_output_list = []
            for i in range(x_tensor.shape[0]):
                idx = selected_network_indices[i].item()
                selected_network = self.model.independent_networks[idx]
                final_output_list.append(selected_network(x_tensor[i].unsqueeze(0)))
            final_output = torch.cat(final_output_list, dim=0)

            if self.output_size == 1:
                proba = torch.sigmoid(final_output).cpu().numpy().flatten()
                return np.vstack((1 - proba, proba)).T
            return torch.softmax(final_output, dim=1).cpu().numpy()

    def get_hyperparameter_search_space(self) -> dict[str, Any]:
        """Defines the hyperparameter search space for IndependentWeightsFlexibleNN."""
        space = super().get_hyperparameter_search_space()
        space.update(
            {
                "max_hidden_layers": {"type": "int", "low": 1, "high": 3},
                "hidden_size": {"type": "int", "low": 32, "high": 128, "step": 32},
                "activation": {
                    "type": "categorical",
                    "choices": [e.value for e in ActivationFunction],
                },
                "gumbel_tau": {"type": "float", "low": 1e-8, "high": 1.0, "log": True},
                "n_predictor_layers": {"type": "int", "low": 0, "high": 2},
                "n_predictor_learning_rate": {
                    "type": "float",
                    "low": 1e-8,
                    "high": 1e-2,
                    "log": True,
                },
                "layer_selection_method": {
                    "type": "categorical",
                    "choices": [LayerSelectionMethod.GUMBEL_SOFTMAX],
                },
            }
        )

        if "hidden_layers" in space:
            del space["hidden_layers"]

        if self.search_space_override:
            space.update(self.search_space_override)

        return space

    def _setup_optimizers(self, model: torch.nn.Module) -> None:
        """Sets up the optimizers for IndependentWeightsFlexibleNN, handling n_predictor separately."""
        # Collect parameters for the main networks (all independent_networks)
        main_params = []
        for net in model.independent_networks:
            main_params.extend(list(net.parameters()))

        self.optimizer = torch.optim.Adam(main_params, lr=self.learning_rate)

        if model.n_predictor:
            policy_params = model.n_predictor.parameters()
            self.strategy.setup_optimizers(policy_params)
