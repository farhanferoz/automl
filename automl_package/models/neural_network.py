"""Neural Network model implemented in PyTorch."""

import math
from typing import Any

import torch
import torch.nn as nn

from automl_package.enums import TaskType, UncertaintyMethod

from .base_pytorch import PyTorchModelBase


class _PyTorchNNModule(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: int,
        hidden_size: int,
        use_batch_norm: bool,
        dropout_rate: float,
        is_regression: bool,
        uncertainty_method: UncertaintyMethod,
    ) -> None:
        super().__init__()
        self.is_regression = is_regression
        self.uncertainty_method = uncertainty_method

        layers = []
        current_output_size = output_size
        if is_regression and uncertainty_method == UncertaintyMethod.PROBABILISTIC:
            current_output_size = 2

        layers.append(nn.Linear(input_size, hidden_size))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(nn.ReLU())

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            if is_regression and uncertainty_method == UncertaintyMethod.MC_DROPOUT and dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(hidden_size, current_output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        predictions = self.layers(x)
        return predictions, None, None, None


class PyTorchNeuralNetwork(PyTorchModelBase):
    """A simple Feed-Forward Neural Network implemented using PyTorch.

    Can be configured with variable hidden layers and neurons.
    Supports optional Batch Normalization.
    Supports constant, MC-Dropout, and probabilistic layer uncertainty estimation for regression.
    Includes L1 and L2 regularization.
    """

    def __init__(
        self,
        hidden_layers: int = 1,
        hidden_size: int = 64,
        activation: Any = nn.ReLU,
        **kwargs: Any,
    ) -> None:
        """Initializes the PyTorchNeuralNetwork.

        Args:
            hidden_layers (int): Number of hidden layers.
            hidden_size (int): Number of neurons in each hidden layer.
            activation (Any): Activation function to use.
            **kwargs: Additional keyword arguments for PyTorchModelBase.
        """
        super().__init__(**kwargs)
        self._returns_multiple_outputs = True  # This model returns multiple outputs
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.activation = activation

    @property
    def name(self) -> str:
        """Returns the name of the model."""
        return "PyTorchNeuralNetwork"

    def build_model(self) -> None:
        """Dynamically builds the neural network architecture."""
        self.model = _PyTorchNNModule(
            input_size=self.input_size,
            output_size=self.output_size,
            hidden_layers=self.hidden_layers,
            hidden_size=self.hidden_size,
            use_batch_norm=self.use_batch_norm,
            dropout_rate=self.dropout_rate,
            is_regression=self.is_regression_model,
            uncertainty_method=self.uncertainty_method,
        ).to(self.device)

        # Define criterion based on task type and uncertainty method
        if self.task_type == TaskType.REGRESSION:
            if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC:
                # Custom Negative Log-Likelihood Loss for Gaussian output
                def nll_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
                    mean = outputs[:, 0]
                    log_var = outputs[:, 1]
                    # Ensure targets has the same shape as mean for element-wise operations
                    targets = targets.squeeze(-1) if targets.ndim > 1 else targets
                    # Calculate per-sample NLL
                    per_sample_nll = 0.5 * (math.log(2 * math.pi) + log_var + (targets - mean) ** 2 / torch.exp(log_var))
                    # Average over the batch
                    return torch.mean(per_sample_nll)

                self.criterion = nll_loss
            else:  # Standard MSE loss for other regression methods
                self.criterion = nn.MSELoss()
        elif self.task_type == TaskType.CLASSIFICATION:
            if self.output_size == 1:  # Binary classification (e.g., outputs logits for BCEWithLogitsLoss)
                self.criterion = nn.BCEWithLogitsLoss()
            else:  # Multi-class classification
                self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError("task_type must be 'regression' or 'classification'")

    def get_hyperparameter_search_space(self) -> dict[str, Any]:
        """Defines the hyperparameter search space for PyTorchNeuralNetwork."""
        space = super().get_hyperparameter_search_space()
        space.update(
            {
                "hidden_layers": {"type": "int", "low": 1, "high": 3},
                "hidden_size": {"type": "int", "low": 32, "high": 128, "step": 32},
                "activation": {"type": "categorical", "choices": ["ReLU", "Tanh"]},
            }
        )
        return space
