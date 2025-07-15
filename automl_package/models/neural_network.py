import math
from typing import Any, Dict
import torch
import torch.nn as nn

from .base_pytorch import PyTorchModelBase
from ..enums import UncertaintyMethod, TaskType


class PyTorchNeuralNetwork(PyTorchModelBase):
    """
    A simple Feed-Forward Neural Network implemented using PyTorch.
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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.activation = activation

    @property
    def name(self) -> str:
        return "PyTorchNeuralNetwork"

    def build_model(self):
        """Dynamically builds the neural network architecture."""
        layers = []
        current_output_size = self.output_size

        if self.is_regression_model and self.uncertainty_method == UncertaintyMethod.PROBABILISTIC:
            # For probabilistic regression, output 2 values: mean and log-variance
            current_output_size = 2

        # Input layer
        layers.append(nn.Linear(self.input_size, self.hidden_size))
        if self.use_batch_norm:
            layers.append(nn.BatchNorm1d(self.hidden_size))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(self.hidden_layers - 1):  # -1 because input layer is already counted
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(self.hidden_size))
            layers.append(nn.ReLU())
            # Add dropout for MC-Dropout if regression and uncertainty method is MC_DROPOUT and dropout_rate > 0
            if self.is_regression_model and self.uncertainty_method == UncertaintyMethod.MC_DROPOUT and self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))

        # Output layer
        layers.append(nn.Linear(self.hidden_size, current_output_size))

        self.model = nn.Sequential(*layers).to(self.device)

        # Define criterion based on task type and uncertainty method
        if self.task_type == TaskType.REGRESSION:
            if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC:
                # Custom Negative Log-Likelihood Loss for Gaussian output
                def nll_loss(outputs, targets):
                    mean = outputs[:, 0]
                    log_var = outputs[:, 1]
                    # Ensure targets has the same shape as mean for element-wise operations
                    targets = targets.squeeze(-1) if targets.ndim > 1 else targets
                    # Calculate per-sample NLL
                    per_sample_nll = 0.5 * (math.log(2 * math.pi) + log_var + (targets - mean) ** 2 / torch.exp(log_var))
                    # Average over the batch
                    loss = torch.mean(per_sample_nll)
                    return loss

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

    def get_hyperparameter_search_space(self) -> Dict[str, Any]:
        """
        Defines the hyperparameter search space for PyTorchNeuralNetwork.
        """
        space = super().get_hyperparameter_search_space()
        space.update({
            "hidden_layers": {"type": "int", "low": 1, "high": 3},
            "hidden_size": {"type": "int", "low": 32, "high": 128, "step": 32},
            "activation": {"type": "categorical", "choices": ["ReLU", "Tanh"]},
        })
        return space

    
