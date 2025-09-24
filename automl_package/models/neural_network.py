# ruff: noqa: ERA001
"""Neural Network model implemented in PyTorch."""

from typing import Any, ClassVar

import torch
import torch.nn as nn

from automl_package.enums import ActivationFunction, ExplainerType, TaskType, UncertaintyMethod
from automl_package.models.base_pytorch import PyTorchModelBase
from automl_package.utils.losses import nll_loss
from automl_package.utils.pytorch_utils import get_activation_function_map


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
        activation: nn.Module,
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
        layers.append(activation())

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(activation())
            if is_regression and uncertainty_method == UncertaintyMethod.MC_DROPOUT and dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(hidden_size, current_output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class PyTorchNeuralNetwork(PyTorchModelBase):
    """A simple Feed-Forward Neural Network implemented using PyTorch.

    Can be configured with variable hidden layers and neurons.
    Supports optional Batch Normalization.
    Supports constant, MC-Dropout, and probabilistic layer uncertainty estimation for regression.
    Includes L1 and L2 regularization.
    """

    _defaults: ClassVar[dict[str, Any]] = {"hidden_layers": 1, "hidden_size": 64, "activation": ActivationFunction.RELU}

    def __init__(self, **kwargs: Any) -> None:
        """Initializes the PyTorchNeuralNetwork."""
        # Apply this class's defaults and then pass to the parent constructor
        for key, value in PyTorchNeuralNetwork._defaults.items():
            kwargs.setdefault(key, value)
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        """Returns the name of the model."""
        return "PyTorchNeuralNetwork"

    def build_model(self) -> None:
        """Dynamically builds the neural network architecture."""
        activation_function_map = get_activation_function_map()
        activation_module = activation_function_map.get(self.activation)
        if activation_module is None:
            raise ValueError(f"Unsupported activation function: {self.activation}")

        self.model = _PyTorchNNModule(
            input_size=self.input_size,
            output_size=self.output_size,
            hidden_layers=self.hidden_layers,
            hidden_size=self.hidden_size,
            use_batch_norm=self.use_batch_norm,
            dropout_rate=self.dropout_rate,
            is_regression=self.is_regression_model,
            uncertainty_method=self.uncertainty_method,
            activation=activation_module,
        ).to(self.device)

        # Define criterion based on task type and uncertainty method
        if self.is_regression_model:
            if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC:
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
                "hidden_size": {"type": "int", "low": 16, "high": 128, "step": 16},
                # "activation": {
                #     "type": "categorical",
                #     "choices": [e.value for e in ActivationFunction],
                # },
            }
        )
        if self.early_stopping_rounds is None:
            space["n_epochs"] = {"type": "int", "low": 5, "high": 50, "step": 10}
        if self.search_space_override:
            space.update(self.search_space_override)
        return space

    def get_params(self) -> dict[str, Any]:
        """Gets the parameters of the model."""
        params = super().get_params()
        for key in self._defaults:
            params[key] = getattr(self, key)
        return params

    def get_shap_explainer_info(self) -> dict[str, Any]:
        """Gets the SHAP explainer type and the model to be explained."""
        return {"explainer_type": ExplainerType.DEEP, "model": self.get_internal_model()}
