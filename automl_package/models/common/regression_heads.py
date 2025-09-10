"""Shared regression head modules for neural network-based models."""

import torch
import torch.nn as nn

from automl_package.enums import ActivationFunction, UncertaintyMethod
from automl_package.utils.pytorch_utils import get_activation_function_map


class BaseRegressionHead(nn.Module):
    """A single regression head.

    Handles its own layers and probabilistic output processing.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: int,
        hidden_size: int,
        use_batch_norm: bool,
        dropout_rate: float,
        uncertainty_method: UncertaintyMethod,
        activation: nn.Module,
    ) -> None:
        """Initializes the BaseRegressionHead.

        Args:
            input_size: The number of input features.
            output_size: The number of output features.
            hidden_layers: The number of hidden layers.
            hidden_size: The size of the hidden layers.
            use_batch_norm: Whether to use batch normalization.
            dropout_rate: The dropout rate.
            uncertainty_method: The uncertainty estimation method.
            activation: The activation function to use.
        """
        super().__init__()
        self.uncertainty_method = uncertainty_method
        self.output_size = output_size

        activation_map = get_activation_function_map()
        activation_module = activation_map.get(activation)
        if activation_module is None:
            raise ValueError(f"Unsupported activation function: {activation}")

        layers = [nn.Linear(input_size, hidden_size)]
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(activation_module())

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(activation())
            if dropout_rate > 0 and self.uncertainty_method == UncertaintyMethod.MC_DROPOUT:
                layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(hidden_size, output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass.

        Args:
            x: The input tensor.

        Returns:
            The output tensor.
        """
        return self.layers(x)


class SeparateHeadsRegressionModule(nn.Module):
    """Manages multiple BaseRegressionHead instances for the SEPARATE_HEADS strategy."""

    def __init__(
        self,
        n_classes: int,
        regression_head_params: dict,
        uncertainty_method: UncertaintyMethod,
        regression_output_size: int,
        activation: ActivationFunction = ActivationFunction.RELU,
    ) -> None:
        """Initializes the SeparateHeadsRegressionModule.

        Args:
            n_classes: The number of classes (and thus, heads).
            regression_head_params: Parameters for the regression heads.
            uncertainty_method: The uncertainty estimation method.
            regression_output_size: The output size for each regression head.
            activation: The activation function to use.
        """
        super().__init__()
        self.heads = nn.ModuleList()
        for _ in range(n_classes):
            self.heads.append(
                BaseRegressionHead(
                    input_size=1,
                    output_size=regression_output_size,
                    hidden_layers=regression_head_params.get("hidden_layers", 1),
                    hidden_size=regression_head_params.get("hidden_size", 32),
                    use_batch_norm=regression_head_params.get("use_batch_norm", True),
                    dropout_rate=regression_head_params.get("dropout_rate", 0.0),
                    uncertainty_method=uncertainty_method,
                    activation=activation,
                )
            )

    def forward(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass.

        Args:
            probabilities: A tensor of class probabilities.

        Returns:
            The final prediction, which is a weighted sum of the head outputs.
        """
        final_predictions = torch.zeros(probabilities.size(0), self.heads[0].output_size).to(probabilities.device)
        for i in range(len(self.heads)):
            p_i = probabilities[:, i].unsqueeze(1)
            y_i_processed = self.heads[i](p_i)
            final_predictions += p_i * y_i_processed
        return final_predictions

    def forward_per_class(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass and returns the output of each head."""
        head_outputs = []
        for i in range(len(self.heads)):
            p_i = probabilities[:, i].unsqueeze(1)
            head_outputs.append(self.heads[i](p_i))
        return torch.stack(head_outputs, dim=1)

    


class SingleHeadNOutputsRegressionModule(nn.Module):
    """Manages a single BaseRegressionHead instance for the SINGLE_HEAD_N_OUTPUTS strategy."""

    def __init__(
        self,
        input_size: int,
        n_classes: int,
        regression_head_params: dict,
        uncertainty_method: UncertaintyMethod,
        regression_output_size: int,
        activation: ActivationFunction = ActivationFunction.RELU,
    ) -> None:
        """Initializes the SingleHeadNOutputsRegressionModule.

        Args:
            input_size: The input size for the regression head.
            n_classes: The number of classes.
            regression_head_params: Parameters for the regression head.
            uncertainty_method: The uncertainty estimation method.
            regression_output_size: The output size for each class.
            activation: The activation function to use.
        """
        super().__init__()
        self.n_classes = n_classes
        self.regression_output_size = regression_output_size
        self.head = BaseRegressionHead(
            input_size=input_size,
            output_size=n_classes * regression_output_size,
            hidden_layers=regression_head_params.get("hidden_layers", 1),
            hidden_size=regression_head_params.get("hidden_size", 32),
            use_batch_norm=regression_head_params.get("use_batch_norm", True),
            dropout_rate=regression_head_params.get("dropout_rate", 0.0),
            uncertainty_method=uncertainty_method,
            activation=activation,
        )

    def forward(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass.

        Args:
            probabilities: A tensor of class probabilities.

        Returns:
            The final prediction, which is a weighted sum of the reshaped head outputs.
        """
        y_output_all_classes = self.head(probabilities)
        regression_outputs = y_output_all_classes.reshape(probabilities.shape[0], self.n_classes, self.regression_output_size)
        return torch.sum(probabilities.unsqueeze(-1) * regression_outputs, dim=1)

    def forward_per_class(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass and returns the reshaped output of the head."""
        y_output_all_classes = self.head(probabilities)
        return y_output_all_classes.reshape(probabilities.shape[0], self.n_classes, self.regression_output_size)


class SingleHeadFinalOutputRegressionModule(nn.Module):
    """Manages a single BaseRegressionHead instance for the SINGLE_HEAD_FINAL_OUTPUT strategy."""

    def __init__(self, input_size: int, regression_head_params: dict, uncertainty_method: UncertaintyMethod, regression_output_size: int) -> None:
        """Initializes the SingleHeadFinalOutputRegressionModule.

        Args:
            input_size: The input size for the regression head.
            regression_head_params: Parameters for the regression head.
            uncertainty_method: The uncertainty estimation method.
            regression_output_size: The final output size.
        """
        super().__init__()
        self.head = BaseRegressionHead(
            input_size=input_size,
            output_size=regression_output_size,
            hidden_layers=regression_head_params.get("hidden_layers", 1),
            hidden_size=regression_head_params.get("hidden_size", 32),
            use_batch_norm=regression_head_params.get("use_batch_norm", True),
            dropout_rate=regression_head_params.get("dropout_rate", 0.0),
            uncertainty_method=uncertainty_method,
            activation=regression_head_params.get("activation", ActivationFunction.RELU),
        )

    def forward(self, head_input_probas: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass.

        Args:
            head_input_probas: The input tensor of probabilities.

        Returns:
            The final prediction.
        """
        return self.head(head_input_probas)
