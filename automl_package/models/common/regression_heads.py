"""Shared regression head modules for neural network-based models."""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as f

from automl_package.enums import ActivationFunction, Monotonicity, UncertaintyMethod
from automl_package.utils.pytorch_utils import get_activation_function_map, monotonic_linear


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
        monotonic_constraint: Monotonicity = Monotonicity.NONE,
        use_logit_initialization: bool = True,
    ) -> None:
        """Initializes the BaseRegressionHead."""
        super().__init__()
        self.uncertainty_method = uncertainty_method
        self.output_size = output_size
        self.monotonic_constraint = monotonic_constraint
        self.use_logit_initialization = use_logit_initialization

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

        final_layer = nn.Linear(hidden_size, output_size)
        if self.use_logit_initialization:
            if self.monotonic_constraint != Monotonicity.NONE:
                # Initialize weights to be small negative numbers, so that softplus brings them close to zero
                nn.init.normal_(final_layer.weight, mean=-3.0, std=0.1)
                nn.init.constant_(final_layer.bias, 0.0)
            else:
                nn.init.xavier_uniform_(final_layer.weight, gain=nn.init.calculate_gain("linear"))
                nn.init.constant_(final_layer.bias, 0.0)

        layers.append(final_layer)
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, boundaries: torch.Tensor | None = None) -> torch.Tensor:
        """Performs the forward pass."""
        logit = x
        for layer in self.layers:
            logit = monotonic_linear(logit, layer, self.monotonic_constraint) if isinstance(layer, nn.Linear) else layer(logit)

        if self.monotonic_constraint == Monotonicity.NEGATIVE:
            logit = -logit

        if boundaries is not None:
            lower_bound = boundaries[:, 0].unsqueeze(-1)
            upper_bound = boundaries[:, 1].unsqueeze(-1)
            # Transform the unbounded logit to be within the desired [lower, upper] range.
            return lower_bound + (upper_bound - lower_bound) * f.hardsigmoid(logit)
        return logit


class ConstantHead(nn.Module):
    """A head that learns and returns a constant value."""

    def __init__(self, uncertainty_method: UncertaintyMethod, regression_output_size: int) -> None:
        """Initializes the ConstantHead."""
        super().__init__()
        self.regression_output_size = regression_output_size
        self.mean = nn.Parameter(torch.randn(1))
        self.monotonic_constraint = Monotonicity.NONE
        if uncertainty_method == UncertaintyMethod.PROBABILISTIC:
            self.log_variance = nn.Parameter(torch.randn(1))

    def forward(self, x: torch.Tensor, boundaries: torch.Tensor | None = None) -> torch.Tensor:
        """Returns the learned constant value, expanded to the batch size."""
        batch_size = x.size(0)
        output_mean = self.mean.expand(batch_size, 1)
        if self.regression_output_size == 2:
            output_log_var = self.log_variance.expand(batch_size, 1)
            return torch.cat([output_mean, output_log_var], dim=1)
        return output_mean


class SeparateHeadsRegressionModule(nn.Module):
    """Manages multiple BaseRegressionHead instances for the SEPARATE_HEADS strategy."""

    def __init__(
        self,
        n_classes: int,
        regression_head_params: dict,
        uncertainty_method: UncertaintyMethod,
        regression_output_size: int,
        activation: ActivationFunction = ActivationFunction.RELU,
        use_monotonic_constraints: bool = False,
        constrain_middle_class: bool = True,
    ) -> None:
        """Initializes the SeparateHeadsRegressionModule."""
        super().__init__()
        self.heads = nn.ModuleList()
        self.n_classes = n_classes
        self.regression_output_size = regression_output_size

        for i in range(n_classes):
            middle_point = (n_classes - 1) / 2.0
            if (i == middle_point) and constrain_middle_class:
                self.heads.append(ConstantHead(uncertainty_method, regression_output_size))
            else:
                slope_type = Monotonicity.NONE
                if use_monotonic_constraints:
                    if i < middle_point:
                        slope_type = Monotonicity.NEGATIVE
                    elif i > middle_point:
                        slope_type = Monotonicity.POSITIVE

                self.heads.append(
                    BaseRegressionHead(
                        input_size=1,
                        output_size=regression_output_size,
                        hidden_layers=regression_head_params.get("hidden_layers", 1),
                        hidden_size=regression_head_params.get("hidden_size", 32),
                        use_batch_norm=regression_head_params.get("use_batch_norm", False),
                        dropout_rate=regression_head_params.get("dropout_rate", 0.0),
                        uncertainty_method=uncertainty_method,
                        activation=activation,
                        monotonic_constraint=slope_type,
                    )
                )

    def forward(self, probabilities: torch.Tensor, return_head_outputs: bool = False, boundaries: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Performs the forward pass.

        Args:
            probabilities: A tensor of class probabilities.
            return_head_outputs: If True, return per-head outputs alongside the final prediction.
            boundaries: Optional boundaries for the sigmoid transformation.

        Returns:
            The final prediction, or a tuple of (final_prediction, per_head_outputs).
        """
        final_predictions = torch.zeros(probabilities.size(0), self.heads[0].output_size, device=probabilities.device)
        per_head_outputs = []
        for i in range(len(self.heads)):
            p_i = probabilities[:, i].unsqueeze(1)
            y_i_processed = self.heads[i](p_i) if isinstance(self.heads[i], ConstantHead) else self.heads[i](p_i, boundaries=boundaries)
            final_predictions += p_i * y_i_processed
            per_head_outputs.append(y_i_processed)

        per_head_outputs = torch.stack(per_head_outputs, dim=1)

        return final_predictions, per_head_outputs if return_head_outputs else final_predictions

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
            use_batch_norm=regression_head_params.get("use_batch_norm", False),
            dropout_rate=regression_head_params.get("dropout_rate", 0.0),
            uncertainty_method=uncertainty_method,
            activation=activation,
        )

    def forward(self, probabilities: torch.Tensor, return_head_outputs: bool = False, boundaries: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Performs the forward pass.

        Args:
            probabilities: A tensor of class probabilities.
            return_head_outputs: If True, return per-head outputs alongside the final prediction.
            boundaries: Optional boundaries for the sigmoid transformation.

        Returns:
            The final prediction, or a tuple of (final_prediction, per_head_outputs).
        """
        y_output_all_classes = self.head(probabilities, boundaries=boundaries)
        per_head_outputs = y_output_all_classes.reshape(probabilities.shape[0], self.n_classes, self.regression_output_size)
        final_predictions = torch.sum(probabilities.unsqueeze(-1) * per_head_outputs, dim=1)

        return final_predictions, per_head_outputs if return_head_outputs else final_predictions

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
            use_batch_norm=regression_head_params.get("use_batch_norm", False),
            dropout_rate=regression_head_params.get("dropout_rate", 0.0),
            uncertainty_method=uncertainty_method,
            activation=regression_head_params.get("activation", ActivationFunction.RELU),
        )

    def forward(self, head_input_probas: torch.Tensor, boundaries: torch.Tensor | None = None, **_kwargs: Any) -> torch.Tensor:
        """Performs the forward pass.

        Args:
            head_input_probas: The input tensor of probabilities.
            boundaries: Optional boundaries for the sigmoid transformation.

        Returns:
            The final prediction.
        """
        return self.head(head_input_probas, boundaries=boundaries)
