"""Shared regression head modules for neural network-based models."""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as f

from automl_package.enums import ActivationFunction, Monotonicity, UncertaintyMethod
from automl_package.utils.pytorch_utils import apply_law_of_total_variance, get_activation_function_map, monotonic_linear


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
        self.is_probabilistic_monotonic = self.uncertainty_method == UncertaintyMethod.PROBABILISTIC and self.monotonic_constraint != Monotonicity.NONE

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

        if self.is_probabilistic_monotonic:
            # Special case: Separate heads for mean and variance
            self.mean_head = nn.Linear(hidden_size, 1)
            self.variance_head = nn.Linear(hidden_size, 1)
            if self.use_logit_initialization:
                self._initialize_monotonic_head(self.mean_head)
                self._initialize_standard_head(self.variance_head)
        else:
            # Standard case: Single final layer
            final_layer = nn.Linear(hidden_size, output_size)
            if self.use_logit_initialization:
                if self.monotonic_constraint != Monotonicity.NONE:
                    self._initialize_monotonic_head(final_layer)
                else:
                    self._initialize_standard_head(final_layer)
            layers.append(final_layer)

        self.layers = nn.ModuleList(layers)

    def _initialize_monotonic_head(self, layer: nn.Linear) -> None:
        """Applies special initialization for monotonic heads."""
        nn.init.normal_(layer.weight, mean=-3.0, std=0.1)
        nn.init.constant_(layer.bias, 0.0)

    def _initialize_standard_head(self, layer: nn.Linear) -> None:
        """Applies standard Xavier uniform initialization."""
        nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("linear"))
        nn.init.constant_(layer.bias, 0.0)

    def _apply_boundary_constraints(self, logit: torch.Tensor, boundaries: torch.Tensor) -> torch.Tensor:
        """Applies hardsigmoid transformation to constrain logits within boundaries."""
        # Handle probabilistic case by separating mean and log_var
        if self.output_size == 2:  # Probabilistic
            # De-interleave mean and log_var
            # logit shape is (batch, n_classes * 2) for SingleHeadNOutputs, or (batch, 2)
            means = logit[:, 0::2]
            log_vars = logit[:, 1::2]
        else:  # Non-probabilistic
            means = logit
            log_vars = None

        # Now, apply boundaries ONLY to the means
        if means.dim() > 1 and means.shape[1] > 1 and boundaries.dim() > 2 and boundaries.shape[1] == means.shape[1]:
            # Per-output boundaries for SingleHeadNOutputs
            lower_bound = boundaries[..., 0]
            upper_bound = boundaries[..., 1]
        else:
            # Standard boundaries
            lower_bound = boundaries[:, 0].unsqueeze(-1)
            upper_bound = boundaries[:, 1].unsqueeze(-1)

        constrained_means = lower_bound + (upper_bound - lower_bound) * f.hardsigmoid(means)

        # Re-interleave if necessary
        if log_vars is not None:
            # Create a new tensor to hold the results
            output = torch.zeros_like(logit)
            output[:, 0::2] = constrained_means
            output[:, 1::2] = log_vars
            return output
        else:
            return constrained_means

    def forward(self, x: torch.Tensor, boundaries: torch.Tensor | None = None) -> torch.Tensor:
        """Performs the forward pass."""

        # Determine which layers are the shared layers
        shared_layers = self.layers if self.is_probabilistic_monotonic else self.layers[:-1]

        hidden_output = x
        for layer in shared_layers:
            hidden_output = layer(hidden_output)

        if self.is_probabilistic_monotonic:
            # --- Path for Probabilistic Monotonic ---
            mean_logit = monotonic_linear(hidden_output, self.mean_head, self.monotonic_constraint)
            log_var_logit = self.variance_head(hidden_output)

            # The boundary logic for the mean is applied here for this specific path
            if boundaries is not None:
                mean_logit = self._apply_boundary_constraints(logit=mean_logit, boundaries=boundaries)

            output = torch.cat([mean_logit, log_var_logit], dim=1)
        else:
            # --- Path for Standard and Non-Probabilistic Monotonic ---
            final_layer = self.layers[-1]

            if self.monotonic_constraint != Monotonicity.NONE:
                logit = monotonic_linear(hidden_output, final_layer, self.monotonic_constraint)
            else:
                logit = final_layer(hidden_output)

            output = logit if boundaries is None else self._apply_boundary_constraints(logit=logit, boundaries=boundaries)
        return output


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
            output = torch.cat([output_mean, output_log_var], dim=1)
        else:
            output = output_mean
        return output


class ProbabilisticMiddleClassHead(nn.Module):
    """A special head for the middle class in probabilistic monotonic models.

    It learns a constant mean but a flexible, non-constant variance.
    """

    def __init__(self, regression_head_params: dict, activation: ActivationFunction = ActivationFunction.RELU) -> None:
        """Initializes the ProbabilisticMiddleClassHead."""
        super().__init__()
        self.mean = nn.Parameter(torch.randn(1))  # Learnable constant mean
        self.monotonic_constraint = Monotonicity.NONE

        # Network to learn log_variance from probability input
        hidden_layers = regression_head_params.get("hidden_layers", 1)
        hidden_size = regression_head_params.get("hidden_size", 32)

        activation_map = get_activation_function_map()
        activation_module = activation_map.get(activation)
        if activation_module is None:
            raise ValueError(f"Unsupported activation function: {activation}")

        layers = [nn.Linear(1, hidden_size)]
        layers.append(activation_module())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation_module())
        layers.append(nn.Linear(hidden_size, 1))  # Output is log_var

        self.variance_net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, boundaries: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass for the probabilistic middle class head."""
        batch_size = x.size(0)
        output_mean = self.mean.expand(batch_size, 1)
        output_log_var = self.variance_net(x)
        return torch.cat([output_mean, output_log_var], dim=1)


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
            is_middle_class = i == middle_point

            if is_middle_class and constrain_middle_class:
                # If it's the middle class and we are constraining it
                if use_monotonic_constraints and uncertainty_method == UncertaintyMethod.PROBABILISTIC:
                    # User's special case: constant mean, flexible variance
                    self.heads.append(ProbabilisticMiddleClassHead(regression_head_params, activation))
                else:
                    # Original behavior: constant mean, constant variance
                    self.heads.append(ConstantHead(uncertainty_method, regression_output_size))
            else:
                # This is for non-middle-class heads, or if the middle class is not constrained
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
        per_head_outputs = []
        for i in range(len(self.heads)):
            p_i = probabilities[:, i].unsqueeze(1)
            y_i_processed = self.heads[i](p_i) if isinstance(self.heads[i], ConstantHead) else self.heads[i](p_i, boundaries=boundaries)
            per_head_outputs.append(y_i_processed)
        per_head_outputs = torch.stack(per_head_outputs, dim=1)
        final_predictions = _calculate_final_predictions(probabilities, per_head_outputs, self.regression_output_size)
        return (final_predictions, per_head_outputs) if return_head_outputs else final_predictions

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
        constrain_middle_class: bool = False,
    ) -> None:
        """Initializes the SingleHeadNOutputsRegressionModule.

        Args:
            input_size: The input size for the regression head.
            n_classes: The number of classes.
            regression_head_params: Parameters for the regression head.
            uncertainty_method: The uncertainty estimation method.
            regression_output_size: The output size for each class.
            activation: The activation function to use.
            constrain_middle_class: Whether to apply special constraints to the middle class.
        """
        super().__init__()
        self.n_classes = n_classes
        self.regression_output_size = regression_output_size
        self.constrain_middle_class = constrain_middle_class

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

        if self.constrain_middle_class and self.n_classes % 2 != 0:
            self.middle_class_mean = nn.Parameter(torch.randn(1))
            if self.regression_output_size == 2:
                self.middle_class_log_var = nn.Parameter(torch.randn(1))

    def _get_per_head_outputs(self, probabilities: torch.Tensor, boundaries: torch.Tensor | None = None) -> torch.Tensor:
        y_output_all_classes = self.head(probabilities, boundaries=boundaries)
        per_head_outputs = y_output_all_classes.reshape(probabilities.shape[0], self.n_classes, self.regression_output_size).clone()

        if self.constrain_middle_class and self.n_classes % 2 != 0:
            middle_class_idx = self.n_classes // 2
            batch_size = probabilities.shape[0]
            per_head_outputs[:, middle_class_idx, 0] = self.middle_class_mean.expand(batch_size)
            if self.regression_output_size == 2:
                per_head_outputs[:, middle_class_idx, 1] = self.middle_class_log_var.expand(batch_size)

        return per_head_outputs

    def forward(self, probabilities: torch.Tensor, return_head_outputs: bool = False, boundaries: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Performs the forward pass.

        Args:
            probabilities: A tensor of class probabilities.
            return_head_outputs: If True, return per-head outputs alongside the final prediction.
            boundaries: Optional boundaries for the sigmoid transformation.

        Returns:
            The final prediction, or a tuple of (final_prediction, per_head_outputs).
        """
        per_head_outputs = self._get_per_head_outputs(probabilities, boundaries)
        final_predictions = _calculate_final_predictions(probabilities, per_head_outputs, self.regression_output_size)

        return (final_predictions, per_head_outputs) if return_head_outputs else final_predictions

    def forward_per_class(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass and returns the reshaped output of the head."""
        return self._get_per_head_outputs(probabilities)


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


def _calculate_final_predictions(probabilities: torch.Tensor, per_head_outputs: torch.Tensor, regression_output_size: int) -> torch.Tensor:
    """Helper to calculate final predictions from per-head outputs."""
    return apply_law_of_total_variance(probabilities, per_head_outputs) if regression_output_size == 2 else torch.sum(probabilities.unsqueeze(-1) * per_head_outputs, dim=1)
