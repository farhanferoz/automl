"""Utility functions for PyTorch models."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from automl_package.enums import ActivationFunction, Monotonicity, LearnedRegularizationType
from automl_package.utils.numerics import aggregate_stats, log_erfc


def calculate_regularization_loss(
    base_loss: torch.Tensor,
    model: nn.Module,
    learn_regularization: bool,
    learned_regularization_type: LearnedRegularizationType,
    l1_lambda: float,
    l2_lambda: float,
    l1_log_lambda: nn.Parameter | None,
    l2_log_lambda: nn.Parameter | None,
) -> torch.Tensor:
    """Calculates the regularization loss for a model, supporting both fixed and learnable lambdas."""

    loss = base_loss
    d, l1_sum, l2_sum = aggregate_stats(model=model, include_bias=False)

    if learn_regularization:
        l1_lambda_val = torch.exp(l1_log_lambda) if l1_log_lambda is not None else None
        l2_lambda_val = torch.exp(l2_log_lambda) if l2_log_lambda is not None else None

        if learned_regularization_type == LearnedRegularizationType.L1_ONLY and l1_lambda_val is not None:
            loss = loss - d * torch.log(l1_lambda_val / 2.0) + l1_lambda_val * l1_sum
        elif learned_regularization_type == LearnedRegularizationType.L2_ONLY and l2_lambda_val is not None:
            loss = loss - (d / 2.0) * torch.log(l2_lambda_val / torch.pi) + l2_lambda_val * l2_sum
        elif learned_regularization_type == LearnedRegularizationType.L1_L2 and l1_lambda_val is not None and l2_lambda_val is not None:
            log_z = torch.log(torch.pi / l2_lambda_val) / 2.0 + torch.square(l1_lambda_val) / (4.0 * l2_lambda_val) + log_erfc(l1_lambda_val / (2.0 * torch.sqrt(l2_lambda_val)))
            loss = loss + d * log_z + l1_lambda_val * l1_sum + l2_lambda_val * l2_sum
    else:
        if l1_lambda > 0:
            loss = loss + l1_lambda * l1_sum
        if l2_lambda > 0:
            loss = loss + l2_lambda * l2_sum

    return loss


def get_device() -> torch.device:
    """Returns the device to use for PyTorch models."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_activation_function_map() -> dict[ActivationFunction, nn.Module]:
    """Returns a map of activation functions."""
    return {
        ActivationFunction.RELU: nn.ReLU,
        ActivationFunction.TANH: nn.Tanh,
        ActivationFunction.SIGMOID: nn.Sigmoid,
        ActivationFunction.LEAKY_RELU: nn.LeakyReLU,
        ActivationFunction.ELU: nn.ELU,
        ActivationFunction.SELU: nn.SELU,
        ActivationFunction.SOFTPLUS: nn.Softplus,
        ActivationFunction.SWISH: nn.SiLU,  # PyTorch's SiLU is equivalent to Swish
        ActivationFunction.MISH: nn.Mish,
        ActivationFunction.GELU: nn.GELU,
        ActivationFunction.PRELU: nn.PReLU,
        ActivationFunction.RRELU: nn.RReLU,
        ActivationFunction.HARDSHRINK: nn.Hardshrink,
        ActivationFunction.SOFTSHRINK: nn.Softshrink,
        ActivationFunction.TANHSHRINK: nn.Tanhshrink,
        ActivationFunction.SOFTMIN: nn.Softmin,
        ActivationFunction.SOFTMAX: nn.Softmax,
        ActivationFunction.LOG_SOFTMAX: nn.LogSoftmax,
        ActivationFunction.GLU: nn.GLU,
        ActivationFunction.LOGSIGMOID: nn.LogSigmoid,
        ActivationFunction.HARDTANH: nn.Hardtanh,
        ActivationFunction.THRESHOLD: nn.Threshold,
        ActivationFunction.RELU6: nn.ReLU6,
        ActivationFunction.CELU: nn.CELU,
        ActivationFunction.SILU: nn.SiLU,
        ActivationFunction.HARDSWISH: nn.Hardswish,
        ActivationFunction.IDENTITY: nn.Identity,
        ActivationFunction.LINEAR: nn.Identity,  # Linear is equivalent to Identity
    }


def monotonic_linear(input_tensor: torch.Tensor, layer: nn.Linear, monotonic_constraint: Monotonicity) -> torch.Tensor:
    """
    Performs a linear operation with an optional monotonic constraint.

    Args:
        input_tensor: The input tensor to the linear layer.
        layer: The nn.Linear layer module.
        monotonic_constraint: The monotonicity constraint to apply.

    Returns:
        The result of the linear operation.
    """

    # Apply softplus to weights to make them non-negative if monotonic constraint needs to be applied
    weight = layer.weight if monotonic_constraint == Monotonicity.NONE else F.softplus(layer.weight)
    return F.linear(input_tensor, weight, layer.bias)
