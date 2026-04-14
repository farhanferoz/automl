"""Utility functions for PyTorch models."""

import torch
import torch.nn as nn
import torch.nn.functional as f

from automl_package.enums import ActivationFunction, LearnedRegularizationType, Monotonicity
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
    """Returns the best available device: CUDA > XPU > CPU.

    On first call, disables a broken triton-xpu stub that crashes torch._dynamo.
    """
    _disable_broken_triton()
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def _disable_broken_triton() -> None:
    """Disables triton if it's a broken namespace stub (triton-xpu >=3.6 issue).

    triton-xpu ships a namespace package without real module contents.
    torch._dynamo imports it and crashes accessing triton.language.dtype,
    triton.backends.compiler, etc. Since we don't use Triton kernels,
    the fix is to make torch think triton isn't installed by patching
    has_triton_package() to return False.
    """
    if getattr(_disable_broken_triton, "_done", False):
        return
    _disable_broken_triton._done = True  # type: ignore[attr-defined]

    try:
        import triton.language as tl

        if not hasattr(tl, "dtype"):
            # Broken stub — disable triton detection in torch
            import torch.utils._triton

            torch.utils._triton.has_triton_package.cache_clear()
            torch.utils._triton.has_triton_package = lambda: False  # type: ignore[assignment]
    except ImportError:
        pass  # No triton installed — nothing to patch


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
    """Performs a linear operation with an optional monotonic constraint.

    Args:
        input_tensor: The input tensor to the linear layer.
        layer: The nn.Linear layer module.
        monotonic_constraint: The monotonicity constraint to apply.

    Returns:
        The result of the linear operation.
    """
    # Apply softplus to weights to make them non-negative if monotonic constraint needs to be applied
    weight = layer.weight if monotonic_constraint == Monotonicity.NONE else f.softplus(layer.weight)
    output = f.linear(input_tensor, weight, layer.bias)

    if monotonic_constraint == Monotonicity.NEGATIVE:
        output = -output

    return output

def apply_law_of_total_variance(probabilities: torch.Tensor, per_head_outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies the Law of Total Variance to compute the final mean and log_variance.

    Args:
        probabilities: A tensor of class probabilities, shape (batch_size, n_classes).
        per_head_outputs: A tensor of predictions from each head, shape (batch_size, n_classes, 2).

    Returns:
        A tuple containing:
            - final_mean: The final mean prediction, shape (batch_size, 1).
            - final_log_var: The final log_variance prediction, shape (batch_size, 1).
    """
    per_head_means = per_head_outputs[..., 0]
    per_head_vars = torch.exp(per_head_outputs[..., 1])

    # 1. Final Mean (E[Y]) = E[E[Y|C]] = sum(P(C=i) * E[Y|C=i])
    final_mean = torch.sum(probabilities * per_head_means, dim=1)

    # 2. Final Variance (Var(Y)) = E[Var(Y|C)] + Var(E[Y|C])
    # E[Var(Y|C)] = sum(P(C=i) * Var(Y|C=i))
    expected_variance = torch.sum(probabilities * per_head_vars, dim=1)

    # Var(E[Y|C]) = E[(E[Y|C] - E[E[Y|C]])^2] = sum(P(C=i) * (E[Y|C=i] - E[Y])^2)
    variance_of_expectation = torch.sum(probabilities * torch.square(per_head_means - final_mean.unsqueeze(1)), dim=1)

    final_variance = expected_variance + variance_of_expectation

    # Clamp final_variance to avoid log(0)
    final_log_var = torch.log(torch.clamp(final_variance, min=1e-9))

    # Combine final mean and log_var into a single tensor
    final_predictions = torch.stack([final_mean, final_log_var], dim=1)

    return final_predictions
