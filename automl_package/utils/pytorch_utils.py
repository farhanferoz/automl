"""Utility functions for PyTorch models."""

import torch.nn as nn

from automl_package.enums import ActivationFunction


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
