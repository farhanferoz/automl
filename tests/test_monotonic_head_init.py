"""Regression tests for B1: monotonic head init on all-positive targets.

Historical bug: ``_initialize_monotonic_head`` used
``nn.init.normal_(weight, mean=-3.0)``, catastrophically breaking
exponential (all-positive) datasets. Fixed in commit bae7c36 to use
centroid-aware init: with a non-None centroid, bias is set to the
centroid; without, both weight and bias are zero.
"""

import torch

from automl_package.enums import ActivationFunction, Monotonicity, UncertaintyMethod
from automl_package.models.common.regression_heads import BaseRegressionHead


def _make_head(centroid):
    return BaseRegressionHead(
        input_size=1,
        output_size=2,
        hidden_layers=1,
        hidden_size=8,
        use_batch_norm=False,
        dropout_rate=0.0,
        uncertainty_method=UncertaintyMethod.PROBABILISTIC,
        activation=ActivationFunction.RELU,
        monotonic_constraint=Monotonicity.POSITIVE,
        centroid=centroid,
    )


def test_monotonic_head_bias_initialized_to_centroid():
    head = _make_head(centroid=10.0)
    assert torch.allclose(head.mean_head.bias, torch.tensor([10.0])), (
        f"Expected bias=centroid=10.0, got {head.mean_head.bias.tolist()}. "
        "B1 regression — bias no longer initialised to centroid."
    )


def test_monotonic_head_zero_bias_without_centroid():
    head = _make_head(centroid=None)
    assert torch.allclose(head.mean_head.bias, torch.tensor([0.0])), (
        f"Expected zero bias without centroid, got {head.mean_head.bias.tolist()}"
    )
    assert torch.allclose(head.mean_head.weight, torch.zeros_like(head.mean_head.weight)), (
        f"Expected zero weight without centroid, got {head.mean_head.weight.tolist()}"
    )
