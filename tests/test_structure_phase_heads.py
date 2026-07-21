"""Tests for the structure phase's model additions (PS-A1): HeadSpread.ALL_CONSTANT,
ProbRegLossType.FIXED_SIGMA_MIXTURE, and the narrowed uncertainty-method force-override.

See docs/plans/capacity_programme structure.md PS-A1.
"""

import pytest
import torch

from automl_package.enums import (
    HeadSpread,
    ProbabilisticRegressionOptimizationStrategy,
    ProbRegLossType,
    UncertaintyMethod,
)
from automl_package.models.common.regression_heads import SeparateHeadsRegressionModule
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel


def test_all_constant_ignores_input():
    """ALL_CONSTANT module: identical outputs for two different probability vectors."""
    torch.manual_seed(0)
    k = 4
    module = SeparateHeadsRegressionModule(
        n_classes=k,
        regression_head_params={"hidden_size": 16},
        uncertainty_method=UncertaintyMethod.PROBABILISTIC,
        regression_output_size=2,
        head_spread=HeadSpread.ALL_CONSTANT,
    )
    probs_a = torch.softmax(torch.randn(5, k), dim=-1)
    probs_b = torch.softmax(torch.randn(5, k), dim=-1)

    out_a = module.forward_per_class(probs_a)
    out_b = module.forward_per_class(probs_b)

    assert torch.allclose(out_a, out_b)


def test_fixed_sigma_mixture_requires_sigma():
    """Constructing with FIXED_SIGMA_MIXTURE and fixed_sigma_train=None raises ValueError."""
    with pytest.raises(ValueError, match="fixed_sigma_train"):
        ProbabilisticRegressionModel(
            input_size=3,
            n_classes=3,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            prob_reg_loss_type=ProbRegLossType.FIXED_SIGMA_MIXTURE,
            fixed_sigma_train=None,
        )


def test_ce_keeps_constant_uncertainty():
    """CE_STOP_GRAD (escape hatch on) + uncertainty_method=CONSTANT constructs without the flip."""
    model = ProbabilisticRegressionModel(
        input_size=3,
        n_classes=3,
        uncertainty_method=UncertaintyMethod.CONSTANT,
        optimization_strategy=ProbabilisticRegressionOptimizationStrategy.CE_STOP_GRAD,
        allow_retired_capacity_selection=True,
    )
    assert model.uncertainty_method == UncertaintyMethod.CONSTANT


def test_head_spread_guard():
    """ALL_CONSTANT + use_anchored_heads=True raises."""
    with pytest.raises(ValueError, match="head_spread"):
        SeparateHeadsRegressionModule(
            n_classes=3,
            regression_head_params={"hidden_size": 16},
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            regression_output_size=2,
            head_spread=HeadSpread.ALL_CONSTANT,
            use_anchored_heads=True,
        )


def test_head_spread_accepts_string_and_refuses_junk():
    """A raw string coerces (the file's convention for every other enum kwarg); junk raises.

    Regression guard: the branches are `is` identity checks, so before coercion existed a string
    matched no branch and silently behaved as PER_INPUT -- a silent fall-through to a default,
    the same defect class that produced mislabelled width cells.
    """
    model = ProbabilisticRegressionModel(input_size=3, n_classes=3, head_spread="all_constant")
    assert model.head_spread is HeadSpread.ALL_CONSTANT

    with pytest.raises(TypeError, match="head_spread must be a HeadSpread member"):
        ProbabilisticRegressionModel(input_size=3, n_classes=3, head_spread=object())
