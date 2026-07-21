"""Tests for HARDSIGMOID boundary constraints on SEPARATE_HEADS.

Regression coverage for a bug this path carried undetected: `boundaries` is the per-class value
range table, shape (n_classes, 2), but `SeparateHeadsRegressionModule.forward` handed the WHOLE
table to every head, and `_apply_boundary_constraints` reads dim 0 as a broadcastable batch axis.
Head i therefore tried to broadcast n_classes bounds against a batch of means, which raises unless
n_classes happens to equal batch_size. Boundary regularization is off by default and had no test,
so it only surfaced when the structure phase's PS-2 patch audit first exercised it.
"""

import pytest
import torch

from automl_package.enums import UncertaintyMethod
from automl_package.models.common.regression_heads import SeparateHeadsRegressionModule

_BATCH = 64  # deliberately != n_classes, which is what the original bug needed to stay hidden
_PROBABILISTIC_OUTPUT_SIZE = 2  # (mean, log_variance) per class
_CLASS_RANGE_WIDTH = 10.0


def _module(k: int, output_size: int = _PROBABILISTIC_OUTPUT_SIZE) -> SeparateHeadsRegressionModule:
    method = UncertaintyMethod.PROBABILISTIC if output_size == _PROBABILISTIC_OUTPUT_SIZE else UncertaintyMethod.NONE
    return SeparateHeadsRegressionModule(
        n_classes=k,
        regression_head_params={"hidden_size": 8},
        uncertainty_method=method,
        regression_output_size=output_size,
        # OFF deliberately: at ODD k the middle head becomes a ConstantHead, which has no input to
        # squash and ignores `boundaries` by construction. That interaction is covered by its own
        # test below; leaving it on here would conflate it with the per-head routing under test.
        constrain_middle_class=False,
    )


@pytest.mark.parametrize("k", [2, 3, 6])
def test_forward_with_boundaries_runs_when_batch_differs_from_n_classes(k: int):
    """The shape the bug crashed on: batch_size != n_classes."""
    torch.manual_seed(0)
    module = _module(k)
    probs = torch.softmax(torch.randn(_BATCH, k), dim=-1)
    lowers = torch.arange(k, dtype=torch.float32)
    boundaries = torch.stack([lowers, lowers + 1.0], dim=1)

    out = module.forward(probs, boundaries=boundaries)

    assert out.shape[0] == _BATCH
    assert torch.isfinite(out).all()


def test_each_head_is_constrained_to_its_own_class_range():
    """Head i's mean must land inside boundaries[i], not inside some other class's range.

    This is the half of the bug a shape fix alone would not catch: passing the whole table without
    crashing (e.g. when k happens to equal the batch size) would still constrain every head by the
    WRONG rows. Ranges here are widely separated so a mix-up cannot hide inside an overlap.
    """
    torch.manual_seed(0)
    k = 3
    module = _module(k)
    probs = torch.softmax(torch.randn(_BATCH, k), dim=-1)
    lowers = torch.tensor([0.0, 100.0, 200.0])
    boundaries = torch.stack([lowers, lowers + _CLASS_RANGE_WIDTH], dim=1)

    per_head = module.forward(probs, return_head_outputs=True, boundaries=boundaries)[1]

    for i in range(k):
        means = per_head[:, i, 0]
        assert (means >= lowers[i]).all(), f"head {i} fell below its own lower bound"
        assert (means <= lowers[i] + _CLASS_RANGE_WIDTH).all(), f"head {i} exceeded its own upper bound"


def test_middle_class_head_is_exempt_from_boundaries_at_odd_k():
    """Documents a real interaction rather than asserting it is desirable.

    With `constrain_middle_class=True` and ODD k, the middle head is a constant/middle-class head:
    it has no input to squash, so `boundaries` cannot apply to it and its mean is free to sit
    outside its class's range. Pinned here so that if the middle-class head ever starts honouring
    boundaries, that is a deliberate change with a failing test attached, not a silent one.

    (At EVEN k there is no middle head at all -- `middle_point = (n_classes - 1) / 2` is a
    half-integer, so `i == middle_point` never holds and `constrain_middle_class` is inert.)
    """
    torch.manual_seed(0)
    k = 3
    module = SeparateHeadsRegressionModule(
        n_classes=k,
        regression_head_params={"hidden_size": 8},
        uncertainty_method=UncertaintyMethod.PROBABILISTIC,
        regression_output_size=_PROBABILISTIC_OUTPUT_SIZE,
        constrain_middle_class=True,
    )
    probs = torch.softmax(torch.randn(_BATCH, k), dim=-1)
    lowers = torch.tensor([0.0, 100.0, 200.0])
    boundaries = torch.stack([lowers, lowers + _CLASS_RANGE_WIDTH], dim=1)

    per_head = module.forward(probs, return_head_outputs=True, boundaries=boundaries)[1]

    middle = per_head[:, 1, 0]
    assert torch.allclose(middle, middle[0].expand_as(middle)), "middle head should emit one constant"
    assert not (middle >= lowers[1]).all(), "middle head is expected to be exempt from its class range"
