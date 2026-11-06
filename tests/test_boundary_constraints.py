"""Tests for HARDSIGMOID boundary constraints on SEPARATE_HEADS.

Regression coverage for a bug this path carried undetected: `boundaries` is the per-class value
range table, shape (n_classes, 2), but `SeparateHeadsRegressionModule.forward` handed the WHOLE
table to every head, and `_apply_boundary_constraints` reads dim 0 as a broadcastable batch axis.
Head i therefore tried to broadcast n_classes bounds against a batch of means, which raises unless
n_classes happens to equal batch_size. Boundary regularization is off by default and had no test,
so it only surfaced when the structure phase's PS-2 patch audit first exercised it.
"""

import numpy as np
import pytest
import torch

from automl_package.enums import BoundaryRegularizationMethod, RegressionStrategy, UncertaintyMethod
from automl_package.models.common.regression_heads import SeparateHeadsRegressionModule
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel
from automl_package.utils.plotting import evaluate_mapper_heads

_BATCH = 64  # deliberately != n_classes, which is what the original bug needed to stay hidden
_PROBABILISTIC_OUTPUT_SIZE = 2  # (mean, log_variance) per class
_CLASS_RANGE_WIDTH = 10.0


def _module(k: int, output_size: int = _PROBABILISTIC_OUTPUT_SIZE, **head_params) -> SeparateHeadsRegressionModule:
    method = UncertaintyMethod.PROBABILISTIC if output_size == _PROBABILISTIC_OUTPUT_SIZE else UncertaintyMethod.NONE
    return SeparateHeadsRegressionModule(
        n_classes=k,
        regression_head_params={"hidden_size": 8, **head_params},
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

    # Head depth: `hidden_layers=d` builds d hidden layers (0 = none) and runs for d = 0, 1, 2 (M1); a positive
    # dropout rate builds one dropout layer per hidden layer under any uncertainty method (I5).
    for depth in (0, 1, 2):
        sized = _module(k, hidden_layers=depth, dropout_rate=0.5)
        head_layers = list(sized.heads[0].layers)
        assert sum(isinstance(m, torch.nn.Linear) for m in head_layers) == depth + 1
        assert sum(isinstance(m, torch.nn.Dropout) for m in head_layers) == depth
        assert torch.isfinite(sized.forward(probs, boundaries=boundaries)).all()


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


_LIMIT_TOLERANCE = 1e-5


def _bin_limits_model(**extra) -> ProbabilisticRegressionModel:
    return ProbabilisticRegressionModel(
        input_size=1,
        n_classes=4,
        boundary_regularization_method=BoundaryRegularizationMethod.BIN_LIMITS,
        n_epochs=3,
        learning_rate=0.05,
        early_stopping_rounds=2,
        validation_fraction=0.2,
        random_seed=0,
        calculate_feature_importance=False,
        **extra,
    )


def test_bin_limits_bound_every_head_in_training_prediction_and_curves():
    """Each head's mean lies in its bin's [lo, hi] on every path, and `predict` applies the limits (it is not the unbounded head)."""
    rng = np.random.default_rng(0)
    x = rng.uniform(-3, 3, (120, 1)).astype(np.float32)
    y = (3.0 * x.ravel() + rng.normal(0, 0.5, 120)).astype(np.float32)
    model = _bin_limits_model()
    model.fit(x, y)
    limits = model._per_class_limits
    assert limits.shape == (4, 2)
    lower, upper = limits[:, 0], limits[:, 1]

    module = model.model.regression_module
    x_t = torch.tensor(x)
    model.model.eval()
    with torch.no_grad():
        _, _, _, _, per_head = model.model(x_t)
        means = per_head[:, :, 0].numpy()
        assert (means >= lower - _LIMIT_TOLERANCE).all()
        assert (means <= upper + _LIMIT_TOLERANCE).all()

        _, grid = evaluate_mapper_heads(module, RegressionStrategy.SEPARATE_HEADS, 4, torch.device("cpu"))
        assert (grid >= lower - _LIMIT_TOLERANCE).all()
        assert (grid <= upper + _LIMIT_TOLERANCE).all()

        # The prediction is the probability-weighted mean of the SQUASHED heads: rebuild it from the unbounded heads.
        probabilities = model.get_classifier_predictions(x, y)[1]
        saved, module.bin_limits = module.bin_limits, None
        raw = module(torch.tensor(probabilities), return_head_outputs=True)[1][:, :, 0].numpy()
        module.bin_limits = saved
    squashed = lower + (upper - lower) / (1.0 + np.exp(-raw))
    assert not np.allclose(raw, squashed)  # the limits do something here

    # Structure only: the training loss is the fit term plus the classification term, nothing else.
    outputs = model.model(x_t)
    y_t = torch.tensor(y)
    k_values = torch.full((len(y),), 4)
    expected = model._fit_loss(outputs[0], y_t) + model._class_cross_entropy(outputs[1], y_t, k_values)[0]
    assert float(model._calculate_custom_loss(outputs, y_t).detach()) == pytest.approx(float(expected.detach()), rel=1e-6)
    assert np.allclose(model.predict(x), (probabilities * squashed).sum(axis=1), atol=1e-4)


@pytest.mark.parametrize(
    "extra",
    [{"use_anchored_heads": True}, {"use_monotonic_constraints": True}, {"constrain_middle_class": True}, {"regression_strategy": RegressionStrategy.SINGLE_HEAD_N_OUTPUTS}],
    ids=["anchored", "monotonic", "middle_class", "single_head"],
)
def test_bin_limits_refused_where_it_is_undefined(extra):
    with pytest.raises(ValueError, match="BIN_LIMITS"):
        _bin_limits_model(**extra)
