"""Tests for the ordering-constraint identifiability penalty.

Mathematical spec (from docs/probreg_identifiability_research.md §3.3):

For each head i, let S_i be the training samples with p_i(x) in the top
decile. Define the probability-weighted mean
    M_i = sum_{x in S_i} p_i(x) * h_i(p_i(x)) / sum_{x in S_i} p_i(x)
Then
    L_order = lambda * sum_{i=1..k-1} max(0, M_{i-1} - M_i + delta)^2

Reversing any two heads violates ordering on at least one pair, triggering
the penalty.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from automl_package.utils.ordering_loss import (
    compute_ordering_means,
    ordering_penalty,
)


@pytest.fixture(scope="module")
def probreg_pair():
    """Two fitted ProbReg models sharing config: ordering off and on.

    Module-scoped so the two expensive .fit() calls are paid once for the
    whole file, not per-test.
    """
    from automl_package.enums import RegressionStrategy, UncertaintyMethod
    from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel

    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    x = rng.uniform(-2, 2, 200).reshape(-1, 1).astype(np.float32)
    y = x.ravel().astype(np.float32)

    common = dict(
        input_size=1, n_classes=3,
        regression_strategy=RegressionStrategy.SEPARATE_HEADS,
        uncertainty_method=UncertaintyMethod.PROBABILISTIC,
        n_epochs=1, random_seed=42, early_stopping_rounds=None,
        calculate_feature_importance=False, validation_fraction=0.2,
    )
    off = ProbabilisticRegressionModel(ordering_constraint_weight=0.0, **common)
    off.fit(x, y)
    on = ProbabilisticRegressionModel(ordering_constraint_weight=10.0, **common)
    on.fit(x, y)
    return off, on


def _synthetic_model_outputs(head_means_by_class: list[float]):
    """Build a model_outputs tuple for _calculate_custom_loss."""
    batch = 20
    k = len(head_means_by_class)
    classifier_logits = torch.randn(batch, k)
    per_head_outputs = torch.zeros(batch, k, 2)
    for i, m in enumerate(head_means_by_class):
        per_head_outputs[:, i, 0] = m
    final_predictions = torch.zeros(batch, 2)
    selected_k_values = torch.full((batch,), float("inf"))
    return (final_predictions, classifier_logits, selected_k_values, None, per_head_outputs), torch.zeros(batch, 1)


# -----------------------------------------------------------------------------
# compute_ordering_means — the M_i statistic
# -----------------------------------------------------------------------------


def test_means_shape_k_classes():
    """Returns a tensor of shape (k,), one mean per head."""
    batch, k = 100, 3
    torch.manual_seed(0)
    logits = torch.randn(batch, k)
    head_means = torch.randn(batch, k)
    means = compute_ordering_means(logits, head_means, top_decile_fraction=0.1)
    assert means.shape == (k,)


def test_means_probability_weighted_not_arithmetic():
    """Weighted mean differs from arithmetic mean when weights vary."""
    # Synthetic: k=2, 10 samples. For head 0: p_0 either 0.95 or 0.7, head_mean
    # split so arithmetic vs weighted disagree.
    logits_raw = torch.tensor([[3.0, -3.0]] * 5 + [[0.85, -0.85]] * 5)  # p_0 ~0.998 vs ~0.85
    # Decile=50% puts all 10 into top-half.
    head_means = torch.tensor([[10.0, 0.0]] * 5 + [[0.0, 0.0]] * 5)
    means = compute_ordering_means(logits_raw, head_means, top_decile_fraction=1.0)
    # Arithmetic mean of head-0 outputs over all 10 samples: (10*5 + 0*5)/10 = 5.0
    # Weighted mean: (sum p_0 * h_0) / (sum p_0) = (5 * ~0.998 * 10 + 5 * ~0.85 * 0) / (5*~0.998 + 5*~0.85)
    #              ~= (~49.9) / (~9.24) ~= 5.4
    # So weighted > arithmetic when the high-p samples also have high h.
    assert means[0].item() > 5.1, (
        f"Expected weighted mean > arithmetic mean (5.0); got {means[0].item():.4f}"
    )


def test_means_respects_top_decile_subset():
    """Samples outside the top-decile for head i must not contribute to M_i."""
    batch, k = 100, 2
    torch.manual_seed(1)
    # Make class 0 confident on first half, class 1 on second half.
    logits = torch.cat([torch.tensor([[5.0, -5.0]]).repeat(50, 1),
                        torch.tensor([[-5.0, 5.0]]).repeat(50, 1)])
    # Head-0 outputs large for high-p_0 samples, small for low-p_0 samples.
    h0 = torch.cat([torch.full((50,), 100.0), torch.full((50,), -100.0)])
    h1 = torch.zeros(batch)
    head_means = torch.stack([h0, h1], dim=1)

    # Top 10% of samples by p_0 are all in the first 50; M_0 should reflect
    # them, not be diluted by the second 50.
    means = compute_ordering_means(logits, head_means, top_decile_fraction=0.1)
    assert means[0].item() > 50.0, (
        f"Top-decile subset should pick high-p_0 samples only; got M_0={means[0].item():.3f}"
    )


def test_means_handles_empty_decile_safely():
    """When top_decile_fraction is too small to cover any sample, M_i is safe."""
    batch, k = 3, 3  # top 10% of 3 = 0 samples per head if naively rounded down
    logits = torch.randn(batch, k)
    head_means = torch.randn(batch, k)
    # Must not NaN or raise — fall back to at least one sample per head.
    means = compute_ordering_means(logits, head_means, top_decile_fraction=0.1)
    assert not torch.isnan(means).any(), f"NaN in means: {means}"


# -----------------------------------------------------------------------------
# ordering_penalty — the hinge
# -----------------------------------------------------------------------------


def test_penalty_zero_when_ordered_with_margin():
    means = torch.tensor([1.0, 2.0, 3.0])
    loss = ordering_penalty(means, margin=0.1)
    assert loss.item() == 0.0, f"Expected 0 when ordered, got {loss.item()}"


def test_penalty_positive_when_reversed():
    means = torch.tensor([3.0, 2.0, 1.0])  # completely reversed
    loss = ordering_penalty(means, margin=0.0)
    # Two adjacent violations: (3-2)^2 + (2-1)^2 = 2.0
    assert loss.item() > 1.0, f"Expected > 1 for reversed means, got {loss.item()}"


def test_penalty_margin_respected():
    means = torch.tensor([1.0, 1.05])  # ordered but below margin
    loss_zero_margin = ordering_penalty(means, margin=0.0)
    loss_small_margin = ordering_penalty(means, margin=0.1)
    assert loss_zero_margin.item() == 0.0
    # With margin 0.1, need M_1 > M_0 + 0.1 = 1.1; we have 1.05, so
    # violation = 1.0 - 1.05 + 0.1 = 0.05, penalty = 0.05^2 = 0.0025.
    assert 0.001 < loss_small_margin.item() < 0.01


def test_penalty_gradient_flows_to_means():
    """Gradient should push M_{i-1} down and M_i up."""
    means = torch.tensor([3.0, 2.0], requires_grad=True)
    loss = ordering_penalty(means, margin=0.0)
    loss.backward()
    # Gradient of max(0, M_0 - M_1)^2 = 2*(M_0 - M_1) for M_0 > M_1.
    # d/d M_0 = 2*(3-2) = 2.0; d/d M_1 = -2.0.
    assert means.grad[0].item() > 0.0, f"M_0 should see positive grad, got {means.grad[0].item()}"
    assert means.grad[1].item() < 0.0, f"M_1 should see negative grad, got {means.grad[1].item()}"


def test_penalty_k_equals_1_no_pairs():
    """Single head: no inequalities to enforce; penalty should be zero."""
    means = torch.tensor([5.0])
    loss = ordering_penalty(means, margin=0.0)
    assert loss.item() == 0.0


# -----------------------------------------------------------------------------
# Integration: end-to-end with ProbReg's loss
# -----------------------------------------------------------------------------


def test_probreg_ordering_weight_zero_by_default():
    """The new parameter exists and defaults to 0 (off)."""
    from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel

    model = ProbabilisticRegressionModel(input_size=1, n_classes=3)
    assert hasattr(model, "ordering_constraint_weight")
    assert model.ordering_constraint_weight == 0.0


def test_probreg_ordering_loss_adds_to_total_when_enabled(probreg_pair):
    off, on = probreg_pair
    model_outputs, y_t = _synthetic_model_outputs([10.0, 5.0, 0.0])  # reversed
    loss_off = off._calculate_custom_loss(model_outputs, y_t, include_boundary_loss=False)
    loss_on = on._calculate_custom_loss(model_outputs, y_t, include_boundary_loss=False)
    assert loss_on.item() > loss_off.item() + 0.1, (
        f"Reversed means should trigger the ordering penalty: "
        f"off={loss_off.item():.4f}, on={loss_on.item():.4f}"
    )


def test_probreg_ordering_loss_is_zero_when_means_ordered(probreg_pair):
    off, on = probreg_pair
    model_outputs, y_t = _synthetic_model_outputs([0.0, 5.0, 10.0])  # ordered
    loss_off = off._calculate_custom_loss(model_outputs, y_t, include_boundary_loss=False)
    loss_on = on._calculate_custom_loss(model_outputs, y_t, include_boundary_loss=False)
    assert abs(loss_on.item() - loss_off.item()) < 1e-4, (
        f"Ordered means should produce zero ordering contribution: "
        f"off={loss_off.item():.6f}, on={loss_on.item():.6f}"
    )


def test_probreg_ordering_under_ce_stop_grad_does_not_leak_to_classifier():
    """Gradient-stop contract: under CE_STOP_GRAD, ordering must not flow to logits."""
    from automl_package.enums import (
        ProbabilisticRegressionOptimizationStrategy,
        RegressionStrategy,
        UncertaintyMethod,
    )
    from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    x = rng.uniform(-2, 2, 200).reshape(-1, 1).astype(np.float32)
    y = x.ravel().astype(np.float32)
    model = ProbabilisticRegressionModel(
        input_size=1, n_classes=3,
        regression_strategy=RegressionStrategy.SEPARATE_HEADS,
        uncertainty_method=UncertaintyMethod.PROBABILISTIC,
        optimization_strategy=ProbabilisticRegressionOptimizationStrategy.CE_STOP_GRAD,
        ordering_constraint_weight=100.0,
        n_epochs=1, random_seed=0, early_stopping_rounds=None,
        calculate_feature_importance=False, validation_fraction=0.2,
    )
    model.fit(x, y)

    # Construct a reversed-means scenario with a logits tensor we OWN, so we
    # can check whether gradient flows back into it after .backward() on the
    # ordering contribution.
    batch, k = 20, 3
    classifier_logits = torch.randn(batch, k, requires_grad=True)
    # per_head_outputs has requires_grad so the ordering term has a grad_fn
    # even when logits get detached by the stop-grad.
    per_head_outputs = torch.zeros(batch, k, 2, requires_grad=True)
    with torch.no_grad():
        per_head_outputs[:, 0, 0] = 10.0
        per_head_outputs[:, 1, 0] = 5.0
        per_head_outputs[:, 2, 0] = 0.0
    # Only the ordering term is relevant here; pass final_predictions and y
    # placeholders that don't contribute via the regression-loss path.
    final_predictions = torch.zeros(batch, 2)
    selected_k_values = torch.full((batch,), float("inf"))
    model_outputs = (final_predictions, classifier_logits, selected_k_values, None, per_head_outputs)
    y_t = torch.zeros(batch, 1)

    loss = model._calculate_custom_loss(model_outputs, y_t, include_boundary_loss=False)
    loss.backward()
    # Under CE_STOP_GRAD the ordering loss must not send gradient back to logits.
    assert classifier_logits.grad is None or torch.all(classifier_logits.grad == 0), (
        f"CE_STOP_GRAD violated: ordering loss leaked gradient to classifier logits "
        f"(max |grad| = {classifier_logits.grad.abs().max().item() if classifier_logits.grad is not None else 0:.2e})"
    )


def test_probreg_ordering_under_regression_only_does_flow_to_classifier():
    """Complementary check: under REGRESSION_ONLY the gradient path is open."""
    from automl_package.enums import (
        ProbabilisticRegressionOptimizationStrategy,
        RegressionStrategy,
        UncertaintyMethod,
    )
    from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    x = rng.uniform(-2, 2, 200).reshape(-1, 1).astype(np.float32)
    y = x.ravel().astype(np.float32)
    model = ProbabilisticRegressionModel(
        input_size=1, n_classes=3,
        regression_strategy=RegressionStrategy.SEPARATE_HEADS,
        uncertainty_method=UncertaintyMethod.PROBABILISTIC,
        optimization_strategy=ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY,
        ordering_constraint_weight=100.0,
        n_epochs=1, random_seed=0, early_stopping_rounds=None,
        calculate_feature_importance=False, validation_fraction=0.2,
    )
    model.fit(x, y)

    batch, k = 20, 3
    classifier_logits = torch.randn(batch, k, requires_grad=True)
    # per_head_outputs has requires_grad so the ordering term has a grad_fn
    # even when logits get detached by the stop-grad.
    per_head_outputs = torch.zeros(batch, k, 2, requires_grad=True)
    with torch.no_grad():
        per_head_outputs[:, 0, 0] = 10.0
        per_head_outputs[:, 1, 0] = 5.0
        per_head_outputs[:, 2, 0] = 0.0
    final_predictions = torch.zeros(batch, 2)
    selected_k_values = torch.full((batch,), float("inf"))
    model_outputs = (final_predictions, classifier_logits, selected_k_values, None, per_head_outputs)
    y_t = torch.zeros(batch, 1)

    loss = model._calculate_custom_loss(model_outputs, y_t, include_boundary_loss=False)
    loss.backward()
    assert classifier_logits.grad is not None and classifier_logits.grad.abs().sum() > 0.0, (
        "Under REGRESSION_ONLY the ordering loss should flow gradient to logits."
    )


def test_probreg_ordering_skipped_for_non_sep_heads():
    """Ordering should be a no-op for SINGLE_HEAD_* strategies."""
    from automl_package.enums import RegressionStrategy, UncertaintyMethod
    from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel

    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    x = rng.uniform(-2, 2, 200).reshape(-1, 1).astype(np.float32)
    y = x.ravel().astype(np.float32)

    on = ProbabilisticRegressionModel(
        input_size=1, n_classes=3,
        regression_strategy=RegressionStrategy.SINGLE_HEAD_N_OUTPUTS,
        uncertainty_method=UncertaintyMethod.PROBABILISTIC,
        ordering_constraint_weight=100.0,
        n_epochs=1, random_seed=42, early_stopping_rounds=None,
        calculate_feature_importance=False, validation_fraction=0.2,
    )
    # Should fit without crashing even with large weight — ordering must be skipped
    # because SINGLE_HEAD_N_OUTPUTS has no permutation symmetry to break.
    on.fit(x, y)
    assert on.model is not None
