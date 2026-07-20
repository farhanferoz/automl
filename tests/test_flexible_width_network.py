"""Tests for FlexibleWidthNN (capacity-programme Task F2): shared trunk, per-width heads.

Covers: fit/predict at every configured width on synthetic data (MSE and CE tasks); per-width
heads are genuinely distinct parameters (not shared/aliased); the prefix-nesting property --
width-w's prediction is invariant to perturbing the trunk's hidden units w: onward -- holds
numerically, not just architecturally assumed.
"""

import numpy as np
import pytest
import torch
from sklearn.model_selection import train_test_split

from automl_package.enums import ActivationFunction, TaskType, UncertaintyMethod, WidthSelectionMethod
from automl_package.models.flexible_width_network import FlexibleWidthNN
from automl_package.utils.pytorch_utils import get_device

DEVICE = get_device()
TEST_WIDTHS = (2, 4, 6, 8)
PREFIX_INVARIANCE_TOLERANCE = 1e-5
PERTURBATION_SCALE = 5.0
MSE_TOLERANCE = 1.0
N_CLASSES = 3
CLASSIFICATION_ACCURACY_FLOOR = 0.6
RANDOM_SEED = 42


def _make_width_model(**overrides):
    defaults = {
        "input_size": 1, "output_size": 1, "task_type": TaskType.REGRESSION,
        "widths": TEST_WIDTHS, "activation": ActivationFunction.TANH,
        "n_epochs": 30, "learning_rate": 0.02, "random_seed": RANDOM_SEED,
        "calculate_feature_importance": False,
    }
    defaults.update(overrides)
    return FlexibleWidthNN(**defaults)


class TestFlexibleWidthNNSmoke:
    """fit/predict at every configured width, on both MSE and CE tasks."""

    def test_fit_predict_at_every_width_regression(self, simple_linear_data):
        x, y = simple_linear_data
        model = _make_width_model()
        model.fit(x, y)

        for width in model.widths:
            y_pred = model.predict(x, width=width)
            assert y_pred.shape == (len(x),)
            assert not np.any(np.isnan(y_pred))

    def test_largest_width_reasonable_mse_on_linear(self, simple_linear_data):
        x, y = simple_linear_data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=RANDOM_SEED)

        model = _make_width_model(n_epochs=80)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test, width=max(model.widths))
        mse = float(np.mean((y_test - y_pred) ** 2))

        assert mse < MSE_TOLERANCE, f"FlexibleWidthNN MSE ({mse:.4f}) too high on y=2x+1 data at w_max."

    def test_fit_predict_multiclass_classification(self):
        rng = np.random.default_rng(RANDOM_SEED)
        x = rng.normal(size=(240, 2)).astype(np.float32)
        y = (np.argmax(x @ rng.normal(size=(2, N_CLASSES)), axis=1)).astype(np.int64)

        model = _make_width_model(
            input_size=2, output_size=N_CLASSES, task_type=TaskType.CLASSIFICATION,
            n_epochs=60, learning_rate=0.01,
        )
        model.fit(x, y)

        for width in model.widths:
            y_pred = model.predict(x, width=width)
            assert y_pred.shape == (len(x),)
            assert set(np.unique(y_pred)).issubset(set(range(N_CLASSES)))

        accuracy = float(np.mean(model.predict(x, width=max(model.widths)) == y))
        assert accuracy > CLASSIFICATION_ACCURACY_FLOOR, f"w_max accuracy ({accuracy:.3f}) too low on separable synthetic classes."

    def test_fit_predict_binary_classification(self):
        rng = np.random.default_rng(RANDOM_SEED)
        x = rng.normal(size=(200, 2)).astype(np.float32)
        y = (x[:, 0] + x[:, 1] > 0).astype(np.int64)

        model = _make_width_model(
            input_size=2, output_size=1, task_type=TaskType.CLASSIFICATION,
            n_epochs=60, learning_rate=0.01,
        )
        model.fit(x, y)

        y_pred = model.predict(x, width=max(model.widths))
        assert y_pred.shape == (len(x),)
        assert set(np.unique(y_pred)).issubset({0, 1})


class TestPerWidthHeadsAreDistinctParameters:
    """Per-width heads must be genuinely separate parameter tensors, not aliases."""

    def test_heads_do_not_share_parameter_storage(self, simple_linear_data):
        x, y = simple_linear_data
        model = _make_width_model()
        model.fit(x, y)

        head_param_ids = {}
        for width in model.widths:
            head = model.model.heads[str(width)]
            head_param_ids[width] = {id(p) for p in head.parameters()}

        all_ids = [pid for ids in head_param_ids.values() for pid in ids]
        assert len(all_ids) == len(set(all_ids)), "Some per-width heads share parameter storage -- heads are not independent."

        trunk_ids = {id(p) for p in model.model.trunk_linear.parameters()}
        for width, ids in head_param_ids.items():
            assert ids.isdisjoint(trunk_ids), f"width={width}'s head aliases the trunk's own parameters."

    def test_heads_have_independent_values_after_training(self, simple_linear_data):
        x, y = simple_linear_data
        model = _make_width_model()
        model.fit(x, y)

        weights = [model.model.heads[str(w)].weight.detach().clone() for w in model.widths]
        for i in range(len(weights)):
            for j in range(i + 1, len(weights)):
                assert not torch.equal(weights[i], weights[j]), (
                    f"heads for widths {model.widths[i]} and {model.widths[j]} have identical weights after training."
                )


class TestPrefixNestingProperty:
    """Width-w's prediction must depend only on the trunk's first w hidden units."""

    def test_forward_width_invariant_to_trunk_perturbation_beyond_width(self):
        torch.manual_seed(RANDOM_SEED)
        model = _make_width_model(input_size=3)
        model.build_model()
        net = model.model
        net.eval()

        x = torch.randn(17, 3, device=DEVICE)
        max_err = 0.0
        for width in model.widths:
            with torch.no_grad():
                out_before = net.forward_width(x, width)

            orig_weight = net.trunk_linear.weight.detach().clone()
            orig_bias = net.trunk_linear.bias.detach().clone()
            with torch.no_grad():
                if width < net.w_max:
                    net.trunk_linear.weight[width:, :] += torch.randn_like(net.trunk_linear.weight[width:, :]) * PERTURBATION_SCALE
                    net.trunk_linear.bias[width:] += torch.randn_like(net.trunk_linear.bias[width:]) * PERTURBATION_SCALE
                out_after = net.forward_width(x, width)
                net.trunk_linear.weight.copy_(orig_weight)
                net.trunk_linear.bias.copy_(orig_bias)

            err = (out_before - out_after).abs().max().item()
            max_err = max(max_err, err)
            assert err < PREFIX_INVARIANCE_TOLERANCE, (
                f"width={width}: forward_width changed by {err:.3e} after perturbing trunk units "
                f"{width}..{net.w_max - 1} -- prefix-nesting property violated."
            )

        print(f"[test_prefix_nesting] max_abs_err over all widths = {max_err:.3e} (tol={PREFIX_INVARIANCE_TOLERANCE:.0e})")

    def test_forward_width_matches_manual_prefix_mask_of_shared_trunk(self):
        torch.manual_seed(RANDOM_SEED + 1)
        model = _make_width_model(input_size=3)
        model.build_model()
        net = model.model
        net.eval()

        x = torch.randn(13, 3, device=DEVICE)
        with torch.no_grad():
            h_full = net.hidden(x)
            for width in model.widths:
                expected_masked = h_full.clone()
                expected_masked[:, width:] = 0.0
                expected_out = net.heads[str(width)](expected_masked)
                actual_out = net.forward_width(x, width)
                err = (actual_out - expected_out).abs().max().item()
                assert err < PREFIX_INVARIANCE_TOLERANCE, (
                    f"width={width}: forward_width does not equal head applied to the first {width} "
                    f"columns of the shared full-width trunk activation (err={err:.3e})."
                )


class TestFlexibleWidthNNValidation:
    """Constructor and predict()-time validation."""

    def test_empty_widths_rejected(self):
        with pytest.raises(ValueError, match="widths"):
            FlexibleWidthNN(input_size=1, output_size=1, widths=())

    def test_nonpositive_width_rejected(self):
        with pytest.raises(ValueError, match="widths"):
            FlexibleWidthNN(input_size=1, output_size=1, widths=(0, 4))

    def test_distilled_selection_not_implemented(self):
        with pytest.raises(NotImplementedError, match="DistilledCapacityRouter"):
            FlexibleWidthNN(input_size=1, output_size=1, width_selection_method=WidthSelectionMethod.DISTILLED)

    def test_routed_inference_mode_without_router_raises(self, simple_linear_data):
        x, y = simple_linear_data
        model = _make_width_model(n_epochs=1)
        model.fit(x, y)
        with pytest.raises(RuntimeError, match="fit_router"):
            model.predict(x, inference_mode="routed")

    def test_predict_rejects_unconfigured_width(self, simple_linear_data):
        x, y = simple_linear_data
        model = _make_width_model(n_epochs=1)
        model.fit(x, y)
        with pytest.raises(ValueError, match="not one of the configured widths"):
            model.predict(x, width=3)

    def test_unknown_inference_mode_rejected(self, simple_linear_data):
        x, y = simple_linear_data
        model = _make_width_model(n_epochs=1)
        model.fit(x, y)
        with pytest.raises(ValueError, match="Unknown inference_mode"):
            model.predict(x, inference_mode="bogus")


class TestFlexibleWidthNNRouting:
    """`fit_router()` + `predict(inference_mode="routed")` -- capacity-programme Task F3."""

    def test_routed_predict_shape_and_no_nan(self, simple_linear_data):
        x, y = simple_linear_data
        model = _make_width_model()
        model.fit(x, y)
        model.fit_router(x, y)

        y_pred = model.predict(x, inference_mode="routed")
        assert y_pred.shape == (len(x),)
        assert not np.any(np.isnan(y_pred))

    def test_router_routes_only_within_capacity_grid(self, simple_linear_data):
        x, y = simple_linear_data
        model = _make_width_model()
        model.fit(x, y)
        router = model.fit_router(x, y)

        routed_widths = {capacity[0] for capacity in router.route(x)}
        assert routed_widths.issubset(set(model.widths))

    def test_fit_router_default_cost_fn_uses_s2_executed_flops(self, simple_linear_data):
        """`fit_router`'s default `cost_fn` must come from `capacity_accounting.executed_flops`
        (S2 accounting), not a re-derived FLOPs formula -- confirm the two agree at every width.
        """
        from automl_package.examples.capacity_accounting import executed_flops

        x, y = simple_linear_data
        model = _make_width_model()
        model.fit(x, y)
        router = model.fit_router(x, y)

        for i, width in enumerate(model.widths):
            assert router.costs_[i] == pytest.approx(executed_flops(model.model, width))


class TestFlexibleWidthNNPredictUncertainty:
    """WD1: `predict_uncertainty` must be correct for every `uncertainty_method`, or raise.

    `CONSTANT` / `BINNED_RESIDUAL_STD` never touch the stacked-over-widths `forward()` output
    and must keep working; `MC_DROPOUT` / `PROBABILISTIC` cannot be served by this architecture
    (no dropout layer, no log-variance head) and must raise explicitly rather than silently
    mis-indexing that `(len(widths), N, output_size)` tensor.
    """

    def test_constant_uncertainty_returns_one_value_per_sample(self, simple_linear_data):
        x, y = simple_linear_data
        model = _make_width_model()  # default uncertainty_method=CONSTANT
        model.fit(x, y)

        uncertainty = model.predict_uncertainty(x)
        assert uncertainty.shape == (len(x),)
        assert np.all(uncertainty >= 0)

    def test_binned_residual_std_uncertainty_returns_one_value_per_sample(self, simple_linear_data):
        x, y = simple_linear_data
        model = _make_width_model(uncertainty_method=UncertaintyMethod.BINNED_RESIDUAL_STD)
        model.fit(x, y)
        model.calibrate_uncertainty(x, y)

        uncertainty = model.predict_uncertainty(x)
        assert uncertainty.shape == (len(x),)

    def test_mc_dropout_uncertainty_raises_explicit_error(self, simple_linear_data):
        x, y = simple_linear_data
        model = _make_width_model(uncertainty_method=UncertaintyMethod.MC_DROPOUT)
        model.fit(x, y)

        with pytest.raises(NotImplementedError, match="MC_DROPOUT"):
            model.predict_uncertainty(x)

    def test_probabilistic_uncertainty_raises_explicit_error(self, simple_linear_data):
        x, y = simple_linear_data
        model = _make_width_model(uncertainty_method=UncertaintyMethod.PROBABILISTIC)
        model.fit(x, y)

        with pytest.raises(NotImplementedError, match="PROBABILISTIC"):
            model.predict_uncertainty(x)
