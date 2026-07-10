"""Tests for the NESTED layer-selection strategy (capacity-ladder task F2, code phase).

NESTED trains `FlexibleHiddenLayersNN` with per-sample depth draws
d ~ Uniform{1..max_hidden_layers} as a TRAINING SCHEDULE, not a learned selector and
not a loss term (strictly-probabilistic premise, `docs/plans/capacity_ladder_2026-07-09/
EXECUTION_PLAN.md` §0b). The prefix-property checks here mirror the F0 audit
(`automl_package/examples/_flexnn_prefix_selftest.py`) through the NESTED path instead
of `SoftGatingStrategy`.
"""

import numpy as np
import pytest
import torch

from automl_package.enums import ActivationFunction, LayerSelectionMethod, TaskType, UncertaintyMethod
from automl_package.models.flexible_neural_network import FlexibleHiddenLayersNN
from automl_package.models.independent_weights_flexible_neural_network import IndependentWeightsFlexibleNN
from automl_package.utils.losses import nll_loss

TOL = 1e-6


def _make_nested_model(
    max_hidden_layers: int = 4,
    hidden_size: int = 8,
    uncertainty_method: UncertaintyMethod = UncertaintyMethod.PROBABILISTIC,
    random_seed: int = 0,
    input_size: int = 3,
    output_size: int = 1,
) -> FlexibleHiddenLayersNN:
    """Builds a small NESTED FlexibleHiddenLayersNN, BN off per the F0 ladder-arm decision."""
    model = FlexibleHiddenLayersNN(
        task_type=TaskType.REGRESSION,
        max_hidden_layers=max_hidden_layers,
        hidden_size=hidden_size,
        activation=ActivationFunction.RELU,
        layer_selection_method=LayerSelectionMethod.NESTED,
        n_predictor_layers=0,
        use_batch_norm=False,
        uncertainty_method=uncertainty_method,
        output_size=output_size,
        input_size=input_size,
        random_seed=random_seed,
    )
    model.build_model()
    return model


class TestNestedRegistration:
    """NESTED must be registered in both places: the enum and the strategy_map."""

    def test_enum_member_exists(self):
        assert LayerSelectionMethod.NESTED.value == "nested"

    def test_model_constructs_and_builds(self):
        model = _make_nested_model()
        assert model.strategy.__class__.__name__ == "NestedStrategy"
        assert model.model is not None

    def test_n_predictor_layers_must_be_zero(self):
        """NESTED conditions on no n_predictor, same as NoneStrategy."""
        with pytest.raises(ValueError, match="n_predictor_layers must be 0"):
            FlexibleHiddenLayersNN(
                task_type=TaskType.REGRESSION,
                max_hidden_layers=3,
                layer_selection_method=LayerSelectionMethod.NESTED,
                n_predictor_layers=1,
                input_size=2,
                output_size=1,
            )


class TestNestedDepthDrawUniform:
    """The per-sample draw is a schedule: Uniform{1..max_hidden_layers}, fixed seed."""

    def test_draws_are_roughly_uniform(self):
        torch.manual_seed(0)
        model = _make_nested_model(max_hidden_layers=4, uncertainty_method=UncertaintyMethod.CONSTANT)
        model.model.train()
        x = torch.randn(4000, 3, device=model.device)
        _, n_actual, _, _, _ = model.model(x)

        assert n_actual.min().item() >= 1
        assert n_actual.max().item() <= 4

        counts = torch.bincount(n_actual, minlength=5)[1:5].float()
        fractions = counts / counts.sum()
        expected = torch.full_like(fractions, 1.0 / 4)
        assert torch.allclose(fractions, expected, atol=0.03), f"depth draw histogram not roughly uniform: {fractions.tolist()}"


class TestNestedPrefixProperty:
    """The prefix property (F0's invariant) must hold through the NESTED path too."""

    def test_all_depth_outputs_match_independent_truncated_forward(self):
        model = _make_nested_model(max_hidden_layers=4, uncertainty_method=UncertaintyMethod.CONSTANT)
        model.model.eval()
        x = torch.randn(37, 3, device=model.device)

        with torch.no_grad():
            all_outputs = model.strategy.all_depth_outputs(x)  # (N, D, out_features)

        assert tuple(all_outputs.shape) == (37, 4, 1)

        max_abs_err = 0.0
        with torch.no_grad():
            for depth_idx in range(model.max_hidden_layers):
                depth = depth_idx + 1
                current = x
                for i in range(depth):
                    current = model.model.hidden_layers_blocks[i](current)
                independent = model.model.output_layer(current)
                err = (all_outputs[:, depth_idx, :] - independent).abs().max().item()
                max_abs_err = max(max_abs_err, err)

        assert max_abs_err < TOL, f"NESTED prefix property violated: max_abs_err={max_abs_err:.3e} (tol={TOL:.0e})"

    def test_forward_selected_output_matches_all_depth_gather(self):
        """The per-sample training readout must equal that sample's entry in all_depth_outputs."""
        torch.manual_seed(1)
        model = _make_nested_model(max_hidden_layers=4, uncertainty_method=UncertaintyMethod.CONSTANT)
        model.model.eval()
        x = torch.randn(19, 3, device=model.device)

        with torch.no_grad():
            all_outputs = model.strategy.all_depth_outputs(x)
            final_output, n_actual, _, _, _ = model.model(x)

        for i in range(x.size(0)):
            depth_idx = n_actual[i].item() - 1
            expected = all_outputs[i, depth_idx, :]
            assert torch.allclose(final_output[i], expected, atol=TOL), f"sample {i}: forward output != all_depth_outputs at its drawn depth"


class TestNestedAllDepthScoreTable:
    """Eval-time export: the all-depth per-sample log-likelihood table, from one forward."""

    def test_score_table_shape_and_finite(self):
        model = _make_nested_model(max_hidden_layers=4, uncertainty_method=UncertaintyMethod.PROBABILISTIC)
        model.model.eval()
        x = torch.randn(23, 3, device=model.device)
        y = torch.randn(23, device=model.device)

        with torch.no_grad():
            table = model.strategy.all_depth_log_likelihood(x, y)

        assert tuple(table.shape) == (23, 4)
        assert torch.isfinite(table).all()

    def test_score_table_matches_manual_gaussian_log_likelihood(self):
        """Column d must equal the manual Gaussian log-likelihood of the depth-d readout."""
        import math

        model = _make_nested_model(max_hidden_layers=3, uncertainty_method=UncertaintyMethod.PROBABILISTIC)
        model.model.eval()
        x = torch.randn(11, 3, device=model.device)
        y = torch.randn(11, device=model.device)

        with torch.no_grad():
            table = model.strategy.all_depth_log_likelihood(x, y)
            all_outputs = model.strategy.all_depth_outputs(x)

        for depth_idx in range(model.max_hidden_layers):
            mean = all_outputs[:, depth_idx, 0]
            log_var = all_outputs[:, depth_idx, 1]
            variance = torch.exp(log_var)
            expected = -0.5 * (math.log(2 * math.pi) + log_var + (y - mean) ** 2 / variance)
            assert torch.allclose(table[:, depth_idx], expected, atol=1e-5), f"depth={depth_idx + 1} score mismatch"


class TestNestedTrainingStep:
    """Smoke: backprop must actually flow through the per-sample drawn-depth gather."""

    def test_training_reduces_loss(self):
        torch.manual_seed(0)
        model = _make_nested_model(max_hidden_layers=3, hidden_size=16, uncertainty_method=UncertaintyMethod.PROBABILISTIC)
        model._setup_optimizers(model.model)

        x = torch.randn(64, 3, device=model.device)
        y = (0.5 * x.sum(dim=1, keepdim=True))

        model.model.train()
        first_loss = None
        last_loss = None
        for step in range(60):
            model.optimizer.zero_grad()
            final_output, _, _, _, _ = model.model(x)
            loss = nll_loss(final_output, y)
            loss.backward()
            model.optimizer.step()
            if step == 0:
                first_loss = loss.item()
            last_loss = loss.item()

        assert last_loss < first_loss, f"NESTED training did not reduce loss: first={first_loss:.4f}, last={last_loss:.4f}"

    def test_gradients_reach_every_block(self):
        """Every block must receive a nonzero gradient (each is on the trunk of some depth)."""
        torch.manual_seed(0)
        model = _make_nested_model(max_hidden_layers=3, hidden_size=16, uncertainty_method=UncertaintyMethod.PROBABILISTIC)
        x = torch.randn(64, 3, device=model.device)
        y = torch.randn(64, 1, device=model.device)

        model.model.train()
        model.model.zero_grad()
        final_output, _, _, _, _ = model.model(x)
        loss = nll_loss(final_output, y)
        loss.backward()

        for i, block in enumerate(model.model.hidden_layers_blocks):
            grads = [p.grad for p in block.parameters() if p.grad is not None]
            assert len(grads) > 0, f"block {i} received no gradient"
            assert any(g.abs().max() > 1e-9 for g in grads), f"block {i} gradients are all near-zero"


class TestNestedSmoke:
    """Smoke (NOT the F2 coherence run): a tiny NESTED model through the real `.fit()` path."""

    def test_fits_1d_data_and_exports_finite_score_table(self):
        rng = np.random.default_rng(0)
        n = 200
        x = rng.uniform(-3.0, 3.0, size=(n, 1)).astype(np.float32)
        y = (np.sin(x[:, 0]) + 0.1 * rng.standard_normal(n)).astype(np.float32)

        model = FlexibleHiddenLayersNN(
            task_type=TaskType.REGRESSION,
            max_hidden_layers=4,
            hidden_size=8,
            activation=ActivationFunction.RELU,
            layer_selection_method=LayerSelectionMethod.NESTED,
            n_predictor_layers=0,
            use_batch_norm=False,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            output_size=1,
            input_size=1,
            n_epochs=50,
            learning_rate=0.01,
            random_seed=0,
            calculate_feature_importance=False,
        )
        model.fit(x, y)

        model.model.eval()
        x_tensor = torch.tensor(x, dtype=torch.float32).to(model.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(model.device)
        with torch.no_grad():
            table = model.strategy.all_depth_log_likelihood(x_tensor, y_tensor)

        assert tuple(table.shape) == (n, 4)
        assert torch.isfinite(table).all(), "NESTED smoke: all-depth score table has non-finite entries"


class TestIndependentWeightsNested:
    """The conditioned control: IndependentWeightsFlexibleNN must also accept NESTED."""

    def _make(self, max_hidden_layers: int = 4, hidden_size: int = 8, uncertainty_method: UncertaintyMethod = UncertaintyMethod.PROBABILISTIC) -> IndependentWeightsFlexibleNN:
        model = IndependentWeightsFlexibleNN(
            task_type=TaskType.REGRESSION,
            max_hidden_layers=max_hidden_layers,
            hidden_size=hidden_size,
            activation=ActivationFunction.RELU,
            layer_selection_method=LayerSelectionMethod.NESTED,
            n_predictor_layers=0,
            use_batch_norm=False,
            uncertainty_method=uncertainty_method,
            output_size=1,
            input_size=3,
            random_seed=0,
        )
        model.build_model()
        return model

    def test_constructs_and_forward(self):
        model = self._make()
        x = torch.randn(50, 3, device=model.device)
        model.model.train()
        final_output, n_actual, n_probs, _, _ = model.model(x)

        assert tuple(final_output.shape) == (50, 2)
        assert tuple(n_probs.shape) == (50, 4)
        # one-hot per row: the draw is hard selection, not a soft weighting.
        assert torch.allclose(n_probs.sum(dim=1), torch.ones(50, device=model.device))
        assert torch.all((n_probs == 0.0) | (n_probs == 1.0))
        assert n_actual.min().item() >= 1
        assert n_actual.max().item() <= 4

    def test_selected_output_matches_the_drawn_independent_network(self):
        """Because n_probs is one-hot, final_output must equal the drawn network's own output."""
        model = self._make(max_hidden_layers=3)
        x = torch.randn(17, 3, device=model.device)
        model.model.eval()
        with torch.no_grad():
            final_output, n_actual, _, _, _ = model.model(x)
            per_network_outputs = [net(x) for net in model.model.independent_networks]

        for i in range(x.size(0)):
            depth = n_actual[i].item()
            expected = per_network_outputs[depth - 1][i]
            assert torch.allclose(final_output[i], expected, atol=TOL), f"sample {i}: output != independent_networks[{depth - 1}] output"

    def test_draws_are_roughly_uniform(self):
        torch.manual_seed(0)
        model = self._make(max_hidden_layers=4)
        x = torch.randn(4000, 3, device=model.device)
        model.model.train()
        _, n_actual, _, _, _ = model.model(x)

        counts = torch.bincount(n_actual, minlength=5)[1:5].float()
        fractions = counts / counts.sum()
        expected = torch.full_like(fractions, 1.0 / 4)
        assert torch.allclose(fractions, expected, atol=0.03), f"depth draw histogram not roughly uniform: {fractions.tolist()}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
