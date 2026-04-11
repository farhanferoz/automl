"""Phase 2 tests: FlexibleNN gradient fixes and depth complexity control."""

import numpy as np
import pytest
import torch
from sklearn.model_selection import train_test_split

from automl_package.enums import DepthRegularization, LayerSelectionMethod, UncertaintyMethod
from automl_package.models.flexible_neural_network import FlexibleHiddenLayersNN
from automl_package.models.independent_weights_flexible_neural_network import IndependentWeightsFlexibleNN
from automl_package.utils.pytorch_utils import get_device

DEVICE = get_device()


class TestBug5ReinforceLogProb:
    """Verify REINFORCE log_prob is in the correct tuple position."""

    def test_reinforce_log_prob_nonzero(self):
        """The log_prob returned by REINFORCE should be non-trivial."""
        model = FlexibleHiddenLayersNN(
            input_size=1, output_size=1,
            layer_selection_method=LayerSelectionMethod.REINFORCE,
            max_hidden_layers=3, n_predictor_layers=1, hidden_size=16,
            n_epochs=1, random_seed=42,
        )
        model.build_model()
        x = torch.randn(8, 1, device=DEVICE)
        model.model.train()
        _output, _n, _probs, _logits, log_prob = model.model(x)

        assert log_prob is not None
        assert not torch.all(log_prob == 0.0), "log_prob is all zeros -- Bug 5 still present"

    def test_reinforce_policy_gradient_nonzero(self):
        """Policy loss from REINFORCE should produce nonzero gradient on n_predictor."""
        model = FlexibleHiddenLayersNN(
            input_size=1, output_size=1,
            layer_selection_method=LayerSelectionMethod.REINFORCE,
            max_hidden_layers=3, n_predictor_layers=1, hidden_size=16,
            n_epochs=1, random_seed=42,
        )
        model.build_model()
        x = torch.randn(8, 1, device=DEVICE)
        y = torch.randn(8, 1, device=DEVICE)

        model.model.train()
        model.model.zero_grad()
        final_output, _, _, _, log_prob = model.model(x)
        main_loss = torch.nn.MSELoss()(final_output, y)
        reward = -main_loss.detach()
        policy_loss = -log_prob * reward
        total_loss = main_loss + policy_loss.mean()
        total_loss.backward()

        predictor_grads = [p.grad for p in model.model.n_predictor.parameters() if p.grad is not None]
        assert len(predictor_grads) > 0, "n_predictor received no gradients"
        assert any(g.abs().max() > 1e-8 for g in predictor_grads), "n_predictor gradients are all near-zero"


class TestBug6SteGradientFlow:
    """Verify STE passes gradients to n_predictor."""

    def test_ste_n_predictor_gradient_nonzero(self):
        """After backward pass, STE n_predictor should have non-zero gradients."""
        model = FlexibleHiddenLayersNN(
            input_size=1, output_size=1,
            layer_selection_method=LayerSelectionMethod.STE,
            max_hidden_layers=3, n_predictor_layers=1, hidden_size=16,
            n_epochs=1, random_seed=42,
        )
        model.build_model()
        x = torch.randn(16, 1, device=DEVICE)
        y = torch.randn(16, 1, device=DEVICE)

        model.model.train()
        model.model.zero_grad()
        final_output, _, _, _, _ = model.model(x)
        loss = torch.nn.MSELoss()(final_output, y)
        loss.backward()

        predictor_grads = [p.grad for p in model.model.n_predictor.parameters() if p.grad is not None]
        assert len(predictor_grads) > 0, "STE n_predictor received no gradients -- Bug 6 still present"


class TestBug7IndependentWeightsSte:
    """Verify IndependentWeights STE returns proper one-hot."""

    def test_ste_returns_hard_selection(self):
        """STE strategy should return near-one-hot n_probs, not soft probabilities."""
        model = IndependentWeightsFlexibleNN(
            input_size=1, output_size=1,
            layer_selection_method=LayerSelectionMethod.STE,
            max_hidden_layers=3, n_predictor_layers=1, hidden_size=16,
            n_epochs=1, random_seed=42,
        )
        model.build_model()
        x = torch.randn(8, 1, device=DEVICE)

        model.model.eval()
        with torch.no_grad():
            _, _, n_probs, _, _ = model.model(x)

        for row in n_probs:
            assert torch.sum(row == 1.0) == 1, (
                f"STE n_probs row is not one-hot: {row.tolist()}. Bug 7 still present -- returning soft probs."
            )


class TestDepthComplexityControl:
    """Tests for ELBO and depth penalty mechanisms."""

    def test_elbo_prefers_shallow_on_linear(self, simple_linear_data):
        """With ELBO, linear data should result in shallower depth selections."""
        x, y = simple_linear_data
        model = FlexibleHiddenLayersNN(
            input_size=1, output_size=1,
            layer_selection_method=LayerSelectionMethod.GUMBEL_SOFTMAX,
            max_hidden_layers=3, n_predictor_layers=1, hidden_size=32,
            n_epochs=50, depth_regularization=DepthRegularization.ELBO, random_seed=42,
            calculate_feature_importance=False,
        )
        model.fit(x, y)

        x_tensor = torch.tensor(x, dtype=torch.float32).to(model.device)
        model.model.eval()
        with torch.no_grad():
            _, n_actual, _, _, _ = model.model(x_tensor)

        mean_depth = n_actual.float().mean().item()
        assert mean_depth < model.max_hidden_layers, (
            f"Mean depth {mean_depth:.2f} == max {model.max_hidden_layers}. ELBO complexity control not working."
        )

    def test_depth_penalty_reduces_mean_depth(self, simple_linear_data):
        """Depth penalty should reduce mean selected depth vs no regularization."""
        x, y = simple_linear_data

        model_none = FlexibleHiddenLayersNN(
            input_size=1, output_size=1,
            layer_selection_method=LayerSelectionMethod.GUMBEL_SOFTMAX,
            max_hidden_layers=3, n_predictor_layers=1, hidden_size=32,
            n_epochs=50, depth_regularization=DepthRegularization.NONE, random_seed=42,
            calculate_feature_importance=False,
        )
        model_none.fit(x, y)

        model_pen = FlexibleHiddenLayersNN(
            input_size=1, output_size=1,
            layer_selection_method=LayerSelectionMethod.GUMBEL_SOFTMAX,
            max_hidden_layers=3, n_predictor_layers=1, hidden_size=32,
            n_epochs=50, depth_regularization=DepthRegularization.DEPTH_PENALTY,
            depth_penalty_weight=0.05, random_seed=42,
            calculate_feature_importance=False,
        )
        model_pen.fit(x, y)

        x_tensor = torch.tensor(x, dtype=torch.float32).to(model_none.device)
        with torch.no_grad():
            _, n_none, _, _, _ = model_none.model(x_tensor)
            _, n_pen, _, _, _ = model_pen.model(x_tensor)

        assert n_pen.float().mean() <= n_none.float().mean(), "Depth penalty did not reduce mean depth"


class TestFlexibleNNSmoke:
    """Smoke tests: all strategies should train and predict."""

    @pytest.mark.parametrize("method", [
        LayerSelectionMethod.GUMBEL_SOFTMAX,
        LayerSelectionMethod.SOFT_GATING,
        LayerSelectionMethod.STE,
        LayerSelectionMethod.REINFORCE,
    ])
    def test_shared_weights_trains(self, simple_linear_data, method):
        """Each strategy should train without crash."""
        x, y = simple_linear_data
        model = FlexibleHiddenLayersNN(
            input_size=1, output_size=1,
            layer_selection_method=method,
            max_hidden_layers=3, n_predictor_layers=1, hidden_size=16, n_epochs=10, random_seed=42,
            calculate_feature_importance=False,
        )
        model.fit(x, y)
        y_pred = model.predict(x)
        assert y_pred.shape == (len(x),)
        assert not np.any(np.isnan(y_pred))

    @pytest.mark.parametrize("method", [
        LayerSelectionMethod.GUMBEL_SOFTMAX,
        LayerSelectionMethod.SOFT_GATING,
        LayerSelectionMethod.STE,
    ])
    def test_independent_weights_trains(self, simple_linear_data, method):
        """Each strategy should train without crash."""
        x, y = simple_linear_data
        model = IndependentWeightsFlexibleNN(
            input_size=1, output_size=1,
            layer_selection_method=method,
            max_hidden_layers=3, n_predictor_layers=1, hidden_size=16, n_epochs=10, random_seed=42,
            calculate_feature_importance=False,
        )
        model.fit(x, y)
        y_pred = model.predict(x)
        assert y_pred.shape == (len(x),)
        assert not np.any(np.isnan(y_pred))


class TestFlexibleNNModelComparison:
    """Model-level tests: verify FlexibleNN achieves expected relative performance on designed problems."""

    def _make_flexible(self, method=LayerSelectionMethod.GUMBEL_SOFTMAX, depth_reg=DepthRegularization.NONE, **kwargs):
        defaults = dict(
            input_size=1, output_size=1, max_hidden_layers=3, n_predictor_layers=1,
            hidden_size=32, n_epochs=80, learning_rate=0.01, early_stopping_rounds=15,
            validation_fraction=0.2, random_seed=42, calculate_feature_importance=False,
            layer_selection_method=method, depth_regularization=depth_reg,
        )
        defaults.update(kwargs)
        return FlexibleHiddenLayersNN(**defaults)

    def _make_fixed_nn(self, hidden_layers=1):
        from automl_package.models.neural_network import PyTorchNeuralNetwork
        return PyTorchNeuralNetwork(
            input_size=1, output_size=1, hidden_layers=hidden_layers, hidden_size=32,
            learning_rate=0.01, n_epochs=80, early_stopping_rounds=15,
            validation_fraction=0.2, uncertainty_method=UncertaintyMethod.CONSTANT,
            random_seed=42,
        )

    def test_flexible_mse_reasonable_on_linear(self, simple_linear_data):
        """On linear data, FlexibleNN should achieve MSE comparable to a fixed 1-layer NN."""
        x, y = simple_linear_data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        fixed_nn = self._make_fixed_nn(hidden_layers=1)
        fixed_nn.fit(x_train, y_train)
        fixed_pred = fixed_nn.predict(x_test)
        fixed_mse = float(np.mean((y_test - fixed_pred) ** 2))

        flex = self._make_flexible()
        flex.fit(x_train, y_train)
        flex_pred = flex.predict(x_test)
        flex_mse = float(np.mean((y_test - flex_pred) ** 2))

        assert flex_mse < fixed_mse * 3.0, (
            f"FlexibleNN MSE ({flex_mse:.4f}) is >3x worse than fixed 1-layer NN ({fixed_mse:.4f}) "
            f"on simple linear data. Depth selection may be hurting."
        )

    def test_flexible_beats_shallow_on_complex_data(self, piecewise_data):
        """On piecewise data, FlexibleNN with depth should beat a 1-layer NN."""
        x, y, _ = piecewise_data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        shallow = self._make_fixed_nn(hidden_layers=1)
        shallow.fit(x_train, y_train)
        shallow_pred = shallow.predict(x_test)
        shallow_mse = float(np.mean((y_test - shallow_pred) ** 2))

        flex = self._make_flexible(n_epochs=100)
        flex.fit(x_train, y_train)
        flex_pred = flex.predict(x_test)
        flex_mse = float(np.mean((y_test - flex_pred) ** 2))

        assert flex_mse < shallow_mse, (
            f"FlexibleNN MSE ({flex_mse:.4f}) should beat shallow 1-layer NN ({shallow_mse:.4f}) "
            f"on piecewise data that requires depth."
        )

    def test_elbo_selects_shallower_on_linear_vs_complex(self, simple_linear_data, piecewise_data):
        """ELBO should select shallower depth on linear data than on piecewise data."""
        x_lin, y_lin = simple_linear_data
        x_pw, y_pw, _ = piecewise_data

        flex_lin = self._make_flexible(depth_reg=DepthRegularization.ELBO, n_epochs=60)
        flex_lin.fit(x_lin, y_lin)

        flex_pw = self._make_flexible(depth_reg=DepthRegularization.ELBO, n_epochs=60)
        flex_pw.fit(x_pw, y_pw)

        x_lin_t = torch.tensor(x_lin, dtype=torch.float32).to(flex_lin.device)
        x_pw_t = torch.tensor(x_pw, dtype=torch.float32).to(flex_pw.device)

        flex_lin.model.eval()
        flex_pw.model.eval()
        with torch.no_grad():
            _, n_lin, _, _, _ = flex_lin.model(x_lin_t)
            _, n_pw, _, _, _ = flex_pw.model(x_pw_t)

        mean_depth_lin = n_lin.float().mean().item()
        mean_depth_pw = n_pw.float().mean().item()

        assert mean_depth_lin <= mean_depth_pw, (
            f"ELBO should prefer shallower on linear (depth={mean_depth_lin:.2f}) "
            f"than piecewise (depth={mean_depth_pw:.2f})."
        )

    def test_independent_weights_mse_reasonable(self, simple_linear_data):
        """IndependentWeightsFlexibleNN should achieve reasonable MSE on linear data."""
        x, y = simple_linear_data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        model = IndependentWeightsFlexibleNN(
            input_size=1, output_size=1, max_hidden_layers=3, n_predictor_layers=1,
            hidden_size=32, n_epochs=80, learning_rate=0.01, early_stopping_rounds=15,
            validation_fraction=0.2, random_seed=42, calculate_feature_importance=False,
            layer_selection_method=LayerSelectionMethod.GUMBEL_SOFTMAX,
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        mse = float(np.mean((y_test - y_pred) ** 2))

        assert mse < 2.0, f"IndependentWeightsFlexibleNN MSE ({mse:.4f}) too high on y=2x+1 data."

    def test_per_input_depth_variation(self, piecewise_data):
        """On piecewise data (linear x<0, sinusoidal x>=0), n_predictor should select
        shallower depth for the linear region and deeper for the sinusoidal region."""
        x, y, _ = piecewise_data

        model = self._make_flexible(
            depth_reg=DepthRegularization.ELBO, n_epochs=120, hidden_size=64,
            max_hidden_layers=4,
        )
        model.fit(x, y)

        x_tensor = torch.tensor(x, dtype=torch.float32).to(model.device)
        model.model.eval()
        with torch.no_grad():
            _, n_actual, _, _, _ = model.model(x_tensor)

        n_depth = n_actual.float().cpu().numpy().ravel()
        x_flat = x.ravel()

        mean_depth_linear = n_depth[x_flat < 0].mean()
        mean_depth_sinusoidal = n_depth[x_flat >= 0].mean()

        assert mean_depth_linear < mean_depth_sinusoidal, (
            f"Expected shallower depth on linear region (x<0, depth={mean_depth_linear:.2f}) "
            f"than sinusoidal region (x>=0, depth={mean_depth_sinusoidal:.2f}). "
            f"n_predictor is not learning input-dependent depth selection."
        )
