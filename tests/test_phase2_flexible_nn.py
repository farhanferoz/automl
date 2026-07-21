"""Phase 2 tests: FlexibleNN gradient fixes and depth complexity control."""

import numpy as np
import pytest
import torch
from sklearn.model_selection import train_test_split

from automl_package.enums import CapacitySelection, DepthRegularization, LayerSelectionMethod, UncertaintyMethod
from automl_package.models.flexible_neural_network import FlexibleHiddenLayersNN
from automl_package.models.independent_weights_flexible_neural_network import IndependentWeightsFlexibleNN
from automl_package.utils.pytorch_utils import get_device

DEVICE = get_device()
NONZERO_GRAD_THRESHOLD = 1e-8
INDEPENDENT_WEIGHTS_MSE_TOLERANCE = 2.0


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
        assert any(g.abs().max() > NONZERO_GRAD_THRESHOLD for g in predictor_grads), "n_predictor gradients are all near-zero"


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

    def test_elbo_uniform_prior_does_not_collapse(self, piecewise_data):
        """Regression test for the M0 depth-collapse bug: the earlier linspace(3,1)
        ELBO prior caused complete depth-collapse to its argmax regardless of input
        (flexnn-moe.md Done ledger M0). With the uniform prior, depth selection on a
        2-cluster dataset (linear region wants shallow, sinusoidal wants deep) must
        stay input-dependent rather than collapsing to a single prior-favored depth.
        """
        x, y, _ = piecewise_data
        model = FlexibleHiddenLayersNN(
            input_size=1, output_size=1,
            layer_selection_method=LayerSelectionMethod.GUMBEL_SOFTMAX,
            max_hidden_layers=4, n_predictor_layers=1, hidden_size=64,
            n_epochs=120, depth_regularization=DepthRegularization.ELBO, random_seed=42,
            calculate_feature_importance=False,
        )
        model.fit(x, y)

        x_tensor = torch.tensor(x, dtype=torch.float32).to(model.device)
        model.model.eval()
        with torch.no_grad():
            _, n_actual, _, _, _ = model.model(x_tensor)

        n_depth = n_actual.float().cpu().numpy().ravel()
        assert n_depth.std() > 0, (
            "All samples selected the same depth -- ELBO posterior collapsed to a single "
            "value regardless of input (the M0 depth-collapse failure mode)."
        )

        x_flat = x.ravel()
        mean_depth_linear = n_depth[x_flat < 0].mean()
        mean_depth_sinusoidal = n_depth[x_flat >= 0].mean()
        assert mean_depth_linear != mean_depth_sinusoidal, (
            f"Depth selection identical across clusters (linear={mean_depth_linear:.2f}, "
            f"sinusoidal={mean_depth_sinusoidal:.2f}) -- posterior is prior-dominated, not input-dependent."
        )

    def test_depth_penalty_reduces_mean_depth(self, piecewise_data):
        """Depth penalty should reduce mean selected depth vs no regularization.

        Needs a problem where the unregularised optimum actually wants depth>1,
        otherwise the penalty has nothing to compress. Piecewise data with a
        nonlinear branch forces the baseline toward deeper layers.
        """
        x, y, _ = piecewise_data

        model_none = FlexibleHiddenLayersNN(
            input_size=1, output_size=1,
            layer_selection_method=LayerSelectionMethod.GUMBEL_SOFTMAX,
            max_hidden_layers=5, n_predictor_layers=1, hidden_size=32,
            n_epochs=40, depth_regularization=DepthRegularization.NONE, random_seed=42,
            calculate_feature_importance=False,
        )
        model_none.fit(x, y)

        model_pen = FlexibleHiddenLayersNN(
            input_size=1, output_size=1,
            layer_selection_method=LayerSelectionMethod.GUMBEL_SOFTMAX,
            max_hidden_layers=5, n_predictor_layers=1, hidden_size=32,
            n_epochs=40, depth_regularization=DepthRegularization.DEPTH_PENALTY,
            depth_penalty_weight=1.0, random_seed=42,
            calculate_feature_importance=False,
        )
        model_pen.fit(x, y)

        x_tensor = torch.tensor(x, dtype=torch.float32).to(model_none.device)
        with torch.no_grad():
            _, n_none, _, _, _ = model_none.model(x_tensor)
            _, n_pen, _, _, _ = model_pen.model(x_tensor)

        assert n_pen.float().mean() <= n_none.float().mean() + 0.1, (
            f"Depth penalty did not reduce mean depth (pen={n_pen.float().mean():.3f}, "
            f"none={n_none.float().mean():.3f})"
        )


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

        assert mse < INDEPENDENT_WEIGHTS_MSE_TOLERANCE, f"IndependentWeightsFlexibleNN MSE ({mse:.4f}) too high on y=2x+1 data."

    def test_per_input_depth_variation(self, piecewise_data):
        """On piecewise data (linear x<0, sinusoidal x>=0), n_predictor should learn
        input-dependent depth: the selected depth must vary across samples (not collapse
        to a single value) and differ between the linear and sinusoidal regions.

        This originally asserted the specific direction mean_depth_linear <
        mean_depth_sinusoidal. That direction was an artifact of the removed linspace(3,1)
        ELBO prior's depth-collapse (flexnn-moe.md M0; superseded by the uniform-prior fix
        in flexible_neural_network.py). Under the corrected uniform prior the direction is
        not a robust property of this toy -- it flips roughly 50/50 across seeds under
        GumbelSoftmax (3/6) and is no better under SoftGating (4/6), and both fail at the
        pinned seed=42 -- so a fixed-direction assertion is a seed lottery, not a
        capability check. The stable claim the test actually means to make (the model
        allocates depth as an input-dependent resource rather than collapsing to a
        prior-favored constant) is captured by non-collapse plus a cross-region
        difference.
        """
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

        assert n_depth.std() > 0, (
            "Depth selection collapsed to a single value for all inputs -- n_predictor is "
            "not learning input-dependent depth (the M0 depth-collapse failure mode)."
        )

        mean_depth_linear = n_depth[x_flat < 0].mean()
        mean_depth_sinusoidal = n_depth[x_flat >= 0].mean()

        assert mean_depth_linear != mean_depth_sinusoidal, (
            f"Depth allocation identical across regions (linear={mean_depth_linear:.2f}, "
            f"sinusoidal={mean_depth_sinusoidal:.2f}) -- selection is prior-dominated, "
            f"not input-dependent."
        )


class TestDD1IndependentWeightsUniformPrior:
    """DD1 regression test: `IndependentWeightsFlexibleNN`'s ELBO/cost-aware priors must be
    uniform, matching the fix already applied to its shared-weights sibling
    (`FlexibleHiddenLayersNN` -- see the M0 depth-collapse removal comment in
    `flexible_neural_network.py`). The earlier `linspace(3.0, 1.0, ...)` prefer-shallow prior
    reached one twin and not the other.
    """

    def test_elbo_uniform_prior_is_not_shallow_skewed(self, piecewise_data):
        """The selected-depth distribution's shallow/deep balance is the quantity DD1's fix
        changes: the removed `linspace(3,1)` prior is monotonically higher-mass at low depth,
        which empirically skews `IndependentWeightsFlexibleNN`'s ELBO selection toward depth 1
        on this toy (measured at seed=42: depths [1,2,3,4] land counts [204,118,105,73] under
        the unfixed prior -- 2.8x more mass on the shallowest depth than the deepest). The
        uniform prior removes that structural bias and reverses the skew (measured:
        [103,113,137,147]).

        Full posterior collapse to a single value -- the M0 failure mode already fixed on the
        shared-weights sibling `FlexibleHiddenLayersNN` -- does not reproduce reliably on this
        independent-networks architecture at this scale (std stays > 0 under both the buggy and
        fixed priors here), so that is not usable as the discriminating assertion for this
        class. The directional bias the removed prior imposed does reproduce reliably and is
        the actual mechanism DD1's diff changes, so it is what this test asserts on.
        """
        x, y, _ = piecewise_data
        model = IndependentWeightsFlexibleNN(
            input_size=1, output_size=1,
            layer_selection_method=LayerSelectionMethod.GUMBEL_SOFTMAX,
            max_hidden_layers=4, n_predictor_layers=1, hidden_size=64,
            n_epochs=120, depth_regularization=DepthRegularization.ELBO, random_seed=42,
            calculate_feature_importance=False,
        )
        model.fit(x, y)

        x_tensor = torch.tensor(x, dtype=torch.float32).to(model.device)
        model.model.eval()
        with torch.no_grad():
            _, n_actual, _, _, _ = model.model(x_tensor)

        n_depth = n_actual.float().cpu().numpy().ravel()
        assert n_depth.std() > 0, (
            "All samples selected the same depth -- ELBO posterior collapsed to a single value."
        )

        depths, counts = np.unique(n_depth, return_counts=True)
        count_by_depth = dict(zip(depths.astype(int).tolist(), counts.tolist(), strict=True))
        shallowest, deepest = int(depths.min()), int(depths.max())
        assert count_by_depth[deepest] >= count_by_depth[shallowest], (
            f"Depth selection is skewed toward the shallowest depth (depth={shallowest}: "
            f"{count_by_depth[shallowest]} samples, depth={deepest}: {count_by_depth[deepest]} "
            f"samples) -- consistent with the removed prefer-shallow linspace(3,1) prior (DD1)."
        )


class TestDD2PairedDepthUncertainty:
    """DD2 regression test: `predict_uncertainty` must read its spread off the same per-sample
    depth `predict()` used for the mean, not silently always use the soft selection. Assert
    directly against a manual `forward_at_depth` computation at each sample's actual selected
    depth -- the quantity the fix changes -- not a coarse downstream view.
    """

    def _make_model(self, **overrides):
        defaults = {
            "input_size": 1, "output_size": 1,
            "layer_selection_method": LayerSelectionMethod.GUMBEL_SOFTMAX,
            "max_hidden_layers": 3, "n_predictor_layers": 1, "hidden_size": 16,
            "uncertainty_method": UncertaintyMethod.PROBABILISTIC,
            "n_epochs": 15, "random_seed": 42, "calculate_feature_importance": False,
        }
        defaults.update(overrides)
        return FlexibleHiddenLayersNN(**defaults)

    def test_hard_mode_uncertainty_matches_hard_mode_depths(self, simple_linear_data):
        """`predict_uncertainty(hard_execution=True)` must read `log_var` off the same
        per-sample argmax depth `predict(hard_execution=True)` uses -- not the soft mixture.
        """
        x, y = simple_linear_data
        model = self._make_model()
        model.fit(x, y)

        x_tensor = torch.tensor(x, dtype=torch.float32).to(model.device)
        model.model.eval()
        with torch.no_grad():
            n_logits = model.model.n_predictor(x_tensor)
            n_actual = torch.argmax(n_logits, dim=1) + 1  # 1-indexed depth, matches hard_forward

        got = model.predict_uncertainty(x, hard_execution=True)

        expected = np.empty_like(got)
        with torch.no_grad():
            for depth in torch.unique(n_actual).tolist():
                mask = (n_actual == depth).cpu().numpy()
                x_subset = x_tensor[torch.as_tensor(mask, device=model.device)]
                raw = model.model.forward_at_depth(x_subset, depth)
                expected[mask] = torch.exp(0.5 * raw[:, 1]).cpu().numpy()

        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)

    def test_routed_mode_uncertainty_matches_routed_depths(self, simple_linear_data):
        """Same check for `CapacitySelection.PER_INPUT` -- the literal motivating case from the
        plan's DD2 description: PER_INPUT-routed `predict()` paired with an always-soft
        `predict_uncertainty`.
        """
        x, y = simple_linear_data
        model = self._make_model(capacity_selection=CapacitySelection.PER_INPUT)
        model.fit(x, y)
        router = model.fit_router(x, y)

        routed_depths = np.array([capacity[0] for capacity in router.route(x)], dtype=np.int64)
        got = model.predict_uncertainty(x)

        x_tensor = torch.tensor(x, dtype=torch.float32).to(model.device)
        expected = np.empty_like(got)
        with torch.no_grad():
            for depth in np.unique(routed_depths).tolist():
                mask = routed_depths == depth
                x_subset = x_tensor[torch.as_tensor(mask, device=model.device)]
                raw = model.model.forward_at_depth(x_subset, depth)
                expected[mask] = torch.exp(0.5 * raw[:, 1]).cpu().numpy()

        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)

    def test_routed_uncertainty_without_fitted_router_raises(self, simple_linear_data):
        x, y = simple_linear_data
        model = self._make_model(n_epochs=1, capacity_selection=CapacitySelection.PER_INPUT)
        model.fit(x, y)
        with pytest.raises(RuntimeError, match="fit_router"):
            model.predict_uncertainty(x)

    def test_inference_mode_kwarg_rejected(self, simple_linear_data):
        """FP-3.b: `inference_mode` is removed entirely -- no `**kwargs` on `predict_uncertainty`,
        so passing it raises `TypeError` for free once the parameter is deleted."""
        x, y = simple_linear_data
        model = self._make_model(n_epochs=1)
        model.fit(x, y)
        with pytest.raises(TypeError):
            model.predict_uncertainty(x, inference_mode="routed")


class TestNPredictorInMainOptimizer:
    """For non-REINFORCE strategies, n_predictor weights must train via backprop,
    so they must be included in the main optimizer. Reinforce keeps its own policy
    optimizer.
    """

    @pytest.mark.parametrize("method", [
        "soft_gating", "gumbel_softmax", "ste", "none",
    ])
    def test_n_predictor_params_in_main_optimizer(self, method):
        import torch
        import numpy as np
        from automl_package.enums import LayerSelectionMethod, UncertaintyMethod
        from automl_package.models.flexible_neural_network import FlexibleHiddenLayersNN

        np.random.seed(0)
        x = np.random.randn(64, 3).astype(np.float32)
        y = np.random.randn(64).astype(np.float32)

        layer_method = LayerSelectionMethod(method)
        predictor_layers = 0 if layer_method == LayerSelectionMethod.NONE else 1
        m = FlexibleHiddenLayersNN(
            input_size=3, max_hidden_layers=4,
            layer_selection_method=layer_method, n_predictor_layers=predictor_layers,
            uncertainty_method=UncertaintyMethod.CONSTANT,
            n_epochs=2, learning_rate=0.01, random_seed=42,
            calculate_feature_importance=False,
        )
        m.build_model()
        m._setup_optimizers(m.model)

        if m.model.n_predictor is None:
            return  # NONE strategy can omit n_predictor
        opt_ids = {id(p) for g in m.optimizer.param_groups for p in g["params"]}
        pred_ids = {id(p) for p in m.model.n_predictor.parameters()}
        assert pred_ids <= opt_ids, (
            f"n_predictor params missing from main optimizer for layer_selection={method}. "
            "Non-REINFORCE strategies train n_predictor by backprop and must be in the main optimizer."
        )

    def test_reinforce_uses_separate_policy_optimizer(self):
        import numpy as np
        from automl_package.enums import LayerSelectionMethod, UncertaintyMethod
        from automl_package.models.flexible_neural_network import FlexibleHiddenLayersNN

        np.random.seed(0)
        x = np.random.randn(64, 3).astype(np.float32)
        y = np.random.randn(64).astype(np.float32)
        m = FlexibleHiddenLayersNN(
            input_size=3, max_hidden_layers=4,
            layer_selection_method=LayerSelectionMethod.REINFORCE,
            uncertainty_method=UncertaintyMethod.CONSTANT,
            n_epochs=2, learning_rate=0.01, random_seed=42,
            calculate_feature_importance=False,
        )
        m.build_model(); m._setup_optimizers(m.model)
        opt_ids = {id(p) for g in m.optimizer.param_groups for p in g["params"]}
        pred_ids = {id(p) for p in m.model.n_predictor.parameters()}
        assert pred_ids.isdisjoint(opt_ids), "REINFORCE must NOT place n_predictor in main optimizer"
        assert m.strategy.policy_optimizer is not None


class TestConvergenceSummary:
    """Task F4: `convergence_summary_`/`trustworthy` exposure, promoted from `utils/convergence.py`."""

    def test_diverging_fit_is_not_trustworthy(self):
        """An absurdly high learning rate must blow up the held-out loss -> trustworthy=False."""
        rng = np.random.default_rng(42)
        x = rng.normal(size=(64, 1)).astype(np.float32)
        y = (x[:, 0] * 2.0).astype(np.float32)
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.25, random_state=42)

        model = FlexibleHiddenLayersNN(
            input_size=1, output_size=1,
            layer_selection_method=LayerSelectionMethod.NONE,
            max_hidden_layers=1, n_predictor_layers=0, hidden_size=16,
            n_epochs=50, learning_rate=10.0, early_stopping_rounds=5, random_seed=42,
        )
        model._fit_single(x_train, y_train, x_val=x_val, y_val=y_val)

        assert hasattr(model, "convergence_summary_")
        assert model.convergence_summary_["trustworthy"] is False
        assert model.get_params()["trustworthy"] is False


class TestCapacityRouterPerInput:
    """`fit_router()` + `predict()` under `CapacitySelection.PER_INPUT` -- capacity-programme Task FP-3."""

    def _make_model(self, **overrides):
        defaults = dict(
            input_size=1, output_size=1,
            layer_selection_method=LayerSelectionMethod.GUMBEL_SOFTMAX,
            max_hidden_layers=3, n_predictor_layers=1, hidden_size=16,
            n_epochs=10, random_seed=42, calculate_feature_importance=False,
            capacity_selection=CapacitySelection.PER_INPUT,
        )
        defaults.update(overrides)
        return FlexibleHiddenLayersNN(**defaults)

    def test_routed_predict_shape_and_no_nan(self, simple_linear_data):
        """FP-3 test 1: a router-fitted model constructed with `CapacitySelection.PER_INPUT`
        routes on a plain `predict(x)` call, with no caller flag."""
        x, y = simple_linear_data
        model = self._make_model()
        model.fit(x, y)
        model.fit_router(x, y)

        y_pred = model.predict(x)
        assert y_pred.shape == (len(x),)
        assert not np.any(np.isnan(y_pred))

    def test_routed_without_fitted_router_raises(self, simple_linear_data):
        x, y = simple_linear_data
        model = self._make_model(n_epochs=1)
        model.fit(x, y)
        with pytest.raises(RuntimeError, match="fit_router"):
            model.predict(x)

    def test_inference_mode_kwarg_rejected_at_predict(self, simple_linear_data):
        """FP-3.b: `inference_mode` is removed entirely -- `predict` has no `**kwargs`, so passing
        it raises `TypeError` for free once the parameter is deleted."""
        x, y = simple_linear_data
        model = self._make_model(n_epochs=1)
        model.fit(x, y)
        with pytest.raises(TypeError):
            model.predict(x, inference_mode="routed")

    def test_inference_mode_kwarg_rejected_at_construction(self):
        """FP-3.d: a removed selection kwarg passed to the CONSTRUCTOR must raise `TypeError`, not
        be silently swallowed into `self.params` (`BaseModel.__init__`, `base.py:45,52`)."""
        with pytest.raises(TypeError):
            FlexibleHiddenLayersNN(input_size=1, output_size=1, inference_mode="hard")

    def test_hard_execution_matches_todays_hard_forward(self, simple_linear_data):
        """FP-3 test 5: the `hard_execution` boolean survives (orthogonal to `CapacitySelection`,
        not swept up as a selection mode) and produces the same predictions `hard_forward` always
        has."""
        x, y = simple_linear_data
        model = self._make_model(capacity_selection=CapacitySelection.FIXED)
        model.fit(x, y)

        x_tensor = torch.tensor(x, dtype=torch.float32).to(model.device)
        model.model.eval()
        with torch.no_grad():
            expected = model.model.hard_forward(x_tensor).cpu().numpy().flatten()

        got = model.predict(x, hard_execution=True)
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)

    def test_router_routes_only_within_capacity_grid(self, simple_linear_data):
        x, y = simple_linear_data
        model = self._make_model()
        model.fit(x, y)
        router = model.fit_router(x, y)

        routed_depths = {capacity[0] for capacity in router.route(x)}
        assert routed_depths.issubset(set(range(1, model.max_hidden_layers + 1)))

    def test_fit_router_default_cost_fn_uses_s2_executed_flops(self, simple_linear_data):
        """`fit_router`'s default `cost_fn` must come from `capacity_accounting.executed_flops`
        (S2 accounting), not a re-derived FLOPs formula -- confirm the two agree at every depth.
        """
        from automl_package.examples.capacity_accounting import executed_flops

        x, y = simple_linear_data
        model = self._make_model()
        model.fit(x, y)
        router = model.fit_router(x, y)

        for depth in range(1, model.max_hidden_layers + 1):
            assert router.costs_[depth - 1] == pytest.approx(executed_flops(model.model, depth))
