"""Phase 4 tests: performance regression, cross-model integration, and new features.

Tests use fixed seeds and assert MSE/NLL don't degrade below known baselines.
Update thresholds only when intentional changes are made.
"""

import time

import numpy as np
import pytest
import torch
from sklearn.model_selection import train_test_split

from automl_package.enums import (
    DepthRegularization,
    LayerSelectionMethod,
    NClassesSelectionMethod,
    RegressionStrategy,
    UncertaintyMethod,
)
from automl_package.enums import ExplainerType
from automl_package.explainers.feature_explainer import FeatureExplainer
from automl_package.models.conformal import ConformalWrapper
from automl_package.models.flexible_neural_network import FlexibleHiddenLayersNN
from automl_package.models.independent_weights_flexible_neural_network import IndependentWeightsFlexibleNN
from automl_package.models.neural_network import PyTorchNeuralNetwork
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel
from automl_package.utils.transforms import symexp, symlog


# ----- Helpers ---------------------------------------------------------------


def _compute_nll(model, x, y):
    """Compute Gaussian NLL from model predictions + uncertainty."""
    y_pred = model.predict(x)
    y_std = model.predict_uncertainty(x)
    y = y.ravel() if y.ndim > 1 else y
    log_var = 2 * np.log(np.clip(y_std, 1e-6, None))
    return float(0.5 * np.mean(log_var + ((y - y_pred) ** 2) / np.exp(log_var)))


def _calibration_score(model, x, y):
    """Fraction of test points within predicted ±1σ."""
    y_pred = model.predict(x)
    y_std = model.predict_uncertainty(x)
    y = y.ravel() if y.ndim > 1 else y
    return float(np.mean(np.abs(y - y_pred) <= y_std))


def _make_probreg(**overrides):
    base = dict(
        input_size=1, n_classes=5,
        uncertainty_method=UncertaintyMethod.PROBABILISTIC,
        n_classes_selection_method=NClassesSelectionMethod.NONE,
        regression_strategy=RegressionStrategy.SEPARATE_HEADS,
        n_epochs=50, learning_rate=0.01,
        early_stopping_rounds=10, validation_fraction=0.2,
        random_seed=42, calculate_feature_importance=False,
    )
    base.update(overrides)
    return ProbabilisticRegressionModel(**base)


# ----- Performance baselines -------------------------------------------------


class TestPerformanceBaselines:
    """Lock down known performance — thresholds set from Phase 1-3 results."""

    def test_prob_regression_heteroscedastic_nll(self, heteroscedastic_data):
        x, y, _, _ = heteroscedastic_data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        model = _make_probreg()
        model.fit(x_train, y_train)
        nll = _compute_nll(model, x_test, y_test)
        # Phase 1 baseline: ProbReg_k5 NLL=1.38. Allow slack.
        assert nll < 1.8, f"ProbReg NLL ({nll:.4f}) regressed past 1.8 (known: 1.38)"

    def test_prob_regression_heteroscedastic_mse(self, heteroscedastic_data):
        x, y, _, _ = heteroscedastic_data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        model = _make_probreg()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        mse = float(np.mean((y_test - y_pred) ** 2))
        # Phase 1 baseline: ProbReg_k5 MSE=1.54. Allow slack.
        assert mse < 2.5, f"ProbReg MSE ({mse:.4f}) regressed past 2.5 (known: 1.54)"

    def test_flexible_nn_piecewise_mse(self, piecewise_data):
        x, y, _ = piecewise_data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        model = FlexibleHiddenLayersNN(
            input_size=1, output_size=1, max_hidden_layers=3, n_predictor_layers=1,
            hidden_size=32, n_epochs=100, learning_rate=0.01, early_stopping_rounds=15,
            validation_fraction=0.2, random_seed=42, calculate_feature_importance=False,
            layer_selection_method=LayerSelectionMethod.GUMBEL_SOFTMAX,
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        mse = float(np.mean((y_test - y_pred) ** 2))
        assert mse < 2.0, f"FlexibleNN piecewise MSE ({mse:.4f}) regressed past 2.0"


# ----- Cross-model ranking ---------------------------------------------------


class TestCrossModelRanking:
    """Verify expected model ranking holds on designed problems."""

    def test_probreg_beats_constant_variance_nn_on_heteroscedastic(self, heteroscedastic_data):
        x, y, _, _ = heteroscedastic_data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        probreg = _make_probreg()
        probreg.fit(x_train, y_train)
        probreg_nll = _compute_nll(probreg, x_test, y_test)

        nn = PyTorchNeuralNetwork(
            input_size=1, output_size=1, hidden_layers=2, hidden_size=32,
            learning_rate=0.01, n_epochs=50, early_stopping_rounds=10,
            validation_fraction=0.2, uncertainty_method=UncertaintyMethod.CONSTANT,
            random_seed=42, calculate_feature_importance=False,
        )
        nn.fit(x_train, y_train)
        nn_nll = _compute_nll(nn, x_test, y_test)

        assert probreg_nll < nn_nll, (
            f"ProbReg NLL ({probreg_nll:.4f}) should beat constant-variance NN ({nn_nll:.4f}) "
            f"on heteroscedastic data."
        )

    def test_flexible_nn_competitive_with_shallow(self, piecewise_data):
        x, y, _ = piecewise_data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        shallow = PyTorchNeuralNetwork(
            input_size=1, output_size=1, hidden_layers=1, hidden_size=32,
            learning_rate=0.01, n_epochs=80, early_stopping_rounds=15,
            validation_fraction=0.2, uncertainty_method=UncertaintyMethod.CONSTANT,
            random_seed=42, calculate_feature_importance=False,
        )
        shallow.fit(x_train, y_train)
        shallow_mse = float(np.mean((y_test - shallow.predict(x_test)) ** 2))

        flex = FlexibleHiddenLayersNN(
            input_size=1, output_size=1, max_hidden_layers=3, n_predictor_layers=1,
            hidden_size=32, n_epochs=100, learning_rate=0.01, early_stopping_rounds=15,
            validation_fraction=0.2, random_seed=42, calculate_feature_importance=False,
            layer_selection_method=LayerSelectionMethod.GUMBEL_SOFTMAX,
            depth_regularization=DepthRegularization.ELBO,
        )
        flex.fit(x_train, y_train)
        flex_mse = float(np.mean((y_test - flex.predict(x_test)) ** 2))

        # Allow some slack — flex should be in the same ballpark or better
        assert flex_mse <= shallow_mse * 1.2, (
            f"FlexibleNN+ELBO ({flex_mse:.4f}) should be competitive with shallow ({shallow_mse:.4f})."
        )


# ----- End-to-end pipeline ---------------------------------------------------


class TestEndToEndAllModels:
    """Full pipeline: train → predict → uncertainty for every model variant."""

    @pytest.mark.parametrize("model_factory", [
        "probreg_fixed_k",
        "flexible_nn_gumbel",
        "flexible_nn_elbo",
    ])
    def test_full_pipeline(self, heteroscedastic_data, model_factory):
        x, y, _, _ = heteroscedastic_data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        if model_factory == "probreg_fixed_k":
            model = _make_probreg(n_epochs=10)
        elif model_factory == "flexible_nn_gumbel":
            model = FlexibleHiddenLayersNN(
                input_size=1, output_size=1, max_hidden_layers=3, n_predictor_layers=1,
                hidden_size=16, n_epochs=10, random_seed=42,
                calculate_feature_importance=False,
                layer_selection_method=LayerSelectionMethod.GUMBEL_SOFTMAX,
            )
        elif model_factory == "flexible_nn_elbo":
            model = FlexibleHiddenLayersNN(
                input_size=1, output_size=1, max_hidden_layers=3, n_predictor_layers=1,
                hidden_size=16, n_epochs=10, random_seed=42,
                calculate_feature_importance=False,
                layer_selection_method=LayerSelectionMethod.GUMBEL_SOFTMAX,
                depth_regularization=DepthRegularization.ELBO,
            )

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        assert y_pred.shape == (len(y_test),)
        assert not np.any(np.isnan(y_pred))

        if hasattr(model, "predict_uncertainty"):
            y_std = model.predict_uncertainty(x_test)
            assert y_std.shape == (len(y_test),)
            assert np.all(y_std >= 0), "Predicted std should be non-negative"


# ----- β-NLL loss ------------------------------------------------------------


class TestBetaNLL:
    """Tests for β-NLL loss variant."""

    def test_beta_nll_trains(self, heteroscedastic_data):
        """β-NLL should train without crash and produce finite predictions."""
        x, y, _, _ = heteroscedastic_data
        x_train, x_test, y_train, _ = train_test_split(x, y, test_size=0.3, random_state=42)
        model = _make_probreg(loss_type="beta_nll", beta=0.5)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        assert not np.any(np.isnan(y_pred))
        y_std = model.predict_uncertainty(x_test)
        assert np.all(y_std >= 0)

    def test_beta_nll_uncertainty_correlates_with_noise(self, heteroscedastic_data):
        """β-NLL should still learn input-dependent variance."""
        x, y, _, noise = heteroscedastic_data
        x_train, x_test, y_train, _ = train_test_split(x, y, test_size=0.3, random_state=42)
        _, noise_test = train_test_split(noise, test_size=0.3, random_state=42)
        model = _make_probreg(loss_type="beta_nll", beta=0.5)
        model.fit(x_train, y_train)
        y_std = model.predict_uncertainty(x_test)
        r = float(np.corrcoef(noise_test.ravel(), y_std.ravel())[0, 1])
        assert r > 0.2, f"β-NLL learned variance correlates poorly with noise (r={r:.3f})"


# ----- Hard inference --------------------------------------------------------


class TestHardInference:
    """Tests for inference-time hard selection optimization."""

    def test_hard_inference_predictions_close_to_soft(self, piecewise_data):
        x, y, _ = piecewise_data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        model = FlexibleHiddenLayersNN(
            input_size=1, output_size=1, max_hidden_layers=3, n_predictor_layers=1,
            hidden_size=32, n_epochs=80, learning_rate=0.01, early_stopping_rounds=15,
            validation_fraction=0.2, random_seed=42, calculate_feature_importance=False,
            layer_selection_method=LayerSelectionMethod.GUMBEL_SOFTMAX,
        )
        model.fit(x_train, y_train)

        y_soft = model.predict(x_test, inference_mode="soft")
        y_hard = model.predict(x_test, inference_mode="hard")

        assert y_soft.shape == y_hard.shape
        assert not np.any(np.isnan(y_hard))
        # Hard selects argmax depth — predictions should track soft reasonably
        mse_diff = float(np.mean((y_soft - y_hard) ** 2))
        var_y = float(np.var(y_test))
        assert mse_diff < var_y, (
            f"Hard vs soft MSE diff ({mse_diff:.4f}) exceeds target variance ({var_y:.4f})"
        )

    def test_hard_inference_runs_on_large_batch(self, piecewise_data):
        """Hard inference should not crash and complete in reasonable time on large batch."""
        x, y, _ = piecewise_data
        x_train, x_test, y_train, _ = train_test_split(x, y, test_size=0.3, random_state=42)
        model = FlexibleHiddenLayersNN(
            input_size=1, output_size=1, max_hidden_layers=4, n_predictor_layers=1,
            hidden_size=64, n_epochs=30, learning_rate=0.01, early_stopping_rounds=10,
            validation_fraction=0.2, random_seed=42, calculate_feature_importance=False,
            layer_selection_method=LayerSelectionMethod.GUMBEL_SOFTMAX,
        )
        model.fit(x_train, y_train)
        x_large = np.tile(x_test, (20, 1))

        t0 = time.perf_counter()
        for _ in range(5):
            y_hard = model.predict(x_large, inference_mode="hard")
        t_hard = time.perf_counter() - t0

        assert y_hard.shape == (len(x_large),)
        assert t_hard < 30.0, f"Hard inference took too long: {t_hard:.2f}s"


# ----- Conformal prediction --------------------------------------------------


class TestConformalWrapper:
    """Tests for split conformal prediction wrapper."""

    def test_conformal_coverage(self, heteroscedastic_data):
        x, y, _, _ = heteroscedastic_data
        # 50/30/20 train/cal/test
        x_train, x_rest, y_train, y_rest = train_test_split(x, y, test_size=0.5, random_state=42)
        x_cal, x_test, y_cal, y_test = train_test_split(x_rest, y_rest, test_size=0.4, random_state=42)

        model = _make_probreg(n_epochs=30)
        model.fit(x_train, y_train)

        cw = ConformalWrapper(model)
        alpha = 0.1
        cw.calibrate(x_cal, y_cal, alpha=alpha)
        lower, upper = cw.predict_interval(x_test)

        coverage = float(np.mean((y_test >= lower) & (y_test <= upper)))
        assert coverage >= (1 - alpha) - 0.10, (
            f"Conformal coverage ({coverage:.3f}) below target {1 - alpha:.2f} - 0.10."
        )

    def test_conformal_quantile_increases_for_harder_cal(self):
        """A noisier calibration set should produce a larger quantile."""
        np.random.seed(0)
        x = np.random.randn(200, 1).astype(np.float32)
        y_easy = (2 * x.ravel() + np.random.randn(200) * 0.1).astype(np.float32)
        y_hard = (2 * x.ravel() + np.random.randn(200) * 1.0).astype(np.float32)

        # A trivial wrapper around predicting the mean to test ConformalWrapper directly
        class IdentityModel:
            def predict(self, xx):
                return 2 * xx.ravel()

        cw_easy = ConformalWrapper(IdentityModel())
        cw_easy.calibrate(x, y_easy, alpha=0.1)
        cw_hard = ConformalWrapper(IdentityModel())
        cw_hard.calibrate(x, y_hard, alpha=0.1)

        assert cw_hard.quantile > cw_easy.quantile, (
            f"Harder calibration set should produce larger quantile "
            f"(easy={cw_easy.quantile:.3f}, hard={cw_hard.quantile:.3f})"
        )


# ----- Symlog transform ------------------------------------------------------


class TestSymlogTransform:
    """Tests for symlog/symexp target transform."""

    def test_symlog_roundtrip(self):
        x = torch.tensor([-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0])
        roundtrip = symexp(symlog(x))
        assert torch.allclose(x, roundtrip, atol=1e-4), (
            f"symlog/symexp roundtrip failed: {x} -> {roundtrip}"
        )

    def test_symlog_compresses_large_values(self):
        x = torch.tensor([0.0, 1.0, 10.0, 100.0, 1000.0])
        s = symlog(x)
        # symlog(1000) = log(1001) ≈ 6.9, much less than 1000
        assert s[-1] < 10.0
        # Monotonically increasing
        assert torch.all(s[1:] > s[:-1])

    def test_symlog_trains_on_exponential_targets(self):
        """symlog should produce reasonable predictions on wide-range targets."""
        np.random.seed(42)
        x = np.random.uniform(-3, 3, 500).reshape(-1, 1).astype(np.float32)
        y = (np.exp(x.ravel()) + np.random.normal(0, 0.1, 500)).astype(np.float32)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        model = _make_probreg(n_epochs=50, target_transform="symlog")
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        assert not np.any(np.isnan(y_pred))
        # Sanity check — predictions in original scale should bear some correlation with truth
        r = float(np.corrcoef(y_test, y_pred)[0, 1])
        assert r > 0.5, f"symlog model predictions poorly correlate with targets (r={r:.3f})"


# ----- SHAP explainer --------------------------------------------------------


def _small_regression_data(n: int = 200, n_features: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, (n, n_features)).astype(np.float32)
    y = (np.sin(x[:, 0]) + 0.5 * x[:, 1] - x[:, 2]).astype(np.float32)
    return x, y


class TestSHAP:
    """SHAP explainer compatibility after tuple-output fixes."""

    def test_shap_flexible_nn(self):
        """FlexibleNN (5-tuple forward) must produce valid SHAP values via DeepExplainer."""
        x, y = _small_regression_data()
        model = FlexibleHiddenLayersNN(
            input_size=3, output_size=1, max_hidden_layers=3,
            n_predictor_layers=1, hidden_size=32, n_epochs=20,
            learning_rate=0.01, validation_fraction=0.2, random_seed=0,
            calculate_feature_importance=False,
            layer_selection_method=LayerSelectionMethod.GUMBEL_SOFTMAX,
        )
        model.fit(x, y)
        info = model.get_shap_explainer_info()
        assert info["explainer_type"] == ExplainerType.DEEP
        explainer = FeatureExplainer(model, x_background=x[:50], device=model.device, random_state=0)
        shap_vals = explainer.explain(x[:20])
        assert np.array(shap_vals).shape[0] == 20, "Expected one SHAP row per input sample"
        assert not np.any(np.isnan(np.array(shap_vals))), "SHAP values contain NaN"

    def test_shap_independent_weights_flexible_nn(self):
        """IndependentWeightsFlexibleNN (5-tuple forward) must work with DeepExplainer."""
        x, y = _small_regression_data()
        model = IndependentWeightsFlexibleNN(
            input_size=3, output_size=1, max_hidden_layers=3,
            n_predictor_layers=1, hidden_size=32, n_epochs=20,
            learning_rate=0.01, validation_fraction=0.2, random_seed=0,
            calculate_feature_importance=False,
            layer_selection_method=LayerSelectionMethod.GUMBEL_SOFTMAX,
        )
        model.fit(x, y)
        info = model.get_shap_explainer_info()
        assert info["explainer_type"] == ExplainerType.DEEP
        explainer = FeatureExplainer(model, x_background=x[:50], device=model.device, random_state=0)
        shap_vals = explainer.explain(x[:20])
        assert np.array(shap_vals).shape[0] == 20
        assert not np.any(np.isnan(np.array(shap_vals)))

    def test_shap_probreg_fixed_k_uses_deep_explainer(self):
        """ProbReg fixed-k must use DeepExplainer (wrapper returns prediction tensor)."""
        x, y = _small_regression_data()
        model = _make_probreg(input_size=3)
        model.fit(x, y)
        info = model.get_shap_explainer_info()
        assert info["explainer_type"] == ExplainerType.DEEP
        explainer = FeatureExplainer(model, x_background=x[:50], device=model.device, random_state=0)
        shap_vals = explainer.explain(x[:20])
        assert np.array(shap_vals).shape[0] == 20
        assert not np.any(np.isnan(np.array(shap_vals)))

    def test_shap_probreg_dynamic_k_falls_back_to_kernel(self):
        """Dynamic n_classes ProbReg must fall back to KernelExplainer (DeepLIFT incompatible)."""
        x, y = _small_regression_data()
        model = ProbabilisticRegressionModel(
            input_size=3, max_n_classes_for_probabilistic_path=4,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.SOFT_GATING,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            n_epochs=20, learning_rate=0.01, validation_fraction=0.2, random_seed=0,
            calculate_feature_importance=False,
        )
        model.fit(x, y)
        info = model.get_shap_explainer_info()
        assert info["explainer_type"] == ExplainerType.KERNEL, (
            "Dynamic-k ProbReg must use KernelExplainer to avoid DeepLIFT batch-dispatch crash"
        )
        explainer = FeatureExplainer(model, x_background=x[:30], device=model.device, random_state=0)
        shap_vals = explainer.explain(x[:10])
        assert np.array(shap_vals).shape[0] == 10
        assert not np.any(np.isnan(np.array(shap_vals)))


# ----- Symlog MC uncertainty -------------------------------------------------


class TestSymlogMCUncertainty:
    """Monte Carlo uncertainty propagation through symexp replaces linearized Jacobian."""

    def test_mc_uncertainty_near_zero_differs_from_jacobian(self):
        """Near μ_symlog=0 with large σ, MC std should exceed the linearized Jacobian estimate.

        Linearized: σ_orig ≈ exp(|μ|) · σ = exp(0) · σ = σ (underestimates when σ is large).
        MC: integrates over the full nonlinear curvature of symexp, giving a larger spread.
        """
        mean_symlog = np.zeros(100, dtype=np.float32)  # μ = 0, near zero crossing
        std_symlog = np.ones(100, dtype=np.float32)     # σ = 1 in symlog space (large)

        _, std_mc = ProbabilisticRegressionModel._symlog_mc_moments(mean_symlog, std_symlog, n_samples=2000, seed=42)

        jacobian_approx = np.exp(np.abs(mean_symlog)) * std_symlog  # = 1.0 everywhere
        # MC must be strictly larger: symexp is convex, so Jensen's inequality applies
        assert float(np.mean(std_mc)) > float(np.mean(jacobian_approx)), (
            f"MC std ({np.mean(std_mc):.4f}) should exceed linearized Jacobian ({np.mean(jacobian_approx):.4f}) near zero"
        )

    def test_mc_uncertainty_consistent_with_predict(self):
        """predict() and predict_uncertainty() for symlog must both be in original scale.

        For exp(x) targets the mean prediction and std should be positive (original scale,
        not symlog space). Also verifies no NaN and that std is strictly positive.
        """
        np.random.seed(0)
        x = np.random.uniform(-3, 3, 300).reshape(-1, 1).astype(np.float32)
        y = (np.exp(x.ravel()) + np.random.normal(0, 0.1, 300)).astype(np.float32)
        x_train, x_test, y_train, _ = train_test_split(x, y, test_size=0.3, random_state=0)

        model = _make_probreg(n_epochs=40, target_transform="symlog")
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        y_std = model.predict_uncertainty(x_test)

        assert not np.any(np.isnan(y_pred)), "predict() returned NaN"
        assert not np.any(np.isnan(y_std)), "predict_uncertainty() returned NaN"
        assert np.all(y_std > 0), "Uncertainty must be strictly positive"
        # Both outputs should be in original (non-symlog) scale: positive for exp(x) targets
        assert np.mean(y_pred) > 0, "Mean predictions should be positive for exp(x) targets"
