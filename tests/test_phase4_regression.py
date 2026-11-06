"""Phase 4 tests: performance regression, cross-model integration, and new features.

Tests use fixed seeds and assert MSE/NLL don't degrade below known baselines.
Update thresholds only when intentional changes are made.
"""

import time

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.model_selection import train_test_split

from automl_package.enums import (
    DataSplitStrategy,
    DepthRegularization,
    ExplainerType,
    ForecastLoss,
    InputScaling,
    LayerSelectionMethod,
    LRSchedule,
    MissingInputs,
    NClassesSelectionMethod,
    OptimizerType,
    RegressionStrategy,
    TargetScaling,
    TaskType,
    UncertaintyMethod,
)
from automl_package.explainers.feature_explainer import FeatureExplainer
from automl_package.models.classifier_regression import ClassifierRegressionModel
from automl_package.models.common.training import TrainingSchedule
from automl_package.models.conformal import ConformalWrapper
from automl_package.models.flexible_neural_network import FlexibleHiddenLayersNN
from automl_package.models.independent_weights_flexible_neural_network import IndependentWeightsFlexibleNN
from automl_package.models.lightgbm_model import LightGBMModel
from automl_package.models.lstm_model import PyTorchLSTMForecaster, SequenceLayout
from automl_package.models.neural_network import PyTorchNeuralNetwork
from automl_package.models.normal_equation_linear_regression import NormalEquationLinearRegression
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel
from automl_package.models.xgboost_model import XGBoostModel
from automl_package.utils.cv import TimeSeriesSplit
from automl_package.utils.data_handler import create_train_val_split
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
        max_acceptable_nll = 1.8
        assert nll < max_acceptable_nll, f"ProbReg NLL ({nll:.4f}) regressed past {max_acceptable_nll} (known: 1.38)"

    def test_prob_regression_heteroscedastic_mse(self, heteroscedastic_data):
        x, y, _, _ = heteroscedastic_data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        model = _make_probreg()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        mse = float(np.mean((y_test - y_pred) ** 2))
        # Phase 1 baseline: ProbReg_k5 MSE=1.54. Allow slack.
        max_acceptable_mse = 2.5
        assert mse < max_acceptable_mse, f"ProbReg MSE ({mse:.4f}) regressed past {max_acceptable_mse} (known: 1.54)"

    def test_flexible_nn_piecewise_mse(self, piecewise_data):
        """No assertion here touches depth selection -- just an absolute MSE ceiling on the basic
        training/prediction pipeline, so GUMBEL_SOFTMAX was incidental. Rewritten against NONE (a
        survivor under MASTER Decision 29); verified empirically (mse=0.302, well under the 2.0
        ceiling) before this edit.
        """
        x, y, _ = piecewise_data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        max_acceptable_mse = 2.0
        model = FlexibleHiddenLayersNN(
            input_size=1, output_size=1, max_hidden_layers=3, n_predictor_layers=0,
            hidden_size=32, n_epochs=100, learning_rate=0.01, early_stopping_rounds=15,
            validation_fraction=0.2, random_seed=42, calculate_feature_importance=False,
            layer_selection_method=LayerSelectionMethod.NONE,
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        mse = float(np.mean((y_test - y_pred) ** 2))
        assert mse < max_acceptable_mse, f"FlexibleNN piecewise MSE ({mse:.4f}) regressed past {max_acceptable_mse}"


# ----- Cross-model ranking ---------------------------------------------------


class TestCrossModelRanking:
    """Verify expected model ranking holds on designed problems."""

    def test_probreg_beats_constant_variance_nn_on_heteroscedastic(self, heteroscedastic_data):
        x, y, _, _ = heteroscedastic_data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        probreg = _make_probreg(optimize_hyperparameters=False, n_epochs=150, early_stopping_rounds=20)
        probreg.fit(x_train, y_train)
        probreg_nll = _compute_nll(probreg, x_test, y_test)

        nn = PyTorchNeuralNetwork(
            input_size=1, output_size=1, hidden_layers=2, hidden_size=32,
            learning_rate=0.01, n_epochs=100, early_stopping_rounds=15,
            validation_fraction=0.2, uncertainty_method=UncertaintyMethod.CONSTANT,
            random_seed=42, calculate_feature_importance=False, optimize_hyperparameters=False,
        )
        nn.fit(x_train, y_train)
        nn_nll = _compute_nll(nn, x_test, y_test)

        assert probreg_nll < nn_nll, (
            f"ProbReg NLL ({probreg_nll:.4f}) should beat constant-variance NN ({nn_nll:.4f}) "
            f"on heteroscedastic data."
        )

    def test_flexible_nn_competitive_with_shallow(self, piecewise_data):
        """No assertion here reads depth selection or the ELBO posterior -- only final MSE within
        a 1.2x slack of the shallow baseline. Rewritten against NONE (a survivor); verified
        empirically (flex_mse=0.302 vs shallow_mse*1.2=0.324) before this edit. GUMBEL_SOFTMAX +
        ELBO were incidental -- ELBO is dropped too since it is inert without a learned selector
        (docs/flexnn/package/plan.md FP-12: "makes depth_regularization inert for layer_selection_method=NONE").
        """
        x, y, _ = piecewise_data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        competitive_slack = 1.2

        shallow = PyTorchNeuralNetwork(
            input_size=1, output_size=1, hidden_layers=1, hidden_size=32,
            learning_rate=0.01, n_epochs=80, early_stopping_rounds=15,
            validation_fraction=0.2, uncertainty_method=UncertaintyMethod.CONSTANT,
            random_seed=42, calculate_feature_importance=False,
        )
        shallow.fit(x_train, y_train)
        shallow_mse = float(np.mean((y_test - shallow.predict(x_test)) ** 2))

        flex = FlexibleHiddenLayersNN(
            input_size=1, output_size=1, max_hidden_layers=3, n_predictor_layers=0,
            hidden_size=32, n_epochs=100, learning_rate=0.01, early_stopping_rounds=15,
            validation_fraction=0.2, random_seed=42, calculate_feature_importance=False,
            layer_selection_method=LayerSelectionMethod.NONE,
            depth_regularization=DepthRegularization.NONE,
        )
        flex.fit(x_train, y_train)
        flex_mse = float(np.mean((y_test - flex.predict(x_test)) ** 2))

        # Allow some slack — flex should be in the same ballpark or better
        assert flex_mse <= shallow_mse * competitive_slack, (
            f"FlexibleNN ({flex_mse:.4f}) should be competitive with shallow ({shallow_mse:.4f})."
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
        """`flexible_nn_gumbel`/`flexible_nn_elbo` parametrize GUMBEL_SOFTMAX (RETIRED, MASTER
        Decision 29) BY NAME -- this test's whole point is covering the pipeline under each named
        variant, so those two stay labelled comparison arms via the explicit opt-out rather than
        collapsing onto a survivor (which would silently stop testing what the variant names say
        they test).
        """
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
                allow_retired_capacity_selection=True,
            )
        elif model_factory == "flexible_nn_elbo":
            model = FlexibleHiddenLayersNN(
                input_size=1, output_size=1, max_hidden_layers=3, n_predictor_layers=1,
                hidden_size=16, n_epochs=10, random_seed=42,
                calculate_feature_importance=False,
                layer_selection_method=LayerSelectionMethod.GUMBEL_SOFTMAX,
                depth_regularization=DepthRegularization.ELBO,
                allow_retired_capacity_selection=True,
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
        min_correlation = 0.2
        assert r > min_correlation, f"β-NLL learned variance correlates poorly with noise (r={r:.3f})"


# ----- Hard inference --------------------------------------------------------


class TestHardInference:
    """Tests for inference-time hard selection optimization.

    Both tests below need a genuine soft/hard DISTINCTION -- under NONE (a survivor) `predict()`
    and `predict(hard_execution=True)` are identical by construction (verified empirically: soft
    and hard predictions match exactly under NONE), which would make either test vacuous. GUMBEL_SOFTMAX
    is RETIRED (MASTER Decision 29) but load-bearing here, so both stay labelled comparison arms.
    """

    def test_hard_inference_predictions_close_to_soft(self, piecewise_data):
        x, y, _ = piecewise_data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        model = FlexibleHiddenLayersNN(
            input_size=1, output_size=1, max_hidden_layers=3, n_predictor_layers=1,
            hidden_size=32, n_epochs=80, learning_rate=0.01, early_stopping_rounds=15,
            validation_fraction=0.2, random_seed=42, calculate_feature_importance=False,
            layer_selection_method=LayerSelectionMethod.GUMBEL_SOFTMAX,
            allow_retired_capacity_selection=True,
        )
        model.fit(x_train, y_train)

        y_soft = model.predict(x_test)
        y_hard = model.predict(x_test, hard_execution=True)

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
            allow_retired_capacity_selection=True,
        )
        model.fit(x_train, y_train)
        x_large = np.tile(x_test, (20, 1))

        t0 = time.perf_counter()
        for _ in range(5):
            y_hard = model.predict(x_large, hard_execution=True)
        t_hard = time.perf_counter() - t0

        assert y_hard.shape == (len(x_large),)
        max_acceptable_seconds = 30.0
        assert t_hard < max_acceptable_seconds, f"Hard inference took too long: {t_hard:.2f}s"


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
        compression_ceiling = 10.0
        assert s[-1] < compression_ceiling
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
        min_correlation = 0.5
        assert r > min_correlation, f"symlog model predictions poorly correlate with targets (r={r:.3f})"


# ----- SHAP explainer --------------------------------------------------------


def _small_regression_data(n: int = 200, n_features: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, (n, n_features)).astype(np.float32)
    y = (np.sin(x[:, 0]) + 0.5 * x[:, 1] - x[:, 2]).astype(np.float32)
    return x, y


class TestSHAP:
    """SHAP explainer compatibility after tuple-output fixes."""

    def test_shap_flexible_nn(self):
        """FlexibleNN (5-tuple forward) must produce valid SHAP values via DeepExplainer.

        The subject is the 5-tuple forward SIGNATURE, which NoneStrategy (a survivor) shares with
        GUMBEL_SOFTMAX -- verified empirically (DeepExplainer, shape (20, 3, 1), no NaN) before
        this edit. GUMBEL_SOFTMAX was incidental; rewritten against NONE.
        """
        n_explain = 20
        x, y = _small_regression_data()
        model = FlexibleHiddenLayersNN(
            input_size=3, output_size=1, max_hidden_layers=3,
            n_predictor_layers=0, hidden_size=32, n_epochs=20,
            learning_rate=0.01, validation_fraction=0.2, random_seed=0,
            calculate_feature_importance=False,
            layer_selection_method=LayerSelectionMethod.NONE,
        )
        model.fit(x, y)
        info = model.get_shap_explainer_info()
        assert info["explainer_type"] == ExplainerType.DEEP
        explainer = FeatureExplainer(model, x_background=x[:50], device=model.device, random_state=0)
        shap_vals = explainer.explain(x[:n_explain])
        assert np.array(shap_vals).shape[0] == n_explain, "Expected one SHAP row per input sample"
        assert not np.any(np.isnan(np.array(shap_vals))), "SHAP values contain NaN"

    def test_shap_independent_weights_flexible_nn(self):
        """IndependentWeightsFlexibleNN (5-tuple forward) must work with DeepExplainer."""
        n_explain = 20
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
        shap_vals = explainer.explain(x[:n_explain])
        assert np.array(shap_vals).shape[0] == n_explain
        assert not np.any(np.isnan(np.array(shap_vals)))

    def test_shap_probreg_fixed_k_uses_deep_explainer(self):
        """ProbReg fixed-k must use DeepExplainer (wrapper returns prediction tensor)."""
        n_explain = 20
        x, y = _small_regression_data()
        model = _make_probreg(input_size=3)
        model.fit(x, y)
        info = model.get_shap_explainer_info()
        assert info["explainer_type"] == ExplainerType.DEEP
        explainer = FeatureExplainer(model, x_background=x[:50], device=model.device, random_state=0)
        shap_vals = explainer.explain(x[:n_explain])
        assert np.array(shap_vals).shape[0] == n_explain
        assert not np.any(np.isnan(np.array(shap_vals)))

    def test_shap_probreg_dynamic_k_falls_back_to_kernel(self):
        """Dynamic n_classes ProbReg must fall back to KernelExplainer (DeepLIFT incompatible).

        `get_shap_explainer_info` gates on `n_classes_selection_method != NONE` generically
        (probabilistic_regression.py:1552) -- NESTED (a survivor) triggers the identical KERNEL
        fallback, verified empirically before this edit. SOFT_GATING was incidental.
        """
        n_explain = 10
        x, y = _small_regression_data()
        model = ProbabilisticRegressionModel(
            input_size=3, max_n_classes_for_probabilistic_path=4,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.NESTED, allow_unredesigned_nested=True,
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
        shap_vals = explainer.explain(x[:n_explain])
        assert np.array(shap_vals).shape[0] == n_explain
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


class TestPredictShapes:
    """N11: regression models must return predict(x) shape (N,) and predict_uncertainty(x) shape (N,)."""

    @pytest.mark.parametrize("uncertainty_method", [
        UncertaintyMethod.CONSTANT,
        UncertaintyMethod.PROBABILISTIC,
        UncertaintyMethod.MC_DROPOUT,
        UncertaintyMethod.BINNED_RESIDUAL_STD,
    ])
    def test_pytorch_nn_shapes(self, uncertainty_method):
        """PyTorchNeuralNetwork.predict / predict_uncertainty must return (N,)."""
        np.random.seed(0)
        x = np.random.randn(80, 3).astype(np.float32)
        y = np.random.randn(80).astype(np.float32)
        model = PyTorchNeuralNetwork(
            input_size=3, output_size=1,
            uncertainty_method=uncertainty_method,
            n_epochs=5, dropout_rate=0.1,
            calculate_feature_importance=False,
        )
        model.fit(x, y)
        y_pred = model.predict(x)
        y_std = model.predict_uncertainty(x)
        assert y_pred.shape == (80,), f"predict shape {y_pred.shape} for {uncertainty_method.value}"
        assert y_std.shape == (80,), f"predict_uncertainty shape {y_std.shape} for {uncertainty_method.value}"


class TestTargetScaling:
    """`BaseModel` standardises the targets once (TargetScaling.PER_OUTPUT); the outputs come back in target units."""

    @staticmethod
    def _data(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        x = rng.uniform(-1, 1, size=(300, 2)).astype(np.float32)
        y = (100.0 + 20.0 * (x[:, 0] - 0.5 * x[:, 1]) + rng.normal(0, 2.0, 300)).astype(np.float32)  # far from zero mean, sd well above 1
        return x, y

    def test_plain_network_trains_scaled_and_answers_in_target_units(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AUTOML_DEVICE", "cpu")
        x, y = self._data()
        model = PyTorchNeuralNetwork(
            input_size=2, output_size=1, hidden_layers=1, hidden_size=16, learning_rate=0.01, n_epochs=60, early_stopping_rounds=60,
            refit_on_train_val=False, random_seed=0, calculate_feature_importance=False,
        )
        assert model.target_scaling is TargetScaling.PER_OUTPUT  # the family default
        seen: list[np.ndarray] = []
        original = model._fit_single
        model._fit_single = lambda x_train, y_train, **kw: (seen.append(np.asarray(y_train)), original(x_train, y_train, **kw))[1]
        model.fit(x[:200], y[:200], x_val=x[200:], y_val=y[200:])
        assert abs(seen[0].mean()) < 1e-4 and seen[0].std() == pytest.approx(1.0, abs=1e-4)  # the network trained on standardised targets
        scale = model.target_scaler_.scale_[0]
        pred = model.predict(x[200:])
        assert np.sqrt(np.mean((pred - y[200:]) ** 2)) < 0.5 * y.std()  # forecasts are in target units (around 100), not in scaled units
        assert model._held_out_fit_score(x[200:], y[200:]) == pytest.approx(np.mean((pred - y[200:]) ** 2) / scale**2, rel=1e-3)  # the fit term in scaled space
        assert model.predict_uncertainty(x[200:])[0] == pytest.approx(np.std(y[:200] - model.predict(x[:200])), rel=1e-3)  # spread in target units

    @pytest.mark.parametrize(
        "build",
        [
            lambda: PyTorchNeuralNetwork(input_size=2, task_type=TaskType.CLASSIFICATION, target_scaling="per_output"),
            lambda: PyTorchLSTMForecaster(layout=SequenceLayout(2, 2, 1), output_size=1, loss="poisson_deviance"),  # PER_OUTPUT is this model's default
            lambda: LightGBMModel(loss=ForecastLoss.POISSON_DEVIANCE, target_scaling="per_output", calculate_feature_importance=False),
            lambda: XGBoostModel(offset_feature="feature_0", target_scaling="per_output", calculate_feature_importance=False),
            lambda: ProbabilisticRegressionModel(input_size=2, target_transform="symlog", target_scaling="per_output"),
        ],
        ids=["classifier", "lstm-poisson", "lightgbm-poisson", "xgboost-offset", "probreg-symlog"],
    )
    def test_refusals_name_the_remedy(self, build, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AUTOML_DEVICE", "cpu")
        x, y = self._data()
        with pytest.raises(ValueError, match="target_scaling=NONE"):
            build().fit(x, y)
        assert PyTorchNeuralNetwork(input_size=2, task_type=TaskType.CLASSIFICATION).target_scaling is TargetScaling.NONE  # unset: a classifier never scales

    def test_closed_form_model_scores_in_scaled_space_and_keeps_the_enum(self) -> None:
        """The base-class held-out score (models without a fit-term override) also reads scaled targets; a subclass copying its kwargs keeps the enum."""
        x, y = self._data()
        model = NormalEquationLinearRegression(target_scaling="per_output", uncertainty_method=UncertaintyMethod.CONSTANT)
        assert model.target_scaling is TargetScaling.PER_OUTPUT  # the class re-sets every kwarg after the base constructor
        model.fit(x[:200], y[:200], x_val=x[200:], y_val=y[200:])
        pred = model.predict(x[200:])
        assert np.sqrt(np.mean((pred - y[200:]) ** 2)) < 0.5 * y.std()
        assert model._held_out_fit_score(x[200:], y[200:]) == pytest.approx(np.mean((pred - y[200:]) ** 2) / model.target_scaler_.scale_[0] ** 2, rel=1e-4)
        assert model._held_out_fit_per_sample(x[200:], y[200:]).mean() == pytest.approx(model._held_out_fit_score(x[200:], y[200:]), rel=1e-6)  # the per-sample path (selection sweeps)

    def test_normal_equation_fits_its_intercept_on_uncentred_inputs(self) -> None:
        rng = np.random.default_rng(0)
        flag = rng.integers(0, 2, 300).astype(np.float64)  # a binary input: no scaling centres it
        y = 5.0 + 2.0 * flag + 0.1 * rng.normal(size=300)
        model = NormalEquationLinearRegression(input_scaling="none", refit_on_train_val=False, calculate_feature_importance=False)
        model.fit(flag[:200, None], y[:200], x_val=flag[200:, None], y_val=y[200:])
        np.testing.assert_allclose(model.predict(np.array([[0.0], [1.0]])), [5.0, 7.0], atol=0.05)  # the intercept counted once

    def test_class_that_does_not_map_its_outputs_back_refuses(self) -> None:
        with pytest.raises(ValueError, match="target_scaling=NONE"):
            ProbabilisticRegressionModel(input_size=2, target_scaling="per_output")
        assert ProbabilisticRegressionModel(input_size=2).target_scaling is TargetScaling.NONE  # its family default


class TestInputSettings:
    """`BaseModel` applies the inputs' missing-value rule and scaling once (`missing_inputs`, `input_scaling`); callers pass raw inputs."""

    @staticmethod
    def _data(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        big = rng.uniform(1000.0, 3000.0, 300)  # a column far from zero mean and unit sd
        flag = rng.integers(0, 2, 300).astype(np.float64)  # a binary column: left alone
        x = np.column_stack([big, flag])
        y = 0.01 * big + 5.0 * flag + rng.normal(0, 1.0, 300)
        return x, y

    def test_network_trains_on_standardised_inputs_and_takes_raw_ones(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AUTOML_DEVICE", "cpu")
        x, y = self._data()
        model = PyTorchNeuralNetwork(
            input_size=2, output_size=1, hidden_layers=1, hidden_size=16, learning_rate=0.01, n_epochs=40, early_stopping_rounds=40,
            refit_on_train_val=False, random_seed=0, calculate_feature_importance=False,
        )
        assert model.input_scaling is InputScaling.STANDARD  # the family defaults
        assert model.missing_inputs is MissingInputs.RAISE
        seen: list[np.ndarray] = []
        original = model._fit_single
        model._fit_single = lambda x_train, y_train, **kw: (seen.append(np.asarray(x_train)), original(x_train, y_train, **kw))[1]
        model.fit(x[:200], y[:200], x_val=x[200:], y_val=y[200:])
        assert abs(seen[0][:, 0].mean()) < 1e-6  # the network saw the big column standardised
        assert seen[0][:, 0].std() == pytest.approx(1.0, abs=1e-6)
        assert set(np.unique(seen[0][:, 1])) == {0.0, 1.0}  # and the binary column untouched
        pred = model.predict(x[200:])
        assert np.sqrt(np.mean((pred - y[200:]) ** 2)) < 0.5 * y.std()  # raw inputs were converted once: the forecasts are sensible
        outside = ((x[200:] < x[:200].min(axis=0)) | (x[200:] > x[:200].max(axis=0))).any(axis=1).sum()
        assert model.last_prediction_report_.n_rows_input_out_of_range == outside  # the envelope check read the raw inputs, as recorded
        frame = pd.DataFrame(x[200:], columns=model.feature_names)  # a DataFrame goes through the same, name-keyed conversion
        np.testing.assert_allclose(model.predict(frame), pred, rtol=1e-6)
        assert np.isfinite(model.predict_uncertainty(x[200:])).all()  # a wrapped public method also takes raw inputs

    def test_missing_inputs_rules(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AUTOML_DEVICE", "cpu")
        x_clean, y = self._data()
        x = x_clean.copy()
        x[::10, 0] = np.nan
        settings = {"input_size": 2, "output_size": 1, "n_epochs": 3, "refit_on_train_val": False, "random_seed": 0, "calculate_feature_importance": False}
        with pytest.raises(ValueError, match="missing_inputs=TRAINING_MEAN"):
            PyTorchNeuralNetwork(**settings).fit(x, y)
        filled = PyTorchNeuralNetwork(missing_inputs="training_mean", **settings)
        filled.fit(x[:200], y[:200], x_val=x[200:], y_val=y[200:])
        assert np.isfinite(filled.predict(x[200:])).all()
        clean = PyTorchNeuralNetwork(**settings)
        clean.fit(x_clean[:200], y[:200], x_val=x_clean[200:], y_val=y[200:])
        with pytest.raises(ValueError, match="missing_inputs=TRAINING_MEAN"):
            clean.predict(x[200:])  # a model fitted under RAISE refuses NaN at predict time too
        assert XGBoostModel(calculate_feature_importance=False).missing_inputs is MissingInputs.NATIVE  # trees take NaN natively

    @pytest.mark.parametrize(
        "build",
        [
            lambda: PyTorchLSTMForecaster(layout=SequenceLayout(2, 2, 1), output_size=1, input_scaling="standard"),
            lambda: ClassifierRegressionModel(input_scaling="standard"),
        ],
        ids=["lstm", "classreg"],
    )
    def test_input_scaling_refusals_name_the_remedy(self, build) -> None:
        with pytest.raises(ValueError, match="input_scaling=NONE"):
            build()


class TestTrainingDetails:
    """The time gap between fitted and held-out parts (`time_gap`) and the networks' clipping / learning-rate schedule (`TrainingSchedule`)."""

    def test_time_gap_leaves_out_timestamps_in_splits_and_folds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AUTOML_DEVICE", "cpu")
        dates = np.repeat(np.arange(50), 4)  # a panel: 50 dates x 4 series
        rows = np.zeros((200, 1))
        train, val, test = create_train_val_split(rows, 0.2, 0.2, DataSplitStrategy.TIME_ORDERED, dates, None, time_gap=3)
        assert (dates[train].max(), dates[val].min(), dates[val].max(), dates[test].min()) == (29, 33, 39, 43)  # 3 dates left out after each fitted part
        _, val_no_gap, _ = create_train_val_split(rows, 0.2, 0.2, DataSplitStrategy.TIME_ORDERED, dates, None)
        assert dates[val_no_gap].min() == 30  # no gap: as before the setting existed
        shuffled = np.random.default_rng(0).permutation(dates)
        for fold_train, fold_val in TimeSeriesSplit(2, gap=2, timestamps=shuffled).split(rows):
            assert shuffled[fold_val].min() == shuffled[fold_train].max() + 3  # rows ordered by time, the gap counted in dates
        assert [v[0] for _, v in TimeSeriesSplit(2, gap=1).split(rows)] == [81, 181]  # without timestamps the gap counts rows
        folds = PyTorchNeuralNetwork(input_size=1, split_strategy=DataSplitStrategy.TIME_ORDERED, cv_folds=2, validation_fraction=None, time_gap=2).get_kfolds(shuffled)
        assert all(shuffled[v].min() == shuffled[t].max() + 3 for t, v in folds.split(rows))  # the model's folds carry its gap and timestamps
        with pytest.raises(ValueError, match="lower time_gap"):
            create_train_val_split(rows, 0.2, 0.2, DataSplitStrategy.TIME_ORDERED, dates, None, time_gap=10)
        with pytest.raises(ValueError, match="split_strategy=TIME_ORDERED"):
            PyTorchNeuralNetwork(input_size=1, time_gap=1)
        gaps: list[int] = []
        recording = lambda *a, **kw: (gaps.append(kw["time_gap"]), create_train_val_split(*a, **kw))[1]  # noqa: E731
        monkeypatch.setattr("automl_package.models.base.create_train_val_split", recording)
        x = np.random.default_rng(1).normal(size=(200, 1)).astype(np.float32)
        model = PyTorchNeuralNetwork(
            input_size=1, n_epochs=2, split_strategy=DataSplitStrategy.TIME_ORDERED, time_gap=2, optimize_hyperparameters=True, n_trials=1,
            calculate_feature_importance=False, random_seed=0,
        )
        model.fit(x, x[:, 0], timestamps=dates)
        assert gaps == [2, 2]  # the outer split and the search's draw both leave the gap
        assert dates[model.val_indices].min() == dates[model.train_indices].max() + 3

    def test_schedule_and_clipping_in_both_training_loops(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AUTOML_DEVICE", "cpu")
        seen: list[tuple[float, float | None]] = []
        clips: list[float] = []
        end_epoch, clip = TrainingSchedule.end_epoch, TrainingSchedule.clip
        monkeypatch.setattr(TrainingSchedule, "end_epoch", lambda self, fit: (seen.append((self._optimizer.param_groups[0]["lr"], fit)), end_epoch(self, fit))[1])
        monkeypatch.setattr(TrainingSchedule, "clip", lambda self, model: (clips.append(self.grad_clip_norm), clip(self, model))[1])
        rng = np.random.default_rng(0)
        x = rng.normal(size=(120, 2)).astype(np.float32)
        y = x[:, 0] + 0.1 * rng.normal(size=120).astype(np.float32)
        base_rate, n_epochs, warmup = 0.01, 6, 2
        network = PyTorchNeuralNetwork(
            input_size=2, n_epochs=n_epochs, batch_size=40, early_stopping_rounds=n_epochs, refit_on_train_val=False, learning_rate=base_rate,
            lr_schedule="cosine", lr_warmup_epochs=warmup, grad_clip_norm=0.5, calculate_feature_importance=False, random_seed=0,
        )
        held_out: list[float] = []
        network.epoch_callback = lambda epoch, model, val_loss: held_out.append(val_loss)
        network.fit(x[:80], y[:80], x_val=x[80:], y_val=y[80:])
        expected = [base_rate * (e + 1) / warmup if e < warmup else base_rate * 0.5 * (1 + np.cos(np.pi * (e - warmup) / (n_epochs - warmup))) for e in range(n_epochs)]
        np.testing.assert_allclose([rate for rate, _ in seen], expected, rtol=1e-6)  # warmup, then cosine over the budget
        assert [fit for _, fit in seen] == held_out  # each epoch's held-out fit term reached the schedule
        assert clips == [0.5] * n_epochs * 2  # every step of the generic loop clipped (80 rows / 40 per batch)
        seen.clear(), clips.clear()
        layout = SequenceLayout(2, 4, 1)
        panel = np.column_stack([rng.integers(0, 2, 100), rng.normal(size=(100, 4))]).astype(np.float32)
        lstm = PyTorchLSTMForecaster(
            layout=layout, output_size=1, window=4, hidden_size=8, embedding_dim=2, n_epochs=3, batch_size=50, early_stopping_rounds=3,
            refit_on_train_val=False, lr_schedule="reduce_on_plateau", lr_schedule_patience=0, calculate_feature_importance=False, random_seed=0,
        )
        lstm.fit(panel[:80], panel[:80, -1:], x_val=panel[80:], y_val=panel[80:, -1:])
        assert [fit for _, fit in seen] == [loss for _, loss in lstm.convergence_.trajectory] and len(seen) == 3  # every epoch's held-out fit term, handed over
        assert clips == [1.0] * 3 * 2  # the LSTM keeps its default clipping, now through the shared helper

    def test_plateau_rule_and_refusals_name_the_remedy(self) -> None:
        layer = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(layer.parameters(), lr=1.0)
        schedule = TrainingSchedule(optimizer, grad_clip_norm=None, lr_schedule=LRSchedule.REDUCE_ON_PLATEAU, n_epochs=10, factor=0.5, patience=1, warmup_epochs=0)
        for _ in range(3):
            schedule.end_epoch(1.0)  # no new best after the first epoch
        assert optimizer.param_groups[0]["lr"] == 0.5  # more than `patience` epochs without a new best: one reduction
        with pytest.raises(ValueError, match="give it validation rows"):
            schedule.end_epoch(None)
        (layer(torch.ones(1, 1)) * 1e6).sum().backward()
        TrainingSchedule(optimizer, grad_clip_norm=1.0, lr_schedule=LRSchedule.NONE, n_epochs=10, factor=0.5, patience=1, warmup_epochs=0).clip(layer)
        assert torch.nn.utils.get_total_norm([p.grad for p in layer.parameters()]) <= 1.0 + 1e-6
        refusals = [
            (lambda: FlexibleHiddenLayersNN(input_size=1, grad_clip_norm=1.0), "loop of its own"),
            (lambda: PyTorchNeuralNetwork(input_size=1, lr_schedule="reduce_on_plateau", early_stopping_rounds=5), "refit_on_train_val=False"),
            (lambda: PyTorchNeuralNetwork(input_size=1, lr_schedule="cosine", optimizer_type=OptimizerType.HESSIAN_FREE), "optimizer_type=ADAM"),
        ]
        for build, remedy in refusals:
            with pytest.raises(ValueError, match=remedy):
                build()

    def test_search_journal_ignores_settings_that_are_off(self, tmp_path) -> None:
        x, y = np.zeros((10, 1)), np.zeros(10)

        def journal(**settings: object) -> str:
            return PyTorchNeuralNetwork(input_size=1, hpo_storage_dir=str(tmp_path), **settings)._hpo_journal(x, y, None, None)["study_name"]

        assert journal() == journal(lr_schedule_factor=0.5, time_gap=0, grad_clip_norm=None)  # a factor without the plateau schedule changes nothing
        assert journal() != journal(lr_schedule="cosine")
        assert journal(lr_schedule="reduce_on_plateau", early_stopping_rounds=5, refit_on_train_val=False) != journal(
            lr_schedule="reduce_on_plateau", early_stopping_rounds=5, refit_on_train_val=False, lr_schedule_factor=0.5
        )
