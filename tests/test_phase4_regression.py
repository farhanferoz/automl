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
from automl_package.models.catboost_model import CatBoostModel
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
from automl_package.models.pytorch_linear_regression import PyTorchLinearRegression
from automl_package.models.xgboost_model import XGBoostModel
from automl_package.utils.cv import TimeSeriesSplit
from automl_package.utils.data_handler import create_train_val_split
from automl_package.utils.losses import poisson_deviance
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


class TestSampleWeights:
    """`fit(sample_weight=...)`: the weights reach every training loss and held-out score of the classes that take them; the rest refuse."""

    @staticmethod
    def _data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(0)
        x = rng.uniform(size=(400, 1))
        y = np.where(np.arange(400) % 2, 10.0, 0.0)  # independent of x: the best constant is the weighted mean, 7.5 (unweighted 5)
        return x, y, np.where(np.arange(400) % 2, 3.0, 1.0)

    @pytest.mark.parametrize(
        "build",
        [
            lambda: PyTorchNeuralNetwork(input_size=1, n_epochs=60, learning_rate=0.01, early_stopping_rounds=20, random_seed=0),
            lambda: PyTorchLinearRegression(input_size=1, n_epochs=60, learning_rate=0.05, early_stopping_rounds=20, random_seed=0),
            lambda: XGBoostModel(early_stopping_rounds=10, random_seed=0),
            lambda: LightGBMModel(early_stopping_rounds=10, random_seed=0, verbose=-1),
            lambda: CatBoostModel(early_stopping_rounds=10, random_seed=0),
            NormalEquationLinearRegression,
        ],
        ids=["network", "torch-linear", "xgboost", "lightgbm", "catboost", "normal-equation"],
    )
    def test_weights_move_the_fit(self, build, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AUTOML_DEVICE", "cpu")
        x, y, w = self._data()
        model = build()
        model.refit_on_train_val, model.calculate_feature_importance = False, False
        model.fit(x[:300], y[:300], x_val=x[300:], y_val=y[300:], sample_weight=w[:300], sample_weight_val=w[300:])
        assert model.predict(x[300:]).mean() == pytest.approx(7.5, abs=0.4)
        if getattr(model, "convergence_", None) is not None:  # the networks: early stopping read the weighted held-out fit term of the restored epoch
            assert model.convergence_.best_val == pytest.approx(model._held_out_fit_score(x[300:], y[300:], sample_weight=w[300:]), rel=1e-5)

    def test_lstm_weighs_rows_over_their_observed_horizons(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AUTOML_DEVICE", "cpu")
        rng = np.random.default_rng(0)
        lstm = PyTorchLSTMForecaster(
            layout=SequenceLayout(2, 4, 1), output_size=2, window=4, hidden_size=8, embedding_dim=2, n_epochs=40, batch_size=64, learning_rate=0.02,
            early_stopping_rounds=40, refit_on_train_val=False, target_scaling="none", calculate_feature_importance=False, random_seed=0,
        )
        prediction, y = torch.zeros(3, 2), torch.tensor([[1.0, float("nan")], [2.0, 2.0], [3.0, 3.0]])
        weight = torch.tensor([4.0, 1.0, 0.0])
        assert lstm._fit_loss(prediction, y, weight=weight).item() == pytest.approx((4 * 1 + 1 * 4 + 1 * 4) / (4 + 1 + 1))  # NaN entry skipped
        panel = np.column_stack([rng.integers(0, 2, 400), rng.normal(size=(400, 4))]).astype(np.float32)
        y = np.repeat(np.where(np.arange(400) % 2, 10.0, 0.0)[:, None], 2, axis=1)
        w = np.where(np.arange(400) % 2, 3.0, 1.0)
        lstm.fit(panel[:300], y[:300], x_val=panel[300:], y_val=y[300:], sample_weight=w[:300], sample_weight_val=w[300:])
        assert lstm.predict(panel[300:]).mean() == pytest.approx(7.5, abs=0.5)
        assert lstm.convergence_.best_val == pytest.approx(lstm._held_out_fit_score(panel[300:], y[300:], sample_weight=w[300:]), rel=1e-5)  # stopping read the weights
        weight_val = torch.tensor(w[300:] / w[300:].mean(), dtype=torch.float32)
        expected = lstm._fit_loss(torch.tensor(lstm.predict(panel[300:])), torch.tensor(y[300:], dtype=torch.float32), weight=weight_val)
        assert lstm._held_out_fit_score(panel[300:], y[300:], sample_weight=w[300:]) == pytest.approx(expected.item(), rel=1e-5)

    def test_held_out_scores_and_library_stopping_sets_read_the_weights(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AUTOML_DEVICE", "cpu")
        rng = np.random.default_rng(1)
        x = rng.uniform(size=(400, 1))
        y = 5 * x[:, 0] + rng.normal(size=400)
        w = rng.uniform(0, 2, 400)
        xv, yv, wv = x[300:], y[300:], w[300:]
        boosted = {
            "xgboost": (XGBoostModel(early_stopping_rounds=5, random_seed=0), lambda m: m.model.best_score),
            "lightgbm": (LightGBMModel(early_stopping_rounds=5, random_seed=0, verbose=-1), lambda m: m.model.best_score_["valid_0"]["rmse"]),
            "catboost": (CatBoostModel(early_stopping_rounds=5, random_seed=0), lambda m: m.model.get_best_score()["validation"]["RMSE"]),
        }
        for name, (model, best) in boosted.items():
            model.calculate_feature_importance = False
            model._fit_single(x[:300], y[:300], xv, yv, sample_weight=w[:300], sample_weight_val=wv)
            stopped = {"xgboost": lambda m: m.model.predict(xv, iteration_range=(0, m.model.best_iteration + 1)),
                       "lightgbm": lambda m: m.model.predict(xv, num_iteration=m.model.best_iteration_), "catboost": lambda m: m.model.predict(xv)}[name](model)
            assert best(model) == pytest.approx(np.sqrt(np.average((stopped - yv) ** 2, weights=wv)), rel=1e-6), name  # stopping read the weighted error
        for model in (NormalEquationLinearRegression(calculate_feature_importance=False), PyTorchNeuralNetwork(input_size=1, n_epochs=3, calculate_feature_importance=False)):
            model._fit_single(x[:300], y[:300])
            per_row = model._held_out_fit_per_sample(xv, yv)
            assert model._held_out_fit_score(xv, yv, sample_weight=wv) == pytest.approx(np.average(per_row, weights=wv), rel=1e-6)
        poisson = LightGBMModel(loss=ForecastLoss.POISSON_DEVIANCE, random_seed=0, verbose=-1, calculate_feature_importance=False)
        poisson._fit_single(x[:300], np.log1p(np.abs(y[:300])))
        per_row = poisson_deviance(np.log1p(np.abs(yv)), poisson._predict_for_scoring(xv))
        assert poisson._held_out_fit_score(xv, np.log1p(np.abs(yv)), sample_weight=wv) == pytest.approx(np.average(per_row, weights=wv), rel=1e-6)

    def test_weights_follow_folds_and_search_trials(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setenv("AUTOML_DEVICE", "cpu")
        x, y, w = self._data()
        seen: list[tuple[int, int | None, int | None, int | None]] = []

        def record(cls: type) -> None:
            original = cls._fit_single

            def recording(self, x_train, y_train, x_val=None, y_val=None, forced_iterations=None, sample_weight=None, sample_weight_val=None):
                lengths = [None if part is None else len(part) for part in (x_train, sample_weight, x_val, sample_weight_val)]
                seen.append(tuple(lengths))
                return original(self, x_train, y_train, x_val, y_val, forced_iterations, sample_weight=sample_weight, sample_weight_val=sample_weight_val)

            monkeypatch.setattr(cls, "_fit_single", recording)

        record(NormalEquationLinearRegression)
        record(XGBoostModel)
        settings = {"optimize_hyperparameters": True, "n_trials": 2, "calculate_feature_importance": False, "random_seed": 0, "hpo_storage_dir": str(tmp_path)}
        NormalEquationLinearRegression(cv_folds=2, validation_fraction=None, **settings).fit(x, y, sample_weight=w)  # search over CV folds, refit
        NormalEquationLinearRegression(**settings).fit(x, y, sample_weight=w)  # search on a drawn split, refit
        NormalEquationLinearRegression(refit_on_train_val=False, **settings).fit(x, y, sample_weight=w)  # search on the explicit one-pass split
        selecting = XGBoostModel(cv_folds=2, validation_fraction=None, early_stopping_rounds=5, feature_selection_threshold=0.99, random_seed=0)
        selecting.fit(x, y, sample_weight=w)  # CV epochs, feature selection, refit
        assert len(seen) >= 13  # the folds, trials and refits of the four fits above
        assert all(n_train == n_weight and n_val == n_weight_val for n_train, n_weight, n_val, n_weight_val in seen)  # every call weighted
        scaled = [XGBoostModel(random_seed=0, calculate_feature_importance=False, refit_on_train_val=False) for _ in range(2)]
        for model, factor in zip(scaled, (1.0, 1000.0), strict=True):
            model.fit(x[:300], y[:300], x_val=x[300:], y_val=y[300:], sample_weight=factor * w[:300], sample_weight_val=w[300:])
        np.testing.assert_allclose(scaled[0].predict(x[300:]), scaled[1].predict(x[300:]))  # weights are relative
        unweighted = NormalEquationLinearRegression(**settings)
        assert unweighted._hpo_journal(x, y, None, None)["study_name"] != unweighted._hpo_journal(x, y, None, None, sample_weight=w)["study_name"]
        exact = NormalEquationLinearRegression(input_scaling="none", calculate_feature_importance=False)
        twice = np.concatenate([x, x[:50]]), np.concatenate([y + x[:, 0], (y + x[:, 0])[:50]])
        exact._fit_single(*twice)
        doubled = NormalEquationLinearRegression(input_scaling="none", calculate_feature_importance=False)
        doubled._fit_single(x, y + x[:, 0], sample_weight=np.where(np.arange(400) < 50, 2.0, 1.0))
        np.testing.assert_allclose(np.append(doubled.weights, doubled.bias), np.append(exact.weights, exact.bias), rtol=1e-10)  # weight 2 = the row twice

    @pytest.mark.parametrize(
        ("build", "weights", "remedy"),
        [
            (lambda: ProbabilisticRegressionModel(input_size=1), {}, "fit it without sample_weight"),
            (lambda: FlexibleHiddenLayersNN(input_size=1), {}, "fit it without sample_weight"),
            (lambda: LightGBMModel(uncertainty_method=UncertaintyMethod.PROBABILISTIC), {}, "uncertainty_method=CONSTANT"),
            (lambda: NormalEquationLinearRegression(uncertainty_method=UncertaintyMethod.BINNED_RESIDUAL_STD), {}, "uncertainty_method=CONSTANT"),
            (NormalEquationLinearRegression, {"sample_weight": -np.ones(400)}, "non-negative"),
            (NormalEquationLinearRegression, {"sample_weight": np.zeros(400)}, "sums to 0"),
            (NormalEquationLinearRegression, {"sample_weight": np.ones(3)}, "one weight per row"),
            (NormalEquationLinearRegression, {"x_val": np.zeros((5, 1)), "y_val": np.zeros(5)}, "exactly when an explicit x_val"),
        ],
        ids=["probreg", "flexnn", "lightgbm-nll", "binned-spread", "negative", "zero-total", "length", "validation-weights"],
    )
    def test_refusals_name_the_remedy(self, build, weights, remedy) -> None:
        x, y, w = self._data()
        with pytest.raises(ValueError, match=remedy):
            build().fit(x, y, **({"sample_weight": w} | weights))


class TestColumnRolesEveryModel:
    """`offset_feature` / `categorical_features` for the models that do not split natively: `fit` works on the inner problem (offset dropped and
    subtracted from y, categorical columns one-hot), the public methods take the raw columns and `predict` adds the offset back."""

    FEATURES = ("f0", "f1")  # the inner problem of the offset-only model
    N_ROWS, N_FIT = 300, 200

    @classmethod
    def _frame(cls) -> tuple[pd.DataFrame, np.ndarray]:
        """f0, f1, a category code (3 levels, one NaN) and a starting score; y = offset + f(x) + category effect."""
        rng = np.random.default_rng(0)
        frame = pd.DataFrame(
            {"f0": rng.normal(size=cls.N_ROWS), "f1": rng.normal(size=cls.N_ROWS), "cat": rng.integers(0, 3, cls.N_ROWS).astype(float), "off": 3.0 * rng.normal(size=cls.N_ROWS)}
        )
        frame.loc[0, "cat"] = np.nan
        return frame, (frame["off"] + 2.0 * frame["f0"] - frame["f1"] + frame["cat"].fillna(0.0) + 0.1 * rng.normal(size=cls.N_ROWS)).to_numpy()

    @staticmethod
    def _build(kind: str, **kwargs):
        if kind == "linear":
            return NormalEquationLinearRegression(validation_fraction=0.0, test_fraction=0.0, calculate_feature_importance=False, **kwargs)
        return PyTorchNeuralNetwork(
            input_size=2, hidden_layers=1, hidden_size=8, learning_rate=0.02, n_epochs=8, early_stopping_rounds=8, refit_on_train_val=False,
            random_seed=0, calculate_feature_importance=False, **kwargs,
        )

    @pytest.mark.parametrize("kind", ["linear", "network"])
    def test_offset_model_equals_the_residual_model_plus_the_offset(self, kind: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AUTOML_DEVICE", "cpu")
        frame, y = self._frame()
        fit, held = slice(0, self.N_FIT), slice(self.N_FIT, self.N_ROWS)
        offset_model, residual_model = self._build(kind, offset_feature="off"), self._build(kind)
        seen: list[tuple] = []  # what a training path sees of the installed state: nothing is installed until training is over
        original = offset_model._fit_single
        offset_model._fit_single = lambda *a, **kw: (seen.append((offset_model.column_roles_, offset_model.input_feature_names_)), original(*a, **kw))[1]
        raw = frame[[*self.FEATURES, "off"]]
        offset_model.fit(raw.iloc[fit], y[fit], x_val=raw.iloc[held], y_val=y[held])
        off = frame["off"].to_numpy()
        residual_model.fit(frame[list(self.FEATURES)].iloc[fit], y[fit] - off[fit], x_val=frame[list(self.FEATURES)].iloc[held], y_val=y[held] - off[held])
        assert seen
        assert all(state == (None, None) for state in seen)
        assert offset_model.column_roles_ is not None
        assert offset_model.input_feature_names_ == [*self.FEATURES, "off"]

        expected = residual_model.predict(frame[list(self.FEATURES)].iloc[held]) + off[held]
        np.testing.assert_allclose(offset_model.predict(raw.iloc[held]), expected, rtol=1e-12 if kind == "linear" else 1e-6, atol=0.0 if kind == "linear" else 1e-6)
        np.testing.assert_allclose(offset_model.predict(raw.iloc[held].to_numpy()), expected, rtol=1e-6, atol=1e-6)  # a numpy X is read in the raw column order
        # the held-out score of a fitted model, called with raw inputs and raw targets, is the residual model's score on the inner problem
        score = offset_model._held_out_fit_score(raw.iloc[held], y[held])
        assert score == pytest.approx(residual_model._held_out_fit_score(frame[list(self.FEATURES)].iloc[held], y[held] - off[held]), rel=1e-9)
        assert offset_model.target_scaling is residual_model.target_scaling
        if offset_model.target_scaler_ is not None:  # PER_OUTPUT standardises the residual
            np.testing.assert_allclose(offset_model.target_scaler_.mean_, [np.mean(y[fit] - off[fit])], rtol=1e-6)
            np.testing.assert_allclose(offset_model.target_scaler_.scale_, residual_model.target_scaler_.scale_)
        with pytest.raises(NotImplementedError, match="offset_feature"):
            offset_model.predict_proba(raw.iloc[held])

    def test_one_hot_levels_come_from_the_training_rows_and_unseen_or_missing_is_all_zeros(self) -> None:
        frame, y = self._frame()
        train = frame.iloc[: self.N_FIT].copy()
        train["cat"] = train["cat"].where(train["cat"] != 2.0, 1.0)  # the fit rows hold levels {0, 1} (+ one NaN)
        model = self._build("linear", categorical_features=["cat"], l2_lambda=1e-6, out_of_range_action="flag")
        model.fit(train[["f0", "cat", "f1"]], y[: self.N_FIT])
        roles = model.column_roles_
        assert roles.inner_names_ == ["f0", "cat=0", "cat=1", "f1"]  # replaced in place, sorted levels, NaN is no level
        probe = pd.DataFrame({"f0": [0.0, 0.0, 0.0], "cat": [1.0, np.nan, 7.0], "f1": [0.0, 0.0, 0.0]})
        inner, offset = roles.transform(probe)
        assert offset is None
        np.testing.assert_array_equal(inner[["cat=0", "cat=1"]].to_numpy(), [[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])  # NaN and the unseen code 7: all zeros
        model.predict(probe)
        assert model.training_envelope_.feature_names == ["f0", "cat", "f1"]  # the envelope is on the raw columns
        assert model.training_envelope_.categorical_columns == ["cat"]
        assert model.last_prediction_report_.n_rows_unseen_category == 1

    def test_clones_made_inside_fit_work_on_the_inner_problem_and_boosting_clones_keep_the_roles(self) -> None:
        generic = self._build("linear", offset_feature="off", categorical_features=["cat"])
        clone = generic._clone_for_trial()
        assert (clone.offset_feature, clone.categorical_features) == (None, None)
        assert (generic.offset_feature, generic.categorical_features) == ("off", ["cat"])
        native = LightGBMModel(offset_feature="off", categorical_features=["cat"])._clone_for_trial()
        assert (native.offset_feature, native.categorical_features) == ("off", ["cat"])

    @pytest.mark.parametrize(
        ("build", "data", "match"),
        [
            (lambda: PyTorchLSTMForecaster(layout=SequenceLayout(2, 2, 1), output_size=1, offset_feature="off"), None, "cannot take a per-row starting score"),
            (lambda: ProbabilisticRegressionModel(input_size=2, offset_feature="off"), None, "cannot take a per-row starting score"),
            (lambda: PyTorchLSTMForecaster(layout=SequenceLayout(2, 2, 1), output_size=1, categorical_features=["cat"]), None, "own input layout"),
            (lambda: NormalEquationLinearRegression(categorical_encoding="native"), None, "use categorical_encoding=ONE_HOT"),
            (lambda: LightGBMModel(categorical_encoding="one_hot"), None, "boosting takes NATIVE"),
            (lambda: PyTorchNeuralNetwork(input_size=2, task_type=TaskType.CLASSIFICATION, offset_feature="off"), None, "task_type=REGRESSION"),
            (lambda: NormalEquationLinearRegression(offset_feature="cat", categorical_features=["cat"]), None, "also listed in categorical_features"),
            (lambda: NormalEquationLinearRegression(offset_feature="off", loss_function="poisson", calculate_feature_importance=False), "frame", "Poisson"),
            (lambda: NormalEquationLinearRegression(offset_feature="off"), "frame", "calculate_feature_importance"),
            (
                lambda: NormalEquationLinearRegression(offset_feature="off", calculate_feature_importance=False, feature_selection_threshold=1.0),
                "frame",
                "feature_selection_threshold",
            ),
            (lambda: NormalEquationLinearRegression(offset_feature="off", calculate_feature_importance=False), "array", "give a DataFrame"),
        ],
        ids=[
            "lstm-offset", "probreg-offset", "lstm-categorical", "native-on-linear", "one-hot-on-boosting", "offset-classification", "offset-also-categorical", "offset-poisson",
            "importance", "feature-selection", "numpy-x",
        ],
    )
    def test_column_role_refusals_name_the_remedy(self, build, data: str | None, match: str) -> None:
        frame, y = self._frame()

        def construct_and_fit() -> None:  # a setting alone is refused at construction, the rest at the start of `fit`
            model = build()
            if data is not None:
                model.fit(frame if data == "frame" else frame.to_numpy(), y)

        with pytest.raises(ValueError, match=match):
            construct_and_fit()
