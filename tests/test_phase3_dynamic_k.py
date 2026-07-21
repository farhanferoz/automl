"""Phase 3 tests: Dynamic n_classes strategies for ProbabilisticRegression."""

import numpy as np
import pytest
import torch
from sklearn.model_selection import train_test_split

from automl_package.enums import (
    CapacitySelection,
    NClassesRegularization,
    NClassesSelectionMethod,
    ProbabilisticRegressionOptimizationStrategy,
    ProbRegLossType,
    RegressionStrategy,
    UncertaintyMethod,
)
from automl_package.models.common.distilled_router import DEFAULT_TOLERANCE, DistilledCapacityRouter, _cheapest_within_tolerance_labels
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel
from automl_package.utils.pytorch_utils import get_device
from automl_package.utils.transforms import symlog

# Calibrated against measured error-table means on TestFitRouterSymlogSpaceAlignment's fixture
# (400 heavy-tailed points, max_k=3, 150 epochs), fix present vs fix removed:
#   data seed 0: correct 1.057 | unit-mismatched 20.920
#   data seed 1: correct 0.959 | unit-mismatched 15.369
# The original 20.0 bound landed INSIDE the mismatched range and let seed 1 (the seed the test
# used) pass with the fix removed -- the bound must sit well below the smallest mismatched value.
_SYMLOG_ROUTER_NLL_SANITY_BOUND = 5.0  # ~5x above the correct value, ~3x below the smallest mismatched one
_BATCHED_TENSOR_NDIM = 2  # (N, regression_output_size) vs a bare (N,) tensor
_MSE_CONVERGENCE_THRESHOLD = 10.0  # "didn't explode" bar for these small toy fits, not a tuned target
_MIN_UNCERTAINTY_NOISE_CORRELATION = 0.2
_REGIME_SPLIT_X = 0.5  # _two_regime_data's boundary between its unimodal and bimodal halves


class TestBugs1to4DynamicStrategiesNoCrash:
    """All dynamic strategies should run without ValueError/TypeError."""

    @pytest.mark.parametrize("method", [
        NClassesSelectionMethod.GUMBEL_SOFTMAX,
        NClassesSelectionMethod.SOFT_GATING,
        NClassesSelectionMethod.STE,
        NClassesSelectionMethod.REINFORCE,
        NClassesSelectionMethod.NESTED,
    ])
    def test_dynamic_strategy_trains(self, heteroscedastic_data, method):
        """Each dynamic n_classes strategy should train without crash."""
        x, y, _, _ = heteroscedastic_data
        model = ProbabilisticRegressionModel(
            input_size=1, n_classes=3, max_n_classes_for_probabilistic_path=5,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=method,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            n_epochs=10, learning_rate=0.01, random_seed=42,
            calculate_feature_importance=False,
        )
        model.fit(x, y)
        y_pred = model.predict(x)
        assert y_pred.shape == (len(x),)
        assert not np.any(np.isnan(y_pred))


class TestBugN2SteGradientPath:
    """N2: STE strategy must propagate gradients to n_classes_predictor.

    Uses weighted-sum pattern (prob * output) with the STE trick from
    f.gumbel_softmax(hard=True) which returns `y_hard - y_soft.detach() + y_soft`.
    """

    def test_ste_gradients_reach_n_classes_predictor(self, heteroscedastic_data):
        """After a backward pass under STE, n_classes_predictor params must have non-zero grads."""
        x, y, _, _ = heteroscedastic_data
        model = ProbabilisticRegressionModel(
            input_size=1, n_classes=3, max_n_classes_for_probabilistic_path=5,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.STE,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            n_epochs=2, learning_rate=0.01, random_seed=42,
            calculate_feature_importance=False,
        )
        model.fit(x, y)

        x_t = torch.tensor(x[:16], dtype=torch.float32).to(model.device)
        y_t = torch.tensor(y[:16], dtype=torch.float32).to(model.device)
        model.model.train()
        for p in model.model.parameters():
            if p.grad is not None:
                p.grad.zero_()
        preds, _, _, _, _ = model.model(x_t)
        # preds is (N, regression_output_size) — use column 0 (mean) for loss.
        mean_pred = preds[:, 0] if preds.dim() == _BATCHED_TENSOR_NDIM else preds.ravel()
        loss = ((mean_pred - y_t) ** 2).mean()
        loss.backward()

        predictor = model.model.n_classes_predictor
        grad_norms = [
            p.grad.detach().abs().sum().item()
            for p in predictor.parameters()
            if p.grad is not None
        ]
        assert grad_norms, "n_classes_predictor has no grads at all — STE path is severed."
        assert max(grad_norms) > 0, (
            f"All n_classes_predictor gradients are zero under STE: {grad_norms}. "
            "Bug N2 regression — _hard_selection_logic must preserve gradient via weighted-sum."
        )


class TestBug4ClassifierLogits:
    """Verify _weighted_average_logic receives classifier logits, not raw features."""

    def test_weighted_average_uses_classifier_output(self):
        """Predictions should change when classifier weights change."""
        np.random.seed(42)
        x = np.random.randn(50, 1).astype(np.float32)
        y = np.random.randn(50).astype(np.float32)

        model = ProbabilisticRegressionModel(
            input_size=1, n_classes=3, max_n_classes_for_probabilistic_path=5,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.GUMBEL_SOFTMAX,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            n_epochs=3, random_seed=42,
            calculate_feature_importance=False,
        )
        model.fit(x, y)
        y_pred_1 = model.predict(x)

        # Perturb classifier weights
        for p in model.model.classifier_layers.parameters():
            p.data += 1.0
        y_pred_2 = model.predict(x)

        assert not np.allclose(y_pred_1, y_pred_2, atol=1e-3), (
            "Predictions unchanged after perturbing classifier — "
            "raw features may still be passed instead of logits (Bug 4)"
        )


class TestBug9MiddleClassParams:
    """Verify middle_class_dist_params stored correctly per k."""

    def test_middle_class_params_not_self_referential(self, simple_linear_data):
        """Each k should have its own independent params, not circular references."""
        x, y = simple_linear_data
        model = ProbabilisticRegressionModel(
            input_size=1, n_classes=3, max_n_classes_for_probabilistic_path=5,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.GUMBEL_SOFTMAX,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            n_epochs=3, random_seed=42,
            use_middle_class_nll_penalty=True,
            calculate_feature_importance=False,
        )
        model.fit(x, y)

        params = model.middle_class_dist_params_
        assert isinstance(params, dict), "middle_class_dist_params_ should be a dict"

        # Should have per-k entries (k=2,3,4,5)
        for k in range(2, 6):
            if k % 2 != 0:
                assert k in params, f"middle_class_dist_params_ missing key k={k}"
                v = params[k]
                assert v is not params, f"middle_class_dist_params_[{k}] is a circular reference to itself"
                assert isinstance(v, dict), f"middle_class_dist_params_[{k}] should be a dict, got {type(v)}"
                assert "mean" in v and "std" in v, f"middle_class_dist_params_[{k}] missing mean/std"

    def test_middle_class_params_differ_across_k(self, simple_linear_data):
        """Different k values should produce different middle class params."""
        x, y = simple_linear_data
        model = ProbabilisticRegressionModel(
            input_size=1, n_classes=3, max_n_classes_for_probabilistic_path=7,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.GUMBEL_SOFTMAX,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            n_epochs=3, random_seed=42,
            use_middle_class_nll_penalty=True,
            calculate_feature_importance=False,
        )
        model.fit(x, y)

        params = model.middle_class_dist_params_
        odd_k_params = {k: v for k, v in params.items() if isinstance(v, dict)}
        # At least two odd k values should have different means
        means = [v["mean"] for v in odd_k_params.values()]
        assert len(set(means)) > 1, f"All k values have identical middle class mean: {means}"


class TestDynamicKModelComparison:
    """Model-level tests: verify dynamic k achieves expected behavior on designed problems."""

    def _make_probreg(self, n_classes=3, max_n_classes=7, method=NClassesSelectionMethod.GUMBEL_SOFTMAX, **kwargs):
        defaults = dict(
            input_size=1,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=method,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            n_epochs=60, learning_rate=0.01, early_stopping_rounds=15,
            validation_fraction=0.2, random_seed=42,
            calculate_feature_importance=False,
        )
        defaults.update(kwargs)
        return ProbabilisticRegressionModel(
            n_classes=n_classes, max_n_classes_for_probabilistic_path=max_n_classes, **defaults,
        )

    def test_dynamic_k_mse_reasonable(self, heteroscedastic_data):
        """Dynamic-k ProbReg should achieve reasonable MSE (not exploded)."""
        x, y, _, _ = heteroscedastic_data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        model = self._make_probreg(n_classes=3, max_n_classes=7)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        mse = float(np.mean((y_test - y_pred) ** 2))

        assert mse < _MSE_CONVERGENCE_THRESHOLD, f"Dynamic-k MSE ({mse:.4f}) is too high — model didn't converge"

    def test_dynamic_k_nll_competitive_with_fixed_k(self, heteroscedastic_data):
        """Dynamic-k ProbReg NLL should be competitive with fixed-k=5."""
        x, y, _, _ = heteroscedastic_data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        fixed = ProbabilisticRegressionModel(
            input_size=1, n_classes=5,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.NONE,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            n_epochs=60, learning_rate=0.01, early_stopping_rounds=15,
            validation_fraction=0.2, random_seed=42,
            calculate_feature_importance=False,
        )
        fixed.fit(x_train, y_train)
        fixed_nll = _compute_nll(fixed, x_test, y_test)

        dynamic = self._make_probreg(n_classes=3, max_n_classes=7)
        dynamic.fit(x_train, y_train)
        dynamic_nll = _compute_nll(dynamic, x_test, y_test)

        # Dynamic has a harder optimization problem (joint k + regression), allow 2x slack
        assert dynamic_nll < fixed_nll * 2.0, (
            f"Dynamic-k NLL ({dynamic_nll:.4f}) is >2x worse than fixed-k=5 ({fixed_nll:.4f})"
        )

    def test_per_input_k_variation(self, heteroscedastic_data):
        """Dynamic k should not be constant across all inputs."""
        x, y, _, _ = heteroscedastic_data

        model = self._make_probreg(n_classes=3, max_n_classes=7, n_epochs=80)
        model.fit(x, y)

        x_tensor = torch.tensor(x, dtype=torch.float32).to(model.device)
        model.model.eval()
        with torch.no_grad():
            _, _, k_actual, _, _ = model.model(x_tensor)

        k_values = k_actual.cpu().numpy().ravel()
        unique_k = np.unique(k_values)

        assert len(unique_k) > 1, (
            f"Dynamic k is constant ({unique_k[0]}) across all inputs. "
            f"n_classes_predictor is not learning input-dependent k selection."
        )

    def test_dynamic_k_uncertainty_correlates_with_noise(self, heteroscedastic_data):
        """Dynamic-k ProbReg uncertainty should correlate with actual noise level."""
        x, y, _, noise_level = heteroscedastic_data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        _, noise_test = train_test_split(noise_level, test_size=0.3, random_state=42)

        model = self._make_probreg(n_classes=3, max_n_classes=7, n_epochs=80)
        model.fit(x_train, y_train)
        pred_std = model.predict_uncertainty(x_test)

        correlation = np.corrcoef(noise_test.ravel(), pred_std.ravel())[0, 1]
        assert correlation > _MIN_UNCERTAINTY_NOISE_CORRELATION, (
            f"Dynamic-k predicted uncertainty poorly correlates with actual noise "
            f"(r={correlation:.3f}). Uncertainty estimation may be broken."
        )


class TestELBOkSelection:
    """Tests for ELBO-based dynamic k selection."""

    def _make_probreg(self, n_classes=3, max_n_classes=7, method=NClassesSelectionMethod.GUMBEL_SOFTMAX, **kwargs):
        defaults = dict(
            input_size=1,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=method,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            n_epochs=60, learning_rate=0.01, early_stopping_rounds=15,
            validation_fraction=0.2, random_seed=42,
            calculate_feature_importance=False,
        )
        defaults.update(kwargs)
        return ProbabilisticRegressionModel(
            n_classes=n_classes, max_n_classes_for_probabilistic_path=max_n_classes, **defaults,
        )

    def test_elbo_k_converges(self, heteroscedastic_data):
        """ELBO-based k selection should train and converge to reasonable MSE."""
        x, y, _, _ = heteroscedastic_data

        model = self._make_probreg(n_classes=3, max_n_classes=7, n_classes_regularization="elbo", n_epochs=40)
        model.fit(x, y)
        y_pred = model.predict(x)
        mse = float(np.mean((y - y_pred) ** 2))

        assert mse < _MSE_CONVERGENCE_THRESHOLD, f"ELBO-k model MSE={mse:.4f} — did not converge"

    def test_elbo_raises_mode_selection_entropy_vs_unregularized(self, heteroscedastic_data):
        """Uniform-prior ELBO (default) pulls q(k|not-bypass) toward uniform, so the
        entropy of the per-sample mode-selection distribution over probabilistic
        modes should be >= unregularized on average. This replaces the old
        `prefers_lower_k` test, which encoded the buggy linspace-prior behavior
        (see probabilistic_regression.py ELBO block)."""
        import torch.nn.functional as F

        x, y, _, _ = heteroscedastic_data

        model_none = self._make_probreg(n_classes=3, max_n_classes=7, n_classes_regularization="none", n_epochs=80)
        model_none.fit(x, y)

        model_elbo = self._make_probreg(n_classes=3, max_n_classes=7, n_classes_regularization="elbo", n_epochs=80)
        model_elbo.fit(x, y)

        x_t = torch.tensor(x, dtype=torch.float32).to(model_none.device)
        model_none.model.eval()
        model_elbo.model.eval()

        def _mean_entropy_over_probabilistic_modes(m):
            with torch.no_grad():
                probs = F.softmax(m.model.n_classes_predictor(x_t), dim=-1)
                prob_modes = probs[:, :-1]
                prob_modes = prob_modes / prob_modes.sum(-1, keepdim=True).clamp_min(1e-8)
                ent = -(prob_modes * torch.log(prob_modes.clamp_min(1e-8))).sum(-1)
                return float(ent.mean().item())

        ent_none = _mean_entropy_over_probabilistic_modes(model_none)
        ent_elbo = _mean_entropy_over_probabilistic_modes(model_elbo)

        assert ent_elbo >= ent_none - 0.05, (
            f"ELBO (uniform prior) should not reduce entropy of q(k|not-bypass) "
            f"below unregularized (elbo={ent_elbo:.3f}, none={ent_none:.3f})."
        )

    def test_elbo_bypass_prior_pulls_toward_configured_value(self, heteroscedastic_data):
        """ELBO with bypass_prior_prob=0.9 should produce a higher mean bypass
        fraction than bypass_prior_prob=0.1 on the same data. Validates that the
        bypass Bernoulli KL is actually wired up and pulls q(bypass) toward p(bypass)."""
        import torch.nn.functional as F

        x, y, _, _ = heteroscedastic_data

        def _mean_bypass_fraction(model):
            x_t = torch.tensor(x, dtype=torch.float32).to(model.device)
            model.model.eval()
            with torch.no_grad():
                probs = F.softmax(model.model.n_classes_predictor(x_t), dim=-1)
                return float(probs[:, -1].mean().item())

        m_low = self._make_probreg(
            n_classes=3, max_n_classes=7, n_classes_regularization="elbo",
            bypass_prior_prob=0.1, n_epochs=60,
        )
        m_low.fit(x, y)

        m_high = self._make_probreg(
            n_classes=3, max_n_classes=7, n_classes_regularization="elbo",
            bypass_prior_prob=0.9, n_epochs=60,
        )
        m_high.fit(x, y)

        bp_low = _mean_bypass_fraction(m_low)
        bp_high = _mean_bypass_fraction(m_high)

        assert bp_high > bp_low, (
            f"bypass_prior_prob=0.9 should pull bypass fraction higher than 0.1 "
            f"(high={bp_high:.3f}, low={bp_low:.3f})."
        )

    def test_k_penalty_reduces_mean_k(self, heteroscedastic_data):
        """K penalty should reduce mean k vs no regularization."""
        x, y, _, _ = heteroscedastic_data

        model_none = self._make_probreg(n_classes=3, max_n_classes=7, n_classes_regularization="none", n_epochs=50)
        model_none.fit(x, y)

        model_pen = self._make_probreg(n_classes=3, max_n_classes=7, n_classes_regularization="k_penalty", k_penalty_weight=0.05, n_epochs=50)
        model_pen.fit(x, y)

        x_t = torch.tensor(x, dtype=torch.float32).to(model_none.device)
        model_none.model.eval()
        model_pen.model.eval()
        with torch.no_grad():
            _, _, k_none, _, _ = model_none.model(x_t)
            _, _, k_pen, _, _ = model_pen.model(x_t)

        assert k_pen.float().mean() <= k_none.float().mean(), (
            f"K penalty did not reduce mean k (pen={k_pen.float().mean():.2f}, none={k_none.float().mean():.2f})"
        )


def _compute_nll(model, x, y):
    """Helper: compute gaussian NLL from model predictions + uncertainty.

    Follows `model.capacity_selection` -- set at construction, not per-call (FP-3).
    """
    y_pred = model.predict(x)
    y_std = model.predict_uncertainty(x)
    log_var = 2 * np.log(np.clip(y_std, 1e-6, None))
    nll = 0.5 * np.mean(log_var + ((y - y_pred) ** 2) / np.exp(log_var))
    return float(nll)


def _two_regime_data(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic 2-regime problem for the F9 distilled-router verify bullet (K6 reproduction).

    x < 0.5: unimodal linear trend + small noise -- the bypass (k=1) wins here
    ([[project_kselection_bypass_confound]]: smooth unimodal data is a negative control for any
    k>=2 mixture). x >= 0.5: bimodal (y = trend +/- 1.5, separation >> noise) -- no fixed k=1
    bypass can represent this, k=2 wins. No single GLOBAL fixed k is best on both regimes, so a
    per-input router should beat every global fixed k.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    y = np.empty(n, dtype=np.float64)
    low = x < _REGIME_SPLIT_X
    high = ~low
    y[low] = 2.0 * x[low] + rng.normal(0.0, 0.1, int(low.sum()))
    sign = rng.choice([-1.0, 1.0], size=int(high.sum()))
    y[high] = 2.0 * x[high] + sign * 1.5 + rng.normal(0.0, 0.1, int(high.sum()))
    return x.reshape(-1, 1).astype(np.float32), y.astype(np.float32)


class TestNestedKTraining:
    """Task F9 item 1: the nested-k training mode (ported from `_capacity_ladder_nested.py`)."""

    def _fit_nested(self, x, y, max_k=5, n_epochs=15, **kwargs):
        defaults = {
            "input_size": 1, "n_classes": 3, "max_n_classes_for_probabilistic_path": max_k,
            "uncertainty_method": UncertaintyMethod.PROBABILISTIC,
            "n_classes_selection_method": NClassesSelectionMethod.NESTED,
            "regression_strategy": RegressionStrategy.SEPARATE_HEADS,
            "n_epochs": n_epochs, "learning_rate": 0.01, "random_seed": 42,
            "calculate_feature_importance": False,
        }
        defaults.update(kwargs)
        model = ProbabilisticRegressionModel(**defaults)
        model.fit(x, y)
        return model

    def test_nested_trains_without_crash(self, heteroscedastic_data):
        """Mirrors TestBugs1to4DynamicStrategiesNoCrash for the new NESTED strategy."""
        x, y, _, _ = heteroscedastic_data
        model = self._fit_nested(x, y)
        y_pred = model.predict(x)
        assert y_pred.shape == (len(x),)
        assert not np.any(np.isnan(y_pred))

    def test_nested_no_n_classes_predictor_built(self, heteroscedastic_data):
        """No k input to the network: n_classes_predictor must not be built for NESTED."""
        x, y, _, _ = heteroscedastic_data
        model = self._fit_nested(x, y, n_epochs=2)
        assert model.model.n_classes_predictor is None
        assert model.model.direct_regression_head is not None

    def test_nested_requires_probabilistic_uncertainty(self):
        """K4's scheme is a strictly-probabilistic Gaussian-NLL objective."""
        with pytest.raises(ValueError, match="NESTED requires uncertainty_method"):
            ProbabilisticRegressionModel(
                input_size=1, n_classes=3, max_n_classes_for_probabilistic_path=5,
                uncertainty_method=UncertaintyMethod.CONSTANT,
                n_classes_selection_method=NClassesSelectionMethod.NESTED,
                calculate_feature_importance=False,
            )

    def test_nested_bypass_rung_differs_from_mixture_rung(self, multimodal_data):
        """forward_at_k(k=1) (bypass) and forward_at_k(k>=2) (mixture) must be genuinely
        different sub-networks, not the same computation under two names."""
        x, y = multimodal_data
        model = self._fit_nested(x, y, max_k=4, n_epochs=20)
        x_t = torch.tensor(x[:16], dtype=torch.float32).to(model.device)
        with torch.no_grad():
            out_bypass = model.model.forward_at_k(x_t, 1)
            out_mixture = model.model.forward_at_k(x_t, 3)
        assert not torch.allclose(out_bypass, out_mixture, atol=1e-4)

    @pytest.mark.skipif(get_device().type == "cpu", reason="requires a non-CPU device (CUDA/XPU) to exercise cross-device placement in forward_at_k")
    def test_forward_at_k_runs_on_accelerator_device(self, multimodal_data):
        """Regression guard: `forward_at_k` (used directly here, and internally by
        `fit_router`/PER_INPUT-routed `predict()`) takes a raw tensor and does no device
        transfer of its own -- callers must place the input on `model.device` first. Every
        production call site already does this (see `_forward_routed`/`fit_router` in
        `probabilistic_regression.py`); this test would have caught the two now-fixed sibling
        tests above, which called `forward_at_k` with a CPU-default `torch.tensor(...)` while the
        model itself was on `xpu`."""
        x, y = multimodal_data
        model = self._fit_nested(x, y, max_k=3, n_epochs=5)
        assert model.device.type != "cpu"

        x_t = torch.tensor(x[:8], dtype=torch.float32).to(model.device)
        with torch.no_grad():
            for k in range(1, 4):
                out = model.model.forward_at_k(x_t, k)
                assert out.device.type == model.device.type


class TestNestedKDistilledRouting:
    """Task F9 items 2 & 4: DistilledCapacityRouter-routed inference + the arbiter diagnostic."""

    def _fit_nested(self, x, y, max_k=3, n_epochs=250, seed=0, capacity_selection=CapacitySelection.FIXED):
        model = ProbabilisticRegressionModel(
            input_size=1, n_classes=3, max_n_classes_for_probabilistic_path=max_k,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.NESTED,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            optimization_strategy=ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY,
            n_epochs=n_epochs, learning_rate=0.01, random_seed=seed,
            calculate_feature_importance=False,
            capacity_selection=capacity_selection,
        )
        model.fit(x, y)
        return model

    def test_fit_router_requires_nested_strategy(self, heteroscedastic_data):
        x, y, _, _ = heteroscedastic_data
        model = ProbabilisticRegressionModel(
            input_size=1, n_classes=3, max_n_classes_for_probabilistic_path=5,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.GUMBEL_SOFTMAX,
            n_epochs=5, random_seed=42, calculate_feature_importance=False,
        )
        model.fit(x, y)
        with pytest.raises(RuntimeError, match="NESTED"):
            model.fit_router(x, y)

    def test_per_input_routes_with_no_caller_flag(self, heteroscedastic_data):
        """FP-3 test 1: a router-fitted model constructed with `CapacitySelection.PER_INPUT`
        routes on a plain `predict(x)` call, with no caller flag."""
        x, y, _, _ = heteroscedastic_data
        model = self._fit_nested(x, y, max_k=4, n_epochs=15, capacity_selection=CapacitySelection.PER_INPUT)
        model.fit_router(x, y)

        y_pred = model.predict(x)
        assert y_pred.shape == (len(x),)
        assert not np.any(np.isnan(y_pred))

    def test_predict_routed_requires_fitted_router(self, heteroscedastic_data):
        x, y, _, _ = heteroscedastic_data
        model = self._fit_nested(x, y, max_k=4, n_epochs=15, capacity_selection=CapacitySelection.PER_INPUT)
        with pytest.raises(RuntimeError, match="No router fitted"):
            model.predict(x)

    def test_inference_mode_kwarg_rejected_at_predict(self, heteroscedastic_data):
        """FP-3.b: `inference_mode` is removed entirely -- `predict` has no `**kwargs`, so passing
        it raises `TypeError` for free once the parameter is deleted."""
        x, y, _, _ = heteroscedastic_data
        model = self._fit_nested(x, y, max_k=4, n_epochs=5)
        with pytest.raises(TypeError):
            model.predict(x, inference_mode="routed")

    def test_inference_mode_kwarg_rejected_at_construction(self):
        """FP-3.d: a removed selection kwarg passed to the CONSTRUCTOR must raise `TypeError`, not
        be silently swallowed into `self.params`/`setattr` (`base.py:45,52`, `base_pytorch.py:49-50`)."""
        with pytest.raises(TypeError):
            ProbabilisticRegressionModel(input_size=1, inference_mode="hard")

    def test_held_out_arbiter_advantage_requires_nested_strategy(self, heteroscedastic_data):
        x, y, _, _ = heteroscedastic_data
        model = ProbabilisticRegressionModel(
            input_size=1, n_classes=3, max_n_classes_for_probabilistic_path=5,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.GUMBEL_SOFTMAX,
            n_epochs=5, random_seed=42, calculate_feature_importance=False,
        )
        model.fit(x, y)
        with pytest.raises(RuntimeError, match="NESTED"):
            model.held_out_arbiter_advantage(x.ravel(), y)

    def test_held_out_arbiter_advantage_shape(self, heteroscedastic_data):
        x, y, _, _ = heteroscedastic_data
        model = self._fit_nested(x, y, max_k=4, n_epochs=15)
        arb = model.held_out_arbiter_advantage(x.ravel(), y, width=1.0)
        assert arb.shape == (len(x),)
        assert not np.all(np.isnan(arb))

    def test_distilled_router_matches_or_beats_best_global_fixed_k(self):
        """F9 verify bullet: reproduce the K6 result through the package API.

        A synthetic 2-regime case (`_two_regime_data`) where no single global fixed k is
        optimal everywhere: the distilled-k routed model's held-out NLL must be at least as
        good as the BEST achievable global fixed k (computed directly on the test set, which
        gives the fixed-k baseline test-set-optimal information the router never sees --
        a conservative, favorable-to-the-baseline comparison).
        """
        x_train, y_train = _two_regime_data(600, seed=0)
        x_val, y_val = _two_regime_data(300, seed=1)
        x_test, y_test = _two_regime_data(600, seed=2)

        max_k = 3
        model = self._fit_nested(x_train, y_train, max_k=max_k, n_epochs=250, seed=0, capacity_selection=CapacitySelection.PER_INPUT)
        model.fit_router(x_val, y_val)

        routed_nll = _compute_nll(model, x_test, y_test)

        global_nlls = []
        x_test_t = torch.tensor(x_test, dtype=torch.float32).to(model.device)
        for k in range(1, max_k + 1):
            with torch.no_grad():
                out = model.model.forward_at_k(x_test_t, k)
            mean = out[:, 0].cpu().numpy()
            log_var = out[:, 1].cpu().numpy()
            global_nlls.append(float(np.mean(0.5 * (log_var + (y_test - mean) ** 2 / np.exp(log_var)))))
        best_global_nll = min(global_nlls)

        assert routed_nll <= best_global_nll + 0.05, (
            f"routed NLL ({routed_nll:.4f}) worse than the best global fixed k NLL "
            f"({best_global_nll:.4f}) by more than tolerance; per-k global NLLs={global_nlls}"
        )


class TestFitRouterSymlogSpaceAlignment:
    """Regression for F9-fix-b (docs/plans/capacity_programme/flexnn-core.md): `fit()`
    symlog-transforms y internally when `target_transform="symlog"`, so `forward_at_k`'s
    outputs live in symlog space -- but `fit_router()` used to score the caller's raw-space
    `y_val` against those symlog-space outputs with no transform and no exception, silently
    producing a wrong per-capacity error table. Fix: `fit_router` now symlog-transforms
    `y_val` internally (same contract as `fit()`) when `target_transform="symlog"`.
    """

    def _fit_symlog_nested(self, x, y, max_k=3, n_epochs=150, seed=0):
        model = ProbabilisticRegressionModel(
            input_size=1, n_classes=3, max_n_classes_for_probabilistic_path=max_k,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.NESTED,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            optimization_strategy=ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY,
            target_transform="symlog",
            n_epochs=n_epochs, learning_rate=0.01, random_seed=seed,
            calculate_feature_importance=False,
        )
        model.fit(x, y)
        return model

    @staticmethod
    def _heavy_tailed_data(n, seed):
        """Targets spanning ~2 orders of magnitude (roughly -55..55) -- the regime
        `target_transform="symlog"` exists for, and large enough that a raw-space-y-vs-
        symlog-space-mean unit mismatch is unmissable in an assertion."""
        rng = np.random.default_rng(seed)
        x = rng.uniform(0.0, 1.0, n).astype(np.float32)
        sign = rng.choice([-1.0, 1.0], size=n)
        y = (sign * (np.exp(4.0 * x) - 1.0) + rng.normal(0.0, 0.5, n)).astype(np.float32)
        return x.reshape(-1, 1), y

    def test_routing_matches_between_raw_and_pretransformed_y_val(self, monkeypatch):
        """`fit_router(x_val, y_val)` with raw `y_val` (the documented, `fit()`-matching
        contract) must agree with calling it with `y_val` pre-transformed via `symlog` on a
        model whose `target_transform` is temporarily cleared -- both constructions end up
        scoring `y_val` against `forward_at_k`'s outputs in the same (symlog) space, so a
        correctly space-aligned `fit_router` must agree with itself regardless of which
        convention got it there.

        Asserted at THREE depths, cheapest-to-fool last. The final routing is a lossy view of
        the fix: `DistilledCapacityRouter.fit` trains a cross-entropy MLP on labels that are
        ~97% capacity-0 on this fixture, so it collapses to a constant router in BOTH arms and
        compares equal even when the tables underneath differ. Measured with the fix removed:
        the error tables differ by an order of magnitude (means 1.06 vs 20.92) and the labels
        differ on 3.0% of samples, yet `route_index` agreed exactly -- which is why this test
        passed against the unfixed code when it asserted on routing alone. Assert on the error
        table (what the fix actually changes) first; keep the routing check as the contract
        statement, not as the detector.
        """
        x, y = self._heavy_tailed_data(400, seed=0)
        model = self._fit_symlog_nested(x, y, max_k=3, n_epochs=150, seed=0)

        captured: list[np.ndarray] = []
        original_fit = DistilledCapacityRouter.fit

        def _spy_fit(router_self, eval_fn, x_val, y_val, capacity_grid, *args, **kwargs):
            captured.append(np.stack([np.asarray(eval_fn(x_val, capacity)) for capacity in capacity_grid], axis=1))
            return original_fit(router_self, eval_fn, x_val, y_val, capacity_grid, *args, **kwargs)

        monkeypatch.setattr(DistilledCapacityRouter, "fit", _spy_fit)

        router_raw = model.fit_router(x, y)
        routing_raw = router_raw.route_index(x)

        y_pretransformed = symlog(torch.tensor(y, dtype=torch.float32)).numpy()
        original_transform = model.target_transform
        model.target_transform = None
        try:
            router_pretransformed = model.fit_router(x, y_pretransformed)
        finally:
            model.target_transform = original_transform
        routing_pretransformed = router_pretransformed.route_index(x)

        table_raw, table_pretransformed = captured
        np.testing.assert_allclose(
            table_raw, table_pretransformed, rtol=1e-5, atol=1e-6,
            err_msg=(
                "fit_router's per-capacity error table must be identical whether y_val arrives raw "
                "(auto-transformed internally) or pre-transformed -- a difference means one arm scored "
                "raw-space y against forward_at_k's symlog-space outputs (F9-fix-b)."
            ),
        )
        np.testing.assert_array_equal(
            _cheapest_within_tolerance_labels(table_raw, DEFAULT_TOLERANCE),
            _cheapest_within_tolerance_labels(table_pretransformed, DEFAULT_TOLERANCE),
            err_msg="the cheapest-within-tolerance capacity labels the router trains on must match between the two arms",
        )
        np.testing.assert_array_equal(
            routing_raw, routing_pretransformed,
            err_msg=(
                "fit_router(raw y) must route identically to fit_router(pre-transformed y) run "
                "against a model with target_transform temporarily cleared -- both should compare "
                "y_val against forward_at_k's outputs in the same (symlog) space."
            ),
        )

    @pytest.mark.parametrize("data_seed", [0, 1])
    def test_error_table_finite_and_on_training_loss_scale(self, monkeypatch, data_seed):
        """The per-capacity NLL `fit_router`'s `eval_fn` scores against -- captured via a spy
        on `DistilledCapacityRouter.fit` since `fit()` discards the error table into labels
        and never retains it -- must be finite and on the scale of a converged per-sample
        Gaussian NLL, not the raw-y-vs-symlog-space-mean blowup F9-fix-b produced on this
        heavy-tailed data.

        Parametrized over both data seeds: the blowup magnitude is seed-dependent (20.92 on
        seed 0, 15.37 on seed 1), so a single seed plus a loose bound is how the original
        version of this test passed against the unfixed code. See
        `_SYMLOG_ROUTER_NLL_SANITY_BOUND` for the measured calibration.
        """
        x, y = self._heavy_tailed_data(400, seed=data_seed)
        model = self._fit_symlog_nested(x, y, max_k=3, n_epochs=150, seed=0)

        captured: dict[str, np.ndarray] = {}
        original_fit = DistilledCapacityRouter.fit

        def _spy_fit(router_self, eval_fn, x_val, y_val, capacity_grid, *args, **kwargs):
            captured["error_table"] = np.stack([np.asarray(eval_fn(x_val, capacity)) for capacity in capacity_grid], axis=1)
            return original_fit(router_self, eval_fn, x_val, y_val, capacity_grid, *args, **kwargs)

        monkeypatch.setattr(DistilledCapacityRouter, "fit", _spy_fit)
        model.fit_router(x, y)

        error_table = captured["error_table"]
        assert np.all(np.isfinite(error_table)), "fit_router's per-capacity error table must be finite"
        assert error_table.mean() < _SYMLOG_ROUTER_NLL_SANITY_BOUND, (
            f"fit_router's error table mean ({error_table.mean():.2f}) exceeds the sanity bound "
            f"({_SYMLOG_ROUTER_NLL_SANITY_BOUND}) for a converged per-sample Gaussian NLL -- looks "
            "like a raw-y-vs-symlog-space-predictions unit mismatch (F9-fix-b), not real fit quality"
        )


class TestArbiterAdvantageSymlogSpaceAlignment:
    """Regression for D2 (docs/plans/capacity_programme/probreg.md §3): `held_out_arbiter_advantage`
    forces `forward_at_k`, whose outputs live in symlog space when `target_transform="symlog"`
    (fit() transforms y_train/y_val before training, :526-529) -- but
    `_per_sample_log_likelihood_at_k` used to score the caller's raw-space `y` against those
    symlog-space outputs with no transform, the identical units mismatch `fit_router` had before
    the D1 fix (see `TestFitRouterSymlogSpaceAlignment` above). Fix:
    `_per_sample_log_likelihood_at_k` now symlog-transforms `y` internally when
    `target_transform="symlog"`, exactly as `fit_router` does.
    """

    def _fit_symlog_nested(self, x, y, max_k=3, n_epochs=150, seed=0):
        model = ProbabilisticRegressionModel(
            input_size=1, n_classes=3, max_n_classes_for_probabilistic_path=max_k,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.NESTED,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            optimization_strategy=ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY,
            target_transform="symlog",
            n_epochs=n_epochs, learning_rate=0.01, random_seed=seed,
            calculate_feature_importance=False,
        )
        model.fit(x, y)
        return model

    @staticmethod
    def _heavy_tailed_data(n, seed):
        """Same fixture as `TestFitRouterSymlogSpaceAlignment._heavy_tailed_data` -- targets
        spanning ~2 orders of magnitude, large enough that a raw-space-y-vs-symlog-space-mean
        unit mismatch is unmissable in an assertion."""
        rng = np.random.default_rng(seed)
        x = rng.uniform(0.0, 1.0, n).astype(np.float32)
        sign = rng.choice([-1.0, 1.0], size=n)
        y = (sign * (np.exp(4.0 * x) - 1.0) + rng.normal(0.0, 0.5, n)).astype(np.float32)
        return x.reshape(-1, 1), y

    def test_per_sample_log_likelihood_matches_between_raw_and_pretransformed_y(self):
        """`_per_sample_log_likelihood_at_k(x, y_raw, k)` on a `target_transform="symlog"` model
        must equal calling it with `y` pre-transformed via `symlog` on a model whose
        `target_transform` is temporarily cleared -- both constructions end up scoring y against
        `forward_at_k`'s outputs in the same (symlog) space. Asserted directly on the per-sample
        log-likelihood array -- the quantity the fix changes -- rather than a neighbour-averaged
        or routed downstream view, which is exactly how D1's first tests came out blind (see
        `TestFitRouterSymlogSpaceAlignment`'s docstring).
        """
        x, y = self._heavy_tailed_data(400, seed=0)
        model = self._fit_symlog_nested(x, y, max_k=3, n_epochs=150, seed=0)
        x_arr = np.asarray(x, dtype=np.float64)

        ll_raw = model._per_sample_log_likelihood_at_k(x_arr, y, 2)

        y_pretransformed = symlog(torch.tensor(y, dtype=torch.float32)).numpy()
        original_transform = model.target_transform
        model.target_transform = None
        try:
            ll_pretransformed = model._per_sample_log_likelihood_at_k(x_arr, y_pretransformed, 2)
        finally:
            model.target_transform = original_transform

        np.testing.assert_allclose(
            ll_raw, ll_pretransformed, rtol=1e-5, atol=1e-6,
            err_msg=(
                "_per_sample_log_likelihood_at_k must return the same per-sample likelihood "
                "whether y arrives raw (auto-transformed internally) or pre-transformed -- a "
                "difference means raw-space y was scored against forward_at_k's symlog-space "
                "outputs (D2)."
            ),
        )

    def test_held_out_arbiter_advantage_matches_between_raw_and_pretransformed_y(self):
        """Contract statement for `held_out_arbiter_advantage` itself, one level up from the
        per-sample likelihood the fix actually changes: the arbiter's advantage must be
        identical whether `y` arrives raw or pre-transformed. Kept as the contract check, not
        the detector -- the per-sample test above is what would catch a regression that this
        neighbour-averaged view might dilute.
        """
        x, y = self._heavy_tailed_data(400, seed=0)
        model = self._fit_symlog_nested(x, y, max_k=3, n_epochs=150, seed=0)

        advantage_raw = model.held_out_arbiter_advantage(x.ravel(), y, width=1.0)

        y_pretransformed = symlog(torch.tensor(y, dtype=torch.float32)).numpy()
        original_transform = model.target_transform
        model.target_transform = None
        try:
            advantage_pretransformed = model.held_out_arbiter_advantage(x.ravel(), y_pretransformed, width=1.0)
        finally:
            model.target_transform = original_transform

        np.testing.assert_allclose(
            advantage_raw, advantage_pretransformed, rtol=1e-5, atol=1e-6, equal_nan=True,
            err_msg=(
                "held_out_arbiter_advantage must return the same advantage whether y arrives raw "
                "or pre-transformed -- a difference means the underlying per-sample likelihood "
                "scored raw-space y against forward_at_k's symlog-space outputs (D2)."
            ),
        )


class TestCeStopGradDynamicKSentinelFilter:
    """Regression for the `KeyError: 1073741824` bug.

    The loss path filtered samples by `selected_k_values < self.n_classes_inf`.
    n_classes_inf defaults to inf, so the direct-regression sentinel (2**30)
    passed the filter and crashed precomputed_class_boundaries[2**30]. The bug
    only tripped when CE supervision was active (CE_STOP_GRAD/GRADIENT_STOP)
    *and* a dynamic-k strategy selected the bypass mode for at least one
    sample in a batch. Cell B (REGRESSION_ONLY) skipped the buggy block.
    """

    @pytest.mark.parametrize("dynamic", [
        NClassesSelectionMethod.SOFT_GATING,
        NClassesSelectionMethod.GUMBEL_SOFTMAX,
    ])
    @pytest.mark.parametrize("k_reg", [
        NClassesRegularization.NONE,
        NClassesRegularization.K_PENALTY,
        NClassesRegularization.ELBO,
    ])
    def test_ce_stop_grad_with_dynamic_k_trains(self, heteroscedastic_data, dynamic, k_reg):
        x, y, _, _ = heteroscedastic_data
        model = ProbabilisticRegressionModel(
            input_size=1, n_classes=3, max_n_classes_for_probabilistic_path=3,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            optimization_strategy=ProbabilisticRegressionOptimizationStrategy.CE_STOP_GRAD,
            prob_reg_loss_type=ProbRegLossType.GAUSSIAN_LTV,
            n_classes_selection_method=dynamic,
            n_classes_regularization=k_reg,
            use_anchored_heads=False, constrain_middle_class=True,
            n_epochs=5, learning_rate=0.01, random_seed=42,
            calculate_feature_importance=False,
        )
        model.fit(x, y)
        y_pred = model.predict(x)
        assert y_pred.shape == (len(x),)
        assert not np.any(np.isnan(y_pred))
