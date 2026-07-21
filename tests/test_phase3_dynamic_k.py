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
from automl_package.utils.capacity_selection import cheapest_within_tolerance
from automl_package.utils.pytorch_utils import get_device
from automl_package.utils.transforms import symlog

# Re-calibrated for flexnn-package.md FP-12's fixed-sigma MIXTURE scorer (sigma=1.0) against
# measured error-table means on TestFitRouterSymlogSpaceAlignment's fixture (400 heavy-tailed
# points, max_k=3, 150 epochs), fix present vs fix removed:
#   data seed 0: correct 3.849 | unit-mismatched 160.6
#   data seed 1: correct 3.438 | unit-mismatched (same order, not separately measured)
# (Pre-FP-12 values, at the retired learned-log_var Gaussian NLL, no longer apply: correct 1.057/
# 0.959, unit-mismatched 20.920/15.369 -- the scoring FUNCTION changed, not just its inputs.)
_SYMLOG_ROUTER_NLL_SANITY_BOUND = 10.0  # ~2.5x above the correct value, ~16x below the mismatched one
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
        """Each dynamic n_classes strategy should train without crash.

        GUMBEL_SOFTMAX/SOFT_GATING/STE/REINFORCE are RETIRED under MASTER Decision 29
        (flexnn-package.md FP-12) -- `allow_retired_capacity_selection=True` is the escape hatch
        that keeps this labelled-comparison-arm coverage running; NESTED needs no flag (a
        no-op there since it isn't a retired member).
        """
        x, y, _, _ = heteroscedastic_data
        model = ProbabilisticRegressionModel(
            input_size=1, n_classes=3, max_n_classes_for_probabilistic_path=5,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=method,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            n_epochs=10, learning_rate=0.01, random_seed=42,
            calculate_feature_importance=False,
            allow_retired_capacity_selection=True,
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
            allow_retired_capacity_selection=True,  # STE is RETIRED, MASTER Decision 29
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
            allow_retired_capacity_selection=True,  # GUMBEL_SOFTMAX is RETIRED, MASTER Decision 29
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
            allow_retired_capacity_selection=True,  # GUMBEL_SOFTMAX is RETIRED, MASTER Decision 29
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
            allow_retired_capacity_selection=True,  # GUMBEL_SOFTMAX is RETIRED, MASTER Decision 29
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
            allow_retired_capacity_selection=True,  # GUMBEL_SOFTMAX (+ n_classes_regularization
            # below, in TestELBOkSelection) is RETIRED, MASTER Decision 29 -- this class exercises
            # it deliberately as the labelled comparison arm.
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
            allow_retired_capacity_selection=True,  # GUMBEL_SOFTMAX (+ n_classes_regularization
            # below, in TestELBOkSelection) is RETIRED, MASTER Decision 29 -- this class exercises
            # it deliberately as the labelled comparison arm.
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
        """`n_classes_selection_method=NONE` -- any non-NESTED method demonstrates the guard;
        NONE needs no `allow_retired_capacity_selection` escape hatch (unlike GUMBEL_SOFTMAX,
        used here before MASTER Decision 29 retired it -- switched so this test does not couple
        an unrelated "requires NESTED" check to the retirement guard)."""
        x, y, _, _ = heteroscedastic_data
        model = ProbabilisticRegressionModel(
            input_size=1, n_classes=3, max_n_classes_for_probabilistic_path=5,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.NONE,
            n_epochs=5, random_seed=42, calculate_feature_importance=False,
        )
        model.fit(x, y)
        with pytest.raises(RuntimeError, match="NESTED"):
            model.fit_router(x, y, sigma=1.0)

    def test_per_input_routes_with_no_caller_flag(self, heteroscedastic_data):
        """FP-3 test 1: a router-fitted model constructed with `CapacitySelection.PER_INPUT`
        routes on a plain `predict(x)` call, with no caller flag."""
        x, y, _, _ = heteroscedastic_data
        model = self._fit_nested(x, y, max_k=4, n_epochs=15, capacity_selection=CapacitySelection.PER_INPUT)
        model.fit_router(x, y, sigma=1.0)

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
        """NONE, not GUMBEL_SOFTMAX (retired) -- see `test_fit_router_requires_nested_strategy`."""
        x, y, _, _ = heteroscedastic_data
        model = ProbabilisticRegressionModel(
            input_size=1, n_classes=3, max_n_classes_for_probabilistic_path=5,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.NONE,
            n_epochs=5, random_seed=42, calculate_feature_importance=False,
        )
        model.fit(x, y)
        with pytest.raises(RuntimeError, match="NESTED"):
            model.held_out_arbiter_advantage(x.ravel(), y, sigma=1.0)

    def test_held_out_arbiter_advantage_shape(self, heteroscedastic_data):
        x, y, _, _ = heteroscedastic_data
        model = self._fit_nested(x, y, max_k=4, n_epochs=15)
        arb = model.held_out_arbiter_advantage(x.ravel(), y, sigma=1.0, width=1.0)
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
        model.fit_router(x_val, y_val, sigma=0.1)  # _two_regime_data's construction noise std

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

        router_raw = model.fit_router(x, y, sigma=1.0)
        routing_raw = router_raw.route_index(x)

        y_pretransformed = symlog(torch.tensor(y, dtype=torch.float32)).numpy()
        original_transform = model.target_transform
        model.target_transform = None
        try:
            router_pretransformed = model.fit_router(x, y_pretransformed, sigma=1.0)
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
        model.fit_router(x, y, sigma=1.0)

        error_table = captured["error_table"]
        assert np.all(np.isfinite(error_table)), "fit_router's per-capacity error table must be finite"
        assert error_table.mean() < _SYMLOG_ROUTER_NLL_SANITY_BOUND, (
            f"fit_router's error table mean ({error_table.mean():.2f}) exceeds the sanity bound "
            f"({_SYMLOG_ROUTER_NLL_SANITY_BOUND}) for a converged per-sample fixed-sigma mixture NLL "
            "-- looks like a raw-y-vs-symlog-space-predictions unit mismatch (F9-fix-b), not real fit quality"
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

        ll_raw = model._per_sample_log_likelihood_at_k(x_arr, y, 2, sigma=1.0)

        y_pretransformed = symlog(torch.tensor(y, dtype=torch.float32)).numpy()
        original_transform = model.target_transform
        model.target_transform = None
        try:
            ll_pretransformed = model._per_sample_log_likelihood_at_k(x_arr, y_pretransformed, 2, sigma=1.0)
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

        advantage_raw = model.held_out_arbiter_advantage(x.ravel(), y, sigma=1.0, width=1.0)

        y_pretransformed = symlog(torch.tensor(y, dtype=torch.float32)).numpy()
        original_transform = model.target_transform
        model.target_transform = None
        try:
            advantage_pretransformed = model.held_out_arbiter_advantage(x.ravel(), y_pretransformed, sigma=1.0, width=1.0)
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
        """CE_STOP_GRAD, `dynamic` and `k_reg` are all RETIRED under MASTER Decision 29
        (flexnn-package.md FP-12); this regression guard deliberately exercises them together as
        the labelled comparison arm the escape hatch exists for."""
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
            allow_retired_capacity_selection=True,
        )
        model.fit(x, y)
        y_pred = model.predict(x)
        assert y_pred.shape == (len(x),)
        assert not np.any(np.isnan(y_pred))


# Calibrated against measured error-table means on TestFitGlobalSelectorSymlogSpaceAlignment's
# fixture (300 heavy-tailed points, max_k=3, 60 epochs), fix present vs fix removed: the correct
# (pre-transformed) mean log-likelihood is ~-2.0 nats; scoring the same rungs against raw
# (untransformed) y instead measures ~-19 to -23 nats -- an order of magnitude off, the same
# blowup shape D1/D2/F9-fix-b produced.
_SYMLOG_GLOBAL_SELECTOR_LL_SANITY_BOUND = -5.0  # a converged per-sample symlog-space LL; way above the mismatched range
_SYMLOG_GLOBAL_SELECTOR_MIN_GAP_NAT = 5.0  # correct-vs-raw-y curve gap floor; measured gap is ~17-21 nats


class TestFitGlobalSelector:
    """Task PA: `CapacitySelection.GLOBAL_CHEAP` (M1) -- `fit_global_selector`.

    `CapacitySelection.GLOBAL_CHEAP` itself is not yet shipped by `CapacitySelection`
    (capacity-programme FP-3.a: PA reports the exact member text to the root, which applies it to
    `enums.py` -- outside this task's write set). These tests exercise the mechanism directly
    (`fit_global_selector`/`_forward_global_k`), which is everything `fit()`'s dormant
    `CapacitySelection.GLOBAL_CHEAP` dispatch branch calls once the enum member lands.
    """

    def _fit_nested(self, x, y, max_k=4, n_epochs=20, seed=0, target_transform=None):
        model = ProbabilisticRegressionModel(
            input_size=1, n_classes=3, max_n_classes_for_probabilistic_path=max_k,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.NESTED,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            optimization_strategy=ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY,
            target_transform=target_transform,
            n_epochs=n_epochs, learning_rate=0.01, random_seed=seed,
            calculate_feature_importance=False,
        )
        model.fit(x, y)
        return model

    def test_requires_nested_strategy(self, heteroscedastic_data):
        """NONE, not GUMBEL_SOFTMAX (retired) -- see the identical substitution in
        `TestNestedKDistilledRouting.test_fit_router_requires_nested_strategy`."""
        x, y, _, _ = heteroscedastic_data
        model = ProbabilisticRegressionModel(
            input_size=1, n_classes=3, max_n_classes_for_probabilistic_path=4,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.NONE,
            n_epochs=5, random_seed=42, calculate_feature_importance=False,
        )
        model.fit(x, y)
        with pytest.raises(RuntimeError, match="NESTED"):
            model.fit_global_selector(x, y, sigma=1.0)

    def test_requires_fitted_model(self):
        model = ProbabilisticRegressionModel(
            input_size=1, n_classes=3, max_n_classes_for_probabilistic_path=4,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.NESTED,
            calculate_feature_importance=False,
        )
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.fit_global_selector(np.zeros((5, 1)), np.zeros(5), sigma=1.0)

    def test_selects_valid_k_and_matches_manual_curve(self, heteroscedastic_data):
        """PA's core requirement (D5): the selected k, and the curve it was read off, must be
        EXACTLY what `all_rung_log_likelihood` + `cheapest_within_tolerance` produce -- i.e. M1's
        selector is built on `all_rung_log_likelihood`
        (`n_classes_strategies.py:230`), never on `held_out_arbiter_advantage`."""
        x, y, _, _ = heteroscedastic_data
        model = self._fit_nested(x, y, max_k=4, n_epochs=20)
        x_val, y_val = x[:100], y[:100]
        sigma = 1.0

        selected_k = model.fit_global_selector(x_val, y_val, sigma=sigma, n_bootstrap=200)
        assert selected_k in range(1, 5)
        assert model.selected_k_ == selected_k

        x_tensor = torch.tensor(x_val, dtype=torch.float32).to(model.device)
        y_tensor = torch.tensor(y_val, dtype=torch.float32).to(model.device)
        model.model.eval()
        with torch.no_grad():
            ll_table = model.model.n_classes_strategy.all_rung_log_likelihood(x_tensor, y_tensor, sigma=sigma).cpu().numpy()
        expected_idx = cheapest_within_tolerance(-ll_table, n_boot=200, seed=model.random_seed or 0)
        assert selected_k == expected_idx + 1, "selected k must match the cheapest-within-tolerance index into all_rung_log_likelihood's curve"
        np.testing.assert_allclose(
            model.global_selector_curve_["mean_log_likelihood"], ll_table.mean(axis=0),
            rtol=1e-5, atol=1e-6,
        )

    def test_forward_global_k_forces_selected_k_everywhere(self, heteroscedastic_data):
        x, y, _, _ = heteroscedastic_data
        model = self._fit_nested(x, y, max_k=4, n_epochs=15)
        selected_k = model.fit_global_selector(x[:100], y[:100], sigma=1.0, n_bootstrap=100)

        mean, log_var = model._forward_global_k(x)
        x_tensor = torch.tensor(model._filter_predict_data(x), dtype=torch.float32).to(model.device)
        model.model.eval()
        with torch.no_grad():
            expected = model.model.forward_at_k(x_tensor, selected_k)
        np.testing.assert_allclose(mean, expected[:, 0].cpu().numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(log_var, expected[:, 1].cpu().numpy(), rtol=1e-5, atol=1e-6)

    def test_forward_global_k_requires_fitted_selector(self, heteroscedastic_data):
        x, y, _, _ = heteroscedastic_data
        model = self._fit_nested(x, y, max_k=4, n_epochs=5)
        with pytest.raises(RuntimeError, match="No global k selected"):
            model._forward_global_k(x)


class TestFitGlobalSelectorSymlogSpaceAlignment:
    """D2's units-mismatch pattern, applied to `fit_global_selector` (`all_rung_log_likelihood`
    does not transform `y_target` itself -- see its docstring, D6): the caller must
    symlog-transform `y_val` before scoring, or the per-rung curve is silently wrong by an order
    of magnitude, exactly like F9-fix-b/D2 before their fixes.
    """

    @staticmethod
    def _heavy_tailed_data(n, seed):
        rng = np.random.default_rng(seed)
        x = rng.uniform(0.0, 1.0, n).astype(np.float32)
        sign = rng.choice([-1.0, 1.0], size=n)
        y = (sign * (np.exp(4.0 * x) - 1.0) + rng.normal(0.0, 0.5, n)).astype(np.float32)
        return x.reshape(-1, 1), y

    def _fit_symlog_nested(self, x, y, max_k=3, n_epochs=60, seed=0):
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

    def test_curve_matches_manually_pretransformed_y(self):
        """`fit_global_selector(x, y)` with raw `y` (the documented contract) must produce
        EXACTLY the curve `all_rung_log_likelihood` gives when called directly with
        `symlog`-pretransformed `y` -- proving the internal transform runs."""
        x, y = self._heavy_tailed_data(300, seed=0)
        model = self._fit_symlog_nested(x, y)

        model.fit_global_selector(x, y, sigma=1.0, n_bootstrap=100)
        curve = np.asarray(model.global_selector_curve_["mean_log_likelihood"])

        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_pretransformed = symlog(torch.tensor(y, dtype=torch.float32))
        model.model.eval()
        with torch.no_grad():
            ll_correct = model.model.n_classes_strategy.all_rung_log_likelihood(x_tensor, y_pretransformed, sigma=1.0).numpy()

        np.testing.assert_allclose(curve, ll_correct.mean(axis=0), rtol=1e-4, atol=1e-5)
        assert curve.mean() > _SYMLOG_GLOBAL_SELECTOR_LL_SANITY_BOUND, (
            f"fit_global_selector's curve mean ({curve.mean():.2f}) is far below a converged "
            f"symlog-space per-sample log-likelihood -- looks like a raw-y-vs-symlog-space-"
            "predictions unit mismatch (D2's bug class), not real fit quality"
        )

    def test_curve_differs_from_scoring_raw_y_directly(self):
        """The contrast case: scoring the SAME rungs against raw (untransformed) `y` directly
        through `all_rung_log_likelihood` -- what `fit_global_selector` would silently do if it
        forgot the transform -- gives a curve an order of magnitude worse, demonstrating the
        transform is load-bearing, not cosmetic."""
        x, y = self._heavy_tailed_data(300, seed=0)
        model = self._fit_symlog_nested(x, y)
        model.fit_global_selector(x, y, sigma=1.0, n_bootstrap=100)
        curve_correct = np.asarray(model.global_selector_curve_["mean_log_likelihood"])

        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_raw = torch.tensor(y, dtype=torch.float32)
        model.model.eval()
        with torch.no_grad():
            ll_wrong = model.model.n_classes_strategy.all_rung_log_likelihood(x_tensor, y_raw, sigma=1.0).numpy()

        assert curve_correct.mean() - ll_wrong.mean(axis=0).mean() > _SYMLOG_GLOBAL_SELECTOR_MIN_GAP_NAT, (
            "the correctly-transformed curve should be nats better than scoring raw y directly "
            "against symlog-space rungs -- if these are close, the internal transform silently "
            "stopped running"
        )


class TestFitSweepSelector:
    """Task PA: `CapacitySelection.GLOBAL_SWEEP` (M3) -- `fit_sweep_selector`.

    Like `TestFitGlobalSelector`, `CapacitySelection.GLOBAL_SWEEP` is not yet shipped (PA reports
    the exact enum text to the root); these tests exercise the mechanism directly.
    """

    @staticmethod
    def _toy_data(n, seed):
        rng = np.random.default_rng(seed)
        x = rng.uniform(-5, 5, n).astype(np.float32).reshape(-1, 1)
        y_true = np.sin(x).ravel() * 2 + 0.5 * x.ravel()
        noise_std = 0.1 + 0.4 * np.abs(x.ravel())
        y = (y_true + rng.normal(0, noise_std)).astype(np.float32)
        return x, y

    def _base_model(self, max_k=3, n_epochs=10, seed=0):
        return ProbabilisticRegressionModel(
            input_size=1, n_classes=3, max_n_classes_for_probabilistic_path=max_k,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.NONE,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            optimization_strategy=ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY,
            n_epochs=n_epochs, learning_rate=0.01, random_seed=seed,
            calculate_feature_importance=False,
        )

    def test_selects_valid_k_and_submodels_are_ordinary(self):
        """§1's M3 ruling: every per-k sub-model is trained ORDINARILY
        (`n_classes_selection_method=NONE`), never with k-dropout."""
        x, y = self._toy_data(200, seed=0)
        x_tr, y_tr = x[:150], y[:150]
        x_val, y_val = x[150:], y[150:]

        model = self._base_model(max_k=3, n_epochs=10)
        selected_k = model.fit_sweep_selector(x_tr, y_tr, x_val, y_val, sigma=1.0, n_bootstrap=100)

        assert selected_k in range(1, 4)
        assert model.selected_k_ == selected_k
        winner = model._sweep_submodel()
        assert winner.n_classes_selection_method == NClassesSelectionMethod.NONE
        assert winner.n_classes == selected_k

    def test_predict_delegates_to_winning_submodel(self):
        x, y = self._toy_data(200, seed=0)
        x_tr, y_tr = x[:150], y[:150]
        x_val, y_val = x[150:], y[150:]

        model = self._base_model(max_k=3, n_epochs=10)
        model.fit_sweep_selector(x_tr, y_tr, x_val, y_val, sigma=1.0, n_bootstrap=100)

        winner = model._sweep_submodel()
        np.testing.assert_array_equal(model._sweep_submodel().predict(x_val), winner.predict(x_val))
        np.testing.assert_array_equal(model._sweep_submodel().predict_uncertainty(x_val), winner.predict_uncertainty(x_val))

    def test_sweep_submodel_requires_fitted_selector(self):
        model = self._base_model(max_k=3, n_epochs=5)
        with pytest.raises(RuntimeError, match="No sweep selector fitted"):
            model._sweep_submodel()

    def test_selection_rule_uses_shared_cheapest_within_tolerance_primitive(self, monkeypatch):
        """PA's requirement: M3 uses the SAME selection primitive as M1
        (`automl_package.utils.capacity_selection.cheapest_within_tolerance`), not a re-derived
        rule -- verified by spying on the actual call `fit_sweep_selector` makes."""
        import automl_package.models.probabilistic_regression as pr_module

        x, y = self._toy_data(200, seed=1)
        x_tr, y_tr = x[:150], y[:150]
        x_val, y_val = x[150:], y[150:]

        captured: dict = {}
        original = pr_module.cheapest_within_tolerance

        def _spy(error_table, n_boot, seed):
            captured["error_table"] = error_table
            return original(error_table, n_boot=n_boot, seed=seed)

        monkeypatch.setattr(pr_module, "cheapest_within_tolerance", _spy)

        model = self._base_model(max_k=3, n_epochs=10, seed=1)
        selected_k = model.fit_sweep_selector(x_tr, y_tr, x_val, y_val, sigma=1.0, n_bootstrap=150)

        assert "error_table" in captured, "fit_sweep_selector must call the shared cheapest_within_tolerance primitive"
        error_table = captured["error_table"]
        assert error_table.shape == (len(y_val), 3)
        expected_idx = original(error_table, n_boot=150, seed=model.random_seed or 0)
        assert selected_k == expected_idx + 1


class TestSelectionFractionConfigurable:
    """FP-3.e: the selection-set fraction is a constructor parameter, not a baked-in constant."""

    def test_fraction_controls_selection_set_size(self):
        x, y = TestFitSweepSelector._toy_data(400, seed=0)

        model_small = ProbabilisticRegressionModel(
            input_size=1, n_classes=3, max_n_classes_for_probabilistic_path=3,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.NESTED,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            optimization_strategy=ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY,
            n_epochs=5, learning_rate=0.01, random_seed=0,
            calculate_feature_importance=False, selection_fraction=0.1,
        )
        model_large = ProbabilisticRegressionModel(
            input_size=1, n_classes=3, max_n_classes_for_probabilistic_path=3,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.NESTED,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            optimization_strategy=ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY,
            n_epochs=5, learning_rate=0.01, random_seed=0,
            calculate_feature_importance=False, selection_fraction=0.4,
        )
        model_small._fit_global_cheap(x, y, sigma=1.0)
        model_large._fit_global_cheap(x, y, sigma=1.0)

        n_small = model_small.global_selector_curve_["n_selection"]
        n_large = model_large.global_selector_curve_["n_selection"]
        assert n_small < n_large
        np.testing.assert_allclose(n_small / len(y), 0.1, atol=0.02)
        np.testing.assert_allclose(n_large / len(y), 0.4, atol=0.02)


class TestGlobalSelectionEndToEndThroughFit:
    """Capacity-programme PA + FP-10 SEAM: `fit()` must complete under the two global modes.

    Regression guard for an integration defect neither task could see alone (root, 2026-07-21).
    `PyTorchModelBase._fit_residual_std` (FP-10) computes the CONSTANT-uncertainty residual std by
    calling the caller-facing `self.predict()`. Under `CapacitySelection.GLOBAL_CHEAP` that hook
    runs at the END of training, BEFORE `fit_global_selector` has chosen a k -- so the public
    predict path correctly refuses, and `fit()` raised
    `RuntimeError: No global k selected`. PA could not catch it (the enum members did not exist
    while PA ran, so the branch was unreachable); FP-10 could not (the modes did not exist yet).

    ⚠️ **SCOPE CORRECTION (root, 2026-07-21) — the two end-to-end tests below do NOT discriminate.**
    They were run with `ProbabilisticRegressionModel._fit_residual_std` DELETED and both still
    PASSED, so they are coverage, not evidence. The reason: `PyTorchModelBase._fit_single` calls
    that hook ONLY under `uncertainty_method=CONSTANT`
    (`automl_package/models/base_pytorch.py:217`), while `GLOBAL_CHEAP` legally requires
    `NESTED`, which itself requires `PROBABILISTIC` -- so in a VALID configuration the hook never
    fires and the defect is unreachable.

    The defect's real (narrower) blast radius: `CONSTANT` is ProbReg's DEFAULT
    `uncertainty_method`, so asking for `GLOBAL_CHEAP` WITHOUT also setting `NESTED` -- a plausible
    caller mistake -- used to crash mid-`fit()` with a confusing `No global k selected` from the
    residual-std bookkeeping path, instead of the precondition error that names the actual mistake.
    `test_global_cheap_without_nested_raises_precondition_not_midfit_crash` below is the
    discriminating guard: it FAILS on the unfixed code (wrong exception message) and passes with the
    fix. The two end-to-end tests are retained for coverage of the happy path.
    """

    @staticmethod
    def _xy(n: int = 200):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(n, 3))
        return x, x[:, 0] * 2.0 + rng.normal(scale=0.3, size=n)

    def test_fit_completes_and_predicts_under_global_cheap(self):
        x, y = self._xy()
        model = ProbabilisticRegressionModel(
            capacity_selection=CapacitySelection.GLOBAL_CHEAP,
            n_classes_selection_method=NClassesSelectionMethod.NESTED,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes=5, epochs=10, validation_fraction=0.2, early_stopping_rounds=3,
            calculate_feature_importance=False,
        )
        model.fit(x, y, sigma=1.0)             # raised RuntimeError before the fix
        assert model.selected_k_ is not None, "fit() under GLOBAL_CHEAP must select a global k"
        preds = model.predict(x)              # no caller flag
        assert preds.shape == (len(x),)
        assert not np.isnan(preds).any()

    def test_fit_completes_and_predicts_under_global_sweep(self):
        x, y = self._xy()
        model = ProbabilisticRegressionModel(
            capacity_selection=CapacitySelection.GLOBAL_SWEEP,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes=5, epochs=10, validation_fraction=0.2, early_stopping_rounds=3,
            calculate_feature_importance=False,
        )
        model.fit(x, y, sigma=1.0)
        preds = model.predict(x)
        assert preds.shape == (len(x),)
        assert not np.isnan(preds).any()

    def test_global_cheap_without_nested_raises_precondition_not_midfit_crash(self):
        """GLOBAL_CHEAP + the DEFAULT CONSTANT heads must name the real mistake, not crash bookkeeping.

        Discriminating guard for the FP-10/PA seam (see class docstring). Without ProbReg's
        `_fit_residual_std` override, `PyTorchModelBase._fit_single`'s residual-std block calls the
        caller-facing `predict()` mid-fit, which the GLOBAL_CHEAP branch refuses because no k has
        been selected yet -- surfacing as `No global k selected` and pointing the caller at the
        wrong thing entirely. The actual mistake is the missing NESTED strategy.
        """
        x, y = self._xy(n=120)
        model = ProbabilisticRegressionModel(
            capacity_selection=CapacitySelection.GLOBAL_CHEAP,   # no NESTED, no PROBABILISTIC:
            n_classes=4, epochs=5, validation_fraction=0.2,      # uncertainty_method defaults to CONSTANT
            early_stopping_rounds=2, calculate_feature_importance=False,
        )
        with pytest.raises(RuntimeError, match="NESTED"):
            model.fit(x, y, sigma=1.0)


class TestDecision29Guard:
    """MASTER Decision 29's one-enforcement-point guard, and probreg.md P10's head-layout gate
    (merged into the same guard, flexnn-package.md FP-12). FlexNN depth's identical guard
    (`layer.py`'s `_RetiredDepthSelectionStrategy`) is tested in `test_fixed_sigma_scorer.py`.
    """

    @pytest.mark.parametrize("method", [
        NClassesSelectionMethod.SOFT_GATING,
        NClassesSelectionMethod.GUMBEL_SOFTMAX,
        NClassesSelectionMethod.STE,
        NClassesSelectionMethod.REINFORCE,
    ])
    def test_retired_selection_method_raises_at_construction(self, method):
        with pytest.raises(ValueError, match="RETIRED"):
            ProbabilisticRegressionModel(
                input_size=1, n_classes=3, max_n_classes_for_probabilistic_path=5,
                uncertainty_method=UncertaintyMethod.PROBABILISTIC,
                n_classes_selection_method=method,
                calculate_feature_importance=False,
            )

    @pytest.mark.parametrize("regularization", [NClassesRegularization.K_PENALTY, NClassesRegularization.ELBO])
    def test_retired_regularization_raises_at_construction(self, regularization):
        """Reachable even under NESTED (not just the retired selection methods): `NestedStrategy`
        sets a per-sample one-hot `mode_selection_probs` too, so NESTED + a regularizer is a LIVE
        combination the guard must block independently (flexnn-package.md FP-12's report)."""
        with pytest.raises(ValueError, match="RETIRED"):
            ProbabilisticRegressionModel(
                input_size=1, n_classes=3, max_n_classes_for_probabilistic_path=5,
                uncertainty_method=UncertaintyMethod.PROBABILISTIC,
                n_classes_selection_method=NClassesSelectionMethod.NESTED,
                n_classes_regularization=regularization,
                calculate_feature_importance=False,
            )

    @pytest.mark.parametrize("strategy", [
        ProbabilisticRegressionOptimizationStrategy.COMPOSITE_LOSS,
        ProbabilisticRegressionOptimizationStrategy.GRADIENT_STOP,
        ProbabilisticRegressionOptimizationStrategy.CE_STOP_GRAD,
    ])
    def test_retired_optimization_strategy_raises_at_construction(self, strategy):
        with pytest.raises(ValueError, match="RETIRED"):
            ProbabilisticRegressionModel(
                input_size=1, n_classes=3,
                uncertainty_method=UncertaintyMethod.PROBABILISTIC,
                optimization_strategy=strategy,
                calculate_feature_importance=False,
            )

    def test_escape_hatch_allows_retired_selection_method(self):
        model = ProbabilisticRegressionModel(
            input_size=1, n_classes=3, max_n_classes_for_probabilistic_path=5,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.GUMBEL_SOFTMAX,
            allow_retired_capacity_selection=True,
            calculate_feature_importance=False,
        )
        assert model.n_classes_selection_method == NClassesSelectionMethod.GUMBEL_SOFTMAX

    def test_escape_hatch_is_off_by_default(self):
        model = ProbabilisticRegressionModel(input_size=1, n_classes=3, calculate_feature_importance=False)
        assert model.allow_retired_capacity_selection is False

    def test_head_layout_guard_raises_at_construction_not_mid_fit(self):
        """P10's original verify (a): `n_classes_selection_method=NESTED` +
        `regression_strategy=SINGLE_HEAD_FINAL_OUTPUT` raises ValueError naming the layout."""
        with pytest.raises(ValueError, match="SINGLE_HEAD_FINAL_OUTPUT") as excinfo:
            ProbabilisticRegressionModel(
                input_size=3, n_classes=4,
                uncertainty_method=UncertaintyMethod.PROBABILISTIC,
                n_classes_selection_method=NClassesSelectionMethod.NESTED,
                regression_strategy=RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT,
            )
        assert "SINGLE_HEAD_FINAL_OUTPUT" in str(excinfo.value)

    def test_head_layout_guard_has_no_escape_hatch(self):
        """Unlike the three in-training-selection retirements, the head-layout gate is a
        structurally invalid configuration (no components exist), not a labelled comparison arm
        -- `allow_retired_capacity_selection=True` must NOT unblock it."""
        with pytest.raises(ValueError, match="SINGLE_HEAD_FINAL_OUTPUT"):
            ProbabilisticRegressionModel(
                input_size=3, n_classes=4,
                uncertainty_method=UncertaintyMethod.PROBABILISTIC,
                n_classes_selection_method=NClassesSelectionMethod.NESTED,
                regression_strategy=RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT,
                allow_retired_capacity_selection=True,
            )

    def test_sanctioned_layouts_still_construct_and_train_under_nested(self, heteroscedastic_data):
        """P10's original verify (b): the two sanctioned layouts still construct AND train
        unchanged under NESTED."""
        x, y, _, _ = heteroscedastic_data
        for strategy in (RegressionStrategy.SEPARATE_HEADS, RegressionStrategy.SINGLE_HEAD_N_OUTPUTS):
            model = ProbabilisticRegressionModel(
                input_size=1, n_classes=3, max_n_classes_for_probabilistic_path=3,
                uncertainty_method=UncertaintyMethod.PROBABILISTIC,
                n_classes_selection_method=NClassesSelectionMethod.NESTED,
                regression_strategy=strategy,
                n_epochs=3, random_seed=42, calculate_feature_importance=False,
            )
            model.fit(x, y)
            y_pred = model.predict(x)
            assert y_pred.shape == (len(x),)
            assert not np.any(np.isnan(y_pred))

    def test_search_space_excludes_retired_optimization_strategies_by_default(self):
        """P10's original verify (c), generalised to Decision 29's optimization_strategy trap:
        the search space must not offer a member the constructor will reject."""
        model = ProbabilisticRegressionModel(input_size=1, n_classes=3, calculate_feature_importance=False)
        choices = model.get_hyperparameter_search_space()["optimization_strategy"]["choices"]
        assert choices == [ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY.value]

    def test_search_space_includes_all_optimization_strategies_with_escape_hatch(self):
        model = ProbabilisticRegressionModel(
            input_size=1, n_classes=3, calculate_feature_importance=False,
            allow_retired_capacity_selection=True,
        )
        choices = set(model.get_hyperparameter_search_space()["optimization_strategy"]["choices"])
        assert choices == {s.value for s in ProbabilisticRegressionOptimizationStrategy}

    def test_search_space_excludes_single_head_final_output_under_nested(self):
        """P10's original verify (c): the search space must not offer the blocked head layout
        when NESTED is set."""
        model = ProbabilisticRegressionModel(
            input_size=1, n_classes=3, max_n_classes_for_probabilistic_path=5,
            n_classes_selection_method=NClassesSelectionMethod.NESTED,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            calculate_feature_importance=False,
        )
        choices = model.get_hyperparameter_search_space()["regression_strategy"]["choices"]
        assert RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT.value not in choices
        assert set(choices) == {RegressionStrategy.SEPARATE_HEADS.value, RegressionStrategy.SINGLE_HEAD_N_OUTPUTS.value}

    def test_search_space_includes_single_head_final_output_under_none(self):
        """The exclusion is NESTED-specific -- under NONE the layout is legal and offered."""
        model = ProbabilisticRegressionModel(input_size=1, n_classes=3, calculate_feature_importance=False)
        choices = model.get_hyperparameter_search_space()["regression_strategy"]["choices"]
        assert RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT.value in choices
