"""Phase 3 tests: Dynamic n_classes strategies for ProbabilisticRegression."""

import numpy as np
import pytest
import torch
from sklearn.model_selection import train_test_split

from automl_package.enums import (
    NClassesSelectionMethod,
    RegressionStrategy,
    UncertaintyMethod,
)
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel


class TestBugs1to4DynamicStrategiesNoCrash:
    """All dynamic strategies should run without ValueError/TypeError."""

    @pytest.mark.parametrize("method", [
        NClassesSelectionMethod.GUMBEL_SOFTMAX,
        NClassesSelectionMethod.SOFT_GATING,
        NClassesSelectionMethod.STE,
        NClassesSelectionMethod.REINFORCE,
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

        assert mse < 10.0, f"Dynamic-k MSE ({mse:.4f}) is too high — model didn't converge"

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
        assert correlation > 0.2, (
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

        assert mse < 10.0, f"ELBO-k model MSE={mse:.4f} — did not converge"

    def test_elbo_prefers_lower_k_than_unregularized(self, heteroscedastic_data):
        """ELBO should select lower mean k than unregularized dynamic k on the same data."""
        x, y, _, _ = heteroscedastic_data

        model_none = self._make_probreg(n_classes=3, max_n_classes=7, n_classes_regularization="none", n_epochs=80)
        model_none.fit(x, y)

        model_elbo = self._make_probreg(n_classes=3, max_n_classes=7, n_classes_regularization="elbo", n_epochs=80)
        model_elbo.fit(x, y)

        x_t = torch.tensor(x, dtype=torch.float32).to(model_none.device)
        model_none.model.eval()
        model_elbo.model.eval()
        with torch.no_grad():
            _, _, k_none, _, _ = model_none.model(x_t)
            _, _, k_elbo, _, _ = model_elbo.model(x_t)

        mean_k_none = k_none.float().mean().item()
        mean_k_elbo = k_elbo.float().mean().item()

        assert mean_k_elbo <= mean_k_none, (
            f"ELBO should prefer lower k (elbo={mean_k_elbo:.2f}) "
            f"than unregularized (none={mean_k_none:.2f})."
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
    """Helper: compute gaussian NLL from model predictions + uncertainty."""
    y_pred = model.predict(x)
    y_std = model.predict_uncertainty(x)
    log_var = 2 * np.log(np.clip(y_std, 1e-6, None))
    nll = 0.5 * np.mean(log_var + ((y - y_pred) ** 2) / np.exp(log_var))
    return float(nll)
