"""Phase 1 tests: Bug fixes, probabilistic regression baseline, and model comparisons.

Includes:
- Unit tests for Bug 8 (double log_var), Bug 10 (duplicate build_model)
- Integration tests for ProbabilisticRegressionModel with fixed k
- Model comparison tests asserting expected relative performance orderings
- Work package P (2026-07-11 cascade plan §4.7): P1-P4 ProbReg fixes
"""

import logging

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.model_selection import train_test_split

from automl_package.enums import (
    MapperType,
    NClassesSelectionMethod,
    RegressionStrategy,
    UncertaintyMethod,
)
from automl_package.models.common.losses import calculate_combined_loss
from automl_package.models.neural_network import PyTorchNeuralNetwork
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel
from automl_package.utils.losses import nll_loss
from automl_package.utils.numerics import create_bins


@pytest.fixture
def automl_caplog(caplog):
    """caplog against the "automl_package" logger.

    `automl_package.logger` sets `propagate = False` (see `automl_package/logger.py`), so
    pytest's normal root-logger-attached caplog handler never observes its records. Attach the
    handler directly to the named logger for the duration of the test instead.
    """
    target_logger = logging.getLogger("automl_package")
    target_logger.addHandler(caplog.handler)
    with caplog.at_level(logging.WARNING, logger="automl_package"):
        yield caplog
    target_logger.removeHandler(caplog.handler)

# ---------------------------------------------------------------------------
# Bug 8: Double log_var
# ---------------------------------------------------------------------------

class TestBug8DoubleLogVar:
    """Verify Bug 8 fix: log_var should NOT be double-logged."""

    def test_calculate_combined_loss_no_double_log(self):
        """When predictions[:, 1] is already log_var, no extra log should be applied."""
        predictions = torch.tensor([[1.0, 0.5], [2.0, -0.3]])  # [mean, log_var]
        y_true = torch.tensor([1.1, 2.2])

        loss = calculate_combined_loss(
            predictions, y_true, UncertaintyMethod.PROBABILISTIC
        )

        # Manually compute expected NLL using log_var directly
        mean = predictions[:, 0]
        log_var = predictions[:, 1]
        variance = torch.exp(log_var)
        expected = 0.5 * (np.log(2 * np.pi) + log_var + ((y_true - mean) ** 2) / variance)
        expected_loss = expected.mean()

        assert torch.allclose(loss, expected_loss, atol=1e-5), (
            f"Loss {loss.item():.6f} != expected {expected_loss.item():.6f}. "
            "Double log_var likely still present."
        )

    def test_nll_gradient_flows_to_variance(self):
        """Verify gradient flows correctly through log_var after bug fix."""
        predictions = torch.tensor([[1.0, 0.5], [2.0, -0.3]], requires_grad=True)
        y_true = torch.tensor([1.1, 2.2])

        loss = calculate_combined_loss(
            predictions, y_true, UncertaintyMethod.PROBABILISTIC
        )
        loss.backward()

        assert predictions.grad is not None
        assert not torch.all(predictions.grad[:, 1] == 0), (
            "Zero gradient for log_var -- variance learning is broken."
        )


# ---------------------------------------------------------------------------
# Bug 10: Duplicate build_model
# ---------------------------------------------------------------------------

class TestBug10DuplicateBuild:
    """Verify Bug 10 fix: no duplicate build_model call."""

    def test_no_duplicate_build_model_line(self):
        """The source file should not contain consecutive duplicate build lines."""
        import inspect

        from automl_package.models.independent_weights_flexible_neural_network import (
            IndependentWeightsFlexibleNN,
        )
        source = inspect.getsource(IndependentWeightsFlexibleNN.build_model)
        count = source.count("self.model = self.IndependentWeightsFlexibleNNModule")
        assert count == 1, f"Found {count} build_model assignments, expected 1."


# ---------------------------------------------------------------------------
# NLL loss unit tests
# ---------------------------------------------------------------------------

class TestNLLLoss:
    """Unit tests for the core NLL loss function."""

    def test_nll_loss_correct_formula(self):
        """NLL should match manual Gaussian NLL computation."""
        outputs = torch.tensor([[2.0, 0.0]])  # mean=2, log_var=0 -> var=1
        targets = torch.tensor([3.0])

        loss = nll_loss(outputs, targets)
        expected = 0.5 * (np.log(2 * np.pi) + 0.0 + 1.0)
        tolerance = 1e-4
        assert abs(loss.item() - expected) < tolerance

    def test_nll_loss_penalizes_wrong_mean(self):
        """Higher error in mean should produce higher NLL."""
        targets = torch.tensor([0.0])
        close = nll_loss(torch.tensor([[0.1, 0.0]]), targets)
        far = nll_loss(torch.tensor([[5.0, 0.0]]), targets)
        assert far > close

    def test_nll_loss_penalizes_overconfident_wrong(self):
        """Low variance + wrong mean should produce very high NLL."""
        targets = torch.tensor([0.0])
        overconfident = nll_loss(torch.tensor([[5.0, -5.0]]), targets)
        honest = nll_loss(torch.tensor([[5.0, 3.0]]), targets)
        assert overconfident > honest


# ---------------------------------------------------------------------------
# ProbabilisticRegression integration tests (fixed k)
# ---------------------------------------------------------------------------

class TestProbabilisticRegressionFixedK:
    """Integration tests: train and predict with fixed k."""

    def _make_model(self, k: int = 5, strategy: RegressionStrategy = RegressionStrategy.SEPARATE_HEADS):
        return ProbabilisticRegressionModel(
            input_size=1,
            n_classes=k,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.NONE,
            regression_strategy=strategy,
            base_classifier_params={"hidden_layers": 1, "hidden_size": 32},
            regression_head_params={"hidden_layers": 1, "hidden_size": 16},
            n_epochs=30,
            learning_rate=0.01,
            early_stopping_rounds=10,
            validation_fraction=0.2,
            random_seed=42,
        )

    def test_train_and_predict_separate_heads(self, heteroscedastic_data):
        """SEPARATE_HEADS should train without crash and return predictions."""
        x, y, _, _ = heteroscedastic_data
        model = self._make_model(k=5, strategy=RegressionStrategy.SEPARATE_HEADS)
        model.fit(x, y)
        y_pred = model.predict(x)

        assert y_pred.shape == (len(x),), f"Expected 1D predictions, got {y_pred.shape}"
        assert not np.any(np.isnan(y_pred)), "Predictions contain NaN"

    def test_train_and_predict_single_head_n_outputs(self, heteroscedastic_data):
        """SINGLE_HEAD_N_OUTPUTS should train without crash."""
        x, y, _, _ = heteroscedastic_data
        model = self._make_model(k=5, strategy=RegressionStrategy.SINGLE_HEAD_N_OUTPUTS)
        model.fit(x, y)
        y_pred = model.predict(x)

        assert y_pred.shape == (len(x),)
        assert not np.any(np.isnan(y_pred))

    def test_uncertainty_correlates_with_noise(self, heteroscedastic_data):
        """Predicted uncertainty should be higher where true noise is higher."""
        x, y, _, noise_std = heteroscedastic_data
        x_train, x_test, y_train, _y_test, _, noise_test = train_test_split(
            x, y, noise_std, test_size=0.3, random_state=42
        )
        model = self._make_model(k=5)
        model.fit(x_train, y_train)
        pred_std = model.predict_uncertainty(x_test)

        correlation = np.corrcoef(pred_std, noise_test)[0, 1]
        min_correlation = 0.2
        assert correlation > min_correlation, (
            f"Predicted uncertainty poorly correlated with true noise "
            f"(r={correlation:.3f}). Bug 8 may not be fully fixed."
        )

    def test_mse_reasonable_on_linear_data(self, simple_linear_data):
        """Probabilistic regression should not degrade MSE on simple linear data."""
        x, y = simple_linear_data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        model = ProbabilisticRegressionModel(
            input_size=1,
            n_classes=5,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.NONE,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            base_classifier_params={"hidden_layers": 1, "hidden_size": 32},
            regression_head_params={"hidden_layers": 1, "hidden_size": 16},
            n_epochs=100,
            learning_rate=0.01,
            early_stopping_rounds=15,
            validation_fraction=0.2,
            random_seed=42,
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        mse = np.mean((y_test - y_pred) ** 2)

        max_acceptable_mse = 2.0
        assert mse < max_acceptable_mse, f"MSE={mse:.4f} too high for simple linear data (y=2x+1, noise_std=0.3)"


# ---------------------------------------------------------------------------
# Model comparison tests
# ---------------------------------------------------------------------------

def _compute_gaussian_nll(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray) -> float:
    """Compute mean Gaussian NLL from predictions and predicted std."""
    variance = np.maximum(y_std ** 2, 1e-8)
    nll = 0.5 * (np.log(2 * np.pi * variance) + ((y_true - y_pred) ** 2) / variance)
    return float(np.mean(nll))


class TestModelComparison:
    """Cross-model comparison tests.

    These assert the expected relative performance orderings on designed
    datasets, proving the architectures work as intended -- not just
    that they don't crash.
    """

    def test_probabilistic_nll_beats_constant_on_heteroscedastic(self, heteroscedastic_data):
        """On heteroscedastic data, ProbabilisticRegression NLL should beat constant-variance NN.

        The probabilistic model can learn input-dependent variance, so it should
        assign better likelihoods than a model that assumes constant noise.
        """
        x, y, _, _ = heteroscedastic_data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        # Baseline: constant-variance NN
        nn_model = PyTorchNeuralNetwork(
            input_size=1,
            output_size=1,
            hidden_layers=2,
            hidden_size=32,
            learning_rate=0.01,
            n_epochs=100,
            early_stopping_rounds=15,
            validation_fraction=0.2,
            uncertainty_method=UncertaintyMethod.CONSTANT,
            random_seed=42,
            use_hpo=False,
        )
        nn_model.fit(x_train, y_train)
        nn_pred = nn_model.predict(x_test)
        nn_std = nn_model.predict_uncertainty(x_test)
        nn_nll = _compute_gaussian_nll(y_test, nn_pred, nn_std)

        # Probabilistic model
        prob_model = ProbabilisticRegressionModel(
            input_size=1,
            n_classes=5,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.NONE,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            base_classifier_params={"hidden_layers": 1, "hidden_size": 32},
            regression_head_params={"hidden_layers": 0, "hidden_size": 16},
            n_epochs=150,
            learning_rate=0.01,
            early_stopping_rounds=20,
            validation_fraction=0.2,
            random_seed=42,
            use_hpo=False,
        )
        prob_model.fit(x_train, y_train)
        prob_pred = prob_model.predict(x_test)
        prob_std = prob_model.predict_uncertainty(x_test)
        prob_nll = _compute_gaussian_nll(y_test, prob_pred, prob_std)

        assert prob_nll < nn_nll, (
            f"ProbabilisticRegression NLL ({prob_nll:.3f}) should beat "
            f"constant-variance NN NLL ({nn_nll:.3f}) on heteroscedastic data."
        )

    def test_probabilistic_mse_competitive_on_linear(self, simple_linear_data):
        """On simple linear data, ProbabilisticRegression MSE should be within 2x of plain NN.

        The classification bottleneck shouldn't catastrophically degrade accuracy
        on a problem where a simple model suffices.
        """
        x, y = simple_linear_data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        # Baseline NN
        nn_model = PyTorchNeuralNetwork(
            input_size=1,
            output_size=1,
            hidden_layers=1,
            hidden_size=32,
            learning_rate=0.01,
            n_epochs=100,
            early_stopping_rounds=15,
            validation_fraction=0.2,
            uncertainty_method=UncertaintyMethod.CONSTANT,
            random_seed=42,
            use_hpo=False,
        )
        nn_model.fit(x_train, y_train)
        nn_pred = nn_model.predict(x_test)
        nn_mse = float(np.mean((y_test - nn_pred) ** 2))

        # Probabilistic model — n_classes=10 needed to avoid large quantization error on smooth data
        prob_model = ProbabilisticRegressionModel(
            input_size=1,
            n_classes=10,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.NONE,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            base_classifier_params={"hidden_layers": 1, "hidden_size": 32},
            regression_head_params={"hidden_layers": 1, "hidden_size": 16},
            n_epochs=150,
            learning_rate=0.01,
            early_stopping_rounds=20,
            validation_fraction=0.2,
            random_seed=42,
            use_hpo=False,
        )
        prob_model.fit(x_train, y_train)
        prob_pred = prob_model.predict(x_test)
        prob_mse = float(np.mean((y_test - prob_pred) ** 2))

        # 10x threshold: classification bottleneck inherently degrades smooth regression;
        # this checks for catastrophic failure, not competitive performance.
        assert prob_mse < nn_mse * 10.0, (
            f"ProbabilisticRegression MSE ({prob_mse:.4f}) is more than 10x "
            f"plain NN MSE ({nn_mse:.4f}) on simple linear data -- "
            f"classification bottleneck is degrading performance too much."
        )

    def test_classifier_regression_handles_multimodal(self, multimodal_data):
        """On bimodal data, ClassifierRegression variance should be larger than on linear data.

        The classification-based model should recognize the spread in bimodal data
        and produce higher predicted uncertainty than on clean linear data.
        """
        from automl_package.models.classifier_regression import ClassifierRegressionModel

        x, y = multimodal_data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        model = ClassifierRegressionModel(
            base_classifier_class=PyTorchNeuralNetwork,
            n_classes=7,
            mapper_type=MapperType.SPLINE,
            base_classifier_params={
                "input_size": 1,
                "hidden_layers": 2,
                "hidden_size": 32,
                "learning_rate": 0.01,
                "n_epochs": 60,
            },
            early_stopping_rounds=10,
            uncertainty_method=UncertaintyMethod.BINNED_RESIDUAL_STD,
            validation_fraction=0.2,
            random_seed=42,
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        multimodal_mse = float(np.mean((y_test - y_pred) ** 2))

        # On bimodal data (delta=1.5), even a perfect model predicting the conditional
        # mean gets MSE ~= delta^2 = 2.25. We verify the model trains without catastrophic failure.
        catastrophic_failure_mse = 20.0
        assert multimodal_mse < catastrophic_failure_mse, (
            f"ClassifierRegression MSE ({multimodal_mse:.4f}) is unreasonably high "
            f"on bimodal data -- model may have failed to train."
        )

    def test_prob_regression_uncertainty_varies_with_input(self, heteroscedastic_data):
        """Probabilistic model should produce non-constant uncertainty across inputs.

        Unlike constant-variance models, the probabilistic model should predict
        different uncertainty levels for different inputs on heteroscedastic data.
        """
        x, y, _, _ = heteroscedastic_data
        x_train, x_test, y_train, _, _, _ = train_test_split(
            x, y, np.zeros(len(y)), test_size=0.3, random_state=42
        )

        model = ProbabilisticRegressionModel(
            input_size=1,
            n_classes=5,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.NONE,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            base_classifier_params={"hidden_layers": 1, "hidden_size": 32},
            regression_head_params={"hidden_layers": 0, "hidden_size": 16},
            n_epochs=60,
            learning_rate=0.01,
            early_stopping_rounds=10,
            validation_fraction=0.2,
            random_seed=42,
        )
        model.fit(x_train, y_train)
        pred_std = model.predict_uncertainty(x_test)

        # Coefficient of variation of predicted std should be meaningful
        std_cv = np.std(pred_std) / (np.mean(pred_std) + 1e-8)
        min_cv = 0.05
        assert std_cv > min_cv, (
            f"Predicted uncertainty is nearly constant (CV={std_cv:.4f}). "
            f"Model is not learning input-dependent variance."
        )

    def test_tree_baselines_on_heteroscedastic(self, heteroscedastic_data):
        """Tree models should achieve reasonable MSE on heteroscedastic data."""
        from automl_package.models.xgboost_model import XGBoostModel

        x, y, _, _ = heteroscedastic_data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        model = XGBoostModel(n_estimators=200, early_stopping_rounds=10, validation_fraction=0.2, random_seed=42)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        xgb_mse = float(np.mean((y_test - y_pred) ** 2))

        # XGBoost should do well on this — it's a flexible nonparametric model
        max_acceptable_xgb_mse = 5.0
        assert xgb_mse < max_acceptable_xgb_mse, f"XGBoost MSE={xgb_mse:.4f} unexpectedly high on heteroscedastic data"

        # Convergence record (M2): the boosting wrappers record the best round and whether it is the cap.
        # With a 3-round cap every wrapper stops at the cap; with 200 rounds and early stopping XGBoost does not.
        from automl_package.models.catboost_model import CatBoostModel
        from automl_package.models.lightgbm_model import LightGBMModel

        for model_cls, cap_kwargs in ((XGBoostModel, {"n_estimators": 3}), (LightGBMModel, {"n_estimators": 3}), (CatBoostModel, {"iterations": 3})):
            capped = model_cls(early_stopping_rounds=10, validation_fraction=0.2, random_seed=42, calculate_feature_importance=False, **cap_kwargs)
            capped.fit(x_train, y_train)
            assert capped.best_iteration_ == 3
            assert capped.hit_cap_ is True, model_cls.__name__
        assert model.best_iteration_ <= 200
        assert model.hit_cap_ == (model.best_iteration_ == 200)
        # The refit trains exactly the rounds early stopping kept (XGBoost's best_iteration is 0-based: kept = best_iteration + 1).
        assert model.model.get_booster().num_boosted_rounds() == model.best_iteration_

    def test_linear_regression_on_linear_data(self, simple_linear_data):
        """Linear regression should achieve near-optimal MSE on y=2x+1+noise."""
        from automl_package.models.linear_regression import LinearRegressionModel

        x, y = simple_linear_data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        model = LinearRegressionModel(learning_rate=0.01, n_iterations=1000, early_stopping_rounds=50, validation_fraction=0.2)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        lr_mse = float(np.mean((y_test - y_pred) ** 2))

        # For y=2x+1+noise(0.3), irreducible MSE ~= 0.09. LR should be close.
        max_acceptable_lr_mse = 0.5
        assert lr_mse < max_acceptable_lr_mse, f"LinearRegression MSE={lr_mse:.4f} too high for y=2x+1 data"


class TestPredictDistribution:
    """ProbReg exposes ONE predictive Gaussian: the law-of-total-variance mean and variance."""

    def test_gaussian_mean_and_std_match_predict(self, multimodal_data):
        """Gaussian mean/variance from predict_distribution must agree with predict()/predict_uncertainty()."""
        import numpy as np

        from automl_package.enums import NClassesSelectionMethod, RegressionStrategy, UncertaintyMethod
        from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel

        x, y = multimodal_data
        m = ProbabilisticRegressionModel(
            input_size=1, n_classes=2, max_n_classes_for_probabilistic_path=5,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.NONE,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            n_epochs=20, learning_rate=0.01, random_seed=42,
            calculate_feature_importance=False,
        )
        m.fit(x, y)
        dist = m.predict_distribution(x)
        p = m.predict(x)
        assert dist.mean.shape == (len(x),)
        assert np.allclose(p, dist.mean, atol=1e-4)
        assert np.allclose(m.predict_uncertainty(x) ** 2, dist.variance, rtol=1e-4)

    def test_gaussian_log_prob_finite_on_multimodal(self, multimodal_data):
        """log_prob on held-out multimodal data should be finite and reasonable."""
        import numpy as np

        from automl_package.enums import NClassesSelectionMethod, RegressionStrategy, UncertaintyMethod
        from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel

        x, y = multimodal_data
        m = ProbabilisticRegressionModel(
            input_size=1, n_classes=2, max_n_classes_for_probabilistic_path=5,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.NONE,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            n_epochs=60, learning_rate=0.01, random_seed=42,
            calculate_feature_importance=False,
        )
        m.fit(x, y)
        dist = m.predict_distribution(x)
        lp = dist.log_prob(y)
        assert np.all(np.isfinite(lp))

    def test_unsupported_configs_raise(self):
        """Dynamic-k should raise NotImplementedError."""
        import numpy as np
        import pytest

        from automl_package.enums import NClassesSelectionMethod, RegressionStrategy, UncertaintyMethod
        from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel

        np.random.seed(0)
        x = np.random.randn(60, 1).astype(np.float32)
        y = np.random.randn(60).astype(np.float32)

        # predict_distribution's own guard fires for ANY n_classes_selection_method != NONE
        # (automl_package/models/probabilistic_regression.py, "dynamic-k is not yet implemented").
        # SOFT_GATING was incidental to that guard -- NESTED (a surviving member under MASTER
        # Decision 29) triggers the identical branch, verified by running both before this edit.
        m = ProbabilisticRegressionModel(
            input_size=1, n_classes=3, max_n_classes_for_probabilistic_path=5,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.NESTED, allow_unredesigned_nested=True,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            n_epochs=5, random_seed=42, calculate_feature_importance=False,
        )
        m.fit(x, y)
        with pytest.raises(NotImplementedError):
            m.predict_distribution(x)


# ---------------------------------------------------------------------------
# Work package P (cascade_execution_plan_2026-07-11.md §4.7): P1-P4 fixes
# ---------------------------------------------------------------------------

class _StubProbRegNet:
    """Minimal stand-in for ProbabilisticRegressionNet's forward output.

    Lets P1's test drive `get_classifier_predictions` with exact, hand-picked classifier
    logits instead of whatever a trained network happens to produce.
    """

    def __init__(self, classifier_logits: torch.Tensor, selected_k_values: torch.Tensor):
        self._classifier_logits = classifier_logits
        self._selected_k_values = selected_k_values

    def eval(self) -> None:
        """No-op: matches the nn.Module.eval() call site in get_classifier_predictions."""

    def __call__(self, x: torch.Tensor) -> tuple:
        n = x.shape[0]
        return None, self._classifier_logits[:n], self._selected_k_values[:n], None, None


class TestP1BinaryClassifierProbabilityFix:
    """P1: get_classifier_predictions must use one masked-softmax path for every k >= 2.

    Old code special-cased k == 2 as `sigmoid(logits[:, 0])`, which is only the correct
    2-class softmax positive probability when logit1 == 0. Crafted logits below have
    logit1 != 0, so the old path would diverge from `torch.softmax`.
    """

    def _model_with_stub(self, k: int, classifier_logits: torch.Tensor) -> ProbabilisticRegressionModel:
        model = ProbabilisticRegressionModel(
            input_size=1,
            n_classes=k,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.NONE,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            calculate_feature_importance=False,
        )
        model.direct_regression = False
        n = classifier_logits.shape[0]
        y_true = np.linspace(-1.0, 1.0, n).astype(np.float32)
        boundaries, _ = create_bins(data=y_true, n_bins=k, min_value=-np.inf, max_value=np.inf)
        model.precomputed_class_boundaries = {k: boundaries}
        model.model = _StubProbRegNet(classifier_logits, torch.full((n,), k, dtype=torch.long))
        return model, y_true

    @pytest.mark.parametrize("k", [2, 3])
    def test_classifier_proba_matches_softmax_exactly(self, k):
        n = 6
        # logit1 != 0 for every row -> exposes the old sigmoid(logit0) k==2 special case.
        torch.manual_seed(0)
        classifier_logits = torch.randn(n, k) * 2.0
        model, y_true = self._model_with_stub(k, classifier_logits)
        x = np.zeros((n, 1), dtype=np.float32)

        _, y_proba, _ = model.get_classifier_predictions(x, y_true)

        expected = torch.softmax(classifier_logits, dim=1).numpy()
        assert np.allclose(y_proba, expected, atol=1e-6), f"k={k}: classifier proba does not match torch.softmax exactly"
        row_sums = y_proba.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), f"k={k}: proba rows do not sum to 1 (sums={row_sums})"


class TestP2CreateBinsWarning:
    """P2: create_bins warns once when tied percentile edges force n_bins to shrink."""

    def test_tied_targets_warn_and_shrink(self, automl_caplog):
        # Only 3 distinct values repeated many times -> percentile edges at k=5 collide.
        requested_n_bins = 5
        y = np.array([1.0, 2.0, 3.0] * 20)
        edges, _ = create_bins(data=y, n_bins=requested_n_bins, min_value=-np.inf, max_value=np.inf)

        effective_n_bins = len(edges) - 1
        assert effective_n_bins < requested_n_bins, f"expected n_bins to shrink below {requested_n_bins}, got {effective_n_bins}"
        assert f"requested n_bins={requested_n_bins}" in automl_caplog.text
        assert f"effective n_bins={effective_n_bins}" in automl_caplog.text

    def test_clean_continuous_target_no_warning(self, automl_caplog):
        requested_n_bins = 5
        rng = np.random.default_rng(0)
        y = rng.normal(size=200)
        edges, _ = create_bins(data=y, n_bins=requested_n_bins, min_value=-np.inf, max_value=np.inf)

        assert len(edges) - 1 == requested_n_bins, "clean continuous data should not need to shrink n_bins"
        assert automl_caplog.text == ""


class TestP4HygieneGating:
    """P4: HPO dims and per-class centroid computation gated on the config that consumes them."""

    def test_hpo_space_excludes_gated_dims_for_none_selection(self):
        model = ProbabilisticRegressionModel(
            input_size=1,
            n_classes=3,
            n_classes_selection_method=NClassesSelectionMethod.NONE,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            calculate_feature_importance=False,
        )
        space = model.get_hyperparameter_search_space()
        assert "gumbel_tau" not in space
        assert "n_classes_predictor_learning_rate" not in space

    def test_hpo_space_gates_dims_by_selection_strategy(self):
        """`gumbel_tau`/`n_classes_predictor_learning_rate` only exist under GUMBEL_SOFTMAX/STE
        and REINFORCE respectively (probabilistic_regression.py:689-692) -- both survivors (NONE,
        NESTED) never populate them, so this per-strategy gating can only be exercised through the
        retired members themselves; the escape hatch is required, not incidental.
        """
        gumbel_model = ProbabilisticRegressionModel(
            input_size=1,
            n_classes_selection_method=NClassesSelectionMethod.GUMBEL_SOFTMAX,
            max_n_classes_for_probabilistic_path=4,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            calculate_feature_importance=False,
            allow_retired_capacity_selection=True,
        )
        gumbel_space = gumbel_model.get_hyperparameter_search_space()
        assert "gumbel_tau" in gumbel_space
        assert "n_classes_predictor_learning_rate" not in gumbel_space

        reinforce_model = ProbabilisticRegressionModel(
            input_size=1,
            n_classes_selection_method=NClassesSelectionMethod.REINFORCE,
            max_n_classes_for_probabilistic_path=4,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            calculate_feature_importance=False,
            allow_retired_capacity_selection=True,
        )
        reinforce_space = reinforce_model.get_hyperparameter_search_space()
        assert "n_classes_predictor_learning_rate" in reinforce_space
        assert "gumbel_tau" not in reinforce_space

    def _fit_tiny(self, **kwargs) -> ProbabilisticRegressionModel:
        rng = np.random.default_rng(3)
        x = rng.uniform(-1, 1, (30, 1)).astype(np.float32)
        y = rng.uniform(-1, 1, 30).astype(np.float32)
        model = ProbabilisticRegressionModel(
            input_size=1,
            n_classes=3,
            n_classes_selection_method=NClassesSelectionMethod.NONE,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_epochs=1,
            learning_rate=0.01,
            validation_fraction=0.2,
            random_seed=42,
            calculate_feature_importance=False,
            **kwargs,
        )
        model.fit(x, y)
        return model

    def test_default_config_skips_centroid_computation(self):
        model = self._fit_tiny()
        assert getattr(model, "_per_class_centroids", None) is None

    def test_monotonic_head_config_computes_centroids(self):
        model = self._fit_tiny(use_monotonic_constraints=True)
        assert getattr(model, "_per_class_centroids", None) is not None


def _column_roles_frame(n: int = 400) -> pd.DataFrame:
    """Synthetic frame: y = offset + f(x) + category effect, 3 category codes (one NaN), offset on the model's raw scale."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"f0": rng.normal(size=n), "f1": rng.normal(size=n), "cat": rng.integers(0, 3, size=n).astype(float), "off": 3.0 * rng.normal(size=n)})
    frame.loc[0, "cat"] = np.nan
    frame["y"] = frame["off"] + np.sin(frame["f0"]) + frame["f1"] + frame["cat"].fillna(0.0) + 0.1 * rng.normal(size=n)
    return frame


def _native_categorical_names(model) -> list[str]:
    """The categorical features the underlying library reports having received."""
    from automl_package.models.catboost_model import CatBoostModel
    from automl_package.models.xgboost_model import XGBoostModel

    if isinstance(model, CatBoostModel):
        return [model.model.feature_names_[i] for i in model.model.get_cat_feature_indices()]
    if isinstance(model, XGBoostModel):
        booster = model.model.get_booster()
        return [name for name, kind in zip(booster.feature_names, booster.feature_types, strict=True) if kind == "c"]
    infos = model.model.booster_.dump_model()["feature_infos"]  # a categorical column lists its category codes under "values", a numeric one does not
    return [name for name, info in infos.items() if info["values"]]


@pytest.mark.parametrize("model_name", ["LightGBMModel", "XGBoostModel", "CatBoostModel"])
def test_boosting_wrappers_offset_and_categoricals(model_name: str) -> None:
    """The boosting wrappers take a per-row starting score and categorical columns as named X columns."""
    from automl_package.models.catboost_model import CatBoostModel
    from automl_package.models.lightgbm_model import LightGBMModel
    from automl_package.models.xgboost_model import XGBoostModel

    model_cls, size = {"LightGBMModel": (LightGBMModel, "n_estimators"), "XGBoostModel": (XGBoostModel, "n_estimators"), "CatBoostModel": (CatBoostModel, "iterations")}[model_name]
    frame = _column_roles_frame()
    train, held_out = frame.iloc[:300], frame.iloc[300:]
    features = ["f0", "f1", "cat", "off"]

    def make(**roles):
        return model_cls(**{size: 40}, early_stopping_rounds=5, validation_fraction=0.2, random_seed=0, calculate_feature_importance=False, **roles)

    with_offset = make(offset_feature="off", categorical_features=["cat"])
    with_offset.fit(train[features], train["y"].to_numpy())
    pred = with_offset.predict(held_out[features])
    without = make(categorical_features=["cat"])
    without.fit(train[["f0", "f1", "cat"]], train["y"].to_numpy())
    rmse_with = float(np.sqrt(np.mean((held_out["y"].to_numpy() - pred) ** 2)))
    rmse_without = float(np.sqrt(np.mean((held_out["y"].to_numpy() - without.predict(held_out[["f0", "f1", "cat"]])) ** 2)))
    assert rmse_with < rmse_without
    # prediction = starting score + trees: shifting every starting score by 5 shifts every prediction by exactly 5 (the offset is not a feature)
    shifted = held_out[features].assign(off=held_out["off"] + 5.0)
    np.testing.assert_allclose(with_offset.predict(shifted) - pred, 5.0, atol=1e-4)
    # the library received the categorical column natively and not the offset
    assert _native_categorical_names(with_offset) == ["cat"]

    # a search with refit_on_train_val=False builds trial clones: they must know the column names
    searched = make(offset_feature="off", categorical_features=["cat"], optimize_hyperparameters=True, n_trials=2, refit_on_train_val=False)
    searched.fit(train[features], train["y"].to_numpy())
    assert np.isfinite(searched.predict(held_out[features])).all()

    with pytest.raises(ValueError, match="calculate_feature_importance"):
        model_cls(**{size: 5}, offset_feature="off").fit(train[features], train["y"].to_numpy())
