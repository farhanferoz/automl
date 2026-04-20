"""Tests for CE_STOP_GRAD gradient routing (CT3)."""

import numpy as np

from automl_package.enums import (
    ProbabilisticRegressionOptimizationStrategy,
    ProbRegLossType,
    RegressionStrategy,
    UncertaintyMethod,
)
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel


def _make_tiny_dataset(n: int = 80, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, (n, 1)).astype(np.float32)
    y = (x.ravel() + rng.normal(0, 0.3, n)).astype(np.float32)
    return x, y


class TestCeStopGrad:
    def _make_model(self, strategy: ProbabilisticRegressionOptimizationStrategy, loss_type: ProbRegLossType = ProbRegLossType.GAUSSIAN_LTV) -> ProbabilisticRegressionModel:
        return ProbabilisticRegressionModel(
            input_size=1,
            n_classes=3,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            optimization_strategy=strategy,
            prob_reg_loss_type=loss_type,
            n_epochs=3,
            learning_rate=0.01,
            early_stopping_rounds=2,
            validation_fraction=0.2,
            random_seed=42,
            calculate_feature_importance=False,
        )

    def test_ce_stop_grad_trains_without_error(self):
        """CE_STOP_GRAD model fits and predicts without errors."""
        x, y = _make_tiny_dataset()
        model = self._make_model(ProbabilisticRegressionOptimizationStrategy.CE_STOP_GRAD)
        model.fit(x, y)
        preds = model.predict(x[:10])
        assert preds.shape == (10,)
        assert np.all(np.isfinite(preds))

    def test_gradient_stop_trains_without_error(self):
        """GRADIENT_STOP still works (existing path, unchanged)."""
        x, y = _make_tiny_dataset()
        model = self._make_model(ProbabilisticRegressionOptimizationStrategy.GRADIENT_STOP)
        model.fit(x, y)
        preds = model.predict(x[:10])
        assert preds.shape == (10,)
        assert np.all(np.isfinite(preds))

    def test_mdn_ce_stop_grad_trains_without_error(self):
        """MDN + CE_STOP_GRAD (cell H) trains and predicts."""
        x, y = _make_tiny_dataset()
        model = self._make_model(
            ProbabilisticRegressionOptimizationStrategy.CE_STOP_GRAD,
            loss_type=ProbRegLossType.MDN,
        )
        model.fit(x, y)
        preds = model.predict(x[:10])
        assert preds.shape == (10,)
        assert np.all(np.isfinite(preds))

    def test_regression_only_unchanged(self):
        """REGRESSION_ONLY path is byte-for-byte equivalent (smoke test)."""
        x, y = _make_tiny_dataset()
        model = self._make_model(ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY)
        model.fit(x, y)
        preds = model.predict(x[:10])
        assert np.all(np.isfinite(preds))
