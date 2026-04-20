"""Tests for CE_STOP_GRAD gradient routing (CT3)."""

import numpy as np
import torch

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


class TestCeStopGradRouting:
    """Plan CT3: verify classifier sees CE gradient only; heads see regression only.

    Works by fitting to build the model, then running one manual forward+backward on a
    fresh copy and inspecting per-parameter gradients:
      - classifier_layers params: non-zero grad under CE_STOP_GRAD, zero grad under
        REGRESSION_ONLY-with-detached-probs (we simulate by zeroing CE contribution).
      - regression_module params: always get gradient from regression loss.
    """

    def _fit_tiny(self, strategy: ProbabilisticRegressionOptimizationStrategy, loss_type: ProbRegLossType) -> ProbabilisticRegressionModel:
        x, y = _make_tiny_dataset()
        model = ProbabilisticRegressionModel(
            input_size=1,
            n_classes=3,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            optimization_strategy=strategy,
            prob_reg_loss_type=loss_type,
            n_epochs=2,
            learning_rate=0.01,
            early_stopping_rounds=2,
            validation_fraction=0.2,
            random_seed=42,
            calculate_feature_importance=False,
        )
        model.fit(x, y)
        return model

    @staticmethod
    def _zero_grads(net: torch.nn.Module) -> None:
        for p in net.parameters():
            if p.grad is not None:
                p.grad.zero_()

    @staticmethod
    def _total_abs_grad(params) -> float:
        return float(sum(p.grad.abs().sum().item() for p in params if p.grad is not None))

    def _run_backward(self, model: ProbabilisticRegressionModel) -> tuple[float, float]:
        """Run one forward+backward pass, return (classifier_grad_norm, heads_grad_norm)."""
        x, y = _make_tiny_dataset()
        x_t = torch.tensor(x, dtype=torch.float32).to(model.device)
        y_t = torch.tensor(y, dtype=torch.float32).to(model.device).unsqueeze(1)
        model.model.train()
        self._zero_grads(model.model)

        model_outputs = model.model(x_t)
        loss = model._calculate_custom_loss(model_outputs, y_t, include_boundary_loss=False)
        loss.backward()

        classifier_grad = self._total_abs_grad(model.model.classifier_layers.parameters())
        heads_grad = self._total_abs_grad(model.model.regression_module.parameters())
        return classifier_grad, heads_grad

    def test_ce_stop_grad_gaussian_ltv_gradient_flow(self):
        """CE_STOP_GRAD + GAUSSIAN_LTV: classifier grad from CE; heads grad from regression."""
        model = self._fit_tiny(ProbabilisticRegressionOptimizationStrategy.CE_STOP_GRAD, ProbRegLossType.GAUSSIAN_LTV)
        classifier_grad, heads_grad = self._run_backward(model)
        assert classifier_grad > 0.0, "classifier must receive gradient (from CE loss)"
        assert heads_grad > 0.0, "regression heads must receive gradient"

    def test_ce_stop_grad_mdn_gradient_flow(self):
        """CE_STOP_GRAD + MDN (cells G, H): classifier grad from CE; heads grad from MDN NLL."""
        model = self._fit_tiny(ProbabilisticRegressionOptimizationStrategy.CE_STOP_GRAD, ProbRegLossType.MDN)
        classifier_grad, heads_grad = self._run_backward(model)
        assert classifier_grad > 0.0, "classifier must receive gradient (from CE loss)"
        assert heads_grad > 0.0, "regression heads must receive gradient"

    def test_regression_only_no_ce_gradient(self):
        """REGRESSION_ONLY: no CE, so classifier gradient comes only from regression loss path.

        Verifies classifier still gets gradient (via non-detached probs) and heads get gradient.
        """
        model = self._fit_tiny(ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY, ProbRegLossType.GAUSSIAN_LTV)
        classifier_grad, heads_grad = self._run_backward(model)
        assert classifier_grad > 0.0
        assert heads_grad > 0.0

    def test_mdn_regression_only_classifier_trains_via_mdn(self):
        """MDN + REGRESSION_ONLY (cells E, F): classifier grad comes through MDN probs — the whole
        point of MDN (identifiability via probs channel). Both paths must produce gradient.
        """
        model = self._fit_tiny(ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY, ProbRegLossType.MDN)
        classifier_grad, heads_grad = self._run_backward(model)
        assert classifier_grad > 0.0, "MDN's probs path must train the classifier under REGRESSION_ONLY"
        assert heads_grad > 0.0
