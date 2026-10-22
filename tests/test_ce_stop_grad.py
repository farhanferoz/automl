"""Tests for the cross-entropy training modes and CE_STOP_GRAD gradient routing (CT3).

CE_STOP_GRAD is the default `optimization_strategy` (owner ruling 2026-10-22): the classifier is
trained on cross-entropy against the y bins only, and its probabilities are detached before the
regression heads, so the regression loss cannot pull them off the true bin probabilities. All
cross-entropy modes construct without `allow_retired_capacity_selection`.
"""

import numpy as np
import pytest
import torch

from automl_package.enums import (
    Metric,
    ProbabilisticRegressionOptimizationStrategy,
    RegressionStrategy,
    UncertaintyMethod,
)
from automl_package.models.common.losses import calculate_combined_loss
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel


def _make_tiny_dataset(n: int = 80, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, (n, 1)).astype(np.float32)
    y = (x.ravel() + rng.normal(0, 0.3, n)).astype(np.float32)
    return x, y


def _make_model(strategy: ProbabilisticRegressionOptimizationStrategy | None = None, n_epochs: int = 3) -> ProbabilisticRegressionModel:
    """A tiny ProbReg; `strategy` left unset exercises the constructor default."""
    kwargs = {} if strategy is None else {"optimization_strategy": strategy}
    return ProbabilisticRegressionModel(
        input_size=1,
        n_classes=3,
        regression_strategy=RegressionStrategy.SEPARATE_HEADS,
        uncertainty_method=UncertaintyMethod.PROBABILISTIC,
        n_epochs=n_epochs,
        learning_rate=0.01,
        early_stopping_rounds=2,
        validation_fraction=0.2,
        random_seed=42,
        calculate_feature_importance=False,
        **kwargs,
    )


class TestCeModesConstructAndTrain:
    def test_default_is_ce_stop_grad(self):
        model = _make_model()
        assert model.optimization_strategy is ProbabilisticRegressionOptimizationStrategy.CE_STOP_GRAD
        assert model._get_optimization_metric() is Metric.MSE, "search and CV score on held-out regression error only"

    @pytest.mark.parametrize(
        "extra",
        [
            {"cv_folds": 2, "validation_fraction": None},
            {"optimize_hyperparameters": True, "n_trials": 2, "validation_fraction": 0.2},
            {"n_classes": 5, "n_classes_inf": 3, "validation_fraction": 0.2},
        ],
        ids=["cv", "hpo", "direct_regression"],
    )
    def test_default_fits_under_cv_hpo_and_direct_regression(self, extra):
        """The CE_STOP_GRAD default must not crash where trials are scored on a clone (CV, search) or where no classes exist."""
        x, y = _make_tiny_dataset()
        kwargs = {"input_size": 1, "n_classes": 3, "n_epochs": 2, "early_stopping_rounds": 2, "random_seed": 0, "calculate_feature_importance": False} | extra
        model = ProbabilisticRegressionModel(**kwargs)
        model.fit(x, y)
        assert np.all(np.isfinite(model.predict(x[:10])))

    @pytest.mark.parametrize("strategy", list(ProbabilisticRegressionOptimizationStrategy))
    def test_every_mode_constructs_without_escape_hatch_and_trains(self, strategy):
        """All four modes (including the cross-entropy ones) construct with no `allow_retired_capacity_selection`."""
        x, y = _make_tiny_dataset()
        model = _make_model(strategy)
        assert model.allow_retired_capacity_selection is False
        model.fit(x, y)
        preds = model.predict(x[:10])
        assert preds.shape == (10,)
        assert np.all(np.isfinite(preds))


class TestCeStopGradRouting:
    """Plan CT3: verify classifier sees CE gradient only; heads see regression only.

    Works by fitting to build the model, then running one manual forward+backward on a
    fresh copy and inspecting per-parameter gradients:
      - classifier_layers params: non-zero grad under CE_STOP_GRAD, zero grad under
        REGRESSION_ONLY-with-detached-probs (we simulate by zeroing CE contribution).
      - regression_module params: always get gradient from regression loss.
    """

    def _fit_tiny(self, strategy: ProbabilisticRegressionOptimizationStrategy | None = None) -> ProbabilisticRegressionModel:
        x, y = _make_tiny_dataset()
        model = _make_model(strategy, n_epochs=2)
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

    def test_default_classifier_gets_ce_gradient_and_regression_loss_sends_none(self):
        """Default model: the classifier is trained by cross-entropy alone; the regression loss never reaches it.

        The regression term alone must leave every classifier gradient at zero (probabilities are
        detached before the heads) while the heads still learn from it; the full loss must give the
        classifier a gradient, and that gradient can only come from the cross-entropy term.
        """
        model = self._fit_tiny()
        assert model.optimization_strategy is ProbabilisticRegressionOptimizationStrategy.CE_STOP_GRAD
        x, y = _make_tiny_dataset()
        x_t = torch.tensor(x, dtype=torch.float32).to(model.device)
        y_t = torch.tensor(y, dtype=torch.float32).to(model.device).unsqueeze(1)
        model.model.train()

        self._zero_grads(model.model)
        outputs = model.model(x_t)
        regression_loss = calculate_combined_loss(
            predictions=outputs[0], y_true=y_t, uncertainty_method=model.uncertainty_method,
            include_boundary_loss=False, loss_type=model.loss_type, beta=model.beta,
        )
        regression_loss.backward()
        assert self._total_abs_grad(model.model.classifier_layers.parameters()) == 0.0, "regression loss must send no gradient into the classifier"
        assert self._total_abs_grad(model.model.regression_module.parameters()) > 0.0, "regression heads must learn from the regression loss"

        classifier_grad, heads_grad = self._run_backward(model)
        assert classifier_grad > 0.0, "classifier must receive gradient from the cross-entropy loss"
        assert heads_grad > 0.0

    def test_regression_only_no_ce_gradient(self):
        """REGRESSION_ONLY: no CE, so classifier gradient comes only from regression loss path.

        Verifies classifier still gets gradient (via non-detached probs) and heads get gradient.
        """
        model = self._fit_tiny(ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY)
        classifier_grad, heads_grad = self._run_backward(model)
        assert classifier_grad > 0.0
        assert heads_grad > 0.0

    def test_gradient_stop_classifier_still_receives_regression_gradient(self):
        """GRADIENT_STOP with fixed k: CE is off, but regression loss still flows through probs.

        Pre-change semantics (preserved): GRADIENT_STOP only disables the CE term; the classifier
        is still supervised by regression via the non-detached probs path in NoneStrategy.forward.
        If detach were incorrectly applied here the classifier grad would be zero (no CE, no grad
        path), which was the bug flagged for restoration.
        """
        model = self._fit_tiny(ProbabilisticRegressionOptimizationStrategy.GRADIENT_STOP)
        classifier_grad, heads_grad = self._run_backward(model)
        assert classifier_grad > 0.0, "GRADIENT_STOP + fixed k must route regression gradient to classifier (CE is off)"
        assert heads_grad > 0.0
