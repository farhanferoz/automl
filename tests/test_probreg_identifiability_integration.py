"""Integration smoke tests for ProbReg identifiability cells (CT5).

Runs all 8 cells (2 loss types × 2 supervision modes × 2 head parametrizations)
for 10 epochs on a tiny synthetic dataset and checks for no crashes, finite losses,
and sensible predict_distribution output.

Total runtime: ~30 seconds.
"""

import numpy as np
import pytest

from automl_package.enums import (
    ProbabilisticRegressionOptimizationStrategy,
    ProbRegLossType,
    RegressionStrategy,
    UncertaintyMethod,
)
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel

_CELLS = [
    ("A", ProbRegLossType.GAUSSIAN_LTV, ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY, False),
    ("B", ProbRegLossType.GAUSSIAN_LTV, ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY, True),
    ("C", ProbRegLossType.GAUSSIAN_LTV, ProbabilisticRegressionOptimizationStrategy.CE_STOP_GRAD, False),
    ("D", ProbRegLossType.GAUSSIAN_LTV, ProbabilisticRegressionOptimizationStrategy.CE_STOP_GRAD, True),
    ("E", ProbRegLossType.MDN, ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY, False),
    ("F", ProbRegLossType.MDN, ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY, True),
    ("G", ProbRegLossType.MDN, ProbabilisticRegressionOptimizationStrategy.CE_STOP_GRAD, False),
    ("H", ProbRegLossType.MDN, ProbabilisticRegressionOptimizationStrategy.CE_STOP_GRAD, True),
]


def _tiny_dataset(n: int = 100, seed: int = 99) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, (n, 1)).astype(np.float32)
    y = (np.sin(x.ravel()) + rng.normal(0, 0.2, n)).astype(np.float32)
    return x, y


@pytest.mark.parametrize(("cell_id", "loss_type", "opt_strategy", "use_anchored"), _CELLS)
def test_cell_smoke(cell_id, loss_type, opt_strategy, use_anchored):
    """Each cell trains without crash and produces finite predictions."""
    x, y = _tiny_dataset()
    x_tr, y_tr = x[:80], y[:80]
    x_te = x[80:]

    constrain_mid = not use_anchored  # anchored heads subsume middle-class constraint

    model = ProbabilisticRegressionModel(
        input_size=1,
        n_classes=3,
        regression_strategy=RegressionStrategy.SEPARATE_HEADS,
        uncertainty_method=UncertaintyMethod.PROBABILISTIC,
        optimization_strategy=opt_strategy,
        prob_reg_loss_type=loss_type,
        use_anchored_heads=use_anchored,
        constrain_middle_class=constrain_mid,
        n_epochs=10,
        learning_rate=0.01,
        early_stopping_rounds=5,
        validation_fraction=0.2,
        random_seed=42,
        calculate_feature_importance=False,
    )

    model.fit(x_tr, y_tr)

    preds = model.predict(x_te)
    assert preds.shape == (len(x_te),), f"Cell {cell_id}: wrong prediction shape"
    assert np.all(np.isfinite(preds)), f"Cell {cell_id}: non-finite predictions"

    mse = float(np.mean((preds - y[80:]) ** 2))
    assert mse < 100.0, f"Cell {cell_id}: unreasonably high MSE {mse:.2f}"

    dist = model.predict_distribution(x_te)
    assert dist is not None, f"Cell {cell_id}: predict_distribution returned None"
