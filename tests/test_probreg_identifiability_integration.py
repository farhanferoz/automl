"""Integration smoke tests for ProbReg identifiability cells (CT5).

Runs all 4 cells (2 supervision modes × 2 head parametrizations) for 10 epochs on a tiny
synthetic dataset and checks for no crashes, finite losses, and a sensible predict_distribution
output (the single predictive Gaussian).
"""

import numpy as np
import pytest

from automl_package.enums import (
    ProbabilisticRegressionOptimizationStrategy,
    RegressionStrategy,
    UncertaintyMethod,
)
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel

_CELLS = [
    ("A", ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY, False),
    ("B", ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY, True),
    ("C", ProbabilisticRegressionOptimizationStrategy.CE_STOP_GRAD, False),
    ("D", ProbabilisticRegressionOptimizationStrategy.CE_STOP_GRAD, True),
]


def _tiny_dataset(n: int = 100, seed: int = 99) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, (n, 1)).astype(np.float32)
    y = (np.sin(x.ravel()) + rng.normal(0, 0.2, n)).astype(np.float32)
    return x, y


@pytest.mark.parametrize(("cell_id", "opt_strategy", "use_anchored"), _CELLS)
def test_cell_smoke(cell_id, opt_strategy, use_anchored):
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
    catastrophic_failure_mse = 100.0
    assert mse < catastrophic_failure_mse, f"Cell {cell_id}: unreasonably high MSE {mse:.2f}"

    dist = model.predict_distribution(x_te)
    np.testing.assert_allclose(dist.mean, preds, rtol=1e-5, atol=1e-6, err_msg=f"Cell {cell_id}: Gaussian mean must equal predict()")
    assert np.all(dist.variance > 0.0), f"Cell {cell_id}: non-positive predictive variance"
