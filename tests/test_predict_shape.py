"""Regression tests: PyTorchNeuralNetwork.predict() returns 1-D array, not (N, 1).

Open issue historically flagged in RESUME.md. Adding an explicit test so
the invariant is preserved across refactors.
"""

import numpy as np
import pytest

from automl_package.enums import TaskType, UncertaintyMethod
from automl_package.models.neural_network import PyTorchNeuralNetwork


@pytest.fixture
def trained_regression_nn():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 3)).astype(np.float32)
    y = (x[:, 0] + 0.5 * x[:, 1] + rng.normal(0, 0.1, 200)).astype(np.float32)
    model = PyTorchNeuralNetwork(
        input_size=3,
        output_size=1,
        task_type=TaskType.REGRESSION,
        hidden_layers=1,
        hidden_size=8,
        n_epochs=5,
        random_seed=0,
        calculate_feature_importance=False,
    )
    model.fit(x, y)
    return model, x


def test_predict_returns_1d_array(trained_regression_nn):
    model, x = trained_regression_nn
    preds = model.predict(x)
    assert preds.ndim == 1, f"predict should return (N,) not {preds.shape}"
    assert preds.shape[0] == x.shape[0]


def test_predict_uncertainty_returns_1d_array(trained_regression_nn):
    model, x = trained_regression_nn
    std = model.predict_uncertainty(x)
    assert std.ndim == 1, f"predict_uncertainty should return (N,) not {std.shape}"
    assert std.shape[0] == x.shape[0]
