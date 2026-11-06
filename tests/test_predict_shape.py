"""Regression tests: PyTorchNeuralNetwork.predict() returns 1-D array, not (N, 1).

Open issue historically flagged in RESUME.md. Adding an explicit test so
the invariant is preserved across refactors.
"""

import numpy as np
import pandas as pd
import pytest

from automl_package.enums import OutOfRangeAction, TaskType, UncertaintyMethod
from automl_package.models.neural_network import PyTorchNeuralNetwork
from automl_package.models.normal_equation_linear_regression import NormalEquationLinearRegression


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


# --- training envelope: out-of-range reporting in BaseModel.predict ---------------------------------------------------


def _fitted_line(action: OutOfRangeAction = OutOfRangeAction.FLAG, categorical: bool = False) -> NormalEquationLinearRegression:
    """y = 2 * x0 on x0 in [-5, 5] (zero-mean, as this solver has no centred intercept); column x1 holds category codes {-1, 1}."""
    x = np.column_stack([np.linspace(-5.0, 5.0, 50), np.tile([-1.0, 1.0], 25)])
    model = NormalEquationLinearRegression(
        out_of_range_action=action, calculate_feature_importance=False, validation_fraction=0.0, test_fraction=0.0,
        input_scaling="none", missing_inputs="native",  # the envelope on raw inputs; `_predict` is called directly and a NaN input is passed on purpose
        # a role needs column names (fitted on a DataFrame); the one-hot columns of a category sum to 1, so the solve needs a ridge to stay non-singular
        **({"categorical_features": ["feature_1"], "l2_lambda": 1e-6} if categorical else {}),
    )
    model.fit(pd.DataFrame(x, columns=["feature_0", "feature_1"]), 2.0 * x[:, 0])
    return model


def test_envelope_report_counts_and_actions():
    x_new = np.array([[0.0, 1.0], [10.0, -1.0], [-10.0, 1.0]])  # rows 1 and 2 are beyond the training x0 range
    flagged = _fitted_line()
    plain = flagged._predict(x_new)
    out = flagged.predict(x_new)
    np.testing.assert_array_equal(out, plain)  # FLAG leaves the values bit-identical
    rep = flagged.last_prediction_report_
    assert (rep.n_rows, rep.n_outputs, rep.n_outputs_out_of_range) == (3, 3, 2)
    assert rep.share_outputs_out_of_range == pytest.approx(2 / 3)
    assert rep.worst_exceedance == pytest.approx(0.5, abs=0.01)  # +-20 predicted vs +-10 range edge, over range width 20
    assert (rep.n_rows_input_out_of_range, rep.top_input_columns) == (2, (("feature_0", 2),))
    assert rep.as_dict()["range_top_input_columns"] == "feature_0:2"

    clipped = _fitted_line(OutOfRangeAction.CLIP).predict(x_new)
    np.testing.assert_allclose(clipped, [0.0, 10.0, -10.0], atol=0.2)
    with pytest.raises(ValueError, match="2/3"):
        _fitted_line("raise").predict(x_new)  # the value form of the enum is accepted


def test_envelope_in_range_zero_and_unseen_category():
    model = _fitted_line(categorical=True)
    model.predict(np.array([[-1.0, 1.0], [3.0, -1.0]]))
    rep = model.last_prediction_report_
    assert (rep.n_outputs_out_of_range, rep.worst_exceedance, rep.n_rows_input_out_of_range, rep.n_rows_unseen_category) == (0, 0.0, 0, 0)
    model.predict(np.array([[-1.0, 1.0], [3.0, 7.0]]))
    assert model.last_prediction_report_.n_rows_unseen_category == 1
    model.predict(np.array([[np.nan, 1.0]]))  # a NaN input gives a NaN prediction: a fault, counted as out of range, while the NaN input itself is not
    assert (model.last_prediction_report_.n_outputs_out_of_range, model.last_prediction_report_.n_rows_input_out_of_range) == (1, 0)
