"""PyTorchLSTMForecaster: learns a window-driven signal, skips NaN targets, and standardises on its own fit rows."""

import math

import numpy as np
import pytest
import torch

from automl_package.enums import ForecastLoss, OutputBound, RnnBackend, TargetScaling, UncertaintyMethod
from automl_package.models.common.output_bound import TrainingRangeBound
from automl_package.models.common.unknown_category_embedding import UnknownCategoryEmbedding
from automl_package.models.lstm_model import PyTorchLSTMForecaster, SequenceLayout
from automl_package.models.neural_network import PyTorchNeuralNetwork
from automl_package.models.pytorch_linear_regression import PyTorchLinearRegression
from automl_package.utils.losses import poisson_deviance

N_SERIES, N_STEPS, N_CHANNELS, N_DENSE, N_HORIZONS = 3, 8, 2, 3, 2


def _panel(n_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows in the flat layout; horizon 0 is the mean of the last four window values, horizon 1 the series offset."""
    rng = np.random.default_rng(seed)
    series = rng.integers(0, N_SERIES, n_rows)
    window = rng.normal(size=(n_rows, N_STEPS, N_CHANNELS)).astype(np.float32)
    dense = rng.normal(size=(n_rows, N_DENSE)).astype(np.float32)
    x = np.concatenate([series[:, None].astype(np.float32), window.reshape(n_rows, -1), dense], axis=1)
    y = np.column_stack([window[:, -4:, 0].mean(axis=1), 0.5 * series]).astype(np.float32)
    return x, y


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> PyTorchLSTMForecaster:
    monkeypatch.setenv("AUTOML_DEVICE", "cpu")
    return PyTorchLSTMForecaster(
        layout=SequenceLayout(N_SERIES, N_STEPS, N_CHANNELS),
        output_size=N_HORIZONS,
        window=N_STEPS,
        hidden_size=16,
        embedding_dim=4,
        learning_rate=1e-2,
        batch_size=64,
        n_epochs=40,
        early_stopping_rounds=40,
        refit_on_train_val=False,
        calculate_feature_importance=False,
        random_seed=0,
    )


def test_fits_through_the_public_fit_and_predicts_every_horizon(model: PyTorchLSTMForecaster) -> None:
    x_train, y_train = _panel(600, 1)
    x_val, y_val = _panel(200, 2)
    y_train[::2, 0] = np.nan  # purged horizon-0 targets are skipped, not treated as zero
    model.fit(x_train, y_train, x_val=x_val, y_val=y_val)
    pred = model.predict(x_val)
    assert pred.shape == (200, N_HORIZONS)
    assert model.convergence_.best_val < 0.5 * float(np.nanvar(y_val))  # the signal was learned
    assert model._held_out_fit_score(x_val, y_val) == pytest.approx(model.convergence_.best_val, rel=1e-4)  # best epoch restored
    nan = float("nan")
    targets = torch.tensor([[1.0, nan], [nan, nan], [2.0, 3.0], [nan, 1.0]])
    assert model._fit_loss(torch.zeros(4, 2), targets).item() == pytest.approx(15 / 4)  # mean over the 4 observed entries
    assert model._fit_loss(torch.zeros(4, 2), targets, reduction="none").tolist() == pytest.approx([1.0, 0.0, 6.5, 1.0])
    with pytest.raises(NotImplementedError):
        model.predict_uncertainty(x_val)
    # rnn_backend: the default (cuDNN) reaches the network; NATIVE switches cuDNN off around the LSTM layer only, then restores it
    assert model.model.use_cudnn
    seen: list[bool] = []
    model.model.lstm.register_forward_pre_hook(lambda _module, _args: seen.append(torch.backends.cudnn.enabled))
    model.model.use_cudnn = False
    model.predict(x_val[:5])
    assert seen == [False]
    assert torch.backends.cudnn.enabled
    assert PyTorchLSTMForecaster(layout=model.layout, output_size=N_HORIZONS, rnn_backend="native").rnn_backend is RnnBackend.NATIVE
    # the head's shape and activation, the embedding's initial scale and the prediction chunk come from the settings
    head = [type(layer) for layer in model.model.head]
    assert head == [torch.nn.Linear, torch.nn.ReLU, torch.nn.Dropout, torch.nn.Linear]
    assert model.model.head[0].out_features == model.hidden_size
    widths, chunk = [8, 4], 7
    other = PyTorchLSTMForecaster(layout=model.layout, output_size=N_HORIZONS, head_hidden_sizes=widths, head_activation="tanh", embedding_init_std=0.0, eval_batch_size=chunk)
    other._input_stats, other.input_size, other.device = model._input_stats, model.input_size, model.device
    other.build_model()
    assert [layer.out_features for layer in other.model.head if isinstance(layer, torch.nn.Linear)] == [*widths, N_HORIZONS]
    assert sum(isinstance(layer, torch.nn.Tanh) for layer in other.model.head) == len(widths)
    assert not other.model.embedding.weight.any()  # std 0: the known rows and the unknown row are all zeros
    calls: list[int] = []
    other.model.register_forward_pre_hook(lambda _module, args: calls.append(len(args[0])))
    other.predict(x_val)
    assert max(calls) == chunk
    assert sum(calls) == len(x_val)


def test_input_statistics_come_from_the_fit_rows_only(model: PyTorchLSTMForecaster) -> None:
    x_train, y_train = _panel(300, 3)
    x_val, y_val = _panel(100, 4)
    x_val[:, 1 + N_STEPS * N_CHANNELS :] += 100.0  # a shifted held-out block must not move the fitted statistics
    model.fit(x_train, y_train, x_val=x_val, y_val=y_val)
    assert np.allclose(model._input_stats["dense_mean"], x_train[:, 1 + N_STEPS * N_CHANNELS :].mean(axis=0), atol=1e-5)


def test_per_output_target_scaling_is_fitted_on_the_fit_rows_and_forecasts_return_in_target_units(model: PyTorchLSTMForecaster) -> None:
    assert model.target_scaling is TargetScaling.PER_OUTPUT  # the default
    x_train, y_train = _panel(300, 3)
    y_train = y_train * np.array([5.0, 40.0], dtype=np.float32) + np.array([100.0, -7.0], dtype=np.float32)  # horizons on very different scales
    y_train[::3, 0] = np.nan
    x_val, y_val = _panel(100, 4)
    y_val = y_val * np.array([5.0, 40.0], dtype=np.float32) + np.array([100.0, -7.0], dtype=np.float32)
    seen: list[torch.Tensor] = []
    original = model._target_tensor
    model._target_tensor = lambda y: seen.append(original(y)) or seen[-1]  # the targets the network trains on
    model.fit(x_train, y_train, x_val=x_val, y_val=y_val)
    trained_on = seen[0].numpy()
    assert np.allclose(np.nanmean(trained_on, axis=0), 0.0, atol=1e-5)
    assert np.allclose(np.nanstd(trained_on, axis=0), 1.0, atol=1e-5)
    assert np.array_equal(np.isnan(trained_on), np.isnan(y_train))  # NaN stays NaN
    scaler = model.target_scaler_
    assert np.allclose(scaler.mean_, np.nanmean(y_train, axis=0))  # fit rows only
    assert np.allclose(scaler.scale_, np.nanstd(y_train, axis=0))
    raw = model._predict_tensor(torch.tensor(x_val)).numpy()  # in the network's scale
    np.testing.assert_allclose(model.predict(x_val), raw * scaler.scale_ + scaler.mean_, rtol=1e-5)  # forecasts in target units
    np.testing.assert_allclose(scaler.inverse(scaler.transform(y_val)), y_val, rtol=1e-5, atol=1e-4)  # exact round trip
    assert model._held_out_fit_score(x_val, y_val) == pytest.approx(model.convergence_.best_val, rel=1e-4)  # the held-out score reads scaled targets
    assert model.convergence_.best_val < 1.0  # in standardised units (the squared error of the mean forecast is 1)
    # an output bound is built from the scaled training targets
    bounded = PyTorchLSTMForecaster(**{**model.get_params(), "output_bound": "training_range", "n_epochs": 2, "early_stopping_rounds": None})
    bounded.fit(x_train, y_train)
    assert float(bounded._output_bound.centre.abs().max()) < 2.0  # raw targets centre near 100; scaled ones near 0
    with pytest.raises(ValueError, match="target_scaling=NONE"):
        PyTorchLSTMForecaster(layout=model.layout, output_size=N_HORIZONS, loss="poisson_deviance")  # PER_OUTPUT is the default


def test_target_scaling_none_reproduces_the_unscaled_predictions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Golden values from the code before target scaling existed (automl 95ee1aa, CPU, one thread)."""
    monkeypatch.setenv("AUTOML_DEVICE", "cpu")
    torch.set_num_threads(1)
    x_train, y_train = _panel(150, 1)
    x_val, y_val = _panel(60, 2)
    y_train[::2, 0] = np.nan
    unscaled = PyTorchLSTMForecaster(
        layout=SequenceLayout(N_SERIES, N_STEPS, N_CHANNELS), output_size=N_HORIZONS, window=N_STEPS, hidden_size=16, embedding_dim=4, learning_rate=1e-2,
        batch_size=64, n_epochs=4, early_stopping_rounds=4, refit_on_train_val=False, calculate_feature_importance=False, random_seed=0, target_scaling="none",
    )
    unscaled.fit(x_train, y_train, x_val=x_val, y_val=y_val)
    pred = unscaled.predict(x_val)
    assert unscaled.target_scaler_ is None
    np.testing.assert_allclose(pred[:2], [[-0.15867209, 0.90455155], [-0.12159690, 0.40740809]], rtol=1e-4)
    assert float(pred.sum()) == pytest.approx(30.399474, rel=1e-4)


def test_poisson_deviance_formula_and_offset_reach_the_loss(model: PyTorchLSTMForecaster) -> None:
    # hand example: count u = 2, mean mu = 1 -> 2 * (2 log 2 - (2 - 1)) = 4 log 2 - 2; zero when mu == u; numpy and torch agree
    expected = 4 * math.log(2) - 2
    assert poisson_deviance(np.array([math.log(2)]), np.array([0.0]))[0] == pytest.approx(expected)
    assert poisson_deviance(torch.tensor([math.log(2)]), torch.tensor([0.0])).item() == pytest.approx(expected, rel=1e-6)
    assert poisson_deviance(np.array([1.3]), np.array([1.3]))[0] == pytest.approx(0.0, abs=1e-12)
    # the loss: z = log-count minus offset, so the deviance sees exp(z + o) against exp(prediction + o); NaN entries are skipped
    nan = float("nan")
    offset = torch.tensor([[math.log(2)], [math.log(5)]])
    target = torch.tensor([[0.0, nan], [0.0, 0.0]])
    loss = PyTorchLSTMForecaster(layout=model.layout, output_size=N_HORIZONS, loss="poisson_deviance", target_scaling="none")
    assert loss.loss is ForecastLoss.POISSON_DEVIANCE
    got = loss._fit_loss(torch.zeros(2, 2), target, offset=offset)
    assert got.item() == pytest.approx(0.0, abs=1e-6)  # prediction == target: deviance 0
    pred = torch.tensor([[-math.log(2), 7.0], [0.0, 0.0]])  # row 0: mu = exp(-log 2 + log 2) = 1 against u = 2
    assert loss._fit_loss(pred, target, offset=offset).item() == pytest.approx(expected / 3)  # one non-zero entry of three observed
    assert loss._fit_loss(pred, target, offset=offset, reduction="none").tolist() == pytest.approx([expected, 0.0])
    # a bigger row weighs more: the same relative miss costs in proportion to the count
    miss = torch.full((2, 1), 0.5)
    sizes = loss._fit_loss(miss, torch.zeros(2, 1), offset=offset, reduction="none")
    assert sizes[1] / sizes[0] == pytest.approx(5 / 2, rel=1e-5)
    # squared error ignores the offset
    assert model._fit_loss(pred, target, offset=offset).item() == model._fit_loss(pred, target).item()


def test_offset_column_is_not_an_input_and_poisson_model_trains(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTOML_DEVICE", "cpu")
    layout = SequenceLayout(N_SERIES, N_STEPS, N_CHANNELS, offset_dense_index=1)
    x, y = _panel(400, 5)
    x[:, 1 + N_STEPS * N_CHANNELS + 1] = np.log(np.random.default_rng(0).integers(1, 20, len(x)))  # per-row offset in dense column 1
    y = np.abs(y) * 0.1
    common = {
        "layout": layout,
        "output_size": N_HORIZONS,
        "window": N_STEPS,
        "hidden_size": 8,
        "embedding_dim": 4,
        "n_epochs": 3,
        "batch_size": 64,
        "refit_on_train_val": False,
        "calculate_feature_importance": False,
        "random_seed": 0,
    }
    poisson = PyTorchLSTMForecaster(**common, loss=ForecastLoss.POISSON_DEVIANCE, target_scaling=TargetScaling.NONE, early_stopping_rounds=3)
    poisson.fit(x[:300], y[:300], x_val=x[300:], y_val=y[300:])
    assert poisson._input_stats["dense_mean"].shape == (N_DENSE - 1,)  # the offset column is not standardised or fed in
    changed = x[300:].copy()
    changed[:, 1 + N_STEPS * N_CHANNELS + 1] += 3.0
    np.testing.assert_array_equal(poisson.predict(changed), poisson.predict(x[300:]))  # predictions do not read the offset
    held_out = poisson._held_out_fit_score(x[300:], y[300:])
    assert held_out == pytest.approx(poisson.convergence_.best_val, rel=1e-4)  # early stopping read the deviance with the offset
    assert held_out != pytest.approx(poisson._held_out_fit_score(changed, y[300:]))  # ... which does depend on the offset
    with pytest.raises(ValueError, match="offset"):
        poisson._held_out_fit_score(np.where(np.arange(x.shape[1]) == 1 + N_STEPS * N_CHANNELS + 1, np.nan, x[300:]).astype(np.float32), y[300:])
    with pytest.raises(ValueError, match="dense block"):
        PyTorchLSTMForecaster(**{**common, "layout": SequenceLayout(N_SERIES, N_STEPS, N_CHANNELS, offset_dense_index=N_DENSE)}).fit(x[:300], y[:300])


def test_output_bound_confines_the_output_and_is_saved(model: PyTorchLSTMForecaster) -> None:
    x, y = _panel(300, 6)
    bounded = PyTorchLSTMForecaster(**{**model.get_params(), "output_bound": "training_range", "n_epochs": 3, "early_stopping_rounds": None})
    assert bounded.output_bound is OutputBound.TRAINING_RANGE
    y = y.copy()
    y[x[:, 0] == 0] = np.nan  # series 0 has rows but no observed target: its vector never trains, so it counts as unseen
    bounded.fit(x, y)
    assert bounded.model.embedding.seen.tolist() == [False] + [True] * (N_SERIES - 1)
    lo, hi = np.nanmin(y, axis=0), np.nanmax(y, axis=0)
    wild = x.copy()
    wild[:, 1:] = wild[:, 1:] * 1e4  # a runaway input must not give a runaway output
    pred = bounded.predict(wild)
    assert (pred >= lo).all()
    assert (pred <= hi).all()
    assert {"output_bound.centre", "output_bound.half_width"} <= set(bounded.model.state_dict())
    assert bounded.get_params()["output_bound"] is OutputBound.TRAINING_RANGE  # clones keep it
    # the bound is a midpoint-centred tanh: identity at the midpoint, the ends at +-infinity
    bound = TrainingRangeBound(np.array([0.0]), np.array([4.0]))
    assert bound(torch.tensor([[2.0], [100.0], [-100.0]])).flatten().tolist() == pytest.approx([2.0, 4.0, 0.0], abs=1e-4)
    with pytest.raises(ValueError, match=r"column\(s\) \[1\].*zero-width"):
        TrainingRangeBound.from_targets(np.array([[1.0, 2.0], [3.0, 2.0], [2.0, np.nan]]))
    with pytest.raises(NotImplementedError, match="PyTorchLinearRegression"):
        PyTorchLinearRegression(output_bound=OutputBound.TRAINING_RANGE)


def test_plain_network_output_bound_confines_extreme_inputs() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 3)).astype(np.float32)
    y = (2.0 * x[:, 0] + rng.normal(scale=0.1, size=400)).astype(np.float32)
    # no target scaling: its rounding clip in `predict` would hide whether the network itself is bounded
    common = {"n_epochs": 20, "early_stopping_rounds": None, "random_seed": 0, "batch_size": 64, "learning_rate": 1e-2, "target_scaling": TargetScaling.NONE}
    wild = x * 1e3  # far outside the training inputs
    free = PyTorchNeuralNetwork(**common)
    free.fit(x, y)
    bounded = PyTorchNeuralNetwork(**common, output_bound=OutputBound.TRAINING_RANGE)
    bounded.fit(x, y)
    assert np.abs(np.ravel(free.predict(wild))).max() > np.abs(y).max()  # unbounded, the network extrapolates past the targets
    pred = np.ravel(bounded.predict(wild))
    assert pred.min() >= y.min()
    assert pred.max() <= y.max()
    assert {"output_bound.centre", "output_bound.half_width"} <= set(bounded.model.state_dict())
    with pytest.raises(ValueError, match="regression mean"):
        PyTorchNeuralNetwork(output_bound=OutputBound.TRAINING_RANGE, uncertainty_method=UncertaintyMethod.PROBABILISTIC)


def test_unknown_category_embedding_keeps_the_known_rows_and_maps_unseen_ones() -> None:
    seen = torch.tensor([True, True, False])
    torch.manual_seed(0)
    plain = torch.nn.Embedding(3, 4)
    torch.nn.init.normal_(plain.weight, std=0.1)
    after_plain = torch.rand(1)
    torch.manual_seed(0)
    table = UnknownCategoryEmbedding(3, 4, 0.1, seen)
    assert torch.equal(table.weight[:3], plain.weight)  # the known rows are bit-identical to a plain embedding
    assert torch.rand(1) == after_plain  # the unknown row consumed no random draw
    assert not table.weight[3].any()
    assert table(torch.tensor([2, 0])).tolist() == [[0.0] * 4, plain.weight[0].tolist()]  # never-seen index 2 reads the unknown row, a seen one its own
    assert table.n_unseen(torch.tensor([2, 0, 2])) == 2
    state = torch.get_rng_state()
    table.train()
    table(torch.arange(2))
    assert torch.equal(torch.get_rng_state(), state)  # rate 0 draws nothing
    hidden = UnknownCategoryEmbedding(3, 4, 0.1, torch.ones(3, dtype=torch.bool), unknown_rate=0.5)
    hidden.embedding.weight.data[3] = 1.0  # mark the unknown row
    hidden.train()
    out = hidden(torch.zeros(2000, dtype=torch.long))
    assert (out[:, 0] == 1.0).float().mean().item() == pytest.approx(0.5, abs=0.05)  # about half the rows are hidden in training
    hidden.eval()
    assert not (hidden(torch.zeros(50, dtype=torch.long))[:, 0] == 1.0).any()  # nothing is hidden at prediction
