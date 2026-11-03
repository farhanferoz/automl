"""PyTorchLSTMForecaster: learns a window-driven signal, skips NaN targets, and standardises on its own fit rows."""

import numpy as np
import pytest
import torch

from automl_package.enums import RnnBackend
from automl_package.models.lstm_model import PyTorchLSTMForecaster, SequenceLayout

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
    assert not other.model.embedding.weight.any()
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
