"""Numbers the weekly dependency upgrade compares between the old and the new virtual environment.

Run as ``<venv>/bin/python scripts/upgrade_probe.py --out FILE`` from the project root. It fits, on the CPU with one
thread and fixed seeds, (1) a ProbabilisticRegressionModel on toy dataset A and (2) a PyTorchLSTMForecaster on a small
synthetic panel, and writes the predictions and fit scores as JSON. Two environments that agree on the numbers agree
on the fits; any difference is reported by the upgrade check, which then holds the new lock.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ["AUTOML_DEVICE"] = "cpu"  # set before the package reads it; the probe never touches an accelerator

import numpy as np
import torch

from automl_package.examples._toy_datasets import make_toy_a
from automl_package.models.lstm_model import PyTorchLSTMForecaster, SequenceLayout
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel

N_THREADS = 1
SEED = 0
# ProbReg on toy A
PROBREG_N_TRAIN = 400
PROBREG_N_TEST = 200
PROBREG_TEST_SEED = 1
PROBREG_EPOCHS = 30
PROBREG_PATIENCE = 10
# LSTM on a synthetic panel
PANEL_N_SERIES = 3
PANEL_N_STEPS = 8
PANEL_N_CHANNELS = 2
PANEL_N_DENSE = 3
PANEL_N_HORIZONS = 2
PANEL_N_TRAIN = 300
PANEL_N_VAL = 100
PANEL_TRAIN_SEED = 1
PANEL_VAL_SEED = 2
LSTM_EPOCHS = 2
LSTM_HIDDEN = 16
LSTM_EMBEDDING = 4
LSTM_BATCH = 64
LSTM_LEARNING_RATE = 1e-2
HORIZON_MEAN_WINDOW = 4  # horizon 0 target: mean of the newest this-many days of channel 0


def _as_list(values: np.ndarray) -> list[float]:
    """Float64 values as a plain list, full precision."""
    return np.asarray(values, dtype=np.float64).ravel().tolist()


def probreg_numbers() -> dict[str, Any]:
    """Fits ProbReg on toy A and returns the test predictions and their mean squared error."""
    x, y = make_toy_a(n=PROBREG_N_TRAIN, seed=SEED)
    x_test, y_test = make_toy_a(n=PROBREG_N_TEST, seed=PROBREG_TEST_SEED)
    model = ProbabilisticRegressionModel(
        input_size=1, n_epochs=PROBREG_EPOCHS, early_stopping_rounds=PROBREG_PATIENCE, random_seed=SEED, calculate_feature_importance=False
    )
    model.fit(x, y)
    pred = np.asarray(model.predict(x_test), dtype=np.float64).ravel()
    return {"test_mse": float(np.mean((pred - np.ravel(y_test)) ** 2)), "test_predictions": _as_list(pred)}


def _panel(n_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows in the forecaster's flat layout; horizon 0 is a window mean, horizon 1 the series offset."""
    rng = np.random.default_rng(seed)
    series = rng.integers(0, PANEL_N_SERIES, n_rows)
    window = rng.normal(size=(n_rows, PANEL_N_STEPS, PANEL_N_CHANNELS)).astype(np.float32)
    dense = rng.normal(size=(n_rows, PANEL_N_DENSE)).astype(np.float32)
    x = np.concatenate([series[:, None].astype(np.float32), window.reshape(n_rows, -1), dense], axis=1)
    y = np.column_stack([window[:, -HORIZON_MEAN_WINDOW:, 0].mean(axis=1), 0.5 * series]).astype(np.float32)
    return x, y


def lstm_numbers() -> dict[str, Any]:
    """Fits the LSTM forecaster for LSTM_EPOCHS epochs and returns its validation predictions and fit score."""
    x_train, y_train = _panel(PANEL_N_TRAIN, PANEL_TRAIN_SEED)
    x_val, y_val = _panel(PANEL_N_VAL, PANEL_VAL_SEED)
    model = PyTorchLSTMForecaster(
        layout=SequenceLayout(PANEL_N_SERIES, PANEL_N_STEPS, PANEL_N_CHANNELS),
        output_size=PANEL_N_HORIZONS,
        window=PANEL_N_STEPS,
        hidden_size=LSTM_HIDDEN,
        embedding_dim=LSTM_EMBEDDING,
        learning_rate=LSTM_LEARNING_RATE,
        batch_size=LSTM_BATCH,
        n_epochs=LSTM_EPOCHS,
        early_stopping_rounds=LSTM_EPOCHS,
        refit_on_train_val=False,
        calculate_feature_importance=False,
        random_seed=SEED,
    )
    model.fit(x_train, y_train, x_val=x_val, y_val=y_val)
    return {"val_fit_score": float(model._held_out_fit_score(x_val, y_val)), "val_predictions": _as_list(model.predict(x_val))}


def main() -> None:
    """Writes the probe numbers to the file named by --out."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="JSON file to write")
    args = parser.parse_args()
    torch.set_num_threads(N_THREADS)
    numbers = {"probreg_toy_a": probreg_numbers(), "lstm_panel": lstm_numbers()}
    args.out.write_text(json.dumps(numbers, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
