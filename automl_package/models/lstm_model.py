"""LSTM forecaster for panels of daily series: one network across all series, direct multi-horizon output.

Searched before writing: `automl_package/` has no sequence or recurrent model (grep for LSTM, GRU, RNN found
nothing), so this is a new `PyTorchModelBase` subclass. It reuses the base class's optimiser setup,
regularisation hooks, `ConvergenceTracker` and the best-epoch restore (`snapshot_state`).

Input layout. `fit` and `predict` take a flat 2-D array, so the generic `BaseModel` machinery (partitions,
search, clone) works unchanged. Each row is one forecast origin:

    column 0                          series index, 0 .. n_series - 1 (float, cast to long)
    columns 1 .. 1 + T*C              the trailing window, oldest day first, C channels per day (`SequenceLayout`)
    remaining columns                 dense features at the origin (calendar, covariates, static metadata)

The window length actually used is the hyperparameter `window` (<= T, the newest `window` days). Inputs are
standardised INSIDE the model with statistics taken from the rows it is fitted on (nan-aware; a missing dense
value becomes the training mean), so a search or refit never sees statistics from held-out rows.

Targets are `(N, n_horizons)`; NaN means "no target for this row and horizon" (unobserved day, or purged by the
caller's protocol) and is excluded from the loss. The fit term is the mean squared error over the observed
entries: the model is mean-only, its spread is not fitted (DECISIONS.md, "Spread is never fitted in-sample").
"""

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn as nn

from automl_package.enums import ExplainerType
from automl_package.logger import logger
from automl_package.models.base_pytorch import PyTorchModelBase
from automl_package.utils.convergence import ConvergenceTracker, snapshot_state
from automl_package.utils.pytorch_utils import get_device

_MIN_STD = 1e-6
_EVAL_BATCH_SIZE = 8192


@dataclass(frozen=True)
class SequenceLayout:
    """Column layout of the flat input: series index, then `n_steps` days of `n_channels`, then dense features."""

    n_series: int
    n_steps: int
    n_channels: int

    @property
    def n_sequence_columns(self) -> int:
        """Number of columns the window occupies."""
        return self.n_steps * self.n_channels


def _standardiser(values: np.ndarray, axis: int | tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Nan-aware mean and standard deviation (floored) along `axis`, as float32."""
    mean = np.nan_to_num(np.nanmean(values, axis=axis))
    std = np.nan_to_num(np.nanstd(values, axis=axis))
    return mean.astype(np.float32), np.where(std < _MIN_STD, 1.0, std).astype(np.float32)


class _LSTMForecasterNet(nn.Module):
    """Series embedding + LSTM over the window + MLP head over [last hidden state, embedding, dense features]."""

    def __init__(
        self,
        layout: SequenceLayout,
        window: int,
        n_dense: int,
        n_horizons: int,
        hidden_size: int,
        num_layers: int,
        embedding_dim: int,
        dropout_rate: float,
        stats: dict[str, np.ndarray],
    ) -> None:
        super().__init__()
        self.layout, self.window = layout, window
        self.embedding = nn.Embedding(layout.n_series, embedding_dim)
        nn.init.normal_(self.embedding.weight, std=0.1)
        self.lstm = nn.LSTM(layout.n_channels + embedding_dim, hidden_size, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + embedding_dim + n_dense, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, n_horizons),
        )
        for name, value in stats.items():
            self.register_buffer(name, torch.from_numpy(value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`(B, D)` flat rows to `(B, n_horizons)` predictions."""
        n_seq = self.layout.n_sequence_columns
        series = x[:, 0].long()
        window = x[:, 1 : 1 + n_seq].reshape(len(x), self.layout.n_steps, self.layout.n_channels)[:, -self.window :]
        dense = x[:, 1 + n_seq :]
        window = torch.nan_to_num((window - self.seq_mean) / self.seq_std)
        dense = torch.nan_to_num((dense - self.dense_mean) / self.dense_std)
        emb = self.embedding(series)
        steps = torch.cat([window, emb.unsqueeze(1).expand(-1, window.shape[1], -1)], dim=-1)
        out, _ = self.lstm(steps)
        return self.head(torch.cat([out[:, -1], emb, dense], dim=-1))


class PyTorchLSTMForecaster(PyTorchModelBase):
    """One LSTM across all series of a panel, with a learned embedding per series and several forecast horizons at once."""

    _scores_by_fit_term: ClassVar[bool] = True

    _defaults: ClassVar[dict[str, Any]] = {
        "layout": None,
        "window": 56,
        "hidden_size": 64,
        "num_layers": 1,
        "embedding_dim": 8,
        "grad_clip_norm": 1.0,
        "batch_size": 1024,
        "n_epochs": 100,
    }

    def __init__(self, **kwargs: Any) -> None:
        """Initialises the forecaster. `layout` (a `SequenceLayout`) and `output_size` (the number of horizons) are required."""
        for key, value in PyTorchLSTMForecaster._defaults.items():
            kwargs.setdefault(key, value)
        if kwargs["layout"] is None:
            raise ValueError("PyTorchLSTMForecaster needs a SequenceLayout (layout=...).")
        if kwargs.get("output_size", 1) < 1:
            raise ValueError("output_size is the number of forecast horizons and must be at least 1.")
        super().__init__(**kwargs)
        self._input_stats: dict[str, np.ndarray] | None = None

    @property
    def name(self) -> str:
        """Returns the name of the model."""
        return "PyTorchLSTMForecaster"

    def _compute_input_stats(self, x_train: np.ndarray) -> dict[str, np.ndarray]:
        """Standardisation statistics of the rows being fitted: per channel over the window, per column for the dense block."""
        n_seq = self.layout.n_sequence_columns
        seq = x_train[:, 1 : 1 + n_seq].reshape(len(x_train), self.layout.n_steps, self.layout.n_channels)
        seq_mean, seq_std = _standardiser(seq, axis=(0, 1))
        dense_mean, dense_std = _standardiser(x_train[:, 1 + n_seq :], axis=0)
        return {"seq_mean": seq_mean, "seq_std": seq_std, "dense_mean": dense_mean, "dense_std": dense_std}

    def build_model(self) -> None:
        """Builds the network; `_fit_single` sets `self._input_stats` first."""
        if self._input_stats is None:
            raise RuntimeError("Input statistics are missing: build_model is called from _fit_single after they are computed.")
        window = min(int(self.window), self.layout.n_steps)
        n_dense = self.input_size - 1 - self.layout.n_sequence_columns
        self.model = _LSTMForecasterNet(
            layout=self.layout,
            window=window,
            n_dense=n_dense,
            n_horizons=self.output_size,
            hidden_size=int(self.hidden_size),
            num_layers=int(self.num_layers),
            embedding_dim=int(self.embedding_dim),
            dropout_rate=self.dropout_rate,
            stats=self._input_stats,
        ).to(self.device)
        self.criterion = self._fit_loss

    def _fit_loss(self, prediction: torch.Tensor, y: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        """Fit term: squared error over the observed `(row, horizon)` entries of `y` (NaN entries are skipped)."""
        observed = ~torch.isnan(y)
        squared = torch.where(observed, prediction - torch.nan_to_num(y), torch.zeros_like(prediction)) ** 2
        if reduction == "none":
            return squared.sum(dim=1) / observed.sum(dim=1).clamp(min=1)
        return squared.sum() / observed.sum().clamp(min=1)

    def _target_tensor(self, y: np.ndarray) -> torch.Tensor:
        """Targets as a float `(N, n_horizons)` tensor on the model's device."""
        return torch.tensor(np.asarray(y, dtype=np.float32)).to(self.device)

    def _predict_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluation-mode predictions for all rows of `x`, in chunks so large folds fit on the device."""
        self.model.eval()
        with torch.no_grad():
            return torch.cat([self.model(x[i : i + _EVAL_BATCH_SIZE]) for i in range(0, len(x), _EVAL_BATCH_SIZE)])

    def _held_out_fit_score(self, x_val: np.ndarray, y_val: np.ndarray) -> float:
        """Mean held-out fit term (lower is better), the quantity early stopping reads."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        x_tensor = torch.tensor(np.asarray(x_val, dtype=np.float32)).to(self.device)
        return float(self._fit_loss(self._predict_tensor(x_tensor), self._target_tensor(y_val)).item())

    def _fit_single(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        forced_iterations: int | None = None,
        forward_pass_kwargs: dict | None = None,  # noqa: ARG002
    ) -> tuple[int, list[float]]:
        """Fits one network by mini-batch Adam; with a validation set, stops on its fit term and restores the best epoch.

        Returns the number of epochs up to and including the best one, and the validation fit-term history.
        """
        x_train = np.asarray(x_train, dtype=np.float32)
        self.input_size = x_train.shape[1]
        self.device = get_device()
        logger.info(f"Using device: {self.device}")
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)
        self._input_stats = self._compute_input_stats(x_train)
        self.build_model()
        self._setup_regularization_parameters()
        self._setup_optimizers(self.model)

        x_t = torch.tensor(x_train, device=self.device)
        y_t = self._target_tensor(y_train)
        use_early_stopping = self.early_stopping_rounds is not None and forced_iterations is None and x_val is not None
        x_v = torch.tensor(np.asarray(x_val, dtype=np.float32)).to(self.device) if use_early_stopping else None
        y_v = self._target_tensor(y_val) if use_early_stopping else None

        tracker = ConvergenceTracker(patience=self.early_stopping_rounds or 0, min_delta=0.0)
        best_state, history = None, []
        n_epochs = int(forced_iterations if forced_iterations is not None else self.n_epochs)
        final_epoch = 0
        for epoch in range(n_epochs):
            final_epoch = epoch
            self.model.train()
            order = torch.randperm(len(x_t), device=self.device)
            for start in range(0, len(order), self.batch_size):
                batch = order[start : start + self.batch_size]
                self.optimizer.zero_grad()
                loss = self._calculate_regularization_loss(self._fit_loss(self.model(x_t[batch]), y_t[batch]), self.model)
                loss.backward()
                if self.grad_clip_norm:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()
            if use_early_stopping:
                history.append(float(self._fit_loss(self._predict_tensor(x_v), y_v).item()))
                if tracker.update(epoch, history[-1]):
                    best_state = snapshot_state(self.model)
                if tracker.done:
                    break
        if use_early_stopping:
            best_epoch = tracker.best_epoch
            logger.info(f"Best epoch {best_epoch + 1} of {final_epoch + 1} run")
        else:
            best_epoch = final_epoch
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.convergence_ = tracker.result(final_epoch=final_epoch)
        return best_epoch + 1, history

    def predict(self, x: np.ndarray, filter_data: bool = True) -> np.ndarray:  # noqa: ARG002
        """Predictions of shape `(N, n_horizons)`, in the units of the training targets."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        x_tensor = torch.tensor(np.asarray(x, dtype=np.float32)).to(self.device)
        return self._predict_tensor(x_tensor).cpu().numpy()

    def predict_uncertainty(self, x: np.ndarray, filter_data: bool = True) -> np.ndarray:
        """Not available: the model is mean-only and its spread is not fitted in-sample."""
        raise NotImplementedError("PyTorchLSTMForecaster predicts the mean only; the spread is fixed by the study, never fitted.")

    def get_hyperparameter_search_space(self) -> dict[str, Any]:
        """Search space: optimiser, size, window and regularisation of the network; `search_space_override` replaces entries."""
        space = {
            "learning_rate": {"type": "float", "low": 3e-4, "high": 1e-2, "log": True},
            "hidden_size": {"type": "int", "low": 32, "high": 256, "step": 32},
            "num_layers": {"type": "int", "low": 1, "high": 2},
            "embedding_dim": {"type": "int", "low": 4, "high": 32, "step": 4},
            "window": {"type": "int", "low": 14, "high": self.layout.n_steps, "step": 14},
            "dropout_rate": {"type": "float", "low": 0.0, "high": 0.3, "step": 0.1},
        }
        space.update(self.search_space_override)
        return space

    def get_params(self) -> dict[str, Any]:
        """Gets the parameters of the model."""
        params = super().get_params()
        for key in self._defaults:
            params[key] = getattr(self, key)
        return params

    def get_shap_explainer_info(self) -> dict[str, Any]:
        """SHAP is not supported for the windowed input; the explainer entry exists for interface completeness."""
        return {"explainer_type": ExplainerType.DEEP, "model": self.get_internal_model()}
