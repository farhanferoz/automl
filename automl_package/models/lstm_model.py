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

Options (each defaults to the original behaviour, bit for bit, except `target_scaling`, a `BaseModel` setting):

- `loss=ForecastLoss.POISSON_DEVIANCE`: the fit term becomes the Poisson deviance of the count `exp(target + o)` against `exp(output + o)`, with
  `o` a per-row offset (log scale) read from the dense column `SequenceLayout.offset_dense_index`. That column is NOT a network input. The
  prediction stays in target space; the caller adds the offset back as it does today.
- `output_bound=OutputBound.TRAINING_RANGE`: the network output passes through `TrainingRangeBound` (range of the fitted rows' targets, per horizon).
- `target_scaling` (`BaseModel`; default `PER_OUTPUT` for this model): each horizon's targets are standardised by the mean and standard deviation of the
  training part before any fit, so the loss, early stopping, the held-out fit term and the output bound all read the scaled targets, and `predict`
  returns forecasts in target units. It cannot be combined with the Poisson deviance, which reads the counts (`_target_scaling_conflict`).
- `unknown_category_rate`: share of training rows whose series embedding is hidden (replaced by the "unknown" row), so a series never seen in
  training gets a trained fallback vector. Series without a training row that has an observed target are always looked up as unknown at prediction.
"""

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn as nn

from automl_package.enums import ActivationFunction, ExplainerType, ForecastLoss, MissingInputs, OutputBound, RnnBackend
from automl_package.logger import logger
from automl_package.models.base_pytorch import PyTorchModelBase
from automl_package.models.common.output_bound import TrainingRangeBound
from automl_package.models.common.unknown_category_embedding import UnknownCategoryEmbedding
from automl_package.utils.convergence import ConvergenceTracker, snapshot_state
from automl_package.utils.data_handler import standardiser_stats
from automl_package.utils.losses import poisson_deviance
from automl_package.utils.pytorch_utils import cudnn_enabled, get_activation_function_map, get_device


@dataclass(frozen=True)
class SequenceLayout:
    """Column layout of the flat input: series index, then `n_steps` days of `n_channels`, then dense features.

    `offset_dense_index` (default None) names one dense column, by its position within the dense block, that holds the per-row offset of
    `ForecastLoss.POISSON_DEVIANCE` (log scale). It is read by the loss only and left out of the network's inputs and of the standardisation.
    """

    n_series: int
    n_steps: int
    n_channels: int
    offset_dense_index: int | None = None

    @property
    def n_sequence_columns(self) -> int:
        """Number of columns the window occupies."""
        return self.n_steps * self.n_channels

    @property
    def dense_start(self) -> int:
        """Column where the dense block starts."""
        return 1 + self.n_sequence_columns

    def network_dense_columns(self, n_dense_total: int) -> list[int] | None:
        """Positions (within the dense block of `n_dense_total` columns) the network reads; None when every dense column is an input."""
        if self.offset_dense_index is None:
            return None
        if not 0 <= self.offset_dense_index < n_dense_total:
            raise ValueError(f"offset_dense_index {self.offset_dense_index} is outside the dense block of {n_dense_total} columns.")
        return [i for i in range(n_dense_total) if i != self.offset_dense_index]


class _LSTMForecasterNet(nn.Module):
    """Series embedding + LSTM over the window + MLP head over [last hidden state, embedding, dense features].

    The head is `Linear -> activation -> Dropout` per entry of `head_hidden_sizes`, then a `Linear` to the horizons.
    """

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
        use_cudnn: bool,
        embedding_init_std: float,
        head_hidden_sizes: list[int],
        head_activation: type[nn.Module],
        seen_series: np.ndarray,
        unknown_category_rate: float,
        dense_columns: list[int] | None,
        output_bound: TrainingRangeBound | None,
    ) -> None:
        super().__init__()
        self.layout, self.window, self.use_cudnn, self.dense_columns = layout, window, use_cudnn, dense_columns
        self.embedding = UnknownCategoryEmbedding(layout.n_series, embedding_dim, embedding_init_std, torch.from_numpy(seen_series), unknown_category_rate)
        self.output_bound = output_bound
        self.lstm = nn.LSTM(layout.n_channels + embedding_dim, hidden_size, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0.0)
        layers: list[nn.Module] = []
        width = hidden_size + embedding_dim + n_dense
        for next_width in head_hidden_sizes:
            layers += [nn.Linear(width, next_width), head_activation(), nn.Dropout(dropout_rate)]
            width = next_width
        self.head = nn.Sequential(*layers, nn.Linear(width, n_horizons))
        for name, value in stats.items():
            self.register_buffer(name, torch.from_numpy(value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`(B, D)` flat rows to `(B, n_horizons)` predictions."""
        n_seq = self.layout.n_sequence_columns
        series = x[:, 0].long()
        window = x[:, 1 : 1 + n_seq].reshape(len(x), self.layout.n_steps, self.layout.n_channels)[:, -self.window :]
        dense = x[:, 1 + n_seq :]
        if self.dense_columns is not None:
            dense = dense[:, self.dense_columns]  # the offset column is not an input
        window = torch.nan_to_num((window - self.seq_mean) / self.seq_std)
        dense = torch.nan_to_num((dense - self.dense_mean) / self.dense_std)
        emb = self.embedding(series)
        steps = torch.cat([window, emb.unsqueeze(1).expand(-1, window.shape[1], -1)], dim=-1)
        with cudnn_enabled(self.use_cudnn):
            out, _ = self.lstm(steps)
        raw = self.head(torch.cat([out[:, -1], emb, dense], dim=-1))
        return raw if self.output_bound is None else self.output_bound(raw)


class PyTorchLSTMForecaster(PyTorchModelBase):
    """One LSTM across all series of a panel, with a learned embedding per series and several forecast horizons at once."""

    _scores_by_fit_term: ClassVar[bool] = True
    _supports_output_bound: ClassVar[bool] = True  # the head's output is the prediction
    _supports_input_scaling: ClassVar[bool] = False  # it standardises its own sequence layout (`_compute_input_stats`); column 0 is a series index
    _default_missing_inputs: ClassVar[MissingInputs] = MissingInputs.NATIVE  # missing dense values become the training mean inside the model

    _defaults: ClassVar[dict[str, Any]] = {
        "layout": None,
        "window": 56,
        "hidden_size": 64,
        "num_layers": 1,
        "embedding_dim": 8,
        "grad_clip_norm": 1.0,
        "batch_size": 1024,
        "n_epochs": 100,
        "rnn_backend": RnnBackend.CUDNN,  # kernels of the LSTM layer on CUDA (enums.RnnBackend: cuDNN is very slow at some sizes on the GB10)
        "embedding_init_std": 0.1,  # standard deviation of the normal draw that initialises the series embedding
        "head_hidden_sizes": None,  # widths of the head's hidden layers; None: one layer as wide as the LSTM (hidden_size); []: linear head
        "head_activation": ActivationFunction.RELU,
        "eval_batch_size": 8192,  # rows per chunk when predicting; memory only, the predictions do not depend on it
        "loss": ForecastLoss.SQUARED_ERROR,  # the fit term (enums.ForecastLoss); POISSON_DEVIANCE reads the offset column of the layout
        "unknown_category_rate": 0.0,  # share of training rows whose series embedding is replaced by the unknown row (0: nothing is drawn)
    }

    def __init__(self, **kwargs: Any) -> None:
        """Initialises the forecaster. `layout` (a `SequenceLayout`) and `output_size` (the number of horizons) are required."""
        for key, value in PyTorchLSTMForecaster._defaults.items():
            kwargs.setdefault(key, value)
        kwargs["rnn_backend"] = RnnBackend(kwargs["rnn_backend"])  # a config may hand over the value ("native") or the member
        kwargs["head_activation"] = ActivationFunction(kwargs["head_activation"])
        kwargs["loss"] = ForecastLoss(kwargs["loss"])
        if kwargs["layout"] is None:
            raise ValueError("PyTorchLSTMForecaster needs a SequenceLayout (layout=...).")
        if kwargs.get("output_size", 1) < 1:
            raise ValueError("output_size is the number of forecast horizons and must be at least 1.")
        if not 0.0 <= kwargs["unknown_category_rate"] < 1.0:
            raise ValueError(f"unknown_category_rate must be in [0, 1), got {kwargs['unknown_category_rate']}.")
        super().__init__(**kwargs)
        self._input_stats: dict[str, np.ndarray] | None = None
        self._seen_series: np.ndarray | None = None  # bool (n_series,): which series have fit rows with an observed target; set by `_fit_single`
        self._output_bound: TrainingRangeBound | None = None  # set by `_fit_single` under OutputBound.TRAINING_RANGE

    @property
    def name(self) -> str:
        """Returns the name of the model."""
        return "PyTorchLSTMForecaster"

    def _target_scaling_conflict(self) -> str | None:
        """The Poisson deviance reads the counts exp(target + offset), which a rescaled target no longer is."""
        return "loss=POISSON_DEVIANCE (the deviance reads the counts)" if self.loss is ForecastLoss.POISSON_DEVIANCE else None

    def _compute_input_stats(self, x_train: np.ndarray) -> dict[str, np.ndarray]:
        """Standardisation statistics of the rows being fitted: per channel over the window, per column for the dense block."""
        n_seq = self.layout.n_sequence_columns
        seq = x_train[:, 1 : 1 + n_seq].reshape(len(x_train), self.layout.n_steps, self.layout.n_channels)
        seq_mean, seq_std = standardiser_stats(seq, axis=(0, 1))
        dense = x_train[:, 1 + n_seq :]
        columns = self.layout.network_dense_columns(dense.shape[1])
        dense_mean, dense_std = standardiser_stats(dense if columns is None else dense[:, columns], axis=0)
        return {"seq_mean": seq_mean, "seq_std": seq_std, "dense_mean": dense_mean, "dense_std": dense_std}

    def build_model(self) -> None:
        """Builds the network; `_fit_single` sets `self._input_stats` first."""
        if self._input_stats is None:
            raise RuntimeError("Input statistics are missing: build_model is called from _fit_single after they are computed.")
        window = min(int(self.window), self.layout.n_steps)
        n_dense_total = self.input_size - self.layout.dense_start
        dense_columns = self.layout.network_dense_columns(n_dense_total)
        n_dense = n_dense_total if dense_columns is None else len(dense_columns)
        seen_series = np.ones(self.layout.n_series, dtype=bool) if self._seen_series is None else self._seen_series
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
            use_cudnn=self.rnn_backend is RnnBackend.CUDNN,
            embedding_init_std=float(self.embedding_init_std),
            head_hidden_sizes=[int(self.hidden_size)] if self.head_hidden_sizes is None else [int(w) for w in self.head_hidden_sizes],
            head_activation=get_activation_function_map()[self.head_activation],
            seen_series=seen_series,
            unknown_category_rate=float(self.unknown_category_rate),
            dense_columns=dense_columns,
            output_bound=self._output_bound if self.output_bound is OutputBound.TRAINING_RANGE else None,
        ).to(self.device)
        self.criterion = self._fit_loss

    def _fit_loss(self, prediction: torch.Tensor, y: torch.Tensor, reduction: str = "mean", offset: torch.Tensor | None = None) -> torch.Tensor:
        """Fit term over the observed `(row, horizon)` entries of `y` (NaN entries are skipped), per `self.loss`.

        `SQUARED_ERROR`: mean of `(prediction - y)^2`; `offset` is ignored. `POISSON_DEVIANCE`: mean of the Poisson deviance of the count
        `exp(y + offset)` against `exp(prediction + offset)`, so a row weighs in proportion to its size; `offset` is `(N, 1)` on the log
        scale (see `_row_offset`), None meaning 0.
        """
        observed = ~torch.isnan(y)
        prediction = torch.where(observed, prediction, torch.zeros_like(prediction))  # unobserved entries must not reach exp() or the gradient
        target = torch.nan_to_num(y)
        match self.loss:
            case ForecastLoss.SQUARED_ERROR:
                entry = (prediction - target) ** 2
            case ForecastLoss.POISSON_DEVIANCE:
                shift = 0.0 if offset is None else offset
                entry = torch.where(observed, poisson_deviance(target + shift, prediction + shift), torch.zeros_like(prediction))
        if reduction == "none":
            return entry.sum(dim=1) / observed.sum(dim=1).clamp(min=1)
        return entry.sum() / observed.sum().clamp(min=1)

    def _row_offset(self, x: torch.Tensor | np.ndarray) -> torch.Tensor | None:
        """The per-row offset `(N, 1)` the Poisson loss needs: the layout's offset column of `x`; None when the loss ignores it or the layout has none."""
        if self.loss is not ForecastLoss.POISSON_DEVIANCE or self.layout.offset_dense_index is None:
            return None
        column = self.layout.dense_start + self.layout.offset_dense_index
        offset = x[:, column : column + 1]
        return torch.tensor(offset, dtype=torch.float32, device=self.device) if isinstance(offset, np.ndarray) else offset

    def _check_offset(self, x: torch.Tensor) -> None:
        """Raises ValueError when the rows' offset (see `_row_offset`) holds NaN; called once per data set, not per batch."""
        offset = self._row_offset(x)
        if offset is not None and torch.isnan(offset).any():
            raise ValueError("The offset column holds NaN: every row needs an offset under ForecastLoss.POISSON_DEVIANCE.")

    def _target_tensor(self, y: np.ndarray) -> torch.Tensor:
        """Targets (in the space `_fit_single` was given: scaled under `target_scaling`) as a float `(N, n_horizons)` tensor on the model's device."""
        return torch.tensor(np.asarray(y, dtype=np.float32)).to(self.device)

    def _predict_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluation-mode predictions for all rows of `x`, in chunks so large folds fit on the device."""
        self.model.eval()
        with torch.no_grad():
            step = int(self.eval_batch_size)
            return torch.cat([self.model(x[i : i + step]) for i in range(0, len(x), step)])

    def _held_out_fit_score(self, x_val: np.ndarray, y_val: np.ndarray) -> float:
        """Mean held-out fit term (lower is better), the quantity early stopping reads."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        x_tensor = torch.tensor(np.asarray(x_val, dtype=np.float32)).to(self.device)
        self._check_offset(x_tensor)
        return float(self._fit_loss(self._predict_tensor(x_tensor), self._target_tensor(self._to_fit_space(y_val)), offset=self._row_offset(x_tensor)).item())

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
        series = x_train[:, 0].astype(np.int64)
        if series.min() < 0 or series.max() >= self.layout.n_series:
            raise ValueError(f"Series indices must be in 0..{self.layout.n_series - 1}; the fit rows hold {series.min()}..{series.max()}.")
        trained = ~np.isnan(np.asarray(y_train, dtype=float).reshape(len(series), -1)).all(axis=1)  # a row with no observed target never trains its series' vector
        self._seen_series = np.isin(np.arange(self.layout.n_series), series[trained])
        self._output_bound = TrainingRangeBound.from_targets(y_train) if self.output_bound is OutputBound.TRAINING_RANGE else None
        self.build_model()
        self._setup_regularization_parameters()
        self._setup_optimizers(self.model)

        x_t = torch.tensor(x_train, device=self.device)
        self._check_offset(x_t)
        y_t = self._target_tensor(y_train)
        use_early_stopping = self.early_stopping_rounds is not None and forced_iterations is None and x_val is not None
        x_v = torch.tensor(np.asarray(x_val, dtype=np.float32)).to(self.device) if use_early_stopping else None
        y_v = self._target_tensor(y_val) if use_early_stopping else None
        if use_early_stopping:
            self._check_offset(x_v)

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
                x_batch = x_t[batch]
                loss = self._calculate_regularization_loss(self._fit_loss(self.model(x_batch), y_t[batch], offset=self._row_offset(x_batch)), self.model)
                loss.backward()
                if self.grad_clip_norm:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()
            if use_early_stopping:
                history.append(float(self._fit_loss(self._predict_tensor(x_v), y_v, offset=self._row_offset(x_v)).item()))
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

    def _predict(self, x: np.ndarray, filter_data: bool = True) -> np.ndarray:  # noqa: ARG002
        """Predictions of shape `(N, n_horizons)`, in the units of the training targets."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        x_tensor = torch.tensor(np.asarray(x, dtype=np.float32)).to(self.device)
        n_unseen = self.model.embedding.n_unseen(x_tensor[:, 0].long())
        logger.info(f"{n_unseen} of {len(x_tensor)} rows are of a series without training rows and use the unknown embedding")
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
