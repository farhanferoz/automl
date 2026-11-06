"""Data handling utilities."""

from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from automl_package.enums import DataSplitStrategy, InputScaling, MissingInputs

if TYPE_CHECKING:
    from automl_package.models.base import BaseModel

MIN_STD = 1e-6  # a standard deviation below this is treated as 0 and replaced by 1
MATRIX_NDIM = 2  # a (rows, columns) array


def standardiser_stats(values: np.ndarray, axis: int | tuple[int, ...] | None, dtype: type | np.dtype = np.float32) -> tuple[np.ndarray, np.ndarray]:
    """Nan-aware mean and population standard deviation along `axis`; the deviation is floored (below `MIN_STD` it becomes 1). Results are cast to `dtype`."""
    mean = np.nan_to_num(np.nanmean(values, axis=axis))
    std = np.nan_to_num(np.nanstd(values, axis=axis))
    return mean.astype(dtype), np.where(std < MIN_STD, 1.0, std).astype(dtype)


class TargetScaler:
    """Per-output standardisation of targets by the mean and standard deviation of the rows passed to `fit`.

    NaN means "no target" and stays NaN. A 1-D `y` is one output and comes back 1-D. The `transform`/`inverse_transform`/`mean_`/`scale_` names
    mirror sklearn's `StandardScaler`, so a caller written against that works unchanged (those work on `(n, 1)` arrays too).
    """

    def __init__(self) -> None:
        """Creates an unfitted scaler."""
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, y: np.ndarray) -> "TargetScaler":
        """Statistics per output column of `y`; returns self."""
        y = np.asarray(y)
        y = y.astype(np.float64) if not np.issubdtype(y.dtype, np.floating) else y
        self.mean_, self.scale_ = standardiser_stats(y.reshape(len(y), -1), axis=0, dtype=y.dtype)
        return self

    def _fitted(self) -> tuple[np.ndarray, np.ndarray]:
        """Mean and scale, both `(n_outputs,)` and so broadcast along the last axis of a 1-D, `(n, 1)` or `(n, n_outputs)` array."""
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("TargetScaler is not fitted.")
        return self.mean_, self.scale_

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Targets in scaled units (the dtype of `y` when floating)."""
        y = np.asarray(y)
        y = y if np.issubdtype(y.dtype, np.floating) else y.astype(np.float64)
        mean, scale = self._fitted()
        return (y - mean.astype(y.dtype)) / scale.astype(y.dtype)

    def inverse(self, pred: np.ndarray) -> np.ndarray:
        """Scaled values back in target units (float64 for float64 input, else the input's floating dtype)."""
        pred = np.asarray(pred)
        pred = pred if np.issubdtype(pred.dtype, np.floating) else pred.astype(np.float64)
        mean, scale = self._fitted()
        return pred * scale.astype(pred.dtype) + mean.astype(pred.dtype)

    inverse_transform = inverse

    def _spread_scale(self, spread: np.ndarray) -> np.ndarray:
        """The scale a spread of one output (1-D) or of `n_outputs` columns is multiplied by; anything else is not mapped with certainty."""
        _, scale = self._fitted()
        spread = np.asarray(spread)
        if len(scale) == 1 and spread.ndim <= 1:
            return scale[0]
        if spread.ndim == MATRIX_NDIM and spread.shape[1] == len(scale):
            return scale
        raise NotImplementedError(f"A spread of shape {spread.shape} cannot be mapped to {len(scale)} scaled outputs: set target_scaling=NONE.")

    def spread_inverse(self, spread: np.ndarray) -> np.ndarray:
        """A spread (standard deviation) in scaled units to target units."""
        return np.asarray(spread) * self._spread_scale(spread)

    def spread_transform(self, spread: np.ndarray) -> np.ndarray:
        """A spread in target units to scaled units."""
        return np.asarray(spread) / self._spread_scale(spread)

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """`fit(y)` then `transform(y)`."""
        return self.fit(y).transform(y)


class InputTransform:
    """Training-part statistics for the input columns: the missing-value rule and per-column standardisation.

    The settings are `BaseModel.missing_inputs` and `BaseModel.input_scaling`. Statistics are kept per column name, so `transform` takes
    all the fitted columns or any subset of them (e.g. the selected features). Columns that are not standardised: binary 0/1 columns
    of the training part and the names in ``unscaled_columns`` (categorical codes, a
    per-row offset). Under RAISE a NaN raises; under TRAINING_MEAN it becomes the training part's column mean (before scaling); under
    NATIVE it is passed through.
    """

    def __init__(self, scaling: InputScaling, missing: MissingInputs, unscaled_columns: Collection[str] = ()) -> None:
        """Creates an unfitted transform for the given settings."""
        self.scaling, self.missing = scaling, missing
        self.unscaled_columns = frozenset(unscaled_columns)
        self.columns_: dict[str, int] | None = None  # column name -> position in the fitted columns
        self.fill_: np.ndarray | None = None  # per-column training mean (TRAINING_MEAN)
        self.mean_: np.ndarray | None = None  # per-column centre (STANDARD; 0 for a column left alone)
        self.scale_: np.ndarray | None = None  # per-column scale (STANDARD; 1 for a column left alone)

    @property
    def changes_values(self) -> bool:
        """Whether `transform` changes any value (scaling or filling); a RAISE/NATIVE transform without scaling only checks."""
        return self.scaling is InputScaling.STANDARD or self.missing is MissingInputs.TRAINING_MEAN

    def check(self, x: np.ndarray, what: str) -> None:
        """Under RAISE, a NaN in `x` raises ValueError naming the remedy."""
        if self.missing is not MissingInputs.RAISE:
            return
        n_missing = int(np.isnan(np.asarray(x, dtype=np.float64)).sum())
        if n_missing:
            raise ValueError(
                f"{what} hold {n_missing} missing values and missing_inputs=RAISE: clean them, or set missing_inputs=TRAINING_MEAN "
                "(filled with the training mean) or NATIVE (for a model that handles NaN itself)."
            )

    def fit(self, x: np.ndarray, columns: Sequence[str]) -> "InputTransform":
        """Statistics of the training part `x` whose columns are named `columns`; returns self."""
        x = np.asarray(x, dtype=np.float64)
        self.check(x, "The training inputs")
        self.columns_ = {name: j for j, name in enumerate(columns)}
        if self.missing is MissingInputs.TRAINING_MEAN:
            self.fill_ = np.nan_to_num(np.nanmean(x, axis=0))
            x = np.where(np.isnan(x), self.fill_, x)
        if self.scaling is InputScaling.STANDARD:
            mean, scale = standardiser_stats(x, axis=0, dtype=np.float64)
            observed = np.where(np.isnan(x), 0.0, x)
            binary = np.all((observed == 0.0) | (observed == 1.0), axis=0)
            named = np.array([name in self.unscaled_columns for name in columns], dtype=bool)
            left_alone = binary | named
            self.mean_, self.scale_ = np.where(left_alone, 0.0, mean), np.where(left_alone, 1.0, scale)
        return self

    def transform(self, x: np.ndarray, columns: Sequence[str] | None = None) -> np.ndarray:
        """`x` (all fitted columns, or the named subset `columns` in that order) after the missing-value rule and scaling.

        Without a value change (scaling NONE, missing RAISE or NATIVE) `x` is only checked and returned as given.
        """
        self.check(x, "The inputs")
        if not self.changes_values:
            return x
        if self.columns_ is None:
            raise RuntimeError("InputTransform is not fitted.")
        idx = slice(None) if columns is None else np.array([self.columns_[name] for name in columns], dtype=int)
        x = np.asarray(x, dtype=np.float64)
        if self.fill_ is not None:
            x = np.where(np.isnan(x), self.fill_[idx], x)
        if self.mean_ is not None:
            x = (x - self.mean_[idx]) / self.scale_[idx]
        return x


class DataHandler:
    """Handles data scaling and splitting."""

    def __init__(self, scale_x: bool = True, scale_y: bool = True, scale_binary_features: bool = False, log_transform_y: bool = False) -> None:
        """Initializes the DataHandler."""
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.scale_binary_features = scale_binary_features
        self.log_transform_y = log_transform_y
        self.x_scaler = StandardScaler() if scale_x else None
        self.y_scaler = TargetScaler() if scale_y else None
        self.binary_feature_indices_ = []
        self.non_binary_feature_indices_ = []
        self.y_was_log_transformed = False
        self.smearing_factor = None

    def _detect_binary_features(self, x: np.ndarray) -> None:
        if not self.scale_binary_features:
            self.binary_feature_indices_ = [i for i in range(x.shape[1]) if np.all(np.isin(x[:, i], [0, 1]))]
            self.non_binary_feature_indices_ = [i for i in range(x.shape[1]) if i not in self.binary_feature_indices_]
        else:
            self.non_binary_feature_indices_ = list(range(x.shape[1]))

    def fit_transform(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fits and transforms the data."""
        if self.log_transform_y:
            y = np.log1p(y)
            self.y_was_log_transformed = True

        if self.scale_x:
            self._detect_binary_features(x)
            x_scaled = x.copy()
            if self.non_binary_feature_indices_:
                x_scaled[:, self.non_binary_feature_indices_] = self.x_scaler.fit_transform(x[:, self.non_binary_feature_indices_])
        else:
            x_scaled = x

        y_scaled = self.y_scaler.fit_transform(y.reshape(-1, 1)).flatten() if self.scale_y else y
        return x_scaled, y_scaled

    def transform(self, x: np.ndarray, y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        """Transforms the data using the fitted scalers."""
        if self.scale_x:
            x_scaled = x.copy()
            if self.non_binary_feature_indices_:
                x_scaled[:, self.non_binary_feature_indices_] = self.x_scaler.transform(x[:, self.non_binary_feature_indices_])
        else:
            x_scaled = x

        y_scaled = self.y_scaler.transform(y.reshape(-1, 1)).flatten() if self.scale_y and y is not None else y
        return x_scaled, y_scaled

    def inverse_transform_y(self, y_pred: np.ndarray) -> np.ndarray:
        """Inverse transforms the predictions.

        This method assumes y_pred is on the same scale as the transformed y from fit_transform.
        It will inverse scale and inverse log transform if they were applied.
        """
        y_processed = y_pred
        if self.scale_y:
            y_processed = self.y_scaler.inverse_transform(y_processed.reshape(-1, 1)).flatten()

        if self.y_was_log_transformed:
            y_processed = np.expm1(y_processed)
            if self.smearing_factor is not None:
                y_processed *= self.smearing_factor

        return y_processed

    def fit_smearing_correction(
        self,
        y_train_log: np.ndarray,
        model: Optional["BaseModel"] = None,
        x_train: np.ndarray | None = None,
        y_train_pred_log: np.ndarray | None = None,
        are_logs_scaled: bool = True,
    ) -> None:
        """Calculates and stores the smearing factor for correction."""
        if not self.y_was_log_transformed:
            return

        # --- Determine the unscaled, log-space predicted values ---
        if y_train_pred_log is None:
            if model is None or x_train is None:
                raise ValueError("Either (model, x_train) or y_train_pred_log must be provided.")
            # Predictions from the model are always on the scaled log space (if scaling is enabled).
            # Internal bookkeeping call: routes through `_predict_for_scoring`, not the public
            # `predict()`, so it works for a family whose public contract requires caller state
            # beyond `x` (e.g. FlexibleWidthNN under CapacitySelection.FIXED -- capacity-programme
            # FP-10).
            y_train_pred_log_scaled = model._predict_for_scoring(x_train)

            # Unscale them if necessary
            y_train_pred_log_unscaled = self.y_scaler.inverse_transform(y_train_pred_log_scaled.reshape(-1, 1)).flatten() if self.scale_y else y_train_pred_log_scaled
        else:
            # A prediction log is provided directly. Check if we need to unscale it.
            y_train_pred_log_unscaled = self.y_scaler.inverse_transform(y_train_pred_log.reshape(-1, 1)).flatten() if self.scale_y and are_logs_scaled else y_train_pred_log

        # --- Determine the unscaled, log-space true values ---
        y_train_log_unscaled = self.y_scaler.inverse_transform(y_train_log.reshape(-1, 1)).flatten() if self.scale_y and are_logs_scaled else y_train_log

        # --- Calculate smearing factor ---
        residuals = y_train_log_unscaled - y_train_pred_log_unscaled
        self.smearing_factor = np.mean(np.exp(residuals))


def create_train_val_split(
    x: np.ndarray,
    validation_fraction: float,
    test_fraction: float,
    split_strategy: DataSplitStrategy,
    timestamps: np.ndarray | None,
    random_state: int | None,
    force_to_test_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Creates train, validation, and test splits and returns the indices."""
    if timestamps is None and split_strategy != DataSplitStrategy.RANDOM:
        raise ValueError("timestamps cannot be None for non-random split strategies.")

    if force_to_test_indices is not None:
        all_indices = np.arange(x.shape[0])
        forced_test_indices = force_to_test_indices
        remaining_indices = np.setdiff1d(all_indices, forced_test_indices)
    else:
        remaining_indices = np.arange(x.shape[0])
        forced_test_indices = np.array([], dtype=int)

    if split_strategy == DataSplitStrategy.RANDOM:
        train_indices, val_indices, test_indices = _random_split(x[remaining_indices], validation_fraction, test_fraction, random_state)
    elif split_strategy == DataSplitStrategy.DISTINCT_DATES:
        if timestamps is None:
            raise ValueError("timestamps must be provided for distinct_dates split strategy.")
        if not np.issubdtype(timestamps.dtype, np.datetime64):
            try:
                timestamps = pd.to_datetime(timestamps)
            except (ValueError, TypeError) as e:
                raise ValueError("timestamps must be of type date or datetime for distinct_dates split strategy.") from e
        train_indices, val_indices, test_indices = _distinct_dates_split(validation_fraction, test_fraction, timestamps[remaining_indices], random_state)
    elif split_strategy == DataSplitStrategy.TIME_ORDERED:
        if timestamps is None:
            raise ValueError("timestamps must be provided for time_ordered split strategy.")
        if not np.issubdtype(timestamps.dtype, np.datetime64) and not np.issubdtype(timestamps.dtype, np.number):
            try:
                timestamps = pd.to_datetime(timestamps)
            except (ValueError, TypeError) as e:
                raise ValueError("timestamps must be of type date, datetime or numeric for time_ordered split strategy.") from e
        train_indices, val_indices, test_indices = _time_ordered_split(validation_fraction, test_fraction, timestamps[remaining_indices])
    else:
        raise ValueError(f"Unknown split strategy: {split_strategy}")

    # Map back to original indices
    train_indices = remaining_indices[train_indices]
    val_indices = remaining_indices[val_indices]
    test_indices = np.concatenate([forced_test_indices, remaining_indices[test_indices]])

    return train_indices, val_indices, test_indices


def _random_split(x: np.ndarray, validation_fraction: float, test_fraction: float, random_state: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Performs a random split of the data."""
    indices = np.arange(x.shape[0])
    if test_fraction > 0:
        train_val_indices, test_indices = train_test_split(indices, test_size=test_fraction, random_state=random_state)
    else:
        train_val_indices, test_indices = indices, np.array([], dtype=int)

    if validation_fraction > 0 and (1 - test_fraction) > 0:
        train_indices, val_indices = train_test_split(train_val_indices, test_size=validation_fraction / (1 - test_fraction), random_state=random_state)
    else:
        train_indices, val_indices = train_val_indices, np.array([], dtype=int)

    return train_indices, val_indices, test_indices


def _distinct_dates_split(validation_fraction: float, test_fraction: float, timestamps: np.ndarray | None, random_state: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Performs a split based on distinct dates."""
    if timestamps is None:
        raise ValueError("timestamps must be provided for distinct_dates split strategy.")

    unique_dates = pd.to_datetime(timestamps).date.unique()
    if test_fraction > 0:
        train_val_dates, test_dates = train_test_split(unique_dates, test_size=test_fraction, random_state=random_state)
    else:
        train_val_dates, test_dates = unique_dates, np.array([], dtype=object)

    if validation_fraction > 0 and (1 - test_fraction) > 0:
        train_dates, val_dates = train_test_split(train_val_dates, test_size=validation_fraction / (1 - test_fraction), random_state=random_state)
    else:
        train_dates, val_dates = train_val_dates, np.array([], dtype=object)

    timestamps_series = pd.Series(timestamps)
    train_indices = np.where(timestamps_series.dt.date.isin(train_dates))[0]
    val_indices = np.where(timestamps_series.dt.date.isin(val_dates))[0]
    test_indices = np.where(timestamps_series.dt.date.isin(test_dates))[0]

    return train_indices, val_indices, test_indices


def _time_ordered_split(validation_fraction: float, test_fraction: float, timestamps: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Performs a time-ordered split of the data."""
    if timestamps is None:
        raise ValueError("timestamps must be provided for time_ordered split strategy.")

    sorted_indices = timestamps.argsort()
    n_test = int(len(sorted_indices) * test_fraction)
    n_val = int(len(sorted_indices) * validation_fraction)
    n_train = len(sorted_indices) - n_test - n_val

    train_indices = sorted_indices[:n_train]
    val_indices = sorted_indices[n_train : n_train + n_val]
    test_indices = sorted_indices[n_train + n_val :]

    return train_indices, val_indices, test_indices
