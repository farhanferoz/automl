"""Data handling utilities."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from automl_package.enums import DataSplitStrategy


class DataHandler:
    """Handles data scaling and splitting."""

    def __init__(self, scale_x: bool = True, scale_y: bool = True, scale_binary_features: bool = False) -> None:
        """Initializes the DataHandler."""
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.scale_binary_features = scale_binary_features
        self.x_scaler = StandardScaler() if scale_x else None
        self.y_scaler = StandardScaler() if scale_y else None
        self.binary_feature_indices_ = []
        self.non_binary_feature_indices_ = []

    def _detect_binary_features(self, x: np.ndarray):
        if not self.scale_binary_features:
            self.binary_feature_indices_ = [i for i in range(x.shape[1]) if np.all(np.isin(x[:, i], [0, 1]))]
            self.non_binary_feature_indices_ = [i for i in range(x.shape[1]) if i not in self.binary_feature_indices_]
        else:
            self.non_binary_feature_indices_ = list(range(x.shape[1]))

    def fit_transform(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fits and transforms the data."""
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

    def inverse_transform_y(self, y_pred_scaled: np.ndarray) -> np.ndarray:
        """Inverse transforms the scaled y predictions."""
        if not self.scale_y:
            return y_pred_scaled
        return self.y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()


def create_train_val_split(
    x: np.ndarray,
    validation_fraction: float,
    test_fraction: float,
    split_strategy: DataSplitStrategy,
    timestamps: np.ndarray | None,
    random_state: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Creates train, validation, and test splits and returns the indices."""
    if timestamps is None and split_strategy != DataSplitStrategy.RANDOM:
        raise ValueError("timestamps cannot be None for non-random split strategies.")

    if split_strategy == DataSplitStrategy.RANDOM:
        train_indices, val_indices, test_indices = _random_split(x, validation_fraction, test_fraction, random_state)
    elif split_strategy == DataSplitStrategy.DISTINCT_DATES:
        if timestamps is None:
            raise ValueError("timestamps must be provided for distinct_dates split strategy.")
        if not np.issubdtype(timestamps.dtype, np.datetime64):
            try:
                timestamps = pd.to_datetime(timestamps)
            except (ValueError, TypeError) as e:
                raise ValueError("timestamps must be of type date or datetime for distinct_dates split strategy.") from e
        train_indices, val_indices, test_indices = _distinct_dates_split(validation_fraction, test_fraction, timestamps, random_state)
    elif split_strategy == DataSplitStrategy.TIME_ORDERED:
        if timestamps is None:
            raise ValueError("timestamps must be provided for time_ordered split strategy.")
        if not np.issubdtype(timestamps.dtype, np.datetime64) and not np.issubdtype(timestamps.dtype, np.number):
            try:
                timestamps = pd.to_datetime(timestamps)
            except (ValueError, TypeError) as e:
                raise ValueError("timestamps must be of type date, datetime or numeric for time_ordered split strategy.") from e
        train_indices, val_indices, test_indices = _time_ordered_split(validation_fraction, test_fraction, timestamps)
    else:
        raise ValueError(f"Unknown split strategy: {split_strategy}")

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


def _distinct_dates_split(
    validation_fraction: float,
    test_fraction: float,
    timestamps: np.ndarray | None,
    random_state: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
