"""Cross-validation utilities."""

from collections.abc import Generator

import numpy as np
from sklearn.model_selection._split import BaseCrossValidator

from automl_package.enums import DataSplitStrategy
from automl_package.utils.data_handler import check_time_gap, rows_after_gap


class TimeSeriesSplit(BaseCrossValidator):
    """Time series cross-validator."""

    def __init__(self, n_splits: int = 5, gap: int = 0, timestamps: np.ndarray | None = None) -> None:
        """Initialize the TimeSeriesSplit.

        Parameters
        ----------
        n_splits : int, default=5
            Number of splits. Must be at least 2.
        gap : int, default=0
            Distinct timestamps left out between each fold's training rows and its validation rows (`data_handler.rows_after_gap`).
        timestamps : array-like of shape (n_samples,), default=None
            Time of each row of the data `split` receives. Given, the rows are ordered by it (stable sort) and the gap counts its
            distinct values; None, the rows are taken in the order given and the gap counts rows.
        """
        if n_splits < 2:
            raise ValueError(f"n_splits={n_splits} cannot be smaller than 2.")
        check_time_gap(gap, DataSplitStrategy.TIME_ORDERED)
        self.n_splits = n_splits
        self.gap = gap
        self.timestamps = None if timestamps is None else np.asarray(timestamps)

    def get_n_splits(self, _x: np.ndarray = None, _y: np.ndarray = None, _groups: np.ndarray = None) -> int:
        """Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        _x : object
            Always ignored, exists for compatibility.
        _y : object
            Always ignored, exists for compatibility.
        _groups : object
            Always ignored, exists for compatibility.

        Returns:
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits

    def split(self, x: np.ndarray, _y: np.ndarray = None, _groups: np.ndarray = None) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,), default=None
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,), default=None
            Always ignored, exists for compatibility.

        Yields:
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        n_samples = len(x)
        k_fold_size = n_samples // self.n_splits
        if self.timestamps is None:
            indices, times = np.arange(n_samples), np.arange(n_samples)
        elif len(self.timestamps) != n_samples:
            raise ValueError(f"TimeSeriesSplit holds {len(self.timestamps)} timestamps for {n_samples} rows: build it with the timestamps of the rows it splits.")
        else:
            indices, times = np.argsort(self.timestamps, kind="stable"), self.timestamps
        distinct = np.unique(times) if self.gap else None

        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.8 * (stop - start)) + start
            train, val = indices[start:mid], indices[mid:stop]
            if self.gap:
                val = rows_after_gap(val, times, times[train].max(), distinct, self.gap, f"fold {i + 1} validation")
            yield train, val
