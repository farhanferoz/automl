"""Cross-validation utilities."""

from collections.abc import Generator

import numpy as np
from sklearn.model_selection._split import BaseCrossValidator


class TimeSeriesSplit(BaseCrossValidator):
    """Time series cross-validator."""

    def __init__(self, n_splits: int = 5) -> None:
        """Initialize the TimeSeriesSplit.

        Parameters
        ----------
        n_splits : int, default=5
            Number of splits. Must be at least 2.
        """
        if n_splits < 2:
            raise ValueError(f"n_splits={n_splits} cannot be smaller than 2.")
        self.n_splits = n_splits

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
        indices = np.arange(n_samples)

        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.8 * (stop - start)) + start
            yield indices[start:mid], indices[mid + margin : stop]
