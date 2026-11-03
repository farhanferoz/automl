"""Preprocessing utilities for AutoML."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder


class OrderedTargetEncoder(BaseEstimator, TransformerMixin):
    """CatBoost-style ordered target statistics for categorical columns, usable in front of any model.

    ``fit_transform`` (the training rows): each row's category is replaced by the smoothed target mean of the rows BEFORE
    it that share the category, ``(sum_before + prior * smoothing) / (count_before + smoothing)``, ``prior`` being the
    training target mean, so no row sees its own target. "Before" follows ``order`` when given (e.g. timestamps for a
    time series; rows with equal order values do not see each other), else a permutation seeded by ``random_state``.
    ``transform`` (any other rows): the same statistic over all training rows of the category; an unseen category gets
    ``prior``. A missing value is a category of its own. Fit on training rows only, inside each fold.
    """

    def __init__(self, cols: list[str | int] | None = None, smoothing: float = 1.0, random_state: int | None = None) -> None:
        """Initializes the encoder.

        Args:
            cols: names (DataFrame input) or positions (array input) of the columns to encode; None encodes every object or
                category column of a DataFrame.
            smoothing: weight of the prior, in rows.
            random_state: seed of the permutation used when ``fit_transform`` gets no ``order``.
        """
        self.cols = cols
        self.smoothing = smoothing
        self.random_state = random_state

    def _columns(self, frame: pd.DataFrame) -> list:
        if self.cols is not None:
            missing = [c for c in self.cols if c not in frame.columns]
            if missing:
                raise ValueError(f"Columns {missing} not found in X.")
            return list(self.cols)
        return list(frame.select_dtypes(include=["object", "category"]).columns)

    def fit(self, x: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "OrderedTargetEncoder":
        """Stores, per encoded column, each category's target sum and count over the training rows, and the prior."""
        frame, target = _as_frame(x), np.asarray(y, dtype=float)
        self.prior_ = float(target.mean())
        self.columns_ = self._columns(frame)
        self.stats_ = {}
        for col in self.columns_:
            codes, levels = pd.factorize(frame[col], use_na_sentinel=False)
            sums, counts = np.bincount(codes, weights=target, minlength=len(levels)), np.bincount(codes, minlength=len(levels))
            self.stats_[col] = (pd.Index(levels), (sums + self.prior_ * self.smoothing) / (counts + self.smoothing))
        return self

    def transform(self, x: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """Encodes rows the encoder was not fitted on: each category's statistic over all training rows."""
        frame = _as_frame(x).copy()
        for col, (levels, stat) in self.stats_.items():
            pos = levels.get_indexer(frame[col])
            frame[col] = np.where(pos >= 0, stat[pos], self.prior_)
        return _like(frame, x)

    def fit_transform(self, x: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray | None = None, order: np.ndarray | None = None) -> pd.DataFrame | np.ndarray:
        """Fits on the training rows and returns their ordered (leakage-free) encoding."""
        if y is None:
            raise TypeError("fit_transform() missing 1 required argument: 'y'")
        self.fit(x, y)
        frame, target = _as_frame(x).copy(), np.asarray(y, dtype=float)
        if order is None:
            order = np.random.default_rng(self.random_state).permutation(len(frame))
        order_codes = pd.factorize(np.asarray(order), sort=True)[0]
        for col in self.columns_:
            codes = pd.factorize(frame[col], use_na_sentinel=False)[0]
            groups = pd.DataFrame({"c": codes, "o": order_codes, "y": target}).groupby(["c", "o"], sort=True)["y"].agg(["sum", "count"])
            before = groups.groupby(level="c").cumsum() - groups
            stat = (before["sum"] + self.prior_ * self.smoothing) / (before["count"] + self.smoothing)
            frame[col] = stat.reindex(pd.MultiIndex.from_arrays([codes, order_codes], names=["c", "o"])).to_numpy()
        return _like(frame, x)


def _as_frame(x: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """A DataFrame view of the input; array columns are named by position."""
    return x if isinstance(x, pd.DataFrame) else pd.DataFrame(x)


def _like(frame: pd.DataFrame, x: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
    """The result in the input's type."""
    return frame if isinstance(x, pd.DataFrame) else frame.to_numpy()


class OneHotEncoder(BaseEstimator, TransformerMixin):
    """A wrapper around sklearn.preprocessing.OneHotEncoder for categorical features."""

    def __init__(self, cols: list[str | int] | None = None, handle_unknown: str = "ignore") -> None:
        """Initializes the OneHotEncoder.

        Args:
            cols (list[str | int], optional): List of column names or indices to encode.
                                              If None, all categorical columns will be encoded.
            handle_unknown (str): Whether to raise an error or ignore unknown categories.
        """
        self.cols = cols
        self.handle_unknown = handle_unknown
        self.encoder = SklearnOneHotEncoder(handle_unknown=handle_unknown, sparse_output=False)
        self.feature_names_out_ = None  # To store feature names after transformation

    def fit(
        self,
        x: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,  # noqa: ARG002
    ) -> "OneHotEncoder":
        """Fits the encoder to the training data.

        Args:
            x (pd.DataFrame | np.ndarray): Training features.
            y (pd.Series | np.ndarray, optional): Training target (ignored).

        Returns:
            self: Fitted encoder.
        """
        if isinstance(x, np.ndarray):
            x_temp = pd.DataFrame(x)
            x_selected = x_temp.iloc[:, self.cols] if self.cols is not None else x_temp
        else:  # X is pandas DataFrame
            x_selected = x[self.cols] if self.cols is not None else x

        self.encoder.fit(x_selected)
        return self

    def transform(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Transforms the input data using the fitted encoder.

        Args:
            x (pd.DataFrame | np.ndarray): Data to transform.

        Returns:
            np.ndarray: Transformed data.
        """
        if isinstance(x, np.ndarray):
            x_temp = pd.DataFrame(x)
            x_selected = x_temp.iloc[:, self.cols] if self.cols is not None else x_temp
        else:  # X is pandas DataFrame
            x_selected = x[self.cols] if self.cols is not None else x

        transformed_data = self.encoder.transform(x_selected)

        # Store feature names for consistency if needed later
        if self.feature_names_out_ is None:
            if hasattr(self.encoder, "get_feature_names_out"):
                self.feature_names_out_ = self.encoder.get_feature_names_out(self.cols if self.cols else x_selected.columns)
            else:  # Fallback for older sklearn versions or if method is missing
                self.feature_names_out_ = [f"x{i}" for i in range(transformed_data.shape[1])]  # Generic names

        return transformed_data

    def fit_transform(
        self,
        x: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        **_fit_params: Any,
    ) -> np.ndarray:
        """Fits the encoder and transforms the data.

        Args:
            x (pd.DataFrame | np.ndarray): Training features.
            y (pd.Series | np.ndarray, optional): Training target (ignored).
            **_fit_params: Additional fit parameters (ignored).

        Returns:
            np.ndarray: Transformed data.
        """
        self.fit(x, y)
        return self.transform(x)
