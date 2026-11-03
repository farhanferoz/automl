"""Preprocessing utilities for AutoML."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder


class OrderedTargetEncoder(BaseEstimator, TransformerMixin):
    """CatBoost-style ordered target statistics for categorical columns, usable in front of any model.

    The statistic of a category is the smoothed target mean ``(sum + prior * smoothing) / (count + smoothing)`` over some
    set of contributing training rows that share the category; ``prior`` is the mean of the known targets. Rows whose
    target is NaN are encoded like any other row but contribute nothing to any sum or count.

    One time rule decides which contributions a row may use. Each contribution has a ``known_at`` value (when its target
    became known) and each encoded row an ``order`` value (when it is encoded); a row uses exactly the contributions of
    its category with ``known_at < order``, strictly: equal values do not see each other, and with ``known_at == order``
    (the default in ``fit_transform``) a row never sees its own target. ``order`` and ``known_at`` are numbers or
    datetime64 values; they are compared on one common rank scale. Without ``order``, ``fit_transform`` uses a
    permutation seeded by ``random_state`` and ``transform`` uses every contribution (an unseen category gets
    ``prior``). A row with no usable contribution gets ``prior``. A missing category value is a category of its own.
    Fit on training rows only, inside each fold.
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

    def fit(self, x: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray, known_at: np.ndarray | None = None) -> "OrderedTargetEncoder":
        """Stores, per encoded column, each category's target sum and count over the rows with a known target, and the prior.

        Args:
            x: training features.
            y: training targets; NaN rows contribute nothing.
            known_at: when each target became known; needed to later call ``transform`` with an ``order``. The
                contributions are then kept sorted by (category, known_at).
        """
        frame, target = _as_frame(x), np.asarray(y, dtype=float)
        known = ~np.isnan(target)
        self.prior_ = float(target[known].mean())
        self.columns_ = self._columns(frame)
        self.stats_, self.contributions_ = {}, {}
        when = None if known_at is None else np.asarray(known_at)
        for col in self.columns_:
            codes, levels = pd.factorize(frame[col], use_na_sentinel=False)
            sums = np.bincount(codes[known], weights=target[known], minlength=len(levels))
            counts = np.bincount(codes[known], minlength=len(levels))
            self.stats_[col] = (pd.Index(levels), (sums + self.prior_ * self.smoothing) / (counts + self.smoothing))
            if when is not None:
                by_time = np.lexsort((when[known], codes[known]))
                self.contributions_[col] = (codes[known][by_time], when[known][by_time], target[known][by_time])
        return self

    def _ordered_stat(self, col: str | int, codes: np.ndarray, order: np.ndarray) -> np.ndarray:
        """Smoothed statistic of each row from the contributions of its category with ``known_at`` strictly before its ``order``."""
        c_codes, c_when, c_y = self.contributions_[col]
        scale = np.unique(np.concatenate([c_when, order]))  # one integer rank scale for both
        width = len(scale) + 1
        keys = c_codes.astype(np.int64) * width + np.searchsorted(scale, c_when)
        sums, counts = np.concatenate([[0.0], np.cumsum(c_y)]), np.arange(len(c_y) + 1)
        end = np.searchsorted(keys, codes.astype(np.int64) * width + np.searchsorted(scale, order), side="left")
        start = np.searchsorted(keys, codes.astype(np.int64) * width, side="left")
        seen = codes >= 0  # an unseen category (-1) has no contributions
        n, total = np.where(seen, counts[end] - counts[start], 0), np.where(seen, sums[end] - sums[start], 0.0)
        return (total + self.prior_ * self.smoothing) / (n + self.smoothing)

    def transform(self, x: pd.DataFrame | np.ndarray, order: np.ndarray | None = None) -> pd.DataFrame | np.ndarray:
        """Encodes rows: with ``order``, each from the contributions known strictly before its order value; else from all of them."""
        if order is not None and not self.contributions_:
            raise ValueError("transform(order=...) needs known_at: fit the encoder with fit(x, y, known_at=...).")
        frame = _as_frame(x).copy()
        for col, (levels, stat) in self.stats_.items():
            pos = levels.get_indexer(frame[col])
            pos = np.where(pd.isna(frame[col]).to_numpy(), levels.get_indexer([np.nan])[0], pos)  # None and NaN are one missing category
            if order is None:
                frame[col] = np.where(pos >= 0, stat[pos], self.prior_)
            else:
                frame[col] = self._ordered_stat(col, pos, np.asarray(order))
        return _like(frame, x)

    def fit_transform(
        self, x: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray | None = None, order: np.ndarray | None = None, known_at: np.ndarray | None = None
    ) -> pd.DataFrame | np.ndarray:
        """Fits on the training rows and returns their ordered (leakage-free) encoding.

        Args:
            x: training features.
            y: training targets (NaN rows are encoded but contribute nothing).
            order: the encoding order of the rows; None uses a permutation seeded by ``random_state``.
            known_at: when each target became known; defaults to ``order``, so a row sees only strictly earlier rows. Must
                be None when ``order`` is None.
        """
        if y is None:
            raise TypeError("fit_transform() missing 1 required argument: 'y'")
        if order is None:
            if known_at is not None:
                raise ValueError("known_at needs an order: pass order=... or leave known_at unset.")
            order = np.random.default_rng(self.random_state).permutation(len(_as_frame(x)))
        return self.fit(x, y, known_at=order if known_at is None else known_at).transform(x, order=order)


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
