"""Column roles for every model: a per-row starting score (``offset_feature``) and categorical columns (``categorical_features``).

Searched before writing: ``offset`` / ``categorical`` in ``automl_package/models`` (only the boosting wrappers handled them), ``OneHotEncoder`` in
``automl_package/preprocessing.py`` (reused below), the target-scaling build in ``BaseModel`` (the pattern for installing a transform only when
training is over), and ``tests/`` for the boosting column-role tests that must keep passing.

Both settings name columns of the X handed to ``fit`` / ``predict`` and live on `BaseModel`; ``categorical_encoding`` says how the model reads
the categorical columns. Two routes:

- Native (`ColumnRolesMixin`, the boosting wrappers): the library takes the starting score (LightGBM ``init_score``, XGBoost ``base_margin``,
  CatBoost ``Pool(baseline=...)``) and splits on the category codes itself (``CategoricalEncoding.NATIVE``).
- Generic (`ColumnRolesTransform`, every other model): `BaseModel.fit` turns the problem into an inner one: the offset column is dropped and
  subtracted from y, each categorical column is replaced in place by one 0/1 column per level (``CategoricalEncoding.ONE_HOT``). The model
  fits the inner problem unchanged; once training is over the public methods take the raw columns and `predict` adds the offset back.
"""

from typing import Any, ClassVar

import numpy as np
import pandas as pd

from automl_package.enums import CategoricalCast, CategoricalEncoding, MissingInputs
from automl_package.preprocessing import OneHotEncoder

MISSING_CATEGORY_CODE = -1  # CatBoost: a NaN category becomes this integer code, its own category
LEVEL_SEPARATOR = "="  # one-hot column name: ``<column>=<level>``


def check_named_columns(frame: pd.DataFrame, offset_feature: str | None, categorical_features: list[str] | None) -> None:
    """Raises ValueError when a column named by a role is not in ``frame`` (the message lists the valid columns)."""
    named = ([offset_feature] if offset_feature is not None else []) + (categorical_features or [])
    unknown = [name for name in named if name not in frame.columns]
    if unknown:
        raise ValueError(f"Columns {unknown} named in offset_feature / categorical_features are not in X; valid columns: {list(frame.columns)}.")


def offset_values(frame: pd.DataFrame, offset_feature: str) -> np.ndarray:
    """The starting score of every row of ``frame`` as floats; every row needs one."""
    offset = frame[offset_feature].to_numpy(dtype=float)
    if np.isnan(offset).any():
        raise ValueError(f"offset_feature {offset_feature!r} holds NaN: every row needs a starting score.")
    return offset


def broadcast_offset(offset: np.ndarray, like: np.ndarray) -> np.ndarray:
    """``offset`` (one value per row) shaped to broadcast over the output columns of ``like``."""
    return offset.reshape(-1, *([1] * (np.ndim(like) - 1)))


def _level_label(level: Any) -> str:
    """Integer-valued codes read as integers (``cat=1``), anything else as it prints."""
    return str(int(level)) if float(level).is_integer() else str(level)


class ColumnRolesTransform:
    """The generic route: raw X to the inner problem the model fits (offset column dropped, categorical columns one-hot encoded).

    `fit` takes the rows that define the levels: the sorted non-NaN values of each categorical column. When `BaseModel.fit` draws its own
    inner validation split from X the levels come from all of X; a caller giving an explicit ``x_val`` fixes them on the training rows only.
    `transform` replaces each categorical column IN PLACE by its one-hot columns ``<column>=<level>`` (sorted level order, float 0/1); a missing
    or unseen value is all zeros (`preprocessing.OneHotEncoder` with ``handle_unknown="ignore"``, fitted on non-NaN values only).
    """

    def __init__(self, offset_feature: str | None, categorical_features: list[str] | None) -> None:
        """Creates an unfitted transform for the named columns."""
        self.offset_feature = offset_feature
        self.categorical_features = list(categorical_features or [])
        self.input_names_: list[str] | None = None  # the raw columns, in the order a numpy X is read
        self.inner_names_: list[str] | None = None  # the columns of the inner problem
        self.levels_: dict[str, np.ndarray] = {}
        self._encoders: dict[str, OneHotEncoder] = {}

    def fit(self, x: pd.DataFrame) -> "ColumnRolesTransform":
        """Records the raw column names and the category levels of ``x``; returns self."""
        check_named_columns(x, self.offset_feature, self.categorical_features)
        self.input_names_ = list(x.columns)
        for name in self.categorical_features:
            observed = x[[name]].dropna()
            self._encoders[name] = OneHotEncoder(cols=[name]).fit(observed)
            self.levels_[name] = np.asarray(self._encoders[name].encoder.categories_[0])
        self.inner_names_ = [label for name in self.input_names_ if name != self.offset_feature for label in self._labels(name)]
        return self

    def _labels(self, name: str) -> list[str]:
        """The inner columns of raw column ``name``: itself, or ``<column>=<level>`` per level when it is categorical."""
        if name not in self._encoders:
            return [name]
        return [f"{name}{LEVEL_SEPARATOR}{_level_label(level)}" for level in self.levels_[name]]

    def transform(self, x: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray | None]:
        """Raw X to ``(inner X, starting score per row or None)``.

        Raises:
            ValueError: a raw column is missing from ``x``, or the starting score holds NaN.
        """
        if self.input_names_ is None:
            raise RuntimeError("ColumnRolesTransform is not fitted.")
        missing = [name for name in self.input_names_ if name not in x.columns]
        if missing:
            raise ValueError(f"Columns {missing} the model was fitted on are not in X; valid columns: {list(x.columns)}.")
        offset = None if self.offset_feature is None else offset_values(x, self.offset_feature)
        parts = []
        for name in self.input_names_:
            if name == self.offset_feature:
                continue
            if name not in self._encoders:
                parts.append(x[[name]])
                continue
            parts.append(pd.DataFrame(self._encoders[name].transform(x[[name]]), columns=self._labels(name), index=x.index))
        return pd.concat(parts, axis=1), offset

    def apply(self, x: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
        """``(inner X, y - starting score)`` for the training-side arrays of `BaseModel.fit` (y broadcast over the output columns)."""
        inner, offset = self.transform(x)
        y = np.asarray(y)
        return inner, y if offset is None else y - broadcast_offset(offset, y)


class ColumnRolesMixin:
    """The native route of the column roles: a boosting wrapper whose library takes the starting score and the category codes itself.

    The settings (``offset_feature``, ``categorical_features``, ``categorical_encoding``) and their checks live on `BaseModel`; this mixin
    separates the starting score from the features and casts the categorical columns for the library.

    - ``offset_feature``: removed from the features and given to the library (LightGBM ``init_score``, XGBoost ``base_margin``, CatBoost
      ``Pool(baseline=...)``); ``predict`` returns starting score plus trees.
    - ``categorical_features``: integer category codes (float dtype allowed, NaN = missing), split on natively.

    Subclasses set ``_CATEGORICAL_CAST``.
    """

    _default_missing_inputs: ClassVar[MissingInputs] = MissingInputs.NATIVE  # the boosting libraries handle NaN inputs themselves
    _native_column_roles: ClassVar[bool] = True  # the library takes the roles itself: `BaseModel.fit` leaves X and y alone
    _default_categorical_encoding: ClassVar[CategoricalEncoding] = CategoricalEncoding.NATIVE
    _CATEGORICAL_CAST: ClassVar[CategoricalCast]

    def __init__(self, **kwargs: Any) -> None:
        """Initializes the native route's state; ``kwargs`` (the column roles included) go to the model base class."""
        super().__init__(**kwargs)
        self.category_levels_: dict[str, np.ndarray] = {}  # PANDAS_CATEGORICAL only: the levels seen at fit, reused at predict

    def _split_column_roles(self, x: np.ndarray | pd.DataFrame, fit: bool = False) -> tuple[np.ndarray | pd.DataFrame, np.ndarray | None]:
        """Separates the starting score from the features and casts the categorical columns for the library.

        Args:
            x: the full X (numpy in ``self.feature_names`` order, or a DataFrame holding the named columns).
            fit: True on the training part, where the XGBoost category levels are recorded; False elsewhere (validation, predict).

        Returns:
            ``(features, offset)``; ``features`` is ``x`` itself and ``offset`` None when no role is set.
        """
        if not self._has_column_roles:
            return x, None
        if isinstance(x, pd.DataFrame):
            frame = x
        elif self.feature_names is None:
            raise ValueError("Column roles need the column names: fit the model (or give a DataFrame) before calling this on a numpy array.")
        else:
            frame = pd.DataFrame(x, columns=self.feature_names)

        check_named_columns(frame, self.offset_feature, self.categorical_features)
        offset = None if self.offset_feature is None else offset_values(frame, self.offset_feature)
        features = frame.drop(columns=[self.offset_feature] if self.offset_feature is not None else [])
        return self._cast_categoricals(features, fit), offset

    def _cast_categoricals(self, features: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Casts the categorical columns of ``features`` per the library's ``_CATEGORICAL_CAST`` rule."""
        columns = self.categorical_features
        if not columns:
            return features
        match self._CATEGORICAL_CAST:
            case CategoricalCast.NATIVE_CODES:
                pass
            case CategoricalCast.INTEGER_CODES:
                features[columns] = features[columns].fillna(MISSING_CATEGORY_CODE).astype(int)
            case CategoricalCast.PANDAS_CATEGORICAL:
                if fit:
                    self.category_levels_ = {name: np.sort(features[name].dropna().unique()).astype(int) for name in columns}
                elif not self.category_levels_:
                    raise RuntimeError("Category levels are recorded at fit: the model has not been fitted.")
                for name in columns:
                    features[name] = pd.Categorical(features[name], categories=self.category_levels_[name])
        return features
