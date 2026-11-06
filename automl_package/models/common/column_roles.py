"""Mixin giving the boosting wrappers two named column roles: a per-row starting score and categorical columns.

Searched before writing: no offset or categorical handling existed in ``automl_package/models``. One split method serves
LightGBM, XGBoost and CatBoost, so the rule lives here once; each wrapper only declares how its library wants the
categorical codes (``_CATEGORICAL_CAST``) and passes the natives (``init_score`` / ``base_margin`` / ``baseline``).
"""

from typing import Any, ClassVar

import numpy as np
import pandas as pd

from automl_package.enums import CategoricalCast, MissingInputs, TaskType, UncertaintyMethod

MISSING_CATEGORY_CODE = -1  # CatBoost: a NaN category becomes this integer code, its own category


class ColumnRolesMixin:
    """Per-row starting score (``offset_feature``) and categorical columns (``categorical_features``) for a boosting wrapper.

    Both settings name columns of the X handed to ``fit`` / ``predict``, so they follow every split, CV fold and search trial.

    - ``offset_feature``: the column holding each row's starting score on the model's raw prediction scale. It is removed from
      the features and given to the library (LightGBM ``init_score``, XGBoost ``base_margin``, CatBoost ``Pool(baseline=...)``);
      ``predict`` returns starting score plus trees. Supported only with ``TaskType.REGRESSION`` and ``UncertaintyMethod.CONSTANT``.
    - ``categorical_features``: columns holding integer category codes (float dtype allowed, NaN = missing).

    Subclasses set ``_CATEGORICAL_CAST``.
    """

    _default_missing_inputs: ClassVar[MissingInputs] = MissingInputs.NATIVE  # the boosting libraries handle NaN inputs themselves
    _CATEGORICAL_CAST: ClassVar[CategoricalCast]

    def __init__(self, offset_feature: str | None = None, categorical_features: list[str] | None = None, **kwargs: Any) -> None:
        """Initializes the column roles.

        Args:
            offset_feature: name of the column holding each row's starting score, or None.
            categorical_features: names of the columns holding integer category codes, or None.
            **kwargs: passed on to the model base class.
        """
        super().__init__(**kwargs)
        self.offset_feature = offset_feature
        self.categorical_features = list(categorical_features) if categorical_features else None
        self.category_levels_: dict[str, np.ndarray] = {}  # PANDAS_CATEGORICAL only: the levels seen at fit, reused at predict

    def get_params(self) -> dict[str, Any]:
        """Gets parameters for this estimator, including the two column roles (so clones keep them)."""
        params = super().get_params()
        params.update({"offset_feature": self.offset_feature, "categorical_features": self.categorical_features})
        return params

    @property
    def _has_column_roles(self) -> bool:
        return self.offset_feature is not None or bool(self.categorical_features)

    def _target_scaling_conflict(self) -> str | None:
        """A starting score lives on the raw prediction scale and a Poisson objective reads counts: neither survives a rescaled target."""
        if self.offset_feature is not None:
            return "offset_feature (the per-row starting score is on the raw prediction scale)"
        objectives = [getattr(self, "objective", None), self.params.get("loss_function"), getattr(self, "loss", None)]
        if any("poisson" in str(getattr(o, "value", o)).lower() for o in objectives if o is not None):
            return "a Poisson objective (it reads counts)"
        return None

    def _validate_column_roles(self) -> None:
        """Raises ValueError when the model's other settings cannot work with the column roles."""
        if self.offset_feature is not None and (self.task_type != TaskType.REGRESSION or self.uncertainty_method != UncertaintyMethod.CONSTANT):
            raise ValueError(f"offset_feature is supported only with task_type=REGRESSION and uncertainty_method=CONSTANT (got {self.task_type}, {self.uncertainty_method}).")
        if self.calculate_feature_importance:
            raise ValueError("offset_feature / categorical_features need calculate_feature_importance=False: SHAP would see the offset column and the raw codes.")
        if self.feature_selection_threshold is not None:
            raise ValueError("offset_feature / categorical_features cannot be combined with feature_selection_threshold: selection could drop the named columns.")
        if self.offset_feature is not None and self.offset_feature in (self.categorical_features or []):
            raise ValueError(f"offset_feature {self.offset_feature!r} is also listed in categorical_features.")

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
        self._validate_column_roles()
        if isinstance(x, pd.DataFrame):
            frame = x
        elif self.feature_names is None:
            raise ValueError("Column roles need the column names: fit the model (or give a DataFrame) before calling this on a numpy array.")
        else:
            frame = pd.DataFrame(x, columns=self.feature_names)

        named = ([self.offset_feature] if self.offset_feature is not None else []) + (self.categorical_features or [])
        unknown = [name for name in named if name not in frame.columns]
        if unknown:
            raise ValueError(f"Columns {unknown} named in offset_feature / categorical_features are not in X; valid columns: {list(frame.columns)}.")

        offset = None
        if self.offset_feature is not None:
            offset = frame[self.offset_feature].to_numpy(dtype=float)
            if np.isnan(offset).any():
                raise ValueError(f"offset_feature {self.offset_feature!r} holds NaN: every row needs a starting score.")
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
