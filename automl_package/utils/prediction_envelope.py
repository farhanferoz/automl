"""Training envelope: the range of the data a model was fitted on, and a check of predictions against it.

Why this exists: a neural forecaster once returned predictions of 1e18 for a provider with almost no history and nobody noticed
for two days. Every ``BaseModel`` therefore records, at fit time, where its training data lived (``TrainingEnvelope``) and at
every ``predict`` reports how many predictions and input rows fall outside it (``PredictionRangeReport``). The default action is
to FLAG (log and keep the report in ``model.last_prediction_report_``); a fault is never hidden silently. ``OutOfRangeAction.CLIP``
and ``RAISE`` are explicit opt-ins.

What the envelope holds:
    - per output column of y (1-D or 2-D ``(N, H)``): the NaN-aware minimum and maximum;
    - per numeric input column: the NaN-aware minimum and maximum (all-NaN columns are skipped);
    - per categorical input column: the set of seen values instead of a range.

What "out of range" means: a prediction is out of range when it is below its output column's training minimum or above its maximum.
NaN and infinite predictions are counted as out of range too (they are a fault as well); they do not enter ``worst_exceedance``,
which measures finite distances only. An input row is out of range when any numeric input lies outside its training minimum or
maximum; NaN inputs are not counted. An input row has an unseen category when a categorical column holds a non-NaN value never seen in training.
Being outside the range is not by itself an error (a model may extrapolate legitimately); the report says how much, and the caller decides.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from automl_package.logger import logger

DEFAULT_TOP_INPUT_COLUMNS = 5
REPORT_KEY_PREFIX = "range_"


def _as_columns(values: np.ndarray | pd.DataFrame | pd.Series) -> np.ndarray:
    """Returns ``values`` as a float 2-D array with one column per output or input."""
    arr = np.asarray(values, dtype=float)
    return arr.reshape(arr.shape[0], -1)


@dataclass
class TrainingEnvelope:
    """Range of the data a model was fitted on.

    Attributes:
        feature_names: input column names, in training order.
        input_min: per input column minimum (NaN for categorical or all-NaN columns, which have no range).
        input_max: per input column maximum (same convention).
        categorical_values: seen values per categorical input column index.
        output_min: per output column minimum of y.
        output_max: per output column maximum of y.
    """

    feature_names: list[str]
    input_min: np.ndarray
    input_max: np.ndarray
    categorical_values: dict[int, np.ndarray]
    output_min: np.ndarray
    output_max: np.ndarray
    categorical_columns: list[str] = field(default_factory=list)

    @classmethod
    def from_fit_data(
        cls,
        x: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.DataFrame | pd.Series,
        feature_names: Sequence[str],
        categorical_columns: Sequence[str] | None = None,
    ) -> TrainingEnvelope:
        """Builds the envelope from the fit data (vectorised; NaN targets and inputs are ignored).

        Args:
            x: training inputs, shape ``(N, F)``.
            y: training targets, shape ``(N,)`` or ``(N, H)``.
            feature_names: the ``F`` input column names.
            categorical_columns: names of the columns that hold category codes; their seen values are kept instead of min and max.

        Returns:
            The envelope.
        """
        x_arr = _as_columns(x)
        y_arr = _as_columns(y)
        names = list(feature_names)
        categorical = [c for c in (categorical_columns or []) if c in names]
        cat_idx = [names.index(c) for c in categorical]
        # fmin/fmax skip NaN without a temporary copy and give NaN (no warning) for an all-NaN column.
        input_min = np.fmin.reduce(x_arr, axis=0)
        input_max = np.fmax.reduce(x_arr, axis=0)
        input_min[cat_idx] = np.nan
        input_max[cat_idx] = np.nan
        seen = {i: np.unique(x_arr[~np.isnan(x_arr[:, i]), i]) for i in cat_idx}
        return cls(
            feature_names=names,
            input_min=input_min,
            input_max=input_max,
            categorical_values=seen,
            output_min=np.fmin.reduce(y_arr, axis=0),
            output_max=np.fmax.reduce(y_arr, axis=0),
            categorical_columns=categorical,
        )


@dataclass(frozen=True)
class PredictionRangeReport:
    """How many predictions and input rows of one ``predict`` call fall outside the training envelope.

    Attributes:
        n_rows: rows predicted.
        n_outputs: prediction entries checked (rows times output columns).
        n_outputs_out_of_range: entries below the training minimum, above the maximum, NaN or infinite.
        share_outputs_out_of_range: ``n_outputs_out_of_range / n_outputs`` (0.0 when nothing was checked).
        worst_exceedance: largest finite distance beyond the training range, in units of that output column's training range width
            (absolute distance for a zero-width column); 0.0 if none.
        n_rows_input_out_of_range: rows with any numeric input outside its training minimum or maximum.
        share_rows_input_out_of_range: that count over ``n_rows``.
        top_input_columns: up to ``top_n`` ``(name, count)`` pairs, the input columns most often out of range.
        n_rows_unseen_category: rows with a category value never seen in training.
    """

    n_rows: int
    n_outputs: int
    n_outputs_out_of_range: int
    share_outputs_out_of_range: float
    worst_exceedance: float
    n_rows_input_out_of_range: int
    share_rows_input_out_of_range: float
    top_input_columns: tuple[tuple[str, int], ...]
    n_rows_unseen_category: int

    def as_dict(self) -> dict[str, float | int | str]:
        """Flat dict with stable ``range_``-prefixed snake_case keys (for CSV columns); ``top_input_columns`` becomes ``name:count;name:count``."""
        out: dict[str, float | int | str] = {f"{REPORT_KEY_PREFIX}{key}": value for key, value in self.__dict__.items() if key != "top_input_columns"}
        out[f"{REPORT_KEY_PREFIX}top_input_columns"] = ";".join(f"{name}:{count}" for name, count in self.top_input_columns)
        return out

    def summary(self) -> str:
        """One-line description for the log."""
        return (
            f"{self.n_outputs_out_of_range}/{self.n_outputs} predictions out of training range ({self.share_outputs_out_of_range:.2%}), "
            f"worst exceedance {self.worst_exceedance:.3g} ranges; {self.n_rows_input_out_of_range}/{self.n_rows} rows with an input out of range "
            f"({self.share_rows_input_out_of_range:.2%}), {self.n_rows_unseen_category} rows with an unseen category"
        )


def _output_exceedance(envelope: TrainingEnvelope, predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per entry: (out-of-range mask incl. non-finite, finite distance beyond the range in range widths)."""
    below = envelope.output_min - predictions
    above = predictions - envelope.output_max
    distance = np.fmax(np.fmax(below, above), 0.0)  # NaN-safe: fmax ignores NaN
    width = envelope.output_max - envelope.output_min
    distance = distance / np.where(width > 0, width, 1.0)
    finite = np.isfinite(predictions)
    out = ~finite | (below > 0) | (above > 0)
    return out, np.where(finite, distance, 0.0)


def check_prediction_range(
    envelope: TrainingEnvelope,
    x: np.ndarray | pd.DataFrame,
    predictions: np.ndarray,
    input_columns: Sequence[str] | None = None,
    top_n: int = DEFAULT_TOP_INPUT_COLUMNS,
) -> PredictionRangeReport:
    """Compares one prediction call's inputs and outputs with the training envelope. NaN and infinite predictions count as out of range.

    Args:
        envelope: the fit-time envelope.
        x: the inputs of the call. A DataFrame is matched to the envelope by column name; an array by position.
        predictions: the predictions, shape ``(N,)`` or ``(N, H)``.
        input_columns: names of the columns of an array ``x`` when it is narrower than the training inputs (feature selection).
        top_n: how many input columns to list in ``top_input_columns``.

    Returns:
        The report. If the prediction has a different number of columns than the training targets, the output check is skipped
        (``n_outputs`` is 0) and a warning is logged.
    """
    pred = _as_columns(predictions)
    n_rows = pred.shape[0]
    if pred.shape[1] == envelope.output_min.shape[0]:
        out, distance = _output_exceedance(envelope, pred)
        n_out, n_bad, worst = out.size, int(out.sum()), float(distance.max(initial=0.0))
    else:
        logger.warning(f"Prediction has {pred.shape[1]} columns but the training targets had {envelope.output_min.shape[0]}: output range not checked.")
        n_out = n_bad = 0
        worst = 0.0

    names = list(x.columns) if isinstance(x, pd.DataFrame) else list(input_columns if input_columns is not None else envelope.feature_names)
    x_arr = _as_columns(x)
    if x_arr.shape[1] != len(names):
        raise ValueError(f"x has {x_arr.shape[1]} columns but {len(names)} column names were given.")
    pos = np.array([envelope.feature_names.index(n) for n in names if n in envelope.feature_names], dtype=int)
    keep = np.array([n in envelope.feature_names for n in names])
    x_arr = x_arr[:, keep]
    lo, hi = envelope.input_min[pos], envelope.input_max[pos]
    outside = (x_arr < lo) | (x_arr > hi)  # NaN compares False; categorical and all-NaN columns have NaN bounds and never trigger
    per_column = outside.sum(axis=0)
    top = np.argsort(-per_column, kind="stable")[:top_n]
    top_columns = tuple((envelope.feature_names[pos[i]], int(per_column[i])) for i in top if per_column[i] > 0)
    unseen_rows = np.zeros(n_rows, dtype=bool)
    for col, seen in envelope.categorical_values.items():
        where = np.flatnonzero(pos == col)
        if where.size:
            values = x_arr[:, where[0]]
            unseen_rows |= ~np.isnan(values) & ~np.isin(values, seen)
    n_in_bad = int(outside.any(axis=1).sum())
    return PredictionRangeReport(
        n_rows=n_rows,
        n_outputs=n_out,
        n_outputs_out_of_range=n_bad,
        share_outputs_out_of_range=n_bad / n_out if n_out else 0.0,
        worst_exceedance=worst,
        n_rows_input_out_of_range=n_in_bad,
        share_rows_input_out_of_range=n_in_bad / n_rows if n_rows else 0.0,
        top_input_columns=top_columns,
        n_rows_unseen_category=int(unseen_rows.sum()),
    )


def clip_to_envelope(envelope: TrainingEnvelope, predictions: np.ndarray) -> tuple[np.ndarray, int]:
    """Clips each output column to its training [min, max]; returns the clipped array (same shape) and how many entries moved. NaN stays NaN."""
    pred = np.asarray(predictions, dtype=float)
    cols = pred.reshape(pred.shape[0], -1)
    if cols.shape[1] != envelope.output_min.shape[0]:
        return predictions, 0
    clipped = np.clip(cols, envelope.output_min, envelope.output_max)
    n_moved = int(((clipped != cols) & ~np.isnan(cols)).sum())
    return clipped.reshape(pred.shape), n_moved
