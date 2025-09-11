"""Lookup mapper for probability mapping."""

from enum import Enum
from typing import Any

import numpy as np

from automl_package.enums import MapperType, UncertaintyMethod
from automl_package.logger import logger
from automl_package.models.mappers.base_mapper import BaseMapper
from automl_package.utils.numerics import create_bins, calculate_binned_stats


class LookupTableEntries(Enum):
    """Enum for lookup table entries."""

    KEYS = "keys"
    VALUES = "values"
    VARIANCES = "variances"


class LookupMapper(BaseMapper):
    """Mapper that uses a lookup table to map probabilities to regression values."""

    def __init__(self, mapper_type: MapperType, uncertainty_method: UncertaintyMethod | None = None, **kwargs: Any) -> None:
        """Initializes the LookupMapper."""
        super().__init__(mapper_type=mapper_type, uncertainty_method=uncertainty_method)

        if self.uncertainty_method == UncertaintyMethod.BINNED_RESIDUAL_STD:
            logger.warning(f"Uncertainty method '{self.uncertainty_method.value}' is not directly supported by {self.__class__.__name__}. Falling back to '{UncertaintyMethod.PROBABILISTIC.value}'.")
            self.uncertainty_method = UncertaintyMethod.PROBABILISTIC

        self.lookup_table = None
        self.n_partitions_min = kwargs.get("n_partitions_min", 5)
        self.n_partitions_max = kwargs.get("n_partitions_max", np.inf)
        self._residual_variance = 0.0

    def _fit(self, probas: np.ndarray, y_original: np.ndarray, **kwargs: Any) -> dict[str, Any]:  # noqa: ARG002
        min_partitions = max(self.n_partitions_min, len(probas) // 2000)
        max_partitions = min(self.n_partitions_max, len(probas) // 100)
        num_partitions = max(1, min_partitions, max_partitions)

        aggregation_func = np.mean if self._mapper_type == MapperType.LOOKUP_MEAN else np.median
        aggregations = {"values": aggregation_func, "variances": np.var}

        bin_edges, stats_lookups = calculate_binned_stats(
            probas=probas, values=y_original, n_bins=num_partitions, aggregations=aggregations, min_value=0.0, max_value=1.0
        )

        self.lookup_table = {
            LookupTableEntries.KEYS: bin_edges[1:],
            LookupTableEntries.VALUES: stats_lookups["values"],
            LookupTableEntries.VARIANCES: stats_lookups["variances"],
        }

        # For constant uncertainty, calculate overall residual variance
        y_pred_train = self._predict(probas)
        self._residual_variance = np.var(y_original - y_pred_train)
        if np.isnan(self._residual_variance):
            self._residual_variance = 0.0

        return {}

    def _fit_empty(self) -> None:
        self.lookup_table = {LookupTableEntries.KEYS: np.array([0.0, 1.0]), LookupTableEntries.VALUES: np.array([0.0, 0.0]), LookupTableEntries.VARIANCES: np.array([0.0, 0.0])}
        self._residual_variance = 0.0

    def _find_indices(self, probas_new: np.ndarray) -> np.ndarray:
        keys = self.lookup_table[LookupTableEntries.KEYS]
        if len(keys) == 0:
            indices = np.zeros_like(probas_new, dtype=int)
        else:
            _, indices = create_bins(data=probas_new, unique_bin_edges=np.insert(keys, 0, 0))
            # Clip indices to be within the valid range of the lookup table keys
            indices = np.clip(indices, 0, len(keys) - 1)
        return indices

    def _predict(self, probas_new: np.ndarray) -> np.ndarray:
        """Predicts the mapped values.

        Args:
            probas_new (np.ndarray): The new probabilities.

        Returns:
            np.ndarray: The predicted values.
        """
        if self.lookup_table is None:
            raise RuntimeError("Lookup mapper has not been fitted yet.")

        values = self.lookup_table[LookupTableEntries.VALUES]
        indices = self._find_indices(probas_new=probas_new).astype(np.int64)
        return values[indices].flatten()

    def _predict_variance(self, probas_new: np.ndarray) -> np.ndarray:
        """Predicts the variance of the mapped values.

        Args:
            probas_new (np.ndarray): The new probabilities.

        Returns:
            np.ndarray: The predicted variances.
        """
        if self.lookup_table is None:
            raise RuntimeError("Lookup mapper has not been fitted yet.")

        if self.uncertainty_method == UncertaintyMethod.CONSTANT:
            return np.full(probas_new.shape[0], self._residual_variance)
        elif self.uncertainty_method == UncertaintyMethod.PROBABILISTIC:
            # This is the mapper's own internal probabilistic method
            variances = self.lookup_table[LookupTableEntries.VARIANCES]
            indices = self._find_indices(probas_new=probas_new).astype(np.int64)
            return variances[indices]
        else:
            raise NotImplementedError(f"Uncertainty method '{self.uncertainty_method}' is not directly supported by LookupMapper's internal _predict_variance method.")
