"""Lookup mapper for probability mapping."""

from enum import Enum
from typing import Any

import numpy as np

from automl_package.enums import MapperType
from automl_package.models.mappers.base_mapper import BaseMapper
from automl_package.utils.numerics import create_bins


class LookupTableEntries(Enum):
    """Enum for lookup table entries."""

    KEYS = "keys"
    VALUES = "values"
    VARIANCES = "variances"


class LookupMapper(BaseMapper):
    """Mapper that uses a lookup table to map probabilities to regression values."""

    def __init__(self, mapper_type: MapperType, **kwargs: Any) -> None:
        """Initializes the LookupMapper."""
        self.mapper_type = mapper_type
        self.lookup_table = None
        self.n_partitions_min = kwargs.get("n_partitions_min", 5)
        self.n_partitions_max = kwargs.get("n_partitions_max", np.inf)

    def _fit(self, probas: np.ndarray, y_original: np.ndarray) -> None:
        min_partitions = max(self.n_partitions_min, len(probas) // 2000)
        max_partitions = min(self.n_partitions_max, len(probas) // 100)
        num_partitions = max(1, min_partitions, max_partitions)

        bin_edges, bin_indices = create_bins(data=probas, n_bins=num_partitions, min_value=0.0, max_value=1.0)
        bin_edges = bin_edges[1:]

        temp_lookup_keys = bin_edges
        temp_lookup_values = []
        temp_lookup_variances = []

        for i in range(len(bin_edges)):
            mask = bin_indices == i
            partition_y = y_original[mask]
            # Handle empty partitions by taking the overall mean/median and variance
            if len(partition_y) == 0:
                partition_y = y_original
            expected_y = np.mean(partition_y) if self.mapper_type == MapperType.LOOKUP_MEAN else np.median(partition_y)
            partition_variance = np.var(partition_y)

            if np.isnan(partition_variance):
                partition_variance = 0.0

            temp_lookup_values.append(expected_y)
            temp_lookup_variances.append(partition_variance)

        self.lookup_table = {
            LookupTableEntries.KEYS: np.array(temp_lookup_keys),
            LookupTableEntries.VALUES: np.array(temp_lookup_values),
            LookupTableEntries.VARIANCES: np.array(temp_lookup_variances),
        }

    def _fit_empty(self) -> None:
        self.lookup_table = {
            LookupTableEntries.KEYS: np.array([0.0, 1.0]),
            LookupTableEntries.VALUES: np.array([0.0, 0.0]),
            LookupTableEntries.VARIANCES: np.array([0.0, 0.0]),
        }

    def _find_indices(self, probas_new: np.ndarray) -> np.ndarray:
        keys = self.lookup_table[LookupTableEntries.KEYS]
        if len(keys) == 0:
            indices = np.zeros_like(probas_new)
        else:
            _, indices = create_bins(data=probas_new, unique_bin_edges=np.insert(keys, 0, 0))
        return indices

    def predict(self, probas_new: np.ndarray) -> np.ndarray:
        """Predicts the mapped values.

        Args:
            probas_new (np.ndarray): The new probabilities.

        Returns:
            np.ndarray: The predicted values.
        """
        if self.lookup_table is None:
            raise RuntimeError("Lookup mapper has not been fitted yet.")

        values = self.lookup_table[LookupTableEntries.VALUES]
        indices = self._find_indices(probas_new=probas_new)
        return values[indices].flatten()

    def predict_variance(self, probas_new: np.ndarray) -> np.ndarray:
        """Predicts the variance of the mapped values.

        Args:
            probas_new (np.ndarray): The new probabilities.

        Returns:
            np.ndarray: The predicted variances.
        """
        if self.lookup_table is None:
            raise RuntimeError("Lookup mapper has not been fitted yet.")

        variances = self.lookup_table[LookupTableEntries.VARIANCES]
        indices = self._find_indices(probas_new=probas_new)
        return variances[indices]
