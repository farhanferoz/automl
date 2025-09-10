"""Probability mapper for regression tasks."""

from typing import Any

import numpy as np

from automl_package.enums import MapperType, UncertaintyMethod
from automl_package.models.mappers.linear_mapper import LinearMapper
from automl_package.models.mappers.lookup_mapper import LookupMapper
from automl_package.models.mappers.spline_mapper import SplineMapper


class ClassProbabilityMapper:
    """Maps classification probabilities for a single class back to original regression values."""

    def __init__(self, mapper_type: MapperType = MapperType.LINEAR, **kwargs: Any) -> None:
        """Initializes the ClassProbabilityMapper.

        Args:
            mapper_type (MapperType): The type of mapping strategy to use.
            **kwargs: Additional parameters specific to the mapper type.
        """
        self.mapper_type = mapper_type
        self.uncertainty_method = kwargs.get("uncertainty_method", UncertaintyMethod.CONSTANT)
        self.mapper = self._create_mapper(**kwargs)

    def _filter_kwargs(self, mapper_type: MapperType, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Filters kwargs for the specific mapper type."""
        if mapper_type == MapperType.SPLINE:
            return {k: v for k, v in kwargs.items() if k in ["spline_k", "spline_s"]}
        if mapper_type in [MapperType.LOOKUP_MEAN, MapperType.LOOKUP_MEDIAN]:
            return {k: v for k, v in kwargs.items() if k in ["lookup_n_partitions"]}
        return {}

    def _create_mapper(self, **kwargs: Any) -> Any:
        filtered_kwargs = self._filter_kwargs(self.mapper_type, kwargs)
        filtered_kwargs["uncertainty_method"] = self.uncertainty_method

        if self.mapper_type == MapperType.LINEAR:
            return LinearMapper(**filtered_kwargs)
        if self.mapper_type in [MapperType.LOOKUP_MEAN, MapperType.LOOKUP_MEDIAN]:
            return LookupMapper(mapper_type=self.mapper_type, **filtered_kwargs)
        if self.mapper_type == MapperType.SPLINE:
            return SplineMapper(**filtered_kwargs)
        raise ValueError(f"Unsupported mapper_type: {self.mapper_type.label}")

    def fit(self, probas: np.ndarray, y_original: np.ndarray) -> None:
        """Fits the mapping from probabilities to corresponding original target values.

        Args:
            probas (np.ndarray): 1D array of classification probabilities for a specific class.
            y_original (np.ndarray): 1D array of original regression target values.
        """
        self.mapper.fit(probas, y_original)

    def predict(self, probas_new: np.ndarray) -> np.ndarray:
        """Predicts original regression values from new classification probabilities.

        Args:
            probas_new (np.ndarray): 1D array of new classification probabilities for a specific class.

        Returns:
            np.ndarray: Predicted original regression values.
        """
        return self.mapper.predict(probas_new)

    def predict_variance_contribution(self, probas_new: np.ndarray) -> np.ndarray:
        """Predicts the variance contribution for a given probability for this specific class.

        Args:
            probas_new (np.ndarray): 1D array of new classification probabilities for a specific class.

        Returns:
            np.ndarray: Variance contribution for each probability.
        """
        return self.mapper.predict_variance(probas_new)
