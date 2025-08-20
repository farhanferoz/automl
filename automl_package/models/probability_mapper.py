from typing import Any

import numpy as np

from automl_package.enums import MapperType
from automl_package.models.mappers.linear_mapper import LinearMapper
from automl_package.models.mappers.lookup_mapper import LookupMapper
from automl_package.models.mappers.spline_mapper import SplineMapper


class ClassProbabilityMapper:
    """Maps classification probabilities for a single class back to original regression values.
    This class acts as a factory for different mapping strategies.
    """

    def __init__(self, mapper_type: MapperType = MapperType.LINEAR, **kwargs: Any) -> None:
        """Initializes the ClassProbabilityMapper.

        Args:
            mapper_type (MapperType): The type of mapping strategy to use.
            **kwargs: Additional parameters specific to the mapper type.
        """
        self.mapper_type = mapper_type
        self.mapper = self._create_mapper(**kwargs)

    def _create_mapper(self, **kwargs):
        if self.mapper_type == MapperType.LINEAR:
            return LinearMapper(**kwargs)
        if self.mapper_type in [MapperType.LOOKUP_MEAN, MapperType.LOOKUP_MEDIAN]:
            return LookupMapper(mapper_type=self.mapper_type, **kwargs)
        if self.mapper_type == MapperType.SPLINE:
            return SplineMapper(**kwargs)
        raise ValueError(f"Unsupported mapper_type: {self.mapper_type.value}")

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
