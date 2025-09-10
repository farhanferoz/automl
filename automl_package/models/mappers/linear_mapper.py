"""Linear mapper for probability mapping."""

from typing import Any

import numpy as np
from sklearn.linear_model import LinearRegression

from automl_package.enums import MapperType, UncertaintyMethod
from automl_package.logger import logger
from automl_package.models.mappers.base_mapper import BaseMapper


class LinearMapper(BaseMapper):
    """Linear mapper for probability mapping."""

    def __init__(self, uncertainty_method: UncertaintyMethod = UncertaintyMethod.CONSTANT) -> None:
        """Initializes the LinearMapper."""
        super().__init__(MapperType.LINEAR, uncertainty_method)
        self.model = None
        self._linear_mapper_residual_variance = 0.0

    def _fit(self, probas: np.ndarray, y_original: np.ndarray, **kwargs: Any) -> dict[str, Any]:  # noqa: ARG002
        self.model = LinearRegression()
        self.model.fit(probas.reshape(-1, 1), y_original)
        y_pred_train = self.model.predict(probas.reshape(-1, 1))
        _linear_mapper_residual_variance = np.var(y_original - y_pred_train)
        self._linear_mapper_residual_variance = 0.0 if np.isnan(_linear_mapper_residual_variance) else _linear_mapper_residual_variance
        return {}

    def _fit_empty(self) -> None:
        self.model = LinearRegression()
        self.model.fit(np.array([[0], [1]]), np.array([0, 0]))
        self._linear_mapper_residual_variance = 0.0

    def _predict(self, probas_new: np.ndarray) -> np.ndarray:
        """Predicts the mapped values.

        Args:
            probas_new (np.ndarray): The new probabilities.

        Returns:
            np.ndarray: The predicted values.
        """
        if self.model is None:
            raise RuntimeError("Linear mapper has not been fitted yet.")
        return self.model.predict(probas_new.reshape(-1, 1)).flatten()

    def _predict_variance(self, probas_new: np.ndarray) -> np.ndarray:
        """Predicts the variance of the mapped values.

        Args:
            probas_new (np.ndarray): The new probabilities.

        Returns:
            np.ndarray: The predicted variances.
        """
        if self.model is None:
            raise RuntimeError("Linear mapper has not been fitted yet.")
        if self.uncertainty_method != UncertaintyMethod.CONSTANT:
            logger.warning("LinearMapper only supports CONSTANT uncertainty. Falling back to constant variance.")
        return np.full(probas_new.shape[0], self._linear_mapper_residual_variance)
