"""Base mapper for probability mapping."""

from abc import ABC, abstractmethod

import numpy as np


class BaseMapper(ABC):
    """Base class for probability mappers."""

    def fit(self, probas: np.ndarray, y_original: np.ndarray) -> None:
        """Fits the mapper.

        Args:
            probas (np.ndarray): The probabilities.
            y_original (np.ndarray): The original y values.
        """
        if probas.ndim == 2 and probas.shape[1] == 1:
            probas = probas.flatten()

        if len(probas) != len(y_original):
            raise ValueError("Lengths of probabilities and original y must match.")

        if len(probas) == 0:
            self._fit_empty()
            return

        sort_indices = np.argsort(probas)
        sorted_probas = probas[sort_indices]
        sorted_y_original = y_original[sort_indices]

        self._fit(sorted_probas, sorted_y_original)

    @abstractmethod
    def _fit(self, probas: np.ndarray, y_original: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def _fit_empty(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, probas_new: np.ndarray) -> np.ndarray:
        """Predicts the mapped values.

        Args:
            probas_new (np.ndarray): The new probabilities.

        Returns:
            np.ndarray: The predicted values.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_variance(self, probas_new: np.ndarray) -> np.ndarray:
        """Predicts the variance of the mapped values.

        Args:
            probas_new (np.ndarray): The new probabilities.

        Returns:
            np.ndarray: The predicted variances.
        """
        raise NotImplementedError
