from abc import ABC, abstractmethod

import numpy as np


class BaseMapper(ABC):
    def fit(self, probas: np.ndarray, y_original: np.ndarray) -> None:
        if probas.ndim == 2 and probas.shape[1] == 1:
            probas = probas.flatten()

        if len(probas) != len(y_original):
            raise ValueError("Lengths of probabilities and original y must match.")

        if len(probas) == 0:
            self._fit_empty(probas, y_original)
            return

        sort_indices = np.argsort(probas)
        sorted_probas = probas[sort_indices]
        sorted_y_original = y_original[sort_indices]

        self._fit(sorted_probas, sorted_y_original)

    @abstractmethod
    def _fit(self, probas: np.ndarray, y_original: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def _fit_empty(self, probas: np.ndarray, y_original: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, probas_new: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def predict_variance(self, probas_new: np.ndarray) -> np.ndarray:
        raise NotImplementedError
