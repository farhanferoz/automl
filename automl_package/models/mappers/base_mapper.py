"""Base mapper for probability mapping."""

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np

from automl_package.enums import MapperType, UncertaintyMethod
from automl_package.utils.numerics import create_bins, calculate_binned_stats


class BaseMapper(ABC):
    """Base class for probability mappers."""

    def __init__(self, mapper_type: MapperType | None = None, uncertainty_method: Optional[UncertaintyMethod] = None) -> None:
        """Initializes the BaseMapper."""
        self._constant_prediction_value: float | None = None
        self._constant_variance_value: float | None = None
        self._mapper_type = mapper_type
        self.uncertainty_method = uncertainty_method
        self._bypass_sorting = False
        self._binned_uncertainty_lookup: dict[int, float] = {}
        self._binned_uncertainty_edges: np.ndarray | None = None

    def fit(self, probas: np.ndarray, y_original: np.ndarray, **kwargs: Any) -> Any:
        """Fits the mapper.

        Args:
            probas (np.ndarray): The probabilities.
            y_original (np.ndarray): The original y values.
            **kwargs: Additional arguments for the fit method of subclasses.
        """
        if probas.ndim == 2 and probas.shape[1] == 1:
            probas = probas.flatten()

        if len(probas) != len(y_original):
            raise ValueError("Lengths of probabilities and original y must match.")

        if len(probas) == 0:
            return self._fit_empty()
        # Edge case: If all probabilities are the same, set a constant prediction.
        unique_probas = np.unique(probas)
        if len(unique_probas) == 1:
            if self._mapper_type == MapperType.LOOKUP_MEDIAN:
                self._constant_prediction_value = np.median(y_original)
            else:
                assert self._mapper_type == MapperType.LOOKUP_MEAN
                self._constant_prediction_value = np.mean(y_original)
            self._constant_variance_value = np.var(y_original)
            return None  # Skip the actual fitting process
        if self._bypass_sorting:
            return self._fit(probas, y_original, **kwargs)
        sort_indices = np.argsort(probas)
        sorted_probas = probas[sort_indices]
        sorted_y_original = y_original[sort_indices]
        return self._fit(sorted_probas, sorted_y_original, **kwargs)

    def calibrate_uncertainty(self, X: np.ndarray, y: np.ndarray, n_bins: int = 10) -> None:
        """Calibrates the uncertainty model for the mapper.

        Args:
            X (np.ndarray): The input features (probabilities).
            y (np.ndarray): The true target values.
            n_bins (int): The number of bins to use for calibration.
        """
        if self.uncertainty_method == UncertaintyMethod.BINNED_RESIDUAL_STD:
            predictions = self.predict(X)
            residuals = y - predictions

            bin_edges, stats = calculate_binned_stats(
                probas=X, values=residuals, n_bins=n_bins, aggregations={"std": np.std}
            )

            self._binned_uncertainty_edges = bin_edges
            self._binned_uncertainty_lookup = stats["std"]

    @abstractmethod
    def _fit(self, probas: np.ndarray, y_original: np.ndarray, **kwargs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def _fit_empty(self) -> None:
        raise NotImplementedError

    def predict(self, probas_new: np.ndarray) -> np.ndarray:
        """Predicts the mapped values.

        Args:
            probas_new (np.ndarray): The new probabilities.

        Returns:
            np.ndarray: The predicted values.
        """
        return np.full(probas_new.shape[0], self._constant_prediction_value) if self._constant_prediction_value is not None else self._predict(probas_new)

    @abstractmethod
    def _predict(self, probas_new: np.ndarray) -> np.ndarray:
        """Subclass-specific prediction logic."""
        raise NotImplementedError

    def predict_variance(self, probas_new: np.ndarray) -> np.ndarray:
        """Predicts the variance of the mapped values.

        Args:
            probas_new (np.ndarray): The new probabilities.

        Returns:
            np.ndarray: The predicted variances.
        """
        if self._constant_variance_value is not None:
            return np.full(probas_new.shape[0], self._constant_variance_value)

        if self.uncertainty_method == UncertaintyMethod.BINNED_RESIDUAL_STD:
            if self._binned_uncertainty_edges is None or self._binned_uncertainty_lookup is None:
                raise RuntimeError("Uncertainty model has not been calibrated. Call `calibrate_uncertainty` first.")

            # Find which bin each new probability falls into
            bin_indices = np.searchsorted(self._binned_uncertainty_edges[1:], probas_new, side="left")
            bin_indices = np.clip(bin_indices, 0, len(self._binned_uncertainty_edges) - 2)

            # Look up the standard deviation and square it to get variance
            stds = self._binned_uncertainty_lookup[bin_indices]
            variances = np.nan_to_num(stds, nan=np.nanmean(self._binned_uncertainty_lookup)) ** 2
            return variances

        return self._predict_variance(probas_new)

    @abstractmethod
    def _predict_variance(self, probas_new: np.ndarray) -> np.ndarray:
        """Subclass-specific variance prediction logic."""
        raise NotImplementedError

    def predict_mean_and_variance_per_class(self, probas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predicts the mean and variance for each class.

        Args:
            probas (np.ndarray): The probabilities for each class.

        Returns:
            A tuple containing two arrays: the means and the variances for each class.
        """
        raise NotImplementedError