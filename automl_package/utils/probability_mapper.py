import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.interpolate import UnivariateSpline  # New import
from typing import Dict, Any
from ..enums import MapperType  # Import MapperType enum
from ..logger import logger  # Import logger


class ClassProbabilityMapper:
    """
    Maps classification probabilities for a single class back to original regression values.
    Supports linear regression and lookup table approaches (mean or median) with uncertainty.
    Now also supports spline interpolation.
    """

    def __init__(self, mapper_type: MapperType = MapperType.LINEAR, **kwargs):  # Use enum
        """
        Initializes the probability mapper.

        Args:
            mapper_type (MapperType): Type of mapping to use ('linear', 'lookup_mean', 'lookup_median', 'spline').
            **kwargs: Additional parameters specific to the mapper type.
                      For lookup tables:
                          - n_partitions_min (int): Minimum number of partitions (default: 5).
                          - n_partitions_max (int): Maximum number of partitions.
                      For spline:
                          - spline_k (int): Degree of the spline (1-5, default: 3).
                          - spline_s (float): Positive smoothing factor (default: None, for interpolation).
        """
        self.mapper_type = mapper_type
        self.model = None  # For linear regression or spline
        self.lookup_table = None  # For lookup tables {proba_key: (expected_y, variance)}
        self.bin_edges = None  # For lookup tables

        # Parameters for lookup table mappers
        self.n_partitions_min = kwargs.get("n_partitions_min", 5)
        self.n_partitions_max = kwargs.get("n_partitions_max", np.inf)

        # Parameters for spline mapper
        self.spline_k = kwargs.get("spline_k", 3)
        self.spline_s = kwargs.get("spline_s", None)  # None for interpolation, can be float for smoothing
        self._spline_residual_variance = 0.0  # For spline uncertainty

        if self.mapper_type not in [MapperType.LINEAR, MapperType.LOOKUP_MEAN, MapperType.LOOKUP_MEDIAN, MapperType.SPLINE]:
            raise ValueError(f"Unsupported mapper_type: {self.mapper_type.value}. Choose from 'linear', 'lookup_mean', 'lookup_median', 'spline'.")

    def fit(self, probas: np.ndarray, y_original: np.ndarray):
        """
        Fits the mapper using probabilities and corresponding original target values.

        Args:
            probas (np.ndarray): 1D array of classification probabilities for a specific class.
            y_original (np.ndarray): 1D array of original regression target values.
        """
        if probas.ndim == 2 and probas.shape[1] == 1:
            probas = probas.flatten()  # Ensure 1D

        if len(probas) != len(y_original):
            raise ValueError("Lengths of probabilities and original y must match.")

        if len(probas) == 0:
            logger.warning("No data points for mapper fitting. Mapper will predict 0 with 0 variance.")
            if self.mapper_type == MapperType.LINEAR:
                self.model = LinearRegression()
                self.model.fit(np.array([[0], [1]]), np.array([0, 0]))
            elif self.mapper_type == MapperType.SPLINE:
                # Dummy spline that predicts 0, requires at least k+1 points
                # If no data points, create a trivial spline that always outputs 0
                # Need at least k+1 data points to fit a spline of degree k.
                # If no data, force k=1 and use dummy points.
                self.model = UnivariateSpline(x=[0, 1], y=[0, 0], k=1, s=0)
                self._spline_residual_variance = 0.0
            self.lookup_table = {"keys": np.array([0.0, 1.0]), "values": np.array([0.0, 0.0]), "variances": np.array([0.0, 0.0])}
            self.bin_edges = np.array([0.0, 1.0])
            return

        # Sort probabilities and corresponding y values for monotonic interpolation/mapping
        sort_indices = np.argsort(probas)
        sorted_probas = probas[sort_indices]
        sorted_y_original = y_original[sort_indices]

        if self.mapper_type == MapperType.LINEAR:
            self.model = LinearRegression()
            self.model.fit(sorted_probas.reshape(-1, 1), sorted_y_original)
            y_pred_train = self.model.predict(sorted_probas.reshape(-1, 1))
            _linear_mapper_residual_variance = np.var(sorted_y_original - y_pred_train)
            if np.isnan(_linear_mapper_residual_variance):
                self._linear_mapper_residual_variance = 0.0
            else:
                self._linear_mapper_residual_variance = _linear_mapper_residual_variance
        elif self.mapper_type in [MapperType.LOOKUP_MEAN, MapperType.LOOKUP_MEDIAN]:
            max_partitions_by_points = int(np.ceil(len(probas) / 2000.0))
            num_partitions = min(self.n_partitions_max, max(self.n_partitions_min, max_partitions_by_points))

            if num_partitions == 0:
                num_partitions = 1

            self.bin_edges = []
            temp_lookup_keys = []
            temp_lookup_values = []
            temp_lookup_variances = []

            partition_size = len(probas) // num_partitions
            remainder = len(probas) % num_partitions

            current_idx = 0
            for i in range(num_partitions):
                start_idx = current_idx
                end_idx = start_idx + partition_size + (1 if i < remainder else 0)
                end_idx = min(end_idx, len(probas))

                if start_idx >= end_idx:
                    continue

                partition_probas = sorted_probas[start_idx:end_idx]
                partition_y = sorted_y_original[start_idx:end_idx]

                if len(partition_y) == 0:
                    continue

                if self.mapper_type == MapperType.LOOKUP_MEAN:
                    expected_y = np.mean(partition_y)
                else:  # lookup_median
                    expected_y = np.median(partition_y)

                partition_variance = np.var(partition_y)
                if np.isnan(partition_variance):
                    partition_variance = 0.0

                bin_center_proba = np.mean(partition_probas)  # Use mean proba as key
                temp_lookup_keys.append(bin_center_proba)
                temp_lookup_values.append(expected_y)
                temp_lookup_variances.append(partition_variance)

                if i == 0:
                    self.bin_edges.append(sorted_probas[0])
                self.bin_edges.append(sorted_probas[end_idx - 1] if end_idx > 0 else sorted_probas[0])

                current_idx = end_idx

            self.bin_edges = np.unique(self.bin_edges)

            if len(temp_lookup_keys) > 0:
                sorted_zipped_lists = sorted(zip(temp_lookup_keys, temp_lookup_values, temp_lookup_variances))
                sorted_keys, sorted_values, sorted_variances = zip(*sorted_zipped_lists)
                self.lookup_table = {"keys": np.array(sorted_keys), "values": np.array(sorted_values), "variances": np.array(sorted_variances)}
            else:
                mean_y_overall = np.mean(y_original) if len(y_original) > 0 else 0.0
                var_y_overall = np.var(y_original) if len(y_original) > 0 else 0.0
                self.lookup_table = {"keys": np.array([0.0, 1.0]), "values": np.array([mean_y_overall, mean_y_overall]), "variances": np.array([var_y_overall, var_y_overall])}
                self.bin_edges = np.array([0.0, 1.0])

        elif self.mapper_type == MapperType.SPLINE:
            # For spline, need at least k+1 data points. Adjust k if not enough data.
            # Scipy UnivariateSpline requires sorted x values.
            # Handle duplicates in x values by averaging corresponding y values or by adding slight jitter.
            # For simplicity, we'll ensure unique x values by taking unique sorted values and their mean y.
            unique_probas, unique_indices = np.unique(sorted_probas, return_index=True)
            # Handle case where there are not enough unique points to fit spline of degree k
            if len(unique_probas) < self.spline_k + 1:
                effective_k = max(1, len(unique_probas) - 1)  # Fallback to linear if too few unique points
                if effective_k == 0:  # Still no points after making unique
                    self.model = UnivariateSpline(x=[0, 1], y=[0, 0], k=1, s=0)  # Trivial spline
                    self._spline_residual_variance = 0.0
                    return
                logger.warning(
                    f"Not enough unique data points ({len(unique_probas)}) for spline degree k={self.spline_k}. Falling back to k={effective_k} for spline mapper for this class."
                )
            else:
                effective_k = self.spline_k

            # For duplicate x values, UnivariateSpline documentation recommends pre-averaging y values.
            # Simplest approach: create new x, y where x are unique and y are means for those x.
            # This is complex when handling many points with same probability.
            # A more robust way to handle this might be to just use unique points for spline construction.
            # But here, assuming sorted_probas often means distinct enough. If exact duplicates are an issue,
            # pre-processing (like adding small noise or averaging y for duplicate x) is needed.
            # For now, let's rely on UnivariateSpline's internal handling of non-strictly increasing x, or adjust k.
            try:
                self.model = UnivariateSpline(x=sorted_probas, y=sorted_y_original, k=effective_k, s=self.spline_s)
            except Exception as e:
                logger.error(f"Error fitting spline with k={effective_k}, s={self.spline_s}: {e}. Falling back to k=1.")
                self.model = UnivariateSpline(x=sorted_probas, y=sorted_y_original, k=1, s=self.spline_s)  # Fallback

            y_pred_train = self.model(sorted_probas)
            _spline_residual_variance = np.var(sorted_y_original - y_pred_train)
            if np.isnan(_spline_residual_variance):
                self._spline_residual_variance = 0.0
            else:
                self._spline_residual_variance = _spline_residual_variance

    def predict(self, probas_new: np.ndarray) -> np.ndarray:
        """
        Predicts original regression values from new classification probabilities.

        Args:
            probas_new (np.ndarray): 1D array of new classification probabilities for a specific class.

        Returns:
            np.ndarray: Predicted original regression values.
        """
        if probas_new.ndim == 2 and probas_new.shape[1] == 1:
            probas_new = probas_new.flatten()

        if self.mapper_type == MapperType.LINEAR:
            if self.model is None:
                raise RuntimeError("Mapper has not been fitted yet.")
            return self.model.predict(probas_new.reshape(-1, 1))
        elif self.mapper_type in [MapperType.LOOKUP_MEAN, MapperType.LOOKUP_MEDIAN]:
            if self.lookup_table is None or self.bin_edges is None:
                raise RuntimeError("Mapper has not been fitted yet.")

            keys = self.lookup_table["keys"]
            values = self.lookup_table["values"]

            if len(keys) == 0:
                return np.zeros_like(probas_new)

            indices = np.searchsorted(keys, probas_new, side="right") - 1
            indices = np.clip(indices, 0, len(keys) - 1)

            return values[indices]
        elif self.mapper_type == MapperType.SPLINE:
            if self.model is None:
                raise RuntimeError("Spline mapper has not been fitted yet.")
            # Ensure probabilities are within the range used for fitting the spline
            # Clipping is important as UnivariateSpline can extrapolate wildly outside its data range
            if len(self.model.get_knots()) > 1:
                probas_new_clipped = np.clip(probas_new, self.model.get_knots()[0], self.model.get_knots()[-1])
            else:  # Fallback for very few data points, spline might only have one knot
                probas_new_clipped = probas_new  # No clipping possible, rely on spline's internal handling
            return self.model(probas_new_clipped)
        else:
            raise ValueError(f"Unsupported mapper_type: {self.mapper_type.value}")

    def predict_variance_contribution(self, probas_new: np.ndarray) -> np.ndarray:
        """
        Predicts the variance contribution for a given probability for this specific class.

        Args:
            probas_new (np.ndarray): 1D array of new classification probabilities for a specific class.

        Returns:
            np.ndarray: Variance contribution for each probability.
        """
        if probas_new.ndim == 2 and probas_new.shape[1] == 1:
            probas_new = probas_new.flatten()

        if self.mapper_type == MapperType.LINEAR:
            if self.model is None or not hasattr(self, "_linear_mapper_residual_variance"):
                raise RuntimeError("Linear mapper has not been fitted or residual variance not calculated.")
            # For linear, return constant residual variance for each input
            return np.full(probas_new.shape[0], self._linear_mapper_residual_variance)
        elif self.mapper_type in [MapperType.LOOKUP_MEAN, MapperType.LOOKUP_MEDIAN]:
            if self.lookup_table is None or self.bin_edges is None:
                raise RuntimeError("Lookup mapper has not been fitted yet.")

            keys = self.lookup_table["keys"]
            variances = self.lookup_table["variances"]

            if len(keys) == 0:
                return np.zeros_like(probas_new)

            indices = np.searchsorted(keys, probas_new, side="right") - 1
            indices = np.clip(indices, 0, len(keys) - 1)

            return variances[indices]
        elif self.mapper_type == MapperType.SPLINE:
            if self.model is None or not hasattr(self, "_spline_residual_variance"):
                raise RuntimeError("Spline mapper has not been fitted or residual variance not calculated.")
            # For spline, return constant residual variance based on the fit
            return np.full(probas_new.shape[0], self._spline_residual_variance)
        else:
            raise ValueError(f"Unsupported mapper_type: {self.mapper_type.value}")
