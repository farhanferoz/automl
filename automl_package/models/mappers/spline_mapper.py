import numpy as np
from scipy.interpolate import UnivariateSpline

from automl_package.logger import logger
from automl_package.models.mappers.base_mapper import BaseMapper


class SplineMapper(BaseMapper):
    def __init__(self, **kwargs):
        self.model = None
        self.spline_k = kwargs.get("spline_k", 3)
        self.spline_s = kwargs.get("spline_s")
        self._spline_residual_variance = 0.0

    def _fit(self, probas: np.ndarray, y_original: np.ndarray) -> None:
        unique_probas = np.unique(probas)
        effective_k = self.spline_k
        if len(unique_probas) < self.spline_k + 1:
            effective_k = max(1, len(unique_probas) - 1)
            if effective_k == 0:
                self.model = UnivariateSpline(x=[0, 1], y=[0, 0], k=1, s=0)
                self._spline_residual_variance = 0.0
                logger.warning("Not enough unique data points to fit a non-trivial spline. Using a trivial linear spline.")
                return
            logger.warning(f"Not enough unique data points ({len(unique_probas)}) for spline degree k={self.spline_k}. Falling back to k={effective_k}.")

        try:
            self.model = UnivariateSpline(x=probas, y=y_original, k=effective_k, s=self.spline_s)
        except Exception as e:
            logger.error(f"Error fitting spline with k={effective_k}, s={self.spline_s}: {e}. Falling back to k=1.")
            self.model = UnivariateSpline(x=probas, y=y_original, k=1, s=self.spline_s)

        y_pred_train = self.model(probas)
        _spline_residual_variance = np.var(y_original - y_pred_train)
        self._spline_residual_variance = 0.0 if np.isnan(_spline_residual_variance) else _spline_residual_variance

    def _fit_empty(self, probas: np.ndarray, y_original: np.ndarray) -> None:
        self.model = UnivariateSpline(x=[0, 1], y=[0, 0], k=1, s=0)
        self._spline_residual_variance = 0.0

    def predict(self, probas_new: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Spline mapper has not been fitted yet.")
        probas_new_clipped = np.clip(probas_new, self.model.get_knots()[0], self.model.get_knots()[-1]) if len(self.model.get_knots()) > 1 else probas_new
        return self.model(probas_new_clipped).flatten()

    def predict_variance(self, probas_new: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Spline mapper has not been fitted yet.")
        return np.full(probas_new.shape[0], self._spline_residual_variance)
