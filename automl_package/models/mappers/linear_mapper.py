import numpy as np
from sklearn.linear_model import LinearRegression

from automl_package.models.mappers.base_mapper import BaseMapper


class LinearMapper(BaseMapper):
    def __init__(self, **kwargs):
        self.model = None
        self._linear_mapper_residual_variance = 0.0

    def _fit(self, probas: np.ndarray, y_original: np.ndarray) -> None:
        self.model = LinearRegression()
        self.model.fit(probas.reshape(-1, 1), y_original)
        y_pred_train = self.model.predict(probas.reshape(-1, 1))
        _linear_mapper_residual_variance = np.var(y_original - y_pred_train)
        if np.isnan(_linear_mapper_residual_variance):
            self._linear_mapper_residual_variance = 0.0
        else:
            self._linear_mapper_residual_variance = _linear_mapper_residual_variance

    def _fit_empty(self, probas: np.ndarray, y_original: np.ndarray) -> None:
        self.model = LinearRegression()
        self.model.fit(np.array([[0], [1]]), np.array([0, 0]))
        self._linear_mapper_residual_variance = 0.0

    def predict(self, probas_new: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Linear mapper has not been fitted yet.")
        return self.model.predict(probas_new.reshape(-1, 1)).flatten()

    def predict_variance(self, probas_new: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Linear mapper has not been fitted yet.")
        return np.full(probas_new.shape[0], self._linear_mapper_residual_variance)
