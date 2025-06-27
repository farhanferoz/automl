import xgboost as xgb
import lightgbm as lgb
import numpy as np
from .base import BaseModel
from typing import Dict, Any


class XGBoostModel(BaseModel):
    """XGBoost model wrapper."""

    def __init__(self, objective: str = "reg:squarederror", eval_metric: str = "rmse", **kwargs):
        super().__init__(**kwargs)
        self.objective = objective
        self.eval_metric = eval_metric
        self.model = None
        # Determine if it's a regression model based on objective
        self._is_regression_model = self.objective in ["reg:squarederror", "reg:logistic", "count:poisson", "reg:gamma", "reg:tweedie", "reg:quantile"]
        self._train_residual_std = 0.0  # For regression uncertainty

    @property
    def name(self) -> str:
        return "XGBoost"

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Handle classification objectives for XGBoost based on objective string
        if self.objective in ["binary:logistic", "multi:softmax", "multi:softprob"]:
            # For classification, use XGBClassifier
            self.model = xgb.XGBClassifier(objective=self.objective, eval_metric=self.eval_metric, **self.params)
        else:
            # For regression, use XGBRegressor
            self.model = xgb.XGBRegressor(objective=self.objective, eval_metric=self.eval_metric, **self.params)
        self.model.fit(X, y)

        if self._is_regression_model:
            y_pred_train = self.predict(X)
            _train_residual_std = np.std(y - y_pred_train)
            if np.isnan(_train_residual_std):
                self._train_residual_std = 0.0
            else:
                self._train_residual_std = _train_residual_std

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self.model.predict(X)

    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        if not self._is_regression_model:
            raise ValueError("predict_uncertainty is only available for regression models.")
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        # For simplicity, return a constant uncertainty based on training residuals
        return np.full(X.shape[0], self._train_residual_std)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if not hasattr(self.model, "predict_proba"):
            raise ValueError("predict_proba is not available for the current XGBoost configuration (likely regression).")
        return self.model.predict_proba(X)

    def get_hyperparameter_search_space(self) -> Dict[str, Any]:
        return {
            "n_estimators": {"type": "int", "low": 50, "high": 200, "step": 50},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "max_depth": {"type": "int", "low": 3, "high": 9, "step": 2},
            "subsample": {"type": "float", "low": 0.6, "high": 1.0, "step": 0.1},
            "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0, "step": 0.1},
            "gamma": {"type": "float", "low": 0.0, "high": 0.2, "step": 0.05},
        }

    def get_internal_model(self):
        """
        Returns the raw underlying LightGBM model.
        """
        return self.model
