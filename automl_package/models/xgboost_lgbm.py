import xgboost as xgb
import lightgbm as lgb
import numpy as np
from typing import Dict, Any

from .base import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost model wrapper."
    Note: Methods in this class are intentionally named identically to those in BaseModel
    as they implement the BaseModel interface. This is not a redeclaration error.
    """

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
        if self.objective in ["binary:logistic", "multi:softmax", "multi:softprob"]:
            # For binary classification, predict_proba returns probabilities [:, 1] for positive class
            # For multi-class, predict_proba returns (N, num_classes) array of probabilities
            if self.objective == "binary:logistic":
                return self.model.predict_proba(X)[:, 1]
            else:  # For 'multi:softmax' or 'multi:softprob'
                # .predict() for multi:softmax returns class labels, predict_proba gives probabilities
                if self.objective == "multi:softmax":
                    return self.model.predict(X)
                else:  # multi:softprob
                    return self.model.predict_proba(X)
        else:
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
            "reg_alpha": {"type": "float", "low": 1e-6, "high": 1.0, "log": True},  # L1 regularization
            "reg_lambda": {"type": "float", "low": 1e-6, "high": 1.0, "log": True},  # L2 regularization
        }

    def get_internal_model(self):
        """
        Returns the raw underlying XGBoost model.
        """
        return self.model

    def get_classifier_predictions(self, X: np.ndarray, y_true_original: np.ndarray):
        raise NotImplementedError("XGBoostModel is not a composite model and does not have an internal classifier for separate prediction.")


class LightGBMModel(BaseModel):
    """LightGBM model wrapper."""

    def __init__(self, objective: str = "regression", metric: str = "rmse", **kwargs):
        super().__init__(**kwargs)
        self.objective = objective
        self.metric = metric
        self.model = None
        # Determine if it's a regression model based on objective
        self._is_regression_model = self.objective in ["regression", "regression_l1", "huber", "fair", "poisson", "quantile", "mape", "gamma", "tweedie"]
        self._train_residual_std = 0.0  # For regression uncertainty

    @property
    def name(self) -> str:
        return "LightGBM"

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Handle classification objectives for LightGBM based on objective string
        if self.objective in ["binary", "multiclass", "multiclassova"]:
            # For classification, use LGBMClassifier
            self.model = lgb.LGBMClassifier(objective=self.objective, metric=self.metric, **self.params)
        else:
            # For regression, use LGBMRegressor
            self.model = lgb.LGBMRegressor(objective=self.objective, metric=self.metric, **self.params)
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
        if self.objective in ["binary", "multiclass", "multiclassova"]:
            # For binary classification, predict_proba returns probabilities of classes [:, 1] for positive class
            return self.model.predict_proba(X)[:, 1]
        else:
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
            raise ValueError("predict_proba is not available for the current LightGBM configuration (likely regression).")
        return self.model.predict_proba(X)

    def get_hyperparameter_search_space(self) -> Dict[str, Any]:
        return {
            "n_estimators": {"type": "int", "low": 50, "high": 200, "step": 50},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "num_leaves": {"type": "int", "low": 20, "high": 60, "step": 10},
            "max_depth": {"type": "int", "low": 5, "high": 15, "step": 2},
            "min_child_samples": {"type": "int", "low": 20, "high": 40, "step": 5},
            "subsample": {"type": "float", "low": 0.6, "high": 1.0, "step": 0.1},
            "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0, "step": 0.1},
            "reg_alpha": {"type": "float", "low": 1e-6, "high": 1.0, "log": True},  # L1 regularization
            "reg_lambda": {"type": "float", "low": 1e-6, "high": 1.0, "log": True},  # L2 regularization
        }

    def get_internal_model(self):
        """
        Returns the raw underlying LightGBM model.
        """
        return self.model

    def get_classifier_predictions(self, X: np.ndarray, y_true_original: np.ndarray):
        raise NotImplementedError("LightGBMModel is not a composite model and does not have an internal classifier for separate prediction.")
