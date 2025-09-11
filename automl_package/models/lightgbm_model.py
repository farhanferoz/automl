"""LightGBM model wrapper for AutoML."""

from typing import Any, NoReturn

import lightgbm as lgb
import numpy as np

from automl_package.enums import ExplainerType, Metric, TaskType
from automl_package.models.base import BaseModel
from automl_package.models.common.common import get_loss_history
from automl_package.utils.numerics import ensure_proba_shape


class LightGBMModel(BaseModel):
    """LightGBM model wrapper."""

    def __init__(self, random_seed: int | None = None, **kwargs: Any) -> None:
        """Initializes the LightGBMModel."""
        super().__init__(**kwargs)
        self.random_seed = random_seed
        self.model = None
        self._train_residual_std = 0.0
        self.num_iterations_used = 0

        self.params.setdefault("verbose", -1)
        if self.task_type == TaskType.REGRESSION:
            self.objective = "regression"
            self.metric = Metric.RMSE.label
        elif self.task_type == TaskType.CLASSIFICATION:
            self.objective = "binary"
            self.metric = Metric.LOG_LOSS.label
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

    @property
    def name(self) -> str:
        """Returns the name of the model."""
        return "LightGBMModel"

    def _fit_single(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        forced_iterations: int | None = None,
    ) -> tuple[int, list[float]]:
        """Fits a single model instance.

        Args:
            x_train (np.ndarray): The training features.
            y_train (np.ndarray): The training targets.
            x_val (np.ndarray | None): The validation features.
            y_val (np.ndarray | None): The validation targets.
            forced_iterations (int | None): If provided, train for this many iterations, ignoring early stopping.

        Returns:
            tuple[int, list[float]]: A tuple containing:
                - The number of iterations the model was trained for.
                - A list of the validation loss values for each epoch.
        """
        eval_set = None
        callbacks = []
        use_early_stopping = self.early_stopping_rounds is not None and forced_iterations is None

        if use_early_stopping and x_val is not None:
            eval_set = [(x_val, y_val)]
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))

        model_instance = lgb.LGBMClassifier if self.task_type == TaskType.CLASSIFICATION else lgb.LGBMRegressor
        params = self.params.copy()
        params.setdefault("n_estimators", 500)
        self.model = model_instance(
            objective=self.objective,
            metric=self.metric,
            random_state=self.random_seed,
            **params,
        )

        if forced_iterations is not None:
            self.model.n_estimators = forced_iterations

        self.model.fit(x_train, y_train, eval_set=eval_set, callbacks=callbacks)

        best_iteration = self.model.best_iteration_ if use_early_stopping and self.model.best_iteration_ is not None else self.model.n_estimators
        loss_history = get_loss_history(self.model, use_early_stopping)

        return best_iteration, loss_history

    def _clone(self) -> "LightGBMModel":
        """Creates a new instance of the model with the same parameters."""
        return LightGBMModel(**self.get_params())

    def get_params(self) -> dict[str, Any]:
        """Gets parameters for this estimator."""
        params = super().get_params()
        params.update({"task_type": self.task_type, "random_seed": self.random_seed})
        return params

    def predict(self, x: np.ndarray, filter_data: bool = True) -> np.ndarray:
        """Makes predictions on new data."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if filter_data:
            x = self._filter_predict_data(x)
        return self.model.predict(x)

    def predict_proba(self, x: np.ndarray, filter_data: bool = True) -> np.ndarray:
        """Predicts class probabilities."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if not hasattr(self.model, "predict_proba"):
            raise ValueError("predict_proba is not available for this model.")
        if filter_data:
            x = self._filter_predict_data(x)
        proba = self.model.predict_proba(x)
        return ensure_proba_shape(proba, self.model.n_classes_)

    def predict_uncertainty(self, x: np.ndarray, filter_data: bool = True) -> np.ndarray:
        """Estimates uncertainty for predictions."""
        if not self.is_regression_model:
            raise ValueError("predict_uncertainty is only available for regression models.")
        if filter_data:
            x = self._filter_predict_data(x)
        return np.full(x.shape[0], self._train_residual_std)

    def get_hyperparameter_search_space(self) -> dict[str, Any]:
        """Gets the hyperparameter search space for the model."""
        space = {
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "num_leaves": {"type": "int", "low": 20, "high": 150, "step": 10},
            "max_depth": {"type": "int", "low": 5, "high": 15, "step": 2},
            "min_child_samples": {"type": "int", "low": 20, "high": 40, "step": 5},
            "subsample": {"type": "float", "low": 0.6, "high": 1.0, "step": 0.1},
            "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0, "step": 0.1},
            "reg_alpha": {"type": "float", "low": 1e-6, "high": 1.0, "log": True},
            "reg_lambda": {"type": "float", "low": 1e-6, "high": 1.0, "log": True},
        }
        if self.early_stopping_rounds is None:
            space["n_estimators"] = {"type": "int", "low": 5, "high": 550, "step": 50}
        if self.search_space_override:
            space.update(self.search_space_override)
        return space

    def get_internal_model(self) -> Any:
        """Returns the raw underlying LightGBM model."""
        return self.model

    def get_shap_explainer_info(self) -> dict[str, Any]:
        """Gets the SHAP explainer type and the model to be explained."""
        return {"explainer_type": ExplainerType.TREE, "model": self.get_internal_model()}

    def get_num_parameters(self) -> int:
        """Returns the number of estimators in the LightGBM model."""
        if self.model is None:
            num_parameters = 0
        elif hasattr(self.model, "best_iteration_") and self.model.best_iteration_ is not None:
            num_parameters = self.model.best_iteration_
        else:
            num_parameters = self.model.n_estimators
        return num_parameters

    def get_classifier_predictions(self, x: np.ndarray, y_true_original: np.ndarray) -> NoReturn:
        """Not implemented for LightGBMModel."""
        raise NotImplementedError("LightGBMModel is not a composite model.")

    def cross_validate(self, x: np.ndarray, y: np.ndarray, cv: int) -> dict[str, Any]:
        """Performs cross-validation."""
        self.cv_folds = cv
        self.fit(x, y)
        return {"test_score": self.cv_score_mean_}
