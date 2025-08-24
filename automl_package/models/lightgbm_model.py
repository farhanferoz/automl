"""LightGBM model wrapper for AutoML."""

from typing import Any, NoReturn

import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error

from automl_package.enums import TaskType
from automl_package.models.base import BaseModel
from automl_package.utils.metrics import Metrics


class LightGBMModel(BaseModel):
    """LightGBM model wrapper."""

    def __init__(self, task_type: TaskType = TaskType.REGRESSION, random_seed: int | None = None, **kwargs: Any) -> None:
        """Initializes the LightGBMModel."""
        super().__init__(**kwargs)
        self.task_type = task_type
        self.random_seed = random_seed
        self.model = None
        self.is_regression_model = task_type == TaskType.REGRESSION
        self._train_residual_std = 0.0
        self.num_iterations_used = 0

        self.params.setdefault("verbose", -1)
        if self.task_type == TaskType.REGRESSION:
            self.objective = "regression"
            self.metric = "rmse"
        elif self.task_type == TaskType.CLASSIFICATION:
            self.objective = "binary"
            self.metric = "binary_logloss"
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    @property
    def name(self) -> str:
        """Returns the name of the model."""
        return "LightGBMModel"

    def _fit_single(
        self, x: np.ndarray, y: np.ndarray, x_val: np.ndarray | None = None, y_val: np.ndarray | None = None, forced_iterations: int | None = None
    ) -> tuple[int, list[float]]:
        """Fits a single model instance.

        Args:
            x (np.ndarray): The training features.
            y (np.ndarray): The training targets.
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

        x_train, y_train, x_val, y_val = self._prepare_train_val_data(x, y, x_val, y_val)

        if use_early_stopping and x_val is not None:
            eval_set = [(x_val, y_val)]
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))

        if self.task_type == TaskType.CLASSIFICATION:
            self.model = lgb.LGBMClassifier(objective=self.objective, metric=self.metric, random_state=self.random_seed, **self.params)
        else:
            self.model = lgb.LGBMRegressor(objective=self.objective, metric=self.metric, random_state=self.random_seed, **self.params)

        if forced_iterations is not None:
            self.model.n_estimators = forced_iterations

        self.model.fit(x_train, y_train, eval_set=eval_set, callbacks=callbacks)

        best_iteration = self.model.best_iteration_ if use_early_stopping and self.model.best_iteration_ is not None else self.model.n_estimators

        return best_iteration, []

    def _evaluate_trial(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Evaluates a trial for hyperparameter optimization."""
        if self.is_regression_model:
            return np.sqrt(mean_squared_error(y_true, y_pred))
        return accuracy_score(y_true, np.round(y_pred))

    def _clone(self) -> "LightGBMModel":
        """Creates a new instance of the model with the same parameters."""
        return LightGBMModel(**self.get_params())

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Makes predictions on new data."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self.model.predict(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predicts class probabilities."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if not hasattr(self.model, "predict_proba"):
            raise ValueError("predict_proba is not available for this model.")
        return self.model.predict_proba(x)

    def predict_uncertainty(self, x: np.ndarray) -> np.ndarray:
        """Estimates uncertainty for predictions."""
        if not self.is_regression_model:
            raise ValueError("predict_uncertainty is only available for regression models.")
        return np.full(x.shape[0], self._train_residual_std)

    def get_hyperparameter_search_space(self) -> dict[str, Any]:
        """Gets the hyperparameter search space for the model."""
        space = {
            "n_estimators": {"type": "int", "low": 50, "high": 500, "step": 50},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "num_leaves": {"type": "int", "low": 20, "high": 150, "step": 10},
            "max_depth": {"type": "int", "low": 5, "high": 15, "step": 2},
            "min_child_samples": {"type": "int", "low": 20, "high": 40, "step": 5},
            "subsample": {"type": "float", "low": 0.6, "high": 1.0, "step": 0.1},
            "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0, "step": 0.1},
            "reg_alpha": {"type": "float", "low": 1e-6, "high": 1.0, "log": True},
            "reg_lambda": {"type": "float", "low": 1e-6, "high": 1.0, "log": True},
        }
        if self.search_space_override:
            space.update(self.search_space_override)
        return space

    def get_internal_model(self) -> Any:
        """Returns the raw underlying LightGBM model."""
        return self.model

    def get_num_parameters(self) -> int:
        """Returns the number of estimators in the LightGBM model."""
        if self.model is None:
            return 0
        return self.model.n_estimators

    def get_classifier_predictions(self, x: np.ndarray, y_true_original: np.ndarray) -> NoReturn:
        """Not implemented for LightGBMModel."""
        raise NotImplementedError("LightGBMModel is not a composite model.")

    def evaluate(self, x: np.ndarray, y: np.ndarray, save_path: str = "metrics") -> np.ndarray:
        """Evaluates the model on a given dataset and saves the metrics."""
        y_pred = self.predict(x)
        y_proba = None
        if self.task_type == TaskType.CLASSIFICATION:
            y_proba = self.predict_proba(x)
        metrics_calculator = Metrics(task_type=self.task_type.value, model_name=self.name, x_data=x, y_true=y, y_pred=y_pred, y_proba=y_proba)
        metrics_calculator.save_metrics(save_path)
        return y_pred

    def cross_validate(self, x: np.ndarray, y: np.ndarray, cv: int) -> dict[str, Any]:
        """Performs cross-validation."""
        self.cv_folds = cv
        self.fit(x, y)
        return {"test_score": self.cv_score_mean_}
