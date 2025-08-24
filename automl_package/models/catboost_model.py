"""CatBoost model wrapper for AutoML."""

from typing import Any, Never

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.metrics import accuracy_score, mean_squared_error

from automl_package.enums import TaskType
from automl_package.models.base import BaseModel
from automl_package.utils.metrics import Metrics


class CatBoostModel(BaseModel):
    """CatBoost model wrapper."""

    def __init__(self, task_type: TaskType = TaskType.REGRESSION, random_seed: int | None = None, **kwargs: Any) -> None:
        """Initializes the CatBoostModel.

        Args:
            task_type (TaskType): The type of machine learning task (regression or classification).
            random_seed (int, optional): Random seed for reproducibility.
            **kwargs: Additional keyword arguments for the CatBoost model.
        """
        super().__init__(**kwargs)
        self.task_type = task_type
        self.random_seed = random_seed
        self.model: CatBoostRegressor | CatBoostClassifier | None = None
        self.is_regression_model = task_type == TaskType.REGRESSION
        self._train_residual_std = 0.0
        self.num_iterations_used = 0

        self.params.setdefault("verbose", 0)
        if self.random_seed is not None:
            self.params.setdefault("random_seed", self.random_seed)
        if self.task_type == TaskType.REGRESSION:
            self.params.setdefault("loss_function", "RMSE")
            self.params.setdefault("eval_metric", "RMSE")
        elif self.task_type == TaskType.CLASSIFICATION:
            self.params.setdefault("loss_function", "Logloss")
            self.params.setdefault("eval_metric", "Logloss")

    @property
    def name(self) -> str:
        """Returns the name of the model."""
        return "CatBoostModel"

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
        use_early_stopping = self.early_stopping_rounds is not None and forced_iterations is None

        x_train, y_train, x_val, y_val = self._prepare_train_val_data(x, y, x_val, y_val)

        if use_early_stopping and x_val is not None:
            eval_set = Pool(x_val, y_val)

        if self.task_type == TaskType.REGRESSION:
            self.model = CatBoostRegressor(**self.params)
        else:
            self.model = CatBoostClassifier(**self.params)

        if forced_iterations is not None:
            self.model.set_params(iterations=forced_iterations)

        fit_params: dict[str, Any] = {}
        if eval_set is not None:
            fit_params["eval_set"] = eval_set
            fit_params["early_stopping_rounds"] = self.early_stopping_rounds

        self.model.fit(x_train, y_train, **fit_params)

        best_iteration = self.model.tree_count_ if use_early_stopping and self.model.tree_count_ is not None else self.params.get("iterations", 100)
        return best_iteration, []

    def _evaluate_trial(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Evaluates a trial for hyperparameter optimization.

        Args:
            y_true (np.ndarray): The true target values.
            y_pred (np.ndarray): The predicted target values.

        Returns:
            float: The evaluation score.
        """
        if self.is_regression_model:
            return np.sqrt(mean_squared_error(y_true, y_pred))
        return accuracy_score(y_true, np.round(y_pred))

    def _clone(self) -> "CatBoostModel":
        """Creates a new instance of the model with the same parameters."""
        return CatBoostModel(**self.get_params())

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Makes predictions on new data.

        Args:
            x (np.ndarray): Feature matrix for prediction.

        Returns:
            np.ndarray: Predicted values.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self.model.predict(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predicts class probabilities for classification tasks.

        Args:
            x (np.ndarray): Features for probability prediction.

        Returns:
            np.ndarray: Predicted probabilities.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if self.task_type == TaskType.REGRESSION:
            raise ValueError("predict_proba is not available for regression tasks.")
        return self.model.predict_proba(x)

    def predict_uncertainty(self, x: np.ndarray) -> np.ndarray:
        """Estimates uncertainty for predictions.

        Args:
            x (np.ndarray): Feature matrix for uncertainty estimation.

        Returns:
            np.ndarray: Uncertainty estimates (e.g., standard deviation).
        """
        if not self.is_regression_model:
            raise ValueError("predict_uncertainty is only available for regression models.")
        return np.full(x.shape[0], self._train_residual_std)

    def get_hyperparameter_search_space(self) -> dict[str, Any]:
        """Gets the hyperparameter search space for the model.

        Returns:
            dict[str, Any]: A dictionary defining the hyperparameter search space.
        """
        space = {
            "iterations": {"type": "int", "low": 50, "high": 200, "step": 50},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "depth": {"type": "int", "low": 3, "high": 8},
            "l2_leaf_reg": {"type": "float", "low": 1e-2, "high": 10.0, "log": True},
        }
        if self.search_space_override:
            space.update(self.search_space_override)
        return space

    def get_internal_model(self) -> CatBoostRegressor | CatBoostClassifier | None:
        """Returns the raw underlying CatBoost model."""
        return self.model

    def get_num_parameters(self) -> int:
        """Returns the number of trees in the CatBoost model.

        Returns:
            int: The number of trees.
        """
        if self.model is None:
            return 0
        return self.model.tree_count_

    def get_classifier_predictions(self, x: np.ndarray, y_true_original: np.ndarray) -> Never:
        """Not implemented for CatBoostModel.

        Raises:
            NotImplementedError: CatBoostModel is not a composite model.
        """
        raise NotImplementedError("CatBoostModel is not a composite model.")

    def evaluate(self, x: np.ndarray, y: np.ndarray, save_path: str = "metrics") -> np.ndarray:
        """Evaluates the model on a given dataset and saves the metrics.

        Args:
            x (np.ndarray): Feature matrix for evaluation.
            y (np.ndarray): True labels for evaluation.
            save_path (str): Directory to save the metrics files.

        Returns:
            np.ndarray: The predictions made by the model.
        """
        y_pred = self.predict(x)
        y_proba = None
        if self.task_type == TaskType.CLASSIFICATION:
            y_proba = self.predict_proba(x)
        metrics_calculator = Metrics(task_type=self.task_type.value, model_name=self.name, x_data=x, y_true=y, y_pred=y_pred, y_proba=y_proba)
        metrics_calculator.save_metrics(save_path)
        return y_pred

    def cross_validate(self, x: np.ndarray, y: np.ndarray, cv: int) -> dict[str, Any]:
        """Performs cross-validation.

        Args:
            x (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
            cv (int): Number of folds for cross-validation.

        Returns:
            dict[str, Any]: A dictionary containing the cross-validation scores.
        """
        self.cv_folds = cv
        self.fit(x, y)
        return {"test_score": self.cv_score_mean_}
