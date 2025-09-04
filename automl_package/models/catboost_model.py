"""CatBoost model wrapper for AutoML."""

from typing import Any, Never

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.metrics import accuracy_score, mean_squared_error

from automl_package.enums import Metric, TaskType
from automl_package.models.base import BaseModel
from automl_package.models.common.common import get_loss_history
from automl_package.utils.numerics import ensure_proba_shape


class CatBoostModel(BaseModel):
    """CatBoost model wrapper."""

    def __init__(self, random_seed: int | None = None, **kwargs: Any) -> None:
        """Initializes the CatBoostModel.

        Args:
            task_type (TaskType): The type of machine learning task (regression or classification).
            random_seed (int, optional): Random seed for reproducibility.
            **kwargs: Additional keyword arguments for the CatBoost model.
        """
        super().__init__(**kwargs)
        self.random_seed = random_seed
        self.model: CatBoostRegressor | CatBoostClassifier | None = None
        self._train_residual_std = 0.0
        self.num_iterations_used = 0

        self.params.setdefault("verbose", 0)
        if self.random_seed is not None:
            self.params.setdefault("random_seed", self.random_seed)
        if self.task_type == TaskType.REGRESSION:
            self.params.setdefault("loss_function", "RMSE")
            self.params.setdefault("eval_metric", "RMSE")
        elif self.task_type == TaskType.CLASSIFICATION:
            self.params.setdefault("loss_function", Metric.LOG_LOSS.label)
            self.params.setdefault("eval_metric", Metric.LOG_LOSS.label)

    @property
    def name(self) -> str:
        """Returns the name of the model."""
        return "CatBoostModel"

    def _get_optimization_metric(self) -> Metric:
        """Gets the optimization metric for the model."""
        return Metric.RMSE if self.is_regression_model else Metric.ACCURACY

    def _fit_single(
        self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray | None = None, y_val: np.ndarray | None = None, forced_iterations: int | None = None
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
        use_early_stopping = self.early_stopping_rounds is not None and forced_iterations is None

        if use_early_stopping and x_val is not None:
            eval_set = Pool(x_val, y_val)

        params = self.params.copy()
        iteration_synonyms = ["iterations", "n_estimators", "num_boost_round", "num_trees"]
        for synonym in iteration_synonyms:
            params.pop(synonym, None)

        if forced_iterations is not None:
            params["iterations"] = forced_iterations
        else:
            # Use original iterations if provided, otherwise default to 500
            iterations = self.params.get("iterations")
            if iterations is None:
                iterations = self.params.get("n_estimators")
            if iterations is None:
                iterations = 500
            params["iterations"] = iterations

        self.model = CatBoostRegressor(**params) if self.task_type == TaskType.REGRESSION else CatBoostClassifier(**params)

        fit_params: dict[str, Any] = {}
        if eval_set is not None:
            fit_params["eval_set"] = eval_set
            fit_params["early_stopping_rounds"] = self.early_stopping_rounds

        self.model.fit(x_train, y_train, **fit_params)

        best_iteration = self.model.tree_count_ if (use_early_stopping and (self.model.tree_count_ is not None)) else params["iterations"]
        loss_history = get_loss_history(self.model, use_early_stopping)

        return best_iteration, loss_history

    def _evaluate_trial(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Evaluates a trial for hyperparameter optimization.

        Args:
            y_true (np.ndarray): The true target values.
            y_pred (np.ndarray): The predicted target values.

        Returns:
            float: The evaluation score.
        """
        return self._calculate_metric(y_true, y_pred, Metric.RMSE if self.is_regression_model else Metric.ACCURACY)

    def _calculate_metric(self, y_true: np.ndarray, y_pred: np.ndarray, metric: Metric) -> float:
        """Calculates a metric."""
        if metric == Metric.RMSE:
            return np.sqrt(mean_squared_error(y_true, y_pred))
        if metric == Metric.ACCURACY:
            return accuracy_score(y_true, np.round(y_pred))
        raise ValueError(f"Unknown metric: {metric}")

    def _clone(self) -> "CatBoostModel":
        """Creates a new instance of the model with the same parameters."""
        return CatBoostModel(**self.get_params())

    def get_params(self) -> dict[str, Any]:
        """Gets parameters for this estimator."""
        params = super().get_params()
        params.update({"task_type": self.task_type, "random_seed": self.random_seed})
        return params

    def predict(self, x: np.ndarray, filter_data: bool = True) -> np.ndarray:
        """Makes predictions on new data.

        Args:
            x (np.ndarray): Feature matrix for prediction.
            filter_data (bool): If True, filter the input data using the feature selection mask.

        Returns:
            np.ndarray: Predicted values.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if filter_data:
            x = self._filter_predict_data(x)
        return self.model.predict(x)

    def predict_proba(self, x: np.ndarray, filter_data: bool = True) -> np.ndarray:
        """Predicts class probabilities for classification tasks.

        Args:
            x (np.ndarray): Features for probability prediction.
            filter_data (bool): If True, filter the input data using the feature selection mask.

        Returns:
            np.ndarray: Predicted probabilities.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if self.task_type == TaskType.REGRESSION:
            raise ValueError("predict_proba is not available for regression tasks.")
        if filter_data:
            x = self._filter_predict_data(x)
        proba = self.model.predict_proba(x)
        return ensure_proba_shape(proba, self.model.classes_.size)

    def predict_uncertainty(self, x: np.ndarray, filter_data: bool = True) -> np.ndarray:
        """Estimates uncertainty for predictions.

        Args:
            x (np.ndarray): Feature matrix for uncertainty estimation.
            filter_data (bool): If True, filter the input data using the feature selection mask.

        Returns:
            np.ndarray: Uncertainty estimates (e.g., standard deviation).
        """
        if not self.is_regression_model:
            raise ValueError("predict_uncertainty is only available for regression models.")
        if filter_data:
            x = self._filter_predict_data(x)
        return np.full(x.shape[0], self._train_residual_std)

    def get_hyperparameter_search_space(self) -> dict[str, Any]:
        """Gets the hyperparameter search space for the model.

        Returns:
            dict[str, Any]: A dictionary defining the hyperparameter search space.
        """
        space = {
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "depth": {"type": "int", "low": 3, "high": 8},
            "l2_leaf_reg": {"type": "float", "low": 1e-2, "high": 10.0, "log": True},
        }
        if self.early_stopping_rounds is None:
            space["iterations"] = {"type": "int", "low": 5, "high": 550, "step": 50}
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
        return 0 if self.model is None else self.model.tree_count_

    def get_classifier_predictions(self, x: np.ndarray, y_true_original: np.ndarray) -> Never:
        """Not implemented for CatBoostModel.

        Raises:
            NotImplementedError: CatBoostModel is not a composite model.
        """
        raise NotImplementedError("CatBoostModel is not a composite model.")

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
