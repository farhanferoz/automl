"""XGBoost model wrapper for AutoML."""

from typing import Any, Never

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, mean_squared_error

from automl_package.enums import TaskType
from automl_package.models.base import BaseModel
from automl_package.utils.metrics import Metrics


class XGBoostModel(BaseModel):
    """XGBoost model wrapper.

    Note: Methods in this class are intentionally named identically to those in BaseModel
    as they implement the BaseModel interface. This is not a redeclaration error.
    """

    def __init__(
        self,
        objective: str = "reg:squarederror",
        eval_metric: str = "rmse",
        random_seed: int | None = None,
        task_type: TaskType = TaskType.REGRESSION,
        **kwargs: Any,
    ) -> None:
        """Initializes the XGBoostModel.

        Args:
            objective (str): The learning objective function.
            eval_metric (str): The evaluation metric.
            random_seed (int, optional): Random seed for reproducibility.
            task_type (TaskType): The type of task (regression or classification).
            **kwargs: Additional keyword arguments for the BaseModel.
        """
        super().__init__(**kwargs)
        self.objective = objective
        self.eval_metric = eval_metric
        self.random_seed = random_seed
        self.task_type = task_type
        self.model: xgb.XGBRegressor | xgb.XGBClassifier | None = None
        self.is_regression_model = self.task_type == TaskType.REGRESSION
        self._train_residual_std = 0.0
        self.num_iterations_used = 0
        self.params.setdefault("verbosity", 0)

    @property
    def name(self) -> str:
        """Returns the name of the model."""
        return "XGBoostModel"

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
            eval_set = [(x_val, y_val)]
            self.params["early_stopping_rounds"] = self.early_stopping_rounds
        else:
            self.train_indices = np.arange(x.shape[0])
            self.val_indices = None

        if self.task_type == TaskType.CLASSIFICATION:
            self.model = xgb.XGBClassifier(objective=self.objective, eval_metric=self.eval_metric, random_state=self.random_seed, **self.params)
        else:
            self.model = xgb.XGBRegressor(objective=self.objective, eval_metric=self.eval_metric, random_state=self.random_seed, **self.params)

        if forced_iterations is not None:
            self.model.n_estimators = forced_iterations

        fit_params: dict[str, Any] = {}
        if eval_set is not None:
            fit_params["eval_set"] = eval_set
            fit_params["verbose"] = False

        self.model.fit(x_train, y_train, **fit_params)

        if self.is_regression_model:
            y_pred_train = self.predict(x)
            self._train_residual_std = np.std(y - y_pred_train)
            if np.isnan(self._train_residual_std):
                self._train_residual_std = 0.0

        best_iteration = self.model.best_iteration if use_early_stopping and self.model.best_iteration is not None else self.model.n_estimators
        return best_iteration, []

    def _evaluate_trial(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Evaluates a trial for hyperparameter optimization."""
        if self.is_regression_model:
            return np.sqrt(mean_squared_error(y_true, y_pred))

        y_pred_labels = np.argmax(y_pred, axis=1) if y_pred.ndim == 2 else np.round(y_pred)
        return accuracy_score(y_true, y_pred_labels)

    def _clone(self) -> "XGBoostModel":
        """Creates a new instance of the model with the same parameters."""
        return XGBoostModel(**self.get_params())

    def get_params(self) -> dict[str, Any]:
        """Gets parameters for this estimator.

        Returns:
            dict: Parameter names mapped to their values.
        """
        return {
            "objective": self.objective,
            "eval_metric": self.eval_metric,
            "random_seed": self.random_seed,
            "task_type": self.task_type,
            **self.params,
        }

    def cross_validate(self, x: np.ndarray, y: np.ndarray, cv: int) -> dict[str, Any]:
        """Performs cross-validation and returns the scores."""
        self.cv_folds = cv
        self.fit(x, y)
        return {"test_score": self.cv_score_mean_}

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Makes predictions on new data.

        Args:
            x (np.ndarray): Feature matrix for prediction.

        Returns:
            np.ndarray: Predicted values.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if self.objective in ["binary:logistic", "multi:softmax", "multi:softprob"]:
            # For binary classification, predict_proba returns probabilities [:, 1] for positive class
            # For multi-class, predict_proba returns (N, num_classes) array of probabilities
            if self.objective == "binary:logistic":
                return self.model.predict_proba(x)[:, 1]
            # For 'multi:softmax' or 'multi:softprob'
            # .predict() for multi:softmax returns class labels, predict_proba gives probabilities
            if self.objective == "multi:softmax":
                return self.model.predict(x)
            return self.model.predict_proba(x)
        return self.model.predict(x)

    def predict_uncertainty(self, x: np.ndarray) -> np.ndarray:
        """Estimates uncertainty for predictions.

        Args:
            x (np.ndarray): Feature matrix for uncertainty estimation.

        Returns:
            np.ndarray: Uncertainty estimates (e.g., standard deviation).
        """
        if not self.is_regression_model:
            raise ValueError("predict_uncertainty is only available for regression models.")
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        # For simplicity, return a constant uncertainty based on training residuals
        return np.full(x.shape[0], self._train_residual_std)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predicts class probabilities for classification tasks.

        Args:
            x (np.ndarray): Features for probability prediction.

        Returns:
            np.ndarray: Predicted probabilities.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if not hasattr(self.model, "predict_proba"):
            raise ValueError("predict_proba is not available for the current XGBoost configuration (likely regression).")
        return self.model.predict_proba(x)

    def get_hyperparameter_search_space(self) -> dict[str, Any]:
        """Defines the hyperparameter search space for XGBoost.

        Returns:
            Dict[str, Any]: A dictionary defining the hyperparameter search space.
        """
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

    def get_internal_model(self) -> xgb.XGBRegressor | xgb.XGBClassifier | None:
        """Returns the raw underlying XGBoost model."""
        return self.model

    def get_num_parameters(self) -> int:
        """Returns the number of estimators in the XGBoost model.

        Returns:
            int: The number of estimators.
        """
        if self.model is None:
            return 0
        # For tree-based models, n_estimators (number of trees) is a reasonable proxy for complexity.
        return self.num_iterations_used + 1

    def get_classifier_predictions(self, x: np.ndarray, y_true_original: np.ndarray) -> Never:
        """Not implemented for XGBoostModel.

        Raises:
            NotImplementedError: XGBoostModel is not a composite model.
        """
        raise NotImplementedError("XGBoostModel is not a composite model and does not have an internal classifier for separate prediction.")

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
        task_type = "regression" if self.is_regression_model else "classification"
        y_proba = self.predict_proba(x) if task_type == "classification" else None
        metrics_calculator = Metrics(task_type=task_type, model_name=self.name, x_data=x, y_true=y, y_pred=y_pred, y_proba=y_proba)
        metrics_calculator.save_metrics(save_path)
        return y_pred
