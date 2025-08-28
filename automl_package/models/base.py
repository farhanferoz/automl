"""Base classes for machine learning models."""

import abc
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import KFold

from automl_package.enums import DataSplitStrategy, Metric, TaskType
from automl_package.logger import logger
from automl_package.optimizers.optuna_optimizer import OptunaOptimizer
from automl_package.utils.cv import TimeSeriesSplit
from automl_package.utils.data_handler import create_train_val_split
from automl_package.utils.metrics import Metrics
from automl_package.utils.numerics import find_optimal_iterations


class BaseModel(abc.ABC):
    """Abstract base class for all machine learning models in the package.

    Defines a common interface for fitting, predicting, and hyperparameter search.
    """

    def __init__(
        self,
        early_stopping_rounds: int | None = None,
        validation_fraction: float = 0.1,
        test_fraction: float = 0.1,
        cv_folds: int | None = None,
        split_strategy: DataSplitStrategy = DataSplitStrategy.RANDOM,
        optimize_hyperparameters: bool = False,
        search_space_override: dict | None = None,
        output_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the base model with given parameters."""
        if validation_fraction is not None and cv_folds is not None:
            raise ValueError("validation_fraction and cv_folds cannot both be set.")

        self.model = None
        self.params = kwargs
        self.is_regression_model = False
        self.is_composite_regression_model = False
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.test_fraction = test_fraction
        self.split_strategy = split_strategy
        self.cv_folds = cv_folds
        self.optimize_hyperparameters = optimize_hyperparameters
        self.search_space_override = search_space_override or {}
        self.output_dir = output_dir
        self.feature_names = None

        self.best_params_ = None
        self.optimal_iterations_ = None
        self.cv_score_mean_ = None
        self.cv_score_std_ = None

        self.num_iterations_used = 0
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None

    def get_params(self) -> dict[str, Any]:
        """Gets the parameters of the model."""
        params = self.params.copy()
        params.update(
            {
                "early_stopping_rounds": self.early_stopping_rounds,
                "validation_fraction": self.validation_fraction,
                "test_fraction": self.test_fraction,
                "cv_folds": self.cv_folds,
                "split_strategy": self.split_strategy,
                "optimize_hyperparameters": self.optimize_hyperparameters,
                "search_space_override": self.search_space_override,
                "output_dir": self.output_dir,
            }
        )
        return params

    def fit(self, x: np.ndarray, y: np.ndarray, timestamps: np.ndarray | None = None) -> None:
        """Fits the model to the training data and evaluates it on all available partitions."""
        logger.info(f"--- Starting training for {self.name} ---")

        if isinstance(x, pd.DataFrame):
            self.feature_names = x.columns.tolist()
            x = x.values
        if isinstance(y, pd.Series | pd.DataFrame):
            y = y.values

        # Step 1: Create all data partitions up front
        x_train, y_train, x_val, y_val, x_test, y_test, _, _, _, x_train_val, y_train_val, timestamps_train_val = self._prepare_data_partitions(x, y, timestamps)

        # Step 2: Train the Model
        if self.cv_folds:
            if self.optimize_hyperparameters:
                logger.info("--- Starting hyperparameter optimization ---")
                self.best_params_ = self._find_best_hyperparameters(x_train_val, y_train_val)
                self.params.update(self.best_params_)
                logger.info(f"--- Hyperparameter optimization finished. Best params: {self.best_params_} ---")

            logger.info("--- Finding optimal iterations with cross-validation ---")
            self.optimal_iterations_, self.cv_score_mean_, self.cv_score_std_ = self._find_optimal_iterations_with_cv(x_train_val, y_train_val, timestamps_train_val)
            logger.info(f"--- Optimal iterations found: {self.optimal_iterations_} ---")

            # Final fit on the entire training+validation pool
            logger.info("--- Training final model with optimal parameters ---")
            self._fit_single(x_train_val, y_train_val, forced_iterations=self.optimal_iterations_)
            logger.info("--- Final model training finished ---")
        else:
            # Fit with a single train/validation split to find the best number of iterations
            logger.info("--- Finding optimal iterations with train/validation split ---")
            self.num_iterations_used, _ = self._fit_single(x_train, y_train, x_val=x_val, y_val=y_val)
            logger.info(f"--- Optimal iterations found: {self.num_iterations_used} ---")

            # Retrain on the full training + validation set with the optimal number of iterations
            logger.info("--- Training final model with optimal parameters ---")
            self._fit_single(x_train_val, y_train_val, forced_iterations=self.num_iterations_used)
            logger.info("--- Final model training finished ---")

        # Step 3: Automatic Final Evaluation
        if self.output_dir:
            logger.info("--- Starting final evaluation ---")
            if self.cv_folds:
                self.evaluate(x_train_val, y_train_val, "train_after_cv", self.output_dir)
            else:
                if x_train is not None:
                    self.evaluate(x_train, y_train, "train", self.output_dir)
                if x_val is not None:
                    self.evaluate(x_val, y_val, "validation", self.output_dir)
            if x_test is not None:
                self.evaluate(x_test, y_test, "test", self.output_dir)
            logger.info("--- Final evaluation finished ---")

    def _prepare_data_partitions(self, x: np.ndarray, y: np.ndarray, timestamps: np.ndarray | None = None) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
    ]:
        """Creates train, validation, and test partitions."""
        self.train_indices, self.val_indices, self.test_indices = create_train_val_split(
            x,
            self.validation_fraction if self.cv_folds is None else 0,
            self.test_fraction,
            self.split_strategy,
            timestamps,
            getattr(self, "random_seed", None),
        )

        x_train, y_train = x[self.train_indices], y[self.train_indices]
        x_val, y_val = (x[self.val_indices], y[self.val_indices]) if self.val_indices.size > 0 else (None, None)
        x_test, y_test = (x[self.test_indices], y[self.test_indices]) if self.test_indices.size > 0 else (None, None)

        timestamps_train = timestamps[self.train_indices] if timestamps is not None else None
        timestamps_val = timestamps[self.val_indices] if timestamps is not None and self.val_indices.size > 0 else None
        timestamps_test = timestamps[self.test_indices] if timestamps is not None and self.test_indices.size > 0 else None

        train_val_indices = np.concatenate([self.train_indices, self.val_indices])
        x_train_val, y_train_val = x[train_val_indices], y[train_val_indices]
        timestamps_train_val = timestamps[train_val_indices] if timestamps is not None else None

        return x_train, y_train, x_val, y_val, x_test, y_test, timestamps_train, timestamps_val, timestamps_test, x_train_val, y_train_val, timestamps_train_val

    @abc.abstractmethod
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
        raise NotImplementedError

    def _find_best_hyperparameters(self, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Finds the best hyperparameters using Optuna."""

        def _perform_cv_for_trial(model_instance: "BaseModel", x_cv: np.ndarray, y_cv: np.ndarray) -> float:
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=getattr(self, "random_seed", None))
            scores = []
            for train_idx, val_idx in kf.split(x_cv, y_cv):
                x_train_fold, x_val_fold = x_cv[train_idx], x_cv[val_idx]
                y_train_fold, y_val_fold = y_cv[train_idx], y_cv[val_idx]
                model_instance._fit_single(x_train_fold, y_train_fold, x_val=x_val_fold, y_val=y_val_fold)
                preds = model_instance.predict(x_val_fold)
                score = self._evaluate_trial(y_val_fold, preds)
                scores.append(score)
            return float(np.mean(scores))

        def objective(trial: optuna.Trial) -> float:
            search_space = self.get_hyperparameter_search_space()
            params = {}
            for name, space in search_space.items():
                if space["type"] == "int":
                    params[name] = trial.suggest_int(name, space["low"], space["high"], step=space.get("step", 1), log=space.get("log", False))
                elif space["type"] == "float":
                    params[name] = trial.suggest_float(name, space["low"], space["high"], step=space.get("step"), log=space.get("log", False))
                elif space["type"] == "categorical":
                    params[name] = trial.suggest_categorical(name, space["choices"])

            model_instance = self.__class__(**params)
            return _perform_cv_for_trial(model_instance, x, y)

        direction = "minimize" if self._get_optimization_metric() in [Metric.RMSE, Metric.MSE] else "maximize"
        optimizer = OptunaOptimizer(direction=direction)
        study = optimizer.optimize(objective)
        return study.best_params

    @abc.abstractmethod
    def _get_optimization_metric(self) -> Metric:
        """Gets the optimization metric for the model."""
        raise NotImplementedError

    def _find_optimal_iterations_with_cv(self, x: np.ndarray, y: np.ndarray, timestamps: np.ndarray | None = None) -> tuple[int, float, float]:
        """Finds the optimal number of iterations using cross-validation."""
        if self.split_strategy == DataSplitStrategy.TIME_ORDERED:
            if timestamps is None:
                raise ValueError("timestamps must be provided for time_ordered split strategy.")
            kf = TimeSeriesSplit(n_splits=self.cv_folds)
        else:
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=getattr(self, "random_seed", None))
        fold_results = []
        for i, (train_idx, val_idx) in enumerate(kf.split(x, y)):
            logger.info(f"--- Training CV fold {i+1}/{self.cv_folds} ---")
            x_train_fold, x_val_fold = x[train_idx], x[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            model_instance = self._clone()
            best_iter, loss_history = model_instance._fit_single(x_train_fold, y_train_fold, x_val=x_val_fold, y_val=y_val_fold)
            preds = model_instance.predict(x_val_fold)
            score = self._evaluate_trial(y_val_fold, preds)
            fold_results.append({"best_iter": best_iter, "loss_history": loss_history, "score": score})

        scores = [res["score"] for res in fold_results]
        avg_score, std_score = np.mean(scores), np.std(scores)

        optimal_iterations = find_optimal_iterations(fold_results)

        return optimal_iterations, avg_score, std_score

    @abc.abstractmethod
    def _evaluate_trial(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Evaluates a trial for hyperparameter optimization."""
        raise NotImplementedError

    @abc.abstractmethod
    def _clone(self) -> "BaseModel":
        """Creates a new instance of the model with the same parameters."""
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Makes predictions on new data."""
        raise NotImplementedError

    @abc.abstractmethod
    def predict_uncertainty(self, x: np.ndarray) -> np.ndarray:
        """Estimates the uncertainty of predictions."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_hyperparameter_search_space(self) -> dict[str, Any]:
        """Gets the hyperparameter search space for the model."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Returns the name of the model."""
        raise NotImplementedError

    @abc.abstractmethod
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predicts class probabilities."""
        raise NotImplementedError

    @abc.abstractmethod
    def cross_validate(self, x: np.ndarray, y: np.ndarray, cv: int) -> dict[str, Any]:
        """Performs cross-validation."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_classifier_predictions(self, x: np.ndarray, y_true_original: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gets predictions from the internal classifier."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_num_parameters(self) -> int:
        """Returns the number of trainable parameters in the model."""
        raise NotImplementedError

    def evaluate(self, x: np.ndarray, y: np.ndarray, partition_name: str, save_path: str) -> np.ndarray:
        """Evaluates the model on a given dataset and saves the metrics."""
        y_pred = self.predict(x)
        y_proba = self.predict_proba(x) if self.task_type == TaskType.CLASSIFICATION else None
        metrics_calculator = Metrics(
            task_type=self.task_type, model_name=self.name, x_data=x, y_true=y, y_pred=y_pred.flatten(), y_proba=y_proba, partition_name=partition_name
        )
        metrics_calculator.save_metrics(save_path)

        if self.is_composite_regression_model:
            try:
                y_pred_internal, y_proba_internal, y_true_discretized = self.get_classifier_predictions(x, y)
                classification_metrics = Metrics(
                    task_type=TaskType.CLASSIFICATION,
                    model_name=f"{self.name}_internal_classifier",
                    x_data=x,
                    y_true=y_true_discretized,
                    y_pred=y_pred_internal,
                    y_proba=y_proba_internal,
                    partition_name=f"{partition_name}_classification",
                )
                classification_metrics.save_metrics(save_path)
            except NotImplementedError:
                pass

        if hasattr(self, "plot_probability_mappers") and callable(self.plot_probability_mappers):
            self.plot_probability_mappers(plot_path=f"{save_path}/{partition_name}_probability_mappers.png")

        return y_pred

    def get_internal_model(self) -> Any:
        """Returns the internal model."""
        return self.model
