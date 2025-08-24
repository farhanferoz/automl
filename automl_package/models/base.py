"""Base classes for machine learning models."""

import abc
from typing import Any

import numpy as np
import optuna
from sklearn.model_selection import KFold

from automl_package.optimizers.optuna_optimizer import OptunaOptimizer
from automl_package.utils.data_handler import create_train_val_split


class BaseModel(abc.ABC):
    """Abstract base class for all machine learning models in the package.

    Defines a common interface for fitting, predicting, and hyperparameter search.
    """

    def __init__(
        self,
        early_stopping_rounds: int | None = None,
        validation_fraction: float = 0.1,
        cv_folds: int | None = None,
        optimize_hyperparameters: bool = False,
        search_space_override: dict | None = None,
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
        self.cv_folds = cv_folds
        self.optimize_hyperparameters = optimize_hyperparameters
        self.search_space_override = search_space_override or {}

        self.best_params_ = None
        self.optimal_iterations_ = None
        self.cv_score_mean_ = None
        self.cv_score_std_ = None

        self.num_iterations_used = 0
        self.train_indices = None
        self.val_indices = None

    def get_params(self) -> dict[str, Any]:
        """Gets the parameters of the model."""
        return self.params

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fits the model to the training data."""
        if self.optimize_hyperparameters:
            if not self.cv_folds:
                raise ValueError("cv_folds must be set to perform hyperparameter optimization.")
            self.best_params_ = self._find_best_hyperparameters(x, y)
            self.params.update(self.best_params_)

        if self.cv_folds:
            self.optimal_iterations_, self.cv_score_mean_, self.cv_score_std_ = self._find_optimal_iterations_with_cv(x, y)
            self._fit_single(x, y, forced_iterations=self.optimal_iterations_)
        else:
            self.num_iterations_used, _ = self._fit_single(x, y)

    def _prepare_train_val_data(
        self, x: np.ndarray, y: np.ndarray, x_val: np.ndarray | None = None, y_val: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Prepares training and validation data.

        If validation data is not provided, it splits the training data.

        Args:
            x (np.ndarray): The training features.
            y (np.ndarray): The training targets.
            x_val (np.ndarray | None): The validation features.
            y_val (np.ndarray | None): The validation targets.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]: A tuple containing:
                - The training features.
                - The training targets.
                - The validation features.
                - The validation targets.
        """
        if x_val is None and y_val is None and self.early_stopping_rounds is not None:
            self.train_indices, self.val_indices = create_train_val_split(x, y, self.validation_fraction, self.random_seed)
            return x[self.train_indices], y[self.train_indices], x[self.val_indices], y[self.val_indices]
        return x, y, x_val, y_val

    @abc.abstractmethod
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
        raise NotImplementedError

    def _perform_cv_for_trial(self, x: np.ndarray, y: np.ndarray) -> float:
        """Performs cross-validation for a single trial."""
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=getattr(self, "random_seed", None))
        scores = []
        for train_idx, val_idx in kf.split(x, y):
            x_train_fold, x_val_fold = x[train_idx], x[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            model_instance = self._clone()
            model_instance._fit_single(x_train_fold, y_train_fold, x_val_fold, y_val_fold)
            preds = model_instance.predict(x_val_fold)
            score = self._evaluate_trial(y_val_fold, preds)
            scores.append(score)
        return float(np.mean(scores))

    def _find_best_hyperparameters(self, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Finds the best hyperparameters using Optuna."""

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
            return model_instance._perform_cv_for_trial(x, y)

        optimizer = OptunaOptimizer(direction="minimize" if self.is_regression_model else "maximize")
        study = optimizer.optimize(objective)
        return study.best_params

    def _find_optimal_iterations_with_cv(self, x: np.ndarray, y: np.ndarray) -> tuple[int, float, float]:
        """Finds the optimal number of iterations using cross-validation."""
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=getattr(self, "random_seed", None))
        fold_results = []
        for train_idx, val_idx in kf.split(x, y):
            x_train_fold, x_val_fold = x[train_idx], x[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            model_instance = self._clone()
            best_iter, loss_history = model_instance._fit_single(x_train_fold, y_train_fold, x_val_fold, y_val_fold)
            preds = model_instance.predict(x_val_fold)
            score = self._evaluate_trial(y_val_fold, preds)
            fold_results.append({"best_iter": best_iter, "loss_history": loss_history, "score": score})

        scores = [res["score"] for res in fold_results]
        avg_score, std_score = np.mean(scores), np.std(scores)

        max_len = max(len(res["loss_history"]) for res in fold_results)
        if max_len == 0:
            optimal_iterations = int(np.mean([res["best_iter"] for res in fold_results]))
        else:
            avg_loss_curve = np.full(max_len, np.nan)
            for i in range(max_len):
                epoch_losses = [res["loss_history"][i] for res in fold_results if i < len(res["loss_history"])]
                if epoch_losses:
                    avg_loss_curve[i] = np.mean(epoch_losses)

            optimal_iterations = np.nanargmin(avg_loss_curve) + 1
            min_best_iter = min(res["best_iter"] for res in fold_results)

            if optimal_iterations >= min_best_iter:
                optimal_iterations = int(np.mean([res["best_iter"] for res in fold_results]))

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

    @abc.abstractmethod
    def evaluate(self, x: np.ndarray, y: np.ndarray, save_path: str = "metrics") -> np.ndarray:
        """Evaluates the model and saves metrics."""
        raise NotImplementedError

    def get_internal_model(self) -> Any:
        """Returns the internal model."""
        return self.model
