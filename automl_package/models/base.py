"""Base classes for machine learning models."""

import abc
import inspect
from typing import Any, ClassVar

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import KFold

from automl_package.enums import DataSplitStrategy, HpoSampler, Metric, TaskType, UncertaintyMethod
from automl_package.logger import logger
from automl_package.models.common.mixins import BinnedUncertaintyMixin
from automl_package.optimizers.optuna_optimizer import OptunaOptimizer
from automl_package.utils.cv import TimeSeriesSplit
from automl_package.utils.data_handler import create_train_val_split
from automl_package.utils.feature_selection import select_features_by_cumulative_importance
from automl_package.utils.metrics import Metrics, calculate_performance_score
from automl_package.utils.numerics import find_optimal_iterations
from automl_package.utils.plotting import plot_feature_importance


class BaseModel(abc.ABC, BinnedUncertaintyMixin):
    """Abstract base class for all machine learning models in the package.

    Defines a common interface for fitting, predicting, and hyperparameter search.
    """

    def __init__(
        self,
        task_type: TaskType = TaskType.REGRESSION,
        early_stopping_rounds: int | None = None,
        validation_fraction: float = 0.1,
        test_fraction: float = 0.1,
        cv_folds: int | None = None,
        split_strategy: DataSplitStrategy = DataSplitStrategy.RANDOM,
        optimize_hyperparameters: bool = False,
        n_trials: int = 50,
        search_space_override: dict | None = None,
        search_space_exclude: list[str] | None = None,
        hpo_sampler: HpoSampler = HpoSampler.TPE,
        hpo_seed: int | None = None,
        output_dir: str | None = None,
        calculate_feature_importance: bool = True,
        feature_selection_threshold: float | None = None,
        shap_max_data_points: int | None = 50000,
        uncertainty_method: UncertaintyMethod = UncertaintyMethod.CONSTANT,
        refit_on_train_val: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initializes the base model with given parameters.

        ``refit_on_train_val=True`` (default) ends `fit` with a refit on train + validation. ``False`` is the
        one-pass protocol: `fit` keeps the early-stopped model of a single pass (and the search keeps its best
        trial's fitted model), so the validation rows stay held out. It needs a single validation split, so it
        cannot be combined with ``cv_folds`` or feature selection.
        """
        if validation_fraction is not None and cv_folds is not None:
            raise ValueError("validation_fraction and cv_folds cannot both be set.")
        if not refit_on_train_val and (cv_folds is not None or feature_selection_threshold is not None):
            raise ValueError("refit_on_train_val=False keeps one fitted model on one validation split: it cannot be combined with cv_folds or feature_selection_threshold.")

        self.model = None
        self.params = kwargs
        self.task_type = task_type
        self.is_regression_model = task_type == TaskType.REGRESSION
        self.is_composite_regression_model = False
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.test_fraction = test_fraction
        self.split_strategy = split_strategy
        self.cv_folds = cv_folds
        self.optimize_hyperparameters = optimize_hyperparameters
        self.n_trials = n_trials
        self.search_space_override = search_space_override or {}
        self.search_space_exclude = list(search_space_exclude or [])
        self.hpo_sampler = hpo_sampler
        self.hpo_seed = hpo_seed
        self.output_dir = output_dir
        self.feature_names: list[str] | None = None
        self.feature_to_idx_: dict[str, int] | None = None
        self.calculate_feature_importance = calculate_feature_importance
        self.feature_selection_threshold = feature_selection_threshold
        self.shap_max_data_points = shap_max_data_points
        self.selected_features_: list[str] | None = None
        self.uncertainty_method = uncertainty_method
        self.refit_on_train_val = refit_on_train_val
        self._best_trial_model_: BaseModel | None = None

        if self.feature_selection_threshold is not None and self.feature_selection_threshold < 1.0:
            assert self.feature_selection_threshold > 0
            self.calculate_feature_importance = True

        self.cv_score_mean_: float | None = None
        self.cv_score_std_: float | None = None

        self.num_iterations_used = 0
        self.train_indices: np.ndarray | None = None
        self.val_indices: np.ndarray | None = None
        self.test_indices: np.ndarray | None = None

        self._init_binned_uncertainty()

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
                "n_trials": self.n_trials,
                "search_space_override": self.search_space_override,
                "search_space_exclude": self.search_space_exclude,
                "hpo_sampler": self.hpo_sampler,
                "hpo_seed": self.hpo_seed,
                "output_dir": self.output_dir,
                "uncertainty_method": self.uncertainty_method,
                "refit_on_train_val": self.refit_on_train_val,
                "task_type": self.task_type,
                "calculate_feature_importance": self.calculate_feature_importance,
                "feature_selection_threshold": self.feature_selection_threshold,
                "shap_max_data_points": self.shap_max_data_points,
            }
        )
        return params

    def _update_params(self, params: dict[str, Any]) -> None:
        """Updates the model's parameters from a given dictionary."""
        for key, value in params.items():
            setattr(self, key, value)
        self.params.update(params)

    def _filter_predict_data(self, x: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
        """Filters the input data to include only the selected features."""
        filtered_data = x
        if self.selected_features_ is not None:
            if isinstance(x, pd.DataFrame):
                filtered_data = x[self.selected_features_]
            # If the data is a numpy array, we can only filter it if the number of columns matches the original number of features. Otherwise, we assume it's already filtered.
            elif isinstance(x, np.ndarray) and (x.shape[1] == len(self.feature_names)):
                if self.feature_to_idx_ is None:
                    raise ValueError("feature_to_idx_ mapping is not available for numpy array filtering.")
                selected_indices = [self.feature_to_idx_[feature] for feature in self.selected_features_]
                filtered_data = x[:, selected_indices]
            # If it's a numpy array that doesn't match the original number of features, or an unsupported type,
            # return it as is, assuming it's already been handled.
        return filtered_data

    def calculate_feature_importances(self, model_instance: "BaseModel", x_background: pd.DataFrame) -> dict[str, float]:
        """Calculates and returns feature importances."""
        from automl_package.explainers.feature_explainer import FeatureExplainer  # noqa: PLC0415

        logger.info("--- Calculating feature importance ---")

        explainer = FeatureExplainer(
            model_instance=model_instance,
            x_background=x_background.values,
            feature_names=self.selected_features_ if self.selected_features_ else self.feature_names,
            max_data_points=self.shap_max_data_points,
            device=getattr(model_instance, "device", None),
        )
        shap_values = explainer.explain(x_background.values)
        return explainer.get_feature_importance_summary(shap_values)

    def _save_feature_importance(self, feature_importance: dict[str, float]) -> None:
        if self.output_dir:
            plot_feature_importance(feature_importance, f"{self.output_dir}/feature_importance.png")
            feature_importance_df = pd.DataFrame(list(feature_importance.items()), columns=["feature", "importance"])
            feature_importance_df.to_csv(f"{self.output_dir}/feature_importance.csv", index=False)

    def _calculate_and_save_feature_importance(self, model_instance: "BaseModel", x_background: pd.DataFrame) -> None:
        """Calculates and saves feature importance."""
        feature_importance = self.calculate_feature_importances(model_instance=model_instance, x_background=x_background)
        self._save_feature_importance(feature_importance=feature_importance)

    def _perform_feature_selection(self, x_train_val: pd.DataFrame, y_train_val: np.ndarray, iterations: int | None) -> None:
        """Performs feature selection and returns the filtered training data."""
        temp_model = self._clone_for_trial()
        logger.info("--- Training model on all the features in preparation for feature selection ---")
        temp_model._fit_single(x_train_val.values, y_train_val, forced_iterations=iterations)

        feature_importance = self.calculate_feature_importances(model_instance=temp_model, x_background=x_train_val)

        if self.feature_selection_threshold is not None:
            self.selected_features_ = select_features_by_cumulative_importance(feature_importance, self.feature_selection_threshold)
            logger.info(f"--- Selected {len(self.selected_features_)} features ---")

    # Fields `fit` owns on the outer model; `_adopt_fitted_state` never overwrites them with a trial model's values.
    _OUTER_OWNED_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"feature_names", "feature_to_idx_", "selected_features_", "train_indices", "val_indices", "test_indices", "cv_score_mean_", "cv_score_std_", "num_iterations_used"}
    )

    def _adopt_fitted_state(self, fitted: "BaseModel") -> None:
        """Takes over the fitted state of a trial model (the one-pass search keeps its best trial's model, no refit).

        The outer model's own settings (every `BaseModel.__init__` argument: search budget, feature importance, refit
        protocol, ...) are never taken from the trial clone; its searched hyperparameters are already the best trial's.
        """
        keep = self._OUTER_OWNED_FIELDS | _BASE_SETTINGS
        self.__dict__.update({k: v for k, v in fitted.__dict__.items() if k not in keep})

    def _determine_optimal_parameters(
        self,
        x_train: pd.DataFrame | None,
        y_train: np.ndarray,
        x_train_val: pd.DataFrame,
        y_train_val: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        timestamps_train_val: np.ndarray | None = None,
        use_explicit_split: bool = False,
    ) -> None:
        """Runs the search / early-stopping pass.

        With ``use_explicit_split`` (a caller-given validation set, or the one-pass protocol) the search scores
        its trials on ``(x_train, x_val)`` instead of drawing an inner split from train + validation. Under
        ``refit_on_train_val=False`` this method ends with `self` fitted (best trial adopted, or one early-stopped pass).
        """
        if self.optimize_hyperparameters:
            logger.info("--- Starting hyperparameter optimization ---")
            split = (x_train.values, y_train, x_val.values, y_val) if use_explicit_split and x_val is not None else None
            best_params, self.num_iterations_used = self._find_best_hyperparameters(x=x_train_val.values, y=y_train_val, timestamps=timestamps_train_val, explicit_split=split)
            self._update_params(best_params)
            if not self.refit_on_train_val:
                self._adopt_fitted_state(self._best_trial_model_)
                self._best_trial_model_ = None
            logger.info(f"--- Hyperparameter optimization finished. Best params: {self.get_params()} ---")
            logger.info(f"--- Optimal iterations from HPO: {self.num_iterations_used} ---")
        elif not self.refit_on_train_val:
            logger.info("--- One-pass fit on the training part, validation held out ---")
            self.num_iterations_used, _ = self._fit_single(x_train.values, y_train, x_val=x_val.values if x_val is not None else None, y_val=y_val)
        elif self.early_stopping_rounds is not None:
            # Step 2: Train the Model
            if self.cv_folds:
                logger.info("--- Finding optimal iterations with cross-validation ---")
                self.num_iterations_used, self.cv_score_mean_, self.cv_score_std_ = self._find_optimal_iterations_with_cv(
                    x=x_train_val.values, y=y_train_val, timestamps=timestamps_train_val
                )
            else:
                # Fit with a single train/validation split to find the best number of iterations
                logger.info("--- Finding optimal iterations with train/validation split ---")
                self.num_iterations_used, _ = self._fit_single(x_train.values, y_train, x_val=x_val.values if x_val is not None else None, y_val=y_val)

            logger.info(f"--- Optimal iterations found: {self.num_iterations_used} ---")

    def _fit_final_model(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Retrain on the full training + validation set with the optimal number of iterations.

        ``num_iterations_used`` is 0 until HPO or an early-stopping pass sets it. Passing
        0 to tree models treats it as literal "train 0 rounds" and crashes. When no
        optimisation pass has run we hand ``None`` (use the model's configured default).
        """
        logger.info("--- Training final model with optimal parameters ---")
        forced = self.num_iterations_used if self.num_iterations_used and self.num_iterations_used > 0 else None
        self._fit_single(x_train, y_train, forced_iterations=forced)
        logger.info("--- Final model training finished ---")

    def fit(
        self,
        x: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.DataFrame | pd.Series,
        timestamps: np.ndarray | None = None,
        x_val: np.ndarray | pd.DataFrame | None = None,
        y_val: np.ndarray | pd.DataFrame | pd.Series | None = None,
    ) -> None:
        """Fits the model to the training data and evaluates it on all available partitions.

        Args:
            x: training inputs.
            y: training targets.
            timestamps: optional timestamps for time-ordered splits (ignored with an explicit validation set).
            x_val: optional explicit validation inputs. When given (with ``y_val``) they are the validation set,
                ``(x, y)`` is the training part whole, and no inner split is drawn (so no test rows are set aside).
            y_val: validation targets for ``x_val``.
        """
        logger.info(f"--- Starting training for {self.name} ---")

        if isinstance(x, np.ndarray):
            self.feature_names = [f"feature_{i}" for i in range(x.shape[1])]
            x = pd.DataFrame(x, columns=self.feature_names)
        else:
            self.feature_names = x.columns.tolist()
        self.feature_to_idx_ = {feature: i for i, feature in enumerate(self.feature_names)}

        if isinstance(y, pd.Series | pd.DataFrame):
            y = y.values

        explicit_val = x_val is not None
        if explicit_val:
            if y_val is None:
                raise ValueError("x_val was given without y_val.")
            if self.cv_folds:
                raise ValueError("An explicit validation set cannot be combined with cv_folds.")
            x_val = x_val.values if isinstance(x_val, pd.DataFrame) else np.asarray(x_val)
            y_val = y_val.values if isinstance(y_val, pd.Series | pd.DataFrame) else np.asarray(y_val)

        # Step 1: Create all data partitions up front
        x_train, y_train, x_val, y_val, x_test, y_test, _, _, _, x_train_val, y_train_val, timestamps_train_val = self._prepare_data_partitions(
            x.values, y, timestamps, x_val=x_val, y_val=y_val
        )

        x_train = pd.DataFrame(x_train, columns=self.feature_names)
        x_val = pd.DataFrame(x_val, columns=self.feature_names) if x_val is not None else None
        x_test = pd.DataFrame(x_test, columns=self.feature_names) if x_test is not None else None
        x_train_val = pd.DataFrame(x_train_val, columns=self.feature_names)

        self._determine_optimal_parameters(
            x_train=x_train,
            y_train=y_train,
            x_train_val=x_train_val,
            y_train_val=y_train_val,
            x_val=x_val,
            y_val=y_val,
            timestamps_train_val=timestamps_train_val,
            use_explicit_split=explicit_val or not self.refit_on_train_val,
        )

        if self.refit_on_train_val:
            if self.feature_selection_threshold is not None:
                self._perform_feature_selection(x_train_val, y_train_val, self.num_iterations_used)

            x_train_val_final = x_train_val[self.selected_features_] if self.selected_features_ else x_train_val
            self._fit_final_model(x_train=x_train_val_final.values, y_train=y_train_val)
            x_model, y_model = x_train_val_final, y_train_val
        else:
            x_model, y_model = x_train, y_train  # the model was fitted on the training part only

        if self.uncertainty_method == UncertaintyMethod.BINNED_RESIDUAL_STD:
            logger.info("--- Calibrating uncertainty model ---")
            self.calibrate_uncertainty(x=x_model.values, y=y_model)

        if self.calculate_feature_importance:
            self._calculate_and_save_feature_importance(model_instance=self, x_background=x_model)

        # Step 3: Automatic Final Evaluation
        if self.output_dir:
            logger.info("--- Starting final evaluation ---")
            if self.cv_folds:
                self.evaluate(x_train_val, y_train_val, "train_after_cv", self.output_dir)
            else:
                if x_train_val is not None:
                    self.evaluate(x_train_val, y_train_val, "train_val", self.output_dir)
            if x_test is not None:
                self.evaluate(x_test, y_test, "test", self.output_dir)
            logger.info("--- Final evaluation finished ---")

    def _prepare_data_partitions(
        self, x: np.ndarray, y: np.ndarray, timestamps: np.ndarray | None = None, x_val: np.ndarray | None = None, y_val: np.ndarray | None = None
    ) -> tuple[
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
        """Creates train, validation, and test partitions.

        With an explicit ``(x_val, y_val)`` no split is drawn: ``(x, y)`` is the training part whole, there is no
        test part, and ``val_indices`` index the concatenation ``[x; x_val]``.
        """
        if x_val is not None:
            n_train = len(x)
            self.train_indices, self.val_indices, self.test_indices = np.arange(n_train), np.arange(n_train, n_train + len(x_val)), np.array([], dtype=int)
            return x, y, x_val, y_val, None, None, timestamps, None, None, np.concatenate([x, x_val]), np.concatenate([y, y_val]), timestamps
        self.train_indices, self.val_indices, self.test_indices = create_train_val_split(
            x=x,
            validation_fraction=self.validation_fraction if self.cv_folds is None else 0,
            test_fraction=self.test_fraction,
            split_strategy=self.split_strategy,
            timestamps=timestamps,
            random_state=getattr(self, "random_seed", None),
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

        return (
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            y_test,
            timestamps_train,
            timestamps_val,
            timestamps_test,
            x_train_val,
            y_train_val,
            timestamps_train_val,
        )

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

    def _evaluate_trial_performance(
        self,
        model_instance: "BaseModel",
        x: np.ndarray,
        y: np.ndarray,
        timestamps: np.ndarray | None = None,
        explicit_split: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> tuple[float, int]:
        """Evaluates a trial's performance and returns the score and optimal iterations.

        ``explicit_split = (x_train, y_train, x_val, y_val)`` replaces the inner split drawn from ``(x, y)``.
        """
        if self.cv_folds:
            kf = self.get_kfolds(timestamps=timestamps)
            fold_results = []
            for train_idx, val_idx in kf.split(x, y):
                x_train_fold, x_val_fold = x[train_idx], x[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                best_iter, loss_history = model_instance._fit_single(x_train_fold, y_train_fold, x_val=x_val_fold, y_val=y_val_fold)
                score = model_instance._held_out_fit_score(x_val_fold, y_val_fold)
                fold_results.append({"best_iter": best_iter, "loss_history": loss_history, "score": score})

            final_score = float(np.mean([res["score"] for res in fold_results]))
            optimal_iterations = find_optimal_iterations(fold_results)
        else:
            if explicit_split is not None:
                x_train, y_train, x_val, y_val = explicit_split
            else:
                train_indices, val_indices, _ = create_train_val_split(
                    x=x,
                    validation_fraction=self.validation_fraction,
                    test_fraction=0,
                    split_strategy=self.split_strategy,
                    timestamps=timestamps,
                    random_state=getattr(self, "random_seed", None),
                )
                x_train, y_train = x[train_indices], y[train_indices]
                x_val, y_val = x[val_indices], y[val_indices]
            optimal_iterations, _ = model_instance._fit_single(x_train, y_train, x_val=x_val, y_val=y_val)
            final_score = model_instance._held_out_fit_score(x_val, y_val)
        return final_score, optimal_iterations

    def _find_best_hyperparameters(
        self,
        x: np.ndarray,
        y: np.ndarray,
        timestamps: np.ndarray | None = None,
        explicit_split: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> tuple[dict[str, Any], int]:
        """Finds the best hyperparameters and optimal iterations using Optuna.

        Under ``refit_on_train_val=False`` the best trial's fitted model is kept in ``self._best_trial_model_``
        (only the best so far is held in memory), for `fit` to adopt instead of refitting.
        """
        smaller_is_better = self._held_out_score_is_smaller_better()
        best_trial: dict[str, Any] = {"score": None, "model": None}

        def objective(trial: optuna.Trial) -> float:
            search_space = self._searched_space()
            trial_params = {}
            for name, space in search_space.items():
                if space["type"] == "int":
                    trial_params[name] = trial.suggest_int(name, space["low"], space["high"], step=space.get("step", 1), log=space.get("log", False))
                elif space["type"] == "float":
                    trial_params[name] = trial.suggest_float(name, space["low"], space["high"], step=space.get("step"), log=space.get("log", False))
                elif space["type"] == "categorical":
                    trial_params[name] = trial.suggest_categorical(name, space["choices"])

            # Create a new model instance with the original parameters, updated with the trial's specific hyperparameters
            model_instance = self._clone_for_trial()
            model_instance._update_params(trial_params)

            score, iterations = self._evaluate_trial_performance(model_instance, x, y, timestamps=timestamps, explicit_split=explicit_split)
            trial.set_user_attr("iterations", iterations)
            if not self.refit_on_train_val and (best_trial["score"] is None or (score < best_trial["score"] if smaller_is_better else score > best_trial["score"])):
                best_trial.update(score=score, model=model_instance)
            return score

        direction = "minimize" if smaller_is_better else "maximize"
        seed = self.hpo_seed if self.hpo_seed is not None else getattr(self, "random_seed", None)
        optimizer = OptunaOptimizer(direction=direction, n_trials=self.n_trials, random_seed=seed, sampler=self.hpo_sampler)
        study = optimizer.optimize(objective)
        best_iterations = study.best_trial.user_attrs["iterations"]
        self._best_trial_model_ = best_trial["model"]
        return study.best_trial.params, best_iterations

    def _get_optimization_metric(self) -> Metric:
        """Gets the optimization metric for the model."""
        if self.task_type == TaskType.REGRESSION:
            optimization_metric = Metric.NLL if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC else Metric.MSE
        else:
            optimization_metric = Metric.LOG_LOSS
        return optimization_metric

    def _held_out_fit_score(self, x_val: np.ndarray, y_val: np.ndarray) -> float:
        """Held-out score for hyperparameter search and cross-validation, on a model ALREADY fitted on the training part.

        Call it on the trial / fold model instance, never on the outer model (scoring the unfitted outer
        object was the F1 crash). Default: the model's optimisation metric on its predictions (trees and
        linear models). `PyTorchModelBase` overrides it with the held-out fit term of the training loss
        (owner ruling 2026-11-01, DECISIONS.md "Validation score = the held-out FIT TERM"). Direction is
        given by `_held_out_score_is_smaller_better`.
        """
        preds = self._predict_for_scoring(x_val)
        y_pred_std = self.predict_uncertainty(x_val)
        return self._evaluate_trial(y_val, preds, y_pred_std=y_pred_std)

    def _held_out_score_is_smaller_better(self) -> bool:
        """Direction of `_held_out_fit_score` for the Optuna study."""
        return self._get_optimization_metric().is_smaller_better

    def _calculate_performance_score(self, y_true: np.ndarray, y_pred: np.ndarray, x_val: np.ndarray | None = None, y_pred_std: np.ndarray | None = None) -> float:
        """Calculates the performance score based on the model's optimization metric."""
        metric_to_use = self._get_optimization_metric()
        if metric_to_use == Metric.NLL and y_pred_std is None:
            if x_val is None:
                raise ValueError("x_val must be provided to calculate NLL if y_pred_std is not given.")
            y_pred_std = self.predict_uncertainty(x_val, filter_data=False)
        return calculate_performance_score(metric=metric_to_use, y_true=y_true, y_pred=y_pred, y_pred_std=y_pred_std)

    def get_kfolds(self, timestamps: np.ndarray | None = None) -> TimeSeriesSplit | KFold:
        """Gets the cross-validation folds splitter."""
        if self.split_strategy == DataSplitStrategy.TIME_ORDERED:
            if timestamps is None:
                raise ValueError("timestamps must be provided for time_ordered split strategy.")
            kf = TimeSeriesSplit(n_splits=self.cv_folds)
        elif self.split_strategy == DataSplitStrategy.RANDOM:
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=getattr(self, "random_seed", None))
        else:
            raise ValueError(f"Split strategy {self.split_strategy.value} is not supported.")
        return kf

    def _find_optimal_iterations_with_cv(self, x: np.ndarray, y: np.ndarray, timestamps: np.ndarray | None = None) -> tuple[int, float, float]:
        """Finds the optimal number of iterations using cross-validation."""
        kf = self.get_kfolds(timestamps=timestamps)
        fold_results = []
        for i, (train_idx, val_idx) in enumerate(kf.split(x, y)):
            logger.info(f"--- Training CV fold {i + 1}/{self.cv_folds} ---")
            x_train_fold, x_val_fold = x[train_idx], x[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            model_instance = self._clone_for_trial()
            best_iter, loss_history = model_instance._fit_single(x_train_fold, y_train_fold, x_val=x_val_fold, y_val=y_val_fold)
            score = model_instance._held_out_fit_score(x_val_fold, y_val_fold)
            fold_results.append({"best_iter": best_iter, "loss_history": loss_history, "score": score})

        scores = [res["score"] for res in fold_results]
        avg_score, std_score = np.mean(scores), np.std(scores)

        optimal_iterations = find_optimal_iterations(fold_results)

        return optimal_iterations, avg_score, std_score

    def _evaluate_trial(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_std: np.ndarray | None = None) -> float:
        """Evaluates a trial for hyperparameter optimization."""
        return self._calculate_performance_score(y_true=y_true, y_pred=y_pred, x_val=None, y_pred_std=y_pred_std)

    def _clone_for_trial(self) -> "BaseModel":
        """Clones the model for a search trial, CV fold or selection pass, carrying the column names `fit` recorded (column roles need them)."""
        clone = self._clone()
        clone.feature_names = self.feature_names
        clone.feature_to_idx_ = self.feature_to_idx_
        return clone

    @abc.abstractmethod
    def _clone(self) -> "BaseModel":
        """Creates a new instance of the model with the same parameters."""
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, x: np.ndarray | pd.DataFrame, filter_data: bool = True) -> np.ndarray:
        """Makes predictions on new data."""
        raise NotImplementedError

    def _predict_for_scoring(self, x: np.ndarray | pd.DataFrame, filter_data: bool = True) -> np.ndarray:
        """Internal, non-caller-facing prediction path for CV/HPO/evaluation bookkeeping (capacity-programme FP-10).

        Generic bookkeeping call sites -- CV folds and the HPO objective in this class, `evaluate()`
        below, `utils/data_handler.py`'s `fit_smearing_correction` -- call polymorphic predictions
        with no extra caller state, on the assumption that some concrete prediction is always
        obtainable. That assumption breaks for a family whose public `predict()` contract requires
        caller-supplied state beyond `x` (e.g. `FlexibleWidthNN`, which requires an explicit
        `width` under `CapacitySelection.FIXED` -- FP-3.b.4, and correctly raises without one).
        Defaults to the public `predict()`; override where that contract cannot be satisfied
        without state the bookkeeping caller doesn't have.
        """
        return self.predict(x, filter_data=filter_data)

    def predict_uncertainty(self, x: np.ndarray, filter_data: bool = True) -> np.ndarray:
        """Estimates the uncertainty of predictions."""
        if self.uncertainty_method == UncertaintyMethod.BINNED_RESIDUAL_STD:
            if filter_data:
                x = self._filter_predict_data(x)
            return self.predict_binned_uncertainty(x)
        raise NotImplementedError(f"Uncertainty method {self.uncertainty_method} not supported by this model.")

    def _searched_space(self) -> dict[str, Any]:
        """The search space a search samples: the model's own space (with ``search_space_override`` applied) minus ``search_space_exclude``.

        Raises:
            ValueError: If an excluded name is not in the space (the message lists the valid names).
        """
        space = dict(self.get_hyperparameter_search_space())
        unknown = [name for name in self.search_space_exclude if name not in space]
        if unknown:
            raise ValueError(f"search_space_exclude names {unknown} are not in the search space; valid names: {sorted(space)}.")
        return {name: value for name, value in space.items() if name not in self.search_space_exclude}

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
    def predict_proba(self, x: np.ndarray | pd.DataFrame, filter_data: bool = True) -> np.ndarray:
        """Predicts class probabilities."""
        raise NotImplementedError

    @abc.abstractmethod
    def cross_validate(self, x: np.ndarray, y: np.ndarray, cv: int) -> dict[str, Any]:
        """Performs cross-validation."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_classifier_predictions(self, x: np.ndarray | pd.DataFrame, y_true_original: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gets predictions from the internal classifier."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_num_parameters(self) -> int:
        """Returns the number of trainable parameters in the model."""
        raise NotImplementedError

    def evaluate(self, x: pd.DataFrame | np.ndarray, y: np.ndarray, partition_name: str, save_path: str, y_scaler: Any | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        """Evaluates the model on a given dataset and saves the metrics."""
        x_eval = x.values if isinstance(x, pd.DataFrame) else x

        y_pred_scaled = self._predict_for_scoring(x_eval)
        y_std_scaled = self.predict_uncertainty(x_eval) if self.is_regression_model else None

        # Inverse transform predictions and uncertainty if a scaler is provided
        y_pred_unscaled = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten() if y_scaler is not None else y_pred_scaled

        y_std_unscaled = y_std_scaled
        if y_scaler is not None and y_std_scaled is not None:
            scale_ = getattr(y_scaler, "scale_", None)
            if scale_ is not None:
                y_std_unscaled = y_std_scaled * scale_[0]

        y_proba = self.predict_proba(x_eval) if self.task_type == TaskType.CLASSIFICATION else None

        # Calculate metrics using the unscaled values
        metrics_calculator = Metrics(
            task_type=self.task_type,
            model_name=self.name,
            x_data=x_eval,
            y_true=y,
            y_pred=y_pred_unscaled.flatten(),
            y_proba=y_proba,
            y_std=y_std_unscaled.flatten() if y_std_unscaled is not None else None,
            partition_name=partition_name,
        )
        metrics_calculator.save_metrics(save_path)

        if hasattr(self, "plot_probability_mappers") and callable(self.plot_probability_mappers):
            self.plot_probability_mappers(plot_path=f"{save_path}/{partition_name}_probability_mappers.png")

        # Return the unscaled predictions and uncertainty
        return y_pred_unscaled, y_std_unscaled

    def get_internal_model(self) -> Any:
        """Returns the internal model."""
        return self.model

    @abc.abstractmethod
    def get_shap_explainer_info(self) -> dict[str, Any]:
        """Gets the SHAP explainer type and the model to be explained.

        This method must be implemented by all concrete model subclasses.

        Returns:
            A dictionary containing the explainer type and the model object.
        """
        raise NotImplementedError


# BaseModel's own settings (its constructor arguments): a one-pass search never takes these from a trial clone.
_BASE_SETTINGS: frozenset[str] = frozenset(p for p in inspect.signature(BaseModel.__init__).parameters if p not in ("self", "kwargs"))
