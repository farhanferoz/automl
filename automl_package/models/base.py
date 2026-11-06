"""Base classes for machine learning models."""

import abc
import functools
import inspect
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import KFold

from automl_package.enums import (
    DataSplitStrategy,
    HpoSampler,
    InputScaling,
    KSelectionRule,
    Metric,
    MissingInputs,
    OutOfRangeAction,
    OutputBound,
    TargetScaling,
    TaskType,
    UncertaintyMethod,
)
from automl_package.logger import logger
from automl_package.models.common.mixins import BinnedUncertaintyMixin
from automl_package.optimizers.optuna_optimizer import OptunaOptimizer
from automl_package.utils.capacity_selection import cheapest_within_tolerance
from automl_package.utils.checkpoint import stable_hash
from automl_package.utils.cv import TimeSeriesSplit
from automl_package.utils.data_handler import InputTransform, TargetScaler, check_sample_weight, check_time_gap, create_train_val_split
from automl_package.utils.feature_selection import select_features_by_cumulative_importance
from automl_package.utils.metrics import Metrics, calculate_performance_score
from automl_package.utils.numerics import find_optimal_iterations
from automl_package.utils.plotting import plot_feature_importance
from automl_package.utils.prediction_envelope import PredictionRangeReport, TrainingEnvelope, check_prediction_range, clip_to_envelope

HPO_IDENTITY_EXCLUDED = ("n_trials", "hpo_storage_dir", "output_dir")  # settings that do not change what a search computes
HPO_HASH_LENGTH = 16
NESTED_PARAM_SEPARATOR = "__"  # ``a__b`` in a search space names key ``b`` of the dict parameter ``a``
MIN_SWEEP_K = 2  # a one-class model predicts one constant, so a sweep over the number of classes starts at two


SPREAD_METHODS = ("predict_uncertainty",)  # public methods returning a spread in target units: wrapped to multiply by the target scale (see `__init_subclass__`)


def _spread_in_target_units(method: Any) -> Any:
    """Wraps a spread method so that, when the model scales its targets, the spread it returns is in target units.

    Only the outermost call scales: a subclass that calls `super().predict_uncertainty` is not scaled twice.
    """

    @functools.wraps(method)
    def wrapper(self: "BaseModel", *args: Any, **kwargs: Any) -> np.ndarray:
        if self.target_scaler_ is None or self._spread_call_depth:
            return method(self, *args, **kwargs)
        self._spread_call_depth += 1
        try:
            spread = method(self, *args, **kwargs)
        finally:
            self._spread_call_depth -= 1
        return self.target_scaler_.spread_inverse(spread)

    wrapper._spread_wrapped = True  # type: ignore[attr-defined]
    return wrapper


INPUT_METHODS = (  # methods taking raw inputs x first when called on a fitted model (the held-out scores: callers such as the ae adapter)
    "predict_uncertainty",
    "predict_proba",
    "predict_distribution",
    "predict_quantiles",
    "_held_out_fit_score",
    "_held_out_fit_per_sample",
    "fit_router",  # FlexNN / ProbReg routers: fitted on held-out inputs; the router then works in fit space
)


def _inputs_in_fit_space(method: Any) -> Any:
    """Wraps a public method taking ``x`` first so that, when the model transforms its inputs, ``x`` reaches it in fit space.

    Only the outermost public call transforms (`predict` takes part in the same count): an inner `self.predict(x)` is not transformed twice.
    """

    @functools.wraps(method)
    def wrapper(self: "BaseModel", *args: Any, **kwargs: Any) -> Any:
        if self.input_transform_ is None or self._input_call_depth:
            return method(self, *args, **kwargs)
        if args:
            args = (self._inputs_to_fit_space(args[0]), *args[1:])
        else:
            kwargs["x"] = self._inputs_to_fit_space(kwargs["x"])
        self._input_call_depth += 1
        try:
            return method(self, *args, **kwargs)
        finally:
            self._input_call_depth -= 1

    wrapper._inputs_wrapped = True  # type: ignore[attr-defined]
    return wrapper


class BaseModel(abc.ABC, BinnedUncertaintyMixin):
    """Abstract base class for all machine learning models in the package.

    Defines a common interface for fitting, predicting, and hyperparameter search.

    Target scaling (`target_scaling`, `enums.TargetScaling`) lives here once, like sklearn's ``TransformedTargetRegressor``: `fit` fits a
    `TargetScaler` on the TRAINING part of ``y`` and hands scaled targets to every internal path (search trials, CV folds, early stopping,
    the final refit, selection passes); `predict` and the spread methods return target units. The scaler is installed on ``target_scaler_``
    only when training is over, so during training the model is entirely in "fit space" (scaled), afterwards entirely in target units.

    Which space each caller passes y in:

    - `_fit_single`, and `_held_out_fit_score` / `_held_out_fit_per_sample` called by `fit` on a trial, fold or selection model (no scaler
      installed on it): fit space, the space `fit` handed down.
    - `_held_out_fit_score` / `_held_out_fit_per_sample` called from outside on a fitted model (scaler installed): raw target units; they go
      through `_to_fit_space`, the one conversion, so the score stays the fit term of the trained loss.
    """

    _default_target_scaling: ClassVar[TargetScaling] = TargetScaling.NONE  # what `target_scaling=None` means for this family
    _supports_target_scaling: ClassVar[bool] = True  # False: every public y-unit output of the class is not mapped back, so PER_OUTPUT is refused
    _default_input_scaling: ClassVar[InputScaling] = InputScaling.NONE  # what `input_scaling=None` means for this family
    _supports_input_scaling: ClassVar[bool] = True  # False: the class scales its inputs itself or bypasses `predict`, so STANDARD is refused
    _default_missing_inputs: ClassVar[MissingInputs] = MissingInputs.RAISE  # what `missing_inputs=None` means for this family
    _supports_sample_weight: ClassVar[bool] = False  # True: `_fit_single` takes sample_weight / sample_weight_val and `_held_out_fit_score` sample_weight

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Wraps the spread methods a subclass defines (see `_spread_in_target_units`)."""
        super().__init_subclass__(**kwargs)
        for name in SPREAD_METHODS:
            method = cls.__dict__.get(name)
            if callable(method) and not getattr(method, "_spread_wrapped", False):
                setattr(cls, name, _spread_in_target_units(method))
        for name in INPUT_METHODS:
            method = cls.__dict__.get(name)
            if callable(method) and not getattr(method, "_inputs_wrapped", False):
                setattr(cls, name, _inputs_in_fit_space(method))

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
        hpo_storage_dir: str | None = None,
        output_dir: str | None = None,
        calculate_feature_importance: bool = True,
        feature_selection_threshold: float | None = None,
        shap_max_data_points: int | None = 50000,
        uncertainty_method: UncertaintyMethod = UncertaintyMethod.CONSTANT,
        refit_on_train_val: bool = True,
        out_of_range_action: OutOfRangeAction = OutOfRangeAction.FLAG,
        target_scaling: TargetScaling | str | None = None,
        input_scaling: InputScaling | str | None = None,
        missing_inputs: MissingInputs | str | None = None,
        time_gap: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initializes the base model with given parameters.

        ``hpo_storage_dir`` (default None: the search lives in memory) keeps the search's Optuna journal in that directory, one file per
        model class, config and data, so a killed run resumes with the trials it had finished.

        ``refit_on_train_val=True`` (default) ends `fit` with a refit on train + validation. ``False`` is the
        one-pass protocol: `fit` keeps the early-stopped model of a single pass (and the search keeps its best
        trial's fitted model), so the validation rows stay held out. It needs a single validation split, so it
        cannot be combined with ``cv_folds`` or feature selection.

        ``out_of_range_action`` (member or its value) is what `predict` does with predictions outside the training targets' range
        (see `predict`): FLAG (default) only reports, CLIP clips to the range, RAISE raises.

        ``target_scaling`` (member, value or None): `TargetScaling.PER_OUTPUT` standardises each target column by the training part's mean
        and sd inside `fit` (class docstring). None takes the family default (`_default_target_scaling`: PER_OUTPUT for the PyTorch
        regression networks that map every output back, NONE otherwise). A classifier never scales: it gets NONE, and PER_OUTPUT raises.

        ``input_scaling`` (member, value or None): `InputScaling.STANDARD` standardises each numeric input column by the training part's mean
        and sd inside `fit` (binary 0/1, categorical and offset columns left alone); `predict` and the public methods taking x transform raw
        inputs once. None takes the family default (`_default_input_scaling`: STANDARD for the PyTorch networks and the linear models).
        ``missing_inputs`` (member, value or None): what a NaN input does: RAISE (default for networks and linear models), TRAINING_MEAN
        (filled with the training part's column mean), NATIVE (passed through; default for gradient-boosted trees and the LSTM).
        ``time_gap`` (default 0): distinct timestamps left out between each fitted part and the held-out part after it, in the TIME_ORDERED
        split `fit` and the search draw and in the time-ordered CV folds (`data_handler.create_train_val_split`, `cv.TimeSeriesSplit`);
        needs ``split_strategy=TIME_ORDERED``. An explicit ``(x_val, y_val)`` is the caller's split and is used as given.
        """
        if validation_fraction is not None and cv_folds is not None:
            raise ValueError("validation_fraction and cv_folds cannot both be set.")
        check_time_gap(time_gap, split_strategy)
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
        self.time_gap = time_gap
        self.cv_folds = cv_folds
        self.optimize_hyperparameters = optimize_hyperparameters
        self.n_trials = n_trials
        self.search_space_override = search_space_override or {}
        self.search_space_exclude = list(search_space_exclude or [])
        self.hpo_sampler = hpo_sampler
        self.hpo_seed = hpo_seed
        self.hpo_storage_dir = hpo_storage_dir
        self.output_dir = output_dir
        self.feature_names: list[str] | None = None
        self.feature_to_idx_: dict[str, int] | None = None
        self.calculate_feature_importance = calculate_feature_importance
        self.feature_selection_threshold = feature_selection_threshold
        self.shap_max_data_points = shap_max_data_points
        self.selected_features_: list[str] | None = None
        self.uncertainty_method = uncertainty_method
        self.refit_on_train_val = refit_on_train_val
        self.out_of_range_action = OutOfRangeAction(out_of_range_action)
        self._target_scaling = self._resolve_target_scaling(target_scaling)
        self.target_scaler_: TargetScaler | None = None  # installed by `fit` when training is over (PER_OUTPUT)
        self._raw_target_range_: tuple[np.ndarray, np.ndarray] | None = None  # raw-unit min, max of the training targets; kept for a bounded output
        self._spread_call_depth = 0
        self._input_scaling = self._resolve_input_scaling(input_scaling)
        self._missing_inputs = self._default_missing_inputs if missing_inputs is None else MissingInputs(missing_inputs)
        self.input_transform_: InputTransform | None = None  # installed by `fit` when training is over (input_scaling / missing_inputs)
        self._input_call_depth = 0
        self.training_envelope_: TrainingEnvelope | None = None
        self.last_prediction_report_: PredictionRangeReport | None = None
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
                "time_gap": self.time_gap,
                "optimize_hyperparameters": self.optimize_hyperparameters,
                "n_trials": self.n_trials,
                "search_space_override": self.search_space_override,
                "search_space_exclude": self.search_space_exclude,
                "hpo_sampler": self.hpo_sampler,
                "hpo_seed": self.hpo_seed,
                "hpo_storage_dir": self.hpo_storage_dir,
                "output_dir": self.output_dir,
                "uncertainty_method": self.uncertainty_method,
                "refit_on_train_val": self.refit_on_train_val,
                "out_of_range_action": self.out_of_range_action,
                "target_scaling": self.target_scaling,
                "input_scaling": self.input_scaling,
                "missing_inputs": self.missing_inputs,
                "task_type": self.task_type,
                "calculate_feature_importance": self.calculate_feature_importance,
                "feature_selection_threshold": self.feature_selection_threshold,
                "shap_max_data_points": self.shap_max_data_points,
            }
        )
        return params

    @property
    def target_scaling(self) -> TargetScaling:
        """The resolved target scaling (a `TargetScaling` member)."""
        return self._target_scaling

    @target_scaling.setter
    def target_scaling(self, value: TargetScaling | str | None) -> None:
        """Coerces a value a subclass copies from its constructor kwargs (a string from a config); None leaves the resolved setting."""
        if value is not None:
            self._target_scaling = TargetScaling(value)

    def _resolve_target_scaling(self, requested: TargetScaling | str | None) -> TargetScaling:
        """The setting this model runs with: the request, else the family default; classifiers never scale.

        Raises:
            ValueError: PER_OUTPUT requested for a classifier or for a class that does not map its outputs back.
        """
        requested = None if requested is None else TargetScaling(requested)
        if self.task_type == TaskType.CLASSIFICATION:
            if requested is TargetScaling.PER_OUTPUT:
                raise ValueError("target_scaling=PER_OUTPUT is for regression targets: a classifier is fitted on class labels, set target_scaling=NONE (or leave it unset).")
            return TargetScaling.NONE
        resolved = (self._default_target_scaling if self._supports_target_scaling else TargetScaling.NONE) if requested is None else requested
        if resolved is TargetScaling.PER_OUTPUT and not self._supports_target_scaling:
            raise ValueError(f"{type(self).__name__} does not map all its outputs back to target units under target_scaling=PER_OUTPUT: set target_scaling=NONE.")
        return resolved

    @property
    def input_scaling(self) -> InputScaling:
        """The resolved input scaling (an `InputScaling` member)."""
        return self._input_scaling

    @input_scaling.setter
    def input_scaling(self, value: InputScaling | str | None) -> None:
        """Coerces a value a subclass copies from its constructor kwargs (a string from a config); None leaves the resolved setting."""
        if value is not None:
            self._input_scaling = self._resolve_input_scaling(value)

    @property
    def missing_inputs(self) -> MissingInputs:
        """The missing-input rule (a `MissingInputs` member)."""
        return self._missing_inputs

    @missing_inputs.setter
    def missing_inputs(self, value: MissingInputs | str | None) -> None:
        """Coerces a value a subclass copies from its constructor kwargs; None leaves the setting."""
        if value is not None:
            self._missing_inputs = MissingInputs(value)

    def _resolve_input_scaling(self, requested: InputScaling | str | None) -> InputScaling:
        """The input scaling this model runs with: the request, else the family default.

        Raises:
            ValueError: STANDARD requested for a class that scales its inputs itself or bypasses `predict`.
        """
        requested = None if requested is None else InputScaling(requested)
        resolved = (self._default_input_scaling if self._supports_input_scaling else InputScaling.NONE) if requested is None else requested
        if resolved is InputScaling.STANDARD and not self._supports_input_scaling:
            raise ValueError(f"{type(self).__name__} scales its own inputs or predicts without `predict` under input_scaling=STANDARD: set input_scaling=NONE.")
        return resolved

    def _make_input_transform(self) -> InputTransform | None:
        """A fresh `InputTransform` for this model's settings, or None when it would neither check nor change anything (NONE + NATIVE)."""
        if self.input_scaling is InputScaling.NONE and self.missing_inputs is MissingInputs.NATIVE:
            return None
        unscaled = [*(getattr(self, "categorical_features", None) or []), *([self.offset_feature] if getattr(self, "offset_feature", None) else [])]
        return InputTransform(self.input_scaling, self.missing_inputs, unscaled_columns=unscaled)

    def _inputs_to_fit_space(self, x: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
        """Raw inputs to the space the model trained in (the one conversion; `predict` and the wrapped public methods call it).

        A DataFrame keeps its columns and index; an array holds either all fitted columns or the selected features, in order.
        """
        transform = self.input_transform_
        if transform is None:
            return x
        if isinstance(x, pd.DataFrame):
            values = transform.transform(x.values, list(x.columns))
            return x if values is x.values else pd.DataFrame(values, columns=x.columns, index=x.index)
        x = np.asarray(x)
        n_fitted = len(transform.columns_ or {})
        if x.shape[1] == n_fitted:
            return transform.transform(x)
        if self.selected_features_ and x.shape[1] == len(self.selected_features_):
            return transform.transform(x, self.selected_features_)
        raise ValueError(f"{self.name}: inputs with {x.shape[1]} columns match neither the {n_fitted} fitted columns nor the selected features.")

    def _target_scaling_conflict(self) -> str | None:
        """What in this model's settings cannot work with a rescaled target (a count loss, a per-row offset), or None. Subclasses extend it."""
        return None

    def _check_target_scaling(self) -> None:
        """Raises ValueError when PER_OUTPUT is set together with a setting named by `_target_scaling_conflict`."""
        conflict = self._target_scaling_conflict() if self.target_scaling is TargetScaling.PER_OUTPUT else None
        if conflict is not None:
            raise ValueError(f"target_scaling=PER_OUTPUT cannot be combined with {conflict}: set target_scaling=NONE.")

    def _to_fit_space(self, y: np.ndarray) -> np.ndarray:
        """Raw target units to the space the model trained in: scaled once training is over, unchanged while it runs and on trial models."""
        return y if self.target_scaler_ is None else self.target_scaler_.transform(y)

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

    def _perform_feature_selection(self, x_train_val: pd.DataFrame, y_train_val: np.ndarray, iterations: int | None, sample_weight: np.ndarray | None = None) -> None:
        """Performs feature selection and returns the filtered training data."""
        temp_model = self._clone_for_trial()
        logger.info("--- Training model on all the features in preparation for feature selection ---")
        temp_model._fit_single(x_train_val.values, y_train_val, forced_iterations=iterations, **self._weight_kwargs(sample_weight))

        feature_importance = self.calculate_feature_importances(model_instance=temp_model, x_background=x_train_val)

        if self.feature_selection_threshold is not None:
            self.selected_features_ = select_features_by_cumulative_importance(feature_importance, self.feature_selection_threshold)
            logger.info(f"--- Selected {len(self.selected_features_)} features ---")

    # Fields `fit` owns on the outer model; `_adopt_fitted_state` never overwrites them with a trial model's values.
    _OUTER_OWNED_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "feature_names",
            "feature_to_idx_",
            "selected_features_",
            "train_indices",
            "val_indices",
            "test_indices",
            "cv_score_mean_",
            "cv_score_std_",
            "num_iterations_used",
            "training_envelope_",
            "last_prediction_report_",
            "target_scaler_",
            "_raw_target_range_",
            "_spread_call_depth",
            "input_transform_",
            "_input_call_depth",
        }
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
        weights_train: np.ndarray | None = None,
        weights_val: np.ndarray | None = None,
        weights_train_val: np.ndarray | None = None,
    ) -> None:
        """Runs the search / early-stopping pass.

        With ``use_explicit_split`` (a caller-given validation set, or the one-pass protocol) the search scores
        its trials on ``(x_train, x_val)`` instead of drawing an inner split from train + validation. Under
        ``refit_on_train_val=False`` this method ends with `self` fitted (best trial adopted, or one early-stopped pass).
        The ``weights_*`` are the row weights of those parts (`fit`'s ``sample_weight``), None without weights.
        """
        fit_weights = self._weight_kwargs(weights_train, weights_val)
        if self.optimize_hyperparameters:
            logger.info("--- Starting hyperparameter optimization ---")
            split = (x_train.values, y_train, x_val.values, y_val) if use_explicit_split and x_val is not None else None
            search_weights = {} if weights_train_val is None else {"sample_weight": weights_train_val, "explicit_weights": (weights_train, weights_val) if split else None}
            best_params, self.num_iterations_used = self._find_best_hyperparameters(
                x=x_train_val.values, y=y_train_val, timestamps=timestamps_train_val, explicit_split=split, **search_weights
            )
            self._update_params(best_params)
            if not self.refit_on_train_val:
                self._adopt_fitted_state(self._best_trial_model_)
                self._best_trial_model_ = None
            logger.info(f"--- Hyperparameter optimization finished. Best params: {self.get_params()} ---")
            logger.info(f"--- Optimal iterations from HPO: {self.num_iterations_used} ---")
        elif not self.refit_on_train_val:
            logger.info("--- One-pass fit on the training part, validation held out ---")
            self.num_iterations_used, _ = self._fit_single(x_train.values, y_train, x_val=x_val.values if x_val is not None else None, y_val=y_val, **fit_weights)
        elif self.early_stopping_rounds is not None:
            # Step 2: Train the Model
            if self.cv_folds:
                logger.info("--- Finding optimal iterations with cross-validation ---")
                self.num_iterations_used, self.cv_score_mean_, self.cv_score_std_ = self._find_optimal_iterations_with_cv(
                    x=x_train_val.values, y=y_train_val, timestamps=timestamps_train_val, **self._sample_weight_kwarg(weights_train_val)
                )
            else:
                # Fit with a single train/validation split to find the best number of iterations
                logger.info("--- Finding optimal iterations with train/validation split ---")
                self.num_iterations_used, _ = self._fit_single(x_train.values, y_train, x_val=x_val.values if x_val is not None else None, y_val=y_val, **fit_weights)

            logger.info(f"--- Optimal iterations found: {self.num_iterations_used} ---")

    def _fit_final_model(self, x_train: np.ndarray, y_train: np.ndarray, sample_weight: np.ndarray | None = None) -> None:
        """Retrain on the full training + validation set with the optimal number of iterations.

        ``num_iterations_used`` is 0 until HPO or an early-stopping pass sets it. Passing
        0 to tree models treats it as literal "train 0 rounds" and crashes. When no
        optimisation pass has run we hand ``None`` (use the model's configured default).
        """
        logger.info("--- Training final model with optimal parameters ---")
        forced = self.num_iterations_used if self.num_iterations_used and self.num_iterations_used > 0 else None
        self._fit_single(x_train, y_train, forced_iterations=forced, **self._weight_kwargs(sample_weight))
        logger.info("--- Final model training finished ---")

    def fit(
        self,
        x: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.DataFrame | pd.Series,
        timestamps: np.ndarray | None = None,
        x_val: np.ndarray | pd.DataFrame | None = None,
        y_val: np.ndarray | pd.DataFrame | pd.Series | None = None,
        sample_weight: np.ndarray | None = None,
        sample_weight_val: np.ndarray | None = None,
    ) -> None:
        """Fits the model to the training data and evaluates it on all available partitions.

        Args:
            x: training inputs.
            y: training targets.
            timestamps: optional timestamps for time-ordered splits (ignored with an explicit validation set).
            x_val: optional explicit validation inputs. When given (with ``y_val``) they are the validation set,
                ``(x, y)`` is the training part whole, and no inner split is drawn (so no test rows are set aside).
            y_val: validation targets for ``x_val``.
            sample_weight: optional weight per row of ``x``, relative (rescaled to mean 1). The weights follow every split, fold and search
                trial: the training loss is the weighted mean of the per-row loss and every held-out score (early stopping, search, CV) the
                weighted mean of the per-row fit term. Scaling statistics, the in-sample residual spread and the evaluation reports stay
                unweighted. A class that cannot apply weights refuses them (`_sample_weight_conflict`).
            sample_weight_val: the weight per row of ``x_val``; required with ``sample_weight`` and an explicit validation set.
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
        self.training_envelope_ = TrainingEnvelope.from_fit_data(x, y, self.feature_names, getattr(self, "categorical_features", None))
        self._check_target_scaling()
        self.target_scaler_, self._raw_target_range_ = None, None  # a refit starts in fit space again

        explicit_val = x_val is not None
        weights = self._check_fit_weights(sample_weight, sample_weight_val, len(x), None if x_val is None else len(x_val))
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

        if weights is not None:  # the same row indices as every other array (`val_indices` index [x; x_val] under an explicit split)
            weights_train = weights[self.train_indices]
            weights_val = weights[self.val_indices] if self.val_indices.size else None
            weights_train_val = weights[np.concatenate([self.train_indices, self.val_indices])]
            weighted = {"weights_train": weights_train, "weights_val": weights_val, "weights_train_val": weights_train_val}
        else:
            weights_train_val, weighted = None, {}

        self.input_transform_ = None  # a refit starts with raw inputs again
        input_transform = self._make_input_transform()
        x_train_raw, x_train_val_raw = x_train, x_train_val
        if input_transform is not None:  # the inputs' missing-value rule and scaling: training-part statistics, applied to every internal path
            input_transform.fit(x_train, self.feature_names)
            x_train, x_train_val = input_transform.transform(x_train), input_transform.transform(x_train_val)
            x_val = None if x_val is None else input_transform.transform(x_val)
            if x_test is not None:
                input_transform.check(x_test, "The test inputs")  # the test part stays raw: it is evaluated through `predict`

        scaler = TargetScaler().fit(y_train) if self.target_scaling is TargetScaling.PER_OUTPUT else None
        y_train_raw, y_train_val_raw = y_train, y_train_val
        if scaler is not None:  # from here to the end of training every path sees fit space; the test part and the post-fit calls stay raw
            logger.info(f"Target scaling is on: per-output mean {scaler.mean_.tolist()}, sd {scaler.scale_.tolist()}")
            y_train, y_train_val = scaler.transform(y_train), scaler.transform(y_train_val)
            y_val = None if y_val is None else scaler.transform(y_val)

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
            **weighted,
        )

        if self.refit_on_train_val:
            if self.feature_selection_threshold is not None:
                self._perform_feature_selection(x_train_val, y_train_val, self.num_iterations_used, **self._sample_weight_kwarg(weights_train_val))

            x_train_val_final = x_train_val[self.selected_features_] if self.selected_features_ else x_train_val
            self._fit_final_model(x_train=x_train_val_final.values, y_train=y_train_val, **self._sample_weight_kwarg(weights_train_val))
            x_model, y_model = x_train_val_final, y_train_val
        else:
            x_model, y_model = x_train, y_train  # the model was fitted on the training part only

        if self.uncertainty_method == UncertaintyMethod.BINNED_RESIDUAL_STD:
            logger.info("--- Calibrating uncertainty model ---")
            self.calibrate_uncertainty(x=x_model.values, y=y_model)

        if scaler is not None:  # training is over: from here the public outputs are in target units
            self.target_scaler_ = scaler
            if getattr(self, "output_bound", OutputBound.NONE) is OutputBound.TRAINING_RANGE:  # the bound holds in fit space; unscaling must not leave the range by rounding
                raw = np.asarray(y_train_raw, dtype=np.float64).reshape(len(y_train_raw), -1)
                self._raw_target_range_ = (np.nanmin(raw, axis=0), np.nanmax(raw, axis=0))
            y_train_val = y_train_val_raw

        if input_transform is not None:  # training is over: from here the public methods take raw inputs
            self.input_transform_ = input_transform
            x_train_val = pd.DataFrame(x_train_val_raw, columns=self.feature_names)
            x_model_raw = x_train_val if self.refit_on_train_val else pd.DataFrame(x_train_raw, columns=self.feature_names)
            x_model = x_model_raw[self.selected_features_] if self.refit_on_train_val and self.selected_features_ else x_model_raw

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

    def _sample_weight_conflict(self) -> str | None:
        """Why this model, as configured, cannot apply sample weights, with the remedy; None when it can."""
        if not self._supports_sample_weight:
            return f"{type(self).__name__} does not apply sample weights: fit it without sample_weight"
        if self.uncertainty_method == UncertaintyMethod.BINNED_RESIDUAL_STD:
            return "the binned residual spread is calibrated without weights: use uncertainty_method=CONSTANT, or fit without sample_weight"
        return None

    def _check_fit_weights(self, sample_weight: np.ndarray | None, sample_weight_val: np.ndarray | None, n_rows: int, n_val_rows: int | None) -> np.ndarray | None:
        """The checked weights of the rows `fit` splits (``[x; x_val]`` under an explicit validation set), or None without weights.

        Raises:
            ValueError: weights this model cannot apply (`_sample_weight_conflict`), validation weights without training weights or the
                other way round under an explicit validation set, or invalid weights (`data_handler.check_sample_weight`).
        """
        if sample_weight is None and sample_weight_val is None:
            return None
        if (conflict := self._sample_weight_conflict()) is not None:
            raise ValueError(f"sample_weight cannot be used here: {conflict}.")
        if sample_weight is None or (n_val_rows is None) != (sample_weight_val is None):
            raise ValueError("Give sample_weight for the rows of x, and sample_weight_val exactly when an explicit x_val is given.")
        weights = check_sample_weight(sample_weight, n_rows, "sample_weight")
        weights = weights / weights.mean()  # weights are relative: each array is rescaled to mean 1, so penalties and leaf-size limits keep their size
        if n_val_rows is None:
            return weights
        weights_val = check_sample_weight(sample_weight_val, n_val_rows, "sample_weight_val")
        return np.concatenate([weights, weights_val / weights_val.mean()])

    @staticmethod
    def _weight_kwargs(train: np.ndarray | None, val: np.ndarray | None = None) -> dict[str, np.ndarray | None]:
        """`_fit_single` keyword arguments for row weights: none without weights, so a class that takes none is called as before."""
        return {} if train is None else {"sample_weight": train, "sample_weight_val": val}

    @staticmethod
    def _sample_weight_kwarg(weights: np.ndarray | None) -> dict[str, np.ndarray]:
        """``sample_weight=`` for a held-out score or a single-set call: none without weights."""
        return {} if weights is None else {"sample_weight": weights}

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
            time_gap=self.time_gap,
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
        sample_weight: np.ndarray | None = None,
        explicit_weights: tuple[np.ndarray, np.ndarray | None] | None = None,
    ) -> tuple[float, int]:
        """Evaluates a trial's performance and returns the score and optimal iterations.

        ``explicit_split = (x_train, y_train, x_val, y_val)`` replaces the inner split drawn from ``(x, y)``. ``sample_weight`` weighs the
        rows of ``(x, y)``; ``explicit_weights = (train, val)`` the parts of the explicit split.
        """
        if self.cv_folds:
            kf = self.get_kfolds(timestamps=timestamps)
            fold_results = []
            for train_idx, val_idx in kf.split(x, y):
                x_train_fold, x_val_fold = x[train_idx], x[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                w_train, w_val = (None, None) if sample_weight is None else (sample_weight[train_idx], sample_weight[val_idx])
                best_iter, loss_history = model_instance._fit_single(x_train_fold, y_train_fold, x_val=x_val_fold, y_val=y_val_fold, **self._weight_kwargs(w_train, w_val))
                score = model_instance._held_out_fit_score(x_val_fold, y_val_fold, **self._sample_weight_kwarg(w_val))
                fold_results.append({"best_iter": best_iter, "loss_history": loss_history, "score": score})

            final_score = float(np.mean([res["score"] for res in fold_results]))
            optimal_iterations = find_optimal_iterations(fold_results)
        else:
            if explicit_split is not None:
                x_train, y_train, x_val, y_val = explicit_split
                w_train, w_val = explicit_weights or (None, None)
            else:
                train_indices, val_indices, _ = create_train_val_split(
                    x=x,
                    validation_fraction=self.validation_fraction,
                    test_fraction=0,
                    split_strategy=self.split_strategy,
                    timestamps=timestamps,
                    random_state=getattr(self, "random_seed", None),
                    time_gap=self.time_gap,
                )
                x_train, y_train = x[train_indices], y[train_indices]
                x_val, y_val = x[val_indices], y[val_indices]
                w_train, w_val = (None, None) if sample_weight is None else (sample_weight[train_indices], sample_weight[val_indices])
            optimal_iterations, _ = model_instance._fit_single(x_train, y_train, x_val=x_val, y_val=y_val, **self._weight_kwargs(w_train, w_val))
            final_score = model_instance._held_out_fit_score(x_val, y_val, **self._sample_weight_kwarg(w_val))
        return final_score, optimal_iterations

    def _find_best_hyperparameters(
        self,
        x: np.ndarray,
        y: np.ndarray,
        timestamps: np.ndarray | None = None,
        explicit_split: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
        sample_weight: np.ndarray | None = None,
        explicit_weights: tuple[np.ndarray, np.ndarray | None] | None = None,
    ) -> tuple[dict[str, Any], int]:
        """Finds the best hyperparameters and optimal iterations using Optuna (weights as in `_evaluate_trial_performance`).

        Under ``refit_on_train_val=False`` the best trial's fitted model is kept in ``self._best_trial_model_``
        (only the best so far is held in memory), for `fit` to adopt instead of refitting.
        """
        smaller_is_better = self._held_out_score_is_smaller_better()
        best_trial: dict[str, Any] = {"score": None, "model": None, "number": None}
        weights = {} if sample_weight is None else {"sample_weight": sample_weight, "explicit_weights": explicit_weights}

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

            score, iterations = self._evaluate_trial_performance(model_instance, x, y, timestamps=timestamps, explicit_split=explicit_split, **weights)
            trial.set_user_attr("iterations", iterations)
            if not self.refit_on_train_val and (best_trial["score"] is None or (score < best_trial["score"] if smaller_is_better else score > best_trial["score"])):
                best_trial.update(score=score, model=model_instance, number=trial.number)
            return score

        direction = "minimize" if smaller_is_better else "maximize"
        seed = self.hpo_seed if self.hpo_seed is not None else getattr(self, "random_seed", None)
        journal = self._hpo_journal(x, y, timestamps, explicit_split, **weights)
        optimizer = OptunaOptimizer(direction=direction, n_trials=self.n_trials, random_seed=seed, sampler=self.hpo_sampler, **journal)
        study = optimizer.optimize(objective)
        best_iterations = study.best_trial.user_attrs["iterations"]
        if not self.refit_on_train_val and best_trial["number"] != study.best_trial.number:
            # The best trial finished before a restart: its fitted model is gone, so fit its parameters again on the same split and seed.
            logger.info(f"Best trial {study.best_trial.number} was finished by an earlier process: refitting its parameters.")
            best_trial["model"] = self._clone_for_trial()
            best_trial["model"]._update_params(study.best_trial.params)
            _, best_iterations = self._evaluate_trial_performance(best_trial["model"], x, y, timestamps=timestamps, explicit_split=explicit_split, **weights)
        self._best_trial_model_ = best_trial["model"]
        return study.best_trial.params, best_iterations

    def _hpo_journal(
        self,
        x: np.ndarray,
        y: np.ndarray,
        timestamps: np.ndarray | None,
        explicit_split: tuple | None,
        sample_weight: np.ndarray | None = None,
        explicit_weights: tuple | None = None,
    ) -> dict[str, Any]:
        """``OptunaOptimizer`` keyword arguments that make the search resumable (none without ``hpo_storage_dir``).

        The journal file is named by the model class and a hash of its settings (not ``n_trials``, so raising it continues the search)
        and of the data (with the row weights when there are any), so a different config or data never continues another search.
        """
        if self.hpo_storage_dir is None:
            return {}
        settings = {k: v for k, v in self.get_params().items() if k not in HPO_IDENTITY_EXCLUDED}
        for key in self._settings_without_effect() & settings.keys():
            settings.pop(key)  # as before the setting existed: a search that does not use it keeps its journal
        data = [x, y, timestamps, explicit_split] + ([sample_weight, explicit_weights] if sample_weight is not None else [])
        name = f"{type(self).__name__}-{stable_hash([settings, *data])[:HPO_HASH_LENGTH]}"
        return {"storage_path": Path(self.hpo_storage_dir) / f"{name}.journal", "study_name": name}

    def _settings_without_effect(self) -> set[str]:
        """Names of the settings whose current value runs the model as it ran before the setting existed; `_hpo_journal` leaves them out."""
        off = {
            "target_scaling": self.target_scaling is TargetScaling.NONE,
            "input_scaling": self.input_scaling is InputScaling.NONE,
            "missing_inputs": self.missing_inputs is not MissingInputs.TRAINING_MEAN,  # RAISE and NATIVE never change a fit on data without NaN
            "time_gap": self.time_gap == 0,
        }
        return {name for name, is_off in off.items() if is_off}

    def _get_optimization_metric(self) -> Metric:
        """Gets the optimization metric for the model."""
        if self.task_type == TaskType.REGRESSION:
            optimization_metric = Metric.NLL if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC else Metric.MSE
        else:
            optimization_metric = Metric.LOG_LOSS
        return optimization_metric

    def _held_out_fit_score(self, x_val: np.ndarray, y_val: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
        """Held-out score for hyperparameter search and cross-validation, on a model ALREADY fitted on the training part.

        Call it on the trial / fold model instance, never on the outer model (scoring the unfitted outer
        object was the F1 crash). Default: the model's optimisation metric on its predictions (trees and
        linear models). `PyTorchModelBase` overrides it with the held-out fit term of the training loss
        (owner ruling 2026-11-01, DECISIONS.md "Validation score = the held-out FIT TERM"). Direction is
        given by `_held_out_score_is_smaller_better`. With ``sample_weight``: the weighted mean of `_held_out_fit_per_sample`.
        """
        if sample_weight is not None:
            return float(np.average(self._held_out_fit_per_sample(x_val, y_val), weights=sample_weight))
        preds = self._to_fit_space(self._predict_for_scoring(x_val))
        y_pred_std = self.predict_uncertainty(x_val)
        if self.target_scaler_ is not None and y_pred_std is not None:
            y_pred_std = self.target_scaler_.spread_transform(y_pred_std)
        return self._evaluate_trial(self._to_fit_space(y_val), preds, y_pred_std=y_pred_std)

    def _held_out_score_is_smaller_better(self) -> bool:
        """Direction of `_held_out_fit_score` for the Optuna study."""
        return self._get_optimization_metric().is_smaller_better

    def _held_out_fit_per_sample(self, x_val: np.ndarray, y_val: np.ndarray) -> np.ndarray:
        """Per-sample held-out fit term (lower is better) on a model already fitted on the training part, shape ``(N,)``.

        Default: the squared error of the prediction, the fit term of a model scored by MSE (`_get_optimization_metric`).
        `PyTorchModelBase` overrides it with the fit term of its training loss.

        Raises:
            NotImplementedError: if the model is not scored by MSE (its fit term is then not the squared error).
        """
        if self._get_optimization_metric() != Metric.MSE:
            raise NotImplementedError(f"{type(self).__name__} is scored by {self._get_optimization_metric().name}: it must define its own _held_out_fit_per_sample.")
        predictions = np.asarray(self._to_fit_space(self._predict_for_scoring(x_val)), dtype=np.float64).ravel()
        return (predictions - np.asarray(self._to_fit_space(y_val), dtype=np.float64).ravel()) ** 2

    def fit_sweep_selector(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        *,
        capacity_grid: list[int] | None = None,
        n_bootstrap: int = 1000,
        rule: KSelectionRule = KSelectionRule.CHEAPEST_WITHIN_TOLERANCE,
        early_stop_on_val: bool = False,
    ) -> int:
        """Brute-force choice of the number of classes: one model per candidate k, the best on held-out data kept.

        Any model with an ``n_classes`` parameter. Each candidate is this model's own settings at its k
        (`_sweep_candidate_params`), fitted by `_fit_sweep_candidate` and scored per sample on ``(x_val, y_val)`` by its
        held-out fit term (`_held_out_fit_per_sample`). ``rule`` picks the winner: MINIMUM the lowest mean,
        CHEAPEST_WITHIN_TOLERANCE the smallest k within tolerance of the best (`utils.capacity_selection.cheapest_within_tolerance`,
        ``n_bootstrap`` resamples). The winner is kept as ``_sweep_submodel_`` and `predict` delegates to it; ``selected_k_``
        and ``sweep_val_errors_`` (mean held-out fit term per k) are stored, then `_record_sweep` runs.

        Args:
            x_train: inputs every candidate is trained on.
            y_train: targets every candidate is trained on.
            x_val: held-out inputs the candidates are scored on.
            y_val: held-out targets the candidates are scored on.
            capacity_grid: candidate k values, cheapest first (default `_default_sweep_grid`).
            n_bootstrap: bootstrap resamples of the CHEAPEST_WITHIN_TOLERANCE rule.
            rule: how the winner is read off the held-out curve.
            early_stop_on_val: fit each candidate in one pass, early-stopped on ``(x_val, y_val)`` (which then also choose k:
                the study protocol), instead of on an inner split of ``(x_train, y_train)``.

        Returns:
            The selected k.

        Raises:
            TypeError: if the model has no ``n_classes`` parameter.
            ValueError: if a candidate is below `MIN_SWEEP_K`.
        """
        if not hasattr(self, "n_classes"):
            raise TypeError(f"{type(self).__name__} has no n_classes parameter for fit_sweep_selector to sweep.")
        grid = list(capacity_grid) if capacity_grid is not None else self._default_sweep_grid()
        if min(grid) < MIN_SWEEP_K:
            raise ValueError(f"fit_sweep_selector candidates start at k={MIN_SWEEP_K}: a one-class {type(self).__name__} is a constant predictor (got {grid}).")
        error_table = np.empty((len(np.asarray(y_val).ravel()), len(grid)), dtype=np.float64)
        candidates = []
        for col, k in enumerate(grid):
            candidate = type(self)(**self._sweep_candidate_params(k, early_stop_on_val=early_stop_on_val))
            self._fit_sweep_candidate(candidate, x_train, y_train, x_val, y_val, early_stop_on_val=early_stop_on_val)
            error_table[:, col] = candidate._held_out_fit_per_sample(x_val, y_val)
            candidates.append(candidate)
        mean_error = error_table.mean(axis=0)
        seed = getattr(self, "random_seed", None) or 0
        idx = int(np.argmin(mean_error)) if rule == KSelectionRule.MINIMUM else cheapest_within_tolerance(error_table, n_boot=n_bootstrap, seed=seed)
        self.selected_k_ = grid[idx]
        self.sweep_val_errors_ = dict(zip(grid, mean_error.tolist(), strict=True))
        self._sweep_submodel_ = candidates[idx]
        self._record_sweep(grid, error_table, self.selected_k_)
        return self.selected_k_

    def _default_sweep_grid(self) -> list[int]:
        """Candidate k when `fit_sweep_selector` is given none: the model's ``n_classes`` search range.

        Raises:
            ValueError: if the search space has no ``n_classes`` range.
        """
        space = self.get_hyperparameter_search_space().get("n_classes")
        if space is None:
            raise ValueError(f"{type(self).__name__} has no n_classes search range: pass capacity_grid to fit_sweep_selector.")
        return list(range(max(MIN_SWEEP_K, int(space["low"])), int(space["high"]) + 1))

    def _sweep_candidate_params(self, k: int, *, early_stop_on_val: bool) -> dict[str, Any]:
        """Constructor settings of the sweep candidate at ``k``: this model's own settings with ``n_classes = k``."""
        params = self.get_params()
        params["n_classes"] = k
        params["calculate_feature_importance"] = self.calculate_feature_importance
        if early_stop_on_val:
            params["refit_on_train_val"] = False  # one pass: the validation rows stay held out, they also choose k
        return params

    def _fit_sweep_candidate(
        self, candidate: "BaseModel", x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, *, early_stop_on_val: bool
    ) -> None:
        """Fits one sweep candidate: one pass early-stopped on ``(x_val, y_val)``, or on its own inner split of the training rows."""
        if early_stop_on_val:
            candidate.fit(x_train, y_train, x_val=x_val, y_val=y_val)
        else:
            candidate.fit(x_train, y_train)

    def _record_sweep(self, capacity_grid: list[int], error_table: np.ndarray, selected_k: int) -> None:
        """Hook run after `fit_sweep_selector` has chosen ``selected_k`` (per-sample held-out errors ``error_table``, one column per k)."""

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
            kf = TimeSeriesSplit(n_splits=self.cv_folds, gap=self.time_gap, timestamps=timestamps)
        elif self.split_strategy == DataSplitStrategy.RANDOM:
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=getattr(self, "random_seed", None))
        else:
            raise ValueError(f"Split strategy {self.split_strategy.value} is not supported.")
        return kf

    def _find_optimal_iterations_with_cv(
        self, x: np.ndarray, y: np.ndarray, timestamps: np.ndarray | None = None, sample_weight: np.ndarray | None = None
    ) -> tuple[int, float, float]:
        """Finds the optimal number of iterations using cross-validation (``sample_weight``: the weight of each row of ``(x, y)``)."""
        kf = self.get_kfolds(timestamps=timestamps)
        fold_results = []
        for i, (train_idx, val_idx) in enumerate(kf.split(x, y)):
            logger.info(f"--- Training CV fold {i + 1}/{self.cv_folds} ---")
            x_train_fold, x_val_fold = x[train_idx], x[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            model_instance = self._clone_for_trial()
            w_train, w_val = (None, None) if sample_weight is None else (sample_weight[train_idx], sample_weight[val_idx])
            best_iter, loss_history = model_instance._fit_single(x_train_fold, y_train_fold, x_val=x_val_fold, y_val=y_val_fold, **self._weight_kwargs(w_train, w_val))
            score = model_instance._held_out_fit_score(x_val_fold, y_val_fold, **self._sample_weight_kwarg(w_val))
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
    def _predict(self, x: np.ndarray | pd.DataFrame, filter_data: bool = True, **kwargs: Any) -> np.ndarray:
        """Model-specific prediction; subclasses implement this, callers use `predict`."""
        raise NotImplementedError

    def predict(self, x: np.ndarray | pd.DataFrame, filter_data: bool = True, **kwargs: Any) -> np.ndarray:
        """Makes predictions on new data and checks them against the training envelope.

        Calls `_predict`, then (once the model is fitted) compares the predictions and the input rows with the range of the
        training data (``training_envelope_``, recorded by `fit`; see ``utils/prediction_envelope.py``). The resulting
        `PredictionRangeReport` is stored in ``last_prediction_report_`` and one INFO line with the counts is logged on every call,
        also when everything is in range. Then ``out_of_range_action`` applies: FLAG returns the predictions unchanged (bit-identical
        to `_predict`), CLIP clips each output column to its training [min, max], RAISE raises ValueError. NaN and infinite
        predictions count as out of range.

        Args:
            x: inputs.
            filter_data: whether to restrict ``x`` to the selected features.
            **kwargs: model-specific options passed to `_predict` (e.g. ``hard_execution``, ``width``).

        Returns:
            The predictions.

        Raises:
            ValueError: under RAISE, when any prediction is outside the training range.
        """
        x_raw = x
        if self.input_transform_ is not None and not self._input_call_depth:
            x = self._inputs_to_fit_space(x)
        self._input_call_depth += 1
        try:
            winner = getattr(self, "_sweep_submodel_", None)
            if winner is not None:  # after `fit_sweep_selector` the winning candidate is the fitted model
                return winner.predict(x, filter_data=filter_data, **kwargs)
            predictions = self._predict(x, filter_data=filter_data, **kwargs)
        finally:
            self._input_call_depth -= 1
        x = x_raw  # the envelope is recorded on raw inputs
        if self.target_scaler_ is not None:  # the one place forecasts go back to target units (before the envelope check, which is in target units)
            dtype = predictions.dtype
            predictions = self.target_scaler_.inverse(predictions)
            if self._raw_target_range_ is not None:
                predictions = np.clip(predictions, *self._raw_target_range_)
            predictions = predictions.astype(dtype)
        envelope = self.training_envelope_
        if envelope is None:
            return predictions
        input_columns = None
        if isinstance(x, np.ndarray) and self.selected_features_ and x.shape[1] == len(self.selected_features_) != len(envelope.feature_names):
            input_columns = self.selected_features_
        action = OutOfRangeAction(self.out_of_range_action)  # subclasses may re-set the raw constructor value
        report = check_prediction_range(envelope, x, predictions, input_columns=input_columns)
        self.last_prediction_report_ = report
        logger.info(f"{self.name} prediction range check: {report.summary()}")
        if action == OutOfRangeAction.RAISE and report.n_outputs_out_of_range:
            raise ValueError(f"{self.name}: {report.summary()}. The inputs or the model need fixing.")
        if action == OutOfRangeAction.CLIP:
            predictions, n_clipped = clip_to_envelope(envelope, predictions)
            logger.info(f"{self.name} clipped {n_clipped} prediction entries to the training range")
        return predictions

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


for _name in SPREAD_METHODS:
    setattr(BaseModel, _name, _spread_in_target_units(BaseModel.__dict__[_name]))
for _name in INPUT_METHODS:
    if callable(BaseModel.__dict__.get(_name)):
        setattr(BaseModel, _name, _inputs_in_fit_space(BaseModel.__dict__[_name]))

# BaseModel's own settings (its constructor arguments): a one-pass search never takes these from a trial clone.
_BASE_SETTINGS: frozenset[str] = frozenset(p for p in inspect.signature(BaseModel.__init__).parameters if p not in ("self", "kwargs"))
