# ruff: noqa: ERA001
"""Composite model for regression using classification and probability mapping."""

import os
from typing import Any, ClassVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from automl_package.enums import BoundaryRegularizationMethod, MapperType, Metric, MissingInputs, ModelName, OptimizerType, RegressionStrategy, TaskType, UncertaintyMethod
from automl_package.logger import logger
from automl_package.models.base import NESTED_PARAM_SEPARATOR, BaseModel
from automl_package.models.common.middle_class_penalty_mixin import MiddleClassPenaltyMixin
from automl_package.models.mappers.base_mapper import BaseMapper
from automl_package.models.mappers.nn_mapper import NeuralNetworkMapper
from automl_package.models.probability_mapper import ClassProbabilityMapper
from automl_package.models.registry import resolve_model_class
from automl_package.utils.data_handler import create_train_val_split
from automl_package.utils.feature_selection import select_features_by_cumulative_importance
from automl_package.utils.metrics import Metrics
from automl_package.utils.numerics import calculate_class_value_ranges, create_bins
from automl_package.utils.plotting import plot_nn_probability_mappers
from automl_package.utils.pytorch_utils import apply_law_of_total_variance, get_device


class ClassifierRegressionModel(BaseModel, MiddleClassPenaltyMixin):
    """A composite model that combines a classification model with a probability mapper.

    to perform regression. It first discretizes the target into N classes,
    trains a classifier on these classes, and then converts classification probabilities
    back to continuous regression output using various mapping strategies.
    """

    _supports_target_scaling: ClassVar[bool] = False  # target_scaling=PER_OUTPUT refused: two stages and a mapper-derived spread
    _supports_input_scaling: ClassVar[bool] = False  # its classifier applies its own input settings
    _default_missing_inputs: ClassVar[MissingInputs] = MissingInputs.NATIVE  # likewise for missing inputs

    # The model's own options and their defaults (the rest go to `BaseModel`). Declared so a config's keys are checked
    # against them and its strings turned into these enums (`experiments/registry.py`).
    _defaults: ClassVar[dict[str, Any]] = {
        "base_classifier_class": None,  # a model class, or its ModelName (member or name): `models/registry.py`
        "n_classes": 3,
        "auto_include_nn_mappers": True,
        "mapper_type": MapperType.SPLINE,
        "base_classifier_params": None,
        "mapper_params": None,
        "nn_mapper_params": None,
        "regression_strategy": None,
        "boundary_regularization_method": BoundaryRegularizationMethod.NONE,
        "boundary_loss_weight": 1.0,
        "use_monotonic_constraints": False,
        "constrain_middle_class": False,
        "apply_boundary_loss_during_validation": False,
        "use_middle_class_nll_penalty": False,
        "optimizer_type": OptimizerType.ADAM,
        "random_seed": None,  # seeds the classifier and the inner split
    }

    def __init__(self, **kwargs: Any) -> None:
        """Initializes the ClassifierRegressionModel (options: `_defaults`, plus every `BaseModel` setting)."""
        for key, default in self._defaults.items():
            setattr(self, key, kwargs.pop(key, default))
        self.device = get_device()

        if self.boundary_regularization_method == BoundaryRegularizationMethod.PENALTY and kwargs.get("uncertainty_method") == UncertaintyMethod.PROBABILISTIC:
            raise ValueError(
                "The PENALTY boundary method is not compatible with the PROBABILISTIC uncertainty method from a pure statistical perspective. Use the HARDSIGMOID method instead."
            )

        self.mapper_to_strategy_map = {
            MapperType.NN_SEPARATE_HEADS: RegressionStrategy.SEPARATE_HEADS,
            MapperType.NN_SINGLE_HEAD_N_OUTPUTS: RegressionStrategy.SINGLE_HEAD_N_OUTPUTS,
            MapperType.NN_SINGLE_HEAD_FINAL_OUTPUT: RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT,
        }

        super().__init__(**kwargs)
        MiddleClassPenaltyMixin.__init__(self, use_middle_class_nll_penalty=self.use_middle_class_nll_penalty, **kwargs)
        assert self.is_regression_model

        if isinstance(self.mapper_type, str):
            self.mapper_type = MapperType[self.mapper_type]

        if self.mapper_type.is_nn and self.regression_strategy is None:
            self.regression_strategy = self.mapper_to_strategy_map.get(self.mapper_type)

        if self.base_classifier_class is None:
            raise ValueError("base_classifier_class must be provided.")
        self.base_classifier_class = resolve_model_class(self.base_classifier_class)

        # Own copies: a search trial updates these dicts, which must never reach the model it was cloned from.
        self.base_classifier_params = dict(self.base_classifier_params or {})
        self.mapper_params = dict(self.mapper_params or {})
        self.nn_mapper_params = dict(self.nn_mapper_params or {})

        self.base_classifier: BaseModel | None = None
        self.class_boundaries: np.ndarray | None = None
        self.class_value_ranges_: np.ndarray | None = None
        self.precomputed_class_value_ranges_ = {}
        self.class_mappers: list[ClassProbabilityMapper | NeuralNetworkMapper | None] = []
        self.is_composite_regression_model = True
        self.optimal_mapper_params_ = {}
        self.task_type = TaskType.REGRESSION
        self._train_residual_std = 0.0

        if self.n_classes < 2:
            raise ValueError("n_classes must be at least 2 for classification-regression strategy.")

        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """Validates the combination of parameters provided to the model."""
        # This validation logic is moved from NeuralNetworkMapper to fail early.
        if self.mapper_type.is_nn:
            if self.regression_strategy == RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT and self.boundary_regularization_method == BoundaryRegularizationMethod.HARDSIGMOID:
                raise ValueError("The HARDSIGMOID boundary regularization method cannot be used with the SingleHeadFinalOutput strategy, as boundaries are defined per class.")

            if self.use_monotonic_constraints and self.regression_strategy != RegressionStrategy.SEPARATE_HEADS:
                raise ValueError(f"Monotonic constraints are only supported for the SEPARATE_HEADS strategy, but strategy is set to {self.regression_strategy.name}.")

            middle_class_options = self.constrain_middle_class or self.use_middle_class_nll_penalty
            if middle_class_options:
                if self.regression_strategy == RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT:
                    if self.constrain_middle_class:
                        raise ValueError(f"The `constrain_middle_class` option is not supported for the SINGLE_HEAD_FINAL_OUTPUT strategy.")

                    if self.use_middle_class_nll_penalty:
                        raise ValueError(f"The `use_middle_class_nll_penalty` option is not supported for the SINGLE_HEAD_FINAL_OUTPUT strategy.")

                if self.n_classes % 2 == 0:
                    logger.warning(
                        f"The `constrain_middle_class` or `use_middle_class_nll_penalty` options are enabled, but n_classes is {self.n_classes} (an even number). "
                        f"These options will have no effect."
                    )

    @property
    def name(self) -> str:
        """Returns the name of the model."""
        try:
            if self.base_classifier_class.__name__ in [ModelName.XGBOOST.value, ModelName.LIGHTGBM.value, ModelName.CATBOOST.value]:
                base_name = self.base_classifier_class(is_classification=True).name
            elif self.base_classifier_class.__name__ == ModelName.PYTORCH_NEURAL_NETWORK.value:
                base_name = self.base_classifier_class(input_size=10, output_size=1, task_type=TaskType.CLASSIFICATION).name
            else:
                base_name = self.base_classifier_class().name
        except Exception as e:
            logger.warning(f"Could not instantiate base_classifier_class to get name: {e}. Using 'UnknownBase'.")
            base_name = "UnknownBase"

        return f"{base_name}_to_Reg_{self.mapper_type.label}"

    def _fit_single_mapper(
        self,
        mapper_type: MapperType | str,
        probas: np.ndarray,
        y: np.ndarray,
        val_probas: np.ndarray | None = None,
        val_y: np.ndarray | None = None,
        forced_mapper_params: dict | None = None,
    ) -> tuple[list, dict]:
        """Fits a single mapper type and returns the fitted mapper and any learned parameters."""
        if isinstance(mapper_type, str):
            mapper_type = MapperType[mapper_type]
        learned_params = {}
        current_nn_params = self.nn_mapper_params.copy()
        if forced_mapper_params:
            current_nn_params.update(forced_mapper_params)

        current_nn_params["middle_class_dist_params"] = self.middle_class_dist_params_

        if mapper_type.is_nn:
            nn_mapper = NeuralNetworkMapper(
                n_classes=self.n_classes,
                regression_strategy=self.mapper_to_strategy_map[mapper_type],
                mapper_params=current_nn_params,
                uncertainty_method=self.uncertainty_method,
                early_stopping_rounds=self.early_stopping_rounds,
                boundary_regularization_method=self.boundary_regularization_method,
                boundary_loss_weight=self.boundary_loss_weight,
                class_value_ranges=self.class_value_ranges_,
                class_boundaries=self.class_boundaries,
                optimizer_type=self.optimizer_type,
                apply_boundary_loss_during_validation=self.apply_boundary_loss_during_validation,
                use_middle_class_nll_penalty=self.use_middle_class_nll_penalty,
                use_monotonic_constraints=self.use_monotonic_constraints,
                constrain_middle_class=self.constrain_middle_class,
            )

            # The NN mapper handles the train/val split internally if val_probas is provided
            train_indices = np.arange(probas.shape[0])
            val_indices = np.arange(probas.shape[0], probas.shape[0] + val_probas.shape[0]) if val_probas is not None else None
            combined_probas = np.vstack((probas, val_probas)) if val_probas is not None else probas
            combined_y = np.concatenate((y, val_y)) if val_y is not None else y

            learned_params = nn_mapper.fit(combined_probas, combined_y, train_indices=train_indices, val_indices=val_indices)
            fitted_mappers = [nn_mapper]
        else:
            # Non-NN mappers are simpler and fit only on the training data provided.
            fitted_mappers = []
            for c in range(self.n_classes):
                mapper = ClassProbabilityMapper(mapper_type, uncertainty_method=self.uncertainty_method, **self.mapper_params)
                mapper.fit(probas[:, c].reshape(-1, 1), y)
                fitted_mappers.append(mapper)
        return fitted_mappers, learned_params

    def _calibrate_mappers_if_needed(
        self,
        mapper_type: MapperType,
        mappers: list[BaseMapper],
        classifier: BaseModel,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        """Checks the uncertainty method and calibrates the mappers if required."""
        active_mapper = mappers[0]
        if isinstance(active_mapper, NeuralNetworkMapper):
            mapper_to_check = active_mapper
        else:
            mapper_to_check = active_mapper.mapper

        if mapper_to_check and mapper_to_check.uncertainty_method == UncertaintyMethod.BINNED_RESIDUAL_STD:
            x_full = np.vstack((x_train, x_val)) if x_val is not None else x_train
            y_full = np.concatenate((y_train.flatten(), y_val.flatten())) if y_val is not None else y_train.flatten()
            probas_full = classifier.predict_proba(x_full)

            if mapper_type.is_nn:
                mappers[0].calibrate_uncertainty(probas_full, y_full)
            else:
                for c, mapper in enumerate(mappers):
                    if mapper:
                        mapper.mapper.calibrate_uncertainty(probas_full[:, c].reshape(-1, 1), y_full)

    def predict_uncertainty(self, x: np.ndarray, filter_data: bool = True) -> np.ndarray:
        """Estimates uncertainty for regression predictions.

        The method depends on the uncertainty_method set for the model.
        - CONSTANT: Returns a constant uncertainty based on training residuals.
        - PROBABILISTIC: Applies the Law of Total Variance using learned uncertainties from mappers.

        Args:
            x (np.ndarray): Features for uncertainty estimation.
            filter_data (bool): If True, filter the input data using the feature selection mask.

        Returns:
            np.ndarray: Uncertainty estimates (standard deviation) for each prediction.
        """
        if self.base_classifier is None or not self.class_mappers:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        # Check the actual uncertainty method of the active mapper, which may have been remapped.
        active_mapper = self.class_mappers[0]
        if isinstance(active_mapper, NeuralNetworkMapper):
            active_uncertainty_method = active_mapper.uncertainty_method
        else:
            active_uncertainty_method = active_mapper.mapper.uncertainty_method

        uncertainties = None
        if active_uncertainty_method == UncertaintyMethod.CONSTANT:
            uncertainties = np.full(x.shape[0], self._train_residual_std)

        elif active_uncertainty_method in [UncertaintyMethod.BINNED_RESIDUAL_STD, UncertaintyMethod.PROBABILISTIC]:
            if filter_data:
                x = self._filter_predict_data(x)
            probas = self.base_classifier.predict_proba(x, filter_data=False)
            active_mapper = self.class_mappers[0]

            if isinstance(active_mapper, NeuralNetworkMapper):
                if active_mapper.regression_strategy == RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT:
                    total_variance = active_mapper.predict_variance(probas)
                    uncertainties = np.sqrt(total_variance)
                else:
                    # Apply the Law of Total Variance
                    mapper_means, mapper_variances = active_mapper.predict_mean_and_variance_per_class(probas)
                    probas_tensor = torch.from_numpy(probas).float().to(self.device)
                    means_tensor = torch.from_numpy(mapper_means).float().to(self.device).unsqueeze(-1)
                    log_vars_tensor = torch.log(torch.from_numpy(mapper_variances).float().to(self.device).clamp(min=1e-9)).unsqueeze(-1)
                    per_head_outputs = torch.cat([means_tensor, log_vars_tensor], dim=-1)
                    final_predictions = apply_law_of_total_variance(probas_tensor, per_head_outputs)
                    uncertainties = np.sqrt(torch.exp(final_predictions[:, 1]).cpu().numpy())
            else:
                # Logic for non-NN mappers
                mapper_means = np.zeros_like(probas)
                mapper_variances = np.zeros_like(probas)
                for c in range(self.n_classes):
                    if self.class_mappers[c] is not None:
                        class_probas = probas[:, c].reshape(-1, 1)
                        mapper_means[:, c] = self.class_mappers[c].predict(class_probas).flatten()
                        mapper_variances[:, c] = self.class_mappers[c].predict_variance_contribution(class_probas).flatten()

                probas_tensor = torch.from_numpy(probas).float().to(self.device)
                means_tensor = torch.from_numpy(mapper_means).float().to(self.device).unsqueeze(-1)
                log_vars_tensor = torch.log(torch.from_numpy(mapper_variances).float().to(self.device).clamp(min=1e-9)).unsqueeze(-1)
                per_head_outputs = torch.cat([means_tensor, log_vars_tensor], dim=-1)
                final_predictions = apply_law_of_total_variance(probas_tensor, per_head_outputs)
                uncertainties = np.sqrt(torch.exp(final_predictions[:, 1]).cpu().numpy())

        if uncertainties is None:
            raise NotImplementedError(f"Uncertainty method '{active_uncertainty_method}' is not implemented in predict_uncertainty.")

        return uncertainties

    def _mapper_predict(self, probas: np.ndarray, mapper_type_to_test: MapperType, fitted_mappers: list[BaseMapper]) -> np.ndarray:
        if mapper_type_to_test.is_nn:
            predictions = fitted_mappers[0].predict(probas)
        else:
            predictions = np.zeros(probas.shape[0])
            for c in range(self.n_classes):
                proba_for_current_class = probas[:, c].reshape(-1, 1)
                expected_y_from_mapper = fitted_mappers[c].predict(proba_for_current_class)
                predictions += proba_for_current_class.flatten() * expected_y_from_mapper
        return predictions

    def _get_base_classifier_params(self, input_size: int) -> dict[str, Any]:
        """Helper to construct the parameter dictionary for the base classifier."""
        classifier_output_size = 1 if self.n_classes == 2 else self.n_classes
        params = self.base_classifier_params.copy()
        params.update(
            {
                "task_type": TaskType.CLASSIFICATION,
                "early_stopping_rounds": self.early_stopping_rounds,
                "validation_fraction": self.validation_fraction,
                "split_strategy": self.split_strategy,
                "cv_folds": None,
            }
        )
        if self.random_seed is not None:
            params.setdefault("random_seed", self.random_seed)
        if self.base_classifier_class.__name__ == ModelName.PYTORCH_NEURAL_NETWORK.value:
            params["input_size"] = input_size
            params["output_size"] = classifier_output_size
        return params

    def _perform_feature_selection(self, x_train_val: pd.DataFrame, y_train_val: np.ndarray, iterations: int | None) -> None:
        """Performs feature selection on the base classifier."""
        temp_model = self._clone()
        temp_model.class_boundaries = self.class_boundaries
        logger.info("--- Training model on all the features in preparation for feature selection ---")
        temp_model._fit_single(x_train_val.values, y_train_val, forced_iterations=iterations, fit_mappers=False)
        feature_importance = self.calculate_feature_importances(model_instance=temp_model.base_classifier, x_background=x_train_val)

        if self.feature_selection_threshold is not None:
            self.selected_features_ = select_features_by_cumulative_importance(feature_importance, self.feature_selection_threshold)
            logger.info(f"--- Selected {len(self.selected_features_)} features ---")

    def _calculate_and_save_feature_importance(self, model_instance: "BaseModel", x_background: pd.DataFrame) -> None:
        """Calculates and saves feature importance."""
        feature_importance = self.calculate_feature_importances(model_instance=model_instance.base_classifier, x_background=x_background)
        self._save_feature_importance(feature_importance=feature_importance)

    def _evaluate_trial_performance(
        self, model_instance: "ClassifierRegressionModel", x: np.ndarray, y: np.ndarray, timestamps: np.ndarray | None = None, explicit_split: tuple | None = None
    ) -> tuple[float, int]:
        """Evaluates a trial's performance and returns the score and optimal iterations.

        With ``explicit_split = (x_train, y_train, x_val, y_val)`` (the one-pass protocol) the trial is one `_fit_single` on the
        training part, the classifier early-stopped on its held-out cross-entropy, and the score is the held-out fit term of the
        final prediction (`BaseModel._evaluate_trial_performance`).
        """
        # Set the class boundaries for the trial based on the trial's n_classes
        model_instance.class_boundaries = self.precomputed_boundaries_[model_instance.n_classes]
        if self.boundary_regularization_method != BoundaryRegularizationMethod.NONE:
            model_instance.class_value_ranges_ = self.precomputed_class_value_ranges_[model_instance.n_classes]
        if explicit_split is not None:
            return super()._evaluate_trial_performance(model_instance, x, y, timestamps=timestamps, explicit_split=explicit_split)

        if self.cv_folds:
            kf = self.get_kfolds(timestamps=timestamps)
            fold_iterations = []
            # Use a temporary dict to hold scores for each mapper type for this trial
            trial_mapper_scores = {m: [] for m in MapperType if m != MapperType.AUTO}

            for train_idx, val_idx in kf.split(x, y):
                x_train_fold, x_val_fold = x[train_idx], x[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]

                # We need to call the instance's _evaluate_fold_performance
                best_iter, mapper_scores, _ = model_instance._evaluate_fold_performance(x_train_fold, y_train_fold, x_val_fold, y_val_fold)
                fold_iterations.append(best_iter)
                for m, score in mapper_scores.items():
                    trial_mapper_scores[m].append(score)

            optimal_iterations = int(np.mean(fold_iterations)) if fold_iterations else 0

            # Determine the best mapper type for this trial based on average CV score
            avg_mapper_scores = {m: np.mean(scores) for m, scores in trial_mapper_scores.items() if scores}
            if not avg_mapper_scores:
                # Return a high score if no mappers were successfully evaluated
                return float("inf"), optimal_iterations

            # If the trial is for a specific mapper, use its score. If AUTO, find the best one.
            if model_instance.mapper_type == MapperType.AUTO:
                best_mapper_for_trial = min(avg_mapper_scores, key=avg_mapper_scores.get)
                final_score = avg_mapper_scores[best_mapper_for_trial]
            else:
                final_score = avg_mapper_scores.get(model_instance.mapper_type, float("inf"))

        else:  # Non-CV case for HPO
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

            # Use _evaluate_fold_performance for consistency even in the non-CV case
            optimal_iterations, mapper_scores, _ = model_instance._evaluate_fold_performance(x_train, y_train, x_val, y_val)

            if model_instance.mapper_type == MapperType.AUTO:
                final_score = min(mapper_scores.values()) if mapper_scores else float("inf")
            else:
                final_score = mapper_scores.get(model_instance.mapper_type, float("inf"))

        return final_score, optimal_iterations

    def _calibrate_and_score_mappers(
        self,
        mapper_type: MapperType,
        fitted_mappers: list[BaseMapper],
        temp_classifier: BaseModel,
        x_train_fold: np.ndarray,
        y_train_fold: np.ndarray,
        x_val_fold: np.ndarray,
        y_val_fold: np.ndarray,
        predictions: np.ndarray,
    ) -> float:
        """Calibrates the mappers if necessary and calculates the performance score."""
        # Temporarily set the classifier and mappers on the instance for score calculation
        original_classifier = self.base_classifier
        original_mappers = self.class_mappers
        original_uncertainty_method = self.uncertainty_method
        self.base_classifier = temp_classifier
        self.class_mappers = fitted_mappers

        # Force MSE for mapper evaluation
        self.uncertainty_method = UncertaintyMethod.CONSTANT

        self._calibrate_mappers_if_needed(
            mapper_type=mapper_type, mappers=fitted_mappers, classifier=temp_classifier, x_train=x_train_fold, y_train=y_train_fold, x_val=x_val_fold, y_val=y_val_fold
        )

        score = self._calculate_performance_score(y_val_fold, predictions, x_val_fold)

        # Restore the original state
        self.base_classifier = original_classifier
        self.class_mappers = original_mappers
        self.uncertainty_method = original_uncertainty_method

        return score

    def _evaluate_fold_performance(
        self, x_train_fold: np.ndarray, y_train_fold: np.ndarray, x_val_fold: np.ndarray, y_val_fold: np.ndarray
    ) -> tuple[int, dict[MapperType, float], dict[MapperType, dict[str, Any]]]:
        """Evaluates the performance of the classifier and mappers on a single fold.

        Returns:
            A tuple containing the best classifier iterations, a dictionary of mapper scores,
            and a dictionary of learned mapper parameters.
        """
        # 1. Discretize targets for the fold
        _, y_train_clf = create_bins(y_train_fold, unique_bin_edges=self.class_boundaries)
        _, y_val_clf = create_bins(y_val_fold, unique_bin_edges=self.class_boundaries)

        # 2. Train a temporary classifier to get optimal iterations and validation probabilities
        temp_classifier = self.base_classifier_class(**self._get_base_classifier_params(x_train_fold.shape[1]))
        best_iter, _ = temp_classifier._fit_single(x_train=x_train_fold, y_train=y_train_clf, x_val=x_val_fold, y_val=y_val_clf)
        val_probas = temp_classifier.predict_proba(x_val_fold)
        train_probas = temp_classifier.predict_proba(x_train_fold)

        # 3. Evaluate all candidate mappers
        mapper_scores = {}
        mapper_params = {}
        mapper_types_to_evaluate = [self.mapper_type] if self.mapper_type != MapperType.AUTO else [m for m in MapperType if m != MapperType.AUTO]

        for mapper_type in mapper_types_to_evaluate:
            if not self.auto_include_nn_mappers and mapper_type.is_nn:
                continue

            fitted_mappers, learned_params = self._fit_single_mapper(mapper_type, train_probas, y_train_fold, val_probas, y_val_fold)
            predictions = self._mapper_predict(val_probas, mapper_type, fitted_mappers)

            score = self._calibrate_and_score_mappers(
                mapper_type=mapper_type,
                fitted_mappers=fitted_mappers,
                temp_classifier=temp_classifier,
                x_train_fold=x_train_fold,
                y_train_fold=y_train_fold,
                x_val_fold=x_val_fold,
                y_val_fold=y_val_fold,
                predictions=predictions,
            )

            mapper_scores[mapper_type] = score
            mapper_params[mapper_type] = learned_params
            logger.info(f"  - Fold mapper score ({mapper_type.label}): {score:.4f}")

        return best_iter, mapper_scores, mapper_params

    def _fit_single(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        forced_iterations: int | None = None,
        fit_mappers: bool = True,
        forced_mapper_params: dict | None = None,
    ) -> tuple[int, list[float]]:
        """Fits a single model instance."""
        if not self.is_regression_model:
            raise ValueError("This model is designed for regression tasks.")

        y_flat = y_train.flatten() if y_train.ndim > 1 else y_train
        _, y_clf = create_bins(data=y_flat, unique_bin_edges=self.class_boundaries)
        self._calculate_middle_class_dist_params(y_flat, y_clf)

        y_val_clf = None
        if y_val is not None:
            _, y_val_clf = create_bins(data=y_val.flatten() if y_val.ndim > 1 else y_val, unique_bin_edges=self.class_boundaries)

        base_classifier_init_params = self._get_base_classifier_params(x_train.shape[1])
        if forced_iterations:
            base_classifier_init_params["early_stopping_rounds"] = None
            base_classifier_init_params["validation_fraction"] = None
            if self.base_classifier_class.__name__ in [ModelName.XGBOOST.value, ModelName.LIGHTGBM.value, ModelName.CATBOOST.value]:
                base_classifier_init_params["n_estimators"] = forced_iterations

        self.base_classifier = self.base_classifier_class(**base_classifier_init_params)
        best_iter, _ = self.base_classifier._fit_single(
            x_train=x_train,
            y_train=y_clf.astype(np.int64),
            x_val=x_val,
            y_val=y_val_clf.astype(np.int64) if y_val_clf is not None else None,
            forced_iterations=forced_iterations,
        )
        self.num_iterations_used = best_iter
        self.train_indices = self.base_classifier.train_indices
        self.val_indices = self.base_classifier.val_indices

        if fit_mappers:
            y_proba_train = self.base_classifier.predict_proba(x_train)
            y_proba_val = self.base_classifier.predict_proba(x_val) if x_val is not None else None

            self.class_mappers, _ = self._fit_single_mapper(self.mapper_type, y_proba_train, y_flat, val_probas=y_proba_val, val_y=y_val, forced_mapper_params=forced_mapper_params)

            self._calibrate_mappers_if_needed(
                mapper_type=self.mapper_type, mappers=self.class_mappers, classifier=self.base_classifier, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val
            )

            if self.uncertainty_method == UncertaintyMethod.CONSTANT:
                y_pred_train = self.predict(x_train, filter_data=False)
                self._train_residual_std = np.std(y_train - y_pred_train)

        return self.num_iterations_used, []

    def _evaluate_fold_and_update(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        fold_iterations: list[int],
        fold_mapper_scores: dict[MapperType, list[float]],
        fold_mapper_params: dict[MapperType, list[Any]],
    ) -> None:
        best_iter, mapper_scores, mapper_params = self._evaluate_fold_performance(x_train, y_train, x_val, y_val)
        fold_iterations.append(best_iter)
        for m, score in mapper_scores.items():
            fold_mapper_scores[m].append(score)
        for m, params in mapper_params.items():
            fold_mapper_params[m].append(params)

    def _update_params(self, best_params: dict[str, Any]) -> None:
        """Updates the model's parameters from a search trial.

        ``a__b`` sets key ``b`` of the dict parameter ``a`` (``base_classifier_params``, ``mapper_params``, ``nn_mapper_params``);
        other names are attributes (``n_classes``, ``mapper_type``).
        """
        flat = {}
        for key, value in best_params.items():
            outer, _, inner = key.partition(NESTED_PARAM_SEPARATOR)
            if inner:
                getattr(self, outer)[inner] = value
            else:
                flat[key] = MapperType[value] if key == "mapper_type" else value
        super()._update_params(flat)

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
        """Finds the optimal parameters for the model.

        Under ``use_explicit_split`` (a caller-given validation set, or ``refit_on_train_val=False``) the class boundaries come from
        the TRAINING targets only and `BaseModel._determine_optimal_parameters` runs the search on the caller's split (or the one
        early-stopped pass), so the validation rows are never binned, fitted or used for anything but stopping and scoring.
        """
        if use_explicit_split:
            self._precompute_class_boundaries(y_train)
            self.class_boundaries = self.precomputed_boundaries_[self.n_classes]
            if self.boundary_regularization_method != BoundaryRegularizationMethod.NONE:
                self.class_value_ranges_ = self.precomputed_class_value_ranges_[self.n_classes]
            super()._determine_optimal_parameters(
                x_train, y_train, x_train_val, y_train_val, x_val=x_val, y_val=y_val, timestamps_train_val=timestamps_train_val, use_explicit_split=True
            )
            return
        if self.optimize_hyperparameters:
            logger.info("--- Starting hyperparameter optimization for ClassifierRegressionModel ---")
            self._precompute_class_boundaries(y_train_val)
            best_params, best_iterations = self._find_best_hyperparameters(x=x_train_val.values, y=y_train_val, timestamps=timestamps_train_val)
            self._update_params(best_params)
            self.num_iterations_used = best_iterations
            self.class_boundaries = self.precomputed_boundaries_[self.n_classes]
            if self.boundary_regularization_method != BoundaryRegularizationMethod.NONE:
                self.class_value_ranges_ = self.precomputed_class_value_ranges_[self.n_classes]

            logger.info(f"--- Hyperparameter optimization finished. Best params: {self.get_params()} ---")
            logger.info(f"--- Optimal iterations from HPO: {self.num_iterations_used} ---")
        elif self.early_stopping_rounds is not None:
            y_flat = y_train_val.flatten() if y_train_val.ndim > 1 else y_train_val
            self.class_boundaries, y_binned = create_bins(data=y_flat, n_bins=self.n_classes, min_value=-np.inf, max_value=np.inf)
            self._calculate_middle_class_dist_params(y_flat, y_binned)
            if self.boundary_regularization_method != BoundaryRegularizationMethod.NONE:
                y_min, y_max = np.min(y_flat), np.max(y_flat)
                self.class_value_ranges_ = calculate_class_value_ranges(y_flat=y_flat, y_binned=y_binned, k=self.n_classes, y_min=y_min, y_max=y_max, device=self.device)

            # --- 3. Determine Optimal Iterations and Best Mapper ---
            fold_iterations = []
            fold_mapper_scores = {m: [] for m in MapperType if m != MapperType.AUTO}
            fold_mapper_params = {m: [] for m in MapperType if m != MapperType.AUTO}

            if self.cv_folds:
                logger.info(f"--- Evaluating performance across {self.cv_folds} CV folds ---")
                kf = self.get_kfolds(timestamps=timestamps_train_val)

                for i, (train_idx, val_idx) in enumerate(kf.split(x_train_val, y_train_val)):
                    logger.info(f"--- Training CV fold {i + 1}/{self.cv_folds} ---")
                    x_train_fold, y_train_fold = x_train_val.values[train_idx], y_train_val[train_idx]
                    x_val_fold, y_val_fold = x_train_val.values[val_idx], y_train_val[val_idx]
                    self._evaluate_fold_and_update(
                        x_train=x_train_fold,
                        y_train=y_train_fold,
                        x_val=x_val_fold,
                        y_val=y_val_fold,
                        fold_iterations=fold_iterations,
                        fold_mapper_scores=fold_mapper_scores,
                        fold_mapper_params=fold_mapper_params,
                    )
            else:  # Non-CV case
                logger.info("--- Evaluating performance on single train/validation split ---")
                self._evaluate_fold_and_update(
                    x_train=x_train.values,
                    y_train=y_train,
                    x_val=x_val.values,
                    y_val=y_val,
                    fold_iterations=fold_iterations,
                    fold_mapper_scores=fold_mapper_scores,
                    fold_mapper_params=fold_mapper_params,
                )

            # --- 4. Aggregate Results and Select Final Parameters ---
            self.num_iterations_used = int(np.mean(fold_iterations))
            logger.info(f"--- Aggregated optimal classifier iterations: {self.num_iterations_used} ---")

            if self.mapper_type == MapperType.AUTO:
                avg_mapper_scores = {m: np.mean(scores) for m, scores in fold_mapper_scores.items() if scores}
                self.mapper_type = min(avg_mapper_scores, key=avg_mapper_scores.get)
                logger.info(f"--- Auto-selected best mapper: {self.mapper_type.label} (Avg. MSE: {avg_mapper_scores[self.mapper_type]:.4f}) ---")

            # For NN mappers, average the learned epochs across folds
            if self.mapper_type.is_nn:
                all_params = fold_mapper_params[self.mapper_type]
                if all_params and "epochs_used" in all_params[0]:
                    avg_epochs = int(np.mean([p["epochs_used"] for p in all_params]))
                    self.optimal_mapper_params_ = {"epochs": avg_epochs}
                    logger.info(f"--- Aggregated optimal NN mapper epochs: {avg_epochs} ---")

    def _precompute_class_boundaries(self, y: np.ndarray) -> None:
        """Class boundaries (percentile bins of ``y``) for every k a fit may use: the searched ``n_classes`` range and this model's own k."""
        y = np.asarray(y).ravel()
        space = self._searched_space().get("n_classes") if self.optimize_hyperparameters else None
        k_values = sorted({self.n_classes, *(range(int(space["low"]), int(space["high"]) + 1) if space else ())})
        self.precomputed_boundaries_ = {}
        self.precomputed_class_value_ranges_ = {}
        y_min, y_max = np.min(y), np.max(y)
        for n in k_values:
            boundaries, y_binned = create_bins(data=y, n_bins=n, min_value=-np.inf, max_value=np.inf)
            self.precomputed_boundaries_[n] = boundaries
            self._calculate_middle_class_dist_params(y, y_binned, n_classes=n)
            if self.boundary_regularization_method != BoundaryRegularizationMethod.NONE:
                self.precomputed_class_value_ranges_[n] = calculate_class_value_ranges(y_flat=y, y_binned=y_binned, k=n, y_min=y_min, y_max=y_max, device=self.device)

    def _fit_final_model(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Retrain on the full training + validation set with the optimal number of iterations."""
        logger.info("--- Training final model with optimal parameters ---")
        self._fit_single(x_train, y_train, forced_iterations=self.num_iterations_used, fit_mappers=True, forced_mapper_params=self.optimal_mapper_params_)
        logger.info("--- Final model training finished ---")

    def _clone(self) -> "ClassifierRegressionModel":
        """Creates a new instance of the model with the same parameters."""
        return ClassifierRegressionModel(**self.get_params())

    def get_params(self) -> dict[str, Any]:
        """Gets parameters for this estimator."""
        params = super().get_params()
        params.update({key: getattr(self, key) for key in self._defaults})
        return params

    def _predict(self, x: np.ndarray, filter_data: bool = True) -> np.ndarray:
        """Makes predictions (expected regression output) using the composite model."""
        if self.base_classifier is None or not self.class_mappers:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        if filter_data:
            x = self._filter_predict_data(x)
        return self._mapper_predict(probas=self.base_classifier.predict_proba(x, filter_data=False), mapper_type_to_test=self.mapper_type, fitted_mappers=self.class_mappers)

    def predict_proba(self, x: np.ndarray, filter_data: bool = True) -> np.ndarray:
        """Returns the predicted probabilities from the internal base classifier.

        Args:
            x (np.ndarray): Feature matrix.
            filter_data (bool): If True, filter the input data using the feature selection mask.

        Returns:
            np.ndarray: Predicted probabilities from the internal classifier.
        """
        if self.base_classifier is None:
            raise RuntimeError("Base classifier has not been fitted yet.")
        if filter_data:
            x = self._filter_predict_data(x)
        return self.base_classifier.predict_proba(x, filter_data=False)

    def get_hyperparameter_search_space(self) -> dict[str, Any]:
        """Defines the hyperparameter search space for the ClassifierRegressionModel.

        ``n_classes``; the classifier's own space as ``base_classifier_params__<name>`` (read off a classifier built with this
        model's classifier settings, so e.g. its epochs are searched only when it does not stop early); the parameters of the
        chosen mapper only (every mapper's when ``mapper_type`` is AUTO, which also searches the mapper); and the boundary
        weight only when a boundary method is set. The boundary METHOD is a design choice, never searched.
        """
        sep = NESTED_PARAM_SEPARATOR
        space: dict[str, Any] = {"n_classes": {"type": "int", "low": 2, "high": 10}}
        if self.boundary_regularization_method != BoundaryRegularizationMethod.NONE:
            space["boundary_loss_weight"] = {"type": "float", "low": 0.1, "high": 10.0, "log": True}

        if self.base_classifier_class:
            classifier = self.base_classifier_class(**self._get_base_classifier_params(input_size=1))
            space.update({f"base_classifier_params{sep}{name}": params for name, params in classifier.get_hyperparameter_search_space().items()})

        auto = self.mapper_type == MapperType.AUTO
        searched_mappers = [m for m in MapperType if m != MapperType.AUTO and (self.auto_include_nn_mappers or not m.is_nn)] if auto else [self.mapper_type]
        if auto:
            space["mapper_type"] = {"type": "categorical", "choices": [m.name for m in searched_mappers]}
        if any(m in (MapperType.LOOKUP_MEAN, MapperType.LOOKUP_MEDIAN) for m in searched_mappers):
            space[f"mapper_params{sep}lookup_n_partitions"] = {"type": "int", "low": 5, "high": 50}
        if MapperType.SPLINE in searched_mappers:
            space[f"mapper_params{sep}spline_k"] = {"type": "int", "low": 1, "high": 3}
            space[f"mapper_params{sep}spline_s"] = {"type": "float", "low": 0.01, "high": 10.0, "log": True}
        if any(m.is_nn for m in searched_mappers):
            nn_space = {
                "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
                "hidden_layers": {"type": "int", "low": 0, "high": 2},
                "hidden_size": {"type": "int", "low": 16, "high": 64, "step": 16},
                "l1_lambda": {"type": "float", "low": 1e-8, "high": 1e-2, "log": True},
                "l2_lambda": {"type": "float", "low": 1e-8, "high": 1e-2, "log": True},
                "learn_regularization_lambdas": {"type": "categorical", "choices": [True, False]},
            }
            if self.early_stopping_rounds is None:  # epochs are searched only without early stopping
                nn_space["epochs"] = {"type": "int", "low": 5, "high": 100, "step": 20}
            space.update({f"nn_mapper_params{sep}{name}": params for name, params in nn_space.items()})

        if self.search_space_override:
            space.update(self.search_space_override)
        return space

    def get_classifier_predictions(self, x: np.ndarray | pd.DataFrame, y_true_original: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns the internal classifier's predicted classes, probabilities, and.

        the corresponding (discretized) true labels for this composite model.

        Args:
            x (np.ndarray): Feature matrix.
            y_true_original (np.ndarray): Original true labels (will be discretized internally).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - Predicted classes from the internal classifier.
                - Predicted probabilities from the internal classifier.
                - Discretized true labels corresponding to the internal classifier's task.
        """
        if self.base_classifier is None:
            raise RuntimeError("Base classifier has not been fitted yet.")

        # 1. Discretize the original true labels using the boundaries calculated during training.
        _, y_true_discretized = create_bins(data=y_true_original, unique_bin_edges=self.class_boundaries)

        # 2. Get predictions and probabilities from the internal classifier
        x_filtered = self._filter_predict_data(x)
        y_pred_internal = self.base_classifier.predict(x_filtered)
        y_proba_internal = self.base_classifier.predict_proba(x_filtered)

        return y_pred_internal, y_proba_internal, y_true_discretized

    def plot_probability_mappers(self, plot_path: str = "probability_mappers.png") -> None:
        """Plots the n functions (one for each class) calculated in the probability mapper.

        Each plot shows the mapping from class probability to the original regression value.

        Args:
            plot_path (str): The path to the file where the plot will be saved.
        """
        if not self.class_mappers:
            logger.warning("No class mappers found. Please fit the model first.")
            return

        if self.mapper_type.is_nn:
            nn_mapper = self.class_mappers[0]
            plot_nn_probability_mappers(
                mapper_model=nn_mapper.model,
                regression_strategy=nn_mapper.regression_strategy,
                n_classes=nn_mapper.n_classes,
                class_boundaries=self.class_boundaries,
                device=nn_mapper.device,
                plot_path=plot_path,
                model_name=self.name,
            )
        else:
            logger.info(f"\n--- Plotting Probability Mappers to {plot_path} ---")
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.figure(figsize=(12, 8))
            probas_range = np.linspace(0, 1, 100).reshape(-1, 1)
            for i, mapper in enumerate(self.class_mappers):
                if mapper is None:
                    continue
                mapped_values = mapper.predict(probas_range)
                lower_bound_str = f"{self.class_boundaries[i]:.2f}"
                upper_bound_str = f"{self.class_boundaries[i + 1]:.2f}"
                label_text = f"Class {i} (Range: {lower_bound_str}-{upper_bound_str})"
                plt.plot(probas_range, mapped_values, label=label_text)

            plt.title(f"Probability Mappers for {self.name}")
            plt.xlabel("Class Probability")
            plt.ylabel("Mapped Original Regression Value")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            logger.info("Probability mappers plot saved successfully.")

    def get_internal_model(self) -> Any:
        """Returns the raw underlying model object, if applicable.

        Useful for explainability libraries like SHAP.
        """
        return self.base_classifier.get_internal_model()

    def get_shap_explainer_info(self) -> dict[str, Any]:
        """Gets the SHAP explainer type and the model to be explained."""
        return self.base_classifier.get_shap_explainer_info()

    def get_num_parameters(self) -> int:
        """Returns the total number of trainable parameters in the model.

        Returns:
            int: The total number of parameters.
        """
        total_params = 0
        if self.base_classifier:
            total_params += self.base_classifier.get_num_parameters()

        if self.mapper_type.is_nn:
            if self.class_mappers and hasattr(self.class_mappers[0], "get_num_parameters"):
                total_params += self.class_mappers[0].get_num_parameters()
        elif self.mapper_type == MapperType.LINEAR:
            total_params += self.n_classes * 2  # n_classes * (slope + intercept)

        return total_params

    def evaluate(self, x: pd.DataFrame | np.ndarray, y: np.ndarray, partition_name: str, save_path: str, y_scaler: Any | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        """First, performs standard regression evaluation by calling the parent method.

        Then, adds specialized evaluation for the internal classifier.
        """
        # Call parent method to get standard regression evaluation and results
        y_pred_unscaled, y_std_unscaled = super().evaluate(x, y, partition_name, save_path, y_scaler)

        # Add specialized logic for internal classifier metrics
        try:
            # We need the scaled version of y to get correct discretized labels if a scaler was used
            y_scaled = y_scaler.transform(y.reshape(-1, 1)) if y_scaler is not None else y
            x_eval = x.values if isinstance(x, pd.DataFrame) else x

            y_pred_internal, y_proba_internal, y_true_discretized = self.get_classifier_predictions(x_eval, y_scaled)

            classification_metrics = Metrics(
                task_type=TaskType.CLASSIFICATION,
                model_name=f"{self.name}_internal_classifier",
                x_data=x_eval,
                y_true=y_true_discretized,
                y_pred=y_pred_internal,
                y_proba=y_proba_internal,
                partition_name=f"{partition_name}_classification",
            )
            classification_metrics.save_metrics(save_path)
        except NotImplementedError:
            pass

        return y_pred_unscaled, y_std_unscaled

    def _get_optimization_metric(self) -> Metric:
        """Gets the optimization metric, considering the mapper's potentially overridden uncertainty method."""
        # If self.uncertainty_method has been temporarily set to CONSTANT,
        # it means we are in the mapper evaluation phase and should force MSE.
        if self.uncertainty_method == UncertaintyMethod.CONSTANT:
            return Metric.MSE

        # Otherwise, proceed with the normal logic to determine the effective uncertainty method.
        effective_uncertainty_method = self.uncertainty_method
        if self.class_mappers:
            # For composite models, the mapper's uncertainty method is the source of truth
            active_mapper = self.class_mappers[0]
            if hasattr(active_mapper, "uncertainty_method"):
                effective_uncertainty_method = active_mapper.uncertainty_method

        if self.task_type == TaskType.REGRESSION:
            optimization_metric = Metric.NLL if effective_uncertainty_method == UncertaintyMethod.PROBABILISTIC else Metric.MSE
        else:
            optimization_metric = Metric.LOG_LOSS
        return optimization_metric

    def cross_validate(self, x: np.ndarray, y: np.ndarray, cv: int) -> dict[str, Any]:
        """Performs cross-validation."""
        self.cv_folds = cv
        self.fit(x, y)
        return {"test_score": self.cv_score_mean_}
