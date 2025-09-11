# ruff: noqa: ERA001
"""Composite model for regression using classification and probability mapping."""

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from automl_package.enums import MapperType, ModelName, RegressionStrategy, TaskType, UncertaintyMethod
from automl_package.logger import logger
from automl_package.models.base import BaseModel
from automl_package.models.mappers.base_mapper import BaseMapper
from automl_package.models.mappers.nn_mapper import NeuralNetworkMapper
from automl_package.models.probability_mapper import ClassProbabilityMapper
from automl_package.utils.data_handler import create_train_val_split
from automl_package.utils.feature_selection import select_features_by_cumulative_importance
from automl_package.utils.numerics import create_bins
from automl_package.utils.plotting import plot_nn_probability_mappers


class ClassifierRegressionModel(BaseModel):
    """A composite model that combines a classification model with a probability mapper.

    to perform regression. It first discretizes the target into N classes,
    trains a classifier on these classes, and then converts classification probabilities
    back to continuous regression output using various mapping strategies.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initializes the ClassifierRegressionModel."""
        self.base_classifier_class: type[BaseModel] | None = kwargs.pop("base_classifier_class", None)
        self.n_classes: int = kwargs.pop("n_classes", 3)
        self.auto_include_nn_mappers: bool = kwargs.pop("auto_include_nn_mappers", True)
        self.mapper_type: MapperType | str = kwargs.pop("mapper_type", MapperType.SPLINE)
        self.base_classifier_params: dict | None = kwargs.pop("base_classifier_params", None)
        self.mapper_params: dict | None = kwargs.pop("mapper_params", None)
        self.nn_mapper_params: dict | None = kwargs.pop("nn_mapper_params", None)
        self.regression_strategy: RegressionStrategy | None = kwargs.pop("regression_strategy", None)
        self.mapper_to_strategy_map = {
            MapperType.NN_SEPARATE_HEADS: RegressionStrategy.SEPARATE_HEADS,
            MapperType.NN_SINGLE_HEAD_N_OUTPUTS: RegressionStrategy.SINGLE_HEAD_N_OUTPUTS,
            MapperType.NN_SINGLE_HEAD_FINAL_OUTPUT: RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT,
        }

        super().__init__(**kwargs)
        assert self.is_regression_model

        if isinstance(self.mapper_type, str):
            self.mapper_type = MapperType[self.mapper_type]

        if self.mapper_type.is_nn and self.regression_strategy is None:
            self.regression_strategy = self.mapper_to_strategy_map.get(self.mapper_type)

        if self.base_classifier_class is None:
            raise ValueError("base_classifier_class must be provided.")

        self.base_classifier_params = self.base_classifier_params if self.base_classifier_params is not None else {}
        self.mapper_params = self.mapper_params if self.mapper_params is not None else {}
        self.nn_mapper_params = self.nn_mapper_params if self.nn_mapper_params is not None else {}

        self.base_classifier: BaseModel | None = None
        self.class_boundaries: np.ndarray | None = None
        self.class_mappers: list[ClassProbabilityMapper | NeuralNetworkMapper | None] = []
        self.is_composite_regression_model = True
        self.optimal_mapper_params_ = {}
        self.task_type = TaskType.REGRESSION
        self._train_residual_std = 0.0

        if self.n_classes < 2:
            raise ValueError("n_classes must be at least 2 for classification-regression strategy.")

    @property
    def name(self) -> str:
        """Returns the name of the model."""
        try:
            if self.base_classifier_class.__name__ in [ModelName.XGBOOST.value, ModelName.LIGHTGBM.value, ModelName.CATBOOST.value, ModelName.SKLEARN_LOGISTIC_REGRESSION.value]:
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

        if mapper_type.is_nn:
            nn_mapper = NeuralNetworkMapper(
                n_classes=self.n_classes,
                regression_strategy=self.mapper_to_strategy_map[mapper_type],
                mapper_params=current_nn_params,
                uncertainty_method=self.uncertainty_method,
                early_stopping_rounds=self.early_stopping_rounds,
                validation_fraction=self.validation_fraction,
            )

            # The NN mapper handles the train/val split internally if val_probas is provided
            train_indices = np.arange(probas.shape[0])
            val_indices = np.arange(val_probas.shape[0]) if val_probas is not None else None
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
        mapper_to_check = active_mapper.mapper if not mapper_type.is_nn else active_mapper

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
        active_uncertainty_method = active_mapper.mapper.uncertainty_method if not self.mapper_type.is_nn else active_mapper.uncertainty_method

        uncertainties = None
        if active_uncertainty_method == UncertaintyMethod.CONSTANT:
            uncertainties = np.full(x.shape[0], self._train_residual_std)

        elif active_uncertainty_method in [UncertaintyMethod.BINNED_RESIDUAL_STD, UncertaintyMethod.PROBABILISTIC]:
            if filter_data:
                x = self._filter_predict_data(x)
            probas = self.base_classifier.predict_proba(x, filter_data=False)

            if self.mapper_type.is_nn and self.regression_strategy == RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT:
                total_variance = self.class_mappers[0].predict_variance(probas)
            else:
                # Apply the Law of Total Variance
                if self.mapper_type.is_nn:
                    mapper = self.class_mappers[0]
                    mapper_means, mapper_variances = mapper.predict_mean_and_variance_per_class(probas)
                else:
                    mapper_means = np.zeros_like(probas)
                    mapper_variances = np.zeros_like(probas)
                    for c in range(self.n_classes):
                        if self.class_mappers[c] is not None:
                            class_probas = probas[:, c].reshape(-1, 1)
                            mapper_means[:, c] = self.class_mappers[c].predict(class_probas).flatten()
                            mapper_variances[:, c] = self.class_mappers[c].predict_variance_contribution(class_probas).flatten()

                # E[Y|X] = sum(P(C=i|X) * E[Y|X, C=i])
                total_mean = np.sum(probas * mapper_means, axis=1)
                # E[Var(Y|X)] = sum(P(C=i|X) * Var(Y|X, C=i))
                expected_variance = np.sum(probas * mapper_variances, axis=1)
                # Var(E[Y|X]) = sum(P(C=i|X) * (E[Y|X, C=i] - E[Y|X])^2)
                variance_of_expectation = np.sum(probas * np.square(mapper_means - total_mean[:, np.newaxis]), axis=1)
                # Total Variance = E[Var(Y|X)] + Var(E[Y|X])
                total_variance = expected_variance + variance_of_expectation
            uncertainties = np.sqrt(total_variance)

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

    def _evaluate_trial_performance(self, model_instance: "ClassifierRegressionModel", x: np.ndarray, y: np.ndarray, timestamps: np.ndarray | None = None) -> tuple[float, int]:
        """Evaluates a trial's performance and returns the score and optimal iterations."""
        # Set the class boundaries for the trial based on the trial's n_classes
        model_instance.class_boundaries = self.precomputed_boundaries_[model_instance.n_classes]

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
        self.base_classifier = temp_classifier
        self.class_mappers = fitted_mappers

        self._calibrate_mappers_if_needed(
            mapper_type=mapper_type, mappers=fitted_mappers, classifier=temp_classifier, x_train=x_train_fold, y_train=y_train_fold, x_val=x_val_fold, y_val=y_val_fold
        )

        score = self._calculate_performance_score(y_val_fold, predictions, x_val_fold)

        # Restore the original state
        self.base_classifier = original_classifier
        self.class_mappers = original_mappers

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

    def _evaluate_trial_performance(self, model_instance: "ClassifierRegressionModel", x: np.ndarray, y: np.ndarray, timestamps: np.ndarray | None = None) -> tuple[float, int]:
        """Evaluates a trial's performance and returns the score and optimal iterations."""
        # Set the class boundaries for the trial based on the trial's n_classes
        model_instance.class_boundaries = self.precomputed_boundaries_[model_instance.n_classes]

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
        self.base_classifier = temp_classifier
        self.class_mappers = fitted_mappers

        # Calibrate uncertainty if needed before scoring
        active_mapper = fitted_mappers[0]
        mapper_to_check = active_mapper.mapper if not mapper_type.is_nn else active_mapper
        if mapper_to_check and mapper_to_check.uncertainty_method == UncertaintyMethod.BINNED_RESIDUAL_STD:
            x_full_fold = np.vstack((x_train_fold, x_val_fold))
            y_full_fold = np.concatenate((y_train_fold.flatten(), y_val_fold.flatten()))
            probas_full_fold = temp_classifier.predict_proba(x_full_fold)

            if mapper_type.is_nn:
                fitted_mappers[0].calibrate_uncertainty(probas_full_fold, y_full_fold)
            else:
                for c, mapper in enumerate(fitted_mappers):
                    if mapper:
                        mapper.mapper.calibrate_uncertainty(probas_full_fold[:, c].reshape(-1, 1), y_full_fold)

        score = self._calculate_performance_score(y_val_fold, predictions, x_val_fold)

        # Restore the original state
        self.base_classifier = original_classifier
        self.class_mappers = original_mappers

        return score

    def _evaluate_fold_performance(
        self, x_train_fold: np.ndarray, y_train_fold: np.ndarray, x_val_fold: np.ndarray, y_val_fold: np.ndarray,
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
        """Updates the model's parameters from a successful hyperparameter optimization trial."""
        super()._update_params(best_params)
        base_classifier_params = {}
        mapper_params = {}
        nn_mapper_params = {}
        known_mapper_params = {"lookup_n_partitions", "spline_k", "spline_s"}

        for key, value in best_params.items():
            if key.startswith("base_classifier__"):
                param_name = key.split("__", 1)[1]
                base_classifier_params[param_name] = value
            elif key.startswith("nn_mapper_"):
                param_name = key.replace("nn_mapper_", "")
                nn_mapper_params[param_name] = value
            elif key in known_mapper_params:
                mapper_params[key] = value
            else:
                # Handle direct attributes like 'n_classes' and 'mapper_type'
                if key == "mapper_type":
                    self.mapper_type = MapperType[value]
                else:
                    setattr(self, key, value)

        self.base_classifier_params.update(base_classifier_params)
        self.mapper_params.update(mapper_params)
        self.nn_mapper_params.update(nn_mapper_params)

    def _determine_optimal_parameters(
        self,
        x_train: pd.DataFrame | None,
        y_train: np.ndarray,
        x_train_val: pd.DataFrame,
        y_train_val: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        timestamps_train_val: np.ndarray | None = None,
    ) -> None:
        """Finds the optimal parameters for the model."""
        if self.optimize_hyperparameters:
            logger.info("--- Starting hyperparameter optimization for ClassifierRegressionModel ---")
            # Pre-calculate class boundaries for all possible n_classes values to speed up trials
            search_space = self.get_hyperparameter_search_space()
            n_classes_space = search_space["n_classes"]
            self.precomputed_boundaries_ = {}
            for n in range(n_classes_space["low"], n_classes_space["high"] + 1):
                boundaries, _ = create_bins(data=y_train_val, n_bins=n, min_value=-np.inf, max_value=np.inf)
                self.precomputed_boundaries_[n] = boundaries

            best_params, best_iterations = self._find_best_hyperparameters(x=x_train_val.values, y=y_train_val, timestamps=timestamps_train_val)
            self._update_params(best_params)
            self.num_iterations_used = best_iterations
            self.class_boundaries = self.precomputed_boundaries_[self.n_classes]

            logger.info(f"--- Hyperparameter optimization finished. Best params: {self.get_params()} ---")
            logger.info(f"--- Optimal iterations from HPO: {self.num_iterations_used} ---")
        elif self.early_stopping_rounds is not None:
            self.class_boundaries, _ = create_bins(data=y_train_val, n_bins=self.n_classes, min_value=-np.inf, max_value=np.inf)
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
        params.update(
            {
                "n_classes": self.n_classes,
                "base_classifier_class": self.base_classifier_class,
                "base_classifier_params": self.base_classifier_params,
                "mapper_type": self.mapper_type,
                "mapper_params": self.mapper_params,
                "nn_mapper_params": self.nn_mapper_params,
                "auto_include_nn_mappers": self.auto_include_nn_mappers,
            }
        )
        return params

    def predict(self, x: np.ndarray, filter_data: bool = True) -> np.ndarray:
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
        """Defines the hyperparameter search space for the ClassifierRegressionModel."""
        space = {"n_classes": {"type": "int", "low": 2, "high": 10}}

        # Add base classifier hyperparameters
        if self.base_classifier_class:
            base_classifier_space = self.base_classifier_class().get_hyperparameter_search_space()
            for name, params in base_classifier_space.items():
                space[f"base_classifier__{name}"] = params

        # Add mapper type to search space only if it's set to AUTO
        if self.mapper_type == MapperType.AUTO:
            # Exclude AUTO itself from the choices
            mapper_choices = [m.name for m in MapperType if m != MapperType.AUTO and (self.auto_include_nn_mappers or not m.is_nn)]
            space["mapper_type"] = {"type": "categorical", "choices": mapper_choices}

        # Add parameters for all possible mappers that might be chosen
        # Optuna will only sample parameters for the mapper_type chosen in a given trial
        mapper_params = {
            # Parameters for lookup mappers
            "lookup_n_partitions": {"type": "int", "low": 5, "high": 50},
            # Parameters for spline mapper
            "spline_k": {"type": "int", "low": 1, "high": 3},
            "spline_s": {"type": "float", "low": 0.01, "high": 10.0, "log": True},
            # Parameters for NeuralNetworkMapper
            "nn_mapper_learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
            "nn_mapper_hidden_layers": {"type": "int", "low": 0, "high": 2},
            "nn_mapper_hidden_size": {"type": "int", "low": 16, "high": 64, "step": 16},
        }
        # Only add epochs to the search space if early stopping is disabled
        if self.early_stopping_rounds is None:
            mapper_params["nn_mapper_epochs"] = {"type": "int", "low": 5, "high": 100, "step": 20}

        space.update(mapper_params)

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
                upper_bound_str = f"{self.class_boundaries[i+1]:.2f}"
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

    def cross_validate(self, x: np.ndarray, y: np.ndarray, cv: int) -> dict[str, Any]:
        """Performs cross-validation."""
        self.cv_folds = cv
        self.fit(x, y)
        return {"test_score": self.cv_score_mean_}
