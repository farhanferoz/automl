"""Composite model for regression using classification and probability mapping."""

import os
from typing import Any, ClassVar
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import KFold

from automl_package.enums import MapperType, Metric, ModelName, RegressionStrategy, TaskType
from automl_package.logger import logger
from automl_package.models.base import BaseModel
from automl_package.models.mappers.base_mapper import BaseMapper
from automl_package.models.mappers.nn_mapper import NeuralNetworkMapper
from automl_package.models.probability_mapper import ClassProbabilityMapper
from automl_package.utils.numerics import create_bins
from automl_package.utils.plotting import plot_nn_probability_mappers


class ClassifierRegressionModel(BaseModel):
    """A composite model that combines a classification model with a probability mapper.

    to perform regression. It first discretizes the target into N classes,
    trains a classifier on these classes, and then converts classification probabilities
    back to continuous regression output using various mapping strategies.
    """

    _defaults: ClassVar[dict[str, Any]] = {
        "n_classes": 3,
        "base_classifier_class": None,
        "base_classifier_params": None,
        "mapper_type": MapperType.SPLINE,
        "mapper_params": None,
        "nn_mapper_params": None,
        "auto_include_nn_mappers": True,
    }

    def __init__(self, **kwargs: Any) -> None:
        """Initializes the ClassifierRegressionModel."""
        self.base_classifier_class: type[BaseModel] | None = None
        self.n_classes: int = 0
        self.auto_include_nn_mappers: bool = True
        self.mapper_type: MapperType = MapperType.SPLINE
        self.base_classifier_params: dict | None = None
        self.mapper_params: dict | None = None
        self.nn_mapper_params: dict | None = None

        for key, value in self._defaults.items():
            kwargs.setdefault(key, value)

        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.base_classifier_class is None:
            raise ValueError("base_classifier_class must be provided.")

        self.base_classifier_params = self.base_classifier_params if self.base_classifier_params is not None else {}
        self.mapper_params = self.mapper_params if self.mapper_params is not None else {}
        self.nn_mapper_params = self.nn_mapper_params if self.nn_mapper_params is not None else {}

        self.base_classifier: BaseModel | None = None
        self.class_boundaries: np.ndarray | None = None
        self.class_mappers: list[ClassProbabilityMapper | NeuralNetworkMapper | None] = []
        self.is_regression_model = True
        self.is_composite_regression_model = True
        self.optimal_mapper_params_ = {}
        self.task_type = TaskType.REGRESSION

        if self.n_classes < 2:
            raise ValueError("n_classes must be at least 2 for classification-regression strategy.")

    def _get_optimization_metric(self) -> Metric:
        """Gets the optimization metric for the model."""
        return Metric.RMSE

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
        mapper_type: MapperType,
        probas: np.ndarray,
        y: np.ndarray,
        val_probas: np.ndarray | None = None,
        val_y: np.ndarray | None = None,
        forced_mapper_params: dict | None = None,
    ) -> tuple[list, dict]:
        """Fits a single mapper type and returns the fitted mapper and any learned parameters."""
        learned_params = {}
        current_nn_params = self.nn_mapper_params.copy()
        if forced_mapper_params:
            current_nn_params.update(forced_mapper_params)

        if mapper_type.is_nn:
            mapper_to_strategy_map = {
                MapperType.NN_SEPARATE_HEADS: RegressionStrategy.SEPARATE_HEADS,
                MapperType.NN_SINGLE_HEAD_N_OUTPUTS: RegressionStrategy.SINGLE_HEAD_N_OUTPUTS,
                MapperType.NN_SINGLE_HEAD_FINAL_OUTPUT: RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT,
            }
            strategy = mapper_to_strategy_map[mapper_type]
            nn_mapper = NeuralNetworkMapper(
                n_classes=self.n_classes,
                regression_strategy=strategy,
                mapper_params=current_nn_params,
                early_stopping_rounds=self.early_stopping_rounds,
                validation_fraction=self.validation_fraction,
            )

            # The NN mapper handles the train/val split internally if val_probas is provided
            train_indices = np.arange(probas.shape[0])
            val_indices = np.arange(val_probas.shape[0]) if val_probas is not None else None
            combined_probas = np.vstack((probas, val_probas)) if val_probas is not None else probas
            combined_y = np.concatenate((y, val_y)) if val_y is not None else y

            learned_params = nn_mapper.fit(combined_probas, combined_y, train_indices, val_indices)
            fitted_mappers = [nn_mapper]
        else:
            # Non-NN mappers are simpler and fit only on the training data provided.
            fitted_mappers = []
            for c in range(self.n_classes):
                mapper = ClassProbabilityMapper(mapper_type, **self.mapper_params)
                mapper.fit(probas[:, c].reshape(-1, 1), y)
                fitted_mappers.append(mapper)
        return fitted_mappers, learned_params

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
        self.class_boundaries, y_clf = create_bins(data=y_flat, n_bins=self.n_classes, min_value=-np.inf, max_value=np.inf)

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
            x_train, y_clf.astype(np.int64), x_val, y_val_clf.astype(np.int64) if y_val_clf is not None else None, forced_iterations=forced_iterations,
        )
        self.num_iterations_used = best_iter
        self.train_indices = self.base_classifier.train_indices
        self.val_indices = self.base_classifier.val_indices

        if fit_mappers:
            y_proba_all = self.base_classifier.predict_proba(x_train)
            if y_proba_all.shape[1] != self.n_classes:
                if self.n_classes == 2 and y_proba_all.ndim == 1:
                    y_proba_all = np.vstack((1 - y_proba_all, y_proba_all)).T
                elif self.n_classes == 2 and y_proba_all.shape[1] == 1:
                    y_proba_all = np.hstack((1 - y_proba_all, y_proba_all))
                else:
                    raise ValueError(f"Classifier predict_proba output shape {y_proba_all.shape} does not match n_classes {self.n_classes}.")

            self.class_mappers, _ = self._fit_single_mapper(self.mapper_type, y_proba_all, y_flat, forced_mapper_params=forced_mapper_params)

        return self.num_iterations_used, []

    def _evaluate_trial(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Evaluates a trial for hyperparameter optimization."""
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def _find_optimal_iterations_with_cv(self, x: np.ndarray, y: np.ndarray, timestamps: np.ndarray | None = None) -> tuple[int, np.ndarray, np.ndarray]:
        """Uses cross-validation to find the optimal number of iterations for the base classifier."""
        y_flat = y.flatten() if y.ndim > 1 else y
        self.class_boundaries, y_clf = create_bins(data=y_flat, n_bins=self.n_classes, min_value=-np.inf, max_value=np.inf)
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=getattr(self, "random_seed", None))
        oof_probas = np.zeros((x.shape[0], self.n_classes))
        oof_true_y = np.zeros(x.shape[0])
        fold_results = []

        for i, (train_idx, val_idx) in enumerate(kf.split(x)):
            logger.info(f"--- Training CV fold {i+1}/{self.cv_folds} ---")
            x_train_fold, x_val_fold = x[train_idx], x[val_idx]
            y_clf_train_fold, y_clf_val_fold = y_clf[train_idx], y_clf[val_idx]
            classifier_instance = self.base_classifier_class(**self._get_base_classifier_params(x.shape[1]))
            best_iter, loss_history = classifier_instance._fit_single(x_train_fold, y_clf_train_fold, x_val_fold, y_clf_val_fold)
            probas = classifier_instance.predict_proba(x_val_fold)
            oof_probas[val_idx] = probas
            oof_true_y[val_idx] = y[val_idx]
            score = log_loss(y_clf_val_fold, probas)
            fold_results.append({"best_iter": best_iter, "loss_history": loss_history, "score": score})

        max_len = max(len(res["loss_history"]) for res in fold_results if res["loss_history"])
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

        self.cv_score_mean_ = np.mean([res["score"] for res in fold_results])
        self.cv_score_std_ = np.std([res["score"] for res in fold_results])
        return optimal_iterations, oof_probas, oof_true_y

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

    def _find_best_mapper(self, probas_train: np.ndarray, y_train: np.ndarray, probas_val: np.ndarray, y_val: np.ndarray) -> tuple[MapperType, dict]:
        """Finds the best mapper type by fitting on training data and evaluating on validation data."""
        best_mapper_type = None
        min_loss = float("inf")
        best_params = {}

        for mapper_type_to_test in MapperType:
            if mapper_type_to_test == MapperType.AUTO or ((not self.auto_include_nn_mappers) and mapper_type_to_test.is_nn):
                continue

            logger.info(f"Auto-testing mapper type: {mapper_type_to_test.label}")
            fitted_mappers, learned_params = self._fit_single_mapper(mapper_type_to_test, probas_train, y_train, probas_val, y_val)

            predictions = self._mapper_predict(probas=probas_val, mapper_type_to_test=mapper_type_to_test, fitted_mappers=fitted_mappers)

            loss = mean_squared_error(y_val, predictions)
            logger.info(f"  - Mapper Loss on validation (MSE): {loss}")

            if loss < min_loss:
                min_loss = loss
                best_mapper_type = mapper_type_to_test
                best_params = learned_params

        return best_mapper_type, best_params

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

    def fit(self, x: np.ndarray, y: np.ndarray, timestamps: np.ndarray | None = None) -> None:
        """Custom fit method for ClassifierRegressionModel to decouple classifier tuning from mapper selection."""
        logger.info(f"--- Starting training for {self.name} ---")
        x_train, y_train, x_val, y_val, x_test, y_test, _, _, _, x_train_val, y_train_val, timestamps_train_val = self._prepare_data_partitions(x, y, timestamps)

        if self.optimize_hyperparameters:
            if not self.cv_folds:
                raise ValueError("cv_folds must be set to perform hyperparameter optimization.")
            logger.info("--- Starting hyperparameter optimization ---")
            self.best_params_ = self._find_best_hyperparameters(x_train_val, y_train_val)
            self.params.update(self.best_params_)
            logger.info(f"--- Hyperparameter optimization finished. Best params: {self.best_params_} ---")

        if self.cv_folds:
            logger.info("--- Finding optimal iterations with cross-validation ---")
            optimal_iterations, oof_probas, oof_true_y = self._find_optimal_iterations_with_cv(x_train_val, y_train_val, timestamps_train_val)
            self.optimal_iterations_ = optimal_iterations
            logger.info(f"--- Optimal iterations found: {self.optimal_iterations_} ---")

            if self.mapper_type == MapperType.AUTO:
                logger.info("--- Finding best mapper type ---")
                best_mapper_type, best_mapper_params = self._find_best_mapper(oof_probas, oof_true_y, oof_probas, oof_true_y)
                self.mapper_type = best_mapper_type
                self.optimal_mapper_params_ = best_mapper_params
                logger.info(f"Auto-selected mapper type '{self.mapper_type.label}' with params {self.optimal_mapper_params_}")

            logger.info("--- Training final model with optimal parameters ---")
            self._fit_single(x_train_val, y_train_val, forced_iterations=self.optimal_iterations_, fit_mappers=True, forced_mapper_params=self.optimal_mapper_params_)
            logger.info("--- Final model training finished ---")

        else:  # No CV
            # 1. Find optimal classifier iterations using a validation set
            logger.info("--- Finding optimal classifier iterations with train/validation split ---")
            self.num_iterations_used, _ = self._fit_single(x_train, y_train, x_val, y_val, fit_mappers=False)
            self.optimal_iterations_ = self.num_iterations_used
            logger.info(f"--- Optimal classifier iterations found: {self.optimal_iterations_} ---")

            # 2. Find best mapper type and its optimal params using the validation set
            y_proba_train = self.base_classifier.predict_proba(x_train)
            y_proba_val = self.base_classifier.predict_proba(x_val)

            if self.mapper_type == MapperType.AUTO:
                logger.info("--- Finding best mapper type ---")
                best_mapper_type, self.optimal_mapper_params_ = self._find_best_mapper(y_proba_train, y_train, y_proba_val, y_val)
                self.mapper_type = best_mapper_type
                logger.info(f"Auto-selected mapper type '{self.mapper_type.label}' with params {self.optimal_mapper_params_}")
            elif self.mapper_type.is_nn:
                logger.info(f"Finding optimal epochs for specified NN mapper: {self.mapper_type.label}")
                _, self.optimal_mapper_params_ = self._fit_single_mapper(self.mapper_type, y_proba_train, y_train, y_proba_val, y_val)
                logger.info(f"Found optimal mapper params: {self.optimal_mapper_params_}")

            # 3. Train the final model on all data with optimal parameters
            logger.info("--- Training final model with optimal parameters ---")
            self._fit_single(x_train_val, y_train_val, forced_iterations=self.optimal_iterations_, fit_mappers=True, forced_mapper_params=self.optimal_mapper_params_)
            logger.info("--- Final model training finished ---")

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

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Makes predictions (expected regression output) using the composite model."""
        if self.base_classifier is None or not self.class_mappers:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        return self._mapper_predict(probas=self.base_classifier.predict_proba(x), mapper_type_to_test=self.mapper_type, fitted_mappers=self.class_mappers)

    def predict_uncertainty(self, x: np.ndarray) -> np.ndarray:
        """Estimates uncertainty for regression predictions using this composite model.

        It sums the variance contributions from each class's mapper, weighted by their probabilities.

        Args:
            x (np.ndarray): Features for uncertainty estimation.

        Returns:
            np.ndarray: Uncertainty estimates (standard deviation) for each prediction.
        """
        if not self.is_regression_model:
            raise ValueError("predict_uncertainty is only available for regression models.")
        if self.base_classifier is None or not self.class_mappers:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        y_proba_all = self.base_classifier.predict_proba(x)

        # Initialize an array to store the sum of variance contributions (P_i^2 * Var_mapper_i)
        total_variance = np.zeros(x.shape[0])

        for c in range(self.n_classes):
            if self.class_mappers[c] is None:
                logger.warning(f"Mapper for class {c} is None. Skipping its variance contribution.")
                continue

            proba_c = y_proba_all[:, c]  # 1D array of probabilities for class c

            # Get variance from the mapper for this class's probabilities
            mapper_variances = self.class_mappers[c].predict_variance_contribution(proba_c)

            # Add this class's weighted variance contribution
            total_variance += (proba_c**2) * mapper_variances

        return np.sqrt(total_variance)  # Standard deviation is sqrt of total variance

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Returns the predicted probabilities from the internal base classifier.

        Args:
            x (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted probabilities from the internal classifier.
        """
        if self.base_classifier is None:
            raise RuntimeError("Base classifier has not been fitted yet.")
        return self.base_classifier.predict_proba(x)

    def get_hyperparameter_search_space(self) -> dict[str, Any]:
        """Defines the hyperparameter search space for the ClassifierRegressionModel.

        Returns:
            Dict[str, Any]: A dictionary defining the hyperparameter search space.
        """
        # This model's search space defines its own parameters and the choice of base classifier.
        # The base_classifier_params themselves will be dynamically sampled within AutoML._sample_params_for_trial
        space = {
            "n_classes": {"type": "int", "low": 2, "high": 10},  # Number of classes for discretization
            # Base classifier chosen from a categorical list of model names
            "base_classifier_name": {
                "type": "categorical",
                "choices": [
                    ModelName.PYTORCH_NEURAL_NETWORK.value,
                    ModelName.XGBOOST.value,
                    ModelName.LIGHTGBM.value,
                    ModelName.SKLEARN_LOGISTIC_REGRESSION.value,
                    ModelName.CATBOOST.value,
                ],
            },
            "mapper_type": {"type": "categorical", "choices": [e.label for e in MapperType if e != MapperType.AUTO]},
            # Parameters for lookup mappers (will be sampled conditionally in AutoML)
            "n_partitions_min_lookup": {"type": "int", "low": 5, "high": 10},
            "n_partitions_max_lookup": {"type": "int", "low": 10, "high": 50},
            # Parameters for spline mapper (will be sampled conditionally in AutoML)
            "spline_k": {"type": "int", "low": 1, "high": 3},
            "spline_s": {"type": "float", "low": 0.01, "high": 10.0, "log": True},
            # Parameters for NeuralNetworkMapper (will be sampled conditionally in AutoML)
            "nn_mapper_params__epochs": {"type": "int", "low": 50, "high": 200},
            "nn_mapper_params__learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
            "nn_mapper_params__regression_head_params__hidden_layers": {"type": "int", "low": 0, "high": 2},
            "nn_mapper_params__regression_head_params__hidden_size": {"type": "int", "low": 16, "high": 64, "step": 16},
            "nn_mapper_params__regression_head_params__use_batch_norm": {"type": "categorical", "choices": [True, False]},
        }
        if self.search_space_override:
            space.update(self.search_space_override)
        return space

    def get_classifier_predictions(self, x: np.ndarray, y_true_original: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        y_pred_internal = self.base_classifier.predict(x)
        y_proba_internal = self.base_classifier.predict_proba(x)

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
