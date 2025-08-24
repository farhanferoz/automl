"""Composite model for regression using classification and probability mapping."""

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from torch.nn import ReLU, Tanh

from automl_package.enums import (
    MapperType,
    ModelName,
    RegressionStrategy,
    TaskType,
    UncertaintyMethod,
)
from automl_package.logger import logger
from automl_package.models.base import BaseModel
from automl_package.models.mappers.nn_mapper import NeuralNetworkMapper
from automl_package.models.probability_mapper import ClassProbabilityMapper
from automl_package.utils.metrics import Metrics
from automl_package.utils.numerics import create_bins
from automl_package.utils.plotting import plot_nn_probability_mappers


class ClassifierRegressionModel(BaseModel):
    """A composite model that combines a classification model with a probability mapper.

    to perform regression. It first discretizes the target into N classes,
    trains a classifier on these classes, and then converts classification probabilities
    back to continuous regression output using various mapping strategies.
    """

    def __init__(
        self,
        n_classes: int = 3,
        base_classifier_class: type[BaseModel] | None = None,  # The class of the internal classifier (e.g., PyTorchNeuralNetwork, XGBoostModel)
        base_classifier_params: dict[str, Any] | None = None,  # Parameters for the base classifier
        mapper_type: MapperType = MapperType.SPLINE,  # Use enum for mapper type
        mapper_params: dict[str, Any] | None = None,  # Parameters for the ClassProbabilityMapper
        nn_mapper_params: dict[str, Any] | None = None,  # Parameters for the NeuralNetworkMapper
        auto_include_nn_mappers: bool = True,  # Whether to include NN mappers in AUTO mode
        **kwargs: Any,
    ) -> None:
        """Initializes the ClassifierRegressionModel.

        Args:
            n_classes (int): The number of classes to discretize the target into.
            base_classifier_class (type[BaseModel]): The class of the internal classifier.
            base_classifier_params (dict[str, Any]): Parameters for the base classifier.
            mapper_type (MapperType): The type of probability mapping strategy.
            mapper_params (dict[str, Any] | None): Parameters for the ClassProbabilityMapper.
            nn_mapper_params (dict[str, Any] | None): Parameters for the NeuralNetworkMapper.
            auto_include_nn_mappers (bool): Whether to include NN mappers in AUTO mode.
            **kwargs: Additional keyword arguments for the BaseModel.
        """
        super().__init__(**kwargs)
        if base_classifier_class is None:
            raise ValueError("base_classifier_class must be provided.")
        self.n_classes = n_classes
        self.base_classifier_class = base_classifier_class
        self.base_classifier_params = base_classifier_params if base_classifier_params is not None else {}
        self.mapper_type = mapper_type
        self.mapper_params = mapper_params if mapper_params is not None else {}
        self.nn_mapper_params = nn_mapper_params if nn_mapper_params is not None else {}
        self.auto_include_nn_mappers = auto_include_nn_mappers

        self.base_classifier: BaseModel | None = None  # The instantiated base classifier model
        self.class_boundaries: np.ndarray | None = None  # Stores percentile values for discretization
        self.class_mappers: list[ClassProbabilityMapper | NeuralNetworkMapper | None] = []  # Stores a mapper for each class or a single NN mapper
        self.is_regression_model = True  # This composite model is for regression
        self.is_composite_regression_model = True  # Flag for AutoML to identify composite models

        if self.n_classes < 2:
            raise ValueError("n_classes must be at least 2 for classification-regression strategy.")

    @property
    def name(self) -> str:
        """Returns the name of the model."""
        # Include base model name and mapper type for unique identification
        # Dynamically get the name of the base classifier
        try:
            # Instantiate a dummy base_classifier_class to get its name property
            # Pass a default is_classification=True for models that require it in init
            if self.base_classifier_class.__name__ in [ModelName.XGBOOST.value, ModelName.LIGHTGBM.value, ModelName.CATBOOST.value, ModelName.SKLEARN_LOGISTIC_REGRESSION.value]:
                base_name = self.base_classifier_class(is_classification=True).name
            elif self.base_classifier_class.__name__ == ModelName.PYTORCH_NEURAL_NETWORK.value:
                # PyTorchNeuralNetwork needs input_size, output_size, task_type
                base_name = self.base_classifier_class(input_size=10, output_size=1, task_type=TaskType.CLASSIFICATION).name
            else:
                base_name = self.base_classifier_class().name
        except Exception as e:
            logger.warning(f"Could not instantiate base_classifier_class to get name: {e}. Using 'UnknownBase'.")
            base_name = "UnknownBase"

        return f"{base_name}_to_Reg_{self.mapper_type.label}"  # Dynamically append base name and mapper type

    def _fit_mappers(self, y_proba_all: np.ndarray, y_flat: np.ndarray, mapper_type: MapperType) -> list[ClassProbabilityMapper]:
        class_mappers = [None] * self.n_classes
        for c in range(self.n_classes):
            # Define the range for the current class in terms of original y values
            # (used to select original y values belonging to this class)
            mapper = ClassProbabilityMapper(mapper_type, **self.mapper_params)
            mapper.fit(y_proba_all[:, c].reshape(-1, 1), y_flat)
            class_mappers[c] = mapper
        return class_mappers

    def _fit_and_predict_mapper(
        self,
        mapper_type: MapperType,
        y_proba_all: np.ndarray,
        y_flat: np.ndarray,
        train_indices: np.ndarray | None,
        val_indices: np.ndarray | None,
    ) -> tuple[list, np.ndarray]:
        """Fits a given mapper type and returns the fitted mappers and predictions."""
        fitted_mappers = []
        predictions = np.zeros(y_proba_all.shape[0])

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
                mapper_params=self.nn_mapper_params,
                early_stopping_rounds=self.early_stopping_rounds,
                validation_fraction=self.validation_fraction,
            )
            nn_mapper.fit(y_proba_all, y_flat, train_indices, val_indices)
            predictions = nn_mapper.predict(y_proba_all)
            fitted_mappers = [nn_mapper]
        else:
            fitted_mappers = self._fit_mappers(y_proba_all=y_proba_all, y_flat=y_flat, mapper_type=mapper_type)
            for c in range(self.n_classes):
                proba_for_current_class = y_proba_all[:, c].reshape(-1, 1)
                expected_y_from_mapper = fitted_mappers[c].predict(proba_for_current_class)
                predictions += proba_for_current_class.flatten() * expected_y_from_mapper

        return fitted_mappers, predictions

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
        if not self.is_regression_model:
            raise ValueError("This model is designed for regression tasks.")

        y_flat = y.flatten() if y.ndim > 1 else y
        self.class_boundaries, y_clf = create_bins(data=y_flat, n_bins=self.n_classes, min_value=-np.inf, max_value=np.inf)

        y_val_clf = None
        if y_val is not None:
            _, y_val_clf = create_bins(data=y_val.flatten() if y_val.ndim > 1 else y_val, unique_bin_edges=self.class_boundaries)

        classifier_output_size = 1 if self.n_classes == 2 else self.n_classes
        base_classifier_init_params = self.base_classifier_params.copy()
        base_classifier_init_params["task_type"] = TaskType.CLASSIFICATION
        base_classifier_init_params["early_stopping_rounds"] = self.early_stopping_rounds
        base_classifier_init_params["validation_fraction"] = self.validation_fraction

        if self.base_classifier_class.__name__ == ModelName.PYTORCH_NEURAL_NETWORK.value:
            base_classifier_init_params["input_size"] = x.shape[1]
            base_classifier_init_params["output_size"] = classifier_output_size
            if "activation" in base_classifier_init_params and isinstance(base_classifier_init_params["activation"], str):
                base_classifier_init_params["activation"] = ReLU if base_classifier_init_params["activation"] == "ReLU" else Tanh
            if base_classifier_init_params.get("uncertainty_method") == UncertaintyMethod.MC_DROPOUT.value:
                base_classifier_init_params.setdefault("dropout_rate", 0.1)
                base_classifier_init_params.setdefault("n_mc_dropout_samples", 100)
            else:
                base_classifier_init_params["dropout_rate"] = 0.0
        elif self.base_classifier_class.__name__ == ModelName.XGBOOST.value:
            base_classifier_init_params.setdefault("objective", "binary:logistic" if self.n_classes == 2 else "multi:softmax")
            base_classifier_init_params.setdefault("eval_metric", "logloss" if self.n_classes == 2 else "mlogloss")
        elif self.base_classifier_class.__name__ == ModelName.LIGHTGBM.value:
            base_classifier_init_params.setdefault("objective", "binary" if self.n_classes == 2 else "multiclass")
            base_classifier_init_params.setdefault("metric", "binary_logloss" if self.n_classes == 2 else "multi_logloss")
        elif self.base_classifier_class.__name__ == ModelName.CATBOOST.value:
            base_classifier_init_params.setdefault("task_type", TaskType.CLASSIFICATION)
            base_classifier_init_params.setdefault("loss_function", "Logloss" if self.n_classes == 2 else "MultiClass")
            base_classifier_init_params.setdefault("eval_metric", "Logloss" if self.n_classes == 2 else "MultiClass")

        self.base_classifier = self.base_classifier_class(**base_classifier_init_params)
        self.base_classifier._fit_single(x, y_clf.astype(np.int64), x_val, y_val_clf.astype(np.int64) if y_val_clf is not None else None, forced_iterations=forced_iterations)
        self.num_iterations_used = self.base_classifier.num_iterations_used
        self.train_indices = self.base_classifier.train_indices
        self.val_indices = self.base_classifier.val_indices

        y_proba_all = self.base_classifier.predict_proba(x)

        if y_proba_all.shape[1] != self.n_classes:
            if self.n_classes == 2 and y_proba_all.ndim == 1:
                y_proba_all = np.vstack((1 - y_proba_all, y_proba_all)).T
            elif self.n_classes == 2 and y_proba_all.shape[1] == 1:
                y_proba_all = np.hstack((1 - y_proba_all, y_proba_all))
            else:
                raise ValueError(f"Classifier predict_proba output shape {y_proba_all.shape} does not match n_classes {self.n_classes}.")

        if self.mapper_type == MapperType.AUTO:
            best_mapper_type = None
            min_loss = float("inf")

            for mapper_type_to_test in MapperType:
                if mapper_type_to_test in [MapperType.AUTO, MapperType.LINEAR] or ((not self.auto_include_nn_mappers) and mapper_type_to_test.is_nn):
                    continue
                logger.info(f"Auto-testing mapper type: {mapper_type_to_test.label}")
                temp_mappers, final_predictions = self._fit_and_predict_mapper(mapper_type_to_test, y_proba_all, y_flat, self.train_indices, self.val_indices)
                loss = np.mean((y_flat - final_predictions) ** 2)
                logger.info(f"  - Mapper Loss (MSE): {loss}")

                if loss < min_loss:
                    min_loss = loss
                    best_mapper_type = mapper_type_to_test
                    self.class_mappers = temp_mappers

            self.mapper_type = best_mapper_type
            logger.info(f"Auto-selected mapper type: {self.mapper_type.label}")
        else:
            self.class_mappers, _ = self._fit_and_predict_mapper(self.mapper_type, y_proba_all, y_flat, self.train_indices, self.val_indices)
        return 0, []

    def _evaluate_trial(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Evaluates a trial for hyperparameter optimization."""
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def _clone(self) -> "ClassifierRegressionModel":
        """Creates a new instance of the model with the same parameters."""
        return ClassifierRegressionModel(**self.get_params())

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Makes predictions (expected regression output) using the composite model.

        Args:
            x (np.ndarray): Features for prediction.

        Returns:
            np.ndarray: Predicted regression values.
        """
        if self.base_classifier is None or not self.class_mappers:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        # Get classification probabilities from the base classifier for all classes
        y_proba_all = self.base_classifier.predict_proba(x)

        if self.mapper_type.is_nn:
            return self.class_mappers[0].predict(y_proba_all)

        # Initialize final predictions array
        final_predictions = np.zeros(x.shape[0])

        # Sum the product of classification probability and expected regression output for each class
        for c in range(self.n_classes):
            if self.class_mappers[c] is None:
                logger.warning(f"Mapper for class {c} is None. Skipping its contribution.")
                continue

            # Probability of current class `c` for each sample
            proba_for_current_class = y_proba_all[:, c].reshape(-1, 1)

            # Expected regression value mapped from this probability for class `c`
            expected_y_from_mapper = self.class_mappers[c].predict(proba_for_current_class)

            # Add this class's weighted contribution to the final prediction
            final_predictions += proba_for_current_class.flatten() * expected_y_from_mapper

        return final_predictions

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
        return self.model

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
        metrics_calculator = Metrics(task_type="regression", model_name=self.name, x_data=x, y_true=y, y_pred=y_pred)
        metrics_calculator.save_metrics(save_path)

        y_pred_internal_clf, y_proba_internal_clf, y_true_discretized = self.get_classifier_predictions(x, y)

        logger.info(f"Evaluating internal classifier of {self.name} for classification metrics.")
        internal_metrics_calculator = Metrics(
            task_type=TaskType.CLASSIFICATION.value,
            model_name=f"{self.name}_InternalClassifier",
            x_data=x,
            y_true=y_true_discretized,
            y_pred=y_pred_internal_clf,
            y_proba=y_proba_internal_clf,
        )
        internal_metrics_calculator.save_metrics(save_path)

        self.plot_probability_mappers(plot_path=os.path.join(save_path, "probability_mappers.png"))
        return y_pred

    def cross_validate(self, x: np.ndarray, y: np.ndarray, cv: int) -> dict[str, Any]:
        """Performs cross-validation."""
        self.cv_folds = cv
        self.fit(x, y)
        return {"test_score": self.cv_score_mean_}
