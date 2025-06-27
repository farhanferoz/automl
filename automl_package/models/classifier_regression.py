from typing import Dict, Any, Type, List
import numpy as np

from .base import BaseModel
from .catboost_model import CatBoostModel
from .neural_network import PyTorchNeuralNetwork
from .sklearn_logistic_regression import SKLearnLogisticRegression
from .xgboost_lgbm import XGBoostModel
from ..enums import MapperType, TaskType  # Import enums
from ..utils.probability_mapper import ClassProbabilityMapper


class ClassifierRegressionModel(BaseModel):
    """
    A regression model that leverages a classification approach.
    It first discretizes the target into N classes, trains a classifier,
    and then converts classification probabilities back to regression output.
    """

    def __init__(
        self,
        n_classes: int = 5,
        base_classifier_class: Type[BaseModel] = None,
        base_classifier_params: Dict[str, Any] = None,
        mapper_type: MapperType = MapperType.LINEAR,  # Use enum
        mapper_params: Dict[str, Any] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        if base_classifier_class is None:
            raise ValueError("base_classifier_class must be provided.")
        self.n_classes = n_classes
        self.base_classifier_class = base_classifier_class
        self.base_classifier_params = base_classifier_params if base_classifier_params is not None else {}
        self.mapper_type = mapper_type
        self.mapper_params = mapper_params if mapper_params is not None else {}

        self.base_classifier = None
        self.class_boundaries = None  # Stores percentile values
        self.class_mappers: List[ClassProbabilityMapper] = []  # Stores a ClassProbabilityMapper for each class
        self._is_regression_model = True  # Set regression flag

    @property
    def name(self) -> str:
        return "ClassifierRegression"

    def fit(self, X: np.ndarray, y: np.ndarray):
        if not (self.n_classes >= 2):
            raise ValueError("n_classes must be at least 2 for classification.")

        # 1. Discretize the target variable into N balanced classes
        percentiles = np.linspace(0, 100, self.n_classes + 1)[1:-1]  # Exclude 0 and 100
        self.class_boundaries = np.percentile(y, percentiles)

        # Create discrete class labels (y_clf)
        y_clf = np.zeros_like(y, dtype=int)
        for i, boundary in enumerate(self.class_boundaries):
            y_clf[y > boundary] = i + 1

        # For binary classification with some models, output_size should be 1
        classifier_output_size = 1 if self.n_classes == 2 else self.n_classes

        # Pass task_type, input_size, and output_size explicitly for neural networks
        if self.base_classifier_class.__name__ == "PyTorchNeuralNetwork":
            base_classifier_init_params = {
                **self.base_classifier_params,
                "input_size": X.shape[1],
                "output_size": classifier_output_size,
                "task_type": TaskType.CLASSIFICATION,  # Use enum
            }
        else:
            base_classifier_init_params = {**self.base_classifier_params}
            # For XGBoost/LightGBM/CatBoost, set objective/metric for classification if not already set
            if self.base_classifier_class.__name__ == "XGBoostModel":
                if "objective" not in base_classifier_init_params:
                    base_classifier_init_params["objective"] = "binary:logistic" if self.n_classes == 2 else "multi:softmax"
                if "eval_metric" not in base_classifier_init_params:
                    base_classifier_init_params["eval_metric"] = "logloss"
            elif self.base_classifier_class.__name__ == "LightGBMModel":
                if "objective" not in base_classifier_init_params:
                    base_classifier_init_params["objective"] = "binary" if self.n_classes == 2 else "multiclass"
                if "metric" not in base_classifier_init_params:
                    base_classifier_init_params["metric"] = "binary_logloss" if self.n_classes == 2 else "multi_logloss"
            elif self.base_classifier_class.__name__ == "CatBoostModel":
                base_classifier_init_params["task_type"] = TaskType.CLASSIFICATION  # Use enum
                if self.n_classes > 2:
                    base_classifier_init_params.setdefault("loss_function", "MultiClass")
                    base_classifier_init_params.setdefault("eval_metric", "MultiClass")
                else:
                    base_classifier_init_params.setdefault("loss_function", "Logloss")
                    base_classifier_init_params.setdefault("eval_metric", "Logloss")

        # 2. Train the base classifier on the discretized labels
        self.base_classifier = self.base_classifier_class(**base_classifier_init_params)
        self.base_classifier.fit(X, y_clf)

        # 3. Construct a mapper for each class using classification probabilities
        # Get probabilities for all classes
        y_proba_all = self.base_classifier.predict_proba(X)

        self.class_mappers = [None] * self.n_classes

        for c in range(self.n_classes):
            # Define the range for the current class in terms of original y values
            lower_bound = -np.inf if c == 0 else self.class_boundaries[c - 1]
            upper_bound = np.inf if c == self.n_classes - 1 else self.class_boundaries[c]

            # Find indices of data points that fall into the current class
            # This is based on original y values, not predicted class
            indices_in_true_class = np.where((y >= lower_bound) & (y <= upper_bound))

            probas_for_current_class = y_proba_all[indices_in_true_class[0], c].reshape(-1, 1)
            original_y_for_current_class = y[indices_in_true_class[0]]

            mapper = ClassProbabilityMapper(self.mapper_type, **self.mapper_params)
            mapper.fit(probas_for_current_class, original_y_for_current_class)
            self.class_mappers[c] = mapper

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.base_classifier is None or not self.class_mappers:
            raise RuntimeError("Model has not been fitted yet.")

        # Get classification probabilities from the base classifier
        y_proba_all = self.base_classifier.predict_proba(X)

        # Initialize predictions array
        final_predictions = np.zeros(X.shape[0])

        # Sum the product of classification probability and expected regression output for each class
        for c in range(self.n_classes):
            if self.class_mappers[c] is None:
                continue

            proba_for_current_class = y_proba_all[:, c].reshape(-1, 1)
            expected_y_from_mapper = self.class_mappers[c].predict(proba_for_current_class)
            final_predictions += proba_for_current_class.flatten() * expected_y_from_mapper

        return final_predictions

    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        if not self._is_regression_model:
            raise ValueError("predict_uncertainty is only available for regression models.")
        if self.base_classifier is None or not self.class_mappers:
            raise RuntimeError("Model has not been fitted yet.")

        y_proba_all = self.base_classifier.predict_proba(X)

        # Initialize an array to store the sum of squared (proba * std_dev) contributions
        # We will sum variances, then take the square root at the end.
        total_variance = np.zeros(X.shape[0])

        for c in range(self.n_classes):
            if self.class_mappers[c] is None:
                continue

            proba_c = y_proba_all[:, c]  # 1D array of probabilities for class c
            mapper_variances = self.class_mappers[c].predict_variance_contribution(proba_c)

            total_variance += (proba_c**2) * mapper_variances

        uncertainty_estimates = np.sqrt(total_variance)
        return uncertainty_estimates

    def get_hyperparameter_search_space(self) -> Dict[str, Any]:
        # This model's search space defines its own parameters and the choice of base classifier.
        # The base_classifier_params themselves will be dynamically sampled in AutoML.train
        return {
            "n_classes": {"type": "int", "low": 2, "high": 10},  # Number of classes for discretization
            "base_classifier_name": {
                "type": "categorical",
                "choices": [
                    # bm.name for bm in [PyTorchNeuralNetwork(), XGBoostModel(), LightGBMModel(), SKLearnLogisticRegression(), CatBoostModel(task_type=TaskType.CLASSIFICATION)]
                    bm.name for bm in [PyTorchNeuralNetwork(), XGBoostModel(), SKLearnLogisticRegression(), CatBoostModel(task_type=TaskType.CLASSIFICATION)]
                ],
            },  # Use name property of models
            "mapper_type": {"type": "categorical", "choices": [e.value for e in MapperType]},  # Use enum values
            # Parameters for lookup mappers, these will be sampled conditionally in AutoML
            "n_partitions_min_lookup": {"type": "int", "low": 5, "high": 10},  # Added lookup-specific params
            "n_partitions_max_lookup": {"type": "int", "low": 10, "high": 50},  # Added lookup-specific params
            # Parameters for spline mapper
            "spline_k": {"type": "int", "low": 1, "high": 3},  # Added spline degree
            "spline_s": {"type": "float", "low": 0.01, "high": 10.0, "log": True},  # Added spline smoothing factor
        }

    def get_internal_model(self):
        """
        Returns the raw underlying base classifier.
        Note: The ClassProbabilityMappers are not returned here as they are simple scikit-learn models or lookups.
        """
        return self.base_classifier.get_internal_model()
