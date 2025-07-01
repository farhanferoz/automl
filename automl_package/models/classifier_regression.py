from typing import Dict, Any, Type, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt # New import
from torch.nn import ReLU, Tanh # Explicitly import ReLU and Tanh

from .base import BaseModel  # Import BaseModel
from ..enums import MapperType, TaskType, ModelName, UncertaintyMethod  # Import enums
from ..logger import logger  # Import logger
from ..utils.probability_mapper import ClassProbabilityMapper


class ClassifierRegressionModel(BaseModel):
    """
    A composite model that combines a classification model with a probability mapper
    to perform regression. It first discretizes the target into N classes,
    trains a classifier on these classes, and then converts classification probabilities
    back to continuous regression output using various mapping strategies.
    """

    def __init__(
        self,
        n_classes: int = 5,
        base_classifier_class: Type[BaseModel] = None,  # The class of the internal classifier (e.g., PyTorchNeuralNetwork, XGBoostModel)
        base_classifier_params: Dict[str, Any] = None,  # Parameters for the base classifier
        mapper_type: MapperType = MapperType.LINEAR,  # Use enum for mapper type
        mapper_params: Dict[str, Any] = None,  # Parameters for the ClassProbabilityMapper
        **kwargs,
    ):
        super().__init__(**kwargs)
        if base_classifier_class is None:
            raise ValueError("base_classifier_class must be provided.")
        self.n_classes = n_classes
        self.base_classifier_class = base_classifier_class
        self.base_classifier_params = base_classifier_params if base_classifier_params is not None else {}
        self.mapper_type = mapper_type
        self.mapper_params = mapper_params if mapper_params is not None else {}

        self.base_classifier: Optional[BaseModel] = None  # The instantiated base classifier model
        self.class_boundaries: Optional[np.ndarray] = None  # Stores percentile values for discretization
        self.class_mappers: List[Optional[ClassProbabilityMapper]] = []  # Stores a ClassProbabilityMapper for each class
        self._is_regression_model = True  # This composite model is for regression
        self.is_composite_regression_model = True # Flag for AutoML to identify composite models

        if self.n_classes < 2:
            raise ValueError("n_classes must be at least 2 for classification-regression strategy.")

    @property
    def name(self) -> str:
        # Include base model name and mapper type for unique identification
        # Dynamically get the name of the base classifier
        try:
            # Instantiate a dummy base_classifier_class to get its name property
            # Pass a default is_classification=True for models that require it in init
            if self.base_classifier_class.__name__ in ["XGBoostModel", "LightGBMModel", "CatBoostModel", "SKLearnLogisticRegression"]:
                temp_instance = self.base_classifier_class(is_classification=True)
                base_name = temp_instance.name
            elif self.base_classifier_class.__name__ == "PyTorchNeuralNetwork":
                # PyTorchNeuralNetwork needs input_size, output_size, task_type
                temp_instance = self.base_classifier_class(input_size=10, output_size=1, task_type=TaskType.CLASSIFICATION)
                base_name = temp_instance.name
            else:
                temp_instance = self.base_classifier_class()
                base_name = temp_instance.name
        except Exception as e:
            logger.warning(f"Could not instantiate base_classifier_class to get name: {e}. Using 'UnknownBase'.")
            base_name = "UnknownBase"

        return f"{base_name}_to_Reg_{self.mapper_type.value}"  # Dynamically append base name and mapper type

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits the underlying regression model and then trains the probability mapper.
        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Original continuous target values.
        """
        if not self._is_regression_model:  # This model should always be for regression
            raise ValueError("This model is designed for regression tasks.")

        # 1. Discretize the target variable into N balanced classes
        # Ensure y is 1D for percentile calculation
        y_flat = y.flatten() if y.ndim > 1 else y

        # Calculate class boundaries based on percentiles of the original target
        # e.g., for n_classes=5, percentiles would be [20, 40, 60, 80]
        percentiles = np.linspace(0, 100, self.n_classes + 1)[1:-1]
        self.class_boundaries = np.percentile(y_flat, percentiles)

        # Create discrete class labels (y_clf) based on these boundaries
        y_clf = np.zeros_like(y_flat, dtype=int)
        for i, boundary in enumerate(self.class_boundaries):
            y_clf[y_flat > boundary] = i + 1  # Assign class index from 0 to n_classes-1

        # 2. Train the base classifier on the discretized labels
        # Prepare parameters for the base classifier, ensuring it's set for classification
        classifier_output_size = 1 if self.n_classes == 2 else self.n_classes  # Output 1 for binary, N for multi-class

        base_classifier_init_params = self.base_classifier_params.copy()
        base_classifier_init_params["task_type"] = TaskType.CLASSIFICATION  # Ensure classifier knows its task

        # Handle specific parameters for PyTorchNN if it's the base classifier
        if self.base_classifier_class.__name__ == "PyTorchNeuralNetwork":
            base_classifier_init_params["input_size"] = X.shape[1]
            base_classifier_init_params["output_size"] = classifier_output_size
            # Convert string activation to Type[torch.nn.Module]
            if "activation" in base_classifier_init_params and isinstance(base_classifier_init_params["activation"], str):
                base_classifier_init_params["activation"] = ReLU if base_classifier_init_params["activation"] == "ReLU" else Tanh  # Assuming ReLU/Tanh
            # Ensure dropout_rate and n_mc_dropout_samples are set if uncertainty_method is MC_DROPOUT
            if base_classifier_init_params.get("uncertainty_method") == UncertaintyMethod.MC_DROPOUT.value:
                base_classifier_init_params.setdefault("dropout_rate", 0.1)
                base_classifier_init_params.setdefault("n_mc_dropout_samples", 100)
            else:  # If not MC_DROPOUT, ensure dropout is off
                base_classifier_init_params["dropout_rate"] = 0.0

        # Handle specific parameters for other boosting models if they are the base classifier
        elif self.base_classifier_class.__name__ == "XGBoostModel":
            base_classifier_init_params.setdefault("objective", "binary:logistic" if self.n_classes == 2 else "multi:softmax")
            base_classifier_init_params.setdefault("eval_metric", "logloss" if self.n_classes == 2 else "mlogloss")
        elif self.base_classifier_class.__name__ == "LightGBMModel":
            base_classifier_init_params.setdefault("objective", "binary" if self.n_classes == 2 else "multiclass")
            base_classifier_init_params.setdefault("metric", "binary_logloss" if self.n_classes == 2 else "multi_logloss")
        elif self.base_classifier_class.__name__ == "CatBoostModel":
            base_classifier_init_params.setdefault("task_type", TaskType.CLASSIFICATION)
            base_classifier_init_params.setdefault("loss_function", "Logloss" if self.n_classes == 2 else "MultiClass")
            base_classifier_init_params.setdefault("eval_metric", "Logloss" if self.n_classes == 2 else "MultiClass")

        self.base_classifier = self.base_classifier_class(**base_classifier_init_params)
        self.base_classifier.fit(X, y_clf)

        # 3. Construct a mapper for each class using classification probabilities
        # Get probabilities for all classes from the trained classifier
        # y_proba_all will be (n_samples, n_classes)
        y_proba_all = self.base_classifier.predict_proba(X)
        if y_proba_all.shape[1] != self.n_classes:
            # Handle binary classification case where predict_proba might return (N,) or (N,1) for positive class only.
            # Ensure y_proba_all is (N, 2) if it was a binary classifier.
            if self.n_classes == 2 and y_proba_all.ndim == 1:  # If 1D for positive class
                y_proba_all = np.vstack((1 - y_proba_all, y_proba_all)).T  # Convert to (N, 2)
            elif self.n_classes == 2 and y_proba_all.shape[1] == 1:  # If 2D (N,1) for positive class
                y_proba_all = np.hstack((1 - y_proba_all, y_proba_all))  # Convert to (N, 2)
            else:
                raise ValueError(f"Classifier predict_proba output shape {y_proba_all.shape} does not match n_classes {self.n_classes}.")

        self.class_mappers = [None] * self.n_classes

        for c in range(self.n_classes):
            # Define the range for the current class in terms of original y values
            # (used to select original y values belonging to this class)
            lower_bound = -np.inf if c == 0 else self.class_boundaries[c - 1]
            upper_bound = np.inf if c == self.n_classes - 1 else self.class_boundaries[c]

            # Find indices of data points whose *original* y values fall into the current class's range
            indices_in_true_class = np.where((y_flat >= lower_bound) & (y_flat < upper_bound))

            # Extract probabilities for class 'c' and original y values for these data points
            probas_for_current_class = y_proba_all[indices_in_true_class[0], c].reshape(-1, 1)
            original_y_for_current_class = y_flat[indices_in_true_class[0]]

            mapper = ClassProbabilityMapper(self.mapper_type, **self.mapper_params)
            mapper.fit(probas_for_current_class, original_y_for_current_class)
            self.class_mappers[c] = mapper

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions (expected regression output) using the composite model.
        Args:
            X (np.ndarray): Features for prediction.
        Returns:
            np.ndarray: Predicted regression values.
        """
        if self.base_classifier is None or not self.class_mappers:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        # Get classification probabilities from the base classifier for all classes
        y_proba_all = self.base_classifier.predict_proba(X)

        # Initialize final predictions array
        final_predictions = np.zeros(X.shape[0])

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

    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """
        Estimates uncertainty for regression predictions using this composite model.
        It sums the variance contributions from each class's mapper, weighted by their probabilities.

        Args:
            X (np.ndarray): Features for uncertainty estimation.
        Returns:
            np.ndarray: Uncertainty estimates (standard deviation) for each prediction.
        """
        if not self._is_regression_model:
            raise ValueError("predict_uncertainty is only available for regression models.")
        if self.base_classifier is None or not self.class_mappers:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        y_proba_all = self.base_classifier.predict_proba(X)

        # Initialize an array to store the sum of variance contributions (P_i^2 * Var_mapper_i)
        total_variance = np.zeros(X.shape[0])

        for c in range(self.n_classes):
            if self.class_mappers[c] is None:
                logger.warning(f"Mapper for class {c} is None. Skipping its variance contribution.")
                continue

            proba_c = y_proba_all[:, c]  # 1D array of probabilities for class c

            # Get variance from the mapper for this class's probabilities
            mapper_variances = self.class_mappers[c].predict_variance_contribution(proba_c)

            # Add this class's weighted variance contribution
            total_variance += (proba_c**2) * mapper_variances

        uncertainty_estimates = np.sqrt(total_variance)  # Standard deviation is sqrt of total variance
        return uncertainty_estimates

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns the predicted probabilities from the internal base classifier.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted probabilities from the internal classifier.
        """
        if self.base_classifier is None:
            raise RuntimeError("Base classifier has not been fitted yet.")
        return self.base_classifier.predict_proba(X)

    def get_hyperparameter_search_space(self) -> Dict[str, Any]:
        # This model's search space defines its own parameters and the choice of base classifier.
        # The base_classifier_params themselves will be dynamically sampled within AutoML._sample_params_for_trial
        return {
            "n_classes": {"type": "int", "low": 2, "high": 10},  # Number of classes for discretization
            # Base classifier chosen from a categorical list of model names
            "base_classifier_name": {
                "type": "categorical",
                "choices": [
                    ModelName.PYTORCH_NEURAL_NETWORK.value,
                    ModelName.XGBOOST.value,
                    ModelName.LIGHTGBM.value,
                    ModelName.SKLEARN_LOGISTIC_REGRESSION.value,
                    ModelName.CATBOOST.value,  # Ensure CatBoost is initialized for classification
                ],
            },
            "mapper_type": {"type": "categorical", "choices": [e.value for e in MapperType]},  # Use enum values
            # Parameters for lookup mappers (will be sampled conditionally in AutoML)
            "n_partitions_min_lookup": {"type": "int", "low": 5, "high": 10},
            "n_partitions_max_lookup": {"type": "int", "low": 10, "high": 50},
            # Parameters for spline mapper (will be sampled conditionally in AutoML)
            "spline_k": {"type": "int", "low": 1, "high": 3},  # Spline degree
            "spline_s": {"type": "float", "low": 0.01, "high": 10.0, "log": True},  # Spline smoothing factor
        }

    def get_classifier_predictions(self, X: np.ndarray, y_true_original: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns the internal classifier's predicted classes, probabilities, and
        the corresponding (discretized) true labels for this composite model.

        Args:
            X (np.ndarray): Feature matrix.
            y_true_original (np.ndarray): Original true labels (will be discretized internally).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - Predicted classes from the internal classifier.
                - Predicted probabilities from the internal classifier.
                - Discretized true labels corresponding to the internal classifier's task.
        """
        if self.base_classifier is None:
            raise RuntimeError("Base classifier has not been fitted yet.")

        # 1. Discretize the original true labels using the same boundaries as during training
        y_flat = y_true_original.flatten() if y_true_original.ndim > 1 else y_true_original
        y_true_discretized = np.zeros_like(y_flat, dtype=int)
        for i, boundary in enumerate(self.class_boundaries):
            y_true_discretized[y_flat > boundary] = i + 1

        # 2. Get predictions and probabilities from the internal classifier
        y_pred_internal = self.base_classifier.predict(X)
        y_proba_internal = self.base_classifier.predict_proba(X)

        return y_pred_internal, y_proba_internal, y_true_discretized

    def plot_probability_mappers(self, plot_path: str = "probability_mappers.png"):
        """
        Plots the n functions (one for each class) calculated in the probability mapper.
        Each plot shows the mapping from class probability to the original regression value.

        Args:
            plot_path (str): The path to the file where the plot will be saved.
        """
        if not self.class_mappers:
            logger.warning("No class mappers found. Please fit the model first.")
            return

        logger.info(f"\n--- Plotting Probability Mappers to {plot_path} ---")

        plt.figure(figsize=(12, 8))
        probas_range = np.linspace(0, 1, 100).reshape(-1, 1) # Probabilities from 0 to 1

        for i, mapper in enumerate(self.class_mappers):
            if mapper is None:
                logger.warning(f"Mapper for class {i} is None. Skipping plot for this class.")
                continue
            
            # Get mapped values
            mapped_values = mapper.predict(probas_range)
            if i == 0:
                lower_bound_str = "-inf"
                upper_bound_str = f"{self.class_boundaries[0]:.2f}"
            elif i == self.n_classes - 1:
                lower_bound_str = f"{self.class_boundaries[self.n_classes - 2]:.2f}"
                upper_bound_str = "inf"
            else:
                lower_bound_str = f"{self.class_boundaries[i-1]:.2f}"
                upper_bound_str = f"{self.class_boundaries[i]:.2f}"

            label_text = f'Class {i} (Range: {lower_bound_str}-{upper_bound_str})'
            plt.plot(probas_range, mapped_values, label=label_text)

        plt.title(f'Probability Mappers for {self.name}')
        plt.xlabel('Class Probability')
        plt.ylabel('Mapped Original Regression Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        logger.info("Probability mappers plot saved successfully.")

    def get_internal_model(self):
        """
        Returns the raw underlying base classifier.
        """
        if self.base_classifier:
            return self.base_classifier.get_internal_model()
        return None
