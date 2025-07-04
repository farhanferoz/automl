import abc
from typing import Dict, Any, Tuple
import numpy as np


class BaseModel(abc.ABC):
    """
    Abstract base class for all machine learning models in the package.
    Defines a common interface for fitting, predicting, and hyperparameter search.
    """

    def __init__(self, **kwargs):
        """
        Initializes the base model with given parameters.
        Args:
            **kwargs: Arbitrary keyword arguments for model parameters.
        """
        self.model = None
        self.params = kwargs
        self.is_regression_model = False  # Flag to indicate if model is for regression
        self.is_composite_regression_model = False  # New flag to identify composite models that have an internal classifier

    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits the model to the training data.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
        """
        pass

    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions on new data.

        Args:
            X (np.ndarray): Feature matrix for prediction.

        Returns:
            np.ndarray: Predicted values.
        """
        pass

    # New abstract method for uncertainty
    @abc.abstractmethod
    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """
        Estimates the uncertainty of predictions for new data.
        This method is only applicable for regression models.
        For classification models, this method should raise an error or return None.

        Args:
            X (np.ndarray): Feature matrix for uncertainty estimation.

        Returns:
            np.ndarray: Uncertainty estimates (e.g., standard deviation) for each prediction.
        """
        pass

    @abc.abstractmethod
    def get_hyperparameter_search_space(self) -> Dict[str, Any]:
        """
        Defines the hyperparameter search space for Optuna.

        Returns:
            Dict[str, Any]: A dictionary where keys are hyperparameter names
                            and values are dictionaries defining the type
                            and range for Optuna.
                            Example: {'param_name': {'type': 'int', 'low': 1, 'high': 10}}
        """
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        Returns the name of the model.
        """
        pass

    @abc.abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Makes probability predictions on new data. Applicable for classification models.

        Args:
            X (np.ndarray): Feature matrix for prediction.

        Returns:
            np.ndarray: Predicted probabilities.
        """
        pass

    @abc.abstractmethod
    def get_classifier_predictions(self, X: np.ndarray, y_true_original: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns the internal classifier's predicted classes, probabilities, and
        the corresponding (discretized) true labels for composite models.
        For non-composite models, this method should raise NotImplementedError.

        Args:
            X (np.ndarray): Feature matrix.
            y_true_original (np.ndarray): Original true labels (will be discretized internally).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - Predicted classes from the internal classifier.
                - Predicted probabilities from the internal classifier.
                - Discretized true labels corresponding to the internal classifier's task.
        """
        pass

    def get_internal_model(self):
        """
        Returns the raw underlying model object, if applicable.
        Useful for explainability libraries like SHAP.
        """
        return self.model
