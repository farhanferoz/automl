"""Base classes for machine learning models."""

import abc
from typing import Any

import numpy as np


class BaseModel(abc.ABC):
    """Abstract base class for all machine learning models in the package.

    Defines a common interface for fitting, predicting, and hyperparameter search.
    """

    def __init__(self, early_stopping_rounds: int | None = None, validation_fraction: float = 0.1, **kwargs: Any) -> None:
        """Initializes the base model with given parameters.

        Args:
            early_stopping_rounds (int, optional): Activates early stopping. Training will stop if validation
                                                   metric doesn't improve for this many consecutive rounds.
            validation_fraction (float): The fraction of the training data to use as a validation set for early stopping.
            **kwargs: Arbitrary keyword arguments for model parameters.
        """
        self.model = None
        self.params = kwargs
        self.is_regression_model = False  # Flag to indicate if model is for regression
        self.is_composite_regression_model = False  # New flag to identify composite models that have an internal classifier
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction

    @abc.abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> int:
        """Fits the model to the training data.

        Args:
            x (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.

        Returns:
            int: Number of iterations/epochs/estimators.
        """

    @abc.abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Makes predictions on new data.

        Args:
            x (np.ndarray): Feature matrix for prediction.

        Returns:
            np.ndarray: Predicted values.
        """

    # New abstract method for uncertainty
    @abc.abstractmethod
    def predict_uncertainty(self, x: np.ndarray) -> np.ndarray:
        """Estimates the uncertainty of predictions for new data.

        This method is only applicable for regression models.
        For classification models, this method should raise an error or return None.

        Args:
            x (np.ndarray): Feature matrix for uncertainty estimation.

        Returns:
            np.ndarray: Uncertainty estimates (e.g., standard deviation) for each prediction.
        """

    @abc.abstractmethod
    def get_hyperparameter_search_space(self) -> dict[str, Any]:
        """Defines the hyperparameter search space for Optuna.

        Returns:
            Dict[str, Any]: A dictionary where keys are hyperparameter names
                            and values are dictionaries defining the type
                            and range for Optuna.
                            Example: {'param_name': {'type': 'int', 'low': 1, 'high': 10}}
        """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Returns the name of the model."""

    @abc.abstractmethod
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Makes probability predictions on new data. Applicable for classification models.

        Args:
            x (np.ndarray): Feature matrix for prediction.

        Returns:
            np.ndarray: Predicted probabilities.
        """

    @abc.abstractmethod
    def get_classifier_predictions(self, x: np.ndarray, y_true_original: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns the internal classifier's predicted classes, probabilities, and.

        the corresponding (discretized) true labels for composite models.
        For non-composite models, this method should raise NotImplementedError.

        Args:
            x (np.ndarray): Feature matrix.
            y_true_original (np.ndarray): Original true labels (will be discretized internally).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - Predicted classes from the internal classifier.
                - Predicted probabilities from the internal classifier.
                - Discretized true labels corresponding to the internal classifier's task.
        """

    @abc.abstractmethod
    def get_num_parameters(self) -> int:
        """Returns the total number of trainable parameters in the model."""

    @abc.abstractmethod
    def evaluate(self, x: np.ndarray, y: np.ndarray, save_path: str = "metrics") -> np.ndarray:
        """Evaluates the model on a given dataset and saves the metrics.

        Args:
            x (np.ndarray): Feature matrix for evaluation.
            y (np.ndarray): True labels for evaluation.
            save_path (str): Directory to save the metrics files.

        Returns:
            np.ndarray: The predictions made by the model.
        """

    def get_internal_model(self) -> Any:
        """Returns the raw underlying model object, if applicable.

        Useful for explainability libraries like SHAP.
        """
        return self.model
