"""Scikit-learn Logistic Regression model wrapper."""

from typing import Any, Never

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from automl_package.models.base import BaseModel
from automl_package.utils.data_handler import create_train_val_split
from automl_package.utils.metrics import Metrics


class SKLearnLogisticRegression(BaseModel):
    """Logistic Regression model using scikit-learn.

    Primarily for classification, but used internally by ClassifierRegression.
    Supports L1, L2, and ElasticNet regularization.
    """

    def __init__(self, penalty: str = "l2", c: float = 1.0, l1_ratio: float | None = None, random_seed: int | None = None, **kwargs: Any) -> None:
        """Initializes the SKLearnLogisticRegression model.

        Args:
            penalty (str): The type of regularization to use ('l1', 'l2', or 'elasticnet').
            c (float): Inverse of regularization strength.
            l1_ratio (float, optional): The ElasticNet mixing parameter, between 0 and 1.
            random_seed (int, optional): Random seed for reproducibility.
            **kwargs: Additional keyword arguments for the BaseModel.
        """
        super().__init__(**kwargs)
        self.penalty = penalty
        self.C = c  # Inverse of regularization strength
        self.l1_ratio = l1_ratio  # For elasticnet
        self.random_seed = random_seed
        self.model: LogisticRegression | None = None
        self.is_regression_model = False  # Not a regression model

    @property
    def name(self) -> str:
        """Returns the name of the model."""
        return "SKLearnLogisticRegression"

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fits the Logistic Regression model to the training data.

        Args:
            x (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
        """
        if self.penalty == "elasticnet" and self.l1_ratio is None:
            raise ValueError("l1_ratio must be specified when penalty is 'elasticnet'.")

        solver = "lbfgs"  # Default solver, supports L2
        if self.penalty == "l1":
            solver = "liblinear"
        elif self.penalty == "elasticnet":
            solver = "saga"  # Supports ElasticNet

        # Initialize the model with warm_start for iterative fitting
        if self.penalty == "elasticnet":
            base_model = LogisticRegression(penalty=self.penalty, C=self.C, solver=solver, l1_ratio=self.l1_ratio, warm_start=True, max_iter=1, **self.params)
        else:
            base_model = LogisticRegression(penalty=self.penalty, C=self.C, solver=solver, warm_start=True, max_iter=1, **self.params)

        if self.early_stopping_rounds is not None and self.validation_fraction > 0:
            self.train_indices, self.val_indices = create_train_val_split(x, y, self.validation_fraction, self.random_seed)
            x_train, x_val = x[self.train_indices], x[self.val_indices]
            y_train, y_val = y[self.train_indices], y[self.val_indices]
        else:
            x_train, y_train = x, y
            x_val, y_val = None, None
            self.train_indices = np.arange(x.shape[0])
            self.val_indices = None

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None
        best_i = 0

        # Determine the number of iterations. Use a large number if early stopping is enabled,
        # otherwise use the model's default max_iter or a specified one.
        n_iterations = self.params.get("max_iter", 1000)

        for i in range(n_iterations):
            base_model.fit(x_train, y_train)

            if self.early_stopping_rounds is not None and x_val is not None:
                y_pred_proba_val = base_model.predict_proba(x_val)
                val_loss = log_loss(y_val, y_pred_proba_val)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Store a deep copy of the model's state
                    best_model_state = {"coef_": base_model.coef_.copy(), "intercept_": base_model.intercept_.copy()}
                    best_i = i
                else:
                    patience_counter += 1

                if patience_counter >= self.early_stopping_rounds:
                    break
            else:
                best_i = i

        # Set the model to the best state found during early stopping
        if best_model_state:
            base_model.coef_ = best_model_state["coef_"]
            base_model.intercept_ = best_model_state["intercept_"]

        self.model = base_model
        self.num_iterations_used = best_i + 1

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Makes predictions on new data.

        Args:
            x (np.ndarray): Feature matrix for prediction.

        Returns:
            np.ndarray: Predicted class labels.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self.model.predict(x)

    def predict_uncertainty(self, x: np.ndarray) -> np.ndarray:
        """Estimates uncertainty for predictions.

        Args:
            x (np.ndarray): Feature matrix for uncertainty estimation.

        Returns:
            np.ndarray: Uncertainty estimates (e.g., 1 - confidence).
        """
        if not self.is_regression_model:  # This condition is always True for this class
            # For classification, uncertainty is typically 1 - confidence (max probability)
            if self.model is None:
                raise RuntimeError("Model has not been fitted yet.")
            probabilities = self.predict_proba(x)
            # Find the max probability for each sample (confidence)
            max_probs = np.max(probabilities, axis=1)
            # Uncertainty is higher when confidence is lower (closer to 0.5 for binary)
            # Normalize to be between 0 and 1, where 1 is max uncertainty (prob=0.5)
            # 1 - 2 * abs(prob - 0.5) if proba is single value for positive class
            # Or simpler: 1 - max_prob for general classification
            return 1.0 - max_probs
        # This part of the code would technically not be reached given is_regression_model = False
        # but it's good practice to ensure all abstract methods are fully implemented conceptually.
        return np.array([])

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predicts class probabilities for classification tasks.

        Args:
            x (np.ndarray): Features for probability prediction.

        Returns:
            np.ndarray: Predicted probabilities.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self.model.predict_proba(x)

    def get_hyperparameter_search_space(self) -> dict[str, Any]:
        """Defines the hyperparameter search space for SKLearnLogisticRegression.

        Returns:
            Dict[str, Any]: A dictionary defining the hyperparameter search space.
        """
        space = {
            "penalty": {"type": "categorical", "choices": ["l1", "l2", "elasticnet"]},
            "C": {"type": "float", "low": 1e-4, "high": 10.0, "log": True},  # Inverse of regularization strength
            "max_iter": {"type": "int", "low": 100, "high": 1000, "step": 100},
        }
        # l1_ratio is only relevant for 'elasticnet' penalty
        space["l1_ratio"] = {"type": "float", "low": 0.0, "high": 1.0, "step": 0.1}  # Conditional parameter
        return space

    def get_internal_model(self) -> LogisticRegression | None:
        """Returns the raw underlying scikit-learn model."""
        return self.model

    def get_num_parameters(self) -> int:
        """Returns the total number of trainable parameters in the model.

        Returns:
            int: The total number of parameters.
        """
        if self.model is None:
            return 0  # Or raise an error if model not fitted
        return self.model.coef_.size + self.model.intercept_.size

    def get_classifier_predictions(self, x: np.ndarray, y_true_original: np.ndarray) -> Never:
        """Not implemented for SKLearnLogisticRegression.

        Raises:
            NotImplementedError: SKLearnLogisticRegression is not a composite model.
        """
        raise NotImplementedError("SKLearnLogisticRegression is not a composite model and does not have an internal classifier for separate prediction.")

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
        y_proba = self.predict_proba(x)
        metrics_calculator = Metrics(task_type="classification", model_name=self.name, x_data=x, y_true=y, y_pred=y_pred, y_proba=y_proba)
        metrics_calculator.save_metrics(save_path)
        return y_pred
