"""Scikit-learn Logistic Regression model wrapper."""

from typing import Any, ClassVar, Never

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from automl_package.enums import ExplainerType, Penalty, TaskType
from automl_package.models.base import BaseModel


class SklearnLogisticRegression(BaseModel):
    """Logistic Regression model using scikit-learn."""

    _defaults: ClassVar[dict[str, Any]] = {
        "penalty": Penalty.L2,
        "C": 1.0,
        "l1_ratio": None,
        "random_seed": None,
    }

    def __init__(self, **kwargs: Any) -> None:
        """Initializes the SklearnLogisticRegression model."""
        for key, value in self._defaults.items():
            kwargs.setdefault(key, value)

        # Ensure the task_type is always CLASSIFICATION for this model
        kwargs["task_type"] = TaskType.CLASSIFICATION
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.model: LogisticRegression | None = None

    @property
    def name(self) -> str:
        """Returns the name of the model."""
        return "SKLearnLogisticRegression"

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
        if self.penalty == Penalty.ELASTICNET and self.l1_ratio is None:
            raise ValueError("l1_ratio must be specified when penalty is 'elasticnet'.")

        solver = "lbfgs"  # Default solver, supports L2
        if self.penalty == Penalty.L1:
            solver = "liblinear"
        elif self.penalty == Penalty.ELASTICNET:
            solver = "saga"  # Supports ElasticNet

        use_early_stopping = self.early_stopping_rounds is not None and forced_iterations is None

        if use_early_stopping:
            self.model = (
                LogisticRegression(penalty=self.penalty, C=self.C, solver=solver, l1_ratio=self.l1_ratio, warm_start=True, max_iter=1, **self.params)
                if self.penalty == Penalty.ELASTICNET
                else LogisticRegression(penalty=self.penalty, C=self.C, solver=solver, warm_start=True, max_iter=1, **self.params)
            )

            best_val_loss = float("inf")
            patience_counter = 0
            best_model_state = None
            best_iter = 0
            loss_history = []

            n_iterations = forced_iterations or self.params.get("max_iter", 1000)

            for i in range(n_iterations):
                self.model.fit(x_train, y_train)

                y_pred_proba_val = self.model.predict_proba(x_val)
                val_loss = log_loss(y_val, y_pred_proba_val)
                loss_history.append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = {
                        "coef_": self.model.coef_.copy(),
                        "intercept_": self.model.intercept_.copy(),
                        "classes_": self.model.classes_.copy(),
                    }
                    best_iter = i
                else:
                    patience_counter += 1

                if patience_counter >= self.early_stopping_rounds:
                    break

            if best_model_state:
                self.model.coef_ = best_model_state["coef_"]
                self.model.intercept_ = best_model_state["intercept_"]
                self.model.classes_ = best_model_state["classes_"]

            n_iterations = best_iter + 1
        else:
            n_iterations = forced_iterations or self.params.get("max_iter", 1000)
            self.model = (
                LogisticRegression(penalty=self.penalty, C=self.C, solver=solver, l1_ratio=self.l1_ratio, max_iter=n_iterations, **self.params)
                if self.penalty == Penalty.ELASTICNET
                else LogisticRegression(penalty=self.penalty, C=self.C, solver=solver, max_iter=n_iterations, **self.params)
            )
            self.model.fit(x_train, y_train)
            n_iterations = self.model.n_iter_[0]
            loss_history = []
        return n_iterations, loss_history

    def _clone(self) -> "SklearnLogisticRegression":
        """Creates a new instance of the model with the same parameters."""
        return SklearnLogisticRegression(**self.get_params())

    def get_params(self) -> dict[str, Any]:
        """Gets parameters for this estimator.

        Returns:
            dict: Parameter names mapped to their values.
        """
        params = super().get_params()
        params.update(
            {
                "penalty": self.penalty,
                "C": self.C,
                "l1_ratio": self.l1_ratio,
                "random_seed": self.random_seed,
            }
        )
        return params

    def cross_validate(self, x: np.ndarray, y: np.ndarray, cv: int) -> dict[str, Any]:
        """Performs cross-validation and returns the scores."""
        self.cv_folds = cv
        self.fit(x, y)
        return {"test_score": self.cv_score_mean_}

    def predict(self, x: np.ndarray, filter_data: bool = True) -> np.ndarray:
        """Makes predictions on new data.

        Args:
            x (np.ndarray): Feature matrix for prediction.
            filter_data (bool): If True, filter the input data using the feature selection mask.

        Returns:
            np.ndarray: Predicted class labels.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if filter_data:
            x = self._filter_predict_data(x)
        return self.model.predict(x.values)

    def predict_uncertainty(self, x: np.ndarray, filter_data: bool = True) -> np.ndarray:
        """Estimates uncertainty for predictions.

        Args:
            x (np.ndarray): Feature matrix for uncertainty estimation.
            filter_data (bool): If True, filter the input data using the feature selection mask.

        Returns:
            np.ndarray: Uncertainty estimates (e.g., 1 - confidence).
        """
        if not self.is_regression_model:  # This condition is always True for this class
            # For classification, uncertainty is typically 1 - confidence (max probability)
            if self.model is None:
                raise RuntimeError("Model has not been fitted yet.")
            if filter_data:
                x = self._filter_predict_data(x)
            probabilities = self.predict_proba(x.values, filter_data=False)
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

    def predict_proba(self, x: np.ndarray, filter_data: bool = True) -> np.ndarray:
        """Predicts class probabilities for classification tasks.

        Args:
            x (np.ndarray): Features for probability prediction.
            filter_data (bool): If True, filter the input data using the feature selection mask.

        Returns:
            np.ndarray: Predicted probabilities.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if filter_data:
            x = self._filter_predict_data(x)
        return self.model.predict_proba(x.values)

    def get_hyperparameter_search_space(self) -> dict[str, Any]:
        """Defines the hyperparameter search space for SKLearnLogisticRegression.

        Returns:
            Dict[str, Any]: A dictionary defining the hyperparameter search space.
        """
        space = {
            "penalty": {"type": "categorical", "choices": [p.value for p in Penalty]},
            "C": {
                "type": "float",
                "low": 1e-4,
                "high": 10.0,
                "log": True,
            },  # Inverse of regularization strength
            "max_iter": {"type": "int", "low": 100, "high": 1000, "step": 100},
        }
        # l1_ratio is only relevant for 'elasticnet' penalty
        space["l1_ratio"] = {
            "type": "float",
            "low": 0.0,
            "high": 1.0,
            "step": 0.1,
        }  # Conditional parameter

        if self.search_space_override:
            space.update(self.search_space_override)

        return space

    def get_internal_model(self) -> LogisticRegression | None:
        """Returns the raw underlying scikit-learn model."""
        return self.model

    def get_shap_explainer_info(self) -> dict[str, Any]:
        """Gets the SHAP explainer type and the model to be explained."""
        return {"explainer_type": ExplainerType.LINEAR, "model": self.get_internal_model()}

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
