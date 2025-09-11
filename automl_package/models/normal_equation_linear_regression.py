"""Linear Regression model using the Normal Equation."""

from typing import Any, ClassVar, Never

import numpy as np

from automl_package.enums import ExplainerType, Metric
from automl_package.models.base import BaseModel


class NormalEquationLinearRegression(BaseModel):
    """Linear Regression model implemented using the Normal Equation (Ridge Regression)."""

    _defaults: ClassVar[dict[str, Any]] = {"l2_lambda": 0.0}

    def __init__(self, **kwargs: Any) -> None:
        """Initializes the NormalEquationLinearRegression model."""
        for key, value in self._defaults.items():
            kwargs.setdefault(key, value)

        super().__init__(**kwargs)
        assert self.is_regression_model
        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.early_stopping_rounds is not None:
            raise ValueError("Early stopping is not applicable to NormalEquationLinearRegression as it is a direct solution method.")
        if "l1_lambda" in kwargs and kwargs["l1_lambda"] > 0:
            raise ValueError(
                "L1 regularization (Lasso) is not supported by NormalEquationLinearRegression due to non-differentiability at zero. "
                "Use iterative solvers for L1 e.g. JAXLinearRegression."
            )

        self.weights: np.ndarray | None = None
        self.bias: np.ndarray | None = None
        self._train_residual_std = 0.0

    @property
    def name(self) -> str:
        """Returns the name of the model."""
        return "NormalEquationLinearRegression"

    def _fit_single(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,  # noqa: ARG002
        y_val: np.ndarray | None = None,  # noqa: ARG002
        forced_iterations: int | None = None,  # noqa: ARG002
    ) -> tuple[int, list[float]]:
        """Fits a single model instance.

        Args:
            x_train (np.ndarray): The training features.
            y_train (np.ndarray): The training targets.
            x_val (np.ndarray | None): The validation features (unused).
            y_val (np.ndarray | None): The validation targets (unused).
            forced_iterations (int | None): If provided, train for this many iterations, ignoring early stopping (unused).

        Returns:
            tuple[int, list[float]]: A tuple containing:
                - The number of iterations the model was trained for.
                - A list of the validation loss values for each epoch.
        """
        identity_matrix = np.identity(x_train.shape[1])
        a = x_train.T @ x_train + self.l2_lambda * identity_matrix
        b = x_train.T @ y_train

        self.weights = np.linalg.solve(a, b)

        self.bias = np.mean(y_train)

        y_pred_train = self.predict(x_train, filter_data=False)
        self._train_residual_std = np.std(y_train - y_pred_train)
        if np.isnan(self._train_residual_std):
            self._train_residual_std = 0.0

        return 1, []

    def _clone(self) -> "NormalEquationLinearRegression":
        """Creates a new instance of the model with the same parameters."""
        return NormalEquationLinearRegression(**self.get_params())

    def get_params(self) -> dict[str, Any]:
        """Gets parameters for this estimator.

        Returns:
            dict: Parameter names mapped to their values.
        """
        params = super().get_params()
        params.update(
            {
                "l2_lambda": self.l2_lambda,
            }
        )
        return params

    def predict(self, x: np.ndarray, filter_data: bool = True) -> np.ndarray:
        """Makes predictions on new data.

        Args:
            x (np.ndarray): Feature matrix for prediction.
            filter_data (bool): If True, filter the input data using the feature selection mask.

        Returns:
            np.ndarray: Predicted values.
        """
        if self.weights is None or self.bias is None:
            raise RuntimeError("Model has not been fitted yet.")
        if filter_data:
            x = self._filter_predict_data(x)
        return x @ self.weights + self.bias

    def predict_uncertainty(self, x: np.ndarray, filter_data: bool = True) -> np.ndarray:
        """Estimates uncertainty for predictions.

        Args:
            x (np.ndarray): Feature matrix for uncertainty estimation.
            filter_data (bool): If True, filter the input data using the feature selection mask.

        Returns:
            np.ndarray: Uncertainty estimates (e.g., standard deviation).
        """
        if not self.is_regression_model:
            raise ValueError("predict_uncertainty is only available for regression models.")
        if self.weights is None or self.bias is None:
            raise RuntimeError("Model has not been fitted yet.")
        if filter_data:
            x = self._filter_predict_data(x)
        return np.full(x.shape[0], self._train_residual_std)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Not implemented for NormalEquationLinearRegression.

        Raises:
            NotImplementedError: NormalEquationLinearRegression is a regression model.
        """
        raise NotImplementedError("NormalEquationLinearRegression is a regression model and does not support predict_proba.")

    def get_hyperparameter_search_space(self) -> dict[str, Any]:
        """Defines the hyperparameter search space for NormalEquationLinearRegression.

        Returns:
            Dict[str, Any]: A dictionary defining the hyperparameter search space.
        """
        space = {
            "l2_lambda": {
                "type": "float",
                "low": 1e-6,
                "high": 1.0,
                "log": True,
            },  # L2 regularization
        }
        if self.search_space_override:
            space.update(self.search_space_override)
        return space

    def get_num_parameters(self) -> int:
        """Returns the total number of trainable parameters in the model.

        Returns:
            int: The total number of parameters.
        """
        if self.weights is None:
            return 0
        return self.weights.size + 1  # weights + bias

    def get_classifier_predictions(self, x: np.ndarray, y_true_original: np.ndarray) -> Never:
        """Not implemented for NormalEquationLinearRegression.

        Raises:
            NotImplementedError: NormalEquationLinearRegression is not a composite model.
        """
        raise NotImplementedError("NormalEquationLinearRegression is not a composite model and does not have an internal classifier for separate prediction.")

    def get_internal_model(self) -> Any:
        """Returns the internal model."""

        class ShapModel:
            def __init__(self, coef: np.ndarray, intercept: np.ndarray) -> None:
                self.coef_ = coef
                self.intercept_ = intercept

        return ShapModel(self.weights, self.bias)

    def get_shap_explainer_info(self) -> dict[str, Any]:
        """Gets the SHAP explainer type and the model to be explained."""
        return {"explainer_type": ExplainerType.LINEAR, "model": self.get_internal_model()}

    def cross_validate(self, x: np.ndarray, y: np.ndarray, cv: int) -> dict[str, Any]:
        """Performs cross-validation and returns the scores."""
        self.cv_folds = cv
        self.fit(x, y)
        return {"test_score": self.cv_score_mean_}
