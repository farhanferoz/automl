"""PyTorch Linear Regression model."""

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from models.base_pytorch import PyTorchModelBase

from automl_package.enums import ExplainerType, TaskType, UncertaintyMethod
from automl_package.utils.losses import nll_loss


class PyTorchLinearRegression(PyTorchModelBase):
    """A Linear Regression model implemented in PyTorch.

    This model benefits from the base class's features, including support for
    L1, L2, and automatically learned regularization.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initializes the PyTorchLinearRegression model.

        Args:
            **kwargs: Additional keyword arguments for PyTorchModelBase.
        """
        # Ensure the task_type is always REGRESSION for this model
        kwargs["task_type"] = TaskType.REGRESSION
        self.positive_features = kwargs.pop("positive_features", None)
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        """Returns the name of the model."""
        return "PyTorchLinearRegression"

    def get_internal_model(self) -> Any:
        """Returns the internal model."""

        class ShapModel:
            def __init__(self, coef: np.ndarray, intercept: np.ndarray) -> None:
                self.coef_ = coef
                self.intercept_ = intercept

        # For SHAP, we only explain the mean prediction.
        # The weights for the mean are the first row of the weight matrix.
        linear_layer = self.model[0]
        coef = linear_layer.weight.data[0].cpu().numpy().flatten()
        intercept = linear_layer.bias.data[0].cpu().numpy()

        return ShapModel(coef, intercept)

    def get_shap_explainer_info(self) -> dict[str, Any]:
        """Gets the SHAP explainer type and the model to be explained."""
        return {"explainer_type": ExplainerType.LINEAR, "model": self.get_internal_model()}

    def _after_step(self) -> None:
        """Applies positivity constraints to the weights after each optimizer step."""
        if self.positive_features and self.feature_to_idx_:
            positive_indices = [self.feature_to_idx_[feat] for feat in self.positive_features if feat in self.feature_to_idx_]
            if positive_indices:
                linear_layer = self.model[0]
                with torch.no_grad():
                    weights = linear_layer.weight.data
                    weights[0, positive_indices] = torch.clamp(weights[0, positive_indices], min=0)

    def get_hyperparameter_search_space(self) -> dict[str, Any]:
        """Gets the hyperparameter search space for the model."""
        space = super().get_hyperparameter_search_space()
        if self.early_stopping_rounds is None:
            space["n_epochs"] = {"type": "int", "low": 5, "high": 100, "step": 10}
        if self.search_space_override:
            space.update(self.search_space_override)
        return space

    def build_model(self) -> None:
        """Builds the model architecture.

        For linear regression, this is a single linear layer.
        If probabilistic uncertainty is used, the output size is 2 (mean and log_variance).
        """
        output_dim = 2 if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC else 1
        self.model = nn.Sequential(nn.Linear(self.input_size, output_dim)).to(self.device)
        self.criterion = nll_loss if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC else nn.MSELoss()
