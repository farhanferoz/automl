"""PyTorch Logistic Regression model."""

from typing import Any

import numpy as np
import torch.nn as nn
from models.base_pytorch import PyTorchModelBase

from automl_package.enums import ExplainerType, TaskType


class PyTorchLogisticRegression(PyTorchModelBase):
    """A Logistic Regression model implemented in PyTorch.

    This model supports both binary and multi-class classification and leverages
    the base class features like L1, L2, and learned regularization.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initializes the PyTorchLogisticRegression model.

        Args:
            **kwargs: Additional keyword arguments for PyTorchModelBase.
        """
        # Ensure the task_type is always CLASSIFICATION for this model
        kwargs["task_type"] = TaskType.CLASSIFICATION
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        """Returns the name of the model."""
        return "PyTorchLogisticRegression"

    def build_model(self) -> None:
        """Builds the model architecture.

        For logistic regression, this is a single linear layer.
        The output activation (sigmoid or softmax) is handled by the loss function.
        """
        self.model = nn.Sequential(nn.Linear(self.input_size, self.output_size)).to(self.device)

        # The criterion is set in the base class based on task_type and output_size
        if self.task_type == TaskType.CLASSIFICATION:
            if self.output_size == 1:  # Binary classification
                self.criterion = nn.BCEWithLogitsLoss()
            else:  # Multi-class classification
                self.criterion = nn.CrossEntropyLoss()
        else:
            # This should not be reached due to the __init__ override
            raise ValueError("PyTorchLogisticRegression only supports 'classification' task_type.")

    def get_internal_model(self) -> Any:
        """Returns the internal model."""

        class ShapModel:
            def __init__(self, coef: np.ndarray, intercept: np.ndarray) -> None:
                self.coef_ = coef
                self.intercept_ = intercept

        # Extract weights and bias from the linear layer
        linear_layer = self.model[0]
        coef = linear_layer.weight.data.cpu().numpy().flatten()
        intercept = linear_layer.bias.data.cpu().numpy()

        return ShapModel(coef, intercept)

    def get_shap_explainer_info(self) -> dict[str, Any]:
        """Gets the SHAP explainer type and the model to be explained."""
        return {"explainer_type": ExplainerType.LINEAR, "model": self.get_internal_model()}

    def get_hyperparameter_search_space(self) -> dict[str, Any]:
        """Gets the hyperparameter search space for the model."""
        space = super().get_hyperparameter_search_space()
        if self.early_stopping_rounds is None:
            space["n_epochs"] = {"type": "int", "low": 5, "high": 100, "step": 10}
        if self.search_space_override:
            space.update(self.search_space_override)
        return space