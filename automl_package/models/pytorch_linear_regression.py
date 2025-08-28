"""PyTorch Linear Regression model."""

from typing import Any

import torch.nn as nn
from models.base_pytorch import PyTorchModelBase

from automl_package.enums import TaskType


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
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        """Returns the name of the model."""
        return "PyTorchLinearRegression"

    def build_model(self) -> None:
        """Builds the model architecture.

        For linear regression, this is a single linear layer.
        """
        # A linear regression is a single layer mapping inputs to outputs
        self.model = nn.Sequential(nn.Linear(self.input_size, self.output_size)).to(self.device)

        # The criterion is set in the base class based on task_type
        if self.task_type == TaskType.REGRESSION:
            self.criterion = nn.MSELoss()
        else:
            # This should not be reached due to the __init__ override
            raise ValueError("PyTorchLinearRegression only supports 'regression' task_type.")
