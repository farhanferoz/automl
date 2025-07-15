import torch.nn as nn

from .base_pytorch import PyTorchModelBase
from ..enums import TaskType


class PyTorchLogisticRegression(PyTorchModelBase):
    """
    A Logistic Regression model implemented in PyTorch.
    This model supports both binary and multi-class classification and leverages
    the base class features like L1, L2, and learned regularization.
    """

    def __init__(self, **kwargs):
        # Ensure the task_type is always CLASSIFICATION for this model
        kwargs["task_type"] = TaskType.CLASSIFICATION
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "PyTorchLogisticRegression"

    def build_model(self):
        """
        Builds the model architecture.
        For logistic regression, this is a single linear layer.
        The output activation (sigmoid or softmax) is handled by the loss function.
        """
        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.output_size)
        ).to(self.device)

        # The criterion is set in the base class based on task_type and output_size
        if self.task_type == TaskType.CLASSIFICATION:
            if self.output_size == 1:  # Binary classification
                self.criterion = nn.BCEWithLogitsLoss()
            else:  # Multi-class classification
                self.criterion = nn.CrossEntropyLoss()
        else:
            # This should not be reached due to the __init__ override
            raise ValueError("PyTorchLogisticRegression only supports 'classification' task_type.")
