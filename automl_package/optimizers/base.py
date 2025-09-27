"""Base class for optimizer wrappers."""

from abc import ABC, abstractmethod

import torch


class OptimizerWrapper(ABC):
    """Abstract base class for optimizer wrappers."""

    def _calculate_total_loss(
        self,
        model: torch.nn.Module,
        loss_fn: callable,
        regularization_fn: callable,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        forward_pass_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Helper method to calculate the total loss, including regularization."""
        outputs = model(batch_x, **(forward_pass_kwargs or {}))
        loss = loss_fn(outputs, batch_y)
        return regularization_fn(loss, model), outputs

    @abstractmethod
    def create_optimizer(self, model_params: list, lr: float) -> torch.optim.Optimizer:
        """Creates and returns an optimizer instance."""
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        model: torch.nn.Module,
        loss_fn: callable,
        regularization_fn: callable,
        optimizer: torch.optim.Optimizer,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        forward_pass_kwargs: dict | None = None,
    ) -> None:
        """Performs a single, complete optimization step."""
        raise NotImplementedError
