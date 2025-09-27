"""Concrete implementations of optimizer wrappers."""

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.optim as optim
from hessianfree.optimizer import HessianFree

from .base import OptimizerWrapper


class AdamWrapper(OptimizerWrapper):
    """Wrapper for the Adam optimizer."""

    def create_optimizer(self, model_params: list, lr: float) -> torch.optim.Optimizer:
        """Creates an Adam optimizer instance."""
        return optim.Adam(model_params, lr=lr)

    def step(
        self,
        model: nn.Module,
        loss_fn: Callable,
        regularization_fn: Callable,
        optimizer: torch.optim.Optimizer,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        forward_pass_kwargs: dict | None = None,
    ) -> None:
        """Performs a single optimization step for Adam."""
        optimizer.zero_grad()
        loss, _ = self._calculate_total_loss(model, loss_fn, regularization_fn, batch_x, batch_y, forward_pass_kwargs)
        loss.backward()
        optimizer.step()


class HessianFreeWrapper(OptimizerWrapper):
    """Wrapper for the Hessian-free optimizer."""

    def create_optimizer(self, model_params: list, lr: float) -> torch.optim.Optimizer:  # noqa: ARG002
        """Creates a Hessian-free optimizer instance."""
        # Note: learning rate is not used by this optimizer
        return HessianFree(model_params, verbose=False)

    def step(
        self,
        model: nn.Module,
        loss_fn: Callable,
        regularization_fn: Callable,
        optimizer: torch.optim.Optimizer,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        forward_pass_kwargs: dict | None = None,
    ) -> None:
        """Performs a single optimization step for Hessian-free."""

        def forward() -> tuple[torch.Tensor, torch.Tensor]:
            """The forward function required by the HessianFree optimizer."""
            return self._calculate_total_loss(model, loss_fn, regularization_fn, batch_x, batch_y, forward_pass_kwargs)

        optimizer.step(forward=forward)
