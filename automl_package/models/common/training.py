"""Common training loop for PyTorch models."""

from collections.abc import Callable
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

from automl_package.logger import logger


def train_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    probas_train: torch.Tensor,
    y_train: torch.Tensor,
    probas_val: torch.Tensor | None,
    y_val: torch.Tensor | None,
    epochs: int,
    batch_size: int,
    early_stopping_rounds: int | None,
    device: torch.device,
    optimizer_wrapper: Any,
    regularization_fn: Callable | None = None,
    lambda_optimizer: torch.optim.Optimizer | None = None,
    forward_pass_kwargs: dict | None = None,
    apply_boundary_loss_during_validation: bool = False,
) -> tuple[int, float | None]:
    """Trains a PyTorch model.

    Args:
        model: The model to train.
        optimizer: The optimizer to use.
        loss_fn: The loss function to use.
        probas_train: The training probabilities.
        y_train: The training labels.
        probas_val: The validation probabilities.
        y_val: The validation labels.
        epochs: The number of epochs to train for.
        batch_size: The batch size to use.
        early_stopping_rounds: The number of epochs with no improvement after which training will be stopped.
        device: The device to train on.
        optimizer_wrapper: The optimizer wrapper to use.
        regularization_fn: An optional function to apply regularization.
        lambda_optimizer: An optional optimizer for learnable regularization lambdas.
        forward_pass_kwargs: Optional keyword arguments to pass to the model's forward pass.
        apply_boundary_loss_during_validation: If True, include boundary loss in validation.

    Returns:
        A tuple containing:
            - The number of epochs the model was trained for.
            - The best validation loss.
    """
    dataset = TensorDataset(probas_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_state = None
    best_epoch = epochs

    use_early_stopping = early_stopping_rounds and early_stopping_rounds > 0 and probas_val is not None and y_val is not None

    # If no regularization function is provided, use a default one that does nothing.
    if regularization_fn is None:
        regularization_fn = lambda loss, model: loss

    for epoch in range(epochs):
        model.train()
        for probas_batch_cpu, y_batch_cpu in loader:
            probas_batch = probas_batch_cpu.to(device)
            y_batch = y_batch_cpu.to(device)

            if lambda_optimizer:
                lambda_optimizer.zero_grad()

            optimizer_wrapper.step(
                model=model, loss_fn=loss_fn, optimizer=optimizer, batch_x=probas_batch, batch_y=y_batch, regularization_fn=regularization_fn, forward_pass_kwargs=forward_pass_kwargs
            )

            if lambda_optimizer:
                lambda_optimizer.step()

        if use_early_stopping:
            model.eval()
            with torch.no_grad():
                val_preds = model(probas_val, **(forward_pass_kwargs or {}))
                val_loss = loss_fn(val_preds, y_val, include_boundary_loss=apply_boundary_loss_during_validation)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model_state = model.state_dict()
                best_epoch = epoch + 1
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stopping_rounds:
                logger.info(f"Early stopping at epoch {epoch + 1}, best epoch was {best_epoch}")
                break

    if use_early_stopping and best_model_state:
        model.load_state_dict(best_model_state)

    return best_epoch, best_val_loss
