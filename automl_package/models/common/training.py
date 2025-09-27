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
    y_binned_train: torch.Tensor | None = None,
) -> tuple[int, float | None]:
    """Trains a PyTorch model."""
    dataset = TensorDataset(probas_train, y_train, y_binned_train) if y_binned_train is not None else TensorDataset(probas_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_state = None
    best_epoch = epochs

    use_early_stopping = early_stopping_rounds and early_stopping_rounds > 0 and probas_val is not None and y_val is not None

    if regularization_fn is None:

        def regularization_fn(loss: torch.Tensor, _: torch.nn.Module) -> torch.Tensor:
            return loss

    for epoch in range(epochs):
        model.train()
        for batch in loader:
            if y_binned_train is not None:
                probas_batch_cpu, y_batch_cpu, y_binned_batch_cpu = batch
                y_binned_batch = y_binned_batch_cpu.to(device)
            else:
                probas_batch_cpu, y_batch_cpu = batch
                y_binned_batch = None

            probas_batch = probas_batch_cpu.to(device)
            y_batch = y_batch_cpu.to(device)

            batch_forward_pass_kwargs = forward_pass_kwargs.copy() if forward_pass_kwargs else {}
            if y_binned_batch is not None and "class_value_ranges" in batch_forward_pass_kwargs:
                class_value_ranges = batch_forward_pass_kwargs.pop("class_value_ranges")
                if "y_binned_tensor" in batch_forward_pass_kwargs:
                    del batch_forward_pass_kwargs["y_binned_tensor"]
                batch_forward_pass_kwargs["boundaries"] = class_value_ranges[y_binned_batch]

            if lambda_optimizer:
                lambda_optimizer.zero_grad()

            optimizer_wrapper.step(
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                batch_x=probas_batch,
                batch_y=y_batch,
                regularization_fn=regularization_fn,
                forward_pass_kwargs=batch_forward_pass_kwargs,
            )

            if lambda_optimizer:
                lambda_optimizer.step()

        if use_early_stopping:
            model.eval()
            with torch.no_grad():
                val_forward_pass_kwargs = forward_pass_kwargs.copy() if forward_pass_kwargs else {}
                if "y_binned_tensor" in val_forward_pass_kwargs:
                    del val_forward_pass_kwargs["y_binned_tensor"]
                if "class_value_ranges" in val_forward_pass_kwargs:
                    del val_forward_pass_kwargs["class_value_ranges"]
                val_preds = model(probas_val, **(val_forward_pass_kwargs or {}))
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
