"""Common training loop for PyTorch models, and the clipping / learning-rate schedule every `PyTorchModelBase` loop shares."""

import math
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from automl_package.enums import LRSchedule
from automl_package.logger import logger
from automl_package.utils.convergence import ConvergenceTracker, snapshot_state


class TrainingSchedule:
    """Gradient clipping and the learning-rate schedule of one training pass (`PyTorchModelBase` settings, built once per `_fit_single`).

    `clip` runs between the backward pass and the optimizer step; `end_epoch` once at the end of every epoch with that epoch's held-out
    fit term (None when the pass reads none). Over epoch ``e`` (from 0) of an ``n_epochs`` budget, with warmup ``w`` and base rate ``lr``:

    - warmup, ``e < w``: ``lr * (e + 1) / w`` (linear, the FT-Transformer baseline's form);
    - COSINE, ``e >= w``: ``lr * (1 + cos(pi * (e - w) / (n_epochs - w))) / 2``. The budget, not the epochs this pass runs, sets the curve,
      so a refit for a fixed number of epochs follows the same rates as the early-stopped pass that chose that number;
    - REDUCE_ON_PLATEAU, ``e >= w``: torch's `ReduceLROnPlateau` (mode min) on the held-out fit term: the rate is multiplied by ``factor``
      after more than ``patience`` epochs without a new best;
    - NONE: ``lr``.

    Every setting off (no clipping, NONE, ``w = 0``): no scheduler is built and both calls do nothing, so the loop runs unchanged.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        grad_clip_norm: float | None,
        lr_schedule: LRSchedule,
        n_epochs: int,
        factor: float,
        patience: int,
        warmup_epochs: int,
    ) -> None:
        """Builds the schedulers the settings need on ``optimizer`` (none when every setting is off)."""
        self.grad_clip_norm = grad_clip_norm
        self.lr_schedule = lr_schedule
        self.warmup_epochs = warmup_epochs
        self.n_epochs = n_epochs
        self._optimizer = optimizer
        self._epochs_done = 0
        by_epoch = warmup_epochs > 0 or lr_schedule is LRSchedule.COSINE
        self._by_epoch = LambdaLR(optimizer, self.rate_factor) if by_epoch else None
        plateau = lr_schedule is LRSchedule.REDUCE_ON_PLATEAU
        self._plateau = ReduceLROnPlateau(optimizer, mode="min", factor=factor, patience=patience) if plateau else None
        if grad_clip_norm or by_epoch or plateau:
            logger.info(
                f"Training schedule: clipping at norm {grad_clip_norm}, learning rate {lr_schedule.value} (factor {factor}, patience {patience}), warmup {warmup_epochs} epochs"
            )

    @property
    def clips(self) -> bool:
        """Whether `clip` changes the gradients."""
        return bool(self.grad_clip_norm)

    def rate_factor(self, epoch: int) -> float:
        """The multiple of the base learning rate over ``epoch`` set by warmup and COSINE (REDUCE_ON_PLATEAU acts after warmup, on top)."""
        if epoch < self.warmup_epochs:
            return (epoch + 1) / self.warmup_epochs
        if self.lr_schedule is LRSchedule.COSINE:
            return 0.5 * (1.0 + math.cos(math.pi * (epoch - self.warmup_epochs) / max(self.n_epochs - self.warmup_epochs, 1)))
        return 1.0

    def clip(self, model: nn.Module) -> None:
        """Clips the gradients of ``model`` to the total norm ``grad_clip_norm`` (nothing when it is None or 0)."""
        if self.grad_clip_norm:
            nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip_norm)

    def end_epoch(self, held_out_fit: float | None) -> None:
        """Moves the learning rate on after one epoch; ``held_out_fit`` is that epoch's held-out fit term (None: the pass reads none).

        Raises:
            ValueError: REDUCE_ON_PLATEAU past warmup with no held-out fit term.
        """
        self._epochs_done += 1
        if self._plateau is not None and self._epochs_done > self.warmup_epochs:
            if held_out_fit is None:
                raise ValueError("lr_schedule=REDUCE_ON_PLATEAU reads the held-out fit term every epoch and this pass has none: give it validation rows, or use COSINE.")
            before = self._optimizer.param_groups[0]["lr"]
            self._plateau.step(held_out_fit)
            after = self._optimizer.param_groups[0]["lr"]
            if after != before:
                logger.info(f"Learning rate {before:.3g} -> {after:.3g} after epoch {self._epochs_done} (no new best held-out fit term for more than the patience)")
        elif self._by_epoch is not None and (self.lr_schedule is LRSchedule.COSINE or self._epochs_done <= self.warmup_epochs):
            self._by_epoch.step()


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

    best_model_state = None
    best_epoch = epochs

    use_early_stopping = early_stopping_rounds and early_stopping_rounds > 0 and probas_val is not None and y_val is not None
    tracker = ConvergenceTracker(patience=early_stopping_rounds or 0, min_delta=0.0)  # strict-decrease best, this loop's own rule

    if regularization_fn is None:

        def regularization_fn(loss: torch.Tensor, _: torch.nn.Module) -> torch.Tensor:
            return loss

    final_epoch = 0
    for epoch in range(epochs):
        final_epoch = epoch
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

            if tracker.update(epoch, val_loss):
                best_model_state = snapshot_state(model)
                best_epoch = epoch + 1

            if tracker.done:
                logger.info(f"Early stopping at epoch {epoch + 1}, best epoch was {best_epoch}")
                break

    if use_early_stopping and best_model_state:
        model.load_state_dict(best_model_state)

    if use_early_stopping:
        model.convergence_ = tracker.result(final_epoch=final_epoch)
    return best_epoch, tracker.best_val
