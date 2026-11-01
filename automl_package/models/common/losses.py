"""Common loss functions for PyTorch models."""

from typing import Literal

import torch
import torch.nn as nn

from automl_package.enums import UncertaintyMethod
from automl_package.utils.losses import beta_nll_loss, boundary_regularization_loss, nll_loss
from automl_package.utils.numerics import create_bins


def calculate_combined_loss(
    predictions: torch.Tensor,
    y_true: torch.Tensor,
    uncertainty_method: UncertaintyMethod,
    class_boundaries: torch.Tensor | None = None,
    class_value_ranges: torch.Tensor | None = None,
    boundary_loss_weight: float | None = None,
    device: torch.device | None = None,
    include_boundary_loss: bool = False,
    loss_type: str = "nll",
    beta: float = 0.5,
    reduction: Literal["mean", "none"] = "mean",
) -> torch.Tensor:
    """Calculates the combined loss for a regression model.

    Args:
        predictions: The model's predictions.
        y_true: The true labels.
        uncertainty_method: The uncertainty estimation method.
        use_boundary_regularization: Whether to use boundary regularization.
        class_boundaries: The boundaries for each class.
        class_value_ranges: The value ranges for each class.
        boundary_loss_weight: The weight of the boundary loss.
        device: The device to use.
        include_boundary_loss: Whether to include the boundary loss.
        loss_type: `"nll"` or `"beta_nll"` for the PROBABILISTIC fit term.
        beta: beta-NLL exponent.
        reduction: `"mean"` (default) or `"none"` for the `(N,)` per-sample fit term. `"none"` is only
            defined for the fit term, so it cannot be combined with `include_boundary_loss`.

    Returns:
        The combined loss.
    """
    if reduction == "none" and include_boundary_loss:
        raise ValueError("reduction='none' returns the per-sample fit term only; it cannot include the boundary loss.")
    y_true_squeezed = y_true.squeeze(-1) if y_true.ndim > 1 else y_true

    if uncertainty_method == UncertaintyMethod.PROBABILISTIC:
        mean = predictions[:, 0]
        log_var = predictions[:, 1]
        stacked = torch.stack((mean, log_var), dim=1)
        if loss_type == "beta_nll":
            regression_loss = beta_nll_loss(stacked, y_true_squeezed, beta=beta, reduction=reduction)
        else:
            regression_loss = nll_loss(stacked, y_true_squeezed, reduction=reduction)
    else:
        regression_loss = nn.MSELoss(reduction=reduction)(predictions.squeeze(-1) if predictions.ndim > 1 else predictions, y_true_squeezed)

    total_loss = regression_loss

    if include_boundary_loss:
        if class_boundaries is None or class_value_ranges is None:
            raise ValueError("class_boundaries and class_value_ranges must be provided for boundary regularization.")

        _, y_binned = create_bins(data=y_true_squeezed.cpu().numpy(), unique_bin_edges=class_boundaries)
        y_binned_tensor = torch.tensor(y_binned, dtype=torch.long, device=device)

        sample_boundaries = class_value_ranges[y_binned_tensor]
        boundary_loss = boundary_regularization_loss(predictions, sample_boundaries)
        total_loss += boundary_loss_weight * boundary_loss

    return total_loss
