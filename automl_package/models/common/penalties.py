"""Shared penalty calculation logic for regression models."""

import torch

from automl_package.enums import BoundaryRegularizationMethod
from automl_package.utils.numerics import create_bins


def apply_additional_penalties(
    total_loss: torch.Tensor,
    per_head_outputs: torch.Tensor,
    y_true_squeezed: torch.Tensor,
    model_instance: torch.nn.Module,
    include_boundary_loss: bool,
    class_boundaries: torch.Tensor,
    class_value_ranges: torch.Tensor,
    middle_class_dist_params: dict | None = None,
) -> torch.Tensor:
    """
    Applies middle class NLL penalty and boundary regularization loss.

    Args:
        total_loss: The current total loss.
        per_head_outputs: The outputs from the regression heads.
        y_true_squeezed: The squeezed true y values.
        model_instance: The model instance (used to access params and methods).
        include_boundary_loss: Whether to include the boundary loss.
        class_boundaries: The class boundaries.
        class_value_ranges: The class value ranges.
        middle_class_dist_params: The distribution parameters for the middle class.

    Returns:
        The updated total loss.
    """
    is_boundary_reg_penalty = model_instance.boundary_regularization_method == BoundaryRegularizationMethod.PENALTY
    should_apply_boundary_loss = is_boundary_reg_penalty and include_boundary_loss

    if not should_apply_boundary_loss and not model_instance.use_middle_class_nll_penalty:
        return total_loss

    _, y_binned_tensor = create_bins(data=y_true_squeezed.cpu().numpy(), unique_bin_edges=class_boundaries)
    y_binned_tensor = torch.tensor(y_binned_tensor, dtype=torch.long, device=model_instance.device)

    # Middle Class NLL Penalty
    if model_instance.use_middle_class_nll_penalty and middle_class_dist_params:
        total_loss += model_instance.calculate_middle_class_nll_penalty(
            heads=model_instance.regression_heads,
            per_head_outputs=per_head_outputs,
            y_binned_tensor=y_binned_tensor,
            middle_class_dist_params=middle_class_dist_params,
        )

    # Boundary Regularization Loss
    if should_apply_boundary_loss:
        total_loss += model_instance.calculate_boundary_loss(
            heads=model_instance.regression_heads,
            per_head_outputs=per_head_outputs,
            y_true=y_true_squeezed,
            y_binned_tensor=y_binned_tensor,
            class_value_ranges=class_value_ranges,
            boundary_loss_weight=model_instance.boundary_loss_weight,
        )

    return total_loss
