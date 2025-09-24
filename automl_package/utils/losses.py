"""Loss functions."""

import math

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812


def nll_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Negative Log-Likelihood loss for a Gaussian distribution."""
    mean = outputs[:, 0]
    log_var = outputs[:, 1]
    variance = torch.exp(log_var)
    targets = targets.squeeze(-1) if targets.ndim > 1 else targets

    per_sample_nll = 0.5 * (math.log(2 * math.pi) + log_var + (targets - mean) ** 2 / variance)
    return torch.mean(per_sample_nll)


def masked_cross_entropy_loss(logits: torch.Tensor, y_binned: torch.Tensor, k_values: torch.Tensor) -> torch.Tensor:
    """Calculates cross-entropy loss with masking for valid class logits."""
    max_k_in_batch = logits.shape[1]
    col_indices = torch.arange(max_k_in_batch, device=logits.device)
    mask = col_indices < k_values.unsqueeze(1)
    masked_logits = torch.where(mask, logits, float("-inf"))
    return F.cross_entropy(masked_logits, y_binned)


def tree_model_gaussian_nll_objective(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Custom Gaussian NLL objective for tree-based models (scikit-learn API).

    Args:
        y_true (np.ndarray): The true target values, shape (n_samples,).
        y_pred (np.ndarray): The predictions from the model, shape (n_samples, 2).

    Returns:
        A tuple of (gradient, hessian) arrays, both of shape (n_samples, 2).
    """
    # y_pred is shape (n_samples, 2), directly from the booster
    mean = y_pred[:, 0]
    log_var = y_pred[:, 1]
    variance = np.exp(log_var)

    # Gradient of NLL w.r.t. mean
    grad_mean = (mean - y_true) / variance
    # Gradient of NLL w.r.t. log_var
    grad_log_var = 0.5 * (1 - ((y_true - mean) ** 2) / variance)

    # Hessian of NLL w.r.t. mean
    hess_mean = 1.0 / variance
    # Hessian of NLL w.r.t. log_var
    hess_log_var = 0.5 * (((y_true - mean) ** 2) / variance)

    # Stack gradients and hessians to match y_pred's shape: (n_samples, 2)
    grad = np.stack([grad_mean, grad_log_var], axis=1)
    hess = np.stack([hess_mean, hess_log_var], axis=1)

    return grad, hess


def tree_model_gaussian_nll_eval_metric(y_true: np.ndarray, y_pred: np.ndarray) -> list[tuple[str, float, bool]]:
    """Custom evaluation metric for Gaussian NLL for tree-based models (scikit-learn API).

    Args:
        y_true (np.ndarray): The true target values, shape (n_samples,).
        y_pred (np.ndarray): The predictions from the model, shape (n_samples, 2).

    Returns:
        A list containing the metric name, value, and whether higher is better.
    """
    # y_pred is shape (n_samples, 2), directly from the booster
    mean = y_pred[:, 0]
    log_var = y_pred[:, 1]
    variance = np.exp(log_var)

    # Calculate NLL using the full formula, including the constant term
    nll = 0.5 * (np.log(2 * np.pi) + log_var + ((y_true - mean) ** 2) / variance)
    # The metric name, the result, and whether higher is better
    return [("nll", np.mean(nll), False)]


def boundary_regularization_loss(predictions: torch.Tensor, boundaries: torch.Tensor) -> torch.Tensor:
    """Calculates a penalty for predictions that fall outside the given boundaries.

    Args:
        predictions: The predictions from the model, shape (batch_size, 1) or (batch_size, 2).
        boundaries: A tensor of shape (batch_size, 2) where each row is [min_val, max_val].

    Returns:
        The mean boundary penalty loss.
    """
    # If predictions have more than one dimension (e.g., mean and variance), only use the mean for boundary loss
    if predictions.shape[1] > 1:
        predictions = predictions[:, 0]

    min_vals = boundaries[:, 0]
    max_vals = boundaries[:, 1]

    # Penalty for predictions below the minimum boundary
    lower_violation = torch.clamp(min_vals - predictions.squeeze(), min=0)
    # Penalty for predictions above the maximum boundary
    upper_violation = torch.clamp(predictions.squeeze() - max_vals, min=0)

    # The total violation is the sum of lower and upper violations (only one can be non-zero)
    total_violation = lower_violation + upper_violation

    # We use the squared violation as the penalty
    penalty = torch.square(total_violation)

    return torch.mean(penalty)
