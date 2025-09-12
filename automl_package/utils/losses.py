"""Loss functions."""

import math
from typing import Any

import numpy as np
import torch


def nll_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Negative Log-Likelihood loss for a Gaussian distribution."""
    mean = outputs[:, 0]
    log_var = outputs[:, 1]
    variance = torch.exp(log_var)
    targets = targets.squeeze(-1) if targets.ndim > 1 else targets

    per_sample_nll = 0.5 * (math.log(2 * math.pi) + log_var + (targets - mean) ** 2 / variance)
    return torch.mean(per_sample_nll)


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
