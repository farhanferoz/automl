"""Loss functions."""

import math
from typing import Literal, TypeVar

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

ArrayT = TypeVar("ArrayT", np.ndarray, torch.Tensor)


def _reduce(per_sample: torch.Tensor, reduction: Literal["mean", "none"]) -> torch.Tensor:
    """Averages per-sample losses (`"mean"`) or returns them as they are (`"none"`)."""
    return per_sample if reduction == "none" else per_sample.mean()


def nll_loss(outputs: torch.Tensor, targets: torch.Tensor, reduction: Literal["mean", "none"] = "mean") -> torch.Tensor:
    """Negative Log-Likelihood loss for a Gaussian distribution. `reduction="none"` returns the `(N,)` per-sample values."""
    mean = outputs[:, 0]
    log_var = outputs[:, 1]
    variance = torch.exp(log_var)
    targets = targets.squeeze(-1) if targets.ndim > 1 else targets

    per_sample_nll = 0.5 * (math.log(2 * math.pi) + log_var + ((targets - mean) ** 2) / variance)
    return _reduce(per_sample_nll, reduction)


def beta_nll_loss(outputs: torch.Tensor, targets: torch.Tensor, beta: float = 0.5, reduction: Literal["mean", "none"] = "mean") -> torch.Tensor:
    """β-NLL loss (Seitzer et al., 2022) that stabilizes variance learning.

    Weights the NLL by detached variance^β, preventing the model from trivially
    increasing variance to reduce loss (variance collapse).

    Args:
        outputs: Shape (N, 2) with columns [mean, log_variance].
        targets: Shape (N,) or (N, 1).
        beta: Weighting exponent. 0.0 = standard NLL, 1.0 = full reweighting.
        reduction: `"mean"` (default) or `"none"` for the `(N,)` per-sample weighted values.
    """
    mean = outputs[:, 0]
    log_var = outputs[:, 1]
    variance = torch.exp(log_var)
    targets = targets.squeeze(-1) if targets.ndim > 1 else targets

    per_sample_nll = 0.5 * (math.log(2 * math.pi) + log_var + ((targets - mean) ** 2) / variance)
    weighted_nll = (variance.detach() ** beta) * per_sample_nll
    return _reduce(weighted_nll, reduction)


def poisson_deviance(log_count: ArrayT, linear_predictor: ArrayT) -> ArrayT:
    """Poisson deviance per entry, `2 * (u * log(u / mu) - (u - mu))` with `u = exp(log_count)` and `mu = exp(linear_predictor)`.

    One formula for numpy arrays and torch tensors (same operators, the matching `exp`), so a model's training loss and its
    numpy held-out score cannot drift apart. Zero when `mu == u`, never negative for `u > 0`. The count is given on the
    log scale: for `u >= 1` ("1 + attendances") the argument is `log1p(y)`, plus any per-row offset the caller adds to both arguments.

    Args:
        log_count: log of the observed count `u`, any shape.
        linear_predictor: log of the modelled mean `mu`, same shape.

    Returns:
        The per-entry deviance, same shape and type as the inputs.
    """
    exp = torch.exp if isinstance(linear_predictor, torch.Tensor) else np.exp
    count = exp(log_count)
    return 2.0 * (count * (log_count - linear_predictor) - count + exp(linear_predictor))


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
    # Fisher information for log_var (constant 0.5, always positive, more stable than observed Hessian)
    hess_log_var = np.full_like(y_true, 0.5)

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
