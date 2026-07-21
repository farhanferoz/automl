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

    per_sample_nll = 0.5 * (math.log(2 * math.pi) + log_var + ((targets - mean) ** 2) / variance)
    return torch.mean(per_sample_nll)


def beta_nll_loss(outputs: torch.Tensor, targets: torch.Tensor, beta: float = 0.5) -> torch.Tensor:
    """β-NLL loss (Seitzer et al., 2022) that stabilizes variance learning.

    Weights the NLL by detached variance^β, preventing the model from trivially
    increasing variance to reduce loss (variance collapse).

    Args:
        outputs: Shape (N, 2) with columns [mean, log_variance].
        targets: Shape (N,) or (N, 1).
        beta: Weighting exponent. 0.0 = standard NLL, 1.0 = full reweighting.
    """
    mean = outputs[:, 0]
    log_var = outputs[:, 1]
    variance = torch.exp(log_var)
    targets = targets.squeeze(-1) if targets.ndim > 1 else targets

    per_sample_nll = 0.5 * (math.log(2 * math.pi) + log_var + ((targets - mean) ** 2) / variance)
    weighted_nll = (variance.detach() ** beta) * per_sample_nll
    return weighted_nll.mean()


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


def mdn_nll(
    y: torch.Tensor,
    probs: torch.Tensor,
    mus: torch.Tensor,
    log_vars: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Bishop 1994 Mixture Density Network negative log-likelihood.

    L = -mean_i log Σ_j p_j · N(y_i; μ_j, σ_j²)

    Uses log-sum-exp for numerical stability.

    Args:
        y: [batch] or [batch, 1] target values.
        probs: [batch, k] softmax mixture weights.
        mus: [batch, k] per-component means.
        log_vars: [batch, k] per-component log-variances.
        eps: floor for probs before log to prevent -inf.
    """
    y = y.view(-1, 1)
    log_component = -0.5 * (math.log(2 * math.pi) + log_vars + (y - mus) ** 2 * torch.exp(-log_vars))
    log_weights = torch.log(probs.clamp_min(eps))
    log_mixture = torch.logsumexp(log_weights + log_component, dim=-1)
    return -log_mixture.mean()


def fixed_sigma_mixture_log_likelihood(
    y: torch.Tensor,
    probs: torch.Tensor,
    mus: torch.Tensor,
    sigma: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-sample fixed-sigma mixture log-likelihood: log Σ_c p_c(x) N(y; μ_c(x), σ²).

    The capacity-programme's k-selection readout (MASTER Decision 24,
    `docs/plans/capacity_programme/MASTER.md`): σ is ONE shared constant across every component,
    every input and every arm, so nothing is fitted here -- this is not a variance metric.
    **HIGHER IS BETTER** (log-likelihood, not NLL) -- `mdn_nll` above returns the *negative* mean
    of a closely related quantity; mixing the two signs up is a bug waiting to happen, hence the
    distinct name and the explicit direction here.

    `sigma` is **REQUIRED and has NO DEFAULT**: a default is exactly how a per-arm σ would
    silently reappear, which Decision 24 forbids -- arms scored at different σ are not comparable.

    Args:
        y: [batch] or [batch, 1] target values.
        probs: [batch, k] per-component mixture weights (e.g. classifier posteriors, possibly
            renormalized over a rung's surviving classes); need not be pre-softmaxed, but must be
            non-negative and sum to ~1 per row.
        mus: [batch, k] per-component means.
        sigma: the ONE shared standard deviation, fixed before scoring
            (`docs/probreg_benchmark/benchmark_spec.md` §2): the known construction value on toys,
            the held-out RMSE of the plain-NN baseline on real data. Same value for every arm and
            every component -- never fitted, never per-arm.
        eps: floor for `probs` before `log`, to prevent `-inf`.

    Returns:
        [batch] per-sample log Σ_c p_c(x) N(y; μ_c(x), σ²), in nats. Higher is better.
    """
    y = y.view(-1, 1)
    log_var = 2.0 * math.log(sigma)
    log_component = -0.5 * (math.log(2 * math.pi) + log_var + (y - mus) ** 2 / (sigma * sigma))
    log_weights = torch.log(probs.clamp_min(eps))
    return torch.logsumexp(log_weights + log_component, dim=-1)


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
