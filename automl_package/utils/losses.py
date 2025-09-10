"""Loss functions."""

import math
import torch


def nll_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Negative Log-Likelihood loss for a Gaussian distribution."""
    mean = outputs[:, 0]
    log_var = outputs[:, 1]
    targets = targets.squeeze(-1) if targets.ndim > 1 else targets
    per_sample_nll = 0.5 * (math.log(2 * math.pi) + log_var + (targets - mean) ** 2 / torch.exp(log_var))
    return torch.mean(per_sample_nll)
