"""Target transforms for regression models."""

import torch


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric log transform: sign(x) * log(1 + |x|).

    Compresses targets spanning multiple orders of magnitude while preserving sign.
    """
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog: sign(x) * (exp(|x|) - 1)."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
