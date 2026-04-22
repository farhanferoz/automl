"""Ordering-constraint identifiability penalty for ProbReg SEP_HEADS.

See ``docs/probreg_identifiability_research.md`` §3.3 for full derivation.

Given classifier logits and per-head means, for each class i we
    1. Take the top ``top_decile_fraction`` of training samples by p_i(x).
    2. Compute a probability-weighted mean head output:
           M_i = sum_{x in S_i} p_i(x) * h_i(p_i(x)) / sum_{x in S_i} p_i(x)
    3. Apply a squared-hinge penalty on ordering:
           L = sum_{i=1..k-1} max(0, M_{i-1} - M_i + delta)^2

The penalty is zero when M_0 < M_1 < ... < M_{k-1} with separation at
least ``margin``. Breaks the S_k permutation symmetry of the Gaussian-LTV
loss with the minimal k-1 inequalities.
"""

from __future__ import annotations

import math

import torch


def compute_ordering_means(
    classifier_logits: torch.Tensor,
    head_means: torch.Tensor,
    top_decile_fraction: float = 0.1,
) -> torch.Tensor:
    """Return the probability-weighted head mean over each head's top-p subset.

    Args:
        classifier_logits: (B, k) raw logits; softmax gives class probabilities.
        head_means: (B, k) per-sample-per-head mean outputs, i.e. the mu_i that
            enter the LTV combination.
        top_decile_fraction: fraction of samples to include in the top-p
            subset per head. Default 0.1 = top decile. At least 1 sample
            is always retained per head to avoid empty-subset NaNs.

    Returns:
        (k,) tensor of M_i values.
    """
    batch_size, k = classifier_logits.shape
    probs = torch.softmax(classifier_logits, dim=-1)  # (B, k)

    subset_size = max(1, math.ceil(batch_size * top_decile_fraction))

    # For each head i, find indices of the top subset_size samples by p_i.
    # topk over dim=0 picks the B highest along batch for each column.
    _, top_idx = probs.topk(subset_size, dim=0)  # (subset_size, k)

    # Gather the corresponding p_i and h_i values.
    # top_idx[:, i] are the row indices for column i.
    p_top = probs.gather(0, top_idx)  # (subset_size, k)
    h_top = head_means.gather(0, top_idx)  # (subset_size, k)

    numerator = (p_top * h_top).sum(dim=0)  # (k,)
    denominator = p_top.sum(dim=0).clamp_min(1e-8)  # avoid /0
    return numerator / denominator


def ordering_penalty(means: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """Squared-hinge penalty on ``means`` being monotonically increasing.

    For k = 1 (no adjacent pairs), returns 0.

    Args:
        means: (k,) tensor of M_i values.
        margin: required separation M_i >= M_{i-1} + margin. Set > 0 to
            push heads further apart.

    Returns:
        Scalar tensor.
    """
    if means.numel() < 2:
        return torch.zeros((), dtype=means.dtype, device=means.device)
    diffs = means[:-1] - means[1:] + margin  # should all be <= 0 when ordered
    hinge = torch.clamp(diffs, min=0.0)
    return (hinge ** 2).sum()


def ordering_loss(
    classifier_logits: torch.Tensor,
    head_means: torch.Tensor,
    top_decile_fraction: float = 0.1,
    margin: float = 0.0,
) -> torch.Tensor:
    """End-to-end: subset → weighted means → hinge penalty."""
    means = compute_ordering_means(classifier_logits, head_means, top_decile_fraction)
    return ordering_penalty(means, margin=margin)
