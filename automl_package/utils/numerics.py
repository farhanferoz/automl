"""Numerical utility functions."""

import re

import numpy as np
import torch
import torch.nn as nn


def create_bins(
    data: np.ndarray,
    n_bins: int | None = None,
    unique_bin_edges: np.ndarray | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Creates bins for a given dataset and returns the bin edges and the bin index for each data point.

    It handles duplicate bin edges by reducing the number of bins until all bin edges are unique.

    Args:
        data (np.ndarray): The input data.
        n_bins (Optional int): The desired number of bins.
        unique_bin_edges (Optional np.ndarray): If provided, these bin edges will be used instead of creating new ones.
        min_value: (Optional float): If provided then the lower value of first bin
        max_value: (Optional float): If provided then the upper value of last bin

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the bin edges and the bin index for each data point.
    """
    if unique_bin_edges is None:
        assert n_bins is not None
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(data, percentiles)

        # Ensure bin edges are unique
        unique_bin_edges = np.unique(bin_edges)
        while len(unique_bin_edges) < len(bin_edges) and n_bins > 1:
            n_bins -= 1
            percentiles = np.linspace(0, 100, n_bins + 1)
            bin_edges = np.percentile(data, percentiles)
            unique_bin_edges = np.unique(bin_edges)
        if min_value is not None:
            assert min_value <= unique_bin_edges[0]
            unique_bin_edges[0] = min_value
        if max_value is not None:
            assert max_value >= unique_bin_edges[-1]
            unique_bin_edges[-1] = max_value
    else:
        if n_bins is not None:
            assert len(unique_bin_edges) == n_bins + 1

    bin_indices = np.searchsorted(unique_bin_edges[1:], data, side="left")
    bin_indices = np.clip(bin_indices, 0, len(unique_bin_edges) - 1)

    return unique_bin_edges, bin_indices


def aggregate_stats(
    model: nn.Sequential,
    include_bias: bool = True,
    exclude_names_pattern: str | None = None,
) -> tuple[int, float, float]:
    """Return (d, sum|w|, sum w²) across selected parameters."""
    d = 0  # total #elements
    l1_sum = 0.0
    l2_sum = 0.0
    for name, p in model.named_parameters():
        if exclude_names_pattern and re.search(exclude_names_pattern, name):
            continue
        if p.requires_grad and (include_bias or "bias" not in name):
            flat = p.view(-1)
            d += flat.numel()
            l1_sum = l1_sum + flat.abs().sum()
            l2_sum = l2_sum + (flat**2).sum()
    return d, l1_sum, l2_sum


def log_erfc(x: torch.Tensor) -> torch.Tensor:
    """Numerically-stable log(erfc(x))."""
    x64 = x.to(dtype=torch.float64)
    mask = x64 > 5.0
    out = torch.empty_like(x64)

    out[~mask] = torch.log(torch.special.erfc(x64[~mask]))
    out[mask] = torch.log(torch.special.erfcx(x64[mask])) - x64[mask] ** 2
    return out.to(dtype=x.dtype)


def find_optimal_iterations(fold_results: list[dict]) -> int:
    """Finds the optimal number of iterations from a list of fold results."""
    best_iter_label = "best_iter"
    loss_history_label = "loss_history"

    max_best_iter = int(np.max([res[best_iter_label] for res in fold_results]))
    min_best_iter = int(np.min([res[best_iter_label] for res in fold_results]))
    if max_best_iter == min_best_iter:
        optimal_iterations = max_best_iter
    else:
        max_valid_len = min(len(res[loss_history_label]) for res in fold_results if res[loss_history_label])
        optimal_iterations = 0
        if (max_valid_len == 0) or (max_best_iter > max_valid_len):
            optimal_iterations = int(np.mean([res[best_iter_label] for res in fold_results]))
        if (max_valid_len > 0) and (optimal_iterations < max_valid_len):
            avg_loss_curve = np.full(max_valid_len, np.nan)
            for i in range(max_valid_len):
                epoch_losses = [res[loss_history_label][i] for res in fold_results if i < len(res[loss_history_label])]
                if epoch_losses:
                    avg_loss_curve[i] = np.mean(epoch_losses)
            optimal_iterations = np.nanargmin(avg_loss_curve) + 1

    return optimal_iterations
