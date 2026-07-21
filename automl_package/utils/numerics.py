"""Numerical utility functions."""

import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from automl_package.logger import logger


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
        requested_n_bins = n_bins
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(data, percentiles)

        # Ensure bin edges are unique
        unique_bin_edges = np.unique(bin_edges)
        while len(unique_bin_edges) < len(bin_edges) and n_bins > 1:
            n_bins -= 1
            percentiles = np.linspace(0, 100, n_bins + 1)
            bin_edges = np.percentile(data, percentiles)
            unique_bin_edges = np.unique(bin_edges)
        if n_bins < requested_n_bins:
            logger.warning(
                "create_bins: requested n_bins=%d produced tied percentile edges on the target "
                "distribution; shrunk to effective n_bins=%d (%d unique edge(s) dropped). This "
                "happens on discrete/tied targets where percentile boundaries coincide.",
                requested_n_bins,
                n_bins,
                requested_n_bins - n_bins,
            )
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


def ensure_proba_shape(y_proba: np.ndarray, n_classes: int) -> np.ndarray:
    """Ensures that the probability array has the correct shape, especially for binary classification."""
    if y_proba is None:
        return None

    # Case 1: Binary classification with a single output column (e.g., from sigmoid)
    if n_classes == 2 and y_proba.ndim == 2 and y_proba.shape[1] == 1:
        return np.hstack((1 - y_proba, y_proba))

    # Case 2: Binary classification with a 1D array of probabilities for the positive class
    if n_classes == 2 and y_proba.ndim == 1:
        return np.vstack((1 - y_proba, y_proba)).T

    # Case 3: Mismatch between columns and n_classes (error condition)
    if y_proba.shape[1] != n_classes:
        raise ValueError(f"Classifier predict_proba output shape {y_proba.shape} does not match n_classes {n_classes}.")

    # Case 4: Shape is already correct
    return y_proba


def calculate_binned_stats(
    probas: np.ndarray,
    values: np.ndarray,
    n_bins: int,
    aggregations: dict[str, callable],
    min_value: float | None = None,
    max_value: float | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Bins probabilities and calculates aggregate statistics for corresponding values.

    Args:
        probas (np.ndarray): The probabilities to bin.
        values (np.ndarray): The values to aggregate.
        n_bins (int): The number of bins.
        aggregations (dict[str, callable]): A dictionary where keys are stat names
                                            and values are aggregation functions (e.g., np.mean).
        min_value (Optional[float]): Minimum value for bin edges.
        max_value (Optional[float]): Maximum value for bin edges.

    Returns:
        tuple[np.ndarray, dict[str, np.ndarray]]: A tuple containing:
            - The bin edges used.
            - A dictionary of lookup tables (numpy arrays), one for each aggregation.
    """
    probas = probas.flatten()
    values = values.flatten()
    bin_edges, bin_indices = create_bins(data=probas, n_bins=n_bins, min_value=min_value, max_value=max_value)

    stats_lookups = {name: [] for name in aggregations}

    for i in range(len(bin_edges)):
        mask = bin_indices == i
        partition_values = values[mask]

        if len(partition_values) == 0:
            # Handle empty bins, e.g., by using all values as a fallback
            partition_values = values

        for name, func in aggregations.items():
            stat_value = func(partition_values)
            if np.isnan(stat_value):
                stat_value = 0.0  # Replace NaN with 0 or another suitable default
            stats_lookups[name].append(stat_value)

    # Convert lists to numpy arrays
    final_lookups = {name: np.array(lookup) for name, lookup in stats_lookups.items()}

    return bin_edges, final_lookups


def calculate_class_value_ranges(y_flat: np.ndarray, y_binned: np.ndarray, k: int, y_min: float, y_max: float, device: str) -> torch.Tensor:
    """Calculates the value ranges for each class based on the sophisticated bounding logic."""
    # Coerce to Python float — callers may pass numpy scalars which PyTorch 2.6+ rejects
    # when assigned to tensor elements via tensor[i, j] = value.
    y_min, y_max = float(y_min), float(y_max)
    df = pd.DataFrame({"y": y_flat, "bin": y_binned})
    agg = df.groupby("bin")["y"].agg(["min", "max"])
    per_class_ranges = torch.tensor(agg.values, dtype=torch.float32)

    # Apply the sophisticated bounding logic
    final_bounds = torch.zeros_like(per_class_ranges)
    mid_point = k / 2

    for i in range(k):
        class_min, class_max = per_class_ranges[i]
        if i < mid_point:  # Lower half
            final_bounds[i, 0] = class_min
            final_bounds[i, 1] = y_max
        else:  # Upper half
            final_bounds[i, 0] = y_min
            final_bounds[i, 1] = class_max

    # Special case for the middle class if k is odd
    if k % 2 != 0:
        middle_class_idx = int(mid_point)
        final_bounds[middle_class_idx, 0] = per_class_ranges[middle_class_idx, 0]
        final_bounds[middle_class_idx, 1] = per_class_ranges[middle_class_idx, 1]

    return final_bounds.to(device)


# ---------------------------------------------------------------------------
# Bootstrap standard error (capacity-programme Task FP-9.b) -- ONE shared helper covering the
# three shapes the capacity-programme example drivers use: plain (SE of a 1-D vector's mean),
# paired (SE of a paired-difference vector's mean -- mathematically the SAME operation as plain,
# applied to a difference vector, so `bootstrap_se` below covers both), and two-sample
# (`two_sample_bootstrap_se`). See `docs/plans/capacity_programme/shared/zero-caller-inventory.md`
# Part (ii) for the inventory this consolidates: 7 "plain" sites (`_boot_se` x3, `_plain_boot_se`
# x4) that are already identical thin wrappers around one shared primitive (a pure rename here),
# 3 "paired" sites (`_paired_bootstrap_se`, `paired_bootstrap_se`, `_paired_point_bootstrap_se`)
# that each independently reimplement the same resample loop (genuine deduplication), and exactly
# 1 "two-sample" site (`sinc_width_experiment.py::_two_sample_boot_se`).
#
# That inventory's shared low-level primitive, `automl_package/examples/_capacity_ladder.py
# ::_bootstrap_col_means`, cannot be imported here: package code under `utils/` never imports from
# `examples/` (capacity-programme boundary rule, `docs/plans/capacity_programme/flexnn-package.md`
# §2). `bootstrap_se` below reimplements that primitive's single-column algorithm exactly (row-wise
# i.i.d. resample-with-replacement, `n_boot` times, `std(ddof=1)` of the resampled means) rather
# than inventing a new one.
# ---------------------------------------------------------------------------


def bootstrap_se(values: np.ndarray, n_boot: int, seed: int) -> float:
    """Bootstrap standard error of a 1-D vector's mean (the "plain" and "paired" shapes).

    A paired-difference SE is the identical resampling operation applied to the (already computed)
    per-example difference vector, so this one function serves both shapes -- see module comment.

    Args:
        values: `(n,)` values -- raw observations for the "plain" shape, or a per-example paired
            difference for the "paired" shape.
        n_boot: number of bootstrap resamples. No default: a selection rule whose answer moves
            between runs is not a rule (FP-9.b).
        seed: RNG seed for the resample. No default, for the same reason.

    Returns:
        Standard deviation (`ddof=1`) of the `n_boot` resampled means. Exactly `0.0` for a constant
        vector (every resampled mean equals the same constant).
    """
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = arr.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = arr[idx].mean(axis=1)
    return float(boot_means.std(ddof=1))


def two_sample_bootstrap_se(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int) -> float:
    """Bootstrap standard error of `mean(a) - mean(b)` for two INDEPENDENT (unpaired) samples.

    Vectorized (a single `rng.integers(..., size=(n_boot, n))` draw per sample), unlike the sole
    prior site (`sinc_width_experiment.py::_two_sample_boot_se`, an explicit Python
    `for i in range(n_boot)` loop) -- this matches that site's SEMANTICS (the same
    resample-and-difference statistic), not its performance profile, per FP-9.b's instruction.

    Args:
        a: `(n_a,)` first independent sample.
        b: `(n_b,)` second independent sample.
        n_boot: number of bootstrap resamples. No default -- see `bootstrap_se`.
        seed: RNG seed for the resample. No default -- see `bootstrap_se`.

    Returns:
        Standard deviation (`ddof=1`) of the `n_boot` resampled `mean(a) - mean(b)` differences.
    """
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n_a, n_b = arr_a.shape[0], arr_b.shape[0]
    idx_a = rng.integers(0, n_a, size=(n_boot, n_a))
    idx_b = rng.integers(0, n_b, size=(n_boot, n_b))
    diffs = arr_a[idx_a].mean(axis=1) - arr_b[idx_b].mean(axis=1)
    return float(diffs.std(ddof=1))
