"""Structural bound on a network's regression output: the training targets' range, applied inside the network.

Searched before writing: `models/common/` and `models/mappers/` hold boundary PENALTIES (`BoundaryLossMixin`, `boundary_regularization_loss`)
that push predictions towards class bounds during training; nothing confines the output of the network itself, and no module wrapped a
final activation. This one does it with a smooth, differentiable squash, so training and prediction see the same bounded function and a
runaway input cannot produce a runaway output (no post-hoc clipping).
"""

import numpy as np
import torch
import torch.nn as nn


class TrainingRangeBound(nn.Module):
    """`out = c + h * tanh((raw - c) / h)` per output column, with `c` the midpoint and `h` the half-width of the training targets.

    The output lies strictly inside `(lo, hi)`, equals `raw` to first order near the midpoint, and saturates smoothly at the ends.
    The midpoint and half-width are buffers: saved with the state dict and applied in training and in prediction alike.
    """

    centre: torch.Tensor
    half_width: torch.Tensor

    def __init__(self, low: np.ndarray, high: np.ndarray) -> None:
        """Builds the bound from per-column `low` and `high` (same shape `(n_outputs,)`)."""
        super().__init__()
        low, high = np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)
        self.register_buffer("centre", torch.from_numpy(((low + high) / 2).astype(np.float32)))
        self.register_buffer("half_width", torch.from_numpy(((high - low) / 2).astype(np.float32)))

    @classmethod
    def from_targets(cls, y: np.ndarray) -> "TrainingRangeBound":
        """The bound for targets `y` of shape `(N, n_outputs)`, NaN entries ignored.

        Raises:
            ValueError: when a column has no observed value or all its observed values are equal (zero-width range), naming the column.
        """
        y = np.asarray(y, dtype=np.float64)
        y = y.reshape(len(y), -1)
        observed = ~np.isnan(y)
        empty = np.flatnonzero(~observed.any(axis=0))
        if empty.size:
            raise ValueError(f"output_bound=TRAINING_RANGE: output column(s) {empty.tolist()} have no observed training target.")
        low, high = np.nanmin(y, axis=0), np.nanmax(y, axis=0)
        flat = np.flatnonzero(high <= low)
        if flat.size:
            raise ValueError(f"output_bound=TRAINING_RANGE: output column(s) {flat.tolist()} have a zero-width training range (a single target value {low[flat].tolist()}).")
        return cls(low, high)

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        """Squashes `(B, n_outputs)` raw outputs into the training range."""
        return self.centre + self.half_width * torch.tanh((raw - self.centre) / self.half_width)
