"""I2: Diagnostic checks for ProbReg regression-head structure.

Expected structure on well-learned configs:
- n=2: head_0 and head_1 are mirror images along the probability axis (one
  monotonically increasing, one monotonically decreasing in p).
- n=3: middle head (index 1) is ~flat; outer heads mirror.
- n>=4: outer heads follow a similar mirror pattern near the extremes.

A "good" fit should satisfy: the two outermost heads have opposite monotonicity
with respect to their own probability, and their means differ by at least a
margin. Configs violating this likely did not learn the intended representation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from automl_package.enums import RegressionStrategy
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel


@dataclass
class HeadStructureReport:
    n_classes: int
    strategy: str
    head_slopes: np.ndarray  # (n_classes,) sign of slope: +1 / -1 / 0
    head_output_range: np.ndarray  # (n_classes,) max-min per head
    mirror_ok: bool  # True if outer heads have opposite monotonicity
    middle_flat_ok: bool | None  # True if middle head flat (only applicable for odd n>=3)
    mean_sep_ok: bool  # True if outer head means differ by > 0.5 * total y-range observed

    def as_dict(self) -> dict:
        return {
            "n_classes": self.n_classes,
            "strategy": self.strategy,
            "head_slopes": self.head_slopes.tolist(),
            "head_output_range": self.head_output_range.tolist(),
            "mirror_ok": bool(self.mirror_ok),
            "middle_flat_ok": None if self.middle_flat_ok is None else bool(self.middle_flat_ok),
            "mean_sep_ok": bool(self.mean_sep_ok),
            "passed": bool(self.mirror_ok and self.mean_sep_ok and (self.middle_flat_ok is not False)),
        }


def _head_outputs_vs_p(model: ProbabilisticRegressionModel, n_points: int = 40) -> np.ndarray:
    """Returns (n_classes, n_points, output_size) — head i's output when only p_i varies.

    For each class i, construct a synthetic probability vector [0, 0, ..., p, ..., 0, (1-p)]
    (probability p on class i, rest on class 0 or the other outer class) and query the
    regression module. The slope of head i's own output (column 0) over p shows whether
    it is increasing or decreasing in its own assignment probability.
    """
    n_classes = model.n_classes
    probs = np.zeros((n_classes, n_points, n_classes), dtype=np.float32)
    p_grid = np.linspace(0.01, 0.99, n_points, dtype=np.float32)
    for i in range(n_classes):
        other = 0 if i != 0 else n_classes - 1
        for j in range(n_points):
            probs[i, j, i] = p_grid[j]
            probs[i, j, other] = 1.0 - p_grid[j]

    if model.regression_strategy not in (RegressionStrategy.SEPARATE_HEADS, RegressionStrategy.SINGLE_HEAD_N_OUTPUTS):
        raise NotImplementedError(f"Diagnostic not supported for {model.regression_strategy}")
    p_tensor = torch.tensor(probs.reshape(-1, n_classes), dtype=torch.float32).to(model.device)
    model.model.eval()
    with torch.no_grad():
        _, per_head = model.model.regression_module(p_tensor, return_head_outputs=True)
    per_head = per_head.cpu().numpy().reshape(n_classes, n_points, n_classes, -1)
    # For class-i sweep, keep head i's output over p.
    own_output = np.stack([per_head[i, :, i, :] for i in range(n_classes)], axis=0)
    return own_output, p_grid  # shape (n_classes, n_points, output_size)


def analyse_head_structure(
    model: ProbabilisticRegressionModel,
    y_scale: float | None = None,
    slope_tol: float = 1e-3,
    flat_tol_frac: float = 0.05,
) -> HeadStructureReport:
    """Check whether a fitted ProbReg has the expected head structure.

    Args:
        model: fitted ProbabilisticRegressionModel.
        y_scale: (max - min) of the training targets. Used to set thresholds; if
            None, inferred from the outputs.
        slope_tol: absolute slope below which a head counts as "flat".
        flat_tol_frac: for the middle head, |range| <= flat_tol_frac * y_scale
            counts as flat.
    """
    own_output, p_grid = _head_outputs_vs_p(model)
    n_classes = own_output.shape[0]
    means = own_output[..., 0]  # (n_classes, n_points)

    # Slope of each head's mean over its own probability (monotonicity).
    slopes = np.polyfit(p_grid, means.T, deg=1)[0]  # shape (n_classes,)
    slope_signs = np.where(np.abs(slopes) < slope_tol, 0, np.sign(slopes))
    ranges = means.max(axis=1) - means.min(axis=1)

    if y_scale is None:
        y_scale = float(ranges.max()) if ranges.max() > 0 else 1.0

    # Outer heads: 0 and n-1. Expect opposite slope signs (mirror pattern).
    mirror_ok = bool(slope_signs[0] != 0 and slope_signs[-1] != 0 and slope_signs[0] != slope_signs[-1])

    # Middle-flat check only applies for odd n>=3.
    middle_flat_ok: bool | None = None
    if n_classes >= 3 and n_classes % 2 == 1:
        mid = n_classes // 2
        middle_flat_ok = bool(ranges[mid] <= flat_tol_frac * y_scale)

    # Outer means should be well-separated (range gives max span of each head).
    outer_mean_sep = abs(means[-1].mean() - means[0].mean())
    mean_sep_ok = bool(outer_mean_sep > 0.3 * y_scale)

    return HeadStructureReport(
        n_classes=n_classes,
        strategy=model.regression_strategy.value,
        head_slopes=slope_signs,
        head_output_range=ranges,
        mirror_ok=mirror_ok,
        middle_flat_ok=middle_flat_ok,
        mean_sep_ok=mean_sep_ok,
    )
