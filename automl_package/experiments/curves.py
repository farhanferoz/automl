"""Probability curves and head scores of a fitted ProbReg cell (design section 3, step 6), as arrays, an NPZ and a PNG.

Reused: ``get_classifier_predictions`` and ``precomputed_class_boundaries`` of ``ProbabilisticRegressionModel`` for the
class probabilities and bin edges, ``evaluate_mapper_heads`` (``utils/plotting.py``, factored out of
``plot_nn_probability_mappers``) for the head outputs on a probability grid, ``regression_module.forward_per_class`` for
the head outputs at the model's own probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import ndtr
from scipy.stats import norm

from automl_package.enums import RegressionStrategy
from automl_package.experiments.registry import AutomlPredictor
from automl_package.experiments.toys import Toy
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel
from automl_package.utils.plotting import evaluate_mapper_heads

N_CURVE_POINTS = 400
MIN_BIN_PROBABILITY = 1e-12  # below this the truncated-normal mean is 0/0 and is left undefined


@dataclass
class HeadArrays:
    """Class probabilities and head outputs (original y units) at some inputs, plus the bin edges."""

    p: np.ndarray  # (n, k)
    mu: np.ndarray | None  # (n, k) head c's output at the model's probabilities; None for the single-head-final-output layout
    prediction: np.ndarray  # (n,)
    edges: np.ndarray  # (k + 1,) original y units


def probreg_inner(predictor: AutomlPredictor) -> ProbabilisticRegressionModel | None:
    """The fitted fixed-k ProbReg model behind ``predictor``, or ``None`` if it is not one."""
    inner = predictor.inner
    if not isinstance(inner, ProbabilisticRegressionModel) or getattr(inner, "direct_regression", False) or inner.model is None:
        return None
    return inner


def head_arrays(predictor: AutomlPredictor, inner: ProbabilisticRegressionModel, x: np.ndarray) -> HeadArrays:
    """Probabilities, head outputs and prediction of the fitted model at the original-unit inputs ``x``."""
    x_model = predictor.scaler.x(x)
    k = int(inner.n_classes)
    p = inner.get_classifier_predictions(x_model, np.zeros(len(x)))[1]
    mu = None
    if inner.regression_strategy != RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT:
        with torch.no_grad():
            heads = inner.model.regression_module.forward_per_class(torch.tensor(p, dtype=torch.float32, device=inner.device))
        mu = predictor.scaler.y_back(heads[:, :, 0].cpu().numpy())
    edges = predictor.scaler.y_back(inner.precomputed_class_boundaries[k])
    return HeadArrays(p=p, mu=mu, prediction=predictor.predict(x), edges=edges)


def head_scores(arrays: HeadArrays) -> dict[str, float | int] | None:
    """Card 5c-4 part 1, no threshold: out of order and out of bin, each averaged over the inputs.

    out of order = sum_c p_c p_{c+1} 1[mu_c > mu_{c+1}] / sum_c p_c p_{c+1};  out of bin = sum_c p_c 1[mu_c outside bin c].
    """
    if arrays.mu is None:
        return None
    p, mu, edges = arrays.p, arrays.mu, arrays.edges
    pair = p[:, :-1] * p[:, 1:]
    weight = pair.sum(axis=1)
    valid = weight > 0
    inverted = (pair * (mu[:, :-1] > mu[:, 1:])).sum(axis=1)
    outside = (mu < edges[:-1]) | (mu > edges[1:])
    k = p.shape[1]
    return {
        "k": k,
        "k_parity": k % 2,
        "out_of_order": float(np.mean(inverted[valid] / weight[valid])) if valid.any() else float("nan"),
        "out_of_bin": float(np.mean((p * outside).sum(axis=1))),
    }


def true_bin_arrays(toy: Toy, x: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """True ``P(y in bin c | x)`` and ``E[y | y in bin c, x]`` (truncated normal: every study toy's noise is Gaussian given x), each ``(n, k)``."""
    sd = toy.noise_sd(x)
    mean = toy.true_mean(x)[:, None]
    full = np.concatenate([[-np.inf], edges[1:-1], [np.inf]])
    a, b = (full[:-1] - mean) / sd[:, None], (full[1:] - mean) / sd[:, None]
    prob = ndtr(b) - ndtr(a)
    with np.errstate(divide="ignore", invalid="ignore"):
        cond = np.where(prob > MIN_BIN_PROBABILITY, mean + sd[:, None] * (norm.pdf(a) - norm.pdf(b)) / prob, np.nan)
    return prob, cond


def curve_arrays(predictor: AutomlPredictor, inner: ProbabilisticRegressionModel, toy: Toy) -> dict[str, np.ndarray]:
    """Everything the figure shows, on ``N_CURVE_POINTS`` inputs across the toy's range."""
    low, high = toy.x_range
    x = np.linspace(low, high, N_CURVE_POINTS).reshape(-1, 1)
    arrays = head_arrays(predictor, inner, x)
    out = {"x": x.ravel(), "p_model": arrays.p, "prediction": arrays.prediction, "true_mean": toy.true_mean(x), "edges": arrays.edges}
    if arrays.mu is not None:
        out["mu_model"] = arrays.mu
        p_grid, grid_outputs = evaluate_mapper_heads(inner.model.regression_module, inner.regression_strategy, int(inner.n_classes), inner.device)
        out["head_p_grid"], out["head_curve"] = p_grid, predictor.scaler.y_back(grid_outputs)
    out["p_true"], out["mu_true"] = true_bin_arrays(toy, x, arrays.edges)
    return out


def plot_curves(arrays: dict[str, np.ndarray], path: Path, title: str) -> None:
    """Figure: (a) class probabilities against x, model solid and true dashed; (b) one panel per head, or the prediction for a single head."""
    k = arrays["p_model"].shape[1]
    colours = plt.cm.tab10(np.arange(k) % 10)
    per_head = "mu_model" in arrays
    n_cols = min(k, 5) if per_head else 1
    n_rows = 1 + (int(np.ceil(k / n_cols)) if per_head else 1)
    fig = plt.figure(figsize=(3.2 * max(n_cols, 3), 3.0 * n_rows))
    grid = fig.add_gridspec(n_rows, n_cols)
    ax = fig.add_subplot(grid[0, :])
    for c in range(k):
        ax.plot(arrays["x"], arrays["p_model"][:, c], color=colours[c], label=f"class {c}")
        if "p_true" in arrays:
            ax.plot(arrays["x"], arrays["p_true"][:, c], color=colours[c], linestyle="--")
    ax.set_xlabel("x")
    ax.set_ylabel("probability (solid: model, dashed: true)")
    ax.legend(ncol=min(k, 10), fontsize=7)
    if per_head:
        for c in range(k):
            axc = fig.add_subplot(grid[1 + c // n_cols, c % n_cols])
            axc.axhspan(arrays["edges"][c], arrays["edges"][c + 1], color=colours[c], alpha=0.12)
            axc.plot(arrays["head_p_grid"], arrays["head_curve"][:, c], color=colours[c], label="head on a probability grid")
            axc.scatter(arrays["p_model"][:, c], arrays["mu_model"][:, c], s=3, color="k", label="model, over x")
            if "p_true" in arrays:
                axc.plot(arrays["p_true"][:, c], arrays["mu_true"][:, c], color="r", linestyle="--", label="true E[y | bin, x]")
            axc.set_title(f"head {c}", fontsize=8)
            axc.set_xlabel("probability of class")
            axc.set_ylabel("y")
        fig.axes[1].legend(fontsize=6)
    else:
        axp = fig.add_subplot(grid[1, :])
        axp.plot(arrays["x"], arrays["prediction"], label="model")
        axp.plot(arrays["x"], arrays["true_mean"], linestyle="--", label="true mean")
        axp.set_xlabel("x")
        axp.set_ylabel("y")
        axp.legend()
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_curves(predictor: AutomlPredictor, inner: ProbabilisticRegressionModel, toy: Toy, stem: Path, title: str) -> None:
    """Writes ``<stem>.npz`` and ``<stem>.png``."""
    arrays = curve_arrays(predictor, inner, toy)
    np.savez_compressed(stem.with_suffix(".npz"), **arrays)
    plot_curves(arrays, stem.with_suffix(".png"), title)
