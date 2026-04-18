"""I3: FlexNN per-input selected depth visualization + depth-vs-complexity correlation.

On the piecewise dataset (and a new tunable-complexity synthetic), we train a
FlexibleHiddenLayersNN with SOFT_GATING and plot:
  1. Selected depth (argmax n_probs) vs input x.
  2. Soft depth probabilities as a heatmap (depth on y-axis, input on x-axis).
  3. Depth correlation with |distance-to-regime-boundary| — quantified via
     Spearman correlation and saved to a CSV.

Also produces the headline figure for Paper B: per-input depth heatmap
alongside the target function, showing shallow depth for simple regions and
deep depth for complex regions.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from automl_package.enums import (
    DepthRegularization,
    LayerSelectionMethod,
    UncertaintyMethod,
)
from automl_package.models.flexible_neural_network import FlexibleHiddenLayersNN

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

OUT_DIR = Path(__file__).parent / "flex_nn_depth_viz_results"
OUT_DIR.mkdir(exist_ok=True)


def piecewise_dataset(n: int = 1200, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-5.0, 5.0, n).reshape(-1, 1).astype(np.float32)
    regime = (x.ravel() >= 0).astype(np.int8)
    y_true = np.where(x.ravel() < 0, 0.5 * x.ravel(), 0.5 * x.ravel() + np.sin(4 * np.pi * x.ravel()))
    y = (y_true + rng.normal(0, 0.2, n)).astype(np.float32)
    return x, y, y_true.astype(np.float32), regime


def tunable_complexity_dataset(n: int = 1500, complexity_scale: float = 4.0, seed: int = 43) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """y = linear(x) for |x|<1 else sin(complexity_scale * x). `complexity_scale`
    controls how oscillatory the non-linear regime is.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3.0, 3.0, n).reshape(-1, 1).astype(np.float32)
    is_simple = (np.abs(x.ravel()) < 1.0)
    y_simple = 0.3 * x.ravel()
    y_complex = np.sin(complexity_scale * x.ravel()) * 0.6
    y_true = np.where(is_simple, y_simple, y_complex)
    noise = 0.05 + 0.1 * (~is_simple).astype(np.float32)
    y = (y_true + rng.normal(0, noise, n)).astype(np.float32)
    # "Complexity" feature: |local curvature| ~ |second derivative| of y_true
    complexity_metric = np.where(is_simple, 0.0, complexity_scale ** 2).astype(np.float32)
    return x, y, y_true.astype(np.float32), complexity_metric


def train_model(x: np.ndarray, y: np.ndarray, max_layers: int = 5, depth_reg: DepthRegularization = DepthRegularization.ELBO) -> FlexibleHiddenLayersNN:
    m = FlexibleHiddenLayersNN(
        input_size=x.shape[1], max_hidden_layers=max_layers,
        layer_selection_method=LayerSelectionMethod.SOFT_GATING,
        depth_regularization=depth_reg,
        uncertainty_method=UncertaintyMethod.CONSTANT,
        hidden_size=64, n_predictor_layers=1,
        n_epochs=80, learning_rate=0.01, early_stopping_rounds=20, validation_fraction=0.2,
        random_seed=42, calculate_feature_importance=False,
    )
    m.fit(x, y)
    return m


def extract_depth(model: FlexibleHiddenLayersNN, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (argmax_depth+1 per sample, soft_probs)."""
    x_t = torch.tensor(x, dtype=torch.float32).to(model.device)
    model.model.eval()
    with torch.no_grad():
        out = model.model(x_t)
    # Layer-selection tuple: position 3 is the soft n_probs.
    n_probs = out[3]
    if n_probs is None:
        return np.ones(x.shape[0], dtype=np.int64), np.zeros((x.shape[0], model.max_hidden_layers))
    soft = n_probs.cpu().numpy()
    argmax = soft.argmax(axis=1) + 1
    return argmax, soft


def plot_piecewise_heatmap(x: np.ndarray, y: np.ndarray, argmax_depth: np.ndarray, soft: np.ndarray, save_path: Path) -> None:
    order = np.argsort(x.ravel())
    x_sorted = x.ravel()[order]
    y_sorted = y[order]
    argmax_sorted = argmax_depth[order]
    soft_sorted = soft[order]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].scatter(x_sorted, y_sorted, s=4, alpha=0.5, color="black")
    axes[0].set_ylabel("y")
    axes[0].set_title("Target function")

    axes[1].plot(x_sorted, argmax_sorted, color="C1")
    axes[1].set_ylabel("argmax depth")
    axes[1].set_title("Selected depth per input (argmax)")

    # Heatmap: rows = depth, cols = sorted samples
    axes[2].imshow(soft_sorted.T, aspect="auto", origin="lower",
                    extent=[x_sorted[0], x_sorted[-1], 0.5, soft_sorted.shape[1] + 0.5], cmap="viridis")
    axes[2].set_ylabel("depth")
    axes[2].set_xlabel("x")
    axes[2].set_title("Soft depth probabilities")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def main() -> None:
    for name, data_fn in [("piecewise", piecewise_dataset), ("tunable_complexity", tunable_complexity_dataset)]:
        x, y, y_true, aux = data_fn()
        model = train_model(x, y, max_layers=5, depth_reg=DepthRegularization.ELBO)
        argmax, soft = extract_depth(model, x)

        plot_path = OUT_DIR / f"{name}_depth.png"
        plot_piecewise_heatmap(x, y, argmax, soft, plot_path)
        logger.info(f"Wrote {plot_path}")

        # Correlation between depth and complexity indicator
        if name == "piecewise":
            # distance to threshold (x=0) is informative
            dist = np.abs(x.ravel())
            rho_dist, p_dist = spearmanr(dist, argmax)
            regime = aux
            rho_regime, p_regime = spearmanr(regime, argmax)
            pd.DataFrame({
                "metric": ["spearman_depth_vs_|x|", "spearman_depth_vs_regime(0=linear,1=sin)"],
                "rho": [rho_dist, rho_regime],
                "p_value": [p_dist, p_regime],
            }).to_csv(OUT_DIR / f"{name}_correlations.csv", index=False)
        else:
            rho, p = spearmanr(aux, argmax)
            pd.DataFrame({
                "metric": ["spearman_depth_vs_local_complexity"], "rho": [rho], "p_value": [p],
            }).to_csv(OUT_DIR / f"{name}_correlations.csv", index=False)


if __name__ == "__main__":
    main()
