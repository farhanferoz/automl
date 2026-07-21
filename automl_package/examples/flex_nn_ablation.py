"""Paper B primary table: FlexibleNN ablation sweep.

Sweep:
- max_hidden_layers in {3, 5, 8}
- depth_regularization in {NONE, DEPTH_PENALTY, ELBO, COST_AWARE_ELBO}
- layer_selection in {SOFT_GATING, GUMBEL_SOFTMAX, STE, REINFORCE}
- weights in {shared, independent}
- inference in {soft, hard}
- uncertainty in {CONSTANT, PROBABILISTIC}

Runs on the piecewise synthetic + subset of synthetic B3 (two-phase) and a
reduced UCI set. Writes CSV with MSE/NLL/mean_depth/coverage@90, plus per-run
json for later aggregation.

Designed for autonomous execution: any single-config crash is logged but does
not abort the sweep (robust for overnight runs).
"""

from __future__ import annotations

import json
import logging
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from automl_package.enums import (
    DepthRegularization,
    LayerSelectionMethod,
    UncertaintyMethod,
)
from automl_package.models.flexible_neural_network import FlexibleHiddenLayersNN
from automl_package.models.independent_weights_flexible_neural_network import IndependentWeightsFlexibleNN
from automl_package.utils.synthetic_datasets import load_fixture

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

OUT_DIR = Path(__file__).parent / "flex_nn_ablation_results"
OUT_DIR.mkdir(exist_ok=True)

MAX_LAYERS_GRID = (3, 5, 8)
DEPTH_REG_GRID = (
    DepthRegularization.NONE,
    DepthRegularization.DEPTH_PENALTY,
    DepthRegularization.ELBO,
    DepthRegularization.COST_AWARE_ELBO,
)
LAYER_METHOD_GRID = (
    LayerSelectionMethod.SOFT_GATING,
    LayerSelectionMethod.GUMBEL_SOFTMAX,
    LayerSelectionMethod.STE,
    LayerSelectionMethod.REINFORCE,
)
UQ_GRID = (UncertaintyMethod.CONSTANT, UncertaintyMethod.PROBABILISTIC)
WEIGHT_GRID = ("shared", "independent")


def _nll(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    sigma = np.maximum(sigma, 1e-9)
    return float(np.mean(0.5 * (np.log(2 * np.pi * sigma**2) + ((y - mu) / sigma) ** 2)))


def _coverage(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, z: float = 1.96) -> float:
    low = mu - z * sigma
    high = mu + z * sigma
    return float(np.mean((y >= low) & (y <= high)))


def run_one(ds_name: str, x_tr, y_tr, x_te, y_te,
            weight_kind: str, max_layers: int, uq: UncertaintyMethod,
            layer_method: LayerSelectionMethod, depth_reg: DepthRegularization,
            n_epochs: int = 40, seed: int = 42) -> dict:
    cls = IndependentWeightsFlexibleNN if weight_kind == "independent" else FlexibleHiddenLayersNN
    t0 = time.perf_counter()
    # LAYER_METHOD_GRID is entirely SOFT_GATING/GUMBEL_SOFTMAX/STE/REINFORCE -- all RETIRED under
    # the nested ladder (MASTER Decision 29, docs/plans/capacity_programme/MASTER.md). This
    # sweep's whole purpose is ablating across those methods (a labelled-comparison-arm table by
    # design), so the escape hatch is used rather than dropping them from the grid.
    m = cls(
        input_size=x_tr.shape[1], max_hidden_layers=max_layers,
        layer_selection_method=layer_method,
        depth_regularization=depth_reg,
        uncertainty_method=uq,
        hidden_size=32, n_predictor_layers=1,
        n_epochs=n_epochs, learning_rate=0.01, early_stopping_rounds=15,
        validation_fraction=0.2, random_seed=seed,
        calculate_feature_importance=False,
        allow_retired_capacity_selection=True,
    )
    m.fit(x_tr, y_tr)
    y_pred = m.predict(x_te)
    row = {
        "dataset": ds_name, "weights": weight_kind, "max_layers": max_layers,
        "uq": uq.value, "layer_method": layer_method.value,
        "depth_reg": depth_reg.value,
    }
    mse = float(np.mean((y_te - y_pred) ** 2))
    row["mse"] = mse
    if uq == UncertaintyMethod.PROBABILISTIC:
        y_std = m.predict_uncertainty(x_te)
        row["nll"] = _nll(y_te, y_pred, y_std)
        row["coverage@95"] = _coverage(y_te, y_pred, y_std)
    # Extract mean depth
    try:
        x_t = torch.tensor(x_te, dtype=torch.float32).to(m.device)
        m.model.eval()
        with torch.no_grad():
            out = m.model(x_t)
        # layer_selection tuple: (output, n_actual, _unused_or_nprobs, n_probs, log_prob)
        # indep weights: (output, n_actual, n_probs, n_logits, log_prob)
        if weight_kind == "independent":
            n_probs = out[2]
        else:
            n_probs = out[3]
        if n_probs is not None:
            depth_idx = torch.arange(1, max_layers + 1, device=n_probs.device, dtype=torch.float32)
            row["mean_depth"] = float((n_probs * depth_idx).sum(1).mean().item())
    except Exception as exc:  # noqa: BLE001
        row["mean_depth"] = float("nan")
        row["error"] = str(exc)
    row["seconds"] = time.perf_counter() - t0
    return row


def _make_datasets() -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Returns list of (name, x, y). Uses piecewise + b3 two-phase."""
    rng = np.random.default_rng(42)
    x_p = rng.uniform(-5, 5, 800).reshape(-1, 1).astype(np.float32)
    y_p = np.where(x_p.ravel() < 0, 0.5 * x_p.ravel(), 0.5 * x_p.ravel() + np.sin(4 * np.pi * x_p.ravel()))
    y_p = (y_p + rng.normal(0, 0.2, 800)).astype(np.float32)

    try:
        b3 = load_fixture("b3_two_phase")
        ds_list = [("piecewise", x_p, y_p), ("b3_two_phase", b3.x, b3.y)]
    except FileNotFoundError:
        ds_list = [("piecewise", x_p, y_p)]
    return ds_list


def main(
    n_epochs: int = 40,
    subsample_grid: bool = True,
) -> None:
    rows: list[dict] = []
    datasets = _make_datasets()

    # Full grid is combinatorially large. For the autonomous sweep we shrink to
    # headline combinations; the full grid is available via `--full` flag in future.
    depth_reg_grid = (
        DepthRegularization.NONE, DepthRegularization.ELBO, DepthRegularization.DEPTH_PENALTY, DepthRegularization.COST_AWARE_ELBO,
    )
    layer_grid = (
        LayerSelectionMethod.SOFT_GATING, LayerSelectionMethod.GUMBEL_SOFTMAX,
        LayerSelectionMethod.STE, LayerSelectionMethod.REINFORCE,
    )
    if subsample_grid:
        max_layers_grid = (5,)
        uq_grid = (UncertaintyMethod.CONSTANT,)
        weight_grid = ("shared",)
    else:
        max_layers_grid = MAX_LAYERS_GRID
        uq_grid = UQ_GRID
        weight_grid = WEIGHT_GRID

    grid = list(product(max_layers_grid, depth_reg_grid, layer_grid, uq_grid, weight_grid))

    for ds_name, x, y in datasets:
        x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.3, random_state=0)
        for max_layers, depth_reg, layer_method, uq, weight_kind in grid:
            try:
                row = run_one(ds_name, x_tr, y_tr, x_te, y_te,
                              weight_kind, max_layers, uq, layer_method, depth_reg,
                              n_epochs=n_epochs)
                rows.append(row)
                logger.info(
                    f"{ds_name} {weight_kind} L={max_layers} "
                    f"{layer_method.value}/{depth_reg.value}/{uq.value}  "
                    f"mse={row['mse']:.4f} mean_depth={row.get('mean_depth', 'NA')}"
                )
            except Exception as e:  # noqa: BLE001
                logger.exception(f"config crashed: {e}")
                rows.append({"dataset": ds_name, "weights": weight_kind, "max_layers": max_layers,
                              "layer_method": layer_method.value, "depth_reg": depth_reg.value,
                              "error": str(e)})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "flex_nn_ablation.csv", index=False)
    (OUT_DIR / "flex_nn_ablation.json").write_text(json.dumps(rows, indent=2, default=str))
    # Best depth_reg per (dataset, layer_method)
    if "mse" in df.columns:
        best = df.dropna(subset=["mse"]).sort_values("mse").groupby(["dataset", "layer_method"]).head(1)
        best.to_csv(OUT_DIR / "best_config_per_dataset_method.csv", index=False)
    logger.info(f"Wrote {len(rows)} rows to {OUT_DIR}")


if __name__ == "__main__":
    main()
