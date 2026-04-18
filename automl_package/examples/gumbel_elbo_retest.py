"""I8: re-test Gumbel+ELBO after N1/N2 fixes.

Pre-fix state: Gumbel+ELBO was documented as broken (noisy KL gradients →
poor training dynamics). After Phase 6 (N1 normalisation) and Phase 9 (N2
STE gradient path + FlexNN optimizer inclusion of n_predictor), does it work?

This script trains FlexibleHiddenLayersNN and ProbabilisticRegressionModel
with every combination of (SOFT_GATING, GUMBEL_SOFTMAX) x (NONE, ELBO) on
the piecewise / heteroscedastic toys and reports:
  - final MSE / NLL
  - mean selected depth or mean k
  - depth/k distribution entropy (stability indicator)

Low entropy collapse with Gumbel+ELBO would confirm the original failure mode.
"""

from __future__ import annotations

import logging
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from automl_package.enums import (
    DepthRegularization,
    LayerSelectionMethod,
    NClassesRegularization,
    NClassesSelectionMethod,
    RegressionStrategy,
    UncertaintyMethod,
)
from automl_package.models.flexible_neural_network import FlexibleHiddenLayersNN
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

OUT_DIR = Path(__file__).parent / "gumbel_elbo_retest_results"
OUT_DIR.mkdir(exist_ok=True)


def _piecewise() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    x = rng.uniform(-5, 5, 800).reshape(-1, 1).astype(np.float32)
    y = np.where(x.ravel() < 0, 0.5 * x.ravel(), 0.5 * x.ravel() + np.sin(4 * np.pi * x.ravel()))
    y = (y + rng.normal(0, 0.2, 800)).astype(np.float32)
    return x, y


def _hetero() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    x = rng.uniform(-5, 5, 800).reshape(-1, 1).astype(np.float32)
    y = (np.sin(x.ravel()) * 2 + 0.5 * x.ravel() + rng.normal(0, 0.1 + 0.4 * np.abs(x.ravel()))).astype(np.float32)
    return x, y


def _entropy(probs: np.ndarray) -> float:
    p = np.clip(probs, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum(axis=1).mean())


def run_flex_nn(x_tr, y_tr, x_te, y_te, method: LayerSelectionMethod, reg: DepthRegularization) -> dict:
    m = FlexibleHiddenLayersNN(
        input_size=x_tr.shape[1], max_hidden_layers=5, hidden_size=32,
        layer_selection_method=method, depth_regularization=reg,
        uncertainty_method=UncertaintyMethod.CONSTANT,
        n_epochs=60, learning_rate=0.01, early_stopping_rounds=20, validation_fraction=0.2,
        random_seed=42, calculate_feature_importance=False,
    )
    m.fit(x_tr, y_tr)
    y_pred = m.predict(x_te)
    mse = float(np.mean((y_te - y_pred) ** 2))
    x_t = torch.tensor(x_te, dtype=torch.float32).to(m.device)
    m.model.eval()
    with torch.no_grad():
        out = m.model(x_t)
    n_probs = out[3]
    if n_probs is None:
        return {"model": "FlexNN", "method": method.value, "reg": reg.value, "mse": mse, "mean_depth": None, "entropy": None}
    n_probs_np = n_probs.cpu().numpy()
    depth_idx = np.arange(1, 5 + 1)
    mean_depth = float((n_probs_np * depth_idx).sum(1).mean())
    return {
        "model": "FlexNN", "method": method.value, "reg": reg.value,
        "mse": mse, "mean_depth": mean_depth, "entropy": _entropy(n_probs_np),
    }


def run_probreg(x_tr, y_tr, x_te, y_te, method: NClassesSelectionMethod, reg: NClassesRegularization) -> dict:
    m = ProbabilisticRegressionModel(
        input_size=x_tr.shape[1], n_classes=3, max_n_classes_for_probabilistic_path=7,
        uncertainty_method=UncertaintyMethod.PROBABILISTIC,
        n_classes_selection_method=method, n_classes_regularization=reg,
        regression_strategy=RegressionStrategy.SEPARATE_HEADS,
        base_classifier_params={"hidden_layers": 1, "hidden_size": 64},
        regression_head_params={"hidden_layers": 0, "hidden_size": 32},
        n_epochs=60, learning_rate=0.01, early_stopping_rounds=20, validation_fraction=0.2,
        random_seed=42, calculate_feature_importance=False,
    )
    m.fit(x_tr, y_tr)
    y_pred = m.predict(x_te)
    y_std = m.predict_uncertainty(x_te)
    mse = float(np.mean((y_te - y_pred) ** 2))
    nll = float(np.mean(0.5 * (np.log(2 * np.pi * np.maximum(y_std, 1e-9) ** 2) + ((y_te - y_pred) / np.maximum(y_std, 1e-9)) ** 2)))
    # Extract k probs
    x_t = torch.tensor(x_te, dtype=torch.float32).to(m.device)
    m.model.eval()
    with torch.no_grad():
        _, _, k_actual, _, _ = m.model(x_t)
    mean_k = float(k_actual.float().mean().item())
    mode_probs = getattr(m.model.n_classes_strategy, "mode_selection_probs", None)
    entropy = _entropy(mode_probs.cpu().numpy()) if mode_probs is not None else None
    return {
        "model": "ProbReg", "method": method.value, "reg": reg.value,
        "mse": mse, "nll": nll, "mean_k": mean_k, "entropy": entropy,
    }


def main() -> None:
    rows: list[dict] = []
    x_p, y_p = _piecewise()
    x_h, y_h = _hetero()

    for ds_name, (x, y) in [("piecewise", (x_p, y_p)), ("heteroscedastic", (x_h, y_h))]:
        x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.3, random_state=0)
        for method in (LayerSelectionMethod.SOFT_GATING, LayerSelectionMethod.GUMBEL_SOFTMAX):
            for reg in (DepthRegularization.NONE, DepthRegularization.ELBO):
                row = run_flex_nn(x_tr, y_tr, x_te, y_te, method, reg)
                row["dataset"] = ds_name
                rows.append(row)
                logger.info(f"{ds_name} {row['model']} {method.value}/{reg.value}  mse={row['mse']:.4f} depth={row['mean_depth']} ent={row['entropy']}")

        for method in (NClassesSelectionMethod.SOFT_GATING, NClassesSelectionMethod.GUMBEL_SOFTMAX):
            for reg in (NClassesRegularization.NONE, NClassesRegularization.ELBO):
                row = run_probreg(x_tr, y_tr, x_te, y_te, method, reg)
                row["dataset"] = ds_name
                rows.append(row)
                logger.info(f"{ds_name} {row['model']} {method.value}/{reg.value}  mse={row['mse']:.4f} nll={row.get('nll'):.3f} mean_k={row['mean_k']} ent={row['entropy']}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "gumbel_elbo_retest.csv", index=False)
    logger.info(f"Wrote {len(rows)} rows.")


if __name__ == "__main__":
    main()
