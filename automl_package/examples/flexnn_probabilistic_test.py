"""Test FlexNN with UncertaintyMethod.PROBABILISTIC to close the NLL gap.

Compares three FlexNN configurations head-to-head with ProbReg best config:
  1. FlexNN(CONSTANT) — current benchmark default, poor NLL
  2. FlexNN(PROBABILISTIC) — learned mean+log_var per input, proposed fix
  3. FlexNN(MC_DROPOUT) — known to fail (for completeness)
  4. ProbReg(best) — reference point

Datasets: heteroscedastic (input-dependent noise) and exponential (wide range).

Usage:
    ~/dev/.venv/bin/python -m automl_package.examples.flexnn_probabilistic_test
"""

import time

import numpy as np
from sklearn.model_selection import train_test_split

from automl_package.enums import (
    DepthRegularization,
    LayerSelectionMethod,
    NClassesRegularization,
    NClassesSelectionMethod,
    RegressionStrategy,
    UncertaintyMethod,
)
from automl_package.utils.calibration import calculate_sharpness_from_std, ece_regression
from automl_package.utils.distributions import GaussianDistribution
from automl_package.utils.metrics import calculate_nll
from automl_package.utils.scoring import calculate_crps_gaussian


def _heteroscedastic(n: int = 1000, seed: int = 42) -> dict:
    np.random.seed(seed)
    x = np.random.uniform(-5, 5, n).reshape(-1, 1).astype(np.float32)
    y_true = (np.sin(x) * 2 + 0.5 * x).ravel()
    noise_std = (0.1 + 0.4 * np.abs(x)).ravel()
    y = y_true + np.random.normal(0, noise_std).astype(np.float32)
    return {"x": x, "y": y, "noise_std": noise_std, "name": "heteroscedastic"}


def _exponential(n: int = 800, seed: int = 42) -> dict:
    np.random.seed(seed)
    x = np.random.uniform(-3, 3, n).reshape(-1, 1).astype(np.float32)
    y = (np.exp(x.ravel()) + np.random.normal(0, 0.5, n)).astype(np.float32)
    return {"x": x, "y": y, "name": "exponential"}


def _evaluate(model, x_test, y_test, noise_std_test=None) -> dict:
    y_pred = np.asarray(model.predict(x_test, filter_data=False)).ravel()
    y_std = np.maximum(np.asarray(model.predict_uncertainty(x_test, filter_data=False)).ravel(), 1e-6)
    mse = float(np.mean((y_test - y_pred) ** 2))
    nll = calculate_nll(y_test, y_pred, y_std)
    crps = calculate_crps_gaussian(y_test, y_pred, y_std)
    ece = ece_regression(y_test, GaussianDistribution(y_pred, y_std))
    sharp = calculate_sharpness_from_std(y_std)
    result = {"MSE": mse, "NLL": nll, "CRPS": crps, "ECE": ece, "Sharp": sharp, "std_var": float(np.std(y_std))}
    if noise_std_test is not None:
        with np.errstate(invalid="ignore"):
            result["NoiseR"] = float(np.corrcoef(y_std, noise_std_test)[0, 1])
    return result


def run(ds):
    from automl_package.models.flexible_neural_network import FlexibleHiddenLayersNN
    from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel

    x, y = ds["x"], ds["y"]
    noise_std = ds.get("noise_std")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    noise_std_test = noise_std[len(x_train):] if noise_std is not None else None

    common_flex = dict(
        max_hidden_layers=4, hidden_size=64, n_predictor_layers=1,
        layer_selection_method=LayerSelectionMethod.SOFT_GATING,
        depth_regularization=DepthRegularization.ELBO,
        learning_rate=0.01, n_epochs=120, early_stopping_rounds=20,
        validation_fraction=0.2, random_seed=42, calculate_feature_importance=False,
    )

    configs = [
        ("FlexNN(CONSTANT)", FlexibleHiddenLayersNN(
            uncertainty_method=UncertaintyMethod.CONSTANT, **common_flex,
        )),
        ("FlexNN(PROBABILISTIC)", FlexibleHiddenLayersNN(
            uncertainty_method=UncertaintyMethod.PROBABILISTIC, **common_flex,
        )),
        ("ProbReg(dyn,SG,K_PEN)", ProbabilisticRegressionModel(
            n_classes=10, max_n_classes=10,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.SOFT_GATING,
            n_classes_regularization=NClassesRegularization.K_PENALTY,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            base_classifier_params=dict(hidden_layers=1, hidden_size=64),
            regression_head_params=dict(hidden_layers=0, hidden_size=32),
            learning_rate=0.01, n_epochs=100, early_stopping_rounds=15,
            validation_fraction=0.2, random_seed=42, calculate_feature_importance=False,
        )),
    ]

    print(f"\n{'='*92}")
    print(f"  {ds['name']} — FlexNN PROBABILISTIC test")
    print(f"{'='*92}")
    nr_header = " NoiseR" if noise_std_test is not None else ""
    print(f"  {'Config':<24} {'MSE':>8} {'NLL':>8} {'CRPS':>8} {'ECE':>7} {'Sharp':>7} {'σ_var':>7}{nr_header:>9} {'Time':>6}")
    print(f"  {'-'*92}")

    for name, model in configs:
        try:
            t0 = time.time()
            model.fit(x_train, y_train)
            elapsed = time.time() - t0
            m = _evaluate(model, x_test, y_test, noise_std_test)
            nr_s = f" {m.get('NoiseR', float('nan')):7.3f}" if "NoiseR" in m else ""
            print(f"  {name:<24} {m['MSE']:8.4f} {m['NLL']:8.3f} {m['CRPS']:8.4f} "
                  f"{m['ECE']:7.4f} {m['Sharp']:7.4f} {m['std_var']:7.4f}{nr_s} {elapsed:5.1f}s")
        except Exception as e:
            print(f"  {name:<24} FAILED -- {type(e).__name__}: {e!s:.60}")


if __name__ == "__main__":
    for ds_fn in [_heteroscedastic, _exponential]:
        run(ds_fn())
