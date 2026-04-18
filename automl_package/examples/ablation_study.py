"""Ablation study: FlexNN and ProbReg configuration sweep.

Investigates:
  1. FlexNN: Why is NLL poor? Sweep depth_regularization, layer_selection_method,
     max_hidden_layers, uncertainty methods.
  2. ProbReg: Best regression strategy × n_classes × selection method.

Runs on heteroscedastic and exponential datasets (where UQ matters most).

Usage:
    ~/dev/.venv/bin/python -m automl_package.examples.ablation_study
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
    y_true = np.exp(x.ravel())
    y = (y_true + np.random.normal(0, 0.5, n)).astype(np.float32)
    return {"x": x, "y": y, "name": "exponential"}


def _metrics(y_true, y_pred, y_std, noise_std=None):
    y_pred = np.asarray(y_pred).ravel()
    y_std = np.maximum(np.asarray(y_std).ravel(), 1e-6) if y_std is not None else None
    mse = float(np.mean((y_true - y_pred) ** 2))
    if y_std is None:
        return {"MSE": mse}
    nll = calculate_nll(y_true, y_pred, y_std)
    crps = calculate_crps_gaussian(y_true, y_pred, y_std)
    dist = GaussianDistribution(y_pred, y_std)
    ece = ece_regression(y_true, dist)
    result = {"MSE": mse, "NLL": nll, "CRPS": crps, "ECE": ece, "Sharp": calculate_sharpness_from_std(y_std)}
    if noise_std is not None:
        result["NoiseR"] = float(np.corrcoef(y_std, noise_std)[0, 1])
    return result


def _run_one(model, x_train, y_train, x_test, y_test, noise_std_test=None):
    t0 = time.time()
    model.fit(x_train, y_train)
    y_pred = np.asarray(model.predict(x_test, filter_data=False)).ravel()
    y_std = None
    try:
        y_std = np.asarray(model.predict_uncertainty(x_test, filter_data=False)).ravel()
    except Exception:
        pass
    elapsed = time.time() - t0
    m = _metrics(y_test, y_pred, y_std, noise_std_test)
    m["Time"] = elapsed
    return m


def _print_row(name, m):
    nr = f"{m.get('NoiseR', float('nan')):7.3f}" if "NoiseR" in m else f"{'--':>7}"
    print(f"  {name:<40} {m['MSE']:8.4f} {m.get('NLL', float('nan')):8.3f} {m.get('CRPS', float('nan')):8.4f} "
          f"{m.get('ECE', float('nan')):7.4f} {nr} {m.get('Time', 0):6.1f}s")


def ablation_flexnn(ds):
    """Sweep FlexNN configurations."""
    from automl_package.models.flexible_neural_network import FlexibleHiddenLayersNN

    x, y = ds["x"], ds["y"]
    noise_std = ds.get("noise_std")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    noise_std_test = noise_std[len(x_train):] if noise_std is not None else None

    common = dict(
        hidden_size=64, n_predictor_layers=1,
        learning_rate=0.01, n_epochs=120, early_stopping_rounds=20,
        validation_fraction=0.2, random_seed=42, calculate_feature_importance=False,
    )

    configs = [
        # Baseline: no depth selection, constant uncertainty
        ("FlexNN(depth=2,CONST)", dict(
            max_hidden_layers=2, layer_selection_method=LayerSelectionMethod.NONE,
            depth_regularization=DepthRegularization.NONE,
            uncertainty_method=UncertaintyMethod.CONSTANT, **common)),
        # Constant uncertainty + ELBO
        ("FlexNN(ELBO,CONST)", dict(
            max_hidden_layers=4, layer_selection_method=LayerSelectionMethod.SOFT_GATING,
            depth_regularization=DepthRegularization.ELBO,
            uncertainty_method=UncertaintyMethod.CONSTANT, **common)),
        # MC Dropout uncertainty
        ("FlexNN(ELBO,MC_DROP)", dict(
            max_hidden_layers=4, layer_selection_method=LayerSelectionMethod.SOFT_GATING,
            depth_regularization=DepthRegularization.ELBO,
            uncertainty_method=UncertaintyMethod.MC_DROPOUT,
            dropout_rate=0.1, n_mc_dropout_samples=30, **common)),
        # Depth penalty instead of ELBO
        ("FlexNN(DEPTH_PENALTY,CONST)", dict(
            max_hidden_layers=4, layer_selection_method=LayerSelectionMethod.SOFT_GATING,
            depth_regularization=DepthRegularization.DEPTH_PENALTY,
            uncertainty_method=UncertaintyMethod.CONSTANT, **common)),
        # Gumbel instead of SoftGating
        ("FlexNN(ELBO,Gumbel,CONST)", dict(
            max_hidden_layers=4, layer_selection_method=LayerSelectionMethod.GUMBEL_SOFTMAX,
            depth_regularization=DepthRegularization.ELBO,
            uncertainty_method=UncertaintyMethod.CONSTANT, **common)),
        # Deeper: max_layers=6
        ("FlexNN(ELBO,max=6,CONST)", dict(
            max_hidden_layers=6, layer_selection_method=LayerSelectionMethod.SOFT_GATING,
            depth_regularization=DepthRegularization.ELBO,
            uncertainty_method=UncertaintyMethod.CONSTANT, **common)),
        # Wider: hidden=128
        ("FlexNN(ELBO,h=128,CONST)", dict(
            max_hidden_layers=4, layer_selection_method=LayerSelectionMethod.SOFT_GATING,
            depth_regularization=DepthRegularization.ELBO,
            uncertainty_method=UncertaintyMethod.CONSTANT, hidden_size=128,
            n_predictor_layers=1, learning_rate=0.01, n_epochs=120,
            early_stopping_rounds=20, validation_fraction=0.2, random_seed=42,
            calculate_feature_importance=False)),
    ]

    print(f"\n{'='*90}")
    print(f"  FlexNN Ablation — {ds['name']}")
    print(f"{'='*90}")
    print(f"  {'Config':<40} {'MSE':>8} {'NLL':>8} {'CRPS':>8} {'ECE':>7} {'NoiseR':>7} {'Time':>7}")
    print(f"  {'-'*84}")

    for name, cfg in configs:
        try:
            model = FlexibleHiddenLayersNN(**cfg)
            m = _run_one(model, x_train, y_train, x_test, y_test, noise_std_test)
            _print_row(name, m)
        except Exception as e:
            print(f"  {name:<40} FAILED -- {type(e).__name__}: {e!s:.50}")


def ablation_probreg(ds):
    """Sweep ProbReg configurations."""
    from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel

    x, y = ds["x"], ds["y"]
    noise_std = ds.get("noise_std")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    noise_std_test = noise_std[len(x_train):] if noise_std is not None else None

    common = dict(
        uncertainty_method=UncertaintyMethod.PROBABILISTIC,
        base_classifier_params=dict(hidden_layers=1, hidden_size=64),
        regression_head_params=dict(hidden_layers=0, hidden_size=32),
        learning_rate=0.01, n_epochs=100, early_stopping_rounds=15,
        validation_fraction=0.2, random_seed=42, calculate_feature_importance=False,
    )

    configs = [
        # --- Regression strategy sweep (fixed k=5) ---
        ("ProbReg(k=5,SEP_HEADS)", dict(
            n_classes=5, n_classes_selection_method=NClassesSelectionMethod.NONE,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS, **common)),
        ("ProbReg(k=5,SINGLE_N)", dict(
            n_classes=5, n_classes_selection_method=NClassesSelectionMethod.NONE,
            regression_strategy=RegressionStrategy.SINGLE_HEAD_N_OUTPUTS, **common)),
        ("ProbReg(k=5,SINGLE_FINAL)", dict(
            n_classes=5, n_classes_selection_method=NClassesSelectionMethod.NONE,
            regression_strategy=RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT, **common)),

        # --- k sweep (SEPARATE_HEADS) ---
        ("ProbReg(k=2,SEP)", dict(
            n_classes=2, n_classes_selection_method=NClassesSelectionMethod.NONE,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS, **common)),
        ("ProbReg(k=3,SEP)", dict(
            n_classes=3, n_classes_selection_method=NClassesSelectionMethod.NONE,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS, **common)),
        ("ProbReg(k=10,SEP)", dict(
            n_classes=10, n_classes_selection_method=NClassesSelectionMethod.NONE,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS, **common)),
        ("ProbReg(k=20,SEP)", dict(
            n_classes=20, n_classes_selection_method=NClassesSelectionMethod.NONE,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS, **common)),

        # --- Dynamic k strategies (max_n=10, SEPARATE_HEADS) ---
        ("ProbReg(dyn,SG,NONE)", dict(
            n_classes=10, max_n_classes=10,
            n_classes_selection_method=NClassesSelectionMethod.SOFT_GATING,
            n_classes_regularization=NClassesRegularization.NONE,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS, **common)),
        ("ProbReg(dyn,SG,ELBO)", dict(
            n_classes=10, max_n_classes=10,
            n_classes_selection_method=NClassesSelectionMethod.SOFT_GATING,
            n_classes_regularization=NClassesRegularization.ELBO,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS, **common)),
        ("ProbReg(dyn,SG,K_PEN)", dict(
            n_classes=10, max_n_classes=10,
            n_classes_selection_method=NClassesSelectionMethod.SOFT_GATING,
            n_classes_regularization=NClassesRegularization.K_PENALTY,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS, **common)),
        ("ProbReg(dyn,Gumbel,ELBO)", dict(
            n_classes=10, max_n_classes=10,
            n_classes_selection_method=NClassesSelectionMethod.GUMBEL_SOFTMAX,
            n_classes_regularization=NClassesRegularization.ELBO,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS, **common)),

        # --- Loss type sweep ---
        ("ProbReg(k=5,beta_nll_0.5)", dict(
            n_classes=5, n_classes_selection_method=NClassesSelectionMethod.NONE,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            loss_type="beta_nll", beta=0.5, **common)),
    ]

    print(f"\n{'='*90}")
    print(f"  ProbReg Ablation — {ds['name']}")
    print(f"{'='*90}")
    print(f"  {'Config':<40} {'MSE':>8} {'NLL':>8} {'CRPS':>8} {'ECE':>7} {'NoiseR':>7} {'Time':>7}")
    print(f"  {'-'*84}")

    for name, cfg in configs:
        try:
            model = ProbabilisticRegressionModel(**cfg)
            m = _run_one(model, x_train, y_train, x_test, y_test, noise_std_test)
            _print_row(name, m)
        except Exception as e:
            print(f"  {name:<40} FAILED -- {type(e).__name__}: {e!s:.50}")


if __name__ == "__main__":
    for ds_fn in [_heteroscedastic, _exponential]:
        ds = ds_fn()
        ablation_probreg(ds)
        ablation_flexnn(ds)
