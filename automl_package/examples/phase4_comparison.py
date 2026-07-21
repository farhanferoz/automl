"""Phase 4 model-level comparison.

Runs:
- NLL vs β-NLL on heteroscedastic data (compare calibration, MSE, NLL)
- Symlog vs raw on exponential targets (compare MSE in original scale)
- Hard vs soft inference on FlexibleNN piecewise (compare MSE + timing)
- Conformal coverage on heteroscedastic data
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.model_selection import train_test_split

from automl_package.enums import (
    DepthRegularization,
    LayerSelectionMethod,
    NClassesSelectionMethod,
    RegressionStrategy,
    UncertaintyMethod,
)
from automl_package.models.conformal import ConformalWrapper
from automl_package.models.flexible_neural_network import FlexibleHiddenLayersNN
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel


# ----- Data -----------------------------------------------------------------


def heteroscedastic_data(seed: int = 42, n: int = 1000):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-5, 5, n).reshape(-1, 1).astype(np.float32)
    y_true = (np.sin(x) * 2 + 0.5 * x).ravel()
    noise_std = 0.1 + 0.4 * np.abs(x.ravel())
    y = (y_true + rng.normal(0, noise_std)).astype(np.float32)
    return x, y, y_true.astype(np.float32), noise_std.astype(np.float32)


def piecewise_data(seed: int = 42, n: int = 800):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-5, 5, n).reshape(-1, 1).astype(np.float32)
    y_true = np.where(x.ravel() < 0, 0.5 * x.ravel(), 0.5 * x.ravel() + np.sin(4 * np.pi * x.ravel()))
    y = (y_true + rng.normal(0, 0.2, n)).astype(np.float32)
    return x, y, y_true.astype(np.float32)


def exponential_data(seed: int = 42, n: int = 800):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, n).reshape(-1, 1).astype(np.float32)
    y = (np.exp(x.ravel()) + rng.normal(0, 0.1, n)).astype(np.float32)
    return x, y


# ----- Helpers --------------------------------------------------------------


def fmt(value: float | None, width: int = 8) -> str:
    if value is None:
        return f"{'—':>{width}}"
    return f"{value:>{width}.4f}"


def gaussian_nll(y, y_pred, y_std):
    log_var = 2 * np.log(np.clip(y_std, 1e-6, None))
    return float(0.5 * np.mean(log_var + ((y - y_pred) ** 2) / np.exp(log_var)))


def calibration(y, y_pred, y_std):
    return float(np.mean(np.abs(y - y_pred) <= y_std))


def make_probreg(**kw):
    base = dict(
        input_size=1, n_classes=5,
        uncertainty_method=UncertaintyMethod.PROBABILISTIC,
        n_classes_selection_method=NClassesSelectionMethod.NONE,
        regression_strategy=RegressionStrategy.SEPARATE_HEADS,
        n_epochs=80, learning_rate=0.01,
        early_stopping_rounds=15, validation_fraction=0.2,
        random_seed=42, calculate_feature_importance=False,
    )
    base.update(kw)
    return ProbabilisticRegressionModel(**base)


# ----- Experiments ----------------------------------------------------------


def run_nll_vs_beta_nll():
    print("\n=== NLL vs β-NLL on heteroscedastic data ===")
    x, y, _, noise = heteroscedastic_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    _, noise_test = train_test_split(noise, test_size=0.3, random_state=42)

    rows = []
    for label, kw in [
        ("NLL",        dict(loss_type="nll")),
        ("β-NLL β=0.5", dict(loss_type="beta_nll", beta=0.5)),
        ("β-NLL β=1.0", dict(loss_type="beta_nll", beta=1.0)),
    ]:
        model = make_probreg(**kw)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_std = model.predict_uncertainty(x_test)
        mse = float(np.mean((y_test - y_pred) ** 2))
        nll = gaussian_nll(y_test, y_pred, y_std)
        cal = calibration(y_test, y_pred, y_std)
        r = float(np.corrcoef(noise_test.ravel(), y_std.ravel())[0, 1])
        rows.append((label, mse, nll, cal, r))

    print(f"{'Variant':<14} {'MSE':>8} {'NLL':>8} {'Cal':>8} {'NoiseR':>8}")
    for label, mse, nll, cal, r in rows:
        print(f"{label:<14} {fmt(mse)} {fmt(nll)} {fmt(cal)} {fmt(r)}")


def run_symlog_vs_raw():
    print("\n=== Symlog vs raw on exponential targets ===")
    x, y = exponential_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    rows = []
    for label, kw in [
        ("raw",    dict(target_transform=None)),
        ("symlog", dict(target_transform="symlog")),
    ]:
        model = make_probreg(**kw)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        mse = float(np.mean((y_test - y_pred) ** 2))
        r = float(np.corrcoef(y_test, y_pred)[0, 1])
        rows.append((label, mse, r))

    print(f"{'Transform':<14} {'MSE':>10} {'PearsonR':>10}")
    for label, mse, r in rows:
        print(f"{label:<14} {mse:>10.4f} {r:>10.4f}")


def run_hard_vs_soft():
    print("\n=== Hard vs soft inference on FlexibleNN piecewise ===")
    x, y, _ = piecewise_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    model = FlexibleHiddenLayersNN(
        input_size=1, output_size=1, max_hidden_layers=4, n_predictor_layers=1,
        hidden_size=64, n_epochs=120, learning_rate=0.01, early_stopping_rounds=20,
        validation_fraction=0.2, random_seed=42, calculate_feature_importance=False,
        layer_selection_method=LayerSelectionMethod.GUMBEL_SOFTMAX,
        depth_regularization=DepthRegularization.ELBO,
    )
    model.fit(x_train, y_train)

    n_runs = 20
    x_large = np.tile(x_test, (10, 1))

    t0 = time.perf_counter()
    for _ in range(n_runs):
        y_soft = model.predict(x_large)
    t_soft = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n_runs):
        y_hard = model.predict(x_large, hard_execution=True)
    t_hard = time.perf_counter() - t0

    y_soft_test = model.predict(x_test)
    y_hard_test = model.predict(x_test, hard_execution=True)
    mse_soft = float(np.mean((y_test - y_soft_test) ** 2))
    mse_hard = float(np.mean((y_test - y_hard_test) ** 2))
    diff = float(np.mean((y_soft_test - y_hard_test) ** 2))

    print(f"{'Mode':<14} {'MSE':>10} {'Time(s)':>10}")
    print(f"{'soft':<14} {mse_soft:>10.4f} {t_soft:>10.4f}")
    print(f"{'hard':<14} {mse_hard:>10.4f} {t_hard:>10.4f}")
    print(f"hard-vs-soft pred diff (MSE): {diff:.4f}")
    print(f"speedup (soft/hard): {t_soft / t_hard:.2f}x")


def run_conformal():
    print("\n=== Conformal prediction coverage on heteroscedastic data ===")
    x, y, _, _ = heteroscedastic_data()
    x_train, x_rest, y_train, y_rest = train_test_split(x, y, test_size=0.5, random_state=42)
    x_cal, x_test, y_cal, y_test = train_test_split(x_rest, y_rest, test_size=0.4, random_state=42)

    model = make_probreg()
    model.fit(x_train, y_train)

    print(f"{'α':>6} {'target':>8} {'coverage':>10} {'width':>10}")
    for alpha in [0.05, 0.10, 0.20]:
        cw = ConformalWrapper(model)
        cw.calibrate(x_cal, y_cal, alpha=alpha)
        lower, upper = cw.predict_interval(x_test)
        cov = float(np.mean((y_test >= lower) & (y_test <= upper)))
        width = float(np.mean(upper - lower))
        print(f"{alpha:>6.2f} {1 - alpha:>8.2f} {cov:>10.4f} {width:>10.4f}")


if __name__ == "__main__":
    run_nll_vs_beta_nll()
    run_symlog_vs_raw()
    run_hard_vs_soft()
    run_conformal()
