"""Fast UCI benchmark: all models on 4 UCI regression datasets.

Skips SHAP feature importance and slow models (ClassReg + ProbReg dynamic-k variants
other than best config) to make UCI runs feasible.

Datasets:
  - UCI-Energy(768x8) — building heating load
  - UCI-Yacht(308x6) — hull resistance
  - UCI-Kin8nm(8192x8) — robot kinematics
  - California(20640x8) — house prices (runs only if dataset is small; GP skipped)

Usage:
    ~/dev/.venv/bin/python -m automl_package.examples.uci_benchmark
"""

import concurrent.futures
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from automl_package.enums import (
    DepthRegularization,
    LayerSelectionMethod,
    NClassesRegularization,
    NClassesSelectionMethod,
    RegressionStrategy,
    UncertaintyMethod,
)
from automl_package.utils.calibration import calculate_mpiw, calculate_picp_at_alphas, calculate_sharpness_from_std, ece_regression, miscalibration_area
from automl_package.utils.distributions import GaussianDistribution
from automl_package.utils.metrics import calculate_nll
from automl_package.utils.scoring import calculate_crps_gaussian, calculate_winkler_from_gaussian

OUTPUT_DIR = Path(__file__).parent / "full_benchmark_results"


# --- Datasets ---

def _uci_energy() -> dict:
    from sklearn.datasets import fetch_openml
    d = fetch_openml(name="energy-efficiency", version=1, as_frame=False, parser="auto")
    x = d.data[:, :8].astype(np.float32)
    y = d.target.astype(np.float32) if d.target.ndim == 1 else d.target[:, 0].astype(np.float32)
    return {"x": x, "y": y, "name": "UCI-Energy(768x8)"}


def _uci_yacht() -> dict:
    from sklearn.datasets import fetch_openml
    d = fetch_openml(name="yacht_hydrodynamics", version=1, as_frame=False, parser="auto")
    return {"x": d.data.astype(np.float32), "y": d.target.astype(np.float32), "name": "UCI-Yacht(308x6)"}


def _uci_kin8nm() -> dict:
    from sklearn.datasets import fetch_openml
    d = fetch_openml(name="kin8nm", version=1, as_frame=False, parser="auto")
    return {"x": d.data.astype(np.float32), "y": d.target.astype(np.float32), "name": "UCI-Kin8nm(8192x8)"}


def _uci_california() -> dict:
    from sklearn.datasets import fetch_california_housing
    d = fetch_california_housing()
    return {"x": d.data.astype(np.float32), "y": d.target.astype(np.float32), "name": "California(20640x8)"}


# --- Model factory (feature importance disabled everywhere, fewer ProbReg variants) ---

def _make_models(input_size: int, n_samples: int) -> list[tuple[str, object]]:
    from automl_package.enums import MapperType
    from automl_package.models.baselines.ft_transformer import FTTransformerModel
    from automl_package.models.baselines.gaussian_process import GaussianProcessModel
    from automl_package.models.baselines.mixture_density_network import MixtureDensityNetwork
    from automl_package.models.baselines.quantile_regression_nn import QuantileRegressionNN
    from automl_package.models.catboost_model import CatBoostModel
    from automl_package.models.classifier_regression import ClassifierRegressionModel
    from automl_package.models.flexible_neural_network import FlexibleHiddenLayersNN
    from automl_package.models.lightgbm_model import LightGBMModel
    from automl_package.models.linear_regression import LinearRegressionModel
    from automl_package.models.neural_network import PyTorchNeuralNetwork
    from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel
    from automl_package.models.xgboost_model import XGBoostModel

    common_nn = dict(
        input_size=input_size, learning_rate=0.01, n_epochs=100,
        early_stopping_rounds=15, validation_fraction=0.2,
        random_seed=42, calculate_feature_importance=False,
    )
    common_tree = dict(
        early_stopping_rounds=15, validation_fraction=0.2, random_seed=42,
        calculate_feature_importance=False,
    )

    models: list[tuple[str, object]] = [
        ("LinearReg", LinearRegressionModel(
            learning_rate=0.01, n_iterations=1000, early_stopping_rounds=50,
            validation_fraction=0.2, calculate_feature_importance=False,
        )),
        ("XGBoost", XGBoostModel(n_estimators=200, **common_tree, verbosity=0)),
        ("LightGBM", LightGBMModel(n_estimators=200, **common_tree, verbose=-1)),
        ("CatBoost", CatBoostModel(iterations=200, **common_tree)),
        ("CatBoost+UQ", CatBoostModel(
            iterations=200, uncertainty_method=UncertaintyMethod.PROBABILISTIC, **common_tree,
        )),
        ("ProbReg(k=5)", ProbabilisticRegressionModel(
            n_classes=5, uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.NONE,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            base_classifier_params=dict(hidden_layers=1, hidden_size=64),
            regression_head_params=dict(hidden_layers=0, hidden_size=32),
            **common_nn,
        )),
        ("ProbReg(k=5,SINGLE_FINAL)", ProbabilisticRegressionModel(
            n_classes=5, uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.NONE,
            regression_strategy=RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT,
            base_classifier_params=dict(hidden_layers=1, hidden_size=64),
            regression_head_params=dict(hidden_layers=0, hidden_size=32),
            **common_nn,
        )),
        ("ProbReg(dyn,K_PEN)", ProbabilisticRegressionModel(
            n_classes=10, max_n_classes=10,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.SOFT_GATING,
            n_classes_regularization=NClassesRegularization.K_PENALTY,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            base_classifier_params=dict(hidden_layers=1, hidden_size=64),
            regression_head_params=dict(hidden_layers=0, hidden_size=32),
            **common_nn,
        )),
        ("FlexNN(CONST)", FlexibleHiddenLayersNN(
            max_hidden_layers=4, hidden_size=64, n_predictor_layers=1,
            layer_selection_method=LayerSelectionMethod.SOFT_GATING,
            depth_regularization=DepthRegularization.ELBO,
            uncertainty_method=UncertaintyMethod.CONSTANT, **common_nn,
        )),
        ("FlexNN(PROB)", FlexibleHiddenLayersNN(
            max_hidden_layers=4, hidden_size=64, n_predictor_layers=1,
            layer_selection_method=LayerSelectionMethod.SOFT_GATING,
            depth_regularization=DepthRegularization.ELBO,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC, **common_nn,
        )),
        ("ClassReg(k=7)", ClassifierRegressionModel(
            base_classifier_class=PyTorchNeuralNetwork, n_classes=7, mapper_type=MapperType.LOOKUP_MEDIAN,
            base_classifier_params={"input_size": input_size, "hidden_layers": 2, "hidden_size": 64,
                                    "learning_rate": 0.01, "n_epochs": 100},
            early_stopping_rounds=15, uncertainty_method=UncertaintyMethod.BINNED_RESIDUAL_STD,
            validation_fraction=0.2, random_seed=42, calculate_feature_importance=False,
        )),
        ("MDN(K=5)", MixtureDensityNetwork(n_components=5, hidden_size=64, n_hidden=2, **common_nn)),
        ("QR-NN", QuantileRegressionNN(hidden_size=64, n_hidden=2, **common_nn)),
        ("FT-Transformer", FTTransformerModel(d_model=32, n_heads=4, n_layers=2, **common_nn)),
    ]
    if n_samples <= 5000:
        models.insert(5, ("GP(Matern)", GaussianProcessModel(kernel_nu=2.5, n_restarts_optimizer=3)))
    return models


def _compute_metrics(y_true, y_pred, y_std):
    mse = float(np.mean((y_true - y_pred) ** 2))
    result = {"MSE": mse}
    if y_std is not None:
        y_std = np.maximum(y_std, 1e-6)
        result["NLL"] = calculate_nll(y_true, y_pred, y_std)
        result["CRPS"] = calculate_crps_gaussian(y_true, y_pred, y_std)
        result["Winkler@95"] = calculate_winkler_from_gaussian(y_true, y_pred, y_std, alpha=0.05)
        dist = GaussianDistribution(y_pred, y_std)
        result["ECE"] = ece_regression(y_true, dist)
        result["MiscalArea"] = miscalibration_area(y_true, dist)
        result["Sharpness"] = calculate_sharpness_from_std(y_std)
        picp = calculate_picp_at_alphas(y_true, y_pred, y_std, alphas=(0.05,))
        result["PICP@95"] = picp["picp@95"]
        result["MPIW@95"] = calculate_mpiw(y_std, alpha=0.05)
    return result


def _print_row(name, m, t):
    def fmt(v, w=8, p=3):
        return f"{v:{w}.{p}f}" if v is not None else f"{'--':>{w}}"
    print(f"{name:<25} {fmt(m.get('MSE'))} {fmt(m.get('NLL'))} {fmt(m.get('CRPS'))} "
          f"{fmt(m.get('ECE'), 7, 4)} {fmt(m.get('PICP@95'), 7, 3)} {fmt(m.get('MPIW@95'), 7, 3)} {t:6.1f}s", flush=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ds_fn in [_uci_energy, _uci_yacht, _uci_kin8nm, _uci_california]:
        ds = ds_fn()
        x, y = ds["x"], ds["y"]
        x_train_raw, x_test_raw, y_train_raw, y_test_raw = train_test_split(x, y, test_size=0.3, random_state=42)

        # Standardize features and target for fair NN training
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        x_train = x_scaler.fit_transform(x_train_raw).astype(np.float32)
        x_test = x_scaler.transform(x_test_raw).astype(np.float32)
        y_train = y_scaler.fit_transform(y_train_raw.reshape(-1, 1)).ravel().astype(np.float32)
        y_test = y_scaler.transform(y_test_raw.reshape(-1, 1)).ravel().astype(np.float32)

        print(f"\n{'='*90}", flush=True)
        print(f"  {ds['name']}", flush=True)
        print(f"{'='*90}", flush=True)
        print(f"{'Model':<25} {'MSE':>8} {'NLL':>8} {'CRPS':>8} {'ECE':>7} {'PICP@95':>7} {'MPIW@95':>7} {'Time':>7}", flush=True)
        print("-" * 90, flush=True)

        per_model_timeout_s = 300  # 5 minutes; large datasets (e.g. California) can exceed this
        for name, model in _make_models(input_size=x.shape[1], n_samples=len(x)):
            try:
                t0 = time.time()

                def _fit_call(m=model, xt=x_train, yt=y_train):
                    if hasattr(m, "fit"):
                        m.fit(xt, yt)
                    else:
                        m._fit_single(xt, yt)

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(_fit_call)
                    try:
                        fut.result(timeout=per_model_timeout_s)
                    except concurrent.futures.TimeoutError:
                        # NOTE: ThreadPoolExecutor cannot interrupt the GPU thread; the
                        # worker continues in the background. Tracked in memory/project_future_work.md
                        # until we migrate to multiprocessing.
                        print(f"{name:<25} TIMEOUT after {per_model_timeout_s}s (thread still running)", flush=True)
                        continue
                try:
                    y_pred = np.asarray(model.predict(x_test, filter_data=False)).ravel()
                except TypeError:
                    y_pred = np.asarray(model.predict(x_test)).ravel()
                y_std = None
                try:
                    y_std = np.asarray(model.predict_uncertainty(x_test, filter_data=False)).ravel()
                    if np.all(y_std == 0) or np.all(np.isnan(y_std)):
                        y_std = None
                except (NotImplementedError, RuntimeError, AttributeError, TypeError):
                    pass
                elapsed = time.time() - t0
                m = _compute_metrics(y_test, y_pred, y_std)
                _print_row(name, m, elapsed)
            except Exception as e:
                print(f"{name:<25} FAILED -- {type(e).__name__}: {e!s:.50}", flush=True)


if __name__ == "__main__":
    main()
