"""Full Benchmark: all models (existing + new baselines) on all toy datasets.

Compares MSE, NLL, CRPS, ECE, Winkler@95, PICP@95, MPIW@95, sharpness,
and miscalibration area across:

Datasets:
  1. Heteroscedastic sine — noise grows with |x|
  2. Piecewise — linear + sinusoidal, tests depth selection
  3. Multimodal — y = x ± 1.5, tests mixture heads
  4. Exponential — y = exp(x) + noise, tests symlog

Models (existing):
  - LinearRegression, XGBoost, LightGBM, CatBoost
  - PyTorchNeuralNetwork (constant variance)
  - CatBoost (RMSEWithUncertainty)
  - ClassifierRegression (LOOKUP_MEDIAN)
  - ProbabilisticRegression (k=5, SEPARATE_HEADS)
  - ProbabilisticRegression (dynamic k, ELBO+SoftGating)
  - FlexibleNN (ELBO depth)

Models (new baselines):
  - GaussianProcess
  - MDN (K=5)
  - QuantileRegressionNN
  - DeepEnsemble (M=3, over NeuralNetwork with PROBABILISTIC uncertainty)
  - FT-Transformer

Usage:
    ~/dev/.venv/bin/python -m automl_package.examples.full_benchmark
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from automl_package.enums import (
    DepthRegularization,
    LayerSelectionMethod,
    MapperType,
    NClassesRegularization,
    NClassesSelectionMethod,
    RegressionStrategy,
    UncertaintyMethod,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")

OUTPUT_DIR = Path(__file__).parent / "full_benchmark_results"
MODEL_TIMEOUT_SECS = 300  # 5 minutes per model


# ---------------------------------------------------------------------------
# Dataset generators
# ---------------------------------------------------------------------------

def _heteroscedastic(n: int = 1000, seed: int = 42) -> dict:
    np.random.seed(seed)
    x = np.random.uniform(-5, 5, n).reshape(-1, 1).astype(np.float32)
    y_true = (np.sin(x) * 2 + 0.5 * x).ravel()
    noise_std = (0.1 + 0.4 * np.abs(x)).ravel()
    y = y_true + np.random.normal(0, noise_std).astype(np.float32)
    return {"x": x, "y": y, "y_true": y_true, "noise_std": noise_std, "name": "heteroscedastic"}


def _piecewise(n: int = 800, seed: int = 42) -> dict:
    np.random.seed(seed)
    x = np.random.uniform(-5, 5, n).reshape(-1, 1).astype(np.float32)
    y_true = np.where(x < 0, 0.5 * x, 0.5 * x + np.sin(4 * np.pi * x)).ravel()
    y = (y_true + np.random.normal(0, 0.2, n)).astype(np.float32)
    return {"x": x, "y": y, "y_true": y_true, "name": "piecewise"}


def _multimodal(n: int = 1000, seed: int = 42) -> dict:
    np.random.seed(seed)
    x = np.random.uniform(-3, 3, n).reshape(-1, 1).astype(np.float32)
    sign = np.random.choice([-1, 1], size=n)
    y = (x.ravel() + sign * 1.5 + np.random.normal(0, 0.1, n)).astype(np.float32)
    return {"x": x, "y": y, "name": "multimodal"}


def _exponential(n: int = 800, seed: int = 42) -> dict:
    np.random.seed(seed)
    x = np.random.uniform(-3, 3, n).reshape(-1, 1).astype(np.float32)
    y_true = np.exp(x.ravel())
    y = (y_true + np.random.normal(0, 0.5, n)).astype(np.float32)
    return {"x": x, "y": y, "y_true": y_true, "name": "exponential"}


def _uci_energy() -> dict:
    from sklearn.datasets import fetch_openml
    d = fetch_openml(name="energy-efficiency", version=1, as_frame=False, parser="auto")
    x = d.data[:, :8].astype(np.float32)  # 8 building features, drop heating/cooling split
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


TOY_DATASETS = [_heteroscedastic, _piecewise, _multimodal, _exponential]
UCI_DATASETS = [_uci_energy, _uci_yacht, _uci_kin8nm, _uci_california]
ALL_DATASETS = TOY_DATASETS + UCI_DATASETS


# ---------------------------------------------------------------------------
# Model factories — each returns (name, model_instance)
# ---------------------------------------------------------------------------

def _make_models(input_size: int = 1, n_samples: int = 1000) -> list[tuple[str, object]]:
    """Creates all model instances for benchmarking."""
    # New baselines
    from automl_package.models.baselines.deep_ensemble import DeepEnsemble
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
    )

    models: list[tuple[str, object]] = [
        # --- Existing models ---
        ("LinearReg", LinearRegressionModel(
            learning_rate=0.01, n_iterations=1000, early_stopping_rounds=50, validation_fraction=0.2
        )),
        ("XGBoost", XGBoostModel(n_estimators=200, **common_tree, verbosity=0)),
        ("LightGBM", LightGBMModel(n_estimators=200, **common_tree, verbose=-1)),
        ("CatBoost", CatBoostModel(iterations=200, **common_tree)),
        ("CatBoost+UQ", CatBoostModel(
            iterations=200, uncertainty_method=UncertaintyMethod.PROBABILISTIC, **common_tree,
        )),
        ("NeuralNet", PyTorchNeuralNetwork(
            hidden_layers=2, hidden_size=64,
            uncertainty_method=UncertaintyMethod.CONSTANT,
            learning_rate=0.01, n_epochs=100, early_stopping_rounds=15,
            validation_fraction=0.2, random_seed=42, calculate_feature_importance=False,
        )),
        ("ClassReg(LM,k=7)", ClassifierRegressionModel(
            base_classifier_class=PyTorchNeuralNetwork, n_classes=7,
            mapper_type=MapperType.LOOKUP_MEDIAN,
            base_classifier_params=dict(input_size=input_size, hidden_layers=2, hidden_size=64,
                                        learning_rate=0.01, n_epochs=100),
            uncertainty_method=UncertaintyMethod.BINNED_RESIDUAL_STD,
            early_stopping_rounds=15, validation_fraction=0.2, random_seed=42,
            calculate_feature_importance=False,
        )),
        ("ProbReg(k=5)", ProbabilisticRegressionModel(
            n_classes=5, uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.NONE,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            base_classifier_params=dict(hidden_layers=1, hidden_size=64),
            regression_head_params=dict(hidden_layers=0, hidden_size=32),
            **common_nn,
        )),
        # For n > 10k, dynamic-k is too slow; use fixed k=5 to avoid timeout
        (
            "ProbReg(ELBO+SG)" if n_samples <= 10000 else "ProbReg(k=5,fixed)",
            ProbabilisticRegressionModel(
                n_classes=10, max_n_classes=10,
                uncertainty_method=UncertaintyMethod.PROBABILISTIC,
                n_classes_selection_method=NClassesSelectionMethod.SOFT_GATING if n_samples <= 10000 else NClassesSelectionMethod.NONE,
                n_classes_regularization=NClassesRegularization.ELBO if n_samples <= 10000 else NClassesRegularization.NONE,
                regression_strategy=RegressionStrategy.SEPARATE_HEADS,
                base_classifier_params=dict(hidden_layers=1, hidden_size=64),
                regression_head_params=dict(hidden_layers=0, hidden_size=32),
                **common_nn,
            )
        ),
        ("FlexNN(ELBO)", FlexibleHiddenLayersNN(
            max_hidden_layers=4, n_predictor_layers=1, hidden_size=64, output_size=1,
            layer_selection_method=LayerSelectionMethod.SOFT_GATING,
            depth_regularization=DepthRegularization.ELBO,
            **common_nn,
        )),
        # --- New baselines ---
        *([("GP(Matern)", GaussianProcessModel(kernel_nu=2.5, n_restarts_optimizer=3))] if n_samples <= 5000 else []),
        ("MDN(K=5)", MixtureDensityNetwork(
            n_components=5, hidden_size=64, n_hidden=2, **common_nn,
        )),
        ("QR-NN", QuantileRegressionNN(
            hidden_size=64, n_hidden=2, **common_nn,
        )),
        ("DeepEns(M=3)", DeepEnsemble(
            base_model=PyTorchNeuralNetwork(
                hidden_layers=2, hidden_size=64,
                uncertainty_method=UncertaintyMethod.PROBABILISTIC,
                learning_rate=0.01, n_epochs=100, early_stopping_rounds=15,
                validation_fraction=0.2, random_seed=42, calculate_feature_importance=False,
            ),
            n_members=3, base_seed=42,
        )),
        ("FT-Transformer", FTTransformerModel(
            d_model=32, n_heads=4, n_layers=2, **common_nn,
        )),
    ]
    return models


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray | None) -> dict[str, float]:
    """Compute all metrics for a single model on a single dataset."""
    from automl_package.utils.calibration import calculate_mpiw, calculate_picp_at_alphas, calculate_sharpness_from_std, ece_regression, miscalibration_area
    from automl_package.utils.distributions import GaussianDistribution
    from automl_package.utils.metrics import calculate_nll
    from automl_package.utils.scoring import calculate_crps_gaussian, calculate_winkler_from_gaussian

    mse = float(np.mean((y_true - y_pred) ** 2))
    result = {"MSE": mse}

    if y_std is not None:
        y_std_safe = np.maximum(y_std, 1e-6)
        result["NLL"] = calculate_nll(y_true, y_pred, y_std_safe)
        result["CRPS"] = calculate_crps_gaussian(y_true, y_pred, y_std_safe)
        result["Winkler@95"] = calculate_winkler_from_gaussian(y_true, y_pred, y_std_safe, alpha=0.05)
        dist = GaussianDistribution(y_pred, y_std_safe)
        result["ECE"] = ece_regression(y_true, dist)
        result["MiscalArea"] = miscalibration_area(y_true, dist)
        result["Sharpness"] = calculate_sharpness_from_std(y_std_safe)
        picp = calculate_picp_at_alphas(y_true, y_pred, y_std_safe, alphas=(0.05,))
        result["PICP@95"] = picp["picp@95"]
        result["MPIW@95"] = calculate_mpiw(y_std_safe, alpha=0.05)
    return result


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def _run_model_inner(model: object, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Inner model run — called in a thread for timeout enforcement."""
    t0 = time.time()
    if hasattr(model, "fit"):
        model.fit(x_train, y_train)
    else:
        model._fit_single(x_train, y_train)

    try:
        y_pred = model.predict(x_test, filter_data=False)
    except TypeError:
        y_pred = model.predict(x_test)
    y_pred = np.asarray(y_pred).ravel()

    y_std = None
    try:
        y_std = model.predict_uncertainty(x_test, filter_data=False)
        if y_std is not None:
            y_std = np.asarray(y_std).ravel()
            if np.all(y_std == 0) or np.all(np.isnan(y_std)):
                y_std = None
    except (NotImplementedError, RuntimeError, AttributeError, TypeError):
        pass

    elapsed = time.time() - t0
    metrics = _compute_metrics(y_test, y_pred, y_std)
    metrics["_time"] = elapsed
    metrics["_model"] = model
    return metrics


def _run_model(model_name: str, model: object, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> dict | None:
    """Train one model, predict, compute metrics. Returns metrics dict or None on failure/timeout."""
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_model_inner, model, x_train, y_train, x_test, y_test)
            try:
                return future.result(timeout=MODEL_TIMEOUT_SECS)
            except FuturesTimeoutError:
                print(f"{model_name:<22} {'TIMEOUT':>8} -- exceeded {MODEL_TIMEOUT_SECS}s limit")
                return None
    except Exception as e:
        print(f"{model_name:<22} {'FAILED':>8} -- {type(e).__name__}: {e!s:.60}")
        return None


def _print_table(results: list[tuple[str, dict]]) -> None:
    """Print a formatted results table."""
    print(f"{'Model':<22} {'MSE':>8} {'NLL':>8} {'CRPS':>8} {'ECE':>8} {'PICP@95':>8} {'MPIW@95':>8} {'Sharp':>8} {'Time':>7}")
    print("-" * 102)
    for model_name, m in results:
        mse_s = f"{m['MSE']:8.4f}"
        nll_s = f"{m.get('NLL', float('nan')):8.3f}" if "NLL" in m else f"{'--':>8}"
        crps_s = f"{m.get('CRPS', float('nan')):8.4f}" if "CRPS" in m else f"{'--':>8}"
        ece_s = f"{m.get('ECE', float('nan')):8.4f}" if "ECE" in m else f"{'--':>8}"
        picp_s = f"{m.get('PICP@95', float('nan')):8.3f}" if "PICP@95" in m else f"{'--':>8}"
        mpiw_s = f"{m.get('MPIW@95', float('nan')):8.3f}" if "MPIW@95" in m else f"{'--':>8}"
        sharp_s = f"{m.get('Sharpness', float('nan')):8.4f}" if "Sharpness" in m else f"{'--':>8}"
        time_s = f"{m.get('_time', 0):6.1f}s"
        print(f"{model_name:<22} {mse_s} {nll_s} {crps_s} {ece_s} {picp_s} {mpiw_s} {sharp_s} {time_s}")


def _multimodal_mixture_eval(results: list[tuple[str, dict]], x_test: np.ndarray, y_test: np.ndarray) -> None:
    """Evaluate mixture-capable models on multimodal data using full distributional metrics."""
    from automl_package.utils.calibration import ece_regression
    from automl_package.utils.scoring import calculate_crps

    print("\n  Multimodal mixture evaluation (models with predict_distribution):")
    print(f"  {'Model':<22} {'MixNLL':>10} {'MixCRPS':>10} {'MixECE':>10}")
    print(f"  {'-'*56}")

    for model_name, m in results:
        model = m.get("_model")
        if model is None or not hasattr(model, "predict_distribution"):
            continue
        try:
            dist = model.predict_distribution(x_test, filter_data=False)
            mix_nll = float(-np.mean(dist.log_prob(y_test)))
            mix_crps = calculate_crps(y_test, dist, n_quadrature=300)
            mix_ece = ece_regression(y_test, dist)
            print(f"  {model_name:<22} {mix_nll:10.4f} {mix_crps:10.4f} {mix_ece:10.4f}")
        except Exception as e:
            print(f"  {model_name:<22} {'FAILED':>10} -- {type(e).__name__}: {e!s:.40}")


def _save_markdown_report(ds_name: str, n_train: int, n_test: int, d: int, results: list[tuple[str, dict]], output_dir: Path) -> None:
    """Write a self-contained Markdown report for one dataset."""
    import datetime

    metric_cols = ["MSE", "NLL", "CRPS", "ECE", "PICP@95", "MPIW@95", "Sharpness", "MiscalArea", "Winkler@95"]
    header_row = "| Model | " + " | ".join(metric_cols) + " | Time |"
    # n+1 right-align columns for metric_cols + Time
    sep_row = "|:---|" + "---:|" * (len(metric_cols) + 1)

    rows = []
    for model_name, m in results:
        cells = [model_name]
        for col in metric_cols:
            v = m.get(col)
            cells.append(f"{v:.4f}" if v is not None else "--")
        cells.append(f"{m.get('_time', 0):.1f}s")
        rows.append("| " + " | ".join(cells) + " |")

    best: dict[str, tuple[str, float]] = {}
    for col in ["MSE", "NLL", "CRPS"]:
        best_name, best_val = None, float("inf")
        for model_name, m in results:
            v = m.get(col)
            if v is not None and v < best_val:
                best_val = v
                best_name = model_name
        if best_name:
            best[col] = (best_name, best_val)

    commentary_lines = []
    for col, (name, val) in best.items():
        commentary_lines.append(f"- **Best {col}**: `{name}` ({val:.4f})")

    report = f"""# Benchmark Report: {ds_name}

**Generated:** {datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

## Dataset

| Property | Value |
|:---|---:|
| Name | {ds_name} |
| Train samples | {n_train:,} |
| Test samples | {n_test:,} |
| Features | {d} |

## Results

{header_row}
{sep_row}
""" + "\n".join(rows) + f"""

## Summary

{chr(10).join(commentary_lines)}

_Timeout limit: {MODEL_TIMEOUT_SECS}s per model. TIMEOUT/FAILED models are excluded._
"""

    safe_name = ds_name.replace("(", "").replace(")", "").replace("/", "_").replace(" ", "_")
    report_path = output_dir / f"report_{safe_name}.md"
    report_path.write_text(report)


def run_benchmark() -> None:
    """Run full benchmark: all models x all datasets."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for ds_fn in ALL_DATASETS:
        ds = ds_fn()
        ds_name = ds["name"]
        x, y = ds["x"], ds["y"]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        print(f"\n{'='*80}")
        print(f"  Dataset: {ds_name} (n={len(x)}, d={x.shape[1]})")
        print(f"{'='*80}")

        models = _make_models(input_size=x.shape[1], n_samples=len(x))
        results: list[tuple[str, dict]] = []

        for model_name, model in models:
            m = _run_model(model_name, model, x_train, y_train, x_test, y_test)
            if m is not None:
                results.append((model_name, m))

        _print_table(results)

        # Mixture evaluation on multimodal dataset
        if "multimodal" in ds_name.lower():
            _multimodal_mixture_eval(results, x_test, y_test)

        _save_markdown_report(ds_name, len(x_train), len(x_test), x.shape[1], results, OUTPUT_DIR)

        print()


if __name__ == "__main__":
    run_benchmark()
