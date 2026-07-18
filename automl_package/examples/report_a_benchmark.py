"""Report (a) battery: ProbReg (fixed-k, dynamic-k) vs 4 baselines on toys (Task P2).

(`docs/plans/capacity_programme/probreg-report.md` Task P2; depends on P0/P1/S1-S3.)

Searched before writing (minimum-viable-code ladder rung 2 — reuse, don't reinvent):
`automl_package/examples/model_comparison.py` (fixed-k ProbReg builder, tree/NN baseline
builders, MSE/NLL evaluate pattern), `automl_package/examples/noise_robustness_benchmark.py`
(dynamic-k ProbReg builder: SOFT_GATING selection + ELBO regularization — CLAUDE.md's
documented best combo), `automl_package/examples/full_benchmark.py` (the toy suite:
`_heteroscedastic`/`_piecewise`/`_multimodal`/`_exponential`, confirming the April record),
`automl_package/examples/nested_width_net.py` (`make_hetero3`, the WP-3 high-noise-region
stressor). All four toy generators and the k=5 fixed-ProbReg / dynamic-ProbReg builder shapes
are IMPORTED/adapted from those modules, not reimplemented. Metrics are the canonical S1
callables (`shared/metrics-accounting.md`): `Metrics.calculate_ece()` (PIT-ECE, NOT
`calibration.py:ece_regression` — S1's named duplicate to avoid), `calculate_picp_at_alphas`
(`picp@90` is already in its default-alpha output), `calculate_mpiw(alpha=0.1)` for the 90%
pairing (not emitted by default). Stats/seed-count/convergence-flag conventions are S3's; the
bootstrap helper (`sinc_width_experiment.py::_plain_boot_se`) is left for P3's report-table
contrasts, not duplicated here — this driver's job is per-cell numbers, not aggregate verdicts.

ProbReg framing (binding on every string in this file, per the user's convention): a
CLASSIFIER over k classes with per-class regression heads, combined by the Law of Total
Variance (LTV) into one collapsed Gaussian (mu, sigma). Never "Gaussian mixture" anywhere
below. Fixed-k and dynamic-k are two DISTINCT models throughout.

Hetero-NLL protocol (P1's finding, `shared/hetero_nll_diagnosis.md`): ProbReg's mean is a
bottlenecked function of the k-way class posterior, so a coarse fixed k under-resolves the
mean on wiggly targets. The fix is not a code change — it's val-selecting k per toy from a
small grid before scoring, and evaluating the COLLAPSED Gaussian (`predict`/
`predict_uncertainty`), never the full-mixture `predict_distribution` (worse NLL, P1 Check).

Data roles (disjoint, touched once): k-selection draws its OWN dataset at a dedicated
protocol seed (`KSEL_SEED=999`, outside the 5 report seeds) and picks k from a val split of
that draw only. Each report seed then draws an INDEPENDENT toy sample (true replicate, not
just a re-split of one draw) and splits it train/test; test is scored exactly once per cell.

Convergence flagging: this driver does not re-run the heavier trajectory machinery in
`convergence.py` (built for the width/depth certification strand's single high-stakes
verdicts) — for this broad comparative battery it reads the signal `BaseModel` already
exposes after `.fit()`: `model.num_iterations_used` (the epoch/round at which the
early-stopping phase found its best validation score) against the declared training cap
(`n_epochs` for NN-family models, `n_estimators` for tree models). `hit_cap = (used >= cap)`
flags "best score coincided with running out of budget" — cheap, exact, and the same idiom
`flexnn_revalidation.py` uses (`hit_cap = len(val_loss_history) >= N_EPOCHS_CAP`). A cell with
`hit_cap=True` is still scored and recorded (never silently dropped) but flagged
`convergence_ok=False` so P3 can quarantine it from headline tables (MASTER Decision 9 / S3).

Tree baselines: XGBoost/LightGBM/CatBoost all use `UncertaintyMethod.BINNED_RESIDUAL_STD`
uniformly (an existing, non-improvised uncertainty path already wired into `BaseModel` via
`BinnedUncertaintyMixin`) rather than a mix of CONSTANT/PROBABILISTIC per family — this keeps
the three tree baselines apples-to-apples. Note for the report (P0's N5 flag): LightGBM's
`UncertaintyMethod.PROBABILISTIC` path (`tree_model_gaussian_nll_objective`, confirmed FIXED
at HEAD) is available but NOT used here; BINNED_RESIDUAL_STD was chosen for tree-family
uniformity, not because the PROBABILISTIC path is broken.

Usage:
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=2 ~/dev/.venv/bin/python -m automl_package.examples.report_a_benchmark --smoke
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=2 ~/dev/.venv/bin/python -m automl_package.examples.report_a_benchmark --toys heteroscedastic,hetero3
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=2 ~/dev/.venv/bin/python -m automl_package.examples.report_a_benchmark   # full battery, all toys
"""

from __future__ import annotations

import argparse
import enum
import json
import logging
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from automl_package.enums import (
    NClassesRegularization,
    NClassesSelectionMethod,
    RegressionStrategy,
    TaskType,
    UncertaintyMethod,
)
from automl_package.examples.full_benchmark import _exponential, _heteroscedastic, _multimodal, _piecewise
from automl_package.examples.nested_width_net import make_hetero3
from automl_package.models.catboost_model import CatBoostModel
from automl_package.models.lightgbm_model import LightGBMModel
from automl_package.models.neural_network import PyTorchNeuralNetwork
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel
from automl_package.models.xgboost_model import XGBoostModel
from automl_package.utils.calibration import calculate_mpiw, calibration_curve_gaussian
from automl_package.utils.metrics import Metrics, calculate_nll
from automl_package.utils.scoring import calculate_crps_gaussian, calculate_winkler_from_gaussian

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

RESULTS_DIR = Path(__file__).parent / "report_a_results"

REPORT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
KSEL_SEED = 999  # dedicated k-selection protocol seed, disjoint from REPORT_SEEDS (P1's protocol)
K_GRID: tuple[int, ...] = (5, 8, 10, 12)  # P1's pre-registered small grid


class Toy(enum.Enum):
    """The report's toy suite (closed set — home-turf = heteroscedastic/bimodal/high-sigma)."""

    HETEROSCEDASTIC = "heteroscedastic"
    PIECEWISE = "piecewise"
    MULTIMODAL = "multimodal"
    EXPONENTIAL = "exponential"
    HETERO3 = "hetero3"


HOME_TURF_TOYS = (Toy.HETEROSCEDASTIC, Toy.MULTIMODAL, Toy.HETERO3)
STANDARD_TOYS = (Toy.PIECEWISE, Toy.EXPONENTIAL)
ALL_TOYS = (*HOME_TURF_TOYS, *STANDARD_TOYS)

PRODUCTION_N: dict[Toy, int] = {
    Toy.HETEROSCEDASTIC: 1000,
    Toy.PIECEWISE: 800,
    Toy.MULTIMODAL: 1000,
    Toy.EXPONENTIAL: 800,
    Toy.HETERO3: 900,  # 600*3/2: keeps per-region density comparable to the other toys (make_hetero3 docstring)
}
SMOKE_N = 150  # uniform tiny N for the pipeline-proof smoke run


class ModelKey(enum.Enum):
    """The report's 6-model battery (closed set)."""

    PROBREG_FIXED_K = "probreg_fixed_k"
    PROBREG_DYNAMIC_K = "probreg_dynamic_k"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    NN_CONSTANT = "nn_constant"


MODEL_KEYS: tuple[ModelKey, ...] = tuple(ModelKey)


@dataclass(frozen=True)
class RunConfig:
    """Training-budget knobs, scaled down for `--smoke`."""

    n_epochs: int = 100
    n_estimators: int = 300


FULL_CONFIG = RunConfig(n_epochs=100, n_estimators=300)
SMOKE_CONFIG = RunConfig(n_epochs=8, n_estimators=20)


# ---------------------------------------------------------------------------
# Toy dispatch (reuses `full_benchmark.py`'s generators + `nested_width_net.make_hetero3`)
# ---------------------------------------------------------------------------


def make_dataset(toy: Toy, seed: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Draws one (x, y) sample for `toy` at `seed`. `x` is always shape (n, 1), `y` shape (n,)."""
    if toy is Toy.HETEROSCEDASTIC:
        d = _heteroscedastic(n=n, seed=seed)
        return d["x"], d["y"]
    if toy is Toy.PIECEWISE:
        d = _piecewise(n=n, seed=seed)
        return d["x"], d["y"]
    if toy is Toy.MULTIMODAL:
        d = _multimodal(n=n, seed=seed)
        return d["x"], d["y"]
    if toy is Toy.EXPONENTIAL:
        d = _exponential(n=n, seed=seed)
        return d["x"], d["y"]
    if toy is Toy.HETERO3:
        x, y, _region = make_hetero3(n=n, seed=seed)
        return x.reshape(-1, 1), y
    raise ValueError(f"unhandled toy: {toy}")


# ---------------------------------------------------------------------------
# Model builders — adapted from model_comparison.py / noise_robustness_benchmark.py
# ---------------------------------------------------------------------------


def _probreg_fixed(input_size: int, k: int, seed: int, cfg: RunConfig) -> ProbabilisticRegressionModel:
    """Fixed n_classes ProbReg (`model_comparison.py::build_prob_regression`, adapted)."""
    return ProbabilisticRegressionModel(
        input_size=input_size,
        n_classes=k,
        uncertainty_method=UncertaintyMethod.PROBABILISTIC,
        n_classes_selection_method=NClassesSelectionMethod.NONE,
        regression_strategy=RegressionStrategy.SEPARATE_HEADS,
        base_classifier_params={"hidden_layers": 1, "hidden_size": 64},
        regression_head_params={"hidden_layers": 0, "hidden_size": 32},
        n_epochs=cfg.n_epochs,
        learning_rate=0.01,
        early_stopping_rounds=15,
        validation_fraction=0.2,
        random_seed=seed,
        calculate_feature_importance=False,
    )


def _probreg_dynamic(input_size: int, seed: int, cfg: RunConfig, max_k: int = 12) -> ProbabilisticRegressionModel:
    """Dynamic n_classes ProbReg, ELBO + SoftGating (`noise_robustness_benchmark.py::build_probreg_dyn_k`)."""
    return ProbabilisticRegressionModel(
        input_size=input_size,
        n_classes=3,
        max_n_classes_for_probabilistic_path=max_k,
        uncertainty_method=UncertaintyMethod.PROBABILISTIC,
        n_classes_selection_method=NClassesSelectionMethod.SOFT_GATING,
        n_classes_regularization=NClassesRegularization.ELBO,
        regression_strategy=RegressionStrategy.SEPARATE_HEADS,
        base_classifier_params={"hidden_layers": 1, "hidden_size": 64},
        regression_head_params={"hidden_layers": 0, "hidden_size": 32},
        n_epochs=cfg.n_epochs,
        learning_rate=0.01,
        early_stopping_rounds=15,
        validation_fraction=0.2,
        random_seed=seed,
        calculate_feature_importance=False,
    )


def _nn_constant(input_size: int, seed: int, cfg: RunConfig) -> PyTorchNeuralNetwork:
    """Constant-variance plain NN baseline (`model_comparison.py::build_nn_constant`)."""
    return PyTorchNeuralNetwork(
        input_size=input_size,
        output_size=1,
        hidden_layers=2,
        hidden_size=64,
        uncertainty_method=UncertaintyMethod.CONSTANT,
        learning_rate=0.01,
        n_epochs=cfg.n_epochs,
        early_stopping_rounds=15,
        validation_fraction=0.2,
        random_seed=seed,
        calculate_feature_importance=False,
    )


def _xgboost(seed: int, cfg: RunConfig) -> XGBoostModel:
    return XGBoostModel(
        n_estimators=cfg.n_estimators,
        learning_rate=0.05,
        uncertainty_method=UncertaintyMethod.BINNED_RESIDUAL_STD,
        early_stopping_rounds=15,
        validation_fraction=0.2,
        random_seed=seed,
        calculate_feature_importance=False,
    )


def _lightgbm(seed: int, cfg: RunConfig) -> LightGBMModel:
    return LightGBMModel(
        n_estimators=cfg.n_estimators,
        learning_rate=0.05,
        uncertainty_method=UncertaintyMethod.BINNED_RESIDUAL_STD,
        early_stopping_rounds=15,
        validation_fraction=0.2,
        random_seed=seed,
        calculate_feature_importance=False,
    )


def _catboost(seed: int, cfg: RunConfig) -> CatBoostModel:
    return CatBoostModel(
        iterations=cfg.n_estimators,
        learning_rate=0.05,
        uncertainty_method=UncertaintyMethod.BINNED_RESIDUAL_STD,
        verbose=False,
        early_stopping_rounds=15,
        validation_fraction=0.2,
        random_seed=seed,
        calculate_feature_importance=False,
    )


def build_model(model_key: ModelKey, input_size: int, seed: int, cfg: RunConfig, k_for_toy: int) -> tuple[Any, int]:
    """Returns (model, training_cap) — cap is what `num_iterations_used` is compared against for hit_cap."""
    if model_key is ModelKey.PROBREG_FIXED_K:
        return _probreg_fixed(input_size, k_for_toy, seed, cfg), cfg.n_epochs
    if model_key is ModelKey.PROBREG_DYNAMIC_K:
        return _probreg_dynamic(input_size, seed, cfg), cfg.n_epochs
    if model_key is ModelKey.NN_CONSTANT:
        return _nn_constant(input_size, seed, cfg), cfg.n_epochs
    if model_key is ModelKey.XGBOOST:
        return _xgboost(seed, cfg), cfg.n_estimators
    if model_key is ModelKey.LIGHTGBM:
        return _lightgbm(seed, cfg), cfg.n_estimators
    if model_key is ModelKey.CATBOOST:
        return _catboost(seed, cfg), cfg.n_estimators
    raise ValueError(f"unhandled model_key: {model_key}")


# ---------------------------------------------------------------------------
# Metrics (canonical S1 callables only)
# ---------------------------------------------------------------------------


def compute_cell_metrics(x_test: np.ndarray, y_test: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray | None, model_name: str) -> dict[str, float | None]:
    """MSE always; NLL/PIT-ECE/coverage/width only if the model has an uncertainty path (else explicit None -> '-')."""
    y_test = np.asarray(y_test).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mse = float(np.mean((y_test - y_pred) ** 2))
    if y_std is None:
        return {"mse": mse, "nll": None, "pit_ece": None, "picp@90": None, "mpiw@90": None,
                "crps": None, "winkler@90": None, "reliability_curve": None}

    y_std = np.asarray(y_std).ravel()
    m = Metrics(task_type=TaskType.REGRESSION, model_name=model_name, x_data=x_test, y_true=y_test, y_pred=y_pred, y_std=y_std)
    reg = m.calculate_regression_metrics()
    rel_target, rel_empirical = calibration_curve_gaussian(y_test, y_pred, y_std, n_bins=20)
    return {
        "mse": mse,
        "nll": reg["nll"],
        "pit_ece": reg["ece"],
        "picp@90": reg["picp@90"],
        "mpiw@90": calculate_mpiw(y_std, alpha=0.1),
        "crps": calculate_crps_gaussian(y_test, y_pred, y_std),  # proper score, whole distribution, target units
        "winkler@90": calculate_winkler_from_gaussian(y_test, y_pred, y_std, alpha=0.1),  # interval score: width + miss penalty
        "reliability_curve": {"target": rel_target.tolist(), "empirical": rel_empirical.tolist()},  # for the reliability diagram
    }


# ---------------------------------------------------------------------------
# k-selection (P1's protocol fix: val-select k per toy from K_GRID, one round per toy)
# ---------------------------------------------------------------------------


def select_k_for_toy(toy: Toy, cfg: RunConfig, n: int) -> dict[str, Any]:
    """Fits ProbReg fixed-k at each K_GRID candidate on a dedicated protocol-seed draw; picks argmin val NLL."""
    x, y = make_dataset(toy, KSEL_SEED, n)
    x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=0.3, random_state=KSEL_SEED)
    input_size = x.shape[1]

    val_nll: dict[int, float] = {}
    for k in K_GRID:
        model = _probreg_fixed(input_size, k, KSEL_SEED, cfg)
        model.fit(x_tr, y_tr)
        y_pred = model.predict(x_val)
        y_std = model.predict_uncertainty(x_val)
        val_nll[k] = calculate_nll(y_val, y_pred, y_std)
        logger.info(f"[k-select {toy.value}] k={k} val_nll={val_nll[k]:.4f}")

    selected_k = min(val_nll, key=val_nll.get)
    return {
        "toy": toy.value,
        "protocol_seed": KSEL_SEED,
        "k_grid": list(K_GRID),
        "val_nll_by_k": val_nll,
        "selected_k": selected_k,
        "n_samples": n,
        "n_epochs_cap": cfg.n_epochs,
    }


# ---------------------------------------------------------------------------
# One cell: fit one model on one (toy, seed), score once on held-out test
# ---------------------------------------------------------------------------


def run_cell(toy: Toy, model_key: ModelKey, seed: int, cfg: RunConfig, n: int, k_for_toy: int) -> dict[str, Any]:
    """Fits one model on one (toy, seed) draw and scores it once on a held-out test split."""
    group = "home_turf" if toy in HOME_TURF_TOYS else "standard"
    cell: dict[str, Any] = {
        "toy": toy.value,
        "group": group,
        "model": model_key.value,
        "seed": seed,
        "n_samples": n,
        "k_for_toy": k_for_toy if model_key is ModelKey.PROBREG_FIXED_K else None,
    }
    try:
        x, y = make_dataset(toy, seed, n)
        x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.3, random_state=seed)
        input_size = x.shape[1]

        model, cap = build_model(model_key, input_size, seed, cfg, k_for_toy)
        t0 = time.perf_counter()
        model.fit(x_tr, y_tr)
        elapsed = time.perf_counter() - t0

        y_pred = model.predict(x_te)
        try:
            y_std = model.predict_uncertainty(x_te)
        except (NotImplementedError, RuntimeError, AttributeError, TypeError):
            y_std = None

        metrics = compute_cell_metrics(x_te, y_te, y_pred, y_std, model_key.value)
        used = int(getattr(model, "num_iterations_used", 0) or 0)
        hit_cap = used >= cap
        cell.update(
            {
                "status": "ok",
                "seconds": elapsed,
                "training_cap": cap,
                "iterations_used": used,
                "hit_cap": hit_cap,
                "convergence_ok": not hit_cap,
                **metrics,
            }
        )
        logger.info(
            f"[{toy.value}/{model_key.value}/seed{seed}] mse={metrics['mse']:.4f} "
            f"nll={metrics['nll']} hit_cap={hit_cap} t={elapsed:.1f}s"
        )
    except Exception as e:  # land the failure, never lose the whole run to one bad cell
        cell.update({"status": "failed", "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()})
        logger.exception(f"[{toy.value}/{model_key.value}/seed{seed}] FAILED: {e}")
    return cell


def _save(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parses CLI args (`--smoke`, `--toys`, `--seeds`, `--skip-kselect`)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="Tiny config (1 toy default, 2 seeds, few epochs/estimators, small N); files prefixed smoke__.")
    parser.add_argument("--toys", type=str, default=None, help="Comma-separated toy names (default: all 5).")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds (default: 0,1,2,3,4; smoke default: 0,1).")
    parser.add_argument("--skip-kselect", action="store_true", help="Reuse an existing k_selection__<toy>.json instead of recomputing.")
    return parser.parse_args()


def main() -> None:
    """Runs the report (a) battery: k-selection per toy, then {6 models} x {toy suite} x seeds."""
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        cfg = SMOKE_CONFIG
        toys = (Toy.HETEROSCEDASTIC,) if args.toys is None else tuple(Toy(t) for t in args.toys.split(","))
        seeds = (0, 1) if args.seeds is None else tuple(int(s) for s in args.seeds.split(","))
        n_for = lambda _toy: SMOKE_N  # noqa: E731
        prefix = "smoke__"
    else:
        cfg = FULL_CONFIG
        toys = ALL_TOYS if args.toys is None else tuple(Toy(t) for t in args.toys.split(","))
        seeds = REPORT_SEEDS if args.seeds is None else tuple(int(s) for s in args.seeds.split(","))
        n_for = lambda toy: PRODUCTION_N[toy]  # noqa: E731
        prefix = ""

    for toy in toys:
        n = n_for(toy)
        kfile = RESULTS_DIR / f"{prefix}k_selection__{toy.value}.json"
        if args.skip_kselect and kfile.exists():
            k_info = json.loads(kfile.read_text())
        else:
            k_info = select_k_for_toy(toy, cfg, n)
            _save(kfile, k_info)  # land immediately — standing clause (a)
        k_star = k_info["selected_k"]
        logger.info(f"[{toy.value}] selected k={k_star} (grid={K_GRID})")

        for seed in seeds:
            for model_key in MODEL_KEYS:
                cell = run_cell(toy, model_key, seed, cfg, n, k_star)
                out_path = RESULTS_DIR / f"{prefix}{toy.value}__{model_key.value}__seed{seed}.json"
                _save(out_path, cell)  # land immediately — standing clause (a)


if __name__ == "__main__":
    main()
