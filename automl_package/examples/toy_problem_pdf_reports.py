"""Per-toy-problem PDF report generator.

For each toy dataset (heteroscedastic, bimodal, piecewise, exponential), trains
a consistent panel of models and produces a single self-contained PDF that
bundles, in order:

  1. Dataset overview + data scatter.
  2. Metrics comparison table (MSE, NLL, CRPS, PICP@95, sharpness).
  3. Per-model predicted-vs-actual scatter and residuals.
  4. ProbReg classifier-probability and regression-head-output plots.
  5. ClassReg probability-mapper plot.
  6. FlexNN per-input selected-depth heatmap.
  7. Short commentary placeholder.

Each PDF lives at ``toy_problem_reports/<dataset>.pdf`` and is designed to be
flippable by a reviewer without opening source / CSVs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split

from automl_package.enums import (
    DepthRegularization,
    LayerSelectionMethod,
    MapperType,
    NClassesSelectionMethod,
    RegressionStrategy,
    UncertaintyMethod,
)
from automl_package.models.classifier_regression import ClassifierRegressionModel
from automl_package.models.flexible_neural_network import FlexibleHiddenLayersNN
from automl_package.models.neural_network import PyTorchNeuralNetwork
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel
from automl_package.models.xgboost_model import XGBoostModel
from automl_package.utils.calibration import calculate_mpiw, calculate_picp_at_alphas, calculate_sharpness_from_std
from automl_package.utils.distributions import GaussianDistribution
from automl_package.utils.scoring import calculate_crps_gaussian

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

OUT_DIR = Path(__file__).parent / "toy_problem_reports"
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


@dataclass
class ToyDataset:
    name: str
    description: str
    x: np.ndarray
    y: np.ndarray


def _heteroscedastic() -> ToyDataset:
    rng = np.random.default_rng(42)
    x = rng.uniform(-5, 5, 800).reshape(-1, 1).astype(np.float32)
    y = (np.sin(x.ravel()) * 2 + 0.5 * x.ravel()
         + rng.normal(0.0, 0.1 + 0.4 * np.abs(x.ravel()))).astype(np.float32)
    return ToyDataset(
        name="heteroscedastic",
        description=(
            "y = 2*sin(x) + 0.5*x + Normal(0, 0.1 + 0.4|x|); n=800, d=1. "
            "Noise grows linearly with |x| — tests heteroscedastic uncertainty."
        ),
        x=x, y=y,
    )


def _bimodal() -> ToyDataset:
    rng = np.random.default_rng(42)
    x = rng.uniform(-3, 3, 800).reshape(-1, 1).astype(np.float32)
    sign = rng.choice([-1.0, 1.0], size=800).reshape(-1, 1)
    y = (x + sign * 1.5 + rng.normal(0, 0.1, (800, 1))).ravel().astype(np.float32)
    return ToyDataset(
        name="bimodal",
        description=(
            "y = x + sign*1.5 + Normal(0, 0.1); n=800, d=1, sign in {-1, +1}. "
            "Bimodal posterior per input; tests classification bottleneck."
        ),
        x=x, y=y,
    )


def _piecewise() -> ToyDataset:
    rng = np.random.default_rng(42)
    x = rng.uniform(-5.0, 5.0, 800).reshape(-1, 1).astype(np.float32)
    y_true = np.where(x.ravel() < 0, 0.5 * x.ravel(), 0.5 * x.ravel() + np.sin(4 * np.pi * x.ravel()))
    y = (y_true + rng.normal(0, 0.2, 800)).astype(np.float32)
    return ToyDataset(
        name="piecewise",
        description=(
            "y = 0.5x for x<0 else 0.5x + sin(4*pi*x); Normal(0, 0.2) noise; n=800. "
            "Linear regime left, oscillatory regime right; tests FlexNN depth adaptation."
        ),
        x=x, y=y,
    )


def _exponential() -> ToyDataset:
    rng = np.random.default_rng(42)
    x = rng.uniform(-3, 3, 600).reshape(-1, 1).astype(np.float32)
    y = (np.exp(x.ravel()) + rng.normal(0.0, 0.5, 600)).astype(np.float32)
    return ToyDataset(
        name="exponential",
        description=(
            "y = exp(x) + Normal(0, 0.5); n=600, d=1. "
            "Wide target range tests target_transform='symlog'."
        ),
        x=x, y=y,
    )


DATASETS: list[Callable[[], ToyDataset]] = [_heteroscedastic, _bimodal, _piecewise, _exponential]


# ---------------------------------------------------------------------------
# Model panel
# ---------------------------------------------------------------------------


def _build_panel(input_size: int) -> list[tuple[str, object]]:
    common_nn = dict(
        input_size=input_size, learning_rate=0.01, n_epochs=80,
        early_stopping_rounds=15, validation_fraction=0.2,
        random_seed=42, calculate_feature_importance=False,
    )
    return [
        ("XGBoost", XGBoostModel(n_estimators=300, early_stopping_rounds=15,
                                 random_seed=42, calculate_feature_importance=False)),
        ("NN(PROB)", PyTorchNeuralNetwork(
            output_size=1, hidden_layers=2, hidden_size=64,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC, **common_nn,
        )),
        ("ProbReg(k=3,SEP)", ProbabilisticRegressionModel(
            n_classes=3, uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.NONE,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            base_classifier_params={"hidden_layers": 1, "hidden_size": 64},
            regression_head_params={"hidden_layers": 0, "hidden_size": 32},
            **common_nn,
        )),
        ("ClassReg(k=7)", ClassifierRegressionModel(
            base_classifier_class=PyTorchNeuralNetwork, n_classes=7,
            mapper_type=MapperType.LOOKUP_MEDIAN,
            base_classifier_params={"input_size": input_size, "hidden_layers": 2,
                                    "hidden_size": 64, "learning_rate": 0.01, "n_epochs": 80},
            early_stopping_rounds=15, uncertainty_method=UncertaintyMethod.BINNED_RESIDUAL_STD,
            validation_fraction=0.2, random_seed=42, calculate_feature_importance=False,
        )),
        ("FlexNN(ELBO,PROB)", FlexibleHiddenLayersNN(
            max_hidden_layers=5, hidden_size=64, n_predictor_layers=1,
            layer_selection_method=LayerSelectionMethod.SOFT_GATING,
            depth_regularization=DepthRegularization.ELBO,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC, **common_nn,
        )),
    ]


# ---------------------------------------------------------------------------
# Metrics + PDF pages
# ---------------------------------------------------------------------------


def _eval_model(model, x_te, y_te) -> dict:
    y_pred = np.asarray(model.predict(x_te)).ravel()
    result = {"y_pred": y_pred, "mse": float(np.mean((y_te - y_pred) ** 2))}
    try:
        y_std = np.asarray(model.predict_uncertainty(x_te)).ravel()
        y_std = np.maximum(y_std, 1e-9)
        result["y_std"] = y_std
        result["nll"] = float(np.mean(0.5 * (np.log(2 * np.pi * y_std ** 2) + ((y_te - y_pred) / y_std) ** 2)))
        result["crps"] = float(calculate_crps_gaussian(y_te, y_pred, y_std))
        picp = calculate_picp_at_alphas(y_te, y_pred, y_std, alphas=(0.05,))
        result["picp95"] = picp["picp@95"]
        result["mpiw95"] = float(calculate_mpiw(y_std, alpha=0.05))
        result["sharpness"] = float(calculate_sharpness_from_std(y_std))
    except Exception:  # noqa: BLE001
        pass
    return result


def _plot_dataset_overview(ds: ToyDataset, pdf: PdfPages) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.scatter(ds.x.ravel(), ds.y, s=6, alpha=0.5, color="black")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"Dataset: {ds.name}  (n={len(ds.y)})")
    ax.grid(True, alpha=0.3)
    ax.text(
        0.02, 0.02, ds.description,
        transform=ax.transAxes, fontsize=9, verticalalignment="bottom",
        bbox={"facecolor": "white", "alpha": 0.85, "pad": 4},
        wrap=True,
    )
    pdf.savefig(fig); plt.close(fig)


def _plot_metrics_table(metrics: dict[str, dict], pdf: PdfPages) -> None:
    cols = ["mse", "nll", "crps", "picp95", "mpiw95", "sharpness"]
    rows = []
    for name, m in metrics.items():
        rows.append([name] + [f"{m.get(c, float('nan')):.4f}" if c in m else "--" for c in cols])
    df = pd.DataFrame(rows, columns=["model", *cols])

    fig, ax = plt.subplots(figsize=(8.5, 0.5 + 0.3 * (len(df) + 2)))
    ax.axis("off")
    ax.set_title("Model comparison", fontsize=12, pad=12)
    table = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="right")
    for (r, c), cell in table.get_celld().items():
        if c == 0:
            cell.set_text_props(ha="left")
        if r == 0:
            cell.set_text_props(weight="bold")
    table.auto_set_font_size(False); table.set_fontsize(9)
    table.scale(1.0, 1.3)
    pdf.savefig(fig); plt.close(fig)


def _plot_predicted_vs_actual(ds: ToyDataset, x_te, y_te, metrics: dict, pdf: PdfPages) -> None:
    n = len(metrics)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8.5, 3.2 * rows), squeeze=False)
    axes = axes.ravel()
    for ax, (name, m) in zip(axes, metrics.items()):
        y_pred = m["y_pred"]
        ax.scatter(y_te, y_pred, s=4, alpha=0.4)
        lim = [min(y_te.min(), y_pred.min()), max(y_te.max(), y_pred.max())]
        ax.plot(lim, lim, color="red", linestyle="--", linewidth=1)
        ax.set_title(f"{name}  MSE={m['mse']:.3f}", fontsize=10)
        ax.set_xlabel("y_true"); ax.set_ylabel("y_pred")
        ax.grid(True, alpha=0.3)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"Predicted vs actual — {ds.name}")
    fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)


def _plot_uncertainty_intervals(ds: ToyDataset, x_te, y_te, metrics: dict, pdf: PdfPages) -> None:
    n = sum(1 for m in metrics.values() if "y_std" in m)
    if n == 0:
        return
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8.5, 3.2 * rows), squeeze=False)
    axes = axes.ravel()
    order = np.argsort(x_te.ravel())
    i = 0
    for name, m in metrics.items():
        if "y_std" not in m:
            continue
        ax = axes[i]; i += 1
        xs = x_te.ravel()[order]
        ys = y_te[order]
        yp = m["y_pred"][order]
        sig = m["y_std"][order]
        ax.scatter(xs, ys, s=4, alpha=0.3, color="black", label="y_true")
        ax.plot(xs, yp, color="C0", linewidth=1, label="mean")
        ax.fill_between(xs, yp - 1.96 * sig, yp + 1.96 * sig, alpha=0.25, color="C0", label="95%")
        ax.set_title(f"{name}  PICP@95={m.get('picp95', float('nan')):.2f}", fontsize=10)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.grid(True, alpha=0.3)
        if i == 1:
            ax.legend(fontsize=8)
    for ax in axes[i:]:
        ax.axis("off")
    fig.suptitle(f"Prediction intervals — {ds.name}")
    fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)


def _plot_probreg_internal(model: ProbabilisticRegressionModel, pdf: PdfPages) -> None:
    """Classifier probabilities vs x + regression head outputs vs probability."""
    if model.n_classes_selection_method != NClassesSelectionMethod.NONE:
        return  # dynamic-k doesn't expose per-head outputs in a straightforward way
    if model.regression_strategy == RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT:
        return

    rng = np.random.default_rng(0)
    x_eval = np.linspace(rng.uniform(-5, -3), rng.uniform(3, 5), 400).reshape(-1, 1).astype(np.float32)
    x_t = torch.tensor(x_eval, dtype=torch.float32).to(model.device)
    model.model.eval()
    with torch.no_grad():
        classifier_raw_logits = model.model.classifier_layers(x_t)
        masked = torch.full_like(classifier_raw_logits, float("-inf"))
        masked[:, : model.n_classes] = classifier_raw_logits[:, : model.n_classes]
        probs = torch.softmax(masked, dim=1)[:, : model.n_classes].cpu().numpy()
        _, per_head = model.model.regression_module(torch.softmax(masked, dim=1), return_head_outputs=True)
        per_head = per_head[:, : model.n_classes, 0].cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    x_flat = x_eval.ravel()
    for i in range(model.n_classes):
        axes[0].plot(x_flat, probs[:, i], label=f"class {i}")
    axes[0].set_title("Classifier p(class | x)")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("p")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    for i in range(model.n_classes):
        order = np.argsort(probs[:, i])
        axes[1].plot(probs[order, i], per_head[order, i], label=f"head {i}", alpha=0.7)
    axes[1].set_title("Regression head mean vs class prob")
    axes[1].set_xlabel("p(class=i)"); axes[1].set_ylabel("head mean")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    fig.suptitle("ProbReg internal probabilities + head outputs")
    fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)


def _plot_classreg_mapper(model: ClassifierRegressionModel, ds: ToyDataset, pdf: PdfPages) -> None:
    if not model.class_mappers:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    probas_range = np.linspace(0, 1, 100).reshape(-1, 1)
    for i, mapper in enumerate(model.class_mappers):
        if mapper is None:
            continue
        mapped_values = mapper.predict(probas_range)
        lower = model.class_boundaries[i]
        upper = model.class_boundaries[i + 1]
        ax.plot(probas_range, mapped_values, label=f"class {i} [{lower:.2f}, {upper:.2f}]")
    ax.set_xlabel("p(class=i | x)")
    ax.set_ylabel("mapped regression value")
    ax.set_title(f"ClassReg probability mappers — {ds.name}")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    pdf.savefig(fig); plt.close(fig)


def _plot_flexnn_depth(model: FlexibleHiddenLayersNN, x_te: np.ndarray, ds: ToyDataset, pdf: PdfPages) -> None:
    x_t = torch.tensor(x_te, dtype=torch.float32).to(model.device)
    model.model.eval()
    with torch.no_grad():
        out = model.model(x_t)
    n_probs = out[3]
    if n_probs is None:
        return
    soft = n_probs.cpu().numpy()
    argmax = soft.argmax(axis=1) + 1

    if x_te.shape[1] == 1:
        order = np.argsort(x_te.ravel())
        xs = x_te.ravel()[order]
        soft_s = soft[order]
        argmax_s = argmax[order]

        fig, axes = plt.subplots(2, 1, figsize=(8.5, 5.5), sharex=True)
        axes[0].plot(xs, argmax_s, color="C1")
        axes[0].set_ylabel("argmax depth")
        axes[0].set_title(f"FlexNN selected depth — {ds.name}")
        axes[0].tick_params(labelbottom=True)
        axes[0].grid(True, alpha=0.3)
        im = axes[1].imshow(soft_s.T, aspect="auto", origin="lower",
                             extent=[xs[0], xs[-1], 0.5, soft_s.shape[1] + 0.5],
                             cmap="viridis")
        axes[1].set_xlabel("x"); axes[1].set_ylabel("depth")
        axes[1].set_title("Soft depth probabilities")
        fig.colorbar(im, ax=axes[1], shrink=0.7)
        fig.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        ax.hist(argmax, bins=np.arange(0.5, model.max_hidden_layers + 1.5, 1.0))
        ax.set_xlabel("argmax depth"); ax.set_ylabel("count")
        ax.set_title(f"FlexNN depth histogram — {ds.name}")
        fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)


def _commentary_page(ds: ToyDataset, metrics: dict, pdf: PdfPages) -> None:
    # Rank models by MSE for commentary.
    ranked = sorted(metrics.items(), key=lambda kv: kv[1].get("mse", float("inf")))
    lines = [f"Ranked MSE for {ds.name}:"]
    for name, m in ranked:
        lines.append(f"  - {name}: MSE={m.get('mse', float('nan')):.4f}  NLL={m.get('nll', float('nan')):.3f}  PICP95={m.get('picp95', float('nan')):.2f}")
    # Qualitative notes per dataset
    notes = {
        "heteroscedastic": (
            "Heteroscedastic sin: ProbReg / FlexNN(PROB) should dominate on NLL because "
            "the noise scales with |x|; a constant-variance NN will over/under-cover."
        ),
        "bimodal": (
            "Bimodal: y = x +/- 1.5 has a genuinely two-modal posterior. ClassReg (k even) "
            "or ProbReg SEP (k=2) should win on NLL by modelling each branch separately."
        ),
        "piecewise": (
            "Piecewise linear/oscillatory: FlexNN should use shallow depth in the linear "
            "regime (x<0) and deeper in the oscillatory regime (x>0)."
        ),
        "exponential": (
            "Exponential targets span ~20x; symlog transform and probabilistic uncertainty "
            "should both be beneficial."
        ),
    }
    lines.append("\nNotes:")
    lines.append(notes.get(ds.name, ""))

    fig = plt.figure(figsize=(8.5, 5))
    fig.text(0.05, 0.95, "\n".join(lines), fontsize=10, verticalalignment="top", family="monospace")
    fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_pdf_for(ds: ToyDataset) -> Path:
    logger.info(f"=== {ds.name} ===")
    x_tr, x_te, y_tr, y_te = train_test_split(ds.x, ds.y, test_size=0.3, random_state=0)
    panel = _build_panel(input_size=ds.x.shape[1])

    # Train all models, capture metrics + refs
    metrics: dict[str, dict] = {}
    model_refs: dict[str, object] = {}
    for name, model in panel:
        try:
            model.fit(x_tr, y_tr)
            metrics[name] = _eval_model(model, x_te, y_te)
            model_refs[name] = model
            logger.info(f"{name:<20} mse={metrics[name]['mse']:.4f}")
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"{name} failed: {exc}")
            metrics[name] = {"mse": float("nan"), "error": str(exc)}

    out_path = OUT_DIR / f"{ds.name}.pdf"
    with PdfPages(out_path) as pdf:
        _plot_dataset_overview(ds, pdf)
        _plot_metrics_table(metrics, pdf)
        _plot_predicted_vs_actual(ds, x_te, y_te, metrics, pdf)
        _plot_uncertainty_intervals(ds, x_te, y_te, metrics, pdf)
        if "ProbReg(k=3,SEP)" in model_refs:
            _plot_probreg_internal(model_refs["ProbReg(k=3,SEP)"], pdf)
        if "ClassReg(k=7)" in model_refs:
            _plot_classreg_mapper(model_refs["ClassReg(k=7)"], ds, pdf)
        if "FlexNN(ELBO,PROB)" in model_refs:
            _plot_flexnn_depth(model_refs["FlexNN(ELBO,PROB)"], x_te, ds, pdf)
        _commentary_page(ds, metrics, pdf)
    logger.info(f"Wrote {out_path}")
    return out_path


def main() -> None:
    for fn in DATASETS:
        ds = fn()
        build_pdf_for(ds)


if __name__ == "__main__":
    main()
