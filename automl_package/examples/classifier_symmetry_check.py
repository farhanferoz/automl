"""Classifier symmetry diagnostic.

Checks whether the per-class head output vs class probability shows the expected
mirror-symmetry pattern:
  - Symmetric class-pairs (0 & k-1, 1 & k-2, ...) produce mirror-image curves.
  - Their intersection points fall near the middle-class centroid.

Plot per panel:
  x-axis: p(class=i | x)           (class probability as x sweeps input space)
  y-axis: head output for class i   (h_i(p_i) — or centroid_i for ClassReg)

For ClassReg LOOKUP_MEDIAN: "head output" = constant centroid_i (horizontal lines).
For ProbReg SEPARATE_HEADS: "head output" = h_i(p_i) — a smooth function since
  each SEP_HEAD takes only p_i as input (single-valued, no fan artifact).

Sorting trick: sort eval points by p_i then plot (p_i[sorted], h_i[sorted]).

PDF layout:
  4 pages (one per dataset), each a 2×2 grid:
    rows: ClassReg LOOKUP_MEDIAN (top), ProbReg SEP (bottom)
    cols: k=3 (left), k=5 (right)

Output: classifier_symmetry_results/
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split

from automl_package.enums import MapperType, RegressionStrategy, UncertaintyMethod
from automl_package.models.classifier_regression import ClassifierRegressionModel
from automl_package.models.neural_network import PyTorchNeuralNetwork
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel
from automl_package.utils.numerics import create_bins

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

OUT_DIR = Path(__file__).parent / "classifier_symmetry_results"
OUT_DIR.mkdir(exist_ok=True)

K_VALUES = (3, 5)
N_EVAL = 400


def _make_datasets() -> list[tuple[str, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(42)

    x = rng.uniform(-5, 5, 800).reshape(-1, 1).astype(np.float32)
    y = (np.sin(x.ravel()) * 2 + 0.5 * x.ravel() + rng.normal(0.0, 0.1 + 0.4 * np.abs(x.ravel()))).astype(np.float32)
    datasets = [("heteroscedastic", x, y)]

    x = rng.uniform(-3, 3, 800).reshape(-1, 1).astype(np.float32)
    sign = rng.choice([-1.0, 1.0], size=800).reshape(-1, 1)
    y = (x + sign * 1.5 + rng.normal(0, 0.1, (800, 1))).ravel().astype(np.float32)
    datasets.append(("bimodal", x, y))

    x = rng.uniform(-5.0, 5.0, 800).reshape(-1, 1).astype(np.float32)
    y_true = np.where(x.ravel() < 0, 0.5 * x.ravel(), 0.5 * x.ravel() + np.sin(4 * np.pi * x.ravel()))
    y = (y_true + rng.normal(0, 0.2, 800)).astype(np.float32)
    datasets.append(("piecewise", x, y))

    x = rng.uniform(-3, 3, 600).reshape(-1, 1).astype(np.float32)
    y = (np.exp(x.ravel()) + rng.normal(0.0, 0.5, 600)).astype(np.float32)
    datasets.append(("exponential", x, y))

    return datasets


def _bin_centroids(y_tr: np.ndarray, boundaries: np.ndarray, k: int) -> np.ndarray:
    """Compute median y per bin using the given bin boundaries."""
    y_flat = y_tr.flatten()
    _, y_binned = create_bins(data=y_flat, unique_bin_edges=boundaries)
    return np.array([np.median(y_flat[y_binned == i]) for i in range(k)])


def _train_classreg(
    ds_name: str, x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, y_te: np.ndarray, k: int
) -> ClassifierRegressionModel:
    model = ClassifierRegressionModel(
        base_classifier_class=PyTorchNeuralNetwork,
        n_classes=k,
        mapper_type=MapperType.LOOKUP_MEDIAN,
        uncertainty_method=UncertaintyMethod.CONSTANT,
        n_epochs=80,
        learning_rate=0.01,
        early_stopping_rounds=15,
        validation_fraction=0.2,
        random_seed=42,
        calculate_feature_importance=False,
    )
    model.fit(x_tr, y_tr)
    mse = float(np.mean((model.predict(x_te) - y_te) ** 2))
    logger.warning("%s k=%d [ClassReg]  mse=%.4f", ds_name, k, mse)
    return model


def _train_probreg(
    ds_name: str, x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, y_te: np.ndarray, k: int
) -> ProbabilisticRegressionModel:
    model = ProbabilisticRegressionModel(
        input_size=1,
        n_classes=k,
        regression_strategy=RegressionStrategy.SEPARATE_HEADS,
        uncertainty_method=UncertaintyMethod.PROBABILISTIC,
        constrain_middle_class=True,
        learning_rate=0.01,
        n_epochs=80,
        early_stopping_rounds=15,
        validation_fraction=0.2,
        random_seed=42,
        calculate_feature_importance=False,
    )
    model.fit(x_tr, y_tr)
    mse = float(np.mean((model.predict(x_te) - y_te) ** 2))
    logger.warning("%s k=%d [ProbReg]   mse=%.4f", ds_name, k, mse)
    return model


def _extract_classreg(
    model: ClassifierRegressionModel, x_tr: np.ndarray, y_tr: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (probs [N,k], per_head_means [N,k], centroids [k]) for a dense x grid.

    For LOOKUP_MEDIAN, the "head" for class i is the constant centroid_i, so
    per_head_means[:, i] = centroid_i for all rows.
    """
    x_eval = np.linspace(float(x_tr.min()), float(x_tr.max()), N_EVAL).reshape(-1, 1).astype(np.float32)
    probs = model.predict_proba(x_eval, filter_data=False)[:, :k]
    centroids = _bin_centroids(y_tr, model.class_boundaries, k)
    per_head_means = np.tile(centroids, (N_EVAL, 1))  # constant per class
    return probs, per_head_means, centroids


def _extract_probreg(
    model: ProbabilisticRegressionModel, x_tr: np.ndarray, y_tr: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (probs [N,k], per_head_means [N,k], centroids [k]) for a dense x grid.

    per_head_means[:, i] = h_i(p_i(x)) — each SEP_HEAD takes only p_i as input,
    so this is a genuine single-valued function (no multi-value fan artifact).
    """
    x_eval = np.linspace(float(x_tr.min()), float(x_tr.max()), N_EVAL).reshape(-1, 1).astype(np.float32)
    x_t = torch.tensor(x_eval, dtype=torch.float32).to(model.device)
    model.model.eval()
    with torch.no_grad():
        raw = model.model.classifier_layers(x_t)
        masked = torch.full_like(raw, float("-inf"))
        masked[:, :k] = raw[:, :k]
        probs_t = torch.softmax(masked, dim=1)
        probs = probs_t[:, :k].cpu().numpy()
        _, per_head = model.model.regression_module(probs_t, return_head_outputs=True)
        per_head_means = per_head[:, :k, 0].cpu().numpy()
    centroids = _bin_centroids(y_tr, model.precomputed_class_boundaries[k], k)
    return probs, per_head_means, centroids


def _plot_panel(
    ax: plt.Axes,
    probs: np.ndarray,
    per_head_means: np.ndarray,
    centroids: np.ndarray,
    k: int,
    label: str,
) -> None:
    """Fill one panel: h_i(p_i) vs p(class=i|x), sorted per class by p_i."""
    mid_idx = k // 2

    for i in range(k):
        order = np.argsort(probs[:, i])
        lw = 2.0 if i == mid_idx else 1.0
        ls = "--" if i == mid_idx else "-"
        pair = k - 1 - i
        alpha = 1.0 if i <= mid_idx else 0.6
        ax.plot(
            probs[order, i],
            per_head_means[order, i],
            lw=lw, ls=ls, alpha=alpha,
            label=f"c{i}" + (" ←mid" if i == mid_idx else (f" (pair c{pair})" if i < mid_idx else "")),
        )

    for i, c in enumerate(centroids):
        color = "gray" if i != mid_idx else "black"
        ax.axhline(c, color=color, lw=0.8, ls=":", alpha=0.7)

    ax.set_title(f"[{label}]  k={k}", fontsize=9)
    ax.set_xlabel("p(class=i | x)")
    ax.set_ylabel("head output (centroid for ClassReg, h_i(p_i) for ProbReg)")
    ax.legend(fontsize=6, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)


def _plot_page(
    pdf: PdfPages,
    ds_name: str,
    data: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    """One page per dataset: 2-row × 2-col grid."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"{ds_name} — classifier symmetry check", fontsize=12)

    labels = ["ClassReg", "ProbReg"]
    for col, k in enumerate(K_VALUES):
        for row, lbl in enumerate(labels):
            probs, y_hat, centroids = data[(k, lbl)]
            _plot_panel(axes[row, col], probs, y_hat, centroids, k, lbl)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    datasets = _make_datasets()
    rows: list[dict] = []

    pdf_path = OUT_DIR / "symmetry.pdf"
    with PdfPages(pdf_path) as pdf:
        for ds_name, x, y in datasets:
            x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=42)
            page_data: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

            for k in K_VALUES:
                cr = _train_classreg(ds_name, x_tr, y_tr, x_te, y_te, k)
                pr = _train_probreg(ds_name, x_tr, y_tr, x_te, y_te, k)

                page_data[(k, "ClassReg")] = _extract_classreg(cr, x_tr, y_tr, k)
                page_data[(k, "ProbReg")] = _extract_probreg(pr, x_tr, y_tr, k)

                rows.append({"dataset": ds_name, "k": k, "model": "ClassReg", "mse": round(float(np.mean((cr.predict(x_te) - y_te) ** 2)), 4)})
                rows.append({"dataset": ds_name, "k": k, "model": "ProbReg",  "mse": round(float(np.mean((pr.predict(x_te) - y_te) ** 2)), 4)})

            _plot_page(pdf, ds_name, page_data)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "symmetry.csv", index=False)

    print("\n=== Classifier symmetry check ===")
    print(df.pivot_table(index=["dataset", "k"], columns="model", values="mse").to_string())
    print(f"\nSaved to {OUT_DIR / 'symmetry.pdf'}")


if __name__ == "__main__":
    main()
