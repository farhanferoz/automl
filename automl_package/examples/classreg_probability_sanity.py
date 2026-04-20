"""ClassReg classifier sanity check.

For each toy dataset × k ∈ {3, 5}:
  - Train ClassReg (NN classifier + LOOKUP_MEDIAN).
  - Plot p_i(x) vs x for every class, with true y(x) overlaid on a twin axis.
  - Mark bin boundaries as horizontal lines on the y(x) axis so we can see
    which x-ranges SHOULD activate which class.

Goal: verify that high p_i occurs where the true y falls into bin i —
i.e., the classifier itself is semantically correct.

PDF layout:
  4 pages (one per dataset), each a 1×2 grid:
    cols: k=3 (left), k=5 (right)

Output: classreg_probability_sanity_results/
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split

from automl_package.enums import MapperType, UncertaintyMethod
from automl_package.models.classifier_regression import ClassifierRegressionModel
from automl_package.models.neural_network import PyTorchNeuralNetwork

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

OUT_DIR = Path(__file__).parent / "classreg_probability_sanity_results"
OUT_DIR.mkdir(exist_ok=True)

K_VALUES = (3, 5)
N_EVAL = 500


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


def _train(x_tr: np.ndarray, y_tr: np.ndarray, k: int) -> ClassifierRegressionModel:
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
    return model


def _plot_panel(
    ax: plt.Axes,
    model: ClassifierRegressionModel,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    k: int,
) -> None:
    """Plot p_i(x) vs x with true y overlaid on a twin axis."""
    x_min, x_max = float(x_tr.min()), float(x_tr.max())
    x_eval = np.linspace(x_min, x_max, N_EVAL).reshape(-1, 1).astype(np.float32)
    probs = model.predict_proba(x_eval, filter_data=False)[:, :k]

    mid_idx = k // 2
    for i in range(k):
        lw = 2.0 if i == mid_idx else 1.2
        ls = "--" if i == mid_idx else "-"
        ax.plot(x_eval.ravel(), probs[:, i], lw=lw, ls=ls,
                label=f"p{i}" + (" ←mid" if i == mid_idx else ""))
    ax.set_ylabel("p(class=i | x)", fontsize=9)
    ax.set_xlabel("x", fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    order = np.argsort(x_tr.ravel())
    ax2.scatter(x_tr.ravel()[order], y_tr.ravel()[order], s=3, color="black", alpha=0.25, label="y_train")
    for b in model.class_boundaries:
        if np.isfinite(b):
            ax2.axhline(b, color="red", lw=0.8, ls=":", alpha=0.6)
    ax2.set_ylabel("y (scatter) + bin boundaries (red dotted)", fontsize=8, color="dimgray")
    ax2.tick_params(axis="y", labelcolor="dimgray")


def main() -> None:
    datasets = _make_datasets()
    pdf_path = OUT_DIR / "sanity.pdf"

    with PdfPages(pdf_path) as pdf:
        for ds_name, x, y in datasets:
            x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=42)

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(f"{ds_name} — ClassReg classifier probabilities p_i(x)", fontsize=11)

            for col, k in enumerate(K_VALUES):
                model = _train(x_tr, y_tr, k)
                mse = float(np.mean((model.predict(x_te) - y_te) ** 2))
                logger.warning("%s k=%d  mse=%.4f", ds_name, k, mse)
                axes[col].set_title(f"k={k}   MSE={mse:.3f}", fontsize=10)
                _plot_panel(axes[col], model, x_tr, y_tr, k)

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"\nSaved to {pdf_path}")


if __name__ == "__main__":
    main()
