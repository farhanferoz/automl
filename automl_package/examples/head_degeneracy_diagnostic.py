"""Head degeneracy diagnostic — D1 (strategy) and D3 (monotonic constraints).

For each toy dataset × k ∈ {3, 5, 7} × 3 variants:
  - SEP       : SEPARATE_HEADS, no monotonic constraints (baseline)
  - SEP_MONO  : SEPARATE_HEADS + use_monotonic_constraints=True
  - SINGLE_N  : SINGLE_HEAD_N_OUTPUTS (head sees full probability vector)

Measures MSE and plots head mean vs p_i for each variant side-by-side.
For SEP/SEP_MONO each head is a 1-D function of p_i → smooth curves.
For SINGLE_N each head depends on all probs → scatter ordered by p_i.

PDF pages:
  1. MSE bar chart: k on x-axis, grouped bars per variant, one panel per dataset.
  2+. Per (dataset, k): 3-row × 2-col grid. Row = variant, col = (probs vs x | head mean vs p_i).

Output: head_degeneracy_results/
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

from automl_package.enums import RegressionStrategy, UncertaintyMethod
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

OUT_DIR = Path(__file__).parent / "head_degeneracy_results"
OUT_DIR.mkdir(exist_ok=True)

K_VALUES = (3, 5, 7)

VARIANTS: list[tuple[str, dict]] = [
    ("SEP",      {"regression_strategy": RegressionStrategy.SEPARATE_HEADS,      "use_monotonic_constraints": False}),
    ("SEP_MONO", {"regression_strategy": RegressionStrategy.SEPARATE_HEADS,      "use_monotonic_constraints": True}),
    ("SINGLE_N", {"regression_strategy": RegressionStrategy.SINGLE_HEAD_N_OUTPUTS, "use_monotonic_constraints": False}),
]

VARIANT_COLORS = {"SEP": "C0", "SEP_MONO": "C1", "SINGLE_N": "C2"}


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


def _train(ds_name: str, x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, y_te: np.ndarray,
           k: int, variant_name: str, variant_kwargs: dict) -> tuple[ProbabilisticRegressionModel, float]:
    model = ProbabilisticRegressionModel(
        input_size=1,
        n_classes=k,
        uncertainty_method=UncertaintyMethod.PROBABILISTIC,
        constrain_middle_class=True,
        learning_rate=0.01,
        n_epochs=80,
        early_stopping_rounds=15,
        validation_fraction=0.2,
        random_seed=42,
        calculate_feature_importance=False,
        **variant_kwargs,
    )
    model.fit(x_tr, y_tr)
    y_pred = model.predict(x_te)
    mse = float(np.mean((y_pred - y_te) ** 2))
    logger.warning("%s k=%d [%s]  mse=%.4f", ds_name, k, variant_name, mse)
    return model, mse


def _extract_head_data(model: ProbabilisticRegressionModel, x_tr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_eval, probs [N, k], per_head_means [N, k]) over a dense x grid."""
    x_min, x_max = float(x_tr.min()), float(x_tr.max())
    x_eval = np.linspace(x_min, x_max, 400).reshape(-1, 1).astype(np.float32)
    x_t = torch.tensor(x_eval, dtype=torch.float32).to(model.device)
    model.model.eval()
    with torch.no_grad():
        raw_logits = model.model.classifier_layers(x_t)
        masked = torch.full_like(raw_logits, float("-inf"))
        masked[:, : model.n_classes] = raw_logits[:, : model.n_classes]
        probs = torch.softmax(masked, dim=1)[:, : model.n_classes].cpu().numpy()
        _, per_head = model.model.regression_module(torch.softmax(masked, dim=1), return_head_outputs=True)
        per_head_means = per_head[:, : model.n_classes, 0].cpu().numpy()
    return x_eval.ravel(), probs, per_head_means


def _plot_row(
    ax_left: plt.Axes,
    ax_right: plt.Axes,
    variant_name: str,
    k: int,
    x_eval: np.ndarray,
    probs: np.ndarray,
    per_head_means: np.ndarray,
) -> None:
    """Fill one row (variant) of the per-(dataset,k) page."""
    mid_idx = k // 2

    for i in range(k):
        lw = 2.0 if i == mid_idx else 1.0
        ls = "--" if i == mid_idx else "-"
        label = f"c{i}" + (" ←mid" if i == mid_idx else "")
        ax_left.plot(x_eval, probs[:, i], lw=lw, ls=ls, label=label)

    ax_left.set_ylabel(f"[{variant_name}]\np(class|x)")
    ax_left.legend(fontsize=6, loc="upper right", ncol=2)
    ax_left.grid(True, alpha=0.3)

    for i in range(k):
        order = np.argsort(probs[:, i])
        lw = 2.0 if i == mid_idx else 1.0
        ls = "--" if i == mid_idx else "-"
        ax_right.plot(probs[order, i], per_head_means[order, i], lw=lw, ls=ls,
                      label=f"h{i}" + (" ←mid" if i == mid_idx else ""), alpha=0.85)

    ax_right.set_ylabel("head mean")
    ax_right.legend(fontsize=6, loc="best", ncol=2)
    ax_right.grid(True, alpha=0.3)


def _plot_detail_page(
    pdf: PdfPages,
    ds_name: str,
    k: int,
    variant_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    """One page: 3-row (variants) × 2-col (probs-vs-x | head-vs-pi)."""
    n_variants = len(VARIANTS)
    fig, axes = plt.subplots(n_variants, 2, figsize=(11, 3.8 * n_variants))
    fig.suptitle(f"{ds_name}  k={k}  |  mid_idx={k // 2}", fontsize=11)

    for row, (vname, _) in enumerate(VARIANTS):
        x_eval, probs, per_head_means = variant_data[vname]
        _plot_row(axes[row, 0], axes[row, 1], vname, k, x_eval, probs, per_head_means)

    axes[-1, 0].set_xlabel("x")
    axes[-1, 1].set_xlabel("p(class=i | x)")

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    datasets = _make_datasets()
    rows: list[dict] = []
    trained: dict[tuple[str, int, str], tuple[ProbabilisticRegressionModel, np.ndarray]] = {}

    for ds_name, x, y in datasets:
        x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=42)

        for k in K_VALUES:
            for vname, vkwargs in VARIANTS:
                model, mse = _train(ds_name, x_tr, y_tr, x_te, y_te, k, vname, vkwargs)
                rows.append({"dataset": ds_name, "k": k, "variant": vname, "mse": round(mse, 4)})
                trained[(ds_name, k, vname)] = (model, x_tr)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "degeneracy.csv", index=False)

    pdf_path = OUT_DIR / "degeneracy.pdf"
    with PdfPages(pdf_path) as pdf:
        # Page 1: MSE bar chart
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        fig.suptitle("MSE by variant — D1 (strategy) and D3 (monotonic constraints)")
        for ax, (ds_name, _, _) in zip(axes.ravel(), datasets):
            sub = df[df["dataset"] == ds_name]
            x_pos = np.arange(len(K_VALUES))
            width = 0.25
            for vi, (vname, _) in enumerate(VARIANTS):
                vals = [sub[(sub["k"] == k) & (sub["variant"] == vname)]["mse"].values[0] for k in K_VALUES]
                ax.bar(x_pos + vi * width, vals, width, label=vname, color=VARIANT_COLORS[vname], alpha=0.8)
            ax.set_title(ds_name)
            ax.set_xlabel("k")
            ax.set_ylabel("MSE")
            ax.set_xticks(x_pos + width)
            ax.set_xticklabels(K_VALUES)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Per (dataset, k) detail pages
        for ds_name, x, _ in datasets:
            for k in K_VALUES:
                variant_data = {}
                for vname, _ in VARIANTS:
                    model, x_tr = trained[(ds_name, k, vname)]
                    x_eval, probs, per_head_means = _extract_head_data(model, x_tr)
                    variant_data[vname] = (x_eval, probs, per_head_means)
                _plot_detail_page(pdf, ds_name, k, variant_data)

    print("\n=== Head degeneracy diagnostic ===")
    pivot = df.pivot_table(index=["dataset", "k"], columns="variant", values="mse")
    print(pivot.to_string())
    print(f"\nSaved to {OUT_DIR / 'degeneracy.csv'} and {OUT_DIR / 'degeneracy.pdf'}")


if __name__ == "__main__":
    main()
