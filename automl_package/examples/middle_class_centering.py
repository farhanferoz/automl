"""Middle-class centering diagnostic.

For each toy dataset × k ∈ {3, 5, 7}:
  - Trains ProbReg (SEP_HEADS, fixed k, PROBABILISTIC).
  - Reports the learned middle-head mean vs the data centroid (warm-start value).
  - Gap = |learned_mean - data_centroid|; small but nonzero after training is healthy.
  - Also reports MSE so we can see if centering correlates with fit quality.

PDF pages:
  1. Centroid vs learned mean across k, per dataset.
  2. |gap| bar chart, per dataset.
  3–14. Per (dataset, k): classifier p(class|x) vs x  +  head mean vs p_i.

Single seed, no HPO, 80 epochs. Output saved to middle_class_centering_results/.
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
from automl_package.models.common.regression_heads import ConstantHead, ProbabilisticMiddleClassHead
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel
from automl_package.utils.numerics import create_bins

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

OUT_DIR = Path(__file__).parent / "middle_class_centering_results"
OUT_DIR.mkdir(exist_ok=True)


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


def _get_middle_head_mean(model: ProbabilisticRegressionModel) -> float | None:
    rm = model.model.regression_module
    for head in getattr(rm, "heads", []):
        if isinstance(head, (ConstantHead, ProbabilisticMiddleClassHead)):
            return float(head.mean.item())
    return None


def _get_data_centroid(y_train: np.ndarray, k: int) -> float | None:
    if k % 2 == 0:
        return None
    y_flat = y_train.flatten()
    _, y_binned = create_bins(data=y_flat, n_bins=k, min_value=-np.inf, max_value=np.inf)
    mid_mask = y_binned == k // 2
    return float(y_flat[mid_mask].mean()) if mid_mask.any() else None


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


def _plot_head_page(
    pdf: PdfPages,
    ds_name: str,
    k: int,
    x_eval: np.ndarray,
    probs: np.ndarray,
    per_head_means: np.ndarray,
    data_centroid: float | None,
    learned_mean: float | None,
) -> None:
    """One PDF page: probs vs x (left) + head mean vs p_i (right)."""
    mid_idx = k // 2  # index of middle head (only meaningful when k is odd)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(f"{ds_name}  k={k}  |  middle head idx={mid_idx}", fontsize=11)

    # Left: p(class | x) vs x
    for i in range(k):
        lw = 2.0 if i == mid_idx else 1.0
        ls = "--" if i == mid_idx else "-"
        ax_left.plot(x_eval, probs[:, i], lw=lw, ls=ls, label=f"class {i}" + (" ←mid" if i == mid_idx else ""))
    ax_left.set_xlabel("x")
    ax_left.set_ylabel("p(class | x)")
    ax_left.set_title("Classifier probabilities vs x")
    ax_left.legend(fontsize=7, loc="upper right")
    ax_left.grid(True, alpha=0.3)

    # Right: each head's mean output vs its own class probability (sorted by p_i)
    for i in range(k):
        order = np.argsort(probs[:, i])
        lw = 2.0 if i == mid_idx else 1.0
        ls = "--" if i == mid_idx else "-"
        label = f"head {i}" + (" ←mid" if i == mid_idx else "")
        ax_right.plot(probs[order, i], per_head_means[order, i], lw=lw, ls=ls, label=label, alpha=0.85)

    # Mark data centroid and learned mean as horizontal reference lines for the middle head
    if data_centroid is not None:
        ax_right.axhline(data_centroid, color="k", lw=1, ls=":", label=f"centroid {data_centroid:.3f}")
    if learned_mean is not None:
        ax_right.axhline(learned_mean, color="gray", lw=1, ls="-.", label=f"learned {learned_mean:.3f}")

    ax_right.set_xlabel("p(class=i | x)")
    ax_right.set_ylabel("head mean output")
    ax_right.set_title("Regression head mean vs class probability")
    ax_right.legend(fontsize=7, loc="best")
    ax_right.grid(True, alpha=0.3)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    datasets = _make_datasets()
    rows: list[dict] = []
    # Store trained models so we can plot after the training loop
    trained: dict[tuple[str, int], tuple[ProbabilisticRegressionModel, np.ndarray, float | None, float | None]] = {}

    for ds_name, x, y in datasets:
        x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=42)

        for k in (3, 5, 7):
            data_centroid = _get_data_centroid(y_tr, k)

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

            learned_mean = _get_middle_head_mean(model)
            y_pred = model.predict(x_te)
            mse = float(np.mean((y_pred - y_te) ** 2))
            gap = abs(learned_mean - data_centroid) if (learned_mean is not None and data_centroid is not None) else None

            rows.append({
                "dataset": ds_name,
                "k": k,
                "data_centroid": round(data_centroid, 4) if data_centroid is not None else None,
                "learned_mean": round(learned_mean, 4) if learned_mean is not None else None,
                "gap": round(gap, 4) if gap is not None else None,
                "mse": round(mse, 4),
            })
            trained[(ds_name, k)] = (model, x_tr, data_centroid, learned_mean)

            logger.warning(
                "%s k=%d  centroid=%.4f  learned=%.4f  gap=%.4f  mse=%.4f",
                ds_name, k, data_centroid or 0, learned_mean or 0, gap or 0, mse,
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "centering.csv", index=False)

    _plot(df, OUT_DIR / "centering.pdf", trained)

    print("\n=== Middle-class centering diagnostic ===")
    print(df.to_string(index=False))
    print(f"\nSaved to {OUT_DIR / 'centering.csv'} and {OUT_DIR / 'centering.pdf'}")


def _plot(
    df: pd.DataFrame,
    pdf_path: Path,
    trained: dict[tuple[str, int], tuple[ProbabilisticRegressionModel, np.ndarray, float | None, float | None]],
) -> None:
    datasets = df["dataset"].unique()

    with PdfPages(pdf_path) as pdf:
        # Page 1: centroid vs learned mean across k
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        fig.suptitle("Middle-head: data centroid (warm start) vs learned mean after training")
        for ax, ds in zip(axes.ravel(), datasets):
            sub = df[df["dataset"] == ds].sort_values("k")
            ax.plot(sub["k"], sub["data_centroid"], "o--", color="C0", label="centroid (init)")
            ax.plot(sub["k"], sub["learned_mean"], "s-", color="C1", label="learned")
            ax.set_title(ds); ax.set_xlabel("k"); ax.set_ylabel("middle-head mean")
            ax.set_xticks(sub["k"]); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # Page 2: gap bar chart
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        fig.suptitle("Gap = |learned_mean − data_centroid| after training")
        for ax, ds in zip(axes.ravel(), datasets):
            sub = df[df["dataset"] == ds].sort_values("k")
            ax.bar(sub["k"], sub["gap"], color="C2", alpha=0.7)
            ax.set_title(ds); ax.set_xlabel("k"); ax.set_ylabel("|gap|")
            ax.set_xticks(sub["k"]); ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # Pages 3–14: per (dataset, k) head output plots
        for ds in datasets:
            for k in (3, 5, 7):
                model, x_tr, data_centroid, learned_mean = trained[(ds, k)]
                x_eval, probs, per_head_means = _extract_head_data(model, x_tr)
                _plot_head_page(pdf, ds, k, x_eval, probs, per_head_means, data_centroid, learned_mean)


if __name__ == "__main__":
    main()
