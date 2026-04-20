"""ProbReg identifiability sweep — 8-cell experiment matrix.

Tests three orthogonal identifiability fixes (MDN NLL, CE_STOP_GRAD, anchored heads)
across 4 toy datasets × k ∈ {3, 5} × 3 seeds, plus ClassReg baselines.

Cells:
  A: GAUSSIAN_LTV + REGRESSION_ONLY + free heads  (current baseline)
  B: GAUSSIAN_LTV + REGRESSION_ONLY + anchored heads
  C: GAUSSIAN_LTV + CE_STOP_GRAD    + free heads
  D: GAUSSIAN_LTV + CE_STOP_GRAD    + anchored heads
  E: MDN          + REGRESSION_ONLY + free heads
  F: MDN          + REGRESSION_ONLY + anchored heads
  G: MDN          + CE_STOP_GRAD    + free heads
  H: MDN          + CE_STOP_GRAD    + anchored heads
  ClassReg:        LOOKUP_MEDIAN classifier baseline

Outputs (in probreg_identifiability_results/):
  results.csv       — per-run metrics
  summary.csv       — mean ± std across seeds per (dataset, k, cell)
  results_{dataset}.pdf — 5-page PDF per dataset (metrics table +
                          h_i(p_i) plots k=3/5 + p_i(x) plots k=3/5)
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split

from automl_package.enums import (
    MapperType,
    ProbabilisticRegressionOptimizationStrategy,
    ProbRegLossType,
    RegressionStrategy,
    UncertaintyMethod,
)
from automl_package.models.classifier_regression import ClassifierRegressionModel
from automl_package.models.neural_network import PyTorchNeuralNetwork
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel
from automl_package.utils.numerics import create_bins

mpl.use("Agg")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

OUT_DIR = Path(__file__).parent / "probreg_identifiability_results"
OUT_DIR.mkdir(exist_ok=True)

SEEDS = (42, 123, 7)
K_VALUES = (3, 5)
N_EVAL = 400
N_EPOCHS = 80
LR = 0.01
EARLY_STOP = 15
VAL_FRAC = 0.2

CELLS = [
    ("A", ProbRegLossType.GAUSSIAN_LTV, ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY, False),
    ("B", ProbRegLossType.GAUSSIAN_LTV, ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY, True),
    ("C", ProbRegLossType.GAUSSIAN_LTV, ProbabilisticRegressionOptimizationStrategy.CE_STOP_GRAD, False),
    ("D", ProbRegLossType.GAUSSIAN_LTV, ProbabilisticRegressionOptimizationStrategy.CE_STOP_GRAD, True),
    ("E", ProbRegLossType.MDN, ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY, False),
    ("F", ProbRegLossType.MDN, ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY, True),
    ("G", ProbRegLossType.MDN, ProbabilisticRegressionOptimizationStrategy.CE_STOP_GRAD, False),
    ("H", ProbRegLossType.MDN, ProbabilisticRegressionOptimizationStrategy.CE_STOP_GRAD, True),
]


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _gaussian_nll(y: np.ndarray, mean: np.ndarray, log_var: np.ndarray) -> float:
    var = np.exp(log_var)
    return float(np.mean(0.5 * (math.log(2 * math.pi) + log_var + (y - mean) ** 2 / var)))


def _mdn_nll(y: np.ndarray, probs: np.ndarray, mus: np.ndarray, log_vars: np.ndarray) -> float:
    y = y.reshape(-1, 1)
    log_component = -0.5 * (math.log(2 * math.pi) + log_vars + (y - mus) ** 2 / np.exp(log_vars))
    log_weights = np.log(np.clip(probs, 1e-8, None))
    log_mixture = np.logaddexp.reduce(log_weights + log_component, axis=1)
    return float(-np.mean(log_mixture))


def _bin_centroids(y_tr: np.ndarray, boundaries: np.ndarray, k: int) -> np.ndarray:
    """Per-bin centroids using mean.

    Must match ProbReg model's `_per_class_centroids` so that anchor_error is meaningful for
    anchored heads (where AnchoredHead.centroid is the mean).
    """
    y_flat = y_tr.flatten()
    _, y_binned = create_bins(data=y_flat, unique_bin_edges=boundaries)
    counts = np.bincount(y_binned, minlength=k)
    sums = np.bincount(y_binned, weights=y_flat, minlength=k)
    return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _train_classreg(x_tr: np.ndarray, y_tr: np.ndarray, k: int, seed: int) -> ClassifierRegressionModel:
    model = ClassifierRegressionModel(
        base_classifier_class=PyTorchNeuralNetwork,
        n_classes=k,
        mapper_type=MapperType.LOOKUP_MEDIAN,
        uncertainty_method=UncertaintyMethod.CONSTANT,
        n_epochs=N_EPOCHS,
        learning_rate=LR,
        early_stopping_rounds=EARLY_STOP,
        validation_fraction=VAL_FRAC,
        random_seed=seed,
        calculate_feature_importance=False,
    )
    model.fit(x_tr, y_tr)
    return model


def _train_probreg(x_tr: np.ndarray, y_tr: np.ndarray, k: int, seed: int,
                   loss_type: ProbRegLossType, opt_strategy: ProbabilisticRegressionOptimizationStrategy, use_anchored: bool) -> ProbabilisticRegressionModel:
    constrain_mid = not use_anchored
    model = ProbabilisticRegressionModel(
        input_size=1,
        n_classes=k,
        regression_strategy=RegressionStrategy.SEPARATE_HEADS,
        uncertainty_method=UncertaintyMethod.PROBABILISTIC,
        optimization_strategy=opt_strategy,
        prob_reg_loss_type=loss_type,
        use_anchored_heads=use_anchored,
        constrain_middle_class=constrain_mid,
        use_monotonic_constraints=False,
        n_epochs=N_EPOCHS,
        learning_rate=LR,
        early_stopping_rounds=EARLY_STOP,
        validation_fraction=VAL_FRAC,
        random_seed=seed,
        calculate_feature_importance=False,
    )
    model.fit(x_tr, y_tr)
    return model


# ---------------------------------------------------------------------------
# Extract internal representations for plotting
# ---------------------------------------------------------------------------

def _extract_classreg(model: ClassifierRegressionModel, x_tr: np.ndarray, y_tr: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_eval = np.linspace(float(x_tr.min()), float(x_tr.max()), N_EVAL).reshape(-1, 1).astype(np.float32)
    probs = model.predict_proba(x_eval, filter_data=False)[:, :k]
    centroids = _bin_centroids(y_tr, model.class_boundaries, k)
    per_head_means = np.tile(centroids, (N_EVAL, 1))
    return probs, per_head_means, centroids


def _probreg_masked_probs(model: ProbabilisticRegressionModel, x_eval: np.ndarray, k: int) -> np.ndarray:
    x_t = torch.tensor(x_eval, dtype=torch.float32).to(model.device)
    model.model.eval()
    with torch.no_grad():
        raw = model.model.classifier_layers(x_t)
        masked = torch.full_like(raw, float("-inf"))
        masked[:, :k] = raw[:, :k]
        return torch.softmax(masked, dim=1)[:, :k].cpu().numpy()


def _extract_probreg(model: ProbabilisticRegressionModel, x_tr: np.ndarray, y_tr: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


# ---------------------------------------------------------------------------
# Metrics for one run
# ---------------------------------------------------------------------------

def _compute_metrics(model: ClassifierRegressionModel | ProbabilisticRegressionModel,
                     x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, y_te: np.ndarray,
                     cell_label: str, k: int, is_classreg: bool) -> dict:
    preds = model.predict(x_te)
    mse = float(np.mean((preds - y_te) ** 2))

    # Gaussian NLL from predict_uncertainty (available for both models)
    try:
        std = model.predict_uncertainty(x_te)
        log_var = np.log(np.clip(std ** 2, 1e-8, None))
        nll_gaussian = _gaussian_nll(y_te, preds, log_var)
    except Exception:
        nll_gaussian = float("nan")

    # MDN NLL and anchor_error only for ProbReg
    nll_mdn = float("nan")
    max_p_mid = float("nan")
    anchor_error = float("nan")

    if not is_classreg and hasattr(model, "predict_distribution"):
        try:
            dist = model.predict_distribution(x_te)
            mus = dist.means
            stds = dist.stds
            log_vars_mdn = np.log(np.clip(stds ** 2, 1e-8, None))
            weights = dist.weights
            nll_mdn = _mdn_nll(y_te, weights, mus, log_vars_mdn)
        except Exception:
            pass

        try:
            probs_ext, per_head_ext, centroids_ext = _extract_probreg(model, x_tr, y_tr, k)
            if k == 5:
                max_p_mid = float(probs_ext[:, k // 2].max())
            anchor_errs = [abs(per_head_ext[np.argmax(probs_ext[:, i]), i] - centroids_ext[i]) for i in range(k)]
            anchor_error = float(np.max(anchor_errs))
        except Exception:
            pass

    is_mdn_cell = not is_classreg and cell_label in ("E", "F", "G", "H")
    nll_own = nll_mdn if is_mdn_cell else nll_gaussian

    return {
        "cell": cell_label,
        "mse": round(mse, 6),
        "nll_gaussian": round(nll_gaussian, 6),
        "nll_mdn": round(nll_mdn, 6),
        "nll_own": round(nll_own, 6),
        "max_p_mid": round(max_p_mid, 4),
        "anchor_error": round(anchor_error, 4),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_head_curves_panel(ax: plt.Axes, probs: np.ndarray, per_head_means: np.ndarray, centroids: np.ndarray, k: int, title: str) -> None:
    mid_idx = k // 2
    for i in range(k):
        order = np.argsort(probs[:, i])
        lw = 2.0 if i == mid_idx else 1.0
        ls = "--" if i == mid_idx else "-"
        ax.plot(probs[order, i], per_head_means[order, i], lw=lw, ls=ls, label=f"c{i}")
    for _i, c in enumerate(centroids):
        if np.isfinite(c):
            ax.axhline(c, color="gray", lw=0.6, ls=":", alpha=0.7)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("p(class=i|x)", fontsize=7)
    ax.set_ylabel("head mean", fontsize=7)
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=6)


def _plot_prob_curves_panel(ax: plt.Axes, model: ClassifierRegressionModel | ProbabilisticRegressionModel,
                            x_tr: np.ndarray, y_tr: np.ndarray, k: int, title: str, boundaries: np.ndarray | None) -> None:
    x_min, x_max = float(x_tr.min()), float(x_tr.max())
    x_eval = np.linspace(x_min, x_max, N_EVAL).reshape(-1, 1).astype(np.float32)

    probs = model.predict_proba(x_eval, filter_data=False)[:, :k] if isinstance(model, ClassifierRegressionModel) else _probreg_masked_probs(model, x_eval, k)

    mid_idx = k // 2
    for i in range(k):
        lw = 2.0 if i == mid_idx else 1.0
        ls = "--" if i == mid_idx else "-"
        ax.plot(x_eval.ravel(), probs[:, i], lw=lw, ls=ls, label=f"p{i}")

    ax.set_ylabel("p(class=i|x)", fontsize=7)
    ax.set_xlabel("x", fontsize=7)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)

    ax2 = ax.twinx()
    order = np.argsort(x_tr.ravel())
    ax2.scatter(x_tr.ravel()[order], y_tr.ravel()[order], s=2, color="black", alpha=0.2)
    if boundaries is not None:
        for b in boundaries:
            if np.isfinite(b):
                ax2.axhline(b, color="red", lw=0.6, ls=":", alpha=0.5)
    ax2.tick_params(axis="y", labelsize=6)


def _draw_head_cr(ax: plt.Axes, model: ClassifierRegressionModel, x_tr: np.ndarray, y_tr: np.ndarray, k: int, title: str) -> None:
    probs, per_head, centroids = _extract_classreg(model, x_tr, y_tr, k)
    _plot_head_curves_panel(ax, probs, per_head, centroids, k, title)


def _draw_head_pr(ax: plt.Axes, model: ProbabilisticRegressionModel, x_tr: np.ndarray, y_tr: np.ndarray, k: int, title: str) -> None:
    probs, per_head, centroids = _extract_probreg(model, x_tr, y_tr, k)
    _plot_head_curves_panel(ax, probs, per_head, centroids, k, title)


def _draw_prob_cr(ax: plt.Axes, model: ClassifierRegressionModel, x_tr: np.ndarray, y_tr: np.ndarray, k: int, title: str) -> None:
    _plot_prob_curves_panel(ax, model, x_tr, y_tr, k, title, model.class_boundaries)


def _draw_prob_pr(ax: plt.Axes, model: ProbabilisticRegressionModel, x_tr: np.ndarray, y_tr: np.ndarray, k: int, title: str) -> None:
    _plot_prob_curves_panel(ax, model, x_tr, y_tr, k, title, model.precomputed_class_boundaries.get(k))


def _render_grid_page(pdf: PdfPages, ds_name: str, seed42_models: dict, k: int, page_title: str, draw_cr: Callable, draw_pr: Callable) -> None:
    nrows = math.ceil((len(CELLS) + 1) / 3)
    fig, axes = plt.subplots(nrows, 3, figsize=(15, 4 * nrows))
    fig.suptitle(f"{ds_name} k={k} — {page_title} (seed 42)", fontsize=10)
    axes_flat = axes.ravel() if hasattr(axes, "ravel") else [axes]

    panel_idx = 0
    key_cr = f"ClassReg_{k}"
    if key_cr in seed42_models:
        cr_model, x_tr, y_tr = seed42_models[key_cr]
        draw_cr(axes_flat[panel_idx], cr_model, x_tr, y_tr, k, f"ClassReg k={k}")
        panel_idx += 1

    for cell_id, *_ in CELLS:
        key = f"{cell_id}_{k}"
        if key in seed42_models:
            pr_model, x_tr, y_tr = seed42_models[key]
            try:
                draw_pr(axes_flat[panel_idx], pr_model, x_tr, y_tr, k, f"Cell {cell_id} k={k}")
            except Exception as e:
                axes_flat[panel_idx].set_title(f"Cell {cell_id} — error: {e}", fontsize=7)
            panel_idx += 1

    for ax in axes_flat[panel_idx:]:
        ax.axis("off")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _make_pdf(ds_name: str, seed42_models: dict, summary_df: pd.DataFrame) -> None:
    """Generate a 5-page PDF for one dataset."""
    pdf_path = OUT_DIR / f"results_{ds_name}.pdf"

    with PdfPages(pdf_path) as pdf:
        # Page 1: metrics summary table
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.axis("off")
        fig.suptitle(f"{ds_name} — metrics summary (mean ± std across {len(SEEDS)} seeds)", fontsize=11)

        ds_df = summary_df[summary_df["dataset"] == ds_name].copy()
        if not ds_df.empty:
            ds_df = ds_df.sort_values(["k", "cell"])
            cols = ["k", "cell", "mse_mean", "mse_std", "nll_gaussian_mean", "nll_mdn_mean", "max_p_mid_mean", "anchor_error_mean"]
            present_cols = [c for c in cols if c in ds_df.columns]
            table = ax.table(cellText=ds_df[present_cols].values.tolist(), colLabels=present_cols, cellLoc="center", loc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(7)
            table.scale(1, 1.4)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Pages 2–3: h_i(p_i) curves; pages 4–5: p_i(x) vs x
        for k in K_VALUES:
            _render_grid_page(pdf, ds_name, seed42_models, k, "h_i(p_i) vs p_i", _draw_head_cr, _draw_head_pr)
        for k in K_VALUES:
            _render_grid_page(pdf, ds_name, seed42_models, k, "p_i(x) vs x", _draw_prob_cr, _draw_prob_pr)

    logger.warning("Saved %s", pdf_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the 8-cell ProbReg identifiability sweep and generate result PDFs."""
    datasets = _make_datasets()
    all_rows: list[dict] = []

    # seed42_models[ds_name][key] = (model, x_tr, y_tr) for PDF plotting
    seed42_models: dict[str, dict] = {ds_name: {} for ds_name, _, _ in datasets}

    total = len(datasets) * len(K_VALUES) * len(SEEDS) * (len(CELLS) + 1)
    run_count = 0

    for ds_name, x, y in datasets:
        print(f"\n=== Dataset: {ds_name} ===")
        for k in K_VALUES:
            for seed in SEEDS:
                x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=seed)

                # ClassReg baseline
                run_count += 1
                print(f"  [{run_count}/{total}] ClassReg k={k} seed={seed} ...", end="", flush=True)
                try:
                    cr = _train_classreg(x_tr, y_tr, k, seed)
                    row = _compute_metrics(cr, x_tr, y_tr, x_te, y_te, "ClassReg", k, is_classreg=True)
                    row.update({"dataset": ds_name, "k": k, "seed": seed})
                    all_rows.append(row)
                    mse = row["mse"]
                    print(f" MSE={mse:.4f}")
                    if seed == 42:
                        seed42_models[ds_name][f"ClassReg_{k}"] = (cr, x_tr, y_tr)
                except Exception as e:
                    print(f" ERROR: {e}")
                    all_rows.append({"dataset": ds_name, "k": k, "seed": seed, "cell": "ClassReg", "mse": float("nan")})

                # ProbReg cells
                for cell_id, loss_type, opt_strat, use_anchored in CELLS:
                    run_count += 1
                    print(f"  [{run_count}/{total}] Cell {cell_id} k={k} seed={seed} ...", end="", flush=True)
                    try:
                        pr = _train_probreg(x_tr, y_tr, k, seed, loss_type, opt_strat, use_anchored)
                        row = _compute_metrics(pr, x_tr, y_tr, x_te, y_te, cell_id, k, is_classreg=False)
                        row.update({"dataset": ds_name, "k": k, "seed": seed})
                        all_rows.append(row)
                        print(f" MSE={row['mse']:.4f}")
                        if seed == 42:
                            seed42_models[ds_name][f"{cell_id}_{k}"] = (pr, x_tr, y_tr)
                    except Exception as e:
                        print(f" ERROR: {e}")
                        all_rows.append({"dataset": ds_name, "k": k, "seed": seed, "cell": cell_id, "mse": float("nan")})

    # Save results
    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(OUT_DIR / "results.csv", index=False)

    # Summary: mean ± std across seeds
    numeric_cols = ["mse", "nll_gaussian", "nll_mdn", "nll_own", "max_p_mid", "anchor_error"]
    present = [c for c in numeric_cols if c in results_df.columns]
    agg = {c: ["mean", "std"] for c in present}
    summary = results_df.groupby(["dataset", "k", "cell"]).agg(agg)
    summary.columns = ["_".join(col) for col in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(OUT_DIR / "summary.csv", index=False)

    # Print summary table
    print("\n\n=== SUMMARY (mean across seeds) ===")
    for ds_name, _, _ in datasets:
        ds_sum = summary[summary["dataset"] == ds_name]
        print(f"\n{ds_name}:")
        print(ds_sum[["k", "cell", "mse_mean", "nll_gaussian_mean"]].to_string(index=False))

    # Generate PDFs
    print("\nGenerating PDFs...")
    for ds_name, _, _ in datasets:
        _make_pdf(ds_name, seed42_models[ds_name], summary)

    print(f"\nDone. Results in {OUT_DIR}/")


if __name__ == "__main__":
    main()
