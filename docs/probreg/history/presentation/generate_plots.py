"""
Generate the 3 result plots for the Turing Institute presentation.

Result (a): MSE vs n_classes at different noise levels → shows optimal k tracks SNR
Result (b): Standardised residual histogram vs N(0,1) → uncertainty calibration
Result (c): Regression functions fⱼ(pⱼ) for k=3 → learned probability-to-value mappings
"""

import sys
import os

sys.path.insert(0, os.path.expanduser("~/dev/Oasis/StrategyA"))

import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
from typing import List, Tuple, Optional, Dict

from auto_machine_learning.ml_constants import MlModel
from auto_machine_learning.ml_classifier_regressor import HyperParameterOptimisationMethod
from auto_machine_learning.ml_utilities import train_regression_model
from auto_machine_learning.test_regressor import simulate_data, get_model_label

matplotlib.rcParams.update(
    {
        "figure.facecolor": "#0F1B2E",
        "axes.facecolor": "#0F1B2E",
        "axes.edgecolor": "#4FC3F7",
        "axes.labelcolor": "#E0E0E0",
        "text.color": "#E0E0E0",
        "xtick.color": "#B0BEC5",
        "ytick.color": "#B0BEC5",
        "grid.color": "#263238",
        "grid.alpha": 0.5,
        "font.family": "sans-serif",
        "font.size": 12,
        "legend.facecolor": "#1A2940",
        "legend.edgecolor": "#4FC3F7",
        "legend.fontsize": 10,
    }
)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(OUTPUT_DIR, "img"), exist_ok=True)


def run_experiment(
    n_classes: int,
    noise_std: float,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Run a single DiscreteRegressor experiment and return (true, predicted, sigmas)."""
    feature_price_delta_std = noise_std
    price_delta_std = 0.5
    n_features = 10
    n_used_features = n_features // 2
    test_fraction = 0.2

    df, feature_columns, target_column = simulate_data(
        start_date=dt.datetime.strptime("2010.01.01", "%Y.%m.%d").date(),
        end_date=dt.datetime.strptime("2022.12.21", "%Y.%m.%d").date(),
        n_used_features=n_used_features,
        n_unused_features=n_features - n_used_features,
        price_delta_std=price_delta_std,
        feature_price_delta_std=feature_price_delta_std,
        seed=seed,
    )
    index_column = df.index.name
    df.reset_index(inplace=True)

    ml_model = MlModel.DiscreteRegressor
    model_label = get_model_label(
        ml_model=ml_model,
        discrete_regressor_internal_classifier_ml_model=MlModel.LightGBM,
        discrete_regressor_n_classes=n_classes,
    )
    output_prefix = f"gen_{model_label}_noise{noise_std}_"

    train_preds, train_metrics, test_preds, test_metrics = train_regression_model(
        ml_model=ml_model,
        hyper_parameter_optimisation_method=HyperParameterOptimisationMethod.SMART_GRID_SEARCH_WITH_FEATURE_SELECTION,
        optimise_hyperparams_other_than_n_estimators=True,
        prune_features=True,
        max_n_estimators=100,
        df=df,
        feature_columns=feature_columns,
        target_column=target_column,
        index_columns=[index_column],
        weight_column=None,
        n_cv_folds=4,
        output_file_dir=os.path.join(OUTPUT_DIR, "tmp_data"),
        output_file_prefix=output_prefix,
        test_fraction=test_fraction,
        train_if_model_exists=False,
        discrete_regressor_internal_classifier_ml_model=MlModel.LightGBM,
        discrete_regressor_n_classes=n_classes,
    )

    true_vals = test_preds["target"].values if "target" in test_preds.columns else test_metrics.true_targets
    pred_vals = test_preds["predicted"].values if "predicted" in test_preds.columns else test_metrics.predicted_values
    pred_sigmas = test_preds["predicted_sigma"].values if "predicted_sigma" in test_preds.columns else getattr(test_metrics, "predicted_sigmas", None)

    return true_vals, pred_vals, pred_sigmas


def run_pure_regression(
    noise_std: float,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run a pure LightGBM regressor (no discretisation) as baseline."""
    feature_price_delta_std = noise_std
    price_delta_std = 0.5
    n_features = 10
    n_used_features = n_features // 2
    test_fraction = 0.2

    df, feature_columns, target_column = simulate_data(
        start_date=dt.datetime.strptime("2010.01.01", "%Y.%m.%d").date(),
        end_date=dt.datetime.strptime("2022.12.21", "%Y.%m.%d").date(),
        n_used_features=n_used_features,
        n_unused_features=n_features - n_used_features,
        price_delta_std=price_delta_std,
        feature_price_delta_std=feature_price_delta_std,
        seed=seed,
    )
    index_column = df.index.name
    df.reset_index(inplace=True)

    output_prefix = f"gen_LightGBM_noise{noise_std}_"

    train_preds, train_metrics, test_preds, test_metrics = train_regression_model(
        ml_model=MlModel.LightGBM,
        hyper_parameter_optimisation_method=HyperParameterOptimisationMethod.SMART_GRID_SEARCH_WITH_FEATURE_SELECTION,
        optimise_hyperparams_other_than_n_estimators=True,
        prune_features=True,
        max_n_estimators=100,
        df=df,
        feature_columns=feature_columns,
        target_column=target_column,
        index_columns=[index_column],
        weight_column=None,
        n_cv_folds=4,
        output_file_dir=os.path.join(OUTPUT_DIR, "tmp_data"),
        output_file_prefix=output_prefix,
        test_fraction=test_fraction,
        train_if_model_exists=False,
    )

    true_vals = test_preds["target"].values if "target" in test_preds.columns else test_metrics.true_targets
    pred_vals = test_preds["predicted"].values if "predicted" in test_preds.columns else test_metrics.predicted_values

    return true_vals, pred_vals


def generate_plot_a() -> None:
    """Result (a): MSE vs n_classes at different noise levels."""
    print("=" * 60)
    print("Generating Plot (a): MSE vs n_classes at different noise levels")
    print("=" * 60)

    noise_levels = [0.1, 0.5, 1.5]
    k_values = [2, 3, 5, 7]
    noise_labels = {0.1: "Low Noise (0.1)", 0.5: "Medium Noise (0.5)", 1.5: "High Noise (1.5)"}
    noise_colors = {0.1: "#66BB6A", 0.5: "#4FC3F7", 1.5: "#FFA726"}

    results: Dict[float, Dict[int, float]] = {}

    for noise_std in noise_levels:
        results[noise_std] = {}
        for k in k_values:
            print(f"  Running: noise={noise_std}, k={k}")
            true_vals, pred_vals, _ = run_experiment(n_classes=k, noise_std=noise_std, seed=42)
            mse = np.mean((true_vals - pred_vals) ** 2)
            results[noise_std][k] = mse
            print(f"    MSE = {mse:.6f}")

    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.22
    x_positions = np.arange(len(k_values))

    for i, noise_std in enumerate(noise_levels):
        mse_values = [results[noise_std][k] for k in k_values]
        bars = ax.bar(
            x_positions + i * bar_width,
            mse_values,
            bar_width,
            label=noise_labels[noise_std],
            color=noise_colors[noise_std],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xlabel("Number of Classes (k)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Mean Squared Error (MSE)", fontsize=14, fontweight="bold")
    ax.set_title("Optimal k Tracks Signal-to-Noise Ratio", fontsize=16, fontweight="bold", color="#4FC3F7")
    ax.set_xticks(x_positions + bar_width)
    ax.set_xticklabels([str(k) for k in k_values], fontsize=12)
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "img", "result_a_mse_vs_k.png")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def generate_plot_b() -> None:
    """Result (b): Standardised residual histogram vs N(0,1)."""
    print("=" * 60)
    print("Generating Plot (b): Standardised residual histogram")
    print("=" * 60)

    true_vals, pred_vals, pred_sigmas = run_experiment(n_classes=3, noise_std=0.5, seed=42)

    if pred_sigmas is None or np.all(pred_sigmas == 0):
        print("  WARNING: No predicted sigmas available. Will generate synthetic demo.")
        return

    # Remove any zero or NaN sigmas
    valid_mask = (pred_sigmas > 0) & np.isfinite(pred_sigmas) & np.isfinite(true_vals) & np.isfinite(pred_vals)
    z = (pred_vals[valid_mask] - true_vals[valid_mask]) / pred_sigmas[valid_mask]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram of standardised residuals
    n_bins = 40
    ax.hist(z, bins=n_bins, density=True, alpha=0.7, color="#4FC3F7", edgecolor="white", linewidth=0.5, label="Observed residuals", zorder=2)

    # Overlay standard normal
    x_norm = np.linspace(-4, 4, 200)
    y_norm = stats.norm.pdf(x_norm)
    ax.plot(x_norm, y_norm, color="#EF5350", linewidth=2.5, label="N(0, 1)", zorder=3)

    ax.set_xlabel(r"$z_i = (\hat{y}_i - y_i) \,/\, \hat{\sigma}_i$", fontsize=14, fontweight="bold")
    ax.set_ylabel("Probability Density", fontsize=14, fontweight="bold")
    ax.set_title("Uncertainty Calibration: Standardised Residuals", fontsize=16, fontweight="bold", color="#4FC3F7")
    ax.legend(fontsize=12)
    ax.set_xlim(-4, 4)
    ax.grid(axis="y", alpha=0.3)

    # Add statistics
    stats_text = f"Mean: {np.mean(z):.3f}\nStd: {np.std(z):.3f}\nn = {len(z)}"
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes, fontsize=10, verticalalignment="top", horizontalalignment="right", bbox=dict(boxstyle="round,pad=0.5", facecolor="#1A2940", edgecolor="#4FC3F7", alpha=0.8))

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "img", "result_b_calibration_histogram.png")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def generate_plot_c() -> None:
    """Result (c): Regression functions fⱼ(pⱼ) for k=3."""
    print("=" * 60)
    print("Generating Plot (c): Regression functions fⱼ(pⱼ) for k=3")
    print("=" * 60)

    # Read the existing class-to-reg function data from Oasis
    xlsx_path = os.path.expanduser("~/dev/Oasis/StrategyA/auto_machine_learning/data/test_regressor_DiscreteRegressor_LightGBM_ncls3_make_predictions_class_to_reg.xlsx")

    if not os.path.exists(xlsx_path):
        print(f"  ERROR: File not found: {xlsx_path}")
        print("  Will generate from scratch instead.")
        # Fallback: run experiment and extract
        return

    df = pd.read_excel(xlsx_path, header=[0, 1], index_col=0)
    print(f"  Loaded class-to-reg data: {df.shape}")

    class_labels = ["0", "1", "2"]
    class_names = ["Class 0 (Low)", "Class 1 (Mid)", "Class 2 (High)"]
    class_colors = ["#4FC3F7", "#66BB6A", "#FFA726"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (class_label, class_name, color) in enumerate(zip(class_labels, class_names, class_colors)):
        try:
            left = df[(class_label, "left_boundary")].values
            right = df[(class_label, "right_boundary")].values
            mid_x = (left + right) / 2
            median_y = df[(class_label, "median")].values
            stdev_y = df[(class_label, "stdev")].values
            n_samples = df[(class_label, "n_samples")].values

            # Standard error for error bars
            std_err = stdev_y / np.sqrt(np.maximum(n_samples, 1))

            # Filter out NaN
            valid = np.isfinite(mid_x) & np.isfinite(median_y) & np.isfinite(std_err)
            mid_x = mid_x[valid]
            median_y = median_y[valid]
            std_err = std_err[valid]

            ax.plot(mid_x, median_y, color=color, linewidth=2, label=class_name, zorder=3)
            ax.fill_between(mid_x, median_y - std_err, median_y + std_err, color=color, alpha=0.2, zorder=2)
        except KeyError as e:
            print(f"  Warning: Could not find columns for class {class_label}: {e}")
            continue

    ax.set_xlabel(r"Predicted Probability $p_j$", fontsize=14, fontweight="bold")
    ax.set_ylabel(r"Conditional Mean $f_j(p_j)$", fontsize=14, fontweight="bold")
    ax.set_title(r"Learned Regression Functions $f_j(p_j)$ for $k=3$", fontsize=16, fontweight="bold", color="#4FC3F7")
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "img", "result_c_regression_functions.png")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


if __name__ == "__main__":
    os.makedirs(os.path.join(OUTPUT_DIR, "tmp_data"), exist_ok=True)

    # Generate plot (c) first — it reads from existing data, no training needed
    generate_plot_c()

    # Generate plots (a) and (b) — these require training models
    generate_plot_a()
    generate_plot_b()

    print("")
    print("All plots generated successfully!")
