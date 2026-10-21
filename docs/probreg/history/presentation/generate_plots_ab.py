"""
Generate plots (a) and (b) for the Turing Institute presentation.

Uses a standalone DiscreteRegressor implementation with sklearn's
GradientBoostingClassifier to avoid LightGBM API compatibility issues.

Result (a): MSE vs n_classes at different noise levels
Result (b): Standardised residual histogram vs N(0,1)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from typing import Tuple, Optional, Dict, List

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

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def simulate_data(
    n_samples: int, n_used_features: int, n_unused_features: int, noise_std: float, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data with controllable noise level.

    Used features are correlated with the target; unused features are pure noise.
    The target is a cumulative sum of random deltas plus noise.
    """
    rng = np.random.RandomState(seed)
    n_features = n_used_features + n_unused_features

    # Target: random walk deltas
    target_deltas = rng.normal(0, 0.5, size=n_samples)

    # Used features: target delta + noise
    used_features = np.column_stack([rng.normal(target_deltas, noise_std) for _ in range(n_used_features)])

    # Unused features: pure noise
    unused_features = rng.normal(0, 0.5, size=(n_samples, n_unused_features))

    X = np.hstack([used_features, unused_features])
    y = target_deltas

    return X, y


class SimpleDiscreteRegressor:
    """Minimal standalone DiscreteRegressor for generating presentation plots.

    Discretises the target into k balanced classes, trains a classifier,
    then learns empirical mapping functions from predicted probability to
    conditional mean and std.
    """

    def __init__(self, n_classes: int, n_estimators: int = 100) -> None:
        """Initialise the DiscreteRegressor."""
        self.n_classes = n_classes
        self.n_estimators = n_estimators
        self.classifier = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=4, random_state=42)
        self.bin_edges: Optional[np.ndarray] = None
        self.mapping_functions: Dict[int, Dict[str, np.ndarray]] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleDiscreteRegressor":
        """Fit the discrete regressor: discretise, classify, then learn mappings."""
        # Discretise target into k balanced classes
        _, self.bin_edges = pd.qcut(y, q=self.n_classes, retbins=True, labels=False, duplicates="drop")
        y_classes = np.digitize(y, self.bin_edges[1:-1])
        y_classes = np.clip(y_classes, 0, self.n_classes - 1)

        # Train classifier
        self.classifier.fit(X, y_classes)

        # Get training probabilities (out-of-bag style: use same data for simplicity)
        probs = self.classifier.predict_proba(X)

        # Build mapping functions for each class
        n_bins = 30
        for j in range(self.n_classes):
            p_j = probs[:, j]
            # Bin by predicted probability
            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
            median_vals = np.full(n_bins, np.nan)
            mean_vals = np.full(n_bins, np.nan)
            std_vals = np.full(n_bins, np.nan)
            n_in_bin = np.zeros(n_bins, dtype=int)

            for b in range(n_bins):
                mask = (p_j >= bin_edges[b]) & (p_j < bin_edges[b + 1])
                if b == n_bins - 1:
                    mask = (p_j >= bin_edges[b]) & (p_j <= bin_edges[b + 1])
                if np.sum(mask) >= 5:
                    y_in_bin = y[mask]
                    median_vals[b] = np.median(y_in_bin)
                    mean_vals[b] = np.mean(y_in_bin)
                    std_vals[b] = np.std(y_in_bin)
                    n_in_bin[b] = np.sum(mask)

            # Forward-fill NaNs
            df_tmp = pd.DataFrame({"median": median_vals, "mean": mean_vals, "std": std_vals})
            df_tmp = df_tmp.ffill().bfill()

            self.mapping_functions[j] = {
                "bin_edges": bin_edges,
                "bin_centres": bin_centres,
                "median": df_tmp["median"].values,
                "mean": df_tmp["mean"].values,
                "std": df_tmp["std"].values,
                "n_in_bin": n_in_bin,
            }

        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mean and standard deviation for new data."""
        probs = self.classifier.predict_proba(X)
        n = X.shape[0]
        y_pred = np.zeros(n)
        y_var = np.zeros(n)

        for j in range(self.n_classes):
            p_j = probs[:, j]
            mapping = self.mapping_functions[j]

            # Look up the mapping for each sample
            bin_indices = np.digitize(p_j, mapping["bin_edges"][1:])
            bin_indices = np.clip(bin_indices, 0, len(mapping["median"]) - 1)

            mu_j = mapping["median"][bin_indices]
            sigma_j = mapping["std"][bin_indices]

            y_pred += p_j * mu_j
            # Law of Total Variance components
            y_var += p_j * (sigma_j**2)

        # Add between-class variance term
        y_var_between = np.zeros(n)
        for j in range(self.n_classes):
            p_j = probs[:, j]
            mapping = self.mapping_functions[j]
            bin_indices = np.digitize(p_j, mapping["bin_edges"][1:])
            bin_indices = np.clip(bin_indices, 0, len(mapping["median"]) - 1)
            mu_j = mapping["median"][bin_indices]
            y_var_between += p_j * (mu_j - y_pred) ** 2

        y_sigma = np.sqrt(y_var + y_var_between)

        return y_pred, y_sigma


def generate_plot_a() -> None:
    """Result (a): MSE vs n_classes at different noise levels."""
    print("=" * 60)
    print("Generating Plot (a): MSE vs n_classes at different noise levels")
    print("=" * 60)

    noise_levels = [0.1, 0.5, 1.5]
    k_values = [2, 3, 5, 7]
    noise_labels = {0.1: "Low Noise ($\sigma$ = 0.1)", 0.5: "Medium Noise ($\sigma$ = 0.5)", 1.5: "High Noise ($\sigma$ = 1.5)"}
    noise_colors = {0.1: "#66BB6A", 0.5: "#4FC3F7", 1.5: "#FFA726"}

    n_samples = 5000
    n_used = 5
    n_unused = 5

    results: Dict[float, Dict[int, float]] = {}

    for noise_std in noise_levels:
        results[noise_std] = {}
        X, y = simulate_data(n_samples=n_samples, n_used_features=n_used, n_unused_features=n_unused, noise_std=noise_std, seed=42)

        # 80/20 split
        split_idx = int(0.8 * n_samples)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        for k in k_values:
            print(f"  Running: noise={noise_std}, k={k}")
            model = SimpleDiscreteRegressor(n_classes=k, n_estimators=100)
            model.fit(X_train, y_train)
            y_pred, _ = model.predict(X_test)
            mse = np.mean((y_test - y_pred) ** 2)
            results[noise_std][k] = mse
            print(f"    MSE = {mse:.6f}")

    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.22
    x_positions = np.arange(len(k_values))

    for i, noise_std in enumerate(noise_levels):
        mse_values = [results[noise_std][k] for k in k_values]
        ax.bar(
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
    output_path = os.path.join(OUTPUT_DIR, "result_a_mse_vs_k.png")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def generate_plot_b() -> None:
    """Result (b): Standardised residual histogram vs N(0,1)."""
    print("=" * 60)
    print("Generating Plot (b): Standardised residual histogram")
    print("=" * 60)

    n_samples = 5000
    X, y = simulate_data(n_samples=n_samples, n_used_features=5, n_unused_features=5, noise_std=0.5, seed=42)

    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = SimpleDiscreteRegressor(n_classes=3, n_estimators=100)
    model.fit(X_train, y_train)
    y_pred, y_sigma = model.predict(X_test)

    # Compute standardised residuals
    valid_mask = (y_sigma > 1e-8) & np.isfinite(y_sigma) & np.isfinite(y_test) & np.isfinite(y_pred)
    z = (y_pred[valid_mask] - y_test[valid_mask]) / y_sigma[valid_mask]

    print(f"  Standardised residuals: mean={np.mean(z):.4f}, std={np.std(z):.4f}, n={len(z)}")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram
    ax.hist(z, bins=40, density=True, alpha=0.7, color="#4FC3F7", edgecolor="white", linewidth=0.5, label="Observed residuals", zorder=2)

    # Standard normal overlay
    x_norm = np.linspace(-4, 4, 200)
    y_norm = stats.norm.pdf(x_norm)
    ax.plot(x_norm, y_norm, color="#EF5350", linewidth=2.5, label="$\mathcal{N}(0, 1)$", zorder=3)

    ax.set_xlabel(r"$z_i = (\hat{y}_i - y_i) \,/\, \hat{\sigma}_i$", fontsize=14, fontweight="bold")
    ax.set_ylabel("Probability Density", fontsize=14, fontweight="bold")
    ax.set_title("Uncertainty Calibration: Standardised Residuals", fontsize=16, fontweight="bold", color="#4FC3F7")
    ax.legend(fontsize=12)
    ax.set_xlim(-4, 4)
    ax.grid(axis="y", alpha=0.3)

    # Add statistics box
    stats_text = f"Mean: {np.mean(z):.3f}" + "\n" + f"Std: {np.std(z):.3f}" + "\n" + f"n = {len(z)}"
    ax.text(
        0.97,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#1A2940", edgecolor="#4FC3F7", alpha=0.8),
    )

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "result_b_calibration_histogram.png")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


if __name__ == "__main__":
    generate_plot_a()
    generate_plot_b()
    print("")
    print("Done! Plots saved to:", OUTPUT_DIR)
