"""Debugging script for independent weights neural network."""

import os
import shutil

import numpy as np
from bokeh.io import output_file, save
from bokeh.models import Legend, LegendItem
from bokeh.palettes import Category10
from bokeh.plotting import figure

from automl_package.enums import UncertaintyMethod
from automl_package.logger import logger
from automl_package.models.neural_network import PyTorchNeuralNetwork


# --- Data Generation (Copied from flexible_nn_showcase.py) ---
def create_synthetic_data(n_samples: int = 1000, random_seed: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Creates a synthetic dataset with varying complexity."""
    if random_seed is not None:
        np.random.seed(random_seed)
    x = np.random.uniform(-5, 5, n_samples)

    # Define the piecewise function
    y_true = np.piecewise(
        x,
        [x < 0, x >= 0],
        [
            lambda x_lin: 0.5 * x_lin,
            lambda x_nonlin: 0.5 * x_nonlin + np.sin(2 * np.pi * x_nonlin) * np.cos(1 * np.pi * x_nonlin) + np.sin(4 * np.pi * x_nonlin),
        ],
    )

    # Add Gaussian noise
    noise = np.random.normal(0, 0.2, n_samples)
    y_noisy = y_true + noise

    # Sort for plotting
    sort_indices = np.argsort(x)
    x_sorted = x[sort_indices]
    y_noisy_sorted = y_noisy[sort_indices]
    y_true_sorted = y_true[sort_indices]

    return x_sorted.reshape(-1, 1), y_noisy_sorted, y_true_sorted


# ... (rest of the existing code) ...


def plot_results_bokeh(
    x_data: np.ndarray,
    y_data: np.ndarray,
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    results: dict[str, float],
    output_path: str,
) -> None:
    """Generates an interactive Bokeh plot for model comparison."""
    p = figure(
        width=1200,
        height=700,
        title="Model Predictions vs. True Function",
        x_axis_label="Input (x)",
        y_axis_label="Output (y)",
    )

    # Noisy Data
    noisy_data_glyph = p.scatter(x_data.flatten(), y_data, color="gray", alpha=0.3, size=8)

    # True Function
    true_function_glyph = p.line(x_data.flatten(), y_true, line_dash="dashed", line_color="black", line_width=2)

    colors = Category10[10]
    legend_items = [
        LegendItem(label="Noisy Data", renderers=[noisy_data_glyph]),
        LegendItem(label="True Function", renderers=[true_function_glyph]),
    ]

    for i, (name, y_pred) in enumerate(predictions.items()):
        line_style = "solid"  # Always solid for this plot
        line_glyph = p.line(
            x_data.flatten(),
            y_pred,
            line_color=colors[i % len(colors)],
            line_width=2.5,
            line_dash=line_style,
        )
        legend_items.append(LegendItem(label=f"{name} (MSE: {results[name]:.4f})", renderers=[line_glyph]))

    legend = Legend(
        items=legend_items,
        location="top_left",
        click_policy="hide",
        background_fill_alpha=0.6,
    )
    p.add_layout(legend)

    output_file(output_path)
    save(p)
    logger.info(f"Saved main prediction comparison plot to {output_path}")


# --- Main Debugging Run for 2-Network Approach ---
def run_debug() -> None:
    """Runs the main debugging script."""
    random_seed = 42
    x, y, y_true = create_synthetic_data(random_seed=random_seed)

    output_dir = "debug_two_network_approach_results"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    logger.info("--- Starting 2-Network Approach Debugging Run ---")

    # --- Split Data ---
    x_linear = x[x[:, 0] < 0]
    y_linear = y[x[:, 0] < 0]

    x_nonlinear = x[x[:, 0] >= 0]
    y_nonlinear = y[x[:, 0] >= 0]

    # --- Train Linear Network (1 hidden layer) ---
    logger.info("Training Linear Network (1 hidden layer) for x < 0...")
    linear_model = PyTorchNeuralNetwork(
        hidden_layers=1,
        hidden_size=64,
        n_epochs=200,
        learning_rate=0.01,
        uncertainty_method=UncertaintyMethod.CONSTANT,
        random_seed=random_seed,
        early_stopping_rounds=20,
    )
    linear_model.fit(x_linear, y_linear)
    linear_save_path = os.path.join(output_dir, "Linear_Network_1_Layer")
    y_pred_linear = linear_model.evaluate(x_linear, y_linear, save_path=linear_save_path)
    mse_linear = np.mean((y_linear - y_pred_linear) ** 2)
    logger.info(f"  -> Linear Network MSE (x < 0): {mse_linear:.4f}")

    # --- Train Non-Linear Network (2 hidden layers) ---
    logger.info("Training Non-Linear Network (2 hidden layers) for x >= 0...")
    nonlinear_model = PyTorchNeuralNetwork(
        hidden_layers=3,
        hidden_size=128,
        n_epochs=400,
        learning_rate=0.005,
        uncertainty_method=UncertaintyMethod.CONSTANT,
        random_seed=random_seed,
        early_stopping_rounds=20,
    )
    nonlinear_model.fit(x_nonlinear, y_nonlinear)
    nonlinear_save_path = os.path.join(output_dir, "NonLinear_Network_2_Layers")
    y_pred_nonlinear = nonlinear_model.evaluate(x_nonlinear, y_nonlinear, save_path=nonlinear_save_path)
    mse_nonlinear = np.mean((y_nonlinear - y_pred_nonlinear) ** 2)
    logger.info(f"  -> Non-Linear Network MSE (x >= 0): {mse_nonlinear:.4f}")

    # --- Combine Predictions and Calculate Overall MSE ---
    y_pred_combined = np.zeros_like(y)
    y_pred_combined[x[:, 0] < 0] = y_pred_linear
    y_pred_combined[x[:, 0] >= 0] = y_pred_nonlinear

    overall_mse = np.mean((y - y_pred_combined) ** 2)
    logger.info(f"\nOverall Combined 2-Network Model MSE: {overall_mse:.4f}")
    logger.info(f"Plots saved to: {output_dir}")

    # --- Generate Combined Plot ---
    combined_predictions = {"2-Network Combined Model": y_pred_combined}
    combined_results = {"2-Network Combined Model": overall_mse}
    combined_plot_path = os.path.join(output_dir, "combined_2_network_predictions.html")
    plot_results_bokeh(x, y, y_true, combined_predictions, combined_results, combined_plot_path)

    logger.info("--- 2-Network Approach Debugging Run Complete ---")


if __name__ == "__main__":
    run_debug()
