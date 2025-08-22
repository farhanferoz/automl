"""Showcase of the FlexibleNeuralNetwork model.

This script generates a synthetic dataset with varying complexity and uses it to
demonstrate how a flexible architecture can outperform static ones. It compares
the performance of the FlexibleNeuralNetwork using all available layer selection
methods against a simple linear model and a static neural network.

The script will:
1. Generate a synthetic dataset where the function is linear for x < 0 and
   complex (sinusoidal) for x >= 0.
2. Train multiple models on this data:
   - NormalEquationLinearRegression (baseline)
   - PyTorchNeuralNetwork with a fixed architecture (baseline)
   - FlexibleNeuralNetwork with each of the following layer selection methods:
     - NONE
     - SOFT_GATING
     - GUMBEL_SOFTMAX
     - STE
     - REINFORCE
3. Evaluate all models by calculating the Mean Squared Error (MSE).
4. Generate and save plots:
   - A plot comparing the predictions of all models against the true function.
   - For each flexible model, a plot showing how the number of active layers
     changes with the input, demonstrating architectural adaptability.
"""

import os
import shutil

import numpy as np
import pandas as pd
import torch
from bokeh.io import output_file, save
from bokeh.models import Legend, LegendItem
from bokeh.palettes import Category10
from bokeh.plotting import figure

from automl_package.enums import LayerSelectionMethod, UncertaintyMethod
from automl_package.models.flexible_neural_network import FlexibleHiddenLayersNN
from automl_package.models.neural_network import PyTorchNeuralNetwork
from automl_package.models.normal_equation_linear_regression import NormalEquationLinearRegression


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
            lambda x_nonlin: 0.5 * x_nonlin + np.sin(4 * np.pi * x_nonlin) * np.cos(2 * np.pi * x_nonlin) + np.sin(8 * np.pi * x_nonlin),
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


def plot_results_bokeh(x_data: np.ndarray, y_data: np.ndarray, y_true: np.ndarray, predictions: dict[str, np.ndarray], results: dict[str, float], output_path: str) -> None:
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
        line_style = "dashed" if "Flexible" in name else "solid"
        line_glyph = p.line(
            x_data.flatten(),
            y_pred,
            line_color=colors[i % len(colors)],
            line_width=2.5,
            line_dash=line_style,
        )
        legend_items.append(LegendItem(label=f"{name} (MSE: {results[name]:.4f})", renderers=[line_glyph]))

    legend = Legend(items=legend_items, location="top_left", click_policy="hide", background_fill_alpha=0.6)
    p.add_layout(legend)

    output_file(output_path)
    save(p)
    print(f"Saved main prediction comparison plot to {output_path}")


def run_showcase() -> None:
    """Runs the full showcase demonstration."""
    random_seed = 42
    x, y, y_true = create_synthetic_data(random_seed=random_seed)

    models_to_test = {
        "Linear Regression": NormalEquationLinearRegression(),
        "Static NN (3 Layers)": PyTorchNeuralNetwork(hidden_layers=3, hidden_size=64, n_epochs=100, learning_rate=0.01, uncertainty_method=UncertaintyMethod.CONSTANT),
        "Flexible NN (None)": FlexibleHiddenLayersNN(
            layer_selection_method=LayerSelectionMethod.NONE,
            max_hidden_layers=5,
            n_predictor_layers=0,
            hidden_size=64,
            n_epochs=200,
            learning_rate=0.01,
            uncertainty_method=UncertaintyMethod.CONSTANT,
        ),
        "Flexible NN (Soft Gating)": FlexibleHiddenLayersNN(
            layer_selection_method=LayerSelectionMethod.SOFT_GATING,
            max_hidden_layers=5,
            n_predictor_layers=1,
            hidden_size=64,
            n_epochs=200,
            learning_rate=0.01,
            n_predictor_learning_rate=0.01,
            uncertainty_method=UncertaintyMethod.CONSTANT,
        ),
        "Flexible NN (Gumbel-Softmax)": FlexibleHiddenLayersNN(
            layer_selection_method=LayerSelectionMethod.GUMBEL_SOFTMAX,
            max_hidden_layers=5,
            n_predictor_layers=1,
            hidden_size=64,
            n_epochs=200,
            learning_rate=0.01,
            n_predictor_learning_rate=0.01,
            uncertainty_method=UncertaintyMethod.CONSTANT,
        ),
        "Flexible NN (STE)": FlexibleHiddenLayersNN(
            layer_selection_method=LayerSelectionMethod.STE,
            max_hidden_layers=5,
            n_predictor_layers=1,
            hidden_size=64,
            n_epochs=200,
            learning_rate=0.01,
            n_predictor_learning_rate=0.01,
            uncertainty_method=UncertaintyMethod.CONSTANT,
        ),
        "Flexible NN (Reinforce)": FlexibleHiddenLayersNN(
            layer_selection_method=LayerSelectionMethod.REINFORCE,
            max_hidden_layers=5,
            n_predictor_layers=1,
            hidden_size=64,
            n_epochs=200,
            learning_rate=0.01,
            n_predictor_learning_rate=0.01,
            uncertainty_method=UncertaintyMethod.CONSTANT,
        ),
    }

    results = {}
    predictions = {}

    output_dir = "flexible_nn_showcase_results"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    print("--- Running Model Training and Evaluation ---")
    for name, model in models_to_test.items():
        print(f"Training {name}...")
        # Set random seed for reproducibility of torch models
        if hasattr(model, "random_seed"):
            model.random_seed = random_seed
            torch.manual_seed(random_seed)

        model.fit(x, y)
        save_path = os.path.join(output_dir, name.replace(" ", "_"))
        y_pred = model.evaluate(x, y, save_path=save_path)
        predictions[name] = y_pred
        mse = np.mean((y - y_pred) ** 2)
        results[name] = mse
        print(f"  -> MSE: {mse:.4f}")

    # --- Reporting ---
    print("\n--- Model Performance Summary ---")
    results_df = pd.DataFrame.from_dict(results, orient="index", columns=["MSE"]).sort_values(by="MSE")
    print(results_df)

    # --- Visualization ---
    print(f"\n--- Generating Plots in '{output_dir}' Directory ---")

    # 1. Main Prediction Plot
    main_plot_path = os.path.join(output_dir, "all_model_predictions.html")
    plot_results_bokeh(x, y, y_true, predictions, results, main_plot_path)

    # 2. Individual Layer Activation Plots (already generated by evaluate)
    print("\nIndividual layer activation plots have been generated by the `evaluate` method in their respective folders.")
    print("Showcase complete.")


if __name__ == "__main__":
    run_showcase()
