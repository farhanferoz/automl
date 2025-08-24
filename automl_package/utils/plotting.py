"""Plotting utilities for the AutoML package."""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from automl_package.enums import RegressionStrategy
from automl_package.logger import logger


def plot_nn_probability_mappers(
    mapper_model: nn.Module,
    regression_strategy: RegressionStrategy,
    n_classes: int,
    class_boundaries: np.ndarray,
    device: torch.device,
    plot_path: str,
    model_name: str,
) -> None:
    """Plots the functions that map class probabilities to regression values for NN mappers."""
    logger.info(f"\n--- Plotting Probability Mappers to {plot_path} ---")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.figure(figsize=(12, 8))

    if regression_strategy == RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT:
        logger.info("Plotting probability mappers is not applicable for the SINGLE_HEAD_FINAL_OUTPUT strategy.")
        plt.close()  # Close the figure if not used
        return

    mapper_model.eval()
    with torch.no_grad():
        if regression_strategy == RegressionStrategy.SEPARATE_HEADS:
            probas_range = torch.linspace(0, 1, 100).reshape(-1, 1).to(device)
            for i, head in enumerate(mapper_model.heads):
                mapped_values = head(probas_range).cpu().numpy()
                if mapped_values.shape[1] > 1:
                    mapped_values = mapped_values[:, 0]

                lower_bound_str = f"{class_boundaries[i]:.2f}"
                upper_bound_str = f"{class_boundaries[i+1]:.2f}"
                label_text = f"Class {i} (Range: {lower_bound_str}-{upper_bound_str})"
                plt.plot(probas_range.cpu().numpy(), mapped_values, label=label_text)

        elif regression_strategy == RegressionStrategy.SINGLE_HEAD_N_OUTPUTS:
            probas_range = torch.linspace(0, 1, 100).reshape(-1, 1).to(device)
            for i in range(n_classes):
                input_probas = torch.zeros(100, n_classes).to(device)
                p_i = probas_range.squeeze()
                input_probas[:, i] = p_i
                if n_classes > 1:
                    remaining_p = (1 - p_i) / (n_classes - 1)
                    for j in range(n_classes):
                        if i == j:
                            continue
                        input_probas[:, j] = remaining_p

                y_output_all_classes = mapper_model.head(input_probas)
                all_mapped_values = y_output_all_classes.reshape(input_probas.shape[0], mapper_model.n_classes, mapper_model.regression_output_size).cpu().numpy()
                mapped_values_for_class_i = all_mapped_values[:, i, 0]

                lower_bound_str = f"{class_boundaries[i]:.2f}"
                upper_bound_str = f"{class_boundaries[i+1]:.2f}"
                label_text = f"Class {i} (Range: {lower_bound_str}-{upper_bound_str})"
                plt.plot(probas_range.cpu().numpy(), mapped_values_for_class_i, label=label_text)

    plt.title(f"Probability Mappers for {model_name}")
    plt.xlabel("Class Probability")
    plt.ylabel("Mapped Original Regression Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    logger.info("Probability mappers plot saved successfully.")
