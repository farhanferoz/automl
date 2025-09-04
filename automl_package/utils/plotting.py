"""Plotting utilities."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from automl_package.enums import RegressionStrategy


def plot_feature_importance(
    feature_importances: dict[str, float], file_path: str
) -> None:
    """Plots feature importances and saves the plot to a file.

    Args:
        feature_importances (Dict[str, float]): A dictionary of feature importances.
        file_path (str): The path to save the plot to.
    """
    feature_importance_df = pd.DataFrame(
        list(feature_importances.items()), columns=["Feature", "Importance"]
    )
    feature_importance_df = feature_importance_df.sort_values(
        by="Importance", ascending=False
    )

    plt.figure(figsize=(10, 8))
    sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()


def plot_nn_probability_mappers(
    mapper_model: torch.nn.Module,
    regression_strategy: RegressionStrategy,
    n_classes: int,
    class_boundaries: np.ndarray,
    device: torch.device,
    plot_path: str,
    model_name: str,
) -> None:
    """Plots the functions that map class probabilities to regression values for NN mappers.

    Args:
        mapper_model (torch.nn.Module): The neural network mapper model.
        regression_strategy (RegressionStrategy): The regression strategy used.
        n_classes (int): The number of classes.
        class_boundaries (np.ndarray): The boundaries of the classes.
        device (torch.device): The torch device.
        plot_path (str): The path to save the plot to.
        model_name (str): The name of the model.
    """
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.figure(figsize=(15, 10))
    mapper_model.eval()

    with torch.no_grad():
        # Create a grid of probability distributions
        # For each class, vary its probability from 0 to 1, distributing the rest among others
        for i in range(n_classes):
            probas_range = torch.zeros(100, n_classes, device=device)
            p_i = torch.linspace(0, 1, 100, device=device)
            probas_range[:, i] = p_i

            # Distribute (1 - p_i) among the other n-1 classes
            if n_classes > 1:
                remaining_p = (1 - p_i) / (n_classes - 1)
                for j in range(n_classes):
                    if i != j:
                        probas_range[:, j] = remaining_p

            # Get predictions from the mapper
            if regression_strategy == RegressionStrategy.SEPARATE_HEADS:
                output_to_plot = mapper_model.heads[i](p_i.unsqueeze(1)).cpu().numpy()
            else:
                mapped_values = mapper_model(probas_range).cpu().numpy()
                output_to_plot = mapped_values.mean(axis=1)

            lower_bound_str = f"{class_boundaries[i]:.2f}"
            upper_bound_str = f"{class_boundaries[i+1]:.2f}"
            label_text = f"Class {i} (Range: {lower_bound_str}-{upper_bound_str})"

            plt.plot(p_i.cpu().numpy(), output_to_plot, label=label_text)

    plt.title(f"NN Probability Mappers for {model_name} ({regression_strategy.value})")
    plt.xlabel("Probability of a Single Class (with others distributed)")
    plt.ylabel("Mapped Regression Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
