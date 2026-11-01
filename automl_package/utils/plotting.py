"""Plotting utilities."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from automl_package.enums import RegressionStrategy


def plot_feature_importance(feature_importances: dict[str, float], file_path: str) -> None:
    """Plots feature importances and saves the plot to a file.

    Args:
        feature_importances (Dict[str, float]): A dictionary of feature importances.
        file_path (str): The path to save the plot to.
    """
    feature_importance_df = pd.DataFrame(list(feature_importances.items()), columns=["Feature", "Importance"])
    feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()


def evaluate_mapper_heads(
    mapper_model: torch.nn.Module,
    regression_strategy: RegressionStrategy,
    n_classes: int,
    device: torch.device,
    n_points: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluates every class's regression head on a grid of probabilities.

    Args:
        mapper_model (torch.nn.Module): The neural network mapper model.
        regression_strategy (RegressionStrategy): The regression strategy used.
        n_classes (int): The number of classes.
        device (torch.device): The torch device.
        n_points (int): Grid size.

    Returns:
        tuple[np.ndarray, np.ndarray]: The probability grid, shape ``(n_points,)``, and the head outputs, shape ``(n_points, n_classes)``
        (column ``i`` is class ``i``'s mean output as its own probability runs over the grid; other classes share the remainder
        for the strategies whose output depends on them).
    """
    mapper_model.eval()
    p_range = torch.linspace(0, 1, n_points, device=device)
    columns = []
    with torch.no_grad():
        if regression_strategy == RegressionStrategy.SEPARATE_HEADS:
            # Each head is independent: one batch, every column the same probability range.
            all_head_outputs = mapper_model.forward_per_class(p_range.unsqueeze(1).repeat(1, n_classes))
            columns = [all_head_outputs[:, i, 0].cpu().numpy() for i in range(n_classes)]
        else:
            # The output for one class can depend on the others, so each step is a valid probability distribution.
            for i in range(n_classes):
                probas_range = torch.zeros(n_points, n_classes, device=device)
                probas_range[:, i] = p_range
                if n_classes > 1:
                    remaining_p = (1 - p_range) / (n_classes - 1)
                    for j in range(n_classes):
                        if i != j:
                            probas_range[:, j] = remaining_p
                if regression_strategy == RegressionStrategy.SINGLE_HEAD_N_OUTPUTS:
                    columns.append(mapper_model.forward_per_class(probas_range)[:, i, 0].cpu().numpy())
                else:  # SINGLE_HEAD_FINAL_OUTPUT
                    columns.append(mapper_model(probas_range).cpu().numpy().reshape(n_points, -1)[:, 0])
    return p_range.cpu().numpy(), np.stack(columns, axis=1)


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
    p_grid, outputs = evaluate_mapper_heads(mapper_model, regression_strategy, n_classes, device)
    for i in range(n_classes):
        label_text = f"Class {i} (Range: {class_boundaries[i]:.2f}-{class_boundaries[i + 1]:.2f})"
        plt.plot(p_grid, outputs[:, i], label=label_text)

    plt.title(f"NN Probability Mappers for {model_name} ({regression_strategy.value})")
    plt.xlabel("Probability of a Single Class (with others distributed)")
    plt.ylabel("Mapped Regression Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
