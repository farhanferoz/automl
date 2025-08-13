"""Metrics calculation and plotting utilities."""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from automl_package.enums import RegressionStrategy
from automl_package.logger import logger


class Metrics:
    """A class for calculating and plotting various machine learning metrics."""

    def __init__(
        self,
        task_type: str,
        model_name: str,
        x_data: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the Metrics calculator.

        Args:
            task_type (str): Type of task ('regression' or 'classification').
            model_name (str): Name of the model.
            x_data (np.ndarray): Input features data.
            y_true (np.ndarray): True target values.
            y_pred (np.ndarray): Predicted target values.
            y_proba (np.ndarray, optional): Predicted probabilities for classification tasks.
            **kwargs: Additional keyword arguments for specific plots (e.g., flexible_nn_n_actual).
        """
        self.task_type = task_type
        self.model_name = model_name
        self.x_data = x_data
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba

        # For Flexible NN architecture plots
        self.flexible_nn_n_actual = kwargs.get("flexible_nn_n_actual")
        self.flexible_nn_n_logits = kwargs.get("flexible_nn_n_logits")
        self.flexible_nn_max_hidden_layers = kwargs.get("flexible_nn_max_hidden_layers")
        self.flexible_nn_feature_scaler = kwargs.get("flexible_nn_feature_scaler")

        # For Probabilistic Regression internal plots
        self.prob_reg_classifier_probabilities = kwargs.get("prob_reg_classifier_probabilities")
        self.prob_reg_regression_head_outputs = kwargs.get("prob_reg_regression_head_outputs")
        self.prob_reg_n_classes = kwargs.get("prob_reg_n_classes")
        self.prob_reg_regression_strategy = kwargs.get("prob_reg_regression_strategy")
        self.prob_reg_uncertainty_method = kwargs.get("prob_reg_uncertainty_method")
        self.prob_reg_X_original = kwargs.get("prob_reg_X_original")
        self.prob_reg_probas_for_plotting = kwargs.get("prob_reg_probas_for_plotting")

    def _calculate_bins(self, data: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray, int]:
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(data, percentiles)
        # Ensure bin edges are unique
        unique_bin_edges = np.unique(bin_edges)
        while len(unique_bin_edges) < len(bin_edges) and n_bins > 1:
            n_bins -= 1
            percentiles = np.linspace(0, 100, n_bins + 1)
            bin_edges = np.percentile(data, percentiles)
            unique_bin_edges = np.unique(bin_edges)
        bin_midpoints = (unique_bin_edges[:-1] + unique_bin_edges[1:]) / 2
        return unique_bin_edges, bin_midpoints, n_bins

    def calculate_all_metrics(self) -> dict[str, float]:
        """Calculates all relevant metrics based on the task type.

        Returns:
            dict: A dictionary of calculated metrics.
        """
        if self.task_type == "regression":
            return self.calculate_regression_metrics()
        return self.calculate_classification_metrics()

    def calculate_regression_metrics(self) -> dict[str, float]:
        """Calculates regression-specific metrics.

        Returns:
            dict: A dictionary of regression metrics.
        """
        return {
            "mae": mean_absolute_error(self.y_true, self.y_pred),
            "rmse": np.sqrt(mean_squared_error(self.y_true, self.y_pred)),
            "r2_score": r2_score(self.y_true, self.y_pred),
        }

    def calculate_classification_metrics(self) -> dict[str, float]:
        """Calculates classification-specific metrics.

        Returns:
            dict: A dictionary of classification metrics.
        """
        metrics = {
            "accuracy": accuracy_score(self.y_true, self.y_pred),
            "precision": precision_score(self.y_true, self.y_pred, average="weighted"),
            "recall": recall_score(self.y_true, self.y_pred, average="weighted"),
            "f1_score": f1_score(self.y_true, self.y_pred, average="weighted"),
        }
        if self.y_proba is not None:
            unique_classes = np.unique(self.y_true)
            if len(unique_classes) == 2:
                # For binary classification, y_proba should be (n_samples,) or (n_samples, 2)
                # If (n_samples, 2), take the probability of the positive class
                y_score_for_roc_auc = self.y_proba[:, 1] if self.y_proba.ndim == 2 else self.y_proba
                metrics["roc_auc"] = roc_auc_score(self.y_true, y_score_for_roc_auc)
            else:
                # For multi-class, y_proba should be (n_samples, n_classes)
                # Use 'ovr' (one-vs-rest) strategy for multi-class ROC AUC
                metrics["roc_auc"] = roc_auc_score(self.y_true, self.y_proba, multi_class="ovr", average="weighted")

            # log_loss expects y_proba to be (n_samples, n_classes) for multi-class, or (n_samples,) for binary
            # If binary and y_proba is (n_samples, 2), log_loss handles it.
            # If binary and y_proba is (n_samples, 2), log_loss handles it.
            metrics["log_loss"] = log_loss(self.y_true, self.y_proba)
        return metrics

    def plot_regression_charts(self, save_path: str) -> None:
        """Generates and saves regression-specific plots.

        Args:
            save_path (str): Directory to save the plots.
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)

        # Predicted vs. Actual Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_true, self.y_pred, alpha=0.5)
        plt.plot([self.y_true.min(), self.y_true.max()], [self.y_true.min(), self.y_true.max()], "--r", linewidth=2)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f"Predicted vs. Actual Values for {self.model_name}")
        plt.savefig(f"{save_path}/predicted_vs_actual.png")
        plt.close()

        # Residuals vs. Predicted Plot
        residuals = self.y_true - self.y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color="r", linestyle="--")
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title(f"Residuals vs. Predicted Values for {self.model_name}")
        plt.savefig(f"{save_path}/residuals_vs_predicted.png")
        plt.close()

    def plot_classification_charts(self, save_path: str) -> None:
        """Generates and saves classification-specific plots.

        Args:
            save_path (str): Directory to save the plots.
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)

        # Confusion Matrix
        cm = confusion_matrix(self.y_true, self.y_pred)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix for {self.model_name}")
        plt.colorbar()
        tick_marks = np.arange(len(np.unique(self.y_true)))
        plt.xticks(tick_marks, np.unique(self.y_true), rotation=45)
        plt.yticks(tick_marks, np.unique(self.y_true))
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.savefig(f"{save_path}/confusion_matrix.png")
        plt.close()

        if self.y_proba is not None:
            # ROC Curve
            unique_classes = np.unique(self.y_true)
            if len(unique_classes) == 2:
                # Binary ROC Curve
                fpr, tpr, _ = roc_curve(self.y_true, self.y_proba[:, 1])
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_score(self.y_true, self.y_proba[:, 1]):.2f})")
                plt.plot([0, 1], [0, 1], "k--")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(f"ROC Curve for {self.model_name}")
                plt.legend(loc="lower right")
                plt.savefig(f"{save_path}/roc_curve.png")
                plt.close()

                # Binary Precision-Recall Curve
                precision, recall, _ = precision_recall_curve(self.y_true, self.y_proba[:, 1])
                plt.figure(figsize=(8, 6))
                plt.plot(recall, precision, label="Precision-Recall Curve")
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title(f"Precision-Recall Curve for {self.model_name}")
                plt.legend(loc="lower left")
                plt.savefig(f"{save_path}/precision_recall_curve.png")
                plt.close()
            else:
                # Multi-class ROC Curve (One-vs-Rest)
                plt.figure(figsize=(10, 8))
                for i, class_label in enumerate(unique_classes):
                    # Binarize the true labels for the current class
                    y_true_bin = (self.y_true == class_label).astype(int)
                    # Get the probability estimates for the current class
                    y_proba_class = self.y_proba[:, i]
                    fpr, tpr, _ = roc_curve(y_true_bin, y_proba_class)
                    auc_score = roc_auc_score(y_true_bin, y_proba_class)
                    plt.plot(fpr, tpr, label=f"ROC Curve (Class {class_label}, AUC = {auc_score:.2f})")
                plt.plot([0, 1], [0, 1], "k--")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(f"Multi-class ROC Curve for {self.model_name} (One-vs-Rest)")
                plt.legend(loc="lower right")
                plt.savefig(f"{save_path}/multiclass_roc_curve.png")
                plt.close()

                # Multi-class Precision-Recall Curve (One-vs-Rest)
                plt.figure(figsize=(10, 8))
                for i, class_label in enumerate(unique_classes):
                    y_true_bin = (self.y_true == class_label).astype(int)
                    y_proba_class = self.y_proba[:, i]
                    precision, recall, _ = precision_recall_curve(y_true_bin, y_proba_class)
                    plt.plot(recall, precision, label=f"PR Curve (Class {class_label})")
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title(f"Multi-class Precision-Recall Curve for {self.model_name} (One-vs-Rest)")
                plt.legend(loc="lower left")
                plt.savefig(f"{save_path}/multiclass_precision_recall_curve.png")
                plt.close()

            self.plot_predicted_vs_true_classification_rate(save_path)
            self.plot_completeness_vs_purity(save_path)

    def plot_predicted_vs_true_classification_rate(self, save_path: str) -> None:
        """Plots predicted vs. true classification rate.

        Args:
            save_path (str): Directory to save the plot.
        """
        max_proba = np.max(self.y_proba, axis=1)
        predicted_classes = np.argmax(self.y_proba, axis=1)
        is_correct = predicted_classes == self.y_true

        n_bins = max(5, len(self.y_true) // 20)
        bin_edges, bin_midpoints, n_bins = self._calculate_bins(max_proba, n_bins)

        bin_means: list[float] = []
        bin_medians: list[float] = []
        inverse_cumulative_means: list[float] = []

        for i in range(n_bins):
            mask = (max_proba >= bin_edges[i]) & (max_proba < bin_edges[i + 1]) if i < n_bins - 1 else (max_proba >= bin_edges[i]) & (max_proba <= bin_edges[i + 1])

            if np.any(mask):
                bin_means.append(np.mean(is_correct[mask]))
                bin_medians.append(np.median(is_correct[mask]))
            else:
                bin_means.append(0.0)
                bin_medians.append(0.0)

            inv_cum_mask = max_proba >= bin_edges[i]
            if np.any(inv_cum_mask):
                inverse_cumulative_means.append(np.mean(is_correct[inv_cum_mask]))
            else:
                inverse_cumulative_means.append(0.0)

        plt.figure(figsize=(12, 8))
        plt.plot(bin_midpoints, bin_means, "o-", label="Mean Classification Rate")
        plt.plot(bin_midpoints, bin_medians, "s--", label="Median Classification Rate")
        plt.plot(bin_midpoints, inverse_cumulative_means, "^-.", label="Inverse Cumulative Mean Rate")
        plt.xlabel("Bin Midpoint of Maximum Classification Probability")
        plt.ylabel("Classification Rate")
        plt.title(f"Predicted vs. True Classification Rate for {self.model_name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{save_path}/predicted_vs_true_classification_rate.png")
        plt.close()

    def plot_completeness_vs_purity(self, save_path: str) -> None:
        """Plots completeness vs. purity for classification.

        Args:
            save_path (str): Directory to save the plot.
        """
        max_proba = np.max(self.y_proba, axis=1)
        predicted_classes = np.argmax(self.y_proba, axis=1)
        is_correct = predicted_classes == self.y_true

        n_bins = max(5, len(self.y_proba) // 20)
        thresholds, _, _ = self._calculate_bins(max_proba, n_bins)

        completeness: list[float] = []
        purity: list[float] = []

        for t in thresholds:
            mask = max_proba >= t
            if np.any(mask):
                completeness.append(np.sum(mask) / len(self.y_true))
                purity.append(np.mean(is_correct[mask]))
            else:
                completeness.append(0.0)
                purity.append(1.0)  # Purity is 1 if no examples are selected

        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, completeness, "o-", label="Completeness")
        plt.plot(thresholds, purity, "s--", label="Purity")
        plt.xlabel("Threshold Probability")
        plt.ylabel("Rate")
        plt.title(f"Completeness vs. Purity for {self.model_name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{save_path}/completeness_vs_purity.png")
        plt.close()

    def plot_flexible_nn_architecture(self, save_path: str) -> None:
        """Plots the architecture decisions of a Flexible Neural Network.

        Args:
            save_path (str): Directory to save the plots.
        """
        if self.flexible_nn_n_actual is None or self.flexible_nn_n_logits is None or self.x_data is None:
            return

        logger.info(f"Plotting Flexible NN Architecture Decisions to {save_path}/flexible_nn_architecture.png ---")

        n_actual = self.flexible_nn_n_actual
        n_logits = self.flexible_nn_n_logits
        max_hidden_layers = self.flexible_nn_max_hidden_layers
        feature_scaler = self.flexible_nn_feature_scaler

        # Inverse transform x_data to original scale for plotting
        x_original = self.x_data
        if feature_scaler:
            # Handle both 1D and 2D x_data for inverse_transform
            if x_original.ndim == 1:
                x_original = feature_scaler.inverse_transform(x_original.reshape(-1, 1)).flatten()
            else:
                x_original = feature_scaler.inverse_transform(x_original)[:, 0] # Plot against the first feature
        else:
            x_original = x_original.flatten() if x_original.ndim == 1 else x_original[:, 0]

        # Calculate n_probs from logits
        n_probs = softmax(n_logits, axis=1)

        # Calculate weighted average number of layers
        # Layers are 1-indexed, so multiply probabilities by (index + 1)
        layer_indices = np.arange(1, max_hidden_layers + 1)
        weighted_layers = np.sum(n_probs * layer_indices, axis=1)

        fig, axs = plt.subplots(3, 1, figsize=(10, 18))

        # Plot 1: Distribution of chosen active layers (n_actual)
        axs[0].hist(n_actual, bins=np.arange(1, max_hidden_layers + 2), align="left", rwidth=0.8, color="navy")
        axs[0].set_xticks(np.arange(1, max_hidden_layers + 1))
        axs[0].set_xlabel("Number of Active Layers (Actual)")
        axs[0].set_ylabel("Count")
        axs[0].set_title(f"Flexible NN ({self.model_name}) - Chosen Active Layers Distribution")
        axs[0].grid(axis="y", alpha=0.75)

        # Plot 2: Logits for each layer vs. Input Feature
        colors = plt.cm.get_cmap("viridis", max_hidden_layers)
        for i in range(max_hidden_layers):
            axs[1].scatter(x_original, n_logits[:, i], label=f"Layer {i+1} Logit", alpha=0.6, color=colors(i))
        axs[1].set_xlabel("Input Feature")
        axs[1].set_ylabel("Logit Value")
        axs[1].set_title(f"Flexible NN ({self.model_name}) - Layer Logits vs. Input Feature")
        axs[1].legend(loc="upper left")
        axs[1].grid(True)

        # Plot 3: Weighted Average Number of Layers vs. Input Feature
        axs[2].scatter(x_original, weighted_layers, alpha=0.6, color="purple")
        axs[2].set_xlabel("Input Feature")
        axs[2].set_ylabel("Weighted Avg. Layers")
        axs[2].set_title(f"Flexible NN ({self.model_name}) - Weighted Average Layers vs. Input Feature")
        axs[2].grid(True)
        axs[2].set_yticks(np.arange(1, max_hidden_layers + 1, 1))

        plt.tight_layout()
        plt.savefig(f"{save_path}/flexible_nn_architecture.png")
        plt.close()

        logger.info("Flexible NN architecture plots saved successfully.")

    def plot_prob_reg_internal_plots(self, save_path: str) -> None:
        """Plots internal aspects of the Probabilistic Regression model.

        Args:
            save_path (str): Directory to save the plots.
        """
        if self.prob_reg_classifier_probabilities is None or self.prob_reg_regression_head_outputs is None:
            return

        logger.info(f"Plotting Probabilistic Regression internal plots to {save_path} ---")

        # Plot 1: Internal Classifier Probabilities
        plt.figure(figsize=(10, 6))
        colors = plt.cm.get_cmap("viridis", self.prob_reg_n_classes)
        x_original_plot = self.prob_reg_X_original.flatten() if self.prob_reg_X_original.ndim > 1 else self.prob_reg_X_original

        # Sort data by X_original_plot for cleaner lines
        sort_indices = np.argsort(x_original_plot)
        x_original_plot_sorted = x_original_plot[sort_indices]
        prob_reg_classifier_probabilities_sorted = self.prob_reg_classifier_probabilities[sort_indices]

        if prob_reg_classifier_probabilities_sorted.shape[1] > 1:  # Multi-class
            for i in range(self.prob_reg_n_classes):
                plt.plot(x_original_plot_sorted, prob_reg_classifier_probabilities_sorted[:, i], label=f"Class {i} Probability", color=colors(i))
        else:  # Binary
            plt.plot(x_original_plot_sorted, prob_reg_classifier_probabilities_sorted[:, 1], label="Positive Class Probability", color=colors(0))

        plt.title("Internal Classifier Probabilities vs. Input Feature")
        plt.xlabel("Input Feature Value")
        plt.ylabel("Probability")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{save_path}/prob_reg_internal_classifier_probabilities.png")
        plt.close()

        # Plot 2: Regression Head Outputs
        plt.figure(figsize=(12, 8))
        colors = plt.cm.get_cmap("plasma", self.prob_reg_n_classes)

        # Use probas_for_plotting for the x-axis
        probas_range = self.prob_reg_probas_for_plotting[:, 0] if self.prob_reg_probas_for_plotting.ndim > 1 else self.prob_reg_probas_for_plotting

        # Sort the regression head outputs based on the sorted probas_range
        sort_indices = np.argsort(probas_range)
        probas_range_sorted = probas_range[sort_indices]
        sorted_regression_head_outputs = self.prob_reg_regression_head_outputs[sort_indices]

        if self.prob_reg_regression_strategy == RegressionStrategy.SEPARATE_HEADS:
            for i in range(self.prob_reg_n_classes):
                # output is (N, output_size) for each head, so take the mean (index 0)
                output_for_plot = sorted_regression_head_outputs[:, i, 0] if sorted_regression_head_outputs.ndim == 3 else sorted_regression_head_outputs[:, i]
                plt.plot(probas_range_sorted, output_for_plot, label=f"Head {i} Output", color=colors(i))
        elif self.prob_reg_regression_strategy == RegressionStrategy.SINGLE_HEAD_N_OUTPUTS:
            # This strategy returns (N, n_classes, output_size)
            for i in range(self.prob_reg_n_classes):
                output_for_plot = sorted_regression_head_outputs[:, i, 0] if sorted_regression_head_outputs.ndim == 3 else sorted_regression_head_outputs[:, i]
                plt.plot(probas_range_sorted, output_for_plot, label=f"Class {i} Contribution", color=colors(i))
        elif self.prob_reg_regression_strategy == RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT:
            # This strategy returns (N, output_size)
            output_for_plot = sorted_regression_head_outputs[:, 0] if sorted_regression_head_outputs.ndim == 2 else sorted_regression_head_outputs
            plt.plot(probas_range_sorted, output_for_plot, label="Combined Output", color="red")

        plt.title(f"Regression Head Outputs for {self.model_name}")
        plt.xlabel("Input Probability")
        plt.ylabel("Regression Output (Mean)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{save_path}/prob_reg_regression_head_outputs.png")
        plt.close()

        logger.info("Probabilistic Regression internal plots saved successfully.")

    def save_metrics(self, save_path: str) -> None:
        """Saves calculated metrics to a JSON file and generates plots.

        Args:
            save_path (str): Directory to save the metrics and plots.
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        metrics = self.calculate_all_metrics()
        with open(f"{save_path}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        if self.task_type == "regression":
            self.plot_regression_charts(save_path)
        else:
            self.plot_classification_charts(save_path)

        # Call new plotting methods if data is available
        if self.flexible_nn_n_actual is not None:
            self.plot_flexible_nn_architecture(save_path)

        if self.prob_reg_classifier_probabilities is not None:
            self.plot_prob_reg_internal_plots(save_path)
