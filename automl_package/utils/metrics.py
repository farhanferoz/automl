import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
import matplotlib.pyplot as plt
import json
from pathlib import Path


class Metrics:
    def __init__(self, task_type, model_name, y_true, y_pred, y_proba=None):
        self.task_type = task_type
        self.model_name = model_name
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba

    def _calculate_bins(self, data, n_bins):
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

    def calculate_all_metrics(self):
        if self.task_type == "regression":
            return self.calculate_regression_metrics()
        else:
            return self.calculate_classification_metrics()

    def calculate_regression_metrics(self):
        metrics = {
            "mae": mean_absolute_error(self.y_true, self.y_pred),
            "rmse": np.sqrt(mean_squared_error(self.y_true, self.y_pred)),
            "r2_score": r2_score(self.y_true, self.y_pred),
        }
        return metrics

    def calculate_classification_metrics(self):
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
            # If binary and y_proba is (n_samples,), log_loss handles it.
            # If binary and y_proba is (n_samples, 2), log_loss handles it.
            metrics["log_loss"] = log_loss(self.y_true, self.y_proba)
        return metrics

    def plot_regression_charts(self, save_path):
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

    def plot_classification_charts(self, save_path):
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

    def plot_predicted_vs_true_classification_rate(self, save_path):
        max_proba = np.max(self.y_proba, axis=1)
        predicted_classes = np.argmax(self.y_proba, axis=1)
        is_correct = predicted_classes == self.y_true

        n_bins = max(5, len(self.y_true) // 20)
        bin_edges, bin_midpoints, n_bins = self._calculate_bins(max_proba, n_bins)

        bin_means = []
        bin_medians = []
        inverse_cumulative_means = []

        for i in range(n_bins):
            if i < n_bins - 1:
                mask = (max_proba >= bin_edges[i]) & (max_proba < bin_edges[i + 1])
            else:
                mask = (max_proba >= bin_edges[i]) & (max_proba <= bin_edges[i + 1])

            if np.any(mask):
                bin_means.append(np.mean(is_correct[mask]))
                bin_medians.append(np.median(is_correct[mask]))
            else:
                bin_means.append(0)
                bin_medians.append(0)

            inv_cum_mask = max_proba >= bin_edges[i]
            if np.any(inv_cum_mask):
                inverse_cumulative_means.append(np.mean(is_correct[inv_cum_mask]))
            else:
                inverse_cumulative_means.append(0)

        plt.figure(figsize=(12, 8))
        plt.plot(bin_midpoints, bin_means, "o-", label="Mean Classification Rate")
        plt.plot(bin_midpoints, bin_medians, "s--", label="Median Classification Rate")
        plt.plot(bin_midpoints, inverse_cumulative_means, "^-.", label="Inverse Cumulative Mean Rate")
        plt.xlabel("Bin Midpoint of Maximum Classification Probability")
        plt.ylabel("Classification Rate")
        plt.title(f"Predicted vs. True Classification Rate for {self.model_name}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_path}/predicted_vs_true_classification_rate.png")
        plt.close()

    def plot_completeness_vs_purity(self, save_path):
        max_proba = np.max(self.y_proba, axis=1)
        predicted_classes = np.argmax(self.y_proba, axis=1)
        is_correct = predicted_classes == self.y_true

        n_bins = max(5, len(self.y_true) // 20)
        thresholds, _, _ = self._calculate_bins(max_proba, n_bins)

        completeness = []
        purity = []

        for t in thresholds:
            mask = max_proba >= t
            if np.any(mask):
                completeness.append(np.sum(mask) / len(self.y_true))
                purity.append(np.mean(is_correct[mask]))
            else:
                completeness.append(0)
                purity.append(1)  # Purity is 1 if no examples are selected

        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, completeness, "o-", label="Completeness")
        plt.plot(thresholds, purity, "s--", label="Purity")
        plt.xlabel("Threshold Probability")
        plt.ylabel("Rate")
        plt.title(f"Completeness vs. Purity for {self.model_name}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_path}/completeness_vs_purity.png")
        plt.close()

    def save_metrics(self, save_path):
        Path(save_path).mkdir(parents=True, exist_ok=True)
        metrics = self.calculate_all_metrics()
        with open(f"{save_path}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        if self.task_type == "regression":
            self.plot_regression_charts(save_path)
        else:
            self.plot_classification_charts(save_path)
