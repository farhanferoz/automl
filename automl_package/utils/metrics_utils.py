"""Utility functions for calculating metrics."""

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error

from automl_package.enums import Metric, TaskType


def calculate_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Metric,
    task_type: TaskType,
    y_proba: np.ndarray | None = None,
) -> float:
    """Calculates a metric."""
    if task_type == TaskType.REGRESSION:
        if metric == Metric.RMSE:
            return np.sqrt(mean_squared_error(y_true, y_pred))
        if metric == Metric.MSE:
            return mean_squared_error(y_true, y_pred)
        raise ValueError(f"Unsupported regression metric: {metric}")
    if task_type == TaskType.CLASSIFICATION:
        if metric == Metric.ACCURACY:
            return accuracy_score(y_true, y_pred)
        if metric == Metric.LOG_LOSS:
            if y_proba is None:
                raise ValueError("y_proba is required for 'log_loss' metric.")
            return log_loss(y_true, y_proba)
        raise ValueError(f"Unsupported classification metric: {metric}")
    raise ValueError("Task type must be 'regression' or 'classification'.")
