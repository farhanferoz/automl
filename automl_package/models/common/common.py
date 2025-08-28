"""Common utility functions for models."""

from typing import Any, List


def get_loss_history(model: Any, use_early_stopping: bool) -> List[float]:
    """Extracts the loss history from a trained model.

    Args:
        model: The trained model instance.
        use_early_stopping (bool): Whether early stopping was used during training.

    Returns:
        A list of validation loss values for each epoch.
    """
    loss_history = []
    if use_early_stopping:

        validation_strs = [
            "validation_0", # XGBoost,
            "valid_0", # LightGBM
            "validation" # CatBoost
        ]
        evals_result_str = "evals_result_"

        for validation_str in validation_strs:
            if hasattr(model, evals_result_str):
                evals_result = getattr(model, evals_result_str)
                if evals_result and (validation_str in evals_result):
                    metric_name = list(evals_result[validation_str].keys())[0]
                    loss_history = evals_result[validation_str][metric_name]
                    break

    return loss_history
