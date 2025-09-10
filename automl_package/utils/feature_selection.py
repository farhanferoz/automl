"""Feature selection utilities."""

from automl_package.logger import logger


def select_features_by_cumulative_importance(feature_importances: dict[str, float], threshold: float = 0.95) -> list[str]:
    """Selects features based on a cumulative importance threshold.

    Args:
        feature_importances (Dict[str, float]): A dictionary of feature importances.
        threshold (float): The cumulative importance threshold.

    Returns:
        List[str]: A list of selected feature names.
    """
    sorted_features = sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)

    cumulative_importance = 0.0
    selected_features = []
    for feature, importance in sorted_features:
        cumulative_importance += importance
        selected_features.append(feature)
        if cumulative_importance >= threshold:
            break

    logger.info(f"Selected {len(selected_features)} features with cumulative importance of {cumulative_importance}")
    return selected_features
