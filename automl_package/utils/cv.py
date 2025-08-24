"""Cross-validation utilities."""

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.model_selection import KFold

from automl_package.logger import logger


def cross_validate(
    model_class: type,
    model_params: dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
    cv: int,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    random_seed: int | None = None,
) -> dict[str, Any]:
    """Performs cross-validation for a given model.

    Args:
        model_class (type): The model class to instantiate.
        model_params (dict): The parameters to initialize the model with.
        x (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        cv (int): Number of folds for cross-validation.
        metric_func (Callable[[np.ndarray, np.ndarray], float]): The metric function to use for evaluation.
        random_seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        Dict[str, Any]: A dictionary containing the cross-validation scores.
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_seed)
    scores = []

    logger.info(f"Starting cross-validation with {cv} folds.")

    for fold, (train_index, test_index) in enumerate(kf.split(x, y)):
        logger.info(f"--- Fold {fold + 1}/{cv} ---")
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = model_class(**model_params)

        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        score = metric_func(y_test, preds)
        scores.append(score)
        logger.info(f"Fold {fold + 1} score: {score:.4f}")

    mean_score = np.mean(scores)
    std_score = np.std(scores)
    logger.info(f"Cross-validation finished. Mean score: {mean_score:.4f} (+/- {std_score:.4f})")

    return {"test_score": scores, "mean_test_score": mean_score, "std_test_score": std_score}
