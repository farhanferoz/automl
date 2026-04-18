"""Base class for thin baseline model wrappers.

Provides default implementations for BaseModel abstract methods that are
not critical for basic fit/predict workflows, reducing boilerplate in
concrete baseline wrappers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from automl_package.enums import TaskType
from automl_package.models.base import BaseModel


class BaselineModel(BaseModel):
    """Base class for external-library baseline wrappers.

    Subclasses must implement:
        - name (property)
        - _fit_single(x_train, y_train, x_val, y_val, forced_iterations)
        - predict(x, filter_data)

    Optional overrides for richer functionality:
        - predict_uncertainty(x, filter_data) — for probabilistic models
        - get_hyperparameter_search_space() — for Optuna HPO
        - get_num_parameters() — for parameter counting
    """

    def __init__(self, **kwargs: Any) -> None:
        # Filter kwargs to only those accepted by BaseModel
        base_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            in {
                "task_type",
                "early_stopping_rounds",
                "validation_fraction",
                "test_fraction",
                "cv_folds",
                "split_strategy",
                "optimize_hyperparameters",
                "n_trials",
                "search_space_override",
                "output_dir",
                "calculate_feature_importance",
                "feature_selection_threshold",
                "shap_max_data_points",
                "uncertainty_method",
            }
        }
        base_kwargs.setdefault("task_type", TaskType.REGRESSION)
        base_kwargs.setdefault("calculate_feature_importance", False)
        super().__init__(**base_kwargs)
        self._extra_params = {k: v for k, v in kwargs.items() if k not in base_kwargs}

    def predict_proba(self, x: np.ndarray | pd.DataFrame, filter_data: bool = True) -> np.ndarray:
        raise NotImplementedError(f"{self.name} does not support predict_proba.")

    def _clone(self) -> BaselineModel:
        params = self.get_params()
        params.update(self._extra_params)
        return self.__class__(**params)

    def get_hyperparameter_search_space(self) -> dict[str, Any]:
        return {}

    def cross_validate(self, x: np.ndarray, y: np.ndarray, cv: int) -> dict[str, Any]:
        raise NotImplementedError(f"{self.name} does not support cross_validate via this interface.")

    def get_classifier_predictions(self, x: np.ndarray | pd.DataFrame, y_true_original: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError(f"{self.name} does not support get_classifier_predictions.")

    def get_num_parameters(self) -> int:
        return 0

    def get_shap_explainer_info(self) -> dict[str, Any]:
        return {}
