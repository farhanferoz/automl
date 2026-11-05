"""The model class registered for each ``ModelName`` (imported on demand), for anything that names a model in a config.

Moved here from ``experiments/registry.py`` so the library itself can resolve a named model (a composite model's inner
classifier, ``ClassifierRegressionModel.base_classifier_class``) without importing the experiment runner.
"""

from __future__ import annotations

import importlib
from enum import Enum
from typing import TYPE_CHECKING

from automl_package.enums import ModelName

if TYPE_CHECKING:
    from automl_package.models.base import BaseModel

# ModelName member name -> "module:Class". Keyed by member NAME so a member the enum has not gained yet is simply absent.
_CLASS_PATHS: dict[str, str] = {
    "NORMAL_EQUATION_LINEAR_REGRESSION": "automl_package.models.normal_equation_linear_regression:NormalEquationLinearRegression",
    "PYTORCH_LINEAR_REGRESSION": "automl_package.models.pytorch_linear_regression:PyTorchLinearRegression",
    "PYTORCH_NEURAL_NETWORK": "automl_package.models.neural_network:PyTorchNeuralNetwork",
    "FLEXIBLE_NEURAL_NETWORK": "automl_package.models.flexible_neural_network:FlexibleHiddenLayersNN",
    "XGBOOST": "automl_package.models.xgboost_model:XGBoostModel",
    "LIGHTGBM": "automl_package.models.lightgbm_model:LightGBMModel",
    "CATBOOST": "automl_package.models.catboost_model:CatBoostModel",
    "CLASSIFIER_REGRESSION": "automl_package.models.classifier_regression:ClassifierRegressionModel",
    "PROBABILISTIC_REGRESSION": "automl_package.models.probabilistic_regression:ProbabilisticRegressionModel",
    "LSTM_FORECASTER": "automl_package.models.lstm_model:PyTorchLSTMForecaster",
}


def model_class(name: ModelName) -> type[BaseModel]:
    """The class registered for ``name``."""
    path = _CLASS_PATHS.get(name.name)
    if path is None:
        raise ValueError(f"No class registered for {name.name}; registered: {sorted(_CLASS_PATHS)}")
    module, cls = path.split(":")
    return getattr(importlib.import_module(module), cls)


def resolve_model_class(value: type[BaseModel] | ModelName | str) -> type[BaseModel]:
    """A model class given as the class itself, a ``ModelName`` member, or a member's name or value (as a config writes it)."""
    if isinstance(value, type):
        return value
    if isinstance(value, Enum):
        return model_class(ModelName(value.value))
    text = str(value)
    if text in ModelName.__members__:
        return model_class(ModelName[text])
    try:
        return model_class(ModelName(text))
    except ValueError as exc:
        raise ValueError(f"{value!r} names no model: use a ModelName member name, one of {sorted(ModelName.__members__)}") from exc
