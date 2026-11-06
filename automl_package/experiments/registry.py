"""Builds the predictor of a variant from its ``ModelConfig``: any automl model, or a reference predictor.

Generic by construction: ``ModelName`` maps to a class (imported on demand), ``params`` are checked against the keys
the class accepts (the merged ``_defaults`` of its MRO plus every named ``__init__`` argument in the MRO), strings are
turned into the default's enum type, and the protocol settings are applied by key, not by model. A class that declares
no ``_defaults`` (the tree models, which forward ``**kwargs`` to their booster) has no list of valid keys to check
against, so its extra keys pass through unchecked.

Reused: ``BaseModel``/``PyTorchModelBase._defaults`` for keys and defaults, ``PyTorchNeuralNetwork.build_model`` and
``get_num_parameters`` for the matched width, ``fit_sweep_selector`` for the k sweep, ``Toy`` for the true mean.
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from automl_package.experiments.config import MISSING, HpoConfig, ModelConfig, PredictorKind, ProtocolConfig, coerce_enum
from automl_package.experiments.toys import Toy
from automl_package.models.base import NESTED_PARAM_SEPARATOR, BaseModel
from automl_package.models.registry import model_class
from automl_package.utils.data_handler import TargetScaler

_UNSET = object()


def accepted_keys(cls: type[BaseModel]) -> tuple[dict[str, Any], bool]:
    """``(key -> default, checked)``: every key the class accepts, and whether the list is closed (the class declares ``_defaults``)."""
    keys: dict[str, Any] = {}
    checked = False
    for klass in reversed(cls.__mro__):
        init = klass.__dict__.get("__init__")
        if init is not None:
            for p in inspect.signature(init).parameters.values():
                if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) and p.name != "self":
                    keys[p.name] = _UNSET if p.default is p.empty else p.default
        declared = klass.__dict__.get("_defaults")
        if declared is not None:
            checked = True
            keys.update(declared)
    return keys, checked


def _coerce_params(params: dict[str, Any], keys: dict[str, Any], checked: bool, cls_name: str) -> dict[str, Any]:
    """Checks ``params`` against ``keys`` and turns strings into the default's enum type."""
    unknown = sorted(set(params) - set(keys))
    if unknown and checked:
        raise ValueError(f"{cls_name} has no parameter(s) {unknown}; valid keys: {sorted(keys)}")
    out = {}
    for key, value in params.items():
        default = keys.get(key)
        out[key] = coerce_enum(type(default), value) if isinstance(default, Enum) and value is not None else value
    return out


def build_model(config: ModelConfig, protocol: ProtocolConfig, hpo: HpoConfig, seed: int, overrides: dict[str, Any] | None = None) -> BaseModel:
    """The automl model of ``config`` with the protocol applied.

    Protocol values are defaults a variant's own ``params`` override (a linear model sets ``early_stopping_rounds: null``).
    The runner always forces ``refit_on_train_val``, ``calculate_feature_importance``, ``random_seed``, the search switches
    and the search space. ``overrides`` are applied on top of ``params`` (matched width, tuned settings of a source variant).
    """
    if config.kind != PredictorKind.AUTOML or config.name is None:
        raise ValueError("build_model needs kind AUTOML and a model name")
    cls = model_class(config.name)
    keys, checked = accepted_keys(cls)
    params = _coerce_params(copy.deepcopy({**config.params, **(overrides or {})}), keys, checked, cls.__name__)  # a model never holds the config's own objects
    if protocol.early_stopping_patience is None or protocol.early_stopping_patience == MISSING:
        raise ValueError("protocol.early_stopping_patience is not set for this study")
    params.setdefault("early_stopping_rounds", protocol.early_stopping_patience)
    if "n_epochs" in keys:
        params.setdefault("n_epochs", protocol.max_epochs)
    if "batch_size" in keys:
        params.setdefault("batch_size", protocol.batch_size)
    params.update(
        refit_on_train_val=protocol.refit_on_train_val,
        calculate_feature_importance=protocol.calculate_feature_importance,
        optimize_hyperparameters=config.tuned,
        n_trials=hpo.n_trials,
        hpo_sampler=hpo.sampler,
        hpo_seed=hpo.seed,
    )
    if "random_seed" in keys:
        params["random_seed"] = seed
    if config.search_space:
        params["search_space_override"] = {name: space.as_space() for name, space in config.search_space.items()}
    if config.search_space_exclude:
        params["search_space_exclude"] = list(config.search_space_exclude)
    return cls(**params)


def matched_hidden_size(config: ModelConfig, protocol: ProtocolConfig, target_params: int, input_size: int) -> int:
    """Width of a one-hidden-layer plain network whose trainable parameter count is closest to ``target_params``.

    The count is linear in the width, so two probe networks (built by the model itself) give the line.
    """
    counts = []
    for width in (1, 2):
        probe = build_model(config, protocol, HpoConfig(n_trials=1), seed=0, overrides={"hidden_layers": 1, "hidden_size": width})
        probe.input_size = input_size
        probe.build_model()
        counts.append(probe.get_num_parameters())
    slope = counts[1] - counts[0]
    return max(1, round((target_params - (counts[0] - slope)) / slope))


@dataclass
class Standardiser:
    """Training-set mean and sd of ``x`` (per column) and ``y``; either side can be switched off."""

    standardise_x: bool
    standardise_y: bool
    x_mean: np.ndarray | float = 0.0
    x_sd: np.ndarray | float = 1.0
    y_scaler: TargetScaler = field(default_factory=TargetScaler)

    def fit(self, x: np.ndarray, y: np.ndarray) -> Standardiser:
        """Statistics of the training part (a zero sd becomes 1)."""
        if self.standardise_x:
            self.x_mean, self.x_sd = x.mean(axis=0), np.where(x.std(axis=0) > 0, x.std(axis=0), 1.0)
        if self.standardise_y:
            self.y_scaler.fit(y)
        return self

    def x(self, x: np.ndarray) -> np.ndarray:
        """Inputs in model space."""
        return ((x - self.x_mean) / self.x_sd).astype(np.float32)

    def y(self, y: np.ndarray) -> np.ndarray:
        """Targets in model space."""
        return self.y_scaler.transform(y).astype(np.float32) if self.standardise_y else y.astype(np.float32)

    def y_back(self, y: np.ndarray) -> np.ndarray:
        """Model-space values in original y units."""
        return self.y_scaler.inverse(np.asarray(y, dtype=np.float64)) if self.standardise_y else np.asarray(y, dtype=np.float64)


@dataclass
class Split:
    """The fitting data: training and validation parts."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray


class Predictor(ABC):
    """What a cell fits and scores."""

    @abstractmethod
    def fit(self, split: Split) -> None:
        """Fits on ``split.x_train``, early stopping / search on ``split.x_val``."""

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predictions in original y units, shape ``(n,)``."""

    def num_parameters(self) -> int | None:
        """Trainable parameter count, if the predictor has one."""
        return None

    def tuned_hyperparameters(self) -> dict[str, Any]:
        """Settings the search chose (empty if untuned)."""
        return {}

    def convergence(self) -> dict[str, Any] | None:
        """Validation trajectory, best epoch, stop epoch, converged and cap-hit flags, if the model keeps them."""
        return None

    def extras(self) -> dict[str, Any]:
        """Further record fields (selected k and per-k validation errors, matched width, ...)."""
        return {}

    def expected_noise_free_excess(self, true_mean_x: np.ndarray) -> float | None:  # noqa: ARG002 - the reference predictors use it
        """Known noise-free excess of a reference predictor on the points with true mean ``true_mean_x`` (``None`` for fitted models)."""
        return None


class ReferencePredictor(Predictor):
    """ORACLE (the true mean), PLANTED_OFFSET (plus ``offset_sd_fraction`` x sd of y_train) or PLANTED_SCALE (``scale_factor`` x true mean)."""

    def __init__(self, config: ModelConfig, toy: Toy) -> None:
        """Keeps the toy's true mean; the offset is learnt from y_train in ``fit``."""
        self.config, self.toy, self.offset = config, toy, 0.0

    def fit(self, split: Split) -> None:
        """Only the offset predictor looks at the data (the sd of ``y_train``)."""
        self.offset = self.config.offset_sd_fraction * float(np.std(split.y_train))

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Reference prediction."""
        mean = self.toy.true_mean(x)
        if self.config.kind == PredictorKind.PLANTED_OFFSET:
            return mean + self.offset
        if self.config.kind == PredictorKind.PLANTED_SCALE:
            return self.config.scale_factor * mean
        return mean

    def expected_noise_free_excess(self, true_mean_x: np.ndarray) -> float:
        """offset^2, or (1 - factor)^2 x mean(m(x)^2) on the evaluated points; 0 for the oracle."""
        if self.config.kind == PredictorKind.PLANTED_OFFSET:
            return self.offset**2
        if self.config.kind == PredictorKind.PLANTED_SCALE:
            return float((1.0 - self.config.scale_factor) ** 2 * np.mean(true_mean_x**2))
        return 0.0


class AutomlPredictor(Predictor):
    """An automl model behind a standardiser; with ``k_sweep`` it is fitted through ``fit_sweep_selector``."""

    def __init__(
        self, config: ModelConfig, protocol: ProtocolConfig, hpo: HpoConfig, seed: int, overrides: dict[str, Any] | None = None, extras: dict[str, Any] | None = None
    ) -> None:
        """Builds the model (see :func:`build_model`); ``extras`` are recorded as given."""
        self.config = config
        self.model = build_model(config, protocol, hpo, seed, overrides)
        self.scaler = Standardiser(config.standardise_x, config.standardise_y)
        self._extras = dict(extras or {})

    @property
    def inner(self) -> BaseModel:
        """The model that holds the fitted network (the chosen sub-model after a k sweep)."""
        return getattr(self.model, "_sweep_submodel_", self.model)

    def fit(self, split: Split) -> None:
        """Standardises with training statistics, then fits (one pass; search on the validation part when tuned)."""
        self.scaler.fit(split.x_train, split.y_train)
        x, y = self.scaler.x(split.x_train), self.scaler.y(split.y_train)
        x_val, y_val = self.scaler.x(split.x_val), self.scaler.y(split.y_val)
        sweep = self.config.k_sweep
        if sweep is None:
            self.model.fit(x, y, x_val=x_val, y_val=y_val)
            return
        self.model.fit_sweep_selector(x, y, x_val, y_val, capacity_grid=list(sweep.k_values), rule=sweep.rule, early_stop_on_val=True)
        self._extras.update(selected_k=int(self.model.selected_k_), k_val_errors={int(k): float(v) for k, v in self.model.sweep_val_errors_.items()})

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predictions in original y units."""
        return self.scaler.y_back(np.asarray(self.model.predict(self.scaler.x(x))).ravel())

    def num_parameters(self) -> int | None:
        """The model's trainable parameter count (``None`` when it cannot say)."""
        try:
            return int(self.inner.get_num_parameters())
        except (RuntimeError, NotImplementedError):
            return None

    def tuned_hyperparameters(self) -> dict[str, Any]:
        """Value of every searched dimension after the search (``a__b`` names a key of the dict parameter ``a``)."""
        if not self.config.tuned:
            return {}
        out = {}
        for name in self.model._searched_space():
            outer, _, inner = name.partition(NESTED_PARAM_SEPARATOR)
            out[name] = getattr(self.model, outer, {}).get(inner) if inner else getattr(self.model, name, None)
        return out

    def convergence(self) -> dict[str, Any] | None:
        """``convergence_`` of the fitted model as plain data."""
        result = getattr(self.inner, "convergence_", None)
        return None if result is None else dataclasses.asdict(result) | {"trustworthy": getattr(result, "trustworthy", None)}

    def extras(self) -> dict[str, Any]:
        """Selected k / matched width, whichever applies."""
        return dict(self._extras)


def build_predictor(
    config: ModelConfig, protocol: ProtocolConfig, hpo: HpoConfig, seed: int, toy: Toy, overrides: dict[str, Any] | None = None, extras: dict[str, Any] | None = None
) -> Predictor:
    """The predictor of ``config``."""
    if config.kind == PredictorKind.AUTOML:
        return AutomlPredictor(config, protocol, hpo, seed, overrides, extras)
    return ReferencePredictor(config, toy)
