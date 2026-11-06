"""Typed configs of the experiment runner: plain dataclasses and enums, no Hydra import.

Every field has a default; a study is a YAML tree under ``conf/`` that fills them (``run.py`` registers these classes
with Hydra's ConfigStore so each YAML is type-checked when composed). ``execute.py`` needs no Hydra: it rebuilds the
same dataclasses from the JSON lines ``run.py`` writes, through :func:`from_dict`.

Reused: ``ModelName`` (``enums.py``), ``StudyToy`` (``examples/_toy_datasets.py``), the structured-config layout and
``SearchParamConfig`` of ``/home/ff235/dev/ae/automl_adapter/config.py`` (moved here; ae switches over later).
"""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any, TypeVar

from automl_package.enums import HpoSampler, KSelectionRule, ModelName
from automl_package.examples._toy_datasets import StudyToy
from automl_package.utils.checkpoint import stable_hash

MISSING = "???"  # Hydra's marker for a value the study must set before it can run
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = str(REPO_ROOT / "results" / "probreg")  # the ProbReg toy study; another study family sets output.results_root


class PredictorKind(StrEnum):
    """What a variant is: a model fitted to data, or a reference predictor built from the true mean."""

    AUTOML = "automl"
    ORACLE = "oracle"  # the true mean
    PLANTED_OFFSET = "planted_offset"  # the true mean plus a constant offset
    PLANTED_SCALE = "planted_scale"  # the true mean times a constant factor


class ParamType(StrEnum):
    """Kind of one hyperparameter's range, as ``get_hyperparameter_search_space`` spells it (lower case)."""

    INT = "int"
    FLOAT = "float"
    CATEGORICAL = "categorical"


class EvalSet(StrEnum):
    """Which held-out set a report table reads: the selection draw (cards 5c) or the test draw (cards 5b, 5d)."""

    SELECTION = "selection"
    TEST = "test"


class ReportKind(StrEnum):
    """Which report a study gets."""

    COMPARISON = "comparison"  # the one table against the reference variant
    SIZE_SWEEP = "size_sweep"  # card 5a: reference predictors, trajectories, excess against n


@dataclass
class SearchParamConfig:
    """One hyperparameter's range."""

    type: ParamType = ParamType.FLOAT
    low: float | None = None
    high: float | None = None
    step: float | None = None
    log: bool = False
    choices: list[Any] | None = None

    def as_space(self) -> dict[str, Any]:
        """The dict form ``get_hyperparameter_search_space`` / ``search_space_override`` use (only the fields that are set)."""
        space: dict[str, Any] = {"type": str(self.type)}
        cast = int if self.type == ParamType.INT else float
        for name in ("low", "high", "step"):
            if getattr(self, name) is not None:
                space[name] = cast(getattr(self, name))
        if self.choices is not None:
            space["choices"] = list(self.choices)
        if self.log:
            space["log"] = True
        return space


@dataclass
class KSweepConfig:
    """Card 5c-5: refit one model per k from the named variant's tuned settings and choose k on the validation split."""

    from_variant: str = MISSING
    k_values: list[int] = field(default_factory=lambda: list(range(2, 11)))
    rule: KSelectionRule = MISSING


@dataclass
class ModelConfig:
    """One variant's predictor."""

    kind: PredictorKind = PredictorKind.AUTOML
    name: ModelName | None = None
    params: dict[str, Any] = field(default_factory=dict)  # keys of the class's ``_defaults`` / BaseModel arguments; strings become enums
    tuned: bool = False
    search_space: dict[str, SearchParamConfig] = field(default_factory=dict)  # replaces or adds a dimension
    search_space_exclude: list[str] = field(default_factory=list)  # removes dimensions
    standardise_x: bool = False  # training-set statistics; predictions mapped back, every metric in original y units
    standardise_y: bool = False
    match_params_to: str | None = None  # plain network sized to this variant's trainable parameter count, same cell
    k_sweep: KSweepConfig | None = None
    offset_sd_fraction: float = 0.1  # PLANTED_OFFSET: offset = this * sd(y_train)
    scale_factor: float = 0.9  # PLANTED_SCALE: prediction = this * true mean


@dataclass
class ToyConfig:
    """One toy version and its set sizes (the training size comes from the study's ``sizes``)."""

    name: StudyToy = StudyToy.TOY_A
    params: dict[str, Any] = field(default_factory=dict)  # toy parameters, e.g. {"jump": 1.0}
    n_val: int | None = None  # None = n_train // 4
    n_selection: int = 10_000
    n_test: int = 10_000


@dataclass
class ProtocolConfig:
    """Training protocol shared by the study's variants (a variant's own ``params`` win)."""

    seeds: list[int] = field(default_factory=lambda: [0])
    early_stopping_patience: int | None = MISSING  # type: ignore[assignment]
    max_epochs: int = 5000
    batch_size: int = 32
    refit_on_train_val: bool = False
    calculate_feature_importance: bool = False
    torch_threads: int = 1
    device: str = "cpu"


@dataclass
class HpoConfig:
    """Hyperparameter search: trial budget, Optuna sampler and its seed (``None`` = the cell's seed)."""

    n_trials: int = 20
    sampler: HpoSampler = HpoSampler.TPE
    seed: int | None = None


@dataclass
class OutputConfig:
    """Where records go (small JSON and NPZ; tracked in git per docs/RULES.md)."""

    results_root: str = DEFAULT_RESULTS_ROOT
    results_subdir: str | None = None  # record group under runs/; None = the card; cards that share reference records name one group
    curves: bool = False
    curve_seeds: list[int] | None = None  # None = the first seed


@dataclass
class ReportConfig:
    """How a study's records are read."""

    kind: ReportKind = ReportKind.COMPARISON
    eval_set: EvalSet = EvalSet.TEST
    reference_variant: str | None = None
    noise_results_subdir: str | None = None  # record group (under runs/) of the 5a size-sweep records whose seed spread shades the table
    noise_variant: str | None = None  # the reference model's variant id in those records


@dataclass
class StudyConfig:
    """A study: variants x toys x sizes x seeds."""

    name: str = MISSING
    card: str = MISSING
    variants: dict[str, ModelConfig] = field(default_factory=dict)
    toys: dict[str, ToyConfig] = field(default_factory=dict)  # label -> toy
    toy_filter: list[str] = field(default_factory=list)  # labels to keep; empty = all
    sizes: list[int] = field(default_factory=list)
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    hpo: HpoConfig = field(default_factory=HpoConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    n_workers: int = 1


@dataclass
class RootConfig:
    """Hydra's root: the composed study."""

    study: StudyConfig = field(default_factory=StudyConfig)
    target_fingerprint: str | None = None  # environment fingerprint id the pending count is for; None = this machine


@dataclass
class CellSpec:
    """One cell (variant x toy x size x seed) with everything needed to run it. One JSON line of ``cells_<card>.jsonl``."""

    study: str = MISSING
    card: str = MISSING
    variant: str = MISSING
    model: ModelConfig = field(default_factory=ModelConfig)
    toy_label: str = MISSING
    toy: ToyConfig = field(default_factory=ToyConfig)
    n_train: int = 0
    seed: int = 0
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    hpo: HpoConfig = field(default_factory=HpoConfig)
    curves: bool = False
    results_dir: str = MISSING
    report: ReportConfig = field(default_factory=ReportConfig)

    # Fields that do not change what the cell computes: the same variant in two cards or studies shares one record.
    _NOT_IDENTITY: typing.ClassVar[tuple[str, ...]] = ("study", "card", "results_dir", "report", "curves", "toy_label")

    @property
    def n_val(self) -> int:
        """Validation draw size."""
        return self.toy.n_val if self.toy.n_val is not None else self.n_train // 4

    @property
    def depends_on(self) -> str | None:
        """The variant whose record for this same cell must exist first (matched size or k sweep)."""
        if self.model.match_params_to:
            return self.model.match_params_to
        return self.model.k_sweep.from_variant if self.model.k_sweep else None

    def config_hash(self) -> str:
        """SHA-256 (hex) of the resolved config minus the fields in ``_NOT_IDENTITY``."""
        identity = {k: v for k, v in to_dict(self).items() if k not in self._NOT_IDENTITY}
        return stable_hash(identity)


T = TypeVar("T")


def coerce_enum(enum_type: type[Enum], value: Any) -> Enum:
    """``value`` as a member of ``enum_type``: member name first, then value; failure lists the members."""
    if isinstance(value, enum_type):
        return value
    text = str(value)
    if text in enum_type.__members__:
        return enum_type[text]
    if text.upper() in enum_type.__members__:
        return enum_type[text.upper()]
    for member in enum_type:
        if member.value == value or str(member.value) == text:
            return member
    raise ValueError(f"{value!r} is not a member of {enum_type.__name__}; members: {[m.name for m in enum_type]}")


def to_dict(obj: Any) -> Any:
    """A dataclass tree as plain JSON-ready data (enums become their member NAME so :func:`from_dict` round-trips)."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Enum):
        return obj.name
    if isinstance(obj, dict):
        return {str(k): to_dict(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [to_dict(v) for v in obj]
    return obj


def _convert(tp: Any, value: Any) -> Any:
    """``value`` converted to the annotation ``tp`` (dataclasses, enums, dict/list/Optional of those)."""
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        members = [a for a in typing.get_args(tp) if a is not type(None)]
        return None if value is None else _convert(members[0], value)
    if origin is dict:
        return {k: _convert(typing.get_args(tp)[1], v) for k, v in value.items()}
    if origin is list:
        return [_convert(typing.get_args(tp)[0], v) for v in value]
    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        return from_dict(tp, value)
    if isinstance(tp, type) and issubclass(tp, Enum):
        return coerce_enum(tp, value)
    return value


def from_dict(cls: type[T], data: dict[str, Any]) -> T:
    """Rebuild dataclass ``cls`` (recursively) from :func:`to_dict` output; unknown keys are an error."""
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"{cls.__name__} has no field(s) {sorted(unknown)}; fields: {sorted(names)}")
    return cls(**{k: _convert(hints[k], v) for k, v in data.items()})
