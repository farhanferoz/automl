"""Runs ONE cell (variant x toy x size x seed) and writes its record; also where records live and how they are identified.

A record is ``<results_dir>/<variant>/<toy>[__<param>=<value>...]/n<n_train>/s<seed>.json`` (the card is metadata inside
it, not in the path). A record is identified by its resolved config (``CellSpec.config_hash``) and the environment
fingerprint of the machine that wrote it; the same config under another fingerprint goes to ``s<seed>__<fingerprint8>.json``
so a pair is never made across fingerprints.

Reused: ``run_provenance`` (git commit and dirty flag), the toy table through ``Toy``, ``registry`` for every predictor,
``curves`` for the probability curves and head scores.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import time
from enum import Enum, StrEnum
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from automl_package.experiments import curves
from automl_package.experiments.config import CellSpec, PredictorKind, ToyConfig, to_dict
from automl_package.experiments.registry import NESTED_PARAM_SEPARATOR, AutomlPredictor, Predictor, Split, build_predictor, matched_hidden_size
from automl_package.experiments.toys import Draw, DrawName, Toy
from automl_package.utils.checkpoint import ConfigConflictError, atomic_write
from automl_package.utils.run_provenance import run_provenance

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FINGERPRINT_PACKAGES = ("torch", "numpy", "optuna", "lightgbm", "scikit-learn")
FINGERPRINT_ID_LENGTH = 8


class CellStatus(StrEnum):
    """Whether a cell still has to run on this machine."""

    DONE = "done"
    PENDING = "pending"


@cache
def environment_fingerprint() -> dict[str, str]:
    """Machine architecture and the versions of the packages whose differences change a search or a fit."""
    return {"machine": platform.machine(), **{name: importlib.metadata.version(name) for name in FINGERPRINT_PACKAGES}}


def fingerprint_id(fingerprint: dict[str, str] | None = None) -> str:
    """Short hash of a fingerprint (this machine's by default)."""
    fingerprint = fingerprint or environment_fingerprint()
    return hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:FINGERPRINT_ID_LENGTH]


@cache
def source_hash() -> str:
    """SHA-256 over the sorted ``automl_package/**/*.py`` files (relative path and contents)."""
    digest = hashlib.sha256()
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if "__pycache__" not in path.parts:
            digest.update(str(path.relative_to(PACKAGE_ROOT)).encode() + path.read_bytes())
    return digest.hexdigest()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Enum):
        return obj.name
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Atomic write (a half-written record would look finished)."""
    atomic_write(path, lambda tmp: tmp.write_text(json.dumps(data, indent=1, default=_json_default)))


def load_record(path: Path) -> dict[str, Any]:
    """Reads one record."""
    return json.loads(path.read_text())


def base_path(results_dir: str, variant: str, toy: ToyConfig, n_train: int, seed: int) -> Path:
    """The un-suffixed record path of a cell."""
    return Path(results_dir) / variant / Toy.dir_name(toy) / f"n{n_train}" / f"s{seed}.json"


def record_path(cell: CellSpec, fp_id: str, variant: str | None = None) -> Path:
    """Where this cell's record for fingerprint ``fp_id`` lives (``variant`` names another variant's record of the same cell)."""
    primary = base_path(cell.results_dir, variant or cell.variant, cell.toy, cell.n_train, cell.seed)
    if primary.exists() and load_record(primary)["fingerprint_id"] != fp_id:
        return primary.with_name(f"{primary.stem}__{fp_id}.json")
    return primary


def all_record_paths(cell: CellSpec) -> list[Path]:
    """Every record of the cell, whatever its fingerprint (the primary file and the ``__<fingerprint8>`` ones)."""
    primary = base_path(cell.results_dir, cell.variant, cell.toy, cell.n_train, cell.seed)
    if not primary.parent.exists():
        return []
    return sorted(p for pattern in (primary.name, f"{primary.stem}__*.json") for p in primary.parent.glob(pattern))


def cell_status(cell: CellSpec, fp_id: str) -> CellStatus:
    """DONE when a record with the same resolved config and fingerprint exists (and the curves, if the cell makes them)."""
    path = record_path(cell, fp_id)
    if not path.exists():
        return CellStatus.PENDING
    record = load_record(path)
    if record["config_hash"] != cell.config_hash():
        raise ConfigConflictError(f"{path} was written for a different config of variant '{cell.variant}'; give the changed variant a new id or delete that record.")
    if cell.curves and not path.with_suffix(".npz").exists():
        return CellStatus.PENDING
    return CellStatus.DONE


def _set_metrics(toy: Toy, predictor: Predictor, draw: Draw) -> dict[str, Any]:
    """MSE, MAE, excess over the floor and noise-free excess in original y units (also per region and, for references, the expected value)."""
    pred = predictor.predict(draw.x)
    mean = toy.true_mean(draw.x)
    y = draw.y.astype(np.float64)
    sq_free = (pred - mean) ** 2

    def block(mask: np.ndarray | slice) -> dict[str, float]:
        mse = float(np.mean((pred[mask] - y[mask]) ** 2))
        return {"mse": mse, "mae": float(np.mean(np.abs(pred[mask] - y[mask]))), "noise_free_excess": float(np.mean(sq_free[mask]))}

    sq_err = (pred - y) ** 2
    out = block(slice(None)) | {
        "excess": float(np.mean(sq_err) - toy.floor),
        "excess_se": float(sq_err.std(ddof=1) / np.sqrt(len(sq_err))),  # label noise of the MSE on this draw
        "noise_free_excess_se": float(sq_free.std(ddof=1) / np.sqrt(len(sq_free))),
    }
    expected = predictor.expected_noise_free_excess(mean)
    if expected is not None:
        out["expected_noise_free_excess"] = expected
    out["regions"] = {name: block(mask) for name, mask in toy.region_masks(draw.x).items()}
    return out


def _overrides_from_tuned(base_params: dict[str, Any], tuned: dict[str, Any], drop: tuple[str, ...]) -> dict[str, Any]:
    """Another variant's tuned settings as constructor overrides (``a__b`` becomes key ``b`` of dict ``a``, starting from ``base_params[a]``)."""
    out: dict[str, Any] = {}
    for name, value in tuned.items():
        if name in drop:
            continue
        outer, _, inner = name.partition(NESTED_PARAM_SEPARATOR)
        if inner:
            out.setdefault(outer, dict(base_params.get(outer) or {}))[inner] = value
        else:
            out[name] = value
    return out


def _build(cell: CellSpec, toy: Toy, split: Split, fp_id: str) -> Predictor:
    """The cell's predictor, with the declared dependency (matched width or k sweep) resolved from the source variant's record."""
    overrides: dict[str, Any] = {}
    extras: dict[str, Any] = {}
    source = cell.depends_on
    if source is not None:
        path = record_path(cell, fp_id, variant=source)
        if not path.exists():
            raise FileNotFoundError(f"variant '{cell.variant}' needs the record of '{source}' for the same cell first: {path}")
        record = load_record(path)
        if cell.model.match_params_to:
            width = matched_hidden_size(cell.model, cell.protocol, int(record["n_parameters"]), split.x_train.shape[1])
            overrides = {"hidden_layers": 1, "hidden_size": width}
            extras["matched"] = {"to_variant": source, "target_parameters": record["n_parameters"], "hidden_size": width}
        else:
            overrides = _overrides_from_tuned(cell.model.params, record["tuned_hyperparameters"], drop=("n_classes",))
            extras["sweep_source"] = source
    return build_predictor(cell.model, cell.protocol, cell.hpo, cell.seed, toy, overrides, extras)


def run_cell(cell: CellSpec) -> Path:
    """Runs the cell (see the module docstring) and returns the path of its record."""
    started = time.perf_counter()
    os.environ["AUTOML_DEVICE"] = cell.protocol.device  # read by get_device() when the model is built
    torch.set_num_threads(cell.protocol.torch_threads)
    fp_id = fingerprint_id()
    toy = Toy(cell.toy)
    draws = toy.draws({DrawName.TRAIN: cell.n_train, DrawName.VALIDATION: cell.n_val, DrawName.SELECTION: cell.toy.n_selection, DrawName.TEST: cell.toy.n_test}, cell.seed)
    split = Split(draws[DrawName.TRAIN].x, draws[DrawName.TRAIN].y, draws[DrawName.VALIDATION].x, draws[DrawName.VALIDATION].y)
    predictor = _build(cell, toy, split, fp_id)
    predictor.fit(split)

    path = record_path(cell, fp_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {name: _set_metrics(toy, predictor, draws[name]) for name in (DrawName.SELECTION, DrawName.TEST)}
    record: dict[str, Any] = {
        "key": {"study": cell.study, "card": cell.card, "variant": cell.variant, "toy": Toy.dir_name(cell.toy), "n_train": cell.n_train, "seed": cell.seed},
        "config": to_dict(cell) | {"results_dir": None},
        "config_hash": cell.config_hash(),
        "fingerprint": environment_fingerprint(),
        "fingerprint_id": fp_id,
        "source_hash": source_hash(),
        "git": run_provenance()["git"],
        "data_hashes": {name.value: draw.sha256 for name, draw in draws.items()},
        "metrics": {name.value: value for name, value in metrics.items()},
        "tuned_hyperparameters": predictor.tuned_hyperparameters(),
        "n_parameters": predictor.num_parameters(),
        "convergence": predictor.convergence(),
        **predictor.extras(),
    }
    if cell.model.kind == PredictorKind.AUTOML and isinstance(predictor, AutomlPredictor):
        inner = curves.probreg_inner(predictor)
        if inner is not None:
            record["head_scores"] = curves.head_scores(curves.head_arrays(predictor, inner, draws[DrawName.SELECTION].x))
            if cell.curves:
                title = f"{cell.variant} | {Toy.dir_name(cell.toy)} | n={cell.n_train} | seed {cell.seed}"
                curves.write_curves(predictor, inner, toy, path.with_suffix(""), title)
    record["wall_seconds"] = time.perf_counter() - started
    _write_json(path, record)
    return path
