"""Pausable runs: a store of finished units of work, and the stable hash that says which config a unit belongs to.

A runner splits its work into units (a cell, an outer fold, a horizon x fold), saves each unit's frames and metadata the
moment it finishes, and on restart skips the units that are already saved. Inside a unit, a hyperparameter search resumes
from its own Optuna journal (``OptunaOptimizer(storage_path=...)``); this module owns the finished-unit side.

Searched before writing: ``automl_package/experiments/cell.py`` (ProbReg record: JSON plus ``config_hash``, skip when the
hash matches, ``ConfigConflictError`` otherwise) and a grep of ``automl_package`` and ``utils`` for ``to_parquet`` /
``os.replace`` / ``checkpoint``. No generic store exists: cell.py's logic is tied to the ProbReg record layout, which a
running study depends on, so it keeps its layout and shares only the hash function and the conflict error defined here.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

META_SUFFIX = ".json"
FRAME_SUFFIX = ".parquet"
TEMP_SUFFIX = ".tmp"


class ConfigConflictError(RuntimeError):
    """A saved record or unit exists under this key with a different resolved config."""


def _json_default(obj: Any) -> Any:
    """Arrays and pandas objects hash by content; anything else by its ``str`` (what the ProbReg config hash has always done).

    A DataFrame's ``str`` shows only its first and last rows, so two different frames could share it: frames and series are
    hashed row by row (values and index) together with their column names and dtypes.
    """
    if isinstance(obj, np.ndarray):
        return hashlib.sha256(np.ascontiguousarray(obj).tobytes()).hexdigest()
    if isinstance(obj, pd.DataFrame | pd.Series):
        layout = str(list(obj.columns) + list(obj.dtypes.astype(str))) if isinstance(obj, pd.DataFrame) else f"{obj.name}{obj.dtype}"
        return hashlib.sha256(pd.util.hash_pandas_object(obj, index=True).to_numpy().tobytes() + layout.encode()).hexdigest()
    return str(obj)


def stable_hash(data: Any) -> str:
    """SHA-256 (hex) of ``data`` serialised as sorted-key JSON (arrays by content, other non-JSON values by ``str``)."""
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=_json_default).encode()).hexdigest()


def atomic_write(path: Path, write: Any) -> None:
    """Writes ``path`` through a temp file in the same directory and ``os.replace`` (a half-written file would look finished).

    ``write`` is a callable taking the temp path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + TEMP_SUFFIX)
    write(tmp)
    os.replace(tmp, path)


class UnitStore:
    """Finished units under ``root``: ``<key>.json`` (config hash and metadata, written last) and ``<key>.<frame>.parquet``."""

    def __init__(self, root: Path) -> None:
        """Args: root: Directory that holds the units (created on the first save)."""
        self.root = Path(root)

    def _meta_path(self, key: str) -> Path:
        return self.root / f"{key}{META_SUFFIX}"

    def _frame_path(self, key: str, name: str) -> Path:
        return self.root / f"{key}.{name}{FRAME_SUFFIX}"

    def _read(self, key: str) -> dict[str, Any]:
        return json.loads(self._meta_path(key).read_text())

    def is_done(self, key: str, config_hash: str, expect: dict[str, Any] | None = None) -> bool:
        """True when ``key`` was saved for ``config_hash`` and its metadata holds every ``expect`` entry.

        A unit saved for another ``config_hash`` is not skipped silently.

        Raises:
            ConfigConflictError: If the unit exists under another config hash.
        """
        if not self._meta_path(key).exists():
            return False
        record = self._read(key)
        if record["config_hash"] != config_hash:
            raise ConfigConflictError(f"{self._meta_path(key)} was written for a different config; change the run's checkpoint directory or delete that unit.")
        return all(record["meta"].get(name) == value for name, value in (expect or {}).items())

    def save(self, key: str, config_hash: str, frames: dict[str, pd.DataFrame], meta: dict[str, Any]) -> None:
        """Writes the frames, then the metadata file: the unit counts as done only once the metadata is in place."""
        for name, frame in frames.items():
            atomic_write(self._frame_path(key, name), lambda tmp, frame=frame: frame.to_parquet(tmp, index=False, compression="zstd"))
        record = {"config_hash": config_hash, "frames": list(frames), "meta": meta}
        atomic_write(self._meta_path(key), lambda tmp: tmp.write_text(json.dumps(record, indent=1)))

    def load(self, key: str) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
        """The frames and metadata saved for ``key``."""
        record = self._read(key)
        return {name: pd.read_parquet(self._frame_path(key, name)) for name in record["frames"]}, record["meta"]
