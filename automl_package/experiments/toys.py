"""The study toys as the runner uses them: four hash-checked draws per cell, the true mean, noise sd and regions.

Reused: ``STUDY_TOYS`` / ``ToySpec`` (``examples/_toy_datasets.py``) for generators, true means, noise sd, floors, x
ranges and regions; toy parameters (e.g. the jump size) are passed through as keywords.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from automl_package.examples._toy_datasets import STUDY_TOYS, ToySpec
from automl_package.experiments.config import ToyConfig


class DrawName(StrEnum):
    """The four disjoint draws of a cell, in the order their seeds are spawned."""

    TRAIN = "train"
    VALIDATION = "validation"
    SELECTION = "selection"
    TEST = "test"


@dataclass(frozen=True)
class Draw:
    """One sample ``(x, y)`` (``x`` is ``(n, 1)``) and the SHA-256 of its bytes."""

    x: np.ndarray
    y: np.ndarray
    sha256: str


def _hash(x: np.ndarray, y: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes() + np.ascontiguousarray(y).tobytes()).hexdigest()


class Toy:
    """A configured study toy: draws, true mean, noise sd, floor, regions."""

    def __init__(self, config: ToyConfig) -> None:
        """Looks the toy up in ``STUDY_TOYS`` (unknown name: ``KeyError`` listing the table)."""
        if config.name not in STUDY_TOYS:
            raise KeyError(f"{config.name} is not in STUDY_TOYS: {sorted(STUDY_TOYS)}")
        self.config = config
        self.spec: ToySpec = STUDY_TOYS[config.name]
        self.params = dict(config.params)

    @property
    def floor(self) -> float:
        """Bayes squared-error floor in y units."""
        return float(self.spec.floor)

    def true_mean(self, x: np.ndarray) -> np.ndarray:
        """``E[y | x]`` as float64, shape ``(n,)``."""
        return np.asarray(self.spec.true_mean(x, **self.params), dtype=np.float64).ravel()

    def noise_sd(self, x: np.ndarray) -> np.ndarray:
        """Noise sd at each ``x``."""
        return np.asarray(self.spec.noise_sd(x, **self.params), dtype=np.float64).ravel()

    @property
    def x_range(self) -> tuple[float, float]:
        """Range of the toy's inputs."""
        return float(self.spec.x_range[0]), float(self.spec.x_range[1])

    def region_masks(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """Boolean mask over ``x`` for each named region (empty if the toy has none)."""
        return {name: np.asarray(fn(x.ravel()), dtype=bool) for name, fn in self.spec.regions.items()}

    def _draw_one(self, n: int, seed: int) -> Draw:
        x, y = self.spec.generator(n, seed, **self.params)
        x = np.asarray(x).reshape(len(x), -1)
        y = np.asarray(y).ravel()
        return Draw(x=x, y=y, sha256=_hash(x, y))

    def draws(self, sizes: dict[DrawName, int], seed: int) -> dict[DrawName, Draw]:
        """The cell's four draws, from disjoint integer seeds spawned off ``SeedSequence(seed)`` (identical on every machine)."""
        children = np.random.SeedSequence(seed).spawn(len(DrawName))
        seeds = {name: int(child.generate_state(1)[0]) for name, child in zip(DrawName, children, strict=True)}
        return {name: self._draw_one(sizes[name], seeds[name]) for name in DrawName}

    @staticmethod
    def dir_name(config: ToyConfig) -> str:
        """Folder name of a toy version: ``name`` plus ``__param=value`` for each parameter, sorted."""
        return str(config.name) + "".join(f"__{k}={config.params[k]}" for k in sorted(config.params))
