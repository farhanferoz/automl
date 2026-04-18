"""Synthetic benchmark datasets B1-B6.

Original designs from docs/research_plan.md §2.2. Each generator returns
``(x, y, meta)`` with ``meta`` carrying auxiliary quantities useful for
diagnostics (e.g., subpopulation labels, true noise levels, latent parameters).

All generators accept a ``seed`` to allow fixed-seed regeneration. Running
``python -m automl_package.utils.synthetic_datasets`` writes parquet snapshots
under ``tests/fixtures/`` for bit-exact reproduction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


G_CONST = 1.0  # gravitational constant (dimensionless for this benchmark)


@dataclass
class SyntheticDataset:
    name: str
    x: np.ndarray
    y: np.ndarray
    meta: dict

    def to_npz(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {"x": self.x, "y": self.y}
        for k, v in self.meta.items():
            if isinstance(v, np.ndarray):
                arrays[f"meta_{k}"] = v
            elif isinstance(v, (int, float)):
                arrays[f"meta_{k}"] = np.array(v)
        path = directory / f"{self.name}.npz"
        np.savez_compressed(path, **arrays)
        return path


# ---------------------------------------------------------------------------
# B1: Gravitational inverse problem (d=10, n=10k)
# ---------------------------------------------------------------------------

def generate_b1_gravitational(n: int = 10_000, seed: int = 1) -> SyntheticDataset:
    """g_i = G * m / (x_i - x0)^2 + eps_i, eps_i ~ N(0, (0.02*|g_i|)^2).

    Features: 10 noisy gravitational readings at fixed track positions.
    Target: x0 (position of the point mass).
    """
    rng = np.random.default_rng(seed)
    x_positions = np.linspace(-5.0, 5.0, 10)
    x0 = rng.uniform(-2.0, 2.0, n)
    m = rng.lognormal(mean=0.0, sigma=0.5, size=n)

    g_clean = G_CONST * m[:, None] / np.maximum((x_positions[None, :] - x0[:, None]) ** 2, 1e-6)
    noise_std = 0.02 * np.abs(g_clean)
    eps = rng.normal(0.0, noise_std)
    g = g_clean + eps
    return SyntheticDataset(
        name="b1_gravitational",
        x=g.astype(np.float32),
        y=x0.astype(np.float32),
        meta={"m": m.astype(np.float32), "noise_std_feature": noise_std.astype(np.float32), "track_positions": x_positions.astype(np.float32)},
    )


# ---------------------------------------------------------------------------
# B2: Oscillator with phase ambiguity (d=8, n=10k)
# ---------------------------------------------------------------------------

def generate_b2_oscillator(n: int = 10_000, seed: int = 2) -> SyntheticDataset:
    """Damped harmonic oscillator: y(t_i) = A*exp(-gamma*t_i)*cos(2*pi*f*t_i + phi) + eps.

    Features: 8 oscillator readings at fixed times.
    Target: frequency f. Aliasing: f and f_aliased = N/(dt) - f are indistinguishable.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 5.0, 8)
    f = rng.uniform(0.1, 0.9, n)
    a = rng.uniform(0.5, 1.5, n)
    gamma = rng.uniform(0.05, 0.3, n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    y_series = a[:, None] * np.exp(-gamma[:, None] * t[None, :]) * np.cos(2.0 * np.pi * f[:, None] * t[None, :] + phi[:, None])
    eps = rng.normal(0.0, 0.02, y_series.shape)
    y_obs = y_series + eps
    return SyntheticDataset(
        name="b2_oscillator",
        x=y_obs.astype(np.float32),
        y=f.astype(np.float32),
        meta={"amplitude": a.astype(np.float32), "gamma": gamma.astype(np.float32), "phase": phi.astype(np.float32), "t": t.astype(np.float32)},
    )


# ---------------------------------------------------------------------------
# B3: Two-phase transition (d=12, n=10k)
# ---------------------------------------------------------------------------

def generate_b3_two_phase(n: int = 10_000, seed: int = 3) -> SyntheticDataset:
    """y = f_1(x) if x.w < tau else f_2(x); noise high near threshold."""
    rng = np.random.default_rng(seed)
    d = 12
    x = rng.uniform(-1.0, 1.0, (n, d)).astype(np.float32)
    w = rng.normal(0.0, 1.0, d)
    w /= np.linalg.norm(w)
    tau = 0.0

    z = x @ w
    f1 = 2.0 * z + 0.5 * x[:, 0]
    f2 = np.sin(3.0 * z) + 0.8 * x[:, 1] ** 2 - 0.4 * x[:, 2]
    y_clean = np.where(z < tau, f1, f2)

    # Noise peaks at |z-tau| ~ 0, decays with distance.
    dist = np.abs(z - tau)
    sigma = 0.05 + 0.5 * np.exp(-(dist * 4.0) ** 2)
    noise = rng.normal(0.0, sigma)
    y = (y_clean + noise).astype(np.float32)
    return SyntheticDataset(
        name="b3_two_phase",
        x=x,
        y=y,
        meta={"projection": z.astype(np.float32), "regime": (z >= tau).astype(np.int8), "noise_std": sigma.astype(np.float32), "w": w.astype(np.float32)},
    )


# ---------------------------------------------------------------------------
# B4: Conditional heteroscedasticity with latent subpopulations (d=10, n=10k)
# ---------------------------------------------------------------------------

def generate_b4_latent_groups(n: int = 10_000, seed: int = 4) -> SyntheticDataset:
    """Three subpopulations with distinct f(x), sigma(x).

    Group assignment: nonlinear boundary in feature space.
    """
    rng = np.random.default_rng(seed)
    d = 10
    x = rng.normal(0.0, 1.0, (n, d)).astype(np.float32)
    # Nonlinear group assignment via two basis features.
    u = np.sin(x[:, 0] * 2.0) + 0.5 * x[:, 1] ** 2
    group = np.digitize(u, bins=[-0.5, 0.8])  # 0 / 1 / 2

    y = np.zeros(n, dtype=np.float32)
    noise_std = np.zeros(n, dtype=np.float32)

    # Group 0: low noise, linear
    mask = group == 0
    y[mask] = 1.5 * x[mask, 0] + 0.3 * x[mask, 2]
    noise_std[mask] = 0.05

    # Group 1: moderate noise, quadratic
    mask = group == 1
    y[mask] = x[mask, 3] ** 2 - 0.5 * x[mask, 4]
    noise_std[mask] = 0.2

    # Group 2: high noise, sinusoidal
    mask = group == 2
    y[mask] = np.sin(2.0 * x[mask, 5]) + 0.3 * x[mask, 6]
    noise_std[mask] = 0.6

    y = y + rng.normal(0.0, noise_std)
    return SyntheticDataset(
        name="b4_latent_groups",
        x=x,
        y=y.astype(np.float32),
        meta={"group": group.astype(np.int8), "noise_std": noise_std.astype(np.float32)},
    )


# ---------------------------------------------------------------------------
# B5: Exponentially-distributed feature importance (d=30, n=10k)
# ---------------------------------------------------------------------------

def generate_b5_sparse_importance(n: int = 10_000, seed: int = 5) -> SyntheticDataset:
    """3 informative features, 27 noise. Importance decays exponentially."""
    rng = np.random.default_rng(seed)
    d_total = 30
    x = rng.normal(0.0, 1.0, (n, d_total)).astype(np.float32)
    weights = np.zeros(d_total, dtype=np.float32)
    weights[:3] = np.array([1.0, 0.5, 0.25], dtype=np.float32)
    y = x @ weights + 0.1 * np.sin(3.0 * x[:, 0]) + rng.normal(0.0, 0.2, n)
    return SyntheticDataset(
        name="b5_sparse_importance",
        x=x,
        y=y.astype(np.float32),
        meta={"weights": weights},
    )


# ---------------------------------------------------------------------------
# B6: Null homoscedastic unimodal regression (d=8, n=5k)
# ---------------------------------------------------------------------------

def generate_b6_null(n: int = 5_000, seed: int = 6) -> SyntheticDataset:
    """Smooth nonlinear f(x) + N(0, sigma^2) with constant sigma."""
    rng = np.random.default_rng(seed)
    d = 8
    x = rng.uniform(-1.0, 1.0, (n, d)).astype(np.float32)
    y = (
        0.5 * x[:, 0]
        + 0.4 * np.tanh(2.0 * x[:, 1])
        + 0.3 * x[:, 2] * x[:, 3]
        + 0.2 * np.sin(np.pi * x[:, 4])
    )
    y = y + rng.normal(0.0, 0.1, n)
    return SyntheticDataset(
        name="b6_null",
        x=x,
        y=y.astype(np.float32),
        meta={"noise_std": 0.1},
    )


ALL_GENERATORS: dict[str, Callable[[], SyntheticDataset]] = {
    "b1": generate_b1_gravitational,
    "b2": generate_b2_oscillator,
    "b3": generate_b3_two_phase,
    "b4": generate_b4_latent_groups,
    "b5": generate_b5_sparse_importance,
    "b6": generate_b6_null,
}


def load_all() -> list[SyntheticDataset]:
    return [gen() for gen in ALL_GENERATORS.values()]


def dump_fixtures(directory: Path | None = None) -> None:
    if directory is None:
        directory = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "synthetic"
    for ds in load_all():
        path = ds.to_npz(directory)
        print(f"Wrote {path}  n={len(ds.y)} d={ds.x.shape[1]}")


def load_fixture(name: str, directory: Path | None = None) -> SyntheticDataset:
    if directory is None:
        directory = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "synthetic"
    with np.load(directory / f"{name}.npz") as arr:
        x = arr["x"]
        y = arr["y"]
        meta = {k.removeprefix("meta_"): arr[k] for k in arr.files if k.startswith("meta_")}
    return SyntheticDataset(name=name, x=x, y=y, meta=meta)


if __name__ == "__main__":
    dump_fixtures()
