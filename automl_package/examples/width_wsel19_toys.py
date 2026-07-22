"""WSEL-19 multi-feature toy construction — the canonical toy lifted into `d` dimensions.

`docs/plans/capacity_programme/shared/wsel19-toy-design.md` (amended, GO-WITH-AMENDMENTS,
2026-07-22) §2/§2b/§3b/§5.1/§6 is the build authority for this module; read it before changing
anything here. This module is a pure toy-construction library (generators + provisioning helpers)
-- it trains nothing and runs no grid; `automl_package/examples/width_wsel19.py` is the driver that
consumes it.

**The construction is INVERTED (F2).** A forward draft (`x ~ N`, project, feed the target function)
cannot reuse the canonical generator: `nested_width_net.make_hetero` draws its own scalar input
internally (`rng.uniform(-r, r, n)`) with no injection point, and uniform-vs-normal RNG consumption
can never bit-match. Instead:

  1. `(t, y, region) = make_hetero(n, seed)` -- the canonical call, VERBATIM (imported below, never
     re-implemented -- genuine §3.9 reuse).
  2. `u = Φ⁻¹((t + r) / (2r))`, the probability clipped away from `{0, 1}` at float eps -- the
     probability-integral-transform lift, so `u ~ N(0, 1)` EXACTLY.
  3. `x = u·v + (I − vvᵀ)z`, `z ~ N(0, I_d)` drawn from a DECOUPLED RNG stream (never
     `make_hetero`'s own generator instance), projected onto `v`'s orthogonal complement -- so
     `x ~ N(0, I_d)` exactly and `v·x = u` exactly (`v` a unit vector by construction, both
     geometries below).

Every `d` and every unit `v` therefore carries the SAME `(t, y, region)` bit-for-bit -- the identity
check (§5.1, `identity_holds` below) holds BY CONSTRUCTION, and confound C1 (geometry vs input
distribution) is exact, not approximate.

**Geometry (§3 grid axis).** `Geometry.AXIS` (`v = e_1`) or `Geometry.OBLIQUE` (`v = s / sqrt(d)`,
`s` a per-seed DETERMINISTIC ±1 sign pattern -- maximally oblique, no near-axis draws, no per-seed
direction variance, per the adjudicator's amendment). Sparse-oblique is EXCLUDED (recorded
decision, unchanged). `v` is a pure function of `(d, geometry, seed)` -- the CELL's base seed, not
whichever split reads it -- so every split at a cell shares the identical `v`.

**Provisioning (§3b, pre-registered).** `make_train_split` = the canonical size (`TRAIN_N`, 1500) at
the cell's own base seed; `make_selection_split` = an INDEPENDENT draw of exactly `n_sel` points at
`seed + SELECTION_SEED_OFFSET`; `make_report_split` = an INDEPENDENT draw of `REPORT_N` (2000)
points at `seed + REPORT_SEED_OFFSET`. Selection is never carved from the training set (C8).

Usage:
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel19_toys.py --selftest
"""

from __future__ import annotations

import argparse
import enum
import os
import sys

import numpy as np
from scipy.stats import norm

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root, so `import automl_package` works

import converged_width_experiment as cwe  # noqa: E402
import nested_width_net as nwn  # noqa: E402

# --- §3 grid axes and §3b provisioning constants, closed sets, no magic scatter (§3.10) --------

TRAIN_N = cwe.N_TRAIN  # 1500, the canonical training size (§3b), reused not re-derived.
REPORT_N = 2000  # §3b: the report/held-out set size, pre-registered.

D_GRID = (2, 8, 32)  # §3 grid axis: multi-feature input dimension.
N_SEL_GRID = (75, 300, 1200)  # §3 grid axis: selection-set size.

SELECTION_SEED_OFFSET = 1000  # §3b: selection set = an independent §2 draw at seed + this offset.
REPORT_SEED_OFFSET = 2000  # §3b: report set = an independent §2 draw at seed + this offset.

# Both offsets below decouple an AUXILIARY draw (the orthogonal-complement noise `z`, or the
# OBLIQUE sign pattern) from a §2-construction call's own `(t, y, region)` stream -- the same
# "seed + fixed offset" idiom §3b already uses for SELECTION_SEED_OFFSET/REPORT_SEED_OFFSET, and
# already on disk elsewhere (`converged_width_experiment.py`'s held-out `seed + 500` test draw).
# Chosen far enough from every combination of a cell seed x {0, SELECTION_SEED_OFFSET,
# REPORT_SEED_OFFSET} in this strand's grid that no two draws can ever collide.
_Z_SEED_OFFSET = 5_000  # decouples the orthogonal-complement noise `z` from make_hetero's own RNG stream.
_SIGN_PATTERN_SEED_OFFSET = 9_000  # decouples OBLIQUE's sign-pattern draw; keyed on the CELL seed only (shared by every split).

_PROBIT_EPS = float(np.finfo(np.float64).eps)  # §2 step 2: clip the probability away from {0, 1} before Φ⁻¹.

_SELFTEST_D = (1, *D_GRID)  # d=1 additionally exercises the reduction the (separately-owned) 1-D calibration cell relies on.
_SELFTEST_N = 5_000
_SELFTEST_SEED = 0
_MARGINAL_TOL = 0.1  # loose: a 5000-sample N(0,1) mean/std has SE ~0.014, so 0.1 is ~7 SE of margin.
_DOT_TOL = 1e-5  # v.x == u to float32 precision (x is stored float32; see make_hetero_multifeature).


class Geometry(enum.Enum):
    """Closed set of `v` geometries (`width.md` / toy-design §3). Sparse-oblique is EXCLUDED (recorded decision)."""

    AXIS = "axis"  # v = e_1
    OBLIQUE = "oblique"  # v = s / sqrt(d), s a per-seed deterministic +-1 sign pattern


def _uniform_to_standard_normal(t: np.ndarray, r: float) -> np.ndarray:
    """`Φ⁻¹((t + r) / (2r))` -- the probability-integral-transform lift (§2 step 2).

    Args:
        t: `make_hetero`'s own `Uniform(-r, r)` draw.
        r: the domain half-width `make_hetero` was called with.

    Returns:
        `u`, float64, exactly `N(0, 1)`-distributed. The probability is clipped away from `{0, 1}`
        at float eps (`_PROBIT_EPS`) before `norm.ppf` -- `t` landing exactly on `-r`/`r` would
        otherwise map to `-inf`/`+inf`.
    """
    p = (t.astype(np.float64) + r) / (2.0 * r)
    p = np.clip(p, _PROBIT_EPS, 1.0 - _PROBIT_EPS)
    return norm.ppf(p)


def oblique_sign_pattern(d: int, seed: int) -> np.ndarray:
    """Deterministic ±1 Rademacher sign pattern, length `d`, keyed on `seed` ALONE (§3 OBLIQUE geometry).

    A CELL-level draw -- independent of which split (train/selection/report) reads it, so every
    split at a given `(d, geometry, seed)` cell shares the identical `v` (confound C1). Uses a
    decoupled RNG stream (`_SIGN_PATTERN_SEED_OFFSET`), never `make_hetero`'s own generator.

    Args:
        d: pattern length (the input dimension).
        seed: the cell's base seed.

    Returns:
        `(d,)` float64 array of exactly `+1.0`/`-1.0` entries.
    """
    rng = np.random.default_rng(int(seed) + _SIGN_PATTERN_SEED_OFFSET)
    return (rng.integers(0, 2, size=int(d)) * 2 - 1).astype(np.float64)


def geometry_vector(d: int, geometry: Geometry, seed: int) -> np.ndarray:
    """The unit vector `v` a `(d, geometry, seed)` cell routes its signal along (§3).

    `AXIS`: `v = e_1`. `OBLIQUE`: `v = s / sqrt(d)`, `s` from `oblique_sign_pattern` -- DETERMINISTIC
    and maximally oblique (no near-axis draws, no per-seed direction variance -- the adjudicator's
    amendment). Both are exact unit vectors (`||v|| = 1`), which is what makes `x`'s covariance come
    out to `I_d` exactly in `make_hetero_multifeature` below.

    Args:
        d: input dimension (`d >= 1`).
        geometry: `Geometry.AXIS` or `Geometry.OBLIQUE`.
        seed: the cell's base seed (never a split's own draw seed -- see module docstring).

    Returns:
        `(d,)` float64 unit vector.
    """
    if int(d) < 1:
        raise ValueError(f"d={d} must be >= 1")
    if geometry is Geometry.AXIS:
        v = np.zeros(int(d), dtype=np.float64)
        v[0] = 1.0
        return v
    if geometry is Geometry.OBLIQUE:
        s = oblique_sign_pattern(d, seed)
        return s / np.sqrt(float(d))
    raise ValueError(f"unknown geometry: {geometry!r}")


def make_hetero_multifeature(
    n: int,
    seed: int,
    d: int,
    geometry: Geometry,
    *,
    draw_seed: int | None = None,
    r: float = nwn.HETERO_R_DEFAULT,
    sigma: float = nwn.HETERO_NOISE_SIGMA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The `d`-dimensional lift of `nested_width_net.make_hetero` (§2, the INVERTED construction).

    1. `(t, y, region) = make_hetero(n, draw_seed, r=r, sigma=sigma)` -- VERBATIM, never
       re-implemented (`draw_seed` defaults to `seed`, the train-split shape; §3b's selection/report
       provisioning helpers pass an offset `draw_seed` instead).
    2. `u = _uniform_to_standard_normal(t, r)`.
    3. `x = u·v + (I − vvᵀ)z`, `v = geometry_vector(d, geometry, seed)` (a CELL-level draw, so every
       split at this `(d, geometry, seed)` cell uses the SAME `v`), `z ~ N(0, I_d)` from a decoupled
       stream keyed on `draw_seed` (`_Z_SEED_OFFSET`) -- never `make_hetero`'s own generator, and
       never `v`'s own stream, so `u`, `z` and `v` are three independent draws.

    Args:
        n: number of points.
        seed: the CELL's base seed -- determines `v` (shared by every split of this cell).
        d: input dimension.
        geometry: `Geometry.AXIS` or `Geometry.OBLIQUE`.
        draw_seed: the seed `make_hetero` (and the decoupled `z` stream) actually draws from.
            Defaults to `seed`. Passing `seed + SELECTION_SEED_OFFSET` / `seed + REPORT_SEED_OFFSET`
            makes this split's `(t, y, region)` an INDEPENDENT draw from train's, while `v` --
            always read off `seed`, never `draw_seed` -- stays identical across all three (C1).
        r: domain half-width, passed straight through to `make_hetero`.
        sigma: noise std, passed straight through to `make_hetero`.

    Returns:
        `(x, y, region, t)`. `x` is `(n, d)` float32, exactly `N(0, I_d)`-distributed with
        `v . x[i] == u[i]` (`u` the PIT-lifted `t`) to float32 precision. `y`/`region`/`t` are
        `make_hetero`'s own outputs, untouched -- the §5.1 identity.
    """
    resolved_draw_seed = seed if draw_seed is None else draw_seed
    t, y, region = nwn.make_hetero(n, resolved_draw_seed, r=r, sigma=sigma)
    v = geometry_vector(d, geometry, seed)

    u = _uniform_to_standard_normal(t, r)
    z_rng = np.random.default_rng(int(resolved_draw_seed) + _Z_SEED_OFFSET)
    z = z_rng.standard_normal((n, int(d)))
    z_along_v = z @ v
    x = np.outer(u, v) + z - np.outer(z_along_v, v)

    return x.astype(np.float32), y, region, t


def make_train_split(
    seed: int, d: int, geometry: Geometry, r: float = nwn.HETERO_R_DEFAULT, sigma: float = nwn.HETERO_NOISE_SIGMA
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The per-cell TRAINING split (§3b): `TRAIN_N` (1500) points at the cell's own base seed."""
    return make_hetero_multifeature(TRAIN_N, seed, d, geometry, r=r, sigma=sigma)


def make_selection_split(
    seed: int, d: int, geometry: Geometry, n_sel: int, r: float = nwn.HETERO_R_DEFAULT, sigma: float = nwn.HETERO_NOISE_SIGMA
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The per-cell SELECTION split (§3b): an INDEPENDENT draw of `n_sel` points, `seed + SELECTION_SEED_OFFSET`."""
    return make_hetero_multifeature(n_sel, seed, d, geometry, draw_seed=seed + SELECTION_SEED_OFFSET, r=r, sigma=sigma)


def make_report_split(
    seed: int, d: int, geometry: Geometry, r: float = nwn.HETERO_R_DEFAULT, sigma: float = nwn.HETERO_NOISE_SIGMA
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The per-cell REPORT/held-out split (§3b): an INDEPENDENT draw of `REPORT_N` (2000) points, `seed + REPORT_SEED_OFFSET`."""
    return make_hetero_multifeature(REPORT_N, seed, d, geometry, draw_seed=seed + REPORT_SEED_OFFSET, r=r, sigma=sigma)


def identity_holds(
    n: int, seed: int, d: int, geometry: Geometry, r: float = nwn.HETERO_R_DEFAULT, sigma: float = nwn.HETERO_NOISE_SIGMA
) -> bool:
    """§5.1: the produced `(t, y, region)` equals `make_hetero(n, seed)`'s raw output bit-for-bit.

    Holds BY CONSTRUCTION -- §2's inversion never touches `t`/`y`/`region`, it only reparameterizes
    them into `x`. This is the pre-registered check that ASSERTS the construction did not
    accidentally perturb them, not a design choice with two possible outcomes.

    Returns:
        `True` iff `t`, `y` and `region` are bit-for-bit identical to a direct `make_hetero` call
        with the same `(n, seed, r, sigma)`.
    """
    _x, y, region, t = make_hetero_multifeature(n, seed, d, geometry, r=r, sigma=sigma)
    t_ref, y_ref, region_ref = nwn.make_hetero(n, seed, r=r, sigma=sigma)
    return bool(np.array_equal(t, t_ref)) and bool(np.array_equal(y, y_ref)) and bool(np.array_equal(region, region_ref))


def run_selftest() -> bool:
    """No-training checks: §5.1 identity, `x`'s marginals, `v.x == u`, and provisioning independence."""
    ok = True

    for d in _SELFTEST_D:
        for geometry in Geometry:
            ok_identity = identity_holds(_SELFTEST_N, _SELFTEST_SEED, d, geometry)
            ok = ok and ok_identity
            print(f"[wsel19_toys selftest] identity d={d} geometry={geometry.value}  {'PASS' if ok_identity else 'FAIL'}")

            x, _y, _region, t = make_hetero_multifeature(_SELFTEST_N, _SELFTEST_SEED, d, geometry)

            mean_err = float(np.abs(x.mean(axis=0)).max())
            std_err = float(np.abs(x.std(axis=0) - 1.0).max())
            ok_marginal = mean_err < _MARGINAL_TOL and std_err < _MARGINAL_TOL
            ok = ok and ok_marginal
            print(
                f"[wsel19_toys selftest] marginals d={d} geometry={geometry.value} max|mean|={mean_err:.3e} "
                f"max|std-1|={std_err:.3e} (tol={_MARGINAL_TOL:.0e})  {'PASS' if ok_marginal else 'FAIL'}"
            )

            v = geometry_vector(d, geometry, _SELFTEST_SEED)
            u = _uniform_to_standard_normal(t, nwn.HETERO_R_DEFAULT)
            dot_err = float(np.abs(x.astype(np.float64) @ v - u).max())
            ok_dot = dot_err < _DOT_TOL
            ok = ok and ok_dot
            print(f"[wsel19_toys selftest] v.x==u d={d} geometry={geometry.value} max_abs_err={dot_err:.3e} (tol={_DOT_TOL:.0e})  {'PASS' if ok_dot else 'FAIL'}")

    # Provisioning independence (§3b/C8): train/selection/report are genuinely different draws at
    # one representative cell -- not just non-crashing, but bitwise-DIFFERENT `t` per split.
    d, geometry, n_sel = D_GRID[0], Geometry.AXIS, N_SEL_GRID[0]
    x_tr, _y_tr, _region_tr, t_tr = make_train_split(_SELFTEST_SEED, d, geometry)
    x_sel, _y_sel, _region_sel, t_sel = make_selection_split(_SELFTEST_SEED, d, geometry, n_sel)
    x_rep, _y_rep, _region_rep, t_rep = make_report_split(_SELFTEST_SEED, d, geometry)
    ok_shapes = x_tr.shape == (TRAIN_N, d) and x_sel.shape == (n_sel, d) and x_rep.shape == (REPORT_N, d)
    ok_independent = not np.array_equal(t_tr[: min(len(t_tr), len(t_sel))], t_sel[: min(len(t_tr), len(t_sel))]) and not np.array_equal(
        t_tr[: min(len(t_tr), len(t_rep))], t_rep[: min(len(t_tr), len(t_rep))]
    )
    ok_provisioning = ok_shapes and ok_independent
    ok = ok and ok_provisioning
    print(f"[wsel19_toys selftest] provisioning shapes_ok={ok_shapes} independent_ok={ok_independent}  {'PASS' if ok_provisioning else 'FAIL'}")

    print(f"[wsel19_toys selftest] {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    """Parses args and runs the selftest, or prints help (no standalone real-run mode -- the driver owns the grid)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="No-training identity/marginal/projection/provisioning checks across the d x geometry selftest grid.")
    args = parser.parse_args()
    if args.selftest:
        sys.exit(0 if run_selftest() else 1)
    parser.print_help()


if __name__ == "__main__":
    main()
