"""WSEL-19 multi-feature toy construction v2 — the canonical toy embedded into `d` dimensions by a rotation (the rotated-box construction).

`docs/plans/capacity_programme/shared/wsel19-toy-redesign.md` (amended, GO, 2026-07-22) is the
build authority for this module; read it before changing anything here. The v1 probability-
integral-transform lift (`wsel19-toy-design.md`) FAILED its own pre-registered d=1 calibration —
the traced mechanism (`capacity_ladder_results/WSEL19/warp_trace.json`) is two-sided width
convergence caused by the Φ-warp of the model-visible coordinate; redesign requirement R1 bans
any nonlinear reparameterization between the canonical coordinate and what the model sees. This
module is a pure toy-construction library (generators + provisioning helpers) — it trains nothing
and runs no grid; `automl_package/examples/width_wsel19.py` is the driver that consumes it.

**The construction stays INVERTED (F2)** — `nested_width_net.make_hetero` draws its own scalar
input internally (`rng.uniform(-r, r, n)`) with no injection point — but the warp is REMOVED
entirely (R1):

  1. `(t, y, region) = make_hetero(n, draw_seed)` -- the canonical call, VERBATIM (imported below,
     never re-implemented -- genuine §3.9 reuse). `t ~ U(-r, r)` i.i.d.
  2. Decoy coordinates `c_2..c_d ~ U(-r, r)` i.i.d. from a DECOUPLED RNG stream
     (`draw_seed + _DECOY_SEED_OFFSET` -- never `make_hetero`'s own generator).
  3. `x = H t_vec`, `t_vec = (t, c_2, ..., c_d)`, with `H = basis_matrix(d, geometry, seed)` a
     DETERMINISTIC orthogonal matrix whose first column is `v = geometry_vector(d, geometry,
     seed)` (AXIS: `H = I` exactly; OBLIQUE: the Householder reflection mapping `e_1 -> v`).

Exact consequences (each asserted by the selftest, not assumed): `v . x = t` -- the model-visible
coordinate along `v` IS the canonical coordinate, unwarped (R1); at `d = 1` AXIS, `x = t` exactly
(the calibration cell is the canonical toy bit-for-bit); `x` is an exact `H`-rotation of the
i.i.d. `U(-r, r)^d` box, so at AXIS all `d` coordinates are i.i.d. `U(-r, r)` and the input law
is EXCHANGEABLE under coordinate permutation -- no x-only statistic identifies the signal
coordinate (the R4 marginal-sniffing closure; the redesign spec's §6 ledger derives this from
scratch).

**Geometry (§3 grid axis, unchanged from v1).** `Geometry.AXIS` (`v = e_1`) or `Geometry.OBLIQUE`
(`v = s / sqrt(d)`, `s` a per-seed DETERMINISTIC ±1 sign pattern -- maximally oblique, no
near-axis draws). Sparse-oblique is EXCLUDED (recorded decision, unchanged). `v` is a pure
function of `(d, geometry, seed)` -- the CELL's base seed -- so every split at a cell shares the
identical `v`.

**Provisioning (§3b, amended).** `make_train_split` = the canonical size (`TRAIN_N`, 1500) at the
cell's own base seed; `make_report_split` = an INDEPENDENT draw of `REPORT_N` (2000) points at
`seed + REPORT_SEED_OFFSET`. `make_selection_split` is POOL-AND-PREFIX (adjudicator finding 5,
mirroring the certified 1-D slice's own protocol -- `width_wsel19.py`'s selection pool): ONE
independent draw of `SELECTION_POOL_N` (1200) points at `seed + SELECTION_SEED_OFFSET`, shuffled
by a seeded permutation (`np.random.default_rng(seed)`); every smaller `n_sel` is that shuffled
pool's PREFIX, so the three sizes are NESTED and a size-vs-size comparison reflects added
selection data, never independently-resampled noise. Selection is never carved from the training
set (C8).

Usage:
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel19_toys.py --selftest
"""

from __future__ import annotations

import argparse
import enum
import os
import sys

import numpy as np

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
SELECTION_POOL_N = max(N_SEL_GRID)  # §3b (amended): the ONE selection pool every n_sel prefixes.

SELECTION_SEED_OFFSET = 1000  # §3b: the selection POOL = an independent §2 draw at seed + this offset.
REPORT_SEED_OFFSET = 2000  # §3b: report set = an independent §2 draw at seed + this offset.

# Both offsets below decouple an AUXILIARY draw (the decoy coordinates, or the OBLIQUE sign
# pattern) from a §2-construction call's own `(t, y, region)` stream -- the same "seed + fixed
# offset" idiom §3b already uses for SELECTION_SEED_OFFSET/REPORT_SEED_OFFSET, and already on
# disk elsewhere (`converged_width_experiment.py`'s held-out `seed + 500` test draw). Chosen far
# enough from every combination of a cell seed x {0, SELECTION_SEED_OFFSET, REPORT_SEED_OFFSET}
# in this strand's grid that no two draws can ever collide.
_DECOY_SEED_OFFSET = 5_000  # decouples the decoy-coordinate draw from make_hetero's own RNG stream (v1's z-stream offset, same slot).
_SIGN_PATTERN_SEED_OFFSET = 9_000  # decouples OBLIQUE's sign-pattern draw; keyed on the CELL seed only (shared by every split).

# §5.1 selftest tolerances (redesign spec §5.1 -- scale-aware, adjudicator finding 6: the v1
# N(0,1)-era 0.1 constant legitimately fails on box scale; measured on a correct n=5000, d=32
# AXIS draw: max|mean_j|=0.28, max|std_j - r/sqrt(3)|=0.157 -- both well inside 5% of scale,
# while a 10%-of-scale bug trips both bounds).
_MEAN_TOL_FRAC = 0.05  # |mean_j| <= this fraction of r, AXIS only.
_STD_TOL_FRAC = 0.05  # |std_j - r/sqrt(3)| <= this fraction of r/sqrt(3), AXIS only.
_DOT_TOL = 1e-5  # v.x == t to float32 precision (x is stored float32; measured cast error <= 7.4e-7 at d=32 OBLIQUE).
_ORTHO_TOL = 1e-12  # Householder identities: ||H^T H - I||_max and ||H e_1 - v||_max (measured <= 1.6e-15).

_SELFTEST_D = (1, *D_GRID)  # d=1 additionally exercises the exact reduction the calibration cell relies on.
_SELFTEST_N = 5_000
_SELFTEST_SEED = 0


class Geometry(enum.Enum):
    """Closed set of `v` geometries (`width.md` / toy-design §3). Sparse-oblique is EXCLUDED (recorded decision)."""

    AXIS = "axis"  # v = e_1
    OBLIQUE = "oblique"  # v = s / sqrt(d), s a per-seed deterministic +-1 sign pattern


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
    and maximally oblique (no near-axis draws, no per-seed direction variance -- the v1
    adjudicator's amendment, retained). Both are exact unit vectors (`||v|| = 1`).

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


def basis_matrix(d: int, geometry: Geometry, seed: int) -> np.ndarray:
    """The deterministic orthogonal matrix `H` whose first column is `v` (§2 v2 step 3).

    AXIS: `H = I_d` exactly (no arithmetic at all -- `v = e_1`). OBLIQUE: the Householder
    reflection `H = I - 2 w w^T / ||w||^2` with `w = e_1 - v`, which is orthogonal, symmetric,
    and maps `e_1 -> v` (standard identities; asserted numerically by the selftest at
    `_ORTHO_TOL`, not taken on faith). If `w = 0` (i.e. `v = e_1`, e.g. the OBLIQUE `d=1`
    all-plus sign pattern), `H = I`.

    Args:
        d: input dimension (`d >= 1`).
        geometry: `Geometry.AXIS` or `Geometry.OBLIQUE`.
        seed: the cell's base seed (`v` is a pure function of `(d, geometry, seed)`).

    Returns:
        `(d, d)` float64 orthogonal matrix with `H[:, 0] == geometry_vector(d, geometry, seed)`.
    """
    v = geometry_vector(d, geometry, seed)
    if geometry is Geometry.AXIS:
        return np.eye(int(d), dtype=np.float64)
    e1 = np.zeros(int(d), dtype=np.float64)
    e1[0] = 1.0
    w = e1 - v
    w_norm_sq = float(w @ w)
    if w_norm_sq == 0.0:
        return np.eye(int(d), dtype=np.float64)
    return np.eye(int(d), dtype=np.float64) - (2.0 / w_norm_sq) * np.outer(w, w)


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
    """The `d`-dimensional rotated-box embedding of `nested_width_net.make_hetero` (§2 v2, the INVERTED construction).

    1. `(t, y, region) = make_hetero(n, draw_seed, r=r, sigma=sigma)` -- VERBATIM, never
       re-implemented (`draw_seed` defaults to `seed`, the train-split shape; §3b's
       selection/report provisioning helpers pass an offset `draw_seed` instead).
    2. Decoys `c_2..c_d ~ U(-r, r)` i.i.d. from a decoupled stream keyed on `draw_seed`
       (`_DECOY_SEED_OFFSET`) -- never `make_hetero`'s own generator, and never `v`'s own stream,
       so `t`, the decoys and `v` are three independent draws.
    3. `x = H t_vec`, `t_vec = (t, c_2, ..., c_d)`, `H = basis_matrix(d, geometry, seed)` (a
       CELL-level object, so every split at this `(d, geometry, seed)` cell uses the SAME `H`).

    Args:
        n: number of points.
        seed: the CELL's base seed -- determines `v`/`H` (shared by every split of this cell).
        d: input dimension.
        geometry: `Geometry.AXIS` or `Geometry.OBLIQUE`.
        draw_seed: the seed `make_hetero` (and the decoupled decoy stream) actually draws from.
            Defaults to `seed`. Passing `seed + SELECTION_SEED_OFFSET` / `seed + REPORT_SEED_OFFSET`
            makes this split's `(t, y, region)` an INDEPENDENT draw from train's, while `v`/`H` --
            always read off `seed`, never `draw_seed` -- stay identical across all three (C1).
        r: domain half-width, passed straight through to `make_hetero`.
        sigma: noise std, passed straight through to `make_hetero`.

    Returns:
        `(x, y, region, t)`. `x` is `(n, d)` float32, an exact `H`-rotation of the i.i.d.
        `U(-r, r)^d` box, with `v . x[i] == t[i]` to float32 precision (R1: the model-visible
        coordinate along `v` IS the canonical coordinate, unwarped). `y`/`region`/`t` are
        `make_hetero`'s own outputs, untouched -- the §5.1 identity.
    """
    resolved_draw_seed = seed if draw_seed is None else draw_seed
    t, y, region = nwn.make_hetero(n, resolved_draw_seed, r=r, sigma=sigma)
    h_mat = basis_matrix(d, geometry, seed)

    t_vec = np.empty((n, int(d)), dtype=np.float64)
    t_vec[:, 0] = t
    if int(d) > 1:
        decoy_rng = np.random.default_rng(int(resolved_draw_seed) + _DECOY_SEED_OFFSET)
        t_vec[:, 1:] = decoy_rng.uniform(-r, r, size=(n, int(d) - 1))
    x = t_vec @ h_mat.T

    return x.astype(np.float32), y, region, t


def make_train_split(
    seed: int, d: int, geometry: Geometry, r: float = nwn.HETERO_R_DEFAULT, sigma: float = nwn.HETERO_NOISE_SIGMA
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The per-cell TRAINING split (§3b): `TRAIN_N` (1500) points at the cell's own base seed."""
    return make_hetero_multifeature(TRAIN_N, seed, d, geometry, r=r, sigma=sigma)


def make_selection_split(
    seed: int, d: int, geometry: Geometry, n_sel: int, r: float = nwn.HETERO_R_DEFAULT, sigma: float = nwn.HETERO_NOISE_SIGMA
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The per-cell SELECTION split (§3b, amended -- POOL-AND-PREFIX, adjudicator finding 5).

    ONE independent draw of `SELECTION_POOL_N` (1200) points at `seed + SELECTION_SEED_OFFSET`,
    shuffled by a seeded permutation (`np.random.default_rng(seed)`, the certified 1-D slice's own
    convention); the returned split is that shuffled pool's first `n_sel` rows. The three §3 sizes
    are therefore NESTED -- a size-vs-size comparison reflects added selection data, never
    independently-resampled noise.

    Args:
        seed: the CELL's base seed.
        d: input dimension.
        geometry: `Geometry.AXIS` or `Geometry.OBLIQUE`.
        n_sel: selection-set size (`<= SELECTION_POOL_N`).
        r: domain half-width, passed through.
        sigma: noise std, passed through.

    Returns:
        `(x, y, region, t)`, each the shuffled pool's `n_sel`-prefix.
    """
    if int(n_sel) > SELECTION_POOL_N:
        raise ValueError(f"n_sel={n_sel} exceeds the selection pool size {SELECTION_POOL_N}")
    x, y, region, t = make_hetero_multifeature(SELECTION_POOL_N, seed, d, geometry, draw_seed=seed + SELECTION_SEED_OFFSET, r=r, sigma=sigma)
    perm = np.random.default_rng(int(seed)).permutation(SELECTION_POOL_N)
    idx = perm[: int(n_sel)]
    return x[idx], y[idx], region[idx], t[idx]


def make_report_split(
    seed: int, d: int, geometry: Geometry, r: float = nwn.HETERO_R_DEFAULT, sigma: float = nwn.HETERO_NOISE_SIGMA
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The per-cell REPORT/held-out split (§3b): an INDEPENDENT draw of `REPORT_N` (2000) points, `seed + REPORT_SEED_OFFSET`."""
    return make_hetero_multifeature(REPORT_N, seed, d, geometry, draw_seed=seed + REPORT_SEED_OFFSET, r=r, sigma=sigma)


def identity_holds(
    n: int, seed: int, d: int, geometry: Geometry, r: float = nwn.HETERO_R_DEFAULT, sigma: float = nwn.HETERO_NOISE_SIGMA
) -> bool:
    """§5.1: the produced `(t, y, region)` equals `make_hetero(n, seed)`'s raw output bit-for-bit.

    Holds BY CONSTRUCTION -- §2's inversion never touches `t`/`y`/`region`, it only embeds them
    into `x`. This is the pre-registered check that ASSERTS the construction did not accidentally
    perturb them, not a design choice with two possible outcomes.

    Returns:
        `True` iff `t`, `y` and `region` are bit-for-bit identical to a direct `make_hetero` call
        with the same `(n, seed, r, sigma)`.
    """
    _x, y, region, t = make_hetero_multifeature(n, seed, d, geometry, r=r, sigma=sigma)
    t_ref, y_ref, region_ref = nwn.make_hetero(n, seed, r=r, sigma=sigma)
    return bool(np.array_equal(t, t_ref)) and bool(np.array_equal(y, y_ref)) and bool(np.array_equal(region, region_ref))


def run_selftest() -> bool:
    """No-training checks (§5.1 v2): identity, Householder identities, `v.x == t`, the AXIS box law, the d=1 exact reduction, and provisioning."""
    ok = True
    r = nwn.HETERO_R_DEFAULT

    for d in _SELFTEST_D:
        for geometry in Geometry:
            ok_identity = identity_holds(_SELFTEST_N, _SELFTEST_SEED, d, geometry)
            ok = ok and ok_identity
            print(f"[wsel19_toys selftest] identity d={d} geometry={geometry.value}  {'PASS' if ok_identity else 'FAIL'}")

            # Householder identities: H orthogonal, H e1 = v (AXIS: H = I trivially satisfies both).
            v = geometry_vector(d, geometry, _SELFTEST_SEED)
            h_mat = basis_matrix(d, geometry, _SELFTEST_SEED)
            ortho_err = float(np.abs(h_mat.T @ h_mat - np.eye(d)).max())
            map_err = float(np.abs(h_mat[:, 0] - v).max())
            ok_ortho = ortho_err <= _ORTHO_TOL and map_err <= _ORTHO_TOL
            ok = ok and ok_ortho
            print(
                f"[wsel19_toys selftest] householder d={d} geometry={geometry.value} ||HtH-I||={ortho_err:.3e} "
                f"||He1-v||={map_err:.3e} (tol={_ORTHO_TOL:.0e})  {'PASS' if ok_ortho else 'FAIL'}"
            )

            x, _y, _region, t = make_hetero_multifeature(_SELFTEST_N, _SELFTEST_SEED, d, geometry)

            # R1: the model-visible coordinate along v IS t (float32 storage cast only).
            dot_err = float(np.abs(x.astype(np.float64) @ v - t).max())
            ok_dot = dot_err < _DOT_TOL
            ok = ok and ok_dot
            print(f"[wsel19_toys selftest] v.x==t d={d} geometry={geometry.value} max_abs_err={dot_err:.3e} (tol={_DOT_TOL:.0e})  {'PASS' if ok_dot else 'FAIL'}")

            # AXIS ONLY (spec §5.1): box support + scale-aware moment checks. At OBLIQUE the
            # per-coordinate support exceeds r BY DESIGN (rotated box, measured up to ~2.4r at
            # d=32) -- support/moment checks are AXIS-scoped structurally, not leniently.
            if geometry is Geometry.AXIS:
                support_err = float(np.abs(x).max())
                mean_err = float(np.abs(x.mean(axis=0)).max())
                std_err = float(np.abs(x.std(axis=0) - r / np.sqrt(3.0)).max())
                ok_box = support_err <= r and mean_err <= _MEAN_TOL_FRAC * r and std_err <= _STD_TOL_FRAC * r / np.sqrt(3.0)
                ok = ok and ok_box
                print(
                    f"[wsel19_toys selftest] axis-box d={d} max|x|={support_err:.4f}<=r max|mean|={mean_err:.3f} "
                    f"max|std-r/sqrt3|={std_err:.3f}  {'PASS' if ok_box else 'FAIL'}"
                )

            # d=1 AXIS: the exact reduction the calibration cell relies on -- x IS t, bit-for-bit
            # after the float32 storage cast.
            if d == 1 and geometry is Geometry.AXIS:
                ok_reduction = bool(np.array_equal(x[:, 0], t.astype(np.float32)))
                ok = ok and ok_reduction
                print(f"[wsel19_toys selftest] d=1 axis exact reduction x==t(float32)  {'PASS' if ok_reduction else 'FAIL'}")

    # Provisioning (§3b/C8): train/selection/report are genuinely different draws at one
    # representative cell, and the amended selection splits are NESTED (pool-and-prefix).
    d, geometry = D_GRID[0], Geometry.AXIS
    x_tr, _y_tr, _region_tr, t_tr = make_train_split(_SELFTEST_SEED, d, geometry)
    x_rep, _y_rep, _region_rep, t_rep = make_report_split(_SELFTEST_SEED, d, geometry)
    x_sel_small, _y_s, _r_s, t_sel_small = make_selection_split(_SELFTEST_SEED, d, geometry, N_SEL_GRID[0])
    x_sel_mid, _y_m, _r_m, t_sel_mid = make_selection_split(_SELFTEST_SEED, d, geometry, N_SEL_GRID[1])
    ok_shapes = x_tr.shape == (TRAIN_N, d) and x_sel_small.shape == (N_SEL_GRID[0], d) and x_rep.shape == (REPORT_N, d)
    ok_independent = not np.array_equal(t_tr[: len(t_sel_small)], t_sel_small) and not np.array_equal(t_tr[: min(len(t_tr), len(t_rep))], t_rep[: min(len(t_tr), len(t_rep))])
    ok_nested = bool(np.array_equal(t_sel_small, t_sel_mid[: N_SEL_GRID[0]])) and bool(np.array_equal(x_sel_small, x_sel_mid[: N_SEL_GRID[0]]))
    ok_provisioning = ok_shapes and ok_independent and ok_nested
    ok = ok and ok_provisioning
    print(f"[wsel19_toys selftest] provisioning shapes_ok={ok_shapes} independent_ok={ok_independent} nested_ok={ok_nested}  {'PASS' if ok_provisioning else 'FAIL'}")

    print(f"[wsel19_toys selftest] {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    """Parses args and runs the selftest, or prints help (no standalone real-run mode -- the driver owns the grid)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="No-training identity/householder/projection/box/provisioning checks across the d x geometry selftest grid.")
    args = parser.parse_args()
    if args.selftest:
        sys.exit(0 if run_selftest() else 1)
    parser.print_help()


if __name__ == "__main__":
    main()
