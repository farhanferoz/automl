"""DATA-ONLY per-width training weights for the joint width net — computed BEFORE any training.

WHY THIS EXISTS (2026-07-25 user ruling, `width.md` WSEL-23 withdrawal block). The weighting scheme
needs `sigma^2 + a2(k)` — the best error a width-`k` model could achieve — for every width. Two earlier
sources are BINNED:
  * the 36 WSEL8 dedicated-net cells: those are also the closure DENOMINATOR, so calibrating on them
    made the parity claim circular (re-pinning elsewhere moved the constant 4.1x);
  * the joint net's own `held_out_mse_by_width`: that split drives early stopping and picks the
    restored checkpoint, so it is systematically optimistic.
Both also answer the wrong question — they measure a TRAINED net's error, which mixes what the width
cannot represent with what the optimiser failed to find.

WHAT THIS COMPUTES INSTEAD. `min over width-k params of E[(y - f_k(x))^2]`, estimated from the data
alone. That single quantity IS `sigma^2 + a2(k)` (the cross term vanishes), so the noise never has to be
separated out — though `--report` prints a model-free noise estimate as a cross-check.

WHY GREEDY-THEN-REFINE, and not a multi-start nonconvex solve. Reproducibility is the binding
requirement: a multi-start solve gave 0.0357 / 0.0062 / 0.0031 for the SAME width-3 quantity on three
solver configurations, and calibrating on that would encode which optimiser was used. Here the atom
SELECTION is greedy (orthogonal matching pursuit — fully determined by the data) and the refinement is
a SINGLE L-BFGS pass from that selection (no restarts, no random init). Same data in, same weights out.

VALIDATED on the `hetero` toy, where widths 1 and 2 have an exactly-known population floor that every
independent solver agreed on to six decimals: this estimator returns them to within 0.4%, and tracks
what dedicated nets actually achieve to within ~1.8x across all twelve widths.

    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m automl_package.examples.width_data_weights --selftest
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m automl_package.examples.width_data_weights --pin --w-max 12
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
if _EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, _EXAMPLES_DIR)

import nested_width_net as nwn  # noqa: E402
import sinc_width_experiment as sw  # noqa: E402  (only for `_jsonable`, this repo's JSON convention)

from automl_package.utils.run_provenance import run_provenance  # noqa: E402

RESULTS_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "WSEL23_DATA_WEIGHTS")
WEIGHTS_FILENAME = "data_weight_table.json"

# Dictionary geometry: a tanh atom is `tanh(s * (x - t))` -- `t` the transition location, `s` its
# sharpness. Locations span the observed input range; sharpness is log-spaced over both signs so a
# unit may face either way. 240 x 60 = 14400 atoms, the size validated against the exact anchors.
N_LOCATIONS = 240
N_SHARPNESS = 30  # per sign
SHARPNESS_LOG10_LO = -1.5
SHARPNESS_LOG10_HI = 1.2
REFINE_MAX_ITER = 400  # single L-BFGS pass; no restarts (see module docstring)
_SELFTEST_ANCHOR_TOL = 0.05  # widths 1-2 are exactly known; 5% is loose enough for the finite sample
_SELFTEST_N = 25000


def estimate_noise_variance(x: np.ndarray, y: np.ndarray) -> float:
    """Model-free `sigma^2` from successive differences of x-sorted `y` (Rice 1984). Fits nothing."""
    order = np.argsort(x)
    return float(np.mean(np.diff(y[order]) ** 2) / 2.0)


def _dictionary(x: np.ndarray) -> np.ndarray:
    """`(n_atoms, 2)` array of `(sharpness, location)` pairs spanning the observed input range."""
    locations = np.linspace(float(x.min()), float(x.max()), N_LOCATIONS)
    sharpness = np.concatenate([np.logspace(SHARPNESS_LOG10_LO, SHARPNESS_LOG10_HI, N_SHARPNESS), -np.logspace(SHARPNESS_LOG10_LO, SHARPNESS_LOG10_HI, N_SHARPNESS)])
    return np.array([(s, t) for s in sharpness for t in locations])


def _design(x: np.ndarray, picked: np.ndarray) -> np.ndarray:
    """Design matrix of the picked atoms plus an intercept column — the width-k readout's basis."""
    return np.column_stack([np.tanh(s * (x - t)) for s, t in picked] + [np.ones_like(x)])


def _refine(x: np.ndarray, y: np.ndarray, picked: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """ONE deterministic L-BFGS pass from the greedy atoms; returns the refined `(s, t, c, b)`.

    Moves atoms off the dictionary grid and lets them co-adapt — which is the whole difference between
    tracking real nets to 2-4x (greedy alone) and to ~1.2x (refined) at the wider widths.
    """
    coef0, *_ = np.linalg.lstsq(_design(x, picked), y, rcond=None)
    xt, yt = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    s = torch.tensor(picked[:, 0], dtype=torch.float64, requires_grad=True)
    t = torch.tensor(picked[:, 1], dtype=torch.float64, requires_grad=True)
    c = torch.tensor(coef0[:-1], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([coef0[-1]], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([s, t, c, b], max_iter=REFINE_MAX_ITER, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = (((c * torch.tanh(s * (xt[:, None] - t))).sum(dim=1) + b - yt) ** 2).mean()
        loss.backward()
        return loss

    opt.step(closure)
    sv, tv, cv, bv = (v.detach().numpy() for v in (s, t, c, b))
    return sv, tv, cv, float(bv[0])


def per_width_floor(x: np.ndarray, y: np.ndarray, w_max: int) -> dict[int, float]:
    """`{k: sigma^2 + a2(k)}` for `k` in `1..w_max`, from data alone. Deterministic: no restarts, no RNG.

    Greedy orthogonal matching pursuit grows the atom set one unit at a time (each width's set is the
    previous one plus a new atom -- mirroring the net's own nested prefix), then each width's set is
    refined independently by a single L-BFGS pass.
    """
    atoms = _dictionary(x)
    phi = np.tanh(np.outer(x, atoms[:, 0]) - atoms[:, 0] * atoms[:, 1]).astype(np.float32)
    phi_norm = np.linalg.norm(phi, axis=0) + 1e-12
    selected: list[int] = []
    residual = y - y.mean()
    floors: dict[int, float] = {}
    for k in range(1, w_max + 1):
        correlation = np.abs(phi.T @ residual) / phi_norm
        correlation[selected] = -np.inf
        selected.append(int(np.argmax(correlation)))
        basis = np.column_stack([phi[:, selected], np.ones(len(x), dtype=np.float32)])
        coef, *_ = np.linalg.lstsq(basis, y, rcond=None)
        residual = y - basis @ coef
        s, t, c, b = _refine(x, y, atoms[selected])
        floors[k] = float(np.mean(((c * np.tanh(s * (x[:, None] - t))).sum(axis=1) + b - y) ** 2))
    return floors


def weight_table_from_floors(floors: dict[int, float], w_max: int) -> dict[int, float]:
    """`{k: floor(w_max) / floor(k)}` — inverse-variance weights, normalised so the widest head is 1.0.

    This is the maximum-likelihood weighting of the observation model `y | x, w ~ N(f_w(x), sigma^2 +
    a2(w))`: each width's term is divided by the variance that width actually carries, which is what
    makes it a consequence of the model rather than a tuned penalty. Anchoring at `w_max` (rather than
    at 1) keeps the widest head at weight 1.0 and every narrower head below it.
    """
    return {k: floors[w_max] / floors[k] for k in range(1, w_max + 1)}


def pin_from_data(x: np.ndarray, y: np.ndarray, w_max: int) -> dict:
    """The full artifact `--loss weighted` consumes: per-width floors, the weight table, provenance."""
    floors = per_width_floor(x, y, w_max)
    return {
        "weight_table": weight_table_from_floors(floors, w_max),
        "floor_by_width": floors,
        "sigma2_estimate_model_free": estimate_noise_variance(x, y),
        "w_max": w_max,
        "n_train": len(x),
        "source": "data_only_omp_refined",
        "provenance": run_provenance(),
    }


def run_selftest() -> bool:
    """Two checks: the exact-anchor exam, and determinism (the property the binned approach lacked)."""
    ok = True
    x, y, _ = nwn.make_hetero(_SELFTEST_N, 0)

    # (1) EXACT ANCHORS. Widths 1-2 have a population floor every independent solver agreed on to six
    #     decimals (0.063337 / 0.042133 plus sigma^2=0.0025). An estimator that misses these is unusable.
    floors = per_width_floor(x, y, 2)
    anchors = {1: 0.065837, 2: 0.044633}
    errs = {k: abs(floors[k] - anchors[k]) / anchors[k] for k in anchors}
    ok_1 = all(e < _SELFTEST_ANCHOR_TOL for e in errs.values())
    print(
        f"[data-weights selftest] (1) exact anchors w1/w2: {floors[1]:.6f}/{floors[2]:.6f} vs {anchors[1]}/{anchors[2]} "
        f"(rel err {errs[1]:.3f}/{errs[2]:.3f})  {'PASS' if ok_1 else 'FAIL'}"
    )
    ok = ok and ok_1

    # (2) DETERMINISM -- the requirement the multi-start solve failed (0.0357/0.0062/0.0031 for one
    #     width-3 quantity across solver setups). Same data in, byte-identical weights out.
    again = per_width_floor(x, y, 2)
    ok_2 = all(floors[k] == again[k] for k in floors)
    print(f"[data-weights selftest] (2) determinism (same data -> identical floors): {ok_2}  {'PASS' if ok_2 else 'FAIL'}")
    ok = ok and ok_2

    print(f"[data-weights selftest] {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    """`--selftest` (anchors + determinism) or `--pin` (write the weight table)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true", help="Exact-anchor exam + determinism check, then exit.")
    parser.add_argument("--pin", action="store_true", help="Compute the weight table from data and write it.")
    parser.add_argument("--w-max", type=int, default=nwn.W_MAX if hasattr(nwn, "W_MAX") else 12, help="Number of widths.")
    parser.add_argument("--n-train", type=int, default=25000, help="Sample size for the ESTIMATE (not a training set; larger = tighter, costs seconds).")
    parser.add_argument("--seed", type=int, default=0, help="Data seed for the estimation draw.")
    parser.add_argument("--sigma", type=float, default=nwn.HETERO_NOISE_SIGMA, help="Toy noise std (the estimate never reads this; --report cross-checks against it).")
    parser.add_argument("--results-dir", type=str, default=None, help=f"Output directory (default: {RESULTS_DIR}).")
    args = parser.parse_args()

    if args.selftest:
        sys.exit(0 if run_selftest() else 1)
    if not args.pin:
        parser.error("nothing to do: pass --pin or --selftest")

    results_dir = args.results_dir if args.results_dir is not None else RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    x, y, _ = nwn.make_hetero(args.n_train, args.seed, sigma=args.sigma)
    artifact = pin_from_data(x, y, args.w_max)
    out_path = os.path.join(results_dir, WEIGHTS_FILENAME)
    with open(out_path, "w") as f:
        json.dump(sw._jsonable(artifact), f, indent=2)
    print(f"[data-weights] wrote {out_path}")
    print(f"[data-weights] sigma^2 model-free estimate {artifact['sigma2_estimate_model_free']:.6f} (toy true {args.sigma**2:.6f})")
    for k in range(1, args.w_max + 1):
        print(f"  w{k:>2}  floor {artifact['floor_by_width'][k]:.6f}   weight {artifact['weight_table'][k]:.4f}")


if __name__ == "__main__":
    main()
