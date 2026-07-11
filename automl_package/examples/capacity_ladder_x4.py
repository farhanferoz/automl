"""X4: E-lane non-nested discriminator on the identical K4 data (WS1 capacity-ladder follow-up, task X4).

(docs/plans/capacity_ladder_2026-07-09/EXECUTION_PLAN.md §8.5 X4; pre-registration in
 capacity_ladder_results/X4/PREREGISTRATION.md)

K4 (`capacity_ladder_k4.py`) trains a NESTED-k surrogate on toy E (moving modes: two components
merge at both ends of x and resolve only in the middle band, ground truth ``1 -> 2 -> 1``) and
finds it seed-fragile: the global knee reads flat/partial on 2/3 seeds. The mid-June NON-nested
instrument -- the per-input held-out arbiter (mixture-vs-best-single-Gaussian NLL advantage,
neighbour-averaged) -- apparently recovered the hump on a single fit (tercile read approximately
-0.018 / +0.149 / -0.026: ~0 at both x-tails, positive in the middle). W8's hypothesis: E's
nested failure is NESTING-specific (one global component ordering cannot serve an x-varying
importance ordering), not generic per-input-count unidentifiability. X4 tests this by running the
SAME non-nested arbiter on the identical K4 E data across 3 seeds.

Instrument (reused verbatim, not reimplemented): the per-input held-out arbiter and gold-standard
machinery from `probreg_variational_em_toy_e_hump.py::run_condition`, which itself drives
`probreg_variational_em_step2_perinput_arbiter.py` (independent plain-Gaussian baseline +
per-point held-out NLL) and `probreg_variational_em_step3_perinput_model.py` /
`_variational_em_perinput.py` (the per-input `AggregateSparsityKSelector` mixture and its
effective-count readout). `run_condition` bakes its data size (N_TR=1500, N_TE=4000) into module
globals rather than parameters, so X4 replicates its per-seed body -- calling the SAME training
and scoring primitives (`vemp.train_aggregate_sparsity`, `p2.train_cond_gaussian`,
`p2.pure_nll_per_point`, `p3.bucket_nll_per_point`, `p3.effective_count`) -- with K4's exact
N_TR=1000/N_TE=2500 and keeps the raw per-point values (`run_condition` only returns the
already-pooled 3-seed mean/std) so X4 can report a per-seed, per-tercile verdict. Everything else
(K_MAX, ALPHA0, SIGMA, SEP_MIN, SEP_MAX, N_EPOCHS, M_GOLD, N_GRID) is the arbiter's own instrument
configuration and is reused unchanged from `probreg_variational_em_toy_e_hump`.

Pre-registered bars (PREREGISTRATION.md):
  * X4-recovery: per seed, the middle x-tercile Δ̂(x) mean is significantly positive (own two-sided
    band) AND exceeds both tail terciles, on ALL 3 seeds -> W8 CONFIRMED (nesting-specific).
    Fewer than 3/3 -> W8 NOT supported (moving-mode count is hard for any instrument).
  * X4-no-false-positive: on E_broad (variance humps, never bimodal) the arbiter must NOT credit
    the middle band on any of the 3 seeds.
  * Consistency (soft): recovered tercile means are in the neighbourhood of the June single-fit
    read (-0.018 / +0.149 / -0.026); large divergence is noted, not failed.

Strictly probabilistic: the ELBO/MAP objective's Dirichlet-usage prior is the model's own term
(coefficient 1, no tuned lambda) -- unchanged from `_variational_em_perinput.py`.

Usage:
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=2 ~/dev/.venv/bin/python automl_package/examples/capacity_ladder_x4.py --selftest
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=2 ~/dev/.venv/bin/python automl_package/examples/capacity_ladder_x4.py --smoke
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=2 ~/dev/.venv/bin/python automl_package/examples/capacity_ladder_x4.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from collections.abc import Callable

import numpy as np
import torch

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root, so `import automl_package` works

import _toy_datasets as td  # noqa: E402
import _variational_em_perinput as vemp  # noqa: E402
import probreg_variational_em_step2_perinput_arbiter as p2  # noqa: E402
import probreg_variational_em_step3_perinput_model as p3  # noqa: E402
import probreg_variational_em_toy_e_hump as hump  # noqa: E402

OUT_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "X4")
K4_SUMMARY_PATH = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "K4", "k4_summary.json")

# Identical-to-K4 data spec (PREREGISTRATION.md "Identical-K4-E-data requirement"): K4's
# `run_structured_toy("E", seed)` used N_TR=1000/N_TE=2500 with `cl._N_TR`/`cl._N_TE`.
N_TR, N_TE = 1000, 2500
SEEDS = (0, 1, 2)
TORCH_THREADS = 2

# The arbiter's own instrument config, reused UNCHANGED from probreg_variational_em_toy_e_hump.py.
N_EPOCHS = hump.N_EPOCHS
M_GOLD, N_GRID = hump.M_GOLD, hump.N_GRID

# The June single-fit tercile read this task discriminates against (soft consistency check only).
JUNE_SINGLE_FIT_READ = {"edge_lo": -0.018, "middle": 0.149, "edge_hi": -0.026}

TOY_SPECS: dict[str, tuple[Callable[..., tuple[np.ndarray, np.ndarray]], Callable[..., np.ndarray]]] = {
    "E": (td.make_toy_e, td.sample_toy_e_given_x),
    "E_broad": (td.make_toy_e_broad, td.sample_toy_e_broad_given_x),
}


# ---------------------------------------------------------------------------
# Small shared utilities.
# ---------------------------------------------------------------------------


def _jsonable(obj: object) -> object:
    """Recursively converts numpy/torch scalars and arrays to plain Python/JSON types."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def _tercile_masks(x: np.ndarray) -> list[np.ndarray]:
    """Boolean masks selecting the edge / middle / edge x-terciles of ``[0, 1]``."""
    edges = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
    masks = []
    for i, (lo, hi) in enumerate(itertools.pairwise(edges)):
        masks.append((x >= lo) & (x <= hi) if i == 2 else (x >= lo) & (x < hi))
    return masks


def _tercile_means(x: np.ndarray, v: np.ndarray) -> dict[str, float]:
    """Mean of ``v`` within each x-tercile (edge_lo / middle / edge_hi), no dispersion."""
    lo_m, mid_m, hi_m = _tercile_masks(x)
    return {"edge_lo": float(v[lo_m].mean()), "middle": float(v[mid_m].mean()), "edge_hi": float(v[hi_m].mean())}


def _tercile_means_se(x: np.ndarray, v: np.ndarray) -> dict[str, dict[str, float]]:
    """Per-tercile mean, standard error of the mean, and point count, for a per-point array ``v``."""
    out: dict[str, dict[str, float]] = {}
    for name, m in zip(("edge_lo", "middle", "edge_hi"), _tercile_masks(x), strict=False):
        vv = v[m]
        se = float(vv.std(ddof=1) / np.sqrt(vv.size)) if vv.size > 1 else float("nan")
        out[name] = {"mean": float(vv.mean()), "se": se, "n": int(vv.size)}
    return out


# ---------------------------------------------------------------------------
# Per-seed fit + score: replicates run_condition's body with K4-matched N/seed, keeping raw values.
# ---------------------------------------------------------------------------


def fit_and_score_seed(
    make_fn: Callable[..., tuple[np.ndarray, np.ndarray]],
    given_fn: Callable[..., np.ndarray],
    seed: int,
    n_tr: int,
    n_te: int,
    n_epochs: int,
    m_gold: int,
    n_grid: int,
) -> dict[str, np.ndarray]:
    """Fits the per-input arbiter for ONE seed and returns its per-point and gold-standard curves.

    Calls the exact primitives `probreg_variational_em_toy_e_hump.run_condition` calls per seed
    (`vemp.train_aggregate_sparsity`, `p2.train_cond_gaussian`, `p2.pure_nll_per_point`,
    `p3.bucket_nll_per_point`, `p3.effective_count`) -- the mixture-vs-best-single-Gaussian
    per-input advantage is NOT reimplemented here, only reduced differently (per-seed, per-tercile,
    with the raw per-point values retained instead of only the pooled 3-seed mean/std).

    Args:
        make_fn: toy generator, e.g. `_toy_datasets.make_toy_e` (signature ``(n, seed) -> (x, y)``).
        given_fn: gold-standard resampler at fixed x, e.g. `_toy_datasets.sample_toy_e_given_x`.
        seed: K4-matched seed (train uses `seed`, test uses `seed + 500`).
        n_tr: training set size (K4: 1000).
        n_te: held-out test set size (K4: 2500).
        n_epochs: training epochs for both the mixture and the plain-Gaussian baseline.
        m_gold: resample draws per grid point for the gold-standard curve.
        n_grid: number of grid points in ``[0, 1]`` for the gold-standard curve.

    Returns:
        Dict with ``x`` (test inputs, shape ``(n_te,)``), ``delta`` (per-point held-out NLL
        advantage of the mixture, shape ``(n_te,)``), ``eff`` (per-point effective bucket count,
        shape ``(n_te,)``), ``grid`` (shape ``(n_grid,)``), and ``gold`` (gold-standard Δ*(x) on
        ``grid``, shape ``(n_grid,)``).
    """
    x_tr, y_tr = make_fn(n=n_tr, seed=seed)
    x_te, y_te = make_fn(n=n_te, seed=seed + 500)
    model = vemp.train_aggregate_sparsity(x_tr, y_tr, k_max=hump.K_MAX, alpha0=hump.ALPHA0, n_epochs=n_epochs, lr=1e-2, seed=seed)
    pure = p2.train_cond_gaussian(x_tr, y_tr, seed=seed)

    delta = p2.pure_nll_per_point(pure, x_te, y_te) - p3.bucket_nll_per_point(model, x_te, y_te)
    eff = p3.effective_count(model.weights(p2._to_xy(x_te, y_te)[0]).cpu().numpy())
    x_flat = np.asarray(x_te, dtype=np.float64).ravel()

    grid = np.linspace(0.0, 1.0, n_grid).astype(np.float32)
    gold = np.empty(n_grid)
    for j, xg in enumerate(grid):
        xs = np.full(m_gold, xg, dtype=np.float32)
        ys = given_fn(xs, sigma=hump.SIGMA, sep_min=hump.SEP_MIN, sep_max=hump.SEP_MAX, seed=seed * 100_003 + j)
        gold[j] = float(
            (p2.pure_nll_per_point(pure, xs.reshape(-1, 1), ys) - p3.bucket_nll_per_point(model, xs.reshape(-1, 1), ys)).mean()
        )

    return {"x": x_flat, "delta": delta, "eff": eff, "grid": grid.astype(np.float64), "gold": gold}


def tercile_verdict(x: np.ndarray, delta: np.ndarray) -> dict[str, object]:
    """Reduces a per-point Δ̂(x) array to the edge/middle/edge tercile means and the recovery read.

    Recovery (PREREGISTRATION.md X4-recovery, read per seed): the middle tercile mean is
    significantly positive by its OWN two-sided band (``mean - 2*SE > 0``, SE from the per-point
    values within that tercile) AND exceeds both tail tercile means.

    Args:
        x: per-point test inputs, shape ``(N,)``.
        delta: per-point held-out NLL advantage of the mixture, shape ``(N,)``.

    Returns:
        Dict with per-tercile mean/SE/n and the ``recovered`` bool.
    """
    stats = _tercile_means_se(x, delta)
    mid = stats["middle"]
    sig_positive = (mid["mean"] - 2.0 * mid["se"]) > 0.0 if not np.isnan(mid["se"]) else mid["mean"] > 0.0
    above_both_tails = mid["mean"] > stats["edge_lo"]["mean"] and mid["mean"] > stats["edge_hi"]["mean"]
    return {
        "tercile": {k: {"mean": v["mean"], "se": v["se"], "n": v["n"]} for k, v in stats.items()},
        "middle_significantly_positive": bool(sig_positive),
        "middle_above_both_tails": bool(above_both_tails),
        "recovered": bool(sig_positive and above_both_tails),
    }


# ---------------------------------------------------------------------------
# Selftest -- the gold-standard Δ*(x) as a known-answer oracle (independent of arbiter recovery).
# ---------------------------------------------------------------------------


def run_selftest() -> bool:
    """Known-answer oracle: the gold-standard Δ*(x) humps on E and stays flat/≤0 on E_broad.

    This does NOT depend on whether the fitted one-sample arbiter recovers the hump (that is the
    X4-recovery bar, checked by the real 3-seed reader) -- only on the generator's ground truth,
    read through a single small/fast fit (small N, few epochs) so the check is cheap. Uses a
    single seed and reduced sizes; not the real read.
    """
    n_tr, n_te, n_epochs, m_gold, n_grid = 500, 600, 300, 600, 24
    seed = 0

    print(f"[x4 selftest] fitting toy E   (n_tr={n_tr} n_te={n_te} epochs={n_epochs})...")
    make_e, given_e = TOY_SPECS["E"]
    res_e = fit_and_score_seed(make_e, given_e, seed, n_tr, n_te, n_epochs, m_gold, n_grid)
    gold_e = _tercile_means(res_e["grid"], res_e["gold"])

    print(f"[x4 selftest] fitting toy E_broad (n_tr={n_tr} n_te={n_te} epochs={n_epochs})...")
    make_b, given_b = TOY_SPECS["E_broad"]
    res_b = fit_and_score_seed(make_b, given_b, seed, n_tr, n_te, n_epochs, m_gold, n_grid)
    gold_b = _tercile_means(res_b["grid"], res_b["gold"])

    print(f"[x4 selftest] gold Δ*  E:       edge_lo={gold_e['edge_lo']:+.3f}  middle={gold_e['middle']:+.3f}  edge_hi={gold_e['edge_hi']:+.3f}")
    print(f"[x4 selftest] gold Δ*  E_broad: edge_lo={gold_b['edge_lo']:+.3f}  middle={gold_b['middle']:+.3f}  edge_hi={gold_b['edge_hi']:+.3f}")

    ok_e = gold_e["middle"] > 0.0 and gold_e["middle"] > gold_e["edge_lo"] and gold_e["middle"] > gold_e["edge_hi"]
    ok_broad = gold_b["middle"] <= 0.05  # "flat/<=0" with a small numerical-noise tolerance
    ok = ok_e and ok_broad
    print(f"[x4 selftest] E humps={ok_e}  E_broad flat/<=0={ok_broad}  {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Real reader -- 3-seed non-nested arbiter on the K4-identical E / E_broad data.
# ---------------------------------------------------------------------------


def load_k4_nested_e_contrast() -> list[dict[str, object]] | None:
    """Best-effort load of K4's nested-E per-seed knee/arbiter read, for the side-by-side contrast.

    Returns:
        List of per-seed dicts (``seed``, ``knee_r_star``, ``b_coh_any_worse``) if the summary
        exists and has the expected layout; a single-element list carrying an ``error`` string if
        the layout differs; ``None`` if the file does not exist yet.
    """
    if not os.path.exists(K4_SUMMARY_PATH):
        return None
    with open(K4_SUMMARY_PATH) as f:
        k4 = json.load(f)
    try:
        rows = k4["structured"]["E"]
        return [{"seed": r["seed"], "knee_r_star": r["knee_r_star"], "b_coh_any_worse": r["b_coh_any_worse"]} for r in rows]
    except (KeyError, TypeError) as exc:
        return [{"error": f"unexpected K4 summary layout: {exc}"}]


def run_toy(toy: str, seeds: tuple[int, ...], n_tr: int, n_te: int, n_epochs: int, m_gold: int, n_grid: int) -> list[dict[str, object]]:
    """Runs `fit_and_score_seed` + `tercile_verdict` for one toy across all seeds.

    Args:
        toy: key into `TOY_SPECS` (``"E"`` or ``"E_broad"``).
        seeds: seeds to run (K4-matched: ``(0, 1, 2)``).
        n_tr: training set size.
        n_te: held-out test set size.
        n_epochs: training epochs.
        m_gold: gold-standard resample draws per grid point.
        n_grid: gold-standard grid resolution.

    Returns:
        One dict per seed: seed id, Δ̂(x) tercile verdict, gold Δ*(x) tercile means, and eff#(x)
        tercile means.
    """
    make_fn, given_fn = TOY_SPECS[toy]
    rows = []
    for seed in seeds:
        t0 = time.time()
        res = fit_and_score_seed(make_fn, given_fn, seed, n_tr, n_te, n_epochs, m_gold, n_grid)
        verdict = tercile_verdict(res["x"], res["delta"])
        gold_tercile = _tercile_means(res["grid"], res["gold"])
        eff_tercile = _tercile_means(res["x"], res["eff"])
        wall = time.time() - t0
        print(
            f"[x4] {toy} s{seed}: Δ̂ tercile edge/mid/edge = "
            f"{verdict['tercile']['edge_lo']['mean']:+.3f}/{verdict['tercile']['middle']['mean']:+.3f}/{verdict['tercile']['edge_hi']['mean']:+.3f}"
            f"  recovered={verdict['recovered']}  ({wall:.0f}s)"
        )
        rows.append({"toy": toy, "seed": seed, **verdict, "gold_tercile": gold_tercile, "eff_tercile": eff_tercile, "wall_time_sec": round(wall, 1)})
    return rows


def _run(smoke: bool) -> None:
    """Runs the X4 battery (real read, or a 1-seed/reduced-size smoke check) and writes the summary."""
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()

    if smoke:
        seeds, n_tr, n_te, n_epochs, m_gold, n_grid = (0,), 300, 400, 100, 400, 24
    else:
        seeds, n_tr, n_te, n_epochs, m_gold, n_grid = SEEDS, N_TR, N_TE, N_EPOCHS, M_GOLD, N_GRID
    print(f"[x4] smoke={smoke} seeds={seeds} n_tr={n_tr} n_te={n_te} epochs={n_epochs} m_gold={m_gold} n_grid={n_grid}")

    e_rows = run_toy("E", seeds, n_tr, n_te, n_epochs, m_gold, n_grid)
    broad_rows = run_toy("E_broad", seeds, n_tr, n_te, n_epochs, m_gold, n_grid)

    n_seeds_recovered = int(sum(r["recovered"] for r in e_rows))
    broad_stays_flat = bool(all(not r["recovered"] for r in broad_rows))

    e_mid_means = [r["tercile"]["middle"]["mean"] for r in e_rows]
    e_lo_means = [r["tercile"]["edge_lo"]["mean"] for r in e_rows]
    e_hi_means = [r["tercile"]["edge_hi"]["mean"] for r in e_rows]
    consistency = {
        "mean_across_seeds": {
            "edge_lo": float(np.mean(e_lo_means)),
            "middle": float(np.mean(e_mid_means)),
            "edge_hi": float(np.mean(e_hi_means)),
        },
        "june_single_fit_read": JUNE_SINGLE_FIT_READ,
    }

    k4_contrast = load_k4_nested_e_contrast()

    verdict = {
        "n_seeds_recovered": n_seeds_recovered,
        "w8_confirmed": bool(n_seeds_recovered == len(seeds)),
        "broad_stays_flat": broad_stays_flat,
    }

    summary = {
        "task": "X4",
        "config": {
            "n_tr": n_tr,
            "n_te": n_te,
            "n_epochs": n_epochs,
            "m_gold": m_gold,
            "n_grid": n_grid,
            "seeds": list(seeds),
            "k_max": hump.K_MAX,
            "alpha0": hump.ALPHA0,
            "sigma": hump.SIGMA,
            "sep": [hump.SEP_MIN, hump.SEP_MAX],
            "smoke": smoke,
        },
        "E": e_rows,
        "E_broad": broad_rows,
        "n_seeds_recovered": n_seeds_recovered,
        "broad_stays_flat": broad_stays_flat,
        "consistency_vs_june_single_fit": consistency,
        "k4_nested_e_contrast": k4_contrast,
        "verdict": verdict,
        "wall_time_sec": round(time.time() - t_start, 1),
    }

    name = "x4_summary_smoke.json" if smoke else "x4_summary.json"
    out_path = os.path.join(OUT_DIR, name)
    with open(out_path, "w") as f:
        json.dump(_jsonable(summary), f, indent=2)
    print(f"\n[x4] wrote {out_path}  ({summary['wall_time_sec']:.0f}s)")
    print("[x4] VERDICT:", json.dumps(_jsonable(verdict), indent=2))


def main() -> None:
    """Parses args and dispatches to the selftest, the smoke run, or the real 3-seed reader."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="Gold-standard known-answer oracle check (fast); exits with a status code.")
    parser.add_argument("--smoke", action="store_true", help="1 seed, reduced N/epochs -- proves the pipeline runs end-to-end; not the real read.")
    args = parser.parse_args()

    torch.set_num_threads(TORCH_THREADS)
    if args.selftest:
        sys.exit(0 if run_selftest() else 1)

    _run(smoke=args.smoke)


if __name__ == "__main__":
    main()
