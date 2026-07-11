"""X2: CRPS-knee arm on the V3 sigma-capacity ladder (WS3 capacity-ladder follow-up, task X2).

(docs/plans/capacity_ladder_2026-07-09/EXECUTION_PLAN.md §8.5 X2; pre-registration in
 capacity_ladder_results/X2/PREREGISTRATION.md)

The V3 held-out-NLL knee UNDER-RESOLVES the sigma-capacity ladder (v0 global -> v1 tercile ->
v2 linear-in-x log-sigma^2 -> v3 MLP): on the heteroscedastic toy it stops at v1 even though the
sigma-ratio-error truth says v2 (N=1000) / v3 (N=4000) is the best sigma-model. §8.3: the log
score's sensitivity to a relative sigma error delta is SECOND-order (~E[delta^2] nat), so the
v1->v2 NLL increment (~0.003 nat, Sivula-small) is below the knee's 2*SE bar. X2 asks whether the
CRPS -- a strictly proper but NON-local, shape-sensitive score (Gneiting & Raftery 2007) -- read
by the SAME `_capacity_ladder.knee`, resolves v1->v2 where NLL cannot, while still abstaining on
the homoscedastic twin.

Construction reuses `capacity_ladder_v3` VERBATIM (same early-stopped frozen mean, K=5 cross-
fitted residuals, same four sigma rungs, same eval pool). The ONLY change is the score: alongside
V3's per-example Gaussian log-density (`v3._score_rung`, reused so the NLL column is bit-identical
-> the NLL-knee reproduces V3 exactly = the consistency guard), X2 also builds a CRPS table from
the same (mean, sigma) pair and reads `cl.knee(-crps_mat, ...)` (CRPS is negatively oriented, so
negate so "higher = better" holds for the shared reader).

    CRPS(N(mu, sigma), y) = sigma * [ z*(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ],  z = (y-mu)/sigma
    (Gneiting & Raftery 2007; Phi/phi the standard normal cdf/pdf).

No model training beyond V3's own (pure diagnostic plumbing). No touching `automl_package/models/`.

Usage:
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/capacity_ladder_x2.py --selftest
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/capacity_ladder_x2.py --smoke
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/capacity_ladder_x2.py
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/capacity_ladder_x2.py --n-grid 4000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Callable

import numpy as np
import torch

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root, so `import automl_package` works

import _capacity_ladder as cl  # noqa: E402
import capacity_ladder_v2 as v2  # noqa: E402
import capacity_ladder_v3 as v3  # noqa: E402
import capacity_ladder_variance_v1 as v1  # noqa: E402

from automl_package.utils.pytorch_utils import get_device  # noqa: E402

RESULTS_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "X2")

_SQRT2 = math.sqrt(2.0)
_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


# --------------------------------------------------------------------------------------
# Closed-form Gaussian CRPS.
# --------------------------------------------------------------------------------------


def crps_gaussian(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-example closed-form CRPS of `N(mu, sigma)` against `y` (Gneiting & Raftery 2007).

    `CRPS = sigma * [ z*(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ]`, `z = (y - mu)/sigma`; Phi/phi
    the standard-normal cdf/pdf (Phi via `torch.erf`, no scipy dependency). Negatively oriented
    (lower is better). Returns a `(N,)` float64 array.
    """
    z = torch.as_tensor((np.asarray(y, np.float64) - np.asarray(mu, np.float64)) / np.asarray(sigma, np.float64))
    phi = torch.exp(-0.5 * z * z) * _INV_SQRT_2PI
    cdf = 0.5 * (1.0 + torch.erf(z / _SQRT2))
    s = torch.as_tensor(np.asarray(sigma, np.float64))
    crps = s * (z * (2.0 * cdf - 1.0) + 2.0 * phi - _INV_SQRT_PI)
    return crps.numpy()


def _crps_gaussian_mc(mu: float, sigma: float, y: float, n_samples: int, seed: int) -> float:
    """Monte-Carlo CRPS estimate: `E|X-y| - 0.5*E|X-X'|`, `X,X' ~ N(mu, sigma)` (selftest oracle)."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(mu, sigma, n_samples)
    x2 = rng.normal(mu, sigma, n_samples)
    return float(np.mean(np.abs(x1 - y)) - 0.5 * np.mean(np.abs(x1 - x2)))


# --------------------------------------------------------------------------------------
# Per-unit runner: reuse V3's fitting verbatim, add the CRPS score table + CRPS knee.
# --------------------------------------------------------------------------------------


def run_unit_crps(
    x: np.ndarray,
    y: np.ndarray,
    eval_x: np.ndarray,
    eval_y: np.ndarray,
    sigma_true_fn: Callable[[np.ndarray], np.ndarray],
    hidden_sizes: tuple[int, ...],
    lr: float,
    epochs: int,
    k_folds: int,
    seed: int,
    device: torch.device,
) -> dict:
    """Fits the V3 ladder (identical to `v3.run_unit`) and reads BOTH the NLL knee and the CRPS knee.

    The four sigma rungs, the frozen early-stopped mean, the cross-fitted residuals and the eval
    scoring are V3's, unchanged (the NLL column comes from `v3._score_rung`, so the NLL knee is
    bit-identical to V3's recorded read = the consistency guard). The CRPS column is built from the
    same (mean, sigma) pair per rung and read by `cl.knee(-crps_mat, ...)`.
    """
    t0 = time.time()

    # --- V3.run_unit fitting, replicated verbatim (same calls, same order, same seeds) ---
    mean_model, *_ = v2.train_mean_mlp_earlystop(
        x, y, v2.EARLYSTOP_VAL_FRAC, hidden_sizes, lr, epochs, v2.EARLYSTOP_CHECK_EVERY, v2.EARLYSTOP_PATIENCE_CHECKS, seed, device
    )
    mean_fn = v2._numpy_predict_fn(mean_model, device, column=None)
    fold_fit_fn = v3._earlystop_fit_fn(hidden_sizes, lr, epochs, seed, device)
    x_oof, resid_oof = v2.cross_fitted_residuals(x, y, fold_fit_fn, k=k_folds, seed=seed)
    sigma_fns = {
        "v0": v3._fit_rung_v0(resid_oof),
        "v1": v3._fit_rung_v1(x_oof, resid_oof),
        "v2": v3._fit_rung_v2(x_oof, resid_oof, lr, epochs, seed, device),
        "v3": v3._fit_rung_v3(x_oof, resid_oof, hidden_sizes, lr, epochs, seed, device),
    }

    # --- Score every rung: V3's exact NLL/summaries + the new CRPS, from the SAME (mu, sigma) ---
    y_eval = np.asarray(eval_y, np.float64).ravel()
    nll_cols, crps_cols, rung_summ = [], [], {}
    for name in v3.RUNG_NAMES:
        sigma_fn = sigma_fns[name]
        m = v3._score_rung(mean_fn, sigma_fn, eval_x, eval_y, sigma_true_fn)  # exact V3 log_density + diagnostics
        mu = mean_fn(eval_x)
        sigma = np.maximum(sigma_fn(eval_x), 1e-6)  # identical floor to `_score_rung`
        crps = crps_gaussian(mu, sigma, y_eval)
        nll_cols.append(m["log_density"])
        crps_cols.append(crps)
        rung_summ[name] = {
            "sigma_ratio_error": m["sigma_ratio_error"],
            "nll": m["nll"],
            "ssr": m["ssr"],
            "mean_crps": float(np.mean(crps)),
        }

    nll_mat = np.stack(nll_cols, axis=1)
    crps_mat = np.stack(crps_cols, axis=1)

    nll_r, nll_delta, _ = cl.knee(nll_mat, ref_c=1, block=None, seed=seed)
    # CRPS is negatively oriented -> negate so the shared "higher = better" knee reads CRPS reduction.
    crps_r, crps_delta, crps_se = cl.knee(-crps_mat, ref_c=1, block=None, seed=seed)

    return {
        "wall_time_sec": time.time() - t0,
        "nll_r_star": nll_r,
        "nll_rung_selected": v3._rung_label(nll_r),
        "crps_r_star": crps_r,
        "crps_rung_selected": v3._rung_label(crps_r),
        "crps_delta_curve": {v3.RUNG_NAMES[k - 1]: v for k, v in crps_delta.items()},
        "crps_se": {v3.RUNG_NAMES[k - 1]: v for k, v in crps_se.items()},
        "nll_delta_curve": {v3.RUNG_NAMES[k - 1]: v for k, v in nll_delta.items()},
        "rungs": rung_summ,
    }


# --------------------------------------------------------------------------------------
# Selftest.
# --------------------------------------------------------------------------------------


def selftest() -> bool:
    """Known-answer checks for the CRPS reader.

    (0) closed-form CRPS == MC CRPS; (a) constant-sigma -> CRPS knee abstains to v0; (b)
    linear-log-sigma -> CRPS knee resolves >= v2 (real sigma(x) structure detected).
    """
    ok = True

    # (0) CRPS formula vs Monte Carlo on a few (mu, sigma, y).
    print("[x2 selftest] (0) closed-form CRPS vs Monte-Carlo:")
    for mu, sigma, yv in [(0.0, 1.0, 0.0), (0.5, 2.0, 3.0), (-1.0, 0.3, -1.4), (2.0, 1.5, -0.5)]:
        cf = float(crps_gaussian(np.array([mu]), np.array([sigma]), np.array([yv]))[0])
        mc = _crps_gaussian_mc(mu, sigma, yv, n_samples=400_000, seed=0)
        close = abs(cf - mc) < 5e-3
        ok = ok and close
        print(f"    mu={mu:+.1f} sigma={sigma:.1f} y={yv:+.1f}: closed={cf:.4f} mc={mc:.4f} close={close}")

    device = get_device()
    print(f"[x2 selftest] device={device}")

    # (a) constant-sigma known-answer toy: CRPS knee must NOT grow past v0.
    x_c, y_c = v3._make_constant_sigma_toy(v3.SELFTEST_N, seed=0)
    ex_c, ey_c = v3._make_constant_sigma_toy(v3.SELFTEST_N, seed=1)
    uc = run_unit_crps(x_c, y_c, ex_c, ey_c, v3._v_toy1h_sigma_true, v3.SELFTEST_HIDDEN, v1.NN_LR, v3.SELFTEST_EPOCHS, v3.K_FOLDS, 0, device)
    print(f"[x2 selftest] (a) constant-sigma: CRPS r*={uc['crps_r_star']} ({uc['crps_rung_selected']}) | NLL r*={uc['nll_r_star']} ({uc['nll_rung_selected']})")
    ok_a = uc["crps_r_star"] in (0, 1)
    ok = ok and ok_a

    # (b) linear-log-sigma known-answer toy: CRPS knee must resolve real structure (>= v2).
    x_l, y_l = v3._make_linear_logsigma_toy(v3.SELFTEST_N, seed=0)
    ex_l, ey_l = v3._make_linear_logsigma_toy(v3.SELFTEST_N, seed=1)
    ul = run_unit_crps(x_l, y_l, ex_l, ey_l, v3._linear_logsigma_true(-2.0, 2.0), v3.SELFTEST_HIDDEN, v1.NN_LR, v3.SELFTEST_EPOCHS, v3.K_FOLDS, 0, device)
    print(f"[x2 selftest] (b) linear-log-sigma: CRPS r*={ul['crps_r_star']} ({ul['crps_rung_selected']}) | NLL r*={ul['nll_r_star']} ({ul['nll_rung_selected']})")
    ok_b = ul["crps_r_star"] >= 3
    ok = ok and ok_b

    print(f"[x2 selftest] formula={ok and True}  constant->v0={ok_a}  linear->=v2={ok_b}  {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------------------
# Main.
# --------------------------------------------------------------------------------------


def _consistency_note(unit: dict, v3_recorded: dict | None) -> str:
    """Flags whether the recomputed NLL knee matches V3's recorded r_star for this unit."""
    if v3_recorded is None:
        return "no-v3-baseline"
    return "MATCH" if unit["nll_r_star"] == v3_recorded else f"MISMATCH(recomputed={unit['nll_r_star']} vs v3={v3_recorded})"


def _load_v3_baseline(n_grid: list[int]) -> dict:
    """Loads V3's recorded r_star per (toy, N, seed) for the consistency guard, if the summary exists."""
    fn = "v3_summary.json" if n_grid == [1000] else f"v3_summary_N{'_'.join(str(n) for n in n_grid)}.json"
    path = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "V3", fn)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        d = json.load(f)
    return {(u["toy"], u["n"], u["seed"]): u["r_star"] for u in d.get("units", [])}


def main(smoke: bool = False, n_grid_override: list[int] | None = None) -> None:
    """Runs the CRPS-knee arm over both toys/seeds and writes the X2 summary JSON."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = get_device()
    cfg = v3.SMOKE_CONFIG if smoke else v3.FULL_CONFIG
    if n_grid_override is not None:
        cfg = {**cfg, "n_grid": n_grid_override}
    print(f"[x2] device={device} smoke={smoke} cfg={cfg}")

    v3_base = _load_v3_baseline(cfg["n_grid"]) if not smoke else {}
    results: dict = {"task": "X2", "config": cfg, "units": []}
    t_start = time.time()
    for toy_name, (make_fn, sigma_true_fn) in v3.TOYS.items():
        for n in cfg["n_grid"]:
            for seed in cfg["seeds"]:
                x, y = make_fn(n=n, seed=seed)
                eval_x, eval_y = make_fn(n=cfg["n_eval"], seed=seed + v3.EVAL_SEED_OFFSET)
                unit = run_unit_crps(x, y, eval_x, eval_y, sigma_true_fn, v3.NN_HIDDEN, cfg["lr"], cfg["epochs"], v3.K_FOLDS, seed, device)
                note = _consistency_note(unit, v3_base.get((toy_name, n, seed)))
                results["units"].append({"toy": toy_name, "n": n, "seed": seed, "nll_consistency": note, **unit})
                print(f"[x2] {toy_name} N={n} s{seed}: CRPS r*={unit['crps_r_star']} ({unit['crps_rung_selected']})  "
                      f"NLL r*={unit['nll_r_star']} ({unit['nll_rung_selected']})  guard={note}  ({unit['wall_time_sec']:.0f}s)")

    # Re-issued verdicts.
    hetero = [u for u in results["units"] if u["toy"] == "v_toy1"]
    homo = [u for u in results["units"] if u["toy"] == "v_toy1h"]

    def _resolves_toward_truth(u: dict) -> bool:
        """Full pre-registered resolution clause for one hetero unit.

        True iff CRPS reaches a rung strictly above the NLL knee AND >= v2 (r_star>=3) AND that
        rung's sigma-ratio-error is <= v1's (resolves toward the truth, not past it into noise).
        """
        r = u["crps_r_star"]
        if not (r > u["nll_r_star"] and r >= 3):
            return False
        selected = v3.RUNG_NAMES[r - 1]  # r_star 1..4 -> v0..v3
        return u["rungs"][selected]["sigma_ratio_error"] <= u["rungs"]["v1"]["sigma_ratio_error"]

    n_above_nll = int(sum(u["crps_r_star"] > u["nll_r_star"] and u["crps_r_star"] >= 3 for u in hetero))
    n_resolves = int(sum(_resolves_toward_truth(u) for u in hetero))
    verdicts = {
        "resolution": {
            "hetero_crps_r_star": [u["crps_r_star"] for u in hetero],
            "hetero_nll_r_star": [u["nll_r_star"] for u in hetero],
            "n_seeds_crps_above_nll": n_above_nll,
            "n_seeds_resolves_toward_truth": n_resolves,
            "resolves_v1_to_v2": bool(hetero and n_resolves >= 2),
        },
        "no_false_positive": {
            "homo_crps_r_star": [u["crps_r_star"] for u in homo],
            "holds": bool(homo and all(u["crps_r_star"] in (0, 1) for u in homo)),
        },
        "nll_consistency_all_match": bool(all(u["nll_consistency"] in ("MATCH", "no-v3-baseline") for u in results["units"])),
    }
    results["verdicts"] = verdicts
    results["total_wall_time_sec"] = time.time() - t_start

    if smoke:
        name = "x2_summary_smoke.json"
    elif n_grid_override is not None:
        name = f"x2_summary_N{'_'.join(str(n) for n in n_grid_override)}.json"
    else:
        name = "x2_summary.json"
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[x2] wrote {path}  ({results['total_wall_time_sec']:.0f}s)")
    print("[x2] VERDICTS:", json.dumps(verdicts, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="1 seed, N=200, tiny epochs -- proves the pipeline runs; not for the real read.")
    parser.add_argument("--n-grid", type=int, nargs="+", default=None, help="Override N grid (e.g. --n-grid 4000) for a confirmatory larger-N run; writes a size-tagged summary.")
    args = parser.parse_args()

    torch.set_num_threads(v3.TORCH_THREADS)
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    main(smoke=args.smoke, n_grid_override=args.n_grid)
