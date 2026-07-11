"""WS3/V3 -- the variance-CAPACITY ladder (unification): the SAME knee reader that selects mixture-component count k also selects sigma-model capacity v.

Capacity-ladder program, `docs/plans/capacity_ladder_2026-07-09/EXECUTION_PLAN.md` sec 4, AS
AMENDED by the R4 adjudication (`capacity_ladder_results/R4_verdict.md` sec 4, "V3 spec check").
V1 ranked GLOBAL variance-estimation mechanisms; V2 ranked the heteroscedastic sigma(x) FIX
BATTERY and found the disease (in-sample variance collapse) is finite-sample and the
cross-fitted-residual fix is noise-limited below N~1000. V3 reframes "which fix wins" as "how
much sigma-CAPACITY does the data support", scored by the same post-hoc machinery WS1 uses for
mixture-component count:

Four sigma-model rungs (increasing capacity), all mapping x -> log-sigma^2(x):
  v0: global scalar log-sigma^2 (constant) -- the closed-form Gaussian MLE of the residual table.
  v1: per-tercile log-sigma^2 -- 3 piecewise-constant closed-form MLEs on quantile bins of x.
  v2: linear-in-x log-sigma^2 = a + b*x -- `capacity_ladder_v2.SigmaHead` with NO hidden layers
      (a single `Linear(d, 1)`), fit by the SAME Gaussian-NLL machinery as v3.
  v3: small MLP log-sigma^2 head (hidden (16, 16)) -- `capacity_ladder_v2.SigmaHead` at full
      capacity, the arm (d) sigma head unchanged.

Honest-mean / cross-fit construction (R4 sec 4, refinement 1 -- the load-bearing one): V2 arm
(d) deployed an OVERFIT full-data mean (`v1.train_mean_mlp`, fixed epochs, no early stop) for
held-out scoring, which V0 proved dominates held-out NLL when the mean overfits -- so a V3 knee
built on that scoring mean would read mean-overfit, not sigma-capacity. V3 instead: (i) fits ONE
EARLY-STOPPED mean (`capacity_ladder_v2.train_mean_mlp_earlystop`, held-out val split), FROZEN,
and deploys it -- unchanged across all four rungs -- for every held-out scoring call; (ii) fits
all four sigma rungs on K=5 cross-fitted (out-of-fold) residuals of an EARLY-STOPPED mean per
fold (mirroring `capacity_ladder_variance_v1.cross_fitted_sigma2`'s fold structure, but with
`train_mean_mlp_earlystop` in place of arm (d)'s fixed-epoch `v1.mlp_fit_fn`, so a non-converged
fold mean can't inflate a fold's residuals -- V1's own honest-residual convergence caveat).

Toys: V-toy1 (`_toy_datasets.make_v_toy1`, heteroscedastic, known smooth
sigma(x) = 0.1 + 0.3*sigmoid(4x)) and its HOMOSCEDASTIC twin (`_toy_datasets.make_v_toy1h`,
identical mean, constant sigma). Per R4 refinement 3, the 5-D V-toy2 generalization
(`make_v_toy2`) does not exist in this repo and is NOT run here -- a registered gap, not a
finding. Per R4 refinement 2, only N >= 1000 is run (cross-fit rung fitting is noise-limited at
N=200, per V2). Per R4 refinement 4, beta-NLL is NOT a rung (it is a training-loss axis, not a
sigma-capacity class) -- it is simply absent from this ladder.

A held-out per-example log-likelihood table `score[i, v]` (columns v0..v3, SAME deployed mean,
rung-specific sigma) is built per (toy, N, seed) unit and read by
`_capacity_ladder.knee(score, ref_c=1, block=None, ...)` -- the SAME, unmodified reader WS1 uses
for k -- which is the unification claim under test.

No touching `automl_package/models/` -- pure diagnostic plumbing, reusing V1's cross-fit
machinery, V2's early-stopped mean / sigma-head trainer, and K0's knee reader.

Usage:
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/capacity_ladder_v3.py --selftest
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/capacity_ladder_v3.py --smoke
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/capacity_ladder_v3.py
"""

from __future__ import annotations

import argparse
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
import _capacity_ladder as cl  # noqa: E402
import _toy_datasets as td  # noqa: E402
import capacity_ladder_v2 as v2  # noqa: E402
import capacity_ladder_variance_v1 as v1  # noqa: E402

from automl_package.utils.pytorch_utils import get_device  # noqa: E402

RESULTS_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "V3")

RUNG_NAMES = ["v0", "v1", "v2", "v3"]
N_BINS_TERCILE = 3

SEEDS = [0, 1, 2]
N_GRID = [1000]  # R4 refinement 2: N>=1000 only; N=2000 can be added if a second N is wanted
N_EVAL = v2.N_EVAL  # 2000, fresh held-out pool per (N, seed) unit
EVAL_SEED_OFFSET = v2.EVAL_SEED_OFFSET
K_FOLDS = v1.K_FOLDS
NN_HIDDEN = v1.NN_HIDDEN  # (16, 16) -- the mean net AND the v3 (full-capacity) sigma head
TORCH_THREADS = 2

SELFTEST_N = 1000
# The sigma rungs are gradient-fit (v2 linear head, v3 MLP head) via full-batch Adam on a
# Gaussian-NLL loss that is slow to converge from a random init (measured: at 300-800 epochs the
# 2-parameter v2 head is still under-converged and scores WORSE than the closed-form v1 tercile
# rung, which would make the selftest's own synthetic ground truth unrecoverable) -- so the
# selftest reuses the full production hidden size/epoch budget, just at N=1000 with 2 units.
SELFTEST_HIDDEN = NN_HIDDEN
SELFTEST_EPOCHS = v1.NN_EPOCHS

FULL_CONFIG = {"seeds": SEEDS, "n_grid": N_GRID, "n_eval": N_EVAL, "epochs": v1.NN_EPOCHS, "lr": v1.NN_LR}
SMOKE_CONFIG = {"seeds": [0], "n_grid": [200], "n_eval": 200, "epochs": 60, "lr": v1.NN_LR}

SigmaFn = Callable[[np.ndarray], np.ndarray]


def _v_toy1h_sigma_true(x: np.ndarray) -> np.ndarray:
    """Ground-truth sigma(x) for the homoscedastic twin -- constant `V_TOY1H_SIGMA`."""
    return np.full(np.asarray(x).shape[0], td.V_TOY1H_SIGMA, dtype=np.float64)


TOYS: dict[str, tuple[Callable[..., tuple[np.ndarray, np.ndarray]], Callable[[np.ndarray], np.ndarray]]] = {
    "v_toy1": (td.make_v_toy1, td.v_toy1_sigma),
    "v_toy1h": (td.make_v_toy1h, _v_toy1h_sigma_true),
}


# --------------------------------------------------------------------------------------
# Honest-mean / cross-fit construction (R4 refinement 1)
# --------------------------------------------------------------------------------------


def _earlystop_fit_fn(hidden_sizes: tuple[int, ...], lr: float, n_epochs: int, seed: int, device: torch.device) -> v1.FitFn:
    """Wraps `train_mean_mlp_earlystop` into the `fit_fn(x, y) -> pred_fn(x)` contract.

    `cross_fitted_residuals` expects this contract; every fold's mean is EARLY-STOPPED, not arm
    (d)'s fixed-epoch `v1.mlp_fit_fn` refit, so a non-converged fold mean can't leak into the
    honest-residual table the sigma rungs are fit on.
    """

    def _fit(x_train: np.ndarray, y_train: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        model, *_ = v2.train_mean_mlp_earlystop(
            x_train, y_train, v2.EARLYSTOP_VAL_FRAC, hidden_sizes, lr, n_epochs, v2.EARLYSTOP_CHECK_EVERY, v2.EARLYSTOP_PATIENCE_CHECKS, seed, device
        )
        return v2._numpy_predict_fn(model, device, column=None)

    return _fit


# --------------------------------------------------------------------------------------
# Sigma rungs -- all fit on the SAME (x_oof, resid_oof) cross-fitted-residual table
# --------------------------------------------------------------------------------------


def _fit_rung_v0(resid_oof: np.ndarray) -> SigmaFn:
    """v0: global scalar log-sigma^2 -- the closed-form Gaussian MLE, mean(resid_oof^2)."""
    sigma = float(np.sqrt(np.mean(resid_oof.astype(np.float64) ** 2)))
    return lambda x: np.full(np.asarray(x).shape[0], sigma, dtype=np.float64)


def _tercile_bin_fn(x_fit: np.ndarray, n_bins: int = N_BINS_TERCILE) -> Callable[[np.ndarray], np.ndarray]:
    """Fits quantile-bin edges on `x_fit`, returning a callable that bins any x by them.

    Uses `_capacity_ladder.quantile_bins`'s exact edge formula (equal-mass bins plus tiny
    outward padding on the extreme edges). `quantile_bins` itself always re-derives edges from
    whatever `x` it is given (no separate fit/query split, unlike `perinput_curve`), so it cannot
    be called once on `x_oof` and reused on `eval_x` directly -- this wraps it to support exactly
    that split while keeping the identical edge convention.
    """
    x_arr = np.asarray(x_fit, dtype=np.float64).ravel()
    edges = np.quantile(x_arr, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9

    def _assign(x: np.ndarray) -> np.ndarray:
        xr = np.asarray(x, dtype=np.float64).ravel()
        return np.clip(np.digitize(xr, edges) - 1, 0, n_bins - 1)

    return _assign


def _fit_rung_v1(x_oof: np.ndarray, resid_oof: np.ndarray, n_bins: int = N_BINS_TERCILE) -> SigmaFn:
    """v1: per-tercile log-sigma^2, the closed-form Gaussian MLE within each quantile bin.

    Bins are fit on `x_oof` (`n_bins` quantile bins) and applied to any x via the SAME edges
    (see `_tercile_bin_fn`).
    """
    bin_fn = _tercile_bin_fn(x_oof, n_bins)
    bins_oof = bin_fn(x_oof)
    resid2 = resid_oof.astype(np.float64) ** 2
    sigma_per_bin = np.array([np.sqrt(max(float(np.mean(resid2[bins_oof == b])), 1e-12)) for b in range(n_bins)])
    return lambda x: sigma_per_bin[bin_fn(x)]


def _fit_rung_v2(x_oof: np.ndarray, resid_oof: np.ndarray, lr: float, n_epochs: int, seed: int, device: torch.device) -> SigmaFn:
    """v2: linear-in-x log-sigma^2, a `SigmaHead` with NO hidden layers (a single `Linear(d, 1)`).

    Fit by the SAME `train_sigma_head` Gaussian-NLL machinery as v3, just with less capacity.
    """
    model = v2.train_sigma_head(x_oof, resid_oof, hidden_sizes=(), lr=lr, n_epochs=n_epochs, seed=seed, device=device)
    return v2._numpy_sigma_fn(v2._numpy_predict_fn(model, device, column=None))


def _fit_rung_v3(x_oof: np.ndarray, resid_oof: np.ndarray, hidden_sizes: tuple[int, ...], lr: float, n_epochs: int, seed: int, device: torch.device) -> SigmaFn:
    """v3: small MLP log-sigma^2 head -- `train_sigma_head` at full capacity (default hidden (16, 16))."""
    model = v2.train_sigma_head(x_oof, resid_oof, hidden_sizes=hidden_sizes, lr=lr, n_epochs=n_epochs, seed=seed, device=device)
    return v2._numpy_sigma_fn(v2._numpy_predict_fn(model, device, column=None))


# --------------------------------------------------------------------------------------
# Scoring, knee read, per-unit runner
# --------------------------------------------------------------------------------------


def _score_rung(mean_fn: v2.PredFn, sigma_fn: SigmaFn, eval_x: np.ndarray, eval_y: np.ndarray, sigma_true_fn: Callable[[np.ndarray], np.ndarray]) -> dict:
    """Scores one rung's frozen (mean, sigma) pair on held-out data.

    Returns the per-example log density (the knee's score-table column) plus the summary
    NLL/SSR/sigma-ratio-error diagnostics against the KNOWN sigma(x).
    """
    mu = mean_fn(eval_x)
    sigma = np.maximum(sigma_fn(eval_x), 1e-6)
    sigma_true = sigma_true_fn(eval_x)
    y = eval_y.astype(np.float64).ravel()
    log_density = -0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(sigma) + ((y - mu) / sigma) ** 2)
    return {
        "log_density": log_density,
        "sigma_ratio_error": float(np.mean(np.abs(sigma / sigma_true - 1.0))),
        "nll": float(-np.mean(log_density)),
        "ssr": float(np.mean(((y - mu) / sigma) ** 2)),
    }


def _rung_label(r_star: int) -> str:
    """Maps a `knee` `r_star` (grid values 1..4, or the G2 ABSTAIN sentinel 0) to a rung name.

    `r_star == 0` (ABSTAIN) means growth beyond the reference capacity (`ref_c=1`, i.e. v0) was
    never confirmed as significant; since v0 is already the smallest rung in this ladder, that
    reads operationally as "stays at v0" and is reported as such -- but flagged explicitly as
    the ABSTAIN case, not silently folded into "v0" as if growth had been tested and failed at
    v0->v1 specifically (per `_capacity_ladder.knee`'s own G2 warning against that conflation).
    """
    if r_star == 0:
        return "v0 (ABSTAIN -- no significant growth beyond v0 confirmed)"
    return RUNG_NAMES[r_star - 1]


def run_unit(
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
    """Runs the full V3 pipeline for one (toy, N, seed) unit -- a single training draw.

    Honest mean, cross-fitted residuals, all four sigma rungs, held-out scoring, and the knee.
    """
    t0 = time.time()

    # Deployed mean: EARLY-STOPPED, fit ONCE on the full training pool, FROZEN -- reused
    # unchanged as the mean for every rung's held-out scoring (R4 refinement 1).
    mean_model, *_ = v2.train_mean_mlp_earlystop(x, y, v2.EARLYSTOP_VAL_FRAC, hidden_sizes, lr, epochs, v2.EARLYSTOP_CHECK_EVERY, v2.EARLYSTOP_PATIENCE_CHECKS, seed, device)
    mean_fn = v2._numpy_predict_fn(mean_model, device, column=None)

    # Cross-fitted (honest, out-of-fold) residuals of an EARLY-STOPPED mean per fold -- same
    # K-fold structure as v1.cross_fitted_sigma2 / V2 arm (d).
    fold_fit_fn = _earlystop_fit_fn(hidden_sizes, lr, epochs, seed, device)
    x_oof, resid_oof = v2.cross_fitted_residuals(x, y, fold_fit_fn, k=k_folds, seed=seed)

    sigma_fns: dict[str, SigmaFn] = {
        "v0": _fit_rung_v0(resid_oof),
        "v1": _fit_rung_v1(x_oof, resid_oof),
        "v2": _fit_rung_v2(x_oof, resid_oof, lr, epochs, seed, device),
        "v3": _fit_rung_v3(x_oof, resid_oof, hidden_sizes, lr, epochs, seed, device),
    }

    rung_metrics = {name: _score_rung(mean_fn, sigma_fn, eval_x, eval_y, sigma_true_fn) for name, sigma_fn in sigma_fns.items()}
    score_mat = np.stack([rung_metrics[name]["log_density"] for name in RUNG_NAMES], axis=1)

    r_star, delta_curve, se = cl.knee(score_mat, ref_c=1, block=None, seed=seed)
    wall_time = time.time() - t0

    return {
        "wall_time_sec": wall_time,
        "r_star": r_star,
        "rung_selected": _rung_label(r_star),
        "delta_curve": {RUNG_NAMES[k - 1]: v for k, v in delta_curve.items()},
        "se": {RUNG_NAMES[k - 1]: v for k, v in se.items()},
        "rungs": {name: {k: v for k, v in m.items() if k != "log_density"} for name, m in rung_metrics.items()},
    }


# --------------------------------------------------------------------------------------
# Selftest -- synthetic known-answer checks (must PASS before any real read)
# --------------------------------------------------------------------------------------


def _make_constant_sigma_toy(n: int, seed: int, sigma0: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic known-answer toy: sin(2*pi*x) mean, CONSTANT noise `sigma0` -- selftest check (a)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n).reshape(-1, 1).astype(np.float32)
    y = (np.sin(2 * np.pi * x.ravel()) + sigma0 * rng.normal(0.0, 1.0, n)).astype(np.float32)
    return x, y


def _make_linear_logsigma_toy(n: int, seed: int, a: float = -2.0, b: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic known-answer toy: sin(2*pi*x) mean, log-sigma(x) = a + b*x -- selftest check (b)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n).reshape(-1, 1).astype(np.float32)
    sigma_x = np.exp(a + b * x.ravel())
    y = (np.sin(2 * np.pi * x.ravel()) + sigma_x * rng.normal(0.0, 1.0, n)).astype(np.float32)
    return x, y


def _linear_logsigma_true(a: float, b: float) -> Callable[[np.ndarray], np.ndarray]:
    """Ground-truth sigma(x) = exp(a + b*x) for `_make_linear_logsigma_toy`."""

    def _sigma_true(x: np.ndarray) -> np.ndarray:
        return np.exp(a + b * np.asarray(x, dtype=np.float64).ravel())

    return _sigma_true


def selftest() -> None:
    """Synthetic in-memory known-answer checks, N~1000, no disk.

    (a) constant-sigma data: the knee must select v0 (`r_star` in `{0 (ABSTAIN), 1}`) -- no
        false structure. (b) linear-log-sigma data: the knee must select a rung >= v2
        (`r_star >= 3`) -- it must NOT abstain to v0 when real sigma(x) structure exists.
    """
    device = get_device()
    print(f"[selftest] device={device}")

    x_c, y_c = _make_constant_sigma_toy(SELFTEST_N, seed=0)
    eval_x_c, eval_y_c = _make_constant_sigma_toy(SELFTEST_N, seed=1)
    unit_c = run_unit(x_c, y_c, eval_x_c, eval_y_c, _v_toy1h_sigma_true, SELFTEST_HIDDEN, v1.NN_LR, SELFTEST_EPOCHS, K_FOLDS, seed=0, device=device)
    print(f"[selftest] (a) constant-sigma: r_star={unit_c['r_star']} rung={unit_c['rung_selected']} delta_curve={ {k: round(v, 4) for k, v in unit_c['delta_curve'].items()} }")
    assert unit_c["r_star"] in (0, 1), f"constant-sigma data must not grow past v0, got r_star={unit_c['r_star']} ({unit_c['rung_selected']})"

    x_l, y_l = _make_linear_logsigma_toy(SELFTEST_N, seed=0)
    eval_x_l, eval_y_l = _make_linear_logsigma_toy(SELFTEST_N, seed=1)
    sigma_true_l = _linear_logsigma_true(-2.0, 2.0)
    unit_l = run_unit(x_l, y_l, eval_x_l, eval_y_l, sigma_true_l, SELFTEST_HIDDEN, v1.NN_LR, SELFTEST_EPOCHS, K_FOLDS, seed=0, device=device)
    print(f"[selftest] (b) linear-log-sigma: r_star={unit_l['r_star']} rung={unit_l['rung_selected']} delta_curve={ {k: round(v, 4) for k, v in unit_l['delta_curve'].items()} }")
    assert unit_l["r_star"] >= 3, f"linear-log-sigma data must select >= v2, got r_star={unit_l['r_star']} ({unit_l['rung_selected']})"

    print("[selftest] PASS")


# --------------------------------------------------------------------------------------
# Summary checks, main
# --------------------------------------------------------------------------------------


def compute_summary_checks(all_results: dict) -> dict:
    """Scores the pre-registered V3 expectations (R4 sec 4) against the collected units.

    Homoscedastic twin: knee should land at v0 on every seed. V-toy1: R4 predicts v2
    (linear-log-sigma), NOT v1 -- reported as met/not without fudging, alongside where the
    knee actually landed on every seed.
    """
    checks: dict = {}
    for toy_name in {u["toy"] for u in all_results["units"]}:
        units = [u for u in all_results["units"] if u["toy"] == toy_name]
        checks[toy_name] = {
            "rung_selected_per_seed": [u["rung_selected"] for u in units],
            "r_star_per_seed": [u["r_star"] for u in units],
        }
    if "v_toy1h" in checks:
        checks["v_toy1h_knee_is_v0"] = all(r in (0, 1) for r in checks["v_toy1h"]["r_star_per_seed"])
    if "v_toy1" in checks:
        checks["v_toy1_knee_matches_v2_prereg"] = all(r == 3 for r in checks["v_toy1"]["r_star_per_seed"])
    return checks


def main(smoke: bool = False, n_grid_override: list[int] | None = None) -> None:
    """Runs the full (or `--smoke`) V3 ladder over both toys and writes the summary JSON.

    `n_grid_override` (from `--n-grid`) replaces the config's `n_grid` for a confirmatory
    larger-N run (e.g. testing whether the held-out-NLL knee resolves v2 as N grows); the
    output then lands in a size-tagged file so it never clobbers the pre-registered N=1000 read.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = get_device()
    cfg = SMOKE_CONFIG if smoke else FULL_CONFIG
    if n_grid_override is not None:
        cfg = {**cfg, "n_grid": n_grid_override}
    print(f"[main] device={device} smoke={smoke} cfg={cfg}")

    all_results: dict = {"config": cfg, "units": []}
    t_start = time.time()
    for toy_name, (make_fn, sigma_true_fn) in TOYS.items():
        for n in cfg["n_grid"]:
            for seed in cfg["seeds"]:
                print(f"[main] toy={toy_name} N={n} seed={seed} ...")
                x, y = make_fn(n=n, seed=seed)
                eval_x, eval_y = make_fn(n=cfg["n_eval"], seed=seed + EVAL_SEED_OFFSET)
                unit = run_unit(x, y, eval_x, eval_y, sigma_true_fn, NN_HIDDEN, cfg["lr"], cfg["epochs"], K_FOLDS, seed, device)
                all_results["units"].append({"toy": toy_name, "n": n, "seed": seed, **unit})
                delta_r = {k: round(v, 4) for k, v in unit["delta_curve"].items()}
                print(f"  wall_time={unit['wall_time_sec']:.1f}s r_star={unit['r_star']} rung={unit['rung_selected']} delta_curve={delta_r}")

    all_results["summary_checks"] = compute_summary_checks(all_results)
    all_results["total_wall_time_sec"] = time.time() - t_start

    if smoke:
        summary_name = "v3_summary_smoke.json"
    elif n_grid_override is not None:
        summary_name = f"v3_summary_N{'_'.join(str(n) for n in n_grid_override)}.json"
    else:
        summary_name = "v3_summary.json"
    summary_path = os.path.join(RESULTS_DIR, summary_name)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[main] wrote {summary_path}")
    print(f"[main] total wall time: {all_results['total_wall_time_sec']:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="1 seed, N=200, tiny epochs -- proves the pipeline runs end-to-end; not for the real read.")
    parser.add_argument("--n-grid", type=int, nargs="+", default=None, help="Override N grid (e.g. --n-grid 4000) for a confirmatory larger-N run; writes a size-tagged summary.")
    args = parser.parse_args()

    torch.set_num_threads(TORCH_THREADS)

    if args.selftest:
        selftest()
    else:
        main(smoke=args.smoke, n_grid_override=args.n_grid)
