"""WS3/V2 -- heteroscedastic sigma(x)-estimation FIX BATTERY, ranked against known ground truth.

Capacity-ladder program, `docs/plans/capacity_ladder_2026-07-09/EXECUTION_PLAN.md` sec 4. V0
established the joint-NLL variance-collapse disease; V1 ranked the fix family for GLOBAL variance
on well-specified linear-Gaussian data. V2 moves to V-toy1 (`_toy_datasets.make_v_toy1`), the 1-D
HETEROSCEDASTIC toy with a KNOWN smooth sigma(x), and ranks six practical remedies for estimating
that sigma(x) itself -- not just a single global scalar.

Six arms, all scored on a shared fresh held-out set against the KNOWN sigma(x):
  (a) joint NLL       -- a (mean, log-variance) MLP trained end-to-end with `nll_loss` (the disease).
  (b) beta-NLL        -- same net, `beta_nll_loss` (beta=0.5), the labelled variance-collapse fix.
  (c) mean-first, IN-SAMPLE (diseased control) -- a mean MLP (early-stopped on a held-out val
      split), FROZEN, then a small sigma(x) head fit on the residuals of the points the mean model
      directly trained on -- biased low, same disease as V1's in-sample MLE, now with an input-
      dependent sigma head instead of a single scalar.
  (d) mean-first + CROSS-FITTED residuals (the fix) -- K=5 out-of-fold mean predictions (mirrors
      `capacity_ladder_variance_v1.cross_fitted_sigma2`'s fold structure) give every training point
      an honest residual it was never fit on; ONE sigma(x) head is trained on all of them.
  (e) per-bin scale recalibration on top of (c) -- a scalar multiplier per x-tercile, fit on half of
      (c)'s held-out val split and applied to the other half, then SWAPPED and averaged (the
      STACK-2b remedy) so the whole calibration pool is used without leaking into the sigma fit.
  (f) LocallyAdaptiveConformalWrapper around (c)'s (mean, sigma) pair -- INTERVAL metrics only
      (empirical coverage, mean width); it is not a density estimator, so no NLL/ratio score.

Per-arm metrics (a)-(e): sigma_ratio_error = mean_x |sigma_hat(x)/sigma_true(x) - 1| on the held-out
set, held-out Gaussian NLL, and SSR = mean((y-mu_hat)^2/sigma_hat^2) (should -> 1.0 when calibrated).

Pre-registered expectations (report whether met, do not fudge):
  - sigma_ratio_error ranks (d) <= (e) < (b) < (c) < (a).
  - (d) attains SSR in [0.9, 1.1] on 3/3 seeds at N=1000.

No touching `automl_package/models/` -- pure diagnostic plumbing, reusing V1's mean-MLP/cross-fit
infra and the library's own `nll_loss`/`beta_nll_loss`/`LocallyAdaptiveConformalWrapper`.

Usage:
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/capacity_ladder_v2.py --smoke
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/capacity_ladder_v2.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from functools import partial

import numpy as np
import torch
import torch.nn as nn

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root, so `import automl_package` works
import _toy_datasets as td
import capacity_ladder_variance_v1 as v1

from automl_package.models.conformal import LocallyAdaptiveConformalWrapper
from automl_package.utils.losses import beta_nll_loss, nll_loss
from automl_package.utils.pytorch_utils import get_device

RESULTS_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "V2")

SEEDS = [0, 1, 2]
N_GRID = [200, 1000]
N_EVAL = 2000  # shared held-out evaluation pool size, fresh draw per (N, seed) unit
EVAL_SEED_OFFSET = 10_000  # keeps the eval draw disjoint from the training draw for a given seed
K_FOLDS = v1.K_FOLDS
NN_HIDDEN = v1.NN_HIDDEN  # (16, 16) -- reused for the joint net, the mean net, AND the sigma head
EARLYSTOP_VAL_FRAC = 0.2  # fraction of arm (c)'s training pool held out for early stopping
EARLYSTOP_CHECK_EVERY = 10
EARLYSTOP_PATIENCE_CHECKS = 30  # 30 * EARLYSTOP_CHECK_EVERY = 300 epochs without improvement
ALPHA_CONFORMAL = 0.1
N_BINS_RECAL = 3  # terciles
TORCH_THREADS = 2

FULL_CONFIG = {"seeds": SEEDS, "n_grid": N_GRID, "n_eval": N_EVAL, "epochs": v1.NN_EPOCHS, "lr": v1.NN_LR}
SMOKE_CONFIG = {"seeds": [0], "n_grid": [200], "n_eval": 200, "epochs": 60, "lr": v1.NN_LR}

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
PredFn = Callable[[np.ndarray], np.ndarray]


# --------------------------------------------------------------------------------------
# Networks: joint (mean, log-variance) head, and a standalone sigma(x) head
# --------------------------------------------------------------------------------------


class JointMLP(nn.Module):
    """Joint (mean, log-variance) MLP: a shared trunk with a 2-wide output head -- the disease net."""

    def __init__(self, input_dim: int, hidden_sizes: tuple[int, ...] = NN_HIDDEN) -> None:
        """Builds the shared `Tanh` trunk and the 2-wide (mean, log-variance) output head."""
        super().__init__()
        layers: list[nn.Module] = []
        d = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(d, h), nn.Tanh()]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(d, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the (N, 2) [mean, log_variance] output."""
        return self.head(self.trunk(x))


class SigmaHead(nn.Module):
    """Small MLP mapping x -> log sigma^2, fit on precomputed (honest or in-sample) residuals."""

    def __init__(self, input_dim: int, hidden_sizes: tuple[int, ...] = NN_HIDDEN) -> None:
        """Builds the `Tanh` MLP mapping x to a single log-variance output."""
        super().__init__()
        layers: list[nn.Module] = []
        d = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(d, h), nn.Tanh()]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the (N,) log-variance output."""
        return self.net(x).squeeze(-1)


def train_joint_mlp(
    x: np.ndarray,
    y: np.ndarray,
    loss_fn: LossFn,
    hidden_sizes: tuple[int, ...] = NN_HIDDEN,
    lr: float = v1.NN_LR,
    n_epochs: int = v1.NN_EPOCHS,
    seed: int = 0,
    device: torch.device | None = None,
) -> JointMLP:
    """Trains a :class:`JointMLP` via full-batch Adam on the given loss (`nll_loss` or `beta_nll_loss`)."""
    device = device or get_device()
    torch.manual_seed(seed)
    xt = torch.tensor(x, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    model = JointMLP(x.shape[1], hidden_sizes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(n_epochs):
        opt.zero_grad()
        loss = loss_fn(model(xt), yt)
        loss.backward()
        opt.step()
    model.eval()
    return model


def train_sigma_head(
    x: np.ndarray,
    resid: np.ndarray,
    hidden_sizes: tuple[int, ...] = NN_HIDDEN,
    lr: float = v1.NN_LR,
    n_epochs: int = v1.NN_EPOCHS,
    seed: int = 0,
    device: torch.device | None = None,
) -> SigmaHead:
    """Fits log-sigma^2(x) on precomputed residuals via the Gaussian NLL with the mean pinned at 0.

    Reuses `nll_loss` directly: since the residual (y minus a FROZEN mean prediction) is already
    computed, feeding it in as the target with a zero mean column reduces `nll_loss` to exactly
    0.5*(log(2*pi) + log_var + residual^2/variance) -- the correct NLL for the sigma head alone.
    """
    device = device or get_device()
    torch.manual_seed(seed)
    xt = torch.tensor(x, dtype=torch.float32, device=device)
    rt = torch.tensor(resid, dtype=torch.float32, device=device)
    zeros = torch.zeros_like(rt)
    model = SigmaHead(x.shape[1], hidden_sizes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(n_epochs):
        opt.zero_grad()
        outputs = torch.stack([zeros, model(xt)], dim=1)
        loss = nll_loss(outputs, rt)
        loss.backward()
        opt.step()
    model.eval()
    return model


def train_mean_mlp_earlystop(
    x: np.ndarray,
    y: np.ndarray,
    val_frac: float = EARLYSTOP_VAL_FRAC,
    hidden_sizes: tuple[int, ...] = NN_HIDDEN,
    lr: float = v1.NN_LR,
    n_epochs: int = v1.NN_EPOCHS,
    check_every: int = EARLYSTOP_CHECK_EVERY,
    patience_checks: int = EARLYSTOP_PATIENCE_CHECKS,
    seed: int = 0,
    device: torch.device | None = None,
) -> tuple[v1.MeanMLP, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fits a mean MLP with early stopping on a held-out val split -- arm (c)'s in-sample fit stage.

    Returns the model plus the (x_fit, y_fit) rows it was actually trained on (the caller uses these
    to compute arm (c)'s IN-SAMPLE, diseased residuals) and the untouched (x_val, y_val) rows (reused
    downstream as arm (e)/(f)'s calibration pool, since they never entered the mean model's gradient).
    """
    device = device or get_device()
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    n_val = max(int(n * val_frac), 1)
    idx_val, idx_fit = idx[:n_val], idx[n_val:]
    x_fit, y_fit = x[idx_fit], y[idx_fit]
    x_val, y_val = x[idx_val], y[idx_val]

    torch.manual_seed(seed)
    xt_fit = torch.tensor(x_fit, dtype=torch.float32, device=device)
    yt_fit = torch.tensor(y_fit, dtype=torch.float32, device=device)
    xt_val = torch.tensor(x_val, dtype=torch.float32, device=device)
    yt_val = torch.tensor(y_val, dtype=torch.float32, device=device)
    model = v1.MeanMLP(x.shape[1], hidden_sizes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val, best_state, checks_since_best = float("inf"), None, 0
    for epoch in range(n_epochs):
        opt.zero_grad()
        loss = torch.mean((model(xt_fit) - yt_fit) ** 2)
        loss.backward()
        opt.step()
        if (epoch + 1) % check_every == 0:
            model.eval()
            with torch.no_grad():
                val_mse = float(torch.mean((model(xt_val) - yt_val) ** 2).item())
            model.train()
            if val_mse < best_val:
                best_val, checks_since_best = val_mse, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                checks_since_best += 1
                if checks_since_best >= patience_checks:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, x_fit, y_fit, x_val, y_val


# --------------------------------------------------------------------------------------
# Cross-fitted (honest) residuals -- mirrors v1.cross_fitted_sigma2's fold structure
# --------------------------------------------------------------------------------------


def cross_fitted_residuals(x: np.ndarray, y: np.ndarray, fit_fn: v1.FitFn, k: int = K_FOLDS, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Honest out-of-fold residuals: every point is scored by a fit trained on the OTHER k-1 folds.

    Same fold split as `v1.cross_fitted_sigma2` (identical rng/seed), but returns the per-point
    (x, residual) pairs rather than a pooled MSE scalar -- needed here to train a sigma(x) head on
    every point's honest residual instead of a single global variance.
    """
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    x_out, resid_out = [], []
    for i in range(k):
        idx_test = folds[i]
        idx_train = np.concatenate([folds[j] for j in range(k) if j != i])
        pred_fn = fit_fn(x[idx_train], y[idx_train])
        x_out.append(x[idx_test])
        resid_out.append(y[idx_test].astype(np.float64) - pred_fn(x[idx_test]))
    return np.concatenate(x_out), np.concatenate(resid_out)


# --------------------------------------------------------------------------------------
# Predict-fn wrappers and scoring
# --------------------------------------------------------------------------------------


def _numpy_predict_fn(model: nn.Module, device: torch.device, column: int | None = None) -> PredFn:
    """Wraps a trained torch model into a `pred_fn(x) -> np.ndarray` callable (no-grad, numpy in/out).

    `column=None` returns the model's raw (squeezed) output; `column=0`/`1` selects one column of a
    two-headed output, as produced by :class:`JointMLP` (mean, log-variance).
    """

    def _pred(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float32, device=device)
            out = model(xt)
            if column is not None:
                out = out[:, column]
            return out.cpu().numpy().astype(np.float64)

    return _pred


def _numpy_sigma_fn(log_var_pred_fn: PredFn) -> PredFn:
    """Converts a log-variance `pred_fn` into a sigma (std-dev) `pred_fn`."""
    return lambda x: np.sqrt(np.exp(log_var_pred_fn(x)))


def score_density(mean_fn: PredFn, sigma_fn: PredFn, eval_x: np.ndarray, eval_y: np.ndarray) -> dict:
    """Scores a (mean, sigma) pair on a held-out set against the KNOWN sigma(x): ratio error, NLL, SSR."""
    mu = mean_fn(eval_x)
    sigma = np.maximum(sigma_fn(eval_x), 1e-6)
    sigma_true = td.v_toy1_sigma(eval_x)
    y = eval_y.astype(np.float64).ravel()
    ratio_error = float(np.mean(np.abs(sigma / sigma_true - 1.0)))
    nll = float(np.mean(0.5 * (np.log(2 * np.pi) + 2.0 * np.log(sigma) + ((y - mu) / sigma) ** 2)))
    ssr = float(np.mean(((y - mu) / sigma) ** 2))
    return {"sigma_ratio_error": ratio_error, "nll": nll, "ssr": ssr}


def score_interval(lower: np.ndarray, upper: np.ndarray, eval_y: np.ndarray) -> dict:
    """Interval-only metrics for the conformal arm: empirical coverage and mean width."""
    y = eval_y.astype(np.float64).ravel()
    return {"coverage": float(np.mean((y >= lower) & (y <= upper))), "mean_width": float(np.mean(upper - lower))}


# --------------------------------------------------------------------------------------
# Per-bin recalibration helpers (arm (e))
# --------------------------------------------------------------------------------------


def _tercile_edges(x: np.ndarray, n_bins: int = N_BINS_RECAL) -> np.ndarray:
    """Interior quantile cut points splitting `x` into `n_bins` equal-count bins (default: terciles)."""
    return np.quantile(x.ravel(), [i / n_bins for i in range(1, n_bins)])


def _bin_index(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Assigns each `x` to a bin index in `[0, len(edges)]` via the interior cut points `edges`."""
    return np.digitize(x.ravel(), edges)


class _MeanSigmaAdapter:
    """Adapter exposing `predict()`/`predict_uncertainty()` over a frozen (mean, sigma) pair.

    This is the interface `LocallyAdaptiveConformalWrapper` requires.
    """

    def __init__(self, mean_fn: PredFn, sigma_fn: PredFn) -> None:
        """Stores the frozen mean/sigma predict-fns to expose under the model interface."""
        self.mean_fn = mean_fn
        self.sigma_fn = sigma_fn

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.mean_fn(x)

    def predict_uncertainty(self, x: np.ndarray) -> np.ndarray:
        return self.sigma_fn(x)


# --------------------------------------------------------------------------------------
# Per-arm runners
# --------------------------------------------------------------------------------------


def run_arm_joint(x: np.ndarray, y: np.ndarray, eval_x: np.ndarray, eval_y: np.ndarray, loss_fn: LossFn, epochs: int, lr: float, seed: int, device: torch.device) -> dict:
    """Arm (a)/(b): a joint (mean, log-variance) MLP trained end-to-end on `loss_fn`."""
    model = train_joint_mlp(x, y, loss_fn, NN_HIDDEN, lr, epochs, seed, device)
    mean_fn = _numpy_predict_fn(model, device, column=0)
    sigma_fn = _numpy_sigma_fn(_numpy_predict_fn(model, device, column=1))
    return score_density(mean_fn, sigma_fn, eval_x, eval_y)


def fit_arm_c(x: np.ndarray, y: np.ndarray, epochs: int, lr: float, seed: int, device: torch.device) -> dict:
    """Arm (c): mean-first two-stage, IN-SAMPLE (diseased control).

    Fits a mean MLP with early stopping on a held-out val split, freezes it, then fits a sigma(x)
    head on the IN-SAMPLE squared residuals of the points the mean model directly trained on --
    biased low, same disease V1 found for a single global scalar, now for a per-input sigma(x).
    """
    mean_model, x_fit, y_fit, x_val, y_val = train_mean_mlp_earlystop(
        x, y, EARLYSTOP_VAL_FRAC, NN_HIDDEN, lr, epochs, EARLYSTOP_CHECK_EVERY, EARLYSTOP_PATIENCE_CHECKS, seed, device
    )
    mean_fn = _numpy_predict_fn(mean_model, device, column=None)
    resid_in_sample = y_fit.astype(np.float64) - mean_fn(x_fit)
    sigma_model = train_sigma_head(x_fit, resid_in_sample, NN_HIDDEN, lr, epochs, seed, device)
    sigma_fn = _numpy_sigma_fn(_numpy_predict_fn(sigma_model, device, column=None))
    return {"mean_fn": mean_fn, "sigma_fn": sigma_fn, "x_val": x_val, "y_val": y_val}


def run_arm_crossfit(x: np.ndarray, y: np.ndarray, eval_x: np.ndarray, eval_y: np.ndarray, epochs: int, lr: float, k: int, seed: int, device: torch.device) -> dict:
    """Arm (d): mean-first + CROSS-FITTED residuals -- the fix.

    Every training point gets an honest out-of-fold residual (K-fold cross-fit, mirroring
    `v1.cross_fitted_sigma2`'s fold structure); ONE sigma(x) head is trained on all of them. The
    deployed mean predictor is refit on the FULL training pool (the fold models exist only to
    produce honest residuals, not to predict on new points).
    """
    fit_fn = v1.mlp_fit_fn(hidden_sizes=NN_HIDDEN, lr=lr, n_epochs=epochs, seed=seed, device=device)
    x_oof, resid_oof = cross_fitted_residuals(x, y, fit_fn, k=k, seed=seed)
    sigma_model = train_sigma_head(x_oof, resid_oof, NN_HIDDEN, lr, epochs, seed, device)
    sigma_fn = _numpy_sigma_fn(_numpy_predict_fn(sigma_model, device, column=None))
    final_mean_model, _ = v1.train_mean_mlp(x, y, NN_HIDDEN, lr, epochs, seed, device)
    mean_fn = _numpy_predict_fn(final_mean_model, device, column=None)
    return score_density(mean_fn, sigma_fn, eval_x, eval_y)


def run_arm_recalibrated(arm_c_fit: dict, eval_x: np.ndarray, eval_y: np.ndarray, n_bins: int = N_BINS_RECAL) -> dict:
    """Arm (e): per-bin scale recalibration on top of (c).

    A scalar multiplier per x-tercile is fit on half of (c)'s held-out val split and applied to the
    other half, then SWAPPED and averaged (the STACK-2b remedy), so both halves of the calibration
    pool contribute without either one leaking into the other's multiplier.
    """
    x_val, y_val = arm_c_fit["x_val"], arm_c_fit["y_val"]
    mean_fn, sigma_fn = arm_c_fit["mean_fn"], arm_c_fit["sigma_fn"]
    n = len(y_val)
    idx = np.random.default_rng(0).permutation(n)
    half = n // 2
    idx_a, idx_b = idx[:half], idx[half:]
    edges = _tercile_edges(x_val, n_bins)

    def _bin_multipliers(idx_calib: np.ndarray) -> np.ndarray:
        resid2 = (y_val[idx_calib].astype(np.float64) - mean_fn(x_val[idx_calib])) ** 2
        sigma2 = sigma_fn(x_val[idx_calib]) ** 2
        bins = _bin_index(x_val[idx_calib], edges)
        mult = np.ones(n_bins)
        for b in range(n_bins):
            mask = bins == b
            if mask.any():
                mult[b] = np.sqrt(max(float(np.mean(resid2[mask] / sigma2[mask])), 1e-6))
        return mult

    mult_final = 0.5 * (_bin_multipliers(idx_a) + _bin_multipliers(idx_b))  # swap-and-average

    def _recalibrated_sigma_fn(x: np.ndarray) -> np.ndarray:
        return sigma_fn(x) * mult_final[_bin_index(x, edges)]

    return score_density(mean_fn, _recalibrated_sigma_fn, eval_x, eval_y)


def run_arm_conformal(arm_c_fit: dict, eval_x: np.ndarray, eval_y: np.ndarray, alpha: float = ALPHA_CONFORMAL) -> dict:
    """Arm (f): `LocallyAdaptiveConformalWrapper` around (c)'s (mean, sigma) pair -- INTERVAL metrics only."""
    x_val, y_val = arm_c_fit["x_val"], arm_c_fit["y_val"]
    lacw = LocallyAdaptiveConformalWrapper(_MeanSigmaAdapter(arm_c_fit["mean_fn"], arm_c_fit["sigma_fn"]))
    lacw.calibrate(x_val, y_val, alpha=alpha)
    lower, upper = lacw.predict_interval(eval_x)
    return score_interval(lower, upper, eval_y)


# --------------------------------------------------------------------------------------
# Per-unit runner, ranking checks, main
# --------------------------------------------------------------------------------------


def run_unit(n: int, seed: int, cfg: dict, device: torch.device) -> dict:
    """Runs all six arms for one (N, seed) unit and scores each against the shared held-out set."""
    x, y = td.make_v_toy1(n=n, seed=seed)
    eval_x, eval_y = td.make_v_toy1(n=cfg["n_eval"], seed=seed + EVAL_SEED_OFFSET)
    epochs, lr = cfg["epochs"], cfg["lr"]

    t0 = time.time()
    res_a = run_arm_joint(x, y, eval_x, eval_y, nll_loss, epochs, lr, seed, device)
    res_b = run_arm_joint(x, y, eval_x, eval_y, partial(beta_nll_loss, beta=0.5), epochs, lr, seed, device)
    arm_c_fit = fit_arm_c(x, y, epochs, lr, seed, device)
    res_c = score_density(arm_c_fit["mean_fn"], arm_c_fit["sigma_fn"], eval_x, eval_y)
    res_d = run_arm_crossfit(x, y, eval_x, eval_y, epochs, lr, K_FOLDS, seed, device)
    res_e = run_arm_recalibrated(arm_c_fit, eval_x, eval_y)
    res_f = run_arm_conformal(arm_c_fit, eval_x, eval_y)
    wall_time = time.time() - t0

    return {
        "n": n, "seed": seed, "wall_time_sec": wall_time,
        "joint_nll": res_a, "beta_nll": res_b, "mean_first_insample": res_c,
        "mean_first_crossfit": res_d, "recalibrated": res_e, "conformal": res_f,
    }


def compute_ranking_checks(all_results: dict) -> dict:
    """Scores the pre-registered expectations against the collected units (report only, no fudging)."""
    density_arms = ["joint_nll", "beta_nll", "mean_first_insample", "mean_first_crossfit", "recalibrated"]
    checks: dict = {}
    for n in {u["n"] for u in all_results["units"]}:
        units = [u for u in all_results["units"] if u["n"] == n]
        mean_ratio = {arm: float(np.mean([u[arm]["sigma_ratio_error"] for u in units])) for arm in density_arms}
        a, b, c, d, e = (mean_ratio["joint_nll"], mean_ratio["beta_nll"], mean_ratio["mean_first_insample"],
                         mean_ratio["mean_first_crossfit"], mean_ratio["recalibrated"])
        ordering_met = (d <= e) and (e < b) and (b < c) and (c < a)
        checks[f"n={n}"] = {"mean_sigma_ratio_error": mean_ratio, "ordering_expectation_met": ordering_met}

    units_1000 = [u for u in all_results["units"] if u["n"] == 1000]
    ssr_d = [u["mean_first_crossfit"]["ssr"] for u in units_1000]
    in_range = [0.9 <= s <= 1.1 for s in ssr_d]
    checks["crossfit_ssr_n1000"] = {"ssr_values": ssr_d, "n_in_range": int(sum(in_range)), "n_total": len(ssr_d), "all_in_range": bool(all(in_range))}
    return checks


def main(smoke: bool = False) -> None:
    """Runs the full (or `--smoke`) V2 battery and writes the summary JSON."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = get_device()
    cfg = SMOKE_CONFIG if smoke else FULL_CONFIG
    print(f"[main] device={device} smoke={smoke} cfg={cfg}")

    all_results: dict = {"config": cfg, "units": []}
    t_start = time.time()
    for n in cfg["n_grid"]:
        for seed in cfg["seeds"]:
            print(f"[main] N={n} seed={seed} ...")
            unit = run_unit(n, seed, cfg, device)
            all_results["units"].append(unit)
            print(f"  wall_time={unit['wall_time_sec']:.1f}s "
                  f"a_ratio={unit['joint_nll']['sigma_ratio_error']:.3f} "
                  f"b_ratio={unit['beta_nll']['sigma_ratio_error']:.3f} "
                  f"c_ratio={unit['mean_first_insample']['sigma_ratio_error']:.3f} "
                  f"d_ratio={unit['mean_first_crossfit']['sigma_ratio_error']:.3f} "
                  f"e_ratio={unit['recalibrated']['sigma_ratio_error']:.3f} "
                  f"d_ssr={unit['mean_first_crossfit']['ssr']:.3f} "
                  f"f_coverage={unit['conformal']['coverage']:.3f} f_width={unit['conformal']['mean_width']:.3f}")

    all_results["ranking_checks"] = compute_ranking_checks(all_results)
    all_results["total_wall_time_sec"] = time.time() - t_start

    summary_name = "v2_summary_smoke.json" if smoke else "v2_summary.json"
    summary_path = os.path.join(RESULTS_DIR, summary_name)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[main] wrote {summary_path}")
    print(f"[main] total wall time: {all_results['total_wall_time_sec']:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="1 seed, N=200 only, tiny epochs -- proves the 6 arms run end-to-end.")
    args = parser.parse_args()

    torch.set_num_threads(TORCH_THREADS)
    main(smoke=args.smoke)
