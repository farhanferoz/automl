"""WS3/V1 -- global-variance-estimation mechanisms, ranked against analytic ground truth.

Capacity-ladder program, `docs/plans/capacity_ladder_2026-07-09/EXECUTION_PLAN.md` sec 4. V0
established the joint-NLL variance-collapse disease; V1 ranks the FIX FAMILY for GLOBAL variance
against exact ground truth on well-specified linear-Gaussian data, plus a deliberate mean-model
misspecification arm. No touching `automl_package/models/` -- this is pure diagnostic plumbing.

Mechanisms (all on a per-(toy, N, seed) unit):
  (i)   in-sample MLE  sigma_hat^2 = RSS/N              -- biased LOW.
  (ii)  exact evidence -- MacKay alpha,beta fixed-point on a LINEAR (or linear-in-basis) mean
        model; reports evidence noise variance = 1/beta, learned weight decay = alpha/beta, and
        the effective-parameter count gamma.
  (iii) held-out sigma_hat^2 -- mean fit on one half, scored (MSE) on the other half.
  (iv)  K=5-fold cross-fitted sigma_hat^2 -- every point scored by a mean fit that never saw it;
        pooled MSE, plus the per-fold MSE variance.
  NN mean variant of (iii)/(iv): a small MLP mean (MSE-only, no variance head -- the
  honest-residual family, not joint NLL), with a plateau convergence check (V0 lesson: a fixed
  epoch budget can under-train a GD mean fit; verify convergence rather than trust the budget).

Toys:
  - V-toy0 (`_toy_datasets.make_v_toy0`): exactly linear mean, well-specified.
  - V-toy1h WELL-SPECIFIED arm (`_toy_datasets.make_v_toy1h`): the mean basis
    phi(x) = [1, sin(2πx), cos(2πx)] exactly spans f(x)=sin(2πx) -- a correct-capacity LINEAR-IN-
    BASIS model, so the same closed-form evidence machinery applies.
  - V-toy1h MISSPECIFIED arm: deliberately fit phi(x) = [1, x] (pure linear) to the curved mean --
    the evidence-sigma2 and held-out-sigma2 are expected to ABSORB the bias (physically inflated,
    calibration-correct), per the plan's pre-registration -- not a failure.

Usage:
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/capacity_ladder_variance_v1.py --selftest
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/capacity_ladder_variance_v1.py --probe
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/capacity_ladder_variance_v1.py
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
import torch.nn as nn

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root, so `import automl_package` works
import _toy_datasets as td

from automl_package.utils.pytorch_utils import get_device

RESULTS_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "V1")

SEEDS = [0, 1, 2]
N_GRID = [200, 1000]
K_FOLDS = 5
NN_HIDDEN = (16, 16)
NN_LR = 1e-2
NN_EPOCHS = 2000
PLATEAU_FRAC = 0.1  # fraction of epochs compared against the preceding equal-size slice
PLATEAU_TOL = 0.02  # relative-change tolerance for the plateau convergence check
TORCH_THREADS = 3


# --------------------------------------------------------------------------------------
# (ii) exact evidence -- MacKay alpha,beta fixed-point (linear-in-basis model)
# --------------------------------------------------------------------------------------


def evidence_fixed_point(phi: np.ndarray, y: np.ndarray, alpha_init: float = 1e-3, max_iter: int = 100, tol: float = 1e-6) -> dict:
    """MacKay type-II ML alpha,beta fixed-point for linear-Gaussian regression.

    Prior w ~ N(0, alpha^-1 I), noise eps ~ N(0, beta^-1). Iterates:
        A = alpha*I + beta*phi^T phi;  m = beta * A^-1 phi^T y
        gamma = sum_i lambda_i / (lambda_i + alpha),  {lambda_i} = eigenvalues of beta*phi^T phi
        alpha <- gamma / (m^T m);  beta <- (N - gamma) / ||y - phi m||^2
    until |Delta log alpha|, |Delta log beta| < tol or max_iter is reached.

    Args:
        phi: (N, p) design matrix (float64).
        y: (N,) targets (float64).
        alpha_init: initial prior precision.
        max_iter: fixed-point iteration cap.
        tol: convergence tolerance on the log-parameter deltas.

    Returns:
        dict with keys alpha, beta, m (posterior mean weights), gamma (effective params),
        n_iter, converged.
    """
    n, p = phi.shape
    phit_phi = phi.T @ phi
    phit_y = phi.T @ y
    alpha = alpha_init
    beta = 1.0 / max(np.var(y), 1e-12)
    log_alpha_prev, log_beta_prev = np.log(alpha), np.log(beta)
    m = np.zeros(p)
    gamma = 0.0

    for it in range(1, max_iter + 1):
        a_mat = alpha * np.eye(p) + beta * phit_phi
        a_inv = np.linalg.inv(a_mat)
        m = beta * (a_inv @ phit_y)
        eigvals = np.linalg.eigvalsh(beta * phit_phi)
        gamma = float(np.sum(eigvals / (eigvals + alpha)))
        alpha = gamma / max(float(m @ m), 1e-12)
        resid = y - phi @ m
        beta = (n - gamma) / max(float(resid @ resid), 1e-12)

        log_alpha, log_beta = np.log(alpha), np.log(beta)
        converged = abs(log_alpha - log_alpha_prev) < tol and abs(log_beta - log_beta_prev) < tol
        log_alpha_prev, log_beta_prev = log_alpha, log_beta
        if converged:
            return {"alpha": alpha, "beta": beta, "m": m, "gamma": gamma, "n_iter": it, "converged": True}

    return {"alpha": alpha, "beta": beta, "m": m, "gamma": gamma, "n_iter": max_iter, "converged": False}


# --------------------------------------------------------------------------------------
# Honest-residual family: (iii) held-out-half, (iv) K-fold cross-fitted
# --------------------------------------------------------------------------------------

FitFn = Callable[[np.ndarray, np.ndarray], Callable[[np.ndarray], np.ndarray]]


def ols_fit_fn(phi_fn: Callable[[np.ndarray], np.ndarray]) -> FitFn:
    """Wraps a design-matrix builder `phi_fn(x) -> (n, p)` into a `fit_fn(x, y) -> pred_fn(x)`."""

    def _fit(x_train: np.ndarray, y_train: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        phi = phi_fn(x_train).astype(np.float64)
        w_hat, _, _, _ = np.linalg.lstsq(phi, y_train.astype(np.float64), rcond=None)

        def _pred(x: np.ndarray) -> np.ndarray:
            return phi_fn(x).astype(np.float64) @ w_hat

        return _pred

    return _fit


def held_out_sigma2(x: np.ndarray, y: np.ndarray, fit_fn: FitFn, seed: int = 0) -> float:
    """(iii) Fits the mean on a random half, scores mean squared residual on the OTHER half."""
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    half = n // 2
    idx_fit, idx_score = idx[:half], idx[half:]
    pred_fn = fit_fn(x[idx_fit], y[idx_fit])
    resid = y[idx_score].astype(np.float64) - pred_fn(x[idx_score])
    return float(np.mean(resid**2))


def cross_fitted_sigma2(x: np.ndarray, y: np.ndarray, fit_fn: FitFn, k: int = K_FOLDS, seed: int = 0) -> tuple[float, list[float]]:
    """(iv) K-fold cross-fitted: every point scored by a mean fit trained on the OTHER K-1 folds.

    Returns:
        (pooled_sigma2, per_fold_mse): the pooled out-of-fold MSE over all N points, and the
        per-fold MSE list (its variance is the requested fold-to-fold dispersion).
    """
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    all_resid_sq = []
    per_fold_mse = []
    for i in range(k):
        idx_test = folds[i]
        idx_train = np.concatenate([folds[j] for j in range(k) if j != i])
        pred_fn = fit_fn(x[idx_train], y[idx_train])
        resid = y[idx_test].astype(np.float64) - pred_fn(x[idx_test])
        all_resid_sq.append(resid**2)
        per_fold_mse.append(float(np.mean(resid**2)))
    pooled = float(np.mean(np.concatenate(all_resid_sq)))
    return pooled, per_fold_mse


# --------------------------------------------------------------------------------------
# NN mean variant (for (iii)/(iv) only -- the honest-residual family, MSE-only, no variance head)
# --------------------------------------------------------------------------------------


class MeanMLP(nn.Module):
    """Mean-only MLP: no variance head, trained by plain MSE (the honest-residual family)."""

    def __init__(self, input_dim: int, hidden_sizes: tuple[int, ...] = NN_HIDDEN):
        super().__init__()
        layers: list[nn.Module] = []
        d = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(d, h), nn.Tanh()]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_mean_mlp(
    x: np.ndarray,
    y: np.ndarray,
    hidden_sizes: tuple[int, ...] = NN_HIDDEN,
    lr: float = NN_LR,
    n_epochs: int = NN_EPOCHS,
    seed: int = 0,
    device: torch.device | None = None,
) -> tuple[MeanMLP, dict]:
    """Trains a mean-only MLP via full-batch Adam on plain MSE.

    Convergence is a PLATEAU check (V0 lesson: a fixed epoch budget can under-train a GD fit) --
    the mean training loss over the last `PLATEAU_FRAC` of epochs is compared against the equal-
    size slice preceding it; `converged=True` iff the relative change is below `PLATEAU_TOL`.
    """
    device = device or get_device()
    torch.manual_seed(seed)
    xt = torch.tensor(x, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    model = MeanMLP(x.shape[1], hidden_sizes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    losses = np.zeros(n_epochs, dtype=np.float64)
    for epoch in range(n_epochs):
        opt.zero_grad()
        loss = torch.mean((model(xt) - yt) ** 2)
        loss.backward()
        opt.step()
        losses[epoch] = float(loss.item())
    model.eval()

    window = max(int(n_epochs * PLATEAU_FRAC), 10)
    prev_slice = losses[-2 * window : -window]
    last_slice = losses[-window:]
    rel_change = abs(prev_slice.mean() - last_slice.mean()) / max(prev_slice.mean(), 1e-12)
    conv_info = {"final_loss": float(losses[-1]), "rel_change": float(rel_change), "converged": bool(rel_change < PLATEAU_TOL)}
    return model, conv_info


def mlp_fit_fn(hidden_sizes: tuple[int, ...] = NN_HIDDEN, lr: float = NN_LR, n_epochs: int = NN_EPOCHS, seed: int = 0, device: torch.device | None = None) -> FitFn:
    """Wraps `train_mean_mlp` into the `fit_fn(x, y) -> pred_fn(x)` contract used by
    :func:`held_out_sigma2`/:func:`cross_fitted_sigma2` -- retrains a fresh MLP on every call
    (once per fold), which is what honest cross-fitting requires.
    """
    device = device or get_device()

    def _fit(x_train: np.ndarray, y_train: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        model, _ = train_mean_mlp(x_train, y_train, hidden_sizes, lr, n_epochs, seed, device)

        def _pred(x: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                xt = torch.tensor(x, dtype=torch.float32, device=device)
                return model(xt).cpu().numpy().astype(np.float64)

        return _pred

    return _fit


# --------------------------------------------------------------------------------------
# Per-unit mechanism runners
# --------------------------------------------------------------------------------------


def run_mechanisms_linear(x: np.ndarray, y: np.ndarray, phi_fn: Callable[[np.ndarray], np.ndarray], sigma_true: float, seed: int = 0) -> dict:
    """Runs (i) MLE, (ii) evidence, (iii) held-out, (iv) cross-fitted for a linear-in-`phi_fn` mean."""
    phi = phi_fn(x).astype(np.float64)
    yf = y.astype(np.float64)
    n, p = phi.shape

    w_hat, _, _, _ = np.linalg.lstsq(phi, yf, rcond=None)
    resid = yf - phi @ w_hat
    mle_sigma2 = float(np.sum(resid**2) / n)

    ev = evidence_fixed_point(phi, yf)
    evidence_sigma2 = 1.0 / ev["beta"]

    fit_fn = ols_fit_fn(phi_fn)
    heldout_sigma2 = held_out_sigma2(x, y, fit_fn, seed=seed)
    crossfit_sigma2, per_fold_mse = cross_fitted_sigma2(x, y, fit_fn, k=K_FOLDS, seed=seed)

    return {
        "n": n, "p": p,
        "sigma_true2": sigma_true**2,
        "mle_sigma2": mle_sigma2,
        "mle_bias_factor_expected": (n - p) / n,
        "evidence_sigma2": evidence_sigma2,
        "evidence_alpha": ev["alpha"], "evidence_beta": ev["beta"],
        "evidence_weight_decay": ev["alpha"] / ev["beta"],
        "evidence_gamma": ev["gamma"],
        "evidence_converged": ev["converged"], "evidence_n_iter": ev["n_iter"],
        "heldout_sigma2": heldout_sigma2,
        "crossfit_sigma2": crossfit_sigma2,
        "crossfit_per_fold_mse": per_fold_mse,
        "crossfit_fold_variance": float(np.var(per_fold_mse)),
    }


def run_mechanisms_nn(x: np.ndarray, y: np.ndarray, sigma_true: float, seed: int = 0, device: torch.device | None = None) -> dict:
    """NN mean variant of (iii)/(iv): honest held-out/cross-fitted sigma2 with a small MLP mean."""
    device = device or get_device()
    _, conv_info = train_mean_mlp(x, y, seed=seed, device=device)  # upfront full-data convergence check

    fit_fn = mlp_fit_fn(seed=seed, device=device)
    heldout_sigma2 = held_out_sigma2(x, y, fit_fn, seed=seed)
    crossfit_sigma2, per_fold_mse = cross_fitted_sigma2(x, y, fit_fn, k=K_FOLDS, seed=seed)

    return {
        "sigma_true2": sigma_true**2,
        "nn_convergence": conv_info,
        "heldout_sigma2": heldout_sigma2,
        "crossfit_sigma2": crossfit_sigma2,
        "crossfit_per_fold_mse": per_fold_mse,
        "crossfit_fold_variance": float(np.var(per_fold_mse)),
    }


# --------------------------------------------------------------------------------------
# Basis / phi_fn definitions
# --------------------------------------------------------------------------------------


def _phi_v_toy0(x: np.ndarray) -> np.ndarray:
    return x  # already the raw p=5 design matrix


def _phi_v_toy1h_wellspecified(x: np.ndarray) -> np.ndarray:
    """phi(x) = [1, sin(2πx), cos(2πx)] -- exactly spans f(x)=sin(2πx): correct-capacity, well-specified."""
    xr = x.ravel().astype(np.float64)
    return np.column_stack([np.ones_like(xr), np.sin(2 * np.pi * xr), np.cos(2 * np.pi * xr)])


def _phi_v_toy1h_misspecified(x: np.ndarray) -> np.ndarray:
    """phi(x) = [1, x] -- pure linear, cannot represent the curved sin(2πx) mean: deliberate misspecification."""
    xr = x.ravel().astype(np.float64)
    return np.column_stack([np.ones_like(xr), xr])


# --------------------------------------------------------------------------------------
# Selftest / probe / main
# --------------------------------------------------------------------------------------


def selftest() -> None:
    """Synthetic known-answer checks (must PASS before any real read, per sec 0b)."""
    rng = np.random.default_rng(0)
    n, p, sigma_true = 10_000, 4, 0.8
    w_true = rng.normal(0.0, 1.0, p)
    x = rng.normal(0.0, 1.0, (n, p))
    y = x @ w_true + sigma_true * rng.normal(0.0, 1.0, n)

    res = run_mechanisms_linear(x.astype(np.float32), y.astype(np.float32), _phi_v_toy0, sigma_true, seed=0)
    print(f"[selftest] mle_sigma2={res['mle_sigma2']:.4f} evidence_sigma2={res['evidence_sigma2']:.4f} "
          f"heldout_sigma2={res['heldout_sigma2']:.4f} crossfit_sigma2={res['crossfit_sigma2']:.4f} "
          f"true={sigma_true**2:.4f} evidence_converged={res['evidence_converged']}")
    assert res["evidence_converged"], "evidence fixed-point must converge on well-specified data"
    assert abs(res["evidence_sigma2"] - sigma_true**2) < 0.05, "evidence σ² must recover truth"
    assert abs(res["heldout_sigma2"] - sigma_true**2) < 0.08, "held-out σ² must recover truth"
    assert abs(res["crossfit_sigma2"] - sigma_true**2) < 0.05, "cross-fitted σ² must recover truth"
    assert res["mle_sigma2"] < res["evidence_sigma2"], "MLE must be biased low relative to evidence"

    device = get_device()
    x1 = rng.uniform(0.0, 1.0, 2000).reshape(-1, 1).astype(np.float32)
    y1 = (np.sin(2 * np.pi * x1.ravel()) + 0.3 * rng.normal(0.0, 1.0, 2000)).astype(np.float32)
    res_nn = run_mechanisms_nn(x1, y1, sigma_true=0.3, seed=0, device=device)
    print(f"[selftest] NN heldout_sigma2={res_nn['heldout_sigma2']:.4f} crossfit_sigma2={res_nn['crossfit_sigma2']:.4f} "
          f"true={0.3**2:.4f} nn_converged={res_nn['nn_convergence']['converged']}")
    assert res_nn["nn_convergence"]["converged"], f"NN mean fit did not plateau: {res_nn['nn_convergence']}"
    assert abs(res_nn["heldout_sigma2"] - 0.3**2) < 0.03, "NN held-out σ² must recover truth"
    assert abs(res_nn["crossfit_sigma2"] - 0.3**2) < 0.02, "NN cross-fitted σ² must recover truth"
    print("[selftest] PASS")


def probe() -> None:
    """No-unmeasured-time rule (sec 0b): run ONE unit per run shape, measure wall-time, extrapolate."""
    device = get_device()
    print(f"[probe] device={device}")
    n = 1000
    x0, y0 = td.make_v_toy0(n=n, seed=0)
    x1h, y1h = td.make_v_toy1h(n=n, seed=0)

    t0 = time.time()
    run_mechanisms_linear(x0, y0, _phi_v_toy0, sigma_true=1.0, seed=0)
    t_v0_linear = time.time() - t0

    t0 = time.time()
    run_mechanisms_linear(x1h, y1h, _phi_v_toy1h_wellspecified, sigma_true=td.V_TOY1H_SIGMA, seed=0)
    run_mechanisms_linear(x1h, y1h, _phi_v_toy1h_misspecified, sigma_true=td.V_TOY1H_SIGMA, seed=0)
    t_v1h_linear = time.time() - t0

    t0 = time.time()
    run_mechanisms_nn(x1h, y1h, sigma_true=td.V_TOY1H_SIGMA, seed=0, device=device)
    t_nn = time.time() - t0

    print(f"[probe] v_toy0 linear (i-iv): {t_v0_linear:.2f}s")
    print(f"[probe] v_toy1h linear (i-iv) x2 arms: {t_v1h_linear:.2f}s")
    print(f"[probe] NN variant (iii-iv, 7 MLP fits): {t_nn:.2f}s")
    per_unit = t_v0_linear + t_nn  # v_toy0 arm (linear + NN)
    per_unit_1h = t_v1h_linear + t_nn  # v_toy1h arm (2 linear + NN)
    n_units = len(SEEDS) * len(N_GRID)
    total_est = n_units * (per_unit + per_unit_1h)
    print(f"[probe] extrapolated full matrix ({n_units} (seed,N) units x [v_toy0 + v_toy1h]): {total_est:.1f}s ({total_est / 60:.1f} min)")


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = get_device()
    print(f"[main] device={device}")

    all_results: dict = {
        "config": {
            "seeds": SEEDS, "n_grid": N_GRID, "k_folds": K_FOLDS,
            "nn_hidden": NN_HIDDEN, "nn_lr": NN_LR, "nn_epochs": NN_EPOCHS,
            "plateau_frac": PLATEAU_FRAC, "plateau_tol": PLATEAU_TOL,
        },
        "v_toy0": [], "v_toy1h_wellspecified": [], "v_toy1h_misspecified": [],
    }

    t_start = time.time()
    for n in N_GRID:
        for seed in SEEDS:
            print(f"[main] v_toy0 N={n} seed={seed} ...")
            x0, y0 = td.make_v_toy0(n=n, sigma=1.0, seed=seed)
            t0 = time.time()
            lin0 = run_mechanisms_linear(x0, y0, _phi_v_toy0, sigma_true=1.0, seed=seed)
            nn0 = run_mechanisms_nn(x0, y0, sigma_true=1.0, seed=seed, device=device)
            wt0 = time.time() - t0
            rec0 = {"n": n, "seed": seed, "wall_time_sec": wt0, "linear": lin0, "nn": nn0}
            all_results["v_toy0"].append(rec0)
            print(f"  wall_time={wt0:.1f}s mle={lin0['mle_sigma2']:.4f} evidence={lin0['evidence_sigma2']:.4f} "
                  f"heldout={lin0['heldout_sigma2']:.4f} crossfit={lin0['crossfit_sigma2']:.4f} "
                  f"nn_heldout={nn0['heldout_sigma2']:.4f} nn_crossfit={nn0['crossfit_sigma2']:.4f} true=1.0000")

            print(f"[main] v_toy1h N={n} seed={seed} (well-specified + misspecified + NN) ...")
            x1h, y1h = td.make_v_toy1h(n=n, seed=seed)
            t0 = time.time()
            lin_ws = run_mechanisms_linear(x1h, y1h, _phi_v_toy1h_wellspecified, sigma_true=td.V_TOY1H_SIGMA, seed=seed)
            lin_mis = run_mechanisms_linear(x1h, y1h, _phi_v_toy1h_misspecified, sigma_true=td.V_TOY1H_SIGMA, seed=seed)
            nn1h = run_mechanisms_nn(x1h, y1h, sigma_true=td.V_TOY1H_SIGMA, seed=seed, device=device)
            wt1 = time.time() - t0
            all_results["v_toy1h_wellspecified"].append({"n": n, "seed": seed, "wall_time_sec": wt1, **lin_ws})
            all_results["v_toy1h_misspecified"].append({"n": n, "seed": seed, "wall_time_sec": wt1, **lin_mis})
            all_results.setdefault("v_toy1h_nn", []).append({"n": n, "seed": seed, "wall_time_sec": wt1, **nn1h})
            print(f"  wall_time={wt1:.1f}s ws_evidence={lin_ws['evidence_sigma2']:.4f} ws_heldout={lin_ws['heldout_sigma2']:.4f} "
                  f"mis_evidence={lin_mis['evidence_sigma2']:.4f} mis_heldout={lin_mis['heldout_sigma2']:.4f} "
                  f"nn_heldout={nn1h['heldout_sigma2']:.4f} nn_crossfit={nn1h['crossfit_sigma2']:.4f} true={td.V_TOY1H_SIGMA**2:.4f}")

    all_results["total_wall_time_sec"] = time.time() - t_start
    summary_path = os.path.join(RESULTS_DIR, "v1_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[main] wrote {summary_path}")
    print(f"[main] total wall time: {all_results['total_wall_time_sec']:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(TORCH_THREADS)

    if args.selftest:
        selftest()
    elif args.probe:
        probe()
    else:
        main()
