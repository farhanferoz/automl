"""WS3/V0 -- the joint-NLL variance-collapse pathology, demonstrated and quantified (the WS3 null).

Capacity-ladder program, `docs/plans/capacity_ladder_2026-07-09/EXECUTION_PLAN.md` sec 4. This
task DEMONSTRATES and QUANTIFIES the disease only -- no fixes (V1/V2 own those); this script never
touches `automl_package/models/`.

Pre-registered BEFORE running (verbatim from the plan; outcomes are written up against these, no
post-hoc reframing):

  (P1) On V-toy1 the in-sample sigma_hat ratio (sigma_hat/sigma_true, integrated over x) falls
       materially below 1 as training proceeds, while held-out SSR (standardized squared
       residuals) rises above 1 -- the collapse and its tell.
  (P2) On V-toy0 with the LINEAR model, in-sample MLE sigma_hat^2 is biased by the classical
       factor (N-p)/N, and the closed-form unbiased estimator RSS/(N-p) corrects it exactly.
       (This is the "evidence" estimate in the no-prior limit; the FULL MacKay alpha,beta fixed
       point is deferred to V1 -- NOT implemented here.)

  STOP (hard): if P1 fails (no collapse on V-toy1) -- STOP, write up, do NOT tune. Adjudicator
  (fresh context) matter, not this script's to fix.

The disease mechanism (registered, WS3 sec 4 of the plan): joint (mu, sigma) Gaussian NLL
in-sample lets the mean overfit, shrinking in-sample residuals, so sigma_hat is biased low
exactly proportionally to the overfit. Scored with the repo's own `nll_loss`
(`automl_package/utils/losses.py`) -- the actual disease site, not a reinvented objective.

Usage:
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/capacity_ladder_variance_v0.py --selftest
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/capacity_ladder_variance_v0.py --probe
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/capacity_ladder_variance_v0.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root, so `import automl_package` works
import _toy_datasets as td

from automl_package.utils.losses import nll_loss
from automl_package.utils.pytorch_utils import get_device

RESULTS_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "V0")

SEEDS = [0, 1, 2]
N_GRID = [200, 1000]
HIDDEN_SIZES = (32, 32)  # V-toy1 mean+variance MLP heads (overparameterized relative to N=200)
LR = 1e-3
N_EPOCHS_TOY1 = 8000  # N=200 collapses dramatically by ~8k (measured); N=1000 shows the onset within budget
N_EPOCHS_TOY0 = 3000  # linear model converges far faster; kept modest per the CPU-modest rule
GRID_N = 500  # dense x-grid for the in-sample sigma_hat/sigma_true integral
HELDOUT_N = 2000
TORCH_THREADS = 2  # tiny nets: PyTorch's default thread pool overhead dominates otherwise


class MeanVarNet(nn.Module):
    """Mean + log-variance heads, jointly trained with the repo's `nll_loss`.

    `hidden_sizes=()` gives a genuinely LINEAR mean head (a single `nn.Linear`, no nonlinearity)
    with one GLOBAL homoscedastic log-variance parameter -- the well-specified linear model used
    for V-toy0 (P2). Non-empty `hidden_sizes` gives an MLP mean head with a heteroscedastic MLP
    variance head -- the disease-demonstration model used for V-toy1 (P1).
    """

    def __init__(self, input_dim: int, hidden_sizes: tuple[int, ...] = (32, 32), heteroscedastic: bool = True):
        super().__init__()
        layers: list[nn.Module] = []
        d = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(d, h), nn.Tanh()]
            d = h
        self.trunk = nn.Sequential(*layers) if layers else nn.Identity()
        self.mean_head = nn.Linear(d, 1)
        self.heteroscedastic = heteroscedastic
        if heteroscedastic:
            self.logvar_head = nn.Linear(d, 1)
        else:
            self.log_var_param = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        mean = self.mean_head(h).squeeze(-1)
        log_var = self.logvar_head(h).squeeze(-1) if self.heteroscedastic else self.log_var_param.expand(x.shape[0])
        return torch.stack([mean, log_var], dim=-1)  # matches nll_loss's outputs[:, 0]/outputs[:, 1] contract


def train_meanvar(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_epochs: int,
    x_grid: np.ndarray | None = None,
    sigma_true_grid: np.ndarray | None = None,
    x_heldout: np.ndarray | None = None,
    y_heldout: np.ndarray | None = None,
    hidden_sizes: tuple[int, ...] = (32, 32),
    heteroscedastic: bool = True,
    lr: float = 1e-3,
    seed: int = 0,
    device: torch.device | None = None,
) -> tuple[MeanVarNet, dict[str, np.ndarray]]:
    """Trains a `MeanVarNet` on (x_train, y_train) via full-batch Adam on the repo's `nll_loss`.

    Tracks PER-EPOCH (post-update): train NLL, SSR on train, and -- when the optional probes are
    given -- the in-sample sigma_hat/sigma_true ratio integrated over `x_grid` (needs known
    `sigma_true_grid`) and SSR + NLL on a held-out set (needs `x_heldout`/`y_heldout`).

    Returns:
        (model, curves): curves is a dict of float64 numpy arrays, each of length n_epochs
        (untracked quantities -- e.g. sigma_ratio when no grid is given -- are filled with NaN).
    """
    device = device or get_device()
    torch.manual_seed(seed)
    xt = torch.tensor(x_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.float32, device=device)
    model = MeanVarNet(x_train.shape[1], hidden_sizes=hidden_sizes, heteroscedastic=heteroscedastic).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    track_grid = x_grid is not None and sigma_true_grid is not None
    track_heldout = x_heldout is not None and y_heldout is not None
    xg = torch.tensor(x_grid, dtype=torch.float32, device=device) if track_grid else None
    xh = torch.tensor(x_heldout, dtype=torch.float32, device=device) if track_heldout else None
    yh = torch.tensor(y_heldout, dtype=torch.float32, device=device) if track_heldout else None

    curves = {
        "train_nll": np.zeros(n_epochs, dtype=np.float64),
        "sigma_ratio": np.full(n_epochs, np.nan, dtype=np.float64),
        "ssr_train": np.zeros(n_epochs, dtype=np.float64),
        "ssr_heldout": np.full(n_epochs, np.nan, dtype=np.float64),
        "heldout_nll": np.full(n_epochs, np.nan, dtype=np.float64),
    }

    for epoch in range(n_epochs):
        model.train()
        opt.zero_grad()
        loss = nll_loss(model(xt), yt)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out_tr = model(xt)
            curves["train_nll"][epoch] = float(nll_loss(out_tr, yt))
            curves["ssr_train"][epoch] = float((((yt - out_tr[:, 0]) ** 2) / torch.exp(out_tr[:, 1])).mean())

            if track_grid:
                sigma_hat_grid = torch.exp(0.5 * model(xg)[:, 1]).cpu().numpy()
                curves["sigma_ratio"][epoch] = float(np.mean(sigma_hat_grid / sigma_true_grid))

            if track_heldout:
                out_he = model(xh)
                curves["ssr_heldout"][epoch] = float((((yh - out_he[:, 0]) ** 2) / torch.exp(out_he[:, 1])).mean())
                curves["heldout_nll"][epoch] = float(nll_loss(out_he, yh))

    return model, curves


def ols_variance_bias(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Closed-form OLS on well-specified linear-Gaussian data: the P2 anchor (no gradient descent).

    Returns the fitted coefficients, RSS, the biased in-sample MLE variance RSS/N, and the
    classical unbiased correction RSS/(N-p).
    """
    x64, y64 = x.astype(np.float64), y.astype(np.float64)
    n, p = x64.shape
    w_hat, _, _, _ = np.linalg.lstsq(x64, y64, rcond=None)
    resid = y64 - x64 @ w_hat
    rss = float(np.sum(resid**2))
    return {
        "w_hat": w_hat.tolist(),
        "n": n,
        "p": p,
        "rss": rss,
        "mle_variance": rss / n,
        "unbiased_variance": rss / (n - p),
    }


def run_v_toy1(n: int, seed: int, n_epochs: int = N_EPOCHS_TOY1, device: torch.device | None = None) -> dict:
    """Runs one (seed, N) unit of the V-toy1 disease demonstration; returns summary + curves."""
    x_tr, y_tr = td.make_v_toy1(n=n, seed=seed)
    x_he, y_he = td.make_v_toy1(n=HELDOUT_N, seed=seed + 10_000)  # disjoint seed offset -> fresh draw
    grid = np.linspace(0.0, 1.0, GRID_N).astype(np.float32).reshape(-1, 1)
    sigma_true_grid = td.v_toy1_sigma(grid.ravel())

    t0 = time.time()
    _, curves = train_meanvar(
        x_tr, y_tr, n_epochs,
        x_grid=grid, sigma_true_grid=sigma_true_grid,
        x_heldout=x_he, y_heldout=y_he,
        hidden_sizes=HIDDEN_SIZES, heteroscedastic=True, lr=LR, seed=seed, device=device,
    )
    wall_time = time.time() - t0

    return {
        "toy": "v_toy1", "n": n, "seed": seed, "n_epochs": n_epochs, "wall_time_sec": wall_time,
        "sigma_ratio_final": curves["sigma_ratio"][-1],
        "ssr_train_final": curves["ssr_train"][-1],
        "ssr_heldout_final": curves["ssr_heldout"][-1],
        "heldout_nll_final": curves["heldout_nll"][-1],
        "heldout_nll_min": float(np.nanmin(curves["heldout_nll"])),
        "heldout_nll_min_epoch": int(np.nanargmin(curves["heldout_nll"])),
        "curves": curves,
    }


def run_v_toy0(n: int, seed: int, n_epochs: int = N_EPOCHS_TOY0, device: torch.device | None = None) -> dict:
    """Runs one (seed, N) unit of the V-toy0 linear-Gaussian anchor; returns summary + curves.

    Trains the SAME harness (a genuinely linear `MeanVarNet(hidden_sizes=())`, homoscedastic,
    joint `nll_loss`, gradient descent) alongside the closed-form OLS reference so the gradient
    trajectory's converged in-sample MLE variance can be checked against the analytic
    RSS/N vs RSS/(N-p) bias correction (P2).
    """
    x_tr, y_tr = td.make_v_toy0(n=n, seed=seed)
    x_he, y_he = td.make_v_toy0(n=HELDOUT_N, seed=seed + 10_000)

    t0 = time.time()
    model, curves = train_meanvar(
        x_tr, y_tr, n_epochs,
        x_heldout=x_he, y_heldout=y_he,
        hidden_sizes=(), heteroscedastic=False, lr=LR, seed=seed, device=device,
    )
    wall_time = time.time() - t0

    closed_form = ols_variance_bias(x_tr, y_tr)
    device = device or get_device()
    with torch.no_grad():
        gd_log_var = float(model.log_var_param.item())
    gd_variance = math.exp(gd_log_var)

    return {
        "toy": "v_toy0", "n": n, "seed": seed, "n_epochs": n_epochs, "wall_time_sec": wall_time,
        "closed_form": closed_form,
        "gd_mle_variance": gd_variance,  # gradient-descent-trained joint-NLL in-sample variance
        "ssr_train_final": curves["ssr_train"][-1],
        "ssr_heldout_final": curves["ssr_heldout"][-1],
        "heldout_nll_final": curves["heldout_nll"][-1],
        "curves": curves,
    }


def selftest() -> None:
    """Synthetic known-answer check: on data with a KNOWN CONSTANT sigma, the harness recovers
    sigma_hat approx sigma within tolerance. MUST pass before any real read (per sec 0b).
    """
    device = get_device()
    rng = np.random.default_rng(0)
    n, sigma_true = 2000, 0.7
    x = rng.uniform(-1.0, 1.0, n).reshape(-1, 1).astype(np.float32)
    y = (2.0 * x.ravel() + sigma_true * rng.normal(0.0, 1.0, n)).astype(np.float32)

    model, curves = train_meanvar(
        x, y, n_epochs=1500, hidden_sizes=(16, 16), heteroscedastic=True, lr=1e-2, seed=0, device=device,
    )
    model.eval()
    with torch.no_grad():
        grid = torch.linspace(-1.0, 1.0, 200, device=device).reshape(-1, 1)
        sigma_hat = torch.exp(0.5 * model(grid)[:, 1]).cpu().numpy()
    ratio = float(np.mean(sigma_hat / sigma_true))
    print(f"[selftest] recovered sigma_hat/sigma_true ratio = {ratio:.4f} (tolerance: within 15% of 1.0)")
    assert 0.85 < ratio < 1.15, f"selftest FAILED: sigma_hat/sigma_true ratio {ratio:.4f} outside [0.85, 1.15]"
    assert curves["ssr_train"][-1] > 0 and np.isfinite(curves["ssr_train"][-1])
    print("[selftest] PASS")


def probe() -> None:
    """No-unmeasured-time rule (sec 0b): run ONE unit per run shape, measure wall-time, extrapolate."""
    device = get_device()
    print(f"[probe] device={device}")
    shapes = [
        ("v_toy1", 200, run_v_toy1, N_EPOCHS_TOY1),
        ("v_toy1", 1000, run_v_toy1, N_EPOCHS_TOY1),
        ("v_toy0", 200, run_v_toy0, N_EPOCHS_TOY0),
        ("v_toy0", 1000, run_v_toy0, N_EPOCHS_TOY0),
    ]
    total_est = 0.0
    for toy, n, fn, n_epochs in shapes:
        result = fn(n, seed=0, n_epochs=n_epochs, device=device)
        wt = result["wall_time_sec"]
        print(f"[probe] {toy} N={n} n_epochs={n_epochs}: wall_time={wt:.1f}s")
        total_est += wt * len(SEEDS)  # 3 seeds per (toy, N) shape
    print(f"[probe] extrapolated full-matrix time (3 seeds x 2 toys x {len(N_GRID)} N): {total_est:.1f}s ({total_est / 60:.1f} min)")


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = get_device()
    print(f"[main] device={device}")

    all_results: dict = {"config": {
        "seeds": SEEDS, "n_grid": N_GRID, "hidden_sizes": HIDDEN_SIZES,
        "lr": LR, "n_epochs_toy1": N_EPOCHS_TOY1, "n_epochs_toy0": N_EPOCHS_TOY0,
        "grid_n": GRID_N, "heldout_n": HELDOUT_N,
    }, "v_toy1": [], "v_toy0": []}

    t_start = time.time()
    for n in N_GRID:
        for seed in SEEDS:
            print(f"[main] v_toy1 N={n} seed={seed} ...")
            res = run_v_toy1(n, seed, device=device)
            curve_path = os.path.join(RESULTS_DIR, f"v_toy1_N{n}_seed{seed}_curves.pt")
            torch.save(res["curves"], curve_path)
            summary = {k: v for k, v in res.items() if k != "curves"}
            summary["curve_path"] = curve_path
            all_results["v_toy1"].append(summary)
            print(f"  wall_time={summary['wall_time_sec']:.1f}s sigma_ratio_final={summary['sigma_ratio_final']:.4f} "
                  f"ssr_train_final={summary['ssr_train_final']:.4f} ssr_heldout_final={summary['ssr_heldout_final']:.4f}")

            print(f"[main] v_toy0 N={n} seed={seed} ...")
            res0 = run_v_toy0(n, seed, device=device)
            curve_path0 = os.path.join(RESULTS_DIR, f"v_toy0_N{n}_seed{seed}_curves.pt")
            torch.save(res0["curves"], curve_path0)
            summary0 = {k: v for k, v in res0.items() if k != "curves"}
            summary0["curve_path"] = curve_path0
            all_results["v_toy0"].append(summary0)
            print(f"  wall_time={summary0['wall_time_sec']:.1f}s gd_mle_variance={summary0['gd_mle_variance']:.4f} "
                  f"closed_form_mle={summary0['closed_form']['mle_variance']:.4f} "
                  f"closed_form_unbiased={summary0['closed_form']['unbiased_variance']:.4f}")

    all_results["total_wall_time_sec"] = time.time() - t_start
    summary_path = os.path.join(RESULTS_DIR, "v0_summary.json")
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
