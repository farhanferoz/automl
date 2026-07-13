"""Step-2: a per-input held-out arbiter for the variational-EM k-selector.

Step-1 (``probreg_variational_em_step1.py``) judged the WHOLE dataset at once: it asked
whether the genuine mixture beats a single Gaussian on held-out data. Basis B needs the
same question asked *per input* — but each input carries only one target, too noisy to
judge alone. This script builds and validates the machinery that closes that gap, on the
new input-varying toy (``make_toy_c`` / ``make_toy_c_broad``):

  1. Per-point signal. On held-out data, score each point under the bucket mixture and
     under an INDEPENDENTLY trained plain (single-Gaussian) regressor, and take the
     difference  Δ(x_i) = nll_pure(x_i) − nll_bucket(x_i)  (>0 ⇒ buckets help here).
     One point is noisy but unbiased; averaging neighbours in x cancels the noise →
     the binned curve  Δ̂(x).
  2. Gold standard (toy only). Because we control the generator we can draw MANY y at a
     fixed x and average exactly → Δ*(x), the population answer Δ̂(x) is estimating.
     Validating Δ̂(x) ≈ Δ*(x) earns the right to use the one-sample estimate off-toy.

Expected reading. On ``make_toy_c`` the buckets earn held-out credit only past the KNOWN
bimodality boundary spacing = 2 (vertical line); on the variance-matched single-mode
``make_toy_c_broad`` they never do. The fitted bucket COUNT, by contrast, is inflated on
both — that contrast is the whole point.

Note: this runs against the Basis A model (GLOBAL mixing weights), so it validates the
MEASUREMENT, not per-input adaptivity (which needs the Basis B model). Run with any
torch+numpy interpreter:
    python3 automl_package/examples/probreg_variational_em_step2_perinput_arbiter.py
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _kselection_metrics as km
import _toy_datasets as td
import _variational_em as vem

K_MAX = 6
ALPHA0 = 0.1
SIGMA = 0.3
SEP_MIN, SEP_MAX = 0.3, 4.0
N_TR, N_TE = 1500, 4000
N_EPOCHS = 500
SEEDS = [0, 1, 2]
M_GOLD = 1500          # draws per grid point for the gold-standard arbiter
N_GRID = 40            # x grid points for the gold standard
N_BINS = 24            # x bins for the one-sample binned arbiter
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "variational_em_step2_results")

# x at which sep_schedule(x) crosses the bimodality boundary (means 2σ apart).
X_STAR = (2.0 - SEP_MIN) / (SEP_MAX - SEP_MIN)


class CondGaussian(nn.Module):
    """Independent conditional single-Gaussian regressor — the fair 'pure regression' baseline."""

    def __init__(self, in_dim: int, hidden: int = 32, lv_min: float = -8.0, lv_max: float = 4.0) -> None:
        """Builds an ``x -> (mean, log_var)`` MLP with clamped log-variance."""
        super().__init__()
        self.net = vem._MLP(in_dim, 2, hidden)
        self.lv_min, self.lv_max = lv_min, lv_max

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(mean, clamped log_var)``, each shape ``(N,)``."""
        out = self.net(x)
        return out[:, 0], out[:, 1].clamp(self.lv_min, self.lv_max)


def _to_xy(x: np.ndarray, y: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Numpy ``(x, y)`` to float32 tensors of shape ``(N, D)`` and ``(N,)``."""
    x_arr = np.asarray(x, dtype=np.float32)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(-1, 1)
    y_arr = np.asarray(y, dtype=np.float32).ravel()
    return torch.as_tensor(x_arr), torch.as_tensor(y_arr)


def train_cond_gaussian(x: np.ndarray, y: np.ndarray, n_epochs: int = N_EPOCHS, lr: float = 1e-2, hidden: int = 32, seed: int = 0) -> CondGaussian:
    """Fits a :class:`CondGaussian` by full-batch Gaussian NLL (bias seeded at the marginal)."""
    torch.manual_seed(seed)
    x_t, y_t = _to_xy(x, y)
    model = CondGaussian(x_t.shape[1], hidden)
    with torch.no_grad():
        model.net.net[-1].bias[0] = float(y_t.mean())
        model.net.net[-1].bias[1] = float(np.log(max(float(y_t.var()), 1e-4)))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(n_epochs):
        opt.zero_grad()
        mu, lv = model(x_t)
        nll = (0.5 * (math.log(2.0 * math.pi) + lv + (y_t - mu) ** 2 * torch.exp(-lv))).mean()
        nll.backward()
        opt.step()
    return model


@torch.no_grad()
def pure_nll_per_point(model: CondGaussian, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-point held-out NLL under the independent plain regressor."""
    x_t, y_t = _to_xy(x, y)
    mu, lv = model(x_t)
    return (0.5 * (math.log(2.0 * math.pi) + lv + (y_t - mu) ** 2 * torch.exp(-lv))).cpu().numpy()


@torch.no_grad()
def bucket_nll_per_point(model: vem.VariationalEMKSelector, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-point held-out NLL under the bucket mixture (global posterior-mean weights)."""
    x_t, y_t = _to_xy(x, y)
    mu, log_var = model.per_class_params(x_t)
    log_phi = vem.gaussian_log_density(y_t, mu, log_var)
    log_w = torch.log((model.gamma / model.gamma.sum()).clamp_min(1e-12))
    return (-torch.logsumexp(log_w.unsqueeze(0) + log_phi, dim=-1)).cpu().numpy()


def binned_mean(x: np.ndarray, v: np.ndarray, n_bins: int) -> np.ndarray:
    """Mean of ``v`` within each of ``n_bins`` equal-width bins of ``x`` over ``[0, 1]`` (NaN if empty)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    out = np.full(n_bins, np.nan)
    for b in range(n_bins):
        m = idx == b
        if m.any():
            out[b] = float(v[m].mean())
    return out


def run_condition(make_fn, given_fn, label: str) -> dict:
    """Fit both models per seed and return the binned arbiter, gold-standard arbiter, and counts."""
    grid = np.linspace(0.0, 1.0, N_GRID).astype(np.float32)
    bin_curves, gold_curves, surv_eff, surv_k = [], [], [], []
    for seed in SEEDS:
        x_tr, y_tr = make_fn(n=N_TR, seed=seed)
        x_te, y_te = make_fn(n=N_TE, seed=seed + 500)
        bucket = vem.train_variational_em(x_tr, y_tr, k_max=K_MAX, alpha0=ALPHA0, n_epochs=N_EPOCHS, lr=1e-2, seed=seed)
        pure = train_cond_gaussian(x_tr, y_tr, seed=seed)

        delta = pure_nll_per_point(pure, x_te, y_te) - bucket_nll_per_point(bucket, x_te, y_te)
        bin_curves.append(binned_mean(x_te.ravel(), delta, N_BINS))

        gold = np.empty(N_GRID)
        for j, xg in enumerate(grid):
            xs = np.full(M_GOLD, xg, dtype=np.float32)
            ys = given_fn(xs, sigma=SIGMA, sep_min=SEP_MIN, sep_max=SEP_MAX, seed=seed * 100_003 + j)
            gold[j] = float((pure_nll_per_point(pure, xs.reshape(-1, 1), ys) - bucket_nll_per_point(bucket, xs.reshape(-1, 1), ys)).mean())
        gold_curves.append(gold)

        met = km.weight_survival_metrics(bucket.mean_weights())
        surv_eff.append(met["effective_number"])
        surv_k.append(met["surviving_k"])

    bin_arr, gold_arr = np.array(bin_curves), np.array(gold_curves)
    bin_centers = (np.linspace(0.0, 1.0, N_BINS + 1)[:-1] + np.linspace(0.0, 1.0, N_BINS + 1)[1:]) / 2.0
    print(f"  [{label}] global surviving_k median={np.median(surv_k):.0f}  effective_number mean={np.mean(surv_eff):.2f}")
    print(f"  [{label}] gold Δ* at x<x*: {np.nanmean(gold_arr[:, grid < X_STAR]):+.3f} nats   at x>x*: {np.nanmean(gold_arr[:, grid > X_STAR]):+.3f} nats")
    return {
        "label": label,
        "grid": grid.tolist(),
        "bin_centers": bin_centers.tolist(),
        "binned_delta_mean": np.nanmean(bin_arr, axis=0).tolist(),
        "binned_delta_std": np.nanstd(bin_arr, axis=0).tolist(),
        "gold_delta_mean": np.mean(gold_arr, axis=0).tolist(),
        "gold_delta_std": np.std(gold_arr, axis=0).tolist(),
        "global_surviving_k": {"median": float(np.median(surv_k)), "values": surv_k},
        "global_effective_number": {"mean": float(np.mean(surv_eff)), "values": surv_eff},
    }


def _plot(results: list[dict]) -> None:
    """Money plot: binned arbiter Δ̂(x) vs gold standard Δ*(x), with the sep=2 boundary."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(results), figsize=(6.2 * len(results), 4.6), sharey=True)
    if len(results) == 1:
        axes = [axes]
    for ax, r in zip(axes, results, strict=False):
        bc = np.array(r["bin_centers"])
        ax.errorbar(bc, r["binned_delta_mean"], yerr=r["binned_delta_std"], fmt="o", ms=4, capsize=2,
                    color="#1f77b4", label="Δ̂(x): one-sample, neighbour-averaged")
        ax.plot(r["grid"], r["gold_delta_mean"], "-", color="#d62728", lw=2, label="Δ*(x): gold standard (resampled)")
        ax.fill_between(r["grid"], np.array(r["gold_delta_mean"]) - np.array(r["gold_delta_std"]),
                        np.array(r["gold_delta_mean"]) + np.array(r["gold_delta_std"]), color="#d62728", alpha=0.15)
        ax.axhline(0.0, color="k", lw=0.8, ls=":")
        ax.axvline(X_STAR, color="gray", lw=1.2, ls="--", label=f"two-peaks boundary (sep=2, x={X_STAR:.2f})")
        ax.set_title(f"{r['label']}  (fitted count≈{r['global_effective_number']['mean']:.1f})")
        ax.set_xlabel("input x  (low → high mode spacing)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("held-out NLL advantage of buckets (nats)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Per-input arbiter: do the extra buckets actually help on fresh data?", fontsize=12)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "perinput_arbiter.png")
    fig.savefig(path, dpi=130)
    print(f"  saved plot -> {path}")


def main() -> None:
    """Run both conditions, validate the arbiter, save JSON + the money plot."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("== make_toy_c (genuine, structure varies with x) ==")
    c = run_condition(td.make_toy_c, td.sample_toy_c_given_x, "make_toy_c (genuine)")
    print("== make_toy_c_broad (variance-matched single mode: over-chopping trap) ==")
    b = run_condition(td.make_toy_c_broad, td.sample_toy_c_broad_given_x, "make_toy_c_broad (trap)")
    results = [c, b]
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump({"config": {"k_max": K_MAX, "alpha0": ALPHA0, "sigma": SIGMA, "sep": [SEP_MIN, SEP_MAX],
                              "n_tr": N_TR, "n_te": N_TE, "epochs": N_EPOCHS, "seeds": SEEDS, "x_star": X_STAR},
                   "conditions": results}, f, indent=2)
    try:
        _plot(results)
    except Exception as exc:
        print(f"  (plot skipped: {exc})")
    print(f"\nSaved results to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
