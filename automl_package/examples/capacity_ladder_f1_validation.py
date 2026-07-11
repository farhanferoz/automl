"""F1 validation — a fixed-capacity MLP depth sweep on the Toy G / G-flat / H input-varying-
capacity toys (docs/plans/capacity_ladder_2026-07-09/EXECUTION_PLAN.md §3 WS2 F1).

PRE-REGISTERED (written before running, per §0b):
  (Pred) A fixed-capacity sweep of plain MLPs reproduces the qualitative need ordering on Toy G:
  the held-out-NLL-minimizing depth (and the knee, read via `_capacity_ladder.knee` since K0's
  library exists) is HIGHER on the x >= 0 half (compositional region) than the x < 0 half (linear
  region) — the toys must clear this bar before the F2 nested ladder touches them.
  Secondary reads (reported facts-first, not bars — K0's readers are not yet validated for THIS
  toy family, so these are exploratory): Toy G-flat's need ordering should stay FLAT across
  halves (no false capacity signal — the bypass-confound negative control); Toy H's read should
  track the per-half SNR, not structure (its mean function `toy_h_f` is identical on both halves,
  only `toy_h_sigma` differs).

<<< OUTCOME: see the run's stdout + capacity_ladder_results/F1/summary.json >>>

Method: one plain (mean, log_var) MLP per (toy, depth, seed) — depths {1..6}, seeds {0,1,2} — each
trained on a MIXED train split (both halves together; a single net sees the whole domain, as a
real ladder model would), loss = the repo's `nll_loss` (`automl_package/utils/losses.py`, not
reinvented). Depth-c nets for one (toy, seed) plug into `_capacity_ladder.score_table`'s
`Mapping[c, model]` branch via a thin `.log_prob_per_example` adapter (`_FittedDepthModel`),
producing a `(N_val, 6)` held-out log-likelihood table; rows are split into the x<0/x>=0 halves
post hoc (`_capacity_ladder_toys.toy_g_region`). `_capacity_ladder.knee` (K0, already exists — its
bootstrap SE is REUSED, not reimplemented) reads the per-half knee on the POOLED 3-seed table,
block-bootstrapped over seed as the independent unit.

Usage:
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/capacity_ladder_f1_validation.py --selftest
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/capacity_ladder_f1_validation.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root, so `import automl_package` works

import _capacity_ladder as cladder  # noqa: E402
import _capacity_ladder_toys as toys  # noqa: E402

from automl_package.utils.losses import nll_loss  # noqa: E402
from automl_package.utils.pytorch_utils import get_device  # noqa: E402

RESULTS_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "F1")

DEPTHS = [1, 2, 3, 4, 5, 6]
SEEDS = [0, 1, 2]
N_TRAIN = 1000
N_VAL = 500
HIDDEN_SIZE = 24
EPOCHS = 300
LR = 5e-3

TOY_GENERATORS = {
    "G": toys.make_toy_g,
    "G_flat": toys.make_toy_g_flat,
    "H": toys.make_toy_h,
}


class _MLP(nn.Module):
    """Minimal (mean, log_var) MLP: `depth` hidden Linear+ReLU blocks of width `hidden_size`, then
    a linear head to 2 outputs. Depth IS the capacity dial for this sweep — a SEPARATE net is
    trained per depth (independent-weights style; the shared-trunk nested ladder is F2's job, not
    this validation).
    """

    def __init__(self, depth: int, hidden_size: int = HIDDEN_SIZE, input_dim: int = 1):
        super().__init__()
        blocks: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(depth):
            blocks.append(nn.Linear(in_dim, hidden_size))
            blocks.append(nn.ReLU())
            in_dim = hidden_size
        self.trunk = nn.Sequential(*blocks)
        self.head = nn.Linear(in_dim, 2)  # (mean, log_var)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))


def _per_example_log_likelihood(outputs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-example Gaussian held-out log-likelihood: the identical formula `nll_loss` uses before
    its `torch.mean` reduction. `score_table` needs the unreduced `(N,)` vector; `nll_loss` itself
    is used, unmodified, for training.
    """
    mean = outputs[:, 0]
    log_var = outputs[:, 1]
    variance = torch.exp(log_var)
    y = y.squeeze(-1) if y.ndim > 1 else y
    per_sample_nll = 0.5 * (math.log(2 * math.pi) + log_var + ((y - mean) ** 2) / variance)
    return -per_sample_nll


class _FittedDepthModel:
    """Adapter so a trained depth-`d` `_MLP` plugs into `_capacity_ladder.score_table`'s
    `Mapping[c, model]` branch (`.log_prob_per_example` duck-type) — K0 reuse, no bootstrap/knee/EM
    logic reimplemented here.
    """

    def __init__(self, net: nn.Module, device: torch.device):
        self.net = net
        self.device = device

    def log_prob_per_example(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            x_t = torch.as_tensor(np.asarray(x, dtype=np.float32), device=self.device)
            y_t = torch.as_tensor(np.asarray(y, dtype=np.float32), device=self.device)
            ll = _per_example_log_likelihood(self.net(x_t), y_t)
        return ll.cpu().numpy()


def _train_mlp(depth: int, x_train: np.ndarray, y_train: np.ndarray, seed: int, device: torch.device) -> _FittedDepthModel:
    """Trains one depth-`d` MLP full-batch with Adam on `nll_loss`, seed controlling init only (data is pre-drawn)."""
    torch.manual_seed(seed)
    net = _MLP(depth=depth).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    x_t = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    net.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = nll_loss(net(x_t), y_t)
        loss.backward()
        opt.step()
    return _FittedDepthModel(net, device)


def _run_toy_sweep(toy_name: str, make_fn, device: torch.device, unit_time_holder: list[float]) -> tuple[dict, dict]:
    """Trains all (depth, seed) MLPs for one toy; returns (per_depth_seed_nll, knee_results).

    `unit_time_holder` is a 1-element list used to report the FIRST (depth, seed) unit's wall-time
    (the no-unmeasured-time rule) without discarding that unit's model — it is also the real first
    unit of the real sweep, not a throwaway probe.
    """
    per_depth_seed_nll = {"lo": np.full((len(DEPTHS), len(SEEDS)), np.nan), "hi": np.full((len(DEPTHS), len(SEEDS)), np.nan)}
    pooled_scores: dict[str, list[np.ndarray]] = {"lo": [], "hi": []}
    pooled_blocks: dict[str, list[np.ndarray]] = {"lo": [], "hi": []}

    for si, seed in enumerate(SEEDS):
        x_tr, y_tr = make_fn(n=N_TRAIN, seed=seed)
        x_va, y_va = make_fn(n=N_VAL, seed=seed + 200)
        region_va = toys.toy_g_region(x_va.ravel())

        models_by_depth = {}
        for depth in DEPTHS:
            t0 = time.time()
            models_by_depth[depth] = _train_mlp(depth, x_tr, y_tr, seed=seed, device=device)
            elapsed = time.time() - t0
            if not unit_time_holder:
                unit_time_holder.append(elapsed)
                total_units = len(TOY_GENERATORS) * len(DEPTHS) * len(SEEDS)
                print(
                    f"[timing] first unit (toy={toy_name}, depth={depth}, seed={seed}) took {elapsed:.3f}s; "
                    f"extrapolated total for {total_units} units ~= {elapsed * total_units:.1f}s"
                )

        score = cladder.score_table(models_by_depth, x_va, y_va, c_grid=DEPTHS)  # (N_val, 6) held-out log-likelihood
        for half, mask in (("lo", region_va == 0), ("hi", region_va == 1)):
            half_score = score[mask]
            per_depth_seed_nll[half][:, si] = -half_score.mean(axis=0)  # NLL = -log-likelihood
            pooled_scores[half].append(half_score)
            pooled_blocks[half].append(np.full(half_score.shape[0], seed))

    knee_results = {}
    for half in ("lo", "hi"):
        pooled = np.concatenate(pooled_scores[half], axis=0)
        block = np.concatenate(pooled_blocks[half], axis=0)
        # G1: for i.i.d. toy points the correct bootstrap is PLAIN (block=None). Block-by-seed gives
        # only 3 blocks -> hugely inflated SE -> the knee reads spuriously low; kept only for transparency.
        r_star, delta_curve, se = cladder.knee(pooled, ref_c=1, c_grid=DEPTHS, block=None, seed=0)
        r_star_block_seed, _, _ = cladder.knee(pooled, ref_c=1, c_grid=DEPTHS, block=block, seed=0)
        knee_results[half] = {
            "r_star": r_star,
            "r_star_block_seed": r_star_block_seed,
            "delta_curve": delta_curve,
            "se": se,
            "n_pooled": int(pooled.shape[0]),
        }

    return per_depth_seed_nll, knee_results


def main() -> dict:
    device = get_device()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t_start = time.time()

    summary: dict = {
        "device": str(device),
        "depths": DEPTHS,
        "seeds": SEEDS,
        "n_train": N_TRAIN,
        "n_val": N_VAL,
        "hidden_size": HIDDEN_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
    }

    # Headline matched-marginal-variance numbers (the F1 registration proof for Toy G).
    target_var = toys.toy_g_region_variance()
    x_g, y_g = toys.make_toy_g(n=200_000, seed=999)
    region_g = toys.toy_g_region(x_g.ravel())
    var_lo_g = float(np.var(y_g[region_g == 0]))
    var_hi_g = float(np.var(y_g[region_g == 1]))
    summary["toy_g_marginal_variance"] = {"target": target_var, "lo": var_lo_g, "hi": var_hi_g}
    print(f"Toy G marginal variance check: target={target_var:.4f} lo(x<0)={var_lo_g:.4f} hi(x>=0)={var_hi_g:.4f}")

    unit_time_holder: list[float] = []
    for toy_name, make_fn in TOY_GENERATORS.items():
        print(f"\n=== sweeping toy {toy_name} ===")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            per_depth_seed_nll, knee_results = _run_toy_sweep(toy_name, make_fn, device, unit_time_holder)
            cap_sat_warnings = [str(w.message) for w in caught]

        mean_nll_lo = per_depth_seed_nll["lo"].mean(axis=1)
        mean_nll_hi = per_depth_seed_nll["hi"].mean(axis=1)
        argmin_lo = DEPTHS[int(np.argmin(mean_nll_lo))]
        argmin_hi = DEPTHS[int(np.argmin(mean_nll_hi))]

        print(f"  x<0 half  mean NLL by depth: {dict(zip(DEPTHS, np.round(mean_nll_lo, 4).tolist()))}  argmin depth={argmin_lo}")
        print(f"  x>=0 half mean NLL by depth: {dict(zip(DEPTHS, np.round(mean_nll_hi, 4).tolist()))}  argmin depth={argmin_hi}")
        print(f"  knee (pooled 3 seeds, block=None/plain): lo r_star={knee_results['lo']['r_star']}  hi r_star={knee_results['hi']['r_star']}  "
              f"(block=seed underpowered: lo={knee_results['lo']['r_star_block_seed']} hi={knee_results['hi']['r_star_block_seed']})")
        if cap_sat_warnings:
            print(f"  WARNINGS: {cap_sat_warnings}")

        torch.save(
            {
                "toy": toy_name,
                "depths": DEPTHS,
                "seeds": SEEDS,
                "nll_lo": per_depth_seed_nll["lo"],
                "nll_hi": per_depth_seed_nll["hi"],
                "argmin_depth_lo": argmin_lo,
                "argmin_depth_hi": argmin_hi,
                "knee_lo": knee_results["lo"],
                "knee_hi": knee_results["hi"],
            },
            os.path.join(RESULTS_DIR, f"toy_{toy_name.lower()}_nll_table.pt"),
        )

        summary[toy_name] = {
            "mean_nll_lo": mean_nll_lo.tolist(),
            "mean_nll_hi": mean_nll_hi.tolist(),
            "argmin_depth_lo": argmin_lo,
            "argmin_depth_hi": argmin_hi,
            "knee_r_star_lo": knee_results["lo"]["r_star"],
            "knee_r_star_hi": knee_results["hi"]["r_star"],
            "knee_delta_curve_lo": knee_results["lo"]["delta_curve"],
            "knee_delta_curve_hi": knee_results["hi"]["delta_curve"],
            "knee_se_lo": knee_results["lo"]["se"],
            "knee_se_hi": knee_results["hi"]["se"],
            "cap_saturation_warnings": cap_sat_warnings,
        }

    wall_time = time.time() - t_start
    summary["wall_time_sec"] = wall_time
    print(f"\nTotal wall-time: {wall_time:.1f}s")

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ---------------------------------------------------------------------------
# --selftest: trivial known-answer checks, must PASS before any real read.
# ---------------------------------------------------------------------------


def _selftest_toys() -> bool:
    ok = True
    for name, fn in TOY_GENERATORS.items():
        x, y = fn(n=37, seed=0)
        if x.shape != (37, 1) or y.shape != (37,):
            print(f"FAIL toy {name}: bad shapes x={x.shape} y={y.shape}")
            ok = False
    print(f"{'PASS' if ok else 'FAIL'} toy shapes")
    return ok


def _selftest_mlp_forward() -> bool:
    net = _MLP(depth=3)
    x = torch.randn(11, 1)
    out = net(x)
    ok = tuple(out.shape) == (11, 2)
    print(f"{'PASS' if ok else 'FAIL'} MLP forward shape: {tuple(out.shape)} (expected (11, 2))")
    return ok


def _selftest_score_table_knee_integration() -> bool:
    """Known-answer check exercising the EXACT `score_table` + `knee` call path the real sweep
    uses (K0 reuse), without training: a model with a clear, NOISY (not degenerate zero-variance —
    that gives bootstrap SE=0 and defeats the knee's significance test) per-example advantage must
    win over a clearly worse model. Only checks OUR plumbing (adapter + call convention); K0's own
    bootstrap/abstain/saturation behavior is validated by `_capacity_ladder.py`'s own selftest, not
    re-verified here.
    """

    class _NoisyLL:
        def __init__(self, mean_ll: float, noise_sd: float, seed: int):
            self._rng = np.random.default_rng(seed)
            self.mean_ll = mean_ll
            self.noise_sd = noise_sd

        def log_prob_per_example(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
            n = len(np.asarray(y).ravel())
            return self.mean_ll + self.noise_sd * self._rng.standard_normal(n)

    models = {1: _NoisyLL(-5.0, 0.3, seed=1), 2: _NoisyLL(2.0, 0.3, seed=2)}  # 2 has a clear, noisy ~7-nat advantage
    x = np.zeros(300)
    y = np.zeros(300)
    score = cladder.score_table(models, x, y, c_grid=[1, 2])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # cap-saturation warning expected: only 2 columns, advantage is real
        r_star, delta_curve, _se = cladder.knee(score, ref_c=1, c_grid=[1, 2], seed=0)
    ok = r_star == 2 and delta_curve[2] > 5.0
    print(f"{'PASS' if ok else 'FAIL'} score_table+knee integration: r_star={r_star} (expected 2), delta_curve={delta_curve}")
    return ok


def run_selftest() -> bool:
    print("Running F1 validation selftest...")
    checks = [_selftest_toys(), _selftest_mlp_forward(), _selftest_score_table_knee_integration()]
    all_ok = all(checks)
    print("ALL PASS" if all_ok else "SOME FAILED")
    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="Run the trivial known-answer selftest and exit.")
    args = parser.parse_args()
    if args.selftest:
        raise SystemExit(0 if run_selftest() else 1)
    main()
