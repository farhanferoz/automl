"""X1: hierarchical partial-pooled per-bin stacking (WS1/WS2 capacity-ladder follow-up, task X1).

(docs/plans/capacity_ladder_2026-07-09/EXECUTION_PLAN.md §8.5 X1 / W8; pre-registration in
 capacity_ladder_results/X1/PREREGISTRATION.md)

`_capacity_ladder.perbin_stack` runs `stack_em` INDEPENDENTLY per bin (no partial pooling — the
W7 correction). With ~83 held-out points per tercile that per-bin stack is high-variance, which is
what left X3's per-input depth read at the noise floor. X1 replaces it with a hierarchical stacker
that lets the DATA decide how much each bin borrows strength from the global stack:

    theta_b ~ Normal(mu, tau^2 * I),   pi_b = softmax(theta_b)

fit by MAP-EM on the fit half: theta-step = gradient ascent on the held-out-style mixture log-score
+ the Gaussian prior's own term (coeff 1); mu = mean_b theta_b (closed form); tau^2 =
mean_{b,c}(theta_bc - mu_c)^2 (empirical-Bayes closed form — a global low-dim nuisance FITTED by
evidence, §0 B6, NOT a tuned lambda). tau->0 pools to the global stack; large tau frees the bins.

It slots into the SAME repeated-cross-fit harness X3 used (50 random fit/score splits, Nadeau-
Bengio split-aware SE), as a third arm beside the global stack and the independent per-bin stack,
on the SAME F2 nested-depth tables / toys G, G_flat, H. Bars (PREREGISTRATION.md): (1) hierarchical
beats independent per-bin on held-out mixture log-score on G (pooling helps); (2) does pooling lift
G's per-input advantage above the G_flat control (power-limited) or not (genuinely absent — X3's
closure stands on the strongest estimator); (3) G_flat still abstains (no manufactured structure).

Usage:
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/capacity_ladder_x1.py --selftest
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/capacity_ladder_x1.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root, so `import automl_package` works

import _capacity_ladder as cl  # noqa: E402
from capacity_ladder_f3 import SEEDS, TOYS, _jsonable, load_f2_table  # noqa: E402

OUT_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "X1")

N_SPLITS = 50
N_BINS = 3  # terciles, the F3/X3 per-bin convention
TORCH_THREADS = 2

# MAP-EM optimiser settings for the hierarchical stacker (tiny (B, C) problem).
_HIER_OUTER = 12  # closed-form mu/tau^2 updates
_HIER_INNER = 60  # Adam theta steps per outer round
_HIER_LR = 0.05
_TAU2_FLOOR = 1e-4


# ---------------------------------------------------------------------------
# The one new estimator: hierarchical partial-pooled per-bin stacking.
# ---------------------------------------------------------------------------


def hierarchical_perbin_stack(score: np.ndarray, bins: np.ndarray) -> dict[int, np.ndarray]:
    """Partially-pooled per-bin stacking weights via a hierarchical logit-Gaussian prior (MAP-EM).

    Drop-in for `_capacity_ladder.perbin_stack`, but the per-bin weight vectors share a hierarchical
    prior `theta_b ~ N(mu, tau^2 I)` (pi_b = softmax(theta_b)) whose pooling scale tau^2 is fitted by
    empirical Bayes, so a bin with little data is shrunk toward the global stack. Fit on `score`/`bins`
    only (the caller's fit half). Strictly probabilistic: the prior is the model's own term (coeff 1),
    tau^2 is estimated, no tuned penalty.

    Args:
        score: `(N, C)` held-out log-likelihood table (the fit half).
        bins: `(N,)` integer bin id per row (e.g. from `_capacity_ladder.quantile_bins`).

    Returns:
        dict `bin_id -> (C,) pi_b`, one partially-pooled stacking vector per bin.
    """
    s = torch.as_tensor(np.asarray(score, dtype=np.float64))
    bins_arr = np.asarray(bins)
    unique_bins = [int(b) for b in np.unique(bins_arr)]
    bin_index = {b: j for j, b in enumerate(unique_bins)}
    row_bin = torch.as_tensor(np.array([bin_index[int(b)] for b in bins_arr]), dtype=torch.long)
    n = s.shape[0]
    big_b = len(unique_bins)

    # Init every bin's logits at the global-stack logits (a strong, sensible starting point).
    global_pi = cl.stack_em(score)
    init_logit = torch.log(torch.as_tensor(np.clip(global_pi, 1e-12, None), dtype=torch.float64))
    theta = init_logit.unsqueeze(0).repeat(big_b, 1).clone().requires_grad_(True)
    mu = init_logit.clone()
    tau2 = torch.tensor(1.0, dtype=torch.float64)
    opt = torch.optim.Adam([theta], lr=_HIER_LR)

    for _ in range(_HIER_OUTER):
        for _ in range(_HIER_INNER):
            opt.zero_grad()
            log_pi = torch.log_softmax(theta, dim=1)  # (B, C)
            logmix = torch.logsumexp(log_pi[row_bin] + s, dim=1)  # (N,)
            prior = ((theta - mu) ** 2).sum() / (2.0 * tau2)
            loss = -(logmix.sum() - prior) / n  # /n keeps Adam well-scaled; preserves data/prior balance
            loss.backward()
            opt.step()
        with torch.no_grad():
            mu = theta.mean(dim=0)
            tau2 = ((theta - mu) ** 2).mean().clamp_min(_TAU2_FLOOR)

    pi = torch.softmax(theta.detach(), dim=1).numpy()
    return {b: pi[bin_index[b]] for b in unique_bins}


# ---------------------------------------------------------------------------
# Three-arm split: global vs independent per-bin vs hierarchical per-bin.
# ---------------------------------------------------------------------------


def _perbin_logscore(pi_bins: dict[int, np.ndarray], pi_global: np.ndarray, score_s: np.ndarray, bins_s: np.ndarray) -> np.ndarray:
    """Per-example held-out mixture log-score using `pi_bins[b]` per bin, falling back to `pi_global`."""
    out = np.empty(score_s.shape[0], dtype=np.float64)
    for b in np.unique(bins_s):
        mask = bins_s == b
        pi_b = pi_bins.get(int(b), pi_global)
        out[mask] = cl.mixture_logscore(pi_b, score_s[mask])
    return out


def three_arm_split(score: np.ndarray, x: np.ndarray, n_bins: int, split_seed: int) -> dict:
    """One fit/score split; returns per-split mean advantages (indep vs global, hier vs global, hier vs indep)."""
    n = score.shape[0]
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n)
    half = n // 2
    fit, sc = perm[:half], perm[half:]
    score_fit, x_fit = score[fit], x[fit]
    score_s, x_s = score[sc], x[sc]

    pi_global = cl.stack_em(score_fit)
    bins_fit = cl.quantile_bins(x_fit, n_bins)
    pi_indep = cl.perbin_stack(score_fit, bins_fit)
    pi_hier = hierarchical_perbin_stack(score_fit, bins_fit)

    bins_s = cl.quantile_bins(x_s, n_bins)
    global_ls = cl.mixture_logscore(pi_global, score_s)
    indep_ls = _perbin_logscore(pi_indep, pi_global, score_s, bins_s)
    hier_ls = _perbin_logscore(pi_hier, pi_global, score_s, bins_s)

    def _stat(diff: np.ndarray, boot_seed: int) -> dict:
        return {"mean": float(diff.mean()), "se": _boot_se(diff, boot_seed), "n": int(diff.size)}

    return {
        "indep_adv": _stat(indep_ls - global_ls, 1),
        "hier_adv": _stat(hier_ls - global_ls, 2),
        "hier_vs_indep": _stat(hier_ls - indep_ls, 3),
        "n_fit": int(half),
        "n_score": int(n - half),
    }


def _boot_se(vec: np.ndarray, seed: int, n_boot: int = 1000) -> float:
    """Plain i.i.d. bootstrap SE of a 1-D vector's mean (reuses `_capacity_ladder._bootstrap_col_means`)."""
    rng = np.random.default_rng(seed)
    boot = cl._bootstrap_col_means(vec.reshape(-1, 1), n_boot, None, rng)
    return float(boot[:, 0].std(ddof=1))


# ---------------------------------------------------------------------------
# Nadeau-Bengio aggregation over splits (identical estimator to X3).
# ---------------------------------------------------------------------------


def _aggregate(split_stats: list[dict], key: str, n_splits: int) -> dict:
    """Nadeau-Bengio split-aware aggregate of one advantage across splits (as X3)."""
    means = np.array([r[key]["mean"] for r in split_stats], dtype=np.float64)
    beats = np.array([r[key]["mean"] > 2.0 * r[key]["se"] for r in split_stats])
    rho = split_stats[0]["n_score"] / split_stats[0]["n_fit"]
    var_s = float(np.var(means, ddof=1))
    mu_bar = float(means.mean())
    se_nb = float(np.sqrt(max(0.0, (1.0 / n_splits + rho) * var_s)))
    return {
        "mu_bar": mu_bar,
        "se_nadeau_bengio": se_nb,
        "t_corrected": (mu_bar / se_nb if se_nb > 0 else float("inf")),
        "split_pass_fraction": float(beats.mean()),
        "sd_across_splits": float(np.sqrt(var_s)),
    }


def run_repeated(score: np.ndarray, x: np.ndarray, n_splits: int = N_SPLITS, n_bins: int = N_BINS) -> dict:
    """Runs `three_arm_split` over `n_splits` splits and aggregates each advantage (Nadeau-Bengio)."""
    stats = [three_arm_split(score, x, n_bins, s) for s in range(n_splits)]
    return {
        "n_splits": n_splits,
        "indep_vs_global": _aggregate(stats, "indep_adv", n_splits),
        "hier_vs_global": _aggregate(stats, "hier_adv", n_splits),
        "hier_vs_indep": _aggregate(stats, "hier_vs_indep", n_splits),
    }


# ---------------------------------------------------------------------------
# Selftest.
# ---------------------------------------------------------------------------


def _synthetic_table(peaks: tuple[int, int, int], rng: np.random.Generator, n_per: int, noise_sd: float) -> tuple[np.ndarray, np.ndarray]:
    """Three equal x-regions whose score tables peak at `peaks` (reuses `cl._peaked_table`)."""
    x = np.concatenate([rng.uniform(-1.0, -1 / 3, n_per), rng.uniform(-1 / 3, 1 / 3, n_per), rng.uniform(1 / 3, 1.0, n_per)])
    score = np.concatenate([cl._peaked_table(n_per, peak_c=pc, gap=2.0, noise_sd=noise_sd, rng=rng) for pc in peaks], axis=0)
    return score, x


def run_selftest() -> bool:
    """Synthetic known-answer checks for the hierarchical stacker.

    (a) real per-bin structure at FEW points/bin -> hierarchical RECOVERS it over global where the
    independent per-bin stack cannot (hier_vs_global t > 2 and > indep_vs_global t), and is better on
    average (hier_vs_indep mu > 0). (The direct hier_vs_indep t is deliberately NOT gated: the
    independent arm is erratic at few pts/bin, so the split-to-split difference is high-variance even
    when pooling clearly helps -- the robust signal is recovery-over-global.)
    (b) no-structure table -> hierarchical ~ global (advantage ~0, abstains) and does not underperform
    independent (pooling never hurts).
    """
    ok = True
    rng = np.random.default_rng(0)

    # (a) genuine per-region structure, few points/bin, noisy -> pooling should recover it.
    pos_score, pos_x = _synthetic_table((1, 3, 6), rng, n_per=60, noise_sd=1.0)
    agg_pos = run_repeated(pos_score, pos_x, n_splits=15)
    hvi, hvg, ivg = agg_pos["hier_vs_indep"], agg_pos["hier_vs_global"], agg_pos["indep_vs_global"]
    ok_a = hvg["t_corrected"] > 2.0 and hvg["t_corrected"] > ivg["t_corrected"] and hvi["mu_bar"] > 0
    print(f"[x1 selftest] (a) structured, few pts/bin: hier_vs_global t={hvg['t_corrected']:.2f} (recovers) "
          f"indep_vs_global t={ivg['t_corrected']:.2f} (can't)  hier_vs_indep mu={hvi['mu_bar']:+.4f}  helps={ok_a}")
    ok = ok and ok_a

    # (b) no per-bin structure (same peak everywhere) -> hierarchical must not manufacture an advantage.
    neg_score, neg_x = _synthetic_table((3, 3, 3), rng, n_per=60, noise_sd=1.0)
    agg_neg = run_repeated(neg_score, neg_x, n_splits=15)
    hvg = agg_neg["hier_vs_global"]
    hvi_n = agg_neg["hier_vs_indep"]
    ok_b = hvg["t_corrected"] <= 2.0 and hvi_n["mu_bar"] >= -1e-3
    print(f"[x1 selftest] (b) no structure: hier_vs_global mu={hvg['mu_bar']:+.4f} t={hvg['t_corrected']:.2f} (must abstain)  "
          f"hier_vs_indep mu={hvi_n['mu_bar']:+.4f} (>=0, pooling never hurts)  ok={ok_b}")
    ok = ok and ok_b

    print(f"[x1 selftest] {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Real reader.
# ---------------------------------------------------------------------------


def _run_reader(n_splits: int = N_SPLITS) -> None:
    """Runs the three-arm repeated cross-fit on every available F2 table; writes `x1_summary.json`."""
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()

    tables = {}
    for toy in TOYS:
        for seed in SEEDS:
            t = load_f2_table(toy, seed)
            if t is not None:
                tables[(toy, seed)] = t
    if not tables:
        print("no F2 tables available -- run capacity_ladder_f2.py first.")
        return

    per_case = []
    for (toy, seed), t in sorted(tables.items()):
        agg = run_repeated(t["score"], t["x"], n_splits=n_splits)
        per_case.append({"toy": toy, "seed": seed, "n": int(t["score"].shape[0]), "agg": agg})
        print(f"[x1] {toy} s{seed}: hier_vs_indep t={agg['hier_vs_indep']['t_corrected']:+.2f}  "
              f"hier_vs_global t={agg['hier_vs_global']['t_corrected']:+.2f} (pass={agg['hier_vs_global']['split_pass_fraction']:.2f})  "
              f"indep_vs_global t={agg['indep_vs_global']['t_corrected']:+.2f}")

    def _pool(toy: str, arm: str) -> dict:
        rows = [r for r in per_case if r["toy"] == toy]
        t_by_seed = [r["agg"][arm]["t_corrected"] for r in rows]
        mu_by_seed = [r["agg"][arm]["mu_bar"] for r in rows]
        pass_by_seed = [r["agg"][arm]["split_pass_fraction"] for r in rows]
        return {
            "t_by_seed": t_by_seed,
            "mu_by_seed": mu_by_seed,
            "mean_pass_fraction": float(np.mean(pass_by_seed)),
            "n_seeds_t_gt_2": int(sum(t > 2.0 for t in t_by_seed)),
        }

    verdicts = {}
    if any(r["toy"] == "G" for r in per_case):
        g_hvi, g_hvg = _pool("G", "hier_vs_indep"), _pool("G", "hier_vs_global")
        gflat_hvg = _pool("G_flat", "hier_vs_global") if any(r["toy"] == "G_flat" for r in per_case) else None
        verdicts["pooling_helps_on_G"] = {"hier_vs_indep": g_hvi, "pass": bool(g_hvi["n_seeds_t_gt_2"] >= 2 and np.mean(g_hvi["mu_by_seed"]) > 0)}
        verdicts["signal_recovery_on_G"] = {
            "G_hier_vs_global": g_hvg,
            "Gflat_hier_vs_global": gflat_hvg,
            "power_limited_signal_recovered": bool(
                g_hvg["n_seeds_t_gt_2"] >= 2 and gflat_hvg is not None and g_hvg["mean_pass_fraction"] - gflat_hvg["mean_pass_fraction"] > 0.2
            ),
        }
        if gflat_hvg is not None:
            verdicts["no_false_positive_on_Gflat"] = {"hier_vs_global": gflat_hvg, "holds": bool(gflat_hvg["n_seeds_t_gt_2"] == 0 and gflat_hvg["mean_pass_fraction"] < 0.2)}

    wall = time.time() - t_start
    summary = {"task": "X1", "n_splits": n_splits, "per_case": per_case, "verdicts": verdicts, "wall_time_sec": wall}
    out_path = os.path.join(OUT_DIR, "x1_summary.json")
    with open(out_path, "w") as f:
        json.dump(_jsonable(summary), f, indent=2)
    print(f"\n[x1] wrote {out_path}  ({wall:.0f}s)")
    print("[x1] VERDICTS:", json.dumps(_jsonable(verdicts), indent=2))


def main() -> None:
    """Parses args and runs the selftest or the real three-arm repeated cross-fit read."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="Synthetic known-answer check (pooling helps at few pts/bin, abstains on no structure).")
    parser.add_argument("--n-splits", type=int, default=N_SPLITS, help="Number of random fit/score splits (default 50).")
    args = parser.parse_args()

    torch.set_num_threads(TORCH_THREADS)
    if args.selftest:
        sys.exit(0 if run_selftest() else 1)
    _run_reader(n_splits=args.n_splits)


if __name__ == "__main__":
    main()
