"""Step-1 validation of the variational-EM k-selector (Basis A grounding).

Runs the controlled test of docs/kselection_variational_em_2026-06-13 (note §9) on the
constant-baseline regime, where the data are i.i.d. draws from one fixed mixture (the
setting the α₀ pruning theory applies to). The bypass is ON throughout.

  Check 1 (positive sweep): on a 2-mode mixture with the modes held in place, sweep the
    spacing (in noise-widths). Surviving k should rise from ~1 (merged) toward the true
    number as the modes resolve; the bypass should hand over to the bins.
  Check 2 (prior is weak, not a knob): at a resolved spacing, vary α₀ inside the pruning
    regime; surviving k should be stable.
  Check 3 (full shape matters): bimodal vs a broad unimodal with the SAME mean and variance.
    Arbiter is held-out NLL — the genuine mixture should beat a single Gaussian (the summary
    model) on the two-peaked data and not on the broad bell. The surviving-k count is reported
    only as a caveated resolution diagnostic (it over-counts both via bin tiling; see FINDINGS.md).
  Negative control: smooth unimodal data (x informative) — surviving k should stay ~1 with
    the bypass carrying the weight (the method must not invent classes).

Results (JSON + a short markdown note) are written under
``variational_em_step1_results/``. Run with any torch+numpy interpreter:
    python3 automl_package/examples/probreg_variational_em_step1.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _kselection_metrics as km  # noqa: E402
import _toy_datasets as td  # noqa: E402
import _variational_em as vem  # noqa: E402

K_MAX = 6
N = 800
SIGMA = 0.3
N_EPOCHS = 500
SEEDS = [0, 1, 2, 3, 4]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "variational_em_step1_results")


def _fit_model(x: np.ndarray, y: np.ndarray, alpha0: float, seed: int) -> vem.VariationalEMKSelector:
    """Fit one variational-EM model on ``(x, y)`` with the shared run config."""
    return vem.train_variational_em(x, y, k_max=K_MAX, alpha0=alpha0, n_epochs=N_EPOCHS, lr=1e-2, seed=seed)


def _fit_survivors(x: np.ndarray, y: np.ndarray, alpha0: float, seed: int) -> dict:
    """Fit one model and return its weight-survival metrics."""
    return km.weight_survival_metrics(_fit_model(x, y, alpha0, seed).mean_weights())


def _single_gaussian_nll(y_tr: np.ndarray, y_te: np.ndarray) -> float:
    """Held-out NLL of one Gaussian fit by moments on ``y_tr`` — the summary (matched mean & variance) model."""
    mu = float(np.mean(y_tr))
    var = float(max(np.var(y_tr), 1e-8))
    y = np.asarray(y_te, dtype=np.float64).ravel()
    return float(np.mean(0.5 * (np.log(2.0 * np.pi * var) + (y - mu) ** 2 / var)))


def _heldout_nlls(model: vem.VariationalEMKSelector, x_te: np.ndarray, y_te: np.ndarray) -> tuple[float, float]:
    """Held-out ``(mixture NLL, bypass-only NLL)`` for a fitted model on fresh data.

    The mixture NLL scores the genuine blend of probabilities (the principled objective); the
    bypass-only NLL scores the single direct-regression Gaussian for reference.
    """
    x_t = torch.as_tensor(np.asarray(x_te, dtype=np.float32).reshape(len(x_te), -1))
    y_t = torch.as_tensor(np.asarray(y_te, dtype=np.float32).ravel())
    with torch.no_grad():
        mixture = model.mixture_nll(x_t, y_t)
        mu, log_var = model.per_class_params(x_t)
        log_phi = vem.gaussian_log_density(y_t, mu, log_var)
        bypass = float((-log_phi[:, -1]).mean().item())
    return mixture, bypass


def _agg(values: list[float]) -> dict:
    """Mean / median / min / max of a list."""
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "median": float(np.median(a)), "min": float(a.min()), "max": float(a.max())}


def check1_separation_sweep() -> list[dict]:
    """Surviving k vs mode spacing (2 true modes, constant baseline)."""
    rows = []
    for sep in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]:
        sk, eff, byp = [], [], []
        for s in SEEDS:
            x, y = td.make_toy_b(n=N, k_true=2, separation=sep, sigma=SIGMA, baseline="zero", seed=100 + s)
            met = _fit_survivors(x, y, alpha0=0.1, seed=s)
            sk.append(met["surviving_k"]); eff.append(met["effective_number"]); byp.append(met["bypass_weight"])
        rows.append({"separation": sep, "surviving_k": _agg(sk), "effective_number": _agg(eff), "bypass_weight": _agg(byp)})
        print(f"  sep={sep:>4}: surv_k median={np.median(sk):.0f} mean={np.mean(sk):.2f}  bypass_w mean={np.mean(byp):.3f}")
    return rows


def check2_alpha_insensitivity() -> list[dict]:
    """Surviving k at a resolved spacing, across α₀ in the pruning regime."""
    rows = []
    inv_k = 1.0 / (K_MAX + 1)
    for a0 in [0.05, 0.1, 0.2, 0.5, inv_k]:
        sk = []
        for s in SEEDS:
            x, y = td.make_toy_b(n=N, k_true=2, separation=4.0, sigma=SIGMA, baseline="zero", seed=100 + s)
            met = _fit_survivors(x, y, alpha0=a0, seed=s)
            sk.append(met["surviving_k"])
        rows.append({"alpha0": round(a0, 4), "surviving_k": _agg(sk)})
        print(f"  alpha0={a0:.3f}: surv_k median={np.median(sk):.0f} mean={np.mean(sk):.2f}")
    return rows


def check3_shape_pair() -> dict:
    """Bimodal vs matched-variance broad unimodal, judged by held-out NLL.

    The surviving-k count over-counts both shapes (bin tiling; see FINDINGS.md), so the
    principled arbiter is held-out NLL: the genuine mixture should beat a single Gaussian
    (the summary model, matched mean & variance) on the two-peaked data and NOT on the
    broad bell. Train and test are independent draws from the same fixed generators.
    """
    sep = 4.0
    bm_sk, br_sk = [], []
    bm_mix, bm_sg, bm_byp = [], [], []
    br_mix, br_sg, br_byp = [], [], []
    bm_stats, br_stats = [], []
    for s in SEEDS:
        xb, yb = td.make_toy_b(n=N, k_true=2, separation=sep, sigma=SIGMA, baseline="zero", seed=200 + s)
        xr, yr = td.make_broad_unimodal(n=N, separation=sep, sigma=SIGMA, baseline="zero", seed=300 + s)
        xb_te, yb_te = td.make_toy_b(n=N, k_true=2, separation=sep, sigma=SIGMA, baseline="zero", seed=2200 + s)
        xr_te, yr_te = td.make_broad_unimodal(n=N, separation=sep, sigma=SIGMA, baseline="zero", seed=3300 + s)
        mb = _fit_model(xb, yb, alpha0=0.1, seed=s)
        mr = _fit_model(xr, yr, alpha0=0.1, seed=s)
        bm_sk.append(km.weight_survival_metrics(mb.mean_weights())["surviving_k"])
        br_sk.append(km.weight_survival_metrics(mr.mean_weights())["surviving_k"])
        mix, byp = _heldout_nlls(mb, xb_te, yb_te); bm_mix.append(mix); bm_byp.append(byp); bm_sg.append(_single_gaussian_nll(yb, yb_te))
        mix, byp = _heldout_nlls(mr, xr_te, yr_te); br_mix.append(mix); br_byp.append(byp); br_sg.append(_single_gaussian_nll(yr, yr_te))
        bm_stats.append((float(yb.mean()), float(yb.var())))
        br_stats.append((float(yr.mean()), float(yr.var())))
    bm_mean = np.mean([m for m, _ in bm_stats]); bm_var = np.mean([v for _, v in bm_stats])
    br_mean = np.mean([m for m, _ in br_stats]); br_var = np.mean([v for _, v in br_stats])
    adv_bimodal = float(np.mean(bm_sg) - np.mean(bm_mix))  # mixture's held-out edge over the summary model (nats)
    adv_broad = float(np.mean(br_sg) - np.mean(br_mix))
    print(f"  bimodal surv_k median={np.median(bm_sk):.0f}  broad surv_k median={np.median(br_sk):.0f}")
    print(f"  bimodal (mean,var)=({bm_mean:.2f},{bm_var:.2f})  broad (mean,var)=({br_mean:.2f},{br_var:.2f})")
    print(f"  held-out NLL  bimodal: mix={np.mean(bm_mix):.3f} single_gauss={np.mean(bm_sg):.3f} (mixture edge {adv_bimodal:+.3f})")
    print(f"  held-out NLL  broad:   mix={np.mean(br_mix):.3f} single_gauss={np.mean(br_sg):.3f} (mixture edge {adv_broad:+.3f})")
    return {
        "bimodal_surviving_k": _agg(bm_sk),
        "broad_surviving_k": _agg(br_sk),
        "bimodal_empirical_mean_var": [bm_mean, bm_var],
        "broad_empirical_mean_var": [br_mean, br_var],
        "bimodal_heldout_nll": {"mixture": _agg(bm_mix), "single_gaussian": _agg(bm_sg), "bypass_only": _agg(bm_byp)},
        "broad_heldout_nll": {"mixture": _agg(br_mix), "single_gaussian": _agg(br_sg), "bypass_only": _agg(br_byp)},
        "mixture_advantage_nats": {"bimodal": adv_bimodal, "broad": adv_broad},
    }


def negative_control() -> list[dict]:
    """Smooth unimodal (x informative): surviving k should stay ~1, bypass dominant."""
    rows = []
    for noise in [0.05, 0.1, 0.2, 0.4]:
        sk, byp = [], []
        for s in SEEDS:
            x, y = td.make_toy_a(n=N, sigma=noise, seed=100 + s)
            met = _fit_survivors(x, y, alpha0=0.1, seed=s)
            sk.append(met["surviving_k"]); byp.append(met["bypass_weight"])
        rows.append({"noise": noise, "surviving_k": _agg(sk), "bypass_weight": _agg(byp)})
        print(f"  noise={noise}: surv_k median={np.median(sk):.0f}  bypass_w mean={np.mean(byp):.3f}")
    return rows


def main() -> None:
    """Run all four checks, save JSON + a markdown summary."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("== Check 1: separation sweep =="); c1 = check1_separation_sweep()
    print("== Check 2: alpha0 insensitivity =="); c2 = check2_alpha_insensitivity()
    print("== Check 3: shape pair =="); c3 = check3_shape_pair()
    print("== Negative control: smooth =="); nc = negative_control()

    results = {
        "config": {"k_max": K_MAX, "n": N, "sigma": SIGMA, "n_epochs": N_EPOCHS, "seeds": SEEDS},
        "check1_separation_sweep": c1,
        "check2_alpha_insensitivity": c2,
        "check3_shape_pair": c3,
        "negative_control_smooth": nc,
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    _write_markdown(results)
    print(f"\nSaved results to {RESULTS_DIR}/")


def _write_markdown(r: dict) -> None:
    """Render a short human-readable results note."""
    lines = ["# Variational-EM k-selector — Step-1 (Basis A grounding) results", ""]
    lines.append(f"Config: k_max={r['config']['k_max']} (K={r['config']['k_max']+1} incl bypass), "
                 f"n={r['config']['n']}, sigma={r['config']['sigma']}, epochs={r['config']['n_epochs']}, "
                 f"{len(r['config']['seeds'])} seeds. Constant baseline (fixed mixture), bypass ON.")
    lines += ["", "## Check 1 — surviving k vs mode spacing (resolvability)", "",
              "| spacing (÷σ) | surviving k (median) | surviving k (mean) | bypass weight (mean) |",
              "|---:|---:|---:|---:|"]
    for row in r["check1_separation_sweep"]:
        lines.append(f"| {row['separation']} | {row['surviving_k']['median']:.0f} | "
                     f"{row['surviving_k']['mean']:.2f} | {row['bypass_weight']['mean']:.3f} |")
    lines += ["", "## Check 2 — α₀ insensitivity (resolved, separation=4)", "",
              "| α₀ | surviving k (median) | surviving k (mean) |", "|---:|---:|---:|"]
    for row in r["check2_alpha_insensitivity"]:
        lines.append(f"| {row['alpha0']} | {row['surviving_k']['median']:.0f} | {row['surviving_k']['mean']:.2f} |")
    c3 = r["check3_shape_pair"]
    adv_bm = c3["mixture_advantage_nats"]["bimodal"]
    adv_br = c3["mixture_advantage_nats"]["broad"]
    nll_separates = adv_bm > 0.05 and (adv_bm - adv_br) > 0.05
    verdict = ("the genuine mixture's HELD-OUT NLL separates them (it helps on two peaks, not on the broad bell)"
               if nll_separates else
               "the held-out NLL does NOT cleanly separate them at this setting (see FINDINGS.md)")
    bm_nll, br_nll = c3["bimodal_heldout_nll"], c3["broad_heldout_nll"]
    lines += ["", "## Check 3 — full shape vs summary (matched mean & variance)", "",
              "Principled arbiter: held-out NLL of the genuine mixture vs a single Gaussian fit by moments "
              "(the summary model). Surviving-k is reported only as a caveated resolution diagnostic — it "
              "over-counts both shapes via bin tiling (FINDINGS.md).", "",
              "| | bimodal | broad |", "|---|---:|---:|",
              f"| surviving k (median, diagnostic) | {c3['bimodal_surviving_k']['median']:.0f} | {c3['broad_surviving_k']['median']:.0f} |",
              f"| empirical mean | {c3['bimodal_empirical_mean_var'][0]:.2f} | {c3['broad_empirical_mean_var'][0]:.2f} |",
              f"| empirical variance | {c3['bimodal_empirical_mean_var'][1]:.2f} | {c3['broad_empirical_mean_var'][1]:.2f} |",
              f"| held-out NLL — mixture | {bm_nll['mixture']['mean']:.3f} | {br_nll['mixture']['mean']:.3f} |",
              f"| held-out NLL — single Gaussian | {bm_nll['single_gaussian']['mean']:.3f} | {br_nll['single_gaussian']['mean']:.3f} |",
              f"| held-out NLL — bypass only | {bm_nll['bypass_only']['mean']:.3f} | {br_nll['bypass_only']['mean']:.3f} |",
              f"| **mixture edge over summary (nats)** | **{adv_bm:+.3f}** | **{adv_br:+.3f}** |",
              "",
              f"Matched mean & variance → a moment/summary objective sees them as identical; {verdict}."]
    lines += ["", "## Negative control — smooth unimodal (must not invent classes)", "",
              "| noise σ | surviving k (median) | bypass weight (mean) |", "|---:|---:|---:|"]
    for row in r["negative_control_smooth"]:
        lines.append(f"| {row['noise']} | {row['surviving_k']['median']:.0f} | {row['bypass_weight']['mean']:.3f} |")
    lines.append("")
    with open(os.path.join(RESULTS_DIR, "results.md"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
