"""Prior ablation: does the aggregate sparsity prior earn its keep, or is the readout the whole story?

The reframed contribution is a PIPELINE — fit a flexible mixture, then read the per-input cluster
count off a held-out test (the mixture's advantage over a single Gaussian) — that works on ANY
flexible mixture. This script puts the aggregate sparsity prior (`AggregateSparsityKSelector`: a
single ``Dir(alpha0 < 1)`` log-prior on the dataset-mean bucket usage) ON TRIAL. It is kept only if
it beats the SAME model with the prior switched off, read by the SAME arbiter.

Controlled ablation (identical architecture; ONLY the prior changes, via ``alpha0``):
  * prior ON  : ``alpha0 = 0.1``  (sparsity-favouring)
  * prior OFF : ``alpha0 = 1.0``  (``(alpha0 - 1) = 0`` so the log-prior term vanishes — pure NLL fit)
run in BOTH bin regimes:
  * fixed tiles    (``adaptive_bin_means=False``): the prior's design regime — the weights must carry
    the count, so a usage prior can bite.
  * adaptive heads (``adaptive_bin_means=True``) : the REAL-model regime — heads can slide to fit, so
    the weights may go uniform and the prior may do nothing.

External anchor: a plain MDN at ``k_max`` (a different no-prior flexible mixture), read the same way.

Every mixture is scored by the SAME held-out arbiter against the SAME independently-trained single
Gaussian (`CondGaussian`): ``Delta(x) = nll_single(x) - nll_mixture(x)`` at probe inputs of KNOWN
true count (``> 0`` => extra clusters genuinely earn their keep there). Held-out fit is the mean test
NLL against the oracle (the true generator — the best achievable). ``eff#(x)`` (exp-entropy of the
per-input weights) is reported for the sparsity models as the count-vs-arbiter contrast.

Verdict rule: if prior-ON does not beat prior-OFF on per-input count recovery (and ties on fit), the
prior is dropped — the readout pipeline is the contribution, on any flexible mixture.

Usage:
    python3 automl_package/examples/probreg_kselection_prior_ablation.py D 1 150 --no-mdn   # smoke
    python3 automl_package/examples/probreg_kselection_prior_ablation.py                    # full
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root, so `import automl_package` works

import probreg_kselection_comparison as cmp  # oracle_nll, _given, fit_mdn, mdn_nll, PROBES, MAKE, constants
import probreg_variational_em_step2_perinput_arbiter as p2  # CondGaussian, arbiter helpers
import probreg_variational_em_step3_perinput_model as p3  # effective_count, bucket_nll_per_point
import _variational_em_perinput as vemp  # train_aggregate_sparsity

K_MAX = cmp.K_MAX
N_TR, N_TE = cmp.N_TR, cmp.N_TE
N_PROBE = 3000
RESULTS_DIR = os.path.join(_EXAMPLES_DIR, "prior_ablation_results")

# (label, adaptive_bin_means, alpha0) — the four sparsity-model cells of the 2x2 ablation.
VARIANTS = [
    ("agg_fixed_prior", False, 0.1),
    ("agg_fixed_noprior", False, 1.0),
    ("agg_adaptive_prior", True, 0.1),
    ("agg_adaptive_noprior", True, 1.0),
]
VARIANT_LABELS = [v[0] for v in VARIANTS]


def _agg_mean_nll(model: vemp.AggregateSparsityKSelector, x: np.ndarray, y: np.ndarray) -> float:
    """Mean held-out NLL of a sparsity model on ``(x, y)``."""
    return float(p3.bucket_nll_per_point(model, x, y).mean())


def _agg_eff(model: vemp.AggregateSparsityKSelector, x: np.ndarray, y: np.ndarray) -> float:
    """Mean effective bucket count ``exp(entropy of per-input weights)`` on ``x``."""
    w = model.weights(p2._to_xy(x, y)[0]).cpu().numpy()
    return float(p3.effective_count(w).mean())


def _mean(d: dict[str, list[float]]) -> dict[str, float]:
    """Per-key mean of a ``label -> list-of-per-seed-floats`` accumulator."""
    return {k: float(np.mean(v)) for k, v in d.items()}


def run_toy(name: str, seeds: list[int], epochs: int, with_mdn: bool) -> dict:
    """Fit the controlled ablation (+ optional MDN anchor) per seed; return fit and per-input readout."""
    probes = cmp.PROBES[name]
    read_labels = VARIANT_LABELS + (["mdn_kmax"] if with_mdn else [])  # all mixtures get a readout
    fit: dict[str, list[float]] = {k: [] for k in ["oracle", "cond_gaussian", *read_labels]}
    # readout: every mixture vs the single-Gaussian; eff: sparsity variants only — over seeds
    readout: dict[float, dict[str, list[float]]] = {xv: {lab: [] for lab in read_labels} for xv, _ in probes}
    eff: dict[float, dict[str, list[float]]] = {xv: {lab: [] for lab in VARIANT_LABELS} for xv, _ in probes}

    for seed in seeds:
        x_tr, y_tr = cmp.MAKE[name](n=N_TR, seed=seed)
        x_te, y_te = cmp.MAKE[name](n=N_TE, seed=seed + 500)

        cond = p2.train_cond_gaussian(x_tr, y_tr, n_epochs=epochs, seed=seed)
        models = {
            label: vemp.train_aggregate_sparsity(x_tr, y_tr, k_max=K_MAX, alpha0=a0, n_epochs=epochs,
                                                 adaptive_bin_means=adapt, seed=seed)
            for label, adapt, a0 in VARIANTS
        }
        mdn = cmp.fit_mdn(x_tr, y_tr, K_MAX, epochs, seed) if with_mdn else None

        fit["oracle"].append(cmp.oracle_nll(name, x_te, y_te))
        fit["cond_gaussian"].append(float(p2.pure_nll_per_point(cond, x_te, y_te).mean()))
        for label, model in models.items():
            fit[label].append(_agg_mean_nll(model, x_te, y_te))
        if mdn is not None:
            fit["mdn_kmax"].append(cmp.mdn_nll(mdn, x_te, y_te))

        for xv, _ in probes:
            xs = np.full(N_PROBE, xv, dtype=np.float32)
            ys = cmp._given(name, xs, seed * 31 + int(xv * 100))
            xs2d = xs.reshape(-1, 1)
            pure_m = float(p2.pure_nll_per_point(cond, xs2d, ys).mean())
            for label, model in models.items():
                readout[xv][label].append(pure_m - _agg_mean_nll(model, xs2d, ys))
                eff[xv][label].append(_agg_eff(model, xs2d, ys))
            if mdn is not None:
                readout[xv]["mdn_kmax"].append(pure_m - cmp.mdn_nll(mdn, xs2d, ys))

    return {
        "toy": name,
        "fit": _mean(fit),
        "probes": [
            {"input": xv, "true_count": kt, "readout": _mean(readout[xv]), "eff_count": _mean(eff[xv])}
            for xv, kt in probes
        ],
    }


def _report(res: dict, with_mdn: bool) -> None:
    """Print a readable fit table and a prior-on-vs-off readout table per toy."""
    name, fit = res["toy"], res["fit"]
    o = fit["oracle"]
    print(f"\n========== Toy {name} ==========")
    print(f"  Held-out fit (mean NLL, lower=better; oracle/best-possible = {o:+.3f}):")
    print(f"    single Gaussian (reference):       {fit['cond_gaussian']:+.3f}   (gap {fit['cond_gaussian'] - o:+.3f})")
    if with_mdn:
        print(f"    plain MDN, k_max (no prior):       {fit['mdn_kmax']:+.3f}   (gap {fit['mdn_kmax'] - o:+.3f})")
    for label, _, _ in VARIANTS:
        print(f"    {label:<24} {fit[label]:+.3f}   (gap {fit[label] - o:+.3f})")
    print("  Per-input readout  Delta = nll_single - nll_mixture  (>0 = extra clusters earn their keep):")
    print(f"    {'input / true k':>16} | {'fixed:ON':>9} {'fixed:OFF':>9} | {'adapt:ON':>9} {'adapt:OFF':>9}" + ("   MDN" if with_mdn else ""))
    for p in res["probes"]:
        r = p["readout"]
        row = (f"    {str(p['input']) + ' / ' + str(p['true_count']):>16} | "
               f"{r['agg_fixed_prior']:>+9.3f} {r['agg_fixed_noprior']:>+9.3f} | "
               f"{r['agg_adaptive_prior']:>+9.3f} {r['agg_adaptive_noprior']:>+9.3f}")
        if with_mdn:
            row += f"  {r['mdn_kmax']:>+.3f}"
        print(row)
    print("  eff#(x) (clusters engaged; honest under fixed tiles, collapses under adaptive heads):")
    for p in res["probes"]:
        e = p["eff_count"]
        print(f"    {str(p['input']) + ' / ' + str(p['true_count']):>16} | "
              f"{e['agg_fixed_prior']:>9.2f} {e['agg_fixed_noprior']:>9.2f} | "
              f"{e['agg_adaptive_prior']:>9.2f} {e['agg_adaptive_noprior']:>9.2f}")


def main() -> None:
    """Run the prior ablation over the requested toys/seeds and save JSON + a readable report."""
    argv = sys.argv[1:]
    with_mdn = "--no-mdn" not in argv
    argv = [a for a in argv if a != "--no-mdn"]
    toys = argv[0].split(",") if len(argv) > 0 else ["C", "D", "E"]
    n_seeds = int(argv[1]) if len(argv) > 1 else 3
    epochs = int(argv[2]) if len(argv) > 2 else 800
    seeds = list(range(n_seeds))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"prior ablation: toys={toys} seeds={seeds} epochs={epochs} with_mdn={with_mdn}", flush=True)
    out = []
    for name in toys:
        res = run_toy(name, seeds, epochs, with_mdn)
        _report(res, with_mdn)
        out.append(res)
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump({"config": {"k_max": K_MAX, "n_tr": N_TR, "n_te": N_TE, "seeds": seeds, "epochs": epochs,
                              "with_mdn": with_mdn, "alphas": {"prior": 0.1, "noprior": 1.0}},
                   "toys": out}, f, indent=2)
    print(f"\nSaved -> {RESULTS_DIR}/results.json", flush=True)


if __name__ == "__main__":
    main()
