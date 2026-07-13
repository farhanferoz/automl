"""Phase 1: our per-input count-by-inference vs standard supported models, on the known-answer toys.

Slim head-to-head, using the LIBRARY'S OWN supported models (no re-implementations):

  * XGBoost (`XGBoostModel`) — the everyday point predictor + a single uncertainty band. It must
    predict the average on multi-cluster data (a value between the clusters, where nothing lands).
  * Mixture density network (`MixtureDensityNetwork`) at a count chosen by validation — the standard
    "fix the number of clusters" density method, at its best honest single count.
  * Ours — the SAME flexible mixture (k_max components) but the per-input count is READ from the
    held-out structure test (advantage of the full mixture over a single Gaussian, locally), instead
    of committing to one count.

Scored on a held-out test set against the ORACLE (the true generator) — the best achievable, since
on the toys we know the truth. Fit quality is mixture NLL (lower = better). The per-input readout is
the held-out advantage of the flexible model over a single Gaussian (a 1-component MDN) at probe
inputs, vs the known true count.

Usage:
    python3 automl_package/examples/probreg_kselection_comparison.py D 1 400   # smoke: toy D, 1 seed
    python3 automl_package/examples/probreg_kselection_comparison.py           # full: C,D,E / 3 seeds
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys

import numpy as np

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root, so `import automl_package` works
import _toy_datasets as td

from automl_package.enums import UncertaintyMethod
from automl_package.models.baselines.mixture_density_network import MixtureDensityNetwork
from automl_package.models.xgboost_model import XGBoostModel
from automl_package.utils.distributions import GaussianDistribution

logging.getLogger("automl_package").setLevel(logging.WARNING)

SIG = 0.3
SEP_D = 4.0
K_RANGE = [1, 2, 3, 4, 5, 6]
K_MAX = 6
N_TR, N_VAL, N_TE = 1500, 1500, 4000
HIDDEN, N_HIDDEN = 64, 2
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "variational_em_comparison_results")

PROBES = {"C": [(0.0, 1), (0.5, 2), (1.0, 2)], "E": [(0.0, 1), (0.5, 2), (1.0, 1)], "D": [(0.1, 1), (0.5, 2), (0.9, 3)]}
MAKE = {"C": td.make_toy_c, "E": td.make_toy_e, "D": td.make_toy_d}
GIVEN = {"C": td.sample_toy_c_given_x, "E": td.sample_toy_e_given_x, "D": td.sample_toy_d_given_x}


def _lognorm(y: np.ndarray, m: float | np.ndarray, s: float) -> np.ndarray:
    return -0.5 * (math.log(2.0 * math.pi) + 2.0 * math.log(s) + ((y - m) / s) ** 2)


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    mx = np.max(a, axis=axis, keepdims=True)
    return (mx + np.log(np.sum(np.exp(a - mx), axis=axis, keepdims=True))).squeeze(axis)


def oracle_nll(name: str, x: np.ndarray, y: np.ndarray) -> float:
    """Mean NLL of the held-out set under the TRUE generator — the best achievable (the floor)."""
    x, y = x.ravel(), y.ravel()
    if name in ("C", "E"):
        sep = (td.sep_schedule if name == "C" else td.sep_hump)(x)
        o = 0.5 * sep * SIG
        comp = np.stack([_lognorm(y, -o, SIG), _lognorm(y, o, SIG)], axis=0) + math.log(0.5)
        return float(-_logsumexp(comp, axis=0).mean())
    k = td._staircase_k(x)
    lp = np.empty(x.size)
    for kk in np.unique(k):
        mask = k == kk
        means = (np.arange(kk) - (kk - 1) / 2.0) * SEP_D * SIG
        comp = np.stack([_lognorm(y[mask], m, SIG) for m in means], axis=0) + math.log(1.0 / kk)
        lp[mask] = _logsumexp(comp, axis=0)
    return float(-lp.mean())


def _given(name: str, xs: np.ndarray, seed: int) -> np.ndarray:
    if name == "D":
        return GIVEN[name](xs, sigma=SIG, separation=SEP_D, seed=seed)
    return GIVEN[name](xs, sigma=SIG, sep_min=0.3, sep_max=4.0, seed=seed)


def fit_mdn(x: np.ndarray, y: np.ndarray, k: int, epochs: int, seed: int) -> MixtureDensityNetwork:
    m = MixtureDensityNetwork(
        n_components=k, hidden_size=HIDDEN, n_hidden=N_HIDDEN, n_epochs=epochs, learning_rate=1e-2,
        random_seed=seed, calculate_feature_importance=False, optimize_hyperparameters=False,
    )
    m.fit(np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32).ravel())
    return m


def mdn_nll(m: MixtureDensityNetwork, x: np.ndarray, y: np.ndarray) -> float:
    d = m.predict_distribution(np.asarray(x, dtype=np.float32), filter_data=False)
    return float(-d.log_prob(np.asarray(y, dtype=np.float32).ravel()).mean())


def run_toy(name: str, seeds: list[int], epochs: int) -> dict:
    make_fn = MAKE[name]
    nlls = {f"mdn_k{k}": [] for k in K_RANGE}
    nlls.update({"oracle": [], "xgboost": [], "ours_kmax": [], "best_k_val": []})
    arb = {xv: [] for xv, _ in PROBES[name]}
    for seed in seeds:
        x_tr, y_tr = make_fn(n=N_TR, seed=seed)
        x_va, y_va = make_fn(n=N_VAL, seed=seed + 200)
        x_te, y_te = make_fn(n=N_TE, seed=seed + 500)
        nlls["oracle"].append(oracle_nll(name, x_te, y_te))

        mdns, val = {}, {}
        for k in K_RANGE:
            mdns[k] = fit_mdn(x_tr, y_tr, k, epochs, seed)
            nlls[f"mdn_k{k}"].append(mdn_nll(mdns[k], x_te, y_te))
            val[k] = mdn_nll(mdns[k], x_va, y_va)
        khat = min(K_RANGE, key=lambda k: val[k])
        nlls["best_k_val"].append(khat)
        nlls["ours_kmax"].append(nlls[f"mdn_k{K_MAX}"][-1])  # same model; the difference is the readout

        xg = XGBoostModel(calculate_feature_importance=False, optimize_hyperparameters=False,
                          uncertainty_method=UncertaintyMethod.BINNED_RESIDUAL_STD, random_seed=seed, n_estimators=300)
        xg.fit(np.asarray(x_tr, dtype=np.float32), np.asarray(y_tr, dtype=np.float32).ravel())
        gd = GaussianDistribution(xg.predict(np.asarray(x_te, dtype=np.float32), filter_data=False),
                                  xg.predict_uncertainty(np.asarray(x_te, dtype=np.float32), filter_data=False))
        nlls["xgboost"].append(float(-gd.log_prob(np.asarray(y_te, dtype=np.float32).ravel()).mean()))

        # per-input readout: held-out advantage of the flexible model over a single Gaussian (k=1 MDN)
        for xv, _ in PROBES[name]:
            xs = np.full(3000, xv, dtype=np.float32)
            ys = _given(name, xs, seed * 31 + int(xv * 100))
            arb[xv].append(mdn_nll(mdns[1], xs.reshape(-1, 1), ys) - mdn_nll(mdns[K_MAX], xs.reshape(-1, 1), ys))

    res = {"toy": name, "nll": {k: (float(np.mean(v)) if k != "best_k_val" else [int(z) for z in v]) for k, v in nlls.items()},
           "probes": [{"input": xv, "true_count": kt, "our_readout": float(np.mean(arb[xv]))} for xv, kt in PROBES[name]]}
    return res


def _report(res: dict) -> None:
    name, n = res["toy"], res["nll"]
    o = n["oracle"]
    print(f"\n========== Toy {name} ==========")
    print(f"  Held-out fit (NLL, lower=better; oracle/best-possible = {o:+.3f}):")
    print(f"    XGBoost (everyday point model):  {n['xgboost']:+.3f}   (gap {n['xgboost'] - o:+.3f})")
    for k in K_RANGE:
        print(f"    mixture, fixed {k}:                {n[f'mdn_k{k}']:+.3f}   (gap {n[f'mdn_k{k}'] - o:+.3f})")
    print(f"    standard pick (best count by validation): {n['best_k_val']}")
    print(f"    OURS (flexible + per-input readout):  {n['ours_kmax']:+.3f}   (gap {n['ours_kmax'] - o:+.3f})")
    print("  Per-input structure (our readout; >0 = extra clusters genuinely earn their keep there):")
    for p in res["probes"]:
        print(f"    input {p['input']:>4}: true clusters={p['true_count']}  | our readout {p['our_readout']:+.3f}")


def main() -> None:
    toys = sys.argv[1].split(",") if len(sys.argv) > 1 else ["D", "E", "C"]
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    seeds = list(range(n_seeds))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"comparison (supported models): toys={toys} seeds={seeds} epochs={epochs}", flush=True)
    out = []
    for name in toys:
        res = run_toy(name, seeds, epochs)
        _report(res)
        out.append(res)
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump({"config": {"sig": SIG, "k_range": K_RANGE, "n_tr": N_TR, "n_te": N_TE, "seeds": seeds, "epochs": epochs}, "toys": out}, f, indent=2)
    print(f"\nSaved -> {RESULTS_DIR}/results.json", flush=True)


if __name__ == "__main__":
    main()
