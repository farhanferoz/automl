"""K4 — build the VALID nested-k ladder and check its coherence bars (WS1, post-R1).

R1 (`capacity_ladder_results/R1_verdict.md`) ruled GO for K4 with amendments A-D. This
script trains the nested-k surrogate (`_capacity_ladder_nested.NestedKSurrogate`) — the
construction that makes the prefix ladder valid — on toys C/D/E and the single-mode broad
twins, then reads the pre-registered bars:

  * B-coh (nesting-costless): per-k held-out NLL of the ONE nested model vs SEPARATELY-trained
    fixed-k models. The primary reference is a DEDICATED fixed-k model of the SAME architecture
    (`fit_same_arch_fixed_k_sweep`) — this isolates the nesting cost from the surrogate-vs-MDN
    architecture gap, which confounded the first K4 pass (b_coh_worse everywhere was largely
    "k independent MLP heads fit worse than an MDN's shared mixture head", not a nesting cost).
    The leaner 75-epoch K0 `mdn_sweep` table on disk is retained as the amendment-B external
    cross-reference (amendment A: the K0 `aggregate_sparsity` tables are instrument-invalid, NOT
    used). One-sided read: the failure direction is nested WORSE than fixed-k by >2·SE.
  * B-knee: the global knee (`_capacity_ladder.knee`, plain i.i.d. bootstrap) reproduces
    across seeds — R1's DECISIVE check (not B-coh alone).
  * B-order: prefix/ladder content is stable across seeds (a reported diagnostic; instability
    is the registered trigger for the sandwich-draw fallback arm, not acted on here).
  * Broad twins (C_broad/E_broad): a valid ladder must ABSTAIN (knee r_star == 0) — the
    negative control that the invalid prefix ladder catastrophically failed (+24/+229 nats).

The per-input read (K5) loads the score tables this script saves; it is deliberately NOT
computed here (K0→K1K2K3 table-builder/reader split). k=1 rung == the direct single Gaussian
(amendment D). Strictly probabilistic: uniform draws are a schedule, no penalty, no tuned λ.
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

import _capacity_ladder as cl
import _capacity_ladder_nested as ckn
import _toy_datasets as td

from automl_package.utils.pytorch_utils import get_device

OUT_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "K4")
K0_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "K0")

SEEDS = (0, 1, 2)
N_TR, N_TE = cl._N_TR, cl._N_TE  # 1000, 2500 — matched to K0 so the B-coh comparison is apples-to-apples
N_EPOCHS = 800
BOOT_N = 1000
PERINPUT_WIDTH = 0.075  # ~7.5% of the x∈[0,1] range, matching K1K2K3

# k_max per toy: G3 headroom of +2 above the analytic ceiling (D staircase tops at 3, C/E at 2).
KMAX = {"C": 6, "D": 8, "E": 6, "C_broad": 6, "E_broad": 6}
STRUCTURED_TOYS = ("C", "D", "E")
BROAD_TOYS = ("C_broad", "E_broad")
TOY_MAKERS = {
    "C": td.make_toy_c,
    "D": td.make_toy_d,
    "E": td.make_toy_e,
    "C_broad": td.make_toy_c_broad,
    "E_broad": td.make_toy_e_broad,
}


def _test_coord(x_te: np.ndarray) -> np.ndarray:
    """The 1-D input coordinate paired with each test row (toys are 1-D, x∈[0,1])."""
    return np.asarray(x_te, dtype=np.float64).ravel()


def paired_bootstrap_se(diff: np.ndarray, n_boot: int, seed: int) -> float:
    """Bootstrap SE of the mean of a per-sample paired difference (plain i.i.d. resample, G1)."""
    rng = np.random.default_rng(seed)
    n = diff.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[idx].mean(axis=1)
    return float(boot_means.std(ddof=1))


def fit_same_arch_fixed_k_sweep(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, y_te: np.ndarray, ref_grid: list[int], seed: int, device: str, n_epochs: int) -> np.ndarray:
    """Fixed-k control in the SAME architecture: one `NestedKSurrogate(fixed_k=k)` per k, 800ep.

    This isolates the nesting cost from the surrogate-vs-MDN architecture gap: a `b_coh_worse`
    read against an MDN could just be "the MDN mixture head is a better fitter than k independent
    MLP heads." Comparing the nested model's rung k against a DEDICATED k-component model of the
    identical architecture (spread-init, same lr/epochs) answers the real question — is nesting
    costless? Column j is the fixed-k=ref_grid[j] mixture's held-out per-example log-likelihood.
    """
    cols = []
    for k in ref_grid:
        m = ckn.train_nested_k_surrogate(x_tr, y_tr, k_max=k, n_epochs=n_epochs, lr=1e-2, adaptive_bin_means=True, device=device, seed=seed, fixed_k=k)
        cols.append(cl.score_table(m, x_te, y_te, [k])[:, 0])
    return np.stack(cols, axis=1)


def load_k0_mdn_colmeans(toy: str, seed: int) -> list[float] | None:
    """Loads the K0 75-epoch `mdn_sweep` column means for toy/seed (secondary cross-ref), or None."""
    path = os.path.join(K0_DIR, f"mdn_sweep_toy{toy}_seed{seed}.pt")
    if not os.path.exists(path):
        return None
    d = torch.load(path, weights_only=False)
    return [float(v) for v in np.asarray(d["score_mat"]).mean(axis=0)]


def run_structured_toy(toy: str, seed: int, device: str, smoke: bool) -> dict:
    """Trains the nested ladder + same-arch fixed-k reference for one structured toy/seed; reads B-coh/B-knee."""
    n_tr = 300 if smoke else N_TR
    n_te = 400 if smoke else N_TE
    n_epochs = 8 if smoke else N_EPOCHS
    k_max = KMAX[toy]
    c_grid = list(range(1, k_max + 1))
    ref_grid = list(range(1, min(k_max, 6) + 1))  # fixed-k reference exists to k=6 (K0 grid); D's 7,8 are headroom-only

    maker = TOY_MAKERS[toy]
    x_tr, y_tr = maker(n=n_tr, seed=seed)
    x_te, y_te = maker(n=n_te, seed=seed + 500)

    t0 = time.time()
    nested = ckn.train_nested_k_surrogate(x_tr, y_tr, k_max=k_max, n_epochs=n_epochs, lr=1e-2, adaptive_bin_means=True, device=device, seed=seed)
    score = cl.score_table(nested, x_te, y_te, c_grid)  # (N, k_max)
    nested_wall = time.time() - t0

    t1 = time.time()
    ref_score = fit_same_arch_fixed_k_sweep(x_tr, y_tr, x_te, y_te, ref_grid, seed, device, n_epochs)
    ref_wall = time.time() - t1

    # B-coh (nesting-costless): per-k paired held-out log-lik difference, nested-rung-k minus a
    # DEDICATED fixed-k model of the SAME architecture. One-sided: the failure is nested WORSE by
    # >2·SE (nested better is costless-and-then-some). This isolates nesting from the architecture
    # gap that confounded a raw nested-vs-MDN read (see fit_same_arch_fixed_k_sweep).
    bcoh = {}
    for j, k in enumerate(ref_grid):
        diff = score[:, k - 1] - ref_score[:, j]  # >0 => nested better at rung k
        mean_diff = float(diff.mean())
        se = paired_bootstrap_se(diff, BOOT_N if not smoke else 100, seed=0)
        bcoh[k] = {
            "mean_nested_minus_fixedk": mean_diff,
            "se": se,
            "worse_by_gt_2se": bool(mean_diff < -2.0 * se),  # the actual failure direction
            "nested_col_mean": float(score[:, k - 1].mean()),
            "fixedk_col_mean": float(ref_score[:, j].mean()),
        }

    # B-knee: global knee, plain i.i.d. bootstrap (block=None).
    r_star, delta_curve, se_curve = cl.knee(score, ref_c=1, n_boot=BOOT_N if not smoke else 100, block=None, c_grid=c_grid, seed=0)

    return {
        "toy": toy,
        "seed": seed,
        "k_max": k_max,
        "c_grid": c_grid,
        "b_coh": bcoh,
        "b_coh_any_worse": any(v["worse_by_gt_2se"] for v in bcoh.values()),
        "knee_r_star": int(r_star),
        "delta_curve": {int(k): float(v) for k, v in delta_curve.items()},
        "knee_se": {int(k): float(v) for k, v in se_curve.items()},
        "k0_mdn75_colmeans": load_k0_mdn_colmeans(toy, seed),
        "nested_wall_s": round(nested_wall, 1),
        "ref_wall_s": round(ref_wall, 1),
        "_score": score,  # kept in-memory for the .pt dump + B-order; stripped from JSON
        "_x": _test_coord(x_te),
        "_y": np.asarray(y_te, dtype=np.float64).ravel(),
    }


def run_broad_toy(toy: str, seed: int, device: str, smoke: bool) -> dict:
    """Trains the nested ladder on a single-mode broad twin; the knee MUST abstain (r_star==0)."""
    n_tr = 300 if smoke else N_TR
    n_te = 400 if smoke else N_TE
    n_epochs = 8 if smoke else N_EPOCHS
    k_max = KMAX[toy]
    c_grid = list(range(1, k_max + 1))

    maker = TOY_MAKERS[toy]
    x_tr, y_tr = maker(n=n_tr, seed=seed)
    x_te, y_te = maker(n=n_te, seed=seed + 500)

    nested = ckn.train_nested_k_surrogate(x_tr, y_tr, k_max=k_max, n_epochs=n_epochs, lr=1e-2, adaptive_bin_means=True, device=device, seed=seed)
    score = cl.score_table(nested, x_te, y_te, c_grid)
    r_star, delta_curve, se_curve = cl.knee(score, ref_c=1, n_boot=BOOT_N if not smoke else 100, block=None, c_grid=c_grid, seed=0)

    return {
        "toy": toy,
        "seed": seed,
        "k_max": k_max,
        "c_grid": c_grid,
        "knee_r_star": int(r_star),
        "abstains": bool(r_star == 0),
        "delta_curve": {int(k): float(v) for k, v in delta_curve.items()},
        "knee_se": {int(k): float(v) for k, v in se_curve.items()},
        "_score": score,
        "_x": _test_coord(x_te),
        "_y": np.asarray(y_te, dtype=np.float64).ravel(),
    }


def b_order_across_seeds(per_seed: list[dict]) -> dict:
    """Cross-seed stability of the global ladder shape: pairwise Pearson corr of the delta curve.

    A stable importance ordering ⇒ the Δ_c(ladder) vector looks the same across seeds. This is
    the reported B-order diagnostic; instability is the registered sandwich-draw fallback trigger.
    """
    curves = [np.array([r["delta_curve"][k] for k in sorted(r["delta_curve"])]) for r in per_seed]
    corrs = []
    for i in range(len(curves)):
        for j in range(i + 1, len(curves)):
            if curves[i].std() < 1e-12 or curves[j].std() < 1e-12:
                corrs.append(float("nan"))
            else:
                corrs.append(float(np.corrcoef(curves[i], curves[j])[0, 1]))
    valid = [c for c in corrs if not np.isnan(c)]
    return {"pairwise_delta_corr": corrs, "min_corr": (min(valid) if valid else None), "mean_corr": (float(np.mean(valid)) if valid else None)}


def _save_table(res: dict, toy: str, seed: int) -> None:
    """Saves the per-(toy, seed) held-out score table + input coordinate for K5 to read."""
    torch.save(
        {
            "score": torch.as_tensor(res["_score"], dtype=torch.float64),
            "x": torch.as_tensor(res["_x"]),
            "y": torch.as_tensor(res["_y"]),
            "c_grid": res["c_grid"],
            "toy": toy,
            "seed": seed,
        },
        os.path.join(OUT_DIR, f"nested_toy{toy}_seed{seed}.pt"),
    )


def _jsonable(obj: object) -> object:
    """Recursively drops private in-memory arrays (keys starting with `_`) and casts numpy scalars."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items() if not (isinstance(k, str) and k.startswith("_"))}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def main() -> None:
    """Runs the K4 battery: nested ladders + matched fixed-k reference; writes tables + summary."""
    parser = argparse.ArgumentParser(description="K4 nested-k ladder builder + coherence bars.")
    parser.add_argument("--smoke", action="store_true", help="tiny 1-toy/1-seed/8-epoch run to prove the pipeline end-to-end")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    device = str(get_device())
    print(f"[k4] device={device} smoke={args.smoke} N_TR={N_TR} N_TE={N_TE} epochs={N_EPOCHS}")

    structured = ["C"] if args.smoke else list(STRUCTURED_TOYS)
    broad = [] if args.smoke else list(BROAD_TOYS)
    seeds = (0,) if args.smoke else SEEDS

    summary: dict[str, object] = {"config": {"N_TR": N_TR, "N_TE": N_TE, "n_epochs": N_EPOCHS, "kmax": KMAX, "seeds": list(seeds)}, "structured": {}, "broad": {}, "b_order": {}}

    for toy in structured:
        per_seed = []
        for seed in seeds:
            res = run_structured_toy(toy, seed, device, args.smoke)
            per_seed.append(res)
            _save_table(res, toy, seed)
            print(f"[k4] {toy} s{seed}: knee r*={res['knee_r_star']} b_coh_worse={res['b_coh_any_worse']} nested={res['nested_wall_s']}s ref={res['ref_wall_s']}s")
        summary["structured"][toy] = [_jsonable(r) for r in per_seed]
        summary["b_order"][toy] = b_order_across_seeds(per_seed)
        print(f"[k4] {toy} B-knee r* across seeds = {[r['knee_r_star'] for r in per_seed]}  B-order min_corr={summary['b_order'][toy]['min_corr']}")

    for toy in broad:
        per_seed = []
        for seed in seeds:
            res = run_broad_toy(toy, seed, device, args.smoke)
            per_seed.append(res)
            _save_table(res, toy, seed)
            print(f"[k4] {toy} s{seed}: knee r*={res['knee_r_star']} abstains={res['abstains']}")
        summary["broad"][toy] = [_jsonable(r) for r in per_seed]

    out_path = os.path.join(OUT_DIR, "k4_summary.json")
    with open(out_path, "w") as f:
        json.dump(_jsonable(summary), f, indent=2)
    print(f"[k4] wrote {out_path}")


if __name__ == "__main__":
    main()
