"""K5 — per-input knee on the nested ladder: the per-input count (headline deliverable, WS1).

Reads the K4 nested score tables (`capacity_ladder_results/K4/nested_toy*_seed*.pt`) and
produces a full per-input advantage-vs-k CURVE Δ_c(x) from ONE model, replacing the single
mixture-vs-Gaussian advantage of the June instrument (EXECUTION_PLAN.md K5). Reads:

  * per-input knee k̂(x): smallest capacity whose next Δ increment stops helping, neighbour-
    averaged over a box-car of half-width 0.075 in x (`_capacity_ladder.perinput_curve`).
  * G5 locality guard: the same read at half-width; a knee that MOVES under shrinkage is a
    pooling artifact, not structure. We report the fraction of query points whose knee is
    stable under the half-width re-read.
  * analytic-ceiling recovery: k̂(x) vs the KNOWN k*(x) — C rises 1→2 at the 2σ boundary
    (x*=0.459), D staircases 1→2→3 by thirds, E humps 1→2→1 at x≈0.23/0.77, broad twins
    stay at 1 (abstain). Interior recovery excludes a ±width margin around each transition
    (a box-car cannot resolve a step within its own half-width) and the domain edges.
  * June-arbiter cross-check (R1 amendment C): the mixture-vs-single-Gaussian advantage
    A(x) = Δ_{k_max}(x) (top rung vs the k=1 direct Gaussian), neighbour-averaged — the exact
    read the June instrument used — region-averaged and compared to June's documented recovery
    (D staircase −0.009/+0.147/+0.201 at k*=1/2/3; E hump −0.018/+0.149/−0.026 at 1/2/1).

Pure reader (no training). Run after K4 writes its tables.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root

import _capacity_ladder as cl  # noqa: E402
import _toy_datasets as td  # noqa: E402

OUT_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "K5")
K4_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "K4")

SEEDS = (0, 1, 2)
WIDTH = 0.075
QUERY_X = np.linspace(0.02, 0.98, 49)
STRUCTURED_TOYS = ("C", "D", "E")
BROAD_TOYS = ("C_broad", "E_broad")

# The 2σ bimodality boundary crossings (sep == 2.0), from the toy generators.
C_STAR = (2.0 - 0.3) / (4.0 - 0.3)  # 0.459: C's monotone sep crosses 2σ here (k* rises 1 -> 2)
_E_HALF = 1.0 - C_STAR  # E's triangle sep crosses 2σ symmetrically about x=0.5
E_LO, E_HI = (1.0 - _E_HALF) / 2.0, (1.0 + _E_HALF) / 2.0  # 0.23, 0.77: E humps 1 -> 2 -> 1

# June arbiter documented recovery (R1 §3 / kselection_variational_em.md §14): the
# mixture-vs-single-Gaussian advantage the nested top-vs-bottom read must reproduce.
JUNE_BENCH = {
    "D": {"first": -0.009, "second": 0.147, "third": 0.201, "truth": {"first": 1, "second": 2, "third": 3}},
    "E": {"low": -0.018, "mid": 0.149, "high": -0.026, "truth": {"low": 1, "mid": 2, "high": 1}},
}


def analytic_kstar(toy: str, x: np.ndarray) -> np.ndarray:
    """Ground-truth per-input component count k*(x) for a toy (x∈[0,1])."""
    xr = np.asarray(x, dtype=np.float64).ravel()
    if toy == "D":
        return td._staircase_k(xr)
    if toy == "C":
        return np.where(td.sep_schedule(xr) >= 2.0, 2, 1).astype(int)
    if toy == "E":
        return np.where(td.sep_hump(xr) >= 2.0, 2, 1).astype(int)
    if toy in BROAD_TOYS:
        return np.ones(xr.shape, dtype=int)
    raise ValueError(f"unknown toy {toy}")


def perinput_knee_curve(delta: np.ndarray, c_grid: list[int]) -> np.ndarray:
    """Per-input hard knee: smallest c whose next Δ increment fails to improve (mirrors K3's reader).

    `ref_c = c_grid[0]` is a genuine truth value (k*(x) ≥ 1), so "no further capacity helps"
    reads as `c_grid[0]`, NOT an abstain sentinel (`delta[:, 0] == 0` by construction).
    """
    _q, c = delta.shape
    knee_c = np.full(delta.shape[0], c_grid[-1], dtype=np.int64)
    found = np.zeros(delta.shape[0], dtype=bool)
    for j in range(c - 1):
        step_fails = (delta[:, j + 1] - delta[:, j]) <= 0
        newly = step_fails & ~found
        knee_c[newly] = c_grid[j]
        found |= newly
    return knee_c


def _transition_mask(toy: str, qx: np.ndarray, margin: float) -> np.ndarray:
    """True where a query point is NOT within `margin` of an analytic k*(x) transition or the edges."""
    interior = (qx > qx.min() + margin) & (qx < qx.max() - margin)
    if toy == "C":
        near = np.abs(qx - C_STAR) <= margin
    elif toy == "E":
        near = (np.abs(qx - E_LO) <= margin) | (np.abs(qx - E_HI) <= margin)
    elif toy == "D":
        near = (np.abs(qx - 1.0 / 3.0) <= margin) | (np.abs(qx - 2.0 / 3.0) <= margin)
    else:  # broad twins have no transition
        near = np.zeros(qx.shape, dtype=bool)
    return interior & ~near


def read_case(toy: str, seed: int) -> dict | None:
    """Reads one K4 nested table into per-input knee + G5 guard + recovery + arbiter cross-check."""
    path = os.path.join(K4_DIR, f"nested_toy{toy}_seed{seed}.pt")
    if not os.path.exists(path):
        return None
    d = torch.load(path, weights_only=False)
    score = np.asarray(d["score"], dtype=np.float64)
    x = np.asarray(d["x"], dtype=np.float64).ravel()
    c_grid = list(d["c_grid"])

    out = cl.perinput_curve(score, x, WIDTH, ref_c=0, query_x=QUERY_X)
    delta = np.asarray(out["delta"])
    delta_half = np.asarray(out["delta_half"])
    knee = perinput_knee_curve(delta, c_grid)
    knee_half = perinput_knee_curve(delta_half, c_grid)

    # G5 locality guard: stability of the knee under the half-width re-read.
    g5_stable_frac = float(np.mean(knee == knee_half))

    # Analytic-ceiling recovery (interior, transition-margin excluded).
    kstar = analytic_kstar(toy, QUERY_X)
    interior = _transition_mask(toy, QUERY_X, WIDTH)
    recovery_interior = float(np.mean(knee[interior] == kstar[interior])) if interior.any() else None
    recovery_all = float(np.mean(knee == kstar))

    # June-arbiter cross-check: A(x) = Δ_{k_max}(x) = top rung vs the k=1 direct Gaussian.
    arbiter = delta[:, -1]

    res = {
        "toy": toy,
        "seed": seed,
        "c_grid": c_grid,
        "query_x": QUERY_X.tolist(),
        "perinput_knee": knee.tolist(),
        "perinput_knee_half": knee_half.tolist(),
        "kstar": kstar.tolist(),
        "g5_stable_frac": g5_stable_frac,
        "recovery_interior": recovery_interior,
        "recovery_all": recovery_all,
        "arbiter_advantage": arbiter.tolist(),
        "n_neighbors": np.asarray(out["n_neighbors"]).tolist(),
    }
    res.update(_region_reads(toy, QUERY_X, knee, arbiter))
    return res


def _region_reads(toy: str, qx: np.ndarray, knee: np.ndarray, arbiter: np.ndarray) -> dict:
    """Toy-specific region summaries: D thirds, E hump regions — modal knee + arbiter mean vs June."""
    if toy == "D":
        regions = {"first": (0.0, 1 / 3), "second": (1 / 3, 2 / 3), "third": (2 / 3, 1.0)}
    elif toy == "E":
        regions = {"low": (0.0, E_LO), "mid": (E_LO, E_HI), "high": (E_HI, 1.0)}
    else:
        return {}
    modal, arb = {}, {}
    for name, (lo, hi) in regions.items():
        m = (qx >= lo) & (qx <= hi)
        if not m.any():
            modal[name], arb[name] = None, None
            continue
        vals, counts = np.unique(knee[m], return_counts=True)
        modal[name] = int(vals[np.argmax(counts)])
        arb[name] = float(arbiter[m].mean())
    return {"region_modal_knee": modal, "region_arbiter_mean": arb, "june_bench": JUNE_BENCH.get(toy)}


def main() -> None:
    """Reads every available K4 table into the K5 per-input summary."""
    os.makedirs(OUT_DIR, exist_ok=True)
    config = {"width": WIDTH, "n_query": len(QUERY_X), "seeds": list(SEEDS), "c_star": C_STAR, "e_crossings": [E_LO, E_HI]}
    summary: dict[str, object] = {"config": config, "structured": {}, "broad": {}}

    for toy in STRUCTURED_TOYS:
        cases = [read_case(toy, s) for s in SEEDS]
        cases = [c for c in cases if c is not None]
        if not cases:
            print(f"[k5] {toy}: no K4 tables yet, skipping")
            continue
        summary["structured"][toy] = cases
        rec = [c["recovery_interior"] for c in cases if c["recovery_interior"] is not None]
        g5 = [c["g5_stable_frac"] for c in cases]
        print(f"[k5] {toy}: interior recovery per seed = {[round(r, 3) for r in rec]}  G5-stable = {[round(g, 3) for g in g5]}")
        for c in cases:
            if "region_modal_knee" in c:
                arb = {k: (round(v, 3) if v is not None else None) for k, v in c["region_arbiter_mean"].items()}
                print(f"       seed{c['seed']} region modal knee = {c['region_modal_knee']}  arbiter mean = {arb}")

    for toy in BROAD_TOYS:
        cases = [read_case(toy, s) for s in SEEDS]
        cases = [c for c in cases if c is not None]
        if not cases:
            print(f"[k5] {toy}: no K4 tables yet, skipping")
            continue
        summary["broad"][toy] = cases
        abstain = [float(np.mean(np.asarray(c["perinput_knee"]) == 1)) for c in cases]
        print(f"[k5] {toy} (broad, must abstain to k=1): abstain fraction per seed = {[round(a, 3) for a in abstain]}")

    out_path = os.path.join(OUT_DIR, "k5_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[k5] wrote {out_path}")


if __name__ == "__main__":
    main()
