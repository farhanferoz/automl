"""K1/K2/K3: post-hoc reads on the K0 score tables (WS1 capacity-ladder program, task K1-K3).

Pure post-hoc — no model training except a small over-chop-control addendum (the two broad-twin
aggregate_sparsity tables K0 did not build). Everything here consumes the 18 K0 score tables in
`capacity_ladder_results/K0/` through the shared readers in `_capacity_ladder.py` (`stack_em`,
`knee`, `perbin_stack`, `perinput_curve`, `mixture_logscore`, `quantile_bins`) — reused, not
reimplemented, per the K0 contract.

Three reads, per `docs/plans/capacity_ladder_2026-07-09/EXECUTION_PLAN.md` sec 2:

  K1 — global stacking (B4) + global knee (B2) per (method, toy, seed), pooled across seeds, and
       the over-chop control on broad twins (bias-free single-mode data; the knee must not railroad
       into "structure" that isn't there).
  K2 — per-bin stacking (B5): terciles (+ sextiles, the G5 shrink check) of x, fit/score split so the
       per-bin-vs-global comparison is honest, held-out log-score margin + bootstrap SE.
  K3 — the per-input advantage curve (`perinput_curve`) read against the analytic per-input ceilings
       on toys D (staircase) and E (hump), plus the toy-D MDN-sweep seed-coherence check (the
       "[6,3,4]" case from the June validation run).

G-rules honoured throughout (EXECUTION_PLAN sec 0b): G1 plain bootstrap for i.i.d. toy points
(`block=None`, never block-by-seed); G2 abstain is a real sentinel, not "capacity 0 confirmed"; G4
per-bin cells accompany every global read; G5 half-width/half-bin-count shrink checks; G6 full
vectors, never scalar-only summaries.

Usage:
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python -m automl_package.examples.capacity_ladder_k1k2k3
"""

from __future__ import annotations

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
import _toy_datasets as td
import _variational_em_perinput as vemp

K0_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "K0")
OUT_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "K1K2K3")

METHODS = ("aggregate_sparsity", "mdn_sweep")
TOYS = ("C", "D", "E")
SEEDS = (0, 1, 2)
C_GRID = cl._K_RANGE  # [1, 2, 3, 4, 5, 6]

# Adequate-band registrations (K1 P2, EXECUTION_PLAN sec 2 K1) — evaluated against the POOLED knee.
ADEQUATE_BAND = {"C": (1, 2), "D": (2, 3), "E": (1, 2)}

# Broad-twin generators (the over-chop control, K1 P3 / K2 P2) — toy D has no broad twin.
_BROAD_MAKE = {"C": td.make_toy_c_broad, "E": td.make_toy_e_broad}

_FIT_SCORE_SPLIT_SEED = 2026  # fixed seed for every K2 50/50 fit/score split (reproducible, not tuned)
_BOOT_N = 1000


# ---------------------------------------------------------------------------
# Loading (K0 tables) + the one addendum build (broad-twin aggregate_sparsity tables).
# ---------------------------------------------------------------------------


def load_k0_table(method: str, toy: str, seed: int) -> dict:
    """Loads one K0 score table; returns `score` (N,6) float64 and `x` (N,) float64."""
    path = os.path.join(K0_DIR, f"{method}_toy{toy}_seed{seed}.pt")
    d = torch.load(path, weights_only=False)
    score = d["score_mat"].double().numpy()
    x = d["x"].double().numpy().ravel()
    assert list(d["c_grid"]) == C_GRID, f"{path}: c_grid {d['c_grid']} != {C_GRID}"
    return {"score": score, "x": x}


def build_broad_table(toy: str, seed: int) -> dict:
    """Builds one broad-twin aggregate_sparsity score table (over-chop control, not in K0).

    Exact same building blocks K0 used for the real toys (`train_aggregate_sparsity` +
    `_AggregateSparsityNested` + `score_table`, same N_tr/N_te/epochs/k_max), with the toy
    generator swapped to the single-mode broad twin (`make_toy_c_broad` / `make_toy_e_broad`).
    CPU-only via `train_aggregate_sparsity`'s own default (`device="cpu"`).
    """
    make_fn = _BROAD_MAKE[toy]
    x_tr, y_tr = make_fn(n=cl._N_TR, seed=seed)
    x_te, y_te = make_fn(n=cl._N_TE, seed=seed + 500)
    model = vemp.train_aggregate_sparsity(
        x_tr, y_tr, k_max=cl._K_MAX, alpha0=cl._AGG_ALPHA0, n_epochs=cl._AGG_EPOCHS,
        lr=1e-2, hidden=cl._AGG_HIDDEN, adaptive_bin_means=True, seed=seed,
    )
    adapter = cl._AggregateSparsityNested(model)
    mat = cl.score_table(adapter, x_te, y_te, C_GRID)
    x_arr = np.asarray(x_te, dtype=np.float64).ravel()
    return {"score": mat, "x": x_arr}


def save_broad_table(toy: str, seed: int, table: dict) -> str:
    """Saves a broad-twin table in the K0 artifact convention, under the K1K2K3 folder."""
    out_path = os.path.join(OUT_DIR, f"aggregate_sparsity_toy{toy}broad_seed{seed}.pt")
    torch.save(
        {
            "score_mat": torch.as_tensor(table["score"], dtype=torch.float64),
            "x": torch.as_tensor(table["x"], dtype=torch.float64),
            "split": "test",
            "c_grid": C_GRID,
            "seed": seed,
        },
        out_path,
    )
    return out_path


# ---------------------------------------------------------------------------
# Small shared utility: plain bootstrap SE of a 1-D vector's mean (K2's per-bin-vs-global margin).
# Reuses `_capacity_ladder._bootstrap_col_means` (the library's own bootstrap machinery) rather
# than reimplementing the resampling loop.
# ---------------------------------------------------------------------------


def _plain_boot_se(vec: np.ndarray, n_boot: int = _BOOT_N, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    boot = cl._bootstrap_col_means(vec.reshape(-1, 1), n_boot, None, rng)
    return float(boot[:, 0].std(ddof=1))


def _jsonable(obj):
    """Recursively converts numpy/torch scalars and arrays to plain Python/JSON types."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


# ---------------------------------------------------------------------------
# K1 — global stacking + global knee.
# ---------------------------------------------------------------------------


def _stack_and_knee(score: np.ndarray) -> dict:
    pi_hat = cl.stack_em(score)
    r_star, delta_curve, se = cl.knee(score, ref_c=1, n_boot=_BOOT_N, block=None, c_grid=C_GRID, seed=0)
    return {
        "pi_hat": pi_hat.tolist(),
        "argmax_c": int(C_GRID[int(np.argmax(pi_hat))]),
        "r_star": int(r_star),
        "delta_curve": {str(k): v for k, v in delta_curve.items()},
        "se": {str(k): v for k, v in se.items()},
    }


def run_k1(tables: dict, broad_tables: dict) -> dict:
    per_case = []
    for method in METHODS:
        for toy in TOYS:
            for seed in SEEDS:
                t = tables[(method, toy, seed)]
                res = _stack_and_knee(t["score"])
                res.update({"method": method, "toy": toy, "seed": seed, "n": t["score"].shape[0]})
                per_case.append(res)

    pooled = []
    for method in METHODS:
        for toy in TOYS:
            pooled_score = np.concatenate([tables[(method, toy, s)]["score"] for s in SEEDS], axis=0)
            res = _stack_and_knee(pooled_score)
            res.update({"method": method, "toy": toy, "n_pooled": pooled_score.shape[0]})
            pooled.append(res)

    broad_per_seed = []
    for toy in ("C", "E"):
        for seed in SEEDS:
            t = broad_tables[(toy, seed)]
            res = _stack_and_knee(t["score"])
            res.update({"toy": f"{toy}_broad", "seed": seed, "n": t["score"].shape[0]})
            broad_per_seed.append(res)

    broad_pooled = []
    for toy in ("C", "E"):
        pooled_score = np.concatenate([broad_tables[(toy, s)]["score"] for s in SEEDS], axis=0)
        res = _stack_and_knee(pooled_score)
        res.update({"toy": f"{toy}_broad", "n_pooled": pooled_score.shape[0]})
        broad_pooled.append(res)

    # P1: pi_hat does NOT rail to c_max=6 on any bias-free (C/D/E) toy/seed.
    c_max_idx = C_GRID.index(6)
    p1_rows = [
        {"method": r["method"], "toy": r["toy"], "seed": r["seed"], "pi_hat_6": r["pi_hat"][c_max_idx], "argmax_c": r["argmax_c"], "railed": r["pi_hat"][c_max_idx] > 0.5}
        for r in per_case
    ]
    p1_pass = not any(row["railed"] for row in p1_rows)

    # P2: pooled knee lands in the adequate band, per (method, toy).
    p2_rows = []
    for r in pooled:
        lo, hi = ADEQUATE_BAND[r["toy"]]
        in_band = lo <= r["r_star"] <= hi
        p2_rows.append({"method": r["method"], "toy": r["toy"], "r_star_pooled": r["r_star"], "band": [lo, hi], "in_band": in_band})
    p2_pass = all(row["in_band"] for row in p2_rows)

    # P3: broad-twin knee reads 1 or abstain (r_star in {0,1}), per-seed and pooled.
    p3_seed_rows = [{"toy": r["toy"], "seed": r["seed"], "r_star": r["r_star"], "ok": r["r_star"] in (0, 1)} for r in broad_per_seed]
    p3_pooled_rows = [{"toy": r["toy"], "r_star_pooled": r["r_star"], "ok": r["r_star"] in (0, 1)} for r in broad_pooled]
    p3_pass = all(row["ok"] for row in p3_seed_rows) and all(row["ok"] for row in p3_pooled_rows)

    return {
        "per_case": per_case,
        "pooled": pooled,
        "broad_twin": {"per_seed": broad_per_seed, "pooled": broad_pooled},
        "checks": {
            "P1_no_rail": {"pass": p1_pass, "rows": p1_rows},
            "P2_adequate_band_pooled": {"pass": p2_pass, "rows": p2_rows},
            "P3_broad_twin_abstain_or_1": {"pass": p3_pass, "per_seed": p3_seed_rows, "pooled": p3_pooled_rows},
        },
        "contrast_note": (
            "B1 (in-fit selector, recorded history): unregularized in-fit selection cap-tracks "
            "E[k] -> (k_max+2)/2 = 4 on this k_max=6 grid (April cap-tracking result, not "
            "recomputed here) -- the failure mode the held-out knee/stacking reads above must NOT "
            "reproduce."
        ),
    }


# ---------------------------------------------------------------------------
# K2 — per-bin stacking (terciles + the G5 sextile shrink check).
# ---------------------------------------------------------------------------


def _perbin_vs_global(score: np.ndarray, x: np.ndarray, n_bins: int) -> dict:
    """One fit/score-split per-bin-vs-global comparison at `n_bins` quantile bins."""
    n = score.shape[0]
    rng = np.random.default_rng(_FIT_SCORE_SPLIT_SEED)
    perm = rng.permutation(n)
    half = n // 2
    fit_idx, score_idx = perm[:half], perm[half:]
    score_fit, x_fit = score[fit_idx], x[fit_idx]
    score_score, x_score = score[score_idx], x[score_idx]

    pi_global = cl.stack_em(score_fit)
    bins_fit = cl.quantile_bins(x_fit, n_bins)
    pi_bins = cl.perbin_stack(score_fit, bins_fit)

    bins_score = cl.quantile_bins(x_score, n_bins)
    global_ls = cl.mixture_logscore(pi_global, score_score)
    perbin_ls = np.empty(score_score.shape[0], dtype=np.float64)
    for b in np.unique(bins_score):
        mask = bins_score == b
        pi_b = pi_bins.get(b, pi_global)  # a bin absent from the fit-half (rare, small n) falls back to global
        perbin_ls[mask] = cl.mixture_logscore(pi_b, score_score[mask])

    diff = perbin_ls - global_ls
    mean_diff = float(diff.mean())
    se_diff = _plain_boot_se(diff, seed=1)
    return {
        "n_bins": n_bins,
        "pi_global": pi_global.tolist(),
        "pi_bins": {str(int(b)): v.tolist() for b, v in pi_bins.items()},
        "mean_diff": mean_diff,
        "se_diff": se_diff,
        "beats_global_2se": mean_diff > 2.0 * se_diff,
        "n_fit": int(half),
        "n_score": int(n - half),
    }


def run_k2(tables: dict, broad_tables: dict) -> dict:
    per_case = []
    for method in METHODS:
        for toy in TOYS:
            for seed in SEEDS:
                t = tables[(method, toy, seed)]
                tercile = _perbin_vs_global(t["score"], t["x"], 3)
                sextile = _perbin_vs_global(t["score"], t["x"], 6)
                per_case.append({"method": method, "toy": toy, "seed": seed, "tercile": tercile, "sextile": sextile})

    for toy in ("C", "E"):
        for seed in SEEDS:
            t = broad_tables[(toy, seed)]
            tercile = _perbin_vs_global(t["score"], t["x"], 3)
            sextile = _perbin_vs_global(t["score"], t["x"], 6)
            per_case.append({"method": "aggregate_sparsity", "toy": f"{toy}_broad", "seed": seed, "tercile": tercile, "sextile": sextile})

    # P1: per-bin beats global by > 2 SE on D and E (tercile read).
    p1_rows = [{"method": r["method"], "toy": r["toy"], "seed": r["seed"], "mean_diff": r["tercile"]["mean_diff"], "se_diff": r["tercile"]["se_diff"], "beats": r["tercile"]["beats_global_2se"]} for r in per_case if r["toy"] in ("D", "E")]
    p1_pass_D = any(row["beats"] for row in p1_rows if row["toy"] == "D")
    p1_pass_E = any(row["beats"] for row in p1_rows if row["toy"] == "E")

    # P2: ties on C and on the broad twins (tercile read) -- report, do not assert a hard pass/fail bar.
    p2_rows = [{"method": r["method"], "toy": r["toy"], "seed": r["seed"], "mean_diff": r["tercile"]["mean_diff"], "se_diff": r["tercile"]["se_diff"], "beats": r["tercile"]["beats_global_2se"]} for r in per_case if r["toy"] in ("C", "C_broad", "E_broad")]

    # P3: per-bin pi_hat per tercile, D/E aggregate_sparsity cases only (where per-input truth is defined).
    p3_rows = [{"toy": r["toy"], "seed": r["seed"], "pi_bins": r["tercile"]["pi_bins"]} for r in per_case if r["method"] == "aggregate_sparsity" and r["toy"] in ("D", "E")]

    # G5: sextile stability -- does the beats-2SE verdict on D/E survive shrinkage?
    g5_rows = [{"method": r["method"], "toy": r["toy"], "seed": r["seed"], "tercile_beats": r["tercile"]["beats_global_2se"], "sextile_beats": r["sextile"]["beats_global_2se"], "tercile_mean_diff": r["tercile"]["mean_diff"], "sextile_mean_diff": r["sextile"]["mean_diff"]} for r in per_case if r["toy"] in ("D", "E")]

    return {
        "per_case": per_case,
        "checks": {
            "P1_perbin_beats_global_D_or_E": {"pass_D": p1_pass_D, "pass_E": p1_pass_E, "rows": p1_rows},
            "P2_ties_on_C_and_broad": {"rows": p2_rows},
            "P3_perbin_pihat_D_E": {"rows": p3_rows},
        },
        "G5_sextile_stability": g5_rows,
    }


# ---------------------------------------------------------------------------
# K3 — per-input curve + the toy-D MDN-sweep seed-coherence ("[6,3,4]") case.
# ---------------------------------------------------------------------------


def _e_hump_thresholds(sep_min: float = 0.3, sep_max: float = 4.0, boundary: float = 2.0) -> tuple[float, float]:
    """Analytic x where `sep_hump(x) == boundary` (the 2-sigma bimodality crossing), both sides of the hump."""
    frac = (boundary - sep_min) / (sep_max - sep_min)
    half = 1.0 - frac
    return (1.0 - half) / 2.0, (1.0 + half) / 2.0


def _perinput_knee_curve(delta: np.ndarray, c_grid: list) -> np.ndarray:
    """Per-input hard knee: smallest c whose next increment in `Delta_c(x)` fails to improve.

    Distinct from `_capacity_ladder.knee`'s G2 abstain=0 sentinel: here `ref_c=1` (`c_grid[0]`)
    IS itself a genuine member of the toys' truth grid (k*(x) is never 0), so "no further capacity
    helps" reads as `c_grid[0]`, not an abstain code. `delta[:, 0] == 0` by construction
    (`perinput_curve`'s own reference column).
    """
    q, c = delta.shape
    knee_c = np.full(q, c_grid[-1], dtype=np.int64)
    found = np.zeros(q, dtype=bool)
    for j in range(c - 1):
        step_fails = (delta[:, j + 1] - delta[:, j]) <= 0
        newly = step_fails & ~found
        knee_c[newly] = c_grid[j]
        found |= newly
    return knee_c


def run_k3(tables: dict, k1_results: dict) -> dict:
    width = 0.075  # ~7.5% of the [0,1] x-range
    query_x = np.linspace(0.02, 0.98, 49)

    curves = []
    for toy in TOYS:
        for seed in SEEDS:
            t = tables[("aggregate_sparsity", toy, seed)]
            out = cl.perinput_curve(t["score"], t["x"], width, ref_c=0, query_x=query_x)
            delta = np.asarray(out["delta"])
            delta_half = np.asarray(out["delta_half"])
            knee_c = _perinput_knee_curve(delta, C_GRID)
            knee_c_half = _perinput_knee_curve(delta_half, C_GRID)
            curves.append({
                "toy": toy, "seed": seed, "width": width, "width_half": width / 2.0,
                "query_x": query_x.tolist(), "n_neighbors": np.asarray(out["n_neighbors"]).tolist(),
                "delta": delta.tolist(), "delta_half": delta_half.tolist(),
                "perinput_knee": knee_c.tolist(), "perinput_knee_half": knee_c_half.tolist(),
            })

    # D: read by thirds against truth k*=1,2,3.
    d_thirds = []
    for r in curves:
        if r["toy"] != "D":
            continue
        qx = np.asarray(r["query_x"])
        kn = np.asarray(r["perinput_knee"])
        thirds = {}
        for name, (lo, hi) in (("first", (0.0, 1 / 3)), ("second", (1 / 3, 2 / 3)), ("third", (2 / 3, 1.0))):
            mask = (qx >= lo) & (qx < hi if hi < 1.0 else qx <= hi)
            vals, counts = np.unique(kn[mask], return_counts=True)
            thirds[name] = {"knee_values": vals.tolist(), "counts": counts.tolist(), "modal_knee": int(vals[np.argmax(counts)]) if len(vals) else None}
        d_thirds.append({"seed": r["seed"], "thirds": thirds, "truth": {"first": 1, "second": 2, "third": 3}})

    # E: read against the analytic hump crossings.
    e_lo, e_hi = _e_hump_thresholds()
    e_hump = []
    for r in curves:
        if r["toy"] != "E":
            continue
        qx = np.asarray(r["query_x"])
        kn = np.asarray(r["perinput_knee"])
        # crossing points: midpoints between adjacent query_x where the knee value changes.
        transitions = [((qx[i] + qx[i + 1]) / 2.0, int(kn[i]), int(kn[i + 1])) for i in range(len(qx) - 1) if kn[i] != kn[i + 1]]
        e_hump.append({
            "seed": r["seed"], "knee_by_query_x": list(zip(qx.tolist(), kn.tolist())),
            "observed_transitions": transitions, "analytic_crossings": [e_lo, e_hi],
        })

    # [6,3,4] coherence: toy D's MDN sweep, per-seed argmax-pi_hat and knee (already in K1's per_case).
    d_mdn_rows = [r for r in k1_results["per_case"] if r["method"] == "mdn_sweep" and r["toy"] == "D"]
    d_mdn_winners = sorted(d_mdn_rows, key=lambda r: r["seed"])
    argmax_set = {r["argmax_c"] for r in d_mdn_winners}
    knee_set = {r["r_star"] for r in d_mdn_winners}
    coherence = {
        "per_seed": [{"seed": r["seed"], "argmax_pi_hat_c": r["argmax_c"], "pi_hat": r["pi_hat"], "knee_r_star": r["r_star"]} for r in d_mdn_winners],
        "argmax_winner_set": sorted(argmax_set),
        "knee_winner_set": sorted(knee_set),
        "argmax_coherent": len(argmax_set) == 1,
        "knee_coherent": len(knee_set) == 1,
        "june_reference_incoherent_set": [6, 3, 4],
    }

    return {
        "width": width,
        "curves": curves,
        "D_thirds_vs_truth": d_thirds,
        "E_hump_vs_truth": e_hump,
        "D_mdn_seed_coherence": coherence,
    }


# ---------------------------------------------------------------------------


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()

    print("loading K0 tables...")
    tables = {}
    for method in METHODS:
        for toy in TOYS:
            for seed in SEEDS:
                tables[(method, toy, seed)] = load_k0_table(method, toy, seed)
    for (method, toy, seed), t in tables.items():
        print(f"  {method} toy{toy} seed{seed}: score={t['score'].shape} x={t['x'].shape}")

    print("building broad-twin tables (over-chop control, not in K0)...")
    t0 = time.time()
    broad_tables = {}
    for toy in ("C", "E"):
        for seed in SEEDS:
            bt = build_broad_table(toy, seed)
            broad_tables[(toy, seed)] = bt
            path = save_broad_table(toy, seed, bt)
            print(f"  {toy}_broad seed{seed}: score={bt['score'].shape} col_means={np.round(bt['score'].mean(axis=0), 3).tolist()} -> {path}")
    wall_broad = time.time() - t0
    print(f"broad-twin build wall time: {wall_broad:.1f}s")

    print("K1: global stacking + knee...")
    t0 = time.time()
    k1 = run_k1(tables, broad_tables)
    wall_k1 = time.time() - t0
    print(f"K1 wall time: {wall_k1:.1f}s  checks={ {k: v['pass'] if 'pass' in v else None for k, v in k1['checks'].items()} }")

    print("K2: per-bin stacking...")
    t0 = time.time()
    k2 = run_k2(tables, broad_tables)
    wall_k2 = time.time() - t0
    print(f"K2 wall time: {wall_k2:.1f}s")

    print("K3: per-input curve + coherence...")
    t0 = time.time()
    k3 = run_k3(tables, k1)
    wall_k3 = time.time() - t0
    print(f"K3 wall time: {wall_k3:.1f}s")

    wall_total = time.time() - t_start
    meta = {"wall_broad_twin_build": wall_broad, "wall_k1": wall_k1, "wall_k2": wall_k2, "wall_k3": wall_k3, "wall_total": wall_total}
    print(f"total wall time: {wall_total:.1f}s")

    for name, payload in (("k1_global", k1), ("k2_perbin", k2), ("k3_perinput", k3), ("meta", meta)):
        out_path = os.path.join(OUT_DIR, f"{name}.json")
        with open(out_path, "w") as f:
            json.dump(_jsonable(payload), f, indent=2)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
