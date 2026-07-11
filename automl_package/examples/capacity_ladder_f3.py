"""F3: global + per-bin reads on the F2 nested depth-ladder tables (WS2 capacity-ladder program, task F3).

(docs/plans/capacity_ladder_2026-07-09/EXECUTION_PLAN.md WS2, F3 section)

Pure post-hoc reader — no model training. F2 (`capacity_ladder_f2.py`) already trained the
NESTED-strategy `FlexibleHiddenLayersNN` depth ladder on toys G / G-flat / H and saved one
`(N, max_depth)` held-out score table per (toy, seed) to
`capacity_ladder_results/F2/nested_toy{toy}_seed{seed}.pt`. This script runs the SAME battery
K1/K2 ran on the WS1 score tables (`capacity_ladder_k1k2k3.py`), on those F2 tables instead,
through the shared readers in `_capacity_ladder.py` (`stack_em`, `knee`, `quantile_bins`,
`perbin_stack`, `mixture_logscore`) — reused, not reimplemented:

  * global knee (`cl.knee`, B2) + global stacking weights pi-hat (`cl.stack_em`, B4), per
    (toy, seed) and pooled across whatever seeds are available (K1-analogue).
  * per-bin stacking on terciles of x (`cl.perbin_stack`, B5) vs the global stack, held-out
    log-score margin + bootstrap SE, with the G4 sextile stability re-check (K2-analogue).

G-rules honoured (EXECUTION_PLAN.md sec 0b): G1 plain i.i.d. bootstrap (`block=None`) for the
knee read on these i.i.d. toy points — NEVER block-by-seed, a load-bearing decision; G2 abstain
(`r_star=0`) is a real sentinel, not "capacity 0 confirmed"; G4 per-bin cells (terciles) always
accompany the global read, with a sextile stability re-check; G6 full vectors (pi_hat, delta
curves), never scalar-only summaries.

Pre-registered (EXECUTION_PLAN.md F3 section, reported met/not — not fudged post hoc):
  * toy G: per-bin stacking beats the global stack by > 2 SE (the varying-capacity signal).
  * toy G-flat: per-bin TIES the global stack (no seed beats it by > 2 SE) — the
    no-false-positive bar (bypass-confound negative control: uniform complexity everywhere).
  * toy H: the per-bin read varies with SNR (the resolution-dial signature), read here as the
    tercile-stacked argmax capacity trending DOWN from the highest-SNR tercile (lowest x,
    sigma_low) to the lowest-SNR tercile (highest x, sigma_high) — there is no analytic per-input
    depth ceiling for H (only sigma(x) varies, not the mean function), so this is F3's own
    operational reading of "varies with SNR", not a reused template criterion.

F2's tables may not all exist yet (F2 trains as it goes); any missing (toy, seed) is skipped and
noted, exactly like K5's `read_case` returning `None` for an absent K4 table.

Usage:
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/capacity_ladder_f3.py --selftest
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/capacity_ladder_f3.py
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

F2_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "F2")
OUT_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "F3")

TOYS = ("G", "G_flat", "H")
SEEDS = (0, 1, 2)
C_GRID = list(range(1, 7))  # F2's max_depth = 6

_FIT_SCORE_SPLIT_SEED = 2026  # fixed seed for every tercile/sextile 50/50 fit/score split (K1K2K3 convention)
_BOOT_N = 1000


# ---------------------------------------------------------------------------
# Small shared utilities (K1K2K3 conventions, reused verbatim).
# ---------------------------------------------------------------------------


def _jsonable(obj: object) -> object:
    """Recursively converts numpy/torch scalars and arrays to plain Python/JSON types."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def _plain_boot_se(vec: np.ndarray, n_boot: int = _BOOT_N, seed: int = 0) -> float:
    """Plain i.i.d. bootstrap SE of a 1-D vector's mean, via `_capacity_ladder._bootstrap_col_means`."""
    rng = np.random.default_rng(seed)
    boot = cl._bootstrap_col_means(vec.reshape(-1, 1), n_boot, None, rng)
    return float(boot[:, 0].std(ddof=1))


# ---------------------------------------------------------------------------
# Loading (F2 per-case tables).
# ---------------------------------------------------------------------------


def load_f2_table(toy: str, seed: int) -> dict | None:
    """Loads one F2 nested-ladder score table; returns `None` if F2 hasn't produced it yet.

    Args:
        toy: one of `TOYS` ("G", "G_flat", "H").
        seed: one of `SEEDS`.

    Returns:
        `{"score": (N, 6) float64, "x": (N,) float64}`, or `None` if the `.pt` file is absent
        (mirrors K5's `read_case` — F2 may still be training).
    """
    path = os.path.join(F2_DIR, f"nested_toy{toy}_seed{seed}.pt")
    if not os.path.exists(path):
        return None
    d = torch.load(path, weights_only=False)
    score = np.asarray(d["score"], dtype=np.float64)
    x = np.asarray(d["x"], dtype=np.float64).ravel()
    c_grid = list(d["c_grid"])
    if c_grid != C_GRID:
        raise AssertionError(f"{path}: c_grid {c_grid} != {C_GRID}")
    return {"score": score, "x": x}


# ---------------------------------------------------------------------------
# K1-analogue: global stacking + global knee.
# ---------------------------------------------------------------------------


def _stack_and_knee(score: np.ndarray) -> dict:
    """Global pi-hat (`cl.stack_em`) + global knee (`cl.knee`, plain i.i.d. bootstrap, `block=None`)."""
    pi_hat = cl.stack_em(score)
    r_star, delta_curve, se = cl.knee(score, ref_c=1, n_boot=_BOOT_N, block=None, c_grid=C_GRID, seed=0)
    return {
        "pi_hat": pi_hat.tolist(),
        "argmax_c": int(C_GRID[int(np.argmax(pi_hat))]),
        "r_star": int(r_star),
        "delta_curve": {str(k): v for k, v in delta_curve.items()},
        "se": {str(k): v for k, v in se.items()},
    }


def run_global(tables: dict[tuple[str, int], dict]) -> dict:
    """Global stacking + knee, per (toy, seed) and pooled across whatever seeds are available.

    Args:
        tables: `{(toy, seed): {"score": ..., "x": ...}}`, only entries F2 has produced so far.

    Returns:
        `{"per_case": [...], "pooled": [...]}` — pooled entries only exist for toys with >= 1
        available seed; pooling concatenates rows across seeds and reads ONE plain i.i.d. knee
        on the pooled table (never block-by-seed — the toy points are i.i.d. within a seed and
        seeds share the same generator, so pooling then reading plainly is the K1 convention).
    """
    per_case = []
    for (toy, seed), t in sorted(tables.items()):
        res = _stack_and_knee(t["score"])
        res.update({"toy": toy, "seed": seed, "n": t["score"].shape[0]})
        per_case.append(res)

    pooled = []
    for toy in TOYS:
        seeds_avail = sorted(s for (ty, s) in tables if ty == toy)
        if not seeds_avail:
            continue
        pooled_score = np.concatenate([tables[(toy, s)]["score"] for s in seeds_avail], axis=0)
        res = _stack_and_knee(pooled_score)
        res.update({"toy": toy, "seeds_pooled": seeds_avail, "n_pooled": pooled_score.shape[0]})
        pooled.append(res)

    return {"per_case": per_case, "pooled": pooled}


# ---------------------------------------------------------------------------
# K2-analogue: per-bin stacking (terciles + the sextile shrink check).
# ---------------------------------------------------------------------------


def _perbin_vs_global(score: np.ndarray, x: np.ndarray, n_bins: int, split_seed: int = _FIT_SCORE_SPLIT_SEED) -> dict:
    """One fit/score-split per-bin-vs-global comparison at `n_bins` quantile bins (K1K2K3 convention).

    `split_seed` selects the 50/50 fit/score partition; it defaults to `_FIT_SCORE_SPLIT_SEED` so the
    original F3 single-split read is reproduced bit-identically. X3 (`capacity_ladder_x3.py`) sweeps it
    over many values to average the per-bin advantage across re-randomized splits (repeated cross-fit).
    """
    n = score.shape[0]
    rng = np.random.default_rng(split_seed)
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


def _any_pass(rows: list[dict]) -> bool | None:
    """`True` if any row beats global by > 2 SE, `None` if `rows` is empty (no data yet)."""
    if not rows:
        return None
    return any(row["beats"] for row in rows)


def _all_tie(rows: list[dict]) -> bool | None:
    """`True` if NO row beats global by > 2 SE (the ties/no-false-positive bar), `None` if empty."""
    if not rows:
        return None
    return not any(row["beats"] for row in rows)


def run_perbin(tables: dict[tuple[str, int], dict]) -> dict:
    """Per-bin (tercile + sextile) stacking vs global, per (toy, seed), plus the F3 pre-reg checks.

    Args:
        tables: `{(toy, seed): {"score": ..., "x": ...}}`, only entries F2 has produced so far.

    Returns:
        `{"per_case": [...], "checks": {...}, "sextile_stability": [...]}`.
    """
    per_case = []
    for (toy, seed), t in sorted(tables.items()):
        tercile = _perbin_vs_global(t["score"], t["x"], 3)
        sextile = _perbin_vs_global(t["score"], t["x"], 6)
        per_case.append({"toy": toy, "seed": seed, "tercile": tercile, "sextile": sextile})

    def _toy_rows(toy: str) -> list[dict]:
        return [
            {"seed": r["seed"], "mean_diff": r["tercile"]["mean_diff"], "se_diff": r["tercile"]["se_diff"], "beats": r["tercile"]["beats_global_2se"]}
            for r in per_case
            if r["toy"] == toy
        ]

    g_rows = _toy_rows("G")
    gflat_rows = _toy_rows("G_flat")

    h_rows = []
    for r in per_case:
        if r["toy"] != "H":
            continue
        pi_bins = r["tercile"]["pi_bins"]
        bin_ids = sorted(int(b) for b in pi_bins)
        if len(bin_ids) < 2:
            continue
        argmax_lo = C_GRID[int(np.argmax(pi_bins[str(bin_ids[0])]))]  # lowest-x tercile = high-SNR (sigma_low) side
        argmax_hi = C_GRID[int(np.argmax(pi_bins[str(bin_ids[-1])]))]  # highest-x tercile = low-SNR (sigma_high) side
        h_rows.append({
            "seed": r["seed"],
            "argmax_capacity_high_snr_bin": argmax_lo,
            "argmax_capacity_low_snr_bin": argmax_hi,
            "varies": argmax_lo != argmax_hi,
            "snr_trend_down": argmax_lo > argmax_hi,
            "mean_diff": r["tercile"]["mean_diff"],
            "se_diff": r["tercile"]["se_diff"],
            "beats": r["tercile"]["beats_global_2se"],
        })
    h_pass = any(row["varies"] and row["snr_trend_down"] for row in h_rows) if h_rows else None

    sextile_stability = [
        {
            "toy": r["toy"], "seed": r["seed"],
            "tercile_beats": r["tercile"]["beats_global_2se"], "sextile_beats": r["sextile"]["beats_global_2se"],
            "tercile_mean_diff": r["tercile"]["mean_diff"], "sextile_mean_diff": r["sextile"]["mean_diff"],
        }
        for r in per_case
    ]

    return {
        "per_case": per_case,
        "checks": {
            "G_perbin_beats_global_2se": {"pass": _any_pass(g_rows), "rows": g_rows},
            "G_flat_ties_no_false_positive": {"pass": _all_tie(gflat_rows), "rows": gflat_rows},
            "H_varies_with_snr": {"pass": h_pass, "rows": h_rows},
        },
        "sextile_stability": sextile_stability,
    }


# ---------------------------------------------------------------------------
# Selftest — synthetic in-memory known-answer table. MUST pass before trusting any real read.
# ---------------------------------------------------------------------------


def run_selftest() -> bool:
    """Synthetic, in-memory known-answer check of `_perbin_vs_global` (no disk I/O, no training).

    Builds a tiny fake `(N, 6)` score table with three x-regions favouring capacities 1, 3 and 6
    respectively (reuses `_capacity_ladder._peaked_table`, the K0 selftest's own synthetic-table
    generator) and asserts: (a) tercile stacking beats the global stack by > 2 SE, and (b) the
    recovered per-bin argmax capacity is 1 in the lowest-x tercile and 6 in the highest-x tercile
    — proving the reader logic independently of whether F2's real tables exist yet.
    """
    rng = np.random.default_rng(0)
    n_per = 2000
    x = np.concatenate([
        rng.uniform(-1.0, -1.0 / 3.0, n_per),
        rng.uniform(-1.0 / 3.0, 1.0 / 3.0, n_per),
        rng.uniform(1.0 / 3.0, 1.0, n_per),
    ])
    score = np.concatenate([
        cl._peaked_table(n_per, peak_c=1, gap=4.0, noise_sd=0.3, rng=rng),
        cl._peaked_table(n_per, peak_c=3, gap=4.0, noise_sd=0.3, rng=rng),
        cl._peaked_table(n_per, peak_c=6, gap=4.0, noise_sd=0.3, rng=rng),
    ], axis=0)

    result = _perbin_vs_global(score, x, n_bins=3)
    ok_beats = bool(result["beats_global_2se"])
    print(f"[f3 selftest] tercile per-bin - global mean_diff={result['mean_diff']:+.4f} nats, SE={result['se_diff']:.4f} (> 2SE: {ok_beats})")

    pi_bins = result["pi_bins"]
    bin_ids = sorted(int(b) for b in pi_bins)
    argmax_by_bin = {b: C_GRID[int(np.argmax(pi_bins[str(b)]))] for b in bin_ids}
    ok_structure = argmax_by_bin[bin_ids[0]] == 1 and argmax_by_bin[bin_ids[-1]] == 6
    print(f"[f3 selftest] per-bin argmax capacity by bin = {argmax_by_bin} (expect bin{bin_ids[0]}=1, bin{bin_ids[-1]}=6: {ok_structure})")

    ok = ok_beats and ok_structure
    print(f"[f3 selftest] {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Real reader — whatever F2 tables exist.
# ---------------------------------------------------------------------------


def _run_reader() -> None:
    """Reads every available F2 table and writes `f3_summary.json` (skips missing (toy, seed) gracefully)."""
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()

    print("loading F2 tables...")
    tables: dict[tuple[str, int], dict] = {}
    missing: list[tuple[str, int]] = []
    for toy in TOYS:
        for seed in SEEDS:
            t = load_f2_table(toy, seed)
            if t is None:
                missing.append((toy, seed))
                continue
            tables[(toy, seed)] = t
            print(f"  toy{toy} seed{seed}: score={t['score'].shape} x={t['x'].shape}")
    if missing:
        print(f"  missing (F2 not finished yet, skipped): {missing}")

    if not tables:
        print("no F2 tables available yet -- nothing to read. Run capacity_ladder_f2.py first.")
        summary = {"missing": _jsonable(missing), "note": "no F2 tables available at read time"}
        out_path = os.path.join(OUT_DIR, "f3_summary.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"wrote {out_path}")
        return

    print("F3 global: stacking + knee...")
    global_res = run_global(tables)

    print("F3 per-bin: tercile + sextile stacking vs global...")
    perbin_res = run_perbin(tables)

    wall_total = time.time() - t_start

    bar_key_by_toy = {"G": "G_perbin_beats_global_2se", "G_flat": "G_flat_ties_no_false_positive", "H": "H_varies_with_snr"}
    for toy in TOYS:
        pooled_row = next((r for r in global_res["pooled"] if r["toy"] == toy), None)
        toy_rows = [r for r in perbin_res["per_case"] if r["toy"] == toy]
        if pooled_row is None or not toy_rows:
            print(f"[f3] {toy}: no F2 tables yet, skipped")
            continue
        advantages = [(r["tercile"]["mean_diff"], r["tercile"]["se_diff"]) for r in toy_rows]
        advantage_str = ", ".join(f"{d:+.4f}+/-{s:.4f}" for d, s in advantages)
        bar_pass = perbin_res["checks"][bar_key_by_toy[toy]]["pass"]
        print(f"[f3] {toy}: pooled knee r*={pooled_row['r_star']} argmax_c={pooled_row['argmax_c']}  tercile advantage per seed = [{advantage_str}]  pre-reg bar pass={bar_pass}")

    summary = {"missing": _jsonable(missing), "global": global_res, "perbin": perbin_res, "wall_time_sec": wall_total}
    out_path = os.path.join(OUT_DIR, "f3_summary.json")
    with open(out_path, "w") as f:
        json.dump(_jsonable(summary), f, indent=2)
    print(f"wrote {out_path}")
    print(f"total wall time: {wall_total:.1f}s")


def main() -> None:
    """Parses args and either runs the synthetic selftest or reads whatever F2 tables exist."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="Run the in-memory synthetic known-answer selftest and exit.")
    args = parser.parse_args()

    if args.selftest:
        ok = run_selftest()
        sys.exit(0 if ok else 1)

    _run_reader()


if __name__ == "__main__":
    main()
