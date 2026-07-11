"""T1 path-2 — bounded easier-target search for a depth-requiring AND GD-learnable positive control.

(docs/plans/perinput_selector_2026-07-10/EXECUTION_PLAN.md WS-B, task T1; terminal G-FORK path 2,
user-selected 2026-07-11. Disposition packet: capacity_ladder_results/T1/T1_PATH1_ADJUDICATION.md.)

Path-1 HARDENED that the 5-fold tent map (`make_toy_t1` n_iter=5, 32 linear pieces) at trunk width
8 is GD-UNLEARNABLE at every depth even best-of-8 (region B stays ~1.1-1.2 nat below the sigma=0.1
oracle +0.8836), so it cannot be the depth lane's positive control. Path-2 asks the terminal
question the user greenlit: is there an EASIER target that is BOTH depth-requiring AND learnable?
It runs the SAME leak-free apparatus as path-1 (R=8 independent restarts per fixed depth, keep-best
by TRAIN loss, untouched test scored, the verbatim `_construction_bar`) on a bounded set of <=3
easier configs:

    tent4_w8  : n_iter=4 (16 pieces), trunk width 8   -- the primary sweet-spot candidate
    tent3_w8  : n_iter=3 (8 pieces),  trunk width 8   -- weaker depth requirement, easiest to learn
    tent4_w6  : n_iter=4 (16 pieces), trunk width 6   -- narrower trunk, more depth-demanding

Unlike path-1 (seed 0 only), path-2 is the CONSTRUCTION bar itself, so it runs all three T1 seeds
(`capacity_ladder_t1.SEEDS`). A config is a FOUND positive control when the construction bar's
`construction_pass` (region B improves d1->d2 AND d2->d3 by > 2*SE -- depth-requiring AND reached --
while region A stays flat) holds on >= 2/3 seeds under the best-of-8 keep-best-by-train read. If no
config clears it, the depth lane has no learnable positive control among the bounded set and the
finding stands as the representable-but-not-learnable asymmetry (path-3-as-fallback).

Everything scientific is reused verbatim from path-1 / t1: the toy (`make_toy_t1`, only `n_iter`
varied), the region split (`toy_t1_region`), `_build_model`, `_fixed_depth_log_likelihood`, the
oracle constant, and `_construction_bar`. The ONLY new code is (a) lifting trunk width to a
parameter (path-1's `run_multistart_depth` reads the module-global `HIDDEN_SIZE`), and (b) looping
the seeds + aggregating the per-seed construction bar. No penalty/lambda; strictly the same
held-out log-likelihood bar.

Usage:
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/capacity_ladder_t1_path2.py --selftest
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/capacity_ladder_t1_path2.py --smoke
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -u automl_package/examples/capacity_ladder_t1_path2.py --config tent4_w8
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -u automl_package/examples/capacity_ladder_t1_path2.py   # all three
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

import _capacity_ladder_toys as toys  # noqa: E402
from capacity_ladder_f2 import _build_model, _fixed_depth_log_likelihood, _jsonable  # noqa: E402
from capacity_ladder_t1 import (  # noqa: E402
    LEARNING_RATE,
    MAX_DEPTH,
    N_EPOCHS,
    N_TEST,
    N_TRAIN,
    RESULTS_DIR,
    SEEDS,
    _construction_bar,
)
from capacity_ladder_t1_path1 import _UNLEARNABLE_GAP_NAT, N_RESTARTS, ORACLE_LL, SIGMA  # noqa: E402

from automl_package.enums import LayerSelectionMethod  # noqa: E402
from automl_package.models.flexible_neural_network import FlexibleHiddenLayersNN  # noqa: E402

# (label, n_iter, hidden_size) — the bounded <=3-config search set, in the advisory order.
CONFIGS: list[tuple[str, int, int]] = [
    ("tent4_w8", 4, 8),
    ("tent3_w8", 3, 8),
    ("tent4_w6", 4, 6),
]


# ---------------------------------------------------------------------------
# One depth: R restarts, keep-best by TRAIN loss. Path-1's `run_multistart_depth` with trunk width
# (and learning rate / epochs) lifted to explicit parameters — otherwise byte-for-byte identical.
# ---------------------------------------------------------------------------


def _run_multistart_depth(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te_t: torch.Tensor,
    y_te_t: torch.Tensor,
    region: np.ndarray,
    depth: int,
    n_restarts: int,
    n_epochs: int,
    hidden_size: int,
    learning_rate: float,
    timing: list[tuple[str, float]],
) -> dict:
    """Trains `n_restarts` fixed-depth nets at one depth and keeps the best by TRAIN mean LL.

    Identical to `capacity_ladder_t1_path1.run_multistart_depth` except `hidden_size`,
    `learning_rate`, and `n_epochs` are explicit parameters (path-1 read them from module globals),
    so a config can vary the trunk width. Selection is on TRAIN mean LL only; the untouched test set
    is scored but never selected on.

    Args:
        x_tr: `(n_train, 1)` training inputs (identical across restarts).
        y_tr: `(n_train,)` training targets.
        x_te_t: `(n_test, 1)` untouched-test inputs as a float32 tensor.
        y_te_t: `(n_test,)` untouched-test targets as a float32 tensor.
        region: `(n_test,)` region labels (0 = A/linear, 1 = B/tent).
        depth: the fixed hidden-layer depth (`1..MAX_DEPTH`).
        n_restarts: independent random-init restarts.
        n_epochs: training epochs per restart.
        hidden_size: trunk width (the T1 pin is 8; a config may narrow it).
        learning_rate: optimizer learning rate (T1's `LEARNING_RATE`).
        timing: mutated in place with `(label, seconds)` per restart.

    Returns:
        A dict with the kept-best restart's per-sample test LL vector (`kept_test_ll`), the selection
        index (`kept_restart`), and per-restart / summary region diagnostics.
    """
    x_tr_t = torch.as_tensor(x_tr, dtype=torch.float32)
    y_tr_t = torch.as_tensor(y_tr, dtype=torch.float32)
    mask_a = region == 0
    mask_b = region == 1

    restarts: list[dict] = []
    kept_idx = -1
    kept_train_ll = -np.inf
    kept_test_ll: np.ndarray | None = None

    for r in range(n_restarts):
        t0 = time.time()
        model = _build_model(FlexibleHiddenLayersNN, depth, LayerSelectionMethod.NONE, r, n_epochs, hidden_size, learning_rate)
        model.fit(x_tr, y_tr)
        timing.append((f"depth={depth} restart={r}", time.time() - t0))

        train_ll = float(_fixed_depth_log_likelihood(model, x_tr_t, y_tr_t).mean())  # SELECTION metric (no test)
        test_ll = _fixed_depth_log_likelihood(model, x_te_t, y_te_t)  # scored but NOT used for selection
        rec = {
            "restart": r,
            "train_mean_ll": train_ll,
            "test_mean_ll": float(test_ll.mean()),
            "test_mean_ll_region_a": float(test_ll[mask_a].mean()),
            "test_mean_ll_region_b": float(test_ll[mask_b].mean()),
        }
        restarts.append(rec)
        if train_ll > kept_train_ll:
            kept_train_ll, kept_idx, kept_test_ll = train_ll, r, test_ll
        print(
            f"  [depth {depth}] restart {r}: train_LL={train_ll:+.4f}  test_LL(A)={rec['test_mean_ll_region_a']:+.4f}  "
            f"test_LL(B)={rec['test_mean_ll_region_b']:+.4f}"
        )

    if kept_test_ll is None:
        raise AssertionError("no restarts run")
    best_by_test_b = max(rec["test_mean_ll_region_b"] for rec in restarts)  # OPTIMISTIC (test-peeking) bound
    return {
        "depth": depth,
        "kept_restart": kept_idx,
        "kept_train_mean_ll": kept_train_ll,
        "kept_test_ll": kept_test_ll,
        "kept_test_mean_ll_region_a": float(kept_test_ll[mask_a].mean()),
        "kept_test_mean_ll_region_b": float(kept_test_ll[mask_b].mean()),
        "best_by_test_region_b": best_by_test_b,
        "restarts": restarts,
    }


# ---------------------------------------------------------------------------
# One config: all seeds x all depths, per-seed construction bar, aggregate verdict.
# ---------------------------------------------------------------------------


def _config_verdict(per_seed: list[dict]) -> dict:
    """Aggregates the per-seed construction bar into the path-2 FOUND / NOT_FOUND verdict.

    A config is a FOUND learnable depth-requiring positive control when `construction_pass` holds on
    >= 2/3 seeds (region B climbs d1->d2 AND d2->d3 by > 2*SE while region A stays flat) under the
    best-of-8 keep-best-by-train read. The smallest kept and optimistic region-B gaps vs the oracle
    quantify how close region B got when the bar is not met.
    """
    n_b_pass = sum(1 for s in per_seed if s["construction"]["region_b_pass"])
    n_full_pass = sum(1 for s in per_seed if s["construction"]["construction_pass"])
    # smallest (best) region-B gap vs oracle, over all (seed, depth) — how close region B ever got.
    min_kept_gap = min(ORACLE_LL - d["kept_test_mean_ll_region_b"] for s in per_seed for d in s["per_depth"])
    min_opt_gap = min(ORACLE_LL - d["best_by_test_region_b"] for s in per_seed for d in s["per_depth"])

    found = n_full_pass >= 2
    if found:
        verdict = "FOUND_LEARNABLE_DEPTH_CONTROL"
        detail = (
            f"construction_pass on {n_full_pass}/{len(per_seed)} seeds (region B depth-requiring AND reached, "
            f"region A flat) under best-of-8 keep-best-by-train: a learnable depth-requiring positive control."
        )
    elif min_opt_gap > _UNLEARNABLE_GAP_NAT:
        verdict = "NOT_FOUND_UNLEARNABLE"
        detail = (
            f"construction_pass 0-1/{len(per_seed)} seeds; even best-of-8-BY-TEST region B stays "
            f"> {_UNLEARNABLE_GAP_NAT:.2f} nat below the {ORACLE_LL:+.4f} oracle at every depth "
            f"(min optimistic gap {min_opt_gap:.4f}) -- still GD-unlearnable, like tent^5."
        )
    else:
        verdict = "NOT_FOUND_AMBIGUOUS"
        detail = (
            f"construction_pass on {n_full_pass}/{len(per_seed)} seeds (< 2 needed); region B closes to within "
            f"{min_opt_gap:.4f} nat of oracle under the optimistic read but the depth-requiring bar is not met on a "
            f"majority -- neither a clean positive control nor airtight-unlearnable; flag for adjudication."
        )
    return {
        "verdict": verdict,
        "detail": detail,
        "n_seeds_region_b_pass": n_b_pass,
        "n_seeds_construction_pass": n_full_pass,
        "n_seeds": len(per_seed),
        "min_kept_region_b_gap_nat": float(min_kept_gap),
        "min_optimistic_region_b_gap_nat": float(min_opt_gap),
    }


def run_config(
    label: str,
    n_iter: int,
    hidden_size: int,
    n_restarts: int = N_RESTARTS,
    n_epochs: int = N_EPOCHS,
    seeds: list[int] = SEEDS,
    save: bool = True,
) -> dict:
    """Runs one path-2 config (a tent-fold count + trunk width) across all seeds and reports the bar.

    Args:
        label: the config label (e.g. ``"tent4_w8"``), used for the output filename.
        n_iter: tent-map fold count for region B (4 -> 16 pieces, 3 -> 8 pieces).
        hidden_size: trunk width (8 = the T1 pin; 6 = the narrower variant).
        n_restarts: restarts per depth (8, matching path-1's best-of-8 read).
        n_epochs: training epochs per restart (T1's 800 for the real run).
        seeds: the T1 seeds to run (all three for the construction bar).
        save: write the JSON summary under ``RESULTS_DIR`` when True.

    Returns:
        The config summary dict (also written to ``T1/t1_path2_{label}_summary.json`` when `save`).
    """
    print(f"\n########## CONFIG {label}: n_iter={n_iter} (2**{n_iter}={2 ** n_iter} pieces), width={hidden_size} ##########")
    timing: list[tuple[str, float]] = []
    per_seed: list[dict] = []
    for seed in seeds:
        x_tr, y_tr = toys.make_toy_t1(n=N_TRAIN, seed=seed, n_iter=n_iter)
        x_te, y_te = toys.make_toy_t1(n=N_TEST, seed=seed + 500, n_iter=n_iter)  # T1's held-out seed-offset convention
        x_te_t = torch.as_tensor(x_te, dtype=torch.float32)
        y_te_t = torch.as_tensor(y_te, dtype=torch.float32)
        region = toys.toy_t1_region(x_te)

        per_depth: list[dict] = []
        fixed_depth_ll: dict[int, np.ndarray] = {}
        for depth in range(1, MAX_DEPTH + 1):
            res = _run_multistart_depth(x_tr, y_tr, x_te_t, y_te_t, region, depth, n_restarts, n_epochs, hidden_size, LEARNING_RATE, timing)
            fixed_depth_ll[depth] = res["kept_test_ll"]
            per_depth.append({k: v for k, v in res.items() if k != "kept_test_ll"})
            gap = ORACLE_LL - res["kept_test_mean_ll_region_b"]
            print(
                f"[{label} seed {seed}] depth {depth}: kept restart {res['kept_restart']}  kept B={res['kept_test_mean_ll_region_b']:+.4f} "
                f"(gap {gap:.4f})  best-by-test B={res['best_by_test_region_b']:+.4f}"
            )
        construction = _construction_bar(fixed_depth_ll, region)
        per_seed.append({"seed": seed, "construction": construction, "per_depth": per_depth})
        print(
            f"[{label} seed {seed}] region_b_pass={construction['region_b_pass']} region_a_flat={construction['region_a_flat']} "
            f"construction_pass={construction['construction_pass']}"
        )

    verdict = _config_verdict(per_seed)
    total_wall = sum(s for _, s in timing)
    print(f"\n=== CONFIG {label} VERDICT: {verdict['verdict']} — {verdict['detail']}")
    print(f"(construction_pass on {verdict['n_seeds_construction_pass']}/{verdict['n_seeds']} seeds; "
          f"min optimistic region-B gap {verdict['min_optimistic_region_b_gap_nat']:.4f} nat; total wall {total_wall:.1f}s)")

    summary = {
        "task": "T1_path2",
        "config": label,
        "n_iter": n_iter,
        "n_pieces": 2 ** n_iter,
        "hidden_size": hidden_size,
        "seeds": list(seeds),
        "n_restarts": n_restarts,
        "n_epochs": n_epochs,
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "learning_rate": LEARNING_RATE,
        "oracle_ll": ORACLE_LL,
        "sigma": SIGMA,
        "selection_metric": "train_mean_ll (untouched-test scored, never selected on)",
        "verdict": verdict,
        "per_seed": per_seed,
        "total_wall_sec": total_wall,
    }
    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out = os.path.join(RESULTS_DIR, f"t1_path2_{label}_summary.json")
        with open(out, "w") as f:
            json.dump(_jsonable(summary), f, indent=2)
        print(f"wrote {out}")
    return summary


# ---------------------------------------------------------------------------
# Selftest — known-answer checks of the width-parameterized selection + verdict wiring (NO training).
# ---------------------------------------------------------------------------


def run_selftest() -> bool:
    """Checks keep-best-by-train selection (width-parameterized) and the FOUND / NOT_FOUND verdict.

    (a) `_run_multistart_depth`'s keep-best picks the max-TRAIN-LL restart (not test), with a
        synthetic trainer monkeypatched in so no torch runs — a case where best-train != best-test.
    (b) a planted "region B climbs d1<d2<d3 to oracle on 2/3 seeds, region A flat" set reads
        FOUND_LEARNABLE_DEPTH_CONTROL; a "region B flat ~1.1 nat below oracle" set reads
        NOT_FOUND_UNLEARNABLE.
    """
    ok = True

    # (a) keep-best-by-train, monkeypatched trainer (mirrors path-1's selftest (a)).
    n_te = 60
    region = np.array([0] * (n_te // 2) + [1] * (n_te // 2))
    x_te_t = torch.zeros((n_te, 1), dtype=torch.float32)
    y_te_t = torch.zeros(n_te, dtype=torch.float32)
    train_lls = [-0.50, -0.10, -0.90]  # restart 1 has the BEST train LL
    test_vecs = [np.full(n_te, 0.30), np.full(n_te, -0.20), np.full(n_te, 0.10)]  # but a WORSE test LL than restart 0
    calls = {"i": 0}

    class _Dummy:
        def fit(self, *_a: object, **_k: object) -> None:
            return None

    def _fake_build(*_a: object, **_k: object) -> _Dummy:
        return _Dummy()

    def _fake_fit_ll(_model: object, x_t: torch.Tensor, _y_t: object) -> np.ndarray:
        i = calls["i"]
        if x_t.shape[0] == n_te:
            calls["i"] += 1
            return test_vecs[i]
        return np.full(4, train_lls[i])

    g = globals()
    orig_build, orig_ll = g["_build_model"], g["_fixed_depth_log_likelihood"]
    g["_build_model"], g["_fixed_depth_log_likelihood"] = _fake_build, _fake_fit_ll
    try:
        res = _run_multistart_depth(np.zeros((4, 1), dtype=np.float32), np.zeros(4, dtype=np.float32), x_te_t, y_te_t, region, 3, 3, 3, 8, 5e-3, [])
    finally:
        g["_build_model"], g["_fixed_depth_log_likelihood"] = orig_build, orig_ll
    ok_a = res["kept_restart"] == 1 and abs(res["kept_test_mean_ll_region_b"] - (-0.20)) < 1e-9
    kb = res["kept_test_mean_ll_region_b"]
    print(f"[selftest a] keep-best-by-train picked restart {res['kept_restart']} (want 1), kept test-B={kb:+.3f} (want -0.200)  {'PASS' if ok_a else 'FAIL'}")
    ok = ok and ok_a

    # (b) verdict wiring on planted per-seed construction reads.
    rng = np.random.default_rng(0)
    n = 400
    reg = np.array([0] * (n // 2) + [1] * (n // 2))
    mb, ma = reg == 1, reg == 0

    def _planted_seed(b_levels: dict[int, float], seed_lbl: int) -> dict:
        fdll = {}
        for d in range(1, MAX_DEPTH + 1):
            v = np.empty(n)
            v[ma] = ORACLE_LL + rng.normal(0, 0.001, ma.sum())
            v[mb] = b_levels.get(d, b_levels[3]) + rng.normal(0, 0.001, mb.sum())
            fdll[d] = v
        con = _construction_bar(fdll, reg)
        per_depth = [{"depth": d, "kept_test_mean_ll_region_b": float(fdll[d][mb].mean()), "best_by_test_region_b": float(fdll[d][mb].mean())} for d in fdll]
        return {"seed": seed_lbl, "construction": con, "per_depth": per_depth}

    climb = {1: ORACLE_LL - 0.6, 2: ORACLE_LL - 0.25, 3: ORACLE_LL - 0.02}  # region B climbs to ~oracle
    flat = {1: ORACLE_LL - 1.1, 2: ORACLE_LL - 1.1, 3: ORACLE_LL - 1.1}  # region B flat far below
    found = _config_verdict([_planted_seed(climb, 0), _planted_seed(climb, 1), _planted_seed(flat, 2)])
    unlearn = _config_verdict([_planted_seed(flat, s) for s in range(3)])
    ok_b1 = found["verdict"] == "FOUND_LEARNABLE_DEPTH_CONTROL" and found["n_seeds_construction_pass"] == 2
    ok_b2 = unlearn["verdict"] == "NOT_FOUND_UNLEARNABLE"
    print(f"[selftest b1] planted 2/3-climb -> {found['verdict']} (n_pass={found['n_seeds_construction_pass']})  {'PASS' if ok_b1 else 'FAIL'}")
    print(f"[selftest b2] planted all-flat -> {unlearn['verdict']}  {'PASS' if ok_b2 else 'FAIL'}")
    ok = ok and ok_b1 and ok_b2

    print(f"[selftest] {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    """Parses args and runs the selftest, a fast smoke run, one config, or all configs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="No-training known-answer selftest, then exit.")
    parser.add_argument("--smoke", action="store_true", help="Fast plumbing check (tent4_w8, 2 restarts, 3 epochs, 1 seed); no save.")
    parser.add_argument("--config", choices=[c[0] for c in CONFIGS], help="Run only this one config (for parallel launching).")
    args = parser.parse_args()

    if args.selftest:
        sys.exit(0 if run_selftest() else 1)
    if args.smoke:
        run_config("tent4_w8", 4, 8, n_restarts=2, n_epochs=3, seeds=[0], save=False)
        return
    configs = [c for c in CONFIGS if (args.config is None or c[0] == args.config)]
    for label, n_iter, hidden in configs:
        run_config(label, n_iter, hidden)


if __name__ == "__main__":
    main()
