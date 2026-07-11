"""P1 — depth power curve (X10, trimmed 3-point) — per-input selector program WS-D.

(docs/plans/perinput_selector_2026-07-10/EXECUTION_PLAN.md WS-D, P1 section)

The moving-mode power curve (T3) turns "absent" into "absent, or under-powered?" by sweeping
N. P1 asks the SAME question of the depth lane: F2/F3/X3 measured the toy-G per-input depth
signal at N_test=500 and found it modest (X3: `capacity_ladder_results/X3/PREREGISTRATION.md`);
P1 sweeps N_test up to grow the detection floor down and see whether the signal crosses it.

Nature: retrains the F2 nested-depth ladder (`capacity_ladder_f2.py`'s `_build_model` +
`_nested_all_depth_log_likelihood`, NESTED-strategy `FlexibleHiddenLayersNN`, F2's fixed
hyperparameters) at larger N, then reads each resulting score table with X3's repeated
cross-fit machinery (`capacity_ladder_x3.run_repeated_crossfit`, 50 splits, Nadeau-Bengio
2003 corrected SE) exactly as X3 does. Unlike F2, P1 trains ONLY the nested model per case —
it does not need F2's control/fixed-depth baselines, which exist solely to certify F2's B-coh
bar (a bar P1 does not re-check).

Definitions (locked before any real run):
  DETECTION FLOOR at N  = 2 * the Nadeau-Bengio corrected SE of the tercile per-bin-stacking
    advantage over global, at that N (`run_repeated_crossfit`'s `se_nadeau_bengio`) — the
    smallest |signal| that would register as significant at that N.
  MEASURED SIGNAL at N  = the point estimate of that same advantage (`mu_bar`).
  CROSSES = |signal| > floor for a given (toy, N, seed).

Toys: G (the varying-required-capacity toy) and G_flat (its uniform-complexity negative
control), `_capacity_ladder_toys.py`, 3 seeds {0, 1, 2}. N_test in {500, 2000, 8000}, N_train
scaled to F2's own train:test ratio (1000:500 = 2.0) at every point. The N_test=500 point
reuses F2's existing tables verbatim (same toy defaults + hyperparameters, key-compatible)
instead of retraining.

Pre-registered readings (`PREREGISTRATION.md`, written before any real run):
  toy G crosses its floor (>= 2/3 seeds) at some N -> "recoverable at N=<that N>".
  toy G never crosses up to N=8000 -> "below the floor up to N=8000", with the quantified
    floor/signal bound at N=8000.
  Control bar: toy G_flat must have 0/3 seeds crossing its OWN floor at EVERY N.

Usage:
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -u automl_package/examples/capacity_ladder_p1.py --selftest
    # orchestrator: measure ONE N=8000 fit's wall-time before the real matrix
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -u automl_package/examples/capacity_ladder_p1.py --only-n 8000 --toys G --seeds 0
    # orchestrator: run (or reuse) one N-point at a time, serialized; the last one folds the curve
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -u automl_package/examples/capacity_ladder_p1.py --only-n 500
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -u automl_package/examples/capacity_ladder_p1.py --only-n 2000
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -u automl_package/examples/capacity_ladder_p1.py --only-n 8000
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
from capacity_ladder_f2 import C_GRID, HIDDEN_SIZE, LEARNING_RATE, MAX_DEPTH, N_EPOCHS, _build_model, _nested_all_depth_log_likelihood  # noqa: E402
from capacity_ladder_f3 import _jsonable, load_f2_table  # noqa: E402
from capacity_ladder_x3 import _pooled_across_seeds, _synthetic_table, run_repeated_crossfit  # noqa: E402

from automl_package.enums import LayerSelectionMethod  # noqa: E402
from automl_package.models.flexible_neural_network import FlexibleHiddenLayersNN  # noqa: E402

RESULTS_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "P1")

TOY_GENERATORS = {"G": toys.make_toy_g, "G_flat": toys.make_toy_g_flat}
TOYS = list(TOY_GENERATORS)
SEEDS = [0, 1, 2]
N_TEST_POINTS = [500, 2000, 8000]
N_SPLITS = 50  # X3 convention, reused verbatim

# F2's train:test ratio (N_TRAIN=1000, N_TEST=500), held fixed as N scales (P1 spec: "N_train
# scaled proportionally (same ratio as F2's train:test)").
_TRAIN_TEST_RATIO = 1000.0 / 500.0


def _n_train(n_test: int) -> int:
    """N_train scaled to F2's train:test ratio (2.0) at a given N_test."""
    return round(n_test * _TRAIN_TEST_RATIO)


# ---------------------------------------------------------------------------
# Score-table acquisition: reuse F2's N=500 tables, reuse a prior P1 run's cached table, or
# train ONE nested-depth ladder (F2's builder + reader, no control/fixed-depth baselines).
# ---------------------------------------------------------------------------


def _train_and_score(toy: str, seed: int, n_train: int, n_test: int) -> tuple[np.ndarray, np.ndarray]:
    """Trains one nested-depth ladder at `(n_train, n_test)` and returns its held-out score table.

    Reuses F2's `_build_model` (NESTED-strategy `FlexibleHiddenLayersNN`, F2's fixed
    hyperparameters) and `_nested_all_depth_log_likelihood` (F2's own reader) verbatim. Unlike
    F2's `run_one_case`, this trains ONLY the nested model — P1 does not need F2's
    control/fixed-depth baselines, which exist to certify F2's B-coh bar, not re-checked here.

    Args:
        toy: one of `TOYS` ("G", "G_flat").
        seed: fit seed (also the toy's train-data seed; test data uses F2's `seed + 500` offset).
        n_train: training sample count.
        n_test: held-out sample count.

    Returns:
        `(score, x)`: `(n_test, MAX_DEPTH)` held-out score table and `(n_test,)` inputs.
    """
    make_fn = TOY_GENERATORS[toy]
    x_tr, y_tr = make_fn(n=n_train, seed=seed)
    x_te, y_te = make_fn(n=n_test, seed=seed + 500)  # F2's held-out seed-offset convention
    x_te_t = torch.as_tensor(x_te, dtype=torch.float32)
    y_te_t = torch.as_tensor(y_te, dtype=torch.float32)

    model = _build_model(FlexibleHiddenLayersNN, MAX_DEPTH, LayerSelectionMethod.NESTED, seed, N_EPOCHS, HIDDEN_SIZE, LEARNING_RATE)
    model.fit(x_tr, y_tr)
    score = _nested_all_depth_log_likelihood(model, x_te_t, y_te_t)
    if score.shape != (n_test, MAX_DEPTH):
        raise AssertionError(f"score shape {score.shape} != {(n_test, MAX_DEPTH)}")
    x = np.asarray(x_te, dtype=np.float64).ravel()
    return score, x


def _p1_table_path(toy: str, seed: int, n_test: int) -> str:
    """Path for a P1-trained (non-F2) score table, keyed by N unlike F2's toy/seed-only naming."""
    return os.path.join(RESULTS_DIR, f"nested_toy{toy}_N{n_test}_seed{seed}.pt")


def _get_score_table(toy: str, seed: int, n_train: int, n_test: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Gets one (toy, seed, N) score table: F2's table, a cached P1 table, or a fresh fit.

    Returns:
        `(score, x, source)`; `source` is `"F2"` (F2's own N=500 table — same toy defaults and
        hyperparameters, key-compatible), `"cache"` (a P1 table already trained by an earlier
        serialized `--only-n` invocation), or `"trained"` (trained now).
    """
    if n_test == 500 and n_train == _n_train(500):
        t = load_f2_table(toy, seed)
        if t is not None:
            return t["score"], t["x"], "F2"

    cache_path = _p1_table_path(toy, seed, n_test)
    if os.path.exists(cache_path):
        d = torch.load(cache_path, weights_only=False)
        return np.asarray(d["score"], dtype=np.float64), np.asarray(d["x"], dtype=np.float64).ravel(), "cache"

    score, x = _train_and_score(toy, seed, n_train, n_test)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.save(
        {
            "score": torch.as_tensor(score, dtype=torch.float64),
            "x": torch.as_tensor(x, dtype=torch.float64),
            "c_grid": C_GRID,
            "n_train": n_train,
            "n_test": n_test,
        },
        cache_path,
    )
    return score, x, "trained"


# ---------------------------------------------------------------------------
# Floor/signal extraction (X3's reader, reused verbatim, restated as a bound rather than a
# pass/fail verdict).
# ---------------------------------------------------------------------------


def _floor_and_signal(agg: dict) -> dict:
    """The P1 floor/signal/crosses triple from one `run_repeated_crossfit` aggregate.

    `floor` = 2 * the Nadeau-Bengio corrected SE (the minimum `|signal|` that would register as
    significant at this N — X3's `beats_global_2se` threshold restated as a bound rather than a
    verdict); `signal` = the measured tercile per-bin-stacking advantage over global (`mu_bar`,
    pooled across the score-half). `crosses` = `|signal| > floor`.
    """
    floor = 2.0 * agg["se_nadeau_bengio"]
    signal = agg["mu_bar"]
    return {"floor": floor, "signal": signal, "crosses": bool(abs(signal) > floor)}


# ---------------------------------------------------------------------------
# One N-point: every (toy, seed) score table + its X3 read.
# ---------------------------------------------------------------------------


def run_one_n(n_test: int, toy_names: list[str], seeds: list[int]) -> dict:
    """Gets (or trains) every requested (toy, seed) score table at one N_test and reads it via X3.

    Args:
        n_test: held-out sample count for this ladder point.
        toy_names: subset of `TOYS` to run.
        seeds: subset of `SEEDS` to run.

    Returns:
        `{"n_test", "n_train", "toys", "seeds", "per_case": [...]}`; each `per_case` row carries
        the full X3 `run_repeated_crossfit` aggregate (G6 full-vector reporting) plus the P1
        `floor`/`signal`/`crosses` triple.
    """
    n_train = _n_train(n_test)
    per_case = []
    for toy in toy_names:
        for seed in seeds:
            t0 = time.time()
            score, x, source = _get_score_table(toy, seed, n_train, n_test)
            elapsed = time.time() - t0
            agg = run_repeated_crossfit(score, x, n_splits=N_SPLITS)
            fs = _floor_and_signal(agg)
            per_case.append({"toy": toy, "seed": seed, "source": source, "wall_time_sec": elapsed, "agg": agg, **fs})
            print(f"[p1] N={n_test} {toy} seed={seed} ({source}, {elapsed:.1f}s): signal={fs['signal']:+.4f} floor={fs['floor']:.4f} crosses={fs['crosses']}")
    return {"n_test": n_test, "n_train": n_train, "toys": toy_names, "seeds": seeds, "per_case": per_case}


# ---------------------------------------------------------------------------
# Final aggregate: the floor-vs-N curve + the pre-registered readings (needs every N-point's
# full-matrix shard on disk).
# ---------------------------------------------------------------------------


def _reading_for_toy(curve: dict[int, dict], toy: str) -> str:
    """The locked P1 reading wording: first crossing N, or the quantified below-floor bound at N_max."""
    crossing_ns = [n for n in sorted(curve) if curve[n][toy]["crosses_2of3"]]
    if crossing_ns:
        return f"recoverable at N={crossing_ns[0]}"
    n_max = max(curve)
    row = curve[n_max][toy]
    return (
        f"below the floor up to N={n_max} (mean floor={row['mean_floor']:.4f} nats, "
        f"mean signal={row['mean_signal']:+.4f} nats, {row['n_seeds_crossing']}/3 seeds crossing)"
    )


def _aggregate(shards: dict[int, dict]) -> dict:
    """Builds the floor-vs-N curve and the pre-registered readings from the full-matrix N-shards.

    Crossing rule (locked in `PREREGISTRATION.md`, T3's sibling N-sweep-power-curve convention):
    `>= 2/3` seeds crossing their own floor at a given N. Control rule (also T3's convention,
    "control flat 3/3"): toy G_flat must have `0/3` seeds crossing at EVERY N.
    """
    curve: dict[int, dict] = {}
    for n_test, res in sorted(shards.items()):
        by_toy = {}
        for toy in TOY_GENERATORS:
            rows = [r for r in res["per_case"] if r["toy"] == toy]
            pooled = _pooled_across_seeds(rows)
            n_crossing = sum(1 for r in rows if r["crosses"])
            by_toy[toy] = {
                "n_train": res["n_train"],
                "floor_by_seed": [r["floor"] for r in rows],
                "signal_by_seed": [r["signal"] for r in rows],
                "mean_floor": float(np.mean([r["floor"] for r in rows])),
                "mean_signal": pooled["mean_mu_bar"],
                "n_seeds_crossing": n_crossing,
                "crosses_2of3": n_crossing >= 2,
                "pooled": pooled,
            }
        curve[int(n_test)] = by_toy

    g_reading = _reading_for_toy(curve, "G")
    gflat_holds = all(curve[n]["G_flat"]["n_seeds_crossing"] == 0 for n in curve)
    return {"curve": curve, "g_reading": g_reading, "g_flat_control_holds": gflat_holds}


# ---------------------------------------------------------------------------
# Selftest — synthetic known-answer discrimination (a) and a monotone-shrinking floor (b).
# MUST pass before trusting any real N-sweep read.
# ---------------------------------------------------------------------------


def run_selftest() -> bool:
    """Checks the floor/signal wrapper's discrimination (a) and the floor's N-scaling (b).

    (a) the floor/signal wrapper recovers X3's known-answer discrimination on synthetic tables.
    (b) the detection floor is finite and shrinks monotonically as N grows, on synthetic tables
    of increasing size built with a FIXED per-bin signal template (X3's `_synthetic_table`,
    reused verbatim).
    """
    rng = np.random.default_rng(0)
    pos_score, pos_x = _synthetic_table((1, 3, 6), rng)
    neg_score, neg_x = _synthetic_table((3, 3, 3), rng)
    pos_fs = _floor_and_signal(run_repeated_crossfit(pos_score, pos_x, n_splits=20))
    neg_fs = _floor_and_signal(run_repeated_crossfit(neg_score, neg_x, n_splits=20))
    print(f"[p1 selftest a] positive (per-region 1/3/6): signal={pos_fs['signal']:+.4f} floor={pos_fs['floor']:.4f} crosses={pos_fs['crosses']}")
    print(f"[p1 selftest a] negative (flat 3/3/3):        signal={neg_fs['signal']:+.4f} floor={neg_fs['floor']:.4f} crosses={neg_fs['crosses']}")
    ok_a = pos_fs["crosses"] and not neg_fs["crosses"]
    print(f"[p1 selftest a] positive-crosses={pos_fs['crosses']}  negative-stays-below={not neg_fs['crosses']}  {'PASS' if ok_a else 'FAIL'}")

    n_per_grid = [170, 670, 2670]  # ~ N_TEST_POINTS // 3 (three tercile regions), increasing N
    floors = []
    for n_per in n_per_grid:
        rng_n = np.random.default_rng(1000 + n_per)
        score, x = _synthetic_table((1, 3, 6), rng_n, n_per=n_per)
        floor = _floor_and_signal(run_repeated_crossfit(score, x, n_splits=10))["floor"]
        floors.append(floor)
        print(f"[p1 selftest b] n_per={n_per} (N~{3 * n_per}): floor={floor:.4f}  finite={np.isfinite(floor)}")
    ok_b = all(np.isfinite(f) for f in floors) and all(floors[i] > floors[i + 1] for i in range(len(floors) - 1))
    print(f"[p1 selftest b] floors={[f'{f:.4f}' for f in floors]}  monotone-decreasing={ok_b}  {'PASS' if ok_b else 'FAIL'}")

    ok = ok_a and ok_b
    print(f"[p1 selftest] {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Real N-sweep driver.
# ---------------------------------------------------------------------------


def main() -> None:
    """Parses args and either runs the selftest or one/all N-points of the real depth power curve.

    `--only-n` runs (or reuses) a single N_test point and writes its shard so the orchestrator
    can serialize the matrix one N at a time; once every N's FULL (all toys, all seeds) shard is
    on disk, the same invocation that completes the set also folds the floor-vs-N curve and the
    pre-registered readings into `p1_summary.json`. `--toys`/`--seeds` narrow a single invocation
    (e.g. one fit, for a wall-time probe) and write a separately-tagged shard that is excluded
    from the final aggregate.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="Run the synthetic known-answer selftest and exit.")
    parser.add_argument("--only-n", type=int, default=None, choices=N_TEST_POINTS, help="Run a single N_test point only (serialized orchestrator launches).")
    parser.add_argument("--toys", default=None, help="Comma-separated subset of toys (default G,G_flat); for wall-time probes / sharded runs.")
    parser.add_argument("--seeds", default=None, help="Comma-separated subset of seeds (default 0,1,2); for wall-time probes / sharded runs.")
    args = parser.parse_args()

    if args.selftest:
        sys.exit(0 if run_selftest() else 1)

    toy_names = TOYS if args.toys is None else [t.strip() for t in args.toys.split(",")]
    seeds = SEEDS if args.seeds is None else [int(s) for s in args.seeds.split(",")]
    is_full = toy_names == TOYS and seeds == SEEDS
    n_points = [args.only_n] if args.only_n else N_TEST_POINTS

    os.makedirs(RESULTS_DIR, exist_ok=True)
    for n_test in n_points:
        t0 = time.time()
        res = run_one_n(n_test, toy_names, seeds)
        res["wall_time_sec"] = time.time() - t0
        tag = "" if is_full else "_" + "-".join(toy_names) + "_s" + "".join(str(s) for s in seeds)
        shard_path = os.path.join(RESULTS_DIR, f"p1_summary_N{n_test}{tag}.json")
        with open(shard_path, "w") as f:
            json.dump(_jsonable(res), f, indent=2)
        print(f"wrote {shard_path}  ({res['wall_time_sec']:.1f}s)")

    if not is_full:
        print("[p1] partial (--toys/--seeds) run; skipping the full floor-vs-N aggregate (needs the untagged shard for every N).")
        return

    shards: dict[int, dict] = {}
    for n_test in N_TEST_POINTS:
        shard_path = os.path.join(RESULTS_DIR, f"p1_summary_N{n_test}.json")
        if not os.path.exists(shard_path):
            print(f"[p1] shard for N={n_test} not yet present; final floor-vs-N aggregate deferred.")
            continue
        with open(shard_path) as f:
            shards[n_test] = json.load(f)

    if len(shards) < len(N_TEST_POINTS):
        print(f"[p1] {len(shards)}/{len(N_TEST_POINTS)} N-points done; run the remaining --only-n shards, then re-run to fold the final read.")
        return

    agg = _aggregate(shards)
    summary = {
        "task": "P1",
        "n_test_points": N_TEST_POINTS,
        "seeds": SEEDS,
        "train_test_ratio": _TRAIN_TEST_RATIO,
        "n_splits": N_SPLITS,
        "crossing_rule": ">= 2/3 seeds with |signal| > 2*SE_nadeau_bengio at that N",
        "curve": agg["curve"],
        "readings": {"G": agg["g_reading"], "G_flat_control_holds": agg["g_flat_control_holds"]},
    }
    out_path = os.path.join(RESULTS_DIR, "p1_summary.json")
    with open(out_path, "w") as f:
        json.dump(_jsonable(summary), f, indent=2)
    print(f"wrote {out_path}")
    print("READINGS:", json.dumps(_jsonable(summary["readings"]), indent=2))


if __name__ == "__main__":
    main()
