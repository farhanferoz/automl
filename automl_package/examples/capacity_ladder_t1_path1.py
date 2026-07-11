"""T1 path-1 — one-seed R=8 multi-restart disambiguation (GD-unlearnable vs single-restart-unlucky).

(docs/plans/perinput_selector_2026-07-10/EXECUTION_PLAN.md WS-B, task T1; disposition:
capacity_ladder_results/T1/FAIL_I_ADJUDICATION.md Q2 "Step A / path 1")

T1 bar-(i) FAILED 0/3 (`capacity_ladder_results/T1/FAIL_I_ADJUDICATION.md`): region B (the
5-fold-composed tent map) sat ~1.1 nat below the sigma=0.1 oracle (+0.8836) at EVERY depth on all
three seeds, while region A (linear) reached the oracle. The fresh-context adjudicator ruled that
the SINGLE-restart config used by `capacity_ladder_t1.py` cannot, by itself, distinguish "tent^5 is
GD-unlearnable at every depth" (the diagnosis) from "this one random init was unlucky." This driver
runs the adjudicator-approved, within-plan disambiguation:

  * R=8 independent random-init restarts per fixed depth, on SEED 0 ONLY;
  * keep-best by TRAINING loss (NOT test — else the held-out bar-(i) LL is contaminated by
    selection-on-test);
  * score the untouched test set's per-depth held-out LL of the kept-best restart, and re-run the
    CONSTRUCTION bar (`capacity_ladder_t1._construction_bar`) on that kept-best sweep.

Data are byte-identical to `capacity_ladder_t1.py`'s seed-0 case (`make_toy_t1(n=N_TRAIN, seed=0)`
for train, `make_toy_t1(n=N_TEST, seed=500)` for the untouched test); only the model's `random_seed`
varies across restarts (restart r uses `random_seed=r`, so restart 0 reproduces `capacity_ladder_t1`'s
exact single-restart fixed-depth fit as an anchor). Everything else — `_build_model` (NoneStrategy
fixed-depth net, width-8 T1 pin), `_fixed_depth_log_likelihood`, `_construction_bar`, and the T1
hyperparameters — is reused verbatim; only the keep-best-by-train selection is new.

Locked readings (from the adjudication, not re-invented here):
  * region B STAYS ~1 nat below the +0.8836 oracle at all depths even best-of-8 -> hardens
    "GD-unlearnable, not restart-luck" (the expected outcome; strengthens the terminal G-FORK).
  * region B CLOSES much of the gap at d>=3 under best-of-8 (construction bar `region_b_pass=True`)
    -> bar-(i) is rescuable on the SAME Telgarsky toy via multi-restart, a pure optimization fix
    with no premise change (the best outcome; would overturn the diagnosis).

Alongside the gated keep-best-by-train construction bar, this driver also reports the
best-of-8-BY-TEST region-B mean LL per depth: an OPTIMISTIC bound (it peeks at test, so it is
diagnostic-only, never gated). If even that most-generous read stays far below the oracle at every
depth, GD-unlearnability is airtight regardless of the selection metric.

Usage:
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/capacity_ladder_t1_path1.py --selftest
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/capacity_ladder_t1_path1.py --smoke
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -u automl_package/examples/capacity_ladder_t1_path1.py
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
    HIDDEN_SIZE,
    LEARNING_RATE,
    MAX_DEPTH,
    N_EPOCHS,
    N_TEST,
    N_TRAIN,
    RESULTS_DIR,
    _construction_bar,
)

from automl_package.enums import LayerSelectionMethod  # noqa: E402
from automl_package.models.flexible_neural_network import FlexibleHiddenLayersNN  # noqa: E402

SEED = 0  # path 1 is a SINGLE-seed disambiguation (adjudication Step A: "on seed 0 only")
N_RESTARTS = 8  # adjudication: "R=8 independent random-init restarts per depth"
SIGMA = 0.1  # make_toy_t1's noise std; fixes the oracle LL below
ORACLE_LL = float(-np.log(SIGMA) - 0.5 * np.log(2.0 * np.pi) - 0.5)  # +0.8836, the adjudication's anchor
# A best-of-8-BY-TEST region-B gap this large at EVERY depth => "airtight unlearnable" even under the
# most generous (test-peeking) selection. 0.7 nat is a conservative fraction of the observed ~1.1-nat
# single-restart gap; it only sharpens the verdict wording and gates NOTHING scientific (the gated read
# is the construction bar on the keep-best-by-train sweep).
_UNLEARNABLE_GAP_NAT = 0.7


# ---------------------------------------------------------------------------
# One depth: R restarts, keep-best by TRAIN loss; also track the test-peeking optimistic bound.
# ---------------------------------------------------------------------------


def run_multistart_depth(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te_t: torch.Tensor,
    y_te_t: torch.Tensor,
    region: np.ndarray,
    depth: int,
    n_restarts: int,
    n_epochs: int,
    timing: list[tuple[str, float]],
) -> dict:
    """Trains `n_restarts` fixed-depth nets at one depth and keeps the best by TRAIN mean LL.

    Args:
        x_tr: `(n_train, 1)` training inputs (identical across restarts).
        y_tr: `(n_train,)` training targets (identical across restarts).
        x_te_t: `(n_test, 1)` untouched-test inputs as a float32 tensor.
        y_te_t: `(n_test,)` untouched-test targets as a float32 tensor.
        region: `(n_test,)` region labels for the test set (0 = A/linear, 1 = B/tent).
        depth: the fixed hidden-layer depth being swept (`1..MAX_DEPTH`).
        n_restarts: number of independent random-init restarts.
        n_epochs: training epochs per restart.
        timing: mutated in place with `(label, seconds)` per restart, for the wall-time report.

    Returns:
        A dict with the kept-best restart's per-sample test LL vector (`kept_test_ll`), the
        selection index (`kept_restart`), and per-restart / summary region diagnostics.
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
        model = _build_model(FlexibleHiddenLayersNN, depth, LayerSelectionMethod.NONE, r, n_epochs, HIDDEN_SIZE, LEARNING_RATE)
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
        "kept_test_ll": kept_test_ll,  # (n_test,) vector, kept for the construction bar
        "kept_test_mean_ll_region_a": float(kept_test_ll[mask_a].mean()),
        "kept_test_mean_ll_region_b": float(kept_test_ll[mask_b].mean()),
        "best_by_test_region_b": best_by_test_b,
        "restarts": restarts,
    }


# ---------------------------------------------------------------------------
# Full path-1 sweep over all depths + the kept-best construction bar + the verdict.
# ---------------------------------------------------------------------------


def _verdict(construction: dict, per_depth: list[dict]) -> dict:
    """Maps the kept-best construction bar + the optimistic test bound onto the adjudication's readings."""
    region_b_pass = bool(construction["region_b_pass"])
    kept_gap_by_depth = {d["depth"]: ORACLE_LL - d["kept_test_mean_ll_region_b"] for d in per_depth}
    optimistic_gap_by_depth = {d["depth"]: ORACLE_LL - d["best_by_test_region_b"] for d in per_depth}
    min_kept_gap = min(kept_gap_by_depth.values())  # smallest (best) kept-best region-B gap over depth
    min_optimistic_gap = min(optimistic_gap_by_depth.values())  # smallest gap under the most generous read

    if region_b_pass:
        verdict = "RESCUABLE"
        detail = "region_b_pass=True under best-of-8 keep-best-by-train: tent^5 IS multi-restart-learnable; bar-(i) rescuable on the SAME toy (overturns diagnosis)."
    elif min_optimistic_gap > _UNLEARNABLE_GAP_NAT:
        verdict = "HARDENED_UNLEARNABLE"
        detail = (
            f"even best-of-8-BY-TEST region B stays > {_UNLEARNABLE_GAP_NAT:.2f} nat below the {ORACLE_LL:+.4f} oracle at EVERY depth "
            f"(min optimistic gap {min_optimistic_gap:.4f}): GD-unlearnable, not restart-luck (strengthens the terminal G-FORK)."
        )
    else:
        verdict = "AMBIGUOUS"
        detail = (
            f"construction bar not met (region_b_pass=False) yet the optimistic best-of-8-by-test region-B gap closes to "
            f"{min_optimistic_gap:.4f} nat at some depth: neither cleanly rescued nor airtight-unlearnable — flag for adjudication."
        )
    return {
        "verdict": verdict,
        "detail": detail,
        "region_b_pass": region_b_pass,
        "min_kept_region_b_gap_nat": float(min_kept_gap),
        "min_optimistic_region_b_gap_nat": float(min_optimistic_gap),
        "kept_region_b_gap_by_depth": {int(k): float(v) for k, v in kept_gap_by_depth.items()},
        "optimistic_region_b_gap_by_depth": {int(k): float(v) for k, v in optimistic_gap_by_depth.items()},
    }


def run_path1(n_restarts: int = N_RESTARTS, n_epochs: int = N_EPOCHS, save: bool = True) -> dict:
    """Runs the full seed-0 multi-restart sweep, the kept-best construction bar, and the verdict.

    Args:
        n_restarts: restarts per depth (adjudication: 8).
        n_epochs: training epochs per restart (T1's 800 for the real run; small for `--smoke`).
        save: write the JSON summary + kept-best `.pt` shard under `RESULTS_DIR` when True.

    Returns:
        The full summary dict (also written to `T1/t1_path1_summary.json` when `save`).
    """
    x_tr, y_tr = toys.make_toy_t1(n=N_TRAIN, seed=SEED)
    x_te, y_te = toys.make_toy_t1(n=N_TEST, seed=SEED + 500)  # T1's held-out seed-offset convention
    x_te_t = torch.as_tensor(x_te, dtype=torch.float32)
    y_te_t = torch.as_tensor(y_te, dtype=torch.float32)
    region = toys.toy_t1_region(x_te)

    timing: list[tuple[str, float]] = []
    per_depth: list[dict] = []
    fixed_depth_ll: dict[int, np.ndarray] = {}
    for depth in range(1, MAX_DEPTH + 1):
        res = run_multistart_depth(x_tr, y_tr, x_te_t, y_te_t, region, depth, n_restarts, n_epochs, timing)
        fixed_depth_ll[depth] = res["kept_test_ll"]
        per_depth.append(res)
        print(
            f"[path1] depth {depth}: kept restart {res['kept_restart']} (train_LL={res['kept_train_mean_ll']:+.4f})  "
            f"kept test_LL(A)={res['kept_test_mean_ll_region_a']:+.4f}  kept test_LL(B)={res['kept_test_mean_ll_region_b']:+.4f}  "
            f"[best-by-test B={res['best_by_test_region_b']:+.4f}]"
        )

    construction = _construction_bar(fixed_depth_ll, region)
    verdict = _verdict(construction, per_depth)

    total_wall = sum(s for _, s in timing)
    print("\n=== PATH-1 CONSTRUCTION BAR (kept-best-by-train) ===")
    print(f"region_b_pass={construction['region_b_pass']}  region_a_flat={construction['region_a_flat']}  construction_pass={construction['construction_pass']}")
    print(f"oracle LL (sigma={SIGMA}) = {ORACLE_LL:+.4f}")
    print("per-depth kept region-B mean test LL / gap vs oracle:")
    for d in per_depth:
        gap = ORACLE_LL - d["kept_test_mean_ll_region_b"]
        opt_gap = ORACLE_LL - d["best_by_test_region_b"]
        print(f"  d{d['depth']}: kept B={d['kept_test_mean_ll_region_b']:+.4f} (gap {gap:.4f})  |  best-by-test B={d['best_by_test_region_b']:+.4f} (gap {opt_gap:.4f})")
    print(f"\nVERDICT: {verdict['verdict']} — {verdict['detail']}")
    print(f"(total wall {total_wall:.1f}s over {len(timing)} fits)")

    summary = {
        "task": "T1_path1",
        "seed": SEED,
        "n_restarts": n_restarts,
        "n_epochs": n_epochs,
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "hidden_size": HIDDEN_SIZE,
        "oracle_ll": ORACLE_LL,
        "selection_metric": "train_mean_ll (untouched-test scored, never selected on)",
        "construction": construction,
        "verdict": verdict,
        "per_depth": [{k: v for k, v in d.items() if k != "kept_test_ll"} for d in per_depth],
        "total_wall_sec": total_wall,
    }

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        shard = os.path.join(RESULTS_DIR, "path1_toyT1_seed0.pt")
        torch.save(
            {
                "fixed_depth_ll": {d: torch.as_tensor(v, dtype=torch.float64) for d, v in fixed_depth_ll.items()},
                "x": torch.as_tensor(np.asarray(x_te, dtype=np.float64).ravel()),
                "region": torch.as_tensor(np.asarray(region, dtype=np.int64)),
                "oracle_ll": ORACLE_LL,
            },
            shard,
        )
        out = os.path.join(RESULTS_DIR, "t1_path1_summary.json")
        with open(out, "w") as f:
            json.dump(_jsonable(summary), f, indent=2)
        print(f"\nwrote {out}\nwrote {shard}")
    return summary


# ---------------------------------------------------------------------------
# Selftest — known-answer checks of the selection + verdict wiring, NO training.
# ---------------------------------------------------------------------------


def run_selftest() -> bool:
    """Checks keep-best-by-train selection, the oracle constant, and the construction-bar/verdict wiring.

    (a) `run_multistart_depth`'s keep-best picks the restart with the max TRAIN mean LL (not test) — a
        synthetic case where the best-train restart is deliberately NOT the best-test restart.
    (b) the oracle constant matches the adjudication's +0.8836.
    (c) a planted "region B climbs d1->d2->d3, region A flat" fixed-depth table reads
        `region_b_pass=True` / RESCUABLE, and a "region B flat far below oracle" table reads
        `region_b_pass=False` / HARDENED_UNLEARNABLE.
    """
    ok = True

    # (a) keep-best-by-train, verified against a monkeypatched trainer so no torch runs.
    rng = np.random.default_rng(0)
    n_te = 60
    region = np.array([0] * (n_te // 2) + [1] * (n_te // 2))
    x_te_t = torch.zeros((n_te, 1), dtype=torch.float32)
    y_te_t = torch.zeros(n_te, dtype=torch.float32)
    # restart 1 has the BEST train LL but a WORSE test LL than restart 0 -> selection must pick 1.
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
        # first call in each restart is on TRAIN (x has n_train rows), second is on TEST (n_te rows)
        if x_t.shape[0] == n_te:
            calls["i"] += 1
            return test_vecs[i]
        return np.full(4, train_lls[i])  # train vector (its mean is the selection metric)

    # Patch the EXECUTING module's globals (not a re-import: when run as __main__, `import
    # capacity_ladder_t1_path1` would load a distinct second copy whose globals are not the ones
    # run_multistart_depth actually looks up).
    g = globals()
    orig_build, orig_ll = g["_build_model"], g["_fixed_depth_log_likelihood"]
    g["_build_model"], g["_fixed_depth_log_likelihood"] = _fake_build, _fake_fit_ll
    try:
        res = run_multistart_depth(np.zeros((4, 1), dtype=np.float32), np.zeros(4, dtype=np.float32), x_te_t, y_te_t, region, 3, 3, 3, [])
    finally:
        g["_build_model"], g["_fixed_depth_log_likelihood"] = orig_build, orig_ll
    ok_a = res["kept_restart"] == 1 and abs(res["kept_test_mean_ll_region_b"] - (-0.20)) < 1e-9
    kept_b = res["kept_test_mean_ll_region_b"]
    print(f"[selftest a] keep-best-by-train picked restart {res['kept_restart']} (want 1), kept test-B={kept_b:+.3f} (want -0.200)  {'PASS' if ok_a else 'FAIL'}")
    ok = ok and ok_a

    # (b) oracle constant.
    ok_b = abs(ORACLE_LL - 0.8836) < 1e-3
    print(f"[selftest b] oracle LL = {ORACLE_LL:+.4f} (want +0.8836)  {'PASS' if ok_b else 'FAIL'}")
    ok = ok and ok_b

    # (c) construction-bar + verdict wiring on planted tables.
    n = 400
    reg = np.array([0] * (n // 2) + [1] * (n // 2))
    mb, ma = reg == 1, reg == 0
    climb = {}  # region B improves d1<d2<d3 then flat; region A flat at oracle
    for d in range(1, MAX_DEPTH + 1):
        v = np.empty(n)
        v[ma] = ORACLE_LL + rng.normal(0, 0.001, ma.sum())
        b_level = {1: -0.20, 2: 0.30, 3: 0.60}.get(d, 0.60)
        v[mb] = b_level + rng.normal(0, 0.001, mb.sum())
        climb[d] = v
    con_climb = _construction_bar(climb, reg)
    per_depth_climb = [{"depth": d, "kept_test_mean_ll_region_b": float(climb[d][mb].mean()), "best_by_test_region_b": float(climb[d][mb].mean())} for d in climb]
    ver_climb = _verdict(con_climb, per_depth_climb)
    ok_c1 = con_climb["region_b_pass"] and ver_climb["verdict"] == "RESCUABLE"
    print(f"[selftest c1] planted-climb: region_b_pass={con_climb['region_b_pass']} verdict={ver_climb['verdict']} (want True/RESCUABLE)  {'PASS' if ok_c1 else 'FAIL'}")

    flat = {}  # region B flat ~1.1 nat below oracle at every depth; region A at oracle
    for d in range(1, MAX_DEPTH + 1):
        v = np.empty(n)
        v[ma] = ORACLE_LL + rng.normal(0, 0.001, ma.sum())
        v[mb] = (ORACLE_LL - 1.1) + rng.normal(0, 0.001, mb.sum())
        flat[d] = v
    con_flat = _construction_bar(flat, reg)
    per_depth_flat = [{"depth": d, "kept_test_mean_ll_region_b": float(flat[d][mb].mean()), "best_by_test_region_b": float(flat[d][mb].mean())} for d in flat]
    ver_flat = _verdict(con_flat, per_depth_flat)
    ok_c2 = (not con_flat["region_b_pass"]) and ver_flat["verdict"] == "HARDENED_UNLEARNABLE"
    print(f"[selftest c2] planted-flat: region_b_pass={con_flat['region_b_pass']} verdict={ver_flat['verdict']} (want False/HARDENED_UNLEARNABLE)  {'PASS' if ok_c2 else 'FAIL'}")
    ok = ok and ok_c1 and ok_c2

    print(f"[selftest] {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    """Parses args and runs the selftest, a fast smoke run, or the real seed-0 R=8 disambiguation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="Run the no-training known-answer selftest and exit.")
    parser.add_argument("--smoke", action="store_true", help="Fast real-training plumbing check (2 restarts, 3 epochs); does not overwrite the real summary.")
    args = parser.parse_args()

    if args.selftest:
        sys.exit(0 if run_selftest() else 1)
    if args.smoke:
        run_path1(n_restarts=2, n_epochs=3, save=False)
        return
    run_path1()


if __name__ == "__main__":
    main()
