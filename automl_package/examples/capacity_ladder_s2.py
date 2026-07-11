"""S2 — direct held-out-likelihood selector (the principled objective), per-input selector program.

S1 (`capacity_ladder_s1.py`) ran a target-construction FACTORIAL: every arm trains the router by
imitating some derived per-input label or responsibility (soft, soft-no-prior, soft-smoothed,
hard-knee, raw-argmax), then scores it with a blended held-out log-score. The winner was `soft`
(per-tercile-prior responsibilities, mean blend NLL 0.6807 on the structured toys — the lowest
of the five arms; `capacity_ladder_results/S1/s1_summary.json`). S2 asks the question the
factorial never asked directly: what if the router is trained ON the deployment metric itself,
with no derived label in between?

Objective (train half only, L5-compliant — a proper held-out log-score, no label, no prior, no
tuned lambda): maximize the router's own per-input BLEND log-score against the frozen train-half
score table,

    mean_i logsumexp_c( log_softmax(w(x_i))_c + score_tr[i, c] )

which is exactly `capacity_ladder_s1._blend_nll`'s own computation, evaluated on the TRAIN half
with gradients flowing into the router only (`score_tr` is a detached constant tensor — the
ladder itself is never touched). `_train_router_direct` (below) is the new trainer this requires;
`capacity_ladder_k6._train_router` only supports cross-entropy to a precomputed label/target and
cannot express this loss. Router architecture, optimizer, and the 300-epoch schedule are held
IDENTICAL to S1/K6 (`_RouterMLP`, hidden (32,32), Adam lr 1e-2) so the comparison isolates the
objective, not the model; a second 3000-epoch arm is also trained per case to check whether 300
epochs under-trains the direct objective (reported, not separately gated).

Every case reuses S1's split (`capacity_ladder_s1.run_case`'s pattern: TRAIN = even rows, EVAL =
odd rows, EVAL split again by index parity into eval-A/eval-B for the honest `oracle-x` bound)
and S1's evaluation machinery verbatim (`cs1._eval_arm` for blend + hard NLL, `cs1._oracle_reads`
for the honest bound, `cs1._paired_bootstrap_se` for paired-diff SEs). The S1 `soft` winner is
rebuilt on the SAME split via `cs1._train_arm("soft", ...)` for a same-data head-to-head — S1's
`run_case` is not reused directly because it trains all five factorial arms, four of which are
irrelevant here.

Run `--selftest` before any real read (synthetic known-answer table, in-memory, no disk, reusing
`ck6._selftest_table`). Run with no flags to read every K4 table on disk (missing files are
skipped, matching S1/K6/K5).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as nnf

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root, so `import automl_package` works

import _capacity_ladder as cl  # noqa: E402
import capacity_ladder_k6 as ck6  # noqa: E402 — reuse _RouterMLP/read_table/_selftest_table/_jsonable verbatim
import capacity_ladder_s1 as cs1  # noqa: E402 — reuse _train_arm/_eval_arm/_oracle_reads/_paired_bootstrap_se/protocol constants

from automl_package.utils.pytorch_utils import get_device  # noqa: E402

OUT_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "S2")

N_EPOCHS = ck6.N_EPOCHS  # 300, matches S1/K6 router config
N_EPOCHS_LONG = 3000  # under-training check arm (spec: "ALSO run a 3000-epoch arm")
LR = ck6.LR  # 1e-2
DIRECT_ARM_PRIMARY = "direct_300ep"  # the config-matched arm — the "S2" bars (i)-(iii) are read against this one
DIRECT_ARM_LONG = "direct_3000ep"  # diagnostic-only (under-training check); folds into the X7 "best of {S2, soft}" read


# ---------------------------------------------------------------------------
# The direct objective — a new trainer (K6's `_train_router` cannot express this loss).
# ---------------------------------------------------------------------------


def _direct_loss(model: ck6._RouterMLP, x_t: torch.Tensor, score_t: torch.Tensor) -> torch.Tensor:
    """Negative mean per-input blended log-score — the S2 training objective (minimized).

    Identical computation to `cs1._blend_nll`, kept differentiable and evaluated on the TRAIN
    half: `score_t` is a frozen constant (never itself receiving gradient), so the only thing
    gradient descent can move is the router's own soft weights.
    """
    log_w = nnf.log_softmax(model(x_t), dim=1).double()
    blend = torch.logsumexp(log_w + score_t, dim=1)
    return -blend.mean()


def _train_router_direct(
    x: np.ndarray,
    score: np.ndarray,
    *,
    n_cols: int,
    device: str,
    n_epochs: int = N_EPOCHS,
    lr: float = LR,
    seed: int = 0,
) -> ck6._RouterMLP:
    """Trains `w(x)` directly on the deployment objective: maximize mean_i logsumexp_c(log softmax(w(x_i))_c + score[i,c]).

    No derived label, no prior, no tuned lambda (L5) — an ordinary held-out log-score operation
    optimized by gradient descent, on the TRAIN half only. This is the arm the S1 factorial never
    ran: every S1 arm imitates some derived per-input target; this one optimizes the blend metric
    it will be scored on directly.

    Args:
        x: `(N,)` scalar input coordinate, TRAIN half.
        score: `(N, n_cols)` held-out log-likelihood table, TRAIN half (frozen — never backprop
            into it; only the router's own parameters are optimized).
        n_cols: number of router output columns (== `len(c_grid)`).
        device: torch device string.
        n_epochs: full-batch Adam epochs.
        lr: Adam learning rate.
        seed: torch RNG seed for init (reproducibility).

    Returns:
        The trained `_RouterMLP`, in eval mode.
    """
    torch.manual_seed(seed)
    model = ck6._RouterMLP(n_cols).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x_t = torch.as_tensor(np.asarray(x, dtype=np.float32), device=device).reshape(-1, 1)
    score_t = torch.as_tensor(np.asarray(score, dtype=np.float64), device=device).detach()

    model.train()
    for _ in range(n_epochs):
        opt.zero_grad()
        loss = _direct_loss(model, x_t, score_t)
        loss.backward()
        opt.step()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# One case (one toy/seed table): S1's `soft` winner rebuilt + the two S2 direct-objective arms.
# ---------------------------------------------------------------------------


def run_case(toy: str, seed: int, device: str) -> dict | None:
    """Runs one K4 table: rebuilds S1's `soft` winner + trains the S2 direct-objective arms, same split.

    Deterministic index-parity split, identical to `cs1.run_case`: TRAIN = even rows, EVAL = odd
    rows; EVAL is split again by index parity into eval-A (even-of-eval) / eval-B (odd-of-eval)
    for the honest `oracle-x` bound. Targets/objectives are built on TRAIN only.

    Args:
        toy: toy name (e.g. "D", "C_broad").
        seed: table seed (0, 1, or 2).
        device: torch device string.

    Returns:
        A dict of raw per-case numbers (no pass/fail — bars are aggregated across cases in
        `main()`), or None if the K4 table for `(toy, seed)` doesn't exist yet.
    """
    tbl = ck6.read_table(toy, seed)
    if tbl is None:
        return None
    score, x, c_grid = tbl["score"], tbl["x"], tbl["c_grid"]
    n = score.shape[0]

    train_idx = np.arange(0, n, 2)
    eval_idx = np.arange(1, n, 2)
    score_tr, x_tr = score[train_idx], x[train_idx]
    score_ev, x_ev = score[eval_idx], x[eval_idx]
    score_ev_a, x_ev_a = score_ev[0::2], x_ev[0::2]
    score_ev_b, x_ev_b = score_ev[1::2], x_ev[1::2]

    arms: dict[str, dict[str, float]] = {}

    model_soft = cs1._train_arm("soft", score_tr, x_tr, c_grid, device, seed)
    nll_blend, nll_hard = cs1._eval_arm(model_soft, x_ev, score_ev, device)
    arms["soft"] = {"nll_blend": nll_blend, "nll_hard": nll_hard}

    for tag, n_epochs in ((DIRECT_ARM_PRIMARY, N_EPOCHS), (DIRECT_ARM_LONG, N_EPOCHS_LONG)):
        model_direct = _train_router_direct(x_tr, score_tr, n_cols=len(c_grid), device=device, n_epochs=n_epochs, seed=seed)
        nll_blend, nll_hard = cs1._eval_arm(model_direct, x_ev, score_ev, device)
        arms[tag] = {"nll_blend": nll_blend, "nll_hard": nll_hard}

    col_of = {c: j for j, c in enumerate(c_grid)}
    r_star, _delta_curve, _se = cl.knee(score_tr, ref_c=1, n_boot=cs1.BOOT_N, c_grid=c_grid, seed=seed)
    k_global = 1 if r_star == 0 else r_star  # G2 abstain => the k=1 bypass single Gaussian
    global_col = col_of[k_global]
    nll_global = ck6._routed_nll(score_ev, np.full(score_ev.shape[0], global_col, dtype=np.int64))

    oracle_x_score, _route_col = cs1._oracle_reads(score_ev_a, x_ev_a, score_ev_b, x_ev_b, ck6.WIDTH)
    oracle_noisy_score = float(score_ev.max(axis=1).mean())  # continuity label only, as in S1

    return {
        "toy": toy,
        "seed": seed,
        "n_train": len(train_idx),
        "n_eval": len(eval_idx),
        "k_global": int(k_global),
        "nll_global": nll_global,
        "arms": arms,
        "oracle_x_score": oracle_x_score,
        "oracle_noisy_score": oracle_noisy_score,
        "oracle_x_nll": -oracle_x_score,
        "oracle_noisy_nll": -oracle_noisy_score,
    }


# ---------------------------------------------------------------------------
# Selftest — reuses K6's synthetic known-answer table, in-memory, no disk.
# ---------------------------------------------------------------------------


def run_selftest() -> bool:
    """Runs the S2 selftest: the direct objective must reach blend NLL <= S1-soft blend NLL + 0.005."""
    device = str(get_device())
    print(f"S2 selftest (N={ck6._ST_N}, C={len(ck6._ST_C_GRID)}, device={device})")
    score, x = ck6._selftest_table(seed=0)
    c_grid = ck6._ST_C_GRID
    n = score.shape[0]

    train_idx = np.arange(0, n, 2)
    eval_idx = np.arange(1, n, 2)
    score_tr, x_tr = score[train_idx], x[train_idx]
    score_ev, x_ev = score[eval_idx], x[eval_idx]

    model_soft = cs1._train_arm("soft", score_tr, x_tr, c_grid, device, seed=0)
    nll_blend_soft, _nll_hard_soft = cs1._eval_arm(model_soft, x_ev, score_ev, device)

    model_direct = _train_router_direct(x_tr, score_tr, n_cols=len(c_grid), device=device, n_epochs=N_EPOCHS, seed=0)
    nll_blend_direct, nll_hard_direct = cs1._eval_arm(model_direct, x_ev, score_ev, device)

    ok = nll_blend_direct <= nll_blend_soft + 0.005
    print(f"  soft-imitation blend NLL={nll_blend_soft:.4f}; direct-objective blend NLL={nll_blend_direct:.4f} (hard={nll_hard_direct:.4f})")
    print(f"  direct <= soft + 0.005 ({nll_blend_direct:.4f} <= {nll_blend_soft + 0.005:.4f}): {'PASS' if ok else 'FAIL'}")
    print("all checks passed" if ok else "FAILURES PRESENT")
    return ok


# ---------------------------------------------------------------------------
# Real read: every K4 table on disk, the 3 pre-registered bars, the X7-trigger arithmetic.
# ---------------------------------------------------------------------------


def _bar_i(structured_cases: list[dict]) -> dict:
    """(i) S2 (direct_300ep) blend NLL <= S1-soft blend NLL on >= 6/9 structured cases AND mean paired diff <= 0."""
    diffs = np.array([c["arms"][DIRECT_ARM_PRIMARY]["nll_blend"] - c["arms"]["soft"]["nll_blend"] for c in structured_cases])
    n_pass = int(np.sum(diffs <= 0))
    return {
        "n_pass": n_pass,
        "n_cases": len(structured_cases),
        "mean_diff": float(diffs.mean()),
        "se": cs1._paired_bootstrap_se(diffs),
        "pass": bool(n_pass >= 6 and diffs.mean() <= 0),
    }


def _bar_ii(broad_cases: list[dict]) -> dict:
    """(ii) S2 (direct_300ep) blend advantage over global (`nll_global - nll_blend`) <= 0.02 nat, all 6 broad cases."""
    advantages = [c["nll_global"] - c["arms"][DIRECT_ARM_PRIMARY]["nll_blend"] for c in broad_cases]
    return {"advantages": advantages, "max_advantage": max(advantages) if advantages else None, "pass": all(a <= 0.02 for a in advantages)}


def _bar_iii(structured_cases: list[dict]) -> dict:
    """(iii) S2 (direct_300ep) hard-routed NLL <= global on >= 8/9 structured cases (degrades gracefully)."""
    n_pass = sum(1 for c in structured_cases if c["arms"][DIRECT_ARM_PRIMARY]["nll_hard"] <= c["nll_global"])
    return {"n_pass": n_pass, "n_cases": len(structured_cases), "pass": n_pass >= 8}


def _x7_trigger(structured_cases: list[dict]) -> dict:
    """Toy-D gap closure: `mean_seeds((nll_global - best_selector_blend) / (nll_global - oracle_x_nll))`.

    `best_selector_blend` = min blend NLL over {soft, direct_300ep, direct_3000ep} per case — the
    S1 winner and both S2 arms, per the dispatch spec's "min blend NLL over {S2, soft}" (S2 here
    spans both trained epoch schedules). All quantities are NLLs (lower is better) throughout, so
    `nll_global - best_selector_blend` is the gap actually closed and `nll_global - oracle_x_nll`
    is S1's honest achievable gap; `x7_trigger_fires` per L6/EXECUTION_PLAN.md §5: closure < 0.5
    means the deployable-selector question is still open (present at G-X7, do not ask otherwise).
    """
    toy_d = [c for c in structured_cases if c["toy"] == "D"]
    closures = []
    for c in toy_d:
        best_selector_blend = min(c["arms"]["soft"]["nll_blend"], c["arms"][DIRECT_ARM_PRIMARY]["nll_blend"], c["arms"][DIRECT_ARM_LONG]["nll_blend"])
        denom = c["nll_global"] - c["oracle_x_nll"]
        closures.append((c["nll_global"] - best_selector_blend) / denom if denom != 0 else float("nan"))
    gap_closure = float(np.mean(closures)) if closures else float("nan")
    return {"toy": "D", "n_seeds": len(closures), "per_seed_closure": closures, "gap_closure": gap_closure, "x7_trigger_fires": bool(gap_closure < 0.5)}


def main() -> None:
    """Reads every available K4 table, runs the S2 head-to-head + bars + X7-trigger arithmetic, writes the summary json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="Run the S2 synthetic known-answer selftest and exit.")
    args = parser.parse_args()

    if args.selftest:
        ok = run_selftest()
        sys.exit(0 if ok else 1)

    os.makedirs(OUT_DIR, exist_ok=True)
    device = str(get_device())
    print(f"[s2] device={device}")

    all_cases: list[dict] = []
    for toy in (*cs1.STRUCTURED_TOYS, *cs1.BROAD_TOYS):
        for seed in cs1.SEEDS:
            case = run_case(toy, seed, device)
            if case is None:
                print(f"[s2] {toy} s{seed}: no K4 table yet, skipping")
                continue
            all_cases.append(case)
            print(
                f"[s2] {toy} s{seed}: blend NLL soft={case['arms']['soft']['nll_blend']:.4f} "
                f"{DIRECT_ARM_PRIMARY}={case['arms'][DIRECT_ARM_PRIMARY]['nll_blend']:.4f} "
                f"{DIRECT_ARM_LONG}={case['arms'][DIRECT_ARM_LONG]['nll_blend']:.4f}  "
                f"global={case['nll_global']:.4f} oracle_x={case['oracle_x_nll']:.4f}"
            )

    structured_cases = [c for c in all_cases if c["toy"] in cs1.STRUCTURED_TOYS]
    broad_cases = [c for c in all_cases if c["toy"] in cs1.BROAD_TOYS]

    bars = {
        "i_s2_ge_soft": _bar_i(structured_cases),
        "ii_broad_advantage_bounded": _bar_ii(broad_cases),
        "iii_hard_read_beats_global": _bar_iii(structured_cases),
    }
    for name, res in bars.items():
        print(f"[s2] bar {name}: {res}")

    x7 = _x7_trigger(structured_cases)
    print(f"[s2] X7 trigger: {x7}")

    summary = {
        "config": {
            "hidden": list(ck6.HIDDEN),
            "n_epochs_primary": N_EPOCHS,
            "n_epochs_long": N_EPOCHS_LONG,
            "lr": LR,
            "seeds": list(cs1.SEEDS),
            "direct_arm_primary": DIRECT_ARM_PRIMARY,
            "direct_arm_long": DIRECT_ARM_LONG,
        },
        "structured_cases": structured_cases,
        "broad_cases": broad_cases,
        "bars": bars,
        "x7_trigger": x7,
    }
    out_path = os.path.join(OUT_DIR, "s2_summary.json")
    with open(out_path, "w") as f:
        json.dump(ck6._jsonable(summary), f, indent=2)
    print(f"[s2] wrote {out_path}")


if __name__ == "__main__":
    main()
