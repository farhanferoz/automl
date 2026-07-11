"""S1 — evaluation protocol + target-construction factorial (ProbReg), per-input selector program.

Successor to `capacity_ladder_k6.py`'s 3-arm HARD/SOFT/PILOT comparison. K6 established that
SOFT (imitating per-bin-stacked responsibilities) beats HARD (imitating the noisy per-input
knee label) under a routed-NLL read. S1 replaces that confirmatory comparison with (1) a clean
5-arm target-construction FACTORIAL that isolates the two mechanisms K6's SOFT arm bundled
together — the per-tercile PRIOR and the neighbour SMOOTHING — and (2) a corrected evaluation
protocol: a per-input WEIGHTED-BLEND read (`_capacity_ladder_2026-07-10/EXECUTION_PLAN.md` L3:
every capacity costs one forward pass under prefix renormalization, so the deliverable is a
soft weighting head, not a hard pick) alongside the old hard argmax read, plus an HONEST
oracle bound (`oracle-x`, routed from a disjoint half of the data — no peeking at a point's own
held-out score) to replace the old oracle's per-point cheating max (kept as `oracle-noisy`, a
continuity label only).

Reads the K4 nested score tables (`capacity_ladder_results/K4/nested_toy*_seed*.pt`, already on
disk — no ladder retraining). Reuses `capacity_ladder_k6.py` verbatim for `_RouterMLP`,
`_train_router`, `_route`, `_routed_nll`, `_knee_labels`, `_soft_targets`, `read_table`,
`_selftest_table`, `_jsonable`; reuses `_capacity_ladder.py` for `perinput_curve`, `perbin_stack`,
`quantile_bins`, `knee`.

Five factorial arms (identical router config; only the TRAIN-half target changes):
  (1) soft            = K6 SOFT: responsibilities `softmax_c(score[i,c] + log pi_bin(i))`,
                         `pi_bin` a per-tercile EM-stacked prior (`ck6._soft_targets`).
  (2) soft_no_prior    = per-row `softmax_c(score[i,:])`, no prior, no smoothing — isolates
                         the prior's contribution (1v2).
  (3) soft_smoothed    = softmax of the NEIGHBOUR-AVERAGED score rows (box-car, `ck6.WIDTH`),
                         no prior — isolates smoothing's contribution (3v2), and contrasts
                         against the prior mechanism directly (1v3).
  (4) hard_knee        = K6 HARD: cross-entropy to the neighbour-averaged per-input knee label.
  (5) raw_argmax       = K6 PILOT: cross-entropy to the raw per-row argmax (no averaging at all,
                         isolates plain softness vs argmax commitment, 2v5).

Every arm is evaluated on the router-EVAL half two ways: BLEND (primary, per L3) — the
router's own soft weights combined with the score table via `logsumexp`, a valid held-out
log-score of a weighted mixture, no labels/lambda involved — and HARD — argmax-routed NLL
(K6's read). A GLOBAL baseline (K6's global-knee route, one-hot weight) and an honest
`oracle-x` bound (routed via a DISJOINT half's neighbour-averaged advantage curve, see
`_oracle_reads`) round out the comparison set. Strictly probabilistic throughout (L5): every
number here is a likelihood, a proper-score average, or a prior — no tuned penalty anywhere.

KNOB SWEEP (winning arm's soft-target recipe, `_soft_targets_knobs`): prior bins ∈ {2,3,5},
neighbour width ∈ {0.0375,0.075,0.15}, temperature τ ∈ {0.5,1,2} (soft targets raised to
power 1/τ then renormalized; τ=1 is a no-op). Judgment call (flagged to the orchestrator,
see the dispatch reply, not guessed silently): `_soft_targets_knobs` exposes prior-bins AND
smoothing-width as two INDEPENDENT knobs that can be on or off together, so it strictly
generalizes arms (1)/(2)/(3). If the factorial winner is a soft arm (1/2/3), the knob sweep
uses that arm's own (n_bins, width) pair as the two knobs' identity and only varies the
knob(s) that arm's recipe actually has (the OTHER knob stays at the winner's own value, which
means at most 2 of the 3 axes vary per winner — width never varies for arm (1), n_bins never
varies for arm (3), both vary for arm (2)). Temperature always varies. If the winner is a
HARD arm (4/5) — for which "prior bins"/"temperature" have no meaning — the sweep falls back
to the best-performing SOFT arm among (1)/(2)/(3) and logs the substitution explicitly.

Run `--selftest` before any real read (synthetic known-answer table, in-memory, no disk,
reusing `ck6._selftest_table`). Run with no flags to read every K4 table on disk (missing
files are skipped, matching K5/K6).
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
import capacity_ladder_k6 as ck6  # noqa: E402 — reuse router/target/read/selftest machinery verbatim

from automl_package.utils.pytorch_utils import get_device  # noqa: E402

OUT_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "S1")

SEEDS = (0, 1, 2)
STRUCTURED_TOYS = ("C", "D", "E")
BROAD_TOYS = ("C_broad", "E_broad")
ARM_NAMES = ("soft", "soft_no_prior", "soft_smoothed", "hard_knee", "raw_argmax")
SOFT_ARMS = ("soft", "soft_no_prior", "soft_smoothed")
BOOT_N = ck6.BOOT_N  # 1000, matches K6's global-knee bootstrap

# knob-sweep grid (winning arm's recipe only; toys D + C_broad, all seeds).
KNOB_TOYS = ("D", "C_broad")
KNOB_BINS = (2, 3, 5)
KNOB_WIDTH = (0.0375, 0.075, 0.15)
KNOB_TAU = (0.5, 1.0, 2.0)

# (n_bins, width) identity for each soft arm's own recipe — see `_soft_targets_knobs` docstring
# for how these reduce to arms (1)/(2)/(3); `None` disables that mechanism.
_ARM_RECIPE: dict[str, dict[str, float | None]] = {
    "soft": {"n_bins": 3, "width": None},
    "soft_no_prior": {"n_bins": None, "width": None},
    "soft_smoothed": {"n_bins": None, "width": ck6.WIDTH},
}


# ---------------------------------------------------------------------------
# Target construction — the 5 factorial arms + the knob-parametrized generalization.
# ---------------------------------------------------------------------------


def _softmax_rows(mat: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax (max-subtracted)."""
    z = mat - mat.max(axis=1, keepdims=True)
    w = np.exp(z)
    return w / w.sum(axis=1, keepdims=True)


def _soft_targets_knobs(
    score_tr: np.ndarray,
    x_tr: np.ndarray,
    *,
    n_bins: int | None,
    width: float | None,
    tau: float = 1.0,
) -> np.ndarray:
    """General knob-parametrized soft-target constructor: prior bins x smoothing width x temperature.

    Reduces to the S1 factorial's soft arms as special cases (bit-identical, checked by the
    selftest for the `n_bins=3, width=None, tau=1.0` case against `ck6._soft_targets`):
      * `n_bins=3, width=None, tau=1.0`  -> arm (1) soft.
      * `n_bins=None, width=None, tau=1.0` -> arm (2) soft_no_prior (plain row softmax).
      * `n_bins=None, width=ck6.WIDTH, tau=1.0` -> arm (3) soft_smoothed.

    `width=None` disables neighbourhood smoothing (uses the raw per-row score as the softmax
    base); `n_bins in (None, 1)` disables the per-bin prior (`log_prior == 0` everywhere). The
    prior is always computed off the RAW `score_tr` (matching `ck6._soft_targets`) regardless
    of the smoothing knob — smoothing only affects the softmax base, not the prior's own fit.
    `tau` applies AFTER normalization: targets are raised to power `1/tau` and renormalized;
    `tau == 1.0` is skipped entirely (a true no-op, not merely a numerically-close one).

    Args:
        score_tr: `(N, C)` held-out log-likelihood table, TRAIN half only.
        x_tr: `(N,)` input coordinate paired with each row of `score_tr`.
        n_bins: number of quantile bins for the per-bin stacked prior, or None/1 to disable it.
        width: neighbour box-car half-width for smoothing, or None to disable smoothing.
        tau: target temperature (targets ∝ target^(1/tau)); 1.0 is a no-op.

    Returns:
        `(N, C)` per-row target distribution, rows sum to 1.
    """
    if width is None:
        base = np.asarray(score_tr, dtype=np.float64)
    else:
        out = cl.perinput_curve(score_tr, x_tr, width, ref_c=0, query_x=x_tr)
        base = np.asarray(out["delta"], dtype=np.float64)  # softmax-shift-invariant surrogate for the smoothed score

    if n_bins is None or n_bins <= 1:
        log_prior = np.zeros_like(base)
    else:
        bins = cl.quantile_bins(x_tr, n_bins)
        pi_bins = cl.perbin_stack(score_tr, bins)
        log_prior = np.log(np.clip(np.stack([pi_bins[b] for b in bins], axis=0), 1e-300, None))

    w = _softmax_rows(base + log_prior)
    if tau != 1.0:
        w = np.power(np.clip(w, 1e-300, None), 1.0 / tau)
        w = w / w.sum(axis=1, keepdims=True)
    return w


def _build_targets(arm: str, score_tr: np.ndarray, x_tr: np.ndarray, c_grid: list[int]) -> dict:
    """Builds one factorial arm's TRAIN-half target; returns `{"kind": "soft"|"hard", ...}`."""
    if arm == "soft":
        return {"kind": "soft", "soft_targets": ck6._soft_targets(score_tr, x_tr)}
    if arm == "soft_no_prior":
        return {"kind": "soft", "soft_targets": _softmax_rows(score_tr)}
    if arm == "soft_smoothed":
        out = cl.perinput_curve(score_tr, x_tr, ck6.WIDTH, ref_c=0, query_x=x_tr)
        return {"kind": "soft", "soft_targets": _softmax_rows(np.asarray(out["delta"]))}
    if arm == "hard_knee":
        col_of = {c: j for j, c in enumerate(c_grid)}
        hard_c = ck6._knee_labels(score_tr, x_tr, x_tr, c_grid)
        return {"kind": "hard", "hard_labels": np.array([col_of[c] for c in hard_c], dtype=np.int64)}
    if arm == "raw_argmax":
        return {"kind": "hard", "hard_labels": score_tr.argmax(axis=1)}
    raise ValueError(f"unknown arm {arm!r}")


def _train_arm(arm: str, score_tr: np.ndarray, x_tr: np.ndarray, c_grid: list[int], device: str, seed: int) -> ck6._RouterMLP:
    """Builds `arm`'s TRAIN-half target and trains its router (K6 config: hidden (32,32), 300 ep, lr 1e-2)."""
    t = _build_targets(arm, score_tr, x_tr, c_grid)
    if t["kind"] == "soft":
        return ck6._train_router(x_tr, n_cols=len(c_grid), device=device, soft_targets=t["soft_targets"], seed=seed)
    return ck6._train_router(x_tr, n_cols=len(c_grid), device=device, hard_labels=t["hard_labels"], seed=seed)


# ---------------------------------------------------------------------------
# Evaluation — blend (primary) + hard reads, the honest oracle-x bound.
# ---------------------------------------------------------------------------


def _blend_nll(model: ck6._RouterMLP, x: np.ndarray, score: np.ndarray, device: str) -> float:
    """Blended NLL: `-mean_i logsumexp_c(log softmax(logits(x_i))_c + score[i,c])` (L3 primary read).

    A valid held-out log-score: the log of the router's own soft-weighted average of the
    per-capacity densities in `score` — no derived label, no tuned lambda (L5).
    """
    with torch.no_grad():
        x_t = torch.as_tensor(np.asarray(x, dtype=np.float32), device=device).reshape(-1, 1)
        log_w = nnf.log_softmax(model(x_t), dim=1).double()
    score_t = torch.as_tensor(np.asarray(score, dtype=np.float64), device=device)
    blend = torch.logsumexp(log_w + score_t, dim=1)
    return float(-blend.mean().item())


def _eval_arm(model: ck6._RouterMLP, x_ev: np.ndarray, score_ev: np.ndarray, device: str) -> tuple[float, float]:
    """Blend NLL + hard-routed NLL of one trained router on the eval half."""
    nll_blend = _blend_nll(model, x_ev, score_ev, device)
    col_hard = ck6._route(model, x_ev, device)
    nll_hard = ck6._routed_nll(score_ev, col_hard)
    return nll_blend, nll_hard


def _oracle_reads(score_ev_a: np.ndarray, x_ev_a: np.ndarray, score_ev_b: np.ndarray, x_ev_b: np.ndarray, width: float) -> tuple[float, np.ndarray]:
    """The honest `oracle-x` bound: routes eval-B via eval-A's neighbour-averaged advantage curve alone.

    `cl.perinput_curve` is evaluated with `query_x=x_ev_b` but its NEIGHBOURHOOD source is
    `(score_ev_a, x_ev_a)` only — no eval-B point ever informs its own route (the "nearest
    neighbour in x" language in the spec is exactly this box-car mechanism, not a separate
    step). The argmax-routed capacity's ACTUAL score on eval-B is reported, raw (not
    negated) — the spec's "mean actual score" — so bar (iii)'s `oracle-x <= oracle-noisy`
    compares like-for-like: both are RAW mean log-scores, higher is better, and `oracle-x`
    (information-constrained, no peeking at a point's own y) can never legitimately exceed
    `oracle-noisy` (the per-point cheating max) in that raw-score sense — see `main()`'s bar
    (iii) comment for the direction check spelled out in NLL terms too.

    Args:
        score_ev_a: `(NA, C)` neighbourhood-source held-out log-likelihood table.
        x_ev_a: `(NA,)` neighbourhood-source input coordinate.
        score_ev_b: `(NB, C)` routed-and-scored held-out log-likelihood table.
        x_ev_b: `(NB,)` routed-and-scored input coordinate.
        width: box-car half-width (`ck6.WIDTH` at the primary read).

    Returns:
        `(oracle_x_score, route_col)`: the raw mean actual score on eval-B, and the `(NB,)`
        routed column index per eval-B row (reused by the selftest's tercile-peak check).
    """
    out = cl.perinput_curve(score_ev_a, x_ev_a, width, ref_c=0, query_x=x_ev_b)
    delta = np.asarray(out["delta"])
    route_col = delta.argmax(axis=1)
    rows = np.arange(score_ev_b.shape[0])
    oracle_x_score = float(score_ev_b[rows, route_col].mean())
    return oracle_x_score, route_col


# ---------------------------------------------------------------------------
# Stats — plain paired bootstrap SE over per-case diffs (no existing case-level helper to reuse).
# ---------------------------------------------------------------------------


def _paired_bootstrap_se(diffs: np.ndarray, n_boot: int = BOOT_N, seed: int = 0) -> float:
    """SE of the mean via i.i.d. resampling of the paired per-case diffs (case-level bootstrap)."""
    rng = np.random.default_rng(seed)
    n = len(diffs)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs[idx].mean(axis=1)
    return float(boot_means.std(ddof=1))


def _factorial_read(cases: list[dict], arm_a: str, arm_b: str) -> dict:
    """Paired per-case blend-NLL diff (`arm_b - arm_a`) + bootstrap SE, over `cases`.

    Positive `mean_diff` means `arm_a` has the LOWER (better) blend NLL of the pair
    (diffs = arm_b - arm_a, so arm_a is better iff arm_b - arm_a > 0).
    """
    diffs = np.array([c["arms"][arm_b]["nll_blend"] - c["arms"][arm_a]["nll_blend"] for c in cases])
    return {"arm_a": arm_a, "arm_b": arm_b, "mean_diff": float(diffs.mean()), "se": _paired_bootstrap_se(diffs), "n": len(diffs), "a_better": bool(diffs.mean() > 0)}


# ---------------------------------------------------------------------------
# One case (one toy/seed table): splits, the 5-arm factorial, global + oracle reads.
# ---------------------------------------------------------------------------


def run_case(toy: str, seed: int, device: str) -> dict | None:
    """Runs the full S1 protocol on one K4 table: 5-arm factorial + global + oracle-x/-noisy.

    Deterministic index-parity split (K6's pattern, no unseeded RNG): TRAIN = even rows,
    EVAL = odd rows; EVAL is split again by index parity into eval-A (even-of-eval) / eval-B
    (odd-of-eval) for the honest `oracle-x` bound. Targets are built on TRAIN only.

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
    for arm in ARM_NAMES:
        model = _train_arm(arm, score_tr, x_tr, c_grid, device, seed)
        nll_blend, nll_hard = _eval_arm(model, x_ev, score_ev, device)
        arms[arm] = {"nll_blend": nll_blend, "nll_hard": nll_hard}

    col_of = {c: j for j, c in enumerate(c_grid)}
    r_star, _delta_curve, _se = cl.knee(score_tr, ref_c=1, n_boot=BOOT_N, c_grid=c_grid, seed=seed)
    k_global = 1 if r_star == 0 else r_star  # G2 abstain => the k=1 bypass single Gaussian
    global_col = col_of[k_global]
    nll_global = ck6._routed_nll(score_ev, np.full(score_ev.shape[0], global_col, dtype=np.int64))

    oracle_x_score, _route_col = _oracle_reads(score_ev_a, x_ev_a, score_ev_b, x_ev_b, ck6.WIDTH)
    oracle_noisy_score = float(score_ev.max(axis=1).mean())  # k6's nll_oracle, un-negated; full eval half (continuity label)

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
    """Runs the S1 selftest on K6's synthetic table: checks (a) blend>=hard, (b) oracle-x, (c) tau=1."""
    device = str(get_device())
    print(f"S1 selftest (N={ck6._ST_N}, C={len(ck6._ST_C_GRID)}, device={device})")
    score, x = ck6._selftest_table(seed=0)
    c_grid = ck6._ST_C_GRID
    n = score.shape[0]

    train_idx = np.arange(0, n, 2)
    eval_idx = np.arange(1, n, 2)
    score_tr, x_tr = score[train_idx], x[train_idx]
    score_ev, x_ev = score[eval_idx], x[eval_idx]
    score_ev_a, x_ev_a = score_ev[0::2], x_ev[0::2]
    score_ev_b, x_ev_b = score_ev[1::2], x_ev[1::2]

    # (a) blend >= hard for the soft arm, i.e. blend NLL <= hard NLL (see module docstring's L3 note).
    model_soft = _train_arm("soft", score_tr, x_tr, c_grid, device, seed=0)
    nll_blend, nll_hard = _eval_arm(model_soft, x_ev, score_ev, device)
    ok_a = nll_blend <= nll_hard
    print(f"  (a) soft arm: blend NLL={nll_blend:.4f} <= hard NLL={nll_hard:.4f}: {'PASS' if ok_a else 'FAIL'}")

    # (b) oracle-x recovers the designed tercile peaks {1,3,6}.
    _oracle_x_score, route_col = _oracle_reads(score_ev_a, x_ev_a, score_ev_b, x_ev_b, ck6.WIDTH)
    routed_c = np.asarray(c_grid)[route_col]
    tercile_bins = cl.quantile_bins(x_ev_b, 3)
    modal_by_tercile = []
    for b in sorted(np.unique(tercile_bins)):
        vals, counts = np.unique(routed_c[tercile_bins == b], return_counts=True)
        modal_by_tercile.append(int(vals[np.argmax(counts)]))
    ok_b = modal_by_tercile == list(ck6._ST_PEAKS)
    print(f"  (b) oracle-x modal routed capacity by tercile = {modal_by_tercile} (designed {list(ck6._ST_PEAKS)}): {'PASS' if ok_b else 'FAIL'}")

    # (c) tau=1 reproduces the tau-free soft target bit-identically (max abs diff 0.0).
    q_tau1 = _soft_targets_knobs(score_tr, x_tr, n_bins=3, width=None, tau=1.0)
    q_ref = ck6._soft_targets(score_tr, x_tr)
    max_abs_diff = float(np.max(np.abs(q_tau1 - q_ref)))
    ok_c = max_abs_diff == 0.0
    print(f"  (c) tau=1 vs tau-free soft target: max abs diff = {max_abs_diff} (==0.0: {'PASS' if ok_c else 'FAIL'})")

    all_pass = ok_a and ok_b and ok_c
    print("all checks passed" if all_pass else "FAILURES PRESENT")
    return all_pass


# ---------------------------------------------------------------------------
# Real read: every K4 nested table on disk, the 5 pre-registered bars, the knob sweep.
# ---------------------------------------------------------------------------


def _bar_i(structured_cases: list[dict]) -> dict:
    """(i) blend NLL <= hard NLL for every arm, on >= 8/9 structured cases."""
    out = {}
    for arm in ARM_NAMES:
        n_pass = sum(1 for c in structured_cases if c["arms"][arm]["nll_blend"] <= c["arms"][arm]["nll_hard"])
        out[arm] = {"n_pass": n_pass, "n_cases": len(structured_cases), "pass": n_pass >= 8}
    return out


def _bar_ii(structured_cases: list[dict]) -> dict:
    """(ii) some soft arm's blend NLL <= hard_knee's blend NLL, on >= 7/9 structured cases."""
    n_pass = sum(1 for c in structured_cases if any(c["arms"][a]["nll_blend"] <= c["arms"]["hard_knee"]["nll_blend"] for a in SOFT_ARMS))
    return {"n_pass": n_pass, "n_cases": len(structured_cases), "pass": n_pass >= 7}


def _bar_iii(structured_cases: list[dict]) -> dict:
    """(iii) oracle-x raw score <= oracle-noisy raw score, on 9/9 structured cases.

    Raw (un-negated) mean log-score, higher-is-better: the honest bound cannot legitimately
    exceed the cheating per-point-max bound. In NLL terms (lower-is-better) this is the
    opposite-looking `oracle_x_nll >= oracle_noisy_nll` — the honest bound's NLL is no BETTER
    than the cheating bound's, exactly as expected from an information-constrained estimator.
    """
    n_pass = sum(1 for c in structured_cases if c["oracle_x_score"] <= c["oracle_noisy_score"])
    return {"n_pass": n_pass, "n_cases": len(structured_cases), "pass": n_pass == len(structured_cases)}


def _bar_iv(broad_cases: list[dict]) -> dict:
    """(iv) every arm's blend advantage over global (`nll_global - nll_blend`) <= 0.02 nat, all 6 broad cases."""
    out = {}
    for arm in ARM_NAMES:
        advantages = [c["nll_global"] - c["arms"][arm]["nll_blend"] for c in broad_cases]
        out[arm] = {"max_advantage": max(advantages) if advantages else None, "advantages": advantages, "pass": all(a <= 0.02 for a in advantages)}
    return out


def _run_knob_sweep(all_cases: list[dict], winner: str, device: str) -> tuple[list[dict], str]:
    """Sweeps prior-bins/width/tau on the winning arm's own (n_bins, width) recipe (see module docstring).

    Falls back to the best-performing soft arm if `winner` is a hard-labeled arm (4/5), for
    which prior-bins/temperature have no meaning; logs the substitution.

    Returns:
        `(knob_results, recipe_arm)` — the per-combo results and the arm whose recipe was used.
    """
    case_by_key = {(c["toy"], c["seed"]): c for c in all_cases}
    if winner in _ARM_RECIPE:
        recipe_arm = winner
    else:
        mean_blend = {a: float(np.mean([c["arms"][a]["nll_blend"] for c in all_cases if c["toy"] in STRUCTURED_TOYS])) for a in SOFT_ARMS}
        recipe_arm = min(mean_blend, key=mean_blend.get)
        print(f"[s1] winner '{winner}' has no soft-target recipe (hard-label arm); knob sweep falls back to best soft arm '{recipe_arm}'")

    own_n_bins = _ARM_RECIPE[recipe_arm]["n_bins"]
    own_width = _ARM_RECIPE[recipe_arm]["width"]
    bins_grid = KNOB_BINS if own_n_bins is not None else (None,)
    width_grid = KNOB_WIDTH if own_width is not None else (None,)

    results: list[dict] = []
    for toy in KNOB_TOYS:
        for seed in SEEDS:
            case = case_by_key.get((toy, seed))
            if case is None:
                continue
            tbl = ck6.read_table(toy, seed)
            score, x, c_grid = tbl["score"], tbl["x"], tbl["c_grid"]
            n = score.shape[0]
            train_idx = np.arange(0, n, 2)
            eval_idx = np.arange(1, n, 2)
            score_tr, x_tr = score[train_idx], x[train_idx]
            score_ev, x_ev = score[eval_idx], x[eval_idx]

            hard_knee_blend = case["arms"]["hard_knee"]["nll_blend"]
            base_ordering = case["arms"][recipe_arm]["nll_blend"] <= hard_knee_blend

            for n_bins in bins_grid:
                for width in width_grid:
                    for tau in KNOB_TAU:
                        q = _soft_targets_knobs(score_tr, x_tr, n_bins=n_bins, width=width, tau=tau)
                        model = ck6._train_router(x_tr, n_cols=len(c_grid), device=device, soft_targets=q, seed=seed)
                        nll_blend, _nll_hard = _eval_arm(model, x_ev, score_ev, device)
                        combo_ordering = nll_blend <= hard_knee_blend
                        results.append(
                            {
                                "toy": toy,
                                "seed": seed,
                                "n_bins": n_bins,
                                "width": width,
                                "tau": tau,
                                "nll_blend": nll_blend,
                                "hard_knee_blend": hard_knee_blend,
                                "ordering_matches_base": bool(combo_ordering == base_ordering),
                            }
                        )
    return results, recipe_arm


def main() -> None:
    """Reads every available K4 table, runs the S1 factorial + bars + knob sweep, writes the summary json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="Run the S1 synthetic known-answer selftest and exit.")
    args = parser.parse_args()

    if args.selftest:
        ok = run_selftest()
        sys.exit(0 if ok else 1)

    os.makedirs(OUT_DIR, exist_ok=True)
    device = str(get_device())
    print(f"[s1] device={device}")

    all_cases: list[dict] = []
    for toy in (*STRUCTURED_TOYS, *BROAD_TOYS):
        for seed in SEEDS:
            case = run_case(toy, seed, device)
            if case is None:
                print(f"[s1] {toy} s{seed}: no K4 table yet, skipping")
                continue
            all_cases.append(case)
            arm_str = ", ".join(f"{a}={case['arms'][a]['nll_blend']:.4f}" for a in ARM_NAMES)
            print(f"[s1] {toy} s{seed}: blend NLL {arm_str}  global={case['nll_global']:.4f}  oracle_x={case['oracle_x_nll']:.4f}  oracle_noisy={case['oracle_noisy_nll']:.4f}")

    structured_cases = [c for c in all_cases if c["toy"] in STRUCTURED_TOYS]
    broad_cases = [c for c in all_cases if c["toy"] in BROAD_TOYS]

    bars = {
        "i_blend_le_hard": _bar_i(structured_cases),
        "ii_soft_beats_hard_knee": _bar_ii(structured_cases),
        "iii_oracle_x_le_oracle_noisy": _bar_iii(structured_cases),
        "iv_broad_advantage_bounded": _bar_iv(broad_cases),
    }
    for name, res in bars.items():
        print(f"[s1] bar {name}: {res}")

    factorial = {
        "1v2_prior_effect": _factorial_read(structured_cases, "soft", "soft_no_prior"),
        "2v5_softness_effect": _factorial_read(structured_cases, "soft_no_prior", "raw_argmax"),
        "3v2_smoothing_effect": _factorial_read(structured_cases, "soft_smoothed", "soft_no_prior"),
        "1v3_prior_vs_smoothing": _factorial_read(structured_cases, "soft", "soft_smoothed"),
    }
    print(f"[s1] factorial reads: {factorial}")

    mean_blend = {arm: float(np.mean([c["arms"][arm]["nll_blend"] for c in structured_cases])) for arm in ARM_NAMES}
    winner = min(mean_blend, key=mean_blend.get)
    print(f"[s1] winner (lowest mean blend NLL on structured cases) = {winner}  ({mean_blend})")

    knob_results, recipe_arm = _run_knob_sweep(all_cases, winner, device)
    n_matching = sum(1 for r in knob_results if r["ordering_matches_base"])
    bar_v = {"n_combos": len(knob_results), "n_matching": n_matching, "pass": bool(knob_results) and n_matching == len(knob_results), "recipe_arm": recipe_arm}
    bars["v_knob_robustness"] = bar_v
    print(f"[s1] bar v_knob_robustness: {bar_v}")

    summary = {
        "config": {
            "hidden": list(ck6.HIDDEN),
            "n_epochs": ck6.N_EPOCHS,
            "lr": ck6.LR,
            "width": ck6.WIDTH,
            "seeds": list(SEEDS),
            "arm_names": list(ARM_NAMES),
            "knob_bins": list(KNOB_BINS),
            "knob_width": list(KNOB_WIDTH),
            "knob_tau": list(KNOB_TAU),
        },
        "structured_cases": structured_cases,
        "broad_cases": broad_cases,
        "bars": bars,
        "factorial": factorial,
        "winner": winner,
        "mean_blend_by_arm": mean_blend,
        "knob_results": knob_results,
    }
    out_path = os.path.join(OUT_DIR, "s1_summary.json")
    with open(out_path, "w") as f:
        json.dump(ck6._jsonable(summary), f, indent=2)
    print(f"[s1] wrote {out_path}")


if __name__ == "__main__":
    main()
