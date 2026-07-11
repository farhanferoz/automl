"""T2 — multi-dimensional count toys: the per-input selector port de-risk.

(docs/plans/perinput_selector_2026-07-10/EXECUTION_PLAN.md WS-B, task T2; pre-registration in
capacity_ladder_results/T2/PREREGISTRATION.md)

Everything in the per-input selector program so far is 1-D. The real-model ports are gated
precisely on "neighbourhood reads may break in many dims" — T2 measures exactly that
degradation on toys with analytic ground truth: a multi-dimensional generalization of Toy D
(`_toy_datasets.make_toy_d_ndim`, the count-beyond-binary staircase, extended to `dim` inputs
with `dim - 1` nuisance coordinates and an optional fixed-rotation variant) read with kNN
neighbourhoods in place of the 1-D box-car.

**REGRESSION GUARD (load-bearing).** `_capacity_ladder_nested.NestedKSurrogate` is NOT modified
by this script — it already accepts an arbitrary `input_dim` (K4 already calls it with
`input_dim=x_t.shape[1]`, generic). The only NEW behaviour T2 adds is choosing a WIDER trunk
(hidden=64, still 2 hidden layers via `_variational_em._MLP`'s own `layers=2` default — the
EXECUTION_PLAN.md T2 spec's "(64,64)") for `dim > 1`, via `_fit_ladder`'s dim-dependent choice of
the EXISTING `hidden` constructor arg (`_capacity_ladder_nested.py:56`) — never a code-path
change. Because the underlying function is untouched, `dim=1` behaviour is bit-identical to the
current code STRUCTURALLY, not merely by testing; `run_selftest`'s regression guard (a) still
asserts this explicitly (byte-identical score tables from two independent calls: one through
`_fit_ladder(dim=1, ...)`, one calling `_capacity_ladder_nested.train_nested_k_surrogate`
directly with the literal current 1-D convention `hidden=32`) so any FUTURE change to
`_fit_ladder`'s dim->hidden mapping that accidentally touches the dim=1 case is caught.

**Router/reader generalization (local, additive — K6's and `_capacity_ladder.py`'s files are
NOT touched).** `capacity_ladder_k6._RouterMLP`/`_train_router` hardcode a scalar
(`x.reshape(-1, 1)`) input layer, and `_capacity_ladder.perinput_curve` is a 1-D box-car keyed
on a single `x` coordinate — neither generalizes to `dim > 1` inputs as-is, and K6 is another
task's file (T2 must not edit it). This script therefore carries its OWN local `_RouterMLP`
(input_dim generalized, otherwise IDENTICAL architecture/hyperparameters: hidden (32, 32), Adam
lr 1e-2, 300 epochs, soft-label cross-entropy) and its OWN local `knn_curve` (kNN neighbourhoods,
Euclidean, `n_nbr=50` primary / `n_nbr=25` the G5 half-width analog — mirrors
`perinput_curve`'s `delta`/`delta_half` dict interface so the S1 "arm 3" target construction and
S1's honest `oracle-x` bound both port over with a one-line neighbourhood-source swap).

**CRITICAL: which S1 recipe transfers, and which does not (read this before the bars).** S1
certified `soft` (per-tercile PRIOR-stacked responsibilities) as the winning target construction,
and separately certified that its own neighbour-SMOOTHING mechanism (S1 arm 3, "soft_smoothed")
contributes essentially NOTHING on top of the prior (+0.0001 nat, ~60x below the prior's own
effect — `capacity_ladder_results/RESULTS.md` "## S1" section). The per-tercile prior needs a
BINNABLE SCALAR to define bins on; in `dim > 1` there is no such scalar the selector may see (the
selector is `x` alone, per L3/L5 — using the analytic staircase index `s` to build a prior would
leak the very ground truth T2 is trying to test recovery of). So T2's targets can ONLY use the
mechanism S1 found near-zero in 1-D: `_knn_soft_targets` is the kNN generalization of S1's arm 3
("soft_smoothed"), NOT of the winning arm 1 ("soft"). "Selector = S1/S2 winner recipe, input
dim = dim" (EXECUTION_PLAN.md T2 spec) is read here as the TRAINING SCHEME transferring (soft-
label cross-entropy, router architecture/hyperparameters, blend-primary evaluation, honest
oracle-x bound) — NOT the target-construction mechanism, which structurally cannot transfer. A
weak or failing bar (i) is therefore a LEGITIMATE PRE-REGISTERED OUTCOME (report it, escalate per
EXECUTION_PLAN.md §0b), not a build defect — see `capacity_ladder_results/T2/PREREGISTRATION.md`
for the full prediction this is written against.

Matrix (5 configs x 3 seeds = 15 units, orchestrator-run — NOT executed by this authoring pass):
  dim=2 axis, dim=2 broad (variance-matched k*=1-everywhere twin), dim=5 axis, dim=5 rotated
  (fixed unit-vector projection), dim=10 axis. `N_TRAIN = N_TEST = 2500` (K4-scale).

Run `--selftest` before any real read (regression guard + a 2-D synthetic known-answer staircase
recovery, no disk I/O). Run `--measure-one` to time ONE dim=5 axis unit and extrapolate the full
matrix's wall-time (EXECUTION_PLAN.md §0b measure-one-unit-before-matrix; this script does not
launch the 15-unit matrix itself — run ownership is the orchestrator's, per §0b "Run ownership").
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root, so `import automl_package` works

import _capacity_ladder as cl  # noqa: E402
import _capacity_ladder_nested as ckn  # noqa: E402
import _toy_datasets as td  # noqa: E402

from automl_package.utils.pytorch_utils import get_device  # noqa: E402

RESULTS_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "T2")

SEEDS = (0, 1, 2)
N_TRAIN = 2500
N_TEST = 2500
K_MAX = 8  # matches K4's toy-D k_max (KMAX["D"]=8) — "same component geometry as toy D"
C_GRID = list(range(1, K_MAX + 1))
N_EPOCHS_LADDER = 800  # K4's convention for the nested surrogate
LADDER_LR = 1e-2

TRUNK_HIDDEN_1D = 32  # NestedKSurrogate's own current default (unchanged)
TRUNK_HIDDEN_NDIM = 64  # EXECUTION_PLAN.md T2 spec: "trunk widens to (64,64) for dim>1"

ROUTER_HIDDEN = (32, 32)  # K6/S1's router config, unchanged — only input_dim generalizes
ROUTER_N_EPOCHS = 300
ROUTER_LR = 1e-2

N_NBR = 50  # primary kNN neighbourhood size (replaces the 1-D box-car width)
N_NBR_HALF = N_NBR // 2  # the G5 locality-guard analog: "re-read at n_nbr=25"

BOOT_N = 1000

CONFIGS: list[dict] = [
    {"name": "dim2_axis", "dim": 2, "rotated": False, "broad": False},
    {"name": "dim2_broad", "dim": 2, "rotated": False, "broad": True},
    {"name": "dim5_axis", "dim": 5, "rotated": False, "broad": False},
    {"name": "dim5_rotated", "dim": 5, "rotated": True, "broad": False},
    {"name": "dim10_axis", "dim": 10, "rotated": False, "broad": False},
]


# ---------------------------------------------------------------------------
# Ladder training — thin dim-dependent wrapper around the UNCHANGED
# `_capacity_ladder_nested.train_nested_k_surrogate` (see module docstring's REGRESSION GUARD).
# ---------------------------------------------------------------------------


def _trunk_hidden(dim: int) -> int:
    """Trunk hidden width: unchanged 32 at `dim=1`, widened to 64 (still 2 layers) for `dim>1`."""
    return TRUNK_HIDDEN_1D if dim == 1 else TRUNK_HIDDEN_NDIM


def _fit_ladder(
    x: np.ndarray, y: np.ndarray, *, dim: int, k_max: int, n_epochs: int, device: str, seed: int, lr: float = LADDER_LR
) -> ckn.NestedKSurrogate:
    """Fits the nested-k surrogate via the UNCHANGED `train_nested_k_surrogate`, choosing the trunk width by `dim`."""
    return ckn.train_nested_k_surrogate(
        x, y, k_max=k_max, n_epochs=n_epochs, lr=lr, hidden=_trunk_hidden(dim), adaptive_bin_means=True, device=device, seed=seed
    )


# ---------------------------------------------------------------------------
# kNN neighbourhood reader — the multi-D analog of `_capacity_ladder.perinput_curve`'s box-car
# (local to this script; `_capacity_ladder.py` is shared by many other certified tasks and is
# not touched).
# ---------------------------------------------------------------------------


def _pairwise_sqdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """`(M, N)` squared Euclidean distance between rows of `a` (M, D) and `b` (N, D)."""
    a2 = (a**2).sum(axis=1)[:, None]
    b2 = (b**2).sum(axis=1)[None, :]
    return a2 + b2 - 2.0 * a @ b.T


def _knn_delta(score_src: np.ndarray, x_src: np.ndarray, query_x: np.ndarray, n_nbr: int, ref_c: int) -> tuple[np.ndarray, np.ndarray]:
    """kNN-neighbour-averaged advantage `Delta_c(x)` at each row of `query_x`, sourced from `(score_src, x_src)`.

    Averages `score_src[:, c] - score_src[:, ref_c]` over each query point's `n_nbr` nearest
    neighbours (Euclidean, ties broken arbitrarily by `np.argpartition`) among `x_src`'s rows —
    the multi-D generalization of `perinput_curve`'s box-car average.

    Returns:
        `(delta (Q, C), n_used (Q,))` — `n_used` is `min(n_nbr, x_src.shape[0])`, constant here
        (unlike the box-car's data-dependent neighbour count) since kNN always returns exactly k.
    """
    delta_full = score_src - score_src[:, [ref_c]]
    n_src = x_src.shape[0]
    k = int(min(n_nbr, n_src))
    d2 = _pairwise_sqdist(query_x, x_src)  # (Q, N)
    nn_idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]  # (Q, k), unsorted within k
    delta = delta_full[nn_idx].mean(axis=1)
    return delta, np.full(query_x.shape[0], k, dtype=np.int64)


def knn_curve(score: np.ndarray, x: np.ndarray, n_nbr: int, ref_c: int = 0, query_x: np.ndarray | None = None) -> dict:
    """kNN-neighbour-averaged advantage curve — the multi-D analog of `cl.perinput_curve`.

    Same dict shape as `perinput_curve` (`query_x`, `delta`, `delta_half`, plus the neighbourhood
    size in place of a box-car width) so downstream target-construction / oracle-bound code ports
    over from S1 with only the neighbourhood-source function swapped. `delta_half` is the G5
    locality-guard analog at `n_nbr // 2` — EXECUTION_PLAN.md T2: "G5 analog: re-read at n_nbr=25".

    Args:
        score: `(N, C)` held-out log-likelihood table.
        x: `(N, D)` or `(N,)` input coordinates paired with each row of `score`.
        n_nbr: primary neighbourhood size.
        ref_c: positional reference column (`Delta = 0` there).
        query_x: points at which to evaluate the curve; defaults to `x` itself.

    Returns:
        dict with `query_x` `(Q, D)`, `delta` `(Q, C)` at `n_nbr`, `delta_half` `(Q, C)` at
        `n_nbr // 2`, `n_nbr`, `n_nbr_half`, `n_neighbors` `(Q,)`.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(-1, 1)
    qx = x_arr if query_x is None else np.asarray(query_x, dtype=np.float64)
    if qx.ndim == 1:
        qx = qx.reshape(-1, 1)
    s = np.asarray(score, dtype=np.float64)

    n_nbr_half = max(1, n_nbr // 2)
    delta, n_used = _knn_delta(s, x_arr, qx, n_nbr, ref_c)
    delta_half, _ = _knn_delta(s, x_arr, qx, n_nbr_half, ref_c)
    return {"query_x": qx, "delta": delta, "delta_half": delta_half, "n_nbr": n_nbr, "n_nbr_half": n_nbr_half, "n_neighbors": n_used}


# ---------------------------------------------------------------------------
# Target construction — T2's targets are the kNN generalization of S1's arm 3
# ("soft_smoothed": neighbour-averaged score, softmaxed, NO prior term — see module docstring's
# CRITICAL note on why the prior mechanism does not transfer to dim > 1).
# ---------------------------------------------------------------------------


def _softmax_rows(mat: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax (max-subtracted); duplicated locally (S1's own copy, `capacity_ladder_s1._softmax_rows`, is another task's file)."""
    z = mat - mat.max(axis=1, keepdims=True)
    w = np.exp(z)
    return w / w.sum(axis=1, keepdims=True)


def _knn_soft_targets(score_tr: np.ndarray, x_tr: np.ndarray, n_nbr: int, ref_c: int = 0) -> np.ndarray:
    """Per-row soft target `softmax_c(kNN-averaged Delta_c(x_i))` — the S1-arm-3 kNN generalization."""
    out = knn_curve(score_tr, x_tr, n_nbr, ref_c=ref_c, query_x=x_tr)
    return _softmax_rows(np.asarray(out["delta"]))


# ---------------------------------------------------------------------------
# Router — local, input_dim-generalized port of `capacity_ladder_k6._RouterMLP`/`_train_router`
# (K6's file is another task's file and is not touched; same architecture/training scheme).
# ---------------------------------------------------------------------------


class _RouterMLP(nn.Module):
    """`x (R^input_dim) -> logits over c_grid columns`, hidden (32, 32) + ReLU."""

    def __init__(self, input_dim: int, n_cols: int, hidden: tuple[int, ...] = ROUTER_HIDDEN) -> None:
        super().__init__()
        dims = [input_dim, *hidden]
        layers: list[nn.Module] = []
        for d_in, d_out in itertools.pairwise(dims):
            layers += [nn.Linear(d_in, d_out), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], n_cols))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`(N, input_dim) -> (N, n_cols)` logits."""
        return self.net(x)


def _as_2d(x: np.ndarray, dtype: type = np.float32) -> np.ndarray:
    """`(N,)` -> `(N, 1)`; `(N, D)` passed through — the multi-D-aware analog of K6's `.reshape(-1, 1)`."""
    x_arr = np.asarray(x, dtype=dtype)
    return x_arr.reshape(-1, 1) if x_arr.ndim == 1 else x_arr


def _train_router(x: np.ndarray, *, n_cols: int, device: str, soft_targets: np.ndarray, n_epochs: int = ROUTER_N_EPOCHS, lr: float = ROUTER_LR, seed: int = 0) -> _RouterMLP:
    """Trains one router by soft-label cross-entropy (K6/S1's SOFT scheme, input_dim generalized).

    `loss = -mean_i sum_c q_i[c] log softmax(logits)_c`, i.e. `KL(q_i || router(x_i))` up to the
    (parameter-independent) target entropy — identical training scheme to
    `capacity_ladder_k6._train_router`'s soft-target branch, only the input layer width changes.
    """
    torch.manual_seed(seed)
    x_arr = _as_2d(x)
    model = _RouterMLP(x_arr.shape[1], n_cols).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x_t = torch.as_tensor(x_arr, device=device)
    q_t = torch.as_tensor(np.asarray(soft_targets, dtype=np.float32), device=device)

    model.train()
    for _ in range(n_epochs):
        opt.zero_grad()
        logits = model(x_t)
        loss = -(q_t * nnf.log_softmax(logits, dim=1)).sum(dim=1).mean()
        loss.backward()
        opt.step()
    model.eval()
    return model


def _route(model: _RouterMLP, x: np.ndarray, device: str) -> np.ndarray:
    """Argmax-routed column index per row of `x`."""
    with torch.no_grad():
        x_t = torch.as_tensor(_as_2d(x), device=device)
        logits = model(x_t)
    return logits.argmax(dim=1).cpu().numpy()


# ---------------------------------------------------------------------------
# Evaluation — blend (primary, per L3) + hard reads; honest oracle-x bound (S1's protocol,
# neighbourhood source swapped to kNN).
# ---------------------------------------------------------------------------


def _blend_scores(model: _RouterMLP, x: np.ndarray, score: np.ndarray, device: str) -> np.ndarray:
    """Per-example blended log-score `logsumexp_c(log softmax(logits(x_i))_c + score[i,c])`."""
    with torch.no_grad():
        x_t = torch.as_tensor(_as_2d(x), device=device)
        log_w = nnf.log_softmax(model(x_t), dim=1).double()
    score_t = torch.as_tensor(np.asarray(score, dtype=np.float64), device=device)
    return torch.logsumexp(log_w + score_t, dim=1).cpu().numpy()


def _blend_nll(model: _RouterMLP, x: np.ndarray, score: np.ndarray, device: str) -> float:
    """Blended NLL (L3 primary read): `-mean` of `_blend_scores`."""
    return float(-_blend_scores(model, x, score, device).mean())


def _routed_nll(score: np.ndarray, col_idx: np.ndarray) -> float:
    """`mean_i(-score[i, col_idx[i]])` — the routed held-out mixture NLL."""
    rows = np.arange(score.shape[0])
    return float(-score[rows, col_idx].mean())


def _eval_router(model: _RouterMLP, x_ev: np.ndarray, score_ev: np.ndarray, device: str) -> tuple[float, float, np.ndarray]:
    """Blend NLL + hard-routed NLL (and the hard route itself) of one trained router on the eval half."""
    nll_blend = _blend_nll(model, x_ev, score_ev, device)
    col_hard = _route(model, x_ev, device)
    nll_hard = _routed_nll(score_ev, col_hard)
    return nll_blend, nll_hard, col_hard


def _knn_oracle_reads(score_ev_a: np.ndarray, x_ev_a: np.ndarray, score_ev_b: np.ndarray, x_ev_b: np.ndarray, n_nbr: int) -> tuple[float, np.ndarray]:
    """Honest `oracle-x` bound (S1's protocol): routes eval-B via eval-A's kNN-averaged advantage curve alone.

    No eval-B point ever informs its own route. Returns the RAW (un-negated) mean actual score on
    eval-B — S1's convention, so `oracle_x_score <= oracle_noisy_score` is the expected direction.
    """
    out = knn_curve(score_ev_a, x_ev_a, n_nbr, ref_c=0, query_x=x_ev_b)
    delta = np.asarray(out["delta"])
    route_col = delta.argmax(axis=1)
    rows = np.arange(score_ev_b.shape[0])
    oracle_x_score = float(score_ev_b[rows, route_col].mean())
    return oracle_x_score, route_col


def _paired_point_bootstrap_se(diff: np.ndarray, n_boot: int, seed: int) -> float:
    """Plain i.i.d. bootstrap SE of the mean of a per-example paired difference (G1 convention)."""
    rng = np.random.default_rng(seed)
    n = diff.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[idx].mean(axis=1)
    return float(boot_means.std(ddof=1))


# ---------------------------------------------------------------------------
# Gold read — per-region capture against the analytic staircase k*(s) (read only; s is NEVER
# selector-visible, only x is).
# ---------------------------------------------------------------------------


def _gold_region_capture(routed_c: np.ndarray, s_eval: np.ndarray) -> dict:
    """Selector's routed capacity vs the analytic gold `k*(s)` (exact-match rate + modal-by-tercile).

    `s_eval` is already the resolved scalar staircase coordinate (`td.toy_d_ndim_s`'s output);
    `toy_d_ndim_k_star(x, rotated)` recomputes `s` from `x` internally, so rather than fabricate a
    fake `(N, 1)` `x` this calls the underlying tercile rule (`td._staircase_k`, the same function
    `toy_d_ndim_k_star` itself calls) directly on the coordinate already in hand.
    """
    k_star = td._staircase_k(s_eval)
    capture_rate = float(np.mean(routed_c == k_star))
    tercile_bins = cl.quantile_bins(s_eval, 3)
    modal_by_tercile = []
    for b in sorted(np.unique(tercile_bins)):
        vals, counts = np.unique(routed_c[tercile_bins == b], return_counts=True)
        modal_by_tercile.append(int(vals[np.argmax(counts)]))
    return {"capture_rate": capture_rate, "modal_by_tercile": modal_by_tercile, "designed_modal": [1, 2, 3]}


# ---------------------------------------------------------------------------
# One (config, seed) unit.
# ---------------------------------------------------------------------------


def run_case(config: dict, seed: int, device: str, n_train: int = N_TRAIN, n_test: int = N_TEST, n_epochs_ladder: int = N_EPOCHS_LADDER) -> dict:
    """Trains the ladder + router for one (config, seed) unit; runs the full S1-style protocol via kNN."""
    dim, rotated, broad = config["dim"], config["rotated"], config["broad"]

    if broad:
        x_tr, y_tr = td.make_toy_d_ndim_broad(n=n_train, dim=dim, seed=seed)
        x_te, y_te = td.make_toy_d_ndim_broad(n=n_test, dim=dim, seed=seed + 500)
    else:
        x_tr, y_tr = td.make_toy_d_ndim(n=n_train, dim=dim, rotated=rotated, seed=seed)
        x_te, y_te = td.make_toy_d_ndim(n=n_test, dim=dim, rotated=rotated, seed=seed + 500)

    nested = _fit_ladder(x_tr, y_tr, dim=dim, k_max=K_MAX, n_epochs=n_epochs_ladder, device=device, seed=seed)
    score = cl.score_table(nested, x_te, y_te, C_GRID)

    n = score.shape[0]
    train_idx = np.arange(0, n, 2)
    eval_idx = np.arange(1, n, 2)
    score_tr, x_tr_r = score[train_idx], x_te[train_idx]
    score_ev, x_ev_r = score[eval_idx], x_te[eval_idx]
    score_ev_a, x_ev_a = score_ev[0::2], x_ev_r[0::2]
    score_ev_b, x_ev_b = score_ev[1::2], x_ev_r[1::2]

    reads: dict[str, dict] = {}
    for n_nbr, label in ((N_NBR, "n_nbr50"), (N_NBR_HALF, "n_nbr25")):
        targets = _knn_soft_targets(score_tr, x_tr_r, n_nbr)
        model = _train_router(x_tr_r, n_cols=K_MAX, device=device, soft_targets=targets, seed=seed)
        nll_blend, nll_hard, col_hard = _eval_router(model, x_ev_r, score_ev, device)
        reads[label] = {"n_nbr": n_nbr, "nll_blend": nll_blend, "nll_hard": nll_hard, "col_hard": col_hard, "model": model}

    primary = reads["n_nbr50"]

    r_star, _delta_curve, _se_curve = cl.knee(score_tr, ref_c=1, n_boot=BOOT_N, c_grid=C_GRID, seed=seed)
    k_global = 1 if r_star == 0 else r_star
    global_col = C_GRID.index(k_global)
    nll_global = _routed_nll(score_ev, np.full(score_ev.shape[0], global_col, dtype=np.int64))

    oracle_x_score, _route_col = _knn_oracle_reads(score_ev_a, x_ev_a, score_ev_b, x_ev_b, N_NBR)
    oracle_noisy_score = float(score_ev.max(axis=1).mean())

    blend_scores = _blend_scores(primary["model"], x_ev_r, score_ev, device)
    diff_vec = blend_scores - score_ev[:, global_col]
    advantage = float(diff_vec.mean())
    advantage_se = _paired_point_bootstrap_se(diff_vec, BOOT_N, seed)

    gold = None
    if not broad:
        s_ev = td.toy_d_ndim_s(x_ev_r, rotated)
        routed_c = np.asarray(C_GRID)[primary["col_hard"]]
        gold = _gold_region_capture(routed_c, s_ev)

    return {
        "config": config["name"],
        "dim": dim,
        "rotated": rotated,
        "broad": broad,
        "seed": seed,
        "n_train_router": len(train_idx),
        "n_eval_router": len(eval_idx),
        "nll_blend_n_nbr50": primary["nll_blend"],
        "nll_hard_n_nbr50": primary["nll_hard"],
        "nll_blend_n_nbr25": reads["n_nbr25"]["nll_blend"],
        "nll_hard_n_nbr25": reads["n_nbr25"]["nll_hard"],
        "g5_blend_shift": primary["nll_blend"] - reads["n_nbr25"]["nll_blend"],
        "nll_global": nll_global,
        "k_global": int(k_global),
        "advantage": advantage,
        "advantage_se": advantage_se,
        "oracle_x_score": oracle_x_score,
        "oracle_noisy_score": oracle_noisy_score,
        "oracle_x_le_oracle_noisy": bool(oracle_x_score <= oracle_noisy_score),
        "gold": gold,
        "_score": score,
        "_x": np.asarray(x_te, dtype=np.float64),
        "_col_hard_n_nbr50": primary["col_hard"],
    }


def _jsonable(obj: object) -> object:
    """Recursively drops private/model keys and casts numpy scalars (K4/K5/K6/S1 convention)."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items() if not (isinstance(k, str) and (k.startswith("_") or k == "model"))}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---------------------------------------------------------------------------
# Pre-registered bars.
# ---------------------------------------------------------------------------


def _bar_i(dim2_axis_cases: list[dict]) -> dict:
    """(i) dim=2 axis: selector blend beats global > 2*SE on >= 2/3 seeds."""
    per_seed = [{"seed": c["seed"], "advantage": c["advantage"], "se": c["advantage_se"], "beats_2se": bool(c["advantage"] > 2.0 * c["advantage_se"])} for c in dim2_axis_cases]
    n_pass = sum(1 for r in per_seed if r["beats_2se"])
    return {"per_seed": per_seed, "n_pass": n_pass, "n_seeds": len(per_seed), "pass": n_pass >= 2}


def _bar_ii(axis_cases_by_dim: dict[int, list[dict]]) -> dict:
    """(ii) DELIVERABLE: the degradation curve (advantage + gold region-capture vs dim). No pass/fail gate."""
    curve = {}
    for dim, cases in sorted(axis_cases_by_dim.items()):
        advs = [c["advantage"] for c in cases]
        ses = [c["advantage_se"] for c in cases]
        captures = [c["gold"]["capture_rate"] for c in cases if c["gold"] is not None]
        curve[dim] = {
            "mean_advantage": float(np.mean(advs)),
            "per_seed_advantage": advs,
            "per_seed_se": ses,
            "n_seeds_above_2se": int(sum(a > 2.0 * s for a, s in zip(advs, ses, strict=True))),
            "mean_gold_capture_rate": float(np.mean(captures)) if captures else None,
            "per_seed_gold_capture_rate": captures,
        }
    dims_sorted = sorted(curve)
    crossing_dim = next((d for d in dims_sorted if curve[d]["n_seeds_above_2se"] < 2), None)
    return {"curve": curve, "dims": dims_sorted, "crossing_dim_below_2se_majority": crossing_dim}


def _bar_iii(axis_dim5: list[dict], rotated_dim5: list[dict]) -> dict:
    """(iii) rotated-vs-axis at dim=5: paired diff reported (prediction: |diff| within 2*SE). Not gated."""
    axis_by_seed = {c["seed"]: c for c in axis_dim5}
    rot_by_seed = {c["seed"]: c for c in rotated_dim5}
    seeds = sorted(set(axis_by_seed) & set(rot_by_seed))
    diffs = np.array([rot_by_seed[s]["advantage"] - axis_by_seed[s]["advantage"] for s in seeds])
    se = _paired_point_bootstrap_se(diffs, BOOT_N, seed=0) if len(diffs) > 1 else float("nan")
    mean_diff = float(diffs.mean()) if len(diffs) else float("nan")
    within_2se = bool(abs(mean_diff) <= 2.0 * se) if se == se else None  # NaN-safe (se==se is False for NaN)
    return {"seeds": seeds, "per_seed_diff": diffs.tolist(), "mean_diff": mean_diff, "se": se, "within_2se": within_2se}


def _bar_iv(dim2_broad_cases: list[dict]) -> dict:
    """(iv) dim=2 broad twin: selector advantage over global <= 0.02 nat, every seed."""
    advantages = [c["advantage"] for c in dim2_broad_cases]
    return {"advantages": advantages, "max_advantage": max(advantages) if advantages else None, "pass": all(a <= 0.02 for a in advantages)}


# ---------------------------------------------------------------------------
# Selftest — regression guard (a) + 2-D synthetic known-answer staircase recovery (b). No disk I/O.
# ---------------------------------------------------------------------------


def _regression_guard(device: str) -> bool:
    """(a) `dim=1` must reproduce `_capacity_ladder_nested.train_nested_k_surrogate`'s CURRENT behaviour, bit-identically.

    Two independent calls, same hyperparameters/seed: one through `_fit_ladder(dim=1, ...)`, one
    calling `ckn.train_nested_k_surrogate` directly with the literal current 1-D convention
    (`hidden=32`, matching K4's implicit default). Both funnel into the SAME unmodified function,
    so this asserts `_fit_ladder`'s dim->hidden mapping — the only thing T2 adds for `dim=1` —
    never diverges from that convention, not merely that the two happen to agree today.
    """
    x, y = td.make_toy_d(n=300, seed=0)
    k_max = 4
    kwargs = {"k_max": k_max, "n_epochs": 40, "lr": 1e-2, "device": device, "seed": 0}

    model_t2 = _fit_ladder(x, y, dim=1, **kwargs)
    model_ref = ckn.train_nested_k_surrogate(x, y, hidden=TRUNK_HIDDEN_1D, adaptive_bin_means=True, **kwargs)

    c_grid = list(range(1, k_max + 1))
    score_t2 = cl.score_table(model_t2, x, y, c_grid)
    score_ref = cl.score_table(model_ref, x, y, c_grid)
    ok = bool(np.array_equal(score_t2, score_ref))
    print(f"[t2 selftest] (a) regression guard: dim=1 score table bit-identical to hidden=32 reference: {ok} (max abs diff={np.max(np.abs(score_t2 - score_ref)):.2e})")
    return ok


def _synthetic_2d_table(n_per: int = 500, noise_sd: float = 0.8, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A synthetic `(N, 6)` 2-D table whose best column is 1/3/6 by tercile of `x[:, 0]` (K6 `_selftest_table`'s design, ported to 2-D input with one nuisance coordinate)."""
    rng = np.random.default_rng(seed)
    n = n_per * 3
    x = rng.uniform(0.0, 1.0, size=(n, 2))
    s = x[:, 0]
    peak = np.select([s < 1.0 / 3.0, s < 2.0 / 3.0], [1.0, 3.0], default=6.0)
    c = np.arange(1, 7, dtype=np.float64)
    base = -0.5 * (c[None, :] - peak[:, None]) ** 2
    score = base + rng.normal(0.0, noise_sd, size=(n, 6))
    return score, x


def _fold_equivalence_check() -> bool:
    """(c) `--fold`'s aggregation, exercised through `_save_case`/`_load_case`, reproduces `_aggregate_bars` computed directly on in-memory cases.

    No training, no real `RESULTS_DIR` I/O.

    Builds one synthetic, plausible per-`(config, seed)` case dict for the FULL `CONFIGS` x
    `SEEDS` matrix, containing exactly the fields `_bar_i.._bar_iv` read. Computes the 4 bars
    directly on that in-memory list via `_aggregate_bars`, then round-trips every case through
    `_save_case`/`_load_case` into a throwaway temp directory (never `RESULTS_DIR`) and recomputes
    the bars from the reloaded cases the same way `--fold` would. Equal `bars` dicts prove fold
    aggregation is equivalent to monolithic aggregation by construction: same `_aggregate_bars`
    function, same `_bar_*` calls, same numeric inputs surviving the JSON round trip.
    """
    synth_cases: list[dict] = []
    for cfg in CONFIGS:
        for seed in SEEDS:
            advantage = 0.05 - 0.008 * cfg["dim"] + 0.001 * seed
            gold = None if cfg["broad"] else {"capture_rate": 0.75 - 0.02 * cfg["dim"], "modal_by_tercile": [1, 2, 3], "designed_modal": [1, 2, 3]}
            synth_cases.append(
                {
                    "config": cfg["name"],
                    "dim": cfg["dim"],
                    "rotated": cfg["rotated"],
                    "broad": cfg["broad"],
                    "seed": seed,
                    "n_train_router": 1250,
                    "n_eval_router": 1250,
                    "nll_blend_n_nbr50": 1.0,
                    "nll_hard_n_nbr50": 1.05,
                    "nll_blend_n_nbr25": 1.02,
                    "nll_hard_n_nbr25": 1.06,
                    "g5_blend_shift": -0.02,
                    "nll_global": 1.1,
                    "k_global": 2,
                    "advantage": advantage,
                    "advantage_se": 0.01,
                    "oracle_x_score": -1.0,
                    "oracle_noisy_score": -0.95,
                    "oracle_x_le_oracle_noisy": True,
                    "gold": gold,
                }
            )

    bars_direct = _aggregate_bars(synth_cases)

    with tempfile.TemporaryDirectory(prefix="t2_selftest_fold_") as tmp_dir:
        for case in synth_cases:
            _save_case(case, results_dir=tmp_dir)
        loaded_cases = []
        for cfg in CONFIGS:
            for seed in SEEDS:
                loaded = _load_case(cfg["name"], seed, results_dir=tmp_dir)
                assert loaded is not None, f"selftest fold round-trip: missing ({cfg['name']}, {seed})"
                loaded_cases.append(loaded)
        bars_folded = _aggregate_bars(loaded_cases)

    ok = bars_direct == bars_folded
    print(f"[t2 selftest] (c) fold-equivalence: {'PASS' if ok else 'FAIL'}")
    return ok


def run_selftest() -> bool:
    """Runs all selftest checks; must PASS before any real T2 read."""
    device = str(get_device())
    print(f"T2 selftest (device={device})")

    ok_a = _regression_guard(device)

    score, x = _synthetic_2d_table(seed=0)
    n = score.shape[0]
    train_idx = np.arange(0, n, 2)
    eval_idx = np.arange(1, n, 2)
    score_tr, x_tr = score[train_idx], x[train_idx]
    x_ev = x[eval_idx]

    targets = _knn_soft_targets(score_tr, x_tr, n_nbr=30)
    model = _train_router(x_tr, n_cols=6, device=device, soft_targets=targets, n_epochs=200, seed=0)
    col_hard = _route(model, x_ev, device)
    routed_c = np.arange(1, 7)[col_hard]
    tercile_bins = cl.quantile_bins(x_ev[:, 0], 3)
    modal_by_tercile = []
    for b in sorted(np.unique(tercile_bins)):
        vals, counts = np.unique(routed_c[tercile_bins == b], return_counts=True)
        modal_by_tercile.append(int(vals[np.argmax(counts)]))
    ok_b = modal_by_tercile == [1, 3, 6]
    print(f"[t2 selftest] (b) 2-D kNN-routed modal capacity by tercile = {modal_by_tercile} (designed [1, 3, 6]): {'PASS' if ok_b else 'FAIL'}")

    ok_c = _fold_equivalence_check()

    all_pass = ok_a and ok_b and ok_c
    print("all checks passed" if all_pass else "FAILURES PRESENT")
    return all_pass


# ---------------------------------------------------------------------------
# Real reader.
# ---------------------------------------------------------------------------


def _save_table(res: dict) -> None:
    """Saves the per-(config, seed) held-out score table + input coordinates (K4 table-key convention)."""
    torch.save(
        {
            "score": torch.as_tensor(res["_score"], dtype=torch.float64),
            "x": torch.as_tensor(res["_x"]),
            "c_grid": C_GRID,
            "config": res["config"],
            "dim": res["dim"],
            "rotated": res["rotated"],
            "broad": res["broad"],
            "seed": res["seed"],
            "col_hard_n_nbr50": res["_col_hard_n_nbr50"],
        },
        os.path.join(RESULTS_DIR, f"nested_toyD_ndim_{res['config']}_seed{res['seed']}.pt"),
    )


def _save_case(res: dict, results_dir: str | None = None) -> None:
    """Persists the full json-able per-case result dict (`_jsonable(res)`) to `nested_toyD_ndim_{config}_seed{seed}_case.json`, sibling to `_save_table`'s `.pt` file.

    Extends `_save_table` (untouched) so a no-training `--fold` step can reconstruct `all_cases`
    and recompute the bars without re-training. `results_dir` defaults to `RESULTS_DIR`; the
    selftest's fold-equivalence check overrides it with a throwaway temp directory so it never
    touches real T2 results.
    """
    d = RESULTS_DIR if results_dir is None else results_dir
    path = os.path.join(d, f"nested_toyD_ndim_{res['config']}_seed{res['seed']}_case.json")
    with open(path, "w") as f:
        json.dump(_jsonable(res), f, indent=2)


def _load_case(config_name: str, seed: int, results_dir: str | None = None) -> dict | None:
    """Loads one persisted per-case JSON written by `_save_case`, or returns `None` if it does not exist."""
    d = RESULTS_DIR if results_dir is None else results_dir
    path = os.path.join(d, f"nested_toyD_ndim_{config_name}_seed{seed}_case.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _case_pairs(configs: list[dict], seeds: list[int]) -> list[tuple[dict, int]]:
    """Ordered `(config, seed)` pairs, CONFIGS-major then SEEDS-minor — the exact iteration order the loop over `CONFIGS`/`SEEDS` used before sharding existed."""
    return [(cfg, seed) for cfg in configs for seed in seeds]


def _aggregate_bars(all_cases: list[dict]) -> dict:
    """Computes the 4 pre-registered bars from a full `all_cases` list — the SOLE bar-aggregation logic.

    Shared verbatim by the monolithic training path and `--fold`, so bars are identical by
    construction whether `all_cases` came straight out of `run_case` or was reconstructed from
    `--fold`'s persisted per-case JSON files.
    """
    by_config: dict[str, list[dict]] = {}
    for c in all_cases:
        by_config.setdefault(c["config"], []).append(c)
    return {
        "i_dim2_axis_beats_global": _bar_i(by_config.get("dim2_axis", [])),
        "ii_degradation_curve": _bar_ii({cfg["dim"]: by_config.get(cfg["name"], []) for cfg in CONFIGS if not cfg["rotated"] and not cfg["broad"]}),
        "iii_rotated_vs_axis_dim5": _bar_iii(by_config.get("dim5_axis", []), by_config.get("dim5_rotated", [])),
        "iv_dim2_broad_bounded": _bar_iv(by_config.get("dim2_broad", [])),
    }


def _summary_config_block() -> dict:
    """The static per-run `"config"` block written into `t2_summary.json`.

    Pure constants, no training-derived state, so it is identical whether populated by the
    monolithic path or `--fold`.
    """
    return {
        "N_TRAIN": N_TRAIN, "N_TEST": N_TEST, "k_max": K_MAX, "n_epochs_ladder": N_EPOCHS_LADDER,
        "router_hidden": list(ROUTER_HIDDEN), "router_n_epochs": ROUTER_N_EPOCHS, "router_lr": ROUTER_LR,
        "n_nbr": N_NBR, "n_nbr_half": N_NBR_HALF, "seeds": list(SEEDS), "configs": CONFIGS,
    }


def _resolve_only_configs(parser: argparse.ArgumentParser, raw: str | None) -> list[dict] | None:
    """Parses `--only-configs` into the matching `CONFIGS` entries (in the order named), or `None` if the flag was not passed. Exits via `parser.error` on an unknown name."""
    if raw is None:
        return None
    names = [s.strip() for s in raw.split(",") if s.strip()]
    by_name = {c["name"]: c for c in CONFIGS}
    unknown = [n for n in names if n not in by_name]
    if unknown:
        parser.error(f"--only-configs: unknown config name(s) {unknown}; valid names: {sorted(by_name)}")
    return [by_name[n] for n in names]


def _resolve_only_seeds(parser: argparse.ArgumentParser, raw: str | None) -> list[int] | None:
    """Parses `--only-seeds` into a list of validated seeds (must be members of `SEEDS`), or `None` if the flag was not passed.

    Exits via `parser.error` on a malformed or unknown seed.
    """
    if raw is None:
        return None
    try:
        seeds = [int(s.strip()) for s in raw.split(",") if s.strip()]
    except ValueError:
        parser.error(f"--only-seeds: could not parse integer seeds from {raw!r}")
    unknown = [s for s in seeds if s not in SEEDS]
    if unknown:
        parser.error(f"--only-seeds: unknown seed(s) {unknown}; valid seeds: {list(SEEDS)}")
    return seeds


def _run_fold() -> None:
    """`--fold`: no training, no device work.

    Reconstructs `all_cases` from persisted per-case JSON files for the FULL `CONFIGS` x `SEEDS`
    matrix and writes `t2_summary.json` exactly as the monolithic path would.

    Aggregation runs through the identical `_aggregate_bars`/`_summary_config_block` helpers the
    monolithic path uses, so the `bars` and `config` blocks are identical by construction to what
    the monolithic path would have produced from the same underlying `run_case` results. Any
    missing `(config, seed)` case file aborts before a summary is written (no partial output).
    """
    pairs = _case_pairs(CONFIGS, list(SEEDS))
    cases: list[dict] = []
    missing: list[tuple[str, int]] = []
    wall_total = 0.0
    for cfg, seed in pairs:
        case = _load_case(cfg["name"], seed)
        if case is None:
            missing.append((cfg["name"], seed))
            continue
        wall_total += float(case.pop("wall_sec", 0.0))
        cases.append(case)

    if missing:
        print(f"[t2] --fold: missing persisted case file(s) for {missing}; run the missing shard(s) first (each writes nested_toyD_ndim_<config>_seed<seed>_case.json).")
        sys.exit(1)

    bars = _aggregate_bars(cases)
    for name, res in bars.items():
        print(f"[t2] bar {name}: {res}")

    summary = {
        "config": _summary_config_block(),
        "cases": _jsonable(cases),
        "bars": bars,
        "wall_time_sec": wall_total,
    }
    out_path = os.path.join(RESULTS_DIR, "t2_summary.json")
    with open(out_path, "w") as f:
        json.dump(_jsonable(summary), f, indent=2)
    print(f"[t2] --fold: wrote {out_path}")
    print(f"[t2] --fold: aggregated wall-time (sum of persisted per-case times): {wall_total:.1f}s ({wall_total / 3600.0:.2f} h)")


def main() -> None:
    """Runs the T2 battery: `--selftest`, `--measure-one`, a full/sharded training run (`--only-configs`/`--only-seeds`), or `--fold`.

    `--fold` performs no-training aggregation of a sharded run's persisted per-case results into
    `t2_summary.json`.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="Run the T2 selftest (regression guard + 2-D synthetic recovery + fold-equivalence) and exit.")
    parser.add_argument("--measure-one", action="store_true", help="Time ONE dim=5-axis (config, seed=0) unit, print the extrapolated matrix wall-time, and exit.")
    parser.add_argument(
        "--only-configs", type=str, default=None,
        help="Comma-separated CONFIG names (see CONFIGS) to train as a shard; omit to include all configs. "
        "Passing this (or --only-seeds) makes this a shard run: bars/t2_summary.json are NOT written — run again with --fold once the full matrix is persisted.",
    )
    parser.add_argument(
        "--only-seeds", type=str, default=None,
        help="Comma-separated seeds (see SEEDS) to train as a shard; omit to include all seeds. Passing this (or --only-configs) makes this a shard run — see --only-configs.",
    )
    parser.add_argument(
        "--fold", action="store_true",
        help="No training, no device work: fold persisted per-(config, seed) case files for the FULL CONFIGS x SEEDS matrix into t2_summary.json, "
        "byte-for-byte as the monolithic (no-flags) path would have written it. Cannot be combined with --only-configs/--only-seeds.",
    )
    args = parser.parse_args()

    if args.selftest:
        sys.exit(0 if run_selftest() else 1)

    if args.fold:
        if args.only_configs or args.only_seeds:
            parser.error("--fold cannot be combined with --only-configs/--only-seeds")
        _run_fold()
        return

    device = str(get_device())

    if args.measure_one:
        cfg = next(c for c in CONFIGS if c["name"] == "dim5_axis")
        print(f"[t2] measure-one: config={cfg['name']} seed=0 device={device} N_TRAIN={N_TRAIN} N_TEST={N_TEST} k_max={K_MAX} n_epochs_ladder={N_EPOCHS_LADDER}")
        t0 = time.time()
        run_case(cfg, seed=0, device=device)
        wall = time.time() - t0
        n_units = len(CONFIGS) * len(SEEDS)
        print(f"[t2] one unit wall-time: {wall:.1f}s ({wall / 60.0:.2f} min)")
        print(f"[t2] extrapolated {n_units}-unit matrix (serialized): {wall * n_units:.0f}s ({wall * n_units / 3600.0:.2f} h)")
        return

    configs_only = _resolve_only_configs(parser, args.only_configs)
    seeds_only = _resolve_only_seeds(parser, args.only_seeds)
    is_shard = configs_only is not None or seeds_only is not None
    configs_eff = configs_only if configs_only is not None else CONFIGS
    seeds_eff = seeds_only if seeds_only is not None else list(SEEDS)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"[t2] device={device} N_TRAIN={N_TRAIN} N_TEST={N_TEST} k_max={K_MAX} n_epochs_ladder={N_EPOCHS_LADDER} n_nbr={N_NBR}/{N_NBR_HALF}")
    if is_shard:
        print(f"[t2] shard run (configs={[c['name'] for c in configs_eff]} seeds={list(seeds_eff)}); aggregate deferred to --fold")

    t_start = time.time()
    all_cases: list[dict] = []
    for cfg, seed in _case_pairs(configs_eff, seeds_eff):
        print(f"=== T2 {cfg['name']} seed={seed} ===")
        t0 = time.time()
        res = run_case(cfg, seed, device)
        wall = time.time() - t0
        all_cases.append(res)
        _save_table(res)
        case_for_save = dict(res)
        case_for_save["wall_sec"] = wall
        _save_case(case_for_save)
        gold_str = f" gold_capture={res['gold']['capture_rate']:.3f} modal={res['gold']['modal_by_tercile']}" if res["gold"] else ""
        print(f"  advantage={res['advantage']:+.4f} (se={res['advantage_se']:.4f})  nll_global={res['nll_global']:.4f} nll_blend={res['nll_blend_n_nbr50']:.4f}"
              f"  oracle_x<=oracle_noisy={res['oracle_x_le_oracle_noisy']}  g5_shift={res['g5_blend_shift']:+.4f}{gold_str}  wall={wall:.1f}s")

    if is_shard:
        print(f"[t2] shard run complete ({len(all_cases)} case(s) persisted); run with --fold once the full {len(CONFIGS)}x{len(SEEDS)} matrix is persisted.")
        return

    bars = _aggregate_bars(all_cases)
    for name, res in bars.items():
        print(f"[t2] bar {name}: {res}")

    wall_total = time.time() - t_start
    summary = {
        "config": _summary_config_block(),
        "cases": _jsonable(all_cases),
        "bars": bars,
        "wall_time_sec": wall_total,
    }
    out_path = os.path.join(RESULTS_DIR, "t2_summary.json")
    with open(out_path, "w") as f:
        json.dump(_jsonable(summary), f, indent=2)
    print(f"[t2] wrote {out_path}")
    print(f"[t2] total wall-time: {wall_total:.1f}s ({wall_total / 3600.0:.2f} h)")


if __name__ == "__main__":
    main()
