"""K6 — router distillation (deployable per-input k), WS1's post-R1 continuation.

R2 (`capacity_ladder_results/R2_verdict.md`) certified that the per-input latent count is
recovered on toy D via the held-out ARBITER (`capacity_ladder_k5.py`'s neighbour-averaged
top-vs-bottom advantage), while the per-input hard KNEE label is UNFAITHFUL — noisy and
count-collapsing across seeds. K6 distills the per-input read into a small, deployable
router π(x) that trains in seconds off the ALREADY-TRAINED K4 nested score tables (no
retraining of the capacity ladder), and in doing so tests R2's finding directly: a router
trained on the unfaithful hard knee labels is expected to underperform one trained on the
smoother per-bin stacked responsibilities.

Three router arms, one small MLP classifier `x -> logits over c_grid` each (hidden (32,32),
Adam, softmax over the k columns):

  * HARD:  cross-entropy to the neighbour-averaged per-input knee label (the K5 read).
  * SOFT:  soft-label cross-entropy (== KL up to the target's own entropy) to per-input
    responsibilities `q_i = softmax_c(score[i,c] + log pi_bin(i))`, `pi_bin` from per-bin
    (tercile) EM stacking — the smoother, less-discretized target.
  * PILOT: cross-entropy to the raw per-example argmax_c score[i,c] — no neighbour
    averaging at all; registered as a HYPOTHESIS test expected to underperform SOFT.

Each toy/seed's held-out K4 rows are split (deterministic, index parity — no unseeded RNG)
into a router-TRAIN half (targets computed here, routers fit here) and a router-EVAL half
(routed held-out mixture NLL measured here only). Strictly probabilistic: ordinary
supervised likelihood, no penalty, no tuned lambda.

Run `--selftest` before any real read (synthetic known-answer table, in-memory, no disk).
Run with no flags to read every `capacity_ladder_results/K4/nested_toy*_seed*.pt` table
(missing files are skipped, matching `capacity_ladder_k5.py`'s `read_case`).

NOTE: this is a standalone x -> c_grid classifier over toy data. Retargeting the real
model's `n_classes_predictor` "k=2..k_max then bypass-LAST" logit layout is a K7 concern
(EXECUTION_PLAN.md's K6 "real-model hook (survey)") — out of scope here.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root, so `import automl_package` works

import _capacity_ladder as cl  # noqa: E402
import capacity_ladder_k5 as ck5  # noqa: E402 — reuse perinput_knee_curve verbatim (the K5 read)

from automl_package.utils.pytorch_utils import get_device  # noqa: E402

OUT_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "K6")
K4_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "K4")

SEEDS = (0, 1, 2)
STRUCTURED_TOYS = ("C", "D", "E")
WIDTH = 0.075  # neighbourhood half-width, matches K4/K5's box-car
HIDDEN = (32, 32)
N_EPOCHS = 300
LR = 1e-2
BOOT_N = 1000  # for the global-knee baseline's bootstrap SE


# ---------------------------------------------------------------------------
# Router model.
# ---------------------------------------------------------------------------


class _RouterMLP(nn.Module):
    """A small `x (scalar) -> logits over c_grid columns` classifier, hidden (32, 32) + ReLU."""

    def __init__(self, n_cols: int, hidden: tuple[int, ...] = HIDDEN) -> None:
        super().__init__()
        dims = [1, *hidden]
        layers: list[nn.Module] = []
        for d_in, d_out in itertools.pairwise(dims):
            layers += [nn.Linear(d_in, d_out), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], n_cols))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`(N, 1) -> (N, n_cols)` logits."""
        return self.net(x)


def _train_router(
    x: np.ndarray,
    *,
    n_cols: int,
    device: str,
    hard_labels: np.ndarray | None = None,
    soft_targets: np.ndarray | None = None,
    n_epochs: int = N_EPOCHS,
    lr: float = LR,
    seed: int = 0,
    hidden: tuple[int, ...] = HIDDEN,
) -> _RouterMLP:
    """Trains one router: hard cross-entropy if `hard_labels` given, else soft-label CE to `soft_targets`.

    Exactly one of `hard_labels` (integer column index per row) or `soft_targets` (`(N, n_cols)`
    per-row distribution) must be given. Soft-label CE is `-mean_i sum_c q_i[c] log softmax(logits)_c`,
    which minimizes `KL(q_i || router(x_i))` up to the additive `q_i`-entropy term (a constant w.r.t.
    router parameters), so gradients are identical to the true KL objective.

    Args:
        x: `(N,)` scalar input coordinate.
        n_cols: number of router output columns (== `len(c_grid)`).
        device: torch device string.
        hard_labels: `(N,)` int class index per row, or None (use `soft_targets`).
        soft_targets: `(N, n_cols)` per-row target distribution, or None (use `hard_labels`).
        n_epochs: full-batch Adam epochs.
        lr: Adam learning rate.
        seed: torch RNG seed for init (reproducibility).
        hidden: router MLP hidden-layer sizes (default `HIDDEN=(32, 32)`; width-cert W6 threads a
            non-default value through here to probe whether the deploy/dial bars are sensitive to
            router capacity, not just net capacity).

    Returns:
        The trained `_RouterMLP`, in eval mode.
    """
    if (hard_labels is None) == (soft_targets is None):
        raise ValueError("exactly one of hard_labels or soft_targets must be given")
    torch.manual_seed(seed)
    model = _RouterMLP(n_cols, hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x_t = torch.as_tensor(np.asarray(x, dtype=np.float32), device=device).reshape(-1, 1)
    if hard_labels is not None:
        y_t = torch.as_tensor(np.asarray(hard_labels, dtype=np.int64), device=device)
    else:
        q_t = torch.as_tensor(np.asarray(soft_targets, dtype=np.float32), device=device)

    model.train()
    for _ in range(n_epochs):
        opt.zero_grad()
        logits = model(x_t)
        loss = nnf.cross_entropy(logits, y_t) if hard_labels is not None else -(q_t * nnf.log_softmax(logits, dim=1)).sum(dim=1).mean()
        loss.backward()
        opt.step()
    model.eval()
    return model


def _route(model: _RouterMLP, x: np.ndarray, device: str) -> np.ndarray:
    """Argmax-routed column index per row of `x`."""
    with torch.no_grad():
        x_t = torch.as_tensor(np.asarray(x, dtype=np.float32), device=device).reshape(-1, 1)
        logits = model(x_t)
    return logits.argmax(dim=1).cpu().numpy()


# ---------------------------------------------------------------------------
# Targets (computed on the router-TRAIN half only — no eval leakage).
# ---------------------------------------------------------------------------


def _knee_labels(score_src: np.ndarray, x_src: np.ndarray, query_x: np.ndarray, c_grid: list[int], width: float = WIDTH) -> np.ndarray:
    """Neighbour-averaged per-input knee label (c_grid VALUES) at `query_x`, the exact K5 read.

    `(score_src, x_src)` supply the neighbourhood; `query_x` is where the curve is evaluated —
    they may differ (e.g. label the eval half's x-points using only the train half's table, so
    no eval information ever informs a router or a baseline).
    """
    out = cl.perinput_curve(score_src, x_src, width, ref_c=0, query_x=query_x)
    delta = np.asarray(out["delta"])
    return ck5.perinput_knee_curve(delta, c_grid)


def _soft_targets(score_train: np.ndarray, x_train: np.ndarray) -> np.ndarray:
    """Per-example soft responsibilities `q_i = softmax_c(score[i,c] + log pi_bin(i))`.

    `pi_bin` is the per-tercile EM-stacked mixture weight (`_capacity_ladder.perbin_stack`) —
    the smoother, less-discretized alternative to the hard knee label; this is exactly the E-step
    of `stack_em` but with a per-bin `pi` instead of a single global one.
    """
    bins = cl.quantile_bins(x_train, 3)
    pi_bins = cl.perbin_stack(score_train, bins)
    log_pi_bin = np.log(np.clip(np.stack([pi_bins[b] for b in bins], axis=0), 1e-300, None))  # (N, C)
    unnorm = score_train + log_pi_bin
    unnorm = unnorm - unnorm.max(axis=1, keepdims=True)
    w = np.exp(unnorm)
    return w / w.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Evaluation.
# ---------------------------------------------------------------------------


def _routed_nll(score: np.ndarray, col_idx: np.ndarray) -> float:
    """`mean_i(-score[i, col_idx[i]])` — the routed held-out mixture NLL."""
    rows = np.arange(score.shape[0])
    return float(-score[rows, col_idx].mean())


def _agreement(routed_c: np.ndarray, knee_c: np.ndarray, interior: np.ndarray) -> float | None:
    """Fraction of interior points where the router and the K5 knee read differ by <= 1 in k."""
    if not interior.any():
        return None
    return float(np.mean(np.abs(routed_c[interior] - knee_c[interior]) <= 1))


def route_case(score: np.ndarray, x: np.ndarray, c_grid: list[int], device: str, seed: int = 0) -> dict:
    """Trains the 3 router arms on a router-TRAIN half; evaluates routed NLL on the router-EVAL half.

    Args:
        score: `(N, len(c_grid))` held-out per-example log-likelihood table (a K4 nested table).
        x: `(N,)` input coordinate paired with each row of `score`.
        c_grid: capacity value per column of `score`.
        device: torch device string.
        seed: RNG/torch seed (router init + global-knee bootstrap).

    Returns:
        dict of the 3 routed NLLs, the GLOBAL-knee and ORACLE baselines, per-arm agreement with
        the K5 knee read, and the pre-registered checks (i)-(iii), all as plain python scalars.
    """
    n = score.shape[0]
    n_cols = len(c_grid)
    col_of = {c: j for j, c in enumerate(c_grid)}
    c_arr = np.asarray(c_grid)

    train_idx = np.arange(0, n, 2)  # deterministic index-parity split, no RNG
    eval_idx = np.arange(1, n, 2)
    score_tr, x_tr = score[train_idx], x[train_idx]
    score_ev, x_ev = score[eval_idx], x[eval_idx]

    # --- targets, TRAIN half only ---
    hard_label_c = _knee_labels(score_tr, x_tr, x_tr, c_grid)
    hard_label_j = np.array([col_of[c] for c in hard_label_c], dtype=np.int64)
    soft_q = _soft_targets(score_tr, x_tr)
    pilot_label_j = score_tr.argmax(axis=1)

    # --- train the 3 arms ---
    m_hard = _train_router(x_tr, n_cols=n_cols, device=device, hard_labels=hard_label_j, seed=seed)
    m_soft = _train_router(x_tr, n_cols=n_cols, device=device, soft_targets=soft_q, seed=seed)
    m_pilot = _train_router(x_tr, n_cols=n_cols, device=device, hard_labels=pilot_label_j, seed=seed)

    # --- global-knee baseline, TRAIN half only (same info budget as the routers) ---
    r_star, _delta_curve, _se = cl.knee(score_tr, ref_c=1, n_boot=BOOT_N, c_grid=c_grid, seed=seed)
    k_global = 1 if r_star == 0 else r_star  # G2 abstain => the k=1 bypass single Gaussian
    global_col = col_of[k_global]

    # --- routed NLL, EVAL half ---
    col_hard = _route(m_hard, x_ev, device)
    col_soft = _route(m_soft, x_ev, device)
    col_pilot = _route(m_pilot, x_ev, device)
    nll_hard = _routed_nll(score_ev, col_hard)
    nll_soft = _routed_nll(score_ev, col_soft)
    nll_pilot = _routed_nll(score_ev, col_pilot)
    nll_global = _routed_nll(score_ev, np.full(score_ev.shape[0], global_col, dtype=np.int64))
    nll_oracle = float(-score_ev.max(axis=1).mean())

    # --- agreement vs the K5 knee read, interior eval points (domain-edge margin = WIDTH) ---
    knee_eval_c = _knee_labels(score_tr, x_tr, x_ev, c_grid)
    interior = (x_ev > x_ev.min() + WIDTH) & (x_ev < x_ev.max() - WIDTH)
    agreement_hard = _agreement(c_arr[col_hard], knee_eval_c, interior)
    agreement_soft = _agreement(c_arr[col_soft], knee_eval_c, interior)
    agreement_pilot = _agreement(c_arr[col_pilot], knee_eval_c, interior)

    return {
        "n_train": len(train_idx),
        "n_eval": len(eval_idx),
        "k_global": int(k_global),
        "nll_hard": nll_hard,
        "nll_soft": nll_soft,
        "nll_pilot": nll_pilot,
        "nll_global": nll_global,
        "nll_oracle": nll_oracle,
        "agreement_hard": agreement_hard,
        "agreement_soft": agreement_soft,
        "agreement_pilot": agreement_pilot,
        "checks": {
            "soft_le_hard": bool(nll_soft <= nll_hard),  # (i) soft never worse than hard
            "soft_le_global": bool(nll_soft <= nll_global),  # (ii) beats global on D/E, ties on C (toy-dependent read)
            "pilot_ge_soft": bool(nll_pilot >= nll_soft),  # (iii) pilot underperforms soft (hypothesis test)
        },
    }


# ---------------------------------------------------------------------------
# Selftest — synthetic known-answer table (N=1500, C=6), in-memory, no disk.
# ---------------------------------------------------------------------------

_ST_N = 1500
_ST_C_GRID = [1, 2, 3, 4, 5, 6]
_ST_PEAKS = (1, 3, 6)  # true best column by x-tercile: x<1/3 -> c=1, 1/3<=x<2/3 -> c=3, x>=2/3 -> c=6


def _selftest_table(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A synthetic `(N, 6)` table whose best column is 1/3/6 by x-tercile, plus noise.

    `k_max=6` (not 3) and widely-spaced tercile peaks are deliberate, not cosmetic: they
    reproduce the failure mode `perinput_knee_curve` actually has on the real toys (R2
    verdict: "the knee is noisy and count-collapsing"). The knee is a SEQUENTIAL walk up
    the ladder that locks in at the first step whose neighbour-averaged increment fails —
    one noisy dip anywhere between c=1 and the true peak causes premature stopping, and a
    longer walk (more intermediate rungs, as in the real toys' k_max=6-8) gives that noise
    more chances to bite. A 3-column table with adjacent peaks (1/2/3) doesn't exercise this
    at all (checked empirically: SOFT does not reliably beat HARD there); this table does.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=_ST_N)
    peak = np.select([x < 1.0 / 3.0, x < 2.0 / 3.0], _ST_PEAKS[:2], default=_ST_PEAKS[2]).astype(np.float64)
    c = np.asarray(_ST_C_GRID, dtype=np.float64)
    base = -0.5 * (c[None, :] - peak[:, None]) ** 2
    score = base + rng.normal(0.0, 0.8, size=(_ST_N, len(_ST_C_GRID)))
    return score, x


def run_selftest() -> bool:
    """Runs the K6 selftest: known-answer table, full router pipeline, checks (a)/(b)."""
    device = str(get_device())
    print(f"K6 router selftest (N={_ST_N}, C={len(_ST_C_GRID)}, device={device})")
    score, x = _selftest_table(seed=0)
    res = route_case(score, x, _ST_C_GRID, device, seed=0)

    ok_global = res["checks"]["soft_le_global"]
    ok_hard = res["checks"]["soft_le_hard"]
    print(f"  NLL: hard={res['nll_hard']:.4f} soft={res['nll_soft']:.4f} pilot={res['nll_pilot']:.4f} global={res['nll_global']:.4f} oracle={res['nll_oracle']:.4f}")
    print(f"  agreement vs knee read: hard={res['agreement_hard']:.3f} soft={res['agreement_soft']:.3f} pilot={res['agreement_pilot']:.3f}")
    print(f"  (a) soft NLL <= global-knee NLL ({res['nll_soft']:.4f} <= {res['nll_global']:.4f}): {'PASS' if ok_global else 'FAIL'}")
    print(f"  (b) soft NLL <= hard NLL ({res['nll_soft']:.4f} <= {res['nll_hard']:.4f}): {'PASS' if ok_hard else 'FAIL'}")
    all_pass = ok_global and ok_hard
    print("all checks passed" if all_pass else "FAILURES PRESENT")
    return all_pass


# ---------------------------------------------------------------------------
# Real read: every K4 nested table on disk.
# ---------------------------------------------------------------------------


def read_table(toy: str, seed: int) -> dict | None:
    """Loads one K4 nested score table, or None if it hasn't been built yet."""
    path = os.path.join(K4_DIR, f"nested_toy{toy}_seed{seed}.pt")
    if not os.path.exists(path):
        return None
    d = torch.load(path, weights_only=False)
    return {
        "score": np.asarray(d["score"], dtype=np.float64),
        "x": np.asarray(d["x"], dtype=np.float64).ravel(),
        "c_grid": list(d["c_grid"]),
    }


def _jsonable(obj: object) -> object:
    """Recursively drops private in-memory keys (leading `_`) and casts numpy scalars, matching K4/K5."""
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
    """Reads every available K4 table into the K6 router-distillation summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="Run the K6 synthetic known-answer selftest and exit.")
    args = parser.parse_args()

    if args.selftest:
        ok = run_selftest()
        sys.exit(0 if ok else 1)

    os.makedirs(OUT_DIR, exist_ok=True)
    device = str(get_device())
    print(f"[k6] device={device}")
    summary: dict[str, object] = {
        "config": {"width": WIDTH, "hidden": list(HIDDEN), "n_epochs": N_EPOCHS, "lr": LR, "seeds": list(SEEDS)},
        "structured": {},
    }

    for toy in STRUCTURED_TOYS:
        per_seed = []
        for seed in SEEDS:
            tbl = read_table(toy, seed)
            if tbl is None:
                print(f"[k6] {toy} s{seed}: no K4 table yet, skipping")
                continue
            res = route_case(tbl["score"], tbl["x"], tbl["c_grid"], device, seed=seed)
            res["toy"], res["seed"] = toy, seed
            per_seed.append(res)
            print(
                f"[k6] {toy} s{seed}: NLL hard={res['nll_hard']:.4f} soft={res['nll_soft']:.4f} pilot={res['nll_pilot']:.4f} "
                f"global={res['nll_global']:.4f} oracle={res['nll_oracle']:.4f}  agreement(soft)={res['agreement_soft']}  checks={res['checks']}"
            )
        if per_seed:
            summary["structured"][toy] = per_seed

    out_path = os.path.join(OUT_DIR, "k6_summary.json")
    with open(out_path, "w") as f:
        json.dump(_jsonable(summary), f, indent=2)
    print(f"[k6] wrote {out_path}")


if __name__ == "__main__":
    main()
