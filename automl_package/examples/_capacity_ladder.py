"""Shared post-hoc capacity-ladder library (WS1/WS2/WS3, task K0).

One score table, five readers. Every WS in `docs/plans/capacity_ladder_2026-07-09/
EXECUTION_PLAN.md` (taxonomy in its §0, governance in its §0b) reduces to the SAME
post-hoc object: `score[i, c]` = held-out log-likelihood of example i under capacity c
(§0's B2/B4/B5). This module builds that table from either of the two model shapes the
program uses, then reads it four ways — global stacking (B4), the knee arbiter (B2),
per-bin stacking (B5), and the per-input neighbour-averaged curve (also B5) — with the
§0b G-rules (bootstrap SE, abstain, cap-saturation, per-bin cells, locality guard,
full-vector reporting) built into the readers rather than left to callers to remember.

Run `--selftest` before any real read (synthetic known-answer tables); run
`--build-tables` to (re)generate the K0 score tables from the existing June-arc
artifacts (fixed-k MDN sweep + `AggregateSparsityKSelector`) on toys C/D/E.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root, so `import automl_package` works

import _toy_datasets as td  # only used by the artifact-building section, below
import _variational_em_perinput as vemp

from automl_package.models.baselines.mixture_density_network import MixtureDensityNetwork

RESULTS_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "K0")

# ---------------------------------------------------------------------------
# Score table (the primary artifact — everything else consumes only this).
# ---------------------------------------------------------------------------


@runtime_checkable
class NestedComponentsModel(Protocol):
    """A single fitted model exposing ALL components from one cached forward pass.

    Matches the real architecture's own masking convention
    (`probabilistic_regression_net.py:132-149`, `_compute_predictions_for_k`): capacity
    c is read by masking components beyond index c and renormalizing, never by
    re-running the model.
    """

    def nested_components(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns `(logits, means, log_vars)`, each `(N, K_max)`, component 0 first."""
        ...


def _per_example_logprob(model: Any, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Duck-typed per-example held-out log-likelihood of one separately-trained model.

    Tries, in order: `.log_prob_per_example(x, y)`, then the automl_package convention
    `.predict_distribution(x, filter_data=False).log_prob(y)` (MDN, ProbReg, XGBoost, ...).
    """
    if hasattr(model, "log_prob_per_example"):
        return np.asarray(model.log_prob_per_example(x, y), dtype=np.float64)
    if hasattr(model, "predict_distribution"):
        dist = model.predict_distribution(np.asarray(x, dtype=np.float32), filter_data=False)
        y_arr = np.asarray(y, dtype=np.float32).ravel()
        return np.asarray(dist.log_prob(y_arr), dtype=np.float64)
    raise TypeError(f"model {model!r} exposes neither log_prob_per_example nor predict_distribution")


def score_table(model_like: Any, x: np.ndarray, y: np.ndarray, c_grid: Sequence[int]) -> np.ndarray:
    """Builds the `(N, C)` held-out per-example log-likelihood table (the shared artifact).

    `model_like` may be either of the two shapes the program's models come in:

    * `Mapping[int, model]` — k-indexed SEPARATELY-TRAINED models (e.g. the fixed-k MDN
      sweep). Column c stacks the per-example log-likelihood of `model_like[c]` on
      `(x, y)`, one full model evaluation per column.
    * a `NestedComponentsModel` — ONE over-provisioned model whose per-component
      `(logit, mean, log_var)` are cached from a SINGLE forward pass
      (`.nested_components(x)`). Column c renormalizes the prefix of the first c
      components (`logits[:, :c]`, softmax-masked exactly like
      `probabilistic_regression_net.py:132-149`) and scores the resulting c-component
      Gaussian mixture — no re-running the model per capacity.

    Args:
        model_like: see above.
        x: held-out features, `(N, D)` or `(N,)`.
        y: held-out targets, `(N,)`.
        c_grid: capacities to score, e.g. `[1, 2, ..., 6]`.

    Returns:
        `(N, len(c_grid))` float64 array; column j is capacity `c_grid[j]`.
    """
    c_list = list(c_grid)
    y_t = torch.as_tensor(np.asarray(y, dtype=np.float64).ravel())
    n = y_t.shape[0]

    if isinstance(model_like, Mapping):
        cols = []
        for c in c_list:
            if c not in model_like:
                raise KeyError(f"c_grid entry {c} missing from model_like mapping (keys={sorted(model_like.keys())})")
            cols.append(torch.as_tensor(_per_example_logprob(model_like[c], x, y)))
        mat = torch.stack(cols, dim=1)
    elif hasattr(model_like, "nested_components"):
        logits_np, means_np, log_vars_np = model_like.nested_components(x)
        logits = torch.as_tensor(np.asarray(logits_np, dtype=np.float64))
        means = torch.as_tensor(np.asarray(means_np, dtype=np.float64))
        log_vars = torch.as_tensor(np.asarray(log_vars_np, dtype=np.float64))
        k_max = logits.shape[1]
        cols = []
        for c in c_list:
            if not (1 <= c <= k_max):
                raise ValueError(f"c={c} outside available component range 1..{k_max}")
            prefix_logits = logits[:, :c]
            log_w = prefix_logits - torch.logsumexp(prefix_logits, dim=1, keepdim=True)
            var = log_vars[:, :c].exp().clamp_min(1e-12)
            y_col = y_t.unsqueeze(1).expand(-1, c)
            log_phi = -0.5 * (math.log(2.0 * math.pi) + var.log() + (y_col - means[:, :c]) ** 2 / var)
            cols.append(torch.logsumexp(log_w + log_phi, dim=1))
        mat = torch.stack(cols, dim=1)
    else:
        raise TypeError("model_like must be a Mapping[c, model] (k-indexed sweep) or a NestedComponentsModel")

    if tuple(mat.shape) != (n, len(c_list)):
        raise AssertionError(f"score_table shape {tuple(mat.shape)} != expected {(n, len(c_list))}")
    return mat.double().numpy()


# ---------------------------------------------------------------------------
# B4 — global stacking (EM).
# ---------------------------------------------------------------------------


def stack_em(score: np.ndarray, *, tol: float = 1e-10, max_iter: int = 500) -> np.ndarray:
    """EM stacking: maximizes `mean_i logsumexp_c(log pi_c + score[i,c])` over the simplex.

    E-step: `q_i = softmax_c(log pi_c + score[i,c])`. M-step: `pi <- mean_i q_i`. Init
    uniform; stops at `max|Δpi| < tol` or `max_iter` (defaults per K0 spec: 1e-10, 500).
    `torch.logsumexp` throughout.

    Args:
        score: `(N, C)` held-out log-likelihood table.
        tol: convergence tolerance on `max |Δpi|`.
        max_iter: iteration cap.

    Returns:
        `(C,)` float64 `pi_hat`, sums to 1.
    """
    s = torch.as_tensor(np.asarray(score, dtype=np.float64))
    n, c = s.shape
    log_pi = torch.full((c,), -math.log(c), dtype=torch.float64)
    for _ in range(max_iter):
        log_q_unnorm = log_pi.unsqueeze(0) + s
        log_q = log_q_unnorm - torch.logsumexp(log_q_unnorm, dim=1, keepdim=True)
        new_log_pi = torch.logsumexp(log_q, dim=0) - math.log(n)
        delta = (new_log_pi.exp() - log_pi.exp()).abs().max().item()
        log_pi = new_log_pi
        if delta < tol:
            break
    return log_pi.exp().numpy()


def mixture_logscore(pi_hat: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Per-example held-out log score of a stacked mixture: `logsumexp_c(log pi_c + score[i,c])`.

    The evaluation counterpart of `stack_em` — how a (global or per-bin) `pi_hat` scores
    on held-out data, used to compare candidate stacked reads (e.g. per-bin vs global,
    K0 selftest check (ii); K2's per-bin-beats-global registration).

    Args:
        pi_hat: `(C,)` mixing weights, sums to 1.
        score: `(N, C)` held-out log-likelihood table.

    Returns:
        `(N,)` per-example log score.
    """
    p = torch.as_tensor(np.asarray(pi_hat, dtype=np.float64))
    s = torch.as_tensor(np.asarray(score, dtype=np.float64))
    log_pi = torch.log(p.clamp_min(1e-300))
    return torch.logsumexp(log_pi.unsqueeze(0) + s, dim=1).numpy()


# ---------------------------------------------------------------------------
# Bootstrap SE (G1) — shared by knee and any caller wanting a paired resample.
# ---------------------------------------------------------------------------


def _bootstrap_col_means(s: np.ndarray, n_boot: int, block: np.ndarray | None, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap distribution of per-column means, `(n_boot, C)`.

    PAIRED: the same resample is applied to every column, so differences between
    columns (the increments/deltas the readers care about) are compared on the SAME
    resampled held-out examples — the correct design for this kind of comparison.
    Plain i.i.d. row bootstrap when `block` is None; block-bootstrap over the unique
    values of `block` (resample independent units with replacement) otherwise.
    """
    n = s.shape[0]
    if block is None:
        idx = rng.integers(0, n, size=(n_boot, n))
        return s[idx].mean(axis=1)
    block_arr = np.asarray(block)
    uniq = np.unique(block_arr)
    block_rows = [np.where(block_arr == u)[0] for u in uniq]
    out = np.empty((n_boot, s.shape[1]), dtype=np.float64)
    for b in range(n_boot):
        chosen = rng.integers(0, len(uniq), size=len(uniq))
        rows = np.concatenate([block_rows[k] for k in chosen])
        out[b] = s[rows].mean(axis=0)
    return out


# ---------------------------------------------------------------------------
# B2 — the knee arbiter.
# ---------------------------------------------------------------------------


def knee(
    score: np.ndarray,
    ref_c: int = 1,
    n_boot: int = 1000,
    block: np.ndarray | None = None,
    c_grid: Sequence[int] | None = None,
    seed: int = 0,
) -> tuple[int, dict[int, float], dict[int, float]]:
    """Post-hoc knee arbiter (B2): smallest capacity whose NEXT increment fails 2·SE.

    Implements §0b G1-G3:
      G1 bootstrap SE: SE by bootstrap (block-bootstrap over independent units if
        `block` is given, else plain i.i.d. row bootstrap), `B=n_boot`.
      G2 abstain: if even the FIRST increment (`ref_c` -> the next larger grid value)
        is insignificant, returns `r_star=0` — an explicit ABSTAIN sentinel, distinct
        from any real capacity in `c_grid`. Never read `r_star=0` as "capacity 0
        confirmed", and never read `r_star=ref_c` as a synonym for it either — this
        function reserves `ref_c` for a case where growth WAS established.
      G3 cap-saturation: if the knee search never finds an insignificant increment (every
        step up to `max(c_grid)` was significant), `r_star == max(c_grid)` and this
        function warns — an INVALID read; the caller must widen `c_grid` and rerun.

    `c_grid` defaults to consecutive integers `1..C` (`score.shape[1]`), matching the
    column convention of `score_table`'s stacked-columns tables.

    Args:
        score: `(N, C)` held-out log-likelihood table, columns in `c_grid` order.
        ref_c: reference (baseline) capacity; must be in `c_grid`.
        n_boot: bootstrap replicates.
        block: optional `(N,)` array of independent-unit ids for block bootstrap.
        c_grid: capacity value per column; defaults to `range(1, C + 1)`.
        seed: RNG seed for the bootstrap (reproducibility).

    Returns:
        `(r_star, delta_curve, se)`. `delta_curve` maps every c in `c_grid` to
        `mean_i(score[i,c] - score[i,ref_c])` — the FULL vector (G6): every scalar
        summary, including `r_star`, is derived from it, never the primary read. `se`
        maps every c beyond the first walked step to the bootstrap SE of the increment
        `Δ_c - Δ_{prev(c)}`.
    """
    s = np.asarray(score, dtype=np.float64)
    _n_rows, c_cols = s.shape
    grid = list(range(1, c_cols + 1)) if c_grid is None else list(c_grid)
    if len(grid) != c_cols:
        raise ValueError(f"c_grid length {len(grid)} != score column count {c_cols}")
    if ref_c not in grid:
        raise ValueError(f"ref_c={ref_c} not in c_grid={grid}")
    ref_idx = grid.index(ref_c)

    ref_mean = float(s[:, ref_idx].mean())
    delta_curve = {cv: float(s[:, j].mean() - ref_mean) for j, cv in enumerate(grid)}

    rng = np.random.default_rng(seed)
    boot_means = _bootstrap_col_means(s, n_boot, block, rng)  # (B, C), paired resample

    se: dict[int, float] = {}
    r_star = grid[-1]
    found = False
    for j in range(ref_idx, len(grid) - 1):
        inc_mean = float(s[:, j + 1].mean() - s[:, j].mean())
        inc_boot = boot_means[:, j + 1] - boot_means[:, j]
        inc_se = float(inc_boot.std(ddof=1))
        se[grid[j + 1]] = inc_se
        if inc_mean < 2.0 * inc_se:
            r_star = 0 if j == ref_idx else grid[j]
            found = True
            break

    if not found:
        warnings.warn(
            f"knee: G3 cap-saturation — every increment up to c_max={grid[-1]} was significant; "
            "widen c_grid and rerun (this read is INVALID as a knee).",
            stacklevel=2,
        )
        r_star = grid[-1]

    return r_star, delta_curve, se


# ---------------------------------------------------------------------------
# B5 — per-bin stacking (G4 cells) and the per-input curve (G5 locality guard).
# ---------------------------------------------------------------------------


def quantile_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Integer bin id (`0..n_bins-1`) per row of `x`, by quantile.

    Convenience for the G4/G5 conventions: terciles (`n_bins=3`) for the standard
    per-bin cell read, sextiles (`n_bins=6`) for the stability re-check.

    Args:
        x: `(N,)` observable to bin (e.g. the input coordinate).
        n_bins: number of equal-mass bins.

    Returns:
        `(N,)` int array of bin ids.
    """
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    edges = np.quantile(x_arr, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return np.clip(np.digitize(x_arr, edges) - 1, 0, n_bins - 1)


def perbin_stack(score: np.ndarray, bins: np.ndarray, *, tol: float = 1e-10, max_iter: int = 500) -> dict[Any, np.ndarray]:
    """Per-bin stacking (B5): runs `stack_em` independently within each bin (the G4 cell reader).

    Args:
        score: `(N, C)` held-out log-likelihood table.
        bins: `(N,)` bin label per row (any hashable), e.g. from `quantile_bins`.
        tol: forwarded to `stack_em`.
        max_iter: forwarded to `stack_em`.

    Returns:
        dict `bin_label -> (C,) pi_hat`, one independent stacking fit per bin. Cell
        disagreement with a monotone pattern across bins is the confound tell (§0b G4);
        report the full per-bin vectors, never a single collapsed summary.
    """
    bins_arr = np.asarray(bins)
    out: dict[Any, np.ndarray] = {}
    for b in np.unique(bins_arr):
        mask = bins_arr == b
        out[b] = stack_em(score[mask], tol=tol, max_iter=max_iter)
    return out


def perinput_curve(
    score: np.ndarray,
    x: np.ndarray,
    width: float,
    ref_c: int = 0,
    query_x: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    """Neighbour-averaged held-out advantage curve `Δ_c(x)`, with the G5 half-width re-read.

    `Δ_c(x)` for every capacity column c, relative to column `ref_c` (a POSITIONAL column
    index — this function does not carry a `c_grid`, per the K0 contract's literal 3-arg
    signature; default 0 = the smallest/first capacity column). Neighbourhood is a
    box-car of half-width `width` in units of `x`.

    G5 locality guard: also computes the same curve at `width / 2` and returns both, so a
    caller can check that a read is not a pooling artifact — a read that moves materially
    under shrinkage is manufactured multimodality, not a finding (§0b G5).

    Args:
        score: `(N, C)` held-out log-likelihood table.
        x: `(N,)` scalar input coordinate paired with each row of `score`.
        width: neighbourhood half-width (box-car) in units of `x`.
        ref_c: positional column index used as the reference capacity (`Δ = 0` there).
        query_x: points at which to evaluate the curve; defaults to `x` itself.

    Returns:
        dict with `query_x` `(Q,)`, `delta` `(Q, C)` at `width`, `delta_half` `(Q, C)` at
        `width / 2` (the G5 re-read), `width`, `width_half`, and `n_neighbors` `(Q,)` — the
        neighbourhood size used at `width` (diagnostic: thin neighbourhoods are noisy).
    """
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    s = np.asarray(score, dtype=np.float64)
    qx = x_arr if query_x is None else np.asarray(query_x, dtype=np.float64).ravel()
    delta_full = s - s[:, [ref_c]]

    def _sweep(w: float) -> tuple[np.ndarray, np.ndarray]:
        mask = (np.abs(x_arr[None, :] - qx[:, None]) <= w).astype(np.float64)  # (Q, N)
        counts = mask.sum(axis=1)
        summed = mask @ delta_full  # (Q, C)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = summed / counts[:, None]
        out[counts == 0] = np.nan
        return out, counts.astype(np.int64)

    delta, n_neighbors = _sweep(width)
    delta_half, _ = _sweep(width / 2.0)
    return {
        "query_x": qx,
        "delta": delta,
        "delta_half": delta_half,
        "width": width,
        "width_half": width / 2.0,
        "n_neighbors": n_neighbors,
    }


# ---------------------------------------------------------------------------
# Selftest — synthetic known-answer tables (N=4096, C=6). MUST pass before any real read.
# ---------------------------------------------------------------------------

_ST_N = 4096
_ST_C = 6
_ST_GRID = list(range(1, _ST_C + 1))


def _peaked_table(n: int, peak_c: int, gap: float, noise_sd: float, rng: np.random.Generator) -> np.ndarray:
    """A synthetic `(n, _ST_C)` table quadratically peaked at `peak_c` plus i.i.d. noise."""
    c = np.asarray(_ST_GRID, dtype=np.float64)
    base = -gap * (c - peak_c) ** 2
    return base[None, :] + rng.normal(0.0, noise_sd, size=(n, _ST_C))


def _check_i(rng: np.random.Generator) -> bool:
    """(i) all rows favor c=3 => pi_hat_3 > 0.8 and knee == 3."""
    score = _peaked_table(_ST_N, peak_c=3, gap=4.0, noise_sd=0.3, rng=rng)
    pi_hat = stack_em(score)
    r_star, delta_curve, se = knee(score, ref_c=1, n_boot=1000, c_grid=_ST_GRID, seed=1)
    ok_pi = pi_hat[_ST_GRID.index(3)] > 0.8
    ok_knee = r_star == 3
    dc_r = {k: round(v, 3) for k, v in delta_curve.items()}
    se_r = {k: round(v, 4) for k, v in se.items()}
    print(f"  (i) pi_hat={np.round(pi_hat, 3).tolist()}  pi_3={pi_hat[_ST_GRID.index(3)]:.3f} (>0.8: {ok_pi})")
    print(f"  (i) knee r_star={r_star} (==3: {ok_knee})  delta_curve={dc_r}  se={se_r}")
    print(f"  (i) {'PASS' if ok_pi and ok_knee else 'FAIL'}")
    return ok_pi and ok_knee


def _check_ii(rng: np.random.Generator) -> bool:
    """(ii) two-regime bait: half favor c=1, half favor c=4.

    Global pi_hat ~= {.5,.5} on {1,4}; per-bin recovers the split; per-bin held-out
    mixture score beats global by > 2*SE on a FRESH held-out draw of the same regimes.
    """
    half = _ST_N // 2
    x = np.concatenate([np.full(half, 0.25), np.full(half, 0.75)])

    def _make(seed_offset: int) -> tuple[np.ndarray, np.ndarray]:
        r = np.random.default_rng(1000 + seed_offset)
        top = _peaked_table(half, peak_c=1, gap=4.0, noise_sd=0.3, rng=r)
        bot = _peaked_table(_ST_N - half, peak_c=4, gap=4.0, noise_sd=0.3, rng=r)
        return x, np.concatenate([top, bot], axis=0)

    _, score = _make(0)
    pi_global = stack_em(score)
    ok_global = pi_global[_ST_GRID.index(1)] > 0.4 and pi_global[_ST_GRID.index(4)] > 0.4 and (pi_global[_ST_GRID.index(1)] + pi_global[_ST_GRID.index(4)]) > 0.9

    bins = quantile_bins(x, 2)
    pi_bins = perbin_stack(score, bins)
    b_low, b_high = sorted(pi_bins.keys())
    ok_bins = pi_bins[b_low][_ST_GRID.index(1)] > 0.8 and pi_bins[b_high][_ST_GRID.index(4)] > 0.8

    # held-out re-draw of the SAME regimes, scored under global vs per-bin pi_hat.
    _, score_ho = _make(1)
    bins_ho = quantile_bins(x, 2)
    global_ls = mixture_logscore(pi_global, score_ho)
    perbin_ls = np.empty(_ST_N)
    perbin_ls[bins_ho == b_low] = mixture_logscore(pi_bins[b_low], score_ho[bins_ho == b_low])
    perbin_ls[bins_ho == b_high] = mixture_logscore(pi_bins[b_high], score_ho[bins_ho == b_high])
    diff = perbin_ls - global_ls
    n_boot = 1000
    boot_idx = rng.integers(0, _ST_N, size=(n_boot, _ST_N))
    boot_mean_diff = diff[boot_idx].mean(axis=1)
    se_diff = float(boot_mean_diff.std(ddof=1))
    mean_diff = float(diff.mean())
    ok_beat = mean_diff > 2.0 * se_diff

    print(f"  (ii) global pi_hat={np.round(pi_global, 3).tolist()} (~.5/.5 on 1,4: {ok_global})")
    print(f"  (ii) per-bin pi_hat[{b_low}]={np.round(pi_bins[b_low], 3).tolist()}  pi_hat[{b_high}]={np.round(pi_bins[b_high], 3).tolist()} (split recovered: {ok_bins})")
    print(f"  (ii) per-bin - global held-out mixture score: {mean_diff:+.4f} nats, SE={se_diff:.4f} (> 2SE: {ok_beat})")
    print(f"  (ii) {'PASS' if ok_global and ok_bins and ok_beat else 'FAIL'}")
    return ok_global and ok_bins and ok_beat


def _check_iii(rng: np.random.Generator) -> bool:
    """(iii) flat table => knee abstains (G2, r_star=0) and pi_hat ~= uniform."""
    score = rng.normal(0.0, 1.0, size=(_ST_N, _ST_C))
    pi_hat = stack_em(score)
    r_star, delta_curve, se = knee(score, ref_c=1, n_boot=1000, c_grid=_ST_GRID, seed=2)
    ok_abstain = r_star == 0
    ok_uniform = bool(np.max(np.abs(pi_hat - 1.0 / _ST_C)) < 0.1)
    dc_r = {k: round(v, 4) for k, v in delta_curve.items()}
    se_r = {k: round(v, 4) for k, v in se.items()}
    print(f"  (iii) pi_hat={np.round(pi_hat, 3).tolist()}  max|pi-1/C|={np.max(np.abs(pi_hat - 1.0 / _ST_C)):.4f} (<0.1: {ok_uniform})")
    print(f"  (iii) knee r_star={r_star} (G2 abstain==0: {ok_abstain})  delta_curve={dc_r}  se={se_r}")
    print(f"  (iii) {'PASS' if ok_abstain and ok_uniform else 'FAIL'}")
    return ok_abstain and ok_uniform


def run_selftest() -> bool:
    """Runs all K0 selftest checks with a fixed RNG seed; prints PASS/FAIL per check."""
    rng = np.random.default_rng(0)
    print("capacity-ladder selftest (N=4096, C=6)")
    results = [
        ("all-rows-favor-c3", _check_i(rng)),
        ("two-regime-bait", _check_ii(rng)),
        ("flat-table-abstain", _check_iii(rng)),
    ]
    n_pass = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    all_pass = n_pass == len(results)
    print(f"all {len(results)} checks passed" if all_pass else f"{n_pass}/{len(results)} checks passed — FAILURES PRESENT")
    return all_pass


# ---------------------------------------------------------------------------
# Secondary: score tables from existing artifacts (SECONDARY — selftest must pass first).
# ---------------------------------------------------------------------------

_TOYS = ("C", "D", "E")
_SEEDS = (0, 1, 2)
_K_MAX = 6
_K_RANGE = list(range(1, _K_MAX + 1))
_N_TR, _N_TE = 1000, 2500  # de-risk tables (K1-K3): leaner than the June comparison to fit the shared machine; K4 is the definitive fit
_MDN_HIDDEN, _MDN_N_HIDDEN, _MDN_EPOCHS = 64, 2, 75
_AGG_HIDDEN, _AGG_EPOCHS, _AGG_ALPHA0 = 32, 75, 0.1


class _AggregateSparsityNested:
    """Adapts a fitted `AggregateSparsityKSelector` to the `NestedComponentsModel` protocol.

    Capacity c = number of (percentile-tiled) bins admitted, 1..k_max; the bypass
    component is excluded from the ladder (it represents "no structure", not a capacity
    level) — mirrors `classifier_raw_logits[:, :k_val]` masking in
    `probabilistic_regression_net.py:132-149`.
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    def nested_components(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)
        x_t = torch.as_tensor(x_arr)
        k_max = self._model.k_max
        with torch.no_grad():
            logits_full = self._model.weight_net(x_t)
            mu_full, lv_full = self._model.per_class_params(x_t)
        return (
            logits_full[:, :k_max].double().numpy(),
            mu_full[:, :k_max].double().numpy(),
            lv_full[:, :k_max].double().numpy(),
        )


def _make_toy(name: str, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    make_fn = {"C": td.make_toy_c, "D": td.make_toy_d, "E": td.make_toy_e}[name]
    return make_fn(n=n, seed=seed)


def _build_mdn_table(toy: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trains the fixed-k MDN sweep (k=1..6) and returns `(score_mat, x_te, y_te)`."""
    x_tr, y_tr = _make_toy(toy, _N_TR, seed)
    x_te, y_te = _make_toy(toy, _N_TE, seed + 500)
    models: dict[int, MixtureDensityNetwork] = {}
    for k in _K_RANGE:
        m = MixtureDensityNetwork(
            n_components=k, hidden_size=_MDN_HIDDEN, n_hidden=_MDN_N_HIDDEN, n_epochs=_MDN_EPOCHS,
            learning_rate=1e-2, random_seed=seed, calculate_feature_importance=False, optimize_hyperparameters=False,
        )
        m.fit(np.asarray(x_tr, dtype=np.float32), np.asarray(y_tr, dtype=np.float32).ravel())
        models[k] = m
    mat = score_table(models, x_te, y_te, _K_RANGE)
    return mat, x_te, y_te


def _build_aggregate_sparsity_table(toy: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trains `AggregateSparsityKSelector` ONCE (k_max=6) and returns `(score_mat, x_te, y_te)`."""
    x_tr, y_tr = _make_toy(toy, _N_TR, seed)
    x_te, y_te = _make_toy(toy, _N_TE, seed + 500)
    model = vemp.train_aggregate_sparsity(
        x_tr, y_tr, k_max=_K_MAX, alpha0=_AGG_ALPHA0, n_epochs=_AGG_EPOCHS, lr=1e-2,
        hidden=_AGG_HIDDEN, adaptive_bin_means=True, seed=seed,
    )
    adapter = _AggregateSparsityNested(model)
    mat = score_table(adapter, x_te, y_te, _K_RANGE)
    return mat, x_te, y_te


def build_tables() -> None:
    """Builds and saves the K0 score tables (fixed-k MDN sweep + AggregateSparsityKSelector).

    On toys C/D/E, seeds {0,1,2}, into `capacity_ladder_results/K0/`. Runs the selftest
    first and aborts if it fails.
    """
    if not run_selftest():
        raise RuntimeError("selftest failed — refusing to build tables (K0 contract: selftest must pass before any real read)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    # AggregateSparsity first: the faithful June surrogate (per-input reads for K2/K3/K5), only 9 fits — the more central de-risk lands before the 54-fit MDN sweep.
    builders = {"aggregate_sparsity": _build_aggregate_sparsity_table, "mdn_sweep": _build_mdn_table}
    for method, builder in builders.items():
        for toy in _TOYS:
            for seed in _SEEDS:
                t0 = time.time()
                mat, x_te, _y_te = builder(toy, seed)
                wall = time.time() - t0
                out_path = os.path.join(RESULTS_DIR, f"{method}_toy{toy}_seed{seed}.pt")
                torch.save(
                    {
                        "score_mat": torch.as_tensor(mat, dtype=torch.float64),
                        "x": torch.as_tensor(np.asarray(x_te, dtype=np.float64)),
                        "split": "test",
                        "c_grid": _K_RANGE,
                        "seed": seed,
                    },
                    out_path,
                )
                col_means = mat.mean(axis=0)
                print(f"[{method}] toy={toy} seed={seed}  wall={wall:.1f}s  shape={mat.shape}  col_means={np.round(col_means, 3).tolist()} -> {out_path}")


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="Run the K0 synthetic known-answer selftest and exit.")
    parser.add_argument("--build-tables", action="store_true", help="Build the K0 score tables from existing artifacts (runs --selftest first).")
    args = parser.parse_args()

    if args.selftest:
        ok = run_selftest()
        sys.exit(0 if ok else 1)
    if args.build_tables:
        build_tables()
        return
    parser.print_help()


if __name__ == "__main__":
    main()
