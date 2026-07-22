"""WSEL-13 — is the induced importance ordering real? (`docs/plans/capacity_programme/width.md:1089-1156`).

The certified width design's whole mechanistic account — and the argument that it ports to a
transformer (`docs/plans/capacity_programme/shared/width_transformer_port.md` SS1-SS2) — rests on one
property nobody has measured: because hidden unit `j` only receives gradient from widths `k >= j`,
importance should DECREASE with unit index. This driver trains the certified `SharedTrunkPerWidthHeadNet`
on the SS3.8 canonical toy suite (tier 1 + tier 2, per WSEL-13's ROOT ruling), then runs two non-circular
diagnostics on the frozen trunk. The two tiers train under DIFFERENT schedules and objectives (RE-
AUTHORIZED 2026-07-22, `width.md` SS3.8): tier 1 is k-dropout SANDWICH schedule, plain MSE loss
(`automl_package.examples.kdropout_converged_width_experiment._train_kdropout_to_convergence`, imported
not reimplemented); tier 2 is the ALL schedule (every width, every step) under the fixed-sigma weighted
squared error (SS3.7, `width_candidates.weighted_squared_error`), via this module's own
`_train_all_schedule_weighted_to_convergence` — every per-cell JSON records which (`training_schedule`,
`objective`) so a table mixing the two never hides it.

  Step 2 (single-unit ablation): zero one hidden unit at a time in the width-`w_max` head's input,
  read the MSE increase on the REPORT split. `importance_j` should decrease with `j`.

  Step 3 (prefix vs greedy, the non-circular test): the per-width heads were trained on PREFIX masks,
  so scoring an arbitrary unit subset with them would be circular. Instead a FRESH linear readout is
  re-fit by ordinary least squares (closed form, WITH intercept) on the frozen trunk's hidden features,
  for both the natural prefix order and a greedily-selected order. THREE SPLITS carved from the
  driver's held-out test set (never touched by gradient training), the same two-sequential-binary-split
  shape `automl_package/examples/probreg_p8.py:170-172` uses to carve its selection/report split:
  FIT (the OLS solve) -> SELECT (greedy picks its next unit here) -> REPORT (both curves are finally
  scored here, touched by neither fit nor selection) -- collapsing SELECT into REPORT would let greedy
  win by construction and make the secondary bar meaningless.

Pre-registered bars (fixed before any run; no re-run on failure, no bar edits after seeing numbers) are
read on TIER 1 only: primary, Spearman(index, importance) <= -0.5 on >= 2 of 3 seeds; secondary, mean
over k of (prefix_k - greedy_k) / greedy_k <= 0.10. Tier 2 (SS3.8's noisy-easy control) is computed and
reported as corroboration under a separate `tier2` key and may NOT be used to re-read, rescue, or
override the tier-1 bars. A FAIL is a finding, not a bug (see module-level ROOT note in the task brief).

Driver CLI contract (root-run grid; this file is never run over the full grid by its author):
  `--tier {1,2} --seed <int>` runs ONE cell, writing its per-cell JSON and `state_dict` immediately.
  `--summarize` aggregates every per-cell JSON on disk into `WSEL13/frozen.json`.
  `--selftest` runs a tiny w_max=2 toy (the true ordering known by construction: unit 0 is read by
  BOTH width-1 and width-2's mean heads, unit 1 only by width-2's) and asserts the diagnostic recovers
  it -- no real cell is ever run here.

Usage:
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel13.py --selftest
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel13.py --tier 1 --seed 0
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel13.py --tier 2 --seed 0
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel13.py --summarize
"""

from __future__ import annotations

import argparse
import enum
import importlib
import json
import math
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr
from sklearn.model_selection import train_test_split

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root

import converged_width_experiment as cwe  # noqa: E402
import convergence as cvg  # noqa: E402
import kdropout_converged_width_experiment as kce  # noqa: E402
import nested_width_net as nwn  # noqa: E402
import width_candidates as wc  # noqa: E402

from automl_package.utils.run_provenance import run_provenance  # noqa: E402

RESULTS_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "WSEL13")

# Certified configuration, reused verbatim from the k-dropout driver (WSEL-13 does not retune).
SEEDS = cwe.SEEDS  # (0, 1, 2)
W_MAX = cwe.W_MAX  # 12
ARCH = kce.Arch.SHARED_TRUNK  # the certified G-WIDTH arch (SharedTrunkPerWidthHeadNet)
LOSS = kce.LossType.MSE  # sigma fixed (never learned); SharedTrunkPerWidthHeadNet's log_var is a dummy zero anyway


SCHEDULE = nwn.WidthSchedule.SANDWICH
DEFAULT_MAX_EPOCHS = kce.DEFAULT_MAX_EPOCHS  # 200000; convergence gate decides the real stop
DEFAULT_CHECK_EVERY = cvg.DEFAULT_CHECK_EVERY
DEFAULT_PATIENCE = cvg.DEFAULT_PATIENCE
DEFAULT_MIN_DELTA = cvg.DEFAULT_MIN_DELTA

# Step-3 three-way split of the driver's held-out TEST set (never touched by gradient training), carved
# by two sequential train_test_split calls -- the same shape as probreg_p8.py:170-172's selection/report
# carve, extended by one more cut. FIT solves the OLS; SELECT picks greedy's next unit; REPORT scores.
_HELDOUT_FIT_FRACTION = 1.0 / 3.0  # FIT gets 1/3 of the held-out test set
_HELDOUT_SELECT_REPORT_SPLIT = 0.5  # the remaining 2/3 splits 50/50 into SELECT vs REPORT (each 1/3 overall)

_MIN_DENOM = 1e-12  # guards the relative-gap division; real (noisy) data never actually hits this
_PRIMARY_BAR_MIN_SEEDS_PASSING = 2  # pre-registered: Spearman <= threshold on >= 2 of 3 seeds


class Tier(enum.IntEnum):
    """SS3.8's canonical toy suite tiers this task runs (WSEL-13's row: tier 1 + tier 2, 6 runs total)."""

    ONE = 1  # the reference cell -- pre-registered bars are read HERE.
    TWO = 2  # the noisy-easy control -- corroboration only, never gates the bars (ROOT ruling in the brief).


@dataclass(frozen=True)
class _TierConfig:
    """One tier's toy/size config, SS3.8 lines 437-443."""

    toy: nwn.Toy
    n_train: int
    n_test: int
    sigma: float  # ignored by hetero3 downstream (nwn.make_hetero3 has no sigma arg); kept for uniform bookkeeping.


_TIER_CONFIG: dict[Tier, _TierConfig] = {
    Tier.ONE: _TierConfig(toy=nwn.Toy.HETERO, n_train=1500, n_test=500, sigma=nwn.HETERO_NOISE_SIGMA),
    Tier.TWO: _TierConfig(toy=nwn.Toy.HETERO3, n_train=2250, n_test=750, sigma=nwn.HETERO_NOISE_SIGMA),
}


def _assert_tier_objective_available(tier: Tier) -> None:
    """Refuse a tier whose sanctioned objective is not importable (ROOT ruling 2026-07-21; RE-AUTHORIZED 2026-07-22).

    SS3.7 fixes sigma at the generator's true value on EVERY tier, but the two tiers need DIFFERENT
    objectives to do it. Tier 1's `hetero` toy has one constant sigma, so plain MSE is exactly
    equivalent. Tier 2's `hetero3` toy has TWO sigmas, so its sanctioned objective is a fixed-sigma
    WEIGHTED squared error, `(pred - y)**2 / sigma_true(x)**2`, which down-weights the noisy region
    100x -- `width.md:377-379` states outright that this is NOT the same objective as plain MSE.

    That weighted loss is implemented ONCE, in `automl_package/examples/width_candidates.py`
    (`weighted_squared_error`), and imported (SS3.7, SS3.9 -- never re-derived per driver). It landed
    with WSEL-16's authoring and WSEL-13 tier 2 was separately RE-AUTHORIZED 2026-07-22
    (`width.md` SS3.8's RESOLUTION UPDATE / RE-AUTHORIZED paragraphs) -- so this guard no longer
    refuses tier 2 unconditionally. It still refuses MECHANICALLY: tier 2 proceeds only if
    `weighted_squared_error` actually imports right now, so a future revert or breakage of
    `width_candidates.py` fails loudly here rather than silently falling through to plain MSE. `run_cell`
    itself has no plain-MSE branch for tier 2 to fall into -- see `_train_all_schedule_weighted_to_convergence`,
    which is the ONLY trainer tier 2 ever calls, hardcoded to the weighted objective with no toggle.

    This raises rather than silently running tier 2 under plain MSE. Such a cell would be measured on
    an objective the plan explicitly calls incomparable to what the noisy-easy control exists to
    measure -- and it would look exactly like a valid result. That failure mode is what voided
    WSEL-11's first run.
    """
    if tier is Tier.ONE:
        return
    try:
        module = importlib.import_module("automl_package.examples.width_candidates")
        if not hasattr(module, "weighted_squared_error"):
            raise ImportError("width_candidates imports but has no weighted_squared_error attribute")
    except ImportError as exc:
        raise SystemExit(
            f"WSEL-13 tier {tier.value} is BLOCKED: its sanctioned objective, `weighted_squared_error`, "
            f"failed to import from automl_package.examples.width_candidates ({exc}). Running it under "
            "plain MSE would silently measure the wrong objective. Fix the import before running tier 2."
        ) from exc


_HETERO3_NOISY_REGION = 2  # nwn.make_hetero3's region id for the noisy-easy tail (HETERO3_NOISY_SIGMA).


def _sigma_true_tensor(toy: nwn.Toy, region: np.ndarray, sigma: float, norm: dict) -> torch.Tensor:
    """Per-point TRUE noise sigma (`width.md` SS3.7), IN STANDARDIZED-y UNITS (`/ norm["sy"]`) -- never estimated, never learned.

    `hetero`: `sigma` (the toy's own constant) everywhere -- both regions share it. `hetero3`: region 2
    (noisy-easy) is `HETERO3_NOISY_SIGMA`, regions 0/1 are `HETERO_NOISE_SIGMA` -- read from the
    generator's own `region` output, never re-derived by thresholding `x`. A pure linear rescale of `y`
    (`y_std = (y - my) / sy`) rescales its noise std by the same factor `sy`, so `sigma_true` must be
    divided by it too or the weighted loss trains on a mismatched scale.

    Mirrors `width_wsel16._sigma_true_tensor` (`automl_package/examples/width_wsel16.py:369`) exactly,
    reimplemented locally rather than imported: `width_wsel16.py` imports THIS module
    (`import width_wsel13 as w13`, for the OLS/greedy ordering helpers), so importing back would be
    circular.
    """
    if toy is nwn.Toy.HETERO3:
        sigma_raw = np.where(region == _HETERO3_NOISY_REGION, nwn.HETERO3_NOISY_SIGMA, nwn.HETERO_NOISE_SIGMA)
    else:
        sigma_raw = np.full(region.shape, sigma, dtype=np.float64)
    return torch.as_tensor(sigma_raw / norm["sy"], dtype=torch.float32)


def _train_all_schedule_weighted_to_convergence(
    net: nwn.SharedTrunkPerWidthHeadNet,
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    sigma_tr: torch.Tensor,
    sigma_val: torch.Tensor,
    *,
    w_max: int,
    max_epochs: int,
    check_every: int,
    patience: int,
    min_delta: float,
) -> dict[int, cvg.ConvergenceResult]:
    """Tier-2-ONLY trainer: `WidthSchedule.ALL`, every width every step, under the SS3.7 fixed-sigma weighted squared error; gated by PER-WIDTH held-out convergence.

    Tier 1 keeps training through `kce._train_kdropout_to_convergence` under the certified SANDWICH
    schedule, unchanged (see `run_cell`). Tier 2's RE-AUTHORIZED spec (`width.md` SS3.8 "RE-AUTHORIZED
    2026-07-22") trains under ALL instead, and `kce._train_kdropout_to_convergence` cannot express the
    weighted objective at all (its `LossType` is `{NLL, MSE}` only, and
    `kdropout_converged_width_experiment.py` is out of this task's write set to extend) -- so this
    reproduces that function's own ALL-schedule branch (deterministic `range(1, w_max + 1)`, no RNG
    draw, `kdropout_converged_width_experiment.py:293-295`) with the weighted loss substituted in,
    using `net.all_widths_forward` for a single shared-trunk evaluation per step (the WSEL-12
    vectorisation `SharedTrunkPerWidthHeadNet` already exposes, rather than `w_max` separate
    `forward_width` calls). Checkpointing follows the SAME whole-net-at-best-mean-per-width-val rule
    `kce._train_kdropout_to_convergence` uses for `Arch.SHARED_TRUNK` (not per-width independent
    restoration -- this class has one shared trunk, restoring one width's best state would clobber
    another's).

    There is deliberately no `loss` or `total_loss_fn` parameter: tier 2 always trains on
    `wc.weighted_squared_error`, never plain squared error -- no branch here could silently fall back
    to it (the purpose `_assert_tier_objective_available` exists to guard).

    Args:
        net: a `SharedTrunkPerWidthHeadNet` to train in place.
        x_tr: standardized training inputs, shape `(N, 1)`.
        y_tr: standardized training targets, shape `(N,)`.
        x_val: standardized held-out inputs used only for convergence monitoring.
        y_val: standardized held-out targets used only for convergence monitoring.
        sigma_tr: per-training-point true noise sigma (standardized-y units), same length as `y_tr`.
        sigma_val: per-val-point true noise sigma (standardized-y units), same length as `y_val`.
        w_max: maximum hidden width (the largest prefix the net can express).
        max_epochs: safety cap on optimizer steps (== epochs; full-batch).
        check_every: epochs between per-width held-out checkpoints.
        patience: consecutive flat checkpoints that declare one width converged.
        min_delta: held-out-loss decrease counted as a real improvement.

    Returns:
        `{width -> ConvergenceResult}`, best weights already restored.
    """
    opt = torch.optim.Adam(net.parameters(), lr=cwe.LR)
    trackers = {k: cvg.ConvergenceTracker(patience=patience, min_delta=min_delta) for k in range(1, w_max + 1)}
    best_mean_val = math.inf
    best_net_state: dict | None = None

    net.train()
    final_epoch = max_epochs
    for epoch in range(1, max_epochs + 1):
        opt.zero_grad()
        mean_all, _logvar_all = net.all_widths_forward(x_tr)  # (N, w_max); ALL schedule -- every width, every step.
        total_loss = torch.zeros((), device=x_tr.device)
        for k in range(1, w_max + 1):
            total_loss = total_loss + wc.weighted_squared_error(mean_all[:, k - 1], y_tr, sigma_tr)
        total_loss.backward()
        opt.step()

        if epoch % check_every == 0:
            net.eval()
            with torch.no_grad():
                mean_val_all, _ = net.all_widths_forward(x_val)
                per_width_val = {k: float(wc.weighted_squared_error(mean_val_all[:, k - 1], y_val, sigma_val).item()) for k in range(1, w_max + 1)}
            for k, v in per_width_val.items():
                trackers[k].update(epoch, v)
            mean_val = sum(per_width_val.values()) / w_max
            if mean_val < best_mean_val:
                best_mean_val = mean_val
                best_net_state = {n: t.detach().clone() for n, t in net.state_dict().items()}
            net.train()
            if all(t.done for t in trackers.values()):
                final_epoch = epoch
                break

    if best_net_state is not None:
        net.load_state_dict(best_net_state)
    net.eval()
    return {k: trackers[k].result(final_epoch=final_epoch) for k in range(1, w_max + 1)}


def _fit_ols(h: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed-form OLS WITH intercept: `h` is `(N, |subset|)`, `y` is `(N,)`. Returns `beta` (intercept first)."""
    design = np.concatenate([np.ones((h.shape[0], 1)), h], axis=1)
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return beta


def _ols_mse(h: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
    """MSE of the fitted `beta` (from `_fit_ols`) on a (possibly different) split `(h, y)`."""
    design = np.concatenate([np.ones((h.shape[0], 1)), h], axis=1)
    pred = design @ beta
    return float(np.mean((pred - y) ** 2))


def _greedy_forward_selection(h_fit: np.ndarray, y_fit: np.ndarray, h_select: np.ndarray, y_select: np.ndarray, w_max: int) -> tuple[list[int], list[float]]:
    """Greedy forward selection: fits candidates on FIT, picks the next unit by lowest SELECT MSE.

    Returns `(order, select_mse_by_step)`, both length `w_max`; `order` is 0-indexed unit ids in pick
    order, `select_mse_by_step[i]` is the SELECT-split MSE of the subset `order[:i+1]` that won step `i`.
    """
    remaining = list(range(w_max))
    order: list[int] = []
    select_mse_by_step: list[float] = []
    for _ in range(w_max):
        best_unit, best_mse = -1, math.inf
        for cand in remaining:
            cols = [*order, cand]
            beta = _fit_ols(h_fit[:, cols], y_fit)
            mse = _ols_mse(h_select[:, cols], y_select, beta)
            if mse < best_mse:
                best_unit, best_mse = cand, mse
        order.append(best_unit)
        remaining.remove(best_unit)
        select_mse_by_step.append(best_mse)
    return order, select_mse_by_step


def run_cell(
    tier: Tier,
    seed: int,
    *,
    w_max: int = W_MAX,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    check_every: int = DEFAULT_CHECK_EVERY,
    patience: int = DEFAULT_PATIENCE,
    min_delta: float = DEFAULT_MIN_DELTA,
    schedule: nwn.WidthSchedule = SCHEDULE,
) -> tuple[dict, dict]:
    """Runs one (tier, seed) cell: trains to convergence, then Step 2 + Step 3. Returns `(case, state_dict)`.

    Data prep mirrors `kdropout_converged_width_experiment.run_case` verbatim for `arch=SHARED_TRUNK`
    (the only architecture this task runs) -- reimplemented here rather than calling `run_case` because
    this task needs the trained net object and the held-out test set directly (for the ablation/OLS
    diagnostics), neither of which `run_case` returns.

    Tier 1 (`loss=MSE`, `schedule=SANDWICH` by default): the TRAINING LOOP itself is never
    reimplemented -- `_train_kdropout_to_convergence` is imported and called as-is, `schedule`
    overridable for the nested-vs-sandwich ablation this task's `--schedule` flag exists for.

    Tier 2 (RE-AUTHORIZED `width.md` SS3.8): trains under `WidthSchedule.ALL` on the fixed-sigma
    weighted objective (SS3.7) -- neither expressible through `_train_kdropout_to_convergence`
    (`LossType` has no weighted member), so `_train_all_schedule_weighted_to_convergence` (this module)
    is called instead. `schedule` is IGNORED for tier 2 -- ALL is mandatory, not a choice (see `main`).
    """
    cfg = _TIER_CONFIG[tier]
    if cfg.toy is nwn.Toy.HETERO3:
        x_tr, y_tr, region_tr = nwn.make_hetero3(cfg.n_train, seed)
        x_te, y_te, _reg_te = nwn.make_hetero3(cfg.n_test, seed + 500)
    else:
        x_tr, y_tr, region_tr = nwn.make_hetero(cfg.n_train, seed, sigma=cfg.sigma)
        x_te, y_te, _reg_te = nwn.make_hetero(cfg.n_test, seed + 500, sigma=cfg.sigma)

    # Same phase-1 train/val carve as the k-dropout battery (rest = train, every VAL_EVERY-th = val).
    p1_idx = np.arange(0, cfg.n_train, 2)
    x_p1, y_p1, region_p1 = x_tr[p1_idx], y_tr[p1_idx], region_tr[p1_idx]
    val_mask = (np.arange(len(x_p1)) % cwe.VAL_EVERY) == 0
    norm = cwe._standardize_fit(x_p1[~val_mask], y_p1[~val_mask])
    x_tr_t, y_tr_t = cwe._to_std_tensors(x_p1[~val_mask], y_p1[~val_mask], norm)
    x_val_t, y_val_t = cwe._to_std_tensors(x_p1[val_mask], y_p1[val_mask], norm)

    torch.manual_seed(seed)
    net = nwn.SharedTrunkPerWidthHeadNet(w_max=w_max)
    if tier is Tier.TWO:
        effective_schedule = nwn.WidthSchedule.ALL
        objective = "weighted_squared_error"
        sigma_tr = _sigma_true_tensor(cfg.toy, region_p1[~val_mask], cfg.sigma, norm)
        sigma_val = _sigma_true_tensor(cfg.toy, region_p1[val_mask], cfg.sigma, norm)
        conv = _train_all_schedule_weighted_to_convergence(
            net,
            x_tr_t,
            y_tr_t,
            x_val_t,
            y_val_t,
            sigma_tr,
            sigma_val,
            w_max=w_max,
            max_epochs=max_epochs,
            check_every=check_every,
            patience=patience,
            min_delta=min_delta,
        )
    else:
        effective_schedule = schedule
        objective = LOSS.value  # "mse" -- exactly equivalent to the fixed-sigma likelihood on tier 1's constant sigma (SS3.7).
        conv, _best_mean_val_epoch = kce._train_kdropout_to_convergence(
            net,
            x_tr_t,
            y_tr_t,
            x_val_t,
            y_val_t,
            arch=ARCH,
            loss=LOSS,
            max_epochs=max_epochs,
            check_every=check_every,
            patience=patience,
            min_delta=min_delta,
            seed=seed,
            schedule=schedule,
        )
    n_trustworthy = sum(1 for r in conv.values() if r.trustworthy)
    all_trustworthy = n_trustworthy == w_max

    # Held-out test set, standardized on TRAIN stats -- never touched by gradient training. Splitting
    # first (on the raw arrays, like probreg_p8.py:170-172) then standardizing each split with the SAME
    # `norm` is equivalent to standardizing-then-splitting (an affine map commutes with a row subset).
    x_rem, x_fit_raw, y_rem, y_fit_raw = train_test_split(x_te, y_te, test_size=_HELDOUT_FIT_FRACTION, random_state=seed)
    x_select_raw, x_report_raw, y_select_raw, y_report_raw = train_test_split(x_rem, y_rem, test_size=_HELDOUT_SELECT_REPORT_SPLIT, random_state=seed)
    x_fit_t, y_fit_t = cwe._to_std_tensors(x_fit_raw, y_fit_raw, norm)
    x_select_t, y_select_t = cwe._to_std_tensors(x_select_raw, y_select_raw, norm)
    x_report_t, y_report_t = cwe._to_std_tensors(x_report_raw, y_report_raw, norm)

    net.eval()
    with torch.no_grad():
        h_fit = net.hidden(x_fit_t)
        h_select = net.hidden(x_select_t)
        h_report = net.hidden(x_report_t)

        # Step 2 -- single-unit ablation, widest head only (mask is all-ones at k=w_max, so
        # `mean_heads[w_max - 1]` applied to the full hidden vector IS `forward_width(x, w_max)`).
        widest_head = net.mean_heads[w_max - 1]
        baseline_pred = widest_head(h_report).squeeze(1)
        baseline_mse = float(torch.mean((baseline_pred - y_report_t) ** 2).item())
        importance_by_unit: dict[int, float] = {}
        for j in range(1, w_max + 1):
            h_ablated = h_report.clone()
            h_ablated[:, j - 1] = 0.0
            pred_j = widest_head(h_ablated).squeeze(1)
            mse_j = float(torch.mean((pred_j - y_report_t) ** 2).item())
            importance_by_unit[j] = mse_j - baseline_mse

    spearman_res = spearmanr(list(range(1, w_max + 1)), [importance_by_unit[j] for j in range(1, w_max + 1)])

    h_fit_np, y_fit_np = h_fit.numpy(), y_fit_t.numpy()
    h_select_np, y_select_np = h_select.numpy(), y_select_t.numpy()
    h_report_np, y_report_np = h_report.numpy(), y_report_t.numpy()

    # Step 3 -- prefix vs greedy, re-fit by OLS (FIT split), greedy picks on SELECT, both scored on REPORT.
    prefix_report_mse: dict[int, float] = {}
    for k in range(1, w_max + 1):
        cols = list(range(k))
        beta = _fit_ols(h_fit_np[:, cols], y_fit_np)
        prefix_report_mse[k] = _ols_mse(h_report_np[:, cols], y_report_np, beta)

    greedy_order, greedy_select_mse_by_step = _greedy_forward_selection(h_fit_np, y_fit_np, h_select_np, y_select_np, w_max)
    greedy_report_mse: dict[int, float] = {}
    for k in range(1, w_max + 1):
        cols = greedy_order[:k]
        beta = _fit_ols(h_fit_np[:, cols], y_fit_np)
        greedy_report_mse[k] = _ols_mse(h_report_np[:, cols], y_report_np, beta)

    kendall_res = kendalltau(greedy_order, list(range(w_max)))
    relative_gaps = [(prefix_report_mse[k] - greedy_report_mse[k]) / max(greedy_report_mse[k], _MIN_DENOM) for k in range(1, w_max + 1)]
    mean_relative_prefix_gap = float(np.mean(relative_gaps))

    case = {
        "tier": tier.value,
        "seed": seed,
        "toy": cfg.toy.value,
        "n_train": cfg.n_train,
        "n_test": cfg.n_test,
        "sigma": cfg.sigma if cfg.toy is nwn.Toy.HETERO else None,
        "w_max": w_max,
        "convergence": {k: r.summary() for k, r in conv.items()},
        "n_widths_trustworthy": n_trustworthy,
        "all_widths_trustworthy": all_trustworthy,
        "splits": {"n_fit": len(x_fit_raw), "n_select": len(x_select_raw), "n_report": len(x_report_raw)},
        "step2_ablation": {
            "baseline_report_mse": baseline_mse,
            "importance_by_unit": {str(j): v for j, v in importance_by_unit.items()},
            "spearman_index_vs_importance": {"rho": float(spearman_res.statistic), "p": float(spearman_res.pvalue)},
        },
        "step3_prefix_vs_greedy": {
            "prefix_report_mse_by_k": {str(k): v for k, v in prefix_report_mse.items()},
            "greedy_report_mse_by_k": {str(k): v for k, v in greedy_report_mse.items()},
            "greedy_select_mse_by_step": greedy_select_mse_by_step,
            "greedy_order_0indexed": greedy_order,
            "kendall_tau_greedy_vs_index": {"tau": float(kendall_res.statistic), "p": float(kendall_res.pvalue)},
            "mean_relative_prefix_gap": mean_relative_prefix_gap,
        },
        "training_schedule": effective_schedule.value,  # tier 2 is always "all" (RE-AUTHORIZED, width.md SS3.8); tier 1 reads --schedule.
        "objective": objective,  # "mse" (tier 1) or "weighted_squared_error" (tier 2) -- machine-readable per the RE-AUTHORIZED labelling requirement.
        "provenance": run_provenance(),
    }
    return case, {name: t.detach().clone() for name, t in net.state_dict().items()}


def _schedule_suffix(schedule: nwn.WidthSchedule) -> str:
    """Filename suffix per training schedule.

    The certified schedule keeps the ORIGINAL, suffix-free filenames on purpose: those exact paths are
    cited in `width.md` and in `shared/width_transformer_port.md`, and the plan's citation gate checks
    that cited paths resolve. Renaming them to be symmetric would break both. The alternative schedule
    is additive.
    """
    return "" if schedule is SCHEDULE else f"_{schedule.value}"


def _canonical_schedule_for_tier(tier: Tier) -> nwn.WidthSchedule:
    """Which schedule a CANONICAL cell of this tier was (or will be) trained under -- for filenames.

    Tier 1 trains under the certified SANDWICH schedule by default (`SCHEDULE`), overridable via
    `--schedule` for the nested-vs-sandwich ablation this task also runs -- callers wanting a specific
    tier-1 alternate-schedule file pass their own `schedule` to `_cell_json_path`/`_state_path` directly,
    bypassing this helper. Tier 2 is fixed at ALL by the RE-AUTHORIZED ruling (`width.md` SS3.8) -- not
    a user choice -- so its canonical (and only) on-disk schedule is always ALL, regardless of what
    `--schedule` defaults to.
    """
    return nwn.WidthSchedule.ALL if tier is Tier.TWO else SCHEDULE


def _cell_json_path(tier: Tier, seed: int, schedule: nwn.WidthSchedule = SCHEDULE) -> str:
    return os.path.join(RESULTS_DIR, f"wsel13_tier{tier.value}{_schedule_suffix(schedule)}_seed{seed}.json")


def _state_path(tier: Tier, seed: int, schedule: nwn.WidthSchedule = SCHEDULE) -> str:
    return os.path.join(RESULTS_DIR, f"state_tier{tier.value}{_schedule_suffix(schedule)}_seed{seed}.pt")


def _jsonable(obj: object) -> object:
    """Minimal JSON-safe coercion (numpy/torch scalars, dict int-keys) -- local twin of `sinc_width_experiment._jsonable`."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj


def run_selftest() -> bool:
    """Tiny w_max=2 toy with the true ordering known by construction; no real cell is run here.

    Unit 0's readout weight into `mean_heads[1]` (the width-2 head) is trained by BOTH the width-1
    step (where unit 0 is the only unmasked node) and the width-2 step; unit 1's is trained ONLY by the
    width-2 step (masked out at width 1). So unit 0 should come out strictly more important than unit 1,
    and greedy forward selection (a single binary choice at w_max=2) should pick unit 0 first.
    """
    w_max = 2
    torch.manual_seed(0)
    case, _state = run_cell(Tier.ONE, seed=0, w_max=w_max, max_epochs=3000, check_every=100, patience=3, min_delta=2e-3)

    conv_ok = all(len(case["convergence"][k]["trajectory"]) >= 1 for k in (1, 2)) and all(isinstance(case["convergence"][k]["converged"], bool) for k in (1, 2))
    print(f"[wsel13 selftest] convergence trajectories recorded, flags present: {conv_ok}  {'PASS' if conv_ok else 'FAIL'}")

    rho = case["step2_ablation"]["spearman_index_vs_importance"]["rho"]
    importance = case["step2_ablation"]["importance_by_unit"]
    ordering_ok = rho < 0.0 and importance["1"] > importance["2"]  # unit index 1 (0-indexed unit 0) more important
    print(f"[wsel13 selftest] (step2) importance decreases with index: unit1={importance['1']:.4g} unit2={importance['2']:.4g} rho={rho:.3g}  {'PASS' if ordering_ok else 'FAIL'}")

    greedy_order = case["step3_prefix_vs_greedy"]["greedy_order_0indexed"]
    tau = case["step3_prefix_vs_greedy"]["kendall_tau_greedy_vs_index"]["tau"]
    greedy_ok = greedy_order[0] == 0 and tau == 1.0
    print(f"[wsel13 selftest] (step3) greedy picks unit 0 first, order matches index: order={greedy_order} tau={tau:.3g}  {'PASS' if greedy_ok else 'FAIL'}")

    ok = conv_ok and ordering_ok and greedy_ok

    # Tier-2 wiring (RE-AUTHORIZED, width.md SS3.8): the weighted objective actually engages, and the
    # ALL schedule -- not sandwich -- is what tier-2 training sees. No real cell is run beyond the tiny
    # w_max=2 toy below.
    #
    # (a) formula engagement: two points with an IDENTICAL residual (1.0) but different true sigma --
    # `weighted_squared_error` must differ from plain MSE by EXACTLY `mean(1/sigma**2)` (the down-
    # weighting factor SS3.7 exists to apply), proving this module calls the real formula, not a copy.
    pred = torch.tensor([2.0, 2.0])
    y_target = torch.tensor([1.0, 1.0])  # residual = 1.0 at both points
    sigma_quiet, sigma_noisy = nwn.HETERO_NOISE_SIGMA, nwn.HETERO3_NOISY_SIGMA
    sigma_true = torch.tensor([sigma_quiet, sigma_noisy])
    weighted = wc.weighted_squared_error(pred, y_target, sigma_true)
    plain = ((pred - y_target) ** 2).mean()
    expected_ratio = float(torch.mean(1.0 / sigma_true**2))
    actual_ratio = float(weighted / plain)
    engagement_ok = math.isclose(actual_ratio, expected_ratio, rel_tol=1e-5)
    print(f"[wsel13 selftest] (tier2 wiring a) weighted/plain={actual_ratio:.6g} expected={expected_ratio:.6g}  {'PASS' if engagement_ok else 'FAIL'}")

    # (b) a real (tiny) tier-2 cell records ALL as its training schedule and the weighted objective by
    # name -- proves the cell that actually trains sees both, not just that the formula above works.
    torch.manual_seed(0)
    case_tier2, _state_tier2 = run_cell(Tier.TWO, seed=0, w_max=w_max, max_epochs=3000, check_every=100, patience=3, min_delta=2e-3)
    conv_ok_tier2 = all(len(case_tier2["convergence"][k]["trajectory"]) >= 1 for k in (1, 2))
    schedule_ok = case_tier2["training_schedule"] == nwn.WidthSchedule.ALL.value
    objective_ok = case_tier2["objective"] == "weighted_squared_error"
    wiring_ok = conv_ok_tier2 and schedule_ok and objective_ok
    print(
        f"[wsel13 selftest] (tier2 wiring b) schedule={case_tier2['training_schedule']!r} "
        f"objective={case_tier2['objective']!r} convergence recorded={conv_ok_tier2}  {'PASS' if wiring_ok else 'FAIL'}"
    )

    ok = ok and engagement_ok and wiring_ok
    print(f"[wsel13 selftest] {'PASS' if ok else 'FAIL'}")
    return ok


def summarize() -> None:
    """Aggregates every per-cell JSON on disk into `WSEL13/frozen.json`. Does not train anything."""
    per_tier: dict[Tier, dict[int, dict]] = {Tier.ONE: {}, Tier.TWO: {}}
    for tier in Tier:
        for seed in SEEDS:
            path = _cell_json_path(tier, seed, _canonical_schedule_for_tier(tier))
            if os.path.exists(path):
                with open(path) as f:
                    per_tier[tier][seed] = json.load(f)

    def _tier_block(cells: dict[int, dict]) -> dict:
        spearman = {str(s): c["step2_ablation"]["spearman_index_vs_importance"]["rho"] for s, c in cells.items()}
        kendall = {str(s): c["step3_prefix_vs_greedy"]["kendall_tau_greedy_vs_index"]["tau"] for s, c in cells.items()}
        gaps_per_seed = {str(s): c["step3_prefix_vs_greedy"]["mean_relative_prefix_gap"] for s, c in cells.items()}
        greedy_order = {str(s): c["step3_prefix_vs_greedy"]["greedy_order_0indexed"] for s, c in cells.items()}
        per_k_mse = {
            str(s): {"prefix": c["step3_prefix_vs_greedy"]["prefix_report_mse_by_k"], "greedy": c["step3_prefix_vs_greedy"]["greedy_report_mse_by_k"]}
            for s, c in cells.items()
        }
        return {
            "spearman_index_vs_importance": spearman,
            "kendall_tau_greedy_vs_index": kendall,
            # ASSUMPTION (flagged in the report): `mean_relative_prefix_gap` is not marked "(per seed)" in
            # the spec, unlike the other two fields, so the top-level scalar here is the mean ACROSS seeds
            # of each seed's own mean-over-k gap; the per-seed breakdown is kept alongside so the root can
            # re-read the bar either way.
            "mean_relative_prefix_gap": float(np.mean(list(gaps_per_seed.values()))) if gaps_per_seed else None,
            "mean_relative_prefix_gap_per_seed": gaps_per_seed,
            "greedy_selection_order": greedy_order,
            "per_k_report_mse": per_k_mse,
        }

    tier1_block = _tier_block(per_tier[Tier.ONE])
    tier2_block = _tier_block(per_tier[Tier.TWO])

    primary_threshold = -0.5
    secondary_threshold = 0.10
    n_seeds_tier1 = len(per_tier[Tier.ONE])
    n_primary_pass = sum(1 for rho in tier1_block["spearman_index_vs_importance"].values() if rho <= primary_threshold)
    primary_pass = n_primary_pass >= _PRIMARY_BAR_MIN_SEEDS_PASSING and n_seeds_tier1 >= _PRIMARY_BAR_MIN_SEEDS_PASSING
    secondary_value = tier1_block["mean_relative_prefix_gap"]
    secondary_pass = secondary_value is not None and secondary_value <= secondary_threshold
    ordering_holds = bool(primary_pass and secondary_pass)

    primary_bar = {
        "threshold": primary_threshold,
        "rule": "Spearman(index, importance) <= threshold on >= 2 of 3 seeds",
        "n_seeds_passing": n_primary_pass,
        "n_seeds_present": n_seeds_tier1,
        "pass": primary_pass,
    }
    secondary_bar = {
        "threshold": secondary_threshold,
        "rule": "mean over k of (prefix_k - greedy_k) / greedy_k <= threshold",
        "value": secondary_value,
        "pass": secondary_pass,
    }
    tier2_note = (
        "corroboration only -- computed per SS3.8's row for WSEL-13, but per the ROOT ruling in the "
        "task brief it may NOT be used to re-read, rescue, or override the tier-1 bars. No "
        "ordering_holds/bars are computed here on purpose."
    )
    config = {
        "arch": ARCH.value,
        "loss": LOSS.value,
        "schedule": f"kdropout_{SCHEDULE.value}",
        "w_max": W_MAX,
        "seeds": list(SEEDS),
        "tier1": {
            "toy": _TIER_CONFIG[Tier.ONE].toy.value,
            "n_train": _TIER_CONFIG[Tier.ONE].n_train,
            "n_test": _TIER_CONFIG[Tier.ONE].n_test,
            "sigma": _TIER_CONFIG[Tier.ONE].sigma,
            "schedule": SCHEDULE.value,
            "objective": LOSS.value,
        },
        "tier2": {
            "toy": _TIER_CONFIG[Tier.TWO].toy.value,
            "n_train": _TIER_CONFIG[Tier.TWO].n_train,
            "n_test": _TIER_CONFIG[Tier.TWO].n_test,
            # RE-AUTHORIZED 2026-07-22 (width.md SS3.8): tier 2 trains under ALL, not the tier-1 SANDWICH
            # default -- named explicitly here since this config block tabulates both tiers together.
            "schedule": nwn.WidthSchedule.ALL.value,
            "objective": "weighted_squared_error",
        },
        "heldout_fit_fraction": _HELDOUT_FIT_FRACTION,
        "heldout_select_report_split": _HELDOUT_SELECT_REPORT_SPLIT,
        "max_epochs_cap": DEFAULT_MAX_EPOCHS,
        "check_every": DEFAULT_CHECK_EVERY,
        "patience": DEFAULT_PATIENCE,
        "min_delta": DEFAULT_MIN_DELTA,
    }
    frozen = {
        **tier1_block,
        "primary_bar": primary_bar,
        "secondary_bar": secondary_bar,
        "ordering_holds": ordering_holds,
        "tier2": {**tier2_block, "note": tier2_note},
        "config": config,
        "n_cells_present": {"tier1": n_seeds_tier1, "tier2": len(per_tier[Tier.TWO])},
        "provenance": run_provenance(),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "frozen.json")
    with open(path, "w") as f:
        json.dump(_jsonable(frozen), f, indent=2)
    print(f"wrote {path}")
    print(
        f"ordering_holds={ordering_holds} (primary: {n_primary_pass}/{n_seeds_tier1} seeds <= {primary_threshold}; "
        f"secondary: {secondary_value} <= {secondary_threshold} -> {secondary_pass})"
    )


def main() -> None:
    """Parses args and dispatches to `--selftest` / `--summarize` / one real `--tier`/`--seed` cell."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="Tiny w_max=2 known-answer wiring check, then exit.")
    parser.add_argument("--summarize", action="store_true", help="Aggregate every per-cell JSON on disk into WSEL13/frozen.json.")
    parser.add_argument("--tier", type=int, choices=[t.value for t in Tier], default=None, help="SS3.8 tier this cell runs (1 = reference, 2 = noisy-easy control).")
    parser.add_argument(
        "--schedule", type=str, choices=[s.value for s in nwn.WidthSchedule], default=SCHEDULE.value,
        help=(
            "TIER 1 ONLY -- which width-training schedule to train under. "
            "'sandwich' (default, the CERTIFIED one) trains the narrowest AND the widest every step plus 2 random "
            "middles -- so the widest trains every step and the LAST unit is never starved. "
            "'nested' draws ONE width per example per pass, so the last unit trains ~1/w_max of the time. "
            "The decreasing-importance prediction is derived for the 'nested' case; running both is what separates "
            "'the mechanism does not exist' from 'our schedule trains the asymmetry away'. "
            "Not selectable for tier 2, which is fixed at 'all' by the RE-AUTHORIZED ruling (width.md SS3.8) -- "
            "passing anything but the default alongside --tier 2 is a hard error."
        ),
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for this cell (canonical suite: 0, 1, 2).")
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--check-every", type=int, default=DEFAULT_CHECK_EVERY)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA)
    args = parser.parse_args()

    if args.selftest:
        sys.exit(0 if run_selftest() else 1)
    if args.summarize:
        summarize()
        return
    if args.tier is None or args.seed is None:
        parser.error("--tier and --seed are both required for a real cell (or pass --selftest / --summarize).")

    tier = Tier(args.tier)
    _assert_tier_objective_available(tier)
    if tier is Tier.TWO and args.schedule != SCHEDULE.value:
        parser.error(
            f"--schedule is fixed at {nwn.WidthSchedule.ALL.value!r} for tier 2 (RE-AUTHORIZED, width.md SS3.8) "
            "and is not user-selectable; omit --schedule for a tier-2 cell."
        )
    os.makedirs(RESULTS_DIR, exist_ok=True)
    schedule = nwn.WidthSchedule.ALL if tier is Tier.TWO else nwn.WidthSchedule(args.schedule)
    objective_label = LOSS.value if tier is Tier.ONE else "weighted_squared_error"
    print(
        f"[wsel13] tier={tier.value} toy={_TIER_CONFIG[tier].toy.value} seed={args.seed} w_max={W_MAX} "
        f"arch={ARCH.value} schedule={schedule.value} objective={objective_label}",
        flush=True,
    )
    case, state_dict = run_cell(
        tier, args.seed, max_epochs=args.max_epochs, check_every=args.check_every,
        patience=args.patience, min_delta=args.min_delta, schedule=schedule,
    )

    if not case["all_widths_trustworthy"]:
        print(
            f"*** DO-NOT-CONCLUDE GUARD: tier={tier.value} seed={args.seed} has widths that did NOT converge "
            "trustworthily. Raise --max-epochs before drawing any conclusion. ***"
        )

    cell_path = _cell_json_path(tier, args.seed, schedule)
    with open(cell_path, "w") as f:
        json.dump(_jsonable(case), f, indent=2)
    print(f"wrote {cell_path}")

    state_path = _state_path(tier, args.seed, schedule)
    torch.save(state_dict, state_path)
    print(f"wrote {state_path}")

    spearman = case["step2_ablation"]["spearman_index_vs_importance"]["rho"]
    gap = case["step3_prefix_vs_greedy"]["mean_relative_prefix_gap"]
    tau = case["step3_prefix_vs_greedy"]["kendall_tau_greedy_vs_index"]["tau"]
    print(f"spearman_index_vs_importance={spearman:.4f}  mean_relative_prefix_gap={gap:.4f}  kendall_tau_greedy_vs_index={tau:.4f}")


if __name__ == "__main__":
    main()
