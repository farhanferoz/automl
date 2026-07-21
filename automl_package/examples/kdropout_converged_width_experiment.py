"""Convergence-gated per-input WIDTH dial trained with K-DROPOUT (not per-width separate training).

`converged_width_experiment.py` trained each width as its OWN separate network (its own optimizer, its
own convergence) — a clean upper bound, but NOT the deployable model: it costs one training run PER
width. The model the user actually proposed is ONE joint training pass with k-dropout — every step
trains a sampled subset of widths (the SANDWICH schedule: always width=1 and width=w_max, plus 2 random
middle widths) with a single continuous optimizer. This driver runs exactly that, but gated by the same
per-width convergence rule (`convergence.py`; agent-memory
`feedback_check_loss_trajectory_before_concluding`): training continues until EVERY width's held-out
loss has flattened (per-width `ConvergenceTracker`) or the safety cap is hit, each width keeps its OWN
best-not-last weights, and NO conclusion is drawn from a width whose result is not `trustworthy`.

Why this is not subsumed by the separate-training battery: with independent weights the two schemes
reach the same networks at convergence, but (a) the whole efficiency case for k-dropout is ONE pass vs
twelve, and (b) under sandwich each middle width is trained only ~1/5 of the steps AND a single shared
Adam optimizer applies its bias-correction bookkeeping globally, so k-dropout can converge differently
(and slower on the mids) than dedicated training — which is exactly what the per-width gate measures.

Architecture / data / split / scoring / bars are `IndependentWidthNet` +
`converged_width_experiment` + `sinc_width_experiment` reused verbatim, so the summary is directly
comparable to `capacity_ladder_results/W_CONVERGED/w_converged_summary.json`. The ONLY thing that
changes is the training scheme (k-dropout sandwich vs per-width separate).

Width-MSE program extension (`docs/plans/width_mse_2026-07-16/EXECUTION_PLAN.md` WP-1): `--arch`
selects `NestedWidthNet` (shared trunk + shared heads, 1x params -- the charter question),
`IndependentWidthNet` (K disjoint sub-nets, the old default / positive control), or
`SharedTrunkPerWidthHeadNet` (shared trunk + per-width heads -- the G-WIDTH certified arm and, since
`docs/plans/capacity_programme/width.md`'s WD4 fix, the CLI default); `--loss` selects Gaussian NLL (old default, unchanged) or plain
MSE on the mean output (`logvar_head` untouched). MSE runs are scored with a separate MSE score table
+ the WP-1 fit/curve-shape/dial/deploy bars (`sinc_width_experiment._score_all_widths_mse` and
friends) instead of the LL-based bars. `--loss nll --arch independent` reproduces the pre-existing
path bit-for-bit (no longer the CLI defaults -- pass both flags explicitly).

Usage:
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/kdropout_converged_width_experiment.py --selftest
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/kdropout_converged_width_experiment.py --smoke
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/kdropout_converged_width_experiment.py --smoke --arch nested --loss mse
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/kdropout_converged_width_experiment.py --smoke --arch affine_seam --loss mse
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/kdropout_converged_width_experiment.py --config 0
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/kdropout_converged_width_experiment.py            # all seeds
"""

from __future__ import annotations

import argparse
import enum
import json
import math
import os
import sys
import time

import numpy as np
import torch

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root

import converged_width_experiment as cwe  # noqa: E402
import convergence as cvg  # noqa: E402
import nested_width_net as nwn  # noqa: E402
import sinc_width_experiment as sw  # noqa: E402

from automl_package.utils.pytorch_utils import get_device  # noqa: E402
from automl_package.utils.run_provenance import run_provenance  # noqa: E402

RESULTS_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "W_KDROPOUT_CONVERGED")

# Regime reused verbatim from the separate-training battery so the two summaries are directly comparable.
SEEDS = cwe.SEEDS
W_MAX = cwe.W_MAX
N_TRAIN = cwe.N_TRAIN
N_TEST = cwe.N_TEST
# Mids are trained only ~1/5 of steps under sandwich, so the honest per-width cap is well above the
# separate-training battery's 40k (where each width trained every step); convergence still decides the
# real stop, the cap is only the safety net.
DEFAULT_MAX_EPOCHS = 200000

# Width-cert W5: widths drawn per step under the `--schedule uniform` ablation (see `--schedule` help).
_UNIFORM_SCHEDULE_DRAW_N = 4
# Width-cert W7: report-grade delta_tie sweep for the MSE deploy baseline; the 0.25 row must reproduce
# the canonical `deploy_bar` (regression guard asserted in `run_case`), since `sw.DELTA_TIE == 0.25`.
DELTA_TIE_SWEEP = (0.0, 0.1, 0.25, 0.5)
_DEPLOY_SWEEP_REGRESSION_TOL = 1e-9  # exact-reproduction tolerance for the delta_tie=0.25 regression guard


class Arch(enum.Enum):
    """Width-net architecture to instantiate (`--arch`; closed set, width-MSE program WP-1)."""

    NESTED = "nested"  # NestedWidthNet: shared trunk + shared heads, 1x params -- the charter question.
    INDEPENDENT = "independent"  # IndependentWidthNet: K disjoint sub-nets -- old default / positive control.
    SHARED_TRUNK = "shared_trunk"  # SharedTrunkPerWidthHeadNet: shared trunk + per-width heads -- WP-2 Contingency C, the middle rung.
    AFFINE_SEAM = "affine_seam"  # SharedReadoutPerWidthAffineNet: shared trunk + SHARED readout + 2-param/width affine -- width-cert minimum-seam probe.


class LossType(enum.Enum):
    """Per-width training loss (`--loss`; closed set, width-MSE program WP-1)."""

    NLL = "nll"  # Gaussian NLL, old default -- unchanged bit-for-bit.
    MSE = "mse"  # Mean squared error of the mean output; `logvar_head` untouched/unused.


def _width_loss(
    loss: LossType,
    net: nwn.NestedWidthNet | nwn.IndependentWidthNet | nwn.SharedTrunkPerWidthHeadNet | nwn.SharedReadoutPerWidthAffineNet,
    k: int,
    x_t: torch.Tensor,
    y_t: torch.Tensor,
) -> torch.Tensor:
    """Dispatches to the NLL (`converged_width_experiment._width_nll`, old default) or MSE (`nested_width_net._width_mse`) per-width loss."""
    if loss is LossType.NLL:
        return cwe._width_nll(net, k, x_t, y_t)
    if loss is LossType.MSE:
        return nwn._width_mse(net, k, x_t, y_t)
    raise ValueError(f"unknown loss: {loss!r}")


def _loss_from_readout(loss: LossType, mean: torch.Tensor, log_var: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
    """Applies `_width_loss`'s NLL/MSE formula to an ALREADY-computed `(mean, log_var)` readout.

    `_width_loss` computes the readout itself via `net.forward_width(x_t, k)`, which is exactly the
    per-width trunk recompute WSEL-12 removes for shared-trunk architectures. `_sampled_widths_total_loss`
    calls this once per sampled width off a SINGLE `net.sampled_widths_forward(x_t, widths)` call instead.
    Same formulas as `cwe._width_nll` (NLL) / `nwn._width_mse` (MSE) -- those two functions cannot be
    reused directly here because they take `(net, k)` and always redo the forward internally; neither
    lives in this task's write set to refactor their signature (`converged_width_experiment.py` is out
    of scope entirely, and `nested_width_net.py`'s `_width_mse` is called from other sites unchanged).
    """
    if loss is LossType.NLL:
        ll = nwn.gaussian_log_likelihood(mean.squeeze(1), log_var.squeeze(1), y_t)
        return -ll.mean()
    if loss is LossType.MSE:
        return ((mean.squeeze(1) - y_t) ** 2).mean()
    raise ValueError(f"unknown loss: {loss!r}")


def _sampled_widths_total_loss(
    loss: LossType,
    net: nwn.NestedWidthNet | nwn.IndependentWidthNet | nwn.SharedTrunkPerWidthHeadNet | nwn.SharedReadoutPerWidthAffineNet,
    widths: list[int],
    x_t: torch.Tensor,
    y_t: torch.Tensor,
) -> torch.Tensor:
    """Sum of `_width_loss(loss, net, k, x_t, y_t)` over `widths`.

    Bit-for-bit equivalent to the old `for k in widths: total += _width_loss(...)` accumulation
    (`tests/test_nested_width_single_trunk.py`). WSEL-12: for shared-trunk architectures (`NestedWidthNet`, `SharedTrunkPerWidthHeadNet`,
    `SharedReadoutPerWidthAffineNet`, all of which expose `sampled_widths_forward`) this evaluates the
    trunk ONCE regardless of `len(widths)`, instead of once per width as the naive per-k loop does.
    `IndependentWidthNet` has no shared trunk (`sampled_widths_forward` absent) -- there is nothing to
    reuse there, so it keeps the unchanged per-k loop.
    """
    if hasattr(net, "sampled_widths_forward"):
        mean_all, logvar_all = net.sampled_widths_forward(x_t, widths)
        total = torch.zeros((), device=x_t.device)
        for i in range(len(widths)):
            total = total + _loss_from_readout(loss, mean_all[:, i : i + 1], logvar_all[:, i : i + 1], y_t)
        return total
    total = torch.zeros((), device=x_t.device)
    for k in widths:
        total = total + _width_loss(loss, net, k, x_t, y_t)
    return total


def _trunk_evals_per_step(
    net: nwn.NestedWidthNet | nwn.IndependentWidthNet | nwn.SharedTrunkPerWidthHeadNet | nwn.SharedReadoutPerWidthAffineNet,
    schedule: nwn.WidthSchedule,
    w_max: int,
) -> int:
    """WSEL-12 cost instrumentation: how many shared-trunk forward evaluations one training step pays.

    Shared-trunk architectures (`net.sampled_widths_forward` present) pay ONE trunk eval per step
    regardless of how many widths SANDWICH/UNIFORM samples -- that is the fix. `IndependentWidthNet`
    has no shared trunk to amortize, so it still pays one eval per sampled width, same as before this
    task (its per-width sub-nets are disjoint; there is nothing to reuse).
    """
    if hasattr(net, "sampled_widths_forward"):
        return 1
    if schedule is nwn.WidthSchedule.UNIFORM:
        return _UNIFORM_SCHEDULE_DRAW_N
    mid_candidates = list(range(2, w_max))  # {2..w_max-1}, mirrors _train_kdropout_to_convergence's sandwich draw
    return 2 + min(2, len(mid_candidates))


def _train_kdropout_to_convergence(
    net: nwn.NestedWidthNet | nwn.IndependentWidthNet | nwn.SharedTrunkPerWidthHeadNet | nwn.SharedReadoutPerWidthAffineNet,
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    arch: Arch,
    loss: LossType,
    max_epochs: int,
    check_every: int,
    patience: int,
    min_delta: float,
    seed: int,
    schedule: nwn.WidthSchedule = nwn.WidthSchedule.SANDWICH,
) -> tuple[dict[int, cvg.ConvergenceResult], int | None]:
    """One joint k-dropout run (SANDWICH default), gated by PER-WIDTH held-out convergence.

    A single continuous Adam optimizer over the whole net (the sandwich step is replicated inline from
    `nested_width_net.train_nested_width` rather than called, because chunking that function would rebuild
    the optimizer each chunk and reset Adam's momentum — unfaithful to one continuous run). Every
    `check_every` epochs each width's held-out loss (NLL or MSE, per `loss`) is checkpointed into its own
    `ConvergenceTracker`; the loop stops when ALL widths have flattened (or the cap is hit).

    Checkpointing is ARCH-dependent (width-MSE program WP-1 item 3): `Arch.INDEPENDENT`'s disjoint
    sub-nets restore each width's OWN best-not-last weights independently (unchanged from the original
    single-arch version of this function). `Arch.NESTED`'s shared trunk+heads make per-width restoration
    impossible — restoring width-3's best state would clobber width-7's — so instead the WHOLE net is
    checkpointed at the epoch with the best MEAN per-width validation loss, and that single state is
    restored at the end. Per-width `ConvergenceTracker`s still track every width's own trajectory for the
    trustworthiness flags and the stop rule in both cases.

    Args:
        net: the net to train in place (`NestedWidthNet` for `Arch.NESTED`, `IndependentWidthNet` for
            `Arch.INDEPENDENT`).
        x_tr: standardized training inputs, shape `(N, 1)`.
        y_tr: standardized training targets, shape `(N,)`.
        x_val: standardized held-out inputs used only for convergence monitoring.
        y_val: standardized held-out targets used only for convergence monitoring.
        arch: which checkpointing scheme to use (must match `net`'s actual type).
        loss: which per-width loss (`_width_loss`) trains and validates each width.
        max_epochs: safety cap on optimizer steps (== epochs; full-batch).
        check_every: epochs between per-width held-out checkpoints.
        patience: consecutive flat checkpoints that declare one width converged.
        min_delta: held-out-loss decrease counted as a real improvement.
        seed: RNG seed for the per-step middle-width draw.
        schedule: `WidthSchedule.SANDWICH` (default, byte-identical to the original hardcoded behavior:
            always {1, w_max} + 2 random mids) or `WidthSchedule.UNIFORM` (W5 ablation: `_UNIFORM_SCHEDULE_DRAW_N`
            widths drawn uniformly from {1..w_max}, NO guaranteed inclusion of {1, w_max}).

    Returns:
        `({width -> ConvergenceResult}, best_mean_val_epoch)`. Best weights are already restored (per
        width for `Arch.INDEPENDENT`, whole-net for `Arch.NESTED`). `best_mean_val_epoch` is the epoch
        of the whole-net checkpoint for `Arch.NESTED`, else `None` (inapplicable under per-width
        checkpointing).
    """
    w_max = net.w_max
    opt = torch.optim.Adam(net.parameters(), lr=cwe.LR)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    mid_candidates = list(range(2, w_max))  # {2..w_max-1}
    n_mid_draw = min(2, len(mid_candidates))

    trackers = {k: cvg.ConvergenceTracker(patience=patience, min_delta=min_delta) for k in range(1, w_max + 1)}
    best_states: dict[int, dict | None] = dict.fromkeys(range(1, w_max + 1))
    best_mean_val = math.inf
    best_mean_val_epoch: int | None = None
    best_net_state: dict | None = None

    net.train()
    final_epoch = max_epochs
    for epoch in range(1, max_epochs + 1):
        opt.zero_grad()
        if schedule is nwn.WidthSchedule.UNIFORM:
            # W5 ablation: N uniform draws from {1..w_max}, no guaranteed {1, w_max} sandwich.
            widths = torch.randint(1, w_max + 1, (_UNIFORM_SCHEDULE_DRAW_N,), generator=gen).tolist()
        else:
            widths = [1, w_max]
            if n_mid_draw:
                perm = torch.randperm(len(mid_candidates), generator=gen)[:n_mid_draw]
                widths += [mid_candidates[i] for i in perm.tolist()]
        total_loss = _sampled_widths_total_loss(loss, net, widths, x_tr, y_tr)
        total_loss.backward()
        opt.step()

        if epoch % check_every == 0:
            net.eval()
            with torch.no_grad():
                per_width_val = {k: float(_width_loss(loss, net, k, x_val, y_val).item()) for k in range(1, w_max + 1)}
            if arch is Arch.INDEPENDENT:
                for k, v in per_width_val.items():
                    if trackers[k].update(epoch, v):
                        best_states[k] = {n: t.detach().clone() for n, t in net.subnets[k - 1].state_dict().items()}
            else:
                for k, v in per_width_val.items():
                    trackers[k].update(epoch, v)
                mean_val = sum(per_width_val.values()) / w_max
                if mean_val < best_mean_val:
                    best_mean_val = mean_val
                    best_mean_val_epoch = epoch
                    best_net_state = {n: t.detach().clone() for n, t in net.state_dict().items()}
            net.train()
            if all(t.done for t in trackers.values()):
                final_epoch = epoch
                break

    if arch is Arch.INDEPENDENT:
        for k in range(1, w_max + 1):
            if best_states[k] is not None:
                net.subnets[k - 1].load_state_dict(best_states[k])
    elif best_net_state is not None:
        net.load_state_dict(best_net_state)
    net.eval()
    return {k: trackers[k].result(final_epoch=final_epoch) for k in range(1, w_max + 1)}, best_mean_val_epoch


def run_case(
    seed: int,
    w_max: int,
    n_train: int,
    n_test: int,
    max_epochs: int,
    device: str,
    *,
    arch: Arch,
    loss: LossType,
    check_every: int,
    patience: int,
    min_delta: float,
    toy: nwn.Toy = nwn.Toy.HETERO,
    sigma: float = nwn.HETERO_NOISE_SIGMA,
    schedule: nwn.WidthSchedule = nwn.WidthSchedule.SANDWICH,
    router_hidden_mult: float = 1.0,
) -> dict:
    """Trains the k-dropout net to per-width convergence, then scores the frozen net for the pre-registered bars.

    `loss=LossType.NLL` reproduces the pre-existing path bit-for-bit (LL-based construction/recovery/
    deploy bars, reused verbatim from `sinc_width_experiment`). `loss=LossType.MSE` scores an MSE table
    instead and reads the width-MSE program's fit/curve-shape/dial/deploy bars
    (`sinc_width_experiment`'s `_mse`-suffixed twins) plus a report-only soft-target sensitivity arm.

    `toy=nwn.Toy.HETERO` (default) is the 2-region WP-2/WP-4 toy at common-mode noise `sigma` (WP-4
    sweeps `sigma`, so `floor_hard = sigma**2`). `toy=nwn.Toy.HETERO3` (WP-3) adds a noisy-easy third
    region (fixed structure, `sigma` ignored — its hard region keeps the quiet `HETERO_NOISE_SIGMA`, so
    `floor_hard = HETERO_NOISE_SIGMA**2`) and, under `--loss mse`, the §5.4 noisy-easy negative-control
    bar is added to the case.
    """
    if toy is nwn.Toy.HETERO3:
        x_tr, y_tr, _reg_tr = nwn.make_hetero3(n_train, seed)
        x_te, y_te, region_te = nwn.make_hetero3(n_test, seed + 500)
        floor_hard = nwn.HETERO_NOISE_SIGMA**2  # hard region keeps the quiet sigma; noisy-easy region is scored separately.
    else:
        x_tr, y_tr, _reg_tr = nwn.make_hetero(n_train, seed, sigma=sigma)
        x_te, y_te, region_te = nwn.make_hetero(n_test, seed + 500, sigma=sigma)
        floor_hard = sigma**2  # WP-4 ladder: common-mode noise, analytic floor stays sigma**2.

    p1_idx = np.arange(0, n_train, 2)
    p2_idx = np.arange(1, n_train, 2)
    x_p1, y_p1 = x_tr[p1_idx], y_tr[p1_idx]
    x_p2, y_p2 = x_tr[p2_idx], y_tr[p2_idx]

    # Same phase-1 train/val carve as the separate-training battery (rest = train, every 5th = val).
    val_mask = (np.arange(len(x_p1)) % cwe.VAL_EVERY) == 0
    norm = cwe._standardize_fit(x_p1[~val_mask], y_p1[~val_mask])
    x_tr_t, y_tr_t = cwe._to_std_tensors(x_p1[~val_mask], y_p1[~val_mask], norm)
    x_val_t, y_val_t = cwe._to_std_tensors(x_p1[val_mask], y_p1[val_mask], norm)

    torch.manual_seed(seed)
    if arch is Arch.NESTED:
        net = nwn.NestedWidthNet(w_max=w_max)
    elif arch is Arch.INDEPENDENT:
        net = nwn.IndependentWidthNet(w_max=w_max)
    elif arch is Arch.SHARED_TRUNK:
        net = nwn.SharedTrunkPerWidthHeadNet(w_max=w_max)
    else:
        net = nwn.SharedReadoutPerWidthAffineNet(w_max=w_max)
    # WSEL-12 cost instrumentation (Step 3): wall-clock wraps ONLY the training call itself, unchanged
    # otherwise -- `_train_kdropout_to_convergence`'s own signature/return contract is untouched, since
    # other in-flight work on this branch imports it.
    _train_t0 = time.perf_counter()
    conv, best_mean_val_epoch = _train_kdropout_to_convergence(
        net,
        x_tr_t,
        y_tr_t,
        x_val_t,
        y_val_t,
        arch=arch,
        loss=loss,
        max_epochs=max_epochs,
        check_every=check_every,
        patience=patience,
        min_delta=min_delta,
        seed=seed,
        schedule=schedule,
    )
    train_wall_clock_s = time.perf_counter() - _train_t0
    # W6: the deploy router's hidden sizes, scaled off the ck6 default (mult 1.0 reproduces it exactly).
    router_hidden = tuple(max(1, round(h * router_hidden_mult)) for h in sw.ck6.HIDDEN)
    n_trustworthy = sum(1 for r in conv.values() if r.trustworthy)
    case = {
        "seed": seed,
        "arch": arch.value,
        "loss": loss.value,
        "convergence": {k: r.summary() for k, r in conv.items()},
        "n_widths_trustworthy": n_trustworthy,
        "all_widths_trustworthy": n_trustworthy == w_max,
        "best_mean_val_epoch": best_mean_val_epoch,
        # WSEL-12 cost instrumentation (Step 3, additive only): per-seed training wall clock and the
        # (arch, schedule)-determined trunk-eval count per step, aggregated into summary["config"] by
        # `main()` (`docs/plans/capacity_programme/width.md` lines 1071-1074).
        "train_wall_clock_s": train_wall_clock_s,
        "trunk_evals_per_step": _trunk_evals_per_step(net, schedule, w_max),
    }

    if loss is LossType.NLL:
        # Frozen-net scoring on TEST — the 3 bars, reused verbatim from sinc_width_experiment.
        score_te = sw._score_all_widths(net, norm, x_te, y_te, device)
        fixed_width_ll = {k: score_te[:, k - 1] for k in range(1, w_max + 1)}
        construction = sw._construction_bar(fixed_width_ll, region_te, k_lo=1, k_mid=max(2, w_max // 2), w_max=w_max)

        score_p2 = sw._score_all_widths(net, norm, x_p2, y_p2, device)
        router = sw._fit_selector(score_p2, x_p2, w_max, seed, device)
        selector_nll, marginal_p, expected_width = sw._selector_eval(router, score_te, x_te, device)
        recovery = sw._recovery_bar(expected_width, region_te)
        per_k_nll = {k: float(-fixed_width_ll[k].mean()) for k in range(1, w_max + 1)}
        oracle_nll = float(-score_te.max(axis=1).mean())
        deploy = sw._deploy_bar(selector_nll, per_k_nll, oracle_nll)

        case.update(
            {
                "hard_curve": [float(score_te[region_te == 1][:, k - 1].mean()) for k in range(1, w_max + 1)],
                "easy_curve": [float(score_te[region_te == 0][:, k - 1].mean()) for k in range(1, w_max + 1)],
                "construction": construction,
                "recovery": recovery,
                "deploy": deploy,
                "marginal_p": marginal_p.tolist(),
                "per_k_nll": per_k_nll,
            }
        )
        return case

    # loss is LossType.MSE — separate MSE score table + the width-MSE program's §5 bars.
    err2_te = sw._score_all_widths_mse(net, norm, x_te, y_te, device)
    err2_by_width_te = {k: err2_te[:, k - 1] for k in range(1, w_max + 1)}
    # floor_hard computed above from `toy`/`sigma` (sigma**2 for HETERO; quiet HETERO_NOISE_SIGMA**2 for HETERO3).

    fit_bar = sw._fit_bar_mse(err2_by_width_te, region_te, w_max, floor_hard)
    curve_gate = sw._curve_shape_gate_mse(err2_by_width_te, region_te, w_max)

    err2_p2 = sw._score_all_widths_mse(net, norm, x_p2, y_p2, device)
    router = sw._fit_selector_mse(err2_p2, x_p2, w_max, seed, device, delta_tie=sw.DELTA_TIE, hidden=router_hidden)
    soft_blend_mse, expected_width = sw._selector_eval_mse(router, err2_te, x_te, device)
    dial_bar = sw._recovery_bar(expected_width, region_te)  # reused verbatim -- see module docstring note

    err2_hardpick, executed_width = sw._route_hardpick_mse(router, err2_te, x_te, device)
    deploy_bar = sw._deploy_bar_mse(err2_hardpick, executed_width, err2_by_width_te)

    # W7: report-grade deploy sweep -- non-hindsight (val-selected) baseline + a delta_tie grid. Report
    # payoff numbers come from HERE (verdict-doc §6: the old deploy bar's best_fixed_k is test-selected).
    err2_p2_by_width = {k: err2_p2[:, k - 1] for k in range(1, w_max + 1)}
    deploy_sweep = {
        "router_hidden": list(router_hidden),
        "baseline_test_selected": {"mse_best_fixed": deploy_bar["mse_best_fixed"], "best_fixed_k": deploy_bar["best_fixed_k"]},
        "baseline_val_selected": sw._deploy_bar_mse_valselected(err2_p2_by_width, err2_by_width_te),
        "per_delta_tie": {},
    }
    for _dt in DELTA_TIE_SWEEP:
        _router_dt = sw._fit_selector_mse(err2_p2, x_p2, w_max, seed, device, delta_tie=_dt, hidden=router_hidden)
        _e2_hp, _ew = sw._route_hardpick_mse(_router_dt, err2_te, x_te, device)
        _db = sw._deploy_bar_mse(_e2_hp, _ew, err2_by_width_te)
        deploy_sweep["per_delta_tie"][f"{_dt:g}"] = {"mse_hardpick": _db["mse_hardpick"], "mean_executed_width": _db["mean_executed_width"]}
    # Regression guard: delta_tie=0.25 refit must reproduce the canonical deploy_bar (same seed/hidden/tie).
    if abs(deploy_sweep["per_delta_tie"][f"{sw.DELTA_TIE:g}"]["mse_hardpick"] - deploy_bar["mse_hardpick"]) >= _DEPLOY_SWEEP_REGRESSION_TOL:
        raise AssertionError("W7 deploy_sweep: delta_tie=0.25 row must reproduce deploy_bar (regression guard)")

    # Sensitivity arm (report only, not gating any bar): soft targets, one global scale, not tuned.
    router_soft = sw._fit_selector_mse_soft(err2_p2, x_p2, w_max, seed, device)
    soft_blend_mse_sens, expected_width_sens = sw._selector_eval_mse(router_soft, err2_te, x_te, device)
    dial_sens = sw._recovery_bar(expected_width_sens, region_te)

    case.update(
        {
            "norm": norm,
            "hard_curve_mse": [float(err2_te[region_te == 1][:, k - 1].mean()) for k in range(1, w_max + 1)],
            "easy_curve_mse": [float(err2_te[region_te == 0][:, k - 1].mean()) for k in range(1, w_max + 1)],
            "fit_bar": fit_bar,
            "curve_gate": curve_gate,
            "dial_bar": dial_bar,
            "deploy_bar": deploy_bar,
            "deploy_sweep": deploy_sweep,
            "soft_blend_mse_secondary": {"mse": soft_blend_mse, "note": "executes ALL widths (soft blend); not the compute/deploy claim"},
            "sensitivity_soft_targets": {"soft_blend_mse": soft_blend_mse_sens, "dial": dial_sens},
        }
    )
    if toy is nwn.Toy.HETERO3:
        # §5.4 noisy-easy negative control (WP-3 only): the dial must read capacity-hunger, not raw error.
        case["noisy_easy_bar"] = sw._noisy_easy_bar_mse(expected_width, region_te)
    return case


def _selftest_bars_present(case: dict, loss: LossType) -> bool:
    """Wiring check: the loss-appropriate bar keys landed in the case dict."""
    if loss is LossType.NLL:
        return all(key in case for key in ("construction", "recovery", "deploy", "per_k_nll"))
    return all(key in case for key in ("fit_bar", "curve_gate", "dial_bar", "deploy_bar", "norm"))


def _selftest_masking_agreement(net: nwn.SharedReadoutPerWidthAffineNet, x: torch.Tensor) -> tuple[bool, float]:
    """AFFINE_SEAM-only wiring check: `forward_width` vs `all_widths_forward` agreement.

    `forward_width` (explicit mask) is what training uses; `all_widths_forward` (cumsum + per-width
    affine) is what `sinc_width_experiment._score_all_widths_mse` uses -- the two code paths must
    compute the same thing at every k, or the net gets trained against one readout and scored against
    another.
    """
    with torch.no_grad():
        mean_all, _logvar_all = net.all_widths_forward(x)
    max_err = 0.0
    for k in range(1, net.w_max + 1):
        with torch.no_grad():
            mean_k, _logvar_k = net.forward_width(x, k)
        max_err = max(max_err, (mean_all[:, k - 1 : k] - mean_k).abs().max().item())
    return max_err < nwn._CONSISTENCY_TOL, max_err


def run_selftest() -> bool:
    """Wiring check: tiny k-dropout net trains to a tiny cap for BOTH arch x BOTH loss (4 combos)."""
    ok = True
    for arch in Arch:
        for loss in LossType:
            case = run_case(
                seed=0, w_max=3, n_train=200, n_test=100, max_epochs=1500, device="cpu", arch=arch, loss=loss, check_every=100, patience=3, min_delta=2e-3
            )
            conv = case["convergence"]
            ok_traj = all(len(conv[k]["trajectory"]) >= 1 for k in range(1, 4))
            ok_flags = all(isinstance(conv[k]["converged"], bool) for k in range(1, 4))
            ok_bars = _selftest_bars_present(case, loss)
            combo_ok = ok_traj and ok_flags and isinstance(case["all_widths_trustworthy"], bool) and ok_bars
            print(
                f"[kdropout-converged selftest] arch={arch.value} loss={loss.value} "
                f"trajectories_recorded={ok_traj} flags_present={ok_flags} bars_present={ok_bars}  {'PASS' if combo_ok else 'FAIL'}"
            )
            print(f"  (per-width stop epochs: {[conv[k]['stop_epoch'] for k in range(1, 4)]})")
            ok = ok and combo_ok

    # AFFINE_SEAM-only: forward_width (training) vs all_widths_forward (scoring) must agree numerically.
    affine_net = nwn.SharedReadoutPerWidthAffineNet(w_max=6)
    affine_net.eval()
    mask_ok, mask_err = _selftest_masking_agreement(affine_net, torch.randn(37, 1))
    print(
        f"[kdropout-converged selftest] arch=affine_seam masking agreement (forward_width vs all_widths_forward): "
        f"max_abs_err={mask_err:.3e} (tol={nwn._CONSISTENCY_TOL:.0e})  {'PASS' if mask_ok else 'FAIL'}"
    )
    ok = ok and mask_ok

    # WP-3 wiring: the 3-region toy + the noisy-easy negative-control bar land on the case (arch=independent, mse).
    case3 = run_case(
        seed=0, w_max=3, n_train=300, n_test=150, max_epochs=1500, device="cpu",
        arch=Arch.INDEPENDENT, loss=LossType.MSE, check_every=100, patience=3, min_delta=2e-3, toy=nwn.Toy.HETERO3,
    )
    ne = case3.get("noisy_easy_bar")
    hetero3_ok = isinstance(ne, dict) and all(key in ne for key in ("mean_width_noisy_easy", "stays_narrow", "hard_beats_noisy_2se", "noisy_easy_pass"))
    print(f"[kdropout-converged selftest] toy=hetero3 arch=independent loss=mse noisy_easy_bar_present={hetero3_ok}  {'PASS' if hetero3_ok else 'FAIL'}")
    ok = ok and hetero3_ok

    print(f"[kdropout-converged selftest] {'PASS' if ok else 'FAIL'}")
    return ok


def _print_case_mse(case: dict, w_max: int) -> None:
    """MSE twin of `converged_width_experiment._print_case` — prints the §5 fit/curve-gate/dial/deploy bars."""
    print("  per-width convergence (val MSE, lower=better):")
    for k in range(1, w_max + 1):
        s = case["convergence"][k]
        flag = "OK " if s["trustworthy"] else ("CAP" if s["hit_cap"] else "crp")
        print(f"    w{k:2d} [{flag}] stop@{s['stop_epoch']:6d} best_val={s['best_val']:.4f} recent_impr={s['recent_improvement']:.4f}")
    print(f"  ALL WIDTHS CONVERGED (trustworthy): {case['all_widths_trustworthy']}  ({case['n_widths_trustworthy']}/{w_max})")
    if case["best_mean_val_epoch"] is not None:
        print(f"  best_mean_val_epoch={case['best_mean_val_epoch']} (shared-arch whole-net checkpoint)")
    print(f"  hard MSE by width: {np.array2string(np.array(case['hard_curve_mse']), precision=4, floatmode='fixed')}")
    print(f"  easy MSE by width: {np.array2string(np.array(case['easy_curve_mse']), precision=4, floatmode='fixed')}")
    fit_bar, curve_gate, dial_bar, deploy_bar = case["fit_bar"], case["curve_gate"], case["dial_bar"], case["deploy_bar"]
    print(f"  fit: ratio_to_floor={fit_bar['ratio_to_floor']:.3f} pass={fit_bar['pass']} strong_pass={fit_bar['strong_pass']}")
    print(
        f"  curve_gate: pass={curve_gate['curve_gate_pass']} "
        f"(hard_drops_to_mid={curve_gate['hard_drops_to_mid']} hard_plateaus={curve_gate['hard_plateaus_at_wmax']} easy_flat={curve_gate['easy_flat']})"
    )
    print(
        f"  dial: hard_width={dial_bar['mean_expected_width_centre']:.3f} easy_width={dial_bar['mean_expected_width_tail']:.3f} "
        f"sep_pass={dial_bar['separation_beats_2se']}"
    )
    print(
        f"  deploy: mse_hardpick={deploy_bar['mse_hardpick']:.4f} mse_best_fixed={deploy_bar['mse_best_fixed']:.4f} "
        f"best_fixed_k={deploy_bar['best_fixed_k']} mean_executed_width={deploy_bar['mean_executed_width']:.3f} pass={deploy_bar['deploy_pass']}",
        flush=True,
    )
    if "deploy_sweep" in case:
        ds = case["deploy_sweep"]
        bv = ds["baseline_val_selected"]
        print(f"  deploy_sweep: router_hidden={ds['router_hidden']} baseline_val_selected(k={bv.get('best_fixed_k_valselected')},"
              f" test_mse={bv.get('mse_best_fixed_valselected_test'):.4f})")
        for _dt, _row in ds["per_delta_tie"].items():
            print(f"    delta_tie={_dt}: mse_hardpick={_row['mse_hardpick']:.4f} mean_executed_width={_row['mean_executed_width']:.3f}")
    if "noisy_easy_bar" in case:
        ne = case["noisy_easy_bar"]
        print(
            f"  noisy_easy (WP-3): width easy={ne['mean_width_easy']:.3f} hard={ne['mean_width_hard']:.3f} noisy_easy={ne['mean_width_noisy_easy']:.3f} "
            f"stays_narrow={ne['stays_narrow']} hard_beats_noisy_2se={ne['hard_beats_noisy_2se']} pass={ne['noisy_easy_pass']}",
            flush=True,
        )


def _summary_filename(arch: Arch, loss: LossType, toy: nwn.Toy, n_train: int, sigma: float, tag: str | None) -> str:
    """Collision-free summary filename; the canonical WP-2 config keeps its HISTORIC name (cited downstream).

    The canonical WP-2 cell (`toy=hetero`, `n_train=N_TRAIN=1500`, `sigma=HETERO_NOISE_SIGMA=0.05`, no
    tag) resolves to the pre-existing `w_kdropout_converged_summary_{arch}_{loss}.json` — the file the
    verdict doc and RESUME already reference, so re-running the canonical cell reproduces in place.
    Every non-canonical cell (WP-3's `hetero3`, WP-4's `n`/`sigma` grid) or any explicit `--tag`
    encodes `toy`/`n`/`sigma`/`tag` into the name so the 36-cell ladder and the nested confirmation
    never collide with each other or clobber the canonical files (the plan §7 two-writers warning).
    """
    base = f"w_kdropout_converged_summary_{arch.value}_{loss.value}"
    _sigma_eq_tol = 1e-12  # float-equality slop for "sigma == the canonical 0.05"
    is_canonical = toy is nwn.Toy.HETERO and n_train == N_TRAIN and abs(sigma - nwn.HETERO_NOISE_SIGMA) < _sigma_eq_tol
    if is_canonical and not tag:
        return base + ".json"
    parts = [base]
    if toy is not nwn.Toy.HETERO:
        parts.append(toy.value)
    parts += [f"n{n_train}", f"s{sigma:g}"]
    if tag:
        parts.append(tag)
    return "_".join(parts) + ".json"


def main() -> None:
    """Runs the k-dropout convergence-gated width battery (or `--smoke` / `--selftest`)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="Tiny wiring check, then exit.")
    parser.add_argument("--smoke", action="store_true", help="Tiny config (w_max=4, n_train=200, n_test=100, low cap, 1 seed); no save.")
    parser.add_argument("--config", type=int, default=None, help="One seed only (any int, W9 5-seed bump; default = all seeds).")
    parser.add_argument("--arch", choices=[a.value for a in Arch], default=Arch.SHARED_TRUNK.value, help="Width-net architecture (default: shared_trunk, G-WIDTH certified).")
    parser.add_argument("--loss", choices=[loss_type.value for loss_type in LossType], default=LossType.NLL.value, help="Per-width training loss (default: nll, the old default).")
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS, help="Safety cap (convergence decides the real stop).")
    parser.add_argument("--check-every", type=int, default=cvg.DEFAULT_CHECK_EVERY, help="Epochs between per-width held-out checkpoints.")
    parser.add_argument("--patience", type=int, default=cvg.DEFAULT_PATIENCE, help="Flat checkpoints that declare a width converged.")
    parser.add_argument("--min-delta", type=float, default=cvg.DEFAULT_MIN_DELTA, help="Held-out-loss decrease (nats, or sq-error units under --loss mse) counted as improvement.")
    parser.add_argument("--toy", choices=[t.value for t in nwn.Toy], default=nwn.Toy.HETERO.value, help="Synthetic toy (hetero = 2-region WP-2/WP-4; hetero3 = WP-3 control).")
    parser.add_argument("--n-train", type=int, default=None, help="Override train N (default 1500 hetero / 2250 hetero3). WP-4 sweeps this.")
    parser.add_argument("--n-test", type=int, default=None, help="Override test N (default 500 hetero / 750 hetero3).")
    parser.add_argument("--sigma", type=float, default=nwn.HETERO_NOISE_SIGMA, help="Common-mode noise std for hetero (WP-4 ladder; floor_hard=sigma**2). Ignored by hetero3.")
    parser.add_argument("--tag", type=str, default=None, help="Optional filename suffix (keeps ladder cells / confirmation runs from clobbering canonical files).")
    parser.add_argument("--schedule", choices=[nwn.WidthSchedule.SANDWICH.value, nwn.WidthSchedule.UNIFORM.value], default=nwn.WidthSchedule.SANDWICH.value,
                        help="W5 width schedule: sandwich (default, {1,w_max}+2 mid) or uniform (N uniform draws, no guarantee).")
    parser.add_argument("--router-hidden-mult", type=float, default=1.0,
                        help="W6: scale deploy-router hidden off ck6.HIDDEN=(32,32) (0.5=half, 2.0=double; 1.0=canonical).")
    parser.add_argument("--w-max", type=int, default=None, help="W8: max width / number of levels (default 12=W_MAX; scan couples #levels with total trunk size).")
    args = parser.parse_args()

    if args.selftest:
        sys.exit(0 if run_selftest() else 1)

    arch = Arch(args.arch)
    loss = LossType(args.loss)
    toy = nwn.Toy(args.toy)
    sigma = args.sigma
    tag = args.tag
    schedule = nwn.WidthSchedule(args.schedule)

    device = str(get_device())
    if args.smoke:
        seeds, w_max, max_epochs, save = [0], 4, 4000, False
        n_train = args.n_train if args.n_train is not None else 200
        n_test = args.n_test if args.n_test is not None else 100
    else:
        seeds = list(SEEDS) if args.config is None else [int(args.config)]
        w_max, max_epochs, save = (args.w_max if args.w_max is not None else W_MAX), args.max_epochs, True
        default_n_train, default_n_test = (2250, 750) if toy is nwn.Toy.HETERO3 else (N_TRAIN, N_TEST)
        n_train = args.n_train if args.n_train is not None else default_n_train
        n_test = args.n_test if args.n_test is not None else default_n_test

    print(
        f"[kdropout-converged] device={device} arch={arch.value} loss={loss.value} toy={toy.value} sigma={sigma:g} "
        f"seeds={seeds} w_max={w_max} n_train={n_train} n_test={n_test} max_epochs_cap={max_epochs} check_every={args.check_every} tag={tag}",
        flush=True,
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)

    per_case = []
    for seed in seeds:
        print(f"=== W_KDROPOUT_CONVERGED seed={seed} arch={arch.value} loss={loss.value} toy={toy.value} n_train={n_train} sigma={sigma:g} ===", flush=True)
        case = run_case(
            seed, w_max, n_train, n_test, max_epochs, device,
            arch=arch, loss=loss, check_every=args.check_every, patience=args.patience, min_delta=args.min_delta, toy=toy, sigma=sigma,
            schedule=schedule, router_hidden_mult=args.router_hidden_mult,
        )
        per_case.append(case)
        if loss is LossType.NLL:
            cwe._print_case(case, w_max)
        else:
            _print_case_mse(case, w_max)

    any_untrustworthy = [c["seed"] for c in per_case if not c["all_widths_trustworthy"]]
    if any_untrustworthy:
        print(f"\n*** DO-NOT-CONCLUDE GUARD: seeds {any_untrustworthy} have widths that did NOT converge (hit cap / still creeping). ***")
        print("*** Raise --max-epochs for those before drawing any conclusion from their curves. ***")
    else:
        print("\nAll widths converged on all seeds — curves are trustworthy to interpret.")

    if save:
        # Filename encodes arch/loss (+ toy/n/sigma/tag for non-canonical cells) so concurrent battery
        # arms and the WP-3/WP-4 ladders never write the same file — the plan's §7 two-writers warning.
        path = os.path.join(RESULTS_DIR, _summary_filename(arch, loss, toy, n_train, sigma, tag))
        summary = {
            "config": {
                "schedule": f"kdropout_{schedule.value}",
                "router_hidden_mult": args.router_hidden_mult,
                "arch": arch.value,
                "loss": loss.value,
                "toy": toy.value,
                "tag": tag,
                "w_max": w_max,
                "n_train": n_train,
                "n_test": n_test,
                "sigma": sigma,
                "seeds": seeds,
                "max_epochs_cap": max_epochs,
                "lr": cwe.LR,
                "val_every": cwe.VAL_EVERY,
                "check_every": args.check_every,
                "patience": args.patience,
                "min_delta": args.min_delta,
                # WSEL-12 cost instrumentation (Step 3, additive only -- no key above this line changed):
                # per-seed training wall clock, index-aligned with "seeds" above, and the trunk-eval count
                # one training STEP pays (constant across seeds for one config: same arch/schedule/w_max).
                "train_wall_clock_s": [c["train_wall_clock_s"] for c in per_case],
                "trunk_evals_per_step": per_case[0]["trunk_evals_per_step"] if per_case else None,
            },
            "per_case": per_case,
            "untrustworthy_seeds": any_untrustworthy,
            # What produced these numbers -- library versions, git commit, and above all THREAD
            # COUNT, which changes float reduction order and therefore the low-order bits. Added
            # 2026-07-21 after a saved reference could not be reproduced by its own commit and the
            # artifact recorded nothing that could explain why
            # (docs/plans/capacity_programme/shared/fp5-stale-reference-finding.md).
            "provenance": run_provenance(),
        }
        with open(path, "w") as f:
            json.dump(sw._jsonable(summary), f, indent=2)
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
