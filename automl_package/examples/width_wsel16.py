"""WSEL-16 stage 1 -- can the CHEAP structure (`NestedWidthNet`, Design A: one output weight per unit) be TRAINED to work? (`docs/plans/capacity_programme/width.md` ~1527-1682).

Design A is 6x cheaper in output parameters than the certified `SharedTrunkPerWidthHeadNet` (Design B)
and is the structure the strand's importance-ordering account is written for (`shared/
width_transformer_port.md` §1) -- but trained normally (the plain summed loss) it FAILS (MASTER
Decision 1): hidden unit `w_1` receives gradient from all `w_max` width terms at once and is pulled
`w_max` ways in a tug-of-war. This task asks whether a TRAINING change (not a new architecture) fixes
that, on the SAME canonical toy suite (`width.md` §3.8: tier 1 = `hetero`, tier 2 = `hetero3`) and the
SAME convergence gate as the rest of the width strand.

**The five stage-1 arms.** `h = hidden(x)` is the shared `(N, w_max)` trunk output, `c_j` unit `j`'s own
contribution, `S_k = b + sum_{j<=k} c_j` the width-k running sum:

  - `B_HEADS` -- `SharedTrunkPerWidthHeadNet`, loss `sum_k MSE(head_k(mask_k(h)), y)`. THE REFERENCE,
    unchanged from the certified run. No new code.
  - `A_JOINT` -- `NestedWidthNet`, loss `sum_k MSE(S_k, y)`, `c_j = w_j * h_j`. NEGATIVE CONTROL -- this
    must FAIL (Step 3 halt condition (a)). No new code.
  - `A_STOPGRAD` -- SAME structure as `A_JOINT` (literally `NestedWidthNet`, no wrapper class -- §3.9's
    reuse inventory: "Only the gate arm is new"), one change to the LOSS: `sum_k MSE(detach(S_k - c_k) +
    c_k, y)`. Each unit trains only against what the units before it left over. Computed in ONE cumsum
    pass by `stopgrad_all_widths_pred` below -- no python loop over widths, no staging, no extra forward.
  - `A_GATES` -- `width_candidates.MonotoneGateWidthNet`, the ONLY new class this task adds: `A_JOINT`'s
    structure with `c_j = g_j * w_j * h_j`, `g_j` a monotone-decreasing gate driven by ONE learnable
    scalar. `A_JOINT`'s plain summed loss, no penalty term. OUR SIMPLIFICATION of the published
    monotone-gate mechanism (see that class's docstring).
  - `INDEPENDENT` -- `IndependentWidthNet`, 12 disjoint sub-nets. POSITIVE CONTROL / ceiling. No shared
    hidden trunk at all, so the ordering diagnostic below does not apply to it (see
    `_ordering_statistic`).

**Reuse discipline (§3.9).** Four of the five arms are package classes trained through the EXISTING
`kdropout_converged_width_experiment._train_kdropout_to_convergence` at tier 1 (plain `--loss mse`,
byte-identical dispatch to the certified driver). `A_STOPGRAD`'s loss shape, and EVERY arm's tier-2
fixed-sigma WEIGHTED objective (`width.md` §3.7, assigned to this task by §3.8's RESOLUTION UPDATE
2026-07-22), cannot be expressed through that function's `LossType` (NLL/MSE only, and
`kdropout_converged_width_experiment.py` is read-only for this task) -- `_train_custom_to_convergence`
below covers exactly those two cases, reproducing the SANDWICH width-draw and `convergence.
ConvergenceTracker` gate verbatim (that function does not factor its draw out into a reusable helper).

**Step 3 controls (run BEFORE any candidate cell, `MASTER Decision 14`).** `--summarize` reads whatever
tier-1 `A_JOINT`/`B_HEADS`/`INDEPENDENT` cells are on disk and reports two HALT conditions: (a) `A_JOINT`
does NOT fail (its per-width held-out MSE is within 10% of `B_HEADS`' at every width); (b) `INDEPENDENT`
does not reach its certified fit bar against `capacity_ladder_results/W_KDROPOUT_CONVERGED/
w_kdropout_converged_summary_independent_mse.json`. Both are reported in `frozen.json`; ACTING on a
triggered halt (stopping the grid, escalating) is the ROOT's job, not this driver's -- this task only
authors the check.

**Pre-registered bars, fixed before any run.** PRIMARY: the candidate arm's FULL-WIDTH held-out MSE
within 10% of `B_HEADS`', tier 1 AND tier 2, ALL 3 seeds. ORDERING: the candidate's
`spearman_index_vs_importance` at least as strong (<=) as `B_HEADS`' (imported from `width_wsel13.py`,
same three-split carve, so the two tasks' numbers are directly comparable). COST (reported, not gating):
both parameter counts, and wall-clock per step within 1.3x `B_HEADS`'. Decision rule (mechanical):
`stage1_winner = A_STOPGRAD` if PRIMARY and ORDERING both pass; else `A_GATES` if it passes both; else
`B_HEADS`, and `stage2_required = true`.

**Stage 2 -- CONDITIONAL, authored here now that `frozen.json` records `stage2_required: true`**
(`width.md` "Stage 2 -- CONDITIONAL", tier 1 only, 3 arms x 3 seeds = 9 cells). Purpose: separate
"greedy hurts" from "the moving target hurts" -- under `A_STOPGRAD` each unit fits a predecessor that
is STILL CHANGING (every optimizer step moves every unit's weight simultaneously), which staged
boosting never does (each stage trains against a prefix already frozen at convergence).

  - `A_CORRECTIVE` -- `A_STOPGRAD`'s own loss, PLUS: every `_A_CORRECTIVE_INTERVAL` (2000) epochs,
    `_A_CORRECTIVE_DURATION` (200) optimizer steps on the PLAIN summed loss (no detach, `A_JOINT`'s own
    loss), then resume stop-gradient training -- same net, same optimizer, no reset. Removes
    greediness periodically while keeping the moving target. `NestedWidthNet`, no new class; the ONLY
    new mechanism is `_train_custom_to_convergence`'s optional corrective-phase schedule.
  - `A_STOPGRAD_DISTILL` -- `A_STOPGRAD`'s own predictions (`stopgrad_all_widths_pred`, reused
    unchanged, "costs nothing extra per step"), target for every `k < w_max` replaced by
    `detach(S_{w_max})` (that column's VALUE already equals the plain full-width sum exactly, Step 1's
    own identity test); `k = w_max` keeps the true target `y`. `NestedWidthNet`, no new class.
  - `A_CASCADE_STAGED` -- `cascade_width_net.ResidualCascadeNet` + `train_cascade`, ALREADY
    IMPLEMENTED, reused unchanged except for the sigma-fix port landed alongside this file
    (`cascade_width_net.StageLoss.SQUARED_ERROR`, additions-only -- see that module's docstring). The
    literal staged frozen cascade: each block trains against a CONVERGED, FROZEN prefix, no moving
    target at all -- the arm that isolates whether the moving target (not the greediness) is what hurts
    `A_STOPGRAD`. TIER 1 ONLY -- `run_cell` raises if invoked at tier 2 (the port's own scope is the
    constant-sigma `hetero` toy; a tier-2 per-point-weighted cascade objective was never authorized).
    Has NO shared hidden trunk (each block owns its own `Linear(1,1)`), so `_ordering_statistic`
    returns `None` for it, same as `INDEPENDENT`; and it carries one EXTRA freedom vs `A_STOPGRAD` -- a
    per-prefix readout bias instead of ONE shared bias (`cascade_width_net.py:11-14`) -- named in every
    cell's own JSON via the `cascade_extra_readout_bias_freedom` field (`True` only for this arm) so
    tables carry the caveat without a lookup.

Stage 2 has no pre-registered pass/fail bar of its own (unlike stage 1) -- it is a diagnostic, read by
whoever consumes the 9 cells' JSON, not by a decision rule computed here; `summarize()` is unchanged
beyond automatically picking the new arms up into its existing generic per-(arm, tier) aggregation.

**Non-goals**: no real data, no transformer, no multi-layer net, no variance fitting, no new selection
rule, no change to the toy suite, no promotion of any candidate into the package (WSEL-17), no new
nested-width class, no re-opening of `G-WIDTH = PASS`, no stage-2 grid EXECUTION (the root runs the 9
cells), no stage-3 grid EXECUTION (the root runs the 36 cells; this file only adds `--tier 3` CLI
support -- SANCTIONED 2026-07-22, `width.md`'s stage-3 sign-off ruling: "coverage APPROVED, single-arm
-- `B_HEADS` alone across the 36 tier-3 ladder cells", the driver-extension authoring contract named
there), no stage-2 decision rule (diagnostic only, see above), no redesign of the stop-gradient
frozen-readout-bias quirk on record in `frozen.json`.

Driver CLI contract (root-run grid; this file is never run over the full grid by its author):
  `--arm {b_heads,a_joint,a_stopgrad,a_gates,independent,a_corrective,a_stopgrad_distill,
  a_cascade_staged} --tier {1,2,3} --seed <int> [--tag TAG]` runs ONE cell, writing its per-cell JSON +
  `state_dict` immediately. `--tier 3` additionally requires `--n-train {200,500,1500,4000}` and
  `--sigma {0.05,0.15,0.5}` (`width.md` SS3.8's data x noise ladder; `--n-test` is never a tier-3 flag,
  it stays at the driver default). `a_cascade_staged` only accepts `--tier 1`.
  `--summarize` aggregates every per-cell JSON on disk into `WSEL16/frozen.json`.
  `--selftest` runs tiny cells for every (arm, tier) combo (skipping `a_cascade_staged` x tiers 2/3, the
  combos `run_cell` refuses) plus the stop-grad/weighted-loss identity checks -- no real cell is
  ever run here.

Usage:
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel16.py --selftest
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel16.py --arm b_heads --tier 1 --seed 0
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel16.py --arm b_heads --tier 3 --n-train 1500 --sigma 0.05 --seed 0
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel16.py --summarize
"""

from __future__ import annotations

import argparse
import enum
import json
import math
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root

import cascade_width_net as cascade  # noqa: E402
import converged_width_experiment as cwe  # noqa: E402
import convergence as cvg  # noqa: E402
import kdropout_converged_width_experiment as kce  # noqa: E402
import nested_width_net as nwn  # noqa: E402
import sinc_width_experiment as sw  # noqa: E402
import width_candidates as wc  # noqa: E402
import width_wsel13 as w13  # noqa: E402

from automl_package.models.flexnn.routing import DEFAULT_TOLERANCE as ROUTER_DEFAULT_TOLERANCE  # noqa: E402
from automl_package.models.flexnn.routing import DistilledCapacityRouter  # noqa: E402
from automl_package.utils.capacity_accounting import LOGVAR_HEAD_PATH_SUBSTRING, executed_flops, param_count  # noqa: E402
from automl_package.utils.capacity_selection import cheapest_within_tolerance  # noqa: E402
from automl_package.utils.run_provenance import run_provenance  # noqa: E402

RESULTS_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "WSEL16")

# Certified configuration, reused verbatim (this task does not retune).
W_MAX = cwe.W_MAX  # 12
SEEDS = cwe.SEEDS  # (0, 1, 2)
DEFAULT_MAX_EPOCHS = kce.DEFAULT_MAX_EPOCHS
DEFAULT_CHECK_EVERY = cvg.DEFAULT_CHECK_EVERY
DEFAULT_PATIENCE = cvg.DEFAULT_PATIENCE
DEFAULT_MIN_DELTA = cvg.DEFAULT_MIN_DELTA

_INDEPENDENT_CERT_REF_PATH = os.path.join(
    _EXAMPLES_DIR, "capacity_ladder_results", "W_KDROPOUT_CONVERGED", "w_kdropout_converged_summary_independent_mse.json"
)

# Pre-registered bars (`width.md` "Pre-registered bars" block), fixed BEFORE any run.
_REL_TOL_10PCT = 0.10  # both halt condition (a) (A_JOINT vs B_HEADS, every width) and the PRIMARY bar (full-width only).
_REPRO_RATIO_RTOL = 0.02  # halt (b): |ours - certified| ratio-to-floor tolerance for "reproduces the certified reference" (observed deviations ~1e-3; thread noise ~5% ceiling).
_COST_WALLCLOCK_RATIO_BAR = 1.3

_HETERO3_NOISY_REGION = 2  # nwn.make_hetero3's region id for the noisy-easy tail (HETERO3_NOISY_SIGMA).
_SELFTEST_STOPGRAD_LOOP_TOL = 1e-6  # one-pass vs explicit-loop reference, float32 non-associativity slop.
_SELFTEST_WEIGHTED_LOSS_TOL = 1e-9  # weighted_squared_error vs a hand-computed case, float64 exactness.

# Stage 2's A_CORRECTIVE schedule (`width.md` "Stage 2 -- CONDITIONAL"): stop-gradient epochs between
# corrective phases, and plain-summed-loss optimizer steps per phase. Neither is w_max-dependent -- both
# are fixed spec constants, not something needing generalization under §3.10.
_A_CORRECTIVE_INTERVAL = 2000
_A_CORRECTIVE_DURATION = 200


class Arm(enum.Enum):
    """WSEL-16's eight arms (closed set): five stage-1, three stage-2 (conditional on `stage2_required`)."""

    B_HEADS = "b_heads"  # SharedTrunkPerWidthHeadNet -- THE REFERENCE, unchanged.
    A_JOINT = "a_joint"  # NestedWidthNet, plain summed loss -- NEGATIVE CONTROL, must fail.
    A_STOPGRAD = "a_stopgrad"  # NestedWidthNet, stop-gradient loss (this file's only new LOSS, no new class).
    A_GATES = "a_gates"  # width_candidates.MonotoneGateWidthNet -- the only new CLASS.
    INDEPENDENT = "independent"  # IndependentWidthNet -- POSITIVE CONTROL / ceiling.
    A_CORRECTIVE = "a_corrective"  # STAGE 2: A_STOPGRAD + periodic plain-loss correction. NestedWidthNet, no new class.
    A_STOPGRAD_DISTILL = "a_stopgrad_distill"  # STAGE 2: A_STOPGRAD's structure, target for k<w_max distilled from S_wmax. NestedWidthNet, no new class.
    A_CASCADE_STAGED = "a_cascade_staged"  # STAGE 2: cascade_width_net.ResidualCascadeNet + train_cascade, ALREADY IMPLEMENTED. Tier 1 only.


class Tier(enum.IntEnum):
    """SS3.8's canonical toy suite tiers.

    WSEL-16's own grid: tier 1 + tier 2, 30 cells; tier 3 is the data x noise ladder, CLI support added
    under `width.md`'s 2026-07-22 stage-3 sign-off ruling -- the root, not this driver, decides which
    arms/cells actually run there.
    """

    ONE = 1  # the reference cell -- PRIMARY/ORDERING/COST bars and the Step-3 controls read here.
    TWO = 2  # the noisy-easy control -- PRIMARY/ORDERING bars ALSO hold here (unlike WSEL-13's tier 2).
    THREE = 3  # the data x noise ladder -- no single fixed row (n_train x sigma varies per cell); see _TIER3_* below.


@dataclass(frozen=True)
class _TierConfig:
    """One tier's toy/size config, `width.md` §3.8 lines 437-443."""

    toy: nwn.Toy
    n_train: int
    n_test: int
    sigma: float  # ignored by hetero3 downstream (nwn.make_hetero3 has no sigma arg); kept for uniform bookkeeping.


_TIER_CONFIG: dict[Tier, _TierConfig] = {
    Tier.ONE: _TierConfig(toy=nwn.Toy.HETERO, n_train=1500, n_test=500, sigma=nwn.HETERO_NOISE_SIGMA),
    Tier.TWO: _TierConfig(toy=nwn.Toy.HETERO3, n_train=2250, n_test=750, sigma=nwn.HETERO_NOISE_SIGMA),
    # Tier.THREE has NO entry here -- its n_train/sigma vary per cell (the ladder itself), resolved by
    # _resolve_tier_config() into an ephemeral _TierConfig instead of a fixed row.
}

# Tier 3 (`width.md` §3.8's data x noise ladder): --toy hetero always, --n-test at this driver's own
# default (cwe.N_TEST=500, matching tier 1 and the `_n*_s*_wp4` ledger precedent, which used n_test=500
# at every n_train/sigma combo). --n-train and --sigma are CLOSED SETS, validated at the CLI boundary
# (argparse `choices=`, main()) -- not re-validated inside run_cell(), which only requires them to be
# GIVEN, so --selftest can still exercise the wiring with a tiny synthetic n_train.
_TIER3_TOY = nwn.Toy.HETERO
_TIER3_N_TEST = cwe.N_TEST
_TIER3_N_TRAIN_CHOICES: tuple[int, ...] = (200, 500, 1500, 4000)
_TIER3_SIGMA_CHOICES: tuple[float, ...] = (0.05, 0.15, 0.5)

# Tiers whose training objective is the fixed-sigma WEIGHTED squared error (`width_candidates.
# weighted_squared_error`, §3.7) rather than tier 1's plain `--loss mse`. Tier 3 shares tier 2's
# objective verbatim -- both are `hetero`/`hetero3` cells that need per-point sigma weighting, never
# plain MSE (§3.7: "Tier 2 and Tier 3: a fixed-sigma weighted squared error").
_WEIGHTED_OBJECTIVE_TIERS = (Tier.TWO, Tier.THREE)


def _resolve_tier_config(tier: Tier, n_train: int | None, n_test: int | None, sigma: float | None) -> _TierConfig:
    """Tier's `_TierConfig` -- the fixed row for tiers 1/2, or an ephemeral one built from `n_train`/`sigma` for tier 3.

    Tier 3 has no single row (`width.md` §3.8: it is a 4 n_train x 3 sigma ladder, `--n-test` left at
    this driver's own default) -- `n_train` and `sigma` must both be given by the caller; the closed-set
    membership check itself lives at the CLI boundary (`main()`'s argparse `choices=`), not here, so
    `--selftest` can still pass a tiny synthetic `n_train` through this same path.

    Raises:
        ValueError: `tier is Tier.THREE` and either `n_train` or `sigma` is `None`.
    """
    if tier is Tier.THREE:
        if n_train is None or sigma is None:
            raise ValueError("Tier.THREE (the SS3.8 data x noise ladder) has no single fixed config -- n_train and sigma must both be given explicitly.")
        return _TierConfig(toy=_TIER3_TOY, n_train=n_train, n_test=n_test if n_test is not None else _TIER3_N_TEST, sigma=sigma)
    return _TIER_CONFIG[tier]

# Checkpointing-branch selector for the FOUR arms trained through the EXISTING kce trainer at tier 1
# (arch-agnostic beyond the INDEPENDENT/whole-net split -- WSEL-15's own comment for its arms B/C
# applies here too: this is not an architecture identity check, just which checkpoint branch to take).
_KCE_ARCH_FOR_CHECKPOINTING: dict[Arm, kce.Arch] = {
    Arm.B_HEADS: kce.Arch.SHARED_TRUNK,
    Arm.A_JOINT: kce.Arch.NESTED,
    Arm.A_GATES: kce.Arch.NESTED,
    Arm.INDEPENDENT: kce.Arch.INDEPENDENT,
}


def _build_net(arm: Arm, w_max: int) -> nn.Module:
    """Builds arm's net.

    `A_JOINT`, `A_STOPGRAD`, `A_CORRECTIVE` and `A_STOPGRAD_DISTILL` are all the SAME class
    (`NestedWidthNet`) -- only the training loss differs between them; see the module docstring.
    `A_CASCADE_STAGED` is the one arm with a genuinely different net, `cascade_width_net.
    ResidualCascadeNet` (ALREADY IMPLEMENTED, reused unchanged).
    """
    if arm is Arm.B_HEADS:
        return nwn.SharedTrunkPerWidthHeadNet(w_max=w_max)
    if arm in (Arm.A_JOINT, Arm.A_STOPGRAD, Arm.A_CORRECTIVE, Arm.A_STOPGRAD_DISTILL):
        return nwn.NestedWidthNet(w_max=w_max)
    if arm is Arm.A_GATES:
        return wc.MonotoneGateWidthNet(w_max=w_max)
    if arm is Arm.A_CASCADE_STAGED:
        return cascade.ResidualCascadeNet(w_max=w_max)
    return nwn.IndependentWidthNet(w_max=w_max)


def _params_effective(arm: Arm, w_max: int, params_allocated: int) -> int:
    """Output-layer effective parameter count, extending WSEL-15's `_params_effective_triangle` convention per arm.

    `B_HEADS`: per-width heads mask away input columns `>= k`, so only the `1+2+...+w_max` triangle of
    the `w_max` allocated weight columns per head is ever read at ANY width (`width_wsel15.
    _params_effective_triangle`, reproduced here rather than imported -- that function lives in a driver
    this task does not touch, `width_wsel15.py`). `A_JOINT`/`A_STOPGRAD`/`A_CORRECTIVE`/
    `A_STOPGRAD_DISTILL`: ONE shared `mean_head`, `w_max` weights, every one of which is read by SOME
    width -- no masking waste, effective == `w_max` (the loss differs between these four arms, the
    architecture and its effective-parameter count do not). `A_GATES`: the same `w_max` shared weights
    plus the ONE gate scalar `nu` (always active, no masking waste either) == `w_max + 1`.
    `INDEPENDENT`/`A_CASCADE_STAGED`: no sharing or masking at all -- every allocated parameter (`w_max`
    disjoint sub-nets, or `w_max` disjoint cascade blocks) is read by SOME width/rung -- effective ==
    allocated.
    """
    if arm is Arm.B_HEADS:
        return w_max * (w_max + 1) // 2
    if arm in (Arm.A_JOINT, Arm.A_STOPGRAD, Arm.A_CORRECTIVE, Arm.A_STOPGRAD_DISTILL):
        return w_max
    if arm is Arm.A_GATES:
        return w_max + 1
    return params_allocated  # INDEPENDENT, A_CASCADE_STAGED


# ---------------------------------------------------------------------------
# A_STOPGRAD -- the one-pass stop-gradient loss (`width.md` ~1575-1578). Step 1's identity test target.
# ---------------------------------------------------------------------------


def stopgrad_all_widths_pred(net: nwn.NestedWidthNet, x: torch.Tensor) -> torch.Tensor:
    """`(N, w_max)` table of `detach(S_k - c_k) + c_k` for every `k=1..w_max`, in ONE cumsum pass.

    `S_k = b + sum_{j<=k} c_j` is `NestedWidthNet`'s plain width-k running sum, `c_j = w_j * h_j` unit
    j's own contribution. Algebraically `S_k - c_k = S_{k-1}` (with `S_0 := b`, the empty sum), so this
    computes `detach(S_{k-1}) + c_k` per width directly: unit j's weight `w_j` appears UNDETACHED only
    in column `j-1` (via `c_j`) -- every column `k > j` also contains `c_j` inside `raw_cum`, but that
    whole column is wrapped in `.detach()`. So `w_j`'s gradient comes ONLY from the width-j term, which
    is exactly what "each unit trains only against what the units before it left over" means (Step 1's
    load-bearing identity, `tests/test_stopgrad_width_loss.py`).

    No python loop over widths, no staging, no extra forward -- `hidden(x)` and the cumsum are each
    computed exactly once, covering every width at once (the width analog of `NestedWidthNet.
    all_widths_forward`'s own one-pass cumsum trick, which this function's un-detached twin literally is
    -- see the prove-it-fails test).

    Args:
        net: a `NestedWidthNet` (this arm reuses the SAME class as `A_JOINT` -- no wrapper).
        x: `(N, 1)` standardized input.

    Returns:
        `(N, w_max)`, column `k-1` is width-k's stop-gradient-adjusted prediction.
    """
    h = net.hidden(x)  # (N, w_max)
    contrib = h * net.mean_head.weight.squeeze(0)  # c_j = w_j * h_j, (N, w_max)
    raw_cum = torch.cumsum(contrib, dim=1)  # raw_cum[:, k-1] = sum_{j<=k} c_j
    raw_cum_prev = torch.cat([torch.zeros_like(raw_cum[:, :1]), raw_cum[:, :-1]], dim=1)  # raw_cum shifted right, S_0's sum = 0
    s_prev = raw_cum_prev + net.mean_head.bias  # S_{k-1}, broadcasts the (1,)-shaped bias
    return s_prev.detach() + contrib  # detach(S_{k-1}) + c_k


def _make_standard_total_loss_fn(tier: Tier, sigma_tr: torch.Tensor) -> Callable[[nn.Module, list[int], torch.Tensor, torch.Tensor], torch.Tensor]:
    """`(net, widths, x, y) -> scalar`: `sum_k` plain (tier 1) or fixed-sigma weighted (tier 2) squared error, for any net exposing `forward_width`."""

    def _fn(net: nn.Module, widths: list[int], x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        total = torch.zeros((), device=x.device)
        for k in widths:
            mean, _log_var = net.forward_width(x, k)
            pred = mean.squeeze(1)
            total = total + (wc.weighted_squared_error(pred, y, sigma_tr) if tier in _WEIGHTED_OBJECTIVE_TIERS else ((pred - y) ** 2).mean())
        return total

    return _fn


def _make_stopgrad_total_loss_fn(tier: Tier, sigma_tr: torch.Tensor) -> Callable[[nn.Module, list[int], torch.Tensor, torch.Tensor], torch.Tensor]:
    """`(net, widths, x, y) -> scalar`: `A_STOPGRAD`'s one-pass loss, gathering only the SANDWICH-sampled width columns."""

    def _fn(net: nn.Module, widths: list[int], x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        preds = stopgrad_all_widths_pred(net, x)  # (N, w_max)
        cols = [k - 1 for k in widths]
        sel = preds[:, cols]  # (N, len(widths))
        y_b = y.unsqueeze(1)
        sq = (sel - y_b) ** 2 / sigma_tr.unsqueeze(1) ** 2 if tier in _WEIGHTED_OBJECTIVE_TIERS else (sel - y_b) ** 2
        return sq.mean(dim=0).sum()

    return _fn


def _make_stopgrad_distill_total_loss_fn(tier: Tier, sigma_tr: torch.Tensor, w_max: int) -> Callable[[nn.Module, list[int], torch.Tensor, torch.Tensor], torch.Tensor]:
    """`(net, widths, x, y) -> scalar`: `A_STOPGRAD_DISTILL`'s loss -- `A_STOPGRAD`'s structure, target for `k < w_max` replaced by `detach(S_wmax)`.

    Reuses `stopgrad_all_widths_pred` unchanged ("costs nothing extra per step", `width.md` Stage 2):
    column `w_max-1`'s VALUE already equals the plain (undetached) `S_{w_max}` exactly (Step 1's own
    identity test, part (a) -- `detach()` changes gradients, not values), so `.detach()`-ing that one
    column gives the distillation target for free, no second forward pass. `k = w_max` itself keeps the
    true target `y` (the spec's own carve); every other sampled width regresses toward the frozen
    full-width prediction instead.
    """

    def _fn(net: nn.Module, widths: list[int], x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        preds = stopgrad_all_widths_pred(net, x)  # (N, w_max)
        s_wmax_detached = preds[:, w_max - 1].detach()  # distillation target for every k < w_max
        cols = [k - 1 for k in widths]
        sel = preds[:, cols]  # (N, len(widths))
        targets = torch.stack([y if k == w_max else s_wmax_detached for k in widths], dim=1)
        sq = (sel - targets) ** 2 / sigma_tr.unsqueeze(1) ** 2 if tier in _WEIGHTED_OBJECTIVE_TIERS else (sel - targets) ** 2
        return sq.mean(dim=0).sum()

    return _fn


def _make_per_width_val_fn(tier: Tier, sigma_val: torch.Tensor) -> Callable[[nn.Module, int, torch.Tensor, torch.Tensor], float]:
    """`(net, k, x, y) -> float`: held-out metric checkpointed per width, ARM-AGNOSTIC.

    Stop-gradient changes GRADIENTS, not VALUES (Step 1's identity test, part (a)) -- `net.
    forward_width(x, k)` already returns the plain `S_k` for `A_STOPGRAD`'s `NestedWidthNet`, so the
    SAME plain-or-weighted MSE serves every arm, including `A_STOPGRAD`.
    """

    def _fn(net: nn.Module, k: int, x: torch.Tensor, y: torch.Tensor) -> float:
        mean, _log_var = net.forward_width(x, k)
        pred = mean.squeeze(1)
        loss = wc.weighted_squared_error(pred, y, sigma_val) if tier in _WEIGHTED_OBJECTIVE_TIERS else ((pred - y) ** 2).mean()
        return float(loss.item())

    return _fn


def _sigma_true_tensor(toy: nwn.Toy, region: np.ndarray, sigma: float, norm: dict) -> torch.Tensor:
    """Per-point TRUE noise sigma (`width.md` §3.7), IN STANDARDIZED-y UNITS (`/ norm["sy"]`) -- never estimated, never learned.

    `hetero`: `sigma` (the toy's own constant) everywhere -- both regions share it. `hetero3`: region 2
    (noisy-easy) is `HETERO3_NOISY_SIGMA`, regions 0/1 are `HETERO_NOISE_SIGMA` -- read from the
    generator's own `region` output, never re-derived by thresholding `x`. A pure linear rescale of `y`
    (`y_std = (y - my) / sy`) rescales its noise std by the same factor `sy`, so `sigma_true` must be
    divided by it too or the weighted loss trains on a mismatched scale.
    """
    if toy is nwn.Toy.HETERO3:
        sigma_raw = np.where(region == _HETERO3_NOISY_REGION, nwn.HETERO3_NOISY_SIGMA, nwn.HETERO_NOISE_SIGMA)
    else:
        sigma_raw = np.full(region.shape, sigma, dtype=np.float64)
    return torch.as_tensor(sigma_raw / norm["sy"], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Generic trainer -- needed for A_STOPGRAD (any tier), tier 2 (any arm), and STAGE 2's A_CORRECTIVE /
# A_STOPGRAD_DISTILL (tier 1); see module docstring.
# ---------------------------------------------------------------------------


def _train_custom_to_convergence(
    net: nn.Module,
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    total_loss_fn: Callable[[nn.Module, list[int], torch.Tensor, torch.Tensor], torch.Tensor],
    per_width_val_fn: Callable[[nn.Module, int, torch.Tensor, torch.Tensor], float],
    w_max: int,
    max_epochs: int,
    check_every: int,
    patience: int,
    min_delta: float,
    seed: int,
    independent_checkpointing: bool,
    corrective_loss_fn: Callable[[nn.Module, list[int], torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    corrective_interval: int = 0,
    corrective_duration: int = 0,
) -> dict[int, cvg.ConvergenceResult]:
    """Generic SANDWICH-schedule k-dropout trainer, gated by PER-WIDTH held-out convergence.

    A near-duplicate of `kce._train_kdropout_to_convergence`'s training loop, needed because that
    function's per-width loss is hardcoded to its own `LossType` (NLL/MSE) dispatch and cannot express
    (a) the tier-2 fixed-sigma WEIGHTED objective (`width.md` §3.7 -- no `LossType` member for it, and
    `kdropout_converged_width_experiment.py` is out of this task's write set), or (b) `A_STOPGRAD`'s
    per-width total-loss SHAPE, which is not a sum of independently-detachable `forward_width(x, k)`
    calls (see `stopgrad_all_widths_pred`). The SANDWICH width draw below (`widths = [1, w_max] + 2
    random mid`) is reproduced VERBATIM from `kce._train_kdropout_to_convergence` -- not imported: that
    function does not factor its draw out into a standalone helper, and this task's write set excludes
    editing it. Same RNG usage, same optimizer (`torch.optim.Adam`, `cwe.LR`), same
    `cvg.ConvergenceTracker` convergence gate, unchanged.

    Args:
        net: the net to train in place (`_build_net`'s output for this arm).
        x_tr: standardized training inputs, shape `(N, 1)`.
        y_tr: standardized training targets, shape `(N,)`.
        x_val: standardized held-out inputs used only for convergence monitoring.
        y_val: standardized held-out targets used only for convergence monitoring.
        total_loss_fn: `(net, widths, x, y) -> scalar` -- ONE training step's total loss over the
            SANDWICH-sampled widths (arm+tier specific; see `_make_standard_total_loss_fn`/
            `_make_stopgrad_total_loss_fn`).
        per_width_val_fn: `(net, k, x, y) -> float` -- held-out metric checkpointed per width, the SAME
            for every arm (`_make_per_width_val_fn`).
        w_max: maximum hidden width (the largest prefix the net can express).
        max_epochs: safety cap on optimizer steps (== epochs; full-batch).
        check_every: epochs between per-width held-out checkpoints.
        patience: consecutive flat checkpoints that declare one width converged.
        min_delta: held-out-loss decrease counted as a real improvement.
        seed: RNG seed for the per-step middle-width draw.
        independent_checkpointing: `True` for `Arm.INDEPENDENT` (disjoint sub-nets restore their OWN
            best state independently, matching `kce._train_kdropout_to_convergence`'s `Arch.INDEPENDENT`
            branch); `False` restores the WHOLE net at its best mean-per-width-validation epoch.
        corrective_loss_fn: `A_CORRECTIVE`'s only extra mechanism (`width.md` Stage 2): if given, the
            training loop periodically substitutes THIS loss for `total_loss_fn` on the SAME
            sandwich-sampled widths, same optimizer, no state reset. `None` (default) reproduces the
            EXACT prior behavior -- `A_STOPGRAD` and every tier-2 arm's existing call sites pass none of
            the three `corrective_*` args, so this is byte-identical for them.
        corrective_interval: epochs of `total_loss_fn` training between corrective phases. Ignored if
            `corrective_loss_fn` is `None`; required `> 0` otherwise.
        corrective_duration: consecutive `corrective_loss_fn` steps per phase. Same requirement.

    Returns:
        `{width -> ConvergenceResult}`, best weights already restored.
    """
    if corrective_loss_fn is not None and (corrective_interval <= 0 or corrective_duration <= 0):
        raise ValueError("corrective_loss_fn requires corrective_interval > 0 and corrective_duration > 0")
    corrective_block = corrective_interval + corrective_duration  # unused (corrective_loss_fn is None) => never divided by zero below.

    opt = torch.optim.Adam(net.parameters(), lr=cwe.LR)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    mid_candidates = list(range(2, w_max))
    n_mid_draw = min(2, len(mid_candidates))

    trackers = {k: cvg.ConvergenceTracker(patience=patience, min_delta=min_delta) for k in range(1, w_max + 1)}
    best_states: dict[int, dict | None] = dict.fromkeys(range(1, w_max + 1))
    best_mean_val = math.inf
    best_net_state: dict | None = None

    net.train()
    final_epoch = max_epochs
    for epoch in range(1, max_epochs + 1):
        opt.zero_grad()
        widths = [1, w_max]
        if n_mid_draw:
            perm = torch.randperm(len(mid_candidates), generator=gen)[:n_mid_draw]
            widths += [mid_candidates[i] for i in perm.tolist()]
        # A_CORRECTIVE's periodic plain-loss phase: epochs [corrective_interval, corrective_block) of
        # every corrective_block-epoch cycle use corrective_loss_fn instead of total_loss_fn -- "every
        # corrective_interval epochs [of total_loss_fn], corrective_duration steps of corrective_loss_fn,
        # then resume" (`width.md` Stage 2's own wording). corrective_loss_fn is None for every OTHER
        # arm, so `in_corrective_phase` is always False there and this branch never fires.
        in_corrective_phase = corrective_loss_fn is not None and (epoch - 1) % corrective_block >= corrective_interval
        step_loss_fn = corrective_loss_fn if in_corrective_phase else total_loss_fn
        total_loss = step_loss_fn(net, widths, x_tr, y_tr)
        total_loss.backward()
        opt.step()

        if epoch % check_every == 0:
            net.eval()
            with torch.no_grad():
                per_width_val = {k: per_width_val_fn(net, k, x_val, y_val) for k in range(1, w_max + 1)}
            if independent_checkpointing:
                for k, v in per_width_val.items():
                    if trackers[k].update(epoch, v):
                        best_states[k] = {n: t.detach().clone() for n, t in net.subnets[k - 1].state_dict().items()}
            else:
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

    if independent_checkpointing:
        for k in range(1, w_max + 1):
            if best_states[k] is not None:
                net.subnets[k - 1].load_state_dict(best_states[k])
    elif best_net_state is not None:
        net.load_state_dict(best_net_state)
    net.eval()
    return {k: trackers[k].result(final_epoch=final_epoch) for k in range(1, w_max + 1)}


# ---------------------------------------------------------------------------
# Ordering statistic -- imported from width_wsel13.py (its private OLS/greedy helpers + split
# fractions), generalized ONLY on "apply the width-w_max readout to an (possibly ablated) hidden
# vector" -- width_wsel13.py hardcodes `net.mean_heads[w_max-1]`, a SharedTrunkPerWidthHeadNet-specific
# attribute.
# ---------------------------------------------------------------------------


def _widest_head_from_hidden(net: nn.Module, arm: Arm, h: torch.Tensor) -> torch.Tensor:
    """Applies arm's width-w_max readout to an ALREADY-COMPUTED (possibly unit-ablated) hidden vector `h`.

    At `k=w_max` the prefix mask is all-ones, so no masking is needed (matches `width_wsel13.run_cell`'s
    own reasoning for its `net.mean_heads[w_max-1](h_report)` shortcut).
    """
    if arm is Arm.B_HEADS:
        return net.mean_heads[net.w_max - 1](h)
    if arm is Arm.A_GATES:
        contrib = h * net.base.mean_head.weight.squeeze(0) * net._gates(h.dtype, h.device)
        return contrib.sum(dim=1, keepdim=True) + net.base.mean_head.bias
    # A_JOINT, A_STOPGRAD, A_CORRECTIVE, A_STOPGRAD_DISTILL -- plain NestedWidthNet, ONE shared mean_head.
    return net.mean_head(h)


def _ordering_statistic(
    net: nn.Module,
    arm: Arm,
    w_max: int,
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    x_select: torch.Tensor,
    y_select: torch.Tensor,
    x_report: torch.Tensor,
    y_report: torch.Tensor,
) -> dict | None:
    """WSEL-13's ablation-Spearman + prefix-vs-greedy ordering diagnostic, read off THIS arm's frozen trunk.

    Reuses `width_wsel13.py`'s private split-fraction/OLS/greedy-selection helpers UNCHANGED (imported,
    never reimplemented -- that file is read-only for this task) on the SAME three-way FIT/SELECT/REPORT
    carve, so the two tasks' numbers sit on the same statistic.

    Returns `None` for `Arm.INDEPENDENT` and `Arm.A_CASCADE_STAGED`: neither has a SINGLE SHARED hidden
    trunk to ablate a unit in or slice a prefix of -- `IndependentWidthNet` is `w_max` disjoint sub-nets,
    each its own `Linear(1, k)`; `ResidualCascadeNet` is `w_max` disjoint per-block trunks
    (`cascade_width_net.ResidualCascadeNet.blocks`, each `Linear(1, 1)`). The diagnostic's premise (a
    shared-trunk prefix property) does not hold for either architecture.
    """
    if arm in (Arm.INDEPENDENT, Arm.A_CASCADE_STAGED):
        return None
    net.eval()
    with torch.no_grad():
        h_fit = net.hidden(x_fit)
        h_select = net.hidden(x_select)
        h_report = net.hidden(x_report)

        baseline_pred = _widest_head_from_hidden(net, arm, h_report).squeeze(1)
        baseline_mse = float(torch.mean((baseline_pred - y_report) ** 2).item())
        importance_by_unit: dict[int, float] = {}
        for j in range(1, w_max + 1):
            h_ablated = h_report.clone()
            h_ablated[:, j - 1] = 0.0
            pred_j = _widest_head_from_hidden(net, arm, h_ablated).squeeze(1)
            mse_j = float(torch.mean((pred_j - y_report) ** 2).item())
            importance_by_unit[j] = mse_j - baseline_mse

    spearman_res = spearmanr(list(range(1, w_max + 1)), [importance_by_unit[j] for j in range(1, w_max + 1)])

    h_fit_np, y_fit_np = h_fit.numpy(), y_fit.numpy()
    h_select_np, y_select_np = h_select.numpy(), y_select.numpy()
    h_report_np, y_report_np = h_report.numpy(), y_report.numpy()

    prefix_report_mse: dict[int, float] = {}
    for k in range(1, w_max + 1):
        cols = list(range(k))
        beta = w13._fit_ols(h_fit_np[:, cols], y_fit_np)
        prefix_report_mse[k] = w13._ols_mse(h_report_np[:, cols], y_report_np, beta)

    greedy_order, _greedy_select_mse_by_step = w13._greedy_forward_selection(h_fit_np, y_fit_np, h_select_np, y_select_np, w_max)
    greedy_report_mse: dict[int, float] = {}
    for k in range(1, w_max + 1):
        cols = greedy_order[:k]
        beta = w13._fit_ols(h_fit_np[:, cols], y_fit_np)
        greedy_report_mse[k] = w13._ols_mse(h_report_np[:, cols], y_report_np, beta)

    relative_gaps = [(prefix_report_mse[k] - greedy_report_mse[k]) / max(greedy_report_mse[k], w13._MIN_DENOM) for k in range(1, w_max + 1)]

    return {
        "baseline_report_mse": baseline_mse,
        "importance_by_unit": {str(j): v for j, v in importance_by_unit.items()},
        "spearman_index_vs_importance": {"rho": float(spearman_res.statistic), "p": float(spearman_res.pvalue)},
        "prefix_report_mse_by_k": {str(k): v for k, v in prefix_report_mse.items()},
        "greedy_report_mse_by_k": {str(k): v for k, v in greedy_report_mse.items()},
        "greedy_order_0indexed": greedy_order,
        "mean_relative_prefix_gap": float(np.mean(relative_gaps)),
    }


# ---------------------------------------------------------------------------
# Width selection -- BOTH rules the strand binds (`width.md` §1): the global cheapest-within-tolerance
# reader, and the distilled per-input router. Both reused from shared machinery, never reimplemented.
# ---------------------------------------------------------------------------


def _global_selected_width(net: nn.Module, w_max: int, x_report: torch.Tensor, y_report: torch.Tensor) -> dict:
    """The strand's binding GLOBAL rule (`width.md` §1): cheapest-within-tolerance at 2x bootstrap SE over the held-out per-width squared-error curve."""
    net.eval()
    with torch.no_grad():
        cols = []
        for k in range(1, w_max + 1):
            mean, _log_var = net.forward_width(x_report, k)
            cols.append(((mean.squeeze(1) - y_report) ** 2).numpy())
    error_table = np.stack(cols, axis=1)  # (N, w_max), column 0 = width 1 (cheapest first, capacity_selection's contract)
    idx = cheapest_within_tolerance(error_table)
    return {"selected_width": idx + 1, "held_out_curve_by_width": {str(k): float(error_table[:, k - 1].mean()) for k in range(1, w_max + 1)}}


def _distilled_router_selected_width(net: nn.Module, w_max: int, x_fit: torch.Tensor, y_fit: torch.Tensor, x_route: torch.Tensor, seed: int) -> dict:
    """The strand's PER-INPUT rule (`width.md` §1): `DistilledCapacityRouter` at its own flat `DEFAULT_TOLERANCE=0.25`.

    A DIFFERENT tolerance rule from the global one, deliberately (§1's "W-PERINPUT runs on a DIFFERENT
    tolerance rule" note).

    Fits on `(x_fit, y_fit)` (the driver's phase-1 validation split -- held out from training, disjoint
    from the report split), routes `x_route` (the ordering statistic's REPORT split, touched by nothing
    else). A per-input rule produces a DISTRIBUTION of widths, not one scalar the way the global rule
    does, so the mean plus the full histogram is what Step 5 asks this arm's field to carry.
    """
    x_fit_np = x_fit.numpy().reshape(-1, 1)
    y_fit_np = y_fit.numpy()
    x_route_np = x_route.numpy().reshape(-1, 1)
    capacity_grid = [(k,) for k in range(1, w_max + 1)]

    net.eval()

    def eval_fn(x: np.ndarray, capacity: tuple[int, ...]) -> np.ndarray:
        k = capacity[0]
        x_t = torch.as_tensor(x, dtype=torch.float32).reshape(-1, 1)
        with torch.no_grad():
            mean, _log_var = net.forward_width(x_t, k)
        return (mean.squeeze(1).numpy() - y_fit_np) ** 2

    def cost_fn(capacity: tuple[int, ...]) -> float:
        return float(executed_flops(net, capacity[0]))

    router = DistilledCapacityRouter(device="cpu", seed=seed)
    router.fit(eval_fn=eval_fn, x_val=x_fit_np, y_val=y_fit_np, capacity_grid=capacity_grid, tolerance=ROUTER_DEFAULT_TOLERANCE, cost_fn=cost_fn)
    routed_widths = [c[0] for c in router.route(x_route_np)]
    return {
        "tolerance": ROUTER_DEFAULT_TOLERANCE,
        "mean_selected_width": float(np.mean(routed_widths)),
        "width_counts": {str(k): routed_widths.count(k) for k in range(1, w_max + 1)},
    }


# ---------------------------------------------------------------------------
# One cell.
# ---------------------------------------------------------------------------


def run_cell(
    arm: Arm,
    tier: Tier,
    seed: int,
    *,
    w_max: int = W_MAX,
    n_train: int | None = None,
    n_test: int | None = None,
    sigma: float | None = None,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    check_every: int = DEFAULT_CHECK_EVERY,
    patience: int = DEFAULT_PATIENCE,
    min_delta: float = DEFAULT_MIN_DELTA,
    corrective_interval: int | None = None,
    corrective_duration: int | None = None,
) -> tuple[dict, dict]:
    """Trains one (arm, tier, seed) cell to per-width convergence, then records every Step-5 field.

    Data prep / phase-1 train-val carve mirrors `kdropout_converged_width_experiment.run_case` /
    `width_wsel13.run_cell` / `width_wsel15.run_cell` verbatim -- reimplemented here (not called)
    because this task needs the trained net object directly for the ordering diagnostic, the width
    selectors, and `capacity_accounting`, none of which `run_case` returns. The TRAINING LOOP itself is
    never hand-rolled per cell: tier-1 arms among `{B_HEADS, A_JOINT, A_GATES, INDEPENDENT}` go through
    the EXISTING `kce._train_kdropout_to_convergence`; `A_CASCADE_STAGED` goes through the ALREADY-
    IMPLEMENTED `cascade.train_cascade` (tier 1 only); everything else (`A_STOPGRAD`/`A_CORRECTIVE`/
    `A_STOPGRAD_DISTILL` at any tier, and tiers 2/3 for the four kce-trained arms) goes through
    `_train_custom_to_convergence` (see module docstring for why) -- tier 3 shares tier 2's weighted
    objective verbatim (`_WEIGHTED_OBJECTIVE_TIERS`), never the kce trainer's MSE-only path.

    Args:
        arm: which of the 8 arms to train (5 stage-1, 3 stage-2).
        tier: which SS3.8 canonical-suite tier (1 = reference, 2 = noisy-easy control, 3 = the data x
            noise ladder -- no single fixed row, see `n_train`/`sigma` below).
        seed: RNG seed for this cell (canonical suite: 0, 1, 2).
        w_max: maximum hidden width.
        n_train: overrides `_TIER_CONFIG[tier].n_train` for tiers 1/2 (used by `--selftest` for a tiny
            toy; `None` = the tier's canonical value). REQUIRED (not `None`) for `tier is Tier.THREE`,
            which has no canonical value to fall back to -- see `_resolve_tier_config`.
        n_test: overrides `_TIER_CONFIG[tier].n_test` for tiers 1/2, same convention; `None` at tier 3
            falls back to `_TIER3_N_TEST` (this driver's own default), not a per-cell requirement.
        sigma: tier-3-only override of the cell's noise sigma (ignored for tiers 1/2, whose sigma is
            fixed by `_TIER_CONFIG`). REQUIRED (not `None`) for `tier is Tier.THREE`.
        max_epochs: safety cap on optimizer steps (convergence decides the real stop).
        check_every: epochs between per-width held-out checkpoints.
        patience: consecutive flat checkpoints that declare one width converged.
        min_delta: held-out-loss decrease counted as a real improvement.
        corrective_interval: `A_CORRECTIVE`-only (`width.md` Stage 2); `None` = `_A_CORRECTIVE_INTERVAL`
            (the real 2000-epoch spec value). Overridable so `--selftest` can exercise the corrective
            phase inside its tiny epoch budget without touching the real constant.
        corrective_duration: `A_CORRECTIVE`-only, same override convention; `None` =
            `_A_CORRECTIVE_DURATION`.

    Returns:
        `(case, state_dict)` -- `case` is the JSON-able per-cell record; `state_dict` is the trained
        net's own (already best-checkpoint-restored) parameters.

    Raises:
        ValueError: `tier is Tier.THREE` and `n_train`/`sigma` are not both given (see
            `_resolve_tier_config`); or `arm is Arm.A_CASCADE_STAGED` and `tier is not Tier.ONE` -- the
            cascade's squared-error stage-loss port (`cascade_width_net.StageLoss.SQUARED_ERROR`) is
            scoped to tier 1's constant-sigma `hetero` toy only; a tier-2/tier-3 per-point-weighted
            cascade objective was never authorized (`width.md` Stage 2: "tier 1 only, 3 arms x 3 seeds").
    """
    cfg = _resolve_tier_config(tier, n_train, n_test, sigma)
    n_tr = n_train if n_train is not None else cfg.n_train
    n_te = n_test if n_test is not None else cfg.n_test

    if cfg.toy is nwn.Toy.HETERO3:
        x_tr, y_tr, region_tr = nwn.make_hetero3(n_tr, seed)
        x_te, y_te, region_te = nwn.make_hetero3(n_te, seed + 500)
    else:
        x_tr, y_tr, region_tr = nwn.make_hetero(n_tr, seed, sigma=cfg.sigma)
        x_te, y_te, region_te = nwn.make_hetero(n_te, seed + 500, sigma=cfg.sigma)

    p1_idx = np.arange(0, n_tr, 2)
    x_p1, y_p1, region_p1 = x_tr[p1_idx], y_tr[p1_idx], region_tr[p1_idx]
    val_mask = (np.arange(len(x_p1)) % cwe.VAL_EVERY) == 0
    norm = cwe._standardize_fit(x_p1[~val_mask], y_p1[~val_mask])
    x_tr_t, y_tr_t = cwe._to_std_tensors(x_p1[~val_mask], y_p1[~val_mask], norm)
    x_val_t, y_val_t = cwe._to_std_tensors(x_p1[val_mask], y_p1[val_mask], norm)
    x_te_t, y_te_t = cwe._to_std_tensors(x_te, y_te, norm)
    region_tr_split = region_p1[~val_mask]
    region_val_split = region_p1[val_mask]

    sigma_tr = _sigma_true_tensor(cfg.toy, region_tr_split, cfg.sigma, norm)
    sigma_val = _sigma_true_tensor(cfg.toy, region_val_split, cfg.sigma, norm)

    torch.manual_seed(seed)
    net = _build_net(arm, w_max)

    _train_t0 = time.perf_counter()
    if arm is Arm.A_CASCADE_STAGED:
        if tier is not Tier.ONE:
            raise ValueError(
                "A_CASCADE_STAGED's squared-error stage-loss port is scoped to tier 1 only "
                "(width.md Stage 2: 'tier 1 only, 3 arms x 3 seeds') -- neither tier 2 nor tier 3 was ever authorized for this arm."
            )
        stage_results = cascade.train_cascade(
            net,
            x_tr_t,
            y_tr_t,
            x_val_t,
            y_val_t,
            seed=seed,
            max_epochs=max_epochs,
            check_every=check_every,
            patience=patience,
            min_delta=min_delta,
            stage_loss=cascade.StageLoss.SQUARED_ERROR,
            sigma_true=None,  # tier 1 hetero is constant-sigma -- PLAIN squared error, `width.md` §3.7.
        )
        conv = {b: stage_results[b]["conv"] for b in range(1, w_max + 1)}  # stage b IS rung/width b.
    elif arm is Arm.A_STOPGRAD or arm is Arm.A_STOPGRAD_DISTILL or arm is Arm.A_CORRECTIVE or tier in _WEIGHTED_OBJECTIVE_TIERS:
        if arm is Arm.A_STOPGRAD_DISTILL:
            total_loss_fn = _make_stopgrad_distill_total_loss_fn(tier, sigma_tr, w_max)
            corrective_kwargs = {}
        elif arm is Arm.A_CORRECTIVE:
            total_loss_fn = _make_stopgrad_total_loss_fn(tier, sigma_tr)
            corrective_kwargs = {
                "corrective_loss_fn": _make_standard_total_loss_fn(tier, sigma_tr),
                "corrective_interval": corrective_interval if corrective_interval is not None else _A_CORRECTIVE_INTERVAL,
                "corrective_duration": corrective_duration if corrective_duration is not None else _A_CORRECTIVE_DURATION,
            }
        elif arm is Arm.A_STOPGRAD:
            total_loss_fn = _make_stopgrad_total_loss_fn(tier, sigma_tr)
            corrective_kwargs = {}
        else:
            total_loss_fn = _make_standard_total_loss_fn(tier, sigma_tr)
            corrective_kwargs = {}
        per_width_val_fn = _make_per_width_val_fn(tier, sigma_val)
        conv = _train_custom_to_convergence(
            net,
            x_tr_t,
            y_tr_t,
            x_val_t,
            y_val_t,
            total_loss_fn=total_loss_fn,
            per_width_val_fn=per_width_val_fn,
            w_max=w_max,
            max_epochs=max_epochs,
            check_every=check_every,
            patience=patience,
            min_delta=min_delta,
            seed=seed,
            independent_checkpointing=(arm is Arm.INDEPENDENT),
            **corrective_kwargs,
        )
    else:
        conv, _best_epoch = kce._train_kdropout_to_convergence(
            net,
            x_tr_t,
            y_tr_t,
            x_val_t,
            y_val_t,
            arch=_KCE_ARCH_FOR_CHECKPOINTING[arm],
            loss=kce.LossType.MSE,
            max_epochs=max_epochs,
            check_every=check_every,
            patience=patience,
            min_delta=min_delta,
            seed=seed,
            schedule=nwn.WidthSchedule.SANDWICH,
        )
    train_wall_clock_s = time.perf_counter() - _train_t0

    n_trustworthy = sum(1 for r in conv.values() if r.trustworthy)
    all_trustworthy = n_trustworthy == w_max
    hit_cap = any(r.hit_cap for r in conv.values())
    steps_to_converge = max(r.stop_epoch for r in conv.values())

    net.eval()
    with torch.no_grad():
        train_mse_by_width = {k: float(nwn._width_mse(net, k, x_tr_t, y_tr_t).item()) for k in range(1, w_max + 1)}
        held_out_mse_by_width = {k: float(nwn._width_mse(net, k, x_val_t, y_val_t).item()) for k in range(1, w_max + 1)}
    train_minus_heldout_gap_by_width = {k: train_mse_by_width[k] - held_out_mse_by_width[k] for k in range(1, w_max + 1)}
    full_width_held_out_mse = held_out_mse_by_width[w_max]

    params_allocated = param_count(net, path_filter=LOGVAR_HEAD_PATH_SUBSTRING)
    params_effective = _params_effective(arm, w_max, params_allocated)
    executed_flops_by_width = {k: executed_flops(net, k) for k in range(1, w_max + 1)}

    # fit_bar -- original y-units, region-conditioned, reused sw functions verbatim (halt condition (b) reads this).
    floor_hard = nwn.HETERO_NOISE_SIGMA**2 if cfg.toy is nwn.Toy.HETERO3 else cfg.sigma**2
    err2_te = sw._score_all_widths_mse(net, norm, x_te, y_te, "cpu")
    err2_by_width_te = {k: err2_te[:, k - 1] for k in range(1, w_max + 1)}
    fit_bar = sw._fit_bar_mse(err2_by_width_te, region_te, w_max, floor_hard)

    # Ordering statistic (WSEL-13, imported) -- three-way FIT/SELECT/REPORT carve of the driver's OWN
    # held-out test set, exactly width_wsel13.run_cell's split shape.
    x_rem, x_fit_raw, y_rem, y_fit_raw = train_test_split(x_te, y_te, test_size=w13._HELDOUT_FIT_FRACTION, random_state=seed)
    x_select_raw, x_report_raw, y_select_raw, y_report_raw = train_test_split(x_rem, y_rem, test_size=w13._HELDOUT_SELECT_REPORT_SPLIT, random_state=seed)
    x_fit_t, y_fit_t = cwe._to_std_tensors(x_fit_raw, y_fit_raw, norm)
    x_select_t, y_select_t = cwe._to_std_tensors(x_select_raw, y_select_raw, norm)
    x_report_t, y_report_t = cwe._to_std_tensors(x_report_raw, y_report_raw, norm)
    ordering_statistic = _ordering_statistic(net, arm, w_max, x_fit_t, y_fit_t, x_select_t, y_select_t, x_report_t, y_report_t)

    # Width selection -- BOTH rules (`width.md` §1). Global reads the driver's whole held-out test set;
    # the router fits on the phase-1 val split and routes the report split (see that function's docstring).
    selected_width_global = _global_selected_width(net, w_max, x_te_t, y_te_t)
    selected_width_distilled_router = _distilled_router_selected_width(net, w_max, x_val_t, y_val_t, x_report_t, seed)

    case = {
        "arm": arm.value,
        "tier": tier.value,
        "seed": seed,
        "toy": cfg.toy.value,
        "n_train": n_tr,
        "n_test": n_te,
        "sigma": cfg.sigma,
        "w_max": w_max,
        "convergence": {k: r.summary() for k, r in conv.items()},
        "n_widths_trustworthy": n_trustworthy,
        "all_widths_trustworthy": all_trustworthy,
        "hit_cap": hit_cap,
        "train_wall_clock_s": train_wall_clock_s,
        "steps_to_converge": steps_to_converge,
        "train_mse_by_width": {str(k): v for k, v in train_mse_by_width.items()},
        "held_out_mse_by_width": {str(k): v for k, v in held_out_mse_by_width.items()},
        "full_width_held_out_mse": full_width_held_out_mse,
        "train_minus_heldout_gap_by_width": {str(k): v for k, v in train_minus_heldout_gap_by_width.items()},
        "params_allocated": params_allocated,
        "params_effective": params_effective,
        "executed_flops_by_width": {str(k): v for k, v in executed_flops_by_width.items()},
        "fit_bar": fit_bar,
        "ordering_statistic": ordering_statistic,
        "selected_width_global": selected_width_global,
        "selected_width_distilled_router": selected_width_distilled_router,
        "training_schedule": nwn.WidthSchedule.SANDWICH.value,
        # True only for A_CASCADE_STAGED: its per-prefix readout bias (one bias PER BLOCK) is an extra
        # freedom vs A_STOPGRAD's ONE shared bias (`cascade_width_net.py:11-14`) -- named here so any
        # table that tabulates the two arms together carries the caveat without a separate lookup.
        "cascade_extra_readout_bias_freedom": arm is Arm.A_CASCADE_STAGED,
        "provenance": run_provenance(),
    }
    return case, {name: t.detach().clone() for name, t in net.state_dict().items()}


def _tier3_suffix(n_train: int | None, sigma: float | None) -> str:
    """`_n{n_train}_s{sigma}` filename suffix, present ONLY when both are given (tier 3's ladder cells).

    Tiers 1/2 never pass `n_train`/`sigma` here, so their filenames are byte-identical to before this
    task -- this is precisely what keeps a tier-3 cell from ever clobbering a tier-1/2 canonical one,
    since no tier-1/2 filename contains `_n`/`_s` segments.
    """
    suffix = ""
    if n_train is not None:
        suffix += f"_n{n_train}"
    if sigma is not None:
        suffix += f"_s{sigma}"
    return suffix


def _cell_json_path(arm: Arm, tier: Tier, seed: int, tag: str | None = None, *, n_train: int | None = None, sigma: float | None = None) -> str:
    base = f"wsel16_{arm.value}_tier{tier.value}_seed{seed}" + _tier3_suffix(n_train, sigma)
    if tag:
        base += f"_{tag}"
    return os.path.join(RESULTS_DIR, base + ".json")


def _state_path(arm: Arm, tier: Tier, seed: int, tag: str | None = None, *, n_train: int | None = None, sigma: float | None = None) -> str:
    base = f"state_{arm.value}_tier{tier.value}_seed{seed}" + _tier3_suffix(n_train, sigma)
    if tag:
        base += f"_{tag}"
    return os.path.join(RESULTS_DIR, base + ".pt")


def _jsonable(obj: object) -> object:
    """Minimal JSON-safe coercion (numpy/torch scalars, dict int-keys) -- local twin of `width_wsel13._jsonable`."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj


# ---------------------------------------------------------------------------
# --summarize: Step-3 controls + pre-registered bars + the mechanical decision rule.
# ---------------------------------------------------------------------------


def _mean_by_width(cells: dict[int, dict], field: str, w_max: int) -> dict[str, float] | None:
    if not cells:
        return None
    return {str(k): float(np.mean([c[field][str(k)] for c in cells.values()])) for k in range(1, w_max + 1)}


def _mean_scalar(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _arm_tier_block(cells: dict[int, dict], w_max: int) -> dict:
    """One (arm, tier)'s aggregate over however many seeds are present on disk."""
    rhos = [c["ordering_statistic"]["spearman_index_vs_importance"]["rho"] for c in cells.values() if c.get("ordering_statistic") is not None]
    fit_ratios = [c["fit_bar"]["ratio_to_floor"] for c in cells.values()]
    return {
        "n_cells_present": len(cells),
        "held_out_mse_by_width_mean": _mean_by_width(cells, "held_out_mse_by_width", w_max),
        "full_width_held_out_mse_mean": _mean_scalar([c["full_width_held_out_mse"] for c in cells.values()]),
        "full_width_held_out_mse_by_seed": {str(s): c["full_width_held_out_mse"] for s, c in cells.items()},
        "train_wall_clock_s_mean": _mean_scalar([c["train_wall_clock_s"] for c in cells.values()]),
        "steps_to_converge_mean": _mean_scalar([c["steps_to_converge"] for c in cells.values()]),
        "params_allocated": next(iter(cells.values()))["params_allocated"] if cells else None,
        "params_effective": next(iter(cells.values()))["params_effective"] if cells else None,
        "ordering_spearman_rho_mean": _mean_scalar(rhos) if rhos else None,
        "fit_bar_ratio_to_floor_mean": _mean_scalar(fit_ratios),
        "all_widths_trustworthy_all_seeds": all(c["all_widths_trustworthy"] for c in cells.values()) if cells else None,
        "hit_cap_any_seed": any(c["hit_cap"] for c in cells.values()) if cells else None,
    }


def _check_controls(per_cell: dict[Arm, dict[Tier, dict[int, dict]]]) -> dict:
    """Step 3's two HALT conditions, read from whatever tier-1 cells are on disk (`width.md` "Step 3").

    Reporting a triggered halt is this driver's job; ACTING on it (stopping the grid, escalating to the
    user) is the ROOT's -- this function only computes the check.
    """
    a_joint_cells = per_cell[Arm.A_JOINT][Tier.ONE]
    b_heads_cells = per_cell[Arm.B_HEADS][Tier.ONE]
    independent_cells = per_cell[Arm.INDEPENDENT][Tier.ONE]

    halt_a = None
    if a_joint_cells and b_heads_cells:
        a_mean = _mean_by_width(a_joint_cells, "held_out_mse_by_width", W_MAX)
        b_mean = _mean_by_width(b_heads_cells, "held_out_mse_by_width", W_MAX)
        rel_gap_by_width = {k: abs(a_mean[k] - b_mean[k]) / b_mean[k] for k in a_mean}
        a_joint_within_10pct_everywhere = all(gap <= _REL_TOL_10PCT for gap in rel_gap_by_width.values())
        halt_a = {
            "rel_gap_a_joint_vs_b_heads_by_width": rel_gap_by_width,
            "a_joint_within_10pct_of_b_heads_at_every_width": a_joint_within_10pct_everywhere,
            # HALT (a) fires iff A_JOINT does NOT fail, i.e. IS within 10% everywhere.
            "halt_condition_a_triggered": a_joint_within_10pct_everywhere,
        }

    halt_b = None
    if independent_cells and os.path.exists(_INDEPENDENT_CERT_REF_PATH):
        with open(_INDEPENDENT_CERT_REF_PATH) as f:
            cert = json.load(f)
        cert_strong_pass_by_seed = {c["seed"]: c["fit_bar"]["strong_pass"] for c in cert["per_case"]}
        cert_ratio_by_seed = {c["seed"]: c["fit_bar"]["ratio_to_floor"] for c in cert["per_case"]}
        per_seed = {
            str(s): {
                "ours_ratio_to_floor": cell["fit_bar"]["ratio_to_floor"],
                "ours_strong_pass": cell["fit_bar"]["strong_pass"],
                "certified_ratio_to_floor": cert_ratio_by_seed.get(s),
                "certified_strong_pass": cert_strong_pass_by_seed.get(s),
            }
            for s, cell in independent_cells.items()
        }
        halt_b = {
            "certified_reference": _INDEPENDENT_CERT_REF_PATH,
            "per_seed": per_seed,
            # HALT (b) fires iff INDEPENDENT fails to REPRODUCE its certified reference (width.md Step 3:
            # "does not reach its certified fit bar" -- the bar the certified run itself reached, PER SEED).
            # ROOT FIX 2026-07-22: the original predicate demanded strong_pass on every seed, which the
            # CERTIFIED run itself does not satisfy (its seed 2 is 1.1887, strong_pass=False) -- so a
            # bit-faithful reproduction (ours: 1.1887, matching to 4 d.p.) tripped the halt. Reproduction
            # means: no seed regresses below the certified outcome, and ratios match within tolerance.
            "halt_condition_b_triggered": any(
                (v["certified_strong_pass"] and not v["ours_strong_pass"])
                or v["certified_ratio_to_floor"] is None
                or abs(v["ours_ratio_to_floor"] - v["certified_ratio_to_floor"]) / v["certified_ratio_to_floor"] > _REPRO_RATIO_RTOL
                for v in per_seed.values()
            )
            if per_seed
            else None,
        }

    controls_passed = bool(
        halt_a is not None and not halt_a["halt_condition_a_triggered"] and halt_b is not None and not halt_b["halt_condition_b_triggered"]
    )
    return {"halt_condition_a": halt_a, "halt_condition_b": halt_b, "controls_passed": controls_passed}


def _primary_bar_for(arm: Arm, per_cell: dict[Arm, dict[Tier, dict[int, dict]]]) -> dict | None:
    """`arm`'s full-width held-out MSE within 10% of `B_HEADS`', tier 1 AND tier 2, EVERY present seed.

    Per-cell, not aggregated -- the bar is pre-registered to hold on every seed, not on average.
    """
    per_cell_result: dict[str, dict] = {}
    all_pass = True
    any_present = False
    for tier in Tier:
        arm_cells = per_cell[arm][tier]
        b_cells = per_cell[Arm.B_HEADS][tier]
        for s in sorted(set(arm_cells) & set(b_cells)):
            any_present = True
            arm_val = arm_cells[s]["full_width_held_out_mse"]
            b_val = b_cells[s]["full_width_held_out_mse"]
            rel_gap = abs(arm_val - b_val) / b_val if b_val else math.inf
            seed_pass = rel_gap <= _REL_TOL_10PCT
            per_cell_result[f"tier{tier.value}_seed{s}"] = {"arm_value": arm_val, "b_heads_value": b_val, "rel_gap": rel_gap, "pass": seed_pass}
            all_pass = all_pass and seed_pass
    if not any_present:
        return None
    return {"threshold": _REL_TOL_10PCT, "per_cell": per_cell_result, "pass": bool(all_pass)}


def _ordering_bar_for(arm: Arm, per_cell: dict[Arm, dict[Tier, dict[int, dict]]]) -> dict | None:
    """`arm`'s ordering statistic at least as strong as `B_HEADS`'.

    Mean Spearman rho (more negative = stronger decreasing-importance ordering) over every present cell
    where the diagnostic applies.
    """
    arm_rhos = [c["ordering_statistic"]["spearman_index_vs_importance"]["rho"] for t in Tier for c in per_cell[arm][t].values() if c.get("ordering_statistic")]
    b_rhos = [c["ordering_statistic"]["spearman_index_vs_importance"]["rho"] for t in Tier for c in per_cell[Arm.B_HEADS][t].values() if c.get("ordering_statistic")]
    if not arm_rhos or not b_rhos:
        return None
    arm_mean, b_mean = float(np.mean(arm_rhos)), float(np.mean(b_rhos))
    return {"arm_mean_rho": arm_mean, "b_heads_mean_rho": b_mean, "pass": bool(arm_mean <= b_mean)}


def _cost_bar_for(arm: Arm, per_cell: dict[Arm, dict[Tier, dict[int, dict]]]) -> dict | None:
    """`arm`'s per-step wall-clock within 1.3x `B_HEADS`', tier 1 (report only -- not part of the decision rule's gate)."""
    arm_cells = per_cell[arm][Tier.ONE]
    b_cells = per_cell[Arm.B_HEADS][Tier.ONE]
    if not arm_cells or not b_cells:
        return None
    arm_per_step = _mean_scalar([c["train_wall_clock_s"] / c["steps_to_converge"] for c in arm_cells.values() if c["steps_to_converge"]])
    b_per_step = _mean_scalar([c["train_wall_clock_s"] / c["steps_to_converge"] for c in b_cells.values() if c["steps_to_converge"]])
    if not arm_per_step or not b_per_step:
        return None
    ratio = arm_per_step / b_per_step
    return {
        "threshold": _COST_WALLCLOCK_RATIO_BAR,
        "wall_clock_per_step_arm": arm_per_step,
        "wall_clock_per_step_b_heads": b_per_step,
        "ratio": ratio,
        "pass": bool(ratio <= _COST_WALLCLOCK_RATIO_BAR),
        "params_allocated_arm": next(iter(arm_cells.values()))["params_allocated"],
        "params_allocated_b_heads": next(iter(b_cells.values()))["params_allocated"],
    }


def _decide_stage1(per_cell: dict[Arm, dict[Tier, dict[int, dict]]]) -> dict:
    """The mechanical decision rule (`width.md` "Pre-registered bars").

    `stage1_winner = A_STOPGRAD` if PRIMARY+ORDERING pass, else `A_GATES` if it passes both, else
    `B_HEADS` with `stage2_required=True`.
    """
    a_stopgrad_primary = _primary_bar_for(Arm.A_STOPGRAD, per_cell)
    a_stopgrad_ordering = _ordering_bar_for(Arm.A_STOPGRAD, per_cell)
    a_gates_primary = _primary_bar_for(Arm.A_GATES, per_cell)
    a_gates_ordering = _ordering_bar_for(Arm.A_GATES, per_cell)

    a_stopgrad_passes = bool(a_stopgrad_primary and a_stopgrad_primary["pass"] and a_stopgrad_ordering and a_stopgrad_ordering["pass"])
    a_gates_passes = bool(a_gates_primary and a_gates_primary["pass"] and a_gates_ordering and a_gates_ordering["pass"])

    if a_stopgrad_passes:
        winner, stage2_required = Arm.A_STOPGRAD.value, False
    elif a_gates_passes:
        winner, stage2_required = Arm.A_GATES.value, False
    else:
        winner, stage2_required = Arm.B_HEADS.value, True

    return {
        "a_stopgrad": {
            "primary_bar": a_stopgrad_primary,
            "ordering_bar": a_stopgrad_ordering,
            "cost_bar": _cost_bar_for(Arm.A_STOPGRAD, per_cell),
            "passes_decision_rule": a_stopgrad_passes,
        },
        "a_gates": {
            "primary_bar": a_gates_primary,
            "ordering_bar": a_gates_ordering,
            "cost_bar": _cost_bar_for(Arm.A_GATES, per_cell),
            "passes_decision_rule": a_gates_passes,
        },
        "stage1_winner": winner,
        "stage2_required": stage2_required,
    }


def summarize() -> None:
    """Aggregates every per-cell JSON on disk into `WSEL16/frozen.json`. Does not train anything."""
    per_cell: dict[Arm, dict[Tier, dict[int, dict]]] = {arm: {tier: {} for tier in Tier} for arm in Arm}
    for arm in Arm:
        for tier in Tier:
            for seed in SEEDS:
                path = _cell_json_path(arm, tier, seed)
                if os.path.exists(path):
                    with open(path) as f:
                        per_cell[arm][tier][seed] = json.load(f)

    per_arm_tier = {f"{arm.value}_tier{tier.value}": _arm_tier_block(per_cell[arm][tier], W_MAX) for arm in Arm for tier in Tier}
    controls = _check_controls(per_cell)
    decision = _decide_stage1(per_cell)

    frozen = {
        "per_arm_tier": per_arm_tier,
        **controls,
        **decision,
        "config": {
            "w_max": W_MAX,
            "seeds": list(SEEDS),
            "schedule": nwn.WidthSchedule.SANDWICH.value,
            "tier1": {
                "toy": _TIER_CONFIG[Tier.ONE].toy.value,
                "n_train": _TIER_CONFIG[Tier.ONE].n_train,
                "n_test": _TIER_CONFIG[Tier.ONE].n_test,
                "sigma": _TIER_CONFIG[Tier.ONE].sigma,
            },
            "tier2": {
                "toy": _TIER_CONFIG[Tier.TWO].toy.value,
                "n_train": _TIER_CONFIG[Tier.TWO].n_train,
                "n_test": _TIER_CONFIG[Tier.TWO].n_test,
            },
            "lr": cwe.LR,
            "max_epochs_cap": DEFAULT_MAX_EPOCHS,
            "check_every": DEFAULT_CHECK_EVERY,
            "patience": DEFAULT_PATIENCE,
            "min_delta": DEFAULT_MIN_DELTA,
        },
        "n_cells_present": {f"{arm.value}_tier{tier.value}": len(per_cell[arm][tier]) for arm in Arm for tier in Tier},
        "provenance": run_provenance(),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "frozen.json")
    with open(path, "w") as f:
        json.dump(_jsonable(frozen), f, indent=2)
    print(f"wrote {path}")
    print(f"controls_passed={controls['controls_passed']}  stage1_winner={decision['stage1_winner']}  stage2_required={decision['stage2_required']}")


# ---------------------------------------------------------------------------
# --selftest
# ---------------------------------------------------------------------------

_SELFTEST_KW = {"w_max": 3, "n_train": 60, "n_test": 40, "max_epochs": 400, "check_every": 50, "patience": 2, "min_delta": 5e-2}

# A_CORRECTIVE-only override: the real spec constants (2000/200) never trigger inside _SELFTEST_KW's
# 400-epoch budget, which would leave the corrective branch of _train_custom_to_convergence completely
# unexercised by this wiring check. Tiny values here make it fire a few times within the tiny budget.
_SELFTEST_CORRECTIVE_KW = {"corrective_interval": 40, "corrective_duration": 10}

# Tier.THREE-only override: run_cell requires sigma explicitly for that tier (no canonical fallback,
# see _resolve_tier_config) -- _SELFTEST_KW's own n_train/n_test stand in for tier 3's normally-required
# n_train too (run_cell does not re-validate closed-set membership, only the CLI's argparse does), so
# this tiny wiring check never needs a real {200,500,1500,4000} x {0.05,0.15,0.5} cell.
_SELFTEST_TIER3_KW = {"sigma": 0.05}

# Arms with no SHARED hidden trunk (`_ordering_statistic` returns None for them -- see its docstring).
_ORDERING_NONE_ARMS = (Arm.INDEPENDENT, Arm.A_CASCADE_STAGED)

_SELFTEST_REQUIRED_KEYS = (
    "convergence",
    "n_widths_trustworthy",
    "all_widths_trustworthy",
    "hit_cap",
    "train_wall_clock_s",
    "steps_to_converge",
    "train_mse_by_width",
    "held_out_mse_by_width",
    "full_width_held_out_mse",
    "train_minus_heldout_gap_by_width",
    "params_allocated",
    "params_effective",
    "executed_flops_by_width",
    "fit_bar",
    "ordering_statistic",
    "selected_width_global",
    "selected_width_distilled_router",
    "cascade_extra_readout_bias_freedom",
)


def run_selftest() -> bool:
    """Wiring check (<60s): tiny cells across every (arm, tier), the stop-grad one-pass identity vs a loop reference, the weighted-loss formula, and JSON round-tripping."""
    ok = True
    for arm in Arm:
        for tier in Tier:
            if arm is Arm.A_CASCADE_STAGED and tier is not Tier.ONE:
                continue  # A_CASCADE_STAGED's squared-error port is tier-1-only -- run_cell refuses tiers 2/3 (width.md Stage 2).
            kw = dict(_SELFTEST_KW)
            if arm is Arm.A_CORRECTIVE:
                kw.update(_SELFTEST_CORRECTIVE_KW)
            if tier is Tier.THREE:
                kw.update(_SELFTEST_TIER3_KW)
            case, _state = run_cell(arm, tier, seed=0, **kw)
            keys_ok = all(key in case for key in _SELFTEST_REQUIRED_KEYS)
            conv_ok = all(len(case["convergence"][k]["trajectory"]) >= 1 for k in range(1, _SELFTEST_KW["w_max"] + 1))
            ordering_none_as_expected = (case["ordering_statistic"] is None) == (arm in _ORDERING_NONE_ARMS)
            roundtrip = json.loads(json.dumps(_jsonable(case)))
            roundtrip_ok = all(key in roundtrip for key in _SELFTEST_REQUIRED_KEYS)
            combo_ok = keys_ok and conv_ok and ordering_none_as_expected and roundtrip_ok
            print(
                f"[wsel16 selftest] arm={arm.value} tier={tier.value} keys_present={keys_ok} convergence_recorded={conv_ok} "
                f"ordering_None_as_expected={ordering_none_as_expected} json_roundtrip_ok={roundtrip_ok}  {'PASS' if combo_ok else 'FAIL'}"
            )
            ok = ok and combo_ok

    # Stop-grad one-pass form vs an EXPLICIT per-width loop reference (tiny case, independent of Step 1's pytest).
    torch.manual_seed(0)
    net_sg = nwn.NestedWidthNet(w_max=4)
    x_tiny = torch.randn(16, 1)
    one_pass = stopgrad_all_widths_pred(net_sg, x_tiny)
    with torch.no_grad():
        h = net_sg.hidden(x_tiny)
        loop_cols = []
        for k in range(1, 5):
            if k == 1:
                s_prev = net_sg.mean_head.bias.expand(h.shape[0], 1)
            else:
                mask = torch.zeros_like(h)
                mask[:, : k - 1] = 1.0
                s_prev = net_sg.mean_head(h * mask)
            c_k = h[:, k - 1 : k] * net_sg.mean_head.weight[0, k - 1]
            loop_cols.append(s_prev + c_k)
        loop_pred = torch.cat(loop_cols, dim=1)
    stopgrad_err = (one_pass - loop_pred).abs().max().item()
    stopgrad_match = stopgrad_err < _SELFTEST_STOPGRAD_LOOP_TOL
    print(f"[wsel16 selftest] stopgrad one-pass matches loop reference: max_abs_err={stopgrad_err:.3e}  {'PASS' if stopgrad_match else 'FAIL'}")
    ok = ok and stopgrad_match

    # weighted_squared_error matches a hand-computed case.
    pred = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([1.5, 2.5, 2.0])
    sigma_true = torch.tensor([0.5, 1.0, 2.0])
    expected = float((((pred - y) ** 2) / sigma_true**2).mean().item())
    got = float(wc.weighted_squared_error(pred, y, sigma_true).item())
    weighted_ok = abs(expected - got) < _SELFTEST_WEIGHTED_LOSS_TOL
    print(f"[wsel16 selftest] weighted_squared_error matches hand-computed case: expected={expected:.6f} got={got:.6f}  {'PASS' if weighted_ok else 'FAIL'}")
    ok = ok and weighted_ok

    print(f"[wsel16 selftest] {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    """Parses args and dispatches to `--selftest` / `--summarize` / one real `--arm`/`--tier`/`--seed` cell."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="Tiny wiring check, then exit.")
    parser.add_argument("--summarize", action="store_true", help="Aggregate every per-cell JSON on disk into WSEL16/frozen.json.")
    parser.add_argument("--arm", type=str, choices=[a.value for a in Arm], default=None, help="Which of the 8 arms this cell trains (5 stage-1, 3 stage-2).")
    parser.add_argument("--tier", type=int, choices=[t.value for t in Tier], default=None, help="SS3.8 tier (1 = reference, 2 = noisy-easy control, 3 = data x noise ladder).")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for this cell (canonical suite: 0, 1, 2).")
    parser.add_argument("--tag", type=str, default=None, help="Optional filename suffix (keeps a re-run from clobbering the canonical grid cell).")
    parser.add_argument("--n-train", type=int, choices=_TIER3_N_TRAIN_CHOICES, default=None, help="Tier-3-only: training set size (SS3.8's data x noise ladder).")
    parser.add_argument("--sigma", type=float, choices=_TIER3_SIGMA_CHOICES, default=None, help="Tier-3-only: noise sigma (SS3.8's data x noise ladder).")
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--check-every", type=int, default=DEFAULT_CHECK_EVERY)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA)
    parser.add_argument("--w-max", type=int, default=None)
    args = parser.parse_args()

    if args.selftest:
        sys.exit(0 if run_selftest() else 1)
    if args.summarize:
        summarize()
        return
    if args.arm is None or args.tier is None or args.seed is None:
        parser.error("--arm, --tier and --seed are all required for a real cell (or pass --selftest / --summarize).")

    arm = Arm(args.arm)
    tier = Tier(args.tier)
    if tier is Tier.THREE:
        if args.n_train is None or args.sigma is None:
            parser.error("--tier 3 (the SS3.8 data x noise ladder) requires both --n-train and --sigma.")
    elif args.n_train is not None or args.sigma is not None:
        parser.error("--n-train/--sigma only apply to --tier 3 (SS3.8's data x noise ladder); tiers 1/2 use their own fixed config.")
    w_max = args.w_max if args.w_max is not None else W_MAX
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg_preview = _resolve_tier_config(tier, args.n_train, None, args.sigma)
    print(f"[wsel16] arm={arm.value} tier={tier.value} toy={cfg_preview.toy.value} seed={args.seed} w_max={w_max}", flush=True)
    case, state_dict = run_cell(
        arm,
        tier,
        args.seed,
        w_max=w_max,
        n_train=args.n_train,
        sigma=args.sigma,
        max_epochs=args.max_epochs,
        check_every=args.check_every,
        patience=args.patience,
        min_delta=args.min_delta,
    )

    if not case["all_widths_trustworthy"]:
        print(f"*** DO-NOT-CONCLUDE GUARD: arm={arm.value} tier={tier.value} seed={args.seed} has widths that did NOT converge trustworthily. Raise --max-epochs. ***")

    path_kw = {"n_train": args.n_train, "sigma": args.sigma} if tier is Tier.THREE else {}
    cell_path = _cell_json_path(arm, tier, args.seed, tag=args.tag, **path_kw)
    with open(cell_path, "w") as f:
        json.dump(_jsonable(case), f, indent=2)
    print(f"wrote {cell_path}")

    state_path = _state_path(arm, tier, args.seed, tag=args.tag, **path_kw)
    torch.save(state_dict, state_path)
    print(f"wrote {state_path}")

    print(f"full_width_held_out_mse={case['full_width_held_out_mse']:.5f}  train_wall_clock_s={case['train_wall_clock_s']:.2f}  steps_to_converge={case['steps_to_converge']}")


if __name__ == "__main__":
    main()
