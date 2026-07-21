r"""The ProbReg structure-phase battery driver (stages PS-1..PS-4).

The "structure phase" settles what ProbabilisticRegressionModel structurally IS by experiment,
before Problem 2 (per-input capacity) is allowed to ask anything about a dial. Four stages, one
driver, run cell-by-cell (the ROOT runs the grid; this file never runs one itself):

    PS-1 (trunk)       -- supervision x spread x likelihood, on the pinned k in {2, 6}.
    PS-2 (patches)      -- ordering / anchored / monotonic / middle-class / boundary, one at a time,
                           off the trunk winner.
    PS-3 (layout)       -- SEPARATE_HEADS vs SINGLE_HEAD_N_OUTPUTS vs SINGLE_HEAD_FINAL_OUTPUT,
                           optionally parameter-matched.
    PS-4 (certification) -- the winning recipe re-run on the full toy suite (T_FULL), plus the
                           mandatory sigma_toy/2 and 2*sigma_toy re-score of stored predictions.

Every axis is a closed set of CLI choices; the (spread, likelihood) pair additionally has a LOCKED
coherence matrix (`_coherence_kwargs`) that REFUSES the one excluded combination
(`all_constant x collapsed`) rather than silently falling through to a default -- that silent-
fallthrough failure mode is exactly what this programme has been bitten by before (see
MASTER Decision 29's enforcement style in `probabilistic_regression.py`, mirrored here).

Reuse trail (minimum-viable-code ladder, rung 2 -- see the task report for the full search log):
  - Toy generators, `sep_schedule`/`sep_hump`, and the gold-standard `sample_toy_*_given_x`
    resamplers: `automl_package/examples/_toy_datasets.py`, imported as a module (never
    reimplemented) so this file can call its underscore-prefixed helpers the same way
    `width_wsel13.py` calls `kdropout_converged_width_experiment`'s.
  - `fixed_sigma_mixture_log_likelihood`: `automl_package/utils/losses.py` (added by commit
    ef7e1f8 for capacity selection; the fixed-sigma common yardstick here is the SAME function).
  - `calculate_combined_loss`, `mdn_nll`: the exact functions
    `ProbabilisticRegressionModel._calculate_custom_loss` itself calls for its regression-loss
    term -- reused directly rather than re-derived, so "own NLL" here is bit-for-bit the model's
    own training objective, just isolated from the classification/penalty terms added on top.
  - `compute_ordering_means`: `automl_package/utils/ordering_loss.py`, the same function
    `_calculate_custom_loss` calls for its ordering-constraint penalty; `ordering_violations`
    here is a one-line hinge-count twin of `ordering_penalty`'s squared-hinge sum.
  - The driver SHAPE (per-cell JSON written the moment a cell finishes, `--selftest` tiny
    known-answer check, `--summarize` -> frozen.json, `_jsonable` coercion): the pattern
    `width_wsel13.py` and the absorbed `probreg_variance_degeneracy_check.py` both use.
  - `run_provenance()`: `automl_package/utils/run_provenance.py`, reused verbatim.

Absorbed and replaces `automl_package/examples/probreg_variance_degeneracy_check.py` (uncommitted,
deleted by this task): its toy wiring, `epoch_callback` trajectory hook, per-class spread
extraction via `forward_per_class`, classifier-confidence computation, and `--selftest`/
`--summarize` patterns are the direct ancestors of the corresponding code below.

Design decisions this driver had to make that the task brief left to "arithmetic, not judgment"
but the underlying data structures forced a concrete choice for (flagged here, and again in the
task report, exactly as the absorbed predecessor flagged its own judgment call):

  1. "own NLL" (`train_nll_own`/`heldout_nll_own`) is the arm's regression-loss term ONLY --
     `_calculate_custom_loss`'s step 1, computed by calling the SAME functions it calls, on a
     fresh forward pass at the current weights. It excludes the classification/ordering/boundary/
     middle-class penalty terms that step 2-4 of that method add during TRAINING: those are
     regularizers, not likelihood, and mixing them into a "NLL" trajectory would make the D1
     degeneracy check (a gap-in-NLL diagnostic) noisy for reasons that have nothing to do with
     variance collapse.
  2. The convergence gate ("early-stop on held-out own-NLL, patience 8 checks, min-delta 1e-4")
     is implemented as an INCREMENTAL early stop raised from inside `epoch_callback` (a
     `_TrainingStoppedError` exception caught around `model.fit()`), evaluated once per recorded
     checkpoint (not once per epoch) -- patience is counted in CHECKS, matching the spec's
     wording literally, without needing to extend `PyTorchModelBase`'s own early-stopping
     primitive (which has no min-delta and counts epochs, not checks) or reimplement the
     training loop.
  3. sigma_toy MUST be one shared constant per toy, identical across every seed and arm (Decision
     24, cited in `fixed_sigma_mixture_log_likelihood`'s own docstring) -- so for the two
     input-varying-noise toys (`toy_c`/`toy_c_broad`/`toy_e`; `toy_d`'s per-third variance is
     the same idea) the "RMS over the training inputs" the readouts describe is computed over a
     LARGE, FIXED-SEED synthetic population (`np.random.default_rng(0)`, n=200000), not the
     current cell's own (seed-varying) training draw -- using the literal per-seed draw would
     make sigma_toy itself seed-dependent, breaking the "fixed-sigma score column"'s entire
     point.
  4. `min_sigma_ratio` appears in the LOCKED trajectory schema's D1 arithmetic ("at any logged
     checkpoint") but only in the LOCKED per-cell schema's `final` block, not per-checkpoint --
     `per_class_sigma`/`sigma_toy` (both already-recorded, per-checkpoint-or-cell-level fields)
     resolve this: D1's checkpoint-level min_sigma_ratio is approximated as
     `per_class_sigma.min / sigma_toy` (exact for the three homoscedastic screen toys, an
     RMS-based approximation for the heteroscedastic ones, which is exactly what sigma_toy
     itself already is under decision 3 above). The FINAL block's own `min_sigma_ratio` is the
     more precise version: per-class fitted sigma divided by a per-class TRUE within-slice
     residual sigma, estimated by resampling `y | x` many times at each held-out point via the
     toys' own `sample_toy_*_given_x` gold-standard arbiters (toy_a/toy_b/broad_unimodal have no
     such public helper; `_resample_y_given_x` below adds the three-line inline equivalent for
     each, mirroring their `make_toy_*` formula exactly).
  5. D2's win rule (structure.md §4.5, AMENDED 2026-07-21 by the root after this concern was
     raised against the original wording, which compared `heldout_nll_own` across arms that do
     not share a likelihood family -- exactly what PS-1's trunk battery varies). The PRIMARY
     readout is now `heldout_fixed_sigma_mll` (HIGHER is better -- it is identically defined for
     every arm, one shared sigma over the arm's own component means/weights, so no arm can win on
     variance machinery rather than structure). `heldout_nll_own` is now a reported column and a
     RESTRICTED tie-break only: `fixed_shared` arms have no fitted predictive density at all and
     record `final.heldout_nll_own = None`; the tie-break is only attempted when EVERY surviving
     arm has a defined own-NLL (if even one survivor is `fixed_shared`, the tie-break cannot
     arbitrate against it and the decision falls straight through to D4, rather than manufacture a
     comparison against an arm own-NLL cannot see).
  6. Per-point own-NLL, per-point fixed-sigma-mixture log-likelihood, and (PS-4 only) per-point
     (probs, mus, y) are NOT part of the locked per-cell JSON schema (which has no array field for
     them) but D2's bootstrap (both the primary and the restricted tie-break) and PS-4's
     `--rescore` all need per-point values. All three are saved to a companion
     `<cell>_perpoint.npz` next to the per-cell JSON -- an additional artifact, not a schema
     violation, the same role WSEL-13's companion `state_*.pt` files play.
  7. D3's identifiability disqualifier (structure.md §4.5, AMENDED 2026-07-21 by the root after
     inspecting real PS-1 cells mid-grid: the original "any toy, any seed" wording would have
     disqualified every `ce` arm -- `broad_unimodal`/`toy_c_broad` are single-mode twins built so
     y's fine structure is NOT recoverable from x, so chance-level `slice_accuracy` there is
     correct, not broken; and "any seed" tripped a working mechanism on ~1% seed noise). `ce` arms
     now gate ONLY on the SEED-MEAN `slice_accuracy` on the reference toy `toy_b` (the one toy
     with genuine, known-k components), per k; every other toy's `slice_accuracy` is a recorded,
     non-gating diagnostic. An arm with no `toy_b` cell at all is recorded as not-evaluated, never
     disqualified by omission. `none` arms (`ordering_violations`, any seed) are unchanged; the
     1.5/k bar itself is unchanged.

CLI (locked):
    --stage {1,2,3,4} --supervision {none,ce} --spread {per_input,fixed_shared,all_constant}
    --likelihood {collapsed,mixture} --ordering {auto,on,off} --anchored {on,off}
    --monotonic {on,off} --middle-class {on,off} --boundary {off,hardsigmoid,penalty}
    --layout {separate,single_n,single_final} --param-match {on,off}
    --toy {toy_a,toy_b,broad_unimodal,toy_c_broad,toy_c,toy_e,toy_d} --k INT --seed INT
    --max-epochs INT --check-every INT --out DIR --selftest --summarize --decide
    --rescore {half,double}

Usage:
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python -m automl_package.examples.probreg_structure_battery --selftest
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python -m automl_package.examples.probreg_structure_battery \\
        --stage 1 --supervision none --spread fixed_shared --likelihood collapsed \\
        --toy toy_a --k 2 --seed 0 --max-epochs 300 --check-every 10
    ~/dev/.venv/bin/python -m automl_package.examples.probreg_structure_battery --stage 1 --summarize
    ~/dev/.venv/bin/python -m automl_package.examples.probreg_structure_battery --stage 1 --decide
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import json
import math
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch

import automl_package.examples._toy_datasets as toy_datasets
from automl_package.enums import (
    BoundaryRegularizationMethod,
    HeadSpread,
    NClassesSelectionMethod,
    ProbabilisticRegressionOptimizationStrategy,
    ProbRegLossType,
    RegressionStrategy,
    UncertaintyMethod,
)
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel
from automl_package.utils.losses import fixed_sigma_mixture_log_likelihood
from automl_package.utils.numerics import create_bins
from automl_package.utils.ordering_loss import compute_ordering_means
from automl_package.utils.run_provenance import run_provenance

RESULTS_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "capacity_ladder_results")

# Training config, held constant across every cell (not on the locked CLI, so not exposed as flags
# -- matches the absorbed predecessor's own reasoning: keep the contract to exactly what was named).
LEARNING_RATE = 0.01
BATCH_SIZE = 64
DEFAULT_MAX_EPOCHS = 300
DEFAULT_CHECK_EVERY = 10
PATIENCE_CHECKS = 8  # "patience 8 checks" -- global constraints' convergence gate.
MIN_DELTA = 1e-4
N_RESAMPLES_TRUE_SIGMA = 200  # per held-out point, for the final block's min_sigma_ratio.
SIGMA_TOY_POPULATION_SEED = 0  # fixed, seed-independent -- see design decision 3 above.
SIGMA_TOY_POPULATION_N = 200_000

# Canonical (n_train, n_holdout) per toy. toy_d's 3-region staircase needs the most coverage;
# toy_a/toy_b/broad_unimodal mirror the absorbed predecessor's 600/300.
_TOY_N: dict[str, tuple[int, int]] = {
    "toy_a": (600, 300),
    "toy_b": (600, 300),
    "broad_unimodal": (600, 300),
    "toy_c_broad": (1200, 500),
    "toy_c": (1200, 500),
    "toy_e": (1200, 500),
    "toy_d": (1800, 600),
}

# Toy construction constants. toy_b's k_true=2 is NOT the module's own default (3) -- the readouts
# name toy_b explicitly as "reference, known intrinsic k=2" (T_SCREEN definition), and
# broad_unimodal's own docstring calls itself "moment-matched twin of the k_true=2 bimodal toy",
# so both must share this pairing for the mixture-vs-collapsed discriminator to be well-posed.
TOY_A_SIGMA = 0.5
TOY_B_K_TRUE = 2
TOY_B_SEPARATION = 4.0
TOY_B_SIGMA = 0.3
BROAD_UNIMODAL_SEPARATION = TOY_B_SEPARATION
BROAD_UNIMODAL_SIGMA = TOY_B_SIGMA
TOY_C_SIGMA = 0.3
TOY_C_SEP_MIN = 0.3
TOY_C_SEP_MAX = 4.0
TOY_E_SIGMA = 0.3
TOY_E_SEP_MIN = 0.3
TOY_E_SEP_MAX = 4.0
TOY_D_SIGMA = 0.3
TOY_D_SEPARATION = 4.0

_MIN_HIDDEN_SIZE = 32
_MAX_HIDDEN_SIZE_SEARCH = 4096  # safety cap on the param-matching search below.

_PROBABILISTIC_OUTPUT_WIDTH = 2  # (mean, log_var) width `forward_per_class`/`final_predictions` return under PROBABILISTIC.
_MIN_CLASSES_FOR_ORDERING = 2  # ordering has no adjacent pair to violate below this.

# D1 -- degeneracy disqualifier (§4.5), locked thresholds.
D1_MIN_EPOCH = 50
D1_MIN_CHECKPOINTS_FOR_TREND = 3  # "increasing over the last 3 checkpoints" -- needs indices i-2, i-1, i.
D1_MIN_SIGMA_RATIO_THRESHOLD = 0.1
D1_NLL_GAP_THRESHOLD = 0.5

# D2 -- win rule (§4.5), locked threshold.
D2_MIN_WINNING_TOYS = 2
D2_SE_MULTIPLE = 2


class Stage(Enum):
    """The four structure-phase battery stages (§ locked CLI)."""

    TRUNK = 1
    PATCHES = 2
    LAYOUT = 3
    CERTIFICATION = 4


class Supervision(Enum):
    """--supervision. `ce` is a MASTER-Decision-29-retired optimization strategy, opted back in."""

    NONE = "none"
    CE = "ce"


class Spread(Enum):
    """--spread. Exactly three values (a per-class-scalar fourth was rejected -- do not add one)."""

    PER_INPUT = "per_input"
    FIXED_SHARED = "fixed_shared"
    ALL_CONSTANT = "all_constant"


class Likelihood(Enum):
    """--likelihood."""

    COLLAPSED = "collapsed"
    MIXTURE = "mixture"


class OrderingMode(Enum):
    """--ordering. `auto` defers to the model's own (recipe, GAUSSIAN_LTV, REGRESSION_ONLY)-only default."""

    AUTO = "auto"
    ON = "on"
    OFF = "off"


class OnOff(Enum):
    """Generic on/off patch toggle, shared by --anchored / --monotonic / --middle-class / --param-match."""

    ON = "on"
    OFF = "off"


class Boundary(Enum):
    """--boundary."""

    OFF = "off"
    HARDSIGMOID = "hardsigmoid"
    PENALTY = "penalty"


class Layout(Enum):
    """--layout."""

    SEPARATE = "separate"
    SINGLE_N = "single_n"
    SINGLE_FINAL = "single_final"


class Toy(Enum):
    """--toy. Canonical toys only (`_toy_datasets.py`); `make_datasets()`/`make_v_toy*` are forbidden."""

    TOY_A = "toy_a"
    TOY_B = "toy_b"
    BROAD_UNIMODAL = "broad_unimodal"
    TOY_C_BROAD = "toy_c_broad"
    TOY_C = "toy_c"
    TOY_E = "toy_e"
    TOY_D = "toy_d"


class RescoreFactor(Enum):
    """--rescore. PS-4's mandatory robustness re-score of stored predictions (no retraining)."""

    HALF = "half"
    DOUBLE = "double"


_LAYOUT_TO_STRATEGY: dict[Layout, RegressionStrategy] = {
    Layout.SEPARATE: RegressionStrategy.SEPARATE_HEADS,
    Layout.SINGLE_N: RegressionStrategy.SINGLE_HEAD_N_OUTPUTS,
    Layout.SINGLE_FINAL: RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT,
}


@dataclass(frozen=True)
class Arm:
    """One point in the arm space: supervision x spread x likelihood x 5 patches x layout x param-match."""

    supervision: Supervision
    spread: Spread
    likelihood: Likelihood
    ordering: OrderingMode
    anchored: OnOff
    monotonic: OnOff
    middle_class: OnOff
    boundary: Boundary
    layout: Layout
    param_match: OnOff

    def arm_id(self) -> str:
        """Filename-safe, fully-descriptive identifier (no internal codenames -- literal CLI values)."""
        return (
            f"sup-{self.supervision.value}_spr-{self.spread.value}_lik-{self.likelihood.value}"
            f"_ord-{self.ordering.value}_anc-{self.anchored.value}_mono-{self.monotonic.value}"
            f"_mc-{self.middle_class.value}_bnd-{self.boundary.value}_lay-{self.layout.value}"
            f"_pm-{self.param_match.value}"
        )

    def to_json(self) -> dict[str, Any]:
        """The locked `arm` block (§4.4) -- exactly these keys, `param_match` recorded alongside, not inside."""
        return {
            "supervision": self.supervision.value,
            "spread": self.spread.value,
            "likelihood": self.likelihood.value,
            "patches": {
                "ordering": self.ordering.value,
                "anchored": self.anchored.value,
                "monotonic": self.monotonic.value,
                "middle_class": self.middle_class.value,
                "boundary": self.boundary.value,
            },
            "layout": self.layout.value,
        }


# ---------------------------------------------------------------------------
# Toy construction, sigma_toy, and the true-conditional-sigma resampler.
# ---------------------------------------------------------------------------


def _make_toy_xy(toy: Toy, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Builds `(x, y)` for one toy, dispatching to `_toy_datasets`'s canonical generators verbatim."""
    if toy is Toy.TOY_A:
        return toy_datasets.make_toy_a(n=n, sigma=TOY_A_SIGMA, seed=seed)
    if toy is Toy.TOY_B:
        return toy_datasets.make_toy_b(n=n, k_true=TOY_B_K_TRUE, separation=TOY_B_SEPARATION, sigma=TOY_B_SIGMA, seed=seed)
    if toy is Toy.BROAD_UNIMODAL:
        return toy_datasets.make_broad_unimodal(n=n, separation=BROAD_UNIMODAL_SEPARATION, sigma=BROAD_UNIMODAL_SIGMA, seed=seed, baseline="zero")
    if toy is Toy.TOY_C_BROAD:
        return toy_datasets.make_toy_c_broad(n=n, sigma=TOY_C_SIGMA, sep_min=TOY_C_SEP_MIN, sep_max=TOY_C_SEP_MAX, seed=seed)
    if toy is Toy.TOY_C:
        return toy_datasets.make_toy_c(n=n, sigma=TOY_C_SIGMA, sep_min=TOY_C_SEP_MIN, sep_max=TOY_C_SEP_MAX, seed=seed)
    if toy is Toy.TOY_E:
        return toy_datasets.make_toy_e(n=n, sigma=TOY_E_SIGMA, sep_min=TOY_E_SEP_MIN, sep_max=TOY_E_SEP_MAX, seed=seed)
    if toy is Toy.TOY_D:
        return toy_datasets.make_toy_d(n=n, sigma=TOY_D_SIGMA, separation=TOY_D_SEPARATION, seed=seed)
    raise AssertionError(f"unreachable: every Toy member is dispatched above (got {toy})")


def _resample_y_given_x(toy: Toy, x: np.ndarray, seed: int) -> np.ndarray:
    """Draws a FRESH `y` at the SAME `x` (gold-standard-arbiter resampling), for the true-sigma estimator.

    toy_c/toy_e/toy_d reuse `_toy_datasets`'s own public `sample_toy_*_given_x` resamplers verbatim.
    toy_a/toy_b/broad_unimodal have no such helper (nothing else needs one); their three-line
    formulas mirror the corresponding `make_toy_*` generator exactly, evaluated at a supplied `x`
    instead of a fresh draw.
    """
    xr = np.asarray(x, dtype=np.float64).ravel()
    rng = np.random.default_rng(seed)
    if toy is Toy.TOY_A:
        return (np.sin(2 * np.pi * xr) + TOY_A_SIGMA * rng.normal(0.0, 1.0, xr.size)).astype(np.float32)
    if toy is Toy.TOY_B:
        base = np.sin(2 * np.pi * xr)
        comp = rng.integers(0, TOY_B_K_TRUE, size=xr.size)
        offset = (comp - (TOY_B_K_TRUE - 1) / 2.0) * TOY_B_SEPARATION * TOY_B_SIGMA
        return (base + offset + TOY_B_SIGMA * rng.normal(0.0, 1.0, xr.size)).astype(np.float32)
    if toy is Toy.BROAD_UNIMODAL:
        sigma_broad = BROAD_UNIMODAL_SIGMA * math.sqrt(1.0 + BROAD_UNIMODAL_SEPARATION**2 / 4.0)
        return (sigma_broad * rng.normal(0.0, 1.0, xr.size)).astype(np.float32)
    if toy is Toy.TOY_C_BROAD:
        return toy_datasets.sample_toy_c_broad_given_x(xr, sigma=TOY_C_SIGMA, sep_min=TOY_C_SEP_MIN, sep_max=TOY_C_SEP_MAX, seed=seed)
    if toy is Toy.TOY_C:
        return toy_datasets.sample_toy_c_given_x(xr, sigma=TOY_C_SIGMA, sep_min=TOY_C_SEP_MIN, sep_max=TOY_C_SEP_MAX, seed=seed)
    if toy is Toy.TOY_E:
        return toy_datasets.sample_toy_e_given_x(xr, sigma=TOY_E_SIGMA, sep_min=TOY_E_SEP_MIN, sep_max=TOY_E_SEP_MAX, seed=seed)
    if toy is Toy.TOY_D:
        return toy_datasets.sample_toy_d_given_x(xr, sigma=TOY_D_SIGMA, separation=TOY_D_SEPARATION, seed=seed)
    raise AssertionError(f"unreachable: every Toy member is dispatched above (got {toy})")


def _sigma_for_toy(toy: Toy) -> float:
    """sigma_toy: the toy's construction noise value (design decision 3 -- ONE shared constant, never per-seed).

    Homoscedastic toys (toy_a, toy_b, broad_unimodal) return their single generator constant
    directly. The input-varying-noise toys (toy_c/toy_c_broad/toy_e/toy_d) return the RMS, over a
    large fixed-seed synthetic `x` population, of the generator's true total-variance formula --
    the SAME formula `_toy_datasets._broad_targets`/`_toy_d_ndim_total_var` already implement.
    """
    if toy is Toy.TOY_A:
        return TOY_A_SIGMA
    if toy is Toy.TOY_B:
        return TOY_B_SIGMA
    if toy is Toy.BROAD_UNIMODAL:
        return BROAD_UNIMODAL_SIGMA * math.sqrt(1.0 + BROAD_UNIMODAL_SEPARATION**2 / 4.0)

    xr = np.random.default_rng(SIGMA_TOY_POPULATION_SEED).uniform(0.0, 1.0, SIGMA_TOY_POPULATION_N)
    if toy is Toy.TOY_C_BROAD or toy is Toy.TOY_C:
        sep = toy_datasets.sep_schedule(xr, sep_min=TOY_C_SEP_MIN, sep_max=TOY_C_SEP_MAX)
        variance = TOY_C_SIGMA**2 * (1.0 + sep**2 / 4.0)
        return float(np.sqrt(np.mean(variance)))
    if toy is Toy.TOY_E:
        sep = toy_datasets.sep_hump(xr, sep_min=TOY_E_SEP_MIN, sep_max=TOY_E_SEP_MAX)
        variance = TOY_E_SIGMA**2 * (1.0 + sep**2 / 4.0)
        return float(np.sqrt(np.mean(variance)))
    if toy is Toy.TOY_D:
        k = toy_datasets._staircase_k(xr)
        variance = toy_datasets._toy_d_ndim_total_var(k, TOY_D_SIGMA, TOY_D_SEPARATION)
        return float(np.sqrt(np.mean(variance)))
    raise AssertionError(f"unreachable: every Toy member is dispatched above (got {toy})")


# ---------------------------------------------------------------------------
# Coherence matrix, patch kwargs, and param-matching (PS-3).
# ---------------------------------------------------------------------------


def _coherence_kwargs(spread: Spread, likelihood: Likelihood, sigma_toy: float) -> dict[str, Any]:
    """The locked spread x likelihood coherence matrix. REFUSES the excluded combo -- no catch-all default."""
    if spread is Spread.PER_INPUT and likelihood is Likelihood.COLLAPSED:
        return {"uncertainty_method": UncertaintyMethod.PROBABILISTIC, "head_spread": HeadSpread.PER_INPUT, "prob_reg_loss_type": ProbRegLossType.GAUSSIAN_LTV}
    if spread is Spread.PER_INPUT and likelihood is Likelihood.MIXTURE:
        return {"uncertainty_method": UncertaintyMethod.PROBABILISTIC, "head_spread": HeadSpread.PER_INPUT, "prob_reg_loss_type": ProbRegLossType.MDN}
    if spread is Spread.FIXED_SHARED and likelihood is Likelihood.COLLAPSED:
        return {"uncertainty_method": UncertaintyMethod.CONSTANT}
    if spread is Spread.FIXED_SHARED and likelihood is Likelihood.MIXTURE:
        return {"uncertainty_method": UncertaintyMethod.CONSTANT, "prob_reg_loss_type": ProbRegLossType.FIXED_SIGMA_MIXTURE, "fixed_sigma_train": sigma_toy}
    if spread is Spread.ALL_CONSTANT and likelihood is Likelihood.MIXTURE:
        return {"uncertainty_method": UncertaintyMethod.PROBABILISTIC, "head_spread": HeadSpread.ALL_CONSTANT, "prob_reg_loss_type": ProbRegLossType.MDN}
    if spread is Spread.ALL_CONSTANT and likelihood is Likelihood.COLLAPSED:
        raise ValueError(
            "spread=all_constant x likelihood=collapsed is EXCLUDED by the locked coherence matrix "
            "(redundant with the per-class questions; keeps the grid bounded). This is a refusal, "
            "not a bug -- no other (spread, likelihood) combination reaches this branch."
        )
    raise AssertionError(f"unreachable: every (Spread, Likelihood) pair is enumerated above (got {spread}, {likelihood})")


def _supervision_kwargs(supervision: Supervision) -> dict[str, Any]:
    """--supervision ce opts back into the MASTER-Decision-29-retired CE_STOP_GRAD strategy."""
    if supervision is Supervision.NONE:
        return {}
    if supervision is Supervision.CE:
        return {"optimization_strategy": ProbabilisticRegressionOptimizationStrategy.CE_STOP_GRAD, "allow_retired_capacity_selection": True}
    raise AssertionError(f"unreachable: every Supervision member is dispatched above (got {supervision})")


def _patch_kwargs(arm: Arm) -> dict[str, Any]:
    """The 5 trunk-stage patches. `ordering=auto` leaves `ordering_constraint_weight` at the model's own default."""
    kwargs: dict[str, Any] = {}
    if arm.ordering is OrderingMode.ON:
        kwargs["ordering_constraint_weight"] = 1.0
    elif arm.ordering is OrderingMode.OFF:
        kwargs["ordering_constraint_weight"] = 0.0
    elif arm.ordering is not OrderingMode.AUTO:
        raise AssertionError(f"unreachable: every OrderingMode member is dispatched above (got {arm.ordering})")

    kwargs["use_anchored_heads"] = arm.anchored is OnOff.ON
    kwargs["use_monotonic_constraints"] = arm.monotonic is OnOff.ON
    kwargs["constrain_middle_class"] = arm.middle_class is OnOff.ON

    if arm.boundary is Boundary.OFF:
        kwargs["boundary_regularization_method"] = BoundaryRegularizationMethod.NONE
    elif arm.boundary is Boundary.HARDSIGMOID:
        kwargs["boundary_regularization_method"] = BoundaryRegularizationMethod.HARDSIGMOID
    elif arm.boundary is Boundary.PENALTY:
        kwargs["boundary_regularization_method"] = BoundaryRegularizationMethod.PENALTY
    else:
        raise AssertionError(f"unreachable: every Boundary member is dispatched above (got {arm.boundary})")
    return kwargs


def _base_kwargs(arm: Arm, k: int, seed: int, max_epochs: int, sigma_toy: float) -> dict[str, Any]:
    """Constructor kwargs shared by every layout -- everything except regression_strategy/regression_head_params."""
    return {
        "input_size": 1,
        "n_classes": k,
        "max_n_classes_for_probabilistic_path": k,
        "n_classes_selection_method": NClassesSelectionMethod.NONE,
        "n_epochs": max_epochs,
        "learning_rate": LEARNING_RATE,
        "early_stopping_rounds": None,  # driver-side incremental early stop instead -- design decision 2.
        "validation_fraction": 0.2,
        "test_fraction": 0.0,  # else BaseModel.fit() silently drops 10% of x_tr into an unused internal test split.
        "batch_size": BATCH_SIZE,
        "random_seed": seed,
        "calculate_feature_importance": False,
        **_coherence_kwargs(arm.spread, arm.likelihood, sigma_toy),
        **_supervision_kwargs(arm.supervision),
        **_patch_kwargs(arm),
    }


def _count_params(model: ProbabilisticRegressionModel) -> int:
    """Trainable-parameter count of `model.model`.

    NOT `model.get_num_parameters()`: `ProbabilisticRegressionModel`'s own override
    (`probabilistic_regression.py`, near its end) calls `sum(p.numel() for p in self.parameters()
    ...)` on the wrapper itself, which is not an `nn.Module` and has no `.parameters()` -- a
    pre-existing bug in code this task may not edit (models/ is PS-A1's). Bypassed here by going
    straight to the underlying `nn.Module`, exactly what `PyTorchModelBase.get_num_parameters`
    (the method this override shadows) already does correctly.
    """
    return sum(p.numel() for p in model.model.parameters() if p.requires_grad)


def _param_matched_hidden_size(base_kwargs: dict[str, Any], layout: Layout) -> tuple[int, int]:
    """PS-3's locked param-matching algorithm: raise hidden_size from 32 until params >= 0.9x separate-heads'.

    Builds throwaway (untrained) models via `ProbabilisticRegressionModel.build_model()` --
    cheap, no forward/backward pass needed just to count parameters.
    """
    ref_model = ProbabilisticRegressionModel(**{**base_kwargs, "regression_strategy": RegressionStrategy.SEPARATE_HEADS})
    ref_model.build_model()
    threshold = 0.9 * _count_params(ref_model)

    hidden_size = _MIN_HIDDEN_SIZE
    while hidden_size <= _MAX_HIDDEN_SIZE_SEARCH:
        cand_model = ProbabilisticRegressionModel(**{**base_kwargs, "regression_strategy": _LAYOUT_TO_STRATEGY[layout], "regression_head_params": {"hidden_size": hidden_size}})
        cand_model.build_model()
        count = _count_params(cand_model)
        if count >= threshold:
            return hidden_size, count
        hidden_size += 1
    raise RuntimeError(f"param-matching search exceeded hidden_size={_MAX_HIDDEN_SIZE_SEARCH} without reaching 0.9x the separate-heads reference ({threshold:.0f} params)")


def _build_constructor_kwargs(arm: Arm, k: int, seed: int, max_epochs: int, sigma_toy: float) -> tuple[dict[str, Any], int | None]:
    """Full constructor kwargs for one cell. Returns (kwargs, hidden_size_used_or_None)."""
    base = _base_kwargs(arm, k, seed, max_epochs, sigma_toy)
    kwargs = {**base, "regression_strategy": _LAYOUT_TO_STRATEGY[arm.layout]}
    if arm.layout is Layout.SEPARATE or arm.param_match is OnOff.OFF:
        return kwargs, None
    hidden_size, _ = _param_matched_hidden_size(base, arm.layout)
    kwargs["regression_head_params"] = {"hidden_size": hidden_size}
    return kwargs, hidden_size


# ---------------------------------------------------------------------------
# Forward pass diagnostics ("own NLL", per-class sigma, classifier confidence, ordering violations).
# ---------------------------------------------------------------------------


def _forward(model: ProbabilisticRegressionModel, x_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any, torch.Tensor | None]:
    """One no-grad forward pass, returning the model's raw `(final_predictions, classifier_logits_out, ..., per_head_outputs)` tuple."""
    model.model.eval()
    with torch.no_grad():
        return model.model(x_tensor)


def _own_nll_per_point(
    model: ProbabilisticRegressionModel,
    y_tensor: torch.Tensor,
    final_predictions: torch.Tensor,
    classifier_logits_out: torch.Tensor,
    per_head_outputs: torch.Tensor | None,
) -> torch.Tensor:
    """Per-point "own" NLL: `_calculate_custom_loss`'s regression-loss term ONLY, per point not averaged.

    Dispatches on `model.prob_reg_loss_type`/`model.uncertainty_method` exactly as
    `_calculate_custom_loss` does. FIXED_SIGMA_MIXTURE reuses `fixed_sigma_mixture_log_likelihood`
    directly (already per-point). MDN needs a per-point twin of `mdn_nll` (which reduces
    internally) -- the twin mirrors `mdn_nll`'s body verbatim, minus the final `.mean()`. The
    GAUSSIAN_LTV/CONSTANT branch mirrors `nll_loss`'s per-point formula (also reduced internally)
    for PROBABILISTIC, or plain per-point squared error for CONSTANT.
    """
    n = model.n_classes
    y_flat = y_tensor.view(-1)
    if model.prob_reg_loss_type is ProbRegLossType.MDN and per_head_outputs is not None:
        probs = torch.softmax(classifier_logits_out[:, :n], dim=-1)
        mus = per_head_outputs[:, :n, 0]
        log_vars = per_head_outputs[:, :n, 1]
        y = y_flat.view(-1, 1)
        log_component = -0.5 * (math.log(2 * math.pi) + log_vars + (y - mus) ** 2 * torch.exp(-log_vars))
        log_weights = torch.log(probs.clamp_min(1e-8))
        return -torch.logsumexp(log_weights + log_component, dim=-1)
    if model.prob_reg_loss_type is ProbRegLossType.FIXED_SIGMA_MIXTURE and per_head_outputs is not None:
        probs = torch.softmax(classifier_logits_out[:, :n], dim=-1)
        mus = per_head_outputs[:, :n, 0]
        return -fixed_sigma_mixture_log_likelihood(y_flat, probs, mus, sigma=model.fixed_sigma_train)
    if model.uncertainty_method is UncertaintyMethod.PROBABILISTIC:
        mean, log_var = final_predictions[:, 0], final_predictions[:, 1]
        return 0.5 * (math.log(2 * math.pi) + log_var + (y_flat - mean) ** 2 * torch.exp(-log_var))
    return (y_flat - final_predictions.squeeze(-1)) ** 2


def _per_class_sigma_stats(per_head_outputs: torch.Tensor, sigma_toy: float) -> tuple[dict[str, float], np.ndarray]:
    """{min, median, max} of the per-class median-over-batch fitted sigma, plus the raw per-class array.

    Under CONSTANT uncertainty (fixed_shared spread, either likelihood) there is no log-variance
    column by construction -- every class reports sigma_toy exactly (asserted in `--selftest`).
    """
    if per_head_outputs.shape[-1] == _PROBABILISTIC_OUTPUT_WIDTH:
        sigma = torch.exp(0.5 * per_head_outputs[..., 1])  # (N, k)
        per_class_median = sigma.median(dim=0).values.cpu().numpy()
    else:
        per_class_median = np.full(per_head_outputs.shape[1], sigma_toy)
    return {"min": float(np.min(per_class_median)), "median": float(np.median(per_class_median)), "max": float(np.max(per_class_median))}, per_class_median


def _classifier_max_prob_stats(probs: torch.Tensor) -> dict[str, float]:
    """{mean, max} over the batch of max_i p_i(x) -- the classifier's own confidence."""
    max_p = probs.max(dim=1).values
    return {"mean": float(max_p.mean().item()), "max": float(max_p.max().item())}


def _ordering_violations(model: ProbabilisticRegressionModel, classifier_logits_out: torch.Tensor, per_head_outputs: torch.Tensor) -> int:
    """Count of adjacent-class-mean pairs violating M_0 < M_1 < ... < M_{k-1} -- a hinge-count twin of `ordering_penalty`."""
    n = model.n_classes
    means = compute_ordering_means(classifier_logits_out[:, :n], per_head_outputs[:, :n, 0], top_decile_fraction=model.ordering_top_decile_fraction)
    if means.numel() < _MIN_CLASSES_FOR_ORDERING:
        return 0
    diffs = means[:-1] - means[1:] + model.ordering_constraint_margin
    return int((diffs > 0).sum().item())


# ---------------------------------------------------------------------------
# The convergence gate (design decision 2) and run_cell.
# ---------------------------------------------------------------------------


class _TrainingStoppedError(Exception):
    """Raised from `epoch_callback` once the driver-side convergence gate fires. Caught around `model.fit()`."""


def _true_class_sigma(toy: Toy, x_holdout: np.ndarray, y_binned: np.ndarray, n_classes: int, seed: int) -> np.ndarray:
    """Per-class TRUE within-slice residual sigma, by resampling y|x at every held-out point assigned to that class.

    Design decision 4's more precise final-block estimator: `_resample_y_given_x` is called
    `N_RESAMPLES_TRUE_SIGMA` times per class (once per resample, vectorized over that class's
    held-out x's); the per-point conditional variance across resamples is averaged, then
    sqrt'd -- the population RMS conditional sigma, not a single noisy draw.
    """
    true_sigma = np.full(n_classes, np.nan)
    for c in range(n_classes):
        mask = y_binned == c
        if not np.any(mask):
            continue
        x_c = x_holdout[mask]
        draws = np.stack([_resample_y_given_x(toy, x_c, seed=seed * 1_000_003 + r) for r in range(N_RESAMPLES_TRUE_SIGMA)], axis=0)
        per_point_var = draws.astype(np.float64).var(axis=0)
        true_sigma[c] = float(np.sqrt(np.mean(per_point_var)))
    return true_sigma


def run_cell(arm: Arm, toy: Toy, k: int, seed: int, max_epochs: int, check_every: int, sigma_toy: float) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Runs one (arm, toy, k, seed) cell: trains with the driver-side convergence gate, returns (case, perpoint arrays)."""
    n_train, n_holdout = _TOY_N[toy.value]
    x_tr, y_tr = _make_toy_xy(toy, n_train, seed)
    x_ho, y_ho = _make_toy_xy(toy, n_holdout, seed + 1000)  # disjoint seed offset (WSEL13's convention).

    kwargs, hidden_size_used = _build_constructor_kwargs(arm, k, seed, max_epochs, sigma_toy)
    model = ProbabilisticRegressionModel(**kwargs)
    device = model.device
    x_tr_t = torch.tensor(x_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
    x_ho_t = torch.tensor(x_ho, dtype=torch.float32, device=device)
    y_ho_t = torch.tensor(y_ho, dtype=torch.float32, device=device)

    trajectory: list[dict[str, Any]] = []
    gate_state = {"best": math.inf, "patience_counter": 0, "hit_cap": True}

    def on_epoch_end(epoch: int, model: ProbabilisticRegressionModel, val_loss: float | None) -> None:
        del val_loss  # base loop's own internal val split -- this driver scores its own held-out set instead.
        is_check = (epoch % check_every == 0) or (epoch == max_epochs - 1)
        if not is_check:
            return

        outputs_tr = _forward(model, x_tr_t)
        outputs_ho = _forward(model, x_ho_t)
        final_tr, logits_tr, _, _, heads_tr = outputs_tr
        final_ho, logits_ho, _, _, heads_ho = outputs_ho

        train_nll_own = float(_own_nll_per_point(model, y_tr_t, final_tr, logits_tr, heads_tr).mean().item())
        heldout_nll_own = float(_own_nll_per_point(model, y_ho_t, final_ho, logits_ho, heads_ho).mean().item())
        n = model.n_classes
        probs_ho = torch.softmax(logits_ho[:, :n], dim=-1)
        per_class_sigma, _ = _per_class_sigma_stats(heads_ho, sigma_toy)
        classifier_max_prob = _classifier_max_prob_stats(probs_ho)
        ordering_violations = _ordering_violations(model, logits_ho, heads_ho)

        trajectory.append({
            "epoch": epoch,
            "train_nll_own": train_nll_own,
            "heldout_nll_own": heldout_nll_own,
            "per_class_sigma": per_class_sigma,
            "classifier_max_prob": classifier_max_prob,
            "ordering_violations": ordering_violations,
        })

        if heldout_nll_own < gate_state["best"] - MIN_DELTA:
            gate_state["best"] = heldout_nll_own
            gate_state["patience_counter"] = 0
        else:
            gate_state["patience_counter"] += 1
        if gate_state["patience_counter"] >= PATIENCE_CHECKS:
            gate_state["hit_cap"] = False
            raise _TrainingStoppedError

    model.epoch_callback = on_epoch_end
    with contextlib.suppress(_TrainingStoppedError):
        model.fit(x_tr, y_tr)

    final_block, perpoint_final = _finalize(model, arm, toy, x_ho, y_ho, x_ho_t, y_ho_t, sigma_toy)
    final_block["hit_cap"] = gate_state["hit_cap"]
    final_block["converged"] = not gate_state["hit_cap"]
    final_block["params_count"] = _count_params(model)

    resolved_kwargs = {key: (value.value if isinstance(value, Enum) else value) for key, value in kwargs.items()}
    case = {
        "arm": arm.to_json(),
        "param_match": arm.param_match.value,
        "hidden_size_used": hidden_size_used,
        "toy": toy.value,
        "k": k,
        "seed": seed,
        "sigma_toy": sigma_toy,
        "resolved_kwargs": resolved_kwargs,
        "allow_retired_recorded": bool(kwargs.get("allow_retired_capacity_selection", False)),
        "provenance": run_provenance(),
        "trajectory": trajectory,
        "final": final_block,
    }
    perpoint: dict[str, np.ndarray] = {}
    if _own_nll_is_defined(arm):
        final_ho, logits_ho, _, _, heads_ho = _forward(model, x_ho_t)
        own_nll_perpoint = _own_nll_per_point(model, y_ho_t, final_ho, logits_ho, heads_ho)
        perpoint["own_nll_per_point"] = own_nll_perpoint.detach().cpu().numpy().astype(np.float32)
    perpoint.update(perpoint_final)
    return case, perpoint


def _own_nll_is_defined(arm: Arm) -> bool:
    """Whether `arm` has a defined own-NLL (D2 amendment, structure.md §4.5).

    `fixed_shared` arms have no fitted predictive density at all (a mean-only or
    fixed-sigma-scored network), so their own-NLL is UNDEFINED, not merely different --
    `final.heldout_nll_own` records `None` for them and they are excluded from the restricted
    own-NLL tie-break, never compared against.
    """
    return arm.spread is not Spread.FIXED_SHARED


def _own_nll_is_defined_json(arm_json: dict[str, Any]) -> bool:
    """`_own_nll_is_defined`'s twin over the on-disk `arm` JSON block (`--decide` reads JSON, not `Arm` instances)."""
    return arm_json["spread"] != Spread.FIXED_SHARED.value


def _finalize(
    model: ProbabilisticRegressionModel, arm: Arm, toy: Toy, x_ho: np.ndarray, y_ho: np.ndarray, x_ho_t: torch.Tensor, y_ho_t: torch.Tensor, sigma_toy: float
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """The `final` block (minus hit_cap/converged/params_count, attached by the caller)."""
    final_ho, logits_ho, _, _, heads_ho = _forward(model, x_ho_t)
    n = model.n_classes
    probs_ho = torch.softmax(logits_ho[:, :n], dim=-1)
    mus = heads_ho[:, :n, 0]

    fixed_sigma_ll_perpoint = fixed_sigma_mixture_log_likelihood(y_ho_t, probs_ho, mus, sigma=sigma_toy)
    heldout_fixed_sigma_mll = float(fixed_sigma_ll_perpoint.mean().item())
    heldout_nll_own = float(_own_nll_per_point(model, y_ho_t, final_ho, logits_ho, heads_ho).mean().item()) if _own_nll_is_defined(arm) else None

    pred_mean = final_ho[:, 0]
    pred_std = torch.exp(0.5 * final_ho[:, 1]) if final_ho.shape[-1] == _PROBABILISTIC_OUTPUT_WIDTH else torch.full_like(pred_mean, sigma_toy)
    resid = y_ho_t.view(-1) - pred_mean
    rmse = float(torch.sqrt((resid**2).mean()).item())
    coverage_1sigma = float((resid.abs() <= pred_std).float().mean().item())
    coverage_2sigma = float((resid.abs() <= 2 * pred_std).float().mean().item())

    boundaries = model.precomputed_class_boundaries[n]
    _, y_ho_binned = create_bins(data=y_ho, unique_bin_edges=boundaries)
    pred_class = probs_ho.argmax(dim=1).cpu().numpy()
    slice_accuracy = float(np.mean(pred_class == y_ho_binned))

    _, per_class_median = _per_class_sigma_stats(heads_ho, sigma_toy)
    true_sigma = _true_class_sigma(toy, x_ho, y_ho_binned, n, seed=int(model.random_seed))
    with np.errstate(invalid="ignore", divide="ignore"):
        ratios = per_class_median / true_sigma
    min_sigma_ratio = float(np.nanmin(ratios)) if np.any(~np.isnan(ratios)) else float("nan")

    final_block = {
        "heldout_nll_own": heldout_nll_own,
        "heldout_fixed_sigma_mll": heldout_fixed_sigma_mll,
        "rmse": rmse,
        "coverage_1sigma": coverage_1sigma,
        "coverage_2sigma": coverage_2sigma,
        "min_sigma_ratio": min_sigma_ratio,
        "slice_accuracy": slice_accuracy,
    }
    perpoint = {
        "fixed_sigma_ll_per_point": fixed_sigma_ll_perpoint.detach().cpu().numpy().astype(np.float32),
        "probs": probs_ho.detach().cpu().numpy().astype(np.float32),
        "mus": mus.detach().cpu().numpy().astype(np.float32),
        "y_true": np.asarray(y_ho, dtype=np.float32),
    }
    return final_block, perpoint


# ---------------------------------------------------------------------------
# Disk I/O: paths, _jsonable, --summarize, --decide, --rescore.
# ---------------------------------------------------------------------------


def _stage_dir(stage: Stage, out: str | None) -> str:
    return out if out is not None else os.path.join(RESULTS_BASE_DIR, f"PS{stage.value}")


def _cell_json_path(stage_dir: str, stage: Stage, arm: Arm, toy: Toy, k: int, seed: int) -> str:
    return os.path.join(stage_dir, f"ps{stage.value}_{arm.arm_id()}_{toy.value}_k{k}_seed{seed}.json")


def _perpoint_npz_path(cell_json_path: str) -> str:
    return cell_json_path[: -len(".json")] + "_perpoint.npz"


def _jsonable(obj: object) -> object:
    """Minimal JSON-safe coercion (numpy/torch scalars), matching `width_wsel13.py`'s helper."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def _load_cells(stage_dir: str, stage: Stage) -> list[dict[str, Any]]:
    cells = []
    for path in sorted(glob.glob(os.path.join(stage_dir, f"ps{stage.value}_*.json"))):
        if path.endswith("_decision.json") or os.path.basename(path) == "frozen.json":
            continue
        with open(path) as f:
            cell = json.load(f)
        cell["_arm_id"] = os.path.basename(path)[len(f"ps{stage.value}_") :].rsplit(f"_{cell['toy']}_k{cell['k']}_seed{cell['seed']}.json", 1)[0]
        cell["_json_path"] = path
        cells.append(cell)
    return cells


def summarize(stage_dir: str, stage: Stage) -> None:
    """Aggregates every per-cell JSON in `stage_dir` into `stage_dir/frozen.json`. Trains nothing."""
    cells = _load_cells(stage_dir, stage)
    groups: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        key = f"{cell['_arm_id']}__{cell['toy']}__k{cell['k']}"
        groups.setdefault(key, []).append(cell)

    per_group: dict[str, Any] = {}
    for key, group_cells in groups.items():
        finals = [c["final"] for c in group_cells]
        per_group[key] = {
            "n_seeds": len(group_cells),
            "seeds_present": sorted(c["seed"] for c in group_cells),
            "heldout_nll_own": [f["heldout_nll_own"] for f in finals],
            "heldout_fixed_sigma_mll": [f["heldout_fixed_sigma_mll"] for f in finals],
            "rmse": [f["rmse"] for f in finals],
            "coverage_1sigma": [f["coverage_1sigma"] for f in finals],
            "coverage_2sigma": [f["coverage_2sigma"] for f in finals],
            "min_sigma_ratio": [f["min_sigma_ratio"] for f in finals],
            "slice_accuracy": [f["slice_accuracy"] for f in finals],
            "params_count": [f["params_count"] for f in finals],
            "hit_cap": [f["hit_cap"] for f in finals],
            "converged": [f["converged"] for f in finals],
        }

    frozen = {"stage": stage.value, "n_cells": len(cells), "groups": per_group, "provenance": run_provenance()}
    os.makedirs(stage_dir, exist_ok=True)
    path = os.path.join(stage_dir, "frozen.json")
    with open(path, "w") as f:
        json.dump(_jsonable(frozen), f, indent=2)
    print(f"wrote {path}  ({len(cells)} cells, {len(groups)} groups)")


def _bootstrap_se(diff: np.ndarray, rng: np.random.Generator, n_boot: int = 1000) -> float:
    """Bootstrap SE of the mean of `diff`: 1000 point-resamples, vectorized."""
    n = diff.size
    idx = rng.integers(0, n, size=(n_boot, n))
    resample_means = diff[idx].mean(axis=1)
    return float(resample_means.std(ddof=1))


def _load_perpoint(cell: dict[str, Any], key: str) -> np.ndarray | None:
    """Loads one named array from a cell's companion `_perpoint.npz`, or None if absent."""
    path = _perpoint_npz_path(cell["_json_path"])
    if not os.path.exists(path):
        return None
    with np.load(path) as data:
        if key not in data:
            return None
        return data[key]


# k=2 is the truth-scale, decision-bearing pinned k (§4.2: "k is a condition, never a claim"; k=6
# is the deliberate-overcapacity observation cell). Genuinely un-locked ambiguity, resolved here as
# the reversible default: D2's comparisons run on k=2 only. Flagged for the root in the task report;
# now locked into structure.md §4.5 by the root's 2026-07-21 D2 amendment.
_D2_DECISION_K = 2

# D3-ce's amended reference toy (structure.md §4.5, amended 2026-07-21): the ONE toy whose classes
# are genuine components with a known intrinsic k, so a cross-entropy-supervised classifier's
# accuracy is a meaningful signal there. Every other toy's slice_accuracy is diagnostic only.
_D3_REFERENCE_TOY = Toy.TOY_B.value


def _pairwise_metric_result(cells_a: list[dict[str, Any]], cells_b: list[dict[str, Any]], rng: np.random.Generator, perpoint_key: str) -> dict[str, Any]:
    """Per-toy {mean_diff, combined_se} of arm A's vs arm B's per-point `perpoint_key` array, at k=2.

    Shared by BOTH the D2 primary comparison (`fixed_sigma_ll_per_point`) and the restricted
    own-NLL tie-break (`own_nll_per_point`) -- same bootstrap machinery, different array.
    """
    by_toy_seed_a = {(c["toy"], c["seed"]): c for c in cells_a if c["k"] == _D2_DECISION_K}
    by_toy_seed_b = {(c["toy"], c["seed"]): c for c in cells_b if c["k"] == _D2_DECISION_K}
    toys = sorted({toy for (toy, _seed) in by_toy_seed_a if any(t == toy for (t, _s) in by_toy_seed_b)})

    per_toy: dict[str, Any] = {}
    for toy in toys:
        seeds = sorted({s for (t, s) in by_toy_seed_a if t == toy} & {s for (t, s) in by_toy_seed_b if t == toy})
        if not seeds:
            continue
        seed_means, seed_ses = [], []
        for seed in seeds:
            arr_a = _load_perpoint(by_toy_seed_a[(toy, seed)], perpoint_key)
            arr_b = _load_perpoint(by_toy_seed_b[(toy, seed)], perpoint_key)
            if arr_a is None or arr_b is None or arr_a.shape != arr_b.shape:
                continue
            diff = arr_a.astype(np.float64) - arr_b.astype(np.float64)
            seed_means.append(float(diff.mean()))
            seed_ses.append(_bootstrap_se(diff, rng))
        if not seed_means:
            continue
        combined_se = float(np.sqrt(np.mean(np.square(seed_ses)) / len(seed_means)))
        per_toy[toy] = {"mean_diff": float(np.mean(seed_means)), "combined_se": combined_se, "n_seeds": len(seed_means)}
    return per_toy


def _compare_metric(per_toy: dict[str, Any], *, higher_is_better: bool) -> tuple[bool, bool]:
    """(a_beats_b, b_beats_a): A's metric better than B's by > 2xSE on >=2 toys, not worse beyond 2xSE anywhere.

    `mean_diff` is always (A - B). `higher_is_better=True` (the D2 primary, fixed-sigma-mll) reads
    it directly; `higher_is_better=False` (the restricted own-NLL tie-break, a loss) flips the sign
    -- same threshold arithmetic either way.
    """
    sign = 1.0 if higher_is_better else -1.0
    a_better = [t for t, r in per_toy.items() if sign * r["mean_diff"] > D2_SE_MULTIPLE * r["combined_se"]]
    b_better = [t for t, r in per_toy.items() if sign * r["mean_diff"] < -D2_SE_MULTIPLE * r["combined_se"]]
    a_beats_b = len(a_better) >= D2_MIN_WINNING_TOYS and len(b_better) == 0
    b_beats_a = len(b_better) >= D2_MIN_WINNING_TOYS and len(a_better) == 0
    return a_beats_b, b_beats_a


def _dominant(pool: list[str], beats: dict[str, set[str]]) -> str | None:
    """The arm in `pool` that beats every OTHER arm in `pool` (per `beats`), or None if no such arm exists."""
    if len(pool) == 1:
        return pool[0]
    for a in pool:
        if beats.get(a, set()) == set(pool) - {a}:
            return a
    return None


def _pairwise_matrix(
    pool: list[str], by_arm: dict[str, list[dict[str, Any]]], rng: np.random.Generator, perpoint_key: str, *, higher_is_better: bool
) -> tuple[dict[str, Any], dict[str, set[str]]]:
    """Every pairwise comparison within `pool` on `perpoint_key`. Returns (matrix for the decision JSON, beats-graph)."""
    matrix: dict[str, Any] = {}
    beats: dict[str, set[str]] = {a: set() for a in pool}
    for i, a in enumerate(pool):
        for b in pool[i + 1 :]:
            per_toy = _pairwise_metric_result(by_arm[a], by_arm[b], rng, perpoint_key)
            a_beats_b, b_beats_a = _compare_metric(per_toy, higher_is_better=higher_is_better)
            matrix[f"{a}__vs__{b}"] = {"per_toy": per_toy, "a_beats_b": a_beats_b, "b_beats_a": b_beats_a}
            if a_beats_b:
                beats[a].add(b)
            if b_beats_a:
                beats[b].add(a)
    return matrix, beats


_SPREAD_TIE_RANK = {Spread.FIXED_SHARED.value: 0, Spread.PER_INPUT.value: 1, Spread.ALL_CONSTANT.value: 2}
_SUPERVISION_TIE_RANK = {Supervision.CE.value: 0, Supervision.NONE.value: 1}
_LIKELIHOOD_TIE_RANK = {Likelihood.MIXTURE.value: 0, Likelihood.COLLAPSED.value: 1}


def _d4_rank(arm_json: dict[str, Any]) -> tuple[int, ...]:
    """D4's tie-break order: spread (fixed_shared>per_input>all_constant), supervision (ce), likelihood (mixture), patches (off)."""
    patches = arm_json["patches"]
    patch_rank = tuple(0 if patches[key] == "off" else 1 for key in ("ordering", "anchored", "monotonic", "middle_class", "boundary"))
    return (_SPREAD_TIE_RANK[arm_json["spread"]], _SUPERVISION_TIE_RANK[arm_json["supervision"]], _LIKELIHOOD_TIE_RANK[arm_json["likelihood"]], *patch_rank)


def decide(stage_dir: str, stage: Stage) -> dict[str, Any]:
    """Implements D1-D4 (§4.5) literally over every per-cell JSON found in `stage_dir`."""
    cells = _load_cells(stage_dir, stage)
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for c in cells:
        by_arm.setdefault(c["_arm_id"], []).append(c)

    disqualified: dict[str, list[str]] = {}
    d3_ce_report: dict[str, Any] = {}
    for arm_id, arm_cells in by_arm.items():
        reasons: list[str] = []
        # D1 -- degeneracy disqualifier. min_sigma_ratio at a checkpoint is approximated as
        # per_class_sigma.min / sigma_toy (design decision 4: exact for the homoscedastic screen
        # toys, an RMS-based approximation for the heteroscedastic ones).
        #
        # D1 is INAPPLICABLE to an arm with no fitted spread, and is skipped EXPLICITLY rather than
        # by luck. D1 asks "did a LEARNED spread collapse onto individual targets"; a `fixed_shared`
        # arm has none, so it cannot fail that way. The ratio guard below happens to skip it today
        # (per_class_sigma == sigma_toy exactly, so the ratio is 1.0), but that is accidental: for
        # such an arm the trajectory's `*_nll_own` entries hold squared error, not nats, so if the
        # ratio definition or D1_MIN_SIGMA_RATIO_THRESHOLD ever changed, the gap test would compare
        # a squared-error gap against a threshold in nats and could disqualify the Decision-26
        # control arm on a units mismatch. Same principle as the D2 amendment: where a quantity is
        # undefined for an arm, refuse it rather than let a number that happens to exist stand in.
        d1_cells = arm_cells if _own_nll_is_defined_json(arm_cells[0]["arm"]) else []
        for c in d1_cells:
            traj = c["trajectory"]
            gaps = [entry["heldout_nll_own"] - entry["train_nll_own"] for entry in traj]
            for i, entry in enumerate(traj):
                if entry["epoch"] <= D1_MIN_EPOCH or i < D1_MIN_CHECKPOINTS_FOR_TREND - 1:
                    continue
                ratio = entry["per_class_sigma"]["min"] / c["sigma_toy"] if c["sigma_toy"] else math.inf
                if ratio >= D1_MIN_SIGMA_RATIO_THRESHOLD or gaps[i] <= D1_NLL_GAP_THRESHOLD:
                    continue
                if gaps[i - 2] < gaps[i - 1] < gaps[i]:
                    reasons.append(
                        f"D1: toy={c['toy']} k={c['k']} seed={c['seed']} epoch={entry['epoch']} "
                        f"min_sigma_ratio~={ratio:.4f} gap={gaps[i]:.4f} (increasing over last 3 checkpoints)"
                    )
        # D3 -- identifiability disqualifier. AMENDED 2026-07-21 (see module docstring, design
        # decision 7): `ce` arms gate ONLY on the reference toy's (toy_b) SEED-MEAN slice_accuracy,
        # per k -- every other toy's slice_accuracy is a non-gating diagnostic, and an arm with no
        # toy_b cell at all is recorded as not-evaluated rather than disqualified by omission.
        # `none` arms are unchanged.
        supervision = arm_cells[0]["arm"]["supervision"]
        if supervision == Supervision.CE.value:
            reference_cells_by_k: dict[int, list[dict[str, Any]]] = {}
            for c in arm_cells:
                if c["toy"] == _D3_REFERENCE_TOY:
                    reference_cells_by_k.setdefault(c["k"], []).append(c)
            per_k_report: dict[str, Any] = {}
            for k, k_cells in reference_cells_by_k.items():
                seed_mean_sa = float(np.mean([c["final"]["slice_accuracy"] for c in k_cells]))
                bar = 1.5 / k
                passed = seed_mean_sa >= bar
                per_k_report[str(k)] = {"seed_mean_slice_accuracy": seed_mean_sa, "bar": bar, "n_seeds": len(k_cells), "pass": passed}
                if not passed:
                    reasons.append(f"D3(ce): toy={_D3_REFERENCE_TOY} k={k} seed_mean_slice_accuracy={seed_mean_sa:.4f} < 1.5/k={bar:.4f}")
            other_toy_diagnostics = {f"{c['toy']}_seed{c['seed']}_k{c['k']}": c["final"]["slice_accuracy"] for c in arm_cells if c["toy"] != _D3_REFERENCE_TOY}
            d3_ce_report[arm_id] = {
                "reference_toy": _D3_REFERENCE_TOY,
                "evaluated": bool(per_k_report),
                "per_k": per_k_report,
                "other_toy_slice_accuracy_diagnostic_only": other_toy_diagnostics,
            }
        else:
            for c in arm_cells:
                ov = c["trajectory"][-1]["ordering_violations"] if c["trajectory"] else 0
                if ov > 0:
                    reasons.append(f"D3(none): toy={c['toy']} k={c['k']} seed={c['seed']} ordering_violations={ov} > 0")
        if reasons:
            disqualified[arm_id] = reasons

    survivors = [a for a in by_arm if a not in disqualified]
    rng = np.random.default_rng(12345)

    # D2 PRIMARY (structure.md §4.5, amended 2026-07-21): heldout_fixed_sigma_mll, HIGHER is
    # better -- identically defined for every arm, so this is the metric that actually decides.
    primary_matrix, primary_beats = _pairwise_matrix(survivors, by_arm, rng, "fixed_sigma_ll_per_point", higher_is_better=True)
    primary_winner = _dominant(survivors, primary_beats) if survivors else None

    # D2 RESTRICTED TIE-BREAK: heldout_nll_own, lower is better, but ONLY when every surviving arm
    # has a defined own-NLL -- a `fixed_shared` survivor cannot be compared on this axis at all, so
    # its presence blocks the tie-break outright rather than being silently skipped past (design
    # decision 5 in the module docstring).
    null_own_nll_arms = [a for a in survivors if not _own_nll_is_defined_json(by_arm[a][0]["arm"])]
    tiebreak_matrix: dict[str, Any] = {}
    tiebreak_winner = None
    if primary_winner is None and not null_own_nll_arms and len(survivors) > 1:
        tiebreak_matrix, tiebreak_beats = _pairwise_matrix(survivors, by_arm, rng, "own_nll_per_point", higher_is_better=False)
        tiebreak_winner = _dominant(survivors, tiebreak_beats)

    if primary_winner is not None:
        winner, decided_by = primary_winner, "primary"
    elif tiebreak_winner is not None:
        winner, decided_by = tiebreak_winner, "own_nll_tiebreak"
    elif survivors:
        winner, decided_by = min(survivors, key=lambda a: _d4_rank(by_arm[a][0]["arm"])), "d4_tiebreak"
    else:
        winner, decided_by = None, None

    per_arm_summary = {
        arm_id: {
            "fixed_sigma_mll_by_toy_seed": {f"{c['toy']}_seed{c['seed']}": c["final"]["heldout_fixed_sigma_mll"] for c in arm_cells if c["k"] == _D2_DECISION_K},
            "heldout_nll_own_by_toy_seed": {f"{c['toy']}_seed{c['seed']}": c["final"]["heldout_nll_own"] for c in arm_cells if c["k"] == _D2_DECISION_K},
            "own_nll_defined": _own_nll_is_defined_json(arm_cells[0]["arm"]),
        }
        for arm_id, arm_cells in by_arm.items()
    }

    decision = {
        "stage": stage.value,
        "k_read": _D2_DECISION_K,
        "primary_metric": "heldout_fixed_sigma_mll",
        "n_arms": len(by_arm),
        "disqualified": disqualified,
        "d3_ce_report": d3_ce_report,
        "survivors": survivors,
        "null_own_nll_arms": null_own_nll_arms,
        "per_arm_summary": per_arm_summary,
        "primary_matrix": primary_matrix,
        "primary_winner": primary_winner,
        "tiebreak_matrix": tiebreak_matrix,
        "tiebreak_winner": tiebreak_winner,
        "decided_by": decided_by,
        "winner": winner,
        "winner_arm": by_arm[winner][0]["arm"] if winner is not None else None,
        "provenance": run_provenance(),
    }
    os.makedirs(stage_dir, exist_ok=True)
    path = os.path.join(stage_dir, f"ps{stage.value}_decision.json")
    with open(path, "w") as f:
        json.dump(_jsonable(decision), f, indent=2)
    print(f"wrote {path}  winner={winner}  decided_by={decided_by}")
    return decision


def rescore(stage_dir: str, factor: RescoreFactor) -> None:
    """PS-4's mandatory robustness re-score at sigma_toy/2 or 2*sigma_toy, from stored (probs, mus, y) -- no retraining."""
    results: dict[str, Any] = {}
    for path in sorted(glob.glob(os.path.join(stage_dir, "ps4_*.json"))):
        if path.endswith("_decision.json") or os.path.basename(path) == "frozen.json":
            continue
        with open(path) as f:
            cell = json.load(f)
        perpoint_path = _perpoint_npz_path(path)
        if not os.path.exists(perpoint_path):
            continue
        with np.load(perpoint_path) as data:
            if "probs" not in data:
                continue
            probs = torch.tensor(data["probs"])
            mus = torch.tensor(data["mus"])
            y_true = torch.tensor(data["y_true"])
        sigma_toy = cell["sigma_toy"]
        new_sigma = sigma_toy / 2.0 if factor is RescoreFactor.HALF else sigma_toy * 2.0
        mll = float(fixed_sigma_mixture_log_likelihood(y_true, probs, mus, sigma=new_sigma).mean().item())
        results[os.path.basename(path)] = {"sigma_toy": sigma_toy, "sigma_used": new_sigma, "heldout_fixed_sigma_mll": mll}

    out = {"factor": factor.value, "n_cells": len(results), "cells": results, "provenance": run_provenance()}
    os.makedirs(stage_dir, exist_ok=True)
    path = os.path.join(stage_dir, f"rescore_{factor.value}.json")
    with open(path, "w") as f:
        json.dump(_jsonable(out), f, indent=2)
    print(f"wrote {path}  ({len(results)} cells)")


# ---------------------------------------------------------------------------
# --selftest
# ---------------------------------------------------------------------------


def run_selftest() -> bool:
    """Tiny known-answer wiring check. No real cell (full epoch budget) is ever run here."""
    ok = True

    # sigma_toy matches the generator constants for every toy.
    expected_homoscedastic = {Toy.TOY_A: TOY_A_SIGMA, Toy.TOY_B: TOY_B_SIGMA, Toy.BROAD_UNIMODAL: BROAD_UNIMODAL_SIGMA * math.sqrt(1.0 + BROAD_UNIMODAL_SEPARATION**2 / 4.0)}
    for toy, expected in expected_homoscedastic.items():
        got = _sigma_for_toy(toy)
        this_ok = math.isclose(got, expected, rel_tol=1e-9)
        print(f"[structure-battery selftest] sigma_toy({toy.value})={got:.6f} (want {expected:.6f})  {'PASS' if this_ok else 'FAIL'}")
        ok = ok and this_ok
    for toy in (Toy.TOY_C_BROAD, Toy.TOY_C, Toy.TOY_E, Toy.TOY_D):
        got = _sigma_for_toy(toy)
        this_ok = math.isfinite(got) and got > 0.0
        print(f"[structure-battery selftest] sigma_toy({toy.value})={got:.6f} finite and positive  {'PASS' if this_ok else 'FAIL'}")
        ok = ok and this_ok

    # The coherence matrix refuses all_constant x collapsed.
    try:
        _coherence_kwargs(Spread.ALL_CONSTANT, Likelihood.COLLAPSED, sigma_toy=0.3)
        refused = False
    except ValueError:
        refused = True
    print(f"[structure-battery selftest] all_constant x collapsed refused: {refused}  {'PASS' if refused else 'FAIL'}")
    ok = ok and refused

    # Every OTHER (spread, likelihood) pair builds a real, trainable model at tiny scale.
    torch.manual_seed(0)
    sigma_toy = _sigma_for_toy(Toy.TOY_A)
    legal_pairs = [
        (Spread.PER_INPUT, Likelihood.COLLAPSED),
        (Spread.PER_INPUT, Likelihood.MIXTURE),
        (Spread.FIXED_SHARED, Likelihood.COLLAPSED),
        (Spread.FIXED_SHARED, Likelihood.MIXTURE),
        (Spread.ALL_CONSTANT, Likelihood.MIXTURE),
    ]
    for spread, likelihood in legal_pairs:
        arm = Arm(Supervision.NONE, spread, likelihood, OrderingMode.AUTO, OnOff.OFF, OnOff.OFF, OnOff.ON, Boundary.OFF, Layout.SEPARATE, OnOff.OFF)
        case = None
        try:
            case, perpoint = run_cell(arm, Toy.TOY_A, k=3, seed=0, max_epochs=2, check_every=1, sigma_toy=sigma_toy)
            heldout_nll_own = case["final"]["heldout_nll_own"]
            # D2 amendment (structure.md §4.5): fixed_shared arms have no fitted predictive
            # density, so own-NLL is UNDEFINED (None) for them and finite for every other arm.
            if spread is Spread.FIXED_SHARED:
                own_nll_ok = heldout_nll_own is None and "own_nll_per_point" not in perpoint
            else:
                own_nll_ok = heldout_nll_own is not None and math.isfinite(heldout_nll_own) and "own_nll_per_point" in perpoint
            built_ok = len(case["trajectory"]) >= 1 and math.isfinite(case["final"]["heldout_fixed_sigma_mll"]) and own_nll_ok
        except Exception as exc:  # selftest must report every arm's status, not abort on the first exception.
            built_ok = False
            print(f"[structure-battery selftest] {spread.value} x {likelihood.value}: raised {type(exc).__name__}: {exc}")
        print(
            f"[structure-battery selftest] {spread.value} x {likelihood.value}: trains, finite fixed-sigma mll, own-NLL null iff fixed_shared: "
            f"{built_ok}  {'PASS' if built_ok else 'FAIL'}"
        )
        ok = ok and built_ok

        if spread is Spread.FIXED_SHARED and case is not None:
            per_class = case["trajectory"][-1]["per_class_sigma"]
            fixed_ok = math.isclose(per_class["min"], sigma_toy, rel_tol=1e-6) and math.isclose(per_class["max"], sigma_toy, rel_tol=1e-6)
            print(f"[structure-battery selftest] fixed_shared x {likelihood.value}: per_class_sigma == sigma_toy for every class: {fixed_ok}  {'PASS' if fixed_ok else 'FAIL'}")
            ok = ok and fixed_ok

    print(f"[structure-battery selftest] {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """Parses args and dispatches to `--selftest` / `--summarize` / `--decide` / `--rescore` / one real cell."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", type=int, choices=[s.value for s in Stage], default=None)
    parser.add_argument("--supervision", type=str, choices=[s.value for s in Supervision], default=Supervision.NONE.value)
    parser.add_argument("--spread", type=str, choices=[s.value for s in Spread], default=None)
    parser.add_argument("--likelihood", type=str, choices=[likelihood.value for likelihood in Likelihood], default=None)
    parser.add_argument("--ordering", type=str, choices=[o.value for o in OrderingMode], default=OrderingMode.AUTO.value)
    parser.add_argument("--anchored", type=str, choices=[o.value for o in OnOff], default=OnOff.OFF.value)
    parser.add_argument("--monotonic", type=str, choices=[o.value for o in OnOff], default=OnOff.OFF.value)
    parser.add_argument("--middle-class", dest="middle_class", type=str, choices=[o.value for o in OnOff], default=OnOff.ON.value)
    parser.add_argument("--boundary", type=str, choices=[b.value for b in Boundary], default=Boundary.OFF.value)
    parser.add_argument("--layout", type=str, choices=[layout.value for layout in Layout], default=Layout.SEPARATE.value)
    parser.add_argument("--param-match", dest="param_match", type=str, choices=[o.value for o in OnOff], default=OnOff.OFF.value)
    parser.add_argument("--toy", type=str, choices=[t.value for t in Toy], default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--check-every", type=int, default=DEFAULT_CHECK_EVERY)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--decide", action="store_true")
    parser.add_argument("--rescore", type=str, choices=[r.value for r in RescoreFactor], default=None)
    args = parser.parse_args()

    if args.selftest:
        raise SystemExit(0 if run_selftest() else 1)

    if args.stage is None:
        parser.error("--stage is required (or pass --selftest).")
    stage = Stage(args.stage)
    stage_dir = _stage_dir(stage, args.out)

    if args.summarize:
        summarize(stage_dir, stage)
        return
    if args.decide:
        decide(stage_dir, stage)
        return
    if args.rescore is not None:
        rescore(stage_dir, RescoreFactor(args.rescore))
        return

    if args.spread is None or args.likelihood is None or args.toy is None or args.k is None or args.seed is None:
        parser.error("--spread, --likelihood, --toy, --k and --seed are all required for a real cell (or pass --selftest / --summarize / --decide / --rescore).")

    arm = Arm(
        supervision=Supervision(args.supervision),
        spread=Spread(args.spread),
        likelihood=Likelihood(args.likelihood),
        ordering=OrderingMode(args.ordering),
        anchored=OnOff(args.anchored),
        monotonic=OnOff(args.monotonic),
        middle_class=OnOff(args.middle_class),
        boundary=Boundary(args.boundary),
        layout=Layout(args.layout),
        param_match=OnOff(args.param_match),
    )
    toy = Toy(args.toy)
    sigma_toy = _sigma_for_toy(toy)
    print(f"[structure-battery] stage={stage.value} arm={arm.arm_id()} toy={toy.value} k={args.k} seed={args.seed} max_epochs={args.max_epochs}", flush=True)

    os.makedirs(stage_dir, exist_ok=True)
    case, perpoint = run_cell(arm, toy, args.k, args.seed, args.max_epochs, args.check_every, sigma_toy)

    cell_path = _cell_json_path(stage_dir, stage, arm, toy, args.k, args.seed)
    with open(cell_path, "w") as f:
        json.dump(_jsonable(case), f, indent=2)
    print(f"wrote {cell_path}")

    perpoint_path = _perpoint_npz_path(cell_path)
    np.savez(perpoint_path, **perpoint)
    print(f"wrote {perpoint_path}")

    final = case["final"]
    print(
        f"final heldout_nll_own={final['heldout_nll_own']:.4f}  heldout_fixed_sigma_mll={final['heldout_fixed_sigma_mll']:.4f}  "
        f"hit_cap={final['hit_cap']}  converged={final['converged']}"
    )


if __name__ == "__main__":
    main()
