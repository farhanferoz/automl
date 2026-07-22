"""WSEL-6 -- how much data does width selection need? (`docs/plans/capacity_programme/width.md` ~908-930).

The selection split every other width driver inherits is a hardcoded 50/50 even/odd carve of the
training portion (`kdropout_converged_width_experiment.py:273-276`'s `p1_idx`/`p2_idx`, the SAME
carve `converged_width_experiment.run_case` and `sinc_width_experiment`'s phase-1/phase-2 arm use):
`p1` (even indices) trains the underlying net(s), `p2` (odd indices, the "held-out-within-train"
half) is the SELECTION POOL the three arms of `width.md` SS1 read from. Nobody has ever measured
how much of `p2` the selection step actually needs. This driver sweeps the FRACTION of the training
portion (`n_train`) handed to selection -- `{5, 10, 15, 25, 40}%`, always <= p2's own 50%, so every
fraction is a genuine SUBSET of the selection pool, never an over-read of it -- on the SS3.8 canonical
toy suite's tier 1 + tier 2 (this task's ROOT-assigned suite row), for all three SS1 arms, holding
everything else fixed.

**Nested subsampling.** `p2` is shuffled ONCE per (tier, seed) with a seeded RNG; a fraction `f`'s
selection subsample is the first `round(f * n_train)` shuffled `p2` points. Fractions are therefore
strictly nested (5% subset of 10% subset of ... of 40%), so a saturation curve reflects added DATA,
not sampling noise between independently-drawn subsets at each fraction.

**Training is independent of `fraction` -- caching choice, stated per the task brief.** Only the
SELECTION step reads a fraction of `p2`; the underlying net(s) always train on the full `p1` split,
regardless of fraction or which arm is asked for. W-SHARED and W-PERINPUT are read off the SAME
trained multi-head `FlexibleWidthNN(widths=1..w_max)` (`width.md` SS1: "training is NOT a variable
between those two"); W-SWEEP needs 12 dedicated single-width nets. Both are therefore CACHED to disk,
keyed by `(tier, seed[, width])`, the first time any cell needs them, and every later cell (a
different fraction, or the other of W-SHARED/W-PERINPUT) loads the cached `state_dict` instead of
retraining -- one training run per (tier, seed) for the shared net (`_get_or_train_shared`), one per
(tier, seed, width) for each sweep net (`_get_or_train_sweep_all`), instead of paying full training
cost on every one of the 5 fractions x 3 arms that reads it. Cost: one extra `torch.save`/`torch.load`
round trip per cache hit, negligible next to a training run.

**Training protocol -- `FlexibleWidthNN`, `width_wsel4.py`'s ported-arm protocol reused verbatim**
(Tanh activation, full-batch [`w4.PORTED_BATCH_SIZE`], LR `w4.PORTED_LR_DEFAULT`, epoch cap
`w4.PORTED_N_EPOCHS_CAP`, patience `w4.PORTED_PATIENCE`/`w4.PORTED_MIN_DELTA`) -- including that
file's own finding that this class trains stably on RAW (unstandardized) x/y at this toy's scale, so
no `converged_width_experiment._standardize_fit` detour is needed here. `FlexibleWidthNN` always sums
its loss over EVERY configured width every step (`models/flexnn/width/model.py`'s module docstring:
the class's own ALL-schedule shape, the MASTER-Decision-31 programme default) -- true for both tiers
here, so `training_schedule` is always `"all"` in every cell this driver writes.

**Tier 1** (`--toy hetero`) trains via the established `_fit_single` bypass (`width_wsel4._train_ported_
default`'s exact pattern) under the class's own built-in MSE criterion -- exactly SS3.7's fixed-sigma
likelihood on `hetero`'s single constant sigma. **Tier 2** (`--toy hetero3`) needs the fixed-sigma
WEIGHTED squared error (SS3.7, `width_candidates.weighted_squared_error`); `FlexibleWidthNN`'s built-in
criterion has no sigma-weighting hook, so `_train_tier2_weighted` reproduces the SAME full-batch-Adam
/ best-state-checkpointing shape `width_wsel4._train_ported_escalated` and `width_wsel13._train_all_
schedule_weighted_to_convergence` / `width_wsel16._train_custom_to_convergence`'s tier-2 branch already
use for exactly this class of problem, wired onto `FlexibleWidthNN.model` (`FlexibleWidthNNModule`)
instead of a raw `nested_width_net` class.

**Selection, one rule for all three arms (`width.md` SS1: "the same rule applies to W-SWEEP's curve, or
W-SHARED and W-SWEEP are not answering the same question").** W-SHARED: `FlexibleWidthNN.
fit_global_selector` (landed 2026-07-22). W-PERINPUT: `FlexibleWidthNN.fit_router` (same class, same
trained net). W-SWEEP: no single net spans every width, so its held-out error TABLE is built by hand
(the same public `predict(..., width=k)` call `fit_global_selector` uses internally, once per dedicated
net) and fed to the SAME `automl_package.utils.capacity_selection.cheapest_within_tolerance` selector --
never re-derived.

**Cost fields** (WSEL-5's `automl_package/utils/capacity_accounting.py`, landed 2026-07-22): W-SHARED /
W-PERINPUT use `global_cheap_cost`/`per_input_cost` with `training_macs` computed via `sweep_cost` on
the shared net's actual (widths, epochs, n_train) -- exact, not approximate: `FlexibleWidthNNModule.
forward` recomputes the trunk once per configured width with no sharing in its own computation graph
(`model.py:158-165`), which is exactly what `sweep_cost`'s formula prices. W-SWEEP sums `sweep_cost`/
`held_out_read_cost` per dedicated net (its 12 nets are not one `executed_flops`-registered object, so
`global_sweep_cost`'s single-`net` signature does not fit it directly). Reported, not gating.

**Emits** (`width.md` SS3.6, this task's two owned frozen constants): `WSEL6/frozen.json`'s `fraction`
(the smallest fraction at which EVERY (tier, arm) is within its own twice-bootstrap-SE noise band of its
best fraction, via `cheapest_within_tolerance` read over a `(seeds, fractions)` curve per (tier, arm) --
the same selection rule turned on itself; the largest swept fraction plus `saturated: false` if any
(tier, arm) never gets there) and `data_limited` (one boolean per `(toy, arm)`: true iff that (tier, arm)
curve has NOT saturated by the largest fraction -- SS3.5's "W-PERINPUT still improving at the largest
fraction" branch, generalized to every arm since the same non-saturation reasoning applies to any of
them). A saturation plot (`WSEL6/saturation.png`, mean held-out MSE +/- SE vs fraction, one line per arm,
one panel per tier) backs the frozen numbers visually.

Driver CLI contract (root-run grid; this file is never run over the full grid by its author):
  `--tier {1,2} --seed <int> --fraction {5,10,15,25,40} --arm {w_shared,w_perinput,w_sweep}` runs ONE
  cell, writing its per-cell JSON immediately (and, on a cache miss, the underlying net's `state_dict`
  under `WSEL6/_cache/`).
  `--summarize` aggregates every per-cell JSON on disk into `WSEL6/frozen.json` plus the saturation plot.
  `--selftest` runs every (tier, arm) at 2 tiny fractions on a w_max=3 toy, in a temp dir, plus an
  explicit cache-reuse check -- no real cell is ever run here.

Non-goals (task brief): no real data (WSEL-9's budget); no architecture changes; no router-architecture
variation (WSEL-7's job); no single-head arms (2026-07-22 coupling ruling: selection studies run
multi-head-only); no plan-doc edits.

Usage:
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel6.py --selftest
    AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel6.py --tier 1 --seed 0 --fraction 10 --arm w_shared
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/width_wsel6.py --summarize
"""

from __future__ import annotations

import argparse
import enum
import glob
import json
import math
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

mpl.use("Agg")

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root, so `import automl_package` works

import converged_width_experiment as cwe  # noqa: E402 -- VAL_EVERY, the phase-1 train/val carve convention
import nested_width_net as nwn  # noqa: E402 -- Toy, WidthSchedule, make_hetero/make_hetero3, HETERO_NOISE_SIGMA, HETERO3_NOISY_SIGMA
import width_candidates as wc  # noqa: E402 -- weighted_squared_error, SS3.7's tier-2 objective, implemented once
import width_wsel4 as w4  # noqa: E402 -- PORTED_* ported-arm protocol constants + _replay, reused verbatim (see module docstring)

from automl_package.enums import ActivationFunction, CapacitySelection, TaskType  # noqa: E402
from automl_package.models.flexnn.routing import (  # noqa: E402 -- WSEL-7 re-run: direct construction with new_default args (wsel7/wsel16's sanctioned pattern)
    DEFAULT_TOLERANCE,
    DistilledCapacityRouter,
)
from automl_package.models.flexnn.width.model import FlexibleWidthNN  # noqa: E402
from automl_package.utils.capacity_accounting import executed_flops, global_cheap_cost, held_out_read_cost, per_input_cost, sweep_cost  # noqa: E402
from automl_package.utils.capacity_selection import DEFAULT_N_BOOT, cheapest_within_tolerance  # noqa: E402
from automl_package.utils.run_provenance import run_provenance  # noqa: E402

RESULTS_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "WSEL6")
_CACHE_DIRNAME = "_cache"

SEEDS = (0, 1, 2)
W_MAX = 12
FRACTIONS_PCT = (5, 10, 15, 25, 40)  # of the training portion (n_train) -- width.md WSEL-6's suggested sweep, taken as-is.

_HETERO3_NOISY_REGION = 2  # nwn.make_hetero3's region id for the noisy-easy tail (HETERO3_NOISY_SIGMA).


class Tier(enum.IntEnum):
    """SS3.8's canonical toy suite tiers this task runs (WSEL-6's ROOT-assigned row: tier 1 + tier 2)."""

    ONE = 1  # the reference cell (hetero, n_train=1500).
    TWO = 2  # the noisy-easy control (hetero3, n_train=2250).


class Arm(enum.Enum):
    """Closed set: `width.md` SS1's three complete systems -- WSEL-6's whole comparison."""

    W_SHARED = "w_shared"  # FlexibleWidthNN.fit_global_selector -- ONE width for the dataset, cheap read.
    W_PERINPUT = "w_perinput"  # FlexibleWidthNN.fit_router -- a width PER INPUT, distilled router.
    W_SWEEP = "w_sweep"  # 12 dedicated per-width nets, cheapest_within_tolerance over their held-out curve.


@dataclass(frozen=True)
class _TierConfig:
    """One tier's toy/size config, `width.md` SS3.8's WSEL-6 row (~908-930)."""

    toy: nwn.Toy
    n_train: int
    n_test: int
    sigma: float  # ignored by hetero3 downstream (nwn.make_hetero3 has no sigma arg); kept for uniform bookkeeping.


_TIER_CONFIG: dict[Tier, _TierConfig] = {
    Tier.ONE: _TierConfig(toy=nwn.Toy.HETERO, n_train=1500, n_test=500, sigma=nwn.HETERO_NOISE_SIGMA),
    Tier.TWO: _TierConfig(toy=nwn.Toy.HETERO3, n_train=2250, n_test=750, sigma=nwn.HETERO_NOISE_SIGMA),
}


# ---------------------------------------------------------------------------
# Data -- ONE split builder for every arm (MASTER Decision 15's single-difference rule), the
# canonical p1 (train)/p2 (selection pool) carve, plus the fraction-limited selection subsample.
# ---------------------------------------------------------------------------


def _sigma_true(toy: nwn.Toy, region: np.ndarray, sigma: float) -> np.ndarray:
    """Per-point TRUE noise sigma (`width.md` SS3.7), RAW scale -- never estimated, never learned.

    Unlike `width_wsel13`/`width_wsel16`'s twins of this helper, no `/ norm["sy"]` correction is
    needed: `FlexibleWidthNN` trains on RAW (unstandardized) `x`/`y` here (`width_wsel4.py`'s
    already-vetted, non-escalated ported-arm protocol -- see module docstring), so `sigma_true` stays
    in the generator's own units throughout.
    """
    if toy is nwn.Toy.HETERO3:
        return np.where(region == _HETERO3_NOISY_REGION, nwn.HETERO3_NOISY_SIGMA, nwn.HETERO_NOISE_SIGMA).astype(np.float32)
    return np.full(region.shape, sigma, dtype=np.float32)


def _build_split(tier: Tier, seed: int, *, n_train: int | None = None, n_test: int | None = None) -> dict[str, Any]:
    """Builds the canonical p1 (trains the net)/p2 (selection pool)/test split for one (tier, seed).

    `p1`/`p2` are the SAME hardcoded 50/50 even/odd carve every other width driver uses
    (`kdropout_converged_width_experiment.py:273-276`; `converged_width_experiment.run_case`;
    `sinc_width_experiment`'s phase-1/phase-2 arm) -- the carve this task exists to measure, not to
    replace. `p1` is further split into train/val via the standard `VAL_EVERY` convergence-monitoring
    carve. `p2` is shuffled ONCE, seeded, so every fraction's subsample (`_selection_subsample`) is a
    strict prefix of every larger fraction's -- nested, not independently resampled.
    """
    cfg = _TIER_CONFIG[tier]
    n_train = cfg.n_train if n_train is None else n_train
    n_test = cfg.n_test if n_test is None else n_test
    if cfg.toy is nwn.Toy.HETERO3:
        x_tr, y_tr, region_tr = nwn.make_hetero3(n_train, seed)
        x_te, y_te, region_te = nwn.make_hetero3(n_test, seed + 500)
    else:
        x_tr, y_tr, region_tr = nwn.make_hetero(n_train, seed, sigma=cfg.sigma)
        x_te, y_te, region_te = nwn.make_hetero(n_test, seed + 500, sigma=cfg.sigma)

    p1_idx = np.arange(0, n_train, 2)
    p2_idx = np.arange(1, n_train, 2)
    x_p1, y_p1, region_p1 = x_tr[p1_idx], y_tr[p1_idx], region_tr[p1_idx]
    x_p2, y_p2, region_p2 = x_tr[p2_idx], y_tr[p2_idx], region_tr[p2_idx]

    val_mask = (np.arange(len(x_p1)) % cwe.VAL_EVERY) == 0

    perm = np.random.default_rng(seed).permutation(len(x_p2))
    x_p2, y_p2, region_p2 = x_p2[perm], y_p2[perm], region_p2[perm]

    return {
        "n_train": n_train,
        "n_test": n_test,
        "x_train": x_p1[~val_mask],
        "y_train": y_p1[~val_mask],
        "region_train": region_p1[~val_mask],
        "x_p1_val": x_p1[val_mask],
        "y_p1_val": y_p1[val_mask],
        "region_p1_val": region_p1[val_mask],
        "x_p2": x_p2,
        "y_p2": y_p2,
        "region_p2": region_p2,
        "x_test": x_te,
        "y_test": y_te,
        "region_test": region_te,
    }


def _selection_subsample(split: dict[str, Any], fraction_pct: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The first `round(fraction_pct% * n_train)` shuffled `p2` points -- always <= p2's own 50% of `n_train`."""
    n_select = round((fraction_pct / 100.0) * split["n_train"])
    n_select = max(1, min(n_select, len(split["x_p2"])))
    return split["x_p2"][:n_select], split["y_p2"][:n_select], split["region_p2"][:n_select]


# ---------------------------------------------------------------------------
# Training -- ONE net-construction helper and ONE dispatcher (tier 1 plain MSE / tier 2 weighted),
# reused for BOTH the multi-head net (W-SHARED/W-PERINPUT) and each W-SWEEP dedicated single-width net.
# ---------------------------------------------------------------------------


def _new_model(widths: tuple[int, ...], seed: int, *, max_epochs: int, patience: int, lr: float) -> FlexibleWidthNN:
    """Builds an untrained `FlexibleWidthNN`, `width_wsel4.py`'s ported-arm protocol verbatim (module docstring)."""
    return FlexibleWidthNN(
        input_size=1,
        output_size=1,
        task_type=TaskType.REGRESSION,
        widths=widths,
        learning_rate=lr,
        n_epochs=max_epochs,
        early_stopping_rounds=patience,
        batch_size=w4.PORTED_BATCH_SIZE,
        random_seed=seed,
        calculate_feature_importance=False,
        capacity_selection=CapacitySelection.FIXED,
        # Tanh, not the class's ReLU default -- width_wsel4.py's confound-doctrine finding (its module
        # docstring: the control arm's width classes are Tanh; a ReLU ported arm under-fit the train
        # set at every width >= 4). Reused verbatim, not re-derived.
        activation=ActivationFunction.TANH,
    )


def _train_tier1(
    widths: tuple[int, ...], seed: int, x_tr: np.ndarray, y_tr: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, *, max_epochs: int, patience: int, lr: float
) -> tuple[FlexibleWidthNN, list[float]]:
    """Tier 1 (`hetero`): the established `_fit_single` bypass, `width_wsel4._train_ported_default`'s exact pattern.

    `FlexibleWidthNN`'s own built-in criterion (`nn.MSELoss`, summed over every configured width every
    step) IS SS3.7's fixed-sigma likelihood on `hetero`'s single constant sigma -- no custom loop needed.
    """
    model = _new_model(widths, seed, max_epochs=max_epochs, patience=patience, lr=lr)
    x_tr_in = x_tr.reshape(-1, 1).astype(np.float32)
    x_val_in = x_val.reshape(-1, 1).astype(np.float32)
    _best_epoch, val_loss_history = model._fit_single(x_tr_in, y_tr, x_val=x_val_in, y_val=y_val)
    return model, val_loss_history


def _train_tier2_weighted(
    widths: tuple[int, ...],
    seed: int,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    sigma_tr: np.ndarray,
    sigma_val: np.ndarray,
    *,
    max_epochs: int,
    patience: int,
    min_delta: float,
    lr: float,
) -> tuple[FlexibleWidthNN, list[float]]:
    """Tier 2 (`hetero3`): full-batch Adam under SS3.7's fixed-sigma weighted squared error.

    `FlexibleWidthNN`'s built-in criterion has no sigma-weighting hook, so this reproduces the SAME
    full-batch / best-state-checkpointing shape `width_wsel4._train_ported_escalated` and
    `width_wsel13._train_all_schedule_weighted_to_convergence` / `width_wsel16._train_custom_to_
    convergence`'s tier-2 branch already use for this exact problem (module docstring), wired onto
    `FlexibleWidthNN.model` (`FlexibleWidthNNModule`, whose `.forward` already sums every configured
    width every step -- the class's own ALL-schedule shape) instead of a raw `nested_width_net` class.
    """
    model = _new_model(widths, seed, max_epochs=max_epochs, patience=patience, lr=lr)
    model.input_size = 1
    torch.manual_seed(seed)
    model.build_model()
    opt = torch.optim.Adam(model.model.parameters(), lr=lr)

    x_tr_t = torch.tensor(x_tr, dtype=torch.float32, device=model.device).reshape(-1, 1)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=model.device)
    x_val_t = torch.tensor(x_val, dtype=torch.float32, device=model.device).reshape(-1, 1)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=model.device)
    sigma_tr_t = torch.tensor(sigma_tr, dtype=torch.float32, device=model.device)
    sigma_val_t = torch.tensor(sigma_val, dtype=torch.float32, device=model.device)

    best_val, patience_counter, best_state = float("inf"), 0, None
    val_loss_history: list[float] = []
    for _epoch in range(max_epochs):
        model.model.train()
        opt.zero_grad()
        stacked = model.model(x_tr_t)  # (len(widths), N, 1) -- every configured width, every step.
        total = stacked.new_zeros(())
        for i in range(stacked.shape[0]):
            total = total + wc.weighted_squared_error(stacked[i].squeeze(1), y_tr_t, sigma_tr_t)
        total.backward()
        opt.step()

        model.model.eval()
        with torch.no_grad():
            stacked_val = model.model(x_val_t)
            val_loss = sum(float(wc.weighted_squared_error(stacked_val[i].squeeze(1), y_val_t, sigma_val_t).item()) for i in range(stacked_val.shape[0]))
        val_loss_history.append(val_loss)
        if val_loss < best_val - min_delta:
            best_val, patience_counter = val_loss, 0
            best_state = {name: t.detach().clone() for name, t in model.model.state_dict().items()}
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    if best_state is not None:
        model.model.load_state_dict(best_state)
    model.model.eval()
    return model, val_loss_history


def _train_net(
    widths: tuple[int, ...],
    tier: Tier,
    cfg: _TierConfig,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    region_tr: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    region_val: np.ndarray,
    seed: int,
    *,
    max_epochs: int,
    patience: int,
    min_delta: float,
    lr: float,
) -> tuple[FlexibleWidthNN, dict[str, Any]]:
    """Trains one `FlexibleWidthNN(widths=widths)` for `tier`, gated by a convergence-tracker replay.

    Dispatches to `_train_tier1`/`_train_tier2_weighted`, then replays the resulting per-epoch val-loss
    trajectory through `width_wsel4._replay` (same convergence-tracker verdict shape every ported-arm
    cell in this programme uses) for the `trustworthy`/`hit_cap` flags this task's contract requires.
    """
    if tier is Tier.ONE:
        model, val_loss_history = _train_tier1(widths, seed, x_tr, y_tr, x_val, y_val, max_epochs=max_epochs, patience=patience, lr=lr)
        objective = "mse"
    else:
        sigma_tr = _sigma_true(cfg.toy, region_tr, cfg.sigma)
        sigma_val = _sigma_true(cfg.toy, region_val, cfg.sigma)
        model, val_loss_history = _train_tier2_weighted(
            widths, seed, x_tr, y_tr, x_val, y_val, sigma_tr, sigma_val, max_epochs=max_epochs, patience=patience, min_delta=min_delta, lr=lr
        )
        objective = "weighted_squared_error"

    replay = w4._replay(val_loss_history, patience, min_delta)
    hit_cap = bool(len(val_loss_history) >= max_epochs)
    trustworthy = bool(replay.trustworthy and not hit_cap)
    return model, {
        "trajectory": replay.summary()["trajectory"],
        "actual_epochs": len(val_loss_history),
        "n_train_used": len(x_tr),
        "trustworthy": trustworthy,
        "hit_cap": hit_cap,
        "objective": objective,
    }


# ---------------------------------------------------------------------------
# Get-or-train caching (this task's stated caching choice, see module docstring): training is
# independent of `fraction`, so the underlying net(s) are cached to disk per (tier, seed[, width])
# and reused across every fraction/arm cell that shares them.
# ---------------------------------------------------------------------------


def _cache_dir(results_dir: str) -> str:
    path = os.path.join(results_dir, _CACHE_DIRNAME)
    os.makedirs(path, exist_ok=True)
    return path


def _shared_cache_paths(results_dir: str, tier: Tier, seed: int) -> tuple[str, str]:
    base = _cache_dir(results_dir)
    return os.path.join(base, f"shared_tier{tier.value}_seed{seed}.pt"), os.path.join(base, f"shared_tier{tier.value}_seed{seed}_meta.json")


def _sweep_cache_paths(results_dir: str, tier: Tier, seed: int, width: int) -> tuple[str, str]:
    base = _cache_dir(results_dir)
    return os.path.join(base, f"sweep_tier{tier.value}_seed{seed}_w{width}.pt"), os.path.join(base, f"sweep_tier{tier.value}_seed{seed}_w{width}_meta.json")


def _load_cached_model(widths: tuple[int, ...], seed: int, state_path: str, *, max_epochs: int, patience: int, lr: float) -> FlexibleWidthNN:
    """Rebuilds a `FlexibleWidthNN`'s module and loads a previously-cached `state_dict` back into it."""
    model = _new_model(widths, seed, max_epochs=max_epochs, patience=patience, lr=lr)
    model.input_size = 1
    model.build_model()
    model.model.load_state_dict(torch.load(state_path, map_location=model.device))
    model.model.eval()
    return model


def _get_or_train(
    widths: tuple[int, ...],
    tier: Tier,
    seed: int,
    split: dict[str, Any],
    state_path: str,
    meta_path: str,
    *,
    max_epochs: int,
    patience: int,
    min_delta: float,
    lr: float,
) -> tuple[FlexibleWidthNN, dict[str, Any], bool]:
    """Get-or-train ONE `FlexibleWidthNN(widths=widths)`, cached at `state_path`/`meta_path`.

    Shared by both the multi-head net (`widths=1..w_max`, W-SHARED/W-PERINPUT) and each W-SWEEP
    dedicated per-width net (`widths=(k,)`) -- same training path, only `widths`'s length differs.

    Returns:
        `(model, meta, cache_hit)`. `meta` carries the training trajectory/epoch count/trustworthy
        flags plus `training_macs` (computed once, at training time, and cached alongside).
    """
    if os.path.exists(state_path) and os.path.exists(meta_path):
        model = _load_cached_model(widths, seed, state_path, max_epochs=max_epochs, patience=patience, lr=lr)
        with open(meta_path) as f:
            meta = json.load(f)
        return model, meta, True

    cfg = _TIER_CONFIG[tier]
    model, meta = _train_net(
        widths,
        tier,
        cfg,
        split["x_train"],
        split["y_train"],
        split["region_train"],
        split["x_p1_val"],
        split["y_p1_val"],
        split["region_p1_val"],
        seed,
        max_epochs=max_epochs,
        patience=patience,
        min_delta=min_delta,
        lr=lr,
    )
    # Exact, not approximate: `FlexibleWidthNNModule.forward` recomputes the trunk once per configured
    # width with no sharing in its own computation graph (model.py:158-165) -- exactly what `sweep_cost`
    # prices (see module docstring).
    meta["training_macs"] = sweep_cost(model.model, list(widths), meta["n_train_used"], meta["actual_epochs"])
    torch.save(model.model.state_dict(), state_path)
    with open(meta_path, "w") as f:
        json.dump(_jsonable(meta), f, indent=2)
    return model, meta, False


def _get_or_train_shared(tier: Tier, seed: int, split: dict[str, Any], w_max: int, *, results_dir: str, **train_kwargs: Any) -> tuple[FlexibleWidthNN, dict[str, Any], bool]:
    """The ONE multi-head net (`widths=1..w_max`) W-SHARED and W-PERINPUT are both read off."""
    state_path, meta_path = _shared_cache_paths(results_dir, tier, seed)
    return _get_or_train(tuple(range(1, w_max + 1)), tier, seed, split, state_path, meta_path, **train_kwargs)


def _get_or_train_sweep_all(
    tier: Tier, seed: int, split: dict[str, Any], w_max: int, *, results_dir: str, **train_kwargs: Any
) -> tuple[dict[int, FlexibleWidthNN], dict[int, dict[str, Any]], bool]:
    """The 12 dedicated single-width nets W-SWEEP reads its held-out curve off, `{width -> model/meta}`."""
    models: dict[int, FlexibleWidthNN] = {}
    metas: dict[int, dict[str, Any]] = {}
    all_cache_hit = True
    for width in range(1, w_max + 1):
        state_path, meta_path = _sweep_cache_paths(results_dir, tier, seed, width)
        model, meta, cache_hit = _get_or_train((width,), tier, seed, split, state_path, meta_path, **train_kwargs)
        models[width], metas[width] = model, meta
        all_cache_hit = all_cache_hit and cache_hit
    return models, metas, all_cache_hit


# ---------------------------------------------------------------------------
# Arm selection + scoring -- `width.md` SS1's "each model is the complete system, including its
# selection machinery": every arm's held-out quality is read off `predict()`, the public API a
# downstream consumer would actually call, never a side-channel score.
# ---------------------------------------------------------------------------


def _run_w_shared(model: FlexibleWidthNN, meta: dict[str, Any], split: dict[str, Any], fraction_pct: int, seed: int) -> dict[str, Any]:
    """W-SHARED: `fit_global_selector` on the fraction-limited selection subsample, scored held out."""
    x_sel, y_sel, _region_sel = _selection_subsample(split, fraction_pct)
    x_sel_in = x_sel.reshape(-1, 1).astype(np.float32)
    model.fit_global_selector(x_sel_in, y_sel, seed=seed)
    model.capacity_selection = CapacitySelection.GLOBAL_CHEAP

    x_test_in = split["x_test"].reshape(-1, 1).astype(np.float32)
    pred = model.predict(x_test_in, filter_data=False)
    mse = float(np.mean((pred - split["y_test"]) ** 2))

    cost = global_cheap_cost(training_macs=meta["training_macs"], net=model.model, capacity_grid=list(model.widths), n_samples=len(x_sel))
    return {
        "selected_width": int(model.selected_width_),
        "selection_curve": model.global_selector_curve_,
        "held_out_mse": mse,
        "n_selection_used": len(x_sel),
        "cost_macs": {"training_macs": cost.training_macs, "selection_macs": cost.selection_macs, "total_macs": cost.total_macs},
    }


def _run_w_perinput(
    model: FlexibleWidthNN, meta: dict[str, Any], split: dict[str, Any], fraction_pct: int, seed: int, router_override: dict[str, Any] | None = None
) -> dict[str, Any]:
    """W-PERINPUT: `fit_router` on the fraction-limited selection subsample, scored held out.

    Router hyperparameters (hidden/epochs/lr) are read back off the FITTED router
    (`model.capacity_router_.hidden`/`.n_epochs`) for the cost accounting, never re-chosen here --
    WSEL-6 does not vary router architecture (WSEL-7's job; SS3.6: the router stays frozen at its
    current defaults until WSEL-7 rules otherwise).

    `router_override` is the ONE sanctioned exception -- the SS3.5-mandated re-run at WSEL-7's
    `new_default` after its non-invariance verdict (`width.md` WSEL-7 result block, 2026-07-22).
    When set (`{"hidden": tuple, "epochs": int, "lr": float, "source": path}`), the router is
    constructed DIRECTLY with those args and fitted through its unmodified public `.fit()` -- the
    same reuse pattern `width_wsel7.run_cell` and `width_wsel16._distilled_router_selected_width`
    already use, with eval/cost wiring mirroring `FlexibleWidthNN.fit_router`'s own (per-sample
    squared error at each fixed width via the public `predict(..., width=k)`, `executed_flops` cost).
    Everything downstream (labelling tolerance, selection, scoring, cost accounting) is IDENTICAL to
    the default path -- the router architecture is the single difference. Never combined with the
    primary results dir: re-run cells land in their own `--results-dir`.
    """
    del seed  # DistilledCapacityRouter.fit's own seeding is unconditional on this class's router-weight init (self.seed), unused here.
    x_sel, y_sel, _region_sel = _selection_subsample(split, fraction_pct)
    x_sel_in = x_sel.reshape(-1, 1).astype(np.float32)
    if router_override is None:
        model.fit_router(x_sel_in, y_sel)
    else:
        capacity_grid = [(width,) for width in model.widths]

        def eval_fn(x: np.ndarray, capacity: tuple[int, ...]) -> np.ndarray:
            return (model.predict(x, filter_data=False, width=capacity[0]) - y_sel) ** 2

        def cost_fn(capacity: tuple[int, ...]) -> float:
            return float(executed_flops(model.model, capacity[0]))

        router = DistilledCapacityRouter(hidden=tuple(router_override["hidden"]), n_epochs=int(router_override["epochs"]), lr=float(router_override["lr"]), device=model.device)
        router.fit(eval_fn=eval_fn, x_val=x_sel_in, y_val=y_sel, capacity_grid=capacity_grid, tolerance=DEFAULT_TOLERANCE, cost_fn=cost_fn)
        model.capacity_router_ = router
    model.capacity_selection = CapacitySelection.PER_INPUT

    x_test_in = split["x_test"].reshape(-1, 1).astype(np.float32)
    pred = model.predict(x_test_in, filter_data=False)
    mse = float(np.mean((pred - split["y_test"]) ** 2))

    routed_widths = [capacity[0] for capacity in model.capacity_router_.route(x_test_in)]
    width_distribution = {str(width): routed_widths.count(width) for width in sorted(set(routed_widths))}

    router = model.capacity_router_
    cost = per_input_cost(training_macs=meta["training_macs"], in_dim=1, n_capacities=len(model.widths), n_samples=len(x_sel), n_epochs=router.n_epochs, hidden=router.hidden)
    result = {
        "width_distribution": width_distribution,
        "mean_routed_width": float(np.mean(routed_widths)),
        "held_out_mse": mse,
        "n_selection_used": len(x_sel),
        "cost_macs": {"training_macs": cost.training_macs, "selection_macs": cost.selection_macs, "total_macs": cost.total_macs},
    }
    if router_override is not None:
        result["router_config"] = {
            "hidden": list(router_override["hidden"]),
            "depth": len(router_override["hidden"]),
            "epochs": int(router_override["epochs"]),
            "lr": float(router_override["lr"]),
            "source": router_override["source"],
        }
    return result


def _run_w_sweep(models: dict[int, FlexibleWidthNN], metas: dict[int, dict[str, Any]], split: dict[str, Any], fraction_pct: int, seed: int, w_max: int) -> dict[str, Any]:
    """W-SWEEP: builds a held-out error TABLE across the 12 dedicated nets, applies the SAME selector.

    No single net spans every width here, so `FlexibleWidthNN.fit_global_selector` (built for ONE
    multi-width net) does not apply directly; the error table is built by hand, one `predict(...,
    width=k)` call per dedicated net -- the SAME public API `_run_w_shared` uses -- then fed to the
    SAME `cheapest_within_tolerance` selector `fit_global_selector` calls internally (`width.md` SS1:
    "the same rule applies to W-SWEEP's curve").
    """
    x_sel, y_sel, _region_sel = _selection_subsample(split, fraction_pct)
    x_sel_in = x_sel.reshape(-1, 1).astype(np.float32)
    x_test_in = split["x_test"].reshape(-1, 1).astype(np.float32)

    error_table = np.stack([(models[width].predict(x_sel_in, filter_data=False, width=width) - y_sel) ** 2 for width in range(1, w_max + 1)], axis=1)
    idx = cheapest_within_tolerance(error_table, n_boot=DEFAULT_N_BOOT, seed=seed)
    selected_width = idx + 1  # column i (0-indexed, cheapest-first) <-> width i+1.

    pred = models[selected_width].predict(x_test_in, filter_data=False, width=selected_width)
    mse = float(np.mean((pred - split["y_test"]) ** 2))

    training_macs = sum(sweep_cost(models[width].model, [width], metas[width]["n_train_used"], metas[width]["actual_epochs"]) for width in range(1, w_max + 1))
    selection_macs = sum(held_out_read_cost(models[width].model, [width], len(x_sel)) for width in range(1, w_max + 1))
    return {
        "selected_width": int(selected_width),
        "held_out_mse": mse,
        "n_selection_used": len(x_sel),
        "cost_macs": {"training_macs": training_macs, "selection_macs": selection_macs, "total_macs": training_macs + selection_macs},
        "hit_cap": any(metas[width]["hit_cap"] for width in range(1, w_max + 1)),
        "trustworthy": all(metas[width]["trustworthy"] for width in range(1, w_max + 1)),
    }


# ---------------------------------------------------------------------------
# One cell.
# ---------------------------------------------------------------------------


def run_cell(
    tier: Tier,
    seed: int,
    fraction_pct: int,
    arm: Arm,
    *,
    w_max: int = W_MAX,
    max_epochs: int = w4.PORTED_N_EPOCHS_CAP,
    patience: int = w4.PORTED_PATIENCE,
    min_delta: float = w4.PORTED_MIN_DELTA,
    lr: float = w4.PORTED_LR_DEFAULT,
    results_dir: str = RESULTS_DIR,
    n_train: int | None = None,
    n_test: int | None = None,
    router_override: dict[str, Any] | None = None,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    """Runs one (tier, seed, fraction, arm) cell: get-or-train, select on the fraction, score held out.

    `cache_dir`, when set, points the get-or-train step at ANOTHER results dir's `_cache` (the WSEL-7
    re-run reuses the primary grid's trained nets -- training is independent of both fraction and
    router config, so retraining them into the re-run dir would be pure waste). Cell JSONs still land
    in `results_dir`.
    """
    cfg = _TIER_CONFIG[tier]
    split = _build_split(tier, seed, n_train=n_train, n_test=n_test)
    train_kwargs = {"max_epochs": max_epochs, "patience": patience, "min_delta": min_delta, "lr": lr, "results_dir": cache_dir or results_dir}

    if arm in (Arm.W_SHARED, Arm.W_PERINPUT):
        model, meta, cache_hit = _get_or_train_shared(tier, seed, split, w_max, **train_kwargs)
        if arm is Arm.W_SHARED:
            arm_result = _run_w_shared(model, meta, split, fraction_pct, seed)
        else:
            arm_result = _run_w_perinput(model, meta, split, fraction_pct, seed, router_override=router_override)
        hit_cap, trustworthy, objective = meta["hit_cap"], meta["trustworthy"], meta["objective"]
        training_block = {"trajectory": meta["trajectory"], "actual_epochs": meta["actual_epochs"]}
    else:
        models, metas, cache_hit = _get_or_train_sweep_all(tier, seed, split, w_max, **train_kwargs)
        arm_result = _run_w_sweep(models, metas, split, fraction_pct, seed, w_max)
        hit_cap, trustworthy = arm_result.pop("hit_cap"), arm_result.pop("trustworthy")
        objective = metas[1]["objective"]
        training_block = {
            "trajectory": {str(width): metas[width]["trajectory"] for width in metas},
            "actual_epochs": {str(width): metas[width]["actual_epochs"] for width in metas},
        }

    case = {
        "tier": tier.value,
        "toy": cfg.toy.value,
        "seed": seed,
        "arm": arm.value,
        "fraction_pct": fraction_pct,
        "fraction": fraction_pct / 100.0,
        "w_max": w_max,
        "n_train": split["n_train"],
        "n_test": split["n_test"],
        "hit_cap": bool(hit_cap),
        "trustworthy": bool(trustworthy),
        "cache_hit": bool(cache_hit),
        "training_schedule": nwn.WidthSchedule.ALL.value,  # FlexibleWidthNN always sums every configured width every step (module docstring).
        "objective": objective,
        "training": training_block,
        "provenance": run_provenance(),
    }
    case.update(arm_result)
    return case


def _cell_json_path(results_dir: str, tier: Tier, seed: int, fraction_pct: int, arm: Arm) -> str:
    return os.path.join(results_dir, f"wsel6_tier{tier.value}_seed{seed}_frac{fraction_pct}_{arm.value}.json")


def _jsonable(obj: object) -> object:
    """Minimal JSON-safe coercion (numpy/torch scalars, dict int-keys) -- local twin of every sibling WSEL driver's helper."""
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
# --summarize -- frozen.json (the two SS3.6 constants this task owns) + the saturation plot.
# ---------------------------------------------------------------------------


def _plot_saturation(cells: list[dict[str, Any]], results_dir: str) -> str:
    """One panel per tier, mean held-out MSE +/- SE vs fraction, one line per arm."""
    tiers = list(Tier)
    fig, axes = plt.subplots(1, len(tiers), figsize=(6 * len(tiers), 5), squeeze=False)
    for tier in tiers:
        ax = axes[0][tier.value - 1]
        for arm in Arm:
            fracs, means, ses = [], [], []
            for frac in sorted(FRACTIONS_PCT):
                vals = [c["held_out_mse"] for c in cells if c["tier"] == tier.value and c["arm"] == arm.value and c["fraction_pct"] == frac]
                if not vals:
                    continue
                fracs.append(frac)
                means.append(float(np.mean(vals)))
                ses.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)
            if fracs:
                ax.errorbar(fracs, means, yerr=ses, marker="o", label=arm.value)
        ax.set_xlabel("selection fraction (% of training portion)")
        ax.set_ylabel("held-out MSE")
        ax.set_title(f"tier {tier.value} ({_TIER_CONFIG[tier].toy.value})")
        ax.legend()
    fig.tight_layout()
    path = os.path.join(results_dir, "saturation.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def summarize(results_dir: str = RESULTS_DIR) -> None:
    """Aggregates every per-cell JSON on disk into `WSEL6/frozen.json` plus `WSEL6/saturation.png`."""
    cells: list[dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(results_dir, "wsel6_tier*_seed*_frac*_*.json"))):
        with open(path) as f:
            cells.append(json.load(f))

    by_key: dict[tuple[int, str, int], list[float]] = defaultdict(list)
    for cell in cells:
        by_key[(cell["tier"], cell["arm"], cell["fraction_pct"])].append(cell["held_out_mse"])

    fracs_sorted = sorted(FRACTIONS_PCT)
    fraction_choice: dict[tuple[int, str], dict[str, Any]] = {}
    for tier in Tier:
        for arm in Arm:
            rows = []
            complete = True
            for frac in fracs_sorted:
                vals = by_key.get((tier.value, arm.value, frac))
                if not vals or len(vals) < len(SEEDS):
                    complete = False
                    break
                rows.append(vals)
            if not complete:
                continue
            # cheapest_within_tolerance turned on itself: "capacities" = fractions (cheapest = smallest
            # first), "samples" = seeds. Returns the smallest fraction within 2 bootstrap SE of the best.
            error_table = np.array(rows).T  # (n_seeds, n_fractions)
            idx = cheapest_within_tolerance(error_table, n_boot=DEFAULT_N_BOOT, seed=0)
            chosen = fracs_sorted[idx]
            fraction_choice[(tier.value, arm.value)] = {"fraction_pct": chosen, "saturated": chosen < fracs_sorted[-1]}

    n_expected_pairs = len(Tier) * len(Arm)
    if fraction_choice:
        overall_pct = max(v["fraction_pct"] for v in fraction_choice.values())
        overall_saturated = len(fraction_choice) == n_expected_pairs and all(v["saturated"] for v in fraction_choice.values())
    else:
        overall_pct = max(FRACTIONS_PCT)
        overall_saturated = False

    data_limited: dict[str, dict[str, bool]] = {}
    for (tier_value, arm_value), choice in fraction_choice.items():
        toy = _TIER_CONFIG[Tier(tier_value)].toy.value
        data_limited.setdefault(toy, {})[arm_value] = not choice["saturated"]

    frozen = {
        "fraction": overall_pct / 100.0,
        "fraction_pct": overall_pct,
        "saturated": bool(overall_saturated),
        "data_limited": data_limited,
        "per_tier_arm_fraction_choice": {f"{Tier(tier_value).value}:{arm_value}": choice for (tier_value, arm_value), choice in fraction_choice.items()},
        "n_tier_arm_pairs_complete": len(fraction_choice),
        "n_tier_arm_pairs_expected": n_expected_pairs,
        "fractions_swept_pct": list(FRACTIONS_PCT),
        "seeds": list(SEEDS),
        "n_cells_present": len(cells),
        "provenance": run_provenance(),
    }
    os.makedirs(results_dir, exist_ok=True)
    frozen_path = os.path.join(results_dir, "frozen.json")
    with open(frozen_path, "w") as f:
        json.dump(_jsonable(frozen), f, indent=2)
    print(f"wrote {frozen_path}")

    plot_path = _plot_saturation(cells, results_dir)
    print(f"wrote {plot_path}")
    print(
        f"fraction={frozen['fraction']} ({frozen['fraction_pct']}%) saturated={overall_saturated} "
        f"({frozen['n_tier_arm_pairs_complete']}/{frozen['n_tier_arm_pairs_expected']} (tier, arm) pairs complete)"
    )


# ---------------------------------------------------------------------------
# Selftest -- every (tier, arm) at 2 tiny fractions on a w_max=3 toy, plus an explicit cache-reuse
# check, all inside a throwaway tmp dir. No real cell is ever run here.
# ---------------------------------------------------------------------------


def run_selftest() -> bool:
    """Tiny end-to-end pass: every (tier, arm) combo at 2 fractions, `--summarize`, and cache reuse."""
    tmp_dir = tempfile.mkdtemp(prefix="width_wsel6_selftest_")
    try:
        w_max = 3
        n_train, n_test = 60, 30
        fractions = (20, 40)
        max_epochs, patience, min_delta, lr = 40, 3, 1e-2, 0.05

        ok = True
        for tier in Tier:
            for arm in Arm:
                for frac in fractions:
                    case = run_cell(
                        tier,
                        seed=0,
                        fraction_pct=frac,
                        arm=arm,
                        w_max=w_max,
                        max_epochs=max_epochs,
                        patience=patience,
                        min_delta=min_delta,
                        lr=lr,
                        results_dir=tmp_dir,
                        n_train=n_train,
                        n_test=n_test,
                    )
                    with open(_cell_json_path(tmp_dir, tier, 0, frac, arm), "w") as f:
                        json.dump(_jsonable(case), f, indent=2)

                    cell_ok = (
                        math.isfinite(case["held_out_mse"])
                        and isinstance(case["hit_cap"], bool)
                        and isinstance(case["trustworthy"], bool)
                        and case["training_schedule"] == nwn.WidthSchedule.ALL.value
                        and (case["objective"] == "mse") == (tier is Tier.ONE)
                        and (case["objective"] == "weighted_squared_error") == (tier is Tier.TWO)
                        and case["cost_macs"]["total_macs"] > 0
                    )
                    cell_ok = cell_ok and (("width_distribution" in case) if arm is Arm.W_PERINPUT else ("selected_width" in case))
                    ok = ok and cell_ok
                    print(
                        f"[wsel6 selftest] tier={tier.value} arm={arm.value} frac={frac} mse={case['held_out_mse']:.4g} "
                        f"objective={case['objective']} cache_hit={case['cache_hit']}  {'PASS' if cell_ok else 'FAIL'}"
                    )

        summarize(results_dir=tmp_dir)
        with open(os.path.join(tmp_dir, "frozen.json")) as f:
            frozen = json.load(f)
        summarize_ok = "fraction" in frozen and "data_limited" in frozen and os.path.exists(os.path.join(tmp_dir, "saturation.png"))
        print(f"[wsel6 selftest] summarize: fraction={frozen.get('fraction')} data_limited={frozen.get('data_limited')}  {'PASS' if summarize_ok else 'FAIL'}")

        # Cache-reuse check: a W_SHARED cell repeated (same tier/seed, any fraction) after the loop
        # above already trained that (tier, seed)'s shared net must hit the cache, not retrain.
        case_again = run_cell(
            Tier.ONE,
            seed=0,
            fraction_pct=fractions[0],
            arm=Arm.W_SHARED,
            w_max=w_max,
            max_epochs=max_epochs,
            patience=patience,
            min_delta=min_delta,
            lr=lr,
            results_dir=tmp_dir,
            n_train=n_train,
            n_test=n_test,
        )
        cache_ok = case_again["cache_hit"] is True
        print(f"[wsel6 selftest] cache reuse on repeat cell: cache_hit={case_again['cache_hit']}  {'PASS' if cache_ok else 'FAIL'}")

        ok = ok and summarize_ok and cache_ok
        print(f"[wsel6 selftest] {'PASS' if ok else 'FAIL'}")
        return ok
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Runs one per-cell training+selection (`--tier/--seed/--fraction/--arm`), `--summarize`, or `--selftest`."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true", help="Tiny end-to-end wiring check (every tier/arm + cache reuse), then exit.")
    parser.add_argument("--summarize", action="store_true", help="Aggregate landed cells into WSEL6/frozen.json + saturation.png, then exit.")

    parser.add_argument("--tier", type=int, choices=[t.value for t in Tier], default=None, help="Required outside --selftest/--summarize.")
    parser.add_argument("--seed", type=int, default=None, help="Required outside --selftest/--summarize (canonical suite: 0, 1, 2).")
    parser.add_argument(
        "--fraction", type=int, choices=list(FRACTIONS_PCT), default=None, help="Selection-set fraction, %% of the training portion. Required outside --selftest/--summarize."
    )
    parser.add_argument("--arm", choices=[a.value for a in Arm], default=None, help="Required outside --selftest/--summarize.")
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR)
    parser.add_argument(
        "--router-from-wsel7",
        type=str,
        default=None,
        metavar="FROZEN_JSON",
        help="Path to WSEL-7's frozen.json. Fits the W-PERINPUT router at its `new_default` config instead of the frozen "
        "defaults -- the SS3.5-mandated re-run after WSEL-7's non-invariance verdict. Requires --arm w_perinput and a "
        "NON-DEFAULT --results-dir (re-run cells must never pool with the primary grid's).",
    )
    parser.add_argument(
        "--cache-from",
        type=str,
        default=None,
        metavar="RESULTS_DIR",
        help="Reuse another results dir's `_cache` for the get-or-train step (the re-run reuses the primary grid's "
        "trained nets; training is independent of fraction and router config). Cell JSONs still land in --results-dir.",
    )
    parser.add_argument(
        "--epoch-cap",
        type=int,
        default=w4.PORTED_N_EPOCHS_CAP,
        help="Safety cap on training epochs for any net this cell has to train (default: the w4 ported protocol's cap). "
        "Raise it ONLY to repair a cell whose cached net hit the cap (the DO-NOT-CONCLUDE guard); the capped cache entry "
        "must be moved aside first or the cell will just reload it. A raised cap is a named deviation: the rebuilt cache "
        "meta records the actual epochs, and convergence is still decided by the patience gate, not the cap.",
    )

    args = parser.parse_args()

    if args.selftest and args.summarize:
        parser.error("--selftest and --summarize are mutually exclusive.")
    if args.selftest:
        sys.exit(0 if run_selftest() else 1)
    if args.summarize:
        summarize(results_dir=args.results_dir)
        return

    if args.tier is None or args.seed is None or args.fraction is None or args.arm is None:
        parser.error("--tier, --seed, --fraction and --arm are all required for a real cell (or pass --selftest / --summarize).")

    tier = Tier(args.tier)
    arm = Arm(args.arm)

    router_override = None
    if args.router_from_wsel7 is not None:
        if arm is not Arm.W_PERINPUT:
            parser.error("--router-from-wsel7 only applies to --arm w_perinput (the other arms construct no router).")
        if os.path.abspath(args.results_dir) == os.path.abspath(RESULTS_DIR):
            parser.error("--router-from-wsel7 requires a NON-DEFAULT --results-dir: re-run cells must never pool with the primary grid's.")
        with open(args.router_from_wsel7) as f:
            wsel7_frozen = json.load(f)
        new_default = wsel7_frozen["new_default"]
        if new_default is None:
            parser.error(f"{args.router_from_wsel7} carries new_default=null (invariant verdict) -- there is nothing to re-run at.")
        router_override = {
            "hidden": (int(new_default["hidden"]),) * int(new_default["depth"]),
            "epochs": int(new_default["epochs"]),
            "lr": float(new_default["lr"]),
            "source": os.path.abspath(args.router_from_wsel7),
        }

    os.makedirs(args.results_dir, exist_ok=True)
    print(f"[wsel6] tier={tier.value} toy={_TIER_CONFIG[tier].toy.value} seed={args.seed} fraction={args.fraction}% arm={arm.value} w_max={W_MAX}", flush=True)
    case = run_cell(tier, args.seed, args.fraction, arm, results_dir=args.results_dir, max_epochs=args.epoch_cap, router_override=router_override, cache_dir=args.cache_from)

    if not case["trustworthy"]:
        print(
            f"*** DO-NOT-CONCLUDE GUARD: tier={tier.value} seed={args.seed} arm={arm.value} did NOT converge trustworthily "
            f"(hit_cap={case['hit_cap']}). Raise the epoch cap before drawing any conclusion. ***"
        )

    out_path = _cell_json_path(args.results_dir, tier, args.seed, args.fraction, arm)
    with open(out_path, "w") as f:
        json.dump(_jsonable(case), f, indent=2)
    print(f"[wsel6] wrote {out_path} (held_out_mse={case['held_out_mse']:.6f} trustworthy={case['trustworthy']} hit_cap={case['hit_cap']} cache_hit={case['cache_hit']})")


if __name__ == "__main__":
    main()
