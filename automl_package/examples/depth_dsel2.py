"""DSEL-2 — feed-forward depth: does the benefit hold at all? STAGE 1 ONLY (the positive control).

`docs/plans/capacity_programme/depth-selection.md`, task DSEL-2. This module builds and runs
**only** the two-part positive control MASTER Decision 14 requires before any per-depth reading is
trusted:

  (i)  the feed-forward (distinct-weights-per-layer) net at FULL DEPTH reaches held-out accuracy
       >= 0.90 AND `trustworthy=true`, on BOTH seeds (§3.6 `positive-control bar`);
  (ii) only if (i) passes — the parameter-matched wide-shallow net does NOT clear the same bar.

The per-depth ladder and the formal two-arm (SHARED vs PER_DEPTH) readout comparison the plan's
"CARRIED RULING" block calls for are **out of scope here** — the root gates both on (i)/(ii)
passing. Nothing in this module runs them.

## DSEL-2c extension (2026-07-21) — re-running the positive control at specification

Stage 1's control (above) failed on both seeds (`capacity_ladder_results/DSEL2/control_rung*.json`)
and turned out to deviate from DSEL-1b's own ratified design in two unflagged ways: it trained every
exit against the FULL-word label (never the depth-*d* prefix product DSEL-1b's own spec bullet calls
for) and it never exercised DSEL-1b §1b's scheme (a), the per-sample uniform depth draw. USER RULINGS
2026-07-21 (`depth-selection.md`, DSEL-2's halt block) order both deviations repaired and re-run as a
four-arm grid before the stage-1 failure is read as a substrate finding. `TargetScheme`/`ScheduleType`
below are the two axes; `(FULL, ALL_RUNGS)` is stage 1's own arm above — cited, never rerun — and
`run_dsel2c_arm` trains the other three at the best-found config (width 32, train_frac 0.75, DSEL-2's
own `rung4_data75`), three-way split (train/val/test, test is the reported number), against the same
`FIT_ACC` bar. `freeze_dsel2c` lands the verdict.

**Substrate/protocol, fixed from DSEL-1b, not varied (one-variable rule — this task varies depth
capacity only):** group A5, word length 6, `NestedFeedForwardClf`
(`depth_composition_toy.NestedFeedForwardClf`) trained under DSEL-1b's all-rungs scheme (scheme
(b): mean cross-entropy over every rung of the ladder, every step, no sampling —
`depth_composition_toy.train_nested_clf`, reused). The depth ladder is read from
`capacity_ladder_results/DSEL1b/frozen.json` at runtime, never hardcoded (plan §3.6).

**Readout arm, provisional for this stage only:** SHARED (one `nn.Linear` reused at every depth's
exit) — the certified-winning shape from the recurrent depth mechanism
(`docs/depth_capacity/verdict_per_input_depth.md`) and the shape `FlexibleHiddenLayersNN` already
ships. DSEL-1b's own readout_comparison JSONs are silent on which arm should be read as *the*
feed-forward net for Decision-14 purposes (`readable_as_verdict=false` on both seeds — neither
arm converged cleanly parameter-matched); this module does not resolve that question, it picks one
arm to attempt the control with and says so here, in writing, rather than by silent default. If
clause (i) passes on SHARED, PER_DEPTH remains untested by this module and is explicitly named as
open in this module's own report.

## Protocol diff vs DSEL-1b (MASTER Decision 15 — every difference justified in writing here)

DSEL-1b's `run_readout_arm_comparison` (`depth_composition_toy.py:804-878`) trains via
`train_nested_clf` (`:720-782`), which gates ONLY on the ladder-mean held-out cross-entropy —
best weights are restored on CE, and CE is the only stopping/trustworthy signal. Its own landed
artifacts (`capacity_ladder_results/DSEL1b/readout_comparison_a5_n6_seed{0,1}.json`) show, for the
SHARED arm at full depth (6), seed 0: CE trajectory falls to a minimum at epoch 500 (2.130 nats)
then rises monotonically to 2.576 nats by the epoch-3000 stop, while `best_epoch=500` — i.e. the
weights actually returned and scored are from epoch 500, four-fifths of the way through training
having already been thrown away by a patience clock reading the wrong series. Held-out accuracy at
those early-restored weights is 0.282 (train_acc 0.456, both far below the 0.90 bar) — the SAME
selection defect F5c-b's history names explicitly ("best-CE weight restore would have selected a
LOWER accuracy than the run's own best").

**Single difference introduced here:** DSEL-2 adds a second convergence tracker on FULL-DEPTH
HELD-OUT ACCURACY (the metric clause (i)'s bar actually reads — MASTER Decision 17: "the
convergence gate must be computed on the metric the bar reads"), AND-stopped against the existing
CE tracker, with best weights restored on accuracy — `train_nested_clf_dual_gate` below. This is
the exact dual-gate PATTERN `depth_composition_toy._train_dual_gate` (`:535-612`) already
implements for the single-target `train_clf` trainer; it cannot be called on the nested/all-rungs
step function directly (different loss shape — summed over the ladder, not one target), so the
pattern is REUSED (tracker construction, AND-stop, best-on-accuracy, the `acc_gate` dict shape)
and re-implemented against `train_nested_clf`'s step/val functions rather than copied verbatim.
Every other setting (LR, optimizer, batching — full-batch Adam, CHECK_EVERY/PATIENCE/MIN_DELTA
constants) is read from `depth_composition_toy` unchanged. No clipping is introduced at Rung 0 —
DSEL-1b's own run used none; clipping is escalation-ladder rung 2 if Rung 0 (gate fix alone)
under-fits.

## Doctrine / reuse (minimum-viable-code ladder)

Searched `depth_composition_toy.py` (the substrate + all four architecture classes),
`depth_selection_toy.py` (the certified recurrent dual-gate precedent — `train_anytime`, not
reused directly since its target is per-`t`-prefix, not full-word, matching the recurrent
per-step-consumption shape this module's flat-input architecture does not share),
`automl_package/utils/convergence.py` (`ConvergenceTracker`, `ConvergenceResult` — reused
directly, not reimplemented). Reused without modification: `Group`, `ReadoutMode`,
`NestedFeedForwardClf`, `make_word_data`, `build_wide_shallow_clf`, `train_clf` (with
`GateMode.DUAL`, already built for exactly the wide-shallow falsifier's needs — no ladder there,
so DSEL-1b's nested trainer does not apply and `train_clf`'s existing dual gate is the correct
fit), `count_params`. Net-new here: `train_nested_clf_dual_gate` (the accuracy-gated nested
trainer DSEL-1b's version lacks) and this module's escalation-ladder orchestration
(`ControlConfig`, `run_full_depth_control`, `run_wide_shallow_falsifier`, `main`).

## Non-goals (this module)

No recurrent runs of any kind (⛔ parked, §1a). No per-depth ladder (DSEL-2's own dispatch gates
it on (i)/(ii)). No formal two-arm readout comparison / PER_DEPTH training. No new toy. No changes
to `depth_composition_toy.py`, any `automl_package/models/` file, or `tests/`.

Usage:
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/depth_dsel2.py --selftest
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/depth_dsel2.py \
        --control --rung rung0 --seed 0
    AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/depth_dsel2.py \
        --falsifier --seed 0
"""

from __future__ import annotations

import argparse
import enum
import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_EXAMPLES_DIR)))  # repo root, so `import automl_package` works

import convergence as cvg  # noqa: E402 — full-trajectory convergence gate, shared with the width/depth toys
import depth_composition_toy as dct  # noqa: E402 — the certified group-word substrate + net families

DSEL1B_FROZEN_PATH = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "DSEL1b", "frozen.json")
OUT_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "DSEL2")

# Substrate, fixed from DSEL-1b's readout comparison (not varied here — see module docstring).
GROUP = dct.Group.A5
SEQ_LEN = 6
READOUT = dct.ReadoutMode.SHARED

POSITIVE_CONTROL_BAR_ACC = dct.FIT_ACC  # 0.90, imported unchanged (plan §3.6, "positive-control bar" row)


def load_depth_ladder(path: str = DSEL1B_FROZEN_PATH) -> tuple[int, ...]:
    """Read the frozen depth ladder DSEL-1b owns — never a locally hardcoded copy (plan §3.6)."""
    with open(path) as f:
        frozen = json.load(f)
    return tuple(int(d) for d in frozen["depth_ladder"])


# ---------------------------------------------------------------------------
# Dual-gated nested (all-rungs) trainer — the one piece of net-new machinery this module adds.
# Pattern reused from `depth_composition_toy._train_dual_gate`; re-implemented against
# `train_nested_clf`'s ladder-summed step/val functions (see module docstring's protocol diff).
# ---------------------------------------------------------------------------


def train_nested_clf_dual_gate(
    net: dct.NestedFeedForwardClf,
    data: dict,
    ladder: tuple[int, ...],
    device: str = "cpu",
    max_epochs: int = dct.MAX_EPOCHS,
    *,
    lr: float = dct.LR,
    clip_max_norm: float | None = None,
    check_every: int = dct.CHECK_EVERY,
    ce_patience: int = dct.PATIENCE,
    ce_min_delta: float = dct.MIN_DELTA,
    acc_patience: int = dct.ACC_PATIENCE,
    acc_min_delta: float = dct.ACC_MIN_DELTA,
    acc_divergence_abs_eps: float = dct.ACC_DIVERGENCE_ABS_EPS,
    acc_still_improving_eps: float = dct.ACC_STILL_IMPROVING_EPS,
) -> dict:
    """Nested (scheme-b, all-rungs) training with a SECOND gate on FULL-DEPTH held-out accuracy.

    The training step is unchanged from `depth_composition_toy.train_nested_clf`: mean CE over
    every depth in `ladder`, every step, no sampling. What is added is a second
    `cvg.ConvergenceTracker` reading FULL-DEPTH (`max(ladder)`) held-out accuracy specifically —
    the metric clause (i)'s bar reads (Decision 17) — AND-stopped against the CE tracker (so
    neither series is declared done while the other is still moving, mirroring
    `_train_dual_gate`'s "an OR would leave the accuracy clock firing on an arm still improving on
    CE" rationale, inverted here since accuracy is now the metric of record). Best weights are
    restored on ACCURACY, not CE — the fix to the exact defect diagnosed in this module's
    docstring.

    Returns a dict (not a `ConvergenceResult`, since two independent trackers are involved):
    `ce_gate`/`acc_gate` (each a full-trajectory summary dict), `trustworthy_ce`/`trustworthy_acc`,
    per-depth train/val accuracy (post-restore-on-accuracy weights), and the full accuracy/CE
    trajectories (Decision 9 — never an endpoint).
    """
    net.to(device)
    x_tr = torch.as_tensor(data["x_tr"], dtype=torch.float32, device=device)
    y_tr = torch.as_tensor(data["y_tr"], dtype=torch.long, device=device)
    x_val = torch.as_tensor(data["x_val"], dtype=torch.float32, device=device)
    y_val = torch.as_tensor(data["y_val"], dtype=torch.long, device=device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    full_depth = max(ladder)

    ce_tracker = cvg.ConvergenceTracker(patience=ce_patience, min_delta=ce_min_delta)
    acc_tracker = cvg.ConvergenceTracker(patience=acc_patience, min_delta=acc_min_delta)  # tracks -val_acc (lower-is-better contract)
    best_state: dict | None = None
    final_epoch = max_epochs

    val_ce_traj: list[tuple[int, float]] = []
    val_acc_full_traj: list[tuple[int, float]] = []
    train_acc_full_traj: list[tuple[int, float]] = []

    net.train()
    for epoch in range(1, max_epochs + 1):
        opt.zero_grad()
        outs = net.forward_all_depths(x_tr)
        loss = sum(ce(outs[d - 1], y_tr) for d in ladder) / len(ladder)  # scheme (b): every rung, every step
        loss.backward()
        if clip_max_norm is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=clip_max_norm)
        opt.step()

        if epoch % check_every == 0:
            net.eval()
            with torch.no_grad():
                outs_val = net.forward_all_depths(x_val)
                v_ce = sum(ce(outs_val[d - 1], y_val).item() for d in ladder) / len(ladder)
                v_acc_full = float((outs_val[full_depth - 1].argmax(1) == y_val).float().mean().item())
                outs_tr = net.forward_all_depths(x_tr)
                tr_acc_full = float((outs_tr[full_depth - 1].argmax(1) == y_tr).float().mean().item())
            net.train()

            val_ce_traj.append((epoch, v_ce))
            val_acc_full_traj.append((epoch, v_acc_full))
            train_acc_full_traj.append((epoch, tr_acc_full))

            ce_tracker.update(epoch, v_ce)
            is_new_best_acc = acc_tracker.update(epoch, -v_acc_full)
            if is_new_best_acc:
                best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}

            if ce_tracker.done and acc_tracker.done:  # AND, not OR
                final_epoch = epoch
                break

    if best_state is not None:
        net.load_state_dict(best_state)

    ce_result = ce_tracker.result(final_epoch=final_epoch)
    acc_result_neg = acc_tracker.result(final_epoch=final_epoch)  # fields below on the NEGATED (-acc) series
    best_acc = -acc_result_neg.best_val
    final_val_acc = -acc_result_neg.trajectory[-1][1] if acc_result_neg.trajectory else float("nan")
    diverged_acc = bool((best_acc - final_val_acc) > acc_divergence_abs_eps)
    still_improving_acc = bool(acc_result_neg.recent_improvement > acc_still_improving_eps)
    trustworthy_acc = bool(acc_result_neg.converged and not acc_result_neg.hit_cap and not still_improving_acc and not diverged_acc)

    acc_gate = {
        "converged": acc_result_neg.converged,
        "hit_cap": acc_result_neg.hit_cap,
        "still_improving": still_improving_acc,
        "diverged": diverged_acc,
        "trustworthy": trustworthy_acc,
        "stop_epoch": acc_result_neg.stop_epoch,
        "best_val": best_acc,
        "best_epoch": acc_result_neg.best_epoch,
        "recent_improvement": acc_result_neg.recent_improvement,
        "trajectory": [[int(e), float(-v)] for e, v in acc_result_neg.trajectory],
    }

    net.eval()
    with torch.no_grad():
        outs_tr = net.forward_all_depths(x_tr)
        outs_val = net.forward_all_depths(x_val)
        train_acc_per_depth = {d: float((outs_tr[d - 1].argmax(1) == y_tr).float().mean().item()) for d in ladder}
        val_acc_per_depth = {d: float((outs_val[d - 1].argmax(1) == y_val).float().mean().item()) for d in ladder}

    return {
        "full_depth": full_depth,
        "ce_gate": ce_result.summary(),
        "acc_gate": acc_gate,
        "trustworthy_ce": bool(ce_result.trustworthy),
        "trustworthy_acc": trustworthy_acc,
        "train_acc_per_depth": {str(d): v for d, v in train_acc_per_depth.items()},
        "val_acc_per_depth": {str(d): v for d, v in val_acc_per_depth.items()},
        "train_acc_full": train_acc_per_depth[full_depth],
        "val_acc_full": val_acc_per_depth[full_depth],
        "val_ce_trajectory": [[int(e), float(v)] for e, v in val_ce_traj],
        "val_acc_full_trajectory": [[int(e), float(v)] for e, v in val_acc_full_traj],
        "train_acc_full_trajectory": [[int(e), float(v)] for e, v in train_acc_full_traj],
    }


# ---------------------------------------------------------------------------
# Escalation-ladder configs (MASTER Decision 16: LR sweep -> clipping -> warmup -> init scheme ->
# normalization; brief also licenses capacity/data — trunk width, n_train, epoch cap). Each rung is
# a named, logged config; `main()`'s `--rung` selects one. Warmup/init-scheme/normalization rungs
# are not pre-built here (added only if the sweep below needs them — no unused escalation code).
# ---------------------------------------------------------------------------


@dataclass
class ControlConfig:
    """One escalation rung's hyperparameters for the full-depth positive control."""

    label: str
    width: int = dct.NARROW_WIDTH
    train_frac: float = dct.DEFAULT_TRAIN_FRAC
    lr: float = dct.LR
    clip_max_norm: float | None = None
    max_epochs: int = dct.MAX_EPOCHS
    note: str = ""


# Rung 0: DSEL-1b's exact config (width=16, train_frac=0.5 -> n_train=2048/4096 words, lr=1e-2, no
# clip, 40000-epoch cap), with ONLY the gate fixed (dual, not CE-only). Isolates the gate-fix effect
# from every other change, per Decision 15 (one difference at a time).
RUNG0 = ControlConfig(label="rung0_gatefix", note="DSEL-1b's exact config; only the convergence gate changes (dual, not CE-only).")

# Rung 1 (LR): depth_selection_toy.py's certified L=10 recipe uses lr=3e-3 (RUNG0_LR in
# depth_composition_toy.py, cited "1e-2 stalls the deep unroll, 3e-3 reaches 0.99" —
# depth_graded_toy.py:74) — a smaller step for a deep composed stack. Try it before touching clip,
# since RUNG0_LR alone (no clip) is the smaller, single-variable move.
RUNG1_LR = ControlConfig(label="rung1_lr3e-3", lr=dct.RUNG0_LR, note="LR sweep: 1e-2 -> 3e-3 (depth_graded_toy.py:74 precedent).")

# Rung 2 (+ clipping): RUNG1's LR plus RUNG0_CLIP_MAX_NORM=1.0 (depth_selection_toy.py: "L=10 needs
# clipping to stay GD-trainable") — the full certified-recipe pair.
RUNG2_LR_CLIP = ControlConfig(label="rung2_lr_clip", lr=dct.RUNG0_LR, clip_max_norm=dct.RUNG0_CLIP_MAX_NORM, note="+ grad clipping (depth_selection_toy.py RUNG0 pair).")

# Rung 3 (capacity): double the trunk width. Params grow with width^2 per hidden block.
RUNG3_WIDTH = ControlConfig(
    label="rung3_width32", width=2 * dct.NARROW_WIDTH, lr=dct.RUNG0_LR, clip_max_norm=dct.RUNG0_CLIP_MAX_NORM, note="width 16 -> 32 (capacity, not substrate)."
)

# Rung 4 (data): raise train_frac 0.5 -> 0.75 (n_train 2048 -> 3072 of the 4096 enumerated words),
# still leaving >=1024 held-out words to score generalization on.
RUNG4_DATA = ControlConfig(
    label="rung4_data75", width=2 * dct.NARROW_WIDTH, train_frac=0.75, lr=dct.RUNG0_LR, clip_max_norm=dct.RUNG0_CLIP_MAX_NORM,
    note="train_frac 0.5 -> 0.75 (n_train 2048 -> 3072/4096).",
)

# Rung 5 (more data still): rung3->rung4 (train_frac 0.5 -> 0.75, width held at 32) closed most of
# the memorization gap (val 0.377 -> 0.747, seed 0) while train_acc stayed ~1.0 throughout -- push
# the same single lever (train_frac) further before touching width again (one variable at a time).
RUNG5_DATA90 = ControlConfig(
    label="rung5_data90", width=2 * dct.NARROW_WIDTH, train_frac=0.9, lr=dct.RUNG0_LR, clip_max_norm=dct.RUNG0_CLIP_MAX_NORM,
    note="train_frac 0.75 -> 0.9 (n_train 3072 -> 3686/4096).",
)

# Rung 6 (more capacity, at rung 4's data fraction — rung 5 regressed at train_frac=0.9 on seed 0,
# see escalation log; going wider at the BEST-so-far data fraction (0.75) isolates capacity as the
# single next lever rather than compounding it with the unexplained train_frac=0.9 regression).
RUNG6_WIDTH64 = ControlConfig(
    label="rung6_width64", width=4 * dct.NARROW_WIDTH, train_frac=0.75, lr=dct.RUNG0_LR, clip_max_norm=dct.RUNG0_CLIP_MAX_NORM,
    note="width 32 -> 64, train_frac held at rung4's 0.75.",
)

# Rung 7 (interpolate the data anomaly): rung4 (train_frac=0.75, width=32) reached val 0.747; rung5
# (train_frac=0.9, same width) REGRESSED to val 0.410 -- unexplained (n_val shrinks to 410/60
# classes at 0.9, a noisier held-out read, but a 34pp drop is far past sampling noise at that n).
# Interpolating at 0.8 checks whether 0.75 sits near a real local ceiling or the anomaly starts
# closer in -- needed before this ladder's data lever can be read at all.
RUNG7_DATA80 = ControlConfig(
    label="rung7_data80", width=2 * dct.NARROW_WIDTH, train_frac=0.8, lr=dct.RUNG0_LR, clip_max_norm=dct.RUNG0_CLIP_MAX_NORM,
    note="train_frac=0.8, width held at rung4's 32 -- interpolates the rung4->rung5 data anomaly.",
)

RUNGS: dict[str, ControlConfig] = {c.label: c for c in (RUNG0, RUNG1_LR, RUNG2_LR_CLIP, RUNG3_WIDTH, RUNG4_DATA, RUNG5_DATA90, RUNG6_WIDTH64, RUNG7_DATA80)}


def run_full_depth_control(cfg: ControlConfig, seed: int, ladder: tuple[int, ...], device: str = "cpu", out_dir: str = OUT_DIR) -> dict:
    """Train the SHARED-readout `NestedFeedForwardClf` at `cfg`'s hyperparameters; land + return the JSON."""
    data = dct.make_word_data(GROUP, SEQ_LEN, seed, train_frac=cfg.train_frac)
    n_classes = data["n_classes"]
    in_dim = data["seq_len"] * data["n_gen"]
    max_depth = max(ladder)

    torch.manual_seed(1000 * seed + 0)  # DSEL-1b's own seeding convention: `1000*seed + i`, i=0 for SHARED
    net = dct.NestedFeedForwardClf(max_depth, cfg.width, in_dim, n_classes, readout=READOUT)
    params = dct.count_params(net)

    result = train_nested_clf_dual_gate(
        net, data, ladder, device=device, max_epochs=cfg.max_epochs, lr=cfg.lr, clip_max_norm=cfg.clip_max_norm,
    )

    out = {
        "arm": "full_depth_control",
        "rung": cfg.label,
        "rung_note": cfg.note,
        "group": GROUP.value,
        "seq_len": SEQ_LEN,
        "readout": READOUT.value,
        "seed": seed,
        "ladder": list(ladder),
        "full_depth": max_depth,
        "width": cfg.width,
        "params": params,
        "n_classes": n_classes,
        "chance": 1.0 / n_classes,
        "n_train": int(data["x_tr"].shape[0]),
        "n_val": int(data["x_val"].shape[0]),
        "hyperparameters": {
            "lr": cfg.lr, "clip_max_norm": cfg.clip_max_norm, "max_epochs": cfg.max_epochs,
            "train_frac": cfg.train_frac,
            "ce_check_every": dct.CHECK_EVERY, "ce_patience": dct.PATIENCE, "ce_min_delta": dct.MIN_DELTA,
            "acc_check_every": dct.ACC_CHECK_EVERY, "acc_patience": dct.ACC_PATIENCE, "acc_min_delta": dct.ACC_MIN_DELTA,
            "acc_divergence_abs_eps": dct.ACC_DIVERGENCE_ABS_EPS, "acc_still_improving_eps": dct.ACC_STILL_IMPROVING_EPS,
        },
        "positive_control_bar_acc": POSITIVE_CONTROL_BAR_ACC,
        "val_acc": result["val_acc_full"],
        "train_acc": result["train_acc_full"],
        "trustworthy_ce": result["trustworthy_ce"],
        "trustworthy_acc": result["trustworthy_acc"],
        "trustworthy": bool(result["trustworthy_ce"] and result["trustworthy_acc"]),  # Decision 17: dual-gated, both reported
        "clears_bar": bool(result["val_acc_full"] >= POSITIVE_CONTROL_BAR_ACC and result["trustworthy_ce"] and result["trustworthy_acc"]),
        "train_acc_per_depth": result["train_acc_per_depth"],
        "val_acc_per_depth": result["val_acc_per_depth"],
        "ce_gate": result["ce_gate"],
        "acc_gate": result["acc_gate"],
        "val_ce_trajectory": result["val_ce_trajectory"],
        "val_acc_full_trajectory": result["val_acc_full_trajectory"],
        "train_acc_full_trajectory": result["train_acc_full_trajectory"],
    }

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"control_{cfg.label}_seed{seed}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    out["_path"] = path
    return out


# ---------------------------------------------------------------------------
# Wide-shallow falsifier (clause (ii)) — parameter-matched to whichever control config passed
# clause (i). Reuses `depth_composition_toy.build_wide_shallow_clf` + `train_clf(..., gate_mode=
# GateMode.DUAL)` UNCHANGED: a single Linear->Tanh->Linear net has no depth ladder of its own, so
# DSEL-1b's nested trainer does not apply, and `train_clf`'s existing dual gate (built for exactly
# this — the F5c-b recurrent positive control) is the correct, already-built fit.
# ---------------------------------------------------------------------------


def _matched_wide_width(target_params: int, in_dim: int, n_classes: int) -> int:
    """Smallest wide-shallow width whose param count is closest to `target_params` (both directions checked)."""
    # params(w) = in_dim*w + w + w*n_classes + n_classes = w*(in_dim + 1 + n_classes) + n_classes
    denom = in_dim + 1 + n_classes
    est = max(1, round((target_params - n_classes) / denom))
    candidates = range(max(1, est - 2), est + 3)
    return min(candidates, key=lambda w: abs(dct.count_params(dct.build_wide_shallow_clf(w, in_dim, n_classes)) - target_params))


def run_wide_shallow_falsifier(cfg: ControlConfig, target_params: int, seed: int, device: str = "cpu", out_dir: str = OUT_DIR) -> dict:
    """Train the parameter-matched wide-shallow control at the SAME protocol as the passing control rung."""
    data = dct.make_word_data(GROUP, SEQ_LEN, seed, train_frac=cfg.train_frac)
    n_classes = data["n_classes"]
    in_dim = data["seq_len"] * data["n_gen"]
    wide_width = _matched_wide_width(target_params, in_dim, n_classes)

    torch.manual_seed(1000 * seed + 777)  # matches this file's/F5b's wide-shallow seeding convention
    net = dct.build_wide_shallow_clf(wide_width, in_dim, n_classes)
    params = dct.count_params(net)

    result, train_acc, val_acc = dct.train_clf(
        net, data, device=device, max_epochs=cfg.max_epochs, lr=cfg.lr, clip_max_norm=cfg.clip_max_norm, gate_mode=dct.GateMode.DUAL,
    )
    acc_gate = result.acc_gate

    out = {
        "arm": "wide_shallow_falsifier",
        "matched_to_rung": cfg.label,
        "group": GROUP.value,
        "seq_len": SEQ_LEN,
        "seed": seed,
        "wide_width": wide_width,
        "params": params,
        "target_params": target_params,
        "param_match_relative_diff": abs(params - target_params) / target_params,
        "n_classes": n_classes,
        "chance": 1.0 / n_classes,
        "n_train": int(data["x_tr"].shape[0]),
        "n_val": int(data["x_val"].shape[0]),
        "hyperparameters": {
            "lr": cfg.lr, "clip_max_norm": cfg.clip_max_norm, "max_epochs": cfg.max_epochs, "train_frac": cfg.train_frac,
        },
        "positive_control_bar_acc": POSITIVE_CONTROL_BAR_ACC,
        "val_acc": val_acc,
        "train_acc": train_acc,
        "trustworthy_ce": bool(result.trustworthy),
        "trustworthy_acc": bool(acc_gate["trustworthy"]) if acc_gate is not None else None,
        "trustworthy": bool(result.trustworthy and (acc_gate["trustworthy"] if acc_gate is not None else False)),
        "clears_bar": bool(val_acc >= POSITIVE_CONTROL_BAR_ACC and result.trustworthy and (acc_gate["trustworthy"] if acc_gate is not None else False)),
        "ce_gate": result.summary(),
        "acc_gate": acc_gate,
        "val_ce_trajectory": [[int(e), float(v)] for e, v in result.trajectory],
        "val_acc_trajectory": [[int(e), float(a)] for e, a in result.val_acc_trajectory],
        "train_acc_trajectory": [[int(e), float(a)] for e, a in result.train_acc_trajectory],
    }

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"falsifier_wide_seed{seed}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    out["_path"] = path
    return out


# ---------------------------------------------------------------------------
# DSEL-2c — re-run the positive control at specification (USER RULINGS 2026-07-21).
# Two switches on the existing dual-gate loop above: `targets` (ruling (ii)) and `schedule`
# (DSEL-1b §1b's schemes (a)/(b), MASTER Decision 20). (FULL, ALL_RUNGS) is `train_nested_clf_dual_gate`
# above, cited via CITED_CONTROL_PATHS, never rerun here.
# ---------------------------------------------------------------------------


class TargetScheme(enum.StrEnum):
    """Which label each ladder depth's exit is trained against (closed set, ruling (ii))."""

    FULL = "full"  # every exit trained against the full-word product label — the as-run stage-1 control
    PREFIX = "prefix"  # exit d trained against the product of the FIRST d generators (ruling (ii))


class ScheduleType(enum.StrEnum):
    """Which rungs get a gradient contribution on a given step (closed set, DSEL-1b §1b / Decision 20)."""

    ALL_RUNGS = "all_rungs"  # scheme (b): every rung, every step — DSEL-1b's default, the as-run stage-1 control
    SAMPLED = "sampled"  # scheme (a): a fresh per-sample d ~ Uniform{1..max_depth} draw every step, never yet run


DSEL2C_WIDTH = 2 * dct.NARROW_WIDTH  # 32 — "best-found config" (DSEL-2 stage 1's rung4_data75)
DSEL2C_TRAIN_FRAC = 0.75
DSEL2C_VAL_FRAC = 0.125  # remainder (0.125) is test — three-way split, binding (DSEL-2c spec)
DSEL2C_OUT_DIR = os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "DSEL2c")

# The existing (FULL, ALL_RUNGS) control DSEL-2c cites rather than reruns ("cite its artifacts, do
# not re-run it") — DSEL-2 stage 1's rung4_data75 cells, same width/train_frac/lr/clip as below.
CITED_CONTROL_PATHS = {
    0: os.path.join(OUT_DIR, "control_rung4_data75_seed0.json"),
    1: os.path.join(OUT_DIR, "control_rung4_data75_seed1.json"),
}

# The three corrected arms this task runs (FULL, ALL_RUNGS excluded — it is CITED_CONTROL_PATHS above).
ARM_LABELS: dict[tuple[TargetScheme, ScheduleType], str] = {
    (TargetScheme.FULL, ScheduleType.SAMPLED): "full_sampled",
    (TargetScheme.PREFIX, ScheduleType.ALL_RUNGS): "prefix_all_rungs",
    (TargetScheme.PREFIX, ScheduleType.SAMPLED): "prefix_sampled",
}


def _prefix_word_products(word_elem_ids: np.ndarray, grp: dict, ladder: tuple[int, ...]) -> dict[int, np.ndarray]:
    """Snapshot the running word product at every depth in `ladder` — ruling (ii)'s per-exit target.

    Same left-to-right fold `dct.word_product` (`depth_composition_toy.py:249-260`) uses — `P_0 =
    identity`, `P_t = elements[g_t] . P_{t-1}` — extended here to record EVERY ladder depth's prefix
    product rather than only the final one. `dct.word_product` itself is not called because it
    returns just the `seq_len`-deep fold; this needs every intermediate depth's product too (the
    fold at depth `max(ladder)` is identical to `dct.word_product`'s output — asserted in the
    selftest below).
    """
    mult = grp["mult"]
    n_rows, seq_len = word_elem_ids.shape
    prod = np.full(n_rows, grp["identity"], dtype=np.int64)
    wanted = set(ladder)
    snapshots: dict[int, np.ndarray] = {}
    for t in range(seq_len):
        prod = mult[word_elem_ids[:, t], prod]  # mult[g_t, P_{t-1}] = g_t . P_{t-1}
        depth = t + 1
        if depth in wanted:
            snapshots[depth] = prod.copy()
    return snapshots


def make_word_data_3way(
    group: dct.Group, seq_len: int, seed: int, ladder: tuple[int, ...], train_frac: float = DSEL2C_TRAIN_FRAC, val_frac: float = DSEL2C_VAL_FRAC
) -> dict:
    """Three-way train/val/test split (DSEL-2c spec: 0.75/0.125/0.125) with every ladder depth's prefix-product label alongside the full-word label.

    Reuses `dct.build_group` / `dct._all_or_sampled_words` UNCHANGED — the same word generation
    `dct.make_word_data` uses, so a given `(group, seq_len, seed)` produces the identical word order
    the cited (FULL, ALL_RUNGS) control trained on; only the split boundaries differ (two-way ->
    three-way).
    """
    grp = dct.build_group(group)
    n_gen = len(grp["generators"])
    gen_arr = np.array(grp["generators"], dtype=np.int64)

    words_gen = dct._all_or_sampled_words(n_gen, seq_len, seed)
    word_elem_ids = gen_arr[words_gen]
    prefix = _prefix_word_products(word_elem_ids, grp, ladder)
    full_depth = max(ladder)
    y_full = prefix[full_depth]  # prefix at the deepest rung IS the full-word product

    n = words_gen.shape[0]
    n_tr = round(train_frac * n)
    n_val = round(val_frac * n)
    n_test = n - n_tr - n_val
    onehot = np.eye(n_gen, dtype=np.float32)[words_gen].reshape(n, seq_len * n_gen)

    bounds = {"tr": (0, n_tr), "val": (n_tr, n_tr + n_val), "test": (n_tr + n_val, n)}
    out: dict = {
        "n_classes": len(grp["elements"]), "n_gen": n_gen, "seq_len": seq_len, "grp": grp,
        "n_train": n_tr, "n_val": n_val, "n_test": n_test,
    }
    for split, (i0, i1) in bounds.items():
        out[f"x_{split}"] = onehot[i0:i1]
        out[f"y_{split}_full"] = y_full[i0:i1]
        out[f"y_{split}_prefix"] = {d: prefix[d][i0:i1] for d in ladder}
    return out


def train_nested_clf_dsel2c(
    net: dct.NestedFeedForwardClf,
    data: dict,
    ladder: tuple[int, ...],
    targets: TargetScheme,
    schedule: ScheduleType,
    device: str = "cpu",
    max_epochs: int = dct.MAX_EPOCHS,
    *,
    lr: float = dct.RUNG0_LR,
    clip_max_norm: float | None = dct.RUNG0_CLIP_MAX_NORM,
    sample_seed: int = 0,
    check_every: int = dct.ACC_CHECK_EVERY,
    ce_patience: int = dct.PATIENCE,
    ce_min_delta: float = dct.MIN_DELTA,
    acc_patience: int = dct.ACC_PATIENCE,
    acc_min_delta: float = dct.ACC_MIN_DELTA,
    acc_divergence_abs_eps: float = dct.ACC_DIVERGENCE_ABS_EPS,
    acc_still_improving_eps: float = dct.ACC_STILL_IMPROVING_EPS,
) -> dict:
    """DSEL-2c's dual-gated nested trainer — extends `train_nested_clf_dual_gate` above.

    Adds the two USER RULING 2026-07-21 axes (`targets`, `schedule`) plus the three-way split (val
    gates stopping and best-weight restore exactly as before; test is scored and returned but never
    drives either).

    Gate pattern (CE tracker AND val-accuracy tracker, best weights on accuracy) reused verbatim from
    `train_nested_clf_dual_gate` above — only the training step (`targets`/`schedule`-dependent loss)
    and the eval block (three-way, not two-way) differ, so this is NOT called for (FULL, ALL_RUNGS) —
    that arm is `train_nested_clf_dual_gate` itself, cited via `CITED_CONTROL_PATHS`.

    `schedule=SAMPLED` requires `ladder` to be the contiguous range `1..max(ladder)` (asserted below)
    so a per-sample drawn depth indexes directly into `net.forward_all_depths`' output list.
    """
    if schedule is ScheduleType.SAMPLED and tuple(ladder) != tuple(range(1, max(ladder) + 1)):
        raise ValueError(f"SAMPLED schedule needs a contiguous 1..max_depth ladder, got {ladder}")

    net.to(device)
    x_tr = torch.as_tensor(data["x_tr"], dtype=torch.float32, device=device)
    x_val = torch.as_tensor(data["x_val"], dtype=torch.float32, device=device)
    x_test = torch.as_tensor(data["x_test"], dtype=torch.float32, device=device)
    y_tr_full = torch.as_tensor(data["y_tr_full"], dtype=torch.long, device=device)
    y_val_full = torch.as_tensor(data["y_val_full"], dtype=torch.long, device=device)
    y_test_full = torch.as_tensor(data["y_test_full"], dtype=torch.long, device=device)

    full_depth = max(ladder)
    if targets is TargetScheme.PREFIX:
        y_tr_by_depth = {d: torch.as_tensor(data["y_tr_prefix"][d], dtype=torch.long, device=device) for d in ladder}
        y_val_by_depth = {d: torch.as_tensor(data["y_val_prefix"][d], dtype=torch.long, device=device) for d in ladder}
        y_test_by_depth = {d: torch.as_tensor(data["y_test_prefix"][d], dtype=torch.long, device=device) for d in ladder}
    else:
        y_tr_by_depth = dict.fromkeys(ladder, y_tr_full)
        y_val_by_depth = dict.fromkeys(ladder, y_val_full)
        y_test_by_depth = dict.fromkeys(ladder, y_test_full)

    opt = torch.optim.Adam(net.parameters(), lr=lr)  # no weight_decay -- Decision 21: no regularisation in this task
    ce = nn.CrossEntropyLoss()
    n_tr = x_tr.shape[0]
    sample_gen = torch.Generator(device="cpu").manual_seed(sample_seed)  # SAMPLED's per-step depth draw, reproducible

    def train_step() -> None:
        opt.zero_grad()
        outs = net.forward_all_depths(x_tr)  # [logits_depth_1, ..., logits_depth_max] -- one forward, every exit
        if schedule is ScheduleType.ALL_RUNGS:
            loss = sum(ce(outs[d - 1], y_tr_by_depth[d]) for d in ladder) / len(ladder)
        else:  # SAMPLED -- scheme (a): a fresh per-sample depth draw every step (Decision 20)
            depths = torch.randint(1, full_depth + 1, (n_tr,), generator=sample_gen).to(device)  # Uniform{1..full_depth}
            row = torch.arange(n_tr, device=device)
            per_sample_logits = torch.stack(outs, dim=0)[depths - 1, row]  # each sample's OWN drawn depth's logits
            if targets is TargetScheme.PREFIX:
                stacked_targets = torch.stack([y_tr_by_depth[d] for d in ladder], dim=0)
                per_sample_targets = stacked_targets[depths - 1, row]
            else:
                per_sample_targets = y_tr_full
            loss = ce(per_sample_logits, per_sample_targets)
        loss.backward()
        if clip_max_norm is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=clip_max_norm)
        opt.step()

    ce_tracker = cvg.ConvergenceTracker(patience=ce_patience, min_delta=ce_min_delta)
    acc_tracker = cvg.ConvergenceTracker(patience=acc_patience, min_delta=acc_min_delta)
    best_state: dict | None = None
    final_epoch = max_epochs

    val_ce_traj: list[tuple[int, float]] = []
    val_acc_full_traj: list[tuple[int, float]] = []
    train_acc_full_traj: list[tuple[int, float]] = []

    net.train()
    for epoch in range(1, max_epochs + 1):
        train_step()

        if epoch % check_every == 0:
            net.eval()
            with torch.no_grad():
                outs_val = net.forward_all_depths(x_val)
                v_ce = sum(ce(outs_val[d - 1], y_val_by_depth[d]).item() for d in ladder) / len(ladder)
                v_acc_full = float((outs_val[full_depth - 1].argmax(1) == y_val_full).float().mean().item())
                outs_tr = net.forward_all_depths(x_tr)
                tr_acc_full = float((outs_tr[full_depth - 1].argmax(1) == y_tr_full).float().mean().item())
            net.train()

            val_ce_traj.append((epoch, v_ce))
            val_acc_full_traj.append((epoch, v_acc_full))
            train_acc_full_traj.append((epoch, tr_acc_full))

            ce_tracker.update(epoch, v_ce)
            is_new_best_acc = acc_tracker.update(epoch, -v_acc_full)
            if is_new_best_acc:
                best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}

            if ce_tracker.done and acc_tracker.done:  # AND, not OR
                final_epoch = epoch
                break

    if best_state is not None:
        net.load_state_dict(best_state)

    ce_result = ce_tracker.result(final_epoch=final_epoch)
    acc_result_neg = acc_tracker.result(final_epoch=final_epoch)  # fields below on the NEGATED (-acc) series
    best_acc = -acc_result_neg.best_val
    final_val_acc = -acc_result_neg.trajectory[-1][1] if acc_result_neg.trajectory else float("nan")
    diverged_acc = bool((best_acc - final_val_acc) > acc_divergence_abs_eps)
    still_improving_acc = bool(acc_result_neg.recent_improvement > acc_still_improving_eps)
    trustworthy_acc = bool(acc_result_neg.converged and not acc_result_neg.hit_cap and not still_improving_acc and not diverged_acc)

    acc_gate = {
        "converged": acc_result_neg.converged,
        "hit_cap": acc_result_neg.hit_cap,
        "still_improving": still_improving_acc,
        "diverged": diverged_acc,
        "trustworthy": trustworthy_acc,
        "stop_epoch": acc_result_neg.stop_epoch,
        "best_val": best_acc,
        "best_epoch": acc_result_neg.best_epoch,
        "recent_improvement": acc_result_neg.recent_improvement,
        "trajectory": [[int(e), float(-v)] for e, v in acc_result_neg.trajectory],
    }

    net.eval()
    with torch.no_grad():
        outs_tr = net.forward_all_depths(x_tr)
        outs_val = net.forward_all_depths(x_val)
        outs_test = net.forward_all_depths(x_test)
        train_acc_per_depth = {d: float((outs_tr[d - 1].argmax(1) == y_tr_by_depth[d]).float().mean().item()) for d in ladder}
        val_acc_per_depth = {d: float((outs_val[d - 1].argmax(1) == y_val_by_depth[d]).float().mean().item()) for d in ladder}
        test_acc_per_depth = {d: float((outs_test[d - 1].argmax(1) == y_test_by_depth[d]).float().mean().item()) for d in ladder}

    ce_summary = ce_result.summary()
    return {
        "full_depth": full_depth,
        "ce_gate": ce_summary,
        "acc_gate": acc_gate,
        "trustworthy_ce": bool(ce_result.trustworthy),
        "trustworthy_acc": trustworthy_acc,
        "hit_cap": bool(ce_summary["hit_cap"] or acc_gate["hit_cap"]),
        "train_acc_per_depth": {str(d): v for d, v in train_acc_per_depth.items()},
        "val_acc_per_depth": {str(d): v for d, v in val_acc_per_depth.items()},
        "test_acc_per_depth": {str(d): v for d, v in test_acc_per_depth.items()},
        "train_acc_full": train_acc_per_depth[full_depth],
        "val_acc_full": val_acc_per_depth[full_depth],
        "test_acc_full": test_acc_per_depth[full_depth],  # prefix-at-full-depth == full-word label either scheme
        "val_ce_trajectory": [[int(e), float(v)] for e, v in val_ce_traj],
        "val_acc_full_trajectory": [[int(e), float(v)] for e, v in val_acc_full_traj],
        "train_acc_full_trajectory": [[int(e), float(v)] for e, v in train_acc_full_traj],
    }


def run_dsel2c_arm(
    targets: TargetScheme, schedule: ScheduleType, seed: int, ladder: tuple[int, ...], device: str = "cpu", out_dir: str = DSEL2C_OUT_DIR
) -> dict:
    """Train + land one of DSEL-2c's three corrected arms (targets, schedule) at one seed."""
    arm_label = ARM_LABELS[(targets, schedule)]
    data = make_word_data_3way(GROUP, SEQ_LEN, seed, ladder, train_frac=DSEL2C_TRAIN_FRAC, val_frac=DSEL2C_VAL_FRAC)
    n_classes = data["n_classes"]
    in_dim = data["seq_len"] * data["n_gen"]
    max_depth = max(ladder)

    arm_index = 10 + sorted(ARM_LABELS.values()).index(arm_label)  # offset from DSEL-2 stage 1's own i=0 seeding
    torch.manual_seed(1000 * seed + arm_index)
    net = dct.NestedFeedForwardClf(max_depth, DSEL2C_WIDTH, in_dim, n_classes, readout=READOUT)
    params = dct.count_params(net)

    result = train_nested_clf_dsel2c(
        net, data, ladder, targets, schedule, device=device,
        max_epochs=dct.MAX_EPOCHS, lr=dct.RUNG0_LR, clip_max_norm=dct.RUNG0_CLIP_MAX_NORM,
        sample_seed=2000 * seed + arm_index,
    )

    out = {
        "arm": arm_label,
        "targets": targets.value,
        "schedule": schedule.value,
        "group": GROUP.value,
        "seq_len": SEQ_LEN,
        "readout": READOUT.value,
        "seed": seed,
        "ladder": list(ladder),
        "full_depth": max_depth,
        "width": DSEL2C_WIDTH,
        "params": params,
        "n_classes": n_classes,
        "chance": 1.0 / n_classes,
        "n_train": data["n_train"],
        "n_val": data["n_val"],
        "n_test": data["n_test"],
        "hyperparameters": {
            "lr": dct.RUNG0_LR, "clip_max_norm": dct.RUNG0_CLIP_MAX_NORM, "max_epochs": dct.MAX_EPOCHS,
            "train_frac": DSEL2C_TRAIN_FRAC, "val_frac": DSEL2C_VAL_FRAC, "test_frac": 1.0 - DSEL2C_TRAIN_FRAC - DSEL2C_VAL_FRAC,
            "weight_decay": 0.0, "dropout": 0.0,  # Decision 21 -- explicit, not accidental
        },
        "positive_control_bar_acc": POSITIVE_CONTROL_BAR_ACC,
        "test_acc": result["test_acc_full"],
        "val_acc": result["val_acc_full"],
        "train_acc": result["train_acc_full"],
        "trustworthy_ce": result["trustworthy_ce"],
        "trustworthy_acc": result["trustworthy_acc"],
        "trustworthy": bool(result["trustworthy_ce"] and result["trustworthy_acc"]),
        "hit_cap": result["hit_cap"],
        "clears_bar": bool(result["test_acc_full"] >= POSITIVE_CONTROL_BAR_ACC and result["trustworthy_ce"] and result["trustworthy_acc"]),
        "train_acc_per_depth": result["train_acc_per_depth"],
        "val_acc_per_depth": result["val_acc_per_depth"],
        "test_acc_per_depth": result["test_acc_per_depth"],
        "ce_gate": result["ce_gate"],
        "acc_gate": result["acc_gate"],
        "val_ce_trajectory": result["val_ce_trajectory"],
        "val_acc_full_trajectory": result["val_acc_full_trajectory"],
        "train_acc_full_trajectory": result["train_acc_full_trajectory"],
    }

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"dsel2c_{arm_label}_seed{seed}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    out["_path"] = path
    return out


def freeze_dsel2c(out_dir: str = DSEL2C_OUT_DIR) -> str:
    """Land `frozen.json`: which of DSEL-2c's arms cleared the bar on BOTH seeds.

    Reads each arm's two already-landed per-seed JSONs from disk (does not retrain). If several arms
    clear, prefers a SAMPLED-schedule clearer as cheapest (one forward's worth of gradient per sample
    vs `len(ladder)` for ALL_RUNGS, DSEL-2c spec's "cheapest if several clear"). If none clear, records
    the pre-authorized option-(a) finding rather than choosing among (a)/(b)/(c) itself.
    """
    per_arm: dict[str, dict] = {}
    for (targets, schedule), label in ARM_LABELS.items():
        seeds: dict[int, dict] = {}
        for seed in (0, 1):
            path = os.path.join(out_dir, f"dsel2c_{label}_seed{seed}.json")
            with open(path) as f:
                seeds[seed] = json.load(f)
        clears_both = all(seeds[s]["clears_bar"] for s in (0, 1))
        per_arm[label] = {
            "targets": targets.value,
            "schedule": schedule.value,
            "test_acc": {str(s): seeds[s]["test_acc"] for s in (0, 1)},
            "val_acc": {str(s): seeds[s]["val_acc"] for s in (0, 1)},
            "train_acc": {str(s): seeds[s]["train_acc"] for s in (0, 1)},
            "trustworthy": {str(s): seeds[s]["trustworthy"] for s in (0, 1)},
            "trustworthy_acc": {str(s): seeds[s]["trustworthy_acc"] for s in (0, 1)},
            "clears_bar_both_seeds": clears_both,
            # Path is DERIVED from the naming contract above, not read back out of the record:
            # the per-cell writer does not store its own path (the original `seeds[s]["_path"]`
            # read raised KeyError and left `frozen.json` unlandable). Root fix, 2026-07-21.
            "paths": {str(s): os.path.join(out_dir, f"dsel2c_{label}_seed{s}.json") for s in (0, 1)},
        }

    cleared = [label for label, a in per_arm.items() if a["clears_bar_both_seeds"]]
    cheapest = min(cleared, key=lambda label: 0 if per_arm[label]["schedule"] == ScheduleType.SAMPLED.value else 1) if cleared else None

    payload = {
        "cited_full_all_rungs_control": {
            "0": CITED_CONTROL_PATHS[0],
            "1": CITED_CONTROL_PATHS[1],
            "note": "existing (FULL, ALL_RUNGS) DSEL-2 stage-1 rung4_data75 cells -- cited by path, NOT re-run here",
        },
        "arms": per_arm,
        "cleared_arms": cleared,
        "unhalt_scheme": cheapest,
        "all_four_failed": len(cleared) == 0,
    }
    if not cleared:
        payload["finding"] = (
            "All four (targets, schedule) arms fail the positive-control bar (test_acc>=0.90 and "
            "trustworthy on both gates, both seeds). Per DSEL-2c's pre-authorized branch: recorded as "
            "option (a) -- this substrate does not carry a feed-forward depth signal -- an "
            "evidence-backed finding, never a default. DSEL-2/2b and the downstream feed-forward "
            "selection studies are PARKED pending user review; other strands continue."
        )

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "frozen.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Selftest — cheap wiring checks, no real training (mirrors depth_composition_toy.py's convention).
# ---------------------------------------------------------------------------


_SELFTEST_SMOKE_WIDTH = 4  # tiny width: wiring check only, not a capacity/optimization read
_SELFTEST_SMOKE_MAX_EPOCHS = 20
_SELFTEST_SMOKE_CHECK_EVERY = 5
_SELFTEST_SMOKE_PATIENCE = 100  # >> max_epochs/check_every so the smoke run always hits the epoch cap, never patience-stops
_SELFTEST_SMOKE_EXPECTED_CHECKPOINTS = _SELFTEST_SMOKE_MAX_EPOCHS // _SELFTEST_SMOKE_CHECK_EVERY


def run_selftest() -> bool:
    """Ladder-loading + tiny (few-epoch) dual-gate smoke run — wiring only, no bar is evaluated."""
    ladder = load_depth_ladder()
    ladder_ok = ladder == tuple(dct.PROBE_DEPTHS)
    print(f"[dsel2 selftest] ladder loaded from frozen.json: {ladder} (expect {tuple(dct.PROBE_DEPTHS)})  {'PASS' if ladder_ok else 'FAIL'}")

    data = dct.make_word_data(GROUP, SEQ_LEN, seed=0)
    net = dct.NestedFeedForwardClf(max(ladder), _SELFTEST_SMOKE_WIDTH, data["seq_len"] * data["n_gen"], data["n_classes"], readout=READOUT)
    result = train_nested_clf_dual_gate(
        net, data, ladder,
        max_epochs=_SELFTEST_SMOKE_MAX_EPOCHS, check_every=_SELFTEST_SMOKE_CHECK_EVERY,
        ce_patience=_SELFTEST_SMOKE_PATIENCE, acc_patience=_SELFTEST_SMOKE_PATIENCE,
    )
    shape_ok = "val_acc_full" in result and "trustworthy_acc" in result and len(result["val_acc_full_trajectory"]) == _SELFTEST_SMOKE_EXPECTED_CHECKPOINTS
    print(
        f"[dsel2 selftest] dual-gate smoke run ({_SELFTEST_SMOKE_MAX_EPOCHS} epochs, width={_SELFTEST_SMOKE_WIDTH}): "
        f"val_acc_full={result['val_acc_full']:.3f} keys/shape ok={shape_ok}  {'PASS' if shape_ok else 'FAIL'}"
    )

    # DSEL-2c checks: prefix-product correctness + wiring smoke runs for the three corrected arms.
    dct_full = dct.make_word_data(GROUP, SEQ_LEN, seed=0)
    grp = dct_full["grp"]
    gen_arr = np.array(grp["generators"], dtype=np.int64)
    words_gen = dct._all_or_sampled_words(len(grp["generators"]), SEQ_LEN, seed=0)
    word_elem_ids = gen_arr[words_gen]
    prefix = _prefix_word_products(word_elem_ids, grp, ladder)
    ref_full = dct.word_product(word_elem_ids, grp)
    prefix_matches_full_ok = bool(np.array_equal(prefix[max(ladder)], ref_full))
    print(f"[dsel2 selftest] prefix product at max(ladder) matches dct.word_product: {'PASS' if prefix_matches_full_ok else 'FAIL'}")

    data3 = make_word_data_3way(GROUP, SEQ_LEN, seed=0, ladder=ladder)
    split_ok = data3["n_train"] + data3["n_val"] + data3["n_test"] == words_gen.shape[0]
    print(
        f"[dsel2c selftest] 3-way split n_train={data3['n_train']} n_val={data3['n_val']} n_test={data3['n_test']} "
        f"sums to n={words_gen.shape[0]}  {'PASS' if split_ok else 'FAIL'}"
    )

    arm_shape_ok = True
    for (targets, schedule), label in ARM_LABELS.items():
        net_c = dct.NestedFeedForwardClf(max(ladder), _SELFTEST_SMOKE_WIDTH, data3["seq_len"] * data3["n_gen"], data3["n_classes"], readout=READOUT)
        res_c = train_nested_clf_dsel2c(
            net_c, data3, ladder, targets, schedule,
            max_epochs=_SELFTEST_SMOKE_MAX_EPOCHS, check_every=_SELFTEST_SMOKE_CHECK_EVERY,
            ce_patience=_SELFTEST_SMOKE_PATIENCE, acc_patience=_SELFTEST_SMOKE_PATIENCE,
        )
        ok_c = (
            "test_acc_full" in res_c and "val_acc_full" in res_c and "hit_cap" in res_c
            and len(res_c["val_acc_full_trajectory"]) == _SELFTEST_SMOKE_EXPECTED_CHECKPOINTS
        )
        print(f"[dsel2c selftest] arm={label} smoke run: test_acc_full={res_c['test_acc_full']:.3f} shape ok={ok_c}  {'PASS' if ok_c else 'FAIL'}")
        arm_shape_ok = arm_shape_ok and ok_c

    ok = bool(ladder_ok and shape_ok and prefix_matches_full_ok and split_ok and arm_shape_ok)
    print(f"[dsel2 selftest] {'PASS' if ok else 'FAIL'}")
    return ok


def _format_traj(pts: list, label: str) -> str:
    """One-line `epoch:value` rendering of a trajectory for console output."""
    s = " ".join(f"{int(e)}:{v:.3f}" for e, v in pts)
    return f"{label}: {s}"


def main() -> None:
    """CLI: run one escalation rung's full-depth control (or the falsifier / selftest)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--control", action="store_true", help="Run the full-depth positive control at --rung.")
    parser.add_argument("--rung", type=str, choices=sorted(RUNGS), default=RUNG0.label)
    parser.add_argument("--falsifier", action="store_true", help="Run the wide-shallow falsifier, param-matched to --rung's control JSON.")
    parser.add_argument("--dsel2c-arm", type=str, choices=sorted(ARM_LABELS.values()), help="Run one DSEL-2c corrected arm (targets x schedule) at --seed.")
    parser.add_argument("--dsel2c-freeze", action="store_true", help="Land DSEL2c/frozen.json from the already-landed per-arm-per-seed JSONs.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default=OUT_DIR)
    args = parser.parse_args()

    torch.set_num_threads(dct.TORCH_THREADS)
    device = os.environ.get("AUTOML_DEVICE", "cpu")

    if args.selftest:
        sys.exit(0 if run_selftest() else 1)

    ladder = load_depth_ladder()

    if args.control:
        cfg = RUNGS[args.rung]
        out = run_full_depth_control(cfg, args.seed, ladder, device=device, out_dir=args.out_dir)
        print(f"[dsel2] control rung={cfg.label} seed={args.seed} -> {out['_path']}")
        print(
            f"  params={out['params']} width={out['width']} lr={out['hyperparameters']['lr']} "
            f"clip={out['hyperparameters']['clip_max_norm']} train_frac={out['hyperparameters']['train_frac']}"
        )
        print(f"  val_acc={out['val_acc']:.4f} train_acc={out['train_acc']:.4f} bar={POSITIVE_CONTROL_BAR_ACC} clears_bar={out['clears_bar']}")
        print(f"  trustworthy_ce={out['trustworthy_ce']} trustworthy_acc={out['trustworthy_acc']}")
        print("    " + _format_traj(out["val_acc_full_trajectory"], "val_acc_full_traj "))
        print("    " + _format_traj(out["train_acc_full_trajectory"], "train_acc_full_traj"))
        sys.exit(0)

    if args.falsifier:
        cfg = RUNGS[args.rung]
        ctrl_path = os.path.join(args.out_dir, f"control_{cfg.label}_seed{args.seed}.json")
        with open(ctrl_path) as f:
            ctrl = json.load(f)
        out = run_wide_shallow_falsifier(cfg, ctrl["params"], args.seed, device=device, out_dir=args.out_dir)
        print(f"[dsel2] falsifier matched_to={cfg.label} seed={args.seed} -> {out['_path']}")
        print(f"  wide_width={out['wide_width']} params={out['params']} (target {out['target_params']}, rel_diff={out['param_match_relative_diff']:.4f})")
        print(f"  val_acc={out['val_acc']:.4f} train_acc={out['train_acc']:.4f} bar={POSITIVE_CONTROL_BAR_ACC} clears_bar={out['clears_bar']}")
        print(f"  trustworthy_ce={out['trustworthy_ce']} trustworthy_acc={out['trustworthy_acc']}")
        sys.exit(0)

    if args.dsel2c_arm:
        targets, schedule = next((t, s) for (t, s), label in ARM_LABELS.items() if label == args.dsel2c_arm)
        out = run_dsel2c_arm(targets, schedule, args.seed, ladder, device=device, out_dir=os.path.join(_EXAMPLES_DIR, "capacity_ladder_results", "DSEL2c"))
        print(f"[dsel2c] arm={out['arm']} targets={out['targets']} schedule={out['schedule']} seed={args.seed} -> {out['_path']}")
        print(f"  params={out['params']} width={out['width']} n_train={out['n_train']} n_val={out['n_val']} n_test={out['n_test']}")
        print(f"  test_acc={out['test_acc']:.4f} val_acc={out['val_acc']:.4f} train_acc={out['train_acc']:.4f} bar={POSITIVE_CONTROL_BAR_ACC} clears_bar={out['clears_bar']}")
        print(f"  trustworthy_ce={out['trustworthy_ce']} trustworthy_acc={out['trustworthy_acc']} hit_cap={out['hit_cap']}")
        print("    " + _format_traj(out["val_acc_full_trajectory"], "val_acc_full_traj "))
        print("    " + _format_traj(out["train_acc_full_trajectory"], "train_acc_full_traj"))
        sys.exit(0)

    if args.dsel2c_freeze:
        path = freeze_dsel2c()
        with open(path) as f:
            payload = json.load(f)
        print(f"[dsel2c] frozen -> {path}")
        print(f"  cleared_arms={payload['cleared_arms']} unhalt_scheme={payload['unhalt_scheme']} all_four_failed={payload['all_four_failed']}")
        sys.exit(0)

    parser.print_help()


if __name__ == "__main__":
    main()
