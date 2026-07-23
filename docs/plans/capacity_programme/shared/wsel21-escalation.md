# WSEL-21 — the d ≥ 8 training-protocol escalation: the written spec (rung (i))

**Scope of this document.** This is rung (i) of WSEL-21 (`docs/plans/capacity_programme/width.md`,
the `### WSEL-21` block): the protocol ladder, its graduation bar, its failure-branch done-state,
and a cost estimate. Authoring only — no runs, no driver edits, no new toy constructions. Rungs
(ii) (calibration) and (iii) (grid re-run) execute later, after this spec's adversarial read and
root go, per the standing delegation.

## 1. The wall this escalates (grounded in the recorded run, not restated from memory)

The WSEL-19 v2 bake-off (`shared/wsel19-toy-redesign.md`; ledger
`automl_package/examples/capacity_ladder_results/WSEL19/frozen_mf.json`) trains, per (d, geometry,
seed) triple, one dedicated `FlexibleWidthNN` per width ∈ {1..12} via the PORTED-arm protocol:
`batch_size = PORTED_BATCH_SIZE = 1_000_000_000` (one full-batch step per epoch — the constant's
own comment: "≥ any n_train => ONE batch per epoch"), `lr = PORTED_LR_DEFAULT = 0.01`,
`patience = PORTED_PATIENCE = 60`, `max_epochs = PORTED_N_EPOCHS_CAP = 6000`,
`min_delta = PORTED_MIN_DELTA = 1e-4`. <!-- source: `automl_package/examples/width_wsel4.py:105-108` (`PORTED_N_EPOCHS_CAP`, `PORTED_PATIENCE`, `PORTED_BATCH_SIZE`), `width_wsel4.py:97` (`PORTED_LR_DEFAULT`), `width_wsel4.py:107` (`PORTED_MIN_DELTA`) --> The multi-feature driver's per-width
constructor calls this protocol verbatim, with `batch_size` hardcoded (no parameter to override
it). <!-- source: `automl_package/examples/width_wsel19.py:900-922` (`_mf_new_model`, `batch_size=w4.PORTED_BATCH_SIZE`) -->

At d ∈ {8, 32}, every triple is `fit_status = void_for_fit`: best-fixed held-out MSE / noise floor
(σ² = 0.0025) runs 17.8–31.7× the noise floor, never approaching the anchor bar (§3 below).
<!-- source: `automl_package/examples/capacity_ladder_results/WSEL19/frozen_mf.json` `triples.d8_*`/`triples.d32_*` `ratio_to_noise_floor` --> <!-- numcheck-ignore: 17.8/31.7 are the min/max of the per-triple leaves, already recorded in width.md's verdict block -->
More data alone does not fix this at fixed step count: the F6 n_train fallback (1500 → 4000) pulls
the probed d=32 triple from 26.740 to 26.712 — a rounding-level move — <!-- numcheck-ignore: rounded readbacks; exact leaves live in `automl_package/examples/capacity_ladder_results/WSEL19/frozen_mf.json` (26.74) and `..._d32_axis_frozen_mlp_hard_nsel300_seed0_f6fallback.json` (26.712250709533688) --> while the SAME fallback on
`d8_oblique_seed2` moves 17.8x-class → 2.553, still 2.26× the anchor bar and still void. <!-- source: `automl_package/examples/capacity_ladder_results/WSEL19/wsel19_mf_d8_oblique_frozen_mlp_hard_nsel300_seed2.json` `validity_checks.ratio_to_noise_floor` -->
<!-- source: `automl_package/examples/capacity_ladder_results/WSEL19/wsel19_mf_d8_oblique_frozen_mlp_hard_nsel300_seed2.json` `validity_checks.ratio_to_noise_floor` --> The mixed
picture (one triple nearly unmoved by more data, another materially improved, neither reaching the
bar) is consistent with a **training-protocol** wall rather than a data-quantity wall: more data
changes what a fixed number of full-batch gradient steps sees per step, not how many steps are
taken or how they are taken.

**Mechanism, read off the recorded per-epoch trajectories (not conjectured):** at
`d8_axis_seed0_ntrain1500`, width 6's validation loss reaches a minimum of 0.0910974 at epoch 85, <!-- source: `automl_package/examples/capacity_ladder_results/WSEL19/_mf_cache/box_d8_axis_seed0_ntrain1500_w6_meta.json` (`trajectory`) -->
then *rises* to 0.0916373 by epoch 210 before patience (60 epochs without a ≥ 1e-4 improvement) <!-- source: `automl_package/examples/capacity_ladder_results/WSEL19/_mf_cache/box_d8_axis_seed0_ntrain1500_w6_meta.json` (`trajectory`) -->
stops training at epoch 217; width 12 shows the same pattern twice over (minima near epoch 61 and
epoch 173, rising in between and after) before stopping at epoch 233.
<!-- source: `automl_package/examples/capacity_ladder_results/WSEL19/_mf_cache/box_d8_axis_seed0_ntrain1500_w6_meta.json`, `..._w12_meta.json` (`trajectory`) --> Width 1's curve is
monotone but its useful improvement is essentially exhausted by epoch ~150 of the 552 it actually
runs (loss 0.07607→0.07205 from epoch 100 to 552, three significant figures already reached by <!-- numcheck-ignore: rounded readbacks of `automl_package/examples/capacity_ladder_results/WSEL19/_mf_cache/box_d8_axis_seed0_ntrain1500_w1_meta.json` `trajectory` leaves (e.g. 0.07205073535442352) -->
epoch ~200). <!-- source: same directory, `..._w1_meta.json` --> This is the "stalls within tens of
[full-batch] steps" described in the v2 verdict block <!-- source: `docs/plans/capacity_programme/width.md:1786` -->: full-batch gradient descent on this construction finds a shallow, non-monotone
region of the loss surface and spends most of its epoch budget drifting inside it rather than
escaping it. Stochastic per-step noise (mini-batching) is the standard lever for escaping exactly
this failure mode; it is why the ladder below starts there, not with the LR/clip/warmup/init/
normalize ladder already on record.

**This is a different escalation axis from two existing mechanisms, not a replacement for either:**
- **MASTER Decision 16's ported-arm escalation ladder** (`width_wsel4.py`'s `_train_ported_escalated`:
  LR sweep → grad-clip → warmup → init scheme → normalize-inputs, `--lr`/`--grad-clip-norm`/
  `--warmup-epochs`/`--init-scheme`/`--normalize-inputs`) <!-- source: `automl_package/examples/width_wsel4.py:296-309,715-722` --> tunes optimization *around* a full-batch
  base case that is assumed to work. WSEL-21 questions the full-batch base case itself at d ≥ 8;
  it is a ladder underneath that one, not a duplicate of it.
- **F6's n_train fallback** (1500 → 4000, pre-authorized in the fit gate) is a data-quantity lever,
  orthogonal to the training-protocol levers below. It remains available per-triple during rung
  (iii) exactly as today, composed with whichever rung graduates (see §5).

## 2. Scope boundary: training protocol, not measurement semantics

Every rung below changes only how the per-width nets are optimized (`batch_size`, an LR schedule,
`patience`); none changes:
- how held-out/report error is computed, or what `noise_floor`, `best_fixed_held_out_mse`, or
  `ratio_to_noise_floor` mean (`width_wsel19.py`'s existing `_load_calibration_or_refuse`/fit-gate
  arithmetic is untouched);
- what a "trustworthy" trajectory is. `trustworthy = replay.trustworthy and not hit_cap` is a fixed
  function of a per-epoch validation-loss list, `patience`, and `min_delta`
  (`w4._replay`/`cvg.ConvergenceTracker`). <!-- source: `automl_package/examples/width_wsel19.py:981-983` (`trustworthy = bool(replay.trustworthy and not hit_cap)`); `automl_package/examples/width_wsel4.py:272-282` (`_replay`) --> Changing `batch_size` or the LR schedule changes the *trajectory* fed
  into this fixed function, not the function; this is training protocol.
- how validation loss itself is scored: validation is evaluated on the fixed, pre-declared
  validation split once per epoch regardless of the training batch size — mini-batching changes
  only the gradient steps taken *between* those evaluations, never what is evaluated.

**Patience, flagged explicitly per this task's instruction.** Rung D (§4) raises `patience` as a
single-difference change. `patience` is a parameter into the same fixed trustworthy/hit_cap
function above, not a redefinition of it — and raising it has direct strand precedent as a
protocol-level (not measurement-level) move: the calibration's own failure taxonomy already treats
"raise the epoch cap ×2, one retrain" as a same-precedent repair, not a measurement change.
<!-- source: `docs/plans/capacity_programme/shared/wsel19-toy-redesign.md` §5.2(a) ("exactly ONE raised-cap retrain (cap × 2 — the WSEL-8 same-precedent repair)") --> Rung D is the same move applied to `patience`
instead of `max_epochs`. It is named here explicitly, as instructed, rather than folded silently
into the ladder.

**Nothing proposed below touches measurement semantics.** No rung changes the report/train/val
split boundaries, the noise-floor definition, or the anchor-ratio arithmetic. If a future rung ever
proposed such a change (e.g., a different validation cadence, a different held-out split), it would
require the full toy-gate process (adversarial read as a measurement-semantics change), not this
delegation — flagged here so the boundary is checkable, not because any current rung crosses it.

**Considered and excluded from this ladder (scope discipline):** `min_delta` (currently 1e-4,
`PORTED_MIN_DELTA`) is a fourth knob that interacts with patience under noisier mini-batch
trajectories, but the task's spec names only mini-batch size, LR schedule, and patience — `min_delta`
is left untouched to keep each rung a single difference from its predecessor; if rungs A–D all fail,
a follow-up ladder extension (not this one) would be the place to raise it.

## 3. The designated calibration-scale cell

**(d = 8, geometry = AXIS, seed = 2), n_train = 1500** (the grid's own provisioning, matching the
anchor block's scale) is the single cell every rung is tested against.

**Why this one, not another d = 8 or d = 32 triple:** among the six d = 8 triples that ran under the
*unmodified* full-batch protocol (excluding `d8_oblique_seed2`, which already received the F6
n_train = 4000 data fallback and would confound "did mini-batching help" with "did more data
help"), `d8_axis_seed2` has the lowest (least-void) ratio: 17.835, versus 26.726/20.751 for <!-- source: `automl_package/examples/capacity_ladder_results/WSEL19/frozen_mf.json` (`triples.d8_*.ratio_to_noise_floor`) -->
`d8_axis_seed{0,1}` and 17.889/17.83 for `d8_oblique_seed{0,1}`. <!-- source: `automl_package/examples/capacity_ladder_results/WSEL19/frozen_mf.json` (`triples.d8_oblique_seed0.ratio_to_noise_floor`, `triples.d8_oblique_seed1.ratio_to_noise_floor`) -->
<!-- source: `automl_package/examples/capacity_ladder_results/WSEL19/frozen_mf.json` `triples.d8_axis_seed2.ratio_to_noise_floor`, `triples.d8_axis_seed0.ratio_to_noise_floor`, `triples.d8_axis_seed1.ratio_to_noise_floor`, `triples.d8_oblique_seed0.ratio_to_noise_floor`, `triples.d8_oblique_seed1.ratio_to_noise_floor` --> It is therefore the single most sensitive test available at unit cost: if the
closest-to-passing d = 8 triple cannot graduate a rung, no farther-from-passing d = 8 or d = 32
triple would either, and no calibration compute is wasted confirming that. If it *does* graduate, rung
(iii) still re-runs the full 12-triple void set (§5) before any bake-off number is read — the
calibration cell only decides which rung's protocol earns that re-run, never substitutes for it.
AXIS (not OBLIQUE) is chosen for the calibration cell specifically because `H = I_d` exactly there
(no Householder arithmetic), keeping the calibration test's own geometry as simple as the anchor's.
<!-- source: `docs/plans/capacity_programme/shared/wsel19-toy-redesign.md` §2 ("AXIS: H = I_d exactly (no arithmetic at all)") -->

Current state of the designated cell (full-batch protocol, for reference — not re-cited per rung,
cited once here): `ratio_to_noise_floor = 17.835`, `regime_visible = false`, `fit_status = <!-- source: `automl_package/examples/capacity_ladder_results/WSEL19/frozen_mf.json` (`triples.d8_axis_seed2`) -->
void_for_fit`, `n_train_fallback_applied = false`.
<!-- source: `automl_package/examples/capacity_ladder_results/WSEL19/frozen_mf.json` `triples.d8_axis_seed2` -->

## 4. The rung ladder (single-difference discipline)

Each rung changes exactly one named knob from its predecessor (rung A's predecessor is the
unmodified full-batch protocol in §1); every other knob is held at its predecessor's value. The
ladder stops at the first rung that graduates (§5); later rungs are not run once one graduates.

| Rung | Single change | Value | Held fixed |
|---|---|---|---|
| A — `MINIBATCH_128` | `batch_size` | 1,000,000,000 → 128 | `lr=0.01`, `patience=60`, `max_epochs=6000`, `min_delta=1e-4` |
| B — `MINIBATCH_32` | `batch_size` | 128 → 32 | same as A otherwise |
| C — `LR_DECAY` | LR schedule | constant 0.01 → cosine-annealed 0.01 → 0 over `max_epochs` | `batch_size` at B's value (32); `patience`, `max_epochs`, `min_delta` as A |
| D — `PATIENCE_RAISED` | `patience` | 60 → 180 (×3) | `batch_size`/LR-schedule at C's values; `max_epochs`, `min_delta` as A |

**Rationale per rung, tied to §1's mechanism:**
- **A/B (mini-batch first):** the recorded trajectories show full-batch GD drifting inside a
  shallow, non-monotone region rather than escaping it (§1). Gradient noise from small batches is
  the standard lever for escaping such regions; two sizes are tested because the right amount of
  noise is not knowable in advance — 128 gives ⌈1200/128⌉ = 10 steps/epoch at this cell's
  `n_train_used = 1200`, 32 gives ⌈1200/32⌉ = 38 steps/epoch, an order of magnitude more stochastic
  updates per epoch than full-batch's 1. <!-- source: `automl_package/examples/capacity_ladder_results/WSEL19/_mf_cache/box_d8_axis_seed0_ntrain1500_w1_meta.json` (`n_train_used: 1200`, shared across seeds at this n_train per §3b's provisioning) -->
- **C (LR decay), only if A and B both fail:** the same trajectories show late-training *rises* in
  validation loss (§1's width-6/width-12 examples), consistent with the optimizer overshooting a
  shallow minimum once it is near one. A decaying LR reduces step size as training proceeds without
  reintroducing full-batch determinism, directly targeting that overshoot without touching batch
  size again (single difference from B).
- **D (patience), only if A–C fail:** mini-batch training produces an inherently noisier epoch-to-epoch
  validation curve than full-batch's (each epoch's gradient direction now depends on the shuffle,
  not just the fixed full dataset); `patience = 60` was calibrated against the full-batch protocol's
  smoother curve. Raising it (×3, deliberately more headroom than the ×2 raised-cap precedent since
  the noise source here is per-epoch stochasticity, not merely a slow deterministic approach) gives
  the trajectory room to recover from a noise-induced plateau before being declared converged. This
  multiplier is a judgment call, stated as one — not derived from a formula — and is the reason it is
  its own rung rather than bundled into A.

**Existing machinery this reuses, not reinvents:** `width_wsel4.py::_train_ported_escalated` already
implements a mini-batch training loop with a configurable `batch_size` and per-epoch validation,
best-weights early stopping (`automl_package/examples/width_wsel4.py:296-361`) — the shape every
rung above should follow at rung (ii), generalized to `in_dim` exactly as `_mf_new_model` already
generalizes the protocol-default path (`width_wsel19.py:900-922`). Rung (ii) is parameter-threading
into existing shapes, not new training code.

**Two build-time hazards for rung (ii), named here so they are not rediscovered mid-run:**
1. `_mf_new_model`/`_mf_train_one_width`/`_mf_get_or_train_one_width` currently hardcode
   `batch_size=w4.PORTED_BATCH_SIZE` with no parameter to override it.
   <!-- source: `automl_package/examples/width_wsel19.py:900-922,925-936,957-994` --> Rung (ii) must thread `batch_size` (and, for rung C, an LR-schedule
   argument) through these exactly as `width_wsel4.py`'s `_run_ported_cell` already threads
   `args.batch_size`/`args.lr` (`automl_package/examples/width_wsel4.py:386-394`).
2. `_mf_model_cache_paths`' cache tag encodes `d`, `geometry`, `seed`, `n_train`, `width` but not
   batch size, LR schedule, or patience. <!-- source: `automl_package/examples/width_wsel19.py:939-944` --> Without an
   additional distinguishing `cache_tag` (the function already accepts one — used today for the
   raised-cap retrain), a rung's nets would collide with the existing full-batch cache at the same
   (d, geometry, seed, n_train, width) key and silently load the wrong model. Rung (ii) must pass a
   per-rung `cache_tag` (e.g. `_mb128`, `_mb32_lrdecay`, `_mb32_lrdecay_pat180`).

## 5. Graduation bar (reusing the v2 calibration machinery exactly — no new bar invented)

A rung **graduates** iff, on the designated calibration-scale cell (§3), retrained under that
rung's protocol:

1. **Every one of the 12 per-width nets is trustworthy:** `trustworthy = True` and `hit_cap = False`
   for widths 1–12 — the identical field and identical semantics the calibration's own regime/anchor
   blocks already require ("every gate-counted net trustworthy per the cached meta's own field").
   <!-- source: `docs/plans/capacity_programme/shared/wsel19-toy-redesign.md` §5.2, amendment A3 --> AND
2. **`ratio_to_noise_floor ≤ anchor_ratio_to_noise_floor`**, where `ratio_to_noise_floor =
   best_fixed_held_out_mse / 0.0025` recomputed at this cell under the rung's protocol, and
   `anchor_ratio_to_noise_floor = 1.1278630234301088` is the SAME pinned value the v2 fit gate <!-- numcheck-ignore: the pin; exact leaf verified at the adversarial read against `automl_package/examples/capacity_ladder_results/WSEL19/wsel19_calibration_d1.json` top-level `anchor_ratio_to_noise_floor` -->
   already uses everywhere else in the grid — read once, from the d = 1 calibration artifact, never
   recomputed at d = 8. <!-- source: `automl_package/examples/capacity_ladder_results/WSEL19/wsel19_calibration_d1.json` top-level `anchor_ratio_to_noise_floor` (= `anchor_block.anchor_ratio_to_noise_floor`, the worst-of-3-seeds ratio at the grid's own n_train = 1500 provisioning) --> This is exactly the §5.3/F6 fit-gate comparison
   already implemented in the grid driver's `validity_checks.ratio_to_noise_floor` vs
   `validity_checks.calibration_anchor_ratio_to_noise_floor` fields.
   <!-- source: `automl_package/examples/capacity_ladder_results/WSEL19/wsel19_mf_d8_axis_frozen_mlp_hard_nsel300_seed0.json` `validity_checks.ratio_to_noise_floor`, `validity_checks.calibration_anchor_ratio_to_noise_floor` -->

**Scope of the bar, stated so it is not over-read:** only the fit gate governs ladder graduation —
not the full §5 validity chain (regime visibility, the hiddenness falsifier). Those are properties
of the *aggregated* bake-off at the full grid, evaluated once at rung (iii) on all 12 void triples;
requiring them at the single-cell calibration stage would mean re-deriving multi-seed,
multi-fraction regime statistics from n = 1, which the v2 design itself does not do even for its own
d = 1 calibration (the regime block there uses 3 seeds specifically to apply the majority bar). The
ladder's job is narrower: decide whether best-fixed fit quality reaches the trustworthiness bar at
all; whether the *regime* (practical-floor gap) becomes visible is rung (iii)'s question, on the
full re-run.

**Sequential rule:** rungs run in order A → B → C → D; the ladder stops at the first graduating
rung. A graduating rung authorizes rung (iii): the full grid re-run on all 12 void (d, geometry,
seed) triples (d ∈ {8, 32} × geometry ∈ {axis, oblique} × seed ∈ {0, 1, 2}) at exactly that rung's
protocol, with A1 aggregation and `mf_aggregate.py` unchanged, per the task block.

## 6. Failure branch as a done-state (checkable ledger fields, no user gate)

If no rung (A–D) graduates, **d ≥ 8 is recorded UNREACHABLE under the protocol family** — a
done-state, not a halt. This is recorded in a new sibling ledger file (not inside `frozen_mf.json`,
which stays cell-glob-driven and untouched by this escalation per the task's non-goals):

`automl_package/examples/capacity_ladder_results/WSEL19/wsel21_escalation_ledger.json`

```json
{
  "calibration_cell": {"d": 8, "geometry": "axis", "seed": 2, "n_train": 1500},
  "anchor_ratio_to_noise_floor": null,
  "anchor_source": "automl_package/examples/capacity_ladder_results/WSEL19/wsel19_calibration_d1.json",
  "baseline": {"ratio_to_noise_floor": null, "all_widths_trustworthy": null},
  "rungs": [
    {"name": "MINIBATCH_128", "batch_size": 128, "lr": 0.01, "lr_schedule": "none", "patience": 60,
     "ratio_to_noise_floor": null, "all_widths_trustworthy": null, "graduated": null},
    {"name": "MINIBATCH_32", "batch_size": 32, "lr": 0.01, "lr_schedule": "none", "patience": 60,
     "ratio_to_noise_floor": null, "all_widths_trustworthy": null, "graduated": null},
    {"name": "LR_DECAY", "batch_size": 32, "lr": 0.01, "lr_schedule": "cosine_to_zero", "patience": 60,
     "ratio_to_noise_floor": null, "all_widths_trustworthy": null, "graduated": null},
    {"name": "PATIENCE_RAISED", "batch_size": 32, "lr": 0.01, "lr_schedule": "cosine_to_zero", "patience": 180,
     "ratio_to_noise_floor": null, "all_widths_trustworthy": null, "graduated": null}
  ],
  "graduated_rung": null,
  "verdict": "UNREACHABLE",
  "verdict_scope": "d>=8, protocol family = {full-batch->minibatch{128,32}, LR-constant->cosine-decay, patience 60->180}; d>=8 not reachable at any tested combination",
  "provenance": {"timestamp_utc": null, "git": null}
}
```

(`null` fields above are the pre-registered *shape*; rung (ii) fills them as each rung is measured —
including `anchor_ratio_to_noise_floor` (read from `anchor_source`, never retyped),
`baseline.ratio_to_noise_floor` (read from `frozen_mf.json`), and `baseline.all_widths_trustworthy`
(READ from the per-width `_mf_cache/*_meta.json` trustworthy flags — see the §9 addendum, finding 3).
`verdict` becomes `"GRADUATED"` and `graduated_rung` names the rung the moment any rung's two
conditions hold — at which point rung (ii) stops early and rung (iii) dispatches.)

**Report consequence, verbatim template:** *"d ≥ 8 is recorded UNREACHABLE under the protocol
family tested (full-batch → mini-batch {128, 32}; constant LR → cosine decay; patience 60 → 180);
ruling 6's decisive d ∈ {8, 32} test and the large-d half of the input-size-relative rule stay OPEN
— protocol-limited, not architecture-decided."* This is the terminal state for WSEL-21 absent a
graduating rung: not a halt, not a user gate — `flexnn-package.md` FP-13 evaluates its trigger
against this recorded UNREACHABLE state exactly as it would against a GRADUATED one.

## 7. Cost estimate per rung

**Method.** No wall-clock field is recorded anywhere in the WSEL-19 v2 cell JSONs or the
calibration artifact (`train_wall_clock_s` is a field other width tasks record; this driver does
not). Costs below are derived from the per-width-net cache files' filesystem mtimes — the delta
between consecutive `_mf_cache/box_*_w{k}.pt` write times is that width's own training time, and the
span from the first to the last width's write time is the full 12-width sweep's wall-clock. This is
a measurement of what already ran, not a projection, for the full-batch baseline; the mini-batch
projections in this section are explicitly marked as estimates, not measurements.

**Measured full-batch baseline (existing runs, CPU, `OMP_NUM_THREADS=4`):**
- Designated calibration cell (`d8_axis_seed2_ntrain1500`, 12 widths): first write 23:45:18.05,
  last write 23:46:04.62 (2026-07-22 local) → **≈ 46.6 s** for the full 12-width sweep.
  <!-- source: filesystem mtimes, `automl_package/examples/capacity_ladder_results/WSEL19/_mf_cache/box_d8_axis_seed2_ntrain1500_w{1,12}.pt` --> <!-- numcheck-ignore: derived from file mtimes, not a JSON field; method stated in this section -->
- A second d = 8 triple for range (`d8_axis_seed0_ntrain1500`): **≈ 31.2 s** for 12 widths.
  <!-- source: filesystem mtimes, `..._box_d8_axis_seed0_ntrain1500_w{1,12}.pt` -->
- A d = 32 triple (`d32_axis_seed0_ntrain1500`): **≈ 11.2 s** for 12 widths — *cheaper* than d = 8,
  consistent with §1's mechanism (the higher-decoy-count construction stalls even sooner, so
  patience trips faster). <!-- source: filesystem mtimes, `..._box_d32_axis_seed0_ntrain1500_w{1,12}.pt` -->
- A DECIDED level for contrast (`d2_axis_seed0_ntrain1500`, genuinely converging, not stalled):
  **≈ 255.5 s** for 12 widths — roughly 5–8× the d = 8/d = 32 full-batch cost, because these nets
  actually spend their epoch budget improving rather than stalling.
  <!-- source: filesystem mtimes, `..._box_d2_axis_seed0_ntrain1500_w{1,12}.pt` -->

**Rung (ii) — calibration only (4 rungs × 1 twelve-width cell):** batch size 128 gives
⌈1200/128⌉ = 10 gradient steps/epoch versus full-batch's 1; batch size 32 gives ⌈1200/32⌉ = 38.
Per-step compute is lower on CPU for the smaller batches but Python/DataLoader overhead per step
does not fall proportionally for these tiny (32, 32)-hidden nets, so a step-count-only projection
(10–38× the full-batch per-net cost) is a *lower* bound, not an estimate of the true multiplier.
Even a pessimistic 50–100× multiplier on the ≈ 46.6 s baseline is 40–80 minutes for one rung's
12-width sweep; four rungs sequentially is at most a few hours on a single CPU core, and each rung
after the first only runs if its predecessor failed. **Rung (ii) is cheap under any plausible
multiplier** — the absolute baseline times are tens of seconds, not minutes.

**Rung (iii) — grid re-run at the graduated rung, if any (12 void triples × 12 widths = 144 nets,
plus router refits at seconds-scale per the v2 spec's own §7):** applying the same lower-bound
multiplier range to the two observed baselines — d = 8 triples (≈ 31–47 s baseline × 6 triples) and
d = 32 triples (≈ 11 s baseline × 6 triples) — projects to roughly 30 minutes to several hours
depending on which rung graduated (higher rungs cost more: rung B's 38 steps/epoch costs more than
rung A's 10). **This projection must not be trusted for bulk dispatch without confirmation:**
per the v2 spec's own pre-dispatch discipline ("the d=1 calibration plus the FIRST d=32 sweep
measure per-net cost before the bulk dispatches" <!-- source: `docs/plans/capacity_programme/shared/wsel19-toy-redesign.md` §7 --> ), rung (iii) must re-run ONE d = 8 and ONE d = 32 triple first, measure
their actual wall-clock at the graduated protocol, and only then dispatch the remaining 10 triples
in the background (`systemd-inhibit`, `OMP_NUM_THREADS=4`, ≤ 3 concurrent heavy processes — the
existing convention, unchanged).

## 8. Non-goals

Restated from the task block, binding on rungs (i)–(iii): no new toy constructions; no
labelling/tolerance changes (WSEL-22 owns those); no re-run of decided levels (1-D, d = 2); no
router-default changes (FP-5.b binds). Added by this spec: no `min_delta` rung (§2); no change to
the fit-gate arithmetic, the noise-floor definition, or the report/train/val split boundaries at any
rung (§2); no bake-off number is read from any rung's calibration-cell retrain — only rung (iii)'s
full grid re-run produces bake-off-eligible cells.

## 9. ADVERSARIAL-READ ADDENDUM (root, 2026-07-23) — GO, with one required amendment

The read verified the pinned anchor against the calibration artifact (top-level
`anchor_ratio_to_noise_floor`, = the anchor block's, per the v2 spec's own gate definition — the
regime block's 1.889 is a different provisioning and would have been the wrong pin <!-- source: `automl_package/examples/capacity_ladder_results/WSEL19/wsel19_calibration_d1.json` (`regime_block.anchor_ratio_to_noise_floor`) -->), both §4 build
hazards against the driver source, and the §2 protocol-not-measurement boundary (accepted under the
standing delegation). Three findings:

1. **REQUIRED AMENDMENT — rung C's schedule horizon is mismatched as written and would make the
   rung a near-no-op.** Cosine-to-zero over `max_epochs = 6000` delivers almost no decay at the
   epochs where these runs actually stop: the recorded full-batch trajectories stop at epochs
   ~217–552 (§1), and at epoch 300 of a 6000-epoch cosine the LR still sits at ~99% of its initial
   value. Rung C would then test "constant LR with a cosmetic schedule attached", and its failure
   would be misread as "LR decay does not help". **Amendment (binding on rung (ii) before rung C
   runs; rungs A/B are unaffected):** rung C's schedule is cosine-to-zero with
   `T_max = 3 × (the median stopping epoch across the 12 widths of rung B's own recorded
   trajectories)` — a deterministic function of data that necessarily exists whenever rung C runs
   (C runs only after B fails), introducing no new free constant beyond the ×3 headroom multiplier,
   which is the same judgment-call class as rung D's ×3 and is stated as such. Training may run past
   `T_max` at the floor LR-of-zero only until patience stops it.
2. **The §3 "most sensitive cell" implication is a heuristic, not a theorem** — a protocol change
   need not preserve the full-batch ordering of triples, so "the closest-to-passing cell failed ⇒
   no other would graduate" is an economy argument for the screen, not a proof. Acceptable as
   written BECAUSE rung (iii) re-runs all 12 triples regardless of which rung graduates; recorded
   so the screen's verdict is never quoted as evidence about other triples.
3. **The §6 ledger template's baseline `"all_widths_trustworthy": false` was asserted, not read** —
   the cell JSON records no such field (root-checked: `validity_checks` carries fit/regime fields
   only). Corrected in the template above's semantics: rung (ii) fills the baseline field by
   READING the per-width `_mf_cache/*_meta.json` trustworthy flags, and until then it is `null`,
   never an assumed `false`.

**Verdict: GO** for rung (ii) authoring (flag-threading + per-rung `cache_tag` + ledger writer) and
for rungs A/B execution at the root once that authoring lands; rung C is gated on amendment 1,
which this addendum fully specifies — no further design round needed.
