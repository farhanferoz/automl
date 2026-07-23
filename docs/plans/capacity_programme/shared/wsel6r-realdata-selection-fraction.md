# WSEL-6-R — the real-data selection-fraction reopen: written spec

**Scope of this document.** This is the spec for **WSEL-6-R**
(`docs/plans/capacity_programme/width.md`, wave line — grep `WSEL-6-R`, `width.md:694`), the task
fired by WSEL-24's causal verdict that the frozen 15% SELECT fraction (`width.md:990`, WSEL-6) is
toys-only and underpowered on real data. Authoring only — no runs, no driver edits, no ledger
writes. The driver is built and the grid is executed by a later task, per the standing
worker-writes-the-driver / root-runs-the-grid split
(`~/.claude/model-routing-policy.md`; the section 6 contract below binds that later task).

## 0. Grounding (read off disk, not restated from memory)

**WSEL-6's toys verdict and its real-data reopen.** WSEL-6 answered "15% of the training portion
suffices for every arm" on the two SYNTHETIC toy tiers only
(`docs/plans/capacity_programme/width.md:990`, ledger
`automl_package/examples/capacity_ladder_results/WSEL6/frozen.json` — `fraction: 0.15`,
`fractions_swept_pct: [5, 10, 15, 25, 40]`, `seeds: [0, 1, 2]`, `n_cells_present: 90`). The block's
own 2026-07-23 note narrows its scope label to "toys" and reopens the real-data half as WSEL-6-R
(`width.md:992-997`).

**WSEL-24's causal demonstration.** WSEL-9's real-data battery (5 datasets, 3 seeds, 12 widths,
ledger `automl_package/examples/capacity_ladder_results/WSEL9/`) fields the current frozen 15%
fraction and yields SELECT-set sizes of 37 / 53 / 92 rows on yacht / diabetes / energy (vs 983 /
2477 on kin8nm / california) — `n_selection_used`,
`automl_package/examples/capacity_ladder_results/WSEL9/attribution.json` (`yacht.n_selection_used`,
`diabetes.n_selection_used`, `energy.n_selection_used`, `kin8nm.n_selection_used`,
`california.n_selection_used`). A winning width exists but was lost by selection on ALL FIVE
datasets (`winning_width_exists: true`, `selection_lost_it: true` for every one of `diabetes`,
`yacht`, `energy`, `kin8nm`, `california` — same file). The dominant-cause split is clean:
`diabetes`/`yacht`/`energy` → `selection_set_size`; `kin8nm`/`california` → `rule_objective_mismatch`
(same file, `<dataset>.dominant_cause`).

A **5-cell probe** (inside WSEL-24's pre-declared 6-cell training cap) held the rule
(`cheapest_within_tolerance`), the 2·SE multiplier (`TOLERANCE_SE_MULTIPLE = 2.0`,
`automl_package/utils/capacity_selection.py:37`) and the bootstrap estimator
(`automl_package/utils/numerics.py:251`) IDENTICAL and only enlarged the SELECT carve to
`probe_fraction_pct = 50` — CAUSAL, not correlational, on both datasets it touched:

| dataset | seed | pick vs oracle | at 15% (`n_selection_used`) | at 50% (`n_selection_used`) | `within_tolerance` 15% → 50% |
|---|---:|---|---:|---:|---|
| diabetes | 1 | w_lo=1 (shipped) vs w_hi=3 (oracle) | 53 | 177 | `True` → `False` (flips to CORRECTLY rejecting w_lo) |
| yacht | 0 | w_lo=2 (shipped) vs w_hi=8 (oracle) | 37 | 123 | `True` → `False` (flips to CORRECTLY rejecting w_lo) |

<!-- source: `automl_package/examples/capacity_ladder_results/WSEL9/attribution.json` (`diabetes.selection_set_size_probe`, `yacht.selection_set_size_probe`) -->
Both probe cells used `probe_fraction_pct = 50`, which is the split builder's OWN hard ceiling —
`_build_split`'s p1/p2 halves make p2 exactly half the pool, so `fraction_pct=50` always grabs the
FULL p2 pool regardless of dataset size (verified in code, not assumed):
`automl_package/examples/width_wsel24.py:92-95` (`PROBE_FRACTION_PCT = 50` docstring) and the split
construction itself, `automl_package/examples/width_wsel9.py:290-304` (`n_pool`, `p2_idx`,
`n_select = max(1, min(round((fraction_pct/100.0) * n_pool), len(x_p2)))`). **50% is therefore not
an arbitrary upper anchor — it is the largest SELECT carve reachable without changing the
FIT/STOP/SELECT split architecture itself**, which is out of scope here (see §7).
**Energy's `selection_set_size` label is inferred by analogy** (same n-clustering, worst seed
variance of the five) — its own probe cell was inside the cap but deliberately unspent
(`width.md:1559-1560`). §4's grid retrains energy as part of the systematic ladder, not a one-off
probe, so it settles this causally as a side effect (§5, Outcome D).

**The binding capture gap (WSEL-22(a)'s catch).** The per-input (dial-router) arm's SELECT-split
per-sample error table was never cached to disk for ANY dataset — verified directly, not assumed:
every `<dataset>_<seed>_dial.json` cell (read `automl_package/examples/capacity_ladder_results/WSEL9/yacht_0_dial.json`)
stores only the DERIVED outputs (`w_shared.selected_width`, `w_perinput.mean_routed_width`,
`width_distribution`, `held_out_mse`) — the `(n_selection_used, w_max)` per-sample error table that
`fit_router`/`fit_global_selector` build internally
(`automl_package/models/flexnn/width/model.py:450`, `error_table = np.stack([self._per_sample_error_at_width(x_arr, y_val_arr, width) for width in self.widths], axis=1)`)
is computed in memory and discarded. `width.md:1352-1365` names this a ROOT-GRADE CATCH and records
the consequence explicitly at WSEL-6-R: **whenever per-input cells are next regenerated, this table
MUST be cached at run time.** §4/§6 below are this spec's concrete answer to that requirement.

**MASTER Decision 36 (the binding success criterion).** The fielded GLOBAL width-selection rule is
graded OBJECTIVE-MATCHED with DUAL REPORTING: every battery table reports both the
smallest-sufficient pick and the accuracy-optimal pick, side by side, with accuracy and compute cost,
both read off the SAME recorded held-out curve at zero retraining
(`docs/plans/capacity_programme/MASTER.md:654-668`). Default when no objective is stated:
smallest-sufficient. This is mechanically trivial to satisfy once an error table exists — both picks
are two different reads of `cheapest_within_tolerance`'s own inputs
(`automl_package/utils/capacity_selection.py:71-72`, `best_idx = int(np.argmin(column_means))` is the
accuracy-optimal pick; the function's return value is the smallest-sufficient pick) — so "dual
reporting" costs nothing beyond building the error table once. §3 below is this spec's binding
success criterion, verbatim from Decision 36.

## 1. Question

What SELECT-set fraction (equivalently, absolute SELECT-set size) does the fielded GLOBAL
width-selection rule (`cheapest_within_tolerance` reading a per-width held-out error curve) need on
each of the five real battery datasets — yacht, diabetes, energy, kin8nm, california — before its
smallest-sufficient pick matches the accuracy-optimal pick within the noise-aware 2·SE bar (MASTER
Decision 33(i), `docs/plans/capacity_programme/MASTER.md:594-596` — no flat percentage bars)? The
frozen 15% is causally demonstrated underpowered on the three small datasets at n ≤ ~100 (§0); this
task finds where (if anywhere, within the 50%-ceiling split architecture) it stops being underpowered.

## 2. Grid design

### 2.1 The nested-prefix insight (load-bearing; flagged for adversarial read in §8)

`_build_split` fixes `x_fit`/`x_stop` (the training data) independent of `fraction_pct` — only the
SELECT slice `x_select = x_p2[:n_select]` changes, and `x_p2`'s row order is fixed by a permutation
seeded on `seed` alone, not on `fraction_pct`
(`automl_package/examples/width_wsel9.py:292-304`). Two consequences, both already verified by
WSEL-24's own probe (`prefix_consistent: True` on both probe cells, same
`attribution.json` fields cited in §0):

1. **SELECT sets at different fractions are exact prefixes of one another** (smaller fraction's
   rows are the first N rows of the larger fraction's rows, same order). A per-sample error TABLE
   captured once at the ladder's maximum fraction (50%, the hard ceiling) answers every smaller
   fraction in the ladder by slicing rows — no retraining, no rescoring beyond the one pass already
   taken.
2. **Training is fraction-invariant.** Since `_run_w_sweep_cell` and `_run_dial_cell` both train on
   `x_fit`/`x_stop` only (`width_wsel9.py:479-525`, `:528-598`) and both call `torch.manual_seed(seed)`
   deterministically before training (`width_wsel9.py:397`), a retrain at `fraction_pct=50` on CPU
   (this battery's own recorded provenance is `automl_device: cpu` on every existing cell —
   `automl_package/examples/capacity_ladder_results/WSEL9/yacht_0_dial.json` `provenance.automl_device`)
   is expected to reproduce the IDENTICAL trained net as the already-cached `fraction_pct=15` cell for
   the same (dataset, seed, width) — same `held_out_mse`, same `actual_epochs`, same trajectory.

**Design decision: one retrain per (dataset, seed, width) at the 50% ceiling, not one retrain per
(dataset, seed, width, fraction).** The full SELECT-split error array/table is cached at that single
retrain; the fraction ladder is read off it at `--summarize` time by prefix-slicing, at zero
additional compute. This departs from a literal reading of "per-cell CLI
(--dataset/--seed/--fraction/--arm)" as one training cell per fraction (WSEL-6's own toy design,
`automl_package/examples/capacity_ladder_results/WSEL6/frozen.json`'s 90 cells = 2 tiers × 3 seeds × 5 fractions × 3 arms) — flagged explicitly in
§8 for the root's adversarial read, because it changes the shape of the grid the orchestrator
described. The `--fraction-pct` CLI flag is retained (§6) so a specific smaller fraction CAN still be
trained directly, both for the faithfulness check below and as an escape hatch if §2.1's
determinism assumption fails.

**Pre-registered faithfulness check (cheap, decisive, self-correcting):** the retrained
`fraction_pct=50` cell's `held_out_mse` for a given (dataset, seed, width) must equal the ALREADY-
LANDED WSEL9 cell's `held_out_mse` at the same (dataset, seed, width)
(`automl_package/examples/capacity_ladder_results/WSEL9/<dataset>_<seed>_w_sweep_<width>.json`
<!-- citecheck-ignore: placeholder path with `<...>` fields naming the existing per-cell naming convention, not a citation -->,
within float tolerance). If it does NOT match on a spot check (do this on the first landed cell of
each dataset before running the rest), §2.1's design is invalidated for that dataset and the grid
falls back to one retrain per fraction for it — a mechanical, checkable failure branch, not a
judgment call.

### 2.2 Fraction ladder

`{5, 10, 15, 20, 25, 30, 35, 40, 45, 50}` (percent). 15 is the current frozen anchor (causally
shown insufficient at this value, §0); 50 is the hard ceiling of the current split architecture
(§0, §2.1). **The two rungs BELOW the frozen anchor (5, 10) are included by root amendment
(adversarial read, 2026-07-23): they are zero-cost prefix reads off the same cached table, and
together with 15/25/40 they reproduce WSEL-6's toy ladder EXACTLY
(`fractions_swept_pct: [5, 10, 15, 25, 40]`,
`automl_package/examples/capacity_ladder_results/WSEL6/frozen.json`) — making the toys↔real
fraction-response comparison direct at matched rungs, and charting the starvation side of the
curve the report will want.** Because §2.1 makes every rung at or below the trained maximum free
at `--summarize` time, this specific 10-point list is a REPORTING-resolution choice, not a
compute-driven one — a finer ladder can be added later by re-running `--summarize` alone, no new
training.

### 2.3 Seeds, widths, datasets, arms

- **Seeds:** `{0, 1, 2}` — the same three seeds WSEL-9/WSEL-24 already used, so every new cell is
  directly comparable to the existing record, not a fresh sample.
- **Widths:** `1..12` (`w_max = 12` for every one of the five datasets — verified,
  `automl_package/examples/capacity_ladder_results/WSEL9/<dataset>_0_dial.json` `w_max` field, read
  for all five datasets and identical).
- **Datasets:** all five — yacht, diabetes, energy, kin8nm, california. kin8nm/california are
  pre-registered as a NEGATIVE CONTROL (§5, Outcome C): their SE is already well-estimated at
  n = 983 / 2477 (`rule_objective_mismatch`, not `selection_set_size` — §0), so the fraction ladder
  is expected to move their picks little to none; a large shift on either would be a genuine
  surprise worth an adversarial re-read of their WSEL-24 label.
- **Arms in scope:**
  1. **W-SWEEP (global rule, primary — the arm WSEL-24's causal probe actually used).** Retrain
     each of the 12 dedicated per-width nets per (dataset, seed) at `fraction_pct=50`
     (`_run_w_sweep_cell`, reused verbatim per §6), caching the FULL `select_squared_error` array.
     `5 × 3 × 12 = 180` retrains.
  2. **DIAL / per-input (`w_shared` + `w_perinput`) — IN SCOPE, justified below.** Retrain the
     jointly-trained multi-head dial net per (dataset, seed) at `fraction_pct=50`
     (`_run_dial_cell`, reused verbatim per §6), this time capturing the raw
     `(n_selection_used, w_max)` error table (§0's capture gap) alongside the existing derived
     outputs. `5 × 3 = 15` retrains.
  - **Justification for including the dial/per-input arm now, rather than deferring it again:**
    (a) the capture requirement is BINDING for this spec regardless (`width.md:694-697`); (b) §2.1's
    determinism argument applies identically to the dial net (same `_build_split` invariance), so
    the marginal cost is 15 retrains, not 195 — small relative to the sweep arm; (c) deferring again
    would leave the per-input arm's real-data fraction-sensitivity permanently unanswerable, exactly
    the gap WSEL-22(a) flagged. **Scope boundary, so this inclusion does not quietly expand the
    task:** the dial/per-input arm's own labelling rule (WSEL-22's tolerance-band mechanism, a
    DIFFERENT selector from `cheapest_within_tolerance` — MASTER Decision 18) is NOT graded by this
    task's §3 success criterion, which is Decision 36's GLOBAL-rule criterion only. Capturing the
    table and reporting the per-input pick's behavior across the ladder is DESCRIPTIVE here; grading
    it against a success bar is WSEL-22's business, not re-litigated by this task. Flagged in §8 for
    confirmation.
- **Reused from cache, zero retraining:** `plain_nn`, `lightgbm`, `linear_reg` control cells for all
  five datasets — verified by reading `_run_plain_nn_cell`
  (`automl_package/examples/width_wsel9.py:632-663`), which trains and scores only on
  `x_fit`/`x_stop`/`x_test`, never touching `x_select`; `fraction_pct` cannot affect these cells'
  output, so the already-landed WSEL9 cells are reused verbatim (`45` cells: `3 seeds × 3 arms ×
  5 datasets`, no compute).

### 2.4 Compute estimate

`180 (sweep) + 15 (dial) = 195` NN retrains, reusing `45` existing control cells at zero cost. By
§2.1's determinism argument this is expected to cost the SAME wall-clock as WSEL-9's original
sweep+dial pass already did (same architectures, same `x_fit`/`x_stop`, same early-stopping
trajectories — only the scored SELECT slice differs) — a known, already-amortized-once quantity,
not a novel unknown. <!-- numcheck-ignore: no wall-clock was recorded in any WSEL9 cell (only
`actual_epochs`/`training_macs`), so this is a qualitative equal-to-precedent claim, not a
ledger-sourced figure -->
`--summarize`'s per-fraction dual-pick computation (§1, §3) is then two `numpy` reads
(`argmin`, `cheapest_within_tolerance`) per (dataset, seed, fraction) off the cached arrays — no
additional training or scoring, so the 8-point ladder (§2.2), or any finer one, is free at this
stage.

## 3. Success criterion (MASTER Decision 36, binding — restated precisely for this task)

At every rung of the fraction ladder (§2.2), for the W-SWEEP arm (§2.3.1), on every dataset:

1. Build the `(n_selection_at_fraction, 12)` error table by slicing the cached `fraction=50` array
   to the fraction's prefix length (§2.1).
2. Report BOTH picks from that table, same object, zero retraining:
   - **accuracy-optimal** = `argmin` of the column means (`automl_package/utils/capacity_selection.py:71-72`).
   - **smallest-sufficient** = `cheapest_within_tolerance(error_table, n_boot=1000, seed=0)`
     (`DEFAULT_N_BOOT`/`DEFAULT_SEED`, same file, `:35-36`) — the SAME function, SAME 2·SE
     multiplier (`TOLERANCE_SE_MULTIPLE = 2.0`, `:37`), SAME bootstrap estimator WSEL-24's probe
     held fixed.
3. Both picks' held-out TEST accuracy and compute cost are reported side by side (never one without
   the other — this is the "dual reporting" half of Decision 36).
4. **"Selection works at fraction f" (per dataset, per objective) — mechanical definition (root
   amendment, adversarial read 2026-07-23; the original draft's "TEST errors within the same bar
   already used to build the picks" mixed a SELECT-side SE with TEST-side errors and is
   superseded):** the fielded pick at f (smallest-sufficient = `cheapest_within_tolerance` on the
   f-prefix table; accuracy-optimal = `argmin` of its column means) either
   - **matches the TEST-oracle-best width's identity** (the width minimizing recorded TEST mse at
     that (dataset, seed) — the `width_hi_comparator` semantics of WSEL-24's probe), OR
   - **passes TEST-side paired validation**: mean per-sample TEST squared-error difference
     (pick − TEST-oracle-best) ≤ 2 × the bootstrap SE of that difference — same estimator and
     constants as the rule itself (`bootstrap_se`, `n_boot=1000`, `seed=0`,
     `automl_package/utils/capacity_selection.py:35-37`; MASTER Decision 33(i) — no flat
     percentage substitute). Requires the per-sample TEST squared-error capture (§4).
   **Continuity readout (reported, never gating):** the SELECT-side within-tolerance flag between
   the fielded pick and the TEST-oracle-best at f — the exact computation WSEL-24's probe recorded
   (`orig`/`probe` → `mean_diff`/`se`/`within_tolerance`,
   `automl_package/examples/capacity_ladder_results/WSEL9/attribution.json`) — so this grid's
   numbers sit directly beside WSEL-24's probe rows.
5. **Default objective when none is stated: smallest-sufficient** (Decision 36) — the fraction
   ladder's headline number per dataset is the smallest f at which check 4 passes and stays passing
   at every larger f in the ladder (no re-flipping), analogous to WSEL-6's own `saturated: true`
   semantics (`automl_package/examples/capacity_ladder_results/WSEL6/frozen.json`).

This criterion is unchanged from what Decision 36 already ruled; nothing here is a new policy
decision — it is Decision 36 made mechanically checkable against this specific grid's outputs.

## 4. The binding capture requirement (file contract)

Whenever a dial/per-input cell is (re)generated by this task's eventual driver, in addition to the
already-existing derived fields (`w_shared`, `w_perinput`, `shared_training`, unchanged in shape),
the per-cell JSON gains one new top-level key:

```json
"select_error_table": {
  "widths": [1, 2, 3, "...", 12],
  "n_selection_used": "<int, length of the row axis at capture time (the fraction_pct this cell trained at)>",
  "fraction_pct_captured_at": "<int, matches config.fraction_pct on the same cell>",
  "table": "[[float, ...] * 12] * n_selection_used  -- row i, col j = _per_sample_error_at_width(x_select[i:i+1], y_select_std[i:i+1], widths[j]), i.e. exactly the array `fit_global_selector`/`fit_router` already build in memory (automl_package/models/flexnn/width/model.py:450) and currently discard"
}
```

- **What gets written:** the `(n_selection_used, 12)` per-sample squared-error table, one row per
  SELECT-split example, one column per width, in the SAME row order `_build_split` produces for
  `x_select` (deterministic given `dataset`/`seed`/`fraction_pct` — no separate index array is
  needed, since re-deriving the split from the same three arguments reproduces the same order).
- **Where (ROOT AMENDMENT, adversarial read 2026-07-23 — two corrections to the original draft):**
  1. **This task's cells land under a NEW results dir,
     `automl_package/examples/capacity_ladder_results/WSEL6R/` <!-- citecheck-ignore: forward reference -- created by the grid run -->,
     with the same per-cell naming convention. `WSEL9/` is a READ-ONLY historical record for this
     task** — the faithfulness check (§2.1) reads its landed cells and the driver must refuse to
     write into it. (The original draft's "embedded in the existing `<dataset>_<seed>_dial.json`
     cell" + §6's "same defaults" would have pointed the driver at `WSEL9/` and overwritten the
     committed record.)
  2. **Raw per-sample arrays go to a git-ignored sidecar, never into the committed cell.** The
     `WSEL9/` per-cell JSONs are committed (251 files, 22 MB); at the 50% ceiling the california
     dial table is ≈8k rows × 12 widths and each california sweep cell's select array ≈8k floats —
     embedding them repeats the recorded case-law mistake (an 11 MB calibration JSON with embedded
     trajectories, slimmed to 12 KB; big arrays live in git-ignored cache metas). Contract: each
     cell's raw arrays (`select_error_table.table`, the 50%-ceiling `select_squared_error`, and
     §3's per-sample TEST squared errors) are written to
     `WSEL6R/_cache/<dataset>_<seed>_<arm>[_<width>]_arrays.npz` <!-- citecheck-ignore: forward reference -- naming template for files the grid run creates -->
     the moment the cell completes; the committed per-cell JSON carries the sidecar's relative
     path, array shapes, a SHA-256 checksum, and the per-width column MEANS (the reviewably-small
     summary). `--summarize` consumes the sidecars; a fresh clone without them must retrain to
     re-read — the same accepted consequence the programme already records for state dicts.
- **Scale note (explicit so consumers don't mis-join):** `select_squared_error` and the dial table
  are on the STANDARDIZED-y scale (`automl_package/examples/width_wsel9.py:499-500` computes them
  from `y_select_std`); `held_out_mse` and §3's per-sample TEST squared errors are RAW-y scale
  (`width_wsel9.py:495-497`). Each sidecar array records its scale in an adjacent JSON field.
- **Format:** `npz` arrays in the sidecar; the committed cell keeps plain JSON (pointer + shapes +
  checksum + means), matching the committed-artifacts-stay-reviewably-small case law.
- **When captured:** at `fraction_pct=50` (the ceiling, §2.1) — captured ONCE per (dataset, seed);
  every smaller fraction in the ladder reads a row-prefix of this same table, exactly as for the
  sweep arm.
- **Additional capture (root amendment, adversarial read 2026-07-23 — feeds §3 check 4): every
  retrained W-SWEEP cell also lands its per-sample TEST squared errors** (raw-y scale, one value
  per TEST row, computed at the same `model.predict` call that already produces `held_out_mse` —
  `automl_package/examples/width_wsel9.py:495-497`) in the cell's sidecar. This is what makes the
  TEST-side paired validation between the fielded pick and the TEST-oracle-best width computable
  at `--summarize` time without reloading any net. Dial cells capture the analogous per-sample
  TEST arrays where the existing cell code already computes the predictions (descriptive, §2.3's
  scope boundary).
- **Consumers:** `--summarize` (§6) uses this table to compute BOTH `fit_global_selector`'s pick
  (mirroring `w_shared`) and a `fit_router`-equivalent per-input labelling at every fraction in the
  ladder, descriptively (§2.3's scope boundary — not graded against §3's bar).

This closes `width.md:1363-1365`'s named consequence exactly as instructed, and is the FIRST time
this table exists on disk for any dataset.

## 5. Pre-registered outcomes and failure branches

- **Outcome A (success, per dataset).** §3's check passes at some `f* ≤ 50` for diabetes/yacht/
  energy. Done-state: WSEL-6's real-data half becomes ANSWERED with a per-dataset (or, if they
  converge, a single shared) `f*`; the candidate new frozen constant is reported alongside the
  existing 15% for comparison, never silently overwritten — a follow-on task adopts it if the user
  ratifies.
- **Outcome B (failure, checkable, no new ruling needed).** §3's check still FAILS at `f = 50` (the
  architecture's ceiling) on one or more of diabetes/yacht/energy. Done-state: recorded as a negative
  result, filed the same way WSEL-8's negative result was (recorded, not blocking) — "this dataset's
  selection gap is not closeable by enlarging the SELECT fraction within the current 50/50 split
  architecture; Decision 36's dual-reporting is the durable mitigation, not a stopgap for this
  dataset." Changing the split architecture itself (beyond 50%) is explicitly OUT of this task's
  scope (§7) — a distinct, larger task if ever pursued.
- **Outcome C (negative control, pre-registered).** kin8nm/california's picks move little or not at
  all across the ladder (consistent with `rule_objective_mismatch`, not `selection_set_size` — §0).
  If either shows a LARGE shift instead, that is a surprise: flag for adversarial re-read of its
  WSEL-24 label rather than silently reinterpreting it here (non-goal, §7).
- **Outcome D.** Energy's `selection_set_size` label (currently inferred by analogy, §0) is settled
  causally as a byproduct of being retrained inside the systematic ladder rather than as a one-off
  probe — no separate action needed to close this WSEL-24 open item.
- **Outcome E (capture requirement, descriptive, ungated).** The per-input arm's fraction-sensitivity
  becomes readable for the first time on every dataset (§4); reported alongside §3's results but not
  graded by §3's bar (§2.3's scope boundary). Feeds a later WSEL-22 task if the user wants the
  per-input rule itself re-examined.
- **Failure branch on the EFFICIENCY DESIGN itself (§2.1), distinct from the scientific outcomes
  above:** the pre-registered faithfulness check (§2.1) fails on the first cell of some dataset. Done-
  state: that dataset's grid falls back to one retrain per (fraction, width, seed) instead of one
  retrain per (width, seed) — a mechanical, pre-declared fallback, not a new judgment call or a
  reason to halt the rest of the grid.

None of these branches requires a new user ruling to reach a checkable done-state (no invented gates
— every branch above is a terminal, reportable outcome).

## 6. Driver contract

A later task builds `automl_package/examples/width_wsel6r.py` <!-- citecheck-ignore: forward reference -- Create target of a later task, not yet built by this spec-authoring task --> per this
contract. **The driver's author never runs the full 195-cell grid — the ROOT runs it
`Bash(run_in_background: true)`, per the standing dispatch discipline.**

- **Per-cell CLI**, mirroring `width_wsel9.py`'s existing flag shape
  (`automl_package/examples/width_wsel9.py:996-1014`) so the two drivers stay recognizable siblings:
  - `--dataset {yacht,diabetes,energy,kin8nm,california}` (required outside `--selftest`/`--summarize`)
  - `--seed {0,1,2}` (required outside `--selftest`/`--summarize`)
  - `--arm {w_sweep,dial}` (required outside `--selftest`/`--summarize`)
  - `--width <int 1..12>` (required iff `--arm w_sweep`, forbidden otherwise — same convention as
    `width_wsel9.py`'s `--width`)
  - `--fraction-pct <int>` (default: `50`, the ceiling; override to train a specific smaller
    fraction directly — used only by the §2.1 faithfulness check or as its fallback escape hatch)
  - `--results-dir` — **default `automl_package/examples/capacity_ladder_results/WSEL6R/`
    <!-- citecheck-ignore: forward reference -- created by the grid run --> (root amendment, §4):
    the driver REFUSES a results dir that resolves to the landed `WSEL9/` record (hard error, not
    a warning) — `WSEL9/` is read-only to this task**
  - `--wsel7-path`, `--wsel8-dir` (reused verbatim from `width_wsel9.py`'s own flags, same
    defaults, so the router/width-ladder constants are sourced identically — this task does not
    touch either constant, §7)
  - `--epoch-cap`, `--test-fraction`, `--max-train`, `--tag` (reused verbatim from `width_wsel9.py`)
- **One JSON per cell + its sidecar `.npz` (§4), both written the moment the cell is produced** —
  never held in memory until a batch completes (standing clause). Sweep cells land their
  50%-ceiling `select_squared_error` and per-sample TEST arrays in the sidecar (committed JSON
  keeps pointer + shapes + checksum + means, §4); dial cells add `select_error_table` per §4, same
  sidecar split.
- **`--summarize`** aggregate mode: for each fraction in the ladder (§2.2, default
  `{15,20,25,30,35,40,45,50}`, overridable), for each dataset, slice every landed cell's cached array
  to that fraction's prefix length, compute §3's dual picks (W-SWEEP) and §4's descriptive per-input
  readout (dial), cross-check §5 Outcome C's negative control, and write
  `automl_package/examples/capacity_ladder_results/WSEL6R/frozen.json` <!-- citecheck-ignore: forward reference -- Create target of a later task's grid run, not yet built by this spec-authoring task -->.
- **`--selftest`**: synthetic in-memory check (no real dataset, no real training — mirrors
  `width_wsel9.py`'s/`width_wsel24.py`'s own `--selftest` pattern), asserting at minimum:
  1. Prefix-slicing a synthetic `(n_max, 12)` error table to a smaller `n` and calling
     `cheapest_within_tolerance`/`argmin` on the SLICE reproduces the SAME pick as building the table
     directly at that smaller `n` from the same underlying per-sample errors (the core §2.1
     assumption, testable without real data or real training).
  2. The dual-pick output always carries both objectives' identity, accuracy, and cost (Decision 36
     shape check).
  3. `select_error_table` round-trips through the sidecar write/read (`npz`) with the same shape
     and values it was constructed with, and the committed cell's recorded checksum matches the
     sidecar on disk (§4's file-contract check, both halves).
  4. A fraction ladder point outside `[5, 50]` is rejected (keeps the ladder inside the split
     architecture's valid range, §2.1/§2.2).
  5. A `--results-dir` resolving to the landed `WSEL9/` record is rejected with a hard error
     (§4's read-only guarantee).

## 7. Non-goals

- No changes to the GLOBAL selection rule, its 2·SE multiplier, or its bootstrap estimator (WSEL-22
  owns those; this task only measures where the EXISTING rule needs more data, per WSEL-24's own
  non-goals it inherits).
- No changes to the per-input labelling tolerance or router-fitting logic (WSEL-22 owns those; §4/§6
  only capture data these mechanisms already compute internally).
- No changes to `_build_split`'s FIT/STOP/SELECT split architecture itself — the 50% ceiling (§0,
  §2.1) is accepted as a hard boundary of this task, not something it relaxes. Reopening the p1/p2
  halving itself (e.g. to reach above 50%) is a distinct, larger task if Outcome B (§5) is ever hit
  on every dataset.
- No new datasets beyond the five already in the WSEL-9 battery.
- No learned variances, no architecture changes to any of the three model families (W-SWEEP nets,
  the dial multi-head net, the plain/lightgbm/linear controls).
- No re-grading of kin8nm/california's `rule_objective_mismatch` label (Outcome C, §5, is a
  consistency check, not a re-litigation).
- No report prose (a later WSEL-10-consuming task owns that, per the programme's existing division).
- The driver's author does not run the full grid (standing clause, §6).

## 8. Open questions flagged for the root's adversarial read — ✅ ALL RESOLVED (root, 2026-07-23; verdict in §9)

1. **§2.1's single-retrain-at-the-ceiling design** departs from a literal one-cell-per-fraction
   reading of the orchestrator's own brief phrasing. The evidence for it (fraction-invariant
   training, prefix-consistent SELECT sets, both verified against WSEL-24's own probe fields) is
   laid out in full in §0/§2.1; flagging in case the root prefers the literal (more expensive, more
   conservative) per-fraction design instead, e.g. as a hedge against §2.1's determinism assumption
   turning out to have an escape hatch I haven't found (CPU non-determinism in some op, a hidden RNG
   draw inside `_train_flexwidth_single`/`_train_dial_to_convergence` not covered by the single
   `torch.manual_seed(seed)` call at `width_wsel9.py:397`). The pre-registered faithfulness check
   (§2.1, §5) is the proposed mitigation rather than a full literal-design fallback.
2. **Including the dial/per-input arm in this task's scope** (§2.3) rather than deferring it again.
   Justified on bounded marginal cost and the binding capture requirement, with an explicit scope
   boundary keeping its OWN success grading out of §3. Flagging because it is a scope call, not a
   mechanical derivation.
3. **The 8-point fraction ladder** (§2.2) is a reporting-resolution choice enabled by §2.1's
   zero-marginal-cost property — flagging in case a coarser or finer list is preferred; it costs
   nothing to change post-hoc once §2.1's design is accepted.
4. **Energy's probe cell being folded into the systematic grid** rather than spent as a dedicated
   one-off (§0, §5 Outcome D) — flagging as a design choice, not asking for a new probe-cap
   authorization (the systematic grid subsumes what a probe would have shown).

**Resolutions (root):** (1) **APPROVED** — the prefix property, the fit-only standardization stats
(`width_wsel9.py:306-315`), the seed-only `perm2` (`:301`), and fit/stop-only training were each
re-verified at source by the root, and the faithfulness check is BINDING as the FIRST landed cell
of each dataset (grid ordering constraint on the run, not a fallback afterthought). (2) **APPROVED**
with the stated scope boundary. (3) **AMENDED** — ladder extended down to {5, 10} (§2.2, free
prefix reads, exact toys-ladder reproduction). (4) **APPROVED**.

## 9. Adversarial-read verdict (root, 2026-07-23) — **GO, with four root amendments applied above**

Every load-bearing citation was re-verified against source by the root before this verdict:
`_build_split`'s prefix/invariance structure (`automl_package/examples/width_wsel9.py:273-336`),
the probe constant and its docstring (`automl_package/examples/width_wsel24.py:92-95`), the
discarded in-memory error table (`automl_package/models/flexnn/width/model.py:450`), the rule's
constants and argmin line (`automl_package/utils/capacity_selection.py:35-37`, `:71-72`), the
attribution ledger's per-dataset fields and both probe sub-objects
(`automl_package/examples/capacity_ladder_results/WSEL9/attribution.json`), and cell shape/commit
status of the WSEL9 record (251 committed files; sweep-cell keys; dial cells lacking any error
table). The four amendments: (i) results-dir isolation — `WSEL6R/` new, `WSEL9/` read-only
(original draft would have overwritten the committed record); (ii) raw arrays to git-ignored
sidecars, committed cells stay reviewably small (case-law precedent); (iii) §3 check 4 pinned to
the TEST-side paired 2·SE computation + WSEL-24 continuity readout (draft phrasing mixed a
SELECT-side SE with TEST-side errors); (iv) ladder extended to {5, 10}. Next step per the wave
line: the driver task builds `automl_package/examples/width_wsel6r.py` against §6 <!-- citecheck-ignore: forward reference -- Create target of the driver task -->; the ROOT runs
the 195-cell grid backgrounded.
