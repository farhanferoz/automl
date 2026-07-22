# Width wave-2 close — the MASTER Decision-32 checklist, discharged with evidence

Date: 2026-07-22. Scope: every width-strand task the wave claims done. The task blocks in
`docs/plans/capacity_programme/width.md` remain the authoritative ledgers; this file is the
wave-close audit index — each row points at the ledger block or artifact that carries the numbers.

## Item 1 — every task's verify clause, re-run against DISK at close

Sweep script + raw output: session scratchpad (`wave2_verify_recheck.sh` / `.out`); every line
re-derived from disk on 2026-07-22, not from task-list status.

| task | verify re-check | result |
|---|---|---|
| WSEL-3 cheap global read | shared-selector primitive call + `tests/test_flexible_width_network.py` | PASS (with one DRIFT, below) |
| WSEL-4 sweep reference | 2% reproduction bar, 36/36 cells; ported cells carry trajectory + `hit_cap: false` | PASS |
| WSEL-5 selection costs | shim import resolves; `tests/test_capacity_accounting.py` | PASS |
| WSEL-6 data fraction | frozen constants + 90 cells + saturation plot; re-run cells cite the router ledger | PASS |
| WSEL-7 router architecture | frozen keys + `invariant` field + non-null `new_default` on the False branch | PASS |
| WSEL-8 dial vs sweep | control `reproduces: true`; per-seed cells carry the full output contract | PASS |
| WSEL-11 regularisation | 9/9 re-run cells + `selection_moved` (in `WSEL11/rerun/` — the top level retains the VOID first run as case law) | PASS |
| WSEL-13 ordering | driver selftest; frozen step-4 fields; all 6 cells parse | PASS |
| WSEL-16 architecture comparison | stopgrad test; driver selftest; frozen trio (`controls_passed`/`stage1_winner`/`stage2_required`); no top-level `hit_cap` across 41 JSONs | PASS |
| WSEL-18 vectorisation | fused-equivalence test; `bench.json` carries both arms' `mean_step_wall_clock_ms` | PASS |
| WSEL-17 Steps 1-6 | equivalence test (with prove-it-fails); manifest present (41 KEEP / 2 SHIM / 0 DELETE-ELIGIBLE) | PASS |

**DRIFT, recorded:** WSEL-3's verify clause greps `automl_package/models/flexible_width_network.py`,
which the later module reorganisation turned into a shim; the shared-primitive call is confirmed at
the real home `automl_package/models/flexnn/width/model.py`. Same drift family as WSEL-17's
write-set amendment (its task block, 2026-07-22 note).

**Two first-pass check corrections, named so the audit is honest:** the close sweep's own first pass
mis-read WSEL-11 (checked the voided top-level run instead of `rerun/`) and WSEL-18 (guessed a
`per_step` field name); both re-checked correctly same turn against the real layout/field names.

## Item 2 — every pre-registered prediction/bar, swept and reconciled

Rules quoted verbatim from the registering spec; measured values live in the pointed-at ledger.

1. **WSEL-13 primary** — "Spearman correlation between index and ablation importance `<= -0.5` on
   at least 2 of 3 seeds" → **FAIL 0/3, correlation runs the opposite way** (tier-1 table, WSEL-13
   result block). Graded at landing, unchanged at close.
2. **WSEL-13 secondary** — "mean over `k` of `(prefix_k - greedy_k) / greedy_k` `<= 0.10`" →
   **FAIL** (mean 0.310, same block). `ordering_holds: false` stands. <!-- numcheck-ignore: 0.310 restates the tier-1 secondary-bar mean already sourced in the WSEL-13 result block (`automl_package/examples/capacity_ladder_results/WSEL13/frozen.json`) -->
3. **WSEL-13 tier-2** — pre-registered as corroboration only ("may NOT be used to re-read, rescue,
   or override the tier-1 bars") → honored; no bars computed there (tier-2 block).
4. **WSEL-4** — "relative error ≤ 2% on `W_CONVERGED`'s reported MSE, per (toy, seed, width) cell"
   → **PASS 36/36** under the logged anchor correction (the reference records log-likelihood, not
   MSE; WSEL-4 result block).
5. **WSEL-16 primary** — "`A_STOPGRAD`'s full-width held-out MSE within 10% of `B_HEADS`', on
   tier 1 and tier 2, on all 3 seeds" → graded at landing; `stage1_winner: b_heads`,
   `stage2_required: true`, stage-2 winner NONE (WSEL-16 block +
   `automl_package/examples/capacity_ladder_results/WSEL16/frozen.json`).
6. **WSEL-7 invariance rule** — "INVARIANT iff the frozen default's own ratio is within
   `_PLATEAU_REL_TOL` (5%) of the best ratio achieved anywhere in the sweep" → **NOT invariant**
   under the registered rule; recorded WITH the noise caveats; the discriminating re-run showed no
   end-task gain; **user ruling at sign-off: frozen default stays** (WSEL-7 block).
7. **WSEL-6 rule** — "the smallest fraction at which every arm is within its own noise band of its
   best (twice-standard-error rule); if none saturates, take the largest swept and record the study
   inconclusive" → every (tier, arm) pair saturates; fraction 0.15 frozen (WSEL-6 block).
8. **WSEL-6 branch** — "W-PERINPUT still improving at the largest fraction → mark router
   data-limited" → not triggered; every `data_limited` flag false (same block).
9. **WSEL-8 gate (MASTER Decision 14)** — "the known-good arm runs first, alone; here that is
   W-SWEEP reproducing WSEL-4's control before any W-SHARED number is read" → enforced
   mechanically by the driver; control exact (WSEL-8 block).
10. **WSEL-8 claim under test** — §2's "(b) same choice" plus quality-at-matched-width → **BOTH
    HALVES FAIL** (0/3 agreement, dial always wider; matched-width premium up to 7.2×; WSEL-8
    block). §2's row corrected in the same turn.
11. **WSEL-11 question** — does explicit regularisation move the selected width →
    `selection_moved: false` at the corrected objective (WSEL-11 ✅ block).
12. **WSEL-18 premise (Decision 31's cost clause)** — fusion removes the dispatch premium →
    **HALF-TRUE**: dispatch premium gone, a ~21% per-step arithmetic (coverage) premium vs
    sandwich remains (WSEL-18 block;
    `automl_package/examples/capacity_ladder_results/WSEL18/bench.json`). Batched for user review.
13. **§2 exposure prediction** — "W-PERINPUT must learn a function from x to width and should be
    hungriest" → confirmed: it is the binding arm for the frozen fraction (WSEL-6 block).

## Item 3 — gate/halt outcomes carry their rule verbatim

Every graded outcome above quotes its registering rule beside the verdict; the ledger blocks carry
the measured values beside the quoted bars (spot-audited at close: WSEL-13's both bars, WSEL-7's
rule, WSEL-4's bar, WSEL-8's gate — all quoted verbatim in their result blocks).

## Items 4-7 — standing craft, confirmed over the wave

- Grid launches direct + verified started + notified; every cell landed to disk as produced
  (three studies' lanes plus repairs; zero lost cells).
- Gate→commit chains conditioned on the test's own exit code throughout (one historical
  pipeline-tail incident predates this close and is retained as case law in RESUME).
- Worker briefs carried the standing clauses; both wave workers reported per contract and were
  verified against disk before their reports were acted on.
- Status reported by research approach in the ratified names throughout the attended review.

## Wave-end full test suite

`pytest tests/` at close: **480 passed, 2 failed, 1 skipped** — the two failures are exactly the
pre-existing ACCEPTED pair (`test_probabilistic_nll_beats_constant_on_heteroscedastic`,
`test_prob_regression_heteroscedastic_mse`), the skip is the baseline one, and passes grew from
the 431 baseline by this wave's new tests. **No regressions.**

## Deviations & preserved-evidence registry (this wave)

- Superseded/confounded runs preserved, never tabulated: `WSEL4/relu_confounded_run/`,
  `WSEL4/tanh_minibatch_run/`, `WSEL6/capped_at_6000/` (+ `_cache/capped_at_6000/`),
  `WSEL8/capped_at_6000/` (+ cache twin), `WSEL11/` top level (VOID first run).
- The joint-training stop-rule defect (latched done → non-stationary stop) is FIXED in the
  ordering driver and mirrored in the verdict driver; **the same latent pattern remains in
  `kdropout_converged_width_experiment.py:322` by choice** (not edited mid-wave; flagged for a
  follow-up outside this wave).
- Path drifts from the module reorganisation recorded at WSEL-3 (verify clause) and WSEL-17
  (write set, amended in-block).

## Still open at close (attended)

Sign-off items 2-6 of the end-of-run review (vectorisation cost clause · stage-3 coverage ·
sweep-reference verify corrections · easy-tier ordering state-dict commit · cleanup manifest +
near-miss deletion candidates + the headstone finding on the zero-caller inventory's location);
then the local merge to master per protocol. The deletion pass (WSEL-17 Step 7) runs attended
only, from the signed-off manifest.
