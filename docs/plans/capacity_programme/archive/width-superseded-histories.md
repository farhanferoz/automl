# Width strand — superseded histories & retained pre-registrations (archive)

**Created 2026-07-23 by the root during the Decision-33(vi)+35(iv) hygiene pass.** Verbatim text
moved out of `width.md` task blocks; each donor block retains a one-paragraph verdict + pointer.
**Frozen history — NEVER dispatch from this file.** Citations and numbers in here are exempt from
the plan gates by the archive rule and are allowed to rot; the citable records are the ledgers and
the live blocks in `width.md`.

---

## §WSEL-11 — the DISCARDED first run (case law) + the original pre-registration

### WSEL-11 — the DISCARDED first run — ⛔ **RESULTS DISCARDED — RUN ON A FORBIDDEN OBJECTIVE (variance fitting). The verdict below is VOID.**

**Why it is void.** The run trained on the **Gaussian negative log-likelihood** —
`automl_package/examples/width_wsel11.py:98-101` calls
`gaussian_log_likelihood(mean, log_var, y)` on `IndependentWidthNet`, whose variance heads are real —
so it fitted **mean AND variance**. MASTER Decision 2 parks variance for this strand and §3.7 makes
sigma-fixed-at-truth binding on every width run. The study is therefore measured on a **different objective from
every arm it is compared against**, and its verdict may not stand. *(User ruling 2026-07-21: results
produced in violation of a constraint are DISCARDED, not reinterpreted.)*

**Status of the artifacts.** The six per-λ/seed JSONs and `frozen.json` under
`automl_package/examples/capacity_ladder_results/WSEL11/` **stay on disk as a record of what was run
and may NOT be cited as evidence for anything.** Deletion is user-gated and is not proposed — the
record of a discarded run is worth keeping.

**Downstream consequence — this is the part that matters.** MASTER Decision 21 requires this check to
pass BEFORE the battery is read. With the check void, the precondition is **unmet, not satisfied**:
**WSEL-8 and WSEL-10 are blocked again**, exactly as they would be had the check never run. The
earlier "Battery NOT blocked" line is withdrawn.

**What the re-run must change (and ONLY this):** train with sigma FIXED at the generator's true value — on tier 1 that is `--loss mse`,
explicit, everything else — toy, seeds, λ grid, convergence gates, selection rule — byte-identical to
the original spec below, so the only moving part is the objective. **Do not widen the λ grid or change
the selection rule while re-running; that would make the re-run incomparable to its own
pre-registration.**

*(The recorded verdict text is retained below, struck, because the case law is the point: a check
designed to protect a battery was itself run outside the constraint it was protecting.)*
~~✅ DONE 2026-07-21. VERDICT: SELECTION DOES NOT MOVE. Battery NOT blocked.~~

**RESULT: `selection_moved: false`** — ledger `automl_package/examples/capacity_ladder_results/WSEL11/frozen.json`, six per-cell JSONs (λ ∈ {0, 1e-4, 1e-2} × seeds {0,1}) in the same directory.
Selected width is **invariant across the whole weight-decay grid on both seeds**: seed 0 selects the
same width at λ=0, 1e-4 and 1e-2; seed 1 likewise selects its own same width at all three. No cell
moves beyond tolerance, so `moved_vs_baseline_by_seed_and_lambda` is `false` throughout.

**Consequence (MASTER Decision 21):** the strand-local block does **not** fire. **WSEL-8 and WSEL-10
proceed**, and **WSEL-10's report MUST cite this as the robustness note** Decision 21 requires — the
"does not move" branch is not a silent pass, it is a citable result.

**What this does and does NOT establish** *(root, 2026-07-21 — recorded so the report cannot
over-claim it)*:
- **Does:** width's cheapest-within-tolerance selection is not an artefact of the research loop being
  unregularised. The Decision-21 worry — that small capacity wins because small OVERFITS LESS rather
  than because small SUFFICES — is not operating here, on this toy, at this grid.
- **Does NOT:** establish agreement ACROSS seeds. The two seeds select **different** widths (7 and 6),
  which is a different question this task never asked and its grid cannot answer. Stability *under
  penalty* ≠ stability *across seeds*; do not present one as the other.
- **Does NOT:** generalise beyond the one toy the task specifies. This is a discriminating check by
  design (§4 non-goals: "no sweep over toys or seeds beyond what's specified"), not a survey.

**Depth inheritance is now MOOT for this cycle:** Decision 21 has depth inherit this treatment after
its positive control passes — `DSEL-2c` failed all four arms the same day and the depth strand is
⏸ PARKED, so no depth regularisation check is scheduled. ProbReg's `P8` is the remaining live half.

*(Original task spec follows, retained verbatim as the pre-registration this run was judged against.)*

### WSEL-11 — does explicit regularisation move the selected width? (MASTER Decision 21)

**Inserted 2026-07-21 (cross-strand repair, MASTER Decision 21).** The programme's research training
is entirely unregularised — no weight decay, dropout, norm layers, or mini-batching — so the
cheapest-within-tolerance rule (§1) may partly select small widths because small overfits less, not
because small suffices. That confound would bias every dial the same direction and would otherwise be
invisible.
**Files (write set):** `automl_package/examples/width_wsel11.py` (Create) ·
`automl_package/examples/capacity_ladder_results/WSEL11/`
**Spec:** Discriminating check, one toy: train the per-width sweep (the existing converged-width
machinery, ordinary per-width models) at AdamW weight_decay `λ ∈ {0, 1e-4, 1e-2}`, 2 seeds, unchanged
convergence gates; apply the strand's selection rule (§1, cheapest-within-tolerance) to each curve.
Report whether the selected width moves beyond tolerance. **It moves → block THIS strand's battery
reads (WSEL-8/WSEL-10 may not proceed), log the finding prominently, continue the OTHER strands, and
batch it for end-of-run user review** *(pre-authorized 2026-07-21 — a strand-local block, not a
whole-run halt)* — the
strand's numbers conflate capacity with regularisation, and the battery may not be read until
re-derived. **It does not move → robustness note that WSEL-10's report MUST cite.**
**Non-goals:** no sweep over toys or seeds beyond what's specified; no change to §1's selection rule
itself; no re-run of WSEL-4's or WSEL-8's numbers from here — this is a discriminating check, not a
re-derivation.
*Orchestration:* parallel: yes · deps: none · tier: sonnet high · scale: dynamic · shape: research ·
verify: one JSON per (λ, seed) under `automl_package/examples/capacity_ladder_results/WSEL11/` each
carrying `selected_width`, `held_out_trajectory`, `hit_cap: false`; a `frozen.json` with
`selection_moved: bool` and, if `true`, the per-λ selected widths. The reported numbers come from a
split not used for stopping or selection.

---

## §WSEL-13-spec — the retained pre-registration

### WSEL-13 — the task spec (retained as the pre-registration this run was judged against)

**Why.** The certified design's account (`shared/width_transformer_port.md` §1–§2) rests on a property
nobody has measured: because hidden unit `j` receives gradient only from widths `k >= j`, the summed
loss should induce **decreasing importance with index** — unit 1 the most important, the last unit the
least. If it does not hold, "nested prefix" is a naming convention rather than a mechanism, and the
transformer port argument (§4 of that note) loses its basis. It matters MORE with many rungs, which is
the transformer regime.

**⚠️ Correction to an earlier claim in `RESUME.md`: this task DOES retrain.** "No retraining, uses the
landed models" was wrong — verified 2026-07-21, `find` over
`automl_package/examples/capacity_ladder_results/` returns **no saved width state dicts** (the `.pt`
files there belong to F1/V0, not to any W_ dir). The canonical cell is a 1-D toy and retraining 3 seeds
is minutes, so this is cheap, but it is not free and the task must say so.

**Files (write set):** `automl_package/examples/width_wsel13.py` (Create) ·
`automl_package/examples/capacity_ladder_results/WSEL13/` (Create by runs)
**Reads (never writes):** `automl_package/examples/kdropout_converged_width_experiment.py` (imports
`_train_kdropout_to_convergence`, `:118`) · `automl_package/models/flexnn/width/architectures.py`
*(corrected 2026-07-21: this line previously named `automl_package/models/architectures/nested_width_net.py`,
which FP-11 turned into a re-export shim. The architectures themselves now live at the flexnn path.
The citation gate passed either way — the shim still exists — which is exactly why a stale citation
survives it. Drivers may keep importing through the shim; the plan must name the real home.)*

**Spec (execution-level).**
- [ ] **Step 1 — train the canonical cell, 3 seeds.** `SharedTrunkPerWidthHeadNet`, `w_max=12`,
  `hetero`, `n_train=1500`, `sigma=0.05`, `lr=1e-2`, MSE loss, sandwich schedule, the driver's own
  convergence gate — i.e. the certified configuration, by importing the driver's training function, not
  by reimplementing a loop. Save each trained `state_dict` to `WSEL13/state_seed<S>.pt` so the
  diagnostic is re-runnable without retraining ever again.
- [ ] **Step 2 — diagnostic A, single-unit ablation (uses only the widest head).** For each hidden
  unit `j` in `1..w_max`: zero unit `j` alone in the hidden vector (all others intact), read the
  width-`w_max` head, and record the MSE increase vs the unablated net **on the REPORT split defined in
  Step 3** (never on training data, never on the split used for any selection). That is
  `importance_j`. Report Spearman correlation between `j` and `importance_j`.
- [ ] **Step 3 — diagnostic B, prefix vs greedy (the non-circular test).** The per-width heads were
  trained on prefix masks, so scoring an arbitrary unit subset with them is circular. Instead **re-fit a
  fresh linear readout by ordinary least squares (closed form, WITH intercept) on the frozen trunk's
  hidden features** for each candidate subset.
  **THREE SPLITS, and the separation is load-bearing — do not collapse them:**
  1. **FIT split** (the training split): the least-squares solve for a given unit subset.
  2. **SELECT split**: greedy forward selection picks the next unit by lowest error **here**.
  3. **REPORT split**: both `prefix_k` (units `1..k`) and `greedy_k` are finally scored **here**, and
     this split is touched by neither the fit nor the selection.
  **Why this matters: if greedy selected on the split it is scored on, it would beat the prefix by
  construction and the secondary bar would be meaningless.** Splits are carved from the held-out data
  the same way `automl_package/examples/probreg_p8.py:170-172` carves its selection/report split — the
  precedent exists; reuse the shape.
  Report per-`k` REPORT-split MSE for both, the greedy selection order, and Kendall tau between the
  greedy order and the index order.
- [ ] **Step 4 — write `automl_package/examples/capacity_ladder_results/WSEL13/frozen.json`** carrying `spearman_index_vs_importance` (per seed),
  `mean_relative_prefix_gap`, `kendall_tau_greedy_vs_index` (per seed), and `ordering_holds: bool` per
  the bars below.

**Pre-registered bars (fixed BEFORE the run; no re-run on failure, no bar edits after seeing numbers).**
- **Primary:** Spearman correlation between index and ablation importance `<= -0.5` on **at least 2 of
  3 seeds**.
- **Secondary:** mean over `k` of `(prefix_k - greedy_k) / greedy_k` `<= 0.10`.
- `ordering_holds = primary AND secondary`.
**A FAIL is a finding, not a bug** — it does not block this strand's battery; it invalidates §2/§4 of
`shared/width_transformer_port.md`, which the root then corrects in the same turn, and it is reported
prominently for end-of-run user review.

**Non-goals:** no retuning, no architecture change, no new selection rule, no other toy, no other arch
(`NestedWidthNet` is closed). Do not touch the driver — import it.
*Orchestration:* parallel: yes (write set disjoint from WSEL-12/WSEL-14) · deps: none · tier: sonnet
high · scale: static (3 seeds) · shape: research ·
verify: `AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel13.py --selftest`
PASS (a 2-unit toy where the true ordering is known by construction, asserting the diagnostic recovers
it — the driver's own correctness check); then one JSON per seed plus
`automl_package/examples/capacity_ladder_results/WSEL13/frozen.json` exist and carry every field named
in Step 4.

---

## §decision-20-draft — the root's draft recommendation, superseded by the user ruling

#### DECISION-20 REVISIT — DRAFT RECOMMENDATION, FOR USER SIGN-OFF (root, 2026-07-22; NOT decided)

Inputs: the schedule grid (WSEL-14), the cost probe, and the 5-seed confound extension. Facts on
record: sandwich holds every accuracy bar and is the cheapest adequate schedule; its mid widths are
a per-seed LOTTERY (variance, not bias) that the ALL schedule eliminates; ALL costs ~2.4× per step
(intrinsic per-width-head work — head loop/backward/optimizer, per the probe), partially offset by
converging in ~0.8× the steps → **~2× total wall-clock at `w_max=12`**; <!-- source: `automl_package/examples/capacity_ladder_results/WSEL14/cost_probe/result_orderA_all.json` + `automl_package/examples/capacity_ladder_results/WSEL15_ALLSCHED/frozen.json` -->
the cost gap vanishes under a vectorised readout (fused heads are EXACT under ALL; the cheap
running-sum structure is natively vectorised); and starvation worsens with width count
(§3.10 — each mid trains on ~2/(w_max−2) of sandwich steps).

**Draft recommendation (three clauses):**
1. **Sandwich stays the default** for cost-sensitive training of the certified architecture at
   `w_max=12` — nothing it is certified for is contradicted.
2. **ALL becomes REQUIRED for any run whose READOUT consumes mid-width values** (per-width
   profiles, ordering diagnostics, architecture-comparison per-width tables): under sandwich those
   numbers carry a 6× seed lottery and are not measurement-grade. ~2× cost is the price of a
   trustworthy readout. *(Consequence to note, not retrofit: stage-1 WSEL-16 per-width PROFILES
   carry the sandwich caveat; its PRIMARY bar reads full-width only and is unaffected.)*
3. **At larger `w_max`, or once any vectorised/cheap readout lands, ALL becomes the default** —
   the cost argument for sandwich is architecture-bound and scale-bound, and both bounds are
   expected to fall.
