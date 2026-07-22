# Strand: width — per-input and global width selection

**Owns the width workstream from the certified architecture forward**: the three ways of choosing a
width, the defects, the studies that fix the method's parameters, the comparison battery, and the
report. Read `MASTER.md` + this file — that is the whole context. If another document disagrees with
this one about width *selection*, **this one wins and the other is a bug to fix**.

**Division of ownership, explicit, so this file does not repeat the failure it was written after.**
`width-cert.md` owns the **architecture certification** (`G-WIDTH = PASS`, 2026-07-16,
`SharedTrunkPerWidthHeadNet` = the architecture of record). That work is **CLOSED and is not
reopened here.** This file cites it and never restates it. Everything about *choosing* a width —
which was never inside the G-WIDTH gate rule (`width-cert.md:308-318`, two clauses, both about the
dial's behaviour, neither about selection) — is owned here.

**Why this file exists (2026-07-20).** The ProbReg re-scope identified four planning failures. An
audit of the width workstream against the same bar found all four open for width, plus two live code
defects and two soft spots in the record. None of that contradicts `G-WIDTH = PASS`; the gate simply
never covered it. Writing width's remaining work to the same shape as `probreg.md` is the fix.

**Cross-plan dependencies.** This strand no longer owns the selection API — `flexnn-package.md`
does, so several tasks below block on it explicitly rather than re-deriving what it builds:
- **`flexnn-package.md` FP-1** — moves the cost/FLOP accounting module out of
  `automl_package/examples/capacity_accounting.py` into a new module under `automl_package/utils/`,
  leaving the old path as a re-export shim. **WSEL-5** depends on it.
- **`flexnn-package.md` FP-3** — the one `CapacitySelection` enum (`automl_package/enums.py`) shared
  by every capacity family, replacing the per-family `inference_mode` string. **WSEL-2** and
  **WSEL-3** depend on it.
- **`flexnn-package.md` FP-9** — "the shared selection primitives" (the cheapest-within-tolerance
  selector + its bootstrap standard-error helper), consumed rather than re-implemented per family.
  **Correction, 2026-07-21:** this line originally read "not yet a landed task as of this writing";
  that was stale even at the time — `flexnn-package.md` records FP-1, FP-3, and FP-9 all
  **DONE/LANDED 2026-07-20**, the same day this file was drafted. `automl_package/utils/capacity_accounting.py:258,276,298`
  (`router_fit_cost`, `held_out_read_cost`, `sweep_cost`) is FP-9's/FP-1's landed code, merged in
  commit `e3cc52b`. **WSEL-3** depends on it, and the dependency is now satisfied.
- **`flexnn-package.md` FP-4** — the package width class's shared-training schedule deviates from
  the certified sandwich schedule (sums ALL configured widths every step instead of sampling a
  subset, `automl_package/models/flexnn/width/model.py:12-16`); FP-4 owns establishing whether
  that deviation is material. Added 2026-07-21 (repair audit): **WSEL-3** and **WSEL-4** depend on
  it — see §1's warning note and their own deps lines.

---

## 1. Model definitions — the three ways of choosing a width

The comparison is between three ways of choosing **w**, the hidden width the network computes at.
**W-SHARED and W-PERINPUT are read off the SAME trained network** — the certified
`SharedTrunkPerWidthHeadNet` under its certified joint width-dial schedule (the sandwich: every step
always trains the smallest and largest width plus two random middles,
`automl_package/examples/nested_width_net.py:93`). Training is therefore NOT a variable between
those two; they differ in exactly one thing, how the width is chosen.

⚠️ **Warning, added 2026-07-21 (cross-strand repair audit) — the paragraph above describes the
certified schedule, not necessarily the schedule the shipping code runs.** The shipping class
`FlexibleWidthNN` (`automl_package/models/flexnn/width/model.py:12-16`, self-documented) sums
the loss over **ALL** configured widths every step, not the sandwich's smallest+largest+2-random-mid
subset — an **unvalidated deviation**, owned by `flexnn-package.md` FP-4, which has not yet shown it
material or immaterial. Both W-SHARED (**WSEL-3**) and W-PERINPUT
(`automl_package/models/flexnn/width/model.py:239-298`, the file WSEL-3 also writes) are built on
this same class. **W-SHARED and W-PERINPUT may not be read off it until FP-4 rules the deviation
material or immaterial** — see WSEL-3's and WSEL-4's deps in §4, and the cross-plan dependencies note
above. If FP-4 finds it MATERIAL, this paragraph's "under its certified joint width-dial schedule"
claim is false for the package port and must be re-argued together with the "not a confound"
argument below.

⚠️ **W-SWEEP DOES NOT USE WIDTH-DROPOUT (user ruling, 2026-07-20). Each of its per-width models is
trained ORDINARILY at its own fixed width** — that is what "a network dedicated to that width"
means. A per-width model trained with the sandwich schedule across the whole ladder is not dedicated
to anything; it is the same network read at a different point and cannot serve as an independent
reference. **Deliberate, and not a confound**: W-SWEEP is not an arm in a controlled contrast, it is
the expensive ceiling, and training dedicated networks is what makes it expensive. The
single-difference rule binds **W-SHARED vs W-PERINPUT**; it does not bind the reference.
*(This is what `automl_package/examples/converged_width_experiment.py` already does — each width
trained independently to its own convergence. The existing artifact is therefore the right shape.)*

🔑 **EACH MODEL IS THE COMPLETE SYSTEM, INCLUDING ITS SELECTION MACHINERY.** Every one is scored
end-to-end and costed end-to-end: the selection step is *inside* the model, never a side-analysis
reported beside it. A table row for W-SHARED is the selector's answer, not the network's answer at
some width with the selector mentioned in a companion field. This is binding on the driver, the
metrics and the report. *(This is exactly what the current width results do NOT do — see §3, WD5.)*

| Model | = the certified net + | How w is chosen | Cost | Mechanism |
|---|---|---|---|---|
| **W-SHARED** | a cheap held-out read | ONE w for the dataset | cheap | to be built — **WSEL-3** |
| **W-PERINPUT** | the distilled router | a w **per input** | cheap | `fit_router` + routed predict (`automl_package/models/flexnn/width/model.py:239-298`) |
| **W-SWEEP** | a per-width sweep | ONE w for the dataset, by training a **separate model per width** and scoring each held out | **expensive — the reference** | `automl_package/examples/converged_width_experiment.py` (exists; results `automl_package/examples/capacity_ladder_results/W_CONVERGED/w_converged_summary.json`) — **needs porting to the package class, WSEL-4** |

**What W-SWEEP is for.** It is the honest, expensive way to pick a width. The efficiency claim of
this strand is that **W-SHARED reaches W-SWEEP's answer at a fraction of the cost**. It is therefore
not optional and not a baseline — it is the thing W-SHARED is measured against. W-SHARED and W-SWEEP
differ only in *how the same global width is found*; W-PERINPUT is the separate question of whether
the width should be global at all.

**Both halves of the W-SHARED ≈ W-SWEEP claim must be tested, and they are different claims:**
- **(a) same quality** — read at a given width, does the jointly-trained dial network match a network
  dedicated to that width? *(Partially addressed by the certification's fit-at-floor evidence, but
  never against the converged per-width sweep — see §2.)*
- **(b) same choice** — does the cheap read pick the width the sweep would pick? **Never tested, and
  currently unplanned anywhere.** → **WSEL-8**

**Selection rule, binding on the two global arms (imported from the ProbReg decision, deliberately):**
**cheapest-within-tolerance, NOT argmax.** w is the **smallest** width whose held-out score is not
meaningfully worse than the best width's, where "meaningfully" = exceeding twice a bootstrap-estimated
standard error (the rule published in `docs/reports/probreg_kselection/probreg_kselection.md` §3.2).
Rationale: held-out curves are noisy, so argmax systematically overshoots and reports more capacity
than the data supports; and the question this strand asks is "how much width is needed", which is a
smallest-sufficient question. **The same rule applies to W-SWEEP's curve**, or W-SHARED and W-SWEEP
are not answering the same question. This supersedes the plain `argmin` currently used by the
script-level readouts (`automl_package/examples/sinc_width_experiment.py:486-525`).

**✅ CONFIRMED BY USER RULING, 2026-07-20 — the tolerance split stands as written below.** The flat
`0.25` is an **inherited** constant (copied from `sinc_width_experiment.py`'s tie threshold), never
measured. That is the known soft spot, and it is accepted deliberately: because the arms are compared
on held-out error and cost rather than on their chosen widths, the value does not carry the
comparison. **Noted follow-up, NOT a blocker and NOT scheduled:** if a reviewer leans on the `0.25`,
a sensitivity sweep would turn it from inherited into measured and freeze the result in a
`frozen.json`. Do not run it pre-emptively.

**W-PERINPUT runs on a DIFFERENT tolerance rule, and that is legitimate, not an oversight.** The
distilled router does not read a curve — it labels each row independently at a flat relative margin,
`DEFAULT_TOLERANCE = 0.25` (`automl_package/models/flexnn/routing.py:57`), applied per the
docstring at `:64` and enforced in code at `:79-80` as `error <= (1 + tolerance) * row_min`. The two
rules differ because the two selection problems differ: a **per-input** labelling decision has one
row's worth of evidence and no standard error to estimate from it, while a **global** chooser reads
a whole held-out curve, over which a bootstrap standard error is exactly the right notion of noise.
Forcing W-PERINPUT onto the twice-SE rule would mean estimating a standard error from a single
observation, which is not meaningful. **Consequence, stated so the report does not paper over it:
W-PERINPUT's width choices are not directly comparable to W-SHARED's or W-SWEEP's on tolerance
grounds** — the three arms share a cost objective, not a shared statistical selection rule — and
**WSEL-10's report must say so explicitly**, not merely list the two tolerance values side by side.

**Confound doctrine (MASTER Decision 15 generalised):** an arm that differs from its comparator in
more than one respect is NOT dispatchable. State the single difference in the task, or do not run it.

---

## 2. State — what is established, and what is not

**Established.** `G-WIDTH = PASS` (2026-07-16, re-derived from disk over 21 W-strand result files).
The gate's two pre-registered clauses — (a) the dial reads capacity rather than error on the
noisy-easy clause, (b) dial-separation ≥3/4 cells and fit-at-floor at both σ=0.05 corners — both
passed. `SharedTrunkPerWidthHeadNet` is the architecture of record. Evidence:
`docs/width_mse_2026-07-16/verdict_variable_width_mse.md` §10, `width-cert.md:318-328`.
**This is an architecture result. It certifies none of §1's three models.**

> ⛔ **REVIEW FINDING 2026-07-22 — G-WIDTH's GATE LOGIC DOES NOT RE-DERIVE. USER RULING REQUIRED.**
> Independent adversarial re-derivation (full report:
> `automl_package/examples/capacity_ladder_results/REVIEW_2026-07-22/gwidth_rederivation.md`; the two
> decisive findings root-verified against the raw JSONs before recording). The NUMBERS are faithful
> (21/21 cited files provenance-clean; the old mislabel history was handled correctly) and the
> empirical substance stands (floor reached where the ceiling control reaches it; dial separation
> 12–70 SE). What fails is the GATE AS PRE-REGISTERED:
> **(a)** the curve-shape quarantine was silently dropped exactly where it failed — BOTH counted
> seeds of the noisy-easy clause fail it (seed 0 at 3.6× threshold + a missing hard-region drop;
> seed 2 at 2.5×), leaving ZERO eligible seeds as written, and the pre-registered remediation
> (add seeds 3/4) never ran; <!-- source: `automl_package/examples/capacity_ladder_results/W_KDROPOUT_CONVERGED/w_kdropout_converged_summary_shared_trunk_mse_hetero3_n2250_s0.05_wp3.json` -->
> **(b)** the small-data corner FAILS fit on 3/3 seeds countable under the plan's own
> trustworthiness bar (1.52–1.57 vs the 1.25 bar), and the rule's own FAIL branch — "run W8 FIRST,
> then escalate to user" — was not taken; the "not certifiable" framing rested on a stricter
> all-widths rule the plan does not prescribe. <!-- source: `automl_package/examples/capacity_ladder_results/W_KDROPOUT_CONVERGED/w_kdropout_converged_summary_shared_trunk_mse_n200_s0.05_wp4c.json` -->
> Also on record: an architecture-specific starved-narrow-width signature visible in cert-era data
> went unreported (the certified design's rarely-sampled narrow heads sit stale — feeds the
> WSEL-14/schedule story); one ladder cell misreported three ways; the "5/5" headline contains one
> inside-noise pass; three cert-cited files were regenerated post-cert (cert-era bytes recoverable
> at `bb7e9dc` and matching).
> **STANDING CONSEQUENCES until the user rules:** the PASS is procedurally unsound as written
> though empirically plausible; **WSEL-3..8 dispatch HELD** (they build on the certified
> architecture); **WSEL-16 CONTINUES** (its own non-goal: `B_HEADS` is the reference, not a
> defendant — and its comparison is informative under ANY ruling). ~~Options prepared for the
> ruling~~ **SUPERSEDED 2026-07-22 — the plan itself prescribed the remediation
> (`EXECUTION_PLAN.md:192-193` "add seeds 3/4 if any seed is quarantined by the curve gate"); run
> without a ruling, as it should have been in July. The W8-scan precondition of the (b)-fail
> branch was already satisfied on disk.**
>
> **RE-GRADE 2026-07-22 (seeds 3/4 landed; gate re-read mechanically as written):**
> - **Seed 4 — the FIRST VALID measurement of the noise-vs-difficulty clause, and it PASSES:**
>   curve gate clean (easy-flat 0.0030 vs allowed 0.0036); dial expected width 2.31 on the
>   noisy-easy region vs 8.34 on the hard region, separation ≫ 2·SE. <!-- source: `automl_package/examples/capacity_ladder_results/W_KDROPOUT_CONVERGED/w_kdropout_converged_summary_shared_trunk_mse_hetero3_n2250_s0.05_wp3s4.json` -->
> - **Seed 3 — QUARANTINED by the same curve-gate failure as the originals** (easy-region error at
>   width 2: 0.0109 vs allowed 0.0033, 3.3× over). <!-- source: `automl_package/examples/capacity_ladder_results/W_KDROPOUT_CONVERGED/w_kdropout_converged_summary_shared_trunk_mse_hetero3_n2250_s0.05_wp3s3.json` -->
>   **The stale-narrow-heads pathology now hits 3 of 5 seeds on this toy variant** — a real, named
>   limitation of the certified architecture under the sandwich schedule (plausibly the same
>   narrow/mid-width starvation VARIANCE the WSEL-15 FOLLOW-UP measured; testable under the ALL
>   schedule if the user wants that evidence).
> - **Formal state:** clause (a) now has 1 valid PASS against a pre-registered quorum of ≥2
>   (all other seeds quarantined: 0/2/3 by curve gate, 1 by convergence) → the clause cannot PASS
>   as written on current data. Clause (b)'s small-data fit failure stands (universal — the
>   ceiling control fails identically). **⇒ G-WIDTH cannot be marked PASS under its own rules;
>   the plan-reserved user escalation is NOW properly constituted**, with every existing
>   measurement — valid and quarantined alike — pointing one way (the dial reads difficulty, not
>   noise). User options: extend seeds beyond the pre-registered 3/4 to seek quorum (itself a
>   rule amendment); amend to PASS-with-caveats on evidence-weight; or demote.
>
> ✅ **USER RULING 2026-07-22: G-WIDTH = PASS WITH CAVEATS (evidence-weight).** Binding caveats,
> which every downstream consumer carries: (1) the stale-narrow-heads pathology on the noisy-easy
> variant (3/5 seeds under the sandwich schedule) is a NAMED LIMITATION of the certified
> architecture; (2) the small-data corner failure is real and UNIVERSAL (ceiling control fails
> identically) — the architecture is not certified at n=200; (3) the noise-vs-difficulty clause
> rests on 1 valid seed (passing decisively) + directionally-consistent quarantined seeds, not on
> its pre-registered quorum. **WSEL-3..8 are UNBLOCKED** (dispatch on the user's go).

**Established (MECHANISM, recorded 2026-07-21 after a user discussion — an ACCOUNT, not new evidence).**
Why the shared readout fails and the per-width readout succeeds is now written down once, so it is not
re-derived: `shared/width_transformer_port.md` §1. Summary: the summed-over-widths loss asks a single
shared output weight for two incompatible things (width 1 wants hidden unit 1 to carry the whole
prediction; width `w_max` wants it as one contributor of `w_max`), and per-width output nodes end the
conflict by giving each width its own copy of the contested parameter. Same note carries the parameter
cost arithmetic (§3), the conditions under which this design ports to a transformer (§4–§5), and the
structural argument that DEPTH is the better port target (§7, an input to `depth-selection.md`, not a
width claim). **`NestedWidthNet` (arch #1) is CLOSED — no further compute (user, 2026-07-21).**

**NOT established — each is a task below:**
- **The importance-ordering property the account above rests on.** Hidden unit `j` receives gradient
  only from widths `k >= j`, so the summed loss *induces* a decreasing importance ordering — never
  measured. It matters more at transformer scale, not less. → **WSEL-13**
- **What the width schedules cost and whether sampling buys anything once the trunk is computed
  once.** Four-widths-per-step is measured (`width-cert.md:210-220`); one-width-per-batch and
  all-widths-every-step are not, and no schedule cell has ever recorded parameters, executed FLOPs or
  wall-clock. → **WSEL-14**
- **W-SHARED does not exist as library code.** The only cheap global readouts are script-level
  `argmin` (`automl_package/examples/sinc_width_experiment.py:486-525`;
  `automl_package/examples/moe_flexnn_comparison.py:309-336`), which is the wrong rule per §1. → **WSEL-3**
- **W-SWEEP is not usable as a reference.** The converged per-width sweep exists and ran, but against
  the research module's own width classes, not `FlexibleWidthNN`; and `width-cert.md` never cites it
  (`grep` for the script name returns zero hits there). The "#3 positive control" used throughout the
  certification is `IndependentWidthNet` trained under the *same cheap joint schedule* as everything
  else — a different object. → **WSEL-4**
- **(b) same choice.** Never tested. → **WSEL-8**
- **How much data the selection step needs.** Never measured. The selection split is a hardcoded
  50/50 even/odd index carve (`automl_package/examples/kdropout_converged_width_experiment.py:273-276`).
  → **WSEL-6**
- **Whether the router's architecture matters.** Task W6 varied ONE dimension (hidden size,
  half/double), on one toy, 3 seeds, as a binary pass/fail on the downstream deploy claim
  (`width-cert.md:234`, invariance at `:237`). That is a does-it-break check, not a search. → **WSEL-7**
- **Selection cost — WSEL-5's dependency landed; the wiring itself has not.** **Superseded 2026-07-21
  (repair audit):** this line originally read "Nothing anywhere charges the cost of choosing the
  width," citing `metrics-accounting.md` S2 (accurate as of that section's 2026-07-16 writing) and
  the pre-move `automl_package/examples/capacity_accounting.py`. As of `flexnn-package.md` FP-1/FP-9
  (DONE 2026-07-20, commit `e3cc52b`), the three cost functions now exist —
  `automl_package/utils/capacity_accounting.py:258` `router_fit_cost`, `:276` `held_out_read_cost`,
  `:298` `sweep_cost`. **This is FP-1/FP-9 landing the primitives, not WSEL-5 completing** — WSEL-5's
  own job is wiring §1's three models onto them and updating `metrics-accounting.md`'s S2 scope, and
  that task's spec (§4) is unchanged. → **WSEL-5**
- **Anything on real data, and any baseline.** Width has been run on two synthetic toys only, and no
  external comparator (tree model / plain NN / linear) has ever been run for this strand. → **WSEL-9**
- **Any width report.** Width's reportable content was folded into a unified report that has never
  run. → **WSEL-10**

---

## 3. Known defects

**WD1 — FIXED 2026-07-20, commit `63ab6bc`** ("fix(width): correct predict_uncertainty; certify the
driver default"). **Correction to this entry's own original claim:** it previously said
`FlexibleWidthNN` had zero test coverage on this path; that was true when written and is no longer
true. `FlexibleWidthNN.predict_uncertainty` is now overridden
(`automl_package/models/flexnn/width/model.py:361-397`): `CONSTANT` / `BINNED_RESIDUAL_STD`
delegate to the inherited implementation (neither touches the stacked
`(len(widths), N, output_size)` tensor), and `MC_DROPOUT` / `PROBABILISTIC` raise
`NotImplementedError` explicitly instead of silently mis-indexing it. Covered by
`TestFlexibleWidthNNPredictUncertainty` (`tests/test_flexible_width_network.py:277-318`), one test
per `uncertainty_method` value including both raise cases. **WSEL-1 is DONE by this commit — see
§4.**

**WD2 — FIXED, landed in commit `e3cc52b` (`flexnn-package.md` FP-3's write set); recognised
2026-07-21.** *(Description of the defect as it stood, with its original line citations REMOVED
rather than repaired: they pointed at `automl_package/enums.py:105-109` and
`automl_package/models/flexnn/width/model.py:92`, `:192`, `:204-205`, none of which describe the
current file — the symbol they named no longer exists anywhere in the repo, so there is nothing left
to cite. Per the repair-pass rule, this entry was re-verified against disk, not re-worded.)* The
typed enum sat on the broken knob and the magic string on the working one: `WidthSelectionMethod.DISTILLED`
was documented as "not yet landed" and raised `NotImplementedError` at construction, while the
mechanism that *does* work was reached by the raw string `inference_mode="routed"`, validated by a
hand-rolled membership check. Omitting that string after fitting a router silently gave the largest
fixed width — no error, no warning, nothing recording that a router was fitted and unused.
**As fixed:** `capacity_selection` is a `CapacitySelection` member passed at construction
(`automl_package/models/flexnn/width/model.py:79`); `predict`
(`automl_package/models/flexnn/width/model.py:264`) and `predict_uncertainty`
(`automl_package/models/flexnn/width/model.py:365`) carry no `inference_mode` parameter;
`grep -rn "WidthSelectionMethod" automl_package/ tests/` returns nothing — the enum is gone from the
repo entirely, so the closed-set violation cannot recur. → **WSEL-2, now DONE (§4).**

**WD3 — NOT A DEFECT. Withdrawn 2026-07-20 after reading the verdict's own §2.1.**
An audit flagged `NestedWidthNet`'s **FAILS** verdict as resting on non-converged runs
(`untrustworthy_seeds: [0,1,2]`), and this file originally recorded that as a defect. **That was
wrong, and the error was mine for propagating the flag without reading the section that answers it.**
`docs/width_mse_2026-07-16/verdict_variable_width_mse.md` §2.1 — *"Nested convergence — not
under-training (the load-bearing check)"* — closes the under-training door explicitly: a confirmation
run with early stopping DISABLED out to **120,000 epochs** (~12× the canonical budget) finds a
**global minimum over the whole run** never below 0.063 on any seed, against ~0.024 for the arms that
do reach floor; the trajectory is flat from ~10k to 120k with no late breakthrough, and the test fit
ratios reproduce the canonical run.

**⇒ The pure nested architecture (shared trunk + SHARED readout) genuinely FAILS: it plateaus at
3.7–5.9× the hard-region noise floor and stays there.** The verdict is safe to cite. The
`untrustworthy` flag means the automated convergence gate did not certify those seeds; §2.1 is the
manual analysis that resolves it. **No task, no re-run.**

*Case law worth keeping: an `untrustworthy`/non-converged flag is a prompt to go read the verdict's
own convergence analysis, not a licence to call the verdict unsafe. This programme's own rule
([[feedback_check_loss_trajectory_before_concluding]]) cuts both ways — the original authors DID
check the trajectory, and the audit that flagged them did not check whether they had.*

**WD4 — FIXED 2026-07-20, commit `63ab6bc`** ("fix(width): correct predict_uncertainty; certify the
driver default"). **Correction to this entry's own citation, dated 2026-07-21:** the original text
cited `kdropout_converged_width_experiment.py:549` for the stale default and attributed a "the old
default" comment to the `--arch` flag. Both were wrong even as a description of the pre-fix state:
`:549` is `--smoke`, not `--arch`; and the "the old default" comment belongs to a different flag
entirely, `--loss` (`:552`, `default=LossType.NLL.value, help="... (default: nll, the old
default)."`) — a different training axis (loss function, not architecture). As fixed, `--arch`
defaults to the certified architecture at `automl_package/examples/kdropout_converged_width_experiment.py:551`
(`default=Arch.SHARED_TRUNK.value, help="Width-net architecture (default: shared_trunk, G-WIDTH
certified)."`). Anyone re-running the driver without an explicit flag now reproduces the certified
arm. **WSEL-1 is DONE by this commit — see §4.**

**WD5 — OPEN, the §1 violation in the existing results.** The best-fixed-width baseline is written as
a `best_fixed_k` / `mse_best_fixed` pair **inside the same case dict** as the routed model's numbers
(`automl_package/examples/kdropout_converged_width_experiment.py:387-399`) — the companion-field
pattern §1 forbids. The router's own fitting cost is charged to nothing. → structural, fixed by
**WSEL-5** + the battery's output contract in **WSEL-8**. **Correction, 2026-07-21:** this line
originally named WSEL-9 as the second half of the fix; WSEL-9 is now **⏸ PARKED** (§4) and
guaranteed not to run, so it cannot deliver anything. WSEL-8 is the live task whose output contract
does this job: its per-(toy, seed) result files each carry `w_shared_width`, `w_sweep_width`,
`held_out_trajectory`, `hit_cap: false`, and a **per-arm `selection_cost` key** (§4, WSEL-8's verify)
— no best-fixed companion field baked in, so WSEL-8's schema avoids the WD5 pattern by
construction rather than by a promise from a task that will never run.

---

## 3.5 Autonomous execution contract

**This strand is scoped to run unattended.** The root is dispatcher + verifier. Every foreseeable
branch below has a **pre-authorised default**; take it, log it, keep going. Only the HALT conditions
stop the run.

**Rule: never block on a question that has a reversible default.** Log every default taken to
`RESUME.md` `### Decisions` with the evidence that triggered it, and batch anything genuinely
user-only for the end of the run.

| Branch | Pre-authorised default | Log |
|---|---|---|
| **WSEL-6**: which selection fraction becomes the frozen default | the **smallest** fraction at which every arm is within its own noise band of its best (twice-standard-error rule); if none saturates, take the largest swept and record the study **inconclusive, floor not found** | fraction + the curve |
| **WSEL-6**: W-PERINPUT still improving at the largest fraction | freeze the largest swept and mark W-PERINPUT's battery result **"router data-limited"** — a loss then does NOT support "per-input width does not pay" | the mark, prominently |
| **WSEL-7**: conclusions invariant to router architecture | keep the current frozen default; record invariance as a finding | table |
| **WSEL-7**: NOT invariant | adopt the **smallest** configuration that reaches the plateau **as this strand's per-dial default**, report it to the root, and re-run WSEL-6 at the new router. **NEVER write `automl_package/models/common/distilled_router.py` from this strand** — that file's write set belongs to `flexnn-package.md` FP-5; the root applies a genuinely global freeze there. **Correction, 2026-07-21 (cross-strand repair, user ruling):** this branch originally said "freeze it globally," which no WSEL task can do — WSEL-7's own write set (§4) is `width_wsel7.py` + `WSEL7/`, not the router file. ProbReg's PC and depth's DSEL-9 face the identical correction; see `flexnn-package.md` FP-5. | old → new + why, and the report to the root |
| **WSEL-4**: the ported sweep does not reproduce `W_CONVERGED`'s numbers | report the discrepancy and **halt WSEL-8** — an unreproduced reference cannot serve as a reference | both number sets |
| **ceiling binds** (selected w = w_max) | re-run that cell with the ladder extended one rung; report the raise | which cells |
| a spec or older document contradicts §1 | **§1 wins**; fix the other document in the same turn | the correction |

**HALT and ask — these only:**
1. A **positive control fails** (MASTER Decision 14) — the protocol is then the defect, not the arms.
2. A study comes back **incoherent rather than merely negative** (e.g. a non-monotone curve beyond
   noise) — that is a broken instrument, and running the battery on it wastes the budget.
3. Any change to **§1's model definitions or the selection rule.**
4. Anything **irreversible or outward-facing** (deleting artifacts, publishing, **pushing to
   `origin`**). *(Amended 2026-07-21, user: COMMITTING per the `MASTER.md` branch protocol —
   wave-branch commits, local merge, branch delete, docs straight to `master` — is
   PRE-AUTHORIZED for the autonomous run and is no longer a HALT trigger. Pushing, publishing
   and deletion remain user-gated; `FP-8` stays attended-only.)*
5. Any result that would **call `G-WIDTH = PASS` into question.** That gate is closed; a finding that
   reopens it is a user decision, never a run's.

## 3.6 Constants the studies FREEZE, and the battery READS

🚫 **DO NOT WRITE THE VALUES INTO THIS TABLE.** The plan holds the *name of the constant* and the
*path of the artifact that owns it*; the value is read from that artifact at build time. A number
copied here is a cache entry that rots.

| Constant | Set by | Owning artifact (single source of truth) |
|---|---|---|
| selection-set fraction | **WSEL-6** | `automl_package/examples/capacity_ladder_results/WSEL6/frozen.json` |
| W-PERINPUT data-limited flag, per dataset | **WSEL-6** | same file, one boolean per (toy, arm) |
| router hidden / depth / epochs / lr | **WSEL-7** | `automl_package/examples/capacity_ladder_results/WSEL7/frozen.json`, else the current frozen defaults at `automl_package/models/flexnn/routing.py:57-60` if that file's `invariant` field is `true` |
| ~~labelling tolerance \| **WSEL-7** \| same file~~ — **struck 2026-07-21.** MASTER Decision 18 rules the tolerance sensitivity sweep **NOT scheduled** (§1's own note above, "Noted follow-up, NOT a blocker and NOT scheduled... Do not run it pre-emptively"); this row and WSEL-7's swept dimensions (§4) contradicted that ruling and are corrected to match it. WSEL-7 keeps router hidden/depth/epochs/lr only. |
| width ladder / `w_max` after any ceiling raise | **WSEL-8** (WSEL-9 is ⏸ PARKED — would apply there too if a later ruling unparks it) | the per-cell result JSON under `automl_package/examples/capacity_ladder_results/WSEL8/` (or `.../WSEL9/`, dormant) that recorded the bind |
| per-model selection cost | **WSEL-5** | the accounting module's own selftest artifact (exact path fixed by `flexnn-package.md` FP-1, see this file's cross-plan dependencies note) |

**Feed-forward rule (binding):** if **WSEL-8 or WSEL-10** runs at a value not justified by the
artifact named here, its results are **not reportable**. **Correction, 2026-07-21:** this rule
originally named WSEL-9, which is ⏸ PARKED and will never run, so it bound nothing; WSEL-8 and
WSEL-10 are the live consumers of these constants and are the tasks this rule must actually
constrain (the same defect depth-selection.md's equivalent rule has, and ProbReg's PC/P3/P4
equivalent — still live — does not).

⚠️ **Anchor warning for every `verify:` line below.** Where a task reproduces a frozen number as its
positive control, that anchor must come from something **not computed by the method under test** — a
second implementation, an invariant, or a published figure. Re-deriving an anchor with the same code
does not verify the worker; it conscripts the worker into confirming our own bug and returns it
stamped *verified*. Concretely: WSEL-8's quality half must NOT anchor solely on the certification's
own fit-at-floor numbers re-run through the same harness — pair it with W-SWEEP's dedicated per-width
models, which are trained independently and are exactly the non-shared implementation this rule asks for.

---

## 3.7 VARIANCE IS FIXED AT THE TRUE VALUE, NEVER LEARNED (user ruling, 2026-07-21)

**⚠️ THIS SUPERSEDES "mean-only" AS THE RULE OF THIS STRAND.** The user's clarification, verbatim in
substance: *"I'm not saying we should drop the functionality of setting it. I'm just saying just fix
it at the true value."* **The variance machinery STAYS. What is forbidden is LEARNING the variance,
because that is what overfits.**

### The rule, mechanically

Every width run uses the Gaussian likelihood with `sigma` **clamped to the generator's true,
per-point value** — never a learned `logvar_head`, never a free parameter. The toys know their own
noise exactly and already return the region label needed to look it up:

- `make_hetero` (`automl_package/examples/nested_width_net.py:143`): a single
  `HETERO_NOISE_SIGMA = 0.05` everywhere.
- `make_hetero3` (`automl_package/examples/nested_width_net.py:180`, the tier-2 control): `0.05` in
  regions 0 and 1, `HETERO3_NOISY_SIGMA = 0.5` (`:95`) in region 2 — and the generator **returns
  `region` as its third value**, so the true per-point sigma is a lookup, not an estimate.

### How each tier implements it — no choice left to the implementer

- **Tier 1:** `--loss mse`. This IS the fixed-sigma likelihood up to a constant scale (see below), and
  it keeps every certified number byte-comparable. Do not add a redundant likelihood path.
- **Tier 2 and Tier 3:** a **fixed-sigma weighted squared error** — per point, `(pred - y)^2 /
  sigma_true(x)^2`, with `sigma_true` read from the generator's `region` output. Implemented ONCE, in
  `automl_package/examples/width_candidates.py`, and imported; never re-derived per driver (§3.9).

**Any variance head present in a class is held at the true value and excluded from the optimiser's
parameter list.** A class whose `log_var` is a dummy zero (the certified `SharedTrunkPerWidthHeadNet`)
satisfies this trivially at fixed unit sigma.

### What this changes versus the squared error already on disk — state it, do not gloss it

With sigma fixed, the Gaussian negative log-likelihood equals the squared error divided by
`2 * sigma^2`, plus a constant. Therefore:

- **Tier 1 (`hetero`, constant sigma): EXACTLY EQUIVALENT to the certified MSE objective** up to one
  positive scale factor. Same optimum, same ordering of arms. **Every certified tier-1 number stays
  comparable** — this is why the ruling is safe to adopt mid-strand.
- **Tier 2 (`hetero3`, two different sigmas): a WEIGHTED squared error** — the noisy region is
  down-weighted 100x relative to the quiet ones. **This is NOT the same objective as the plain MSE
  previously run on that toy, and any table mixing the two must say so.**
- **It sharpens the tier-2 control rather than blunting it.** That toy exists because noise is
  common-mode across widths, so no width fits it down and the honest verdict is "stay narrow"; plain
  squared error over-weights that region, which is the exact failure it was built to catch. Fixing
  sigma at the truth makes the objective read capacity instead of noise, by construction rather than
  by hope.
- **Tier 3 (the `sigma` ladder):** sigma is fixed at whatever that cell's generator used, so each cell
  is internally consistent. Cells at different sigma are NOT on a common loss scale and may not be
  compared on raw loss — compare `ratio_to_floor`, never the raw number.

### The two leaks this rule closes — both were LEARNED variance, which is the forbidden thing

- **WD6 (found 2026-07-21) — the shipping driver's default loss FITS VARIANCE.**
  `automl_package/examples/kdropout_converged_width_experiment.py:553` defaults `--loss` to `nll`
  (Gaussian negative log-likelihood, mean AND log-variance), commented "the old default". Every
  certified run passed `--loss mse` explicitly; nothing stops the next one from forgetting.
  **Binding on every task in §4: never leave the loss flag to the default; tier 1 passes `--loss mse`
  explicitly, tiers 2/3 use the fixed-sigma weighted form, and the run's own provenance asserts which.** *(Not re-defaulted here: flipping a driver default silently changes the meaning of
  every un-flagged historical invocation. Flipping it is a task for the cleanup pass, with a
  reproduction check — not a drive-by edit.)*
- **WD7 (found 2026-07-21) — the regularisation study TRAINED VARIANCE.**
  `automl_package/examples/width_wsel11.py:98-101` trains through
  `gaussian_log_likelihood(mean, log_var, y)` on `IndependentWidthNet`, whose variance heads are
  real. So the study that concluded "explicit regularisation does not move the selected width" was
  run on the **Gaussian-likelihood objective**, while everything it is compared against is
  on a **fixed/absent** one. **Consequence: that verdict is NOT like-for-like and may not be cited as
  clearing the battery until it is re-derived with sigma FIXED at the true value, or the discrepancy
  is argued in writing.**
  Escalated to the user 2026-07-21; **no re-run started** (the user asked that execution hold).
  This does not automatically overturn the verdict — over-fitting the variance would, if anything,
  bias toward *smaller* selected widths, the same direction the check was probing — but the argument
  has to be made explicitly, not assumed.

**Architecture note, so this is not over-read:** the certified `SharedTrunkPerWidthHeadNet` cannot
fit a variance at all — its `log_var` is a dummy zero tensor that never enters the loss
(`automl_package/models/flexnn/width/architectures.py:179-181`). The exposure is confined to the
two OTHER classes (`NestedWidthNet`, `IndependentWidthNet`) and to whichever loss flag a driver picks.

## 3.8 THE CANONICAL TOY SUITE — the SAME SET across every width architecture and experiment (user, 2026-07-21)

**Why this exists.** The strand is about to compare four ARCHITECTURES and several SCHEDULES against
each other. A comparison whose arms quietly ran on different data, widths or seeds is not a
comparison, and the drift is invisible once the numbers reach a table. **User instruction,
2026-07-21: the same toy problems apply across all width architectures and experiments, so the
comparison is like-for-like.** *(Corrected the same day — the first draft of this section said "one
canonical toy". Wrong: this strand has a SUITE, and like-for-like means the same SUITE, not the same
single cell.)* Written here ONCE; §4 tasks reference it and none restates it.

**Fixed on every cell of every tier:** `w_max = 12` · seeds **0, 1, 2** · **sigma FIXED at the
generator's true per-point value, never learned** — tier 1 via `--loss mse` (exactly equivalent), tiers
2/3 via the fixed-sigma weighted squared error (§3.7; the driver's default loss LEARNS sigma and is
forbidden) · `lr = 1e-2` · the strand's
convergence gate unchanged with `hit_cap: false` required · `OMP_NUM_THREADS=4` pinned (the metric
moves up to ~5% with thread count, `shared/fp5-stale-reference-finding.md`).
Constants: `automl_package/examples/converged_width_experiment.py:45-49`.

### The three tiers — exact cells, no judgement calls

- **TIER 1 — the reference cell.** `--toy hetero --n-train 1500 --n-test 500 --sigma 0.05`.
  **3 runs per arm** (seeds 0, 1, 2). The 2-region easy-line + width-hungry-sine target
  (`automl_package/examples/nested_width_net.py:143`); the ladder toy the certification was read on.
- **TIER 2 — the noisy-easy control.** `--toy hetero3 --n-train 2250 --n-test 750 --sigma 0.05`.
  **3 runs per arm** (seeds 0, 1, 2). Adds a NOISY-easy region, which catches an arm that reads
  *error* where it should read *capacity* — the one failure mode the certification specifically
  probed.
- **TIER 3 — the data × noise ladder.** `--toy hetero`, `--n-train ∈ {200, 500, 1500, 4000}` ×
  `--sigma ∈ {0.05, 0.15, 0.5}`, `--n-test` left at the driver default.
  **36 runs per arm** (12 cells × 3 seeds). Separates a property of the design from a property of one
  data size or noise level. Ledger precedent: the `_n*_s*_wp4` cells already on disk.

### Which task runs which tier — FIXED. An implementer reads its row and runs exactly that.

| Task | Tier 1 | Tier 2 | Tier 3 | Runs |
|---|---|---|---|---|
| **WSEL-11** (re-run) | ✅ | ❌ | ❌ | 3 per λ × 3 λ = **9** |
| **WSEL-13** ordering | ✅ | ⛔ **DEFERRED — needs WSEL-15** | ❌ | **3 now, 6 total** |
| **WSEL-14** schedule × bunch | ✅ | ❌ | ❌ | 3 per arm × 5 arms = **15** |
| **WSEL-15** normalisation | ✅ | ✅ | ❌ | 3 per arm × 3 arms × 2 tiers = **18** |
| **WSEL-16** architectures, stage 1 | ✅ | ✅ | ❌ | 3 per arm × 5 arms × 2 tiers = **30** |
| **WSEL-16** architectures, stage 3 | — | — | ✅ | 36 × **the 2 finalist arms only** = **72** |

> ### ⛔ DEFECT IN THIS TABLE, FOUND AND FIXED 2026-07-21 (root, from WSEL-13's task review)
>
> **This table assigns tiers without checking that each tier's sanctioned OBJECTIVE exists in code.
> It does not for tier 2.** §3.7 (lines 361-363) requires tiers 2 and 3 to train on the fixed-sigma
> **weighted** squared error `(pred - y)^2 / sigma_true(x)^2`, implemented ONCE in
> `automl_package/examples/width_candidates.py` and imported, never re-derived per driver (§3.9).
> **That module does not exist yet — WSEL-15 creates it** (line 1248 below). The only loss variants
> a width driver can select today are `NLL` and `MSE`
> (`automl_package/examples/kdropout_converged_width_experiment.py:96-100`); there is no third member.
>
> **⇒ Every tier-2 and tier-3 assignment in this table carries an unstated dependency on WSEL-15.**
> WSEL-13 is the first task to hit it. Resolution, applied 2026-07-21:
> - **WSEL-13 runs TIER 1 ONLY for now (3 cells).** Its pre-registered bars are read on tier 1 in any
>   case, so `ordering_holds` is fully determined without tier 2 — the deferral costs corroboration,
>   not the verdict.
> - **Tier 2 is DEFERRED, not cancelled**, and unblocks the moment WSEL-15 lands `width_candidates.py`.
> - **The driver REFUSES tier 2 mechanically** rather than defaulting to plain MSE
>   (`automl_package/examples/width_wsel13.py`, `_assert_tier_objective_available`). A prose warning
>   would not have survived a later re-run.
>
> **Why this needed a mechanical guard and not a note.** Running tier 2 under plain MSE throws no
> error and produces a plausible number, measured on an objective this file itself calls NOT
> comparable to what the noisy-easy control exists to measure (lines 377-379). **That is precisely
> the failure that voided WSEL-11's first run** — same shape, different task, found this time by
> review rather than by an audit weeks later. The rule below ("a task deviating on any constant
> carries a written justification IN THE TASK") is what this block is discharging.
>
> **Do NOT read this as licence to skip tier 2 elsewhere.** Every other tier-2/tier-3 row above is
> still owed, and each is blocked behind WSEL-15 by the same argument.
>
> **RESOLUTION UPDATE 2026-07-22 (root; user authorized starting WSEL-16 during the width review):**
> WSEL-15 landed `width_candidates.py` **without** the weighted loss — correctly, since WSEL-15's own
> spec never named it; the dependency above was assumed, not specced (the orphaned dependency
> surfaced in RESUME). **The fixed-sigma weighted squared error is hereby assigned to WSEL-16's
> authoring contract**: it lives in `automl_package/examples/width_candidates.py` per §3.7/§3.9, and
> WSEL-16 is the first task that cannot run its own spec (Step 2, tier-2 cells) without it. Formula
> authority is §3.7 — `(pred - y)^2 / sigma_true(x)^2`, sigma from the generator, never learned.
> **WSEL-13's tier-2 driver guard stays in place** until the loss lands AND WSEL-13 tier 2 is
> separately re-authorized; this update does not touch `width_wsel13.py`.

**Rules — mechanical, nothing left to interpret:**
- **Tier 3 runs for exactly two arms**: the certified per-width-head design, and whichever candidate
  wins WSEL-16's stage 1. Never more. 72 runs is already the largest block in this strand.
- **A tier-1-only task may not state an architecture verdict.** It may state cost, timing,
  equivalence and mechanism results. The tasks above are written so that this never has to be
  adjudicated: a task's row IS its licence.
- **A task deviating on any constant above carries a written justification IN THE TASK naming the
  constant and why.** No silent deviation; no deviation discovered afterwards inside a results file.
- **A deviating cell may not be tabulated beside a canonical one** unless the deviation is named in
  the same table.
- `automl_package/examples/sinc_width_experiment.py`'s sinc toy is a DIFFERENT lineage — the certified
  router producer, not an arm here. Its cells never enter these tables.
- **The older comparison chain (WSEL-3, WSEL-4, WSEL-6, WSEL-7, WSEL-8) is NOT assigned above**, and
  that is deliberate rather than an omission: each already carries its own cell spec, written before
  this section existed. **Their rows are added by the root, by reading each spec against this suite,
  at the point WSEL-11's re-run unblocks them.** Nothing is guessed here.
- **WSEL-11's re-run moves to 3 seeds** to comply; its original 2-seed pre-registration is superseded,
  which is safe precisely because the re-run is a new run under a corrected objective, not a
  re-reading of the discarded one.

## 3.9 CODE ORGANISATION FOR WIDTH ARCHITECTURES — reuse-first, ONE home, NO duplication (user, 2026-07-21)

**User instruction:** organise the architectures, keep them under the FlexNN umbrella, **no
duplication**, and clean up. **Answer to "are they all in FlexNN?" — NO, and there is already a
duplicate pair.** Inventory below is read off disk 2026-07-21, not recalled.

### Inventory — every width architecture that exists today

| Class | Lives in | Status | Fits variance? |
|---|---|---|---|
| `NestedWidthNet` (Design A: one output weight per unit) | `automl_package/models/flexnn/width/architectures.py:39-111` | package · FAILED-under-joint-training, kept as negative control | yes (`logvar_head`) |
| `SharedTrunkPerWidthHeadNet` (Design B: per-width output layer, `Linear(w_max -> 1)` + masking) | same file `:164-230` | package · **CERTIFIED** | no (dummy zeros) |
| `IndependentWidthNet` (12 disjoint sub-nets) | same file `:114-161` | package · positive control | yes |
| `SharedReadoutPerWidthAffineNet` (shared readout + 2-param per-width affine) | same file `:233-277` | package · minimum-seam arm | no (dummy zeros) |
| `MatryoshkaWidthNet` (per-rung DEDICATED heads, `Linear(k -> 1)`) + `train_matryoshka` | `automl_package/examples/matryoshka_width_net.py:62,111` | **examples · NEVER RUN, never promoted** | **yes** (per-rung `logvar_head_k`) |
| `ResidualCascadeNet` (frozen residual cascade = staged boosting) + `train_cascade` | `automl_package/examples/cascade_width_net.py:80,157` | **examples · built, never compared** | **yes** (additive log-variance, NGBoost parametrisation) |

### The three findings this inventory produces — act on them, do not re-derive them

1. **DUPLICATE PAIR — `MatryoshkaWidthNet` vs `SharedTrunkPerWidthHeadNet`.** Both are "shared trunk,
   a dedicated output layer per width". They differ in ONE implementation detail: Matryoshka's head
   `k` is `Linear(k -> 1)` (sized to the prefix), the certified head is `Linear(w_max -> 1)` reading a
   masked vector, whose columns `>= k` provably cannot influence the output. **Same design, two
   implementations, different nominal parameter counts.** The certified class's own docstring
   (`:170-177`) already records why the masked form was chosen — it keeps exactly one variable moving
   against `NestedWidthNet`. **Consolidation is WSEL-17. Nothing new may be written against
   `MatryoshkaWidthNet` in the meantime.**
2. **THE BOOSTING ARM IS ALREADY IMPLEMENTED — do NOT write a new one.** `ResidualCascadeNet` is the
   staged frozen cascade. Its docstring records the lemma that matters (`cascade_width_net.py:11-14`):
   **a sum of `k` width-1 tanh blocks is EXACTLY a width-`k` single-hidden-layer tanh network**, so
   the cascade's rung-`k` function class equals `NestedWidthNet`'s width-`k` class, plus one extra
   freedom (a per-prefix readout bias). **This is on-disk confirmation of the account in
   `shared/width_transformer_port.md` §1: the cascade is not a third architecture, it is Design A with
   a different training scheme.**
3. **BOTH examples-side classes FIT VARIANCE and are therefore UNUSABLE as written** (§3.7, MASTER
   Decision 2, as clarified 2026-07-21). Neither may enter a comparison until its variance is FIXED at
   the generator's true value rather than learned. That port is scoped inside the task that first
   needs it, never done ad hoc.

### The rules (binding on WSEL-15, WSEL-16, WSEL-17 and anything later)

- **REUSE FIRST — the ladder, rung 2.** Before writing any width net, class or training loop, check
  this inventory. **Extending an existing class is required; a near-copy is a defect, not a style
  choice.** A task that writes a new architecture states in its report which inventory rows it
  checked and why none fits.
- **ONE home per lifecycle stage.** Certified architectures live in
  `automl_package/models/architectures/nested_width_net.py`. **Candidates under test live in exactly
  ONE module, `automl_package/examples/width_candidates.py`** — not one per driver, and not a new
  file per idea. Created by WSEL-15, extended by WSEL-16 (so those two SERIALISE on that file; see
  their deps).
- **Promotion is a task, never a side effect.** A candidate moves from `examples/` to the package only
  via a task whose verify line reproduces the certified reference numbers. This is what
  `MatryoshkaWidthNet` and `ResidualCascadeNet` never got, which is why they are stranded.
- **Every candidate holds sigma FIXED at the true value** (§3.7). A class with a LEARNED variance head
  is ported — the head is clamped and dropped from the optimiser — never wrapped in a driver that
  quietly passes it a free-sigma likelihood.

## 3.10 GENERALIZABILITY TO ARBITRARY WIDTH (user ruling, 2026-07-22)

**Every mechanism this strand tests must be defined for arbitrary `w_max`, never for 12.** The toy
suite fixes `w_max = 12` as the measurement point; 12 is a sample point, not a design constant.
- **No formula, initialisation, schedule or constant may hard-code 12.** Stated generically once:
  sandwich = `{1, w_max}` + 2 random mids · gates init `softplus(nu) = ln(2)/(w_max - 1)` ·
  `params_effective = w_max(w_max+1)/2` · the prefix-norm and stop-grad cumulative sums are
  `O(w_max)` by construction. An implementer finding a bare `12` outside the toy-suite constants
  treats it as a defect.
- **Any cost or parameter count super-linear in `w_max` is reported WITH its scaling law**, not just
  its value at 12. The certified per-width-head reference carries `O(w_max^2)` output parameters
  (`shared/width_transformer_port.md` §3) — exactly why the `O(w_max)` cheap structure (WSEL-16)
  is the deciding question at transformer scale, and why a 12-only verdict would be worthless.
- **A result whose mechanism cannot be stated for general `w_max` is reported as a 12-specific
  observation, never as a design property.**
- Consequence already visible in data: sandwich trains each middle width on ~`2/(w_max - 2)` of
  steps, so mid-width starvation WORSENS as `w_max` grows — the ALL-schedule finding (WSEL-14
  follow-up) matters MORE at scale, not less.

## 4. Tasks

Order: **WSEL-0 → WSEL-1 → WSEL-2 → WSEL-3 → WSEL-4 → WSEL-5 → (WSEL-6 ∥ WSEL-7 ∥ WSEL-11) → WSEL-8 →
WSEL-10.** *(**WSEL-11** added 2026-07-21, MASTER Decision 21 — parallel, independent of WSEL-6/7,
and must land before WSEL-8 reads its numbers.)*

⚡ **TASK ZERO — `flexnn-package.md` FP-11 RUNS BEFORE ANY TASK IN THIS FILE (user, 2026-07-21).**
It moves the flexible-capacity code under one `models/flexnn/` home. **Every width task deps on it**,
because every one of them either edits or imports a file it moves. Doing it first is cheapest: nothing
is in flight to collide with, and four of the tasks below CREATE files that would otherwise have to be
moved afterwards. *(The root first proposed doing it last and was corrected — see FP-11's rationale.)*

**Efficiency/mechanism track, added 2026-07-21 (user, discussion) — runs alongside the order above:
WSEL-12 → WSEL-14, with WSEL-13 parallel to both.**
- **WSEL-12 and WSEL-14 SHARE A WRITE SET** (`kdropout_converged_width_experiment.py` and
  `automl_package/examples/nested_width_net.py`) and are therefore **NOT independent**: they may not be dispatched in
  the same wave, and WSEL-14 must be briefed only after WSEL-12 has merged. Write-set overlap, not
  topic overlap, is what decides this (MASTER, single-writer rule).
- **WSEL-13 is disjoint** (new file only) and dispatches in parallel with either.
- **WSEL-15 → WSEL-16 → WSEL-17 is a SERIAL chain** (added 2026-07-21): WSEL-15 creates
  `automl_package/examples/width_candidates.py`, WSEL-16 extends it, WSEL-17 consolidates the package
  module. Shared write sets — never the same wave. WSEL-16 additionally needs WSEL-13's ordering
  statistic landed, and WSEL-17 needs WSEL-16's winner.
- **Full order for this track:** `FP-11 (task zero) → WSEL-12 → (WSEL-14 ∥ WSEL-15) → WSEL-16 →
  WSEL-17`, with **WSEL-13 parallel to everything after FP-11** and required before WSEL-16 reads out.
- **Every task in this section holds sigma FIXED at the true value — see §3.7. The driver default
  LEARNS it, which is the forbidden thing.**
- **Every task in this section runs the toy tiers assigned in §3.8. No task chooses its own cells.**
- **Reuse before writing: §3.9's inventory is binding. A new nested-width class is a defect.**
- **Both WSEL-12 and WSEL-14 produce a DRIVER; the ROOT runs the grid** backgrounded — a subagent may
  author a sweep but may never own its execution (MASTER §Rules, Environment).
*(**WSEL-9 is ⏸ PARKED** by the 2026-07-20 toys-only ruling and is deliberately absent from this
order — a dispatcher must skip it, not schedule it. Its spec is retained for a possible later pass.)*
WSEL-0 through WSEL-5 are the "fix it properly and completely" phase; no comparison compute runs
until they close. **WSEL-6 and WSEL-7 must precede WSEL-8/WSEL-9**: both fix a *parameter of the
method*, and running the battery before they are settled produces results nobody could attribute —
the failure this strand exists to stop.

### WSEL-0 — single-source the definitions

**Files (write set):** this file **only** (§1 is authored)
**🚫 NOT in the write set: `docs/plans/capacity_programme/MASTER.md`** — ROOT-ONLY (MASTER naming
key, "SHARED FILES ARE ROOT-ONLY"). Three sibling tasks (FP-0, DSEL-0, P0) need MASTER edits in this
same wave; concurrent writers produce contradictory text in one file. **Deliverable instead:** emit
the exact MASTER text — strand-index entry and naming-key entry — verbatim in this task's report.
**Spec:** Add this strand to `MASTER.md`'s strand index with its ownership split from `width-cert.md`
stated explicitly (architecture = width-cert, CLOSED; selection = here). Add a naming-key entry that
**points here and states no definition of its own** — the same discipline applied to the ProbReg
entry. Record as a correction, not a silent edit, that `G-WIDTH = PASS` certifies the architecture
and not any selection mechanism, so no later reader repeats the misreading.
**Non-goals:** no code; no re-opening of the G-WIDTH gate; no edit to `width-cert.md`'s verdict.
*Orchestration:* parallel: no · deps: none · tier: main loop (definitional) · scale: static ·
shape: design · verify: `grep -n "width.md" docs/plans/capacity_programme/MASTER.md` shows the strand
index and the naming key delegating here; `width-cert.md` is unmodified (`git diff --stat` shows it absent).

### WSEL-1 — fix WD1 and WD4 — ✅ **DONE, found already landed 2026-07-20, commit `63ab6bc`**

**Found already landed during the 2026-07-21 repair audit; no dispatch needed.** Both fixes this
task specifies are on disk, an ancestor of HEAD: see WD1/WD4 in §3 for the evidence (predict_uncertainty
override + coverage, and the driver's certified default). The **prove-it-fails ceremony below is
waived, with reason**: the fix and its tests landed in one commit, `63ab6bc`, before this plan was
repaired to reflect it — there is no unfixed tree left to demonstrate the regression test against,
and re-deriving one against a synthetic revert would verify the audit's own reconstruction, not the
original fix.
**Files (write set):** `automl_package/models/flexible_width_network.py` ·
`tests/test_flexible_width_network.py` · `automl_package/examples/kdropout_converged_width_experiment.py`
**Spec:** (i) **WD1** — make `predict_uncertainty` correct for every `uncertainty_method`, or raise
explicitly for the ones it cannot serve; add the missing coverage. (ii) **WD4** — change the driver's
default architecture to the certified `SharedTrunkPerWidthHeadNet`.
**(WD3 is withdrawn — see §3. `width-cert.md` is NOT edited by this task.)**
**Doctrine:** a regression test is not evidence until it has been shown to FAIL on the unfixed code.
Assert on the quantity the fix changes, never on a coarse downstream view. *(Satisfied historically
by commit `63ab6bc`'s own process, not re-verified by this repair.)*
**Non-goals:** no re-run of any certified cell; no change to the G-WIDTH verdict; no refactor of the
inherited base-class uncertainty machinery beyond what WD1 forces.
*Orchestration:* parallel: no (same file as WSEL-2) · deps: none · tier: sonnet high · scale: static ·
shape: execution · verify: `AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m pytest tests/test_flexible_width_network.py -q`
green, THEN revert the WD1 fix, re-run, show the new test FAILING, restore, and show the file
checksum unchanged.

### WSEL-2 — the width selection API — ✅ **DONE, found already landed; verified at the root 2026-07-21, commit `e3cc52b`**

**Found already landed when dispatched 2026-07-21; no code change was needed or made.** The API this
task specifies was built by `flexnn-package.md` FP-3, which carried
`automl_package/models/flexible_width_network.py` in its own write set — the second instance of the
overlap this programme keeps paying for (cf. WSEL-1, DSEL-3, P1, FP-0/FP-7). `git diff --stat
e3cc52b~1 e3cc52b` shows that commit rewriting both this class (+170/−52) and the one named call
site.
**Verify line EXECUTED at the root** (not taken from the worker's report):
`grep -rn "WidthSelectionMethod" automl_package/ tests/` → empty (the enum is gone repo-wide, so
`WidthSelectionMethod.DISTILLED`'s dead `NotImplementedError` path is retired by construction);
`grep -n "inference_mode" automl_package/examples/moe_flexnn_comparison.py` → empty (call site
migrated, `capacity_selection=CapacitySelection.PER_INPUT` + `fit_router()` + plain `predict()`);
`AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m pytest tests/test_flexible_width_network.py -q` → 24 passed.
**One verify clause is not literally satisfiable, and is read the way FP-3's equivalent already was.**
`grep -rn "inference_mode" automl_package/models/flexible_width_network.py` returns TWO hits
(`automl_package/models/flexnn/width/model.py:84`, `:86`) — both inside the constructor's own
`TypeError` message, which names the rejected kwarg back to the caller. No live call site passes it.
This is the identical situation `flexnn-package.md`'s FP-3 completion note ruled on for its clause
(a) (`docs/plans/capacity_programme/flexnn-package.md:652-655`): once a discoverable rejection
message is required, a literal zero-substring grep is unsatisfiable, and deleting the word from the
message to force the grep clean would degrade the error for callers with no behavioural gain.
**Read the clause as "no live call site passes it."**
**The selection-set fraction requirement is satisfied without a change:** `fit_router` takes
caller-supplied `x_val`/`y_val` (`automl_package/models/flexnn/width/model.py:307`), so no split
fraction is baked into this class at all — there was no constant to make configurable.
**`fit()` does not internally call `fit_router()`, and that is CORRECT, not a gap.** The two-call
pattern is the cross-family contract: `automl_package/models/flexnn/depth/model.py:478` and
`automl_package/models/probabilistic_regression.py:906` expose the same separate `fit_router`, and
their tests use the same two-call shape. A width-local auto-fit would be the "second implementation"
this task's own doctrine section forbids.

**Files (write set):** `automl_package/models/flexible_width_network.py` ·
`tests/test_flexible_width_network.py` · call sites found by grep, not by memory
**Spec:** Adopt `CapacitySelection` — built by `flexnn-package.md` FP-3 in `automl_package/enums.py`,
**not built here** — as `FlexibleWidthNN`'s selection API. Migrate every call site onto it: the enum
is passed **at construction**, `fit()` performs whatever held-out selection the chosen mode needs,
and **`predict` loses `inference_mode` entirely for this class** (clean break, not a shim — the repo
has no external users; a shim keeps the silent-failure route alive and will eventually be used by
accident; known call site beyond the class itself:
`automl_package/examples/moe_flexnn_comparison.py:414`). The explicit `width=` override stays, as the
escape hatch for reading the dial directly. If `flexnn-package.md` FP-3 has not already retired
`WidthSelectionMethod.DISTILLED`'s dead `NotImplementedError` path (WD2) globally, retire it here.
**The selection-set fraction must stay CONFIGURABLE on this class**, not a baked-in constant — it is
a parameter of the method and WSEL-6 is about to measure it.
**Doctrine:** the enum and its cross-family contract are FP-3's design to own; this task's job is to
make `FlexibleWidthNN` a correct, complete consumer — not a second implementation. Do not add a
width-local enum, even temporarily, and do not touch `automl_package/enums.py`.
**Non-goals:** no new selection *algorithms* — this changes how the existing mechanisms are reached,
not what they do. No change to `DistilledCapacityRouter` internals. No change to `CapacitySelection`
itself, its name, or its cross-family contract — file a finding against FP-3 instead of patching it
from here.
*Orchestration:* parallel: no (same file as WSEL-1) · deps: `flexnn-package.md` FP-3, WSEL-1 ·
tier: sonnet high (mechanical migration against an already-designed API) · scale: static ·
shape: execution · verify:
`grep -rn "inference_mode" automl_package/models/flexible_width_network.py` returns nothing;
`grep -n "inference_mode" automl_package/examples/moe_flexnn_comparison.py` returns nothing (call
site migrated); `grep -n "WidthSelectionMethod.DISTILLED" automl_package/models/flexible_width_network.py`
returns nothing; `AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m pytest tests/test_flexible_width_network.py -q`
exits 0.

### WSEL-3 — build W-SHARED as library code

**Files (write set):** `automl_package/models/flexible_width_network.py` ·
`tests/test_flexible_width_network.py`
**Spec:** Wire the cheap global read using the **shared selector built by `flexnn-package.md` FP-9**
("the shared selection primitives") — do not re-implement cheapest-within-tolerance or its bootstrap
standard-error helper here, a **third** from-scratch copy is exactly the failure this repair exists
to stop (see the cross-plan dependencies note near the top of this file). This task's own scope is
width-specific plumbing only: score every width on the held-out selection set, feed that error curve
to FP-9's selector at twice a bootstrap standard error (§1), and store the ONE returned width so
`predict` uses it with no caller flag. This is the mechanism `W-SHARED` names and it does not exist
today.
**Doctrine:** the tolerance rule and its estimator are FP-9's to own so width's numbers stay
comparable with the k-selection report's and with depth's, rather than each strand needing its own
caveat. Do **not** re-derive either.
**Anchor warning §3.6 applies**: do not validate this against the same script-level `argmin` it
replaces — they answer different questions and agreement would be meaningless.
**Non-goals:** no per-input logic; no change to the router; no re-tuning of the certified architecture;
no changes to FP-9's selector or SE estimator — file a finding against FP-9 instead of patching it
from here.
*Orchestration:* parallel: no (same file as WSEL-2) · deps: `flexnn-package.md` FP-9,
`flexnn-package.md` FP-3, `flexnn-package.md` FP-4 *(added 2026-07-21 — §1's warning note: W-SHARED
may not be read off the shipping class until FP-4 rules its schedule deviation material or
immaterial)* · tier: sonnet high · scale: static · shape: execution · verify:
`grep -n "cheapest_within_tolerance\|bootstrap" automl_package/models/flexible_width_network.py`
shows a call into the shared primitive, not a local re-implementation; a test constructs a model
whose held-out curve is flat beyond width *w* and asserts the selector returns *w*, not the argmin;
a second test asserts the returned width is stable under a reshuffle of the selection set;
`AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m pytest tests/test_flexible_width_network.py -q` exits 0.

### WSEL-4 — make W-SWEEP a usable reference

**Files (write set):** `automl_package/examples/width_wsel4.py` (Create) ·
`automl_package/examples/capacity_ladder_results/WSEL4/`
**Spec:** The converged per-width sweep already exists
(`automl_package/examples/converged_width_experiment.py`, results
`automl_package/examples/capacity_ladder_results/W_CONVERGED/w_converged_summary.json`) but trains the research module's own
width classes, not `FlexibleWidthNN`. **Reuse it — do not rewrite it.** Port it to train the package
class per width, and reproduce `W_CONVERGED`'s numbers on the original classes first as the
positive control.
**Reproduction criterion (a chosen default — retune if the user wants a tighter or looser bar):
relative error ≤ 2% on `W_CONVERGED`'s reported MSE, per (toy, seed, width) cell.** A verify that lets
the worker pick its own bar is the exact failure this repair exists to fix — this number is it.
**Trajectory discipline (MASTER Decision 9) and optimization-first (MASTER Decision 16) both bind**:
every ported training run reports its full held-out trajectory (not an endpoint), its convergence
flag is trajectory-verified with `hit_cap=False`, and if the port fails to reproduce, the escalation
ladder (LR sweep → clipping → warmup → init scheme → normalization) is exhausted before the port
itself is called broken.
**Doctrine:** MASTER Decision 14 — run the known-good arm first, alone; it must reproduce before any
new number is read. If it does not reproduce, the branch table in §3.5 fires and WSEL-8 halts.
**Non-goals:** do not change the sweep's training schedule; do not "improve" convergence criteria
mid-port — a port that changes the protocol is not a port.
*Orchestration:* parallel: no · deps: `flexnn-package.md` FP-3, WSEL-2, `flexnn-package.md` FP-4
*(added 2026-07-21 — §1's warning note: the ported package class carries the same schedule
deviation until FP-4 rules on it)* · tier: sonnet high ·
scale: dynamic (a sweep) · shape: execution · verify:
`test -f automl_package/examples/capacity_ladder_results/WSEL4/reproduction.json` and
`python -c "import json; d=json.load(open('automl_package/examples/capacity_ladder_results/WSEL4/reproduction.json')); assert all(c['relative_error'] <= 0.02 for c in d['cells']), d"`
exits 0 — every (toy, seed, width) cell within the 2% bar, or the run halts per §3.5; then
`test -f automl_package/examples/capacity_ladder_results/WSEL4/<toy>_<seed>_<width>.json` for the
ported driver's per-cell output, each containing a `held_out_trajectory` key (Decision 9) and a
`hit_cap` key equal to `false`.

### WSEL-5 — charge the cost of selection

**Files (write set):** the accounting module under `automl_package/utils/` (its exact filename is
fixed by `flexnn-package.md` FP-1, which moves it there from
`automl_package/examples/capacity_accounting.py` and leaves that old path as a re-export shim — this
task extends FP-1's landed module, it does not rename or relocate it again) ·
`docs/plans/capacity_programme/shared/metrics-accounting.md`
**Spec:** The accounting module prices a network at a given width; it has no notion of the cost of
*choosing* that width. Add accounting for: router fitting cost, the cheap held-out read's cost, and
W-SWEEP's total training cost. Every model in §1 must be costable end-to-end, which is what makes
the efficiency claim a ratio with a real denominator.
**Doctrine:** this gap is **programme-wide**, not width-specific — depth, joint and MoE draw from the
same module. Build it in the shared module (post-FP-1 location) so the other strands inherit it; do
NOT build a width-local copy and do NOT add new code to the `automl_package/examples/` shim. Update
`metrics-accounting.md`'s S2 scope in the same turn so the spec and the module agree.
**Non-goals:** no wall-clock benchmarking harness (the module is analytic by design); do not extend
to families outside this programme's architectures.
*Orchestration:* parallel: no (shared module, single writer) · deps: `flexnn-package.md` FP-1,
WSEL-3, WSEL-4 · tier: sonnet high · scale: static · shape: execution · verify:
`AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -c "from automl_package.examples.capacity_accounting import *"`
exits 0 (shim still resolves); hand-computed known-answer checks for one small config per mechanism,
asserted in a test, not eyeballed; a test asserts each of §1's three models returns a finite total
cost including selection.

### WSEL-6 — how much data does width selection need?

**Files (write set):** `automl_package/examples/width_wsel6.py` (Create) ·
`automl_package/examples/capacity_ladder_results/WSEL6/`
**Spec:** The 50/50 even/odd selection carve
(`automl_package/examples/kdropout_converged_width_experiment.py:273-276`) was never measured. Sweep
the selection fraction (suggest `{5, 10, 15, 25, 40}%` of the training portion) on the toys where
ground truth is known, for all three arms, holding everything else fixed. Report each arm's quality
against fraction and the fraction at which each saturates. **The arms are not equally exposed**:
W-PERINPUT must learn a *function* from x to width and should be hungriest; W-SHARED and W-SWEEP need
only rank widths on average. If W-PERINPUT loses at a small fraction, "per-input width does not pay"
and "the router was starved" are indistinguishable.
**Emit the frozen-constants artifact §3.6 promises:**
`automl_package/examples/capacity_ladder_results/WSEL6/frozen.json`, containing exactly the two
constants this task owns per §3.6 — the selection-set fraction (the pre-authorised default per §3.5
if no fraction saturates) and the W-PERINPUT data-limited flag, one boolean per (toy, arm).
**Non-goals:** no real data (WSEL-9's budget); no architecture changes (WSEL-7).
*Orchestration:* parallel: yes (disjoint from WSEL-7 if driven by separate scripts) · deps: WSEL-5 ·
tier: sonnet high · scale: dynamic · shape: research · verify:
`test -f automl_package/examples/capacity_ladder_results/WSEL6/frozen.json` and
`python -c "import json; d=json.load(open('automl_package/examples/capacity_ladder_results/WSEL6/frozen.json')); assert 'fraction' in d and 'data_limited' in d, d"`
exits 0; `ls automl_package/examples/capacity_ladder_results/WSEL6/*.json` shows one file per
(toy, seed, fraction, arm); a saturation plot file exists under the same results dir.

### WSEL-7 — is the router's architecture right for width?

**Files (write set):** `automl_package/examples/width_wsel7.py` (Create) ·
`automl_package/examples/capacity_ladder_results/WSEL7/`
**Spec:** The router is fixed at two hidden layers of 32 units, 300 full-batch Adam epochs, lr 1e-2,
labelling tolerance 0.25 (`automl_package/models/flexnn/routing.py:57-60`), constants
inherited rather than chosen. The only existing evidence is Task W6 — one dimension, two settings,
one toy, 3 seeds, binary pass/fail (`width-cert.md:234`, `:237`). **Extend it to a search**: vary
router width/depth (at least half/double/4× hidden, 1 vs 3 layers) and epochs. Establish whether
width's routing conclusions are invariant, and if not, what the router needs.
~~**and the labelling tolerance**~~ — **struck 2026-07-21.** MASTER Decision 18 rules the labelling
tolerance's sensitivity sweep NOT scheduled ("Do not run it pre-emptively"; §1 restates this
verbatim); the swept dimensions here contradicted that ruling. This task sweeps router
hidden/depth/epochs/lr only.
**Emit the frozen-constants artifact §3.6 promises:**
`automl_package/examples/capacity_ladder_results/WSEL7/frozen.json`, containing exactly the
constant this task owns per §3.6 — router hidden/depth/epochs/lr. **Correction, 2026-07-21:** this
line previously also named "the labelling tolerance" as an owned constant; struck along with the
swept dimension above — §3.6's table row for the tolerance is struck too. If this task finds
invariance, the file records the current frozen defaults
(`automl_package/models/flexnn/routing.py:57-60`) rather than inventing new ones.
**Doctrine:** the router stays FROZEN and untuned inside the battery so the W-SHARED/W-PERINPUT
contrast measures selection rather than search effort. **This task does not unfreeze it** — it
establishes whether the frozen choice is defensible, and any change lands as a new frozen default
*before* **WSEL-8 and WSEL-10** run, never per-dataset. **Correction, 2026-07-21:** this line
previously said "before WSEL-9 runs"; WSEL-9 is ⏸ PARKED and will never run. WSEL-8 and WSEL-10 are
the live consumers this timing clause must actually bind (same repair as §3.6's feed-forward rule).
**Non-goals:** no per-dataset tuning of the router, ever. No change to the labelling rule's meaning.
*Orchestration:* parallel: yes · deps: WSEL-5 · tier: sonnet high · scale: dynamic · shape: research ·
verify: `test -f automl_package/examples/capacity_ladder_results/WSEL7/frozen.json` and
`python -c "import json; d=json.load(open('automl_package/examples/capacity_ladder_results/WSEL7/frozen.json')); assert {'hidden','depth','epochs','lr'} <= d.keys(), d"`
exits 0; `python -c "import json; d=json.load(open('automl_package/examples/capacity_ladder_results/WSEL7/frozen.json')); assert 'invariant' in d, d"`
exits 0 (the invariant-or-not verdict is a field, not prose); if `d['invariant']` is `False`, the same
file's `new_default` key is non-null and cited by WSEL-6's re-run.

### WSEL-8 — the W-SHARED ≈ W-SWEEP claim, both halves, on toys

**Files (write set):** `automl_package/examples/width_wsel8.py` (Create) ·
`automl_package/examples/capacity_ladder_results/WSEL8/`
**Spec:** On the existing width toys: train W-SWEEP (one dedicated model per width over the frozen
ladder), score each held out, record the sweep's chosen width under the §1 tolerance rule. Train the
certified dial network and record W-SHARED's chosen width. Report **(a)** quality at matched width
and **(b)** agreement between the two chosen widths — the untested half. Same seeds throughout so the
numbers are comparable.
**Trajectory discipline (MASTER Decision 9) and optimization-first (MASTER Decision 16) both bind**:
every trained model (both the W-SWEEP per-width models and the certified dial network) reports its
full held-out trajectory, its convergence flag is trajectory-verified with `hit_cap=False`, and any
arm that looks like it lost is run through the escalation ladder before being recorded as an
architecture finding rather than an optimization one.
**Doctrine:** MASTER Decision 14 — the known-good arm runs first, alone; here that is W-SWEEP
reproducing WSEL-4's control before any W-SHARED number is read. §3.6's anchor warning binds: the
quality half is anchored on the independently-trained per-width models, not on the certification's
own numbers re-run through this harness.
**Non-goals:** no real data (WSEL-9); no baselines (WSEL-9); no re-tuning of the selector.
*Orchestration:* parallel: no · deps: WSEL-6, WSEL-7, **WSEL-11** *(added 2026-07-21, MASTER Decision
21 — the regularisation confound must be ruled out or reported before this task's numbers are read)*
· tier: **opus xhigh** (a verdict call) · scale: static ·
shape: execution · verify:
`test -f automl_package/examples/capacity_ladder_results/WSEL8/wsweep_control.json` and
`python -c "import json; d=json.load(open('automl_package/examples/capacity_ladder_results/WSEL8/wsweep_control.json')); assert d['reproduces'] is True, d"`
exits 0 **before** any W-SHARED cell is read (Decision 14); then
`ls automl_package/examples/capacity_ladder_results/WSEL8/*.json` shows one file per (toy, seed),
each containing `w_shared_width`, `w_sweep_width`, `held_out_trajectory`, `hit_cap: false`, and a
per-arm `selection_cost` key. **Added 2026-07-21:** the reported numbers come from a split not used
for stopping or selection.

### WSEL-9 — real data + baselines — ⏸ PARKED (real data deferred; spec retained)

**⏸ PARKED, not cut — the spec below is retained verbatim and is not to be deleted.**
**RULING TAKEN 2026-07-20 (user):** `docs/plans/capacity_programme/MASTER.md` Decision 3's real-data
exemption is **NOT extended to WIDTH**. Width stays **toys-only**. The user's ruling explicitly left
the door open to a later real-data pass, so this task is parked rather than removed: the spec stays
on disk, ready to unpark unchanged if that decision is revisited.

**Consequences, binding now:**
- **Nothing depends on WSEL-9.** It is out of the execution order; **WSEL-10 no longer deps on it.**
- **Do not start this task's compute.** A dispatcher must skip it, not schedule it.
- **WSEL-10's report drops its baseline / real-data section** and says plainly that the strand's
  claims rest on constructed targets — it does not silently omit the section.
- Cross-references to WSEL-9 elsewhere in this plan that read "no real data (WSEL-9's budget)" remain
  correct as non-goal statements: real data is out of scope for this strand either way.
**Files (write set):** `automl_package/examples/width_wsel9.py` (Create) ·
`automl_package/examples/capacity_ladder_results/WSEL9/` ·
`docs/width_benchmark/benchmark_spec.md` (Create)
**Spec:** The three models of §1 against the baseline set — **one tree model (LightGBM), a plain
single-output NN (the key control: the dial network at fixed width ≈ this), and linear regression
(the floor, which makes an essentially-linear dataset visible instead of reading as an uninformative
tie)** — on real datasets frozen in the spec. Write the spec first; it is a deliverable of this task,
not a preamble.
**Binding: the driver READS §3.6's constants from their artifacts at startup and FAILS LOUDLY if any
is missing.** No default may be silently substituted. Each results JSON records the constants it ran
under, so any table row traces to the study that justified its settings. **Every arm's number
includes its selection cost** (§1, WSEL-5) — the companion-field pattern of WD5 is forbidden in this
driver's output schema.
**Trajectory discipline (MASTER Decision 9) and optimization-first (MASTER Decision 16) both bind**
on every model trained here, same as WSEL-4 and WSEL-8: full held-out trajectories, trajectory-verified
convergence flags, `hit_cap=False`, and the escalation ladder run before any arm is called an
architecture loss.
*Orchestration:* parallel: no · deps: WSEL-8 · tier: sonnet high · scale: dynamic · shape: execution ·
verify: `test -f docs/width_benchmark/benchmark_spec.md` (spec lands first); then, with one §3.6
constant artifact deliberately hidden/renamed, `AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/width_wsel9.py`
exits non-zero (the prove-it-fails rule) — restore the artifact and re-run; then
`ls automl_package/examples/capacity_ladder_results/WSEL9/*.csv automl_package/examples/capacity_ladder_results/WSEL9/*.json`
shows the per-dataset CSV and per-model JSON outputs, each JSON containing a `constants` key naming
the study artifact it traces to.

### WSEL-10 — report — ⏸ **PARKED (user, 2026-07-21). Gated behind a joint results review.**

⛔ **NO REPORT WORK STARTS UNTIL THE USER AND THE ROOT HAVE REVIEWED THE RESULTS TOGETHER.** The
user's instruction: when all the work in BOTH live strands (width and ProbReg) is done, they will walk
the results with the root to confirm the numbers make sense and that nothing has been missed — **and
only then is the report written.** The purpose is explicit: a comprehensive report is expensive, and
writing one on results that turn out to be wrong or incomplete wastes that effort. **The same gate
binds `probreg.md` P6**; the review covers both strands in one pass, because they share machinery and
a miss in one is likely a miss in the other. See MASTER Decision 23.

**Files (write set):** `docs/reports/width_selection/`
**Spec:** Author via the `research-report` skill, as the user, no AI/tool provenance
(MASTER Decision 10). **The studies are REPORT CONTENT, not internal detail** — they get their own
sections ahead of the comparison, because they are what license its settings:
- **How much data width selection needs (WSEL-6)** — the curve, the saturation point, the chosen
  fraction. Where an arm was still improving at the largest fraction, say so, and say plainly that
  its comparison result is then a floor, not a verdict.
- **Whether the router's architecture matters (WSEL-7)** — the sensitivity table and an invariance
  verdict.
- **Whether the cheap global read reaches the expensive sweep's answer (WSEL-8)** — both halves
  stated separately, because they are different claims and only one was ever previously touched.
- **Real data + baselines — ⏸ SECTION CUT** (user ruling 2026-07-20: width stays toys-only; WSEL-9
  parked). **The report must state this explicitly, not silently omit it**: say that no external
  comparator — tree model, plain single-output NN, or linear floor — was run for this strand, that
  every claim here therefore rests on constructed targets, and that a real-data pass is deferred
  rather than refused. **Do not present a toys-only result as though it had survived a baseline.**

**Honesty clauses, binding:** report W-SWEEP's full cost next to its accuracy (the efficiency claim
is a ratio and this is its denominator); state every constant the battery ran under and which study
set it; carry the negative results — the data floor, any non-invariance — in the body, not an
appendix. **State plainly what `G-WIDTH = PASS` did and did not certify**, so the report does not
propagate the misreading this strand was created to correct.
*Orchestration:* parallel: no · deps: WSEL-8 *(was WSEL-9 — parked by the 2026-07-20 ruling)* ·
tier: **opus xhigh** (main loop) + adversarial cold-read ·
scale: static · shape: execution · verify: the skill's own cold-read gate (procedural, run per the
skill); then `grep -c "WSEL-[0-9]" docs/reports/width_selection/*.md` is nonzero (studies cited by
task ID, not restated from memory); then for each §3.6 constant name,
`grep -q "<constant name>" docs/reports/width_selection/*.md` exits 0.

### WSEL-11 — does explicit regularisation move the selected width? — ✅ **RE-RUN COMPLETE 2026-07-21 AT THE CORRECTED OBJECTIVE. VERDICT: SELECTION DOES NOT MOVE. Battery NOT blocked.** *(The original run remains ⛔ VOID — see the discard notice below, retained as case law.)*

#### ✅ THE RE-RUN'S VERDICT (2026-07-21, fixed-sigma objective, 9/9 cells)

**RESULT: `selection_moved: false`** — ledger `automl_package/examples/capacity_ladder_results/WSEL11/rerun/frozen.json`
plus nine per-cell JSONs (λ ∈ {0, 1e-4, 1e-2} × seeds {0, 1, 2}) in that same directory.

Every cell verified by the root before the verdict was written: `hit_cap: false` on all nine,
`objective == "mse"` on all nine, all 12 widths trustworthy on all nine,
`sigma_treatment == "fixed_at_generator_true_sigma_via_mse_equivalence"`. Within each seed the
selected width is **identical across the whole weight-decay grid** — seeds 0 and 1 select 6 at every
λ, seed 2 selects 7 at every λ. No cell moves beyond tolerance.

**Consequence (MASTER Decision 21):** the strand-local block does **not** fire. **WSEL-8 and WSEL-10
are unblocked on this axis**, and **WSEL-10's report MUST cite this as the robustness note** Decision
21 requires — "does not move" is a citable result, never a silent pass.

**What this does and does NOT establish** *(binding on the report; do not over-claim it)*:
- **Does:** width's cheapest-within-tolerance selection is not an artefact of the research loop being
  unregularised. Decision 21's worry — that small capacity wins because small OVERFITS LESS rather
  than because small SUFFICES — is not operating on this toy, at this grid, under this objective.
- **Does NOT:** establish agreement ACROSS seeds. The three seeds select 6, 6 and 7 — a different
  question this task never asked and its grid cannot answer. **Stability under penalty is not
  stability across seeds; presenting one as the other is the misread to avoid.** *(The voided run hit
  the same spread, 7 and 6 on its two seeds, so this is a property of the toy, not of the objective
  change.)*
- **Does NOT:** generalise beyond the one toy specified. This is a discriminating check by design,
  not a survey.

**Why the re-run is comparable to its own pre-registration:** exactly two things differ from the
voided run — the objective (now fixed-sigma, the entire point) and the seed count (2 → 3, licensed by
§3.8 line 476). The λ grid, toy, w_max, learning rate, convergence gates, selection rule and reported
split are byte-identical, verified line by line at review.

**Depth inheritance remains MOOT for this cycle** (depth is ⏸ PARKED). ProbReg's `P8` is the
remaining live half of Decision 21.

---

#### WSEL-11 re-run — REVIEWED 2026-07-22 (independent adversarial review): CLEAN — the verdict stands

- Re-derived bit-exact from the 9 raw cells (the aggregate is faithful); fixed-sigma objective
  verified in code AND in all 9 provenance blocks; convergence/trust flags clean on all 108
  width-cells; stopping and selection splits disjoint; per-width AdamW correctly scoped; ONE
  bootstrap-SE-calibrated selection rule across every cell — which rules out the
  coarse-dial-guarantees-a-null failure mode by construction.
- **CAVEAT THAT MUST TRAVEL TO THE REPORT (WSEL-10): the null has BOUNDED POWER.** At the selected
  widths, the λ=1e-2 effect is <2% — an order of magnitude below the plateau's natural 10-17%
  width-to-width jitter that the tolerance rule absorbs. The grid is NOT inert: at the elbow (w=4),
  λ=1e-2 moves seed 2's held-out MSE +35% (0.00379 → 0.00512) and shifts its convergence regime
  (stop 33500 → 23500). <!-- source: `automl_package/examples/capacity_ladder_results/WSEL11/rerun/wsel11_mse_lam0.01_seed2.json` vs `automl_package/examples/capacity_ladder_results/WSEL11/rerun/wsel11_mse_lam0.0_seed2.json` -->
  "Regularisation does not move the selection" is true AND weaker than it reads — the report must
  carry the elbow-vs-plateau effect-size contrast, or the null will be over-read.
- Cosmetic, recorded, no action: §3.7's "excluded from the optimiser" is functionally true
  (grad=None ⇒ AdamW skips) but not literal (the spread head's params sit in the list, untouched);
  the 9 cells ran on a dirty tree (ancestor `fd80b39e` of landing `cd9d0e9`), so exact driver bytes
  are not mechanically provable — internal consistency was re-verified instead.
- Persistence gap, noted for FUTURE drivers (not retrofitted into in-flight contracts): the
  per-sample error table the bootstrap-SE selection reads is not persisted (column means only), so
  the selection call cannot be bit-re-derived offline. Drivers authored after this date should
  persist what their selection rule reads.

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

### WSEL-12 — stop recomputing the shared trunk once per width (efficiency defect, user-raised 2026-07-21)

**The defect.** The k-dropout training loop evaluates one width at a time —
`automl_package/examples/kdropout_converged_width_experiment.py:200-201` loops `for k in widths` and
calls `_width_loss(...)`, which reaches `automl_package/examples/nested_width_net.py:131`
`_width_mse` → `forward_width(x, k)`. **Every one of those calls recomputes the shared trunk from
scratch** (`h = self.hidden(x)` inside `forward_width`,
`automl_package/models/flexnn/width/architectures.py:75`). With the sandwich's four widths per
step the trunk is computed FOUR times and discarded three — defeating the entire point of a shared
trunk.

**The fix already exists and is unused by the training loop:** `all_widths_forward`
(`automl_package/models/flexnn/width/architectures.py:81-91`) computes `h` ONCE and reads every
width off it in a single vectorised pass, no python loop. It is currently used for scoring, not for
training.

**Why this matters MORE at scale, not less.** In this toy the trunk is `Linear(1 -> w_max)`, so the
waste is invisible and nobody noticed. In any realistic network the trunk is nearly all of the
compute, so recomputing it once per sampled width is ~Kx the training cost of the whole forward
pass, for nothing. The per-width readout arithmetic (1+2+...+w_max multiply-adds) is the *only*
genuinely unavoidable extra cost of this architecture, and it is trivial by comparison.

**Consequence for schedule choice — this defect has been distorting a design decision.** MASTER
Decision 20 assigns width the sandwich because "each rung costs a real forward." **That premise is
an artifact of this defect, not a property of the architecture.** Once the trunk is computed once,
training EVERY width every step costs ~one forward plus trivial readout arithmetic — so the
all-rungs schedule becomes nearly free for width too, and the efficiency argument for sampling
(sandwich, or the user's one-width-per-batch "width dropout") largely evaporates. **Do not
re-litigate Decision 20 until this is fixed** — measure the schedules against the fixed loop, since
the old timings were measuring the defect.
*(Prior evidence, for whoever picks this up: W5 already found the guaranteed sandwich is NOT
load-bearing for the certified architecture — the uniform-draw ablation reached floor,
`width-cert.md:218-220`, ledger `automl_package/examples/capacity_ladder_results/W_KDROPOUT_CONVERGED/w_kdropout_converged_summary_shared_trunk_mse_n1500_s0.05_schedU.json`.
That ablation drew FOUR widths per step, not one; one-width-per-batch is untested.)*

**Files (write set):** `automl_package/examples/kdropout_converged_width_experiment.py` ·
`automl_package/examples/nested_width_net.py` (or the package architectures module, whichever holds
the loss helper after FP-2's move)
**Spec (execution-level).** Steps, in order:
- [ ] **Step 1 — the equivalence test FIRST, and prove it fails.** Add
  `tests/test_nested_width_single_trunk.py`: build `SharedTrunkPerWidthHeadNet(w_max=6)` under
  `torch.manual_seed(0)`, a fixed `(64, 1)` input and target; compute (a) the current per-width loop's
  summed loss over `widths=[1, 6, 3, 4]` and its `.grad` for every parameter, (b) the single-trunk
  path's; assert losses equal to `1e-12` and every gradient equal to `1e-10` (`torch.allclose`).
  **Prove-it-fails run, recorded in the task report:** deliberately mis-index the readouts (use
  `widths` reversed) and show the test FAILS; restore. A test that passes both ways is not evidence
  (MASTER Corrections, 2026-07-20).
- [ ] **Step 2 — implement.** In `_train_kdropout_to_convergence`
  (`automl_package/examples/kdropout_converged_width_experiment.py:118`), replace the
  `for k in widths: total_loss += _width_loss(...)` accumulation with ONE trunk evaluation whose
  per-width readouts are summed off it. Reuse `all_widths_forward`
  (`automl_package/models/flexnn/width/architectures.py:81-95`) where the sampled set is all
  widths; otherwise add a helper alongside it that computes `h` once and applies the sampled widths'
  readouts. **Ladder rung 2: extend the existing helper, do not write a parallel one.**
- [ ] **Step 3 — instrument cost (additive only).** Record into the summary JSON's `config` block:
  `train_wall_clock_s` (per seed), `trunk_evals_per_step` (int), and `run_provenance`
  (`automl_package/utils/run_provenance.py`, already attached by this driver). Additive fields only —
  no existing field changes.
- [ ] **Step 4 — root re-runs one canonical cell** (`--arch shared_trunk --loss mse`, seeds 0/1/2,
  defaults otherwise) and diffs against the ledger.
**Bit-for-bit equivalence is the bar, not "close enough".**

#### ✅ DONE 2026-07-21 — and the bar above needed correcting. Precision note: what "bit-for-bit" turned out to mean.

**The `1e-10` gradient bar is NOT achievable at float32 by ANY correct implementation**, so it is
verified in float64 (`tests/test_nested_width_single_trunk.py` casts to `.double()`). Computing the
trunk once and reusing one autograd node across the sampled widths changes the reduction order of the
backward matmul relative to recomputing it per width, and IEEE-754 addition is not associative.
Float32 machine epsilon is ~`1.2e-7`, so a `1e-10` gradient bar sits ~1000x below float32's own
resolution. Measured gradient discrepancy: `0.7-1.9e-7` relative — the rounding floor, no systematic
sign. *(The spec also mis-specified its own falsifier: the suggested "reverse `widths`" mutant does
NOT fail the test, because the accumulation sums the same set. A genuine mis-indexing mutant was
substituted, and the test kills dropped-width, mis-paired-head and off-by-one-mask mutants.)*

**⇒ WHAT IS CLAIMED IS ALGORITHMIC IDENTITY, NOT NUMERICAL IDENTITY: the fix computes the same
mathematical gradient by a different summation order.** Evidence: (a) the FORWARD loss is
**bit-identical — difference exactly `0.0`** — at float32 across three width architectures x both
losses x multiple seeds at the canonical shape (`w_max=12`, `n_train=1500`), asserted in the committed
test; (b) gradients agree to `<=4.4e-16` at float64; (c) `IndependentWidthNet`, which keeps the old
per-width loop, agrees to exactly `0.0` everywhere. For calibration, a genuine semantic fault shifts
the step-one loss by **0.9%-4.2%** at this shape — four to five orders above any drift observed here.

**RESULT: measured effect on the canonical cell** — ledger `automl_package/examples/capacity_ladder_results/W_KDROPOUT_CONVERGED/w_kdropout_converged_summary_shared_trunk_mse.json`
(`--arch shared_trunk --loss mse --max-epochs 300000`, seeds 0/1/2; the pre-fix file is recoverable via
`git show cd9d0e9:<that path>` — the driver overwrites this filename in place):

| | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| `fit_bar.ratio_to_floor` | 1.089283878749 → 1.089289780095 | 1.060856868245 → 1.060847129485 | 1.077473572713 → 1.077476009439 | <!-- numcheck-ignore: a deliberate BEFORE→AFTER pair spanning two cells — the pre-fix values live in `…_prewsel12.json`, the post-fix values in the canonical `…_shared_trunk_mse.json`; both are cited by path in this block. No single cell contains both, by construction. -->
| relative movement | 5.42e-06 (up) | 9.18e-06 (**down**) | 2.26e-06 (up) |
| per-width `final_epoch` (12 widths) | identical | identical | identical |
| `pass` / `strong_pass` | unchanged | unchanged | unchanged |

`untrustworthy_seeds` empty before and after. Movement is **bidirectional** and no discrete quantity
moved. **The non-goal "this must not move a single number" is SUPERSEDED**: the fix moves no verdict,
no flag and no discrete quantity, and moves continuous quantities only in the sixth significant figure.

**What a future reader must NOT do with pre-fix and post-fix numbers side by side:**
- **Do not treat a difference at or below ~`1e-5` relative across this boundary as a finding.** It is
  float32 reduction-order noise. Anything quoted to five or more significant figures across it is
  comparing rounding.
- **Do not pool pre-fix and post-fix runs into one sample** and quote a spread or standard error — the
  spread would be dominated by this artifact, not by seed variance.
- **Do not cite a pre-fix number as the reproduction target for a post-fix run.** Reproduction across
  this boundary means identical convergence epochs, identical bar verdicts, and continuous agreement
  to ~`1e-5` — not equality.
- **Do not read the direction of movement as a signal** — it is bidirectional by construction.
- **Conversely, do NOT dismiss a *discrete* change** — a flipped `pass`, a moved `final_epoch`. None
  moved here, and rounding is not an available explanation if one moves later.
- **⚠️ Do not cite this task as measuring an efficiency win.** `trunk_evals_per_step` correctly drops
  `4 → 1` and `train_wall_clock_s` is 53.9 / 36.2 / 70.3 s per seed, but at smoke scale the fixed loop
  is marginally *slower* — the trunk here is `Linear(1 -> w_max)`, so the masking bookkeeping costs
  more than the recompute it removes. **MASTER Decision 20's premise is corrected in principle, not
  demonstrated in wall-clock on this toy.** WSEL-14 must not present timings measured here as evidence
  that the trunk-once argument pays.

**Known gap, accepted, NOT scheduled:** the efficiency crossover point was never measured — no run
exists at a scale where the trunk dominates. Any claim that this fix pays must measure it there.
**Non-goals:** no change to the schedule, the architecture, the loss, or any bar. This is a pure
efficiency fix and must not move a single number. Do NOT add the new schedules here — that is WSEL-14.
*Orchestration:* parallel: yes (disjoint from the studies; **NOT from WSEL-14**, shared write set) ·
deps: none · tier: sonnet high · scale: static · shape: execution ·
verify: (1) `AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m pytest tests/test_nested_width_single_trunk.py -q`
passes AND the prove-it-fails run is shown; (2) the re-run cell's `fit_bar.ratio_to_floor` is unchanged
for every width against
`automl_package/examples/capacity_ladder_results/W_KDROPOUT_CONVERGED/w_kdropout_converged_summary_shared_trunk_mse.json`;
(3) the before/after wall-clock is reported, with `OMP_NUM_THREADS=4` pinned on both sides (thread
count moves this metric by up to ~5% — `shared/fp5-stale-reference-finding.md`).

#### WSEL-12 — REVIEWED 2026-07-22 (independent adversarial review): CLEAN on correctness — two weakens-tier findings, both now discharged on disk

- **Equivalence verified BEYOND the landed test's coverage:** repeat draws (uniform samples WITH
  replacement can duplicate a width in one step — untested by the landed test), heavy repeats, and
  the full ALL schedule, across all 3 shared-trunk architectures × both losses × 3 seeds — forward
  loss bit-identical, float64 gradient max-err ≤ 4e-16. <!-- numcheck-ignore: reviewer-measured via the preserved script, no JSON exists; script = `automl_package/examples/capacity_ladder_results/REVIEW_2026-07-22/equivalence_check.py` -->
  `IndependentWidthNet` correctly falls through to the untouched per-width loop. Ledger A/B
  (`ratio_to_floor`, stop epochs, trust flags) verified against both JSONs; the pre-fix ledger is
  byte-preserved; `cd9d0e9` → `0fba726` is a clean single-commit A/B.
- **Weakens 1 — the spec's own "prove-it-fails: show that" was satisfied in PROSE ONLY** (the
  asserted-not-shown class). The reviewer independently reproduced it (an off-by-one mask mutant:
  7 targeted failures, unmutated architectures stay green). The reproduction scripts are now ON
  DISK — `automl_package/examples/capacity_ladder_results/REVIEW_2026-07-22/` — so the
  demonstration is checkable from the repo alone.
- **Weakens 2 — `trunk_evals_per_step` is an asserted FORMULA, not a measured counter, and no test
  covers it.** Its SANDWICH branch hand-mirrors the training loop's draw logic — correct today,
  silently wrong the day the draw changes. Every value on disk today verified correct. **Queued
  hygiene (WSEL-17 candidate): derive it from one source or pin it with a counter test.**
- **Scope caveat for anyone quoting the fix:** "the trunk is computed once per step" is a
  K-DROPOUT-DRIVER claim. `train_nested_width` (`nested_width_net.py`) still recomputes the trunk
  per width and serves the sinc/hetero/independent drivers — legitimately outside WSEL-12's write
  set, still unfixed. Never quote the fix as codebase-wide.
- Knock-on check: all 15 WSEL-14 and 9 WSEL-15 cells provenance-descend from the fix commit — the
  cross-arm cost comparisons are uncontaminated.

### WSEL-13 — is the induced importance ordering real? — ⛔ **ANSWERED 2026-07-21: NO. `ordering_holds: false`, 0 of 3 seeds, and the correlation runs the OPPOSITE way. A FAIL is a finding, not a bug.**

**RESULT: `ordering_holds: false`** — ledger `automl_package/examples/capacity_ladder_results/WSEL13/frozen.json` (tier 1, seeds 0/1/2; per-cell JSONs and `state_tier1_seed<S>.pt` in the same directory).

| seed | Spearman(index, importance) | primary (`<= -0.5`) | prefix-vs-greedy gap | secondary (`<= 0.10`) | unit 1 vs unit 12 importance |
|---|---:|---|---:|---|---|
| 0 | **+0.524** | FAIL | 0.496 | FAIL | 0.084 vs 6.698 | <!-- source: `automl_package/examples/capacity_ladder_results/WSEL13/wsel13_tier1_seed0.json` -->
| 1 | **+0.881** | FAIL | 0.379 | FAIL | 0.133 vs 6.830 | <!-- source: `automl_package/examples/capacity_ladder_results/WSEL13/wsel13_tier1_seed1.json` -->
| 2 | **+0.580** | FAIL | 0.055 | pass | 0.292 vs 0.384 | <!-- source: `automl_package/examples/capacity_ladder_results/WSEL13/wsel13_tier1_seed2.json` -->

Primary bar: **0 of 3 seeds pass**, and every seed's correlation is **positive** — importance
*increases* with unit index. Secondary bar: mean gap **0.310** against a `0.10` bar. <!-- source: `automl_package/examples/capacity_ladder_results/WSEL13/frozen.json` --> All three cells
`all_widths_trustworthy: true`.

**What this refutes.** The design's mechanistic account — unit `j` receives gradient only from widths
`k >= j`, therefore the summed loss induces a *decreasing* importance ordering — is **wrong on this
toy**, not merely unsupported. Greedy forward selection finds materially better same-size subsets than
the prefix on 2 of 3 seeds, and its picks are **spread across the vector** (seed 0: units 1, 6, 12
first), not prefix-concentrated. §2 and §4 of `shared/width_transformer_port.md` have been corrected
in the same turn, per this task's own rule.

**What it does NOT refute.** G-WIDTH's certification of the architecture is untouched — that gate is
about the dial's behaviour, not about learned feature ordering. Every width still comes off one hidden
evaluation. **"Nested prefix" is, on this evidence, a statement about COMPUTATION SHARING, not about
learned feature ordering.**

**Two caveats that must travel with this result:**
- **A candidate mechanism, NOT established.** The ablation reads the widest head. Narrow heads force
  early units to be independently sufficient (coarse structure), while late units are read by the
  widest head alone and may carry the fine detail only it supplies — which would make
  importance-through-the-widest-head genuinely anti-correlated with index. **No discriminating
  experiment was run. Do not cite this as the explanation.**
- **A known confound in the primary metric.** Single-unit ablation conflates functional importance
  with outgoing-weight magnitude. This does not rescue the design — the prefix-vs-greedy bar is
  scale-free and fails on its own — but the primary bar alone would not separate the two.

**Scope: tier 1 only** (3 cells, not the 6 in §3.8's row). Tier 2 is deferred behind WSEL-15 — see
§3.8's defect block. The refutation therefore rests on the reference cell alone: enough to strike a
claim that was asserted unconditionally, not enough to characterise the ordering in general.

**Not blocked, not re-run.** Per the pre-registration: no re-run on failure, no bar edits after seeing
numbers, and this does not block the strand's battery.

**⚠️ The saved state dicts are LOCAL-ONLY — the spec's "re-runnable without retraining ever again" is
NOT satisfied on a fresh clone.** `WSEL13/state_tier1_seed{0,1,2}.pt` exist on this machine but are
excluded by the repo-wide `*.pt` rule (`.gitignore:29`), so they are not committed. **Left ignored
deliberately rather than force-added** — overriding a repo-wide convention is not a task-level call,
and the cost of the alternative is small (≈1 min/seed to retrain, and the driver is deterministic
under its seeds). Consequence to know rather than rediscover: **a fresh clone must re-run the three
cells before the diagnostic can be re-read.** *(Flagged for the user: force-adding the three files
would cost ~25 KB total and would make the result re-analysable without any retraining.)*

---

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

### WSEL-14 — schedule × bunch size, measured against the FIXED loop, with costs

**Why.** MASTER Decision 20 gives width the sandwich because "each rung costs a real forward". WSEL-12
removes that premise: with one trunk evaluation per step, training every width every step costs ~one
forward plus cheap readout arithmetic. **So the sampling schedules can no longer be justified on
compute for width, and any remaining case for them is a quality/regularisation claim that must be
pre-registered as one** (`shared/width_transformer_port.md` §6). This task measures the axis properly
and, for the first time in this strand, records what each cell COSTS.

**Already measured — do NOT re-run for error, only for wall-clock:** four widths drawn uniformly per
step reached the floor on 3 seeds (`width-cert.md:210-220`; draw count
`automl_package/examples/kdropout_converged_width_experiment.py:80`). **Untested:** one width per batch,
and all widths every step (the latter is what the shipping class already does unvalidated — §1's
warning, `flexnn-package.md` FP-4).

**Files (write set):** `automl_package/examples/kdropout_converged_width_experiment.py` ·
`automl_package/examples/nested_width_net.py:108-116` (the `WidthSchedule` enum) ·
`automl_package/examples/capacity_ladder_results/WSEL14/` (Create by runs)

**Spec (execution-level).**
- [ ] **Step 1 — the schedule axis becomes a parameter, defaults byte-identical.** Replace the
  hardcoded `_UNIFORM_SCHEDULE_DRAW_N = 4`
  (`automl_package/examples/kdropout_converged_width_experiment.py:80`) with a `--uniform-draw-n` CLI
  argument **defaulting to 4**, and add a `WidthSchedule.ALL` member (all widths every step) to the
  existing enum — **extend `WidthSchedule`, never a new enum and never a string literal** (CLAUDE.md:
  closed sets get a type). Existing invocations must produce byte-identical results; show that.
- [ ] **Step 2 — pin the optimiser footgun in a test.** `tests/test_width_schedule_bunching.py`:
  under bunch size 1, assert that a parameter belonging to an unselected width's head has
  `grad is None` after `backward()` (NOT a zero tensor), and that its values are unchanged after
  `opt.step()`. **Prove-it-fails:** with `zero_grad(set_to_none=False)` the second assertion must FAIL
  (the optimiser steps zero-gradient parameters; with weight decay it would shrink heads no batch asked
  to change). Verified mechanism: plain Adam at
  `automl_package/examples/nested_width_net.py:271`, PyTorch 2.10 `zero_grad(set_to_none=True)` default.
- [ ] **Step 3 — cost instrumentation.** Each cell's summary records: `train_wall_clock_s`,
  `steps_to_converge`, `params_allocated`, `params_effective` (the `1+2+...+w_max` triangle — see
  `shared/width_transformer_port.md` §3), and `executed_flops` per width via
  `automl_package/utils/capacity_accounting.py` (reuse; do not re-derive a FLOP formula). Also record
  **train and held-out MSE per width** so the regularisation question is answerable.
- [ ] **Step 4 — the grid (ROOT runs it, backgrounded; the worker only lands Steps 1–3).**
  Cells: bunch size `b in {1, 2, 4, 12(=ALL)}` under `WidthSchedule.UNIFORM`/`ALL`, plus the
  **sandwich** re-run purely for a post-fix wall-clock reference.
  **Sampling, pinned:** `b` widths drawn per step by `torch.randint(1, w_max+1, (b,))` —
  **uniform WITH replacement, no guaranteed inclusion of width 1 or `w_max`**, byte-identical to the
  existing uniform path (`automl_package/examples/kdropout_converged_width_experiment.py:193`) so
  `b=4` reproduces the ablation already on disk. `b=12` means ALL widths deterministically, not 12
  draws — 5 arms × seeds 0/1/2 = **15 runs**,
  canonical cell throughout (`--arch shared_trunk --loss mse --toy hetero`, `n_train=1500`,
  `sigma=0.05`, `w_max=12`), `--tag wsel14_b<N>`, `OMP_NUM_THREADS=4` pinned on every run (thread count
  moves the metric by up to ~5%, `shared/fp5-stale-reference-finding.md`).

**Pre-registered readouts and bars (fixed BEFORE the run).**
- **Fit:** each arm's per-width held-out MSE vs the sandwich reference. Bar: within **10%** relative at
  every width. *(The 2% bar is retired — it was tighter than its own thread-count noise floor;
  `shared/fp5-stale-reference-finding.md`.)*
- **Cost:** wall-clock per step and to convergence, per arm. Prediction on record: after WSEL-12 the
  ALL arm is within ~1.5× of the b=1 arm per step; if it is not, the fix is incomplete.
- **Regularisation:** per-width `train - held_out` MSE gap per arm. Prediction on record: sampling does
  NOT reduce the gap. **This is the arm of the task that answers "does width dropout buy robustness".**
- **Expected failure, recorded in advance so a confirmation is not read as a discovery:** `b=1` should
  under-fit the widest width — the retired per-example draw already did, and it was *gentler* (every
  width still saw gradient from a slice of every batch).

**Non-goals:** no change to the architecture, the toy, the selection rule or the convergence gate; no
new schedule beyond the bunch-size axis and `ALL`; no re-litigation of MASTER Decision 20 in prose —
this task produces the measurement that Decision 20 gets revisited against, by the ROOT, afterwards.
*Orchestration:* parallel: **no — shares a write set with WSEL-12** · deps: **WSEL-12 merged** · tier:
sonnet high (driver) + root (grid) · scale: dynamic (15 cells) · shape: execution ·
verify: (1) `--selftest` PASS and a default-flag run byte-identical to the pre-change output;
(2) `AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m pytest tests/test_width_schedule_bunching.py -q`
passes with its prove-it-fails run shown; (3) 15 JSONs under
`automl_package/examples/capacity_ladder_results/WSEL14/`, each carrying every field in Step 3 and
`hit_cap: false`; (4) an
`automl_package/examples/capacity_ladder_results/WSEL14/frozen.json` with the three readouts above.

#### WSEL-14 FOLLOW-UP — the ALL-schedule cost anomaly — 🔄 PROBE DISPATCHED 2026-07-22 (user-authorized)

- **The pre-registered cost prediction FAILED and nobody followed up at the time.** ALL measured
  ~4.1x b=1 per step (6.8 ms vs 1.7 ms; 2.6x on total wall-clock), against the on-record "within
  ~1.5x or the fix is incomplete". <!-- source: `automl_package/examples/capacity_ladder_results/WSEL14/frozen.json` -->
- **Signed re-read of the fit numbers (root, 2026-07-22, prompted by user review): ALL matches
  sandwich on accuracy and WINS the mids — sandwich's verdict is a COST verdict.** ALL is better by
  18-70% at widths 4-6 and tied elsewhere; its only two bar misses (+10.8% at w3/w9) sit inside
  sandwich's own 33-35% seed spread at those widths. <!-- source: `automl_package/examples/capacity_ladder_results/WSEL14/frozen.json` -->
  The earlier binary reading ("ALL fails the fit bar") was technically correct per the
  pre-registered bar and interpretively misleading; this note is the correction of record.
- **PROBE VERDICT 2026-07-22 — REPRODUCED AND DIAGNOSED: the extra cost is INTRINSIC to the
  per-width-head architecture at toy scale, NOT a residual defect in the trunk-once fix.** Reproduced
  at 4.6-5.0x per step (frozen: 4.06x). The trunk forward is FLAT across schedules (~0.06-0.09
  ms/step at 1 vs 4 vs 12 sampled widths) — WSEL-12's fix works. The growth is per-width work:
  head loop ~29% / backward ~41% / optimizer step ~27% of the ALL per-step total, each ~linear in
  sampled width count; at this net size per-op dispatch overhead dominates FLOPs. <!-- source: `automl_package/examples/capacity_ladder_results/WSEL14/cost_probe/result_orderA_all.json` -->
- **The ~1.5x prediction was wrong in its PREMISE, not its arithmetic:** it assumed a fixed per-step
  cost dominates a small per-width marginal cost. With 12 SEPARATE head tensors (deliberate — the
  design isolates readout sharing), the marginal cost IS the dominant term. Closing the gap requires
  vectorised heads (one `(w_max, w_max)` tensor / fused step). Semantics caveat (reasoned, not
  coded): fused heads are EXACT under ALL (every head gets gradient every step) but CHANGE optimizer
  behaviour under sampling schedules — a fused tensor always carries a gradient, so Adam stops
  skipping unsampled widths, the very footgun WSEL-14 Step 2 pinned. NOT implemented; it is an
  input to the schedule revisit — and the cheap structure (WSEL-16, in flight) vectorises natively,
  one more reason that comparison decides things.
- **Sequencing consequence: the cost question is RESOLVED (intrinsic, diagnosis on disk) ⇒ the
  WSEL-15 FOLLOW-UP confound check is UNBLOCKED** and root-run. The Decision-20 schedule revisit
  reads: ALL = better mids + no overfitting change (above), 2.6x total wall-clock at `w_max=12`,
  cost gap intrinsic at THIS architecture but vanishing under a vectorised/cheap readout.
  **Recommendation drafted for user sign-off — never decided autonomously.**
- Probe artifacts preserved: `automl_package/examples/capacity_ladder_results/WSEL14/cost_probe/`
  (probe script + 6 result JSONs: orders A/B × schedules b1/sandwich/all).
- **AMENDMENT 2026-07-22 (after the WSEL-15 FOLLOW-UP's 5-seed extension): "ALL wins the mids" above
  is refined — at 3-seed means it looked like a mean improvement; per-seed pairing shows the real
  effect is VARIANCE COLLAPSE** (sandwich's mid widths are a 6× per-seed lottery that ALL removes;
  see the WSEL-15 FOLLOW-UP verdict, item 5). The schedule conclusion is unchanged in direction —
  ALL is the schedule that makes mid-width readouts trustworthy — but the mechanism is stabilisation,
  not mean shift.

#### DECISION-20 REVISIT — ✅ RULED BY THE USER 2026-07-22 (supersedes the draft below)

**The ruling (three parts, binding programme-wide):**
1. **ALL-WIDTHS IS THE DEFAULT TRAINING SCHEDULE** for every future width experiment and training
   run. The sandwich survives in exactly ONE labelled role: like-for-like comparisons against
   already-landed sandwich ledgers (either re-run the reference under ALL, or run sandwich and
   label it). Rationale on record: the sandwich's mid-width variance lottery broke 3/5
   certification seeds and mis-graded a gate; ALL costs ~2x wall-clock at w_max=12 — an
   implementation artifact (per-head bookkeeping), not a law.
2. **VECTORISATION OF THE MULTI-HEAD ARCHITECTURE COMES FIRST — before any further multi-head
   compute.** Fused triangular-masked head tensor behind an explicit flag; equivalence test vs the
   per-head path (float-noise tolerance); hard-error on fused+sampling-schedule; per-width best
   snapshots as row slices. Under ALL the fusion is mathematically exact (every head gets gradient
   every step). Gates stage-3's 36 multi-head cells; single-head work proceeds in parallel.
3. **MULTI-HEAD IS RETAINED as a supported architecture REGARDLESS of the comparison outcome** —
   WSEL-16 now selects the DEFAULT RECOMMENDATION, not a sole survivor; WSEL-17's deletion scope
   is amended: the multi-head architecture is never deleted, whatever wins.

*(The draft below is retained for the reasoning record; where it conflicts with the ruling above,
the ruling governs.)*

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

### WSEL-15 — does the nested design survive a normalisation layer? (the transformer-port repairs, measured)

**Why.** `shared/width_transformer_port.md` §5 lists four repairs for the one obstacle that stops this
design porting to a transformer: a normalisation layer computes its statistics over the whole vector,
so truncating to a prefix changes the divisor for every surviving unit. Three of the four repairs are
testable on a toy TODAY; as written they are reasoning with no code behind them. **Our own toy has no
normalisation, so nothing in the current width work would ever surface a problem here.** This task
converts the argument into measurement, on the same toy, the same seeds and the same bars as the rest
of the strand, so the answer is comparable rather than a side-experiment: **does it work, what does it
cost, does it cost accuracy?**

**Files (write set):** `automl_package/examples/width_wsel15.py` (Create) ·
**`automl_package/examples/width_candidates.py` (CREATE — the ONE home for candidate width
architectures, §3.9; WSEL-16 extends it afterwards)** ·
`automl_package/examples/capacity_ladder_results/WSEL15/` (Create by runs)
**Reads (never writes):** `automl_package/models/architectures/nested_width_net.py` ·
`automl_package/examples/kdropout_converged_width_experiment.py`

**Where the candidate architecture lives, and why (boundary rule, MASTER Decision 19; organisation
rule §3.9).** The normalised variant goes in `automl_package/examples/width_candidates.py` — **not in
the driver, and not in a file of its own.** Candidates under test live in exactly one module; the
package holds architectures of record. **It must be a thin wrapper over
`SharedTrunkPerWidthHeadNet`, not a copy of it** — §3.9's inventory already carries one duplicate pair
and this task must not add another. Promotion into the package is WSEL-17's job, gated on this task
passing. Keeping candidates out of the package also keeps this write set clear of the package chain
(FP-4/FP-10/WSEL-3), so this task runs in parallel with them.

**Arms (all on the certified `SharedTrunkPerWidthHeadNet` shape, MSE-only per §3.7):**
- **A — no normalisation.** The certified net, unchanged. The reference.
- **B — prefix normalisation via running totals. EXACT DEFINITION, no discretion:**
  root-mean-square normalisation (**NO mean subtraction**), applied to the hidden vector between the
  shared hidden layer and the per-width output heads. For width `k`:
  `r_k(x) = sqrt( (1/k) * sum_{j<=k} h_j(x)^2 + eps )`, `eps = 1e-5`; the head then reads
  `h_j / r_k` for `j <= k` and `0` beyond. **Divide by `k`, the ACTIVE count — never by `w_max`.**
  Computed for every `k` at once from `cumsum(h^2)`, so one pass covers all widths.
  **NO affine parameters in this arm.**
  *(Why RMS and not mean-centred: mean subtraction needs a second cumulative sum AND interacts with
  the head's bias term, which would move two things at once. Mean-centred normalisation is a DIFFERENT
  question and is explicitly out of scope for this task — §4 non-goals.)*
- **C — B plus a per-width SCALAR scale and shift.** `gamma_k * (h_j / r_k) + beta_k`, with `gamma_k`
  initialised to 1 and `beta_k` to 0 — **2 parameters per width, 24 total at `w_max=12`.**
  **Scalar, not per-unit, and that is the point:** the thing a per-width head might already absorb is
  the rung-dependent **divisor**, which is one number per width
  (`shared/width_transformer_port.md` §5 repair 3). A per-UNIT affine would test channel recalibration
  — a different question, explicitly out of scope.
- **D — naive per-width normalisation.** Identical formula to B, computed the textbook way: loop over
  `k`, slice `h[:, :k]`, compute its root-mean-square directly. **NOT a science arm — it is the
  correctness oracle for B**, and the only arm whose purpose is a test rather than a result.

**Spec (execution-level).**
- [ ] **Step 1 — exactness first: does the trick work at all?** `tests/test_prefix_norm_equivalence.py`:
  assert arm B's per-width outputs equal arm D's to `1e-5` (`torch.allclose`) on a fixed seed and a
  fixed `(64, 1)` input, for every width `1..w_max`, at initialisation and after 10 training steps.
  **Prove-it-fails:** compute the running totals over the FULL vector instead of the prefix, show the
  test FAILS, restore. If this test cannot be made to pass, **stop and report** — repair 2 of the note
  is then wrong and the note must be corrected before anything else in this task runs.
- [ ] **Step 2 — build arms A/B/C/D** in the driver, sharing the toy, the schedule, the convergence
  gate and the selection rule with the rest of the strand. **MSE-only; `--loss mse` explicit; do NOT
  fit a variance** (§3.7).
- [ ] **Step 3 — cost instrumentation, same fields as WSEL-14** so the numbers sit in one table with
  the rest of the width work: `train_wall_clock_s`, `steps_to_converge`, `params_allocated`,
  `params_effective`, and `executed_flops` per width via `automl_package/utils/capacity_accounting.py`
  (reuse; do not re-derive a FLOP formula). Also record **train and held-out MSE per width**.
- [ ] **Step 4 — the grid (ROOT runs it, backgrounded; the worker lands Steps 1–3 only).** Arms A/B/C
  × seeds 0/1/2 = **9 runs** (D is exercised by the Step 1 test, not by the grid), canonical cell
  throughout (`--arch shared_trunk --loss mse --toy hetero`, `n_train=1500`, `sigma=0.05`, `w_max=12`),
  `--tag wsel15_<arm>`, `OMP_NUM_THREADS=4` pinned on every run.
- [ ] **Step 5 — write `automl_package/examples/capacity_ladder_results/WSEL15/frozen.json`** carrying
  `prefix_norm_exact: bool`, per-arm per-width held-out MSE, the three cost fields per arm, and
  `per_width_affine_needed: bool`.

**Pre-registered bars (fixed BEFORE the run).**
- **Does it work:** Step 1's equivalence test passes, with its prove-it-fails run shown. This is the
  load-bearing claim — repair 2 of the note lives or dies here.
- **Accuracy:** arm B's per-width held-out MSE within **10%** of arm A's at every width. *(Same bar as
  WSEL-14; the old 2% bar is retired as tighter than its own thread-count noise —
  `shared/fp5-stale-reference-finding.md`.)* **A degradation IS a finding**, not a failure to fix: it
  would mean normalisation costs accuracy in a nested net, which is exactly what a transformer port
  needs to know.
- **Cost:** arm B within **1.3×** arm A on wall-clock per step. The running-totals trick adds one
  cumulative sum; if it costs more than that, it was implemented as a loop.
- **Is the per-width affine needed:** `per_width_affine_needed = true` iff arm C beats arm B by more
  than 10% relative held-out MSE at any width. **The prediction on record is FALSE** (the per-width
  output head should already absorb it) — recorded in advance so a confirmation is not mistaken for a
  discovery.

**Non-goals:** no transformer, no attention, no real data, no multi-layer net, no variance fitting, no
change to the toy/schedule/selection rule, and **no promotion of the variant into the package** (that
is a later task, gated on this one passing). Repair 4 of the note (a rung-independent normaliser) is
**explicitly OUT** — its literature is unverified and it may not be built on until surveyed.
*Orchestration:* parallel: yes (write set disjoint from every other live task) ·
deps: **WSEL-12 merged** *(so arm A's cost numbers are measured on the fixed training loop and are
comparable with WSEL-14's)* · tier: sonnet high (driver) + root (grid) · scale: dynamic (9 cells) ·
shape: research ·
verify: (1) `AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m pytest tests/test_prefix_norm_equivalence.py -q`
passes with the prove-it-fails run shown; (2)
`AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel15.py --selftest`
PASS; (3) 9 JSONs under `automl_package/examples/capacity_ladder_results/WSEL15/`, each carrying every
field in Step 3 and `hit_cap: false`; (4)
`automl_package/examples/capacity_ladder_results/WSEL15/frozen.json` carries every field in Step 5.

#### WSEL-15 FOLLOW-UP — schedule-starvation confound in the accuracy readout — 📌 QUEUED 2026-07-22 (user-agreed)

- **The observation (root, 2026-07-22, from the user review; CANDIDATE common cause, not established):**
  normalisation's accuracy effect is a REDISTRIBUTION — widths 3-5 improve 33/41/64%, widths 7-11
  degrade 11-39% — and the improvement region coincides exactly with the widths the sandwich schedule
  STARVES (each mid trains on ~2/(w_max-2) of steps; WSEL-14). The normalised arm also converged 43%
  later (32.5k vs 22.7k steps), i.e. its starved mids received substantially more draws. <!-- source: `automl_package/examples/capacity_ladder_results/WSEL15/frozen.json` -->
  **So part of "normalisation helps the mids" may be "longer training feeds the starved mids" — the
  accuracy readout is potentially schedule-confounded.** The wide-end degradation has no such
  alternative account on record.
- **The discriminating check (9 cells):** re-run arms A/B/C × seeds 0/1/2 under `WidthSchedule.ALL`
  (no width starved), everything else identical to the landed grid. Mid-width improvement VANISHES →
  it was schedule starvation; the true cost of normalisation is the wide-end degradation alone.
  PERSISTS → normalisation genuinely helps the mids. Either answer sharpens the transformer port.
- **Sequencing:** blocked behind the WSEL-14 follow-up (the ALL-schedule cost anomaly) — running the
  check on a schedule whose cost is anomalous would tangle two open questions; resolve cost first,
  then this, then the Decision-20 schedule revisit reads both.
- **CORRECTION 2026-07-22 (root): the line originally here — "driver support already exists,
  expected new code ≈ none" — was WRONG, caught by checking the source before running.**
  `width_wsel15.py` pinned the schedule as a module constant (its parent task's explicit non-goal).
  Root added `--schedule {sandwich,all}`: default byte-identical (sandwich, selftest PASS under both
  schedules, ruff clean), and `all` cells write to `WSEL15_ALLSCHED/` so the landed `WSEL15/` grid
  and its `frozen.json` can never be clobbered nor mixed-schedule-aggregated.
- ✅ **ANSWERED 2026-07-22 — 14 cells landed (arms a/b/c × seeds 0/1/2, then a/b extended to seeds
  3/4 when the 3-seed mid-width readout proved inside seed noise). Verdict, per-seed-paired:**
  1. **Normalisation's claimed mid-width benefit is NOT established — it was a seed lottery.**
     Paired per-seed gaps at widths 3–5 flip sign seed to seed (−87% to +185%; norm better on only
     2–3 of 5 seeds). The sandwich grid's 33–64% mean improvements were 3-seed means over the same
     bimodal fit-regime lottery. **Do not cite normalisation as helping the mid widths.** <!-- source: `automl_package/examples/capacity_ladder_results/WSEL15_ALLSCHED/frozen.json` + per-seed cells in the same dir -->
  2. **The wide-end cost IS established:** +13–28% mean at widths 7–10, worse on 4/5 seeds at w7
     and w10, consistent across BOTH schedules. <!-- source: `automl_package/examples/capacity_ladder_results/WSEL15_ALLSCHED/frozen.json` -->
  3. **The convergence cost IS established:** +34% steps under ALL (24.1k vs 18.0k mean), +43%
     under sandwich. <!-- source: `automl_package/examples/capacity_ladder_results/WSEL15_ALLSCHED/frozen.json` vs `automl_package/examples/capacity_ladder_results/WSEL15/frozen.json` -->
  4. **Port conclusion:** prefix normalisation is mechanically exact and per-step free, but carries
     two established costs and no established benefit in a nested net. A transformer port must
     budget for that or test repairs beyond the two tried here.
  5. **Bonus finding, feeds the Decision-20 revisit: the all-widths schedule STABILIZES the starved
     mid widths rather than shifting their mean.** No-norm w5 per-seed spans 0.068–0.406 under sandwich <!-- source: `automl_package/examples/capacity_ladder_results/WSEL15/wsel15_a_seed0.json` .. seed2 -->
     (a 6× lottery) vs a tight band under ALL — five-seed min 0.0549 <!-- source: `automl_package/examples/capacity_ladder_results/WSEL15_ALLSCHED/wsel15_a_seed2.json` -->
     to max 0.0915. <!-- source: `automl_package/examples/capacity_ladder_results/WSEL15_ALLSCHED/wsel15_a_seed1.json` -->
     **The sandwich mid-width problem is VARIANCE, not bias** — any readout that consumes per-width
     profiles at mid widths under sandwich is reading lottery noise.
  6. Integrity: 14/14 cells converged under the cap; one cell (b, seed 3) has width 2 untrustworthy —
     not verdict-bearing (mid-width claims are lottery-labelled anyway; wide-end conclusions hold
     with and without it, checked both ways).
- ⏸ **PARKED (user, 2026-07-22): all further normalisation work** — the two spec-excluded variants
  (mean-centred/true LayerNorm; the width-independent normaliser, literature still unsurveyed) stay
  closed until after the pending architecture work settles which width architecture is best and how
  it compares. Reopen deliberately then; do not fold into any current task.

### WSEL-16 — the architecture comparison: can the CHEAP structure be trained to work? — 🔄 **IN PROGRESS 2026-07-22**

> **STATUS 2026-07-22 (tracking, root):** user authorized an EARLY START during the width review,
> ahead of the wave-2 orchestration (the rest of the strand dispatches after the review). Scope of
> the start: **authoring only — the stage-1 driver, the Step-1 identity test, and the two
> `width_candidates.py` additions (the gate wrapper + the tier-2 weighted loss per §3.8's RESOLUTION
> UPDATE 2026-07-22)** — dispatched to one worker. **The ROOT runs every grid, controls first
> (Step 3), backgrounded.** Stage-2 arms are NOT in the authoring contract (conditional; separate
> contract if `stage2_required`). The PS-3 halt ruling (structure.md) is a separate matter and
> remains pending; it does not gate this task.
>
> **AUTHORING LANDED 2026-07-22 (worker report harvested from its transcript; root-verified):**
> `width_wsel16.py` (driver) · `width_candidates.py` (+`MonotoneGateWidthNet`, +`weighted_squared_error`,
> both generalized to arbitrary `w_max` per §3.10) · `tests/test_stopgrad_width_loss.py` (identity
> test with prove-it-fails INSIDE the test — checkable from the repo alone). Root re-ran all three
> verifications green: pytest 3 passed · full selftest PASS across every arm × tier · ruff clean.
> **ROOT PRE-GRID CHECK — TRAINER PARITY (a confound the selftest did not cover):** tier-1
> non-stopgrad arms train on the REAL `kce` trainer; `A_STOPGRAD` and ALL tier-2 cells on the
> driver's `_train_custom_to_convergence` (the kce trainer cannot express those losses; its file is
> out of the write set). Verified before any candidate cell: single-step loss IDENTICAL, grads agree
> at 3e-08 (float32 eps); full-cell A/B — all 12 stop epochs identical, per-width readouts to 2e-07,
> weight drift 6.7e-05 over 20k steps = accumulated non-associative float noise, five orders below
> the 10% primary bar. **Mixed-trainer comparisons are SAFE.** Scripts preserved:
> `automl_package/examples/capacity_ladder_results/REVIEW_2026-07-22/wsel16_trainer_parity.py` +
> `wsel16_singlestep_probe.py`. <!-- numcheck-ignore: root-measured via the preserved scripts; no JSON artifact exists for a pre-grid check -->
> **INTERPRETATION CAVEAT ON RECORD BEFORE ANY CANDIDATE CELL RUNS (worker-flagged, spec-literal
> implementation):** under the stop-gradient loss the shared readout bias sits inside the detached
> term at EVERY width, so it NEVER receives gradient — frozen at init. If `A_STOPGRAD` misses the
> primary bar, check this mechanism BEFORE concluding "greedy training fails". Not redesigned
> mid-flight; the spec's formula is implemented exactly, and the quirk is documented in
> `stopgrad_all_widths_pred`'s docstring.
> 🔄 **GRID: controls first (Step 3) launched 2026-07-22 — root-run, backgrounded, sequential.**
>
> **USER RULING 2026-07-22 (selection-studies coupling):** the selection studies' MEASUREMENT tasks
> (WSEL-6, WSEL-7, WSEL-8) are ARCHITECTURE-PARAMETERISED: they run on the vectorised multi-head
> AND on any single-head recipe that lands competitive (at or near the 10% primary bar) after
> stages 2–3, with WSEL-8's central claim compared across both. If no single-head recipe is
> competitive, they run multi-head only and single-head is recorded as
> ordering-success/accuracy-failure. The infrastructure tasks (WSEL-3/4/5) are
> architecture-agnostic and are NOT gated on this.

**The question, stated once.** Two structures produce a width dial from one shared hidden layer:

- **Design A — one output weight per unit.** Width-k prediction `= b + Σ_{j<=k} w_j * h_j`. The weight
  on `h_1` is the same number at every width. `NestedWidthNet`
  (`automl_package/models/flexnn/width/architectures.py:39-111`). At `w_max=12`: **13 output
  parameters.** **Trained normally, this FAILS** (MASTER Decision 1).
- **Design B — a separate output layer per width.** Width-k has its own weights on `h_1..h_k`.
  `SharedTrunkPerWidthHeadNet` (`:164-230`). At `w_max=12`: **78 effective output weights + 12
  biases.** **Certified: `G-WIDTH = PASS`.**

**Design A is 6x cheaper in output parameters and it is the structure the ordering theory is written
for** (the running-sum form). Its failure is a TRAINING failure, not a structural one: `w_1` receives
gradient from all 12 width terms and is pulled 12 ways at once — the tug-of-war
(`shared/width_transformer_port.md` §1). **This task asks whether a training change fixes it.**

**Files (write set):** `automl_package/examples/width_wsel16.py` (Create) ·
`automl_package/examples/width_candidates.py` (EXTEND — created by WSEL-15; **this is why the two
serialise**) · `automl_package/examples/capacity_ladder_results/WSEL16/` (Create by runs)
**Reads (never writes):** `automl_package/models/architectures/nested_width_net.py` ·
`automl_package/examples/cascade_width_net.py` · `automl_package/examples/width_wsel13.py` (imports
its ordering statistic — **rung 2 of the ladder: import it, do not reimplement it**) ·
`automl_package/utils/capacity_accounting.py`

**⚠️ REUSE INVENTORY — READ §3.9 BEFORE WRITING ANY CLASS.** Four of this task's six arms need NO new
architecture at all: three are package classes, and the staged-boosting arm is
`cascade_width_net.ResidualCascadeNet`, **already implemented**. Only the gate arm is new, and it is a
thin wrapper. **Writing a fresh nested-width class here is a defect** — the plan already carries a
duplicate pair (§3.9 finding 1) and this task must not add a third.

#### The five stage-1 arms — exact definitions, nothing left to choose

Let `h = hidden(x)` (shape `(N, 12)`), `c_j = w_j * h[:, j]` the per-unit contribution, and
`S_k = b + Σ_{j<=k} c_j` the width-k running sum.

1. **`B_HEADS`** — `SharedTrunkPerWidthHeadNet`, loss `Σ_k MSE(head_k(mask_k(h)), y)`. **THE
   REFERENCE.** Unchanged from the certified run.
2. **`A_JOINT`** — `NestedWidthNet`, loss `Σ_k MSE(S_k, y)`. **NEGATIVE CONTROL — this must FAIL.**
3. **`A_STOPGRAD`** — same structure and same loss shape, one change:
   `Σ_k MSE(detach(S_k - c_k) + c_k, y)`. Each unit is then trained only against what the units
   before it left over; `w_1` feels only the width-1 term. Computed in ONE pass from the cumulative
   sums — no python loop over widths, no staging, no extra forward.
4. **`A_GATES`** — **the ONLY new code in this task**, a thin wrapper over `NestedWidthNet` added to
   `automl_package/examples/width_candidates.py` (never a new nested-width class). Same structure,
   contribution `c_j = g_j * w_j * h[:, j]` with
   `g_j = exp(-softplus(nu) * (j - 1))`, `nu` a **single learnable scalar** initialised so
   `g_12 = 0.5` (i.e. `softplus(nu) = ln(2)/11`). Monotonically decreasing in `j` **by construction**,
   one extra parameter, **no penalty term added to the loss** (the strand's no-arbitrary-penalty rule
   holds). Loss is `A_JOINT`'s. ⚠️ **This is OUR simplification of the published monotone-gate
   mechanism, which derives its gate from a variational bound; we are not reproducing that
   derivation, and the arm must be labelled as a simplification wherever it is reported.**
5. **`INDEPENDENT`** — `IndependentWidthNet`, 12 disjoint sub-nets. **POSITIVE CONTROL / ceiling.**

#### Steps

- [ ] **Step 1 — the stop-gradient identity test, BEFORE any training.**
  `tests/test_stopgrad_width_loss.py`: on a fixed seed and fixed `(64, 1)` input, assert (a) the
  stop-gradient loss VALUE equals the plain summed loss value exactly (`detach` changes gradients, not
  values), and (b) the gradient of `w_1` under stop-gradient equals the gradient of `w_1` from the
  width-1 term ALONE. **Prove-it-fails:** drop the `detach`, show (b) FAILS, restore. If (b) cannot be
  made to pass, the arm is mis-implemented and the task STOPS here.
- [ ] **Step 2 — build the five arms** in the driver, sharing the toy, schedule, convergence gate and
  selection rule with the rest of the strand. **Sigma FIXED at the true value per §3.7 — `--loss mse`
  on tier 1, the fixed-sigma weighted form on tier 2.**
- [ ] **Step 3 — CONTROLS FIRST, ALONE (MASTER Decision 14).** Run `A_JOINT` and `INDEPENDENT` on
  tier 1 only, 3 seeds, before spending anything on the candidates.
  **HALT CONDITIONS — either one stops the task and escalates:**
  (a) `A_JOINT` does NOT fail — i.e. its per-width held-out MSE is within 10% of `B_HEADS` at every
  width. The premise of this task would then be wrong and the failure that motivates it unreproduced.
  (b) `INDEPENDENT` does not reach its certified fit bar against
  `automl_package/examples/capacity_ladder_results/W_KDROPOUT_CONVERGED/w_kdropout_converged_summary_independent_mse.json`.
- [ ] **Step 4 — stage 1 grid (ROOT runs it, backgrounded).** 5 arms × tier 1 + tier 2 × seeds 0/1/2 =
  **30 runs**, `--tag wsel16_<arm>_<tier>`.
- [ ] **Step 5 — record per arm:** per-width held-out MSE · **full-width held-out MSE** · train-minus-
  held-out gap per width · `params_allocated` and `params_effective` · `executed_flops` per width via
  `automl_package/utils/capacity_accounting.py` · `train_wall_clock_s` · `steps_to_converge` · the
  **ordering statistic imported from WSEL-13** — specifically its `spearman_index_vs_importance`,
  computed by the same function on the same three-split carve, so the two tasks' numbers are directly
  comparable · the width selected by BOTH rules (cheapest-within-
  tolerance globally, and the distilled per-input router).
- [ ] **Step 6 — write `automl_package/examples/capacity_ladder_results/WSEL16/frozen.json`** with
  every field in Step 5 per arm, plus `controls_passed: bool`, `stage1_winner: str`,
  `stage2_required: bool`.

#### Pre-registered bars (fixed BEFORE the run; no bar edits after seeing numbers)

- **PRIMARY — full-width accuracy.** `A_STOPGRAD`'s full-width held-out MSE within **10%** of
  `B_HEADS`', on tier 1 **and** tier 2, on **all 3 seeds**. *(This is where greedy training is
  expected to hurt if it hurts at all: every unit fits only the leftover and no unit is ever adjusted
  to work better in the final combination.)*
- **ORDERING.** `A_STOPGRAD`'s ordering statistic at least as strong as `B_HEADS`' on the same
  measure and the same cells.
- **COST.** Report both parameter counts (13 vs 90 at `w_max=12`) and require `A_STOPGRAD`'s
  wall-clock per step within **1.3x** of `B_HEADS`'.
- **DECISION RULE, mechanical:** `stage1_winner = A_STOPGRAD` if PRIMARY and ORDERING both pass;
  else `A_GATES` if it passes both; else `B_HEADS`, and `stage2_required = true`.

#### Stage 2 — CONDITIONAL, runs only if `stage2_required` (tier 1 only, 3 arms × 3 seeds = 9 runs)

Its purpose is to separate **"greedy hurts"** from **"the moving target hurts"** — under stop-gradient
each unit fits a predecessor that is still changing, which staged boosting never does.
- **`A_CORRECTIVE`** — `A_STOPGRAD`, plus: after every 2000 epochs, 200 optimizer steps on the plain
  summed loss (no `detach`), then resume. Removes greediness, keeps the moving target.
- **`A_STOPGRAD_DISTILL`** — `A_STOPGRAD` with the target for every `k < 12` replaced by
  `detach(S_12)`; `k = 12` keeps the true target `y`. Costs nothing extra per step.
- **`A_CASCADE_STAGED`** — **`cascade_width_net.ResidualCascadeNet` + `train_cascade`, ALREADY
  IMPLEMENTED (`automl_package/examples/cascade_width_net.py:80,157`) — do not rewrite it.** The
  literal staged frozen cascade: the upper bound on what strict ordering buys, and the arm that
  isolates the moving target (it has none — each block trains against a converged, frozen prefix).
  **Required port before it runs: it is variance-fitting** (additive log-variance, NGBoost
  parametrisation) and must be run with sigma FIXED at the generator's true value per §3.7. **That port is scoped HERE and nowhere else** —
  add a squared-error stage loss alongside the existing likelihood one, leaving the existing path
  byte-identical, exactly as the width driver carries both. Its per-prefix readout bias is an extra
  freedom versus `A_STOPGRAD` (`cascade_width_net.py:11-14`) and must be named when the two are
  tabulated together.
  *(This arm was previously written as "staged boosting is explicitly OUT". That was wrong the moment
  the inventory was read: it is not future work to be avoided, it is code sitting on disk, and running
  it is cheaper than arguing about whether staging would have helped.)*

#### Stage 3 — the generality check (tier 3, **2 finalist arms ONLY**, 36 runs each = 72 runs)

`B_HEADS` and the stage-1/stage-2 winner, across the data x noise ladder. **Bar:** the winner holds
the PRIMARY bar at **every** ladder cell. A design that wins at `n=1500, sigma=0.05` and loses at
`n=200` or `sigma=0.5` is a cell-specific result and must be reported as one.

**Compute note for the root:** worst case **111 runs** (30 + 9 + 72). Stage 3 alone is the largest
block in this strand. Run backgrounded, land each cell's JSON the moment it is produced, `--config`
one seed per invocation.

**Non-goals:** no real data, no transformer, no multi-layer net, no variance fitting, no new selection
*rule*, no change to the toy suite, no promotion of any candidate into the package (that is WSEL-17's
job and it is gated on this task's outcome), no new nested-width class (§3.9), and **no re-opening of
`G-WIDTH = PASS`** — `B_HEADS` is the reference here, not a defendant.
*Orchestration:* parallel: **no — shares `automl_package/examples/width_candidates.py` with WSEL-15** ·
deps: **WSEL-12 merged** (cost parity on the fixed loop), **WSEL-13 landed** (its ordering statistic is
imported), **WSEL-15 merged** (it creates the candidates module) ·
tier: sonnet high (driver) + root (grids) · scale: dynamic (30 → 6 → 72) · shape: research ·
verify: (1) `AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m pytest tests/test_stopgrad_width_loss.py -q`
passes with its prove-it-fails run shown; (2)
`AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python automl_package/examples/width_wsel16.py --selftest`
PASS; (3) the Step-3 controls are shown to have run FIRST and their outcome recorded before any
candidate cell exists; (4) every JSON under
`automl_package/examples/capacity_ladder_results/WSEL16/` carries every field in Step 5 with
`hit_cap: false`; (5)
`automl_package/examples/capacity_ladder_results/WSEL16/frozen.json` carries every field in Step 6.

### WSEL-17 — THE GRAND CLEANUP: consolidate the architectures, delete what is superseded

> **⚠️ SCOPE AMENDED BY USER RULING 2026-07-22 (Decision-20 revisit, part 3): the MULTI-HEAD
> architecture (`SharedTrunkPerWidthHeadNet`) is NEVER a deletion candidate, regardless of the
> WSEL-16 outcome** — it is retained as a supported architecture ("it may be better on some
> problems"). WSEL-16 selects the default recommendation, not a sole survivor. Deletion eligibility
> below applies to genuinely superseded duplicates/variants only.

**Why.** §3.9's inventory found six width architectures across two directories, one duplicate pair,
two stranded never-promoted candidates, and two classes that fit a variance this strand parks. **User
instruction 2026-07-21: organise them, no duplication, clean up.** This task is the cleanup, and it
runs LAST because promoting the wrong class is worse than leaving the mess.

**Files (write set):** `automl_package/models/architectures/nested_width_net.py` ·
`automl_package/examples/matryoshka_width_net.py` · `automl_package/examples/width_candidates.py` ·
`docs/plans/capacity_programme/shared/PROTECTED.tsv` · tests
**⚠️ Write-set overlap:** touches the package architectures module, so it may NOT run beside FP-2's
successors or WSEL-16. It is last in the strand for that reason too.

**Spec (execution-level).**
- [ ] **Step 1 — resolve the duplicate pair, by MEASUREMENT not by taste.** `MatryoshkaWidthNet`
  (`automl_package/examples/matryoshka_width_net.py:62`, heads `Linear(k -> 1)`) and the certified
  `SharedTrunkPerWidthHeadNet` (heads `Linear(w_max -> 1)` on a masked vector) are the same design.
  Prove they are equivalent in function: assert their width-`k` outputs match to `1e-5` when the
  masked head's columns `>= k` are zeroed, on a fixed seed. **They differ ONLY in nominal parameter
  count**, and `params_effective` (§3.9) already accounts for that. → **Keep
  `SharedTrunkPerWidthHeadNet`** (it is the certified class and the whole width paper trail cites it);
  **reduce `matryoshka_width_net.py` to a re-export shim** naming the equivalence, exactly as
  `automl_package/examples/convergence.py` does. **Move the logic, leave the shim, do not rewrite
  callers** — a shim is not a deletion and passes the protected-path manifest check.
- [ ] **Step 2 — promote the winner, if there is one.** If WSEL-16 named a stage-1/stage-2 winner
  other than `B_HEADS`, move that class from `automl_package/examples/width_candidates.py` into
  `automl_package/models/architectures/nested_width_net.py`, leaving a re-export shim.
  **Promotion requires the verify clause below to reproduce the certified reference numbers** — this
  is the step `MatryoshkaWidthNet` and `ResidualCascadeNet` never got, which is why they stranded.
  If `B_HEADS` won, promote nothing and record that.
- [ ] **Step 3 — record the variance status of every row of §3.9's inventory** in the class
  docstrings, so the next reader cannot pick up a variance-fitting class by accident (the WSEL-11
  failure mode, §3.7 WD7).
- [ ] **Step 4 — leave no litter.** `git status --short` clean of anything this strand created and did
  not intend to keep; every candidate either promoted, shimmed, or explicitly recorded as retained
  for a named future task.

#### Steps 5-7 — the deletion pass (user authorised 2026-07-21: "delete if not needed")

**Why this is safe to do at all, and why it is LAST.** WSEL-16 re-runs every width architecture on the
canonical suite under the sigma-fixed objective, recording cost and ordering fields that were **never
captured before**. Four independent reasons make the old runs non-reusable rather than merely old —
no saved models (so ordering cannot be computed after the fact), no cost fields, a changed tier-2
objective, and a tier-3 ladder that covers 12 cells for `IndependentWidthNet` but only the **4
corners** for the certified design (verified 2026-07-21 by listing
`automl_package/examples/capacity_ladder_results/W_KDROPOUT_CONVERGED/`). So the replacement is
genuine, and only after it exists is anything provably superseded.

- [ ] **Step 5 — build the disposition inventory. ELIGIBILITY IS MECHANICAL, NOT A JUDGEMENT.**
  One row per width driver, architecture module and results JSON. A file is **DELETE-ELIGIBLE only if
  ALL FOUR hold**, each checked by a command whose output is recorded:
  1. it is **not** listed in `docs/plans/capacity_programme/shared/PROTECTED.tsv`;
  2. **no line in any live plan file cites it** (`grep` across the plan dir at execution time — the
     citation gate's own resolver is the reference implementation);
  3. **nothing under `tests/` and no non-deleted module imports it**;
  4. it is **superseded by a NAMED replacement that exists on disk** — the replacement's path is
     written in the row. "Looks old" is not a reason.
  Everything else is **KEEP** or **SHIM**. **Results that are gate evidence are KEEP by rule** — any
  JSON cited by a ledger `RESULT:` marker, or by `docs/width_mse_2026-07-16/verdict_variable_width_mse.md`, <!-- citecheck-ignore: names the gate's own marker, carries no result -->
  backs a passed gate, and deleting it would break the paper trail that makes the gate citable.
- [ ] **Step 6 — write the manifest and STOP.** `shared/wsel17-cleanup-manifest.tsv`: path, verdict,
  reason, replacement path, and the four eligibility checks' outcomes. **The task ENDS here and the
  manifest goes to the user.** The deletion itself is a separate, attended step — the user authorised
  the category, not a blind sweep, and a 30+ file corpus is exactly where an unreviewed rule
  misfires.
- [ ] **Step 7 — attended deletion.** After sign-off: delete in ONE commit whose body is the manifest.
  `git` history retains every file, so this is recoverable; say so in the commit body.

**Non-goals:** no deletion of anything failing any one of the four checks. No deletion of gate
evidence, ever. No behaviour change to any certified class. No new architecture. No merging of
classes that are NOT proven equivalent in Step 1 — a proof, not a resemblance. **Deleting is not the
same as consolidating: a class with live callers is SHIMMED, never removed** (that is what keeps the
pre-registered driver names resolving).
*Orchestration:* parallel: no (package architectures module is single-writer) · deps: **WSEL-16
complete** (its winner decides Step 2 and its re-runs are what supersede the old results) ·
tier: sonnet high for Steps 1-6, **root + ATTENDED for Step 7** · scale: static · shape: execution ·
verify: (1) the Step-1 equivalence assertion passes, with a prove-it-fails run (perturb one head,
show it FAILS); (2)
`AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m pytest tests/test_flexible_width_network.py -q` green;
(3) every importer of the shimmed paths still resolves — re-derive the importer list by `grep` AT
EXECUTION TIME, never from this plan; (4) the canonical cell reproduces `fit_bar.ratio_to_floor`
unchanged for every width against
`automl_package/examples/capacity_ladder_results/W_KDROPOUT_CONVERGED/w_kdropout_converged_summary_shared_trunk_mse.json`;
(5) no path listed in `docs/plans/capacity_programme/shared/PROTECTED.tsv` is deleted or renamed; (6)
`shared/wsel17-cleanup-manifest.tsv` exists and every DELETE row records all four eligibility checks.

---

## 5. Non-goals for this strand

No re-opening of `G-WIDTH = PASS` or of the architecture comparison behind it. No new selection
*algorithms*. No variance-programme work (MASTER Decision 2). No joint width+depth work
(`width-depth.md`). No revival of in-training width selection as a primary (MASTER Decision 13) — it
may appear only as a labelled comparison arm.
