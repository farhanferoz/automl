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
  **DONE/LANDED 2026-07-20**, the same day this file was drafted. `automl_package/utils/capacity_accounting.py:258,283,300`
  *(line numbers re-verified 2026-07-22 after WSEL-5's additions shifted them)*
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
argument below. ✅ **RESOLVED 2026-07-22 (root): FP-4 graded MATERIAL as pre-registered (two-sided
per-seed variance, not systematic degradation), and the remedy is superseded by MASTER Decision 31
— the ALL schedule is the programme default, so the shipping class runs the ratified default.
W-SHARED and W-PERINPUT may be read off it; the binding residue is Decision 31's per-arm schedule
label wherever ALL-trained and sandwich-trained numbers share a table. Numbers + verbatim bar:
`docs/plans/capacity_programme/shared/fp4-schedule-deviation.md`. The single-difference rule
between W-SHARED and W-PERINPUT is untouched (both still read the SAME trained network).**

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
**→ TRIGGER FIRED 2026-07-23 (user, joint results review: "arbitrary & frankly very large
threshold — we need to be objective"). Scheduled as WSEL-22** (sensitivity sweep + a σ-anchored
replacement candidate). This satisfies rather than contradicts the clause above: the sweep is no
longer pre-emptive — the reviewer it waited for has leaned on the constant.

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
- **(b) same choice.** ~~Never tested.~~ **ANSWERED 2026-07-22: NO — 0/3 seeds, the dial network
  always picks wider (9/11/11 vs the sweep's 7/8/6), as a mechanical consequence of its middle-width
  quality gap. See the WSEL-8 result block.** → **WSEL-8**
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
| **WSEL-13** ordering | ✅ | ✅ **landed 2026-07-22** (ALL schedule + weighted objective — its tier-2 corroboration block) | ❌ | **6 done** |
| **WSEL-14** schedule × bunch | ✅ | ❌ | ❌ | 3 per arm × 5 arms = **15** |
| **WSEL-15** normalisation | ✅ | ⏸ **never run — thread parked in full (Decision 35.i)** | ❌ | **9 done (tier 1); 9 parked** |
| **WSEL-16** architectures, stage 1 | ✅ | ✅ | ❌ | 3 per arm × 5 arms × 2 tiers = **30** |
| **WSEL-16** architectures, stage 3 | — | — | ✅ **run single-arm** | **36 done (B_HEADS only — contrast EMPTY by ruling, stage-3 block)** |

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
> **RE-AUTHORIZED 2026-07-22 (user, as part of the ratified wave-2 execution):** the loss landed
> with WSEL-16's authoring; WSEL-13 tier 2 runs in the wave — under the ALL schedule per the
> Decision-20 ruling, labelled as such (its tier-1 cells were sandwich-trained; the schedule
> difference is named wherever the two tiers are tabulated together).
>
> **STALENESS SWEEP 2026-07-23 (root, Decision 35.iv — the table's rows re-derived against disk):**
> WSEL-13's tier-2 obligation is DISCHARGED (corroboration landed 2026-07-22 under the
> re-authorization above); WSEL-15's tier-2 row will never run as pre-registered — the
> normalisation thread is parked in full (Decision 35.i) — and is marked parked, not owed;
> WSEL-16 stage 3 ran 36 cells SINGLE-ARM under the empty-contrast ruling (its stage-3 block),
> not the 72 two-finalist cells this table pre-registered. The clause above ("every other
> tier-2/tier-3 row is still owed") is hereby fully discharged or parked — no row in this table
> is silently owed any longer.

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

**⚡ 2026-07-23 REVIEW WAVE (Decisions 33/34/35) — ✅ REVIEW CLOSED BY THE USER 2026-07-23
(second sitting; all items walked incl. the normalisation re-pass and the WSEL-16/17
walkthrough). DISPATCH AUTHORIZED from this commit.** Order:
**WSEL-24 ✅ DONE 2026-07-23** (verdict in its block: selection_set_size on the small datasets —
causal; rule_objective_mismatch on the large — policy, batched for the user; NO re-sequencing
forced; WSEL-22(c) stays dormant) → in parallel by disjoint write sets **(dispatched at harvest,
2026-07-23)**: WSEL-20 (partial diff in the session scratchpad, `wsel20-partial-work.patch`) ∥
WSEL-22(a) (+ the WSEL-9 uncapped LightGBM cells, zero/tiny compute; *the WSEL-15 affine re-grade
was struck from this slot — user parking ruling 2026-07-23, see the WSEL-15 FOLLOW-UP block*) ∥
WSEL-21 spec ∥ the Decision-33(vi) plan-hygiene pass (chunk 1 landed `f8b7f72`; remaining chunks
after their reader tasks harvest) → **WSEL-6-R (NEW — fired by WSEL-24's causal verdict: the
real-data selection-fraction reopen; spec-gated, dispatches after the trio harvests)** ∥ WSEL-23
candidate specs (adversarial read each) → WSEL-23 grids → WSEL-10 LAST (its replacement-claim
section gated on WSEL-23's end-state; transfer ledger + arbiter reference required).
`flexnn-package.md` FP-13 stays CONDITIONAL on WSEL-21's outcome.

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

### WSEL-4 — make W-SWEEP a usable reference — ✅ **DONE 2026-07-22 (root-verified, both halves)**

> **PORTED HALF DONE 2026-07-22.** The package-class per-width sweep (36 cells, seeds 0/1/2) is the
> go-forward W-SWEEP reference: every cell converged (`hit_cap: false`; two cells needed a raised
> cap, re-run and clean), and the ported/control held-out-MSE ratio spans 0.66–1.18 across the
> ladder — at the noise floor wherever the control reaches it, and up to ~16x BETTER at the
> mid-width cells where the historical control sat in a plateau (consistent with the reference's
> own quarantine flags). Table: `automl_package/examples/capacity_ladder_results/WSEL4/reproduction.json`
> (`ported_vs_control`).
> **TWO CONFOUNDS were caught and removed before the reference was accepted — both are §1
> single-difference violations the inherited calibration smuggled in, and both runs are retained
> on disk as records, never to be tabulated:**
> 1. **Activation** — the package class defaults to ReLU; the control classes are Tanh. As-run,
>    the ported arm under-fit the TRAIN set at every width ≥ 4 (up to 24x vs control). Fixed by
>    pinning Tanh in the driver. Record: `automl_package/examples/capacity_ladder_results/WSEL4/relu_confounded_run/`.
> 2. **Batch regime** — the inherited calibration trains mini-batch (64, shuffled); the control is
>    full-batch. The mini-batch noise floored the fit ~2.4x above the control at w ≥ 6. Fixed by
>    full-batch + a 6000-epoch cap (the probe at the old 1500 cap capped mid-descent). Probes in
>    the session scratchpad; superseded cells: `automl_package/examples/capacity_ladder_results/WSEL4/tanh_minibatch_run/`.
> The final protocol is pinned as the DRIVER DEFAULT (constants block, with the rationale in
> comments), so an unflagged re-run reproduces the reference.

> **CONTROL REPRODUCTION PASSED 2026-07-22 — 36/36 cells within the 2% bar** (max rel err 1.9e-2;
> `automl_package/examples/capacity_ladder_results/WSEL4/reproduction.json`). Two spec-vs-disk
> corrections, discovered in execution and taken as logged reversible defaults (Decision 32):
> 1. **The anchor is the recorded per-width LOG-LIKELIHOOD, not an MSE** — the reference
>    (`automl_package/examples/capacity_ladder_results/W_CONVERGED/w_converged_summary.json`)
>    carries no MSE anywhere (verified: the file contains no `mse` key); this spec's "reported
>    MSE" was written from memory of an artifact that never recorded one. Same 2% bar, applied to
>    the only number the reference actually holds. `held_out_mse` is a NEW field the driver
>    introduces for both arms going forward (no historical anchor exists for it).
> 2. **The blanket `hit_cap: false` verify clause fails on exactly one cell — (seed 1, width 4)
>    — because the REFERENCE ITSELF capped there** (`converged: false, hit_cap: true` at epoch
>    40000 in the historical summary). The port reproduces that cell to 0.63% INCLUDING the
>    cap-hit; a port that "fixed" it would not be a port. 35/36 satisfy the clause as written.
> 3. **Carry-forward caveat:** the reference's own `untrustworthy_seeds: [0, 1]` means some
>    historical cells are not trajectory-certified; reproduction of their numbers validates the
>    DRIVER, and WSEL-8's quality half must not lean on those specific reference cells.
>
> **⛔ RULED at the 2026-07-22 sign-off (item 4, user): the 2%-RELATIVE bar construction is
> REJECTED as arbitrary — relative percentages of a log-likelihood are not meaningful (the LL's
> absolute scale is set by constants unrelated to fit quality). Replacement standard, ratified:**
> **(a) UNITS — reproduction bars on likelihood anchors are stated as an absolute PER-POINT
> log-likelihood difference; for the fixed-sigma Gaussian setting this equals ΔMSE/(2σ²), i.e.
> "agreement to within x% of the irreducible noise floor". (b) TOLERANCE — never hand-picked:
> anchored ≥10× below the smallest difference any consumer of the reference acts on (the 10%
> decision bar ⇒ reproduction noise invisible at decision scale), with the measured-determinism
> precedent (a sibling driver re-ran bit-identically under pinned settings) justifying a tight
> value. Corrections 1-2 above are ratified UNDER THIS BAR, conditionally: a wave-A verification
> step (scheduled; recompute the landed per-cell gaps in the new units from the JSONs on disk —
> no retraining) confirms every cell passes; any breach halts before the merge and escalates.
> This standard applies to FUTURE reproduction bars strand-wide; the WSEL-13 5% plateau
> tolerance and similar pre-registered bars are not re-graded retroactively.**
>
> **✅ WAVE-A VERIFICATION EXECUTED AND RULED (root, 2026-07-22, under the delegated mandate).**
> Two-stage recompute (`WSEL4/bar_recheck.{py,json}`, commits `8b2cafe` proxy → `8b422b9`
> true-σ): under the ratified formula with the generator's TRUE noise variance (σ² = 0.0025,
> verified at `nested_width_net.py:93,144` — common-mode across regions, so genuinely
> x-independent for hetero), **34/36 cells pass; the 2 failures (seed 1, widths 7 and 12; worst
> ratio 3.749) BOTH sit on reference-UNCERTIFIED cells** <!-- source: `automl_package/examples/capacity_ladder_results/WSEL4/bar_recheck.json` (`all_pass`, per-cell `ratio`/`pass`/`reference_seed_untrustworthy`) --> (`untrustworthy_seeds: [0, 1]` read
> from the reference itself; zero failures on trustworthy seeds). **Ruling: corrections 1-2
> stand RATIFIED; the two failing cells are attributable to the reference's own uncertified
> seed-1 trajectories (the port's runs ARE trajectory-certified), fall under carry-forward
> caveat 3 (already banned from downstream use), and are additionally annotated in
> `bar_recheck.json`. The item-4 MERGE HOLD IS LIFTED.** The earlier proxy construction
> (σ² := achieved MSE) is retained in the artifact as `proxy_*` columns for the record; it
> over-fails underfit widths and is not the standard.

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

### WSEL-6 — how much data does width selection need? — ✅ **ANSWERED 2026-07-22 (primary grid): 15% of the training portion suffices for every arm; NO arm is data-limited. Router-dependent-arm re-run at WSEL-7's `new_default` still owed.**

> 🔄 **REAL-DATA HALF REOPENED 2026-07-23 (WSEL-24's causal verdict): "15% suffices" is TOYS-ONLY
> and does NOT transfer.** On the battery the 15% carve yields selection sets of 37/53/92 rows on
> yacht/diabetes/energy, and WSEL-24's probe demonstrated causally that enlarging the SELECT carve
> ALONE — rule, multiplier and estimator identical — flips the pick to correct on diabetes and
> yacht. Scheduled as **WSEL-6-R** in the wave line (§4), spec-gated. This block's toys verdict is
> unchanged; its scope label narrows to toys.

**RESULT: `fraction: 0.15`, `saturated: true`, `data_limited` all false** — ledger `automl_package/examples/capacity_ladder_results/WSEL6/frozen.json` (90 cells: 2 tiers × 3 seeds × 5 fractions × 3 arms; per-cell JSONs + `saturation.png` in the same directory; per-(tier,seed[,width]) cached nets under `WSEL6/_cache/`).

| tier | arm | smallest fraction within 2·bootstrap-SE of its best | saturated |
|---|---|---:|---|
| 1 (hetero) | W-SHARED | 5% | yes | <!-- source: `automl_package/examples/capacity_ladder_results/WSEL6/frozen.json` `per_tier_arm_fraction_choice` -->
| 1 (hetero) | W-PERINPUT | 5% | yes | <!-- source: `automl_package/examples/capacity_ladder_results/WSEL6/frozen.json` -->
| 1 (hetero) | W-SWEEP | 5% | yes | <!-- source: `automl_package/examples/capacity_ladder_results/WSEL6/frozen.json` -->
| 2 (hetero3) | W-SHARED | 10% | yes | <!-- source: `automl_package/examples/capacity_ladder_results/WSEL6/frozen.json` -->
| 2 (hetero3) | W-PERINPUT | 15% | yes | <!-- source: `automl_package/examples/capacity_ladder_results/WSEL6/frozen.json` -->
| 2 (hetero3) | W-SWEEP | 5% | yes | <!-- source: `automl_package/examples/capacity_ladder_results/WSEL6/frozen.json` -->

**Findings.** (a) Every (tier, arm) pair saturates INSIDE the sweep — the frozen constant is the max
over pairs, 15%, bound by the noisy tier's per-input arm; the historical 50/50 selection carve
(`kdropout_converged_width_experiment.py:273-276`) therefore carries ≥3× more selection data than
selection needs on these toys. (b) **W-PERINPUT is NOT router-starved at the frozen fraction** — the
§2 worry ("per-input width does not pay" vs "the router was starved" indistinguishable) is resolved
within the swept range: every `data_limited` flag is false, so a battery loss is attributable to the
arm, not to selection-data starvation. (c) The exposure ordering came out as §2 predicted: the
per-input arm is the hungriest (15% on the noisy tier), the rank-on-average arms saturate at 5-10%.
(d) Run integrity: 4 of 78 cached nets hit the w4 ported protocol's 6000-epoch cap (the tier-2
seed-0 shared net; one middle-width sweep net in three other groups) and were retrained under a
raised cap (actual epochs 6312-8983, all patience-stopped, none capped); the capped originals are
preserved under `WSEL6/capped_at_6000/` and `WSEL6/_cache/capped_at_6000/`; all 90 primary cells are
`trustworthy: true`. <!-- numcheck-ignore: the four actual-epoch counts live in the four rebuilt `automl_package/examples/capacity_ladder_results/WSEL6/_cache/*_meta.json` files, one per file -->
**Protocol notes.** Trains the package class on RAW x/y per WSEL-4's vetted ported protocol — name
this if these numbers ever share a table with the standardizing drivers (wsel13/wsel16). The router
ran at the CURRENT frozen defaults (`automl_package/models/flexnn/routing.py:57-60`).

#### The §3.5-mandated router re-run — DONE 2026-07-22: the candidate router shows NO end-task gain

**RESULT: 30/30 W-PERINPUT cells re-run at WSEL-7's `new_default` (64×3, 600 epochs, lr 0.01), all trustworthy, zero guards** — ledger `automl_package/examples/capacity_ladder_results/WSEL6_RERUN_ROUTER/frozen.json` (per-cell JSONs + `saturation.png` in the same directory; each cell records `router_config` with `source` = WSEL-7's frozen.json; nets reused read-only from the primary grid's cache — training is identical by construction, the router is the single difference).

Head-to-head, mean held-out MSE across seeds, candidate/frozen ratio (>1 = candidate WORSE):

| tier | 5% | 10% | 15% | 25% | 40% |
|---|---:|---:|---:|---:|---:|
| 1 (hetero) | 1.129 | 1.122 | 1.013 | 1.031 | 0.903 | <!-- numcheck-ignore: ratios of seed-means derived across the per-cell `held_out_mse` values in `automl_package/examples/capacity_ladder_results/WSEL6/` and `automl_package/examples/capacity_ladder_results/WSEL6_RERUN_ROUTER/`, stored in no single file -->
| 2 (hetero3) | 1.036 | 1.083 | 1.018 | 1.014 | 1.026 | <!-- numcheck-ignore: same derived ratios, noisy tier -->

- **On the noisy tier the candidate is consistently slightly worse (5 of 5 fractions), on the easy
  tier mixed** (worse at small fractions — consistent with a ~3.6× bigger router overfitting a small
  selection subsample — parity mid-range, better only at the largest fraction). **There is no
  fraction at which the candidate improves the noisy tier, and adoption would need evidence of
  improvement.**
- **Recommendation carried to the user review alongside WSEL-7's caveat block: KEEP the frozen
  default (32×2, 300 epochs).** WSEL-7's plateau gain on router-fit quality ratios does not
  transfer to the end-task decision that actually consumes the constant. This is the discriminating
  experiment the caveats asked for, not a re-grade of WSEL-7's registered verdict.
- The primary grid's frozen constants (`fraction: 0.15`, no arm data-limited) remain the binding
  §3.6 values; the re-run's own arm-only summarize (both per-input pairs saturating at the smallest
  swept fraction on a flatter, slightly worse curve) is a labelled sibling readout, never pooled.

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

### WSEL-7 — is the router's architecture right for width? — ⛔ **ANSWERED 2026-07-22: NOT INVARIANT under the pre-registered rule. REVIEWED & CLOSED at the 2026-07-22 sign-off: the frozen default STAYS; six user rulings recorded below; the follow-on experiment is WSEL-19.**

> 📐 **RE-GRADED 2026-07-23 under the noise-aware 2·SE rule (WSEL-20; root-applied at harvest;
> ledger regenerated in place with every original leaf preserved).** The NOT-INVARIANT headline
> **stands but NARROWS to depth alone**: layers' plateau gap exceeds its noise band while hidden
> and epochs land WITHIN theirs (epochs borderline) and lr was invariant under both rules — per-dimension numbers in the harvest note of the WSEL-20 block and in `automl_package/examples/capacity_ladder_results/WSEL7/frozen.json` (`per_dimension.*.gap_to_default_at_plateau` vs `twice_se_at_plateau`).
> `invariant` (field of record) = false under 2·SE; the flat-5% grading survives as
> `invariant_5pct_historical`, never overwritten. The frozen default STAYS — the sign-off ruling
> is unchanged; WSEL-10's router section reads THIS grading.

**RESULT: `invariant: false`, `new_default = {hidden: 64, depth: 3, epochs: 600, lr: 0.01}`** — ledger `automl_package/examples/capacity_ladder_results/WSEL7/frozen.json` (78 cells: 4 dimensions × 13 values total × 2 tiers × 3 seeds; per-cell JSONs in the same directory).

| dimension | frozen default | plateau value | mean ratio at plateau (± SE, 6 cells) | invariant (5% rule) |
|---|---:|---:|---:|---|
| hidden | 32 | 64 | 0.9516 ± 0.0355 | FAIL | <!-- source: `automl_package/examples/capacity_ladder_results/WSEL7/frozen.json` -->
| layers | 2 | 3 | 0.9077 ± 0.0387 | FAIL | <!-- source: `automl_package/examples/capacity_ladder_results/WSEL7/frozen.json` -->
| epochs | 300 | 600 | 0.9024 ± 0.0503 | FAIL | <!-- numcheck-ignore: mean is `ratio_to_default_by_value['600']` in `automl_package/examples/capacity_ladder_results/WSEL7/frozen.json`; the SE is derived across the six per-cell JSONs, stored in no single file -->
| lr | 0.01 | — | 0.9884 ± 0.0118 | pass | <!-- source: `automl_package/examples/capacity_ladder_results/WSEL7/frozen.json`; SEs recomputed from the per-cell `quality_ratio_to_default` values -->

Ratio < 1 = variant beats the frozen default on routed held-out squared error (verified in
`width_wsel7.py`, `quality_ratio_to_default = variant_quality / frozen_default_quality`, lower better).

**Caveats recorded WITH the verdict (root, 2026-07-22 — batched for end-of-run user review):**
- **The 5% plateau tolerance was worker-chosen at authoring** (module docstring: "retune
  `_PLATEAU_REL_TOL` if the user wants a tighter or looser bar") and was flagged at authoring for
  review if leaned on. It is now leaned on. It is the pre-registered bar of record, so the verdict
  above is NOT re-graded post hoc — but under the twice-standard-error rule (§3.5's own noise-aware
  rule for the sibling WSEL-6 decision) only **depth** clears, marginally (gap 0.0923 vs 2·SE <!-- numcheck-ignore: gaps and 2·SE values derived across the six per-cell `quality_ratio_to_default` values per dimension under `WSEL7/`, stored in no single file -->
  0.0774); hidden (0.0484 vs 0.0710) and epochs (0.0976 vs 0.1006) do not. <!-- numcheck-ignore: same derived gaps/2·SE values, continuation of the line above -->
- **The epochs gain is tier-1-concentrated**: tier-2 cells at 600 epochs are ≈ parity (1.029, 1.016, <!-- numcheck-ignore: the six `quality_ratio_to_default` values live one-per-file in `WSEL7/wsel7_tier{1,2}_seed{0,1,2}_epochs_600.json`; the per-line checker verifies against a single file -->
  0.972) while tier-1 cells drive the mean (0.744, 0.882, 0.773). Not tier-consistent. <!-- numcheck-ignore: continuation — same six per-cell values -->
- **`new_default` combines one-factor winners never measured jointly** (each dimension was swept with
  the others held at the frozen default); the combined 64×3×600 configuration has no cell of its own.
- **tier-1 seed-0 is a recurring outlier in both directions** across dimensions (e.g. layers-1 ratio
  1.588, hidden-128 ratio 0.730), consistent with a noisy same-cell default-router baseline. <!-- numcheck-ignore: the two ratios live in different files (`automl_package/examples/capacity_ladder_results/WSEL7/wsel7_tier1_seed0_layers_1.json`, `automl_package/examples/capacity_ladder_results/WSEL7/wsel7_tier1_seed0_hidden_128.json`); the per-line checker verifies against a single file -->

**Consequence, per §3.5 (executed, not reinterpreted):** WSEL-6's router-dependent arm (W-PERINPUT
only — W-SHARED/W-SWEEP construct no router) is re-run citing `new_default`, into a separate results
dir so the primary grid's cells are never pooled with re-run cells. **Re-run DONE 2026-07-22 — the
candidate shows NO end-task gain (slightly worse on the noisy tier at every fraction); see WSEL-6's
re-run block for the head-to-head and the keep-the-frozen-default recommendation.**

> **⛔ USER RULINGS AT THE 2026-07-22 SIGN-OFF REVIEW — router disposition SETTLED; sign-off
> item CLOSED. Six rulings: 1-3 recorded at the first sitting, 4-6 at the second sitting the
> same day.**
> 1. **The frozen default STAYS (32×2, 300 epochs, lr 0.01). `new_default` is NOT adopted** — the
>    registered sweep verdict stands as recorded, but the discriminating re-run showed no end-task
>    gain and the user ratified keeping the constant where it is.
> 2. **Router architecture must be INPUT-SIZE-RELATIVE (user).** The router MLP already infers its
>    input layer from the data (`routing.py` builds `_CapacityRouterMLP(x_arr.shape[1], ...)`), but
>    every capacity knob (hidden sizes) is an absolute constant chosen on 1-feature toys. Binding:
>    any future router-architecture specification is a RULE parameterised by input dimensionality;
>    the current constants are only that rule's validated instance for 1-D inputs. No task may
>    freeze new absolute router constants without stating the rule they instantiate.
> 3. **Regularisation / overfitting avoidance for the router is a FIRST-CLASS REQUIREMENT (user),
>    currently unmet** — verified in code: `DistilledCapacityRouter._fit_from_targets` is plain
>    full-batch Adam for a fixed epoch count, no early stopping, no validation split, no weight
>    decay, no dropout. The re-run's overfitting signature (bigger router worse on 75-point
>    selection sets) is the motivating evidence. Implementation lands via the router module's
>    owning strand (`flexnn-package.md` — `routing.py` is outside every WSEL write set); this
>    strand's studies re-read the constants after any such change per §3.6's feed-forward rule.
> 4. **Metric ruling (user): router candidates are ranked on the SAME metric the underlying model
>    trains on, evaluated OUT-OF-SAMPLE — no size/complexity penalty enters the metric.** The
>    held-out error table is pure per-input error; smallness enters only through the declared
>    cheapest-within-tolerance tie-band (`DEFAULT_TOLERANCE`, `routing.py:77`), which stays
>    declared, not tuned (MASTER Decision 18 rules its sensitivity sweep not scheduled). Matches
>    the implementation as-is — no code change.
> 5. **Ruling 2's generalizability requirement may be met EITHER by an input-size-relative sizing
>    rule OR by a router backend with no size-scaling problem (gradient-boosted trees) — decided
>    EMPIRICALLY, not by fiat: the WSEL-19 router-backend bake-off, ratified at this sign-off
>    (task block below).** Blending is part of that test, never hand-waved (user): every arm runs
>    in both routing modes — hard (one width per input) and probabilistic (per-width outputs
>    blended by the router's class probabilities) — with deployed compute reported next to
>    quality.
> 6. **Ruling 3 concretised (user-approved): early stopping on an internal validation split is
>    MANDATORY; mild weight decay is the default; dropout is EXCLUDED** (wrong regime for a tiny
>    router on small selection sets). A tree backend satisfies this natively (early stopping +
>    depth limits + shrinkage + subsampling).

No global freeze from this
strand (`flexnn-package.md` FP-5 owns `routing.py`). **WSEL-8 is unaffected: its spec's output
contract carries `w_shared_width`/`w_sweep_width` only — no router-consuming arm.** Whether
`new_default` becomes the strand's settled per-dial default is decided at user review with this
caveat block and the re-run's outcome on the table. **Decided 2026-07-22: it does not — ruling 1
above.**

> **📌 2026-07-23 REVIEW ADDENDA (user, joint results walkthrough) — two amendments, both given
> plan slots at the moment of decision:**
> 1. **The noise-aware bar is ADOPTED, not merely recorded.** User directive: "just recording is
>    not good enough. We need to adopt it, test it, validate it." The 2·SE rule becomes the
>    invariance verdict of record for this sweep; the 5% grading is retained as history, never
>    overwritten. Scheduled as **WSEL-20** (task block below). Under the re-grade the operative
>    conclusion STRENGTHENS ruling 1: three of four dimensions are invariant within noise, depth
>    is marginally significant, and the discriminating end-task re-run (WSEL-6's re-run block)
>    already validates the no-adoption outcome independently.
> 2. **Rulings 3/6 AMENDED: "mandatory early stopping" is demoted to CONDITIONAL.** The bake-off
>    measured ruling 6's recipe and it never won unconditionally (WSEL-19 findings 2: starved-cell
>    inversion at d=1 and d=2-axis, oblique reversal, d ≥ 8 protocol-blocked). The implementation
>    task ruling 3 delegated to the router's owning strand was NEVER FILED there (verified by grep
>    2026-07-23 — a scheduling miss under "runs must be scheduled"); now filed as
>    **`flexnn-package.md` FP-13 — CONDITIONAL: unblocks only if the d ≥ 8 training-protocol
>    escalation shows the regularised recipe winning there.**

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

### WSEL-20 — adopt the noise-aware (2·SE) invariance rule — ✅ **DONE 2026-07-23 (worker-built, root-harvested from disk): WSEL-7 re-graded — the NOT-INVARIANT headline STANDS but NARROWS to depth alone; hidden/epochs deviations were inside noise. WSEL-10's router section unblocked.**

> ✅ **HARVEST NOTE (root, 2026-07-23).** The worker (`wsel20-regrade-worker-2`) went idle without
> delivering its report — the known named-teammate report failure; the work was harvested from
> DISK per doctrine, not re-dispatched. Root verification: ruff clean · `--selftest` PASS ·
> leaf-preservation diff against the committed original ledger: **0 leaf paths missing, 0 values
> changed except provenance timestamp/commit** · `new_default` byte-identical. Re-graded verdict
> (per dimension, gap vs its 2·SE band): layers 0.0923 > 0.0774 NOT invariant; hidden 0.0484 < 0.0711 invariant; epochs 0.0976 < 0.1006 invariant (borderline, recorded as such); lr invariant under both rules; field-of-record `invariant` = false. <!-- source: `automl_package/examples/capacity_ladder_results/WSEL7/frozen.json` -->
> **ROOT CATCH, recorded per the no-guessed-citations rule: the verify one-liner this block
> originally carried was WRONG AS WRITTEN** — it asserted `ratio_to_default_by_value` within the
> top two key levels, but that key lived at the THIRD level in the ORIGINAL ledger too, so the
> command could never have passed even before the change. The check below is the corrected,
> recursive form (dated); the failure was the check's, not the work's.

**Why.** WSEL-7's caveat block computed the noise-aware grading and left it as prose. The user
ruled that insufficient ("adopt it, test it, validate it"): one statistical rule for the strand,
executable in code, not narrated in caveats.

**Files (write set):** `automl_package/examples/width_wsel7.py` ·
`automl_package/examples/capacity_ladder_results/WSEL7/` (regenerated ledger). Worker-dispatchable;
the WSEL-7 re-grade addendum in THIS file is ROOT-applied at harvest (plan docs are root-only).

**Spec:**
- **(a)** `--summarize` grades per-dimension invariance under the 2·SE rule: a dimension fails
  invariance only if the plateau value's mean quality-ratio gap to the frozen default exceeds
  twice the standard error of that mean across the dimension's cells. The `invariant` field (the
  field of record, asserted by WSEL-7's verify line) is graded under this rule. The 5% grading
  survives as `invariant_5pct_historical` (+ the existing `plateau_rel_tol`) — retained, never
  overwritten. Per-dimension gaps and SEs land as ledger fields, so the caveat block's derived
  numbers become citable leaves instead of numcheck-ignored prose.
- **(b)** regenerate `automl_package/examples/capacity_ladder_results/WSEL7/frozen.json` from the existing 78 per-cell JSONs — **NO retraining,
  no new cells**; every key existing citations consume (`ratio_to_default_by_value`,
  `new_default`, `hidden`/`depth`/`epochs`/`lr`, `invariant`) is preserved.
- **(c)** binding, strand-wide: any future plateau/invariance decision uses the noise-aware rule;
  no new flat-percentage bar may be introduced without a stated noise argument. (This generalises
  §3.5's existing 2·SE rule from the data-fraction decision to all threshold decisions.)
- **(d)** validation for this instance is already on record — the discriminating end-task re-run
  (WSEL-6's re-run block) reached the same operative outcome independently.

**Non-goals:** no retraining; no change to WSEL-7's historical verdict text (the re-grade is an
addendum); no edit to `routing.py`; no new toy cells.
*Orchestration:* parallel: yes (disjoint write set) · deps: none (all inputs on disk) · tier:
sonnet high · scale: static · shape: execution · verify:
`OMP_NUM_THREADS=4 AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/width_wsel7.py --selftest` PASS;
`python -c "import json; ks=set(); walk=lambda o: [ks.add(k) or walk(v) for k,v in o.items()] if isinstance(o,dict) else None; walk(json.load(open('automl_package/examples/capacity_ladder_results/WSEL7/frozen.json'))); assert {'invariant','invariant_5pct_historical','ratio_to_default_by_value','new_default'} <= ks, sorted(ks)"`
exits 0 *(corrected 2026-07-23 at harvest — the original one-liner searched only the top two key
levels for a third-level key and could never have passed; see the harvest note above)*;
`AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m pytest docs/plans/capacity_programme/ -q` green.
**Execution note (user, 2026-07-23):** a first dispatch was recalled mid-run when the user ruled
the review completes before ANY execution; the worker's partial diff is preserved in the session
scratchpad (`wsel20-partial-work.patch`) for the orchestration wave to reuse or discard.

### WSEL-21 — the d ≥ 8 training-protocol escalation — 🗓 **SCHEDULED (user, 2026-07-23 review sitting): the strand's one open compute item; blocks FP-13's trigger evaluation; turns the report's d ≥ 8 claims from OPEN to decided (either way)**

**Why.** The multi-feature bake-off's d ∈ {8, 32} levels are honestly voided: the vetted full-batch
sweep protocol stalls within tens of steps (best-fixed error 17.8–31.7× the noise floor — the v2
verdict block above), so no verdict exists there. The recorded follow-up is a training-protocol
escalation through the strand's optimization-first doctrine; the user's review gave it this slot.

**Spec (three rungs, each gated by the previous):**
- **(i) Written escalation spec** (`docs/plans/capacity_programme/shared/wsel21-escalation.md`,
  Create): the protocol ladder — mini-batch size rungs, LR schedule / patience — with
  pre-registered graduation bars reusing the v2 calibration machinery; adversarial read + ROOT go
  under the standing delegation (it changes training protocol, not measurement semantics; if it
  turns out to touch measurement semantics, full toy-gate process instead).
- **(ii) Calibration first:** each rung runs the d=8 calibration-scale cell only; a rung graduates
  iff it meets the pinned anchor trustworthiness bar (the v2 anchor ratio in
  `automl_package/examples/capacity_ladder_results/WSEL19/wsel19_calibration_d1.json`). **No rung
  graduates → the failure branch IS a done-state:** d ≥ 8 recorded UNREACHABLE under the protocol
  family in the ledger, and the report says exactly that — not a halt, not a user gate.
- **(iii) Grid re-run at the graduated rung** on the voided d ∈ {8, 32} triples; A1 aggregation and
  `automl_package/examples/capacity_ladder_results/WSEL19/mf_aggregate.py` unchanged; the ledger
  regenerated. FP-13's trigger (`flexnn-package.md`) then evaluates on the regularised arm's
  showing, and the input-size-relative sizing rule's large-d half gets its decisive test.
**Non-goals:** no new toy constructions; no labelling/tolerance changes (WSEL-22 owns those); no
re-run of decided levels; no router-default changes (FP-5.b binds).
*Orchestration:* worker authors spec + driver flags; **ROOT runs every grid backgrounded**
(systemd-inhibit, OMP caps) · deps: none (all inputs on disk) · tier: sonnet high · scale:
dynamic · shape: research · verify: plan gates green; decided-level ledger entries bit-stable
under re-aggregation; every rung's verdict a ledger field, not prose.

### WSEL-22 — an OBJECTIVE labelling tolerance — 🗓 **SCHEDULED (user, 2026-07-23 review sitting): §1's pre-registered trigger has FIRED — the reviewer leaned on the 0.25**

**Why.** The per-input labelling band (`DEFAULT_TOLERANCE = 0.25`,
`automl_package/models/flexnn/routing.py:75`) is inherited, never measured; the user's review
called it arbitrary and too large. §1's own clause pre-registered exactly this trigger and remedy.

**Spec, two parts:**
- **(a) Sensitivity sweep — mechanical, pre-registered:** relabel the CACHED per-width error
  tables of the decided bake-off levels (1-D + d=2 only; **NO per-width retraining**) at
  tolerance ∈ {0.05, 0.10, 0.25, 0.50}, refit routers per cell, and report: verdict stability
  (does any recorded bake-off finding flip), label churn vs the 0.25 labels, and the routed
  error / deployed-compute shift per tolerance. Freeze in a `frozen.json` per §1's promise
  ("turn it from inherited into measured"). **Explicitly in scope (user review, 2026-07-23):
  WSEL-9's yacht cells — the strand's ONE real-data-confirmed per-input win must be shown to
  survive the band question, not assumed to (cached battery artifacts; no retraining).**
- **(b) σ-anchored band — replacement candidate, SPEC-GATED:** derive per-row acceptability from
  the fixed true noise level (§3.7) instead of a free constant — a width is acceptable for a row
  iff its excess squared error over the row-best is within what the noise distribution alone
  explains at a stated confidence level (the one remaining free choice is the confidence level, a
  statistical convention rather than a domain constant). Written spec → adversarial read → root
  go, per the standing methodology gate; **adopted only if (a) shows verdict sensitivity OR (b)
  wins on generator-true oracle agreement at the decided levels.**
- **(b′) THE BLEND HAS NO THRESHOLD — user ruling, 2026-07-23 (verbatim intent: "Blend should
  have no threshold. Blend is purely a function of the values on the out-of-sample data for each
  width").** Fact of record: the bake-off's blend trained on the SAME 0.25-band hard labels as
  hard routing (user-confirmed at dispatch: "training targets stay hard-label in both modes") —
  the pipeline DISCRETIZED the continuous held-out error values through the band, then
  re-smoothed them with a learned classifier. The band is legitimate ONLY where a discrete pick
  must prefer cheaper widths (hard routing and its labels); a blend makes no discrete pick and
  needs no band. **An earlier draft of this bullet proposed a "σ-anchored acceptability soft
  target" for the blend — SUPERSEDED (an acceptability notion is a band by another name).** Two
  threshold-free candidates, both pure functions of the out-of-sample values:
  1. **Noise-posterior targets (zero free constants on toys):** with σ fixed at truth (§3.7),
     the per-row weight of width k is the posterior over widths given the observed held-out
     value — ∝ exp(−err_k / (2σ²)), normalized per row; the router learns x → that distribution
     via the EXISTING `fit_soft` path (`automl_package/models/flexnn/routing.py`; no new
     machinery). Real-data use inherits (b)'s estimated-σ caveat.
  2. **Direct-objective router (no targets at all; σ-free, real-data-compatible):** train the
     mixing weights to directly minimize the BLENDED held-out loss — the unresolved research arm
     already protected in the codebase (`capacity_ladder_s2._train_router_direct`, named in the
     router module docstring and PROTECTED.tsv).
  The reopen protocol's blend re-evaluation tests: band-changed hard labels (continuity with the
  recorded verdict) AND both threshold-free candidates. **Report scoping, binding:** the recorded
  bake-off verdict is stated as "the blend AS IMPLEMENTED (a smoothed thresholded object) is
  dominated" — never "blending is dominated"; the threshold-free blend was never tested.
  Historical note: ad-hoc soft-target constructions (EM responsibilities, kNN smoothing) lost a
  certification-era five-arm label-construction comparison and were retired; that comparison
  predates both candidates and does not pre-judge them.
**Implication map (user question, 2026-07-23) — what (b)'s adoption reopens, and what it cannot:**
- **Gatekeeper = (a).** The sweep re-reads every DECIDED per-input verdict at the alternative
  bands from cached artifacts: a verdict that survives stays CLOSED and gains robustness
  evidence; a verdict that flips REOPENS at the adopted band. Nothing reopens unconditionally.
- **Conditional-reopen set (everything that consumed 0.25-band labels):** WSEL-7's invariance
  verdict; WSEL-19's bake-off in **BOTH routing modes — the BLEND verdict is explicitly in
  scope** (blending mixes per-width outputs by the class probabilities of routers TRAINED on
  these labels; a changed band changes the posterior the blend uses); WSEL-6's router-dependent
  arm; WSEL-9's per-input arm. Feasibility per readout: hard-routing re-reads need only the
  cached error tables; BLEND re-evaluation needs per-width predictions — cached state dicts
  exist locally (never committed), so blend re-eval is local-machine-feasible without
  retraining, and (a)'s ledger states that dependency.
- **Untouched by construction:** WSEL-8 (output contract carries `w_shared_width`/`w_sweep_width`
  only — no router-consuming arm); the global arms' selection rule is untouched **by (a)/(b)**
  (already noise-aware 2·SE — a different rule for a different problem, §1) **but is (c)'s subject
  if WSEL-24 convicts it — see (c) below**; certified historical artifacts (frozen paper trail).
- **(c) CONDITIONAL — global-rule band remedy (added 2026-07-23, review re-pass; root catch:
  WSEL-24's remedy routing pointed here while this task's scope disclaimed the global rule —
  a conviction would have had no owning task).** Trigger: WSEL-24's Q2 replay convicts the GLOBAL
  cheapest-within-tolerance rule — the recorded bootstrap-2·SE band swallowed genuinely-better
  widths (`dominant_cause` ∈ {band, se_estimate} on the global curve). Remedy is SPEC-GATED like
  (b), never pre-built: candidate directions are (b)'s σ-anchored criterion lifted to the global
  curve (a width is within-tolerance iff its excess held-out error over the best is within what
  the noise level alone explains at a stated confidence) and/or a recalibrated bootstrap SE;
  a `rule_objective_mismatch` verdict is explicitly NOT (c)'s to fix — that is the selection-POLICY
  ruling reserved to the user (WSEL-24 block). Real-data use inherits (b)'s estimated-σ caveat
  verbatim. Dormant unless triggered; costs nothing if WSEL-24 exonerates the band.
- **Report consequence:** every routed compute-vs-quality trade is a function of the band; the
  report states the band of record next to each such number.
- **(b) real-data caveat, BINDING on its spec:** the σ-anchored form uses the TRUE noise level —
  a toy-only luxury. Real-data adoption requires an estimated-noise variant whose estimator
  passes §3.7's no-learned-variance doctrine; until one does, the flat band remains the
  real-data default and (b) is a toys-only instrument.

**Non-goals:** no change to `routing.py` defaults from this task (FP-5.b binds; adoption goes
through the owning strand); no per-width retraining; no d ≥ 8 cells (WSEL-21 owns those).
*Orchestration:* (a) worker-dispatchable at the orchestration wave (write set: new driver + its
results dir); (b) blocked on its spec · tier: sonnet high · scale: static · shape: research ·
verify: plan gates green; (a)'s verdict-stability table lands as ledger fields with
per-tolerance leaves, and the report's router section cites it wherever the 0.25 appears.

### WSEL-23 — EXHAUST THE ARCHITECTURE before any negative replacement verdict — 🗓 **SCHEDULED (user ruling, 2026-07-23 joint review: "This is an exercise in inventing a new architecture... we need to exhaust every possibility of improving the architecture... before we report a negative result")**

**Why.** WSEL-8's both-halves FAIL is a verdict on the CURRENT architecture instance (nested
prefixes on one shared trunk, uniform joint loss), not on the programme goal. The user ruled the
replacement line does NOT stop: the improvement space is exhausted first; WSEL-10's
replacement-claim section ships only at this task's end-state.

**The candidate ladder (enumerated NOW so "exhausted" is checkable; every candidate spec-gated
before build; §3.7 binds throughout — all variances FIXED/declared, never learned):**
1. **Derived loss weighting — probabilistically principled, PRIMARY.** The current objective
   (unweighted sum over widths) is the MLE of an observation model asserting every width predicts
   y with the SAME fixed variance — false by construction (a narrow head carries irreducible
   approximation deficit). The coherent model, y | x, w ~ N(f_w(x), σ² + a²(w)) with σ² the true
   noise (§3.7) and a²(w) a DECLARED approximation-deficit law, derives per-width weights
   1/(σ² + a²(w)) — a consequence, not a penalty (programme doctrine: no arbitrary penalties).
   Pre-registered mechanism prediction: the unweighted sum lets the narrowest heads' large
   irreducible errors dominate the shared units' gradients; discounting error a head CANNOT
   remove frees the most-shared early units — the matched-width premium at MIDDLE widths shrinks.
   **Generalization clause, BINDING (user, same sitting): a²(w) is a PARAMETRIC LAW in w
   (approximation-theoretic decay), instantiated once and FROZEN — never per-width constants —
   and validated at ≥2 values of w_max (the w_max=12 toy plus at least one other) with the
   profile response pre-registered. Same discipline as the router sizing rule and §3.10.**
   **Pre-registered SELECTION corollary (added 2026-07-23, review re-pass):** the dial's
   right-shifted pick is a mechanical consequence of the mid-width premium (WSEL-8's mechanism
   paragraph: the dial's curve still improves at its widest heads, so the same honest rule picks
   wider — 9/11/11 vs the sweep's 7/8/6). If this candidate shrinks the premium, the end-state
   re-run's dial pick must move NARROWER and pick-agreement with the sweep must rise from 0/3.
   The protocol measures both halves anyway; registering the direction now makes the mechanism
   story falsifiable rather than post-hoc.
   **Evidence-gated MECHANISM DIAGNOSTIC + single-head companion (user conditional approval,
   2026-07-23 re-pass: "I don't mind a more thorough investigation if we have concrete evidence
   that we are missing stuff here & the loss needs redress"):**
   - **(i) Gradient-attribution diagnostic, runs FIRST inside this candidate's spec — zero
     training, a few forward/backward passes on existing code:** under the CURRENT unweighted
     sum, measure each width term's share of gradient magnitude on the SHARED parameters (trunk
     for the multi-head; trunk + output weights for the single-head), at init and at a
     mid-training checkpoint, 3 seeds. **Pre-registered expectation: the narrow widths (1–3,
     whose error is dominated by irreducible approximation deficit) contribute an outsized
     share.** This directly tests the candidate's own premise BEFORE any weighted training —
     evidence either way: confirmed → the "loss needs redress" claim has direct evidence;
     refuted → the mechanism story is weakened and that is recorded against the candidate too.
     *Why a diagnostic at all (recorded so the reasoning survives the conversation): the
     observational ordering — mid-width penalty 1× (nothing shared) / 2–6× (trunk shared) /
     9–22× (everything shared) — is CONFOUNDED by capacity (12 vs 90 effective output
     parameters), so it cannot distinguish gradient distortion from capacity shortage; only a
     direct gradient measurement can. And the stop-grad result (removing the cross-width pull
     entirely was the WORST recipe, 17.7×) already rules out the crude form of the claim — the
     defensible claim is that the interaction's BALANCE is wrong, which is exactly what
     weighting changes and what the diagnostic measures.*
   - **(ii) Single-head companion arm, CONDITIONAL on (i) confirming:** the single-head
     structure (`NestedWidthNet` running-sum form) trained with the same derived weights,
     3 seeds, canonical cell only — a mechanism DISCRIMINATOR, not a resurrection: the two
     architectures share different amounts (trunk-only vs everything), so the weighting should
     help the single-head's mid-widths MORE in relative terms if the mechanism is real, and a
     multi-head-helps/single-head-doesn't outcome localises the single-head failure elsewhere
     (consistent with its 3× full-width gap, which this mechanism cannot reach). **Expectation
     on record: it stays uncompetitive at full width; its retirement (ordering-success /
     accuracy-failure) stands unless it closes the primary bar, in which case the existing
     promotion path applies.** If (i) refutes, this arm is DROPPED and the drop recorded.
2. **Deployment-prior mixture training (the programme's own prior-over-capacity pattern):**
   weights = a DECLARED prior over deployment widths (uniform = today's schedule; any non-uniform
   choice is stated and justified AS a prior) — the width analog of depth's ELBO geometric prior
   and ProbReg's k-prior. Coherent but subjective; secondary to candidate 1.
3. **Per-width private capacity:** small per-width adapters / low-rank corrections over the
   shared trunk — attacks the located mechanism (the nested-prefix compromise) by changing the
   model class rather than the objective; parameter/compute overhead charged honestly via the
   accounting module.
4. **Self-distillation down the ladder:** the widest head (nearest its floor) as teacher for the
   middle heads. NO dedicated-net teacher — that would be circular for the replacement economics.

**End-state ("exhausted"), checkable — no invented gates:** every candidate either (i) closes its
pre-registered bar (matched-width ratio bar set by the noise-aware rule per WSEL-20's binding,
never a flat percentage) → the WSEL-8 protocol re-runs at the winning recipe and the claim is
re-graded; or (ii) fails its bar with trajectory-verified convergence. Either way the ladder's
outcomes ship in the report as the exhaustion evidence — only then may a negative replacement
verdict be reported final.
**Recipe-survival gate (root, recorded with the ruling):** a winning recipe must ALSO re-verify
the dial's certified selection behaviour (G-WIDTH's two clauses re-read on the retrained network —
cheap, no new cells beyond the re-run) — closing the matched-width gap while silently breaking
the dial would trade the architecture's one proven property for the contested one.
**Circularity guard on candidate 1's law:** on the toy, a²(w) MAY be pinned from the existing
dedicated-net curves (already on disk — legitimate for the mechanism test); the ≥2-w_max
generalization validation must pin the law WITHOUT a new sweep (theory or a declared cheap probe),
or the law inherits the sweep's cost and the replacement economics are circular.
**Transfer note:** candidate 1's derivation applies verbatim to the depth family (per-depth heads
on a shared trunk carry the same false equal-variance assertion); recorded here for depth's
unpark — depth's parked plans are NOT edited from this task.

### WSEL-24 — the selection post-mortem: a winning width should exist — did SELECTION throw it away? — ✅ **ANSWERED 2026-07-23: YES a winning width existed on ALL FIVE datasets, and selection lost it on all five. The cause SPLITS cleanly by selection-set size: small datasets → underpowered selection (causally demonstrated); large datasets → the rule answering "smallest sufficient" where the battery scores "most accurate" (a POLICY question, batched for the user).**

> ✅ **HARVESTED 2026-07-23 (worker-run; root re-verified: ruff clean, `--selftest` PASS, ledger
> loads with all three verdict fields per dataset, write set exactly the declared driver + ledger).**
> Driver `automl_package/examples/width_wsel24.py` · ledger
> `automl_package/examples/capacity_ladder_results/WSEL9/attribution.json`.
> - **diabetes / yacht / energy → `selection_set_size`** (n_selection_used 53 / 37 / 92): per-seed
>   gaps vs the plain control are erratic sign-flippers (up to +217% and +408% swings — the
>   underpowered-selection signature). **CAUSAL for diabetes and yacht, not correlational:** a
>   5-cell probe (inside the pre-declared 6-cell cap) held the rule, the 2·SE multiplier and the
>   bootstrap estimator IDENTICAL and only enlarged the SELECT carve — the verdict FLIPPED to
>   correctly rejecting the bad cheap width in both. <!-- source: `automl_package/examples/capacity_ladder_results/WSEL9/attribution.json` -->
>   **energy's label is inferred by analogy** (same n-clustering + the worst seed variance of the
>   five; its direct probe cell deliberately unspent) — recorded, not hidden.
> - **kin8nm / california → `rule_objective_mismatch`** (n_selection_used 983 / 2477): the SE is
>   small and well-estimated, the SELECT curve genuinely near-flat at the top; the rule correctly
>   trades a hair of accuracy (4.3% / 0.6% vs plain) for a cheaper width. The rule answered its
>   own question faithfully; the battery scores a different one. <!-- source: `automl_package/examples/capacity_ladder_results/WSEL9/attribution.json` -->
> - **Exonerated:** `band`/`se_estimate` on the global rule (no dataset — **WSEL-22(c) stays
>   DORMANT**); `data_carve` (config-identical across every arm in every cell); `recipe` (Q4
>   non-issue — no architecture-vs-control gap anywhere; 3 of 15 cells tie plain exactly at w_max).
> **Consequences applied at harvest:** (1) **WSEL-6's real-data selection-fraction half REOPENS as
> WSEL-6-R** (wave line, §4) — the frozen 15% was validated on toys only and is now causally
> demonstrated underpowered at real-data selection sizes ≤ ~100. (2) The kin8nm/california finding
> is a selection-POLICY question (which rule to field per deployment context) — ⛔ **BATCHED FOR THE
> USER**; no task may re-aim the rule at the battery's objective without that ruling. (3) **No
> re-sequencing** of the parallel trio — dispatched at harvest.

**The fact to attribute.** WSEL-9: W-SWEEP wins no dataset and loses to the plain fixed-width
control on all five — yet the control was built single-difference (one hidden layer at
`hidden_size = w_max`, same activation family, `docs/width_benchmark/benchmark_spec.md` §4).
Per-width dedicated results exist per cell (`<dataset>_<seed>_w_sweep_<width>.json` under
`automl_package/examples/capacity_ladder_results/WSEL9/`), so the questions below are DISK READS
in stated order — compute only inside the pre-declared probe cap.

**Q1 — does a winning width exist? (zero compute, decides the branch):** per (dataset, seed),
the best width in the sweep's OWN recorded table vs the plain control. YES → the family was fine
and **selection lost it** → Q2/Q3. NO → even the best dedicated width loses → Q4 (recipe).
**Q2 — why did the pick miss it? (zero compute):** replay selection counterfactually from the
recorded curves: argmin's pick vs the rule's pick; the pick under narrower/wider bands; the
recorded bootstrap SEs — did the band swallow genuinely-better widths, or was the SELECT-split
curve itself misleading (selection noise)? Separates the user's three suspects: the threshold,
the noise estimate, the rule.
**Q3 — was the selection set too small? (cheap):** recorded `n_selection_used` per dataset; the
frozen 15% fraction was validated ON TOYS ONLY (WSEL-6) and consumed on real data unvalidated —
a frozen-constant-crossing-domains risk now named. Where cached weights exist locally, re-score
on larger selection carves; otherwise probe cap.
**Q4 — recipe branch (only if Q1 = NO):** dedicated width-12 vs the plain control at IDENTICAL
capacity — any gap is pure training recipe → the repair enters WSEL-23's ladder scope
(exhaust-before-negative extends to it). Data-carve accounting (effective training rows per arm)
is read alongside either branch.
**Rule-vs-question distinction, binding on the verdict:** cheapest-within-tolerance answers
"smallest sufficient width"; the battery scores "most accurate arm". If Q2 shows the rule
faithfully answered the wrong question for this context, the finding is a selection-POLICY gap
(which rule to field per context — user ruling required), not a rule defect; the verdict field
must say which.
**Deliverable:** per-dataset decomposition, leaf-cited to the WSEL9 JSONs; verdict fields:
`winning_width_exists`, `selection_lost_it`, `dominant_cause` ∈ {band, se_estimate,
selection_set_size, rule_objective_mismatch, recipe, data_carve}; probe cells only if a disk
read is ambiguous (**cap: 6 cells, pre-declared**).
**Non-goals:** no retraining beyond the probe cap; no selector/band CHANGES (WSEL-22 owns those —
this task only replays counterfactuals); no new datasets; no report prose (WSEL-10 consumes).
**Sequencing consequence (user ruling):** this task runs FIRST; its outcome may re-sequence the
wave — a `band`/`se_estimate` verdict on the PER-INPUT side feeds WSEL-22(a)/(b); **on the GLOBAL
rule's side it triggers WSEL-22(c)** (the conditional global-band remedy — added 2026-07-23 at the
review re-pass when the root caught that WSEL-22's original scope declared the global rule
untouched, leaving a conviction of the global band with no owning task); `selection_set_size`
reopens WSEL-6's real-data half, `recipe` feeds WSEL-23.
*Orchestration:* orchestration-wave head; worker-dispatchable (write set:
`automl_package/examples/capacity_ladder_results/WSEL9/attribution.json` + one driver under
`automl_package/examples/`) · tier: sonnet high · scale: static · shape: research · verify: the
attribution JSON exists with the three verdict fields per dataset and per-cause magnitudes;
plan gates green.
**Non-goals:** no test-split tuning of any weight/law (frozen before any comparison read); no
learned variances (§3.7); no new toy constructions; no per-input/router/labelling changes
(WSEL-22 owns those); no depth/ProbReg cells.
*Orchestration:* candidate specs are orchestration-wave deliverables (spec → adversarial read →
root go, standing gates); **ROOT runs every grid backgrounded** · deps: candidates 1-2 spec-ready
now; candidate 3 build binds §3.9's one-home module rules; candidate 4 after 1's verdict (teacher
quality question) · tier: sonnet high (specs/drivers), opus xhigh (verdict calls) · scale:
dynamic · shape: research · verify: each candidate's spec carries its own pre-registered bar and
verify lines; plan gates green at every landing.

### WSEL-19 — the router-backend bake-off — 🗓 **RATIFIED AT THE 2026-07-22 SIGN-OFF; SCHEDULED (user, same day): HEAD OF THE POST-MERGE PHASE — runs before any settled-model comparison experiment that consumes the router**

**Question:** what should the distilled router BE? This is WSEL-7 rulings 2/3/5/6 turned into an
experiment: the input-size-relative architecture requirement and the first-class regularisation
requirement are decided empirically by a backend comparison, not by fiat.

**Scheduling (user ruling, 2026-07-22 second sitting — "schedule it without me"):** runs at the
head of the post-merge phase, BEFORE the settled-model comparison experiments that would consume
the router — the width strand's three-chooser battery (WSEL-9, on unpark) and the ProbReg
M1/M2/M3 selection studies (`probreg.md`, on unpark) alike. The root writes the
decision-complete spec and the multi-feature toy design spec at that phase without further
scheduling input from the user; the toy design spec itself still goes to the user for GO before
anything is built (the standing toy gate — that review is about the toys' design, not about
timing).
**Scheduling amendment (user, 2026-07-22 third sitting — "why can't you just do all the work"):
the post-merge phase BEGINS THE SAME SESSION, as wave D of the autonomous run, immediately
after the wave-C merge.** Wave D contents, split by what the toy gate actually covers: (D1) the
root authors the decision-complete bake-off spec + the multi-feature toy DESIGN spec; (D2) the
bake-off driver is authored (worker contract); (D3) **the 1-D slice of the grid RUNS** — it uses
only the canonical §3.8 toys and existing selection-set-fraction machinery, so no new toy design
is involved and the standing toy gate does not apply to it; (D4) the multi-feature slice stays
GATED on the D1 design spec's GO — **delegated (user, 2026-07-22 fourth sitting: "you do a
final review of these & tell me if its a go"): the ROOT renders the go/no-go after an
independent adversarial review of the spec (adjudicator pass) and records the verdict + any
amendments here; the user is informed, not awaited.** Related same-sitting escalation: the
item-4 bar recompute returned 5/36 failures under a noise-variance PROXY; a refinement to the
generator-TRUE noise variance (more faithful to the ratified formula, no retraining) is in
flight, with reference-untrustworthy-seed annotation; merge stays halted until its verdict. WSEL-9 (toys-only
ruling) and WSEL-10 (joint-review gate) remain parked by their own user rulings — out of scope
for this run by the user's explicit clarification.

**Four arms (user-approved 2026-07-22):**
1. **Frozen recipe** — the current router MLP exactly as shipped (hidden `(32, 32)`, 300
   full-batch Adam epochs, lr 1e-2; `routing.py:77-80`) — the baseline every claim is measured
   against.
2. **Rule-sized + regularised MLP** — the same architecture family with hidden sizes set by an
   input-dimensionality-relative rule (the rule itself is a deliverable, per ruling 2), trained
   under ruling 6: early stopping on an internal validation split, mild weight decay, no dropout.
3. **Gradient-boosted trees (XGBoost — already a project dependency)** — no
   architecture-vs-input-size problem to solve at all, and native regularisation (early stopping,
   depth limits, shrinkage, subsampling). Trained on the same hard labels the
   cheapest-within-tolerance rule produces.
4. **Constant router** — always predicts the single globally-best capacity from the error table;
   the control that tells us whether per-input routing pays at all.

**× two routing modes, both first-class (user ruling: blending is TESTED, never hand-waved):**
- **HARD** — argmax: one width per input; deployed compute = that width's cost only.
- **PROBABILISTIC** — per-width predictions/likelihoods blended by the router's class
  probabilities (for likelihoods, the existing `blend_scores`/`blend_nll` path). Blending
  executes several widths per input, so **deployed compute is reported next to quality in every
  cell** — otherwise blend wins unfairly. *(Interpretation confirmed with the user 2026-07-22:
  "two ways" = hard routing vs probability-blend routing; training targets stay hard-label in
  both modes.)*

**Grid:** arms × modes, crossed over input dimensionality and selection-set size (small
selection sets are where the overfitting signature appeared). ⚠ The multi-feature toys this
requires do NOT exist in the canonical §3.8 suite; their design goes through the written-toy-spec
gate (user review before building) at scheduling time.

**Readout (ruling 4):** routed held-out error on the underlying model's own training metric — no
size penalty, smallness only via the declared tie-band — plus routed-vs-oracle agreement and mean
deployed compute.

**Constraints inherited:**
- `flexnn-package.md` FP-5.b binds: this task MEASURES AND REPORTS. It writes no new shared
  default and does not modify `routing.py`; only the owning strand changes the router's
  constants/machinery, per its process.
- MASTER Decision 18: the labelling tolerance is not swept.
- The depth and ProbReg sibling studies read this design when they unpark, rather than inventing
  their own.

**Non-goals:** no changes to `routing.py` (the owning strand's write set); no soft-target
*construction* sweep (experiment-protocol, stays with the drivers); no depth/ProbReg cells.

#### WSEL-19 D1 — the decision-complete execution spec (root, 2026-07-22, wave D)

**Driver:** `automl_package/examples/width_wsel19.py` (Create) + results dir
`automl_package/examples/capacity_ladder_results/WSEL19/` (Create by runs). **Write set of the
authoring contract: exactly those two paths.**

**REUSE FIRST (§3.9 — read before writing a line):** `width_wsel7.py` already builds per-width
error tables and fits/evaluates the distilled router per cell; `width_wsel6.py` already varies
the selection-set fraction; `automl_package/models/flexnn/routing.py` already provides
`fit`/`route_index`/`blend_scores`/`blend_nll` and the labelling rule. The bake-off driver
IMPORTS/adapts that machinery — the author re-derives the exact reuse points from source at
execution time and reimplements NOTHING that exists. XGBoost is an installed dependency
(`xgboost`); verify the import before writing code that assumes an API.

**The four backends per cell (WSEL-19 header, ratified):** (1) frozen MLP recipe — call
`DistilledCapacityRouter` at its defaults, unchanged; (2) rule-sized + regularised MLP — hidden
sizes from an input-relative rule the driver DEFINES AND RECORDS in every cell JSON (ruling 2:
the rule is a deliverable; the frozen `(32, 32)` must be its d=1 instance), trained with an
internal validation split + early stopping + mild weight decay, dropout excluded (ruling 6) —
implemented INSIDE the driver (routing.py is out of the write set; this arm re-implements only
the small training loop it needs, citing ruling 3's future home); (3) gradient-boosted trees —
`xgboost` classifier on the same hard labels, native early stopping on the same internal split;
(4) constant router — always the globally-best capacity from the training-side table.

**× two evaluation modes per backend (no extra training):** HARD (argmax route; report routed
held-out error + mean deployed FLOPs via the existing cost accounting) and BLEND (probability-
weighted; for likelihood readouts use `blend_scores`/`blend_nll`; deployed compute = expected
cost under the router's class probabilities — report it next to quality in every cell).

**1-D slice cells (runs NOW, canonical toys only):** toy = tier-1 hetero per §3.8 (§3.7 σ fixed
at truth), selection-set sizes {75, 300, 1200} × seeds {0, 1, 2} × 4 backends × 2 modes. The
per-width error table per (seed) comes from per-width models trained ONCE per seed under the
WSEL-4-vetted protocol and SHARED by all backends (confound C3 of the toy-design spec) — reuse
landed per-width machinery/artifacts where the protocol matches exactly; retrain only what
cannot be reused, and say which in the cell JSON. One JSON per (backend, mode, N_sel, seed):
routed held-out error, oracle agreement, deployed compute, labels/table provenance, the sizing
rule (arm 2), fitted-backend config. `--selftest` (tiny synthetic end-to-end across all four
backends × both modes) and `--summarize` (aggregate to
`automl_package/examples/capacity_ladder_results/WSEL19/frozen.json`) required.
**Multi-feature slice:** cells per `shared/wsel19-toy-design.md` — **GO RENDERED (root, under
the delegated authority, 2026-07-22): adjudicator verdict GO-WITH-AMENDMENTS; all six findings
(F1-F6) incorporated into the amended spec, which is now the build authority.** The
multi-feature authoring contract additionally owns the §2b input-dimension generalization of
the package width classes (`architectures.py` in its write set per §3.9; 1-D behavior
byte-identical, guarded by the existing equivalence suites; the d=1 calibration cell gates the
d>1 grid).

**Verify (authoring contract):** selftest PASS · ruff clean · ONE real 1-D cell (backend 1,
HARD, N_sel 300, seed 0, `--tag authsmoke`, deleted after schema check) · `git status --short`
clean but for the driver before commit.

**Non-goals (authoring):** run no grid beyond the smoke cell (the ROOT runs the slices); no
edits to `routing.py`, `width_wsel7.py`, `width_wsel6.py`, or any plan doc; no tolerance sweep
(Decision 18); no new toy code before the multi-feature GO is recorded.

#### WSEL-19 1-D slice — ✅ **RUN 2026-07-22 (root, wave D3): 72/72 cells, 0 failures** — ledger `automl_package/examples/capacity_ladder_results/WSEL19/frozen.json`

- **The constant-router control wins QUALITY at every selection size (0.00296 ± 0.00006 routed held-out MSE) at the HIGHEST deployed compute (11.3 FLOP-units, mean routed width 5.67).** <!-- source: `automl_package/examples/capacity_ladder_results/WSEL19/frozen.json` (`per_group['constant:hard:*']`) -->
  This is the designed shape of the cheapest-within-tolerance policy, not a routing refutation:
  routers trade a bounded quality band for compute. The frozen backend's hard cells deliver the
  compute saving (9.4-9.9 units) within ~1.11-1.17× constant's error at n_sel ≥ 300 <!-- source: `automl_package/examples/capacity_ladder_results/WSEL19/frozen.json` (`per_group['frozen_mlp:hard:{300,1200}']` vs `constant:hard`) -->
  (0.00346 → 0.00328); the other two backends exceed the band at some sizes. <!-- source: `automl_package/examples/capacity_ladder_results/WSEL19/frozen.json` (`per_group`) -->
- **The starved cell (n_sel = 75) INVERTS ruling 6's expectation at d=1: the frozen fixed-epoch
  recipe is the most robust (0.00392 ± 0.00024), while the identically-sized early-stopping
  recipe — at d=1, `rule_mlp` IS the same (32, 32) net, differing only by validation split +
  early stop + weight decay — lands ~6× worse with huge seed variance (0.02324 ± 0.02012);
  xgboost 0.00906 ± 0.00552.** <!-- source: `automl_package/examples/capacity_ladder_results/WSEL19/frozen.json` (`per_group['*:hard:75']`) -->
  Reading: at 75 points the internal validation carve costs more than fixed-epoch overfitting
  gains — mandatory early stopping is NOT free at tiny selection sets. NOT recorded as a
  refutation of ruling 6 (d=1 is the frozen recipe's home turf, and the motivating overfitting
  evidence came from BIGGER routers); **the multi-feature cells at d ∈ {8, 32} are the deciding
  test, now with this 1-D anchor on record.**
- **BLEND is dominated by HARD on this toy — worse quality AND more deployed compute in every
  group** (frozen at 75: 0.00690 vs 0.00392; deployed 10.7 vs 9.9). <!-- source: `automl_package/examples/capacity_ladder_results/WSEL19/frozen.json` (`per_group['frozen_mlp:{hard,blend}:75']`) -->
  First empirical answer to the blending half of ruling 5 at d=1; the multi-feature slice
  re-tests it where the router's job is harder.
- **xgboost is competitive at n_sel ≥ 300** (0.00390 / 0.00343), carries the best
  oracle-agreement (28-34%) and the cheapest deployed compute among the routed backends, but is
  unstable at 75. <!-- source: `automl_package/examples/capacity_ladder_results/WSEL19/frozen.json` (`per_group['xgboost:hard:*']`) -->
- **Oracle agreement is low across the board (routed 20-34%, constant 9.8%)** — per-point labels
  on this toy are noise-dominated (the adjudicator's selection-on-noise finding applies to the
  LABELS as well; the multi-feature falsifier's generator-true oracle exists for exactly this
  reason). <!-- source: `automl_package/examples/capacity_ladder_results/WSEL19/frozen.json` (`per_group` `oracle_agreement` fields) -->
- No backend/default change lands from this slice (FP-5.b binds; the slice MEASURES). The
  multi-feature slice + this anchor together produce the bake-off verdict.

#### WSEL-19 multi-feature slice — ✅ **ANSWERED 2026-07-23 (v2 rotated-box redesign): calibration GREEN, 432/432 cells run, verdict recorded below. d=2 DECIDED on both geometries (frozen default survives; routing pays in compute, not quality); d ∈ {8, 32} OPEN — a full-batch training-protocol wall, honestly voided by the pre-registered gates, never decided from survivors. The v1 Φ-lift failure record below stands as history.**

**The driver extension itself LANDED** (root-applied via the write-set-guard handoff protocol —
the worker built and verified against a scratchpad copy; the root re-ran every verify against
the real path: selftest PASS incl. all multi-feature checks, ruff clean, the input-dim
equivalence suite 9/9, `--summarize` still aggregates the real 72 1-D cells unchanged). 1-D
behavior is byte-identical; multi-feature cells mechanically REFUSE to run while the
calibration artifact records `passed: false`.

**The calibration verdict** (real scale: w_max=12, n_train=1500, axis geometry, seeds 0-2;
artifact `automl_package/examples/capacity_ladder_results/WSEL19/wsel19_calibration_d1.json`):

| seed | best-fixed MSE/floor | best width (easy) | best width (hard) | regime visible | trustworthy |
|---|---:|---:|---:|---|---|
| 0 | 1.123 | 12 | 12 | NO | yes | <!-- numcheck-ignore: v1 FAILED-run record; the cited artifact was regenerated by the v2 PASSING run (git history holds the v1 values); the v2 verdict block below carries the live citations -->
| 1 | 1.041 | 6 | 6 | NO | yes | <!-- numcheck-ignore: v1 FAILED-run record, superseded on disk by the v2 artifact — see the v2 verdict block below -->
| 2 | 1.015 | 5 | 9 | yes | NO | <!-- numcheck-ignore: v1 FAILED-run record, superseded on disk by the v2 artifact — see the v2 verdict block below -->

Two of three seeds show NO best-width differentiation between the easy and width-hungry
regions; the one seed that differentiates fails per-width convergence trust. Fit quality is
fine (all seeds within 1.2% - 12.3% of the noise floor) — the construction fits; it just no
longer carries the regime structure. <!-- numcheck-ignore: the percentages restate the table's ratio column (same artifact) -->

**Mechanism hypothesis (UNVERIFIED — trace before any redesign):** the amended spec's own §2
warp caveat (adjudicator finding F3): the inverse-CDF lift makes the model see the easy
region's flat-linear target through a sigmoid-like reparameterization, plausibly destroying
its easy/cheap character so every region wants width. The worker flagged this as the likely
explanation, explicitly NOT as a verified diagnosis; per-width prediction-curve tracing is the
first step of any redesign.

**Consequences, recorded:** (1) the bake-off verdict currently rests on the 1-D slice alone;
the input-size-relative half of rulings 2/5 and ruling 6's decisive d ∈ {8, 32} test remain
OPEN. (2) A redesigned lift goes through the SAME gate process as the original — written spec +
independent adversarial review — with the go/no-go RENDERED BY THE ROOT under the standing
delegation above ("the user is informed, not awaited"; re-confirmed by the user 2026-07-22
late evening: "why does it need me? I thought we designed it"). Nothing multi-feature runs
before a passing verdict; nothing waits on the user. (3) The pre-registration culture worked:
the failure cost one calibration run (~36 small nets), not a 432-cell grid read as science.

**✅ MECHANISM TRACED (root-dispatched, 2026-07-22 late; `automl_package/examples/capacity_ladder_results/WSEL19/warp_trace.json` + script, commit `8b3796a`): the warp hypothesis is CONFIRMED, refined — TWO-SIDED WIDTH CONVERGENCE.**
- Verdict basis: practical-floor widths (smallest width whose generator-true error reaches
  ≤ 0.2 × the noise floor), because raw argmin proved unusable — the true-error curves are
  genuinely non-monotonic at tiny errors (a methodological finding recorded in the artifact;
  robustness-checked at 0.1/0.3 fractions). <!-- source: `automl_package/examples/capacity_ladder_results/WSEL19/warp_trace.json` (`declared_criteria`) -->
- Canonical control differentiates on 3/3 seeds (easy practical width 3-5 vs hard 6-8); the
  lifted construction TIES on 3/3 seeds — both regions converge onto width 5. Easy needs MORE
  width under the lift, hard needs LESS: the regions converge onto each other, killing the
  per-input routing signal from both sides. <!-- source: `automl_package/examples/capacity_ladder_results/WSEL19/warp_trace.json` (`per_seed`, `classification`) -->
- Cross-check: the trace's lifted argmins reproduce the calibration artifact's own best-width
  fields exactly on all 3 seeds (pipeline validated bit-for-bit).
- Confound, read asymmetrically: the two caches differ in effective training size (canonical
  600 via the carve vs lifted 1200, no carve). <!-- source: `automl_package/examples/capacity_ladder_results/WSEL19/warp_trace.json` (`caveats`) -->
  More data reaches error floors at SMALLER widths, so the easy-side degradation (3 → 5 despite
  2× data) is UNCONFOUNDED; the hard-side convergence is partially confounded.

**✅ REDESIGN v2 GO RENDERED (root, under the standing delegation, 2026-07-22 late):**
`shared/wsel19-toy-redesign.md` (the rotated-box construction — Φ-warp removed entirely; the
canonical coordinate embedded by a deterministic Householder rotation with uniform decoy
coordinates) authored against R1-R4 below, adversarially adjudicated (construction mathematics
CONFIRMED by executed numeric probe; no construction-level defect found; verdict
GO-WITH-AMENDMENTS), all mandatory + recommended amendments incorporated (verdict-aggregation
rule with survivor-bias closure; pre-dispatch regime reads; calibration failure taxonomy
protecting the iteration budget; trustworthiness field pinned; majority pass bar per the traced
measure; selection pool-and-prefix; scale-aware selftest tolerances). **The amended v2 spec is
the build authority for the multi-feature slice; build proceeds on `capacity/wave-3`.**

**Redesign requirements derived from the trace (BINDING on the redesign spec):**
- **R1** — the network-visible coordinate along `v` must preserve the canonical coordinate's
  geometry: no Φ-warp (or any nonlinear reparameterization) between the canonical target's
  input and what the model sees.
- **R2** — the calibration protocol pins effective training size EQUAL to the canonical carve,
  killing the flagged confound at the protocol level.
- **R3** — the §5.2 differentiation check is operationalized on practical-floor widths (the
  traced measure), not raw argmin.
- **R4** — any new construction's axis/oblique distribution-match claim is re-derived from
  scratch in its ledger (the obvious no-warp candidate — a uniform ridge along `v` with a
  Gaussian complement — has a known marginal-sniffing asymmetry at axis geometry that the
  ledger must close, not wave at).

#### WSEL-19 multi-feature v2 VERDICT — ✅ RUN 2026-07-23 (rotated-box construction; spec `shared/wsel19-toy-redesign.md`; ledger `automl_package/examples/capacity_ladder_results/WSEL19/frozen_mf.json`; aggregator `mf_aggregate.py` beside it)

**Calibration (two-block, gate GREEN).** The regime block at the R2-pinned 600-effective
differentiates 2/3 seeds (the traced measure's own majority bar; seed 1's hard region misses the
0.2-fraction floor on the report split); the anchor block at the grid's own 1200-effective
differentiates **3/3** — the §7 pre-dispatch read PASSED, settling the adjudicator's central
unknown (canonical regime at 1200-effective was unmeasured anywhere before this run). Six nets
needed the pre-registered raised-cap retrain; all converged (both blocks fully trustworthy).
Anchor ratio to noise floor 1.128, regime-block gaps 4/5/3. <!-- source: `automl_package/examples/capacity_ladder_results/WSEL19/wsel19_calibration_d1.json` (`anchor_ratio_to_noise_floor`; `anchor_block.per_seed.*.practical_floor_gap`) --> <!-- numcheck-ignore: 4/5/3 are integer gaps (<3 decimals); anchor value carried on the pointer line -->
**The trace's diagnosis is confirmed end-to-end: removing the Φ-warp restores the canonical regime
structure at identical fit quality — the warp, not the embedding, killed v1.**

**Grid: 432/432 cells, 0 failures; F6 fallback second pass 14/14 jobs** (4 full-triple rescues +
per-triple tagged proofs for the stalled levels, `*_f6fallback.json`).

**A1 aggregation (survivor-bias closure applied as pre-registered; ledger
`automl_package/examples/capacity_ladder_results/WSEL19/frozen_mf.json`, aggregator `mf_aggregate.py`):**
- **d=2 axis DECIDED 3/3** — two seeds via F6 fallback rescue, fit ratios 1.108 and 1.040 at n_train=4000. <!-- source: `automl_package/examples/capacity_ladder_results/WSEL19/frozen_mf.json` (`triples.d2_axis_seed0.ratio_to_noise_floor`, `triples.d2_axis_seed2.ratio_to_noise_floor`) -->
- **d=2 oblique DECIDED 2/3** — seed 0's post-fallback fit ratio 1.133 misses the anchor bar (1.128), recorded VOID not bent. <!-- source: `automl_package/examples/capacity_ladder_results/WSEL19/frozen_mf.json` (`triples.d2_oblique_seed0.ratio_to_noise_floor`) --> <!-- numcheck-ignore: 1.128 is the anchor from wsel19_calibration_d1.json cited above; only 1.133 belongs to frozen_mf.json -->
- **d=8 and d=32 OPEN at every geometry (0/3 verdict-weight everywhere): a TRAINING-PROTOCOL wall, not an architecture verdict** — the vetted full-batch sweep protocol stalls within tens of full-batch steps at d ≥ 8, best-fixed error 17.8–31.7× the noise floor, so the practical floors never exist. <!-- numcheck-ignore: 17.8/31.7 are the min/max of the per-triple ratio_to_noise_floor leaves in frozen_mf.json (d8/d32 triples), no single leaf -->
- The F6 fallback moves the probed d=32 triple from 26.740 to 26.712 (full-batch means more data per step, zero extra steps) — proof-recorded per stalled triple. <!-- numcheck-ignore: 26.740 = untagged default record's ratio; 26.712 = the *_f6fallback.json record's ratio; two files, one comparison line -->
- Ruling 6's decisive d ∈ {8, 32} test and the large-d half of the input-size-relative rule therefore remain **OPEN — protocol-limited**; the recorded follow-up is a training-protocol escalation (mini-batching / LR-patience ladder) run through the strand's own optimization-first doctrine, never freelanced mid-grid.

**Findings on the DECIDED levels (1-D anchor + d=2, both geometries — the bake-off verdict):**
1. **The constant-router control again wins QUALITY at every selection size at the HIGHEST deployed compute** (d=2 axis constant hard 0.003053 at n_sel 300, vs frozen 0.005978), replicating the 1-D shape: per-input routing buys compute, never raw quality, at every d measured. <!-- source: `automl_package/examples/capacity_ladder_results/WSEL19/frozen_mf.json` (`per_group.d2:axis:constant:hard:300`, `per_group.d2:axis:frozen_mlp:hard:300` `routed_held_out_quality_mean`) -->
2. **The starved cell (n_sel=75) is GEOMETRY-CONDITIONAL at d=2** — at AXIS the 1-D inversion replicates (frozen 0.009920 more robust than rule_mlp 0.016599), at OBLIQUE it reverses (rule_mlp 0.007444 beats frozen 0.010150). Ruling 6's regularised recipe is neither vindicated nor refuted here — geometry-dependent, and the decisive larger-d test is protocol-blocked. <!-- source: `automl_package/examples/capacity_ladder_results/WSEL19/frozen_mf.json` (`per_group` `d2:{axis,oblique}:{frozen_mlp,rule_mlp}:hard:75` `routed_held_out_quality_mean`) -->
3. **BLEND stays dominated by HARD at n_sel ≥ 300 on every backend and both geometries** (replicating 1-D); at the starved size the gap narrows to the one marginal blend win, frozen oblique 0.009736 blend vs 0.010150 hard. <!-- source: `automl_package/examples/capacity_ladder_results/WSEL19/frozen_mf.json` (`per_group` `d2:oblique:frozen_mlp:{blend,hard}:75` `routed_held_out_quality_mean`) -->
4. **xgboost is geometry-sensitive exactly as C2 anticipated** — best routed backend at the starved AXIS cell (0.004290), degraded and unstable at OBLIQUE (0.016394 at n_sel 300); a tree backend cannot be the default on geometry-robustness grounds. <!-- source: `automl_package/examples/capacity_ladder_results/WSEL19/frozen_mf.json` (`per_group` `d2:axis:xgboost:hard:75`, `d2:oblique:xgboost:hard:300` `routed_held_out_quality_mean`) -->
5. **Oracle agreement stays low** — per-point labels remain noise-dominated at d=2, best 0.202 (xgboost axis), consistent with the 1-D finding. <!-- source: `automl_package/examples/capacity_ladder_results/WSEL19/frozen_mf.json` (`per_group.d2:axis:xgboost:hard:75.oracle_agreement_mean`) -->

**Bake-off verdict (1-D slice + d=2, the decided scope):** no backend change lands (FP-5.b
binds; this task MEASURES). The frozen MLP default SURVIVES: no challenger beats it
unconditionally on quality, robustness-to-starvation AND geometry at once; the
input-size-relative sizing rule (rulings 2/5) earns no unconditional win on the sizes it could
be tested at, and its decisive large-d test is protocol-blocked. Routing's value proposition on
these toys is compute-at-bounded-quality-loss, at every dimensionality where fit succeeds.

### WSEL-8 — the W-SHARED ≈ W-SWEEP claim, both halves, on toys — ⛔ **ANSWERED 2026-07-22: BOTH HALVES FAIL. The dial network is NOT ≈ the exhaustive sweep on this toy — 2.6-7.2× worse at matched middle widths, 0/3 agreement on the chosen width, and the disagreement is a mechanical consequence of the quality gap. A FAIL is a finding, not a bug.**

**RESULT: `agreement_rate: 0.0`, quality NOT matched at middle widths** — ledger `automl_package/examples/capacity_ladder_results/WSEL8/frozen.json` (36 sweep cells + control + 3 dial cells, ALL trustworthy, no cap hits after the same-precedent repair; per-cell JSONs in the same directory).

**Integrity chain, in order:** the fresh sweep reproduces WSEL-4's ported reference EXACTLY (36/36, max relative error 0.0000 — deterministic protocol; `automl_package/examples/capacity_ladder_results/WSEL8/wsweep_control.json`), and the two capped cells were the SAME (seed, width) slow-creep cells the data-fraction study repaired — retrained at a raised cap they converged at the identical epoch counts (8983, 6850), capped originals under `automl_package/examples/capacity_ladder_results/WSEL8/capped_at_6000/`. <!-- numcheck-ignore: the two epoch counts live one-per-file in the rebuilt `automl_package/examples/capacity_ladder_results/WSEL8/_cache/*_meta.json` files; 0.0000 is the control file's max relative error -->
The selection fraction 0.15 was read from the data-fraction study's frozen ledger at runtime with provenance recorded in every cell (`selection.fraction_source`), never hardcoded.

**Half (a) — quality at MATCHED width (REPORT split, mean over 3 seeds, dial/sweep ratio; >1 = dial worse):**

| w | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ratio | 1.081 | 1.159 | 0.997 | 7.207 | 2.932 | 3.268 | 2.641 | 2.638 | 1.751 | 1.680 | 1.571 | 1.486 | <!-- source: `automl_package/examples/capacity_ladder_results/WSEL8/frozen.json` `quality_at_matched_width[].mean_ratio_shared_over_sweep` -->

Parity at widths 1-3; the dedicated nets sit at the noise floor from width ~6 while the dial
network's shared trunk cannot serve the middle heads — the premium peaks at width 4 and shrinks
monotonically toward the widest heads. Consistent with this wave's stage-2 verdict (strict joint
training is what the cheap structure costs) and the frozen-bias diagnostic (≤2% of it).

**Half (b) — agreement on the chosen width: 0/3, and the dial ALWAYS picks wider:**

| seed | dial pick | sweep pick | dial MSE @ its pick | sweep MSE @ its pick |
|---|---:|---:|---:|---:|
| 0 | 9 | 7 | 0.004502 | 0.002848 | <!-- source: `automl_package/examples/capacity_ladder_results/WSEL8/hetero_0.json` -->
| 1 | 11 | 8 | 0.006105 | 0.002634 | <!-- source: `automl_package/examples/capacity_ladder_results/WSEL8/hetero_1.json` -->
| 2 | 11 | 6 | 0.003282 | 0.002857 | <!-- source: `automl_package/examples/capacity_ladder_results/WSEL8/hetero_2.json` -->

**Mechanism (visible in the recorded curves, not conjectured):** the dedicated nets' held-out curve
flattens at ~width 6 (noise floor), so cheapest-within-tolerance stops early; the dial network's own
per-width curve is still IMPROVING at its widest heads (the least trunk-starved ones), so the SAME
selection rule, applied honestly to a right-shifted curve, picks wider. The disagreement is half
(a)'s quality gap expressed through the selector — not a selector defect. No pick was
ceiling-bound (`ceiling_bound` false everywhere).

**The practical trade recorded with the verdict:** at its own pick the dial network's held-out MSE
is 1.15-2.32× the sweep's, while selecting wider (more inference compute), in exchange for a
1.6-2.7× end-to-end selection-cost saving (`selection_cost.total_macs` per arm, per cell — the
saving is far below the naive 1-vs-12-trainings intuition because the ALL schedule trains every
width every step anyway). <!-- numcheck-ignore: the per-seed MSE ratios and cost ratios are derived across fields of `automl_package/examples/capacity_ladder_results/WSEL8/hetero_{0,1,2}.json`, stored in no single leaf -->

**What this does NOT refute:** G-WIDTH's certification (the dial's monotone/converged/selects
behaviour is untouched — this is the first head-to-head against DEDICATED nets); the multi-head
retention ruling (WSEL-16 compared trainable-cheap structures against the multi-head reference, a
different question). What it DOES establish: on this toy, "one training that carries every width"
is not a free replacement for the sweep — it buys ~2× cheaper selection at ~1.1-2.3× worse
end-task error and a wider, costlier deployed width. Reported prominently for end-of-run user
review; WSEL-10 (the report) stays parked and reads this ledger when it unparks.

**→ USER RULING 2026-07-23 (joint review): the FAIL stands as a recorded measurement but is NOT
report-final — "this is an exercise in inventing a new architecture; exhaust every possibility of
improving the architecture before we report a negative result." The replacement-claim section of
WSEL-10 is GATED on WSEL-23's checkable exhaustion end-state (task block above); if any WSEL-23
candidate closes its bar, THIS protocol re-runs at the winning recipe and the verdict re-grades.**

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

### WSEL-9 — real data + baselines — ✅ **ANSWERED 2026-07-22 (wave E, root-run): NO FREE LUNCH — every family wins somewhere; the width-selection machinery is competitive but never beats its own fixed-width control except on yacht (per-input's win), and per-input shows high seed variance on energy. Full verdict block below.**

> **✅ WAVE-E RESULT (root, 2026-07-22; 240 cells + 2 probe extras, 0 failures; spec
> `docs/width_benchmark/benchmark_spec.md`, driver `width_wsel9.py`, per-dataset ledgers
> `automl_package/examples/capacity_ladder_results/WSEL9/<ds>_summary.csv`).** Mean held-out
> MSE across seeds 0-2 (winner bolded per row):
>
> | dataset | linear | lightgbm | plain NN | W-SWEEP | W-SHARED | W-PERINPUT |
> |---|---:|---:|---:|---:|---:|---:|
> | diabetes | **3252.313** | 3872.722 | 3626.904 | 3709.794 | 3791.329 | 3665.728 | <!-- source: `automl_package/examples/capacity_ladder_results/WSEL9/diabetes_summary.csv` (mean of held_out_mse across seeds) -->
> | yacht | 95.653 | 86.710 | 1.754 | 3.190 | 2.414 | **1.492** | <!-- numcheck-ignore: means of held_out_mse across seeds in `automl_package/examples/capacity_ladder_results/WSEL9/yacht_summary.csv` — CSV source, the per-line checker requires .json -->
> | energy | 39.477 | 21.525 | **11.904** | 28.541 | 33.076 | 17.559 | <!-- numcheck-ignore: means of held_out_mse across seeds in `automl_package/examples/capacity_ladder_results/WSEL9/energy_summary.csv` — CSV source, checker requires .json -->
> | kin8nm | 0.04051 | 0.01711 | **0.00876** | 0.00913 | 0.01150 | 0.01751 | <!-- source: `automl_package/examples/capacity_ladder_results/WSEL9/kin8nm_summary.csv` (same construction) -->
> | california | 0.49932 | **0.21534** | 0.30150 | 0.30321 | 0.29682 | 0.32098 | <!-- numcheck-ignore: means of held_out_mse across seeds in `automl_package/examples/capacity_ladder_results/WSEL9/california_summary.csv` — CSV source, checker requires .json -->
>
> **Findings, in the strand's own terms:**
> 1. **No free lunch, cleanly demonstrated:** linear wins diabetes (the floor-detector baseline
>    did exactly its job — the dataset is essentially linear and every richer model overfits
>    it); trees win california; the plain fixed-width NN wins energy and kin8nm; per-input
>    width selection wins yacht.
> 2. **The strand's central real-data question — does width SELECTION pay over a fixed-width
>    net?** Mostly NO on this battery: plain NN (the "dial at fixed width" control) beats all
>    three width-choosing arms on energy and kin8nm and ties the family on diabetes; W-SHARED
>    never wins a dataset outright. The exception is yacht, where W-PERINPUT is the overall
>    winner (small, smooth, few-dim — the regime where per-input capacity genuinely varies).
>    Consistent with the toy-side findings (the 1-D bake-off's constant-control result and the
>    dial-vs-sweep verdict): selection buys flexibility and compute options, not raw accuracy.
> 3. **Per-input's known variance problem shows on real data too:** energy spans 2.972-39.793 <!-- numcheck-ignore: w_perinput held_out_mse per seed in `automl_package/examples/capacity_ladder_results/WSEL9/energy_summary.csv` — CSV source, checker requires .json -->
>    across three seeds.
> 4. **Integrity:** 235/240 cells clean under the battery's trajectory-verified convergence <!-- numcheck-ignore: counts derived by scanning every per-cell JSON's hit_cap field under WSEL9/, stored in no single file -->
>    rules; the 5 `hit_cap` cells are ALL LightGBM (kin8nm ×3, energy seed 2, california seed <!-- numcheck-ignore: same scan, continuation -->
>    2) — its round-cap bound, so the tree baseline's numbers there UNDERSTATE it (it wins
>    california regardless; an uncapped re-run is a cheap follow-up available at report time,
>    deliberately not run now — no strand claim rests on the tree margin).
> 5. Selection costs are recorded per arm in every cell/CSV per the spec (baselines carry the
>    documented non-applicability); the cost story joins the report when WSEL-10 unparks.
>
> **→ 2026-07-23 (user review): W-SWEEP wins NO dataset and loses to the plain fixed-width
> control on ALL FIVE — the gap is scheduled for attribution as WSEL-24 (task block below):
> selection-rule cheapness (by design) vs training-recipe difference vs data carve, decidable
> from the per-width cell JSONs already on disk.**

**Gate added at the 2026-07-22 sign-off (user):** on unpark, **WSEL-19 (the router-backend
bake-off) runs first** — the router backend must be settled before any settled-model comparison
experiment consumes it.
**Unpark record (2026-07-22, third sitting):** the user's 2026-07-20 ruling ("width stays
toys-only", MASTER Decision 3's real-data exemption not extended) is REVERSED by the same
authority — "let's do #1 & #2 now" in the post-review scheduling discussion, #2 = this task.
The spec below runs AS RETAINED, no re-scoping. Ordering per the sign-off gate above: wave E,
after WSEL-19's 1-D slice (wave D) — and if the multi-feature GO is still pending at wave-E
time, after the 1-D slice alone, since WSEL-9's arms consume the router at the real datasets'
own dimensionality (the frozen 1-D-validated default until the bake-off rules otherwise; §3.6
feed-forward applies to any later change).
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
  **⛔ GATED (user, 2026-07-23): this section ships only at WSEL-23's exhaustion end-state** —
  positive (re-graded at a winning recipe) or negative (with the candidate ladder as exhaustion
  evidence); never before. **The ARBITER is a named reference in the comparison suite** (user,
  same sitting): the held-out error-table per-input judge appears explicitly as the routing upper
  bound, with its measured noise limitation and the generator-true oracle beside it.
- **Real data + baselines — ⏸ SECTION CUT** (user ruling 2026-07-20: width stays toys-only; WSEL-9
  parked). **The report must state this explicitly, not silently omit it**: say that no external
  comparator — tree model, plain single-output NN, or linear floor — was run for this strand, that
  every claim here therefore rests on constructed targets, and that a real-data pass is deferred
  rather than refused. **Do not present a toys-only result as though it had survived a baseline.**
  **→ SUPERSEDED 2026-07-22: WSEL-9 was UNPARKED and RAN (wave E — its verdict block above). The
  section is REINSTATED as report content; only the permanent instruction survives — never
  present a toys-only result as though it had survived a baseline.**
- **Real-data inversion, carried AS A HYPOTHESIS (root, 2026-07-23 review):** on 2 of 5 battery
  datasets the jointly-trained network's cheap read BEAT the dedicated sweep (yacht, california —
  `automl_package/examples/capacity_ladder_results/WSEL9/` summaries), inverting the toy result.
  The report states the plausible mechanism — joint training as a regularizer in the
  overfit-limited small-data regime vs its fit-limited toy loss — **explicitly flagged
  unverified**, with the regime-dependence framing beside the WSEL-8/WSEL-23 story. The
  **uncapped LightGBM re-run (5 capped cells) is PROMOTED from report-time option to the
  orchestration wave** — a published baseline table must not understate a baseline.
- **Transfer ledger (user, 2026-07-23 review) — REQUIRED SECTION.** Every adopted verdict states
  its evidence class: **toy-negative** (the method failed on ground rigged in its favour —
  transfers a fortiori), **toy-positive** (works on constructed targets — NOT adopted for
  practice without real-data confirmation), or **real-data-confirmed** (the battery or a product
  dataset agrees). No verdict ships without its class; the generator-true oracle may appear as a
  validation instrument, never as an adoption criterion.

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

### WSEL-11 — the DISCARDED first run + the original pre-registration — 📦 **ARCHIVED (hygiene pass, 2026-07-23)**

Full case-law text and the verbatim pre-registration spec:
`archive/width-superseded-histories.md` §WSEL-11. Load-bearing summary: the first run trained the
Gaussian NLL on real variance heads — a LEARNED variance, forbidden (§3.7, MASTER Decision 2) — and
its verdict is VOID (user ruling 2026-07-21: results produced in violation of a constraint are
DISCARDED, not reinterpreted). The discarded artifacts stay on disk under
`automl_package/examples/capacity_ladder_results/WSEL11/` as a record, citable as evidence for
nothing (exact ruling in the archive text). The RE-RUN above (sigma fixed at truth, 3 seeds, 9/9
cells) is the verdict of record and was judged against the archived pre-registration. The case law
is cited by §3.8's tier-objective defect note (same failure shape, caught pre-run there).

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
*(Tier-2 update, 2026-07-22: the tier-2 state dicts ARE force-added, per the ratified wave-2
instruction for this commit; the tier-1 three remain the user's call as flagged above.)*
**⛔ RULED at the 2026-07-22 sign-off (item 5 CLOSED, user): trained-model state dicts are NEVER
committed — "we don't commit these sort of results." The tier-1 files stay local, and the
force-added tier-2 files are UNTRACKED again in the same commit (they remain on disk and in the
wave branch's history). This supersedes the wave-2 force-add instruction. The repo-wide `*.pt`
ignore rule is the standing policy; the retrain-to-re-read consequence above is accepted.**

#### Tier-2 corroboration — landed 2026-07-22 (RE-AUTHORIZED run: ALL schedule + §3.7 weighted objective, hetero3)

**RESULT: corroboration only, no bar re-grade — the noisy-easy tier also shows NO decreasing importance ordering, and the prefix penalty vanishes there** — ledger `automl_package/examples/capacity_ladder_results/WSEL13/frozen.json` (`tier2` block; per-cell `wsel13_tier2_all_seed{0,1,2}.json` + `state_tier2_all_seed{0,1,2}.pt` in the same directory).

| seed | Spearman(index, importance) | prefix-vs-greedy gap | Kendall(greedy, index) | widths trustworthy |
|---|---:|---:|---:|---:|
| 0 | -0.0979 | 0.0131 | 0.3030 | 12/12 | <!-- numcheck-ignore: values are `step2_ablation.spearman_index_vs_importance.rho` etc. in `automl_package/examples/capacity_ladder_results/WSEL13/wsel13_tier2_all_seed0.json`; the checker's token extractor drops the minus sign, so a signed leaf can never match -->
| 1 | -0.0559 | 0.0123 | 0.3030 | 12/12 | <!-- numcheck-ignore: same fields in `automl_package/examples/capacity_ladder_results/WSEL13/wsel13_tier2_all_seed1.json`; signed-leaf limitation as above -->
| 2 | -0.0559 | 0.0048 | 0.6061 | 12/12 | <!-- numcheck-ignore: same fields in `automl_package/examples/capacity_ladder_results/WSEL13/wsel13_tier2_all_seed2.json`; signed-leaf limitation as above -->

**Grade vs tier 1 (the two tiers differ in schedule, objective AND toy — the re-authorization ruling
requires the schedule difference named wherever they are tabulated: tier 1 = sandwich + plain MSE +
hetero; tier 2 = ALL + fixed-sigma weighted + hetero3. Cross-tier contrasts attribute to that BUNDLE,
never to a single factor):**
- **The refutation is corroborated**: 0/3 tier-2 seeds anywhere near the `<= -0.5` primary bar —
  there is no decreasing-importance ordering on this tier either. The certified design's mechanistic
  ordering account stays refuted on both tiers now, not just the reference cell.
- **Tier 1's positive correlation does NOT replicate**: tier-2 correlations are ≈ zero (-0.0979 to <!-- numcheck-ignore: tier-2 values as in the table above (signed-leaf limitation); tier-1 values restate the tier-1 table -->
  -0.0559), not positive (+0.524/+0.881/+0.580 on tier 1). The reversed-ordering anomaly is <!-- numcheck-ignore: continuation — tier-1 values from `automl_package/examples/capacity_ladder_results/WSEL13/wsel13_tier1_seed{0,1,2}.json` -->
  tier-1-specific on current evidence.
- **The prefix penalty vanishes on tier 2**: mean prefix-vs-greedy gap 0.0100 vs tier 1's 0.310 — <!-- numcheck-ignore: 0.0100 is `tier2.mean_relative_prefix_gap` and 0.310 restates the tier-1 secondary-bar mean, both in `automl_package/examples/capacity_ladder_results/WSEL13/frozen.json` -->
  under this tier's bundle the prefix is within ~1% of greedy, i.e. "nested prefix" as COMPUTATION
  SHARING is essentially free here.
- **Run integrity**: first tier-2 run (2026-07-22, same day) stopped at a non-stationary point via
  the latched joint stop rule and was discarded unconcluded per its own guard; this run, under the
  corrected simultaneous-flatness rule (commit `f3a1c65`), is 12/12 trustworthy on every seed with
  zero guard hits.

---

### WSEL-13 — the task spec — 📦 **ARCHIVED (hygiene pass, 2026-07-23)**

The verbatim pre-registration this run was judged against — the two diagnostics (single-unit
ablation; prefix-vs-greedy with the load-bearing FIT/SELECT/REPORT three-split carve) and the
pre-registered bars (primary: Spearman(index, ablation importance) at or below −0.5 on 2 of 3
seeds; secondary: mean relative prefix-vs-greedy gap at or below 0.10; `ordering_holds` = both) —
now lives at `archive/width-superseded-histories.md` §WSEL-13-spec. The verdict blocks above
(⛔ ANSWERED: NO, plus the tier-2 corroboration) are unchanged and remain the citable record. The
three-split discipline the spec introduced is carried forward by WSEL-16's Step 5 (its ordering
statistic imports WSEL-13's function on the same carve) and stays binding there.

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

#### DECISION-20 REVISIT — the superseded DRAFT — 📦 **ARCHIVED (hygiene pass, 2026-07-23)**

The root's three-clause draft recommendation (written 2026-07-22, before the user ruled) is
preserved verbatim at `archive/width-superseded-histories.md` §decision-20-draft. The RULED block
above supersedes it in full — the draft is history, not guidance.

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
- ⏸ **PARKED 2026-07-23 (user ruling at the joint-review re-pass — superseding the 📌 QUEUED
  status this item carried for a few hours): the re-grade of `per_width_affine_needed` parks with
  the rest of the normalisation thread; the field stays QUARANTINED and un-citable until the
  thread reopens.** Supporting evidence recorded at the same re-pass (root, ledgers re-read off
  disk): the ALL-schedule `frozen.json`'s own mean-graded copy of the field is ALSO true, but its
  triggering widths MOVE — sandwich triggers at widths 5/6, ALL-schedule at widths 2/7 (rel. gains 0.120 and 0.143 vs the 0.100 threshold) <!-- source: `automl_package/examples/capacity_ladder_results/WSEL15_ALLSCHED/frozen.json` (`affine_detail.rel_gain_c_over_b_by_width`) -->
  — a trigger that wanders across schedules is lottery-consistent, strengthening the quarantine.
  Original re-grade rationale kept below for when the thread reopens. The frozen field is TRUE against the
  pre-registered FALSE prediction, but its triggering widths (5 and 6, relative gains 0.222 and 0.180 vs the 0.100 threshold) sit INSIDE the mid-width lottery zone this very follow-up established <!-- source: `automl_package/examples/capacity_ladder_results/WSEL15/frozen.json` (`affine_detail.rel_gain_c_over_b_by_width`) -->
  (the grading grid was the sandwich grid, 3 seeds), while at the wide end the affine HURTS
  (C/B ratios 1.13-1.25 at widths 10-12, same ledger). Re-read arms b vs c per-seed-paired from
  `automl_package/examples/capacity_ladder_results/WSEL15_ALLSCHED/` and re-grade the field;
  **its current value may not be cited by the report until re-graded.**
- ⏸ **PARKED (user, 2026-07-22): all further normalisation work** — the two spec-excluded variants
  (mean-centred/true LayerNorm; the width-independent normaliser, literature still unsurveyed) stay
  closed until after the pending architecture work settles which width architecture is best and how
  it compares. Reopen deliberately then; do not fold into any current task. *(2026-07-23 note:
  "settles" now concretely = WSEL-23's exhaustion end-state — the reopen trigger has a task.)*
  **WIDENED 2026-07-23 (user ruling at the joint-review re-pass): the parking now covers the ENTIRE
  normalisation thread, including the queued affine re-grade above (struck from the review wave).**
  Rationale, user's words: we first need a width architecture that works WITHOUT normalisation —
  we don't have one (separately trained per-width networks still beat the nested structure, the
  open research the review wave exists to run) — so how normalisation behaves on top of a
  not-yet-chosen architecture is not decision-relevant. Reopen trigger unchanged: WSEL-23's
  exhaustion end-state.

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
> ✅ **FROZEN-BIAS DIAGNOSTIC RUN 2026-07-22 (root, before interpreting the stage-1 accuracy
> failure): the mechanism is REAL but does NOT explain the failure.** The saved readout bias is
> bit-for-bit at its init value on 3/3 tier-1 seeds (proving zero gradient from disk), yet a
> closed-form 1-dof bias refit on the train split closes at most ~2% of the full-width gap to the
> multi-head reference on any seed — a scalar offset cannot fake capacity. Greedy training remains
> the live explanation; stage 2's arms (which separate greedy from moving-target) are the right
> probe. Artifact: `automl_package/examples/capacity_ladder_results/WSEL16/frozen_bias_diag.json`
> (script beside it, reproduces the landed full-width MSEs exactly before repairing). <!-- numcheck-ignore: the ~2% is a rounded readback of the cited diagnostic JSON's fraction_of_gap fields -->

> 🔄 **GRID: controls first (Step 3) launched 2026-07-22 — root-run, backgrounded, sequential.**
>
> ✅ **STAGE 2 COMPLETE 2026-07-22 (root-run, 9 cells, tier 1, sandwich schedule, seeds 0/1/2;
> all cells + `frozen.json` re-summarized). THE DIAGNOSTIC ANSWER: GREEDINESS is the binding
> constraint, NOT the moving target.**
> - **Corrective (stop-grad + periodic joint bursts) — best single-head recipe, still a PRIMARY
>   FAIL.** RESULT: full-width held-out MSE 0.089 / 0.193 / 0.098 vs the multi-head reference's 0.024 / 0.031 / 0.028 (`automl_package/examples/capacity_ladder_results/WSEL16/frozen.json`)
>   — 3.7-6.3x the reference against the 10% bar; ordering retained (rho −0.71/−0.58/−0.33).
>   ⚠️ Its convergence flags are structurally unreadable: the burst schedule makes the val
>   trajectory non-monotone BY CONSTRUCTION, so the gate certifies only 2-4/12 widths and flags
>   4-6/12 "diverged" — while endpoint train ≈ val shows no actual divergence. Recorded as an
>   instrument-mismatch (the gate was designed for monotone-ish single-loss runs); the primary
>   verdict does not hinge on it (the ~3.7x best-seed gap dwarfs any reading of the flags).
> - **Distillation-target — dynamical runaway, clean negative.** Self-referential target (all
>   narrow widths chase `detach(S_wmax)`, which inflates in response): 12/12 widths diverged on
>   every seed, fails the TRAIN set itself. RESULT: full-width val 315.541 / 53.679 / 158.210 (`automl_package/examples/capacity_ladder_results/WSEL16/frozen.json`).
> - **Staged cascade (no moving target at all) — converges PERFECTLY and still fails.** 12/12
>   widths trustworthy on every seed, train ≈ val, yet RESULT: full-width val 0.209 / 0.189 / 0.337 (`automl_package/examples/capacity_ladder_results/WSEL16/frozen.json`)
>   — 6-12x the reference. Extra per-prefix readout bias freedom named in every cell JSON.
> - **Inference: removing the moving target entirely (cascade) does not rescue accuracy; the
>   only recipe that periodically un-greeds (corrective) is the best single-head result. Strict
>   importance-ordered training pays multiples of the reference MSE in every recipe tried.**
>   (Consistent with the frozen-bias diagnostic above: the failure is the training principle,
>   not an implementation artifact.)
> - **stage2 winner: NONE — `stage1_winner = b_heads` stands. ⇒ Per the 2026-07-22
>   selection-studies coupling ruling: NO single-head recipe lands competitive, so WSEL-6/7/8
>   run MULTI-HEAD ONLY and single-head is recorded as ordering-success / accuracy-failure.**
> - **Stage 3 (the generality check): NOT RUN — the contrast is EMPTY.** Its spec compares
>   "B_HEADS and the stage-1/stage-2 winner"; the winner IS B_HEADS, so the two-finalist tier-3
>   block collapses to a self-comparison. Logged as the reversible default under Decision 32;
>   batched for user review (if certified-design tier-3 canonical-suite coverage is wanted for
>   its own sake — e.g. for WSEL-17's supersession argument — it is a cheap standalone launch).
>   **RULED at the 2026-07-22 sign-off (item 3 CLOSED, user): coverage APPROVED, single-arm —
>   `B_HEADS` alone across the 36 tier-3 ladder cells (§3.8); the comparison half stays empty by
>   construction. NOT launched during the review (user instruction: no runs mid-sign-off);
>   queued to execute AFTER the review concludes. Prerequisite: the driver excludes the stage-3
>   ladder by design (`width_wsel16.py:90`, `--tier` accepts 1/2 only) — a small tier-3
>   extension is a separate authoring contract first; §3.8 requires tier 3 to train on the
>   fixed-sigma weighted squared error from `width_candidates.py` (landed with WSEL-15), never
>   plain MSE. The residual-recipe convergence audit raised at the same sitting (trajectories
>   re-read off disk: 36/36 widths converged, 0 hit_cap, no overfitting signature, corrective's
>   sawtooth = burst schedule not memorization, reported numbers = best-val snapshots on an
>   independent split) is recorded in the sign-off transcript; the stage-2 verdict stands
>   unchanged.**
>   **AUTONOMOUS HAND-OFF (user, 2026-07-22: "once all sign-offs are done, you can start the
>   orchestration"): the moment sign-off items 4-6 close, the root proceeds WITHOUT further user
>   input under the Decision-32 regime — (wave A) dispatch the tier-3 driver-extension authoring
>   contract; (wave B) root runs the 36-cell grid backgrounded, each cell landing as produced;
>   (wave C) `--summarize`, commit results, execute WSEL-17 Step 7 per the signed-off manifest,
>   then the local merge to master per MASTER's branch protocol. User-only questions are batched
>   to the end; only irreversible/destructive actions outside the ratified manifest halt.**
>   **✅ STAGE-3 COVERAGE RUN 2026-07-22 (root, wave B): 36/36 cells landed, 0 launch failures,
>   `hit_cap` false everywhere — ledger `automl_package/examples/capacity_ladder_results/WSEL16/tier3_summary.json`
>   (aggregator script beside it; the driver's own `summarize()` predates tier 3).** Protocol
>   note, recorded not buried: the cells ran the stage-1 SANDWICH comparison protocol (the
>   driver default and the certification-era protocol); the Decision-31 ALL default governs
>   future training — an ALL-schedule ladder pass is an option, deliberately NOT scheduled.
>   **Quality verdict: coverage HOLDS.** The canonical cell reproduces as the (n=1500, σ=0.05)
>   slice; no cell collapses anywhere on the ladder; seed bands tighten with n (10.92-11.13 <!-- source: `automl_package/examples/capacity_ladder_results/WSEL16/tier3_summary.json` (`mse_over_noise_floor`, n=4000/σ=0.05 cells) -->
>   MSE-over-noise-floor at n=4000, σ=0.05) and the error approaches the noise floor as noise
>   dominates (2.84-2.98 at n=4000, σ=0.5). <!-- source: `automl_package/examples/capacity_ladder_results/WSEL16/tier3_summary.json` (`mse_over_noise_floor`, n=4000/σ=0.5 cells) -->
>   **Integrity, quantified:** the verdict-bearing full-width head is trajectory-certified in
>   28/36 cells; <!-- source: `automl_package/examples/capacity_ladder_results/WSEL16/tier3_summary.json` + per-cell `convergence['12']` fields -->
>   6 exceptions are slow-creep "still improving at stop" (converged at plateau granularity),
>   and 2 (seed 1, n=200, σ ∈ {0.15, 0.5}) are gate-diverged small-data wobble with endpoint
>   train ≈ held-out and in-family quality (8.03 / 3.72 MSE-over-floor vs same-(n,σ) neighbours
>   spanning 7.08-17.68) — reported numbers are best-validation snapshots. <!-- numcheck-ignore: the neighbour ranges span multiple per-cell JSONs summarized in tier3_summary.json's cells array -->
>   Mid-width per-cell readouts from this grid inherit the KNOWN sandwich mid-width-lottery
>   caveat (WSEL-15) — do not consume mid-width profiles from these cells without it.
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

### WSEL-18 — vectorise the multi-head readout (user ruling 2026-07-22: FIRST in the multi-head queue) — ✅ **DONE 2026-07-22, root-verified; benchmark outcome MIXED — RATIFIED at the 2026-07-22 sign-off: ALL stays the default (see below)**

> **LANDED 2026-07-22 (worker-authored, every verify line re-run at the root, prove-it-fails
> ceremony re-run against the committed state — mask dropped → 2 loud failures → restored →
> 21/21).** Fused mode on `SharedTrunkPerWidthHeadNet` behind a constructor flag (default OFF;
> 44 tests across the three suites touching the class confirm unfused paths unchanged); hard
> error on fused + any schedule other than ALL; equivalence exact (values float32-exact, grads at
> float64 tolerance, masked entries pinned EXACTLY zero after 200 real optimizer steps).
>
> ⚠️ **THE ACCEPTANCE BENCHMARK PARTIALLY FAILS THE "dominance, no decision" PREMISE — batched
> for user review (Decision 32 item 2 sweep).** The premise verified: fusion removes the
> per-tensor dispatch/bookkeeping premium (~2.3x vs unfused ALL, consistent with the WSEL-14
> cost probe's attribution). The premise NOT fully borne out: RESULT: fused ALL 1.697 ms/step vs per-head sandwich 1.404 ms/step (ratio 0.827) on the canonical cell — `automl_package/examples/capacity_ladder_results/WSEL18/bench.json` —
> i.e. after fusion a **~21% per-step premium remains, and it is ARITHMETIC (training w_max
> widths vs the sandwich's 4), not bookkeeping** — fusion cannot remove it, and it grows with
> `w_max` (coverage-vs-cost trade at scale, §3.10). **Decision 31's accuracy/variance grounds
> are untouched** (never-less-accurate + 6x lower mid-width variance stand); only the "cost
> premium is an implementation artifact" clause is now HALF-true: dispatch yes, coverage no.
> ALL remains the default per the ruling; this note corrects the cost rationale's scope.
> **RATIFIED at the 2026-07-22 sign-off (item 2 CLOSED, user): ALL STAYS THE DEFAULT — the
> accuracy/variance grounds carry the ruling on their own, and the ~21% coverage premium is
> accepted as the honest price of training every width every step. The user re-affirmed §3.10 in
> the same ruling: 12 widths is this benchmark's sample point, never a design constant — real
> problems may use a different width count entirely (even ~100), so the premium is reported WITH
> its scaling law, never as a fixed number.**
> Stage-3's multi-head cells are UNGATED (the fused mode is available and verified). The ALL-schedule cost premium is per-head bookkeeping (12 separate tensors each paying
forward/backward/optimizer dispatch — the WSEL-14 cost probe's attribution), not arithmetic. Fusing
the heads into ONE lower-triangular-masked `(w_max, w_max)` weight tensor + bias vector removes it.
Under the ALL schedule the fusion is MATHEMATICALLY EXACT: every head receives gradient every step,
and elementwise optimizer updates on one tensor equal per-tensor updates. Under sampling schedules
it is NOT exact (zero-gradients vs no-gradients — the WSEL-14 Step-2 footgun) and must be refused.

**Files (write set):** `automl_package/models/flexnn/width/architectures.py` (fused mode on
`SharedTrunkPerWidthHeadNet`, constructor flag, DEFAULT OFF) ·
`automl_package/examples/kdropout_converged_width_experiment.py` (flag threading + the
fused×non-ALL hard error) · `tests/test_fused_heads_equivalence.py` (Create) ·
`automl_package/examples/capacity_ladder_results/WSEL18/` (Create by the acceptance benchmark).

**Spec.** (1) Fused parameterisation with the upper triangle hard-masked; init maps the per-head
weights row-for-row so a fused net can be constructed equal to a per-head net. (2) Equivalence
test: outputs and gradients equal per-head at float tolerance on fixed seeds; masked entries
remain EXACTLY zero after real optimizer steps (pinned, not assumed); per-width best-weight
snapshots slice rows and round-trip. Prove-it-fails included. (3) Hard error on fused + any
schedule other than ALL. (4) Acceptance benchmark, committed as
`automl_package/examples/capacity_ladder_results/WSEL18/bench.json`: fused ALL-schedule per-step
wall-clock vs per-head SANDWICH per-step on the canonical cell — the measurement that verifies the
"dominance, no decision" premise.
**Non-goals:** no default flip (flag off; existing paths byte-identical), no other architectures,
no use in any grid until the equivalence test is green.
*Orchestration:* parallel: yes (disjoint from WSEL-16 stage-2 authoring) · deps: none ·
tier: sonnet high · **gates: WSEL-16 stage-3's multi-head cells** · verify:
`AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m pytest tests/test_fused_heads_equivalence.py -q` PASS
with prove-it-fails shown; driver `--selftest` PASS with the flag off AND on; the benchmark JSON
exists and carries both per-step numbers.

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
**Write-set amendment, 2026-07-22 (root, logged reversible default under the autonomous mandate):**
this line predates the module reorganisation — `automl_package/models/architectures/nested_width_net.py`
is now a pure re-export shim and the class bodies live at `automl_package/models/flexnn/width/architectures.py`
(with `ResidualCascadeNet` at `automl_package/examples/cascade_width_net.py`). The Steps 1-6 worker
correctly declined to touch those un-named real homes; **Step 3's variance-status docstrings for the
three undocumented classes (`NestedWidthNet`, `IndependentWidthNet`, `ResidualCascadeNet`) were
completed by the root across the two real homes** — documentation-only edits, no behaviour change.
The write set is read as covering the named modules' REAL homes after the reorganisation.
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
- [x] **Step 7 — attended deletion. ✅ EXECUTED 2026-07-22 (root, autonomous run — front-loaded
  from wave C the moment the rulings made it unblocked):** the three ratified files deleted in
  one commit, manifest updated to 40 KEEP / 2 SHIM / 3 DELETED, importer/reference safety
  re-derived by grep immediately before the `git rm` (zero genuine importers; W_INDEP emptied).
  `git` history retains every file, so this is recoverable; said so in the commit body.
  **⛔ SIGN-OFF TAKEN 2026-07-22 (item 6 CLOSED, user — three rulings, execution delegated to the
  autonomous post-review run's wave C):**
  1. **Citation-gate scope AMENDED (the headstone finding fixed at the rule):** pure
     audit/inventory documents — `shared/zero-caller-inventory.md` and any future disposal-audit
     doc — are EXCLUDED from the citation walk. A mention that nominates a file for disposal is
     not a dependency. The manifest's c2 column is re-read under this scope at Step-7 execution.
  2. **DELETE `automl_package/examples/independent_width_experiment.py` + <!-- citecheck-ignore: deleted at Step 7 2026-07-22; this line is part of the deletion record -->
     `automl_package/examples/capacity_ladder_results/W_INDEP/w_indep_summary.json`** — superseded <!-- citecheck-ignore: deleted at Step 7 2026-07-22; this line is part of the deletion record -->
     by name (`converged_width_experiment.py` docstring), zero genuine importers, unprotected;
     was blocked solely by ruling 1's defect. Consequence recorded: `hetero_width_experiment.py`
     loses its only genuine importer and becomes zero-caller, but stays KEEP (fails c4 — no named
     replacement); a future wave may name one and retire it.
  3. **DELETE `automl_package/examples/capacity_ladder_results/W_MRL/w_mrl_summary_run1_partial.json`** <!-- citecheck-ignore: deleted at Step 7 2026-07-22; this line is part of the deletion record -->
     — partial run superseded by its completed directory-mate, zero references repo-wide; the
     producer's directory-umbrella protection is ruled NOT to cover this leaf (explicit leaf
     exemption, recorded here).

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
