# Strand: depth — feed-forward depth (PRIMARY) and depth selection

**Owns depth from the certification forward.** Read `MASTER.md` + this file — that is the whole
context. If another document disagrees with this one about feed-forward depth or about depth
*selection*, **this one wins and the other is a bug to fix**.

**🔑 THIS STRAND IS FEED-FORWARD DEPTH. THE RECURRENT ARM IS PARKED (user, 2026-07-20, restated).**
The object is the **plain feed-forward net with distinct weights per layer**, and nothing else.

**⛔ RECURRENT = PARKED. Do not run it, do not build for it, do not propose it.** The certified
recurrent result (`depth.md`, `G-DEPTH = PASS`) stands as history and may be *cited*; it is not an
arm, not a reference, and not a positive control in this strand. Any task below that would spend
compute on the recurrent architecture is out of scope — if a task appears to need it, that task is
wrong and gets rewritten, not the ruling. **A proposal to "just re-run recurrent to check X" is the
specific thing this ruling forbids.**

**Consequence, logged as a default taken rather than a question asked:** MASTER Decision 14 requires
a known-good arm to run first. With recurrent parked, the positive control lives **inside the
feed-forward arm** — the feed-forward net trained on the all-rungs scheme must reach the bar at full
depth before any per-depth reading is trusted, with the parameter-matched wide-shallow net as the
falsifier. That is a self-contained control and it needs no recurrent run.

**Division of ownership, explicit, because duplicate ownership is this programme's recurring failure:**
- `depth.md` owns the **certification** (`G-DEPTH = PASS`, 2026-07-17: substrate ∧ selection, on the
  recurrent arm with supervision at every depth). **CLOSED. Cited here, never restated, never reopened.**
- **This file absorbs the feed-forward depth study out of `flexnn-core.md`** (the F5/F5b/F5c block),
  which closes the split that file has been pending. `flexnn-core.md` retains no depth content after
  task DSEL-0.
- **Package structure, the shared API, and de-duplication of code are NOT owned here** — they are
  owned by `flexnn-package.md`. This strand consumes that API; it does not define it.

---

## 0. Cross-plan dependencies

Every task in this strand that blocks on a sibling plan, by exact task id. Nothing else in this
strand depends on another plan — a dispatcher can read this table alone to know what to wait for.

| Task here | Depends on | What it needs |
|---|---|---|
| DSEL-4 | `flexnn-package.md` FP-3, FP-5 | the consolidated selection API + the reconciled router |
| DSEL-5 (tracked dependency, **not a dispatchable task**) | `flexnn-package.md` FP-6 | the independent-weights class's missing per-depth primitive |
| DSEL-6 | `flexnn-package.md` FP-9, FP-3 | the shared cheapest-within-tolerance selection primitive + the API it is reached through |

---

## 1. What is being compared

Two independent axes. **Mechanism** is the object of study; **choice** is the machinery.

### 1a. The two mechanisms

| Mechanism | Weights | Role in this strand |
|---|---|---|
| **FEED-FORWARD** | distinct weights per layer | **the primary object.** No result of any kind exists |
| **RECURRENT** | one block reused at every step | ⛔ **PARKED — not an arm, not a control, no compute.** Certified in `depth.md`; cite only |

### 1b. The three ways of choosing a depth

**D-SHARED and D-PERINPUT are read off the SAME trained network** — trained on every rung (DSEL-1b).
Training is therefore not a variable between them; they differ in exactly one thing, how the depth is
chosen.

⚠️ **D-SWEEP DOES NOT TRAIN ON EVERY RUNG (user ruling, 2026-07-20). Each of its per-depth models is
trained ORDINARILY at its own fixed depth** — that is what "a network dedicated to that depth" means.
A per-depth model trained across the whole ladder is not dedicated to anything; it is the same
network read at a different point and cannot serve as an independent reference. **Deliberate, and not
a confound**: D-SWEEP is not an arm in a controlled contrast, it is the expensive ceiling, and
training dedicated networks is what makes it expensive. The single-difference rule binds
**D-SHARED vs D-PERINPUT**; it does not bind the reference.

| Model | = the net + | How the depth is chosen | Cost | State |
|---|---|---|---|---|
| **D-SHARED** | a cheap held-out read | ONE depth for the dataset | cheap | **does not exist** → DSEL-6 |
| **D-PERINPUT** | the distilled router | a depth **per input** | cheap | exists twice, unreconciled — see §3 DD3 |
| **D-SWEEP** | a per-depth sweep | ONE depth for the dataset, by training a **separate model per depth** | **expensive — the reference** | **does not exist** → DSEL-7. Today's "best fixed depth" is the *same* network read at different exits, which is not an independent reference |

🔑 **EACH MODEL IS THE COMPLETE SYSTEM, INCLUDING ITS SELECTION MACHINERY.** Scored end-to-end and
costed end-to-end; the selection step is *inside* the model, never a companion field beside it.

**Selection rule, fixed for all three:** **cheapest-within-tolerance, NOT argmax** — the smallest
depth whose held-out score is not worse than the best by more than twice a bootstrap-estimated
standard error (the rule published in `docs/reports/probreg_kselection/probreg_kselection.md` §3.2,
imported unchanged so depth's numbers stay comparable). **The same rule applies to D-SWEEP's curve**,
or D-SHARED and D-SWEEP are not answering the same question.

⚠️ **Two different tolerance rules exist across this strand's three arms, and both are legitimate —
do not conflate them.** The rule above (twice a bootstrap standard error) binds D-SHARED and D-SWEEP:
each reads a whole held-out curve across the ladder, so a standard error is estimable at that grain.
**D-PERINPUT does not read a curve.** The distilled router labels each row independently at a flat
relative margin, `DEFAULT_TOLERANCE = 0.25` (`automl_package/models/common/distilled_router.py:57`),
applied by `_cheapest_within_tolerance_labels` (`automl_package/models/common/distilled_router.py:63-66`).
A single row has no distribution to bootstrap, so no standard error is estimable at that grain — the
flat margin is the only rule available there. **Consequence, stated so the report does not paper over
it:** D-PERINPUT's per-input depth choices are not directly comparable to D-SHARED's/D-SWEEP's on
tolerance grounds — the three arms share a cost objective, not a shared statistical selection rule —
and **DSEL-12's report must say so explicitly**, not merely list the two tolerance values side by
side. Comparison across the three arms lives on held-out error and cost, not on tolerance.

**Confound doctrine (MASTER Decision 15):** an arm differing from its comparator in more than one
respect is NOT dispatchable. State the single difference in the task, or do not run it. **This
strand exists because that rule was broken** — see DSEL-1.

---

## 2. State

**Established (recurrent arm only).** `G-DEPTH = PASS`, 2026-07-17. Substrate: one weight-shared net
serves every per-input depth on a group-composition task where depth is provably irreducible to
width, all bars on 3/3 seeds, holding at exact parameter parity against a wide-shallow control.
Selection: a distilled, oracle-free router picks each input's depth and deploys it at a real compute
saving, 2/2 seeds. Evidence: `docs/depth_capacity/verdict_per_input_depth.md`, `depth.md`.

**Established (negative, and load-bearing).** Smooth one-dimensional regression toys **cannot** carry
a depth signal. Four candidates failed, including the port of the width toy with depth substituted for
width ("stall is a cliff"). The failures are explained by a theorem — for smooth targets, depth-hunger
and gradient-descent learnability are in tension (Malach & Shalev-Shwartz 2019; Malach et al. 2021).
This is why the combinatorial construction is used and **it is not to be relitigated.**
Evidence: `docs/depth_capacity/depth_toy_negative_note.md`, `CHANGELOG.md` 2026-07-16.

**NOT established — each is a task below:**
- **Feed-forward depth. Nothing.** The first pilot was ruled invalid on four grounds; its replacement's
  positive control failed on both seeds. → **DSEL-1, DSEL-2**
- **D-SHARED and D-SWEEP do not exist**, for either mechanism. → **DSEL-6, DSEL-7**
- **Does the cheap choice pick what the sweep picks.** Never asked, for either mechanism. → **DSEL-10**
- **How much data depth selection needs.** Never measured; fixed at 3000/stratum for wall-clock
  reasons. → **DSEL-8**
- **Whether the router's architecture matters for depth.** No study exists. The only sensitivity
  evidence in the programme is width's, which is a different strand, one dimension, and a
  does-it-break check. → **DSEL-9**
- **Any depth report.** → **DSEL-12** (toys-only; no baseline, no real data — DSEL-11 is ⏸ PARKED,
  user ruling 2026-07-20, not extending width/depth's real-data exemption)

---

## 3. Known defects

**DD1 — FIXED 2026-07-20, commit `84ad94d`.** The prefer-shallow depth prior `linspace(3.0, 1.0, ...)`
was removed from `FlexibleHiddenLayersNN` because it caused complete depth-collapse to the prior; the
comment recording the removal is at `automl_package/models/flexible_neural_network.py:300`. It was
**also still present in `IndependentWeightsFlexibleNN`**, at
`automl_package/models/independent_weights_flexible_neural_network.py:306` and `:312` — the fix had
reached one twin and not the other. Both sites now use `torch.zeros(self.max_hidden_layers, ...)`
(uniform prior), with a comment cross-citing the sibling's removal. Regression test:
`TestDD1IndependentWeightsUniformPrior` (`tests/test_phase2_flexible_nn.py:410`). Found already fixed
during the 2026-07-21 repair audit — this file's DSEL-3 task had not yet been updated to match. →
was **DSEL-3**, now closed (see DSEL-3 below).

**DD2 — FIXED 2026-07-20, commit `84ad94d`.** `FlexibleHiddenLayersNN.predict(inference_mode="routed")`
returned a mean from the routed depth while `predict_uncertainty` — which had **no mode parameter at
all** — always computed its spread from the soft selection. A mean and an uncertainty from
*different depths*, returned together as if consistent, no error, no warning. `predict_uncertainty`
now takes `hard_execution: bool = False`, matching the flag a paired `predict()` call used, so the
spread comes from the same selected depth as the mean it accompanies. Regression test:
`TestDD2PairedDepthUncertainty` (`tests/test_phase2_flexible_nn.py:464`). Found already fixed during
the 2026-07-21 repair audit, same as DD1. → was **DSEL-3**, now closed (see DSEL-3 below).

**DD3 — OPEN, structural.** The certified depth-selection result was produced by a router written
inside `automl_package/examples/depth_selection_toy.py`, which imports nothing from
`automl_package/models/`. The package ships a different implementation of the same idea
(`automl_package/models/common/distilled_router.py`). **The thing that was certified and the thing
that ships are not the same code**, and neither has been run against the other. **Corrected count
(2026-07-20, verified by grep):** exactly FOUR router MLP classes exist —
`automl_package/examples/capacity_ladder_k6.py:75`, `automl_package/examples/capacity_ladder_t2.py:233`,
`automl_package/examples/depth_selection_toy.py:607`, `automl_package/models/common/distilled_router.py:84`
— and about nine example scripts *touch* one of them by import or reuse rather than defining their
own. Reconciling the four implementations is owned by `flexnn-package.md`; this strand's dependency
on that reconciliation is stated in DSEL-4.

**DD4 — OPEN, capability gap.** `IndependentWeightsFlexibleNN` has no `fit_router`, no
`inference_mode`, no convergence gating, and lacks the per-depth forward primitive every other
family's router is built on. It cannot participate in any of §1b's three choices until that primitive
exists. Owned by `flexnn-package.md` FP-6; gates DSEL-5.

---

## 3.5 Autonomous execution contract

The root is dispatcher + verifier. Every branch below has a **pre-authorised default**; take it, log
it, keep going.

| Branch | Pre-authorised default | Log |
|---|---|---|
| **DSEL-1**: loops differ ONLY in supervision | record the finding (see DSEL-1), unblock DSEL-2 | the diff table |
| **DSEL-1**: other differences survive | name each, rebuild the contrast one variable at a time, do NOT record a finding | each residual difference |
| **DSEL-8**: which selection fraction becomes the default | the **smallest** fraction at which every arm is within its own noise band of its best; if none saturates, take the largest swept and record **inconclusive, floor not found** | fraction + curve |
| **DSEL-8**: D-PERINPUT still improving at the largest fraction | freeze the largest and mark its battery result **"router data-limited"** — a loss then does NOT support "per-input depth does not pay" | the mark, prominently |
| **DSEL-9**: conclusions invariant to router architecture | keep the frozen default; record invariance | table |
| **DSEL-9**: NOT invariant | adopt the **smallest** configuration reaching the plateau as **depth's own default; report it to the root, do not write it globally** (cross-strand repair, 2026-07-21 — see DSEL-9's doctrine clause below), re-run DSEL-8 against depth's own reported default | old → new + why, plus the report to the root |
| **ceiling binds** (selected depth = max) | extend the ladder one rung, re-run that cell, report the raise | which cells |
| a document contradicts §1 | **§1 wins**; fix the other document the same turn | the correction |

**HALT and ask — these only:**
1. A **positive control fails** (MASTER Decision 14) — the numeric bar is §3.6's
   `positive-control bar` row: held-out accuracy ≥ 0.90 AND `trustworthy=true`, on BOTH seeds.
2. A study is **incoherent rather than merely negative** — a broken instrument.
3. Any change to **§1's definitions, the primary-object ruling, or the selection rule.**
4. Anything **irreversible or outward-facing** (deleting artifacts, publishing, **pushing to
   `origin`**). *(Amended 2026-07-21, user: COMMITTING per the `MASTER.md` branch protocol —
   wave-branch commits, local merge, branch delete, docs straight to `master` — is
   PRE-AUTHORIZED for the autonomous run and is no longer a HALT trigger. Pushing, publishing
   and deletion remain user-gated; `FP-8` stays attended-only.)*
5. Any result that would **reopen `G-DEPTH = PASS`.** That gate is closed; reopening it is a user
   decision, never a run's.

## 3.6 Constants the studies FREEZE, and the battery READS

🚫 **DO NOT WRITE THE VALUES INTO THIS TABLE.** The plan holds the *name* of the constant and the
*path of the artifact that owns it*; the value is read from that artifact at build time. (Exception,
stated explicitly where it applies below: a constant **imported unchanged from a source outside this
strand** is not "written into the table" in the sense this rule forbids — it is cited, not
re-measured, and its number is stated once in prose next to its citation so no task has to go hunting
for it.)

| Constant | Set by | Owning artifact |
|---|---|---|
| supervision regime for the feed-forward study | **DSEL-1** | `docs/plans/capacity_programme/shared/dsel1_nested_diagnosis.md` |
| initial depth ladder | **DSEL-1b** | `automl_package/examples/capacity_ladder_results/DSEL1b/frozen.json` |
| selection-set fraction | **DSEL-8** | `automl_package/examples/capacity_ladder_results/DSEL8/frozen.json` |
| D-PERINPUT data-limited flag, per dataset | **DSEL-8** | same file, `perinput_data_limited` field, one boolean per (toy, arm) |
| router hidden / depth / epochs / lr | **DSEL-9** | `automl_package/examples/capacity_ladder_results/DSEL9/frozen.json`, else the frozen defaults at `automl_package/models/common/distilled_router.py:57-60` if invariant |
| ~~labelling tolerance~~ | ~~DSEL-9~~ | **struck 2026-07-21 (cross-strand repair — MASTER Decision 18 rules the labelling-tolerance sensitivity sweep NOT scheduled and NOT to be run pre-emptively; DSEL-9's spec listed it before that ruling landed and was never updated to match — the same stale sweep was found in `width.md`'s WSEL-7 and `probreg.md`'s PC. See DSEL-9's spec note below.)** |
| depth ladder ceiling raise (if §3.5's "ceiling binds" branch fires) | **DSEL-10** (DSEL-11 is ⏸ PARKED — would apply there too if a later ruling unparks it) | the per-cell result JSON under `automl_package/examples/capacity_ladder_results/DSEL10/` (or `.../DSEL11/`, dormant) that recorded the bind |
| positive-control bar (feed-forward full-depth control, DSEL-2 (i)/(ii); also the HALT condition in §3.5) | **not set by any task in this strand — imported unchanged** | `docs/plans/capacity_programme/depth-selection.md:1048` (F5c-b's PASS BAR line itself, absorbed into §6 of this file by DSEL-0). **Value, stated once here so no task re-derives it:** held-out accuracy ≥ 0.90 AND `trustworthy=true`, on BOTH seeds. *(Citation repaired 2026-07-20 at the root: it read `flexnn-core.md:235`, which was correct before that file was split 760 → 254 lines and now points at frozen dispatch-wave prose. The citations gate passed throughout — a resolving path is not a correct one. **Re-repaired 2026-07-21: the 2026-07-20 fix landed on the line that then held the F5 task HEADER, not F5c-b's actual PASS BAR line — a resolving path that was still not the correct one, exactly the defect class the note above claims to have fixed. Corrected here to the PASS BAR sentence itself; line number re-verified after this same repair pass shifted the file, since the failure mode being fixed is a stale line number.**)* |

**Feed-forward rule (binding):** if **DSEL-10 or DSEL-12** runs at a value not justified by the
artifact named here, its results are **not reportable**. *(Re-pointed 2026-07-21, cross-strand repair
— this row named DSEL-11, which the 2026-07-20 parking ruling removed from the dispatch order, so the
rule bound nothing; DSEL-11 is ⏸ PARKED and the row above it already flags the consequence for the
depth-ladder-ceiling constant, this row did not. The live consumers before the report are DSEL-10 and
DSEL-12.)*

⚠️ **Anchor warning.** Where a task reproduces a frozen number as its positive control, the anchor
must come from something **not computed by the method under test**. Re-deriving an anchor with the
same code conscripts the worker into confirming our own bug and returns it stamped *verified*.
Concretely: DSEL-10's quality half must be anchored on D-SWEEP's independently trained per-depth
models, not on the certification's numbers re-run through this harness.

---

## 4. Tasks

Order: **DSEL-0 → DSEL-1 → DSEL-1b → DSEL-2 → DSEL-2c → DSEL-2b → DSEL-3 → DSEL-4 → DSEL-6 → DSEL-7 →
(DSEL-8 ∥ DSEL-9) → DSEL-10 → DSEL-12.** *(DSEL-2c inserted 2026-07-21, answering DSEL-2's HALT #1 —
see DSEL-2c and the USER RULINGS recorded in DSEL-2's halt block. DSEL-2b's own read is unchanged by
the insertion: it reads DSEL-2's eventual per-depth ladder, produced once DSEL-2 UNHALTS under
whichever scheme DSEL-2c licenses, never DSEL-2c's own four arms directly.)*

**DSEL-5 is NOT a step in this order line.** It is a tracked dependency on `flexnn-package.md` FP-6
(§0, §3 DD4) — record it, do not dispatch it, do not wait on it in a mechanical scheduler unless
DSEL-6/7 actually route through `IndependentWeightsFlexibleNN`.

**DSEL-11 is NOT a step in this order line either.** ⏸ PARKED (user ruling 2026-07-20, §2, §4
DSEL-11) — real data is deferred for this strand and its spec stays on disk, unscheduled, for a
possible later pass. A mechanical dispatcher must not wait on it: DSEL-12 now runs straight off
DSEL-10.

DSEL-1 and DSEL-2 come first because they carry the primary claim. DSEL-2b measures a property of
DSEL-2's own ladder and does not gate anything downstream. Everything from DSEL-6 on is selection
machinery, and **DSEL-8/DSEL-9 must precede DSEL-10** — both fix a parameter of the method, and
running the battery before they are settled produces results nobody could attribute.

### DSEL-0 — single-source the definitions; absorb the feed-forward study

**Files (write set):** this file **only**
**🚫 NOT in the write set: `docs/plans/capacity_programme/MASTER.md` or
`docs/plans/capacity_programme/flexnn-core.md`** — both are ROOT-ONLY (MASTER naming key, "SHARED
FILES ARE ROOT-ONLY"), and `flexnn-core.md` is READ-ONLY for dispatch until FP-0's dispositions land.
Three sibling tasks (FP-0, WSEL-0, P0) need MASTER edits in this same wave; concurrent writers
produce contradictory text in one file. **Deliverable instead:** emit the exact MASTER text (index
entry + naming-key entry + the `G-DEPTH` scope correction) verbatim in this task's report, and record
the F5/F5b/F5c move as a `move` row for FP-0's `shared/CORE-DISPOSITIONS.tsv` rather than editing
`flexnn-core.md` directly.
**Spec:** Add this strand to `MASTER.md`'s index with the ownership split stated (certification =
`depth.md`, CLOSED; feed-forward + selection = here; package/API = `flexnn-package.md`). Add a naming
key entry that **points here and states no definition of its own**. Move the F5/F5b/F5c block out of
`flexnn-core.md` into this file's history, leaving a pointer, so no depth content survives in two
places. Record as a correction that `G-DEPTH = PASS` certifies the recurrent arm under
supervision-at-every-depth and covers neither the feed-forward mechanism nor any of §1b's choices.
**Non-goals:** no code; no reopening of the depth gate; no edit to `depth.md`'s verdict.
*Orchestration:* parallel: no · deps: none · tier: opus high (definitional) · scale: static ·
shape: design · verify: `grep -c "depth-selection.md" docs/plans/capacity_programme/MASTER.md`
returns ≥2 (one hit in the strand index table, one in the naming key — both already present as of
this repair, so this clause is a regression check, not new work);
`grep -cin "F5c\|feedforward depth" docs/plans/capacity_programme/flexnn-core.md` returns exactly `1`
(today it returns `13` because the full F5/F5b/F5c block is still live in that file; after this task
moves it here and leaves a single pointer sentence, the count must drop to 1); `depth.md` unmodified
(`git diff --stat -- docs/plans/capacity_programme/depth.md` prints no output).

### DSEL-1 — close the halt: the failed run had no nested structure. Record it and move on.

**USER RULING 2026-07-20, and it supersedes both the optimisation escalation ladder in
`docs/depth_capacity/ff_depth_protocol_repair_spec.md` §2 and this task's own earlier framing.**

**The diagnosis is settled, from the code, at the root:**

| | Training step | Nested? | Result |
|---|---|---|---|
| certified recurrent run (`depth_selection_toy.train_anytime:475`) | mean cross-entropy over **every** depth in the ladder, each depth's target being what is actually derivable at that depth | **yes** | pass |
| failed feed-forward run (`depth_composition_toy.train_clf:487`) | `loss = ce(net(x_tr), y_tr)` — **one** target, full depth only | **no** | train 0.970/1.000, held-out 0.432/0.744 |

The failing arm was a plain fixed-depth network trained on a single target. **No sampling over depth,
no per-depth outputs, no nesting.** Every capacity dial in this programme that works is trained the
other way — the classifier samples the number of classes per example; width uses the sandwich
schedule (always the smallest and largest width, plus two random middles) with a head per width; the
certified recurrent depth run trains a target at every step. The feed-forward attempt is the only one
that skipped it, and the only one that failed.

**⇒ The overfitting needs no further diagnosis.** A single deep stack with one target places no
pressure on any intermediate depth to mean anything, so there is nothing to stop at — which is also
why the per-input depth idea has no purchase on that arm. The escalation ladder is irrelevant: every
rung is an optimisation remedy and the optimiser already fits the training set completely.

**This already satisfies MASTER Decision 16** (optimisation exonerated before architecture is
blamed) by inspection, cited here by number so no separate escalation-ladder run is owed: train_acc
0.970/1.000 shows the failed arm already fits the training set, so the failure is not under-fitting —
Decision 16's "low on both train and val ⇒ under-fit" branch does not apply, and the finding may be
read as architectural without running the ladder.

**Spec (all this task does):** write the finding to
`docs/plans/capacity_programme/shared/dsel1_nested_diagnosis.md` with the two code citations above,
and clear the halt marker in `flexnn-core.md` with a pointer to it. **No re-run. No ladder. No further
diagnosis.**

**Defect found while establishing this, and it is in scope here:** the positive-control artifacts
(`automl_package/examples/capacity_ladder_results/D_TOY_PROBES/f5c_poscontrol_a5_seed{0,1}.json`)
**do not record the training-set size.** `n_train`/`n_val` are absent from the JSON, so the question
"was there enough data?" cannot be answered from the artifact — it has to be inferred from the
generator's defaults. Fix the driver to record data sizes in every results file. A run whose data
size is not readable from its own output cannot be audited.

**Files (write set):** `docs/plans/capacity_programme/shared/dsel1_nested_diagnosis.md` ·
`docs/plans/capacity_programme/flexnn-core.md` (halt marker only) ·
`automl_package/examples/depth_composition_toy.py` (results-schema fix only)
**Non-goals:** do NOT walk the escalation ladder. Do NOT re-run the failed control — it is explained,
and the schema fix is verified below by a cheap smoke run, never by re-running the 40,000-epoch
`--poscontrol` that already failed. Do NOT change any training code here; building the nested
feed-forward arm is DSEL-2.
*Orchestration:* parallel: no · deps: none · tier: sonnet high · scale: static · shape: execution ·
verify: `test -f docs/plans/capacity_programme/shared/dsel1_nested_diagnosis.md` exits 0 AND
`grep -c "depth_selection_toy.py:475\|depth_composition_toy.py:487"
docs/plans/capacity_programme/shared/dsel1_nested_diagnosis.md` returns 2 (both citations present);
`grep -n "remain HALTED" docs/plans/capacity_programme/flexnn-core.md` returns nothing (fail if the
F5c-c/F5c-d halt marker, currently at `flexnn-core.md:269`, is still live), and the line that replaces
it cites `dsel1_nested_diagnosis.md` by path; the missing-schema fix is in `run_positive_control`
(`automl_package/examples/depth_composition_toy.py:770-854`, the function `--poscontrol` calls, whose
returned dict at `:816-848` has no `n_train`/`n_val` key today) — demonstrate the fix with a CHEAP
`--poscontrol` smoke run at a fraction of the failed control's budget (50 epochs, not the 40,000 the
forbidden re-run would use), landed to a separate directory so it cannot be mistaken for, or clobber,
the already-explained failed-control evidence:
`AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/depth_composition_toy.py --poscontrol
--seed 0 --max-epochs 50 --out-dir automl_package/examples/capacity_ladder_results/DSEL1/schema_smoke`
then
`python -c "import json; d=json.load(open('automl_package/examples/capacity_ladder_results/DSEL1/schema_smoke/f5c_poscontrol_a5_seed0.json')); assert 'n_train' in d and 'n_val' in d"`
exits 0.

### DSEL-1b — port the nested + capacity-dropout scheme to the feed-forward architecture

**Files (write set):** `automl_package/examples/depth_composition_toy.py` (a new training loop) ·
`tests/test_depth_composition_toy.py` ·
`automl_package/examples/capacity_ladder_results/DSEL1b/frozen.json`
**Spec:** Build the feed-forward analogue of what the other two dials do. Concretely, the design
decisions this task must settle and record, each with the width strand's answer as the precedent:
- **The initial depth ladder has no owner elsewhere in this strand — this task is it.** Freeze the
  ladder used for the group-word substrate and emit it to
  `automl_package/examples/capacity_ladder_results/DSEL1b/frozen.json` (§3.6) as `depth_ladder`; every
  task from DSEL-2 onward reads the ladder from this file, never from a locally hardcoded copy.
- **The readout: an OPEN, PRE-REGISTERED TWO-ARM COMPARISON. Do NOT default to either.**
  *(Corrected 2026-07-20 after review; the earlier draft of this task made a shared readout the
  default and that was an unlicensed transfer — see the method note at the end of this task.)*
  The certified depth result does favour a shared readout — strongest arm on every seed, val acc
  1.000 at ℓ=4/6/8 and ≥0.990 at ℓ=10, at **16,376 parameters against 39,776**
  (`docs/depth_capacity/verdict_per_input_depth.md:70-71`). **But the verdict states the MECHANISM,
  and the mechanism does not carry over.** `verdict_per_input_depth.md:40-43`: one shared network can
  serve every depth *"precisely because the weight-shared recurrent block presents every depth with
  the **same** state space, so one readout suffices."* `depth.md:29-31` states the contrast: readout
  interference is *"WIDTH-SPECIFIC … width prefixes hand each capacity a DIFFERENT representation
  fighting over one readout."*
  **A feed-forward net with distinct weights per layer IS the prefix case** — each depth hands the
  readout a different representation. By the verdict's own mechanism the shared readout is predicted
  to break there, exactly as it broke for width. The certified evidence therefore licenses **no
  default** for this arm.
  **⇒ Run both as pre-registered arms: one shared readout vs one head per depth.** Whichever wins is
  a finding about whether the mechanism is representation-sharing or something else. *(Carry the
  stability caveat on the shared arm: 1 of 6 certified shared-readout runs hit an optimisation
  blow-up — `depth.md:34`, `:136` — flagged by the convergence gate, not hidden.)*
- **Nesting: already correct in the shipping class, and it is prefix-of-layers.** For depth, nested
  means depth *n* executes the FIRST *n* blocks and skips the rest — not run as identity layers,
  simply not executed. `FlexibleHiddenLayersNN` already implements exactly this
  (`automl_package/models/flexible_neural_network.py:29-38`) with a single shared output layer
  (`:133`). **That is the certified-winning shape, already in the package.** This task does not
  redesign it.
- **How the depths are trained: SUM/MEAN THE LOSS OVER EVERY DEPTH, EVERY STEP. No sampling.**
  *(Corrected 2026-07-20 after review. The earlier draft called for a sampled "sandwich" schedule on
  the strength of the phrase "per-depth readout sandwich" at
  `docs/depth_capacity/verdict_per_input_depth.md:70`. The phrase is there; the schedule is not.
  **Both certified depth loops train every rung every step:**
  `automl_package/examples/depth_selection_toy.py:475` — mean CE over every `t` in the ladder;
  `automl_package/examples/depth_graded_toy.py:148` — sum over every length. Width's sandwich
  (`automl_package/examples/nested_width_net.py:93`, "always trains width=1 and width=w_max +2 random
  mid") is a **different scheme on a different axis** and was transferred here without warrant.)*
  **Two nested schemes exist, and both put every rung in the training objective — this is a choice
  between them, not a choice of whether to nest:**
  (a) **per-sample uniform draw over depths** —
  `automl_package/models/selection_strategies/layer_selection_strategies.py:187` `NestedStrategy`
  (confirmed: each sample independently draws `d ~ Uniform{1..max_hidden_layers}` on every forward
  pass and the loss is the readout at that sample's own drawn depth);
  (b) **sum/mean the loss over every rung, every step** —
  `automl_package/examples/depth_selection_toy.py:475` and
  `automl_package/examples/depth_graded_toy.py:148` (confirmed both).
  **⇒ Default: (b), the sum/mean-over-all-rungs scheme**, because that is what the certified depth
  runs used (same two citations). **(a) is recorded as a labelled comparison arm, available if (b)
  proves too slow to train — never a silent alternative.** If (a) is ever run, every artifact it
  produces must name it as the (a) scheme explicitly. Sampling over depths is not the missing piece
  and is not introduced here by default; adopting it as a replacement default would be a Decision-15
  protocol deviation needing its own control arm.
  **The two schemes differ in gradient budget per rung:** (b) gives every rung a gradient
  contribution on every step; (a) gives each rung a gradient only on the steps it happens to be
  drawn. **This is scheme-specific — DSEL-2b's per-depth measurement (below) is a property of
  whichever scheme actually ran and does not automatically transfer to the other.**
  **What the failed feed-forward run actually lacked** was training the intermediate depths *at all*
  (`automl_package/examples/depth_composition_toy.py:487` — a single loss at full depth).
- **What each depth is trained against.** The certified loop's targets are what is *derivable at that
  depth*, not the final answer. Carry that property across; a depth asked for an answer it cannot
  hold is an impossible target and corrupts the ladder.
**Doctrine:** rung 1 and 2 of the minimum-viable-code ladder. The nested prefix structure and the
shared readout **already exist** in `FlexibleHiddenLayersNN` — do not rebuild them. What is genuinely
missing is the sampling schedule, and that exists twice elsewhere (the classifier's per-example class
count, width's sandwich). Reuse; state what you reused.
**⚠️ Method note, binding beyond this task:** a width result is NOT evidence about depth. This
programme pre-registered exactly one such transfer and it was refuted. Any future "width found X,
therefore depth" reasoning must be flagged as a hypothesis and tested, never adopted as a default.
**Non-goals:** no selection machinery yet (DSEL-6/7); no new toy.
*Orchestration:* parallel: no · deps: DSEL-1 · tier: opus high (design) then sonnet high (build) ·
scale: static · shape: design → execution ·
verify: `AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m pytest tests/test_depth_composition_toy.py -q`
passes, where that test asserts every depth in the ladder produces a non-degenerate output on
held-out data (not just the maximum depth) — demonstrate the test is real by truncating the loss sum
to the final depth only, re-running to show the SAME test FAIL, then restoring; `test -f
automl_package/examples/capacity_ladder_results/DSEL1b/frozen.json` exits 0 AND
`python -c "import json; d=json.load(open('automl_package/examples/capacity_ladder_results/DSEL1b/frozen.json')); assert 'depth_ladder' in d"`
exits 0. **No recurrent run** — the control is DSEL-2's, inside the feed-forward arm.

### DSEL-2 — feed-forward depth: does the benefit hold at all?

**Files (write set):** `automl_package/examples/depth_dsel2.py` ·
`automl_package/examples/capacity_ladder_results/DSEL2/`
**Spec:** **The primary claim of this strand, and the only claim that matters.** Using DSEL-1b's
all-rungs training scheme and the depth ladder frozen at
`automl_package/examples/capacity_ladder_results/DSEL1b/frozen.json`, run the feed-forward
(distinct-weights-per-layer) arm on the group-word substrate, with the parameter-matched wide-shallow
net as the falsifier. Answer: does a plain feed-forward net show a per-input depth benefit?
**Doctrine:** MASTER Decision 14 — a known-good arm runs first, alone. **⛔ That arm is NOT the
recurrent net (parked, §1a).** The control is self-contained in the feed-forward arm and has two
parts, both of which must hold before any per-depth reading is trusted:
  (i) the feed-forward net at FULL depth reaches the positive-control bar on held-out data — held-out
      accuracy ≥ 0.90 AND `trustworthy=true`, on BOTH seeds (§3.6's `positive-control bar` row,
      imported from `docs/plans/capacity_programme/flexnn-core.md:235`) — if it cannot clear this bar,
      nothing about intermediate depths is readable;
  (ii) the parameter-matched wide-shallow net does NOT clear the same bar — if width substitutes for
      depth here, the substrate carries no depth signal and the run stops.
**One variable at a time:** this task varies depth only; the training scheme, substrate and protocol
are fixed by DSEL-1b and are not touched here.

**⚠️ CARRIED RULING (root, 2026-07-20) — DSEL-1b's readout comparison is UNRESOLVED and DSEL-2 owns
it.** DSEL-1b ran the two pre-registered readout arms and the result is **not readable as a verdict**:
its own artifacts self-label `readable_as_verdict=false`
(`automl_package/examples/capacity_ladder_results/DSEL1b/readout_comparison_a5_n6_seed{0,1}.json`)
because the `per_depth` arm carries `trustworthy=false` on **both** seeds — a diverged arm did not
lose, it failed to train (Decision 16) — and the two arms differ in **two** ways at once, readout
structure **and** 2.8× parameters (2,780 vs 7,880; Decision 15). **Therefore:**
1. Both readout arms run again here, **parameter-matched** — equalise via trunk width, and record
   both arms' parameter counts in every JSON so the match is checkable, not asserted.
2. Both arms are **dual-gated** (Decision 17): `trustworthy` computed on the metric the bar reads,
   held-out accuracy, **and** on the loss, with both reported. An arm failing the gate is escalated
   through the Decision-16 ladder before any comparison is read from it.
3. **The open design question must be settled in writing, not by default.** The certified verdict's
   "each depth hands the readout a different representation" argument may bind to the **per-step
   consumption** shape (`UntiedPerStepComposer`, letter *t* enters at layer *t*) and NOT to a
   whole-input MLP, where `blocks[0]` sees the entire word at every depth so all depths start from
   identical information. Never specified anywhere. State which shape this task runs, and say
   explicitly which of the two the readout finding does and does not license.

**⚠️ PRIOR EVIDENCE THE CONTROL MUST BE READ AGAINST — do not discover this after spending compute.**
The full-depth feed-forward control has never cleared 0.90 on this substrate at any configuration
tried: DSEL-1b's pilot peaked at val acc **0.28** at full depth on the **shared** readout arm
(`automl_package/examples/capacity_ladder_results/DSEL1b/readout_comparison_a5_n6_seed0.json`); the
**per_depth** arm's ceiling across both seeds is lower, ~0.219 — seed0 max 0.2192 at depth5 (`automl_package/examples/capacity_ladder_results/DSEL1b/readout_comparison_a5_n6_seed0.json`),
seed1 max 0.2153 at depth2 (`automl_package/examples/capacity_ladder_results/DSEL1b/readout_comparison_a5_n6_seed1.json`) —
*(corrected 2026-07-21 — the original text here read "both readout arms," which the per_depth arm's
own numbers do not support)*, and F5c-b's repaired-protocol control on the harder A5/L=10 configuration failed at
**0.432** (`automl_package/examples/capacity_ladder_results/D_TOY_PROBES/f5c_poscontrol_a5_seed0.json`)
and **0.744** (`automl_package/examples/capacity_ladder_results/D_TOY_PROBES/f5c_poscontrol_a5_seed1.json`)
with `train_acc` 0.97/1.00 — memorization-without-generalization, an exhausted Decision-16 ladder, and
an **open user escalation** (§6, F5c-b). ⇒ Clause (i) is a live risk, not a formality. This task may
scale **capacity and data** (trunk width, `n_train`, epoch cap) to give the control a fair chance —
those are not the controlled variable — and must log every such change with its before/after number.
It may **not** change the substrate, the all-rungs training scheme, or the bar. If the control still
fails, that is §3.5 HALT #1: **stop, land the evidence, report; do not run the ladder, and do not
resolve the standing escalation.**
**Non-goals:** **no recurrent runs of any kind.** No selection machinery (DSEL-6/7 build it). No new
toy construction — the smooth one-dimensional lane is closed by theorem (§2) and is not revisited.
*Orchestration:* parallel: no · deps: DSEL-1, **DSEL-1b** · tier: sonnet high (execution against a
fixed spec), opus xhigh for the verdict · scale: dynamic · shape: execution ·
verify: run `automl_package/examples/depth_dsel2.py` for the full-depth control FIRST, alone, on both
seeds; for each seed's control JSON,
`python -c "import json; d=json.load(open(p)); assert d['val_acc']>=0.90 and d['trustworthy']"` must
exit 0 (Decision 14 — non-zero exit HALTS the task); then run the wide-shallow falsifier the same way
and confirm the SAME check exits non-zero (it must NOT clear the bar). Only then run the per-depth
ladder: every `automl_package/examples/capacity_ladder_results/DSEL2/*.json` must contain
`n_train`/`n_val`, the FULL held-out trajectory rather than an endpoint (`hit_cap=False` — MASTER
Decision 9), and `trustworthy` computed on the metric the bar reads (MASTER Decision 17). Any arm
reading low on both train and held-out accuracy must carry the Decision-16 escalation-ladder record
(LR sweep → clipping → warmup → init scheme → normalization) before it is read as an architecture
finding — its absence invalidates that cell's reading.

### DSEL-2 stage 1 RESULT (2026-07-20) — ⛔ **POSITIVE CONTROL FAILS. HALT #1. ESCALATED TO USER.**

Driver: `automl_package/examples/depth_dsel2.py`. Artifacts: ten JSONs under
`automl_package/examples/capacity_ladder_results/DSEL2/`. **Every one was re-checked at the root with
the task's own command; all ten fail `val_acc >= 0.90 and trustworthy`.** Clause (ii), the
wide-shallow falsifier, was correctly NOT run — it is gated on (i). No per-depth ladder, no readout
comparison, no recurrent run.

**The failure is memorization-without-generalization, and optimization is exonerated (Decision 16).**
From the width-32 rung onward the net fits the training set outright — `train_acc` reaching 1.000 (`automl_package/examples/capacity_ladder_results/DSEL2/control_rung6_width64_seed0.json`) —
while held-out accuracy sits between 0.32 and 0.75. Decision 16's exoneration condition is satisfied
with margin, so **no further ladder rung is licensed**: learning-rate, warmup, init and normalization
are remedies for under-fitting, and this is not under-fitting. Best found anywhere, width 32:
seed 0 = **0.747** (`automl_package/examples/capacity_ladder_results/DSEL2/control_rung4_data75_seed0.json`);
seed 1 = **0.590** (`automl_package/examples/capacity_ladder_results/DSEL2/control_rung4_data75_seed1.json`)
— far below the 0.90 bar on both. All runs are patience-stopped with `hit_cap=false` on both gates
(Decision 9), and both the accuracy gate and the loss gate are recorded per run (Decision 17).

**This is the SECOND feed-forward configuration to fail this way**, and the first — F5c-b, §6 — is
still an open user escalation. Two different substrates, ladders and protocols, same shape:
the training set is fitted completely and held-out accuracy does not follow.

**⚠️ ROOT CORRECTION — the data rungs are CONFOUNDED and must NOT be read as a curve.** The worker
reported a "non-monotonic anomaly": more training data appearing to *hurt* held-out accuracy
(0.747 → 0.387 → 0.410 at train fractions 0.75 → 0.80 → 0.90). **That comparison is invalid.** This  <!-- numcheck-ignore: three values from three different evaluation sets, quoted here to be refuted, not asserted -->
toy enumerates its ENTIRE word space (4 generators, length 6 = 4,096 words — `_all_or_sampled_words`,
`automl_package/examples/depth_composition_toy.py:269-280`), so raising the training fraction
**shrinks and rewrites the evaluation set at the same time**: n_val goes 1024 → 819 → 410. Those
three numbers are measured on three different test sets and are not comparable — two variables moved
at once, which Decision 15 forbids. ⇒ **No anomaly is established, and none may be recorded as a
finding.** What survives is the fixed-configuration evidence: at a FIXED 0.75 fraction, on the same
evaluation set, two seeds and two widths all land 0.59–0.75, and widening 32 → 64 moves nothing —
0.747 (`automl_package/examples/capacity_ladder_results/DSEL2/control_rung4_data75_seed0.json`) vs 0.746 (`automl_package/examples/capacity_ladder_results/DSEL2/control_rung6_width64_seed0.json`) — **capacity is not the binding constraint.**

**Also corrected here — DSEL-1b's 0.28 was PARTLY a gate artifact.** DSEL-1b gated on ladder-mean
held-out loss and restored best weights on it, while accuracy was still climbing; fixing the gate to
read the metric the bar reads (Decision 17) moves the same configuration from 0.28 to **0.319** (`automl_package/examples/capacity_ladder_results/DSEL2/control_rung0_gatefix_seed0.json`).

The gate defect was real and is now fixed in this driver; it was **not** the load-bearing cause.

**⛔ THE USER DECISION THIS RAISES** (batched with the standing F5c-b escalation — they are the same
question asked twice): is the feed-forward depth claim to be pursued on THIS substrate at all? The
programme has now spent two protocol-repair cycles here. The options are (a) accept that this
substrate does not carry a feed-forward depth signal and say so as a finding, (b) change the
substrate, or (c) attack generalization directly rather than through the optimization ladder. **No
task may take this decision** — §3.5 HALT #1 and #5.

**USER RULINGS 2026-07-21, recorded verbatim in substance.**

**(i) A fourth option, ahead of (a)/(b)/(c): re-run the positive control at specification before any
of them is chosen.** The as-run control (DSEL-2 stage 1, above) deviated from the ratified DSEL-1b
design in two ways neither of which was flagged as a deviation at the time it ran: no per-rung
sampling arm was ever run (DSEL-1b §1b's scheme (a), the per-sample uniform draw, was recorded as "a
labelled comparison arm, available if (b) proves too slow" and never exercised), and every exit was
trained against the full-word label rather than DSEL-1b's own spec bullet — "what each depth is
trained against... the certified loop's targets are what is *derivable at that depth*, not the final
answer." Both deviations plausibly produce exactly the observed memorization failure on their own, so
the failure as run is not yet clean evidence about the feed-forward substrate — it is evidence about a
control that did not run the design it was meant to test. **⇒ DSEL-2c, below, re-runs the control at
specification before (a)/(b)/(c) is chosen.**

**(ii) "What is derivable at that depth" for the feed-forward net is RULED to mean the prefix
product.** Exit *d* trains against the product of the first *d* generators — the same closed 60-class
space, read at a shorter prefix — not the full-word label. The as-run implementation's
information-deficit argument (an exit sees the whole flattened word, so nothing is informationally
missing at any depth, so full-label training was assumed harmless) confused information with
capacity: depth 1 sees the whole word but cannot compose six group elements, so training it against
the full-word label forces memorization at that exit, and five of the six loss terms (every exit but
the last) inject that same pressure directly into the shared trunk every step. The prefix-product
target removes that pressure by construction — depth *d*'s target is exactly what *d* layers can
compute.

**(iii) READOUT RULING RE-AFFIRMED (user, 2026-07-21): both readout arms stay.** Width certified
SEPARATE per-capacity heads; the certified recurrent depth arm certified the SHARED head; feed-forward
depth decides EMPIRICALLY via the pre-registered parameter-matched two-arm comparison this task
already specifies (above), once a target scheme passes. Neither arm is dropped, neither is a default
— this ruling exists so a future session does not read the shared head's absence from DSEL-2c's arms
as its having been abandoned; it is absent from DSEL-2c only because DSEL-2c's job is settling the
target scheme, not the readout comparison, which stays owned by this task per the two-arm comparison
already specified above and runs once a target scheme clears the bar.

### DSEL-2c — re-run the positive control at specification — ✅ **RAN 2026-07-21. ⛔ ALL FOUR ARMS FAIL. STRAND PARKED (pre-authorized, strand-local).**

**RESULT: no arm clears the bar on either seed** — ledger `automl_package/examples/capacity_ladder_results/DSEL2c/frozen.json` (`cleared_arms: []`, `all_four_failed: true`, `unhalt_scheme: null`), six per-cell JSONs in the same
directory. Bar was full-depth **test** accuracy ≥ 0.90 with `trustworthy` on both gates, both seeds.
Per-cell test accuracies span **0.41–0.65** — not marginal, and not rescued by either corrected
switch (prefix targets, sampled schedule) alone or together.

**THE FAILURE MODE IS GENERALISATION, NOT OPTIMISATION — this is what makes the finding load-bearing.**
Train accuracy is **0.93–1.00 in every one of the six cells** while test sits near 0.5. MASTER
Decision 16 makes an arm low on *both* train and val an under-fitting/optimisation finding; this is
the exact opposite, so the escalation ladder (LR → clipping → warmup → init → normalisation) is
**already exonerated by the training curve** and running it would be wasted compute. The models
memorise their 3,072 enumerated training words and do not generalise to held-out ones.

**Two caveats recorded against over-reading this** *(root, 2026-07-21 — neither is resolvable from
the cells in hand)*:
1. **The re-run made the bar strictly harder in TWO ways at once**, both deliberate repairs: the
   reported number moved from the val set (which had been doing triple duty — stop, restore, report)
   to a genuine held-out **test** split, AND the three-way split leaves less training data. Some of
   the gap to the as-run control is those corrections landing. **The cells cannot separate "the
   substrate was always this weak" from "the honest split revealed it"** — a discriminating contrast
   would be needed, and none is scheduled.
2. **The two convergence gates DISAGREE** on several cells (`trustworthy_ce: false`,
   `trustworthy_acc: true`). Read per MASTER Decision 17 — the gate is taken on the metric the bar
   reads, i.e. **accuracy** — so these are treated as trustworthy. The disagreement is preserved in
   every per-cell JSON rather than resolved silently.

**Branch taken (pre-authorized by this task's own spec + the MASTER autonomous-run authorization; NOT
a run-level decision):** option (a) recorded as an **evidence-backed finding**; `DSEL-2`, `DSEL-2b`
and the downstream feed-forward selection studies (`DSEL-4`, `DSEL-6`…`DSEL-12`) are **⏸ PARKED**
pending user review; the other three strands **continue uninterrupted**; batched for end-of-run
review. **Choosing among options (a)/(b)/(c) is explicitly NOT the run's to take** — it goes to the
user as evidence, per this task's non-goals.

**Driver defect found and fixed at the ROOT while landing this** *(2026-07-21)*: `freeze_dsel2c`
read `seeds[s]["_path"]`, a key the per-cell writer never stores, so the aggregation raised
`KeyError: '_path'` **after every cell's compute had already succeeded** — the verdict was
unlandable despite the run being complete. Path is now DERIVED from the file-naming contract
(`automl_package/examples/depth_dsel2.py:855`), and the frozen record additionally carries
`val_acc`, `train_acc` and `trustworthy_acc` per seed, since the train/test gap is the whole finding.

*(Original task spec follows, retained verbatim as the pre-registration this run was judged against.)*

### DSEL-2c — re-run the positive control at specification (USER RULING 2026-07-21)

**Files (write set):** `automl_package/examples/depth_dsel2.py` (extended, not rewritten) ·
`automl_package/examples/capacity_ladder_results/DSEL2c/`
**Spec:** Two training-scheme switches on the existing dual-gate loop, giving four arms:
- **targets:** FULL (as-run; every exit trained against the full-word label) vs **PREFIX** (exit *d*
  trained against the product of the first *d* generators — ruling (ii) above);
- **schedule:** ALL_RUNGS (as-run, DSEL-1b scheme (b)) vs **SAMPLED** (per-sample *d* ~
  Uniform{1..6}, DSEL-1b's pre-registered scheme (a), never yet run).
(FULL, ALL_RUNGS) is the existing rung4/rung6 control — cite its artifacts, do not re-run it. Run the
other three arms × 2 seeds at the best-found config (width 32, `train_frac` 0.75), shared readout only
(single-difference against the as-run control; the readout comparison stays owned by DSEL-2 per ruling
(iii) above, and runs once a target scheme passes here).
**Three-way split, binding (repairs a defect in the as-run control).** The as-run control's validation
set did triple duty — early stopping, best-weight restore, and the reported number — making the
reported accuracy a model-selection score rather than a held-out one. DSEL-2c splits the enumerated
words train/val/test = 0.75/0.125/0.125; <!-- numcheck-ignore: protocol split ratios, not result numbers --> stopping and best-weight restore read val as before; **the
reported number is TEST accuracy**, never val. Trajectory gates (Decision 9) keep logging val, as
they always have.
**Bar unchanged:** full-depth **test** accuracy ≥ 0.90, `trustworthy=true` on both gates, both seeds
(§3.6's positive-control-bar row, read against test rather than val per the split above). Any
corrected arm clearing the bar → DSEL-2 UNHALTS under that arm's scheme, recorded as the strand's
scheme going forward (if several clear it, the cheapest). All four fail → *(pre-authorized
2026-07-21, so the autonomous run never blocks here)*: record option (a) as the evidence-backed
finding, PARK this strand's downstream feed-forward tasks (DSEL-2/2b and the selection studies that
need the substrate), CONTINUE the other strands, and batch the finding for end-of-run user review.
Option (a) is
evidence-backed and goes to the user as a finding, never taken as a default by any task.
**No weight decay or dropout in this task** (MASTER Decision 21: regularisation is tested by its own
discriminating check, never mixed silently into a control repair).
**Non-goals:** does not choose between (a)/(b)/(c) itself — only ruling (i) above licenses running
this task at all; it answers the halt, it does not resolve it by itself. Does not run the readout
comparison (DSEL-2's, once a target scheme passes here).
*Orchestration:* parallel: no · deps: none (it answers the halt directly) · tier: sonnet high ·
scale: dynamic · shape: execution · verify: `--selftest` passes; one JSON per (arm, seed) under
`automl_package/examples/capacity_ladder_results/DSEL2c/` carrying `targets`, `schedule`, `test_acc`,
`val_acc`, both gate dicts, `hit_cap: false`, and `n_train`/`n_val`/`n_test`; the two cited (FULL,
ALL_RUNGS) cells resolved by path, not re-run; a `frozen.json` recording which arms cleared the bar.

### DSEL-2b — does the ladder cost the middle depths, the way ProbReg's k-ladder does?

**Files (write set):** `automl_package/examples/depth_dsel2b.py` ·
`automl_package/examples/capacity_ladder_results/DSEL2b/` ·
`docs/plans/capacity_programme/shared/dsel2b_middle_depth_cost.md`
**Spec:** The gradient-share question depth has never asked. Under DSEL-1b's default all-rungs
training scheme (b, sum/mean the loss over every rung every step), the early blocks receive a
gradient contribution from every depth's loss term while the deepest block receives one only from the
deepest — the rungs are not equally trained. The sibling strand has already measured a corresponding
cost: `docs/plans/capacity_programme/probreg.md:95-101` (§2) reports that its k-dropout
coherence check, run on 3 toy problems × 3 repeats, found the largest k never fails in 9/9 cases but
**some middle k fails in 8 of the 9** — "a small, real, and currently unexplained cost concentrated in
the middle of the range." Explaining *why* is tracked separately as
`docs/plans/capacity_programme/probreg.md` task P5
(`docs/plans/capacity_programme/probreg.md:567-576`) — **this task does not
diagnose the mechanism, it only measures whether the analogous effect exists for depth.**
Read DSEL-2's per-(seed, depth) held-out JSONs across the whole ladder and test, per depth, whether
its held-out quality underperforms the trend set by its neighbours by more than the same noise band
§1b already uses for selection (twice a bootstrap standard error) — i.e. reuse the tolerance
convention rather than inventing a second one. Report a per-depth table and an explicit yes/no verdict
per (toy, seed): do middle depths underperform their neighbours beyond noise? **If they do, that is a
property of the ladder that any per-input depth selector inherits, and DSEL-12 must carry it.**
**Doctrine:** this measurement is scheme-specific (see DSEL-1b's gradient-budget note) — it speaks to
scheme (b) only; if scheme (a) is ever run as the labelled comparison arm, this measurement does not
transfer to it without being re-run.
**Non-goals:** no mechanism diagnosis — establishing whether the effect exists is the whole task; a
depth-analogue of `probreg.md` P5 is future work, not created here. No training-schedule change to fix
it — that is a design decision for the user, not this task's to make.
*Orchestration:* parallel: yes (disjoint write set from DSEL-3/DSEL-4) · deps: DSEL-2 ·
tier: opus xhigh (discovery-shaped) · scale: static · shape: research ·
verify: `ls automl_package/examples/capacity_ladder_results/DSEL2b/*.json` lists one file per (toy,
seed) with a per-depth `deviation_from_neighbour_trend_in_se_units` field and a `beyond_noise` boolean
per depth; `test -f docs/plans/capacity_programme/shared/dsel2b_middle_depth_cost.md` exits 0 AND that
note states an explicit yes/no verdict citing the JSONs by path, plus `probreg.md` §2 and P5 by
citation (not restated from memory); each JSON's source DSEL-2 artifacts must already carry
`trajectory_verified=True`/`hit_cap=False` (Decision 9) and, for any low-both-train-and-val cell, the
Decision-16 escalation-ladder record — this task's own verify FAILS if it draws a conclusion from a
DSEL-2 cell missing either.

### DSEL-3 — fix DD1 and DD2

**DONE — found already landed (commit `84ad94d`) during the 2026-07-21 repair audit; no dispatch
needed.** Both fixes and their regression tests (`TestDD1IndependentWeightsUniformPrior`,
`TestDD2PairedDepthUncertainty` — cited under DD1/DD2 in §3) were on disk before this task was ever
run. **The doctrine clause's prove-it-fails ceremony (revert each fix, re-run, show the test FAIL,
restore) is waived here, not separately satisfied**: the commit message for `84ad94d` records that it
was followed at the time the fix was authored — "Every test here was confirmed to fail against its
unfixed code before being kept, and each file's checksum was confirmed unchanged after restoring" —
and re-deriving that ceremony now would mean reverting an already-shipped, already-tested fix for no
purpose. Spec retained below, unedited, as the record of what DD1/DD2 required; read as history, not
a live task.

**Files (write set):** `automl_package/models/independent_weights_flexible_neural_network.py` ·
`automl_package/models/flexible_neural_network.py` · `tests/test_phase2_flexible_nn.py`
**Spec:** (i) **DD1** — replace the `linspace(3.0, 1.0, ...)` prefer-shallow prior with the uniform
prior its sibling already uses, at both sites (`:306` ELBO branch, `:312` cost-aware branch), citing
the sibling's removal comment. (ii) **DD2** — give `predict_uncertainty` the same depth selection the
prediction used, or raise explicitly rather than silently pairing readings from different depths.
**Doctrine:** a regression test is not evidence until shown to FAIL on the unfixed code. Assert on
the quantity the fix changes — the selected depth distribution for DD1, the pairing for DD2 — never
on a coarse downstream view. That is exactly how an earlier fix's tests came out blind.
**Non-goals:** no other change to either class; the missing router and forward primitive on the
independent-weights class (DD4) belong to `flexnn-package.md`.
*Orchestration:* parallel: yes (disjoint from DSEL-1/DSEL-2's write sets) · deps: none ·
tier: sonnet high · scale: static · shape: execution ·
verify: `AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m pytest tests/test_phase2_flexible_nn.py -q`
green, THEN revert each fix in turn, re-run, show the corresponding test FAILING, restore, and show
both file checksums unchanged.

### DSEL-4 — adopt the consolidated API and the single router

**Ordering repair, 2026-07-21.** The write set below originally read "this strand's drivers only (the
`depth_dsel*.py` family — DSEL-2, DSEL-2b, DSEL-6 through DSEL-10's scripts...)" and the verify clause
grepped that same glob. As written, DSEL-4 runs at its scheduled position in §4's order line — before
DSEL-6 through DSEL-10, the five tasks that create `depth_dsel6.py`...`depth_dsel10.py` — where only
`depth_dsel2.py` (and `depth_dsel2b.py`, once DSEL-2b lands — DSEL-2c EXTENDS `depth_dsel2.py`
rather than creating a driver of its own, per its Files line) exist on disk.
The migration and both greps would pass vacuously against files that don't yet exist, DSEL-4 would be
marked done without ever inspecting four of the five scripts its write set claimed to own, and nothing
downstream re-ran the check. Worse, DSEL-7 (`deps: DSEL-4`, as originally written) would then depend
on a task that claims to have already migrated `depth_dsel7.py` — a file DSEL-7 itself creates.
Rescoped as follows; DSEL-7's own `deps:` line is corrected to match, below.

**Files (write set):** `automl_package/examples/depth_dsel2.py` **only** — the one depth-strand driver
that exists at this task's scheduled position.
**Spec:** Once `flexnn-package.md` lands `CapacitySelection` (FP-3, `automl_package/enums.py`) and
the single reconciled router (FP-5, DD3), migrate `depth_dsel2.py` onto the enum: pass the enum member
at `FlexibleHiddenLayersNN` construction, never a string; `predict` loses `inference_mode` entirely per
FP-3.b (clean break, no shim) — a call site in `depth_dsel2.py` still passing it must raise
`TypeError`, not be silently swallowed. `FlexibleHiddenLayersNN`'s `"hard"` execution shortcut (FP-3.c)
is untouched — it is an execution mode, not a selection source, and stays a separate boolean argument,
orthogonal to `CapacitySelection`. **The certified depth-selection numbers must be reproduced through
the package router** before any new selection result is read — that reproduction is what retires the
two-implementations problem rather than merely renaming it. (This part of the spec is unchanged from
the original task; only the write set and the binding convention below are new.)
**Binding convention (new, 2026-07-21) — every later driver is BORN on the package API.** DSEL-6
through DSEL-10 must construct `FlexibleHiddenLayersNN` with the `CapacitySelection` enum from the
moment each script is first written — no local router class, no `inference_mode` string, ever, not
even transiently. There is no migration step for them because there is nothing to migrate: this
convention is how they are built. Each of DSEL-6 through DSEL-10's own verify clauses inherits the two
greps below (`inference_mode`, `class.*Router`) scoped to its own new script; the last consumer before
the report, DSEL-10, additionally runs both greps across the whole `depth_dsel*.py` family as the
strand-wide check (moved there from this task — see DSEL-10's verify clause).
**Doctrine:** DD3 means the certified result and the shipping code are different implementations.
Reproduction against the certified numbers is a positive control (Decision 14), not a formality. The
enum and its cross-family contract are FP-3's design to own; this task's job is to be a correct,
complete consumer — not a second implementation. Do not add a depth-local enum, even temporarily.
**Non-goals:** this task does not design the API or edit package code — it consumes both. No change
to `CapacitySelection` itself — file a finding against FP-3 instead of patching it from here. No
migration of DSEL-6 through DSEL-10's scripts — they do not exist yet (see binding convention above).
*Orchestration:* parallel: no · deps: `flexnn-package.md` FP-3, `flexnn-package.md` FP-5; DSEL-3 ·
tier: sonnet high · scale: static · shape: execution ·
verify: `grep -n "inference_mode" automl_package/examples/depth_dsel2.py` returns nothing (call site
migrated); run `automl_package/examples/depth_dsel2.py` (migrated onto the package router) for both
seeds, and for each seed a check script asserting
`abs(reproduced_mean_executed_width - certified_mean_executed_width) / certified_mean_executed_width <= 0.02`
(2% relative, matching `flexnn-package.md` FP-4's tolerance convention) exits 0, where
`certified_mean_executed_width` is read from
`automl_package/examples/capacity_ladder_results/D_TOY_PROBES/depth_selection_deploy_seed{0,1}.json`'s
`deploy.mean_executed_width` field (certified values: seed0 = 8.0, seed1 = 7.99 —
`docs/depth_capacity/verdict_per_input_depth.md:313`); `grep -n "class.*Router\|def fit_router\|def _train_router"
automl_package/examples/depth_dsel2.py` returns nothing (fail if a local router class remains in this
script — `depth_selection_toy.py`'s certified router is FP-5's reconciliation target, not this task's,
and is deliberately out of this grep's scope). **The strand-wide sweep across the whole
`depth_dsel*.py` family (both greps) is DSEL-10's verify, not this task's** — moved there 2026-07-21,
see DSEL-10.

### DSEL-5 — give the independent-weights class the per-depth primitive (tracked dependency, NOT a dispatchable task)

**Files (write set):** none in this strand — **tracked here, owned by `flexnn-package.md` FP-6**
**Spec:** Recorded as a dependency, not a task: `IndependentWeightsFlexibleNN` cannot participate in
any of §1b's three choices until it has the per-depth forward primitive. If the feed-forward study
(DSEL-2) uses that class, this blocks DSEL-6/7 for the feed-forward mechanism. **This entry is not a
step in the order line below** — it closes when `flexnn-package.md` FP-6 closes, not by a run of its
own.
*Orchestration:* not dispatchable here · deps: n/a · verify: n/a — closed when
`flexnn-package.md` FP-6's own verify line passes.

### DSEL-6 — build D-SHARED

**Files (write set):** `automl_package/models/flexible_neural_network.py` ·
`tests/test_phase2_flexible_nn.py`
**Spec:** Wire the cheap global read using the SHARED cheapest-within-tolerance-at-twice-a-bootstrap-
standard-error primitive owned by `flexnn-package.md` FP-9 ("the shared selection primitives") — do
**NOT** re-implement the selector here. This task's own scope is the depth-specific plumbing only:
score every depth on the held-out selection set, call FP-9's primitive to reduce that curve to ONE
depth for the dataset, and store it so `predict` uses it with no caller flag. Two sibling plans
(`width.md`, `probreg.md`) consume the same FP-9 primitive for their own capacity axes — this task
must not diverge from it or fork a depth-local copy.
**Doctrine:** the tolerance rule (§1b) is imported unchanged, through FP-9, so depth's numbers stay
comparable with the other strands'. Do not re-derive it.
**Non-goals:** no per-input logic; no change to the router; no reimplementation of
cheapest-within-tolerance — that is FP-9's, consumed here.
*Orchestration:* parallel: no (same file as DSEL-3) · deps: `flexnn-package.md` FP-9,
`flexnn-package.md` FP-3 · tier: sonnet high · scale: static ·
shape: execution · verify: `AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m pytest
tests/test_phase2_flexible_nn.py -q` green; a test with a held-out curve flat beyond depth *d* asserts
the selector returns *d*, not the argmax; a second asserts stability under a reshuffle of the
selection set; `grep -n "def.*cheapest_within_tolerance"
automl_package/models/flexible_neural_network.py` returns nothing (fail if a local copy of the
selector exists instead of a call into FP-9's primitive).

### DSEL-7 — build D-SWEEP as a real reference

**Files (write set):** `automl_package/examples/depth_dsel7.py` ·
`automl_package/examples/capacity_ladder_results/DSEL7/`
**Spec:** Train a **separate network per depth** over the frozen ladder
(`automl_package/examples/capacity_ladder_results/DSEL1b/frozen.json`) and score each held out. This
does not exist: today's "best fixed depth" reads the *same* network at different exits, which shares
all its weights with the thing it is supposed to be an independent reference for.
**Doctrine:** the whole efficiency claim is that the cheap read reaches this reference's answer at a
fraction of the cost. A reference that shares weights with the method under test cannot serve as one.
**Non-goals:** do not reuse the multi-exit readout as a shortcut — that is the object this task
replaces.
*Orchestration:* parallel: no · deps: `flexnn-package.md` FP-3, `flexnn-package.md` FP-5 (corrected
2026-07-21 — was `DSEL-4`, which is circular: DSEL-4 is now scoped to migrating `depth_dsel2.py` only
and DSEL-7 is born directly on the package API per DSEL-4's binding convention, so it needs the API
itself, not DSEL-4's completion; see DSEL-4's ordering repair) · tier: sonnet high · scale: dynamic ·
shape: execution · verify: `ls automl_package/examples/capacity_ladder_results/DSEL7/*.json` lists one file per (toy,
seed, depth); for each, `python -c "import json,sys; d=json.load(open(sys.argv[1])); assert
'held_out_score' in d and 'train_cost_seconds' in d" <file>` exits 0; the summed per-depth cost is
written as the sweep's headline `total_training_cost` in
`automl_package/examples/capacity_ladder_results/DSEL7/frozen.json`.

### DSEL-8 — how much data does depth selection need?

**Files (write set):** `automl_package/examples/depth_dsel8.py` ·
`automl_package/examples/capacity_ladder_results/DSEL8/`
**Spec:** The current selection set is 3000 per stratum, chosen for wall-clock reasons, never
measured. Sweep the selection fraction — **the grid IS `{5, 10, 15, 25, 40}%` of the training
portion, binding, not a suggestion; a worker may not substitute its own** — for
all three of §1b's arms, holding everything else fixed. Report quality against fraction and the
saturation point per arm. **The arms are not equally exposed**: D-PERINPUT must learn a *function*
from input to depth and should be hungriest; D-SHARED and D-SWEEP need only rank depths on average.
If D-PERINPUT loses at a small fraction, "per-input depth does not pay" and "the router was starved"
are indistinguishable.
**Non-goals:** no real data (DSEL-11); no architecture changes (DSEL-9).
*Orchestration:* parallel: yes (disjoint from DSEL-9 if driven by separate scripts) ·
deps: DSEL-6, DSEL-7 · tier: sonnet high · scale: dynamic · shape: research ·
verify: `ls automl_package/examples/capacity_ladder_results/DSEL8/*.json` lists one file per (toy,
seed, fraction, arm); `test -f
automl_package/examples/capacity_ladder_results/DSEL8/saturation_plot.png` exits 0;
`python -c "import json; d=json.load(open('automl_package/examples/capacity_ladder_results/DSEL8/frozen.json')); assert 'selection_fraction' in d and 'perinput_data_limited' in d and 'justification' in d"`
exits 0, where `justification` names the specific curve point the default was read from.

### DSEL-9 — does the router's architecture matter for depth?

**Files (write set):** `automl_package/examples/depth_dsel9.py` ·
`automl_package/examples/capacity_ladder_results/DSEL9/`
**Spec:** No sensitivity study exists for depth. Vary router width/depth (at least half/double/4×
hidden, 1 vs 3 layers) and epochs. Establish whether depth's routing conclusions are invariant, and if
not, what the router needs. ~~and the labelling tolerance~~ **struck 2026-07-21: MASTER Decision 18
rules the labelling-tolerance sensitivity sweep NOT scheduled and NOT to be run pre-emptively — "do
not run pre-emptively; run it if a reviewer leans on the constant." This task's spec listed it before
that ruling landed and was never updated to match (cross-strand repair — the same stale sweep was
found in `width.md`'s WSEL-7 and `probreg.md`'s PC, both corrected the same way). If a reviewer leans
on the flat `0.25` tolerance, run the sweep as a separate, later task, not folded into this one.**
**Doctrine:** the router stays FROZEN and untuned inside the battery so the D-SHARED/D-PERINPUT
contrast measures selection rather than search effort. **This task does not unfreeze it** — any
change lands as a new default, reported to the root and applied via `flexnn-package.md` FP-5 (the
file that owns `distilled_router.py`; this strand does not write it — cross-strand repair, 2026-07-21,
see §3.5's DSEL-9 branch above), *before* **DSEL-10** runs (re-pointed 2026-07-21 from DSEL-11, which
the 2026-07-20 parking ruling removed from the dispatch order), never per-dataset.
**Non-goals:** no per-dataset tuning, ever. No change to the labelling rule's meaning.
*Orchestration:* parallel: yes · deps: DSEL-6, DSEL-7 · tier: sonnet high · scale: dynamic ·
shape: research · verify: `test -f
automl_package/examples/capacity_ladder_results/DSEL9/sensitivity_table.csv` exits 0;
`python -c "import json; d=json.load(open('automl_package/examples/capacity_ladder_results/DSEL9/frozen.json')); assert isinstance(d['invariant'], bool)"`
exits 0.

### DSEL-10 — does the cheap read pick what the sweep picks?

**Files (write set):** `automl_package/examples/depth_dsel10.py` ·
`automl_package/examples/capacity_ladder_results/DSEL10/`
**Spec:** Both halves, for the feed-forward mechanism ONLY (recurrent is parked, §1a). Read as
reference: **(a) quality at matched depth** — does the jointly trained network read at depth *d*
match a network dedicated to depth *d*; **(b) agreement** — does D-SHARED choose the depth D-SWEEP
chooses. (b) has never been asked for depth.
**Doctrine:** Decision 14 — D-SWEEP reproduces DSEL-7's control before any D-SHARED number is read.
§3.6's anchor warning binds: the quality half anchors on the independently trained per-depth models.
**Non-goals:** no real data; no baselines.
*Orchestration:* parallel: no · deps: DSEL-8, DSEL-9 · tier: sonnet high · scale: static ·
shape: execution · verify: `ls automl_package/examples/capacity_ladder_results/DSEL10/*.json` lists
one file per (toy, seed) — mechanism is fixed to feed-forward (§1a), not a swept dimension; each file
contains `d_shared_chosen`, `d_sweep_chosen`, the per-depth held-out scores, and each arm's end-to-end
cost including selection; `python -c "import json; d=json.load(open(p)); print(d['d_shared_chosen']==d['d_sweep_chosen'])"`
reports agreement per cell (a `False` is a finding, not a bug); each cell's artifact must carry
`trajectory_verified=True`/`hit_cap=False` (Decision 9) and, for any arm reading low on both train and
val, the Decision-16 escalation-ladder record — absence of either blocks that cell's reading.
**Strand-wide sweep (moved here from DSEL-4, 2026-07-21 ordering repair — DSEL-10 is the last consumer
of this strand's drivers before the report):** `grep -rn "inference_mode" automl_package/examples/depth_dsel*.py`
returns nothing AND `grep -rln "class.*Router\|def fit_router\|def _train_router" automl_package/examples/depth_dsel*.py`
returns no files, confirming every driver DSEL-2 through DSEL-10 built was born on (or, for DSEL-2,
migrated onto) the package API, with no local router surviving anywhere in the family.

### DSEL-11 — real data + baselines — ⏸ PARKED — real data deferred (user ruling 2026-07-20); spec retained for a possible later pass

**⏸ PARKED, not deleted.** **USER RULING 2026-07-20: width and depth stay TOYS-ONLY.** `MASTER.md`
Decision 3's real-data exemption is **not** extended to this strand — it stays scoped to ProbReg only
(`docs/plans/capacity_programme/MASTER.md:89-92`). This task does not run, is not in the dispatch
order (§4), and **nothing else in this strand depends on it**. Its spec below stays on disk, unedited
in substance, so a later user ruling can unpark it without a rewrite. **Do not delete this task.**

**Files (write set):** `automl_package/examples/depth_dsel11.py` · its results dir
(`automl_package/examples/capacity_ladder_results/DSEL11/`) · a benchmark spec under
`docs/depth_benchmark/`
**Spec:** The three models of §1b, on the feed-forward mechanism ONLY (recurrent is parked, §1a —
this task does not revive it for a real-data run any more than the toy studies do), against **one
tree model (LightGBM), a plain single-output NN (the key control), and linear regression (the
floor)**, on real datasets frozen in the spec. Write the spec first; it is a deliverable of this task.
**Binding: the driver READS §3.6's constants from their artifacts at startup and FAILS LOUDLY if any
is missing.** Each results JSON records the constants it ran under. **Every arm's number includes its
selection cost.**
**Doctrine:** depth has never been run on real data or against any external comparator. State plainly
in the results whether the substrate that carries the depth signal on toys has any analogue in the
real datasets — if it does not, that is the finding.
*Orchestration:* parallel: no · deps: DSEL-10 · tier: sonnet high · scale: dynamic · shape: execution ·
verify: rename or move one §3.6 constant artifact aside and run
`automl_package/examples/depth_dsel11.py`; confirm (`echo $?`) it exits non-zero with an error naming
the missing constant; restore the artifact and re-run, confirm exit 0. Then: one per-dataset CSV +
one per-model JSON under `automl_package/examples/capacity_ladder_results/DSEL11/`, every headline
number traceable via a `source_artifact` field to a seed-level JSON that itself lists the constants it
ran under and carries Decision 9's trajectory fields (`trajectory_verified`, `hit_cap=False`) and,
where an arm reads low on both train and val, Decision 16's escalation-ladder record.

### DSEL-12 — report

**Files (write set):** `docs/reports/depth_selection/`
**Spec:** Author via the `research-report` skill, as the user, no AI/tool provenance (MASTER
Decision 10). **Lead with the feed-forward result** — it is the primary claim. The studies get their
own sections ahead of the comparison because they license its settings: how much data selection
needs, whether the router's architecture matters, and whether the cheap read reaches the sweep's
answer (both halves stated separately). **This report carries NO baseline / real-data section** —
DSEL-11 is PARKED (user ruling 2026-07-20, width/depth stay toys-only); this is toys-only content,
not a placeholder pending a ruling.

**Honesty clauses, binding:** report D-SWEEP's full cost next to its accuracy; state every constant
the battery ran under and which study set it; carry the negative results in the body, not an
appendix — including **the closed one-dimensional lane and the theorem behind it**, DSEL-1's
supervision finding whichever way it went, and **DSEL-2b's middle-depth verdict whichever way it
went** (§ DSEL-2b). State plainly that D-PERINPUT's tolerance rule differs from D-SHARED's/D-SWEEP's
(§1b) and that their choices are not directly comparable on tolerance grounds — say explicitly why
(no standard error is estimable from one row) and that comparison across arms lives on held-out error
and cost, not on tolerance. **State plainly what `G-DEPTH = PASS` did and did not certify**, so the
report does not propagate the misreading this strand was created to correct.
*Orchestration:* parallel: no · deps: DSEL-10 · tier: opus xhigh (main loop, adversarial cold-read) ·
scale: static · shape: execution ·
verify: the `research-report` skill's cold-read gate passes (its own pass/fail contract); for each
constant row in §3.6, `grep -c "<constant name>" docs/reports/depth_selection/*.md` returns ≥1 AND the
surrounding text names the owning study (DSEL-1/DSEL-1b/DSEL-8/DSEL-9/DSEL-10) — check each row by
hand against the built report; fail if any constant appears without its study named; `grep -ic
"lightgbm\|linear regression\|real.data\|benchmark" docs/reports/depth_selection/*.md` returns nothing
(fail if a real-data/baseline section leaked in while DSEL-11 is PARKED).

---

## 5. Non-goals for this strand

No reopening of `G-DEPTH = PASS`. No revival of smooth one-dimensional depth toys — closed by
theorem (§2), four candidates, not to be relitigated. No new toy construction without a reviewed
written spec. No package-structure or API work (owned by `flexnn-package.md`). No joint width+depth
work (`width-depth.md`). No revival of in-training depth selection as a primary (MASTER Decision 13).

---

## 6. History — the feed-forward depth pilot (absorbed from `flexnn-core.md`, 2026-07-20)

Moved here verbatim by **DSEL-0** so no depth content survives in two places (`flexnn-core.md`
now carries a pointer only). **This is HISTORY: read it, never dispatch from it.** It records
the original pilot (F5), the run that was ruled INVALID on four independent grounds (F5b), and
the protocol-repair attempt whose positive control failed on both seeds (F5c).

**Superseded by this strand as follows.** The halt F5c escalated is **CLOSED** by the user
ruling recorded in DSEL-1 and written up at
`docs/plans/capacity_programme/shared/dsel1_nested_diagnosis.md`: the failed arm had **no
nested structure** — a single loss at full depth — so the overfitting needed no further
diagnosis and the escalation ladder was aimed at the wrong problem. DSEL-1b builds the
replacement all-rungs training scheme; DSEL-2 carries the primary claim. The staged F5c-a/b/c/d
sequence below is **retired, not pending** — do not resume it.

<!-- citecheck-ignore: historical record, superseded; retained for provenance -->

### Task F5: feedforward-depth pilot — spec, ADJUDICATOR GO, build, run (2 seeds)

**F5a — author the spec** (`docs/depth_capacity/ff_depth_toy_spec.md`), then **⛔ ADJUDICATOR GO
gate (user away — ratified 2026-07-18: no user questions mid-run).**
[[feedback_toy_design_needs_reviewed_spec]] is satisfied unattended as follows: the written spec
+ adversarial review are still MANDATORY before any build; the GO call is made by an adjudicator
(Opus/xhigh) instead of the user. Verdict SOUND, or SOUND-WITH-FIXES whose fixes are mechanical →
fold fixes, log the verdict, BUILD. Verdict UNSOUND, or any fix that is a PI design decision →
**PARK F5b** (do not improvise, do not redesign), log it under "Batched user questions", and
carry on with every other unblocked task. Spec + review verdict are delivered for post-hoc user
review either way.

**Files (F5a):** Create: `docs/depth_capacity/ff_depth_toy_spec.md`
**Orchestration (F5a):** parallel: yes · deps: none · tier: opus/high (design) + adversarial
review pass (adjudicator) · scale: static · shape: design · verify: spec contains generative
math, the 2×2 grid below, pre-registered bars with numeric thresholds, confound ledger
(param-count-grows-with-depth for untied arms → param-matched reads mandatory), review verdict §

Design core (settled in main session 2026-07-18, to be made numeric in the spec): task = A5
word composition, L=10, involution generators (reuse `depth_composition_toy.py` machinery — the
certified substrate; input = flattened one-hot word, 50-dim). **2×2 architecture grid** isolating
WHY depth wins there:

| | flat input (all letters at layer 0) | per-step input (letter t at layer t) |
|---|---|---|
| **untied weights** | plain deep MLP — *the user's original claim* | untied unrolled stack |
| **tied weights** | tied stack on flat input | = `RecurrentComposer` (already certified ≥0.90) |

plus the certified wide-shallow control. Depth ladder for stack arms: d ∈ {4, 7, 10}, width 64.
Bars (freeze numerically in spec): FF-CLAIM passes iff the untied-flat arm at some d ≤ 10 reaches
held-out ≥ 0.90 while param-matched wide-shallow ≤ 0.60 (the graded-pilot fit/stall thresholds);
attribution read = which cells generalize, ordered; convergence-gated; 2 seeds. Either outcome is
reportable: PASS → flexible depth demonstrated on a straightforward feedforward net where depth
is provably non-substitutable; FAIL with tied cells passing → weight-tying is the load-bearing
ingredient (goes in F7 + informs the transformer roadmap: weight-tied = the Universal
Transformer shape, arXiv:1807.03819).

**F5b — build + run: RAN 2026-07-18 → ⛔ INVALID (2026-07-20). Superseded by F5c.**
The build landed (`NetKind.TIED_FLAT`, `UNTIED_PERSTEP` + builders + driver flags in
`automl_package/examples/depth_composition_toy.py`) and 28 runs produced
`.../D_TOY_PROBES/ff_depth_pilot_a5_seed{0,1}.json`. **No bar may be read from them**, on three
independent grounds, each verified on disk 2026-07-20:
1. **Positive control FAILED (MASTER Decision 14).** Cell 4 (`RecurrentComposer`, the certified
   arm, made a MANDATORY confirm run by the F5a reviewer precisely to validate the protocol) is
   trustworthy on **1 of 2 seeds**: seed 1 = 0.9257 clean; seed 0 climbed to 0.830 by epoch 3000
   then **collapsed to 0.097**, `diverged=true`. Spec §6 requires ≥2 trustworthy seeds. The
   protocol did not reproduce a known-good result ⇒ nothing else in the battery is readable.
2. **Protocol parity breach (MASTER Decision 15).** These runs used `depth_composition_toy.py`'s
   `train_clf`, which applies **no gradient clipping**, at A5/L=10 — while
   `automl_package/examples/depth_selection_toy.py` sets `GRAD_CLIP_MAX_NORM = 1.0` commented
   "L=10 needs clipping to stay GD-trainable". Single unswept `LR = 1e-2`, no schedule/warmup,
   across arms spanning 12,476–89,660 params.
3. **Gate/bar metric mismatch (MASTER Decision 17).** `trustworthy` is computed on val CE; the
   bars read val accuracy. Val CE explodes (overconfidence) while val accuracy sits flat.
4. **Decision 9 never satisfied.** Runs self-terminated by patience at ~2,750 epochs against a
   40,000 cap; the required early-stop-OFF confirmation at ≥4× budget was never run.

**Recorded observations (NOT findings — no bar attaches):** untied-flat fails to fit the TRAIN
set at d=7 (train 0.195) and d=10 (train 0.055) ⇒ under-fit per Decision 16, an optimization
signal, not an architecture verdict. Shallow (d=4) and deep (d=10) arms fail *differently*
(instability/overconfidence vs. no learning at all) — one protocol fix may not cover both.

### Task F5c: depth protocol repair + diagnosis (replaces F5b's run; added 2026-07-20, user-approved)

**Staged — each stage GATES the next. No stage may be skipped or reordered.**

**F5c-a — spec first (~30 min), then ⛔ ADJUDICATOR GO.**
Create: `docs/depth_capacity/ff_depth_protocol_repair_spec.md`. Must contain: (i) a line-by-line
**training-loop diff** between `depth_composition_toy.py::train_clf` and `depth_selection_toy.py`'s
loop, every difference justified or removed (Decision 15); (ii) the **escalation ladder** — which
remedies, in what fixed order, with the stopping rule (Decision 16); (iii) **instrumentation
plan** for per-layer gradient norms (the vanishing/exploding hypothesis must be MEASURED, not
asserted — no artifact currently exists for it anywhere in the repo); (iv) the convergence gate
recomputed on **val accuracy** as well as CE (Decision 17); (v) bars **unchanged** from
`ff_depth_toy_spec.md` §6 — FIT 0.90 / STALL 0.60, frozen (Decision 9: no bar moves after a run).
*Orchestration:* parallel: yes · deps: none · tier: opus/high design + adjudicator review ·
shape: design · verify: spec contains all five items + a review verdict §.

**F5c-b — positive control ALONE (Decision 14 gate).** `RecurrentComposer`, A5, L=10, 2 seeds,
repaired protocol, plain single-readout (same `train_clf` protocol as the grid arms).
**PASS BAR: held-out ≥ 0.90 AND `trustworthy=true` on BOTH seeds.** FAIL ⇒ **HALT**, escalate the
ladder, re-run this stage only. **No compute is spent on any other arm until this passes.**
*Orchestration:* deps: F5c-a GO · tier: sonnet · verify: 2 JSONs, both ≥0.90 and trustworthy.

**RESULT 2026-07-20 at Rung 0 (`lr=3e-3`, `clip_max_norm=1.0`, dual gate) — ⛔ FAIL, both seeds.**
Artifacts: `automl_package/examples/capacity_ladder_results/D_TOY_PROBES/f5c_poscontrol_a5_seed{0,1}.json`.

| seed | train_acc | val_acc (bar 0.90) | trustworthy_acc | stop epoch (cap 40k) | clip engagement |
|---|---|---|---|---|---|
| 0 | 0.9697 | **0.4324** | true | 10,500 | 98 % |
| 1 | 1.0000 | **0.7442** | true | 7,500 | 59 % |

Both runs are `INTERMEDIATE`, both converged cleanly (not `hit_cap`, not `still_improving`, not
`diverged` on accuracy), and both show the SAME shape: `val_acc` rises then goes flat for the final
third while **held-out CE diverges monotonically** after its minimum (seed 1: 1.374 @ 3,750 →
2.325 @ 7,500; seed 0: 1.55 @ 1,250 → 4.87 @ 10,250; `ce_gate.diverged=true` both). Note this is
also the vindication of the dual gate: best-CE weight restore would have selected ~epoch 3,750 for
seed 1, i.e. a *lower* accuracy than the run's own best — the exact selection defect that invalidated
the previous attempt.

**The failure mode is memorization-without-generalization, NOT under-fitting** — and that is a
problem the ladder cannot address. Every rung (L1 LR, L2 warmup, L3 init) is an *optimization*
remedy, but at `train_acc` 0.97/1.00 the optimizer has already fit the training set completely; there
is no optimization failure left to fix. Concretely, both seeds already SATISFY Decision 16's
exoneration condition (`train_acc ≥ 0.90`) while failing the §5 positive-control bar. Two seeds
differing by 31 pp on held-out accuracy (0.432 vs 0.744) from initialization alone is a further sign
this protocol is fragile on this substrate rather than mis-tuned.

**⛔ ESCALATED TO USER — do NOT resolve this without a ruling.** §2.5 step 2 mechanically prescribes
L1; M7 makes the fail branch a PI-level call. The evidence says L1 is aimed at the wrong problem and
the informative experiment is M6's discriminator (does the certified anytime configuration still
reproduce its ≥ 0.90 per-stratum numbers? — reproduced ⇒ single-exit supervision genuinely cannot do
this task, a substantive finding; not reproduced ⇒ environmental/regression and NOTHING may be
claimed from today's runs), which the spec currently gates behind L3 exhaustion. Running M6 before
L1 is a spec deviation and needs the ruling. F5c-c/F5c-d remain HALTED either way.

**Correction to the prior session's hand-off note:** it recorded "FAIL → run the M6 discriminator".
That contradicts the spec — §2.5 step 2 escalates to L1; M6 is gated at L3 exhaustion (§M6/M7). The
spec is the plan of record.

**F5c-c — instrumented diagnosis of the untied arms.** Only after F5c-b passes. Log per-layer
gradient norms across training for untied-flat d ∈ {4,7,10}; land them as an artifact. Answers
the actual question: does the deep untied stack starve (vanishing) or destabilize (exploding),
and are these two different failure modes? *Orchestration:* deps: F5c-b PASS · tier: sonnet ·
verify: a grad-norm artifact JSON/CSV exists per depth; the mechanism claim cites it (new Rule:
no claim without an artifact).

**F5c-d — the 2×2 grid re-run + verdict.** Only after F5c-b/c. Full matrix per `ff_depth_toy_spec.md`
§9 under the repaired protocol, incl. the Decision 9 early-stop-OFF confirmation at ≥4× budget for
any load-bearing cell. Append the results § to `docs/depth_capacity/verdict_per_input_depth.md`
(the record that F5b owed and never wrote — it records what ran and what is quarantined, and
renders a verdict ONLY where a bar is validly evaluable).
*Orchestration:* deps: F5c-c · tier: sonnet · scale: static · shape: execution · verify: JSONs
exist AND every reported cell is `trustworthy=true` on ≥2 seeds AND bars evaluated in the verdict §;
runs detached per environment rules.

**Non-goals:** no L > 10 (MOD-1 GD wall), no new groups, no selection/router in the pilot
(substrate question only), no curriculum tricks (future work if FF fails).

