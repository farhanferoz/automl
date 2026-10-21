# ProbReg — stream plan

The one plan for the ProbReg stream: the end-to-end classification-bottleneck regressor
(`automl_package/models/probabilistic_regression.py`). It holds the owner's rulings and objectives,
the model definitions, the task order across sub-plans, the stream's decisions and its open
questions. Work items live in the sub-plans. Rules that bind every stream: `docs/RULES.md`.

## Sub-plans

| Sub-plan | What it covers | Status |
|---|---|---|
| `docs/probreg/fixed_k/plan.md` | the base model, k set by the user: the structure study (PS-1 to PS-4), the head-structure comparison (P11), the accepted failing test (P2) | structure study ran 3 of 4 stages; PS-4 certification not run, blocked on the PS-3 head-layout halt ruling |
| `docs/probreg/global_k/plan.md` | one k per dataset, chosen on held-out data: the cheap nested read-out (M1) against the per-k sweep (M3); P3, P5, P8 | open; P8's results are void |
| `docs/probreg/per_input_k/plan.md` | a distilled router picks k per input (M2): how much selection data (PB), the router's architecture (PC) | open |
| `docs/probreg/shared/plan.md` | work serving all three: code organisation (§0.5, P9), execution contract and frozen constants (§3.5, §3.6), P0, P0b, P0-b1, PA, P1, the training-schedule migration (P7), real data (P4), the report (P6), P10 | see each task's header |
| `docs/probreg/shared/toy_suite.md` | the canonical toy suite and its assignment; PT | — |
| `docs/probreg/shared/state.md` | what is established (§2), which records to trust (§2.5), known defects (§3) | — |
| `docs/probreg/review/` | the independent methodology review: terms of reference and inputs (kept local-only) | not yet run |

Also in this stream: `docs/probreg/shared/benchmark_spec.md` and `docs/probreg/shared/toy_design_spec.md`
(the benchmark spec's numeric pass bars are superseded by the owner's no-thresholds ruling below),
`docs/probreg/shared/metrics-accounting.md`, `docs/probreg/shared/reports/`,
`docs/probreg/fixed_k/reports/report_a/` (report (a)), `docs/probreg/history/` (finished work: the
April identifiability programme, the June variational-EM k-selection, the first-generation
capacity-ladder plans and report, the July diagnoses, the talk), and `docs/probreg/archive/`.

## Where the former `probreg.md` sections went <!-- citecheck-ignore -->

Assembled 2026-10-21 from `docs/plans/capacity_programme/probreg.md`, <!-- citecheck-ignore -->
`docs/plans/probreg_review/objectives.md` and the ProbReg entries of the former `MASTER.md` decision <!-- citecheck-ignore -->
register. Text was moved, not rewritten; each moved block starts with a `<!-- from … -->` marker
giving its original file and lines.

| Former section | Now |
|---|---|
| header, §1 model definitions, §4 task order, §5 non-goals | this file |
| §0.5 code organisation (with P9), §3.5, §3.6, P0, P0b, P0-b1, PA, P7, P1, P4, P6, P10 | `docs/probreg/shared/plan.md` |
| the canonical toy suite, PT | `docs/probreg/shared/toy_suite.md` |
| §2, §2.5, §3 | `docs/probreg/shared/state.md` |
| P2, P11 | `docs/probreg/fixed_k/plan.md` (after the structure study) |
| P3, P5, P8 | `docs/probreg/global_k/plan.md` |
| PB, PC | `docs/probreg/per_input_k/plan.md` |
| MASTER Decisions 4, 24, 25, 26, 27, 29 | this file, at the end |

Decision numbers are unchanged; `docs/RULES.md` lists which file holds each decision.

## Parked open question — learned σ

σ is fixed for the current phase (owner ruling Q2, 2026-10-20, below), but learned σ remains a design
option for later. Decision 26 (at the end of this file) records fixed σ as a settled
model-definition change; the owner's ruling amends that to "a design option — start with it fixed".
When it is reopened, start from report (a)
(`docs/probreg/fixed_k/reports/report_a/report_a_probreg_toys.md`, run on the old learned-variance
model) and the June–July variance lane: results `V0`–`V3` and `X2` in
`automl_package/examples/capacity_ladder_results/`, and the verdict
`automl_package/examples/capacity_ladder_results/R4_verdict.md` ("variance estimation without
in-sample collapse"). The hazard that fixed σ creates — extra classes absorbing spread that has no
modes behind it — must be guarded in the k-selection work, not assumed away.

<!-- from docs/plans/probreg_review/objectives.md (whole file) -->
## ProbReg — objectives of the work

Status: **owner-answered 2026-10-20.** Assembled from the plan documents; the owner answered the six
open questions on 2026-10-20 — **their rulings are in the next section and override anything below
that conflicts with them.** Every objective names its source so it can be checked.

### Owner's rulings, 2026-10-20 (answers to Q1–Q6 at the end of this file)

| Q | Ruling | What it means for the work |
|---|---|---|
| Q1 | **Both** claims (calibrated uncertainty AND adaptive resolution) — **but one at a time, k first** | The current focus is k: Fixed-k as the substrate, then Global-k, then Per-input-k. Calibration (O1c) stays an objective, worked later. |
| Q2 | **σ is a design option — start with it fixed** | Work proceeds at fixed σ, but fixed σ is NOT settled truth: learned σ remains a later design option. Hazards that fixed σ creates (e.g. extra components absorbing spread) must be guarded, not assumed away. |
| Q3 | **The classification bottleneck must IMPROVE point accuracy**, not merely not hurt it | O1b is raised: the pre-registered bar (H3, "≤ 1.05× the best baseline") is weaker than the objective. The comparison that tests it is against a plain network of matched capacity (the no-classification/bypass arm). |
| Q4 | **Open** — cross-entropy supervision of the classifier, or regression-loss only | To be settled by evidence; neither the 2026-07-21 retirement nor the structure study's finding is final. |
| Q5 | **Toys only for now** | O4 (real data, baselines on real data) is out of scope for the current phase. |
| Q6 | **Open** — is k claimed to be *useful* (prediction), *true* (structure), or both | Every k claim must say which kind it is; the two are tested differently. |
| — | **k is a hyperparameter, never learned with the regression means during fitting** — chosen from held-out data or genuine Bayesian evidence | Training always prefers more classes; any route by which the training objective chooses k (including early stopping or checkpoint choice on a loss averaged over k) violates this. |
| — | **No hard thresholds or pass bars.** Results are judged by comparison — against a pure regression model, across ProbReg architectures, across ways of choosing k — using judgement, unless a difference is obvious | Numeric bars quoted below (e.g. "≥ 8 of 11 cases", "1.05×", "0.02 nats") are context from the old real-data spec, NOT bars for this phase. Comparisons are reported as paired differences with their seed-to-seed spread; each comparison states its possible outcomes and what each would mean before it is run. |

### The model, in one paragraph

ProbReg is a regression model built as a classifier. The target range is divided into k classes. A
network reads the input and outputs a probability for each class; each class has its own small
regression head; the prediction combines the heads, weighted by the class probabilities, into a
predictive distribution (a k-component mixture; its mean and spread follow from the law of total
variance). The classification step is a bottleneck: the model must say *which part of the target
range* an input belongs to before it refines the value. k is therefore a **resolution dial**.

Three variants, differing only in how k is chosen:

| Name | How k is chosen | Plan code name |
|---|---|---|
| **Fixed-k ProbReg** | set by the user | the base model (`docs/probreg/PLAN.md` §1; `docs/probreg/fixed_k/plan.md` is the study of its design) |
| **Global-k ProbReg** | one k for the dataset, chosen from held-out data — cheaply, by reading one nested network at every k; the expensive reference trains one model per k | M1 (cheap read-out) and M3 (sweep reference), `docs/probreg/PLAN.md` §1 |
| **Per-input-k ProbReg** | a small router, distilled from held-out per-k errors, picks k for each input | M2, `docs/probreg/PLAN.md` §1 |

### The goal, as the owner framed it (2026-07-03)

Recorded in `docs/probreg/history/capacity_ladder_2026-07/plans/handover/REVIEW_HANDOVER_2026-07-03.md` "FRAMING": <!-- citecheck-ignore: local-only note (excluded from git) -->
the architecture is **frozen** as a classifier-regularised regressor (classifier bottleneck over
output-range classes + per-class heads + a bypass head; replacing it with a generic mixture density
network was rejected). **The goal is k selected from data, ultimately per input**, "with strict
adherence to probabilistic inference: priors and likelihoods only, no hand-weighted penalty terms".
The same note distinguishes **three meanings of "k-selection works"**: in-fit concentration,
post-hoc measurement, and predictive adequacy — and asked for a claims ledger (`STATE_OF_PLAY.md`)
that was never written.

### Objectives

#### O1 — Fixed-k: the classification bottleneck is a sound regression model
- **O1a. Settle what the model is.** Classifier supervision, how spread is parameterised, the
  likelihood form, identifiability constraints, the head layout — decided by pre-registered
  experiment and certified as ONE configuration. *(Source: `docs/probreg/fixed_k/plan.md` "Goal";
  "Problem 1 before Problem 2", `docs/probreg/fixed_k/plan.md` §1.)*
- **O1b. It IMPROVES point accuracy** (owner, 2026-10-20, Q3) — over a plain network of matched
  capacity. *(The pre-registered real-data bar, `docs/probreg/shared/benchmark_spec.md` §7.4 H3, asks only
  "≤ 1.05× the best baseline"; that is now the floor, not the objective.)*
- **O1c. It gives a better predictive distribution where the data has structure** — multimodal and
  heteroscedastic targets — than standard regressors. *(Sources: H2 and H5 in `benchmark_spec.md` §7.3, §7.6;
  report (a)'s calibration headline, `docs/probreg/fixed_k/reports/report_a/report_a_probreg_toys.md` Summary.)*

#### O2 — Global-k: k can be chosen from data, cheaply and correctly
- **O2a. Same quality.** Reading the nested network at a given k is as good as a model trained only
  for that k. *(Source: `docs/probreg/PLAN.md` §1 "(a) same quality".)*
- **O2b. Same choice.** The cheap read-out picks the k the expensive sweep picks — on ≥ 8 of 11 real
  cases, with a likelihood gap < 0.02 nats where they differ — and the cost ratio is reported.
  *(Source: `docs/probreg/PLAN.md` §1 "(b) same choice"; `benchmark_spec.md` §7.4b, H6.)*
- **O2c. The chosen k is right where the truth is known.** On the reference toy it returns the true
  k; on smooth data it returns the smallest k; it does not raise k merely because the spread widens.
  *(Source: the canonical toy suite tiers 1–2, `docs/probreg/PLAN.md` "THE CANONICAL TOY SUITE".)*
  ⚠️ O2c is a different kind of claim from O2a/O2b — see Q6.

#### O3 — Per-input-k: the model can spend resolution where the input needs it
- **O3a. It recovers a known per-input k\*(x)** — rising, rising-then-falling, and a staircase that stops
  at 3 rather than running to k_max. *(Source: toy suite tier 3, `docs/probreg/PLAN.md`.)*
- **O3b. It never loses to global-k** (ties or beats on ≥ 10 of 11 real cases) **and strictly pays where
  structure varies** (≥ 0.02 nats better on ≥ 4 of 11). *(Source: `benchmark_spec.md` §7.1–7.2, H1a/H1b.)*
- **O3c. It routes smooth data to the no-classification (bypass) rung.** A falsifiable prediction,
  not a gate. *(Source: `benchmark_spec.md` §7.7.)*

#### O4 — Beyond toys — ⏸ OUT OF SCOPE FOR NOW (owner, 2026-10-20, Q5: toys only)
- Real data and baselines are in scope for ProbReg only (Kepler, UCI Superconductivity, UCI Airfoil,
  then the UCI suite; baselines LightGBM, a plain network, linear regression).
  *(Source: `docs/RULES.md` Decision 3 as amended 2026-07-20; `benchmark_spec.md` §2, §5.)*
- The eventual paper headline is photometric redshift estimation. *(Source: `docs/applications/research_plan.md` §6.2,
  §7 "Paper A".)*

### Principles the owner has ruled, which every objective inherits
1. **σ is fixed, never learned** — so spread must be explained by more components, not by one wider
   component (MASTER Decision 26; but see open question Q2).
2. **k is read on the fixed-σ mixture likelihood** — never on squared error (blind to k) and never on a
   learned variance (MASTER Decision 24).
3. **k is chosen after training, never during** — in-training selection is retired (MASTER Decision 29).
4. **Each variant is scored as a complete system, its selection machinery and cost included**
   (`docs/probreg/PLAN.md` §1).
5. **Settle training before selection** (MASTER Decision 37, ruled for width; the structure phase is
   the same principle for ProbReg).
6. **Probabilistic inference only** — priors and likelihoods, no hand-weighted penalty terms
   (2026-07-03 framing above; `K_PENALTY` is retired by MASTER Decision 29).

### The six questions, as put to the owner (answered 2026-10-20 — rulings at the top of this file)

Kept for the reasoning behind each question; the rulings table at the top is authoritative.
- **Q1. What is the primary claim: calibrated uncertainty, or adaptive resolution?** The April paper
  plan (`docs/applications/research_plan.md` §7) claims *"learned per-class variance … dominates other tabular uncertainty
  estimators"*, with in-training dynamic k as a contribution. Since July both halves are retired
  (σ fixed; in-training k retired). The live plan is about k as a resolution dial. Report (a)'s
  headline was calibration, measured under the old learned-variance model.
- **Q2. Is σ fixed, or is it a design choice still under test?** MASTER Decision 26 fixes it in
  training; `docs/probreg/fixed_k/plan.md` §1 then SUSPENDED Decision 26 and made fixed σ one arm among three. The
  structure study picked fixed σ by a simplicity tie-break, not a measured win.
- **Q3. "Classification as a regulariser" — must it HELP point accuracy, or only not hurt it?** The
  pre-registered bar (H3) only asks it not to hurt (≤ 1.05× the best baseline).
- **Q4. Is the classifier trained against its classes (cross-entropy) or only through the regression
  loss?** Cross-entropy was retired on 2026-07-21 because it imposes a fixed percentile carve-up of y
  (MASTER Decision 29); the structure study then found the only designs whose classes stayed in order
  were the cross-entropy ones (`docs/probreg/fixed_k/plan.md` §0, PS-1 evidence note).
- **Q5. Real data before or after the toy results are settled?** Decision 3 puts real data in scope;
  the structure phase paused all of it (`docs/probreg/fixed_k/plan.md` §1).
- **Q6. Is the claim "k is a useful resolution" or "k equals the true number of modes" — or both?**
  AutoCast's rank-ladder programme separates the two (`docs/probreg/review/sources/autocast_rank_ladder.md` §1, local-only): a <!-- citecheck-ignore: local-only review pack -->
  chosen capacity can be right for *prediction* (held-out likelihood, calibration) while being wrong
  about the *true* structure, and in AutoCast every reader lost the "true structure" claim once the
  model's persistent approximation error reached about a quarter of the noise level. O2a/O2b/O3b are
  prediction claims; O2c/O3a are true-structure claims. ProbReg's toys can supply what AutoCast's
  could not — many draws of y at the same x — which would let the two be separated exactly.

<!-- from docs/plans/capacity_programme/probreg.md lines 1-14 -->
# Strand: ProbReg — per-input resolution (k) selection

**Owns the whole ProbReg workstream**: model definitions, the defects, the comparison battery
(toys + real data + baselines), and the report. Read `docs/flexnn/PLAN.md` + this file — that is the whole
context. Nothing about ProbReg is decided anywhere else; if another document disagrees with this
one, **this one wins and the other is a bug to fix**.

**Why this file exists (2026-07-20).** ProbReg content was scattered across 15 files in 5 plan
directories with no owning strand, and the model definitions disagreed in three places at once
(§1). That scatter produced wrong answers to the user twice in one session. Consolidating here is
the fix; the rule that keeps it fixed is the first line of this file.

---


<!-- from docs/plans/capacity_programme/probreg.md lines 233-474 -->
## 1. Model definitions — SETTLED 2026-07-20 (user, live). Supersedes every other statement.

The comparison is between three ways of choosing **k**, the number of classes the model resolves
the target into.

⚠️ **σ IS FIXED IN TRAINING AS WELL AS IN SELECTION — MODEL-DEFINITION CHANGE (user, 2026-07-21;
MASTER Decision 26).** ProbReg fits no variance anywhere. With σ a shared constant, each rung's NLL
reduces to squared error up to a constant, so **training is MSE on the mixture mean**, the per-class
heads predict **means only**, and the within-component term of the law-of-total-variance combination
is the constant. Predictive spread is therefore the **between-component spread plus σ²** — still
meaningful, never fitted.

**Why (the reason is a confound, not tidiness):** a learned per-class variance lets **one component
absorb dispersion by widening itself**, so the model fits spread-out data *without* needing more
components — precisely the question the k dial exists to answer. Fixing σ forces dispersion to be
explained by structure rather than by width.

**Consequences that bind every task in this strand:**
- **All prior k-dropout results are now OLD-OBJECTIVE** as well as old-schedule (Decision 20). Two
  labels travel with historical numbers; neither may be dropped.
- **The suite bar must be RE-BASELINED before this lands** — the two accepted heteroscedastic
  failures test variance behaviour directly, and this changes what a component can express. The
  "366 passed / 2 failed / 1 skipped, no new failures and no newly-passing tests" bar **does not
  carry across this change.** Re-baseline, record the new expected result, and treat any other
  movement as the regression signal.
- **`docs/flexnn/package/plan.md` FP-12 now covers training as well as scoring**, which overlaps **P7**'s
  rewrite of the training objective. Sequence deliberately or merge them; do not dispatch both.

⚠️ **HEAD LAYOUT PINNED — all three models use `RegressionStrategy.SEPARATE_HEADS` (root, 2026-07-21,
at the user's instruction to correct this omission).** The plan previously never named a head layout
anywhere (`grep -in 'regression_strategy\|separate_heads' docs/probreg/PLAN.md` → **zero hits** before this
block), while every capacity-programme driver on disk already pins `SEPARATE_HEADS`
(`automl_package/examples/report_a_benchmark.py:192` and `:213`,
`probreg_kselection_experiments.py:83`, `probreg_elbo_prior_check.py:57`, `capacity_ladder_h1.py:96`,
`multi_seed_sweep.py:64`). The convention was universal and unwritten; this writes it down.

**Why it is load-bearing and not bookkeeping — the nesting guarantee is about COMPONENTS.** The
prefix property this strand's whole read-out machinery rests on is *"the first c rungs are a genuine
c-component model"*. That sentence requires c components to exist. The three layouts differ in
whether they produce any:

| layout | per-class outputs? | prefix-masking gives | status under `NESTED` |
|---|---|---|---|
| `SEPARATE_HEADS` | yes; head `i` sees **only** `p_i` (`regression_heads.py:368-372`, `input_size=1`) | a probability-weighted combination over the surviving k classes; masked classes carry zero weight and cannot leak into any surviving head | ✅ **the sanctioned layout** |
| `SINGLE_HEAD_N_OUTPUTS` | yes; ONE net reads the whole probability vector and emits every class's output, then the SAME weighted combination (`regression_heads.py:461`) | still a genuine k-class weighted combination — **but** each surviving class's output is computed from the zero-padded vector, so truncation leaks into every component | 🟡 **labelled comparison arm only, never a default** |
| `SINGLE_HEAD_FINAL_OUTPUT` | **no** — the head maps probabilities straight to the prediction (`regression_heads.py:498-508`); uncertainty is whatever that single head emits, not a law-of-total-variance combination over components | a valid predictor, but not a c-component model of any kind | 🚫 **BLOCKED under `NESTED`** |

**On the third row, state the claim precisely — an earlier draft of this analysis overreached.** The
masked vector *is* valid and the rung *is* a well-defined model; k remains a real resolution dial on
the classifier side. Nothing is ill-formed and nothing crashes. What fails is narrower and worse: the
justifying guarantee is **vacuous** there rather than false, so the selector, the router and the
arbiter would emit a perfectly plausible curve over k underwritten by a property that configuration
does not have. Silent, not loud.

**The codebase already draws exactly this line, for exactly this reason — one path over.**
`predict_distribution` hard-refuses the same layout: *"predict_distribution is not available for
SINGLE_HEAD_FINAL_OUTPUT; no per-class (mu, sigma) is produced"*
(`automl_package/models/probabilistic_regression.py:792-794`). Two code paths, one underlying fact,
opposite responses — the nested ladder accepts what `predict_distribution` refuses. **That
inconsistency, not the structural argument above, is the primary evidence the gate is missing rather
than deliberately omitted.** Closing it is **`docs/flexnn/package/plan.md` FP-12** *(originally task P10, merged into FP-12 on 2026-07-21 — one guard for every illegal-under-`NESTED` configuration)*.

**Exposure today is theoretical, and that is the reason to gate now rather than later:** no
programme driver selects a non-sanctioned layout, but `get_hyperparameter_search_space` offers all
three as categorical choices (`automl_package/models/probabilistic_regression.py:530`), so a tuning
run may select one, and the only constructor gate on `NESTED` checks `uncertainty_method`, not the
head layout (`:163-168`).

**M1 and M2 train identically — with k-dropout** (per-sample `k ~ Uniform{1..k_max}`,
`NClassesSelectionMethod.NESTED`,
`automl_package/models/flexnn/strategies/n_classes.py:173`) — and are read off the
SAME trained network. Training is therefore NOT a variable between them; they differ in exactly one
thing, **how k is chosen**, which is what makes that contrast controlled.

⚠️ **TRAINING-SCHEDULE RULING (user, 2026-07-21; MASTER Decision 20) — the per-sample uniform draw
is RETIRED as M1/M2's default schedule; the migration is task P7.** The nested/anytime property —
one network readable at every rung — is unchanged; what changes is how gradient is allocated across
rungs per step: the objective becomes the mean over rungs k ∈ {1..k_max} of that rung's NLL, every
step, from ONE forward. Three facts force this: (i) every rung's output is already computable in
essentially one forward (`NestedStrategy.all_rung_outputs`,
`automl_package/models/flexnn/strategies/n_classes.py:207`), so per-sample sampling
buys no compute here; (ii) the width strand RAN a per-sample uniform draw and recorded its failure —
the top rung trains only 1/k_max of the time
(`docs/flexnn/width/history/width_dial_2026-07/EXECUTION_PLAN.md:119-129`, fixed by the W2 sandwich); (iii) this
strand's own middle-k coherence failures (§2, 8/9 with max-k at 9/9) are plausibly the same
starvation pathology — see P5's prior-art clause. The draw survives only as a LABELLED comparison
arm. **Until P7's re-validation lands, existing k-dropout results remain citable only as the OLD
(per-sample draw) schedule, labelled as such; no NEW result may be produced on the old schedule.**
The M1-vs-M2 single-difference contrast is unaffected — both read the same network and migrate
together.

⚠️ **M3 DOES NOT USE k-DROPOUT (user ruling, 2026-07-20). Each of M3's per-k models is trained
ORDINARILY at its own fixed k** (`NClassesSelectionMethod.NONE`), which is what "a model dedicated
to that k" means. A per-k model trained with dropout across the whole ladder is not dedicated to
anything — it is the same network read at a different point, and it cannot serve as an independent
reference. **This is deliberate and it is not a confound**: M3 is not an arm in a controlled
contrast, it is the expensive ceiling the cheap methods are measured against, and training dedicated
models is precisely what makes it expensive. The single-difference rule binds **M1 vs M2**; it does
not bind the reference.

*(Corrected 2026-07-20 after review. The earlier text said all three trained identically with
k-dropout. That made M3 not-a-reference, and it also broke P3's positive control: the published
coherence check compares against "a separately trained **ordinary** model"
(`docs/probreg/shared/reports/probreg_kselection/probreg_kselection.md` §3.2), and the current per-k sweep code
already builds them that way — `_probreg_fixed` sets `NClassesSelectionMethod.NONE`
(`automl_package/examples/report_a_benchmark.py:185-191`). The plan was wrong; the code was right.
**`docs/probreg/shared/benchmark_spec.md` §2.0 and §2.3 still carry the superseded wording and
must be corrected — see P0-b1 below.**)*

✅ **M3's CANDIDATE SET SPANS THE SAME RUNGS AS THE LADDER, BYPASS INCLUDED — RULED BY THE USER
2026-07-21 (MASTER Decision 25). No longer open; P3 and P4 are unblocked on this count.** The grid is
widened to cover k ∈ {1..k_max}, bypass rung included, so both sides can return the same answer. **The
cost increase is accepted and must be REPORTED, never absorbed silently** — a wider grid makes the
reference more expensive, and the reference's price is the denominator of every efficiency claim
here. The narrow-grid fallback below is therefore NOT taken; it is retained only as the record of
what was rejected.

*(Original framing retained below as the pre-registration this ruling answers.)*

⚠️ **M3's CANDIDATE SET MUST SPAN THE SAME RUNGS AS THE LADDER, BYPASS INCLUDED — raised by the user,
2026-07-21.**

**The asymmetry.** M1/M2 read a ladder over **k ∈ {1..k_max}, bypass rung included** — stated in
this file's own schedule ruling ("the loss is the mean over rungs k ∈ {1..k_max} (bypass rung
included)") and implemented as rung 1 = the direct-regression head. **M3's grid today starts at 5**:
`K_GRID = (5, 8, 10, 12)`, "P1's pre-registered small grid"
(`automl_package/examples/report_a_benchmark.py:102`), consumed by `select_k_for_toy` (`:331-350`),
which fits a dedicated fixed-k model per grid point and takes the argmin validation NLL.

⚠️ **SECOND DEFECT AT THE SAME SITE — M3's selector scores on a LEARNED variance (found 2026-07-21,
root, during the user-instructed variance audit; independent of the grid question above).**
`select_k_for_toy` selects by `calculate_nll(y_val, y_pred, y_std)` where
`y_std = model.predict_uncertainty(x_val)` (`automl_package/examples/report_a_benchmark.py:341-345`)
— i.e. the model's **fitted** variance. That is exactly the violation that voided P8 and that the
σ-scope decision removed. **The benchmark spec's §4.3 lists three code sites that must substitute the
shared constant** (the router's error table, the arbiter's per-rung readout, and the reported
metric); **this driver is a FOURTH site and is not on that list**, so a reader following §4.3 to
completion would still leave M3 selecting on a learned σ. **⇒ M3's selector must be migrated to the
fixed-σ mixture log-likelihood in the same task that generalises it, and §4.3's site list must gain
this entry.** Until then, no M3 number is compliant with §0.5's fixed protocol.

**Why this is not bookkeeping.** The cheap arms can select *"do not discretize at all"*; the
reference structurally **cannot** select anything below k=5. On any cell where the honest answer is
the bypass — and this strand's own negative control is smooth data, where the bypass IS the right
answer — M1 would beat its own ceiling **for a reason that has nothing to do with selection
quality**. An arm that outscores the reference because the reference was not allowed to consider the
winning candidate is not evidence about selection; it is a rigged grid. Note the direction: this
flatters the cheap method, which is the direction a reader will least suspect.

**Status: OPEN, deliberately not ruled here.** This file defines M3 as *generalising*
`select_k_for_toy`, and never pinned the generalised grid — checked before this block was written
(`grep -in 'k_grid\|k grid'` over this file returned only the constant's own mentions, no ruling).
So this is an **unpinned decision surfacing early**, not a defect in shipped code, and the current
grid is the old driver's, not M3's.

**Recommendation to rule on (root, for the user):** M3's candidate set spans the same rung range the
ladder is read over, **including the bypass**, so that both sides may return the same answer. The
cost consequence is honest and must be reported, not hidden: widening the grid raises M3's price,
and M3's price is the denominator of every efficiency claim in this strand. If the grid is instead
left narrow for cost reasons, then **every M1-vs-M3 comparison must exclude cells whose selected
rung lies outside M3's grid**, and say so in the report. What is NOT acceptable is a narrow grid
with an unqualified "M1 matches M3" headline.

**Binds:** **P3** (the M1 ≈ M3 claim, both halves) and **P4** (real data + baselines) — neither may
report an M1-vs-M3 headline until this is settled. Also fold the ruling into §3.6's frozen-constants
table when it lands, since the grid becomes a constant the battery reads.

🔑 **EACH MODEL IS THE COMPLETE SYSTEM, INCLUDING ITS SELECTION MACHINERY** (user, 2026-07-20).
**M1 = ProbReg + arbiter. M2 = ProbReg + distillation. M3 = ProbReg + sweep selector.** Every one is
scored end-to-end and costed end-to-end: the selection step is *inside* the model, never a
side-analysis reported next to it. A table row for M1 is the arbiter's answer, not the network's
answer with the arbiter mentioned in a footnote. This is binding on the driver, the metrics and the
report.

| Model | = ProbReg + | How k is chosen | Cost | Mechanism |
|---|---|---|---|---|
| **M1** | **the arbiter** | ONE k for the dataset | cheap | `all_rung_log_likelihood` (`automl_package/models/flexnn/strategies/n_classes.py:230`) — see ⚠️ below, NOT `held_out_arbiter_advantage` |
| **M2** | **the distillation** | a k **per input** | cheap | `fit_router` + routed predict (`automl_package/models/probabilistic_regression.py:754`) |
| **M3** | **the sweep selector** | ONE k for the dataset, by training a **separate ORDINARY model per k** (no k-dropout, `NClassesSelectionMethod.NONE`) and scoring each on held-out data | **expensive — the reference** | generalise `select_k_for_toy` (`automl_package/examples/report_a_benchmark.py:331`), which already builds fixed-k models via `_probreg_fixed` (`:185-191`) |

⚠️ **M1's mechanism corrected 2026-07-20 (repair pass).** `held_out_arbiter_advantage`
(`automl_package/models/probabilistic_regression.py:845-916`) was named here originally and is
**the wrong shape for M1's job** — it returns a per-input `(N,)` array comparing ONLY the top rung
against the k=1 bypass (`:905-907`), and cannot express "chose k=4" for any middle rung, let alone
one global answer for the whole dataset. The primitive that CAN — a full `(batch, n_classes)`
per-rung held-out likelihood table, exactly what a cheapest-within-tolerance selector needs to run
over — is `all_rung_log_likelihood`
(`automl_package/models/flexnn/strategies/n_classes.py:230-252`), which as of this
repair has **zero callers anywhere in the repo**
(`grep -rn "all_rung_log_likelihood" automl_package/ tests/` matches only its own definition). PA
builds M1's selector on top of it, not on `held_out_arbiter_advantage`. Recorded as **D5/D6** in §3.

**What M3 is for.** M3 is the honest, expensive way to pick k. The efficiency claim of this whole
strand is that **M1 reaches M3's answer at a fraction of the cost**. M3 is therefore not optional
and not a baseline — it is the thing M1 is measured against. M1 and M3 differ only in *how the
same global k is found*; M2 is the separate question of whether k should be global at all.

**Both halves of the M1≈M3 claim must be tested, and they are different claims:**
- **(a) same quality** — read at a given k, does the k-dropout model match a model dedicated to
  that k? *(Partially established on toys — §2.)*
- **(b) same choice** — does the arbiter pick the k that the sweep would pick?
  *(NOT tested anywhere. §2.)*

**M2 runs on a DIFFERENT tolerance rule, and that is legitimate, not an oversight.** The global arms
(M1, M3) select k at **twice a bootstrap-estimated standard error** — the smallest k whose held-out
score is not meaningfully worse than the best (PA's selection rule, reusing the tolerance published
in `docs/probreg/shared/reports/probreg_kselection/probreg_kselection.md` §3.2). M2's distilled router does not
read a curve — it labels each row independently at a flat relative margin,
`DEFAULT_TOLERANCE = 0.25` (`automl_package/models/flexnn/routing.py:57`), applied as
`error <= (1 + tolerance) * row_min` (`:80`). The two rules legitimately differ because the two
selection problems differ: a per-input labelling decision has one row's worth of evidence, and no
standard error is estimable from a single observation, whereas a global chooser reads a whole
held-out curve, over which a bootstrap standard error is exactly the right notion of noise.
**Consequence, stated so the report does not paper over it: M2's chosen k values are NOT directly
comparable to M1's or M3's on tolerance grounds** — the three models share a cost objective, not a
shared statistical selection rule; comparison lives on held-out error and cost.

### 1.1 Contradictions this replaces — all three were live simultaneously

Recorded so the same error cannot be reintroduced by reading an older document:

1. **`docs/flexnn/PLAN.md`'s naming key** defined variable-k as *"dynamic-k (ELBO + SoftGating)"* —
   **in-training** selection, which `docs/RULES.md` Decision 13 itself demotes to a labelled
   comparison arm. Self-contradictory. **CORRECTED 2026-07-20**: that entry now points here and
   states no definition of its own. *(No line number is cited on purpose — a line reference into a
   file this one is actively rewriting is a cache entry that rots within the hour. It did: the
   first draft of this bullet cited `:38-39`, which pointed at an unrelated entry ten minutes
   later.)*
2. **`docs/probreg/shared/benchmark_spec.md:66`** defines Model 1 with
   `n_classes_selection_method=NONE` — **k-dropout OFF** — and k chosen by hyperparameter tuning.
   Against §1 this arm differs from Model 2 in **two** ways at once (training scheme AND k
   choice), so no result could be attributed to either. The spec's own §3.4 claims the two arms
   *"differ in exactly one thing"*; that claim is **false as written**. **→ fix in task P0.**
3. That spec has **no M3 at all** (`grep -c "arbiter"` on it returns 0), so the efficiency claim
   had no reference to be measured against.

**Confound doctrine (MASTER Decision 15 generalised):** an arm that differs from its comparator in
more than one respect is NOT dispatchable. State the single difference in the task, or do not run
it. This is the same failure that invalidated the depth pilot; it was about to repeat here.

---


<!-- from docs/plans/capacity_programme/probreg.md lines 662-690 -->
## 4. Tasks

Order is **`docs/flexnn/package/plan.md` FP-12 → P0 → P0b → P0-b1 → P1 → P2 → PA → P7 → PB ∥ PC ∥ P8 →
P11 → P3 → P4 → P5 → P9 → P6** *(P10 removed 2026-07-21 — merged into FP-12)* *(amended
2026-07-21: P7, the Decision-20 schedule migration, precedes every task that trains M1/M2; P8, the
Decision-21 regularisation check, runs parallel and gates P4's read; P0-b1 and P1 were found
already landed — see their headers)*.

⚠️ **ORDER LINE CORRECTED 2026-07-21 (root, after an execution-level audit). P9, P10 and P11 all had
full specs — write set, deps, verify line — and NONE of them appeared in this order line.** A
dispatcher following the order literally would never have scheduled any of the three, and since P10
gates P11, the omission also severed P11's only path into the sequence. *(Recorded as case law: a
task is not scheduled because it is well-specified; it is scheduled because it is in the order. Both
are required, and this file had drifted so that adding a task no longer meant adding it here.)*

**Two things this linear line cannot express — read them before partitioning waves:**
- **FP-12 is WAVE ZERO and lives in another strand.** It builds the fixed-σ scorer that MASTER
  Decision 24 makes the capacity readout. Until it lands, **every task here that selects or reports a
  chosen k reads through a mechanism that scores on a LEARNED variance** — the forbidden case. That
  includes P8's re-run, P11, PB, PC, P3, P4, P5 and P7's re-validation.
- **FP-12 then P7 are a STRICT SERIAL CHAIN** *(was three tasks; P10 merged into FP-12 on 2026-07-21)*: both write
  `automl_package/models/probabilistic_regression.py`. The write-set guard is **session-scoped, not
  liveness-scoped**, so they cannot be worker-written in one session even sequentially — they need
  separate sessions or a root-applied handoff. **Plan this at dispatch, not on discovery.** (PT is parked and gates nothing — the existing toys are retained.) P0–P2 plus PA are the "fix it properly
and completely" phase the user gated everything else behind (2026-07-20); no comparison compute runs
until they close. **PB and PC must precede P3/P4**: both fix a *parameter of the method* (how much
selection data, what router), and running the battery before they are settled would produce results
nobody could attribute — which is the failure this whole strand was created to stop.


<!-- from docs/plans/capacity_programme/probreg.md lines 1725-1729 -->
## 5. Non-goals for this strand

No new selection strategies. No change to `RegressionStrategy` or the loss family. No revival of
in-training k selection as a primary (MASTER Decision 13) — it may appear only as a labelled
comparison arm. No variance-programme work (MASTER Decision 2).

<!-- from docs/plans/capacity_programme/MASTER.md lines 159-161 -->
4. **Hetero-NLL default** (strand 2): minimal fix if one exists cheaply; otherwise report (a)
   documents the limitation honestly and variance stays parked. Failure there does NOT unpark
   the variance programme.

<!-- from docs/plans/capacity_programme/MASTER.md lines 372-441 -->
24. **The capacity readout is a likelihood at FIXED σ — never squared error, never a learned variance
    (root, 2026-07-21, promoted to MASTER after the strand file was found contradicting it).**
    Promoted here *because* it was ratified only in `docs/probreg/shared/benchmark_spec.md` §4.2/§4.3
    and `docs/probreg/shared/plan.md` §0.5 had drifted to the exact opposite. A rule that lives in one document does not
    bind the plan; this register is where it binds.
    - **The readout for choosing capacity is the per-sample fixed-σ mixture log-likelihood.**
      σ is ONE SHARED CONSTANT, so nothing is fitted and this is not a variance metric.
    - **Squared error / RMSE is a reporting column, NEVER the readout.** Spec §4.3: selecting on it
      *"would be selecting on a quantity that is structurally blind to k for symmetric targets… the
      selection curve would be flat, the cheapest-within-tolerance rule would return k=1 everywhere,
      and the result would look like a clean finding."* A metric that cannot see the dial returns a
      confident answer about the dial — the worst available failure mode.
    - **A likelihood read at a LEARNED `log_var` is FORBIDDEN.** This is the violation that voided P8.
      **The test is not "is it a likelihood?" but "is σ learned?"** — Decision 21's amendment (a check
      runs on its own strand's sanctioned objective) is what this makes operational.
    - **Both errors are live and were found by audit, not by a failing run** — which is the point:
      neither would have thrown. The strand file mandated squared-error selection, and the reference
      arm's own selector still scores on a fitted variance at a code site the spec's migration list
      omits. **Every task that selects, scores, or reports a chosen capacity states its metric
      explicitly, and "as the strand does" is not a statement.**
    - **Width parity:** Decision 2 (amended) already fixes σ at the generator's true value for width.
      Same principle, two dials — neither strand learns a variance it then selects on.

25. **M3's candidate set spans the same rungs the ladder is read over, bypass included (user,
    2026-07-21).** The cheap arms (M1/M2) read k ∈ {1..k_max} including the bypass rung; M3's grid
    started at k=5, so the reference could not consider a candidate the cheap arms could select.
    - **On any cell where the bypass is the honest answer — including the strand's own smooth-data
      negative control — M1 would have beaten its own reference for a reason unrelated to selection
      quality**, and in the direction that flatters the cheap method.
    - **Cost is accepted and must be reported, not hidden:** a wider grid makes M3 more expensive, and
      M3's price is the denominator of every efficiency claim in the strand. This ruling makes the
      cheap arms look *worse*, which is why it is the conservative choice.
    - **Unblocks `docs/probreg/global_k/plan.md` P3 and P4**, whose deps carried this as an explicit blocking entry.

26. **ProbReg trains at FIXED σ — a MODEL-DEFINITION change (user, 2026-07-21).** σ is not fitted
    anywhere in ProbReg: not in selection (Decision 24), and now not in training either. This is the
    ProbReg counterpart of Decision 2 (amended) for width — same principle, two dials, one premise.
    - **The reason is a confound, not consistency.** With a learned per-class variance, **a single
      component can absorb dispersion by widening itself**, so the model fits spread-out data without
      needing more components — which is exactly the question "does this input need more k?" is trying
      to measure. Fixing σ removes that escape hatch and forces dispersion to be explained by
      **structure** (more components) rather than by **width** (one fatter component).
    - **What it means mechanically:** with σ constant the rung NLL reduces to squared error up to a
      constant, so training becomes MSE on the mixture mean. **⇒ ROOT-APPLIED IMPLEMENTATION RULING
      (flagged for the user to overturn if unintended): the per-class heads predict MEANS ONLY, and
      the within-component term in the law-of-total-variance combination becomes the shared constant.**
      The alternative — keeping a `log_var` output that the loss no longer trains — would leave an
      **untrained head that `predict_uncertainty` would happily expose**, which is a worse trap than
      the one this decision removes.
    - **Predictive spread stays meaningful:** it becomes the between-component spread plus the
      constant. The variance *machinery* is not deleted (Decision 2's carry-over); it is simply never
      fitted and never selected on.
    - **⚠️ CONSEQUENCE — the suite bar must be RE-BASELINED before this lands.** The two accepted
      heteroscedastic failures test variance behaviour directly, and this changes what the model can
      express within a component. **The "no new failures, no newly-passing tests" bar cannot be
      carried across this change** — re-baseline first, record the new expected result, and treat any
      *other* movement as the regression signal.
    - **All prior k-dropout numbers become OLD-OBJECTIVE** and are citable only when labelled as such
      — the same treatment Decision 20's retired schedule already receives. Two labels now travel
      with historical ProbReg results: old-schedule and old-objective.
    - **Scope consequence:** `docs/flexnn/package/plan.md` **FP-12** grows from a scoring change to a scoring
      **and training** change, and now overlaps `docs/probreg/PLAN.md` **P7** (which rewrites the training
      objective for the schedule migration). **Those two must be sequenced deliberately or merged.**

27. **ProbReg head-layout arm list (user, 2026-07-21).** Separate per-class heads are the model of
    record (`docs/probreg/PLAN.md` §1). The single-head-with-per-class-outputs layout is **retained as a
    LABELLED comparison arm, never a default** — it distinguishes "components help" from
    "*independent* components help". The single-head-final-output layout stays **blocked under
    `NESTED`** and serves only as P11's mechanism control, because it produces no components at all.


<!-- from docs/plans/capacity_programme/MASTER.md lines 448-522 -->
29. **Under the nested ladder, nothing may choose or shape k DURING TRAINING (user, 2026-07-21).
    Amends Decision 13 from "demoted to a labelled comparison arm" to "unreachable by default".**

    **⚠️ PROGRAMME-WIDE — this binds ProbReg AND FlexNN (width/depth), user 2026-07-21.** It is one
    principle about *when* capacity is chosen, not a fact about one dial. Inventory verified on disk
    before writing:

    | dial | in-training machinery this retires | where |
    |---|---|---|
    | **ProbReg (k)** | `SOFT_GATING`, `GUMBEL_SOFTMAX`, `STE`, `REINFORCE` · `NClassesRegularization`'s `K_PENALTY` and `ELBO` (the enum's own docstring: *"only meaningful with an in-training selection method"*) · the cross-entropy training modes `COMPOSITE_LOSS` / `GRADIENT_STOP` / `CE_STOP_GRAD` | `automl_package/enums.py` · `automl_package/models/flexnn/strategies/n_classes.py` |
    | **FlexNN depth** | `GumbelSoftmaxStrategy`, `SoftGatingStrategy`, `SteStrategy`, `ReinforceStrategy` · `DepthRegularization`'s `DEPTH_PENALTY`, `ELBO`, `COST_AWARE_ELBO` | `automl_package/models/flexnn/strategies/layer.py:52,98,144,278` · `automl_package/enums.py:96` |
    | **FlexNN width** | **nothing left to retire** — `WidthSelectionMethod` was removed entirely by FP-3 and width already selects only through the distilled router. Recorded so nobody hunts for a width equivalent that does not exist. | — |

    **Survivors in every family: `NONE` (fixed capacity) and `NESTED` (the ladder).** Depth's evidence
    points the same way as ProbReg's — the FlexNN ELBO depth-selection claim was **refuted** (M0,
    post-fix ELBO collapsed to depth=1 on all five seeds), which is the depth-side instance of exactly
    the failure Decision 13 recorded for width.

    **Why one rule and not three patches.** Three separate problems were found in one session — the
    head layout with no components, cross-entropy against per-k rebinned targets, and in-training
    k-selection — and **all three share a shape: the plan assumed a configuration, nothing enforced
    it, and `get_hyperparameter_search_space` could select the unsafe one.** A tuning run could reach
    every one of them. One enforcement point closes all three.

    **Two distinct justifications, recorded separately so neither carries the other's weight:**
    - *In-training selection* is retired because **it does not work** — Decision 13's finding
      (failed for width; the FlexNN ELBO depth-selection claim refuted), which is why the arbiter and
      the distilled router exist at all.
    - *Cross-entropy* is retired on a **different and independent** argument: it trains the classifier
      toward a **predetermined percentile carve-up of y**, which contradicts the premise that k is an
      adaptive resolution dial driven by difficulty and signal-to-noise. **It is NOT part of the
      in-training-selection programme** — it applies equally to a fixed-k model — so "it's legacy"
      would have been the wrong reason.

    **Consequence:** `NClassesSelectionMethod` has two live members under the ladder — `NONE` and
    `NESTED`. The design's actual API.

    **The escape hatch — ROOT-APPLIED DEFAULT, flagged for the user to strike.** The user approved the
    rule; the root chose the reversible option on the open sub-question. An **explicit opt-out flag**
    re-enables the retired members **for the labelled-comparison-arm purpose only** (publishing "here
    is what in-training selection does, and here is why we moved to distillation"). Without it,
    Decision 13's comparison-arm provision would be silently overridden and the comparison would be
    **awkward to retrofit later** — the reason the root defaulted to keeping it. The flag is never set
    by a search space, never a default, and any run using it is labelled in its results JSON.

    **RETIREMENT IS NOT DELETION — *yet*.** The code stays for now. Deletion remains user-gated behind
    the four mechanical eligibility checks, attended, via `docs/probreg/shared/plan.md` P9's manifest — several of these
    paths produced results that are still cited.

    **🗑️ CONDITIONAL DELETION TRIGGER (user, 2026-07-21).** *"If we reach a conclusion that the
    arbiter/distillation setup works, we delete all machinery we have for in-sample k selection."*
    - **The condition is the strand's own headline**, not a judgement call: the arbiter matches the
      expensive sweep, and the distilled router works per input — i.e. **P3 and P4 pass** on the
      widened grid (Decision 25) at the fixed-σ readout (Decision 24).
    - **Applies to FlexNN's machinery on the same trigger** (user: *"this applies to both"*). Depth's
      in-training strategies and `DepthRegularization` become deletion-eligible on the same
      conclusion — **with one asymmetry that must not be glossed:** depth is PARKED, so its own
      arbiter/router evidence does not exist yet. **Deleting depth's machinery on ProbReg's evidence
      is an inference across dials, not a measurement.** The M0 refutation makes it a *reasonable*
      inference — but it is one, and the manifest must say so rather than presenting depth's deletion
      as equally evidenced. If depth is ever unparked, that machinery is what its comparison arm
      would have needed.
    - **The trigger point is the joint results review** (Decision 23), where both strands' results are
      walked together. If the condition holds there, the in-sample machinery becomes
      **deletion-ELIGIBLE**, and P9 brings the manifest. Eligibility still runs the four mechanical
      checks; the user's ruling supplies the *intent*, not a bypass.
    - **⚠️ ORDER MATTERS — the trigger and the escape hatch collide, and the collision is one-way.**
      The opt-out flag exists so the comparison *"here is what in-training selection does, and why we
      moved to distillation"* can be run and published. **Deleting the machinery makes that comparison
      permanently unrunnable.** ⇒ **If that comparison is wanted in the report, it must be RUN BEFORE
      the deletion, not after.** Deciding at the review is too late if the run has not happened.
      **Root's recommendation: decide at the review whether the comparison earns a place in the
      report; if yes, run it there and then, and delete afterwards.** Recorded here because this is
      exactly the kind of ordering that is obvious in advance and invisible in the moment.



## Code and results of this stream

Where this stream's experiment scripts and results live; the code stays where it is until the
planned code move. Scripts are in `automl_package/examples/`; `LADDER/` is
`automl_package/examples/capacity_ladder_results/`, whose `RESULTS.md` records what completed.

- **fixed_k** — scripts: `probreg_structure_battery.py`; results: `LADDER/P2/`, `LADDER/PS1/`, `LADDER/PS2/`, `LADDER/PS3/`, `report_a_results/`
- **global_k** — scripts: `probreg_p8.py`, `report_a_benchmark.py`; results: `LADDER/P8/`
- **history** — scripts: `_kselection_metrics.py`, `_phase2_followups_chain.sh`, `_variational_em.py`, `_variational_em_perinput.py`, `capacity_ladder_v2.py`, `capacity_ladder_v3.py`, `capacity_ladder_variance_v0.py`, `capacity_ladder_variance_v1.py`, `capacity_ladder_x2.py`, `classifier_symmetry_check.py`, `dynamic_k_comparison.py`, `exponential_symlog_probe.py`, `gumbel_elbo_retest.py`, `head_degeneracy_diagnostic.py`, `head_structure_diagnostic.py`, `middle_class_centering.py`, `probabilistic_regression_showcase.py`, `probreg_ablation.py`, `probreg_elbo_prior_check.py`, `probreg_identifiability_sweep.py`, `probreg_k10_sweep.py`, `probreg_k20_sweep.py`, `probreg_k_sweep.py`, `probreg_kselection_comparison.py`, `probreg_kselection_diagnostic.py`, `probreg_kselection_experiments.py`, `probreg_kselection_prior_ablation.py`, `probreg_mixture_eval.py`, `probreg_ordering_ablation.py`, `probreg_snr_sweep.py`, `probreg_variational_em_step1.py`, `probreg_variational_em_step2_perinput_arbiter.py`, `probreg_variational_em_step3_perinput_model.py`, `probreg_variational_em_toy_e_hump.py`, `run_phase2_followups_safe.sh`, `run_probreg_k10_sweep_safe.sh`, `run_probreg_k_sweep_safe.sh`, `sep_heads_vs_single_final.py`; results: `LADDER/V0/`, `LADDER/V1/`, `LADDER/V2/`, `LADDER/V3/`, `LADDER/X2/`, `classifier_symmetry_results/`, `exponential_symlog_probe_results/`, `gumbel_elbo_retest_results/`, `head_degeneracy_results/`, `head_structure_results/`, `middle_class_centering_results/`, `prior_ablation_results/`, `probreg_ablation_results/`, `probreg_elbo_prior_check_results/`, `probreg_identifiability_results/`, `probreg_k10_sweep_results/`, `probreg_k20_sweep_results/`, `probreg_k_sweep_results/`, `probreg_kselection_diagnostic_results/`, `probreg_kselection_experiments_results/`, `probreg_ordering_ablation_results/`, `probreg_snr_sweep_results/`, `sep_heads_investigation/`, `variational_em_comparison_results/`, `variational_em_step1_results/`, `variational_em_step2_results/`, `variational_em_step3_results/`, `variational_em_toy_e_results/`
- **per_input_k** — scripts: `_capacity_ladder.py`, `_capacity_ladder_nested.py`, `capacity_ladder_h1.py`, `capacity_ladder_k1k2k3.py`, `capacity_ladder_k4.py`, `capacity_ladder_k5.py`, `capacity_ladder_k6.py`, `capacity_ladder_s1.py`, `capacity_ladder_s2.py`, `capacity_ladder_t2.py`, `capacity_ladder_t3.py`, `capacity_ladder_x4.py`, `capacity_ladder_x4b.py`; results: `LADDER/H1/`, `LADDER/K0/`, `LADDER/K1K2K3/`, `LADDER/K4/`, `LADDER/K5/`, `LADDER/K6/`, `LADDER/S1/`, `LADDER/S2/`, `LADDER/T2/`, `LADDER/T3/`, `LADDER/X4/`, `LADDER/X4b/`
- **shared** — scripts: `_toy_datasets.py`
