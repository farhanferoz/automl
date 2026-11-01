# ProbReg — stream plan

The one plan for the ProbReg stream: the end-to-end classification-bottleneck regressor
(`automl_package/models/probabilistic_regression.py`). It holds the owner's rulings and objectives,
the model definitions, the task order across sub-plans, the stream's decisions and its open
questions. Work items live in the sub-plans. Rules that bind every stream: `docs/RULES.md`.

> ⛔ **Owner ruling 2026-10-22 — overrides every decision and plan text below that conflicts with it.**
> ProbReg is a **regression model regularised by classification**, not a mixture model. The classifier is
> trained on class labels only (cross-entropy against y bins, `CE_STOP_GRAD`); its probabilities reach the
> heads without gradient, so the regression error cannot pull them off the true bin probabilities. The
> yardstick is held-out regression error. The fixed-σ mixture loss and readout, regression-only supervision as the default, and
> Decision 29's retirement of cross-entropy training are withdrawn. Text below that rests on them is
> superseded and is to be removed under an owner-approved plan (pending).

## Plan of record (owner, 2026-11-01) — the order of work; overrides the task order in §4

**Objective (owner, verbatim):** "we need to find the best model combination for PROBREG that works ... we have
different configurations for PROBREG. We need to investigate them. And we need to find the best one which is
better than all the standard models that we have lined up on our toy."

**Rulings this plan rests on (2026-11-01):**
- *Validation reads the fit term only.* Early stopping, hyperparameter search, cross-validation and the choice of k
  score held-out data with only the part of the training loss that measures the final prediction against y; every
  regularisation term (the classifier's cross-entropy included) is ignored. Determined automatically for any loss.
- *Design choices belong to the study, never to the search.* What the classifier learns from, the head layout and
  the boundary-regularisation type are settled once, across problems, by this study. The search tunes only learning
  rate, depth and width, dropout, regularisation weights and k. Defaults meanwhile: class labels only
  (`CE_STOP_GRAD`), separate heads.
- *Only likelihoods and priors train the model.* No penalty without a probabilistic meaning; the boundary,
  ordering and middle-class penalties stay off. If the heads do not stay ordered and inside their bins unaided, the
  remedy is structure (anchored heads, the hard-limit boundary, an ordinal classifier), not a penalty.
- *Fixed spread by default.* Heads predict the mean only and the combined mean is trained by squared error; a spread
  fitted in-sample comes out too small. Mean-and-spread heads are a comparison arm; any spread needed later comes from
  held-out data.
- *Comparators:* the plain network (tuned with the same search budget, and one at ProbReg's parameter count),
  LightGBM, linear regression; each toy's best possible error as the floor.
- *Yardstick:* held-out regression error on a fresh test draw per seed, never used to stop, tune or choose.
- *Toys (for now):* toy_a (easy control), heteroscedastic, toy_h, hetero3, exponential, piecewise; the zero-mean toys
  are dropped from this claim; new toys aimed at regression error come as specs for the owner first.
- *Choosing k:* fixed k and the per-k sweep (without its constant k = 1 candidate) first; the nested model and the
  router stay in scope but only once designed properly (properly nested; a router fitted and checked on separate
  held-out data), following AutoCast Private's nested rank-selection work.

**Order of work** (each step finished and checked before the next):
1. **Fix the tools.** Fit-term validation, the best-epoch restore fix, the trimmed search space, the middle-class
   head tied to the monotonic constraints, the sweep without k = 1. Done when the tests pass, every changed file has
   been read, and the owner gives the go to commit.
2. **Remove the stale mixture wording** from the plan documents and toy docstrings.
3. **The review** (`docs/probreg/review/`, narrowed): checks the code against the rulings above, then proposes the
   experiments for steps 5a-5d, each stating what is run, on which toys and seeds, and what each outcome would mean,
   decided before anything runs.
4. **Owner and analyst go through the proposed experiments one at a time.** Nothing runs until the owner agrees it.
5. **Run, in order:**
   a. *Test bench* — the toys, the comparators and the data rules per seed (train; validation for stopping, tuning
      and k; a separate set for choosing between configurations; a fresh test set for grading). Every model is
      shown to train to the end. Done when the plain network comes close to the best possible error on toy_a.
      *Training protocol (owner, 2026-11-01, approved):* early stopping always on, on the held-out fit term, restoring
      the true best epoch (the library default is no early stopping and 50 epochs, so the protocol sets it explicitly);
      patience set from validation curves measured here, not guessed; an epoch limit high enough never to stop a run,
      with every run that reaches it reported as not converged; one training pass (no refit on train + validation,
      so validation stays held out); no cross-validation — fresh data per seed does its job; tuning, where used, on
      the validation split.
      *Data size (owner, 2026-11-01, approved):* measured, not guessed. Per toy, the plain network and default ProbReg
      at doubling training sizes (250 to 16,000 points), error above the toy's best possible error against size; the
      study then runs at two sizes: a small one where models sit clearly above the floor (where a regulariser should
      help most) and a large one near the plateau. Test sets sized from each toy's per-point spread so their noise is
      far below the gaps (toy_a: 10,000 points give about 0.0035 against a floor of 0.25).
   b. *First comparison* — today's default ProbReg against the comparators.
   c. *Configuration study, one design choice at a time*, keeping each winner: what the classifier learns from
      (with "learns nothing" as the control); the head layout; whether the heads stay ordered and in their bins
      unaided, and structure if not; how k is chosen. Then the winner against its near neighbours, in case two
      choices interact.
   d. *Confirmation* — the final configuration against every comparator on fresh draws no earlier step touched.
      This is the claim.
6. **Real data** — the owner's own dataset (to be briefed), in place of the Kepler/UCI list in O4.

**Speed (owner, 2026-11-01): train many small models at once on the GPU.** One toy run is barely faster on the GPU
(11.0 s against 13.9 s on one CPU thread, 2026-10-21 timing in `docs/probreg/review/groundwork_2026-10-22/runtimes.md`),
because the networks are tiny. Stack the runs of one experiment (its seeds, or its configurations) into one batched
computation on the GPU. Designed by the main session, built by Sonnet at medium effort, adopted only if a measurement
shows it is faster than parallel CPU runs and gives the same results. Before step 5.

## Sub-plans

| Sub-plan | What it covers | Status |
|---|---|---|
| `docs/probreg/fixed_k/plan.md` | the base model, k set by the user: the accepted failing test (P2). The structure study and the head-structure comparison (P11) were removed 2026-10-22 | — |
| `docs/probreg/global_k/plan.md` | one k per dataset, chosen on held-out data: the cheap nested read-out (M1) against the per-k sweep (M3); P3, P5, P8 | open; P8's earlier results are not citable (see its entry) |
| `docs/probreg/per_input_k/plan.md` | a distilled router picks k per input (M2): how much selection data (PB), the router's architecture (PC) | open |
| `docs/probreg/shared/plan.md` | work serving all three: code organisation (§0.5, P9), execution contract and frozen constants (§3.5, §3.6), P0, P0-b1, PA, P1, the training-schedule migration (P7), real data (P4), the report (P6), P10 | see each task's header |
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
| §0.5 code organisation (with P9), §3.5, §3.6, P0, P0-b1, PA, P7, P1, P4, P6, P10 (P0b removed 2026-10-22) | `docs/probreg/shared/plan.md` |
| the canonical toy suite, PT | `docs/probreg/shared/toy_suite.md` |
| §2, §2.5, §3 | `docs/probreg/shared/state.md` |
| P2 (P11 removed 2026-10-22) | `docs/probreg/fixed_k/plan.md` |
| P3, P5, P8 | `docs/probreg/global_k/plan.md` |
| PB, PC | `docs/probreg/per_input_k/plan.md` |
| MASTER Decisions 4, 25, 27, 29 (24 and 26 removed 2026-10-22) | this file, at the end |

Decision numbers are unchanged; `docs/RULES.md` lists which file holds each decision.

## Parked open question — learned σ

Removed 2026-10-22 (mixture framing; docs/probreg/cleanup_plan.md).

<!-- from docs/plans/probreg_review/objectives.md (whole file) -->
## ProbReg — objectives of the work

Status: **owner-answered 2026-10-20.** Assembled from the plan documents; the owner answered the six
open questions on 2026-10-20 — **their rulings are in the next section and override anything below
that conflicts with them.** Every objective names its source so it can be checked.

### Owner's rulings, 2026-10-20 (answers to Q1–Q6 at the end of this file)

| Q | Ruling | What it means for the work |
|---|---|---|
| Q1 | **Both** claims (calibrated uncertainty AND adaptive resolution) — **but one at a time, k first** | The current focus is k: Fixed-k as the substrate, then Global-k, then Per-input-k. Calibration (O1c) stays an objective, worked later. |
| Q3 | **The classification bottleneck must IMPROVE point accuracy**, not merely not hurt it | O1b is raised: the pre-registered bar (H3, "≤ 1.05× the best baseline") is weaker than the objective. The comparison that tests it is against a plain network of matched capacity (the no-classification/bypass arm). |
| Q5 | **Toys only for now** | O4 (real data, baselines on real data) is out of scope for the current phase. |
| Q2, Q4, Q6 | Removed 2026-10-22 (mixture framing; docs/probreg/cleanup_plan.md). | — |
| — | **k is a hyperparameter, never learned with the regression means during fitting** — chosen from held-out data or genuine Bayesian evidence | Training always prefers more classes; any route by which the training objective chooses k (including early stopping or checkpoint choice on a loss averaged over k) violates this. |
| — | **No hard thresholds or pass bars.** Results are judged by comparison — against a pure regression model, across ProbReg architectures, across ways of choosing k — using judgement, unless a difference is obvious | Numeric bars quoted below (e.g. "≥ 8 of 11 cases", "1.05×", "0.02 nats") are context from the old real-data spec, NOT bars for this phase. Comparisons are reported as paired differences with their seed-to-seed spread; each comparison states its possible outcomes and what each would mean before it is run. |

### The model, in one paragraph

ProbReg is a regression model regularised by classification (owner ruling 2026-10-22). The target range is divided into k ordered classes
(bins of y). A network reads the input and outputs a probability for each class, trained on the class labels. Each class has its own
small regression head that predicts a mean and a spread; the prediction combines the heads, weighted by the class probabilities, into
one Gaussian (its mean and spread follow from the law of total variance). The classification step is a bottleneck: the model must say *which part of the target
range* an input belongs to before it refines the value. k is therefore a **resolution dial**.

Three variants, differing only in how k is chosen:

| Name | How k is chosen | Plan code name |
|---|---|---|
| **Fixed-k ProbReg** | set by the user | the base model (`docs/probreg/PLAN.md` §1; `docs/probreg/fixed_k/plan.md` is its sub-plan) |
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
- **O1a.** Removed 2026-10-22 (mixture framing; docs/probreg/cleanup_plan.md).
- **O1b. It IMPROVES point accuracy** (owner, 2026-10-20, Q3) — over a plain network of matched
  capacity. *(The pre-registered real-data bar, `docs/probreg/shared/benchmark_spec.md` §7.4 H3, asks only
  "≤ 1.05× the best baseline"; that is now the floor, not the objective.)*
- **O1c. It gives a better predictive distribution where the data has structure** — heteroscedastic
  targets — than standard regressors. *(Source:
  report (a)'s calibration headline, `docs/probreg/fixed_k/reports/report_a/report_a_probreg_toys.md` Summary.)*

#### O2 — Global-k: k can be chosen from data, cheaply and correctly
- **O2a. Same quality.** Reading the nested network at a given k is as good as a model trained only
  for that k. *(Source: `docs/probreg/PLAN.md` §1 "(a) same quality".)*
- **O2b. Same choice.** The cheap read-out picks the k the expensive sweep picks, and the cost ratio is reported.
  *(Source: `docs/probreg/PLAN.md` §1 "(b) same choice".)*
- **O2c.** Removed 2026-10-22 (mixture framing; docs/probreg/cleanup_plan.md).

#### O3 — Per-input-k: the model can spend resolution where the input needs it
- **O3a.** Removed 2026-10-22 (mixture framing; docs/probreg/cleanup_plan.md).
- **O3b. It never loses to global-k, and pays where structure varies.**
- **O3c. It routes smooth data to the no-classification (bypass) rung.** A falsifiable prediction,
  not a gate. *(Source: `benchmark_spec.md` §7.7.)*

#### O4 — Beyond toys — ⏸ OUT OF SCOPE FOR NOW (owner, 2026-10-20, Q5: toys only)
- Real data and baselines are in scope for ProbReg only (Kepler, UCI Superconductivity, UCI Airfoil,
  then the UCI suite; baselines LightGBM, a plain network, linear regression).
  *(Source: `docs/RULES.md` Decision 3 as amended 2026-07-20; `benchmark_spec.md` §2, §5.)*
- The eventual paper headline is photometric redshift estimation. *(Source: `docs/applications/research_plan.md` §6.2,
  §7 "Paper A".)*

### Principles the owner has ruled, which every objective inherits
1. Removed 2026-10-22 (mixture framing; docs/probreg/cleanup_plan.md).
2. Removed 2026-10-22 (mixture framing; docs/probreg/cleanup_plan.md).
3. **k is chosen after training, never during** — in-training selection is retired (MASTER Decision 29).
4. **Each variant is scored as a complete system, its selection machinery and cost included**
   (`docs/probreg/PLAN.md` §1).
5. **Settle training before selection** (MASTER Decision 37, ruled for width).
6. **Only likelihoods and priors train the model** — restated, and still binding, in the plan of record (2026-11-01): no
   penalty without a probabilistic meaning (`K_PENALTY` stays retired by MASTER Decision 29).

### The six questions, as put to the owner (answered 2026-10-20 — rulings at the top of this file)

Kept for the reasoning behind each question; the rulings table at the top is authoritative.
- **Q1. What is the primary claim: calibrated uncertainty, or adaptive resolution?** Superseded 2026-11-01: the yardstick is held-out regression error; see `docs/probreg/PLAN.md` "Plan of record".
- **Q2.** Removed 2026-10-22 (mixture framing; docs/probreg/cleanup_plan.md).
- **Q3. "Classification as a regulariser" — must it HELP point accuracy, or only not hurt it?** The
  pre-registered bar (H3) only asks it not to hurt (≤ 1.05× the best baseline).
- **Q4.** Removed 2026-10-22 (mixture framing; docs/probreg/cleanup_plan.md).
- **Q5. Real data before or after the toy results are settled?** Decision 3 puts real data in scope.
- **Q6.** Removed 2026-10-22 (mixture framing; docs/probreg/cleanup_plan.md).

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

Removed 2026-10-22 (mixture framing; docs/probreg/cleanup_plan.md).

⚠️ **HEAD LAYOUT PINNED — all three models use `RegressionStrategy.SEPARATE_HEADS` (root, 2026-07-21,
at the user's instruction to correct this omission).** The plan previously never named a head layout
anywhere (`grep -in 'regression_strategy\|separate_heads' docs/probreg/PLAN.md` → **zero hits** before this
block), while every capacity-programme driver on disk already pins `SEPARATE_HEADS`
(`automl_package/examples/report_a_benchmark.py:192` and `:213`,
`probreg_kselection_experiments.py:83`, `probreg_elbo_prior_check.py:57`, `capacity_ladder_h1.py:96`,
`multi_seed_sweep.py:64`). The convention was universal and unwritten; this writes it down.

**Why it is load-bearing and not bookkeeping — the nesting guarantee is about per-class heads.** The
prefix property this strand's whole read-out machinery rests on is *"the first c rungs are a genuine
c-class model"*. That sentence requires c per-class heads to exist. The three layouts differ in
whether they produce any:

| layout | per-class outputs? | prefix-masking gives | status under `NESTED` |
|---|---|---|---|
| `SEPARATE_HEADS` | yes; head `i` sees **only** `p_i` (`regression_heads.py:368-372`, `input_size=1`) | a probability-weighted combination over the surviving k classes; masked classes carry zero weight and cannot leak into any surviving head | ✅ **the sanctioned layout** |
| `SINGLE_HEAD_N_OUTPUTS` | yes; ONE net reads the whole probability vector and emits every class's output, then the SAME weighted combination (`regression_heads.py:461`) | still a genuine k-class weighted combination — **but** each surviving class's output is computed from the zero-padded vector, so truncation leaks into every surviving class's output | 🟡 **labelled comparison arm only, never a default** |
| `SINGLE_HEAD_FINAL_OUTPUT` | **no** — the head maps probabilities straight to the prediction (`regression_heads.py:498-508`); uncertainty is whatever that single head emits, not a law-of-total-variance combination over per-class heads | a valid predictor, but not a c-class model with per-class heads | 🚫 **BLOCKED under `NESTED`** |

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
rungs per step: the objective becomes the mean over rungs k ∈ {1..k_max} of that rung's held-out fit term, every
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
which fits a dedicated fixed-k model per grid point and takes the argmin of the held-out fit term.

Removed 2026-10-22 (mixture framing; docs/probreg/cleanup_plan.md).

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
| **M1** | **the arbiter** | ONE k for the dataset | cheap | `all_rung_fit_loss` (`automl_package/models/flexnn/strategies/n_classes.py:226`) — see ⚠️ below, NOT `held_out_arbiter_advantage` |
| **M2** | **the distillation** | a k **per input** | cheap | `fit_router` + routed predict (`automl_package/models/probabilistic_regression.py:754`) |
| **M3** | **the sweep selector** | ONE k for the dataset, by training a **separate ORDINARY model per k** (no k-dropout, `NClassesSelectionMethod.NONE`) and scoring each on held-out data | **expensive — the reference** | generalise `select_k_for_toy` (`automl_package/examples/report_a_benchmark.py:331`), which already builds fixed-k models via `_probreg_fixed` (`:185-191`) |

⚠️ **M1's mechanism corrected 2026-07-20 (repair pass).** `held_out_arbiter_advantage`
(`automl_package/models/probabilistic_regression.py:845-916`) was named here originally and is
**the wrong shape for M1's job** — it returns a per-input `(N,)` array comparing ONLY the top rung
against the k=1 bypass (`:905-907`), and cannot express "chose k=4" for any middle rung, let alone
one global answer for the whole dataset. The primitive that CAN — a full `(batch, n_classes)`
per-rung held-out fit-term table, exactly what a cheapest-within-tolerance selector needs to run
over — is `all_rung_fit_loss`
(`automl_package/models/flexnn/strategies/n_classes.py:226`). PA
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

Order is **`docs/flexnn/package/plan.md` FP-12 → P0 → P0-b1 → P1 → P2 → PA → P7 → PB ∥ PC ∥ P8 →
P3 → P4 → P5 → P9 → P6** *(P10 removed 2026-07-21 — merged into FP-12)* *(amended
2026-07-21: P7, the Decision-20 schedule migration, precedes every task that trains M1/M2; P8, the
Decision-21 regularisation check, runs parallel and gates P4's read; P0-b1 and P1 were found
already landed — see their headers)*.

⚠️ **ORDER LINE CORRECTED 2026-07-21 (root, after an execution-level audit). P9 and P10 both had
full specs — write set, deps, verify line — and NEITHER appeared in this order line.** A
dispatcher following the order literally would never have scheduled either. *(Recorded as case law: a
task is not scheduled because it is well-specified; it is scheduled because it is in the order. Both
are required, and this file had drifted so that adding a task no longer meant adding it here.)*

**One thing this linear line cannot express — read it before partitioning waves:**
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
24. Removed 2026-10-22 (mixture framing; docs/probreg/cleanup_plan.md).

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

26. Removed 2026-10-22 (mixture framing; docs/probreg/cleanup_plan.md).

27. **ProbReg head-layout arm list (user, 2026-07-21).** Separate per-class heads are the model of
    record (`docs/probreg/PLAN.md` §1). The single-head-with-per-class-outputs layout is **retained as a
    LABELLED comparison arm, never a default** — it distinguishes "per-class heads help" from
    "*independent* per-class heads help". The single-head-final-output layout stays **blocked under
    `NESTED`**, because it produces no per-class heads at all.


<!-- from docs/plans/capacity_programme/MASTER.md lines 448-522 -->
29. **Under the nested ladder, nothing may choose or shape k DURING TRAINING (user, 2026-07-21).
    Amends Decision 13 from "demoted to a labelled comparison arm" to "unreachable by default".**

    **⚠️ PROGRAMME-WIDE — this binds ProbReg AND FlexNN (width/depth), user 2026-07-21.** It is one
    principle about *when* capacity is chosen, not a fact about one dial. Inventory verified on disk
    before writing:

    | dial | in-training machinery this retires | where |
    |---|---|---|
    | **ProbReg (k)** | `SOFT_GATING`, `GUMBEL_SOFTMAX`, `STE`, `REINFORCE` · `NClassesRegularization`'s `K_PENALTY` and `ELBO` (the enum's own docstring: *"only meaningful with an in-training selection method"*) | `automl_package/enums.py` · `automl_package/models/flexnn/strategies/n_classes.py` |
    | **FlexNN depth** | `GumbelSoftmaxStrategy`, `SoftGatingStrategy`, `SteStrategy`, `ReinforceStrategy` · `DepthRegularization`'s `DEPTH_PENALTY`, `ELBO`, `COST_AWARE_ELBO` | `automl_package/models/flexnn/strategies/layer.py:52,98,144,278` · `automl_package/enums.py:96` |
    | **FlexNN width** | **nothing left to retire** — `WidthSelectionMethod` was removed entirely by FP-3 and width already selects only through the distilled router. Recorded so nobody hunts for a width equivalent that does not exist. | — |

    **Survivors in every family: `NONE` (fixed capacity) and `NESTED` (the ladder).** Depth's evidence
    points the same way as ProbReg's — the FlexNN ELBO depth-selection claim was **refuted** (M0,
    post-fix ELBO collapsed to depth=1 on all five seeds), which is the depth-side instance of exactly
    the failure Decision 13 recorded for width.

    **Why one rule and not two patches.** Two separate problems were found in one session — the
    head layout with no per-class heads and in-training k-selection — and **both share a shape: the plan assumed a configuration, nothing enforced
    it, and `get_hyperparameter_search_space` could select the unsafe one.** A tuning run could reach
    either of them. One enforcement point closes both.

    **Justification.** *In-training selection* is retired because **it does not work** — Decision 13's finding
    (failed for width; the FlexNN ELBO depth-selection claim refuted), which is why the arbiter and
    the distilled router exist at all.

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
      widened grid (Decision 25).
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

- **fixed_k** — scripts: none; results: `LADDER/P2/`, `report_a_results/`
- **global_k** — scripts: `probreg_p8.py`, `report_a_benchmark.py`; results: `LADDER/P8/`
- **history** — scripts: `_kselection_metrics.py`, `_phase2_followups_chain.sh`, `_variational_em.py`, `_variational_em_perinput.py`, `capacity_ladder_v2.py`, `capacity_ladder_v3.py`, `capacity_ladder_variance_v0.py`, `capacity_ladder_variance_v1.py`, `capacity_ladder_x2.py`, `classifier_symmetry_check.py`, `dynamic_k_comparison.py`, `gumbel_elbo_retest.py`, `head_degeneracy_diagnostic.py`, `head_structure_diagnostic.py`, `middle_class_centering.py`, `probabilistic_regression_showcase.py`, `probreg_ablation.py`, `probreg_elbo_prior_check.py`, `probreg_k10_sweep.py`, `probreg_k20_sweep.py`, `probreg_k_sweep.py`, `probreg_kselection_comparison.py`, `probreg_kselection_diagnostic.py`, `probreg_kselection_experiments.py`, `probreg_kselection_prior_ablation.py`, `probreg_mixture_eval.py`, `probreg_ordering_ablation.py`, `probreg_snr_sweep.py`, `probreg_variational_em_step1.py`, `probreg_variational_em_step2_perinput_arbiter.py`, `probreg_variational_em_step3_perinput_model.py`, `probreg_variational_em_toy_e_hump.py`, `run_phase2_followups_safe.sh`, `run_probreg_k10_sweep_safe.sh`, `run_probreg_k_sweep_safe.sh`, `sep_heads_vs_single_final.py`; results: `LADDER/V0/`, `LADDER/V1/`, `LADDER/V2/`, `LADDER/V3/`, `LADDER/X2/`, `classifier_symmetry_results/`, `gumbel_elbo_retest_results/`, `head_degeneracy_results/`, `head_structure_results/`, `middle_class_centering_results/`, `prior_ablation_results/`, `probreg_ablation_results/`, `probreg_elbo_prior_check_results/`, `probreg_k10_sweep_results/`, `probreg_k20_sweep_results/`, `probreg_k_sweep_results/`, `probreg_kselection_diagnostic_results/`, `probreg_kselection_experiments_results/`, `probreg_ordering_ablation_results/`, `probreg_snr_sweep_results/`, `sep_heads_investigation/`, `variational_em_comparison_results/`, `variational_em_step1_results/`, `variational_em_step2_results/`, `variational_em_step3_results/`, `variational_em_toy_e_results/`
- **per_input_k** — scripts: `_capacity_ladder.py`, `_capacity_ladder_nested.py`, `capacity_ladder_h1.py`, `capacity_ladder_k1k2k3.py`, `capacity_ladder_k4.py`, `capacity_ladder_k5.py`, `capacity_ladder_k6.py`, `capacity_ladder_s1.py`, `capacity_ladder_s2.py`, `capacity_ladder_t2.py`, `capacity_ladder_t3.py`, `capacity_ladder_x4.py`, `capacity_ladder_x4b.py`; results: `LADDER/H1/`, `LADDER/K0/`, `LADDER/K1K2K3/`, `LADDER/K4/`, `LADDER/K5/`, `LADDER/K6/`, `LADDER/S1/`, `LADDER/S2/`, `LADDER/T2/`, `LADDER/T3/`, `LADDER/X4/`, `LADDER/X4b/`
- **shared** — scripts: `_toy_datasets.py`
