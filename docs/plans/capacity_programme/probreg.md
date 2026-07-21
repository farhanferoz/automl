# Strand: ProbReg — per-input resolution (k) selection

**Owns the whole ProbReg workstream**: model definitions, the defects, the comparison battery
(toys + real data + baselines), and the report. Read `MASTER.md` + this file — that is the whole
context. Nothing about ProbReg is decided anywhere else; if another document disagrees with this
one, **this one wins and the other is a bug to fix**.

**Why this file exists (2026-07-20).** ProbReg content was scattered across 15 files in 5 plan
directories with no owning strand, and the model definitions disagreed in three places at once
(§1). That scatter produced wrong answers to the user twice in one session. Consolidating here is
the fix; the rule that keeps it fixed is the first line of this file.

---

## 0.5 CODE ORGANISATION AND CLEANUP — the SAME standard as `width.md` §3.9 (user, 2026-07-21)

**User instruction:** the classifier work gets the same organisation and cleanup standard as width.
**It does not have it today, and this strand is messier than width was.** Inventory read off disk
2026-07-21, not recalled:

- **24 driver scripts under `automl_package/examples/` matching `probreg*`/`classifier*`**, with
  heavily overlapping names: `probreg_k_sweep.py`, `probreg_k10_sweep.py`, `probreg_k20_sweep.py`,
  `probreg_kselection_comparison.py`, `probreg_kselection_diagnostic.py`,
  `probreg_kselection_experiments.py`, `probreg_kselection_prior_ablation.py`,
  `probreg_elbo_prior_check.py`, `probreg_ordering_ablation.py`, `probreg_ablation.py`,
  `probreg_identifiability_sweep.py`, `probreg_snr_sweep.py`, `probreg_mixture_eval.py`,
  `probreg_variational_em_step1.py` / `_step2_perinput_arbiter.py` / `_step3_perinput_model.py`,
  `probreg_variational_em_toy_e_hump.py`, `probreg_p8.py`, `_kselection_metrics.py`. **Nobody can say
  from the names which of these is current.**
- **~10 results directories in the WRONG PLACE**, as `automl_package/examples/<name>_results/`
  siblings of the drivers (`probreg_k_sweep_results`, `probreg_k10_sweep_results`,
  `probreg_k20_sweep_results`, `probreg_ablation_results`, `probreg_snr_sweep_results`,
  `probreg_ordering_ablation_results`, `probreg_identifiability_results`,
  `probreg_kselection_diagnostic_results`, `probreg_kselection_experiments_results`,
  `probreg_elbo_prior_check_results`, `classifier_symmetry_results`). **MASTER's artifact-naming rule
  says results live under `automl_package/examples/capacity_ladder_results/<TASKID>/`.** These predate
  the rule; they are not exempt from it.
- **Two shell launchers hardcode absolute `/home/ff235/...` paths** on a public repo:
  `automl_package/examples/run_probreg_k_sweep_safe.sh`,
  `automl_package/examples/run_probreg_k10_sweep_safe.sh`.
- **Package files split across three locations**: `automl_package/models/probabilistic_regression.py`,
  `automl_package/models/architectures/probabilistic_regression_net.py`,
  `automl_package/models/selection_strategies/n_classes_strategies.py`.

⚠️ **INVENTORY RE-DERIVED OFF DISK 2026-07-21 (root, user-instructed). The counts above are
UNDERSTATED — P9 must be scoped from these numbers, not those.** Every figure below was produced with
the same pattern the bullet above claims to use, so the difference is real, not a definitional
artefact:

| item | plan says | actually on disk | note |
|---|---|---|---|
| drivers matching `probreg*`/`classifier*` | 24 | **31** | the named list is a sample, not the set |
| misplaced `*_results` dirs under `examples/` | "~10" | **35** (excluding the sanctioned `capacity_ladder_results/`) | of which **15** are ProbReg-owned (`probreg*`, `classifier*`, `classreg*`, `head_*`); the remaining 20 belong to other strands or are legacy, and P9 **does not** own them |
| shell launchers hardcoding `/home/ff235` | 2 | **4** | the two named, plus `run_phase2_followups_safe.sh` and `_phase2_followups_chain.sh` |
| package locations | 3 | **2** | ✅ improved by FP-11: `n_classes_strategies.py` moved to `automl_package/models/flexnn/strategies/n_classes.py`, so only the two ProbReg-proper files remain — and they stay, by the ruling below |

**Consequences for P9, all binding:**
1. **P9's write set must be re-derived at execution time, not copied from the bullet list above.** A
   task scoped to "~10" directories that meets 15 will either stop early or improvise — both bad.
2. **P9 owns the 15 ProbReg-owned directories ONLY.** The other 20 are out of scope; moving another
   strand's artifacts is a write-set violation, and several belong to parked or legacy programmes.
3. **The hardcoded-path fix covers all 4 scripts**, not 2. *(This is a public repo; the paths are a
   portability defect, not a tidiness one.)*
4. **The "three locations" bullet is now stale in the good direction** — do not "fix" it by moving
   the two remaining files. §0.5's ruling below is explicit that ProbReg's model files STAY.

### What moves, and what deliberately does NOT (settled 2026-07-21)

**Only the capacity MECHANISM joins the FlexNN home.** `flexnn-package.md` FP-11 (TASK ZERO) moves
`n_classes_strategies.py` to `automl_package/models/flexnn/strategies/n_classes.py`, so that every
capacity-selection strategy — width, depth, k — sits together. **`probabilistic_regression.py` and
`probabilistic_regression_net.py` STAY where they are**, because ProbReg is a MODEL that *uses* a
capacity dial, not a capacity mechanism; it belongs beside `classifier_regression.py`, its sibling.
FlexNN is the umbrella for the mechanisms, not for every model that consumes one. **P7 therefore deps
on FP-11** (it writes the file FP-11 moves) — one move, not two, which is the same argument that made
FP-11 task zero.

### The rules (identical to `width.md` §3.9 / WSEL-17 — stated here so this strand is self-contained)

- **REUSE FIRST.** Before writing any driver, class or training loop, check what exists. A near-copy
  is a defect. A task writing something new states what it checked.
- **ONE home per lifecycle stage.** Certified mechanisms in the package; candidates under test in ONE
  module; drivers in `examples/`; **results ONLY under
  `automl_package/examples/capacity_ladder_results/<TASKID>/`**, hyphen-free task id.
- **Promotion is a task, never a side effect** — a candidate reaches the package only via a task whose
  verify reproduces reference numbers.
- **Deletion eligibility is FOUR MECHANICAL CHECKS, never a judgement** — see P9.

### P9 — THE CLASSIFIER CLEANUP: consolidate the drivers, relocate the results, delete what is superseded

**Files (write set):** `automl_package/examples/probreg_*.py` · `automl_package/examples/*_results/` ·
`automl_package/examples/run_probreg_*.sh` ·
`docs/plans/capacity_programme/shared/p9-cleanup-manifest.tsv` (Create)

**Spec (execution-level).**
- [ ] **Step 1 — one row per artifact.** Every `probreg*`/`classifier*` driver, every `*_results/`
  directory, both shell launchers. Columns: path · verdict (KEEP / RELOCATE / SHIM / DELETE) · reason ·
  replacement path · the four checks' outcomes.
- [ ] **Step 2 — DELETE-ELIGIBLE only if ALL FOUR hold**, each checked by a recorded command:
  (1) not in `docs/plans/capacity_programme/shared/PROTECTED.tsv`; (2) no live plan line cites it
  (`grep` the plan dir at execution time); (3) nothing under `tests/` and no surviving module imports
  it; (4) superseded by a NAMED replacement that EXISTS on disk, path written in the row.
  **"Looks old" is not a reason.** Everything else is KEEP, RELOCATE or SHIM.
- [ ] **Step 3 — gate evidence is KEEP BY RULE.** Any JSON backing a delivered report
  (`docs/reports/probreg_toys/report_a_probreg_toys.md`) or cited by a ledger marker stays. Deleting
  it would break the paper trail that makes the claim citable.
- [ ] **Step 4 — RELOCATE the misplaced results dirs** under
  `automl_package/examples/capacity_ladder_results/<TASKID>/`, one task id per study, and update every
  driver's output path. A relocation is a `git mv` plus a path constant — never a copy.
- [ ] **Step 5 — genericize the two shell launchers**: replace the hardcoded `/home/ff235/...` with a
  repo-root-relative path derived at run time. This is a public repo.
- [ ] **Step 6 — write the manifest and STOP.** `shared/p9-cleanup-manifest.tsv` goes to the USER.
- [ ] **Step 7 — attended deletion** after sign-off: ONE commit whose body is the manifest. `git`
  history retains every file, so it is recoverable; say so in the commit body.

**Non-goals:** no deletion of anything failing any one check; no deletion of report evidence, ever; no
behaviour change to any driver that a live task will run; no merging of drivers that are not proven
equivalent; **no touching `n_classes_strategies.py`** (FP-11 moves it, P7 rewrites it).
*Orchestration:* parallel: **NO — and this is a hard single-writer constraint, not a preference** ·
deps: **FP-11** and **P7 merged** (P7 rewrites the training schedule, which is what supersedes several
of these drivers) **· AND P3, P4, P8, PB, PC, P11 ALL CLOSED** · tier: sonnet high for Steps 1-6,
**root + ATTENDED for Step 7** · scale: static · shape: execution ·

⚠️ **P9 COLLIDES WITH SIX OPEN TASKS BY CONSTRUCTION (added 2026-07-21 after an execution-level
audit).** P9's write set is the glob `automl_package/examples/probreg_*.py` — which **matches every
driver the open tasks create**: `probreg_p3.py` (P3), `probreg_benchmark.py` (P4), `probreg_p8.py`
(P8), `probreg_pb.py` (PB), `probreg_pc.py` (PC), `probreg_p11.py` (P11). Nothing in the previous
deps line prevented a wave-partitioner from scheduling P9 **concurrently** with any of them, and a
relocate or `git mv` racing another task mid-write is the exact single-writer failure this programme
has a hook to catch. P9's own eligibility checks would probably block a *deletion* of a live-cited
file — but they do not protect a **rename**. **⇒ P9 runs LAST in this strand, after all six close.**
verify: (1) `shared/p9-cleanup-manifest.tsv` exists and every DELETE row records all four checks;
(2) every relocated study's JSONs resolve at their new path and no driver writes outside
`capacity_ladder_results/`; (3) neither shell launcher contains the string `/home/`;
(4) `AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m pytest tests/ -q` matches the pre-cleanup result
exactly — no new failures, no newly-passing tests; (5) no `PROTECTED.tsv` path deleted or renamed.

### THE CANONICAL TOY SUITE — the toys EXIST and are named; what was missing is the ASSIGNMENT

*(Written 2026-07-21. The first draft of this section said the strand "has no canonical toy suite" and
deferred it to the user. **That was wrong and the user corrected it: the toys had been run, they just
needed finding and noting.** They live in ONE module,
`automl_package/examples/_toy_datasets.py`, imported by ~35 drivers, each carrying its ROLE in its own
docstring. Roles below are quoted from those docstrings, not inferred.)*

**This suite is RICHER than width's** and the difference matters: these toys carry **per-input ground
truth** `k*(x)`, and three of them are **matched positive/negative pairs** — a bimodal toy and a
single-mode twin matched in mean and variance — so a selector that keys on spread rather than on
mixture structure is caught by construction. Width has no equivalent.

**Fixed on every cell of every tier:** seeds **0, 1, 2** · the strand's convergence gate with
`hit_cap: false` required · `OMP_NUM_THREADS=4` pinned · **selection on the per-sample fixed-σ
mixture log-likelihood** on the selection split, with **RMSE reported alongside as a point-accuracy
column, NEVER as the k readout**.

⚠️ **CORRECTED 2026-07-21 (root, at the user's instruction to audit this strand for variance and
regularisation problems). The previous text said "selection on the point-prediction metric (squared
error), never Gaussian NLL" — that was WRONG in both halves and it is the most dangerous kind of
wrong, because a battery driven to green on it would have produced a clean-looking false finding.**

- **The squared-error half inverted the ratified rule.** `docs/probreg_benchmark/benchmark_spec.md`
  §4.3 requires all three arms to score rungs on the fixed-σ mixture log-likelihood, and §4.2 keeps
  RMSE as a reporting column *"never the k readout"*. §4.3 states the consequence of getting this
  wrong in terms this strand cannot afford to ignore: *"Selecting k on squared error would be
  selecting on a quantity that is structurally blind to k for symmetric targets: the mixture mean is
  the same whether the model resolves one component or five. The selection curve would be flat, the
  cheapest-within-tolerance rule would return k=1 everywhere, and the result would look like a clean
  finding."* **⇒ Squared-error selection does not merely weaken this strand — it manufactures its
  headline conclusion out of a metric artefact.**
- **The "never Gaussian NLL" half over-banned, by conflating two different things.** What P8's
  reopening actually forbids is a likelihood read at a **LEARNED** `log_var` — that is variance
  fitting, and it is what the σ-scope decision removed. A likelihood at **FIXED** σ fits nothing: σ
  is one shared constant, so the metric is not a variance metric at all. **The test is not "is it a
  likelihood?" but "is σ learned?"**

**The rule, stated so it cannot be misread again:**

| | selection / k readout | reported alongside | status |
|---|---|---|---|
| fixed-σ mixture log-likelihood | ✅ **the readout, everywhere** | — | σ is a constant; nothing is fitted |
| RMSE / squared error | 🚫 **never** — structurally blind to k | ✅ honest point-accuracy column | a disagreement between the two is itself a finding |
| likelihood at a **learned** `log_var` | 🚫 **forbidden** | 🚫 | this is the violation that voided P8 |

**Tolerance rule is unaffected:** the curve stays in nats, so cheapest-within-tolerance at twice a
bootstrap standard error transfers **unchanged**, with its published figures still comparable
(spec §4.3).

- **TIER 1 — the reference cell.** `make_toy_b` (`automl_package/examples/_toy_datasets.py:49`) with
  `baseline="zero"`: "conditional Gaussian mixture with **KNOWN intrinsic k**", the mixture sitting
  still so the data are i.i.d. draws from one fixed mixture — the grounding regime where the pruning
  theory applies. **The only toy with unambiguous ground-truth k.** Every task that trains runs this.
- **TIER 2 — the three negative controls.** Required by any task claiming **the dial works**. Each is
  a trap for a different wrong mechanism:
  - `make_toy_a` (`:35`) — "smooth unimodal with controlled homoscedastic noise", the "no intrinsic k"
    baseline. Honest answer: the smallest k. Catches a selector that raises k on noise.
  - `make_broad_unimodal` (`:97`) — "Single Gaussian matched in mean AND variance to the k_true=2
    bimodal toy", so a moment-matching model "cannot tell them apart; the genuine mixture objective
    can". Catches a selector reading moments instead of structure.
  - `make_toy_c_broad` (`:241`) — "Single-mode twin of make_toy_c, variance-matched at every input
    (over-chopping trap)"; `k*(x) = 1` everywhere while the spread widens with x. "A tiling model is
    tempted to raise its bucket count here as the spread grows; the honest held-out arbiter must give
    those buckets no credit."
- **TIER 3 — the per-input structure set.** Required by any task claiming **per-input** behaviour:
  - `make_toy_c` (`:211`) — "per-input mode structure"; all the x-dependence lives in the SHAPE, not
    the location; `k*(x) = 2` above a spacing threshold, monotone in x.
  - `make_toy_e` (`:276`) — "the x-confound breaker": spacing humps, so the count must rise and then
    **fall again while x increases monotonically** — "which a selector that merely tracks x (or the
    marginal variance, which also humps) cannot fake."
  - `make_toy_d` (`:439`) — "a staircase in component count — the count-beyond-binary / ceiling test";
    `k*(x) in {1, 2, 3}`. Tests that reading generalises past 1-vs-2 **and STOPS at 3 rather than
    tiling out to k_max**.
- **TIER 4 — the generality ladder.** Required only by the headline comparison and the report: the
  separation/sigma sweep over tiers 1-3 (the resolvability dial named in `make_toy_b`'s docstring),
  plus `make_toy_d_ndim` (`:512`) for the dimensionality check.

**Which task runs which tier — a MECHANICAL rule, not a judgement:**
- Every task that trains runs **tier 1**.
- **+ tier 2** if it claims the dial works, or reports a chosen k as correct.
- **+ tier 3** if it makes any **per-input** claim (the distilled router, the per-input arm).
- **+ tier 4** only for the headline comparison and the report.
- **P7** (schedule migration) = tiers 1 + 2 — it re-validates that middle-k rungs are no longer
  starved, which is a claim about the dial. **PB / PC** (data need, router architecture) = tiers 1 + 3,
  both per-input. **P3** (the cheap-vs-expensive comparison) = tiers 1 + 2 + 3. **P6** (report) = all
  four.
- **A deviating cell carries a written justification IN THE TASK**, and may not be tabulated beside a
  canonical one unless the deviation is named in the same table.
- **The four legacy `make_datasets()` toys** (`:10` — heteroscedastic, bimodal, piecewise, exponential)
  are the OLD set, from before the role-labelled toys existed. **They are not a tier.** Results on them
  stay citable as history, labelled as the legacy set; **no NEW cell may be produced on them.**
- **The `make_v_toy*` family** (`:332`, `:369`, `:394`) belongs to the PARKED variance programme. Not a
  tier here; do not run.

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
- **`flexnn-package.md` FP-12 now covers training as well as scoring**, which overlaps **P7**'s
  rewrite of the training objective. Sequence deliberately or merge them; do not dispatch both.

⚠️ **HEAD LAYOUT PINNED — all three models use `RegressionStrategy.SEPARATE_HEADS` (root, 2026-07-21,
at the user's instruction to correct this omission).** The plan previously never named a head layout
anywhere (`grep -in 'regression_strategy\|separate_heads' probreg.md` → **zero hits** before this
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
than deliberately omitted.** Closing it is **`flexnn-package.md` FP-12** *(originally task P10, merged into FP-12 on 2026-07-21 — one guard for every illegal-under-`NESTED` configuration)*.

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
(`docs/plans/width_dial_2026-07-11/EXECUTION_PLAN.md:119-129`, fixed by the W2 sandwich); (iii) this
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
(`docs/reports/probreg_kselection/probreg_kselection.md` §3.2), and the current per-k sweep code
already builds them that way — `_probreg_fixed` sets `NClassesSelectionMethod.NONE`
(`automl_package/examples/report_a_benchmark.py:185-191`). The plan was wrong; the code was right.
**`docs/probreg_benchmark/benchmark_spec.md` §2.0 and §2.3 still carry the superseded wording and
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
in `docs/reports/probreg_kselection/probreg_kselection.md` §3.2). M2's distilled router does not
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

1. **`MASTER.md`'s naming key** defined variable-k as *"dynamic-k (ELBO + SoftGating)"* —
   **in-training** selection, which `MASTER.md` Decision 13 itself demotes to a labelled
   comparison arm. Self-contradictory. **CORRECTED 2026-07-20**: that entry now points here and
   states no definition of its own. *(No line number is cited on purpose — a line reference into a
   file this one is actively rewriting is a cache entry that rots within the hour. It did: the
   first draft of this bullet cited `:38-39`, which pointed at an unrelated entry ten minutes
   later.)*
2. **`docs/probreg_benchmark/benchmark_spec.md:66`** defines Model 1 with
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

## 2. State — what is established, and what is not

**Established (toys only).** The k-dropout coherence check — one k-dropout model read at each
fixed k, against a separately trained model dedicated to that k (i.e. an M1-vs-M3 *quality* check)
— has been run on 3 toy problems × 3 repeats. Write-up and numbers:
`docs/reports/probreg_kselection/probreg_kselection.md` §3.2. Summary of what it shows: the
largest checked k **never** fails the coherence check in any of the 9 cases, but some middle k
fails in **8 of the 9** — a small, real, and currently **unexplained** cost concentrated in the
middle of the range.

**NOT established — and each is a task below:**
- **(b) same choice.** Nothing anywhere compares the arbiter's chosen k against the sweep's best
  k. §3.2 compares *fit quality at each k*; that is a different claim. → **P3**
- **Anything on real data.** All of the above is synthetic. → **P4**
- **Baselines.** No comparison against XGBoost / LightGBM / CatBoost / a standard NN has been
  run for this strand. → **P4**
- **Why the middle-k cost exists.** 8 of 9 is a pattern, not noise, and it is undiagnosed. → **P5**

**Report status.** `docs/reports/probreg_kselection/probreg_kselection.md` exists (880 lines, PDF
built) and covers the toy work. It predates §1 and therefore uses the older framing; it must not be
extended until §1 is reflected in the code and the battery. → **P6**

---

## 2.5 Where the history lives, and which records can be trusted

Surveyed 2026-07-20 (13 files, all read in full, artifact paths and a sample of line citations
checked against disk). Two disconnected generations of ProbReg planning existed, which never
referenced each other — that disconnection is what this strand ends.

**⚠ GROUND-TRUTH RULE, learned from the survey: for the first generation, only
`automl_package/examples/capacity_ladder_results/RESULTS.md` records what actually completed.** No
inline status glyph in those plan files is trustworthy — several tasks carry adjudicated GO verdicts
in the results ledger while their own plan file still shows them open, and one file's status table
contradicts its own prose from the same day. All four are now banner-frozen in place (not moved —
moving them would have broken ~30 citations including ten frozen preregistration records).

**Still-live reasoning in the frozen files — read, never dispatch:**
- `docs/plans/width_dial_2026-07-11/cascade_execution_plan_2026-07-11.md` §3.1 — the cross-k
  class-identity conflict (one component asked to be correct at several resolutions at once) **and
  its resolution**: `optimization_strategy=REGRESSION_ONLY` makes the cross-entropy branch never
  fire. Now baked into the source (`automl_package/models/probabilistic_regression.py:53-60`,
  `:531-543`). **This is why §1 freezes `REGRESSION_ONLY` for all three models** — it is not a
  stylistic choice, it is the thing that keeps the k ladder coherent.
- The same file's §4.7 — four ProbReg library fixes, **all verified landed at HEAD 2026-07-20**,
  none of which had a completion record in any plan document until that check.
- `docs/plans/capacity_programme/shared/hetero_nll_diagnosis.md` — the diagnosed, accepted
  hetero-NLL failure (§3 D3).

**Known citation rot (do not build on these):** three 2026-07-11 documents cite
`automl_package/models/probabilistic_regression.py:796-798` for a classifier-probability bug. That
file has grown ~200 lines since; the function moved to roughly `:1010` and the bug is already fixed
there. Re-verify any line citation from that generation before using it.

## 3. Known defects

**D1 — FIXED + VERIFIED 2026-07-20.** `fit_router` scored raw-space `y_val` against
`forward_at_k`'s symlog-space outputs — silently wrong error table ⇒ silently wrong selector, worst
on exactly the heavy-tailed data this strand targets. Fix at
`automl_package/models/probabilistic_regression.py:813`; contract documented in the docstring.
Verified in BOTH directions: fix present → 3 passed; fix deleted → 3 failed
(`tests/test_phase3_dynamic_k.py`, class `TestFitRouterSymlogSpaceAlignment`). Detail + the
measured blindness table: `docs/plans/capacity_programme/flexnn-core.md` (F9-fix-b block).

**D2 — ✅ FIXED 2026-07-20 (commit `84ad94d`), found already landed during the 2026-07-21 repair
audit.** The units mismatch in `_per_sample_log_likelihood_at_k`
(`automl_package/models/probabilistic_regression.py:835`) is fixed in that commit: caller-space `y`
is symlog-transformed before scoring, the contract is documented on the method and on its only
caller `held_out_arbiter_advantage`, and the regression test P1 asked for exists in the same commit
(`tests/test_phase3_dynamic_k.py:726`, `TestArbiterAdvantageSymlogSpaceAlignment`) — re-executed at
the root 2026-07-21: 2 passed.
*(Corrected 2026-07-21: the 2026-07-20 repair pass re-asserted "D2 still stands as a real bug" —
prose was edited without re-checking disk while the fix already sat in history. The rule this
motivates is now a `MASTER.md` Corrections entry. What SURVIVES of the earlier note, verified still
true 2026-07-21: **the primitive M1 actually needs, `all_rung_log_likelihood`
(`automl_package/models/flexnn/strategies/n_classes.py:230`), has the SAME shape of
bug** — it passes `y_target` straight through with no symlog transform (`:251`), and has zero
callers today. Whoever builds PA's M1 selector on top of it must apply the now-fixed transform
pattern there too, or D2 reopens in the function that actually matters. It is PA's problem to close
as part of building the mechanism.)*

**D4 — OPEN, inherited, user-gated.** The cross-k class-identity conflict is only *avoided*, not
fixed: it is dodged by freezing `REGRESSION_ONLY` (§2.5). For any configuration that DOES activate
cross-entropy, the defect is live, and what shipped is a runtime warning only — the guard is at
`automl_package/models/probabilistic_regression.py:535` and the `logger.warning` it fires at `:536`
— not a behaviour change. The two candidate
fixes on record are k-stable binning, or documenting that k-dropout requires likelihood-only
training. **Not in scope for this strand's battery** (which is `REGRESSION_ONLY` throughout) but it
must not be re-discovered a fourth time — it is recorded here as the owner.

**D3 — two failing tests, both pre-existing, neither caused by current work.**
- `test_probabilistic_nll_beats_constant_on_heteroscedastic` — root-caused and deliberately
  accepted: `docs/plans/capacity_programme/shared/hetero_nll_diagnosis.md`.
- `test_prob_regression_heteroscedastic_mse` — error 2.8812 against a 2.5 bar. Verified identical
  to 16 digits on a clean worktree at commit `600460c`, so it is not new. **Undocumented anywhere**
  — the same silent-failure pattern this programme keeps paying for. → **P2**

**D5 — OPEN, the primary defect this repair pass exists to record.** `held_out_arbiter_advantage`
(`automl_package/models/probabilistic_regression.py:845-916`) is **not M1's selection mechanism**,
despite being named as such in an earlier version of §1's table. Verified by reading the full
function body: it returns a per-input `(N,)` array (`:914-916`), computed as the
neighbour-averaged advantage of ONE fixed top rung over the k=1 bypass (`:905-907`) — a binary
top-vs-bypass comparison, smoothed over an x-neighbourhood. It cannot express "chose k=4" for any
middle rung, and it answers a per-input question, not "one k for the dataset" (§1's M1
definition). → **PA must build M1's selector on `all_rung_log_likelihood` instead** (§1 ⚠️, PA's
task text).

**D6 — OPEN, a stale docstring; recorded here, not fixed (no source edits in this repair).**
`all_rung_log_likelihood`'s docstring
(`automl_package/models/flexnn/strategies/n_classes.py:234-235`) claims *"This is
the all-rung score table `ProbabilisticRegressionModel.held_out_arbiter_advantage` ...
consume[s]"* — **false**. `held_out_arbiter_advantage` never calls `all_rung_log_likelihood`; it
calls `_per_sample_log_likelihood_at_k` twice, once per rung
(`probabilistic_regression.py:905-906`). Confirmed by reading both functions in full. Whichever
task next touches `n_classes_strategies.py` (PA, most likely) should correct the docstring in the
same change that gives `all_rung_log_likelihood` its first real caller.

---

## 3.5 Autonomous execution contract

**This strand is scoped to run unattended (user, 2026-07-20).** The root is dispatcher + verifier.
Every foreseeable branch below has a **pre-authorised default**; take it, log it, keep going. Only
the four HALT conditions stop the run.

**Rule: never block on a question that has a reversible default.** Log every default taken to
`RESUME.md` `### Decisions` with the evidence that triggered it, and batch anything genuinely
user-only for the end of the run.

| Branch | Pre-authorised default | Log |
|---|---|---|
| **PB**: which selection fraction becomes the frozen default | the **smallest** fraction at which every arm is within its own noise band of its best (same twice-standard-error rule as everywhere else); if no fraction saturates, take the largest swept and record the study as **inconclusive, floor not found** | fraction + the curve |
| **PB**: M2 still improving at the largest fraction | freeze the largest swept, and mark M2's battery result **"router data-limited"** in the report — a loss then does NOT support "per-input k does not pay" | the mark, prominently |
| **PC**: router conclusions invariant to architecture | keep the current frozen default; record invariance as a finding | table |
| **PC**: NOT invariant | adopt the **smallest** plateau configuration as THIS STRAND'S default, record it in `automl_package/examples/capacity_ladder_results/PC/frozen.json` under `new_default`, re-run PB at the new router. **This strand never writes `automl_package/models/common/distilled_router.py`** — any change to the SHARED default is landed by the root via `flexnn-package.md` FP-5 *(ruling 2026-07-21: three strands were each pre-authorised to "freeze globally" a file none may write — the classic false-parallel)* | old → new + why |
| **P2**: the undocumented failing test | if it is a stale bar, write the acceptance note; if it is a real regression, fix it. **Never loosen the bar to make it pass.** | either way, a note on disk |
| **ceiling binds** (selected k = k_max) | re-run that cell at `k_max = 20`; report the raise | which cells |
| **Kepler feature leakage** (`benchmark_spec.md` §14.3, was PARK-1) | drop the two leaky features, and pre-register that this dataset tests conditional mean/variance, **not** a density-level multimodality claim | the pre-registration, before the run |
| a spec section contradicts §1 | **§1 wins**; fix the spec in the same turn | the correction |

**HALT and ask — these four only:**
1. A **positive control fails** (MASTER Decision 14) — the protocol is then the defect, not the arms.
2. A study comes back **incoherent rather than merely negative** (e.g. PB's curve is non-monotone
   beyond noise) — that is a broken instrument, and running the battery on it wastes the budget.
3. Any change to **§1's model definitions**. They are the user's, not the run's.
4. Anything **irreversible or outward-facing** (deleting artifacts, publishing, **pushing to
   `origin`**). *(Amended 2026-07-21, user: COMMITTING per the `MASTER.md` branch protocol —
   wave-branch commits, local merge, branch delete, docs straight to `master` — is
   PRE-AUTHORIZED for the autonomous run and is no longer a HALT trigger. Pushing, publishing
   and deletion remain user-gated; `FP-8` stays attended-only.)*

## 3.6 Constants the studies FREEZE, and the battery READS

The studies are not background colour — they set the parameters the comparison runs at.

🚫 **DO NOT WRITE THE VALUES INTO THIS TABLE.** The plan holds the *name of the constant* and the
*path of the artifact that owns it*; the value is read from that artifact at build time. A number
copied here is a cache entry that rots, and this programme has already paid for that twice today (a
line reference that broke within ten minutes, and a diagnosis contradicted by its own sibling
document). The driver reads these from disk; a reviewer resolves them by opening the artifact.

| Constant | Set by | Owning artifact (single source of truth) |
|---|---|---|
| selection-set fraction | **PB** | `automl_package/examples/capacity_ladder_results/PB/frozen.json` |
| M2 data-limited flag, per dataset | **PB** | same file, one boolean per (toy, arm) |
| router hidden / depth / epochs / lr | **PC** | `automl_package/examples/capacity_ladder_results/PC/frozen.json`, else the current frozen defaults at `automl_package/models/flexnn/routing.py:57-60` if that file's `invariant` field is `true` |
| ~~labelling tolerance~~ | ~~PC~~ | STRUCK 2026-07-21 — Decision 18: the sweep is not scheduled; the flat 0.25 (`automl_package/models/flexnn/routing.py:57`) stays inherited-and-accepted |
| `k_max` after any ceiling raise | **P3/P4** | the per-cell result JSON that recorded the bind |

**Feed-forward rule (binding):** if P3 or P4 runs at a value not justified by the artifact named
here, its results are **not reportable**. The point of doing the studies first is that every number
in the final table has a measured reason for the setting it was produced under.

⚠️ **Anchor warning for every `verify:` line below.** Where a task reproduces a frozen number as its
positive control, that anchor must come from something **not computed by the method under test** — a
second implementation, an invariant, or a published figure. Re-deriving an anchor with the same code
does not verify the worker; it conscripts the worker into confirming our own bug and returns it
stamped *verified*. Concretely: P3's coherence half must NOT anchor solely on
`docs/reports/probreg_kselection/probreg_kselection.md` §3.2 re-run through the same harness — pair
it with the dedicated per-k models of M3, which are trained independently and are exactly the
non-shared implementation this rule asks for.

**Compilation note.** Once PA–PC are closed and every task below is decision-complete, this strand
is compilable to a deterministic Workflow (fan out on the sweeps, serialise the writes through the
root) rather than relying on dispatcher discipline — which is the shape the unattended run should
take.

## 4. Tasks

Order is **`flexnn-package.md` FP-12 → P0 → P0b → P0-b1 → P1 → P2 → PA → P7 → PB ∥ PC ∥ P8 →
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

### P0 — make the definitions single-sourced

**Files (write set):** `docs/probreg_benchmark/benchmark_spec.md` **only**
**🚫 NOT in the write set: `docs/plans/capacity_programme/MASTER.md`** — ROOT-ONLY (MASTER naming
key, "SHARED FILES ARE ROOT-ONLY"). Three sibling tasks (FP-0, WSEL-0, DSEL-0) need MASTER edits in
this same wave; concurrent writers produce contradictory text in one file. **Deliverable instead:**
emit the exact MASTER naming-key text verbatim in this task's report, for the root to apply.
**Spec:** Rewrite the `MASTER.md` naming key to point at §1 of this file instead of restating a
definition. Rewrite the benchmark spec's model section to §1's three models: all `NESTED`, M1 =
arbiter-selected global k, M2 = distilled per-input k, M3 = per-k sweep reference. Delete the
"differ in exactly one thing" claim about the old pair and re-state it truthfully for the new
arms. Record the three superseded definitions as corrections, not silent edits.
**Non-goals:** no code; no change to metrics, datasets, or the tuning protocol beyond what §1
forces.
**Status: ✅ DONE 2026-07-20, all three verify conditions executed and shown passing.**
`MASTER.md`: naming key delegates here and states no definition; strand index carries this file;
Corrections entry records the three-way contradiction and its organisational root cause.
`docs/probreg_benchmark/benchmark_spec.md`: §2 rewritten to the three-model set with a new §2.0
stating what each contrast isolates and what the old pair got wrong; M1 rebuilt on `NESTED`; M3
added; baselines renumbered 4–7; the false "differ in exactly one thing" cell corrected in place.

**Three consequences found while editing, each verified, each already folded into the spec:**
- **C3 is now RESOLVED, structurally.** The old arm could not select the bypass rung because that
  head is built only on the dynamic branch
  (`automl_package/models/architectures/probabilistic_regression_net.py:84`, `None` on the
  non-dynamic `else` at `:96`). All three arms now use `NESTED`, so all three reach rungs `1..10`.
  Standing condition added: M1's rung set must stay `1..10` or the defect returns.
- **C1 dissolved into a narrower, sharper condition.** No ProbReg arm tunes `n_classes` any more, so
  the Optuna space is identical across the three; what remains is the split each reads its k from.
  Binding: **all three select k on `cal` only** — an M1 reading `val ∪ cal` against an M2 reading
  `cal` would be rigged.
- **M3 must be EXEMPT from the matched wall-clock budget** (per-k-fit matched instead, total cost
  reported as a headline number). Matching it would starve the reference the efficiency claim is
  measured against.
*Orchestration:* parallel: no (single writer, doc set overlaps everything) · deps: none ·
tier: main loop (definitional) · scale: static · shape: design ·
verify: (i) `grep -n "probreg.md" docs/plans/capacity_programme/MASTER.md` shows the naming key
delegating here — **DONE**; (ii) the M1 code block in `docs/probreg_benchmark/benchmark_spec.md`
sets `NESTED`, not `NONE`; (iii) that spec defines three models, and `grep -c "arbiter"` on it is
no longer 0. *(Do NOT verify by grepping for the retired phrase — the Corrections entry quotes it
deliberately, so such a grep always fires. This line originally did exactly that and was
unrunnable.)*

### P0b — ✅ DONE 2026-07-20. σ-scope sweep completed at requirement level.

**Outcome differs from this task's original premise, and the difference is the point.** P0b was
written assuming σ removal meant squared-error scoring. The user corrected that: σ is **not fitted**
but scoring stays distributional with **one shared fixed σ** (`docs/probreg_benchmark/benchmark_spec.md`
§2, §4.1). The sweep was then executed on the corrected premise.

**Landed in the spec:** §2 scope block rewritten (fixed-σ mixture likelihood, the σ-value rule, the
required ×0.5/×2 ranking-stability check, and the recorded reasoning error); §2.0a naming map added
so remaining older prose cannot be misread; §4 rewritten to fixed-σ mixture NLL as primary with RMSE
retained as a point-accuracy column and a note on why RMSE can never be the k readout; §4.3 selection
score is the same likelihood at fixed σ, so the published tolerance rule transfers **unchanged**;
§6.5 search spaces collapsed to one identical ProbReg space plus the two surviving baselines, with
M3's single-search rule; §7 outcomes restated on the new arm set, thresholds carried over unchanged
because the metric family and units are unchanged; **new outcome H6** (does the cheap read match the
expensive sweep) marked NEW rather than folded into an old one; §1.1/§1.2/§8.2 build requirements for
removed metrics and dropped models struck through rather than deleted.

**Verified:** `grep` for live requirements naming a removed metric or a dropped model returns only
struck-through or explicitly-superseded text. Plan gates 6/6.

**Remaining (cosmetic, non-blocking):** ~41 occurrences of the old "shared-k"/"variable-k" wording
survive in prose. §2.0a maps them binding-ly, so the document is safe to build from; a prose rename
is a mechanical follow-up. **Do not attempt it with a blind find-and-replace** — §2.0a names the one
class of passage where the old term is superseded rather than renamed.

*Original task text follows, retained because its premise being wrong is itself the case law.*

### ~~P0b — finish the σ-scope-change sweep through the spec~~ (original, premise superseded)

**Why.** The user removed variance fitting from scope (variance is separate work). §2 of
`docs/probreg_benchmark/benchmark_spec.md` is updated and the consequences are stated there, but the
change reaches further into that document than one section, and a half-swept spec is worse than an
unswept one — the driver would implement whichever half it read first.

**Known stale regions, all in `docs/probreg_benchmark/benchmark_spec.md`:**
- **§4 (metrics) — wholly stale.** NLL, CRPS, PIT, calibration/ECE, PICP, Winkler all assume σ.
  Replace with a point-prediction set (primary: squared error; per-dataset RMSE; the existing
  matched-split and seed machinery is unaffected).
- **§6.5 search spaces** still list XGBoost and CatBoost rows; §1.2 item 5 still requires CatBoost
  probabilistic wiring. Both models are dropped.
- **§7's bars** are written on distributional metrics (H2/H3/H5 reference calibration-type claims)
  and must be restated on the point-prediction set — or dropped if they no longer mean anything
  without σ. **Do not silently re-point a bar at a different metric; a bar that changes metric is a
  new bar and is labelled as one.**
- **§5.4's `symlog` rows** stay — the transform is about target scale, not variance — but any text
  justifying it by uncertainty conversion needs re-reading.

**Doctrine:** the σ removal is a SCOPE decision, not a metric downgrade to be smuggled. Every bar
that changes must be visibly restated, and the report must carry the caveat already recorded in §2:
squared error scores the mean only, so an MSE-only benchmark **under-detects** what k buys.
**Non-goals:** do not re-add any distributional metric "just for reference" — that reopens the scope.
*Orchestration:* parallel: no (same file as P0) · deps: P0 · tier: sonnet · scale: static ·
shape: execution · verify: `grep -niE "crps|pit|winkler|picp|calibration|NLL" docs/probreg_benchmark/benchmark_spec.md`
returns only historical/superseded notes, no live requirement; and no dropped model name survives as
a live spec row.

### P0-b1 — ✅ DONE (premise found already satisfied on disk; verified 2026-07-21) — propagate the M3 training ruling into the benchmark spec

**✅ No dispatch needed.** All four named spots were checked against disk 2026-07-21: each already
carries the §1 formulation with dated corrections, and the spec file has no uncommitted diff. The
task's own premise ("the spec still carries the superseded wording") was stale when written — the
same repair-pass defect class as D2's note. Spec kept below for the record; verify re-written (the
original `grep -n "train identically"` would flag the CORRECTED text, which legitimately contains
the live claim "M1 and M2 train identically" — it tested the wrong invariant).
**Verify as re-written, all three executed 2026-07-21, all pass:**
`grep -n "All three ProbReg models train identically" docs/probreg_benchmark/benchmark_spec.md`
returns nothing; `grep -n "SAME as M2/M3" docs/probreg_benchmark/benchmark_spec.md` returns
nothing; the §2.3 M3 build text sets `NClassesSelectionMethod.NONE` (`benchmark_spec.md:284`,
`:293`).

**Files (write set):** `docs/probreg_benchmark/benchmark_spec.md`
**Why.** §1 was corrected 2026-07-20 (user ruling): M3's per-k models are trained ORDINARILY at
fixed k, not with k-dropout. The spec still carries the superseded wording as a live requirement in
at least four places, and a driver built from the spec would train M3 the wrong way — destroying the
reference the whole efficiency claim is measured against.
**Spec:** Correct §2.0's "All three ProbReg models train identically — with k-dropout" to the §1
formulation (M1 and M2 identical; M3 ordinary per k). Correct §2.3's M3 build text. Correct the
naming-map row that says the training difference "is gone". Where a code block sets `NESTED` with a
comment claiming it is the "SAME as M2/M3", fix the comment. Record each as a dated correction, not
a silent edit — the superseded claim was in force and other documents may cite it.
**Doctrine:** §1 wins over any spec (this file's opening rule). The spec is the thing a driver is
built from, so a spec that disagrees with §1 is not a documentation nit — it is the defect that
reaches the results.
**Non-goals:** no other spec change; no code.
*Orchestration:* parallel: no (same file as P0) · deps: none · tier: sonnet high · scale: static ·
shape: execution ·
verify: `grep -n "train identically" docs/probreg_benchmark/benchmark_spec.md` returns only
struck-through or explicitly-superseded text; `grep -n "SAME as M2/M3" docs/probreg_benchmark/benchmark_spec.md`
returns nothing; and the M3 code block in §2.3 sets `NClassesSelectionMethod.NONE`.

### PA — the k-selection API — ✅ **DONE 2026-07-21** (built by worker; enum + integration fix applied at the ROOT)

**What PA actually had to build was NARROWER than this spec implies, and that was verified before
work started, not assumed:** FP-3 had already landed the `inference_mode` clean break and
`CapacitySelection.{FIXED, PER_INPUT}` on this model. PA's real gap was the two MECHANISMS —
M1 (`fit_global_selector`) and M3 (`fit_sweep_selector`) — plus the configurable selection fraction
and defect D6.

**Built** (`automl_package/models/probabilistic_regression.py`,
`automl_package/models/selection_strategies/n_classes_strategies.py`,
`tests/test_phase3_dynamic_k.py`; 52 passed, 1 pre-existing skip):
- **M1 = `fit_global_selector`**, built on `all_rung_log_likelihood`
  (`automl_package/models/flexnn/strategies/n_classes.py:230`) — **never** on
  `held_out_arbiter_advantage` (§1 ⚠️, D5). Selects via FP-9's already-landed shared primitive
  `automl_package/utils/capacity_selection.py`; the bootstrap-SE/tolerance rule was IMPORTED, not
  re-derived (the third-copy failure this programme exists to stop).
- **M3 = `fit_sweep_selector`**: one ORDINARY (`NClassesSelectionMethod.NONE`) model per k, scored
  on the held-out remainder, SAME selection rule.
- **`selection_fraction` is a constructor parameter** (default 0.15), not a baked-in constant — the
  parameter `PB` is about to measure.
- **D6 CLOSED**: `all_rung_log_likelihood`'s docstring no longer falsely claims
  `held_out_arbiter_advantage` consumes it (both function bodies re-read to confirm).

**ROOT-APPLIED, because `automl_package/enums.py` is FP-3's/shared and outside every strand's write
set:** `CapacitySelection.GLOBAL_CHEAP` and `GLOBAL_SWEEP` added in the same change that ships their
mechanisms (the naming-key rule; NOT ahead of them — that is the retired
`WidthSelectionMethod.DISTILLED` trap). The worker, unable to touch `enums.py`, had written
`getattr(CapacitySelection, "GLOBAL_CHEAP", None)` dormant guards; the root **replaced them with
direct enum references** once the members existed — comparing a live config value against `None` is
a silent-failure shape, and the stale comment block explaining the guard was deleted rather than
left to rot. `enums.py`'s class docstring now records that **the k family implements these two
members and width/depth do NOT yet** (WSEL-3/4, DSEL-6/7) — a member is a promise about the enum's
contract, not a claim every family implements it.

**⚠️ INTEGRATION DEFECT — found by the ROOT's post-enum smoke test. SCOPE CORRECTED BY ITS OWN
REMOVAL RUN — read the correction below before citing this.** With the members live, `fit()` under
`GLOBAL_CHEAP` crashed *inside training*:
`PyTorchModelBase._fit_residual_std` — the hook **FP-10 created earlier the same wave** — calls the
caller-facing `self.predict()`, which `GLOBAL_CHEAP` correctly REFUSES mid-fit, because selection
runs AFTER training returns (`_fit_global_cheap`: `super().fit(...)` THEN `fit_global_selector(...)`).
**Neither worker could have caught it:** PA built the modes while `enums.py` still lacked their
members, so this path was unreachable in its tests; FP-10 landed the hook with no knowledge that two
selection modes were arriving. **Fixed at the root**, same shape as FP-10's own width fix: a single
`_predict_unselected` internal path, with `_fit_residual_std` and `_predict_for_scoring` overrides on
ProbReg routing to it (under `GLOBAL_SWEEP`, scoring goes through the fitted sub-model, which IS the
model).

**🔻 SCOPE CORRECTION — the root's own first write-up of this OVERSTATED it, and the removal run is
what caught that.** The first two regression tests written for it (end-to-end `fit()` under each
global mode) were re-run with the override DELETED and **both still PASSED** — so they were coverage,
not evidence, and the "load-bearing integration defect" framing did not survive its own verification.
**Why the defect is narrower than it first appeared:** `PyTorchModelBase._fit_single` calls the
residual-std hook ONLY under `uncertainty_method=CONSTANT`
(`automl_package/models/base_pytorch.py:217`), whereas `GLOBAL_CHEAP` legally requires `NESTED`,
which itself requires `PROBABILISTIC` — **so in a VALID configuration the crash path is unreachable.**
**What is genuinely fixed:** `CONSTANT` is ProbReg's DEFAULT `uncertainty_method`, so requesting
`GLOBAL_CHEAP` without also setting `NESTED` — a plausible caller mistake — used to die mid-`fit()`
with `No global k selected`, pointing at the wrong thing entirely; it now raises the precondition
error that names the real mistake. That is an error-quality fix on an invalid configuration, **not a
silent-wrong-answer fix on a valid one**, and it must not be cited as the latter.
**The discriminating guard** (`tests/test_phase3_dynamic_k.py::TestGlobalSelectionEndToEndThroughFit::test_global_cheap_without_nested_raises_precondition_not_midfit_crash`)
**was PROVED in both directions at the root**: with the override deleted it fails with
`RuntimeError: No global k selected`; with it restored it passes.
*Case law, in its surviving form: **a test written against a fix must be run with the fix removed
before the fix is described — the removal run is what sizes the defect, not the author's reading of
it.** The secondary lesson stands: an enum member landed by one task can activate paths in another
task's file, so smoke-test every family that dispatches on a shared enum after adding a member.*

**A worker finding that did NOT survive root verification — recorded so it is not re-filed.** PA
reported that `get_params()` fails to round-trip six fields, making CV/HPO sub-models silently train
without `target_transform="symlog"` — a D1/D2-shaped silent-wrong-answer bug on HEAD. **Checked at
the root by constructing with non-defaults and reading `_clone()` back: FALSE for five of the six.**
`target_transform`, `prob_reg_loss_type`, `use_anchored_heads`, `loss_type` and `beta` are all in
`get_params()` and survive the clone (symlog round-trips). Only `calculate_feature_importance` is
absent, and dropping a diagnostic switch from a clone is arguably correct — it also prevents every
sweep sub-model running SHAP unasked. **No bug, nothing filed**; the two source comments asserting it
were corrected in place (`automl_package/models/probabilistic_regression.py`, `fit_sweep_selector`
docstring + call site).

**Carried forward, NOT done here:** the `GLOBAL_CHEAP` path requires
`n_classes_selection_method=NESTED` **and** `uncertainty_method=PROBABILISTIC` (both enforced with
explicit constructor errors — correct: M1 reads a prefix-nested per-rung likelihood curve, which is
undefined for constant/MC-dropout heads). P7 and PB consume this API and must construct accordingly.

*(Original task spec follows, retained verbatim as the pre-registration this run was judged against.)*

### ~~PA — the k-selection API — OPEN, DISPATCHABLE~~ ➡️ **SUPERSEDED HEADER — the task is ✅ DONE; see the PA header above. Retained only as the pre-registration.** *(Struck 2026-07-21: an audit found a header grep could match this line and re-dispatch finished work. Original note: disambiguated 2026-07-21: the ✅ that stood in this header marked its two design DECISIONS settled 2026-07-20, not the task — an audit flagged the header as readable-as-done)*

**Why this is a task and not an implementation detail.** The three models must be three *options on
one model*, usable standalone and sharing every component. Today they are three different shapes,
and one is a trap. Verified 2026-07-20:
- **M2** takes three separate steps — a constructor option, a separate `fit_router()` call, then
  `inference_mode="routed"` passed to BOTH `predict` and `predict_uncertainty`
  (`automl_package/models/probabilistic_regression.py:598`, `:694`). **Forget that flag and you
  silently get the un-routed model** — the default is `"soft"`, there is no error, and nothing
  records that a router was fitted but unused. Same silent-wrong-answer class as D1 and D2.
- **`inference_mode` is a raw string**, validated by a hand-rolled membership check
  (`automl_package/models/probabilistic_regression.py:613`, `:708`), with no type anywhere. This
  breaks the repo's own closed-set rule (`CLAUDE.md` → enums, never magic strings).
- **M1** has no mechanism in code yet — the *rule* is now specified (`benchmark_spec.md` §2.1:
  cheapest-within-tolerance at twice a bootstrap standard error, bypass competing), so this is a
  build, not a design question. It is PA's main piece of work. **Build it on
  `all_rung_log_likelihood`** (`automl_package/models/flexnn/strategies/n_classes.py:230`)
  — the per-rung `(batch, n_classes)` held-out likelihood table — never on
  `held_out_arbiter_advantage`, which is a different, per-input readout that cannot answer a
  global "which k" question (§1 ⚠️, D5).
- **M3** exists only as a loop inside an example script
  (`automl_package/examples/report_a_benchmark.py:331`), not as a model.

**Shape (settled by this repair pass, 2026-07-20 — supersedes the earlier "propose a `KSelection`
enum" text, which collided with `flexnn-package.md` FP-3's `CapacitySelection` and is retired):**
PA **consumes** `CapacitySelection` (`automl_package/enums.py`, owned by `flexnn-package.md` FP-3)
— it does not define an enum of its own. FP-3 ships exactly two members, `FIXED` and `PER_INPUT`;
**`GLOBAL_CHEAP` and `GLOBAL_SWEEP` are added by the task that builds their mechanism** — PA adds
`GLOBAL_CHEAP` in the same change that builds M1's selector, and `GLOBAL_SWEEP` in the same change
that builds M3's, never as a member whose mechanism raises `NotImplementedError` (the
`WidthSelectionMethod.DISTILLED` trap `flexnn-package.md` FP-3.a retires; do not recreate it here
under a new name). The member is passed at construction. `fit()` performs whatever held-out
selection the chosen mode needs. **`predict` loses its `inference_mode` argument entirely** — the
model already knows how it selects k, and requiring the caller to remember is what creates the
silent failure. All three then drop into any comparison harness unchanged.

**✅ Both blocking decisions SETTLED by the user 2026-07-20:**
1. **Selection rule = cheapest-within-tolerance, NOT argmax.** k is the **smallest** rung whose
   held-out score is not meaningfully worse than the best rung's. Rationale on record: held-out
   curves are noisy, so argmax systematically overshoots (any upward wiggle at a high rung wins by
   chance) and would report more resolution than the data supports; and the question this strand
   asks is "how much resolution is needed", which is a smallest-sufficient question. **Tolerance
   rule: reuse the one already published in
   `docs/reports/probreg_kselection/probreg_kselection.md` §3.2** — a difference counts as real only
   if it exceeds twice a bootstrap-estimated standard error — so results stay comparable with the
   existing report instead of needing a caveat. **The SAME rule is applied to M3's sweep curve**, or
   M1 and M3 are not answering the same question.
2. **Clean break on `inference_mode` — delete it, do not deprecate it.** The repo has no external
   users (user, 2026-07-20), so a compatibility shim is pure cost: it keeps the silent-failure route
   alive, doubles the test surface, and will eventually be used by accident.

**Also required (user, 2026-07-20): the selection-set fraction must be CONFIGURABLE**, not a baked-in
constant — it is a parameter of the method, and PB is about to measure it.

**Files (write set):** `automl_package/models/probabilistic_regression.py` ·
`automl_package/models/selection_strategies/n_classes_strategies.py` ·
`tests/test_phase3_dynamic_k.py` · call sites found by grep, not by memory
**Non-goals:** no new selection *algorithms* — this task changes how the existing three are reached,
not what they do. No change to `DistilledCapacityRouter` internals. No change to `CapacitySelection`
itself, its name, or its cross-family contract — file a finding against `flexnn-package.md` FP-3
instead of patching it from here. Do not touch `automl_package/enums.py`.
*Orchestration:* parallel: no (touches the same file as P1) · deps: `flexnn-package.md` FP-3,
`flexnn-package.md` FP-9, P1 · tier: sonnet high · scale: static · shape: design → execution ·
verify: `grep -rn "inference_mode" automl_package/ tests/` returns nothing (clean break, no shim);
`grep -n "class KSelection" automl_package/enums.py` returns nothing (no home-grown enum);
`grep -n "CapacitySelection" automl_package/models/probabilistic_regression.py` is non-zero; a test
asserts a `CapacitySelection.PER_INPUT`-constructed, router-fitted model routes with no caller flag;
`AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m pytest tests/test_phase3_dynamic_k.py -q` exits 0.

### PB — how much data does the selection step actually need? (NEW, user 2026-07-20)

**Why.** The 15% selection fraction was chosen because it looked reasonable, **not because anything
was measured**. Searched 2026-07-20: no study of selection-set size exists anywhere in this repo for
ProbReg. The entire efficiency claim runs through a mechanism whose data requirement is unknown —
and the three arms are not equally exposed. M2 must learn a *function* from x to rung, so it should
be the hungriest; M1 and M3 need only rank rungs on average. **If M2 loses at 15%, "per-input k does
not pay" and "the router was starved" are indistinguishable** — the same class of unattributable
result that has already cost this programme twice.

**Files (write set):** `automl_package/examples/probreg_pb.py` (Create) ·
`automl_package/examples/capacity_ladder_results/PB/`
**Spec:** sweep the selection fraction (suggest `{5, 10, 15, 25, 40}%` of the training portion) on
the toys where ground truth is known, for all three arms, holding everything else fixed. Report each
arm's quality against fraction, and the fraction at which each saturates. The headline output is a
defensible default and, for M2, an explicit statement of the data floor below which its result is
not readable.
**Emit the frozen-constants artifact §3.6 promises:**
`automl_package/examples/capacity_ladder_results/PB/frozen.json`, containing exactly the two
constants this task owns per §3.6 — the selection-set fraction (the pre-authorised default per §3.5
if no fraction saturates) and the M2 data-limited flag, one boolean per (toy, arm).
**Non-goals:** no real data (that is P4's budget); no architecture changes (PC).
*Orchestration:* parallel: yes (disjoint from PC's write set — separate driver scripts and results
directories, see Files above) · deps: P1, PA, P7 · tier: sonnet high · scale: dynamic (a sweep) ·
shape: research · verify:
`test -f automl_package/examples/capacity_ladder_results/PB/frozen.json` and
`python -c "import json; d=json.load(open('automl_package/examples/capacity_ladder_results/PB/frozen.json')); assert 'fraction' in d and 'data_limited' in d, d"`
exits 0; `ls automl_package/examples/capacity_ladder_results/PB/*.json` shows one file per
(toy, seed, fraction, arm); a saturation plot file exists under the same results dir.

### PC — is the distilled router's architecture right for ProbReg? (NEW, user 2026-07-20)

**What exists.** The router is fixed at two hidden layers of 32 units, ReLU, 300 full-batch Adam
epochs, lr 1e-2, cross-entropy on cheapest-within-tolerance labels (tolerance 0.25) —
`automl_package/models/flexnn/routing.py:57-60`. Those constants were **copied from an
earlier width experiment**, not chosen for ProbReg. The only sensitivity evidence anywhere is from
the WIDTH strand: `docs/plans/capacity_programme/width-cert.md:234` ran half and double router hidden
on 3 seeds and found the deploy claims invariant (`:237`). That is reassuring but it is (a) a
different strand, (b) one dimension, (c) a does-it-break check, not a search.

**Files (write set):** `automl_package/examples/probreg_pc.py` (Create) ·
`automl_package/examples/capacity_ladder_results/PC/`
**Spec:** on the toys, vary router width/depth (at least half/double/4× hidden and 1 vs 3 layers)
and epochs. *(The labelling tolerance is STRUCK from the swept dimensions, 2026-07-21: MASTER
Decision 18 rules that sweep NOT scheduled — do not run pre-emptively; this spec contradicted the
register. The flat 0.25 stays inherited-and-accepted.)* Establish whether ProbReg's routing
conclusions are invariant the way width's were, and if not, what the router actually needs.
**Emit the frozen-constants artifact §3.6 promises:**
`automl_package/examples/capacity_ladder_results/PC/frozen.json`, containing exactly the two
constants this task owns per §3.6 — router hidden/depth/epochs/lr. If
this task finds invariance, the file records the current frozen defaults
(`automl_package/models/flexnn/routing.py:57-60`) rather than inventing new ones.
**Doctrine:** the router stays FROZEN and untuned inside the benchmark (§2.2) so the M1/M2 contrast
measures selection rather than search effort. **This task does not unfreeze it** — it establishes
whether the frozen choice is defensible, and any change lands as a new frozen default *before* P4
runs, never per-dataset.
**Non-goals:** no per-dataset tuning of the router, ever. No change to the labelling rule's meaning.
*Orchestration:* parallel: yes (disjoint from PB's write set — separate driver scripts and results
directories, see Files above) · deps: P1, PA, P7 · tier: sonnet high · scale: dynamic · shape: research ·
verify: `test -f automl_package/examples/capacity_ladder_results/PC/frozen.json` and
`python -c "import json; d=json.load(open('automl_package/examples/capacity_ladder_results/PC/frozen.json')); assert {'hidden','depth','epochs','lr'} <= d.keys(), d"`
exits 0; `python -c "import json; d=json.load(open('automl_package/examples/capacity_ladder_results/PC/frozen.json')); assert 'invariant' in d, d"`
exits 0 (the invariant-or-not verdict is a field, not prose); if `d['invariant']` is `False`, the
same file's `new_default` key is non-null and cited by PB's re-run.

### P7 — the WRITE side: migrate M1/M2 to the all-rungs schedule **AND to fixed-σ training**, re-validate ONCE (MASTER Decisions 20 + 26)

⚠️ **SCOPE WIDENED 2026-07-21 (root): P7 absorbs MASTER Decision 26 — σ fixed in TRAINING.** Both
changes rewrite the same loss. **Landing them separately would waste a re-validation and create a
confound**: whichever ran second would measure the schedule change and the σ change entangled, with
no way to attribute a movement to either. **One change, one re-validation.**

**What the σ half means concretely** (Decision 26):
- With σ a shared constant, each rung's NLL reduces to squared error up to a constant ⇒ **training
  becomes MSE on the mixture mean.**
- **The per-class heads predict MEANS ONLY.** *(Root-applied implementation ruling, flagged in
  Decision 26 for the user to overturn.)* The rejected alternative — keeping a `log_var` output the
  loss no longer trains — leaves an **untrained head that `predict_uncertainty` still exposes**,
  a worse trap than the one being removed.
- **The within-component term of the law-of-total-variance combination becomes the constant**, so
  predictive spread = between-component spread + σ². Never fitted, still meaningful.
- **The variance machinery is NOT deleted** (Decision 2's carry-over) — never fitted, never selected
  on. Deletion is out of scope and user-gated.

**⚠️ RE-BASELINE THE SUITE — the standing bar does NOT carry across this change.** The two accepted
heteroscedastic failures test variance behaviour directly, and this alters what a component can
express. **Run the suite FIRST, record the new expected result in this task's completion note as the
new baseline, and treat only movement beyond that as a regression.** 🚫 **Do NOT drive the old
numbers back to green** — that would mean suppressing the very effect this change makes.

**deps: `flexnn-package.md` FP-12 must be MERGED FIRST** — P7's re-validation selects a k, and until
FP-12 lands there is no compliant metric to select on (it would score on a learned variance, the
forbidden case). **P7 and FP-12 write the same model file and cannot share a session** under the
session-scoped write guard.

**Files (write set):** `automl_package/models/flexnn/strategies/n_classes.py` ·
`automl_package/models/probabilistic_regression.py` *(added 2026-07-21 with the σ half — the loss and
the head-output shape live here)* ·
`automl_package/enums.py` · `tests/test_phase3_dynamic_k.py` ·
`automl_package/examples/capacity_ladder_results/P7/`
**Spec:** Implement the ruled schedule for `NClassesSelectionMethod.NESTED`: per training step, ONE
`all_rung_outputs` forward; the loss is the mean over rungs k ∈ {1..k_max} (bypass rung included)
of that rung's NLL — replacing the per-sample uniform draw as the default. The draw stays available
as a labelled comparison arm: add a `NestedSchedule` enum (`ALL_RUNGS` default, `UNIFORM_DRAW`
legacy) to `automl_package/enums.py` — this task builds the mechanism, so this task adds the member
(the naming-key rule). Then RE-VALIDATE: re-run the §2 middle-k coherence check on the toys, both
seeds, under `ALL_RUNGS`, emitting one JSON per cell plus
`automl_package/examples/capacity_ladder_results/P7/frozen.json` recording the schedule and the
old-vs-new middle-k failure counts (old: 8/9 with max-k 9/9, cited by pointer not copied numbers).
**Positive control (Decision 14):** under `ALL_RUNGS` the max-k rung's coherence must not regress
from 9/9 — it never failed under the old schedule; if it regresses, HALT (the protocol change is
then the defect).
**Why this task exists:** §1's 2026-07-21 training-schedule ruling; the note there lists the three
forcing facts. This run is simultaneously P5's discriminating experiment — see P5's prior-art
clause.
**Non-goals:** no change to the selection mechanisms (arbiter/distillation/sweep — PA's ground); no
change to M3 (it never used the draw); no real data.
*Orchestration:* parallel: no (shares the k-strategy module and the test file with PA's ground)
· deps: PA, **`flexnn-package.md` FP-11** *(added 2026-07-21 — FP-11 moves the k-strategy module to
`automl_package/models/flexnn/strategies/n_classes.py`; running P7 first would mean moving it twice)* ·
tier: sonnet high · scale: static · shape: execution →
verify: `AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m pytest tests/test_phase3_dynamic_k.py -q`
green, including a NEW test asserting the `ALL_RUNGS` loss equals the mean of the per-rung NLLs
computed independently (every rung receives gradient every step); demonstrate the test is real by
truncating the loss to the drawn-rung-only form, showing it FAIL, restoring;
`test -f automl_package/examples/capacity_ladder_results/P7/frozen.json` exits 0 and the file
carries `schedule`, `middle_k_failures_old`, `middle_k_failures_new`; every per-cell JSON carries
`held_out_trajectory` and `hit_cap: false`.

### P8 — does explicit regularisation move the selected k? — ⛔ **REOPENED 2026-07-21. RESULTS DISCARDED — SELECTED ON A FORBIDDEN METRIC (Gaussian NLL). The verdict below is VOID.**

**Why it is void.** The selection was made on **Gaussian negative log-likelihood**, a variance-based
metric: `automl_package/examples/probreg_p8.py:145` builds `sel_nll_per_sample` from the predicted
mean AND predicted standard deviation, and `:180-182` feeds exactly that as the error table into
`cheapest_within_tolerance`. The σ-scope decision removed variance from this strand and replaced the
metric set with **point prediction, primary squared error** (P0b; `docs/probreg_benchmark/benchmark_spec.md`
§4 — "NLL, CRPS, PIT, calibration/ECE, PICP, Winkler all assume σ… replace with a point-prediction
set"). The run already computes `report_mse` (`:150`), so the sanctioned metric was in hand and was
not the one selected on. *(User ruling 2026-07-21: results produced in violation of a constraint are
DISCARDED, not reinterpreted.)*

**Status of the artifacts.** The per-λ/seed JSONs and `frozen.json` under
`automl_package/examples/capacity_ladder_results/P8/` **stay on disk as a record and may NOT be cited
as evidence.** Deletion is user-gated and not proposed.

**Downstream consequence — the block STAYS, and the reason changes.** The earlier state was "the check
ran and the selection MOVED, therefore block". The correct state now is "**the check has not validly
run**", and MASTER Decision 21 blocks a battery read until it has. So **P4 and P6 remain blocked** —
but nobody may now cite "regularisation moves k" as an established finding, because that finding was
produced on the wrong metric. **Both the block and the claim it rested on are open questions.**

⚠️ **THE REMEDY BELOW WAS ITSELF WRONG — CORRECTED 2026-07-21 (root), and this is the second time
this strand has been about to select on a metric that cannot see its own dial.** The text originally
read *"select on the point-prediction metric (squared error)"*. **Do NOT do that.** Per MASTER
Decision 24 and the corrected §0.5, squared error is *structurally blind to k* for symmetric targets:
the mixture mean is identical whether the model resolves one component or five, so the selection
curve flattens, cheapest-within-tolerance returns k=1 at every λ, and P8 reports a confident **"the
selected k does not move"** — a verdict manufactured by the metric, not measured from the data. **On
THIS task that failure is maximally deceptive**, because "does not move" is the outcome that
*unblocks* the battery. A blind metric would have cleared the block it exists to enforce.

**The diagnosis of what voided the original run stands unchanged** — it selected on a **learned**
variance. What changes is only the remedy: the fix is a likelihood at **FIXED σ**, not a retreat to
squared error. Both errors share one root cause, which is why §0.5 now states the test as *"is σ
learned?"* rather than *"is it a likelihood?"*.

**What the re-run must change (and ONLY this):** select on the **per-sample fixed-σ mixture
log-likelihood** (MASTER Decision 24), keeping the λ grid, seeds, splits, convergence gates and the
twice-bootstrap-SE selection rule byte-identical to the spec below. **⇒ P8 therefore deps on FP-12**
(`flexnn-package.md`), which builds that scorer — until it exists there is nothing compliant to
select on, and re-running P8 before it lands would void the result a third time. Recording squared
error alongside as a labelled point-accuracy column is fine;
selecting on it is not.

**⚠️ Same defect, same day, different strand — this is a PATTERN, not an incident.** The width
strand's equivalent check (`width.md` WSEL-11) was reopened at the same time for the same class of
violation, with the OPPOSITE verdict. Two checks whose whole purpose was to protect batteries were
themselves run outside the constraint they protect. → MASTER Decision 21 amended.

*(Recorded verdict retained, struck, as the case law.)*
~~✅ RAN 2026-07-21. VERDICT: MOVES → ⛔ STRAND-LOCAL BLOCK on P4/P6.~~

**RESULT: `selection_moved: true`** — ledger `automl_package/examples/capacity_ladder_results/P8/frozen.json`, thirteen per-cell JSONs in the same directory (λ ∈ {0, 1e-4, 1e-2} × seeds {100, 101}, plus the driver's own epoch-raise and ceiling-raise re-runs). Toy: heteroscedastic. Selection rule:
`cheapest_within_tolerance` at twice a bootstrap SE (`automl_package/utils/capacity_selection.py`).

**Consequence (MASTER Decision 21, block semantics 2026-07-21): P4 and P6 MAY NOT BE READ** until the
confound is re-derived. Strand-local: `PB`/`PC`/`P3` are unaffected, the other strands continue, and
this is **batched for end-of-run user review**, never resolved by the run.

**⚠️ READ THE PATTERN BEFORE READING THE VERDICT — the two seeds behave completely differently, and
the movement is almost certainly INSTRUMENT INSTABILITY, not a clean regularisation effect** *(root
analysis, 2026-07-21)*:
- **Seed 101 is rock stable:** selects the same k at every λ — 5, 5, 5. It does not move at all.
- **Seed 100 moves, but NOT MONOTONICALLY in λ:** 10 → 8 → 12 as λ goes 0 → 1e-4 → 1e-2. More
  regularisation selecting a LARGER k is backwards for the mechanism under test; a genuine
  "regularisation lets you get away with less capacity" effect would push k DOWN, monotonically.
- **The decisive evidence that it is the instrument: the same cell flips selection under a longer
  EPOCH budget, with λ held fixed.** The driver's own convergence escalation (`epoch_raises` in
  `frozen.json`) re-ran cells that hit the epoch cap and recorded: at λ=0 seed 100, selected k went
  **8 → 10**; at λ=1e-4 seed 100, **12 → 8**. Weight decay was CONSTANT across each of those pairs.
  **A selection that moves when only the training budget changes cannot be attributed to weight
  decay.** Seed 101's selection survived every one of those same re-runs unchanged.

⇒ **The honest statement of this result: on this toy, k-selection is UNSTABLE on one of two seeds —
to the weight-decay grid, to the epoch budget, and non-monotonically — while the other seed is
perfectly stable. Decision 21's question ("does regularisation move the selected capacity?") is NOT
cleanly answered, because the instrument moves for reasons unrelated to the treatment.** This is
closer to §3.5's HALT-2 shape (*"a study comes back incoherent rather than merely negative ... that
is a broken instrument"*) than to a clean positive. The pre-authorized strand-local block is taken
(it stops exactly what needs stopping — the battery reads), the run CONTINUES elsewhere, and the
diagnosis goes to the user.

**Do NOT, on the strength of this:** conclude that weight decay reduces the k a model needs; quote
the seed-100 λ-series as a trend (it is non-monotone); or "fix" it by widening the tolerance until
the movement disappears — that would hide the instability rather than resolve it.

**Suggested (NOT scheduled, NOT run — the user decides):** the discriminating follow-up is seed
count, not λ resolution — if k-selection varies this much across two seeds and across epoch budgets
at fixed λ, the selection-set size (`PB`'s question) and seed variance are the live suspects, and
`PB` is already the scheduled task that measures the first of them.

**Protocol note — the driver did this right, unprompted:** it detected cells hitting the 100-epoch
cap without early stopping and re-ran them at 400 epochs (MASTER Decision 9: never read a
non-converged endpoint), and re-ran the one ceiling-bound cell at the raised k_max (§3.5's "ceiling
binds" branch; λ=1e-2 seed 100 re-selected 12, so the raise did not change the answer).
`any_hit_cap: false` across the final record. Those escalations are what surfaced the instability —
had the driver read the capped endpoints, this would have been recorded as a clean "moves".

*(Original task spec follows, retained verbatim as the pre-registration this run was judged against.)*

### P8 — does explicit regularisation move the selected k? (NEW 2026-07-21, MASTER Decision 21)

**Files (write set):** `automl_package/examples/probreg_p8.py` ·
`automl_package/examples/capacity_ladder_results/P8/`
**Spec:** The programme's research training is entirely unregularised (no weight decay, dropout,
norm layers, or mini-batching — audited 2026-07-21), so cheapest-within-tolerance may partly select
small k because small OVERFITS LESS, not because small suffices — a bias identical across all three
dials and invisible to cross-strand agreement. Discriminating check, one toy: train M3's per-k
ORDINARY sweep models at AdamW `weight_decay` λ ∈ {0, 1e-4, 1e-2}, 2 seeds, unchanged convergence
gates; apply the strand's selection rule to each per-k curve; report whether the selected k moves
beyond tolerance, emitting the verdict to
`automl_package/examples/capacity_ladder_results/P8/frozen.json`. **Moves → block THIS strand's
battery reads (P4/P6 may not proceed), log prominently, continue the OTHER strands, batch for
end-of-run user review** *(pre-authorized 2026-07-21 — a strand-local block, not a whole-run halt;
the strand's numbers conflate capacity with regularisation and the battery may not be read until
re-derived)*. **Does not move → robustness note that P6's report MUST cite.** The reported numbers come from a split not used for stopping or
selection. Weight decay is framed as the Gaussian prior it is — the no-arbitrary-penalty premise
binds the SELECTION objective, not MAP training.
**Non-goals:** no change to any model definition; no tuning of λ beyond the fixed grid; no real
data.
*Orchestration:* parallel: yes (disjoint write set) · deps: none · tier: sonnet high · scale:
static · shape: execution →
verify: one JSON per (λ, seed) under `automl_package/examples/capacity_ladder_results/P8/` each
carrying `selected_k`, `held_out_trajectory`, `hit_cap: false`;
`test -f automl_package/examples/capacity_ladder_results/P8/frozen.json` exits 0 and the file
carries `selection_moved` (bool) and, if true, the per-λ selected k values.

### P1 — ✅ DONE (landed 2026-07-20 in commit `84ad94d`; recognised 2026-07-21) — fix D2 (the arbiter units mismatch)

**✅ No dispatch needed.** The exact fix and the exact regression test this task specifies were
found already committed (`84ad94d`) during the 2026-07-21 repair audit; the test was re-executed at
the root: 2 passed. The prove-it-fails ceremony is waived with reason: fix and test landed in one
commit, so the FAIL state no longer exists to demonstrate — and the test asserts on the per-sample
likelihood (the quantity the fix changes), satisfying the doctrine's intent. Spec kept below for
the record.

**Files (write set):** `automl_package/models/probabilistic_regression.py` ·
`tests/test_phase3_dynamic_k.py`
**Spec:** Transform `y` to the fitted space inside `_per_sample_log_likelihood_at_k` when
`target_transform="symlog"`, exactly as `fit_router` now does; document the contract on
`held_out_arbiter_advantage` the way `fit_router`'s docstring does. Add a regression test that the
arbiter's advantage is identical whether given raw or pre-transformed `y`.
**Doctrine:** a regression test is not evidence until it has been shown to FAIL on the unfixed
code. Assert on the quantity the fix changes (the per-sample likelihood), never on a coarse
downstream view — that is exactly how D1's first tests came out blind.
**Non-goals:** no change to the arbiter's neighbourhood averaging or its certified width default;
no touching `fit_router` (already fixed).
*Orchestration:* parallel: no (same file as D1's fix) · deps: none · tier: sonnet high ·
scale: static · shape: execution ·
verify: `AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m pytest tests/test_phase3_dynamic_k.py -q`
green, THEN delete the transform, re-run, and show the new test FAILING, then restore and show the
file checksum unchanged.

### P2 — diagnose or accept the undocumented failing test (D3, second one) — ✅ **DONE 2026-07-20. Outcome (c): ACCEPTED, no fix warranted.**

**Verdict: the test is a brittle single-seed assertion, not a package defect.** Both halves of the
task's own verify were executed at the root, on real `pytest`, one variable, both directions: at HEAD
the test fails at MSE 2.8812, and with the recorded candidate fix applied it **still fails**, at  <!-- source: `automl_package/examples/capacity_ladder_results/P2/p2_warmstart_grid.json` -->
2.8565 — a 0.9 % move against a 0.39 gap. **The recorded "fix applied → 1 passed" was refuted.** It  <!-- source: `automl_package/examples/capacity_ladder_results/P2/p2_warmstart_grid.json` -->
had been transcribed from a killed worker's transcript and never re-executed; re-executing it is what
overturned it. Outcome (a) is therefore excluded (the test does not pass) and (b) is excluded (there
is no fix that survives prove-it-fails), so the outcome is the acceptance branch this task's spec
requires — written up, cited, and in the same shape as `shared/hetero_nll_diagnosis.md`.

Root cause, established across 5 seeds with per-epoch trajectories (MASTER Decision 9 satisfied):
`k=5` on one hard-coded seed, with a bar sitting 0.25 above a distribution of spread 0.41. Four of
five seeds land under the bar in **both** directions; only the seed the test hard-codes exceeds it.
The culprit commit `445315e` is confirmed — the test passes at its parent — but the crossing is
carried by the RNG-stream shift in that commit, not by the centroid warm-start the earlier note
blamed. **The 2.5 bar was NOT loosened and must not be.** No change to
`automl_package/models/probabilistic_regression.py` is warranted; the file is at HEAD, verified by
`git diff --stat`.

Full record, including the 2×2×5 grid and both trajectories:
`docs/plans/capacity_programme/shared/p2_hetero_mse_diagnosis.md`. Ledger cells:
`automl_package/examples/capacity_ladder_results/P2/p2_warmstart_grid.json` and
`automl_package/examples/capacity_ladder_results/P2/p2_warmstart_seed42_trajectories.json`.

**✅ RULED 2026-07-20 (user): the test's protocol IS to be repaired, and the repair WAITS for the
selection rule. Do not patch this test twice.** Until then the failure stands as a known failure and
is **not** to be treated as a regression, and the 2.5 bar does not move.

**Why deferred, and what the deferral is really about — this is the load-bearing part, not the seed
noise.** The configuration under test is `n_classes=5` with
`n_classes_selection_method=NONE` (`tests/test_phase4_regression.py:51-62`, `_make_probreg`): no
arbiter, no distilled router, no sweep, no k-dropout. **It is none of §1's three models** — it is the
bare network trained ordinarily at ONE hand-picked rung, closest to a single *cell* of M3's sweep but
without the selection step that makes M3 a model. Its bar was set from that same hand-picked run.
⇒ **The test cannot distinguish "the model regressed" from "k=5 was never the right resolution for
this data"**, because the dial the whole strand is about is frozen. Multi-seed averaging would make
it stable and leave it arbitrary; that is why the seed fix alone was rejected.

**The repair, when its dependency lands:** select `k` on held-out data using the strand's own rule —
cheapest-within-tolerance at twice a bootstrap standard error — instead of pinning `k=5`, holding the
per-seed bar at 2.5. **Tracked dependency: `flexnn-package.md` FP-9.a/FP-9.b** (the shared selector
and its bootstrap standard-error helper) **and PA** (M1's selector built on `all_rung_log_likelihood`).
Not a dispatchable task until both exist; re-read this block when PA closes.

*(Method note, recorded because it cost two rounds: the first two explanations of this failure led
with the noise floor and the per-seed spread and never stated the configuration. The configuration
was the finding. Pin the variant — and whether the mechanism under study is switched on — before
analysing any number.)*

**Also noted, not scheduled:** the
centroid warm-start is shipped, test-uncovered in both directions, and now measured as null on this
configuration (`+0.0026 ± 0.0374` paired over 5 seeds) — keeping it is the recommendation, but it
wants an executable identifiability check before anyone claims it works. <!-- numcheck-ignore: derived statistics over the grid cell cited above -->

*(Original spec retained below, unchanged.)*

**Files (write set):** `docs/plans/capacity_programme/shared/` (a new diagnosis note) ·
possibly `automl_package/models/probabilistic_regression.py`
**Spec:** Establish whether error 2.8812 against the 2.5 bar is a real regression or a stale bar.
The bar's own comment cites a 1.54 baseline; find when it last held (`git log -S` on the test) and
what changed. Outcome is either a fix, or a written, cited acceptance in the same shape as
`docs/plans/capacity_programme/shared/hetero_nll_diagnosis.md`. **"Known failure" with no document
is not an outcome.**
**Non-goals:** do not loosen the bar to make it pass. Do not touch the other failing test.
*Orchestration:* parallel: **no — runs AFTER P1** (both may touch `automl_package/models/probabilistic_regression.py`; the old "parallel: yes if it lands doc-only" made the safety of a concurrent dispatch depend on an outcome the dispatcher cannot know in advance) · deps: **P1** · tier: sonnet
high · scale: static · shape: research ·
verify: `test -f docs/plans/capacity_programme/shared/<note>.md` exits 0, and
`grep -qE "^[0-9a-f]{7,40}" docs/plans/capacity_programme/shared/<note>.md` shows the commit at
which the bar last held (from `git log -S` on the test). Then exactly one of two outcomes, both
mechanically checkable: **(a) stale bar** —
`AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m pytest tests/test_phase4_regression.py::TestPerformanceBaselines::test_prob_regression_heteroscedastic_mse -q`
passes, and the note says so; **(b) real regression** — same prove-it-fails shape as P1's template:
the fix is applied and the suite is green, THEN the fix is deleted, the same test is re-run and
shown FAILING, then the fix is restored and the file checksum is shown unchanged. "Known failure,
no fix, no acceptance note" is not a valid outcome under either branch.

### PT — ⏸ PARKED. Toys are RETAINED; the redesign answers a different question.

**Resolution 2026-07-20 (user).** σ is not fitted, but scoring stays **distributional** with one
shared fixed σ (`docs/probreg_benchmark/benchmark_spec.md` §2, §4.1). The three existing toys are
homoscedastic by construction — exactly what a fixed σ wants — so **they are retained unchanged and
the multimodality question they were built for survives intact.**

**What actually happened, recorded because it is the instructive part.** The user's decision was
"do not fit σ". The orchestrator inferred "therefore score on squared error", and from that
correctly derived that the toys were degenerate: their two modes are equal-weight and symmetric
about zero (`automl_package/examples/_toy_datasets.py:174-178`), so the conditional mean is
identically zero and squared error has **zero power** to detect k. The derivation was sound; **the
premise was an unrequested inference.** The user challenged it, and the third option — keep the
mixture likelihood, fix σ to a shared constant — removes the in-sample-overfitting pathology
without removing the readout. **Lesson: when a scope decision seems to invalidate an established
artifact, re-examine the inference chain before discarding the artifact.** The cost here was one
spec; had it gone the other way it would have been the whole toy suite and the question with it.

**The spec is not wasted:** `docs/probreg_benchmark/toy_design_spec.md` designs toys where
resolution helps the *conditional mean*, with an analytic floor and a falsifiable
noise-versus-resolution prediction. That is a genuinely interesting and *different* question —
"classification as a regulariser for the mean" rather than "how many components does this input
need". **Parked for a future strand**, not deleted, and explicitly not a dependency of anything
below.
*Orchestration:* PARKED — no deps, no tier, not dispatchable, blocks nothing.

### P3 — the M1 ≈ M3 claim, both halves, on toys

⛔ **BLOCKED on the M3 candidate-set ruling (§1, raised 2026-07-21).** M3's grid starts at k=5 while
the ladder is read from the bypass up, so M1 can currently select a rung M3 is not allowed to
consider. **No M1-vs-M3 headline may be produced until that is settled** — an unqualified "M1
matches M3" off a narrow grid would flatter the cheap arm for a reason unrelated to selection.

**Files (write set):** `automl_package/examples/probreg_p3.py` (Create) ·
`automl_package/examples/capacity_ladder_results/P3/`
**Spec:** On the existing 3 toy problems: train M3 (one dedicated model per k over the frozen k
grid), score each on held-out data, record the sweep's best k. **Emit the positive control first**
— `automl_package/examples/capacity_ladder_results/P3/m3_control.json`, recording whether M3's
per-k scores reproduce §3.2's finding pattern — and check it before training or reading any M1
number (MASTER Decision 14). Train M1 (k-dropout) and record the arbiter's chosen k. Report
**(a)** quality at matched k — this re-runs §3.2's check inside the new harness so both halves come
from one artifact — and **(b)** agreement between the two chosen k's, which is the untested half.
Both across the same seeds as §3.2 so the numbers are comparable.
**Doctrine:** MASTER Decision 14 — run the known-good arm first, alone. Here that is M3 at the k
§3.2 already characterised; it must reproduce before M1 is scored. **MASTER Decisions 9 (trajectory
discipline) and 16 (exonerate optimisation before blaming architecture) both bind**: every trained
model (M1's k-dropout net and M3's per-k dedicated nets) reports its full held-out trajectory, its
convergence flag is trajectory-verified with `hit_cap=False`, and any arm that looks like it lost is
run through the escalation ladder (LR sweep → clipping → warmup → init scheme → normalization)
before being recorded as an architecture finding rather than an optimization one.
**Non-goals:** no real data (P4); no baselines (P4); do not re-tune the arbiter's neighbourhood
width.
*Orchestration:* parallel: no · deps: P1, PA, PB, P7, **FP-12** (the fixed-σ scorer AND fixed-σ
training — without it M1 and M3 both select on a learned variance) · **✅ the §1 M3 candidate-set
ruling is SETTLED (user, 2026-07-21, MASTER Decision 25 — grid spans the ladder, bypass included);
this dep is CLEARED.** *(It was previously stated only in prose above this task — a dependency-driven
dispatcher would have seen the deps clear and dispatched P3 straight past the ⛔ banner. Carried into
the deps field 2026-07-21 after an audit caught exactly that gap, and now discharged here rather than
silently dropped.)* (the arbiter must be correct, M1's
mechanism must exist, and the selection-set fraction must be frozen before M1's choices mean
anything) ·
tier: sonnet high · scale: static · shape: execution · verify:
`test -f automl_package/examples/capacity_ladder_results/P3/m3_control.json` and
`python -c "import json; d=json.load(open('automl_package/examples/capacity_ladder_results/P3/m3_control.json')); assert d['reproduces'] is True, d"`
exits 0 **before** any M1 number is read (Decision 14); then
`ls automl_package/examples/capacity_ladder_results/P3/*.json` shows one file per (problem, seed),
each containing `m1_chosen_k`, `m3_chosen_k`, `per_k_held_out_scores`, `held_out_trajectory`, and
`hit_cap: false`.

### P4 — real data + baselines

⛔ **Inherits P3's block on the M3 candidate-set ruling (§1).** Same reason: the reference's grid must
be able to return the answer the cheap arms can return, or the comparison is not like-for-like. The
plain-network baseline is a separate question from the bypass rung — do not conflate them.

**Files (write set):** `automl_package/examples/probreg_benchmark.py` (Create) ·
`automl_package/examples/capacity_ladder_results/P4/`
**Spec:** The battery in `docs/probreg_benchmark/benchmark_spec.md` **as corrected by P0** — M1,
M2, M3 plus the live baseline set (`MASTER.md` Decision 3 correction, 2026-07-20): **one tree model
(LightGBM), a plain single-output NN (the key control), and linear regression (the floor)**.
XGBoost and CatBoost were dropped the same day and must not appear in the driver. On the datasets
that spec freezes. The Kepler question (§14.3) is resolved by the pre-authorised default in §3.5,
pre-registered before the run.

**Binding: the driver READS §3.6's constants from their artifacts at startup and FAILS LOUDLY if any
is missing.** No default may be silently substituted — a battery that runs at an unjustified
selection fraction or router shape produces exactly the unattributable result this strand exists to
prevent. The startup read is the mechanism; a comment saying "remember to use PB's fraction" is not.
Each results JSON records the constants it ran under, so any table row can be traced to the study
that justified its settings.
**Trajectory discipline (MASTER Decision 9) and optimization-first (MASTER Decision 16) both bind**
on every model trained here, same shape as P3: full held-out trajectories, trajectory-verified
convergence flags, `hit_cap=False`, and the escalation ladder (LR sweep → clipping → warmup → init
scheme → normalization) run before any arm is called an architecture loss.
*Orchestration:* parallel: no · deps: P0, **P0-b1**, P1, P2, PA, **PB, PC**, P3, **P8** · tier: sonnet high ·
scale: dynamic · shape: execution · verify: with one §3.6 constant artifact deliberately
hidden/renamed, the driver exits non-zero (the prove-it-fails rule) — restore the artifact and
re-run; then `ls automl_package/examples/capacity_ladder_results/P4/*.csv automl_package/examples/capacity_ladder_results/P4/*.json`
shows the per-dataset CSV and per-model JSON outputs, each JSON containing a `constants` key naming
the study artifact it traces to, plus `held_out_trajectory` and `hit_cap: false`.

### P5 — explain the middle-k coherence cost

**Files (write set):** `docs/plans/capacity_programme/shared/p5_middle_k_coherence.md` ·
`automl_package/examples/capacity_ladder_results/P5/`
**Spec:** 8 of 9 cases show a middle-k coherence failure (§2). Establish the mechanism. Candidate
to falsify first: middle k's get the least effective gradient share under uniform k-dropout while
the largest k is reinforced by every prefix. If true, the fix is a non-uniform k schedule and that
is a design decision for the user, not an improvisation.
**Prior art to falsify against FIRST (added 2026-07-21):** this candidate is width's W1 diagnosis
restated — the per-sample uniform draw starves rungs, recorded with numbers at
`docs/plans/width_dial_2026-07-11/EXECUTION_PLAN.md:119-129` and fixed there by the W2 sandwich,
with the slimmable-networks literature basis at
`docs/plans/width_dial_2026-07-11/nested_architecture_research_2026-07-11.md:104`. This strand's
plan carried no reference to that record and had budgeted P5 as fresh opus-tier discovery. **P7's
coherence re-check under the all-rungs schedule (Decision 20) is the cheap discriminating
experiment**: if the middle-k failures disappear under P7, the mechanism was the schedule and this
task reduces to recording that with the two runs side by side; P5's discovery budget is spent only
on whatever survives P7.
**Trajectory discipline (MASTER Decision 9) and optimization-first (MASTER Decision 16) both
bind**: before any middle-k pattern is attributed to architecture, show the escalation ladder (LR
sweep → clipping → warmup → init scheme → normalization) was exhausted on the middle-k arms, with
full held-out trajectories on disk, not endpoints.
**Non-goals:** the schedule itself is already ruled (Decision 20 → P7); this task explains what
survives P7, it does not redesign the schedule.
*Orchestration:* parallel: yes · deps: P3, P7 · tier: opus xhigh (discovery-shaped) · scale: static ·
shape: research · verify:
`test -f docs/plans/capacity_programme/shared/p5_middle_k_coherence.md` exits 0, and
`grep -oE "automl_package/examples/capacity_ladder_results/P5/[A-Za-z0-9_./-]+\.json" docs/plans/capacity_programme/shared/p5_middle_k_coherence.md`
returns at least one path, and each returned path resolves with `test -f`.

### P11 — the head-structure comparison (NEW 2026-07-21, user-instructed)

**The question.** All three head layouts admit a k-ladder. Two produce per-class components; the
third does not (§1's table). Does the component structure actually buy anything — in selection
quality or in fit — once capacity is held equal?

**⚠️ THE THIRD ARM IS A MECHANISM CONTROL, NOT A COMPETITOR.** `SINGLE_HEAD_FINAL_OUTPUT` is in this
battery because it is the one configuration where the prefix guarantee is **vacuous** — there are no
components for "the first c rungs are a genuine c-component model" to be about. It is the negative
control for this strand's central mechanism claim, and it must be labelled that way in every table.
**If it ties or wins at matched capacity, that is not a layout preference — it is evidence the
component story is not what is doing the work, and it HALTS the battery** (see halt conditions).

**What varies, and what is held constant.** Exactly ONE thing varies: the head layout. Same trained
network per cell, same schedule (the all-rungs schedule — **deps P7**, see below), same masked-softmax
prefix mechanism, same toys, same seeds, same convergence gate, same selection rules.

**The two selection modes are FREE and both are reported.** Global-k and per-input-k are two
post-hoc READS of the same trained network (§1: M1 and M2 are read off the SAME network). They cost
no extra training. ⇒ **This battery is 3 trained networks per cell, each read twice — NOT a 3×2 grid
of models.** Any plan to train separately per selection mode is a misreading of §1.

**🚫 THE SWEEP REFERENCE IS OUT OF SCOPE.** M3's job is to establish the ceiling for the *pinned*
model; the layout question is answerable entirely from the cheap arms. Running a dedicated per-k
reference per layout is where cost would actually explode, for no added information. It also inherits
§1's unresolved M3 candidate-set question, which this battery must not be blocked behind.

**⚠️ PARAMETER-MATCHING IS MANDATORY — the arms are NOT matched as configured, and the gap GROWS with
k.** Measured at the root 2026-07-21 by instantiating the modules at the standard head config
(`hidden_layers=1, hidden_size=32`, probabilistic heads):

| k | `SEPARATE_HEADS` | `SINGLE_HEAD_N_OUTPUTS` | `SINGLE_HEAD_FINAL_OUTPUT` | <!-- numcheck-ignore: parameter counts computed from the module definitions, not a run ledger; reproduce with the snippet in this task's verify -->
|---:|---:|---:|---:|
| 2 | 260 | 228 (0.88×) | 162 (0.62×) |
| 5 | 650 | 522 (0.80×) | 258 (0.40×) |
| 10 | 1300 | 1012 (0.78×) | 418 (0.32×) |
| 12 | 1560 | 1208 (0.77×) | 482 (0.31×) |

Separate heads grow linearly in k (one head per class); the third layout grows only through its input
dimension, so at the top of the ladder it runs on **roughly a third of the parameters**. **Unmatched,
the control is WASTED**: "no components" and "a third of the capacity" both predict losing, and the
run cannot distinguish them. ⇒ Raise the single-head layouts' `hidden_size` until parameter counts are
comparable **at each k**, and **report the realised counts in every results table** so the match is
checkable rather than asserted. *(Doctrine reuse, not a new rule: the depth strand already requires its
two readout arms be decided empirically and parameter-matched — MASTER Decision on the readout ruling.)*

**Cells — the mechanical tier rule, applied.** This battery reports a chosen k as correct AND makes
per-input claims, so §0.5's rule gives **tiers 1 + 2 + 3** (tier 4 is reserved for the headline
comparison and the report). That is **7 toys × 3 seeds × 3 layouts = 63 training runs**, each read
both ways → 126 selection results, no extra training.

**⭐ The decisive cell is in TIER 2, not tier 3.** `make_broad_unimodal` is a single Gaussian matched
in mean AND variance to the bimodal toy, and its role is quoted in §0.5 as: a moment-matching model
"cannot tell them apart; the genuine mixture objective can." The no-component layout has no per-class
parameters to express mixture structure with, so moments are essentially all it can read. **If
component nesting does real work, this matched pair is where it must show up.** An earlier draft of
this battery deferred tier 2 to a second stage — that would have deferred the one cell that answers
the question. Do not re-defer it.

**Halt conditions — decided BEFORE the run, in the shape `width.md` WSEL-16 already ratified:**
1. **The two component-producing layouts are indistinguishable** (difference within twice a bootstrap
   SE **on the fixed-σ mixture log-likelihood** — §0.5's readout, NOT squared error) → the §1 pin
   stands, drop the middle layout from the programme, **stop**. Do not proceed to any further layout
   work.
2. **The no-component control ties or wins at MATCHED capacity** → **HALT and escalate to the user.**
   Write up what was measured; propose nothing. This contradicts the mechanism the strand is built on
   and must not be absorbed as a routine result.
3. **The control loses at matched capacity** → the intended outcome: first direct positive evidence
   the components carry the work. Record it as such; it becomes report content.

**♻️ PRIOR ART — what was checked before writing this task (§0.5's REUSE-FIRST rule requires this
block; an earlier draft of P11 omitted it and was non-compliant).** Searched
`automl_package/examples/` for existing head-structure work. **Two drivers exist and neither answers
this question, but both are reusable:**
- `automl_package/examples/head_structure_diagnostic.py` — *"I2: run head-structure diagnostics
  across a grid of ProbReg configs"*, with results at
  `automl_package/examples/head_structure_results/head_structure.csv` (32 rows). **Why it does not
  answer P11:** it compares only TWO layouts (`separate_heads`, `single_head_n_outputs` — the
  no-component layout is absent, so the mechanism control was never run); it runs on the **legacy**
  `heteroscedastic`/`bimodal` toys, which §0.5 rules are not a tier and on which **no new cell may be
  produced**; it uses fixed k with no ladder; and it scores MSE under an `nll`/`beta_nll` training
  loss — a learned-variance objective. **Its results are citable as history, labelled legacy; they
  are not a baseline for P11.**
- `automl_package/examples/head_degeneracy_diagnostic.py` — same two layouts plus a monotonic
  variant; same legacy-toy limitation.
- **REUSE, do not reimplement:** `automl_package/utils/head_diagnostics.py::analyse_head_structure`
  computes the per-head structural checks (mirror/middle-flat/mean-separation/dead-head). **P11 must
  call it** and report those flags per arm, because they turn "layout A beat layout B" into *why* —
  a dead or degenerate head is a mechanism, not a score. This is the difference between a result and
  a finding.

**⚠️ METRIC AND VARIANCE STATUS — stated explicitly, per the §0.5 correction.** Score every rung on
the **fixed-σ mixture log-likelihood**; report RMSE alongside as a point-accuracy column only.
**Never score on squared error** (structurally blind to k — it would flatten the curve and hand every
layout the same answer, which on THIS battery would read as "head structure does not matter" when the
metric simply cannot see it). **Never score on a learned `log_var`** — and note this battery makes
that trap unusually easy to fall into, because the three layouts *differ in how they produce σ*
(components combined by law of total variance, versus a single head emitting `log_var` directly).
**Scoring at fixed σ is precisely what neutralises that as a confound**: the learned variance never
enters the score, so the arms are compared on structure, not on their variance machinery.

**⚠️ REGULARISATION STATUS — declared, per MASTER Decision 21.** This battery trains **unregularised**
(no weight decay, dropout, norm layers, or mini-batching), identical to the rest of the strand's
research training. **It therefore inherits P8's block:** if the reopened P8 finds that regularisation
moves the selected k, this battery's selection comparison is confounded in the same way and must be
re-derived. ⇒ **deps include P8** (see orchestration line).

**Files (write set):** `automl_package/examples/probreg_p11.py` (Create) ·
`automl_package/examples/capacity_ladder_results/P11/`
**Non-goals:** no new architecture; no change to any head module (this battery CONFIGURES them, it
does not edit them); no sweep reference; no tier 4; no deletion; no edit to `probabilistic_regression.py`
(FP-12 and P7 own that file).

*Orchestration:* parallel: **the 63 cells are independent COMPUTE — fan out on cells, but the ROOT
runs the grid backgrounded and the worker only AUTHORS the driver** (per-cell CLI with
`--layout/--toy/--seed`, one JSON per cell, `--summarize`, `--selftest`; explicit non-goal: do not run
the full grid) · **deps: FP-12** (the scorer AND every `NESTED` guard — the layout gate this battery relies on now
lives there, P10 having been merged into it) **· P7** (schedule + fixed-σ training — running this on
the retired draw or the old objective would make every number citable only as old-schedule/
old-objective) **· P8** (its block, if it triggers, confounds this battery's selection
comparison identically) · tier: sonnet high for the driver · scale: static (63 cells) ·
shape: execution ·
**verify:**
```bash
cd /home/ff235/dev/MLResearch/automl
# (a) the parameter-match is REAL, not asserted -- reproduce the table above, then the matched config
AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python -m automl_package.examples.probreg_p11 --selftest
# (b) every cell landed: 7 toys x 3 seeds x 3 layouts
[ "$(ls automl_package/examples/capacity_ladder_results/P11/*.json | wc -l)" = "63" ] && echo CELLS-OK
# (c) every result row carries its realised parameter count and its selection mode
# (d) no cell hit the convergence cap (hit_cap must be false everywhere)
AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python -m automl_package.examples.probreg_p11 --summarize
```

### P6 — report — ⏸ **PARKED (user, 2026-07-21). Gated behind a joint results review.**

⛔ **NO REPORT WORK STARTS UNTIL THE USER AND THE ROOT HAVE REVIEWED THE RESULTS TOGETHER.** The user's
instruction: when all the work in BOTH live strands is done, they will walk the results with the root
to confirm the numbers make sense and that nothing has been missed — **and only then is the report
written.** The purpose is explicit: a comprehensive report is expensive, and writing one on results
that turn out to be wrong or incomplete wastes that effort. **This gate applies equally to `width.md`
WSEL-10.** See MASTER Decision 23.

**Files (write set):** `docs/reports/probreg_kselection/`
**Spec:** Extend the existing report to the three-model framing and the real-data results, via the
`research-report` skill, authored as the user, no AI/tool provenance (MASTER Decision 10). The
existing text predates §1 and its framing must be **corrected, not appended to**.

**The studies are REPORT CONTENT, not internal detail (user, 2026-07-20)** — they get their own
sections ahead of the comparison, because they are what license its settings:
- **How much data the selection step needs (PB).** The curve, the saturation point, and the chosen
  fraction. Where an arm was still improving at the largest fraction, say so — and say plainly that
  its comparison result is then a floor, not a verdict.
- **Whether the router's architecture matters (PC).** The sensitivity table and an invariance verdict.
- **Whether the cheap global read reaches the expensive sweep's answer (P3).** Both halves stated
  separately — same quality at matched k, and same chosen k — because they are different claims and
  only one of them was ever previously tested.
- **The middle-k coherence cost (P5)**, including that it was unexplained until P5 and what P5 found.

**Honesty clauses, binding:** report M3's full cost next to its accuracy (the efficiency claim is a
ratio and this is its denominator); state every constant the battery ran under and which study set
it; and carry the negative results — the data floor, any non-invariance, the middle-k cost — in the
body, not an appendix.
*Orchestration:* parallel: no · deps: P4, P5 · tier: opus xhigh (report + verdict) ·
scale: static · shape: execution · verify: the `research-report` skill's own cold-read gate
(procedural, run per the skill); then `grep -c "P[0-9A-Z]*" docs/reports/probreg_kselection/*.md`
is nonzero (studies cited by task ID, not restated from memory); then for each §3.6 constant name
(selection-set fraction, M2 data-limited flag, router hidden/depth/epochs/lr, labelling tolerance,
`k_max`), `grep -q "<constant name>" docs/reports/probreg_kselection/*.md` exits 0.

---

### P10 — gate the head layout under `NESTED` — ➡️ **MERGED INTO `flexnn-package.md` FP-12 (root, 2026-07-21). DO NOT DISPATCH THIS TASK.**

**Why merged, not deleted.** MASTER Decision 29's guard is the same shape as this one — a constructor
error plus a search-space repair under `NESTED`, in the same file
(`automl_package/models/probabilistic_regression.py`), with the same test target
(`tests/test_phase3_dynamic_k.py`). Two tasks editing one constructor's validation block, in
sequence, under a session-scoped write guard, for one logical rule, is pure overhead — and it splits
the rule across two documents so no reader sees it whole. **FP-12 now carries every
illegal-under-`NESTED` configuration in one guard**, this one included.

**Nothing in the spec below is dropped** — it is retained verbatim as FP-12's requirement for the
head-layout clause, and as the pre-registration that clause is judged against. **Orchestration effect:
the serial chain on the model file drops from three tasks to two (FP-12 → P7).**

*(Original spec retained below, unchanged, as FP-12's input.)*

**Why this exists.** §1's head-layout block rules that the nesting guarantee is a statement about
per-class components, and that one of the three layouts produces none. `predict_distribution`
already refuses that layout for precisely this reason
(`automl_package/models/probabilistic_regression.py:792-794`); the nested ladder does not. This task
makes the two paths consistent. **It is a guard, not a new mechanism — no behaviour changes for any
configuration the programme actually runs** (every driver already pins `SEPARATE_HEADS`).

**Files (write set):** `automl_package/models/probabilistic_regression.py` ·
`tests/test_phase3_dynamic_k.py`

**Spec — three changes, all mechanical:**
1. **Constructor gate.** `n_classes_selection_method=NESTED` with
   `regression_strategy=SINGLE_HEAD_FINAL_OUTPUT` raises `ValueError` at construction, beside the
   existing `uncertainty_method` gate (`:163-168`), and for the same reason. The message must name
   the cause — no per-class `(mu, sigma)` is produced, so "the first c rungs are a genuine
   c-component model" is vacuous — and cite `predict_distribution`'s matching refusal, so the next
   reader sees the two are one ruling.
2. **Search-space repair.** `get_hyperparameter_search_space` (`:530`) must not offer a layout the
   constructor will reject: when `NESTED` is set, the `regression_strategy` choices are the two
   component-producing layouts. **This is the live exposure** — a tuning run is the only way the
   programme could currently reach the blocked layout.
3. **Label the middle layout.** `SINGLE_HEAD_N_OUTPUTS` under `NESTED` is legal but is a **labelled
   comparison arm, never a default** (§1's table): its components are computed from the zero-padded
   vector, so truncation leaks into every surviving component. Record this in the `NESTED` member's
   docstring in `automl_package/enums.py`; no runtime restriction.

**⚠️ Do NOT "fix" this by widening the guarantee.** The tempting alternative — redefining the prefix
property so it covers a layout with no components — would retire the one criterion that makes the
ladder auditable. The guarantee stays as written; the configurations that cannot satisfy it are
excluded.

**Non-goals:** no change to any sanctioned-layout numerical path (this must be a pure guard — every
existing certified number stays reproducible); no new selection mechanism; no touching the three
read-out methods; no deletion.

*Orchestration:* parallel: no (single file, and P7 also writes it) · deps: **FP-11** (the strategies
module moved) — **and must be serialised against P7**, which writes the same file · tier: sonnet high ·
scale: static · shape: execution ·
**verify:**
```bash
cd /home/ff235/dev/MLResearch/automl
# (a) the blocked pairing raises at CONSTRUCTION, not mid-fit
AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python - <<'PY'
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel
from automl_package.enums import NClassesSelectionMethod, RegressionStrategy, UncertaintyMethod
try:
    ProbabilisticRegressionModel(input_size=3, n_classes=4,
        uncertainty_method=UncertaintyMethod.PROBABILISTIC,
        n_classes_selection_method=NClassesSelectionMethod.NESTED,
        regression_strategy=RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT)
    print("P10-FAIL: no error raised")
except ValueError as e:
    assert "SINGLE_HEAD_FINAL_OUTPUT" in str(e), e
    print("P10-GATE-OK")
PY
# (b) the two sanctioned layouts still construct AND train unchanged
# (c) the search space no longer offers the blocked layout when NESTED is set
# (d) no certified number moved: the suite matches its known result exactly
AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python -m pytest tests/test_phase3_dynamic_k.py -q
AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python -m pytest tests/ -q   # root runs this at wave end
```
(a) prints `P10-GATE-OK`; (d) must be **366 passed / 2 failed (the two accepted heteroscedastic
tests) / 1 skipped** — no new failures and no newly-passing tests.

---

## 5. Non-goals for this strand

No new selection strategies. No change to `RegressionStrategy` or the loss family. No revival of
in-training k selection as a primary (MASTER Decision 13) — it may appear only as a labelled
comparison arm. No variance-programme work (MASTER Decision 2).
