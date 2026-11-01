# ProbReg — shared work

Tasks and rules that serve all three ProbReg variants. Moved verbatim from the former <!-- citecheck-ignore -->
`docs/plans/capacity_programme/probreg.md` on 2026-10-21; each block's `<!-- from … -->` marker gives <!-- citecheck-ignore -->
its original lines. Index and task order: `docs/probreg/PLAN.md`.

<!-- from docs/plans/capacity_programme/probreg.md lines 15-137 -->
## 0.5 CODE ORGANISATION AND CLEANUP — the SAME standard as `docs/flexnn/width/plan.md` §3.9 (user, 2026-07-21)

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

**Only the capacity MECHANISM joins the FlexNN home.** `docs/flexnn/package/plan.md` FP-11 (TASK ZERO) moves
`n_classes_strategies.py` to `automl_package/models/flexnn/strategies/n_classes.py`, so that every
capacity-selection strategy — width, depth, k — sits together. **`probabilistic_regression.py` and
`probabilistic_regression_net.py` STAY where they are**, because ProbReg is a MODEL that *uses* a
capacity dial, not a capacity mechanism; it belongs beside `classifier_regression.py`, its sibling.
FlexNN is the umbrella for the mechanisms, not for every model that consumes one. **P7 therefore deps
on FP-11** (it writes the file FP-11 moves) — one move, not two, which is the same argument that made
FP-11 task zero.

### The rules (identical to `docs/flexnn/width/plan.md` §3.9 / WSEL-17 — stated here so this strand is self-contained)

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
`docs/flexnn/p9-cleanup-manifest.tsv` (Create)

**Spec (execution-level).**
- [ ] **Step 1 — one row per artifact.** Every `probreg*`/`classifier*` driver, every `*_results/`
  directory, both shell launchers. Columns: path · verdict (KEEP / RELOCATE / SHIM / DELETE) · reason ·
  replacement path · the four checks' outcomes.
- [ ] **Step 2 — DELETE-ELIGIBLE only if ALL FOUR hold**, each checked by a recorded command:
  (1) not in `docs/flexnn/package/PROTECTED.tsv`; (2) no live plan line cites it
  (`grep` the plan dir at execution time); (3) nothing under `tests/` and no surviving module imports
  it; (4) superseded by a NAMED replacement that EXISTS on disk, path written in the row.
  **"Looks old" is not a reason.** Everything else is KEEP, RELOCATE or SHIM.
- [ ] **Step 3 — gate evidence is KEEP BY RULE.** Any JSON backing a delivered report
  (`docs/probreg/fixed_k/reports/report_a/report_a_probreg_toys.md`) or cited by a ledger marker stays. Deleting
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
of these drivers) **· AND P3, P4, P8, PB, PC ALL CLOSED** · tier: sonnet high for Steps 1-6,
**root + ATTENDED for Step 7** · scale: static · shape: execution ·

⚠️ **P9 COLLIDES WITH FIVE OPEN TASKS BY CONSTRUCTION (added 2026-07-21 after an execution-level
audit).** P9's write set is the glob `automl_package/examples/probreg_*.py` — which **matches every
driver the open tasks create**: `probreg_p3.py` (P3), `probreg_benchmark.py` (P4), `probreg_p8.py`
(P8), `probreg_pb.py` (PB), `probreg_pc.py` (PC). Nothing in the previous
deps line prevented a wave-partitioner from scheduling P9 **concurrently** with any of them, and a
relocate or `git mv` racing another task mid-write is the exact single-writer failure this programme
has a hook to catch. P9's own eligibility checks would probably block a *deletion* of a live-cited
file — but they do not protect a **rename**. **⇒ P9 runs LAST in this strand, after all five close.**
verify: (1) `shared/p9-cleanup-manifest.tsv` exists and every DELETE row records all four checks;
(2) every relocated study's JSONs resolve at their new path and no driver writes outside
`capacity_ladder_results/`; (3) neither shell launcher contains the string `/home/`;
(4) `AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m pytest tests/ -q` matches the pre-cleanup result
exactly — no new failures, no newly-passing tests; (5) no `PROTECTED.tsv` path deleted or renamed.


<!-- from docs/plans/capacity_programme/probreg.md lines 594-661 -->
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
| **PC**: NOT invariant | adopt the **smallest** plateau configuration as THIS STRAND'S default, record it in `automl_package/examples/capacity_ladder_results/PC/frozen.json` under `new_default`, re-run PB at the new router. **This strand never writes `automl_package/models/common/distilled_router.py`** — any change to the SHARED default is landed by the root via `docs/flexnn/package/plan.md` FP-5 *(ruling 2026-07-21: three strands were each pre-authorised to "freeze globally" a file none may write — the classic false-parallel)* | old → new + why |
| **P2**: the undocumented failing test | if it is a stale bar, write the acceptance note; if it is a real regression, fix it. **Never loosen the bar to make it pass.** | either way, a note on disk |
| **ceiling binds** (selected k = k_max) | re-run that cell at `k_max = 20`; report the raise | which cells |
| **Kepler feature leakage** (`benchmark_spec.md` §14.3, was PARK-1) | drop the two leaky features, and pre-register what this dataset tests | the pre-registration, before the run |
| a spec section contradicts §1 | **§1 wins**; fix the spec in the same turn | the correction |

**HALT and ask — these four only:**
1. A **positive control fails** (MASTER Decision 14) — the protocol is then the defect, not the arms.
2. A study comes back **incoherent rather than merely negative** (e.g. PB's curve is non-monotone
   beyond noise) — that is a broken instrument, and running the battery on it wastes the budget.
3. Any change to **§1's model definitions**. They are the user's, not the run's.
4. Anything **irreversible or outward-facing** (deleting artifacts, publishing, **pushing to
   `origin`**). *(Amended 2026-07-21, user: COMMITTING per the `docs/flexnn/PLAN.md` branch protocol —
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
`docs/probreg/shared/reports/probreg_kselection/probreg_kselection.md` §3.2 re-run through the same harness — pair
it with the dedicated per-k models of M3, which are trained independently and are exactly the
non-shared implementation this rule asks for.

**Compilation note.** Once PA–PC are closed and every task below is decision-complete, this strand
is compilable to a deterministic Workflow (fan out on the sweeps, serialise the writes through the
root) rather than relying on dispatcher discipline — which is the shape the unattended run should
take.


<!-- from docs/plans/capacity_programme/probreg.md lines 691-979 -->
### P0 — make the definitions single-sourced

**Files (write set):** `docs/probreg/shared/benchmark_spec.md` **only**
**🚫 NOT in the write set: `docs/flexnn/PLAN.md`** — ROOT-ONLY (MASTER naming
key, "SHARED FILES ARE ROOT-ONLY"). Three sibling tasks (FP-0, WSEL-0, DSEL-0) need MASTER edits in
this same wave; concurrent writers produce contradictory text in one file. **Deliverable instead:**
emit the exact MASTER naming-key text verbatim in this task's report, for the root to apply.
**Spec:** Rewrite the `docs/flexnn/PLAN.md` naming key to point at §1 of this file instead of restating a
definition. Rewrite the benchmark spec's model section to §1's three models: all `NESTED`, M1 =
arbiter-selected global k, M2 = distilled per-input k, M3 = per-k sweep reference. Delete the
"differ in exactly one thing" claim about the old pair and re-state it truthfully for the new
arms. Record the three superseded definitions as corrections, not silent edits.
**Non-goals:** no code; no change to metrics, datasets, or the tuning protocol beyond what §1
forces.
**Status: ✅ DONE 2026-07-20, all three verify conditions executed and shown passing.**
`docs/flexnn/PLAN.md`: naming key delegates here and states no definition; strand index carries this file;
Corrections entry records the three-way contradiction and its organisational root cause.
`docs/probreg/shared/benchmark_spec.md`: §2 rewritten to the three-model set with a new §2.0
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
verify: (i) `grep -n "docs/probreg/PLAN.md" docs/flexnn/PLAN.md` shows the naming key
delegating here — **DONE**; (ii) the M1 code block in `docs/probreg/shared/benchmark_spec.md`
sets `NESTED`, not `NONE`; (iii) that spec defines three models, and `grep -c "arbiter"` on it is
no longer 0. *(Do NOT verify by grepping for the retired phrase — the Corrections entry quotes it
deliberately, so such a grep always fires. This line originally did exactly that and was
unrunnable.)*

### P0b — the σ-scope sweep through the benchmark spec

Removed 2026-10-22 (mixture framing; docs/probreg/cleanup_plan.md).

### P0-b1 — ✅ DONE (premise found already satisfied on disk; verified 2026-07-21) — propagate the M3 training ruling into the benchmark spec

**✅ No dispatch needed.** All four named spots were checked against disk 2026-07-21: each already
carries the §1 formulation with dated corrections, and the spec file has no uncommitted diff. The
task's own premise ("the spec still carries the superseded wording") was stale when written — the
same repair-pass defect class as D2's note. Spec kept below for the record; verify re-written (the
original `grep -n "train identically"` would flag the CORRECTED text, which legitimately contains
the live claim "M1 and M2 train identically" — it tested the wrong invariant).
**Verify as re-written, all three executed 2026-07-21, all pass:**
`grep -n "All three ProbReg models train identically" docs/probreg/shared/benchmark_spec.md`
returns nothing; `grep -n "SAME as M2/M3" docs/probreg/shared/benchmark_spec.md` returns
nothing; the §2.3 M3 build text sets `NClassesSelectionMethod.NONE` (`benchmark_spec.md:284`,
`:293`).

**Files (write set):** `docs/probreg/shared/benchmark_spec.md`
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
verify: `grep -n "train identically" docs/probreg/shared/benchmark_spec.md` returns only
struck-through or explicitly-superseded text; `grep -n "SAME as M2/M3" docs/probreg/shared/benchmark_spec.md`
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
- **M1 = `fit_global_selector`**, built on `all_rung_fit_loss`
  (`automl_package/models/flexnn/strategies/n_classes.py:226`) — **never** on
  `held_out_arbiter_advantage` (§1 ⚠️, D5). Selects via FP-9's already-landed shared primitive
  `automl_package/utils/capacity_selection.py`; the bootstrap-SE/tolerance rule was IMPORTED, not
  re-derived (the third-copy failure this programme exists to stop).
- **M3 = `fit_sweep_selector`**: one ORDINARY (`NClassesSelectionMethod.NONE`) model per k, scored
  on the held-out remainder, SAME selection rule.
- **`selection_fraction` is a constructor parameter** (default 0.15), not a baked-in constant — the
  parameter `PB` is about to measure.

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
explicit constructor errors — correct: M1 reads a prefix-nested per-rung fit-term curve, which is
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
  `all_rung_fit_loss`** (`automl_package/models/flexnn/strategies/n_classes.py:226`)
  — the per-rung `(batch, n_classes)` held-out fit-term table — never on
  `held_out_arbiter_advantage`, which is a different, per-input readout that cannot answer a
  global "which k" question (§1 ⚠️, D5).
- **M3** exists only as a loop inside an example script
  (`automl_package/examples/report_a_benchmark.py:331`), not as a model.

**Shape (settled by this repair pass, 2026-07-20 — supersedes the earlier "propose a `KSelection`
enum" text, which collided with `docs/flexnn/package/plan.md` FP-3's `CapacitySelection` and is retired):**
PA **consumes** `CapacitySelection` (`automl_package/enums.py`, owned by `docs/flexnn/package/plan.md` FP-3)
— it does not define an enum of its own. FP-3 ships exactly two members, `FIXED` and `PER_INPUT`;
**`GLOBAL_CHEAP` and `GLOBAL_SWEEP` are added by the task that builds their mechanism** — PA adds
`GLOBAL_CHEAP` in the same change that builds M1's selector, and `GLOBAL_SWEEP` in the same change
that builds M3's, never as a member whose mechanism raises `NotImplementedError` (the
`WidthSelectionMethod.DISTILLED` trap `docs/flexnn/package/plan.md` FP-3.a retires; do not recreate it here
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
   `docs/probreg/shared/reports/probreg_kselection/probreg_kselection.md` §3.2** — a difference counts as real only
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
itself, its name, or its cross-family contract — file a finding against `docs/flexnn/package/plan.md` FP-3
instead of patching it from here. Do not touch `automl_package/enums.py`.
*Orchestration:* parallel: no (touches the same file as P1) · deps: `docs/flexnn/package/plan.md` FP-3,
`docs/flexnn/package/plan.md` FP-9, P1 · tier: sonnet high · scale: static · shape: design → execution ·
verify: `grep -rn "inference_mode" automl_package/ tests/` returns nothing (clean break, no shim);
`grep -n "class KSelection" automl_package/enums.py` returns nothing (no home-grown enum);
`grep -n "CapacitySelection" automl_package/models/probabilistic_regression.py` is non-zero; a test
asserts a `CapacitySelection.PER_INPUT`-constructed, router-fitted model routes with no caller flag;
`AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m pytest tests/test_phase3_dynamic_k.py -q` exits 0.


<!-- from docs/plans/capacity_programme/probreg.md lines 1045-1108 -->
### P7 — the WRITE side: migrate M1/M2 to the all-rungs schedule, re-validate ONCE (MASTER Decision 20)

The σ half of this task (MASTER Decision 26, with its suite re-baseline and its dependency on FP-12's scorer) was removed 2026-10-22
(mixture framing; docs/probreg/cleanup_plan.md): P7 is the schedule migration only.

**Files (write set):** `automl_package/models/flexnn/strategies/n_classes.py` ·
`automl_package/models/probabilistic_regression.py` ·
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
· deps: PA, **`docs/flexnn/package/plan.md` FP-11** *(added 2026-07-21 — FP-11 moves the k-strategy module to
`automl_package/models/flexnn/strategies/n_classes.py`; running P7 first would mean moving it twice)* ·
tier: sonnet high · scale: static · shape: execution →
verify: `AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m pytest tests/test_phase3_dynamic_k.py -q`
green, including a NEW test asserting the `ALL_RUNGS` loss equals the mean of the per-rung NLLs
computed independently (every rung receives gradient every step); demonstrate the test is real by
truncating the loss to the drawn-rung-only form, showing it FAIL, restoring;
`test -f automl_package/examples/capacity_ladder_results/P7/frozen.json` exits 0 and the file
carries `schedule`, `middle_k_failures_old`, `middle_k_failures_new`; every per-cell JSON carries
`held_out_trajectory` and `hit_cap: false`.


<!-- from docs/plans/capacity_programme/probreg.md lines 1237-1262 -->
### P1 — ✅ DONE (landed 2026-07-20 in commit `84ad94d`; recognised 2026-07-21) — fix D2 (the arbiter units mismatch)

**✅ No dispatch needed.** The exact fix and the exact regression test this task specifies were
found already committed (`84ad94d`) during the 2026-07-21 repair audit; the test was re-executed at
the root: 2 passed. The prove-it-fails ceremony is waived with reason: fix and test landed in one
commit, so the FAIL state no longer exists to demonstrate — and the test asserts on the per-sample
likelihood (the quantity the fix changes), satisfying the doctrine's intent. Spec kept below for
the record.

**Files (write set):** `automl_package/models/probabilistic_regression.py` ·
`tests/test_phase3_dynamic_k.py`
**Spec:** Transform `y` to the fitted space inside `_per_sample_fit_at_k` when
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


<!-- from docs/plans/capacity_programme/probreg.md lines 1409-1440 -->
### P4 — real data + baselines

⛔ **Inherits P3's block on the M3 candidate-set ruling (§1).** Same reason: the reference's grid must
be able to return the answer the cheap arms can return, or the comparison is not like-for-like. The
plain-network baseline is a separate question from the bypass rung — do not conflate them.

**Files (write set):** `automl_package/examples/probreg_benchmark.py` (Create) ·
`automl_package/examples/capacity_ladder_results/P4/`
**Spec:** The battery in `docs/probreg/shared/benchmark_spec.md` **as corrected by P0** — M1,
M2, M3 plus the live baseline set (`docs/RULES.md` Decision 3 correction, 2026-07-20): **one tree model
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


<!-- from docs/plans/capacity_programme/probreg.md lines 1605-1724 -->
### P6 — report — ⏸ **PARKED (user, 2026-07-21). Gated behind a joint results review.**

⛔ **NO REPORT WORK STARTS UNTIL THE USER AND THE ROOT HAVE REVIEWED THE RESULTS TOGETHER.** The user's
instruction: when all the work in BOTH live strands is done, they will walk the results with the root
to confirm the numbers make sense and that nothing has been missed — **and only then is the report
written.** The purpose is explicit: a comprehensive report is expensive, and writing one on results
that turn out to be wrong or incomplete wastes that effort. **This gate applies equally to `docs/flexnn/width/plan.md`
WSEL-10.** See MASTER Decision 23.

**Files (write set):** `docs/probreg/shared/reports/probreg_kselection/`
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
(procedural, run per the skill); then `grep -c "P[0-9A-Z]*" docs/probreg/shared/reports/probreg_kselection/*.md`
is nonzero (studies cited by task ID, not restated from memory); then for each §3.6 constant name
(selection-set fraction, M2 data-limited flag, router hidden/depth/epochs/lr, labelling tolerance,
`k_max`), `grep -q "<constant name>" docs/probreg/shared/reports/probreg_kselection/*.md` exits 0.

---

### P10 — gate the head layout under `NESTED` — ➡️ **MERGED INTO `docs/flexnn/package/plan.md` FP-12 (root, 2026-07-21). DO NOT DISPATCH THIS TASK.**

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
per-class heads, and that one of the three layouts produces none. `predict_distribution`
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
   c-class model" is vacuous — and cite `predict_distribution`'s matching refusal, so the next
   reader sees the two are one ruling.
2. **Search-space repair.** `get_hyperparameter_search_space` (`:530`) must not offer a layout the
   constructor will reject: when `NESTED` is set, the `regression_strategy` choices are the two
   layouts that produce per-class heads. **This is the live exposure** — a tuning run is the only way the
   programme could currently reach the blocked layout.
3. **Label the middle layout.** `SINGLE_HEAD_N_OUTPUTS` under `NESTED` is legal but is a **labelled
   comparison arm, never a default** (§1's table): its per-class outputs are computed from the zero-padded
   vector, so truncation leaks into every surviving class's output. Record this in the `NESTED` member's
   docstring in `automl_package/enums.py`; no runtime restriction.

**⚠️ Do NOT "fix" this by widening the guarantee.** The tempting alternative — redefining the prefix
property so it covers a layout with no per-class heads — would retire the one criterion that makes the
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


