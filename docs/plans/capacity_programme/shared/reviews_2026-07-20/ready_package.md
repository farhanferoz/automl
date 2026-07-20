# Autonomous-execution readiness review — `docs/plans/capacity_programme/flexnn-package.md`

Reviewed read-only 2026-07-20. Every plan citation below was opened and checked.

---

## Section 1 — verdict table

| Task | Verdict | Biggest gap |
|---|---|---|
| FP-0 | **NOT-DISPATCHABLE** | Its `verify:` asserts a state of `flexnn-core.md` that is **factually unreachable** — that file retains 7 further live tasks (F5c, F8, F9, F10, F11, F12, F13), not "only MoE + unified report". A worker driving to green will move or delete ProbReg/roadmap workstreams it was never asked to touch. |
| FP-1 | **DISPATCHABLE-WITH-FIX** | The one hard decision (what happens to the four `@executed_flops.register(nwn.*)` dispatch branches) is **explicitly delegated to the worker** ("decide in the task"). Its `verify:` greps `models/` only and cannot see the actual violation vector (bare-name `import nested_width_net` via a `sys.path` hack). |
| FP-2 | **NOT-DISPATCHABLE** | `verify:` = "certified width numbers reproduce … to a stated tolerance" — no command, no source-of-truth file, no seed list, no runtime budget, and the **worker states its own tolerance**. That is the exact unattributable-comparison failure this review exists to prevent. Symbol partition of `nested_width_net.py` is undecided beyond "toy generators". |
| FP-3 | **NOT-DISPATCHABLE** | Does not name the enum or its members, while **two other live plans have already named incompatible member sets** for the same enum in the same file. The three named modes have **no member for "do not select"**, which is today's default in all three families. Two of three modes are unimplemented at FP-3 time → re-creates the `NotImplementedError` trap the same task retires. |
| FP-4 | **DISPATCHABLE-WITH-FIX** | "material-or-not verdict" with **no threshold, no cells, no seeds, no command** — worker sets its own bar. |
| FP-5 | **NOT-DISPATCHABLE** | **Highest blast radius in the plan.** Says "its **four** direct importers" — there are **five** (`capacity_ladder_h1`, `_s1`, `_s2`, `depth_selection_toy`, `sinc_width_experiment`), the fifth being the script that produced the certified width result. Spec ("leave re-export shims for `capacity_ladder_k6`'s router") and Non-goals ("do not delete any script router") are mutually unclear about whether k6 is gutted. |
| FP-6 | **DISPATCHABLE-WITH-FIX** | `verify:` still demands "the pre-registration question is answered in writing with the document cited" — a question the plan's own body says was resolved and carries no halt. Verify contradicts spec. |
| FP-7 | **DISPATCHABLE-WITH-FIX** | Best-specified task in the plan, but it does not mandate sweeping **both** import forms. `automl_package/examples/` has **no `__init__.py`**; nine+ scripts import each other by bare name (`import nested_width_net as nwn`). A qualified-import-only sweep will certify live modules as zero-caller and feed those false positives straight into FP-8's deletions. |
| FP-8 | **NOT-DISPATCHABLE** | Write set is "determined by FP-7's inventory" — unknown at dispatch. Deleting anything that produced a certified number is HALT #1, and **the worker has no mechanical way to decide which files did**. Under an unattended run this either halts entirely or deletes on judgment. |

---

## Section 2 — per-task detail

### FP-0 — ratify the boundary rule and take ownership — NOT-DISPATCHABLE

**1. Write set.** `flexnn-package.md` · `MASTER.md` · `flexnn-core.md`. **Collision:** `MASTER.md` is
also in the declared write set of `width.md` WSEL-0 (`width.md:225`) and `depth-selection.md` DSEL-0
(`depth-selection.md:180`). Three strands write one file. `parallel: no` is declared but scoped only
within this strand — it does not exclude the other two.

**3. `verify:` — refuted.** Two clauses:
- `grep -n "flexnn-package.md" docs/plans/capacity_programme/MASTER.md` — runnable. Currently returns
  nothing (verified: `MASTER.md` contains no occurrence), so this is a valid post-condition.
- *"`flexnn-core.md` retains only MoE + unified report as live workstreams"* — **false as a reachable
  state, and not mechanically checkable.** `flexnn-core.md` currently holds fifteen task headings:
  F0, F1, F2, F3, F4 (`:36`–`:138`), F5 (`:155`), F5c (`:217`), F6 (`:294`), F7 (`:328`), F9 (`:362`),
  F10 (`:481`), F11 (`:511`), F8 (`:537`), F12 (`:569`), F13 (`:620`). Its Done ledger (`:758`) is an
  **empty placeholder** — *"(orchestrator appends: task · date · evidence path)"* — so nothing there
  is marked done by ledger. Only three completion markers exist anywhere in the file (`:308`, `:395`,
  `:626`). Moving "the package-refactor workstream" out cannot leave MoE + report alone: F5c
  (depth protocol repair), F8 (transformer roadmap), F9/F10/F12 (ProbReg) and F11 (roadmap dossiers)
  all remain. **A worker driving this verify to green will move or delete live ProbReg and roadmap
  tasks.** This is a documentation blast-radius, and `MASTER.md:21` names `flexnn-core.md` as the
  programme's plan of record for four workstreams.

**2. Decision-completeness.** The plan never names **which task IDs** constitute "the package-refactor
workstream". The obvious candidate, `flexnn-core.md` Task F13 (`:620`, *"refactor debt from the
F2/F3/F4 package port"*), is **the same subject matter as FP-5 and FP-8 and is never mentioned in
`flexnn-package.md` at all**. Worker must guess whether F13 moves, is superseded, or is duplicated.

**6/7.** Downstream is fine (everything deps FP-0). Deps `none` is correct.

---

### FP-1 — break the circular dependency — DISPATCHABLE-WITH-FIX

**Citations verified.** `capacity_accounting.py:62-63` are exactly the two package imports claimed.
`flexible_width_network.py:290` and `flexible_neural_network.py:492` are exactly the deferred imports,
with the quoted comment verbatim. `automl_package/examples/convergence.py` is a genuine re-export shim
as described. All good.

**2. Decision-completeness — the load-bearing gap.** The spec says:

> *"The accounting branches that dispatch on examples-only width classes either move with those
> classes (FP-2) or register from the script side — **decide in the task, and state which**."*

That is the plan handing its hardest decision to a zero-context worker. The concrete situation
(`capacity_accounting.py`):
- `executed_flops` is a `functools.singledispatch` with four registrations on examples-only classes:
  `:224` `NestedWidthNet`, `:242` `SharedTrunkPerWidthHeadNet`, `:259` `IndependentWidthNet`,
  `:275` `SharedReadoutPerWidthAffineNet`.
- Those classes are reached by `capacity_accounting.py:59` — `import nested_width_net as nwn`, a
  **bare-name import that only resolves because `:55-56` inserts the examples dir into `sys.path`**.

So moving the accounting logic into `automl_package/utils/` at FP-1 time forces one of: (a) the new
package module imports examples (violating the very boundary rule this task enforces), (b) the four
registrations stay behind in the examples shim (a split-brain `singledispatch`), or (c) FP-2's class
move happens first — **but FP-2 deps FP-1, so the order forbids it.** The ordering is inverted at
exactly this seam and the plan does not see it.

**Also undecided:** the new module's name and path (`automl_package/utils/` is given as a directory,
not a file); whether the `sys.path` mutation side-effect (currently triggered inside a shipped model's
`fit_router` path) is acceptable to keep; where the characterisation test lives; what "identical
numbers" means for float outputs.

**3. `verify:` — partially runnable, and evadable.**
`grep -rn "from automl_package.examples" automl_package/models/` is runnable and currently returns
exactly the two known lines. But it **does not cover `automl_package/utils/`** — the destination — so a
worker can satisfy it while the new utils module imports examples. And it matches only the qualified
form; the actual coupling mechanism in this repo is the **bare-name + `sys.path`** form, which this
grep cannot see. "the characterisation test passes before and after with identical numbers" has no
named test path and no command. "full suite green" has no command (the repo convention is
`~/dev/.venv/bin/python -m pytest`, per `CLAUDE.md`).

**Fix to make it dispatchable:** name the module (`automl_package/utils/capacity_accounting.py`); settle
the dispatch question in the plan (recommended: keep the four `nwn` registrations in the examples shim
until FP-2, and say so); replace the verify with
`grep -rnE "from automl_package\.examples|^import (nested_width_net|capacity_ladder_|sinc_width_)" automl_package/models/ automl_package/utils/`
returning nothing, plus the exact pytest command and the exact characterisation-test path.

---

### FP-2 — bring the certified width architectures into the package — NOT-DISPATCHABLE

**Citation verified.** `nested_width_net.py:222` is `class SharedTrunkPerWidthHeadNet` — correct.
"four width-dial architectures" is correct (the four `singledispatch` registrations above).

**2. Decision-completeness.** `nested_width_net.py` holds more than four classes. Verified live
symbols used by other scripts: `make_hetero` (`hetero_width_experiment.py:11`), `make_hetero3`
(`report_a_benchmark.py:85`), `gaussian_log_likelihood` (`sinc_width_experiment.py:13`),
`train_nested_width` (`sinc_width_experiment.py:16`), `WidthSchedule.SANDWICH`
(`hetero_width_experiment.py:13`), `HETERO_R_DEFAULT` (`hetero_width_experiment.py:157`). The plan
assigns only "toy generators (`make_hetero` and friends)" to stay. **`gaussian_log_likelihood`,
`train_nested_width`, `WidthSchedule`, `HETERO_R_DEFAULT` are unassigned** — and
`train_nested_width` / `WidthSchedule` are precisely the training-schedule machinery FP-4 is about, so
"library or protocol" is a real judgment call, not a formality.

Two import styles must both keep working: bare (`import nested_width_net as nwn` — 8 scripts:
`independent_width_experiment.py:43`, `sinc_width_experiment.py:68`, `matryoshka_width_net.py:52`,
`converged_width_experiment.py:38`, `cascade_width_experiment.py:60`, `cascade_width_net.py:57`,
`hetero_width_experiment.py:52`, `moe_regression.py:70`, plus `capacity_accounting.py:59`) and
qualified (`report_a_benchmark.py:85`). The plan does not name this constraint.

**3. `verify:` — prose aspiration, not a check.**

> *"the certified width numbers reproduce through the moved classes to a stated tolerance, shown side
> by side; every existing script imports unchanged; full suite green."*

- **No source of truth named.** The certified numbers live in
  `docs/width_mse_2026-07-16/verdict_variable_width_mse.md` (verified to exist) — the plan never cites
  it. The worker must find it.
- **No command, no cells, no seeds, no runtime budget.** Reproduction runs through
  `kdropout_converged_width_experiment.py`, a convergence-driven training job.
- **"to a stated tolerance"** — the worker chooses the tolerance it will be judged against. This is
  the documented root-cause failure mode (unattributable comparison) written into the verify line.
- "every existing script imports unchanged" is checkable in principle but no command is given; the
  natural one (`python -c "import ..."` over 104 scripts) is not stated.

**4. Reversibility.** Move + shim is reversible in git; low deletion risk **provided** the worker does
not "tidy" the emptied module. Nothing in the task forbids deleting now-unused symbols from
`nested_width_net.py`.

**5. Behaviour preservation.** "no behaviour change" is an instruction, not a check. There is **no
characterisation test required here** (unlike FP-1, which does require one). Asymmetry is unexplained.

---

### FP-3 — the one selection API — NOT-DISPATCHABLE

This is the task both consuming strands block on, and it is the least decision-complete task in the
plan.

**Verified facts the task rests on — mostly correct, one wrong.**
- No `InferenceMode` type exists: confirmed (grep over `automl_package/`, `tests/` returns only string
  literals and `str` annotations).
- Three closed sets: confirmed — width `("fixed","routed")` (`flexible_width_network.py:204`), depth
  `("soft","hard","routed")` (`flexible_neural_network.py:397`), ProbReg `("soft","routed")`
  (`probabilistic_regression.py:613`, `:708`).
- **"three different defaults" is wrong.** Verified defaults: `inference_mode: str = "fixed"`
  (`flexible_width_network.py:192`), `= "soft"` (`flexible_neural_network.py:386`), `= "soft"`
  (`probabilistic_regression.py:598`). That is **two** distinct defaults, not three. Minor, but this
  plan claims every §1 line was re-verified.
- `WidthSelectionMethod.DISTILLED` trap: confirmed at `enums.py:109` (comment *"not yet landed"*),
  `flexible_width_network.py:91-95` (`NotImplementedError`), `tests/test_flexible_width_network.py:199-201`
  (`test_distilled_selection_not_implemented`) — the plan cites `:200`, the test body spans `:199-201`.
  Substantively correct.

**2. Decision-completeness — five unsettled decisions, each fatal alone.**

1. **The enum is not named and its members are not named.** The spec describes three modes in prose
   ("a cheap global held-out read, a per-input distilled router, or an expensive per-value sweep").
   Two other live plans have already named this same enum, **incompatibly**:
   - `width.md:257` — *"One enum passed at construction selecting `SHARED` / `PER_INPUT` / `SWEEP`."*
   - `probreg.md:350` — *"one enum `KSelection` with `GLOBAL_ARBITER` / `PER_INPUT_ROUTER` / `GLOBAL_SWEEP`."*

   A zero-context FP-3 worker will invent a third naming. `width.md` WSEL-3's verify (`width.md:275`)
   asserts *"the enum's `SWEEP` and `SHARED` modes are reachable"* — it will fail against any other
   naming.
2. **There is no member for "do not select".** All three families default today to a non-selecting
   mode (`fixed` / `soft`). The three proposed modes cannot express it. Whether `NONE` exists, and
   what the new default is, is unstated.
3. **`FlexibleHiddenLayersNN`'s `"hard"` mode has no home.** `"hard"` is an *execution* shortcut
   (argmax depth only, documented in `CLAUDE.md` as a compute saving), not a *source* of the capacity
   value. It maps onto none of the three modes. `flexible_neural_network.py:420` implements it. The
   task does not say whether it survives, moves to a separate flag, or is deleted.
4. **`predict_uncertainty` is not addressed.** ProbReg has `predict_uncertainty(..., inference_mode=...)`
   at `probabilistic_regression.py:694`. The spec only says *"`predict` loses `inference_mode`
   entirely"*. `depth-selection.md:100-104` (DD2) separately owns the fact that
   `FlexibleHiddenLayersNN.predict_uncertainty` has **no mode parameter at all** and silently mixes
   depths — that fix is DSEL-3, in a different plan, on the same file.
5. **Two of the three modes do not exist.** The Non-goals say *"building the cheap-global and sweep
   mechanisms is the consuming strands' work"*. `width.md:281` confirms W-SHARED *"does not exist
   today"*; `depth-selection.md:356` confirms D-SHARED *"does not exist today"*. So FP-3 ships an enum
   whose `SHARED` and `SWEEP` members cannot be honoured — **which is precisely the
   `WidthSelectionMethod.DISTILLED` trap that this same task is retiring.** The plan does not say what
   those members do in the interim (raise? fall back? be absent until built?).

**3. `verify:`** — `grep -rn "inference_mode" automl_package/ tests/` returns nothing **is** runnable
and mechanically checkable. Two problems:
- It is **broader than the write set**. Occurrences exist in example scripts not named anywhere:
  `phase4_comparison.py:159,164,167,168` and `moe_flexnn_comparison.py:413-414`. The write set covers
  them only under "call sites found by grep". `moe_flexnn_comparison.py` is the driver for
  `flexnn-core.md` F6 (`:294`).
- It **conflicts with `probreg.md` PA's verify** (`probreg.md:390`), which accepts *"returns only the
  deprecated shim (or nothing)"*. Two plans, same grep, incompatible pass conditions.

**Internal contradiction.** §5 Non-goals: *"No ProbReg selection-mechanism work beyond adopting the
shared API (owned by `probreg.md`)"* — yet FP-3's write set includes
`automl_package/models/probabilistic_regression.py`.

**Contradiction with §3.5.** Row 1 of the autonomous contract pre-authorises *"never rewrite the
calling script"*; §3 item 5 repeats it. FP-3's clean break **requires** rewriting calling scripts in
`automl_package/examples/`. No branch in §3.5 covers an API change (as opposed to a move).

---

### FP-4 — resolve the width class's schedule deviation — DISPATCHABLE-WITH-FIX

**Citation.** The plan cites `flexible_width_network.py:15-16` for *"specialised here to sum ALL
configured widths every step rather than a sampled subset"*. The quoted phrase actually begins on
**line 14** and runs to 16. Trivial drift; substance is correct and the docstring says exactly that.

**3. `verify:`** *"a matched comparison on the certified cells, both schedules, same seeds, on disk;
an explicit material-or-not verdict."* Not runnable. Missing: which cells (the certified cells are not
enumerated anywhere in this plan), how many seeds, what "material" means numerically, where on disk,
what command. **The worker chooses the bar it is judged against** — the same defect as FP-2.

**Fix:** name the cells and the seed set, import the tolerance rule already ratified elsewhere
(twice a bootstrap standard error — `width.md:283`, `probreg.md:360`) rather than letting the worker
invent one, and name the output path under `docs/plans/capacity_programme/shared/` (that directory
exists; it currently holds `bug_audit_head.md`, `hetero_nll_diagnosis.md`, `metrics-accounting.md`).

**1. `parallel: yes (disjoint from FP-5)` is TRUE within this strand** — FP-4 writes
`flexible_width_network.py`, FP-5 writes `distilled_router.py`. **False across strands:**
`width.md` WSEL-1/2/3 all write `flexible_width_network.py`.

---

### FP-5 — reconcile the routers — NOT-DISPATCHABLE

**All four router citations verified exactly.** `capacity_ladder_k6.py:75` `class _RouterMLP`, `:92`
`def _train_router`; `capacity_ladder_t2.py:233` `class _RouterMLP`, `:256` `def _train_router`;
`depth_selection_toy.py:607` `class _VectorRouterMLP`, `:624` `def _train_vector_router`;
`distilled_router.py:84` `class _CapacityRouterMLP`, `:108` `class DistilledCapacityRouter`. The
"package version is a strict subset" claim holds — `distilled_router.py` has hard-label CE only and no
blend path, versus `capacity_ladder_t2.py:295-306` (`_blend_scores`, `_blend_nll`) and
`capacity_ladder_s2.py:86` (`_train_router_direct`, verified). `capacity_ladder_s1.py:84`
(`ARM_NAMES = ("soft", "soft_no_prior", "soft_smoothed", "hard_knee", "raw_argmax")`) confirms the
five-arm factorial. §3's protected list is factually accurate about what exists.

**4. BLAST RADIUS — the importer count is wrong.** §3 item 5 and FP-5 both say *"four scripts import
it directly and one transitively"* / *"its four direct importers"*. Verified by
`grep -rn "^import capacity_ladder_k6" automl_package/examples/*.py` — there are **five**:

```
capacity_ladder_h1.py:68
capacity_ladder_s1.py:75
capacity_ladder_s2.py:55
depth_selection_toy.py:88
sinc_width_experiment.py:67
```

`sinc_width_experiment.py` is **the script the plan itself identifies (§1.3) as having produced the
certified width result**. And it is not one transitive importer downstream but **eight**:
`hetero_width_experiment.py:53`, `cascade_width_experiment.py:61`, `converged_width_experiment.py:39`,
`depth_selection_toy.py:90`, `independent_width_experiment.py:44`,
`kdropout_converged_width_experiment.py:60`, `joint_capacity_toy.py:102` all do
`import sinc_width_experiment as sw`. A worker that builds a shim for "the four direct importers" as
instructed leaves a fifth path — the certified one — unshimmed.

**2. Decision-completeness — the central ambiguity.** Spec: *"Leave re-export shims for
`capacity_ladder_k6`'s router so its four direct importers keep resolving"* — a shim implies k6's
`_RouterMLP`/`_train_router` are **gutted and re-exported from the package**. Non-goals: *"do not
delete any script router."* §3 item 5: *"Replacing the implementation breaks the paper trail unless
the documents are updated in the same change."* Three statements, three different implied actions. A
zero-context worker cannot decide whether k6 keeps its own code, becomes a shim, or both.

Compounding: the shim direction here is **opposite** to the one `flexnn-core.md` F13 (`:640-647`)
records as correct — *"Correct end-state is the REVERSE migration: the three certified drivers
(`depth_selection_toy.py`, `joint_capacity_toy.py`, `kdropout_converged_width_experiment.py`) import
the PACKAGE function."* Two plans in the same directory, opposite migration directions, and
`flexnn-package.md` never mentions F13.

**3. `verify:`** — same defect as FP-2/FP-4: *"certified width selection numbers reproduce … to a
stated tolerance"*, no command, no source file, no seeds, worker-chosen tolerance. The one genuinely
checkable clause is *"a capability table on disk"* — but no path is given. *"no script's imports
break"* has no command; the correct one over 104 scripts is not stated.

**Uncovered doctrine.** `MASTER.md:110` Decision 15 (*"Protocol parity when reusing a substrate"*) and
F13's reading of it — *"it is a protocol-parity change requiring a before/after equivalence run, not a
drive-by edit"* — govern exactly what FP-2 and FP-5 do to certified drivers. `flexnn-package.md` never
cites Decision 15.

---

### FP-6 — independent-weights depth class — DISPATCHABLE-WITH-FIX

**Claim verified.** `automl_package/models/independent_weights_flexible_neural_network.py` has
`predict` at `:388` with **no `inference_mode` parameter**, no `fit_router`, no per-depth forward
(only `IndependentWeightsFlexibleNNModule.forward` at `:151`), and no convergence gating. Matches
`depth-selection.md:113-116` (DD4). Claim is accurate.

**3. `verify:` — one clause contradicts the task's own spec.** The verify ends with *"the
pre-registration question is answered in writing with the document cited."* The spec directly above
(`:316-317`) says that question was *"resolved at the root before dispatch … No halt remains here."*
The verify demands an artifact for a retired question. A worker driving to green will produce a
document about a settled non-issue.

**1. Write-set collision.** `independent_weights_flexible_neural_network.py` is also FP-3's write set
and **`depth-selection.md` DSEL-3's** (`depth-selection.md:308`). Three tasks across two plans.

**7. Deps.** FP-3, FP-5 — correct, and both are currently NOT-DISPATCHABLE, so FP-6 is transitively
blocked.

---

### FP-7 — complete the sweep — DISPATCHABLE-WITH-FIX

Best-specified task here: bounded, deletes nothing, output is an inventory, and *"each absence proven
by the grep that shows it"* is a real evidentiary standard.

**Two fixes needed, both safety-critical because FP-8 consumes this directly:**

1. **The sweep method must mandate both import forms.** Verified: `automl_package/examples/` has **no
   `__init__.py`** (it resolves as a PEP-420 namespace package). Scripts import each other by **bare
   name** after a `sys.path` insert — `import nested_width_net as nwn`, `import capacity_ladder_k6 as
   ck6`, `import sinc_width_experiment as sw`. A sweep that greps only
   `from automl_package.examples.X import` will report live, heavily-imported modules as zero-caller.
   Those false positives become FP-8 deletions. The task must require, per candidate module `M`:
   `grep -rn "\bimport ${M}\b\|from ${M} import\|automl_package\.examples\.${M}\|${M}\.py" automl_package/ docs/ tests/`.
2. **Search scope must include `docs/` and the results directories.** The plan's own §3 item 5 rests on
   preregistration documents citing exact function names; verified — e.g.
   `automl_package/examples/capacity_ladder_results/{H1,S1,T2,W1,W2}/PREREGISTRATION.md` all reference
   `capacity_ladder_k6`. A code-only sweep will miss every such reference.

**Minor:** the note has no named path ("a findings note under `docs/plans/capacity_programme/shared/`").
Name it, so FP-8's dependency is a file, not a description.

---

### FP-8 — the cleanup — NOT-DISPATCHABLE

**1. Write set is unknown at dispatch** — *"determined by FP-7's inventory, listed explicitly in the
task before it runs."* That is a promise that someone will complete the task specification later. In an
unattended run nobody does.

**4. Blast radius — maximal, and the guard does not close.** HALT #1 is *"Any deletion of code on §3's
protected list, or of anything that has produced a certified number."* In an unattended run a HALT
means stop. So FP-8 is either (a) a task that halts on its first action, or (b) a task where the worker
decides, unsupervised, whether a given file "has produced a certified number". **There is no
mechanical predicate for that.** 104 example scripts; results live under
`automl_package/examples/capacity_ladder_results/` in 30+ subdirectories; no file maps result
directories back to producing scripts.

**3. `verify:`** *"full suite green"* (no command); *"every certified result still reproduces from its
own artifacts"* (unbounded — no list of certified results, no reproduction command, would require
re-running the programme); *"`git diff --stat` reviewed against FP-7's list with no unlisted
deletion"* — this one **is** mechanisable and is the only real guard in the task, but "reviewed" is a
human verb and no comparison command is given.

---

## Section 3 — BLAST RADIUS

Ordered by expected damage. Each entry: where a worker following this text destroys or orphans working
code, and the missing guard.

**B1. FP-8 deletes live code because FP-7 mis-swept bare-name imports.** `automl_package/examples/` has
no `__init__.py`; inter-script imports are bare names enabled by `sys.path` inserts
(`capacity_accounting.py:55-59`, and `import X as Y` in at least 12 scripts). A qualified-import sweep
returns zero callers for modules with many. FP-8 deletes them. *Missing guard:* FP-7 does not specify
the grep pattern; FP-8 has no independent re-check before deletion.

**B2. FP-5 breaks the certified width path by shimming four importers when there are five.** The
unnamed fifth is `sinc_width_experiment.py:67` — the script §1.3 names as the producer of the certified
width result. *Missing guard:* no task requires the worker to *derive* the importer list by grep rather
than trust the plan's count.

**B3. FP-0 mutilates `flexnn-core.md` chasing an unreachable verify.** To make *"retains only MoE +
unified report"* true, a worker must remove or relocate F5c, F8, F9, F10, F11, F12, F13 — live tasks
in the programme's plan of record for other strands. *Missing guard:* none. HALT #4 covers
"irreversible or outward-facing"; editing a plan file is neither, so nothing stops it.

**B4. FP-2 / FP-3 / FP-5 edit scripts whose preregistrations cite exact symbol names.** Verified:
`capacity_ladder_results/{H1,S1,T2,W1,W2}/PREREGISTRATION.md` each reference `capacity_ladder_k6`.
§3 item 5 states the risk; **no task carries a mechanical check that a symbol named in a
PREREGISTRATION.md still resolves after the change.** *Missing guard:* a post-condition that greps
every backtick-quoted `module.symbol` out of `capacity_ladder_results/*/PREREGISTRATION.md` and asserts
each still imports.

**B5. FP-3's "clean break, no shim" silently changes the behaviour of two example drivers.**
`phase4_comparison.py:159-168` and `moe_flexnn_comparison.py:413-414` pass `inference_mode` positionally
by keyword. If the parameter is removed, these raise `TypeError` — loud, so recoverable. But if a
worker instead *keeps* a `**kwargs`-swallowing signature, the flag is silently ignored and
`moe_flexnn_comparison.py` (the F6 battery driver) silently stops routing — producing a wrong
comparison with no error. *Missing guard:* the task forbids a shim but does not forbid a
silently-accepting signature; no test asserts `TypeError` on the removed kwarg.

**B6. FP-2 "tidies" `nested_width_net.py` after moving four classes.** Six further live symbols
(`make_hetero`, `make_hetero3`, `gaussian_log_likelihood`, `train_nested_width`, `WidthSchedule`,
`HETERO_R_DEFAULT`) are unassigned by the plan. *Missing guard:* no explicit "these symbols stay"
list; only the prose "toy generators and friends".

**B7. FP-1 moves accounting into `utils/` and drags the examples dependency with it.** The verify
greps `models/` only. A utils module that does `from automl_package.examples...` or the `sys.path`
bare import passes the stated verify while inverting the boundary rule the task exists to establish.

**B8. `capacity_ladder_results/W_CASCADE/` and `W_MRL/` orphaning.** Both directories exist (verified).
§3 item 6 protects `ResidualCascadeNet` / `MatryoshkaWidthNet` in prose. Their defining files are
`cascade_width_net.py:80` and `matryoshka_width_net.py:62`, and their only caller is
`cascade_width_experiment.py:182,196`. *Missing guard:* those three filenames appear nowhere in §3 —
a worker doing FP-7/FP-8 sees a two-caller cluster with no other dependents and a "superseded"
label, which reads exactly like dead code.

---

## Section 4 — plan-level findings

### 4.1 The protected list (§3) is not mechanically enforceable

It is prose describing *concepts*. Items and what a worker can actually resolve:

| § | Names | Resolvable to a path/symbol? |
|---|---|---|
| 1 | "The blend-likelihood router evaluation (`capacity_ladder_t2.py` and the s1/s2/h1 blend machinery)" | File yes; **"the s1/s2/h1 blend machinery" is undefined** — verified functions are `t2:295 _blend_scores`, `t2:304 _blend_nll`, `s1:273` blend reads, `s2:80` blend loss. A worker must find these by inspection. |
| 2 | "The direct-objective router trainer (`capacity_ladder_s2.py:86`)" | **Yes** — `:86 def _train_router_direct` verified. The only precise entry. |
| 3 | "The five-arm label-construction factorial (`capacity_ladder_s1.py`)" | File yes, symbols no (`s1:84 ARM_NAMES`, `s1:96 _ARM_RECIPE`, `s1:283` runner). |
| 4 | "The group word-problem composers" (two files) | Files yes, symbols no. |
| 5 | "The scalar-input router (`capacity_ladder_k6.py`)" | File yes; **importer count wrong** (four stated, five actual). |
| 6 | `ResidualCascadeNet`, `MatryoshkaWidthNet` | Symbols yes; **files never named** (`cascade_width_net.py`, `matryoshka_width_net.py`, `cascade_width_experiment.py`). |

**Concrete enforceable form:** replace §3 with a checked-in manifest,
`docs/plans/capacity_programme/shared/PROTECTED.tsv`, one row per protected unit —
`path <TAB> symbol-or-* <TAB> reason <TAB> certified-artifact-dir` — and make it a **precondition of
every task that touches code**: before any `git rm` or symbol deletion, the worker runs a single
script that fails if the diff touches a manifest row. That converts §3 from "be careful" into a gate.
Seed rows from the verified facts above (`capacity_ladder_k6.py:75,92`; `capacity_ladder_s2.py:86`;
`capacity_ladder_s1.py:84,96`; `capacity_ladder_t2.py:233,256,295,304`; `depth_composition_toy.py`;
`depth_graded_toy.py`; `cascade_width_net.py:80`; `matryoshka_width_net.py:62`;
`cascade_width_experiment.py`).

### 4.2 Cross-plan defects (top findings)

**X1 — FP-3 and `width.md` WSEL-2 are the same task, written twice, with different specificity.**
Side by side:

| | FP-3 (`flexnn-package.md:244-273`) | WSEL-2 (`width.md:253-275`) |
|---|---|---|
| write set | `enums.py`, `flexible_width_network.py`, `flexible_neural_network.py`, `independent_weights_*.py`, `probabilistic_regression.py`, tests, call sites | `enums.py`, `flexible_width_network.py`, `tests/test_flexible_width_network.py`, call sites |
| enum members | **unnamed** | `SHARED` / `PER_INPUT` / `SWEEP` |
| `predict` | loses `inference_mode` entirely, clean break, no shim | identical wording |
| DISTILLED trap | retire it and its test | retire it (WD2) |
| selection fraction | must be a parameter | must be CONFIGURABLE |
| deps | FP-1 | **WSEL-1** — *not FP-3* |

`flexnn-package.md:17-19` claims *"Neither of them defines an API — that is why this file exists."*
**That is false for `width.md`.** WSEL-2 defines it, names the members FP-3 does not, and does not
declare a dependency on FP-3. Whichever runs first wins; the other's verify fails. `depth-selection.md`
is clean here — `:21` correctly says *"owned by `flexnn-package.md`. This strand consumes that API; it
does not define it."*

**X2 — a third, differently-named enum is already ratified in `probreg.md`.** `probreg.md:350` names
`KSelection` with `GLOBAL_ARBITER` / `PER_INPUT_ROUTER` / `GLOBAL_SWEEP`, marked
*"✅ decisions settled 2026-07-20 — DISPATCHABLE"*, write set includes `automl_package/enums.py`, deps
`P1` — **no dependency on FP-3**. `width.md:266-268` even instructs its own worker to *"Coordinate the
enum's home with PA; if PA has already landed one, extend it"* — coordination by hope. Three plans,
one file, three member vocabularies, no arbiter. `flexnn-package.md` §5 disclaims ProbReg work while
FP-3's write set includes `probabilistic_regression.py`.

**X3 — the consuming strands assume a *working* API; FP-3 delivers two dead members.**
`width.md:281` (W-SHARED *"does not exist today"*) and `depth-selection.md:356` (D-SHARED *"does not
exist today"*) are the build tasks; `width.md:295` and `depth-selection.md:364` build the sweeps. So
after FP-3, `SHARED` and `SWEEP` name nothing. FP-3's stated purpose includes killing exactly this
pattern (`DISTILLED` naming an unlanded feature). **The plan does not say what the unbuilt members do
in the interim.**

**X4 — `flexnn-core.md` F13 is FP-5 + FP-8 with the opposite migration direction.** F13 (`:620-660`)
covers the same four router implementations and the same duplicated
`_cheapest_within_tolerance_labels`, and prescribes the **reverse** direction (certified drivers import
the package function) under `MASTER.md:110` Decision 15's before/after equivalence discipline.
`flexnn-package.md` §3.5 row 1 and §3 item 5 prescribe the forward direction (*"never rewrite the
calling script"*). `flexnn-package.md:5-6` claims precedence in a disagreement, but FP-0 never names
F13 as the thing being superseded, so both remain live.

**X5 — cross-plan write-set collisions (none of the `parallel:` flags account for these).**

| File | Tasks claiming it |
|---|---|
| `automl_package/enums.py` | FP-3 · `width.md` WSEL-2 · `probreg.md` PA |
| `automl_package/models/flexible_width_network.py` | FP-1, FP-2, FP-3, FP-4 · WSEL-1, WSEL-2, WSEL-3 |
| `automl_package/models/flexible_neural_network.py` | FP-1, FP-3 · `depth-selection.md` DSEL-6 |
| `automl_package/models/independent_weights_flexible_neural_network.py` | FP-3, FP-6 · DSEL-3 |
| `automl_package/models/probabilistic_regression.py` | FP-3 · `probreg.md` PA |
| `automl_package/examples/capacity_accounting.py` | FP-1 · `width.md` WSEL-5 |
| `docs/plans/capacity_programme/MASTER.md` | FP-0 · WSEL-0 · DSEL-0 |

Every `parallel:` flag in `flexnn-package.md` is judged **within this strand only**, and is true at
that scope. At programme scope the flags are misleading, and the single-writer discipline in
`~/.claude/CLAUDE.md` will block the second worker mid-task.

### 4.3 Branch gaps — decisions arising mid-run covered by neither a default nor a HALT

1. The enum's name and member set (FP-3) — §3.5 has no row.
2. What happens to `FlexibleHiddenLayersNN`'s `"hard"` execution mode (FP-3).
3. Whether `predict_uncertainty` also loses `inference_mode` (FP-3).
4. Where the four `@executed_flops.register(nwn.*)` branches live after FP-1 — the plan explicitly
   defers this to the worker.
5. Which `nested_width_net.py` symbols move and which stay, beyond "toy generators" (FP-2).
6. Whether `flexnn-core.md` F13 moves, is superseded, or stays (FP-0).
7. The numeric tolerance for every "reproduces to a stated tolerance" verify (FP-2, FP-4, FP-5) — the
   programme has a ratified rule (twice a bootstrap standard error, `width.md:283`, `probreg.md:360`)
   that this plan never imports.
8. Runtime budget for the reproduction runs (FP-2, FP-4, FP-5). None stated. Subagents may not
   background jobs (`~/.claude/CLAUDE.md`), so an unbudgeted convergence run will hit the tool timeout
   and orphan a process.
9. Whether a *modification* (not deletion) of a certified-result-producing script is permitted. §3.5
   row 1 covers moves; FP-3 requires modification; HALT #1 covers only deletion. **Gap.**

### 4.4 HALT conditions — not exhaustive, and #1 is unusable as written

- **#1** — *"deletion … of anything that has produced a certified number."* There is no mechanical
  predicate for "produced a certified number". No file maps `capacity_ladder_results/*` back to
  producing scripts. This halt is unenforceable and therefore, in an unattended run, inert.
- **#3** — *"any change that would make a certified result unreproducible from its own artifacts."*
  Requires the worker to predict reproducibility before acting. Unusable as a gate.
- **#4** — *"Anything irreversible or outward-facing (deleting artifacts, publishing, committing)."*
  Clear and enforceable. The only one of the four that is.
- **Missing halts:** (a) a verify line that cannot be satisfied (FP-0 needs this); (b) discovery that
  another live plan claims the same file (X5); (c) a reproduction run exceeding its budget; (d) an
  importer list in the plan disagreeing with grep (FP-5's four-vs-five).

### 4.5 Citation integrity

Every code path and line cited in the plan was opened. **All resolve; all say what the plan claims.**
Specifically confirmed: `capacity_accounting.py:62-63`; `flexible_width_network.py:290`;
`flexible_neural_network.py:492`; `capacity_ladder_k6.py:75,92`; `capacity_ladder_t2.py:233,256`;
`depth_selection_toy.py:607,624`; `distilled_router.py:84,108`; `enums.py:109`;
`flexible_width_network.py:91-95`; `tests/test_flexible_width_network.py:200`;
`nested_width_net.py:222`; `capacity_ladder_s2.py:86`;
`docs/plans/width_dial_2026-07-11/cascade_execution_plan_2026-07-11.md:1` (banner verbatim: *"⛔
FROZEN. NEVER DISPATCH FROM THIS FILE."*, and its §1.6 "width content superseded" reading is correct).
`docs/plans/capacity_programme/shared/` exists. `capacity_ladder_results/W_CASCADE/` and `W_MRL/` exist.

Three **factual** inaccuracies (not rotted links):
1. **"four scripts import it directly"** (§3 item 5, FP-5) — **five**. Safety-relevant; see B2.
2. **"three different defaults"** (FP-3) — **two** (`"fixed"`, `"soft"`, `"soft"`).
3. FP-4 cites `flexible_width_network.py:15-16` for a phrase beginning on line **14**. Cosmetic.

### 4.6 Ordering

Internally the declared order (FP-0→FP-8) is consistent with every `deps:` line. Two real problems:
- **FP-1 before FP-2 is inverted at the `singledispatch` seam** (see FP-1 detail): FP-1 cannot cleanly
  land the accounting module in `utils/` while the four dispatched classes still live in `examples/`,
  which is FP-2's job.
- **FP-3 is deliberately early** *"because `width.md` and `depth-selection.md` block on this"* — but
  `width.md` WSEL-2 does not declare that dependency (deps: WSEL-1), and `probreg.md` PA does not
  either (deps: P1). The early placement buys nothing unless X1/X2 are resolved first.

---

## Section 5 — the three changes that most increase the chance this executes correctly unattended

**1. Settle the selection enum in ONE place, by name, before anything dispatches — and make the two
other plans consume it.** Write the exact enum name, the exact member list (including the member that
means "no selection", which none of the three drafts has), the default per family, the fate of
`"hard"`, and the fate of `predict_uncertainty`, into `flexnn-package.md` FP-3. Then edit `width.md`
WSEL-2 and `probreg.md` PA to **consume** it (deps: FP-3, write set excluding `enums.py`) rather than
define it. Also decide what `SHARED`/`SWEEP` do between FP-3 and their build tasks — if the answer is
"raise `NotImplementedError`", say so explicitly, because that is the trap FP-3 exists to remove and it
should be a conscious choice, not a rediscovery. **Without this, three workers write one enum with
three vocabularies and the two consuming strands' verifies fail against whichever lands first.**

**2. Turn every "reproduces to a stated tolerance" verify into a runnable command with a
plan-supplied bar.** Applies to FP-2, FP-4, FP-5, FP-8. Each needs, in the plan text: the file holding
the certified numbers (`docs/width_mse_2026-07-16/verdict_variable_width_mse.md` for width — the plan
never cites it), the exact cells and seeds, the tolerance **imported from the already-ratified rule**
(twice a bootstrap standard error), the exact invocation, the exact output path, and a wall-clock
budget. As written, four of nine tasks let the worker choose the bar it is judged against — the
documented root cause of the previous failed run, reproduced verbatim in the verify lines.

**3. Replace §3's prose with a machine-checked manifest and gate FP-7/FP-8 on it.** Create
`docs/plans/capacity_programme/shared/PROTECTED.tsv` (path · symbol · reason · certified-artifact-dir),
seeded with the verified rows in §4.1 — including the three files §3 protects by concept but never
names (`cascade_width_net.py`, `matryoshka_width_net.py`, `cascade_width_experiment.py`). Then:
(a) require FP-7's zero-caller sweep to use **both** import forms plus a `docs/` + `PREREGISTRATION.md`
scan, with the exact grep in the task text; (b) require FP-8 to run a diff-vs-manifest check that fails
on any touched protected path, and to re-derive every importer list by grep rather than from plan
prose. This is the only change that mechanically prevents an unattended run from deleting working
research code.

---

## Uncertainty

- I did not execute any test, reproduction, or import of the repo — the review is static. Claims about
  what *would* break (B5's `TypeError`, B1's false zero-callers) are derived from read code, not run.
- "Certified result" provenance is asserted by the plan (§1.3) and by `MASTER.md:16-22`; I verified the
  artifact directories and the verdict file exist, not that the numbers in them came from the named
  scripts.
- `flexnn-core.md`'s per-task live/done status is inferred from an empty Done ledger (`:758`) and three
  inline completion markers (`:308`, `:395`, `:626`). If tasks are tracked done elsewhere, FP-0's
  verify may be closer to reachable than I judge — but no such tracker was found in
  `docs/plans/capacity_programme/`.
- Coverage caveat: this is a single-reviewer read of one plan plus targeted reads of three sibling
  plans. I killed false positives in the plan's own citations; I cannot certify that no *further*
  cross-plan collision exists in `depth.md`, `width-depth.md`, `flexnn-moe.md`, or `width-cert.md`,
  which I did not walk.

## Recommendation

**FIX-FIRST — do not dispatch this plan unattended in its current form.** Three tasks (FP-0, FP-3,
FP-8) cannot be executed correctly by a zero-context worker as written, and two (FP-2, FP-5) will
produce results whose bar the worker sets. FP-5's importer undercount and FP-7's import-form gap are
each independently sufficient to destroy or orphan working research code. FP-1, FP-4, FP-6, FP-7 are
recoverable with the stated fixes. Apply the three Section-5 changes, re-check FP-0's verify against
`flexnn-core.md`'s actual contents, and re-review FP-3 and FP-8 specifically before any dispatch.
