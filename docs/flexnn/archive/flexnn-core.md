> **ARCHIVED 2026-10-21 — never dispatch from this file.** Its two live tasks, F6 and F7, moved to
> `docs/flexnn/moe/plan.md`; its R-stage roadmap moved to `docs/flexnn/transformer/plan.md`. What <!-- citecheck-ignore -->
> remains is kept for the record.

<!-- from docs/plans/capacity_programme/flexnn-core.md lines 1-87 -->
# Strand: FlexNN core — MoE comparison (F6) + unified report (F7)

> ## ⚠️ SCOPE REDUCED 2026-07-20 — this file now holds TWO live tasks: **F6** and **F7**.
>
> The split MASTER had been carrying as pending is **applied.** `docs/flexnn/package/plan.md` **FP-0**
> recorded a disposition for every one of this file's 15 task headings in
> `docs/flexnn/package/history/CORE-DISPOSITIONS.tsv`, and the root applied them. **Every heading is still here** — the
> thirteen non-live ones now carry a pointer instead of a spec, so no incoming citation breaks and
> nobody has to guess where the work went.
>
> | disposition | tasks | where the work lives now |
> |---|---|---|
> | **retain (LIVE)** | F6, F7 | here |
> | moved | F2, F3, F13 | `docs/flexnn/package/plan.md` |
> | moved | F5 and its repair block | `docs/flexnn/depth/plan.md` (full text in its History section) |
> | moved | F12 | `docs/probreg/PLAN.md` |
> | superseded (landed) | F0, F1, F4, F8, F9, F10, F11 | verified on disk 2026-07-20 |
>
> **Why this mattered:** this file was 54KB holding four workstreams, and that grab-bag is what let
> three contradictory ProbReg model definitions coexist unnoticed. **One strand file per workstream
> is the rule** (user, 2026-07-20). It also held live tasks whose write sets collided with the four
> strand plans — a task here could have created a file a newer strand owns.
>
> **The `## Dispatch waves` section near the bottom is STALE and carries its own banner. Do not
> dispatch from it.** The live order is each strand file's own §4.

**Ratified by user 2026-07-18 (live), and still binding where it is not superseded above:** (1) **FlexNN is the umbrella** — all width/depth/joint
capacity work IS the FlexNN programme; the rigorous mechanisms get refactored INTO the package
FlexNN family. (2) Plan is execution-level; **orchestration runs in a SEPARATE session** — nothing
in this file is built in the planning session. (3) A **unified research report** covering all
flexible-architecture work (width + depth) with a **MoE comparison** supersedes the separate
reports (b) and (c) of `docs/flexnn/moe/plan.md` (M4/M5 fold into F7 here; M3 battery is UNGATED from
G-JOINT and rescoped as F6). (4) The **feedforward-depth demonstration** is pursued (spec-first).

**G-JOINT status carried:** BLOCKED after three dead construction families; J-3/R1-with-fixes spec
authored (`docs/flexnn/joint/specs/j3_toy_design.md` §9), direction (Option 1 vs 3) still a USER
decision. Default for THIS strand (reversible): the unified report scopes G-JOINT as an honest
open problem with the three-family post-mortem; Option 1 stays available and is NOT killed by
this plan.

**Review basis (2026-07-18, main session):** `automl_package/models/flexible_neural_network.py`
read in full. Confirmed sound: shared readout over depths (matches the refuted-transfer depth
result — shared readout WINS for depth), `hard_forward` per-depth bucketed compute savings,
router-reads-raw-input interface, cost-aware prior idea. Superseded/missing: in-training-only
selection (no distilled path — Decision 13), no width dial, no convergence gating, ELBO prior =
`linspace(3,1)` prefer-shallow (M0 measured complete depth-collapse to this prior — same trap as
ProbReg's removed linspace k-prior), stale docstring (says first `max−n` layers are identity;
implementation runs the FIRST n blocks and skips the LAST `max−n` — verified against every
strategy in `layer_selection_strategies.py` and `hard_forward`), HPO depth cap 3.

**Standing clauses & environment:** per `docs/flexnn/PLAN.md` Rules. `AUTOML_DEVICE=cpu` on every run;
`~/dev/.venv/bin/python`; convergence trajectory rule on every conclusion
([[feedback_check_loss_trajectory_before_concluding]]). Reports: byline = user, no AI provenance.

**Ledger:** `automl_package/examples/capacity_ladder_results/D_TOY_PROBES/` (F5),
`automl_package/examples/moe_comparison_results/` (F6), `docs/flexnn_unified/` (F7).

---

### Task F0: MASTER reframe — FlexNN umbrella naming + strand registration

**⤳ SUPERSEDED — the reframe has LANDED.** `docs/flexnn/PLAN.md` is retitled *FlexNN programme (flexible-capacity networks)*, carries the FlexNN-umbrella naming key, and registers every strand. Verified on disk 2026-07-20. Nothing to execute.

### Task F1: package hygiene fixes in `flexible_neural_network.py`

**⤳ SUPERSEDED — LANDED, and the residual work is now `docs/flexnn/depth/plan.md` DSEL-3's.** Verified on disk 2026-07-20: the uniform prior and its removal comment are at `automl_package/models/flexnn/depth/model.py:300`, and the HPO range is `high: 6` at `:557`. This task's write set is DSEL-3's; retaining it would put two writers on one file.

### Task F2: `FlexibleWidthNN` — port the certified width mechanism into the package

**⤳ MOVED to `docs/flexnn/package/plan.md` FP-2.** Bringing the certified width architectures into the package is owned there, together with the ruling on when the four accounting dispatch branches move. Do not execute from here.

### Task F3: `DistilledCapacityRouter` — Decision-13 selection as a package API

**⤳ MOVED to `docs/flexnn/package/plan.md` FP-5.** Router reconciliation is owned there. `automl_package/models/common/distilled_router.py` exists but is a behavioural SUBSET of the research routers (no soft-target training, no blend-likelihood path) — FP-5 owns closing or documenting that gap. Do not execute from here.

### Task F4: convergence gating — promote the trajectory rule into the package

**⤳ SUPERSEDED — LANDED.** Verified on disk 2026-07-20: `automl_package/utils/convergence.py` holds the logic and `automl_package/examples/convergence.py` is the thin re-export shim. That shim is the migration precedent `docs/flexnn/package/plan.md` FP-1 copies for the accounting module, and is now ratified programme-wide as `docs/RULES.md` Decision 19.

### Task F5: feedforward-depth pilot — spec, ADJUDICATOR GO, build, run (2 seeds)

**⤳ MOVED to `docs/flexnn/depth/plan.md`** (strand 9), which owns this object end to end. The full original text is preserved verbatim in that file's History section; the live successors are DSEL-1 (the settled diagnosis), DSEL-1b (the replacement all-rungs training scheme) and DSEL-2 (the primary claim). Do not execute from here.

### Task F5c: depth protocol repair + diagnosis (replaces F5b's run; added 2026-07-20, user-approved)

**⤳ MOVED to `docs/flexnn/depth/plan.md`; the HALT it escalated is CLOSED.** The user ruling recorded in DSEL-1 and written up at `docs/flexnn/depth/history/dsel1_nested_diagnosis.md` settles it: the failed arm had no nested structure — one loss at full depth — so the optimisation escalation ladder was aimed at the wrong problem and no re-run is owed. The staged a/b/c/d sequence is **retired, not pending.** Full text preserved in that strand's History section. Do not execute from here.


<!-- from docs/plans/capacity_programme/flexnn-core.md lines 156-181 -->
### Task F9: ProbReg dynamic-k port — the SAME refactor FlexNN is getting (added 2026-07-18)

**⤳ SUPERSEDED — LANDED and verified in BOTH directions.** The device bug and the symlog units bug are both fixed; `docs/probreg/shared/state.md` §3 D1 carries the record, and the programme lesson (*a regression test is not evidence until shown to FAIL on the unfixed code*) is now a `docs/flexnn/PLAN.md` rule. This task's write set is P1/PA's.

### Task F10: ProbReg k-selection report — the missing report (added 2026-07-18)

**⤳ SUPERSEDED — the report EXISTS.** Verified on disk 2026-07-20: `docs/probreg/shared/reports/probreg_kselection/probreg_kselection.md` and its PDF are both present. That directory is `docs/probreg/shared/plan.md` P6's write set; P6 CORRECTS the framing to the three-model set rather than appending to it.

### Task F11: roadmap groundwork dossiers R5a + R5b (promoted from the roadmap; runs Wave 1)

**⤳ SUPERSEDED — both dossiers LANDED.** Verified on disk 2026-07-20: `docs/flexnn/transformer/reports/python_instruction_datasets.md` and `docs/flexnn/transformer/reports/dgx_spark_training_envelope.md`. Roadmap groundwork; owned by no capacity strand.

### Task F8: transformer/coding-model roadmap amendment

**⤳ SUPERSEDED — LANDED.** Verified on disk 2026-07-20: `docs/applications/research_plan.md` carries the staged path and both verified anchors (arXiv:1807.03819, arXiv:2404.02258). Owned by no capacity strand.

### Task F12: ProbReg benchmark — shared-k vs variable-k vs baselines, on REAL data (added 2026-07-20, user-ratified)

**⤳ MOVED to `docs/probreg/PLAN.md`** (P0 / P0-b1 for the spec, P4 for the driver and the run). ⚠️ Its six-model set here is **SUPERSEDED** by `docs/probreg/PLAN.md` §1's three-model set, and its baseline list is superseded by `docs/RULES.md` Decision 3's correction (LightGBM, a plain single-output NN, linear regression; XGBoost and CatBoost dropped). Do not execute from here.

### Task F13: refactor debt from the F2/F3/F4 package port (added 2026-07-20)

**⤳ MOVED to `docs/flexnn/package/plan.md`.** Item 1 (the duplicated cheapest-within-tolerance labeller) is FP-9's; item 2 (the coexisting router MLPs) is FP-5's; item 3 (the `FlexibleWidthNN` port) is FP-2's. All three are specified there in more detail than here. The parity guard and the dead-code removal recorded under this heading are DONE and stay recorded. Do not execute from here.

---


<!-- from docs/plans/capacity_programme/flexnn-core.md lines 230-291 -->
## Dispatch waves — ⛔ STALE. NEVER DISPATCH FROM THIS SECTION.

**Frozen 2026-07-20 when the dispositions were applied.** Every wave below names tasks that have
since **landed, moved to another strand, or been superseded** — F0/F1/F4/F8/F9/F10/F11 are done,
F2/F3/F13 belong to `docs/flexnn/package/plan.md`, F5 and its repair block belong to `docs/flexnn/depth/plan.md`, and F12 belongs to
`docs/probreg/PLAN.md`. A dispatcher working from this list would re-run completed work and execute tasks
another strand now owns, on write sets that strand's tasks are actively using.

**The live dispatch order is each strand file's own §4 Tasks section**, partitioned into waves by
`deps:` + write-set overlap. Only **F6** and **F7** remain live in THIS file, and their order is
stated at their own headings. Retained verbatim below solely as the historical record of how the
refactor was sequenced.

<!-- citecheck-ignore: frozen historical wave list, superseded by the strand files -->

- **Wave 1 (parallel, disjoint writes):** F0 (MASTER) · F1 (flexnn fixes) · F5a (FF spec author →
  review) · F8 (research_plan §5) · F11 (R5a/R5b groundwork dossiers — NOT optional; user agreed
  to run them early, 2026-07-18)
- **⛔ ADJUDICATOR GO on F5 spec** (see F5a — user away; park F5b on UNSOUND/PI-fix, never ask).
  **The G-JOINT Option 1/3 decision stays a USER decision and is NOT taken this run** — it blocks
  no F-task (the report scopes G-JOINT as open regardless); batch it. Context for the eventual
  decision: a passing feedforward substrate may host an entanglement-free joint toy (width =
  hidden units, depth = layer count, independent draws — needs its own confound analysis).
- **Wave 2 (parallel):** F2 (width module) · F4 (convergence promote) · F5b (pilot build+run) ·
  F10 (k-selection report; pure writing, disjoint)
- **Wave 3:** F3 (router; touches both model files — after F1/F2 land)
- **Wave 4 (parallel, disjoint writes):** F6 (battery; ~1–2 h CPU detached, ≤3–4 concurrent
  heavy per environment rules) · F9 (ProbReg dynamic-k port; touches only
  `probabilistic_regression.py` + its tests)
- **Wave 5:** F7 (report; cold-read loop)

Estimated orchestration cost: ~11 worker dispatches (mostly sonnet) + 1 design + adjudicator
review/cold-reads (opus-tier) + 2 research dossiers; CPU batteries ~2–3 h wall detached.

## Unattended-run contract (binding on the orchestrating session; ratified 2026-07-18)

1. **No user questions, ever, mid-run.** Every gate in this plan is either mechanical (verify
   command) or adjudicator-resolved (F5a GO). Anything that would genuinely need the user —
   an UNSOUND F5 verdict, a PI-level design fork, an irreversible/outward-facing action — is
   PARKED and logged under `RESUME.md` "### Batched user questions"; the run continues.
2. **Never idle while any unblocked task remains.** After each wave (or any single task)
   completes, immediately re-derive the unblocked set from `deps:` + disjoint write sets and
   dispatch ALL of it concurrently. A parked task blocks only its own dependents. The run ends
   only when every remaining task is blocked-by-park or done.
3. **Drive to conclusion = verified on disk.** A task is done when its `verify:` line passes,
   run by the orchestrator against disk — never on a worker's claim.
4. **Result-vs-bar ambiguity → adjudicator, never improvisation.** Pre-registered bars are
   evaluated mechanically; if a result is genuinely ambiguous against its bar, dispatch the
   adjudicator for a verdict and log it. No bar is adjusted after its run (Decision 9).
5. **Orchestrator tier: Sonnet/xhigh is SUFFICIENT** — the plan pins every judgment-heavy
   subtask to the adjudicator/opus tier (F5a design + review, F7 cold-reads, rule 4 above); the
   orchestrator's own job is dispatch + disk verification, which is execution-shaped. Route via
   `executing-plans`/`subagent-driven-development` per MASTER; scale is static (11 tasks) — no
   Workflow needed.
6. **End-of-run (in order): focused commit pass is PRE-APPROVED** (authored as user, exclude
   RESUME/CHANGELOG/memory/CLAUDE.md and any AI-instruction files) → `/checkpoint --final`
   (with `--tidy` if end of day) → batched user questions listed at the top of RESUME. Stop all
   agents before the boundary (enumerate-probe; commit before any kill).

## Done ledger

*(orchestrator appends: task · date · evidence path)*

