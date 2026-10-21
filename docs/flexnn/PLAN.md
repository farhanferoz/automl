# FlexNN — stream plan (flexible architectures)

The one plan for the flexible-architecture stream. **On hold since 2026-10-20** while the owner
focuses on ProbReg. Assembled 2026-10-21 from the former programme index <!-- citecheck-ignore -->
`docs/plans/capacity_programme/MASTER.md`; text moved, not rewritten (`<!-- from … -->` markers). <!-- citecheck-ignore -->
Rules and decisions that bind more than one stream moved to `docs/RULES.md`; the ProbReg decisions
moved to `docs/probreg/PLAN.md`. Decision numbers are unchanged.

## Sub-plans (current)

| Sub-plan | Status |
|---|---|
| `docs/flexnn/width/plan.md` — width selection and the architecture comparison (WSEL tasks) | open, on hold; certification record `docs/flexnn/width/history/width-cert.md` |
| `docs/flexnn/depth/plan.md` — feed-forward depth and depth selection | parked (owner, 2026-07-21); the closed substrate strand (G-DEPTH pass) is `docs/flexnn/depth/history/depth.md` |
| `docs/flexnn/joint/plan.md` — the joint width-depth dial | parked: J-1 and J-2 dead, the J-3 spec deferred |
| `docs/flexnn/moe/plan.md` — mixture-of-experts comparison; F6 and F7 appended from the former `flexnn-core.md` | M0–M2 done (2026-07-16); F6 and F7 not started | <!-- citecheck-ignore -->
| `docs/flexnn/transformer/plan.md` — transformer port roadmap (R-stages) | outlined only | <!-- citecheck-ignore -->
| `docs/flexnn/package/plan.md` — the codebase: package boundary, selection API, cleanup | open; the 2026-10-21 reorganisation removed the dead code FP-7 had inventoried; moving the example scripts is deferred |

Earlier records: `docs/flexnn/history/`, and each sub-plan's `history/`. The superseded grab-bag
`docs/flexnn/archive/flexnn-core.md` is never dispatched from. The strand index below is the July
status record, kept as it was; the table above is current.

<!-- from docs/plans/capacity_programme/MASTER.md lines 1-113 -->
# FlexNN programme (flexible-capacity networks)

Index + rules ONLY. Per-strand detail lives in the strand files. Read this file plus the ONE
strand you are working — that is the whole context. Fix this plan IN PLACE; never fork a dated
copy (gate: `test_plan_gates.py`).

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:subagent-driven-development`
> (recommended) or `superpowers:executing-plans`, driven from a strand file. Tasks carry
> checkbox steps and Orchestration lines; dispatch waves are computed from `deps:` + write-set
> overlap.

## Strand index & priority

| # | Strand file | Delivers | Gated on | Dispatchable now? |
|---|---|---|---|---|
| 1 | `docs/flexnn/width/history/width-cert.md` | Certification of arch #2 + seam/robustness analyses → **G-WIDTH** | — | **DONE — G-WIDTH PASS (2026-07-16)** |
| 2 | `docs/probreg/archive/probreg-report.md` | Report (a): ProbReg fixed-k & dynamic-k vs baselines on toys | `docs/probreg/shared/metrics-accounting.md` T-S1..S3 | **DONE — P0-P3 done, report (a) delivered** (`docs/probreg/fixed_k/reports/report_a/report_a_probreg_toys.{pdf,md}`) |
| 3 | `docs/flexnn/depth/history/depth.md` | S5 substrate (D1 **DONE**) + selection-without-oracle (D8) → **G-DEPTH = D5 ∧ D8** | **G-WIDTH ✓** | **DONE 2026-07-17 — G-DEPTH = PASS.** Substrate D5 (3/3) ∧ selection D8b (2/2: S1–S4 pass, S5 surface covariate 100%/Option-A). Verdict `verdict_per_input_depth.md` §12; D8b toy AS-RUN rebuild (L10/involutions/shared-readout, flagged). old D2/D3 retired. → `docs/flexnn/joint/plan.md` J0 |
| 4 | `docs/flexnn/joint/plan.md` | Joint 2-D capacity dial + transformer halting → **G-JOINT** | **G-DEPTH** (D5 ∧ D8) | **J0 RAN 2026-07-17 — J-1/J-2 DEAD (multi-track fold substrate fails S1); G-JOINT BLOCKED. RULED 2026-07-21 (user-delegated): Option 1 — a proper J-3 design spec — DEFERRED until DSEL-2's feed-forward question resolves and WSEL-8/DSEL-10 land; the spec gets user review before any build (the no-improvised-toy gate survives autonomy). NOT in the autonomous run.** (post-mortem in strand) |
| 5 | `docs/flexnn/moe/plan.md` | MoE build (M0-M2 early) + reports (b), (c) | build: none; comparisons+reports: **G-JOINT** | **DONE — M0-M2 DONE 2026-07-16**; M3 rescoped as flexnn-core F6; M4/M5 superseded by F7 (user 2026-07-18) |
| 6 | `docs/flexnn/archive/flexnn-core.md` | Package refactor + FF-depth pilot + unified report | — | **yes** — but see the SPLIT note below; this file is 4 workstreams in one |
| 7 | `docs/probreg/PLAN.md` | **ProbReg k-selection: models M1/M2/M3, defects, battery, report** | — | **yes — the live workstream (user, 2026-07-20)** |
| 8 | `docs/flexnn/width/plan.md` | **Width SELECTION + the ARCHITECTURE comparison, the toy suite, and the code-organisation rules.** Architecture certification in `docs/flexnn/width/history/width-cert.md` — **AMENDED 2026-07-22 to PASS WITH CAVEATS** (user ruling after adversarial re-derivation; see docs/flexnn/width/plan.md §2 ⛔ block) | **`docs/flexnn/package/plan.md` FP-11 (TASK ZERO)**, FP-3, FP-9, FP-4 | **yes — the LIVE strand, under the Decision-32 autonomous mandate.** Wave-2 order: `[WSEL-18 ∥ WSEL-16 stage-2 authoring ∥ WSEL-3 ∥ WSEL-4] → stage-2 cells + WSEL-5 → stage-3 (WSEL-18-gated) → verdict → (WSEL-6 ∥ WSEL-7) → WSEL-8 (architecture-parameterised per the WSEL-16 coupling ruling) → WSEL-17`; WSEL-13 tier 2 + small items slotted. Landed 2026-07-22: WSEL-14∥15 + follow-ups, WSEL-16 stage 1, cost probe, reviews. §3.7 sigma-fixed-at-truth · §3.8 toy suite · §3.9 no-duplication · §3.10 arbitrary-w_max rule · Decision-31 schedule/vectorisation/retention rulings. WSEL-9 ⏸ PARKED, toys-only; normalisation follow-ons ⏸ PARKED (user 2026-07-22) |
| 9 | `docs/flexnn/depth/plan.md` | **FEED-FORWARD depth (the object) + depth selection.** | — | **⛔ ENTIRE STRAND PARKED (user, 2026-07-21): "let's park depth completely". NOTHING dispatchable.** Supersedes the earlier strand-local block. Two untried levers (per-depth output layers; regularisation) and a missing literature survey are recorded in the strand header for whoever unparks it; neither lever has a written task, and writing them is a ROOT action on unpark. **Live programme = width + ProbReg only.** |
| 10 | `docs/flexnn/package/plan.md` | **The codebase**: package-vs-scripts boundary, the ONE selection API, router de-duplication, shared selection primitives, cleanup, **and FP-11 — ONE HOME under `models/flexnn/`** | — | **yes — and FP-11 is TASK ZERO for the whole programme** (user, 2026-07-21: do the reorganisation FIRST, not last — nothing is in flight to collide with, and the new width tasks create files that would otherwise be moved twice). FP-2 DONE. Boundary rule = Decision 19 |

**⚠ SPLIT PENDING on `docs/flexnn/archive/flexnn-core.md` (opened 2026-07-20).** That file is 54KB holding FOUR
workstreams — package refactor (F0–F4/F8/F13), the feed-forward depth attribution study (F5), the
MoE comparison (F6) and the unified report (F7) — plus, until today, the ProbReg tasks. **One
strand file per workstream is the rule (user, 2026-07-20);** a grab-bag file is what let three
contradictory ProbReg model definitions coexist.

**SPLIT STATUS 2026-07-20 (updated).** ProbReg → strand 7. The feed-forward depth study → strand 9
`docs/flexnn/depth/plan.md` (**NOT** `depth-ff.md` — that filename was planned and never used; do not look
for it). The package refactor → strand 10 `docs/flexnn/package/plan.md`. **`docs/flexnn/archive/flexnn-core.md` is therefore
expected to retain only F6 (the MoE comparison) and F7 (the unified report) — but it does NOT yet:
it still holds live tasks whose write sets collide with strands 7–10** (the ProbReg benchmark driver
and spec, the k-selection report directory, and refactor debt that duplicates FP-2/FP-5). **Until
`docs/flexnn/package/plan.md` FP-0 records a disposition for every remaining task there and the root applies
it, `docs/flexnn/archive/flexnn-core.md` is READ-ONLY for dispatch purposes** — nothing in it may be executed, because a
task in it may create a file a newer strand owns.

Gates are decision points written in the owning strand (evidence-backed verdict + branch).
Priority: flexnn-core (6) waves go first. `docs/flexnn/joint/plan.md` J0 is **PARKED** pending user
Option 1/3 decision. *(Strands 1, 2, 3, and 5's M0-M2 are complete; live forward order updated
2026-07-18.)*

## Naming key

- **FlexNN** = the umbrella for ALL per-input capacity work; strands are its width/depth/joint/MoE/core
  facets. **The family gets ONE home, `automl_package/models/flexnn/`, via `docs/flexnn/package/plan.md` FP-11
  (TASK ZERO, 2026-07-21).** Until FP-11 lands the code is split across four locations —
  `models/` (flat), `models/architectures/`, `models/selection_strategies/`, `models/common/` — which
  is the mess FP-11 exists to fix.
- **#1 / #2 / #3** — width architectures: #1 `NestedWidthNet` (shared trunk + shared readout,
  fails), #2 `SharedTrunkPerWidthHeadNet` (shared trunk + per-width heads, **certified winner —
  G-WIDTH PASS 2026-07-16**),
  #3 `IndependentWidthNet` (K disjoint sub-nets, positive control), plus
  `SharedReadoutPerWidthAffineNet` (minimum-seam arm). **All four are in
  `automl_package/models/architectures/nested_width_net.py`** since FP-2;
  `automl_package/examples/nested_width_net.py` is a re-export shim. *(This line previously named the
  examples path as their home — stale since FP-2 landed. FP-11 moves them again, to
  `automl_package/models/flexnn/width/architectures.py`, leaving another shim.)* **Candidate
  architectures under test are NOT here — they live in one module, `docs/flexnn/width/plan.md` §3.9.**
- **G-WIDTH / G-DEPTH / G-JOINT** — the three programme gates.
- **ProbReg models M1 / M2 / M3** — the three ways of choosing k. **Defined in ONE place:
  `docs/probreg/PLAN.md` §1. Do not restate the definition here or anywhere else.** *(This entry previously
  defined variable-k as "dynamic-k (ELBO + SoftGating)" — in-training selection, which Decision 13
  demotes to a labelled comparison arm. It was one of three mutually contradictory definitions live
  at once on 2026-07-20; see `docs/probreg/PLAN.md` §1.1.)*
- **The three ways of choosing a capacity value** — for EVERY dial (k, width, depth): a cheap global
  read, a per-input distilled router, and an expensive per-value sweep, each scored and costed as a
  complete system including its selection machinery. **Defined per dial in exactly one place:**
  `docs/probreg/PLAN.md` §1 (k), `docs/flexnn/width/plan.md` §1 (width), `docs/flexnn/depth/plan.md` §1b (depth). Do not restate here.
- **`CapacitySelection`** — the ONE selection enum, in `automl_package/enums.py`. **Owned by
  `docs/flexnn/package/plan.md` FP-3; every other strand consumes it and none declares its own.** FP-3 ships
  only the members whose mechanisms exist; a member is added by the task that builds its mechanism
  (the rule that retires the `WidthSelectionMethod.DISTILLED` trap rather than repeating it).
- **⚠ `docs/flexnn/joint/plan.md` is STALE** w.r.t. both of the above: it predates the three-way model set, the
  shared selection rule, and `CapacitySelection`. Its J-3 redesign must be re-read against strands
  8–10 before it is dispatched. It is parked pending a user ruling in any case.
- **Artifact naming (binding on every strand — a new task follows this, it does not invent).**
  Settled during the 2026-07-20 plan repair; `docs/flexnn/width/plan.md` is the worked example.
  - **Drivers**: `automl_package/examples/<strand>_<taskid_lower>.py` — `width_wsel6.py`,
    `depth_dsel8.py`, `probreg_pb.py`.
  - **Results**: `automl_package/examples/capacity_ladder_results/<TASKID>/`. **The directory name
    is HYPHEN-FREE** — `DSEL1b/`, not `DSEL-1b/` — matching all 34 pre-existing result dirs and the
    other strands. **Task IDs keep their hyphens** (`DSEL-1b` the task writes to `DSEL1b/` the dir),
    exactly as task `WSEL-6` writes to `WSEL6/`. *(Depth was the lone hyphenated holdout and was
    corrected during the repair; had it shipped, every parked forward reference to it would have
    been dead on arrival.)*
  - **Frozen constants**: `<results dir>/frozen.json` — ONE small file per study holding only the
    constants downstream tasks read, kept distinct from the per-cell result JSONs.
  - **Notes**: `docs/flexnn/<taskid_lower>_<topic>.md`.
  - **Test command**: `AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m pytest <path> -q`.
- **🔑 SHARED FILES ARE ROOT-ONLY — `docs/flexnn/PLAN.md` and `docs/flexnn/archive/flexnn-core.md`.** Four wave-1 tasks (FP-0,
  WSEL-0, DSEL-0, P0) each need a `docs/flexnn/PLAN.md` edit, and two need a `docs/flexnn/archive/flexnn-core.md` edit. They are
  **not independent** — independence is a disjoint WRITE set, not a different topic — and dispatching
  them concurrently as writers is the classic false-parallel that produces contradictory text in one
  file. **The rule: a task PRODUCES the exact text it needs added and reports it; the ROOT applies
  it.** Neither file may appear in any task's write set. `docs/flexnn/archive/flexnn-core.md` additionally stays
  READ-ONLY for dispatch until FP-0's dispositions land (`docs/flexnn/package/history/CORE-DISPOSITIONS.tsv`), which is
  the mechanism for changing it. *(FP-0 already states this rule for `docs/flexnn/archive/flexnn-core.md`; it is
  generalised here so every strand is bound by it, not just the one that noticed.)*
- **`gates_baseline.txt` is ROOT-ONLY.** A worker that needs a forward reference parked **lists it
  in its report**; the root verifies each is a real Create target named by a task — and asserted by
  that task's verify line — before parking it, and records the owning task inline. **No typo may
  ever be parked.** A worker parking its own reference would let a typo grandfather itself.
- **Ledger** — result JSONs under
  `automl_package/examples/capacity_ladder_results/` (width/depth) and the per-strand
  results dirs named in each strand. The plan holds POINTERS to ledger files, never copied
  result numbers. A line carrying a result must be marked `RESULT:` and name its `.json`.


<!-- from docs/plans/capacity_programme/MASTER.md lines 116-118 -->
1. **Charter verdicts to date** — strict nesting (#1) fails; break = shared READOUT; #2 is the
   **certified** architecture of record (**G-WIDTH PASS 2026-07-16**). Evidence:
   `docs/flexnn/width/reports/verdict_variable_width_mse.md` (§10 certification addendum; manifest §8).

<!-- from docs/plans/capacity_programme/MASTER.md lines 119-133 -->
2. **MSE-only** for all capacity strands (width/depth/joint). Variance stays parked.
   *(Amended 2026-07-17: binds the WIDTH strand only. The depth strand is CE classification per
   the `docs/flexnn/depth/history/depth.md` PIVOT; the joint strand's metric follows the toy J0 adopts.)*
   ✅ **AMENDED AND SHARPENED 2026-07-21 (user, live) — "FIXED AT THE TRUE VALUE", NOT "DROPPED".**
   The user's clarification: *"I'm not saying we should drop the functionality of setting it. I'm just
   saying just fix it at the true value."* **The variance machinery STAYS; LEARNING the variance is
   what is forbidden**, because that is what overfits. Every width run uses sigma clamped to the
   generator's true per-point value, excluded from the optimiser. Full rule, the per-tier
   implementation, and the honest statement of what it changes versus the squared error already on
   disk: `docs/flexnn/width/plan.md` §3.7. Key consequence, so nobody re-derives it: on a constant-noise toy the
   fixed-sigma likelihood **equals the squared error up to one positive constant**, so certified
   numbers stay comparable; on the two-noise-level control it becomes a **weighted** squared error
   that down-weights the noisy region 100x — a different objective from what was run there before,
   which must be labelled in any shared table, and which *sharpens* that control rather than blunting
   it.

<!-- from docs/plans/capacity_programme/MASTER.md lines 162-171 -->
5. **Depth/joint strands are structured around #2's pattern** (shared redocs/probreg/history/presentation +
   per-capacity readout heads), conditioned on G-WIDTH; written contingencies apply if it fails.
   *(Amended 2026-07-17: holds for WIDTH only — the depth transfer prediction was REFUTED 3/3;
   shared readout is the depth winner. Joint prior = per-width heads × depth-shared readout;
   see `docs/flexnn/depth/history/depth.md` and `docs/flexnn/joint/plan.md` Settled inputs.)*
6. **Depth starts with the toy, not architecture** — a depth-hungry-but-GD-learnable target
   with analytic floor; kill criterion + escalation if unconstructible.
7. **MoE config**: 8 experts, top-2 routing primary; top-1 ablation; load-balancing auxiliary
   loss; param/FLOP-matched to FlexNN/width nets. Conventions verified against Shazeer 2017 /
   Switch / Mixtral before the build brief freezes (task M1).

<!-- from docs/plans/capacity_programme/MASTER.md lines 326-353 -->
22. **The transformer port target is DEPTH; width ports only where the readout is linear (user,
    2026-07-21).** The programme's stated bet is a flexible-capacity architecture that transfers to a
    transformer. The analysis is recorded ONCE, in `docs/flexnn/transformer/specs/width_transformer_port.md`, and is
    **[ARGUMENT], not measurement** — it may not be cited as a result:
    - **What ports from width:** per-rung readouts over a nested prefix, *wherever the readout is
      linear and no normalisation intervenes* — in a transformer, the feed-forward hidden width and
      the head count. Cost: `+~44%` FFN parameters, **zero extra inference cost** (§3 of the note).
    - **What does not:** any claim that a narrow forward is a free by-product of a wide one across a
      normalisation layer. Normalisation is repairable at bounded cost (cumulative prefix statistics;
      per-rung normalisation parameters — the slimmable-networks fix), but the model-dimension case
      stays global and expensive (§5).
    - **Why depth is structurally better:** a transformer is already a stack of shape-preserving
      blocks, so a depth prefix is *literally* a sub-computation of the full forward — no vector is
      truncated, every normalisation sees a full-width vector, all rungs come off one pass, and ONE
      output projection serves every rung. Width has to engineer all three. Consistent with what we
      measured: depth's certified arm won with a SHARED readout, width with per-rung readouts.
    - **Consequences, binding:** (a) `NestedWidthNet` (arch #1) is **CLOSED — no further compute**;
      (b) width's remaining value is the mechanism account plus the unmeasured importance-ordering
      property (`docs/flexnn/width/plan.md` **WSEL-13**) and the schedule/cost sweep (**WSEL-14**) — not a broader
      width programme; (c) **cost is a first-class recorded output** from here on (parameters,
      executed FLOPs at the deployed rung, wall-clock), reusing
      `automl_package/utils/capacity_accounting.py`, never argued in prose; (d) the depth material is
      an INPUT to `docs/flexnn/depth/plan.md`, not a width claim, and is blocked behind that strand's
      failing positive control (Decision 14); (e) ⚠️ **depth has NO literature pillar** — width got
      one (`docs/flexnn/width/history/width_dial_2026-07/nested_architecture_research_2026-07-11.md`), early
      exiting in transformer stacks is a crowded area, and **no depth result may be positioned as
      novel until an equivalent survey exists.**


<!-- from docs/plans/capacity_programme/MASTER.md lines 442-447 -->
28. **Depth stays parked until both live strands close (user, 2026-07-21).** Confirms the existing
    park rather than changing it. Rationale recorded so it is not re-litigated: width's architecture
    comparison is about to test the ordering/cascade assumptions any depth task would be designed on,
    and Decision 22 already names depth as the better transformer-port target. Unparking earlier
    means designing depth's tasks on premises that are mid-test.


<!-- from docs/plans/capacity_programme/MASTER.md lines 560-614 -->
31. **WIDTH SCHEDULE + ARCHITECTURE RULINGS (user, 2026-07-22, during the width review).** Three
    parts, binding on the width strand and on any strand that later trains multi-width models:
    (a) **the ALL schedule (every width, every step) is the DEFAULT** — the sandwich survives only
    as the labelled comparability mode against already-landed ledgers; rationale: dominance, not
    trade-off (never less accurate, 6× lower mid-width variance, cost premium is an implementation
    artifact removed by (b)); (b) **the multi-head readout is VECTORISED before any further
    multi-head compute** (`docs/flexnn/width/plan.md` WSEL-18 — exact under ALL, refused under sampling);
    (c) **the multi-head architecture is RETAINED regardless of the WSEL-16 outcome** — the
    comparison selects the default recommendation, never a sole survivor, and WSEL-17's deletion
    scope is amended accordingly. Full ruling text: `docs/flexnn/width/plan.md` §Decision-20 REVISIT.

32. **AUTONOMOUS EXECUTION MANDATE — width wave-2 (user, 2026-07-22).** The user ratified the
    execution outline and mandates autonomous execution: **the root is dispatcher and verifier;
    plan-prescribed branches are EXECUTED, never re-asked; ambiguities take the reversible default
    and a logged decision; questions are batched to the end.** Commits are pre-authorized for
    validated results, plan updates, and decision records (gates green, by exit code). The ONLY
    halts: (i) irreversible/destructive/outward-facing actions; (ii) a defect that would make
    continuing produce known-wrong numbers — halting that BRANCH only; (iii) the natural end.
    **Wave-close checklist, derived from the previous run's failures — each item MANDATORY:**
    1. "Done" is claimed ONLY against each task's verify clause, re-checked on disk — never
       against the task list (the previous run reported a half-finished strand as done).
    2. Every pre-registered prediction/bar in the specs of the wave's tasks is SWEPT and
       reconciled at wave close (the 1.5× cost prediction failed silently for weeks).
    3. Any gate/halt outcome is recorded with the rule QUOTED VERBATIM from the plan beside the
       measured values (two gates were mis-graded from memory of their rules).
    4. Grid launches are direct, verified started, and watched; every cell lands to disk as
       produced (two chained/detached launches failed silently).
    5. Gate→commit chains condition on the TEST'S exit code, never a pipeline tail's (two commits
       landed on red gates).
    6. Worker briefs carry: report-via-message clause, no-findings-file clause, land-as-verified
       clause, and explicit non-goals (three report-delivery incidents).
    7. Status and findings are reported by RESEARCH APPROACH in the user's ratified names
       (multi-head / single-head / recipe), never task ids.

33. **THE 2026-07-23 JOINT-REVIEW RULINGS (user, width results walkthrough).** Recorded so none
    is re-asked; each has its plan slot: **(i)** noise-aware (2·SE) thresholds ADOPTED over flat
    percentage bars — `docs/flexnn/width/plan.md` WSEL-20 (the WSEL-7 verdict re-grades under it; strand-wide
    binding: no new flat bar without a stated noise argument); **(ii)** the per-input labelling
    tolerance's pre-registered trigger FIRED — sensitivity sweep + σ-anchored replacement
    candidate, `docs/flexnn/width/plan.md` WSEL-22 (part b spec-gated; implication map incl. the BLEND reopen path
    in the block); **(iii)** the d ≥ 8 training-protocol escalation SCHEDULED — `docs/flexnn/width/plan.md`
    WSEL-21 (failure branch is a done-state); **(iv)** router regularisation demoted from
    mandatory to CONDITIONAL — `docs/flexnn/package/plan.md` FP-13; **(v)** FP-5's stale-reference finding
    RESOLVED by git archaeology, bisect moot — resolution block in `docs/flexnn/package/plan.md` FP-5;
    **(vi)** a PLAN HYGIENE PASS is authorized for the orchestration wave: roll CLOSED tasks'
    verdict histories into `archive/` (or per-task files under `shared/`), a one-paragraph
    verdict + pointer left in each task block; ROOT-ONLY writes, single session, citation AND
    numbers gates green before and after; nothing summarized away that an open task still reads;
    **(vii)** the report (WSEL-10) gains a REQUIRED transfer-ledger section (toy-negative /
    toy-positive / real-data-confirmed per adopted verdict). Standing conduct rule re-affirmed:
    obvious, low-risk, reversible decisions are taken and logged, never parked on the user.
    **Execution regime for ALL of it: NOTHING dispatches until the review completes — the user
    orchestrates the full planned set at the sitting's end (user, mid-review, verbatim intent:
    "Don't start the work. We will orchestrate all planned work.").**


<!-- from docs/plans/capacity_programme/MASTER.md lines 633-670 -->
35. **THE 2026-07-23 REVIEW RE-PASS RULINGS (user + root, same joint review, second sitting —
    review still OPEN at recording time; CLOSED by the user later the same sitting, wave
    dispatch authorized).** **(i)** the NORMALISATION thread is parked IN FULL
    (user: an architecture must first work *without* normalisation) — the parking of 2026-07-22
    is WIDENED to include the queued `per_width_affine_needed` re-grade, struck from the
    orchestration wave; the field stays quarantined/un-citable; reopen trigger unchanged
    (WSEL-23 end-state) — `docs/flexnn/width/plan.md` WSEL-15 FOLLOW-UP block. **(ii)** ROOT CATCH, remedy-seam:
    WSEL-24's band/SE conviction routed to a task whose scope disclaimed the GLOBAL selection
    rule; fixed by a CONDITIONAL WSEL-22(c) (global-band remedy, spec-gated, dormant unless
    WSEL-24 convicts; `rule_objective_mismatch` stays a user policy ruling, not (c)'s) —
    `docs/flexnn/width/plan.md` WSEL-22/WSEL-24 blocks. **(iii)** WSEL-23 candidate 1 gains: a pre-registered
    SELECTION corollary (premium shrinks ⇒ dial pick narrower, agreement rises from 0/3); a
    zero-training gradient-attribution DIAGNOSTIC that tests the candidate's premise before any
    weighted run; and a CONDITIONAL single-head companion arm (3 seeds, mechanism discriminator;
    runs unless the diagnostic refutes; retirement stands unless it closes the primary bar) —
    user-agreed evidence-gated design, `docs/flexnn/width/plan.md` WSEL-23 candidate 1. **(iv)** the Decision
    33(vi) hygiene pass's scope GAINS a staleness sweep: status markers whose stated blockers
    have since landed on disk (found instance: §3.8's tier-2 rows still say "DEFERRED — needs
    WSEL-15" though `width_candidates.py` + the weighted loss are long landed; the deferral may
    stand, the *reason* must be re-derived).

36. **THE SELECTION-POLICY RULING (user, 2026-07-23, post-wave — answers WSEL-24's batched
    question).** **Objective-matched rule + dual reporting.** The fielded GLOBAL width-selection
    rule becomes a stated function of the deployment objective: accuracy-scored contexts field
    the accuracy-optimal pick; efficiency-scored contexts field cheapest-within-tolerance
    (smallest sufficient). Every battery table reports BOTH picks side by side with their
    accuracy and compute cost — both read off the recorded held-out curves, zero retraining.
    **Default when the caller states no objective: SMALLEST-SUFFICIENT** (the programme's own
    question; conservative on compute), the accuracy-optimal pick one documented flag away.
    Rationale on record: WSEL-24 showed neither single rule is universally right (the mismatch
    is genuine, not a defect); accuracy-argmin-everywhere was REJECTED because raw argmin is
    exactly the rule that overshoots on small noisy selection sets — WSEL-24's causally
    demonstrated small-data failure mode. Flows into: **(i)** WSEL-10's battery-section framing
    (dual-pick tables, objective stated per context — bullet added in its block); **(ii)**
    WSEL-6-R's real-data success criterion (selection is graded against the pick matching the
    stated objective, both objectives reported); **(iii)** the library's documented default.
    Implementation travels through the owning tasks — the ruling itself changes no rule code.


<!-- from docs/plans/capacity_programme/MASTER.md lines 820-918 -->
## History (frozen — never dispatch from)

- `docs/flexnn/width/history/width_mse_2026-07/EXECUTION_PLAN.md` — width-MSE programme, COMPLETE
  (WP-0..5). Its §5 bars are re-used by reference throughout this programme.
- `docs/flexnn/width/reports/verdict_variable_width_mse.md` — the width verdict + file manifest.
- `docs/flexnn/width/history/width_dial_2026-07/`, `docs/probreg/per_input_k/history/perinput_selector_2026-07/`,
  `docs/probreg/history/capacity_ladder_2026-07/plans/` — earlier completed programmes (NLL era).
- `docs/applications/research_plan.md` — the two-paper roadmap (2026-04); its §1 bug ledger is PARTIALLY
  STALE (N1 fixed at HEAD; N3 appears fixed at HEAD — see strand 2 re-audit task).

## Corrections

- **2026-07-20, WHAT THE TWO CLOSED GATES DID AND DID NOT CERTIFY** (recorded by `docs/flexnn/width/plan.md`
  WSEL-0 and `docs/flexnn/depth/plan.md` DSEL-0; emitted by those tasks, applied by the root). Both gates
  are PASS and neither is reopened — but both were being read as covering more than they do, and
  the two selection strands exist precisely because of the gap.
  - **`G-WIDTH = PASS` (2026-07-16) certifies the ARCHITECTURE — `SharedTrunkPerWidthHeadNet` —
    and NONE of the three ways of choosing a width.** Its two pre-registered clauses
    (`docs/flexnn/width/history/width-cert.md:308-318`) are both about the dial's behaviour; neither is about selection.
    `docs/flexnn/width/history/width-cert.md` owns that certification and is **CLOSED**; `docs/flexnn/width/plan.md` owns everything about
    *choosing* a width and never restates it. **W-SHARED does not exist as library code at all**
    — the only cheap global readouts are script-level `argmin`, which is the wrong rule per
    `docs/flexnn/width/plan.md` §1.
  - **`G-DEPTH = PASS` (2026-07-17) certifies the RECURRENT arm under supervision-at-every-depth,**
    and covers neither the feed-forward mechanism nor any of `docs/flexnn/depth/plan.md` §1b's three
    choices. `docs/flexnn/depth/history/depth.md` owns that certification and is **CLOSED**; the recurrent arm is **⏸ PARKED**
    — cite it, never run it. For feed-forward depth, **nothing is established.**
  - **Neither gate may be cited as evidence about selection.** A toys-only or architecture-only
    result presented as though it had survived a selection comparison is the misreading these two
    strands were created to correct, and each strand's report is bound to say so explicitly.

- **2026-07-20, ProbReg had THREE contradictory model definitions live at once — root cause was
  organisational.** ProbReg content sat in 15 files across 5 plan directories with **no owning
  strand**; the definitions in this file's naming key, in
  `docs/probreg/shared/benchmark_spec.md`, and in the user's actual design all disagreed, and
  the benchmark's two arms differed in TWO respects (training scheme *and* k choice) while
  claiming to differ in one. Nobody could see it because no single document held them side by
  side. Fixed by creating `docs/probreg/PLAN.md` as the single source (§1), which supersedes every other
  statement. **Generalised rule (user, 2026-07-20): one execution-level strand file per
  workstream, MASTER stays an index.** A definition may exist in exactly one file; everywhere
  else points at it.
- **2026-07-20, a regression test is not evidence until it has been shown to FAIL on the unfixed
  code.** The two tests guarding the `fit_router` units fix both PASSED with the fix deleted — one
  asserted on a downstream view too coarse to see the error, the other used a threshold above the
  broken value. "Tests green" and "the bug is caught" are different claims and only the removal run
  separates them. Both rewritten and re-proved in both directions; detail in `docs/flexnn/archive/flexnn-core.md`
  (F9-fix-b block). **Add the removal run to every bug-fix task's `verify:` line.**
- **2026-07-17, G-DEPTH CLOSED (both halves) = PASS.** D8b selection battery landed: 2 seeds, all
  trustworthy/`diverged=false`. **S1** full-T fit ≥0.90/stratum 2/2; **S2** make-or-break knee 2/2
  (ρ(T*,t*)=1.000/0.993, acc@t*−2=0 cliff); **S3** deploy mean-T 8.0/7.99 < best-fixed-10 with MSE
  *improving* (routing to t* beats full depth); **S4** oracle-free router (raw word only). **S5**
  surface MLP recovers t* at 100% → Option-A covariate, NOT a kill (answer f(x) not surface-computable;
  §3/D7 depth still width-irreducible). Selection verdict appended `verdict_per_input_depth.md`
  §10–§12. **D8b toy AS-RUN rebuild** (the D8a C1‴ trained to chance → root-caused): L16→10,
  five-cycle→A5-involution, per-T-head→shared-readout-on-running-product, n40k→3000, ceiling 0.2875→0.35
  — all reversible, charter intact, flagged for user veto. **G-DEPTH = D5 (3/3) ∧ D8 (2/2) = PASS →
  `docs/flexnn/joint/plan.md` J0.** (Nothing committed — user-gated.)
- **2026-07-17, depth closing waves + substrate PASS + D8a sign-off.** D6 divergence guard landed
  (`convergence.py` `diverged` flag; calibration surfaced TWO genuine Z120 blow-ups, both quarantined,
  neither touches a bar). D7 param-matched wide-101 run landed (16,381 params, stalls 0.447 @ℓ10 →
  G2 holds at parity). **D5 substrate verdict = PASS 3/3** →
  `docs/flexnn/depth/reports/verdict_per_input_depth.md`. **D8a design signed off by user (Option A):**
  concealment dropped as a kill criterion (difficulty-of-detection is not the charter), surface-baseline
  control added, narrow-deep width-substitutable graded toy documented as the fallback. D8b (selection)
  now building; gradedness is the make-or-break. **G-DEPTH = substrate PASS ∧ (D8 pending).**
  *(superseded — G-DEPTH closed, see top entry)*
- **2026-07-17, depth D1b landed + rescope (user-ratified).** Graded battery: ALL pre-registered
  bars pass on 3/3 seeds with the ℓ=4 rung excluded (degenerate by design arithmetic — 128 train
  words vs 120 classes); ledger
  `automl_package/examples/capacity_ladder_results/D_TOY_PROBES/depth_graded_pilot_s5_seed0.json`
  + 5 siblings. Transfer prediction REFUTED 3/3 (Decision 5 amended); Decision 2 amended (CE for
  depth). Old D2/D3 retired → `docs/flexnn/depth/history/depth_d2_d3_retired.md`; verdict target renamed →
  `docs/flexnn/depth/reports/verdict_per_input_depth.md` (baseline −3/+1). Convergence gate certifies
  diverged runs as trustworthy (Z120 seed-1 shared_readout blow-up) → fix = `docs/flexnn/depth/history/depth.md` D6.
  Learned halting moved to the joint strand (`docs/flexnn/joint/plan.md` J0, autonomous protocol).
- **2026-07-16, depth D1 kill criterion.** All three `docs/flexnn/depth/history/depth.md` §D1 candidates (gentle
  composition, hierarchical spline, 2-D multiplicative) exhausted the pre-registered probe bars
  on 2 seeds each with no full pass — see `docs/flexnn/depth/reports/depth_toy_negative_note.md` (probe
  JSONs: `automl_package/examples/capacity_ladder_results/D_TOY_PROBES/*.json`). The depth strand
  does NOT proceed to D2/D3 pending a fresh D1 candidate. D0's citations were re-verified at this
  same HEAD and still hold exactly (`output_layer` at `layer_selection_strategies.py:90,136,180`;
  `independent_weights_flexible_neural_network.py` still present) — no citation fix needed.

- **2026-07-21, THE REPAIR-PASS RULE, from a five-way plan audit.** The 2026-07-20 repair pass
  edited prose without re-checking disk. Result: three strands carried "OPEN" defects whose fixes
  were ALREADY COMMITTED — `docs/probreg/PLAN.md` D2 and `docs/flexnn/depth/plan.md` DD1/DD2 in `84ad94d`,
  `docs/flexnn/width/plan.md` WD1/WD4 in `63ab6bc` — with their fix-tasks (P1, WSEL-1, DSEL-3) sitting as no-ops at
  the head of three dependency chains, and two tasks (`docs/flexnn/package/plan.md` FP-0/FP-7) done on disk
  with no completion marker. **Rule, binding on every future repair pass: a defect entry names its
  regression test, and a repair pass RE-RUNS the named verifies against disk — it never edits the
  prose alone.** The same audit found and the same-day repair fixed: three strands pre-authorised
  to "freeze globally" a shared-file constant none may write (now per-dial defaults; the shared
  file moves only via FP-5, root-applied); three task specs scheduling the tolerance sweep
  Decision 18 forbids (struck); width's and depth's "not reportable" rules naming PARKED tasks and
  therefore binding nothing (re-pointed to live consumers); `docs/flexnn/package/plan.md`'s helper-count
  self-contradiction and impossible as-run order note (corrected); Decisions 20 and 21 added. The
  depth positive-control halt was answered the same day with a re-run-at-spec ruling
  (`docs/flexnn/depth/plan.md` DSEL-2c) after the audit showed the as-run control deviated from its own
  spec twice.


## Code and results of this stream

Where this stream's experiment scripts and results live; the code stays where it is until the
planned code move. Scripts are in `automl_package/examples/`; `LADDER/` is
`automl_package/examples/capacity_ladder_results/`, whose `RESULTS.md` records what completed.

- **depth** — scripts: `_capacity_ladder_toys.py`, `_flexnn_prefix_selftest.py`, `capacity_ladder_f1_validation.py`, `capacity_ladder_f2.py`, `capacity_ladder_f3.py`, `capacity_ladder_p1.py`, `capacity_ladder_t1.py`, `capacity_ladder_t1_path1.py`, `capacity_ladder_t1_path2.py`, `capacity_ladder_x1.py`, `capacity_ladder_x3.py`, `depth_composition_toy.py`, `depth_dsel2.py`, `depth_graded_toy.py`, `depth_selection_toy.py`, `depth_toy.py`, `flex_nn_ablation.py`, `flex_nn_depth_viz.py`, `flexible_nn_showcase.py`, `flexnn_probabilistic_test.py`, `independent_weights_showcase.py`; results: `LADDER/DSEL1/`, `LADDER/DSEL1b/`, `LADDER/DSEL2/`, `LADDER/DSEL2c/`, `LADDER/D_TOY_PROBES/`, `LADDER/F1/`, `LADDER/F2/`, `LADDER/F3/`, `LADDER/P1/`, `LADDER/T1/`, `LADDER/X1/`, `LADDER/X3/`, `flex_nn_ablation_results/`, `flex_nn_depth_viz_results/`
- **joint** — scripts: `joint_capacity_toy.py`; results: `LADDER/J_TOY_PROBES/`
- **moe** — scripts: `flexnn_revalidation.py`, `moe_flexnn_comparison.py`, `moe_regression.py`; results: `report_b_results/`
- **package** — scripts: `capacity_accounting.py`, `convergence.py`
- **width** — scripts: `cascade_width_experiment.py`, `cascade_width_net.py`, `converged_width_experiment.py`, `hetero_width_experiment.py`, `kdropout_converged_width_experiment.py`, `matryoshka_width_net.py`, `nested_width_net.py`, `sinc_width_experiment.py`, `width_candidates.py`, `width_wsel11.py`, `width_wsel13.py`, `width_wsel15.py`, `width_wsel16.py`, `width_wsel19.py`, `width_wsel19_toys.py`, `width_wsel21_capacity.py`, `width_wsel22a.py`, `width_wsel22b.py`, `width_wsel23_candidate1_pin.py`, `width_wsel24.py`, `width_wsel4.py`, `width_wsel6.py`, `width_wsel6r.py`, `width_wsel7.py`, `width_wsel8.py`, `width_wsel9.py`; results: `LADDER/W1/`, `LADDER/W2/`, `LADDER/WSEL11/`, `LADDER/WSEL13/`, `LADDER/WSEL14/`, `LADDER/WSEL15/`, `LADDER/WSEL15_ALLSCHED/`, `LADDER/WSEL16/`, `LADDER/WSEL18/`, `LADDER/WSEL19/`, `LADDER/WSEL22A/`, `LADDER/WSEL22B/`, `LADDER/WSEL23/`, `LADDER/WSEL23_DATA_WEIGHTS/`, `LADDER/WSEL23_PWRESTORE/`, `LADDER/WSEL23_SHARING/`, `LADDER/WSEL4/`, `LADDER/WSEL6/`, `LADDER/WSEL6_RERUN_ROUTER/`, `LADDER/WSEL7/`, `LADDER/WSEL8/`, `LADDER/WSEL9/`, `LADDER/W_CASCADE/`, `LADDER/W_CONVERGED/`, `LADDER/W_KDROPOUT_CONVERGED/`, `LADDER/W_MRL/`
