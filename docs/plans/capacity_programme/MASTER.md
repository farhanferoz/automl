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
| 1 | `width-cert.md` | Certification of arch #2 + seam/robustness analyses → **G-WIDTH** | — | **DONE — G-WIDTH PASS (2026-07-16)** |
| 2 | `probreg-report.md` | Report (a): ProbReg fixed-k & dynamic-k vs baselines on toys | `shared/metrics-accounting.md` T-S1..S3 | **DONE — P0-P3 done, report (a) delivered** (`docs/reports/probreg_toys/report_a_probreg_toys.{pdf,md}`) |
| 3 | `depth.md` | S5 substrate (D1 **DONE**) + selection-without-oracle (D8) → **G-DEPTH = D5 ∧ D8** | **G-WIDTH ✓** | **DONE 2026-07-17 — G-DEPTH = PASS.** Substrate D5 (3/3) ∧ selection D8b (2/2: S1–S4 pass, S5 surface covariate 100%/Option-A). Verdict `verdict_per_input_depth.md` §12; D8b toy AS-RUN rebuild (L10/involutions/shared-readout, flagged). old D2/D3 retired. → `width-depth.md` J0 |
| 4 | `width-depth.md` | Joint 2-D capacity dial + transformer halting → **G-JOINT** | **G-DEPTH** (D5 ∧ D8) | **J0 RAN 2026-07-17 — J-1/J-2 DEAD (multi-track fold substrate fails S1); G-JOINT BLOCKED. RULED 2026-07-21 (user-delegated): Option 1 — a proper J-3 design spec — DEFERRED until DSEL-2's feed-forward question resolves and WSEL-8/DSEL-10 land; the spec gets user review before any build (the no-improvised-toy gate survives autonomy). NOT in the autonomous run.** (post-mortem in strand) |
| 5 | `flexnn-moe.md` | MoE build (M0-M2 early) + reports (b), (c) | build: none; comparisons+reports: **G-JOINT** | **DONE — M0-M2 DONE 2026-07-16**; M3 rescoped as flexnn-core F6; M4/M5 superseded by F7 (user 2026-07-18) |
| 6 | `flexnn-core.md` | Package refactor + FF-depth pilot + unified report | — | **yes** — but see the SPLIT note below; this file is 4 workstreams in one |
| 7 | `probreg.md` | **ProbReg k-selection: models M1/M2/M3, defects, battery, report** | — | **yes — the live workstream (user, 2026-07-20)** |
| 8 | `width.md` | **Width SELECTION + the ARCHITECTURE comparison, the toy suite, and the code-organisation rules.** Architecture certification in `width-cert.md` — **AMENDED 2026-07-22 to PASS WITH CAVEATS** (user ruling after adversarial re-derivation; see width.md §2 ⛔ block) | **`flexnn-package.md` FP-11 (TASK ZERO)**, FP-3, FP-9, FP-4 | **yes — the LIVE strand, under the Decision-32 autonomous mandate.** Wave-2 order: `[WSEL-18 ∥ WSEL-16 stage-2 authoring ∥ WSEL-3 ∥ WSEL-4] → stage-2 cells + WSEL-5 → stage-3 (WSEL-18-gated) → verdict → (WSEL-6 ∥ WSEL-7) → WSEL-8 (architecture-parameterised per the WSEL-16 coupling ruling) → WSEL-17`; WSEL-13 tier 2 + small items slotted. Landed 2026-07-22: WSEL-14∥15 + follow-ups, WSEL-16 stage 1, cost probe, reviews. §3.7 sigma-fixed-at-truth · §3.8 toy suite · §3.9 no-duplication · §3.10 arbitrary-w_max rule · Decision-31 schedule/vectorisation/retention rulings. WSEL-9 ⏸ PARKED, toys-only; normalisation follow-ons ⏸ PARKED (user 2026-07-22) |
| 9 | `depth-selection.md` | **FEED-FORWARD depth (the object) + depth selection.** | — | **⛔ ENTIRE STRAND PARKED (user, 2026-07-21): "let's park depth completely". NOTHING dispatchable.** Supersedes the earlier strand-local block. Two untried levers (per-depth output layers; regularisation) and a missing literature survey are recorded in the strand header for whoever unparks it; neither lever has a written task, and writing them is a ROOT action on unpark. **Live programme = width + ProbReg only.** |
| 10 | `flexnn-package.md` | **The codebase**: package-vs-scripts boundary, the ONE selection API, router de-duplication, shared selection primitives, cleanup, **and FP-11 — ONE HOME under `models/flexnn/`** | — | **yes — and FP-11 is TASK ZERO for the whole programme** (user, 2026-07-21: do the reorganisation FIRST, not last — nothing is in flight to collide with, and the new width tasks create files that would otherwise be moved twice). FP-2 DONE. Boundary rule = Decision 19 |

**⚠ SPLIT PENDING on `flexnn-core.md` (opened 2026-07-20).** That file is 54KB holding FOUR
workstreams — package refactor (F0–F4/F8/F13), the feed-forward depth attribution study (F5), the
MoE comparison (F6) and the unified report (F7) — plus, until today, the ProbReg tasks. **One
strand file per workstream is the rule (user, 2026-07-20);** a grab-bag file is what let three
contradictory ProbReg model definitions coexist.

**SPLIT STATUS 2026-07-20 (updated).** ProbReg → strand 7. The feed-forward depth study → strand 9
`depth-selection.md` (**NOT** `depth-ff.md` — that filename was planned and never used; do not look
for it). The package refactor → strand 10 `flexnn-package.md`. **`flexnn-core.md` is therefore
expected to retain only F6 (the MoE comparison) and F7 (the unified report) — but it does NOT yet:
it still holds live tasks whose write sets collide with strands 7–10** (the ProbReg benchmark driver
and spec, the k-selection report directory, and refactor debt that duplicates FP-2/FP-5). **Until
`flexnn-package.md` FP-0 records a disposition for every remaining task there and the root applies
it, `flexnn-core.md` is READ-ONLY for dispatch purposes** — nothing in it may be executed, because a
task in it may create a file a newer strand owns.

Gates are decision points written in the owning strand (evidence-backed verdict + branch).
Priority: flexnn-core (6) waves go first. `width-depth.md` J0 is **PARKED** pending user
Option 1/3 decision. *(Strands 1, 2, 3, and 5's M0-M2 are complete; live forward order updated
2026-07-18.)*

## Naming key

- **FlexNN** = the umbrella for ALL per-input capacity work; strands are its width/depth/joint/MoE/core
  facets. **The family gets ONE home, `automl_package/models/flexnn/`, via `flexnn-package.md` FP-11
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
  architectures under test are NOT here — they live in one module, `width.md` §3.9.**
- **G-WIDTH / G-DEPTH / G-JOINT** — the three programme gates.
- **ProbReg models M1 / M2 / M3** — the three ways of choosing k. **Defined in ONE place:
  `probreg.md` §1. Do not restate the definition here or anywhere else.** *(This entry previously
  defined variable-k as "dynamic-k (ELBO + SoftGating)" — in-training selection, which Decision 13
  demotes to a labelled comparison arm. It was one of three mutually contradictory definitions live
  at once on 2026-07-20; see `probreg.md` §1.1.)*
- **The three ways of choosing a capacity value** — for EVERY dial (k, width, depth): a cheap global
  read, a per-input distilled router, and an expensive per-value sweep, each scored and costed as a
  complete system including its selection machinery. **Defined per dial in exactly one place:**
  `probreg.md` §1 (k), `width.md` §1 (width), `depth-selection.md` §1b (depth). Do not restate here.
- **`CapacitySelection`** — the ONE selection enum, in `automl_package/enums.py`. **Owned by
  `flexnn-package.md` FP-3; every other strand consumes it and none declares its own.** FP-3 ships
  only the members whose mechanisms exist; a member is added by the task that builds its mechanism
  (the rule that retires the `WidthSelectionMethod.DISTILLED` trap rather than repeating it).
- **⚠ `width-depth.md` is STALE** w.r.t. both of the above: it predates the three-way model set, the
  shared selection rule, and `CapacitySelection`. Its J-3 redesign must be re-read against strands
  8–10 before it is dispatched. It is parked pending a user ruling in any case.
- **Artifact naming (binding on every strand — a new task follows this, it does not invent).**
  Settled during the 2026-07-20 plan repair; `width.md` is the worked example.
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
  - **Notes**: `docs/plans/capacity_programme/shared/<taskid_lower>_<topic>.md`.
  - **Test command**: `AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m pytest <path> -q`.
- **🔑 SHARED FILES ARE ROOT-ONLY — `MASTER.md` and `flexnn-core.md`.** Four wave-1 tasks (FP-0,
  WSEL-0, DSEL-0, P0) each need a `MASTER.md` edit, and two need a `flexnn-core.md` edit. They are
  **not independent** — independence is a disjoint WRITE set, not a different topic — and dispatching
  them concurrently as writers is the classic false-parallel that produces contradictory text in one
  file. **The rule: a task PRODUCES the exact text it needs added and reports it; the ROOT applies
  it.** Neither file may appear in any task's write set. `flexnn-core.md` additionally stays
  READ-ONLY for dispatch until FP-0's dispositions land (`shared/CORE-DISPOSITIONS.tsv`), which is
  the mechanism for changing it. *(FP-0 already states this rule for `flexnn-core.md`; it is
  generalised here so every strand is bound by it, not just the one that noticed.)*
- **`gates_baseline.txt` is ROOT-ONLY.** A worker that needs a forward reference parked **lists it
  in its report**; the root verifies each is a real Create target named by a task — and asserted by
  that task's verify line — before parking it, and records the owning task inline. **No typo may
  ever be parked.** A worker parking its own reference would let a typo grandfather itself.
- **Ledger** — result JSONs under
  `automl_package/examples/capacity_ladder_results/` (width/depth) and the per-strand
  results dirs named in each strand. The plan holds POINTERS to ledger files, never copied
  result numbers. A line carrying a result must be marked `RESULT:` and name its `.json`.

## Decision register (settled 2026-07-16, user-approved — do not re-ask)

1. **Charter verdicts to date** — strict nesting (#1) fails; break = shared READOUT; #2 is the
   **certified** architecture of record (**G-WIDTH PASS 2026-07-16**). Evidence:
   `docs/width_mse_2026-07-16/verdict_variable_width_mse.md` (§10 certification addendum; manifest §8).
2. **MSE-only** for all capacity strands (width/depth/joint). Variance stays parked.
   *(Amended 2026-07-17: binds the WIDTH strand only. The depth strand is CE classification per
   the `depth.md` PIVOT; the joint strand's metric follows the toy J0 adopts.)*
   ✅ **AMENDED AND SHARPENED 2026-07-21 (user, live) — "FIXED AT THE TRUE VALUE", NOT "DROPPED".**
   The user's clarification: *"I'm not saying we should drop the functionality of setting it. I'm just
   saying just fix it at the true value."* **The variance machinery STAYS; LEARNING the variance is
   what is forbidden**, because that is what overfits. Every width run uses sigma clamped to the
   generator's true per-point value, excluded from the optimiser. Full rule, the per-tier
   implementation, and the honest statement of what it changes versus the squared error already on
   disk: `width.md` §3.7. Key consequence, so nobody re-derives it: on a constant-noise toy the
   fixed-sigma likelihood **equals the squared error up to one positive constant**, so certified
   numbers stay comparable; on the two-noise-level control it becomes a **weighted** squared error
   that down-weights the noisy region 100x — a different objective from what was run there before,
   which must be labelled in any shared table, and which *sharpens* that control rather than blunting
   it.
3. **Reports are toys-only.** UCI/real-data belongs to the later Paper A/B roadmap
   (`docs/research_plan.md`), not these reports.
   *(**AMENDED 2026-07-20, user live.** Binds the WIDTH/DEPTH capacity reports only. The ProbReg
   k-selection work is now explicitly IN scope for real data + baselines: the report must carry
   baseline comparisons and real-world datasets. Dataset candidates already sourced in
   `docs/research_plan.md` §6 Layer 2/3.)*
   *(**CORRECTED 2026-07-20, twice, both after the amendment above was written:**
   (i) the baseline set is **one tree model (LightGBM), a plain single-output NN, and linear
   regression** — XGBoost and CatBoost were dropped the same day (`probreg.md` P0b;
   `docs/probreg_benchmark/benchmark_spec.md` baseline section). The list originally written here
   was already stale when written.
   (ii) "shared-k vs variable-k as two distinct models" is the **retired** framing. The live
   definition is the three-way model set, `probreg.md` §1.)*
   ✅ **RULED 2026-07-20 (user) — SETTLED, do not re-ask. WIDTH AND DEPTH STAY TOYS-ONLY.** The
   real-data exemption is **NOT** extended to width or depth; it remains **ProbReg-only**. The user
   explicitly kept the door open to a real-data pass later, so the two affected tasks are
   **⏸ PARKED, not deleted** — their specs are retained on disk verbatim and unpark unchanged if the
   decision is revisited:
   - `width.md` **WSEL-9** — parked; removed from the execution order; **WSEL-10 now deps on WSEL-8.**
   - `depth-selection.md` **DSEL-11** — parked; removed from the execution order; **DSEL-12 now deps
     on its predecessor, not on DSEL-11.**
   **Both report tasks CUT their baseline / real-data sections and must say so explicitly** — naming
   that no external comparator (tree model, plain single-output NN, linear floor) was run, that the
   strand's claims rest on constructed targets, and that real data is *deferred*, not refused.
   A toys-only result may never be presented as though it had survived a baseline.
4. **Hetero-NLL default** (strand 2): minimal fix if one exists cheaply; otherwise report (a)
   documents the limitation honestly and variance stays parked. Failure there does NOT unpark
   the variance programme.
5. **Depth/joint strands are structured around #2's pattern** (shared representation +
   per-capacity readout heads), conditioned on G-WIDTH; written contingencies apply if it fails.
   *(Amended 2026-07-17: holds for WIDTH only — the depth transfer prediction was REFUTED 3/3;
   shared readout is the depth winner. Joint prior = per-width heads × depth-shared readout;
   see `depth.md` and `width-depth.md` Settled inputs.)*
6. **Depth starts with the toy, not architecture** — a depth-hungry-but-GD-learnable target
   with analytic floor; kill criterion + escalation if unconstructible.
7. **MoE config**: 8 experts, top-2 routing primary; top-1 ablation; load-balancing auxiliary
   loss; param/FLOP-matched to FlexNN/width nets. Conventions verified against Shazeer 2017 /
   Switch / Mixtral before the build brief freezes (task M1).
8. **Reports supersede** the old parked "REPORT-2 + `mathematical_guide.tex` fold-in" item.
   Math is LIFTED from `docs/mathematical_guide.tex`, not re-derived.
9. **Trajectory discipline** (binding on every training conclusion): full per-width/depth
   held-out trajectories, convergence flags trajectory-verified, `hit_cap=False` required;
   load-bearing verdicts get an early-stop-OFF confirmation run at ≥4× the self-terminated
   budget. No conclusion from an endpoint. (Case law: verdict doc §2.1.)
10. **Report authorship**: `research-report` skill; authored as the user; full mathematical
    rigor; NO AI/tool provenance anywhere committed or shared.
11. **Commits are user-gated.** Nothing staged without explicit go.
12. **Worker tiering**: haiku = mechanical/complete-spec (tables from JSONs, formatting, path
    cleanup); sonnet = default build/battery/draft against decision-complete specs;
    opus/main = discovery (root-cause, toy construction), verdicts, math review, gate decisions.
    Orchestrate on the strong model; execute on the cheap one.
13. **Selection = post-hoc DISTILLATION (user, 2026-07-17).** Per-input capacity choice (width,
    depth, joint, transformer per-token depth) is learned by distilling a router from held-out
    per-capacity error tables — the certified width mechanism
    (`automl_package/examples/sinc_width_experiment.py::_fit_selector_mse`). In-sample /
    in-training selection is never the primary (failed for width; FlexNN ELBO depth-select
    refuted, M0); it may appear only as a labeled comparison arm after the distilled primary
    passes. **⚠️ AMENDED 2026-07-21 by Decision 29 (user): under the nested ladder the in-training
    strategies are now UNREACHABLE by default, not merely demoted — hard error, and out of the
    hyperparameter search space. The comparison-arm provision survives only behind an explicit
    opt-out flag. Read Decision 29 before citing this one.**
14. **Positive-control gate (2026-07-20).** Any battery containing a known-good arm runs that arm
    **FIRST, alone**, and it must reproduce its certified result at the same bar on **every** seed
    before further compute is spent. A failed positive control **HALTS** the battery: the protocol
    is then the defect under investigation, not the unknown arms. *(Case law: F5b — the certified
    `RecurrentComposer` collapsed on seed 0 (val acc 0.830 → 0.097) and the battery ran to
    completion anyway, leaving all 28 runs unreadable.)*
15. **Protocol parity when reusing a substrate (2026-07-20).** A battery reusing a toy that
    produced a certified result must **diff its training loop** against the loop that produced
    that result; every difference is justified in writing IN THE TASK before the run. *(Case law:
    F5b ran A5/L=10 through `automl_package/examples/depth_composition_toy.py`, which applies no
    gradient clipping, while the certified L=10 work ran through
    `automl_package/examples/depth_selection_toy.py`, whose `GRAD_CLIP_MAX_NORM = 1.0` is
    commented "L=10 needs clipping to stay GD-trainable".)*
16. **Optimization is exonerated BEFORE architecture is blamed (2026-07-20).** No arm is recorded
    as an architecture failure until either it is shown to fit the **training** set, or a
    documented escalation ladder (LR sweep → clipping → warmup → init scheme → normalization) has
    been run. An arm low on **both** train and val is **under-fit** — an optimization finding,
    never a generalization verdict. *(Generalizes the width-control rule in
    `docs/depth_capacity/ff_depth_toy_spec.md` §6 to every arm.)*
17. **The convergence gate must be computed on the metric the bar reads (2026-07-20).** Where a
    pre-registered bar is read on metric M, the `trustworthy`/`diverged` flags are computed on M
    (or on M *and* the loss, with both reported). *(Case law: F5b gated on val cross-entropy while
    its bars read val accuracy — cells whose accuracy had cleanly plateaued were quarantined for
    CE overconfidence, and the reverse error is equally possible.)*
18. **The tolerance split is CONFIRMED as legitimate (user ruling 2026-07-20).** The two selection
    rules in this programme are deliberately different and stay different:
    - **Global arms** (one capacity for the dataset — W-SHARED, W-SWEEP, ProbReg M1/M3, and the depth
      equivalents) select **cheapest-within-tolerance at twice a bootstrap-estimated standard error**:
      the smallest capacity whose held-out score is not meaningfully worse than the best.
    - **Per-input arms** (the distilled router — W-PERINPUT, ProbReg M2, D-PERINPUT) label each row at
      a **flat `0.25` relative margin** (`automl_package/models/flexnn/routing.py:57`,
      `DEFAULT_TOLERANCE`, applied as `error <= (1 + tolerance) * row_min`).

    **Why they legitimately differ:** a per-input decision has one row's worth of evidence, and **no
    standard error is estimable from a single observation**; a global chooser reads a whole held-out
    curve, over which a bootstrap standard error is exactly the right notion of noise. Forcing the
    per-input arm onto the twice-SE rule would mean inventing a standard error it cannot have.

    **Consequence, binding on every report — state it, do not paper over it:** the per-input arm's
    **chosen capacities are NOT directly comparable** to the global arms' on tolerance grounds. The
    arms share a cost objective, not a statistical selection rule. **Comparison lives on held-out
    error and cost, never on the chosen values.** Listing the two tolerance numbers side by side
    without saying this is the failure mode being ruled out.

    **Known soft spot, accepted:** the flat `0.25` is **inherited** (copied from
    `automl_package/examples/sinc_width_experiment.py`'s tie threshold), never measured. Accepted
    because the value does not carry the comparison. **Noted follow-up, NOT scheduled and NOT a
    blocker:** a sensitivity sweep would make it measured rather than inherited. Do not run
    pre-emptively; run it if a reviewer leans on the constant.

19. **The package/experiment BOUNDARY RULE (ratified 2026-07-20 by `flexnn-package.md` FP-0).**
    Authored in `flexnn-package.md` §2; stated here once because it binds every strand:
    - `automl_package/models/` and `automl_package/utils/` contain **library code** — reusable
      architectures, selection mechanisms, the selection API, accounting. They **never** import from
      `automl_package/examples/`.
    - `automl_package/examples/` contains **experiment drivers** — protocols, preregistered
      batteries, toy generators, result production. They may import freely from the package. A
      driver may hold a local implementation ONLY when it encodes an experiment-specific protocol
      the library does not and should not express, and it must say so in a comment, naming what
      differs.

    **The dependency arrow points ONE way.** Today it points both ways, and that is the defect:
    `automl_package/examples/capacity_accounting.py:62-63` imports two model classes from the
    package at module level, while `automl_package/models/flexnn/width/model.py:290` and
    `automl_package/models/flexnn/depth/model.py:492` import `executed_flops` back from that
    example **inside method bodies**, commented as avoiding a load-time circular import. Shipping
    code depending on a research script, held together by deferred imports. → **FP-1** breaks it.

    **The sanctioned migration shape is a re-export shim**, and the precedent already exists in this
    repo: `automl_package/examples/convergence.py` is a thin re-export over
    `automl_package/utils/convergence.py`, so existing scripts' imports keep resolving while the
    logic lives in the package. **Move the logic, leave the shim, do not rewrite callers** — a shim
    is NOT a deletion and passes `shared/PROTECTED.tsv`'s manifest check, because the path still
    exists.

20. **Training-schedule rule for nested/anytime models (user, 2026-07-21).** Train **every rung
    every step wherever all rungs are computable in ~one forward** — depth (exits read off the
    shared prefix) and ProbReg k (`NestedStrategy.all_rung_outputs`,
    `automl_package/models/flexnn/strategies/n_classes.py:207`). Where each extra
    rung costs a real forward — width, each width being its own matmul slice — use the certified
    **sandwich** (always min + max plus 2 random middles,
    `automl_package/examples/nested_width_net.py:93`). The **per-sample uniform draw is retired as
    a default schedule programme-wide**: it buys no compute where all-rungs is free, and width
    recorded its failure — the top rung trains only 1/k_max of the time
    (`docs/plans/width_dial_2026-07-11/EXECUTION_PLAN.md:119-129`; literature basis
    `docs/plans/width_dial_2026-07-11/nested_architecture_research_2026-07-11.md:104`). It survives
    only as a LABELLED comparison arm. Migration: `probreg.md` P7 (k); depth tests both schemes as
    arms in `depth-selection.md` DSEL-2c. A strand deviating from this rule carries a written
    justification in its §1. The three dials previously ran three different schedules with the
    reason (compute cost) written down nowhere — that is what this decision repairs.

21. **Regularisation is explicit, never accidental (user, 2026-07-21).** The research training
    loops are otherwise unregularised — no weight decay, dropout, norm layers, or mini-batching
    (audited 2026-07-21; the library's levers all default off) — so cheapest-within-tolerance
    risks selecting small capacity because small OVERFITS LESS, not because small suffices: a bias
    identical across all three dials and invisible to cross-strand agreement. Every dial runs a
    **discriminating check** (weight decay λ ∈ {0, 1e-4, 1e-2} on its sweep reference; does the
    selected value move beyond tolerance?) BEFORE its battery is read: `width.md` WSEL-11,
    `probreg.md` P8; depth inherits after its positive control passes (DSEL-2c). A check that
    moves the selection BLOCKS that strand's battery reads — strand-local, logged prominently,
    batched for end-of-run user review while the run continues elsewhere — never a silent footnote
    *(block semantics set 2026-07-21 with the autonomous-run authorization)*. Weight decay is the Gaussian prior it is — the
    no-arbitrary-penalty premise binds the SELECTION objective, not MAP training. A schedule's
    stochasticity (or its absence) must never be the de-facto regulariser. Baselines receive the
    same treatment or the comparison is not like-for-like.

    **AMENDED 2026-07-21 (user), after BOTH checks were found to have violated their own strand's
    constraint — a pattern, not an incident.** `width.md` WSEL-11 *trained* on the Gaussian
    likelihood (fitting variance, which Decision 2 parks for width); `probreg.md` P8 *selected* on
    Gaussian NLL (a variance metric, which the σ-scope decision removed from that strand). They
    returned OPPOSITE verdicts and between them either cleared or blocked three downstream tasks.
    **Both are REOPENED and their results DISCARDED.** Binding additions:
    - **A discriminating check runs on its own strand's sanctioned objective and metric.** Width and
      the joint strand: mean-only, squared error. **ProbReg: the fixed-σ mixture log-likelihood —
      ⚠️ AMENDED 2026-07-21, see Decision 24.** Depth: classification. A check measured on a
      different objective from the arms it protects proves nothing about them.
      *(The original text here read "ProbReg: the point-prediction set." **That was wrong and it
      survived being written into this register**, which is how `probreg.md` §0.5 and P8 both came to
      mandate squared-error selection. Decision 24 is the authority; this line is corrected rather
      than deleted so the error is auditable. The diagnosis of what voided P8 stands — it selected on
      a **learned** variance — but the remedy is a likelihood at FIXED σ, never squared error.)*
    - **The check's own compliance is part of its verify line** — the task states the objective and
      the run's provenance shows it. "The driver defaulted to it" is how both of these happened.
    - **RESULTS PRODUCED IN VIOLATION OF A CONSTRAINT ARE DISCARDED, NOT REINTERPRETED** (user ruling,
      2026-07-21). They stay on disk as a record of what was run and may not be cited as evidence for
      anything. A re-run changes ONLY the violated dimension, so it stays comparable to its own
      pre-registration.
    - **A void check does NOT clear a battery and does NOT settle the question it asked.** The
      correct downstream state is *unmeasured* — which under this decision blocks the battery read
      just as a failed check would.

22. **The transformer port target is DEPTH; width ports only where the readout is linear (user,
    2026-07-21).** The programme's stated bet is a flexible-capacity architecture that transfers to a
    transformer. The analysis is recorded ONCE, in `shared/width_transformer_port.md`, and is
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
      property (`width.md` **WSEL-13**) and the schedule/cost sweep (**WSEL-14**) — not a broader
      width programme; (c) **cost is a first-class recorded output** from here on (parameters,
      executed FLOPs at the deployed rung, wall-clock), reusing
      `automl_package/utils/capacity_accounting.py`, never argued in prose; (d) the depth material is
      an INPUT to `depth-selection.md`, not a width claim, and is blocked behind that strand's
      failing positive control (Decision 14); (e) ⚠️ **depth has NO literature pillar** — width got
      one (`docs/plans/width_dial_2026-07-11/nested_architecture_research_2026-07-11.md`), early
      exiting in transformer stacks is a crowded area, and **no depth result may be positioned as
      novel until an equivalent survey exists.**

23. **Reports are PARKED behind a joint results review (user, 2026-07-21).** Both live strands need a
    comprehensive report — `width.md` **WSEL-10** and `probreg.md` **P6**, both authored via the
    `research-report` skill, as the user, zero AI provenance (Decision 10). **Neither may start.**
    The sequence is binding:
    1. All execution work in **both** live strands completes (width battery + ProbReg battery).
    2. **The user and the root review the results together** — do the numbers make sense, has
       anything been missed, does any arm need re-running.
    3. **Only then** is a report written.
    - **Why the gate exists, in the user's own terms:** a comprehensive report is expensive, and
      producing one on results that turn out to be wrong or incomplete wastes that effort. The review
      is a cheap check in front of an expensive irreversible step.
    - **One review, both strands, in one pass** — they share machinery (the selection primitives, the
      router, the accounting), so a defect in one is likely a defect in the other; reviewing them
      separately would miss exactly the cross-strand misses this gate is for.
    - **Consequence for planning:** no task may list "write the report" as its completion criterion,
      and no battery may be scheduled *because* the report needs it. Report content is an OUTPUT of
      the review, not an input to it.

24. **The capacity readout is a likelihood at FIXED σ — never squared error, never a learned variance
    (root, 2026-07-21, promoted to MASTER after the strand file was found contradicting it).**
    Promoted here *because* it was ratified only in `docs/probreg_benchmark/benchmark_spec.md` §4.2/§4.3
    and `probreg.md` §0.5 had drifted to the exact opposite. A rule that lives in one document does not
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
    - **Unblocks `probreg.md` P3 and P4**, whose deps carried this as an explicit blocking entry.

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
    - **Scope consequence:** `flexnn-package.md` **FP-12** grows from a scoring change to a scoring
      **and training** change, and now overlaps `probreg.md` **P7** (which rewrites the training
      objective for the schedule migration). **Those two must be sequenced deliberately or merged.**

27. **ProbReg head-layout arm list (user, 2026-07-21).** Separate per-class heads are the model of
    record (`probreg.md` §1). The single-head-with-per-class-outputs layout is **retained as a
    LABELLED comparison arm, never a default** — it distinguishes "components help" from
    "*independent* components help". The single-head-final-output layout stays **blocked under
    `NESTED`** and serves only as P11's mechanism control, because it produces no components at all.

28. **Depth stays parked until both live strands close (user, 2026-07-21).** Confirms the existing
    park rather than changing it. Rationale recorded so it is not re-litigated: width's architecture
    comparison is about to test the ordering/cascade assumptions any depth task would be designed on,
    and Decision 22 already names depth as the better transformer-port target. Unparking earlier
    means designing depth's tasks on premises that are mid-test.

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
    the four mechanical eligibility checks, attended, via `probreg.md` P9's manifest — several of these
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

30. **THE SUITE BAR IS RE-BASELINED — measured 2026-07-21, by `flexnn-package.md` FP-12, NOT by P7
    (root).** Decision 26 anticipated **P7** carrying this re-baseline. **FP-12's Decision-29 guard
    reached it first**, because retiring a configuration breaks every test that constructs one. The
    bar is recorded here, once, so no later task re-derives it or quietly restores the old one.

    | | old bar (pre-wave-1) | **NEW BAR (measured, post-fix)** |
    |---|---|---|
    | passed | 366 | **372** |
    | failed | 2 | **56** |
    | skipped | 1 | **1** |

    **EVERY failure is attributed. Zero unexplained** — this is the claim that makes the number
    usable, and it was checked per-test, not assumed:
    - **54 = the Decision-29 guard firing on retired configurations.** The ratified decision working,
      not damage. Concentrated in `test_phase2_flexible_nn.py` (28), `test_phase4_regression.py` (8),
      `test_ce_stop_grad.py` (6, whole file — its subject `CE_STOP_GRAD` is retired),
      `test_phase1_probabilistic_regression.py` (4), `test_probreg_identifiability_integration.py`
      (4, all `CE_STOP_GRAD` parametrisations), `test_ordering_constraint.py` (3),
      `test_capacity_accounting.py` (1).
    - **2 = the pre-accepted heteroscedastic pair**, still failing. **They were NOT driven green** —
      Decision 26 forbids it, because forcing them green suppresses the effect the change exists to
      make visible.

    **⚠️ THIS BAR IS A WAYPOINT, NOT THE DESTINATION.** The 54 guard failures are tests that must be
    *updated* — they construct retired members and need the explicit opt-out flag at those sites, or
    rewriting against a surviving member. **A follow-up task owns that**, and it must run BEFORE any
    later task reads "no new failures" as a signal, because 54 failures drown any real regression.
    **Until then, the only usable regression signal is the ATTRIBUTION, not the count**: a new
    failure that is not a retired-member construction is a real regression, whatever the total says.

    **Case law from this measurement, worth more than the number:** the worker wrote a test asserting
    that the now-broken default constructor *should* raise. **A test that asserts a regression is
    correct behaviour permanently blinds the suite to it** — it goes green and no later run flags it.
    Retiring a mechanism never licenses breaking a shipped class's constructor. The test was inverted
    to assert the default constructs, plus a companion asserting **every method the search space
    advertises actually constructs** — the property that was really violated.

31. **WIDTH SCHEDULE + ARCHITECTURE RULINGS (user, 2026-07-22, during the width review).** Three
    parts, binding on the width strand and on any strand that later trains multi-width models:
    (a) **the ALL schedule (every width, every step) is the DEFAULT** — the sandwich survives only
    as the labelled comparability mode against already-landed ledgers; rationale: dominance, not
    trade-off (never less accurate, 6× lower mid-width variance, cost premium is an implementation
    artifact removed by (b)); (b) **the multi-head readout is VECTORISED before any further
    multi-head compute** (`width.md` WSEL-18 — exact under ALL, refused under sampling);
    (c) **the multi-head architecture is RETAINED regardless of the WSEL-16 outcome** — the
    comparison selects the default recommendation, never a sole survivor, and WSEL-17's deletion
    scope is amended accordingly. Full ruling text: `width.md` §Decision-20 REVISIT.

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
    percentage bars — `width.md` WSEL-20 (the WSEL-7 verdict re-grades under it; strand-wide
    binding: no new flat bar without a stated noise argument); **(ii)** the per-input labelling
    tolerance's pre-registered trigger FIRED — sensitivity sweep + σ-anchored replacement
    candidate, `width.md` WSEL-22 (part b spec-gated; implication map incl. the BLEND reopen path
    in the block); **(iii)** the d ≥ 8 training-protocol escalation SCHEDULED — `width.md`
    WSEL-21 (failure branch is a done-state); **(iv)** router regularisation demoted from
    mandatory to CONDITIONAL — `flexnn-package.md` FP-13; **(v)** FP-5's stale-reference finding
    RESOLVED by git archaeology, bisect moot — resolution block in `flexnn-package.md` FP-5;
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

34. **EXHAUST-BEFORE-NEGATIVE (user ruling, 2026-07-23 joint review, verbatim intent: "This is
    an exercise in inventing a new architecture... we need to exhaust every possibility of
    improving the architecture... before we report a negative result").** WSEL-8's both-halves
    FAIL stands as a recorded measurement but is NOT report-final; the replacement line does not
    stop. `width.md` WSEL-23 enumerates the improvement ladder — derived observation-model loss
    weighting as PRIMARY (probabilistically principled: weights 1/(σ² + a²(w)) fall out of an
    explicit per-width observation model, no arbitrary penalties; all variances FIXED, never
    learned), deployment-prior mixture, per-width private capacity, self-distillation — with a
    CHECKABLE exhaustion end-state; WSEL-10's replacement-claim section is gated on it. The
    generalization clause binds every candidate: parametric laws in w validated at ≥2 values of
    w_max, never per-width constants (§3.10 discipline, same as the router sizing rule). The
    ARBITER (the held-out error-table per-input judge the router is distilled from) is a NAMED
    reference in the report's comparison suite — the routing upper bound, shown with its measured
    noise limitation and the generator-true oracle beside it. Also ruled the same exchange, on
    the blend (recorded in `width.md` WSEL-22 b′): the blend has NO threshold — it is purely a
    function of the out-of-sample per-width values; the bake-off's blend verdict is scoped to
    the implemented (thresholded-label) construction.

35. **THE 2026-07-23 REVIEW RE-PASS RULINGS (user + root, same joint review, second sitting —
    review still OPEN at recording time; CLOSED by the user later the same sitting, wave
    dispatch authorized).** **(i)** the NORMALISATION thread is parked IN FULL
    (user: an architecture must first work *without* normalisation) — the parking of 2026-07-22
    is WIDENED to include the queued `per_width_affine_needed` re-grade, struck from the
    orchestration wave; the field stays quarantined/un-citable; reopen trigger unchanged
    (WSEL-23 end-state) — `width.md` WSEL-15 FOLLOW-UP block. **(ii)** ROOT CATCH, remedy-seam:
    WSEL-24's band/SE conviction routed to a task whose scope disclaimed the GLOBAL selection
    rule; fixed by a CONDITIONAL WSEL-22(c) (global-band remedy, spec-gated, dormant unless
    WSEL-24 convicts; `rule_objective_mismatch` stays a user policy ruling, not (c)'s) —
    `width.md` WSEL-22/WSEL-24 blocks. **(iii)** WSEL-23 candidate 1 gains: a pre-registered
    SELECTION corollary (premium shrinks ⇒ dial pick narrower, agreement rises from 0/3); a
    zero-training gradient-attribution DIAGNOSTIC that tests the candidate's premise before any
    weighted run; and a CONDITIONAL single-head companion arm (3 seeds, mechanism discriminator;
    runs unless the diagnostic refutes; retirement stands unless it closes the primary bar) —
    user-agreed evidence-gated design, `width.md` WSEL-23 candidate 1. **(iv)** the Decision
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

## Rules (cache discipline)

⚠️ **THE NUMBERS GATE IS PARTIAL — strengthen it before the next planning round.**
*(This paragraph originally read "there is no numbers gate". **That was wrong, and it was written
without opening the test file** — the same assert-from-recall failure it is warning about. Corrected
within the minute, by running the suite and reading `test_plan_gates.py`.)*

What exists in `test_plan_gates.py`: a **citations** gate (cited paths resolve, with a shrink-only
`gates_baseline.txt` for forward references), a `RESULT:`-line gate (any `RESULT:` line must carry a <!-- citecheck-ignore: describes the gate, does not carry a result -->
`.json` ledger pointer, so numbers live in the ledger rather than in prose), and a baseline
shrink-only check.

What it still cannot see: **whether a number written in a plan matches the ledger's value, and on
what basis.** The `RESULT:` gate proves a pointer is present, not that the figure beside it is <!-- citecheck-ignore: describes the gate, does not carry a result -->
current — the skill's own warning is that a *superseded* number is still in the ledger and passes.

**This gap cost real money on 2026-07-20.** The citations gate was run repeatedly and reported as
`6 passed` while all four strand plans were undispatchable. Every cited path resolved; what was
wrong was what the paths *meant* — a real phrase on a real line used to justify a training schedule
that does not exist, and a "published selection rule" that is a significance threshold in the cited
section rather than a selection rule. **A green citations gate is a FLOOR, never a clean bill, and
must never be presented as evidence a plan is ready.**

Build the numbers gate per the skill's design notes: shrink-only baseline (never a permanently red
suite), ignore markers for deliberately-quoted bad values, and prove-it-fails before trusting it.
Reference implementation path is named in `~/.claude/skills/programme-plan/SKILL.md`.


- **Pointers, not copies.** Result numbers live in ledger JSONs / report artifacts; strands cite
  paths. `RESULT:` lines must name their `.json` (gate-checked).
- **DONE only with evidence attached** — artifact path or ledger line, never a worker claim.
- **DONE = the task's own `verify:` line EXECUTED, with its output shown.** File existence is NOT
  evidence; a `verify:` line with three conditions needs all three checked. *(Case law 2026-07-20:
  F5b, F6 and F9 were all carried as done — battery invalid, driver never executed, tests
  failing.)*
- **No claim without an artifact.** Any result or diagnosis claim written into a strand, `RESUME`,
  `CHANGELOG` or a report cites a path that resolves on disk. Prose-only findings do not exist.
  *(Case law: a "grad-norm diagnosis / Xavier init doesn't rescue it" conclusion survives only as
  narrative — no script, log or JSON anywhere in the repo.)*
- **Update the strand in the SAME TURN work lands.** Deferred = stale.
- **Verify a task is OPEN before dispatching** (open the artifact / grep the symbol first).
- **Single writer:** workers return findings; the ORCHESTRATOR writes strand files and ledgers.
  Fan out on compute; serialise writes. Write sets are declared per task; wave partition =
  `deps:` + write-set overlap.
- **No unverified citation:** every `file:line` in a strand was opened when written; re-verify
  before building on one (they rot).
- **Environment:** `AUTOML_DEVICE=cpu` on every run (shared 22-core box; ≤4 concurrent heavy).
  Heavy runs detached:
  `setsid nohup systemd-inhibit --what=idle:sleep:handle-lid-switch env AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python -u <script> ... > <scratchpad>/<log> 2>&1 &`
  — watch the summary FILE from the main thread, never a subagent.
- **Venv repair** (if `~/dev/.venv` loses deps again): surgical `uv pip install` of the missing
  declared deps only; `uv sync` is blocked by a mis-scoped guard hook; torch must stay XPU.

## Branch & merge protocol (binding on every execution wave)

**The rule that prevents dangling branches: a branch may not outlive its wave.** One short-lived
branch per dependency wave, merged and deleted the moment that wave's tasks verify. Never more than
one execution branch open at a time.

| | |
|---|---|
| **Name** | `capacity/wave-<N>` — e.g. `capacity/wave-1`. One per wave, never per strand or per task. |
| **Branches from** | `master`, at the moment the wave is dispatched. |
| **Carries** | only that wave's task output — code, tests, and the results JSONs those tasks emit. |
| **Merge trigger** | **every task in the wave has had its own `verify:` command executed by the root and seen to pass** (Rule: DONE = the task's own verify line EXECUTED). Not "the worker said done". |
| **Merge gate** | the plan gates green (`test_plan_gates.py` + `test_plan_numbers.py`, 9 tests) **and** the test files the wave touched pass. |
| **Merge how** | `finishing-a-development-branch` skill, option 1 (merge locally to `master`). This repo pushes to `origin` but has no PR workflow in use — `master` was 11 commits ahead of `origin/master` on 2026-07-20, so local-merge-then-push is the established cadence. |
| **Delete** | `git branch -d capacity/wave-<N>` **in the same session as the merge**. `-d` (never `-D`) refuses to delete an unmerged branch — that refusal is the mechanical guard, so never override it. |
| **Docs-only work** | plan edits, dispositions, reports → commit **straight to `master`**. They are not implementation, every strand reads them, and parking them on a wave branch would let workers dispatch against stale plans. |

**⛔ Two standing checks, run at the start of every session that touches this programme:**
```bash
git branch --no-merged master      # expect EMPTY. Anything listed is a dangling branch — merge or delete it.
git branch --merged master | grep -v '^\*\| master$'   # expect EMPTY. Merged-but-undeleted is the same defect.
```
A wave branch still present after its wave is complete is a **defect to fix before new work starts**,
not a thing to leave for later. *(As of 2026-07-20 both checks are clean: no branches besides
`master`.)*

**Autonomous run authorization (user, 2026-07-21).** The execution phase runs unattended in a
fresh session. **Pre-authorized:** wave-branch commits, local merges, branch deletion per this
protocol, and docs-only commits straight to `master`. **Still user-gated:** pushing to `origin`
(standing decision: do not push), publishing or sharing anything outward, deletions (`FP-8` stays
attended-only), and any change to a strand's §1 model definitions beyond what Decisions 20/21
already license. **Strand-local blocks** (a Decision-21 check that moves the selection; DSEL-2c's
all-arms-fail branch) block ONLY their own strand's downstream reads and are batched for
end-of-run review — the run continues on the other strands and never idles awaiting the user.
`width-depth.md`/G-JOINT is OUT of the autonomous run's scope (strand-index row 4).

**Why per-wave and not one long programme branch:** four strands over weeks on a single branch
diverges from `master` far enough that the merge becomes its own risky project — which is precisely
the dangling-branch failure this protocol exists to prevent. Short branches merge trivially.

## Mechanical gates

Run from repo root:
`~/dev/.venv/bin/python -m pytest docs/plans/capacity_programme/test_plan_gates.py -q`
- **Citations**: any `path/with/slash(.py|.md|.json|.tex|.sh)` cited in MASTER/strands/shared
  must resolve on disk (`file:line` → line must exist). `archive/` exempt.
- **Dated names**: no date-shaped filename in this plan dir (`archive/` exempt).
- **Numbers**: every `RESULT:` line must contain a `.json` pointer. (This gate proves existence
  of a pointer, NOT currency of the number — the newest ledger entry wins only if you look.)
- Baseline is shrink-only: `gates_baseline.txt` lists grandfathered violations; the suite fails
  if new ones appear or the baseline grows.

## History (frozen — never dispatch from)

- `docs/plans/width_mse_2026-07-16/EXECUTION_PLAN.md` — width-MSE programme, COMPLETE
  (WP-0..5). Its §5 bars are re-used by reference throughout this programme.
- `docs/width_mse_2026-07-16/verdict_variable_width_mse.md` — the width verdict + file manifest.
- `docs/plans/width_dial_2026-07-11/`, `docs/plans/perinput_selector_2026-07-10/`,
  `docs/plans/capacity_ladder_2026-07-09/` — earlier completed programmes (NLL era).
- `docs/research_plan.md` — the two-paper roadmap (2026-04); its §1 bug ledger is PARTIALLY
  STALE (N1 fixed at HEAD; N3 appears fixed at HEAD — see strand 2 re-audit task).

## Corrections

- **2026-07-20, WHAT THE TWO CLOSED GATES DID AND DID NOT CERTIFY** (recorded by `width.md`
  WSEL-0 and `depth-selection.md` DSEL-0; emitted by those tasks, applied by the root). Both gates
  are PASS and neither is reopened — but both were being read as covering more than they do, and
  the two selection strands exist precisely because of the gap.
  - **`G-WIDTH = PASS` (2026-07-16) certifies the ARCHITECTURE — `SharedTrunkPerWidthHeadNet` —
    and NONE of the three ways of choosing a width.** Its two pre-registered clauses
    (`width-cert.md:308-318`) are both about the dial's behaviour; neither is about selection.
    `width-cert.md` owns that certification and is **CLOSED**; `width.md` owns everything about
    *choosing* a width and never restates it. **W-SHARED does not exist as library code at all**
    — the only cheap global readouts are script-level `argmin`, which is the wrong rule per
    `width.md` §1.
  - **`G-DEPTH = PASS` (2026-07-17) certifies the RECURRENT arm under supervision-at-every-depth,**
    and covers neither the feed-forward mechanism nor any of `depth-selection.md` §1b's three
    choices. `depth.md` owns that certification and is **CLOSED**; the recurrent arm is **⏸ PARKED**
    — cite it, never run it. For feed-forward depth, **nothing is established.**
  - **Neither gate may be cited as evidence about selection.** A toys-only or architecture-only
    result presented as though it had survived a selection comparison is the misreading these two
    strands were created to correct, and each strand's report is bound to say so explicitly.

- **2026-07-20, ProbReg had THREE contradictory model definitions live at once — root cause was
  organisational.** ProbReg content sat in 15 files across 5 plan directories with **no owning
  strand**; the definitions in this file's naming key, in
  `docs/probreg_benchmark/benchmark_spec.md`, and in the user's actual design all disagreed, and
  the benchmark's two arms differed in TWO respects (training scheme *and* k choice) while
  claiming to differ in one. Nobody could see it because no single document held them side by
  side. Fixed by creating `probreg.md` as the single source (§1), which supersedes every other
  statement. **Generalised rule (user, 2026-07-20): one execution-level strand file per
  workstream, MASTER stays an index.** A definition may exist in exactly one file; everywhere
  else points at it.
- **2026-07-20, a regression test is not evidence until it has been shown to FAIL on the unfixed
  code.** The two tests guarding the `fit_router` units fix both PASSED with the fix deleted — one
  asserted on a downstream view too coarse to see the error, the other used a threshold above the
  broken value. "Tests green" and "the bug is caught" are different claims and only the removal run
  separates them. Both rewritten and re-proved in both directions; detail in `flexnn-core.md`
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
  `width-depth.md` J0.** (Nothing committed — user-gated.)
- **2026-07-17, depth closing waves + substrate PASS + D8a sign-off.** D6 divergence guard landed
  (`convergence.py` `diverged` flag; calibration surfaced TWO genuine Z120 blow-ups, both quarantined,
  neither touches a bar). D7 param-matched wide-101 run landed (16,381 params, stalls 0.447 @ℓ10 →
  G2 holds at parity). **D5 substrate verdict = PASS 3/3** →
  `docs/depth_capacity/verdict_per_input_depth.md`. **D8a design signed off by user (Option A):**
  concealment dropped as a kill criterion (difficulty-of-detection is not the charter), surface-baseline
  control added, narrow-deep width-substitutable graded toy documented as the fallback. D8b (selection)
  now building; gradedness is the make-or-break. **G-DEPTH = substrate PASS ∧ (D8 pending).**
  *(superseded — G-DEPTH closed, see top entry)*
- **2026-07-17, depth D1b landed + rescope (user-ratified).** Graded battery: ALL pre-registered
  bars pass on 3/3 seeds with the ℓ=4 rung excluded (degenerate by design arithmetic — 128 train
  words vs 120 classes); ledger
  `automl_package/examples/capacity_ladder_results/D_TOY_PROBES/depth_graded_pilot_s5_seed0.json`
  + 5 siblings. Transfer prediction REFUTED 3/3 (Decision 5 amended); Decision 2 amended (CE for
  depth). Old D2/D3 retired → `archive/depth_d2_d3_retired.md`; verdict target renamed →
  `docs/depth_capacity/verdict_per_input_depth.md` (baseline −3/+1). Convergence gate certifies
  diverged runs as trustworthy (Z120 seed-1 shared_readout blow-up) → fix = `depth.md` D6.
  Learned halting moved to the joint strand (`width-depth.md` J0, autonomous protocol).
- **2026-07-16, depth D1 kill criterion.** All three `depth.md` §D1 candidates (gentle
  composition, hierarchical spline, 2-D multiplicative) exhausted the pre-registered probe bars
  on 2 seeds each with no full pass — see `docs/depth_capacity/depth_toy_negative_note.md` (probe
  JSONs: `automl_package/examples/capacity_ladder_results/D_TOY_PROBES/*.json`). The depth strand
  does NOT proceed to D2/D3 pending a fresh D1 candidate. D0's citations were re-verified at this
  same HEAD and still hold exactly (`output_layer` at `layer_selection_strategies.py:90,136,180`;
  `independent_weights_flexible_neural_network.py` still present) — no citation fix needed.

- **2026-07-21, THE REPAIR-PASS RULE, from a five-way plan audit.** The 2026-07-20 repair pass
  edited prose without re-checking disk. Result: three strands carried "OPEN" defects whose fixes
  were ALREADY COMMITTED — `probreg.md` D2 and `depth-selection.md` DD1/DD2 in `84ad94d`,
  `width.md` WD1/WD4 in `63ab6bc` — with their fix-tasks (P1, WSEL-1, DSEL-3) sitting as no-ops at
  the head of three dependency chains, and two tasks (`flexnn-package.md` FP-0/FP-7) done on disk
  with no completion marker. **Rule, binding on every future repair pass: a defect entry names its
  regression test, and a repair pass RE-RUNS the named verifies against disk — it never edits the
  prose alone.** The same audit found and the same-day repair fixed: three strands pre-authorised
  to "freeze globally" a shared-file constant none may write (now per-dial defaults; the shared
  file moves only via FP-5, root-applied); three task specs scheduling the tolerance sweep
  Decision 18 forbids (struck); width's and depth's "not reportable" rules naming PARKED tasks and
  therefore binding nothing (re-pointed to live consumers); `flexnn-package.md`'s helper-count
  self-contradiction and impossible as-run order note (corrected); Decisions 20 and 21 added. The
  depth positive-control halt was answered the same day with a re-run-at-spec ruling
  (`depth-selection.md` DSEL-2c) after the audit showed the as-run control deviated from its own
  spec twice.
