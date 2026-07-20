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
| 4 | `width-depth.md` | Joint 2-D capacity dial + transformer halting → **G-JOINT** | **G-DEPTH** (D5 ∧ D8) | **J0 RAN 2026-07-17 — J-1/J-2 DEAD (multi-track fold substrate fails S1); G-JOINT BLOCKED, J-3 redesign ESCALATED to user** (post-mortem in strand) |
| 5 | `flexnn-moe.md` | MoE build (M0-M2 early) + reports (b), (c) | build: none; comparisons+reports: **G-JOINT** | **DONE — M0-M2 DONE 2026-07-16**; M3 rescoped as flexnn-core F6; M4/M5 superseded by F7 (user 2026-07-18) |
| 6 | `flexnn-core.md` | Package refactor + FF-depth pilot + unified report | — | **yes** — but see the SPLIT note below; this file is 4 workstreams in one |
| 7 | `probreg.md` | **ProbReg k-selection: models M1/M2/M3, defects, battery, report** | — | **yes — the live workstream (user, 2026-07-20)** |
| 8 | `width.md` | **Width SELECTION: the three ways of choosing a width, the studies, battery, report.** Architecture certification stays in `width-cert.md` (CLOSED) | `flexnn-package.md` FP-3, FP-9 | **yes — DISPATCHABLE** (repair closed 2026-07-20; WSEL-9 ⏸ PARKED, toys-only) |
| 9 | `depth-selection.md` | **FEED-FORWARD depth (the object) + depth selection.** Recurrent arm PARKED. Absorbs the FF-depth study out of `flexnn-core.md` | `flexnn-package.md` FP-3, FP-5, FP-6, FP-9 | **yes — DISPATCHABLE** (repair closed 2026-07-20; DSEL-11 ⏸ PARKED, toys-only) |
| 10 | `flexnn-package.md` | **The codebase**: package-vs-scripts boundary, the ONE selection API, router de-duplication, shared selection primitives, cleanup | — | **yes — DISPATCHABLE** (repair closed 2026-07-20; boundary rule ratified as Decision 19) |

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
  facets; the package family is `flexible_neural_network.py` + `flexible_width_network.py` (F2).
- **#1 / #2 / #3** — width architectures: #1 `NestedWidthNet` (shared trunk + shared readout,
  fails), #2 `SharedTrunkPerWidthHeadNet` (shared trunk + per-width heads, **certified winner —
  G-WIDTH PASS 2026-07-16**),
  #3 `IndependentWidthNet` (K disjoint sub-nets, positive control). All in
  `automl_package/examples/nested_width_net.py`.
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
    passes.
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
      a **flat `0.25` relative margin** (`automl_package/models/common/distilled_router.py:57`,
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
    package at module level, while `automl_package/models/flexible_width_network.py:290` and
    `automl_package/models/flexible_neural_network.py:492` import `executed_flops` back from that
    example **inside method bodies**, commented as avoiding a load-time circular import. Shipping
    code depending on a research script, held together by deferred imports. → **FP-1** breaks it.

    **The sanctioned migration shape is a re-export shim**, and the precedent already exists in this
    repo: `automl_package/examples/convergence.py` is a thin re-export over
    `automl_package/utils/convergence.py`, so existing scripts' imports keep resolving while the
    logic lives in the package. **Move the logic, leave the shim, do not rewrite callers** — a shim
    is NOT a deletion and passes `shared/PROTECTED.tsv`'s manifest check, because the path still
    exists.

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
