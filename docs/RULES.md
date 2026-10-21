# Rules binding every stream

Rules and decisions that bind more than one research stream. Each stream's own decisions are in its
plan: `docs/probreg/PLAN.md`, `docs/flexnn/PLAN.md`. Index of every stream: `docs/README.md`.
Assembled 2026-10-21 from the former `docs/plans/capacity_programme/MASTER.md`; text moved, not <!-- citecheck-ignore -->
rewritten (`<!-- from … -->` markers).

## Decision register index

Decision numbers are unchanged from the former `MASTER.md`, so a citation such as "MASTER Decision <!-- citecheck-ignore -->
29" still names one decision. Where each now lives:

| Decisions | File |
|---|---|
| 3, 8–21, 23, 30, 34, 37, 38 | this file |
| 4, 24, 25, 26, 27, 29 | `docs/probreg/PLAN.md` |
| 1, 2, 5, 6, 7, 22, 28, 31, 32, 33, 35, 36 | `docs/flexnn/PLAN.md` |

## Where data, results and checkpoints live

`~/dev` is one Syncthing folder shared by the DGX, the NAS and the laptop (workspace-sync decision
D136 in `~/dev/workspace-sync/DECISIONS.md`). The laptop receives code and documents only: anything
git does not track that is heavy — a folder of at least 1 GiB holding no tracked file, or a single
file of at least 50 MiB — is kept off it automatically, and the NAS keeps a full copy as the DGX's
backup. Deletions propagate to every machine, so a snapshot taken before a destructive change goes
outside `~/dev`, in `~/cache/automl/`.
- **In the repo, tracked:** code, documents, summary results (JSON, CSV) and small figures.
- **In the repo, untracked:** small checkpoints and caches only (`*.pt` is git-ignored).
- **Outside the repo:** anything heavy, in `~/cache/automl/` or the NAS share for this project, with a
  line in a local `DISK.md` (path, size, reader, delete-when) once it reaches 1 GB.
- **Results from June–July 2026** stay in `automl_package/examples/capacity_ladder_results/`, one
  ledger across streams whose `RESULTS.md` records what completed; each stream plan lists its own
  folders there.
- **Git history** moves between machines only through the remote (workspace-sync D130): after a push
  here, the laptop runs `~/dev/workspace-sync/sync/repo_git_parity.sh catchup <repo>`, not `git pull`.

<!-- from docs/plans/capacity_programme/MASTER.md lines 114-115 -->
## Decision register (settled 2026-07-16, user-approved — do not re-ask)


<!-- from docs/plans/capacity_programme/MASTER.md lines 134-158 -->
3. **Reports are toys-only.** UCI/real-data belongs to the later Paper A/B roadmap
   (`docs/applications/research_plan.md`), not these reports.
   *(**AMENDED 2026-07-20, user live.** Binds the WIDTH/DEPTH capacity reports only. The ProbReg
   k-selection work is now explicitly IN scope for real data + baselines: the report must carry
   baseline comparisons and real-world datasets. Dataset candidates already sourced in
   `docs/applications/research_plan.md` §6 Layer 2/3.)*
   *(**CORRECTED 2026-07-20, twice, both after the amendment above was written:**
   (i) the baseline set is **one tree model (LightGBM), a plain single-output NN, and linear
   regression** — XGBoost and CatBoost were dropped the same day (`docs/probreg/shared/plan.md` P0b;
   `docs/probreg/shared/benchmark_spec.md` baseline section). The list originally written here
   was already stale when written.
   (ii) "shared-k vs variable-k as two distinct models" is the **retired** framing. The live
   definition is the three-way model set, `docs/probreg/PLAN.md` §1.)*
   ✅ **RULED 2026-07-20 (user) — SETTLED, do not re-ask. WIDTH AND DEPTH STAY TOYS-ONLY.** The
   real-data exemption is **NOT** extended to width or depth; it remains **ProbReg-only**. The user
   explicitly kept the door open to a real-data pass later, so the two affected tasks are
   **⏸ PARKED, not deleted** — their specs are retained on disk verbatim and unpark unchanged if the
   decision is revisited:
   - `docs/flexnn/width/plan.md` **WSEL-9** — parked; removed from the execution order; **WSEL-10 now deps on WSEL-8.**
   - `docs/flexnn/depth/plan.md` **DSEL-11** — parked; removed from the execution order; **DSEL-12 now deps
     on its predecessor, not on DSEL-11.**
   **Both report tasks CUT their baseline / real-data sections and must say so explicitly** — naming
   that no external comparator (tree model, plain single-output NN, linear floor) was run, that the
   strand's claims rest on constructed targets, and that real data is *deferred*, not refused.
   A toys-only result may never be presented as though it had survived a baseline.

<!-- from docs/plans/capacity_programme/MASTER.md lines 172-269 -->
8. **Reports supersede** the old parked "REPORT-2 + `mathematical_guide.tex` fold-in" item.
   Math is LIFTED from `docs/package/mathematical_guide.tex`, not re-derived.
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
    `docs/flexnn/depth/specs/ff_depth_toy_spec.md` §6 to every arm.)*
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

19. **The package/experiment BOUNDARY RULE (ratified 2026-07-20 by `docs/flexnn/package/plan.md` FP-0).**
    Authored in `docs/flexnn/package/plan.md` §2; stated here once because it binds every strand:
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
    is NOT a deletion and passes `docs/flexnn/package/PROTECTED.tsv`'s manifest check, because the path still
    exists.


<!-- from docs/plans/capacity_programme/MASTER.md lines 270-325 -->
20. **Training-schedule rule for nested/anytime models (user, 2026-07-21).** Train **every rung
    every step wherever all rungs are computable in ~one forward** — depth (exits read off the
    shared prefix) and ProbReg k (`NestedStrategy.all_rung_outputs`,
    `automl_package/models/flexnn/strategies/n_classes.py:207`). Where each extra
    rung costs a real forward — width, each width being its own matmul slice — use the certified
    **sandwich** (always min + max plus 2 random middles,
    `automl_package/examples/nested_width_net.py:93`). The **per-sample uniform draw is retired as
    a default schedule programme-wide**: it buys no compute where all-rungs is free, and width
    recorded its failure — the top rung trains only 1/k_max of the time
    (`docs/flexnn/width/history/width_dial_2026-07/EXECUTION_PLAN.md:119-129`; literature basis
    `docs/flexnn/width/history/width_dial_2026-07/nested_architecture_research_2026-07-11.md:104`). It survives
    only as a LABELLED comparison arm. Migration: `docs/probreg/shared/plan.md` P7 (k); depth tests both schemes as
    arms in `docs/flexnn/depth/plan.md` DSEL-2c. A strand deviating from this rule carries a written
    justification in its §1. The three dials previously ran three different schedules with the
    reason (compute cost) written down nowhere — that is what this decision repairs.

21. **Regularisation is explicit, never accidental (user, 2026-07-21).** The research training
    loops are otherwise unregularised — no weight decay, dropout, norm layers, or mini-batching
    (audited 2026-07-21; the library's levers all default off) — so cheapest-within-tolerance
    risks selecting small capacity because small OVERFITS LESS, not because small suffices: a bias
    identical across all three dials and invisible to cross-strand agreement. Every dial runs a
    **discriminating check** (weight decay λ ∈ {0, 1e-4, 1e-2} on its sweep reference; does the
    selected value move beyond tolerance?) BEFORE its battery is read: `docs/flexnn/width/plan.md` WSEL-11,
    `docs/probreg/global_k/plan.md` P8; depth inherits after its positive control passes (DSEL-2c). A check that
    moves the selection BLOCKS that strand's battery reads — strand-local, logged prominently,
    batched for end-of-run user review while the run continues elsewhere — never a silent footnote
    *(block semantics set 2026-07-21 with the autonomous-run authorization)*. Weight decay is the Gaussian prior it is — the
    no-arbitrary-penalty premise binds the SELECTION objective, not MAP training. A schedule's
    stochasticity (or its absence) must never be the de-facto regulariser. Baselines receive the
    same treatment or the comparison is not like-for-like.

    **AMENDED 2026-07-21 (user), after BOTH checks were found to have violated their own strand's
    constraint — a pattern, not an incident.** `docs/flexnn/width/plan.md` WSEL-11 *trained* on the Gaussian
    likelihood (fitting variance, which Decision 2 parks for width); `docs/probreg/global_k/plan.md` P8 *selected* on
    Gaussian NLL (a variance metric, which the σ-scope decision removed from that strand). They
    returned OPPOSITE verdicts and between them either cleared or blocked three downstream tasks.
    **Both are REOPENED and their results DISCARDED.** Binding additions:
    - **A discriminating check runs on its own strand's sanctioned objective and metric.** Width and
      the joint strand: mean-only, squared error. **ProbReg: the fixed-σ mixture log-likelihood —
      ⚠️ AMENDED 2026-07-21, see Decision 24.** Depth: classification. A check measured on a
      different objective from the arms it protects proves nothing about them.
      *(The original text here read "ProbReg: the point-prediction set." **That was wrong and it
      survived being written into this register**, which is how `docs/probreg/shared/plan.md` §0.5 and P8 both came to
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


<!-- from docs/plans/capacity_programme/MASTER.md lines 354-371 -->
23. **Reports are PARKED behind a joint results review (user, 2026-07-21).** Both live strands need a
    comprehensive report — `docs/flexnn/width/plan.md` **WSEL-10** and `docs/probreg/PLAN.md` **P6**, both authored via the
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


<!-- from docs/plans/capacity_programme/MASTER.md lines 523-559 -->
30. **THE SUITE BAR IS RE-BASELINED — measured 2026-07-21, by `docs/flexnn/package/plan.md` FP-12, NOT by P7
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


<!-- from docs/plans/capacity_programme/MASTER.md lines 615-632 -->
34. **EXHAUST-BEFORE-NEGATIVE (user ruling, 2026-07-23 joint review, verbatim intent: "This is
    an exercise in inventing a new architecture... we need to exhaust every possibility of
    improving the architecture... before we report a negative result").** WSEL-8's both-halves
    FAIL stands as a recorded measurement but is NOT report-final; the replacement line does not
    stop. `docs/flexnn/width/plan.md` WSEL-23 enumerates the improvement ladder — derived observation-model loss
    weighting as PRIMARY (probabilistically principled: weights 1/(σ² + a²(w)) fall out of an
    explicit per-width observation model, no arbitrary penalties; all variances FIXED, never
    learned), deployment-prior mixture, per-width private capacity, self-distillation — with a
    CHECKABLE exhaustion end-state; WSEL-10's replacement-claim section is gated on it. The
    generalization clause binds every candidate: parametric laws in w validated at ≥2 values of
    w_max, never per-width constants (§3.10 discipline, same as the router sizing rule). The
    ARBITER (the held-out error-table per-input judge the router is distilled from) is a NAMED
    reference in the report's comparison suite — the routing upper bound, shown with its measured
    noise limitation and the generator-true oracle beside it. Also ruled the same exchange, on
    the blend (recorded in `docs/flexnn/width/plan.md` WSEL-22 b′): the blend has NO threshold — it is purely a
    function of the out-of-sample per-width values; the bake-off's blend verdict is scoped to
    the implemented (thresholded-label) construction.


<!-- from docs/plans/capacity_programme/MASTER.md lines 671-819 -->
37. **TRAINING-BEFORE-SELECTION ORDERING (user, 2026-07-24).** Verbatim intent: "We need to find
    the best model that we have, the best training setup that we have. We get the results of that,
    then we try the selection. There is no point in trying the selection while we are still
    figuring out the training problem." **Binding pipeline for the width strand: settle the training
    recipe FIRST (WSEL-23 candidate 1 = the derived per-width weighting, aimed at the weak mid
    widths), THEN redo selection.** Consequence the user made explicit and RATIFIED: settling the
    training INVALIDATES every prior selection result — the band verdicts (WSEL-22), the
    router-backend bake-off (WSEL-19), the selection-fraction freeze (WSEL-6 / WSEL-6-R), the
    oracle-agreement numbers, and the "no free lunch" real-data battery were ALL computed on the
    CURRENT (unweighted, mid-width-weak) per-width held-out error tables; a training change changes
    those tables, so all of it is recomputed after training settles. Verbatim: "all the selection
    work done before... has to be done again after the training has settled." Rationale on record:
    selection is DISTILLED from the per-width error tables (MASTER Decision 13); selection-on-stale-
    tables is wasted work, and it explains why so much selection read as "no free lunch" (measured on
    a model with a broken middle). **De-queued until the weighted verdict lands:** the real-data
    selection-fraction battery (WSEL-6-R), the band experiments (WSEL-22 remainder), the
    threshold-free blend. The band CRITERION may be DESIGNED now (recorded in the WSEL-22 block); its
    EXPERIMENTS wait. The weighted run's embedded selection-agreement corollary (candidate-1 spec
    §5.2) is a PRELIMINARY signal about training, never a settled selection result.

38. **ATTACK NEGATIVES BEFORE RECORDING (user, 2026-07-24 — "why do you keep reporting failures
    that, when I review, turn out to have gaps or things done incorrectly? How do we break the
    cycle?").** Root cause named and accepted: the standing verification (selftest / ruff / gates)
    proves a task was EXECUTED correctly, but nothing attacks whether a NEGATIVE verdict is REAL —
    the adversarial pass has been happening at the user's review instead of before the record, and
    that displacement IS the cycle. A false negative closes a door SILENTLY and is unrecoverable
    downstream (verifiability asymmetry: adjudication kills false positives, cannot recover misses),
    so a negative deserves MORE scrutiny than a positive, not less. **Binding rule:** before ANY
    failure / negative / UNREACHABLE / NOT-ADOPTED verdict is recorded, or any INHERITED default is
    retained, the block must carry an on-disk `flip_attempt:` note answering — (a) the smallest change
    that would flip it, and whether it was actually tried; (b) any number already in the record that
    points the other way; (c) is the failure the phenomenon itself or an artifact of setup (capacity,
    sample size, scale, wrong metric); (d) for an inherited default, justified in terms of WHAT and
    why this value. A pre-registered "failure branch" is valid ONLY if the spec first enumerates the
    FULL lever-space and the branch fires after that space is SPENT, not after one axis — WSEL-21's
    "unreachable" fired after optimizer knobs alone while a ~7× data-lever improvement sat in the
    record as a footnote (the motivating instance). Negative verdicts route to the adjudicator tier.
    **ENFORCEMENT:** a negative-verdict check in `test_plan_gates.py` requires every block carrying
    the explicit `NEGATIVE-VERDICT` marker to hold a `flip_attempt:` field in the same block —
    forward-looking and zero-false-positive (a natural-language scan over the existing FAIL /
    UNREACHABLE / NOT-ADOPTED verdicts was considered and deferred as too brittle to run unattended;
    pre-existing verdicts predate the marker and the judgment rule, not the gate, governs their
    back-fill). The judgment rule binds whether or not the gate fires.

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
`docs/flexnn/joint/plan.md`/G-JOINT is OUT of the autonomous run's scope (strand-index row 4).

**Why per-wave and not one long programme branch:** four strands over weeks on a single branch
diverges from `master` far enough that the merge becomes its own risky project — which is precisely
the dangling-branch failure this protocol exists to prevent. Short branches merge trivially.

## Mechanical gates

Run from repo root:
`~/dev/.venv/bin/python -m pytest docs/test_plan_gates.py -q`
- **Citations**: any `path/with/slash(.py|.md|.json|.tex|.sh)` cited in MASTER/strands/shared
  must resolve on disk (`file:line` → line must exist). `archive/` exempt.
- **Dated names**: no date-shaped filename in this plan dir (`archive/` exempt).
- **Numbers**: every `RESULT:` line must contain a `.json` pointer. (This gate proves existence
  of a pointer, NOT currency of the number — the newest ledger entry wins only if you look.)
- Baseline is shrink-only: `gates_baseline.txt` lists grandfathered violations; the suite fails
  if new ones appear or the baseline grows.


