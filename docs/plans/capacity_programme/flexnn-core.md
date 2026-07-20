# Strand: FlexNN core — refactor, feedforward-depth pilot, unified report

**Ratified by user 2026-07-18 (live):** (1) **FlexNN is the umbrella** — all width/depth/joint
capacity work IS the FlexNN programme; the rigorous mechanisms get refactored INTO the package
FlexNN family. (2) Plan is execution-level; **orchestration runs in a SEPARATE session** — nothing
in this file is built in the planning session. (3) A **unified research report** covering all
flexible-architecture work (width + depth) with a **MoE comparison** supersedes the separate
reports (b) and (c) of `flexnn-moe.md` (M4/M5 fold into F7 here; M3 battery is UNGATED from
G-JOINT and rescoped as F6). (4) The **feedforward-depth demonstration** is pursued (spec-first).

**G-JOINT status carried:** BLOCKED after three dead construction families; J-3/R1-with-fixes spec
authored (`docs/joint_capacity/j3_toy_design.md` §9), direction (Option 1 vs 3) still a USER
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

**Standing clauses & environment:** per `MASTER.md` Rules. `AUTOML_DEVICE=cpu` on every run;
`~/dev/.venv/bin/python`; convergence trajectory rule on every conclusion
([[feedback_check_loss_trajectory_before_concluding]]). Reports: byline = user, no AI provenance.

**Ledger:** `automl_package/examples/capacity_ladder_results/D_TOY_PROBES/` (F5),
`automl_package/examples/moe_comparison_results/` (F6), `docs/reports/flexnn_unified/` (F7).

---

### Task F0: MASTER reframe — FlexNN umbrella naming + strand registration

**Files:**
- Modify: `docs/plans/capacity_programme/MASTER.md`
- Modify (if the gate test asserts strand rows): `docs/plans/capacity_programme/test_plan_gates.py`,
  `docs/plans/capacity_programme/gates_baseline.txt`

**Orchestration:** parallel: yes · deps: none · tier: sonnet · scale: static · shape: execution ·
verify: `~/dev/.venv/bin/python -m pytest docs/plans/capacity_programme/test_plan_gates.py -q` green

Spec: retitle MASTER "Capacity programme" → "FlexNN programme (flexible-capacity networks)";
add naming-key line: "**FlexNN** = the umbrella for ALL per-input capacity work; strands are its
width/depth/joint/MoE/core facets; the package family is `flexible_neural_network.py` +
`flexible_width_network.py` (F2)". Register this strand as row 6 (gated on: none; dispatchable).
Amend row 5: M3–M5 → "M3 rescoped as flexnn-core F6; M4/M5 superseded by F7 (user 2026-07-18)".
Update priority order: flexnn-core waves first; `width-depth.md` J0 marked PARKED (pending user
Option 1/3). Do NOT rewrite strand history or Decision register entries — append, never reword.

**Non-goals:** no edits to strand files other than MASTER (F6 owns the `flexnn-moe.md` amendment);
no RESUME/CHANGELOG edits (local-only).

### Task F1: package hygiene fixes in `flexible_neural_network.py`

**Files:**
- Modify: `automl_package/models/flexible_neural_network.py`
- Test: `tests/test_phase2_flexible_nn.py` (extend, don't rewrite)

**Orchestration:** parallel: yes · deps: none · tier: sonnet · scale: static · shape: execution ·
verify: `~/dev/.venv/bin/python -m pytest tests/test_phase2_flexible_nn.py -q` green; ruff clean

Spec (all evidence-backed, none speculative):
1. **Docstring fix** (line 29): active layers are the FIRST n blocks (`blocks[0..n-1]`); the last
   `max_hidden_layers − n` are skipped. Fix class + `FlexibleNNModule` docstrings to match code.
2. **ELBO prior → uniform default.** Replace `linspace(3,1)` categorical prior with uniform
   (`torch.zeros(max_hidden_layers)` logits) in `DepthRegularization.ELBO`; `COST_AWARE_ELBO`
   becomes uniform base − `cost_aware_lambda·cost` (keep the cost term). Rationale to cite in the
   commit: M0 revalidation measured complete depth-collapse to the linspace prior
   (`report_b_results/`, flexnn-moe.md Done ledger M0); identical trap removed from ProbReg's
   k-prior. Add a regression test: with ELBO on, depth posterior does NOT collapse to argmax of
   the prior on a 2-cluster synthetic where deep is needed for one cluster.
3. **HPO range:** `max_hidden_layers` high 3 → 6.
4. NO other behavior changes (REINFORCE/Gumbel/STE paths untouched).

**Non-goals:** no distillation (F3), no width (F2), no convergence gating (F4), no strategy
rewrites.

### Task F2: `FlexibleWidthNN` — port the certified width mechanism into the package

**Files:**
- Read FIRST: `automl_package/examples/nested_width_net.py:222` (`SharedTrunkPerWidthHeadNet` —
  the G-WIDTH certified pattern: shared full-width trunk, prefix-mask `state[:, w:]=0`, PER-WIDTH
  heads) + its training loop (joint multi-width loss) + selftest pattern;
  `automl_package/models/base_pytorch.py` (fit contract).
- Create: `automl_package/models/flexible_width_network.py` (`FlexibleWidthNN(PyTorchModelBase)`)
- Create: `tests/test_flexible_width_network.py`
- Modify: `automl_package/enums.py` (new `WidthSelectionMethod(StrEnum)`: `NONE` (fixed w),
  `DISTILLED` — closed set ⇒ enum, house rule)

**Orchestration:** parallel: yes · deps: F1 (same-file conflicts: none — new module) · tier:
sonnet · scale: static · shape: execution · verify: new test file green (fit/predict at each
width on synthetic; per-width heads present; prefix-nesting property: activations at width w are
a prefix of w_max's); ruff clean

Spec: constructor takes `widths: tuple[int,...]` (default `(16, 32, 48, 64)`), trunk depth/act
per existing FlexNN defaults; training = sum of per-width losses (the certified regime — lift
from the example, cite it in the docstring); `predict(x, width=...)` for fixed,
`inference_mode="routed"` via F3's router once F3 lands (until then raise NotImplementedError
with pointer). MSE + CE tasks (mirror FlexNN's criterion selection). Search-first clause: reuse
`nested_width_net.py` code by import where clean, else copy with provenance comment — state in
the PR note which was done and why.

**Non-goals:** no NestedWidthNet (#1, failed) or IndependentWidthNet (#3, control) ports; no
in-training width selection strategies (distillation only, Decision 13); no examples changes.

### Task F3: `DistilledCapacityRouter` — Decision-13 selection as a package API

**Files:**
- Read FIRST: `automl_package/examples/sinc_width_experiment.py`
  (`_cheapest_within_tolerance_labels`, `DELTA_TIE` — the programme's labeling rule, reused
  verbatim-in-spirit), `automl_package/examples/depth_selection_toy.py:590-601`
  (`_VectorRouterMLP` — vector-input router + why `ck6._RouterMLP` doesn't generalize),
  `automl_package/examples/capacity_ladder_k6.py:75` (router hyperparameter conventions).
- Create: `automl_package/models/common/distilled_router.py`
- Create: `tests/test_distilled_router.py`
- Modify: `automl_package/models/flexible_neural_network.py` (+`inference_mode="routed"`),
  `automl_package/models/flexible_width_network.py` (same)

**Orchestration:** parallel: no · deps: F1, F2 · tier: sonnet · scale: static · shape: execution ·
verify: test green: on a synthetic 2-regime task the routed model beats the worst fixed capacity
and its mean deployed cost < max capacity's; router never sees oracle difficulty labels; ruff clean

Spec: `DistilledCapacityRouter.fit(eval_fn, x_val, y_val, capacity_grid, tolerance, cost_fn)` —
`eval_fn(x, capacity) → per-sample error`; builds the held-out per-capacity error table; labels =
cheapest capacity within `tolerance` of each input's best (lift the sinc rule); trains a small
MLP router on raw inputs → capacity index; `route(x) → capacity`. Grid is a list of tuples so the
SAME class serves 1-D (depth), 1-D (width), and 2-D (w, T) later — no joint-specific code now.
Cost default: analytic FLOPs per capacity via `shared/metrics-accounting.md` S2 accounting
(import the S2 helpers; do not re-derive).

**Non-goals:** no in-training routing; no transformer/token routing; no changes to selection
strategies.

### Task F4: convergence gating — promote the trajectory rule into the package

**Files:**
- Read FIRST: `automl_package/examples/convergence.py` (the `diverged`/trustworthy trajectory
  rule every certified result uses).
- Create: `automl_package/utils/convergence.py` (moved logic; keep a thin re-export in
  `automl_package/examples/convergence.py` so example imports keep working — verify by grepping importers)
- Modify: `automl_package/models/flexible_neural_network.py`,
  `automl_package/models/flexible_width_network.py` — expose `self.convergence_summary_` from
  `_fit_single` val-loss history; add `trustworthy` to `get_params()` output dict.

**Orchestration:** parallel: yes · deps: F2 · tier: sonnet · scale: static · shape: execution ·
verify: `grep -rn "import convergence\|from convergence" automl_package/ tests/` all resolve;
relevant test files green; a deliberately-diverging fit (lr=10) yields `trustworthy=False`

**Non-goals:** no gating inside HPO/CV loops (surface the flag only); no changes to other models.

### Task F5: feedforward-depth pilot — spec, ADJUDICATOR GO, build, run (2 seeds)

**F5a — author the spec** (`docs/depth_capacity/ff_depth_toy_spec.md`), then **⛔ ADJUDICATOR GO
gate (user away — ratified 2026-07-18: no user questions mid-run).**
[[feedback_toy_design_needs_reviewed_spec]] is satisfied unattended as follows: the written spec
+ adversarial review are still MANDATORY before any build; the GO call is made by an adjudicator
(Opus/xhigh) instead of the user. Verdict SOUND, or SOUND-WITH-FIXES whose fixes are mechanical →
fold fixes, log the verdict, BUILD. Verdict UNSOUND, or any fix that is a PI design decision →
**PARK F5b** (do not improvise, do not redesign), log it under "Batched user questions", and
carry on with every other unblocked task. Spec + review verdict are delivered for post-hoc user
review either way.

**Files (F5a):** Create: `docs/depth_capacity/ff_depth_toy_spec.md`
**Orchestration (F5a):** parallel: yes · deps: none · tier: opus/high (design) + adversarial
review pass (adjudicator) · scale: static · shape: design · verify: spec contains generative
math, the 2×2 grid below, pre-registered bars with numeric thresholds, confound ledger
(param-count-grows-with-depth for untied arms → param-matched reads mandatory), review verdict §

Design core (settled in main session 2026-07-18, to be made numeric in the spec): task = A5
word composition, L=10, involution generators (reuse `depth_composition_toy.py` machinery — the
certified substrate; input = flattened one-hot word, 50-dim). **2×2 architecture grid** isolating
WHY depth wins there:

| | flat input (all letters at layer 0) | per-step input (letter t at layer t) |
|---|---|---|
| **untied weights** | plain deep MLP — *the user's original claim* | untied unrolled stack |
| **tied weights** | tied stack on flat input | = `RecurrentComposer` (already certified ≥0.90) |

plus the certified wide-shallow control. Depth ladder for stack arms: d ∈ {4, 7, 10}, width 64.
Bars (freeze numerically in spec): FF-CLAIM passes iff the untied-flat arm at some d ≤ 10 reaches
held-out ≥ 0.90 while param-matched wide-shallow ≤ 0.60 (the graded-pilot fit/stall thresholds);
attribution read = which cells generalize, ordered; convergence-gated; 2 seeds. Either outcome is
reportable: PASS → flexible depth demonstrated on a straightforward feedforward net where depth
is provably non-substitutable; FAIL with tied cells passing → weight-tying is the load-bearing
ingredient (goes in F7 + informs the transformer roadmap: weight-tied = the Universal
Transformer shape, arXiv:1807.03819).

**F5b — build + run: RAN 2026-07-18 → ⛔ INVALID (2026-07-20). Superseded by F5c.**
The build landed (`NetKind.TIED_FLAT`, `UNTIED_PERSTEP` + builders + driver flags in
`automl_package/examples/depth_composition_toy.py`) and 28 runs produced
`.../D_TOY_PROBES/ff_depth_pilot_a5_seed{0,1}.json`. **No bar may be read from them**, on three
independent grounds, each verified on disk 2026-07-20:
1. **Positive control FAILED (MASTER Decision 14).** Cell 4 (`RecurrentComposer`, the certified
   arm, made a MANDATORY confirm run by the F5a reviewer precisely to validate the protocol) is
   trustworthy on **1 of 2 seeds**: seed 1 = 0.9257 clean; seed 0 climbed to 0.830 by epoch 3000
   then **collapsed to 0.097**, `diverged=true`. Spec §6 requires ≥2 trustworthy seeds. The
   protocol did not reproduce a known-good result ⇒ nothing else in the battery is readable.
2. **Protocol parity breach (MASTER Decision 15).** These runs used `depth_composition_toy.py`'s
   `train_clf`, which applies **no gradient clipping**, at A5/L=10 — while
   `automl_package/examples/depth_selection_toy.py` sets `GRAD_CLIP_MAX_NORM = 1.0` commented
   "L=10 needs clipping to stay GD-trainable". Single unswept `LR = 1e-2`, no schedule/warmup,
   across arms spanning 12,476–89,660 params.
3. **Gate/bar metric mismatch (MASTER Decision 17).** `trustworthy` is computed on val CE; the
   bars read val accuracy. Val CE explodes (overconfidence) while val accuracy sits flat.
4. **Decision 9 never satisfied.** Runs self-terminated by patience at ~2,750 epochs against a
   40,000 cap; the required early-stop-OFF confirmation at ≥4× budget was never run.

**Recorded observations (NOT findings — no bar attaches):** untied-flat fails to fit the TRAIN
set at d=7 (train 0.195) and d=10 (train 0.055) ⇒ under-fit per Decision 16, an optimization
signal, not an architecture verdict. Shallow (d=4) and deep (d=10) arms fail *differently*
(instability/overconfidence vs. no learning at all) — one protocol fix may not cover both.

### Task F5c: depth protocol repair + diagnosis (replaces F5b's run; added 2026-07-20, user-approved)

**Staged — each stage GATES the next. No stage may be skipped or reordered.**

**F5c-a — spec first (~30 min), then ⛔ ADJUDICATOR GO.**
Create: `docs/depth_capacity/ff_depth_protocol_repair_spec.md`. Must contain: (i) a line-by-line
**training-loop diff** between `depth_composition_toy.py::train_clf` and `depth_selection_toy.py`'s
loop, every difference justified or removed (Decision 15); (ii) the **escalation ladder** — which
remedies, in what fixed order, with the stopping rule (Decision 16); (iii) **instrumentation
plan** for per-layer gradient norms (the vanishing/exploding hypothesis must be MEASURED, not
asserted — no artifact currently exists for it anywhere in the repo); (iv) the convergence gate
recomputed on **val accuracy** as well as CE (Decision 17); (v) bars **unchanged** from
`ff_depth_toy_spec.md` §6 — FIT 0.90 / STALL 0.60, frozen (Decision 9: no bar moves after a run).
*Orchestration:* parallel: yes · deps: none · tier: opus/high design + adjudicator review ·
shape: design · verify: spec contains all five items + a review verdict §.

**F5c-b — positive control ALONE (Decision 14 gate).** `RecurrentComposer`, A5, L=10, 2 seeds,
repaired protocol, plain single-readout (same `train_clf` protocol as the grid arms).
**PASS BAR: held-out ≥ 0.90 AND `trustworthy=true` on BOTH seeds.** FAIL ⇒ **HALT**, escalate the
ladder, re-run this stage only. **No compute is spent on any other arm until this passes.**
*Orchestration:* deps: F5c-a GO · tier: sonnet · verify: 2 JSONs, both ≥0.90 and trustworthy.

**F5c-c — instrumented diagnosis of the untied arms.** Only after F5c-b passes. Log per-layer
gradient norms across training for untied-flat d ∈ {4,7,10}; land them as an artifact. Answers
the actual question: does the deep untied stack starve (vanishing) or destabilize (exploding),
and are these two different failure modes? *Orchestration:* deps: F5c-b PASS · tier: sonnet ·
verify: a grad-norm artifact JSON/CSV exists per depth; the mechanism claim cites it (new Rule:
no claim without an artifact).

**F5c-d — the 2×2 grid re-run + verdict.** Only after F5c-b/c. Full matrix per `ff_depth_toy_spec.md`
§9 under the repaired protocol, incl. the Decision 9 early-stop-OFF confirmation at ≥4× budget for
any load-bearing cell. Append the results § to `docs/depth_capacity/verdict_per_input_depth.md`
(the record that F5b owed and never wrote — it records what ran and what is quarantined, and
renders a verdict ONLY where a bar is validly evaluable).
*Orchestration:* deps: F5c-c · tier: sonnet · scale: static · shape: execution · verify: JSONs
exist AND every reported cell is `trustworthy=true` on ≥2 seeds AND bars evaluated in the verdict §;
runs detached per environment rules.

**Non-goals:** no L > 10 (MOD-1 GD wall), no new groups, no selection/router in the pilot
(substrate question only), no curriculum tricks (future work if FF fails).

### Task F6: FlexNN-vs-MoE comparison battery (rescopes M3; UNGATED from G-JOINT)

**Files:**
- Modify: `docs/plans/capacity_programme/flexnn-moe.md` (mark M3 → superseded-by-F6, M4/M5 →
  superseded-by-F7, with date + this file as authority)
- Modify: `automl_package/examples/moe_regression.py` (add `task={mse,ce}` flag: CE head
  `Linear(H→n_classes)` per expert, CrossEntropyLoss, same router/aux-loss — needed because the
  depth toy is CE)
- Create: `automl_package/examples/moe_flexnn_comparison.py` (driver) · Create (by runs):
  `automl_package/examples/moe_comparison_results/*.json`

**STATUS 2026-07-20: NOT RUN.** The driver `automl_package/examples/moe_flexnn_comparison.py`
(564 lines) and the `moe_regression.py` CE-task flag both landed, but the battery was **never
executed** — `automl_package/examples/moe_comparison_results/` does not exist. A prior session
carried this as substantially done; it is not (new Rule: DONE = `verify:` executed).
**Decisions 14 + 15 now bind this task:** before the grid runs, diff the driver's training loop
against the loops that certified the width and depth results, and run a known-good arm first as
the protocol gate.

**Orchestration:** parallel: no · deps: F2, F3, F4, M2 (done) · tier: sonnet · scale: static ·
shape: execution · verify: JSONs cover the grid with convergence flags + MoE collapse
diagnostics per cell; matching arithmetic printed into every JSON (S2), never hand-waved

Grid (inherits M3's pre-registered hypotheses H-flex/H-moe; constants frozen in-driver before
runs): toys = certified width toy config (from `width-cert.md` ledger), `make_hetero3`, A5 depth
toy (CE). Contenders = `FlexibleHiddenLayersNN` (routed, post-F1/F3), `FlexibleWidthNN` (routed),
MoE top-2 (top-1 ablation), val-selected best fixed. Two matching regimes (total params AND
executed FLOPs). 3 seeds minimum (S3 stats); tuned-α rerun clause before any "MoE fails" claim
(carried from M5 verbatim). **Joint toy dropped** (G-JOINT open — grid covers the two certified
1-D dials only; state this in the driver docstring).

**Non-goals:** no MoE variants beyond frozen M1 config; no transformer MoE; no real data; no
HPO beyond the tuned-α clause.

### Task F7: unified research report — FlexNN: flexible-capacity networks (+ MoE comparison)

**Files:**
- Create: `docs/reports/flexnn_unified/` (own folder; md + PDF per
  [[reference_pdf_build]] pandoc/xelatex/FreeSerif)

**Orchestration:** parallel: no · deps: F5b, F6, M0 (done) · tier: session-model draft +
adjudicator cold-reads · scale: static · shape: execution · verify: `research-report` skill gates
pass (self-contained, cold-readable, no code refs); every number traces to a ledger JSON; byline
= user, zero AI provenance; supersession of reports (b)/(c) noted in `flexnn-moe.md`.
**Contingency (unattended):** if F5b is PARKED or F6 partially fails, F7 still runs on the
evidence that DID land — missing pieces become explicit "pending" statements, never blockers.

Structure (execution-level TOC): (1) The FlexNN idea — one network, per-input capacity;
(2) Width: the shared-readout break + certified per-width-heads result (G-WIDTH, verdict §10);
(3) Depth: why smooth toys can't show depth (negative note), the group-composition result
(G-DEPTH D5/D8b), the feedforward 2×2 pilot outcome (F5) — whichever way it lands;
(4) Selection = distillation — THREE certified instances of the same law (in-training selection
fails; held-out distillation works): width (W-strand), depth (D8), and **k (K6 — the SOFT
responsibility-labelled router ≤ global-k on 9/9 audited cases; arbiter-not-knee readout, R2)**;
incl. the honest ELBO-collapse refutation (M0) and the ProbReg linspace-prior precedent;
cross-reference the F10 k-selection report rather than re-deriving; (5) MoE comparison (F6
numbers, symmetric where-wins table); (6) Open problems: G-JOINT three-family post-mortem (J-1/J-2 substrate,
group-order confound, R1 entanglement) stated as open, NOT as failure theater; (7) Roadmap:
per-token/per-module capacity in transformers — our depth substrate is the Universal-Transformer
shape (weight-tied looped layer + per-position halting, Dehghani et al. arXiv:1807.03819,
verified 2026-07-18); our width dial maps to MoE experts; the joint dial maps to
Mixture-of-Depths(+MoE = "MoDE") per-token top-k depth routing (Raposo et al. 2024,
arXiv:2404.02258, verified 2026-07-18); endpoint = flexible encoder/decoder modules trained as a
small Python-code LM (staged, see F8).

**Non-goals:** no new experiments (ledger numbers only); no UCI/real data; no claims past the
toys tested (boundary stated explicitly); no G-JOINT verdict (it is open).

### Task F9: ProbReg dynamic-k port — the SAME refactor FlexNN is getting (added 2026-07-18)

**Files:**
- Read FIRST: `automl_package/examples/_capacity_ladder_nested.py` (K4 nested-k surrogate — the
  VALID prefix ladder, trained per-sample `k ~ Uniform{1..k_max}` = the user's k-dropout scheme,
  strictly probabilistic, no penalties), `automl_package/examples/capacity_ladder_k6.py` (K6
  distilled router π(x): SOFT responsibility-labelled router ≤ global-k on 9/9 audited cases),
  `automl_package/examples/capacity_ladder_results/{R2_verdict.md,RESULTS.md}` (arbiter — not
  knee — is the faithful per-input readout; toy E = honest negative 2/3 seeds),
  `automl_package/models/probabilistic_regression.py` (current in-training dynamic-k).
- Modify: `automl_package/models/probabilistic_regression.py`
- Test: extend the relevant ProbReg test file (locate via `tests/test_phase3_dynamic_k.py` first)

**Orchestration:** parallel: yes (disjoint from F6) · deps: F3 · tier: sonnet · scale: static ·
shape: execution · verify: relevant ProbReg tests green; a synthetic 2-regime case shows the
distilled-k routed model ≥ the best global fixed k on held-out NLL (the K6 result, reproduced
through the package API); ruff clean

Spec: (1) add a **nested-k training mode** to the package model implementing the K4 scheme
(per-sample uniform k over the renormalized first-k prefix; component 1 = bypass rung; no k
input to the network) — lift semantics from `_capacity_ladder_nested.py`, cite it; (2) wire
**per-input k at inference via `DistilledCapacityRouter`** (F3) with K6's SOFT
(responsibility-style) labels as the labeling rule — k is just another capacity axis on the
same API; (3) demote the in-training `NClassesSelectionMethod`/ELBO selection to labeled
comparison arms in docs (they remain functional; the distilled path is the recommended one);
(4) expose the held-out arbiter read as a diagnostic utility (the certified readout — eff-count
tiles, knee collapses; R2). Known boundary to state in docstrings: adaptive-mode drift cases
(toy-E-like) are NOT certified — 2/3 seeds negative.

**STATUS 2026-07-20: LOGIC LANDS, DEVICE BUG OPEN.** Verified on disk: both F9 deliverable tests
(`TestNestedKTraining::test_nested_bypass_rung_differs_from_mixture_rung`,
`TestNestedKDistilledRouting::test_distilled_router_matches_or_beats_best_global_fixed_k`) **PASS
on CPU** (42 s) and **FAIL on XPU** with `Expected all tensors to be on the same device`.
**Added deliverable (F9-fix):** repair the device placement. Strongest candidate found by
inspection — `automl_package/models/probabilistic_regression.py:246-247` creates
`unique_k`/`probabilistic_indices` via `torch.tensor([])` with **no `device=`**, on CPU; the
reassignment at line 258 happens only inside the `if is_ce_active:` branch, so the CPU tensors
survive on the other path. Also tidy `automl_package/models/common/distilled_router.py:121`,
whose `device: ... = "cpu"` default is the only hardcoded device string in `models/` (against the
repo rule); all three call sites already pass `self.device`, so it is a latent trap, not the
active bug. **Verify: the two tests pass under BOTH `AUTOML_DEVICE=cpu` and a non-CPU device.**
Note this bug class is invisible to the suite as normally run — everything passes on CPU while
the library claims CUDA/XPU support.

**Non-goals:** no change to fixed-k behavior or report-(a) configs; no Basis-B work (separate
open research item); no removal of existing selection strategies; no variational-EM harness
port (research instrument, stays in examples).

### Task F10: ProbReg k-selection report — the missing report (added 2026-07-18)

**Files:**
- Read FIRST: `automl_package/examples/capacity_ladder_results/RESULTS.md` (audited results of
  record, incl. K4/K5/K6 + R1–R4 verdicts + independent review 2026-07-10),
  `docs/reports/probreg_kselection/historical/probreg_kselection_findings.md` (moved there by this
  task) + `docs/kselection_variational_em_2026-06-13/` (June
  Basis-A/variational-EM arc — historical context to absorb, not the results of record).
- Create: `docs/reports/probreg_kselection/` (own folder; md + PDF per [[reference_pdf_build]])
- Modify: move/supersede the loose `docs/probreg_kselection_findings.{md,pdf}` into the new
  folder (no stray files in docs/ root)

**Orchestration:** parallel: yes (pure writing; all numbers landed) · deps: none · tier:
session-model draft + adjudicator cold-reads · scale: static · shape: execution · verify:
`research-report` skill gates pass; every number traces to `capacity_ladder_results/` artifacts;
byline = user, no AI provenance

Gap this fills (verified 2026-07-18): report (a) covers ProbReg-vs-baselines ONLY — zero
mentions of k-selection/arbiter/router. The July ladder programme (K4 nested-k prefix ladder =
the k-dropout scheme; R2 arbiter-not-knee; K6 distilled per-input router 9/9) exists only as
working verdicts + a June-era findings note that PREDATES the ladder certification. Structure:
the k question (resolution dial, not component count); why prefix-masking an unordered mixture
failed (R1); the nested-k construction; the arbiter readout; the distilled router result;
honest negatives (toy E 2/3 seeds, V3 under-resolution ceiling); relation to the FlexNN
selection law (pointer to F7). ProbReg is framed as a CLASSIFIER over k classes throughout
([[feedback_explain_probreg_as_classifier]]).

**Non-goals:** no new experiments; no Basis-B (open research, listed as future work); no
package-API content (F9's docstrings own that).

### Task F11: roadmap groundwork dossiers R5a + R5b (promoted from the roadmap; runs Wave 1)

**Files:**
- Create: `docs/roadmap_groundwork/python_instruction_datasets.md` (R5a) and
  `docs/roadmap_groundwork/dgx_spark_training_envelope.md` (R5b) — own folder, two dossiers.

**Orchestration:** parallel: yes (two independent task-workers, one per dossier) · deps: none ·
tier: sonnet · scale: static · shape: research · verify: every factual claim carries a source
URL fetched THIS run (WebSearch/WebFetch — no memory citations); unresolved items listed as
OPEN, never asserted

R5a brief: inventory public Python instruction-level coding datasets. Seed anchors (verified
2026-07-18): CodeAlpaca-20k, Evol-CodeAlpaca-V1, Magicoder-OSS-Instruct-75K (arXiv:2312.02120),
OpenCodeInstruct. Must resolve: license per dataset (The Stack v2 terms explicitly),
size/token counts, generation provenance (human vs model-generated — flags for contamination),
dedup practice, and a recommended Python-only corpus mix with total token estimate.
R5b brief: DGX Spark training envelope. Verified base specs (2026-07-18): GB10, 128 GB unified
LPDDR5X, ~1 PFLOP FP4, 31 TFLOPS FP32, 4 TB NVMe. Must resolve (sourced): BF16/FP8/NVFP4
*training* (not inference) support and realistic sustained throughput, memory budget arithmetic
for training (params+optimizer+activations) at candidate sizes {~125M, ~350M, ~1B}, scaling-law
token budgets (cite the law used), wall-clock estimates, and a go/no-go read on "1B-class Python
model trainable on one unit".

**Non-goals:** no purchases/recommendations of specific vendors; no dataset downloads; no
architecture content (R4's job).

### Task F8: transformer/coding-model roadmap amendment

**Files:**
- Modify: `docs/research_plan.md` §5 + Paper C paragraph

**Orchestration:** parallel: yes · deps: none (content finalizes after F7 drafts §7 — light
touch-up allowed then) · tier: sonnet · scale: static · shape: execution · verify: §5 keeps the
"no sequence transformers IN THE TABULAR PACKAGE" verdict intact and adds the staged path
without contradicting it

Spec: append the staged path (each stage gated on the previous): (i) tabular FT-Transformer
block-depth gating (existing Paper C, unchanged) → (ii) weight-tied sequence-transformer toy
with per-token distilled halting on an algorithmic task (the D8-style router, UT-shaped;
sits OUTSIDE the tabular package, own examples module) → (iii) per-module (encoder/decoder
block) flexible capacity, MoD/MoDE-style top-k routing with OUR distilled (not in-training)
router as the comparison axis → (iv) **probabilistically-grounded architecture research
(ratified theme, 2026-07-18):** start from the STANDARD transformer block (for baseline
comparability), then explore replacing its arbitrary components (Q/K/V projections, heuristic
gating, tuned auxiliary penalties) with statistically/probabilistically motivated constructions —
the programme's standing theme (no arbitrary penalties; coherent probabilistic objectives; cf.
ProbReg ELBO, distilled cheapest-sufficient routing). Research aim, stated as a BET not a
promise: can probabilistic grounding of capacity allocation yield large efficiency gains over
SOTA MoE transformers? Requires a dedicated **literature review task** (deep-research fan-out:
probabilistic/kernel/Bayesian interpretations of attention, principled routing, adaptive-compute
theory — every anchor verified, none from memory) BEFORE any architecture is proposed →
(v) small Python-code LM demonstration (compute plan TBD — explicitly flagged as needing
hardware beyond this box). Cite the two verified anchors (UT arXiv:1807.03819, MoD
arXiv:2404.02258).

**Non-goals:** no implementation; no package scope change; no promises of LM-scale results on
current hardware (state the constraint honestly).

### Task F12: ProbReg benchmark — shared-k vs variable-k vs baselines, on REAL data (added 2026-07-20, user-ratified)

**Authority:** MASTER Decision 3 as amended 2026-07-20 (user, live). This is the scope change
from "reports are toys-only" — it binds THIS task and F10's results §, not the width/depth reports.

**Files:**
- Read FIRST: `automl_package/examples/model_comparison.py` — the EXISTING baseline-comparison
  harness (XGBoost / LightGBM / PyTorch NN / ProbReg on 3 synthetic sets, Apr 2026). **Extend it;
  do not reinvent** (repo search-before-write rule). Also `docs/research_plan.md` §6 (dataset
  dossier, incl. published baselines) and §4 (the missing-baselines analysis).
- Create: `docs/probreg_benchmark/benchmark_spec.md` (F12-a, spec first)
- Create: `automl_package/examples/probreg_benchmark.py` (driver)
- Create (by runs): `automl_package/examples/probreg_benchmark_results/*.json`

**F12-a — spec first, then ⛔ ADJUDICATOR GO** (same discipline as F5c-a; a benchmark grid
improvised mid-run is exactly what invalidated F5b). Spec must freeze, BEFORE any run: the model
set, dataset list + provenance, metrics, split protocol, seed count, and **how tuning budget is
matched across models** (an unmatched comparison is not evidence — `research_plan.md` §3 fixes a
per-model-per-dataset wall-clock budget; reuse it rather than inventing one). Pre-register what
result would count as a win, a loss, and a null, for each claim.

**Models (6).** (1) **shared-k** ProbReg — fixed `n_classes`; (2) **variable-k** ProbReg —
per-input k via `DistilledCapacityRouter` (F9). Treated as two DISTINCT models per the MASTER
Naming key. Baselines: (3) XGBoost, (4) LightGBM, (5) CatBoost, (6) standard PyTorch NN. *(The
wider UQ baseline set — NGBoost, MDN, Deep Ensembles, quantile NN — is `research_plan.md` §4
future work, NOT this task.)*

**Datasets — real, in two tiers.** Tier 1 (tests OUR specific claims, run first):
`Kepler Objects of Interest` (NASA Exoplanet Archive; predict planet radius — genuinely
**multimodal** across rocky / sub-Neptune / Jupiter populations, the strongest real test of the
classification-bottleneck idea), `UCI Superconductivity` (n=21,263, d=81; heavy-tailed T_c 0–185 K
⇒ real symlog test; published XGBoost RMSE ≈ 9.5 K to beat), `UCI Airfoil Self-Noise` (n=1,503,
d=6; heteroscedastic). Tier 2 (literature comparability, Hernández-Lobato & Adams 2015 protocol):
Concrete, Energy, Naval, Power Plant, Protein/CASP, Wine red+white, Yacht. **Boston excluded**
(removed from scikit-learn 1.2). **Every dataset URL in `research_plan.md` §6 must be re-verified
live before download** — no-guessing rule; they were written 2026-04.

**Metrics:** RMSE + NLL + CRPS + calibration (PIT / interval coverage). The headline read is
**variable-k vs shared-k** (does per-input k pay on real data?) and **both vs the tree baselines**.
Prior evidence to reproduce-or-contradict, not to assume: K6 found the per-input router ≤ global-k
on **9/9** synthetic cases (`automl_package/examples/capacity_ladder_results/RESULTS.md`).

**Orchestration:** parallel: yes (disjoint from all F-tasks) · deps: **F9-fix** (variable-k must
work on the run device) · tier: opus/high spec + adjudicator review, then sonnet execution ·
scale: static · shape: design → execution · verify: spec has a review verdict §; every run
convergence-gated (Decisions 9/17); a known-good configuration passes first (Decision 14); results
JSONs carry per-seed numbers, not just means.

**Non-goals:** no astrophysics headline analysis (that is Paper A/B); no NGBoost/MDN/Deep-Ensemble
baselines; no HPO beyond the frozen matched budget; no width/depth content (F7 owns that).

---

## Ultimate-aim roadmap (R-stages) — ratified outline 2026-07-18

Endpoint: a small Python-only instruction-level coding LM trained on DGX-Spark-class hardware
with a flexible-capacity (FlexNN-principled) architecture, benchmarked against dense + MoE
baselines at matched compute. Status flags: **SCOPED** = execution-level, dispatchable;
**PARTIAL** = key artifacts exist, a bounded design round remains; **OUTLINED** = direction +
gates known, needs its own spec/design round before any build.

- **R0 — FlexNN core (this strand, F0–F8): SCOPED.** The only fully execution-level stage.
- **R1 — G-JOINT joint dial: PARTIAL.** J-3/R1 spec + required fixes enumerated
  (`docs/joint_capacity/j3_toy_design.md` §9); decision DEFERRED to post-F5b (see waves) because
  a passing feedforward substrate may offer an entanglement-free joint toy (needs its own
  confound ledger — design round, spec-first).
- **R2 — sequence bridge: OUTLINED.** Weight-tied/looped sequence model + per-token DISTILLED
  halting on an algorithmic sequence task (UT-shaped; D8 router pattern). Needs: toy spec
  (task, bars, confound ledger) per [[feedback_toy_design_needs_reviewed_spec]].
- **R3 — transformer capacity dials at toy scale: OUTLINED.** Standard block first (baseline
  comparability); small MoD / MoE / MoDE reproductions; OUR distilled router as the comparison
  axis vs their in-training routers. Needs: M1-style convention freeze from primary sources +
  design doc.
- **R4 — probabilistically-grounded architectures: OUTLINED (research-shaped).**
  R4a deep literature review (deep-research fan-out; every anchor verified) — DELIBERATELY
  DEFERRED to this stage's gate: the adaptive-compute/MoE field moves fast, a review done now
  decays before it is consumed. R4b candidate constructions + falsifier toys — unscopable until
  R4a lands (by design, not neglect).
- **R5 — endpoint groundwork (research, CHEAP, runs EARLY — dispatchable alongside R0):**
  - **R5a dataset research: SCOPABLE NOW.** Inventory public Python instruction-level coding
    data; verified anchors to seed it (2026-07-18): CodeAlpaca-20k, Evol-CodeAlpaca-V1,
    Magicoder-OSS-Instruct-75K (arXiv:2312.02120), OpenCodeInstruct (large-scale). MUST verify:
    licenses (esp. The Stack v2 terms — NOT confirmed yet), contamination/dedup practice,
    dataset mix + token counts for a Python-only corpus. Deliverable: a sourced dataset dossier.
  - **R5b hardware envelope: SCOPABLE NOW.** DGX Spark verified 2026-07-18: GB10 (20-core Arm +
    Blackwell GPU, 6144 CUDA cores), 128 GB unified LPDDR5X, ~1 PFLOP FP4, 31 TFLOPS FP32, 4 TB
    NVMe (nvidia.com/dgx-spark; docs.nvidia.com/dgx/dgx-spark/hardware.html). Deliverable:
    trainable model-size/token-budget arithmetic (scaling-law based, sources verified), training
    precision support (NVFP4/BF16), wall-clock estimates for candidate model sizes.
  - **R5c eval harness choice: SCOPABLE NOW** (verify current standard Python coding evals +
    contamination policy; do not assert from memory).
- **R6 — the model: UNSCOPED (by design).** Pretrain + instruction-tune the small Python LM with
  the flexible-capacity architecture; benchmark vs dense + MoE at matched compute. Gated on
  R2–R5.
- **R7 — publications: PARTIAL.** Unified report = F7 (scoped); paper-form deliverables follow
  `docs/research_plan.md` (amended by F8).

**Research-now-vs-later rule (ratified):** run R5a/R5b(/R5c) early — cheap, slow-decaying,
de-risks the endpoint and sizes every intermediate decision; defer R4a to its gate — expensive
and fast-decaying. R2/R3 design rounds happen at their own stages under the spec-first rule.

## Dispatch waves (for the orchestration session)

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
