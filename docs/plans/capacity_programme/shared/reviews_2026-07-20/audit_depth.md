# Depth workstream audit vs. the ProbReg rubric

Scope: `docs/plans/capacity_programme/depth.md`, `docs/depth_capacity/ff_depth_protocol_repair_spec.md`,
depth sections of `docs/plans/capacity_programme/flexnn-core.md`,
`automl_package/models/flexible_neural_network.py`,
`automl_package/models/independent_weights_flexible_neural_network.py`,
`automl_package/models/common/distilled_router.py`,
`.../D_TOY_PROBES/f5c_poscontrol_a5_seed{0,1}.json`.

---

## Q1 — The three choices (arbiter / router / sweep)

**VERDICT: PARTIAL — (ii) PRESENT in library code but never applied to the depth research; (i) ABSENT; (iii) ABSENT.**

**(i) cheap global choice / arbiter-equivalent — ABSENT.** `grep -rln "arbiter" automl_package/` returns
files for ProbReg (`probabilistic_regression.py`, `n_classes_strategies.py`) and the WIDTH strand
(`capacity_ladder_results/R2_verdict.md` etc.) only. No depth file, doc, or class implements a
held-out single-global-depth arbiter. `flexnn-core.md` never mentions an arbiter for depth. There is
no depth analogue of `held_out_arbiter_advantage` anywhere.

**(ii) per-input choice / distilled router — PRESENT in the package, but disconnected from the depth
research.** `automl_package/models/flexible_neural_network.py:437-500` (`FlexibleHiddenLayersNN.fit_router`)
builds a per-depth held-out error table via `forward_at_depth` (`:173-185`) and fits a
`DistilledCapacityRouter` (`automl_package/models/common/distilled_router.py:108-232`); `predict(...,
inference_mode="routed")` (`:386-435`, dispatch at `:422-425`) consumes it. This is tested in
`tests/test_phase2_flexible_nn.py` (`test_routed_predict_shape_and_no_nan` etc., lines 508-548) — but
only on synthetic fixtures (`simple_linear_data`), never on the A5/S5 depth substrate the D-tasks
built. **Separately, `automl_package/examples/depth_selection_toy.py` implements its OWN, unrelated
router** (`_VectorRouterMLP`/`fit_router` at `depth_selection_toy.py:607-657`) — same name, same
"cheapest-within-tolerance" idea, but a completely separate, non-imported implementation (verified:
`grep -n "^import\|^from" depth_selection_toy.py` shows no import of `automl_package.models.*`
anywhere in the three depth toy scripts). **The package's distilled-router mechanism (F3, Decision 13)
was never run on the depth research's own certified substrate, and the depth research's certified
D8b router result was never produced with the package's `DistilledCapacityRouter`.** These are two
independently-built, never-reconciled implementations of the same idea.

**(iii) expensive global sweep reference — ABSENT (weaker substitute exists).** ProbReg's M3 trains a
**separate model per k** and scores each held out (`probreg.md:34`, `:487-503`). Depth has no
analogue: D8b's "best-fixed-T" (`depth.md:239-244`, `verdict_per_input_depth.md:313-317`) is **the
same single anytime net, evaluated at each exit T** (val-selected among its own exits), not
independently-retrained models per depth. `grep` for a depth-specific per-capacity sweep driver
analogous to `select_k_for_toy` (`probreg.md:34`) found none. D7's param-matched wide-shallow run
(`depth.md:197-227`) trains separate models per **width**, not per **depth**, and for a different
purpose (the width-substitution falsifier, not a selection reference).

## Q2 — The two mechanisms: shared-weight (recurrent) vs. independent weights per layer (feed-forward)

**VERDICT: The plan's "recurrent"/"feed-forward" vocabulary maps to TOY-SCRIPT classes, NOT to the
package classes with confusingly similar names. Verified, not assumed.**

Toy-script classes (`automl_package/examples/depth_composition_toy.py`):
- `RecurrentComposer` (`:337-364`) — **one shared 2-layer block** applied `seq_len` times:
  `state = tanh(block([state, onehot(g_t)]))` per step, same `self.block` weights reused every step.
  Docstring: *"Unrolling depth = seq_len with SHARED weights."* This is the "recurrent" arm.
- `UntiedPerStepComposer` (`:367-393`) — identical shape but `seq_len` **distinct** blocks, one per
  step (`self.blocks = nn.ModuleList([...])`). Docstring: *"weight-tying is the sole architectural
  difference from RecurrentComposer."*
- `TiedFlatComposer` (`:397+`) and `build_narrow_clf`/`build_wide_shallow_clf` (`:321-336`, plain
  `nn.Sequential` MLP stacks) round out the 2×2 grid (`flexnn-core.md:174-190`).

**All results recorded in `depth.md` and `verdict_per_input_depth.md` — D1b, D5, D7, D8 — were produced
by these toy-script classes**, confirmed by ledger paths (`depth_graded_pilot_*`, `depth_selection_*`,
`ff_depth_pilot_*`, `f5c_poscontrol_*`), never by the `automl_package.models` package.

Package classes, mapped precisely because their names invite the wrong mapping:
- `FlexibleHiddenLayersNN` (`flexible_neural_network.py:29`) is **neither** "recurrent" **nor**
  "untied-flat" in the toy sense. Its `hidden_layers_blocks` (`:121-131`) are `max_hidden_layers`
  **distinct** blocks (own weights each), and depth-n runs a **prefix** `blocks[0:n]` (docstring
  `:29-39`, verified against `forward_at_depth` `:173-185`). This is a "growing/nested" network —
  a third, distinct mechanism from both toy arms.
- `IndependentWeightsFlexibleNN` (`independent_weights_flexible_neural_network.py:31`) is **not**
  "independent weights per layer within one deep net" (the toy's `untied-flat`/Cell-1 sense). It is
  `self.independent_networks` (`:127-149`) — **N wholly separate, independently-parameterized full
  networks**, one per possible depth `1..max_hidden_layers`, chosen by the selection strategy. This is
  a fourth, distinct mechanism (closer to a discrete mixture-of-depths than to either toy arm).

**No plan document ever states this mapping explicitly, and the package classes are never named in
`depth.md` or the `docs/depth_capacity/` directory** (`grep -rln "FlexibleHiddenLayersNN\|
IndependentWeightsFlexibleNN" docs/plans/capacity_programme/depth.md docs/depth_capacity/` → no
match). Anyone reading "depth" as "the FlexNN depth mechanism" and reaching for
`IndependentWeightsFlexibleNN` because of its name would get architecture #4 above, which has never
been evaluated by any depth bar in this programme.

## Q3 — Lesson 1: is each depth arm a complete system (selection cost included)?

**VERDICT: PARTIAL, and weaker than ProbReg's binding rule.** The D8b deploy battery does report an
inference-time cost proxy — executed mean-T **8.00/7.99 vs best-fixed-T 10**
(`verdict_per_input_depth.md:313-317`) — so the deployed *inference* saving is genuinely inside the
reported number, matching the spirit of ProbReg's "selection is inside the model, never a
side-analysis" rule (`probreg.md:23-28`). But: (a) there is no depth "M1"/arbiter arm at all to be
scored end-to-end against (Q1-i), so the *cheap-global* half of the lesson has nothing to apply to;
(b) the router's own **training/fitting cost** (building the per-T error table + training the MLP
router) is never reported alongside the deploy number — only the deployed-step count is; contrast
ProbReg's explicit binding that M3 (the expensive reference) reports its "full cost... next to its
accuracy" (`probreg.md:553-555`) — no such accounting exists for depth's "best-fixed-T" reference,
because there is no independently-trained-per-depth reference to cost (Q1-iii).

## Q4 — Lesson 2a: selection-set-size study

**VERDICT: ABSENT.** `grep -in "slice.b\|selection.set\|selection fraction\|data.floor"` over
`depth.md` and `docs/depth_capacity/*.md` finds only the fixed, unswept use of "slice B" as a held-out
split (`depth.md:239,243`) — never a sweep of its size, no data-floor measurement, no equivalent of
ProbReg's PB task (`probreg.md:384-403`). The router in `depth_selection_toy.py` is trained on
`n=3000 words/stratum` (`verdict_per_input_depth.md:290`), a value chosen for wall-clock reasons
("40k = ~7h/pilot on CPU, infeasible" — `depth.md:382`), not measured as a selection-data
requirement.

## Q5 — Lesson 2b: router architecture sensitivity

**VERDICT: ABSENT.** No sensitivity table, no swept router width/depth/epochs for the depth selector.
`_VectorRouterMLP` (`depth_selection_toy.py:607`) hyperparameters are inherited fixed constants
("reuse HIDDEN/N_EPOCHS/LR router hyperparameter conventions" from `capacity_ladder_k6.py`,
`depth_selection_toy.py:88`), never varied or tested for invariance. The package's
`DistilledCapacityRouter` likewise ships one frozen default (`distilled_router.py:57-60`:
`DEFAULT_HIDDEN=(32,32)`, `DEFAULT_N_EPOCHS=300`, `DEFAULT_LR=1e-2`) with no depth-specific
sensitivity check — the only sensitivity evidence anywhere in the programme is the WIDTH strand's
(`probreg.md:411-413` cites `width-cert.md:234`, explicitly a different strand, one dimension, a
does-it-break check not a search — and that's WIDTH, not depth).

## Q6 — Lesson 3: API and silent-failure paths

**VERDICT: PRESENT but with the same silent-failure class ProbReg's PA task found and fixed
elsewhere, plus one that's worse.**

- `FlexibleHiddenLayersNN.predict(self, x, filter_data=True, inference_mode="soft")`
  (`flexible_neural_network.py:386`) — default `"soft"`. **Forget to pass `inference_mode="routed"`
  after calling `fit_router()` and you silently get the un-routed model** — identical failure class
  to the one `probreg.md:337-339` documents for ProbReg's pre-PA API ("Forget that flag and you
  silently get the un-routed model... the default is 'soft', there is no error").
- `inference_mode` is validated by a bare string membership check
  (`flexible_neural_network.py:397: if inference_mode not in ("soft", "hard", "routed")`) — no enum,
  the same "breaks the repo's own closed-set rule" defect `probreg.md:340-342` flags for ProbReg's
  `inference_mode`.
- `predict_uncertainty(self, x, filter_data=True)` (`flexible_neural_network.py:502`) **has no
  `inference_mode` parameter at all** — even with a fitted router, uncertainty is *always* computed
  from the in-training soft-selection strategy. This is a stricter silent-failure than ProbReg's
  (which at least required, and could omit, a flag on both `predict` and `predict_uncertainty`,
  `probreg.md:336`); here the routed path for uncertainty simply doesn't exist.
- `IndependentWeightsFlexibleNN` has **no `fit_router`, no `DistilledCapacityRouter` import, no
  `inference_mode` parameter anywhere** (`independent_weights_flexible_neural_network.py`, full file
  read — `predict`/`predict_uncertainty`/`predict_proba` at `:388-517` take no such argument). Calling
  `.fit_router()` on this class raises `AttributeError`; there is no distilled-selection path for it
  at all. A caller migrating from `FlexibleHiddenLayersNN` to this "independent-weights" sibling
  loses the entire selection mechanism silently until they try to use it.
- **Convergence gating (F4) landed asymmetrically.** `FlexibleHiddenLayersNN._fit_single` builds
  `self.convergence_summary_` via `ConvergenceTracker` (`:378-382`) and surfaces `trustworthy` in
  `get_params()` (`:612`). `IndependentWeightsFlexibleNN._fit_single` (`:219-379`, full method read)
  has **no `ConvergenceTracker` usage at all**, and its `get_params()` (`:381-386`) does not include a
  `trustworthy` key. F4's "promote convergence gating into the package" (`flexnn-core.md:138-153`)
  reached one twin class, not the other.
- **A live, unfixed regression-collapse bug survives in the untouched twin.** `flexnn-core.md:69-75`
  (Task F1) documents removing the `linspace(3,1)` prefer-shallow ELBO prior from
  `FlexibleHiddenLayersNN` because "M0 revalidation measured complete depth-collapse to the linspace
  prior" — confirmed fixed at `flexible_neural_network.py:299-309` (now `torch.zeros(...)`, uniform,
  comment cites the removal explicitly). **`IndependentWeightsFlexibleNN` still has the old, buggy
  prior**, verbatim: `depth_prior_logits = torch.linspace(3.0, 1.0, self.max_hidden_layers, ...)` at
  `independent_weights_flexible_neural_network.py:306` (ELBO branch) and again at `:312`
  (`COST_AWARE_ELBO` branch, `base = torch.linspace(3.0, 1.0, ...)`). F1's fix was applied to one
  class and never ported to its sibling — this is a live defect, not a documented decision (no
  comment, no task references it), found by reading the file the brief specified in full.

## Q7 — Lesson 4: comparison and report

**VERDICT: ABSENT on disk; planned but not run.** No XGBoost/LightGBM/CatBoost/standard-NN comparison
for depth exists anywhere (`grep -in "baseline\|XGBoost\|LightGBM\|CatBoost"` over `depth.md` and
`docs/depth_capacity/*.md` returns zero comparator-baseline hits — only "surface-baseline control,"
which is a covariate probe for whether t* is surface-detectable, not a competing model). The closest
planned artifact is `flexnn-core.md` Task F6 (FlexNN-vs-MoE comparison battery, includes the A5 depth
toy under a CE task) — **STATUS explicitly "NOT RUN"** (`flexnn-core.md:305-308`: *"the battery was
never executed — `automl_package/examples/moe_comparison_results/` does not exist"*), confirmed:
`ls automl_package/examples/moe_comparison_results/` → No such file or directory. Task F7 (unified
report) depends on F6 and the now-invalid F5b (`flexnn-core.md:334`); its output directory
`docs/reports/flexnn_unified/` also does not exist (confirmed by `ls`). **No depth-vs-baseline, no
depth-vs-expensive-reference report exists on disk.**

## Q8 — The halt (F5c positive control failure)

**VERDICT: reconstructed precisely from the repair spec and the two JSON artifacts.**

**What the positive control measures.** `RecurrentComposer` (weight-shared, `depth_composition_toy.py:337-364`)
on A5/L=10, trained under the **repaired, single-exit, plain protocol** — the same `train_clf` loop the
grid arms (untied-flat MLP stacks) use — as a Decision-14 gate: *if the plain protocol can't even drive
the certified architecture to its known-good performance, nothing else in the battery is readable*
(`flexnn-core.md:233-237`).

**What the 0.90 bar is a bar on.** Held-out (val) word accuracy of `RecurrentComposer` under that
protocol, ≥0.90 AND `trustworthy=true`, on both seeds (`flexnn-core.md:235`, restated unchanged from
`ff_depth_toy_spec.md` §6 per `ff_depth_protocol_repair_spec.md` §5).

**What the repair changed, itemized (`ff_depth_protocol_repair_spec.md` §1.2, table rows 1/2/3/9/12):**
- LR: `1e-2` (unswept, hard-wired, `depth_composition_toy.py:93/:415`, no CLI override) → `3e-3`
  (matches every certified A5/S5 result's recorded `lr`).
- Gradient clipping: **none** → `clip_grad_norm_(..., max_norm=1.0)`, matching
  `depth_selection_toy.py:114/:477`'s own comment "L=10 needs clipping to stay GD-trainable."
- Convergence gate: CE-only → **dual** CE-and-accuracy gate (`ff_depth_protocol_repair_spec.md` §4),
  with best-weights restored on **accuracy**, not CE.
- Widths/depths/seeds: implicit module defaults → explicit, recorded in the output JSON.
- (Row 4 — single-exit supervision — was **kept**, not changed, per the F5a reviewer's explicit ruling
  that Cell 4 must train under the same plain protocol as the grid cells, `ff_depth_protocol_repair_spec.md:82`.)

**RESULT, verified directly from the JSONs**
(`capacity_ladder_results/D_TOY_PROBES/f5c_poscontrol_a5_seed{0,1}.json`):

| seed | train_acc | val_acc | fit_status | trustworthy_acc | trustworthy_ce | diverged (acc) |
|---|---:|---:|---|---|---|---|
| 0 | 0.9697 | 0.4324 | MEMORIZED | True | False | False |
| 1 | 1.0000 | 0.7442 | INTERMEDIATE | True | False | False |

Both seeds pass the accuracy convergence gate cleanly (`acc_gate.converged=true`, `diverged=false`,
`hit_cap=false` for both — read directly from the JSON) but **fail the 0.90 bar** (0.432 and 0.744).
Neither shows a late collapse under the repaired gate — `val_acc_trajectory` plateaus for both (seed 0:
~0.42-0.43 from epoch ~4000 to stop at 10500; seed 1: ~0.74 from epoch ~5000 to stop at 7500) — while
`ce_gate.diverged=true` for both (CE climbs monotonically after its minimum: seed 0 best 1.52@1500 →
trajectory continues rising; seed 1 best 1.37@3750 → rising). This matches the flexnn-core.md narrative
exactly: overconfidence (CE divergence) with a flat, sub-bar accuracy plateau, not an accuracy
collapse.

**Comparison to the pre-repair numbers, verified against the review's own citation
(`ff_depth_protocol_repair_spec.md:701`): pre-repair (loop T, lr=1e-2, no clip) `val 0.7581` (seed 0)
/ `0.9257` (seed 1).** Seed 1 went from a clean pre-repair pass (0.9257 ≥ 0.90) to a post-repair fail
(0.7442) — exactly the "0.926 → 0.744" the brief describes, now traced to source.

**Escalation ladder vs. the failure mode.** §2 of the repair spec pre-registers a fixed-order ladder:
L1 LR sweep → L2 LR warmup → L3 init scheme → L4 normalization/residual (flagged as a separate labelled
arm, not a repair). Every one of L1-L3 is explicitly an *optimization* remedy — the spec's own words:
"zero architectural change" (§2.3 table, L1/L2/L3 rows). **The spec itself states the ladder cannot
address this failure**: "both seeds already SATISFY Decision 16's exoneration condition
(`train_acc ≥ 0.90`) while failing the §5 positive-control bar... every rung... is an optimization
remedy, but at train_acc 0.97/1.00 the optimizer has already fit the training set completely; there is
no optimization failure left to fix" (`flexnn-core.md:255-259`, restated in
`ff_depth_protocol_repair_spec.md §2.5` mechanics). This is memorization-without-generalization — the
Decision-16 `fit_status` field computed from the artifacts is literally `MEMORIZED` (seed 0) /
`INTERMEDIATE` (seed 1, train=1.0 but val=0.744 which is above the 0.60 memorization-stall threshold
so it's not classified `MEMORIZED`) — neither is `UNDER_FIT`, which is the only status the ladder is
built to escalate on (`ff_depth_protocol_repair_spec.md` §4.4 table: `UNDER_FIT` is the sole
ladder-triggering status; `MEMORIZED`/`INTERMEDIATE` are not).

**Current state, per the plan text itself:** `flexnn-core.md:263-269` — *"⛔ ESCALATED TO USER — do NOT
resolve this without a ruling... F5c-c/F5c-d remain HALTED either way."* The spec's own M6/M7
machinery (a discriminator experiment: re-run `depth_selection_toy.py`'s certified anytime
configuration to check whether it still reproduces ≥0.90 — reproduced ⇒ substrate/pipeline intact and
single-exit supervision is the substantive finding; not reproduced ⇒ environmental/regression, no claim
permitted) is explicitly gated behind L3 exhaustion, and running it before L1 is called "a spec
deviation" needing a ruling (`flexnn-core.md:268-269`). No document claims the ladder's remedies *could*
in principle fix memorization-without-generalization; the spec explicitly argues the opposite.

---

## Also report

### Claims in depth.md / flexnn-core.md not backed by an artifact on disk

- **`flexnn-core.md:17-25`** (main-session review of `flexible_neural_network.py`, dated 2026-07-18)
  states the ELBO prior issue and its fix as a plan item, but does not itself constitute an artifact —
  the fix WAS verified on disk for `FlexibleHiddenLayersNN` (see Q6) but **the same review never
  checked `IndependentWeightsFlexibleNN`**, which still has the bug (Q6 finding) — an omission, not a
  false claim, but the plan's F1 "DONE" framing (implied by F5's later tasks proceeding) does not
  distinguish the two classes anywhere.
- No artifact backs a claim that the depth package classes (`FlexibleHiddenLayersNN`,
  `IndependentWeightsFlexibleNN`) implement the certified D5/D8 mechanisms — because no plan document
  makes that claim explicitly, but its absence is easy to assume given the strand's own framing
  ("FlexNN is the umbrella... width/depth/joint capacity work IS FlexNN", `flexnn-core.md:3-4`,
  echoing `MASTER.md`/[[project_flexnn_is_the_umbrella]]). The umbrella claim is aspirational
  (refactor-target), not a statement that the port has happened for depth — confirmed: no depth toy
  imports the package (Q2), and F2/F3's "port the certified width mechanism" (`flexnn-core.md:82-137`)
  is scoped to **width**, not depth; there is no equivalent depth-port task anywhere in `flexnn-core.md`
  (`grep -n "^### Task F"` shows F0-F13, none titled or scoped as a depth port).

### Stated status of the depth work

**depth.md's own header** (`depth.md:1`): *"Strand: depth (→ G-DEPTH) — COMPLETE 2026-07-17. G-DEPTH =
D5 substrate ∧ D8 selection = PASS."* This certification is real and independently verified on disk
(D5 substrate: `verdict_per_input_depth.md` §§0-9, D7 param-matched wide-101 stall; D8 selection:
§§10-12, S1-S4 pass 2/2 seeds, S5 covariate) — **but it rests entirely on the toy scripts' multi-exit /
graded-supervision training loops** (loop **G** = `depth_graded_toy.py::_train_mixed`, loop **S** =
`depth_selection_toy.py::train_anytime`, per `ff_depth_protocol_repair_spec.md` §1.1 table), using
`RecurrentComposer` with **shared-readout, multi-length or prefix-supervised** training.

**The currently-HALTED F5c work (`flexnn-core.md` Task F5/F5c, dated 2026-07-18 through 2026-07-20) is
a separate, later, narrower claim**: whether a **plain deep feed-forward MLP** (untied-flat, Cell 1 of
the 2×2 grid) can learn the SAME task under **single-exit** supervision, with the wide-shallow control
as its falsifier. This claim has never been evaluated — F5b's 28 runs are formally invalid
(`flexnn-core.md:192-215`, four independent grounds) and F5c's own positive control (testing whether
even the *certified* `RecurrentComposer` reaches 0.90 under the plain single-exit protocol) has now
FAILED on both seeds (Q8). **The G-DEPTH "COMPLETE/PASS" certification does NOT rest on this failed
run and is not invalidated by it** — they exercise the same architecture class under materially
different training loops, and the repair spec says so explicitly (§1.1: "Loop T... carries no
certified A5/L=10 result at all" — loop T is F5b/F5c's loop, distinct from G and S which carry the
D5/D8 certifications). **But the "flexible depth demonstrated on a straightforward feedforward net"
claim that F5 was designed to produce (`flexnn-core.md:187-190`) is currently neither PASS nor FAIL —
it is HALTED, escalated to the user, pending a ruling on running the M6 discriminator ahead of the
prescribed ladder order** (`flexnn-core.md:263-273`).

---

## GAPS IF DEPTH WERE HELD TO THE PROBREG BAR

Most serious first.

1. **No arbiter arm exists for depth at all (Q1-i).** ProbReg's whole three-model M1/M2/M3 structure
   has no depth analogue — depth has a router (M2-shaped) and a weak sweep-substitute (a val-selected
   exit of one shared net, not M3's independently-trained-per-k reference), but nothing resembling M1.
   Any "depth selection efficiency" claim modeled on ProbReg's "M1 reaches M3's answer at a fraction of
   the cost" (`probreg.md:36-39`) is currently unaskable for depth.

2. **The package's distilled-router mechanism and the depth research's certified router are two
   unreconciled implementations (Q1-ii, Q2).** `FlexibleHiddenLayersNN.fit_router` has never been run
   on A5/S5; `depth_selection_toy.py`'s router has never been ported to the package. Neither Lesson 2a
   nor 2b (Q4, Q5) has been measured for either implementation. Under ProbReg's PB/PC discipline this
   pair of gaps alone would block any comparison battery from running (`probreg.md:227-231`: "PB and PC
   must precede P3/P4... running the battery before they are settled would produce results nobody
   could attribute").

3. **A live regression bug (the `linspace(3,1)` depth-collapse prior) survives, uncited, in
   `IndependentWeightsFlexibleNN` after being fixed in its sibling class (Q6).** This is exactly the
   "silent-wrong-answer class" ProbReg's D1/D2/PA sections treat as blocking
   (`probreg.md:128-146,338-342`) — here it has no doctrine (D-number), no test, and no plan-document
   mention at all.

4. **No baseline or expensive-reference comparison report exists (Q7).** The one planned artifact (F6,
   FlexNN-vs-MoE) is confirmed not run; its dependent unified report (F7) does not exist. ProbReg's
   equivalent (P4/P6) is scoped, gated, and pre-registered even though also not yet executed — depth
   has no equivalent gated task at all, only the stale F6/F7 dependency chain.

5. **The halted positive control (Q8) means the newest, narrowest depth claim (plain feedforward depth,
   F5) has no path forward under the currently-authored spec** — its own escalation ladder is
   architecturally incapable of addressing the observed failure mode (memorization without
   generalization), and the spec says so in its own words. This is not a hidden gap; it is
   correctly flagged and halted in the plan — but it means "depth is COMPLETE" (the header) and "the
   feedforward depth demonstration's outcome" are two different claims at two different levels of
   confidence, and a reader skimming only the header would miss that the second is unresolved.
