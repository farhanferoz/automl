# Strand: width+depth joint (→ G-JOINT) — G-DEPTH ✓ (2026-07-17); J0 Step 1 DONE, pilots (Step 2) next

**Goal:** one network serving a per-input 2-D capacity dial (width AND depth), built on what
G-WIDTH and G-DEPTH each certified.

**Settled inputs (2026-07-17 — J0 designs FROM these, does not re-derive them):**
- **The two certified toys live in DIFFERENT regimes** — width: 1-D MSE regression with
  analytic σ² floors; depth: S5 word classification, CE on held-out words
  (`automl_package/examples/depth_graded_toy.py`). "Cross the two toys" is therefore a DESIGN
  PROBLEM (J0's job), not a given; the earlier assumption that both sides were regression is
  void.
- **Asymmetric readout prior (from the refuted transfer prediction, `depth.md`):** width needs
  PER-WIDTH heads (shared readout fails — G-WIDTH); depth wants ONE shared readout over a
  weight-shared recurrent state (strongest arm 3/3 — G-DEPTH). Default joint pattern:
  per-width heads × depth-shared readout. The old "full (w,d) head grid vs factorized" question
  is now asymmetric, not symmetric.
- **Depth dial mechanics:** per-input depth = iteration count of a weight-shared block.
  **Depth selection WITHOUT an oracle is certified (or killed) in the depth strand — `depth.md`
  Task D8** (anytime net + router + deploy, the width protocol re-derived for depth on a
  hidden-depth-hunger toy). J0 CONSUMES D8's certified mechanism; this strand owns only the
  transformer-specific per-token/sequential halting question (`docs/research_plan.md` §5) and
  the 2-D crossing.
- **Stability caveat:** shared readouts showed 1-of-6 optimization blow-ups in the depth
  battery; the divergence guard (`depth.md` D6, `automl_package/examples/convergence.py`) must
  be landed BEFORE any J-battery runs, and every J-battery cell reads the new `diverged` flag.
- **Metric regime is decided by toy choice in J0** (MASTER Decision 2 amendment: MSE binds the
  width strand only; the joint strand's metric follows its adopted toy and is recorded in the
  Decision register when J0 closes).
- **Selection mechanism = DISTILLATION (USER doctrine 2026-07-17, MASTER Decision 13):** the
  per-input (width, depth) policy is DISTILLED post-hoc from held-out per-capacity error
  tables — the certified width mechanism (`automl_package/examples/sinc_width_experiment.py`
  `_fit_selector_mse`), re-certified for depth in `depth.md` D8. This is the trajectory for the
  transformer endpoint too: per-token depth is learned as a distilled function of the input,
  not by in-training selection (which failed for width and was refuted for FlexNN depth, M0).
  J* batteries use distilled routers as the primary; in-training selectors only as labeled
  comparison arms.

### Task J0: design the joint toy + expand this strand to execution level ⛔ (orchestrator/opus, same turn G-DEPTH lands)

**Orchestration:** parallel: no · deps: G-DEPTH (depth.md D5) + depth.md D6 landed · tier:
opus/main · scale: static · shape: discovery · verify: this file contains dispatchable tasks
J1..Jn each carrying the five contract elements; adopted-toy decision logged.

**AUTONOMY CONTRACT (user-ratified 2026-07-17):** J0 runs without user questions, under this
pre-registered protocol:
- [x] Step 1 (+ Step-1.5 arithmetic, MOD-5): **DONE 2026-07-17 → `docs/joint_capacity/joint_toy_design.md`.**
      EXACTLY 3 candidates, priority order **J-1 → J-2 → J-3**, each with dial semantics, metric +
      floor/ceiling analog (Bayes-accuracy per (w-demand, T-demand) cell + width info-floor — MOD-2), a
      pre-registered pilot bar set (S1/W-bar/D-bar/X-bar) + kill criterion, and a cost estimate (~20–40
      min/pilot ≪ 2 h). The candidate set HEDGES the one real risk (is the width dial genuine, or does a
      narrow readout free-ride on a full-width computation): J-1 (readout-width) vs J-2 (block-width) on
      the same data; J-3 is a different width mechanism (variable group order). **J-1 arithmetic VERIFIED**
      (`docs/joint_capacity/j1_arithmetic_check.py`, reproducible): A⊥T* pearson −0.028, 12/12 cells
      populated, class balance 60/60, sharp joint Bayes knee, all T*≤10 (MOD-1). J-3 needs its own
      order⊥length probe before its pilot.
- [ ] Step 2 (NEXT SESSION = orchestration): pilot each candidate at seed 0 in priority order (detached,
      ≤4 concurrent, HOLD until box load reasonable). No bar adjusted after its pilot runs. **Pilot
      artifacts:** implement the toy as `automl_package/examples/joint_capacity_toy.py` (reusing
      `depth_selection_toy`/`nested_width_net`/`convergence`; `--selftest`/`--probe` modes like the depth
      toy); pilot JSONs → `automl_package/examples/capacity_ladder_results/J_TOY_PROBES/joint_*_seed0.json`.
- [ ] Step 3: adopt the FIRST candidate that fully passes its own bars (priority J-1→J-2→J-3); log the
      adoption + one-line rationale in `RESUME.md` `### Decisions`.
- [ ] Step 4: rewrite this file in place to execution level (J1..Jn: nets, driver, battery, bars, gate
      rule — to `width-cert.md`/`depth.md` standards, citing only opened files), including the
      pre-registered G-JOINT gate rule (drafted in `joint_toy_design.md` §6, MOD-4: beats best-fixed-(w,T)
      AND both marginal routers on compute-matched accuracy).
- **Escalation (the ONLY sanctioned stop):** all 3 candidates die on their own bars → STOP the
  strand, write the post-mortem into this file, and batch the redesign question for the user
  (mirrors the depth strand's D1 once-then-escalate rule). Do not invent a 4th candidate
  autonomously.

**Design spec = `docs/joint_capacity/joint_toy_design.md` (Step 1 deliverable, 2026-07-17).** It carries
the joint net (§2: recurrent block, prefix-masked state = width dial, unroll count = depth dial, per-width
heads × shared readout), the 3 candidates (§3), the pre-registered bars (§4), the J-1 arithmetic (§5,
verified), and the G-JOINT gate rule (§6). Durable constraints baked in, each from a landed result:
- **MOD-1** depth axis ≤ L=10 (A5 GD-trainable wall). **MOD-2** joint toy is CE-classification, not MSE
  (depth must be group-structural → no analytic σ² floor; ceiling = Bayes-accuracy per cell + width
  info-floor). **MOD-3** inherit Option A — do NOT chase hidden demand (routing may be surface-easy; the
  science is the 2-D crossing). **MOD-4** G-JOINT gate rewards capacity-MATCHING and must beat BOTH
  marginal routers. **MOD-5** the 2-D orthogonality + starvation arithmetic probe is MANDATORY before any
  pilot (done for J-1; J-3 owes its own).
- **Pilot regime (user-set 2026-07-17):** SEMI-autonomous — the user reviews `joint_toy_design.md`, then
  Step-2 pilots run autonomously. **Box-load gate:** pilots wait until the 22-core shared box load is
  reasonable (was ~34).

**Non-goals (J0):** no FlexNN/package model changes (examples-level only until the flexnn-moe
strand); no transformer; no real-data benchmarks (MASTER Decision 3).

**Gate decision (G-JOINT): BLOCKED — J0 candidates J-1/J-2 dead at the substrate; J-3 needs a reviewed
redesign. Escalated to the user (2026-07-17 eve).** See post-mortem below. Gate RULE (unchanged,
pre-registered in `joint_toy_design.md` §6 / `docs/joint_capacity/joint_frozen_bar_spec.md`): adopted toy passes
S1 + W-bar + D-bar + X-bar on ≥2 seeds.

---

## J0 post-mortem (2026-07-17 eve) — parallel-multi-track construction is unfittable; decision required

**What ran.** `automl_package/examples/joint_capacity_toy.py` built (J-1 readout + J-2 block), verified
(selftest, §5 arithmetic reproduced, ruff clean), frozen bars applied
(`docs/joint_capacity/joint_frozen_bar_spec.md`). Seed-0 pilots + one diagnostic on CPU. Per-candidate dossier:
`docs/joint_capacity/dossier_J1_readout_width.md`.

**J-1 (readout-width): DEAD — S1 substrate failure.** Trustworthy convergence
(`automl_package/examples/capacity_ladder_results/J_TOY_PROBES/joint_readout_seed0.json`:
`converged=True, diverged=False, hit_cap=False`; val-CE best
0.465 @ epoch 8000 of 10500, flat tail → genuine ceiling, trajectory-checked). Per-track held-out acc @
(w=64, T=10) = **0.58–0.79 across all (A,T\*) cells, below the 0.90 S1 bar**, degrading monotonically with
A and T\*. **Even A=1 (single active track) = 0.79** vs the depth toy's ≥0.90 on the *same* single-track
A5 task. Discriminating diagnostic (`docs/joint_capacity/diag_fullwidth_only.py`, loss on w=64 only): **0.41–0.61,
worse** — so removing the width grid did not help; the multi-track fold, not grid dilution, is the
bottleneck. **Root cause:** the 4-slot construction feeds a 20-dim per-step input (15 dims NOOP noise even
at A=1) through ONE shared 64-wide block that must route 4 independent tracks without interference — a
multi-task-interference / input-multiplexing failure, not an A5-composition or capacity-bit limit.

**J-2 (block-width): DEAD by shared substrate.** Code-verified: at w=64 the mask is a no-op, so J-2's
full-width fold is *identical* to J-1's; block-mode only adds mid-fold masking burden to the shared block
→ J-2 S1 ≤ J-1 S1 < 0.90. J-2's purpose (force width into the computation to kill a *free-ride*) does not
address an S1 *substrate* failure. **Block-mode confirmation LANDED (2026-07-17):
`automl_package/examples/capacity_ladder_results/J_TOY_PROBES/joint_block_seed0.json`, `pilot_pass=False`, converged trustworthily
(`diverged=false`, best-val 1.038 CE @ epoch 3250) — per-cell S1 = 0.23–0.415, WORSE than J-1's 0.58–0.79
exactly as deduced (block-masking adds burden to the shared fold). J-2 dead, confirmed.**

**J-1/J-2 share the parallel-A5-track width mechanism — so that mechanism is dead for the joint dial.**
The 1-D width strand's dial (prefix-nested regression) and the depth strand's dial (single-track
recurrent fold) do NOT compose into a fittable *multi-track* substrate.

**J-3 (single-track, group-order width) — a genuine NEW toy design, not a drop-in probe.** It is
single-track (K=1), so it reuses the PROVEN single-track fold (≥0.90 in the depth toy) and structurally
avoids the multi-track bottleneck. But its width dial needs design, not improvisation:
- The width-demand is **width-to-COMPUTE the group multiplication** (`depth_composition_toy.py:302-303`:
  width-16 gives only 0.46 val for A5 → A5 needs a wider block; a simple group needs less), NOT
  width-to-hold-the-answer (one element is ~log₂|G| ≤ 6 bits, trivially held). So a genuine width dial
  (narrow for simple groups, wide for A5) is *plausible*.
- **Confound:** width-to-compute-multiplication scales with group **complexity/solvability**, which is the
  SAME axis that drives depth-hardness (Barrington / Liu et al.; `depth_composition_toy.py:24,116`). J-3
  must disentangle a genuine per-input WIDTH demand from the solvability→DEPTH demand, or the two dials
  collapse. This is the confound ledger the design's §3 J-3 risk ("group-order and commitment length
  correlate") pointed at, now sharpened.
- Needs new machinery (a nested chain of A5 subgroups, orders ∈ {2,3,4,5,6,10,12,60}) + its own
  order⊥length + solvability-confound arithmetic probe (MOD-5) BEFORE any pilot.

**DECISION REQUIRED (batched for user — [[feedback_toy_design_needs_reviewed_spec]]: no toy improvised
mid-run).** Options:
1. **Author a proper J-3 design spec** (group-complexity width dial + solvability/depth confound ledger +
   arithmetic probe), review, then build+pilot. Highest chance of a genuine joint result; needs a design
   round.
2. **Fix the multi-track substrate** instead of abandoning J-1/J-2: wider state (128), per-track sub-blocks
   (relaxes "one block"), or fewer tracks (K_MAX=2). Each is a design change with its own charter cost
   (e.g. per-track blocks weaken the "one network" claim).
3. **Re-scope G-JOINT** to a narrower, honestly-stated claim (e.g. depth-only per-input dial with a fixed
   width), or park the joint dial and proceed to the MoE reports (`flexnn-moe.md` M3–M5, currently gated on
   G-JOINT).

Recommendation: **(1)** — J-3 is the design's own next candidate and the single-track substrate is proven
fittable; the width-vs-solvability confound is the crux to nail in the spec. Awaiting user GO on the
direction; the finding above is complete and reversible (nothing committed).

**✅ RULED 2026-07-21 (user-delegated to the root): direction = Option 1, timing = DEFERRED.** The
J-3 design spec is authored only after (i) `depth-selection.md` DSEL-2c/DSEL-2 resolve whether the
feed-forward group-word substrate carries a depth signal at all — J-3 builds on that same substrate
family, and designing on it now would build on sand — and (ii) `width.md` WSEL-8 and
`depth-selection.md` DSEL-10 land their selection verdicts, which the joint dial consumes. The spec
itself REMAINS user-gated (reviewed-spec-before-build; no toy improvised). This strand is OUT of
the 2026-07-21 autonomous run's scope, and this ruling replaces the open "Option 1/3" escalation —
nothing here awaits a user answer mid-run.

---

---

## Done ledger

*(orchestrator appends: task · date · evidence path)*
