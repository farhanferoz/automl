# Cross-cutting plan-state audit — capacity programme (2026-07-20)

Scope: MASTER.md, width-depth.md, flexnn-moe.md, probreg-report.md, shared/metrics-accounting.md,
probreg.md (§1,3.5,3.6,4), research_plan.md, flexnn-core.md (structure only). Read-only; no repo
edits made. All citations opened this session.

---

## C1 — Programme map

**Verdict:** Seven strand files, one (flexnn-core.md) still holds a fully-specified ProbReg
sub-workstream that the programme's own index says was extracted elsewhere — its completion
claims are evidence-cited per-task but never rolled up into a Done ledger or into MASTER's table.

| # | File | Owns | Stated status (MASTER.md:14-22) | Ledger-backed? |
|---|---|---|---|---|
| 1 | width-cert.md | width architecture cert | DONE — G-WIDTH PASS 2026-07-16 | Yes — `docs/width_mse_2026-07-16/verdict_variable_width_mse.md` cited (MASTER.md:60), file verified to exist on disk. |
| 2 | probreg-report.md | report (a): ProbReg vs 4 baselines, toys | DONE — P0-P3 done, report delivered | Yes at the task level — Done ledger (probreg-report.md:130-136) cites `shared/bug_audit_head.md`, `shared/hetero_nll_diagnosis.md`, `report_a_results/`, `report_a_probreg_toys.pdf`, all with per-cell counts (150/150). **But see C8 #2 — the delivered content is built on a model definition since superseded.** |
| 3 | depth.md | S5 substrate + D8 selection | DONE 2026-07-17 — G-DEPTH PASS | Yes — `verdict_per_input_depth.md` §10-12, D8b toy JSONs cited (MASTER.md:196-205). |
| 4 | width-depth.md | joint 2-D dial | BLOCKED — J-1/J-2 dead, J-3 escalated | Yes, unusually well — the "not done" status itself is evidence-cited (`joint_readout_seed0.json`, `joint_block_seed0.json`, `diag_fullwidth_only.py`), verified in this audit (width-depth.md:104-121). |
| 5 | flexnn-moe.md | MoE build + reports (b)(c) | DONE — M0-M2 DONE 2026-07-16; M3-M5 superseded | Yes — Done ledger (flexnn-moe.md:186-189) cites `report_b_results/*.json`, `moe_regression.py` selftest numbers, and states a REFUTED finding (ELBO depth-selection claim) rather than just a pass. M3/M4/M5 correctly NOT claimed done — marked superseded, cross-referencing flexnn-core.md F6/F7. |
| 6 | flexnn-core.md | package refactor, FF-depth pilot, unified report, **+ProbReg tasks F9/F10/F11/F12/F13** | "yes" dispatchable (MASTER.md:21); per-task STATUS blocks inline | **Weak.** Formal "## Done ledger" section at flexnn-core.md:758-761 is EMPTY (`*(orchestrator appends: task · date · evidence path)*` with nothing under it) despite F9 claiming "**DONE + VERIFIED ON XPU HARDWARE**" (flexnn-core.md:395) and F13 claiming "**DONE 2026-07-20 (no further action)**" (flexnn-core.md:626) with real evidence inline (test-pass counts, before/after diffs). MASTER.md's own row for strand 6 gives no per-task rollup. This is a self-reported-glyph pattern one level removed from probreg.md's own warning about untrustworthy inline status (probreg.md:102-107) — the individual claims ARE evidence-backed, but nothing outside the file confirms them, and the one formal ledger meant to do that is blank. |
| 7 | probreg.md | whole ProbReg workstream | "yes — the live workstream" (MASTER.md:22) | Per-task, yes — P0/P0b carry explicit verify conditions with results shown (probreg.md:244-249, 291-293); D1 fix verified in both directions (probreg.md:128-134). No separate Done ledger section exists yet (strand is mid-execution, order is P0→P0b→P1→...→P6, most tasks not yet run). |

**The RESULTS.md ground-truth rule itself is scoped narrower than "programme-wide."** probreg.md
§2.5 (probreg.md:96-107) states the rule for "the first generation" of ProbReg planning — 13 files
across 5 directories that predate this strand — and names `capacity_ladder_results/RESULTS.md` as
authoritative for *that* generation only. Nothing in MASTER.md extends this rule to width/depth/
joint/MoE. Applying it uniformly to all seven strands (as this audit's brief invited) overstates
what the source document actually claims; I've instead applied MASTER's own general rule ("DONE =
the task's own `verify:` line EXECUTED, with its output shown" — MASTER.md:134-137) to every strand
above, which is the correct programme-wide standard.

---

## C2 — The MASTER decisions

All 17, one line each what they bind (MASTER.md:56-127):

1. Charter verdicts to date — #2 (`SharedTrunkPerWidthHeadNet`) is the certified width architecture.
2. MSE-only for capacity strands — **amended**: binds WIDTH only; depth is CE, joint follows its toy.
3. Reports are toys-only — **amended 2026-07-20**: ProbReg now explicitly exempted (real data + baselines in scope).
4. Hetero-NLL default: minimal fix if cheap, else document the limitation.
5. Depth/joint structured around #2's shared-repr + per-capacity-head pattern — **amended**: WIDTH only; depth's transfer prediction was refuted, shared readout won instead.
6. Depth strand starts with toy construction, not architecture.
7. MoE config: 8 experts/top-2 primary, top-1 ablation, load-balance aux loss, param/FLOP-matched.
8. Reports supersede the old "REPORT-2 + mathematical_guide fold-in" item; math is lifted, not re-derived.
9. Trajectory discipline: full held-out trajectories, convergence-verified, no conclusion from an endpoint.
10. Report authorship: `research-report` skill, full math rigor, no AI/tool provenance.
11. Commits are user-gated.
12. Worker tiering: haiku=mechanical, sonnet=default build, opus/main=discovery+verdicts+gates.
13. Selection = post-hoc distillation for width/depth/joint/transformer per-token depth; in-training selection may only be a labeled comparison arm.
14. Positive-control gate: known-good arm runs first, alone, must reproduce before further compute.
15. Protocol parity when reusing a substrate: diff the training loop, justify every difference in writing.
16. Optimization exonerated before architecture is blamed: LR/clip/warmup/init ladder before calling a result a generalization failure.
17. Convergence gate computed on the metric the bar reads.

**(a) Decisions whose text is programme-wide in name but whose stated scope, on inspection, covers
only one strand — or has been *stretched* to cover a strand it never named:**
- Decisions 2 and 5 are self-flagged as WIDTH-only via inline "(Amended ...)" parentheticals — not
  a hidden problem, MASTER.md documents its own narrowing.
- **Decision 13 is the sharp case, and it is NOT self-flagged.** Its text (MASTER.md:97-103) reads:
  *"Per-input capacity choice (width, depth, joint, transformer per-token depth) is learned by
  distilling a router from held-out per-capacity error tables... In-sample / in-training selection
  is never the primary (failed for width; FlexNN ELBO depth-select refuted, M0)."* It names four
  domains — width, depth, joint, transformer-token-depth — and ProbReg/k-selection is not one of
  them. Yet probreg.md's own non-goals section invokes it directly: *"No revival of in-training k
  selection as a primary (MASTER Decision 13)"* (probreg.md:566). Decision 13 is being used as
  authority for a fifth domain (k) that its own text never mentions. The conclusion it's used for is
  almost certainly still correct — ELBO in-training k-selection collapses (documented independently
  in flexnn-moe.md's M0 ledger entry, flexnn-moe.md:188) — but the citation is doing more work than
  the decision it cites actually says.

**(b) Decisions the ProbReg re-scope has superseded or contradicted:** Decision 3 is the one
directly touched, and it is handled correctly — the amendment is written in place with a date
stamp, not silently overwritten. I found no MASTER decision the re-scope contradicts *without*
MASTER's own text acknowledging it, aside from the Decision-13 scope-creep in (a) above, which is a
citation-authority gap rather than a live contradiction (the two decisions don't disagree; one is
just being cited past its stated domain).

**(c) Does "selection is distilled post-hoc" have an equivalent statement for the cheap-global
(arbiter, M1) and expensive-sweep (M3) modes, or only for the router (M2)?** Only for the router.
Decision 13's text is specifically about *per-input* policy learned by *distillation* — that is the
M2 mechanism verbatim. M1 (one k for the whole dataset, chosen by a held-out arbiter) and M3 (one k
for the whole dataset, chosen by sweeping a model per k) are not per-input at all, so Decision 13
doesn't logically reach them either way. Nothing in MASTER.md states a general principle like
"global capacity selection must also be post-hoc/held-out, never in-training" that would cover M1
and M3 by extension — that framing exists only inside probreg.md §1 (probreg.md:15-45), which is
consistent with Decision 13 in spirit but was never written back up into MASTER as a generalized
rule the other strands could reuse.

---

## C3 — Is the three-mode model set stated anywhere programme-wide?

**Verdict: No. It is ProbReg-only.**

```
grep -rn "GLOBAL_ARBITER\|PER_INPUT_ROUTER\|GLOBAL_SWEEP\|cheap global\|expensive global\|three ways of choosing" \
  docs/plans/capacity_programme/*.md docs/plans/capacity_programme/shared/*.md
```
returns zero hits outside probreg.md. The only programme-wide artifact that even gestures at the
three models is MASTER.md's naming key, and it explicitly refuses to restate the definition:
*"ProbReg models M1 / M2 / M3 — the three ways of choosing k. **Defined in ONE place: `probreg.md`
§1. Do not restate the definition here or anywhere else.**"* (MASTER.md:46-47). So the structural
idea — cheap-global vs per-input vs expensive-reference as three *complete, separately-costed*
models — lives exclusively in probreg.md and has not been proposed as a template for width, depth,
joint, or MoE. See C4 and the gaps section below for what that costs those strands.

---

## C4 — The joint width+depth work

**Verdict:** width-depth.md's J0 tried two toy constructions, both died at the substrate (not at
the selection question), and the strand is currently a formal escalation waiting on a user design
decision. No selection-mode work (arbiter/router/sweep) has even been reached yet — the toy itself
doesn't exist.

**What it proposes:** one network serving a per-input 2-D capacity dial (width AND depth), built by
crossing the certified width pattern (per-width heads) with the certified depth pattern (shared
readout over a weight-shared recurrent block) — "Default joint pattern: per-width heads × depth-shared
readout" (width-depth.md:14-15). Selection mechanism is declared up front to be distillation, generalizing
Decision 13: *"the per-input (width, depth) policy is DISTILLED post-hoc from held-out per-capacity
error tables"* (width-depth.md:29-31) — i.e. only the M2 analog is even discussed as a target.

**What ran:** `automl_package/examples/joint_capacity_toy.py`, two candidate constructions (J-1
readout-width, J-2 block-width), both multi-track parallel-A5 substrates. Both died on an S1
substrate-fit bar (0.90 held-out accuracy per cell), not on a selection-quality question: J-1 scored
0.58-0.79 across cells (width-depth.md:107-109), and even the single-track control (A=1) only hit
0.79 vs the depth toy's ≥0.90 on the *same* single-track task (width-depth.md:108-109). J-2 was shown
code-identical to J-1 at full width and scored *worse* (0.23-0.415) once masking was added
(width-depth.md:115-121). Root cause stated plainly: *"a multi-task-interference / input-multiplexing
failure, not an A5-composition or capacity-bit limit"* (width-depth.md:112-113).

**What's open — the decision required, quoted verbatim (width-depth.md:142-152):**

> "DECISION REQUIRED (batched for user — [[feedback_toy_design_needs_reviewed_spec]]: no toy
> improvised mid-run). Options:
> 1. Author a proper J-3 design spec (group-complexity width dial + solvability/depth confound
>    ledger + arithmetic probe), review, then build+pilot. Highest chance of a genuine joint result;
>    needs a design round.
> 2. Fix the multi-track substrate instead of abandoning J-1/J-2: wider state (128), per-track
>    sub-blocks (relaxes "one block"), or fewer tracks (K_MAX=2). Each is a design change with its
>    own charter cost (e.g. per-track blocks weaken the "one network" claim).
> 3. Re-scope G-JOINT to a narrower, honestly-stated claim (e.g. depth-only per-input dial with a
>    fixed width), or park the joint dial and proceed to the MoE reports (flexnn-moe.md M3-M5,
>    currently gated on G-JOINT)."

Recommendation on record is option 1; the strand is explicitly parked pending that user call
(MASTER.md:32 marks J0 "PARKED pending user Option 1/3 decision"; flexnn-core.md:11-15 carries the
same BLOCKED status forward and treats G-JOINT as "an honest open problem" in the unified report
rather than a dependency).

**Would the three-mode structure apply to a joint dial?** Partially, and only by accident of
convention, not by design. The pre-registered gate rule (MOD-4, width-depth.md:79) requires the
adopted toy to *"beat BOTH marginal routers"* **and** *"best-fixed-(w,T)"* — the latter is
functionally an M3 analog (a grid-sweep reference used as the honest ceiling), and "marginal
routers" gesture at M2. But neither is framed, scored, or costed as a *complete system including its
own selection machinery* the way probreg.md §1 demands (probreg.md:23-28) — they appear only as
comparator terms inside a single gate inequality, not as three separately-reportable models. There
is no M1 analog anywhere in width-depth.md: no cheap-global-arbiter concept for a joint (w,T) pair
has been proposed. If J-3 (or any successor) reaches execution, the strand would need its own
version of C3's "each mode is a complete system" doctrine written in — right now it would inherit
only the router half of the pattern by default.

---

## C5 — The mixture-of-experts strand

**Verdict:** M0-M2 done and ledger-backed with real evidence (including a genuine negative
result); M3-M5 correctly not claimed done, cleanly rescoped elsewhere. No contradiction found here.

- **M1** (MoE conventions frozen): 8 experts/top-2 = Mixtral (arXiv:2401.04088 §2.1), top-1 ablation
  = Switch (arXiv:2101.03961 §2), deterministic top-k softmax gating (a correction from the planning
  hypothesis of noisy top-k — flexnn-moe.md:60-62 explicitly documents the correction, citing Shazeer
  2017 as the noisy-top-k *origin* only), load-balance aux loss = Switch eq. 4 with α=1e-2. Done
  ledger: flexnn-moe.md:186.
- **M2** (`moe_regression.py` build): selftest evidence quoted directly in the ledger — "shapes/finite;
  top-k gradient isolation err=0.0; aux-loss ↓ under imbalance 0.080→0.020; `match_to_reference` ≤5%
  on FlexNN & NestedWidthNet" (flexnn-moe.md:187).
- **M0** (FlexNN post-Phase-9 re-validation): ledger entry states a *negative* finding as the
  headline, not a pass — "**Historical ELBO depth-selection claim REFUTED**: post-fix ELBO →
  complete depth-collapse to depth=1 all 5 seeds... ELBO vs NONE test-MSE indistinguishable. Report
  (b) cites the collapse, not the old claim." (flexnn-moe.md:188). This negative result is also what
  justifies F1's prior-fix in flexnn-core.md (flexnn-core.md:69-75), so the two strands agree with
  each other on this point — a genuine cross-strand consistency check that passed.
- **M3/M4/M5**: explicitly marked "superseded" in place (flexnn-moe.md:104-179, 189), pointing at
  flexnn-core.md's F6 (comparison battery, ungated from G-JOINT) and F7 (unified report). F6's own
  status is "**NOT RUN**" with an explicit correction of an earlier false-done claim: *"carried this
  as substantially done; it is not (new Rule: DONE = `verify:` executed)"* (flexnn-core.md:305-308) —
  so the supersession chain is honest about what's actually finished versus just re-pointed.

---

## C6 — Report planning across the programme

**Verdict:** Three separate report deliverables exist or are planned for ProbReg alone, spread
across three different strand files, two of them targeting overlapping content; there is no
standalone report plan for width or depth; and the master paper-strategy document that's supposed
to tie it all together predates almost the entire capacity programme and has not been updated to
match it.

**Report inventory:**
| Report | Owning strand | Status | Output |
|---|---|---|---|
| Report (a): ProbReg fixed-k vs "dynamic-k (ELBO+SoftGating)" vs 4 baselines, toys | probreg-report.md | DONE 2026-07-16 (probreg-report.md:137) | `docs/reports/probreg_toys/report_a_probreg_toys.pdf` |
| k-selection report (arbiter/distilled-router narrative) | flexnn-core.md Task F10 | delivered on disk (`docs/reports/probreg_kselection/probreg_kselection.md`, 880 lines, verified this session) | `docs/reports/probreg_kselection/` |
| ProbReg report, three-model (M1/M2/M3) framing + real data | probreg.md Task P6 | not yet run — gated on P4, P5 | extends `docs/reports/probreg_kselection/` (probreg.md:537) |
| Unified FlexNN report (width+depth+MoE, G-JOINT framed as open) | flexnn-core.md Task F7 | not yet run | `docs/reports/flexnn_unified/` |

**No standalone report plan exists for width or depth as their own deliverables** — width-cert.md
and depth.md each deliver a verdict document (`verdict_variable_width_mse.md`,
`verdict_per_input_depth.md`), not a `research-report`-skill output; their reportable content is
folded into F7's unified report instead. That's a coherent design (one report, not two thin ones),
but it means width/depth have zero report-shaped deliverable of their own right now — everything
routes through F7, which hasn't run.

**Mapping onto research_plan.md — stale, confirmed by direct inspection.** research_plan.md is
dated 2026-04-15 and the only change since is a one-line factual correction (5 vs 6 features on UCI
Airfoil, 2026-07-20 per `git diff docs/research_plan.md`) plus a self-contained §5.1 addendum dated
2026-07-18 (research_plan.md:578-618) about a *transformer* roadmap stage. Its §7 Paper Strategy
(research_plan.md:732-789) and §8 Execution Roadmap (research_plan.md:792-818) — the sections that
actually map work onto papers — are untouched. Direct check:
```
grep -n "arbiter\|G-WIDTH\|G-DEPTH\|G-JOINT\|FlexNN\b" docs/research_plan.md
```
returns zero hits for "arbiter", "G-WIDTH", "G-DEPTH", "G-JOINT" anywhere in the file; "FlexNN" (as
opposed to "FlexibleNN") appears only in the §5.1 transformer addendum (research_plan.md:590,592,597)
and later backlog items (research_plan.md:844-912) that predate the umbrella-naming decision and use
it as a loose synonym for FlexibleNN, not the umbrella sense MASTER.md now defines (MASTER.md:38-39).
Concretely: research_plan.md's §7 "Paper A — ProbReg" contributions list (research_plan.md:744-748)
still describes the single dynamic-k model as "(b) ELBO-based dynamic-k (SoftGating) for automatic
bin complexity allocation" — the same superseded in-training framing flagged in C8 below — with no
mention of M1/M2/M3, the arbiter, or the distilled router anywhere in the paper-strategy section.
The mapping from capacity-programme work onto the two-paper structure predates the re-scope by three
months of programme evolution and has not been reconciled.

---

## C7 — Metrics and cost accounting

**Verdict:** shared/metrics-accounting.md's cost-accounting module (S2, `capacity_accounting.py`)
covers only the network's own params/FLOPs at a given capacity setting. It has no accounting hook
for the cost of the *selection step itself* — no router-fit cost, no arbiter-evaluation cost, no
per-k-sweep training cost. This is exactly the gap probreg.md's re-scope was created to close, and
it has been closed *inside* probreg.md's own tasks (PA/P4/P6), but not inside the shared module
every other strand is supposed to build from.

S2's own spec, quoted in full (metrics-accounting.md:54-59):
> "Spec: `param_count(net, path_filter=None)` (e.g. exclude `logvar` heads — the width nets' MSE-path
> convention) and `executed_flops(net, config)` — analytic multiply-add counts for the programme's
> architectures: width nets at routed width k (3k+1 pattern — derive per class, do not copy), depth
> nets at routed depth d, MoE at top-k, FlexNN at selected depth. Selftest: hand-computed known-answer
> checks for one small config per family. **This module is the ONLY source of params/FLOPs numbers in
> reports (b)/(c) and the strand-4 deploy bars.**"

I verified `automl_package/examples/capacity_accounting.py` directly (grep for class/def names): it
implements `param_count`, `executed_flops` for `NestedWidthNet`, `SharedTrunkPerWidthHeadNet`,
`IndependentWidthNet`, `SharedReadoutPerWidthAffineNet`, FlexNN, plus `DepthNetShapeDescriptor` and
`MoEShapeDescriptor`. The MoE descriptor DOES account for router FLOPs
(`capacity_accounting.py:358`, `router_macs = _linear_macs(net.d_in, net.n_experts)` — the router
always scores every expert before top-k picks). **But there is no descriptor or function anywhere in
the file for `DistilledCapacityRouter`, the ProbReg arbiter, or a per-k-sweep's total training
cost** — grep for `probabilistic_regression|distilled_router|arbiter` against the file returns
nothing. ProbReg's own re-scope had to build this accounting itself, outside the shared module:
probreg.md's P4 task requires the driver to record which constants (selection-set fraction, router
shape) a run used and fail loudly if missing (probreg.md:513-518), and P6's honesty clause requires
"report M3's full cost next to its accuracy (the efficiency claim is a ratio and this is its
denominator)" (probreg.md:553-555) — but this lives entirely inside probreg.md, not in
shared/metrics-accounting.md where width, depth, and MoE would also draw from it. **The MoE router's
FLOP cost is accounted; the width/depth/joint strands' distilled-router *fit* cost is not accounted
anywhere I can find** (S2 counts inference-time FLOPs of the *network*, not the training cost of the
selector that chose its configuration).

---

## C8 — Contradictions

This is the adversarial section; four findings, ordered most to least serious.

### 1. flexnn-core.md still fully owns and claims progress on ProbReg tasks that probreg.md says it alone owns

MASTER.md states, twice, that ProbReg's scatter problem is fixed by consolidation: *"ProbReg is
extracted (strand 7)"* (MASTER.md:27) and *"one execution-level strand file per workstream is the
rule... A definition may exist in exactly one file"* (MASTER.md:187-189). probreg.md opens with:
*"Nothing about ProbReg is decided anywhere else; if another document disagrees with this one, this
one wins and the other is a bug to fix"* (probreg.md:5-6).

But flexnn-core.md (read in full structure this session) still carries five fully-specified ProbReg
tasks with their own status claims, none marked superseded or redirected to probreg.md:
- **F9** "ProbReg dynamic-k port" (flexnn-core.md:362-479) — claims "F9-fix: DONE + VERIFIED ON XPU
  HARDWARE 2026-07-20" (flexnn-core.md:395) and documents a second defect, "F9-fix-b", "VERIFIED
  2026-07-20" (flexnn-core.md:441-465) — this is the *exact same* symlog-space unit-mismatch bug
  that probreg.md independently records as "D1 — FIXED + VERIFIED 2026-07-20" at
  `probabilistic_regression.py:813` (probreg.md:128-131). Same fix, same file, same line number,
  documented twice under two different task IDs (F9-fix-b vs D1) in two strand files that both claim
  sole authority.
- **F10** "ProbReg k-selection report" (flexnn-core.md:481-509) — creates
  `docs/reports/probreg_kselection/`, the exact folder probreg.md's P6 (probreg.md:535-559) is tasked
  to *extend*. F10 has no "superseded" marker; P6 doesn't cite F10 as its predecessor either — the
  two tasks are connected only by sharing a path, not by any written pointer.
- **F12** "ProbReg benchmark — real data" (flexnn-core.md:569-618) — creates
  `docs/probreg_benchmark/benchmark_spec.md`. probreg.md's own §1.1 documents that this exact file
  was one of the three live contradictory definitions (*"`docs/probreg_benchmark/benchmark_spec.md:66`
  defines Model 1 with `n_classes_selection_method=NONE`"*, probreg.md:58-62) and probreg.md's **Task
  P0** (probreg.md:233-271) has a write-set that includes rewriting that same file — so P0 and F12
  are two different strand files both claiming single-writer authority over
  `docs/probreg_benchmark/benchmark_spec.md`. MASTER's own Rules section requires "**Single writer:**
  workers return findings; the ORCHESTRATOR writes strand files and ledgers" (MASTER.md:144) — that
  discipline has not been applied to *which strand file gets to own this artifact*.
- **F11** (roadmap groundwork dossiers) and **F13** (refactor debt) are lower-stakes but same
  pattern: live, undated-as-superseded task blocks inside flexnn-core.md for content that is either
  ProbReg-adjacent (F13's `distilled_router.py` parity guard) or unrelated to the four workstreams
  flexnn-core.md's own header says it holds ("package refactor... FF-depth study... MoE comparison...
  unified report" — flexnn-core.md:1, no fifth item listed).

MASTER.md's SPLIT PENDING note (MASTER.md:24-29) only commits to extracting F5 into `depth-ff.md` —
it does not mention F9/F10/F11/F12/F13 at all, so this isn't a known-and-tracked gap, it's an
unacknowledged one.

### 2. Report (a) — DONE, delivered — is built on the model definition probreg.md's re-scope retired

probreg-report.md's Goal line, unchanged since authoring: *"report (a): Probabilistic Regression
with **shared-k (fixed `n_classes`)** and **variable-k (ELBO + SoftGating dynamic `n_classes`)** as
two distinct models"* (probreg-report.md:3-4) — i.e., the *in-training* ELBO+SoftGating selection
mechanism as the "variable-k" arm. This is delivered and marked DONE 2026-07-16
(probreg-report.md:136-137), one day *before* MASTER Decision 13 (dated 2026-07-17, MASTER.md:97)
demoted in-training selection to a labeled comparison arm only, and four days before probreg.md §1.1
(2026-07-20) formally catalogued this exact definition as one of three mutually contradictory
statements live at once: *"`MASTER.md`'s naming key defined variable-k as 'dynamic-k (ELBO +
SoftGating)' — in-training selection, which `MASTER.md` Decision 13 itself demotes to a labelled
comparison arm. Self-contradictory."* (probreg.md:51-53).

**MASTER.md's Corrections entry for this event catalogues three contradictory sites — its own naming
key, `benchmark_spec.md`, and "the user's actual design" (MASTER.md:180-189) — and does not list
report (a) as a fourth.** But report (a) states the identical superseded definition, as a *delivered,
"DONE"* research artifact with a built PDF, and nothing in MASTER.md, probreg.md, or
probreg-report.md itself flags it as needing revision. probreg.md's own "Report status" section
(probreg.md:90-92) discusses only `docs/reports/probreg_kselection/probreg_kselection.md` as
predating §1 — it doesn't mention `docs/reports/probreg_toys/report_a_probreg_toys.pdf` at all. Report
(a) is currently an orphaned, undated liability: nobody's task list currently includes revising it.

### 3. Citations that looked fragile held up (ruled out, stated for completeness)

I checked one candidate contradiction and it turned out NOT to be one: MASTER.md Decision 1 cites
`docs/width_mse_2026-07-16/verdict_variable_width_mse.md` (no `plans/` segment) while the History
section cites `docs/plans/width_mse_2026-07-16/EXECUTION_PLAN.md` (with `plans/`) for the same
programme. Both paths exist on disk (verified via `test -f`); they are two different, correctly
distinct files, not a citation error. The plan-gates suite also passes clean: `pytest
docs/plans/capacity_programme/test_plan_gates.py -q` → **6 passed** (run this session). Flagging this
only so it's not re-suspected by a later auditor re-reading the same two paths.

---

## PROGRAMME-LEVEL GAPS AGAINST THE PROBREG BAR (most serious first)

1. **The shared cost-accounting module has the same blind spot ProbReg's re-scope was created to
   fix, and nobody has gone back to check the other strands against it.** `shared/metrics-accounting.md`
   S2 (`capacity_accounting.py`) prices a network at a given capacity setting; it has no function for
   the cost of *choosing* that setting (router-fit cost, arbiter-evaluation cost, sweep-training
   cost) for ANY strand, not just ProbReg. It happens to include MoE's router inference FLOPs (a
   different thing — routing cost at serve time, not selection-fit cost), which makes the gap easy to
   miss on a skim. G-WIDTH PASS and G-DEPTH PASS were both certified before this accounting gap was
   even named as a failure mode (2026-07-20); nobody has asked whether the width or depth "certified
   winner" would still win once its distilled router's own fitting cost is charged against it. (C7)

2. **The three-mode "score the selection machinery as part of the model" doctrine exists in exactly
   one strand file.** Width, depth, and joint have a router (M2-equivalent) but no cheap-global
   (M1) analog and no honest expensive-reference (M3) analog used to validate the router's *choice*,
   only its *accuracy at a fixed setting*. Concretely: nobody has checked whether width's or depth's
   distilled router picks the same capacity level a full per-value sweep would have picked — the
   exact "(b) same choice" gap probreg.md itself flags as untested for its own strand (probreg.md:83-85)
   is untested, and unplanned, for width and depth too. (C3, C4)

3. **A delivered, DONE report (report (a)) carries a model definition the programme has since
   retired, and no open task currently owns fixing it.** Unlike the k-selection report
   (`docs/reports/probreg_kselection/`), which probreg.md's P6 explicitly plans to extend and correct,
   report (a) at `docs/reports/probreg_toys/` has no successor task anywhere in the seven strand
   files. (C8 #2)

4. **Organizational duplication that MASTER.md's own "ProbReg extracted" claim says shouldn't exist
   still exists in flexnn-core.md** — five task blocks (F9-F13), one of them (F12) claiming
   single-writer authority over a file probreg.md's P0 also claims to own. This is the identical
   failure class ("no single document held them side by side," MASTER.md:185-186) that motivated
   creating probreg.md in the first place, now reproduced at the strand-file level instead of the
   plan-directory level it was fixed at. (C8 #1)

5. **The paper-strategy document the whole programme is nominally building evidence for has not
   been updated to reflect three months of restructuring.** research_plan.md's §7/§8 still describe
   ProbReg's contribution as "(b) ELBO-based dynamic-k (SoftGating) for automatic bin complexity
   allocation" with zero mention of the arbiter, the distilled router, M1/M2/M3, or the
   G-WIDTH/G-DEPTH/G-JOINT gate structure that now organizes all of the underlying work. Every
   capacity-programme result currently has no stated path onto the paper it's supposedly evidence
   for. (C6)
