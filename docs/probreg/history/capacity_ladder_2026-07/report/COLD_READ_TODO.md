# REPORT-2 cold-read gate — status log (2026-07-10)

Report: `capacity_ladder_report.pdf` (18 pp), guard-clean, 6 figures from real artifacts,
4 verified citations, 6 tables. Build: `bash ~/.claude-automl/skills/research-report/scripts/build_report.sh capacity_ladder_report.md`.

## Gate rounds

**Round 1 (17 findings) — ALL APPLIED.** Recovery-error formula + one term ("noise-recovery
error") throughout; reference-instrument defined; Table 1 caption count; Table 5 real N=1000
numbers (0.054/0.062/0.060/0.063/0.079, verified vs `V2/V2_findings.md`); ladder max stated
(D=8, others=6, verified vs K4 KMAX); {1,2,4} reworded "erratic"; Toy H pre-reg bar explicit;
V1/v1 disambig; NLL sign gloss; likelihood-based arms; conformal gloss; X design matrix; Eq 8
simplified to average responsibility (EM fixed point); Eq 5 cumulative-vs-successive; §1.1 prior
reframed theoretical. Plus reconciled a latent count bug: 3 structured toys (C,D,E) + 2 twins
(C_broad,E_broad) → added Toy C row to Table 1, fixed "each structured toy has a twin".

**Round 2 (1 blocker + minors) — ALL 4 round-1 blockers CONFIRMED RESOLVED; round-2 fixes APPLIED.**
- BLOCKER r* off-by-one + abstain-code clash → rewrote §2.4 rule (accept significant increments,
  r* = largest accepted = one below first failing; r*=0 = abstain flag, never returns 1) +
  Table 2 caption reconciles per-input-knee count (1) vs global abstain (0).
- Added §3 evidence table (new Table 3): Toy C region means (−0.025/−0.016/+0.004 → +0.061/+0.037/
  +0.048), Toy E (recovers seed 0 only), broad-twin controls (≤0.016); global knee r*=0 on ALL incl.
  structured C/E (knee-unfaithful at global level too). Renumbered router→T4, depth→T5, fix→T6.
- §5 noise-ladder-is-ordered-family (not prefix-nested); Fig 5 "early over-reports not matches truth";
  half-nat decoupled from Appendix-A variance-bias + "nat" glossed at first use; 83-pts/bin arithmetic;
  EM monotone-ascent note; Table 1 "dial test"→"signal-to-noise dial"; caption "their twin"→built for C,E.
- Figure labels: fig_ws1 legend "nested surrogate"→"nested ladder"; fig_ws3_coarse rungs
  scalar/tercile/linear/MLP → constant/three-step/smooth/flexible (match prose, de-jargon).

**Round 3 (1 blocker + minors) — ALL round-2 fixes CONFIRMED; round-3 fixes APPLIED.**
- BLOCKER Table 4 (router) header "Beats/ties global count" contradicted rows 2–3 (which compare
  vs the *smooth* router) → restructured to Target | Outcome | Compared-against columns + caption.
- Recurring minors closed: half-nat bridged to chi-square deviance (≈0.5 nat); reference instrument
  described concretely (variational per-region mixture); Summary per-input reworded (clean for count,
  intermittent for depth, aggregate for noise); coherence-cost sign note at §2.4; per-input-knee "1"
  vs global-abstain "0" parenthetical in §2.4; Δ(c)/Δ^{C−1} clash → simplex renamed 𝒮 (Appendix B);
  G-flat "uniform, base-level required depth"; depths-3-and-4 vs r*=3 = idiom-1-vs-2 clarifier.
- Gate 3 CONFIRMED: r* rule fully determinable + self-consistent; all cross-refs resolve; no other
  hard contradiction; Appendices A/B complete; case counts clean; v1/V1 disambig holds.

**Round 4 (1 blocker + minors) — round-3 fixes CONFIRMED; round-4 fixes APPLIED.**
- BLOCKER: Summary/§6 claimed "aggregate reader discriminates in all 3 settings", but the new
  Table 3 shows the aggregate GLOBAL KNEE abstains (r*=0) identically on structured C/E AND controls
  → the knee under-reads component count even globally; the discriminator there is the LOCAL
  ADVANTAGE (large on staircase ≤0.20, ≤0.016 on controls). This is a real scientific nuance my own
  §3 table exposed. Reframed Summary finding 1 + finding 3 + §6 "What holds": knee discriminates for
  depth/noise; local advantage discriminates for count (knee unfaithful even at aggregate level).
- Minors: r* abstain-corner "i.e." precision (or 0 if capacity-2 fails); v-index→r* mapping in §5
  (v0=base r*=0, v1=r*=2); "never tracks"→"does not reliably track" (seed 0 {1,2,4} does rise);
  Table 6 "In-sample joint fit"→"In-sample residual variance (mean fit first)" + "(mean and variance
  together)" to distinguish arms; neighbourhood size (7.5% box-car) in §2.5; Summary/§6 "one nested
  model" softened for the noise ordered-family.

**Round 5 (1 blocker + minors) — component-count reframe CONFIRMED clean across Summary/§3/Table3/§6;
round-5 fixes APPLIED.**
- BLOCKER: my round-4 finding-1 reframe over-corrected — it said the aggregate knee "selects the right
  capacity" for depth AND noise, but for NOISE the knee is only COARSE (detects heteroscedasticity +
  abstains on the constant control, but stops one rung short of the correct smooth class). §5/§6 + my own
  finding 3 already said this → contradiction. Fixed finding 1 + finding 3 to the THREE-WAY gradation:
  depth = knee selects right capacity; noise = knee detects but coarse (stops one rung short); count =
  knee under-reads even presence → local advantage discriminates. Verified verbatim-aligned with §5/§6.
- Minors: reference instrument reframed as "independently computed" (drop "earlier work" → not a dangling
  citation; also fixed Fig 2 legend "(prior study)"→"(variational mixture)"); "raw per-example labels"
  router defined + added to prose (worse 8/9).
- Gate 5 CONFIRMED clean: r* rule exactly determinable; all numeric cross-checks pass; case counts;
  every fig/table referenced; refs resolve; Appendices A/B derive what they assert.

**Round 6 (FINAL) — NO BLOCKERS. REPORT-2 DONE.**
- Knee-consistency verdict: CONSISTENT across Summary(f1,f3)/§3/§4/§5/§6 + all table captions (the
  three-way gradation depth=selects / noise=coarse / count=under-reads-presence holds everywhere; no
  sentence claims the knee "selects right capacity" where another says it under-reports). r* rule exactly
  determinable + consistent with every worked value. All numeric cross-checks, math, sign conventions,
  refs CLEAN.
- 7 minors; applied the cheap correctness-adjacent ones: depth "correct depth" overclaim softened (toy G's
  true depth not stated → "selects a definite depth 3 and 2, the latter matching toy H's known
  requirement"); finding-3 scoped ("under-reports the finer structure … for two of the three at the
  aggregate level; depth is the exception"); §5 "exactly as component count"→"same direction, less severely
  (noise detects, misses only class; count missed structure entirely)"; −0.838 = "local-advantage profiles".
  Reported typo #1 = FALSE POSITIVE (text already correct; cold reader misread the PDF).
- LEFT (acceptable for a research note): #5 no SE bars on Fig 3 aggregate depth curve (r* certified by R3);
  #6 Table 3 control-row cell "max anywhere" in the 2-comp column (caption resolves). Would need data-pull +
  figure regen for marginal gain.

## STATUS: REPORT-2 COMPLETE (6 gate rounds, final clean). PDF 18pp, guard-clean, 6 tables, 6 figures.
## NEXT: NOTE-MOE (stretch — user decides go/skip) + scoped-commit proposal (user approves set). Gate loop CLOSED.

## Deliberately LEFT (cold reader marked acceptable / trivial): "responsibility" 2 senses (both
locally defined); H matrix vs toy H; r* vs k* not contrasted in one place; Table 3 "—" (caption
explains); "pre-registered" framing; several §5/§6 multi-case numbers stated in prose not tabled.

## REVISION 2026-07-10 (post-completion review): findings folded in + gate round 7

**Revision content (review of record: EXECUTION_PLAN.md §8):** F2 nesting-cost correction (large at
depth 1, −0.72 nat, ~50% cells, top rung too); NEW Table 5 nested-vs-dedicated depth knees (G r*=3→2
seed-coherent; H first increment +0.64→+0.049, 13× nesting-inflated) + the Δnested = Δdedicated + Δκ
decomposition displayed in §3; D global-knee gate closure (r*={2,0,0}, pre-registered check reconciled);
G per-input +2.67·SE downgraded (3/9 splits; noise floor); hierarchical-stacking wording fixed (independent
per-bin, no partial pooling; pooling = queued refinement); E-lane June-instrument contrast (nesting-specific
hypothesis); V3 intrinsic claim upgraded to three checks (analytic 0.0029-nat ceiling; Gneiting-Raftery
locality; Camporeale-Carè exact NLPD ties; δ² second-order display); v2-vindication softened (N-dependent
winner, v3 dominates at N=4000); §2.4 guardrails: paired bootstrap + Sivula small-gap caveat. 3 new bib
entries (gneiting2007strictly, camporeale2021accrue, sivula2020uncertainty). PDF 21pp.

**Round 7 (fresh cold reader on the 21pp revised PDF): 2 real blockers + fixes, ALL APPLIED.**
- BLOCKER 1: Summary said dedicated depth read "coherent across seeds" for BOTH problems — Table 5
  contradicts for H ({2,3,abstain}) → "coherently on the first, marginally on the second".
- BLOCKER 2: Summary "over-reports by one rung" over-generalized to H (H nested=dedicated=2) →
  "one rung too deep on one problem, first increment inflated thirteen-fold on the other".
- Math-asserted fixes: Δnested=Δdedicated+Δκ now DISPLAYED (§3); δ²-second-order loss now DERIVED
  (§5 display, log(1+δ)+1/(2(1+δ)²)−1/2 = δ²+O(δ³)); half-nat-per-parameter accounting spelled out
  (§1.1); App A HX=X step + p/H symbol disambiguation; App B EM ascent justified.
- Under-shown evidence closed with inline numbers: E reference-instrument arc (−0.018/+0.149/−0.026);
  D global knee per-seed (2,0,0); 9-partition split values; sextile cells (G-s2 +0.010±0.010, H-s2
  +0.040±0.015); analytic integral per-region breakdown (0.0075/0.0012/0.0001); depth-cost decay
  (≤0.08 by depth 2); −0.838 three-point fragility flag; Table-2 pointer fixed (method, not table);
  Table-4 pairs defined; Fig-6 0.003 tied to the analytic ceiling.
- Reader confirmed CLEAN: App A/B derivations, stacking motivation, k*/r*/v-rung disambiguations,
  instrument naming (nested vs dedicated) everywhere else, all cross-refs, all numeric cross-checks.
Rebuilt: 21pp, guard-clean, 0 unresolved refs.

## R-INT REVISION 2026-07-10/11 — per-input selector program (S/T/H/P) folded in + gate round 8

**Revision content:** new **§6 "From a measurement to a deployable selector"** (renumbering the old
boundary section to §7), with five subsections: §6.1 the selector recipe (five target constructions;
prior-informed soft responsibilities win; prior load-bearing +0.0067±0.0021, smoothing negligible
+0.0001; a direct held-out objective ties 0.6811 vs 0.6807 but does not beat → recipe of record =
soft) → Table 8; §6.2 the recipe validated end-to-end on the classifier-over-classes model (two-phase
post-hoc gate beats jointly-trained on staircase 3/3 by 0.078–0.100, matches standalone router 9/9,
never worse than oracle single count on control, heads on half the data) → Table 9; §6.3 input-dimension
degradation (advantage +0.073 at dim2 3/3, gone by dim5; capture-rate rise is the all-k=1 collapse, not
improvement) → Table 10; §6.4 the two power curves — moving-mode count recovers with data (2/3 at
N=1000 → 3/3 at N=4000, control flat every N) vs per-input depth null at every N up to 8000 (signal
flat, control crosses alongside, t=0.11) → Tables 11, 12; §6.5 representable-but-not-learnable (a
provable tent⁵ depth requirement is GD-unlearnable, ~1.1–1.2 nat below achievable at every depth/restart
→ no learnable positive control for the depth lane; terminal path left OPEN as future work) → Table 13,
cites Telgarsky 2016 (verified this session). Reconciled §3 moving-mode tail (nesting-specific suspect
REFUTED, resolution = power, → §6.4), §4 per-input tail (queued refinements ran, null holds → §6.4/6.5),
§7 boundary (moving-mode no longer the genuine negative; boundary = dimension + depth-learnability),
"What holds" (+validated selector), deployment recs (+recipe), and the Summary. 6 new tables (8–13); no
new figures (power curves surfaced as tables by choice). New bib entry telgarsky2016benefits.

**Round 8 (fresh cold reader on the 28pp revised PDF): 2 real blockers + minors, ALL APPLIED; a
targeted fresh re-read of the two fixed passages certified them clean.**
- BLOCKER 1 (§5, eq for expected log-score lost): the LHS `log(1+δ)+1/(2(1+δ)²)−½` was asserted (round
  7 had derived only the δ² Taylor step, not the LHS provenance) → added the Gaussian-NLL derivation
  inline (E[−log N] = ½log2π + log σ̃ + σ²/2σ̃², minus the honest-σ value, sub σ̃=σ(1+δ)); symbols μ,y
  glossed. Re-read reproduced the algebra + Taylor (f, f′ vanish at 0, f″=2) — correct.
- BLOCKER 2 (§6.4 Table 12): "detection floor" undefined + a confusing collision (structured floor 0.0014
  = control signal 0.0014 at N=8000, in one under-labeled column) → defined detection floor (2·SE bar,
  split-overlap corrected) at first use; restructured Table 12 into separate Structured-signal/floor and
  Control-signal/floor columns with per-seed crossing counts. Re-read confirmed collision resolved,
  argument followable.
- Minors applied: "beats"→"matches or beats a single global count" (Table 4 = beats-or-ties) in Summary,
  §6 intro, §7; §6.5 "more than 0.7 nats"→"1.14–1.17 nats" (match Table 13); dropped undefined/overshooting
  "effective count 1.9→3.2" clause (§6.4). Left (pre-existing, MINOR, prior rounds accepted): dense Summary,
  toy-D global-knee r*=2,0,0 + −0.838 stated in prose (single numbers, not grids), rung/v-rung/p-vs-p_c
  conventions (already flagged in-text), companion-note not in refs (intended).
- Reader confirmed CLEAN otherwise: ~20 numeric cross-checks passed, moving-mode "negative→recovered" and
  per-input-depth null both explicitly reconciled across §3/§4/§6/§7, all cross-refs resolve, all 8
  citations resolve (incl. Telgarsky). Rebuilt: 28pp, guard-clean, 0 unresolved refs.
