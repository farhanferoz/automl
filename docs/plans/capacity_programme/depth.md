# Strand: depth (→ G-DEPTH) — COMPLETE 2026-07-17. G-DEPTH = D5 substrate ∧ D8 selection = PASS. → width-depth.md J0

**Goal:** answer the depth edition of the charter — can one network serve per-input variable
DEPTH? **Evidence says YES (2026-07-17): the D1b graded battery passed every pre-registered bar
on all 3 seeds, with controls** (RESULT lines in D1). Task D5 renders the formal G-DEPTH
verdict; D6/D7 close two measurement-apparatus gaps found on the way. The retired Phase-II
battery spec (old D2/D3/D4, obsoleted by the results — see RESCOPE) is preserved verbatim in
`archive/depth_d2_d3_retired.md`.

**RESCOPE [2026-07-17, user-ratified; amended same day after user correction].** The old
Phase II spec (old D2/D3, retired as written) was 1-D/MSE-specific and its transfer prediction
is REFUTED 3/3 (below) — but its PURPOSE returns as Task D8. **The charter question — per the
user's correction — is per-input depth variation WITHOUT being told what the input needs,
exactly as certified for width** (router trained from data, no oracle). D1b answers only the
SUBSTRATE half (one net serves every depth; width cannot substitute). The S5 graded toy CANNOT
carry the selection half: per-input depth is syntactically visible (word length), and the
error-vs-depth curve is a CLIFF (below ℓ: chance; at ℓ: perfect), so a selector would learn the
identity map and certify nothing. D8 therefore constructs a selection-capable toy (hidden,
graded depth-hunger) and runs the width selection protocol on it. **G-DEPTH = D5 (substrate
verdict) ∧ D8 (selection verdict).** Transformer-specific halting (per-token, sequential) stays
with J0/M3+, which consume D8's certified mechanism.

**The pre-registered transfer prediction — REFUTED 3/3.** Predicted (written before any battery
ran): the shared-readout arm fails the fit bar while per-length heads pass, mirroring the width
failure (FlexNN's shared `output_layer`,
`automl_package/models/flexnn/strategies/layer.py:90`). Outcome: the
opposite. `shared_readout` is the STRONGEST arm on every seed (S5 val_acc 1.000 at ℓ=4/6/8 and
≥0.990 at ℓ=10, all 3 seeds) while `per_length_head` merely passes. Mechanism reading: readout
interference is WIDTH-SPECIFIC — a weight-shared recurrent block presents every depth with the
SAME state space, so one readout serves all depths; width prefixes hand each capacity a
DIFFERENT representation fighting over one readout. This asymmetry replaces "one mechanism
governs both axes" as report (b)'s spine and sets the G-JOINT design prior (per-width heads ×
depth-shared readout — carried into `width-depth.md`). Stability caveat travelling with the
prior: 1 of 6 shared_readout runs (Z120 seed 1) hit an optimization blow-up (val CE 1.34 → 5.85
at epoch 3500, never recovered); flagged, not rerun (D5 ruling R4), and motivates D6.

> **PIVOT [2026-07-16] — the depth toy is a GROUP WORD-PROBLEM, not a 1-D regression toy**
> (construction + grounding, kept for the record). All four 1-D smooth-regression candidates
> failed, and the failure is a THEOREM: depth-hunger ⊥ GD-learnability for smooth 1-D targets
> (Malach & Shalev-Shwartz 2019 arXiv:1903.03488; Malach et al. 2021 arXiv:2102.00434 —
> GD-learnability requires weak shallow-approximability). Full write-up:
> `docs/depth_capacity/depth_toy_negative_note.md`. The escape is COMBINATORIAL: compose a
> length-n word of **S5** (non-solvable) generators → 120-way classification of the product;
> difficulty knob = n = #sequential steps = per-input depth. **Barrington 1989**: width provably
> cannot substitute for depth here. **Z120** (abelian, same order 120) is the
> width-substitutable control (Liu et al. 2022 arXiv:2210.10749). Toy modules:
> `automl_package/examples/depth_composition_toy.py` (single-length pilots, make-or-break) and
> `automl_package/examples/depth_graded_toy.py` (D1b graded battery). LR 3e-3 canonical
> (the n=10 "wall" was an LR artifact — Decisions 2026-07-16).

**Standing clauses & environment:** per `MASTER.md` Rules. This strand is CE classification on
held-out word accuracy (MASTER Decision 2 amended 2026-07-17 — MSE binds the width strand only).
**Ledger:** `automl_package/examples/capacity_ladder_results/D_TOY_PROBES/`.

**AUTONOMY CONTRACT (binding on every task below).** Every decision an executor needs is made in
this file; no task stops to ask the user anything, with exactly TWO sanctioned stops: (1) the
D8a toy-design review gate (user-mandated — the design doc is DELIVERED for review, D8b waits
for GO); (2) the D8b all-candidates-die escalation. Everything else: judgment calls are
pre-ruled (D5 R1–R5, D7 branch rule); a genuinely unforeseen reversible call → take the
conservative default, log ONE line in `RESUME.md` `### Decisions`, continue. User-only items
(commit approval — MASTER rule 11: stage NOTHING) are batched for end of run. No irreversible
or outward-facing action exists in this strand.

---

### Task D0: re-verify the standing citations — **DONE 2026-07-16** (MASTER Corrections entry)

### Task D1: construct the depth toy — **DONE 2026-07-17** (all 6 steps)

**History (compressed):** three 1-D candidates + the user's `fixedwidth` 4th all failed →
theorem closed the 1-D lane (PIVOT above). S5 construction built; make-or-break single-length
pilot PASSED at seed 0 (recurrent learns the algorithm, train≈val; wide-shallow memorizes;
gap widens with n). Standalone Z120 n=8/10 pilots at LR 1e-2 were killed as redundant
(Decisions 2026-07-16) — the graded battery provides the Z120 control at LR 3e-3.

**Bars (pre-registered before the battery ran; length ladder {4,6,8,10}):**
G1 deep arm fits (held-out acc ≥ 0.90 across the whole ladder) · G2 width stalls (wide-shallow
≤ 0.60 at the long end) · G3 the (deep − wide) gap increases monotonically with length on ≥2
seeds · G4 Z120 control shows NO divergence (all arms ≥ 0.90 everywhere, gap ≈ 0).

- [x] Step 1: `depth_composition_toy.py` (S5 + Z120, 3 net families, CE + convergence gate).
- [x] Step 2: make-or-break pilot, S5, n=6/8/10 seed 0 — PASS.
- [x] Step 3: Z120 control — folded into the graded battery (standalone pilots killed).
- [x] Step 4: n=10 trainability spike — LR 3e-3 canonical (0.569 → 0.991).
- [x] Step 5 (D1b): `depth_graded_toy.py` graded battery, 3 seeds × {S5, Z120} — landed
      2026-07-17.
- [x] Step 6: keeper JSONs promoted to `D_TOY_PROBES/`.

RESULT: D1b battery — ledger `automl_package/examples/capacity_ladder_results/D_TOY_PROBES/depth_graded_pilot_s5_seed0.json`
+ `depth_graded_pilot_s5_seed{1,2}.json`, `depth_graded_pilot_z120_seed{0,1,2}.json` (same dir).
With the ℓ=4 rung excluded (degenerate by design arithmetic — 128 train words vs 120 classes;
D5 ruling R1): ALL bars pass on ALL 3 seeds. G1 per_length_head ≥ 0.958/0.963/0.980 at ℓ=10
(shared_readout ≥ 0.990 everywhere); G2 wide-shallow at ℓ=10 = 0.386/0.382/0.408 while its
TRAIN acc ≥ 0.93 (memorizes, cannot generalize); G3 gap at ℓ=6/8/10 ≈ +0.07/+0.26/+0.57 on
every seed (spread ≤ 0.01); G4 all Z120 arms ≥ 0.916 at ℓ∈{6,8,10} on all seeds.

RESULT: transfer auto-eval — ledger `automl_package/examples/capacity_ladder_results/D_TOY_PROBES/depth_graded_pilot_z120_seed1.json`.
Fired True on exactly ONE run — Z120 seed 1 — caused by the shared_readout optimization blow-up,
not capacity (trajectory in that JSON: best CE 1.339 @1500 → 5.85 @3500). It touches NO bar
(G4 reads per_length_head + wide only; G1 is S5-side). On S5, transfer is False 3/3 =
prediction refuted.

---

### Task D5: G-DEPTH **substrate** verdict (half-gate, orchestrator/opus) — deps: D7

**Files:**
- Create: `docs/depth_capacity/verdict_per_input_depth.md` (structure mirrors
  `docs/width_mse_2026-07-16/verdict_variable_width_mse.md`: bars recap → per-seed tables →
  mechanism → claim ledger → file manifest).
- Modify: this file (gate row below) + `MASTER.md` (strand-3 row, Corrections).

**Orchestration:** parallel: no · deps: D7 · tier: opus/main · scale: static · shape: discovery ·
verify: verdict doc exists; gate row filled; `~/dev/.venv/bin/python -m pytest
docs/plans/capacity_programme/test_plan_gates.py -q` green.

**Pre-registered rule (substrate clause; supersedes the retired old-D4 rule):** the SUBSTRATE
half of G-DEPTH = PASS iff bars G1–G4 pass on ≥ 2 trustworthy seeds with rulings R1–R3 applied.
Current evidence: 3/3. The verdict doc must state explicitly that it certifies the substrate
only — selection-without-oracle is D8's clause, and G-DEPTH final = both.

**Rulings — pre-decided; write them into the doc, do not re-litigate:**
- **R1 (ℓ=4 exclusion):** excluded from all bar reads. Defense (must appear verbatim in the
  doc): post-hoc in timing but pre-computable from design alone — 4 generators ⇒ 256 total
  words, split 128/128, vs 120 classes ≈ 1 training example per class; no method can generalize
  from that rung, and its instability signature (per_length_head 0.719/0.625/0.766 across seeds,
  the only unstable cells in the table) matches starvation, not depth. State it as a design
  error caught late, not silently dropped. Future ladders start at ℓ=6.
- **R2 (bar-carrier arm):** `per_length_head` carries G1 (matches the D1b Step-5 deliverable and
  the coded `_check_bars`); `shared_readout` is reported alongside in every table (it dominates).
  Both readings pass — record that fact.
- **R3 (param matching):** G2 was pre-registered as PARAM-MATCHED; the battery's wide arm has
  82,552 params vs the recurrent 16,376 (5.0×, conservative in the null's favour). The BINDING
  G2 read is D7's width-101 run (16,381 params, Δ 0.03%); the 512-wide result stays as the
  over-provisioned robustness point. Both stall → G2 clean at parity and beyond.
- **R4 (blow-up run):** Z120 seed-1 shared_readout is flagged as an optimization failure
  (trajectory pointer above), touches no bar, is NOT rerun — rerunning until clean is
  cherry-picking. Cite D6's guard as the systemic fix.
- **R5 (refuted prediction):** presented as a RESULT, not a failure — the width/depth readout
  asymmetry is the mechanism finding and the report-(b) spine; include the stability caveat.

**Authorship:** as the user, full rigor, zero AI/tool provenance (MASTER Decision 10;
`research-report` skill for the doc pass).

- [x] Step 1: verified D7's JSON on disk (`params` = 16381; wide-101 stalls at ℓ=10 = 0.447 ≤ 0.60;
      R3 branch rule applied → G2 holds at parity, D7 JSON is the binding G2 read).
- [x] Step 2: verdict doc drafted — `docs/depth_capacity/verdict_per_input_depth.md` (rulings R1–R5
      verbatim §6; D6 divergence guard folded into the R4 quarantine §5).
- [x] Step 3: gate row filled (below) + MASTER strand-3 row; plan gates green (6/6).

RESULT: **SUBSTRATE half of G-DEPTH = PASS (3/3 seeds, R1–R5)** — ledger `automl_package/examples/capacity_ladder_results/D_TOY_PROBES/depth_graded_pilot_s5_seed0.json` (+5 siblings) and `.../depth_graded_pilot_s5_seed0_w101.json`.
Reads: G1 deep fits, G2 width stalls at parity (w101 = 0.447 ≤ 0.60), G3 gap monotone 3/3, G4 Z120
clean with the two D6-flagged cells quarantined. Verdict prose in the D5 Files entry above. G-DEPTH
final still pending the D8 selection half. *(superseded — G-DEPTH closed, see top entry)*

**Non-goals:** no new experiments beyond citing D7; no FlexNN/package changes; no halting
design (that is J0/M3+ scope).

### Task D6: divergence guard for the convergence gate (sonnet) — parallel with D7

**Files:**
- Modify: `automl_package/examples/convergence.py` (+ its `--selftest` block).

**Orchestration:** parallel: yes (write-set disjoint from D7) · deps: — · tier: sonnet ·
scale: static · shape: execution · verify: selftest exits 0; `ruff check` clean; calibration
check below passes.

**Defect (verified 2026-07-17):** `trustworthy = converged ∧ ¬hit_cap ∧ ¬still_improving` has
no divergence concept — a run whose held-out loss explodes and plateaus high patience-stops and
is certified trustworthy (case: Z120 seed-1 shared_readout, best CE 1.339 → final 3.032).
Best-weights restore already limits the damage (reported metrics come from the best point);
the flag is what lies.

**Spec (decision-complete):**
- Add module constants `DIVERGENCE_ABS_EPS = 0.2` (nats) and `DIVERGENCE_REL_FACTOR = 0.5`.
- Add property `diverged`: final-trajectory-point loss − `best_val` >
  `max(DIVERGENCE_ABS_EPS, DIVERGENCE_REL_FACTOR × best_val)`. The absolute floor is mandatory:
  a pure ratio misfires near zero (healthy best 0.0015 + harmless 0.003 wobble = ratio 3).
- Fold into `trustworthy` (`∧ ¬diverged`) and add `"diverged"` to `summary()` (additive key;
  no consumer break — landed JSONs are static and are NOT re-scored).
- Extend the selftest with a synthetic diverging trajectory asserting `diverged=True,
  trustworthy=False`, and a healthy near-zero trajectory asserting `diverged=False`.
- **Calibration check (must pass before done):** applying the rule to all 18 arm-convergence
  records in `automl_package/examples/capacity_ladder_results/D_TOY_PROBES/depth_graded_pilot_s5_seed0.json`
  and its 5 siblings flags EXACTLY one — (z120, seed 1, shared_readout). More or fewer ⇒ retune
  the two constants, do not ship.

- [x] Step 1: implement `diverged` property + the two constants; fold into `trustworthy`; add
      `"diverged"` to `summary()`.
- [x] Step 2: extend the selftest (synthetic diverging trajectory → `diverged=True,
      trustworthy=False`; healthy near-zero trajectory → `diverged=False`).
- [x] Step 3: run the calibration check (exactly 1 of 18 flagged); ruff; selftest.

**Non-goals:** no change to `ConvergenceTracker` patience/stopping semantics; no re-scoring or
rewriting of any landed ledger JSON.

### Task D7: param-matched wide-shallow robustness run (sonnet) — parallel with D6

**Files:**
- Modify: `automl_package/examples/depth_graded_toy.py` — add `--wide-width N` (default the
  current `WIDE_WIDTH`); when non-default, train ONLY the wide_shallow arm (skip recurrent arms
  and bars) and write `depth_graded_pilot_{group}_seed{seed}_w{N}.json` (collision-free) with
  that arm's `by_length` + convergence record.
- Create (by run): `.../D_TOY_PROBES/depth_graded_pilot_s5_seed0_w101.json`.

**Orchestration:** parallel: yes · deps: — · tier: sonnet · scale: static · shape: execution ·
verify: selftest exits 0; ruff clean; the JSON lands in `D_TOY_PROBES/` with
`params = 16381` recorded (161·101 + 120; recurrent = 16,376, Δ 0.03%).

**Run:** S5, seed 0, `--wide-width 101`, detached per MASTER environment rule, `AUTOML_DEVICE=cpu
OMP_NUM_THREADS=3`; expected ≈ 20–40 min (wide arm alone; its 512-wide sibling stopped at epoch
3250 of a ~100-min 3-arm job).

**Branch rule (pre-decided — no user question either way):**
- width-101 STALLS at ℓ=10 (expected: 512-wide already fails while memorizing) → R3 footnote:
  G2 holds at parity; verdict cites this JSON as the binding G2 read.
- width-101 FITS (upset) → G2-as-registered fails at parity; the verdict records the separation
  as "vs over-provisioned width", G-DEPTH then rests on G1/G3/G4 + the make-or-break table, and
  the gate row says exactly that. Record honestly; do not suppress.

- [x] Step 1: add `--wide-width` + wide-only mode + tagged output filename; selftest + ruff.
- [x] Step 2: launch the detached run (S5, seed 0, `--wide-width 101`); verify the JSON lands
      with `params` = 16381.
- [x] Step 3: RESULT line here with the branch rule applied.

**Non-goals:** no other seeds/groups; no bar re-computation; no change to default-path behavior
of the script (default invocation must produce byte-identical logic to the landed battery).

### Task D8: depth SELECTION without oracle — the charter's other half (umbrella: D8a design ⛔ + D8b build/battery)

**Why the graded S5 toy cannot carry this (recorded):** required depth = word length is
syntactically visible (padding pattern), and error-vs-depth per input is a step function —
selection is degenerate. The selection toy must have per-input depth-hunger that is (a) HIDDEN
(discoverable only by computing) and (b) ideally GRADED (running shallower degrades gracefully,
giving the router a real tradeoff curve, as width had).

**Protocol (mirrors the certified width selection protocol — reuse, don't reinvent):** train
ONE anytime-depth net (sandwich over unroll budgets T, per-budget convergence gating with the
D6-fixed gate); build the per-input per-T held-out error table on slice B; **DISTILL** a router
input → T from that table with the cheapest-within-tolerance rule (`δ_tie = 0.25`, matching
`docs/width_mse_2026-07-16/verdict_variable_width_mse.md` §6; sweep δ_tie in the battery per
that doc's §recommendation); deploy = executed-steps hard-pick vs best-fixed-T (val-selected),
router trained on slice B, applied to test, NO oracle depth labels anywhere. Selector/deploy
bar implementations in `automl_package/examples/sinc_width_experiment.py` are metric-agnostic
over a `{level -> err}` table — import/adapt, do not re-implement.

**Mechanism doctrine (USER, 2026-07-17 — binding here and on all downstream selection work):**
selection is learned by POST-HOC DISTILLATION from the held-out error table — the certified
width mechanism (`sinc_width_experiment.py` `_fit_selector_mse`: "Distills the PRIMARY
hard-label cheapest-within-tolerance router off slice-B's MSE table"). In-sample / in-training
selection is NOT attempted as the primary — it failed for width, and the FlexNN ELBO
depth-selection attempt was refuted (M0, flexnn-moe strand). An in-training selector may appear
ONLY as a labeled comparison arm after the distilled primary passes, never instead of it.

**D8a — TOY DESIGN SPEC ⛔ USER REVIEW GATE.**

**Files:** Create: `docs/depth_capacity/depth_selection_toy_design.md` (**DRAFTED
2026-07-17** by the design session; probe scripts live in the session scratchpad, promoted as
D8b selftest modes on approval, not as-is).

**Orchestration:** parallel: yes (no repo write-conflicts) · deps: — · tier: opus/main
(design session, NOT a delegated worker) · scale: static · shape: discovery · verify: doc
exists; §7 probe results filled; plan gates green.

- [x] Step 1: design doc drafted (C1′ commitment-point words primary; original C1 demoted on
      the §1 tension; C2 fallback; exact arithmetic in doc §2).
- [x] Step 2: falsifier probe round 1 — CAUGHT a realized-t* scan bug (contiguity assumption)
      before anything was built; definition corrected (doc §3.1), probes fixed.
- [x] Step 3: probes run — THREE falsifier rounds, all REJECTED for concealment (79%/75%/70%
      vs chance; structural: suffix-identity commitment is intrinsically surface-leaky — doc
      §7 table). Gradedness round 1 inconclusive (undertrained; redesigned convergence-gated,
      not yet run — first D8b action under fork option A). Iteration STOPPED per the
      no-4th-unilateral-round rule.
- [x] Step 4: DELIVERED 2026-07-17 with the DESIGN FORK as the review question (doc §6).
- [x] Step 5: **⛔ GATE CLEARED — USER SIGN-OFF 2026-07-17. Decision = Option A** (drop concealment
      as a kill criterion; keep the falsifier as a measured covariate; add a pre-registered
      surface-baseline control), **with the narrow-deep width-substitutable graded toy documented as
      the fallback** (trigger: gradedness make-or-break fails on 2 seeds). Charter per user: *learn
      depth as a function of the input, correctly; difficulty-of-detection is not a concern.* D8a
      COMPLETE. D8b starts (design doc §6 + §6.1 records the resolution).

Toy design is where yesterday's cycle failed (four candidates designed by plausibility, killed
only by experiment; the ℓ=4 starvation was arithmetic-checkable before any run). The selection
toy is therefore designed as an explicit reviewed document BEFORE anything is built:
- Create: `docs/depth_capacity/depth_selection_toy_design.md`, developing (or rejecting) the
  starting candidates below and at most one designer-added alternative. For EACH candidate the
  doc must contain, explicitly:
  1. **Exact generative spec** — the math of data generation (group, generators, hidden-r
     distribution, padding/insertion process, label), and the **dataset arithmetic per rung**
     (#distinct inputs, train/val counts, class balance vs 120 classes) — the ℓ=4 starvation
     check is MANDATORY per rung, before any run.
  2. **Hidden-ness argument + cheap falsifier** — why required depth is not readable from
     surface statistics, PLUS a ≤15-min probe that must FAIL for the design to survive: train a
     shallow classifier input → r; if it succeeds, the hunger is visible and the candidate is
     rejected on arithmetic/probe grounds, not after a 2-h pilot.
  3. **Gradedness argument + micro-probe** — why error(input, T) should degrade gracefully
     (mechanism, e.g. iterated local cancellation), what theory supports it, and a ≤15-min
     micro-probe result. Every unverified hypothesis is LABELED as such (No-guessing gate).
  4. **Confound ledger** — coverage-vs-length, memorization vs generalization split, router
     surface shortcuts, optimization instability (reads the D6 `diverged` flag), seed
     sensitivity; for each: the control that detects it.
  5. **Pre-registered pilot bars + kill criterion + cost estimate** (target ≤ ~2 h CPU/pilot).
- Theory-grounding section: what Barrington 1989 / Liu et al. 2022 / Malach et al. actually
  license here, and which construction assumptions go BEYOND the theorems (labeled).
- **Then STOP and deliver the design doc to the USER for review. D8b does not start without an
  explicit GO.** (User-mandated 2026-07-17: "toy design has to be explicit & well thought out
  … it needs proper guidance or specs for me.")

**Design doc: DRAFTED 2026-07-17 → `docs/depth_capacity/depth_selection_toy_design.md`
(awaiting probe results + user review).** *(SUPERSEDED: signed off Option A 2026-07-17; D8b
complete.)* The design session's analysis replaced the original
sketches: **C1′ commitment-point words is the PREFERRED candidate** (certified sequential
substrate; hidden per-word commitment point t* with a PROVABLE Bayes knee — tail is exactly
identity-product, sampled uniformly by DP; exact arithmetic in the doc §2). The original C1
(reducible words + novel iterated-full-input net) is DEMOTED (design defect: tree reduction
makes required iterations uniform ~log L — doc §4); C2 mixed-solvability stays the fallback.
Reviewer decisions requested in doc §6.

**D8b — build + pilots + battery (starts only on the user's GO on D8a).**

**Files:** Create: `automl_package/examples/depth_selection_toy.py` (toy per the approved
design: generator + DP tail sampler + realized-t* + anytime net + router + deploy; probes as
`--selftest`/`--probe` modes). Create (by runs): `.../D_TOY_PROBES/depth_selection_*.json`.
Modify: this file (RESULT lines, gate row) + `docs/depth_capacity/verdict_per_input_depth.md`
(selection clause appended) same turn results land.

**Orchestration:** parallel: no (sequential within task) · deps: D8a GO + D6 (battery reads
the `diverged` flag) · tier: sonnet build/battery, opus for the kill/proceed reads · scale:
static · shape: execution against the approved design · verify: selftest exits 0; ruff clean;
pilot + battery JSONs land; bars S1–S4 evaluated.

- [x] Step 1: toy implemented — `automl_package/examples/depth_selection_toy.py` (A5 five-cycle
      composition L=16, anytime depth-dropout net, distilled router, gradedness/surface probes; A5 added
      to `depth_composition_toy.py::build_group` with axiom/span self-check). Selftests green (realized-t*
      correctness + sampler assertion at all strata), ruff clean. S2 ceiling frozen = 0.2875 (A5 Bayes(g=2)
      +0.10; `automl_package/examples/capacity_ladder_results/D_TOY_PROBES/depth_selection_arithmetic_a5.json`). Ladder {6,8,10,16}
      confirmed coverage-safe. *(SUPERSEDED by AS-RUN note below: involutions/L=10/{6,8,10}/0.35 —
      this L=16/five-cycle/{6,8,10,16}/0.2875 construction trained to chance and was rebuilt.)*
- [x] Step 2: gradedness pilot (S1–S2, make-or-break) — **PASS 2/2 seeds**. S1 full-T acc ≥ 0.90/stratum
      (seed0 {6:.949,8:.957,10:.946}; seed1 {6:.938,8:.931,10:.901}); S2 knee_ratio ≥ 1.0 all rungs,
      acc@t*−2 = 0.0 ≤ ceiling 0.35. Make-or-break cleared. (First build trained to chance → root-caused
      + rebuilt, see AS-RUN note below; guard "fail S1 twice → escalate" not triggered.)
- [x] Step 3: router distilled (Decision 13, cheapest-within-tolerance over held-out per-T error table);
      deploy battery **PASS 2/2**. S3: mean-T 8.00/7.99 < best-fixed-T 10, held-out MSE *improves*
      (0.023/0.045 vs 0.049/0.077), clears design's ≤0.8× too. S2 correlation ρ(T*,t*)=1.000/0.993 ≥ 0.7.
      S4: router input = raw one-hot word only, `router_features` in each deploy JSON, no oracle. S5
      covariate: surface MLP recovers t* at 100% (Option A → covariate, not kill; answer f(x) still not
      surface-computable, acc@t*−2=0).
- [x] Step 4: selection verdict appended (`docs/depth_capacity/verdict_per_input_depth.md` §10–§12);
      gate row filled below (G-DEPTH = D5 ∧ D8 = PASS).
- **Escalation:** every approved candidate fails its pre-registered pilot bars → STOP, write
  the post-mortem into the design doc, batch the redesign question for the user (mirrors D1's
  once-then-escalate rule). Do not invent new candidates autonomously.

**Pre-registered bars (fixed at D8a sign-off, Option A; pilots may only tune constants BEFORE
their own run, then frozen). Construction = C1‴: A5 five-cycle generators, L=16, ladder {6,8,10,16},
~40k words/stratum.**
- S1 *anytime substrate:* at full T the anytime net matches the D1b recurrent's quality class
  (held-out acc ≥ 0.90) — the sandwich training didn't break the substrate.
- **S2 *graded knee* (THE MAKE-OR-BREAK):** per-input error(T) knees at the input-dependent
  realized commitment point t*(x), degrading GRADUALLY below it, not off a cliff — mean over stratum
  of acc(x, T=t*(x)) ≥ 0.95 × acc(x, full T), and acc at T=t*(x)−2 within 10 pp of the Bayes ceiling
  (≤ 0.35). Correlation form: Spearman ρ(T*, t*(x)) ≥ 0.7 on slice B. **Fail on 2 seeds → FALLBACK**
  (design doc §6: narrow-deep width-substitutable graded toy; taken without further user ceremony).
- S3 *router without oracle:* deploy executed-steps < best-fixed-T compute, with held-out
  error within δ_tie of best-fixed — the width deploy claim, re-derived for depth.
- S4 *no-oracle check:* the router's input is the raw input only (no r, no length labels);
  document the router features in the JSON.
- **S5 *surface-baseline control* (Option A, replaces the concealment kill):** report the distilled
  router's routing/deploy alongside a router of the same shallow architecture trained on the raw input
  directly. Both land in the JSON; the verdict states honestly whether the surface router matches. The
  falsifier (per-stratum surface detectability of t*) is recorded as a COVARIATE, never a kill.
  *Concealment is NOT a bar (user 2026-07-17: difficulty-of-detection is not our concern).*

**AS-RUN construction (deviates from the pre-registered C1‴ above; recorded honestly, all reversible,
charter intact — flagged for user veto on return).** The C1‴ construction trained to chance and was
root-caused + rebuilt before the battery: **L 16 → 10** (16 is past the recurrent block's GD-trainable
wall on A5); **per-T heads → one shared readout on the running (prefix) product** (per-T heads on the
full-word label carry impossible targets for T<t* and corrupt the shared block); **five-cycle → A5
involution generators** (five-cycles have no identity word < length 5, so no early-commitment stratum
is constructible at L=10; involutions fold at length 2 → ladder {6,8,10} feasible, full 60-class
coverage every rung); **n 40k → 3000/stratum** (40k = ~7 h/pilot on CPU, infeasible; 3000 converges
cleanly). **S2 ceiling refrozen 0.2875 → 0.35** for the new construction (arithmetic
`depth_selection_arithmetic_a5.json`). Bars S1–S5 themselves unchanged.

**RESULT (battery closed 2026-07-17, 2 seeds, all trustworthy, `diverged=false`):**
- **S1 PASS 2/2** — full-T acc seed0 {6:.949,8:.957,10:.946} / seed1 {6:.938,8:.931,10:.901} ≥ 0.90.
- **S2 PASS 2/2** — knee_ratio ≥ 1.0 every rung, acc@t*−2 = 0.0 ≤ 0.35; ρ(T*,t*) = 1.000 / 0.993 ≥ 0.7.
- **S3 PASS 2/2** — mean-T 8.00/7.99 < best-fixed 10; MSE improves 0.023/0.045 vs 0.049/0.077 (≤ δ_tie
  trivially; also clears design ≤ 0.8×best-fixed = 8.0).
- **S4 PASS** — router input = raw one-hot word (in_dim 40), labels distilled from held-out per-T error;
  `router_features`/`oracle_depth_labels_used:false` in each deploy JSON.
- **S5 covariate** — surface MLP recovers t* at 100% (both seeds); Option A → not a kill. Answer f(x)
  not surface-computable (S2 cliff), so only routing is surface-easy.
- Files: `depth_selection_{gradedness,deploy,surface}_seed{0,1}.json` + `depth_selection_arithmetic_a5.json`
  (verdict §11). Selection verdict prose: `verdict_per_input_depth.md` §10–§12.

**Non-goals:** no transformer/token-sequential halting (J0/M3+); no FlexNN/package changes; no
modification of the D1b graded toy or its landed ledgers.

---

**Wave plan (dispatcher):** Wave A = D6 ∥ D7 — **DONE 2026-07-17** (D6 divergence guard landed +
calibrated; D7 wide-101 run landed, G2 holds at parity). Wave B = D5 (substrate verdict) **DONE**
∥ D8a (design + probes) **DONE + USER SIGN-OFF (Option A)**. **⛔ GATE CLEARED 2026-07-17.**
Wave C = D8b build → gradedness make-or-break → selection battery (2 seeds) → selection verdict
appended to `docs/depth_capacity/verdict_per_input_depth.md` — **DONE 2026-07-17** (fallback not
needed; make-or-break passed 2/2 after the AS-RUN rebuild). Hand off to `width-depth.md` J0 next.

**Gate decision (G-DEPTH = D5 substrate ∧ D8 selection) = PASS (2026-07-17).** *Substrate clause =
PASS (D5, 3/3 seeds); selection clause = PASS (D8b, 2/2 seeds, bars S1–S4 + S5 covariate). Verdict:
`docs/depth_capacity/verdict_per_input_depth.md` §12. → `width-depth.md` J0.*

---

## Done ledger

- D0 · 2026-07-16 · `MASTER.md` Corrections entry (citations re-verified at HEAD)
- D1 · 2026-07-17 · 6 ledger JSONs in
  `automl_package/examples/capacity_ladder_results/D_TOY_PROBES/` (RESULT lines in Task D1)
- D5 · 2026-07-17 · substrate verdict PASS 3/3 (`docs/depth_capacity/verdict_per_input_depth.md` §§0–9)
- D6 · 2026-07-17 · divergence guard (`convergence.py` `diverged`), 2 genuine blow-ups flagged
- D7 · 2026-07-17 · param-matched wide-101 stalls 0.447 @ℓ10 (`depth_graded_pilot_s5_seed0_w101.json`)
- D8 · 2026-07-17 · selection verdict PASS 2/2 (`verdict_per_input_depth.md` §10–§12); toy
  `depth_selection_toy.py` + 6 battery JSONs. **G-DEPTH = PASS (both halves).**
