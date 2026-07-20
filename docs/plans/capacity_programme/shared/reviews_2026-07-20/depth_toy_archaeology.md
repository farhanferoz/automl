# Depth Toy Archaeology — Working Notes (raw capture, will be organized into final report)

Source: /home/ff235/dev/MLResearch/automl/automl_package/examples/capacity_ladder_results/RESULTS.md (738 lines, read in full, part 1 = lines 1-547 captured below)

## corpus-search results for headline question (searches run, automl index STALE per warning — re-index not attempted, read-only constraint)
- Query "depth as replacement for width in per-input capacity toy" — NO relevant automl hits (all autocast/turing repo hits about depth x width^2 scaling heuristics, unrelated repo). automl hit #8 was docs/architecture_analysis.md "Input-dependent depth selection... not all inputs require the same model complexity" (generic Early Exit lit note, not the headline idea).
- Query "swap the dial from width to depth reuse width toy" — NO automl hits at all in top 10 (all turing/autocast repos).
- Query "cascade anytime-prefix construction transfer to depth" — NO automl hits (top hit automl-adjacent none). 
- PRELIMINARY: corpus-search finds NO trace of the headline idea under these phrasings. Will try more phrasings (W_CASCADE dir exists in capacity_ladder_results — must investigate).

## RESULTS.md lines 1-547 — WS2 (F-lane) = the depth-capacity lane. Verdict: R3_verdict.md (GO, qualified)

Status table (line 10-14): WS2 (FlexNN depth): tasks F0,F1,F2,F3; certified by R3 (GO, qualified) + REV.
Headline: "Aggregate depth DETECTION discriminates w/ clean control (dedicated-sweep knee: G r*=2 {2,2,2}, H marginal, G-flat r*=0); nested-knee reads (G r*=3, H +0.64 first inc) are interference-biased (X5); per-input read at the noise floor at N=500"

### WS2 full section (RESULTS.md:115-186)
- Toys: G (varying required capacity: linear x<0 / compositional x>=0), G_flat (negative control, uniform complexity), H (SNR dial, fixed mean/sigma(x) varies)
- Method: ONE prefix-nested-depth FlexibleHiddenLayersNN surrogate (per-sample depth d~Uniform{1..6}, shared trunk, depth-d readout, BN off), held-out depth score tables N=500/seed.
- G per-bin beats global: NOT ESTABLISHED (split-fragile) [REV W4]. Only seed2 tercile advantage clears (+0.0122±0.0046, 2.67 SE) under ONE hardcoded split; review re-ran under 8 alt split seeds -> clears 2SE on only 3/9 partitions -> noise floor. Global knee detects every seed (per-seed r* {4,2,3}, pooled r*=3).
- G-flat (negative control): MET CLEANLY. 0/3 seeds beat global at terciles, 0/6 at sextiles. Global knee abstains r*=0 all 3 seeds+pooled.
- H (SNR dial): NOT MET (directionally unconfirmed). Tercile argmax varies 2/3 seeds but wrong/no consistent direction (snr_trend_down=False all 3). Global knee still detects (r* {3,2,2} per-seed, pooled r*=2).
- Sextile stability = noise-floor tell: G s2 loses tercile win at sextiles; H s2 gains one it lacked - single-cell flips = noise floor signature.
- Nesting cost LARGE at shallow end (F2): B_coh_all_depths_all_toys_pass=false; 26/54 cells fail (48%), max cost -0.72 nat (~17.5 SE) at depth1 on H; top rung fails 5/9.
- REV X5 cross-check: dedicated fixed-depth knee (not nested) on same held-out rows: G pooled r*=2, per-seed {2,2,2} COHERENT vs nested r*=3 {4,2,3} -- nested G read is one rung HIGH. H: dedicated first increment +0.049 (not +0.64 -- 13x nesting-inflated); pooled r*=2 agrees but per-seed {2,3,0} fragile -- marginal at N=500.
- Mechanism: nested knee inherits GRADIENT of nesting cost (mirrors WS1's B-knee mid-rung stall, but here decaying depth-1 interference INFLATES early increments -> over-read, opposite direction from WS1's under-read).
- Corrected WS2 headline: "aggregate depth DETECTION discriminates with a clean control on both instruments (G strongly, H marginally), but the nested ladder's selected depth is biased by its own interference profile."
- F4 GO'd qualified (a router that ties/beats global + compute saving, NOT decisive per-input NLL win) -- BUT SEE BELOW: X3+X1 later closed F4 as NO-GO.
- F5 stays "far-future user-gated" port.
- Artifacts: F2/nested_toy{G,G_flat,H}_seed{0,1,2}.pt, F2/f2_summary_{G,G_flat,H}_s012.json, F3/f3_summary.json, R3_verdict.md.

### X-queue (RESULTS.md:300-368) — depth-relevant items
- X3 (line 306-312): repeated cross-fit re-issue of F3 per-input DEPTH read. 50 random fit/score splits/seed, Nadeau-Bengio split-aware SE. G per-input advantage corrected-t {-0.34,-0.60,+1.00} (0/3 >2), split-pass 0.047 <= G-flat control 0.073; H SNR-trend 0.30 (no majority). F3's "+2.67 SE" was seed-2 under ONE fixed split (cherry pick by omission, not intent). CONCLUSION: "Per-input depth signal at the noise floor — clean negative on signal-vs-control discrimination at N=500... **⛔F4 = no-go.**" Selftest confirms estimator fires (t=6.4) on genuine signal (so instrument works, toy doesn't show it).
- X1 (line 313-325): hierarchical partial-pooled per-bin stacking (power-limited-vs-absent discriminator). New estimator `hierarchical_perbin_stack`. On real G: hier_vs_global 0/3 sig, pass 0.093 ~= G-flat 0.08 -> recovers NOTHING. Detection curve: resolves per-region gap>=0.1 nat; G's advantage is 2-3 ORDERS OF MAGNITUDE below that floor. Conclusion: "F4 = CONFIRMED no-go on the strongest available estimator ... WS2 per-input depth sub-lane CLOSED."
- X10 mentioned as an open follow-up ("power curve — only if chasing sub-0.1-nat depth") at line 366 — NOT run at time of X-queue closure. (NOTE: later found to have been run as "P1" — see below.)

### P1 — depth power curve (RESULTS.md:512-547+, continues past truncation)
"P1 — depth power curve (depth-lane analog of T3 / X10 3-point, FlexNN nested ladder). Verdict: NO recoverable per-input depth signal up to N=8000; control-violated at N=8000 (adjudicated 2026-07-10, fresh context)"
- Sweep N_test in {500,2000,8000} (N_train x2) on F2 nested-depth ladder, X3's repeated cross-fit (50 splits, Nadeau-Bengio SE).
- Toy G tercile per-bin-stacking advantage over global stays pinned ~0.001-0.0015 nat at every N while detection floor falls ~9x -> G crosses floor on 2/3 seeds at N=8000 BUT G_flat (negative control) ALSO crosses on 2/3 seeds at SAME N, with statistically indistinguishable advantage (+0.0014 vs +0.0015 nat, Welch t=0.107) and LARGER signal/floor ratio (2.06 vs 1.09 for control!).
- Conclusion: "N=8000 crossing is a floor-drops-below-a-fixed-offset ARTIFACT, not a per-input depth signal." "per-input depth structure remains below a trustworthy detection floor up to N=8000 — the crossing is control-matched, so it certifies no recovery."
- This is explicitly framed as "the depth-lane mirror image of the T3 count-lane result: same power-curve apparatus, opposite outcome" (count/k lane DID recover with more data at T3; depth did NOT recover even at N=8000, and its "recovery" is actually a control-crossing artifact).
- No 5-point extension: floors monotone-decreasing on every seed/toy; G's crossing sequence [F,F,T] never un-crosses -> non-monotone-curve extension clause doesn't fire; control-crossing clause fired instead.
- STILL NEED: rest of P1 section past line 547 (truncated) — must re-read offset 548.

## RESULTS.md lines 548-738 — REST OF FILE (T2, H1, T1)

### T2 (line 557-608) — NOT depth. Multi-D count-mechanism port de-risk (kNN-routed selector, toy D in dim). ProbReg/count lane extended to multiple dims. GO. Not depth-relevant except as cross-lane contrast (mentions WS1's "knee overshoots count / arbiter is faithful" pattern reproduced in multi-D). SKIP for depth ledger except contrast note.

### H1 (line 610-664) — NOT depth. ProbReg two-phase selector vs joint gate vs oracle fixed-k. Qualified-GO. Not depth. SKIP for depth ledger.

### T1 — "provably-deep-required toy" (depth-lane discriminator) (RESULTS.md:666-739) *** MAJOR DEPTH FINDING ***
Verdict: "learnability-vs-representability finding; bar-(i) FAILED 0/3 and HARDENED unlearnable under multi-restart; PARKED at a user G-FORK (path 2 vs path 3); H2 LOCKED" (adjudicated 2026-07-10, fresh context)

- **Construction**: region A (x<0.5) linear; region B (x>=0.5) a 5-fold-composed tent map (Telgarsky construction, 32 linear pieces) at trunk width 8, sigma=0.1 (oracle log-lik +0.8836). This is a PROVABLE-by-construction depth requirement (Telgarsky 2016 cited) — the strongest theoretical form of "genuinely needs more depth."
- **Finding**: tent^5 is REPRESENTABLE at depth>=2 (by Telgarsky's theorem) yet GD training sits ~1.1-1.2 nat below the sigma=0.1 oracle AT EVERY TRAINED DEPTH on all seeds, while the linear region A reaches the oracle fine. This is a clean **learnability-vs-representability asymmetry**: depth CAN represent the function (proven) but gradient descent CANNOT learn it in practice.
- Conclusion: "a provable (Telgarsky-style) depth requirement lives in the GD-unlearnable regime, so it cannot serve as the depth-lane positive control."
- Bar (i) construction: FAILED 0/3 seeds. Region B stuck ~1.1 nat below oracle at EVERY depth 1-6 on ALL 3 seeds; does NOT progressively climb with depth (d1->d6 change only +0.12/+0.02/+0.03 nat = "not-learnable signature, not slowly-getting-there"). Region A reaches +0.79-0.84 ~= oracle (proves harness works, not a scoring bug).
- Path-1 (multi-restart hardening, R=8 independent random-init restarts, keep-best by TRAINING LL, no test-leak): region B stays 1.168-1.209 nat below oracle at EVERY depth, no closing at d>=3. Even optimistic best-of-8-BY-TEST (test-peeking) stays 1.141-1.171 nat below oracle at every depth. CONCLUSION: "HARDENED_UNLEARNABLE" — multi-restart does not rescue it; single-init unluckiness hypothesis CLOSED.
- **G-FORK** (user decision point) resolved 2026-07-11: user greenlit "Path 2" (bounded search): a bounded <=3-config, same-bars, all-3-seeds empirical search for a depth-preferring-AND-learnable target: tent^4@w8, tent^3@w8, tent^4@w6 (`capacity_ladder_t1_path2.py`).
- **Path-2 RESULT: ALL THREE configs read NOT_FOUND_UNLEARNABLE** (0/3 construction_pass each; region_b_pass 0/9 seed-config pairs). Even best-of-8-scored-on-test, region B stayed >0.7 nat below +0.8836 oracle at EVERY depth on EVERY seed. Minimum optimistic gaps: 1.095/0.980/1.098 nat for tent^4@w8 / tent^3@w8 / tent^4@w6. (tent^3@w8 the "near miss" at 0.98 nat but still failed — isolated single-depth B increments, never consecutive d1->d2 AND d2->d3 climb.)
- **TERMINAL OUTCOME: "path-3-as-fallback"**: No learnable depth-requiring positive control was found. "The representable-but-not-learnable asymmetry stands as the depth-lane finding; per-input depth closes as a compute-saving story. Whether ANY learnable depth-requiring target exists remains open (a 3-config search demonstrates the limit but cannot prove non-existence)."
- **H2 LOCKED**: every H2-unlock branch requires bar(i) PASS first; bar(i) failed on all paths -> H2 permanently locked under path-3 outcome. (H2 = some downstream hypothesis/task not otherwise detailed in RESULTS.md — NEED to check docs/plans for what H2 actually is, if referenced elsewhere.)
- Binding certification condition (adjudicator): "report T1 as a learnability-vs-representability finding PARKED at the user G-FORK; never as a depth-lane positive control, and never as a machinery failure (region A learnable proves the harness)."
- Artifacts: `capacity_ladder_results/T1/{PREREGISTRATION.md, FAIL_I_ADJUDICATION.md, BAR_II_ADJUDICATION.md, t1_summary.json, nested_toyT1_seed{0,1,2}.pt, t1_path1_summary.json, path1_toyT1_seed0.pt, T1_PATH1_ADJUDICATION.md, t1_path2_{tent4_w8,tent3_w8,tent4_w6}_summary.json, T1_PATH2_ADJUDICATION.md}`, `capacity_ladder_t1.py`, `capacity_ladder_t1_path1.py`, `capacity_ladder_t1_path2.py`.
- THIS IS THE CLOSEST THING FOUND SO FAR TO A "DEPTH-LANE POSITIVE CONTROL" TOY, and it is explicitly a THEORY-DRIVEN construction (Telgarsky tent-map composition), NOT a port of a width toy. Need to check whether T1's tent-map construction has any lineage from width toys, and whether the headline "port the width toy to depth" idea is DIFFERENT from T1 or IS effectively what T1 became after the G-FORK. Will check docs/plans and PREREGISTRATION.md for T1 to see stated rationale/lineage.

## W_CASCADE / W_MRL / W_KDROPOUT_CONVERGED / W_INDEP / W_CONVERGED dirs — WIDTH lane (need to verify not secretly a width->depth port)
Directory listing done (ls -la). All are dated Jul 11-17 (WIDTH lane, later than the Jul 10 capacity-ladder/depth work). Naming (W_CASCADE, W_MRL, W_KDROPOUT, W_INDEP, W_CONVERGED) strongly suggests "Width" prefix experiments — consistent with project memory note `project_width_kdropout_arch_error` (IndependentWidthNet vs NestedWidthNet width work). MUST check W_CASCADE/PREREGISTRATION.md directly to (a) confirm it's width not depth, (b) check if "cascade" construction is described as potentially transferable to depth (the memory note `project_depth_revisit_with_width_ideas` said "after width, first test if cascade/anytime-prefix ideas transfer to depth" — need to find where/if that transfer was attempted).

## *** MAJOR DISCOVERY: A SECOND, LATER, SUCCESSFUL DEPTH PROGRAMME EXISTS ***
Everything above (RESULTS.md WS2/F-lane/T1/P1) is dated 2026-07-10/11 and is part of the OLDER
"capacity-ladder programme" (`automl_package/examples/capacity_ladder_results/`). There is a SEPARATE,
LATER strand under `docs/plans/capacity_programme/` (the FlexNN-core / MASTER-Decision programme) dated
2026-07-16/17 that CONCLUDES DEPTH WORKS. This completely changes the picture and must be the headline
of the ledger. Source: /home/ff235/dev/MLResearch/automl/docs/plans/capacity_programme/depth.md (425 lines, READ IN FULL).

### depth.md header (line 1): "Strand: depth (→ G-DEPTH) — COMPLETE 2026-07-17. G-DEPTH = D5 substrate ∧ D8 selection = PASS. → width-depth.md J0"

**THE PIVOT (depth.md:37-49, 2026-07-16), kept for the record verbatim:**
> "the depth toy is a GROUP WORD-PROBLEM, not a 1-D regression toy (construction + grounding, kept
> for the record). All four 1-D smooth-regression candidates failed, and the failure is a THEOREM:
> depth-hunger ⊥ GD-learnability for smooth 1-D targets (Malach & Shalev-Shwartz 2019 arXiv:1903.03488;
> Malach et al. 2021 arXiv:2102.00434 — GD-learnability requires weak shallow-approximability). Full
> write-up: docs/depth_capacity/depth_toy_negative_note.md. The escape is COMBINATORIAL: compose a
> length-n word of S5 (non-solvable) generators -> 120-way classification of the product; difficulty
> knob = n = #sequential steps = per-input depth. Barrington 1989: width provably cannot substitute
> for depth here. Z120 (abelian, same order 120) is the width-substitutable control (Liu et al. 2022
> arXiv:2210.10749)."

*** THIS IS A DIRECT, CITED, THEORY-BACKED ANSWER TO THE ARCHAEOLOGY BRIEF'S FAILURE-MODE QUESTION (ii):
"the toy required depth but gradient descent could not learn it" — there is now a THEOREM (Malach &
Shalev-Shwartz 2019 + Malach et al. 2021) stating that smooth 1-D regression targets with a genuine
depth requirement are PROVABLY NOT GD-learnable (GD-learnability requires "weak shallow-approximability"
which a genuine depth-hunger target lacks). This explains T1's tent-map "learnability-vs-representability
asymmetry" finding retroactively as an INSTANCE of this theorem, not a one-off curiosity. NEED TO VERIFY
this citation and "all four 1-D smooth-regression candidates failed" claim against depth_toy_negative_note.md
— it may explicitly discuss/reject the width-toy-ported-to-depth idea as one of the "four candidates."**

### D1 — the substrate toy (S5 group word composition + Z120 abelian control) (depth.md:68-101)
- Construction: `depth_composition_toy.py` (S5 + Z120, 3 net families: recurrent/weight-shared, per-length-head, wide-shallow). Difficulty knob = word length n (per-input depth requirement), classification (CE) not regression.
- Bars: G1 deep arm fits (held-out acc>=0.90 whole ladder {4,6,8,10}) / G2 width stalls (wide-shallow <=0.60 at long end) / G3 (deep-wide) gap increases monotonically with length on >=2 seeds / G4 Z120 control shows NO divergence (all arms >=0.90, gap~0).
- RESULT: D1b graded battery — ALL BARS PASS ON ALL 3 SEEDS (with ℓ=4 excluded as a design-arithmetic starvation artifact, ruling R1). G1 per_length_head >=0.958/.963/.980 at ℓ=10 (shared_readout even better, >=0.990 everywhere). G2 wide-shallow at ℓ=10 = 0.386/0.382/0.408 despite TRAIN acc>=0.93 (memorizes, doesn't generalize = clean width-can't-substitute finding). G3 gap grows +0.07/+0.26/+0.57 at ℓ=6/8/10, consistent across seeds. G4 Z120 all arms >=0.916 (control clean, no artificial divergence).
- Artifacts: `capacity_ladder_results/D_TOY_PROBES/depth_graded_pilot_s5_seed{0,1,2}.json`, `depth_graded_pilot_z120_seed{0,1,2}.json`.

### "The pre-registered transfer prediction — REFUTED 3/3" (depth.md:23-35) — a DIFFERENT transfer question than the headline (about readout architecture, not toy construction)
- Prediction (written before battery ran): shared-readout arm would FAIL the fit bar (mirroring width's failure where FlexNN's shared output_layer causes readout interference) while per-length/per-capacity heads would pass.
- OUTCOME: OPPOSITE. shared_readout is the STRONGEST arm on every seed (S5 val_acc 1.000 at ℓ=4/6/8, >=0.990 at ℓ=10, all 3 seeds); per_length_head merely passes.
- Mechanism: "readout interference is WIDTH-SPECIFIC — a weight-shared recurrent block presents every depth with the SAME state space, so one readout serves all depths; width prefixes hand each capacity a DIFFERENT representation fighting over one readout."
- This sets the design prior for width-depth.md's joint strand: per-width heads x depth-shared readout (asymmetric).
- Stability caveat: 1/6 shared_readout runs (Z120 seed1) hit an optimization blow-up (val CE 1.34->5.85 @ epoch3500, never recovered) — motivated Task D6 (divergence guard).

### D5 — G-DEPTH substrate verdict: PASS 3/3 seeds (depth.md:105-154)
Verdict doc: `docs/depth_capacity/verdict_per_input_depth.md`. Rulings R1 (ℓ=4 excluded, starvation not depth failure) / R2 (per_length_head is bar-carrier, shared_readout reported alongside) / R3 (param-matched D7 w101 run is the BINDING G2 read: 16381 params vs recurrent's 16376, stalls 0.447<=0.60 at ℓ=10) / R4 (Z120 seed1 blowup flagged, not rerun, D6 is systemic fix) / R5 (refuted readout prediction reported as a RESULT not failure).

### D6 — divergence guard (depth.md:159-196): added `diverged` property to convergence.py; calibration check flags EXACTLY 1/18 records (Z120 seed1 shared_readout) — confirms the blowup was real and isolated, not systemic.

### D7 — param-matched wide-shallow robustness (depth.md:197-227): width-101 (16381 params, matching recurrent's 16376, Δ0.03%) STALLS at ℓ=10 (0.447<=0.60) — confirms G2 (width can't substitute) holds even at exact parameter parity, not just because the wide arm was under-provisioned.

### D8 — depth SELECTION without oracle (depth.md:229-412) *** THE OTHER HALF OF THE CHARTER — ALSO PASSED ***
- Why S5 graded toy (D1) can't carry selection: required depth = word length is SYNTACTICALLY VISIBLE (padding pattern) and error-vs-depth is a CLIFF (below length: chance; at length: perfect) -> a selector would just learn the identity map, certifying nothing. Needed a toy with HIDDEN, GRADED depth-hunger.
- Protocol explicitly stated as: **"mirrors the certified width selection protocol — reuse, don't reinvent"** (depth.md:237) — train ONE anytime-depth net (sandwich over unroll budgets T), build per-input per-T held-out error table, DISTILL a router via cheapest-within-tolerance rule (same rule as `docs/width_mse_2026-07-16/verdict_variable_width_mse.md` §6), deploy = executed-steps hard-pick vs best-fixed-T. **THIS IS THE "PORT THE WIDTH PROTOCOL TO DEPTH" MOVE THE USER RECALLS — but it is a port of the SELECTOR/DISTILLATION MECHANISM, not a port of the underlying TOY DATA CONSTRUCTION.** Need to check if a literal toy-construction port (e.g., reusing `nested_width_net.make_hetero`'s width-hungry-region idea but swapping in depth) was ALSO considered — check depth_selection_toy_design.md next.
- D8a design gate: doc `docs/depth_capacity/depth_selection_toy_design.md` DRAFTED 2026-07-17, delivered to user for review (⛔ mandatory user gate), THREE candidates considered: C1 (reducible words + novel iterated-full-input net, DEMOTED — tree reduction makes iterations uniform ~log L, design defect), C1' (commitment-point words, PREFERRED — hidden per-word commitment point t* with provable Bayes knee), C2 (mixed-solvability, fallback).
- Falsifier probes: 3 rounds run, ALL REJECTED for concealment (79%/75%/70% vs chance — a shallow classifier could partially predict the hidden commitment point from surface stats) — but "concealment" was later DEMOTED from a kill criterion by the user (Option A, 2026-07-17): "difficulty-of-detection is not our concern" — the user explicitly ruled that the depth requirement doesn't need to be perfectly hidden from a surface probe, only that the ANSWER (not the routing decision) must not be surface-computable.
- **User sign-off 2026-07-17, Option A**: drop concealment as a kill criterion, keep falsifier as a measured covariate, add a pre-registered surface-baseline control (S5 bar). D8a COMPLETE.
- AS-RUN construction deviated from pre-registered C1''' (recorded honestly): L 16->10 (16 past GD-trainable wall on A5), per-T heads -> one shared readout on running product, five-cycle -> A5 involution generators (five-cycles have no identity word <length5, blocking early-commitment stratum), n 40k->3000/stratum (compute). Bars S1-S5 themselves UNCHANGED.
- **RESULT (battery closed 2026-07-17, 2 seeds): S1 PASS 2/2 (anytime substrate matches D1b quality, >=0.90). S2 PASS 2/2 (THE MAKE-OR-BREAK "graded knee" bar: knee_ratio>=1.0 every rung, acc@t*-2=0.0<=0.35 ceiling, Spearman rho(T*,t*)=1.000/0.993>=0.7 — clean graded, non-cliff, per-input depth signal, CORRELATED with the true hidden commitment point). S3 PASS 2/2 (router deploy: mean-T 8.00/7.99 < best-fixed-T 10, held-out MSE IMPROVES 0.023/0.045 vs 0.049/0.077 — router beats fixed budget). S4 PASS (router input = raw one-hot word only, no oracle depth labels — verified in JSON). S5 covariate (surface MLP recovers t* at 100% both seeds, but per Option A this is NOT a kill — the actual answer f(x) remains non-surface-computable, only the ROUTING decision is surface-easy).**
- Artifacts: `depth_selection_{gradedness,deploy,surface}_seed{0,1}.json`, `depth_selection_arithmetic_a5.json`. Verdict: `verdict_per_input_depth.md` §10-12.

### FINAL GATE (depth.md:410-412): **"Gate decision (G-DEPTH = D5 substrate ∧ D8 selection) = PASS (2026-07-17). Substrate clause = PASS (D5, 3/3 seeds); selection clause = PASS (D8b, 2/2 seeds, bars S1-S4 + S5 covariate)."** Hands off to `width-depth.md` Task J0 (joint width+depth dial), which in turn got BLOCKED (see width-depth.md notes above — J-1/J-2 dead at substrate, J-3 needs redesign, escalated to user 2026-07-17 eve).

## ============================================================
## *** THE HEADLINE QUESTION — ANSWERED: DONE (attempted, and it FAILED) ***
## ============================================================

**Verdict: DONE.** The exact idea — take the width toy's construction and substitute depth for width
as the capacity dial — was proposed BY THE USER, built, and run. It is "Candidate 4" /
`FIXEDWIDTH_HETERO` in `automl_package/examples/depth_toy.py`. It FAILED, and its failure (together
with 3 other candidates) is one of the four data points that closed the entire 1-D-regression-toy
depth lane and triggered the PIVOT to the combinatorial (S5 group-word-problem) construction — which
is the construction that eventually succeeded (G-DEPTH = PASS, 2026-07-17; see section below).

**Primary evidence — code docstring, `automl_package/examples/depth_toy.py:40-52`:**
> "Candidate 4 — fixed-width / vary-depth (FIXEDWIDTH_HETERO; the follow-up after candidates 1-3 hit
> the kill criterion...). Candidates 1-3 all tried to INVENT a function that is depth-hungry in an
> absolute sense... Candidate 4 inverts the strategy: it reuses the WIDTH toy's
> `nested_width_net.make_hetero` target VERBATIM — a flat-easy line spliced to `0.5*sin(x)` over
> `[0, 4*pi]`... which is KNOWN GD-learnable by a wide shallow net (that is the whole width program)
> and so clears the spectral-bias wall by construction. The depth question is then not 'is it
> learnable' but 'does DEPTH substitute for WIDTH at a FIXED small width w': pick w small enough that
> a depth-1 net at width w fits the flat region but STALLS on the sine, and ask whether adding depth
> AT THE SAME WIDTH w unlocks the sine."

This is EXACTLY the user's recalled idea, stated almost in the user's own words, and explicitly
attributed to the user: **CHANGELOG.md:97 calls it "the user's 4th `fixedwidth` candidate."**

**The verbatim, final, dated verdict — `/home/ff235/dev/MLResearch/automl/CHANGELOG.md:96-99`
(under the "## 2026-07-16" heading, i.e. dated 2026-07-16):**
> "Dispatched depth D0+D1: D0 passed, **D1 hit kill criterion** (3 candidates unconstructible);
> **user's 4th `fixedwidth` candidate also FAILED (stall is a cliff).** Two web-research workers
> established depth-hunger ⊥ GD-learnability is a THEOREM (Malach et al.); escape = S5
> group-composition (Barrington; Z120 control)."

**Corroborating raw data — the actual pilot run, verified by reading the JSON directly (NOT
paraphrased from memory):**
`automl_package/examples/capacity_ladder_results/D_TOY_PROBES/depth_toy_pilot_fixedwidth_seed{0,1}.json`
(sweeps width in {3,4,5,6,8}, training depth-1 and depth-4 narrow nets + a wide-shallow width-32
control at each; "trustworthy": true on every cell = genuine converged reads, not training-cap
artifacts). Bars: depth-1 "stalls" if `ratio_hard >= HUNGRY_STALL_MULTIPLE(2.0)`; depth-N "fits" if
`ratio_hard <= FIT_PASS_MULTIPLE(1.25)`.
- **width=3, seed 0:** depth1 ratio_hard=**39.43** (stalls hard) -> depth4 ratio_hard=**0.84** (fits
  cleanly) — LOOKS LIKE A CLEAN PASS on this seed alone.
- **width=3, seed 1:** depth1 ratio_hard=**22.59** (stalls) -> depth4 ratio_hard=**20.41** (STILL
  STALLED — barely better than depth1; does NOT clear the 1.25 fit bar by a huge margin). Depth does
  NOT rescue the fit on this seed at the width where depth-1 clearly stalls.
- **width=4, seed 0/1:** depth1 ratio_hard = 1.80 / 1.89 — BELOW the 2.0 stall threshold on both
  seeds (does not clearly stall at all), so this width cannot anchor the "depth-1 stalls, depth-N
  fits" comparison either.
- **Consequence:** there is no single width at which the pilot's own locking rule ("smallest w where
  depth-1 STALLS and depth-max FITS", `depth_toy.py:745-747`) is satisfied consistently across BOTH
  seeds — width=3 only half-passes (seed 0 only), width>=4 never triggers the stall condition. No
  `FIXEDWIDTH_W` could be honestly locked, and (consistent with this) no full-probe JSON
  (`depth_toy_probe_fixedwidth_seed{0,1}.json`, the P1-P4+CONTROL battery that would run AFTER
  locking) exists anywhere in the repo — the pilot-only JSONs are the entire run.
- This numeric pattern (works on one seed, collapses on the other, at the SAME construction) is the
  same seed-instability / optimization-landscape signature that killed Candidate 2
  (`hierarchical_spline`, `docs/depth_capacity/depth_toy_negative_note.md:37-49`), and is consistent
  with, though not verbatim identical wording to, the CHANGELOG's "stall is a cliff" summary (read as:
  the depth-1-stalls -> depth-N-fits transition the construction needs is not a graded, reproducible
  slope — it is either present as a sudden, seed-fragile jump or absent, failing the P3 "graded" bar
  candidates 1-3 also had to clear).

**Was it "considered and rejected on stated grounds", or "done"?** It is best classified as **DONE**
(built, run, produced a real negative result) rather than CONSIDERED-AND-REJECTED-WITHOUT-RUNNING —
unlike, say, alternative J-3 designs in `width-depth.md` that were rejected on paper. The idea was not
dismissed on plausibility; it was executed and it failed empirically, with an artifact on disk.

**Downstream consequence:** this failure (all 4 1-D candidates exhausted) triggered the theoretical
literature search that surfaced Malach & Shalev-Shwartz 2019 / Malach et al. 2021 (depth-hunger ⊥
GD-learnability is a THEOREM for smooth 1-D targets) and the PIVOT to the S5-group-word-problem
construction (Barrington 1989 depth-separation + Liu et al. 2022 width-substitutable Z120 control),
which is the construction that ultimately PASSED both halves of G-DEPTH on 2026-07-17 (see the
"SECOND, LATER, SUCCESSFUL DEPTH PROGRAMME" section above). **In that light, the failed width-toy
port was not a dead end — it was the last of the four negative data points that forced abandoning
smooth 1-D regression entirely, which is what led to the construction that worked.**

## verdict_per_input_depth.md — READ IN FULL (367 lines). Confirms depth.md's summary exactly, with full per-seed tables. Key numbers already captured above. This is the canonical, single, formal G-DEPTH verdict document. Claim ledger (§7) lists 7 certified claims — all traced to JSON artifacts. Nothing new beyond what depth.md already gave, but this is THE authoritative artifact path for the ledger's "where the verdict lives" column for D1/D5/D8.

## *** A THIRD, MOST RECENT, STILL-UNRESOLVED DEPTH THREAD: F5/F5b/F5c "feedforward-depth pilot" (dated 2026-07-18 to 2026-07-20 = TODAY) ***
Source: `/home/ff235/dev/MLResearch/automl/docs/plans/capacity_programme/flexnn-core.md:155-292` (Task F5/F5c).
This is DIFFERENT from D1/D5/D8 above: those certified that a RECURRENT weight-shared block can serve
per-input depth. F5 asks a narrower, harder question: does a PLAIN FEEDFORWARD (untied-weights, non-recurrent)
deep network ALSO show genuine per-input depth benefit, or is the G-DEPTH result specific to weight-tying?
flexnn-core.md:180 explicitly calls the untied-flat arm **"the user's original claim."**

- **Design (F5a, 2026-07-18):** 2x2 grid on A5 word composition (L=10, involutions, reusing the certified
  substrate): {flat input vs per-step input} x {untied weights vs tied weights}, plus the certified
  wide-shallow control and the certified RecurrentComposer (tied+per-step = the already-passing D-strand
  arm, included as the MANDATORY positive control). Depth ladder d in {4,7,10}, width 64.
- **F5b — RAN 2026-07-18 -> RULED INVALID 2026-07-20 (TODAY), on FOUR independent grounds, all
  verified on disk:**
  1. **Positive control FAILED.** The certified RecurrentComposer arm (mandatory confirm run) was
     trustworthy on only 1/2 seeds: seed1 clean at 0.9257, but seed0 climbed to 0.830 by epoch 3000 then
     COLLAPSED to 0.097 (diverged=true). Spec requires >=2 trustworthy seeds. "The protocol did not
     reproduce a known-good result => nothing else in the battery is readable."
  2. Protocol-parity breach: no gradient clipping used (unlike depth_selection_toy.py which sets
     GRAD_CLIP_MAX_NORM=1.0 specifically because "L=10 needs clipping to stay GD-trainable"); single
     unswept LR=1e-2, no schedule, across arms spanning 12,476-89,660 params.
  3. Gate/bar metric mismatch: trustworthy computed on val CE while bars read val accuracy; val CE
     explodes (overconfidence) while val accuracy stays flat -- decoupled signals.
  4. Early-stop-off confirmation (required at >=4x budget) never run.
  - Artifacts (INVALID, not usable for any bar): `D_TOY_PROBES/ff_depth_pilot_a5_seed{0,1}.json` (28 runs).
- **F5c — protocol repair, staged, GATES each stage. Status as of 2026-07-20 (TODAY): STILL FAILING,
  ESCALATED TO USER, UNRESOLVED.**
  - F5c-b (positive control alone, repaired protocol: lr=3e-3, clip_max_norm=1.0, dual gate) —
    **RESULT: FAIL, BOTH SEEDS.** seed0 val_acc=0.4324, seed1 val_acc=0.7442 (bar is >=0.90). BOTH
    runs converged cleanly (not hit_cap, not diverged on accuracy) yet held-out CE diverges
    monotonically after its minimum on both seeds (ce_gate.diverged=true both) -- this is presented as
    validating the NEW dual-gate mechanism (best-CE-weight restore would have picked a WORSE checkpoint
    than the run's own best -- exactly the defect that invalidated F5b).
  - Diagnosis: "failure mode is memorization-without-generalization, NOT under-fitting" -- train_acc
    0.97/1.00 (fully fit) while held-out fails -- so the standard optimization remedies (LR/warmup/init,
    "the ladder") cannot fix it, because there's no under-fitting left to fix. Also notes the two seeds
    differ by 31pp (0.432 vs 0.744) purely from initialization -- flagged as evidence the PROTOCOL/SUBSTRATE
    combination is fragile, not merely mistuned.
  - **"ESCALATED TO USER — do NOT resolve this without a ruling."** The spec mechanically prescribes
    continuing the optimization-remedy ladder (L1/L2/L3), but the evidence suggests the informative next
    experiment is a DIFFERENT one (M6 discriminator: does the ALREADY-certified anytime/selection
    configuration still reproduce its own numbers under this repaired protocol? — reproduces => single-exit
    supervision genuinely can't do this narrower task; doesn't reproduce => environmental regression and
    NOTHING may be claimed from any of today's F5c runs). Running M6 before exhausting L1-L3 is a spec
    deviation requiring a ruling. **F5c-c (gradient-norm diagnosis) and F5c-d (2x2 grid re-run + verdict)
    both remain HALTED pending this decision.**
  - Artifacts: `D_TOY_PROBES/f5c_poscontrol_a5_seed{0,1}.json` — THESE ARE THE TWO UNTRACKED "?? " FILES
    VISIBLE IN THE CURRENT GIT STATUS (git status showed `f5c_poscontrol_a5_seed0.json` /
    `f5c_poscontrol_a5_seed1.json` as untracked) — CONFIRMS this is genuinely live, uncommitted,
    in-progress work as of right now, not settled history.
- **NET STATE: whether a plain feedforward (non-recurrent, untied-weight) deep network shows genuine
  per-input depth benefit is COMPLETELY OPEN. Not "no", not "yes" — the measurement apparatus itself has
  not yet successfully reproduced its own certified positive control under ANY tested protocol variant, so
  no claim (positive or negative) about the feedforward/untied case can be made yet.** This is the single
  most important "currently open" item for the ledger and directly matches the archaeology brief's request
  to flag toys "designed but never run" / conclusions not yet reached — this one is DESIGNED, PARTIALLY RUN
  TWICE, BOTH RUNS INVALIDATED, and AWAITING A USER RULING, not concluded either way.

## STILL TO READ
- RESULTS.md lines 548-738 (rest of file, likely more per-input selector program T2/T1/H1 sections, possibly more depth mentions)
- R3_verdict.md, R1/R2/R4 verdict files (R3 is depth's own verdict; others may reference depth in cross-lane discussion)
- NOV1_novelty.md
- T1/ dir (PATH2 depth-learnability finding mentioned at RESULTS.md:22-23 "G-FORK -> Path 2 RUN -> NOT_FOUND_UNLEARNABLE on all 3 configs -> path-3-as-fallback outcome (T1 finding stands; certified T1/T1_PATH2_ADJUDICATION.md)" — T1 = "representable-not-learnable" per REPORT-2 section list at line 279. NEED TO CHECK: is T1 a DEPTH task or a general per-input-selector task? Line 279 lists "T1 representable-not-learnable" among S1/S2/H1/T2/T3/P1 — need dedicated read of T1/ dir.
- F1/, F2/, F3/ subdirs PREREGISTRATION.md/ADJUDICATION.md
- D_TOY_PROBES/, J_TOY_PROBES/ dirs (untracked new files per git status: f5c_poscontrol_a5_seed0/1.json — recent, un-adjudicated in RESULTS.md yet? NEED TO CHECK if these are depth-related)
- W_CASCADE/ dir — NAME MATCH for "cascade" mentioned in headline question! MUST INVESTIGATE — likely width lane but check if depth-relevant or if this IS the width->depth port.
- docs/plans/capacity_programme/depth.md
- docs/plans/capacity_programme/flexnn-core.md (depth sections)
- docs/plans/capacity_programme/width-depth.md
- docs/depth_capacity/ (all files) — note ?? docs/depth_capacity/ff_depth_protocol_repair_spec.md is UNTRACKED/NEW per git status, recent work
- automl_package/examples/depth_composition_toy.py (MODIFIED per git status) + grep other depth toy generators
- automl_package/examples/_toy_datasets.py
- git log searches for headline phrases
- CHANGELOG.md, RESUME.md
- project_depth_revisit_with_width_ideas memory note referenced this exact question — need to trace where it landed
