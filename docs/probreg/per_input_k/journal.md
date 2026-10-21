# Per-input-k ProbReg — journal

A factual record of the per-input-k variant: a small router, distilled from held-out per-k
scores, picks k for each input. It covers what the variant is, how it is built, the routes tried
(including the retired in-training ones), the toy results with their caveats, and what is already
planned. Assembled 2026-10-21 from the plans, the results on disk and the code at commit `5fc982e`.
It states facts and the conclusions the sources drew, each attributed; it gives no verdicts. The plan
of record is `docs/probreg/per_input_k/plan.md` under `docs/probreg/PLAN.md` (§1 holds the
definitions); earlier router plans are in `docs/probreg/per_input_k/history/`.

Paths are relative to the repository root (`/home/ff235/dev/MLResearch/automl` on the DGX,
`/mnt/dgx/home/ff235/dev/MLResearch/automl` from the laptop). Line numbers are as of `5fc982e`.
The base model is described in `docs/probreg/fixed_k/journal.md` §2 and the nested network this
variant reads in `docs/probreg/global_k/journal.md` §2.1–2.2.

**Labels used below** (defined in `docs/probreg/global_k/journal.md`): **OLD-SCHEDULE** (nested
training with a per-sample random k draw; every nested result to date), **LEARNED-σ** (the plans'
"old-objective"), **SURROGATE** (produced on the stand-alone `NestedKSurrogate` or the June
variational-EM mixture, whose per-component heads read x and learn a variance — not on the library
model, whose heads read only their class probability), **RETIRED** (Decision 29), **VOID**.

---

## 1. What the variant is

**M2 = ProbReg + distillation** (`docs/probreg/PLAN.md` §1): the same nested network M1 reads, plus
a small router trained afterwards to choose k for each input from held-out per-k errors; prediction
routes each input to its k. M1 and M2 are read off the same trained network, so "training is NOT a
variable between them; they differ in exactly one thing, how k is chosen". M2 is scored and costed
as a complete system, the router included.

M2's tolerance rule is different from M1/M3's on purpose: the router labels each row independently
at a flat relative margin (0.25), because "a per-input labelling decision has one row's worth of
evidence, and no standard error is estimable from a single observation"; the plan concludes "M2's
chosen k values are NOT directly comparable to M1's or M3's on tolerance grounds"
(`docs/probreg/PLAN.md` §1).

Objectives (owner rulings of 2026-10-20, `docs/probreg/PLAN.md`): **O3a** recover a known
per-input k\*(x) — rising, rising-then-falling, and a staircase that stops at 3; **O3b** never lose
to global-k and pay where structure varies (its numeric bars came from the real-data spec and are
superseded by the no-thresholds ruling); **O3c** route smooth data to the bypass rung, "a falsifiable
prediction, not a gate". O3a is flagged as a true-structure claim and O3b as a prediction claim
(ruling Q6, open). Per-input work is third in the owner's order (fixed k, then global k, then
per-input k; ruling Q1).

## 2. How it is built

### 2.1 The router (library)

| Part | What it is | Where |
|---|---|---|
| Entry point | `ProbabilisticRegressionModel.fit_router(x_val, y_val, *, sigma, capacity_grid=None, tolerance=0.25)`; requires a fitted `NESTED` model; stores `capacity_router_` | `automl_package/models/probabilistic_regression.py:1105` |
| Error table | for every rung k: minus the per-sample fixed-σ mixture log-likelihood of `mixture_components_at_rung(x, k)` at the caller's σ; y symlog-transformed if the model was fitted in symlog | `automl_package/models/probabilistic_regression.py:1170-1188` |
| Labels | per row, the first (cheapest) column with `error <= (1 + tolerance) * row_min` | `automl_package/models/flexnn/routing.py:95` (`_cheapest_within_tolerance_labels`) |
| Router | MLP, hidden (32, 32), ReLU, 300 full-batch Adam epochs, lr 1e-2, cross-entropy on the hard labels, seed 0 | `automl_package/models/flexnn/routing.py:77-80` |
| Prediction | `predict` routes each input to its k under `CapacitySelection.PER_INPUT` | `docs/probreg/review/groundwork/k_scoring_code_paths.md` (local-only), rows A3 and following <!-- citecheck-ignore: local-only review pack --> |
| Also available | `fit_soft()` (soft targets supplied by the caller) and `blend_scores()`/`blend_nll()`; the July soft-target construction stayed in the drivers | `automl_package/models/flexnn/routing.py` module docstring |

Facts about the router as built (from the code and the fact sheet
`docs/probreg/review/groundwork/k_scoring_code_paths.md`, row A3): <!-- citecheck-ignore: local-only review pack -->
- The router is labelled **and** trained on the same `(x_val, y_val)`; there is no further held-out
  split inside `DistilledCapacityRouter.fit`.
- The labelling rule is a **relative** margin on a per-sample negative log-likelihood. Where a row's
  smallest error is negative (a log-likelihood above zero), `1.25 × min` is below the minimum, no
  column qualifies, and the rule returns column 0 — the cheapest rung. The library's label tests use
  non-negative tables only (`tests/test_distilled_router.py`).
- The router constants were "copied from an earlier width experiment, not chosen for ProbReg"
  (`docs/probreg/per_input_k/plan.md`, PC); the 0.25 tolerance traces to
  `automl_package/examples/sinc_width_experiment.py` (`routing.py` module docstring).
- The hard-label path is not the July recipe of record: S1 certified **soft**, prior-dominant
  targets (§3.3), and the module docstring records that soft-target construction "stays with the
  drivers".
- **No example driver calls `fit_router`, `fit_global_selector`, `fit_sweep_selector` or
  `held_out_arbiter_advantage` on the library model**; they are exercised only by
  `tests/test_phase3_dynamic_k.py` (fact sheet, scope note). Every per-input result below was
  produced by driver code, mostly on the surrogate.

### 2.2 The per-input diagnostic (the "arbiter")

`held_out_arbiter_advantage(x, y, *, sigma, width=0.075)`
(`automl_package/models/probabilistic_regression.py:1436`) returns, per input, the neighbour average
(a box of half-width 0.075 around x) of the held-out log-likelihood of the top rung minus that of the
bypass rung. It chooses nothing; it is the readout the July K-lane certified as faithful (§3.3). It
scores at a fixed σ through `_per_sample_log_likelihood_at_k` (`:1400`), which symlog-transforms y
when needed (defect D2, fixed 2026-07-20, `docs/probreg/shared/state.md` §3).

### 2.3 Training of the network the router reads

The router reads the nested network of M1: per-sample random k draw (OLD-SCHEDULE), collapsed
Gaussian per rung with learned variance in the training path, `REGRESSION_ONLY`, and `NESTED`
requires a learned variance (`docs/probreg/global_k/journal.md` §2.1).

### 2.4 The surrogate the evidence was produced on

`NestedKSurrogate` (`automl_package/examples/_capacity_ladder_nested.py:38`): a weight network over
k_max components and one small network per component that reads x and outputs a mean and a clamped
log-variance (`:90-92`); trained on the renormalised mixture NLL over the first k components with k
drawn per sample; component means initialised at y-percentiles ("an INIT ONLY", `:72-80`). Its
docstring cites AutoCast's rank-ladder design rules. The library port is `NestedStrategy`
(`automl_package/models/flexnn/strategies/n_classes.py`), whose class docstring claims ProbReg "gets
K4's nesting guarantee for free" — the port shares the masked-prefix softmax, but differs in head
inputs (p_i only) and in the training readout (collapsed Gaussian rather than mixture).

## 3. Routes investigated

### 3.1 In-training per-input selection (April–July 2026) — RETIRED

- **April:** soft gating and Gumbel selectors trained jointly with the model "cap-track" — the
  expected k follows (k_max+2)/2 regardless of the data
  (`docs/probreg/history/kselection_findings_2026-04/probreg_kselection_findings.md`; details in
  `docs/probreg/global_k/journal.md` §3.1). The note found the **bypass head** was the part that
  responded to noise ("bypass mass collapses 0.75 → 0.08 as σ rises 0.1 → 2.0"), per input.
- **Report (a) (16 July):** its "adaptive" arm was ELBO + soft gating with a bypass and k_max 12,
  scored against trees (`docs/probreg/fixed_k/reports/report_a/report_a_probreg_toys.md` §2.3).
- **H1 arm (a) (July):** the jointly trained `SOFT_GATING` + `ELBO` "shipping" gate, on the library
  model; it "stays near its prior (arm-a marginal_p ≈ [0.51 bypass, 0.10×5] every seed, barely
  selecting)" (`automl_package/examples/capacity_ladder_results/RESULTS.md:626-630`).
- Labels: LEARNED-σ, RETIRED (Decision 29; reachable only with `allow_retired_capacity_selection=True`).

### 3.2 June 2026 — per-input weights, a prior, and held-out reading (SURROGATE)

From `docs/probreg/history/kselection_variational_em_2026-06/kselection_variational_em.md`:
- **§12:** per-input weights with a prefer-few (Dirichlet) cost charged **per input** collapsed — the
  fitted weights came out "essentially the same for every input"; the note measured the cost
  outweighing the fit "by roughly three to one" and stated this is "not a missing normalising
  constant" but the correctly scaled bound for that model.
- **§13:** one Dirichlet prior (α₀ = 0.1) on the **dataset-average** usage, charged once, "turns the
  dial": on the rising-spacing toy the effective class count rose from 1.0 to 1.9 across the known
  two-peak boundary and the held-out advantage turned positive there; on the single-peak twin the
  count drifted up to about 1.5 while the held-out advantage stayed within a hundredth of a nat of zero
  — "the count alone can mislead, and the held-out advantage is the honest arbiter".
- **§14:** a held-out, neighbour-averaged advantage of the mixture over one Gaussian recovered the
  count at probe inputs on all three families (rising spacing; staircase 1→2→3; hump 1→2→1).
- **§15, prior on trial:** with movable classes, prior on (α₀ = 0.1), prior off (α₀ = 1) and a plain
  mixture density network read "indistinguishable"; with classes pinned, the prior "actively breaks
  the reading" (negative advantage at the two-peak middle of the hump).
- **§16:** "The contribution is the reading, not the prior"; recommendation, "for the reader to
  weigh", to drop the prior; open: novelty against the literature, and real many-dimensional data.
- The owner's 2026-07-03 framing note counted the June work as evidence for this model ("a faithful
  surrogate of the bottleneck design") and recorded the aggregate-prior result as "the one in-fit
  result recorded as working", with the prior-drop recommendation pending
  (`docs/probreg/history/capacity_ladder_2026-07/plans/handover/REVIEW_HANDOVER_2026-07-03.md`, FRAMING 2–4). <!-- citecheck-ignore: local-only handover note -->
- Labels: SURROGATE (mixture-scored, heads read x), LEARNED-σ. The aggregate Dirichlet prior is a
  training-time term; under Decision 29 nothing may choose or shape k during training.

### 3.3 July 2026 — the nested ladder, the arbiter and the distilled router (SURROGATE unless stated)

All on toys C (single step, k\* 1→2), D (staircase 1→2→3), E (moving mode 1→2→1) and the
variance-matched single-mode twins C_broad and E_broad; three seeds; ledger
`automl_package/examples/capacity_ladder_results/RESULTS.md`. Labels on every K/S/T row:
SURROGATE, OLD-SCHEDULE, LEARNED-σ.

| Task | Question | What the ledger records | Source |
|---|---|---|---|
| K5 arbiter | per-input count from one nested model | toy D region means rise first < second < third on 3 of 3 seeds; toy E recovers on 1 of 3 ("a genuine negative"); broad twins max positive arbiter 0.0164 nats; positive claim "scoped to fixed modes" | `RESULTS.md:48-91`; `automl_package/examples/capacity_ladder_results/K5/k5_summary.json` |
| K5 per-input knee | the alternative reader (where the per-input curve stops climbing) | "unfaithful": toy D modal knee {1,2,4}, {1,1,1}, {1,1,2} across seeds | `RESULTS.md:59-61` |
| Nesting cost | nested rung vs dedicated model at the same k | worse by more than 2 SE somewhere in 8 of 9 structured cases, at middle rungs, never the top | `RESULTS.md:84-88` |
| K6 router | distil the per-input read into a router | soft (responsibility) router ≤ the best global k on 9 of 9; soft ≤ knee-labelled router on 7 of 9; raw per-point argmax worse than soft on 8 of 9 | `RESULTS.md:93-108`; `automl_package/examples/capacity_ladder_results/K6/k6_summary.json` |
| S1 target factorial | which training target for the router | soft targets win; "the per-tercile PRIOR is the load-bearing mechanism; neighbour smoothing adds essentially nothing"; two pre-registered bars failed and were adjudicated benign with corrected wording logged beside the originals | `RESULTS.md:378-420`; `automl_package/examples/capacity_ladder_results/S1/s1_summary.json` |
| S2 direct objective | train the router on held-out likelihood directly | ties soft (difference 0.36 SE), does not beat it; recipe of record stays soft | `RESULTS.md:422-467`; `automl_package/examples/capacity_ladder_results/S2/s2_summary.json` |
| T2 more input dimensions | does the read survive nuisance dimensions (kNN neighbourhoods) | toy D in 2-D: advantage over global on 3 of 3 seeds at 6–9 SE, broad twin no advantage; null by 5-D and 10-D; a rotated 5-D test is "VOID" (the rotated toy had k\*=1 everywhere) | `RESULTS.md:557-608`; `automl_package/examples/capacity_ladder_results/T2/t2_summary.json` |
| X4, X4b non-nested arm on toy E | is the toy-E failure specific to nesting | recovers 1 of 3 seeds; after 8 restarts the seed-1 collapse is repaired but no hump appears; "W8 (E-fragility nesting-specific) NOT supported" | `RESULTS.md:335-360`; `automl_package/examples/capacity_ladder_results/X4b/x4b_summary.json` |
| T3 power curve on toy E | was toy E under-powered | non-nested multi-restart instrument at N = 1000, 4000, 16000: gold-standard recovery 2/3, 3/3, 3/3; arbiter-style read 1/3, 3/3, 3/3; control flat at every N; binding condition: "do not read as overturning X4b's fragility finding" | `RESULTS.md:469-510`; `automl_package/examples/capacity_ladder_results/T3/t3_summary.json` |
| H1 two-phase recipe on the **library model** | freeze the nested ProbReg, then distil a gate on a held-out half | beats the jointly trained gate on toy D on 3 of 3 seeds by 0.08–0.10 nats; matches a standalone router (≤ 0.0036 nats, 9 of 9); on C_broad never worse than the best fixed k; heads trained on half the data of the other arms; k=1 asymmetry noted (§2 of `docs/probreg/global_k/journal.md`) | `RESULTS.md:610-664`; `automl_package/examples/capacity_ladder_results/H1/h1_summary.json` |

Further facts the sources attach:
- **RESULTS.md is not internally consistent across its own updates.** Its status table (line 12)
  says the moving mode is "recoverable-with-more-data (T3…)", while the WS1 section it heads still
  says toy E "does NOT recover for 2/3 seeds" and calls the failure "plausibly NESTING-SPECIFIC"
  (`automl_package/examples/capacity_ladder_results/RESULTS.md:60-70`), which X4/X4b later recorded as
  "NOT supported" (`:343-345`, `:353-355`). Read each claim at its own task section, with its date.
- **K6 pre-registered** router agreement with the K5 reads of at least 90% within ±1 k on interior
  points (`docs/probreg/history/capacity_ladder_2026-07/plans/EXECUTION_PLAN.md:343-346`); the ledger
  reports agreement "low/variable (0.03–0.85)" and explains it by the knee's own noise
  (`RESULTS.md:104-105`).
- **Repeated draws at one x** are available for toys C, C_broad, E, E_broad, D and D_ndim through
  `sample_toy_*_given_x` helpers in `automl_package/examples/_toy_datasets.py`; the June comparison,
  X4/X4b and T3 used them for their "gold" reads.
- **Data roles in S1/S2/K6:** targets from even-indexed held-out rows, scoring on odd rows; the
  "honest oracle" routes one half from a disjoint half (`RESULTS.md:411-414`, `:457-461`). The July
  report notes that every selector result scores against one fixed even/odd split and that the
  split-robustness check "has not yet been run for the class-count selector"
  (`docs/probreg/shared/reports/probreg_kselection/probreg_kselection.md` §5); the depth lane's
  equivalent result did not survive re-splitting (X3, `RESULTS.md:306-312`).
- **The soft target's "prior"** is a per-tercile stacking weight fitted on held-out scores at the
  selector stage, not a training-time prior (`RESULTS.md:411-414`). Its terciles are fixed regions of
  the 1-D input; T2 records that the 1-D prior "cannot legitimately transfer to dim>1 without leaking
  the analytic index" and used kNN-smoothed targets without it (`RESULTS.md:594-597`).
- **T3 and X4/X4b used the June non-nested instrument**, whose objective includes a Dirichlet-usage
  term with coefficient 1 (`RESULTS.md:500-503`); the sources do not state how σ is treated there.
- **H1's arm (b) phase 1** is the library model with the per-sample k draw (OLD-SCHEDULE) and the
  gate quiescent; the ledger does not state whether the variance was learned (by date, before
  Decision 26).
- The July report (`docs/probreg/shared/reports/probreg_kselection/probreg_kselection.md`,
  18 July 2026) is the write-up of K4–K6, S1 and the moving-mode follow-up; the plan marks it as
  predating the 2026-07-20 definitions, "must not be extended until §1 is reflected in the code and
  the battery" (`docs/probreg/shared/state.md` §2). Its §1.1 describes per-class sub-models that read
  x and output a variance — the surrogate, not the library heads.

### 3.4 Defects found along the way

- **D1 (fixed 2026-07-20):** `fit_router` scored raw-space y against symlog-space outputs — "silently
  wrong error table ⇒ silently wrong selector, worst on exactly the heavy-tailed data this strand
  targets"; verified both directions with tests (`docs/probreg/shared/state.md` §3).
- **D2 (fixed):** the same units mismatch in `_per_sample_log_likelihood_at_k`.
- **D5:** `held_out_arbiter_advantage` is a per-input top-vs-bypass comparison and cannot choose a
  middle k (`docs/probreg/shared/state.md` §3).

## 4. Toy results on record, with caveats

| Claim (objective) | Strongest evidence on record | Labels |
|---|---|---|
| Recovers a rising per-input k\* (O3a) | K5 arbiter on toy D, 3 of 3 seeds | SURROGATE, OLD-SCHEDULE, LEARNED-σ |
| Recovers a rising-then-falling k\* (O3a) | 1 of 3 at N=1000 on the nested arbiter; 3 of 3 at N=4000 only with the non-nested multi-restart instrument (T3) | SURROGATE, LEARNED-σ |
| Stops at 3 rather than running to k_max (O3a) | toy D's top band reads above its middle band; no direct read of "stops at 3" is recorded (the per-input knee, which would give a count, was ruled unfaithful) | SURROGATE, OLD-SCHEDULE, LEARNED-σ |
| Never loses to global k (O3b) | K6 soft router ≤ best global k on 9 of 9 held-out cases; one fixed even/odd split | SURROGATE, OLD-SCHEDULE, LEARNED-σ |
| Routes smooth data to the bypass (O3c) | broad twins: arbiter ≤ 0.017 nats; H1 on C_broad never worse than the best fixed k | SURROGATE / library (H1), OLD-SCHEDULE |
| Survives more input dimensions | yes at 2-D, null by 5-D | SURROGATE, OLD-SCHEDULE, LEARNED-σ |
| The library `fit_router` (hard labels, 0.25 margin, fixed σ) recovers anything on a toy | **no run on disk** | — |
| How much selection data M2 needs | not studied (PB) | — |
| Whether the router architecture matters for ProbReg | not studied (PC); width's half/double check found its deploy claims invariant on 3 seeds (`docs/flexnn/width/history/width-cert.md:234-237`) | width strand |

## 5. Planned next steps, as the plans state them

| Task | What it asks | State |
|---|---|---|
| PB (`docs/probreg/per_input_k/plan.md`) | sweep the selection fraction {5, 10, 15, 25, 40}% of the training portion on toy tiers 1 + 3, for all three arms; find where each saturates; state M2's data floor; emit a frozen-constants file (PB/frozen.json, not yet created) | not run; deps P1, PA, P7 |
| PC | vary router width/depth (half, double, 4×; 1 vs 3 layers) and epochs on the toys; the 0.25 tolerance is struck from the sweep (Decision 18); router stays frozen and untuned inside any benchmark; emit a frozen-constants file (PC/frozen.json, not yet created) | not run; deps P1, PA, P7 |
| P7 | the all-rungs training schedule | not landed |
| P3 | the M1 ≈ M3 comparison (tiers 1 + 2 + 3), which also gives M2's read on the same networks | not run |
| P4, P6 | real data and baselines; the report | out of scope (toys only); P6 parked |
| Structure-study wall | no k-selection task before the fixed-k certification gate | `docs/probreg/fixed_k/plan.md` §1 |

Per-input claims need toy tier 3 (`make_toy_c`, `make_toy_e`, `make_toy_d`) as well as tier 1
(`docs/probreg/shared/toy_suite.md`). The July report lists one direction as open: input spaces where
the right number of classes "is not a small fixed integer at all, or where the true structure is not
built from cleanly separated peaks" (`probreg_kselection.md` §7).

## 6. Sources

Plans: `docs/probreg/PLAN.md` (§1, Decision 29), `docs/probreg/per_input_k/plan.md`,
`docs/probreg/per_input_k/history/perinput_selector_2026-07/EXECUTION_PLAN.md` (frozen),
`docs/probreg/shared/state.md`, `docs/probreg/shared/toy_suite.md`. Reports:
`docs/probreg/shared/reports/probreg_kselection/probreg_kselection.md`,
`docs/probreg/history/kselection_variational_em_2026-06/kselection_variational_em.md`,
`docs/probreg/history/kselection_findings_2026-04/probreg_kselection_findings.md`,
`docs/probreg/history/capacity_ladder_2026-07/report/capacity_ladder_report.md`. Results:
`automl_package/examples/capacity_ladder_results/RESULTS.md` and `K4/`, `K5/`, `K6/`, `S1/`, `S2/`,
`T2/`, `T3/`, `X4/`, `X4b/`, `H1/`. Code: `automl_package/models/probabilistic_regression.py`
(`fit_router`, `held_out_arbiter_advantage`), `automl_package/models/flexnn/routing.py`,
`automl_package/examples/_capacity_ladder_nested.py`, `automl_package/examples/capacity_ladder_k6.py`,
`automl_package/examples/capacity_ladder_s1.py`, `automl_package/examples/capacity_ladder_h1.py`.
