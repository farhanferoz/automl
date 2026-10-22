> Superseded framing (2026-10-22): text below that treats ProbReg as a mixture model (fixed-σ mixture loss/readout, regression-only supervision) is withdrawn; see the banner in docs/probreg/PLAN.md.

# Global-k ProbReg — journal

A factual record of the global-k variant: one k for the whole dataset, chosen from held-out data.
It covers what the variant is, how it is built, the selection routes tried (including the retired
in-training ones), the toy results with their caveats, and what is already planned. Assembled
2026-10-21 from the plans, the results on disk and the code at commit `5fc982e`. It states facts
and the conclusions the sources drew, each attributed; it gives no verdicts. The plan of record is
`docs/probreg/global_k/plan.md` under `docs/probreg/PLAN.md` (§1 holds the definitions).

Paths are relative to the repository root (`/home/ff235/dev/MLResearch/automl` on the DGX,
`/mnt/dgx/home/ff235/dev/MLResearch/automl` from the laptop). Line numbers are as of `5fc982e`.
The model the variant is built on is described in `docs/probreg/history/journals_2026-10-21/fixed_k_journal.md` §2.

**Labels used below:**
- **OLD-SCHEDULE** — nested training with a per-sample random k draw. Decision 20 (2026-07-21,
  `docs/RULES.md`) retired that draw as the default schedule in favour of training every rung each
  step (task P7). P7 has not landed, so **every nested result to date is old-schedule**.
- **LEARNED-σ** (the plans' "old-objective") — trained with fitted variances (Decision 26).
- **SELECTED ON LEARNED σ** — k chosen by a likelihood read at a fitted variance (Decision 24
  forbids this).
- **SURROGATE** — produced on the stand-alone `NestedKSurrogate` network
  (`automl_package/examples/_capacity_ladder_nested.py:38`) or the June variational-EM mixture,
  not on the library `ProbabilisticRegressionModel`. The surrogate's per-component heads read x
  and output a mean and a log-variance (`automl_package/examples/_capacity_ladder_nested.py:90-92`); the library's heads read only their class
  probability (`docs/probreg/history/journals_2026-10-21/fixed_k_journal.md` §2.1).
- **RETIRED** — in-training k selection or cross-entropy modes (Decision 29).
- **VOID** — results the plan has discarded.

---

## 1. What the variant is

Global-k ProbReg chooses **one k for the whole dataset from held-out data**. Two ways of doing it
are defined (`docs/probreg/PLAN.md` §1):

| Name | = ProbReg + | How k is chosen | Cost |
|---|---|---|---|
| **M1** | the arbiter | train ONE nested network that works at every k; read its held-out score at each k; pick one k | cheap |
| **M3** | the sweep selector | train a separate ordinary model at each k (no k-dropout, `NClassesSelectionMethod.NONE`); score each on held-out data; pick one k | expensive — the reference |

Rules the plan attaches (`docs/probreg/PLAN.md` §1 and Decisions 24–25):
- Each model is scored **as a complete system, selection machinery and cost included**.
- **Two claims, tested separately:** (a) *same quality* — read at a given k, the nested model is
  as good as a model dedicated to that k; (b) *same choice* — M1 picks the k M3 picks.
- The readout is the per-sample **fixed-σ mixture log-likelihood**; squared error is a reporting
  column only ("structurally blind to k for symmetric targets"); a likelihood at a learned
  variance is forbidden (Decision 24).
- M3's candidate set spans the same rungs as the ladder, bypass included (Decision 25), and its
  cost is reported.
- Selection rule for both: the smallest k whose held-out score is not worse than the best by more
  than **twice a bootstrap standard error** (the "cheapest within tolerance" rule). The review terms
  of reference treat this two-SE rule as part of the method, not an evaluation threshold.

Owner rulings of 2026-10-20 that bind it (`docs/probreg/PLAN.md` rulings table): k is a
hyperparameter, never learned with the regression means during fitting, and "any route by which the
training objective chooses k (including early stopping or checkpoint choice on a loss averaged over
k) violates this"; ruling Q6 (is k claimed *useful* or *true*) is open. Objectives O2a (same
quality), O2b (same choice) and O2c (the chosen k is right where the truth is known) — the plan
flags O2c as a different kind of claim from O2a/O2b.

## 2. How it is built

### 2.1 The nested ladder (M1's network)

- `n_classes_selection_method=NESTED` builds `NestedStrategy`
  (`automl_package/models/flexnn/strategies/n_classes.py`); rung 1 is the **direct-regression head**
  (it reads x; built only for dynamic-k methods,
  `automl_package/models/architectures/probabilistic_regression_net.py:66-97`); rungs 2..k_max use
  a softmax over the first k classifier logits only (masked prefix), feeding the same shared heads.
- **Training schedule today:** `NestedStrategy.forward` "draws a per-sample rung ~ Uniform{1..k_max};
  returns that rung's own readout" (`automl_package/models/flexnn/strategies/n_classes.py:328`).
  The 2026-07-21 ruling (Decision 20, task P7) to train every rung each step has not landed.
- **Training loss today:** each rung's readout in the training path is the collapsed
  (law-of-total-variance) Gaussian with a learned log-variance — `all_rung_outputs` is "still used by
  `forward()`, the TRAINING path, untouched by FP-12" (`n_classes.py`, `all_rung_mixture_components`
  docstring). The class docstring states that MDN loss, the ordering penalty, the middle-class
  penalty and boundary regularisation "all silently no-op under NESTED". Construction **requires**
  `uncertainty_method=PROBABILISTIC` under `NESTED`
  (`automl_package/models/probabilistic_regression.py:208-213`), so a nested network cannot
  currently be trained at fixed σ.
- The July surrogate that produced the k-selection evidence (§4.3) trained a different objective:
  the negative log-likelihood of the renormalised k-component **mixture** over the first k
  components, with learned per-component variances (`_capacity_ladder_nested.py`,
  `masked_prefix_nll`).
- **Early stopping and prediction draw rungs at random too** (read from code, not run). The library
  loop early-stops on the full training loss on the validation split
  (`automl_package/models/base_pytorch.py:100` onward); under `NESTED` that loss passes through the
  same `forward`, so it is "a single stochastic draw of rungs per row … not computed at k_max and not
  averaged over rungs" (`docs/probreg/review/groundwork/data_roles_defaults_stopping.md`, fact 4; <!-- citecheck-ignore: local-only review pack -->
  `automl_package/models/flexnn/strategies/n_classes.py:345-347`). `forward` has no training-mode
  check, so a `NESTED` model predicting under `capacity_selection=FIXED` also routes each input
  through a random rung (`automl_package/models/probabilistic_regression.py:841-879`).
- **Cross-entropy is frozen off:** `optimization_strategy=REGRESSION_ONLY` is required for the
  ladder to stay coherent, because cross-entropy re-cuts the bins at every k so one class would mean
  different target ranges at once (defect D4, "only avoided, not fixed",
  `docs/probreg/shared/state.md` §3).
- `SINGLE_HEAD_FINAL_OUTPUT` is refused under `NESTED`
  (`automl_package/models/probabilistic_regression.py:244-258`).

### 2.2 Scoring at every k

`NestedStrategy.all_rung_log_likelihood(x, y, sigma)` (`n_classes.py:293`) returns a
`(batch, k_max)` table of per-sample fixed-σ mixture log-likelihoods; rung 1 is the bypass treated
as a one-component mixture at the bypass head's mean. It does **not** symlog-transform y; the
caller must (`fit_global_selector` does, `automl_package/models/probabilistic_regression.py:1260`).
Before 2026-07-21 the same table was a collapsed Gaussian at a learned variance (commit `ef7e1f8`,
"score capacity on a fixed-sigma mixture").

### 2.3 The two selectors (task PA, landed)

| | M1 `fit_global_selector` | M3 `fit_sweep_selector` |
|---|---|---|
| Where | `automl_package/models/probabilistic_regression.py:1196` | `automl_package/models/probabilistic_regression.py:1284` |
| Requires | a fitted `NESTED` model | any model (it trains its own sub-models) |
| Candidate grid | default every rung 1..`max_n_classes_for_probabilistic_path` | the same default |
| What it scores | `all_rung_log_likelihood` on `(x_val, y_val)` at the caller's σ | each sub-model's `predict_distribution` weights and means at the caller's σ; the sub-model's own learned spread is ignored |
| Space of the curve | the space the model was fitted in (symlog if `target_transform="symlog"`) | raw target units; `predict_distribution` does not support symlog (docstring) — "the two curves are not numerically comparable rung-for-rung" |
| Rule | `cheapest_within_tolerance` (`automl_package/utils/capacity_selection.py:41`): 2× bootstrap SE of the paired difference, `n_boot=1000` | the same function |
| σ | `sigma` required, no default; must equal the value passed to the other selectors | same |
| Tests | `tests/test_phase3_dynamic_k.py`, classes `TestFitGlobalSelector`, `TestFitGlobalSelectorSymlogSpaceAlignment` | `TestFitSweepSelector` |

Facts about the pair that bear on comparing them:
- **k=1 is a different model on each side.** M1's rung 1 is the direct-regression head, which reads
  x. M3's k=1 sub-model is an ordinary one-class ProbReg whose single head is fed a class probability
  of 1 for every input, so it predicts the same value everywhere; the July H1 adjudication recorded
  this asymmetry for a fixed-k arm of the library model
  (`automl_package/examples/capacity_ladder_results/RESULTS.md:641-644`).
- M3's sub-models are built from the parent's parameters with only `n_classes`, the selection method
  and `capacity_selection` overridden; with library defaults they **train** with a learned variance
  and are **scored** at the fixed σ.
- How much data the selection step uses: `selection_fraction = 0.15` of `fit()`'s data is held out
  for the internal selection step (`automl_package/models/probabilistic_regression.py:144`); the
  plan records that 15% "was chosen because it looked reasonable, not because anything was measured"
  (`docs/probreg/per_input_k/plan.md`, PB).

### 2.4 The other reader in the code

`held_out_arbiter_advantage` (`automl_package/models/probabilistic_regression.py:1436`) compares
only the top rung against the bypass, per input, neighbour-averaged. The plan recorded that it is
"not M1's selection mechanism" (defect D5, `docs/probreg/shared/state.md` §3); it is the per-input
diagnostic of `docs/probreg/history/journals_2026-10-21/per_input_k_journal.md`.

## 3. Selection routes investigated

### 3.1 In-training k selection (April 2026) — RETIRED

- **Question:** does the library model, given a generous k_max and a learned per-input selector
  (`SOFT_GATING`, `GUMBEL_SOFTMAX`), settle on the intrinsic k?
- **Setting:** library model, learned σ, collapsed Gaussian; 60-run sweep plus a 12-run ELBO-prior
  check; k_max ∈ {5, 10, 20, 40}; April toys A (smooth) and B (K_true ∈ {2, 3, 5}); 3 seeds
  (`docs/probreg/history/kselection_findings_2026-04/probreg_kselection_findings.md`, Setup). Results
  `automl_package/examples/probreg_kselection_experiments_results/`,
  `automl_package/examples/probreg_elbo_prior_check_results/`.
- **The note found:** without a penalty, the expected k "tracks the uniform-distribution mean
  ((k_max+2)/2)" on both toys — "the model does the same thing whether K_true exists or not";
  `K_PENALTY` drove E[k] to about 3.3 on K_true=3; ELBO with the default uniform prior made the
  selector exactly uniform; a geometric prior (λ=0.5) gave E[k] ≈ 3.0, but with argmax at k=2, which
  the note called "a coincidence — the model isn't finding K_true"; the bypass head, not classifier
  confidence, was the part that responded to noise (`probreg_kselection_findings.md`, summary items
  1–5 and Q4). "No intrinsic-k phase exists to be captured by early stopping" (Q6).
- **Status:** the owner's 2026-07-03 framing banned `K_PENALTY`-class terms "even where they work
  numerically" (`docs/probreg/history/capacity_ladder_2026-07/plans/handover/REVIEW_HANDOVER_2026-07-03.md`, FRAMING 1); <!-- citecheck-ignore: local-only handover note -->
  Decision 29 (2026-07-21) made all in-training selection unreachable without the escape hatch. The
  note's banner marks it superseded. Labels: LEARNED-σ, RETIRED.

### 3.2 June 2026 — a prior on the weights, and held-out reading

- The June note (`docs/probreg/history/kselection_variational_em_2026-06/kselection_variational_em.md`)
  diagnosed why in-fit selection cannot work — "a finer split has more freedom, so it fits the
  training targets at least as well as a coarser one" — and that the then-default blend of per-k
  summaries is "not even connected to a quantity that could reward the right amount of detail"
  (§2). It developed a global prior route ("Basis A: one shared set of weights", §5) before moving
  to per-input weights (see `docs/probreg/history/journals_2026-10-21/per_input_k_journal.md` §3.2).
- **Its fixed-count comparison (§14, SURROGATE):** a mixture density network at a fixed number of
  components, with that number chosen by held-out tuning, matched the best achievable fit within
  about 0.02 nats on three toy families, but on the staircase family "held-out tuning cannot find a
  single right number: it picks six, three, and four on three different starts" — "the tuning is
  choosing among near-ties". (`kselection_variational_em.md` §14, table.)
- **Labels:** SURROGATE (a stand-alone mixture model and an MDN, not the library model), LEARNED-σ.

### 3.3 July 2026 — the nested ladder (K-lane), on the surrogate

- **K1–K3 (invalid instrument):** reading the first c classes off a classifier trained without
  nesting gave non-physical results — on single-mode control data a fabricated advantage of +24 nats
  (C_broad, seed 1) and +229 nats (E_broad, seed 2) per
  `automl_package/examples/capacity_ladder_results/R1_verdict.md:70-71`; the July report quotes the
  first figure as 42 nats (`docs/probreg/shared/reports/probreg_kselection/probreg_kselection.md` §2.1)
  — the two sources differ. The R1 verdict ruled these "an artifact of an invalid de-risk instrument" and K4 was
  built to fix it (`automl_package/examples/capacity_ladder_results/R1_verdict.md`;
  `docs/probreg/shared/reports/probreg_kselection/probreg_kselection.md` §2).
- **K4 (nested surrogate, per-sample k draw):** trained with the renormalised masked-prefix mixture
  NLL, k_max 6 (toys C, E and twins) or 8 (toy D), N_train 1000, N_test 2500, 800 epochs, seeds
  0–2, component means initialised at y-percentiles. Results
  `automl_package/examples/capacity_ladder_results/K4/k4_summary.json`.
- **Global knee on K4 (pooled held-out curve, stop when the next increment fails 2 SE):** toy D read
  r* = 2, 0, 0 across seeds; toys C and E read 0 on every seed; broad twins 0 on all 6 cases.
  RESULTS.md records that this fails R1's "decisive check" at face value and closed the gate by
  reading it as evidence that "the knee under-reads component count even at the aggregate level"
  (`automl_package/examples/capacity_ladder_results/RESULTS.md:71-79`). The mechanism given
  ("the knee walk stalls exactly at the middle rungs where the b_coh nesting costs fire") is labelled
  a hypothesis there.
- **Not recorded:** the K4 plan pre-registered an ordering-penalty ablation to be reported at the R2
  checkpoint (`docs/probreg/history/capacity_ladder_2026-07/plans/EXECUTION_PLAN.md:307-312`,
  `:360-362`); no outcome appears in RESULTS.md, R1 or R2.
- **Floors:** the true density's held-out NLL is stored for toys C, E and D only, from the June
  comparison (n_test 4000, three seeds), in
  `automl_package/examples/variational_em_comparison_results/results.json`; no floor is stored for
  toys A, B, the broad twins or `make_broad_unimodal` (`docs/probreg/review/groundwork/toys.md`, local-only). <!-- citecheck-ignore: local-only review pack -->
  For toys C, E and D the structure battery's σ_toy is the root-mean-square **total** standard
  deviation, wider than the true component σ of 0.3, so the true mixture itself scores lower at σ_toy
  than at its own σ (same sheet; computed there, not in a ledger).
- **Labels:** SURROGATE, OLD-SCHEDULE, LEARNED-σ. The global knee is the closest thing on record to
  M1's selector (a pooled held-out curve with a 2-SE rule) but it was read on the surrogate's mixture
  readout at learned σ, before `fit_global_selector` existed.

### 3.4 The fixed-k chooser of report (a)

Report (a) chose k per toy from {5, 8, 10, 12} by the lowest held-out NLL at the fitted variance,
then froze it — the closest thing on record to an M3 run on the library model; SELECTED ON LEARNED
σ, no bypass, LEGACY TOYS (details in `docs/probreg/history/journals_2026-10-21/fixed_k_journal.md` §3.5 and §4.2). Its chosen
values were 8, 10, 12, 10, 12 — "never the smallest option on the grid"
(`docs/probreg/fixed_k/reports/report_a/report_a_probreg_toys.md` §7).

### 3.5 P8 — does explicit regularisation move the selected k? — VOID

- Ran 2026-07-21: M3's ordinary per-k models at weight decay λ ∈ {0, 1e-4, 1e-2}, seeds {100, 101},
  heteroscedastic toy, with epoch-cap escalation; recorded `selection_moved: true`
  (`automl_package/examples/capacity_ladder_results/P8/frozen.json`).
- **Discarded the same day:** the selection was made on Gaussian NLL from the predicted mean and
  predicted standard deviation — SELECTED ON LEARNED σ. "Results produced in violation of a
  constraint are DISCARDED, not reinterpreted" (owner ruling quoted in
  `docs/probreg/global_k/plan.md`, P8). The files stay on disk and "may NOT be cited as evidence".
- What the plan recorded about the voided run's pattern, as context only: one seed selected the
  same k at every λ, the other moved non-monotonically and also moved when only the epoch budget
  changed; the plan read it as "closer to … a broken instrument than to a clean positive" and
  suggested seed count and selection-set size (task PB) as the live suspects.
- The re-run must change only the metric, to the fixed-σ mixture log-likelihood; not squared error,
  which "would have cleared the block it exists to enforce" (`docs/probreg/global_k/plan.md`, P8).

## 4. Toy results on record, with caveats

| Claim | What exists | Source | Labels |
|---|---|---|---|
| (a) same quality: nested read at k vs dedicated model at k | 9 cases (toys C, D, E × 3 seeds), k = 1..6: difference ranges −0.13 to +0.09 nats; some middle k worse by more than 2 SE in **8 of 9** cases; the top checked k never fails; toy D's ladder goes to 8 but the check stops at 6 | `automl_package/examples/capacity_ladder_results/RESULTS.md:84-88`; `docs/probreg/shared/reports/probreg_kselection/probreg_kselection.md` §3.2; `automl_package/examples/capacity_ladder_results/K4/k4_summary.json` | SURROGATE, OLD-SCHEDULE, LEARNED-σ |
| (b) same choice: M1's k vs M3's k | **never tested** ("Nothing anywhere compares the arbiter's chosen k against the sweep's best k") | `docs/probreg/shared/state.md` §2 | — |
| O2c: the chosen k is right where the truth is known | no run of `fit_global_selector` or `fit_sweep_selector` on any toy is on disk; the nearest evidence is the surrogate's global knee (§3.3) | `automl_package/examples/capacity_ladder_results/` (no `P3/`) | — |
| A single global k cannot represent input-varying structure | in the July selector study the best single fixed count was 1 in 7 of 9 cases and 2 in the other 2, never higher, including the staircase whose true count reaches 3 | `docs/probreg/shared/reports/probreg_kselection/probreg_kselection.md` §5, Table 4 | SURROGATE, OLD-SCHEDULE, LEARNED-σ |
| Held-out tuning of a fixed count is seed-incoherent on input-varying data | staircase: 6, 3, 4 on three starts | `kselection_variational_em.md` §14 | SURROGATE, LEARNED-σ |
| In-training selection does not find intrinsic k | cap tracking on toys A and B | `probreg_kselection_findings.md` | LEARNED-σ, RETIRED |
| Regularisation moves the selected k | — | P8 | VOID |

The plan's explanation for the middle-k cost is unproven: it names width's W1 diagnosis (the
per-sample uniform draw starves rungs, fixed in the width strand by the "sandwich" schedule) as the
prior-art candidate to falsify first, with P7's all-rungs schedule as the cheap discriminating run
(`docs/probreg/global_k/plan.md`, P5).

## 5. Planned next steps, as the plans state them

| Task | What it asks | State |
|---|---|---|
| P7 (`docs/probreg/shared/plan.md`) | migrate the nested training to the all-rungs-every-step schedule (Decision 20) and re-validate | not landed: `NestedStrategy.forward` still draws per sample |
| Decision 26 for the ladder | fixed σ in training | not possible today under `NESTED` (§2.1) |
| P3 (`docs/probreg/global_k/plan.md`) | M1 ≈ M3 on toys, both halves; M3's positive control first (P3/m3_control.json, not yet created, must show it reproduces §3.2's pattern before any M1 number is read); full held-out trajectories; `hit_cap: false` | not run; deps P1, PA, PB, P7, FP-12; the Decision 25 grid dependency is cleared |
| P5 | explain the middle-k coherence cost, after P7 | not run |
| P8 re-run | the voided check on the fixed-σ readout | not run; blocks the battery reads of P4/P6 |
| PB (`docs/probreg/per_input_k/plan.md`) | how much selection data each arm needs (5–40% of the training portion) | not run; P3 depends on it |
| P4, P6 | real data and baselines; the report | out of scope (toys only, ruling Q5); P6 parked |
| Frozen constants (`docs/probreg/shared/plan.md` §3.6) | the selection fraction and M2 data-limited flag (from PB), the router constants (from PC), k_max after any ceiling raise (from P3/P4); results at a value not justified by its artifact are "not reportable" | none of the four artifacts exists |
| FP-12 residuals (`docs/flexnn/package/plan.md`, FP-12 header) | owner sign-off on the σ constant for the two heteroscedastic toys; σ×0.5 and σ×2 re-scoring, which the benchmark spec requires and which "is implemented NOWHERE" | open |
| Structure-study wall | no k-selection task proceeds before the fixed-k certification (PS-4) gate | `docs/probreg/fixed_k/plan.md` §1 |

The toy tiers the plan assigns: P7 runs tiers 1 + 2 (`make_toy_b`; `make_toy_a`,
`make_broad_unimodal`, `make_toy_c_broad`); P3 runs tiers 1 + 2 + 3; the four legacy toys are not a
tier (`docs/probreg/shared/toy_suite.md`). Toy seeds 0, 1, 2 on every cell.

Decision 29's conditional deletion trigger ties to this variant: if the arbiter matches the sweep
and the distilled router works per input (P3 and P4 pass), all in-training k-selection machinery
becomes deletion-eligible — and the in-training comparison, if wanted in the report, must be run
before deletion (`docs/probreg/PLAN.md`, Decision 29).

## 6. Sources

Plans: `docs/probreg/PLAN.md` (§1, Decisions 24–26, 29), `docs/probreg/global_k/plan.md`,
`docs/probreg/shared/state.md`, `docs/probreg/shared/toy_suite.md`, `docs/RULES.md` (Decision 20).
Reports: `docs/probreg/shared/reports/probreg_kselection/probreg_kselection.md`,
`docs/probreg/history/kselection_findings_2026-04/probreg_kselection_findings.md`,
`docs/probreg/history/kselection_variational_em_2026-06/kselection_variational_em.md`,
`docs/probreg/history/capacity_ladder_2026-07/report/capacity_ladder_report.md`.
Results: `automl_package/examples/capacity_ladder_results/RESULTS.md`, `R1_verdict.md`,
`R2_verdict.md`, `K4/`, `K1K2K3/`, `P8/` (void). Code:
`automl_package/models/probabilistic_regression.py` (`fit_global_selector`, `fit_sweep_selector`),
`automl_package/models/flexnn/strategies/n_classes.py` (`NestedStrategy`),
`automl_package/utils/capacity_selection.py`, `automl_package/examples/_capacity_ladder_nested.py`.
