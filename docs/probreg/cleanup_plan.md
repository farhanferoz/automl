# ProbReg cleanup — remove the mixture-model framing (2026-10-22)

**Why.** Owner ruling 2026-10-22: ProbReg is a regression model regularised by classification, not a
mixture model (DECISIONS.md, top section; banner in `docs/probreg/PLAN.md`). The mixture framing entered
in stages: the default moved from composite loss to regression-only on 2025-09-24 (commit `710d22a`);
on 2026-07-21 fixed σ (Decision 26), the fixed-σ mixture loss and readout (commit `ef7e1f8`) and the
retirement of every cross-entropy mode (Decision 29) followed; the 2026-10-21 review built on all three.
This plan removes it from the library, tests, drivers, plans and the review. A new review follows.

## Owner choices (DECIDED 2026-10-22: O1 (c), O2 (a), O3 (a), O4 (a))

O1 (c) on the owner's reasoning: a composite loss lets the regression error pull the class probabilities away
from the true bin probabilities, which breaks the probabilistic argument; cross-entropy alone keeps them
estimates of P(y in bin | x), and the heads read them without changing them.

| # | Choice | Options (recommended first) |
|---|---|---|
| O1 | Default supervision | (a) composite loss: regression error + cross-entropy on y bins, both train the classifier (default Jul–Sep 2025); (b) regression-only + monotonic head constraints (default from 2025-09-24); (c) cross-entropy only (`CE_STOP_GRAD`) |
| O2 | What heads predict | (a) mean and spread per head, combined into one Gaussian (original; reverses the 2026-10-20 "σ fixed" ruling); (b) keep σ fixed |
| O3 | Nested global-k model and router | (a) keep, strip only their mixture scoring, new review judges them; (b) remove now |
| O4 | Old mixture-based results and drivers | (a) delete tracked (git history keeps them), archive untracked to `~/cache/automl/archive/`; (b) keep, marked superseded |

Out of scope unless the owner says otherwise: the separate baseline models that produce mixtures
(`automl_package/models/baselines/mixture_density_network.py`, `deep_ensemble.py`) — they are comparison models, not ProbReg.

## Inventory (what carries the framing)

- **Library.** `automl_package/models/probabilistic_regression.py` (`FIXED_SIGMA_MIXTURE` and `MDN` loss
  branches, `fixed_sigma_train`, `_RETIRED_OPTIMIZATION_STRATEGIES` guard, default `REGRESSION_ONLY`,
  `predict_distribution` returning a mixture); `automl_package/enums.py` (`ProbRegLossType.FIXED_SIGMA_MIXTURE`,
  `MDN`; `HeadSpread`); `automl_package/utils/losses.py` (`mdn_nll`, `fixed_sigma_mixture_log_likelihood`);
  `automl_package/models/flexnn/strategies/n_classes.py` (`mixture_components_at_rung`);
  `automl_package/models/common/regression_heads.py:304` (refuses monotonic/anchored heads unless spread is per input).
- **Tests (12 files).** `test_fixed_sigma_scorer.py`, `test_mdn_nll.py`, `test_distributions_mog_components.py`,
  `test_ce_stop_grad.py`, `test_capacity_accounting.py`, `test_ordering_constraint.py`,
  `test_probreg_identifiability_integration.py`, `test_structure_phase_heads.py`, `test_phase1..4_*.py`.
- **Tracked drivers and results.** `probreg_structure_battery.py` (July structure study) and its results
  `capacity_ladder_results/PS1..PS3/` (284 result files); `probreg_identifiability_sweep.py` (April, MDN arms);
  `exponential_symlog_probe.py`; `report_a_benchmark.py` (imported by `probreg_p8.py`).
- **Untracked review material.** `probreg_review_cards.py`, `probreg_review_c4.py`, `probreg_review_c1_readout_calibration.py`,
  `probreg_review_code_checks.py`, `probreg_review_curves.py`; results `capacity_ladder_results/review_cards/`.
- **Living documents** (reference counts in brackets). `docs/probreg/PLAN.md` [24], `fixed_k/plan.md` [61],
  `global_k/plan.md` [3], `per_input_k/plan.md`, `shared/plan.md` [11], `shared/benchmark_spec.md` [33],
  `shared/toy_suite.md` [12], `shared/state.md` [4], `docs/RULES.md` [6], `docs/applications/research_plan.md` [34],
  `docs/package/mathematical_guide.tex` [10], project `CLAUDE.md`. Journals and `history/` folders are records:
  they get a one-line superseded banner, not edits.
- **The review.** `docs/probreg/review/review.md` [122], `initial_results.md` [26], `groundwork/` and `sources/`.

## Tasks (write sets; order = dependency order)

1. **Library** — the files under Library above. Remove the fixed-σ mixture and MDN losses and their
   parameters; remove the Decision 29 cross-entropy retirement; set the default supervision (O1) and head
   output (O2); make `predict_distribution` return the model's one predictive Gaussian (O2a); strip
   `mixture_components_at_rung` (O3a) or the nested/router code (O3b). Defaults and refusals change, so
   every changed behaviour gets a test.
2. **Tests** — delete tests of removed features; update the rest to the new defaults; one test per new
   default and per removed refusal. Full suite green (`~/cache/automl/venv-dgx/bin/python -m pytest tests/`),
   except the two accepted-failing heteroscedastic tests.
3. **Drivers and results** — per O4. Every driver left must import and run its own self-test.
4. **Plans and documents** — delete superseded text from the living documents (Decision 26, Decision 29,
   fixed-σ readout, useful/true k definitions, structure-study plan); banner the journals and history.
5. **The review** — remove from `review.md` and `initial_results.md` everything judged on the mixture
   likelihood (F1, F5–F8, D2–D5, cards C1, C3, C5–C7 as specified); keep the code and statistics findings
   (F4, F9, F11, F12, F14, F16, F20) and C4's comparison design, each re-checked against the new library.
6. **New review** — terms of reference written after 1–5, judged on held-out regression error.

## Validation

- Full test suite and ruff on `automl_package/` pass.
- A default `ProbabilisticRegressionModel` fits a toy and predicts; its classifier receives the class-label
  loss (checked by a test that breaks when the loss is removed).
- `grep -rE 'FIXED_SIGMA_MIXTURE|fixed_sigma|mdn_nll|ProbRegLossType.MDN'` over `automl_package/` (outside
  the baselines) and the living documents returns nothing.
- Independent review of the library diff before commit (tiered-review).

## Open items left by the cleanup (for the owner or the new review)

- **P8's status.** `docs/probreg/global_k/plan.md` still says P8's results are discarded, but the reason given
  (selection on a metric the fixed-σ rule forbade) is withdrawn. Do they stay void?
- **Cross-entropy with the nested k ladder.** With cross-entropy the default and bins recomputed for every k,
  the class-identity conflict that regression-only once dodged is live again for the nested model
  (`docs/probreg/shared/state.md` D4). The library keeps its refusal of cross-entropy with dynamic or nested k.
- **Wording kept on purpose, to re-read under the ruling:** per-rung "held-out likelihood" tables
  (`docs/probreg/PLAN.md`, `docs/probreg/shared/benchmark_spec.md`); true-k and multimodality language in
  `docs/probreg/shared/toy_suite.md`, `toy_design_spec.md` and `docs/applications/research_plan.md` (a one-Gaussian
  prediction cannot show several peaks); the CatBoost-dropped reasoning in `benchmark_spec.md`.
- **Journals moved** to `docs/probreg/history/journals_2026-10-21/` (records of the mixture-era work; the plan
  gates exempt history folders, and the journals cite files and plan lines the cleanup deleted).
- **FlexNN documents** still describe the fixed-σ scorer (`docs/flexnn/package/plan.md`, FP-12); FlexNN is on hold.
