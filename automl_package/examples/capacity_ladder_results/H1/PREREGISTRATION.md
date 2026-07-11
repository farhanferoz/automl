# H1 preregistration — ProbReg: two-phase schedule vs shipping joint gate vs fixed-k

Source: `docs/plans/perinput_selector_2026-07-10/EXECUTION_PLAN.md`, §3 WS-C, task **H1**
(lines ~283-319), plus §0a decisions L1-L5, §0b governance, §0c. Written BEFORE any real run,
verbatim from the ratified plan (2026-07-10) plus the dispatch message's arm/bar detail. Do not
edit after a real run starts; outcomes are read against this text, not the reverse.

## Purpose

The decisive "does our model work" comparison, on the SAME shipping architecture
(`ProbabilisticRegressionNet` driven from an experiment script): does the program's two-phase
scheme beat the current jointly-trained gate?

## Arms

Toys D and E + C_broad control; 3 seeds; identical net config, k_max=6, held-out NLL of the
per-input blended density as primary metric, G6 full-vector selector distributions logged.

- **(a) SHIPPING JOINT**: `NClassesSelectionMethod.SOFT_GATING` + `NClassesRegularization.ELBO`
  (the library's best combo), trained as today.
- **(b) TWO-PHASE**: phase 1 — per-sample k ~ U{1..k_max} masked-prefix schedule on the SAME
  net, gate quiescent (mirrors `masked_prefix_nll` / the `_compute_predictions_for_k` masking
  pattern from `automl_package/models/selection_strategies/base_selection_strategy.py`); phase
  2 — freeze everything except `n_classes_predictor`, train it with the S1-winning SOFT
  selector objective on a held-out-within-train split. Respects the head's logit layout (k∈
  {2..k_max} then bypass LAST, `probabilistic_regression_net.py:66-75`).
- **(c) FIXED-k SWEEP**: same net, fixed k ∈ {1..6}; best single k by held-out NLL.
- **(d) OPTIONAL FINE-TUNE** (measured, NO bar): arm (b) + 100 ep joint at lr 1e-4. Prediction:
  |ΔNLL| ≤ 0.01 nat (safe no-op). Explicitly droppable under budget pressure.

## Pre-registered bars

(i) (b) ≥ (a) on toy D held-out NLL on 3/3 seeds;
(ii) (b) within 0.02 nat of best (c) on C_broad;
(iii) parity: (b)'s phase-2 gate ≈ the standalone S1-SOFT selector on the same data
(|ΔNLL| ≤ 0.01 nat);
(iv) toy E: report only, NO bar (T3 owns the moving-mode question).

## Selftest

On a tiny toy-D subsample (N=200, 50 ep) the script runs ALL arms end-to-end and produces
finite NLLs + full per-input selector vectors.

## Script/artifacts

`automl_package/examples/capacity_ladder_h1.py` →
`capacity_ladder_results/H1/{PREREGISTRATION.md,h1_summary.json}`.

## Run

`AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python -u
automl_package/examples/capacity_ladder_h1.py` (selftest first with `--selftest`).

## Cost

Measure one arm-(b) fit first; expect ~1 day serialized for ~30 fits (orchestrator-owned —
this worker authors code + selftest only, per the plan's run-ownership rule).

## Non-goals

Library edits (K7 is user-gated; if phase 2 proves impossible without touching library code,
this script must STOP and report `UNRESOLVED`, not improvise a library change — it did not);
XPU; moving-mode verdicts; K7/porting language in any output (H1 is evidence FOR a future K7
decision, not the port).

---

## Implementation notes (not part of the plan text, added by the authoring worker)

- **Arms (a)/(b)/(c)/(d) all drive the real `ProbabilisticRegressionNet` /
  `ProbabilisticRegressionModel`**, no re-implementation and no library edits. Arm (a) and each
  fixed-k point of arm (c) use the ordinary `.fit()` path. Arm (b) cannot use `.fit()` (it needs
  a custom two-phase training loop with the gate held quiescent through phase 1), so it
  constructs the net via `ProbabilisticRegressionModel(...).build_model()` directly and drives
  `net.classifier_layers` / `net.regression_module` / `net.direct_regression_head` /
  `net.n_classes_predictor` by hand. `_seed_head_centroids` replicates the SEPARATE_HEADS
  centroid/middle-class-init bookkeeping `_fit_single` normally performs before `build_model()`
  (`probabilistic_regression.py:519-530`), and the `torch.manual_seed`/`np.random.seed` calls
  before `build_model()` replicate `base_pytorch.py:129-132` — both needed so arm (b)'s net
  starts from the SAME initialization scheme as arms (a)/(c) despite bypassing `.fit()`.
- **The primary metric (blended NLL) is NOT `predict_distribution`.** That method explicitly
  raises `NotImplementedError` for `n_classes_selection_method != NONE`
  (`probabilistic_regression.py:586-589`), so it only covers arm (c)'s fixed-k points. Arms
  (a)/(b)/(d) use a hand-written `_per_k_log_density` + `_gate_probs_c_grid_ascending` +
  `_blended_nll`, which computes the genuine two-level mixture density (mixture over k-modes,
  each mode itself a k-component `SEPARATE_HEADS` mixture) via the same
  `_compute_predictions_for_k`-style masking and `regression_module(..., return_head_outputs=
  True)` per-class read that `predict_distribution` itself uses for the fixed-k case — the
  dynamic-k generalization the library doesn't ship, built here in the examples script only.
- **k=1 is the bypass/direct-regression rung, consistently with the rest of the
  capacity-ladder program** (`_capacity_ladder_nested.py`'s "component 0 = the k=1 rung = the
  direct/bypass single Gaussian"). For arms (a)/(b)/(d) this is `net.direct_regression_head(x)`,
  an x-dependent MLP. For arm (c)'s fixed k=1 point (`n_classes_selection_method=NONE,
  n_classes=1`), `SEPARATE_HEADS`' single head is fed the class probability (always 1.0, not
  x) as its own input — an inherent property of the `SeparateHeadsRegressionModule` design (the
  head's input is `probabilities[:, i]`, never the raw feature), so arm (c)'s k=1 point is
  structurally a CONSTANT (x-independent) Gaussian, architecturally distinct from arms (a)/(b)/
  (d)'s x-dependent bypass. This is not a script bug and not something in scope to fix (it would
  require a library change to how `SEPARATE_HEADS` conditions its heads); flagged here for the
  orchestrator/adjudicator in case a different k=1 baseline is wanted for the fixed-k sweep.
- **Bar (ii)'s "within 0.02 nat" is read as a two-sided `abs(diff) <= 0.02`**, matching bar
  (iii)'s own explicit "|ΔNLL| <= 0.01" formula in the same plan sentence structure (unlike
  S1/S2's "advantage over global <= 0.02 nat" one-sided no-invented-structure framing, which
  this bar's wording does not use). A judgment call — confirm or override before the real run
  if the one-sided "(b) not worse than best (c)" reading is intended instead.
- **Phase 2's epoch/lr budget (300 ep, lr=1e-2) matches `capacity_ladder_k6.N_EPOCHS`/`LR`
  exactly**, not H1's own 100-epoch/1e-2 `.fit()` budget used for phase 1 and arms (a)/(c). This
  is deliberate: bar (iii)'s parity check is "same objective, different host" — the in-net gate
  and the standalone `_RouterMLP` (bar iii's comparison arm) both implement literally the S1
  SOFT recipe, so both are trained with that recipe's own established config, not an arbitrary
  H1-local epoch count.
- **Arm (d) fine-tunes WITHOUT re-enabling ELBO** (`n_classes_regularization` is set to `NONE`
  before the fine-tune loop, even though arm (a)'s "as today" recipe uses ELBO). Judgment call:
  arm (d) is testing whether unfreezing everything under the plain joint NLL is a safe no-op
  relative to arm (b)'s distilled state, not whether re-introducing the ELBO prior perturbs it —
  the auto-gated ordering-constraint penalty (SEPARATE_HEADS + Gaussian-LTV + REGRESSION_ONLY,
  `probabilistic_regression.py:129-135`) is left ON, since that is on by default for this
  combo "as today" and is not an ELBO/n_classes-selection regularizer.
- **Phase 1/phase 2 train on disjoint halves of `x_train`** (deterministic index-parity split,
  even = phase-1 fit, odd = phase-2 targets/fit — matching S1/K6's leak-avoidance convention),
  while arm (a) trains on the model's own internal 80/20 `.fit()` split (validation is for
  early stopping only, not for shaping the objective). This asymmetry is inherent to any
  legitimate two-phase distillation (phase 2's targets must come from data phase 1 didn't fit,
  or the gate would imitate overfit scores) and is not a fairness bug — noted for the record.

## Selftest result (this authoring pass)

`AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python -u
automl_package/examples/capacity_ladder_h1.py --selftest` exits 0; all 8 checks (finite NLL +
full 6-entry selector vector summing to 1.0 for arms a/b/d, finite standalone-router NLL,
finite fixed-k NLL for all 6 k values) PASS. No real read has been performed — the real 3-toy x
3-seed x 4-arm matrix is orchestrator-owned per §0c.
