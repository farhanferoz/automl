# Archive — Completed Work & Historical Context

Read on demand only. Not auto-loaded.

## Session: 2026-04-06 — Phase 1 Complete

### Bugs Fixed

| # | Bug | File | Fix | Status |
|---|-----|------|-----|--------|
| 8 | Double log() on log_variance | `models/common/losses.py:41` | Remove extra `log()` | **Fixed** |
| 10 | Duplicate build_model() call | `independent_weights_flexible_neural_network.py:198` | Delete line | **Fixed** |
| 11 | Showcase indexes 1D array as 2D | `examples/probabilistic_regression_showcase.py:253` | Use `predict()` + `predict_uncertainty()` | **Fixed** |
| — | SINGLE_HEAD_N_OUTPUTS in-place view crash | `models/common/regression_heads.py:346` | Add `.clone()` before reshape | **Fixed** |
| — | `optimizer_type` defaults to `False` | `models/classifier_regression.py:51` | Default to `OptimizerType.ADAM` | **Fixed** |
| — | NN mapper val indices overlap training data | `models/classifier_regression.py:177` | Offset by train size | **Fixed** |
| — | `Metric.MLOGLOSS` doesn't exist | `xgboost_model.py:96`, `lightgbm_model.py:72` | Use framework strings directly | **Fixed** |
| — | CUDA-only device detection | `utils/pytorch_utils.py:45` | Add XPU fallback | **Fixed** |

### Remaining Bugs (from original audit)

| # | Bug | Severity | Status |
|---|-----|----------|--------|
| 1 | `_weighted_average_logic` returns 2 values, callers expect 3 | CRITICAL | Open (Phase 3) |
| 2 | Dynamic n_classes strategies return 4 values, forward expects 5 | CRITICAL | Open (Phase 3) |
| 3 | Dynamic n_classes strategies missing `boundaries` kwarg | CRITICAL | Open (Phase 3) |
| 4 | `_weighted_average_logic` passes raw x_input instead of logits | CRITICAL | Open (Phase 3) |
| 5 | REINFORCE log_prob in wrong tuple position | CRITICAL | Open (Phase 2) |
| 6 | STE argmax destroys gradient flow | CRITICAL | Open (Phase 2) |
| 7 | IndependentWeightsSTE returns soft probs | CRITICAL | Open (Phase 2) |
| 9 | Self-referential middle_class_dist_params_ | HIGH | Open (Phase 3) |

### Model Comparison Results (heteroscedastic dataset, n=1000)

| Model | MSE | NLL | Cal. |
|-------|-----|-----|------|
| LinearReg | 3.88 | 2.10 | 0.64 |
| XGBoost | 2.15 | 1.82 | 0.65 |
| LightGBM | 1.61 | 1.66 | 0.71 |
| ClassReg_median_k10 | 1.57 | 1.62 | 0.83 |
| ClassReg_XGB_median_k7 | 1.73 | 1.73 | 0.85 |
| ClassReg_spline_k7 | 6.38 | 1.88 | 0.72 |
| ClassReg_NN_k7 | 8.02 | 10.73 | 0.32 |
| **ProbReg_k5** | **1.54** | **1.38** | 0.70 |

**Key findings**: ProbReg wins on NLL (learned variance). ClassReg_median competitive on MSE at k=10. ClassReg_NN still broken (training issue, not architecture). Spline mapper unstable.

## Session: 2026-04-06 — Phase 2 Complete

### Bugs Fixed (Phase 2)

| # | Bug | File | Fix | Status |
|---|-----|------|-----|--------|
| 5 | REINFORCE log_prob wrong tuple position | `layer_selection_strategies.py:198-204` | Swapped to position 4 matching caller | **Fixed** |
| 6 | STE argmax kills gradients | `layer_selection_strategies.py:153-167` | Replaced torch.where with weighted-sum pattern (hard=True) | **Fixed** |
| 7 | IndependentWeightsSTE returns soft probs | `independent_weights_strategies.py:193` | Return `gumbel_softmax(hard=True)` as n_probs | **Fixed** |
| — | Missing `setup_optimizers` on 4 layer strategies | `layer_selection_strategies.py` | Added abstract method implementations | **Fixed** |
| — | `x.values` on numpy array | `flexible_neural_network.py`, `independent_weights_flexible_neural_network.py` | Added `hasattr` guard | **Fixed** |
| — | `build_model()` double-call overwrites activation enum | Both FlexibleNN variants | Check `isinstance(ActivationFunction)` before lookup | **Fixed** |

### New Features (Phase 2)

- `DepthRegularization` enum: NONE, DEPTH_PENALTY, ELBO
- ELBO depth control (KL vs geometric prior) in both FlexibleNN variants
- Depth penalty (weighted expected depth) in both FlexibleNN variants

### Model-Level Test Results (Phase 2)

- FlexibleNN MSE competitive with fixed 1-layer NN on linear data (no degradation from depth selection)
- FlexibleNN beats 1-layer NN on piecewise data (depth helps on complex regions)
- ELBO selects shallower on linear vs piecewise data (prior works correctly)
- IndependentWeightsFlexibleNN achieves MSE < 2.0 on linear data
- All 4 strategies (GumbelSoftmax, SoftGating, STE, REINFORCE) train without crash on both variants

### Infrastructure Created

- `tests/conftest.py` — 4 fixtures (heteroscedastic, multimodal, linear, piecewise)
- `tests/test_phase1_probabilistic_regression.py` — 16 tests
- `tests/test_phase2_flexible_nn.py` — 17 tests
- `examples/model_comparison.py` — full model comparison script
- `pyproject.toml` — replaces setup.py, integrated into uv workspace at `~/dev/`
- XPU device detection in `get_device()`, workspace venv at `~/dev/.venv/` with torch 2.10.0+xpu

## Session: 2026-04-06 — Phase 3 Complete

### Bugs Fixed (Phase 3)

| # | Bug | File | Fix | Status |
|---|-----|------|-----|--------|
| 1 | `_weighted_average_logic` returns 2 values, callers expect 3 | `base_selection_strategy.py:48` | Compute classifier_raw_logits inside, return 3-tuple | **Fixed** |
| 2 | Dynamic n_classes strategies return 4 values, forward expects 5 | `n_classes_strategies.py` | All 4 strategies return 5-tuple with `per_head_outputs=None` | **Fixed** |
| 3 | Dynamic n_classes strategies missing `boundaries` kwarg | `n_classes_strategies.py` | Added `boundaries: torch.Tensor | None = None` to all forward() sigs | **Fixed** |
| 4 | `_weighted_average_logic` passes raw x_input instead of logits | `base_selection_strategy.py:68` | Compute `classifier_raw_logits = self.model.classifier_layers(x_input)` first, pass logits | **Fixed** |
| 9 | Self-referential `middle_class_dist_params_` | `probabilistic_regression.py:325-339` | Collect per-k params in temp dict, assign after loop | **Fixed** |
| — | `n_classes_inf=float('inf')` overflows torch.long | `base_selection_strategy.py` | Use `2**30` sentinel for direct regression k value | **Fixed** |
| — | Dead `forward()` on ProbabilisticRegressionModel | `probabilistic_regression.py:538` | Removed (Net's forward is the actual call path) | **Fixed** |
| — | `get_classifier_predictions` unpacks 4 from 5-tuple | `probabilistic_regression.py:448` | Unpack 5 values | **Fixed** |

### New Features (Phase 3)

- `NClassesRegularization` enum: NONE, K_PENALTY, ELBO
- ELBO k control (KL vs normalized prior) in ProbabilisticRegression loss
- K penalty (weighted expected k) in ProbabilisticRegression loss
- GumbelSoftmax n_classes strategy: deterministic softmax at eval time (no Gumbel noise)

### Debugging: ELBO Prior Steepness

Original prior `logits = arange(n_modes, 0, -1)` scaled steepness with n_modes:
- FlexibleNN (3 modes): KL cost depth=1→3 = 1.8 nats
- ProbReg (10 modes): KL cost k=2→10 = 8.0 nats (unbeatable)

Fix: `logits = linspace(3.0, 1.0, n_modes)` — same range regardless of n_modes.
Result: SoftGating+ELBO went from MSE=1.68 to MSE=1.56 (matching best fixed-k).

ELBO works well with SoftGating but not GumbelSoftmax. Gumbel noise during training creates
noisy KL gradients that prevent learning a good spread distribution. SoftGating's deterministic
softmax produces stable training. Recommendation: use SoftGating with ELBO.

### Dynamic k Comparison Results (heteroscedastic sine, n=1000)

| Model | MSE | NLL | Cal(1σ) | Noise r | Mean k |
|-------|-----|-----|---------|---------|--------|
| Fixed k=3 | 1.56 | 0.45 | 0.67 | 0.98 | 3 |
| Fixed k=5 | 1.55 | 0.46 | 0.70 | 0.95 | 5 |
| Fixed k=10 | 1.55 | 0.47 | 0.68 | 0.98 | 10 |
| Dynamic (none) | 1.88 | 0.59 | 0.71 | 0.91 | 10 |
| Dynamic (ELBO+Gumbel) | 2.24 | 0.80 | 0.68 | 0.73 | 2.95 |
| Dynamic (k-penalty) | 1.63 | 0.42 | 0.72 | 0.96 | 4.57 |
| **Dynamic (ELBO+SoftGating)** | **1.56** | **0.44** | **0.69** | **0.99** | **2.00** |

**Key finding**: ELBO+SoftGating matches best fixed-k MSE while achieving best noise correlation (0.99).

### Phase 3 Test Results (14 tests)

- All 4 dynamic strategies (GumbelSoftmax, SoftGating, STE, REINFORCE) train without crash
- Classifier weights influence predictions (Bug 4 verified)
- `middle_class_dist_params_` stores independent per-k params (Bug 9 verified)
- Dynamic-k MSE < 10.0 on heteroscedastic data
- Dynamic-k NLL competitive with fixed-k=5 (within 2x)
- Per-input k varies (n_classes_predictor learns input-dependent k)
- Uncertainty correlates with actual noise (r > 0.2)
- ELBO k converges to reasonable MSE
- ELBO selects lower mean k than unregularized
- K penalty reduces mean k vs no regularization

### Session: 2026-04-06 — Persistence & Architecture Audit

- Produced `docs/architecture_analysis.md` — full bug audit, SOTA comparison
- Produced `docs/implementation_plan.md` — phased implementation plan
- Created `~/dev/antigravity_recovery/antigravity_recovery.py` for session recovery

## Session: 2026-04-07 — Phase 5: SHAP + Symlog Fixes

### Bugs Fixed (Phase 5)

| # | Bug | File | Fix |
|---|-----|------|-----|
| 1 | SHAP DeepExplainer crashes on FlexibleNN (5-tuple forward) | `flexible_neural_network.py` | Added `get_internal_model()` returning `_ShapModelWrapper` that strips tuple to prediction tensor |
| 1 | Same crash on IndependentWeightsFlexibleNN | `independent_weights_flexible_neural_network.py` | Same wrapper pattern |
| 2 | SHAP crashes on dynamic n_classes ProbReg (DeepLIFT batch-dispatch) | `probabilistic_regression.py:get_shap_explainer_info` | Falls back to `ExplainerType.KERNEL` with `self.predict` callable when `n_classes_selection_method != NONE` |
| 3 | Symlog uncertainty uses linearized Jacobian — inaccurate near zero | `probabilistic_regression.py:predict_uncertainty` | Replaced with `_symlog_mc_moments`: 200 samples from `N(μ_symlog, σ²)` through `symexp`, empirical std |
| 3 | `predict()` inconsistent with `predict_uncertainty()` under symlog | `probabilistic_regression.py:predict` | Both now use MC moments from same sample — `predict()` returns MC mean |

### Tests Added (Phase 5)

- `TestSHAP` (4 tests) — FlexibleNN, IndependentWeightsFlexibleNN, ProbReg fixed-k (Deep), ProbReg dynamic-k (Kernel fallback)
- `TestSymlogMCUncertainty` (2 tests) — MC std > Jacobian near zero crossing; end-to-end non-NaN + positive outputs

### Test Suite Totals

- **23 tests** in `test_phase4_regression.py` (was 17). Full suite: 71 tests across Phases 1–5.

## Session: 2026-04-07 — Phase 4 Complete

### New Features (Phase 4)

| Feature | File | Description |
|---------|------|-------------|
| β-NLL loss | `utils/losses.py:beta_nll_loss` | Variance-collapse-resistant NLL (Seitzer et al. 2022) |
| `loss_type`/`beta` params | `models/common/losses.py`, `probabilistic_regression.py` | Wired into ProbReg via `calculate_combined_loss` |
| Symlog/symexp transform | `utils/transforms.py` | **NEW** module with `symlog`/`symexp` |
| `target_transform="symlog"` | `probabilistic_regression.py` | Applied in `_fit_single` before binning, reversed in `predict()`/`predict_uncertainty()` |
| Conformal wrapper | `models/conformal.py` | **NEW** — `ConformalWrapper` with split conformal + finite-sample correction |
| Hard inference | `flexible_neural_network.py:hard_forward` + `predict(inference_mode=...)` | Per-sample argmax depth bucketing |

### Phase 4 Comparison Results

**NLL vs β-NLL on heteroscedastic data (n=1000)**

| Variant | MSE | NLL | Cal | NoiseR |
|---------|-----|-----|-----|--------|
| NLL | 1.80 | 0.54 | 0.69 | 0.94 |
| β-NLL β=0.5 | 1.84 | 0.57 | 0.70 | 0.93 |
| β-NLL β=1.0 | 2.21 | 0.80 | 0.68 | 0.51 |

β=0.5 is competitive with standard NLL; β=1.0 over-corrects on this dataset (likely because the
noise model is well-behaved, so variance reweighting hurts more than it helps).

**Symlog vs raw on exponential targets (n=800)**

| Transform | MSE | PearsonR |
|-----------|-----|----------|
| raw | 1.0008 | 0.9788 |
| **symlog** | **0.1934** | **0.9978** |

5x improvement in MSE on `y = exp(x)` targets — exactly the use case symlog is designed for.

**Hard vs soft inference on FlexibleNN piecewise (max_depth=4, hidden=64)**

| Mode | MSE | Time(s) | Notes |
|------|-----|---------|-------|
| soft | 0.30 | 0.30 | weighted-sum across depths |
| hard | 0.34 | 0.31 | argmax bucket execution |

Predictions track tightly (diff MSE 0.02). No speedup at this scale — bucket-grouping overhead
dominates the savings on tiny networks. Speedup should appear for larger networks
(more depths, wider hidden layers).

**Conformal coverage on heteroscedastic data**

| α | target | coverage | width |
|---|--------|----------|-------|
| 0.05 | 0.95 | 0.965 | 5.82 |
| 0.10 | 0.90 | 0.890 | 4.61 |
| 0.20 | 0.80 | 0.780 | 3.19 |

Coverage matches target almost exactly across all α values. Finite-sample correction
(`ceil((n+1)(1-α))/n` quantile) ensures the coverage guarantee holds.

### Phase 4 Test Results (17 tests)

- `TestPerformanceBaselines` (3): ProbReg NLL < 1.8, MSE < 2.5; FlexibleNN piecewise MSE < 2.0
- `TestCrossModelRanking` (2): ProbReg beats constant-variance NN on heteroscedastic; FlexibleNN+ELBO competitive with shallow on piecewise
- `TestEndToEndAllModels` (3): probreg_fixed_k, flexible_nn_gumbel, flexible_nn_elbo all complete train→predict→uncertainty
- `TestBetaNLL` (2): trains without crash, uncertainty correlates with noise (r > 0.2)
- `TestHardInference` (2): hard predictions close to soft; large-batch hard inference completes in time
- `TestConformalWrapper` (2): coverage ≥ 1-α-0.10; harder calibration → larger quantile
- `TestSymlogTransform` (3): roundtrip identity; monotone compression; trains on exponential targets (r > 0.5)

### Test Suite Totals

- 65 tests across all phases (16 + 18 + 14 + 17), full suite runtime ~7m48s on XPU.
- All tests pass on `~/dev/.venv/bin/python -m pytest tests/ -v`.

### Files Modified/Added

- `utils/losses.py` — added `beta_nll_loss`
- `utils/transforms.py` — **NEW**
- `models/common/losses.py` — `loss_type`/`beta` params in `calculate_combined_loss`
- `models/probabilistic_regression.py` — `loss_type`/`beta`/`target_transform` defaults; `predict`/`predict_uncertainty` overrides for symlog
- `models/conformal.py` — **NEW**
- `models/flexible_neural_network.py` — `hard_forward()` + `inference_mode` param on `predict()`
- `tests/test_phase4_regression.py` — **NEW** (17 tests)
- `examples/phase4_comparison.py` — **NEW** comparison script

---

## Phase 9 fix log (added April 2026 — autonomous pass)

### N2 (STE gradient path) — **verified already fixed in Phase 6**
`_hard_selection_logic` uses the weighted-sum pattern; `f.gumbel_softmax(hard=True)`
bakes in the STE trick so gradients flow back to `n_classes_predictor`. Regression
test `TestBugN2SteGradientPath` (tests/test_phase3_dynamic_k.py) now guards this.

### N11 — PyTorchNeuralNetwork uncertainty shape (new)
- `predict_uncertainty` under `MC_DROPOUT` returned `(N, 1)` instead of `(N,)`.
  Fixed by raveling `np.std(..., axis=0)`.
- `BINNED_RESIDUAL_STD` mixin stored stats as a NumPy array but treated it as a
  dict; `not array` then threw `ValueError`. Replaced dict-indexed lookup with
  direct array indexing and guarded with length check.
- `TestPredictShapes` (tests/test_phase4_regression.py) covers all four methods.

### FlexNN n_predictor was excluded from the main optimizer (new, severe)
`FlexibleHiddenLayersNN._setup_optimizers` filtered `"n_predictor" not in n`,
delegated to `strategy.setup_optimizers`. Non-REINFORCE strategies have a no-op
`setup_optimizers`, so the n_predictor weights stayed frozen at initialization
for SOFT_GATING, GUMBEL_SOFTMAX, STE, and NoneStrategy. **Depth regularisation
(ELBO, DEPTH_PENALTY) never affected training.** Same bug in
`IndependentWeightsFlexibleNN._setup_optimizers`.

Fix: include n_predictor in the main optimizer when the strategy does not use a
policy optimizer (REINFORCE only). Test `TestNPredictorInMainOptimizer`
(tests/test_phase2_flexible_nn.py) locks this in.

### FlexNN training-loop tuple unpacking (new)
`final_output, _, n_probs, n_logits, log_prob = self.model(_batch_x)` mis-unpacked
the strategy return — position 2 is always `None` (legacy slot), position 3 is
`n_probs`. All depth-regularisation branches checked `if n_probs is not None`
which was silently `False`, so the KL term never fired even once trainable
parameters had been set up correctly. Fixed both branches; `IndependentWeights`
uses a different tuple order (position 2 is n_probs) and was already correct.

### Base.fit `forced_iterations=0` crashed tree models (new)
`_fit_final_model` passed `forced_iterations=self.num_iterations_used`. Default
value 0 (no HPO, no early-stopping) crashes LightGBM ("Number of boosting rounds
must be > 0"). Fixed to pass `None` when `num_iterations_used <= 0`.

### Cost-aware ELBO for FlexNN (feature)
Added `DepthRegularization.COST_AWARE_ELBO` and `cost_aware_lambda`. Prior logits
are `linspace(3,1) - lambda * normalised_depth_cost`, so the KL now prefers
shallower depths as a function of the FLOPs cost parameter. Applies to both
shared-weight and independent-weight FlexNN variants.

### FT-Transformer training recipe (fragility fix)
The previous full-batch no-warmup implementation lost every UCI dataset.
Updated to:
- norm_first=True (pre-norm)
- AdamW with configurable weight_decay
- Linear warmup over `warmup_epochs`
- Mini-batching with random permutation per epoch
- Gradient clipping at `grad_clip_norm`
- Validation-driven early stopping via `early_stopping_rounds`

### predict_distribution (feature)
`ProbabilisticRegressionModel.predict_distribution(x)` returns the full mixture-of-
Gaussians predictive distribution built from per-class (mu, log_var) weighted by
softmax classifier probabilities. Supports NONE selection, SEPARATE_HEADS and
SINGLE_HEAD_N_OUTPUTS strategies; raises NotImplementedError for
SINGLE_HEAD_FINAL_OUTPUT / dynamic-k / symlog. On the bimodal toy, mixture-NLL
is ~7% better than the collapsed-Gaussian NLL (tests/test_phase1_probabilistic_regression.py::TestPredictDistribution).
