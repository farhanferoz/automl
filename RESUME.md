# Next Steps

**Current active work (2026-04-20)**: ProbReg identifiability investigation.

Symmetry-check diagnostic on toy datasets revealed ProbReg SEP_HEADS suffers from
head–class index swap degeneracy: under default `REGRESSION_ONLY` optimization
strategy, nothing forces head_i to anchor at centroid_i when p_i → 1. Root cause:
the current Gaussian-LTV loss `−log N(y; ŷ_mean, ŷ_var)` (via law of total
variance) is agnostic to which (p, μ, σ) configuration produced a given
(ŷ_mean, ŷ_var). Many degenerate factorizations satisfy the loss.

ClassReg verified clean end-to-end (classifier p_i(x) semantically correct,
centroids correctly computed, predictions bounded by [min_c, max_c] which is a
known LOOKUP_MEDIAN limitation, not a bug).

**Three orthogonal fixes identified, to be implemented as dials + experiments:**
1. **MDN NLL** (new regression likelihood): probabilities enter the likelihood
   directly → identifiability from likelihood structure
2. **CE_STOP_GRAD** (new classifier supervision mode): classifier trained on
   bin-CE alone; `probs.detach()` before heads → joint-likelihood with observed
   bin labels
3. **Anchored heads** (new head parametrization): `h_i(p_i) = c_i + (1−p_i)·f_i(p_i)`
   → hard structural anchor at p_i = 1

Plus bug fix B1: `_initialize_monotonic_head` uses `nn.init.normal_(weight, mean=-3.0)`
which breaks on all-positive targets (exponential MSE = 11 vs 0.45 for SEP).

Detailed implementation plan: `docs/probreg_identifiability_implementation_plan.md`.
Session workflow: Opus plans → Sonnet implements → Opus reviews → Sonnet experiments
→ PDFs.

---

**Status**: Phase 9 autonomous pass (Apr 18 2026). See `SESSION_JOURNAL_2.md` for the
full change list. Critical fixes + feature additions; script scaffolding for the
Paper A / Paper B primary tables authored and partly executed.

## Phase 9 headline

- **Found + fixed major bug**: `FlexibleHiddenLayersNN` excluded `n_predictor` from
  the main optimizer for all non-REINFORCE strategies. Every prior FlexNN result
  ran on a frozen n_predictor; depth regularisation never fired. Fixed.
- **Found + fixed tuple-unpack bug**: FlexNN training loop mis-unpacked the
  strategy return, silently disabling ELBO / DEPTH_PENALTY branches. Fixed.
- **Added feature**: `DepthRegularization.COST_AWARE_ELBO` with `cost_aware_lambda`.
- **Added feature**: `ProbReg.predict_distribution` returns the full mixture
  (~7% better NLL than collapsed-Gaussian on bimodal).
- **Added feature**: 5 synthetic datasets B1-B6 checked into `tests/fixtures/synthetic/`.
- **Fixed**: FT-Transformer training recipe (pre-norm + AdamW + warmup + grad-clip +
  mini-batch + val early stop). Previous implementation lost every UCI dataset.
- **Fixed**: `PyTorchNeuralNetwork.predict_uncertainty` shape bug for `MC_DROPOUT` and
  the dict-vs-ndarray bug in `BINNED_RESIDUAL_STD`.
- **Fixed**: `Base.fit` passed 0 as forced iterations to tree models when no HPO/
  early-stopping had run, crashing LightGBM.

## Noise-robustness benchmark v2 (after tree-baseline fix)

See `automl_package/examples/noise_robustness_results/`.

| sigma | best model | MSE | note |
|------:|:---|---:|:---|
| 0.05  | NN (PyTorch) | 0.005 | trees (0.007) close; ClassReg worst |
| 0.30  | LightGBM | 0.117 | NN / ProbReg dyn_k competitive |
| 1.00  | NN ≈ ProbReg_dyn_k | 1.099 | ClassReg k=2 (1.162) beats both XGB / LGBM |

Classification-bottleneck story is **partially confirmed**: at sigma=1.0 ClassReg k=2
beats tree baselines by a clear margin but NOT plain NN. Paper narrative should be
reframed accordingly. Dyn-k ProbReg with ELBO stays pinned at k=2 regardless of
sigma — a follow-up pass with `NClassesRegularization.NONE` (added to the benchmark
script but not yet re-run) will show whether dyn-k adapts when the prior is lifted.

## Remaining work

- Run: `head_structure_diagnostic`, `flex_nn_depth_viz`, `gumbel_elbo_retest`,
  `sep_heads_vs_single_final`, `flex_nn_ablation`, `probreg_ablation`,
  `multi_seed_sweep` (batch is executing sequentially).
- Run: `hpo_sweep` (scaffolded, N=50 trials per model; overnight).
- Re-run noise benchmark with the new dual-reg path once batch completes.
- Phase 10 / 11 / 12 need external data (LSST DC2, IllustrisTNG) — blocked.

## Recent session findings (2026-04-16)

**Full benchmark + ablation run.** See `automl_package/examples/full_benchmark_results/REPORT.md`.

**Headline findings:**
1. **FlexNN(PROBABILISTIC)** beats ProbReg on CRPS (0.605 vs 0.609) and MSE (1.496 vs 1.511) on heteroscedastic, matches on NLL within 2%. Previous benchmarks used `CONSTANT` which locks uncertainty to a single scalar. **This is the default to recommend for Paper B.**
2. **K_PENALTY + SoftGating beats ELBO + SoftGating** for dynamic k in ProbReg, marginally on both datasets. Both should be reported as viable dynamic-k regularizers.
3. **SINGLE_HEAD_FINAL_OUTPUT** (previously untested) has best NLL on heteroscedastic (1.326 vs 1.377 SEP_HEADS) and 2.3× better MSE on exponential. Must be in ablation.
4. **Gumbel + ELBO confirmed broken** (MSE 2.2 vs 1.5 for SoftGating) — noisy KL gradients.
5. **MC_DROPOUT + depth gating is catastrophic** (NLL=42). Must document incompatibility.
6. **FT-Transformer** is unexpectedly the NLL winner on heteroscedastic — non-negotiable baseline.
7. **GP(Matern)** dominates small smooth datasets (piecewise, exponential). Expected at n<1k.

## Phase 7/8 artifacts (this session)

| File | Purpose |
|---|---|
| `automl_package/utils/distributions.py` | PredictiveDistribution protocol + Gaussian/MoG/Empirical/Quantile |
| `automl_package/utils/scoring.py` | CRPS (closed-form + numerical), CRPS decomposition, Winkler, pinball |
| `automl_package/utils/calibration.py` | PIT, ECE, miscalibration area, sharpness, isotonic recal, KS/CvM/AD tests |
| `automl_package/utils/domain_metrics.py` | Photo-z metrics (σ_MAD, bias, η, CDE) — domain-gated |
| `automl_package/models/baselines/` | BaselineModel base + 8 wrappers (NGBoost, MDN, QR-NN, DeepEnsemble, GP, CQR, FT-Transformer, TabPFN) |
| `automl_package/models/conformal.py` | Added LocallyAdaptiveConformalWrapper |
| `automl_package/examples/full_benchmark.py` | Unified benchmark runner for all models × all datasets |
| `automl_package/examples/ablation_study.py` | ProbReg and FlexNN configuration sweeps |
| `automl_package/examples/flexnn_probabilistic_test.py` | FlexNN(PROBABILISTIC) validation |

All 71 existing tests still pass.

## Key documents

| File | Purpose |
|---|---|
| `docs/research_plan.md` | Two-paper publication roadmap: bugs, benchmarks, metrics, baselines, astrophysics |
| `docs/mathematical_guide.tex` | Complete math spec: all models, UQ methods, selection strategies, losses, metrics |
| `docs/benchmarks.md` | Current empirical results (Phases 1–5) — needs refresh with new data |
| `docs/architecture_analysis.md` | SOTA comparison + historical bug audit |
| `automl_package/examples/full_benchmark_results/REPORT.md` | Latest benchmark + ablation analysis |
| `ARCHIVE.md` | Per-phase change details |

## Open issues (architectural, not bugs)

- ClassReg `NN` mapper: poor results (two-stage training limitation; use ProbReg instead)
- ELBO + Gumbel for dynamic k: noisy KL gradients (use SoftGating instead)
- MC_DROPOUT + FlexNN: destructive interaction with depth gate (use PROBABILISTIC instead)
- `PyTorchNeuralNetwork.predict()` returns (N,1) not (N,) — minor downstream shape issue
- DeepEnsemble with CONSTANT-uncertainty members: variance of means too tight (use PROBABILISTIC members)

## Canonical model configurations

### ProbReg best overall
```python
ProbabilisticRegressionModel(
    n_classes=10, max_n_classes=10,
    uncertainty_method=UncertaintyMethod.PROBABILISTIC,
    n_classes_selection_method=NClassesSelectionMethod.SOFT_GATING,
    n_classes_regularization=NClassesRegularization.K_PENALTY,  # or ELBO
    regression_strategy=RegressionStrategy.SEPARATE_HEADS,
)
```

### FlexNN best (NEW: PROBABILISTIC is the default)
```python
FlexibleHiddenLayersNN(
    max_hidden_layers=4, hidden_size=64,
    layer_selection_method=LayerSelectionMethod.SOFT_GATING,
    depth_regularization=DepthRegularization.ELBO,
    uncertainty_method=UncertaintyMethod.PROBABILISTIC,  # was CONSTANT
)
```

## Execution roadmap

Phase 7: Metrics ✅ → Phase 8: Baselines ✅ → **Phase 9: Expanded benchmarks (UCI Energy/Yacht/Kin8nm done; California partial)** → Phase 10: Photo-z → Phase 11: Cluster mass → Phase 12A/B: Paper drafts.

**Full 36-item priority-ordered work list:** see `docs/work_order.md`. Next item to pick up: **#1 I4 noise-robustness benchmark** — validates the paper's original motivation (ClassReg/ProbReg as noise-robust alternatives to pure regression).

## Follow-up investigation backlog

See `docs/research_plan.md` §9 for the full list. Parked for later — not active work:

- **I1-I2**: Investigate *why* SEPARATE_HEADS underperforms SINGLE_FINAL on small data; validate with regression-head-output-vs-probability structure diagnostic.
- **I3**: Validate FlexNN's depth-selection hypothesis with per-input depth heatmaps + design stronger complexity-adaptive showcase.
- **I4**: **Noise-robustness benchmark** (critical, matches original ClassReg motivation in finance: n=3 beats pure regression on noisy data; does ProbReg auto-pick the right n?).
- **I5-I8**: Fix bug N2 (STE gradient path), benchmark STE + REINFORCE for both dynamic-k and depth selection, re-test Gumbel+ELBO after N1 fix.
- **I9**: Per-problem self-contained dashboards/reports with commentary.
- Priority items: complete California Housing, add `predict_distribution()` to ProbReg, multi-seed averaging, HPO sweeps, cost-aware ELBO.

## Deferred from current session (2026-04-20 identifiability work)

Discussed in detail but **not in scope for the current session**. Implement after the
primary ProbReg identifiability sweep (`docs/probreg_identifiability_implementation_plan.md`)
lands and we've seen the results.

- **Target transform infrastructure** (general-purpose preprocessing, model-agnostic):
  - Add `log`, `yeo_johnson`, optional `quantile_normal` to `target_transform` (symlog
    already exists)
  - Transform inversion layer cleanup: point-prediction flag (median vs mean with bias
    correction), variance via first-order Jacobian (document approximation limits),
    exact interval/quantile inversion via T⁻¹ endpoint-wise
  - NLL evaluation add `log|T'(y)|` change-of-variables term for fair cross-transform
    comparison
  - Audit `ConformalWrapper` composition with transforms (should be correct by
    construction but verify)
  - Coverage test on synthetic log-normal
  - Motivation: faithful handling of photo-z (σ_z / (1+z)), cluster log-mass, any
    multiplicative-noise domain. Toy data is additive-noise so transform will show
    a null result there — that's the right negative finding to document.

- **Secondary ProbReg experiments** (after primary sweep):
  - E4: Anchored × monotonic on/off (needs B1 bug fix)
  - E5: Control — anchored with/without `constrain_middle_class` to empirically
    confirm the per-class anchor subsumes the middle-class special case
  - E6: Transform diagnostic on exponential for ClassReg + top 2 ProbReg cells

- **Subsumed tasks** (from workslate):
  - #37 "Middle-class centering investigation" — folded into anchored-heads work
  - #38 "Loss functions full grid" — becomes the primary sweep E1–E3
