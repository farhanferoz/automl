# Next Steps

**Status**: Phase 7 (metrics) + Phase 8 (baselines) complete. Toy + ablation benchmarks done.
UCI benchmark running. Key finding: FlexNN(PROBABILISTIC) closes the NLL gap vs ProbReg.

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
