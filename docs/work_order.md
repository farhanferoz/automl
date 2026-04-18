# Work Order — Priority-Ordered Execution List

**Date:** 2026-04-16
**Scope:** Complete Phase 9 → Phase 10 (photo-z) → Phase 11 (cluster mass) → Paper drafts (Phase 12A/B) + cleanup.
**Companion docs:** `docs/research_plan.md` (authoritative plan), `RESUME.md` (session state), `automl_package/examples/full_benchmark_results/REPORT.md` (benchmark findings motivating this ordering).

Phases 1-8 are complete. Phase 9 is partial: toy + 3 UCI datasets were run, but synthetic datasets B1-B6, full ablation sweeps, multi-seed averaging, and California Housing are all pending.

Legend: **S** = <1 day, **M** = 1-3 days, **L** = 1+ week. Check boxes are intentional — tick as items complete.

---

## Phase 9 completion — validate the empirical story

**Must finish before Phase 10. These items directly determine whether the papers have a coherent empirical story.**

- [ ] **1. I4 — Noise-robustness benchmark** — S
  Sweep ClassReg n=1..20 × noise levels σ ∈ {0.01, 0.1, 0.5, 1.0}. Check ProbReg(dynamic k) auto-selected mean-k at each noise level. Compare to baselines (XGBoost, LightGBM, plain NN, CatBoost+UQ). Plot MSE vs n curves per noise level + auto-k vs noise. Validates the **original motivation** of classification-bottleneck regression.
  No dependencies. Deliverable: `automl_package/examples/noise_robustness_benchmark.py` + results table.

- [ ] **2. I2 — Regression-head-vs-probability structure diagnostic** — S
  Run `plot_prob_reg_internal_plots` on all ablation configs. Flag any without expected structure:
  - n=2: head_0 monotonically decreasing, head_1 monotonically increasing (or vice versa), mirror images.
  - n=3: middle head flat, outer heads mirror images.
  Models that fail the diagnostic are not learning what we claim. This likely explains **why SEP_HEADS overfits** (I1).

- [ ] **3. I1 — Investigate why SEPARATE_HEADS underperforms SINGLE_FINAL on small data** — M
  Controlled tests:
  - Parameter-matched (adjust hidden_size so total params are equal across strategies)
  - Per-head gradient-norm logging per batch
  - Frozen-classifier vs learned-classifier variants
  Depends on #2 (structure diagnostic gives half the answer).

- [ ] **4. I3 — FlexNN depth-vs-input visualization + stronger showcase** — M
  Per-input selected depth heatmap on piecewise dataset. Depth vs local-complexity correlation where complexity is quantified (e.g., ∂²y/∂x² or distance from a regime boundary). Design a new synthetic with a tunable local-complexity parameter if piecewise isn't sharp enough.
  Without this plot, **Paper B has no evidence for its headline claim**.

- [ ] **5. Fix bug N2 (STE gradient path for dynamic-k)** — S
  `automl_package/models/selection_strategies/base_selection_strategy.py:82`. Rewrite `_hard_selection_logic` to weighted-sum pattern (see research plan Appendix A). Blocks #6, #7.

- [ ] **6. I6 — Benchmark STE and REINFORCE for dynamic-k** — S
  Requires #5. Add baseline-subtraction variance reduction for REINFORCE. Required ablation row for Paper A strategy comparison.

- [ ] **7. I7 — Benchmark STE and REINFORCE for FlexNN depth selection** — S
  Same strategies, different model. STE already works for depth; REINFORCE needs benchmark. Required ablation row for Paper B.

- [ ] **8. I8 — Re-test Gumbel + ELBO after N1/N2 fixes** — S
  N1 was fixed in Phase 6; #5 fixes N2. Retest whether Gumbel+ELBO is still broken or partly recovered. Informs research plan narrative on Gumbel recommendations.

- [ ] **9. Add ProbReg's `predict_distribution()` method + re-run multimodal** — S
  MDN got 12% CRPS improvement with proper mixture evaluation. ProbReg's classification bottleneck IS a mixture and must be evaluated with `MixtureOfGaussiansDistribution`. Biggest evaluation gap currently.

- [ ] **10. Implement Layer 1 synthetic datasets B1-B6** — M
  From research plan §2.2:
  - B1: Gravitational inverse problem (d=10, n=10k)
  - B2: Oscillator phase ambiguity (d=8, n=10k)
  - B3: Two-phase transition (d=12, n=10k)
  - B4: Conditional heteroscedasticity with latent subpopulations (d=10, n=10k)
  - B5: Exponentially-distributed feature importance (d=30, n=10k)
  - B6: Null homoscedastic problem (d=8, n=5k) — sanity check
  Fixed seeds, checked into `tests/fixtures/`.

- [ ] **11. Fix `PyTorchNeuralNetwork.predict()` shape bug** — S
  Returns (N,1) instead of (N,). Pre-existing minor issue, causes downstream shape errors if not raveled.

- [ ] **12. Complete California Housing benchmark with per-model timeout** — S
  Previous run killed at 66 min on ProbReg dynamic-k. Add 5-minute per-model timeout; drop or lightweight dyn-k for n>10k. Fills the last UCI row.

- [ ] **13. Multi-seed averaging (5-20 seeds) for all Phase 9 numbers** — M
  Currently all results are single-seed=42. Research plan §2.3 mandates 5 seeds for 80/20 splits, 20 for small UCI. Report mean ± std.

- [ ] **14. Full ProbReg ablation sweep (research plan §2.3)** — L
  3 regression strategies × {NLL, β-NLL at β ∈ {0, 0.5, 1.0}} × {symlog yes/no} × {joint, GRADIENT_STOP} × dynamic-k strategies {NONE, SOFT_GATING, GUMBEL_SOFTMAX, STE} × regularization {NONE, K_PENALTY, ELBO}. Run on B1-B6 + UCI. Primary Paper A table.

- [ ] **15. Full FlexibleNN ablation sweep** — M
  max_hidden_layers ∈ {3, 5, 8} × depth_regularization ∈ {NONE, DEPTH_PENALTY, ELBO} × layer_selection ∈ {SOFT_GATING, GUMBEL, STE, REINFORCE} × {shared, independent weights} × {soft, hard inference} × {CONSTANT, PROBABILISTIC}. Primary Paper B table.

- [ ] **16. HPO sweep (Optuna) on UCI datasets** — L
  All current numbers are untuned defaults. Run N≥50 trials per model per dataset. Expose `loss_type`, `beta`, `target_transform`, `uncertainty_method` in search spaces.

---

## Phase 9 polish — after empirics land

**Valuable but not strictly required. Can run in parallel with Paper A draft.**

- [ ] **17. I9 — Per-problem report/dashboard infrastructure** — M
  Extend `Metrics.save_metrics()` to produce self-contained Markdown report per (model, dataset): problem description, dataset stats, metrics table, diagnostic plots, commentary. Dataset-level aggregator combines model reports.

- [ ] **18. Benchmark photo-z domain metrics + locally-adaptive conformal** — S
  `utils/domain_metrics.py` and `LocallyAdaptiveConformalWrapper` implemented in Phase 7/8 but never used. Sanity-check before Phase 10 (photo-z).

- [ ] **19. Deep Ensemble with PROBABILISTIC members** — S
  Current DE uses CONSTANT members → near-identical Gaussian mixture. Rerun with PROBABILISTIC; this is a mandatory baseline, needs to actually work.

- [ ] **20. ClassifierRegression on UCI (SHAP disabled)** — S
  Excluded from UCI runs due to SHAP slowness. Run with `calculate_feature_importance=False` to complete the comparison table.

- [ ] **21. Cost-aware ELBO for FlexNN** — M
  Weight KL term by FLOPs per depth: `cost_aware_prior[d] = linspace(3,1)[d] - λ · flops(d)`. Strengthens Paper B "compute-adaptive" narrative. Related: PonderNet (Banino 2021).

- [ ] **22. Investigate FT-Transformer fragility** — M
  Won heteroscedastic toy, lost all UCI. Either fix hyperparameters (d_model, warmup) or drop from headline comparisons with a footnote explaining.

---

## Phase 10 — Paper A headline experiment (photometric redshift)

- [ ] **23. Acquire LSST DC2 cosmoDC2 data + matched spec-z** — M
  Via GCRCatalogs at NERSC. Catalog `cosmoDC2_v1.1.4_image_with_photoz_v1`. Fallback: SDSS DR17 via SciServer CasJobs.

- [ ] **24. RAIL wrapper for ProbReg** — M
  Plug ProbReg into DESC photo-z evaluation framework (`github.com/LSSTDESC/rail`). Produces directly comparable numbers to 15+ published algorithms.

- [ ] **25. Run full ProbReg + baseline suite on photo-z** — L
  Baselines: NGBoost, MDN, DeepEnsemble, GP, TabPFN v2, Zephyr, nflow-z, Jones+2024 BNN. Domain metrics: σ_MAD, η_0.15, bias, CDE loss, PIT uniformity (KS/CvM/AD).

- [ ] **26. Produce Paper A plots** — M
  - Per-input k heatmap
  - Multimodal predictive density (Lyman break at z≈3)
  - Bin centroid overlay on redshift distribution
  - PIT histogram
  - Reliability diagram
  - Recalibration improvement bar chart
  - Critical-difference diagram across all datasets

---

## Phase 11 — Paper B headline experiment (galaxy cluster mass)

- [ ] **27. Download IllustrisTNG cluster observables** — S
  Via public API at `tng-project.org/data/`. Extract (M200c, M500c, richness, σ_v, T_X_proxy, Y_SZ_proxy).

- [ ] **28. Run FlexibleNN + baseline suite on cluster mass** — L
  Baselines: ProbReg, MDN, DeepEnsemble, Ntampaka+2019 CNN (cite, don't re-implement), Ho+2019 Bayesian NN.

- [ ] **29. Produce Paper B plots** — M
  - Per-input depth vs cluster dynamical state (relaxed vs disturbed)
  - Soft-vs-hard inference scatter
  - Wall-clock inference bar chart at max_depth ∈ {3, 5, 8, 10}
  - Depth histogram
  - Depth entropy vs prediction error scatter (free-contribution diagnostic)
  - Training dynamics (mean depth vs epoch)

---

## Phase 12 — Paper drafts

- [ ] **30. Paper A draft** — L
  Target venue: NeurIPS/ICLR methodology + MNRAS astro companion.
  Freeze tagged commit. Submit to arXiv.

- [ ] **31. Paper B draft** — L
  Target venue: NeurIPS/ICLR methodology + MNRAS astro companion.
  Frame FT-Transformer depth gating as future work (Paper C).

---

## Low-priority cleanup — fill gaps or skip

- [ ] **32. Add missing docstrings on baseline `__init__`/predict methods** — S
  Ruff D107/D102 warnings. Cosmetic.

- [ ] **33. SEPARATE_HEADS benchmark at n > 50k** — M
  Only if Paper A narrative depends on "SEP_HEADS wins at scale" hypothesis. Requires a large synthetic or combining UCI sets.

- [ ] **34. FlexibleNN with separate mean/var heads per depth** — M
  Architecture improvement: currently the output layer is shared across depths. Could specialize per depth.

- [ ] **35. Four-way train/val/cal/test splits** — S
  Clean conformal + recalibration benchmarks. Currently just train/test.

- [ ] **36. Fix Greek character warnings (RUF002/RUF003)** — S
  Cosmetic ruff noise cleanup (α, σ in docstrings/comments).

---

## Critical-path summary

If forced to pick one item per week:

| Week | Primary focus | Why |
|---:|:---|:---|
| 1 | #1 (I4 noise-robustness) | Validates paper premise. If ClassReg/ProbReg don't beat baselines on noisy data, paper needs rewrite. |
| 2 | #2, #3 (I1-I2 head structure) | Explains the SEP vs SINGLE_FINAL finding. Could reshape the paper's regression-strategy story. |
| 3 | #4 (I3 FlexNN visualization) | Paper B headline evidence. Without this we can't claim depth adapts to complexity. |
| 4 | #5-#8 (bug fixes + strategy re-tests) | Removes caveats from ablation tables. |
| 5 | #9 (ProbReg mixture eval) + #10 (synthetics B1-B6) | Biggest evaluation + data gaps. |
| 6-7 | #14, #15 (full ablation sweeps) | Primary paper tables. |
| 8 | #16 (HPO) | Credible untuned → tuned numbers. |
| 9-10 | #23-#26 (Phase 10 photo-z) | Paper A headline experiment. |
| 11-12 | #27-#29 (Phase 11 cluster mass) | Paper B headline experiment. |
| 13-15 | #30, #31 (drafts) | Submit. |

**The first 4 weeks determine whether the papers have a coherent story to tell. Everything after that is execution.**
