# Next Steps

**Status**: Phases 1–5 complete. 71 tests passing.

- Phase 1 (16) — Critical bug fixes + ProbReg baseline
- Phase 2 (18) — FlexibleNN gradient fixes + `DepthRegularization`
- Phase 3 (14) — Dynamic n_classes fixes + `NClassesRegularization`
- Phase 4 (17) — β-NLL, symlog, conformal, hard inference
- Phase 5 (6) — SHAP tuple-output fixes, symlog MC uncertainty

For empirical results see `docs/benchmarks.md`. For per-phase change details see `ARCHIVE.md`.

## Open issues

- ClassReg `NN` mapper: poor results (two-stage training limitation — architectural; use ProbReg)
- ClassReg `SPLINE` mapper: convergence warnings, unstable (dominated by `LOOKUP_MEDIAN` on benchmarks)
- ELBO + Gumbel for dynamic k: noisy KL gradients prevent convergence (use SoftGating instead)

## Recently fixed

- SHAP DeepExplainer on FlexibleNN/ProbReg: added `_ShapModelWrapper` returning only the prediction tensor (`flexible_neural_network.py`, `independent_weights_flexible_neural_network.py`).
- SHAP on dynamic n_classes ProbReg: falls back to `KernelExplainer` (DeepLIFT can't trace per-sample dispatch). See `probabilistic_regression.py:get_shap_explainer_info`.
- Symlog uncertainty: replaced linearized Jacobian with Monte Carlo through `symexp` (`probabilistic_regression.py:_symlog_mc_moments`). Also makes `predict()` consistent with `predict_uncertainty()` (both come from the same posterior sample).

## Possible next directions

- Expose `loss_type` / `beta` / `target_transform` in Optuna search spaces
- Hard inference for ProbReg (currently FlexibleNN only)
- Adaptive conformal (per-input quantile via residuals/variance)
- Multivariate target support
