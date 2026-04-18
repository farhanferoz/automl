# Autonomous session journal (Apr 18)

Mirrors the work_order.md queue. Each section is "what was done, why, how verified".

## Fixes (bug-class)

### N2 verified (regression test added)
`tests/test_phase3_dynamic_k.py::TestBugN2SteGradientPath` asserts that, under STE
dynamic-k, gradients reach `n_classes_predictor` after one backward pass. The
`_hard_selection_logic` weighted-sum + STE trick had already landed in Phase 6.

### PyTorchNeuralNetwork uncertainty shape fixes
Two latent bugs found:
- `MC_DROPOUT` returned `(N, 1)` — raveled.
- `BINNED_RESIDUAL_STD` stored stats as ndarray but treated as dict — rewrote
  lookup with array indexing. `TestPredictShapes` covers all four methods.

### FlexibleNN n_predictor frozen at initialization (severe)
`FlexibleHiddenLayersNN._setup_optimizers` excluded all `n_predictor.*` params
and delegated to `strategy.setup_optimizers`, which is a no-op for every
strategy except REINFORCE. Non-REINFORCE depth selection effectively ran on
randomly-initialized weights and depth regularization never fired.

Same bug in `IndependentWeightsFlexibleNN._setup_optimizers`. Both fixed:
`main_optimizer` now owns `n_predictor` params unless REINFORCE is in use.

`TestNPredictorInMainOptimizer` locks this in.

### FlexNN training-loop tuple unpacking
`final_output, _, n_probs, n_logits, log_prob = self.model(...)` took position 2
(always `None`) as `n_probs`. All depth-regularisation branches silently never
ran. Fixed to use position 3. Independent-weights version uses a different tuple
order (position 2 IS n_probs) — left as-is.

### Base.fit forced_iterations=0 crash on trees
`_fit_final_model` passed `self.num_iterations_used` which defaults to 0 when
no HPO and no early-stopping. LightGBM rejects 0 iterations. Fixed to pass
`None` when `num_iterations_used <= 0`.

## Features

### `ProbabilisticRegressionModel.predict_distribution`
Exposes the full mixture-of-Gaussians (per-class mu, sigma weighted by classifier
probs). On the bimodal toy, mixture-NLL is ~7% better than collapsed-Gaussian NLL.

### `DepthRegularization.COST_AWARE_ELBO`
FlexNN prior logits = `linspace(3,1)` minus `lambda * normalised_depth_cost`. Test
shows the cost-aware variant selects shallower mean depth than standard ELBO at
high lambda.

### FT-Transformer training recipe
Pre-norm + AdamW + warmup + mini-batch + grad-clip + validation early-stopping.

### Synthetic datasets B1-B6
`automl_package/utils/synthetic_datasets.py`. Seeded generators for gravitational
inverse (B1), oscillator phase ambiguity (B2), two-phase transition (B3),
latent-subpopulation heteroscedasticity (B4), sparse-importance (B5), null (B6).
Dumped to `tests/fixtures/synthetic/` as `.npz`.

### Markdown per-run report
`Metrics.save_markdown_report` writes a self-contained `{partition}_report.md`
with dataset stats, metrics table, and links to all generated plots.

### Head-structure diagnostic
`automl_package/utils/head_diagnostics.py` returns a per-config report: outer
heads' slope signs (mirror_ok), middle head flatness, mean-separation. Writes
a CSV alongside ablation runs to correlate learned structure with MSE.

### California Housing per-model timeout
`uci_benchmark.py` wraps each fit in `ThreadPoolExecutor` with 5-minute deadline.
Timeout logs a warning and continues; note that thread cannot be killed (tracked
in `memory/project_future_work.md`).

### Photo-z domain metric + LocallyAdaptive conformal tests
`tests/test_photoz_domain_metrics.py` covers sigma_MAD, bias, outlier fraction,
CDE loss, and heteroscedastic marginal coverage + locally-adaptive width scaling.

## Scripts authored (runs queued)

- `automl_package/examples/noise_robustness_benchmark.py` (I4, ran; results below)
- `automl_package/examples/head_structure_diagnostic.py` (I2)
- `automl_package/examples/flex_nn_depth_viz.py` (I3)
- `automl_package/examples/gumbel_elbo_retest.py` (I8)
- `automl_package/examples/sep_heads_vs_single_final.py` (I1)
- `automl_package/examples/flex_nn_ablation.py` (Paper B primary table)
- `automl_package/examples/probreg_ablation.py` (Paper A primary table)
- `automl_package/examples/multi_seed_sweep.py` (seeds 0..4, three models)
- `automl_package/examples/hpo_sweep.py` (Optuna, N_TRIALS=50, UCI-Yacht default)
- `automl_package/examples/probreg_mixture_eval.py` (mixture vs collapsed)

## Noise robustness benchmark results (first pass)

See `automl_package/examples/noise_robustness_results/`. Highlights:

- sigma=0.05 (low): best ClassReg k=15, ProbReg dyn_k mean=2 got MSE=0.012 (beats ClassReg best of 0.026, beats NN's 0.005 it matched).
- sigma=0.3: best ClassReg k=15, ProbReg dyn_k still mean=2.
- sigma=1.0: best ClassReg k=2 (classification bottleneck wins). ProbReg dyn_k
  mean=2 at MSE=1.10 competitive with NN=1.10.

Follow-up: ProbReg dyn_k with ELBO regularisation is pinned to k=2 regardless of
noise. Need a re-run with `NClassesRegularization.NONE` to check whether
dyn_k actually adapts when the ELBO prior is lifted. Note: this is now meaningful
because the FlexibleNN n_predictor bug (which froze the predictor) does NOT apply
to ProbReg — ProbReg's n_classes_predictor was always in the main optimizer.
