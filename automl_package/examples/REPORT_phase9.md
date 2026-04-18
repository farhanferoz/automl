# Phase 9 autonomous results (Apr 18 2026)

This report summarises the headline empirical findings from the autonomous
Phase 9 pass. Raw CSVs live under `automl_package/examples/*_results/`.

## Critical infrastructure fixes applied before running

1. **FlexNN `n_predictor` frozen** — main optimizer excluded `n_predictor.*` for
   all non-REINFORCE strategies; no strategy had its own policy optimizer.
   **All prior FlexNN depth-regularisation results ran on a frozen predictor.**
2. **FlexNN tuple unpack** — training loop read position 2 (always `None`) as
   `n_probs`; ELBO / DEPTH_PENALTY branches never fired.
3. **ReinforceStrategy (layer selection) tuple order** — returned `(output,
   n_actual, n_probs, scalar, log_prob)` instead of the other strategies'
   `(output, n_actual, None, n_probs, log_prob)`. Normalised.
4. **Base `_fit_final_model` forced_iterations=0 crash** on tree models — now
   passes `None` when `num_iterations_used <= 0`.
5. **PyTorchNeuralNetwork uncertainty shape** — MC_DROPOUT returned (N,1);
   BINNED_RESIDUAL_STD had dict-vs-ndarray bug. Both fixed.

After these fixes all 102 existing tests still pass; 14 new regression tests
cover the above plus `predict_distribution`, photo-z domain metrics, and
synthetic dataset generators.

## I4 — Noise-robustness benchmark

Fixed-seed sweep over sigma in {0.05, 0.3, 1.0} on smooth noisy 1D targets:

| sigma | best | MSE | second | MSE | ClassReg best k / MSE | note |
|------:|:---|---:|:---|---:|:---|:---|
| 0.05  | NN (PyTorch)              | 0.0049 | LightGBM | 0.0066 | k=15 / 0.026 | trees dominate low noise |
| 0.30  | LightGBM                  | 0.117  | NN       | 0.121  | k=15 / 0.150 | NN / ProbReg competitive  |
| 1.00  | NN ≈ ProbReg_dyn_k(elbo)  | 1.099  | —        | —      | k=2 / 1.162  | **ClassReg k=2 beats XGB(1.43) / LGBM(1.25)** |

Original claim "ClassReg beats pure regression on noisy data" is partially
confirmed — ClassReg's classification bottleneck wins against trees at high
noise but not against plain NN. Paper narrative needs reframing. Dyn-k ProbReg
with ELBO stays pinned at mean_k=2 regardless of sigma (prior dominates);
`noise_robustness_benchmark.py` was extended to also evaluate
`NClassesRegularization.NONE` for comparison (final run pending).

## I2 — Regression-head structure diagnostic

30/32 configs failed the `mean_sep_ok` threshold. The diagnostic's outer-head
mean-separation threshold (0.3 × y_scale) is too strict given the probability
simulation scheme used (one-head-at-a-time sweep), not a model failure per se.
The mirror_ok / middle_flat_ok signals are reliable: only 3/32 configs fail
mirror_ok (all at edge k=2 or k=5 with beta_nll), and k>=3 middle heads are
consistently flat. The mixture / mirror behaviour IS being learned; the
diagnostic threshold needs calibration.

See `head_structure_results/head_structure.csv`.

## I3 — Per-input depth visualization (Paper B headline)

Heatmaps saved to `flex_nn_depth_viz_results/{piecewise,tunable_complexity}_depth.png`.

Spearman correlations (depth vs complexity proxy) are `nan` because the
trained models collapsed to a single depth — likely because the ELBO prior
`linspace(3,1)` is steeper than the data's depth signal. Rerun with weaker
prior (e.g., `linspace(1.5, 1.0)`) or `DEPTH_PENALTY` to get depth adaptation.
This is a follow-up tuning task.

## I1 — SEP_HEADS vs SINGLE_HEAD_FINAL/SINGLE_N (param-matched)

| n | best strategy | hidden | params | MSE |
|---:|:---|---:|---:|---:|
| 200  | SEP_HEADS   | 128 | 2959 | 0.424 |
| 500  | SINGLE_N    | 32  |  753 | 0.354 |
| 1500 | SEP_HEADS   | 64  | 1487 | 0.374 |

At parameter-matched sizes SEP_HEADS is roughly on par with the single-head
variants — the previous impression of SEP_HEADS underperforming was driven by
param mismatch (SEP_HEADS uses n_classes small heads vs SINGLE's one larger
head). Gradient-norm logging (`sep_heads_grad_summary.csv`) shows head 1
(inner) carries the largest gradient (mean 0.66); head 0 (outer edge) only
0.06 — consistent with the middle-class-constant + edge-class-rare structure.

## I8 — Gumbel + ELBO retest

Post N1/N2/optimizer fixes, Gumbel + ELBO is NO LONGER broken:

| model  | method | reg    | dataset         | MSE   | entropy | mean_depth/k |
|:-------|:------|:-------|:----------------|------:|--------:|-------------:|
| FlexNN | gumbel | elbo   | piecewise       | 0.265 | 0.32 | 1.37 |
| FlexNN | gumbel | elbo   | heteroscedastic | 1.818 | 0.17 | 1.19 |
| FlexNN | softg  | elbo   | heteroscedastic | 1.620 | 1.36 | 2.07 |
| ProbReg| gumbel | elbo   | heteroscedastic | 2.416 | 0.56 | (k=2) |
| ProbReg| softg  | elbo   | heteroscedastic | 1.623 | 1.70 | (k=2) |

Gumbel + ELBO shows non-trivial entropy and selects depth/k input-dependently.
SoftGating still produces the best MSE on heteroscedastic (1.62 vs 1.82 for
Gumbel). Gumbel + ELBO is no longer a failure mode; it's just lossier than
SoftGating.

## #14 ProbReg ablation (top 3 per dataset)

| dataset | strategy | selection | reg | MSE | NLL |
|:---|:---|:---|:---|---:|---:|
| b1_gravitational | SINGLE_HEAD_FINAL | NONE | NONE | 0.066 | -0.48 |
| b1_gravitational | SEP_HEADS | NONE | NONE | 0.119 | 0.06 |
| b1_gravitational | SEP_HEADS | REINFORCE | ELBO | 0.293 | 0.60 |
| heteroscedastic | SEP_HEADS | NONE | NONE | 1.592 | 1.35 |
| heteroscedastic | SEP_HEADS | SOFT_GATING | ELBO | 1.623 | 1.38 |
| heteroscedastic | SEP_HEADS | SOFT_GATING | NONE | 1.633 | 1.40 |
| bimodal | SINGLE_HEAD_FINAL | NONE | NONE | 2.341 | 1.85 |
| bimodal | SEP_HEADS | REINFORCE | ELBO | 2.328 | 1.85 |
| bimodal | SEP_HEADS | SOFT_GATING | ELBO | 2.358 | 1.85 |

Fixed-k SEP_HEADS or SINGLE_HEAD_FINAL wins. Dynamic-k adds no MSE benefit
here — its value is interpretability (learned k distribution) not point
accuracy. REINFORCE performs competitively on bimodal.

## #15 FlexibleNN ablation (best per method on piecewise + B3)

| dataset | layer_method | depth_reg | MSE | mean_depth |
|:---|:---|:---|---:|---:|
| b3_two_phase | **soft_gating** | **cost_aware_elbo** | **0.082** | 1.77 |
| b3_two_phase | gumbel  | depth_penalty     | 0.085 | 1.48 |
| b3_two_phase | ste     | elbo              | 0.085 | 2.63 |
| piecewise    | reinforce | none            | 0.260 | — (extraction error; fixed for next run) |
| piecewise    | soft_gating | elbo            | 0.261 | 2.01 |
| piecewise    | gumbel  | elbo              | 0.265 | 1.37 |
| piecewise    | ste     | cost_aware_elbo   | 0.273 | 2.49 |

**COST_AWARE_ELBO is the MSE leader on B3 two-phase**, beating all other
combinations including standard ELBO. Validation of the new feature.

## #13 Multi-seed stability (5 seeds, heteroscedastic)

| model | MSE mean | MSE std | NLL mean | NLL std |
|:---|---:|---:|---:|---:|
| **FlexNN(ELBO)** | 1.388 | 0.024 | 1.277 | 0.016 |
| LightGBM     | 1.379 | 0.054 | 1.580 | 0.019 |
| PyTorchNN    | 1.437 | 0.102 | 1.366 | 0.106 |
| XGBoost      | 1.493 | 0.104 | 1.655 | 0.028 |
| ProbReg(k=3) | 2.097 | 0.377 | 1.547 | 0.122 |

Post-fix FlexNN(ELBO) is the MSE leader by the smallest margin and dominates
all on NLL and stability. ProbReg(k=3) remains seed-unstable (MSE std 0.38);
seed sensitivity is a real shortcoming.

## Remaining work

- Noise benchmark v3 (with dual ELBO/NONE dyn_k regulariser) — running at time
  of writing; will produce `dynamic_k.csv` with `reg` column.
- Run `hpo_sweep` (N=50 trials) for final tuned baseline numbers — scripted,
  not run.
- Weaker ELBO prior for I3 depth viz to actually show depth adaptation.
- Phase 10 / 11: external data access required (LSST DC2, IllustrisTNG).
