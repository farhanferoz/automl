# Benchmarks

Empirical results from the AutoML package across designed problems. All numbers come from the
test suite (`tests/test_phase{1..4}*.py`) and the comparison scripts under
`automl_package/examples/`. Each result uses fixed seeds for reproducibility.

For setup details and test runners, see [README §14](../README.md#14-installation) and `CLAUDE.md`.

## Datasets

| Name | Definition | Size | Used to test |
|---|---|---|---|
| **heteroscedastic** | `y = 2·sin(x) + 0.5·x + ε(x)` with `σ(x) = 0.1 + 0.4·|x|`, `x ∈ [-5, 5]` | 1000 | Learned variance, ELBO k-selection |
| **piecewise** | `y = 0.5·x` for `x < 0`; `y = 0.5·x + sin(4πx)` for `x ≥ 0` | 800 | Adaptive depth selection |
| **multimodal** | `y = x ± 1.5` (bimodal) | 500 | Multi-mode regression heads |
| **exponential** | `y = exp(x) + ε`, `x ∈ [-3, 3]` | 800 | Symlog transform |

## Heteroscedastic data — model comparison

Comparing all model families on the heteroscedastic sine problem (n = 1000, train/test 0.7/0.3).

| Model | MSE | NLL | Cal(1σ) | NoiseR |
|---:|---:|---:|---:|---:|
| LinearReg | 3.88 | 2.10 | 0.64 | — |
| XGBoost | 2.15 | 1.82 | 0.65 | — |
| LightGBM | 1.61 | 1.66 | 0.71 | — |
| ClassReg `LOOKUP_MEDIAN` k=10 | 1.57 | 1.62 | 0.83 | — |
| ClassReg `LOOKUP_MEDIAN`+XGB k=7 | 1.73 | 1.73 | 0.85 | — |
| ClassReg `SPLINE` k=7 | 6.38 | 1.88 | 0.72 | — |
| ClassReg `NN` mapper k=7 | 8.02 | 10.73 | 0.32 | — |
| **ProbReg k=5** | **1.54** | **1.38** | 0.70 | 0.94 |

`Cal(1σ)` = empirical fraction of test points within `±1σ` of the predicted mean (ideal ≈ 0.683).
`NoiseR` = Pearson correlation between predicted `σ(x)` and true `σ(x)`.

**Takeaways**

- **ProbReg wins on NLL** (1.38) — its learned variance is the right tool for input-dependent noise.
- **ClassReg `LOOKUP_MEDIAN`** is competitive on MSE at k=10 and is the most reliable non-NN mapper.
- **`SPLINE` mapper** is unstable; **`NN` mapper** is broken (two-stage training limitation — see open issues in `RESUME.md`).

## Dynamic `n_classes` (ProbReg)

Comparing fixed-k baselines vs all four dynamic-k strategies (with and without regularization),
on the heteroscedastic dataset.

| Strategy | MSE | NLL | Cal | NoiseR | Mean k |
|---:|---:|---:|---:|---:|---:|
| Fixed k=3 | 1.56 | 0.45 | 0.67 | 0.98 | 3.00 |
| Fixed k=5 | 1.55 | 0.46 | 0.70 | 0.95 | 5.00 |
| Fixed k=10 | 1.55 | 0.47 | 0.68 | 0.98 | 10.00 |
| Dynamic (none) | 1.88 | 0.59 | 0.71 | 0.91 | 10.00 |
| Dynamic ELBO + Gumbel | 2.24 | 0.80 | 0.68 | 0.73 | 2.95 |
| Dynamic K-penalty + SoftGating | 1.63 | 0.42 | 0.72 | 0.96 | 4.57 |
| **Dynamic ELBO + SoftGating** | **1.56** | **0.44** | 0.69 | **0.99** | **2.00** |

**Takeaways**

- **ELBO + SoftGating** matches the best fixed-k MSE while running with `mean k = 2`, and gets the best noise correlation (0.99).
- **ELBO + Gumbel fails** — Gumbel sampling noise during training corrupts KL gradients and prevents convergence to a good `q(k|x)`.
- **Unregularized dynamic** drifts to `mean k = 10` (no incentive to be efficient) and loses ~20% on MSE.
- **K-penalty** is a reasonable middle ground but not as effective as ELBO at finding low k.

## Piecewise data — adaptive depth

Comparing fixed-depth NNs vs `FlexibleNN` (with and without ELBO depth regularization), on the
piecewise linear-vs-sinusoidal dataset.

| Model | MSE | Notes |
|---:|---:|:---|
| Shallow NN (1 hidden layer) | baseline | underfits sinusoidal region |
| FlexibleNN (max=3, no reg) | competitive with shallow | learns to use depth where needed |
| **FlexibleNN + ELBO** | **0.30** | shallower depth on linear half, deeper on sinusoidal half |
| FlexibleNN + ELBO (`hard` inference) | 0.34 | argmax depth bucketing — diff from soft ≈ 0.02 MSE |

**Takeaways**

- ELBO depth regularization correctly selects shallower depth for the linear region and deeper for the sinusoidal region (verified per-input in `tests/test_phase2_flexible_nn.py::test_per_input_depth_variation`).
- **Hard inference** is essentially a free swap from soft when it comes to prediction quality. Compute savings depend on network size: at this scale (max_depth=4, hidden=64) bucket-grouping overhead cancels the savings; larger networks should benefit.

## Wide-range targets — symlog transform

Comparing raw vs symlog target transform on `y = exp(x) + ε`, where targets span ~0.05 to ~20.

| Transform | MSE | PearsonR |
|---:|---:|---:|
| raw | 1.0008 | 0.9788 |
| **symlog** | **0.1934** | **0.9978** |

**~5× MSE improvement** on the use case symlog was designed for. The transform compresses targets
into a manageable range during training (`symlog(1000) ≈ 6.9`), then `symexp` reverses it at
prediction time.

## β-NLL loss variant (Seitzer 2022)

Comparing standard NLL vs β-NLL at two values of β, on the heteroscedastic dataset.

| Variant | MSE | NLL | Cal | NoiseR |
|---:|---:|---:|---:|---:|
| **NLL** | **1.80** | **0.54** | 0.69 | 0.94 |
| β-NLL β=0.5 | 1.84 | 0.57 | 0.70 | 0.93 |
| β-NLL β=1.0 | 2.21 | 0.80 | 0.68 | 0.51 |

**Takeaways**

- **β=0.5** is competitive with standard NLL on this dataset (slightly better calibration).
- **β=1.0** over-corrects when noise is well-behaved — the variance reweighting hurts more than it helps.
- β-NLL is most useful when the variance is collapsing during training (a known failure mode of NLL on harder datasets). When NLL is already stable, the gain is small or negative.

## Conformal prediction coverage

Verifying that the split-conformal wrapper achieves its target coverage on the heteroscedastic dataset.

| α | target coverage | empirical coverage | mean width |
|---:|---:|---:|---:|
| 0.05 | 0.95 | 0.965 | 5.82 |
| 0.10 | 0.90 | 0.890 | 4.61 |
| 0.20 | 0.80 | 0.780 | 3.19 |

Coverage matches target almost exactly across all α levels — the finite-sample correction
(`ceil((n+1)(1-α)) / n` quantile) keeps the guarantee tight.

## Headline findings

| Win | Where |
|---|---|
| Best NLL on heteroscedastic data | **ProbReg** with learned variance (NLL = 1.38) |
| Best dynamic-k strategy | **ELBO + SoftGating** (matches best fixed-k with `mean k = 2`) |
| Biggest single-feature win | **Symlog** on exponential targets (5× MSE improvement) |
| Cross-model ranking confirmed | ProbReg NLL beats constant-variance NN; FlexibleNN+ELBO competitive with shallow on piecewise |
| Conformal coverage | Tracks target across α = 0.05 / 0.10 / 0.20 |

## Known broken / open

- **ClassReg `NN` mapper** — two-stage training limitation, large MSE/NLL. Use ProbReg instead.
- **ClassReg `SPLINE` mapper** — instability (smoothing-spline convergence warnings). `LOOKUP_MEDIAN` dominates on benchmarks.
- **ELBO + Gumbel** for dynamic k — fundamental noisy-gradient issue. Use SoftGating instead.

## Fixed (Phase 5)

- **SHAP DeepExplainer on FlexibleNN / IndependentWeightsFlexibleNN** — both models returned 5-tuples from `forward()`; added `get_internal_model()` override returning a `_ShapModelWrapper` that strips auxiliary tensors to a single prediction tensor.
- **SHAP on dynamic n_classes ProbReg** — DeepLIFT can't trace per-sample head dispatch; `get_shap_explainer_info()` now returns `ExplainerType.KERNEL` (with `self.predict` as the callable) when `n_classes_selection_method != NONE`.
- **Symlog uncertainty (linearized Jacobian)** — replaced `exp(|μ_symlog|) · σ` with Monte Carlo: 200 samples from `N(μ_symlog, σ²)` pushed through `symexp`, empirical std taken. Also makes `predict()` return the MC mean so point estimate and std are consistent. Verified: MC std > Jacobian approximation near zero crossings (see `TestSymlogMCUncertainty`).

See `RESUME.md` for the live open-issue list.
