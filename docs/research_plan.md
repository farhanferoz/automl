# Research Plan — Probabilistic Regression & Flexible Neural Networks

**Date:** 2026-04-12
**Status:** Draft for review (revised with user feedback)
**Scope:** Path from current benchmark suite to publication-quality evidence for two papers.
**Companion docs:** `docs/architecture_analysis.md` (SOTA comparison + historical bug audit), `docs/benchmarks.md` (current empirical results), `docs/implementation_plan.md` (completed Phase 1–5 roadmap), `docs/mathematical_guide.tex/.pdf` (complete mathematical specification of all models).

This plan addresses six questions:

1. Are there bugs or logical mistakes? Are the theoretical foundations sound?
2. Are the current toy problems sufficient? What physically-motivated benchmarks should replace/extend them?
3. What additional metrics does the evaluation suite need?
4. What comparison models are missing?
5. Should we support transformer-based models?
6. Which astrophysics application should we target for the headline experiment?

It ends with a paper strategy and an execution roadmap.

---

## Executive Summary

**Verdict on current state.** The two research models are theoretically sound *after* the Phase 1–5 fixes. The Probabilistic Regression law-of-total-variance decomposition is correct, the SEPARATE_HEADS + monotonic-constraints path via separate mean/log_var heads is now the cleanest formulation, and ELBO-based dynamic-k with SoftGating empirically achieves the best of both worlds (fixed-k MSE with mean k ≈ 2). FlexibleNN’s ELBO depth regularization works as advertised on the piecewise dataset.

**What still needs to happen before a paper is defensible.**

- **Seven new bugs** (verified against live code, not hypothetical). Some are correctness-affecting (n_classes STE gradient path), some are reporting-affecting (non-standard ECE, β-NLL missing constant), some are latent (NoneStrategy shape mismatch). See §1.
- **The current benchmark suite is inadequate** for a methods paper. Four small synthetic 1-D datasets (heteroscedastic sine, piecewise, bimodal, exponential) prove a *mechanism* works but can’t support claims like “calibrated uncertainty on real-world tabular data.” Reviewers will demand UCI benchmarks and one realistic domain problem. See §2.
- **Metrics are incomplete** — no CRPS, no Winkler score, no sharpness, no PIT histogram / proper calibration curve, no interval-width-vs-coverage tradeoff, no sharpness-calibration plot. See §3.
- **Visualizations are inadequate for publication.** Current plots are basic regression charts. Missing: PIT histograms, reliability diagrams (predicted vs observed quantile), sharpness-calibration scatter plots, critical-difference diagrams (Nemenyi test), per-input uncertainty heatmaps, interval-width-vs-coverage curves, and residual-vs-predicted-σ diagnostics. See §3.8.
- **Comparison set is too narrow** — no NGBoost, no MDN, no quantile regression, no deep ensembles, no Gaussian process baseline, no Laplace-approximation Bayesian NN. Without these, you can’t claim the models are state-of-the-art for tabular probabilistic regression. See §4.
- **Transformer support is a trap** for the tabular use case. The right answer is FT-Transformer and TabPFN, not a sequence transformer, and even those should be added as baselines rather than supported as first-class architectures. See §5.
- **Astrophysics application should be photometric redshift estimation** (primary) with **galaxy cluster mass** as a secondary demonstration. Both are tabular, have well-curated public data (SDSS / DES / LSST PZ DC2 testbed; Planck/SPT cluster catalogs + IllustrisTNG/MDPL2 simulated training sets), are heteroscedastic and partially multimodal, and have a rich published baseline literature. See §6.

**Two papers, two models.** These are independent contributions that warrant separate publications:

- **Paper A (ProbReg):** *”Classification-Based Probabilistic Regression via the Law of Total Variance.”* Claim: learned per-class variance combined via discrete mixture decomposition dominates other tabular uncertainty estimators (NGBoost, MDN, Deep Ensembles) on heteroscedastic, multi-modal problems. Dynamic k (ELBO + SoftGating) provides automatic bin complexity. Astrophysics headline: **photometric redshift estimation** (multimodal posteriors from color-redshift degeneracies; §6.2).

- **Paper B (FlexibleNN):** *”Input-Dependent Depth Selection for Compute-Adaptive Neural Networks.”* Claim: ELBO-regularized depth gating learns to allocate more layers to complex inputs and fewer to simple inputs, yielding compute savings at inference without accuracy loss. Astrophysics headline: **galaxy cluster mass estimation** (complexity varies with cluster richness and dynamical state; §6.3). Future work: applying the same depth-gating mechanism to FT-Transformer blocks (§5).

---

## 1. Code Audit (Q1)

The companion document `docs/architecture_analysis.md` catalogued 11 historical bugs (all fixed in Phases 1–5). This section reports **new findings verified against the code as of HEAD `597a416`**.

### 1.1 Newly discovered bugs

| # | Bug | Severity | Location | Effect |
|---|-----|----------|----------|--------|
| N1 | FlexibleNN ELBO depth prior not normalized | HIGH | `flexible_neural_network.py:250`, `independent_weights_flexible_neural_network.py:303` | Steepness scales with `max_hidden_layers` |
| N2 | n_classes STE gradient path is broken | HIGH | `selection_strategies/base_selection_strategy.py:82-111` (used by `n_classes_strategies.py:111`) | `n_classes_predictor` receives no gradient |
| N3 | `calculate_ece` is a non-standard formulation | MEDIUM | `utils/metrics.py:155-173` | Reported ECE ≠ literature ECE |
| N4 | `beta_nll_loss` omits `log(2π)` constant | LOW (cosmetic) | `utils/losses.py:21-39` | NLL values inconsistent with `nll_loss` |
| N5 | Tree-model Gaussian NLL objective uses observed Hessian | MEDIUM | `utils/losses.py:51-80` | Slow log_variance convergence at init |
| N6 | `build_model` called twice in `_fit_single` | LOW | `models/base_pytorch.py:127-138` | Wasted rebuild; discards first model |
| N7 | `NoneStrategy` (layer selection) n_probs has wrong shape | MEDIUM (latent) | `selection_strategies/layer_selection_strategies.py:43` | Crashes if combined with `DepthRegularization.ELBO/DEPTH_PENALTY` |

**Do these bugs invalidate existing results?** No. N1 affects FlexibleNN ELBO regularization strength (not ProbReg — already fixed there). N2 only affects STE strategy (SoftGating, our best, is unaffected). N3/N4 are reporting issues (absolute values shift, rankings don't). N5 affects tree NLL objectives only. N6 wastes compute but doesn't change outputs. N7 is latent (only triggers with specific unused strategy+regularization combos). **Rankings and conclusions in benchmarks.md remain valid.** Fixes are needed before paper submission, but no retractions required.

**N1 — FlexibleNN ELBO depth prior.** In `probabilistic_regression.py:214` the fix `torch.linspace(3.0, 1.0, n_modes)` was applied so that the prior-over-k has the same steepness regardless of `n_modes`. The docstring explicitly explains the rationale (“without normalization, arange(n_modes, 0, -1) creates a prior whose steepness scales with n_modes…”). The exact same pattern is still broken in both FlexibleNN variants:

```python
# flexible_neural_network.py:250 — still uses unnormalized arange
depth_prior_logits = torch.arange(self.max_hidden_layers, 0, -1, dtype=torch.float, device=...)
depth_prior = torch.distributions.Categorical(logits=depth_prior_logits)
```

**Fix:** change both occurrences to `torch.linspace(3.0, 1.0, self.max_hidden_layers, device=...)`. This single change will matter for any `max_hidden_layers ≥ 4` and is directly relevant to the headline benchmark claim that ELBO produces correct depth selection.

**N2 — n_classes STE gradient path.** `n_classes_strategies.SteStrategy.forward` (line 111) delegates to `_hard_selection_logic` in `base_selection_strategy.py`. That helper uses `torch.argmax(mode_selection_one_hot, dim=1)` → `selected_indices = torch.where(selected_k_indices == i)[0]` → `final_predictions_contribution[selected_indices] = predictions_for_mode`. The problem is that **indexed assignment never uses `mode_selection_one_hot` as a multiplicative factor** — the gradient path from the regression loss back to the `n_classes_predictor` is severed. Contrast with FlexibleNN’s layer-selection `SteStrategy` (layer_selection_strategies.py:176-180), which correctly uses the weighted-sum pattern:

```python
for i in range(self.model.max_hidden_layers):
    prob = n_probs[:, i]
    hidden_rep = hidden_representations[i]
    output_for_n = self.model.model.output_layer(hidden_rep)
    aggregated_output += prob.unsqueeze(1) * output_for_n
```

The multiplicative `prob * output` preserves STE gradients. The n_classes version should be rewritten to the same pattern (accepting the cost of running every `k ∈ [2, max_n]` during training). Historical bug #6 was exactly this pattern for layer selection and was fixed; the fix was not carried over when dynamic n_classes was implemented.

**Severity note:** this partly explains why `ELBO + Gumbel` was reported “broken” in `docs/benchmarks.md`. Gumbel has the same gradient-through-hard-sampling requirement as STE, and while `GumbelSoftmaxStrategy` for n_classes uses `_weighted_average_logic` (which IS correct), `SteStrategy` does not. **Any result for `STE` in the n_classes table is invalid** until this is fixed.

**N3 — ECE formulation.** The current implementation:

```python
# utils/metrics.py:155
z_scores = np.abs(self.y_true - self.y_pred) / np.maximum(self.y_std, 1e-9)
confidences = 2 * (1 - norm.cdf(z_scores))          # two-sided p-value, NOT coverage prob
# ... bins points by "confidence" then checks fraction with z ≤ 1
accuracy_in_bin = np.mean(z_scores[in_bin] <= 1)     # fixed 1σ decision rule
```

This is not ECE as defined for regression calibration (Kuleshov, Fenner & Ermon, 2018). The standard formulation uses the **Probability Integral Transform (PIT)**: for each test point compute `p_i = Φ((y_true_i − μ_i)/σ_i)`, then check whether the empirical CDF of `{p_i}` matches the uniform CDF. The calibration curve plots target coverage (0, 0.1, ..., 1) against empirical coverage; ECE is then the area between that curve and the diagonal.

**Effect:** the number reported as `ece` in the benchmark table is not comparable to anything in the literature, and the bin-wise accuracy rule (`z ≤ 1`, fixed) makes the metric degenerate for all but one z threshold.

**Fix:** rewrite to the PIT-based calibration curve (§3 covers the API).

**N4 — β-NLL missing constant.** `nll_loss` includes `log(2π)` (line 17), `beta_nll_loss` does not (line 37). For optimization this is irrelevant (constants don’t affect gradients), but for *reporting* NLL in a paper this means the two losses produce numbers on different scales — users reading `benchmarks.md` will compare them and draw wrong conclusions about loss scale. Add the constant to `beta_nll_loss` for consistency.

**N5 — Observed vs Fisher Hessian for tree boosters.** `tree_model_gaussian_nll_objective` currently uses:

```python
hess_log_var = 0.5 * (((y_true - mean) ** 2) / variance)   # observed Hessian
```

The Fisher information for `log_var` under a Gaussian is `E[0.5 · ((y−μ)/σ)²] = 0.5` — a constant. At initialization, when `(y − mean)²` is large, the observed Hessian is also large, so the booster’s Newton step `−grad/hess` in the log_var direction is tiny. Using Fisher information (constant 0.5) eliminates that init-phase damping and is the standard choice for natural-gradient / Fisher-scoring methods. The XGBoost / LightGBM custom-objective convention accepts scalar Hessians, so this is a one-line fix:

```python
hess_log_var = np.full_like(y_true, 0.5)   # Fisher information for Gaussian log_var
```

Verify empirically that XGB+NLL actually improves on the heteroscedastic benchmark; if it does, this becomes a third probabilistic tree baseline alongside LightGBM+NLL.

**N6 — Double `build_model`.** In `base_pytorch.py` the `_fit_single` code path is:

```python
if self.input_size != new_input_size:
    self.input_size = new_input_size
    self.build_model()        # L127
if self.model is None:
    self.input_size = new_input_size
    self.build_model()        # L131
...
self.build_model()            # L138 — unconditional
```

Line 138 always runs; it discards whatever was built on L127/131. Delete the conditional duplicates or delete the unconditional one (depending on intent — most likely the unconditional one is authoritative and the conditional block is dead code from a refactor). Historical bug #10 was the same class of mistake in `independent_weights_flexible_neural_network.py`.

**N7 — Layer `NoneStrategy` shape mismatch.** `NoneStrategy.forward` (L43) creates `n_probs = torch.zeros(batch, max_hidden_layers + 1, ...)`. All other layer strategies produce `(batch, max_hidden_layers)` (softmax/gumbel of the `n_predictor` output). If a user combines `layer_selection_method=NONE` with `depth_regularization=ELBO` (or `DEPTH_PENALTY`), the main training loop’s broadcast:

```python
depth_indices = torch.arange(1, self.max_hidden_layers + 1, ...)  # shape (max,)
expected_depth = torch.sum(n_probs * depth_indices, dim=1)         # (B, max+1) * (max,) → error
```

…will either crash or silently produce wrong numbers (torch broadcasting may align the last dim and include a spurious column). This is latent because the combination is unusual, but the shape contract should be enforced. Fix: make `NoneStrategy` produce `(B, max)` with `n_probs[:, -1] = 1.0` (one-hot on the max depth), or assert the combination is disallowed.

### 1.2 Theoretical foundations — are the models sound?

**Probabilistic Regression — YES.**

- **Law of Total Variance.** The implementation in `utils/pytorch_utils.apply_law_of_total_variance` is correct: `E[Y] = Σ_i P(c_i) · μ_i`; `Var(Y) = Σ_i P(c_i) · σ²_i + Σ_i P(c_i) · (μ_i − E[Y])²`. This is the textbook decomposition — aleatoric + epistemic-like split across discrete classes. Reviewers will accept it immediately.
- **Probability bottleneck is regularization, not limitation.** The SEPARATE_HEADS design where each head only sees its own `P(class_i)` is deliberate — the classifier acts as a nonlinear feature extractor projecting onto a probability simplex, and each regression head learns a scalar-to-scalar function (`P(class_i) → μ_i, log σ²_i`). This IS the novelty of the architecture and should be framed that way in the paper.
- **Three regression strategies exist but only SEPARATE_HEADS has been benchmarked.** The code also supports `SINGLE_HEAD_N_OUTPUTS` (one shared head producing N outputs) and `SINGLE_HEAD_FINAL_OUTPUT` (shared head with a final scalar output). These have NOT been tested in the current benchmark suite. A systematic ablation across all three is required for the ProbReg paper — it's a natural table showing that SEPARATE_HEADS' per-class specialization is the source of the performance gain. Add to §2.3 ablation sweep.
- **Monotonic constraints.** The new `is_probabilistic_monotonic` path in `regression_heads.py` uses separate mean and variance heads; monotonicity is applied only to the mean. This is correct — requiring a monotonic standard deviation is rarely meaningful. Document this choice explicitly in the paper.
- **ELBO k-selection.** Treating `k` as a discrete latent variable with a prior favoring small k, and computing KL(q(k|x) ‖ p(k)) as an Occam penalty, is a clean variational formulation. The normalization to `linspace(3, 1, n_modes)` is a sensible heuristic (constant steepness regardless of `n_modes`); a more defensible alternative is a geometric prior `p(k) ∝ λ^k` with `λ ∈ (0, 1)` as a hyperparameter — easier to reason about theoretically.
- **Monte-Carlo symlog uncertainty.** `_symlog_mc_moments` (replaces linearized Jacobian) samples `ε ~ N(μ_s, σ²_s)`, applies `symexp`, and reports the empirical std. This is exact up to Monte-Carlo variance and handles non-linearity near zero correctly. Good.
- **β-NLL (Seitzer 2022).** Variance-weighted NLL prevents variance collapse. Implementation matches the paper (sans the missing constant, bug N4).
- **Conformal prediction.** `ConformalWrapper` implements split-conformal with finite-sample correction `ceil((n+1)(1−α))/n`. Correct. Only supports **marginal** coverage (one interval width for all test points); §3 covers locally-adaptive conformal as a future addition.

**Systematic loss/configuration audit — what works and what doesn’t.**

The code supports several losses and optimization paths. Current testing status:

| Configuration | Tested? | Theoretically sound? | Status |
|---:|---:|---:|:---|
| NLL (standard Gaussian) | Yes | Yes | **Works.** Primary loss, used in all benchmarks. |
| β-NLL (Seitzer 2022) β=0.5 | Yes | Yes | **Works.** Competitive with NLL; slightly better calibration on heteroscedastic data. |
| β-NLL β=1.0 | Yes | Yes | **Over-corrects** when noise is well-behaved — variance reweighting hurts. |
| Cross-entropy + NLL joint loss | Yes (implicitly) | Yes, but **untuned** | **Works but λ_ce is implicit (=1.0).** Needs sweep or learned weighting (Kendall & Gal 2018). |
| GRADIENT_STOP optimization | Partially | Yes | Detaches classifier probabilities before regression — **not benchmarked vs joint**. Needs ablation. |
| Symlog target transform | Yes | Yes | **Works.** 5× MSE improvement on exponential targets. MC uncertainty conversion verified. |
| SEPARATE_HEADS regression | Yes | Yes | **Works.** Only tested strategy in benchmarks. |
| SINGLE_HEAD_N_OUTPUTS | No | Yes | **Untested.** Needs inclusion in ablation sweep. |
| SINGLE_HEAD_FINAL_OUTPUT | No | Yes | **Untested.** Needs inclusion in ablation sweep. |
| Monotonic constraints | Partially | Yes | Applied to mean head only (correct). Not benchmarked against unconstrained on same data. |

**Action items for ProbReg paper:**
1. Full ablation: 3 regression strategies × {NLL, β-NLL β=0.5} × {with/without symlog} × {joint, GRADIENT_STOP}
2. λ_ce sweep: test λ_ce ∈ {0.1, 0.5, 1.0, 2.0} plus learned uncertainty weighting (Kendall & Gal 2018)
3. Monotonic constraints ablation on datasets with known monotonic structure

**FlexibleNN — YES for shared-weights with GumbelSoftmax / SoftGating / STE + ELBO. Independent-weights same.**

- The shared-weights variant trains all blocks on every batch and aggregates via `Σ prob_i · output_layer(block_i(...))`. Gradients flow through every block, so even depth=1 has the full training signal. At inference, `hard_forward` groups samples by argmax depth and runs only the needed blocks per bucket. This is sound.
- The independent-weights variant is conceptually a mixture-of-networks where each expert has a different depth. It uses more parameters but allows per-depth specialization. Also sound.
- The ELBO depth prior regularizes toward shallow networks; the KL term functions as an Occam penalty. Clean variational formulation.
- **Concern:** the training cost of shared-weights is `O(max_depth)` per step, so `max_depth=10` costs 10× a fixed-depth-1 network. At inference `hard_forward` can recover most of that cost *if the gate produces near-one-hot outputs*. The hard-inference tests show ~0.02 MSE divergence from soft, which is acceptable; but the benchmark should include **wall-clock inference time** to validate the compute-savings claim. That’s a paper-grade experiment, not a correctness issue.

**Systematic selection strategy audit — both models.**

The same set of strategies is used in FlexibleNN (depth selection) and ProbReg (n_classes selection). Here is the full status:

| Strategy | FlexibleNN depth | ProbReg k-select | Tested? | Implementation correct? | Theoretically sound? |
|---:|---:|---:|---:|---:|:---|
| NoneStrategy (fixed) | Yes | Yes | Yes | Yes | Yes — trivial baseline. |
| SoftGating | Yes | Yes | Yes (best performer for ProbReg) | Yes | Yes — continuous relaxation, full gradient flow. |
| GumbelSoftmax | Yes | Yes | Partially (ProbReg only) | Yes | Yes in theory, but **noisy KL gradients cause poor ELBO training dynamics**. Documented failure mode. |
| STE (Straight-Through) | Yes | Yes | **No** — untested in benchmarks | **Bug N2** — gradient path broken in `_hard_selection_logic` | Yes if N2 is fixed — standard discrete-relaxation technique. |
| REINFORCE | Yes | Yes | **No** — untested in benchmarks | Believed correct (not audited line-by-line) | Yes — high variance but unbiased. Needs variance reduction (baseline subtraction) for practical use. |
| Hard inference (argmax) | Yes (FlexibleNN only) | N/A | Yes (±0.02 MSE) | Yes | Yes — deterministic at inference, groups by argmax depth. |

**Action items for FlexibleNN paper:**
1. Benchmark all 5 strategies on piecewise + UCI datasets (FlexibleNN depth selection)
2. Benchmark all 5 strategies on heteroscedastic + UCI datasets (ProbReg k-selection)
3. Wall-clock inference comparison: soft vs hard (argmax) at max_depth ∈ {3, 5, 8, 10}
4. Fix STE (bug N2) before benchmarking — or explicitly exclude and document why
5. REINFORCE: add baseline subtraction if variance is too high; otherwise document and include results
6. For ProbReg: same strategy sweep (SoftGating vs Gumbel vs STE vs REINFORCE) on dynamic k

**Supporting code — mostly sound with the caveats in §1.1.**

- `create_bins` (percentile-based) produces unique boundaries and handles ties. Correct.
- `masked_cross_entropy_loss` (for per-sample variable k) correctly masks invalid class logits with `-inf` before softmax. Correct.
- `calculate_combined_loss` now applies NLL directly to model output without double-log (historical bug #8 is fixed).

### 1.3 Known limitations (design, not bugs)

- **Marginal conformal coverage only.** `ConformalWrapper` produces a constant interval width. For heteroscedastic data, this over-covers in low-noise regions and under-covers in high-noise regions on a per-point basis. A locally-adaptive version (normalizing residuals by predicted σ) is a natural extension.
- **No support for multi-output / multi-target regression.** All models assume scalar targets. The generalization is straightforward for the SEPARATE_HEADS design but needs thought for ClassifierRegression (would need per-target binning).
- **Feature selection uses a single SHAP round.** Iterative / stability-selection variants aren’t supported. Not needed for the paper.
- **Optuna search spaces do not include `loss_type` / `beta` / `target_transform`.** RESUME.md flags this. For the paper we’ll want the sweep to pick these jointly; expose them in the search space.
- **Joint classification + regression loss weighting is implicit.** See §1.2. Exposing a learnable or swept `lambda_ce` would strengthen the empirical results.

---

## 2. Benchmark Redesign (Q2)

### 2.1 Assessment of current toy problems

| Fixture | What it tests | Problem for a paper |
|---|---|---|
| `heteroscedastic_data` (1-D sine with `σ(x) = 0.1 + 0.4·|x|`, n=500) | Learned variance; `NoiseR` correlation | Too small, 1-D, no confounders. Reviewers will not accept this as evidence of “real-world” calibration. |
| `piecewise_data` (1-D linear + sinusoidal, n=500) | Input-dependent depth selection | Handcrafted for FlexibleNN to win. Will read as cherry-picked. |
| `multimodal_data` (1-D bimodal `y = x ± 1.5`, n=500) | Multi-mode regression | 1-D. Every baseline that has a mixture or quantile head will also handle this. |
| `exponential_data` (`y = exp(x) + ε`, n=800) | Symlog transform | Useful for symlog but nothing else. Confirms a feature, not a model. |

**Verdict.** These are **mechanism tests**, not benchmarks. They’re fine for the test suite — they prove the components behave correctly — but they are insufficient for a paper. A reviewer will ask: “why should I believe this architecture works on the kind of tabular data people actually care about?”

**Also missing from the current synthetic set:**

1. **Higher-dimensional** (d ≥ 10) with correlated features — closer to real tabular.
2. **Large training size** (n ≥ 10k) to show the models scale and to distinguish statistical significance.
3. **Noisy / contaminated targets** — label noise and outliers stress-test calibration.
4. **Sharp local discontinuities** (step functions, edges) — stress-test the bin structure.
5. **Long-tailed targets** — stress-test symlog + heavy-tailed variance estimation.
6. **Categorical confounders** — stress-test the preprocessing pipeline.

### 2.2 Proposed new benchmark suite

Three layers:

- **Layer 1: Physically-motivated synthetic problems (our own designs, with known Bayesian ground truth).** For controlled mechanism ablations, not leaderboard claims.
- **Layer 2: Real physics-derived tabular benchmarks (published, public).** UCI Airfoil Self-Noise, UCI Superconductivity, NASA Exoplanet Archive Kepler KOI, plus UCI HTRU2 as a classification sanity check. These are real physical measurements and they bridge the gap between the hand-crafted synthetic set and the astrophysics headline.
- **Layer 3: Standard UCI UQ benchmarks under the Hernández-Lobato & Adams 2015 protocol.** Non-negotiable for comparability with the published probabilistic-regression literature.

**Two problem sets for two papers.** Since ProbReg and FlexibleNN are separate contributions, each needs problems that showcase its specific advantage. Both models run on all problems (for completeness), but the narrative focus differs:

**Paper A (ProbReg) — problems where learned uncertainty structure matters:**
- Problems with **heteroscedastic noise** (noise varies with input) → show n_classes adapts: more bins where noise is complex, fewer where it's simple.
- Problems with **multimodal targets** → show per-class regression heads capture distinct modes that a single Gaussian misses.
- Problems with **heavy-tailed / wide-range targets** → show symlog + learned variance handles extreme values.
- **Simplicity baseline:** On well-behaved (homoscedastic, unimodal) problems, ProbReg with k=2 should reduce to near-Gaussian behavior, matching standard baselines. This demonstrates adaptability without overhead.

**Paper B (FlexibleNN) — problems where input-dependent computational complexity matters:**
- Problems with **spatially varying complexity** (smooth in some regions, highly nonlinear in others) → show depth gate assigns more layers to complex regions.
- Problems with **mixed easy/hard subpopulations** → show hard inference (argmax) gives compute savings on the easy subpopulation with minimal accuracy loss.
- Problems with **increasing dimensionality** → show depth gating scales: higher-d problems need more depth on average, but easy instances still get shallow paths.
- **Simplicity baseline:** On uniformly simple problems, FlexibleNN should collapse to depth=1, matching a shallow NN. On uniformly complex problems, it should use max depth uniformly, matching a deep NN. This demonstrates the model doesn't add unnecessary complexity.

**Shared problems (Layer 3 UCI):** Both models run on the full UCI suite. ProbReg paper focuses on NLL/CRPS/calibration; FlexibleNN paper focuses on accuracy-compute tradeoff and wall-clock comparisons.

The astrophysics applications (§6) come on top as headline experiments: photo-z for ProbReg, cluster mass for FlexibleNN.

#### Layer 1: Physically-motivated synthetic problems

Each is designed to have a specific failure mode that the research models should handle better than the baselines. For each dataset we specify the generating process so the ground-truth calibration curve is known analytically.

**Important framing note.** All five synthetic problems below are **original designs for this paper**, not standard benchmarks from the literature. They exist to test specific architectural mechanisms (multimodality, heteroscedasticity, depth adaptation, censoring, feature-selection robustness) against known-correct Bayesian posteriors where available. They are demonstrations and mechanism ablations, not leaderboard entries. The leaderboard work is done by Layer 2 (real physics-derived tabular), Layer 3 (UCI standard protocol), and the astrophysics headline (§6). The paper must be explicit about this to avoid the “handcrafted to show wins” criticism.

**B1 — Gravitational inverse problem (d=10, n=10k).** Observable: noisy gravitational acceleration `g` at 10 points along a 1-D track, around a point-mass `m` at position `x_0`:
`g_i = G · m / (x_i − x_0)² + ε_i`, with `ε_i ~ N(0, σ_i²)` and `σ_i = 0.02 · |g_i|` (multiplicative noise). Target: `x_0` (position). Sampled with `x_0 ~ U(−2, 2)`, `m ~ LogNormal(0, 0.5)`. Properties: **heavy-tailed features**, **input-dependent noise**, **multimodal posterior** (the two near-field points dominate in different mass regimes). Closed-form posterior is computable via importance sampling, so the calibration curve is exact.

*Why this is a good benchmark:* physically interpretable (point-mass gravity), has a correct Bayesian posterior for reference, targets the heteroscedasticity + multimodality that ProbReg should dominate.

**B2 — Oscillator with phase ambiguity (d=8, n=10k).** Target: oscillation frequency `f` of a damped harmonic oscillator observed at 8 time points `t_i`:
`y(t_i) = A · exp(−γ · t_i) · cos(2πf · t_i + φ) + ε`. Features are `y(t_i)` plus the times. The frequency `f` has an aliasing ambiguity (`f` and `N/Δt − f` are indistinguishable at finite sampling), so the posterior over `f` is **genuinely bimodal** for certain observations. Targets: `f ~ U(0.1, 0.9)`, `A ~ U(0.5, 1.5)`, `γ ~ U(0.05, 0.3)`, `φ ~ U(0, 2π)`.

*Why:* physically motivated (oscillator measurements are ubiquitous), has a known bimodal likelihood, and the aliasing is a real-world source of multimodality — exactly where a mixture / classification-based head should beat a Gaussian-only baseline.

**B3 — Two-phase transition (d=12, n=10k).** A regression problem that switches regime at an unknown feature threshold: `y = f_1(x)` for `x · w < τ`, `y = f_2(x)` for `x · w ≥ τ`, with `f_1` simple linear and `f_2` non-linear. Additive noise scales with distance from the threshold: near the transition, noise is high (regime ambiguity); far from it, noise is low. Target: `y`.

*Why:* ideal stress test for FlexibleNN — simple regime should use shallow depth, non-linear regime should use deep. Also ideal for ProbReg — the variance should spike near the threshold. The fact that this is “handcrafted” is a feature here, because it allows us to plot per-input selected depth against distance-to-threshold and validate the architecture mechanistically. This is a demonstration, not a leaderboard entry.

**B4 — Truncated-likelihood regression (d=15, n=20k).** Target values are censored at a threshold `y < y_cutoff` (physical analogue: detection threshold in any observational science). Standard regressors will systematically over-predict below the threshold; ProbReg with its asymmetric per-class variance should handle this better. Also tests that the predicted intervals correctly widen toward the threshold.

**B5 — Exponentially-distributed feature importance (d=30, n=10k).** 3 informative features, 27 noise features. Tests feature selection robustness (SHAP-based) and the pipeline’s ability not to overfit noise. Standard setup in the ML benchmark literature.

All five synthetic datasets should be generated with **fixed seeds** and checked into the repo as small JSON/parquet files (not regenerated at test time) so results are reproducible across versions.

#### Layer 2: Real physics-derived tabular regression benchmarks

These are **real, published, public datasets** where the features and targets come from physical measurements. They bridge the gap between the synthetic problems above and the astrophysics headline (§6), and they are well-established in the ML benchmarking literature so reviewers will recognize them. All are listed with verified access URLs.

**L2.1 — UCI Airfoil Self-Noise (NASA).** 1503 samples, 6 features (frequency in Hz, angle of attack in degrees, chord length in m, free-stream velocity in m/s, suction-side displacement thickness in m), target: scaled sound pressure level (dB). Physics: self-induced airfoil noise from boundary-layer turbulence, measured in an anechoic wind tunnel on NACA 0012 airfoils. Heteroscedastic in the high-frequency regime. Standard tabular regression benchmark since 1989. No missing values. Available at https://archive.ics.uci.edu/dataset/291/airfoil+self+noise.

**L2.2 — UCI Superconductivity (Hamidieh 2018).** 21263 samples, 81 features (derived elemental properties: mean atomic radius, valence, thermal conductivity, electron affinity, atomic mass, and weighted/entropy variants of each), target: critical temperature `T_c` (K). Targets span `T_c ∈ [0, 185]` K — **strongly heavy-tailed**, making this a real-world symlog test case. Published baseline RMSE ≈ 9.5 K with XGBoost. Clear expected win for ProbReg (percentile binning + symlog jointly handle heavy tails). Reference: Hamidieh, *Comput. Mater. Sci.* 154, 346 (2018). Available at https://archive.ics.uci.edu/dataset/464/superconductivty+data.

**L2.3 — Kepler Objects of Interest, NASA Exoplanet Archive.** ~9564 candidate signals (2358 confirmed exoplanets, 2366 candidates, 4840 false positives). Tabular features: orbital period, transit depth, transit duration, signal-to-noise ratio, stellar parameters (T_eff, log g, [Fe/H], R_*). Two regression variants are natural: (a) predict **planet radius** from transit parameters + stellar features (scalar, positive, log-distributed, multimodal across rocky/sub-Neptune/Jupiter populations), or (b) predict **equilibrium temperature**. The planet-radius regression is the cleaner tabular problem and has been used in multiple exoplanet-ML papers. Public API at https://exoplanetarchive.ipac.caltech.edu/.

**L2.4 — UCI HTRU2 Pulsar (sanity-check classification).** 17898 candidates (1639 real pulsars, 16259 RFI/noise), 8 features (integrated profile + DM-SNR statistics). Binary classification, not regression — listed here only as a calibration sanity check on a real astrophysics problem with known labels from the High Time Resolution Universe survey. Reference: Lyon et al. 2016, *MNRAS* 459, 1104. Available at https://archive.ics.uci.edu/dataset/372/htru2.

#### Layer 3: Standard ML UQ benchmarks (UCI protocol, Hernández-Lobato & Adams 2015)

For the paper to be taken seriously as a tabular probabilistic regression paper, it must include the standard UCI suite used in every Bayesian-NN / deep-ensemble paper since Hernández-Lobato & Adams (2015, *Probabilistic Backpropagation*, ICML; arXiv:1502.05336). **Boston Housing is excluded**: it was removed from scikit-learn 1.2 due to an ethically problematic engineered feature (see https://scikit-learn.org/stable/datasets/toy_dataset.html — the dataset was deprecated in v1.0 and removed in v1.2). The post-PBP standard set minus Boston is:

- **Concrete Compressive Strength** (n=1030, d=8) — UCI, physics (cement mixture proportions → compressive strength)
- **Energy Efficiency** (n=768, d=8) — UCI, physics (building parameters → heating and cooling load)
- **Kin8nm** (n=8192, d=8) — Delve, robot-arm forward kinematics
- **Naval Propulsion Plant** (n=11934, d=16) — UCI, marine gas-turbine degradation
- **Combined Cycle Power Plant** (n=9568, d=4) — UCI, thermodynamic measurements → net hourly electrical output
- **Protein Structure (CASP)** (n=45730, d=9) — UCI, protein tertiary-structure RMSD
- **Wine Quality Red / White** (n=1599 / 4898, d=11) — UCI, chemistry → sensory rating
- **Yacht Hydrodynamics** (n=308, d=6) — UCI, physics (hull geometry → residuary resistance)
- **California Housing** (n=20640, d=8) — Kelley Pace & Barry 1997, standard Boston replacement

**Protocol (non-negotiable for comparability).** Twenty random 90/10 train/test splits per dataset, identical preprocessing to the PBP paper, report mean ± std of **test NLL and test RMSE**. This is the protocol everyone in the tabular-UQ literature uses since 2015. Every paper we would compare against (PBP, MC Dropout, Deep Ensembles, NGBoost, SWAG, Laplace) reports numbers under exactly this protocol.

**Note on dataset overlap.** Most of the UCI Layer 3 datasets are themselves physics-derived (concrete, energy, naval, power plant, yacht), so Layer 3 is effectively a continuation of Layer 2 under a stricter evaluation protocol. The distinction matters for the paper: Layer 2 uses our own 5-seed protocol so we can run extensive ablations; Layer 3 uses the 20-seed PBP protocol so the numbers slot directly into the existing leaderboard.

### 2.3 Experimental protocol

**Splits.** 5 random 80/20 train/test splits per dataset, 20 for the small UCI sets. Fixed seeds. Report mean ± std.

**Hyperparameters.** For each model-dataset pair, run Optuna for N trials with a pre-declared search space. N ≥ 50 for the UCI sets, N ≥ 30 for the synthetic/astrophysics. Search spaces are in `configs/benchmark_search_spaces.yaml` (new file).

**Reporting.** A single master table per dataset family, with columns: `model`, `RMSE`, `MAE`, `NLL`, `CRPS`, `PICP@95`, `MPIW@95`, `ECE`, `wall_clock_train`, `wall_clock_inference`, `#params`. See §3 for metric definitions.

**Ablations.** For the research models, sweep:

- ProbReg: `k ∈ {2, 3, 5, 10, 20}`, `regression_strategy ∈ {SEPARATE_HEADS, SINGLE_HEAD_N_OUTPUTS, SINGLE_HEAD_FINAL_OUTPUT}`, `loss_type ∈ {nll, beta_nll}` at `β ∈ {0.0, 0.5, 1.0}`, `n_classes_regularization ∈ {NONE, K_PENALTY, ELBO}` with `n_classes_selection_method ∈ {NONE, SOFT_GATING, GUMBEL_SOFTMAX}` (STE excluded pending N2 fix).
- FlexibleNN: `max_hidden_layers ∈ {3, 5, 8}`, `depth_regularization ∈ {NONE, DEPTH_PENALTY, ELBO}`, `layer_selection_method ∈ {SOFT_GATING, GUMBEL_SOFTMAX}` plus independent-weights variant.

**Compute budget.** Fix a total wall-clock budget per model per dataset (say 10 minutes on CPU for small sets, 30 on XPU for large sets) to make the comparison fair. Models that can’t converge in budget report failure, not worst-case numbers.

**Statistical testing.** Use Wilcoxon signed-rank (paired on seeds) to test each new model’s NLL vs. each baseline. Report p-values in the appendix.

**Versioning.** All results written to `docs/benchmarks.md` with the git SHA they were produced under, so regressions are detectable across commits.

---

## 3. Metrics Extensions (Q3)

Current metrics (`utils/metrics.py`): MAE, RMSE, R², MAPE, median APE, NLL (Gaussian), PICP (at 1.96σ), ECE (non-standard, see N3). For a paper-quality evaluation the following must be added.

### 3.1 Proper scoring rules

**CRPS — Continuous Ranked Probability Score.** The de facto gold standard for probabilistic regression (Gneiting & Raftery 2007). For a Gaussian prediction `N(μ, σ²)`:

```
CRPS(N(μ, σ²), y) = σ · [ (y − μ)/σ · (2Φ((y − μ)/σ) − 1) + 2φ((y − μ)/σ) − 1/√π ]
```

Strictly proper (minimized iff predictive distribution equals true distribution). Decomposes into **reliability + resolution − uncertainty** terms. Must be in the primary table.

**Energy score.** Multivariate generalization of CRPS (Gneiting et al., 2008). Needed only if we extend to multi-output regression; deprioritize for now.

**Log-score / NLL.** Already there — but currently only Gaussian. When we support mixtures (NGBoost, MDN baselines), we need a general mixture-NLL that can consume an arbitrary predictive distribution object. Design: make `calculate_nll` accept a `predictive_distribution` handle with a `.log_prob(y)` method (pattern used in `tensorflow_probability.distributions`).

### 3.2 Interval scores

**Winkler score.** For a central `(1 − α)` interval `[L, U]`:

```
W_α(L, U, y) = (U − L) + (2/α) · (L − y) · 1[y < L] + (2/α) · (y − U) · 1[y > U]
```

Proper score for intervals; penalizes both width and miscoverage. Report at α = 0.05, 0.1, 0.2.

**MPIW — Mean Prediction Interval Width.** Average `U − L` at a fixed α. Reported alongside PICP to show the sharpness-calibration tradeoff (narrow intervals with low coverage beat wide intervals with high coverage in a unified score like Winkler).

**Interval coverage at multiple α.** Extend current `calculate_picp` to return a dict keyed by α instead of hard-coding 1.96. Values: `α ∈ {0.05, 0.1, 0.2, 0.32}` (corresponding to 1σ, 2σ, 95%, 90%).

### 3.3 Calibration

**PIT-based calibration curve + ECE.** Replace `calculate_ece` with the Kuleshov et al. (2018) formulation:

```python
def calibration_curve(y_true, predictive_dist, n_bins: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Returns (target_quantiles, empirical_quantiles) for a reliability diagram.

    Implements Kuleshov et al. 2018: for each target quantile p, computes the
    empirical fraction of test points with CDF(y_true) ≤ p, where CDF is the
    predictive CDF.
    """
    pit = predictive_dist.cdf(y_true)  # shape (n,), should be ~Uniform(0,1) if calibrated
    target = np.linspace(0, 1, n_bins + 1)
    empirical = np.mean(pit[:, None] <= target[None, :], axis=0)
    return target, empirical

def ece_regression(y_true, predictive_dist, n_bins: int = 20) -> float:
    target, empirical = calibration_curve(y_true, predictive_dist, n_bins)
    return np.mean(np.abs(target - empirical))
```

Also save the calibration curve so it can be plotted (reliability diagram is standard in every UQ paper).

**Miscalibration area.** Trapezoidal integral of `|empirical − target|` — a more robust scalar than the binwise mean.

**Sharpness.** Mean predictive variance (or predictive interval width). A perfectly calibrated model that always predicts a very wide interval would look calibrated but be useless; sharpness quantifies how narrow the intervals are, given calibration. Standard plot: sharpness (x-axis) vs miscalibration (y-axis), scatter across ablation runs.

### 3.4 Quantile-specific

**Quantile loss / pinball loss.** For a target quantile `τ`:

```
L_τ(y, q̂) = max(τ · (y − q̂), (τ − 1) · (y − q̂))
```

Needed for comparison with quantile regression baselines (see §4). At `τ = 0.5`, this is MAE.

**QICE — Quantile Interval Coverage Error.** For quantile models, checks whether the nominal quantile matches empirical coverage at multiple levels simultaneously; a vectorized generalization of PICP that avoids the arbitrary “1.96” choice.

### 3.5 Comparative / ranking

**Normalized NLL / CRPS across datasets.** For each dataset, compute `(score − best_score) / (worst_score − best_score)`, average across datasets for a model-level ranking. Standard in multi-dataset benchmark tables (used e.g. by Salinas et al. 2020 for DeepAR benchmarks).

**Critical-difference diagram** (Demšar 2006). Visual presentation of average ranks across datasets with statistical significance bars. Required in any paper comparing more than three models on more than three datasets.

### 3.6 Sanity checks / debugging metrics (not for the paper, but for day-to-day)

- **Noise correlation** (Pearson between predicted σ and true σ) — already exists in some scripts. Promote to `Metrics.calculate_noise_correlation` so it’s a first-class citizen.
- **Residual vs predicted σ plot** — diagnostic for identifying whether σ underestimates or overestimates on average.
- **Per-bin accuracy for FlexibleNN** — which depth did each sample select? Histogram across a dataset. Sanity-check for B3.

### 3.7 Implementation plan

- `utils/metrics.py`: add `calculate_crps`, `calculate_winkler`, `calibration_curve`, `ece_regression`, `calculate_miscalibration_area`, `calculate_sharpness`, `calculate_pinball_loss`. Keep the old `calculate_ece` but rename to `calculate_ece_legacy` and deprecate it with a warning.
- New file `utils/distributions.py`: a `PredictiveDistribution` protocol that wraps Gaussian, Student-t, Mixture-of-Gaussians, and Empirical (samples) with a common interface (`cdf`, `ppf`, `log_prob`, `mean`, `variance`). This is what the CRPS / calibration functions consume.
- `Metrics.calculate_all_metrics` returns the new metrics when `y_std` is provided.
- Add a reliability-diagram plot to `Metrics.plot_regression_charts`.

### 3.8 Publication-quality visualizations

Current plotting is limited to basic regression scatter plots. The following are required for publication:

**Per-paper standard plots:**
1. **PIT histogram** — the gold standard calibration diagnostic. Flat = well-calibrated. One per model per dataset.
2. **Reliability diagram** — predicted vs observed quantiles (Kuleshov et al. 2018). Diagonal = perfect calibration. Overlay all models on one plot per dataset.
3. **Sharpness-calibration scatter** — x = mean interval width (sharpness), y = miscalibration area. One point per model per dataset. Shows the tradeoff: a wide-interval model is calibrated but useless.
4. **Critical-difference diagram** (Demšar 2006) — average rank across datasets with Nemenyi significance bars. One per metric (NLL, CRPS, RMSE). Standard in any multi-model multi-dataset paper.
5. **Interval width vs coverage curve** — sweep α from 0.01 to 0.99, plot PICP vs MPIW. Shows how each model's intervals tighten as coverage relaxes.

**ProbReg-specific plots (Paper A):**
6. **Per-input k heatmap** — show how dynamic k varies across the feature space. On the heteroscedastic dataset: k should be higher where noise variance is higher.
7. **Multimodal prediction density** — for specific inputs with known bimodal targets (e.g., Lyman-break galaxies at z ≈ 3), show the full predictive distribution with two peaks.
8. **Bin centroid overlay** — plot learned classification bin boundaries overlaid on the target distribution. Shows the bins correspond to physically meaningful regions.

**FlexibleNN-specific plots (Paper B):**
9. **Per-input depth map** — show selected depth across the feature space. On piecewise data: depth=1 on linear half, depth=3 on sinusoidal half.
10. **Soft vs hard inference comparison** — scatter plot of soft predictions vs hard predictions, per sample. Show tight correlation (±0.02 MSE).
11. **Wall-clock bar chart** — inference time for soft vs hard at multiple max_depth values.
12. **Depth histogram per dataset** — what fraction of samples select each depth.

**Cross-model comparison plots:**
13. **Residual vs predicted σ** — diagnostic for under/over-estimation of uncertainty.
14. **Per-dataset NLL/CRPS bar chart** — standard grouped bar chart for the master comparison table.

**Implementation:** New file `utils/publication_plots.py` with functions for each plot type. All plots use matplotlib with a consistent style (serif font, tight layout, colorblind-safe palette). Each function takes the master results DataFrame and produces a single figure.

---

## 4. Additional Comparison Models (Q4)

Currently supported baselines: XGBoost, LightGBM, CatBoost, Linear/Logistic Regression, plain PyTorch NN, ClassifierRegression, ProbabilisticRegression, FlexibleNN, IndependentWeightsFlexibleNN. For a tabular probabilistic regression paper this is **missing every model literature expects to see**.

### 4.1 Tier A — must add before the paper

**NGBoost (Duan et al., 2020).** Natural gradient boosting that produces a parametric predictive distribution (Gaussian, lognormal, etc.). The direct tree-based analogue of ProbReg and the most important baseline to beat — if ProbReg doesn’t beat NGBoost on NLL/CRPS for heteroscedastic tabular data, there is no paper. Install via `ngboost` pip package. Thin wrapper to match `BaseModel` interface.

**MDN — Mixture Density Network (Bishop 1994).** Neural network that outputs the parameters of a Gaussian mixture. This is the closest neural-network baseline to ProbReg (both produce multi-modal predictive distributions) and is the *right* comparison for demonstrating that the classification-based decomposition is better than a raw Gaussian mixture head. Implementation: ~150 lines on top of `PyTorchModelBase`.

**Quantile Regression NN (Koenker 2005, pinball-loss NN).** Predicts multiple quantiles (e.g. 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95) simultaneously via the pinball loss. Distribution-free (no Gaussian assumption). Closest non-parametric neural baseline. Important for showing that ProbReg’s parametric form isn’t the bottleneck. Also ~150 lines.

**Deep Ensembles (Lakshminarayanan et al., 2017).** Ensemble of M independently-initialized NNs, each predicting `(μ_i, σ_i²)`, combined as an equal-weighted Gaussian mixture. The de facto baseline for UQ in deep learning since 2017. No paper on probabilistic regression omits this. Implementation: a `DeepEnsembleWrapper` over any `PyTorchModelBase` subclass; run M=5 fits on M different seeds, aggregate predictions.

**Gaussian Process (scikit-learn GaussianProcessRegressor with Matérn kernel).** The textbook Bayesian nonparametric baseline. Small datasets only (scales O(n³) in training). Include on UCI sets where n ≤ 5000. Already in sklearn so only a wrapper is needed.

**Constant-variance NN.** Already supported via ProbReg with `n_classes=1` and `uncertainty_method=PROBABILISTIC` (single class head learns a global σ²). **Do not create a separate class.** Instead, register it as a named configuration alias (e.g., `ConstantVarianceNN = ProbReg(n_classes=1, ...)`) in the benchmark runner, and verify the existing code path produces correct results with k=1. This is the simplest UQ baseline and shows whether input-dependent uncertainty is even worth the complexity.

### 4.2 Tier B — nice to have, deprioritize

**Laplace-approximation Bayesian NN (Daxberger et al., 2021, `laplace-torch` package).** Fits a point-estimate NN, then post-hoc Gaussian-approximates the posterior at the MAP. Provides a principled Bayesian UQ baseline without the MCMC cost. ~50 lines.

**Variational Bayesian NN (Blundell et al., 2015, mean-field Gaussian).** Older baseline, still appears in benchmarks. Implementation effort vs added signal is unfavorable for us; skip unless a reviewer demands it.

**BART — Bayesian Additive Regression Trees.** Strong tabular baseline, available via `PyBART` or `bartpy`. Slow but principled. Consider for the UCI tables only.

**Infinite-width / NTK baselines.** Too exotic for this paper.

**FT-Transformer / TabPFN.** Covered in §5.

### 4.3 Integration

Every new model gets a thin wrapper in `automl_package/models/` following the `BaseModel` interface (fit, predict, predict_uncertainty, `is_regression_model=True`, `name` property). Tests in `tests/test_baselines.py`. Benchmark entries added to `docs/benchmarks.md` in a single pass once the wrappers exist.

**Estimated effort:** ~3 days for Tier A (the wrappers are thin; the tricky parts are NGBoost’s fit API and Deep Ensembles’ multi-model coordination), ~2 days for Tier B cherry-picked subset (Laplace + BART).

---

## 5. Transformer Support (Q5)

**Short answer: no sequence transformers. Yes to FT-Transformer and TabPFN as baselines. No first-class transformer support in the core package.**

**Why not sequence transformers.** The core models in this repo are tabular regressors / classifiers on fixed-shape features. A sequence transformer (BERT, GPT-style) requires a sequential input modality (text, time series, event sequences) and an entirely different preprocessing / tokenization pipeline. Adding it would mean building a parallel abstraction stack for sequences and double the testing surface. The astrophysics application we’re picking (photometric redshift) is tabular — adding sequence transformers adds no value for this paper.

**Yes to tabular transformers, but only as baselines.** Two specific models:

- **FT-Transformer (Gorishniy et al., 2021).** Tabular transformer that tokenizes each feature into an embedding and applies self-attention. On most UCI benchmarks it performs comparably to gradient-boosted trees. If the paper claims to be a general tabular probabilistic regression framework, FT-Transformer is the modern-deep-learning baseline to include. Reference implementation at https://github.com/yandex-research/rtdl. Thin wrapper, ~200 lines.

**Future work: FlexibleNN depth gating applied to FT-Transformer blocks (Paper C).** The FlexibleNN mechanism (gate network predicts `P(depth=d|x)`, ELBO regularizes toward shallow) can be applied to transformer blocks: each tabular input gets routed through a variable number of self-attention layers based on its complexity. This is distinct from existing adaptive-computation approaches:
- **Early exit** (DeeBERT, Xin et al. 2020) — exits at a fixed confidence threshold, no learned prior.
- **Mixture of Depths** (Raposo et al. 2024, Google DeepMind) — skips tokens within layers, different mechanism.
- **MIND** (ICLR 2025) — uses fixed-point iteration for adaptive depth, no per-input discrete depth selection.
Our approach would use the same ELBO + SoftGating framework from FlexibleNN, giving a principled Occam penalty for unnecessary transformer depth. If this works on FT-Transformer for tabular data, the same mechanism applies to LLMs (variable transformer depth per token). **This is explicitly scoped as Paper C — not this round.** The FlexibleNN paper (Paper B) should frame this as the natural extension in the future work section.

- **TabPFN (Hollmann et al., 2023).** A pre-trained transformer that performs in-context tabular regression without task-specific training. State-of-the-art on small tabular datasets (n ≤ 10k, d ≤ 100). Essentially free to evaluate (no fit step). Include on all UCI datasets small enough to fit the model’s context window. `tabpfn` package exists; wrapper is trivial. **Including TabPFN is non-optional** — a 2023-2024 tabular ML paper without it is not credible.

**Neither should be integrated into the core package architecture.** They’re baselines, not building blocks. Wrap them behind `BaseModel`, add to the comparison table, move on.

**On transformer-based probabilistic heads.** There is interesting work on applying the ProbReg-style classification-bottleneck trick inside transformer architectures (e.g. two-hot encoding in DreamerV3). That’s a follow-up paper, not this one.

**Decision: add FT-Transformer and TabPFN as Tier-A baselines in §4, do not add sequence transformer support.**

---

## 6. Astrophysics Case Studies (Q6) — One Per Paper

Each paper needs its own astrophysics headline experiment. The problems were selected for alignment with each model's specific strengths, and for having active post-2022 literature using modern ML methods (transformers, normalizing flows, score-based models) so we can compare against state-of-the-art.

### 6.1 Candidate problems evaluated

| Problem | Tabular? | Public data? | Uncertainty matters? | Multimodality? | ML baselines? | Fit for this paper |
|---|---|---|---|---|---|---|
| **Photometric redshift** | Yes | Yes — SDSS DR17, DES DR2, LSST DC2 simulated data | **Yes**: error budget dominates cosmological analyses | **Yes**: color-redshift degeneracies, Lyman break aliasing | Many: ANNz2, GPz, MDNs, Random Forest, CNN | **Primary recommendation** |
| **Galaxy cluster mass (observables → M_halo)** | Yes | Yes — Planck SZ, SPT, ACT catalogs + IllustrisTNG/MDPL2/Magneticum simulated training data | **Yes**: halo-mass uncertainty drives cosmological constraints | Partially — mostly single-mode heavy tail | Yes: Ntampaka+ (2015-2019), Ho+, Kodi Ramanah+ | **Secondary recommendation** |
| **Exoplanet characterization from transit parameters** | Partially — light curves are sequential, summary stats are tabular | Yes — Kepler, TESS | Yes | Yes (stellar activity confound) | Many | Tabular version possible but less impactful |
| **Gravitational wave source parameters** | No (strain is time-series) — would need CNN/transformer | Yes — GWTC catalog | Yes | Yes (mass degeneracies) | Yes | **Wrong modality** for tabular models |
| **Weak-lensing mass maps** | No (image modality) | Yes — HSC, DES, KiDS | Yes | No | CNNs | **Wrong modality** |
| **Strong-lensing time delays** | Partially | Yes — TDLMC, H0LiCOW | Yes | Yes | Some | Small datasets |
| **Supernova light-curve classification → cosmology** | Sequential | Yes — Open Supernova Catalog, DES-SN, LSST | Yes | Yes | Many | Wrong modality for this paper |

### 6.2 Paper A headline: Photometric redshift estimation (ProbReg)

**What it is.** Galaxies have spectra; spectra reveal their redshift (and hence distance). Measuring spectra is expensive. Photometry (integrated flux in ~5-10 broadband filters: u, g, r, i, z, Y, J, H, K) is cheap. Machine learning maps photometry → redshift. This is the **single most-studied ML problem in observational cosmology**, and it drives the systematic error budget for every large-scale structure, weak-lensing, and galaxy-clustering analysis out of LSST, Euclid, and Roman.

**Why this is the right fit for ProbReg specifically.**

- **Heteroscedastic and multimodal.** Color-redshift degeneracies produce genuinely multi-peaked posteriors for certain galaxy types (the famous “Lyman break” at z ≈ 3 can alias with the 4000Å break at z ≈ 0.4). A Gaussian regressor cannot capture this; a mixture density network can; a classification-based head like ProbReg is purpose-built for it.
- **Uncertainty is the product.** Point estimates are nearly useless; downstream cosmological analyses consume `p(z | photometry)` as a PDF. ProbReg’s native output — a discrete probability over classes plus per-class regression — IS the object the downstream pipeline wants.
- **Well-curated public data (all verified):**
    - **LSST DESC DC2 — cosmoDC2 extragalactic catalog.** The Rubin Observatory science team’s benchmark simulated catalog for PZ algorithms. Catalog name `cosmoDC2_v1.1.4_image_with_photoz_v1`, hosted at NERSC. Access via the `GCRCatalogs` Python package (https://github.com/LSSTDESC/gcr-catalogs). Current photo-z in the catalog comes from the template-based **BPZ** code, and **DESC has publicly flagged a known systematic at high redshift** because the SED template set did not optimize the UV portion of the templates that shifts into the LSST bands at z ≳ 2 — **this is the specific systematic our ML approach can try to fix, and is the most defensible research framing for the paper**. Reference: Korytov et al. 2019 (cosmoDC2); DESC DC2 Data Release Note (arXiv:2101.04855).
    - **SDSS DR17 galaxy photometric + spectroscopic sample.** Millions of galaxies with ugriz photometry and spec-z ground truth from the Legacy and BOSS/eBOSS surveys. Access via SciServer CasJobs (https://skyserver.sdss.org/casjobs/) or the public API. The classical benchmark dataset for empirical PZ work — Pasquet et al. 2019 used ~500k galaxies from this sample.
    - **DES Year 6 / KiDS DR4-5** if we want an extra cross-survey generalization test. Secondary.
- **Rich baseline literature (all verified):**
    - **ANNz2** — Sadeh, Abdalla, Lahav 2016, *PASP* 128, 104502, arXiv:1507.00490. Neural network PZ with full PDF output via bootstrap ensembles and boosted decision trees. **First paper to provide PZ PDFs from a standard ML method.** Code at https://github.com/IftachSadeh/ANNZ.
    - **GPz** — Almosallam, Jarvis, Roberts 2016, *MNRAS* 462, 726, arXiv:1604.03593. Sparse non-stationary Gaussian process that jointly optimizes mean and variance for heteroscedastic PZ. **Direct Bayesian competitor to ProbReg on heteroscedastic PZ.** Matlab + Python code at https://github.com/OxfordML/GPz.
    - **TPZ** — Carrasco Kind & Brunner 2013, *MNRAS* 432, 1483, arXiv:1303.7269. Prediction trees + random forests with measurement errors incorporated into the split criterion. Full PDF output via tree-leaf histograms. Code http://matias-ck.com/mlz/. Has a RAIL wrapper at https://github.com/LSSTDESC/rail_tpz.
    - **DCMDN (D’Isanto & Polsterer 2018)** — A&A 609, A111, arXiv:1706.02467. CNN feature extractor + Mixture Density Network with **5 Gaussians** outputting a full PZ PDF for galaxies *and* quasars without pre-classification. **The most direct comparable to ProbReg’s multimodal head** — a clean MDN-based PDF output. Code: https://github.com/Kafka-pi/DCMDN.
    - **Pasquet et al. 2019** — A&A 621, A26, arXiv:1806.06607. CNN on 64×64 ugriz SDSS images + galactic reddening, treats PZ as a classification problem over redshift bins (this is essentially the non-dynamic ProbReg baseline inside a CNN). Achieves σ_MAD < 0.01 on SDSS at z < 0.4 for ≥100k training galaxies. **A pixel-based comparable which we can beat on the uncertainty front while losing on point accuracy.** Different modality (images not tabular), so it’s a reference point but not a head-to-head.
    - **DeepDISC-photoz (Sánchez et al. 2024)** — arXiv:2411.18769 (November 2024). Instance segmentation + CNN PZ on DC2. The most recent RAIL-integrated PZ method and the direct state-of-the-art target.
    - **Jones, Do et al. 2024** — *ApJ* 964, 130, arXiv:2306.13179. Bayesian neural network for LSST PZ — this is a current paper that directly claims calibrated PZ for cosmology, and is the closest thing to our paper’s framing. Reading priority: high.
    - **Post-2022 modern methods (verified via WebSearch 2026-04-12):**
        - **Zephyr** (arXiv:2310.20125, Oct 2023) — normalizing flow + mixture density for PZ with heterogeneous training data. Modern flow-based competitor.
        - **nflow-z** (arXiv:2510.10032, Dec 2025) — conditional normalizing flow (cINN, cNSF) for PZ PDFs. Tested on CSST, COSMOS2020, DES Y1, SDSS, DECaLS. Direct modern baseline.
        - **ViT-MDNz** (arXiv:2602.22711, Feb 2026) — first vision transformer for PZ, integrated with MDN. Single-band images, σ_MAD = 2.6%. Different modality (images) but demonstrates transformer+MDN is the current frontier.
        - **RAIL v2** (arXiv:2505.02928, May 2025) — the DESC evaluation framework itself was recently published, establishing the standard protocol we should follow.
    These post-2022 papers confirm the field is **actively advancing with modern architectures**. Our ProbReg paper can claim novelty over all of them: none uses a classification-bottleneck decomposition with law-of-total-variance uncertainty.
- **Benchmark pipeline: RAIL (Redshift Assessment Infrastructure Layers).** https://github.com/LSSTDESC/rail — the LSST DESC Photo-z Working Group’s standardized framework for running any PZ algorithm on common data splits and producing the community-agreed metrics. Sub-packages include `rail_delight`, `rail_lephare`, `rail_cmnn`, `rail_som`, `rail_tpz`, `rail_deepdisc`, `rail_fsps`, `rail_pipelines`. **Plugging our model into RAIL is how we get directly comparable to ≥15 published algorithms without reimplementing any of them.** This drops several months of engineering work off the critical path and is the single most important tooling decision in the plan.
- **Clean ground truth.** Spectroscopic redshifts are accurate to `σ_z / (1+z) ≲ 10⁻⁴` for bright galaxies. For fainter galaxies the spec-z itself becomes uncertain, creating a natural "high-noise regime" that tests calibration.

**Target dataset: LSST DESC DC2 `cosmoDC2_v1.1.4_image_with_photoz_v1` + matched truth redshifts.** Simulated but IS the benchmark used by every LSST-era PZ paper. Real LSST data from DR1 will replace it with minimal code changes once available.

**Alternative primary dataset (if DC2 access is slow): SDSS DR17 main galaxy sample via CasJobs.** Fully public, well-understood, directly comparable to Pasquet+2019 and TPZ/ANNz2.

**What to claim in the paper.**

1. ProbReg with SEPARATE_HEADS and dynamic-k (ELBO + SoftGating) produces better calibrated `p(z | photometry)` (lower CRPS, better PIT uniformity) than NGBoost / MDN / Deep Ensembles / TabPFN on the DC2 benchmark.
2. The learned classification bins correspond to physically meaningful redshift intervals (show this by overlaying bin centroids on the redshift distribution and coloring by galaxy type).
3. Multimodal outputs correctly capture Lyman-break aliasing (show a 2-D density plot: for specific galaxies near z ≈ 3, our predictive distribution has two peaks where point-estimate models predict a single mode).
4. The PIT histogram is flat (standard calibration diagnostic; reviewers expect it).
5. FlexibleNN as a secondary demonstration: redshift estimation complexity varies with galaxy brightness (dim galaxies need more capacity). Show per-input depth vs magnitude.

### 6.3 Paper B headline: Galaxy cluster mass estimation (FlexibleNN)

**Why this is the right fit for FlexibleNN specifically.** Cluster mass estimation from multi-observable data (X-ray temperature, SZ signal, optical richness, velocity dispersion) has **spatially varying complexity**: relaxed clusters follow a simple power-law scaling relation (low depth needed), while merging or disturbed clusters have complex, multi-component mass distributions requiring more capacity. FlexibleNN’s input-dependent depth selection should learn to allocate more layers to disturbed clusters and fewer to relaxed ones. The depth histogram across clusters would be a compelling plot showing the model "knows" which clusters are hard.

**Dataset size is appropriate for FlexibleNN.** Cluster catalogs are moderate-sized (~1k–12k), which is exactly the regime where FlexibleNN’s compute-adaptive inference matters most: on small datasets, the overhead of a gate network pays off if it can correctly route easy samples through shallow paths.

**Data sources (all public).**

- **Planck SZ catalog** (~1600 clusters, SZ-selected, has mass proxies).
- **SPT-SZ / SPTpol / ACT** catalogs (~600 / 1500 clusters respectively).
- **eROSITA eRASS:1** (~12k clusters — new, 2024 release; richest single catalog).
- **Simulated training data.** This is the standard approach: train on IllustrisTNG300 or MDPL2 or The Three Hundred Project, test on simulation then apply to real data. Public at https://www.tng-project.org/ and https://www.cosmosim.org/.

**Baselines (all verified):**

- **Ntampaka et al. 2019** — *ApJ* 876, 82, arXiv:1810.07703, "A Deep Learning Approach to Galaxy Cluster X-ray Masses". CNN trained on 7896 mock Chandra X-ray images derived from 329 massive clusters in IllustrisTNG. Reported mass bias −0.02 dex, scatter 8–12%. Tabular version is a natural follow-up using the same IllustrisTNG halo catalog.
- **Ho et al. 2019** — *ApJ* 887, 25. Dynamical mass estimation from galaxy member line-of-sight velocities via a Bayesian NN. Provides calibrated uncertainty intervals for cluster mass — our most direct baseline for the tabular observables → mass task.
- **Kodi Ramanah et al. 2020–2021** — Deep learning mass inference from galaxy velocity phase-space. Worth citing as another Bayesian NN PZ-adjacent baseline.
- **Krippendorf et al. 2024** — most recent DL cluster-mass review (arXiv:2501.04081, "Galaxy cluster characterization with machine learning techniques"). Useful for comparing to the current state of the field.
- **Post-2022 modern methods (verified via WebSearch 2026-04-12):**
    - **Hybrid Neural Network (hNN)** (arXiv:2511.20429, Nov 2025) — CNN + GNN fusion for estimating triaxial geometry of clusters from 2D observables (X-ray + tSZ + optical). Uses MillenniumTNG simulation. **Most modern architecture applied to cluster characterization.**
    - **Score-based generative model for cluster mass maps** (arXiv:2410.02857, Oct 2024) — learns the predictive posterior of gas/DM maps conditioned on SZ+X-ray inputs. **Generative-model approach to cluster mass — the cutting edge.**
    - **AE-CNN for cluster mass** (arXiv:2507.21876, Nov 2025) — autoencoder-CNN for posterior mean estimation, compared with MLE. Recent probabilistic framework.
These confirm the field has active post-2022 work with modern architectures. FlexibleNN's unique contribution: none of these methods has input-dependent depth selection.

Our FlexibleNN should demonstrate compute-adaptive inference on this problem. Additionally, ProbReg should be competitive with or beat these on log-mass NLL/CRPS via percentile-binning + symlog + learned per-bin variance, on the tabular observables → halo-mass task. We would use the IllustrisTNG public API to extract (M200c, M500c, richness, σ_v, T_X_proxy, Y_SZ_proxy) tuples per halo, and treat it as a supervised regression problem.

### 6.4 Data access and protocol

**Primary (PZ).**

1. Download LSST DC2 object catalog + matched spec-z sample (~100k objects) from NERSC via LSSTDESC portal. Alternatively SDSS DR17 via SciServer (requires free account).
2. Preprocessing: compute colors (mag differences between bands), magnitudes, redshift. Missing-value handling follows RAIL convention. Write to parquet.
3. Train/test split: follow RAIL convention (90/10 random, stratified by magnitude) and the DC2 simulation split.
4. Evaluation follows §3 metrics suite + standard PZ metrics: PIT uniformity (KS test), outlier fraction `η = fraction with |Δz|/(1+z) > 0.15`, bias `⟨Δz/(1+z)⟩`, σ_mad, fraction with catastrophic failures.
5. Publication: compare against RAIL-published results on the same split.

**Secondary (cluster mass).**

1. Download cluster observables from IllustrisTNG300-1 (public API).
2. Map to M500c using published scaling relations as the benchmark baseline.
3. Train ProbReg / FlexibleNN / NGBoost / MDN / Deep Ensembles / TabPFN on sim data.
4. Cross-validate on a held-out simulation. Apply once to Planck SZ catalog as a qualitative “real data” demonstration.

**Expected effort.**

- PZ: 2–3 days to obtain data + preprocess, 1 week to run full benchmark suite with all baselines, 1 day to produce plots.
- Cluster mass: 2 days data prep, 3 days benchmark, 1 day plots.

**Total:** ~2 weeks of dedicated work for the astrophysics section.

---

## 7. Paper Strategy

**Two separate papers for two independent contributions.**

### Paper A — ProbReg

**Title:** *”Classification-Based Probabilistic Regression via the Law of Total Variance”*

**Target venues (ranked):**
1. **NeurIPS / ICLR (methodology track).** Claim: classification-bottleneck probabilistic regression with learned per-class variance, combined via law of total variance, dominates existing tabular UQ methods.
2. **MNRAS / A&A (astrophysics companion).** Claim: calibrated photometric redshifts with multi-modal uncertainty from LSST DC2, outperforming MDN/NGBoost/normalizing-flow baselines on CRPS and PIT calibration.

**Contributions:**
(a) SEPARATE_HEADS + law-of-total-variance decomposition as a general probabilistic regression framework.
(b) ELBO-based dynamic-k (SoftGating) for automatic bin complexity allocation.
(c) Empirical validation on UCI benchmarks (Layer 3) + heteroscedastic/multimodal synthetic problems (Layer 1).
(d) Photometric redshift headline experiment on DC2/SDSS via RAIL, comparing against Zephyr, nflow-z, ANNz2, GPz, DCMDN, Jones+ 2024 BNN.

**Story arc:** Motivation (tabular UQ is underserved; trees have NGBoost but NNs lack a principled discrete-mixture approach) → framework (classification bottleneck + per-class heads) → theory (law of total variance, ELBO k-selection) → ablations (regression strategies, loss types, dynamic-k strategies — §1.2 tables) → UCI benchmarks → photo-z headline → discussion.

**Reviewer attack surface:**
- Novelty vs DreamerV3 two-hot → explicit: DreamerV3 uses *fixed* bin midpoints; we learn per-bin regression heads with variance.
- Novelty vs MDN → explicit: MDN learns Gaussian components end-to-end; we impose a classification bottleneck that regularizes the mixture structure.
- Why not normalizing flows? → discuss: flows are more flexible but harder to interpret; our approach gives a discrete mixture with physically interpretable bins.

### Paper B — FlexibleNN

**Title:** *”Input-Dependent Depth Selection for Compute-Adaptive Neural Networks”*

**Target venues (ranked):**
1. **NeurIPS / ICLR (methodology track).** Claim: ELBO-regularized depth gating learns to allocate compute per input, giving hard-inference savings without accuracy loss.
2. **MNRAS / A&A (astrophysics companion).** Claim: compute-adaptive mass estimation for galaxy clusters, where relaxed vs disturbed clusters naturally require different model complexity.

**Contributions:**
(a) Soft depth gating with ELBO Occam penalty — principled variational framing for adaptive computation.
(b) Hard inference via argmax depth bucketing — near-zero accuracy loss with compute savings proportional to mean selected depth.
(c) Empirical validation on UCI + piecewise-complexity synthetic problems (Layer 1).
(d) Galaxy cluster mass headline experiment on IllustrisTNG/eROSITA, showing depth correlates with cluster dynamical state.
(e) **Future work framing:** depth gating applied to FT-Transformer blocks → variable transformer depth per input → path to LLM adaptive computation (Paper C).

**Story arc:** Motivation (fixed-depth NNs waste compute on easy inputs; existing adaptive computation methods lack a principled prior) → framework (FlexibleNN shared/independent weights + gate network) → theory (ELBO depth prior, hard inference) → strategy comparison (SoftGating vs Gumbel vs STE vs REINFORCE) → UCI + synthetic benchmarks → cluster mass headline → future work (transformer depth gating).

**Reviewer attack surface:**
- Novelty vs ACT (Graves 2016) / Universal Transformers → explicit: ACT uses a continuous halting probability; we use a discrete depth selection with an ELBO prior.
- Novelty vs MIND (ICLR 2025) → explicit: MIND uses fixed-point iteration; we use a learned gate with per-input depth prediction.
- Compute savings only matter at scale → address with wall-clock measurements at max_depth ∈ {3, 5, 8, 10}.

### Shared concerns for both papers

- **Benchmark set too narrow if we skip UCI.** §2 addresses this.
- **No comparison with NGBoost / deep ensembles / MDNs.** §4 addresses this.
- **ECE is not standard.** §3 / bug N3 addresses this.
- **No real-world large-scale evaluation.** §6 addresses this (PZ for Paper A, cluster mass for Paper B).
- **No theory-side guarantees.** Fine for empirical methods papers; frame as empirical contributions.

---

## 8. Execution Roadmap

Phases 6–9 are shared infrastructure. Phases 10–11 are paper-specific. Papers can be written in parallel or sequentially (Paper A first is recommended since ProbReg has more benchmark results already).

**Phase 6 — Bug fixes (1 week).** Fix all 7 new bugs (N1–N7). Add regression tests for each. Re-run the existing benchmark suite to confirm no numbers change except those that should (ELBO depth prior, ECE). Update `benchmarks.md` with post-fix results.

**Phase 7 — Metrics + visualizations (1.5 weeks).** Implement §3: `PredictiveDistribution` protocol, CRPS, Winkler, PIT calibration curve, correct ECE, sharpness, pinball loss, critical-difference diagram. Implement §3.8 publication plots: `utils/publication_plots.py` with all 14 plot types. Tests.

**Phase 8 — Comparison baselines (1 week).** Wrap Tier A models (§4.1): NGBoost, MDN, QR-NN, Deep Ensembles, GP, constant-variance NN (as ProbReg config alias). FT-Transformer and TabPFN wrappers. Tests.

**Phase 9 — Expanded benchmarks + full ablations (2.5 weeks).** Implement synthetic datasets B1–B5 (organized by paper). Add UCI loaders (`utils/uci_datasets.py`). Run the full ablation sweeps from §2.3:
- ProbReg ablation: 3 regression strategies × {NLL, β-NLL} × {with/without symlog} × {joint, GRADIENT_STOP} × dynamic-k strategies
- FlexibleNN ablation: depth selection strategies × max_depth values × {shared, independent weights} × {soft, hard inference}
- Baseline sweep: all Tier A models on all datasets
Populate master results tables. This is the bulk of the compute.

**Phase 10 — Paper A: Photometric redshift (2 weeks).** Acquire LSST DC2 (or SDSS) + matched spec-z. Implement RAIL wrapper for ProbReg. Run full ProbReg + baseline suite. Produce ProbReg-specific plots (k heatmap, multimodal density, bin overlay, PIT). Cross-check against RAIL-published numbers.

**Phase 11 — Paper B: Galaxy cluster mass (1.5 weeks).** Download IllustrisTNG observables. Run FlexibleNN + baseline suite. Produce FlexibleNN-specific plots (depth map by cluster state, soft-vs-hard scatter, wall-clock bars, depth histogram). Cross-validate on simulation, apply to eROSITA/Planck.

**Phase 12A — Paper A draft (2 weeks).** ProbReg methods paper. Freeze tagged commit. Submit to arXiv. Target: ICLR 2027 (late September 2026 submission).

**Phase 12B — Paper B draft (2 weeks, can overlap with 12A).** FlexibleNN methods paper. Frame FT-Transformer depth gating as future work. Target: NeurIPS 2027 (May 2027 submission) or same ICLR deadline if ready.

**Total calendar time:** ~10 weeks of focused work for the empirical foundation of both papers, then ~3 weeks for writing (with overlap). **Paper A alone: ~8 weeks to submission.**

---

## Appendix A — Concrete fix locations for §1.1

For the 1-week bug-fix phase, here are the exact changes:

```python
# N1: automl_package/models/flexible_neural_network.py:250
# BEFORE:
depth_prior_logits = torch.arange(self.max_hidden_layers, 0, -1, dtype=torch.float, device=_batch_x.device)
# AFTER:
depth_prior_logits = torch.linspace(3.0, 1.0, self.max_hidden_layers, dtype=torch.float, device=_batch_x.device)
# (Apply same fix to independent_weights_flexible_neural_network.py:303)
```

```python
# N2: automl_package/models/selection_strategies/base_selection_strategy.py:82
# REWRITE _hard_selection_logic to use the weighted-sum pattern that preserves STE gradients,
# mirroring layer_selection_strategies.SteStrategy.forward lines 176-180. The key is:
#     aggregated += mode_selection_one_hot[:, i:i+1] * predictions_for_mode_full_batch
# rather than indexed assignment. This runs all k paths on every sample (cost: O(n_modes) per step)
# but preserves gradients through mode_selection_one_hot.
```

```python
# N3: automl_package/utils/metrics.py:155
# REPLACE calculate_ece with the Kuleshov et al. (2018) PIT-based formulation.
# Keep the old function as calculate_ece_legacy with a DeprecationWarning.
```

```python
# N4: automl_package/utils/losses.py:37
# BEFORE:
per_sample_nll = 0.5 * (log_var + ((targets - mean) ** 2) / variance)
# AFTER:
per_sample_nll = 0.5 * (math.log(2 * math.pi) + log_var + ((targets - mean) ** 2) / variance)
```

```python
# N5: automl_package/utils/losses.py:74
# BEFORE:
hess_log_var = 0.5 * (((y_true - mean) ** 2) / variance)
# AFTER:
hess_log_var = np.full_like(y_true, 0.5)   # Fisher information for log_var (Gaussian)
```

```python
# N6: automl_package/models/base_pytorch.py:125-138
# Delete the conditional build_model calls at L127 and L131; keep only the unconditional call at L138.
# Verify with a test that the code path that previously triggered L127/131 still works.
```

```python
# N7: automl_package/models/selection_strategies/layer_selection_strategies.py:43
# BEFORE:
n_probs = torch.zeros(x_input.size(0), self.model.max_hidden_layers + 1, device=x_input.device)
# AFTER:
n_probs = torch.zeros(x_input.size(0), self.model.max_hidden_layers, device=x_input.device)
if self.model.max_hidden_layers > 0:
    n_probs[:, -1] = 1.0
```

All seven fixes are one-to-three line changes (N2 is the largest at maybe 15 lines). Total fix-phase effort: 1–2 days of code, 2–3 days of tests, 1–2 days of benchmark re-run and comparison.

---

## Appendix B — Recommended reading (verified, with arXiv / DOI where available)

All citations below were verified against the papers’ arXiv abstracts or journal pages on 2026-04-11. Entries marked with `[v]` have had their title, year, and venue explicitly confirmed.

**Proper scoring rules & calibration**

| Paper | Relevance |
|---|---|
| `[v]` Gneiting & Raftery (2007) — *Strictly Proper Scoring Rules, Prediction, and Estimation*, JASA 102, 359 | CRPS, log-score, energy score; the foundational reference for proper scoring. |
| `[v]` Kuleshov, Fenner, Ermon (2018) — *Accurate Uncertainties for Deep Learning Using Calibrated Regression*, ICML 2018, arXiv:1807.00263 | The PIT-based regression calibration used to replace the current ECE (bug N3). |
| `[v]` Hernández-Lobato & Adams (2015) — *Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks*, ICML 2015, arXiv:1502.05336 | The UCI benchmark protocol (20 random 90/10 splits, test NLL and RMSE). |
| `[v]` Lakshminarayanan, Pritzel, Blundell (2017) — *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*, NeurIPS 2017, arXiv:1612.01474 | Deep ensembles baseline. |
| `[v]` Seitzer, Tavakoli, Antić, Martius (2022) — *On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks*, ICLR 2022, arXiv:2203.09168 | β-NLL source paper. Code: github.com/martius-lab/beta-nll. |
| Kendall & Gal (2018) — *Multi-task learning with uncertainty weighting*, CVPR 2018 | For the joint-loss weighting ablation in §1.3. |
| Angelopoulos & Bates (2022) — *A Gentle Introduction to Conformal Prediction*, arXiv:2107.07511 | Conformal extension future work. |

**Tabular ML baselines**

| Paper | Relevance |
|---|---|
| `[v]` Duan et al. (2020) — *NGBoost: Natural Gradient Boosting for Probabilistic Prediction*, ICML 2020, arXiv:1910.03225 | Primary tree-based probabilistic regression baseline. |
| Bishop (1994) — *Mixture Density Networks*, Tech. Report NCRG/4288, Aston University | MDN baseline; the original mixture-density-network reference. |
| `[v]` Gorishniy, Rubachev, Khrulkov, Babenko (2021) — *Revisiting Deep Learning Models for Tabular Data*, NeurIPS 2021, arXiv:2106.11959 | FT-Transformer baseline. Code: github.com/yandex-research/rtdl. |
| `[v]` Hollmann et al. (2023) — *TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second*, ICLR 2023, arXiv:2207.01848 | TabPFN v1 (classification only). Included for completeness. |
| `[v]` Hollmann et al. (2024) — *Accurate predictions on small data with a tabular foundation model*, Nature (December 2024) | **TabPFN v2 (regression)** — this is the version to use as a baseline. Code: github.com/PriorLabs/TabPFN. Non-optional. |
| `[v]` Hamidieh (2018) — *A data-driven statistical model for predicting the critical temperature of a superconductor*, Comput. Mater. Sci. 154, 346 | Source for the UCI Superconductivity dataset used in Layer 2. |
| ProbSAINT (2024) — arXiv:2403.03812 | Recent probabilistic transformer for tabular regression; worth citing as a 2024 comparable. |

**Photometric redshift (for §6 primary)**

| Paper | Relevance |
|---|---|
| `[v]` Sadeh, Abdalla, Lahav (2016) — *ANNz2: Photometric Redshift and Probability Distribution Function Estimation using Machine Learning*, PASP 128, 104502, arXiv:1507.00490 | **NOTE: PASP, not MNRAS** (original plan draft had this wrong). Direct PZ ML baseline with PDFs. Code: github.com/IftachSadeh/ANNZ. |
| `[v]` Almosallam, Jarvis, Roberts (2016) — *GPz: non-stationary sparse Gaussian processes for heteroscedastic uncertainty estimation in photometric redshifts*, MNRAS 462, 726, arXiv:1604.03593 | Bayesian heteroscedastic PZ baseline. Code: github.com/OxfordML/GPz. |
| `[v]` Carrasco Kind & Brunner (2013) — *TPZ: photometric redshift PDFs and ancillary information by using prediction trees and random forests*, MNRAS 432, 1483, arXiv:1303.7269 | Tree-based PZ with PDFs. Code: matias-ck.com/mlz/. RAIL wrapper: github.com/LSSTDESC/rail_tpz. |
| `[v]` D'Isanto & Polsterer (2018) — *Photometric redshift estimation via deep learning: Generalized and pre-classification-less, image based, fully probabilistic redshifts*, A&A 609, A111, arXiv:1706.02467 | CNN+MDN baseline. Uses 5 Gaussians. The closest existing multimodal-head comparable to ProbReg. |
| `[v]` Pasquet, Bertin, Treyer, Arnouts, Fouchez (2019) — *Photometric redshifts from SDSS images using a convolutional neural network*, A&A 621, A26, arXiv:1806.06607 | CNN-as-classifier on SDSS images. Achieves σ_MAD < 0.01 at z < 0.4. Different modality but a core reference. |
| DeepDISC-photoz (2024) — arXiv:2411.18769 | Most recent RAIL-integrated PZ method (instance segmentation + CNN on LSST simulated data). State-of-the-art target. |
| Jones, Do et al. (2024) — *Improving Photometric Redshift Estimation for Cosmology with LSST using Bayesian Neural Networks*, ApJ 964, 130, arXiv:2306.13179 | Bayesian NN PZ for LSST — closest framing to our paper. High reading priority. |
| LSSTDESC RAIL | github.com/LSSTDESC/rail — the PZ evaluation pipeline we plug into (not a paper but the key piece of infrastructure). |
| DESC DC2 Data Release Note (2021) — arXiv:2101.04855 | Reference for the cosmoDC2 catalog used as the headline dataset. |

**Galaxy cluster mass (for §6 — Paper B headline)**

| Paper | Relevance |
|---|---|
| `[v]` Ntampaka et al. (2019) — *A Deep Learning Approach to Galaxy Cluster X-ray Masses*, ApJ 876, 82, arXiv:1810.07703 | CNN cluster-mass baseline. 7896 mock Chandra images from IllustrisTNG, bias −0.02 dex, scatter 8–12%. |
| `[v]` Ho et al. (2019) — *A Robust and Efficient Deep Learning Method for Dynamical Mass Measurements of Galaxy Clusters*, ApJ 887, 25 | Bayesian NN dynamical mass estimation — direct baseline for Bayesian cluster mass. |
| Krippendorf et al. (2024) — *Galaxy cluster characterization with machine learning techniques*, arXiv:2501.04081 | Recent review of ML cluster characterization. Context citation. |
| `[v]` hNN — Hybrid Neural Network for cluster triaxiality (2025) — arXiv:2511.20429 | CNN+GNN fusion on MillenniumTNG. Most modern architecture for cluster characterization. |
| `[v]` Score-based cluster mass maps (2024) — arXiv:2410.02857 | Generative model for gas/DM maps from SZ+X-ray. Cutting-edge probabilistic approach. |
| `[v]` AE-CNN cluster mass (2025) — arXiv:2507.21876 | Autoencoder-CNN posterior mean estimation. Recent probabilistic framework. |
| IllustrisTNG public API | tng-project.org/data/. Web API for extracting halos + observables. Used to build the training set. |

**Photometric redshift — post-2022 modern methods (for §6 — Paper A headline)**

| Paper | Relevance |
|---|---|
| `[v]` Zephyr (2023) — arXiv:2310.20125 | Normalizing flow + mixture density for PZ with heterogeneous training data. Modern flow-based competitor. |
| `[v]` nflow-z (2025) — arXiv:2510.10032 | Conditional normalizing flow (cINN, cNSF) for PZ PDFs. Tested on CSST, COSMOS2020, DES Y1, SDSS, DECaLS. |
| `[v]` ViT-MDNz (2026) — arXiv:2602.22711 | First vision transformer for PZ + MDN. σ_MAD = 2.6%. Image modality but shows transformer+MDN frontier. |
| `[v]` RAIL v2 (2025) — arXiv:2505.02928 | The DESC evaluation framework paper. Establishes the standard protocol for PZ comparison. |

**Adaptive computation (for Paper B related work)**

| Paper | Relevance |
|---|---|
| `[v]` MIND (2025) — ICLR 2025 | Dynamic computation via fixed-point iteration. Distinct from our discrete depth selection + ELBO prior. |

**Historical / classical (referenced in main text)**

| Paper | Relevance |
|---|---|
| Hafner et al. (2023) — *DreamerV3* | Two-hot value encoding — closest prior art for regression-as-classification without uncertainty. |
| Teerapittayanon, McDanel, Kung (2016) — *BranchyNet* | Early exit inference. |
| Dehghani et al. (2019) — *Universal Transformers* | ACT with ponder cost — reference for depth penalty. |
| Graves (2016) — *Adaptive Computation Time for Recurrent Neural Networks*, arXiv:1603.08983 | Original ACT formulation. |

---

## 9. Grounding and Confidence

This section exists because the user explicitly asked “is this plan grounded in research?” The answer is layered. I want to be explicit about what I have verified, what I am confident about from standard training data, and what I have left as informed judgment.

**Strongly grounded (verified by direct code reading or explicit WebFetch/WebSearch on 2026-04-11):**

- **All 7 new bugs in §1.1.** Every bug was verified line-by-line against the code at HEAD 597a416. No inference, no guessing.
- **§1.2 theoretical review.** Law of Total Variance, β-NLL mechanics, ELBO k-selection, Monte-Carlo symlog uncertainty, conformal split-prediction — these are all standard textbook / paper material I have high confidence in.
- **All §3 metrics.** CRPS, Winkler, PIT-based calibration, sharpness, pinball loss, critical-difference diagrams are standard.
- **§4 comparison models with verified citations:** NGBoost (Duan 2020, ICML, arXiv:1910.03225), Deep Ensembles (Lakshminarayanan 2017, NeurIPS, arXiv:1612.01474), FT-Transformer (Gorishniy 2021, NeurIPS, arXiv:2106.11959), TabPFN v2 (Hollmann 2024, Nature).
- **§6 photometric redshift references — all verified.** Sadeh 2016 ANNz2 (**correction: PASP, not MNRAS, as originally written**), Almosallam 2016 GPz (MNRAS 462, 726, arXiv:1604.03593), Carrasco Kind & Brunner 2013 TPZ (MNRAS 432, 1483, arXiv:1303.7269), D'Isanto & Polsterer 2018 DCMDN (A&A 609, A111, arXiv:1706.02467), Pasquet 2019 (A&A 621, A26, arXiv:1806.06607), DeepDISC-photoz (arXiv:2411.18769, Nov 2024).
- **LSSTDESC RAIL** infrastructure verified at github.com/LSSTDESC/rail with multiple sub-packages (rail_delight, rail_lephare, rail_cmnn, rail_som, rail_tpz, rail_deepdisc, rail_fsps, rail_pipelines).
- **LSST DC2 data access verified**: catalog `cosmoDC2_v1.1.4_image_with_photoz_v1` via GCRCatalogs at NERSC. **The known BPZ high-z systematic flagged in the DC2 documentation is a specific research opportunity** our ML approach can address — this is the most defensible framing I found.
- **§6 cluster mass references**: Ntampaka 2019 (arXiv:1810.07703, ApJ 876, 82 — verified title and mass bias statistics), Ho 2019 (ApJ 887, 25 — verified), Krippendorf 2024 review (arXiv:2501.04081).
- **IllustrisTNG** data access at tng-project.org/data/ with full public web API, JupyterLab interface, ~1.1 PB of snapshots.
- **All Layer 2 physics tabular datasets** (UCI Airfoil Self-Noise, UCI Superconductivity with Hamidieh 2018 reference, NASA Exoplanet Archive for Kepler KOI, UCI HTRU2 with Lyon 2016 reference).
- **Boston Housing deprecation**: confirmed removed from scikit-learn 1.2 due to ethically problematic engineered feature. The plan now excludes Boston Housing and uses California Housing as the standard replacement.

**Grounded by judgment, not by a specific cited paper**:

- **The five synthetic problems B1–B5 in §2.2 Layer 1 are my original designs.** They are not standard benchmarks. The plan now explicitly states this up front, so the paper cannot be criticized for presenting them as more than what they are (mechanism tests with known ground-truth Bayesian posteriors where computable).
- **The two-paper split** (Paper A = ProbReg, Paper B = FlexibleNN) reflects user direction. Venue targeting (NeurIPS/ICLR for methods, MNRAS for astro companion) is standard practice.
- **Post-2022 astrophysics references** (Zephyr, nflow-z, ViT-MDNz, hNN, score-based cluster maps) were verified via WebSearch on 2026-04-12 with arXiv IDs confirmed. These confirm both problems have active modern-ML literature.
- **Benchmark compute budgets** (10 min CPU, 30 min XPU per model per dataset) are rules of thumb.

**Unverified details I deliberately left loose** (can be tightened once the actual work begins):

- **Exact SDSS DR17 galaxy count.** I said "~1M galaxies" as order-of-magnitude. The precise number depends on cuts and is only relevant at data-download time.
- **DES Y6 status** — I mentioned it as a secondary generalization test without verifying the latest data release.
- **Kepler KOI exact split between confirmed/candidate/false-positive** — I gave 2358/2366/4840 based on search results; this shifts over time as the archive updates.
- **RAIL sub-package list may be incomplete** — I listed what I saw; the current RAIL ecosystem may have added more since.

**What I did not verify but am confident in from standard ML knowledge:**

- Koenker 2005 pinball-loss reference for quantile regression — textbook.
- Daxberger et al. 2021 `laplace-torch` for Laplace BNN — well-known package.
- Demšar 2006 critical-difference diagram — standard in benchmarking literature.
- Gneiting & Raftery 2007 strictly proper scoring rules — textbook.
- Bishop 1994 MDN reference — foundational, well-known.

If any verification step blocks the plan (e.g. RAIL doesn’t accept our model format, DC2 access is restricted), I will flag it at execution time. None of the known-unverified items are on the critical path.

---

## Open questions for the human

Before committing to the roadmap, confirm:

1. **Astrophysics dataset choice.** Is LSST DC2 + SDSS spec-z pairing acceptable, or do you have a preferred dataset from your prior research (e.g. ACT cluster catalogs, DES PZ)?
2. **Single paper vs two.** Methods paper first, then astro paper? Or combined submission to MNRAS from the start?
3. **Compute envelope.** The full benchmark sweep (UCI + synthetic + PZ + cluster) with all baselines and ablations is ~2 weeks of wall-clock on the XPU. Is that acceptable, or should we prune?
4. **Dynamic n_classes STE (bug N2).** Fix and benchmark, or remove the strategy from the public API since SoftGating empirically dominates?
5. **Scope creep control.** The conformal extension (locally-adaptive), multi-target regression, and transformer internals are all natural extensions but not needed for the paper. Confirm we defer them.

Once these are answered I can start executing Phase 6 (bug fixes) immediately.
