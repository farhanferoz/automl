# AutoML Toolkit — Architecture Analysis & SOTA Comparison

**Date:** 2026-04-06
**Status:** Reviewed and approved by author

This document provides a comprehensive analysis of the two core experimental architectures
in the AutoML toolkit: **Probabilistic Regression** and **Flexible Neural Network**.
It includes a complete bug audit, comparison with state-of-the-art approaches,
theoretical viability assessment, and a concrete improvement roadmap.

---

## 1. Bug Audit

### 1.1 Summary Table

| # | Bug | Severity | Location | Status |
|---|-----|----------|----------|--------|
| 1 | `_weighted_average_logic` returns 2 values, callers expect 3 | CRITICAL | `base_selection_strategy.py:48-72`, `n_classes_strategies.py:75` | Crash — dynamic n_classes path |
| 2 | Dynamic n_classes strategies return 4 values, `ProbabilisticRegressionNet.forward` expects 5 | CRITICAL | `n_classes_strategies.py:76`, `probabilistic_regression_net.py:147` | Crash — dynamic n_classes path |
| 3 | Dynamic n_classes strategies don't accept `boundaries` kwarg | CRITICAL | `probabilistic_regression_net.py:148` | Crash — dynamic n_classes path |
| 4 | `_weighted_average_logic` passes raw `x_input` instead of classifier logits | CRITICAL | `base_selection_strategy.py:68` | Wrong math (unreachable due to Bug 1) |
| 5 | REINFORCE `log_prob` assigned to wrong tuple position | CRITICAL | `layer_selection_strategies.py:198-204`, `flexible_neural_network.py:209` | Dead n_predictor — policy gradient always zero |
| 6 | STE strategy uses `argmax` destroying gradient flow to n_predictor | CRITICAL | `layer_selection_strategies.py:153-167` | Dead n_predictor — no gradient |
| 7 | `IndependentWeightsSteStrategy` returns soft probs, not STE one-hot | CRITICAL | `independent_weights_strategies.py:181-198` | STE functionally identical to SoftGating |
| 8 | Double `log()` on log_variance in `calculate_combined_loss` | CRITICAL | `losses.py:39-42` | Corrupts ALL probabilistic regression training |
| 9 | `middle_class_dist_params_` self-referential dict assignment | HIGH | `probabilistic_regression.py:325-339` | Wrong penalty params for all k except last |
| 10 | Duplicate `build_model()` call in IndependentWeightsFlexibleNN | MODERATE | `independent_weights_flexible_neural_network.py:197-198` | Wasted compute — copy-paste error |
| 11 | Showcase indexes 1D array as 2D | MODERATE | `probabilistic_regression_showcase.py:253-257` | Showcase crash (`IndexError`) |

### 1.2 Net Effect

- **Probabilistic Regression:** Only the fixed-k `NONE` strategy path works at all. Every dynamic
  n_classes strategy (`GUMBEL_SOFTMAX`, `SOFT_GATING`, `STE`, `REINFORCE`) has three independent
  crash bugs (1, 2, 3). Even the working path has corrupted variance estimation (Bug 8).
- **FlexibleNN (shared-weights):** `GumbelSoftmax` and `SoftGating` work correctly. `STE` (Bug 6)
  and `REINFORCE` (Bug 5) have dead n_predictors.
- **FlexibleNN (independent-weights):** `GumbelSoftmax` and `SoftGating` work. `STE` (Bug 7)
  returns wrong values. `REINFORCE` is broken by extension.

### 1.3 Bug Details

#### Bug 1: `_weighted_average_logic` returns 2 values, callers expect 3

**Files:** `base_selection_strategy.py` L48-72 (returns 2), `n_classes_strategies.py` L75 (unpacks 3)

`_weighted_average_logic` returns `(final_predictions_contribution, selected_k_values)` — a 2-tuple.
But `GumbelSoftmaxStrategy.forward()` (L75) and `SoftGatingStrategy.forward()` (L100) unpack 3 vars:

```python
final_predictions_contribution, selected_k_values_for_logging, classifier_raw_logits = self._weighted_average_logic(...)
```

Immediate `ValueError: not enough values to unpack`. Never executed successfully.

#### Bug 2: Dynamic strategies return 4, caller expects 5

**Files:** `n_classes_strategies.py` L76 (returns 4), `probabilistic_regression_net.py` L147 (unpacks 5)

`NoneStrategy.forward()` correctly returns 5 values (including `per_head_outputs`).
All other strategies return only 4. The caller:

```python
final_predictions, selected_k_values, log_prob_for_reinforce, classifier_logits_out, per_head_outputs = self.n_classes_strategy.forward(...)
```

#### Bug 3: Missing `boundaries` kwarg

**File:** `probabilistic_regression_net.py` L148

Call passes `boundaries=boundaries` but `GumbelSoftmaxStrategy.forward()`, `SoftGatingStrategy.forward()`,
`SteStrategy.forward()`, and `ReinforceStrategy.forward()` signatures are `forward(self, x_input, logits)`.
Produces `TypeError`.

**Conclusion (Bugs 1-3):** Dynamic n_classes has never been runnable. Three independent crashes.

#### Bug 4: `_weighted_average_logic` passes raw features as logits

**File:** `base_selection_strategy.py` L68

```python
predictions_for_mode = self.model._compute_predictions_for_k(x_input, k_val)
```

`_compute_predictions_for_k` expects classifier raw logits as first arg. Passing raw `x_input`
means softmaxing arbitrary feature values. Compare with `_hard_selection_logic` L80 which correctly
calls `self.model.classifier_layers(x_input)` first. Currently unreachable due to Bug 1.

#### Bug 5: REINFORCE log_prob in wrong tuple position

**Files:** `layer_selection_strategies.py` L198-204, `flexible_neural_network.py` L209

Return order from `ReinforceStrategy.forward()`:

| Position | Value |
|----------|-------|
| 0 | `final_output` |
| 1 | `n_actual` |
| 2 | **`log_prob`** (real) |
| 3 | `mode_selection_probs` |
| 4 | **`torch.tensor(0.0)`** (dummy) |

Training loop unpacking: `final_output, _, _, n_logits, log_prob = self.model(_batch_x)`

Variable `log_prob` receives `torch.tensor(0.0)` (position 4). Actual is discarded.
Policy loss = `-0.0 * reward = 0.0` every batch.

#### Bug 6: STE destroys gradient via argmax

**File:** `layer_selection_strategies.py` L153-167

```python
n_probs = f.gumbel_softmax(logits, tau=..., hard=True, dim=1)  # Has STE gradients
chosen_indices = torch.argmax(n_probs, dim=1)                  # Destroys graph
n_actual = chosen_indices + 1
active_mask = (i < n_actual).unsqueeze(1)                      # Boolean, no grad
current_output = torch.where(active_mask, block_output, current_output)
```

`n_probs` carries STE gradients but is never used to weight outputs. Only `argmax(n_probs)` is used,
stripping all gradient info. Hidden blocks train normally; **n_predictor receives zero gradient**.

Note: GumbelSoftmax and SoftGating strategies for shared-weights do NOT have this bug — they
correctly use `prob.unsqueeze(1) * output_for_n`.

#### Bug 7: IndependentWeights STE returns soft probs

**File:** `independent_weights_strategies.py` L181-198

```python
def select_n(self, x_input, n_logits):
    n_probs = f.softmax(n_logits, dim=-1)                     # Soft probabilities
    n_actual_one_hot = f.gumbel_softmax(n_logits, hard=True)   # STE one-hot (UNUSED for mixing)
    n_actual = torch.argmax(n_actual_one_hot, dim=-1) + 1
    return n_actual, n_probs, n_logits, log_prob               # Returns n_probs, NOT one-hot
```

Caller uses `n_probs_tensor` for weighted sum. Since it's `f.softmax` (smooth), this is pure
soft gating. The `gumbel_softmax(hard=True)` computation is wasted.

#### Bug 8: Double log() on log_variance

**File:** `losses.py` L39-42

```python
if uncertainty_method == UncertaintyMethod.PROBABILISTIC:
    mean = predictions[:, 0]
    log_var = torch.log(torch.clamp(predictions[:, 1], min=1e-6))  # BUG
    regression_loss = nll_loss(torch.stack((mean, log_var), dim=1), y_true_squeezed)
```

The model architecture already outputs `log_variance` as `predictions[:, 1]`:
- `apply_law_of_total_variance` (pytorch_utils.py:132): `final_log_var = torch.log(torch.clamp(final_variance, min=1e-9))`
- BaseRegressionHead raw output: linear layer outputs scalar intended as log_var

Then `calculate_combined_loss` applies `torch.log()` again → `log(log_var)` into NLL.
Corrupts variance estimation and gradient signal for ALL probabilistic regression training.

**Fix:** Remove the `torch.log(torch.clamp(...))` wrapper. Use `predictions[:, 1]` directly.

#### Bug 9: Self-referential dict assignment

**File:** `probabilistic_regression.py` L325-339

```python
self.middle_class_dist_params_ = {}
for k in k_values:
    self._calculate_middle_class_dist_params(y_flat, y_binned, n_classes=k)
    # Mixin replaces self.middle_class_dist_params_ with {"mean": ..., "std": ...}
    self.middle_class_dist_params_[k] = self.middle_class_dist_params_  # SELF-REFERENTIAL
```

Inserts dict into itself. After loop: `{"mean": last, "std": last, last_k: <circular_ref>}`.
`.get(k_int)` returns `None` for all k except the last.

#### Bug 10: Duplicate build_model

**File:** `independent_weights_flexible_neural_network.py` L197-198

```python
self.model = self.IndependentWeightsFlexibleNNModule(self).to(self.device)
self.model = self.IndependentWeightsFlexibleNNModule(self).to(self.device)  # Duplicate
```

First construction immediately discarded. Harmless but wastes memory.

#### Bug 11: Showcase IndexError

**File:** `probabilistic_regression_showcase.py` L253-257

```python
y_pred_pr = pr_model.predict(x_test)          # Returns 1D (mean only)
mse_pr = mean_squared_error(y_test, y_pred_pr[:, 0])  # IndexError
y_pred_std_pr = np.sqrt(np.maximum(y_pred_pr[:, 1], 1e-9))  # Also wrong: treats as variance not log_var
```

`predict()` for PROBABILISTIC returns `model_output[:, 0]` (mean only), a 1D array.
Additionally, even if 2D, column 1 is **log_variance**, not variance. Correct conversion:
`np.exp(0.5 * y_pred_pr[:, 1])`.

---

## 2. Architecture A: Probabilistic Regression — Design & SOTA Comparison

### 2.1 Core Design

```
Input → Classifier NN → softmax → P(class_i | x)
                                      ↓
                         Regression Heads → E[Y|class_i], Var[Y|class_i]
                                      ↓
                         Law of Total Variance → final mean, final variance
```

Three regression strategies provide different information-access tradeoffs:

| Strategy | Head Input | Head Output | Key Property |
|----------|-----------|-------------|--------------|
| **SEPARATE_HEADS** | `P(class_i)` — single scalar per head | `(mean_i, log_var_i)` per bin | Strongest regularization via probability bottleneck; supports monotonic constraints |
| **SINGLE_HEAD_N_OUTPUTS** | Full probability vector `[P(c_1),...,P(c_k)]` | All `(mean_i, log_var_i)` jointly | Shared network sees all probabilities; moderate regularization |
| **SINGLE_HEAD_FINAL_OUTPUT** | Full probability vector | Single `(mean, log_var)` directly | Lightest architecture — weighted regression |

When `n_classes >= n_classes_inf`, the model falls back to **pure direct regression** — the
classification path is bypassed entirely. This gives the architecture a continuous spectrum
from "pure classification-based" (low k) to "pure regression" (k=∞).

### 2.2 SOTA Comparison

| Dimension | Your Approach | DreamerV3/MuZero (Two-Hot) | Mixture Density Networks | Quantile Regression |
|-----------|--------------|---------------------------|--------------------------|---------------------|
| **Bins** | Flexible (2 to ∞), learnable k | Many fixed (255–601) | N/A (continuous Gaussians) | N/A (quantile based) |
| **Per-bin output** | Learned regression heads (mean+var per bin) | Bin midpoints (fixed) | μ_k, σ_k, π_k per component | τ-th quantile |
| **Decoding** | Law of Total Variance | Expected value over two-hot | Weighted mixture | Quantile interpolation |
| **Loss** | NLL + Cross-entropy (joint) | Cross-entropy only | NLL of mixture | Pinball loss |
| **Uncertainty** | Yes (aleatoric via total variance) | Not explicit | Yes (mixture spread) | Yes (quantile spread) |
| **Regularization** | Built-in via probability bottleneck | None (fixed bins) | None (raw features to components) | None |

### 2.3 Strengths (Genuine Novelty)

1. **Law of Total Variance decomposition is statistically rigorous.** Unlike DreamerV3 which only
   recovers a point estimate, this design recovers both mean AND calibrated variance by correctly
   applying `Var(Y) = E[Var(Y|C)] + Var(E[Y|C])`. No SOTA regression-as-classification system
   does this.

2. **The probability bottleneck in SEPARATE_HEADS is deliberate regularization.** Providing only
   `P(class_i)` to each head is intentional dimensionality reduction. The classifier acts as a
   learned feature extractor that compresses input into a probability simplex. The regression
   heads learn what each confidence level means for the target value. This is analogous to
   autoencoder bottlenecks. The SINGLE_HEAD_N_OUTPUTS variant provides a middle ground.

3. **Per-bin regression heads are strictly more expressive than fixed bins.** DreamerV3 uses fixed
   bin midpoints. This design learns flexible functions `P(class_i) → E[Y|class_i]`, enabling
   non-uniform value mappings within each bin region.

4. **Monotonic constraints are well-motivated.** Left-bin head monotonically decreasing, right-bin
   increasing (in probability) ensures "more confident → more extreme prediction" — correct for
   ordered partitions.

5. **The k=∞ fallback provides graceful degradation.** When data doesn't benefit from
   discretization, the model can use direct regression. No other approach has this safety net.

6. **Three regression strategies span a principled regularization spectrum:**
   - SEPARATE_HEADS: Maximum regularization (each head isolated)
   - SINGLE_HEAD_N_OUTPUTS: Moderate (shared network, all probabilities visible)
   - SINGLE_HEAD_FINAL_OUTPUT: Minimum (direct probability-to-output mapping)

### 2.4 Design Considerations

**Bin count sweet spot:** With very few bins (k=3), the distribution is coarse. With hundreds,
regression heads add little over midpoints. The architecture's unique value lives in the
moderate range (5-20), where bins provide structure but heads still do substantial refinement.
For simple unimodal targets, k=3-5 may suffice.

**Dynamic k — hard but not impossible:** Changing k changes classification targets, creating
non-stationary optimization. Not fundamentally doomed — an ELBO-based formulation where k is
a discrete latent variable with a simplicity-favoring prior provides probabilistic motivation.
The KL term naturally penalizes unnecessarily high k.

**Optimization strategy tradeoffs:**

| Strategy | Classifier gradient source | Risk |
|----------|---------------------------|------|
| REGRESSION_ONLY | Regression loss only | Classifier may not learn meaningful bins |
| GRADIENT_STOP | Classification loss only | Optimizes binning accuracy, not regression quality |
| JOINT | Both losses | Conflicting gradients; needs loss weighting |

### 2.5 Viability Verdict

| Mode | Verdict | Reasoning |
|------|---------|-----------|
| Fixed-k, SEPARATE_HEADS, JOINT | **YES — after Bug 8 fix** | Sound architecture, proper uncertainty |
| Fixed-k, SINGLE_HEAD_N_OUTPUTS | **YES** | Good balance of expressiveness and structure |
| Fixed-k, SINGLE_HEAD_FINAL_OUTPUT | **MARGINAL** | Loses per-class variance structure |
| Dynamic-k (current) | **NO — needs redesign** | Three crash bugs + adversarial optimization |
| Dynamic-k (ELBO-based) | **PLAUSIBLE — research** | Variational formulation makes it tractable |

---

## 3. Architecture B: Flexible Neural Network — Design & SOTA Comparison

### 3.1 Core Design

**Shared-weights (FlexibleHiddenLayersNN):**
```
Input → n_predictor → P(depth=1), ..., P(depth=max)
Input → Block_1 → Block_2 → ... → Block_max → Output_layer
         Soft/hard selection of which depth's output to use
```

**Independent-weights (IndependentWeightsFlexibleNN):**
```
Input → n_predictor → P(depth=1), ..., P(depth=max)
Input → Network_1(x)  [1-layer]
Input → Network_2(x)  [2-layer]
Input → Network_3(x)  [3-layer]
         Weighted sum using P(depth=i) as weights
```

**Key distinction from MoE:** In standard MoE, all experts have the same architecture and differ
only in weights. Here, experts have **genuinely different architectures** (different depths).
The motivation is not "specialist routing" but **input-dependent structural complexity** —
simple inputs should use shallow networks, complex inputs should use deep ones.

### 3.2 SOTA Comparison

| Dimension | Shared-Weights | Independent-Weights | Early Exit (BranchyNet) | ACT (Graves 2016) |
|-----------|---------------|--------------------|-----------------------|-------------------|
| **Mechanism** | Gate selects depth's output | Gate selects network's output | Confidence exits early | Halting prob accumulates |
| **Weight sharing** | All depths share blocks | Each depth owns weights | Shared backbone | Shared recurrent cell |
| **Train compute** | Full (all blocks run) | Full (all networks run) | Full (all layers run) | Full (all steps run) |
| **Inference compute** | Reducible via annealing | Reducible via hard selection | Reduced (early exit) | Reduced (halting) |
| **Complexity control** | None currently | None currently | Implicit (threshold) | Ponder cost penalty |

### 3.3 Strengths

1. **Input-dependent depth selection** is well-supported by Early Exit literature. The intuition
   is correct: not all inputs require the same model complexity.

2. **GumbelSoftmax and SoftGating for shared-weights are correctly implemented.** They create
   differentiable weighted averages of per-depth outputs.

3. **Independent-weights with genuinely different architectures** goes beyond standard MoE.
   Each "expert" is a fundamentally different model (different capacity/depth).

4. **Inference-time efficiency is achievable.** With properly annealed Gumbel-Softmax
   (`hard=True` at low temperature), the gate produces near-one-hot selections at inference.
   Only the selected network executes during deployment.

### 3.4 The Depth Complexity Problem: Probabilistic Solutions

Without complexity control, the gate collapses to max depth (more capacity always reduces
training loss). Ranked by probabilistic motivation:

**Approach 1: ELBO with prior over depth (strongest probabilistic foundation)**

Treat depth as discrete latent variable `d ∈ {1,...,D}` with prior `p(d)`:

```
ELBO = E_q(d|x) [log p(y | x, d)] - KL(q(d|x) || p(d))
```

- `q(d|x)` = gate's output (depth distribution from n_predictor)
- `p(d)` = prior (e.g., geometric `p(d) ∝ λ^d` favoring shallow)
- KL term provides automatic Occam's razor

Implementation sketch:
```python
log_likelihood = -criterion(final_output, batch_y)
depth_prior = torch.distributions.Categorical(
    logits=torch.arange(max_depth, 0, -1, dtype=torch.float)
)
kl_div = torch.distributions.kl_divergence(
    torch.distributions.Categorical(probs=depth_probs),
    depth_prior
).mean()
elbo = log_likelihood - kl_div
loss = -elbo
```

**Approach 2: Concrete / Spike-and-Slab relaxation**

Place spike-and-slab prior on each layer's activation gate. Use Concrete (Gumbel-Softmax)
relaxation. KL divergence to spike-and-slab naturally encourages pruning.

**Approach 3: Practical depth regularization (least probabilistic, proven)**

```python
expected_depth = torch.sum(depth_probs * torch.arange(1, max_depth+1), dim=1)
loss += lambda_depth * expected_depth.mean()
```

De facto standard in adaptive computation literature (Universal Transformers, ACT).

### 3.5 Viability Verdict

| Variant | Strategy | Verdict | Notes |
|---------|----------|---------|-------|
| Shared | GumbelSoftmax | **YES — with complexity control** | Add ELBO or depth penalty |
| Shared | SoftGating | **YES — with complexity control** | Same |
| Shared | STE | **NO (currently)** | Bug 6. Fixable, but GumbelSoftmax is better |
| Shared | REINFORCE | **NO (currently)** | Bug 5. High variance even if fixed |
| Independent | GumbelSoftmax | **YES — with complexity control** | Most promising variant |
| Independent | SoftGating | **YES — with complexity control** | Same |
| Independent | STE | **NO (currently)** | Bug 7 |
| Independent | REINFORCE | **MARGINAL** | High variance |

---

## 4. Toy Problem Assessment

### 4.1 `probabilistic_regression_showcase.py` — INSUFFICIENT

Uses **homoscedastic** (constant-variance) noise:
```python
noise = np.random.normal(0, noise_level, n_samples)
```

Problems:
1. **Uncertainty estimation can't shine.** Constant variance means a single global σ captures
   everything. Dynamic bucket allocation adds no value.
2. **Missing multimodality.** The "regression as classification" approach is uniquely powerful
   for multimodal targets. Current dataset is strictly unimodal.

Recommended test data:
- **Heteroscedastic:** `noise = np.random.normal(0, 0.1 + 0.3 * np.abs(x))`
- **Multimodal:** `y = x + sign * np.sin(x)` where `sign` is randomly ±1

### 4.2 `flexible_nn_showcase.py` — GOOD

The piecewise function (linear for x<0, complex sinusoidal for x≥0) is well-designed for
adaptive depth. Models that route simple inputs to shallow paths should outperform fixed-depth.
The fact that flexible models didn't outperform is fully explained by Bugs 5, 6, 7.

---

## 5. Recommended Reading

| Paper | Relevance |
|-------|-----------|
| DreamerV3 (Hafner et al., 2023) | Gold standard for regression-as-classification with two-hot encoding |
| Mixture Density Networks (Bishop, 1994) | Closest academic relative of per-bin regression heads |
| Unbounded Depth Neural Networks (Doshi et al.) | ELBO-based depth selection |
| Universal Transformers (Dehghani et al., 2019) | ACT with ponder cost — practical depth penalty |
| BranchyNet (Teerapittayanon et al., 2016) | Early exit inference patterns |
| Concrete Dropout (Gal et al., 2017) | Differentiable layer/node pruning via VI |
| β-NLL (Seitzer et al., 2022) | Improved NLL for heteroscedastic regression |
| Conformal Prediction (Shafer & Vovk, 2008) | Distribution-free uncertainty quantification |
| Spike-and-Slab VI (Nalisnick & Smola, 2018) | Principled Bayesian structure selection |
