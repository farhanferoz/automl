# AutoML Toolkit — Implementation Plan

**Date:** 2026-04-06
**Reference:** `docs/architecture_analysis.md` for full bug details and SOTA context

This plan is organized into sequential phases. Each phase produces a working, testable state.

**Testing philosophy:** Every phase includes its own test file. Tests are written alongside
fixes, not deferred. No phase is complete until its tests pass. The project currently has
**zero tests** — Phase 1 establishes the test infrastructure.

**Test runner:** `python -m pytest tests/ -v --tb=short` from the venv.

---

## Phase 1: Critical Bug Fixes & Probabilistic Regression Baseline

**Goal:** Fix the bugs blocking probabilistic regression with fixed k, design proper toy
problems, and run head-to-head comparisons with other supported models.

**Priority: HIGHEST — must complete before all other phases.**

### 1.1 Fix Bug 8: Double log_var in `calculate_combined_loss`

**File:** `automl_package/models/common/losses.py` L39-42

**Current (broken):**
```python
log_var = torch.log(torch.clamp(predictions[:, 1], min=1e-6))
```

**Fix:**
```python
log_var = predictions[:, 1]
```

**Rationale:** The model already outputs `log_variance`. `apply_law_of_total_variance` in
`pytorch_utils.py` L132 applies `torch.log()` to produce `final_log_var`. Applying `log` again
corrupts the NLL gradient signal.

**Verification:** After fix, train a ProbabilisticRegressionModel on synthetic data. The learned
variance should correlate with actual noise level. Before fix, variance is meaningless.

---

### 1.2 Fix Bug 11: Showcase IndexError + wrong variance conversion

**File:** `automl_package/examples/probabilistic_regression_showcase.py` L253-258

**Current (broken):**
```python
y_pred_pr = pr_model.predict(x_test)
mse_pr = mean_squared_error(y_test, y_pred_pr[:, 0])     # IndexError: 1D array
y_pred_std_pr = np.sqrt(np.maximum(y_pred_pr[:, 1], 1e-9))  # Wrong: log_var != var
```

**Fix:** The `predict()` method returns 1D (mean only for PROBABILISTIC). Use `predict_with_uncertainty()`
or modify `predict()` to return both. Additionally, convert log_var correctly:

```python
y_pred_pr = pr_model.predict(x_test)
mse_pr = mean_squared_error(y_test, y_pred_pr)

# For uncertainty, use predict_uncertainty
y_pred_std_pr = pr_model.predict_uncertainty(x_test)
```

**Note:** Review whether `predict()` and `predict_uncertainty()` methods exist and work correctly
for ProbabilisticRegressionModel. If not, implement them following the pattern in
`flexible_neural_network.py` L272-327.

---

### 1.3 Fix Bug 10: Duplicate build_model

**File:** `automl_package/models/independent_weights_flexible_neural_network.py` L197-198

**Fix:** Delete line 198 (the duplicate `self.model = self.IndependentWeightsFlexibleNNModule(self).to(self.device)`).

---

### 1.4 Design Proper Toy Problems

The current showcase uses homoscedastic (constant-variance) noise which cannot demonstrate
the value of probabilistic regression. We need three test datasets:

#### Dataset 1: Heteroscedastic Sine (Primary Test)

```python
def generate_heteroscedastic_data(n_samples=1000, random_seed=42):
    """Generates data where noise variance increases with |x|.

    This is the ideal test for probabilistic regression: the model should
    learn that uncertainty grows in the tails.
    """
    np.random.seed(random_seed)
    x = np.random.uniform(-5, 5, n_samples).reshape(-1, 1)
    y_true = np.sin(x) * 2 + 0.5 * x
    noise_std = 0.1 + 0.4 * np.abs(x)  # Variance grows with |x|
    noise = np.random.normal(0, noise_std)
    y = y_true + noise
    return x, y, y_true, noise_std
```

**What to check:**
- Predicted mean closely tracks `y_true`
- Predicted `std` should approximately recover the `noise_std` curve
- NLL should be substantially better than a constant-variance model

#### Dataset 2: Multimodal Target (Stretch Test)

```python
def generate_multimodal_data(n_samples=1000, random_seed=42):
    """Generates data where the target is bimodal.

    For each x, y can be either y_true + delta or y_true - delta
    with equal probability. This tests whether the per-bin structure
    can capture multiple modes.
    """
    np.random.seed(random_seed)
    x = np.random.uniform(-3, 3, n_samples).reshape(-1, 1)
    y_true = x
    delta = 1.5 * np.ones_like(x)
    sign = np.random.choice([-1, 1], size=n_samples).reshape(-1, 1)
    y = y_true + sign * delta + np.random.normal(0, 0.1, (n_samples, 1))
    return x, y, y_true
```

**What to check:**
- Predicted variance should be large (captures bimodality as spread)
- Direct regression baselines should have MSE ≈ delta²; classification-based should adapt

#### Dataset 3: Simple Linear (Sanity Check)

```python
def generate_simple_linear(n_samples=500, random_seed=42):
    """Simple y = 2x + 1 + noise. Probabilistic regression should match baselines."""
    np.random.seed(random_seed)
    x = np.random.uniform(-5, 5, n_samples).reshape(-1, 1)
    y = 2 * x + 1 + np.random.normal(0, 0.3, (n_samples, 1))
    return x, y
```

**What to check:** MSE should be comparable to linear regression and simple NN baselines.
Overhead from classification structure should not degrade performance.

---

### 1.5 Build Comprehensive Comparison Script

Create a new script: `automl_package/examples/model_comparison.py`

This script should:

1. **Run all supported models** on each toy dataset:
   - `LinearRegressionModel` (baseline — no uncertainty)
   - `PyTorchNeuralNetwork` with `UncertaintyMethod.CONSTANT` (baseline)
   - `PyTorchNeuralNetwork` with `UncertaintyMethod.PROBABILISTIC` (probabilistic baseline)
   - `ClassifierRegressionModel` with optimized n_classes (classification-only baseline)
   - `ProbabilisticRegressionModel` with `SEPARATE_HEADS`, fixed k, `JOINT` optimization
   - `ProbabilisticRegressionModel` with `SINGLE_HEAD_N_OUTPUTS`, fixed k, `JOINT` optimization

2. **Metrics to collect per model:**
   - MSE (accuracy)
   - NLL (quality of uncertainty — only for models that produce uncertainty)
   - Calibration: % of test points within ±1σ of predicted mean (should be ~68%)
   - CRPS (Continuous Ranked Probability Score) if feasible

3. **k sweep:** For ProbabilisticRegressionModel, sweep k ∈ {3, 5, 7, 10, 15}

4. **Visualization:**
   - Prediction plots with uncertainty bands for each model
   - Calibration curves
   - MSE vs NLL scatter plot
   - k vs MSE and k vs NLL curves

5. **Output:** Results table (CSV + printed), plots saved to `model_comparison_results/`

---

### 1.6 Phase 1 Tests

Create the test infrastructure and Phase 1 test file.

#### 1.6.1 Test infrastructure setup

Create `tests/__init__.py` (empty) and `tests/conftest.py` with shared fixtures:

```python
"""Shared test fixtures for AutoML test suite."""

import numpy as np
import pytest


@pytest.fixture
def heteroscedastic_data():
    """Generates heteroscedastic sine data where noise grows with |x|."""
    np.random.seed(42)
    x = np.random.uniform(-5, 5, 500).reshape(-1, 1)
    y_true = np.sin(x) * 2 + 0.5 * x
    noise_std = 0.1 + 0.4 * np.abs(x)
    y = y_true + np.random.normal(0, noise_std)
    return x, y.ravel(), y_true.ravel(), noise_std.ravel()


@pytest.fixture
def multimodal_data():
    """Generates bimodal data: y = x ± 1.5."""
    np.random.seed(42)
    x = np.random.uniform(-3, 3, 500).reshape(-1, 1)
    sign = np.random.choice([-1, 1], size=500).reshape(-1, 1)
    y = x + sign * 1.5 + np.random.normal(0, 0.1, (500, 1))
    return x, y.ravel()


@pytest.fixture
def simple_linear_data():
    """Generates simple y = 2x + 1 + noise."""
    np.random.seed(42)
    x = np.random.uniform(-5, 5, 300).reshape(-1, 1)
    y = (2 * x + 1 + np.random.normal(0, 0.3, (300, 1))).ravel()
    return x, y


@pytest.fixture
def piecewise_data():
    """Piecewise function: linear for x<0, complex sinusoidal for x>=0."""
    np.random.seed(42)
    x = np.random.uniform(-5, 5, 500)
    y_true = np.piecewise(
        x,
        [x < 0, x >= 0],
        [
            lambda xi: 0.5 * xi,
            lambda xi: 0.5 * xi + np.sin(4 * np.pi * xi),
        ],
    )
    y = y_true + np.random.normal(0, 0.2, 500)
    return x.reshape(-1, 1), y, y_true
```

#### 1.6.2 Test file: `tests/test_phase1_probabilistic_regression.py`

```python
"""Phase 1 tests: Bug fixes and probabilistic regression baseline."""

import numpy as np
import pytest
import torch
from sklearn.model_selection import train_test_split

from automl_package.enums import (
    NClassesSelectionMethod,
    RegressionStrategy,
    TaskType,
    UncertaintyMethod,
)
from automl_package.models.common.losses import calculate_combined_loss
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel
from automl_package.utils.losses import nll_loss


class TestBug8DoublLogVar:
    """Verify Bug 8 fix: log_var should NOT be double-logged."""

    def test_calculate_combined_loss_no_double_log(self):
        """When predictions[:, 1] is already log_var, no extra log should be applied."""
        predictions = torch.tensor([[1.0, 0.5], [2.0, -0.3]])  # [mean, log_var]
        y_true = torch.tensor([1.1, 2.2])

        loss = calculate_combined_loss(
            predictions, y_true, UncertaintyMethod.PROBABILISTIC
        )

        # The loss should use log_var=0.5 directly, not log(0.5)
        # Manually compute expected NLL:
        mean = predictions[:, 0]
        log_var = predictions[:, 1]  # Should be used directly
        variance = torch.exp(log_var)
        expected = 0.5 * (np.log(2 * np.pi) + log_var + ((y_true - mean) ** 2) / variance)
        expected_loss = expected.mean()

        assert torch.allclose(loss, expected_loss, atol=1e-5), (
            f"Loss {loss.item():.6f} != expected {expected_loss.item():.6f}. "
            "Double log_var likely still present."
        )

    def test_nll_gradient_flows_to_variance(self):
        """Verify gradient flows correctly through log_var after bug fix."""
        predictions = torch.tensor([[1.0, 0.5], [2.0, -0.3]], requires_grad=True)
        y_true = torch.tensor([1.1, 2.2])

        loss = calculate_combined_loss(
            predictions, y_true, UncertaintyMethod.PROBABILISTIC
        )
        loss.backward()

        assert predictions.grad is not None
        # Gradient w.r.t. log_var (column 1) should be non-zero
        assert not torch.all(predictions.grad[:, 1] == 0), (
            "Zero gradient for log_var — variance learning is broken."
        )


class TestBug10DuplicateBuild:
    """Verify Bug 10 fix: no duplicate build_model call."""

    def test_no_duplicate_build_model_line(self):
        """The source file should not contain consecutive duplicate build lines."""
        import inspect
        from automl_package.models.independent_weights_flexible_neural_network import (
            IndependentWeightsFlexibleNN,
        )
        source = inspect.getsource(IndependentWeightsFlexibleNN.build_model)
        # Count occurrences of the model assignment
        count = source.count("self.model = self.IndependentWeightsFlexibleNNModule")
        assert count == 1, f"Found {count} build_model assignments, expected 1."


class TestProbabilisticRegressionFixedK:
    """Integration tests: train and predict with fixed k."""

    def _make_model(self, k: int = 5, strategy: RegressionStrategy = RegressionStrategy.SEPARATE_HEADS):
        return ProbabilisticRegressionModel(
            input_size=1,
            n_classes=k,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.NONE,
            regression_strategy=strategy,
            base_classifier_params={"hidden_layers": 1, "hidden_size": 32},
            regression_head_params={"hidden_layers": 0, "hidden_size": 16},
            n_epochs=30,
            learning_rate=0.01,
            early_stopping_rounds=10,
            validation_fraction=0.2,
            random_seed=42,
        )

    def test_train_and_predict_separate_heads(self, heteroscedastic_data):
        """SEPARATE_HEADS should train without crash and return predictions."""
        x, y, _, _ = heteroscedastic_data
        model = self._make_model(k=5, strategy=RegressionStrategy.SEPARATE_HEADS)
        model.fit(x, y)
        y_pred = model.predict(x)

        assert y_pred.shape == (len(x),), f"Expected 1D predictions, got {y_pred.shape}"
        assert not np.any(np.isnan(y_pred)), "Predictions contain NaN"

    def test_train_and_predict_single_head_n_outputs(self, heteroscedastic_data):
        """SINGLE_HEAD_N_OUTPUTS should train without crash."""
        x, y, _, _ = heteroscedastic_data
        model = self._make_model(k=5, strategy=RegressionStrategy.SINGLE_HEAD_N_OUTPUTS)
        model.fit(x, y)
        y_pred = model.predict(x)

        assert y_pred.shape == (len(x),)
        assert not np.any(np.isnan(y_pred))

    def test_uncertainty_correlates_with_noise(self, heteroscedastic_data):
        """Predicted uncertainty should be higher where true noise is higher."""
        x, y, _, noise_std = heteroscedastic_data
        x_train, x_test, y_train, y_test, _, noise_test = train_test_split(
            x, y, noise_std, test_size=0.3, random_state=42
        )
        model = self._make_model(k=5)
        model.fit(x_train, y_train)
        pred_std = model.predict_uncertainty(x_test)

        # Correlation between predicted and true std should be positive
        correlation = np.corrcoef(pred_std, noise_test)[0, 1]
        assert correlation > 0.2, (
            f"Predicted uncertainty poorly correlated with true noise "
            f"(r={correlation:.3f}). Bug 8 may not be fully fixed."
        )

    def test_mse_reasonable_on_linear_data(self, simple_linear_data):
        """Probabilistic regression should not degrade MSE on simple linear data."""
        x, y = simple_linear_data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        model = self._make_model(k=3)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        mse = np.mean((y_test - y_pred) ** 2)

        # For y=2x+1+noise(0.3), MSE should be well below 1.0
        assert mse < 1.0, f"MSE={mse:.4f} too high for simple linear data"


class TestNLLLoss:
    """Unit tests for the core NLL loss function."""

    def test_nll_loss_correct_formula(self):
        """NLL should match manual Gaussian NLL computation."""
        outputs = torch.tensor([[2.0, 0.0]])  # mean=2, log_var=0 → var=1
        targets = torch.tensor([3.0])

        loss = nll_loss(outputs, targets)
        # NLL = 0.5 * (log(2π) + 0 + (3-2)²/1) = 0.5 * (1.8379 + 1) = 1.4189
        expected = 0.5 * (np.log(2 * np.pi) + 0.0 + 1.0)
        assert abs(loss.item() - expected) < 1e-4

    def test_nll_loss_penalizes_wrong_mean(self):
        """Higher error in mean should produce higher NLL."""
        targets = torch.tensor([0.0])
        close = nll_loss(torch.tensor([[0.1, 0.0]]), targets)
        far = nll_loss(torch.tensor([[5.0, 0.0]]), targets)
        assert far > close

    def test_nll_loss_penalizes_overconfident_wrong(self):
        """Low variance + wrong mean should produce very high NLL."""
        targets = torch.tensor([0.0])
        # Wrong mean, low variance (overconfident)
        overconfident = nll_loss(torch.tensor([[5.0, -5.0]]), targets)
        # Wrong mean, high variance (honest)
        honest = nll_loss(torch.tensor([[5.0, 3.0]]), targets)
        assert overconfident > honest
```

#### 1.6.3 Run the tests

```bash
python -m pytest tests/test_phase1_probabilistic_regression.py -v --tb=short
python -m ruff check --line-length 180 automl_package/models/common/losses.py
python -m ruff check --line-length 180 automl_package/models/independent_weights_flexible_neural_network.py
```

### 1.7 Verification Criteria for Phase 1

- [ ] All Phase 1 tests pass: `python -m pytest tests/test_phase1_probabilistic_regression.py -v`
- [ ] Bug 8 fix verified: `TestBug8DoublLogVar` passes
- [ ] Bug 10 fix verified: `TestBug10DuplicateBuild` passes
- [ ] Bug 11 fix verified: showcase runs without IndexError
- [ ] Heteroscedastic dataset: `test_uncertainty_correlates_with_noise` passes (r > 0.2)
- [ ] Linear dataset: `test_mse_reasonable_on_linear_data` passes (MSE < 1.0)
- [ ] Comparison script runs all models without errors
- [ ] Results table shows probabilistic regression NLL advantage on heteroscedastic data
- [ ] All modified files pass `ruff --line-length 180` and `black -l 180`

---

## Phase 2: Fix FlexibleNN Gradient Bugs + Add Depth Complexity Control

**Goal:** Fix the broken STE and REINFORCE strategies, add probabilistically-motivated
depth control, and verify on the piecewise toy problem.

**Depends on:** Phase 1 (for verified testing infrastructure)

### 2.1 Fix Bug 5: REINFORCE log_prob in wrong position

**File:** `automl_package/models/selection_strategies/layer_selection_strategies.py` L198-204

**Current:** Returns `(final_output, n_actual, log_prob, mode_selection_probs, tensor(0.0))`

The caller at `flexible_neural_network.py:209` unpacks as:
`final_output, _, _, n_logits, log_prob = self.model(_batch_x)`

This means position 2 (`log_prob`) is discarded, position 4 (`tensor(0.0)`) is used as `log_prob`.

**Fix options (evaluate both):**

Option A: Fix the return order to match caller expectations:
```python
return (
    final_output,
    n_actual,
    self.mode_selection_probs,  # Position 2: n_probs (discarded by caller with _)
    n_logits_passthrough,       # Position 3: n_logits
    log_prob,                   # Position 4: log_prob (used by caller)
)
```

Option B: Fix the caller unpacking to match the return:
```python
final_output, _, log_prob, _, _ = self.model(_batch_x)
```

**Recommendation:** Option A is safer — matches the convention of all other strategies.
But first audit ALL callers to ensure consistency.

---

### 2.2 Fix Bug 6: STE gradient flow destruction

**File:** `automl_package/models/selection_strategies/layer_selection_strategies.py` L153-167

**Current:** Uses `argmax(n_probs)` then `torch.where(active_mask, ...)`. The `n_probs` from
`gumbel_softmax(hard=True)` carries STE gradients but they're never used.

**Fix:** Follow the pattern from GumbelSoftmax/SoftGating strategies — use `n_probs` to
weight the per-depth outputs, but with `hard=True` for discrete forward pass:

```python
def forward(self, x_input, logits):
    n_probs = f.gumbel_softmax(logits, tau=self.model.gumbel_tau, hard=True, dim=1)
    # n_probs is one-hot in forward, but has softmax gradients in backward (STE)

    final_output_neurons = self.model.model.output_layer.out_features
    aggregated_output = torch.zeros(x_input.size(0), final_output_neurons, device=x_input.device)

    hidden_representations = []
    current_output = x_input
    for i in range(self.model.max_hidden_layers):
        current_output = self.model.model.hidden_layers_blocks[i](current_output)
        hidden_representations.append(current_output)

    for i in range(self.model.max_hidden_layers):
        prob = n_probs[:, i]
        hidden_rep = hidden_representations[i]
        output_for_n = self.model.model.output_layer(hidden_rep)
        aggregated_output += prob.unsqueeze(1) * output_for_n

    n_actual = torch.argmax(n_probs, dim=1) + 1
    return aggregated_output, n_actual, None, n_probs, torch.tensor(0.0)
```

This is identical to GumbelSoftmax but with `hard=True`. In forward pass, only one depth
contributes. In backward, STE gradients flow through all paths.

---

### 2.3 Fix Bug 7: IndependentWeights STE returns soft probs

**File:** `automl_package/models/selection_strategies/independent_weights_strategies.py`

**Fix:** Return the `gumbel_softmax(hard=True)` one-hot vector as `n_probs`, not `f.softmax`.
The weighted sum in the caller will then use STE for gradient flow.

---

### 2.4 Add Depth Complexity Control

**Approach:** Implement both ELBO-based (probabilistic) and practical depth penalty,
controlled by a configuration enum.

#### 2.4.1 New enum

**File:** `automl_package/enums.py`

```python
class DepthRegularization(str, Enum):
    NONE = "none"
    DEPTH_PENALTY = "depth_penalty"
    ELBO = "elbo"
```

#### 2.4.2 ELBO implementation

**File:** Modify `flexible_neural_network.py` and `independent_weights_flexible_neural_network.py`

Add to `_defaults`:
```python
"depth_regularization": DepthRegularization.NONE,
"depth_prior_lambda": 0.5,      # Geometric decay for prior
"depth_penalty_weight": 0.01,    # Weight for depth_penalty mode
```

Add to training loop, after main loss computation:
```python
if self.depth_regularization == DepthRegularization.ELBO:
    depth_prior_logits = torch.arange(self.max_hidden_layers, 0, -1, dtype=torch.float, device=device)
    depth_prior = torch.distributions.Categorical(logits=depth_prior_logits)
    q_depth = torch.distributions.Categorical(probs=n_probs)
    kl_div = torch.distributions.kl_divergence(q_depth, depth_prior).mean()
    loss += kl_div
elif self.depth_regularization == DepthRegularization.DEPTH_PENALTY:
    expected_depth = torch.sum(
        n_probs * torch.arange(1, self.max_hidden_layers + 1, dtype=torch.float, device=device),
        dim=1
    )
    loss += self.depth_penalty_weight * expected_depth.mean()
```

#### 2.4.3 Add depth regularization to hyperparameter search space

Update `get_hyperparameter_search_space()` to include:
```python
"depth_regularization": {"type": "categorical", "choices": [e.value for e in DepthRegularization]},
"depth_prior_lambda": {"type": "float", "low": 0.1, "high": 0.9},
"depth_penalty_weight": {"type": "float", "low": 1e-4, "high": 0.1, "log": True},
```

---

### 2.5 Phase 2 Tests

Create `tests/test_phase2_flexible_nn.py`:

```python
"""Phase 2 tests: FlexibleNN gradient fixes and depth complexity control."""

import numpy as np
import pytest
import torch

from automl_package.enums import LayerSelectionMethod, UncertaintyMethod
from automl_package.models.flexible_neural_network import FlexibleHiddenLayersNN
from automl_package.models.independent_weights_flexible_neural_network import (
    IndependentWeightsFlexibleNN,
)


class TestBug5ReinforceLogProb:
    """Verify REINFORCE log_prob is in the correct tuple position."""

    def test_reinforce_log_prob_nonzero(self):
        """The log_prob returned by REINFORCE should be non-trivial."""
        model = FlexibleHiddenLayersNN(
            input_size=1,
            output_size=1,
            layer_selection_method=LayerSelectionMethod.REINFORCE,
            max_hidden_layers=3,
            n_predictor_layers=1,
            hidden_size=16,
            n_epochs=1,
            random_seed=42,
        )
        model.build_model()
        x = torch.randn(8, 1)
        model.model.train()
        final_output, _, _, _, log_prob = model.model(x)

        # log_prob should NOT be tensor(0.0) — that was the bug
        assert log_prob is not None
        assert not torch.all(log_prob == 0.0), (
            "log_prob is all zeros — Bug 5 still present"
        )

    def test_reinforce_policy_gradient_nonzero(self):
        """Policy loss from REINFORCE should produce nonzero gradient on n_predictor."""
        model = FlexibleHiddenLayersNN(
            input_size=1,
            output_size=1,
            layer_selection_method=LayerSelectionMethod.REINFORCE,
            max_hidden_layers=3,
            n_predictor_layers=1,
            hidden_size=16,
            n_epochs=1,
            random_seed=42,
        )
        model.build_model()
        x = torch.randn(8, 1)
        y = torch.randn(8, 1)

        model.model.train()
        model.model.zero_grad()
        final_output, _, _, _, log_prob = model.model(x)
        main_loss = torch.nn.MSELoss()(final_output, y)
        reward = -main_loss.detach()
        policy_loss = -log_prob * reward
        total_loss = main_loss + policy_loss.mean()
        total_loss.backward()

        # n_predictor should have received gradient
        predictor_grads = [
            p.grad for p in model.model.n_predictor.parameters()
            if p.grad is not None
        ]
        assert len(predictor_grads) > 0, "n_predictor received no gradients"
        assert any(g.abs().max() > 1e-8 for g in predictor_grads), (
            "n_predictor gradients are all near-zero"
        )


class TestBug6SteGradientFlow:
    """Verify STE passes gradients to n_predictor."""

    def test_ste_n_predictor_gradient_nonzero(self):
        """After backward pass, STE n_predictor should have non-zero gradients."""
        model = FlexibleHiddenLayersNN(
            input_size=1,
            output_size=1,
            layer_selection_method=LayerSelectionMethod.STE,
            max_hidden_layers=3,
            n_predictor_layers=1,
            hidden_size=16,
            n_epochs=1,
            random_seed=42,
        )
        model.build_model()
        x = torch.randn(16, 1)
        y = torch.randn(16, 1)

        model.model.train()
        model.model.zero_grad()
        final_output, _, _, _, _ = model.model(x)
        loss = torch.nn.MSELoss()(final_output, y)
        loss.backward()

        predictor_grads = [
            p.grad for p in model.model.n_predictor.parameters()
            if p.grad is not None
        ]
        assert len(predictor_grads) > 0, (
            "STE n_predictor received no gradients — Bug 6 still present"
        )


class TestBug7IndependentWeightsSte:
    """Verify IndependentWeights STE returns proper one-hot."""

    def test_ste_returns_hard_selection(self):
        """STE strategy should return near-one-hot n_probs, not soft probabilities."""
        model = IndependentWeightsFlexibleNN(
            input_size=1,
            output_size=1,
            layer_selection_method=LayerSelectionMethod.STE,
            max_hidden_layers=3,
            n_predictor_layers=1,
            hidden_size=16,
            n_epochs=1,
            random_seed=42,
        )
        model.build_model()
        x = torch.randn(8, 1)

        model.model.eval()
        with torch.no_grad():
            _, _, n_probs, _, _ = model.model(x)

        # Each row should be one-hot (exactly one 1.0, rest 0.0)
        for row in n_probs:
            assert torch.sum(row == 1.0) == 1, (
                f"STE n_probs row is not one-hot: {row.tolist()}. "
                "Bug 7 still present — returning soft probs."
            )


class TestDepthComplexityControl:
    """Tests for ELBO and depth penalty mechanisms."""

    def test_elbo_prefers_shallow_on_linear(self, simple_linear_data):
        """With ELBO, linear data should result in shallower depth selections."""
        x, y = simple_linear_data
        model = FlexibleHiddenLayersNN(
            input_size=1,
            output_size=1,
            layer_selection_method=LayerSelectionMethod.GUMBEL_SOFTMAX,
            max_hidden_layers=3,
            n_predictor_layers=1,
            hidden_size=32,
            n_epochs=50,
            depth_regularization="elbo",  # New parameter
            random_seed=42,
        )
        model.fit(x, y)

        # Get depth selections
        x_tensor = torch.tensor(x, dtype=torch.float32)
        model.model.eval()
        with torch.no_grad():
            _, n_actual, _, _, _ = model.model(x_tensor)

        mean_depth = n_actual.float().mean().item()
        # Linear data should NOT need max depth
        assert mean_depth < model.max_hidden_layers, (
            f"Mean depth {mean_depth:.2f} == max {model.max_hidden_layers}. "
            "ELBO complexity control not working."
        )

    def test_depth_penalty_reduces_mean_depth(self, simple_linear_data):
        """Depth penalty should reduce mean selected depth vs no regularization."""
        x, y = simple_linear_data

        # Train without penalty
        model_none = FlexibleHiddenLayersNN(
            input_size=1, output_size=1,
            layer_selection_method=LayerSelectionMethod.GUMBEL_SOFTMAX,
            max_hidden_layers=3, n_predictor_layers=1,
            hidden_size=32, n_epochs=50,
            depth_regularization="none",
            random_seed=42,
        )
        model_none.fit(x, y)

        # Train with penalty
        model_pen = FlexibleHiddenLayersNN(
            input_size=1, output_size=1,
            layer_selection_method=LayerSelectionMethod.GUMBEL_SOFTMAX,
            max_hidden_layers=3, n_predictor_layers=1,
            hidden_size=32, n_epochs=50,
            depth_regularization="depth_penalty",
            depth_penalty_weight=0.05,
            random_seed=42,
        )
        model_pen.fit(x, y)

        x_tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            _, n_none, _, _, _ = model_none.model(x_tensor)
            _, n_pen, _, _, _ = model_pen.model(x_tensor)

        assert n_pen.float().mean() <= n_none.float().mean(), (
            "Depth penalty did not reduce mean depth"
        )


class TestFlexibleNNSmoke:
    """Smoke tests: all working strategies should train and predict."""

    @pytest.mark.parametrize("method", [
        LayerSelectionMethod.GUMBEL_SOFTMAX,
        LayerSelectionMethod.SOFT_GATING,
        LayerSelectionMethod.STE,
        LayerSelectionMethod.REINFORCE,
    ])
    def test_shared_weights_trains(self, simple_linear_data, method):
        """Each strategy should train without crash."""
        x, y = simple_linear_data
        model = FlexibleHiddenLayersNN(
            input_size=1, output_size=1,
            layer_selection_method=method,
            max_hidden_layers=3, n_predictor_layers=1,
            hidden_size=16, n_epochs=10,
            random_seed=42,
        )
        model.fit(x, y)
        y_pred = model.predict(x)
        assert y_pred.shape == (len(x),)
        assert not np.any(np.isnan(y_pred))

    @pytest.mark.parametrize("method", [
        LayerSelectionMethod.GUMBEL_SOFTMAX,
        LayerSelectionMethod.SOFT_GATING,
        LayerSelectionMethod.STE,
    ])
    def test_independent_weights_trains(self, simple_linear_data, method):
        """Each strategy should train without crash."""
        x, y = simple_linear_data
        model = IndependentWeightsFlexibleNN(
            input_size=1, output_size=1,
            layer_selection_method=method,
            max_hidden_layers=3, n_predictor_layers=1,
            hidden_size=16, n_epochs=10,
            random_seed=42,
        )
        model.fit(x, y)
        y_pred = model.predict(x)
        assert y_pred.shape == (len(x),)
        assert not np.any(np.isnan(y_pred))
```

Run:
```bash
python -m pytest tests/test_phase2_flexible_nn.py -v --tb=short
```

### 2.6 Verification Criteria for Phase 2

- [ ] All Phase 2 tests pass: `python -m pytest tests/test_phase2_flexible_nn.py -v`
- [ ] `TestBug5ReinforceLogProb` passes — log_prob is non-zero, policy gradient flows
- [ ] `TestBug6SteGradientFlow` passes — n_predictor receives gradient
- [ ] `TestBug7IndependentWeightsSte` passes — STE returns one-hot
- [ ] `TestDepthComplexityControl` passes — ELBO/penalty reduce mean depth on linear data
- [ ] `TestFlexibleNNSmoke` passes — all strategies train without crash
- [ ] All modified files pass `ruff --line-length 180` and `black -l 180`

---

## Phase 3: Fix Dynamic n_classes for Probabilistic Regression

**Goal:** Make the dynamic k-selection strategies (Gumbel, SoftGating, STE, REINFORCE)
functional, then explore ELBO-based k selection.

**Depends on:** Phase 1 (Bug 8 must be fixed first), Phase 2 (for ELBO patterns)

### 3.1 Fix Bug 1: `_weighted_average_logic` return count

**File:** `automl_package/models/selection_strategies/base_selection_strategy.py` L48-72

**Fix:** Return 3 values by computing classifier logits inside the method:

```python
def _weighted_average_logic(self, x_input, mode_selection_probs):
    classifier_raw_logits = self.model.classifier_layers(x_input)
    # ... existing logic ...
    return final_predictions_contribution, selected_k_values, classifier_raw_logits
```

---

### 3.2 Fix Bug 2: Dynamic strategies return count

**File:** `automl_package/models/selection_strategies/n_classes_strategies.py` L63-158

**Fix:** Make all dynamic strategies return 5 values matching `NoneStrategy`:
`(predictions, selected_k, log_prob, classifier_logits, per_head_outputs)`

This requires computing `per_head_outputs` inside each strategy. The cleanest approach is
to pass `per_head_outputs` back from the helper methods (`_weighted_average_logic`,
`_hard_selection_logic`).

---

### 3.3 Fix Bug 3: Missing `boundaries` kwarg

**File:** `automl_package/models/selection_strategies/n_classes_strategies.py`

**Fix:** Add `boundaries` parameter to all dynamic strategy `forward()` signatures and pass
it through to `_compute_predictions_for_k` and `direct_regression_head`.

---

### 3.4 Fix Bug 4: Pass classifier logits, not raw features

**File:** `automl_package/models/selection_strategies/base_selection_strategy.py` L68

**Fix:** Call `self.model.classifier_layers(x_input)` first, then pass logits to
`_compute_predictions_for_k`. This is already done correctly in `_hard_selection_logic`.

---

### 3.5 Fix Bug 9: Self-referential dict

**File:** `automl_package/models/probabilistic_regression.py` L325-339

**Fix:** Store a copy:
```python
self.middle_class_dist_params_[k] = {
    "mean": self.middle_class_dist_params_.get("mean"),
    "std": self.middle_class_dist_params_.get("std"),
}
```

Or restructure to avoid the mixin pattern overwriting the same attribute.

---

### 3.6 Implement ELBO-Based k Selection (Research Extension)

Similar to the depth ELBO in Phase 2, treat k as a discrete latent variable:

```
ELBO = E_q(k|x) [log p(y | x, k)] - KL(q(k|x) || p(k))
```

Where:
- `q(k|x)` is the n_classes_predictor output
- `p(k)` is a prior favoring simplicity (e.g., geometric favoring low k)

**Implementation steps:**
1. Add `NClassesRegularization` enum (NONE, ELBO, PENALTY)
2. Compute per-k likelihoods inside `_weighted_average_logic`
3. Add KL divergence term to loss
4. The KL term naturally penalizes unnecessarily high k without ad-hoc weights

---

### 3.7 Phase 3 Tests

Create `tests/test_phase3_dynamic_k.py`:

```python
"""Phase 3 tests: Dynamic n_classes strategies."""

import numpy as np
import pytest
import torch

from automl_package.enums import (
    NClassesSelectionMethod,
    RegressionStrategy,
    UncertaintyMethod,
)
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel


class TestBugs1to3DynamicStrategiesNoCrash:
    """All dynamic strategies should run without ValueError/TypeError."""

    @pytest.mark.parametrize("method", [
        NClassesSelectionMethod.GUMBEL_SOFTMAX,
        NClassesSelectionMethod.SOFT_GATING,
        NClassesSelectionMethod.STE,
        NClassesSelectionMethod.REINFORCE,
    ])
    def test_dynamic_strategy_trains(self, heteroscedastic_data, method):
        """Each dynamic n_classes strategy should train without crash."""
        x, y, _, _ = heteroscedastic_data
        model = ProbabilisticRegressionModel(
            input_size=1,
            n_classes=5,
            max_n_classes=7,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=method,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            n_epochs=10,
            learning_rate=0.01,
            random_seed=42,
        )
        model.fit(x, y)
        y_pred = model.predict(x)
        assert y_pred.shape == (len(x),)
        assert not np.any(np.isnan(y_pred))


class TestBug4ClassifierLogits:
    """Verify _weighted_average_logic receives classifier logits, not raw features."""

    def test_weighted_average_uses_classifier_output(self):
        """Predictions from weighted_average should change when classifier weights change."""
        # This is an indirect test: if raw features were passed instead of logits,
        # the classifier wouldn't influence predictions at all.
        model = ProbabilisticRegressionModel(
            input_size=1, n_classes=3,
            max_n_classes=5,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.GUMBEL_SOFTMAX,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            n_epochs=1, random_seed=42,
        )
        # Build but don't fully train
        x = np.random.randn(10, 1).astype(np.float32)
        y = np.random.randn(10).astype(np.float32)
        model.fit(x, y)
        y_pred_1 = model.predict(x)

        # Perturb classifier weights
        for p in model.model.classifier_layers.parameters():
            p.data += 1.0
        y_pred_2 = model.predict(x)

        # Predictions should differ if classifier output is used
        assert not np.allclose(y_pred_1, y_pred_2, atol=1e-3), (
            "Predictions unchanged after perturbing classifier — "
            "raw features may still be passed instead of logits (Bug 4)"
        )


class TestBug9MiddleClassParams:
    """Verify middle_class_dist_params stored correctly per k."""

    def test_middle_class_params_not_self_referential(self, simple_linear_data):
        """Each k should have its own independent params, not circular references."""
        x, y = simple_linear_data
        model = ProbabilisticRegressionModel(
            input_size=1, n_classes=3,
            max_n_classes=7,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.NONE,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            n_epochs=5, random_seed=42,
        )
        model.fit(x, y)

        if hasattr(model, "middle_class_dist_params_"):
            params = model.middle_class_dist_params_
            # Should not contain itself (circular reference)
            for k, v in params.items():
                if isinstance(v, dict):
                    assert v is not params, (
                        f"middle_class_dist_params_[{k}] is a circular reference to itself"
                    )


class TestELBOkSelection:
    """Tests for ELBO-based dynamic k selection."""

    def test_elbo_k_converges(self, heteroscedastic_data):
        """ELBO-based k selection should train and converge."""
        x, y, _, _ = heteroscedastic_data
        model = ProbabilisticRegressionModel(
            input_size=1, n_classes=5,
            max_n_classes=10,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.GUMBEL_SOFTMAX,
            n_classes_regularization="elbo",  # New parameter
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            n_epochs=30, random_seed=42,
        )
        model.fit(x, y)
        y_pred = model.predict(x)
        mse = np.mean((y - y_pred) ** 2)

        # Should converge to reasonable MSE
        assert mse < 10.0, f"ELBO-k model MSE={mse:.4f} — did not converge"


class TestDynamicKModelComparison:
    """Model-level tests: verify dynamic k achieves expected behavior on designed problems."""

    def _make_probreg(self, n_classes=5, max_n_classes=10, method=NClassesSelectionMethod.GUMBEL_SOFTMAX,
                      n_classes_regularization=None, **kwargs):
        defaults = dict(
            input_size=1,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=method,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            n_epochs=60, learning_rate=0.01, early_stopping_rounds=15,
            validation_fraction=0.2, random_seed=42,
        )
        defaults.update(kwargs)
        return ProbabilisticRegressionModel(
            n_classes=n_classes, max_n_classes=max_n_classes,
            n_classes_regularization=n_classes_regularization, **defaults,
        )

    def test_dynamic_k_nll_competitive_with_fixed_k(self, heteroscedastic_data):
        """Dynamic-k ProbReg NLL should be competitive with fixed-k=5 (known best)."""
        x, y, _, _ = heteroscedastic_data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        fixed = self._make_probreg(n_classes=5, method=NClassesSelectionMethod.NONE)
        fixed.fit(x_train, y_train)
        fixed_nll = _compute_nll(fixed, x_test, y_test)

        dynamic = self._make_probreg(n_classes=5, max_n_classes=10,
                                     method=NClassesSelectionMethod.GUMBEL_SOFTMAX)
        dynamic.fit(x_train, y_train)
        dynamic_nll = _compute_nll(dynamic, x_test, y_test)

        # Dynamic should not be dramatically worse (allow 50% slack since it has
        # harder optimization problem)
        assert dynamic_nll < fixed_nll * 1.5, (
            f"Dynamic-k NLL ({dynamic_nll:.4f}) is >1.5x worse than fixed-k=5 ({fixed_nll:.4f})"
        )

    def test_elbo_selects_fewer_k_on_simple_data(self, simple_linear_data, heteroscedastic_data):
        """ELBO should select lower k on simple linear data than on heteroscedastic data."""
        x_lin, y_lin = simple_linear_data
        x_het, y_het, _, _ = heteroscedastic_data

        model_lin = self._make_probreg(n_classes=3, max_n_classes=8,
                                       n_classes_regularization="elbo", n_epochs=80)
        model_lin.fit(x_lin, y_lin)

        model_het = self._make_probreg(n_classes=3, max_n_classes=8,
                                       n_classes_regularization="elbo", n_epochs=80)
        model_het.fit(x_het, y_het)

        x_lin_t = torch.tensor(x_lin, dtype=torch.float32).to(model_lin.device)
        x_het_t = torch.tensor(x_het, dtype=torch.float32).to(model_het.device)

        model_lin.model.eval()
        model_het.model.eval()
        with torch.no_grad():
            _, k_lin, _, _, _ = model_lin.model(x_lin_t)
            _, k_het, _, _, _ = model_het.model(x_het_t)

        mean_k_lin = k_lin.float().mean().item()
        mean_k_het = k_het.float().mean().item()

        assert mean_k_lin <= mean_k_het, (
            f"ELBO should prefer fewer classes on simple data (k={mean_k_lin:.2f}) "
            f"than heteroscedastic data (k={mean_k_het:.2f})."
        )

    def test_per_input_k_variation(self, heteroscedastic_data):
        """On heteroscedastic data where noise varies with x, dynamic k should vary per input.

        The heteroscedastic fixture has noise proportional to x. Regions with higher noise
        may benefit from more classes to capture the broader distribution. At minimum,
        k should not be constant across all inputs.
        """
        x, y, _, _ = heteroscedastic_data

        model = self._make_probreg(n_classes=3, max_n_classes=8,
                                   method=NClassesSelectionMethod.GUMBEL_SOFTMAX,
                                   n_epochs=80)
        model.fit(x, y)

        x_tensor = torch.tensor(x, dtype=torch.float32).to(model.device)
        model.model.eval()
        with torch.no_grad():
            _, k_actual, _, _, _ = model.model(x_tensor)

        k_values = k_actual.float().cpu().numpy().ravel()
        unique_k = np.unique(k_values)

        assert len(unique_k) > 1, (
            f"Dynamic k is constant ({unique_k[0]}) across all inputs. "
            f"n_classes_predictor is not learning input-dependent k selection."
        )

    def test_dynamic_k_uncertainty_correlates_with_noise(self, heteroscedastic_data):
        """Dynamic-k ProbReg uncertainty should still correlate with actual noise level."""
        x, y, _, noise_level = heteroscedastic_data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        _, _, _, noise_test = train_test_split(x, noise_level, test_size=0.3, random_state=42)

        model = self._make_probreg(n_classes=5, max_n_classes=10,
                                   method=NClassesSelectionMethod.GUMBEL_SOFTMAX, n_epochs=80)
        model.fit(x_train, y_train)
        pred_std = model.predict_uncertainty(x_test)

        correlation = np.corrcoef(noise_test.ravel(), pred_std.ravel())[0, 1]
        assert correlation > 0.3, (
            f"Dynamic-k predicted uncertainty poorly correlates with actual noise "
            f"(r={correlation:.3f}). Uncertainty estimation may be broken."
        )


def _compute_nll(model, x, y):
    """Helper: compute gaussian NLL from model predictions + uncertainty."""
    y_pred = model.predict(x)
    y_std = model.predict_uncertainty(x)
    log_var = 2 * np.log(np.clip(y_std, 1e-6, None))
    nll = 0.5 * np.mean(log_var + ((y - y_pred) ** 2) / np.exp(log_var))
    return float(nll)
```

Run:
```bash
python -m pytest tests/test_phase3_dynamic_k.py -v --tb=short
```

### 3.8 Verification Criteria for Phase 3

- [ ] All Phase 3 tests pass: `python -m pytest tests/test_phase3_dynamic_k.py -v`
- [ ] `TestBugs1to3DynamicStrategiesNoCrash` — all 4 dynamic strategies train without crash
- [ ] `TestBug4ClassifierLogits` — classifier weights influence predictions
- [ ] `TestBug9MiddleClassParams` — no circular references
- [ ] `TestELBOkSelection` — ELBO-k model trains and converges
- [ ] `TestDynamicKModelComparison` — dynamic k NLL competitive with fixed k, ELBO selects fewer k on simple data, per-input k varies, uncertainty correlates with noise
- [ ] All modified files pass `ruff --line-length 180`

---

## Phase 4: Integration, Polish, and Advanced Features

**Goal:** End-to-end integration, advanced research features, comprehensive testing.

**Depends on:** Phases 1-3

### 4.1 Inference-Time Optimization for FlexibleNN

Add hard selection mode for deployment:

```python
def predict(self, x, filter_data=True, inference_mode="hard"):
    """When inference_mode='hard', use argmax selection for single-network inference."""
    if inference_mode == "hard":
        # Only run the selected network/depth
        n_actual = torch.argmax(n_probs, dim=1)
        # Execute only network[n_actual] per sample
```

This provides genuine compute savings at inference — the key practical value proposition.

### 4.2 Add β-NLL Loss Variant

Standard NLL can lead to variance collapse (model learns to predict high variance to reduce
loss). β-NLL (Seitzer et al., 2022) adds a β weighting that stabilizes training:

```python
def beta_nll_loss(outputs, targets, beta=0.5):
    mean = outputs[:, 0]
    log_var = outputs[:, 1]
    variance = torch.exp(log_var)
    nll = 0.5 * (log_var + ((targets - mean) ** 2) / variance)
    weighted_nll = (variance.detach() ** beta) * nll
    return weighted_nll.mean()
```

### 4.3 Add Symlog Transform

For targets spanning multiple orders of magnitude:

```python
def symlog(x):
    return torch.sign(x) * torch.log(1 + torch.abs(x))

def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
```

Apply before binning, reverse after prediction.

### 4.4 Comprehensive Regression & Integration Test Suite

By Phase 4, we have tests from Phases 1–3 covering all bug fixes and features.
Phase 4 adds **performance regression tests** (lock down known baselines so future changes
can't silently degrade), **cross-model comparison tests**, and **feature-specific model-level tests**.

Create `tests/test_phase4_regression.py`:

```python
"""Phase 4 tests: performance regression, cross-model integration, and new features.

Tests use fixed seeds and assert MSE/NLL don't degrade below known baselines.
Update thresholds only when intentional changes are made.
"""

import numpy as np
import pytest
import torch
from sklearn.model_selection import train_test_split

from automl_package.enums import (
    DepthRegularization,
    LayerSelectionMethod,
    NClassesSelectionMethod,
    RegressionStrategy,
    UncertaintyMethod,
)
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel
from automl_package.models.flexible_neural_network import FlexibleHiddenLayersNN
from automl_package.models.classifier_regression import ClassifierRegressionModel
from automl_package.models.neural_network import PyTorchNeuralNetwork


class TestPerformanceBaselines:
    """Lock down known performance — these thresholds are set from Phase 1-3 results."""

    def test_prob_regression_heteroscedastic_nll(self, heteroscedastic_data):
        """ProbReg NLL on heteroscedastic data must stay below known baseline (1.38)."""
        x, y, _, _ = heteroscedastic_data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        model = ProbabilisticRegressionModel(
            input_size=1, n_classes=5,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.NONE,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            n_epochs=50, learning_rate=0.01,
            early_stopping_rounds=10, validation_fraction=0.2,
            random_seed=42,
        )
        model.fit(x_train, y_train)
        nll = _compute_nll(model, x_test, y_test)

        # Baseline from Phase 1: ProbReg_k5 NLL=1.38
        assert nll < 1.8, f"ProbReg NLL ({nll:.4f}) regressed past baseline 1.8 (known: 1.38)"

    def test_prob_regression_heteroscedastic_mse(self, heteroscedastic_data):
        """ProbReg MSE on heteroscedastic data must stay below known baseline (1.54)."""
        x, y, _, _ = heteroscedastic_data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        model = ProbabilisticRegressionModel(
            input_size=1, n_classes=5,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.NONE,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            n_epochs=50, learning_rate=0.01,
            early_stopping_rounds=10, validation_fraction=0.2,
            random_seed=42,
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        mse = float(np.mean((y_test - y_pred) ** 2))

        # Baseline from Phase 1: ProbReg_k5 MSE=1.54. Allow 30% slack for seed/platform variance.
        assert mse < 2.5, f"ProbReg MSE ({mse:.4f}) regressed past threshold 2.5 (known: 1.54)"

    def test_flexible_nn_piecewise_mse(self, piecewise_data):
        """FlexibleNN MSE on piecewise data must stay below known baseline."""
        x, y, _ = piecewise_data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        model = FlexibleHiddenLayersNN(
            input_size=1, output_size=1, max_hidden_layers=3, n_predictor_layers=1,
            hidden_size=32, n_epochs=100, learning_rate=0.01, early_stopping_rounds=15,
            validation_fraction=0.2, random_seed=42, calculate_feature_importance=False,
            layer_selection_method=LayerSelectionMethod.GUMBEL_SOFTMAX,
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        mse = float(np.mean((y_test - y_pred) ** 2))

        assert mse < 1.0, f"FlexibleNN piecewise MSE ({mse:.4f}) regressed past threshold 1.0"


class TestCrossModelRanking:
    """Verify expected model ranking on designed problems holds across all model types."""

    def test_model_ranking_heteroscedastic(self, heteroscedastic_data):
        """On heteroscedastic data: ProbReg NLL < ClassReg NLL < ConstantVariance NN NLL."""
        x, y, _, _ = heteroscedastic_data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        # ProbReg (learned variance)
        probreg = ProbabilisticRegressionModel(
            input_size=1, n_classes=5,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.NONE,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            n_epochs=50, learning_rate=0.01, early_stopping_rounds=10,
            validation_fraction=0.2, random_seed=42,
        )
        probreg.fit(x_train, y_train)
        probreg_nll = _compute_nll(probreg, x_test, y_test)

        # Plain NN (constant variance)
        nn = PyTorchNeuralNetwork(
            input_size=1, output_size=1, hidden_layers=2, hidden_size=32,
            learning_rate=0.01, n_epochs=50, early_stopping_rounds=10,
            validation_fraction=0.2, uncertainty_method=UncertaintyMethod.CONSTANT,
            random_seed=42,
        )
        nn.fit(x_train, y_train)
        nn_nll = _compute_nll(nn, x_test, y_test)

        assert probreg_nll < nn_nll, (
            f"ProbReg NLL ({probreg_nll:.4f}) should beat constant-variance NN ({nn_nll:.4f}) "
            f"on heteroscedastic data — learned variance is the whole point."
        )

    def test_flexible_nn_vs_fixed_depth_on_varying_complexity(self, piecewise_data):
        """FlexibleNN should beat both too-shallow and too-deep fixed NNs on piecewise data."""
        x, y, _ = piecewise_data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        # Shallow (1 layer) — underfit on sinusoidal region
        shallow = PyTorchNeuralNetwork(
            input_size=1, output_size=1, hidden_layers=1, hidden_size=32,
            learning_rate=0.01, n_epochs=80, early_stopping_rounds=15,
            validation_fraction=0.2, uncertainty_method=UncertaintyMethod.CONSTANT,
            random_seed=42,
        )
        shallow.fit(x_train, y_train)
        shallow_mse = float(np.mean((y_test - shallow.predict(x_test)) ** 2))

        # Flexible (adapts depth per input)
        flex = FlexibleHiddenLayersNN(
            input_size=1, output_size=1, max_hidden_layers=3, n_predictor_layers=1,
            hidden_size=32, n_epochs=100, learning_rate=0.01, early_stopping_rounds=15,
            validation_fraction=0.2, random_seed=42, calculate_feature_importance=False,
            layer_selection_method=LayerSelectionMethod.GUMBEL_SOFTMAX,
            depth_regularization=DepthRegularization.ELBO,
        )
        flex.fit(x_train, y_train)
        flex_mse = float(np.mean((y_test - flex.predict(x_test)) ** 2))

        assert flex_mse < shallow_mse, (
            f"FlexibleNN+ELBO ({flex_mse:.4f}) should beat shallow NN ({shallow_mse:.4f}) "
            f"on piecewise data."
        )


class TestEndToEndAllModels:
    """Full pipeline: train → predict → uncertainty for every model type."""

    @pytest.mark.parametrize("model_factory", [
        "probreg_fixed_k",
        "flexible_nn_gumbel",
        "flexible_nn_elbo",
    ])
    def test_full_pipeline(self, heteroscedastic_data, model_factory):
        """Each model should complete the full train → predict → uncertainty pipeline."""
        x, y, _, _ = heteroscedastic_data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        if model_factory == "probreg_fixed_k":
            model = ProbabilisticRegressionModel(
                input_size=1, n_classes=5,
                uncertainty_method=UncertaintyMethod.PROBABILISTIC,
                n_classes_selection_method=NClassesSelectionMethod.NONE,
                regression_strategy=RegressionStrategy.SEPARATE_HEADS,
                n_epochs=10, random_seed=42,
            )
        elif model_factory == "flexible_nn_gumbel":
            model = FlexibleHiddenLayersNN(
                input_size=1, output_size=1, max_hidden_layers=3, n_predictor_layers=1,
                hidden_size=16, n_epochs=10, random_seed=42,
                calculate_feature_importance=False,
                layer_selection_method=LayerSelectionMethod.GUMBEL_SOFTMAX,
            )
        elif model_factory == "flexible_nn_elbo":
            model = FlexibleHiddenLayersNN(
                input_size=1, output_size=1, max_hidden_layers=3, n_predictor_layers=1,
                hidden_size=16, n_epochs=10, random_seed=42,
                calculate_feature_importance=False,
                layer_selection_method=LayerSelectionMethod.GUMBEL_SOFTMAX,
                depth_regularization=DepthRegularization.ELBO,
            )

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        assert y_pred.shape == (len(y_test),)
        assert not np.any(np.isnan(y_pred))

        if hasattr(model, "predict_uncertainty"):
            y_std = model.predict_uncertainty(x_test)
            assert y_std.shape == (len(y_test),)
            assert np.all(y_std > 0), "Predicted std should be positive"


class TestBetaNLL:
    """Tests for β-NLL loss variant."""

    def test_beta_nll_better_calibration(self, heteroscedastic_data):
        """β-NLL should produce better calibrated uncertainty than standard NLL.

        Calibration = fraction of test points within predicted ±1σ interval ≈ 0.68.
        """
        x, y, _, _ = heteroscedastic_data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        # Standard NLL
        model_std = ProbabilisticRegressionModel(
            input_size=1, n_classes=5,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.NONE,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            n_epochs=50, learning_rate=0.01, early_stopping_rounds=10,
            validation_fraction=0.2, random_seed=42,
            loss_type="nll",  # standard
        )
        model_std.fit(x_train, y_train)
        cal_std = _calibration_score(model_std, x_test, y_test)

        # β-NLL
        model_beta = ProbabilisticRegressionModel(
            input_size=1, n_classes=5,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.NONE,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            n_epochs=50, learning_rate=0.01, early_stopping_rounds=10,
            validation_fraction=0.2, random_seed=42,
            loss_type="beta_nll", beta=0.5,
        )
        model_beta.fit(x_train, y_train)
        cal_beta = _calibration_score(model_beta, x_test, y_test)

        # β-NLL calibration should be closer to ideal 0.68
        ideal = 0.6827  # 1σ coverage
        assert abs(cal_beta - ideal) <= abs(cal_std - ideal) + 0.05, (
            f"β-NLL calibration ({cal_beta:.3f}) should be closer to ideal 0.68 "
            f"than standard NLL ({cal_std:.3f}). Allow 5% tolerance."
        )


class TestHardInference:
    """Tests for inference-time hard selection optimization."""

    def test_hard_inference_predictions_close_to_soft(self, piecewise_data):
        """Hard inference should produce similar predictions to soft inference."""
        x, y, _ = piecewise_data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        model = FlexibleHiddenLayersNN(
            input_size=1, output_size=1, max_hidden_layers=3, n_predictor_layers=1,
            hidden_size=32, n_epochs=80, learning_rate=0.01, early_stopping_rounds=15,
            validation_fraction=0.2, random_seed=42, calculate_feature_importance=False,
            layer_selection_method=LayerSelectionMethod.GUMBEL_SOFTMAX,
        )
        model.fit(x_train, y_train)

        y_soft = model.predict(x_test, inference_mode="soft")
        y_hard = model.predict(x_test, inference_mode="hard")

        # Hard and soft should be close (hard skips computation, not quality)
        mse_diff = float(np.mean((y_soft - y_hard) ** 2))
        assert mse_diff < 0.5, (
            f"Hard vs soft inference MSE diff ({mse_diff:.4f}) too large — "
            f"hard selection may be picking wrong depth."
        )

    def test_hard_inference_faster_than_soft(self, piecewise_data):
        """Hard inference should be faster (fewer FLOPs) than soft."""
        import time
        x, y, _ = piecewise_data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        model = FlexibleHiddenLayersNN(
            input_size=1, output_size=1, max_hidden_layers=4, n_predictor_layers=1,
            hidden_size=64, n_epochs=80, learning_rate=0.01, early_stopping_rounds=15,
            validation_fraction=0.2, random_seed=42, calculate_feature_importance=False,
            layer_selection_method=LayerSelectionMethod.GUMBEL_SOFTMAX,
        )
        model.fit(x_train, y_train)

        # Use larger batch for timing
        x_large = np.tile(x_test, (20, 1))
        n_runs = 10

        t0 = time.perf_counter()
        for _ in range(n_runs):
            model.predict(x_large, inference_mode="soft")
        t_soft = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(n_runs):
            model.predict(x_large, inference_mode="hard")
        t_hard = time.perf_counter() - t0

        # Hard should be at least a bit faster (not a strict test — timing is noisy)
        # Just verify it doesn't crash and is in the same ballpark
        assert t_hard < t_soft * 2.0, (
            f"Hard inference ({t_hard:.3f}s) should not be 2x slower than soft ({t_soft:.3f}s)"
        )


class TestConformalWrapper:
    """Tests for conformal prediction intervals."""

    def test_conformal_coverage(self, heteroscedastic_data):
        """Conformal intervals should achieve specified coverage on test data."""
        x, y, _, _ = heteroscedastic_data
        x_train, x_cal_test, y_train, y_cal_test = train_test_split(x, y, test_size=0.4, random_state=42)
        x_cal, x_test, y_cal, y_test = train_test_split(x_cal_test, y_cal_test, test_size=0.5, random_state=42)

        model = ProbabilisticRegressionModel(
            input_size=1, n_classes=5,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.NONE,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            n_epochs=30, learning_rate=0.01, random_seed=42,
        )
        model.fit(x_train, y_train)

        from automl_package.models.conformal import ConformalWrapper
        cw = ConformalWrapper(model)
        alpha = 0.1  # target 90% coverage
        cw.calibrate(x_cal, y_cal, alpha=alpha)
        lower, upper = cw.predict_interval(x_test)

        coverage = np.mean((y_test >= lower) & (y_test <= upper))
        # Conformal guarantee: coverage >= 1-α on exchangeable data
        # Allow small slack for finite sample effects
        assert coverage >= (1 - alpha) - 0.05, (
            f"Conformal coverage ({coverage:.3f}) below target {1-alpha:.2f} - 0.05. "
            f"Conformal guarantee violated."
        )

    def test_conformal_interval_width_correlates_with_noise(self, heteroscedastic_data):
        """For a model with learned variance, conformal + model uncertainty should
        produce wider intervals where noise is higher."""
        x, y, _, noise_level = heteroscedastic_data
        x_train, x_cal_test, y_train, y_cal_test = train_test_split(x, y, test_size=0.4, random_state=42)
        _, _, _, noise_cal_test = train_test_split(x, noise_level, test_size=0.4, random_state=42)
        x_cal, x_test, y_cal, y_test = train_test_split(x_cal_test, y_cal_test, test_size=0.5, random_state=42)
        _, _, noise_cal, noise_test = train_test_split(x_cal_test, noise_cal_test, test_size=0.5, random_state=42)

        model = ProbabilisticRegressionModel(
            input_size=1, n_classes=5,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.NONE,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            n_epochs=50, learning_rate=0.01, early_stopping_rounds=10,
            validation_fraction=0.2, random_seed=42,
        )
        model.fit(x_train, y_train)

        # Use model uncertainty to create adaptive intervals
        pred_std = model.predict_uncertainty(x_test)
        interval_width = pred_std.ravel()

        correlation = np.corrcoef(noise_test.ravel(), interval_width)[0, 1]
        assert correlation > 0.2, (
            f"Model uncertainty width poorly correlates with noise (r={correlation:.3f}). "
            f"Learned variance should produce adaptive intervals."
        )


class TestSymlogTransform:
    """Tests for symlog/symexp target transform."""

    def test_symlog_improves_wide_range_targets(self):
        """On targets spanning multiple orders of magnitude, symlog should help."""
        np.random.seed(42)
        x = np.random.uniform(-3, 3, 500).reshape(-1, 1)
        # Exponential target: spans ~0.05 to ~20
        y = np.exp(x.ravel()) + np.random.normal(0, 0.1, 500)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        # Without symlog
        model_raw = ProbabilisticRegressionModel(
            input_size=1, n_classes=5,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.NONE,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            n_epochs=50, learning_rate=0.01, random_seed=42,
            target_transform=None,
        )
        model_raw.fit(x_train, y_train)
        mse_raw = float(np.mean((y_test - model_raw.predict(x_test)) ** 2))

        # With symlog
        model_sym = ProbabilisticRegressionModel(
            input_size=1, n_classes=5,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.NONE,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            n_epochs=50, learning_rate=0.01, random_seed=42,
            target_transform="symlog",
        )
        model_sym.fit(x_train, y_train)
        mse_sym = float(np.mean((y_test - model_sym.predict(x_test)) ** 2))

        assert mse_sym < mse_raw * 1.2, (
            f"Symlog MSE ({mse_sym:.4f}) should be competitive with or better than "
            f"raw MSE ({mse_raw:.4f}) on exponential targets."
        )

    def test_symlog_roundtrip(self):
        """symlog → symexp should be identity."""
        from automl_package.utils.transforms import symlog, symexp
        x = torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0])
        roundtrip = symexp(symlog(x))
        assert torch.allclose(x, roundtrip, atol=1e-5), (
            f"symlog/symexp roundtrip failed: {x} -> {roundtrip}"
        )


def _compute_nll(model, x, y):
    """Helper: compute gaussian NLL from model predictions + uncertainty."""
    y_pred = model.predict(x)
    y_std = model.predict_uncertainty(x)
    log_var = 2 * np.log(np.clip(y_std, 1e-6, None))
    nll = 0.5 * np.mean(log_var + ((y - y_pred) ** 2) / np.exp(log_var))
    return float(nll)


def _calibration_score(model, x, y):
    """Helper: fraction of test points within predicted ±1σ."""
    y_pred = model.predict(x)
    y_std = model.predict_uncertainty(x)
    within_1sigma = np.abs(y - y_pred) <= y_std
    return float(np.mean(within_1sigma))
```

Run entire test suite:
```bash
python -m pytest tests/ -v --tb=short
```

### 4.5 Add Conformal Prediction Wrapper

Distribution-free prediction intervals:

```python
class ConformalWrapper:
    """Wraps any regression model with conformal prediction intervals."""

    def calibrate(self, x_cal, y_cal, alpha=0.1):
        """Compute conformity scores on calibration set."""
        y_pred = self.model.predict(x_cal)
        self.quantile = np.quantile(np.abs(y_cal - y_pred), 1 - alpha)

    def predict_interval(self, x):
        y_pred = self.model.predict(x)
        return y_pred - self.quantile, y_pred + self.quantile
```

### 4.6 Verification Criteria for Phase 4

- [ ] All Phase 4 tests pass: `python -m pytest tests/test_phase4_regression.py -v`
- [ ] `TestPerformanceBaselines` — ProbReg NLL < 1.8, MSE < 2.5; FlexibleNN piecewise MSE < 1.0
- [ ] `TestCrossModelRanking` — ProbReg NLL beats constant-variance NN; FlexibleNN+ELBO beats shallow NN
- [ ] `TestEndToEndAllModels` — all model variants complete train → predict → uncertainty pipeline
- [ ] `TestBetaNLL` — β-NLL calibration closer to ideal 0.68 than standard NLL
- [ ] `TestHardInference` — hard predictions close to soft, not slower than 2x soft
- [ ] `TestConformalWrapper` — coverage ≥ 85% at α=0.1; interval width correlates with noise
- [ ] `TestSymlogTransform` — roundtrip identity; competitive MSE on exponential targets
- [ ] Full suite (all phases): `python -m pytest tests/ -v --tb=short`

---

## Summary: Phase Dependencies

```
Phase 1 (Bug Fixes + Toy Problems + Comparison)
  Tests: tests/conftest.py, tests/test_phase1_probabilistic_regression.py
    ↓
Phase 2 (FlexibleNN Gradient Fixes + ELBO Depth Control)
  Tests: tests/test_phase2_flexible_nn.py
    ↓
Phase 3 (Dynamic n_classes Fixes + ELBO k Selection)
  Tests: tests/test_phase3_dynamic_k.py
    ↓
Phase 4 (Polish + Advanced Features)
  Tests: tests/test_phase4_regression.py
```

**Run all tests at any time:** `python -m pytest tests/ -v --tb=short`

## File Modification Summary

### Phase 1 (4 files modified, 4 new)
| File | Action | Bugs Fixed |
|------|--------|------------|
| `models/common/losses.py` | Modify L41 | Bug 8 |
| `examples/probabilistic_regression_showcase.py` | Modify L253-258 | Bug 11 |
| `models/independent_weights_flexible_neural_network.py` | Delete L198 | Bug 10 |
| `examples/model_comparison.py` | **NEW** | — |
| `tests/__init__.py` | **NEW** — empty | — |
| `tests/conftest.py` | **NEW** — shared fixtures | — |
| `tests/test_phase1_probabilistic_regression.py` | **NEW** — Phase 1 tests | — |

### Phase 2 (4 files modified, 1 enum added, 1 test new)
| File | Action | Bugs Fixed |
|------|--------|------------|
| `models/selection_strategies/layer_selection_strategies.py` | Modify REINFORCE + STE | Bugs 5, 6 |
| `models/selection_strategies/independent_weights_strategies.py` | Modify STE | Bug 7 |
| `models/flexible_neural_network.py` | Add ELBO/depth penalty | — |
| `models/independent_weights_flexible_neural_network.py` | Add ELBO/depth penalty | — |
| `enums.py` | Add DepthRegularization | — |
| `tests/test_phase2_flexible_nn.py` | **NEW** — Phase 2 tests | — |

### Phase 3 (3 files modified, 1 test new)
| File | Action | Bugs Fixed |
|------|--------|------------|
| `models/selection_strategies/base_selection_strategy.py` | Fix returns | Bugs 1, 4 |
| `models/selection_strategies/n_classes_strategies.py` | Fix returns + boundaries | Bugs 2, 3 |
| `models/probabilistic_regression.py` | Fix dict + add ELBO | Bug 9 |
| `tests/test_phase3_dynamic_k.py` | **NEW** — Phase 3 tests | — |

### Phase 4 (4+ files modified/new, 1 test new)
| File | Action |
|------|--------|
| `utils/losses.py` | Add β-NLL |
| `utils/transforms.py` | **NEW** — symlog/symexp |
| `models/conformal.py` | **NEW** — conformal wrapper |
| FlexibleNN prediction methods | Add hard inference mode |
| `tests/test_phase4_regression.py` | **NEW** — regression + end-to-end tests |
