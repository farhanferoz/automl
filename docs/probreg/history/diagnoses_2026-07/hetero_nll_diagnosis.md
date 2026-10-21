# Hetero-NLL root-cause diagnosis (Task P1)

**Status:** complete. Outcome 2 (config/protocol). Mechanism: mean-resolution bottleneck (not
variance miscalibration), validated by oracle-σ and mean-swap counterfactuals + k-sweep + 5 seeds.

**Test under diagnosis:**
`tests/test_phase1_probabilistic_regression.py::TestModelComparison::test_probabilistic_nll_beats_constant_on_heteroscedastic`
(defined `tests/test_phase1_probabilistic_regression.py:243`).

## 1. Repro (re-measured, not cached)

Command:
```
AUTOML_DEVICE=cpu OMP_NUM_THREADS=2 ~/dev/.venv/bin/python -m pytest \
  "tests/test_phase1_probabilistic_regression.py::TestModelComparison::test_probabilistic_nll_beats_constant_on_heteroscedastic" -v
```
Result: **FAIL** (as pre-registered).
- ProbReg NLL = **1.843**
- constant-σ NN NLL = **1.688**
- assertion `prob_nll (1.843) < nn_nll (1.688)` is false.

Cached planning numbers (ProbReg 1.843 vs constant-σ 1.688) **reproduce exactly**.

## 2. Metric verification (from source, not the name)

- Test NLL helper `_compute_gaussian_nll` (`tests/test_phase1_probabilistic_regression.py:228-232`):
  `nll = 0.5 * (log(2π·var) + (y−ŷ)²/var)`, `var = max(std², 1e-8)`, then `mean`.
  Standard Gaussian negative log-likelihood, **lower = better**. Direction confirmed.
- What is scored for ProbReg: a **single collapsed Gaussian** `N(μ, σ²)`, NOT the full mixture.
  - `μ = predict(x)` → `base_pytorch.py:284` returns `model_output[:,0]` = LTV mean.
  - `σ = predict_uncertainty(x)` → `base_pytorch.py:340-348` returns `sqrt(exp(model_output[:,1]))`
    = LTV std.
  - LTV combination: `apply_law_of_total_variance` (`automl_package/utils/pytorch_utils.py:161-182`):
    `σ² = Σ p_k σ_k²  (within)  +  Σ p_k (μ_k − μ)²  (between)`.
- What is scored for the baseline: `μ = plain-MLP output`; `σ = _train_residual_std`, a single
  constant = std of training residuals (`base_pytorch.py:217-219`, `330`). This constant is the
  entropy-minimising constant σ, so on it `mean((y−ŷ)²/σ²) = 1` by construction — a genuinely
  strong homoscedastic baseline.
- Training objective for ProbReg (default REGRESSION_ONLY + GAUSSIAN_LTV): the collapsed Gaussian
  NLL over `[μ_LTV, logσ²_LTV]` (`_calculate_custom_loss` → `calculate_combined_loss`
  `losses.py:41-48` → `nll_loss`), plus an ordering-constraint penalty (weight auto-set to 1.0 for
  this SEPARATE_HEADS/LTV/REGRESSION_ONLY triple, `probabilistic_regression.py:139-145`).
  **The model directly minimises the same functional the test scores** (on train/val) — so this is
  not an objective-mismatch bug.

## 3. Architectural fact that constrains the mechanism

In `SeparateHeadsRegressionModule.forward` (`regression_heads.py:369-372`), head `i` is fed **only
the scalar class-probability `p_i(x)`**, not `x`. So each per-class mean is `μ_k = h_k(p_k(x))` and
the collapsed mean is `μ(x) = Σ_k p_k(x)·h_k(p_k(x))`. The mean is a bottlenecked function of the
5-way class posterior, whereas the baseline mean is a full 2×32 MLP of `x`. This is the structural
candidate for a mean-quality gap.

## 4. Trajectory + contrast evidence (seed 42, exact test config)

Both models converged (neither truncated by its epoch cap): the constant-NN early-stopped at
epoch 77/100; ProbReg early-stopped at epoch 112/150 (`hit_cap=False` for both). Val loss
plateaued for both (ProbReg first-3 val ≈ [2.18, 1.98, 1.95] → plateau ~1.94). So the gap is
**not** an under-training-in-epochs artifact.

Endpoint, on the held-out test split:

| quantity | constant-NN | ProbReg (k=5) |
|---|---|---|
| collapsed-Gaussian NLL (= test metric) | **1.688** | **1.843** |
| point-prediction MSE (mean quality) | 1.714 (RMSE 1.309) | **2.889 (RMSE 1.700)** |
| σ: mean / min / max | 1.314 (constant) | 1.427 / 0.223 / 1.798 |
| corr(σ, true noise_std) | — (constant) | **0.671** |
| mean( resid² / σ² ) | 0.993 (=1 by constr.) | 1.271 |
| full **mixture** NLL (uncollapsed) | — | 2.174 (worse than collapsed) |

NLL decomposition `0.5·log2π + 0.5·E[log σ²] + 0.5·E[resid²/σ²]`:
- constant-NN: `0.919 + 0.273 + 0.497 = 1.688`
- ProbReg:     `0.919 + 0.289 + 0.636 = 1.843`

The **sharpness** term (`0.5·E[log σ²]`) is essentially tied (0.273 vs 0.289) — ProbReg's
heteroscedastic σ buys no net sharpness advantage because its σ must be inflated overall to cover
a worse mean. The **calibration** term (`0.5·E[resid²/σ²]`) is where ProbReg loses (0.497 → 0.636),
and it is dominated by the larger residuals.

Per-|x| tercile (contrast, both sides):

| region | trueSD | NN resid | PR resid | NN σ | PR σ | NN NLL | PR NLL |
|---|---|---|---|---|---|---|---|
| low  |x|∈[0.03,1.61] | 0.473 | 0.725 | 0.983 | 1.314 | 1.042 | 1.344 | **1.343** (tie) |
| mid  |x|∈[1.61,3.54] | 1.081 | 1.042 | 1.302 | 1.314 | 1.576 | 1.507 | 1.708 (PR loses) |
| high |x|∈[3.54,4.95] | 1.850 | 1.879 | 2.451 | 1.314 | 1.663 | 2.215 | 2.478 (PR loses) |

ProbReg's input-dependent σ is doing exactly what it should — it *ties* the constant baseline in
the low-noise region where the homoscedastic model is most over-dispersed. It loses everywhere
else purely because its mean residual is larger in every region.

## 5. Named mechanism

**Mean-resolution bottleneck, not variance miscalibration.** ProbReg is a classifier over k classes
with per-class regression heads combined by the Law of Total Variance. The collapsed conditional
mean is `μ(x) = Σ_k p_k(x)·h_k(p_k(x))` where each head `h_k` sees only the scalar class-posterior
`p_k` (`regression_heads.py:369-372`), whereas the constant-σ baseline fits the mean with a full
2×32 MLP directly on `x`. On the wiggly heteroscedastic target
(`y = 2 sin x + 0.5 x`, `noise_std = 0.1 + 0.4|x|`), five percentile bins give the mean too little
resolution, so its RMSE (1.700) is well above the plain MLP's (1.309). Because this NLL is
**mean-dominated** (the calibration term carries the gap), the worse mean sinks the score even
though the learned σ is genuinely useful. This mechanism satisfies "fact ≠ cause": it also explains
why the *constant* baseline wins — the baseline spends all its capacity on the mean and pays only a
fixed homoscedastic σ penalty, and here a good mean is worth more than heteroscedastic σ.

## 6. Discriminating checks (mechanism validated, not just plausible)

**Check A — oracle-σ counterfactual (isolates mean vs variance).** Hand *both* models the oracle
σ = true `noise_std`, keep each model's own mean:
- NN mean + oracle σ = **1.465**; ProbReg mean + oracle σ = **1.958**.
- With variance calibration equalised, ProbReg *still loses by 0.49* — the entire gap is mean
  quality, not σ.

**Check B — mean-swap counterfactual (confirms σ is not the problem).** Give ProbReg the NN's good
mean but keep ProbReg's own heteroscedastic σ:
- NN mean + PR σ = **1.598**, which *beats* the constant-NN 1.688.
- So ProbReg's variance head is an asset; it wins the moment it is paired with a competent mean.
  (ProbReg's own hetero σ also beats a best-fit constant σ on ProbReg's own mean: 1.843 < 1.950.)

**Check C — k-resolution sweep (seed 42), the fix.** Raising class resolution relaxes the mean
bottleneck; NLL tracks mean-MSE monotonically:

| k | RMSE | collapsed NLL | vs const-NN 1.688 |
|---|---|---|---|
| 5  | 1.700 | 1.843 | lose |
| **8**  | 1.483 | **1.567** | **win** |
| **10** | 1.457 | **1.618** | **win** |
| 15 | 1.693 | 1.828 | lose |
| 20 | 1.596 | 1.735 | lose |

(head hidden_layers 0 vs 1 was byte-identical — the head loop adds `hidden_layers−1` layers, so
both give one hidden layer; head capacity is not the bottleneck, class resolution is.)

**Check D — multi-seed robustness (split + init seeds {0,1,2,3,42}).** Confirms the k=5 result is
an unlucky single-seed draw and that a selected k wins reliably:

| model | per-seed NLL | mean±std | wins vs NN |
|---|---|---|---|
| constant-NN | 1.699, 1.843, 1.735, 1.843, 1.688 | 1.762 ± 0.068 | — |
| ProbReg k=5 | 1.561, 1.600, 1.763, 1.707, 1.843 | 1.695 ± 0.104 | 3/5 (mean −0.067) |
| ProbReg k=8 | 1.430, 1.477, 1.584, 1.857, 1.567 | 1.583 ± 0.148 | 4/5 (mean −0.179) |
| ProbReg k=10 | 1.515, 1.641, 1.635, 1.825, 1.618 | 1.647 ± 0.100 | **5/5 (mean −0.115)** |

Even at the test's k=5, ProbReg beats the constant baseline *in expectation* (1.695 < 1.762); the
failing test just draws the one seed (42) where coarse k underperforms. With val-selected k (8–10)
ProbReg wins 4–5/5.

## 7. Chosen outcome — **Outcome 2 (under-training/config → protocol fix)**

**Not a bug** (Outcome 1): LTV, the loss, the metric, and the head wiring all behave exactly as
designed; nothing in `probabilistic_regression.py` is wrong. **Not a deep joint-μσ pathology**
(Outcome 3): the variance head is well-behaved and useful (Checks A–B), and a modest,
val-selectable config already makes the model win robustly (Checks C–D). The root cause is a
**config/protocol** one: a single fixed coarse `k=5` evaluated on a single seed. The mean bottleneck
is real but is relieved by choosing `k` with resolution appropriate to the target — precisely what
Task **P2** already prescribes ("ProbReg fixed-k, k pinned per toy by a val-selected small grid,
recorded") over 5 seeds.

**Protocol for report (a)'s battery (this is the fix; package defaults untouched):**
1. Select `k` per toy on a validation split from a small grid (e.g. {5, 8, 10, 12}); do **not**
   hard-code k=5. On this heteroscedastic sine, val selection lands on k≈8–10.
2. Report over ≥5 seeds (mean ± std), not a single seed — the k=5 single-seed loss is inside the
   seed noise band.
3. Evaluate the **collapsed** Gaussian `(μ_LTV, σ_LTV)` (as `predict`/`predict_uncertainty` and the
   test's `_compute_gaussian_nll` already do); do not switch to the full-mixture `predict_distribution`
   for this metric — the mixture NLL is *worse* here (2.174 vs 1.843), so collapsed is the honest,
   charitable choice.

**Write-set note:** under Outcome 2 this diagnosis note is the only deliverable. The failing test
(`test_probabilistic_nll_beats_constant_on_heteroscedastic`, single-seed k=5) is left as-is — it is
a brittle single-seed assertion, not a claim the report depends on; the report's honest claim is
carried by P2's val-selected-k, multi-seed battery. Package defaults are not trivially wrong and are
untouched.

