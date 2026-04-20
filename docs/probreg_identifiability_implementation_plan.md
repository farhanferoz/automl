# ProbReg Identifiability — Implementation Plan

**Session date:** 2026-04-20
**Handover:** Opus planned → Sonnet implements → Opus reviews → Sonnet runs experiments.

## 1. Context

Symmetry-check diagnostic (`automl_package/examples/classifier_symmetry_check.py`)
revealed **head–class index swap degeneracy** in ProbReg with SEP_HEADS under the
default `REGRESSION_ONLY` optimization strategy. Concretely, on the heteroscedastic,
bimodal, piecewise, and exponential toy datasets, for k ∈ {3, 5}:

- `h_i(p_i → 1)` does not anchor at centroid `c_i`; instead it drifts to c_{k-1-i}
  or some permuted value.
- Paired curves h_0, h_{k-1} are mirror images but rotated — their intersection is
  not at c_mid.
- Middle class in k=5 is suppressed (p_mid barely exceeds 0.1).
- MSE remains competitive because the (classifier, heads) pair jointly compensates
  — the degeneracy is a permutation-invariance that the Gaussian-LTV loss cannot
  distinguish.

Root cause: the Gaussian-LTV loss `−log N(y; ŷ_mean, ŷ_var)` depends only on
(ŷ_mean, ŷ_var). Many internal (p, μ, σ) configurations give the same output → loss
cannot resolve which indexing the components take.

ClassReg was verified clean end-to-end separately (see
`automl_package/examples/classreg_probability_sanity.py`): classifier p_i(x)
correctly tracks bin membership on all four toy datasets.

## 2. Three orthogonal fixes

Each addresses identifiability via a different probabilistic/architectural lever:

1. **MDN NLL** — replace Gaussian-LTV with mixture NLL
   `L = −log Σⱼ pⱼ · N(y; μⱼ, σⱼ²)`. Probabilities enter the likelihood directly;
   identifiability emerges because p_j gets pushed up when component j fits y.
2. **CE_STOP_GRAD** — treat bin(y) as an observed auxiliary variable. Joint NLL
   factorizes as `CE on classifier + regression NLL given p`. Implement by
   detaching probs before feeding into heads and adding a CE term on original
   logits. No λ — the weight is 1 by construction.
3. **Anchored heads** — reparametrize `h_i(p_i) = c_i + (1 − p_i) · f_i(p_i)`.
   Hard constraint h_i(1) = c_i; f_i retains full expressivity away from p_i = 1.

## 3. Scope for current session

**In scope:**
- B1 bug fix (monotonic head init)
- C1–C9 ProbReg code changes
- CT1–CT5 tests
- E1–E3 primary experiment (8-cell sweep × 4 datasets × k ∈ {3, 5} × 3 seeds,
  with ClassReg k=3, k=5 re-run inside the same sweep for apples-to-apples baseline)
- Report PDFs (one per dataset) with metrics tables and probability-curve plots

**Deferred** (see RESUME.md "Deferred from current session" section):
- Target transform infrastructure (log, Yeo-Johnson, inversion-layer cleanup,
  Jacobian log-det in NLL)
- Secondary experiments E4 (anchored × monotonic), E5 (middle-class subsumption
  control), E6 (transform diagnostic)

## 4. Staging

Sonnet should commit in this order; each stage is self-contained:

1. **B1** monotonic head init fix — standalone small commit, has a test
2. **Enums** (C1) — add new values, no behavior change
3. **MDN NLL helper** (C2) — add + test in isolation
4. **Anchored head** (C6, C7) — add + test in isolation
5. **Loss branching** (C3, C4, C5) — now wire it all together in training loss
6. **predict_distribution** (C9) — both losses
7. **Tests** (CT1–CT5) — add as each piece lands; integration smoke-test at end
8. **Experiment script** (E1)
9. **Run experiment** (E2) + **generate PDFs** (E3)

**Review checkpoint**: after stage 8, before stage 9 runs. Opus reviews the code
and confirms the experiment script is correct before the runs start.

---

## 5. Detailed item specifications

### B1 — Monotonic head init bug fix

**File:** `automl_package/models/common/regression_heads.py`

**Find:** `_initialize_monotonic_head` (or similar) — contains
`nn.init.normal_(weight, mean=-3.0, ...)`.

**Problem:** `mean=-3.0` forces initial monotonic-head outputs negative. Catastrophic
for all-positive targets (exponential dataset: SEP_MONO MSE = 11 vs 0.45 for SEP
without monotonic). Diagnosed in the D1/D3 head-degeneracy work earlier this session.

**Fix:** replace the hardcoded `mean=-3.0` with centroid-aware initialization:
- If a per-class `c_i` is available at construction (same mechanism we add for
  anchored heads), initialize bias ≈ `c_i` and weight ≈ small-positive (e.g., 0.1).
- If not available, default to `mean=0.0` (never negative by construction, symmetric).

**Test:** `tests/test_monotonic_head_init.py` — construct heads with synthetic
positive centroids (e.g., `c = [1.0, 5.0, 15.0]`), check that `head.forward(p=1.0)`
is close to the centroid (within a reasonable init tolerance, e.g., ±0.5), and NOT
strongly negative (i.e., `head(1.0) > 0` for positive-only centroid cases).

**Acceptance:** the `SEP_MONO` cell in the existing
`head_degeneracy_diagnostic.py` no longer produces MSE > 1.0 on the exponential
dataset. (Run a quick validation after the fix lands.)

---

### C1 — Enum and parameter changes

**File:** `automl_package/enums.py`

Add new enum:

```python
class ProbRegLossType(str, Enum):
    GAUSSIAN_LTV = "gaussian_ltv"   # current behavior — law of total variance + N(y; mean, var)
    MDN = "mdn"                     # Bishop 1994 mixture NLL
```

Extend existing enum:

```python
class ProbabilisticRegressionOptimizationStrategy(str, Enum):
    REGRESSION_ONLY = "regression_only"         # existing default
    JOINT = "joint"                             # existing (if present; leave as-is)
    CE_STOP_GRAD = "ce_stop_grad"               # NEW — classifier gets CE gradient only; heads get regression gradient on detached probs
```

**File:** `automl_package/models/probabilistic_regression.py`

Add three parameters to `ProbabilisticRegressionModel.__init__`:

```python
loss_type: ProbRegLossType = ProbRegLossType.GAUSSIAN_LTV,
# optimization_strategy already exists — no new param, just a new enum value
use_anchored_heads: bool = False,
```

Thread these through to the trainer and forward pass.

**Default behavior unchanged:** existing code paths must produce identical results
when `loss_type=GAUSSIAN_LTV`, `optimization_strategy=REGRESSION_ONLY`,
`use_anchored_heads=False`. Verify by running a pre-existing test suite and
confirming no changes in outputs.

---

### C2 — MDN NLL helper

**File:** `automl_package/utils/losses.py`

Add:

```python
def mdn_nll(
    y: torch.Tensor,          # [batch] or [batch, 1]
    probs: torch.Tensor,      # [batch, k]  — softmax output
    mus: torch.Tensor,        # [batch, k]
    log_vars: torch.Tensor,   # [batch, k]
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Bishop 1994 Mixture Density Network negative log-likelihood.
    L = -mean_i log Σ_j p_j · N(y_i; μ_j, σ_j²)

    Uses log-sum-exp for numerical stability.
    """
    y = y.view(-1, 1)  # [batch, 1]
    # per-component log N(y; μ_j, σ_j²) — shape [batch, k]
    log_N = -0.5 * (math.log(2 * math.pi) + log_vars + (y - mus) ** 2 * torch.exp(-log_vars))
    log_weights = torch.log(probs.clamp_min(eps))    # [batch, k]
    log_mixture = torch.logsumexp(log_weights + log_N, dim=-1)   # [batch]
    return -log_mixture.mean()
```

**Tests** (`tests/test_mdn_nll.py`):
- **CT1a:** Hand-compute NLL for k=2 mixture with known p, μ, σ, y. Assert within
  1e-5 of `mdn_nll` output.
- **CT1b:** When one p_j → 1 and others → 0, MDN NLL reduces to pure Gaussian NLL
  of that single component. Verify numerically.
- **CT1c:** Gradient flows to all three of (probs, mus, log_vars) — assert non-zero
  grads on each via backward pass.
- **CT1d:** Numerical stability — pass extreme log_vars (e.g., ±20) and check no
  NaN/Inf.

---

### C3 — Branch training loss on `loss_type`

**File:** `automl_package/models/probabilistic_regression.py`

Find the training loss computation (look for where `apply_law_of_total_variance`
is called — likely in a method like `_compute_loss` or inside the training loop).

Restructure as:

```python
# After forward pass producing:
#   probs: [batch, k]
#   per_head_outputs: [batch, k, 2]  (mean, log_var)
mus = per_head_outputs[..., 0]
log_vars = per_head_outputs[..., 1]

if self.loss_type == ProbRegLossType.GAUSSIAN_LTV:
    # EXISTING CODE PATH — must be byte-for-byte equivalent
    y_hat_mean, y_hat_var = apply_law_of_total_variance(probs_for_heads, per_head_outputs)
    regression_loss = gaussian_nll(y, y_hat_mean, y_hat_var)
elif self.loss_type == ProbRegLossType.MDN:
    regression_loss = mdn_nll(y, probs_for_heads, mus, log_vars)
else:
    raise ValueError(f"Unknown loss_type: {self.loss_type}")
```

Where `probs_for_heads` comes from C4 (gradient routing).

**Test** (`tests/test_probreg_loss_branch.py` — may be part of CT5 smoke test):
- GAUSSIAN_LTV path produces identical loss to pre-change code on a fixed seed.
- MDN path produces a finite positive loss on the same inputs; differs from
  GAUSSIAN_LTV value.

---

### C4, C5 — Classifier supervision branching

**File:** `automl_package/models/probabilistic_regression.py`

Implement gradient routing:

```python
# probs computed from classifier_layers(x) → softmax
# raw logits must be retained for CE term

if self.optimization_strategy == ProbabilisticRegressionOptimizationStrategy.CE_STOP_GRAD:
    probs_for_heads = probs.detach()
else:
    probs_for_heads = probs

# Forward through regression module using probs_for_heads
per_head_outputs = self.model.regression_module(probs_for_heads, return_head_outputs=True)[1]

# Compute regression_loss as in C3, using probs_for_heads

if self.optimization_strategy == ProbabilisticRegressionOptimizationStrategy.CE_STOP_GRAD:
    # Use ORIGINAL logits (with gradient), not detached
    # Bin labels computed from y using precomputed boundaries
    bin_labels = self._compute_bin_labels(y)   # C5 helper below
    ce_loss = F.cross_entropy(classifier_logits, bin_labels)
    total_loss = regression_loss + ce_loss
else:
    total_loss = regression_loss
```

**C5 — Bin-label helper**

Add method on `ProbabilisticRegressionModel` (or helper in same file):

```python
def _compute_bin_labels(self, y: torch.Tensor) -> torch.Tensor:
    """
    Discretize continuous y into bin indices using precomputed class boundaries.
    Returns a LongTensor of shape [batch] suitable for F.cross_entropy.
    """
    y_np = y.detach().cpu().numpy().flatten()
    boundaries = self.precomputed_class_boundaries[self.n_classes]  # or appropriate k
    _, bin_idx = create_bins(data=y_np, unique_bin_edges=boundaries)
    return torch.tensor(bin_idx, dtype=torch.long, device=y.device)
```

For dynamic-k case, look up boundaries per-sample (existing code does this in
the regression loss path — mirror that logic).

**Test** (`tests/test_ce_stop_grad.py`, part of CT3):
- Construct tiny model, forward pass, backward pass under CE_STOP_GRAD.
- Assert: **classifier params have gradient** (from CE).
- Assert: **classifier params have NO gradient path from regression loss** — test
  by zeroing the CE loss weight temporarily and confirming classifier grads are zero.
- Assert: **head params have gradient** (from regression loss), **no gradient from
  CE** (CE only depends on classifier logits).

---

### C6, C7 — Anchored head

**File:** `automl_package/models/common/regression_heads.py`

Add new class:

```python
class AnchoredHead(BaseRegressionHead):   # or same base as existing SEP heads
    """
    Anchored separate-head parametrization:
        h_i(p_i) = c_i + (1 - p_i) * f_i(p_i)
    for the mean; log_var is left free (not anchored).

    At p_i = 1, h_i(1) = c_i exactly — structural identifiability.
    Away from p_i = 1, f_i retains full expressivity (scaled by 1 - p_i).
    """
    def __init__(self, centroid: float, hidden_size: int = 16,
                 regression_output_size: int = 2, ...):
        super().__init__(...)
        self.register_buffer("centroid", torch.tensor(float(centroid)))
        # f_i network — matches existing BaseRegressionHead body, produces 2 outputs (f_mean, log_var)
        self.f = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, regression_output_size),
        )

    def forward(self, p_i: torch.Tensor, **kwargs) -> torch.Tensor:
        # p_i: [batch, 1]
        f_out = self.f(p_i)               # [batch, 2]  (f_mean, log_var)
        gate = 1.0 - p_i                  # [batch, 1]
        mean = self.centroid + gate * f_out[:, 0:1]
        log_var = f_out[:, 1:2]
        return torch.cat([mean, log_var], dim=-1)   # [batch, 2]
```

**C7** — `SeparateHeadsRegressionModule` construction:

**File:** `automl_package/models/common/regression_heads.py`

When a caller passes `use_anchored_heads=True` and per-class centroids `cs`:

```python
if use_anchored_heads:
    self.heads = nn.ModuleList([
        AnchoredHead(centroid=cs[i], ...) for i in range(n_classes)
    ])
    # Disable middle-class ConstantHead special case — anchored heads already cover it
else:
    # Existing behavior — middle class uses ConstantHead / ProbabilisticMiddleClassHead
    self.heads = nn.ModuleList([...])
```

Thread `use_anchored_heads` + per-class centroids from
`ProbabilisticRegressionModel.fit` (computed via `_bin_centroids` on training y,
same method used by existing `constrain_middle_class` warm-start) down to the
regression module constructor.

**Tests** (`tests/test_anchored_head.py`, CT2):
- **CT2a:** For random f.weights, assert `head(p_i=1.0).mean == centroid` to
  within float tolerance.
- **CT2b:** `head(p_i=0.0).mean == centroid + f(0.0)[0]` — verify the formula.
- **CT2c:** Gradient flows to `f` parameters; `centroid` is a buffer (not a
  parameter) and gets zero gradient.
- **CT2d:** Module construction in `SeparateHeadsRegressionModule` with
  `use_anchored_heads=True` produces k `AnchoredHead` instances with correct
  centroids.

---

### C8 — SINGLE_HEAD_* handling

**File:** `automl_package/models/architectures/probabilistic_regression_net.py`
(or wherever the regression strategy is dispatched)

Anchoring only applies to `SEPARATE_HEADS`. For `SINGLE_HEAD_N_OUTPUTS` and
`SINGLE_HEAD_FINAL_OUTPUT`, `use_anchored_heads=True` is a no-op or raises a
warning:

```python
if self.use_anchored_heads and self.regression_strategy != RegressionStrategy.SEPARATE_HEADS:
    logger.warning(
        "use_anchored_heads=True has no effect for regression_strategy=%s; "
        "anchoring is only defined for SEPARATE_HEADS. Ignoring.",
        self.regression_strategy,
    )
    # Proceed without anchoring
```

Document this limitation in the docstring.

---

### C9 — predict_distribution

**File:** `automl_package/models/probabilistic_regression.py`

Existing `predict_distribution` (added in Phase 9) already supports mixture output.
Verify it works correctly under both loss types:

- **GAUSSIAN_LTV**: returns `GaussianDistribution(mean=ŷ_mean, var=ŷ_var)` — existing
  behavior.
- **MDN**: returns the full mixture via `MixtureOfGaussians(probs, mus, vars)` —
  probably already implemented; if not, add.

Add a test (part of CT4) confirming both paths return sensible predictive
distributions with correct shape and finite density at training points.

---

### CT5 — Integration smoke test

**File:** `tests/test_probreg_identifiability_integration.py`

For each of the 8 cells (loss × supervision × head), run a short training loop
(10 epochs) on a tiny synthetic dataset (100 points, 1D x, heteroscedastic y).
Assert:
- No crashes / NaN losses
- Final MSE is finite and < some loose bound (e.g., 100)
- `predict_distribution(x_test)` returns a finite, sensible output

Keep this test fast (< 30 sec total) so it runs in CI without slowing things down.

---

## 6. Experiment specification

### E1 — Primary sweep script

**File:** `automl_package/examples/probreg_identifiability_sweep.py`

**Datasets:** 4 toy datasets, identical definitions to existing
`classifier_symmetry_check.py` (heteroscedastic, bimodal, piecewise, exponential).

**k values:** {3, 5}

**Seeds:** {42, 123, 7}

**Baseline (re-run in same script):** ClassReg(NN + LOOKUP_MEDIAN) for each
(dataset, k, seed). Same config as `classifier_symmetry_check.py`: 80 epochs,
learning_rate=0.01, early_stopping_rounds=15, validation_fraction=0.2,
calculate_feature_importance=False.

**ProbReg cells** — 8 per (dataset, k, seed):

| Cell | loss_type | optimization_strategy | use_anchored_heads |
|---|---|---|---|
| A | GAUSSIAN_LTV | REGRESSION_ONLY | False |
| B | GAUSSIAN_LTV | REGRESSION_ONLY | True |
| C | GAUSSIAN_LTV | CE_STOP_GRAD | False |
| D | GAUSSIAN_LTV | CE_STOP_GRAD | True |
| E | MDN | REGRESSION_ONLY | False |
| F | MDN | REGRESSION_ONLY | True |
| G | MDN | CE_STOP_GRAD | False |
| H | MDN | CE_STOP_GRAD | True |

Common ProbReg settings: `regression_strategy=SEPARATE_HEADS`,
`uncertainty_method=PROBABILISTIC`, `constrain_middle_class=True` for FREE heads
(cells A, C, E, G) and `False` for ANCHORED heads (cells B, D, F, H — subsumed by
the anchor), `n_epochs=80`, `learning_rate=0.01`, `early_stopping_rounds=15`,
`validation_fraction=0.2`, `calculate_feature_importance=False`,
`use_monotonic_constraints=False` (monotonic explicitly off in primary sweep).

**Total runs:** 4 datasets × 2 k × 3 seeds × (1 ClassReg + 8 ProbReg) = **216 runs**.

**Parallelism:** sequential is fine at ~15 s each = ~55 min wall.

**Runtime note:** CE_STOP_GRAD adds ~10% to train time due to extra CE loss term;
MDN adds ~5%. Budget ~70 min total.

### E2 — Metrics collection

For each run, collect:

| Metric | Scope | Notes |
|---|---|---|
| `mse` | test | `mean((y_pred - y_test)^2)` |
| `nll_own` | test | NLL in the model's own loss family (Gaussian for LTV cells and ClassReg-CONSTANT; MDN for MDN cells) |
| `nll_gaussian` | test | Gaussian NLL — computed for all cells for common comparison |
| `nll_mdn` | test | MDN NLL — computed for all ProbReg cells where per-head (μ, σ) are available |
| `max_p_mid` | eval grid | max over x of `p_{k//2}(x)` — middle-class activity (k=5 only) |
| `anchor_error` | eval grid | For FREE heads only: `max_i |h_i(argmax_p p_i) − c_i|`. Diagnostic for index swap. |
| `component_permutation` | eval grid | Ordered list `[argmax_x p_i(x)_center]` for i = 0..k-1. Should be monotone in c_i for well-identified models. |

For MDN cells, `nll_mdn = nll_own`; for GAUSSIAN_LTV cells, `nll_gaussian = nll_own`.
Cross-NLL is `nll_mdn` for LTV cells and `nll_gaussian` for MDN cells — lets us
compare "which model does better even under the other's metric."

Save: `results.csv` with columns
`(dataset, k, seed, cell, mse, nll_own, nll_gaussian, nll_mdn, max_p_mid, anchor_error)`.

Aggregate: `summary.csv` with mean ± std across seeds per (dataset, k, cell).

### E3 — PDF reports

**Four PDFs**, one per dataset. Name: `results_{dataset}.pdf`.

**Per dataset, pages:**

1. **Page 1 — Metrics summary table**
   - Rows: `(k, cell)` combinations — 20 rows (2 ClassReg + 16 ProbReg, covering
     k ∈ {3, 5} × cells)
   - Columns: MSE, NLL_own, NLL_gaussian, NLL_mdn, max_p_mid, anchor_error —
     mean ± std across 3 seeds
   - Highlight: best cell per column (bold), ClassReg row colored for reference
   - Render as a matplotlib table in a single page

2. **Page 2 — h_i(p_i) curves, k=3**
   - 10-panel grid (2 cols × 5 rows): ClassReg + 8 ProbReg cells + 1 blank/legend
   - Seed 42 only (keeps visual clean)
   - Per panel: `h_i(p_i)` vs p_i for each class, dotted horizontal line at each
     centroid c_i
   - Cell label in title

3. **Page 3 — h_i(p_i) curves, k=5**
   - Same layout, k=5

4. **Page 4 — p_i(x) curves, k=3**
   - Same 10-panel grid
   - Per panel: `p_i(x)` vs x for each class, with training y scatter on twin axis
     (as in `classreg_probability_sanity.py`)

5. **Page 5 — p_i(x) curves, k=5**

6. **Page 6 — Training curves** (optional, if time permits)
   - MSE vs epoch for each cell, seed 42
   - Helps diagnose optimization issues (e.g., MDN cell stuck)

---

## 7. Deliverables

At the end of the session:

1. **Code:** B1 fix + C1–C9 merged; all tests passing
2. **Tests:** CT1–CT5 in place; `pytest tests/` green
3. **Experiment artifacts:**
   - `automl_package/examples/probreg_identifiability_sweep.py` (the script)
   - `automl_package/examples/probreg_identifiability_results/results.csv`
   - `automl_package/examples/probreg_identifiability_results/summary.csv`
   - `automl_package/examples/probreg_identifiability_results/results_heteroscedastic.pdf`
   - `automl_package/examples/probreg_identifiability_results/results_bimodal.pdf`
   - `automl_package/examples/probreg_identifiability_results/results_piecewise.pdf`
   - `automl_package/examples/probreg_identifiability_results/results_exponential.pdf`
4. **Docs:**
   - This file, updated with any spec refinements made during implementation
   - `RESUME.md` updated with current-session outcome pointer
   - Brief analysis writeup: `docs/probreg_identifiability_results.md` — key
     findings across the 4 datasets, predicted vs observed behavior per cell,
     recommendation for which cell to adopt as the new ProbReg default

---

## 8. Review checkpoints

**Checkpoint 1 (Opus review, before experiments run):**

Sonnet completes stages 1–8 (all code + tests + experiment script authored but not
run). Opus reviews:
- B1 fix correctness
- Enum additions and parameter threading
- MDN NLL numerical correctness (hand-verify a small example)
- AnchoredHead formula implementation
- Gradient routing under CE_STOP_GRAD (read the code path carefully)
- Experiment script matches E1 spec: correct cell matrix, seeds, metrics, PDF
  layout

Opus gives green light → Sonnet proceeds to stage 9.

**Checkpoint 2 (after experiments complete):**

Sonnet hands back results + PDFs + writeup. Opus (or user) reviews findings.
Follow-up actions may include:
- Secondary experiments (deferred list in RESUME.md)
- Default recommendation update in `CLAUDE.md` (if a winning cell is clear)
- Research-plan update (`docs/research_plan.md`) if findings reframe the ProbReg
  story for Paper A / Paper B

---

## 9. Anti-scope reminders

These came up in planning but are explicitly **NOT** in this session:

- Target transforms (log, Yeo-Johnson, inversion layer, Jacobian log-det, exact
  moment corrections). See RESUME.md deferred section.
- Monotonic × anchored secondary sweep (E4) — needs B1 landed first; defer to
  next session.
- Middle-class subsumption control sweep (E5) — defer.
- Transform diagnostic sweep (E6) — defer.
- SINGLE_HEAD_* anchoring — explicitly out of scope; anchoring applies to
  SEPARATE_HEADS only.
- HPO over the new dials — the sweep uses fixed hyperparameters; Optuna tuning is
  follow-up work if a cell is adopted as default.
- Real-data validation (photo-z, cluster mass) — Papers A/B, not this session.

---

## 10. Style / housekeeping notes for Sonnet

- Follow existing codebase conventions: Google docstrings, type hints, line
  length 180, ruff-clean.
- All new Python files start with a brief module docstring matching the style in
  `automl_package/examples/classifier_symmetry_check.py`.
- Use `~/dev/.venv/bin/python` for all runs (workspace venv with torch+xpu).
- Match seeds exactly: `(42, 123, 7)`.
- Don't break existing tests. Run `pytest tests/` periodically.
- Don't change defaults: `GAUSSIAN_LTV`, `REGRESSION_ONLY`, `use_anchored_heads=False`.
- Keep commits logical and small; each stage should be one commit minimum.
- If a spec detail here contradicts code reality you find, flag it in the commit
  message and continue with the most conservative interpretation.
