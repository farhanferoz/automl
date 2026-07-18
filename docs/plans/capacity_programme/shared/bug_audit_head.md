# Bug ledger re-audit at HEAD (Task P0, `probreg-report.md`)

Audited: the seven N1–N7 findings of `docs/research_plan.md` §1.1, against the current
working tree. Consumed by strand 2 (`probreg-report.md`) and strand 5 (`flexnn-moe.md`).

- HEAD: `6ca4809`
- Date: 2026-07-16
- Method: opened each cited (or line-shifted-successor) site directly; did not trust the
  plan's cached hints. Also located the fix commit (`45e0b32`, "phase 6: fix 7 bugs (N1-N7)")
  as corroborating history, not as a substitute for reading current code.

Report relevance key: **(a)** = ProbReg toys report (`probreg-report.md`), **(b)** =
FlexibleNN report, **(c)** = FlexNN-vs-MoE report (both (b)/(c) in `flexnn-moe.md`).

---

### N1: FlexibleNN ELBO depth prior not normalized

**Verdict: FIXED at HEAD.**

Evidence — `automl_package/models/flexible_neural_network.py:260`:
```
depth_prior_logits = torch.linspace(3.0, 1.0, self.max_hidden_layers, dtype=torch.float, device=_batch_x.device)
```
and the independent-weights variant, `automl_package/models/independent_weights_flexible_neural_network.py:306`
(identical line). Both build the prior logits with `linspace(3.0, 1.0, max_hidden_layers)`,
i.e. the endpoints are fixed regardless of `max_hidden_layers` — steepness no longer scales
with the layer-count hyperparameter, which was the defect.

Relevance: **(b), (c)** only — FlexibleNN depth regularization, not ProbReg. Not relevant to
report (a).

---

### N2: n_classes STE gradient path severed in `_hard_selection_logic`

**Verdict: FIXED at HEAD** (multiplicative-STE pattern is applied) — **but moot for report (a)'s tables**, since the reported dynamic-k model is ELBO+SoftGating, not STE.

Evidence — `automl_package/models/selection_strategies/base_selection_strategy.py:116-153`
(`_hard_selection_logic`), docstring at line 119:
```
"""Applies hard selection using weighted-sum pattern to preserve STE gradients.
```
and the actual accumulation at line 151:
```
final_predictions_contribution += prob_i * predictions_for_mode
```
where `prob_i = mode_selection_one_hot[:, i]` (line 142). The caller,
`automl_package/models/selection_strategies/n_classes_strategies.py:119` (`SteStrategy.forward`):
```
mode_selection_one_hot = f.gumbel_softmax(logits, tau=self.model.gumbel_tau, hard=True, dim=-1)
```
`torch.nn.functional.gumbel_softmax(hard=True)` itself is a straight-through estimator
(`y_hard - y_soft.detach() + y_soft`), so `prob_i` already carries a gradient back to
`logits`/the n_classes predictor; the multiplicative pattern in `_hard_selection_logic`
preserves that path through to the loss. This matches commit `45e0b32`'s stated fix ("STE
gradient path rewritten to weighted-sum in `_hard_selection_logic`").

Note: `automl_package/models/selection_strategies/layer_selection_strategies.py` (FlexNN's
layer-selection strategies) does **not** call this shared helper — it has its own, separately
implemented `SteStrategy` (`layer_selection_strategies.py:144`) — so N2 was ProbReg-only to
begin with and has no bearing on (b)/(c).

Relevance: **(a)** — fixed, but the report's reported dynamic-k model does not use STE
(`_weighted_average_logic`/SoftGating, already correct pre-fix per the original finding), so
this fix does not change any report (a) number. Scope note carried forward:
**STE excluded from report tables** regardless. Not relevant to (b)/(c).

---

### N3: `calculate_ece` is a non-standard formulation

**Verdict: FIXED at HEAD.**

Evidence — `automl_package/utils/metrics.py:171-190` (`calculate_ece`). Full formulation
re-read, not just the first line:
- Line 184: `pit_values = norm.cdf(self.y_true, loc=self.y_pred, scale=y_std)` — standard PIT
  (probability integral transform) under the predicted Gaussian.
- Line 188: `target_levels = np.linspace(1.0 / (n_bins + 1), n_bins / (n_bins + 1), n_bins)` —
  evenly spaced target confidence levels $p_j = j/(m+1)$.
- Line 189: `observed_coverages = np.mean(pit_values[:, np.newaxis] <= target_levels[np.newaxis, :], axis=0)`
  — empirical coverage $\hat p_j = |\{i : F_i(y_i) \le p_j\}| / N$ at each target level.
- Line 190: `return float(np.mean(np.abs(observed_coverages - target_levels)))` — mean absolute
  deviation of empirical from nominal coverage.

This is the PIT-based calibration-curve error of Kuleshov, Fenner & Ermon (2018,
"Accurate Uncertainties for Deep Learning Using Calibrated Regression"): compute the PIT
values, compare empirical vs. nominal coverage at a grid of confidence levels, average the
absolute miscalibration. All four steps are present and match the cited reference, not just
the PIT line. (Cross-check note: worker `s1-calib-metrics` is independently auditing this
same function for the shared metrics-accounting doc; my read — full function, all four
steps — agrees this is the correct PIT-based formulation. Flag for the orchestrator only if
`s1-calib-metrics` disagrees.)

Relevance: **(a)** — directly load-bearing; report (a)'s P2 spec names "PIT-ECE" as a
per-cell metric, which is this function. **(b), (c)** — relevant if either report tables
regression calibration for FlexibleNN/MoE (shared infra, not model-specific).

---

### N4: `beta_nll_loss` omits `log(2π)` constant

**Verdict: FIXED at HEAD** — **not exercised by report (a)'s default config**.

Evidence — `automl_package/utils/losses.py:37` (inside `beta_nll_loss`):
```
per_sample_nll = 0.5 * (math.log(2 * math.pi) + log_var + ((targets - mean) ** 2) / variance)
```
identical in structure to `nll_loss` at `automl_package/utils/losses.py:17`, so the two are
now numerically consistent (same additive constant), which was the defect.

Relevance: **(a)** — only if a reported ProbReg config sets `loss_type="beta_nll"`.
`automl_package/models/probabilistic_regression.py:78` shows the class default is
`"loss_type": "nll"`; report (a)'s P2 spec (fixed-k / dynamic-k models) does not name
`beta_nll`, so this fix is not expected to move any report (a) number unless P2 explicitly
opts in. **(b), (c)** — not relevant (FlexNN/MoE don't use this loss).

---

### N5: Tree-model Gaussian NLL objective uses observed Hessian

**Verdict: FIXED at HEAD.**

Evidence — `automl_package/utils/losses.py:73-74` (`tree_model_gaussian_nll_objective`):
```
# Fisher information for log_var (constant 0.5, always positive, more stable than observed Hessian)
hess_log_var = np.full_like(y_true, 0.5)
```
Uses the (constant, always-positive) Fisher-information Hessian for the `log_var` head
instead of the observed Hessian, which was the defect (slow log_variance convergence at
init from a Hessian that could be near-zero/unstable early in training).

Relevance: **(a)** — only exercised if the report battery's LightGBM baseline sets
`uncertainty_method=PROBABILISTIC` to get an NLL column. Confirmed the objective is wired
conditionally at `automl_package/models/lightgbm_model.py:28-29`
(`if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC: self.objective =
tree_model_gaussian_nll_objective`) — i.e. LightGBM only uses this path when uncertainty is
requested; report (a)'s P2 spec says tree baselines "report NLL via their existing
uncertainty path if one exists, else MSE-only", so if P2 turns on LightGBM's probabilistic
path, this fix is load-bearing for that baseline's NLL column. XGBoost/CatBoost do not use
this function (only `lightgbm_model.py` imports it). Not relevant to **(b), (c)**.

---

### N6: `build_model` called twice in `_fit_single`

**Verdict: FIXED at HEAD.**

Evidence — `automl_package/models/base_pytorch.py`: only two occurrences of `build_model(`
in the file — the abstract definition at line 78 and a single call site at line 132
(inside `_fit_single`, before the training loop). No second call remains later in the
method.

Relevance: **(a), (b), (c)** equally — this is generic `PyTorchModelBase` infrastructure
used by every PyTorch model (ProbReg, FlexNN, plain NN baseline). Per the original finding
this bug wasted compute but did not change outputs, so its fix has no effect on any
reported number, only on wall-clock cost of the report batteries.

---

### N7: `NoneStrategy` (layer selection) `n_probs` has wrong shape

**Verdict: FIXED at HEAD.**

Evidence — `automl_package/models/selection_strategies/layer_selection_strategies.py:44`
(`NoneStrategy.forward`):
```
n_probs = torch.zeros(x_input.size(0), self.model.max_hidden_layers, device=x_input.device)
```
Shape is `(batch, max_hidden_layers)`, matching what `DepthRegularization.ELBO` /
`DEPTH_PENALTY` expect (both index `n_probs` against `depth_prior`/`depth_indices` sized
`max_hidden_layers`, see `flexible_neural_network.py:260-279`) — not the old
`(batch, max_hidden_layers + 1)` that caused a crash when `NoneStrategy` was combined with
either regularizer.

Relevance: **(b), (c)** only — FlexNN layer selection; ProbReg has no `NoneStrategy` (layer
selection) analogue. Not relevant to (a).

---

## Summary table

| # | Verdict | Report-relevant? |
|---|---------|-------------------|
| N1 | FIXED | (b), (c) |
| N2 | FIXED (moot for (a) tables — STE not a reported config) | (a) only, excluded from tables |
| N3 | FIXED | (a) directly load-bearing (PIT-ECE); (b)/(c) if calibration reported |
| N4 | FIXED (not exercised by report (a) default `loss_type="nll"`) | (a) only if `beta_nll` opted in |
| N5 | FIXED | (a) only if LightGBM's probabilistic path is used for its NLL column |
| N6 | FIXED | (a), (b), (c) — compute cost only, no output change |
| N7 | FIXED | (b), (c) only |

No open bugs among N1–N7 at HEAD `6ca4809`. No fix tasks needed from this audit.
