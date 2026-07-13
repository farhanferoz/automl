# Phase 2 Handover Plan — $k$-dimension sweep

**Audience.** A Sonnet follow-up session that will **execute** P2.1, P2.3
and #34, then hand results back for P2.4 analysis.

**Context.** Phase 1 (docs) is done. The current default is Cell B
(Gaussian-LTV + `REGRESSION_ONLY` + ordering, auto-resolved). Philosophical
tension with Cell C is documented in `docs/probreg_identifiability_research.md`
§7.7.1; middle-class-penalty stacking is explicitly rejected in §7.8. Both
hinge on whether dynamic-$k$ absorbs the middle-class emptiness signal. The
$k$-sweep tests that.

Before running anything, re-read:
- `RESUME.md` — status snapshot, plan, task IDs
- `docs/probreg_identifiability_research.md` §7.7, §7.7.1, §7.8, §9.1
- `CLAUDE.md` — uses `~/dev/.venv/bin/python` for **everything**

---

## P2.1 — Dynamic-$k$ × ordering compatibility sanity check (task #47)

**Goal.** Confirm the default auto-resolution (`ordering_constraint_weight=1.0`
for Cell B) composes cleanly with each `NClassesSelectionMethod`. Gate for
P2.3.

**Time.** ~15 min.

**Steps.**

1. Run existing tests touching ordering + dynamic-$k$:
   ```bash
   ~/dev/.venv/bin/python -m pytest tests/test_ordering_constraint.py \
     tests/test_probabilistic_regression.py -v --tb=short
   ```
   All 15 + N tests must pass.

2. Spot-check the 4 selection methods × Cell B on the smallest dataset
   (bimodal, 1 seed, $k=5$, 20 epochs). Minimal script (inline, do not
   commit):
   ```python
   from automl_package.enums import (
       NClassesSelectionMethod, NClassesRegularization,
       ProbRegLossType, ProbabilisticRegressionOptimizationStrategy,
       RegressionStrategy,
   )
   from automl_package.models.probabilistic_regression import (
       ProbabilisticRegressionModel,
   )
   # Sanity: loss finite, gradients finite, predict() runs,
   # ordering_constraint_weight auto-resolves to 1.0 for each.
   ```
   For each `method` in {`SOFT_GATING`, `GUMBEL_SOFTMAX`, `STE`, `REINFORCE`}:
   - Instantiate ProbReg with Cell B defaults + `n_classes_selection_method=method`,
     `n_classes_regularization=K_PENALTY`, `n_classes=7` (upper bound).
   - `fit(x, y)` on 200 bimodal points, 20 epochs.
   - Assert `model.ordering_constraint_weight == 1.0`.
   - Assert no NaN/inf in final loss history.
   - Assert `predict(x)` returns finite, non-degenerate outputs.

3. **Decision gate.**
   - All 4 methods pass → unblock P2.3 (mark #47 completed).
   - One or more fails → **do not proceed to P2.3**. Fix the root cause,
     add a regression test, then re-run step 2. If the failure is in the
     ordering-penalty path specifically for some method, consider
     narrowing P2.3's `dynamic` axis to the passing methods and
     document the omission.

**Do not add** middle-class or other new regularizers. If you think you
need one, stop — §7.8 explicitly rejects that direction.

---

## P2.3 — ProbReg dual-substrate $k$-sweep (task #48)

**Goal.** Two primary questions:
1. Does dynamic-$k$ match or beat best fixed-$k$?
2. Does C + dynamic-$k$ close the exponential gap to B and keep
   non-exponential parity — i.e. does the empirical B-over-C ranking survive?

**Time.** ~2 h wall clock on XPU (Intel Arc). Budget for rerun.

### Grid (trimmed from the raw Cartesian product)

| axis | values | count |
|:---|:---|:---:|
| cell | B, C | 2 |
| loss-type baseline | inherited from cell | — |
| regression_strategy | `SEPARATE_HEADS` | 1 |
| $k$ (or $k_\max$) | 3, 5, 7 | 3 |
| dynamic-$k$ selection | `NONE`, `SOFT_GATING`, `GUMBEL_SOFTMAX` | 3 |
| $k$-regularization | see constraint below | — |
| seed | 5 fresh seeds (42, 123, 7, 2026, 31) | 5 |
| dataset | heteroscedastic, bimodal, piecewise, exponential | 4 |

**Constraint on `(dynamic, k-reg)` pairing:**
- `dynamic == NONE` ⇒ `k-reg = NONE` only (1 combo)
- `dynamic ∈ {SOFT_GATING, GUMBEL_SOFTMAX}` ⇒ `k-reg ∈ {NONE, K_PENALTY, ELBO}` (6 combos)

So valid `(dynamic, k-reg)` pairs = **7**.

**Total runs:** $2 \times 3 \times 7 \times 5 \times 4 = 840$.

**Skip conditions** (can shrink runtime):
- Dynamic-$k$ + $k=3$ is near-degenerate (little room to prune) — keep it
  for completeness but no need to expand.
- `REINFORCE` and `STE` are dropped here — the P2.1 sanity check gates them
  in only if they bring distinct behavior. If P2.1 highlighted a surprise,
  add them; otherwise skip.

### Implementation

Create `automl_package/examples/probreg_k_sweep.py` modelled on
`automl_package/examples/probreg_identifiability_sweep.py`:

- Reuse `_make_datasets()` (factor to `automl_package/examples/_toy_datasets.py`
  if not yet shared — both existing sweeps duplicate it).
- Reuse the per-dataset PDF generator (metrics table + $h_i(p_i)$ plots
  + $p_i(x)$ plots). Add a new page: **"effective $k$" histogram** — for
  dynamic runs, log the soft distribution over $\{1,\dots,k_\max\}$
  post-training and plot its expected value.
- Log to:
  - `probreg_k_sweep_results/results.csv` — per-run row:
    `dataset, cell, k_max, dynamic, k_reg, seed, MSE, NLL_gaussian,
    NLL_mdn, anchor_drop, max_p_mid, effective_k, wall_time_s`
  - `probreg_k_sweep_results/summary.csv` — mean ± std across seeds, grouped
    by `(dataset, cell, k_max, dynamic, k_reg)`
  - `probreg_k_sweep_results/results_{dataset}.pdf` — as above

**Metrics (verify definitions match §4.4).**
- `NLL_gaussian` — always compute for both cells (B uses it as training
  loss; C here too for comparability).
- `NLL_mdn` — optional; populate if the model carries a mixture head.
- `effective_k = sum(p_k * k) where p_k` is the soft selection vector
  over $k \in \{1,\dots,k_\max\}$ averaged over the validation set.
  For `dynamic=NONE`, `effective_k = k_max`.
- `max_p_mid` — max bin probability over the middle third of bins,
  averaged over val set. Matches §6.3.

### Merge with ClassReg k-sweep (#34)

**Task #34 is a prerequisite for P2.4, not for P2.3.** Run it *in parallel*
with P2.3 once P2.1 passes. Existing `classreg_k_sweep.py` has single-seed
MSE-only output from Apr 21; upgrade to:

- $k \in \{2, 3, 5, 7, 10, 15\}$
- 5 seeds (same as P2.3 for direct comparison)
- Add NLL (Gaussian at bin centroid with $\sigma^2 =$ within-bin
  variance) and PICP95 / sharpness
- Add `$k=\infty$` baseline row = `PyTorchNeuralNetwork` direct regression
  (no binning). This is the "pure regression" anchor the user called out.
- Rewrite output to match P2.3 CSV schema where applicable so #49 can
  cross-join easily.

### Gotchas (known)

- **XPU GPU sharing:** `ThreadPoolExecutor` timeout returns control but
  the GPU thread keeps running — see memory note. Run the sweep as a
  single process; do **not** use the per-model timeout wrapper here.
- **BatchNorm + batch-size-1:** Fixed once (`fe63c90`) but watch for
  regressions if val splits create small batches.
- **`auto_resolution` guardrail:** Cell C passes
  `optimization_strategy=CE_STOP_GRAD`, so `ordering_constraint_weight`
  must auto-resolve to `0.0`. Assert this in the sweep harness to catch
  silent default drift.

### Reporting

On completion, produce:
1. `probreg_k_sweep_results/summary.csv` (complete, committed)
2. `probreg_k_sweep_results/results_all.pdf` — merged via
   `pdfunite results_heteroscedastic.pdf results_bimodal.pdf
    results_piecewise.pdf results_exponential.pdf results_all.pdf`
3. A short handoff note (~200 words) in this file under a new
   `## P2.3 — findings` section covering:
   - Which `(dynamic, k-reg)` combo wins per dataset
   - Whether C + dynamic-$k$ closes the exponential gap
   - Whether middle-$k$ emptiness (`max_p_mid` → bin floor) rises with
     $k_\max$ in B (expected: yes), and whether dynamic selection shrinks
     `effective_k` toward data-supported values

Do **not** touch §7.7 or the default policy. P2.4 will decide that.

---

## P2.4 — Analysis (task #49, blocked)

Explicitly **out of scope** for the Sonnet handover. Opus returns to write
§7.7 update based on the produced artifacts.

---

## Handover mechanics

- Branch: work on `master` (repo is already dirty; keep
  `examples/*_results/` out of the commit as usual).
- Commit only code changes (new `probreg_k_sweep.py`, classreg upgrade,
  shared toy-dataset helper if factored), not CSV/PDF artefacts.
- If you need to extend the `_calculate_custom_loss` signature or add a
  new metric to `probabilistic_regression.py`, keep the change minimal
  and add a targeted test. Do not refactor the model.
- If P2.1 reveals a real bug, fix it there and then (memory: "don't
  skip bugs, fix them"). Update §10 of the research doc with a new bug
  entry.
- When P2.3 + #34 are complete, mark #47, #48, #34 as done and leave
  #49, #50, #51 for Opus to claim.
