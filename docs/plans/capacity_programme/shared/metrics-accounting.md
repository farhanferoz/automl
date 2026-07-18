# Shared: metrics, accounting, statistics (gates strands 2 & 5; consumed by 3 & 4)

One definition each for the quantities every report table uses. Built ONCE, cited everywhere —
a report may not privately redefine any of these.

### Task S1: calibration + coverage utilities, confirmed against the literature

**Files:**
- Read: `automl_package/utils/metrics.py:171` (`calculate_ece` — PIT-based at HEAD per the
  planning-session check; CONFIRM the full formulation)
- Modify: `automl_package/utils/metrics.py` ONLY if S1 finds a defect (minimum delta; tests
  first) · Test: extend the existing metrics tests (locate via
  `grep -rn "calculate_ece" tests/`)

**Orchestration:** parallel: yes · deps: none · tier: sonnet · scale: static · shape: execution ·
verify: metrics tests green; a note in this file states {ECE formulation verdict, coverage/width
API to use} with `file:line`

Deliverable: the exact callables report batteries use for (i) PIT-based calibration error
(Kuleshov, Fenner & Ermon 2018 formulation — verify against the paper, cite section), (ii) 90%
central-interval empirical coverage, (iii) mean interval width. If all exist at HEAD: this task
is a confirmation note, zero code.

**S1 note (calibration/coverage/width — confirmed at HEAD 2026-07-16; ZERO code change, no defect):**
The three canonical callables report batteries MUST use (a report may not privately redefine them):
1. **PIT-based calibration error** — `Metrics.calculate_ece()` (`automl_package/utils/metrics.py:171`);
   already emitted as `metrics["ece"]` via `calculate_regression_metrics()` (`automl_package/utils/metrics.py:138`).
   PIT construction `norm.cdf(y_true, μ, σ)` → empirical coverage vs target levels → mean-absolute deviation.
   CORRECT vs Kuleshov, Fenner & Ermon 2018 (§3.1/§3.5, eq. 3/4/6/8): identical PIT/coverage construction; the
   final reduction is the L1-mean ("mean absolute calibration error") variant, NOT the paper's eq.-9
   weighted-squared "cal" score — cite in reports as "PIT-based mean-absolute calibration error (Kuleshov et al.
   2018 calibration-curve construction)", not "eq. 9".
2. **90% central-interval coverage** — `calculate_picp_at_alphas(y_true, μ, σ, alphas=(…,0.1,…))['picp@90']`
   (`automl_package/utils/calibration.py:201`); already emitted as `metrics["picp@90"]` (`automl_package/utils/metrics.py:145`).
3. **Mean interval width** — `calculate_mpiw(σ, alpha=0.1)` (`automl_package/utils/calibration.py:229`) for the 90%
   pairing. `automl_package/utils/metrics.py:144` emits only `mpiw@95` by default; `mpiw@90` needs the explicit
   `alpha=0.1` call (one line against the existing correct function — not a gap).

**DUPLICATE TO AVOID:** `automl_package/utils/calibration.py:68` `ece_regression()` is a SECOND, independent
PIT-ECE (n_bins=20, inclusive grid) used by ad-hoc example scripts. Per "built once, cited everywhere",
programme reports use `Metrics.calculate_ece` ONLY — do not reuse `ece_regression`. (No refactor performed —
out of S1 scope.)

### Task S2: params/FLOPs accounting

**Files:**
- Create: `automl_package/examples/capacity_accounting.py` (search first:
  `grep -rn "def count_params\|flops" automl_package/ --include="*.py"` — extend if anything
  exists, state the search result in the module docstring)

**Orchestration:** parallel: yes · deps: none · tier: sonnet · scale: static · shape: execution ·
verify: `AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/capacity_accounting.py --selftest` exits 0

Spec: `param_count(net, path_filter=None)` (e.g. exclude `logvar` heads — the width nets'
MSE-path convention) and `executed_flops(net, config)` — analytic multiply-add counts for the
programme's architectures: width nets at routed width k (3k+1 pattern — derive per class, do
not copy), depth nets at routed depth d, MoE at top-k, FlexNN at selected depth. Selftest:
hand-computed known-answer checks for one small config per family. This module is the ONLY
source of params/FLOPs numbers in reports (b)/(c) and the strand-4 deploy bars.

**S2 note — DONE 2026-07-16:** `param_count`/`executed_flops` built covering `NestedWidthNet`,
`SharedTrunkPerWidthHeadNet`, `IndependentWidthNet`, `SharedReadoutPerWidthAffineNet`, FlexNN
(with/without predictor), plus `DepthNetShapeDescriptor`/`MoEShapeDescriptor` for the depth/MoE
strands. `--selftest` → 18/18 hand-computed known-answer checks PASS (exit 0). Deliverable:
`automl_package/examples/capacity_accounting.py`.

### Task S3: statistics methodology (definition, no code)

**Orchestration:** parallel: yes · deps: none · tier: haiku · scale: static · shape: execution ·
verify: the block below is filled and cites the bootstrap helpers by `file:line`

Frozen once filled: paired-bootstrap SE for same-test-set model contrasts and two-sample
bootstrap for group contrasts (reuse `automl_package/examples/sinc_width_experiment.py`
`_plain_boot_se`/`_two_sample_boot_se` at lines 181/188 — re-verify at execution); decision
threshold 2·SE everywhere; 5 seeds for headline tables, 3 for ablations; per-seed convergence
flags printed in every table (a seed without a trustworthy flag is quarantined, MASTER
Decision 9).

**S3 block (frozen statistics methodology, 2026-07-16 — constants verified at HEAD):**
- **Same-test-set (paired) model contrasts:** i.i.d. bootstrap SE of the per-example paired-difference
  vector's mean — `automl_package/examples/sinc_width_experiment.py:181` `_plain_boot_se`
  (`n_boot=1000`, `seed=0`; `sinc_width_experiment.py:87-88`).
- **Group (unpaired) contrasts** (e.g. centre vs tail region): two-sample bootstrap SE of `mean(a) − mean(b)`
  — `automl_package/examples/sinc_width_experiment.py:188` `_two_sample_boot_se` (same `n_boot=1000`, `seed=0`).
- **Decision threshold:** 2·SE everywhere (A beats B iff Δμ > 2·SE).
- **Seed counts:** 5 seeds for headline tables, 3 for ablations.
- **Convergence flags:** every table prints per-seed convergence flags; a seed lacking a trustworthy flag is
  quarantined from the verdict (MASTER Decision 9).
