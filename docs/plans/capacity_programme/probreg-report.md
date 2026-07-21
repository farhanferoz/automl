# Strand: ProbReg report (a) — parallel with width/depth strands

**Goal:** report (a): Probabilistic Regression with **shared-k (fixed `n_classes`)** and
**variable-k (ELBO + SoftGating dynamic `n_classes`)** as two distinct models — full
technical/mathematical detail (lifted from `docs/mathematical_guide.tex`, not re-derived),
results on toys, and comparisons against XGBoost / LightGBM / CatBoost / plain NN on
(i) high-noise & heteroscedastic toys (home turf) and (ii) standard toys (general-workflow
suitability). Toys-only (MASTER Decision 3).

**Standing clauses & environment:** per `MASTER.md` Rules.
**Ledger for this strand:** `automl_package/examples/report_a_results/` (created by the battery
driver) + the audit note below.

Dependency shape: P0 ∥ P1 → (with shared/ S1–S3) → P2 → P3.

---

### Task P0: bug-ledger re-audit at HEAD

**Files:**
- Create: `docs/plans/capacity_programme/shared/bug_audit_head.md` (the updated ledger note — shared/ because strand 5 consumes it too)

**Orchestration:** parallel: yes · deps: none · tier: sonnet · scale: static · shape: discovery ·
verify: the note exists and gives each of N1–N7 a verdict {FIXED at HEAD / OPEN / N-A for reports} with a `file:line` opened at audit time

Audit the seven findings of `docs/research_plan.md` §1.1 against CURRENT code. Known already
(2026-07-16, this plan's authoring session — re-confirm, don't trust):
- **N1** (FlexNN depth prior unnormalized): appears FIXED —
  `automl_package/models/flexnn/depth/model.py:260` and
  `automl_package/models/flexnn/depth/independent_weights.py:306` use
  `torch.linspace(3.0, 1.0, ...)`.
- **N3** (non-standard ECE): appears FIXED — `automl_package/utils/metrics.py:184` computes PIT
  values via `norm.cdf(y_true, loc, scale)`. Confirm the full formulation matches the
  Kuleshov/Fenner/Ermon calibration curve, not just the first line.
- **N2** (n_classes STE gradient severed in `_hard_selection_logic`): site is
  `automl_package/models/flexnn/strategies/base.py:116` — determine
  whether the multiplicative-STE pattern was applied. **Only matters for the report if any
  reported config uses STE**; the reported dynamic-k model is ELBO + SoftGating
  (`_weighted_average_logic`, which was already correct). If STE stays broken: verdict = OPEN,
  scope note "STE excluded from report tables", NOT a fix task.
- N4–N7: verdict + one-line relevance-to-reports each.

**Non-goals:** no fixes in this task (fix tasks are spawned per verdict, by the orchestrator);
no re-audit of the 11 pre-Phase-5 historical bugs.

### Task P1: hetero-NLL root-cause  ⚠️ discovery (this is the report's thesis risk)

**Files:**
- Read/instrument: `automl_package/models/probabilistic_regression.py`, `tests/conftest.py:8` (`heteroscedastic_data` fixture), `tests/test_phase1_probabilistic_regression.py:243`
- Create: `docs/plans/capacity_programme/shared/hetero_nll_diagnosis.md`

**Orchestration:** parallel: yes · deps: none · tier: opus · scale: static · shape: discovery ·
verify: diagnosis note exists, states the mechanism with evidence (loss trajectories, per-component σ readouts), and ends in exactly one of the three pre-registered outcomes

Repro FIRST:
`~/dev/.venv/bin/python -m pytest "tests/test_phase1_probabilistic_regression.py::TestModelComparison::test_probabilistic_nll_beats_constant_on_heteroscedastic" -v`
(expected at strand start: FAIL, ProbReg NLL ≈ 1.843 vs constant-σ 1.688 — re-measure, do not
trust these cached numbers).

Discipline: full trajectory analysis (MASTER Decision 9) — NLL/σ trajectories per training
phase, not endpoints; contrast BOTH models (analysis doctrine: pull both sides); verify the
metric direction in code before concluding.

**Pre-registered outcomes (MASTER Decision 4):**
1. **Shallow bug → minimal fix** (apply, test green, note the diff) — report proceeds on HEAD+fix.
2. **Under-training/config → protocol fix** (report's battery uses the corrected protocol;
   package defaults untouched unless trivially wrong).
3. **Deep joint-μσ pathology → DOCUMENT** — report (a) carries an honest limitation section;
   variance programme stays parked; the test stays failing with a skip-marker + pointer to the
   diagnosis note.

**Non-goals:** no variance-programme redesign; no β-NLL mechanism hunt (parked); nothing beyond
the smallest change that makes the report's claims honest.

### Task P2: report battery — 2 ProbReg variants vs 4 baselines on toys

**Files:**
- Create: `automl_package/examples/report_a_benchmark.py` — ONLY if extension is infeasible:
  FIRST read `automl_package/examples/model_comparison.py`,
  `automl_package/examples/noise_robustness_benchmark.py`,
  `automl_package/examples/full_benchmark.py` and extend/reuse (minimum-viable-code ladder rung
  2; state in the report of this task what was searched and reused).
- Create (by runs): `automl_package/examples/report_a_results/*.json`

**Orchestration:** parallel: no · deps: P0, P1, shared S1–S3 · tier: sonnet · scale: static ·
shape: execution · verify: results JSONs exist covering {6 models} × {toy suite} × 5 seeds with
per-cell metrics {NLL, MSE, PIT-ECE, 90% coverage + mean interval width} and convergence flags

Spec:
- **Models (6):** ProbReg fixed-k (k pinned per toy by a val-selected small grid, recorded);
  ProbReg dynamic-k (ELBO + SoftGating — the CLAUDE.md-documented best combo); XGBoost;
  LightGBM; CatBoost; plain NN (`automl_package/models/neural_network.py`). Tree/NN baselines
  report NLL via their existing uncertainty path if one exists, else MSE-only with an explicit
  dash in NLL columns — no improvised variance heads bolted onto baselines.
- **Toys:** enumerate the existing suite from the three drivers above at execution time
  (April record says heteroscedastic sine / piecewise / bimodal / exponential — verify) +
  the width toy `nested_width_net.make_hetero3` as the high-noise-region stressor. Split:
  home-turf group (heteroscedastic/bimodal/high-σ) vs standard group.
- **Protocol:** 5 seeds; convergence-gated NN training (trajectory rule); data roles disjoint
  (train / val-for-selection / test touched once); stats + accounting per
  `shared/metrics-accounting.md`.
- ProbReg explanations in all artifacts follow the user's framing: a CLASSIFIER over k classes
  with per-class regression heads — never "Gaussian mixture" prose.

**Non-goals:** no UCI/real data; no conformal wrapper section (out of report scope); no HPO
sweeps beyond the pinned small grids (record them).

### Task P3: author report (a)

**Files:**
- Create: `docs/reports/probreg_toys/` (report source + built PDF; own folder per deliverable)

**Orchestration:** parallel: no · deps: P2 · tier: sonnet (draft) + opus (cold-read gate) ·
scale: static · shape: execution · verify: the `research-report` skill's build + cold-read gates
pass; every table cites its `report_a_results/*.json`; zero AI provenance; byline = user

- [x] Invoke the `research-report` skill (it carries the contract: self-contained chapter,
  pdflatex build, fresh-reader cold-read). Math lifted from `docs/mathematical_guide.tex`.
  Structure: model (fixed-k), model (dynamic-k, as a separate model), training objective,
  toys + protocol, results (home turf), results (standard), limitation section per P1's
  outcome, when-to-use guidance.

**Non-goals:** no UCI/real-data content (MASTER Decision 3); no conformal-prediction section;
no new experiments — every number comes from `automl_package/examples/report_a_results/`
(a missing number means P2 was incomplete: reopen P2, do not run ad-hoc experiments here);
no sections beyond the structure listed above.

---

## Done ledger

*(orchestrator appends: task · date · evidence path)*
- P0 · 2026-07-16 · bug-ledger re-audit — **N1–N7 all FIXED at HEAD** (6ca4809), no fix tasks; N3 load-bearing for report (a) PIT-ECE; N2 fixed but STE still excluded from tables (dynamic-k = ELBO+SoftGating); N4/N5 conditional on P2 config (flag to P2 author) · `shared/bug_audit_head.md`
- P1 · 2026-07-16 · hetero-NLL root-cause — **Outcome 2 (protocol fix)**: mean-resolution bottleneck (NOT variance miscalibration; oracle-σ + mean-swap + k-sweep + 5-seed validated). Fix = val-select k per toy (k≈8–10 wins 5/5; k=5 single-seed is the brittle failing test, left as-is), evaluate COLLAPSED Gaussian. No package/test code touched. Variance stays parked. · `shared/hetero_nll_diagnosis.md`
- P2 · 2026-07-16 · report battery COMPLETE + verified on disk — **150/150 cells** {6 models × 5 toys (heteroscedastic/multimodal/hetero3/piecewise/exponential) × 5 seeds}, all `status=ok`, each with {mse, nll, pit_ece, picp@90, mpiw@90, convergence_ok, hit_cap}; 5× `k_selection__<toy>.json` present. hetero3 row (29 cells) completed this session detached; other 4 toys already 30/30. **Report caveat for P3:** 7 cells `hit_cap=True`/`convergence_ok=False`, all piecewise × {catboost,lightgbm} (trees hit iteration cap on the discontinuity — note honestly in the limitation section). · `automl_package/examples/report_a_results/`
- P3 · 2026-07-16 · report (a) delivered — 14pp PDF, 3 cold-read rounds, guard clean · `docs/reports/probreg_toys/report_a_probreg_toys.pdf`
