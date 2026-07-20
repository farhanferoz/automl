# Strand: FlexNN re-validation, MoE build, reports (b) & (c)

**Goal:** (1) re-validate FlexibleNN results post-Phase-9-bug; (2) build a realistic toy MoE
baseline; (3) report (b) — FlexibleNN with full technical/mathematical detail, toy results and
baseline comparisons; (4) report (c) — FlexNN vs MoE: where each wins, mechanistically grounded
in the capacity-programme findings.

Early tasks M0–M2 are dispatchable NOW (parallel with width/depth strands); M3–M5 wait for
**G-JOINT** so the reports carry the final capacity story (MASTER Decision 5 ordering).

**Standing clauses & environment:** per `MASTER.md` Rules.
**Ledger:** `automl_package/examples/report_b_results/` and `.../moe_comparison_results/`
(created by drivers); consumes `shared/bug_audit_head.md` from the probreg strand.

---

### Task M0: FlexNN post-bug re-validation runs

**Files:**
- Read: `automl_package/models/flexible_neural_network.py:24` (`FlexibleHiddenLayersNN`),
  `automl_package/models/independent_weights_flexible_neural_network.py`
- Create (by runs): `automl_package/examples/report_b_results/flexnn_revalidation_*.json`

**Orchestration:** parallel: yes · deps: none · tier: sonnet · scale: static · shape: execution ·
verify: revalidation JSONs exist with per-config convergence flags; a one-paragraph note in this
file states which historical claims survived

Context (memory `project_phase9_bugs`): the FlexNN `n_predictor` was frozen + a tuple-unpack
off-by-one until ~Apr 2026 — **any depth-regularization result predating the fix is
untrustworthy and must be re-measured**, not cited. Re-run the depth-selection demonstrations
the report will cite (the piecewise-dataset ELBO depth-selection claim in
`docs/research_plan.md` Executive Summary), 5 seeds, convergence-gated, trajectory rule
(MASTER Decision 9). Cross-check the `n_predictor` actually receives gradients (assert a
nonzero grad-norm during one training step — cheap instrumentation, remove after).

**Non-goals:** no architecture changes; no new toys (that is the depth strand's job); STE
configs excluded if `shared/bug_audit_head.md` says N2 is OPEN.

### Task M1: pin MoE conventions (verify + cite, then freeze)

**Files:**
- Modify: this file (fill the "Frozen MoE config" block below)

**Orchestration:** parallel: yes · deps: none · tier: sonnet · scale: static · shape: execution ·
verify: the block below is filled with citations (paper + section) for expert count, top-k, and
load-balance loss form — via WebSearch/WebFetch, not memory

Working hypotheses to verify (planning-session recollection, NOT yet citable): top-1 routing
(Switch Transformer), 8 experts / top-2 (Mixtral), load-balancing auxiliary loss
`L_aux = α · E · Σ_e f_e · P_e` with α ≈ 0.01 (Switch; f_e = fraction of tokens routed to e,
P_e = mean router probability), noisy top-k gating (Shazeer et al. 2017).

**Frozen MoE config** (verified against primary sources, M1 2026-07-16):
- **Experts & routing — primary: 8 experts, top-2.** Mixtral of Experts (Jiang et al. 2024,
  arXiv:2401.04088, §2.1 / Table 1): `num_experts=8`, router selects top-2 experts per token.
- **Ablation: top-1 routing.** Switch Transformer (Fedus, Zoph & Shazeer 2022, JMLR 23; arXiv:2101.03961,
  §2): each token → exactly one expert (argmax) — a deliberate simplification of Shazeer 2017's k>1.
- **Gating — deterministic top-k softmax (NO noise):** `G(x) = Softmax(TopK(x·W_g))`,
  `y = Σ_i G(x)_i · Expert_i(x)` (TopK sets non-top-k logits to −∞ before the softmax). Mixtral §2.1.
  *Correction to the planning hypothesis:* NOISY top-k gating (Gaussian-perturbed logits) is Shazeer
  2017-specific (arXiv:1701.06538 §2.1, eq. 3–5); Switch/Mixtral — and M2's spec — use plain deterministic
  top-k. Cite Shazeer 2017 as the noisy-top-k ORIGIN only, not as M2's gating equation.
- **Load-balance auxiliary loss:** Switch Transformer §2.2, eq. 4:
  `L_aux = α · E · Σ_{e=1}^{E} f_e · P_e`, where `f_e` = fraction of tokens dispatched to expert e and
  `P_e` = mean router-probability mass on e (minimized when both are uniform at 1/E). **α = 1e-2** (Switch's
  exact value). Mixtral's paper does not restate an aux-loss — it inherits Switch's — so cite Switch, not
  Mixtral, for the form + α. MASTER Decision 7's stated form matches eq. 4 (paper's `N` = our `E`).
- **Param/FLOP-matching** to FlexNN/width nets uses the accounting in `shared/metrics-accounting.md` (S2).
- Sources fetched: arXiv:2401.04088 (Mixtral), arXiv:2101.03961 (Switch), arXiv:1701.06538 (Shazeer 2017).

### Task M2: MoE toy build

**Files:**
- Create: `automl_package/examples/moe_regression.py`
- Test: its own `--selftest` (this repo's examples convention — no pytest file; see
  `automl_package/examples/nested_width_net.py` selftest pattern)

**Orchestration:** parallel: no · deps: M1 · tier: sonnet · scale: static · shape: execution ·
verify: `AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/moe_regression.py --selftest` exits 0; ruff clean

Spec (decision-complete; internals follow the frozen M1 config):
- `MoERegressionNet(n_experts=8, top_k=2, expert_hidden=H, ...)`: experts are small MLPs
  `Linear(d_in→H) → tanh → Linear(H→1)`; router is a linear layer over the input (mirror
  `capacity_ladder_k6._RouterMLP`'s scale, read `capacity_ladder_k6.py:75-145` first — reuse the
  pattern, not the class, since this router trains jointly).
- Forward: router logits → softmax → top-k dispatch; output = probability-renormalized weighted
  sum of the k selected experts; training loss = MSE + `α·L_aux` (form from M1).
- Both routing modes at eval: `top_k` (executed-FLOPs = k experts) and full-soft (diagnostic).
- Diagnostics in every summary: per-expert load histogram, router entropy, collapsed-expert
  count (load < 1/(4·E)) — routing collapse is a first-class result for report (c), not a
  nuisance.
- **Matching helper:** `match_to_reference(params_or_flops, reference)` sizing `expert_hidden`
  so {total params, executed FLOPs/input} match a given FlexNN/width-net config within 5%,
  using the accounting functions from `shared/metrics-accounting.md` S2 — the matching
  arithmetic must be printed into the summary JSON, never hand-waved.
- Selftest: shapes/finiteness; top-k mask really zeroes non-selected experts' gradient paths
  (perturb a non-selected expert's weights → output unchanged, the
  `nested_width_net.py` selftest-(a) pattern); aux-loss decreases under a synthetic imbalanced
  router; matching helper hits 5% on two reference configs.

**Non-goals:** no transformer blocks, no sparse kernels (dense masked compute is fine at toy
scale — measured FLOPs are computed analytically by S2, not timed); no training batteries here.

### Task M3: FlexNN vs MoE comparison battery — gated on G-JOINT — **superseded-by-F6 (2026-07-18)**

**Status: superseded.** `docs/plans/capacity_programme/flexnn-core.md` Task F6 rescopes this task,
UNGATED from G-JOINT (per `flexnn-core.md`'s MASTER Decision 5 reframe: the two certified 1-D
dials — width, depth — carry the comparison; G-JOINT stays open and is explicitly dropped from
the grid rather than blocking it). `flexnn-core.md` is authoritative for the executed grid,
contenders, and matching regimes from this date forward; the pre-registered hypotheses
(H-flex/H-moe) and toy list below are historical context, not superseded by omission.

**Files:**
- Create: `automl_package/examples/moe_flexnn_comparison.py` (driver; reuses bars/metrics per
  shared/) · Create (by runs): `automl_package/examples/moe_comparison_results/*.json`

**Orchestration:** parallel: no · deps: G-JOINT, M2, M0, shared S1–S3 · tier: sonnet ·
scale: static · shape: execution · verify: results JSONs cover the pre-registered grid with
convergence flags + MoE diagnostics per cell

Pre-registered grid & hypotheses (refine constants in place when G-JOINT lands, before runs):
- **Toys:** the certified width toy, depth toy, joint toy (from the three gates) + `make_hetero3`.
- **Contenders:** FlexNN (certified pattern), MoE (top-2 primary, top-1 ablation), best fixed
  net (val-selected), matched on params AND on executed FLOPs (two matching regimes — report
  both; they answer different questions).
- **H-flex:** FlexNN wins on parameter efficiency and ordered/anytime capacity (prefix
  property); degrades gracefully in scarce data (shared weights regularize).
- **H-moe:** MoE wins where regions want DIFFERENT functions (expert specialization avoids the
  interference the width verdict §3 characterized); risks collapse on smooth/unimodal toys —
  the diagnostics decide, per seed.
- 5 seeds; trajectory rule; stats per S3.

**Non-goals:** no toys beyond the four listed; no MoE variants beyond the frozen M1 config
(top-2 primary, top-1 ablation); no hyperparameter tuning beyond the tuned-α rerun clause in
M5's verify line; no FlexNN architecture changes.

### Task M4: author report (b) — FlexibleNN — **superseded-by-F7 (2026-07-18)**

**Status: superseded.** `docs/plans/capacity_programme/flexnn-core.md` Task F7 (the unified FlexNN
research report) absorbs this task's content — `flexnn-core.md` is authoritative for the report's
structure and scope from this date forward.

**Files:**
- Create: `docs/reports/flexnn_toys/` (own folder)

**Orchestration:** parallel: no · deps: G-JOINT, M0 · tier: sonnet (draft) + opus (cold-read) ·
scale: static · shape: execution · verify: `research-report` skill gates pass; tables cite
`report_b_results/` + the capacity-programme ledgers; no AI provenance; byline = user

Structure: FlexNN model + selection strategies (math lifted from
`docs/mathematical_guide.tex`); the capacity-programme mechanism story as evidence spine
(width verdict §3 + G-DEPTH transfer result + G-JOINT), with the width/depth programmes as
mechanism studies informing the package model; post-bug re-validated results (M0 — historical
numbers may NOT be cited); baseline comparisons like report (a); honest nulls (old depth-toy
null; deploy accuracy-vs-compute tradeoff per `width-cert.md` W7 numbers).

**Non-goals:** no pre-Apr-2026 FlexNN depth-regularization numbers (Phase-9 bug — M0 replaces
them); no UCI/real data; no MoE content (that is report (c), M5); no new experiments — numbers
come from `automl_package/examples/report_b_results/` and the capacity-programme ledgers only.

### Task M5: author report (c) — FlexNN vs MoE — **superseded-by-F7 (2026-07-18)**

**Status: superseded.** `docs/plans/capacity_programme/flexnn-core.md` Task F7 (the unified FlexNN
research report, §5 "MoE comparison") absorbs this task's content, including the tuned-α rerun
clause below (carried into F6's spec verbatim) — `flexnn-core.md` is authoritative for the
report's structure and scope from this date forward.

**Files:**
- Create: `docs/reports/flexnn_vs_moe/` (own folder)

**Orchestration:** parallel: no · deps: M3, M4 · tier: sonnet (draft) + opus (cold-read) ·
scale: static · shape: execution · verify: `research-report` gates pass; every claim traces to
`moe_comparison_results/*.json`; the where-better/where-worse table is symmetric (no
straw-manned MoE: collapse cells report the diagnostic, and a tuned-α rerun is required before
any "MoE fails here" claim)

**Non-goals:** no batteries beyond M3's pre-registered grid plus the tuned-α rerun clause; no
transformer/sparse-kernel MoE scope; no real-data claims; conclusions may not extrapolate past
the toys tested (state the boundary explicitly in the report).

---

## Done ledger

*(orchestrator appends: task · date · evidence path)*
- M1 · 2026-07-16 · MoE conventions frozen (8 experts/top-2 = Mixtral; top-1 = Switch; aux-loss + α=1e-2 = Switch eq. 4; noisy-top-k = Shazeer-only, M2 uses deterministic) · "Frozen MoE config" block above · sources arXiv:2401.04088 / 2101.03961 / 1701.06538
- M2 · 2026-07-16 · `moe_regression.py` built + selftest PASS (shapes/finite; top-k gradient isolation err=0.0; aux-loss ↓ under imbalance 0.080→0.020; `match_to_reference` ≤5% on FlexNN & NestedWidthNet) + ruff clean (orchestrator-verified). Loss = `MSE + L_aux` with α baked into L_aux per Switch eq.4 (avoids α² double-multiply). Diagnostics: per-expert load / router entropy / collapsed-expert count. · `automl_package/examples/moe_regression.py`
- M0 · 2026-07-16 · FlexNN post-Phase-9 re-validation — grad-norm cross-check PASS (n_predictor grad=7.43, fix live). **Historical ELBO depth-selection claim REFUTED**: post-fix ELBO → complete depth-collapse to depth=1 all 5 seeds (posterior collapse to prior `linspace(3,1,5)`, echoes [[project_nested_k_component_starvation]]); NONE control varies but not seed-robust (3/5 expected, 1/5 inverted, 1/5 null); ELBO vs NONE test-MSE indistinguishable. Report (b) cites the collapse, not the old claim. · driver `flexnn_revalidation.py`, `report_b_results/*.json`
- M3/M4/M5 · 2026-07-18 · **superseded** by `flexnn-core.md` Tasks F6 (comparison battery, ungated from G-JOINT) and F7 (unified report) — see the superseded-task headers above · `docs/plans/capacity_programme/flexnn-core.md`
