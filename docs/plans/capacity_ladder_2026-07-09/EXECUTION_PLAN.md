# Capacity-ladder execution plan (LIVE — single source of truth)

**Created 2026-07-09. Status: DRAFT pending user ratification (⛔ RATIFY-0) — no
execution before sign-off.** This is the ONLY execution-level plan for the
capacity-ladder program in this repo. It is updated in place as tasks complete
(superseded content is deleted, not copied — the autocast live-plan convention).
Companions in this folder with fixed roles:

- `INPUT_SCOPE_DOC.md` — the user's 2026-07-09 scope document + task framing (frozen input).
- `SCOPE_REVIEW.md` — the critical review of that document (frozen verdict; its
  corrections C1–C8 are folded in here and cited by number).
- Results of record will be appended to `RESULTS.md` (created by the first task
  that produces a verdict; plans live here, results there).
- `/home/ff235/dev/MLResearch/automl/RESUME.md` `### Now` points HERE.

Predecessor/context documents (read-only): `REVIEW_HANDOVER_2026-07-03.md` and
`STACKING_NOTE_2026-07-05.md` (repo root); the June note
`docs/kselection_variational_em_2026-06-13/kselection_variational_em.md`; the
autocast rank-ladder program
`~/dev/turing/autocast-private/notes/design/rank_ladder/` (design, execution
plan, results — the measured evidence base this plan transfers).

Status legend: ✅ DONE · 🔜 NEXT · ⏸ QUEUED · ⛔ GATE (user sign-off required) ·
🔮 REVIEW (interpretation checkpoint — adjudicator tier, fresh context).

---

## 0. Canonical taxonomy — how capacity is read (anti-muddle section)

One problem, three organs. In every case a CAPACITY parameter (k = number of
classes; d = architecture size; v = variance-model flexibility, with the fitted
σ itself as the degenerate v=0 case) cannot be learned by the training fit
because the in-sample gain of extra capacity (~½ nat per spurious parameter,
measured) beats any fixed in-sample charge (O(10) nats at every admissible
setting, measured — SCOPE_REVIEW C7). Every experiment below tests one or more
of these approaches. **Always name which.**

| # | Approach | How it reads capacity | Evidence used | Standing verdict |
|---|---|---|---|---|
| B1 | In-fit selector (learned weights/gates, in-sample priors, K_PENALTY, ELBO-as-selector, evidence on flexible heads) | training objective concentrates on a capacity | training data | DEAD as a selector (April cap-tracking here; autocast A1/A4; Occam race). K_PENALTY banned by ruling. In-fit machinery is kept ONLY as predictive plumbing, never read. |
| B2 | Post-hoc arbiter (held-out advantage knee) | score table on held-out data; knee of Δ_c at 2·SE | fresh data | THE global read. Validated both projects. Needs its guardrails (§0b G-rules). |
| B3 | Nested ladder (trained) + arbiter | per-sample capacity draws train one model whose prefixes are all valid; read = B2 on its ladder | training for fit, fresh for read | Validated in autocast (Tier-0/1); the transfer object of this plan. |
| B4 | Post-hoc stacking (global π̂) | maximize held-out mixture log-score over the ladder (concave, EM) | fresh data | Validated bias-free; = soft B2. Deployment-value read. |
| B5 | Per-bin stacking / per-input distilled router | B4 per bin of a cheap feature; or a classifier trained on neighbour-averaged B2 reads | fresh data (targets), amortized | The ONLY admissible per-input routes (SCOPE_REVIEW C2). Low capacity of the weight function is the safety mechanism. |
| B6 | Evidence / type-II ML (MacKay; Minka) | marginal likelihood over a GLOBAL low-dim nuisance (σ², weight decay, rank) | training data | Principled for global nuisances on (near-)well-specified models; exact for linear-Gaussian. Breaks under persistent bias / flexible heads (Occam race). WS3 uses it exactly in its validity domain, nowhere else. |

Shared vocabulary. **Score table**: `score[i,c]` = held-out log-likelihood of
example i under sub-model capacity c — the primary artifact of every run
(SCOPE_REVIEW C5); everything post-hoc consumes only this. **Δ_c(x)**: the
neighbour-averaged held-out advantage of capacity c over the reference capacity
at input x. **Knee**: smallest c whose next increment fails 2·SE
(bootstrap). **Ceiling**: the analytic per-input truth available on toys C/D/E.

## 0b. Standing governance (applies to every task; violations = STOP)

- **Environment**: `~/dev/.venv/bin/python` for everything (XPU torch). If the
  venv is broken: `cd ~/dev && uv sync --package automl-package`. The
  variational-EM test suite runs DIRECTLY (`python3 tests/test_variational_em.py`),
  not via pytest. Long runs: launch with
  `setsid nohup systemd-inhibit --what=idle:sleep:handle-lid-switch ...`
  (GNOME idle-suspends on AC at 900 s), background + 30-min checks.
- **Strictly probabilistic — THE PREMISE, not a preference** (standing ruling,
  re-affirmed by the user 2026-07-09 as the basis of the whole repo): every
  training objective, selection criterion and router target must be a
  likelihood, a prior, or a proper-score operation on held-out data. No
  K_PENALTY-class terms, no hand-weighted losses, no load-balancing
  auxiliaries, no tuned λ anywhere. Audit line for every arm in this plan:
  nested draws = a schedule (not a loss term); stacking/knee = held-out
  log-score operations; distillation = an ordinary supervised likelihood on
  measured targets; evidence = marginal likelihood. β-NLL appears ONLY as a
  labelled BASELINE arm (it is an arbitrary reweighting — that is the point of
  including it as the thing to beat).
- **Architecture frozen** (standing ruling): ProbReg stays a
  classifier-bottleneck regressor (classes + per-class heads + bypass). Nested
  training changes the TRAINING SCHEME, not the architecture — ratified by
  ⛔ RATIFY-0 covering this plan.
- **Mixture scoring precondition** (SCOPE_REVIEW C4): all WS1 score tables are
  conditional-mixture log-likelihoods. The blend (law-of-total-variance) summary
  is a deployment output, never a selection signal.
- **Pre-registration**: every task states its predictions and bars BEFORE
  running; outcomes are written against them with no post-hoc reframing. 3 seeds
  {0,1,2} minimum for any verdict; no verdict off a single reading.
- **Escalation**: any pre-registered bar fails, or any result forcing a design
  decision → STOP the lane, write up, hand to 🔮 (adjudicator tier, fresh
  context). Do not tune past a failed bar. Other independent lanes may continue.
- **G-rules for every knee/arbiter read** (SCOPE_REVIEW C3):
  - G1 bootstrap SE: knee = smallest c with Δ_{c+1}−Δ_c < 2·SE, SE by bootstrap
    (block over independent units; plain over i.i.d. toy points), B=1000.
  - G2 abstain: knee=0 (no capacity earns its keep) is a NO-READ, never
    "capacity 0 confirmed"; report the full Δ curve + dispersion.
  - G3 cap-saturation: knee = c_max ⇒ INVALID read; double c_max, rerun that
    case. Caps are per-problem, never inherited.
  - G4 cells: every global read is accompanied by per-bin cells (terciles of the
    relevant observable; stability re-check at sextiles). Cell disagreement with
    a monotone pattern = the confound tell; a global read may not ship without
    its cells.
  - G5 locality guard: per-input reads re-checked at half the neighbour
    width/bin count; a read that moves materially under shrinkage is a pooling
    artifact (manufactured multimodality), not a finding.
  - G6 full-vector rule: stacked weights π̂ and any selector distribution are
    reported as FULL vectors; every scalar summary (mode/median/argmax) is
    banned from verdicts.
- **No unmeasured time estimates**: for any new run shape, run ONE unit,
  measure, extrapolate, report before launching the matrix. Record wall-time in
  every results table.
- **Artifacts**: each task writes to its own folder
  `automl_package/examples/capacity_ladder_results/<task-id>/` (score tables as
  `.pt` with keys `score_mat` (N×C float64), `x` (inputs), `split`, `c_grid`,
  `seed`, plus per-component cached quantities where cheap). Nothing at repo
  root; no scratch files outside the task folder. Before re-running anything,
  check the artifact exists — if it does, the task is done; do not redo.
- **Tests**: library-code changes (anything under `automl_package/`) are
  test-first (the repo's TDD convention, cf. commit 50b1f62); experiment
  scripts under `examples/` need a `--selftest` path with synthetic
  known-answer checks that must PASS before any real read (autocast STACK-0
  convention).
- **Git**: local commits per completed task with plain descriptive messages; no
  push/PR without the user. No AI/tool provenance anywhere.
- **RESUME.md**: update the `### Now` pointer + one-line task status at every
  task completion and every checkpoint.

## 0c. Model/effort policy (orchestrator routing)

| work type | tier |
|---|---|
| build scripts from the locked specs below; run + monitor; fill result tables | task-worker (Sonnet/high); truly mechanical table-filling may drop to utility-worker |
| library code changes (`automl_package/`), TDD | task-worker (Sonnet/high); adjudicator reviews the diff before commit |
| interpretation of completed measurements; any failed bar; any surprise; 🔮 checkpoints | adjudicator (Opus/xhigh), FRESH context — never the session that produced the artifact |
| ⛔ user gates | present + WAIT |

Every dispatched task carries: exact files, the pre-registered block from this
plan verbatim, the verification command, and explicit non-goals (input-contract
rule). Workers never interpret beyond their table; a quiet finder is not a
clean bill.

## 0d. Sequencing and dependencies

```
⛔ RATIFY-0 (user signs this plan; covers the nested-training scheme change)
   │
   ├── WS1 lane:  K0 → K1 → K2 → K3 → 🔮 R1 → K4 → K5 → K6 → 🔮 R2 → ⛔ K7 (real-model port)
   ├── WS2 lane:  F0 → F1 → F2 → F3 → 🔮 R3 → F4 → ⛔ F5 (real-model port)
   └── WS3 lane:  V0 → V1 → V2 → 🔮 R4 → V3 → ⛔ V4 (library port)
```

The three lanes are independent until their ports and may run concurrently
(separate workers, separate artifact folders). K0–K3 are pure post-hoc on
existing artifacts (cheapest de-risk, per STACKING_NOTE §4). Nested training
(K4, F2) starts only after its lane's post-hoc stage has validated the readers.
NOV-1 (novelty search, §5) runs any time before K7/F5.

---

## 1. State of record (what already exists — do not rebuild)

| item | where | status |
|---|---|---|
| Toys with analytic per-input count ceilings: C (spacing-growth bimodal + broad twin), D (staircase 1→2→3), E (hump 1→2→1 + broad twin) | `automl_package/examples/_toy_datasets.py` | validated, tests green |
| Faithful surrogate of the frozen architecture (adaptive heads = real regime) | `automl_package/examples/_variational_em_perinput.py` (`AggregateSparsityKSelector`) | validated (June arc) |
| Per-input held-out arbiter (mixture-vs-single-Gaussian NLL, neighbour-averaged, ~0.01-nat validated) | `_variational_em_perinput.py` + step2/step3 scripts | validated; THE reference reader |
| Comparison harness vs real supported models (XGBoost, MDN, oracle) | `probreg_kselection_comparison.py` → `variational_em_comparison_results/results.json` | run of record bpk6u9xjz |
| Prior ablation (prior adds nothing under adaptive heads) | `probreg_kselection_prior_ablation.py` → `prior_ablation_results/` | run of record b0stdka43 |
| Tests | `tests/test_variational_em.py` (19 test functions; run DIRECTLY — repo-wide pytest collection is broken by an omegaconf conflict, but this file also has a `__main__` runner) + ordering/dynamic-k pytest suites | green at last run |
| Measured facts imported from autocast (not re-derived here) | rank-ladder results note | nested=ML-optimum digit-match; nesting costless (B3); cost flat in cap; uniform draws; in-fit π rails; stacking concave/rail-free bias-free; per-bin stacking beats global exactly where truth varies; per-regime cells catch what gauges miss |

**Code facts of record (2026-07-09 survey; every claim carries file:line in the
survey report — key integration points):**

| fact | where |
|---|---|
| ProbReg dynamic mode is ALREADY over-provisioned to `max_n_classes_for_probabilistic_path` heads, with per-k prefix masking implemented: `_compute_predictions_for_k()` masks classifier logits beyond k and renormalizes | `automl_package/models/architectures/probabilistic_regression_net.py:132-149` |
| Class boundaries are precomputed for EVERY k∈{2..k_max} in dynamic mode (only for k_max in fixed mode) | `automl_package/models/probabilistic_regression.py:501-506` |
| Genuine mixture likelihood exists: `ProbRegLossType.MDN` → `mdn_nll` (log-sum-exp); default is `GAUSSIAN_LTV` (blend) | `probabilistic_regression.py:188-213`, `automl_package/utils/losses.py:104-128` |
| Full mixture components publicly exposed via `predict_distribution()` → `MixtureOfGaussiansDistribution`, BUT guarded: refuses dynamic-k, symlog, SINGLE_HEAD_FINAL_OUTPUT | `probabilistic_regression.py:556-618` (guards :582-594) |
| Ordering penalty auto-weight is 1.0 ONLY for (SEPARATE_HEADS, GAUSSIAN_LTV, REGRESSION_ONLY); with MDN loss it defaults 0 and warns (empirically harmful per `docs/probreg_identifiability_research.md` §9.1) — the C1 conflict resolves itself under mixture scoring | `probabilistic_regression.py:129-142` |
| `n_classes_predictor` layout = logits for k∈{2..k_max} + ONE bypass logit LAST; all consumers assume this order | `probabilistic_regression_net.py:66-75` |
| Sentinel gotcha: `DIRECT_REGRESSION_K_SENTINEL = 2**30`; must be checked with `!=`, never `<` (historical `< inf` bug) | `base_selection_strategy.py:12`, `probabilistic_regression.py:225-227` |
| FlexNN capacity = DEPTH (shared `hidden_size`); prefix property CONFIRMED — soft strategies run blocks sequentially, cache every intermediate representation, apply the shared output layer per depth (= a free all-depth score table per forward); `hard_forward()` batches by depth bucket and runs only layers 1..d | `flexible_neural_network.py:101-151`, `layer_selection_strategies.py:79-181` |
| `IndependentWeightsFlexibleNN` = separate full stacks per depth = the conditioned CONTROL; no batched fixed-depth fast path | `independent_weights_flexible_neural_network.py:124-184` |
| FlexNN supports `(mean, log_var)` output (`UncertaintyMethod.PROBABILISTIC`) — NLL scoring native; blocks MAY contain BatchNorm (ladder arms must pin BN off or use per-depth stats) | `flexible_neural_network.py:82-84,101-111` |
| FlexNN inlines its own batch loop (per-sample depth draws go here); ProbReg uses the shared `base_pytorch` loop | `flexible_neural_network.py:236-285`, `base_pytorch.py:168-186` |
| Variance disease sites: joint `(mean, log_var)` NLL heads (`nll_loss`/`beta_nll_loss`); `UncertaintyMethod.CONSTANT` = global scalar from TRAIN residuals (biased low); `BinnedUncertaintyMixin` calibrates per-bin residual std on TRAIN data; tree models share the joint-NLL objective | `utils/losses.py:10-39,51-101`, `base_pytorch.py:217-219`, `models/common/mixins.py:89-118` |
| `learn_regularization_lambdas`: log-normalizer + penalty optimized on the SAME training batch by a separate Adam — empirical-Bayes-flavored but NOT evidence/held-out; confirmed B1-audit target | `base_pytorch.py:24-70,171-186`, `mixins.py:124-139`, `utils/pytorch_utils.py:11-42` |
| Honest calibration machinery already present: `ConformalWrapper` + `LocallyAdaptiveConformalWrapper` (calibration-split based) | `models/conformal.py:14-119` |
| NO evidence/Laplace/type-II machinery anywhere in `automl_package/` (grepped) | survey §3 |
| NO fourth "readout" data partition exists (train/val/test only) — adding one touches `utils/data_handler.py:118-158` + `base.py:266-317` | survey §5 |
| Strategy registration is TWO-PLACE: the enum in `enums.py` AND the `strategy_map` in the consuming net's `__init__` (not auto-synced) | `enums.py:95-119`, `probabilistic_regression_net.py:55-61`, `flexible_neural_network.py:58-64` |

---

## 2. WS1 — ProbReg k-selection (fixed k / global learned k / per-input k)

**The three regimes map onto the readers, not onto three models:** one nested
ladder is trained ONCE; then fixed k = the global knee (B2), global learned k =
global stacking (B4), per-input k = per-bin stacking + distilled router (B5).
That the SAME trained object serves all three is the point of the transfer.

### K0 🔜 — score-table infrastructure on EXISTING artifacts (pure post-hoc)
**Approach tested: none yet (plumbing).** Build
`automl_package/examples/_capacity_ladder.py` — the shared library for this
program (score tables, EM stacking, knee arbiter, per-bin stacking, bootstrap
SE), plus its selftest.
- Functions (exact contracts):
  - `score_table(model_like, X, y, c_grid) -> (N, C) float64` of per-example
    log-likelihoods; for k-indexed separately-trained models (the existing MDN
    fixed-k sweep) it stacks their per-example scores; for a nested/maskable
    model it evaluates renormalized prefixes from ONE forward pass of cached
    per-component (logit, mean, log_var).
  - `stack_em(score_A) -> pi_hat` maximizing
    `mean_i logsumexp_c(log pi_c + score[i,c])`, EM with
    `q_i = softmax_c(log pi_c + score[i,c])`, `pi <- mean_i q_i`, init uniform,
    stop at max|Δpi| < 1e-10 or 500 iters, torch.logsumexp throughout.
  - `knee(score, ref_c=1, n_boot=1000, block=None) -> (r_star, delta_curve, se)`
    per G1–G3.
  - `perbin_stack(score, bins) -> pi_hat[bins]` + the G4 cell reader.
  - `perinput_curve(score, x, width) -> Δ_c(x)` neighbour-averaged, with the G5
    half-width re-read built in.
- **Selftest (must PASS before any real read):** synthetic (N=4096, C=6) tables:
  (i) all rows favor c=3 ⇒ π̂₃>0.8, knee=3; (ii) two-regime bait (half favor 1,
  half favor 4) ⇒ global π̂≈{.5,.5} on {1,4}, per-bin recovers the split,
  per-bin held-out score beats global by >2·SE; (iii) flat table ⇒ knee
  abstains (G2), π̂ ~ uniform.
- Then build the tables from existing artifacts: the fixed-k MDN sweep and the
  June surrogate on C/D/E (3 seeds), saved per §0b conventions.
- **Non-goals:** no retraining; no touching `automl_package/models/`.

### K1 ⏸ — global stacking + global knee on the existing tables
**Approaches: B2 vs B4 vs (recorded) B1.** Run `stack_em` + `knee` per
(toy, seed); tabulate against the in-sample selector history (April
cap-tracking) as the contrast column.
- **Pre-registered:** (P1) π̂ concentrates on adequate k's and does NOT rail to
  k_max on any toy/seed (fresh data supplies the charge — the STACKING_NOTE §4
  registration); (P2) global knee lands in the adequate band (C:1–2, D:2–3
  pooled, E:1–2 pooled — pooled truth is ambiguous BY DESIGN on D/E, which is
  the point of K2); (P3) on the broad twins (C-broad, E-broad) the knee reads 1
  or abstains — the over-chop trap stays shut.
- **STOP:** π̂ railing on any bias-free toy (would contradict both projects'
  measurements — suspect the table build, then 🔮).

### K2 ⏸ — per-bin stacking (the per-input k read, soft form)
**Approach: B5.** Terciles of x (1-D toys); `perbin_stack` + G4 cells + G5
shrink check (sextiles).
- **Pre-registered:** (P1) per-bin beats global held-out mixture score by >2·SE
  on D and E (truth varies with x); (P2) ties on C (variation weak) and on the
  broad twins; (P3) per-bin π̂ per bin concentrates near the per-bin truth
  (D: 1/2/3 across bins; E: low/high/low).
- **STOP:** P1 fails on BOTH D and E ⇒ the per-input premise of the whole WS1
  program is in question → 🔮 with the tables.

### K3 ⏸ — agreement with the existing per-input arbiter + the [6,3,4] case
**Approach: B5 vs the validated reference reader.** Compare per-bin/per-input
stacked reads against the CondGaussian-reference arbiter's neighbour-averaged
reads (the June instrument) on identical splits; then the seed-coherence check:
where validation best-k was incoherent across seeds ([6,3,4] on D), the stacked
read must be coherent (same winner set across seeds within reported SE).
- **Pre-registered:** agreement where the arbiter is confident; stacking adds a
  calibrated soft answer where the arbiter's hard read was unstable; coherence
  restored on D.
- 🔮 **R1** — adjudicator reviews K0–K3 against the registrations; GO/NO-GO +
  any spec amendment for K4 (nested training). Presented to the user with the
  R1 verdict (async — execution may proceed through the ⛔-free tasks).

### K4 ⏸ (post-R1) — nested-k training of the surrogate (the ladder proper)
**Approach: B3.** One surrogate model (adaptive heads, mixture scoring), trained
with per-sample k ~ Uniform{1..k_max}, k_max=6 on C/E and 8 on D (G3 headroom),
loss = NLL of the renormalized masked mixture over components 1..k (masked
softmax over the active prefix — the `_compute_predictions_for_k` pattern; note
`NoneStrategy`'s own mask is model-wide, NOT per-sample, so it is the pattern
to imitate, not the hook to reuse). 800 epochs, 3 seeds, C/D/E + broad twins.
- **Design rules imported (autocast §3/§6):** uniform draws (a schedule, not a
  prior — no evidence-balance audit needed); no k input to the network (nesting
  replaces conditioning); evaluation at ALL k from one forward pass cached into
  the score table; σ_c floors set from the converged residual scale, never init.
- **Integration facts (survey):** the per-k prefix evaluation ALREADY exists —
  `_compute_predictions_for_k()` (`probabilistic_regression_net.py:132-149`)
  masks and renormalizes at arbitrary k, and boundaries exist for every k in
  dynamic mode. K4's surrogate-level version reuses that pattern; the k=1 rung
  IS the direct/bypass head (single Gaussian), aligning the ladder's bottom
  rung with the existing bypass semantics (April lesson: bypass handoff is the
  SNR-adaptive part). Rung grid = {1 (direct head), 2..k_max}. Use `!=` against
  `DIRECT_REGRESSION_K_SENTINEL`, never `<`.
- **Ordering-fallback arms (literature check Q1/Q3 — registered triggers, try
  in this order ONLY if B-order fails):** (i) sandwich draws (every batch
  includes k=1 and k=k_max; cheap); (ii) boosted smallest-prefix weighting
  (the Matryoshka MRL-boost precedent); (iii) unit-sweeping freeze schedule
  (the supervised nested-dropout precedent). Each is a draw/weighting schedule,
  not a loss term — probabilistic gate intact.
- **Ordering-penalty interaction (SCOPE_REVIEW C1) — pre-registered ablation,
  not a silent choice:** arm (a) ordering penalty OFF for the nested model
  (importance-ordering supplied by nesting pressure alone); arm (b) penalty ON
  restricted to the active prefix. Registered expectation: (a) suffices and (b)
  is neutral-to-harmful on ladder coherence; measured ordering check = across-
  seed stability of prefix content (what component 1 captures, etc.).
- **Pre-registered bars:**
  - (B-coh) ladder coherence, the B3 analogue: per-k held-out NLL of the nested
    model within 2·SE of the SEPARATELY-trained fixed-k models from K0's table,
    at every k, every toy — a single trained object matches the fixed-k sweep.
  - (B-knee) global knee reproduces K1's read on every toy/seed.
  - (B-order) prefix content stable across seeds (subspace/assignment overlap
    reported; instability ⇒ the graded-draw fallback arm may be switched ON —
    pre-registered trigger, mirrors autocast's graded-prior arm).
- **STOP:** B-coh fails at any k by >2·SE on 2+ seeds (nesting is NOT costless
  here — a real divergence from the autocast result; 🔮 before any tuning).
- Runtime rule: one (toy, seed) unit first, measure, extrapolate, report.

### K5 ⏸ — per-input knee on the ladder (the per-input count, hard form)
**Approach: B3+B2 per input.** `perinput_curve` on the nested score table:
Δ_k(x) neighbour-averaged, knee per input, G5 guard; read against the analytic
ceilings.
- **Pre-registered:** D staircase read 1→2→3 with transitions within the
  ceiling's transition bands; E humps 1→2→1; C rises with its known 2σ
  boundary; broad twins stay at 1; recovery ≥ the June arc's 85–96% interior
  benchmark; edge instability (C at x=1.0) tolerated and reported (known
  boundary effect, not a bar).
- This is the headline scientific deliverable: a full per-input
  advantage-vs-k CURVE from ONE model, replacing the single
  mixture-vs-Gaussian advantage of the June instrument.

### K6 ⏸ — router distillation (deployable per-input k)
**Approach: B5-hard.** Train the selector π(x) (a small classifier) on K5's
neighbour-averaged per-input reads: hard arm = classification likelihood on the
knee labels; soft arm = KL to the per-input stacked responsibilities. Strictly
probabilistic (an ordinary supervised likelihood — REVIEW_HANDOVER idea 3).
- **Pre-registered:** hard-routed per-input prediction (each x scored under its
  routed k) attains ≥ the global-k model's held-out mixture NLL on D/E and ties
  on C; router agreement with the K5 reads ≥ 90% within ±1 k on interior
  points; the soft arm is never worse than hard.
- **Pilot arm (registered as a HYPOTHESIS test, per the literature check):** a
  raw per-example argmin-label router — expected to underperform the
  neighbour-averaged targets (our local evidence says so; the literature only
  indirectly corroborates). Whatever the outcome, it is a finding worth one
  table row; it must never become the deployed route without beating (b) on
  the bars.
- **Real-model hook (survey):** the distilled router retargets the EXISTING
  `n_classes_predictor` (`probabilistic_regression_net.py:66-75`) — preserve
  its "k=2..k_max then bypass-LAST" logit layout, which every consumer
  assumes; k=1/bypass reads map to the bypass logit.
- **Non-goal:** no in-fit training of the router jointly with the ladder (that
  is B1 — dead).

### 🔮 R2 ⏸ — WS1 synthesis before the port
Adjudicator: the five-column verdict table (B1 history / B2 / B3 / B4 / B5) ×
(C, D, E, twins), the C1 ordering-ablation outcome, and the port
prerequisites checklist. Output = the K7 spec confirmed or amended. ⛔ presented
to the user together with R1–R4 outcomes.

### ⛔ K7 ⏸ — port to the real `ProbabilisticRegressionModel`
Gated on R2 + user. Scope (locked now, spec finalized at R2): a
`NESTED` member of `NClassesSelectionMethod` + a strategy class in
`n_classes_strategies.py` (two-place registration: `enums.py:112-119` AND the
`strategy_map` at `probabilistic_regression_net.py:55-61`); train-time
per-sample k draws + masked renormalized softmax reusing
`_compute_predictions_for_k`; mixture scoring (`ProbRegLossType.MDN`); public
`predict_at_k(k)` + cached score-table export — which means EXTENDING
`predict_distribution()` past its current guards (`probabilistic_regression.py:582-594`
refuses dynamic-k and symlog; the symlog case exports the table in symlog
space with the MC push-through documented); a fourth READOUT partition added
to `create_train_val_split`/`_prepare_data_partitions`
(`utils/data_handler.py:118-158`, `base.py:266-317`) so held-out selection
never touches the early-stopping split; kNN locality (not 1-D bins) for
per-input reads on real data; benchmark on 2 real datasets (California + one
UCI) with the G-rules; no analytic ceiling — bars are seed-coherence,
calibration (SSR/ACE), and beat-fixed-k-sweep on held-out NLL at matched or
lower average k. Library changes TDD per §0b.

---

## 3. WS2 — FlexibleNN per-input architecture size

FlexibleNN selects among architectures indexed by size (the n_predictor
mechanism); the June/April machinery treats size = depth. The capacity ladder
transfers with one structural difference from WS1: sub-model d+1 REUSES the
trunk of sub-model d (weight sharing across the ladder is the architecture's
existing design), so nesting is natural; the `IndependentWeightsFlexibleNN`
variant is precisely the "conditioned control" arm (free per-capacity weights)
and is kept ONLY to price the nesting compromise (retired when the B3-analogue
bar passes, as in autocast).

### F0 🔜 — prefix-property + scoring audit (RESOLVED by the 2026-07-09 survey; remaining work = record + selftest)
Facts of record (survey; verify by selftest, not re-reading): (i) depth d
reuses blocks 1..d — the soft strategies already cache EVERY intermediate
representation and score the shared output layer per depth
(`layer_selection_strategies.py:79-181`), i.e. the all-depth score table is a
free by-product of one forward pass; (ii) the model emits `(mean, log_var)`
under `UncertaintyMethod.PROBABILISTIC` (`flexible_neural_network.py:82-84`) —
NLL scoring is native, no WS3 shim needed for the primary arm; (iii) per-sample
depth draws hook into the model's OWN inlined batch loop
(`flexible_neural_network.py:236-285`), not the shared base loop; (iv)
`hard_forward()` (`flexible_neural_network.py:120-151`) batches samples by
routed depth and runs only blocks 1..d — the deployment path for F4.
Remaining F0 work: a selftest script asserting (i)/(ii) numerically (cached
per-depth outputs == independent truncated forward, to 1e-6) + the BN
decision recorded: ladder arms run BatchNorm OFF (blocks may contain BN,
`flexible_neural_network.py:101-111`; shared BN statistics are corrupted by
mixed-depth batches — the slimmable-networks lesson; per-depth BN is the
fallback if BN-off costs accuracy, registered).

### F1 ⏸ — toys with input-varying required capacity
New generators in `_toy_datasets.py` (test-first, matching the existing
generator conventions):
- **Toy G (varying need):** 1-D x ∈ [−1,1]; y = a·x + ε for x<0 (linear
  region), y = sin(ω x)·g(x) + ε for x≥0 (compositional region), amplitudes
  matched so marginal variance is comparable; known qualitative truth: required
  capacity higher on x≥0.
- **Toy G-flat (negative control):** same marginal stats, uniform complexity —
  per-input reads must NOT vary (the bypass-confound lesson: smooth data =
  negative control).
- **Toy H (SNR dial):** fixed function, noise σ varying with x — capacity need
  varies through SNR, not structure (k-as-resolution-dial thesis, WS1 memory).
- Ground-truth tests for all three (suite extends `tests/`).
- **Pre-registered:** a fixed-capacity sweep of plain MLPs reproduces the
  qualitative need ordering on G (held-out NLL knee higher on the x≥0 half
  than the x<0 half) — the toys are validated before the ladder touches them.

### F2 ⏸ — nested-depth training + coherence bar
**Approach: B3.** Per-sample d ~ Uniform{1..d_max} (d_max=6, G3 headroom),
loss at the depth-d readout, shared trunk, BN off (F0); implemented in the
inlined batch loop (`flexible_neural_network.py:236-285`) as a new `NESTED`
layer-selection strategy (two-place registration: `enums.py` +
`flexible_neural_network.py:58-64`); `IndependentWeightsFlexibleNN` as the
conditioned control on the same draws.
- **Pre-registered bars:** (B-coh) per-depth held-out NLL of the nested model
  within 2·SE of separately-trained fixed-depth baselines at every d (and of
  the independent-weights control — the nesting-costless bar; control retired
  if it passes); (B-order) trunk stability across seeds.
- **Ordering-fallback arms:** same registered ladder as K4 (sandwich draws →
  boosted smallest-prefix weighting → freeze schedule), same triggers. The
  literature check flags depth-prefix regression as UNVALIDATED territory
  (early-exit interference is documented for classification: BranchyNet/MSDNet)
  — B-coh failing here is a publishable finding either way.
- **STOP:** B-coh fails after the registered fallback arms ⇒ 🔮.

### F3 ⏸ — global reads: knee + stacking; per-bin cells
Same battery as K1/K2 on the F2 ladder over toys G/G-flat/H: global knee,
global π̂, per-bin stacking on terciles of x, G4/G5 guards.
- **Pre-registered:** per-bin beats global on G (>2·SE); ties on G-flat; on H
  the read varies with SNR (the resolution-dial signature) — and the G-flat tie
  is the no-false-positive bar.
- 🔮 **R3** — adjudicator synthesis; GO for F4; spec check for F5.

### F4 ⏸ — n_predictor as the distilled router
**Approach: B5-hard.** The existing n_predictor subnetwork is retargeted as the
distilled router: train it on F3's per-input reads (classification likelihood;
soft arm = KL to responsibilities) with the ladder frozen;
`inference_mode="hard"` then runs only the selected depth per sample —
deployment-ready per-input capacity with honest provenance (the in-fit
DEPTH_PENALTY/ELBO selector paths are B1 = never load-bearing; kept as
historical baselines only).
- **Pre-registered:** hard-routed prediction ≥ global-knee model on G (>2·SE
  better on the mixed regions), tie on G-flat; expected active depth tracks the
  per-region need; report (secondary, not a bar) the compute saving of hard
  routing vs full depth.

### ⛔ F5 ⏸ — port to the real `FlexibleHiddenLayersNN` (gated on R3 + user)
`NESTED` member of `DepthRegularization`-adjacent machinery (per-sample draws in
the training loop; per-depth score-table export; n_predictor distillation as a
fit stage), TDD; benchmark against fixed-depth sweeps on the standing example
datasets; note the Phase-9 bug-era results must not be used as baselines
(memory: re-run any FlexNN depth-reg result from before Apr 2026).

---

## 4. WS3 — variance estimation without in-sample collapse

**The disease (registered, to be demonstrated in V0):** joint (μ, σ) Gaussian
NLL in-sample lets the mean overfit, shrinking in-sample residuals, so σ̂ is
biased low exactly proportionally to the overfit — plus the known gradient
pathologies of joint NLL. **The autocast semantics (adopted):** a fitted
variance is a NUISANCE that legitimately absorbs model error — for calibration
that is wanted — but it must absorb HELD-OUT (honest) error, not in-sample
(shrunken) error. Evidence/type-II ML (B6 — the MacKay/"Minka framework"
family) is adopted exactly in its validity domain: global nuisances,
(near-)well-specified model classes.

### V0 🔜 — the pathology, demonstrated and quantified (the WS3 null)
Toy V-toy1: 1-D heteroscedastic, y = f(x) + σ(x)·ε with known smooth f and
σ(x) (e.g. σ = 0.1+0.3·sigmoid(4x)); N ∈ {200, 1000}; MLP mean+variance heads,
joint NLL, 3 seeds; track per-epoch: in-sample σ̂/σ_true (integrated ratio),
held-out NLL, SSR on train vs held-out (the fit-vs-readout gap tell).
Linear-Gaussian twin V-toy0 (y = wᵀx + σε, well-specified linear model) run
through the same harness.
- **Pre-registered:** (P1) on V-toy1 the in-sample σ̂ ratio falls materially
  below 1 as training proceeds while held-out SSR rises above 1 (collapse +
  its tell); (P2) on V-toy0 with the LINEAR model, in-sample MLE σ̂² is biased
  by the classical factor (N−p)/N and the evidence-fitted σ² corrects it
  (exact, closed-form) — the framework anchored where truth is analytic.
- **STOP:** P1 fails (no collapse) ⇒ the premise needs re-scoping before
  building fixes → 🔮 with the curves.

### V1 ⏸ — global-variance mechanisms, ranked on ground truth
On V-toy0 and a homoscedastic V-toy1h: (i) in-sample MLE σ̂²; (ii) exact
evidence (MacKay α,β fixed-point on the linear model; report the learned
weight-decay too); (iii) held-out σ̂² (mean squared residual on a held-out
half); (iv) K-fold cross-fitted σ̂² (out-of-fold residuals, K=5 — every point
scored by a model that never saw it). NN mean variant of (iii)/(iv) included.
- **Pre-registered:** on well-specified data (ii)≈(iii)≈(iv)≈truth, (i) biased
  low; under deliberate mean-model misspecification (fit linear to curved f)
  the evidence-σ² and held-out-σ² both ABSORB the bias (calibration-correct,
  physically inflated) — recorded as the expected nuisance semantics, not a
  failure; cross-fitted ≈ held-out with lower variance.

### V2 ⏸ — heteroscedastic σ(x) without collapse (the fix battery)
On V-toy1 (+ a 5-D tabular variant V-toy2 with known σ(x)): arms —
(a) joint NLL (baseline/disease — the repo's `nll_loss` path); (b) β-NLL
(β=0.5, existing `beta_nll_loss` hook) — labelled baseline; (c) mean-first
two-stage: fit mean (early-stopped on held-out MSE), freeze, fit σ(x) head on
IN-SAMPLE residuals (still-diseased control); (d) mean-first + σ(x) on
CROSS-FITTED residuals (the arbiter-family fix: out-of-fold residuals from
K=5 mean fits, then one σ(x) head fit on all points' honest residuals —
literature precedent: the 2026 practical-heteroscedastic paper + classical
two-stage variance-function estimation); (e) per-bin scale recalibration on
held-out data on top of (c) (the measured STACK-2b remedy: fit a scalar s per
tercile on half the held-out split, score on the other half, swap);
(f) `LocallyAdaptiveConformalWrapper` (`models/conformal.py:64-119`) as the
existing honest-calibration comparator (interval metrics only — it is not a
density).
- **Pre-registered:** σ̂(x)/σ(x) integrated error ranks (d) ≤ (e) < (b) < (c)
  < (a); held-out NLL + SSR/ACE rank the same way within 2·SE; (d) attains
  SSR ∈ [0.9, 1.1] on 3/3 seeds at N=1000.
- **STOP:** (a) beats (d) anywhere ⇒ harness bug until proven otherwise, then 🔮.
- 🔮 **R4** — adjudicator: mechanism ranking table + the V3/V4 spec check.

### V3 ⏸ — the variance-capacity ladder (unification)
**Approach: B2 applied to v.** σ-model rungs: v0 global scalar → v1 per-bin
(terciles) → v2 linear-in-x log-σ → v3 small MLP head; all fitted by (d)'s
cross-fitted mechanism; held-out NLL knee (G1–G4) selects v.
- **Pre-registered:** on homoscedastic twins the knee = v0 (no false
  structure); on V-toy1/V-toy2 the knee lands at the smallest rung matching
  the planted σ(x) shape; the same `_capacity_ladder.py` readers run unchanged
  (the unification claim — one selection machinery for k, d, and v).

### ⛔ V4 ⏸ — library port (gated on R4 + user)
Port the winning mechanism (registered expectation: cross-fitted σ plus
optional per-bin recalibration; evidence for the linear model's global σ²/
weight-decay) into the repo models: the NN regression model's variance path,
linear regression's variance option, and the per-class σ_c floors in ProbReg
(floor-from-converged-residual rule). Deprecate (with a loud docstring, keep
functional) the pure joint-NLL default where a collapse-free option exists.
Specific library fixes queued by the survey (small, high-value, same gate):
- `UncertaintyMethod.CONSTANT` computes its global scalar from TRAIN residuals
  (`base_pytorch.py:217-219`) — biased low by construction; move to
  validation residuals (one-line semantics change + test).
- `BinnedUncertaintyMixin.calibrate_uncertainty` calibrates on the data it is
  handed — `BaseModel.fit()` hands it train+val (`base.py:247-249`); require a
  held-out calibration split (align with the K7/V4 readout partition).
- Tree-model Gaussian-NLL objectives (`utils/losses.py:51-101`, used by
  LightGBM et al.) share the joint-NLL disease — in-scope for the deprecation
  notice, out-of-scope for re-engineering (recorded, not fixed).
- `learn_regularization_lambdas` audit CONFIRMED as a B1-class device: the
  log-normalizer+penalty objective is optimized on the SAME training batch
  (`base_pytorch.py:171-186`, `utils/pytorch_utils.py:11-42`) — a
  joint-MAP-style scheme, which the autocast graveyard (§9.3: joint MAP over
  parameters and their prior scales is pruning-biased vs true type-II ML)
  says is the wrong objective. V4 either moves the lambda objective to the
  evidence (exact for the linear model; Laplace/Immer-2021-style for MLPs) or
  to a held-out criterion — decided at R4 on V1's measured evidence.
TDD throughout.

---

## 5. Cross-cutting tasks

- **NOV-1 ⏸ (before K7/F5):** the owed novelty search — per-input mixture
  cardinality by held-out local likelihood-ratio reads vs the conditional-
  density / mixture-testing / hierarchical-stacking literature; deliverable =
  positioning paragraph + overlap table (adjudicator tier; the 2026-07-09
  focused lit check is the seed, not the substitute). Two positioning angles
  the 2026-07-09 check already established: (i) NO published validation of
  prefix-nested training for REGRESSION exists (all Matryoshka-line evidence is
  classification/retrieval/embeddings) — the K4/F2 coherence results are
  publishable on that gap alone; (ii) hierarchical stacking (Yao et al. 2021)
  is the closest prior art for per-input weights and must be engaged directly
  (our differentiator: the per-input COUNT claim graded against analytic
  ceilings, plus the ladder giving all sub-models from one trained object).
- **REP-1 ⏸ (with each 🔮):** RESULTS.md sections are written facts-first
  against the registrations (autocast results-note convention), one section
  per task, artifacts linked by path.
- **NOTE-MOE ⏸ (user-requested 2026-07-09; any time after K6, ideal after K7;
  EXECUTED AT THE EXECUTION STAGE by the orchestrator's agent — not part of
  the planning session):** a mathematical research note contrasting this
  program with sparse Mixture-of-Experts as used in large language models —
  the two frameworks solve the same problem (per-input activation of a subset
  of a large model's capacity) with opposite methodologies. **MUST be authored
  with the research-report skill** (self-contained book-chapter standard,
  buildable PDF, cold-read gate, no code references, technical/mathematical
  throughout). Content contract:
  (1) Both frameworks written as probability statements: the MoE layer as a
  conditional mixture p(y|x) = Σ_e g_e(x)·p_e(y|x) and what top-k gating,
  capacity factors and token dropping do to that statement (a truncated,
  renormalized — or NOT renormalized — approximation trained by straight-through
  heuristics); our ladder as an ordered family {p_c(y|x)} with selection by
  held-out proper score.
  (2) The arbitrariness audit (verified by the 2026-07-09 literature check,
  Q9): which MoE ingredients are derivable from a single probabilistic
  objective and which are acknowledged heuristic stabilizers (load-balancing
  auxiliary losses with hand-tuned coefficients, router z-loss, capacity
  factor, noisy gating) — the user's claim to confirm/refute WITH CITATIONS,
  including the principled minority lineage (EM/variational MoE from Jacobs &
  Jordan 1991; balanced-assignment and aux-loss-free variants) and what each
  relocates rather than removes.
  (3) The structural correspondence table: in-fit router ↔ B1 (dead here —
  April cap-tracking / autocast π-railing are the SAME mechanism as MoE
  expert collapse: in-fit gate training with no honest complexity charge);
  load-balancing auxiliary ↔ K_PENALTY-class term (banned) whose principled
  counterpart is the aggregate Dirichlet usage prior (June arc) — one prior,
  charged once, on dataset-average usage, in place of a tuned penalty;
  top-k inference ↔ hard routing bridged by arbiter distillation (B5) instead
  of train/inference mismatch; expected-capacity control ↔ per-bin stacking.
  (4) The honest limits of the contrast: scale (their routers train on 10^12
  tokens in-fit because held-out charging at that scale is expensive; the
  Occam-race arithmetic still applies but the bias/variance trade lands
  differently), and the parts of MoE practice our framework has no answer for
  (hardware-driven capacity constraints).
  Deliverable: `docs/moe_contrast_<date>/` note (own folder), buildable PDF,
  cold-read gate per the research-report skill. Adjudicator tier drafts;
  mathematical statements verified against the cited papers, not paraphrased
  from memory.

- **REPORT-2 ⏸ (user-requested 2026-07-09) — the program report, living:** a
  single TECHNICAL, SELF-CONTAINED research report on the results of all three
  workstreams, **authored and every-time updated with the research-report
  skill** (book-chapter standard: readable cold, no code references, buildable
  PDF, fresh-reader cold-read gate on every revision). Location:
  `docs/capacity_ladder_report_<date>/` (own folder). Structure contract:
  (1) the shared failure mechanism (in-sample capacity learning and the Occam
  race), stated once, mathematically; (2) the method (nested ladder; held-out
  arbiter/knee with guardrails; stacking global → per-bin → distilled router;
  evidence for global nuisances) as probability statements; (3) one results
  chapter per workstream (ProbReg k / FlexibleNN depth / variance), written
  facts-first against the pre-registered bars, updated at each 🔮 checkpoint
  (R1–R4) as its lane's results land; (4) limitations + the identifiability
  boundary. First draft due at the FIRST 🔮 checkpoint that closes with
  results (expected R1); every later checkpoint updates it — the report is a
  standing deliverable, not an end-of-program afterthought. NOTE-MOE remains a
  SEPARATE note; REPORT-2 references it rather than absorbing it.

## 6. Risks (beyond the per-task STOPs)

| risk | tell | response |
|---|---|---|
| Prefix-of-components invalid for ProbReg despite renormalization (C1) | K4 B-coh fails while K0's separately-trained table was fine | 🔮; fallback = k-conditioned ladder (per-k binning, shared trunk) — loses weight-sharing elegance, keeps the readers |
| Ordering penalty vs nesting conflict harms identifiability work | K4 arm-(b) materially worse; ordering tests in `tests/test_ordering_constraint.py` regress | keep penalty OFF for ladder arm; document interaction; the identifiability research thread is per-k-fixed and unaffected |
| Neighbour-width pooling manufactures multimodality on real data (K7) | G5 shrink-test moves the read | widen data, not the window; report abstain |
| Cross-fitted variance too expensive on big models (V4) | measured cost > 5× single fit | K=2 cross-fitting arm (registered); or held-out-half fitting |
| Worker drift during autonomous execution | any deviation from a registered block | §0b escalation — STOP the lane, never tune past a bar |
| BatchNorm statistics corrupted by mixed-capacity batches (FlexNN blocks may contain BN) | F2 B-coh fails only with BN on | ladder arms run BN-off (F0 decision); per-depth BN statistics as the registered fallback |
| Weak self-ordering of prefixes in the NN setting (literature: supervised nested dropout needed unit-sweeping; no regression precedent) | K4/F2 B-order instability across seeds | the registered fallback ladder: sandwich draws → boosted smallest-prefix weighting → freeze schedule (all schedules, not loss terms) |
| Fabricated/unverifiable citations entering the notes (one caught 2026-07-09) | any load-bearing quote not verified against the source | REPORT-2/NOTE-MOE rule: quotes verified against the actual paper, else the claim is downgraded to "our measurement/hypothesis" |

## 7. References (verified 2026-07-09; details + caveats in SCOPE_REVIEW §5)

Load-bearing:
1. Yao, Pirš, Vehtari, Gelman (2021/22, Bayesian Analysis). *Bayesian
   Hierarchical Stacking: Some Models Are (Somewhere) Useful.* arXiv:2101.08954 —
   per-input held-out-fitted model weights with partial pooling (B5 reference).
2. Yao, Vehtari, Simpson, Gelman (2018, Bayesian Analysis). *Using Stacking to
   Average Bayesian Predictive Distributions.* — held-out log-score stacking (B4).
3. Rippel, Gelbart, Adams (2014, ICML). *Learning Ordered Representations with
   Nested Dropout.* arXiv:1402.0915 — ordered-prefix training; PCA-equivalence
   for semi-linear autoencoders; supervised follow-up arXiv:1412.7155 (unit
   sweeping).
4. Kusupati et al. (2022, NeurIPS). *Matryoshka Representation Learning.*
   arXiv:2205.13147 — prefix quality preservation at scale + the
   smallest-prefix boosting caveat.
5. Yu & Huang (2019, ICCV). *Universally Slimmable Networks and Improved
   Training Techniques.* arXiv:1903.05134 — sandwich rule, in-place
   distillation; Yu et al. (2019, ICLR) *Slimmable Neural Networks*
   arXiv:1812.08928 — switchable BN.
6. Seitzer et al. (2022, ICLR). *On the Pitfalls of Heteroscedastic Uncertainty
   Estimation with Probabilistic Neural Networks.* arXiv:2203.09168 — β-NLL;
   the pathology documentation (V-baseline).
7. Stirn et al. (2023, AISTATS). *Faithful Heteroscedastic Regression with
   Neural Networks.* arXiv:2212.09184 — joint-NLL mean degradation +
   stop-gradient fix (V2 arm).
8. *Practical Deep Heteroskedastic Regression* (2026). arXiv:2603.01750 —
   mean-first, variance-on-held-out; the direct precedent for V2 arm (d)
   (new, to be reproduced rather than leaned on).
9. Immer, Bauer, Fischer, Rätsch, Khan (2021, ICML). *Scalable Marginal
   Likelihood Estimation for Model Selection in Deep Learning.*
   arXiv:2104.04975 — modern evidence framework (B6, V-arms); Immer et al.
   (2023, NeurIPS) *Effective Bayesian Heteroscedastic Regression with Deep
   Neural Networks* — Laplace marginal likelihood for heteroscedastic heads.
10. MacKay (1992, Neural Computation). *Bayesian Interpolation* / *The Evidence
    Framework Applied to Classification Networks* — type-II ML for weight
    decay and noise (B6 foundation).
11. Minka (2000, NeurIPS). *Automatic Choice of Dimensionality for PCA* — the
    evidence-based rank selector (the family's exemplar; misspecification
    behaviour measured in-house, not in the literature).
12. Skafte, Jørgensen, Hauberg (2019, NeurIPS). *Reliable Training and
    Estimation of Variance Networks.* arXiv:1906.03260 — named post-hoc
    mean-then-variance precedent (mechanism unverified from abstract).

Context (per-input routing / adaptive compute; indirect support only):
Graves (2016) ACT arXiv:1603.08983; Banino et al. (2021) PonderNet
arXiv:2107.05407; Huang et al. (2018) MSDNet arXiv:1703.09844; Wang et al.
(2018) SkipNet; Teerapittayanon et al. (2017) BranchyNet.

MoE-contrast citations (NOTE-MOE): SCOPE_REVIEW §6 (the Q9 annex) — verdict
SUPPORTED; key sources: Shazeer et al. 2017 arXiv:1701.06538; Lepikhin et al.
2020 (GShard); Fedus, Zoph, Shazeer 2021 (Switch Transformer); Zoph et al.
2022 arXiv:2202.08906 (ST-MoE z-loss); Wang et al. 2024 arXiv:2408.15664
(routing collapse; aux-loss-free bias); DeepSeek-V3 arXiv:2412.19437;
arXiv:2512.03915 (dual-ascent framing); Jacobs et al. 1991 + Jordan & Jacobs
1994 (EM lineage); Lewis et al. 2021 arXiv:2103.16716 (BASE); Zhou et al. 2022
arXiv:2202.09368 (Expert Choice); Puigcerver et al. 2023 arXiv:2308.00951
(Soft MoE). Two annex caveats (Shazeer self-framing; dual-ascent reading) were
PRIMARY-VERIFIED 2026-07-10 (§8.4): Shazeer's own register is "hand-tuned
scaling factor" / "soft constraint approach" (the word "heuristic" appears
nowhere — quote their words, do not paraphrase); the dual-ascent paper's own
hyperparameter search over the bias step-size u ∈ {1e-4..10} directly supports
"relocates rather than removes." Switch α=0.01 + token-dropping and ST-MoE
z-loss (c_z=0.001 from a sweep) quotes re-confirmed. NOTE-MOE is cleared to
draft.

---

## 8. Post-completion review (2026-07-10, after REPORT-2 round-6 gate)

User-requested independent review of the completed program: three lane audits
(every load-bearing number re-derived from artifacts by fresh workers; code
read end-to-end) + two literature sweeps (alternatives; scoring-rule theory).
This section is the review of record; the ledger corrections it mandates are
applied in `RESULTS.md` (addendum block) and the report corrections in the
REV-1 revision of REPORT-2.

### 8.1 Verification outcome — the work STANDS, with corrections

- **Every headline number in all three lanes re-derived exactly** (D arbiter
  region means; broad-twin bound 0.0164; K6 9/9-7/9-8/9; b_coh [−0.126,+0.089];
  F3 pooled knees G r\*=3 / H r\*=2 / G-flat r\*=0 with per-increment arithmetic;
  V0 trajectory; V2 SSR + ranking; V3 knees + N=4000 increments).
- **Split hygiene clean everywhere**: K4/F2 train-vs-test disjoint seeds; K5
  arbiter pools held-out rows only; **K6 router leakage-free** (even/odd
  disjoint partition; targets from the train half only; all routed-NLL claims
  scored on the eval half); V1/V2/V3 cross-fitting leakage-free (fold models
  never score their own folds); V3 eval set disjoint from mean-fit and σ-fit.
- **Implementations correct against their sources**: masked-prefix NLL (valid
  renormalized mixture, train/read-consistent); knee (paired row-bootstrap —
  which already accounts for between-rung score correlation, see §8.3);
  NESTED depth strategy (uniform draws, no depth input, BN structurally off);
  β-NLL = Seitzer's σ^(2β) stop-grad form; MacKay α,β fixed-point = PRML.
- **Strictly-probabilistic premise honored in code**: draws are schedules,
  spread-init is an init (uses train labels only), every selection is a
  held-out proper-score operation; no penalty/λ anywhere in the lanes.
- **The V3 headline is now confirmed at three independent levels**: (i)
  empirical (knee stays v1 at N=4000); (ii) ANALYTIC — the population NLL gap
  between the optimal tercile-σ and the true smooth σ(x) is **0.00291 nat**
  (numeric integral; bin nearest x=0 carries 0.00745, the rest ≤0.00118), a
  hard ceiling matching the measured increments, so there is essentially
  nothing for the NLL knee to resolve; (iii) literature — Camporeale & Carè
  (IJUQ 2021, arXiv:2003.05103, primary-verified) construct EXACT per-residual
  NLPD ties σ̃≠σ, the published mechanism for log-score blindness to
  σ-fidelity. The log score's sensitivity to a relative σ error δ is
  second-order (≈E[δ²] nat) while σ-ratio-error is first-order (|δ|) — the
  mathematical form of "the knee is a coarse σ-selector."

### 8.2 Corrections mandated (all applied via RESULTS.md addendum + REV-1)

| # | Lane | Correction |
|---|---|---|
| W1 | WS1 | **Close R1's B-knee gate explicitly.** R1 named global-knee seed-reproducibility "decisive"; on D the nested global knee reads r\*={2,0,0} — fails at face value; R2/ledger never reconciled it. Closure: the failure corroborates the round-4 reframe (the knee under-reads component count even at the aggregate level); mechanism hypothesis — the knee walk stalls exactly at the middle rungs where b_coh costs fire (seed0 rung-3 cost → r\*=2; seeds 1/2 rung-2 cost → abstain). State plainly; the arbiter remains the certified readout. |
| W2 | WS1 | G5 locality guard was run on the knee, not the arbiter. Review filled the gap: arbiter region means move ≤0.086 nat (mean 0.028) under half-width shrink vs 0.13–0.19 effects → PASSES. Record; add G5-on-arbiter to the reader for future lanes. Also: review added a bootstrap SE to the arbiter region means — D regions 2/3 clear 2·SE on 3/3 seeds; E seed1 mid genuinely null (0±0.002), seed2 significantly inverted (−0.006±0.002). |
| W3 | WS2 | **F2 nesting cost is NOT "small/localized mirroring WS1."** Actual: 26/54 vs-fixed cells fail (48%); max −0.72 nat (~17.5σ) at depth 1 (H); the TOP rung fails 5/9 (vs-fixed) — WS1's "never at the top rung" language does not transfer. Consequence to check (X5): depth-1 interference inflates the ladder's first increment, so H's global detection (r\*=2, c1→c2 +0.64) partially conflates capacity need with baseline-rung interference — cross-check against the F1 fixed-depth sweep before calling H's detection clean. |
| W4 | WS2 | **G per-input +2.67·SE is split-fragile**: survives 3/9 re-randomized fit/score partitions (others null/negative); R3's robustness check varied only the bootstrap seed. Downgrade "MET-BUT-UNDERPOWERED, real above-control signal" → "not reproducible under split resampling — at the noise floor." Fix = X3 (repeated-split averaging). 2/18 per-bin cells at ~2.3% nominal is within chance; the `_any_pass` operationalization lowered the effective bar ~3× uncorrected. |
| W5 | WS3 | **"v2 best-or-tied σ-recovery 3/3 seeds" is false**: seed2 has v1 (0.0388) and v3 (0.0393) beating v2 (0.0424) → 2/3; at N=4000 **v3 dominates v2 on all 3 seeds** (e.g. 0.0105 vs 0.0389). Soften to: a higher-than-tercile rung recovers σ best; WHICH rung wins is itself N-dependent (σ(x) is not exactly linear-in-log-σ). The knee-is-coarse framing is unaffected. |
| W6 | WS3 | R4's per-seed "(a) beats (d): seed0; seed2 by 0.00004" is wrong — d beats a on seed2 (by 0.00046); a beats d on 1/3 seeds only. The mean-based conclusion and the NOT-A-STOP ruling stand. Also: NN honest-residual cross-fit over-estimation range is [1.64, 3.96], not "1.6–3.7". |
| W7 | X-cut | REPORT-2 §2.5 "This is exactly hierarchical stacking" is wrong: `perbin_stack` is INDEPENDENT per-bin EM with no partial pooling — pooling is Yao-2021's defining safeguard. Fix wording; partial pooling becomes upgrade X1. |
| W8 | WS1 | E-lane framing: the June NON-nested instrument recovered the hump (−0.018/+0.149/−0.026, apparently single-fit) where the nested ladder fails 2/3 seeds → the moving-mode failure is plausibly NESTING-SPECIFIC (a single global component ordering cannot serve an x-varying importance ordering), not generic per-input-count unidentifiability. Report as hypothesis + discriminating experiment X4. |
| W9 | minor | Lint: `_capacity_ladder.py` (3×E402), `capacity_ladder_k4.py` (7×E402) not ruff-clean (K6's "ruff green" claim is correctly scoped, but fix before commit). Ledger punch-list item "V2_findings order imprecision" was already fixed in V2_findings.md — retire it. |

### 8.3 Readout-machinery findings (from the scoring-rule literature sweep)

- **Knee SE machinery is sound**: the paired row-bootstrap increment SE already
  captures between-rung score covariance (the Yates-et-al. correlation-adjusted
  SE upgrade is MOOT for this implementation — verified against their full
  text). Residual caveats to carry in the G-rules: (i) Sivula–Magnusson–Vehtari
  (arXiv:2008.10296): ANY SE estimator (bootstrap included) is unreliable when
  the summed log-score gap is small (elpd-diff ≲ 4) — V3's 0.003-nat rungs live
  there; near-flat increments should be reported as "unresolvable at this N,"
  not as clean negatives. (ii) The bootstrap captures held-out sampling
  variability only, not training/seed variability — mitigated by the 3-seed
  governance, but say so where per-seed knees disagree.
- **Log score is the only LOCAL proper score** (Gneiting & Raftery 2007) — the
  principled root of the σ-blindness; CRPS/twCRPS (Gneiting & Ranjan 2011) are
  strictly proper, non-local, and shape-sensitive → the X2 upgrade.
- Verified-negative: no published critique "held-out-NLL knee for variance-class
  selection is coarse" exists — that finding is OURS (novelty for REPORT-2).

### 8.4 Literature sweep — alternatives (what we are NOT missing, and upgrades)

**The program's approach is not superseded.** No published method selects a
discrete mixture-component COUNT per-input with validated recovery (DP/
stick-breaking priors produce predictor-varying densities, never validated
count recovery — Rigon & Durante logit stick-breaking is the closest); no
published prefix-nested/Matryoshka training for REGRESSION or density
estimation exists (2024-25 Matryoshka extensions are all embeddings/retrieval)
— NOV-1's negative-existence claim STANDS after a fresh sweep. Amortized
model-comparison networks (Evidence Networks arXiv:2305.11241, BayesFlow) are
whole-dataset only; a per-input p(k|x) variant would be a novel contribution
(X7, exploratory). DirMoE (arXiv:2602.09001) needs a calibrated Dirichlet-
concentration λ — fails the no-tuned-penalty constraint.

**Verified upgrade sources**: Soloff–Guntuboyina–Sen NPMLE heteroscedastic EB
(JRSS-B 2025, arXiv:2109.03466 — pure-likelihood variance estimation, oracle
inequality, no rung ladder needed) → X6. Casa & Ferrari model-selection
confidence SETS (arXiv:2503.18790) → X8. Dendrograms of mixing measures for
softmax-gated Gaussian MoE (arXiv:2510.12744 — our exact model family;
sweep-free consistent GLOBAL k selector) → X9. MoE-contrast quotes for
NOTE-MOE primary-verified (see §7 addendum above).

### 8.5 Queued follow-up tasks (all ⏸ pending user prioritization)

**DISPOSITION (2026-07-10, user-ratified — supersedes the pending states below):** the
remaining queue is absorbed into the successor program
`docs/plans/perinput_selector_2026-07-10/EXECUTION_PLAN.md`: X10 → its P1 (trimmed
3-point ladder); X7 → gated behind its S2 evidence trigger (G-X7); X8 → executed inside
its R-INT report pass (reporting upgrade, no training); X6 and X9 → DEFERRED behind the
program (re-surface in R-INT future-work only). X1–X5 were completed 2026-07-10 (see
RESULTS.md "## X-queue follow-ups"). Execute nothing from this table directly.

| id | task | cost | note |
|---|---|---|---|
| REV-1 ✅ | REPORT-2 revised (W1–W8 + verified citations + new Table 5 nested-vs-dedicated + Δ-decomposition + δ² derivation), rebuilt, gate round 7 CLEAN. Deliverable: `/home/ff235/dev/MLResearch/automl/docs/capacity_ladder_report_2026-07-10/capacity_ladder_report.pdf` (21pp; gate log in same folder's `COLD_READ_TODO.md`). | done | 2026-07-10 |
| NOTE-MOE ✅ | authored per §5 contract with primary-verified quotes (§8.4), gate round 1 CLEAN after fixes. Deliverable: `/home/ff235/dev/MLResearch/automl/docs/moe_contrast_2026-07-10/moe_contrast.pdf` (11pp; gate log `COLD_READ_LOG.md`; figures via `make_figures.py`; verified `references.bib`). Cross-referenced from REPORT-2 §6 and vice versa. | done | 2026-07-10 |
| X1 ⏸ | Partial-pooled (hierarchical) per-bin stacking: hierarchical prior on per-bin log-weights, MAP-EM; re-run K2/F3 per-bin reads. THE targeted fix for the ~83-points-per-bin power limit; strictly probabilistic (a prior). Bars: beats independent per-bin held-out mixture log-score; G-flat still abstains. | ~1 day | highest-value new experiment |
| X2 ⏸ | CRPS-knee arm: same V3 σ-rung tables, CRPS instead of NLL (closed-form Gaussian-mixture CRPS or MC); selftest = strong-signal toy; bar = resolves v1→v2 where NLL-knee cannot, homoscedastic twin still abstains. Gives the σ lane a DEPLOYABLE fine readout (σ-ratio-error needs ground truth). | hours | high value |
| X3 ⏸ | F3 fix: average per-bin advantage over ~50 random fit/score splits (repeated cross-fit), pooled SE; re-issue the G/H per-bin verdicts. | hours | cheap; do before F4 |
| X4 ⏸ | E-lane discriminator: 3-seed NON-nested June-style instrument on the identical K4 E data. Recovers 3/3 → nesting-specific (W8 hypothesis confirmed); fragile too → moving-mode count is hard for any instrument. | hours | settles an open question |
| X5 ✅ | H/G-detection confound check — DONE in-review from F2's own `fixed_depth_ll` tables (`capacity_ladder_results/REVIEW_2026-07-10/fixed_vs_nested_knee.json`). Result: dedicated knee G pooled r\*=2 {2,2,2} coherent (nested r\*=3 was one rung HIGH — interference-differential artifact); H dedicated first increment +0.049 not +0.64 (13× inflated), per-seed {2,3,0} fragile; G-flat abstains on both. Corrected WS2 headline in RESULTS.md; REPORT-2 §4 revised accordingly. | done | the nested knee is biased by the nesting-cost gradient (under-reads count / over-reads depth) — dedicated-sweep knee or arbiter is the honest aggregate read |
| X6 ⏸ | NPMLE heteroscedastic-EB arm on V-toy1 vs the rung ladder (Soloff et al.). | ~day | optional, strengthens WS3 |
| X7 ⏸ | Per-input amortized evidence network p(k|x) prototype (Evidence-Networks loss on simulated (x,y|k)). | days | exploratory/novel |
| X8 ⏸ | Confidence-SET readout: report the plausible rung set per read (Casa & Ferrari) instead of point knee where increments are Sivula-small. | hours | reporting upgrade |
| X9 ⏸ | Dendrogram-of-mixing-measures global-k selector vs the global knee on K4 tables. | ~day | optional cross-check |
| X10 ⏸ | Power curve: N_TEST sweep (500→8k) on G per-bin read → measured sample-complexity curve for the per-input depth signal. | hours-day | turns "power-limited" into a number |

Non-goals of this review: no retraining was run; R-verdict files are historical
adjudicator artifacts and are NOT edited (corrections live here + RESULTS.md
addendum + REV-1).

### 8.6 Resumption protocol (for a fresh orchestrator session)

**State as of 2026-07-10 end-of-review.** All lanes complete + independently audited; both
reports DONE and gate-clean (paths in the §8.5 table); ledger corrected in place
(`automl_package/examples/capacity_ladder_results/RESULTS.md`); review artifact
`capacity_ladder_results/REVIEW_2026-07-10/fixed_vs_nested_knee.json`.

**⛔ USER-GATED (present, then wait):** (1) the scoped COMMIT (heterogeneous tree; capacity-ladder
set = 20 modified tracked files + `capacity_ladder_*` scripts + untracked June deps
`_toy_datasets.py`/`_variational_em.py`/`_variational_em_perinput.py` + both report folders +
plans + RESULTS.md; NEVER AI-instruction files); (2) far-future real-model ports K7/F5/V4;
(3) WS2 F4 distilled depth router (GO'd by R3, but its per-input signal was downgraded by W4 —
re-present with the review evidence before running).

**Next experiments (⏸, user prioritizes; suggested order by value/cost):** X3 (hours, unblocks
F4 decision) → X2 (hours, gives the σ lane a deployable fine readout) → X4 (hours, settles the
E-lane hypothesis) → X1 (~day, the per-input power fix) → X8/X10/X6/X9/X7 as appetite allows.
Every X-task: pre-register bars per §0b before running, artifacts to
`capacity_ladder_results/<task-id>/`, 3 seeds, escalate failed bars to a fresh-context
adjudicator. Routing per §0c (workers build/run; adjudicator interprets; heavy runs on the main
thread, detached, `AUTOML_DEVICE=cpu`).

**Report maintenance invariant:** any future result that touches a claim in REPORT-2 or NOTE-MOE
re-enters that document through the research-report skill loop (guard → build → fresh cold-read
gate); both gate logs live next to their markdown.
