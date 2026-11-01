# ProbReg real-data benchmark — frozen design specification

**Task:** capacity-programme F12-a (`docs/flexnn/archive/flexnn-core.md` §"Task F12").
**Authority:** MASTER Decision 3 as amended 2026-07-20 (`docs/flexnn/PLAN.md:79-83`),
which puts real data and baseline comparisons in scope for the ProbReg k-selection work and requires
**shared-k and variable-k to be reported as two distinct models** (Naming key, `docs/flexnn/PLAN.md:62-63`).
**Status:** SPEC — frozen before any run. Nothing in this document may be changed after the first
result lands (Decision 9: no bar is adjusted after its run).
**Written:** 2026-07-20. Every external fact below was re-verified live on that date; see §11.

**This document freezes, and only freezes:** the model set, the dataset list and its provenance, the
metrics, the split protocol, the seed counts, the matched tuning budget, the convergence gates, and
the pre-registered win/loss/null bars. It contains no driver code — the driver
(`automl_package/examples/probreg_benchmark.py`) is a separate task and must implement this
document without reinterpreting it.

---

## 1. What is reused, and what is genuinely new

Per the repo's search-before-write rule, this is the audit of existing functionality. Everything in
the "reuse" column was opened and read while writing this spec.

### 1.1 Reused unchanged

| Need | Existing artifact | Note |
|---|---|---|
| Baseline-comparison harness shape (build model → `fit` → `predict` → `predict_uncertainty` → metric dict → per-dataset DataFrame → CSV) | `automl_package/examples/model_comparison.py:174-247` | The driver **extends this shape**. `evaluate_model` (`:174-185`) is the exact contract to generalise. |
| Baseline model builders | `model_comparison.py:105-121` (`build_xgboost`, `build_lightgbm`, `build_nn_constant`) | Reused as the starting point; hyperparameters are *replaced* by the tuned search spaces of §6. |
| shared-k ProbReg builder | `model_comparison.py:159-167` (`build_prob_regression`) | Reused; `n_classes` becomes a tuned hyperparameter (§3.1). |
| Convergence gating (trajectory, `converged`/`hit_cap`/`still_improving`/`diverged`/`trustworthy`, `summary()`) | `automl_package/utils/convergence.py:36-80`, re-exported by `automl_package/examples/convergence.py` | The Decision 9/17 gate. `fit_to_convergence` at `convergence.py:153`. |
| variable-k mechanism | `ProbabilisticRegressionModel.fit_router` (`automl_package/models/probabilistic_regression.py:754`) + `predict(inference_mode="routed")` (`:598`) + `DistilledCapacityRouter` (`automl_package/models/common/distilled_router.py:106`) | Decision-13 distilled router; no new selection code. |
| K6 positive control | `automl_package/examples/capacity_ladder_k6.py` (`--selftest`, `main()` writes `k6_summary.json`) + certified artifact `automl_package/examples/capacity_ladder_results/K6/k6_summary.json` | The Decision-14 known-good arm (§8.1). |


### 1.2 Genuinely new (must be built by the driver task)

1. **Real-dataset loaders** with on-disk caching under `automl_package/examples/probreg_benchmark_data/`
   (git-ignored — raw data is never committed). No loader for any of these datasets exists in the repo today.
   `ucimlrepo` is **not installed** in `~/dev/.venv` (checked 2026-07-20); see §10.3.
2. **A matched-wall-clock tuning loop** (§6). `model_comparison.py` has no tuning at all; `BaseModel`'s
   Optuna path is trial-count-bounded, not wall-clock-bounded.
3. Removed 2026-10-22 (mixture framing; docs/probreg/cleanup_plan.md).
4. **Per-seed results JSONs** carrying trajectories and convergence flags (§9), not just means.
5. ~~**CatBoost probabilistic arm wiring**~~ — **REMOVED 2026-07-20: CatBoost is dropped from the
   model set (§2).** Retained struck-through so the removal is legible rather than silent.
   *Superseded text:* `CatBoostModel` already selects `RMSEWithUncertainty`
   when `uncertainty_method=PROBABILISTIC` (`automl_package/models/catboost_model.py:36`); the driver
   must verify the two-column output is consumed correctly (§3.5). CatBoost is absent from
   `model_comparison.py` entirely.

---

## 2. Model set (6) — frozen

> ## ⚠️ SCOPE CHANGE 2026-07-20 (user, live): TWO BASELINES
>
> **1.** The "no variance fitting, fixed shared σ" half of this scope change, with its scoring rule, σ-value rule and
> limitation note: removed 2026-10-22 (mixture framing; docs/probreg/cleanup_plan.md).
>
> **2. Baselines cut from four to two.** Three boosting libraries measured nearly the same thing.
> One well-tuned tree model plus a linear model spans the range that matters.
> - **CatBoost is DROPPED** — its only reason for inclusion was its probabilistic output mode
>   (`RMSEWithUncertainty`, §1.2 item 5).
> - **XGBoost is DROPPED** in favour of one tree learner (below).
> - **Linear regression is ADDED** — the floor. If a dataset's target is essentially linear in its
>   features, every elaborate model converging to the same score is the finding, and without a
>   linear baseline that is invisible.
>
> **Resulting set: M1, M2, M3, one gradient-boosted tree, one plain NN, linear regression = 6.**
> The neural network STAYS — a plain single-output regressor, and it is the
> single most important baseline in the set: ProbReg at k=1 is essentially that model, so it is the
> direct "does the classification bottleneck buy anything at all?" control.

**REWRITTEN 2026-07-20 (user, live). The previous two-ProbReg-arm set was CONFOUNDED and is
superseded — see §2.0.** Three ProbReg models and four baselines. **No other model may be added.**
NGBoost, MDN, Deep Ensembles, quantile NN, MC-Dropout, GP, CQR are explicit non-goals here
(`research_plan.md` §4 future work; F12 non-goals).

All six emit a per-input point prediction. That is the invariant that makes the squared-error metric
set comparable across the whole table.

### 2.0a Naming map — binding for the whole document

Older prose in this file still says "shared-k" and "variable-k" (the superseded two-arm framing).
**Read them as follows, everywhere they appear:**

| Older wording | Means now | Note |
|---|---|---|
| "shared-k" | **M1** — one k for the dataset, chosen by the arbiter | the *training* difference between M1 and M2 is gone: M1 now uses k-dropout like M2. **CORRECTED 2026-07-20**: this row previously said "like the others" (plural) — false as written. M3 is the exception: it is trained ORDINARILY per k, not with k-dropout (`docs/probreg/PLAN.md` §1). |
| "variable-k" | **M2** — a k per input, chosen by the distilled selector | unchanged in substance |
| *(no older term)* | **M3** — the per-k sweep reference | new; wherever an old passage compares only two ProbReg arms it is incomplete, not wrong |

⚠️ **The one place this map is NOT a safe substitution:** any older passage justifying a design
choice by shared-k's *lack of k-dropout* or by its tuning of `n_classes` is **superseded, not
renamed** — that arm no longer exists. §14.2 C1 and C3 record both such passages and their
resolutions. A prose rename of the remaining occurrences is a mechanical follow-up; this map is what
makes the document safe to build from in the meantime.

### 2.0 What the three ProbReg models isolate, and what the old pair got wrong

Definition of record: `docs/probreg/PLAN.md` §1. **CORRECTED 2026-07-20 (user
ruling; this section previously said "all three ProbReg models train identically — with k-dropout",
which is superseded and was in force until this correction).** **M1 and M2 train identically — with
k-dropout** (`NClassesSelectionMethod.NESTED`, per-sample `k ~ Uniform{1..k_max}`) and are read off
the SAME trained network. **M3 does NOT use k-dropout: each of its per-k models is trained
ORDINARILY at its own fixed k** (`NClassesSelectionMethod.NONE`) — a model dedicated to that k, not
the same network read at a different point, because M3 is the expensive reference the cheap methods
are measured against, not an arm in the controlled contrast. Training is therefore NOT a variable
**between M1 and M2**; they differ in exactly one thing: how k is chosen. The single-difference rule
does not bind M3.

🔑 **Each ProbReg model IS the complete system, selection machinery included** — **M1 = ProbReg +
arbiter, M2 = ProbReg + distillation, M3 = ProbReg + sweep selector.** Every metric in every table is
measured on that whole system, and every cost figure includes the selection step. The selector is
never reported as a side-analysis alongside a base model. Binding on the driver, §4's metrics and
§7's tables.

| | = ProbReg + | how k is chosen | cost |
|---|---|---|---|
| **M1** | the arbiter | ONE k for the dataset | cheap |
| **M2** | the distillation | a k per input | cheap |
| **M3** | the sweep selector | ONE k, by training a separate model per k | **expensive — the reference** |

**M1 vs M3 is the efficiency claim** (does the cheap read reach the expensive answer?). **M1 vs M2
is the per-input claim** (should k be global at all?). Two different questions; the old two-arm set
could answer neither.

**The confound this replaces.** The superseded Model 1 set
`n_classes_selection_method=NONE` — k-dropout **off** — and chose k by hyperparameter tuning. It
therefore differed from the variable-k arm in **two** respects at once, training scheme *and* k
choice, so no result could be attributed to either. §3.4 asserted the two arms "differ in exactly
one thing"; that was **false as written**. There was also no M3 at all, so the efficiency claim had
no reference to be measured against.

### 2.1 Model 1 — global-k ProbReg (one k for the dataset, chosen by the arbiter)

```python
ProbabilisticRegressionModel(
    input_size=d,
    n_classes_selection_method=NClassesSelectionMethod.NESTED,   # k-dropout, SAME as M2 (CORRECTED 2026-07-20: M3 does NOT use k-dropout — see §1)
    max_n_classes_for_probabilistic_path=10,                     # FROZEN, §3.2
    uncertainty_method=UncertaintyMethod.PROBABILISTIC,
    regression_strategy=RegressionStrategy.SEPARATE_HEADS,
    optimization_strategy=ProbabilisticRegressionOptimizationStrategy.CE_STOP_GRAD,   # owner ruling 2026-10-22
    base_classifier_params={"hidden_layers": ..., "hidden_size": ...},   # TUNED
    regression_head_params={"hidden_layers": ..., "hidden_size": ...},   # TUNED
    target_transform=<None | "symlog">,           # FROZEN PER DATASET, §5.4
    learning_rate=..., n_epochs=..., early_stopping_rounds=..., validation_fraction=...,
    random_seed=<seed>,
)
model.fit(x_fit, y_fit)
k_global = <SELECTION PROCEDURE — NOT YET SPECIFIED, see below>
```

> ✅ **RESOLVED 2026-07-20 (user). M1's selection procedure is now specified; it still has to be
> BUILT** (owned by `docs/probreg/PLAN.md` task PA).
>
> **The rule: cheapest-within-tolerance on the held-out per-rung curve, NOT argmax.** Score every
> rung `k = 1..k_max` on `cal` (and only `cal` — C1), then select the **smallest** k whose score is
> not meaningfully worse than the best rung's, where "meaningfully" is a difference exceeding
> **twice a bootstrap-estimated standard error** — the same rule already published in
> `docs/probreg/shared/reports/probreg_kselection/probreg_kselection.md` §3.2, reused so results stay comparable.
> Rung 1 (the bypass) competes. **M3 selects off its own sweep curve with the IDENTICAL rule** — if
> the two arms used different rules they would not be answering the same question, and the
> efficiency claim would be meaningless.
>
> *Why not argmax, recorded so it is not revisited:* held-out curves are noisy, so argmax
> systematically overshoots — any upward wiggle at a high rung wins by chance and the arm reports
> more resolution than the data supports. The question this work asks is "how much resolution is
> needed", which is smallest-sufficient, not highest-scoring.
>
> *Original finding, retained as the reason this block exists —* verified 2026-07-20:
> `held_out_arbiter_advantage`
> (`automl_package/models/probabilistic_regression.py:845`) returns an `(N,)` **per-input** array —
> the neighbour-averaged advantage of the top rung over the k=1 bypass. It does **not** return a
> chosen k, and no function anywhere in `automl_package/` selects a single global k from a
> k-dropout model (checked by grep for `select_k`/`choose_k`/`best_k`/`knee`/`elbow`; every hit is
> a per-input knee helper under `examples/`, and the arbiter's own docstring records that the
> per-input knee was found **not faithful**). Three things must be decided before this arm can be
> built, and they are PI decisions, not implementation details:
> 1. **What curve is read** — per-rung held-out fit term across `k = 1..10`, or the arbiter
>    statistic generalised from top-vs-bypass to rung-vs-rung?
> 2. **What rule picks k off it** — argmax, or cheapest-within-tolerance? If tolerance, what value?
>    (`docs/probreg/shared/reports/probreg_kselection/probreg_kselection.md` §3.2 uses twice a bootstrap standard
>    error as its "real difference" rule; adopting it here would be consistent but is a choice.)
> 3. **Does the bypass rung `k=1` compete?** M2's grid includes it (§2.2); if M1's does not, the
>    two arms are again not competing over the same rung set — the exact defect §14.2 C3 raised
>    about the old pair.
>
> ~~**Do not improvise this.** Escalated to the user; until it is settled, M1 is unbuildable and the
> battery cannot run.~~ *(SUPERSEDED 2026-07-21: the escalation was answered — see the ✅ RESOLVED
> 2026-07-20 banner opening this blockquote; the three questions above are settled by that ruling
> and building the mechanism is `docs/probreg/PLAN.md` task PA. This paragraph
> stood unretracted next to the banner for a day — flagged by review, missed by the first repair
> pass, struck by the second.)*

Inference, once `k_global` is settled: `predict(x)` / `predict_uncertainty(x)`, with every input
forced through the single selected rung.

### 2.2 Model 2 — variable-k ProbReg (per-input k via the distilled router)

```python
model = ProbabilisticRegressionModel(
    ...,                                          # identical to §2.1 except the two lines below
    n_classes_selection_method=NClassesSelectionMethod.NESTED,
    max_n_classes_for_probabilistic_path=10,      # FROZEN, §3.2
)
model.fit(x_fit, y_fit)
model.fit_router(x_cal, y_cal_router_space,       # see the HAZARD in §10.1
                 capacity_grid=[(1,), (2,), ..., (10,)])
y_pred = model.predict(x_test, inference_mode="routed")
y_std  = model.predict_uncertainty(x_test, inference_mode="routed")
```

`NESTED` is mandatory: `fit_router` raises unless
`n_classes_selection_method == NESTED` (`probabilistic_regression.py:802`), because routing assumes
the K4 prefix-nesting property. Rung `(1,)` is the bypass (direct-regression) mode. The router is
the package default `DistilledCapacityRouter` (hidden `(32, 32)`, 300 epochs, lr 1e-2, tolerance
0.25 — `distilled_router.py:55-58`); **these router hyperparameters are frozen and are NOT tuned**,
so that the shared-k/variable-k contrast is a contrast in *selection*, not in router search effort.

### 2.3 Model 3 — per-k sweep ProbReg (the expensive reference M1 is measured against)

Train a **separate** model for each `k` in the sweep grid (§3.1), each trained ORDINARILY at its own
fixed k (`NClassesSelectionMethod.NONE`, **not** k-dropout — **CORRECTED 2026-07-20**; this
paragraph previously said "a separate k-dropout model for each k", which is superseded, see §1),
score each on held-out data, keep the best. This is the honest, expensive way to pick one global k;
M1's entire claim is that it reaches this answer far more cheaply.

**Reuse, do not rewrite.** `select_k_for_toy`
(`automl_package/examples/report_a_benchmark.py:331`) already implements exactly this loop — fit at
each candidate k, score on held-out, `argmin` of the held-out fit term — and returns the per-k score table alongside
the selected k. The driver **generalises that function** (its ProbReg builder **stays the ordinary
fixed-k builder it already uses**, `_probreg_fixed` (`:185-191`, `NClassesSelectionMethod.NONE`) —
**CORRECTED 2026-07-20**: this sentence previously said the builder "becomes the k-dropout one of
§2.1", which is exactly the confound §1 rules out for M3 — and its `argmin` becomes whatever rule
§2.1's open question settles, so M1 and M3 are scored by the SAME selection rule on the SAME curve,
read from differently-trained networks). Writing a second sweep loop is a defect.

**Two things M3 must emit, because both halves of the efficiency claim are read off it:**
- the **selected k** (compared against M1's — the *same choice* half), and
- the **full per-k held-out score table** (compared against the k-dropout model read at each k — the
  *same quality* half, i.e. the coherence check of
  `docs/probreg/shared/reports/probreg_kselection/probreg_kselection.md` §3.2, re-run inside this harness so both
  halves come from one artifact).

**Cost note (must be stated in the report, not hidden):** M3 trains `|grid|` models where M1 and M2
train one. Its wall-clock is therefore NOT matched to the other arms and **must not** be forced into
§6's matched-budget rule — matching it would sabotage the reference. Report M3's total fit cost
explicitly; the efficiency claim is quantitative and this is the denominator.

### 2.4 Model 4 — LightGBM (the ONE gradient-boosted tree)

```python
LightGBMModel(n_estimators=..., num_leaves=..., learning_rate=..., subsample=...,   # TUNED
              early_stopping_rounds=15, validation_fraction=0.0, random_seed=<seed>)
```

**Why one tree learner and why this one.** XGBoost, LightGBM and CatBoost measure nearly the same
thing; three of them bought breadth of *library*, not breadth of *method*, and split the tuning
attention three ways. One tree model tuned properly is the stronger, more honest baseline.
LightGBM is chosen because the fairness mechanism here is a **matched wall-clock budget** (§6.4):
under a fixed time box the faster learner completes more tuning trials, so it arrives at the
comparison better tuned — and a strong baseline is what makes any ProbReg win worth reporting.
CatBoost's specific justification (a genuine heteroscedastic output) died with σ. **Reversible:**
swapping in XGBoost changes nothing structural, and the choice is recorded here rather than in code
so it can be revisited in one place.

### 2.5 Model 5 — plain PyTorch NN (the bottleneck control)

```python
PyTorchNeuralNetwork(
    input_size=d, output_size=1, hidden_layers=..., hidden_size=...,   # TUNED
    learning_rate=..., n_epochs=..., early_stopping_rounds=15,
    validation_fraction=0.0, random_seed=<seed>,
)
```

**The most important baseline in the set.** `research_plan.md` §4.1 identifies ProbReg at k=1 as
essentially this model, so it is the direct control for the question the whole architecture rests
on: **does the classification bottleneck buy anything over a plain regression head?** If ProbReg
cannot beat this, k-selection has nothing to select for.



### 2.6 Model 6 — linear regression (the floor)

Ordinary least squares, no tuning beyond standardising the features.

**Why it belongs.** It costs nothing to run and it is the only arm that can reveal a target which is
essentially linear in its features. On such a dataset every elaborate model converges to the same
score — **that convergence is the finding**, and without a linear floor in the table it is invisible
and the whole row reads as an uninformative tie. It also bounds from below: any arm that fails to
beat least squares on a dataset has not earned its complexity there.

---

## 3. Frozen model-level constants

| Constant | Value | Why |
|---|---|---|
| §3.1 **M3 sweep grid** (was: shared-k `n_classes` range) | `{1, 2, 3, 5, 7, 10}` | **Changed 2026-07-20 with the §2 rewrite.** No longer an Optuna range — no ProbReg arm tunes `n_classes` (C1). It is now M3's sweep: the k values at which a dedicated model is trained. `1` (the bypass) is ADDED so M3 competes over the same rung set M1 and M2 reach (C3); the old grid's omission of it was the confound. Still capped at 10 = `max_n_classes_for_probabilistic_path`. |
| §3.2 `max_n_classes_for_probabilistic_path` | `10` **+ a binding ceiling check** | Package default (`probabilistic_regression.py:67`), not a code limit — it is a constructor argument and the rung grid is derived from it, so nothing is hardcoded. **But a frozen ceiling censors the answer if it binds.** ⇒ **CEILING CHECK (binding, added 2026-07-20):** if any (dataset, seed) selects `k = k_max` for M1 or M3, or M2 routes a material share of inputs to the top rung, the ceiling **is** the result and the true answer is unknown. That cell is **re-run at a raised ceiling** (`k_max = 20`) and the raise is reported. Never report a selected k equal to the ceiling as if it were a free choice. *(Related hardcode worth knowing: the Optuna search space falls back to a literal `10` upper bound at `probabilistic_regression.py:464` when `n_classes_inf` is infinite. It does not bite here — no arm tunes this — but it is a real hardcode and would bite anyone who did.)* |
| §3.3 `regression_strategy` | `SEPARATE_HEADS` | The only strategy with per-class `(μ, σ)` (`predict_distribution` rejects `SINGLE_HEAD_FINAL_OUTPUT`, `probabilistic_regression.py:656-659`) and the certified configuration of record. Not swept. |
| §3.4 `optimization_strategy` | `CE_STOP_GRAD` | Owner ruling 2026-10-22: the classifier is trained on the class labels (cross-entropy on y bins) and its probabilities reach the heads without gradient. **Identical across all three ProbReg arms**, so the objective is not a second difference between the reference and the arms it is the reference for. *(The earlier `REGRESSION_ONLY` row and its reasoning: removed 2026-10-22 (mixture framing; docs/probreg/cleanup_plan.md).)* |
| §3.5 router hyperparameters | package defaults, untuned | §2.2. |
| §3.6 device | `AUTOML_DEVICE=cpu` | MASTER environment rule (`docs/RULES.md:46`). |

---

## 4. Metrics — definitions and direction

The fixed-σ mixture-likelihood metric set that stood in §4.0, §4.1 and §4.3: removed 2026-10-22 (mixture framing; docs/probreg/cleanup_plan.md).
The yardstick is held-out regression error (owner ruling 2026-10-22).

### 4.2 RMSE — point accuracy

`RMSE = sqrt(mean((y − μ̄)²))` where `μ̄(x) = Σ_c p_c(x) μ_c(x)` is the class-weighted prediction. Original target
units. **Lower is better.**

### 4.2b and 4.3 (removed metrics; the k-selection score)

Removed 2026-10-22 (mixture framing; docs/probreg/cleanup_plan.md).

### 4.4 Cost — first-class, not a footnote

The efficiency claim (M1 reaches M3's answer far more cheaply) is a **ratio**, so its denominator is
a headline quantity, not an appendix column. Recorded per (model, dataset, seed):
`wall_clock_fit_s` (for M3, the total across all `|grid|` fits), `wall_clock_predict_s`, `n_params`,
`n_trials_completed`, `selected_hyperparameters`, **and the selection cost separately from the
training cost** — the arbiter's read, the router's fit, or the sweep, as applicable. Never report an
arm's accuracy without its cost.

### 4.5 Diagnostics (descriptive, never an outcome)

- **Selected k per arm per dataset** — the object the whole strand is about.
- **Routing spread (M2 only):** fraction of test inputs sent to each rung, mean routed k, bypass
  fraction. Feeds §7.7's prediction.
- **Ceiling flag:** whether the selected k equalled `k_max` (§3.2's ceiling check).
  mean routed k, and the bypass fraction (rung `(1,)`). Descriptive; feeds the §7.7 prediction.

### 4.7 Cost columns (recorded, not scored)

`wall_clock_fit_s`, `wall_clock_predict_s`, `n_params` (neural arms), `n_trials_completed`,
`selected_hyperparameters`. Required so a reader can see what the matched budget actually bought.

---

## 5. Datasets

All provenance below was fetched live on 2026-07-20; the fetch log is §11. Figures are **as stated on
the page fetched**, not as remembered from `research_plan.md` — where the two disagree, §11 records it.

### 5.1 Tier 1 — tests our specific claims (run first)

| | **T1.1 Kepler Objects of Interest** | **T1.2 UCI Superconductivity** | **T1.3 UCI Airfoil Self-Noise** |
|---|---|---|---|
| Why in the set | Planet-radius target (rocky / sub-Neptune / Jupiter populations) | **Heavy-tailed** target — the real symlog test | **Heteroscedastic**, small, unimodal — the negative control for per-input k |
| Source | NASA Exoplanet Archive TAP service | UCI ML Repository, dataset 464 | UCI ML Repository, dataset 291 |
| URL (verified live) | `https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=...&format=csv` (table name `cumulative`) | `https://archive.ics.uci.edu/dataset/464/superconductivty+data` | `https://archive.ics.uci.edu/dataset/291/airfoil+self+noise` |
| Size (verified live) | **9564** rows (`select count(kepid) from cumulative`, executed 2026-07-20) | **21263** instances | **1503** instances |
| Features | See §5.2 for the frozen column list | **81** (`train.csv`, target in the 82nd column) | **5** (page states 5, **not** the 6 claimed in `research_plan.md` §2.2 L2.1) |
| Target | `koi_prad` — planetary radius (Earth radii) | critical temperature `T_c` (K) | `scaled-sound-pressure` (dB) |
| Licence (verified live) | No CC licence; **usage policy** is acknowledgement-based, see §5.3 | "This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license." | Same CC BY 4.0 sentence |
| Downloadable | Yes — TAP sync query returns CSV | Yes — 7.9 MB (`train.csv` + `unique_m.csv`) | Yes — 58.6 KB |
| Published reference number | none used | Hamidieh 2018 XGBoost **out-of-sample RMSE ≈ 9.5 K**, R² ≈ 0.92 (verified verbatim, §11) | none used |

### 5.2 Kepler — frozen task definition

The KOI table is a candidate list, not a curated ML dataset, so the task must be pinned here or it
will drift.

- **Table:** `cumulative` (the KOI Cumulative Delivery table; the legacy `nph-nstedAPI` interface for
  Kepler tables was discontinued in November 2023 — verified §11 — so TAP is the only route).
- **Target:** `koi_prad` (planetary radius; "the product of the planet star radius ratio and the
  stellar radius").
- **Features (frozen, 9):** `koi_period`, `koi_depth`, `koi_duration`, `koi_model_snr`, `koi_teq`,
  `koi_steff`, `koi_slogg`, `koi_smet`, `koi_srad`. All column identifiers verified live (§11).
- **Row filter (frozen):** keep rows with `koi_pdisposition IN ('CANDIDATE', 'FALSE POSITIVE')`
  **and** all 9 features **and** the target non-null. Rationale: `koi_prad` is derived in part from
  `koi_depth` and `koi_srad`, so the task is a *forward-model consistency* regression, not a
  discovery task — this must be stated plainly in the results section rather than sold as a
  scientific prediction. **The realised row count after filtering is recorded in the results JSON
  and in the report; the 9564 above is the unfiltered table size.**
- **Leakage check (mandatory, blocking):** before any model runs, the driver reports the R² of an
  ordinary least-squares fit of `koi_prad` on `[koi_depth, koi_srad]` alone. If that R² > 0.98 the
  task is a closed-form identity and the Kepler arm is **reported as such and excluded from the §7
  bars** — it does not silently become a "win". This gate exists because the confound is structural,
  not hypothetical.
- **Snapshot discipline:** the cumulative table is mutable. The driver records the query string, the
  fetch timestamp, and a SHA-256 of the returned CSV in every results JSON, and caches the CSV so
  every seed and every model sees the identical snapshot.

### 5.3 Kepler licensing and acknowledgement

The archive does not attach a CC licence. The requested acknowledgement, verified verbatim
2026-07-20 at `https://exoplanetarchive.ipac.caltech.edu/docs/acknowledge.html`:

> "This research has made use of the NASA Exoplanet Archive, which is operated by the California
> Institute of Technology, under contract with the National Aeronautics and Space Administration
> under the Exoplanet Exploration Program."

This sentence is **mandatory** in any report or paper that uses the Kepler arm. Whether the archive
formally declares the data "public domain" is **OPEN** (§12) — the page fetched does not say so in
those words. Raw KOI data is not committed to this repo in any case.

### 5.4 `target_transform` — frozen per dataset

| Dataset | `target_transform` | Reason |
|---|---|---|
| T1.2 Superconductivity | `"symlog"` | The heavy-tailed target this feature exists for. |
| T1.1 Kepler | `"symlog"` | `koi_prad` spans roughly rocky→Jupiter scales; log-distributed. |
| Everything else | `None` | Targets are on a single scale. |

`target_transform` is **frozen, not tuned** — otherwise the symlog claim becomes a search result
rather than a prediction. It applies identically to both ProbReg arms; the four baselines see the
raw target (they have no such feature), and this asymmetry is stated in the results section.

### 5.5 Tier 2 — literature comparability (Hernández-Lobato & Adams 2015 suite)

All eight UCI pages fetched live 2026-07-20 and all eight carry the identical licence sentence:
"This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license."
All eight still have working public download links and all carry `ucimlrepo` import snippets.

| Dataset | URL | Instances (page) | Features (page) | Target | PBP RMSE | PBP test LL |
|---|---|---|---|---|---|---|
| Concrete Compressive Strength | `https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength` | 1030 | 8 | concrete compressive strength | 5.667 ± 0.0933 | −3.161 ± 0.019 |
| Energy Efficiency | `https://archive.ics.uci.edu/dataset/242/energy+efficiency` | 768 | 8 | **Y1 heating load** (frozen; Y2 discarded) | 1.804 ± 0.0481 | −2.042 ± 0.019 |
| Naval Propulsion Plants | `https://archive.ics.uci.edu/dataset/316/condition+based+maintenance+of+naval+propulsion+plants` | 11,934 | 16 | **GT compressor decay coefficient** (frozen; turbine coefficient discarded) | 0.006 ± 0.0000 | 3.731 ± 0.006 |
| Combined Cycle Power Plant | `https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant` | 9,568 | 4 | PE (net hourly electrical output) | 4.124 ± 0.0345 | −2.837 ± 0.009 |
| Protein Structure (CASP) | `https://archive.ics.uci.edu/dataset/265/physicochemical+properties+of+protein+tertiary+structure` | 45,730 | 9 | RMSD | 4.732 ± 0.0130 | −2.973 ± 0.003 |
| Wine Quality **red** | `https://archive.ics.uci.edu/dataset/186/wine+quality` (`winequality-red.csv`) | **see §12 OPEN-1** | 11 | quality | 0.635 ± 0.0079 | −0.968 ± 0.014 |
| Wine Quality **white** | same page (`winequality-white.csv`) | 4898 (page headline) | 11 | quality | not in PBP | not in PBP |
| Yacht Hydrodynamics | `https://archive.ics.uci.edu/dataset/243/yacht+hydrodynamics` | 308 | 6 | residuary resistance | 1.015 ± 0.0542 | −1.634 ± 0.016 |

- **Boston Housing is excluded**, per F12 and `research_plan.md` §2.2. Verified 2026-07-20: Boston no
  longer appears anywhere on `https://scikit-learn.org/stable/datasets/toy_dataset.html`. The
  *reason* for its removal is **not** verifiable at that URL any more (§12 OPEN-2); the exclusion
  stands on the F12 contract regardless.
- **Kin8nm and California Housing are excluded** although `research_plan.md` §2.2 Layer 3 lists them:
  F12's Tier 2 list names Concrete, Energy, Naval, Power Plant, Protein/CASP, Wine red+white, Yacht
  and nothing else. Kin8nm is a Delve dataset with no UCI page to licence-verify, and California
  Housing is not in the F12 list. Adding either would be scope creep.
- **PBP numbers are reference points, not targets.** They come from a different architecture (1
  hidden layer, 50 units — 100 for Protein) trained for 40 passes. Our arms are tuned inside a
  wall-clock budget. §7.5 pre-registers that **no "we beat PBP" claim may be made**; the column
  exists so a reader can see we are in the right order of magnitude. Sign convention: PBP reports
  test *log-likelihood* (higher better).

---

## 6. Protocol

### 6.1 Splits and seeds

| Tier | Splits | Ratio | Seeds | Authority |
|---|---|---|---|---|
| Tier 1 (Kepler, Superconductivity, Airfoil) | 5 | 80/20 train/test | 0–4 | `research_plan.md` §2.3 |
| Tier 2, all except Protein | 20 | 90/10 train/test | 0–19 | HLA 2015 §5, verified verbatim (§11) |
| Tier 2, Protein/CASP | 5 | 90/10 train/test | 0–4 | HLA 2015 §5: "In the two largest data sets … we do the train-test splitting only one and five times respectively" |

Splits are drawn with `sklearn.model_selection.train_test_split(random_state=seed)`. **All six models
on a given (dataset, seed) receive byte-identical splits** — the driver builds the split once and
passes it to every arm.

### 6.2 The inner partition — and why it is the fair shared-k/variable-k contrast

The training portion is split **70 / 15 / 15** into `fit` / `val` / `cal`, once per (dataset, seed),
identically for every model.

- `fit` — trains every arm.
- `val` — early stopping and hyperparameter selection, every arm.
- `cal` — **the selection set. ALL THREE ProbReg arms select k on `cal` ONLY** (updated 2026-07-20
  with the §2 rewrite; C1). M1's arbiter reads its curve on `cal`; M2's router is fitted on `cal`;
  M3 scores its per-k sweep on `cal`. Identical data, identical size, identical question.

This is the point of the design: the three ProbReg arms see exactly the same data and spend it on
exactly the same question — *which k?* — one answering it globally by a cheap read, one per input,
one by brute force. **The equal-`cal` rule is what makes the comparison mean anything**: an arm
allowed `val ∪ cal` (30%) while another gets `cal` (15%) would be selecting on twice the evidence,
and its win would be attributable to that alone. *(§6.2 previously gave shared-k `val ∪ cal` while
variable-k got `cal`. That asymmetry is removed.)*

The four baselines use `fit` + `val` and do not touch `cal` — a deliberate, stated asymmetry (they
have no selection problem to spend it on). **Direction of that asymmetry, corrected (C2): it
FAVOURS ProbReg, it does not disadvantage it.** Every arm trains on `fit`; the ProbReg arms
additionally consume `cal` for selection while the baselines are simply denied it. The C2 fix —
baselines rank their Optuna trials on `val ∪ cal` — applies.

`cal` is drawn from the training portion only. **The test split is never touched by tuning,
early stopping, router fitting, or k selection.**

### 6.3 Preprocessing (frozen)

Features and target are z-scored using **train-portion statistics only**; predictions are
de-standardised before any metric is computed, so every number in §4 is in original target units.
Missing values: Tier 1/Tier 2 rows with any missing feature or target are dropped at load time and
the dropped count is recorded. No feature selection, no categorical encoding (all frozen datasets are
numeric).

`research_plan.md` §2.2 Layer 3 calls for "identical preprocessing to the PBP paper". PBP's exact
normalisation was **not** verified in this session (§12 OPEN-3); the scheme above is therefore
declared as **our** choice, not as a reproduction of theirs.

### 6.4 Matched tuning budget — the fairness mechanism

**The budget is wall-clock, identical for the six comparable models on a given dataset, and it covers
the whole tuning-plus-final-fit pipeline for one seed.**

> **M3 (§2.3) is EXEMPT, by design, and the exemption is the point.** M3 is the expensive reference:
> it trains one model per k, so forcing it into the same wall-clock as a single-fit arm would starve
> each of its `|grid|` fits and destroy the very thing M1 is being measured against. M3 instead gets
> the matched budget **per k-candidate fit**, so each of its fits is individually comparable to an
> M1 fit, and its **total** cost — roughly `|grid|×` — is recorded as a headline number. The
> efficiency claim is quantitative: M1's cost is one fit, M3's is `|grid|`, and the question is
> whether they reach the same k. Never report M3's accuracy without its cost.

| Dataset size | Budget `B` per (model, dataset, seed) |
|---|---|
| `n ≤ 12,000` (Airfoil, Kepler, Concrete, Energy, Naval, Power Plant, Wine ×2, Yacht) | **10 minutes CPU** |
| `n > 12,000` (Superconductivity 21,263; Protein/CASP 45,730) | **30 minutes CPU** |

Derived from `research_plan.md` §2.3 ("10 minutes on CPU for small sets, 30 on XPU for large sets").
**Deliberate amendment, recorded here rather than invented silently:** the 30-minute figure was
written for XPU, but MASTER's environment rule pins `AUTOML_DEVICE=cpu` (`docs/RULES.md:46`), so the
30 minutes is re-used as a CPU budget. This makes the large-set budget *tighter* in effective compute
than `research_plan.md` intended, which is conservative for our own models (the two ProbReg arms are
the expensive ones), not favourable to them.

Rules:
1. Within `B`, each model runs an Optuna TPE search over its own pre-declared space (§6.5), scored on
   `val`, until `B` is exhausted or the space is trivially covered. The **number of completed trials
   is recorded, not fixed** — matching budget rather than trial count is the whole point.
2. The final fit at the selected hyperparameters happens **inside** `B`. A model that cannot complete
   one full fit inside `B` reports **failure**, not a truncated number (`research_plan.md` §2.3:
   "Models that can't converge in budget report failure, not worst-case numbers"). A failure is a
   recorded outcome and propagates to §7 as a loss for that model on that dataset.
3. **Concurrency parity (binding).** Wall-clock budgets are meaningless on a shared box unless load is
   matched. All six models for a given (dataset, seed) run at the **same concurrency level**, and the
   driver records `OMP_NUM_THREADS` and the number of concurrent heavy jobs in every results JSON. Per
   `docs/RULES.md:46`, ≤4 concurrent heavy jobs, `OMP_NUM_THREADS=4`. If the concurrency level recorded
   for two arms of the same (dataset, seed) differs, that cell is **invalid** and is re-run.
4. `time.perf_counter()` around the whole per-(model, dataset, seed) pipeline is the measurement.

### 6.5 Search spaces (pre-declared; frozen)

**Updated 2026-07-20 for the §2 model set.** No ProbReg arm tunes `n_classes` — all three use the
frozen `k_max` and choose k afterwards by their own mechanism (§14.2 C1) — so their search spaces are
**identical**, which is what makes the M1/M2/M3 contrast a contrast in *selection* rather than in
search effort.

| Model | Space |
|---|---|
| M1 / M2 / M3 (identical) | classifier `hidden_layers ∈ {1,2}`, `hidden_size ∈ {32,64,128}`; head `hidden_layers ∈ {0,1}`, `hidden_size ∈ {32,64}`; `learning_rate ∈ loguniform(1e-3, 3e-2)`. **No `n_classes` dimension.** |
| LightGBM | `n_estimators ∈ [100, 1000]`, `num_leaves ∈ [15, 255]`, `learning_rate ∈ loguniform(1e-2, 3e-1)`, `feature_fraction ∈ [0.6, 1.0]` |
| PyTorch NN | `hidden_layers ∈ {1,2,3}`, `hidden_size ∈ {32,64,128}`, `learning_rate ∈ loguniform(1e-3, 3e-2)` |
| Linear regression | none — no tuning beyond feature standardisation (§2.6) |

*(XGBoost and CatBoost rows removed — both models are dropped, §2.)*

Selection criterion inside the search: the fixed-σ mixture NLL that stood here was removed 2026-10-22 (mixture framing; docs/probreg/cleanup_plan.md); to be restated on held-out regression error.

**M3's search runs ONCE, not per k.** M3 trains `|grid|` models; tuning each independently would
multiply its already-large cost and would also let it tune its way to a better k for a reason
unrelated to selection. It inherits the configuration selected by the shared search, and only `k`
varies across its fits.

### 6.6 Statistical testing — and its arithmetic limit

Wilcoxon signed-rank, paired on seeds, of each ProbReg arm's held-out error against each baseline's, and of
variable-k against shared-k. Two-sided, reported with the paired mean difference and its std.

**Binding restriction.** The exact two-sided Wilcoxon signed-rank test on `n` pairs cannot return a
p-value below `2/2ⁿ`. For `n = 5` that floor is **0.0625** — no 5-seed comparison in this benchmark
can ever reach p < 0.05, whatever the data. Therefore:

- p-values are computed and reported **only** for the 20-split Tier-2 datasets (`n = 20`, floor
  ≈ 1.9 × 10⁻⁶);
- Tier 1 and Protein/CASP (`n = 5`) are evaluated by **seed-count and effect size** (§7), never by a
  significance claim. No p-value from a 5-seed cell may appear in any table.

---

## 7. Pre-registered outcomes

Set once, here, before any run. Decision 9: **no bar is adjusted after its run.** A result that
matches none of the stated patterns is a NULL — NULL is a reportable outcome, not a prompt to
re-cut the bar.

**Cases** = 11 dataset arms: 3 Tier 1 + 8 Tier 2 (Wine red and Wine white counted separately). A case
where the Kepler leakage gate (§5.2) fires is dropped from the denominator and reported separately.

### 7.1–7.3 H1a, H1b, H2

Removed 2026-10-22 (mixture framing; docs/probreg/cleanup_plan.md). These outcomes (and the note on the 0.02-nat threshold) were stated on the fixed-σ mixture NLL, in nats.
The new review restates them on held-out regression error.

### 7.4 H3 — the bottleneck does not cost point accuracy

- **PASS:** `min(RMSE over M1, M2, M3) ≤ 1.05 × min(RMSE over the three baselines)` on
  **≥ 8 of 11** cases.
- **FAIL:** otherwise. A FAIL is a headline caveat, not a footnote: it would mean the classification
  bottleneck costs accuracy.

*(Restated for the new baseline set.)*

### 7.4b H6

Removed 2026-10-22 (mixture framing; docs/probreg/cleanup_plan.md).

### 7.5 H4 — Superconductivity, symlog, and the published reference

Hamidieh 2018's XGBoost out-of-sample RMSE is **≈ 9.5 K** (verified verbatim, §11).

- **REFERENCE MET:** shared-k ProbReg with `target_transform="symlog"` achieves test RMSE < 9.5 K on
  **≥ 3 of 5** seeds.
- **NOT MET:** otherwise.
- **Binding qualifier:** Hamidieh's split protocol was **not** verified (§12 OPEN-4). This is therefore
  a **reference point, not a head-to-head result**, and must be labelled so wherever it appears. The
  head-to-head comparison on this dataset is against our own in-run XGBoost arm.

**Also pre-registered here:** no "we beat PBP / Deep Ensembles" claim may be made from the Tier-2
table (§5.5). Those columns are descriptive context. Stated in advance so it cannot be relaxed later.

### 7.6 H5

Removed 2026-10-22 (mixture framing; docs/probreg/cleanup_plan.md).

### 7.7 Falsifiable prediction (recorded, not a bar)

Per the k-selection bypass analysis: on **Airfoil** (heteroscedastic, smooth) the
variable-k router should route the **majority** of test inputs to the bypass rung `(1,)`, whereas on
**Kepler** it should not. The rung histogram (§4.6) is reported for every dataset. This
is a prediction the benchmark can falsify; it does not gate any §7 verdict.

---

## 8. Gates

### 8.1 Decision 14 — positive control runs FIRST, ALONE, and can HALT the battery

No real-data compute is spent until both of these pass. A failure means **the protocol is the defect
under investigation**, not the datasets.

**PC-1 — the package path is sound.**
```
AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m pytest tests/test_phase3_dynamic_k.py tests/test_distilled_router.py tests/test_phase1_probabilistic_regression.py -q
```
Must be fully green. (Note: F12's stated dependency is **F9-fix** — variable-k must work on the run
device. This command is that dependency's check.)

**PC-2 — the certified K6 result reproduces.**
```
AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/capacity_ladder_k6.py --selftest
```
plus a clean-room rerun of `capacity_ladder_k6.py` whose per-case held-out NLLs are diffed against
`automl_package/examples/capacity_ladder_results/K6/k6_summary.json`. The certified bar
(`RESULTS.md:106-108`): the 2026-07-10 clean-room rerun reproduced **every per-case NLL
bit-identically, max abs diff 0.0 across 45 values**. Reproduction tolerance here: **max abs diff
≤ 1e-6** on all 45 values, on every seed the script runs.

**On failure of either: HALT.** Do not run any dataset. Report the failure as the finding.

### 8.2 Smoke gate — every arm produces a usable `(μ, σ)`

Before the battery, each of the six models is fitted once on the smallest dataset (Yacht, n=308) and
must return, of the right shape and all finite: a point prediction `μ`, and — for the three ProbReg
arms — per-class means `μ_c` and weights `p_c` summing to 1. *(Superseded: this check previously targeted CatBoost's two-column
`RMSEWithUncertainty` unpacking; that model is dropped.)* Cheap; blocking.

### 8.3 Decision 17 — the convergence gate is computed on the metric the bar reads

Bars in §7 read **RMSE** (H3, H4). Therefore:

- **Neural arms** (both ProbReg arms, the PyTorch NN): the driver records a held-out RMSE trajectory via
  `ConvergenceResult` (`automl_package/utils/convergence.py:36`) and stores `trustworthy_rmse`
  (`converged ∧ ¬hit_cap ∧ ¬still_improving ∧ ¬diverged`, `convergence.py:68-71`).
  A cell may be used for an RMSE bar only if `trustworthy_rmse`. Non-trustworthy cells are **quarantined and reported**, never silently dropped
  and never averaged in.
- **Tree arms:** the analogue is early stopping. Record `best_iteration` and
  `hit_cap = (best_iteration == n_estimators - 1)`. A tree arm that hit its `n_estimators` cap is not
  trustworthy for any bar and is re-run at a raised cap **inside the same budget `B`** (a cap raise
  that cannot fit in `B` is a §6.4 rule-2 failure).
- **Router fit:** `DistilledCapacityRouter.fit` runs a fixed 300 epochs (`distilled_router.py:57`)
  with no convergence flag. The driver records the router's final training cross-entropy and its
  label-agreement rate on `cal`, so a degenerate router (e.g. all-one-rung labels) is visible.

### 8.4 Decision 9 — no conclusion from an endpoint

The headline M1-vs-M2 comparison gets, on each of the 3 Tier-1 datasets at seed 0, one
**early-stopping-OFF confirmation run** at **≥ 4×** the self-terminated epoch count for both ProbReg
arms, with the full trajectory saved. If the confirmation run reverses the sign of the shared-k /
variable-k difference on any Tier-1 dataset, the headline verdict is **held** and escalated to the
adjudicator (unattended-run contract rule 4) rather than reported.

### 8.5 Decision 15 — protocol parity when reusing a substrate

The variable-k arm reuses the F9-ported `NESTED` training schedule that produced the K6 result. The
driver task must **diff its training loop** against `automl_package/examples/_capacity_ladder_nested.py`
(the source of the port) and justify every difference in writing before the battery runs. Known
differences to justify at minimum: real multivariate features vs 1-D synthetic inputs; z-scored
targets; early stopping on `val`; the tuned learning rate.

### 8.6 Decision 16 — optimisation is exonerated before architecture is blamed

Any arm that ends **low on both train and held-out** metrics is recorded as **under-fit — an
optimisation finding**, never as an architecture verdict, until the documented escalation ladder
(LR sweep → gradient clipping → warmup → init scheme → normalisation) has been run and logged.
Train-set metrics are therefore recorded alongside test metrics for every cell.

---

## 9. Results artifacts

One JSON per (dataset, model, seed) under
`automl_package/examples/probreg_benchmark_results/<dataset>/<model>_seed<k>.json`, each containing:

- **per-seed numbers, never only means** (F12 verify line);
- all §4.2–4.5 metrics, plus §4.7 cost columns;
- the full convergence trajectory and both flag sets (§8.3);
- the selected hyperparameters, `n_trials_completed`, and the wall-clock budget actually consumed;
- the concurrency level and `OMP_NUM_THREADS` at run time (§6.4 rule 3);
- the dataset snapshot identity: source URL, fetch timestamp, row count after filtering, SHA-256;
- `git rev-parse HEAD` at run time (`research_plan.md` §2.3 versioning rule);
- for variable-k: the rung histogram, mean routed k, and bypass fraction.

Plus one roll-up `probreg_benchmark_results/SUMMARY.md` holding the §7 bar evaluations — computed
mechanically from the JSONs, with each verdict naming the JSONs it read (MASTER `RESULT:` rule).

---

## 10. Known hazards found while writing this spec

These are properties of the code as it stands at HEAD on 2026-07-20, verified by reading it. The
driver task must handle each explicitly.

### 10.1 HAZARD — `fit_router` and `target_transform="symlog"` disagree about target space

`ProbabilisticRegressionModel.fit` symlog-transforms the targets before training
(`probabilistic_regression.py:526-529`), so the network's outputs live in symlog space. But
`fit_router` compares the caller's `y_val` directly against those outputs:

```python
y_val_arr = np.asarray(y_val, dtype=np.float64)          # probabilistic_regression.py:808
...
    return 0.5 * (log_var + (y_val_arr - mean) ** 2 / np.exp(log_var))
```

`mean` here comes from `forward_at_k`, i.e. symlog space. **Passing raw-space `y_val` to `fit_router`
on a symlog model silently produces a wrong error table and therefore a wrong router**, with no
exception raised. The docstring does not mention this.

**Binding mitigation:** on Superconductivity and Kepler (the two symlog datasets, §5.4) the driver
must pass `symlog(y_cal)` to `fit_router`, and must assert that the resulting per-rung mean errors
are on the same scale as the model's training loss. This hazard is reported upward as a package-level
docstring/API defect for a separate task — **fixing it is a non-goal of F12.**

### 10.2 CONSTRAINT — no mixture distribution under dynamic-k or symlog

Removed 2026-10-22 (mixture framing; docs/probreg/cleanup_plan.md).

### 10.3 DEPENDENCY — `ucimlrepo` is not installed

Checked 2026-07-20: `~/dev/.venv/lib/python3.12/site-packages/` contains no `ucimlrepo`. All eight
UCI dataset pages carry `ucimlrepo` import snippets and the package is live on PyPI (**version
0.0.7**, uploaded 2024-05-21). The driver task must either add it as a declared dependency
(surgical `uv pip install` per `docs/RULES.md:50-51`) or fetch the CSVs over plain HTTPS from the download
links verified in §5. **Frozen preference:** plain HTTPS + local cache, so the benchmark has one
fewer dependency and the cached snapshot is the reproducibility unit.

### 10.4 NOTE — Wilcoxon on 5 seeds is arithmetically incapable of significance

See §6.6. Recorded here too because it is the single easiest way for this benchmark to produce a
misleading table.

---

## 11. Live verification log (2026-07-20)

Everything asserted about an external source above traces to one of these fetches. Figures in
`research_plan.md` §2.2 were written 2026-04 and were **not** trusted.

| # | Source fetched | What it established |
|---|---|---|
| V1 | `https://archive.ics.uci.edu/dataset/291/airfoil+self+noise` | 1503 instances; **5** features (frequency, attack-angle, chord-length, free-stream-velocity, suction-side-displacement-thickness); target `scaled-sound-pressure`; CC BY 4.0; download 58.6 KB; `ucimlrepo` snippet present. **Corrects `research_plan.md` §2.2 L2.1, which says 6 features.** |
| V2 | `https://archive.ics.uci.edu/dataset/464/superconductivty+data` | 21263 instances; 81 features; target = critical temperature, 82nd column of `train.csv`; CC BY 4.0; 7.9 MB; two files (`train.csv`, `unique_m.csv`). |
| V3 | `https://archive.ics.uci.edu/dataset/186/wine+quality` | Page headline instance count is **4898** (white only); no combined or red-only count surfaced; CC BY 4.0; both CSVs downloadable. |
| V4 | `https://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html` | TAP is the documented route; "API support for this table was discontinued in November 2023" for the Kepler tables; the KOI Cumulative Table is "entered as `cumulative`". |
| V5 | `https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html` | TAP sync endpoint `https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=`; KOI table names incl. `cumulative`, `q1_q17_dr25_koi`. |
| V6 | `https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+count(kepid)+as+n+from+cumulative&format=csv` (executed live) | **9564** rows in the `cumulative` table as of 2026-07-20. Confirms the table is publicly queryable without authentication. |
| V7 | `https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html` | Column identifiers: `koi_prad`, `koi_period`, `koi_depth`, `koi_duration`, `koi_model_snr`, `koi_teq`, `koi_steff`, `koi_slogg`, `koi_smet`, `koi_srad`, `koi_pdisposition`, with definitions. |
| V8 | `https://exoplanetarchive.ipac.caltech.edu/docs/acknowledge.html` | The verbatim acknowledgement sentence quoted in §5.3. |
| V9 | `https://scikit-learn.org/stable/datasets/toy_dataset.html` | Boston Housing is **not mentioned at all** on the current page (six toy datasets listed, scikit-learn 1.9.0). |
| V10 | UCI pages for datasets 165, 242, 316, 294, 265, 186, 243 | Instances / features / targets / CC BY 4.0 / download links as tabulated in §5.5. |
| V11 | `https://pypi.org/pypi/ucimlrepo/json` | `ucimlrepo` version **0.0.7**, uploaded 2024-05-21. |
| V12 | `https://arxiv.org/pdf/1502.05336` (PBP, HLA 2015) | §5 protocol verbatim: "The data sets are split into random training and test sets with 90% and 10% of the data, respectively. This splitting process is repeated 20 times… In the two largest data sets, *Year Prediction MSD* and *Protein Structure*, we do the train-test splitting only one and five times respectively." Architecture: one hidden layer, 50 units (100 for Protein/Year), 40 passes. Table 1 RMSE/LL as tabulated in §5.5. |
| V13 | `https://arxiv.org/pdf/1612.01474` (Deep Ensembles, 2017) | §3.3 verbatim confirms the identical 20/5/1-fold, 50/100-unit, 40-epoch protocol. Table 1 values available if needed. |
| V14 | `https://arxiv.org/pdf/1803.10260` (Hamidieh 2018 preprint) | p.3 verbatim: "Our XGBoost model gives reasonable predictions: an out-of-sample error of about 9.5 K based on root-mean-squared-error (rmse), and an out-of-sample R2 values of about 0.92." p.2: "21,263 superconductors are used", "we define and extract 81 features". **Confirms the ≈9.5 K claim and n/d.** |

**Transcription caveat.** V12 and V13 numeric tables were read from rendered PDF pages rather than
machine-extracted text. Protocol *sentences* are quoted verbatim and are solid; individual table
digits carry small transcription risk. Since §7.5 forbids any "we beat PBP" claim, no §7 bar depends
on a V12/V13 digit — they are context only. Any single PBP/DE number that later becomes load-bearing
must be re-verified first.

---

## 12. OPEN — not verified, not to be asserted

1. **OPEN-1 — Wine Quality red instance count.** The UCI page (V3) surfaces only "4898" (white). The
   red count of **1599** appears in PBP Table 1 (V12) as `N` for "Wine Quality Red", and the repo's
   `research_plan.md` §2.2 states 1599/4898 — but the UCI page itself was not confirmed to state it.
   The driver must record the **actual row count of `winequality-red.csv` at load time** and that
   number, not this one, goes in the report.
2. **OPEN-2 — reason for Boston's removal from scikit-learn.** `research_plan.md` §2.2 cites
   `https://scikit-learn.org/stable/datasets/toy_dataset.html` for the deprecation-in-1.0 /
   removal-in-1.2 / ethical rationale. That page no longer mentions Boston at all (V9), and the v1.2
   changelog fetched in this session contains no `load_boston` entry. The **exclusion** of Boston is
   contractual (F12) and stands; the **stated reason** is currently uncited and must not be repeated
   as fact without a fresh source.
3. **OPEN-3 — PBP's exact preprocessing.** Not extracted from the paper. §6.3's normalisation is
   declared as our own choice, not as a reproduction.
4. **OPEN-4 — Hamidieh 2018's split protocol.** The ≈9.5 K RMSE is verified (V14) but the
   train/test scheme that produced it is not. §7.5 is therefore a reference point, not a head-to-head.
5. **OPEN-5 — Kepler data-use terms.** The acknowledgement text is verified (V8); an explicit
   public-domain / open-licence statement was not found. No KOI data is committed to this repo, and
   the acknowledgement is mandatory in any output.
6. **OPEN-6 — Naval Propulsion target column labels.** The UCI page's Variables table was truncated in
   the fetch; the two decay coefficients are confirmed to exist but their exact on-page Role labels
   were not quoted verbatim. §5.5 freezes the **GT compressor** coefficient as the target (the PBP
   convention); the driver must confirm the column index at load time.
7. **OPEN-7 — Combined Cycle Power Plant feature count.** "4" is consistent with the four ambient
   inputs (T, AP, RH, V) with PE as target and matches PBP Table 1's `d=4`, but was not read off the
   page's own Features field verbatim.
8. **OPEN-8 — realised Kepler row count after the §5.2 filter.** Unknown until fetch; 9564 is the
   unfiltered table size (V6). Recorded at run time.

---

## 13. Non-goals (restated so they cannot drift)

- No driver code in this document; `automl_package/examples/probreg_benchmark.py` is a separate task.
- No experiments run and no datasets downloaded while authoring this spec.
- No NGBoost, MDN, Deep Ensembles, quantile NN, MC-Dropout, GP, or CQR baselines.
- No astrophysics headline analysis (Paper A/B owns that).
- No HPO beyond the frozen matched budget of §6.4.
- No width/depth content (F7 owns that).
- No fix to the §10.1 symlog/`fit_router` hazard — reported, not repaired, here.

---

## 14. Review verdict

| Field | Value |
|---|---|
| Verdict | **SOUND-WITH-FIXES → GO for the driver build**, with one **PARK** gate on the Kepler arm only |
| Reviewer | F12-a adjudication pass |
| Date | 2026-07-20 |
| Conditions | C1–C12 below are binding and must land in this document **before the battery runs**. PARK-1 must be settled by the user before **any Kepler compute** is spent; it does not block the driver build or the other ten cases. |

Outcomes H1a, H1b, H2, H5 and H6, cited in the conditions below, were removed from §7 on 2026-10-22 (mixture framing;
docs/probreg/cleanup_plan.md); the conditions are kept as the review record.

Scope of this review: the design only. No experiment was run. Every ruling below was re-derived from
the cited artifact at HEAD on 2026-07-20; where the spec's own citation was checked and found wrong,
that is stated as such.

### 14.1 What is sound

- **The frozen-before-run discipline is real.** Model set, dataset list, metrics, splits, seeds,
  budget and bars are all fixed here, ahead of compute, with WIN/LOSS/NULL stated per claim. This
  satisfies Decision 9's pre-registration requirement.
- **The Wilcoxon arithmetic is exact and correct (§6.6, §10.4).** Re-derived with
  `scipy.stats.wilcoxon(..., method="exact")`: the two-sided floor is **0.0625 at n=5** (`= 2/2⁵`)
  and **1.9073486328125e-06 at n=20**. Confining p-values to the 20-split Tier-2 sets is correct and
  is applied consistently — §7.1, §7.3, §7.4, §7.5 and §7.6 all read counts or effect sizes, and the
  only bar citing significance (§7.2 H1b) restricts it to Tier 2 in the bar text itself. Protein/CASP
  is correctly named as an n=5 exception despite being Tier 2. **CONFIRMED, no fix needed** beyond C10.
- **Tier-2 scope is contractually correct (§5.5).** F12 (`docs/flexnn/archive/flexnn-core.md:513-514`) names exactly
  Concrete, Energy, Naval, Power Plant, Protein/CASP, Wine red+white, Yacht. Excluding Kin8nm and
  California Housing is adherence, not under-scoping. *Caveat (C11d):* §5.5 titles this "the
  Hernández-Lobato & Adams 2015 suite" while omitting datasets that suite contains; full Table-1
  membership was **not** re-verified in this review, so the report must say "a subset of" unless it is.
- **The wall-clock budget is the right choice among the available options (§6.4).** It is not a
  compute-matched comparison and must never be described as one — but matched-trial-count would be
  strictly worse (one ProbReg trial costs orders of magnitude more than one LightGBM trial, so equal
  trials would hand ProbReg far more compute and make any ProbReg win uninterpretable), and matched
  FLOPs is not definable across trees and nets. It also reuses the repo's own rule verbatim
  (`research_plan.md:352`: "10 minutes on CPU for small sets, 30 on XPU for large sets… Models that
  can't converge in budget report failure, not worst-case numbers"). The direction of its bias — trees
  complete far more trials per minute on CPU — runs **against** the ProbReg arms, which is conservative.
  Concurrency parity (rule 3) is a genuine and well-specified safeguard. **CONFIRMED as defensible**,
  subject to C8.
- **The positive-control, convergence and protocol-parity gates (§8.1–§8.6)** map correctly onto
  MASTER Decisions 14, 17, 15, 9 and 16, and §8.3's split into `trustworthy_nll` / `trustworthy_rmse`
  is the correct reading of Decision 17. All cited code exists at the cited lines
  (`convergence.py:36-80`, `:68-71`, `:153`; `distilled_router.py:55-58`, `:106`;
  `probabilistic_regression.py:526-529`, `:598`, `:656-668`, `:754`, `:800-808`;
  `catboost_model.py:36`; `neural_network.py:34`; `scoring.py:18,40,119,158`;
  `_capacity_ladder_nested.py`, `tests/test_distilled_router.py`, `capacity_ladder_k6.py` all present).
- **§10.1's symlog/`fit_router` hazard is real** and correctly diagnosed: `fit()` symlog-transforms
  `y_train`/`y_val` at `probabilistic_regression.py:526-529`, while `fit_router`'s `eval_fn` compares
  the caller's raw `y_val` (`:808`) against `forward_at_k` output that lives in symlog space. Silent,
  no exception. Correctly declared a non-goal here.

### 14.2 Conditions (binding, mechanical)

**C1 — RESTATED for the §2 rewrite (2026-07-20). The original contradiction is dissolved; a
narrower one replaces it.** C1 originally resolved a clash over where the old shared-k arm's
`n_classes` was selected. Under §2.0 **no ProbReg arm tunes `n_classes` through Optuna at all** —
`k_max` is frozen at 10 for all three and k is chosen by each arm's own mechanism afterwards. So the
Optuna search space is now *identical* for M1, M2 and M3 (the non-k hyperparameters only, scored on
`val`), which is what the original fairness argument wanted and could not express.

What survives, and must still be settled: **which split each arm's k-selection reads.** M2's router
is restricted to `cal` and that is intrinsic (its per-rung error table must not be read on the set
used for early stopping). **M1 and M3 must be given the SAME restriction — `cal` only — or the
comparison is rigged**: an M1 that reads `val ∪ cal` while M2 reads `cal` selects on twice the data,
and any M1 win would be attributable to that alone. Binding: **all three ProbReg arms select k on
`cal` only.** Amend §6.2/§6.5 accordingly.

**C2 — §6.2's stated direction of the baseline asymmetry is wrong, and it favours ProbReg.** The
spec says the baselines' not touching `cal` "can only disadvantage the ProbReg arms". The opposite is
true: all six arms train on `fit` (70%), but the **three** ProbReg arms additionally *consume*
`cal` for
selection while the four baselines are simply denied it. ProbReg therefore selects on 30% held-out
where the baselines select on 15%, and no arm gains training data from the difference. This advantages
ProbReg on **H2, H3 and H5** — three of the six bars. Fix (mechanical): give the four baselines the
same selection budget — they train on `fit`, early-stop on `val`, and their Optuna trial ranking is
scored on **`val ∪ cal`**. Then every arm trains on `fit` and selects on `val ∪ cal`, and the only
residual asymmetry is the variable-k router's `cal`-only restriction, which is intrinsic (the router's
per-rung error table must not be read on the set used for early stopping). Correct the sentence's sign
either way.

**C3 — ✅ RESOLVED by the §2 rewrite (2026-07-20), and it must not be reopened.** The defect was
structural: the old shared-k arm used `NClassesSelectionMethod.NONE`, and the bypass head is built
**only** on the dynamic branch (`automl_package/models/architectures/probabilistic_regression_net.py:84`,
with the `None` assignment on the non-dynamic `else` branch at `:96` — verified 2026-07-20), so that
arm could not select the bypass at all while variable-k could route every input to it.

**⚠️ CORRECTED 2026-07-20 (second pass), after the M3 training ruling.** This paragraph previously
read: *"Under §2.0 all three ProbReg arms use `NESTED`, so **all three build the bypass head and
compete over the identical rung set `1..10`**."* **The first clause is false under §1 as ruled.**
M3 does NOT use `NESTED` — each of its per-k models is ordinary (`NClassesSelectionMethod.NONE`),
which takes the **non-dynamic** branch, where `direct_regression_head` is set to `None`
(`automl_package/models/architectures/probabilistic_regression_net.py:93-96`; the head is built only
on the dynamic branch at `:84-92` — both re-verified on disk 2026-07-20). **M3 therefore builds no
bypass head at all.**

**What survives, and what does not:**
- **The practical claim survives, by a DIFFERENT mechanism.** M3 still competes over the same rung
  set because `1` is an explicit member of its sweep grid (§3.1): M3 reaches the bypass by training
  a **dedicated `n_classes=1` model**, not by routing to a bypass head. The confound C3 names — one
  arm structurally unable to select direct regression — is still gone.
- **The equivalence is NOT established, and is recorded here as OPEN rather than assumed.** A
  dedicated `n_classes=1` model and the `direct_regression_head` are *architecturally different
  objects*: the former passes through the classification bottleneck with a single class, the latter
  is a separate MLP. They are expected to behave similarly and neither is obviously favoured, but
  nothing in this repo has measured it. **Do not present M3's k=1 rung as identical to M1/M2's
  bypass rung** — report it as the dedicated k=1 model it is. If a headline result turns on the
  bypass comparison specifically, this gap must be closed first.

**Standing condition (unchanged):** M1's rung set, once §2.1's open selection question is settled,
MUST be `1..10` including the bypass. Any narrower grid re-creates this exact defect.

*Original finding, retained as the reason the condition exists:* §3.1 fixed shared-k's grid at
`{2,3,5,7,10}`; §2.2 fixes
variable-k's `capacity_grid` at `[(1,), …, (10,)]`, where `(1,)` is the direct-regression bypass. This
is structural in the code, not a spec choice: `forward_at_k` raises for `k=1` unless a
`direct_regression_head` was built, and that head exists **only** under a dynamic
`n_classes_selection_method` (`probabilistic_regression_net.py:185-190`). So the shared-k arm cannot
select the bypass **at all**, while variable-k can route every input to it. A variable-k win is
therefore consistent with "the bypass was available", not "per-input k pays" — and §7.7 predicts
exactly that on Airfoil. This is the confound already on record for this line of work
(k-selection bypass confound). No listed bar catches it. Fix (mechanical, uses only data §4.6 already
collects): pre-register in §7.2 that **on any case where variable-k wins and its bypass fraction
exceeds 0.5, the win is additionally checked against the heteroscedastic PyTorch NN baseline (Model 7);
if the NN matches variable-k within 0.02 nat, the case is reported as "bypass availability", not as
per-input k, and is excluded from the H1b WIN count.** Also correct §3.1's rationale, which claims
`{2,3,5,7,10}` is "exactly the set of rungs the variable-k router can reach" — it is not.

**C4 — the router's selection set is starved on the small cases, and no bar notices.** `cal` is 15% of
the training portion: 13.5% of `n` at the Tier-2 90/10 split, 12% at the Tier-1 80/20 split. Realised
sizes: **Yacht ≈ 42**, Energy ≈ 104, Concrete ≈ 139, Airfoil ≈ 180, Wine red ≈ 216, Wine white ≈ 661,
Kepler ≲ 1148, Power Plant ≈ 1292, Naval ≈ 1611, Superconductivity ≈ 2552, Protein ≈ 6174. A
`(32, 32)` MLP over 10 rungs, trained 300 epochs with no early stopping and no convergence flag
(§8.3), fit on 42 labelled points is not a router — it is noise. Three of eleven cases sit at or below
~140 points, and §7.2's H1b WIN additionally requires that **no** case is worse by ≥ 0.02 nat, so a
single starved-router loss on Yacht can block an H1b WIN on its own. Fix (mechanical): pre-register a
**selection-data floor** — cases with `|cal| < 150` are flagged, H1a and H1b are reported **both with
and without** the flagged cases, and a variable-k loss on a flagged case is recorded as a
**selection-data-starvation** finding, never as an architecture verdict (the Decision-16 pattern
applied to the router). §8.3 already records the router's training CE and `cal` label-agreement;
require both to be printed next to every flagged case.

**C5 —** Removed 2026-10-22 (mixture framing; docs/probreg/cleanup_plan.md).

**C6 — the symlog asymmetry confounds H2 on exactly the two cases where it is largest, and no bar
isolates it.** §5.4 freezes `target_transform="symlog"` for the two ProbReg arms on Superconductivity
and Kepler; the four baselines see the raw target because they have no such feature. Heavy-tailed
targets are precisely where a raw-target Gaussian head or tree does worst, so on those two cases the
ProbReg advantage read by **H2** may be entirely the target transform and have nothing to do with the
classification bottleneck. §5.4 states the asymmetry exists but registers no consequence for any bar.
Fix (mechanical, free — no extra compute): pre-register that **H2 is evaluated twice, on all 11 cases
and on the 9 non-symlog cases (bar ≥ 6 of 9 to hold the same proportion); if the two verdicts differ,
the 9-case verdict is the reported one and the difference is attributed to the target transform.**

**C7 — the Kepler leakage gate (§5.2) is specified in the wrong functional form and will not fire.**
The gate is an **ordinary least-squares** fit of `koi_prad` on `[koi_depth, koi_srad]`, thresholded at
R² > 0.98. But the derivation the gate exists to catch is multiplicative and involves a square root:
transit depth δ ≈ (R_p/R_*)², so `koi_prad ∝ koi_srad · √koi_depth`. A **linear** OLS on
`(koi_depth, koi_srad)` cannot represent that and will report a modest R² even when the identity is
exact — i.e. the gate as written is close to a guaranteed pass, and Kepler then enters every bar as a
legitimate case. Fix (mechanical): run the gate as an OLS of **`log(koi_prad)` on
`[log(koi_depth), log(koi_srad)]`**, and additionally report the fitted coefficients (the leakage
signature is ≈ 0.5 and ≈ 1.0). Keep 0.98 — the threshold is not what is wrong here; once the form is
right the R² will be decisive in either direction. **The remedy branch is PARK-1, not this condition.**

**C8 — the trial-count disparity under a wall-clock budget is unread by any bar.** §6.4 rule 1
deliberately records rather than fixes `n_trials_completed`, which is correct, but no §7 reading rule
consumes it. On CPU with `OMP_NUM_THREADS=4` the tree arms will complete one to three orders of
magnitude more Optuna trials than the ProbReg arms inside the same `B`; on the 30-minute sets the
ProbReg arms may complete a handful. An **H2 LOSS, H3 FAIL or H1b LOSS** under those conditions is a
budget finding, not an architecture finding — the same logic as Decision 16. Fix (mechanical):
(a) pre-register that any ProbReg-negative verdict on a case where the ProbReg arm completed fewer
than 25% of the median baseline trial count on that case is reported as **budget-limited**, with the
trial counts printed beside it; (b) record **CPU time** (`time.process_time()` or
`resource.getrusage`) alongside `time.perf_counter()` in every results JSON, so contention on the
shared box is detectable after the fact rather than only assumed away by rule 3.

**C9 — H4 (§7.5) is a §7 outcome that rests on OPEN-4 and must be demoted.** OPEN-4 records that
Hamidieh 2018's train/test protocol is unverified. A test-RMSE threshold of 9.5 K is not comparable to
our 80/20 test RMSE without knowing what split produced it, so "REFERENCE MET / NOT MET" is not a
falsifiable statement about our models. The spec's own binding qualifier says as much. Per the F12
review contract (anything load-bearing either resolves before the run or its bar is dropped): **move
H4 out of §7 into a recorded context line** — "shared-k symlog test RMSE on Superconductivity,
reported beside Hamidieh's ≈ 9.5 K under an unverified split" — with no WIN/LOSS language. The
head-to-head on that dataset stays where §7.5 already puts it: against our own in-run XGBoost arm,
under H2.

**C10 — §7.2's "at least one Tier-2 case with p < 0.05" is an uncorrected max-statistic over 8
datasets.** Family-wise error for eight independent tests at α = 0.05 is ≈ 34%. The requirement is an
AND-clause on top of an effect-size count, so the inflation is bounded, but it is unstated. Fix
(mechanical): require the p < 0.05 to survive **Holm–Bonferroni across the eight Tier-2 datasets**, or
label it explicitly as an uncorrected supporting statistic. Also note in §7.2 that because the
significance clause is Tier-2-only, **an effect confined to Tier 1 can never produce an H1b WIN** —
so that a NULL is not misread as "no effect".

**C11 — stale or inaccurate reuse rationales (each is a one-line correction).**
- (a) §4.5 and §1.1 state that `metrics.py` `calculate_ece` "is the non-standard N3 formulation".
  **Refuted at HEAD:** the implementation now computes PIT values via `norm.cdf` and returns
  `mean|observed_coverage − target_level|` over `linspace(1/(n+1), n/(n+1), n)`, with a docstring
  citing Kuleshov, Fenner & Ermon 2018. The formulation described as N3 in `research_plan.md:85-95`
  (two-sided p-values plus a fixed `z ≤ 1` rule) is **not** what is in the file; MASTER's History
  section already flags N3 as "appears fixed at HEAD". Writing a fresh function is still justified —
  the spec needs the 21-level `j/20` grid, the KS statistic and the raw PIT histogram, none of which
  `calculate_ece` returns — but state that as the reason.
- (b) §4.4 and §1.1 state that `calculate_picp` "hardcodes `z_score=1.96`". It takes
  `z_score: float = 1.96` as a **parameter with a default** (`metrics.py:163`), so multi-α PICP is
  reachable through it. The real reason to write a new one is MPIW, which it does not compute.
- (c) §3.1's rationale — `{2,3,5,7,10}` is "exactly the set of rungs the variable-k router can reach"
  — is false; see C3.
- (d) §5.5's "Hernández-Lobato & Adams 2015 suite" label; see §14.1.
- (e) §6.4 cites `research_plan.md` §2.3 for the budget; the quoted sentence is at
  `research_plan.md:352`, which falls under **§3**, matching F12's own citation ("`research_plan.md`
  §3 fixes a per-model-per-dataset wall-clock budget"). Fix the section number.

Minor, no action required: §4.5's ECE grid `q_j = j/20, j = 0…20` faithfully reproduces
`research_plan.md` §3.3's `ece_regression`, but two of its 21 terms (`q=0` and `q=1`) are structurally
zero, so the reported ECE is deflated by a constant factor of 19/21 relative to an interior-only grid.
It is a fixed monotone rescaling and does not affect any comparison; it is recorded here only so the
number is not read against a literature ECE computed on a different grid.

**C12 — §10.1's mitigation has no acceptance criterion and will pass unconditionally.** "Must assert
that the resulting per-rung mean errors are on the same scale as the model's training loss" is not a
check a driver can fail. Fix (mechanical): give it a number. Minimum acceptable form — on each symlog
dataset the driver builds the per-rung error table **twice**, once from raw `y_cal` and once from
`symlog(y_cal)`, asserts the two tables differ materially, and asserts the symlog-space table's
minimum agrees with the model's final `val` loss at the corresponding k within a pre-registered
tolerance. The tolerance is the driver task's to pick, but it must be written down **before** the run.

### 14.3 PARK — the one item the user must decide (Kepler only)

**PARK-1 — Kepler.** The multimodality argument that stood here (moment-matched Gaussian versus mixture readout) and its options 2 and 3: removed 2026-10-22 (mixture framing; docs/probreg/cleanup_plan.md).

C7 shows the leakage gate as written will very likely not fire,
and if it is corrected to log space it will very likely fire hard, because `koi_prad` is a near-exact
closed-form function of two of the nine frozen features. 
The user must choose one of:
1. **Change Kepler's feature set** — drop `koi_depth` and `koi_srad`, predicting `koi_prad` from the
   remaining seven. This turns a forward-model consistency check into a genuine prediction task and
   makes the leakage gate moot. Costs: the frozen §5.2 feature list changes, and the task gets harder.
2. and 3. Removed 2026-10-22 (mixture framing; docs/probreg/cleanup_plan.md).
4. **Drop the Kepler arm from F12.** Costs: 10 cases instead of 11; every §7 denominator changes.

Option 1 changes frozen sections; 4 changes scope. Both are principal-investigator calls, which is why this is PARKed rather than written as a condition.

**Gate:** the driver build proceeds now. The ten non-Kepler cases proceed once C1–C12 land. **No
Kepler compute until PARK-1 is answered**, and whichever option is chosen is written into §5.2 / §5.4
before that arm runs.

### 14.4 Coverage statement

This review verified the spec's internal consistency and its external citations; it cannot certify
that no further defect exists. Verification restores precision, not recall. The gaps found that **no
listed bar would have caught** are C3 (bypass availability read as per-input k), C4 (router
starvation read as an architecture verdict), C6 (target transform read as a bottleneck advantage),
and C8 (tuning budget read as an architecture verdict) — four, in a document that is otherwise unusually careful. That density
suggests the bar-reading rules, specifically, would repay one more adversarial pass once C1–C12 are
written in; the dataset, metric and protocol sections did not show the same pattern.
