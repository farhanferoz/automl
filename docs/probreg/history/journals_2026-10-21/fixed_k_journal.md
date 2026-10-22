> Superseded framing (2026-10-22): text below that treats ProbReg as a mixture model (fixed-σ mixture loss/readout, regression-only supervision) is withdrawn; see the banner in docs/probreg/PLAN.md.

# Fixed-k ProbReg — journal

A factual record of the fixed-k variant: what it is, how it is built, what has been tried in
training it, what the toy results show and under which caveats, and what is already planned.
Assembled 2026-10-21 from the plans, the results on disk and the code at commit `5fc982e`. It
records facts and the conclusions the sources themselves drew, each attributed; it gives no
verdicts of its own. The plan of record is `docs/probreg/fixed_k/plan.md` (the structure study)
under `docs/probreg/PLAN.md`; this journal does not replace either.

Paths are relative to the repository root (`/home/ff235/dev/MLResearch/automl` on the DGX,
`/mnt/dgx/home/ff235/dev/MLResearch/automl` from the laptop). Line numbers are as of `5fc982e`;
`docs/probreg/review/groundwork/function_index.md` (local-only) re-locates any function by name. <!-- citecheck-ignore: local-only review pack -->

**Labels used below.** Each result carries the labels that apply to it:
- **LEARNED-σ** — trained with a fitted per-class variance. Decision 26 (`docs/probreg/PLAN.md`,
  end) made fixed σ the model definition on 2026-07-21; the owner's 2026-10-20 ruling Q2 softened
  that to "a design option — start with it fixed". Results trained with learned σ are labelled
  old-objective in the plans.
- **LEGACY TOYS** — the four `make_datasets()` toys (heteroscedastic, bimodal, piecewise,
  exponential) in `automl_package/examples/_toy_datasets.py`, which `docs/probreg/shared/toy_suite.md`
  says are "not a tier"; no new cell may be produced on them.
- **SELECTED ON LEARNED σ** — k or an arm was chosen by a likelihood read at a fitted variance,
  which Decision 24 forbids for capacity readouts.
- **PROVISIONAL** — the source itself marks the result as not final (a tie-break winner, a halt
  awaiting a ruling).

---

## 1. What the variant is

ProbReg is a regression model built as a classifier and trained end to end as one network. A
classifier reads the input and gives a probability for each of k classes; each class has its
own regression head; the heads are combined by class probability into a k-component predictive
distribution. In the fixed-k variant **k is set by the user** and is not chosen by the model
(`docs/probreg/PLAN.md`, "The model, in one paragraph" and the variants table). It is the
substrate the other two variants are built on: global-k and per-input-k change only how k is
chosen (`docs/probreg/history/journals_2026-10-21/global_k_journal.md`, `docs/probreg/history/journals_2026-10-21/per_input_k_journal.md`).

What the owner requires of it (rulings of 2026-10-20, `docs/probreg/PLAN.md` rulings table and
objectives O1a–O1c):
- **O1a** — settle what the model is (classifier supervision, spread, likelihood form,
  identifiability constraints, head layout) by pre-registered experiment and certify one
  configuration. This is the structure study, `docs/probreg/fixed_k/plan.md`.
- **O1b** — the classification bottleneck must **improve** point accuracy over a plain network of
  matched capacity (ruling Q3), not merely not hurt it.
- **O1c** — a better predictive distribution than standard regressors where the data has
  structure (multimodal, heteroscedastic). Calibration is a later objective (ruling Q1: k first).
- σ is fixed for the current phase (ruling Q2); real data is out of scope (ruling Q5).

## 2. Architecture as built

### 2.1 Components

| Part | What it is | Where |
|---|---|---|
| Model wrapper | `ProbabilisticRegressionModel`; fit/predict, losses, selectors | `automl_package/models/probabilistic_regression.py:64` |
| Network | `ProbabilisticRegressionNet`: classifier + regression module + (for dynamic k only) a direct-regression head | `automl_package/models/architectures/probabilistic_regression_net.py:15` |
| Classifier | a feed-forward network giving `n_classes` logits, softmaxed to class probabilities | `automl_package/models/architectures/probabilistic_regression_net.py:99` onward |
| Heads, default layout | `SeparateHeadsRegressionModule`: one head per class; head i is built with `input_size=1` and is fed **only its own class probability** p_i, never x | `automl_package/models/common/regression_heads.py:286`, `:350`, `forward` at `:369` |
| Heads, alternatives | `SingleHeadNOutputsRegressionModule` (one net reads the whole probability vector, emits every class's output); `SingleHeadFinalOutputRegressionModule` (maps probabilities straight to one prediction; no per-class components) | `automl_package/models/common/regression_heads.py:406`, `:491` |
| Direct-regression ("bypass") head | built only when `n_classes_selection_method` is not `NONE`; a fixed-k model has none | `automl_package/models/architectures/probabilistic_regression_net.py:66-97` |

With `RegressionStrategy.SEPARATE_HEADS` the prediction is a probability-weighted combination of
the heads' outputs. Whether it is scored as one Gaussian (collapsed by the law of total variance)
or as a k-component mixture depends on the loss (§3.1).

The April 2026 statistical critique of this design described heads that depend on p_i only as a
"strong assumption" (a conditional-independence restriction) and, "if redesigning from scratch",
proposed letting heads depend on x
(`docs/probreg/history/identifiability_2026-04/probreg_identifiability_research.md:1361-1407`,
§13.7–13.8). No head that reads x has been built in the library; the structure study deferred
"head input re-presentation (feeding features beyond p_i to heads)" as a proposal, not an arm
(`docs/probreg/fixed_k/plan.md:289-294`). The July surrogate used for the k-selection
experiments (`automl_package/examples/_capacity_ladder_nested.py:38`) does read x in its heads;
see `docs/probreg/history/journals_2026-10-21/per_input_k_journal.md` §2.4.

### 2.2 Defaults at HEAD

From `ProbabilisticRegressionModel._defaults`
(`automl_package/models/probabilistic_regression.py:95-151`):

| Setting | Default | Meaning |
|---|---|---|
| `n_classes` | 3 | k |
| `regression_strategy` | `SEPARATE_HEADS` | head layout |
| `optimization_strategy` | `REGRESSION_ONLY` | classifier trained only through the regression loss |
| `prob_reg_loss_type` | `GAUSSIAN_LTV` | collapsed Gaussian (law of total variance) |
| `head_spread` | `PER_INPUT` | each head emits a log-variance as a function of p_i |
| `fixed_sigma_train` | `None` | no fixed σ unless the fixed-σ mixture loss is chosen |
| `ordering_constraint_weight` | `None` → auto | resolves to 1.0 only for SEPARATE_HEADS + GAUSSIAN_LTV + REGRESSION_ONLY, else 0.0 (`automl_package/models/probabilistic_regression.py:272-284`) |
| `constrain_middle_class` | `True` | at odd k the middle class gets a constant head (`automl_package/models/common/regression_heads.py:312`, `:331`); inert at even k |
| `boundary_regularization_method` | `NONE` | |
| `use_monotonic_constraints`, `use_anchored_heads` | `False` | |
| `n_classes_selection_method` | `NONE` | fixed k |
| `capacity_selection` | `FIXED` | |
| `allow_retired_capacity_selection` | `False` | escape hatch for retired modes (§3.3) |

**Early stopping** in the library loop (`automl_package/models/base_pytorch.py:100` onward) monitors
the full training loss on the validation split, penalty terms included — under the defaults that is
the collapsed Gaussian NLL plus the ordering penalty at weight 1.0
(`docs/probreg/review/groundwork/data_roles_defaults_stopping.md`, early-stopping table, local-only). <!-- citecheck-ignore: local-only review pack -->
The structure battery replaced it with its own stop on the arm's held-out regression term only
(`automl_package/examples/probreg_structure_battery.py:44-62`).

**Fact worth stating plainly:** the library default still trains a **learned** per-class variance
(`head_spread=PER_INPUT` with the collapsed Gaussian likelihood). Fixed-σ training exists only as an
opt-in (`prob_reg_loss_type=FIXED_SIGMA_MIXTURE` with a required `fixed_sigma_train`,
`automl_package/models/probabilistic_regression.py:261-262`, `:448-459`), added for the structure
study (commit `25fcf9a`). Decision 26's "σ fixed in training" has not become the default.

### 2.3 The three head layouts and the nested ladder

`docs/probreg/PLAN.md` §1 pins `SEPARATE_HEADS` for all three variants and records why the layout
matters to the nested ladder: "the first c rungs are a genuine c-component model" needs
components to exist. `SINGLE_HEAD_N_OUTPUTS` is kept as a labelled comparison arm (Decision 27);
`SINGLE_HEAD_FINAL_OUTPUT` produces no components and is refused under `NESTED` at construction
(`automl_package/models/probabilistic_regression.py:244-258`) and by `predict_distribution`.
For fixed k these restrictions do not apply; all three layouts are constructible.

## 3. Training objectives and choices investigated

### 3.1 Likelihood form

| Form | Code | Status |
|---|---|---|
| Collapsed Gaussian (law of total variance): one Gaussian with the mixture's mean and total variance, scored by Gaussian NLL | default; `calculate_combined_loss` branch, `automl_package/models/probabilistic_regression.py:460-468` | library default |
| Mixture NLL with learned per-class variances (`ProbRegLossType.MDN`) | `automl_package/models/probabilistic_regression.py:432-447` | opt-in |
| Mixture NLL at a fixed shared σ (`FIXED_SIGMA_MIXTURE`) | `automl_package/models/probabilistic_regression.py:448-459`; `fixed_sigma_mixture_log_likelihood` in `automl_package/utils/losses.py` | opt-in since `25fcf9a` (2026-07-21) |
| β-NLL (`loss_type="beta_nll"`, Seitzer 2022), symlog target transform | `loss_type`, `target_transform` defaults | out of scope of the structure study ("orthogonal UQ conveniences", `docs/probreg/fixed_k/plan.md:282`) |

- **April 2026** (LEARNED-σ, LEGACY TOYS): the 8-cell identifiability matrix compared collapsed
  Gaussian and MDN; the note concluded MDN is "situational, not uniform" — better on exponential
  and bimodal at k=3, worse on heteroscedastic and piecewise — and kept the collapsed Gaussian as
  default (`docs/probreg/history/identifiability_2026-04/probreg_identifiability_research.md:628-638`).
  Its §13 critique called the collapsed form "partially sound … misspecifies the mixture;
  components unidentified" and MDN "sound" (`:1361-1380`).
- **July 3 2026**: the owner's framing note recorded a provisional leaning toward mixture scoring
  ("blend scoring makes k nearly invisible to the likelihood"), to be confirmed in-repo
  (`docs/probreg/history/capacity_ladder_2026-07/plans/handover/REVIEW_HANDOVER_2026-07-03.md`, FRAMING 5). <!-- citecheck-ignore: local-only handover note -->
- **July 21–22 2026** (structure study PS-1): likelihood was a trunk axis (collapsed vs mixture).
  No arm won on the primary readout; the D4 tie-break prefers mixture (`docs/probreg/fixed_k/plan.md:395-400`).

### 3.2 Spread (σ)

- Default: each head emits a log-variance as a function of p_i (LEARNED-σ).
- **Decision 26 (2026-07-21)** fixed σ in training as a model-definition change, for a stated
  reason: "a learned per-class variance lets one component absorb dispersion by widening itself"
  (`docs/probreg/PLAN.md` §1). The same day the structure study **suspended** Decision 26 and
  made fixed σ one trunk arm of three (`docs/probreg/fixed_k/plan.md:226-233`):
  `per_input` (learned), `fixed_shared` (one constant σ_toy per toy), `all_constant` (per-class
  constant mean and log-variance, `HeadSpread.ALL_CONSTANT`). A per-class scalar σ was proposed and
  withdrawn by the owner at plan review (`docs/probreg/fixed_k/plan.md:273`).
- How σ_toy is set for fixed-σ arms: the toy's construction noise; for input-varying-noise toys,
  the RMS of the true σ(x) over a fixed-seed population of 200 000 draws
  (`automl_package/examples/probreg_structure_battery.py:63-70`).
- **Owner ruling Q2 (2026-10-20):** σ is a design option; start with it fixed. Learned σ is a
  parked open question (`docs/probreg/PLAN.md`, "Parked open question — learned σ"), with the
  June–July variance lane (`automl_package/examples/capacity_ladder_results/R4_verdict.md`) as the
  starting point when it is reopened.

### 3.3 How the classifier is supervised

`optimization_strategy` (`automl_package/enums.py`):
- `REGRESSION_ONLY` (default) — no cross-entropy; classes are whatever the regression loss makes them.
- `CE_STOP_GRAD` — cross-entropy against percentile-bin labels, with the classifier's probabilities
  detached from the heads (the MDN and fixed-σ branches also detach the logits,
  `automl_package/models/probabilistic_regression.py:437-443`, `:452-457`).
- `COMPOSITE_LOSS` — regression loss plus cross-entropy.
- `GRADIENT_STOP` — the cross-entropy term is skipped (`is_ce_active` excludes it,
  `automl_package/models/probabilistic_regression.py:475-478`) while the mixture branches detach
  the logits; its gradient path therefore differs by loss type.

History:
- **April 2026** (LEARNED-σ, LEGACY TOYS): `CE_STOP_GRAD` was "a situational dial, not default":
  best heteroscedastic MSE (cell C), worse bimodal MSE, and failure by an order of magnitude on the
  exponential toy (`docs/probreg/history/identifiability_2026-04/probreg_identifiability_research.md:615-626`); a symlog transform did not
  rescue it (§8, `:785-865`). The note's §7.7.1 recorded the tension that cross-entropy is a
  likelihood on an observable (bin labels) while the ordering penalty is not
  (`:704-743`).
- **2026-07-11**: the cross-k class-identity conflict — under the nested ladder, bins are re-cut at
  every k, so one class is asked to mean different target ranges at once — was resolved by freezing
  `REGRESSION_ONLY` (`docs/probreg/shared/state.md` §2.5, defect D4: "only avoided, not fixed").
- **2026-07-21, Decision 29**: the three cross-entropy modes were retired at construction, on an
  argument independent of in-training selection: cross-entropy "trains the classifier toward a
  predetermined percentile carve-up of y" (`docs/probreg/PLAN.md`, Decision 29). They raise
  `ValueError` unless `allow_retired_capacity_selection=True`
  (`automl_package/models/probabilistic_regression.py:46-61`, `:218-242`).
- **2026-07-21, structure study**: `CE_STOP_GRAD` re-admitted as a labelled arm through the escape
  hatch (`docs/probreg/fixed_k/plan.md:230-233`). PS-1's winner carries `ce` (§4.3). The plan
  records that PS-1 "supplies no positive evidence that cross-entropy supervision helps — only that
  the unsupervised arms violate the ordering constraint" (`docs/probreg/fixed_k/plan.md:187-192`).
- **Owner ruling Q4 (2026-10-20): open** — "neither the 2026-07-21 retirement nor the structure
  study's finding is final" (`docs/probreg/PLAN.md` rulings table).

### 3.4 Identifiability and constraints

The loss under the collapsed Gaussian depends only on the mixture's mean and variance, so any
relabelling of heads gives the same loss — the "head-index swap" degeneracy
(`docs/probreg/history/identifiability_2026-04/probreg_identifiability_research.md:82-95`). What has been tried:

| Constraint | What it does | Evidence and status |
|---|---|---|
| Ordering penalty (`automl_package/utils/ordering_loss.py`) | penalises out-of-order top-decile head means, fixed weight 1.0, margin 0 | April rerun: 2–7× lower anchor error with neutral-to-better MSE/NLL on 7/8 cells, **only** for collapsed Gaussian + `REGRESSION_ONLY`; redundant under `CE_STOP_GRAD`, harmful with MDN (`docs/probreg/history/identifiability_2026-04/probreg_identifiability_research.md:577-596`, `:974-998`). Auto-on for that one triple. LEARNED-σ, LEGACY TOYS. The review terms of reference ask whether a fixed-weight penalty fits "priors and likelihoods only". |
| Anchored heads (`AnchoredHead`, `automl_package/models/common/regression_heads.py:168`) | pins each head near its bin centroid | rejected in April "on statistical grounds" (§13) and replaced by the ordering penalty (`docs/probreg/history/identifiability_2026-04/probreg_identifiability_research.md:31-35`); a PS-2 arm, recorded inapplicable at the PS-1 winner (`automl_package/examples/capacity_ladder_results/PS2/inapplicable_arms.json`) |
| Middle-class constant head | at odd k, the middle class has a constant mean | default on; inert at even k. PS-2 found it byte-identical at k=2 and k=6, measured it at k=3 only (D-PS-8, `docs/probreg/fixed_k/plan.md:70-83`) |
| Middle-class occupancy penalty | push middle-bin mass toward 1/k | declined in April ("emptiness is information, not a bug", `docs/probreg/history/identifiability_2026-04/probreg_identifiability_research.md:745-783`) |
| Boundary regularisation (`HARDSIGMOID`, `PENALTY`) | bounds head means to class ranges | off by default; broken in the library until `f6ae824` (2026-07-22, D-PS-9) — "had no test anywhere" before PS-2 (`docs/probreg/fixed_k/plan.md:84-98`) |
| Monotonic head slopes | monotone head in p_i | off; PS-2 arm, inapplicable at the PS-1 winner |
| Centroid warm start | initialise head means at bin centroids | shipped; measured null on the P2 configuration, `+0.0026 ± 0.0374` paired over 5 seeds (`docs/probreg/fixed_k/plan.md:741-744`) <!-- numcheck-ignore: quoted from fixed_k/plan.md P2 note (a derived paired statistic, not a single cell) --> |

### 3.5 Choosing k for a fixed-k model

Within the fixed-k variant k is an input; but every fixed-k result so far had to pick one:
- **Report (a)** chose k per toy from the grid {5, 8, 10, 12} by the lowest held-out NLL on a
  dedicated draw (`docs/probreg/fixed_k/reports/report_a/report_a_probreg_toys.md` §2.2 and
  Table 1). That NLL was read at the model's fitted variance — **SELECTED ON LEARNED σ** — because
  the driver was migrated to the fixed-σ mixture only on 2026-07-21 (`ef7e1f8`), after report (a)'s
  results were written (files dated 2026-07-16); the saved selection records carry `val_nll_by_k`, the
  pre-migration field (e.g. `automl_package/examples/report_a_results/k_selection__heteroscedastic.json`,
  selected k=8), whereas the driver at HEAD writes `val_ll_by_k`. The grid starts at 5 and has no bypass.
  `select_k_for_toy` (`automl_package/examples/report_a_benchmark.py:369`) now scores on the fixed-σ
  mixture but still uses `K_GRID = (5, 8, 10, 12)` (`:114`) and an argmax rule.
- **The structure study** pinned k as a condition, never a claim: k ∈ {2, 6} (truth-scale and
  deliberate over-capacity), k=3 added in PS-2 (`docs/probreg/fixed_k/plan.md:308-311`, D-PS-8).
- **The failing test P2** pins k=5 by hand (`tests/test_phase4_regression.py`, `_make_probreg`);
  the plan records that "the test cannot distinguish 'the model regressed' from 'k=5 was never the
  right resolution'" (`docs/probreg/fixed_k/plan.md:720-728`).

## 4. Toy results and their caveats

### 4.1 April 2026 — the identifiability matrix

- **Question:** which of MDN likelihood, `CE_STOP_GRAD` and anchored heads breaks the head-index swap.
- **Setting:** 8 ProbReg cells (2 likelihoods × 2 supervision modes × anchored on/off) plus
  ClassReg; 4 LEGACY TOYS, n=600–800, 1-D; k ∈ {3, 5}; 3 seeds; 80 epochs, early stopping 15,
  20% validation (`docs/probreg/history/identifiability_2026-04/probreg_identifiability_research.md:373-422`). Results
  `automl_package/examples/probreg_identifiability_results/summary.csv`.
- **The note concluded:** ordering penalty for collapsed Gaussian + `REGRESSION_ONLY` (cell B) as
  "the best safe default"; `CE_STOP_GRAD` and MDN as situational dials; ProbReg beats ClassReg on
  all four toys (`docs/probreg/history/identifiability_2026-04/probreg_identifiability_research.md:598-660`).
- **Labels:** LEARNED-σ, LEGACY TOYS. Point and density scores only; no fixed-σ readout existed.

### 4.2 Report (a), 16 July 2026 — fixed-k against trees and a plain network

- **Question:** calibration and point accuracy of ProbReg (fixed k, and an adaptive in-training
  variant) against XGBoost, LightGBM, CatBoost and a plain network.
- **Setting:** 5 toys (heteroscedastic, multimodal, three-region, piecewise, exponential — the
  legacy set plus a three-region toy), 5 fresh draws each, n=800–1000, 70/30 split, k per toy from
  {5, 8, 10, 12} (§3.5); learned per-class variances; scored as a single Gaussian
  (`report_a_probreg_toys.md` §2–§5). Results `automl_package/examples/report_a_results/`.
- **What it reported** (its Tables 2–5): the binning model's 90% intervals covered close to 90% on
  the heteroscedastic, piecewise and exponential toys while the trees under-covered; the trees had
  lower squared error on three-region, piecewise and exponential; on the multimodal toy every
  model's single-Gaussian summary was the wrong shape. The fixed-k model's exponential error was
  unstable across draws (`report_a_probreg_toys.md` §7).
- **Its headline was calibration** (`report_a_probreg_toys.md` Summary), which the owner's ruling
  Q1 has since put after k.
- **Labels:** LEARNED-σ (the model of report (a) is the old learned-variance model,
  `docs/probreg/PLAN.md` "Parked open question"); SELECTED ON LEARNED σ for its bin counts; LEGACY
  TOYS; the "adaptive" arm is ELBO + SoftGating in-training selection, retired by Decision 29.
  `docs/probreg/archive/probreg-report.md` is the task plan that produced it.

### 4.3 The structure study, 21–22 July 2026 (PS-1 to PS-3)

The one fixed-k study run on the canonical role-labelled toys, with σ as an arm, scored on a
readout identically defined for every arm: the held-out fixed-σ mixture log-likelihood
(`heldout_fixed_sigma_mll`, higher is better), compared at k=2 by a 2× bootstrap-SE rule on ≥2 toys
with none worse (D2, `docs/probreg/fixed_k/plan.md:340-363`). Toys: `make_toy_b` (reference,
intrinsic k=2), `make_toy_a` (smooth), `make_broad_unimodal` (moment-matched twin of toy b),
`make_toy_c_broad` (variance trap) (`:298-306`). Every cell: 300-epoch cap, early stop on the
arm's own held-out NLL (patience 8 checks), capped cells re-run once at 600
(`docs/probreg/fixed_k/plan.md:251-254`).

| Stage | Cells | Decision as recorded | Source |
|---|---|---|---|
| PS-1 trunk (supervision × spread × likelihood) | 160 at k ∈ {2, 6}, seeds {0, 1}; 20 capped and re-run; 1 still not converged, excluded | winner `ce × fixed_shared × mixture`, `decided_by: d4_tiebreak`, `primary_winner: null`; every `none` (unsupervised) arm disqualified by D3 on ordering violations | `automl_package/examples/capacity_ladder_results/PS1/ps1_decision.json` |
| PS-2 patch audit | 144 at k ∈ {2, 3, 6}, seeds {0, 1, 2}; 24 capped, all converged on re-run | no patch kept; winner = trunk with every patch off, by D4 | `automl_package/examples/capacity_ladder_results/PS2/ps2_decision.json` |
| PS-3 head layouts (separate / single-n / single-final, parameter-matched) | 72 at k ∈ {2, 6}, seeds {0, 1, 2}; all converged | winner `separate` by D4; the mechanism-control halt rule fired (three-way tie on the aggregate rule) | `automl_package/examples/capacity_ladder_results/PS3/ps3_decision.json` |
| PS-4 certification | — | **not run**; blocked on the owner's ruling on the PS-3 halt | `docs/probreg/fixed_k/plan.md:29` |

What the plan's evidence notes attach to these (`docs/probreg/fixed_k/plan.md:121-192`), quoted or
closely paraphrased:
- **PS-1:** "The winner was chosen by D4's simplicity order, not by D2"; the trunk is
  "underdetermined between `fixed_shared` and `per_input`", and PS-2/PS-3 "inherit it as an
  assumption". The supervision axis "closed by disqualification of the alternative, not by a D2 win".
- **PS-2:** middle-class is inert at even k; at k=3, middle-class on was ahead on 3 of 4 toys
  (+0.087 nats on toy b), "not strong enough to overturn anything". The boundary arm lost on all four <!-- numcheck-ignore: quoted from fixed_k/plan.md PS-2 note (cross-cell seed-mean difference) -->
  toys at k=2.
- **PS-3 (the halt):** per toy, `separate` beat `single_final` by +1.19 nats on toy b (2×SE 0.15),
  `single_final` won by 0.03 on toy a, and the two broad toys tied. The note judged the trigger "an
  aggregation artifact" and set out options (a) accept the halt, (b) rule that the aggregate rule
  misfires on a mixed suite and certify `separate`, (c) amend the halt rule to read the reference toy
  per-toy; it recommended against (a). The mechanism control's scoring (a degenerate one-component
  fixed-σ mixture, D-PS-10) is a reversible default awaiting ratification "because … a different
  rule here could flip the halt outcome" (`:100-119`).
- **Labels:** PROVISIONAL (every stage decided by tie-break; PS-3 awaiting a ruling since
  2026-07-22). k is a condition here, never a claim. Unregularised training (no weight decay,
  dropout or mini-batching), which `docs/probreg/global_k/plan.md` P8 flags as a possible confound
  for any k comparison.
- **Not compared in this study:** a plain regression network of matched capacity (the Q3 / O1b
  comparison). The mechanism control `single_final` still has a classifier; it is not a
  no-classification arm.
- **The twin pair as run is not mean-matched.** The battery builds `toy_b` (k_true=2, separation
  4, σ=0.3) with the generator's default `baseline="sin"`, so both modes ride on sin(2πx), while it
  builds `broad_unimodal`, the study's "moment-matched twin of toy_b", with `baseline="zero"`
  (`automl_package/examples/probreg_structure_battery.py:359-362`; default at
  `automl_package/examples/_toy_datasets.py:55`; the twin's docstring says the baseline "must match the
  bimodal partner", `:118`). The tier-1 reference cell of `docs/probreg/shared/toy_suite.md` is
  `make_toy_b` with `baseline="zero"`.
- **Floors:** no true-density held-out score is stored for any of the four toys this study used
  (`docs/probreg/review/groundwork/toys.md`, local-only). σ_toy for `toy_b` is its true component <!-- citecheck-ignore: local-only review pack -->
  σ; for `broad_unimodal` it is that toy's total standard deviation, σ·√(1 + 4²/4)
  (`automl_package/examples/probreg_structure_battery.py:415-420`).

### 4.4 The two accepted failing tests (defect D3)

- `test_probabilistic_nll_beats_constant_on_heteroscedastic` — root-caused and accepted
  (`docs/probreg/history/diagnoses_2026-07/hetero_nll_diagnosis.md`).
- `test_prob_regression_heteroscedastic_mse` (task P2) — accepted: a single-seed assertion at
  k=5; at HEAD it fails at MSE 2.88 against a 2.5 bar; four of five seeds land under the bar
  (`docs/probreg/fixed_k/plan.md:691-714`; cells `automl_package/examples/capacity_ladder_results/P2/p2_warmstart_grid.json`).
  Ruled by the owner (2026-07-20): the test's protocol is to be repaired by selecting k on held-out
  data once the selector exists, holding the bar.
- Both test learned-variance behaviour; Decision 26 states the suite bar "must be RE-BASELINED"
  if fixed σ becomes the training definition (`docs/probreg/PLAN.md`, Decision 26).

### 4.5 Smaller fixed-k studies on record

- **Head-structure diagnostics** (`automl_package/examples/head_structure_diagnostic.py`,
  `automl_package/examples/head_structure_results/head_structure.csv`): separate heads vs
  single-head-n-outputs on LEGACY TOYS with a learned-variance loss; the plan records them as
  "citable as history, labelled legacy; not a baseline for P11" (`docs/probreg/fixed_k/plan.md:844-852`).
- Further April–July probes (ordering ablation, classifier symmetry, middle-class centering, the
  exponential symlog probe, separate-vs-single-final heads) are listed with their scripts and
  results folders in `docs/probreg/PLAN.md`, "Code and results of this stream" (history row).

## 5. Open decisions and planned next steps

What the plans already name, with their status as the plans state it:

| Item | What it asks | Status per its source |
|---|---|---|
| PS-3 halt ruling | options (a)/(b)/(c) above | awaiting the owner since 2026-07-22 (`docs/probreg/fixed_k/plan.md:146-153`); the review terms of reference ask for a recommendation |
| PS-4 certification | the final structure × 7 toys × k ∈ {2, 4, 6} × 3 seeds, plus σ half/double re-scoring of stored predictions; then a certification memo and an owner gate | not run (`docs/probreg/fixed_k/plan.md:597-607`) |
| Q4 — classifier supervision | cross-entropy or regression-only | open (owner, 2026-10-20) |
| Q3 / O1b — must improve point accuracy | needs a matched-capacity plain network as an arm | no such comparison in the structure study; report (a)'s plain network was not capacity-matched |
| Decision 26 vs ruling Q2 | fixed σ as definition vs as starting design option | the library default is still learned σ (§2.2) |
| Suite re-baseline | the two accepted failures test variance behaviour | required before fixed σ lands as the training definition (Decision 26) |
| P2 test repair | select k on held-out data instead of pinning k=5 | waits on the shared selector (`docs/probreg/fixed_k/plan.md:730-734`) |
| P11 head-structure comparison | three layouts read under the nested ladder at matched capacity | subsumed by PS-3 for fit; its selection reads moved to the k-selection work (`docs/probreg/fixed_k/plan.md:219-222`) |
| Deferred proposals | an order-aware (ordinal) classifier target; heads reading features beyond p_i | proposals for the PS-4 memo, not arms (`docs/probreg/fixed_k/plan.md:289-294`) |
| Learned σ | reopen later | parked (`docs/probreg/PLAN.md`) |

The structure study's wall still stands in the plan: no k-selection work (P7, P8, P11, PB, PC,
P3, P4, P5, P6) proceeds until PS-4's owner gate (`docs/probreg/fixed_k/plan.md:217-222`).

## 6. Sources

Plans: `docs/probreg/PLAN.md`, `docs/probreg/fixed_k/plan.md`, `docs/probreg/shared/state.md`,
`docs/probreg/shared/toy_suite.md`. Reports: `docs/probreg/fixed_k/reports/report_a/report_a_probreg_toys.md`,
`docs/probreg/history/identifiability_2026-04/probreg_identifiability_research.md`,
`docs/probreg/history/diagnoses_2026-07/hetero_nll_diagnosis.md`,
`docs/probreg/history/diagnoses_2026-07/p2_hetero_mse_diagnosis.md`. Results:
`automl_package/examples/capacity_ladder_results/PS1/`, `PS2/`, `PS3/`, `P2/`;
`automl_package/examples/report_a_results/`; `automl_package/examples/probreg_identifiability_results/`.
Code: `automl_package/models/probabilistic_regression.py`,
`automl_package/models/architectures/probabilistic_regression_net.py`,
`automl_package/models/common/regression_heads.py`, `automl_package/utils/ordering_loss.py`,
`automl_package/utils/losses.py`, `automl_package/examples/probreg_structure_battery.py`,
`automl_package/examples/report_a_benchmark.py`.
