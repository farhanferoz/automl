# Nested / shared architecture for the flexible neural network — research record

**Purpose.** Durable record of the focused research done before deciding how to set up the nested
(shared-prefix) architecture for the per-input flexible-width / flexible-depth neural network. Written so
a future implementing session does **not** have to redo this research. Every external claim carries a
citation and a verification-confidence tag; every in-repo claim carries a `file:line`.

**Motivating question.** The per-input capacity dial works for (a) the low-rank covariance ladder and
(b) the ProbReg classifier-over-k-classes, but the *shared-nested* form fails for the neural-network
width and depth knobs. We want to know **why the nested structure is coherent in the first two and
mis-set-up for the neural net**, and what structural fix the evidence supports — before building anything.

**Three research pillars:**
1. How our ProbReg classifier-over-k-classes actually nests, at the parameter level (in-repo). — *pending*
2. Why low-rank rank-dropout reaches the SVD-optimal nested loading (autocast notes). — *pending*
3. What the published literature establishes about ordered/nested-width representations. — **landed (below)**

Status: pillar 3 complete; pillars 1–2 and the cross-pillar synthesis to follow.

---

## Pillar 3 — Literature on ordered / nested representations (landed)

### Per-paper findings

**Matryoshka Representation Learning (MRL).** Kusupati et al., *NeurIPS 2022*, arXiv:2205.13147.
- Mechanism: one shared backbone `F`, and a **dedicated linear head `W(m)` per nesting scale `m`**,
  trained by a joint loss that sums, over O(log d) prefix sizes, the loss of the first-`m` dims:
  `min Σ_i Σ_m c_m · L(W(m)·F(x_i)_{1:m} ; y_i)`.
- Degradation: the **full (per-prefix-head) MRL is at least as accurate as independently-trained
  fixed-feature models at every size**; the cheaper *weight-tied* variant (MRL-E) shows only a small
  (~1%) gap at very low dims. → per-prefix dedicated heads are doing real work.
- Confidence: **medium** (abstract + HTML render; loss equation and "at least as accurate" quote not
  cross-checked by a second search).

**Nested Dropout.** Rippel, Gelbart, Adams, *ICML 2014*, arXiv:1402.0915. (Co-author is Gelbart, not Gebru.)
- Mechanism: stochastically remove *coherent nested sets* of hidden units (sample a truncation index,
  zero every unit past it) instead of independent-unit dropout.
- Theory: **provably enforces unit identifiability**, with an **exact equivalence to PCA** in the
  semi-linear autoencoder case — i.e. training-time nested truncation recovers the same ordered-importance
  structure PCA gets from eigendecomposition. The earliest "ordering via stochastic nested truncation" result.
- Confidence: **medium** (abstract quotes).

**Slimmable Neural Networks.** Yu et al., *ICLR 2019*, arXiv:1812.08928.
- Mechanism: one network runnable at a small discrete set of widths, with **per-width private
  (switchable) BatchNorm**, because different active-channel counts give different feature mean/variance
  that a shared BN mis-estimates.
- Degradation: reports **similar-or-better** accuracy vs individually-trained nets — but only for a fixed
  discrete set (4 widths), not arbitrary width.
- Confidence: **medium** (abstract).

**Universally Slimmable Networks.** Yu & Huang, *ICCV 2019*, arXiv:1903.05134.
- **Sandwich rule** (quote): each iteration trains the model at "smallest width, largest width and (n−2)
  randomly sampled widths" — optimizing the performance lower+upper bound "implicitly optimize[s] the
  model at all widths."
- **Inplace distillation** (quote): use "the predicted label of the model at the largest width as the
  training label for other widths" (ground truth only for the largest), free within the same step.
- **Explicit naive-failure statement** (quote): "**a naively trained model fails to run at different
  widths even if their BN statistics are calibrated**"; sampling only some sub-networks per step leaves BN
  stats "insufficiently accumulated thus inaccurate, leading to much worse accuracy."
- Confidence: **high** (full-text render; quotes cross-checked).

**FjORD / Ordered Dropout.** Horvath et al., *NeurIPS 2021*, arXiv:2102.13451.
- Mechanism: **deterministic contiguous-prefix** retention — submodel for ratio `p` keeps neurons
  `{0..⌈p·K⌉−1}`, prunes the rest, so `p1<p2 ⟹ F_{p1} ⊂ F_{p2}`. Contrast with random dropout: drops
  *adjacent* components, not random ones.
- **Ordered Dropout is equivalent to SVD for linear maps** (direct quote). Deterministic prefix-ordering
  recovers the SVD importance ranking for linear layers (echoes Nested Dropout's PCA equivalence).
- Self-distillation: max-width submodel's softmax teaches every sampled `p`-submodel in the same step.
- Confidence: **high** for mechanism + SVD-equivalence quotes; the "submodel accuracy very close to
  dedicated baselines" (Fig. 3) is **UNVERIFIED** (figure not rendered).

**Once-for-All (OFA).** Cai et al., *ICLR 2020*, arXiv:1908.09791.
- **Progressive shrinking** (quote): train the largest network first, then "progressively fine-tune … to
  support smaller sub-networks that share weights with the larger ones" (order: kernel → depth → width).
- **Interference** (quotes): "different sub-networks are interfering with each other"; "randomly sampling
  a few sub-networks in each step will lead to significant accuracy drops." Progressive shrinking works
  because large sub-nets are "already well-trained" before small ones are introduced.
- **CONTESTED**: Shipard et al., *CVPRW 2022*, arXiv:2204.09210 ("Does Interference Exist…") dispute the
  causal story — interference-mitigation "do[es] not have a large impact," attribute the benefit to
  "subnet architecture selection bias," and find random subnet sampling can **beat** progressive shrinking
  on small/medium datasets. → the "interference" explanation is the original claim, **not settled consensus**.
- Confidence: **high** for OFA quotes; the counter-paper is a flagged contested point.

**Boosting = forward-stagewise additive modeling.** Friedman, "Greedy Function Approximation: A Gradient
Boosting Machine," *Annals of Statistics* 29:1189–1232 (2001); Friedman, Hastie, Tibshirani, "Additive
Logistic Regression," *Annals of Statistics* 28:337–407 (2000).
- Each new additive term is fit to the current residual/functional gradient; earlier terms are not
  revisited → **any prefix (first k rounds) is already a complete, valid, weaker additive model**. This is
  the canonical precedent for "freeze-and-add gives valid nested/additive prefixes."
- **NOT** cited by any of the nested-width NN papers above — the connection is structurally obvious but
  undrawn in that literature.
- Confidence: **verified** for boosting; "NN nested-width papers cite boosting" = **not found / likely absent**.

**Per-input / anytime selection.** BranchyNet (Teerapittayanon et al., 2017, arXiv:1709.01686) and MSDNet
= **early-exit / anytime**, i.e. per-input conditional **depth** (confidence-gated exits). No authoritative
primary source found this pass for per-input **width** routing / mixture-of-widths — flagged as a search
gap, not a proven absence.

### Synthesis (literature only)

- **A — what makes nested width work.** Three families, often combined: (i) a training-time **ordering /
  identifiability** mechanism (Nested Dropout → PCA; Ordered Dropout → SVD-for-linear); (ii) a **joint
  multi-scale / sandwich loss** that supervises the extreme scales every step (MRL; Universally Slimmable);
  (iii) **in-place self-distillation** from the full model to every nested sub-model (Universally Slimmable,
  FjORD; coarser in OFA). Ordering is the deep "why a prefix is meaningful" answer; loss-shaping + distillation
  are the engineering answer for "why training converges without a width starving."
- **B — is the "standalone-vs-component" conflict acknowledged?** Only indirectly. Closest is Universally
  Slimmable's "naively trained model fails to run at different widths," fixed by sandwich + inplace
  distillation (not by a per-unit ordering constraint). OFA frames it as "interference," fixed by
  sequencing — but that mechanism is contested (Shipard 2022).
- **C — does naive shared-trunk nested width underperform, with published fixes?** Yes, stated directly
  (Universally Slimmable; OFA). Published fixes: sandwich + inplace distillation; progressive/sequential
  training + distillation; deterministic importance-ordered truncation (PCA/SVD). MRL is the outlier with
  no degradation — attributable to its **per-prefix dedicated heads**.
- **D — boosting as valid-additive-prefix precedent?** Yes (Friedman 2001; FHT 2000) — verified; but not
  referenced by the NN nested-width papers.
- **E — per-input capacity routing.** Early-exit (BranchyNet, MSDNet) = conditional depth; per-input width
  routing not found with a primary source this pass.

### Direct bearing on our results (flagged for the synthesis)

- Our **shared-nested width failure reproduces a published phenomenon** ("naively trained model fails at
  different widths"). It is not a bug in our setup.
- The theory pins the crux: Ordered Dropout = SVD **only for linear maps**. That is why nesting is clean for
  the (linear-algebraic) covariance and breaks for a **nonlinear** MLP — the hidden units are not an
  eigenbasis and the shared readout entangles them.
- **MRL** (shared backbone + per-prefix dedicated heads, no degradation) is a verified **middle ground**
  between our failed fully-shared net and our working K×-param independent nets — directly relevant to the
  "efficient middle ground" question.
- **Tension to reconcile:** the standard published fixes are sandwich + inplace distillation, and we
  **already tried both** (SANDWICH schedule; inplace distillation) and neither rescued our width toy. So
  the answer is probably **per-prefix dedicated heads / importance-ordering** (not cleanly tried) rather
  than more loss-shaping (tried, insufficient) — to be confirmed against pillars 1–2.

---

## Pillar 1 — ProbReg classifier-over-k-classes nesting (in-repo)

All claims `file:line`-verified against the shippable model.

**Structure (as a classifier over k percentile-bin classes).**
- **Classifier**: one `PyTorchNeuralNetwork`, `output_size = n_classes` (`architectures/probabilistic_regression_net.py:91-100`).
- **Regression heads (default `SEPARATE_HEADS`)**: an **independent** `BaseRegressionHead` per class (own
  `nn.Linear` stack), no shared trunk between class heads (`common/regression_heads.py:283-349`). *Each head
  takes only a scalar — its own class probability `p_i` — never the raw input x* (`:337-338`, `:369-371`).
- **Combination**: Law of Total Variance, probability-weighted — `final_mean = Σ P(C=i)·E[Y|C=i]`,
  `final_var = E[Var(Y|C)] + Var(E[Y|C])` (`utils/pytorch_utils.py:149-177`). Confirmed.
- **Classes**: ordered percentile bins via `create_bins` (`utils/numerics.py:32-55`), **recomputed
  independently for each k** in `_fit_single` (`probabilistic_regression.py:504-506`). Because
  `percentiles = linspace(0,100,k+1)` shifts with k, the k=2 edges are **NOT a subset** of the k=3 edges —
  independently-defined quantile grids, **not** a coarse-to-fine refinement. (Direct code fact.)

**Nesting semantics.**
- Fixed k (`NClassesSelectionMethod.NONE`, the default): **fully separate model instantiations, zero
  parameter reuse across k** (`probabilistic_regression_net.py:86-100`).
- Dynamic k (non-NONE): ONE classifier (`output=max_k`) + ONE head-set, shared across all k via
  **masked-prefix** (mask logits to the first k, softmax, same heads) — `probabilistic_regression_net.py:132-149`,
  `selection_strategies/base_selection_strategy.py:88-153`. So k=2 and k=7 reuse the identical classifier
  forward and head objects 0,1. But it is **masked-prefix, NOT importance-ordered**, and "class i" is
  redefined per k (Q3).

**THE KEY FINDING — the shipped dynamic-k path has the SAME node-0 conflict as NN width (verified, Q5).**
Heads/logits 0,1 are active for every k≥2; the classifier's per-k cross-entropy target is built from the
**k-specific recomputed boundaries** (`_calculate_custom_loss`, `probabilistic_regression.py:240-244`). So
**head 0 (and logit 0) is asked to be simultaneously correct for "bottom 50% of y" (k=2) and "bottom 10%"
(k=10), with no input distinguishing the regime.** This is the exact "one component must double as a good
predictor at multiple resolutions" conflict — code-confirmed, not inferred.

**Crucial distinction — this is NOT the surrogate that WS1 validated.** The per-input count recovery that
"works" (WS1, `RESULTS.md`) used the **K4 nested-k surrogate** (`examples/_capacity_ladder_nested.py`), whose
components are **importance-ordered by the nesting itself** ("an importance ordering created by nesting, not
seeded by location"), with independent heads and a spread-init used only as init — it does **not** recompute
bins per k, so "component i" is stable across k. That is why K4 recovers and the shipped dynamic-k path
carries the conflict. **OPEN VERIFICATION ITEM (before building on it):** confirm whether the shipped
"recipe-of-record" masked-prefix *training* (H1) inherits the recompute-per-k boundaries (`:240-244`) — if
so, the recipe of record shares the defect and only the post-hoc selector rescues it.
**→ RESOLVED 2026-07-11 (good direction): it does NOT.** Every H1 arm sets
`optimization_strategy=REGRESSION_ONLY` (`capacity_ladder_h1.py:99`) which makes `is_ce_active` False
(`probabilistic_regression.py:219-222`), so the CE/per-k-boundary branch structurally never fires;
cross-k identity is carried by the one shared classifier + fixed heads under masked-prefix softmax.
The defect is scoped to dynamic-k configurations with CE active. Also, one precision: boundaries are
precomputed once per k (`:504-506`), not per step — what varies per step is which k's boundary set
applies to which sample (per-sample gate argmax). Full verification table + traces:
`annex_probreg_verification_2026-07-11.md`; consequences folded into
`cascade_execution_plan_2026-07-11.md` §3.1.

**STRUCTURE CRITIQUE (ProbReg) — items the owner should see:**
- **[central, structural]** Masked-prefix sharing is coherent in principle but **sloppy in implementation**:
  per-k boundaries recomputed independently break the assumption that shared index i means the same thing
  across k (`probabilistic_regression.py:504-506` + `240-244`). This is the ProbReg mirror of the width
  node-0 conflict, and the most relevant fact for the nesting design.
- **[expressiveness ceiling]** SEPARATE_HEADS heads never see x, only the classifier's scalar confidence
  (`regression_heads.py:369-371`) — two inputs in the same class with the same confidence are
  indistinguishable to the head. Likely intentional ("classification bottleneck"), but a real ceiling.
- **[likely bug, latent/untested]** `get_classifier_predictions` binary case uses
  `sigmoid(logit0)` (`probabilistic_regression.py:796-798`) while training uses full 2-way softmax CE — the
  true positive prob is `sigmoid(logit1−logit0)`; wrong whenever logit1≠0. Only affects that diagnostic
  method (not `predict`/training). No test exercises it. Cheaply fixable.
- **[silent failure risk]** `create_bins` dedups tied edges and **silently shrinks n_bins** on
  discrete/tied targets (`numerics.py:37-43`), stored directly (`probabilistic_regression.py:506`) with no
  warning in the re-binning path → fewer effective classes than requested k, unsurfaced.
- **[hygiene]** Inert HPO dims (`gumbel_tau`, `n_classes_predictor_learning_rate` added unconditionally,
  inert for NONE + 3/4 dynamic strategies, `:406-435`); unconditional dead centroid computation in the
  default config (`:526-530`, consumed only under monotonic/anchored heads, both default-False).
- **[context, not a defect]** A lower-param alternative already exists (`SINGLE_HEAD_N_OUTPUTS` shares one
  trunk, `regression_heads.py:385-467`); SEPARATE_HEADS is default anyway.

---

## Pillar 2 — Low-rank rank-dropout → SVD-optimal nested loading (autocast notes)

Sources: `~/dev/turing/autocast-private/notes/design/rank_ladder/{rank_ladder_design_2026-07-03.md,
rank_ladder_results_2026-07-03.md, rank_ladder_execution_plan.md, rank_ladder_mixture_discussion_2026-07-04.md}`
and the toy code `~/dev/turing/autocast-private/local_hydra/local_experiment/cal_study/rank_ladder_toy.py`.

**Parametrization (no orthogonality constraint anywhere).** ONE shared, unconstrained loading `M`
(n_dim × r_max=8), a single `nn.Parameter` (`rank_ladder_toy.py:520`), plus a diagonal `D` (`:518`). The
covariance at rung r is **moment-matched**, not a plain truncation:
`Σ_r = D + diag(Σ_{k>r} m_k m_kᵀ) + Σ_{k≤r} m_k m_kᵀ` (design:43; code `:22-30`) — **dropped columns are
folded into the diagonal, so every rung has identical marginals by construction** (this is why
calibration is "rung-invariant"). No QR/Gram-Schmidt/orthogonality/cross-column penalty exists; `M` is raw.
No per-rung parameters, no rank-conditioning input.

**Training loop (`rank_ladder_toy.py:917-920`, `407-446`).** Per-sample uniform draw `r ~ {0..8}`; a hard
0/1 prefix mask `_mask_from_r` (column j kept iff `j < r`); a masked Woodbury NLL `rung_nll` that folds
masked columns into the diagonal and Woodbury-solves the kept ones (literally the moment-matched `Σ_r`).
Objective `E_r[NLL_r]`, one fresh per-sample r per minibatch — "every rung trained to be individually
good." Plus a **flat, isotropic** Gaussian prior `prior_c·Σ(m²)` (column-index-blind).

**Why the prefix becomes importance-ordered — THE MECHANISM (stated design argument, not a theorem;
confidence: stated-as-intuition, no literature citation in the notes).** Two training-induced levers:
- **(a) Prefix-scoring breaks rotational degeneracy** (design:57-58, quote: "Rotation freedom (M → MR)
  reshuffles at constant MMᵀ, bounded, broken by nesting's ordering pressure"). A single fixed-rank
  `MMᵀ` fit is rotationally degenerate (any orthogonal R gives the same `MMᵀ`), so column order carries
  no signal. But truncating a *prefix* is **not** rotation-invariant, so scoring the SAME `M` at every
  prefix length simultaneously breaks the degeneracy — the only rotation good at every truncation is
  (approximately) the importance-sorted (Eckart-Young/PCA-style) ordering.
- **(b) Asymmetric exposure schedule supplies the ordering pressure** (design:68-69, quote: "column k
  directional exposure = (9−k)/9 (89% → 11%)"). Under uniform-r, column j is kept with probability
  `(8−j)/9`, so column 0 acts as a correlated direction in ~89% of steps and the last column in ~11%.
  Gradient pushes the highest-value correlation direction into the highest-exposure slot.

  **KEY CROSS-LINK.** This is exactly Nested Dropout / Ordered Dropout (Pillar 3, [2],[5]) — and those give
  a **provable PCA/SVD equivalence only for LINEAR maps**. The autocast argument (a) is itself a *linear*
  rotation-degeneracy argument. So low-rank nests cleanly **because the covariance is a linear-algebraic
  object**; a nonlinear MLP has no such rotational/SVD structure, which is the mechanistic reason the same
  uniform-prefix-dropout does *not* induce a clean ordering for MLP width. The notes do not cite the
  ordered-representation literature — the connection is drawn here.

**Evidence it reaches ~ML optimum.** Moment-anchored: nested MV matches its own fit split 0.987 (closed-form
ML = 1.000 by construction) (`results:526-538`). The trained rank-dropout ssr sweep across ρ={0,.25,.5,1,2,4}
matches the offline closed-form PPCA-MLE ssr sweep to ~2 decimals at every point (`results:150-157` vs
`268-275`). *(The "digit-for-digit ⇒ training = ML optimum" phrasing (`execution_plan:45`) is a cross-table
inference by the investigator, not a co-located verbatim measurement — flagged.)*

**Prefix stability.** (a) **Trivially guaranteed by code**: one stored `M`, every readout slices `m[:, :r]`
— no per-rank re-fit, so "the first r columns don't change at rank r+1" is true by construction, not an
emergent training property. (b) Whether that fixed ordering is actually *good/stable across seeds* was
**NOT directly measured**: the design specified a `‖m_k‖` / `subspace_r2` column diagnostic and a
graded-prior trigger "if ordering instability > 0.1 across seeds" (design:239-246, 132), but no such number
is reported for the trained nested arm, and the graded-prior arm stayed default-OFF (indirect
non-firing, not a positive measurement).

**Caveat / where the prefix is NOT optimal.** Under heavy bias (ρ=4) the readout calibration goes
over-confident (ssr 0.81), **but** Check 3 attributes this to inter-trajectory bias-energy split-imbalance
(readout moment 1.51× the fit moment), NOT to a training/ML defect — the trained `M` matches its own
fit-split moment ~exactly (`results:540-547`). (The 2.27 SSR blow-up under heavy tails belongs to the
retired MIXTURE arm A4, not nested — easy to misattribute.)

**STRUCTURE CRITIQUE (areas to improve / port):**
- *Unverified assumption (the big one):* the low-rank program **never directly verified that its trained
  loading is importance-ordered at the column level.** Confirmed by grep of both the results note and the
  execution plan: the design's own `‖m_k‖` / cross-seed `subspace_r2` column diagnostic (design:132,239-246)
  was specified but **never reported for the trained (A3) arm** — the only subspace_r2 numbers are for the
  *offline closed-form* fit (A2). Every "reaches ML optimum" data point is **aggregate/calibration-level**
  (MV/fit≈0.987; ssr sweep matching the closed-form). And the ordering *mechanism* itself is a plausibility
  argument (rotation-degeneracy + a *gentle linear* 89→11% exposure gradient), not a proof and not tied to
  any cited theorem. Related unresolved items the notes flag: a pre-built "graded-prior" escape hatch that was
  never validated (its trigger check was never run); untested scaling of ordering fidelity as r_max grows
  (per-column exposure pressure narrows). ⇒ **do NOT assume training induces a clean ordering — measure it.**
- *Surrounding fragilities (autocast-internal, acknowledged in their notes; less relevant to our port):*
  bias-driven readout over-confidence attributed to split-sampling (item 1); preflight gauges have a blind
  spot needing a mandatory cell-check backstop; the arbiter's "safe abstain" is a small-sample power artifact
  (selects bias-carrying rungs at realistic trajectory counts); at their first realistic arena the internal
  identifiability gauge fires on all positive seeds and rank recovery is only a SOFT pass.
- *Nice property worth porting:* **moment-matching** — dropped columns' variance is folded into the
  fallback diagonal so every prefix is a *calibrated* model by construction, not a truncated/mis-calibrated
  one. An NN analog (fold "dropped-unit" capacity into a fallback term rather than hard-zeroing) may make
  every width a valid model by construction.
- *Confidence:* mechanism = stated design argument (not a proven theorem, no lit citation in the notes);
  ML-match = measured but the "digit-for-digit" wording is investigator inference; ordering-stability =
  unmeasured.

---

## Cross-pillar synthesis + design implications

### The coherence invariant (what every working case shares; every failing case violates)

A nested capacity ladder is **coherent** iff, for every rung i, all three hold:
1. **Stable identity** — "rung i" means the SAME thing at every prefix length ≥ i (not redefined per k).
2. **Self-contained contribution** — rung i is a complete additive/standalone piece, not a partial feature
   entangled with the others through one shared readout.
3. **Importance ordering** — rungs are ordered so that any prefix is (near-)optimal for its size.

**Scorecard (all three pillars):**

| Case | (1) Stable identity | (2) Self-contained | (3) Importance-ordered | Works? |
|---|---|---|---|---|
| Low-rank covariance (A3) | ✓ moment-matched column i | ✓ additive rank-1 factor | ✓ training-induced (prefix-dropout + 89→11% exposure) | **yes** — rides on LINEARITY (SVD) |
| ProbReg K4 surrogate (WS1) | ✓ importance-ordered component | ✓ independent head | ✓ manufactured by nested-dropout | **yes** |
| ProbReg shipped dynamic-k | ✗ class i redefined per k (bins recomputed) | ~ independent heads | ✗ masked-prefix over unstable bins | **no — same node-0 conflict** |
| NN width shared trunk | ✗ unit role changes with width | ✗ partial feature + shared readout | ✗ none, + nonlinear (no SVD) | **no** (fails all three) |
| NN width independent (built) | ✓ (whole net per width) | ✓ complete net per width | n/a | **yes but K× params, not "nested"** |

Two independent confirmations that **linearity is the hidden enabler** for the clean cases: Pillar 3
(Ordered Dropout = SVD **only for linear maps**) and Pillar 2 (the low-rank ordering argument is a *linear*
rotation-degeneracy argument). A nonlinear MLP has neither → the same prefix-dropout does not induce a clean
ordering, which is exactly what we observed.

### Why the standard published fixes did NOT rescue our width toy

Sandwich rule + inplace distillation (Universally Slimmable, FjORD) address **gradient starvation** — making
sure every width receives training signal. They do **not** establish stable identity, self-containment, or
importance ordering; they keep the shared-readout entanglement. So they treat a *different* failure mode than
the one binding for us. We ran both and they didn't help — consistent with this diagnosis, not a
contradiction. Our own two results triangulate it: **independent-width works** (self-contained rungs, but
K×params) and **k-dropout-sandwich fails** (loss-shaping without ordering/identity). The missing ingredients
are (1)+(2)+(3), not more loss-shaping.

### Design implications — candidate architectures (PROPOSED, for owner review before any build)

Goal: satisfy the invariant while keeping ONE shared representation (efficiency + cheap prefix inference),
i.e. earn back what independent-width throws away.

1. **Frozen residual cascade (boosting-style) — primary candidate.** Block k is trained to fit the
   *residual* of blocks 1..k-1; earlier blocks are **frozen** when block k is added. Forces all three
   properties *by construction*: stable identity (frozen ⇒ block k never changes), self-contained (each block
   a complete additive correction), importance ordering (greedy residual-reduction ⇒ decreasing importance).
   Keeps ~1× params + cheap width-k inference (run first k blocks). Crucially, boosting (Friedman 2001) works
   for **nonlinear/arbitrary** weak learners — it is the nonlinear generalization of low-rank's SVD nesting,
   sidestepping the "SVD only for linear" barrier. Cost: staged/greedy training (akin to OFA progressive
   shrinking — lit-supported, though the "interference" rationale is contested by Shipard 2022).

2. **Matryoshka-style: shared backbone + per-width dedicated heads — efficiency fallback.** One shared trunk,
   a dedicated readout head per width scale, joint multi-width loss (MRL: no degradation for the full,
   per-head variant). Cheaper than independent nets. Caveat: the shared *hidden units* still lack stable
   identity — the per-width heads absorb the readout-entanglement but may only partially fix the conflict.
   Must be tested, not assumed.

3. **Ordered-dropout + explicit ordering/orthogonality pressure — weakest alone.** Literally porting the
   low-rank mechanism (prefix-dropout + asymmetric exposure) is close to what we already did (k-dropout
   sandwich) and failed, because the SVD backstop is absent for a nonlinear MLP. Only worth trying combined
   with a feature-orthogonality regularizer, and even then speculative.

**Portable trick (from Pillar 2):** *moment-matching* — fold "dropped" capacity into a fallback term so every
prefix is a *calibrated* model by construction (rather than a mis-calibrated truncation). An NN analog (fold
dropped-unit contribution into a bias/fallback rather than hard-zeroing) may make every width valid by design.

**Recommendation:** primary = **frozen residual cascade** (only candidate that forces all three invariant
properties AND generalizes past linearity); fallback = **Matryoshka per-width heads** for efficiency. Test
both against the existing `make_hetero` toy + the convergence gate + the 3 bars we already built, directly
comparable to the W_INDEP / W_CONVERGED / W_KDROPOUT_CONVERGED results.

*Extra weight for the cascade (from the Pillar 2 critique):* the low-rank ordering is **emergent and never
column-level-verified** — an evidentiary gap and a possible fragility (weak middle-column pressure, untested
r_max scaling). The cascade's ordering is **explicit and guaranteed by construction** (greedy residual-fit ⇒
strictly decreasing importance), so it sidesteps exactly that gap. Conversely, this is another mark against
candidate 3 (porting the low-rank exposure mechanism to a nonlinear MLP would inherit the unverified/fragile
ordering with none of the SVD backstop).

### Open verification items before building
- ~~Confirm whether the shipped ProbReg recipe-of-record *training* inherits recompute-per-k bins
  (`probabilistic_regression.py:240-244`); if so, only the post-hoc selector rescues it.~~
  **RESOLVED 2026-07-11: it does not** (REGRESSION_ONLY disables the CE branch — see the Pillar-1
  KEY FINDING block above and `cascade_execution_plan_2026-07-11.md` §3.1).
- The independent-width result is the "each rung self-contained" extreme (works, K× params); the cascade is
  the *efficient* version of the same principle — validate it recovers the dial at ~1× params.
- **Measure the induced ordering, don't assume it** (the gap the low-rank program left): add a per-rung
  contribution-norm + cross-seed stability diagnostic to whatever we build, so "the prefix is
  importance-ordered" is a *measured* result, not an argued one.

### Incidental (non-nesting) findings surfaced during the research — for owner triage
- **Likely bug (cheap fix):** `get_classifier_predictions` binary-prob uses `sigmoid(logit0)` vs the correct
  `sigmoid(logit1−logit0)` (`probabilistic_regression.py:796-798`); latent/untested, diagnostic-only.
- **Silent failure risk:** `create_bins` silently shrinks k on tied/discrete targets (`numerics.py:37-43`).
- **Hygiene:** inert HPO dims; unconditional dead centroid computation in the default config.
