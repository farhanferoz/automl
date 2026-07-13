# Frozen residual cascade vs Matryoshka heads — firmed mathematics + execution plan (2026-07-11)

**Status.** This is the build plan that follows the research record
(`docs/plans/width_dial_2026-07-11/nested_architecture_research_2026-07-11.md`). §1 is the review
verdict on that record (what stands, what is corrected). §2 firms up the mathematics. §3 records what
the verification pass uncovered about ProbReg and the low-rank ladder. §4 is the decision-complete
execution plan (buildable by a lower-capability model with no judgment calls left). §5 is routing/cost.

---

## 1. Review verdict on the research record

**Overall: the diagnosis and the recommendation stand.** The coherence invariant (stable identity +
self-contained + importance-ordered), the scorecard, and the "linearity is the hidden enabler" reading
are consistent with the code, the literature quotes, and our own W1/W2/W_INDEP/W_CONVERGED results.
Primary = frozen residual cascade, fallback = Matryoshka per-width heads is the right ordering.

**Corrections / firmings applied in this plan:**

1. **Overclaim fixed — "greedy residual-reduction ⇒ decreasing importance" is not a theorem.**
   Greedy stagewise fitting does NOT guarantee decreasing marginal gains (a later block can contribute
   more than an earlier one if the earlier stage under-fit). What IS guaranteed, with the zero-init +
   acceptance rule of §2.3, is a **monotone non-increasing held-out NLL across rungs** (weak ordering).
   The strong form (decreasing ΔNLL per rung) is a *measured diagnostic*, never assumed — exactly the
   discipline the low-rank ladder skipped (its column-level ordering was specified but never measured).
2. **Two missing precedents added (verified, §2.6).** NGBoost — stagewise *additive updates in the
   Gaussian's parameter space (μ, log σ)*, i.e. the cascade's exact algebra, published and standard.
   Cascade-Correlation (Fahlman & Lebiere 1990) — grow capacity one unit at a time, train it against
   the residual, then **freeze it forever**: the cascade's exact architecture move. The research doc's
   only precedent was boosting-in-general; these two make both halves (parameter-space additivity;
   frozen greedy width growth) individually standard rather than novel-and-risky.
3. **Missing risk added — stagewise sharpening of the heteroscedastic-NLL pathology (§2.5).** The
   moment-matching property that makes every prefix calibrated also means the predictive variance is
   inflated exactly where signal remains unexplained; the mean-fit gradient is attenuated by 1/σ²
   there (Seitzer 2022). Staging *starts* each new block at those inflated values. Pre-registered
   mitigation (β-NLL per stage) + trigger, rather than discovering this mid-run.
4. **The "portable trick" is stronger than the doc says.** Moment-matching does not need to be ported —
   for a heteroscedastic Gaussian NLL cascade it is *automatic* (one-line stationarity condition,
   §2.4). This is the precise sense in which the cascade is "the low-rank ladder generalized past
   linearity": same fold-the-dropped-capacity-into-the-variance structure, emergent instead of
   hand-built.
5. **Matryoshka arm under-specified in the doc** — rung-loss weights, convergence gating, and the
   identity caveat are pinned down in §2.7/§4 so the builder makes no choices.
6. **ProbReg dynamic-k "it works" needs the precise statement** — see §3: what actually carries the
   validated per-input-k result is the K4 surrogate + post-hoc selector, not the shipped masked-prefix
   training path, which re-bins per k (node-0 conflict). Verified against current code (annex §3.1).

---

## 2. Firmed mathematics

### 2.0 Setting and notation

Scalar input x (standardized on train stats), scalar target y (standardized likewise — so the
marginal of y is ≈ N(0,1) in training units). All models are heteroscedastic Gaussians: a predictor
returns (μ(x), s(x)) with predictive density N(μ(x), e^{s(x)}); per-example NLL

    ℓ(μ, s; y) = ½ [ log 2π + s + (y − μ)² e^{−s} ].

Held-out mean log-likelihood is the ONLY bar (doctrine: strictly probabilistic, no penalties/λ).
Noise floor at σ = 0.05 in raw units; in standardized units the oracle LL is computed exactly as in
the existing drivers (reuse — do not rederive).

### 2.1 The cascade

K blocks, each a single-hidden-layer tanh subnet of width 1. Block b has one hidden unit
h_b(x) = tanh(u_b x + c_b) and two scalar readouts:

    δμ_b(x) = a_b · h_b(x) + d_b        δs_b(x) = a′_b · h_b(x) + d′_b

The **rung-k (prefix-k) model** is the additive composition

    μ_k(x) = Σ_{b≤k} δμ_b(x)           s_k(x) = Σ_{b≤k} δs_b(x)

Rung 0 (no blocks) is (μ, s) = (0, 0) = N(0,1) — exactly the standardized marginal, the analog of the
rank ladder's diagonal-only rung. Inference at rung k runs blocks 1..k only (cheap prefix inference).

**Additivity is in the distribution's parameter space (μ and log σ²)** — additive μ, multiplicative
σ². This is NGBoost's parametrization (§2.6), keeps σ² > 0 unconstrained, and makes every prefix a
valid Gaussian by construction.

### 2.2 Lemma 1 — function class (cascade = nested net + per-prefix bias; training scheme is the only real change)

A sum of k width-1 tanh blocks is exactly a width-k single-hidden-layer tanh network:
μ_k(x) = Σ_{b≤k} a_b tanh(u_b x + c_b) + (Σ_{b≤k} d_b), and identically for s_k. So rung k's function
class equals `NestedWidthNet`'s width-k class, with ONE extra freedom: the nested net shares a single
readout bias across all widths, while the cascade's rung-k bias is Σ_{b≤k} d_b — an independent bias
per prefix (2k scalars total; negligible params, real expressiveness at small k).

Consequence: the cascade is **not a new architecture family** — it is the nested-width net with (a) a
staged, frozen training scheme and (b) per-prefix readout biases. This satisfies the standing decision
that "nesting changes the TRAINING SCHEME only", and makes the comparison to W1/W2/W_CONVERGED an
apples-to-apples training-scheme contrast at matched capacity (rung k ↔ width k, k = 1..12).

### 2.3 Lemma 2 — the monotone ladder (zero-init + acceptance rule)

Training protocol for stage b (blocks < b frozen, their outputs precomputed/cached):

1. **Zero-init** block b's readouts (a_b = d_b = a′_b = d′_b = 0; hidden weights random). Then the
   stage-b model at initialization is EXACTLY the rung-(b−1) model.
2. Train **only block b's parameters** to convergence on the NLL of the full prefix-b model
   (convergence gate: `fit_to_convergence`, best-val weights kept — the standing trajectory rule).
3. **Acceptance rule**: accept block b iff its converged best held-out-val NLL improves on rung (b−1)
   by more than `min_delta`; otherwise reset its readouts to zero (rung b becomes *inert*: identical
   to rung b−1) and record `accepted=False`.

Then by construction: **NLL_val(rung k) is non-increasing in k**, every rung is a valid calibrated
model (it literally *was* the trained model at the end of its stage), and rung identity is stable
forever after (frozen). This delivers the coherence invariant in its guaranteed (weak) form:

- (1) stable identity — frozen ⇒ block b never changes after stage b;
- (2) self-contained — each block is a complete additive correction; each prefix a complete model;
- (3) importance ordering — guaranteed as monotone improvement; the strong form (ΔNLL decreasing in b)
  is a **measured diagnostic** (§4.5), not an assumption.

Multi-restart: each stage trains R = 3 random restarts of block b (fresh hidden-weight init; readouts
zero) and keeps the best-val one. Rationale: a single tanh unit against a multi-modal residual is
seed-sensitive (the W2 probe already flagged keep-best-of-2–3 restarts at high widths); blocks are
tiny and prefixes cached, so restarts are nearly free.

### 2.4 Proposition — emergent moment-matching (the low-rank ladder's trick, for free)

For fixed μ, ℓ is minimized pointwise in s where ∂ℓ/∂s = ½(1 − (y−μ)² e^{−s}) = 0, i.e.
e^{s(x)} = E[(y − μ(x))² | x]. With y = f(x) + ε, ε ~ N(0, σ_ε²):

    σ²_k(x) → σ_ε² + (f(x) − μ_k(x))²   (at each stage's optimum, given block capacity)

So the variance head **automatically folds the not-yet-explained signal into the predictive
variance**. This is precisely the rank ladder's moment-matching (Σ_r = D + diag(Σ_{k>r} m_k m_kᵀ) +
Σ_{k≤r} m_k m_kᵀ — dropped columns folded into the diagonal so every rung has calibrated marginals) —
but here it is *emergent from stagewise NLL optimization*, not hand-built. Every accepted prefix is a
calibrated model at its own capacity **by training**, not a mis-calibrated truncation.

Ladder ↔ cascade correspondence (the "inspired by how the low-rank ladder works" mapping, made exact):

| Low-rank ladder | Cascade | Same? |
|---|---|---|
| column m_b (rank-1 factor) | block b (one tanh unit, additive (δμ, δs)) | additive self-contained rung |
| prefix truncation Σ_{b≤r} m_b m_bᵀ | prefix sum Σ_{b≤k} (δμ_b, δs_b) | prefix = smaller valid model |
| moment-match: fold dropped columns into diag | variance head absorbs unexplained residual (∂ℓ/∂s = 0) | same structure; hand-built vs emergent |
| ordering via prefix-dropout + 89→11% exposure (emergent, **never measured**) | ordering via freeze + greedy residual fit (explicit) + acceptance rule (guaranteed weak form) | cascade strictly stronger here |
| rotation-degeneracy broken by prefix scoring (LINEAR argument) | not available for a nonlinear MLP — replaced by construction | the linearity barrier, sidestepped |

Where the cascade is *weaker*: greedy stagewise can undershoot the jointly-trained width-K optimum at
full capacity (boosting is greedy). That is exactly what the ANCHOR diagnostic (§4.5) measures against
the dedicated-net W_CONVERGED results — same data, same seeds, directly comparable.

### 2.5 Risk — stagewise sharpening of the heteroscedastic-NLL pathology, and the pre-registered mitigation

∂ℓ/∂μ = −(y − μ) e^{−s}: the mean-fit gradient is scaled by 1/σ². By §2.4, after stage k the variance
is largest exactly where residual signal remains. Stage k+1 *starts* at those inflated values
(zero-init ⇒ s init = s_k), so its mean gradient is attenuated by e^{−s_k} precisely on the region it
exists to fix. This is Seitzer et al.'s pitfall (β-NLL, ICLR 2022 — verified §2.6), arguably sharpened
by staging relative to joint training (where σ² starts at init values, not at correctly-inflated ones).

**Mitigation (pre-registered, not improvised mid-run):** default training loss is plain NLL (for
comparability with every W battery). **Escalation trigger:** if the ANCHOR diagnostic shows the
cascade's rung-12 hard-region held-out LL more than 0.10 nat below the dedicated w12 anchor
(per-seed), re-run the cascade arm with per-stage **β-NLL, β = 0.5**: per-point loss
`(e^{s})^β · ℓ` with the `(e^{s})^β` factor **detached** (stop-gradient), Seitzer's estimator. The
bars are always evaluated with plain held-out LL regardless of the training loss (the bar never moves).

### 2.6 Literature anchors (verified this session — full quotes in `scratchpad` lit report, durable copy in §3.3)

- **NGBoost** (Duan et al., ICML 2020, arXiv:1910.03225): gradient boosting with **additive stagewise
  updates to the distribution's parameters** — for Normal, (μ, log σ) — fitted by natural gradient of a
  proper scoring rule. Prefix-validity of the parameter-space ensemble is structural there, same as here.
- **Cascade-Correlation** (Fahlman & Lebiere, NIPS 1990): add one hidden unit at a time; train it
  against the current residual (correlation objective); then **freeze its input weights permanently**.
  The classic precedent for frozen greedy width growth. (Difference: we freeze the whole block including
  its readout deltas and re-train nothing downstream; CasCor keeps re-training the output layer. Our
  freeze is *stricter*, which is what guarantees rung identity — noted so nobody "fixes" it to CasCor.)
- **β-NLL** (Seitzer et al., ICLR 2022, arXiv:2203.09168): the 1/σ² mean-gradient attenuation pathology
  and the `σ^{2β}`-weighted fix, β = 0.5 recommended. Already in-repo for ProbReg (`loss_type="beta_nll"`).
- **Anytime/nested prefix-calibrated boosting**: no primary source found (closest: SpeedBoost, Grubb &
  Bagnell, AISTATS 2012 — anytime boosting, point-prediction only) — treating every boosting prefix as
  a *deployable, calibrated, per-input-selected* model appears to be the novel piece of this program.

### 2.7 The Matryoshka arm (fallback), pinned down

`MatryoshkaWidthNet`: shared trunk `Linear(1 → w_max) → tanh` giving h(x) ∈ R^{w_max}; for EACH rung
k = 1..w_max a dedicated pair of heads `mean_head_k, logvar_head_k = Linear(k → 1)` applied to the
prefix h[:, :k]. Prefix inference at rung k computes k hidden units + one (k→1) head — still cheap.

Training: **joint**, one optimizer over everything; per step the loss is the **unweighted sum of all
w_max rung NLLs** (c_k ≡ 1; MRL's relative-importance weights c_m are an unexplored dial there — we
pre-register uniform and note it as an assumption). No BN anywhere (tanh MLP), so the slimmable
private-BN issue does not arise. Convergence: per-rung `ConvergenceTracker` on each rung's own val
NLL; the loop stops when ALL rungs' trackers are done (or cap) — mirror
`kdropout_converged_width_experiment.py`'s joint-training gating exactly.

What it does and does not fix: the per-rung heads remove the **shared-readout entanglement** (each
rung gets its own readout basis — the mechanism MRL's no-degradation result rides on). They do NOT
give the trunk's hidden units stable identity or ordering (all rungs' gradients hit the shared trunk
jointly, forever). So the invariant scorecard is: (2) partially fixed, (1)/(3) not — which is exactly
why this is the fallback and the comparison is informative: **cascade vs Matryoshka isolates whether
identity+ordering matter beyond readout disentanglement.**

### 2.8 Accounting (w_max = K = 12, 1-D input)

| Arm | Params (≈) | Rung-k inference | Invariant (1)/(2)/(3) |
|---|---|---|---|
| Nested shared (W2, failed) | 4W+2 ≈ 50 | k units, shared readout | ✗/✗/✗ |
| **Cascade** | 6K ≈ 72 | k units + k bias-adds | ✓/✓/✓(weak-guaranteed, strong-measured) |
| **Matryoshka** | (4W+2) + Σ_k 2(k+1) ≈ 230 | k units + one (k→1) head | ✗/~✓/✗ |
| Independent (W_INDEP/W_CONVERGED) | Σ_k (4k+2) ≈ 336 | k units (own net) | ✓/✓/n·a |

All tiny in absolute terms on this toy — the point of the table is the *scaling shape* (cascade ≈ 1×
a single net; independent ≈ K×/2 quadratic) and the invariant column.

---

## 3. What the verification pass uncovered (ProbReg + low-rank ladder deficiencies)

### 3.1 ProbReg dynamic-k — the precise status of "it works" (verified 2026-07-11)

Every research-record claim (1–12) re-verified against current code; full table with verbatim quotes:
`docs/plans/width_dial_2026-07-11/annex_probreg_verification_2026-07-11.md`. The owner's question was
"per-input k works, I think — verify that." The precise answer has three parts:

1. **The identity defect is real and in the shipped code.** Dynamic-k training *with classification
   cross-entropy active* builds each sample's CE target from k-specific, non-nested percentile
   boundaries: precomputed once per candidate k (`probabilistic_regression.py:504-506`), looked up per
   sample by whatever k that sample's gate selected on that forward pass (`:237-244`, via
   `base_selection_strategy.py:95-100`). The selected k varies per sample and per step (the gate is
   itself training), so logit/head 0 is supervised as "bottom 1/k of y" for a k that keeps changing —
   the node-0 conflict, exactly as the research record stated. (One precision vs the record's wording:
   boundaries are precomputed once per k, NOT recomputed per step; what varies per step is which k's
   boundary set applies to which sample. The conflict statement is unaffected.)
2. **The validated recipe of record does NOT inherit the defect.** Every H1 arm sets
   `optimization_strategy=REGRESSION_ONLY` (`capacity_ladder_h1.py:99`), which makes `is_ce_active`
   False (`probabilistic_regression.py:219-222`) — the CE/boundary-lookup branch structurally never
   fires. Cross-k component identity in that path is carried instead by the ONE shared classifier +
   fixed head objects under masked-prefix softmax (`probabilistic_regression_net.py:132-149`) — the
   same stable-identity mechanism as the K4 surrogate (annex trace D). So "per-input k works" is TRUE
   for the recipe of record (likelihood-only training + post-hoc selector), and the coherence
   invariant explains WHY: with CE off, "class i" is never redefined per k.
3. **Scope of the residual defect + fix shape (user-gated, NOT in this build).** Any dynamic-k
   configuration that activates CE (optimization strategies other than REGRESSION_ONLY /
   GRADIENT_STOP) carries the conflict. Coherent fixes, for a later decision: k-stable binning
   (refinement/dyadic bins, or classes anchored to the max-k grid) or documenting that dynamic-k
   requires likelihood-only training.

Incidental verified defects (unchanged from the record, confirmed at current lines):
`get_classifier_predictions` binary case uses `sigmoid(logit0)` instead of `sigmoid(logit1−logit0)`
(`probabilistic_regression.py:796-798`; diagnostic-only — grep confirms no internal caller);
`create_bins` silently shrinks k on tied targets (`numerics.py:37-43`); inert HPO dims added
unconditionally + dead centroid computation in the default config (`:406-435`, `:526-530`).

### 3.2 Low-rank ladder deficiencies (stand as researched; action items)

- **Ordering never measured at the column level** — the design's own `‖m_k‖` / cross-seed
  `subspace_r2` diagnostic was specified but never reported for the trained arm. The cascade plan
  closes this gap on OUR side (§4.5 ordering diagnostics are mandatory outputs); the corresponding
  action for the autocast program (run their specified diagnostic on the trained A3 loading) is THEIR
  backlog — **FILED 2026-07-11 as task ORD-1** in
  `/home/ff235/dev/turing/autocast-private/notes/design/rank_ladder/rank_ladder_execution_plan.md`
  (+ pointer in that repo's RESUME.md), sequenced before their REPORT-PART-I asserts ordering.
- **The ordering mechanism is a linear argument** (rotation-degeneracy + exposure gradient) — fine for
  covariance, void for MLPs; anyone porting rank-dropout to a nonlinear net inherits the fragility
  with none of the SVD backstop (this killed candidate 3).
- **Untested r_max scaling** (per-column exposure pressure (r_max−k)/(r_max+1) flattens as r_max
  grows) — if the ladder is ever widened, ordering fidelity should be re-measured, not assumed.

### 3.3 Literature verification outcomes (verified 2026-07-11, full-text PDFs; quotes in annex)

Durable copy with verbatim quotes: `docs/plans/width_dial_2026-07-11/annex_literature_verification_2026-07-11.md`.

- **NGBoost — CONFIRMED** (Duan et al., ICML 2020, arXiv:1910.03225). Additive stagewise updates in the
  distribution's parameter space — Normal: two base learners per stage, f_μ and f_{log σ}, natural
  gradient of a proper scoring rule, θ = θ⁰ − η Σ_m ρ_m f_m(x). Prefix-validity is a *structural
  byproduct* the paper never discusses or evaluates — so our per-input use of prefixes is new ground,
  not covered territory.
- **Cascade-Correlation — CONFIRMED precisely** (Fahlman & Lebiere, NIPS 1989 conf. / 1990 proceedings,
  pp. 524–532). "Once a new hidden unit has been added to the network, its input-side weights are
  frozen"; candidate units trained to maximize correlation (covariance) with the residual error; only
  output-layer weights keep re-training. Our cascade's freeze is STRICTER (whole block, readout deltas
  included) — deliberate, because full freezing is what guarantees rung identity (§2.3); do not
  "correct" it to CasCor's partial freeze.
- **β-NLL — CONFIRMED** (Seitzer et al., ICLR 2022, arXiv:2203.09168). ∇_μ NLL scales the error by
  1/σ² (their Eq. 3) → "rich get richer" under-fitting of poorly-fit points; β-NLL multiplies the
  per-point NLL by stop-gradient(σ^{2β}), β = 0.5 recommended. Nuance kept from the paper: the causal
  chain is error → variance shrinks/stays high to match → gradient attenuates — which is exactly the
  stagewise mechanism in §2.5 (our prefix variance is *correctly* high where residual signal lives).
- **Anytime prefix-calibrated distributional boosting — NOT FOUND** (medium confidence, targeted
  search). Closest: SpeedBoost (Grubb & Bagnell, AISTATS 2012) — genuine anytime boosting with
  per-stage near-optimality, but point prediction only. Post-hoc boosting-calibration work
  (Niculescu-Mizil & Caruana) calibrates finished ensembles, not prefixes. Treating every prefix of a
  distributional ensemble as a deployable, calibrated, per-input-selected model appears to be the
  novel piece of this program.

---

## 4. Execution plan (decision-complete — no judgment left to the builder)

**Doctrine (bind, unchanged):** strictly probabilistic (per-example Gaussian log-likelihood; no
MSE-only, no penalty/λ); held-out LL is the only bar; frozen model → post-hoc distilled selector
(two-stage); examples-level only, **NO library edits**; depth fixed at a single hidden layer — the
only capacity axis is width/rungs; convergence-gated training everywhere (`convergence.py`), no fixed
epoch counts; NO conclusion from a non-`trustworthy` run.

**Data / split / seeds (identical to `converged_width_experiment.py` — reuse, don't rewrite):**
`make_hetero(1500, seed)` train + `make_hetero(500, seed+500)` test; seeds (0, 1, 2); index-parity
p1/p2 split of train (p1 = phase-1 fit, p2 = selector distillation); within p1, every 5th point is
the convergence-val split; standardize x AND y on the p1-train stats only. w_max = K = 12. LR 1e-2
Adam. Noise σ = 0.05.

### 4.1 File 1 — `automl_package/examples/cascade_width_net.py` (new)

`ResidualCascadeNet(nn.Module)`:
- `__init__(self, w_max: int = 12, activation: type[nn.Module] = nn.Tanh)`: builds
  `self.blocks: nn.ModuleList`, block b (0-indexed) =
  `nn.ModuleDict({"trunk": nn.Linear(1, 1), "act": activation(), "mean_head": nn.Linear(1, 1), "logvar_head": nn.Linear(1, 1)})`.
  Zero-init EVERY block's `mean_head`/`logvar_head` weight AND bias (trunk init: default).
  Expose `self.w_max`.
- `block_contrib(self, x, b) -> (dmu, ds)`: block b's `(N,1)` readout pair.
- `forward_width(self, x, k) -> (mean, log_var)`: `sum(block_contrib(x, b) for b < k)`; validate
  `1 <= k <= w_max` (ValueError, same message style as `NestedWidthNet.forward_width`).
- `all_widths_forward(self, x) -> (mean_all, logvar_all)`, each `(N, w_max)`: stack per-block
  contributions `(N, w_max)` then `torch.cumsum(dim=1)`. Column k−1 == `forward_width(x, k)`.
  (Same interface contract as `NestedWidthNet` — this is what makes ALL of
  `sinc_width_experiment`'s scoring/selector/bars reusable verbatim.)
- `freeze_blocks_below(self, b)`: `requires_grad_(False)` on all blocks index < b.
- Docstring: cite this plan §2.1–2.3 and mirror the tone/structure of `nested_width_net.py`.

`train_cascade(net, x_tr, y_tr, x_val, y_val, *, restarts=3, max_epochs, check_every, patience,
min_delta) -> dict[int, cvg.ConvergenceResult]` (module-level function):
- For b = 1..w_max (stage b trains block index b−1):
  - Cache the frozen prefix on train and val: `mu_prev, s_prev` from `forward_width(·, b−1)`
    (zeros for b=1) under `no_grad`. Compute `val_nll_prev` (for b=1 this is the N(0,1) marginal NLL).
  - For each of `restarts` restarts: re-init block b's trunk (fresh `torch.nn.init` default via
    re-instantiating the Linear, seeded `torch.manual_seed(seed*1000 + b*10 + restart)`), zero its
    readouts; Adam over block-b params only; `step_fn` = one full-batch step on
    `nll(mu_prev + dmu_b, s_prev + ds_b)` (uses the CACHED prefix tensors — do not re-run frozen
    blocks); `val_loss_fn` likewise on the cached val prefix; run `cvg.fit_to_convergence`. Keep the
    restart with the lowest `best_val`.
  - **Acceptance rule:** if `best_val >= val_nll_prev - cvg.DEFAULT_MIN_DELTA`: reset block b's
    readouts to zero (inert rung), `accepted=False`; else keep best-restart weights, `accepted=True`.
  - `freeze_blocks_below(b+1)`; record the winning `ConvergenceResult` + `accepted` + `val_nll_prev`.
- Gaussian NLL: import `gaussian_log_likelihood` from `nested_width_net` (do NOT re-implement).
- `--beta-nll` flag (default off): multiply per-point NLL by `torch.exp(s_total).pow(0.5).detach()`
  inside the stage loss (β = 0.5). Only used if the §2.5 escalation trigger fires — wire it now so the
  escalation is a rerun, not a code change.

**Selftest (`--selftest`, no training)** — mirror `nested_width_net.py`'s style; all on random init:
  (a) zero-init ⇒ `forward_width(x, k)` returns exactly `(0, 0)` for every k (the N(0,1) rung-0
      property propagates through all-zero readouts);
  (b) after randomizing readouts: `all_widths_forward` column k−1 == `forward_width(x, k)` (tol 1e-5),
      and prefix property — rung-k output invariant to perturbing blocks ≥ k (tol 1e-6);
  (c) freeze check — after `freeze_blocks_below(b)`, a backward pass through rung-w_max NLL leaves
      `grad is None or grad == 0` for every frozen-block parameter;
  (d) stage-start identity — with block b's readouts zeroed, rung-b output == rung-(b−1) output exactly;
  (e) acceptance-reset — after writing garbage into block b's readouts then applying the reset, rung-b
      == rung-(b−1) again.

### 4.2 File 2 — `automl_package/examples/matryoshka_width_net.py` (new)

`MatryoshkaWidthNet(nn.Module)`: `trunk = nn.Linear(1, w_max)`, `act = activation()`,
`mean_heads`/`logvar_heads` = `nn.ModuleList([nn.Linear(k, 1) for k in 1..w_max])`.
- `forward_width(x, k)`: `h = act(trunk(x))[:, :k]` → `(mean_heads[k-1](h), logvar_heads[k-1](h))`.
- `all_widths_forward(x)`: loop k (12 small matmuls; no cumsum trick — heads differ per rung), return
  `(N, w_max)` pair. Same interface contract as above.
- Selftest (`--selftest`): (a) rung-k output invariant to perturbing trunk columns ≥ k AND heads ≠ k;
  (b) `all_widths_forward` == per-k `forward_width` (tol 1e-5); (c) finite shapes.

`train_matryoshka(net, x_tr, y_tr, x_val, y_val, *, max_epochs, check_every, patience, min_delta)
-> dict[int, cvg.ConvergenceResult]`: joint full-batch Adam over ALL params; per step, loss =
`sum over k=1..w_max of mean Gaussian NLL at rung k` (unweighted, §2.7); every `check_every` epochs
update a per-rung `cvg.ConvergenceTracker` with that rung's val NLL (verdicts/trajectories only);
stop when all trackers `.done` or cap. Loop shape: mirror
`kdropout_converged_width_experiment._train_kdropout_to_convergence` — BUT note the one deliberate
difference: that driver restored each width's own best weights, which is only possible because its
subnets are DISJOINT; the Matryoshka trunk is SHARED across rungs, so per-rung restoration is
ill-defined. Instead keep ONE extra `ConvergenceTracker` on the SUMMED val NLL (all rungs) that
snapshots the whole-net state_dict on improvement, and restore that best-joint snapshot at the end —
one deployed model, best-not-last in the joint objective. Record this in the summary
(`"best_restore": "joint_sum"`) so the comparability caveat vs the disjoint batteries is explicit.

### 4.3 File 3 — `automl_package/examples/cascade_width_experiment.py` (new driver, covers BOTH arms)

Mirror `converged_width_experiment.py`'s structure verbatim (imports, `_standardize_fit`,
`_to_std_tensors`, RESULTS_DIR pattern, `run_case`, `_print_case`, `main` with
`--selftest/--smoke/--config/--max-epochs`), plus `--arm {cascade,matryoshka}` (default: run BOTH
sequentially) and `--beta-nll` passthrough to the cascade arm.

Per seed × arm, `run_case`:
1. Data/split/standardization: copy `converged_width_experiment.run_case` lines exactly.
2. Phase 1: build net (`torch.manual_seed(seed)` first), train via §4.1/§4.2 trainer.
3. Frozen-net scoring + bars + selector: **reuse `sinc_width_experiment` verbatim** —
   `sw._score_all_widths`, `sw._construction_bar(k_lo=1, k_mid=max(2, w_max//2), w_max)`,
   `sw._fit_selector(score_p2, x_p2, w_max, seed, device)`, `sw._selector_eval`,
   `sw._recovery_bar`, `sw._deploy_bar`, `sw._jsonable` — identical call shapes to
   `converged_width_experiment.run_case`.
4. Extra recorded fields (beyond the `converged_width_experiment` case dict):
   - `accepted_rungs` (cascade only): list of bool + count of inert rungs;
   - `delta_ll_per_rung`: held-out TEST ΔLL (rung k vs k−1), overall + per region (easy/hard) —
     rung 0 baseline = N(0,1) LL on standardized y;
   - `anchor`: this arm's rung-12 test NLL minus the dedicated w12 test NLL read from
     `capacity_ladder_results/W_CONVERGED/w_converged_summary.json` (`per_k_nll["12"]`, same seed) —
     load that file; if missing, record `anchor: null` and print a warning (do NOT fail);
   - `beta_nll_used`: bool.
5. Results dirs: `capacity_ladder_results/W_CASCADE/` and `capacity_ladder_results/W_MRL/`;
   summaries `w_cascade_summary.json` / `w_mrl_summary.json` (same schema as
   `w_converged_summary.json` + the extra fields).

**Driver selftest (`--selftest`):** tiny end-to-end wiring run per arm (w_max=3, n=200, cap 1500,
seed 0, no save) asserting: trajectories recorded per rung; `accepted_rungs` present (cascade);
bars dict keys present; `delta_ll_per_rung` has w_max entries. Mirror
`converged_width_experiment.run_selftest`'s assertions + prints.
**Smoke (`--smoke`):** w_max=4, cap 4000, seed 0, both arms, no save; print per-rung val NLL and the
hard/easy LL curves (must show hard LL climbing with rungs; easy roughly flat).

### 4.4 Pre-registration — `capacity_ladder_results/W_CASCADE/PREREGISTRATION.md` (write BEFORE the real run; covers both arms)

Identical bars and thresholds to the W batteries (comparability is the point):
- **Bar (i) CONSTRUCTION**: hard-region held-out LL climbs k_lo=1 → k_mid=6 by > 2·SE and reaches
  near the noise floor; easy region comparatively flat. ≥ 2/3 seeds. (Verbatim `sw._construction_bar`.)
- **Bar (ii) RECOVERY**: selector's expected rung strictly larger on hard than easy, separation
  > 2·SE, ≥ 2/3 seeds. (Verbatim `sw._recovery_bar`.)
- **Bar (iii) DEPLOY**: selector blended held-out LL ≥ best single rung (one-sided) AND within 0.02
  nat of the per-input oracle. ≥ 2/3 seeds. (Verbatim `sw._deploy_bar`; note: every W battery so far
  fails the 0.02-nat oracle sub-clause — the standing recalibration question stays OPEN and is a
  user decision; report the number, don't move the bar.)
- **ANCHOR diagnostic (new, informative — not a pass/fail bar)**: rung-12 test NLL within 0.10 nat of
  the dedicated w12 anchor (per seed). Cascade-arm miss on the HARD region ⇒ the §2.5 β-NLL
  escalation fires (one pre-registered re-run of the cascade arm with `--beta-nll`).
- **ORDERING diagnostics (mandatory outputs, the "measure don't assume" items)**: per-rung ΔLL
  profile overall + by region; count + positions of inert rungs (cascade); cross-seed Spearman
  correlation of the ΔLL-by-rung profile (report; no threshold — first measurement, sets the baseline
  the low-rank program never took).
- Convergence gate: NO conclusion from any seed×arm whose rungs are not all `trustworthy`
  (cascade: every ACCEPTED stage's winning restart; matryoshka: all per-rung trackers); the only
  valid read is "needs more training" (raise `--max-epochs`, rerun that seed).
- Decision rule (pre-registered): if both arms pass i+ii, the arm of record = better deploy LL, ties
  (within 2·SE) broken toward the **cascade** (guaranteed identity/ordering + fewer params). If only
  one passes, it is the arm of record. If neither passes construction, the finding is "per-rung heads
  and frozen additivity do NOT rescue shared-representation nesting" — a real (negative) answer to
  the middle-ground question; fold into the report, do not iterate ad hoc.

### 4.5 Phases + verify commands

- **Phase B — BUILD (TWO Sonnet `task-worker`s, dispatched concurrently — independent units)**:
  worker B1 = the width build below; worker B2 = work package P (§4.7, input contract = §4.7 verbatim
  + the annex + its per-item test/verify commands). For B1: files §4.1–4.4 in one coherent dispatch
  (they share the interface). Input contract = this §4 verbatim + template paths
  (`nested_width_net.py`, `converged_width_experiment.py`, `kdropout_converged_width_experiment.py`,
  `sinc_width_experiment.py`) + non-goals (§4.0 doctrine; don't touch existing files except imports;
  no library edits) + the verify commands below. Worker must run all green and report the numbers:
  ```
  AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/cascade_width_net.py --selftest
  AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/matryoshka_width_net.py --selftest
  AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/cascade_width_experiment.py --selftest
  AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/cascade_width_experiment.py --smoke
  ~/dev/.venv/bin/python -m ruff check automl_package/examples/cascade_width_net.py automl_package/examples/matryoshka_width_net.py automl_package/examples/cascade_width_experiment.py
  ```
- **Phase C — VERIFY (main thread, cheap)**: re-run the five commands, read numbers only (smoke: hard
  LL climbs with rungs, easy flat, no NaNs; selftests PASS; ruff clean).
- **Phase D — RUN (detached)**:
  `setsid nohup systemd-inhibit --what=idle:sleep:handle-lid-switch env AUTOML_DEVICE=cpu
  OMP_NUM_THREADS=4 ~/dev/.venv/bin/python -u automl_package/examples/cascade_width_experiment.py
  > <scratchpad>/w_cascade.log 2>&1 &` — monitor from the MAIN thread by waiting for BOTH summary
  JSONs (never a subagent; watch the files, not a pid).
- **Phase E — INTERPRET (main thread, capable tier)**: bars + ordering diagnostics + anchor per §4.4's
  decision rule; escalation re-run only if the trigger fires; fold results into RESUME; report/commit
  remain user-gated (author as user, no AI provenance).

### 4.6 Non-goals / guards
- The width-experiment files (§4.1–4.3) are examples-level only — they make NO library edits. The ONLY
  library edits in this plan are work package P (§4.7), a separate, independently-dispatched unit.
- No depth coupling (depth lane closed; width/rung is the only axis).
- Small-data / high-noise regime (user's accuracy-payoff hypothesis) is the NEXT experiment after
  this one, same drivers with `n_train` swept — out of scope here.
- No new toys — `make_hetero` unchanged, seeds unchanged (comparability with W_* batteries).

### 4.7 Work package P — ProbReg fixes (library edits; user-authorized 2026-07-11)

Independent of the width build — dispatched as its OWN worker, in parallel with Phase B. Library
edits allowed ONLY within this scope. Every fix carries a test in the existing ProbReg test file
(locate it under `tests/` by glob; extend, don't create a parallel file). Run only the relevant test
file(s) + `ruff check` on touched files — not the full suite. All the file:line anchors below are
verified in `annex_probreg_verification_2026-07-11.md`.

- **P1 — binary classifier-probability bug** (`probabilistic_regression.py:796-798`).
  `get_classifier_predictions` 2-class case returns `sigmoid(logits[:, 0])`; the correct 2-class
  softmax positive probability is `sigmoid(logit1 − logit0)`. Fix by DELETING the binary special case
  and letting the general masked-softmax path handle every k ≥ 2 (one code path, no special-casing).
  Test: crafted logits where logit1 ≠ 0 — assert rows sum to 1 and match `softmax` exactly for k=2
  and k=3 (the old code fails the k=2 assertion).
- **P2 — silent bin-count shrink** (`utils/numerics.py:37-43`). When the tied-edge dedup loop reduces
  `n_bins` below the request, emit ONE `logger.warning` (module logger, matching the repo's existing
  logging convention — grep `utils/` for the pattern) stating requested vs effective bins and the
  cause (tied percentile edges on discrete/tied targets). Warning lives in `create_bins` itself so
  every caller (incl. `probabilistic_regression.py:504-506`) is covered. Test: tied target array,
  request k=5 → `caplog` captures the warning and effective bins < 5; clean continuous target → no
  warning.
- **P3 — guard the incoherent dynamic-k + CE combination** (the §3.1 structural defect). At fit time,
  if `n_classes_selection_method != NONE` AND the optimization strategy has CE active (`is_ce_active`
  logic at `probabilistic_regression.py:219-222`), emit ONE loud `logger.warning`: per-k re-binned CE
  targets redefine class identity across k (node-0 conflict); the validated mode is
  `REGRESSION_ONLY`; cite this plan §3.1. Also add the same caveat to the class docstring.
  **Decision (reversible default, logged here):** warning, NOT a `ValueError` — existing
  examples/ablations may exercise the combo deliberately; escalating to a hard error is a later user
  call. Test: `caplog` warning fires for the combo, absent under `REGRESSION_ONLY`.
- **P4 — hygiene** (`probabilistic_regression.py:406-435`, `:526-530`). Gate the `gumbel_tau` and
  `n_classes_predictor_learning_rate` HPO dimensions on the selection method/strategy that actually
  consumes them (inert otherwise); gate the per-class centroid computation on
  (monotonic constraint active OR anchored heads active) — today it runs unconditionally in the
  default config and is never read. Tests: HPO space for a `NONE`-selection config lacks the gated
  dims; default-config fit path skips centroid computation (assert attribute absent/None) while a
  monotonic-head config still gets it.
- **P5 — NOT in this package (flagged, user-gated):** the real redesign that would make CE coherent
  under dynamic-k — a k-stable class semantics (single max-k percentile grid anchoring CE on the full
  model only, with prefixes trained by likelihood; or refinement/dyadic bins). Research-shaped, needs
  its own validation battery; do not start.

## 5. Routing & cost
Build = TWO Sonnet task-workers dispatched concurrently (§4.5 Phase B: B1 width build, B2 ProbReg
work package P — independent files, no shared state). Verify/interpret = main thread. Compute:
3 seeds × 2 arms of tiny full-batch nets on CPU ≈ minutes–low tens of minutes (cascade stages train
~6 params each on cached prefixes; matryoshka joint run ≈ one W_KDROPOUT-scale run). Research
verification this session: 2 Sonnet workers (code-claims sweep; literature anchors).
