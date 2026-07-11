# X1 pre-registration — hierarchical partial-pooled per-bin stacking

**Task:** EXECUTION_PLAN §8.5 X1 (highest-value new experiment) / W7 (§8.2). REPORT-2's
"per-bin stacking = hierarchical stacking" was wrong: `_capacity_ladder.perbin_stack` runs
`stack_em` **independently** per bin — no partial pooling, the defining Yao-2022 safeguard. With
~83 held-out points per tercile the independent per-bin stack is high-variance, which is exactly
what left X3's per-input depth read at the noise floor (the adjudicator's certified read: "route
to X1/X10 — the power fix"). X1 supplies the principled fix: a hierarchical prior that lets the
DATA decide how much each bin borrows strength from the global stack, and asks whether a per-bin
signal emerges once the estimator is regularized — or whether the signal is genuinely absent.

## Instrument (the one new estimator)

`hierarchical_perbin_stack(score, bins)` — drop-in for `perbin_stack`, returns `{bin: π_b}`.
Per-bin weights are unconstrained logits θ_b ∈ ℝ^C (π_b = softmax(θ_b)) with a hierarchical
Gaussian prior in logit space:

    θ_b ~ Normal(μ, τ²·I),     π_b = softmax(θ_b)

fit by MAP-EM on the **fit half only**:
- **θ-step:** gradient ascent on Σ_b Σ_{i∈b} logsumexp_c(logπ_{b,c} + score[i,c]) − Σ_b‖θ_b−μ‖²/(2τ²)
  (data log-score + the prior's own term, coefficient 1). Analytic gradient
  ∂/∂θ_b = Σ_{i∈b}(q_i − π_b) − (θ_b−μ)/τ², q_i = softmax_c(logπ_{b,c}+score[i,c]).
- **μ-step (closed form):** μ = mean_b θ_b.
- **τ²-step (empirical Bayes, closed form):** τ² = mean_{b,c}(θ_{b,c}−μ_c)², floored — a global
  low-dim nuisance FITTED by evidence (§0 B6 MacKay/Minka family), **not a tuned λ**. τ→0 pools to
  the global stack; τ large frees the bins. Init θ_b at the global-stack logits.

Strictly probabilistic (§0b): the prior is the model's own term (coeff 1); τ² is estimated, no
hand-weighted penalty. This is the method the plan itself names for X1 ("hierarchical prior on
per-bin log-weights, MAP-EM").

## Protocol (reuses the X3 repeated-cross-fit harness verbatim)

Same F2 nested-depth tables and toys as F3/X3 (G = varying required capacity, G_flat = uniform
control, H = SNR dial), 3 seeds. Per (toy, seed), over **S=50 random 50/50 fit/score splits**
(the X3 fix for single-split fragility): fit global π, independent per-bin π_b (`perbin_stack`),
and hierarchical per-bin π_b on the fit half; on the held-out half compute per-example mixture
log-score for each; form advantages vs global — `indep_adv = indep_ls − global_ls`,
`hier_adv = hier_ls − global_ls`, and the head-to-head `hier_vs_indep = hier_ls − indep_ls`.
Aggregate each per (toy, seed) with the **Nadeau–Bengio** split-aware SE (as X3), report corrected-t
and split-pass-fraction; pool across seeds.

## Pre-registered bars

- **X1-pooling-helps (primary):** on G, hierarchical per-bin **beats independent per-bin** on
  held-out mixture log-score — `hier_vs_indep` mean > 0 with corrected-t > 2 on a majority of seeds.
  *Prediction: PASS — regularization should reduce the per-bin overfit at ~83 pts/bin.*
- **X1-signal-recovery (the discriminator the adjudicator asked for):** does pooling lift G's
  per-input advantage above the noise floor? `hier_adv` corrected-t > 2 AND clearly above the G_flat
  control's, on a majority of seeds.
  - **PASS → the per-input depth signal was POWER-LIMITED, not absent** (revises X3's read from
    "closed sub-lane" to "recoverable with pooling"; re-opens F4 with the pooled reader).
  - **FAIL (hier_adv ≈ G_flat) → the signal is genuinely ABSENT** — pooling helps the estimator
    (X1-pooling-helps can still pass) but there is no per-input structure to recover; X3's closure
    stands, now on the best available estimator. *Prediction: uncertain — this is the point of X1.*
- **X1-no-false-positive (load-bearing):** on G_flat the hierarchical per-bin collapses toward the
  global (τ small) → `hier_adv` corrected-t ≤ 2, split-pass-fraction near nominal. Pooling must not
  manufacture structure.
- **Selftest (known-answer):** on a synthetic table with REAL per-bin structure but few points/bin,
  hierarchical beats independent on held-out log-score; on a no-structure table, hierarchical
  collapses to ≈ global (advantage ~0) and does not underperform independent.

## Outcome interpretation (locked)

X1 cleanly separates "power-limited" from "absent" for the WS2 per-input depth read: if pooling
recovers G above control, the F4 no-go is reopened; if not, X3's negative is confirmed on the
strongest estimator (a stronger close). Either way X1-pooling-helps quantifies the estimator gain.
Interpretation + any F4 re-opening go to a FRESH-context adjudicator (§0c), never this session.
Optional extension (not in this run): repeat on the K2 count tables (toys C/D/E).
