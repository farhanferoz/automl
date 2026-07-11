# X2 pre-registration — CRPS-knee arm on the V3 σ-capacity ladder

**Task:** EXECUTION_PLAN §8.5 X2. The V3 held-out-NLL knee UNDER-RESOLVES the σ-capacity
ladder: on the heteroscedastic toy it stops at v1 (per-tercile) even though the σ-ratio-error
truth says v2 (linear-in-x log-σ²) — and at N=4000, v3 — is the best σ-model. §8.3 explains why:
the log score's sensitivity to a relative σ error δ is **second-order** (≈E[δ²] nat), so the
v1→v2 NLL increment (~0.003 nat, elpd-diff ≪ 4, Sivula-small) is below the knee's 2·SE bar.
X2 asks whether the **CRPS** — a strictly proper but NON-local, shape-sensitive score
(Gneiting & Raftery 2007) — read by the SAME `_capacity_ladder.knee`, resolves v1→v2 where NLL
cannot, while still abstaining on the homoscedastic twin.

**Value proposition:** σ-ratio-error is an oracle metric (needs the true σ(x)); CRPS is a proper
score computable without ground truth. If the CRPS-knee tracks the σ-ratio-error truth, the σ
lane gains a **deployable fine readout**. If it does NOT resolve either, that is an informative
negative — no proper score gives a fine σ-selector at this N, strengthening §8.3.

## Construction (reuses V3 verbatim; only the score changes)

For each (toy, N, seed) unit, replicate `capacity_ladder_v3.run_unit`'s fitting **unchanged**:
one early-stopped frozen mean μ(x); K=5 cross-fitted out-of-fold residuals; four σ-rungs
v0 (global scalar) / v1 (per-tercile) / v2 (linear-in-x log-σ²) / v3 (MLP log-σ² head), each a
`SigmaFn: x → σ_v(x)`. On the held-out eval pool, per rung v and example i the predictive is
`N(μ(x_i), σ_v(x_i))`. Two score tables are built from the **same** fits:
- `nll_mat[i,v]  = log N(y_i; μ_i, σ_{v,i})` — reproduces V3's NLL-knee (consistency guard).
- `crps_mat[i,v] = CRPS(N(μ_i, σ_{v,i}), y_i)` — **closed-form Gaussian CRPS**,
  `CRPS = σ·[ z(2Φ(z)−1) + 2φ(z) − 1/√π ]`, `z=(y−μ)/σ` (Gneiting & Raftery 2007, eq. 20-ish).
  CRPS is negatively oriented (lower better), so the CRPS-knee is `knee(−crps_mat, ref_c=1,
  block=None, seed=seed)` — identical reader, negated score so "higher = better" holds.

Toys/N: v_toy1 (hetero, σ(x)=0.1+0.3·sigmoid(4x)) and homoscedastic twin v_toy1h; N=1000, seeds
{0,1,2} (3-seed governance, §0b), matching V3's grid. N=4000 confirmatory arm optional.

## Consistency guard (must pass before trusting the CRPS-knee)

The NLL-knee recomputed here must reproduce V3's recorded `r_star` per unit
(`capacity_ladder_results/V3/v3_summary.json`: hetero {v1, abstain, v1}, homo {abstain×3}). A
mismatch means the replication is unfaithful and the CRPS result is not comparable — investigate
before reading. (Exact reproduction expected: identical seeds/config/call-order, CRPS computed
after all fitting, no extra RNG draws.)

## Pre-registered bars

- **X2-resolution (the hypothesis):** on v_toy1 (hetero) the CRPS-knee reaches a **strictly
  higher rung than the NLL-knee's v1** — i.e. it confirms the v1→v2 (and possibly v2→v3)
  increment — on a majority of seeds, AND the rung it selects is one whose σ-ratio-error is ≤
  v1's (it resolves toward the truth, not past it into noise). *Prediction: PLAUSIBLE but genuinely
  uncertain — the v1→v2 σ-gain is small; CRPS may also miss it.*
- **X2-no-false-positive (load-bearing STOP bar):** on v_toy1h (homoscedastic twin) the CRPS-knee
  **abstains (r_star=0 → v0)** on all 3 seeds — no false σ(x) structure. *Prediction: PASS.*
  **STOP:** if the CRPS-knee reads structure (≥ v1) on the homo twin, CRPS is manufacturing
  heteroscedasticity → escalate to 🔮 (this would disqualify CRPS as a σ-selector, a real finding).
- **CRPS-formula correctness:** the closed form matches a Monte-Carlo CRPS estimate within
  tolerance in the selftest (guards the novel scoring code).

## Outcome interpretation (locked)

- **Resolves + twin abstains** → CRPS gives the σ lane a deployable fine readout; report as the
  X2 win; feeds V4 (which σ-selector to port).
- **Does not resolve (stays v1/abstain) + twin abstains** → no proper score yields a fine σ
  selector at this N; the σ-ratio-error oracle is irreducible here — corroborates §8.3's
  second-order-δ ceiling; report as an informative negative, NOT a bug.
- **Twin false-positive** → STOP → 🔮.

Interpretation + any STOP go to a FRESH-context adjudicator (§0c), never this session.
