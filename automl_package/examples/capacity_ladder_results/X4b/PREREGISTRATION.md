# X4b pre-registration — E-lane non-nested arm under multi-restart (does s1's collapse flip?)

**Task:** EXECUTION_PLAN §8.5 X4 follow-up (adjudicator-recommended, RESULTS.md "## X-queue
follow-ups" X4 bullet). X4 ran the June NON-nested per-input arbiter (mixture-vs-best-single-Gaussian
held-out advantage) on the identical K4 toy-E data across 3 seeds and found it ~1/3-recovering, with
the gold-standard Δ*(x) decomposition reading:

| seed | Δ̂ edge/mid/edge | arbiter recovered | gold Δ*(x) middle | read |
|---|---|---|---|---|
| s0 | +0.011 / +0.009 / −0.002 | no  | **+0.025** | model captures the hump, estimation-limited at N_te=2500 |
| s1 | −0.061 / −0.065 / −0.048 | no  | **−0.065** | **mixture training COLLAPSE** — fitted model worse than a single Gaussian even mid-band |
| s2 | −0.043 / +0.020 / +0.008 | yes | **+0.023** | model captures the hump; arbiter recovers |

So **model-capture = 2/3** (s0, s2 hump; s1 collapses). X4's verdict — "E-fragility is
instrument-general" — is HEDGED because s1 was not a limit of the instrument on this data but a
**training collapse**: a single unlucky weight-network initialisation (`train_aggregate_sparsity`'s
`seed` argument) landed in a basin where the per-input usage network puts ~all mass on one component,
so the mixture degenerates to the single-Gaussian baseline and the gold oracle (which reads the
*fitted* model) goes negative in the middle.

**Question X4b settles:** was s1's collapse an avoidable optimisation artifact (an unlucky basin),
or is it intrinsic to the model's own objective on this data?

## Lever (the ONLY change from X4)

**Multi-restart with keep-best-by-training-objective.** For each seed, fit the *pinned* primitive
`_variational_em_perinput.train_aggregate_sparsity` **R = 8** times with deterministic restart seeds
`restart_seed = seed * 100 + r`, `r ∈ {0..7}`, and keep the single fit that reached the **lowest
training MAP objective** `model.loss(x_tr, y_tr)` — the exact quantity Adam minimises. Everything
downstream (the delta / eff / gold scoring, the tercile verdict, the E_broad twin, N_TR=1000 /
N_TE=2500, K_MAX / ALPHA0 / SIGMA / SEP / N_EPOCHS / M_GOLD / N_GRID) is **byte-for-byte X4** — X4b
calls the same `p2`/`p3`/`vemp`/`hump` primitives and reuses X4's own helpers verbatim.

- **Spread-init is already on** and unchanged: `train_aggregate_sparsity` initialises component
  centroids at `np.percentile(y, quantiles)` (the percentile spread-init from the nested-k
  starvation finding) and, with `adaptive_bin_means=False`, the component means are FIXED tiles that
  cannot themselves collapse — the collapse is in the usage/weight network, which is exactly what a
  restart re-initialises. So X4b's operative lever is the multi-restart, not a means re-init.
- **No leakage / no p-hacking (load-bearing):** the keep-best criterion is the model's own training
  objective on the TRAINING set only. The held-out test set and the gold-standard resamples are
  never consulted in model selection. This is standard non-convex multi-restart, not test-set
  selection. If the objective genuinely prefers the collapsed (sparse) solution on s1, keep-best
  will *keep the collapse* — that is a valid, reportable outcome, not a failure of the method.
- **Strictly probabilistic:** the MAP objective's Dirichlet-usage prior is the model's own term
  (coefficient 1, no tuned λ), unchanged from `_variational_em_perinput`. No penalty is added.

## Pre-registered bars

- **X4b-primary (model-capture flip on s1):** read the gold-standard Δ*(x) middle-tercile mean of
  s1's kept-best fit.
  - **gold_mid(s1) > 0** (s1 now captures the hump) → **model-capture = 3/3**. The collapse was an
    avoidable optimisation artifact → the "instrument-general" reading weakens and the balance
    **swings back toward W8 (E-fragility is nesting-specific)**: every non-nested fit captures the
    hump when optimised properly, while the nested K5 stays ~1/3-fragile.
  - **gold_mid(s1) ≤ 0** (s1 still collapses across all 8 restarts) → the collapse is intrinsic to
    the model's own objective on this data, not an unlucky basin → **clean W8 refutation stands:
    E-fragility is instrument-general** (the outcome the hedge was waiting on).
- **X4b-arbiter-recovery (secondary, matches X4-recovery):** the number of seeds (of 3) on which the
  neighbour-averaged per-point arbiter Δ̂(x) recovers — middle tercile mean > 0 by its own two-sided
  band AND above both tail tercile means. Reported as `n_seeds_recovered / 3` for the X4 → X4b
  side-by-side.
- **X4b-no-false-positive (HARD guard — invalidates X4b if it fires):** on E_broad (variance humps,
  never bimodal) the kept-best-of-8 arbiter must STILL stay flat / ≤ 0 in the middle band on **3/3**
  seeds and the gold Δ*(x) middle must stay ≤ 0. If multi-restart manufactures a middle-band hump on
  unimodal data, keep-best is cherry-picking noise and the whole X4b read is void.
- **Consistency (soft, not a bar):** s0 and s2 (already model-capturing in X4) keep gold_mid > 0
  under multi-restart — keep-best must not *degrade* an already-good fit. Large divergence noted,
  not failed.

## Governance

Seeds {0,1,2} (§0b), R=8 restarts each, both toys (E, E_broad). Selftest before the real read: the
gold-standard Δ*(x) known-answer oracle (humps on E, flat on E_broad) at reduced N/epochs with a
small R, reusing X4's oracle. Artifacts → `capacity_ladder_results/X4b/`. Heavy run on the main
thread (`AUTOML_DEVICE=cpu`, `OMP_NUM_THREADS=2`). A flipped/held verdict or any design-forcing
surprise → fresh-context adjudicator (§0c), never the producing session.
