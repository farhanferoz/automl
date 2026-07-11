# P1 pre-registration — depth power curve (X10, trimmed 3-point)

**Task:** `docs/plans/perinput_selector_2026-07-10/EXECUTION_PLAN.md` WS-D, P1. Absorbs the
prior plan's X10 in trimmed 3-point form. X3 re-issued the F2/F3 toy-G per-input depth
advantage at N_test=500 under a split-aware corrected SE and found it modest and
split-fragile (`capacity_ladder_results/X3/PREREGISTRATION.md`): "unresolvable at this N,"
not a clean negative. P1 turns that into a power statement, mirroring T3's treatment of the
moving-mode count signal: absent, or under-powered?

**Nature:** retrains the F2 nested-depth ladder (`capacity_ladder_f2.py`'s `_build_model` +
`_nested_all_depth_log_likelihood`, NESTED-strategy `FlexibleHiddenLayersNN`, F2's fixed
hyperparameters — hidden_size=24, lr=5e-3, n_epochs=800, max_depth=6, BN off, no depth
penalty) at three N_test points, then reads each score table with X3's repeated cross-fit
machinery (`run_repeated_crossfit`, 50 random 50/50 fit/score splits, Nadeau-Bengio 2003
corrected SE) exactly as X3 does. Unlike F2, P1 trains ONLY the nested model per case — the
control/fixed-depth baselines exist to certify F2's B-coh bar, a bar P1 does not re-check.

## Design

- **Toys:** G (varying required capacity: linear on x<0, compositional sine on x>=0) and
  G_flat (its uniform-linear-complexity negative control), `_capacity_ladder_toys.py`. Toy H
  is NOT in P1 (its signature is SNR, not depth per se; T1/T3 own the SNR- and mode-shape
  discriminators respectively).
- **N_test in {500, 2000, 8000}.** N_train scaled to F2's own train:test ratio
  (N_TRAIN=1000, N_TEST=500 -> ratio 2.0), so N_train in {1000, 4000, 16000}.
- **N_test=500 reuses F2's existing tables verbatim**
  (`capacity_ladder_results/F2/nested_toy{G,G_flat}_seed{0,1,2}.pt`) instead of retraining —
  same toy defaults (a=1.5, omega=4pi, sigma=0.25) and the same hyperparameters, so the table
  is key-compatible with what a fresh N=500 fit would produce.
- **3 seeds {0, 1, 2}**, F2's train/test seed convention (`seed` for train, `seed + 500` for
  the held-out draw).
- **Hyperparameters held FIXED across N** (same net, same n_epochs=800, at every N_test point)
  — the sweep varies statistical power (more held-out data shrinks the detection floor), not
  model capacity or training budget. This mirrors T1's framing ("the point is signal SIZE, not
  N") turned around: here the point is FLOOR size, not fit quality.

## Definitions (locked before any real run)

- **Detection floor at N** = `2 * se_nadeau_bengio`, the Nadeau-Bengio (2003) corrected SE of
  the tercile per-bin-stacking advantage over global (X3's own `se_nadeau_bengio`, computed
  from 50 random fit/score splits) — the smallest `|signal|` that would register as
  significant at that N under X3's own significance convention (`beats_global_2se`).
- **Measured signal at N** = the point estimate of that same advantage (X3's `mu_bar`),
  pooled across the score-half over the 50 splits.
- **Crosses** = `|signal| > floor`, evaluated per (toy, N, seed).

## Pre-registered readings

- **Toy G crosses its floor on >= 2/3 seeds at some N** -> report "recoverable at N=<that N>"
  (the smallest such N). This threshold mirrors T3's own crossing rule for the sibling
  moving-mode power curve (`>= 2/3 seeds` at the corrected 2*SE) — the two X10-descended
  power-curve tasks use the same convention so their readings are directly comparable in
  R-INT.
- **Toy G never reaches 2/3 seeds up to N=8000** -> report "below the floor up to N=8000,"
  quantified by the mean floor and mean signal at N=8000 (and the seed-crossing count) rather
  than left as a bare negative.
- **Control bar (toy G_flat):** 0/3 seeds may cross G_flat's OWN floor at ANY of the three N
  points (T3's "control flat 3/3" convention, restated for a 0-crossings-per-N floor check
  instead of a flat-gold-Delta check). A control crossing at any N means the instrument itself
  is not trustworthy at that N — escalate to a fresh-context adjudicator (governance §0b). The
  response is NOT to silently drop that N; it is reported as a caveat on the whole curve.
- **5-point extension:** ONLY if the 3-point curve is non-monotone (floor not monotonically
  decreasing with N, or signal crossing then un-crossing) — an orchestrator decision, not a
  worker decision (EXECUTION_PLAN.md WS-D P1).

## Estimator (X3's, reused verbatim — not re-derived)

Per (toy, N, seed): 50 random 50/50 fit/score splits of the (N_test, 6) score table. Per
split: quantile terciles of the fit-half x; global stack (`stack_em`) + per-bin stack
(`perbin_stack`) fit on the fit-half; held-out advantage `diff = perbin_ls - global_ls` scored
on the score-half. Aggregate: `mu_bar = mean_s(mean_diff_s)`,
`se_nadeau_bengio = sqrt((1/S + n_score/n_fit) * Var_s(mean_diff_s))` (`S=50`,
`n_score/n_fit ~= 1` for a 50/50 split). `floor = 2 * se_nadeau_bengio`,
`signal = mu_bar`, `crosses = |signal| > floor`.

## Selftest (synthetic, no training — `--selftest`)

(a) known-answer discrimination: a synthetic table with per-region capacity peaks 1/3/6 must
cross its own floor; a flat 3/3/3 table must not — reuses X3's `_synthetic_table` and
`run_repeated_crossfit` verbatim, only wrapping the result in P1's floor/signal read.
(b) floor scaling: the SAME peaked synthetic template at increasing per-region sample counts
(~170, ~670, ~2670, matching N_test/3 at the three real N points) must produce a finite,
monotonically DECREASING floor as N grows — confirming the floor computation actually gains
power with more data before it is trusted on any real table.

## Non-goals

No library changes. No N-sweep beyond the 3 locked points without the orchestrator invoking
the non-monotone-curve extension clause above. No real training in this authoring pass — the
orchestrator owns the real N-sweep (measure ONE N=8000 fit's wall-time first: `--only-n 8000
--toys G --seeds 0`), including deciding whether the fixed n_epochs=800 needs revisiting at
N_train=16000.
