# T3 adjudication — moving-mode power curve (count-lane analog of X10)

**Fresh-context Opus adjudication, 2026-07-10.** Re-derived every headline number directly from
`capacity_ladder_results/T3/t3_summary.json`; re-read the pre-registration
(`capacity_ladder_results/T3/PREREGISTRATION.md`), the script (`capacity_ladder_t3.py`), and the
prior X4/X4b context (`RESULTS.md:320–353`). Verdict: **GO to certify T3 and fold**, with a mandatory
2/3-marginal-at-N=1000 nuance carried into the headline (Q2). No re-run performed; no other task touched.

## Q1 — Re-derivation (crossing_n, control-flat, per-seed significance)

**CONFIRMED.** The stored `significant_positive` flag is `mean − 2·se > 0` (`capacity_ladder_t3.py:116`).
I recomputed it independently for all 18 cells (3 N × 3 seeds × 2 toys); every recompute matches the
stored flag. Toy E (humped), gold Δ*_mid mean ± bootstrap-SE, and the significance margin `mean − 2·se`:

| N | seed | gold_mid | SE | mean−2·SE | sig+ | arbiter recovered |
|------:|:----:|---------:|-------:|----------:|:----:|:-----------------:|
| 1000 | 0 | +0.02507 | 0.00384 | +0.01738 | yes | no |
| 1000 | 1 | −0.00448 | 0.00097 | −0.00642 | no | no |
| 1000 | 2 | +0.05683 | 0.00444 | +0.04795 | yes | **yes** |
| 4000 | 0 | +0.06373 | 0.00526 | +0.05321 | yes | yes |
| 4000 | 1 | +0.08687 | 0.00881 | +0.06924 | yes | yes |
| 4000 | 2 | +0.06289 | 0.00462 | +0.05366 | yes | yes |
| 16000 | 0 | +0.06251 | 0.00474 | +0.05304 | yes | yes |
| 16000 | 1 | +0.09042 | 0.00918 | +0.07207 | yes | yes |
| 16000 | 2 | +0.10107 | 0.01069 | +0.07969 | yes | yes |

- **n_seeds_hump**: 2/3 (N=1000, s0+s2) → 3/3 (N=4000) → 3/3 (N=16000). Matches stored.
- **arbiter recovered**: 1/3 (N=1000, s2 only) → 3/3 → 3/3. Matches stored.
- **crossing_n**: first N with n_seeds_hump ≥ 2 is **N=1000** — CONFIRMED (independently recomputed = 1000).
- **E_broad control**, gold middle-tercile mean and arbiter, all 9 control cells:

| N | E_broad gold_mid (s0,s1,s2) | all ≤ 0.05 | arbiter recovered | eff_mid |
|------:|-----------------------------|:----------:|:-----------------:|:-------:|
| 1000 | −0.00250, −0.00085, −0.00487 | yes | 0/3 | 1.0 |
| 4000 | −0.00194, −0.00114, −0.00202 | yes | 0/3 | 1.0 |
| 16000 | −0.00005, −0.00071, −0.00076 | yes | 0/3 | 1.0 |

  `control_flat_all_n = True` — CONFIRMED (recomputed = True). The control-flat rule the code enforces
  (`capacity_ladder_t3.py:237`) is `(not recovered) AND gold_tercile["middle"] ≤ 0.05`, which is exactly
  the prereg's "arbiter not recovered AND gold middle-tercile mean ≤ 0.05" — the code is faithful to the prereg.

## Q2 — Is crossing_n=1000 ROBUST or BORDERLINE?

**The honest read is "recoverable at N=1000" per the locked rule, but it must be reported as MARGINAL
in seed-count, strengthening to a robust 3/3 by N=4000.** Two distinct facts, which must not be conflated:

- The prereg bar ("gold Δ*_mid > 0 by 2·SE on ≥2/3 seeds", §"Pre-registered outcome readings") is met
  at N=1000 (exactly 2/3, s0 and s2). The rule is locked and pre-registered, so crossing_n=1000 is the
  correct label — reframing it away would itself be post-hoc.
- **BUT the crossing is marginal at the seed-count level, not at the effect-size level.** The two seeds
  that hump at N=1000 are individually *strongly* significant — s0 at mean/SE = 6.5, s2 at mean/SE = 12.8
  — so those are not near-threshold noise. The marginality is entirely that it is 2/3 exactly, and the
  dissenting seed s1 shows *no* hump (its mean is negative, −0.00448, not merely non-significant), and the
  strict arbiter-recovery bar is only 1/3 at N=1000. At N=4000 both bars jump to 3/3, and the gold hump
  grows monotonically with N (per-seed means climb from 0.025–0.057 → 0.063–0.087 → 0.063–0.101).

Ruling: report as **"recoverable at N=1000 (marginal: 2/3 seeds; strengthens to 3/3 by N=4000, growing
with N)"** — the crossing is real and pre-registered, but presenting it as a clean recovery without the
2/3→3/3 nuance would overstate the N=1000 datum. The bootstrap SE is a within-seed *spatial* dispersion
over the 13 mid-tercile grid points, so "significant by 2·SE" is a per-seed statement about the mid-band
gold curve sitting above zero, not a cross-seed statement — the cross-seed count (2/3) carries the
fragility signal and must be surfaced.

## Q3 — Reconciliation with the prior "absent at N=1000"

**Benign, fully explained — NOT a genuine concern.** The apparent contradiction dissolves on inspection,
but the task framing needs one correction:

1. **T3 does not use a "stronger" instrument than the prior at N=1000 — it uses X4b's instrument verbatim.**
   `capacity_ladder_t3.py:137` calls `x4b.fit_and_score_seed_mr(...)` unchanged (R=8 multi-restart,
   keep-best-by-train-MAP, leak-free). So T3's N=1000 cell is the *same* instrument that produced the prior
   result, and it **reproduces** the prior numbers: X4b at N=1000 already reported **model-capture 2/3 and
   arbiter-recovery 1/3** (`RESULTS.md:337–338`), with s1's hump "genuinely absent." T3's N=1000 = 2/3 gold,
   1/3 arbiter (s2). These agree.
2. **The "contradiction" is a bar/label difference, not a data difference.** X4b's headline "genuinely
   absent / fragile / no-go" was keyed to the *strict arbiter-recovery* bar (1/3) and to s1's genuine
   absence. T3's "recoverable" label is keyed to the *gold model-capture* bar (2/3), which the prereg
   explicitly makes the outcome-defining read. Both statements are true of the same data on different bars;
   the T3 summary even stores both (`n_seeds_hump=2`, `n_seeds_arbiter_recovered=1` at N=1000).
3. **The genuinely new content is the power curve, and it is driven by N, not by a stronger instrument.**
   Holding the instrument fixed, recovery consolidates from 2/3 gold + 1/3 arbiter at N=1000 to **3/3 on
   both bars at N=4000**, and the gold hump grows monotonically with N. This is the correct reading of the
   prior null: **power-(sample-size)-limited, not signal-absent.** (Correction to the task's Q3 phrasing:
   the recovery is not explained by a stronger instrument — X4b's multi-restart step only removed s1's
   *optimization* collapse (RESULTS.md:334–337); it did not manufacture a hump. What T3 adds is that *more
   data*, at the same instrument, turns the marginal N=1000 read into a robust one.)
4. **The E_broad control staying flat 3/3 is a sufficient false-positive guard for trusting the N=1000
   recovery.** The control runs the identical instrument, identical bootstrap-SE machinery, on variance-only
   (never-bimodal) data, and stays flat at *every* N including N=1000: gold_mid within [−0.005, +0.006] of
   zero, never significant-positive, arbiter 0/3, and the mixture collapses to a single component (eff_mid =
   1.0 on every control cell vs 1.9–3.2 on E's mid-band). A spurious mid-band hump generator would fire on
   E_broad and would not scale coherently with N; neither happens. Combined with the two N=1000 E seeds
   being 6.5·SE and 12.8·SE above zero, the false-positive risk at N=1000 is well controlled.

Minor note (non-blocking): X4/X4b used N_test=2500; T3 uses N_test=N_train=n, so at N=1000 the held-out set
is smaller (1000 vs 2500). This makes T3's N=1000 arbiter read if anything noisier, not stronger; the gold
read is N_test-independent. It does not weaken the reconciliation.

## Q4 — Instrument validity at each N

**CONFIRMED.** Per the prereg's HARD control guard, E_broad must stay flat 3/3 at every N (arbiter not
recovered AND gold middle-tercile mean ≤ 0.05). Verified true at N=1000, 4000, and 16000 (Q1 table). No
control false positive fires at any N ⇒ the instrument is valid at each N per the prereg, and the read
resolves into a pre-registered outcome rather than escalating.

## Q5 — Overall

**GO** to certify T3 and fold. The verdict `recoverable_at_N=1000` is the correct pre-registered label,
the control guard passes at every N, and the reconciliation with the prior "absent at N=1000" is benign
(same instrument, different bar; the prior null was power-limited, demonstrated by the clean 3/3
consolidation at N≥4000). The single binding condition: the fold headline must carry the 2/3-marginal-at-
N=1000 → 3/3-at-N=4000 nuance (Q2) and must not be read as overturning X4b's fragility finding — it
*contextualizes* it as under-powered.

---

## Ready-to-paste RESULTS.md fold block

## T3 — moving-mode power curve (count-lane analog of X10, ProbReg). Verdict: GO (adjudicated 2026-07-10, fresh context)

**Headline.** Turning the X4b moving-mode read (toy E, two modes merging at both ends of x, resolving
only mid-band) into an N-sweep at the identical R=8 multi-restart leak-free instrument shows the prior
"absent at N=1000" was **power-limited, not signal-absent**. The gold-standard model-capture hump crosses
the pre-registered ≥2/3-seed bar already at **N=1000 (marginal, 2/3 seeds)** and consolidates to a robust
**3/3 on both the gold and the strict arbiter bar by N=4000**, with the mid-band gold advantage growing
monotonically with N. The variance-only E_broad control stays flat 3/3 at **every** N — no false positive.
Read of record: **moving-mode per-input count is recoverable; the June N=1000 fragility was under-powering, resolved by more data.**

- **crossing_n = N=1000, but marginal.** gold Δ*_mid > 0 by 2·SE on **2/3 seeds** at N=1000 (s0 +0.025
  at 6.5·SE, s2 +0.057 at 12.8·SE — both strongly significant; s1 −0.004, genuinely no hump). The two
  humped seeds are not near-threshold; the marginality is purely the 2/3 seed-count, with s1 absent and
  strict-arbiter recovery only 1/3 (s2) at this N. Per the locked prereg rule (smallest N with ≥2/3) the
  crossing is N=1000.
- **Consolidates and grows with N.** gold hump 2/3 → **3/3 (N=4000)** → 3/3 (N=16000); arbiter recovery
  1/3 → **3/3** → 3/3. Per-seed gold_mid climbs 0.025/−0.004/0.057 (N=1000) → 0.064/0.087/0.063 (N=4000)
  → 0.063/0.090/0.101 (N=16000) — monotone strengthening, and mid-band effective-component count rises
  with N (mid eff ≈1.9 → 2.7 → 3.2). This is the power curve X10 is for the depth lane.
- **Control flat 3/3 at EVERY N (hard guard PASSED).** E_broad (variance-only humps, never bimodal): gold
  middle-tercile mean within [−0.005, +0.006] of zero at all N (never significant-positive), arbiter
  recovered 0/3 at all N, mixture collapses to a single component (eff_mid = 1.0 on every control cell vs
  1.9–3.2 on E). No spurious mid-band hump at any N, including N=1000 where the E hump crosses — a
  sufficient false-positive guard for the N=1000 recovery.
- **Reconciles the prior "absent at N=1000" — benign bar difference, not a contradiction.** T3 reuses
  `x4b.fit_and_score_seed_mr` VERBATIM (`capacity_ladder_t3.py:137`), so the N=1000 cell *reproduces* X4b
  (2/3 gold model-capture, 1/3 arbiter). X4b's "genuinely absent / fragile" headline was keyed to the
  strict arbiter bar (1/3); T3's "recoverable" is keyed to the pre-registered gold model-capture bar
  (2/3). Recovery is driven by **more data**, not a stronger instrument — X4b's multi-restart step only
  removed s1's optimization collapse; the N-sweep is what lifts the read to 3/3. The prior null is thus
  contextualized as under-powered, not overturned.
- **Strict-probabilistic + leak-free, unchanged from X4/X4b.** MAP objective's Dirichlet-usage prior is
  the model's own term (coefficient 1, no tuned λ); keep-best is by the training MAP objective only (model
  selection never sees held-out or gold). The one T3 addition is the shared G1 bootstrap SE
  (`_capacity_ladder._bootstrap_col_means`) on the gold mid-tercile mean — the missing dispersion the
  pre-registered 2·SE bar needs, not an invented statistic.

Artifacts: `capacity_ladder_results/T3/{PREREGISTRATION.md,t3_summary.json}`, `capacity_ladder_t3.py`.
Adjudication: fresh-context Opus, re-derived every headline number from `t3_summary.json` (all 18 cells'
significance calls, crossing_n, control-flat recomputed byte-exact); GO to certify T3 and fold, binding
condition = keep the 2/3-marginal-at-N=1000 → 3/3-at-N=4000 nuance in the headline (do not read as
overturning X4b's fragility finding — it contextualizes it as power-limited).
