# FP-4 — the package width class's schedule deviation: graded and resolved

**2026-07-22, root, under the MASTER Decision 32 autonomous mandate.** FP-4 (`flexnn-package.md`)
asked whether the shipping width class's schedule — summing **all** configured widths every step
(`automl_package/models/flexnn/width/model.py:15-16`) — materially deviates from the certified
sandwich schedule. It gated `width.md` WSEL-3, WSEL-4 and (through them) WSEL-8.

## 1. The pre-registered bar, quoted verbatim

From `flexnn-package.md` FP-4 ("THE COMPARISON — cells, seeds, metric and bar, all named here"):

> **VERDICT BAR — decided in advance, not after seeing the numbers:** the deviation is
> **IMMATERIAL** iff arm B's `ratio_to_floor` is within **2% relative** of arm A's on **all three**
> seeds **and** the paired mean difference is smaller than twice its bootstrap standard error
> (**FP-9**'s helper — do not hand-roll one). Otherwise **MATERIAL**, and the class is brought onto
> the certified schedule.

Arm A = certified sandwich schedule; arm B = the class's sum-all-widths (ALL) schedule. Cell:
toy `hetero`, `n_train=1500`, `sigma=0.05`, `w_max=12`, arch `shared_trunk`, seeds 0/1/2, metric
`per_case[i].fit_bar.ratio_to_floor`.

## 2. Evidence base — the pre-registered comparison already exists on disk

FP-4's verify block prescribed running the two arms under tags `fp4_sandwich` / `fp4_sumall`.
**The identical comparison landed 2026-07-22 as WSEL-14's schedule arms** — same driver
(`automl_package/examples/kdropout_converged_width_experiment.py`), same cell, same matched seeds,
tags `wsel14_sandwich` / `wsel14_b12` (`b12` = every width, every step = the class's sum-all
schedule). Re-running identical cells under new tags would duplicate compute for zero information;
the tag substitution is recorded here instead.

**Positive control, exact:** the sandwich arm is **bit-identical** to the certified reference on
all three seeds — e.g. seed 0 `ratio_to_floor` = 1.08928978 (8 dp shown) in both `automl_package/examples/capacity_ladder_results/WSEL14/w_kdropout_converged_summary_shared_trunk_mse_n1500_s0.05_wsel14_sandwich_c0.json` and the certified `automl_package/examples/capacity_ladder_results/W_KDROPOUT_CONVERGED/w_kdropout_converged_summary_shared_trunk_mse.json`.
*(FP-4's inline seed-0 quote, 1.0892838787488606, differs from disk at the 5th decimal — the known post-cert regeneration; per FP-4's own rule the JSON on disk governs.)* <!-- numcheck-ignore: quotes FP-4's stale inline number to flag it as stale; the governing on-disk value is on the line above -->

## 3. Measured values

| seed | A sandwich `ratio_to_floor` | B ALL `ratio_to_floor` | rel diff (B−A)/A |
|---|---|---|---|
| 0 | 1.0892897800953354 (`automl_package/examples/capacity_ladder_results/WSEL14/w_kdropout_converged_summary_shared_trunk_mse_n1500_s0.05_wsel14_sandwich_c0.json`) | 1.0549928026397444 (`automl_package/examples/capacity_ladder_results/WSEL14/w_kdropout_converged_summary_shared_trunk_mse_n1500_s0.05_wsel14_b12_c0.json`) | **−3.15%** <!-- numcheck-ignore: derived percentage over the two cited cells on this line --> |
| 1 | 1.060847129485201 (`automl_package/examples/capacity_ladder_results/WSEL14/w_kdropout_converged_summary_shared_trunk_mse_n1500_s0.05_wsel14_sandwich_c1.json`) | 1.0632117157585812 (`automl_package/examples/capacity_ladder_results/WSEL14/w_kdropout_converged_summary_shared_trunk_mse_n1500_s0.05_wsel14_b12_c1.json`) | **+0.22%** <!-- numcheck-ignore: derived percentage over the two cited cells on this line --> |
| 2 | 1.077476009439041 (`automl_package/examples/capacity_ladder_results/WSEL14/w_kdropout_converged_summary_shared_trunk_mse_n1500_s0.05_wsel14_sandwich_c2.json`) | 1.2244662579060168 (`automl_package/examples/capacity_ladder_results/WSEL14/w_kdropout_converged_summary_shared_trunk_mse_n1500_s0.05_wsel14_b12_c2.json`) | **+13.64%** <!-- numcheck-ignore: derived percentage over the two cited cells on this line --> |

Paired mean difference (B−A) = +0.038353; bootstrap SE = 0.044649 (10,000 resamples, seed 0,
`automl_package/utils/numerics.py` `bootstrap_se` — FP-9's helper, not hand-rolled); twice the SE =
0.089297 → the paired clause **passes** (|+0.038| < 0.089). <!-- numcheck-ignore: derived statistics over the six cited cells in the table above, not leaves of any one JSON -->
Secondary metrics for the record: `mean_executed_width` 5.15/6.082/6.936 (A) vs 5.146/5.712/5.916 (B); `mse_hardpick` differences are within seed noise. <!-- numcheck-ignore: secondary readouts copied from the six cells cited in the table above -->

## 4. Grade under the bar as written

The per-seed 2% clause **fails on seeds 0 and 2** (−3.15% and +13.64%); the paired-difference
clause passes. The bar requires both.

**VERDICT:** MATERIAL

**What the failure actually is:** two-sided per-seed variance, not a systematic degradation — ALL
is *better* than sandwich by 3.1% on seed 0, worse by 13.6% on seed 2, and the paired mean is not
significant. Seed 2 is the historically weak seed (the certified positive-control arm also fails
its strong-pass there — `automl_package/examples/capacity_ladder_results/WSEL16/frozen.json`,
`halt_condition_b.per_seed.2`).

## 5. Resolution — the pre-registered remedy is superseded by MASTER Decision 31

FP-4's MATERIAL branch ("the class is brought onto the certified schedule") was written 2026-07-20.
On 2026-07-22, with the full WSEL-14 schedule study in hand, the user ruled the opposite direction
(MASTER Decision 31(a), binding): **the ALL schedule is the programme DEFAULT; the sandwich
survives only as the labelled comparability mode against already-landed ledgers** — "dominance,
not trade-off". The dominance evidence is the schedule study's mid-width readout:
mean held-out MSE at width 4 is 0.175574 (ALL) vs 0.344241 (sandwich), and at width 5 it is 0.058548 (ALL) vs 0.198739 (sandwich) — `readouts.fit.per_arm` in `automl_package/examples/capacity_ladder_results/WSEL14/frozen.json` — the mid-width starvation the sandwich's 2-random-middles sampling causes (`width.md` §3.10 notes it worsens with `w_max`). The
per-step cost premium of ALL is per-head bookkeeping, removed by the fused readout (WSEL-18).

**Therefore: the shipping class's sum-all schedule IS the ratified default. No code change.**
The deviation FP-4 was opened to police is no longer a deviation.

## 6. Consequences, binding

- `width.md` §1's warning ("W-SHARED and W-PERINPUT may not be read off it until FP-4 rules") is
  **discharged** — WSEL-3, WSEL-4 and WSEL-8 are unblocked on this dependency.
- **The Decision-31 label rule carries the residue of the MATERIAL grade:** ALL-trained and
  sandwich-trained numbers may not be tabulated together without the schedule named per arm. The
  seed-2 excursion above is exactly why the label is load-bearing, not cosmetic.
- The sandwich remains the comparability mode for reading against certified-era ledgers; nothing
  here reopens `G-WIDTH = PASS` (its caveats stand as amended, `width.md` §2).
