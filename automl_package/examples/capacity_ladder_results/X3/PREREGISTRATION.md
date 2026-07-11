# X3 pre-registration — repeated cross-fit re-issue of the F3 per-bin verdicts

**Task:** EXECUTION_PLAN §8.5 X3 / §8.6. Fixes the W4 finding (§8.2): the F3 toy-G
per-input depth advantage of "+2.67·SE" is **split-fragile** — it survives only 3/9
re-randomized fit/score partitions, and R3's robustness check varied only the bootstrap
seed, not the fit/score split. X3 averages the per-bin advantage over many random splits
with a split-aware pooled SE and re-issues the G / G-flat / H verdicts.

**Nature:** pure post-hoc on the existing F2 nested-depth ladder `.pt` tables
(`capacity_ladder_results/F2/nested_toy{G,G_flat,H}_seed{0,1,2}.pt`). No training. Reuses
F3's validated `_perbin_vs_global` reader verbatim, only sweeping its fit/score `split_seed`.

## Single-split baseline being re-issued (from `F3/f3_summary.json`)

Tercile per-bin advantage over global, held-out log-score, per seed (mean_diff ± boot SE):

| toy    | seed 0            | seed 1            | seed 2            | F3 verdict (`_any_pass`) |
|--------|-------------------|-------------------|-------------------|--------------------------|
| G      | +0.0037 ± 0.0065  | **−0.0139** ± 0.0062 | +0.0122 ± 0.0046 (2.65·SE) | pass=True (seed-2 only)  |
| G-flat | +0.0024 ± 0.0032  | +0.0062 ± 0.0039  | −0.0022 ± 0.0044  | ties (pass)              |
| H      | −0.0078 ± 0.0061  | −0.0060 ± 0.0104  | +0.0184 ± 0.0111  | SNR-trend fail           |

The G "pass" rests entirely on seed 2 under one fixed split; G-flat seed 1 (+0.0062) exceeds
G seed 0 (+0.0037). `_any_pass` across 3 seeds is an uncorrected ~3× multiplicity inflation (W4).

## Estimator (locked before running)

For each (toy, seed), sweep **S = 50** random 50/50 fit/score splits (`split_seed = 0..49`).
Per split *s*: recompute quantile terciles on the fit-half x, stack global + per-bin on the
fit half, score the held-out half → per-example advantage `diff = perbin_ls − global_ls`;
record `mean_diff_s`, per-split paired-bootstrap `se_s`, `beats_s = mean_diff_s > 2·se_s`.

Aggregate per (toy, seed):
- **Point estimate** `μ̄ = mean_s mean_diff_s`.
- **Split-aware SE (Nadeau–Bengio 2003 corrected resampled):**
  `SE = sqrt( (1/S + n_score/n_fit) · Var_s(mean_diff_s) )`. For a 50/50 split
  `n_score/n_fit = 1`, so `SE ≈ sqrt(1/S + 1)·SD_s ≈ SD_s` — i.e. **overlapping splits do NOT
  shrink the SE by √S**; the split-to-split dispersion is the honest uncertainty. Naïve
  `SD_s/√S` would be anti-conservative and is explicitly rejected.
- **Corrected t** `= μ̄ / SE`.
- **Split-pass-fraction** `= mean_s(beats_s)` — the assumption-light generalization of W4's
  "3/9" to "k/50"; co-primary readout.

Pooled across the 3 seeds: report each seed's `μ̄`/corrected-t/pass-fraction, the between-seed
mean ± SD of `μ̄`, and a boolean verdict (below). 3 seeds minimum per §0b.

For **H**, per split also record the tercile argmax capacity in the highest-SNR (lowest-x)
vs lowest-SNR (highest-x) bin; report `trend_down_fraction = mean_s(argmax_hi_snr > argmax_lo_snr)`.

## Pre-registered bars

- **G (signal):** the tercile per-bin advantage is robustly positive under repeated cross-fit —
  pooled `μ̄ > 0` with **corrected-t > 2** and a split-pass-fraction **materially above the
  G-flat control's**, on a majority of seeds. *Prediction:* PASS but markedly weaker than the
  single-split +2.65·SE (W4 says fragile); plausibly it lands at the noise floor.
- **G-flat (no-false-positive, load-bearing):** the negative control does NOT show a robust
  positive advantage — pooled corrected-t **≤ 2** and split-pass-fraction near nominal
  (≲ 5–10%). This is the discrimination that decides everything.
- **H (resolution dial):** the SNR down-trend is **stable across splits** (trend_down_fraction
  a clear majority per seed), not a single-split artifact. *Prediction:* likely FAILS
  (single-split already showed no down-trend on any seed).

## Decision rule (how X3 unblocks the F4 gate — §8.6)

- **G survives** (corrected-t > 2, split-pass-fraction clearly > G-flat) → per-input depth
  signal is **real-but-modest** → F4 (n_predictor distilled router) is worth running, with
  calibrated expectations.
- **G collapses to the G-flat level** (corrected-t ≤ 2, or pass-fraction ≈ control) → W4's "at
  the noise floor" is **confirmed** → **do not run F4** (no per-input signal to distill); report
  the negative and close the WS2 per-input sub-lane.
- **Small-elpd caveat (§8.3, Sivula–Magnusson–Vehtari):** these advantages are ~0.003–0.01 nat
  (elpd-diff ≪ 4). A near-flat pooled `μ̄` is reported as **"unresolvable at this N"**, NOT as a
  clean negative — matching W4's framing.

Interpretation + the F4 ruling go to a FRESH-context adjudicator (§0c), never this session.
