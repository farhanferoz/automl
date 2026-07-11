# V2 findings — heteroscedastic σ(x) fix battery (WS3)

Source: `capacity_ladder_v2.py` on V-toy1 (1-D heteroscedastic, known smooth σ(x)).
6 arms, 3 seeds, N∈{200,1000}, scored on a fresh held-out pool of 2000 vs known σ(x).
Run wall-time 756.7 s. Arms: (a) joint-NLL [disease], (b) β-NLL [labelled fix],
(c) mean-first in-sample [diseased control], (d) mean-first CROSS-FITTED [the fix],
(e) per-tercile recalibration on (c), (f) locally-adaptive conformal [interval only].

## Per-arm results (mean ± sd over 3 seeds)

| N | arm | σ-ratio-err ↓ | held-out NLL ↓ | SSR (→1) |
|---|-----|--------------:|---------------:|---------:|
| 200  | (a) joint   | 0.163 ± 0.031 | 0.573 | 1.640 ± 0.360 |
| 200  | (b) β-NLL   | **0.139** ± 0.030 | **0.513** | 1.475 |
| 200  | (c) insamp  | 0.162 ± 0.021 | 0.658 | 1.701 |
| 200  | (d) x-fit   | 0.146 ± 0.039 | 0.912 | 2.119 ± 1.422 |
| 200  | (e) recal   | 0.280 ± 0.092 | 0.716 | 2.085 |
| 1000 | (a) joint   | 0.063 | 0.364 | 1.012 |
| 1000 | (b) β-NLL   | **0.054** | **0.357** | 0.989 |
| 1000 | (c) insamp  | 0.060 | 0.359 | 0.984 |
| 1000 | (d) x-fit   | 0.062 | 0.360 | **0.970** |
| 1000 | (e) recal   | 0.079 | 0.361 | 1.003 |
| — conformal (f): coverage 0.872 (N=200) / 0.910 (N=1000), target 0.90; width ≈1.18 ||||

SSR = mean((y−μ̂)²/σ̂²); >1 = overconfident (variance-collapse direction).

## Pre-registration outcome

**Criterion 2 (calibration): MET.** Arm (d) cross-fit SSR @N=1000 = [0.958, 0.930, 1.022]
— all in [0.9, 1.1], **3/3 seeds**. Cross-fitting delivers its honest-residual promise at N=1000.

**Criterion 1 (ranking σ-ratio-err (d) ≤ (e) < (b) < (c) < (a)): FAILED** — and the failure is
informative, not a plumbing bug. Observed best→worst: **b < d < c < a < e** at N=200 (arm d
beats c); at N=1000 all density arms sit at 0.054–0.079 within noise, with (e) worst.
Three specific, real reasons:

1. **The disease is finite-sample, so at N=1000 there is nothing to fix.** Joint-NLL SSR goes
   1.64 (N=200, overconfident collapse) → **1.01 (N=1000, perfectly calibrated)**. By N=1000 all
   five arms sit at σ-ratio-err ≈ 0.06 within noise — the ranking hypothesis silently assumed the
   disease persists; it does not. This CONFIRMS and extends V0/V1 (collapse is a small-N pathology,
   none at N=1000).
2. **Per-tercile recalibration (e) is the WORST arm at every N**, not second-best. A 3-step scalar
   multiplier injects piecewise-constant error into a *smooth* σ(x), and at N=200 the tercile
   scales are estimated on a tiny val split → high variance (σ-err 0.28 ± 0.09, SSR 2.09).
   **Finding: coarse recalibration is the wrong tool for smooth heteroscedasticity — it hurts.**
3. **Cross-fitting (d) backfires at very small N.** Two compounding mechanisms at N=200: (i) its
   out-of-fold (K=5) mean predictions on only ~160 points/fold are poor, inflating residual noise;
   and (ii) a design mismatch — the σ head is calibrated to the *fold* means' residuals but is
   deployed alongside a *different, overfit full-data mean* whose held-out generalization error is
   larger → SSR 2.12 ± 1.42 (worst calibration of the density arms, concentrated in seed 1). The
   honest-residual fix needs enough per-fold data AND an honest deployed mean; it is right at
   N=1000, wrong at N=200. (This second mechanism is exactly what R4 requires V3 to fix — deploy an
   early-stopped mean before building the σ-rung table.)

## Headline

- **β-NLL (Seitzer 2022, already in the library) is the most robust σ(x) estimator here** — lowest
  σ-ratio-err AND best held-out NLL at both N. If one fix must be recommended, it is β-NLL.
- **Cross-fitting is the right *calibration* fix once N is adequate** (SSR→1, 3/3 @N=1000) but is
  noise-limited at N=200.
- The two pre-registered criteria point at different arms (β-NLL wins pointwise σ accuracy;
  cross-fit wins global calibration) because σ-ratio-err and SSR measure different things.

Verdict: report honestly — calibration target met, ranking hypothesis refuted with a mechanism.
Material for R4 (WS3 adjudicator). Does not touch `automl_package/models/`.
