# R4 adjudication — WS3 (variance estimation without in-sample collapse)

Scope: adversarial verification of V0/V1/V2, mechanism-ranking table, and the V3/V4 spec check.
Metrics re-derived from source, not from worker prose. No training run; no edits to `automl_package/models/`.

## 1. Bottom line

**WS3's mechanism story is SHIPPABLE in a qualified form.** The disease is real and its
mechanism is correct, but it is a **finite-sample** pathology (gone by N=1000), and the
pre-registered *ranking* of the fix battery is **refuted** and must be reported as such.

The honest shippable claims are:
- Joint (mean, log-σ) Gaussian-NLL in-sample **variance collapse is real at small N** (N=200) and
  **vanishes by N=1000** — demonstrated on the trajectory, not just the end-state.
- For **global σ²** on a well-specified linear model, **evidence (MacKay type-II ML)** is the
  recommendation: exact, closed-form, recovers truth, and yields the weight-decay α/β for free.
- For **global σ²** with a flexible mean, **cross-fitted σ²** recovers truth **when the mean fit
  converges** — and only then (see V1 caveat).
- For **σ(x)** heteroscedastic estimation, **β-NLL is the most robust point estimator** (best
  σ-ratio-err AND held-out NLL at both N); **cross-fitting is the calibration fix once N is
  adequate** (SSR→1, 3/3 seeds @N=1000) but is noise-limited at N=200; **coarse per-bin
  recalibration hurts** on smooth σ(x) and must not be recommended for it.

The naive "one fix wins everything" framing is not supported. β-NLL (a joint, in-sample,
reweighted method) beating the honest-residual cross-fit at small N is mild evidence against the
*strong* form of the WS3 thesis; the *weak* form (make σ absorb honest error for calibration once N
allows) holds.

## 2. Mechanism-ranking table

| Regime | Recommend | Runner-up | Avoid | Evidence |
|---|---|---|---|---|
| Global σ², **linear** / linear-in-basis mean | **Evidence (MacKay α,β)** — exact, lowest variance, gives weight-decay α/β | Cross-fit σ² (model-agnostic, needs K fits) | In-sample MLE (biased low by (N−p)/N) | V1: evidence 0.962/0.994 vs truth 1.0; recovers truth on well-spec, absorbs on misspec |
| Global σ², **flexible/MLP** mean | **Cross-fitted σ²** — recovers truth **iff mean converges**; lower variance than single split | Held-out single-split σ² | In-sample residual variance (collapse) | V1: NN cross-fit on toy1h (converged) ≈0.09=truth; **but** NN on v_toy0 (NOT converged) over-est 2–4× |
| **σ(x)** heteroscedastic, **small N (~200)** | **β-NLL (Seitzer 2022)** — best σ-ratio-err (0.139) + best NLL (0.513) | joint-NLL (a) surprisingly 2nd on NLL/SSR | cross-fit (noise-limited, SSR 2.12±1.42); per-tercile recal (WORST, 0.280) | V2 @N=200 |
| **σ(x)** heteroscedastic, **N≥1000**, calibration target | **Cross-fitted σ(x) head** for SSR→1 (3/3 in [0.9,1.1]) | β-NLL for pointwise σ (all density arms tie within noise) | per-tercile recal (worst) | V2 @N=1000 |
| Interval-only (any density) | LocallyAdaptiveConformal | — | — | V2: coverage 0.910 @N=1000; mild undercoverage 0.872 @N=200 (small cal set, within noise) |

**Cross-cutting caveat for every honest-residual entry:** the mechanism is only as good as the
mean fit's convergence/honesty. V1's own NN-on-v_toy0 arm is the counterexample (non-converged mean
→ σ² over-estimated 2–4×). This is load-bearing for V3 (see §4).

## 3. Per-claim verdicts

### V0 — GO
- **P1 (collapse + tell): CONFIRMED on the trajectory.** Loaded `v_toy1_N200_seed1_curves.pt`:
  in-sample σ̂/σ_true ratio falls 1.55 (ep300) → 0.902 (ep560, the held-out-NLL min) → 0.803
  (ep8000), crossing below 1; held-out SSR rises 1.05 → 1.24 → 3.74; held-out NLL is U-shaped
  (min 0.371 @ep560, up to 1.424 @ep8000). ssr_train stays ≈1.0 throughout (in-sample residuals
  track σ̂). N=1000: sig_ratio 0.96–1.02, ssr_heldout 1.02–1.12 — collapse essentially absent.
- **P2 (linear MLE bias): CONFIRMED, exact.** mle/unbiased = (N−p)/N to 6 dp for all 6 units. But
  note two precisions: (a) this is an **algebraic identity** of RSS/N vs RSS/(N−p), not an empirical
  fit; (b) V0's "correction" is the classical RSS/(N−p) — the *evidence* estimate the team framed as
  P2 is actually V1's machinery (V0 defers it explicitly). **Side-finding:** the GD-trained joint-NLL
  global variance is 2.5–3.3× (vs truth 1.0 / MLE 0.94) even on the well-specified linear model — GD
  on joint NLL does not recover the closed-form MLE; the P2 anchor is the closed form, so P2 stands.
- **Nuance worth stating:** at N=200 the collapse in the σ̂ *function* is modest (~0.88 mean ratio);
  the dramatic held-out SSR blow-up (mean ~4.7) is dominated by **mean overfitting** (held-out
  (y−μ̂)² includes the growing (μ̂−f)² generalization gap), not σ collapse alone. This is *consistent*
  with the registered mechanism (mean overfit is the *cause* of the σ̂ bias) and directly motivates
  the key V3 refinement.

### V1 — GO with CAVEAT
- **Well-specified + misspecified linear/basis arms: CONFIRMED exactly as pre-registered.**
  Well-spec: evidence ≈ held-out ≈ cross-fit ≈ truth, MLE biased low (v_toy0 N=1000 seed0:
  mle 0.989 < evidence 0.994 ≈ truth 1.0; toy1h-ws all ≈0.09). Misspec (linear fit to sin):
  all four estimators ABSORB to ≈0.285 (>> 0.09 noise) — safe inflation, not collapse. Cross-fit ≈
  held-out with **lower across-seed variance** (v_toy0 N=1000 crossfit range 0.948–1.001 vs held-out
  0.914–1.037). Evidence converged 3–5 iters, γ≈p.
- **CAVEAT (omitted from the V1-claim as framed):** the **NN honest-residual variant on v_toy0 did
  NOT converge** (`converged:false`, rel_change 0.05–0.45) and **over-estimates σ² 2–4×** (held-out
  2.4–4.1, cross-fit 1.6–3.7 vs truth 1.0). It works on toy1h (converged → ≈0.09). Honestly recorded,
  but must **not** be read as a cross-fit success: the honest-residual family requires a converged
  mean. This is a boundary condition, not a refutation (linear arms + NN-on-toy1h both confirm the
  mechanism), and it is exactly the risk V3 inherits.

### V2 — GO with CAVEAT (the nuanced verdict; V2_findings is largely honest)
- **Calibration criterion: MET (verified).** Arm (d) cross-fit SSR @N=1000 = [0.958, 0.930, 1.022],
  3/3 in [0.9, 1.1].
- **Ranking criterion: REFUTED (verified).** Pre-reg (d)≤(e)<(b)<(c)<(a). Observed σ-ratio-err
  @N=200: **b(0.139) < d(0.146) < c(0.162) < a(0.163) < e(0.280)**; @N=1000 all arms 0.054–0.079
  (disease gone). β-NLL wins σ-ratio-err AND held-out NLL at both N (verified). Per-tercile recal (e)
  is the WORST arm at both N (verified) — coarse steps on smooth σ(x) + tiny (~20-pt half) calibration
  splits.
- **V2_findings accuracy:** the mechanism account (disease is finite-sample; recal worst; cross-fit
  backfires at small N) is **correct and honest**. Two minor imprecisions to correct in the write-up:
  (1) it renders the N=200 order as "(b)<(c/d)<(a)<(e)" but **d beats c** — the order is b<d<c<a<e;
  (2) the cross-fit N=200 blow-up has a specific extra mechanism worth naming (below), not just "noisy
  OOF residuals."

**"(a) beats (d)" STOP-condition ruling — NOT A STOP.**
- **The premise as stated ("a beats d on σ-ratio-err at N=1000") is metric-misstated.** On the
  seed-**mean** σ-ratio-err at N=1000, **d beats a** (0.0618 < 0.0633); a beats d only in individual
  seeds (seed0; seed2 by 0.00004). Where a **robustly** beats d is **N=200 on held-out NLL
  (0.573 vs 0.912) and SSR (1.640 vs 2.119)**, driven by seed 1 (d NLL 1.799, SSR 4.116).
- **This is not a harness bug.** Verified: `cross_fitted_residuals` produces honest out-of-fold
  residuals (train on other K−1 folds, no leakage); `train_sigma_head` fits log-σ² correctly (mean
  pinned at 0, `nll_loss` reused); the deployed scoring mean is a full refit, not a fold model. No
  plumbing defect.
- **Mechanism (discharges the "until proven otherwise" burden):** at N=200, K=5 folds → each fold
  mean trained on ~160 pts → noisy out-of-fold residuals → a flexible σ(x) head over-fits that noise.
  **Plus a design mismatch specific to arm (d):** the σ head is calibrated to the *fold* means'
  residuals but is deployed alongside a **different, overfit full-data mean** (`v1.train_mean_mlp`,
  2000 epochs, no early stop), whose held-out generalization error is larger → SSR > 1. Concentrated
  in seed 1. At N=1000 both effects wash out and d is the best-calibrated arm and beats a on all three
  mean metrics. Ruling: the finite-sample account holds; **NOT a STOP.** (The pre-registered process
  — trigger → prove-otherwise → escalate to R4 — was followed correctly.)

## 4. V3 / V4 spec check

### V3 — CONFIRM the ladder, with REQUIRED refinements
The v0-scalar → v1-tercile → v2-linear-log-σ → v3-MLP ladder, selected by the held-out-NLL knee
(`_capacity_ladder.knee` on a per-example held-out log-likelihood table), is architecturally sound
and the unification claim is valid in principle (the knee already reads a generic (N,C) table;
C = σ-rungs is a valid instantiation). Changes V3 needs given what V1/V2 actually showed:

1. **REQUIRED — the deployed mean must be honest/early-stopped before the σ-rung score table is
   built.** The knee selects on held-out NLL, and V0 proved held-out NLL is dominated by *mean*
   generalization error when the mean overfits. Arm (d) as implemented deploys an overfit full-data
   mean — reuse **arm (c)'s early-stopped mean**, then cross-fit the σ rungs on *its* honest
   residuals. Otherwise the knee reads mean-overfit, not σ-capacity. This is the single most
   load-bearing refinement (grounded in V0 + the knee reader's definition).
2. **Pre-register V3 at adequate N (≥1000).** V2 showed the registered rung-fitting mechanism
   (cross-fit) is noise-limited at N=200. Small-N is β-NLL territory; do not fit rungs by cross-fit
   there. State this explicitly rather than letting the knee run on unstable N=200 σ tables.
3. **V-toy2 does not exist** (`make_v_toy2` absent; V2 ran on V-toy1 1-D only). Either implement the
   5-D known-σ(x) toy or drop the "V-toy1/V-toy2" pre-reg to V-toy1 only. Real spec gap — the
   mechanism ranking is currently established on 1-D smooth σ(x) alone.
4. **Do NOT add β-NLL as a rung.** β-NLL is a training-loss axis orthogonal to σ-capacity, not a
   σ(x) function class. Keep it as the labelled baseline and the documented small-N fallback (it won
   V2), but the ladder rungs stay as σ-capacity classes.
5. **Tercile rung (v1) — keep but expect the knee to skip it on smooth σ.** V2 arm (e) (tercile
   recalibration) was the worst arm on smooth σ(x). A directly-fit tercile-σ rung differs from (e),
   and is the honest low-capacity option for genuinely piecewise σ; but the V-toy1 pre-reg knee
   should land at **v2 (linear-log-σ)**, not v1 — σ(x)=0.1+0.3·sigmoid(4x) is smooth/monotone.
6. **Homoscedastic-twin knee = v0: SUPPORTED** by V1 (global-scalar mechanisms recover the constant
   σ² → higher rungs add no held-out NLL). Keep this pre-reg.

### V4 — port targets and the lambda-objective decision
- **`UncertaintyMethod.CONSTANT` from train → validation residuals: SUPPORTED.** V1 shows in-sample
  residual variance is biased low (MLE (N−p)/N; collapse for flexible means); held-out recovers truth.
  Prefer **cross-fit** over a single held-out split (lower variance) where N and convergence allow.
- **`BinnedUncertaintyMixin.calibrate_uncertainty` on a held-out split: SUPPORTED** (same logic;
  align the partition with the K7/V4 readout split).
- **Tree-model Gaussian-NLL deprecation notice: JUSTIFIED.** `losses.py:51–101`
  `tree_model_gaussian_nll_objective` is the same joint-NLL gradient
  (grad_log_var = 0.5·(1 − resid²/var)) — same disease. Notice-only, no re-engineering (as scoped).
- **`learn_regularization_lambdas` objective — move to EVIDENCE for the linear model only; to a
  HELD-OUT criterion for MLPs.** V1's evidence machinery is **exact and validated for the
  linear/linear-in-basis case** (recovers truth + weight-decay α/β) — move the lambda objective there
  for that case. **V1 did NOT test evidence on an MLP** (the NN variant only did held-out/cross-fit
  σ², no MLP evidence). So Laplace/Immer-2021 MLP evidence is **unvalidated here** and must not be
  adopted on V1's strength alone; use the **held-out criterion** for MLPs (model-agnostic,
  V1-validated via cross-fit). The plan's "either evidence or held-out, decided at R4" is thus
  decided: **evidence (linear) + held-out (MLP)**, not evidence everywhere.

## 5. Uncertainty / what remains unverifiable
- The unification claim (same `_capacity_ladder.py` readers run unchanged for σ) is a design claim;
  no V3 run exists yet, so it is architecturally endorsed, not empirically verified.
- Mechanism ranking is established on 1-D smooth σ(x) (V-toy1) only; V-toy2 (5-D) was never run.
- Conformal N=200 undercoverage (0.872) is judged within-noise for ~40 calibration points across 3
  seeds; not separately stress-tested.
