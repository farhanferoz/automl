# Capacity-ladder program — running results ledger

Working ledger of certified findings, updated as each adjudicator checkpoint (R1–R4) closes.
This is the FACTS-FIRST accumulation; the polished, cold-readable program report (REPORT-2,
`docs/capacity_ladder_report_<date>/`) is authored separately from this ledger once all lanes land.
All paths absolute. Every claim below is tied to an adjudicator verdict file.

## Status (2026-07-10)

| Lane | Tasks done | Certified by | Headline |
|---|---|---|---|
| WS1 (ProbReg k) | K0, K1–K3, K4, K5 | R1, **R2 (GO, qualified)** | Fixed-mode per-input count recovered via held-out arbiter; moving-mode recoverable-with-more-data (T3, power-limited-not-absent) |
| WS2 (FlexNN depth) | F0, F1, F2, F3 | **R3 (GO, qualified)** + ⟦REV⟧ | Aggregate depth DETECTION discriminates w/ clean control (dedicated-sweep knee: G r\*=2 {2,2,2}, H marginal, G-flat r\*=0); nested-knee reads (G r\*=3, H +0.64 first inc) are interference-biased (X5); per-input read at the noise floor at N=500 |
| WS3 (variance) | V0, V1, V2, **V3** | **R4 (GO, qualified)** | Collapse is finite-sample; β-NLL best point σ(x), cross-fit best calibration; **V3 unifies σ-estimation with the knee (coarse σ-selector)** |

In flight: none — every capacity-ladder lane (WS1–3 + X-queue) and every per-input-selector task
(S/T/H/P) has run and been fresh-context-certified.
Status: **capacity-ladder program COMPLETE** (WS1/WS2/WS3 through R-checkpoints R2/R3/R4; X-queue
X1–X4b certified; F4 closed no-go). **Per-input selector program (S/T/H/P) COMPLETE through compute +
certification** (S1, S2, T3, T2, H1, P1, T1 all adjudicated, folded below). **R-INT folded into
REPORT-2** (new §6 "the per-input selector program" + §7 boundary rewrite + Summary). **Both user gates
resolved:** G-COMMIT done (commit df3982e, deliverables; `*.pt` excluded); **G-FORK → Path 2 RUN →
NOT_FOUND_UNLEARNABLE on all 3 configs → path-3-as-fallback outcome** (T1 finding stands; certified
`T1/T1_PATH2_ADJUDICATION.md`). NOV-1 and NOTE-MOE landed earlier (see "Reports of record").

**Ledger audit (2026-07-10):** every load-bearing WS1/WS3 figure re-derived from the source JSON
artifacts (K4/K5/K6/V0/V2/V3). All verified exactly or within rounding. Fixes applied: K6 pilot count
7/9→8/9 (transcription error) and broad-arbiter bound 0.016→0.017 (0.0164 max). No other discrepancies.

**Independent review audit (2026-07-10, post-REPORT-2; review of record = EXECUTION_PLAN.md §8):**
three fresh-context lane audits re-derived every headline number (all match), verified split hygiene
end-to-end (incl. K6 router: even/odd disjoint target/eval halves — leakage-free), and confirmed the
V3 under-resolution ANALYTICALLY (population NLL gap of optimal-tercile vs true smooth σ = 0.00291 nat —
a hard ceiling at the knee's detection floor; published mechanism: Camporeale & Carè IJUQ 2021, exact
per-residual NLPD ties). Corrections applied in place below (marked ⟦REV⟧): W3 F2-cost characterization,
W4 G per-bin split-fragility, W5 v2-vindication overclaim. Gate closures recorded: W1 (R1's B-knee
"decisive check" fails at face value on D — global knee r\*={2,0,0}; corroborates the knee-under-reads-
count finding; walk stalls exactly at the middle rungs where b_coh costs fire; arbiter remains the
certified readout), W2 (G5 guard re-run ON THE ARBITER: region means move ≤0.086 nat under half-width
shrink vs 0.13–0.19 effects → PASSES; arbiter region means also now carry bootstrap SEs — D regions 2/3
clear 2·SE on 3/3 seeds; E s1 mid genuinely null 0±0.002, s2 significantly inverted −0.006±0.002).
R4's per-seed "(a) beats (d)" sentence is wrong (a beats d 1/3 seeds, not 2/3; mean-based ruling
unaffected); NN honest-residual cross-fit over-estimation range is [1.64, 3.96] not "1.6–3.7".
R-verdict files are historical artifacts and are not edited.

---

## WS1 — per-input mixture-component count (K-lane). Verdict: `R2_verdict.md` (GO, qualified)

**Headline (shippable, fixed-mode only).** Training ONE prefix-nested-k surrogate (per-sample
k~Uniform, masked-prefix renormalized-mixture NLL) and reading the **held-out arbiter** —
the neighbour-averaged advantage of the top capacity rung over the k=1 single Gaussian —
recovers the per-input latent component count on **fixed-mode** data. The read is the arbiter,
**not** the per-input knee.

- **D (staircase k*=1→2→3): RECOVERED on all 3 seeds.** Region arbiter mean rises monotonically
  first<second<third every seed (s0 .008/.127/.189, s1 −.017/.090/.165, s2 .007/.130/.162),
  reproducing the June instrument's documented arc (−0.009/+0.147/+0.201).
- **The per-input KNEE is unfaithful** (D modal knee across seeds {1,2,4}/{1,1,1}/{1,1,2}) while the
  arbiter recovers — the arbiter, not the knee, is the faithful readout. (This directly predicts K6's
  knee-labelled router will underperform its responsibility-labelled router.)
- **E (moving-mode hump k*=1→2→1): does NOT recover for 2/3 seeds** — a genuine negative. Only s0
  shows a weak, incomplete hump; s1 flat (review SE: mid 0±0.002, genuinely null), s2 inverted
  (−0.006±0.002, significantly); cross-seed ladder ordering anti-correlates (b_order min_corr −0.838).
  The positive claim is scoped to fixed modes. **⟦REV W8⟧ Sharper framing:** the June NON-nested
  instrument recovered this hump (−0.018/+0.149/−0.026, apparently single-fit), so the failure is
  plausibly NESTING-SPECIFIC — a single global component-importance ordering cannot serve an
  x-varying ordering — rather than generic per-input-count unidentifiability. Hypothesis + the
  discriminating experiment (3-seed non-nested run on identical E data) queued as EXECUTION_PLAN
  §8.5 X4.
- **⟦REV W1⟧ R1's B-knee "decisive check" — gate now closed explicitly.** The nested D global knee
  reads r\*={2,0,0} across seeds — it FAILS R1's global-knee-reproducibility bar at face value, and
  R2/this ledger had not reconciled that. Closure: the failure is itself evidence FOR the program's
  central finding (the knee under-reads component count even at the aggregate level — the REPORT-2
  round-4 reframe); mechanism hypothesis: the knee walk stalls exactly at the middle rungs where the
  b_coh nesting costs fire (s0 rung-3 cost → r\*=2; s1/s2 rung-2 cost → abstain). The arbiter remains
  the certified readout. **⟦REV W2⟧** The G5 locality guard, originally run only on the knee, was
  re-run ON THE ARBITER by the review: D region means move ≤0.086 nat (mean 0.028) under half-width
  shrink vs 0.13–0.19 effect sizes → PASSES.
- **Single-mode negative controls abstain (clean).** Broad twins C_broad/E_broad: global knee r*=0 on
  all 6 cases and max positive arbiter ≤0.017 nat (max 0.0164, vs 0.16–0.19 for structured D). Report the control
  via arbiter + global knee, NOT the per-input-knee abstain fraction (the C_broad-s2 0.408 abstain
  frac is a knee artifact, corroborating the knee-unfaithfulness, not an instrument failure).
- **Nesting has a small, localized cost (NOT costless).** b_coh (nested rung minus a same-architecture
  dedicated fixed-k model) ranges [−0.126, +0.089] nat; the `worse-by-2SE` flag fires for 8/9
  structured cases but concentrates at the middle transition rungs (2–3) and never at the top rung.
  D s2 carries the largest cost (rung 3, −0.126) yet its arbiter still recovers the staircase — the
  cost does not break the readout.
- **Not p-hacking:** the corrected pipeline (valid nested training + arbiter readout + spread-init)
  was amendment-driven (R1 A–D); the same pipeline discriminates (D recovers, E fails, controls
  abstain). Strictly probabilistic (masked-prefix NLL, no penalty/λ; k~Uniform is a schedule).

**K6 — the per-input read distills into a deployable router (and corroborates the knee finding).**
A small router π(x) trained off the K4 tables (no ladder retraining), routing each input to its k:
- **The responsibility-trained (SOFT) router beats or ties the global-k model on all 9 cases**
  (soft ≤ global 9/9), with clear held-out-NLL wins on D (0.856<0.885, 0.860<0.949, 0.826<0.922)
  and E-s0 — distilling the per-input read is deployable and improves on a single global k.
- **SOFT ≤ HARD(knee-label) router on 7/9** (the two exceptions, C-s2 and E-s1, are marginal ≤0.014
  nat and on the weak/fragile toys) — a direct downstream corroboration of the knee-unfaithfulness
  (R2 S2): training the router on the noisy knee labels is worse than on the smoother responsibilities.
- **PILOT (raw per-example argmax label) underperforms SOFT on 8/9** — as pre-registered; the lone
  exception is C-s2 (the degenerate single-mode case, pilot 0.6030 edges soft 0.6075 by 0.004 nat),
  consistent with a limited-capacity router implicitly smoothing noisy point labels.
- Router-vs-K5-knee agreement is low/variable (0.03–0.85), reflecting the knee's own noise; the router
  nonetheless beats global, i.e. it learns a smoother deployable map than the raw knee read.
Provenance verified (2026-07-10): a clean-room rerun from the current ruff-clean code (selftest + ruff
green) reproduced every per-case NLL bit-identically (max abs diff 0.0 across 45 values) — fully
deterministic, current-code. Artifact: `K6/k6_summary.json`.

Artifacts: `K4/k4_summary.json`, `K5/k5_summary.json`, `K6/k6_summary.json`, `R2_verdict.md`. June
reference: `docs/kselection_variational_em_2026-06-13/kselection_variational_em.md`.

---

## WS2 — per-input depth capacity (F-lane). Verdict: `R3_verdict.md` (GO, qualified)

**Headline (shippable, as REVISED by the 2026-07-10 review ⟦REV X5⟧).** Training ONE prefix-nested-depth
`FlexibleHiddenLayersNN` surrogate (per-sample depth d~Uniform{1..6}, shared trunk, depth-d readout, BN off)
and reading held-out depth score tables (N=500/seed) recovers depth capacity **at the aggregate level with a
clean negative control** — but the honest aggregate read is the **dedicated fixed-depth sweep** (computed by
F2 alongside the ladder): it reads **G r\*=2, per-seed {2,2,2} (coherent)** and **H r\*=2 pooled (marginal;
per-seed {2,3,0})**, and **abstains on the uniform-complexity control G-flat (r\*=0 everywhere)** — no false
positive. The NESTED ladder's own knee (G r\*=3 {4,2,3}; H first increment +0.64) is biased upward by its
depth-decaying interference cost and must not be quoted as the depth requirement. The **per-input
(per-bin) depth read is power-limited at N_TEST=500** — the same aggregate-robust / per-input-power-limited
through-line as WS1 (arbiter vs per-input knee) and WS3 (σ-recovery vs NLL-knee). Every F3 number was
**independently re-derived from the raw F2 `.pt` tables** (from-scratch numpy readers, not a re-call of the
library) and matched `f3_summary.json` to machine precision (max |Δπ̂| = 8.2e-15); the F3 selftest is green
(+0.83-nat synthetic per-bin signal recovered). No reader bug.

- **G (varying required capacity: linear x<0 / compositional x≥0) — per-bin beats global: NOT
  ESTABLISHED (split-fragile) ⟦REV W4⟧.** Only seed 2's tercile advantage clears (+0.0122 ± 0.0046,
  +2.67·SE) under F3's one hardcoded fit/score split (seed 2026) — the review re-ran the identical
  comparison under 8 alternative split seeds and it clears 2·SE on only **3/9 partitions** (most null or
  negative), so the hit reads as one favorable draw of a split-sensitive statistic, not a stable signal.
  (R3's robustness check varied only the bootstrap seed, not the partition.) 2/18 per-bin cells at ~2.3%
  nominal is within chance expectation. Honest read: at N_TEST=500 (~83 points/tercile-half) the per-input
  depth read is at the noise floor — suggestive at best; fix = repeated-split averaging (EXECUTION_PLAN
  §8.5 X3) before F4 leans on it. **Global knee detects on every seed** (per-seed r\* {4,2,3}, pooled
  r\*=3, pooled π̂ concentrates c3=.418 / c4=.356) — the global read is unaffected.
- **G-flat (negative control, uniform complexity) — ties / no false positive: MET CLEANLY (the load-bearing
  bar).** 0/3 seeds beat global at terciles AND 0/6 cells at sextiles; global knee **abstains r\*=0 on all 3
  seeds + pooled** (first Δ increment −0.0076, below 2·SE; full Δ-curve flat/negative). The bypass-confound
  control passes on both readouts — the instrument invents no per-input structure on smooth data, which is
  what makes the underpowered G positive interpretable.
- **H (SNR dial, fixed mean / σ(x) varies) — varies with SNR: NOT MET (directionally unconfirmed).** The
  tercile argmax capacity varies across SNR bins on 2/3 seeds, but the pre-registered *direction* (capacity
  down from high-SNR to low-SNR) holds on none (`snr_trend_down=False` all 3; the varying seeds trend up /
  hump), and no seed clears >2·SE at terciles (s2 only at sextiles). H has no analytic per-input depth ceiling
  (only σ(x) varies), so "varies with SNR" was F3's own operational reading; on the evidence it is
  inconclusive, reported as a non-confirmation rather than a signature. Global knee still detects (per-seed
  r\* {3,2,2}, pooled r\*=2).
- **Sextile stability = the noise-floor tell.** G s2 loses its tercile win at sextiles; H s2 *gains* a sextile
  win it lacked at terciles; G-flat is 0/6. Single-cell flips at half the bin width are the G5-style signal
  that the per-bin reads sit at the noise floor at N=500.
- **Nesting is not costless (F2) — and the cost is LARGE at the shallow end, not "small/localized"
  ⟦REV W3⟧.** F2's coherence bar failed (`B_coh_all_depths_all_toys_pass=false`); the actual pattern
  (review re-derivation from `f2_summary_*.json`): 26/54 vs-fixed cells fail (48%), max cost **−0.72 nat
  (~17.5·SE) at depth 1 on H**, and the TOP rung fails 5/9 — WS1's "concentrated at middle rungs, never
  the top rung" language does NOT transfer. Mechanistically sensible (one shared output layer serves all
  6 depths; depth 1 gets the least dedicated capacity). It does not change the F3 verdict, but (i) F4/F5
  must not claim small nesting cost, and (ii) the interference DECAYS with depth (d1→d3 on H:
  −0.66/−0.04/−0.02), so every early nested increment inherits a positive interference differential.
  **⟦REV X5 — cross-check DONE from the F2 artifacts' own `fixed_depth_ll` tables
  (`REVIEW_2026-07-10/fixed_vs_nested_knee.json`): the dedicated fixed-depth knee on identical held-out
  rows reads G pooled r\*=2, per-seed {2,2,2} (coherent!) vs nested r\*=3 {4,2,3} — the nested G read is
  one rung HIGH (its c2→c3 increment +0.026 is −0.009 for dedicated models ≈ the measured d2−d3
  interference differential). H: dedicated first increment +0.049 (not +0.64 — 13× nesting-inflated);
  pooled r\*=2 agrees but per-seed {2,3,0} is fragile — H detection is real-but-marginal at N=500.
  G-flat abstains on both instruments.** Unifying mechanism with WS1's B-knee failure (W1): the nested
  knee inherits the GRADIENT of the nesting cost — WS1's mid-rung costs stall the walk (under-read,
  D {2,0,0}); WS2's decaying depth-1 interference inflates early increments (over-read, G 3 vs 2). The
  honest aggregate read comes from the dedicated sweep (already computed by F2) or must carry the
  coherence profile; corrected WS2 headline: **aggregate depth DETECTION discriminates with a clean
  control on both instruments (G strongly, H marginally), but the nested ladder's selected depth is
  biased by its own interference profile.**
- **F4 GO'd, qualified; import the K6 lesson.** R3 GOes F4 (the distilled n_predictor router) but scopes the
  deliverable to K6's actual shape — a router that ties-or-beats global + a hard-routing compute saving, NOT a
  decisive per-input NLL win on the thin G signal — and requires training the router on the **soft per-bin
  responsibilities**, not hard knee labels (WS1: soft router beat/tied global 9/9; hard knee-label router
  worse 7/9). F5 stays ⛔ user-gated.
- **Strictly probabilistic.** Nested depth draws = a schedule; stacking/knee = held-out log-score operations;
  no penalty/λ.

Artifacts: `F2/nested_toy{G,G_flat,H}_seed{0,1,2}.pt`, `F2/f2_summary_{G,G_flat,H}_s012.json`,
`F3/f3_summary.json`, `R3_verdict.md`.

---

## WS3 — variance estimation without in-sample collapse (V-lane). Verdict: `R4_verdict.md` (GO, qualified)

**Headline.** Joint (mean, log-σ) Gaussian-NLL in-sample **variance collapse is real but
finite-sample** — present at N=200, essentially gone by N=1000. The remedy depends on regime;
the pre-registered single ranking of the fix battery is **refuted** and reported as such. **V3
closes the lane by unifying variance estimation with the capacity-selection mechanism**: the same
held-out-NLL knee that selects mixture-k (WS1) selects the σ(x)-function-class here — it abstains to
a scalar σ on the homoscedastic control and detects heteroscedasticity, but is a *coarse* selector
(under-resolves the correct smooth rung that σ-recovery identifies), mirroring WS1's knee-vs-arbiter gap.

- **V0 — collapse demonstrated on the trajectory.** N=200: in-sample σ̂/σ ratio falls 1.55→0.90→0.80
  (crosses below 1) while held-out SSR rises 1.05→1.24→3.74; held-out NLL is U-shaped. Nuance: the big
  held-out-SSR blow-up is dominated by **mean overfitting** (the (μ̂−f)² gap), which is the *cause* of
  the σ̂ bias — motivating the key V3 refinement. Linear MLE bias = (N−p)/N exactly (closed form).
- **V1 — mechanism ranking on ground truth.** Well-specified: evidence ≈ held-out ≈ cross-fit ≈ truth,
  in-sample MLE biased low. Misspecified mean: honest-residual estimators ABSORB the bias (safe
  inflation, not collapse). Cross-fit ≈ held-out with lower variance. **Caveat (load-bearing):** the
  NN honest-residual variant on a non-converged mean over-estimates σ² 2–4× — the honest-residual
  family requires a converged/honest mean.
- **V2 — σ(x) fix battery.** Calibration criterion **MET** (cross-fit SSR@N=1000 = [0.958,0.930,1.022],
  3/3 in [0.9,1.1]). Ranking criterion **REFUTED**: at N=200 order is b<d<c<a<e (**β-NLL best**);
  at N=1000 the disease is gone so all density arms tie. **β-NLL is the most robust point estimator**
  (best σ-ratio-err + NLL at both N). **Per-tercile recalibration is the WORST arm** (coarse steps on
  smooth σ(x)). The "(a) beats (d)" STOP condition was adjudicated **NOT a bug** (finite-sample; no
  cross-fit leakage).

- **V3 — the variance-capacity ladder unifies with the k-ladder; the held-out-NLL knee is a
  *coarse* σ-selector.** The SAME `cl.knee` reader that selects k in WS1 selects σ-capacity here,
  reading a held-out per-example log-density `(N, C=4 σ-rungs: v0 scalar / v1 tercile / v2 linear-log-σ
  / v3 MLP)` table built on a cross-fitted, EARLY-STOPPED-mean honest residual (R4 refinement 1).
  Known-answer selftest (2000 ep) PASSED: constant-σ → v0 (abstain), linear-log-σ → v2. Real run,
  N=1000, 3 seeds:
    - **Homoscedastic twin (v_toy1h) → v0 on all 3 seeds (clean ABSTAIN).** Negative control passes
      exactly as R4 refinement 6 predicted; the instrument invents no σ-structure (|Δ-curve| ≤ 0.006
      nat, all within SE).
    - **Heteroscedastic v_toy1: the NLL-knee lands v1 (2/3 seeds) / v0-abstain (1/3), NOT the
      pre-registered v2** — but this is an NLL-*resolution* limit, not a function-class miss. On the
      ground-truth σ-recovery metric (`sigma_ratio_error`), **a higher-than-tercile rung is the best
      σ-recoverer, but WHICH rung wins is itself N-dependent ⟦REV W5⟧**: at N=1000 v2 wins 2/3 seeds
      (seed2 has v1 0.0388 and v3 0.0393 both beating v2 0.0424); at N=4000 **v3 dominates v2 on all
      3 seeds** (e.g. 0.0105 vs 0.0389) — sensible, since σ(x)=0.1+0.3·sigmoid(4x) is not exactly
      linear-in-log-σ, so the MLP out-resolves the linear rung once N supports it. The earlier
      "v2 vindicated on all 3 seeds" wording was an overclaim. The held-out-NLL surface is nearly flat
      across v1/v2/v3 (spread ≤ 0.004 nat; v1→v2 increment ≤ 2 SE), so the parsimony knee stops at v1.
    - **Not instrument bias.** The selftest (strong synthetic σ, v1→v2 increment +0.032 nat) resolved
      v2 cleanly — the knee reads the smooth rung when the NLL signal is present. v_toy1's
      σ(x)=0.1+0.3·sigmoid(4x) is *weak* heteroscedasticity (total NLL gain v0→v3 ≈ 0.014 nat), placing
      the v1→v2 gap below SE at N=1000. **A confirmatory N=4000 run (4× data, `v3_summary_N4000.json`)
      settles the mechanism: the knee STILL lands v1 on all 3 heteroscedastic seeds** (v1→v2 increment
      +0.0006 / +0.0000 / +0.0026 nat — *no larger* than at N=1000), while the weak seed-1 now crosses
      v0→v1 (the σ-signal IS real; more data sharpens v0→v1 *detection*, not the v1→v2 resolution). The
      homoscedastic twin also abstains v0 3/3 at N=4000.
    - **Take-home (parallels WS1's knee-vs-arbiter gap).** The knee / held-out NLL is a *coarse*
      σ-capacity selector: it detects heteroscedasticity and abstains cleanly on the control, but
      under-resolves the correct smooth rung — exactly as the WS1 knee under-reads relative to the
      arbiter. The faithful fine readout here is `sigma_ratio_error` (the σ-recovery analogue of WS1's
      arbiter) — note it needs GROUND TRUTH, so it is a toy-validation instrument, not a deployable
      readout (deployable candidate: a CRPS knee, §8.5 X2). The under-resolution is **intrinsic, not
      ordinary finite-sample — now confirmed analytically ⟦REV⟧**: the population NLL gap between the
      OPTIMAL tercile-σ and the true smooth σ(x) is **0.00291 nat** (numeric integral; the bin nearest
      x=0 carries 0.00745, the others ≤0.00118), a hard ceiling matching every measured v1→v2 increment
      at both N — there is essentially nothing for an NLL knee to resolve, even though σ-recovery
      differs clearly. Published mechanism: log score is the only LOCAL proper score (Gneiting &
      Raftery 2007) and admits exact per-residual NLPD ties σ̃≠σ (Camporeale & Carè, IJUQ 2021,
      primary-verified); quantitatively, the log-score gain from σ-fidelity is second-order (≈E[δ²]
      for relative σ error δ) while σ-ratio-error is first-order (|δ|). A parsimony-on-NLL knee will
      not pay for σ-smoothness the NLL surface is flat to.

**Mechanism recommendation (R4 table):** global σ² linear → **evidence (MacKay α,β)**; global σ²
flexible mean → **cross-fit (iff mean converges)**; σ(x) small-N → **β-NLL**; σ(x) N≥1000 calibration
→ **cross-fit σ(x) head**; intervals → conformal.

**V4 port decisions (locked at R4):** `CONSTANT` uncertainty from validation (prefer cross-fit) not
train residuals; `BinnedUncertaintyMixin` calibrate on a held-out split; tree-model Gaussian-NLL gets
a deprecation notice (same disease, notice-only); `learn_regularization_lambdas` objective → **evidence
for the linear model, held-out criterion for MLPs** (MLP evidence unvalidated here).

Artifacts: `V0/v0_summary.json`, `V1/v1_summary.json`, `V2/v2_summary.json`, `V2/V2_findings.md`,
`V3/v3_summary.json` (+ `V3/v3_summary_N4000.json`, N=4000 confirmatory), `R4_verdict.md`.

---

## Reports of record (2026-07-10, both gate-clean)
- **REPORT-2 (revised post-review; S/T/H/P folded 2026-07-10/11):**
  `/home/ff235/dev/MLResearch/automl/docs/capacity_ladder_report_2026-07-10/capacity_ladder_report.pdf`
  (28pp; 8 cold-read gate rounds; gate log `COLD_READ_TODO.md` in the same folder). Carries every ⟦REV⟧
  correction in this ledger, the nested-vs-dedicated depth table, the analytic V3 ceiling, and a new
  **§6 "From a measurement to a deployable selector"** (S1/S2 recipe, H1 ProbReg two-phase validation,
  T2 dimension degradation, T3+P1 power curves, T1 representable-not-learnable) with §7 boundary rewrite
  (moving-mode resolved as power-limited; boundary = input-dimension + depth-learnability). Tables 8–13,
  Telgarsky 2016 added.
- **NOTE-MOE:** `/home/ff235/dev/MLResearch/automl/docs/moe_contrast_2026-07-10/moe_contrast.pdf`
  (11pp; gate log `COLD_READ_LOG.md`; primary-verified MoE quotes). Cross-referenced with REPORT-2.
- Review of record for the 2026-07-10 corrections: `docs/plans/capacity_ladder_2026-07-09/EXECUTION_PLAN.md` §8.

## Pending / in flight
- WS2 F0–F3 + R3 DONE (2026-07-10): global depth-capacity knee discriminates (G r\*=3, H r\*=2), G-flat
  abstains r\*=0 (no false positive); per-input per-bin read power-limited at N=500 (G 1/3 seeds; H
  directionally unconfirmed). Every F3 figure independently re-derived from the raw F2 `.pt` tables (matched
  to ≤8e-15). **F4 (distilled router): NO-GO — closed by the X-queue (X3+X1 certified, see below); F5 is the ⛔ far-future real-model port (user-gated).**
- V3 confirmatory N=4000 run DONE (`v3_summary_N4000.json`): knee still lands v1 on all 3 heteroscedastic
  seeds → the v1-under-resolution is intrinsic (NLL-insensitive to the v1→v2 σ-fidelity gap), NOT
  finite-sample. Both N=1000 and N=4000 V3 reads recorded. **WS3 lane COMPLETE.**
- NOV-1 novelty/positioning, REPORT-2 (now 28pp, S/T/H/P folded 2026-07-10), and NOTE-MOE all DONE.
- **WS1 (K0–K6 + R2), WS2 (F0–F3 + R3), WS3 (V0–V3 + R4) all COMPLETE through their R-checkpoints.**
  The ⛔ far-future user-gated ports are K7 (WS1), F5 (WS2), V4 (WS3).
- **Per-input selector program (S/T/H/P) COMPLETE through compute + certification** — S1, S2, T3, T2,
  H1, P1, T1 each fresh-context-adjudicated and folded (sections below). R-INT folded; both user gates
  resolved (G-COMMIT = df3982e; G-FORK → Path 2 run → NOT_FOUND → path-3 outcome, T1 finding stands). No live jobs.

## X-queue follow-ups (2026-07-10, post-review; all adjudicator-certified, fresh-context)

Run after the post-completion review (user: "continue on the plan"; order X3→X2→X4→X1). Each has
pre-registered bars in `capacity_ladder_results/X{n}/PREREGISTRATION.md`, artifacts in the same
folder, and every load-bearing number re-derived bit-identically by a fresh Opus adjudicator.

- **X3 — repeated cross-fit re-issue of the F3 per-input DEPTH read.** 50 random fit/score splits
  per seed, Nadeau–Bengio split-aware SE (overlapping-split correction ≈7× the naive SD/√S). G
  per-input advantage corrected-t {−0.34,−0.60,+1.00} (0/3 >2), split-pass 0.047 ≤ G-flat control
  0.073; H SNR-trend 0.30 (no majority). The F3 "+2.67·SE" was **seed-2 under one fixed split**.
  **Per-input depth signal at the noise floor — clean negative on the signal-vs-control
  discrimination at N=500; absolute magnitude (summed elpd≈1.96≪4) unresolvable at this N (Sivula).
  ⛔F4 = no-go.** Selftest confirms the estimator fires (t=6.4) on a genuine signal.
- **X1 — hierarchical partial-pooled per-bin stacking (the power-limited-vs-absent discriminator).**
  New `hierarchical_perbin_stack`: θ_b~N(μ,τ²) logit-Gaussian prior, MAP-EM, **τ² fitted by
  empirical Bayes** (B6 evidence family; adjudicator verified θ-grad↔autograd 1.4e-14, τ² = exact
  joint-MAP fixed point; no over/under-pool bug; strictly-probabilistic, no tuned λ). 3rd arm in
  X3's harness on the F2 depth tables. Selftest: recovers structure independent stacking misses
  (t=6.9 vs 1.1). On real G: hier_vs_global 0/3 sig, pass 0.093 ≈ G-flat 0.08 → recovers **nothing**
  (seed-2 hier_vs_indep NEGATIVE — pooling correctly shed X3's spurious fluke). Detection curve at
  G's ~83-pts/bin density: the estimator resolves a per-region gap≈0.1 (≈0.1 nats/ex, t≈10); **G's
  advantage is 2–3 orders below that floor.** Honest claim (NOT "genuinely absent"): *"no per-input
  depth signal above the detectable/control band; summed elpd ≲0.25≪4 = Sivula-unresolvable at
  N=500."* ⇒ **F4 = CONFIRMED no-go on the strongest available estimator** (stronger close than X3
  alone; sub-0.1-nat structure chaseable only via the registered X10 N-sweep). WS2 per-input depth
  sub-lane CLOSED.
- **X2 — CRPS-knee arm on the V3 σ-ladder.** Closed-form Gaussian CRPS (adjudicator: bit-identical
  to an independent reimpl; MC-agree ≤3e-3) read by the same knee; NLL-consistency guard reproduces
  V3 exactly on all 6 units. **Resolution bar FAILS**: CRPS resolves-toward-truth 0/3 seeds (reaches
  v2 above NLL only on s2, where v2 is WORSE than v1 in σ-error; on s0/s1 where v2 IS best it misses
  it). Homo twin abstains 3/3 (no false positive). **Carry-forward: "CRPS — the canonical
  shape-sensitive non-local proper score — does not rescue a fine σ-selector AT N=1000" (NOT "no
  proper score"; one tested). State the N=1000 limiter up front (§8.1 0.00291-nat ceiling; truth
  shifts to v3 at N=4000 — larger-N CRPS arm UNRUN).** Informative negative corroborating the §8.3
  second-order-δ ceiling: the σ-selection limit is proper-score-general, not log-score-specific.
- **X4 — E-lane non-nested discriminator (W8).** June non-nested per-input arbiter
  (mixture-vs-single-Gaussian held-out advantage; reused `probreg_variational_em_toy_e_hump`
  primitives verbatim, adjudicator-verified faithful) on K4-identical E data, 3 seeds. Recovers the
  moving-mode hump on **1/3 seeds** (E_broad control flat 3/3 — no false positive). **Recovery bar
  FAILS → W8 (E-fragility is nesting-specific) NOT confirmed.** Gold-standard Δ* decomposition:
  model-capture 2/3 (s0 humps but estimation-limited at N_te=2500; s2 recovers), **s1 = mixture
  training COLLAPSE.** Correct nested contrast is **K5** (not K4's global knee r*={0,0,0}, expected
  since E is locally bimodal): K5 is also ~1/3-fragile (different winning seed → coin-flip
  fragility). **Leading read: moving-mode per-input count is hard for ANY instrument at this N — but
  a clean W8 refutation is HEDGED pending X4b** (adjudicator-recommended spread-init/multi-restart
  re-run; if s1's collapse flips → model-capture 3/3 → swings back toward nesting-specific).
- **X4b — E-lane non-nested arm under multi-restart (2026-07-10, DISCHARGES the X4 hedge; fresh-Opus
  adjudicated SOUND).** R=8 restarts, keep-best by the training MAP objective (`model.loss` on train
  only — leakage audit CLEAN; else byte-for-byte X4; E_broad control flat 3/3 on both arbiter and
  gold). s1's X4 "collapse" was an **optimization artifact**: its gold Δ*_mid rises from −0.065
  (mixture *worse* than baseline) to −0.004, where the best-of-8 fit simply reduces to the
  single-Gaussian baseline (eff≈1) — the pathology is repaired but **no positive hump emerges**. The
  moving-mode hump is therefore **genuinely absent on s1 at N=1000, independent of optimization**;
  model-capture stays **2/3** and arbiter-recovery **1/3** (s2 only). **Hedge discharged toward
  instrument-general; W8 (E-fragility nesting-specific) NOT supported** — s1 did not flip, and s2's
  +0.023→+0.057 is keep-best improving an already-capturing fit, neither supporting nesting-specificity.
  Artifacts: `X4b/{PREREGISTRATION.md,x4b_summary.json}`, `capacity_ladder_x4b.py`.
  *(Two adjudicator notes, non-load-bearing: (a) the raw automated label `s1_still_collapses_…`
  OVERCLAIMS — the worse-than-baseline collapse was removed; only the hump's absence persists; this
  certified wording supersedes it. (b) X4's s1 was a badly-fit ~2-component mixture (eff≈1.9), not a
  literal single-component collapse; the literal eff→1 reduction is the X4b outcome.)*

**Net:** the X-queue closes both per-input capacity questions as no-gos on the strongest estimators
(count via WS1; depth via F4 = X3+X1), shows the σ-selection ceiling is proper-score-general (X2),
and — with X4b — localizes E-fragility as **instrument-general**: the s1 optimization confound is
repaired by multi-restart yet the hump stays genuinely absent (X4 + X4b). Open follow-ups:
**X10** (power curve — only if chasing sub-0.1-nat depth) and report integration (§8.6 invariant —
X1/X3 supersede REPORT-2 §4's "power-limited at N=500" wording). The ⛔ commit was resolved
2026-07-10 (curated library-only set landed; experiment artifacts incl. X4b remain uncommitted).

---

# Per-input selector program (S/T/H/P) — successor to the capacity-ladder program

Plan of record: `docs/plans/perinput_selector_2026-07-10/EXECUTION_PLAN.md`. Framing: the ProbReg
model is a CLASSIFIER over k classes (never "Gaussian mixture"). Each certified task appended here
on adjudication.

## S1 — selector target-construction factorial (ProbReg). Verdict: GO (adjudicated 2026-07-10, fresh context)

**Headline.** On the frozen K4 held-out per-example log-likelihood tables, a small selector π(x)
trained to imitate **per-tercile-prior responsibilities ("soft")** is the best of five target
constructions, read by the per-input weighted-density blend. The **per-tercile PRIOR is the
load-bearing mechanism; neighbour smoothing adds essentially nothing.** Recipe of record for the
deployable selector = **soft (prior-dominant)**.

- **Winner = soft**, mean blend NLL 0.6807 < soft_smoothed 0.6873 ≈ soft_no_prior 0.6874 <
  raw_argmax 0.6926 < hard_knee 0.7131 (9 structured cases C/D/E × seeds 0/1/2; re-derived to 6 dp).
- **Factorial** (paired blend-NLL diff ± bootstrap SE, n=9): prior effect (1v2) **+0.0067±0.0021
  (3.13·SE, load-bearing)**; softness vs argmax (2v5) +0.0052±0.0023 (2.25·SE, modest); smoothing
  (3v2) +0.0001±0.00009 (1.28·SE but 0.0001 nat, ~60× below the prior effect → **negligible**);
  prior beats smoothing (1v3) +0.0066±0.0021 (3.11·SE).
- **Bars PASSED:** (ii) some soft arm ≤ hard_knee 7/9; (iii) oracle-x ≤ oracle-noisy 9/9 (honest
  bound below the per-point cheating max); (iv) broad-control advantage over global ≤ 0.02 nat, 6/6
  (no invented structure on C_broad/E_broad).
- **Bars FAILED as pre-registered — both adjudicated BENIGN (corrected wordings logged beside the
  originals; NO silent rewrite, per §0b):**
  - **(i)** original "blend ≤ hard for every arm on ≥8/9" FAILED (soft 7/9; only raw_argmax 9/9).
    Cause: the premise is **not a theorem** — `log Σ_c w_c p_c(y)` is a convex combination of the
    per-class densities and is NOT ≥ `log p_{argmax-w}(y)` pointwise (adjudicator counterexample:
    w=[.8,.1,.1], p=[2,.5,.5] → blend 1.70 < hard 2.00); it fails precisely in the DESIRABLE regime
    where the router is confident and its top-weight class is high-density. `_blend_nll` verified
    correct, no miscalibration. The soft-arm's 2 misses are E s1 (0.00033 nat) / E s2 (0.00088 nat)
    — sub-milli-nat ties on the known-fragile moving-mode toy. **Corrected reading:** "for the
    winning arm, mean blend NLL ≤ mean hard NLL over the 9 structured cases, any per-case regression
    < 0.005 nat" — soft SATISFIES (mean blend 0.6807 ≤ mean hard 0.6884; 2 regressions, max 0.0009 nat).
  - **(v)** original "winner ordering vs hard_knee unchanged across the knob grid" FAILED (44/54).
    All 10 flips are on **C_broad (control)** at ≤0.00097 nat (soft≈hard_knee, no signal → coin-flip);
    structured toy **D is 27/27** with a decisive −0.023…−0.094 nat base margin. **Corrected reading:**
    "on the structured knob toy D the winner's ordering vs hard_knee is unchanged across the full grid
    (27/27); on the control it is not required and flips must be sub-0.005 nat (report as the null)."
- **Strict-probabilistic + leak-free CONFIRMED:** blend = log Σ_c w_c p_c(y); oracle-x scores actual
  held-out log-lik routed from a DISJOINT eval-A neighbourhood (eval-B never informs its own route);
  targets = softmax(score + log π_bin) with π_bin an EM-stacked weight; no tuned λ; WIDTH=0.075 is a
  pre-registered/swept kernel half-width, not a loss coefficient.
- **Code fix (cosmetic, non-blocking):** `_factorial_read`'s `a_better` flag was inverted
  (`capacity_ladder_s1.py:273`, used by no bar/winner/diff) — corrected 2026-07-10 (`> 0`).

Artifacts: `capacity_ladder_results/S1/{PREREGISTRATION.md,s1_summary.json}`, `capacity_ladder_s1.py`.
Adjudication: fresh-context Opus, re-derived every headline number to 6 dp; GO to certify S1 and to
build S2 with `soft` as recipe of record; X7 trigger unaffected.

## S2 — direct held-out-likelihood selector (the principled objective, ProbReg). Verdict: GO (adjudicated 2026-07-10, fresh context)

**Headline.** Training the selector π(x) **directly on the deployment metric** (maximize
`mean_i logsumexp_c(log softmax(w(x_i))_c + score_tr[i,c])`, train-half only — L5-compliant, no
derived label, no prior, no tuned λ; the arm the S1 factorial never ran) reaches **parity** with the
best imitation target `soft`, but does **not** beat it. **Recipe of record stays `soft`** (S1's
prior-dominant winner); the direct objective is validated as a legitimate alternative, not a
replacement. Bars read against the config-matched **300-epoch** arm (`direct_300ep`); the 3000-epoch
arm is an under-training diagnostic only.

- **Direct ties soft, does not displace it.** Mean blend NLL over the 9 structured cases: soft
  **0.6807** vs direct_300ep **0.6811** (Δ = +0.0004 nat). The 3000-epoch arm is also a tie/marginal-loss
  (0.6820, +0.0012 nat) — 10× longer training recovers no win, so the tie is a real property of the
  objective, not under-training.
- **Bars PASSED:** **(ii)** broad-control advantage over global ≤ 0.02 nat on all 6 (max **0.00584** on
  E_broad s2 — no invented structure on the controls); **(iii)** hard-read NLL ≤ global on **9/9**
  structured (degrades gracefully — the hard route never underperforms the fixed global column).
- **Bar FAILED as pre-registered — adjudicated BENIGN (corrected wording logged beside the original;
  NO silent rewrite, per §0b):**
  - **(i)** original: "S2 blend NLL ≤ S1-`soft` blend NLL on ≥ 6/9 structured cases AND mean paired
    diff ≤ 0 nat" (equivalently, plan §1 S2: "S2 blend ≥ S1-winner blend on ≥6/9 and mean paired diff
    ≥ 0; prediction: the direct objective wins or ties"). **FAILED both sub-conditions: 5/9** (needs
    ≥6) and **mean paired diff +0.000413 nat > 0**. Cause: the direct objective **ties** soft rather
    than beating it — the miss is a **0.36·SE** near-tie (SE 0.001148; mean/SE = 0.36), well inside
    noise, with per-case |diff| ≤ 0.0064 nat and a 5-pass/4-miss coin-flip split; the 0.0004-nat gap is
    the order S1 itself certified "negligible". This lands exactly on the "ties" branch the
    prereg's own prediction ("wins or ties") admits. **Corrected reading:** "the direct objective at
    least ties the imitation winner — mean paired blend-NLL diff indistinguishable from zero (0.36·SE),
    means 0.6811 vs 0.6807 essentially equal" — S2 **SATISFIES**. Because the tie means the principled
    objective does not *beat* `soft`, the recipe of record is unchanged (SOFT stands).
- **X7 (toy D) does NOT fire → no G-X7 gate.** gap_closure = mean_seeds((global − best_selector_blend)/
  (global − oracle-x)) = **1.6587** ≥ 0.5 (per-seed [2.90, 1.04, 1.04]); best_selector_blend = min over
  {soft, direct_300ep, direct_3000ep}. All three closures > 1 because the best selector's blend NLL sits
  *below* the honest hard-route oracle-x bound on toy D (blend beats hard — S1's certified finding), so
  the deployable-selector question is closed on D. X7 SKIPPED per §0d.
- **Strict-probabilistic + leak-free CONFIRMED (inherited from S1 verbatim):** S1's split
  (TRAIN=even rows, EVAL=odd, eval-A/eval-B disjoint for the honest oracle bound), `cs1._eval_arm`
  blend+hard reads, and `cs1._oracle_reads` are reused unchanged; the only new code is `_train_router_direct`,
  whose loss is `cs1._blend_nll`'s own computation on the train half with `score_tr` a **detached**
  constant (gradient reaches the router only — the ladder is never touched). No label, no prior, no λ.

Artifacts: `capacity_ladder_results/S2/{PREREGISTRATION.md,s2_summary.json}`, `capacity_ladder_s2.py`.
Adjudication: fresh-context Opus (not the producing session), re-derived every headline number
independently from `s2_summary.json` (all four bars + X7, bootstrap SE reproduced at seed=0/n_boot=1000);
bar (i) failure certified BENIGN (0.36·SE tie, matching the prereg's own "wins or ties" prediction);
recipe of record stays SOFT; X7 no-fire (closure 1.6587), no user gate.

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

## P1 — depth power curve (depth-lane analog of T3 / X10 3-point, FlexNN nested ladder). Verdict: NO recoverable per-input depth signal up to N=8000; control-violated at N=8000 (adjudicated 2026-07-10, fresh context)

**Headline.** Sweeping N_test ∈ {500, 2000, 8000} (N_train ×2) on the F2 nested-depth
`FlexibleHiddenLayersNN` ladder and reading each table with X3's repeated cross-fit (50 splits,
Nadeau–Bengio corrected SE) does **not** turn the "unresolvable at N=500" depth read into a
recovery. The tercile per-bin-stacking advantage over global stays pinned near ~0.001–0.0015 nat at
every N while the detection floor falls ~9×, so toy G crosses its floor on 2/3 seeds at N=8000 — but
the uniform-complexity negative control **G_flat crosses on 2/3 seeds at the SAME N**, with an
advantage (+0.0014 nat) statistically indistinguishable from toy G's (+0.0015 nat; Welch t=0.107),
and a *larger* signal/floor ratio (2.06 vs 1.09). The N=8000 crossing is therefore a
floor-drops-below-a-fixed-offset **artifact**, not a per-input depth signal. Read of record:
**per-input depth structure remains below a trustworthy detection floor up to N=8000 — the crossing
is control-matched, so it certifies no recovery.** This is the depth-lane mirror image of the T3
count-lane result: same power-curve apparatus, opposite outcome.

- **Toy G does not cross a TRUSTWORTHY floor at any N.** Seeds cross 0/3 (N=500), 0/3 (N=2000),
  2/3 (N=8000). But the 2/3 at N=8000 coincides with a control crossing (below), so the pre-registered
  control-governance clause voids "recoverable at N=8000." Mean signal by N = +0.00100 / +0.00041 /
  +0.00151 nat — **flat, not growing**; mean floor by N = 0.01206 / 0.00334 / 0.00139 nat —
  monotonically shrinking. The crossing is floor-driven, not signal-driven.
- **Negative control G_flat crosses at N=8000 (control VIOLATED) — the load-bearing finding.**
  G_flat (uniform linear complexity, no varying required depth) crosses 0/3, 0/3, **2/3** with
  per-seed corrected t up to 6.37 and mean signal +0.00144 nat ≈ toy G's +0.00151 nat. The
  instrument reports as much per-input depth structure in the structure-free control as in the real
  toy → it cannot separate structure from no-structure at N=8000. Per prereg §control-bar, the
  instrument is not trustworthy at N=8000; the N is retained with this caveat, not silently dropped.
- **No 5-point extension.** Floors are strictly monotone-decreasing on every seed and mean for both
  toys, and G's crossing sequence [F, F, T] never un-crosses → the non-monotone-curve extension
  clause does not fire. (The control-crossing clause fired instead and is adjudicated here.)
- **Contrast with T3 (count lane).** T3 recovers because the toy signal GROWS monotonically with N
  and the E_broad control stays flat 0/3 at the crossing N; P1 fails both — G's signal is flat and
  the G_flat control crosses at the same N, indistinguishable from the toy. P1 confirms and extends
  the X3/X1 depth-lane closure (F4 = no-go): even the registered N-sweep up to N=8000 surfaces no
  per-input depth signal above the control band.
- **Strictly probabilistic; estimator reused verbatim.** floor/signal are X3's `run_repeated_crossfit`
  (`mu_bar`, `2·se_nadeau_bengio`) read as a bound; no library changes, no penalty/λ; N=500 reuses
  F2's tables, N=2000/8000 retrain the nested ladder at F2's fixed hyperparameters.

Artifacts: `capacity_ladder_results/P1/{PREREGISTRATION.md,p1_summary.json,p1_summary_N500.json,p1_summary_N2000.json,p1_summary_N8000.json,P1_ADJUDICATION.md}`, `capacity_ladder_p1.py`.
Adjudication: fresh-context Opus (not the producing session), re-derived all 18 (toy,N,seed)
signal/floor/crosses triples from the raw per-split arrays (match < 1e-12), the G-vs-G_flat N=8000
two-sample comparison (Welch t=0.107, indistinguishable), and floor monotonicity (strict, both
toys); ruled the N=8000 crossing a floor-below-offset ARTIFACT (reading B), control VIOLATED at
N=8000, no recovery certified, 5-point extension NOT triggered.

## T2 — multi-dimensional count-mechanism port de-risk (kNN-routed selector, toy D in dim). Verdict: GO (adjudicated 2026-07-10, fresh context)

**Headline.** The per-input count MECHANISM survives into multiple input dimensions on the readout
the program treats as faithful — the NLL-advantage of the kNN-routed selector's blend over the
global single-Gaussian (the multi-D analog of WS1's held-out arbiter). At **dim2 (1 nuisance
coordinate) the selector beats global by +0.073 nat mean on 3/3 seeds at 6–9·SE** (gated bar i),
and the variance-matched broad twin manufactures **no** advantage (max −0.012 nat ≤ 0.02, gated bar
iv). The advantage then **degrades to null by dim5** (0/3 seeds significant, crossing_dim=5) and
stays null at dim10 — a report-only degradation deliverable, **not a failure** (the prereg
warned the read might not even survive dim2). The de-risk is that "neighbourhood reads may break in
many dims" is confirmed as a real, measured degradation with an analytic ground truth, bounded by a
clean control — exactly what the real-model ports are gated on.

- **Bar (i) GATED — PASS (3/3, 6–9·SE).** dim2_axis advantage +0.0756/+0.0780/+0.0645 nat vs 2·SE
  0.0174/0.0217/0.0209 (ratios 8.70/7.18/6.17). Magnitude (~8% relative NLL improvement over global
  ~0.89–0.93) is comparable to the 1-D toy-D K6/S1 wins (0.03–0.10 nat) — a scientifically
  meaningful effect, not a tiny-but-significant one. Reproduced bit-identically end-to-end (|Δ|=0).
- **Bar (iv) GATED — PASS.** dim2_broad advantages all negative (−0.0124/−0.0126/−0.0237), max
  −0.0124 ≤ 0.02 nat. The broad control invents no structure. Reproduced bit-identically.
- **Bar (ii) REPORT-ONLY — degradation curve.** mean advantage +0.0727 (dim2, 3/3 sig) → −0.0122
  (dim5, 0/3) → −0.0377 (dim10, 0/3); crossing below 2·SE majority at **dim5**. **The gold
  `capture_rate` curve (0.164→0.330→0.301) must NOT be read as "capture improves with dim":** the
  dim5/dim10 rise is the degenerate all-k=1 collapse (modal-by-tercile `[1,1,1]`) trivially
  capturing the ~1/3 of points with k*=1, while the advantage is null. At dim2 the argmax route
  OVERSHOOTS the count (modal `[2,7,5]/[1,7,8]/[1,5,5]` vs designed `[1,2,3]`, capture only
  0.10–0.24) even as the NLL-advantage is strong — WS1's certified knee-overshoots-count /
  arbiter-is-faithful pattern, reproduced in multi-D. Faithful degradation readout = advantage curve
  + modal profile, not raw capture.
- **Bar (iii) REPORT-ONLY — rotated-vs-axis paired diff, dim5: material gap, but the test is VOID.**
  mean diff +0.971 nat, SE 0.263, |diff| > 2·SE. This is **not** a rotation-sensitivity signal: the
  fixed rotated projection at dim5 concentrates `s` at mean −0.25 (std 0.13), so with the hardcoded
  1/3 cutoff **k*=1 for 100% of points (train and eval, all seeds)** — the rotated toy has no
  staircase at all. The gap is a distributional-shape/toy-construction artifact (the ladder's k=1
  column is miscalibrated on the degenerate single-mode data, nll_global 1.35–2.32 vs ideal ~0.20,
  which the blend partly repairs), the prereg's documented "axis vs rotated marginals not identically
  shaped" caveat in its extreme form. Do not read as kNN rotation-variance; the rotation sub-question
  is effectively unanswered (future fix: recenter/quantile-bin `s`).
- **Strictly probabilistic + leak-free.** kNN soft targets = softmax of neighbour-averaged held-out
  log-score deltas (S1 arm-3 generalization, no prior — the prereg's structural note that the
  load-bearing 1-D prior cannot legitimately transfer to dim>1 without leaking the analytic index
  `s`); router = soft-label cross-entropy; oracle-x bound routes eval-B from a disjoint eval-A
  neighbourhood; no tuned λ/penalty. The dim=1 regression guard is bit-identical to the unmodified
  `train_nested_k_surrogate` (max abs diff 0.0).

Artifacts: `capacity_ladder_results/T2/{PREREGISTRATION.md,t2_summary.json,T2_ADJUDICATION.md,nested_toyD_ndim_<config>_seed<seed>.pt}`, `capacity_ladder_t2.py`.
Adjudication: fresh-context Opus, re-derived all four bars from the raw `cases` array (bar iii
bootstrap SE reproduced to 6 dp) and reproduced both GATED bars' six dim2 units end-to-end from
scratch on CPU (advantage + advantage_se bit-identical, |Δ|=0); selftest green (regression guard
bit-identical, 2-D known-answer `[1,3,6]`, fold-equivalence). Binding conditions: (1) frame the
dim5/dim10 null as report-only degradation, not failure; (2) do NOT read the gold capture-rate rise
as improvement — it is the all-k=1 collapse; (3) bar (iii)'s gap is a toy-degeneracy artifact
(rotated dim5 is k*=1 everywhere), NOT rotation sensitivity.

## H1 — two-phase post-hoc selector vs shipping joint gate vs oracle fixed-k (ProbReg). Verdict: qualified-GO (adjudicated 2026-07-10, fresh context)

**Headline.** The program's RECIPE OF RECORD for ProbReg — freeze the classifier/regression heads
(trained by a per-sample `k~Uniform{1..6}` masked-prefix schedule with the gate quiescent), then
distil a post-hoc soft gate on a held-out-within-train split — is a **validated selector**: it
matches the jointly-trained shipping recipe, matches a standalone router, and matches-or-**beats**
an oracle fixed-k sweep on the broad control. On toy D the two-phase blend beats the shipping joint
(`SOFT_GATING`+`ELBO`, trained as today) on all 3 seeds by 0.078–0.100 nat (bar i, 3/3). The in-net
post-hoc gate is indistinguishable from the standalone S1-SOFT router on all 9 cases (bar iii, max
|ΔNLL| 0.0036 ≤ 0.01, 9/9). On the C_broad control the two-phase blend has **lower** held-out NLL
than the best held-out fixed-k on all 3 seeds (signed b−best_c = −0.0108 / −0.0415 / −0.0028) — it
never underperforms the oracle single-k. Two-phase achieves all this with its density heads trained
on **half** the data (750 pts) that the shipping-joint and fixed-k arms see (1500). Read of record:
**the two-phase post-hoc recipe matches the shipping joint and a standalone router and does not
underperform the oracle fixed-k on broad targets — a validated ProbReg selector.**

- **Bar (i) PASS 3/3 (clean).** Two-phase beats the shipping joint on toy D every seed: b−a =
  −0.0997 / −0.0782 / −0.0798. Mechanism (report-level): the joint SOFT_GATING+ELBO gate stays
  near its prior (arm-a `marginal_p` ≈ [0.51 bypass, 0.10×5] every seed, barely selecting) while
  the two-phase gate actually routes — decoupling head training from gate training avoids the joint
  pathology.
- **Bar (ii) PASS 3/3 under the one-sided reading; the reported two-sided FAIL is a sign artifact.**
  On C_broad, two-phase is lower-NLL than the oracle fixed-k on all 3 seeds. The prereg's `_bar_ii`
  uses a sign-blind `abs(diff) ≤ 0.02` and reports FAIL because it counts seed-1's **0.0415-nat
  improvement** as a miss; the one-sided "(b) not worse than best (c)" reading — a pre-registered
  adjudicator judgment call (PREREGISTRATION.md §0a) — resolves to PASS 3/3. `best_k` per seed =
  {6, 1, 2}; b beats best-c even after excluding arm-c's x-independent k=1 constant. Not a soft-miss
  — an outperformance in the favourable direction.
- **Bar (iii) PASS 9/9 (clean).** The distilled in-net gate and the standalone `_RouterMLP` (same
  S1-SOFT target, same frozen density source, different host) agree to ≤ 0.0036 nat on every case.
- **k=1 asymmetry (documented, non-invalidating).** Arm (c)'s fixed k=1 is an x-independent constant
  Gaussian (`SeparateHeadsRegressionModule` feeds the head the class probability ≡ 1.0, never x),
  while arms (a)/(b)/(d)'s k=1 rung is the x-dependent `direct_regression_head`. This inflates b's
  seed-1 margin (best-c on seed 1 IS that constant) but does not flip any bar; the seed-1 gap should
  not be read as evidence dynamic routing dramatically helps on broad data.
- **Arm (d) fine-tune (report-only, NO bar) is not a uniform no-op — supports keeping the gate
  frozen.** Predicted |ΔNLL| ≤ 0.01 holds on 7/9; fine-tuning (ELBO off) hurt D s0 (+0.0495) and E
  s1 (+0.0279). The frozen two-phase gate is better left alone.
- **Toy E (bar iv, report-only): two-phase beats both the shipping joint and the oracle fixed-k on
  all 3 seeds** (b < a and b < best_c, 3/3). The moving-mode verdict of record stays T3
  (recoverable, power-limited-not-absent), not H1.
- **Strictly probabilistic.** Head training = masked-prefix NLL (`k~Uniform` is a schedule, no
  tuned λ); the gate distils the S1-certified SOFT target; ELBO is the shipping-arm's own term.
  Phase-1/phase-2 train on disjoint index-parity halves of x_train; test set is `seed+500`
  (leak-free).

Artifacts: `capacity_ladder_results/H1/{PREREGISTRATION.md,h1_summary.json,H1_ADJUDICATION.md}`,
`capacity_ladder_h1.py`.
Adjudication: fresh-context Opus (not the producing session), every bar re-derived from
`h1_summary.json` (bars i/ii/iii/iv, all three §0a judgment calls, the k=1-excluded best-c
recompute, and arm-d deltas reproduced exactly). Bar (ii)'s reported FAIL certified a
sign-blindness artifact (seed-1 is a 0.0415-nat *improvement*, not a miss); resolved to the
pre-registered one-sided reading → PASS 3/3. qualified-GO; binding condition = report bar (ii) as
"matches-or-beats oracle fixed-k on all 3 broad seeds," never as a soft-miss, and note the k=1
x-independence caveat on the seed-1 margin.

## T1 — provably-deep-required toy (depth-lane discriminator). Verdict: learnability-vs-representability finding; bar-(i) FAILED 0/3 and HARDENED unlearnable under multi-restart; PARKED at a user G-FORK (path 2 vs path 3); H2 LOCKED (adjudicated 2026-07-10, fresh context)

**Headline.** T1 built a toy where the per-input depth requirement is PROVABLE by construction —
region A (x<0.5) linear, region B (x≥0.5) a 5-fold-composed tent map (Telgarsky, 32 linear pieces) at
trunk width 8, σ=0.1 (oracle LL +0.8836). The finding is a clean **learnability-vs-representability
asymmetry**: tent⁵ is representable at depth ≥ 2 yet sits ~1.1–1.2 nat below the σ=0.1 oracle at every
trained depth on all seeds, while the linear region reaches the oracle. It is therefore **not** the
large learnable per-input depth signal the depth lane needed as a positive control. Read of record:
**a provable (Telgarsky-style) depth requirement lives in the GD-unlearnable regime, so it cannot serve
as the depth-lane positive control**; whether ANY large learnable-and-depth-requiring target exists is
the open question the parked G-FORK decides.

- **Bar (ii) — PER-INPUT READ reframed within-plan (regime-contingent).** RESUME's "promote the
  decomposed per-region read" fix was REFUTED: the decomposed reader shares the pooled `pi_global`
  baseline and is equally blind (region-B decomposed t = −0.84, mu = −0.0014 on the planted
  flat-A/+0.8-nat-B table). Bar (ii)'s pooled A-vs-B contrast is a per-input *heterogeneity*
  statistic, not a test of region B's *absolute* requirement (bar (i) tests that directly); on T1's
  asymmetric flat-A/dominant-B design it is a **structural zero** only in the truly-indifferent-A
  regime, and CAN co-pass bar (i) in a hurt-by-depth-A regime (both bars fire on the conflicting
  positive control, t ≈ 72–76). Disposition R1: reframe the PREREGISTRATION bar-(ii) text +
  outcome-semantics (done); construction run allowed to proceed. `T1/BAR_II_ADJUDICATION.md`.
- **Bar (i) — CONSTRUCTION FAILED 0/3 (the depth requirement is not GD-learnable).** On the dedicated
  fixed-depth sweep, region B is stuck ~1.1 nat below oracle at EVERY depth 1–6 on ALL 3 seeds
  (deepest-net gap +1.16/+1.12/+1.07 nat), does NOT progressively climb with depth (total d1→d6 change
  +0.12/+0.02/+0.03 nat — the not-learnable signature, not slowly-getting-there), while region A
  reaches +0.79–0.84 ≈ oracle. `region_b_pass=false` 3/3 → construction bar 0/3. The A-vs-B contrast
  validates the harness inside the run (a scoring bug would depress A too). The plan's width-6/tent⁶
  remedy is REFUTED as wrong-direction (it makes an already-unlearnable target harder). Not
  epoch-limited (A converged; B flat-not-climbing). `T1/FAIL_I_ADJUDICATION.md`.
- **Path-1 — HARDENED "GD-unlearnable, not restart-luck" (the within-plan disambiguation).** R=8
  independent random-init restarts per fixed depth on seed 0, keep-best by TRAINING LL (untouched test
  scored, never selected on — no-test-leak confirmed: at all 6 depths the kept-by-train restart is NOT
  the test-best, and the driver's known-answer selftest verifies the wiring). Region B stays
  **1.168–1.209 nat below oracle at every depth** (min 1.168 @ d4), with NO closing at d≥3 (d3/d5 are
  the worst). Even the optimistic best-of-8-BY-TEST region B stays **1.141–1.171 nat below oracle at
  every depth** (min 1.141 @ d3) — > 0.70 nat under the most generous test-peeking read. Region A flat
  and learnable (+0.831…+0.851, ≤0.052 below oracle). Multi-restart on the same Telgarsky toy does not
  rescue bar (i); the single-init unluckiness hypothesis is closed. `T1/T1_PATH1_ADJUDICATION.md`.
- **Terminal path = G-FORK RESOLVED → Path 2 (bounded search) RUN → NOT_FOUND (path-3-as-fallback
  outcome).** User greenlit Path 2 (2026-07-11): a bounded ≤3-config, same-bars, all-3-seeds empirical
  search for a depth-preferring-AND-learnable target (tent⁴@w8, tent³@w8, tent⁴@w6;
  `capacity_ladder_t1_path2.py`). **All three read `NOT_FOUND_UNLEARNABLE`** (0/3 construction_pass each;
  region_b_pass 0/9 seeds): even best-of-8-scored-on-test, region B stayed >0.7 nat below the +0.8836
  oracle at EVERY depth on EVERY seed — min optimistic gaps 1.095 / 0.980 / 1.098 nat for
  tent⁴@w8 / tent³@w8 / tent⁴@w6 (tent³@w8 the near-miss at 0.98 nat: isolated single-depth B increments
  but never the consecutive d1→d2 AND d2→d3 climb). No learnable depth-requiring positive control →
  **path-3-as-fallback outcome: the representable-but-not-learnable asymmetry stands as the depth-lane
  finding**; per-input depth closes as a compute-saving story. Whether ANY learnable depth-requiring
  target exists remains open (a 3-config search demonstrates the limit but cannot prove non-existence).
  Fresh-context certified (independent re-derivation from the summary JSONs; leak check PASS on all 54
  config×seed×depth entries): `T1/T1_PATH2_ADJUDICATION.md`.
- **H2 — LOCKED.** Every H2-unlock branch requires PASS (i) first; bar (i) failed 0/3 → all unlock
  branches unreachable. The path-2 bounded search was RUN and passed bar (i) on 0/3 configs (0/9 seeds),
  so no unlock branch is reachable → H2 stays LOCKED under the path-3 outcome. `T1/FAIL_I_ADJUDICATION.md` Q4.
- **Strictly probabilistic + leak-free.** Fixed-depth LL = per-example Gaussian log-likelihood
  (`_fixed_depth_log_likelihood`, F2's formula, reused verbatim); train = `make_toy_t1(seed=0)`, test =
  `make_toy_t1(seed=500)` (disjoint); keep-best selection reads train only; construction bar =
  per-region paired plain-bootstrap SE (`_construction_bar`, reused verbatim). No penalty/λ. Restart
  `r` → distinct `random_seed=r` init.

Artifacts: `capacity_ladder_results/T1/{PREREGISTRATION.md, FAIL_I_ADJUDICATION.md,
BAR_II_ADJUDICATION.md, t1_summary.json, nested_toyT1_seed{0,1,2}.pt, t1_path1_summary.json,
path1_toyT1_seed0.pt, T1_PATH1_ADJUDICATION.md, t1_path2_{tent4_w8,tent3_w8,tent4_w6}_summary.json,
T1_PATH2_ADJUDICATION.md}`, `capacity_ladder_t1.py`, `capacity_ladder_t1_path1.py`, `capacity_ladder_t1_path2.py`.
Adjudication: fresh-context Opus (not the producing session). Path-1 driver newly authored this
session — BOTH numbers and methodology verified: every per-depth region-A/B mean and gap re-derived
from `path1_toyT1_seed0.pt` (kept vectors match `t1_path1_summary.json` to 8.5e-8), keep-best-by-train
confirmed to select on train not test (kept ≠ test-best at all 6 depths; known-answer selftest green),
oracle/region-split/data-provenance recomputed independently. Ruling: HARDENED_UNLEARNABLE (kept region
B 1.168–1.209 nat below oracle at every depth; optimistic best-of-8-by-test 1.141–1.171 nat below at
every depth; no closing at d≥3). Binding condition: report T1 as a learnability-vs-representability
finding PARKED at the user G-FORK; never as a depth-lane positive control, and never as a machinery
failure (region A learnable proves the harness). H2 LOCKED.
