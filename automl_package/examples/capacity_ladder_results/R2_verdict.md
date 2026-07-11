# R2 adjudication — WS1 (K-lane) headline: per-input count from ONE nested-k model

Adversarial verification + synthesis of the WS1 result across K4 (nested-ladder builder +
coherence bars) and K5 (per-input reader). Every load-bearing number below was re-derived
from the artifacts (`K4/k4_summary.json`, `K5/k5_summary.json`, the `.pt` tables via the
readers) and every metric definition was confirmed in source
(`_capacity_ladder.py`, `_capacity_ladder_nested.py`, `capacity_ladder_k4.py`,
`capacity_ladder_k5.py`) before it was trusted. No training was run; no model code was
changed.

---

## 1. BOTTOM LINE

**Shippable, in a qualified form.** The WS1 headline holds on the readout it actually
names — the held-out **arbiter** A(x) = neighbour-averaged advantage of the top-rung
mixture over the k=1 single Gaussian. On the fixed-mode staircase (toy D) the region-mean
arbiter rises monotonically 1→2→3 on **all three seeds**, reproducing the June benchmark;
the per-input hard **knee** is confirmed *not* to be the faithful readout (it is noisy and
wrong on D); recovery is honestly **seed-fragile-to-failing on the moving-mode hump (toy E)**;
the single-mode negative controls **abstain** at both readouts WS1 relies on; and nesting
carries a **small, localized cost** at the middle transition rungs that does **not** corrupt
the top-rung arbiter. Two framing fixes are required before it goes in the report: (a) do
**not** call nesting "costless" — `worse_by_gt_2se` fires for 8 of 9 structured cases, the
cost is real but small; (b) present the broad-twin negative control via the **arbiter** and
the **global knee r\*=0**, never via the per-input-knee abstain fraction (which is a weak
0.41 on C_broad seed 2 precisely because the knee is unreliable — the very point of S2).

Verdicts: **S1 GO · S2 GO · S3 GO (sharpen wording) · S4 GO (reframe control) · S5 GO
(drop "costless") · S6 GO.**

---

## 2. PER SUB-CLAIM

Metric provenance confirmed first (this is what the whole verdict rests on):
- `capacity_ladder_k5.py:117,133` — `arbiter = perinput_curve(score, x, W, ref_c=0)["delta"][:, -1]`,
  i.e. neighbour-averaged `score[:, k_max] − score[:, 0]`. `_capacity_ladder.py:384` confirms
  `ref_c=0` is column 0. `_capacity_ladder_nested.py` selftest (3) + `score_table` c=1 path
  confirm **column 0 is the k=1 direct single Gaussian**. So the readout is exactly the
  documented "top mixture vs single Gaussian advantage, neighbour-averaged."
- June bench transcription confirmed against source: `kselection_variational_em.md:842-847`
  gives staircase `-0.009/+0.147/+0.201` and hump `-0.018/+0.149/-0.026` — byte-identical to
  `JUNE_BENCH` in `capacity_ladder_k5.py:56-59`.

### S1 — D recovered on all 3 seeds via the arbiter → **GO**
`region_arbiter_mean` for D (first/second/third), truth k\*=1/2/3:
- seed0: `0.0080 / 0.1275 / 0.1894` — monotone rising ✓
- seed1: `-0.0165 / 0.0904 / 0.1648` — monotone rising ✓
- seed2: `0.0065 / 0.1295 / 0.1621` — monotone rising ✓

All three strictly rise first<second<third, matching June's `-0.009/+0.147/+0.201` ordering
and magnitude band. I re-derived seed0 first (`0.00801`) and second (`0.12747`) directly from
the raw `arbiter_advantage` array over the correct query-index ranges — they match the
summary exactly, and the region boundaries (thirds) align with the `kstar` staircase steps.
Recovery is of the regional **ordering/magnitude**, not a hard per-input integer — which is
exactly what the June instrument delivered. **GO.**

### S2 — arbiter, not knee → **GO**
D `region_modal_knee`: seed0 `{1,2,4}`, seed1 `{1,1,1}`, seed2 `{1,1,2}`. Matches the claim
verbatim. The knee under-reads catastrophically (seed1 flat at 1 across the whole staircase;
seed2 misses third=3) and over-reads (seed0 third=4 > truth 3); only seed0/second lands. The
arbiter recovers on all three where the knee recovers on none. Methodological claim supported
by the data. **GO.**

### S3 — E moving-mode recovery is a seed-fragile finding → **GO (sharpen wording)**
E `region_arbiter_mean` (low/mid/high), truth 1/2/1:
- seed0: `-0.0240 / 0.0746 / 0.0372` — weak hump (mid is peak) but the descending arm does
  **not** return to negative (high `+0.037` vs June `-0.026`);
- seed1: `-0.0032 / 0.0004 / 0.0143` — no hump, weakly monotone-up;
- seed2: `+0.0081 / -0.0047 / +0.0003` — **inverted** (mid is the minimum).

`b_order` for E: pairwise `[0.908, -0.656, -0.838]`, `min_corr = -0.838` — two of three
seed-pairs are anti-correlated: the ladder ordering itself is unstable across E seeds. The
finding is genuine and honestly disclosed. Two caveats: (i) the wording should say "recovery
does **not** hold on moving modes for 2 of 3 seeds; one seed shows a weak, incomplete
correct-direction hump," not anything implying partial success; (ii) the sub-claim that a
prior h64/1500-epoch capacity/epochs probe "did not rescue s1/s2" is **not verifiable from
these artifacts** — no such probe output is in K4/K5. It is corroborated by memory note
`project_nested_k_component_starvation` ("fix works on D, E stays fragile") but treat it as an
asserted, not re-verified, prior result. **GO** on the finding; fix the wording; flag the
probe as unverified here.

### S4 — broad-twin negative control clean; C_broad s2 crack → **GO (reframe the control)**
The control holds on **both** readouts WS1 depends on:
- **Global knee (K4):** all 6 broad cases `r*=0` (abstain), delta-curve max `= 0.0000`
  (every capacity ≤ k=1 on pooled data).
- **Per-input arbiter (K5):** max **positive** advantage is `≤0.0164` nat on all 6 cases
  (C_broad `[0.010, 0.004, 0.014]`, E_broad `[0.016, 0.016, 0.010]`) — an order of magnitude
  below structured D (region-mean 0.16–0.19) and below even E's weak seed0 hump (0.075). (The
  larger *absolute* excursions up to 0.079 are **negative** — single Gaussian beating the
  mixture — the correct direction for single-mode data, not a false multimodal signal.)

C_broad seed2's per-input-knee abstain fraction is `0.408` (10% of query points read k≥3).
This is a crack in the **knee** readout only — and it *corroborates* S2 (the knee is
unreliable), it is not an instrument failure. **Reframe:** report the broad control via the
arbiter (≤0.016 nat) and global knee (r\*=0), and do not headline the per-input-knee abstain
fraction. **GO.**

### S5 — small, localized nesting cost that does not break the readout → **GO (drop "costless")**
`b_coh = mean(nested_rung_k − same-arch fixed-k)`, one-sided failure = nested worse by >2·SE
(`capacity_ladder_k4.py:137-147`). Verified:
- Cost range across all 9×6 cells: **[−0.1258, +0.0885]** nat (task claim [−0.13,+0.09] ✓).
- `worse_by_gt_2se` fires at rungs (per seed): C `{2}/{2,3}/{1}`, D `{1,3}/{2,3}/{2,3,4}`,
  E `{1}/{2}/{}` — **concentrated at the middle transition rungs 2–3**, and the **top rung
  (6) is never worse** (all positive: +0.0095…+0.0860).
- Largest single cost is **D seed2 rung 3 = −0.1258**, and **D seed2's arbiter still recovers**
  the staircase (`0.0065/0.1295/0.1621`, monotone) — the cost at the middle rung does not
  propagate to the top-rung readout.

So "small honest cost that does not break the readout" is defensible. **But** it fires
somewhere for 8/9 structured cases — nesting is **not costless**; the report must say "small,
localized cost," never "costless." Minor residual: D's arbiter reads rung 8 (`delta[:,-1]`),
which is **outside** the b_coh-validated grid (1–6, `ref_grid = 1..min(kmax,6)`); it is
immaterial because D's delta-curve saturates by rung 5–6 (`0.099/0.112/0.111/0.111` at
rungs 5/6/7/8) and rung 6 is validated costless — but it should be acknowledged, not hidden.
**GO** with the wording fix.

### S6 — genuine correction, not p-hacking → **GO**
- The fix was **pre-registered** as R1 amendments A–D (drop agg-sparsity; pin B-coh to a
  same-arch fixed-k control; re-instate the June arbiter; k=1 = direct Gaussian; spread-init).
  K4/K5 implement exactly these.
- **Strictly probabilistic, confirmed in code:** `masked_prefix_nll`
  (`_capacity_ladder_nested.py:129-154`) is pure renormalized-mixture NLL — no penalty, no
  tuned λ, no K_PENALTY; `k_draw ~ Uniform{1..k_max}` re-drawn each epoch is a schedule
  (`:252`). The classifier+regressor is one frozen model; nesting changes only the training
  schedule, as required.
- **The instrument discriminates rather than manufactures positives:** D recovers (arbiter
  0.19), the single-mode controls abstain (positive arbiter ≤0.016), and E *fails* on 2/3
  seeds — all with the identical pipeline. A pipeline overfitting toward positives would have
  broken the negative control (as the invalid K1/K2/K3 prefix ladder did: +24/+229 nats) and
  forced E positive; neither happens.
- **spread-init caveat (not p-hacking):** `spread_init=True` seeds component means at
  y-percentiles (`:234-236`) — an INIT only, no loss term. It is load-bearing (memory
  `project_nested_k_component_starvation`: the surrogate collapses without it) and it does
  hand the components their y-locations for free, so the instrument is exercised in a somewhat
  favorable setup. But it is not a "pass everything" knob — **E fails despite it** — so the D
  recovery is not an init artifact. The x→count routing (the actual per-input signal) is
  learned by the weight net, not seeded. **GO.**

---

## 3. SYNTHESIZED WS1 RESULT PARAGRAPH (for the report)

Training a single over-provisioned mixture-regression surrogate with a nested-prefix schedule
— each example drawn a random capacity k and scored by the renormalized log-likelihood of its
first k components — makes "the first c components" a genuine c-component mixture at every c,
and lets the per-input latent component count be read from one model instead of a fixed-k
sweep. On the fixed-mode staircase target, the held-out arbiter (the neighbour-averaged
advantage of the full mixture over a single Gaussian) rises monotonically across the three
regions on every seed (region means 0.01/0.13/0.19, 0.00/0.09/0.16, 0.01/0.13/0.16 against a
true count of 1/2/3), reproducing the previously documented advantages of −0.009/+0.147/+0.201;
the per-input hard knee, by contrast, is noisy and count-collapsing (modal knee per region of
{1,2,4}, {1,1,1}, {1,1,2}), so the held-out arbiter, not the knee, is the faithful readout.
On the moving-mode hump the recovery is seed-fragile and does not hold: only one of three
seeds shows a weak, incomplete mid-region rise, the other two are flat or inverted, and the
ladder ordering is unstable across seeds (pairwise correlations down to −0.84). The single-mode
negative-control twins abstain — the global knee returns no capacity beyond one and the largest
positive per-input arbiter stays at or below 0.016 nat, an order of magnitude beneath the
structured signal. Nesting is not free but its cost is small and localized to the middle
transition rungs (nested-minus-fixed-k held-out log-likelihood within [−0.13,+0.09] nat,
recovering by the top rung), and it does not corrupt the top-rung arbiter — the seed with the
largest middle-rung cost still recovers the full staircase.

---

## 4. RED FLAGS / residual checks I could not close

1. **"Costless" vs the data (must-fix wording, not a result problem).** The K4 bar is named
   "nesting-costless," but `b_coh_any_worse = True` for 8 of 9 structured cases. The summary
   itself records this honestly; only the narrative must not claim costlessness. Not blocking.
2. **The arbiter rung for D is not directly coherence-checked.** D's arbiter reads rung 8;
   b_coh validates only rungs 1–6. Immaterial (saturated curve, rung 6 costless), but it is a
   gap in the literal "the rung we read is the rung we validated" chain. Acknowledge it.
3. **S3's compute-artifact rebuttal is asserted, not re-verified here.** The h64/1500-epoch
   probe that reportedly failed to rescue E s1/s2 is not among the K4/K5 artifacts. The
   seed-fragility itself is fully evidenced; the "not a capacity/epochs artifact" clause rests
   on a prior run I could not load. Low risk (corroborated by memory), but flag it.
4. **Discovery-shaped coverage is thin by construction.** This lane tests recovery on 3 toys ×
   3 seeds. D (fixed modes) recovers; E (moving modes) does not. The honest scope of the
   positive claim is *fixed-mode* per-input counting; generalization to moving modes is an
   open negative, not a solved problem. This is a scope statement, not a defect — but the
   report must not overclaim beyond the staircase.
5. **spread-init favorability (see S6).** D's success is partly enabled by seeding component
   means at y-percentiles. Legitimate and documented, but the setup is favorable; a fully
   from-scratch init is untested here (and, per memory, collapses).

**Recommendation: SHIP** the WS1 headline in the qualified form above (fixed-mode recovery via
the arbiter; knee disavowed; moving-mode fragility stated as a negative; controls abstain;
small localized nesting cost), after the two wording fixes (drop "costless"; reframe the broad
control off the per-input-knee abstain fraction). No re-run required.
