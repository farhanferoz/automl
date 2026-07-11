# P1 adjudication — depth power curve (X10, 3-point), fresh-context ruling

**Adjudicated 2026-07-10, fresh context (Opus).** P1 triggered its own pre-registered
control-violation escalation (`PREREGISTRATION.md` §Pre-registered readings, control bar): the
script emitted `{"G": "recoverable at N=8000", "G_flat_control_holds": false}`, and the prereg
routes a control crossing at any N to a fresh-context adjudicator. Every number below was
re-derived from the raw per-split arrays in the three N-shards, not from the folded summary or the
task framing.

## Ruling (short form)

**Reading (B) is correct.** P1 does **not** certify "recoverable at N=8000." The depth-lane per-input
signal is **below a trustworthy detection floor up to N=8000**; the N=8000 crossing is a
**floor-drops-below-a-fixed-offset artifact**, not a per-input depth signal — the structure-free
negative control G_flat shows the **same ~0.0015-nat advantage** (statistically indistinguishable
from toy G), toy G's signal **does not grow with N**, and the control actually **crosses more
strongly** than the real toy. Reading (A) is refuted.

---

## (a) Re-derived evidence

### Per-(toy, N, seed) signal / floor / crosses

Recomputed from each case's `per_split_mean_diff` (50 splits) via the locked estimator:
`signal = mean_s(mean_diff_s)`, `floor = 2·sqrt((1/50 + n_score/n_fit)·Var_ddof1(mean_diff_s))`,
`crosses = |signal| > floor`. All 18 triples match the shard values to < 1e-12.

| toy    |    N | seed |     signal |     floor | crosses |
|--------|-----:|-----:|-----------:|----------:|:-------:|
| G      |  500 |  0   | -0.0016920 | 0.0098567 |  False  |
| G      |  500 |  1   | -0.0031585 | 0.0105615 |  False  |
| G      |  500 |  2   | +0.0078476 | 0.0157698 |  False  |
| G      | 2000 |  0   | -0.0000408 | 0.0036370 |  False  |
| G      | 2000 |  1   | +0.0003034 | 0.0031304 |  False  |
| G      | 2000 |  2   | +0.0009532 | 0.0032555 |  False  |
| G      | 8000 |  0   | +0.0022247 | 0.0016639 | **True**  |
| G      | 8000 |  1   | +0.0015135 | 0.0013729 | **True**  |
| G      | 8000 |  2   | +0.0008061 | 0.0011433 |  False  |
| G_flat |  500 |  0   | +0.0001741 | 0.0063472 |  False  |
| G_flat |  500 |  1   | +0.0023188 | 0.0062380 |  False  |
| G_flat |  500 |  2   | +0.0032468 | 0.0113314 |  False  |
| G_flat | 2000 |  0   | +0.0001368 | 0.0028871 |  False  |
| G_flat | 2000 |  1   | +0.0004519 | 0.0015275 |  False  |
| G_flat | 2000 |  2   | +0.0010577 | 0.0021598 |  False  |
| G_flat | 8000 |  0   | +0.0017261 | 0.0007793 | **True**  |
| G_flat | 8000 |  1   | +0.0022045 | 0.0006919 | **True**  |
| G_flat | 8000 |  2   | +0.0003960 | 0.0006302 |  False  |

### Per-(toy, N) folded

| toy    |    N | mean_signal | mean_floor | seeds_crossing | signal/floor |
|--------|-----:|------------:|-----------:|:--------------:|-------------:|
| G      |  500 |  +0.0009990 |  0.0120627 |       0        |        0.083 |
| G      | 2000 |  +0.0004053 |  0.0033410 |       0        |        0.121 |
| G      | 8000 |  +0.0015147 |  0.0013934 |     **2**      |        1.087 |
| G_flat |  500 |  +0.0019132 |  0.0079722 |       0        |        0.240 |
| G_flat | 2000 |  +0.0005488 |  0.0021915 |       0        |        0.250 |
| G_flat | 8000 |  +0.0014422 |  0.0007005 |     **2**      |    **2.059** |

### G vs G_flat at N=8000 (the discriminator)

Two-sample comparison of the three per-seed `mu_bar` values (Welch on the between-seed spread):

- G      seeds `[+0.002225, +0.001513, +0.000806]`, mean **+0.001515**, between-seed SD 0.000709.
- G_flat seeds `[+0.001726, +0.002205, +0.000396]`, mean **+0.001442**, between-seed SD 0.000937.
- diff (G − G_flat) = **+0.000073 nat**, SE_diff = 0.000679, **Welch t = 0.107** (p ≈ 0.9).

The real toy's advantage and the structure-free control's advantage are **statistically
indistinguishable** at N=8000. G/G_flat mean-signal ratio = 1.05. If G's ~0.0015 nat were a real
per-input depth signal sitting *on top of* a control baseline, G would exceed G_flat by roughly the
signal amount; it exceeds it by essentially zero.

---

## (b) Ruling A vs B, with reasoning

**Verdict: (B).** Four independent facts, each re-derived above, converge:

1. **Signal does not grow with N.** Toy G mean_signal = +0.00100 / +0.00041 / +0.00151 across
   N = 500 / 2000 / 8000. This is a non-monotone wobble around ~0.001 nat — the signature of a
   fixed small offset, not of a real signal converging to a stable positive value that a shrinking
   floor finally detects. (For comparison, G_flat's is +0.00191 / +0.00055 / +0.00144 — the same
   order, same non-growth.)

2. **The floor, not the signal, produces the crossing.** Mean floor falls monotonically
   0.01206 → 0.00334 → 0.00139 for G (≈8.7×) and 0.00797 → 0.00219 → 0.00070 for G_flat (≈11.4×),
   while both signals stay pinned near 0.001–0.0015. The "crossing" at N=8000 is the floor
   descending *through* a fixed offset, exactly the mechanism (B) describes.

3. **The control shows the identical advantage.** At N=8000 G (+0.001515) and G_flat (+0.001442)
   are indistinguishable (Welch t = 0.107). The negative control exists to establish the null:
   G_flat has no varying-required-depth structure by construction, so whatever produces ~0.0015 nat
   in it is a nuisance/systematic offset (per-bin stacking has more free stacking weights than the
   global stack and captures a sliver of x-dependent nuisance structure in the depth-score table —
   boundary/density effects, the nested trunk's depth-decaying interference varying across x — that
   is *not* "required capacity varying"). Because the same offset appears in the structure-free
   control, the crossing in G cannot be attributed to per-input depth structure.

4. **The control crosses *more* strongly than the toy.** G_flat's signal/floor ratio at N=8000 is
   2.059 vs G's 1.087, and G_flat crosses on 2/3 seeds with corrected per-seed t up to 6.37
   (vs G's max 2.67). The instrument declares *more* per-input depth structure in the toy that has
   **none** by construction than in the toy that supposedly has it. This is the definition of an
   untrustworthy instrument at this N.

Per the locked prereg control rule (verbatim): "A control crossing at any N means the instrument
itself is not trustworthy at that N." G_flat crosses 2/3 at N=8000. Therefore "recoverable at
N=8000" (reading A) **cannot be certified at the one N where the instrument is declared
untrustworthy**. The script's naive reading is a mechanical application of the crossing rule that
the prereg explicitly overrode with the control governance clause — which is why it routed this case
here.

**Contrast with T3 (count lane) — P1 fails the pattern that made T3 a genuine recovery.**

| property                         | T3 (count lane) — genuine recovery | P1 (depth lane) — this ruling |
|----------------------------------|-----|-----|
| toy signal vs N                  | **grows monotonically** (gold_mid 0.025/−0.004/0.057 → 0.064/0.087/0.063 → 0.063/0.090/0.101) | **flat / wobbles** (+0.00100/+0.00041/+0.00151) |
| control at the crossing N        | **flat, 0/3 recovered** (E_broad flat 3/3 at every N, incl. N=1000 where the toy first crosses) | **crosses 2/3** (G_flat crosses at the same N=8000) |
| toy vs control at crossing N     | toy separates from control (mid eff 1.9–3.2 vs control 1.0) | **indistinguishable** (Welch t=0.107) |
| certified reading                | recoverable, power-limited-not-absent | below a trustworthy floor up to N=8000; crossing is a control-matched artifact |

T3 is a recovery because the signal grows and the control stays flat at the crossing N. P1 has
neither property — so the same "power curve" apparatus that certified a real count signal here
returns a clean *negative* on the depth signal. This is consistent with, and strengthens, the prior
depth-lane closure (X3 + X1: per-input depth signal at the noise floor at N=500, F4 = no-go); the
registered X10 N-sweep was the way to chase sub-0.1-nat depth structure, and it finds none that the
control does not also manufacture.

---

## (c) 5-point extension: NOT triggered

The prereg triggers the 5-point extension **only if the 3-point curve is non-monotone** (floor not
monotonically decreasing with N, **or** signal crossing then un-crossing).

- **Floors strictly decrease with N on every seed and on the mean, both toys** (verified above:
  G seeds 0.00986→0.00364→0.00166, 0.01056→0.00313→0.00137, 0.01577→0.00326→0.00114; G_flat
  0.00635→0.00289→0.00078, 0.00624→0.00153→0.00069, 0.01133→0.00216→0.00063). Monotone.
- **No crossing-then-uncrossing**: G's `crosses_2of3` sequence is [False, False, True] — it crosses
  at the last point (N=8000) and never un-crosses.

Both conditions for non-monotonicity are absent → **the 5-point extension is not triggered.** No
orchestrator extension decision is required. (The escalation that *did* fire is the control-crossing
clause, adjudicated here — a distinct trigger from the non-monotone-curve clause.)

---

## (d) Ready-to-paste RESULTS.md section

```
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
```

---

## Verification provenance

- Script: `/tmp/claude-1000/-home-ff235-dev-MLResearch-automl/ea79a76d-b68f-443b-b3c0-b0b76738f061/scratchpad/verify_p1.py`
  (loads the three N-shards, recomputes every triple from `per_split_mean_diff` via the locked
  estimator, the folded means, floor monotonicity, and the G-vs-G_flat two-sample test).
- All 18 per-case re-derivations match the shard `signal`/`floor`/`crosses` fields exactly
  (max |Δ| < 1e-12); the folded `p1_summary.json` `curve` and `readings` reproduce.
- se_nadeau_bengio spot-check (G N=8000 seed 0): sqrt(1.02)·sd_across_splits(0.00082375) =
  0.00083195 = shard `se_nadeau_bengio`; floor = 2× = 0.0016639 = shard `floor`; consistent with the
  prereg formula (1/S + n_score/n_fit = 1/50 + 1 = 1.02).
