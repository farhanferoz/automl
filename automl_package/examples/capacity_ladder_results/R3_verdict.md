# R3 adjudication — WS2 (FlexNN depth) headline: per-input depth capacity from ONE nested-depth model

Adversarial verification + synthesis of the WS2 result on the F2 nested-depth ladder (F2 trained the
`NESTED`-strategy `FlexibleHiddenLayersNN` on toys G / G-flat / H, 3 seeds, and saved a `(500, 6)` held-out
per-example log-density table per (toy, seed)); F3 (`capacity_ladder_f3.py`) ran the K1/K2 battery on those
tables. Every load-bearing number below was **independently re-derived from the raw `.pt` tables** with a
from-scratch numpy reimplementation of the readers (EM stacking, knee, mixture-logscore, quantile bins,
per-bin-vs-global) — NOT by re-calling `_capacity_ladder` — and cross-checked against both `f3_summary.json`
and the `cl` library. No training was run; no model/architecture code was changed. This is a NO-TRAINING
post-hoc read of the F2 tables only.

---

## 1. BOTTOM LINE

**GO-qualified for F4.** The robust, shippable WS2 result is the **global (aggregate) depth-capacity
discrimination with a clean negative control**: the pooled held-out knee reads depth capacity on the
structured toys (**G r\*=3, H r\*=2**) and **abstains on the uniform-complexity control (G-flat r\*=0)** on
every seed and pooled — no false positive. The **per-input (per-bin) depth read is power-limited at
N_TEST=500**: the pre-registered "per-bin beats global by >2·SE on G" is met only on the any-seed
operationalization (1 of 3 seeds, lost at sextiles), and the toy-H "varies with SNR" bar is **not met** on
its pre-registered criterion. This is exactly the WS1/WS3 through-line — aggregate/global capacity selection
is robust; per-input reads are power-limited — and it is stated as such, not papered over.

Verdicts against the three pre-registered F3 bars (EXECUTION_PLAN.md lines 436–438):
**G per-bin-beats-global → MET-BUT-UNDERPOWERED · G-flat ties/no-false-positive → MET (cleanly, the
load-bearing bar) · H varies-with-SNR → NOT MET (directionally unconfirmed).**

The F3 reader is **faithful** — my independent reimplementation matched `f3_summary.json` to machine
precision (max |Δπ̂| = 8.2e-15, all per-bin mean_diff differences ≤ 1e-15) and the selftest is green. **No
reader bug.** The qualifications below are about statistical power at N=500 and one honest interpretation
downgrade on H, not about a defective read.

---

## 2. INDEPENDENT RE-DERIVATION (the core of R3)

Method: loaded the 9 F2 tables directly (`torch.load`, `weights_only=False`); all are `(500, 6)` float64,
`x` is `(500,)`, `c_grid=[1..6]`, all finite. Reimplemented in plain numpy: `logsumexp`, EM stacking (uniform
init, tol 1e-10, 500 iters), the knee (Δ-curve + paired i.i.d. row-bootstrap increment test, `block=None`,
B=1000, seed 0), `mixture_logscore`, `quantile_bins`, and `_perbin_vs_global` (seeded `default_rng(2026)`
250/250 fit-score permutation split, tercile/sextile quantile bins, per-bin stack vs global stack, per-example
log-score diff, plain bootstrap SE seed 1). Verification script:
`/tmp/claude-1000/-home-ff235-dev-MLResearch-automl/9f8f08e1-30bd-46d4-91ea-91601b89f01b/scratchpad/r3_verify.py`
(scratch; not committed).

### Global knee + pooled π̂ — reproduced exactly

| toy | pooled r\* (mine=json) | pooled π̂ (mine) | pooled Δ-curve incr {c2..c6} | per-seed r\* {s0,s1,s2} |
|---|---|---|---|---|
| G      | **3** | [.000, .089, **.418**, **.356**, .002, .135] | +.2613 / +.0257 / **+.0048** / −.0189 / +.0015 | {4, 2, 3} — all >0 (detect), incoherent |
| G-flat | **0** (ABSTAIN) | [.281, .032, .200, .488, .000, .000] | **−.0076** / +.0076 / +.0047 / −.0247 / −.0291 | {0, 0, 0} — coherent abstain |
| H      | **2** | [.000, .334, **.349**, .131, .182, .004] | +.6379 / **+.0118** / −.0134 / −.0096 / −.0027 | {3, 2, 2} — all >0 (detect) |

The knee arithmetic is transparent and I re-walked it by hand from the Δ-curve:
- **G pooled**: c1→c2 +.2613 (≫2·.0153) sig; c2→c3 +.0257 (>2·.0092=.0183) sig; c3→c4 +.0048 (<2·.0062=.0124)
  **fails → r\*=3.** π̂ concentrates on c3(.418)/c4(.356) (the argmax is c3). ✓
- **G-flat pooled**: c1→c2 −.0076 (< 2·.0032) — the FIRST increment already fails → **r\*=0 ABSTAIN** (G2
  sentinel, not "capacity 0 confirmed"). The full Δ-curve is flat/negative (c3 +1.4e-5, c4 +.0047, c5 −.020,
  c6 −.049). ✓ Clean no-read on the uniform-complexity control.
- **H pooled**: c1→c2 +.6379 sig; c2→c3 +.0118 (<2·.0091=.0183) **fails → r\*=2.** ✓

All 9 per-case π̂ vectors, all delta-curves, and all r\* values matched `f3_summary.json` to ≤ 8.2e-15.
`cl.knee`/`cl.stack_em` matched my numpy to ≤ 4.7e-15 (F3 used the library faithfully).

### Per-bin tercile advantage — reproduced exactly (mean_diff to ~1e-16, se_diff to ~1e-17)

| toy | s0 mean_diff ± se (beats) | s1 (beats) | s2 (beats) | F3 bar `pass` |
|---|---|---|---|---|
| G      | +0.00369 ± 0.00655 (no) | −0.01386 ± 0.00619 (no) | **+0.01219 ± 0.00457 (BEATS, +2.67σ)** | `G_perbin_beats_global_2se` = **True** (any-seed; 1/3) |
| G-flat | +0.00236 ± 0.00317 (no) | +0.00625 ± 0.00387 (no) | −0.00225 ± 0.00443 (no) | `G_flat_ties_no_false_positive` = **True** (0/3) |
| H      | −0.00775 ± 0.00614 (no) | −0.00597 ± 0.01045 (no) | +0.01838 ± 0.01113 (no, +1.65σ) | `H_varies_with_snr` = **False** |

Sextile stability re-check: G s2 loses the tercile win at sextiles (+0.00962 ± 0.00998, no); H s2 *gains* a
sextile win (+0.03996 ± 0.01465, +2.73σ) it did not have at terciles. Both are single-cell flips at half the
bin width — the G5-style tell that these per-bin reads sit at the noise floor. G-flat is 0/6 (all cells, both
resolutions) — the control never fabricates a per-bin advantage. I recomputed se_diff with an independent
bootstrap seed (777, 5000 reps): every beats/ties boolean is unchanged (se moves ≤ 0.0002), so the verdicts
are not bootstrap-seed artifacts.

### Selftest — green

`capacity_ladder_f3.py --selftest` PASSES (exit 0): on a synthetic table where capacity genuinely varies by
x-region, the reader recovers a **+0.829-nat per-bin advantage (>2·SE)** and the correct per-bin argmax
capacities {bin0:1, bin1:3, bin2:6}. So the near-zero real-G advantages are **not** a broken reader — the
reader has ~0.8-nat of discriminating power and simply finds ~0.01-nat of real per-input signal on G.

---

## 3. PER-BAR ASSESSMENT (against EXECUTION_PLAN.md F3 pre-registration, lines 436–438)

### Bar 1 — G "per-bin beats global >2·SE" → **MET-BUT-UNDERPOWERED** (power, not absence)
Met on the any-seed operationalization F3 registered (`_any_pass`): seed 2's tercile advantage +0.0122 clears
at +2.67·SE, and it clears the G-flat control ceiling (whose best is s1 +0.0062 at +1.6·SE, never significant)
— so it is a real above-control per-input signal, not a false positive. **But** it is 1 of 3 seeds, the other
two are non-significant (s1 is negative), and the s2 win evaporates at sextiles. The per-bin test at N_TEST=500
uses only the 250-point score-half → **~83 points per tercile** (weights fit on ~83/tercile from the fit-half),
so it is badly underpowered per seed. Honest reading: the varying-capacity signal on G is **present but
power-limited**, not established across seeds. Directly parallels WS1 (arbiter recovers, per-input knee noisy)
and WS3 (knee coarse, fine σ-readout power-limited).

### Bar 2 — G-flat "ties / no false positive" → **MET CLEANLY** (the load-bearing bar)
0 of 3 seeds beat global at terciles **and** 0 of 6 cells beat at sextiles; the global knee **abstains r\*=0
on all 3 seeds and pooled**; the full Δ-curve is flat/negative. On a uniform-complexity toy with matched
marginal stats, the per-input reader manufactures **no** capacity structure and the global reader takes **no**
capacity beyond depth 1. This is the bypass-confound negative control and it passes on both readouts. This is
the bar that makes the (underpowered) G positive interpretable: the instrument does not fire on smooth data.

### Bar 3 — H "varies with SNR (resolution-dial signature)" → **NOT MET** (directionally unconfirmed)
F3's `H_varies_with_snr` = `any(varies AND snr_trend_down)` reports **pass=False**, and my re-derivation
confirms why: the tercile argmax capacity **varies** across SNR bins on 2/3 seeds (s0 high-SNR=3→low-SNR=4;
s1 2→3; s2 3→3 flat), but the pre-registered **direction** — capacity trending *down* from the high-SNR
(low-x) tercile to the low-SNR (high-x) tercile — holds on **none** (`snr_trend_down=False` all 3; where it
varies it trends *up*, with a hump in the middle tercile: argmax by tercile s0 {3,4,4}, s1 {2,5,3}, s2
{3,4,3}). No seed clears the >2·SE per-bin magnitude bar at terciles (s2 clears only at sextiles). Honest
reading: **the read is SNR-sensitive (it varies), but neither the pre-registered downward resolution-dial
direction nor the >2·SE per-bin magnitude was demonstrated.** H has no analytic per-input depth ceiling (only
σ(x) varies, the mean function is fixed), so "varies with SNR" was always F3's own operational reading; on the
measured evidence it is an **inconclusive** bar, not a confirmed signature. I decline to certify it as met.

---

## 4. THE VERDICT — GO-qualified for F4

**GO for F4 (the distilled n_predictor router), qualified.** What F4 may lean on and what it may not:

1. **Robust claim to carry forward.** Aggregate/global depth-capacity selection discriminates (G pooled r\*=3,
   H pooled r\*=2, per-case all >0) **and** the negative control abstains (G-flat r\*=0 everywhere, 0/6 per-bin
   cells). That is the shippable WS2 headline.
2. **Do NOT expect a large per-input NLL win on G.** The per-input signal F4 distills is thin (1/3 seeds,
   ~0.012 nat, at the noise floor). The F4 pre-reg bar "hard-routed ≥ global-knee, >2·SE better on G's mixed
   regions" is **at risk** on this evidence — set the deliverable to K6's actual outcome shape: a router that
   **ties-or-beats global** plus the **hard-routing compute saving**, not a decisive per-input NLL win.
3. **Import the WS1 K6 lesson explicitly.** In WS1 the responsibility-trained (SOFT) router beat/tied global
   on 9/9 while the hard knee-label router was worse on 7/9 — because the smooth responsibilities beat the
   noisy hard labels. F4 should train the router on the **soft per-bin responsibilities** (the stacked π̂ per
   bin), not on hard knee labels, and keep the raw-argmax router only as the registered pilot.
4. **Nesting cost caveat (from F2, not re-adjudicated here).** F2's coherence bar failed
   (`B_coh_all_depths_all_toys_pass=false`) — nesting is not costless on the depth ladder either, mirroring
   WS1. This does not change the F3 verdict (the held-out score-table discrimination and the abstaining control
   are what F3 reads, and they hold), but F4/F5 must not claim costless nesting.

### F5 spec-check note (F5 is ⛔ user-gated far-future — not run here)
The F5 port (EXECUTION_PLAN.md §F5, port to the real `FlexibleHiddenLayersNN`) is sound as scoped, with three
R3 riders: (a) **do not stake F5's bars on the per-input NLL win** — make the port bars the global-knee
discrimination + the no-false-positive control + seed-coherence, which are the robust results; (b) per-input
reads on real data use **kNN locality with the G5 half-width shrink guard**, since the 1-D tercile reads here
are already at the power floor and real data has no analytic ceiling; (c) keep the **BN-off decision (F0)** and
add the **fourth readout partition** (the K7 lesson) so held-out selection never touches the early-stopping
split; and (d) do **not** use pre-Apr-2026 FlexNN depth-reg results as baselines (memory
`project_phase9_bugs`).

---

## 5. F3-READER / DISPATCH-BRIEF DISCREPANCIES FOUND

None are code bugs (my independent reimplementation matches the reader exactly). Recorded for accuracy:

1. **Fit/score split is a seeded permutation, not "index parity."** The dispatch brief described F3's per-bin
   split as "by index parity"; the code (`capacity_ladder_f3.py:182-187`) uses
   `np.random.default_rng(2026).permutation(n)` then a 250/250 head/tail split. Deterministic and reproducible
   either way — immaterial to the numbers, but the brief's characterization is inaccurate.
2. **Per-tercile power is ~83, not ~167.** The brief cited "~167 test points/tercile." The per-bin test uses
   only the 250-point score-half (and fits π̂ on the 250-point fit-half), so it is **~83 points per tercile**
   in each half — i.e. *more* underpowered than the brief stated. Strengthens the power caveat.
3. **H direction downgraded.** The brief framed H as "resolution-dial signature present directionally." The
   measured `snr_trend_down` is **False on all 3 seeds** (the varying seeds trend *up*/hump), so I downgrade
   this to "SNR-sensitive but directionally unconfirmed" — the pre-registered signature is not present.

**Recommendation: GO (qualified) to F4**, with the per-input claim scoped as power-limited (the robust result
is the global knee discrimination + the G-flat no-false-positive control), the K6 soft-router lesson imported,
and H reported as directionally unconfirmed rather than a confirmed resolution-dial. No re-run required; no
model/architecture change.
