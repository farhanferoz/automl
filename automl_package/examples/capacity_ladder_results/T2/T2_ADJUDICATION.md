# T2 adjudication — multi-dimensional count-mechanism port de-risk. Verdict: GO

Fresh-context Opus adjudication, 2026-07-10. Every load-bearing number below was re-derived from
`t2_summary.json`'s raw per-case fields (not its `bars` block) and, for the two GATED bars,
reproduced end-to-end from scratch. Nothing under `T2/` was modified.

Source of truth:
`/home/ff235/dev/MLResearch/automl/automl_package/examples/capacity_ladder_results/T2/t2_summary.json`,
pre-registration `T2/PREREGISTRATION.md`, bar code `automl_package/examples/capacity_ladder_t2.py`
(`_bar_i` L472, `_bar_ii` L479, `_bar_iii` L499, `_bar_iv` L511).

---

## (a) Re-derived bar numbers and PASS/FAIL calls

Independent numpy re-derivation from the raw `cases` array matched the script's reported `bars`
block on every field. Bar (iii)'s bootstrap SE reproduced to 6 dp with the same
`default_rng(0)`/1000-boot generator the script uses.

### Bar (i) — dim2_axis beats global > 2·SE on ≥ 2/3 seeds. **GATED → PASS**
| seed | advantage (nat) | 2·SE | adv/SE | beats 2·SE |
|---|---:|---:|---:|:--:|
| 0 | +0.075642 | 0.017383 | 8.70 | yes |
| 1 | +0.077997 | 0.021724 | 7.18 | yes |
| 2 | +0.064483 | 0.020890 | 6.17 | yes |

n_pass = 3/3 (≥ 2 required) → **PASS**. Margins are 6–9·SE, not marginal. Matches reported
`n_pass=3, pass=true`.

### Bar (iv) — dim2_broad selector advantage ≤ 0.02 nat (S1 convention). **GATED → PASS**
advantages = [−0.012383, −0.012572, −0.023671]; max = **−0.012383** ≤ 0.02. All three are
NEGATIVE (selector does not beat global on the variance-matched single-mode twin — no manufactured
advantage). → **PASS**. Matches reported `max_advantage=−0.012383, pass=true`.

### Bar (ii) — degradation curve (axis configs). **REPORT-ONLY (no gate)**
| dim | mean advantage (nat) | seeds > 2·SE | mean gold capture |
|---|---:|:--:|---:|
| 2  | +0.072707 | 3/3 | 0.16373 |
| 5  | −0.012200 | 0/3 | 0.33013 |
| 10 | −0.037720 | 0/3 | 0.30080 |

**crossing_dim_below_2se_majority = 5** (matches reported). The advantage collapses from a strong,
coherent +0.073 nat at dim2 to null (mean slightly negative, 0/3 significant) by dim5 and stays
null at dim10.

### Bar (iii) — rotated-vs-axis paired diff at dim5. **REPORT-ONLY (no gate)**
per-seed diff (rotated − axis) = [+0.630678, +1.601515, +0.681184]; mean = **+0.971126**,
bootstrap SE = **0.263451**, 2·SE band = 0.526901 → **|mean| > 2·SE → NOT within 2·SE (material
gap)**. Matches reported `mean_diff=0.971126, se=0.263451, within_2se=false`. See the binding
caveat in (b) — this gap is a toy-construction artifact, not a rotation-sensitivity signal.

### Machinery + reproduction checks
- `capacity_ladder_t2.py --selftest` (no disk I/O): (a) dim=1 regression guard **bit-identical**
  (max abs diff 0.00e+00), (b) 2-D known-answer kNN-routed modal capacity recovered `[1,3,6]`,
  (c) fold-equivalence PASS.
- **Both gated bars reproduced from scratch, bit-identical.** Re-ran `run_case` on CPU for all six
  dim2 units (dim2_axis ×3, dim2_broad ×3): `advantage` and `advantage_se` matched the JSON with
  worst |Δ| = **0.00e+00** on both. dim2_axis gold reads recomputed from the saved `col_hard`
  tables matched exactly (capture 0.1024/0.1448/0.2440; modal `[2,7,5]/[1,7,8]/[1,5,5]`).
- Sanity: `oracle_x ≤ oracle_noisy` on all 15 cases; `k_global = 1` on all 15 (single-Gaussian
  global baseline throughout).
- **Strictly probabilistic — confirmed from source.** kNN soft targets = row-softmax of
  neighbour-averaged held-out log-score deltas (no per-tercile prior, per the prereg's structural
  note that the prior cannot transfer to dim>1); router = soft-label cross-entropy; advantage =
  blended log-score `logsumexp_c(log softmax(w)_c + score_c)` minus the global column; SE = i.i.d.
  bootstrap. No tuned λ, no penalty.

---

## (b) Ruling: **GO** to certify and fold, with binding conditions

Both GATED bars pass under the prereg's exact thresholds, reproduced bit-identically end-to-end.
The de-risk succeeded on the readout the program treats as faithful: the per-input count MECHANISM,
read as NLL-advantage over the global single-Gaussian (the multi-D analog of WS1's held-out
arbiter), transfers to dim2 with one nuisance dimension, and the broad control does not manufacture
advantage. The higher-dim degradation (bar ii) and the dim5 rotated gap (bar iii) are the
report-only scientific deliverables. **GO.**

### Binding conditions on the headline wording (all three are load-bearing)

1. **Degradation-to-null-by-dim5 is REPORT-ONLY, framed as degradation, NOT a failure.** The prereg
   pre-committed to no pass/fail at dim 5/10 and explicitly warned bar (i) might even fail at dim2.
   The honest read is: mechanism transfers at dim2 (2 coords, 1 nuisance), advantage crosses below
   2·SE by dim5. Do not label the dim5/dim10 null a "failure."

2. **Do NOT read the gold `capture_rate` curve (0.164 → 0.330 → 0.301) as "capture improves with
   dim."** This is a trap. At dim2 the argmax route OVERSHOOTS the designed `[1,2,3]` count
   (modal-by-tercile `[2,7,5]/[1,7,8]/[1,5,5]`), so exact-count capture is a modest 0.10–0.24 even
   though the NLL-advantage is strong — exactly WS1's certified pattern (the argmax/knee over-reads
   count; the NLL-advantage/arbiter is the faithful readout). At dim5/dim10 the router COLLAPSES to
   modal `[1,1,1]` everywhere (dim10 all seeds; dim5 all but s1's first tercile), and `capture_rate`
   RISES to ~0.30–0.36 precisely because routing every point to k=1 trivially captures the ~1/3 of
   points whose true k*=1. So the rise is the degenerate all-k=1 collapse, not recovery. The faithful
   degradation readout is the ADVANTAGE curve (+0.073 → null) plus the modal profile
   (monotone-overshoot at dim2 → flat collapse at dim5/10) — NOT the raw capture rate.

3. **Bar (iii)'s material gap carries the distributional-shape caveat in its STRONGEST form: the
   rotated dim5 toy is DEGENERATE, so bar (iii) is void as a rotation test — do NOT over-claim
   rotation sensitivity of the kNN read.** Verified directly: the fixed rotated projection
   `s = u·x/√5` at dim5 concentrates at mean −0.25, std 0.13, so with the hardcoded 1/3 cutoff
   **k*=1 for 100% of points on both train and eval draws, all 3 seeds** — the rotated toy has no
   staircase at all (y is a single Gaussian, std ~0.30, ideal 1-Gaussian NLL ~0.20). The axis toy is
   a proper 3-step staircase (k* fractions ~0.33/0.33/0.33). The large rotated "advantage"
   (0.60–1.65 nat) is not the selector recovering rotated per-input count (there is none): it is the
   nested ladder's k=1 column being badly miscalibrated on the degenerate rotated data (reported
   `nll_global` 1.35/2.32/1.35 vs the ~0.20 ideal), which the blend partly repairs. This is the
   prereg's own "axis vs rotated marginals are not identically shaped" caveat, but far beyond a mild
   CLT effect — full degeneracy. Report bar (iii) as a toy-construction/distributional artifact, NOT
   as a kNN-rotation-variance "binning-artifact tell" and NOT as rotation-invariance confirmation.

### Scope note (why the rotated degeneracy does not block GO)
The degeneracy is confined to `dim5_rotated`, which feeds ONLY bar (iii) (report-only). It touches
neither gated bar (both axis: dim2_axis, dim2_broad use `s = x[0] ~ U[0,1]`, proper thirds) nor the
degradation curve (bar ii is axis-only: dim2/5/10 axis). The prereg pre-authorized interpreting —
not pass/failing — a material bar-(iii) gap.

### Recommendation for future work (non-blocking)
If a genuine rotation-invariance test is wanted later, the rotated toy needs a fix (recenter/rescale
`s` to `[0,1]`, or use quantile cutoffs instead of the fixed 1/3, 2/3) so the rotated case is an
actual rotated staircase rather than a single-mode collapse. As built, T2's rotation sub-question is
effectively unanswered.

---

## (c) Ready-to-paste RESULTS.md section

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
