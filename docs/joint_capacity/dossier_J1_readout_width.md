# Dossier — J-1 (parallel A5 tracks, readout-width) joint width+depth candidate

*Strategy dossier for the joint capacity strand (`docs/plans/capacity_programme/width-depth.md`, Task J0).
Covers candidate **J-1** only; J-2/J-3 get their own dossiers if reached. Records the toy, the frozen
pre-registered bars, the seed-0 pilot result, the analysis, and the branch decision.*

Author: (user). No AI provenance.

---

## 1. Summary — VERDICT: J-1 DEAD (multi-track fold substrate bottleneck)

The J-1 readout-width joint toy was built, its data arithmetic re-verified (§5 of the design doc
reproduced exactly), and its seed-0 pilot run to trustworthy convergence. **The pilot FAILED the S1
substrate bar**: the joint network does not fit the multi-track A5 composition task to the required
per-track accuracy (≥0.90) even at full width and full depth — so the width/depth-dial questions
(W-bar/D-bar/X-bar) are unanswerable by gate order. This is a **substrate** failure, not a
selection/routing failure, and it is shared by J-1 and J-2 (same data + fold), so it is **not** a clean
"J-1 free-rides → adopt J-2" fall-through. A discriminating diagnostic (full-width-only training) is
running to decide whether the cause is (a) grid-loss corruption of the shared recurrent block — fixable
without abandoning J-1's data — or (b) intrinsic multi-track-fold difficulty — the toy needs rescoping.

---

## 2. What was built

`automl_package/examples/joint_capacity_toy.py` — one weight-shared recurrent block folding a per-step
input; the 2-D dial is (state-width `w`, unroll-depth `T`), read by per-width heads.
- **Depth dial** `T` = unroll count (anytime exits `T ∈ {2,4,6,8,10}`), reusing the certified
  `RecurrentComposer` mechanism (`depth_composition_toy.py:295`), each exit trained against the running
  (prefix) product per track (the D8b root-cause fix — no impossible targets).
- **Width dial** `w` = active state units of the 64-wide state, ladder `(16,32,48,64)`, prefix-masked
  `state[:, w:]=0` and read by one per-width head (`Linear(64, K_MAX·60)`), reusing the certified
  `SharedTrunkPerWidthHeadNet` mechanism (`nested_width_net.py:222`).
- **J-1 = readout-width:** the block folds at full 64 width; the mask is applied only at the readout.
  (J-2 = block-width applies the mask inside the fold; not yet piloted.)
- **Data (design §3, `j1_arithmetic_check.py`):** `K_MAX=4` parallel A5-involution track slots; per input
  draw `A ~ U{1..4}` and `T* ~ U{6,8,10}` independently; `A` active tracks carry length-10 words committing
  at `T*`, the rest are NOOP (identity). Per-step input = 4 tracks × 5 symbols = 20 dims, 200 over `L=10`.
- **Verification:** selftest PASS; `--probe arithmetic` reproduces design §5 (pearson(A,T\*) = −0.0277,
  12 cells populated, width info-floor 5.9/11.8/17.7/23.6 bits); ruff clean.

## 3. Frozen pre-registered bars (`docs/joint_capacity/joint_frozen_bar_spec.md`, Opus/xhigh adjudicator)

Frozen before the pilot ran (no bar adjusted after). S1 substrate fit (per-track active acc ≥ 0.90 every
(A,T\*) cell); **W-bar** width dial real (acc(w=64)−acc(w=16) ≥ 0.30 on the **A=4** all-4-correct subset,
plus ρ(deployed-w, A) ≥ 0.7); **D-bar** depth graded (per-track knee at w=64 pooled per T\*: acc(T=T\*) ≥
0.95·acc(T=10), acc(T=T\*−2) ≤ 0.35, ρ(deployed-T, T\*) ≥ 0.7); **X-bar** joint router beats val-selected
best-fixed (w,T) and strictly beats both marginal routers on mean-`w·T` compute within 2·SE. Gate order:
S1 → W-bar → D-bar → X-bar.

## 4. Pilot result (seed 0, n=1000/cell ≈ 12k inputs) — `J_TOY_PROBES/joint_readout_seed0.json`

**Convergence trustworthy:** `converged=True, diverged=False, hit_cap=False, trustworthy=True` — the
result is the genuinely-converged state, not undertraining.

**S1 FAILS on every cell** (per-track active acc @ w=64, T=10; need ≥0.90):

| A \ T\* | 6 | 8 | 10 |
|---|---:|---:|---:|
| 1 | 0.788 | 0.756 | 0.746 |
| 2 | 0.734 | 0.667 | 0.662 |
| 3 | 0.692 | 0.645 | 0.634 |
| 4 | 0.676 | 0.602 | 0.577 |

Accuracy degrades monotonically with both the number of active tracks (A) and the commitment depth (T\*).
Downstream bars are consequently all False (moot by gate order): W-bar gap 0.105 (need 0.30) on 1500 A=4
held-out, ρ(deployed-w,A)=0.408; D-bar ρ(deployed-T,T\*)=0.444.

## 5. Analysis

- **Even A=1 underfits (0.75–0.79).** The *same* single-track A5 composition, in the depth-selection toy
  (`depth_selection_toy.py`), fit to ≥0.90 (G-DEPTH S1 passed, 2 seeds). So the joint construction is
  ~12–15pp worse on its easiest cell — the multi-track/grid setting, not the A5 task itself, is the
  regression.
- **Suffix-forgetting telltale.** Per-track accuracy at T=10 is *lower* than at T=T\* (e.g. t\*=6: 0.884 at
  T=6 vs 0.704 at T=10). After a word commits at T\*, the identity-fold suffix (steps T\*+1..10) re-enters
  the shared block and *degrades* the already-correct product — the block does not perfectly preserve a
  committed state across further steps.
- **Two hypotheses for the S1 underfit** (not yet discriminated — a confident-wrong cause is worse than a
  hedged one):
  - **(a) grid-loss / shared-block corruption.** In readout mode one shared block feeds all 20 (w,T) cells;
    the narrow-width heads (w=16/32/48) demand the block nest the full multi-track answer into a prefix of
    the recurrent state. If that nesting is hard/unsatisfiable, those 15 non-full-width cells contribute
    persistent high CE that dominates the mean loss and drags down the full-width fit. Fixable (loss
    weighting, block-mode, or nested training) without abandoning J-1's data.
  - **(b) intrinsic multi-track-fold difficulty.** Folding 4 parallel A5 tracks through one 64-wide block
    from a 20-dim sparse per-step input is simply harder than the depth toy's single 4-dim track, and 64
    units / the shared block cannot fit it regardless of the loss support. Would require rescoping the toy
    (e.g. fewer tracks, wider state, or per-track blocks).

## 6. Discriminating diagnostic (RUNNING)

`docs/joint_capacity/diag_fullwidth_only.py` — retrains the identical net/data/seed with the CE loss on
the **w=64 cell only** (all T). This removes the width-grid dilution while keeping the multi-track fold.
- **If S1@w=64 recovers to ≥0.90** → hypothesis (a): grid corruption. The width-dial mechanism (per-width
  heads on a recurrent state) is the problem, not the data — proceed to a fix (block-mode J-2, or a
  nested-loss weighting) and re-pilot.
- **If S1@w=64 stays ~0.78** → hypothesis (b): the multi-track fold is the bottleneck. Rescope the toy
  (the design's `K_MAX`, state width, or per-track architecture) before any further J-* pilot.

## 7. Verdict & branch decision — J-1 DEAD → J-3

**Diagnostic result (`diag_fullwidth_only.log`):** full-width-only training did NOT recover S1 — it was
*worse* (0.41–0.61) than the full grid (0.58–0.79), also trustworthy-converged. So removing the width-grid
did not help; the grid's auxiliary multi-exit supervision was in fact *helping*. **Verdict: hypothesis
(b) — the multi-track fold is the bottleneck**, not grid corruption.

**Trajectory check (binding — not concluding from an endpoint).** Pilot val CE: 2.45 (epoch 250) → **best
0.465 (epoch 8000)** → flat/rising 0.47–0.49 through epoch 10500 (`converged=True, hit_cap=False`,
patience carried 2500 epochs past best with no improvement). A genuine converged optimum, not a premature
stop — so the ≥4×-budget early-stop-off confirmation (Decision 9) is not needed to settle a gap this large
(0.58–0.79 vs 0.90) with two independent converged runs agreeing.

**Root cause.** Even A=1 underfits (0.79 vs the depth toy's ≥0.90 on the *same* single-track A5 task). The
regression is the 4-slot construction: a 20-dim per-step input (15 dims are NOOP noise even at A=1) folded
through ONE shared 64-wide block that must route 4 independent tracks without interference. It is a
multi-task-interference / input-multiplexing bottleneck, not an A5-composition or a capacity-bit limit.

**Branch:**
- **J-1: DEAD** (S1 substrate fails, converged; not fixable by loss support — diagnostic).
- **J-2: DEAD by shared substrate** (code-verified deduction: at w=64 `_mask` is a no-op, so J-2's
  full-width fold is identical to J-1's; block-mode only adds mid-fold masking burden to the shared block
  → J-2 S1 ≤ J-1 S1 < 0.90). A fast block-mode substrate confirmation is running to convert this deduction
  to a measurement. J-2's purpose (force width into the computation to kill a *free-ride*) does not address
  an S1 *substrate* failure anyway.
- **→ J-3 (single-track, group-order width):** the genuinely-different fallback. It is **single-track
  (K=1)**, reusing the *proven* single-track recurrent fold (the depth toy fit it to ≥0.90), with the
  width dial coming from the order/richness of the group the input's word lives in — so it structurally
  avoids the multi-track-fold bottleneck that killed J-1/J-2. Gated on its own order⊥length arithmetic
  probe (MOD-5) before any pilot. If that probe fails (order & length correlate) → all three designed
  candidates are dead → STOP + escalate (design's once-then-escalate rule; do not invent a 4th).
