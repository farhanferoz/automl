# Joint width+depth toy — design spec (J0 Step 1)

*Design doc for the joint capacity strand (`docs/plans/capacity_programme/width-depth.md`, Task J0).
Delivered for review at the design→orchestration boundary: the three candidates below are specced and
the primary is arithmetic-verified, so J0 Step 2 (pilots) can run autonomously next session. Mirrors the
depth strand's `docs/depth_capacity/depth_selection_toy_design.md` in rigor and discipline.*

Author: (user). No AI provenance. Every number in §5 is reproduced by
`docs/joint_capacity/j1_arithmetic_check.py` (reuses the certified A5 machinery from
`automl_package/examples/depth_selection_toy.py` — no reinvention).

---

## 0. Charter question (joint half)

**Can ONE weight-shared network serve a per-input 2-D capacity dial — width AND depth — where both
demands are genuinely irreducible, and can a distilled router pick each input's (width, depth) with no
oracle labels?** G-JOINT = that positive, with a certified toy, on ≥ 2 seeds. This is the width and depth
charters crossed: G-WIDTH certified per-input width (per-width heads; shared readout broke), G-DEPTH
certified per-input depth (one shared readout over a weight-shared recurrent block) and per-input depth
*selection* by distillation. J0 fuses them.

---

## 1. Settled inputs (from `width-depth.md`) + results-driven constraints

Carried, not re-derived (`width-depth.md` Settled inputs):
- **Asymmetric readout prior:** width needs PER-WIDTH heads (shared readout fails, G-WIDTH); depth wants
  ONE shared readout over a weight-shared recurrent state (G-DEPTH). Joint net = **per-width heads ×
  depth-shared recurrent block**.
- **Depth dial = iteration count** of a weight-shared block. **Selection = DISTILLATION** post-hoc from a
  held-out per-capacity error table (MASTER Decision 13); distilled router is the primary, in-training
  selectors only as labeled comparison arms.
- **Divergence guard** (`convergence.py` `diverged`, depth D6) reads on every J-battery cell.

New constraints forced by the G-DEPTH results (this is what J0 must design around; each traces to a
landed result):
- **MOD-1 — depth axis bounded at L ≤ 10.** The A5 recurrent block hits a GD-trainable wall at L ≥ 12
  (verified: L=16 trained to chance, root-cause in the D8b AS-RUN note). Any candidate whose depth-hunger
  needs > ~10 sequential steps dies the same way. Hard pre-registered constraint + arithmetic check.
- **MOD-2 — the joint toy is CLASSIFICATION (CE), not MSE.** Depth-hunger must be group-structural
  (a smooth 1-D-regression depth axis is closed-by-theorem — `docs/depth_capacity/depth_toy_negative_note.md`,
  Malach et al.), which forces a classification task. Consequence: the width strand's analytic σ² floor
  does NOT transfer. The "floor/ceiling analog" is a **Bayes-accuracy ceiling per (width-demand,
  depth-demand) cell** (computed exactly, §5) plus an **analytic information floor** on state width; the
  realized width-accuracy curve is a PILOT measurement, not a data property.
- **MOD-3 — inherit Option A: do NOT chase hidden demand.** Depth-demand was 100% surface-readable in
  G-DEPTH (S5) and three concealment rounds failed. The contribution is the **2-D crossing** — one net
  serving a (w,T) dial + a distilled 2-D router matching capacity to per-input demand + both dials
  genuinely irreducible — NOT concealment. Surface-readable routing is acceptable per the user's charter
  ("difficulty-of-detection is not our concern").
- **MOD-4 — the G-JOINT gate rewards capacity-MATCHING and must beat BOTH marginal routers** (§6).
- **MOD-5 — a 2-D orthogonality + starvation arithmetic probe is MANDATORY before any pilot** (§5). The
  L=16 failure was arithmetic-checkable pre-run; no pilot on plausibility alone.

---

## 2. The joint net (shared across all candidates)

A weight-shared recurrent block folds a sequence; the 2-D dial is (state-width w, unroll-depth T):

```
state_0 = 0  (w_max-wide)
for t in 1..T:   state = tanh( block([ mask_w(state), input_t ]) )   # unroll T steps = DEPTH dial
logits_track_k = per_width_head_{w,k}( mask_w(state) )               # read at width w = WIDTH dial
```
- **Depth dial T** = unroll count, exactly the G-DEPTH mechanism (`RecurrentComposer`,
  `depth_composition_toy.py:295`). Anytime exits at T ∈ {2,4,6,8,10}; each exit supervised by the running
  (prefix) product (the D8b root-cause fix — no impossible targets).
- **Width dial w** = active state units, exactly the G-WIDTH mechanism (`SharedTrunkPerWidthHeadNet`,
  `nested_width_net.py:222`): prefix-mask `state[:, w:] = 0`, per-width head `Linear(w_max → ·)`.
- `mask_w` placement is the ONLY thing that differs between J-1 and J-2 (§3) — the hedge on the key risk.

---

## 3. The three candidates (priority order)

The candidate set hedges the ONE real uncertainty — **is the width dial a genuine per-input capacity
demand, or can a narrow readout free-ride on a full-width computation?** J-1 and J-2 are the same
task/data with the width mask in two places; J-3 is a different width mechanism.

### J-1 (PRIMARY) — parallel A5 tracks, **readout-width** dial
- **Generative spec.** K_max = 4 parallel track-slots. Per input, draw width-demand **A ~ U{1..4}** and
  depth-demand **T\* ~ U{6,8,10}** INDEPENDENTLY. A active slots each get a length-10 A5-involution word
  with realized commitment == T\* (distinct products, reuse `sample_stratum`); the K_max−A inactive slots
  are all-no-op (5th symbol → identity). Per-step input = concat over slots of onehot₅(letter) → 4×5 = 20
  dims/step, 200 dims over L=10. **Label** = tuple of K_max track-products (inactive → identity).
- **Width-hunger** = A: reading A independent A5 elements out of the state needs ≥ A·log₂60 bits of
  readout capacity (§5[d]). Realized by **masking only the readout** (`mask_w` at readout; block runs at
  w_max). **Depth-hunger** = T\*: track k correct iff T ≥ T\* (below → Bayes(g), §5[c]).
- **Orthogonality** (§5[a], verified): A ⊥ T\* by independent draw + same-t\* per input (avoids the
  max-of-A-draws skew). pearson = −0.028.
- **Metric** = per-track top-1 accuracy; per-input score = all-active-tracks-correct. **Floor/ceiling** =
  the joint Bayes ceiling `bayes(T\*−T)^A` (§5[c]) + width info-floor (§5[d]).
- **Risk it carries:** the full-width block computes all tracks; a narrow readout may still read them, so
  the width dial could be weak. **This is exactly what J-2 hedges.**

### J-2 (HEDGE on J-1's risk) — same task, **block-width** dial
- Identical data to J-1. Difference: `mask_w` is applied to the **recurrent state at every step**
  (`state[:, w:] = 0` inside the fold), so a narrow width genuinely cannot hold A tracks — width-hunger is
  forced into the COMPUTATION, not just the readout. This is the faithful analog of the width toy masking
  the trunk before the readout.
- **Risk:** masking the state mid-recurrence may hurt trainability (the block must work at every width);
  the pilot's S1 (fit) bar catches this.
- Piloted only if J-1's pilot shows the width dial is not real (narrow readout reads A tracks at ≥ the
  Bayes ceiling for A=1) — i.e. J-1 passes fit but fails the width-separation bar (§4 W-bar).

### J-3 (DIFFERENT mechanism) — variable-group-order tracks, width = representation richness
- **Generative spec.** Single track (K=1). Width-demand = the **order of the group** the input's word
  lives in: draw a sub-alphabet from a nested chain of groups (e.g. Z₂ ⊂ Z₆ ⊂ A5-involutions), width-demand
  = the order class (small order → narrow state suffices; A5 → wide). Depth-demand = commitment length as
  in G-DEPTH. Width here is the sinc-toy analog — *richness of the representation the state must hold* —
  rather than parallel count.
- **Risk:** group-order and commitment length can correlate (larger groups mix faster), so orthogonality
  is NOT free — needs its own arithmetic probe (drawn order ⊥ drawn length) BEFORE pilot (MOD-5). Weaker;
  documented as the genuinely-different fallback if BOTH J-1 and J-2 die.
- Metric/floor as J-1 (CE; per-order Bayes ceiling).

---

## 4. Pre-registered pilot bars (seed 0; frozen before each pilot runs)

Per candidate, in priority order. No bar adjusted after its pilot runs (MASTER Decision 9 discipline).
- **S1 — joint substrate fits.** At full (w_max, T=10) the net reaches per-track held-out acc ≥ 0.90 on
  every (A, T\*) cell (the anytime sandwich + width heads didn't break the substrate). Convergence-gated;
  `diverged=false` required.
- **W-bar — the WIDTH dial is real (make-or-break for J-1 vs J-2).** Held-out accuracy at fixed full
  T=10 must RISE with read-width w and the A=4 inputs must need more width than A=1: `acc(w=w_max) −
  acc(w=w_min)` ≥ a pre-registered gap AND width-demand recovers A (Spearman ρ(deployed-w, A) ≥ 0.7). If
  J-1 fails W-bar (narrow reads A tracks fine) → J-1's width dial is a free-ride → adopt J-2.
- **D-bar — the DEPTH dial is graded** (the G-DEPTH S2 knee, re-used): acc(x, T=T\*) ≥ 0.95·acc(x, full)
  per cell; acc(x, T\*−2) ≤ Bayes(g=2)+10pp; ρ(deployed-T, T\*) ≥ 0.7.
- **X-bar — the two dials are jointly routable.** A distilled 2-D router (raw input → (w,T)) deployed
  hard-pick beats the best fixed (w,T) on compute-matched accuracy AND beats both marginal routers (§6).
- **Kill criterion:** S1 fails on 2 seeds, OR neither J-1 nor J-2 passes W-bar (no real width dial), OR
  X-bar shows no joint gain over marginal routing → that candidate dies; fall to the next in priority.
  All three die → STOP, post-mortem into `width-depth.md`, batch the redesign for the user (the D1/D8
  once-then-escalate rule).
- **Cost:** J-1/J-2 share data (12 cells × ~1000/cell, 200-dim input, 10-step fold) ≈ 20–40 min/pilot on
  CPU (the depth pilot was ~9 min at n=3000/3 strata); J-3 similar. All well under the ~2 h target.

---

## 5. J-1 arithmetic verification (MANDATORY gate, MOD-5) — PASS

Reproduced by `docs/joint_capacity/j1_arithmetic_check.py`:
- **[a] Orthogonality** — A drawn independently of T\*, all active tracks at the same T\* (avoids the
  max-of-A-draws skew): **pearson(A, depth-demand) = −0.028 ≈ 0**; all 12 (A, T\*) cells populated
  (min 303 / 4000).
- **[b] Class balance** (per track, stratum t\*=8): 60/60 classes hit, per-class count min/mean/max =
  4/50/123 (the first-hit distribution's inherent non-uniformity, as in G-DEPTH; no starvation).
- **[c] Joint depth Bayes ceiling** `bayes(T\*−T)^A`: at read-T ≥ T\* → 1.0; below, graded — e.g.
  T\*=8: per-track 0.089/0.133/0.250 at read-T 2/4/6 (g=6/4/2); joint-A4 = 1e-4/3e-4/4e-3. A too-shallow
  OR too-narrow read collapses → a genuine 2-D knee.
- **[d] Width info-floor** = A·log₂60: 5.9 / 11.8 / 17.7 / 23.6 bits for A=1..4 (analytic lower bound on
  readout capacity; the realized curve is the pilot's W-bar).
- **[e] Depth wall:** all T\* ≤ 10 ✓ (MOD-1).

J-2 shares this data, so [a]–[e] cover it. **J-3 requires its own [a] probe (order ⊥ length) before its
pilot** — deferred to its turn per the priority order.

---

## 6. Pre-registered G-JOINT gate rule (MOD-4)

**G-JOINT = PASS iff, on ≥ 2 trustworthy seeds, the adopted toy passes S1 + D-bar + (W-bar for its own
width mechanism) + X-bar, where X-bar is:** a distilled 2-D router reading only the raw input deploys a
hard-pick (w,T) per input whose held-out accuracy is ≥ the best fixed (w,T) at strictly less mean
compute (mean w·T), AND strictly beats BOTH marginal routers — width-only (routes w at fixed T=10) and
depth-only (routes T at fixed w=w_max) — on the same compute-matched basis. The joint gain over marginals
is the evidence the 2-D dial is more than two 1-D dials. (Motivated by the G-DEPTH deploy result: routing
to the matched capacity *beat* over-provisioning, so matching — not just saving — is the win.)

Selection is distilled (Decision 13): router labels = cheapest-within-tolerance over the held-out per-(w,T)
error table; no oracle (A, T\*) labels anywhere. In-training (w,T) selectors only as labeled comparison
arms.

---

## 7. Non-goals (J0)

No FlexNN/package model changes (examples-level only); no transformer/token-sequential halting (J*/M3+);
no real data; no MSE crossing (MOD-2); no concealment engineering (MOD-3); no depth axis beyond L=10
(MOD-1).
