# J-3 joint toy — design spec (single-track group-composition width dial)

*Design doc for the J-3 candidate of the joint capacity strand
(`docs/plans/capacity_programme/width-depth.md`, Task J0). J-1 (parallel-track readout-width) and J-2
(parallel-track block-width) are DEAD at the substrate (J0 post-mortem, `width-depth.md` §"J0
post-mortem"): the 4-track fold multiplexes 4 independent A5 words through one shared block and cannot
fit even the A=1 single-track case (0.79 vs the depth toy's ≥0.90). J-3 is the design's own third
candidate. Per the plan's recommendation (Option 1) and [[feedback_toy_design_needs_reviewed_spec]],
this spec is delivered for review BEFORE any build or pilot. Mirrors `joint_toy_design.md` and the depth
strand's `depth_selection_toy_design.md` in rigor.*

Author: (user). No AI provenance. Every group-theory fact in §3 is reproduced by
`docs/joint_capacity/j3_group_probe.py` (to be run as part of MOD-5, §5), reusing the certified A5
machinery from `automl_package/examples/depth_composition_toy.py` (no reinvention).

> **STATUS (2026-07-17, post adversarial review — see §9): R1 is SOUND-WITH-FIXES, NOT build-ready.**
> The structural claim holds (single-track genuinely removes J-1's multiplexing; R1b state-mask is a
> valid width-vs-serial-memory falsifier). But two load-bearing *arguments* are broken as written: (1)
> the width axis (k) and depth axis are entangled on one length-10 word — scoring all-k-correct requires
> reading to the deepest checkpoint p_k, which grows with k, so **k leaks into the depth demand** and
> MOD-5[a]'s scalar `pearson(k,T*)` is blind to it; (2) the `k·log₂60` info-floor is overstated (the
> checkpoint products are correlated; A5 mixes slowly) and is a bits-vs-state-units category error. **§4
> below reads as originally drafted; §9 records the review, the required fixes, and the ONE fix that is a
> PI design decision (fix 2). Do not build until fixes 1–3 are folded in.**

---

## 0. Charter question (unchanged) and J-3's job

**Can ONE weight-shared network serve a per-input 2-D capacity dial — width AND depth — where both
demands are genuinely irreducible, and can a distilled router pick each input's (width, depth) with no
oracle labels?** (`joint_toy_design.md` §0.)

J-3's specific job: supply a WIDTH dial that composes with the proven single-track A5 depth fold WITHOUT
the multi-track substrate that killed J-1/J-2. The two hard requirements J-3 inherits:

1. **Reuse the proven substrate.** The single-track weight-shared A5 fold reaches ≥0.90 held-out accuracy
   at L=10 (G-DEPTH, D5/D8b). J-3 must be single-track (K=1) so it sits on this proven substrate and
   structurally avoids the input-multiplexing interference that made J-1's A=1 case fit to only 0.79.
2. **Keep the depth dial Barrington-clean.** Depth-hunger must come from a source where width provably
   cannot substitute (non-solvable group composition; `depth_composition_toy.py:14-26`). The width dial
   must NOT re-import depth-hunger through the back door (see the confound ledger, §3).

---

## 1. Settled inputs carried into J-3 (not re-derived)

- **Depth mechanism = commitment length T\*** on a FIXED non-solvable group (A5), exactly as G-DEPTH:
  the running product is determined at step T\*; reading at T < T\* collapses to the Bayes floor
  `bayes(T\*−T)` (`joint_toy_design.md` §5[c]). Ladder T\* ∈ {6, 8, 10}, L=10, MOD-1 wall respected.
- **Substrate = the shared 2-layer recurrent block** `RecurrentComposer` (`depth_composition_toy.py:295`),
  `state = tanh(block([state, input_t]))`, one shared readout. The block needs its hidden layer: a single
  Linear reached only 0.46 val representing one A5 multiplication at width 16
  (`depth_composition_toy.py:302-303`) — **this is the empirical hook the width dial will exploit** (§2).
- **Selection = DISTILLATION** post-hoc from a held-out per-(w,T) error table (MASTER Decision 13); the
  distilled 2-D router is the primary readout, in-training (w,T) selectors are labeled comparison arms only.
- **Divergence guard** (`convergence.py` `diverged`) on every J-3 battery cell; MOD-2 (classification/CE,
  not MSE); MOD-3 (Option A — surface-readable routing is acceptable, the contribution is the 2-D
  crossing, not concealment).

---

## 2. The width-demand candidate the design doc sketched — and the empirical hook

The J0 design doc (`joint_toy_design.md` §3, J-3) proposed: **width-demand = the ORDER of the group the
input's word lives in** — draw a sub-alphabet from a nested chain of A5 subgroups, width-demand = the
order class (small order → narrow state suffices; A5 → wide). The empirical hook that makes a width dial
*plausible* at all:

- Representing ONE group multiplication is a genuine per-step WIDTH cost (O(1) depth — it is a table
  lookup, not a sequential computation): the depth toy needed a hidden layer because width-16 could not
  represent the A5 multiplication table (0.46 val, `depth_composition_toy.py:302-303`). A group of order
  o has an o×o multiplication table; a small group (Z₂, o=2) needs ~1 unit, A5 (o=60) needs a wide block.
- So "width-to-compute-the-multiplication" scales with group order and is, per se, an O(1)-depth width
  axis — distinct from "width-to-hold-the-answer" (one element is ≤ log₂ 60 ≈ 6 bits, trivially held).

This is the sketch. §3 shows it is **structurally confounded** and must be replaced.

---

## 3. The confound ledger (the crux) — why the group-order width dial fails, verified

The J0 design doc flagged the risk ("group-order and commitment length correlate"; `joint_toy_design.md`
§3 J-3 risk). Working it out precisely, the confound is worse than a correlation — it is **structural and
inseparable within the A5 subgroup lattice.**

**Fact (Barrington 1989 / Liu et al. 2022, `depth_composition_toy.py:14-26`).** Composition depth-hunger
comes from **non-solvability**: for a non-solvable group, width provably cannot substitute for sequential
program length (depth); for a *solvable* group, the length-L product can be shortcut to an O(1)-depth
shallow-wide net (no depth-hunger). Depth-demand of the *composition* is a function of **solvability**,
not order.

**Fact (verified — `docs/joint_capacity/j3_group_probe.py`, reproduced 2026-07-17).** The distinct
subgroup orders of A5 are exactly **{1, 2, 3, 4, 5, 6, 10, 12, 60}**, and **only order 60 (A5 itself) is
non-solvable** — every proper subgroup (order < 60) is solvable, because A5 is the smallest non-solvable
group. (BFS-closure over singleton and pair generators; solvability by order < 60 ⇒ solvable.)

**Consequence — the width and depth dials collapse onto one point.** On a group-order width ladder built
from A5's subgroup lattice:

| order o (width-demand) | 2 | 3 | 4 | 5 | 6 | 10 | 12 | 60 |
|---|---|---|---|---|---|---|---|---|
| solvable? | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| composition depth-hunger (Barrington) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **> 0** |

Depth-hunger is **zero for every rung except the top**, where it is also the maximal width-demand.
The two axes are not orthogonal — they are absent everywhere and colinear at a single point. A router
that deploys wide-and-deep only for the order-60 input cannot be shown to be matching *two* demands; one
suffices to explain it. **The group-order width dial cannot produce an orthogonal 2-D crossing.** This is
not a tuning problem; it is a property of A5's lattice (A5 simple ⇒ all proper subgroups solvable).

There is no rescue within A5: to have depth-hunger you need non-solvability, which requires order 60,
which maxes width-demand. (Using a larger ambient group with a richer non-solvable subgroup chain — e.g.
S₆, A₆ — would push past the MOD-1 L≤10 GD-trainability wall and reintroduce the multi-element substrate
cost; rejected.)

---

## 4. Resolution — separate the axes onto orthogonal structural features of a FIXED A5

The fix is to **stop varying the group.** Fix the ambient group at A5 (non-solvable ⇒ depth-hunger on
*every* input via commitment length, Barrington-clean), and source the WIDTH demand from a structural
feature that is independent of solvability.

### R1 (RECOMMENDED) — simultaneous k-checkpoint readout on the fixed A5 fold

- **Group:** always A5. **Depth-demand** = commitment length T\* ∈ {6, 8, 10} (as G-DEPTH), the step at
  which the running product is determined; reading at T < T\* → Bayes floor. Depth-hunger present on
  every input, width cannot substitute for it (Barrington).
- **Width-demand = k ∈ {1, 2, 3, 4}:** the input marks **k checkpoint positions** p₁ < … < p_k along the
  length-L word (a per-step `is_checkpoint` bit). The net must, **at the final step**, output the running
  products at all k checkpoints *simultaneously* (k per-checkpoint softmax heads, or one head read k
  times from a held state). Carrying k partial A5 products through to the end forces the state to
  simultaneously encode k independent A5 elements → **state capacity ≥ k·log₂60 ≈ 5.9k bits** — a genuine
  per-input WIDTH demand. k is drawn INDEPENDENTLY of T\*.
- **Why this is single-track and dodges the J-1 failure:** there is ONE A5 recurrence (the proven
  substrate), no multiplexing of independent words. The width demand is on the state's capacity to
  *buffer k snapshots of one trajectory*, not on folding k independent tracks through one block. The
  freeze-and-hold operation is a learnable gate (copy the running product into an accumulator slot when
  `is_checkpoint` fires, then hold), O(1) depth per step → adds NO depth-hunger, keeping k ⊥ T\*.
- **Width mechanism placement (the J-1-vs-J-2 hedge, reused):** the width dial can be masked at the
  **readout** (R1a — read k heads from a full-width state; tests whether a narrow read free-rides) or at
  the **state** (R1b — `state[:, w:] = 0` inside the fold, so a narrow state genuinely cannot buffer k
  snapshots). R1b is the faithful analog; the W-bar (§6) decides whether R1a free-rides and R1b is
  required, exactly as the J-1→J-2 fall-through was meant to.
- **Bayes ceiling:** each checkpoint c has product determined by its prefix; if the read depth reaches
  p_c the checkpoint is exact, else it collapses to `bayes` on the unread suffix. Per-input score =
  all-k-checkpoints-correct (the correctness catch, analogous to J-1's all-active-tracks-correct).

**R1 open risk (for the ledger, §7):** "buffer k snapshots" is a state-capacity demand — is it *width* or
*serial memory*? It is width: the k accumulators must coexist in the state vector at the final step, so
masking state width below k·log₂60 provably prevents it (R1b). The MOD-5 probe (§5) must confirm the
Bayes ceiling actually falls when width < k·log₂60 (the width info-floor), else the demand is not real.

### R2 (REJECTED) — group-order ladder with a solvability-independent depth source

Rejected by §3: within A5 there is no depth source that is both non-trivial and solvability-independent.

### R3 (WEAK fallback) — per-input generator-alphabet richness on fixed A5

Vary the number of distinct generators in the input alphabet per input (2 vs 4 generators). More
generators → richer per-step input decoding → plausibly wider block. But alphabet size changes the
group's **mixing time** and hence the commitment-length distribution → tangles with the depth axis (the
same order⊥length confound in a subtler form). Documented as the fallback if R1's width demand proves not
real (W-bar fails on both R1a and R1b); NOT recommended.

---

## 5. MOD-5 arithmetic probe — MANDATORY before any J-3 pilot

`docs/joint_capacity/j3_group_probe.py` (to be written for R1; reuses `depth_composition_toy.build_group`
and `sample_stratum`). No pilot on plausibility alone — the L=16 depth-wall failure was arithmetic-
checkable pre-run. All checks below must PASS and be committed to the dossier before Step-2 fitting:

- **[a] Orthogonality (width ⊥ depth).** k drawn independently of T\*: `pearson(k, T\*) ≈ 0` (|ρ| < 0.05);
  all (k, T\*) cells populated (k ∈ {1..4} × T\* ∈ {6,8,10} = 12 cells, min count ≥ ~300/battery).
- **[b] Checkpoint class balance.** Per checkpoint slot and per T\* stratum, all 60 A5 classes reachable;
  report per-class min/mean/max (the first-hit non-uniformity is expected, as in G-DEPTH; flag starvation
  if any class count is 0 in a populated cell).
- **[c] Depth Bayes ceiling.** `bayes(T\*−T)` graded per read-depth (read-T ≥ T\* → 1.0; below → the
  A5-uniform floor on the unread suffix), and per-input all-k-correct ceiling = ∏_c bayes(depth at c).
- **[d] Width info-floor (the R1 make-or-break).** Analytic lower bound = k·log₂60 bits
  (5.9/11.8/17.7/23.6 for k=1..4); and — critically for R1 — a **capacity check** that at state width
  w < k·log₂60 the all-k-correct ceiling is provably < 1 (the demand is genuinely a WIDTH demand, not a
  free readout). If this fails (a narrow state can carry k snapshots), R1's width dial is not real → R3.
- **[e] Depth wall.** All T\* ≤ 10 (MOD-1); the freeze-and-hold gate adds no sequential steps (verify the
  reference implementation composes in exactly L unrolls).
- **[f] Barrington-cleanliness.** Confirm the group is A5 on every input (fixed, non-solvable) — so the
  depth axis is the certified non-substitutable one and the §3 order/solvability confound is absent by
  construction.

---

## 6. Pre-registered pilot bars (seed 0; frozen before the pilot runs)

Adapted from `joint_toy_design.md` §4 to the single-track k-checkpoint construction. No bar adjusted
after the pilot runs (MASTER Decision 9). The precise numeric bars (gap sizes, SE multipliers) will be
frozen by the same xhigh-adjudicator pass that set the J-1 bars (`joint_frozen_bar_spec.md`), once the
construction (R1a vs R1b) is chosen — that freeze is part of Step 2, not this spec.

- **S1 — substrate fits.** At full (w_max, T=10), per-checkpoint held-out acc ≥ 0.90 on every (k, T\*)
  cell (the single-track fold + k held accumulators didn't break the proven substrate). Convergence-gated;
  `diverged=false`. **This is the first thing to check — J-1 died here; R1's single-track claim is exactly
  the hypothesis that S1 now passes.**
- **W-bar — the WIDTH dial is real (make-or-break, R1a vs R1b).** At fixed full T=10, all-k-correct acc
  must RISE with state width w, and k=4 inputs must need more width than k=1: `acc(w_max) − acc(w_min)` ≥
  a pre-registered gap AND ρ(deployed-w, k) ≥ 0.7. If R1a (readout-mask) fails W-bar (a narrow read
  buffers k snapshots), adopt R1b (state-mask) — the faithful analog. If BOTH fail → width dial is a
  free-ride → fall to R3.
- **D-bar — the DEPTH dial is graded** (G-DEPTH S2 knee, reused): acc(x, T=T\*) ≥ 0.95·acc(x, full) per
  cell; acc(x, T\*−2) ≤ Bayes(g=2)+10pp; ρ(deployed-T, T\*) ≥ 0.7.
- **X-bar — the two dials are jointly routable** (MOD-4, unchanged): a distilled 2-D router (raw input →
  (w,T)) deployed hard-pick beats the val-selected best fixed (w,T) at strictly less mean compute (w·T),
  AND strictly beats BOTH marginal routers (width-only at fixed T=10; depth-only at fixed w=w_max) on the
  compute-matched basis, each by 2·SE.
- **Kill criterion:** S1 fails on 2 seeds → the single-track joint substrate itself is dead → STOP,
  post-mortem, escalate (do not invent a 4th). Neither R1a nor R1b passes W-bar → width dial is a
  free-ride → try R3 once, else STOP. X-bar shows no joint gain over marginals → the 2-D dial is two 1-D
  dials → honest rescope (Option 3).

---

## 7. Confound ledger (explicit, for the reviewer)

| # | Confound | Present in group-order dial (§2)? | Present in R1 (§4)? | Control / falsifier |
|---|---|---|---|---|
| C1 | width-demand ↔ solvability-driven depth-hunger (Barrington) | **YES, structural** (§3) — fatal | **NO** — group fixed at A5, depth-hunger constant across k | MOD-5[f]: assert A5 on every input |
| C2 | width-demand k ↔ commitment length T\* | (order↔length, §3) | NO by independent draw | MOD-5[a]: pearson(k,T\*)≈0, all 12 cells |
| C3 | width demand is a free readout, not a state-capacity demand | n/a | **the R1 risk** | MOD-5[d] capacity check + W-bar R1a→R1b fall-through |
| C4 | "buffer k snapshots" is serial memory, not width | n/a | possible | R1b state-mask: masking width < k·log₂60 provably blocks it |
| C5 | freeze-and-hold gate secretly adds depth-hunger (breaks k⊥T\*) | n/a | possible | MOD-5[e]: gate is O(1)/step, exactly L unrolls |

---

## 8. Decision required from the user (the fork) + non-goals

This spec resolves the group-order confound by **replacing** the design doc's J-3 sketch (variable group
order) with **R1 (single-track k-checkpoint readout on fixed A5)**. That is a design change and needs
sign-off before any build. Two decisions batched for the user:

1. **Direction (the J0 escalation):** author-J-3-spec (this, Option 1) vs fix-the-multi-track-substrate
   (Option 2) vs rescope/park G-JOINT (Option 3). This spec assumes Option 1.
2. **Within Option 1 — construction:** adopt **R1** (recommended) as J-3's width mechanism, with the
   R1a→R1b readout-vs-state hedge? Or keep pursuing the literal group-order dial despite §3 (not
   recommended — structurally confounded)?

**Superseded by the review (§9):** decision (2) is no longer "adopt R1 as drafted" but "adopt R1 *with
fixes 1–3*, including the fix-2 fork you must resolve." On GO: fold fixes 1–3 into §4/§5, write
`j3_group_probe.py`, run the REVISED MOD-5 (realized-entropy orthogonality surface, not scalar pearson) —
**all checks must pass** — then freeze the numeric bars (xhigh adjudicator, as for J-1) and only then
pilot. Nothing is built until the fixes land and the revised MOD-5 passes.

---

## 9. Adversarial design review (pre-GO, 2026-07-17) — VERDICT: SOUND-WITH-FIXES

An adversarial review of R1 before any build (attacking the five load-bearing claims). The single-track
substrate claim survives; two arguments are broken as written; three more risks are under-weighted. **Do
not authorize a build until fixes 1–3 are in the spec — the current MOD-5 gate would pass a construction
whose two dials are entangled and whose width demand is weaker and flatter-in-k than claimed.**

**What survives (no fatal hole).**
- *k is a genuine width demand, not a free readout / serial memory.* `RecurrentComposer` is a fixed-width
  RNN with NO external memory — the only channel from step p_c to step L is the w-dim state, so "report
  all k at the final step" forces coexistence, and cross-time carrying *is* state width (ledger C4
  collapses: they are the same thing here). **R1b (`state[:, w:]=0` inside the fold) is a valid falsifier.**
  Caveat: this establishes width *exists*, not its *magnitude/gradation in k* — "provably" (§4, §7 C4) is
  earned only empirically by R1b, not by argument.
- *R1 does not re-import J-1's exact failure.* R1's per-step input is 6 clean dims (`onehot₅ + is_checkpoint`),
  one trajectory — the 20-dim/15-NOOP-noise multiplexing that killed J-1 is absent. **S1 at k=1 should
  recover ≈ the depth toy's ≥0.90.**

**Hole 1 — k ⊥ T\* FAILS (the strongest finding).** Both dials are encoded as positions on the same
length-10 word. A checkpoint's label is `prefix(p_c)`, determined at read-depth p_c; all-k-correct needs
read-depth = **max_c p_c = p_k**, which grows with k by order statistics → **k drives the effective depth
demand.** MOD-5[a]'s `pearson(k, T*)` on drawn scalars cannot see this — the leak is through the
checkpoint positions p_c, which the spec never pins relative to T\*. Compounding: the label is the
checkpoints, not the final product, so the imported T\* ladder is **vestigial** (operative depth is p_k).
This is a subtler recurrence of J-1's shape: J-1 shared one *block*; R1 shares one *10-slot sequence*.

**Hole 2 — the info-floor `k·log₂60` is wrong, two ways.** (a) *Overstated:* the k checkpoint products
are NOT independent — consecutive ones differ by an increment word, and A5 over the certified involution
generators mixes SLOWLY (measured in review: increment entropy by gap length — gap 1: 2.00 bits; gap 2:
3.13; gap 4: 4.29; gap 7: 5.02 bits, still 0.49 TV from the 5.907-bit uniform). Realized k=4 joint
entropy ≈ **16.4 bits, not 23.6** (less if checkpoints cluster) → the k=3-vs-4 demand gap compresses and
**W-bar's ρ(deployed-w, k)≥0.7 risks failing on a width dial that is real but saturated.** (b) *Category
error:* for a continuous w-dim tanh state a linear readout has no hard bit-channel bound; ~64 units carry
~6 bits (`REC_STATE_WIDTH`), so "bits" ≠ "state units" by ~10×. **The make-or-break is empirical W-bar,
not an analytic floor** — demote MOD-5[d] accordingly.

**Under-weighted risks.** (i) The substrate is a **vanilla tanh recurrence, no gating** — "freeze-and-hold
k accumulators" needs a near-identity transport of up to 4 discrete A5 elements through ~9 tanh steps
(classic vanilla-RNN long-lag weakness); MOD-5[e] checks unroll *count*, not retention *trainability* —
the wrong check. (ii) **w_max/ladder never set relative to k_max**: the anchor is ~64 units per A5 element
(width-16 → 0.46), so k=4 plausibly needs w_max ≈ 256; an under-provisioned w_max reads as substrate
death when it is capacity-starvation. (iii) S1 kill criterion doesn't distinguish "dead at all k"
(substrate dead → STOP) from "dead only at k=4" (dial range truncated, not death).

**New confounds for the §7 ledger.** **C6** checkpoint-position ↔ depth (Hole 1): falsifier must measure
realized-demand orthogonality *including positions*, not scalar pearson. **C7** gap-dependent width
magnitude (Hole 2): the width demand's *size* depends on uncontrolled checkpoint spacing; the ledger never
asks "is the width demand monotone/graded in k?".

**Required fixes (patch, do not redesign):**
1. **Pin checkpoint-position sampling + re-do the orthogonality probe.** MOD-5[a] must measure the
   **realized joint entropy H(k-checkpoint tuple) as a surface over (k, T\*)** and require it monotone-
   increasing in k at each T\* and flat in T\* at each k. Scalar `pearson(k,T*)` is insufficient. Top gate.
2. **[PI DECISION] Resolve the T\* vs p_k ambiguity explicitly in §4:** either **(a)** declare the depth
   demand is p_k and drop the imported T\* ladder — then break the p_k↔k order-statistic coupling *by
   construction*; or **(b)** keep T\* as final-product commitment and make the final product a co-label,
   then show the checkpoint read-depths don't dominate. State which.
3. **Replace MOD-5[d]'s analytic floor** with the measured entropy ladder (fix 1) + an **enforced minimum
   inter-checkpoint gap** — the probe shows gap ≥ 5–6 is needed to approach independence, and a length-10
   word cannot give k=4 well-separated checkpoints, so **consider k ∈ {1,2,3}** with enforced gaps rather
   than {1,2,3,4}. Demote MOD-5[d] to an empirical capacity check; name W-bar the sole make-or-break.
4. **Set w_max and the width ladder from the empirical anchor** (~64 units/A5-element ⇒ w_max ≈ k_max×64)
   so S1/W-bar test the right range. (Folds into the Step-2 bar freeze.)
5. **Split the S1 kill criterion by k** ("all k" = substrate death → STOP; "high k only" = truncate the
   dial range) and add a retention-trainability check (exact k-element hold through the tanh recurrence)
   to MOD-5[e]. (Folds into the Step-2 bar freeze.)

Fixes 1–3 are load-bearing and gate the build; 4–5 fold into the bar freeze. R1's single-track substrate
is genuinely stronger than the dead J-1/J-2 and worth pursuing — but only after fixes 1–3 land.

**Non-goals (J-3):** no group-order ladder (§3 rejects it); no ambient group larger than A5 (MOD-1 wall);
no multi-track fold (killed J-1/J-2); no FlexNN/package changes (examples-level only); no MSE (MOD-2); no
concealment (MOD-3); no depth axis beyond L=10 (MOD-1).
