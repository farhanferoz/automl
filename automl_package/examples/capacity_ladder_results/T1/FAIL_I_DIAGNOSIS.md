# T1 bar-(i) FAILED (0/3) — orchestrator diagnosis for fresh adjudication (2026-07-10)

The full T1 800-epoch battery ran (main thread, `t1_summary.json` on disk, ~32 min). **OUTCOME:
FAIL_i** (`construction_bar: n_seeds_pass=0/3`). Before following the plan's FAIL(i) redesign path,
the orchestrator diagnosed the failure mode. The plan's prescribed remedy appears COUNTERPRODUCTIVE.
Fresh-context adjudication needed on the PATH (this challenges the ratified T1 toy premise).

## What failed (per-seed, per-region)
`region_b_pass` needs held-out-LL gain > 2·SE at d1→d2 AND d2→d3; `region_a_flat` needs no > 2·SE
gain past d1. Result:
| seed | region_b_pass | region_a_flat | construction_pass |
|---|---|---|---|
| 0 | False (d1→d2 +0.107 sig; d2→d3 −0.001 flat) | False (A gains +0.42–0.48, all sig) | False |
| 1 | False (d1→d2 +0.016; d2→d3 −0.009, neither sig) | False (A +0.10/+0.07/+0.10 at d2/d5/d6 sig) | False |
| 2 | False (d1→d2 +0.005; d2→d3 +0.007, neither sig) | True (A gains ≈0) | False |

Both regions misbehave, seed-dependently. Region B (tent⁵, the *point* of the toy) shows a robust
depth requirement on ZERO seeds.

## Root cause — tent⁵ is NOT learnable by GD at any depth (LL trajectory)
Per-depth, per-region MEAN held-out LOG-LIKELIHOOD (higher = better; `fixed_depth_ll` in the `.pt`):

seed 0  A: d1 +0.362 → d2 +0.787 → d3 +0.845 … d6 +0.836   B: d1 −0.395 → d2 −0.288 → … d6 −0.276
seed 1  A: d1 +0.747 → d2 +0.844 → … d6 +0.844             B: d1 −0.252 → … d6 −0.235
seed 2  A: d1 +0.800 → d2 +0.800 → … d6 +0.819             B: d1 −0.210 → … d6 −0.184

**Oracle LL for the planted noise σ=0.1 is +0.884/point** (a perfect fit leaves only the ε~N(0,0.1²)
noise: `−log σ − ½log 2π − ½`). Then:
- **Region A** reaches +0.80–0.84 at good depths ≈ the +0.88 oracle → **learnable, well-fit** (linear
  is trivial). Seed-0's d1=+0.36 is the depth-1 net failing region A in that seed only (secondary
  optimization noise; capacity fought over with region B at d1).
- **Region B sits at −0.2 to −0.4 at EVERY depth including d6** — a **~+1.2 nat gap** below oracle,
  i.e. effective predictive std ≈ 0.33 (the net covers the 32-piece tent oscillation with wide
  variance instead of fitting it). **No depth (1–6, width 8) learns tent⁵.**

Mechanism (validated by the A-vs-B contrast): Telgarsky depth separation is a **representation**
result — a deep net *can represent* tent⁵ — but iterated tent/sawtooth maps are the canonical family
that **gradient descent from random init cannot LEARN** (rugged loss landscape). So the toy cannot
exhibit a depth *requirement* via a learned fixed-depth sweep: the requirement needs some depth to fit
it and shallower depths to fail, but here NO depth fits it. Region A being learnable (≈ oracle at all
good depths) confirms the failure is specific to tent⁵, not the harness.

## Why the plan's FAIL(i) remedy is the wrong direction
Plan §2/PREREG FAIL(i): "redesign once (width unchanged — first width 6, then a deeper tent⁶ — before
touching anything else)." Both moves make the target HARDER to learn: width 6 is narrower (less
capacity, worse optimization); tent⁶ (64 pieces) is a strictly harder GD target than tent⁵. The plan
assumed the failure would be "region B doesn't need ENOUGH depth" (remedy: harder target). The actual
failure is "region B is unlearnable at ANY depth" (remedy: an EASIER-to-learn target, or a different
mechanism). Following the plan's path would deepen the failure.

## The deeper finding (for the report, if the adjudicator agrees)
The count lane succeeds because different k are ALL learnable (Gaussian mixtures fit easily), so a
per-input count REQUIREMENT is both large and learnable → detectable. Depth "requirements" that are
*provable* (Telgarsky compositions) live exactly in the GD-UNLEARNABLE regime. This asymmetry —
learnable count vs unlearnable-when-large depth — may be a FUNDAMENTAL limit on demonstrating a
per-input DEPTH requirement, not a fixable toy bug. It would reframe the F2 depth-lane null: the null
may reflect that large learnable per-input depth requirements are hard to construct at all.

## Candidate paths (for the adjudicator to rule among)
1. **Multi-restart optimization** (x4b-style R=8 keep-best) on the SAME toy: might rescue some region-B
   fits the single-restart missed. Prediction: partial help at best — a +1.2-nat gap to a needle-in-
   haystack compositional solution is unlikely to close via random restarts. Cheap-ish to test (one
   seed).
2. **An easier depth-requiring-AND-learnable target**: replace tent⁵ with a construction that (a) GD
   can learn and (b) still shows a real held-out depth gain (e.g., a low-order composition tent²/tent³
   that is depth-separated yet learnable, or a piecewise target matched to width-8 capacity). Departs
   from "provably deep-required" toward "empirically depth-preferring but learnable."
3. **Reframe (no new toy):** accept that a large *learnable* per-input depth requirement may not be
   constructible; record the learnability-vs-representability asymmetry as the depth-lane finding;
   H2 stays locked; the depth lane's per-input value (if any) is the compute story only.
4. Bump epochs / lr: RULED OUT — seed-0 region-A gain is +0.41 at 100ep (measure-one) and +0.4257 at
   800ep, i.e. unchanged by 8× epochs; region B flat across depth at 800ep. Not epoch-limited.

## Questions for the adjudicator
1. VERIFY the diagnosis: region B LL ≈ −0.3 (≈ +1.2 nat below the +0.88 oracle) at ALL depths on all
   seeds ⇒ tent⁵ unlearnable at every depth; region A learnable (≈ oracle). Reproduce from
   `T1/t1_summary.json` / `T1/nested_toyT1_seed*.pt`. Overrule if wrong.
2. RULE on the path (1/2/3, or another). Is the plan's width-6/tent⁶ redesign to be followed as-written
   (I argue no — wrong direction), or superseded?
3. GOVERNANCE: within-plan correction the orchestrator applies, or a USER-LEVEL G-FORK? This
   challenges the ratified toy premise ("provably-deep-required" conflated representation with
   learnability) and the ratified FAIL(i) procedure. The user is away → if G-FORK, PARK T1, keep other
   lanes (P1 etc.) moving.
4. H2 gating: H2 was conditional on T1 passing BOTH bars. Bar (i) failed ⇒ H2 stays locked regardless;
   confirm.

## Deliverable
Write `T1/FAIL_I_ADJUDICATION.md`: verdict on each question + the chosen path (with a spec if a
redesign, or the reframe wording if path 3), and within-plan vs G-FORK. This file is the deliverable.
Reproduce from disk; do not take this note on faith.
