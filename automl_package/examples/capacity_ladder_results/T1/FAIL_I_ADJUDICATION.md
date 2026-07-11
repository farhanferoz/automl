# T1 bar-(i) FAILED — fresh-context adjudication (2026-07-10)

Fresh-context Opus adjudicator. Reproduced every load-bearing claim from disk
(`t1_summary.json`, `nested_toyT1_seed{0,1,2}.pt`) and from source
(`_capacity_ladder_toys.py`, `EXECUTION_PLAN.md`) — did **not** take
`FAIL_I_DIAGNOSIS.md` on faith. Verdict: the diagnosis is **CONFIRMED**, the plan's
width-6/tent⁶ remedy is **REFUTED as wrong-direction**, and the terminal path choice is a
**USER-LEVEL G-FORK** gated behind one cheap within-plan disambiguating test.

---

## Q1 — VERIFY the diagnosis: CONFIRMED (from disk)

Per-depth, per-region mean held-out log-likelihood (recomputed from `fixed_depth_ll` in each
`.pt`; region A = `x<0.5`, region B = `x≥0.5`; higher = better). Oracle LL for σ=0.1 is
`−log σ − ½log 2π − ½ = +0.8836`.

| seed | region | d1 | d2 | d3 | d4 | d5 | d6 |
|---|---|---|---|---|---|---|---|
| 0 | A | +0.362 | +0.787 | +0.845 | +0.844 | +0.828 | +0.836 |
| 0 | B | −0.395 | −0.288 | −0.289 | −0.284 | −0.316 | −0.276 |
| 1 | A | +0.747 | +0.844 | +0.806 | +0.785 | +0.821 | +0.844 |
| 1 | B | −0.252 | −0.237 | −0.245 | −0.230 | −0.233 | −0.235 |
| 2 | A | +0.800 | +0.801 | +0.795 | +0.806 | +0.787 | +0.819 |
| 2 | B | −0.210 | −0.205 | −0.198 | −0.188 | −0.192 | −0.184 |

These match `FAIL_I_DIAGNOSIS.md` exactly. Confirmed sub-claims:

1. **Region B is stuck ~1.1 nat below oracle at EVERY depth on ALL seeds.** Gap at the
   deepest net (d6): +1.160 (s0), +1.119 (s1), +1.068 (s2) nat; implied predictive std ≈
   0.29–0.32 (the net covers the 32-piece oscillation with wide variance instead of fitting
   it). **No depth 1–6 learns tent⁵.**
2. **Region B does not progressively climb with depth** (total d1→d6 change: +0.119 s0, +0.017
   s1, +0.026 s2 — all tiny vs the 1.1-nat gap). This is the signature of *not-learnable*, not
   *slowly-getting-there*: the deepest nets, which CAN represent tent⁵, do no better than d2.
3. **Region A is learnable** — reaches +0.79–0.84 ≈ oracle at good depths. Seed-0's d1=+0.362
   is depth-1 underfitting region A in that one seed (capacity contended with region B at d1),
   an optimization artifact, not a real "depth helps region A" signal.
4. **Harness is validated inside the same run** by the A-vs-B contrast: region A near-oracle
   proves the fixed-depth sweep, the LL scoring (σ, the formula), and the training loop all
   work. A scoring bug would depress region A too. The failure is **specific to tent⁵**.
5. `t1_summary.json` `region_b_gains` reproduce (s0 d1→d2 +0.107 sig, d2→d3 −0.001 flat; s1/s2
   both increments non-significant) → `region_b_pass=false` on 3/3 → `construction_bar 0/3`.

**Not epoch-limited — CONFIRMED, with one caveat.** The note's specific "+0.41 at 100ep"
measure-one datum is **not on disk** (no 100-epoch artifact in `T1/`), so that exact number is
*insufficient-evidence* here. But the conclusion it supports is independently corroborated by
the 800ep data on disk: region A converged to oracle (training is not globally under-run) and
region B is **flat-not-climbing** across depth. Under-training would show region B monotonically
approaching oracle at deeper/longer; it does not. The residual "maybe just an unlucky single
optimization run" question is what Q2 path 1 tests — cleaner than an epoch sweep.

**Representability sanity:** d1 width-8 ReLU on 1D gives ≤ 9 linear pieces < 32 → d1 provably
cannot represent tent⁵ (prereg claim holds). Minor: d2 (not only d3) can already exceed 32
regions, so the prereg's "depth ≥ 3" threshold is loose — but this is moot, since learning
fails at *every* depth.

**Verdict Q1: CONFIRMED.** tent⁵ is unlearnable by GD at every depth on every seed; region A is
learnable; the failure is a genuine learning failure, not a harness or scoring artifact. This is
a **learnability-vs-representability** failure (tent⁵ is representable at d≥2 but not SGD-
learnable), NOT a "signal-too-small" failure.

---

## Q2 — RULE on the path

**The plan's width-6/tent⁶ redesign is REFUTED (wrong direction).** `EXECUTION_PLAN.md`
lines 211-212 and the prereg §Outcome both prescribe "first try width 6 then a deeper tent⁶."
Both moves make an already-unlearnable target *harder*: width 6 = less capacity / worse
optimization; tent⁶ = 64 pieces = strictly ruggeder GD landscape than tent⁵. The plan's remedy
was written on the assumption "region B doesn't need *enough* depth" (fix: harder target). The
diagnosed-and-verified failure is "region B unlearnable at *any* depth" (fix: an *easier*
target, or a different mechanism). **Do NOT run width-6/tent⁶.**

**Chosen path: a two-step disposition.**

**Step A (run now, within-plan): path 1 — one-seed multi-restart disambiguation.** Train
R=8 independent random-init restarts per depth on seed 0; **keep-best by TRAINING/validation
loss, not test** (else the held-out bar-(i) LL is contaminated by selection-on-test); then score
the untouched test set's per-depth LL. This is cheap and decision-relevant:
- If region B stays ~1 nat below oracle at all depths (the note's prediction) → hardens
  "GD-unlearnable, not restart-luck," strengthening whatever terminal path the user picks.
- If region B unexpectedly closes much of the gap at d≥3 → bar (i) may be rescuable on the
  *same* Telgarsky toy via multi-restart, a pure optimization fix with **no premise change**
  (the best outcome — keeps the provable toy). That would overturn the diagnosis and is worth
  knowing before forking.

Path 1 changes no premise and is low-risk/diagnostic → the orchestrator runs it under the plan.

**Step B (after path 1, user-gated): terminal choice between path 2 and path 3.** Both revise
the ratified toy premise and decide the depth lane's scientific conclusion → **G-FORK** (see Q3).
Present both to the user with the path-1 result in hand:

- **Path 2 — pivot to an empirically-depth-preferring-AND-learnable target** (premise change:
  "provably deep-required / Telgarsky" → "empirically depth-preferring but learnable"). Concrete
  bounded spec (≤3 configs, same bars, same 3 seeds, width-8 pin unless noted): (a) **tent⁴ at
  width 8** — 16 pieces, d1 (≤9) provably can't fit but d2-d3 can, and 16 pieces is far less
  GD-rugged than 32; (b) if tent⁴ still unlearnable → **tent³ at width 8** (8 pieces; accept a
  smaller-but-real requirement, risk: d1 width-8 may already fit it → no requirement); (c) if
  tent³ shows no requirement → **tent⁴ at width 6** (fewer d1 pieces forces depth; reintroduces
  the width-6 optimization penalty, so last resort). Bar (i) unchanged: region B d1→d2 AND d2→d3
  both > 2·SE, region A flat, ≥ 2/3 seeds. **Stop after 3 configs**; if none thread the needle,
  that empirically *demonstrates* the fundamental limit → fall through to path 3.
- **Path 3 — reframe, no new toy.** Record the learnability-vs-representability asymmetry as the
  depth-lane finding; close the lane; H2 stays locked; the depth lane's per-input value (if any)
  is a compute story only. Reframe wording (drop-in for the report):
  > *"The count lane succeeds because different k are all GD-learnable, so a per-input count
  > requirement is both large and learnable → detectable. Depth requirements that are provable
  > (Telgarsky-style compositions) live exactly in the GD-unlearnable regime: T1's tent⁵ is
  > representable at depth ≥ 2 yet sits ~1.1 nat below the σ=0.1 oracle at every trained depth on
  > all seeds, while its learnable linear region reaches the oracle. A large learnable per-input
  > depth requirement may therefore be difficult or impossible to construct — which reframes the
  > F2 depth-lane null: the null may reflect the scarcity of large learnable per-input depth
  > structure, not a blind instrument. The depth machinery is neither validated nor refuted on a
  > per-input depth signal, because no positive control could be manufactured."*

**Recommendation to the user (advisory, not a unilateral ruling):** if the path-1 test confirms
GD-unlearnability, path 2 is a *bounded* (≤3 configs) empirical attempt to still manufacture the
positive control the depth lane wanted; it is the scientifically stronger outcome **if it
succeeds**. Path 3 is correct **only after** a bounded path-2 search fails — asserting
"unconstructible" (path 3's central claim) without having searched for a counterexample would be
premature. So: prefer **path 1 → (path 2, bounded) → path 3-as-fallback**, but the go/no-go on
spending path-2 compute is a research-priority call for the user.

**Verdict Q2:** plan's width-6/tent⁶ = **REFUTED**. Run path 1 now (within-plan). Terminal
path-2-vs-3 = user fork (Q3).

---

## Q3 — GOVERNANCE: hybrid (within-plan corrections now; terminal choice is a G-FORK)

- **Within-plan (adjudicator applies now):** (1) skip the width-6/tent⁶ remedy — it is
  demonstrably wrong-direction, and the plan already routes FAIL(i) to the adjudicator; reaching
  the adjudicator without running a known-counterproductive step is a sound reading of the plan's
  intent ("try a fix, then escalate" — the fix is known-bad, so escalate). (2) Run the path-1
  multi-restart diagnostic — it changes no premise and is low-risk.
- **G-FORK for the user (T1 parks here):** the terminal path-2-vs-3 choice. It (a) revises the
  **ratified toy premise** — the prereg §Toy defines T1 as "provable by construction (a Telgarsky-
  style composed tent map)"; both terminal paths abandon or reframe that (path 2 → "empirically
  depth-preferring but learnable"; path 3 → "a large learnable depth requirement may be
  unconstructible"); and (b) fixes the **depth lane's scientific conclusion** and the reframing of
  the F2 null — a research-deliverable-level decision the user ratified the plan to answer. This
  also exceeds the plan's own redesign-iteration scope (which was same-premise: "width unchanged —
  width 6, tent⁶"). Consistent with the prereg precedent that even the compute-payoff H2 decision
  was declared a G-FORK, this larger premise revision is a fortiori a G-FORK.

**Verdict Q3: HYBRID.** Within-plan: skip width-6/tent⁶, run path 1. Then **PARK T1** for a
user-level G-FORK on the terminal path. P1 and all other lanes continue unaffected.

---

## Q4 — H2 gating: CONFIRMED LOCKED

Every H2-UNLOCK branch in the prereg outcome semantics requires **PASS (i)** first (including the
compute-payoff structural-zero branch, itself a G-FORK). Bar (i) failed 0/3 → all unlock branches
are unreachable. H2 stays **LOCKED** regardless of the terminal path chosen (path 3 prescribes
locked explicitly; path 2 cannot unlock until a redesigned bar (i) passes). Bar (ii) is **not
reached** — the readers cannot test a per-input depth signal that no depth learned; the prior
`BAR_II_ADJUDICATION.md` is not re-litigated here and is moot for this outcome.

**Verdict Q4: CONFIRMED — H2 locked.**

---

## Uncertainty (what remains unverified)

- The exact "+0.41 at 100 epochs" measure-one datum is not on disk; the *conclusion* it supports
  (not epoch-limited) is confirmed by the 800ep region-A-converged / region-B-flat evidence, so
  this gap is not load-bearing.
- "GD-unlearnable *in principle*" vs "unlearned by this single-restart config" is not fully
  settled by these runs — this is precisely what path 1 (step A) tests before the fork.
- Whether a learnable-AND-depth-requiring target exists at width 8 (path 2's premise) is an open
  empirical question these runs cannot answer; the bounded ≤3-config ladder is the test.

## Recommendation

**Needs-more-discovery, gated.** (1) Adjudicator applies the within-plan corrections: do NOT run
width-6/tent⁶; run the one-seed R=8 keep-best-by-train multi-restart on seed 0 (untouched test LL).
(2) PARK T1 and G-FORK the terminal path-2 (bounded ≤3-config redesign, spec above) vs path-3
(reframe, wording above) decision to the user, with the path-1 result attached. (3) H2 stays
locked. Other lanes proceed.
