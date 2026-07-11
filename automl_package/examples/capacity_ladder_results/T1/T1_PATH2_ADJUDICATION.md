# T1 path-2 — fresh-context certification (2026-07-11)

Fresh-context Opus adjudicator, independent of the producing session. Path-2 is the user-greenlit
**bounded search** (≤3 configs) for an EASIER target that is BOTH depth-requiring AND
gradient-learnable — the positive control the depth lane needs. Every load-bearing figure was
**re-derived from the summary JSONs directly** (per-seed / per-depth / per-restart records), not taken
from the stored `.verdict`; the oracle constant, the verdict rule, and the leak-free selection were
each recomputed from scratch. This certifies a NEGATIVE/hardening result and holds it to the same bar
as a positive.

**Verdict:** all three configs re-derive to **NOT_FOUND_UNLEARNABLE**, in exact agreement with the
stored verdicts (0/3 construction_pass each; min optimistic region-B gap 1.0950 / 0.9805 / 1.0977 nat,
all > 0.70). Selection is **leak-free** (kept restart = argmax TRAIN LL at every (seed, depth); the
optimistic test-peek bound gates nothing and also fails everywhere). The bounded search finds **no**
learnable depth-requiring positive control among the three configs → the **representable-but-not-
learnable asymmetry stands** as the depth-lane finding (path-3-as-fallback). **H2 stays LOCKED.**

Independently recomputed oracle: `−log σ − ½·log 2π − ½` at σ=0.1 = **+0.8836465598**, equal to the
stored `oracle_ll` (0.8836465597893728) to < 1e-12. Construction / verdict logic reused verbatim from
`capacity_ladder_t1._construction_bar` and `capacity_ladder_t1_path1` (`ORACLE_LL`, `SIGMA`,
`_UNLEARNABLE_GAP_NAT=0.7`, `N_RESTARTS=8`); the only path-2-new code is trunk-width parameterization
and the per-seed loop (`capacity_ladder_t1_path2.py`).

---

## (a) Independent re-derivation, per config

All numbers below were computed by the adjudicator from the JSONs (iterating `per_seed[].per_depth[]`
and `per_seed[].construction`), then compared to the stored `verdict` block. Configs share
`n_restarts=8`, `n_epochs=800`, `n_train=1600`, `n_test=500`, `seeds=[0,1,2]`, and the σ=0.1 oracle
+0.8836.

| config    | n_iter (pieces) | width | recomputed n_construction_pass | recomputed n_region_b_pass | recomputed min optimistic B-gap | recomputed min kept B-gap | min opt > 0.7 | my verdict | stored verdict | agree |
|-----------|----------------:|------:|-------------------------------:|---------------------------:|--------------------------------:|--------------------------:|:-------------:|------------|----------------|:-----:|
| tent4_w8  | 4 (16)          | 8     | 0 / 3                          | 0 / 3                      | **1.095015**                    | 1.104102                  | yes           | NOT_FOUND_UNLEARNABLE | NOT_FOUND_UNLEARNABLE | ✓ |
| tent3_w8  | 3 (8)           | 8     | 0 / 3                          | 0 / 3                      | **0.980475**                    | 0.980475                  | yes           | NOT_FOUND_UNLEARNABLE | NOT_FOUND_UNLEARNABLE | ✓ |
| tent4_w6  | 4 (16)          | 6     | 0 / 3                          | 0 / 3                      | **1.097677**                    | 1.097677                  | yes           | NOT_FOUND_UNLEARNABLE | NOT_FOUND_UNLEARNABLE | ✓ |

Every recomputed field matched the stored `verdict` block exactly (min gaps to < 1e-9;
`n_seeds_construction_pass`, `n_seeds_region_b_pass`, and the coded verdict all identical). The
recomputed min optimistic gaps (1.10 / 0.98 / 1.10) match the pre-registered expected values.

**Step 1 (min optimistic region-B gap) — CONFIRMED.** For each config I took the min over all 18
(seed × depth) entries of `ORACLE_LL − per_depth.best_by_test_region_b`. Because the minimum gap is
1.095 / 0.980 / 1.098, the optimistic (test-peeking) region-B LL is > 0.70 nat below oracle at *every
single* (seed, depth) — not just on average — so the "airtight even under the most generous read"
framing is exact.

**Step 2 (n_construction_pass) — CONFIRMED.** Counting `per_seed[].construction.construction_pass ==
True` gives 0 for all three configs (9 seeds total, 0 passes).

**Step 3 (coded verdict rule) — CONFIRMED.** The driver's `_config_verdict` computes `found =
n_full_pass >= 2` (→ FOUND); else `min_opt_gap > 0.7` → NOT_FOUND_UNLEARNABLE; else
NOT_FOUND_AMBIGUOUS. For all three: `n_full_pass = 0 < 2` and `min_opt_gap ∈ {1.095, 0.980, 1.098} >
0.7` → **NOT_FOUND_UNLEARNABLE**. The stored verdict follows the rule and matches my independent
application of it.

**Verdict-flip margin.** NOT_FOUND_UNLEARNABLE would flip to NOT_FOUND_AMBIGUOUS only if the min
optimistic gap fell to ≤ 0.70; the closest config (tent3_w8) sits at 0.98, a 0.28-nat margin. It would
flip to FOUND only with ≥ 2/3 seeds passing the full construction bar; the actual count is 0/3. The
verdict is not near either boundary.

**Region B never makes the required consecutive climb.** The construction bar requires region B to
improve held-out LL by > 2·SE at d1→d2 **AND** d2→d3 on the same seed. Recomputed per-seed region-B
increments:
- **tent4_w8**: no d1→d2 or d2→d3 increment beats 2·SE on any of the 3 seeds. region_b_pass = False 3/3.
- **tent3_w8** (the closest config): isolated single increments beat 2·SE on individual seeds (seed0
  d1→d2, seed1 d2→d3, seed2 d1→d2), but **never both increments on the same seed**, so the required
  progressive d1→d2→d3 climb never holds. region_b_pass = False 3/3. Its best region-B kept LL reaches
  −0.097 (vs the tent4 configs' ~−0.22), i.e. it is the least-unlearnable of the three, yet still ~0.98
  nat short of the +0.8836 oracle.
- **tent4_w6**: no increment beats 2·SE on any seed. region_b_pass = False 3/3.

**Region A is learnable and reaches ≈ oracle on every config (the harness-works contrast).** Kept
region-A LL spans (min gap ↔ max gap vs oracle): tent4_w8 0.024–0.121, tent3_w8 0.032–0.142, tent4_w6
0.031–0.102 nat below oracle. Region A essentially reaches the +0.8836 oracle at its best depth on
every config while region B is pinned ~1 nat below — a scoring/training bug would depress region A too;
it does not. The failure is specific to region B (the tent map), which is the signature of a
learnability limit, not a machinery failure.

---

## (b) Leak check — **PASS** (selection reads TRAIN only)

For all 3 configs × 3 seeds × 6 depths = 54 entries, I verified from the per-restart records:

1. **`kept_restart` = argmax over restarts of `train_mean_ll`** (mirroring the driver's strict-`>`,
   keep-first-max tie rule) — **0 violations** across all 54 entries. The kept restart is selected on
   the training LL, never the test LL.
2. **`kept_test_mean_ll_region_b` equals the kept restart's own `test_mean_ll_region_b`** — 0
   mismatches (the stored kept region-B is that restart's untouched-test score, not a cherry-pick).
3. **`best_by_test_region_b` equals `max` over restarts of `test_mean_ll_region_b`** (the optimistic,
   test-peeking, diagnostic-only bound) — 0 mismatches.
4. **`best_by_test_region_b ≥ kept_test_mean_ll_region_b`** at every entry — 0 violations (the
   test-peek optimum is ≥ the train-selected value, as required).

**Discriminating evidence that selection did not secretly peek at test** (the same test path-1 used):
the train-selected kept restart is genuinely NOT the test-best restart in the large majority of
entries — kept ≠ argmax(test region-B) on 16/18, 11/18, 11/18 entries (tent4_w8 / tent3_w8 /
tent4_w6); kept ≠ argmax(test total LL) on 9/18, 7/18, 9/18. If selection had read test, these would
coincide; they mostly do not. And because the optimistic test-peek bound *also* fails at every depth
(min gap > 0.70), the verdict does not depend on the selection metric at all.

No leak. The held-out region-B bar is not selection-contaminated.

---

## (c) Certified verdicts

| config    | certified verdict         | basis |
|-----------|---------------------------|-------|
| tent4_w8  | **NOT_FOUND_UNLEARNABLE** | 0/3 construction_pass; min optimistic B-gap 1.0950 > 0.70 at every depth |
| tent3_w8  | **NOT_FOUND_UNLEARNABLE** | 0/3 construction_pass; min optimistic B-gap 0.9805 > 0.70 at every depth (closest config; isolated single B increments but never the consecutive d1→d2 AND d2→d3 climb) |
| tent4_w6  | **NOT_FOUND_UNLEARNABLE** | 0/3 construction_pass; min optimistic B-gap 1.0977 > 0.70 at every depth |

**Overall: NOT_FOUND (flavor NOT_FOUND_UNLEARNABLE) on all 3/3 configs.** The bounded ≤3-config
search is exhausted with no learnable depth-requiring positive control. My independent recompute
**AGREES with every stored verdict** (verdict string, n_construction_pass, n_region_b_pass, min
optimistic gap, min kept gap) with zero discrepancies.

---

## (d) region_a_flat note (does NOT change the verdict)

As flagged, `region_a_flat` is False on some seeds: **tent3_w8** seeds 1 and 2, and **tent4_w6** seed
1 (all other seed/config combinations have region_a_flat = True). On those seeds region A showed a
> 2·SE LL change at some depth past d1, so region A was not perfectly flat.

This does **not** affect the NOT_FOUND outcome, and here is the one-line reason:
`construction_pass = region_b_pass AND region_a_flat`, and **`region_b_pass = False` on ALL 9
seeds across all 3 configs** — because region B never improves by > 2·SE at both d1→d2 and d2→d3 on
any single seed. A False on the `region_b_pass` conjunct forces `construction_pass = False`
regardless of `region_a_flat`, so the region-A-flatness wrinkle cannot flip any seed to a pass. The
bar fails on region B irrespective of region A.

---

## (e) Disposition

**Does the bounded search find a learnable depth-requiring positive control? NO — on all three
configs.** Reducing the tent-fold count from 5 (path-1) to 4 or 3, and narrowing the trunk from width
8 to width 6, does not produce a target whose region-B depth requirement is both *realized* (climbs
d1→d2→d3 to the oracle) and *learnable* by gradient descent. Even best-of-8 restarts with the most
generous test-peeking read leave region B ≥ 0.98 nat below the σ=0.1 oracle at every depth on every
seed, while the linear region A reaches the oracle — the harness works, so the shortfall is a learning
limit, not an instrument failure. This is the **learnability-vs-representability asymmetry**: a
provable (Telgarsky-style composed-tent) depth requirement lives in the GD-unlearnable regime across
the bounded set, so it cannot serve as the depth-lane positive control. With path-2 exhausted, that
asymmetry **stands as the depth-lane finding** (path-3-as-fallback); H2 remains LOCKED (every H2-unlock
branch requires a passing construction bar first, and 0/3 pass). What remains genuinely open is whether
**any** learnable-and-depth-requiring target exists at all — a bounded 3-config search demonstrates the
limit empirically but cannot settle non-existence; that question is not closed by these runs and should
not be framed as if it were.

---

## Uncertainty (what remains unverifiable here)

- **Verdict-only certification, no re-run.** Per the task, no training was reproduced; the
  certification is that the stored per-restart/per-depth numbers, when the verdict logic is applied
  independently, yield NOT_FOUND_UNLEARNABLE and the selection is leak-free. Whether the underlying fits
  themselves are optimal is inherited from the path-1-certified apparatus (same `_build_model`,
  `_fixed_depth_log_likelihood`, `_construction_bar`, reused verbatim), not re-audited here. The
  isolated bad-init restarts visible in the records (e.g. tent4_w8 seed0 d4 restart6 train LL −0.157;
  tent4_w6 seed2 d4 restart6 train LL −0.151) are correctly excluded by keep-best-by-train and do not
  affect the kept sweep.
- **Bounded search ≠ impossibility proof.** Three configs cannot establish that no learnable
  depth-requiring target exists; the finding is "not found in the bounded set," which *earns* the
  path-3 reframe without proving unconstructibility. This is the intended scope of the user-greenlit
  bounded search, not a gap.
- **tent3_w8 is the near-miss.** Its 0.98-nat optimistic gap is the smallest of the three and it shows
  isolated single-step region-B gains — but never the required consecutive d1→d2 AND d2→d3 climb on any
  seed, so it is still a clean 0/3 construction fail, comfortably inside NOT_FOUND_UNLEARNABLE (0.28-nat
  margin from the AMBIGUOUS boundary).

## Recommendation

**Ship the certification.** (1) Certify all three path-2 configs SOUND and NOT_FOUND_UNLEARNABLE —
numbers reproduce from the JSONs to < 1e-9, selection is leak-free (kept = argmax train at all 54
entries; optimistic test bound gates nothing and also fails everywhere). (2) The bounded ≤3-config
path-2 search is **exhausted with no positive control found**; fold the path-3-as-fallback outcome into
`RESULTS.md` (orchestrator-owned edit): record the **learnability-vs-representability asymmetry as the
depth-lane finding**, close the lane (per-input depth value becomes a compute-only story), and keep H2
**LOCKED**. (3) Frame strictly as a learning limit, never as a machinery failure — region A reaching
the oracle on every config is the in-run proof the harness is sound. (4) State the open question
honestly: whether any learnable depth-requiring target exists is not settled by a 3-config search.
Binding condition on any downstream write-up: no penalty/λ language, held-out Gaussian-LL bar only, and
the H2 hypothesis stays locked.
