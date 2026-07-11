# T1 path-1 — fresh-context certification (2026-07-10)

Fresh-context Opus adjudicator. The path-1 DRIVER (`capacity_ladder_t1_path1.py`) was newly
authored this session, so BOTH its methodology and its numbers were verified independently: every
load-bearing figure re-derived from disk (`path1_toyT1_seed0.pt` kept-best shard +
`t1_path1_summary.json` per-restart records) and from source
(`capacity_ladder_f2.py`, `capacity_ladder_t1.py`, `_capacity_ladder_toys.py`) — the summary's
framing was NOT taken on faith. This certifies a NEGATIVE/hardening result and holds it to the same
bar as a positive.

**Verdict:** path-1 is **SOUND** and its methodology has **no test leak into selection**. The
disambiguation rules **HARDENED_UNLEARNABLE** — tent⁵ region B is GD-unlearnable at every depth,
not single-restart-unlucky. The T1 lane is a **learnability-vs-representability** finding, **PARKED
at a user-level G-FORK** (path 2 vs path 3); **H2 stays LOCKED**.

---

## (a) Driver methodology audit — CONFIRMED, no test leak

**Science reused verbatim (no re-implementation).** The driver imports `_build_model`,
`_fixed_depth_log_likelihood`, `_jsonable` from `capacity_ladder_f2` and `HIDDEN_SIZE`,
`LEARNING_RATE`, `MAX_DEPTH`, `N_TEST`, `N_TRAIN`, `RESULTS_DIR`, `_construction_bar` from
`capacity_ladder_t1` (`capacity_ladder_t1_path1.py:59-70`). The only new code is the keep-best-by-
train restart loop (`run_multistart_depth`) and the verdict mapper (`_verdict`); no scoring, model
construction, or construction-bar logic is re-derived. Confirmed by reading all three sources.

**Selection uses ONLY the training set — the untouched test set is scored but never selected on
(the load-bearing methodological risk).** In `run_multistart_depth`
(`capacity_ladder_t1_path1.py:129-146`):
- `train_ll = float(_fixed_depth_log_likelihood(model, x_tr_t, y_tr_t).mean())` is the SELECTION
  metric — computed on `(x_tr, y_tr) = make_toy_t1(seed=0)`.
- `test_ll = _fixed_depth_log_likelihood(model, x_te_t, y_te_t)` on the untouched test
  (`make_toy_t1(seed=500)`, `capacity_ladder_t1_path1.py:218`) is stored but the keep-best gate is
  `if train_ll > kept_train_ll:` — test never enters selection.
- Direction is correct: `_fixed_depth_log_likelihood` returns a per-example Gaussian LL (higher =
  better; `capacity_ladder_f2.py:148-151,181-188`), so `train_ll >` keeps the **max train LL = best
  train loss**.

Three independent confirmations that the held-out bar-(i) region-B LL is NOT selection-contaminated:
1. **Known-answer selftest PASS** (`--selftest`, reproduced here). Part (a) plants restart 1 with the
   BEST train LL (−0.10) but a WORSE test LL (−0.20) than restart 0 (test +0.30); the selector picks
   restart 1 and keeps test-B = −0.200 (restart 1's), not +0.30 (the test-best) — proving selection
   reads train, not test. Parts (c1)/(c2) prove the verdict mapper is symmetric: a planted region-B
   climb reads RESCUABLE, a planted flat reads HARDENED_UNLEARNABLE — so HARDENED is not a hardcoded
   outcome.
2. **On the REAL run, at all 6 depths the kept-by-train restart is NOT the test-best restart**
   (kept-by-train region-B ≠ best-of-8-by-test region-B at every depth; re-derived from the
   per-restart records). If selection had peeked at test, these would coincide. They never do.
3. **The optimistic best-of-8-BY-TEST bound is reported separately and gates nothing** (test-peeking,
   diagnostic-only, `capacity_ladder_t1_path1.py:154,162,176`) — and it too fails at every depth, so
   the verdict does not depend on the selection metric at all.

**8 restarts genuinely differ.** Restart `r` builds the model via
`_build_model(FlexibleHiddenLayersNN, depth, LayerSelectionMethod.NONE, r, ...)`
(`capacity_ladder_t1_path1.py:131`); `_build_model`'s 4th positional arg is `seed`, passed to
`random_seed=seed` (`capacity_ladder_f2.py:112-138`). Distinct `r` → distinct `random_seed` → distinct
init. Restart 0 (`random_seed=0`) reproduces `capacity_ladder_t1`'s single-restart fixed-depth fit as
an anchor. The restart LLs are visibly non-degenerate (e.g. d4 r6 train LL −0.161, a bad-init outlier
vs the ~+0.28 cluster; d5 r3 +0.136) — the inits really vary.

**Region split, oracle, data provenance — all CONFIRMED independently:**
- Region A = `x < 0.5`, region B = `x ≥ 0.5` (`toy_t1_region`, `_capacity_ladder_toys.py:202-205`);
  in the driver `mask_a = region==0`, `mask_b = region==1`. Verified from the shard: `region==1 ⟺
  x≥0.5` exactly; n_A = 245, n_B = 255 (matches `construction.n_region_a/​n_region_b`).
- Oracle `−log σ − ½log 2π − ½` at σ=0.1 = **+0.8836465598**, recomputed independently and equal to
  the shard's stored `oracle_ll` to <1e-12 (`capacity_ladder_t1_path1.py:78`).
- Shard `x` matches `make_toy_t1(n=500, seed=500)` exactly; shard `region` matches
  `toy_t1_region` exactly — the untouched test set is the T1 held-out convention (seed+500).

---

## (b) Re-derived per-depth region-A / region-B table (from the `.pt` shard)

Oracle LL (σ=0.1) = **+0.88365**. Region-A/B means recomputed directly from the kept-best test-LL
vectors in `path1_toyT1_seed0.pt`; they match `t1_path1_summary.json` to **max |Δ| = 8.5e-8**
(float32 model-output storage). "kept" = the keep-best-by-**train** restart's untouched-test LL;
"best-by-test" = the optimistic (test-peeking, diagnostic) per-depth max over 8 restarts.

| depth | kept restart | region A LL | gap A | kept region B LL | gap B | best-by-test B LL | gap B (optimistic) |
|------:|-------------:|------------:|------:|-----------------:|------:|------------------:|-------------------:|
|    d1 |            5 |    +0.83125 | 0.052 |         −0.29795 | 1.182 |          −0.28722 |              1.171 |
|    d2 |            7 |    +0.84515 | 0.039 |         −0.28982 | 1.173 |          −0.27376 |              1.157 |
|    d3 |            2 |    +0.85149 | 0.032 |         −0.31549 | 1.199 |          −0.25760 |              1.141 |
|    d4 |            0 |    +0.84419 | 0.039 |         −0.28435 | 1.168 |          −0.27231 |              1.156 |
|    d5 |            6 |    +0.84875 | 0.035 |         −0.32488 | 1.209 |          −0.27276 |              1.156 |
|    d6 |            3 |    +0.84309 | 0.041 |         −0.29088 | 1.175 |          −0.25893 |              1.143 |

Construction bar on the kept-best sweep (re-derived from the shard, matches JSON exactly):
region B **d1→d2 mean Δ = +0.00813** (SE ~0.0123, not > 2·SE) and **d2→d3 mean Δ = −0.02567**
(SE ~0.0191, negative) → `region_b_pass = False`. Region A: no d1→d(k) gain beats 2·SE →
`region_a_flat = True`. Both reproduce `construction` in the summary.

---

## (c) Disambiguation ruling — HARDENED (GD-unlearnable, not restart-luck)

The FAIL_I adjudication set two outcomes: region B stays ~1 nat below oracle at all depths under 8
restarts → **hardens GD-unlearnable**; region B closes much of the gap at d≥3 → **bar-(i) rescuable**.

**Ruling: HARDENED, decisively.**
- **Kept-best-by-train region B is 1.168–1.209 nat below oracle at EVERY depth** (min gap 1.168 @ d4),
  with **no downward trend** — d3 (1.199) and d5 (1.209) are the WORST, not the best. A rescue would
  require the gap to close at d≥3; it does not close at any depth.
- **Even the optimistic best-of-8-BY-TEST region B is 1.141–1.171 nat below oracle at every depth**
  (min 1.141 @ d3) — so even cherry-picking, per depth, the single restart that best fits region B ON
  THE TEST SET, tent⁵ never comes within 1.14 nat of the oracle. This is > 0.70 nat at every depth →
  `HARDENED_UNLEARNABLE` (`capacity_ladder_t1_path1.py:183-188`).
- **Region A is learnable and flat**: reaches +0.831…+0.851 (0.032–0.052 nat below oracle) at every
  depth, flat across depth (`region_a_flat = True`). The A-vs-B contrast validates the harness inside
  the same run — a scoring/training bug would depress region A too; it does not. The failure is
  specific to tent⁵.

Multi-restart on the same Telgarsky toy does NOT rescue bar (i). The residual "maybe one unlucky
single init" question that FAIL_I left open is now closed: 8 independent inits per depth on seed 0,
selected honestly by train and also bounded by a test-peeking optimum, all land ~1.1–1.2 nat short at
every depth. tent⁵ is representable at d≥2 (d2 already exceeds 32 regions) yet **not SGD-learnable**;
its learnable linear region reaches the oracle. This is a genuine learnability-vs-representability
asymmetry, not an optimization fluke.

---

## (d) Ready-to-paste RESULTS.md section (T3 format, full T1 synthesis)

```markdown
## T1 — provably-deep-required toy (depth-lane discriminator). Verdict: learnability-vs-representability finding; bar-(i) FAILED 0/3 and HARDENED unlearnable under multi-restart; PARKED at a user G-FORK (path 2 vs path 3); H2 LOCKED (adjudicated 2026-07-10, fresh context)

**Headline.** T1 built a toy where the per-input depth requirement is PROVABLE by construction —
region A (x<0.5) linear, region B (x≥0.5) a 5-fold-composed tent map (Telgarsky, 32 linear pieces) at
trunk width 8, σ=0.1 (oracle LL +0.8836). The finding is a clean **learnability-vs-representability
asymmetry**: tent⁵ is representable at depth ≥ 2 yet sits ~1.1–1.2 nat below the σ=0.1 oracle at every
trained depth on all seeds, while the linear region reaches the oracle. It is therefore **not** the
large learnable per-input depth signal the depth lane needed as a positive control. Read of record:
**a provable (Telgarsky-style) depth requirement lives in the GD-unlearnable regime, so it cannot serve
as the depth-lane positive control**; whether ANY large learnable-and-depth-requiring target exists is
the open question the parked G-FORK decides.

- **Bar (ii) — PER-INPUT READ reframed within-plan (regime-contingent).** RESUME's "promote the
  decomposed per-region read" fix was REFUTED: the decomposed reader shares the pooled `pi_global`
  baseline and is equally blind (region-B decomposed t = −0.84, mu = −0.0014 on the planted
  flat-A/+0.8-nat-B table). Bar (ii)'s pooled A-vs-B contrast is a per-input *heterogeneity*
  statistic, not a test of region B's *absolute* requirement (bar (i) tests that directly); on T1's
  asymmetric flat-A/dominant-B design it is a **structural zero** only in the truly-indifferent-A
  regime, and CAN co-pass bar (i) in a hurt-by-depth-A regime (both bars fire on the conflicting
  positive control, t ≈ 72–76). Disposition R1: reframe the PREREGISTRATION bar-(ii) text +
  outcome-semantics (done); construction run allowed to proceed. `T1/BAR_II_ADJUDICATION.md`.
- **Bar (i) — CONSTRUCTION FAILED 0/3 (the depth requirement is not GD-learnable).** On the dedicated
  fixed-depth sweep, region B is stuck ~1.1 nat below oracle at EVERY depth 1–6 on ALL 3 seeds
  (deepest-net gap +1.16/+1.12/+1.07 nat), does NOT progressively climb with depth (total d1→d6 change
  +0.12/+0.02/+0.03 nat — the not-learnable signature, not slowly-getting-there), while region A
  reaches +0.79–0.84 ≈ oracle. `region_b_pass=false` 3/3 → construction bar 0/3. The A-vs-B contrast
  validates the harness inside the run (a scoring bug would depress A too). The plan's width-6/tent⁶
  remedy is REFUTED as wrong-direction (it makes an already-unlearnable target harder). Not
  epoch-limited (A converged; B flat-not-climbing). `T1/FAIL_I_ADJUDICATION.md`.
- **Path-1 — HARDENED "GD-unlearnable, not restart-luck" (the within-plan disambiguation).** R=8
  independent random-init restarts per fixed depth on seed 0, keep-best by TRAINING LL (untouched test
  scored, never selected on — no-test-leak confirmed: at all 6 depths the kept-by-train restart is NOT
  the test-best, and the driver's known-answer selftest verifies the wiring). Region B stays
  **1.168–1.209 nat below oracle at every depth** (min 1.168 @ d4), with NO closing at d≥3 (d3/d5 are
  the worst). Even the optimistic best-of-8-BY-TEST region B stays **1.141–1.171 nat below oracle at
  every depth** (min 1.141 @ d3) — > 0.70 nat under the most generous test-peeking read. Region A flat
  and learnable (+0.831…+0.851, ≤0.052 below oracle). Multi-restart on the same Telgarsky toy does not
  rescue bar (i); the single-init unluckiness hypothesis is closed. `T1/T1_PATH1_ADJUDICATION.md`.
- **Terminal path = a PARKED user-level G-FORK.** Both remaining moves revise the ratified toy premise
  ("provable-by-construction Telgarsky") → user decision. **Path 2** = a bounded (≤3 configs, same
  bars, same 3 seeds, width-8 pin) empirical search for a depth-preferring-AND-learnable target
  (tent⁴@w8 → tent³@w8 → tent⁴@w6); scientifically stronger IF it succeeds; stop after 3 configs.
  **Path 3** = reframe with no new toy: record the learnability-vs-representability asymmetry as the
  depth-lane finding, close the lane (per-input value becomes a compute-only story), leave H2 locked.
  Path-1 is the deciding evidence: the "pure optimization fix, no premise change" option is now closed,
  so path 2 (if run) is bounded and premise-revising, and path 3 is correct only after a bounded path-2
  search fails. Advisory ordering: path 1 → (path 2, bounded) → path 3-as-fallback; the go/no-go on
  path-2 compute is the user's research-priority call.
- **H2 — LOCKED.** Every H2-unlock branch requires PASS (i) first; bar (i) failed 0/3 → all unlock
  branches unreachable. H2 stays locked regardless of the terminal path (path 3 locks it explicitly;
  path 2 cannot unlock until a redesigned bar (i) passes). `T1/FAIL_I_ADJUDICATION.md` Q4.
- **Strictly probabilistic + leak-free.** Fixed-depth LL = per-example Gaussian log-likelihood
  (`_fixed_depth_log_likelihood`, F2's formula, reused verbatim); train = `make_toy_t1(seed=0)`, test =
  `make_toy_t1(seed=500)` (disjoint); keep-best selection reads train only; construction bar =
  per-region paired plain-bootstrap SE (`_construction_bar`, reused verbatim). No penalty/λ. Restart
  `r` → distinct `random_seed=r` init.

Artifacts: `capacity_ladder_results/T1/{PREREGISTRATION.md, FAIL_I_ADJUDICATION.md,
BAR_II_ADJUDICATION.md, t1_summary.json, nested_toyT1_seed{0,1,2}.pt, t1_path1_summary.json,
path1_toyT1_seed0.pt, T1_PATH1_ADJUDICATION.md}`, `capacity_ladder_t1.py`, `capacity_ladder_t1_path1.py`.
Adjudication: fresh-context Opus (not the producing session). Path-1 driver newly authored this
session — BOTH numbers and methodology verified: every per-depth region-A/B mean and gap re-derived
from `path1_toyT1_seed0.pt` (kept vectors match `t1_path1_summary.json` to 8.5e-8), keep-best-by-train
confirmed to select on train not test (kept ≠ test-best at all 6 depths; known-answer selftest green),
oracle/region-split/data-provenance recomputed independently. Ruling: HARDENED_UNLEARNABLE (kept region
B 1.168–1.209 nat below oracle at every depth; optimistic best-of-8-by-test 1.141–1.171 nat below at
every depth; no closing at d≥3). Binding condition: report T1 as a learnability-vs-representability
finding PARKED at the user G-FORK; never as a depth-lane positive control, and never as a machinery
failure (region A learnable proves the harness). H2 LOCKED.
```

---

## (e) What the T1 G-FORK now presents to the user

Path-1 is the deciding evidence, and it removes the only within-plan option: bar (i) is **not**
rescuable by a pure optimization fix (multi-restart) on the same Telgarsky toy. The two remaining
moves BOTH revise the ratified toy premise, which is exactly why T1 parks here as a user-level fork:

- **Path 2 — bounded search for a depth-preferring-AND-learnable target (premise change:
  "provably deep-required / Telgarsky" → "empirically depth-preferring but learnable").** Concrete
  ≤3-config ladder, same bars, same 3 seeds, width-8 pin unless noted: (a) tent⁴ @ width 8 (16 pieces —
  d1≤9 provably can't fit, d2–d3 can, far less GD-rugged than 32); (b) if still unlearnable → tent³ @
  width 8 (8 pieces; risk: d1@w8 may already fit → no requirement); (c) if tent³ shows no requirement →
  tent⁴ @ width 6 (last resort, reintroduces the width-6 optimization penalty). Stop after 3 configs;
  if none thread the needle, that *empirically* demonstrates the limit → fall through to path 3.
  Scientifically the stronger outcome **if it succeeds** (it would manufacture the depth-lane positive
  control and could unlock H2 via a redesigned, passing bar (i)).

- **Path 3 — reframe, no new toy.** Record the learnability-vs-representability asymmetry as the
  depth-lane finding, close the lane; the depth lane's per-input value (if any) is a compute story only;
  H2 stays locked. Reframe wording is drop-in ready (`FAIL_I_ADJUDICATION.md` Q2 path 3): the count lane
  succeeds because different k are all GD-learnable, whereas provable depth requirements live in the
  GD-unlearnable regime, so a large learnable per-input depth signal may be difficult or impossible to
  construct — which reframes the F2 depth-lane null as scarcity of learnable per-input depth structure,
  not a blind instrument.

**Advisory (not a unilateral ruling):** prefer **path 1 → (path 2, bounded ≤3) → path 3-as-fallback**.
Asserting "unconstructible" (path 3's central claim) without first searching for a counterexample would
be premature; path-1 has now proven the same-toy optimization rescue fails, so a bounded path-2 search
is the remaining way to either manufacture the positive control or *earn* the path-3 reframe. The
go/no-go on spending path-2 compute is the user's research-priority call. H2 remains LOCKED under either
path.

---

## Uncertainty (what remains unverifiable here)

- Path-1 is a **single-seed** (seed 0) disambiguation by design (FAIL_I Step A: "on seed 0 only"). It
  hardens the seed-0 single-restart result; seeds 1/2 already showed the same ~1.1-nat flat-across-depth
  failure under single-restart (FAIL_I), so the multi-restart hardening is consistent with, but not
  independently re-run on, seeds 1/2. This is the intended scope, not a gap.
- Whether a learnable-AND-depth-requiring target exists at width 8 (path 2's premise) is the open
  empirical question these runs cannot answer — the bounded ≤3-config path-2 ladder is the only test,
  and it is a user-gated decision.
- The best-by-test optimistic bound is a per-depth max over 8 restarts on the untouched test; it is a
  test-peeking UPPER bound on achievable region-B LL, deliberately generous. It gates nothing and is
  used only to make the HARDENED verdict robust to the selection metric — which it does (fails at every
  depth regardless).

## Recommendation

**Ship the hardening; then PARK for the user G-FORK.** (1) Certify path-1 SOUND and
HARDENED_UNLEARNABLE — methodology clean (no test leak), numbers reproduce from disk. (2) Fold the
ready-to-paste `## T1` section (above) into `RESULTS.md` (orchestrator-owned edit; not modified here).
(3) Present the path-2-vs-path-3 G-FORK to the user with path-1 as the deciding evidence, advisory
ordering path 1 → (path 2, bounded) → path 3-as-fallback. (4) H2 stays LOCKED. Other lanes unaffected.
