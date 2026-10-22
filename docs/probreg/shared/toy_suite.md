# ProbReg — the canonical toy suite

Moved verbatim from the former `docs/plans/capacity_programme/probreg.md` on 2026-10-21 (the toy-suite <!-- citecheck-ignore -->
section and task PT). The generators live in `automl_package/examples/_toy_datasets.py`. Index:
`docs/probreg/PLAN.md`.

<!-- from docs/plans/capacity_programme/probreg.md lines 138-232 -->
### THE CANONICAL TOY SUITE — the toys EXIST and are named; what was missing is the ASSIGNMENT

*(Written 2026-07-21. The first draft of this section said the strand "has no canonical toy suite" and
deferred it to the user. **That was wrong and the user corrected it: the toys had been run, they just
needed finding and noting.** They live in ONE module,
`automl_package/examples/_toy_datasets.py`, imported by ~35 drivers, each carrying its ROLE in its own
docstring. Roles below are quoted from those docstrings, not inferred.)*

Removed 2026-10-22 (mixture framing; docs/probreg/cleanup_plan.md).

**Fixed on every cell of every tier:** seeds **0, 1, 2** · the strand's convergence gate with
`hit_cap: false` required · `OMP_NUM_THREADS=4` pinned.

The fixed-σ readout rule, its correction note and its rule table: removed 2026-10-22 (mixture framing; docs/probreg/cleanup_plan.md).

- **TIER 1 — the reference cell.** `make_toy_b` (`automl_package/examples/_toy_datasets.py:49`) with
  `baseline="zero"`: "conditional Gaussian mixture with **KNOWN intrinsic k**", the mixture sitting
  still so the data are i.i.d. draws from one fixed mixture — the grounding regime where the pruning
  theory applies. Every task that trains runs this.
- **TIER 2 — the three negative controls.** Required by any task claiming **the dial works**. Each is
  a trap for a different wrong mechanism:
  - `make_toy_a` (`:35`) — "smooth unimodal with controlled homoscedastic noise", the "no intrinsic k"
    baseline. Honest answer: the smallest k. Catches a selector that raises k on noise.
  - `make_broad_unimodal` (`:97`) — "Single Gaussian matched in mean AND variance to the k_true=2
    bimodal toy", so a moment-matching model "cannot tell them apart; the genuine mixture objective
    can". Catches a selector reading moments instead of structure.
  - `make_toy_c_broad` (`:241`) — "Single-mode twin of make_toy_c, variance-matched at every input
    (over-chopping trap)"; `k*(x) = 1` everywhere while the spread widens with x. "A tiling model is
    tempted to raise its bucket count here as the spread grows; the honest held-out arbiter must give
    those buckets no credit."
- **TIER 3 — the per-input structure set.** Required by any task claiming **per-input** behaviour:
  - `make_toy_c` (`:211`) — "per-input mode structure"; all the x-dependence lives in the SHAPE, not
    the location; `k*(x) = 2` above a spacing threshold, monotone in x.
  - `make_toy_e` (`:276`) — "the x-confound breaker": spacing humps, so the count must rise and then
    **fall again while x increases monotonically** — "which a selector that merely tracks x (or the
    marginal variance, which also humps) cannot fake."
  - `make_toy_d` (`:439`) — "a staircase in component count — the count-beyond-binary / ceiling test";
    `k*(x) in {1, 2, 3}`. Tests that reading generalises past 1-vs-2 **and STOPS at 3 rather than
    tiling out to k_max**.
- **TIER 4 — the generality ladder.** Required only by the headline comparison and the report: the
  separation/sigma sweep over tiers 1-3 (the resolvability dial named in `make_toy_b`'s docstring),
  plus `make_toy_d_ndim` (`:512`) for the dimensionality check.

**Which task runs which tier — a MECHANICAL rule, not a judgement:**
- Every task that trains runs **tier 1**.
- **+ tier 2** if it claims the dial works, or reports a chosen k as correct.
- **+ tier 3** if it makes any **per-input** claim (the distilled router, the per-input arm).
- **+ tier 4** only for the headline comparison and the report.
- **P7** (schedule migration) = tiers 1 + 2 — it re-validates that middle-k rungs are no longer
  starved, which is a claim about the dial. **PB / PC** (data need, router architecture) = tiers 1 + 3,
  both per-input. **P3** (the cheap-vs-expensive comparison) = tiers 1 + 2 + 3. **P6** (report) = all
  four.
- **A deviating cell carries a written justification IN THE TASK**, and may not be tabulated beside a
  canonical one unless the deviation is named in the same table.
- **The four legacy `make_datasets()` toys** (`:10` — heteroscedastic, bimodal, piecewise, exponential)
  are the OLD set, from before the role-labelled toys existed. **They are not a tier.** Results on them
  stay citable as history, labelled as the legacy set; **no NEW cell may be produced on them.**
- **The `make_v_toy*` family** (`:332`, `:369`, `:394`) belongs to the PARKED variance programme. Not a
  tier here; do not run.


<!-- from docs/plans/capacity_programme/probreg.md lines 1340-1365 -->
### PT — ⏸ PARKED. Toys are RETAINED; the redesign answers a different question.

The three existing toys are retained unchanged. The 2026-07-20 fixed-σ resolution and its narrative: removed 2026-10-22 (mixture framing; docs/probreg/cleanup_plan.md).

**The spec is not wasted:** `docs/probreg/shared/toy_design_spec.md` designs toys where
resolution helps the *conditional mean*, with an analytic floor and a falsifiable
noise-versus-resolution prediction. That is a genuinely interesting and *different* question —
"classification as a regulariser for the mean" rather than "how many components does this input
need". **Parked for a future strand**, not deleted, and explicitly not a dependency of anything
below.
*Orchestration:* PARKED — no deps, no tier, not dispatchable, blocks nothing.


