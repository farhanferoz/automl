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

**This suite is RICHER than width's** and the difference matters: these toys carry **per-input ground
truth** `k*(x)`, and three of them are **matched positive/negative pairs** — a bimodal toy and a
single-mode twin matched in mean and variance — so a selector that keys on spread rather than on
mixture structure is caught by construction. Width has no equivalent.

**Fixed on every cell of every tier:** seeds **0, 1, 2** · the strand's convergence gate with
`hit_cap: false` required · `OMP_NUM_THREADS=4` pinned · **selection on the per-sample fixed-σ
mixture log-likelihood** on the selection split, with **RMSE reported alongside as a point-accuracy
column, NEVER as the k readout**.

⚠️ **CORRECTED 2026-07-21 (root, at the user's instruction to audit this strand for variance and
regularisation problems). The previous text said "selection on the point-prediction metric (squared
error), never Gaussian NLL" — that was WRONG in both halves and it is the most dangerous kind of
wrong, because a battery driven to green on it would have produced a clean-looking false finding.**

- **The squared-error half inverted the ratified rule.** `docs/probreg/shared/benchmark_spec.md`
  §4.3 requires all three arms to score rungs on the fixed-σ mixture log-likelihood, and §4.2 keeps
  RMSE as a reporting column *"never the k readout"*. §4.3 states the consequence of getting this
  wrong in terms this strand cannot afford to ignore: *"Selecting k on squared error would be
  selecting on a quantity that is structurally blind to k for symmetric targets: the mixture mean is
  the same whether the model resolves one component or five. The selection curve would be flat, the
  cheapest-within-tolerance rule would return k=1 everywhere, and the result would look like a clean
  finding."* **⇒ Squared-error selection does not merely weaken this strand — it manufactures its
  headline conclusion out of a metric artefact.**
- **The "never Gaussian NLL" half over-banned, by conflating two different things.** What P8's
  reopening actually forbids is a likelihood read at a **LEARNED** `log_var` — that is variance
  fitting, and it is what the σ-scope decision removed. A likelihood at **FIXED** σ fits nothing: σ
  is one shared constant, so the metric is not a variance metric at all. **The test is not "is it a
  likelihood?" but "is σ learned?"**

**The rule, stated so it cannot be misread again:**

| | selection / k readout | reported alongside | status |
|---|---|---|---|
| fixed-σ mixture log-likelihood | ✅ **the readout, everywhere** | — | σ is a constant; nothing is fitted |
| RMSE / squared error | 🚫 **never** — structurally blind to k | ✅ honest point-accuracy column | a disagreement between the two is itself a finding |
| likelihood at a **learned** `log_var` | 🚫 **forbidden** | 🚫 | this is the violation that voided P8 |

**Tolerance rule is unaffected:** the curve stays in nats, so cheapest-within-tolerance at twice a
bootstrap standard error transfers **unchanged**, with its published figures still comparable
(spec §4.3).

- **TIER 1 — the reference cell.** `make_toy_b` (`automl_package/examples/_toy_datasets.py:49`) with
  `baseline="zero"`: "conditional Gaussian mixture with **KNOWN intrinsic k**", the mixture sitting
  still so the data are i.i.d. draws from one fixed mixture — the grounding regime where the pruning
  theory applies. **The only toy with unambiguous ground-truth k.** Every task that trains runs this.
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

**Resolution 2026-07-20 (user).** σ is not fitted, but scoring stays **distributional** with one
shared fixed σ (`docs/probreg/shared/benchmark_spec.md` §2, §4.1). The three existing toys are
homoscedastic by construction — exactly what a fixed σ wants — so **they are retained unchanged and
the multimodality question they were built for survives intact.**

**What actually happened, recorded because it is the instructive part.** The user's decision was
"do not fit σ". The orchestrator inferred "therefore score on squared error", and from that
correctly derived that the toys were degenerate: their two modes are equal-weight and symmetric
about zero (`automl_package/examples/_toy_datasets.py:174-178`), so the conditional mean is
identically zero and squared error has **zero power** to detect k. The derivation was sound; **the
premise was an unrequested inference.** The user challenged it, and the third option — keep the
mixture likelihood, fix σ to a shared constant — removes the in-sample-overfitting pathology
without removing the readout. **Lesson: when a scope decision seems to invalidate an established
artifact, re-examine the inference chain before discarding the artifact.** The cost here was one
spec; had it gone the other way it would have been the whole toy suite and the question with it.

**The spec is not wasted:** `docs/probreg/shared/toy_design_spec.md` designs toys where
resolution helps the *conditional mean*, with an analytic floor and a falsifiable
noise-versus-resolution prediction. That is a genuinely interesting and *different* question —
"classification as a regulariser for the mean" rather than "how many components does this input
need". **Parked for a future strand**, not deleted, and explicitly not a dependency of anything
below.
*Orchestration:* PARKED — no deps, no tier, not dispatchable, blocks nothing.


