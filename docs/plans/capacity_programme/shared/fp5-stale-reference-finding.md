# The W6 router-capacity deploy reference no longer reproduces at `master`

**Status: ✅ RESOLVED 2026-07-23 — diagnosis executed at the user's review by git archaeology
(zero experiment runs); full resolution recorded in `flexnn-package.md` FP-5's resolution block.
Short form: the reference was regenerated with provenance at `1d940a3` and now matches modern runs
bit-for-bit; zero commits touch the deploy code path in either drift window, so no committed code
moved the metric and the recommended bisect is moot; remaining cause candidates are environmental
(pre-provenance reference, unrecorded env/tree). Everything below stands as the historical record.**

**Original status: OPEN FINDING, batched for user review. NOT caused by wave 4. NOT resolved by the run.**
Recorded by the root, 2026-07-21, while verifying `flexnn-package.md` FP-5 clause (d).

## What failed

FP-5's verify (d) reproduces the two W6 router-capacity **deploy-claim** arms and requires, per
`flexnn-package.md` FP-5.c: **≤ 2% relative error on `deploy_bar.mse_hardpick` per seed**, and
**`deploy_bar.mean_executed_width` within ±0.25 absolute**. Both bars were fixed in the plan in
advance, precisely so a worker could not choose its own after seeing the result.

Result — the **`--router-hidden-mult 2.0` (rhx2)** arm misses on 2 of 3 seeds:

| seed | reference `mse_hardpick` | re-run | rel. err | bar | verdict |
|---|---|---|---|---|---|
| 0 | 0.003516 | 0.003594 | **2.22%** | ≤2% | FAIL |
| 1 | 0.003165 | 0.003154 | 0.34% | ≤2% | pass |
| 2 | 0.003268 | 0.003358 | **2.76%** | ≤2% | FAIL |

`mean_executed_width`, seed 0: reference 5.686 → re-run 5.170, **Δ = 0.516** against a ±0.25 bar (FAIL). <!-- numcheck-ignore: the failing comparison itself; both sides are read from the two JSONs named immediately below, and quoting them side by side IS this note's purpose -->
The **`--router-hidden-mult 0.5` (rhhalf)** arm passes all three seeds (worst 1.14%) — but note it is
*also* systematically off, merely under the bar.

Reference JSONs (read from JSON, never from the markdown), both under
`automl_package/examples/capacity_ladder_results/W_KDROPOUT_CONVERGED/`:
`w_kdropout_converged_summary_shared_trunk_mse_n1500_s0.05_rhhalf.json` and `..._rhx2.json`.

## Attribution — three runs, and the answer is unambiguous

1. **Not run-to-run variance.** The rhx2 arm was run TWICE on the wave-4 tree
   (`..._fp5_rhx2.json`, `..._fp5_rhx2_rerun.json`): the two are **bit-identical to each other**
   (0.00% on every seed, widths equal to 3 dp) while both miss the reference identically. The code
   is deterministic; the gap is reproducible, not noise. *(This killed the first hypothesis — that
   the bar was simply too tight for a stochastic component.)*
2. **Not FP-5.** The width drivers do not use the package router that FP-5 modifies. They import
   the research script's own router — `automl_package/examples/sinc_width_experiment.py:67`
   (`import capacity_ladder_k6 as ck6`, reusing `_RouterMLP`/`_train_router`/`_soft_targets`) — and
   `git status` confirms FP-5's only source edit is
   `automl_package/models/common/distilled_router.py`. **FP-5's changes are not on this code path at
   all.**
3. **Not wave 4, at all.** The same arm was run in a clean `master` worktree (at `833c68e`, none of
   wave 4's changes present), and its result was preserved into the main ledger as
   `automl_package/examples/capacity_ladder_results/W_KDROPOUT_CONVERGED/w_kdropout_converged_summary_shared_trunk_mse_n1500_s0.05_masterprobe_rhx2.json`.
   **Master reproduces the wave-4 numbers EXACTLY** — 0.00% difference on
   every seed, `mean_executed_width` identical to 3 dp — and misses the reference by the same
   2.22% / 2.76%. *(Process note, recorded because it nearly cost the finding: the FIRST master
   probe's JSON lived inside the throwaway worktree and was destroyed when the root cleaned the
   worktree up — briefly reducing this claim to prose, which the programme's "no claim without an
   artifact" rule says does not exist. It was re-run and copied into the ledger before this note was
   committed. **A diagnostic run's artifact must be landed in the repo ledger BEFORE its scratch
   worktree is removed.**)*

⇒ **The reference artifact is stale with respect to `master`.** Something already merged — waves 1–3
or earlier, all of it already on `master` — shifted this deploy metric. Wave 4 changes nothing here.

## Why the deploy bar moves while the fit bar does not

Offered as the leading hypothesis, **not** as a verified cause. FP-2's five-seed reproduction of
`fit_bar.ratio_to_floor` on this same cell passes to **0.001%** (see FP-2). The fit bar is a
continuous aggregate; `mse_hardpick`/`mean_executed_width` are **discrete** — the router hard-picks
ONE width per input, so a numerically tiny drift can flip a handful of per-input choices and move
the aggregate by a full 2%, and the mean executed width by half a rung. A continuous metric passing
while its discrete sibling fails is consistent with that, and seed 0 (the largest width move, 0.516) <!-- numcheck-ignore: restates the table above, same two JSONs -->
is also the largest MSE move. **Unverified — it predicts which seeds move, and that prediction was
not tested.**

## What was deliberately NOT done

- **The bar was not widened.** It is pre-registered; grading a result against a bar relaxed after
  seeing it is the exact failure the plan fixes the number to prevent.
- **The reference was not regenerated.** Overwriting a reference to make a check pass destroys the
  evidence that something changed.
- **No bisect was run.** Finding which already-merged commit moved the metric is a real
  investigation, out of wave 4's scope, and the run does not open it unasked.

## Recommended follow-up (for the user — NOT scheduled)

1. **Bisect `master` on this cell** (rhx2, seeds 0/2, `deploy_bar.mse_hardpick`) to find the commit
   that moved it. Cheap to automate: one driver invocation per candidate commit, in a worktree.
2. **Then decide, with the cause known:** if the shift is a legitimate improvement, regenerate the
   reference and say so in `width-cert.md`; if it is a regression, it is a defect in already-merged
   code and matters well beyond FP-5.
3. **Consider whether a ±2% bar is the right instrument for a discrete deploy metric at all** — a
   per-input-decision-agreement bar (what fraction of inputs get the same width) would fail loudly
   for the right reason instead of quietly through an aggregate.

## Consequence for FP-5

FP-5's code half is **complete and independently verified**: the four shared router defaults are
unchanged (`git diff` on those lines is empty, as FP-5.b requires), all five re-derived importers
(FP-5.a) import cleanly including `sinc_width_experiment` — the certified-width producer —
`tests/test_distilled_router.py` passes 22/22, and the capability note classifies every protected
symbol. **Only clause (d) is unmet, and it is unmet for a reason external to FP-5.** Marked
**verify-blocked-on-stale-reference**, not failed.
