# ProbReg — fixed-k sub-plan

The base model, with k set by the user. Index: `docs/probreg/PLAN.md`.

## The structure study (PS-1 to PS-4)

Removed 2026-10-22 (mixture framing; docs/probreg/cleanup_plan.md).


# Tasks moved here from the former `probreg.md` (2026-10-21) <!-- citecheck-ignore -->

P2 (the accepted heteroscedastic test failure). Moved verbatim; index: `docs/probreg/PLAN.md`.

<!-- from docs/plans/capacity_programme/probreg.md lines 1263-1339 -->
### P2 — diagnose or accept the undocumented failing test (D3, second one) — ✅ **DONE 2026-07-20. Outcome (c): ACCEPTED, no fix warranted.**

**Verdict: the test is a brittle single-seed assertion, not a package defect.** Both halves of the
task's own verify were executed at the root, on real `pytest`, one variable, both directions: at HEAD
the test fails at MSE 2.8812, and with the recorded candidate fix applied it **still fails**, at  <!-- source: `automl_package/examples/capacity_ladder_results/P2/p2_warmstart_grid.json` -->
2.8565 — a 0.9 % move against a 0.39 gap. **The recorded "fix applied → 1 passed" was refuted.** It  <!-- source: `automl_package/examples/capacity_ladder_results/P2/p2_warmstart_grid.json` -->
had been transcribed from a killed worker's transcript and never re-executed; re-executing it is what
overturned it. Outcome (a) is therefore excluded (the test does not pass) and (b) is excluded (there
is no fix that survives prove-it-fails), so the outcome is the acceptance branch this task's spec
requires — written up, cited, and in the same shape as `docs/probreg/history/diagnoses_2026-07/hetero_nll_diagnosis.md`.

Root cause, established across 5 seeds with per-epoch trajectories (MASTER Decision 9 satisfied):
`k=5` on one hard-coded seed, with a bar sitting 0.25 above a distribution of spread 0.41. Four of
five seeds land under the bar in **both** directions; only the seed the test hard-codes exceeds it.
The culprit commit `445315e` is confirmed — the test passes at its parent — but the crossing is
carried by the RNG-stream shift in that commit, not by the centroid warm-start the earlier note
blamed. **The 2.5 bar was NOT loosened and must not be.** No change to
`automl_package/models/probabilistic_regression.py` is warranted; the file is at HEAD, verified by
`git diff --stat`.

Full record, including the 2×2×5 grid and both trajectories:
`docs/probreg/history/diagnoses_2026-07/p2_hetero_mse_diagnosis.md`. Ledger cells:
`automl_package/examples/capacity_ladder_results/P2/p2_warmstart_grid.json` and
`automl_package/examples/capacity_ladder_results/P2/p2_warmstart_seed42_trajectories.json`.

**✅ RULED 2026-07-20 (user): the test's protocol IS to be repaired, and the repair WAITS for the
selection rule. Do not patch this test twice.** Until then the failure stands as a known failure and
is **not** to be treated as a regression, and the 2.5 bar does not move.

**Why deferred, and what the deferral is really about — this is the load-bearing part, not the seed
noise.** The configuration under test is `n_classes=5` with
`n_classes_selection_method=NONE` (`tests/test_phase4_regression.py:51-62`, `_make_probreg`): no
arbiter, no distilled router, no sweep, no k-dropout. **It is none of §1's three models** — it is the
bare network trained ordinarily at ONE hand-picked rung, closest to a single *cell* of M3's sweep but
without the selection step that makes M3 a model. Its bar was set from that same hand-picked run.
⇒ **The test cannot distinguish "the model regressed" from "k=5 was never the right resolution for
this data"**, because the dial the whole strand is about is frozen. Multi-seed averaging would make
it stable and leave it arbitrary; that is why the seed fix alone was rejected.

**The repair, when its dependency lands:** select `k` on held-out data using the strand's own rule —
cheapest-within-tolerance at twice a bootstrap standard error — instead of pinning `k=5`, holding the
per-seed bar at 2.5. **Tracked dependency: `docs/flexnn/package/plan.md` FP-9.a/FP-9.b** (the shared selector
and its bootstrap standard-error helper) **and PA** (M1's selector built on `all_rung_log_likelihood`).
Not a dispatchable task until both exist; re-read this block when PA closes.

*(Method note, recorded because it cost two rounds: the first two explanations of this failure led
with the noise floor and the per-seed spread and never stated the configuration. The configuration
was the finding. Pin the variant — and whether the mechanism under study is switched on — before
analysing any number.)*

**Also noted, not scheduled:** the
centroid warm-start is shipped, test-uncovered in both directions, and now measured as null on this
configuration (`+0.0026 ± 0.0374` paired over 5 seeds) — keeping it is the recommendation, but it
wants an executable identifiability check before anyone claims it works. <!-- numcheck-ignore: derived statistics over the grid cell cited above -->

*(Original spec retained below, unchanged.)*

**Files (write set):** `docs/flexnn/` (a new diagnosis note) ·
possibly `automl_package/models/probabilistic_regression.py`
**Spec:** Establish whether error 2.8812 against the 2.5 bar is a real regression or a stale bar.
The bar's own comment cites a 1.54 baseline; find when it last held (`git log -S` on the test) and
what changed. Outcome is either a fix, or a written, cited acceptance in the same shape as
`docs/probreg/history/diagnoses_2026-07/hetero_nll_diagnosis.md`. **"Known failure" with no document
is not an outcome.**
**Non-goals:** do not loosen the bar to make it pass. Do not touch the other failing test.
*Orchestration:* parallel: **no — runs AFTER P1** (both may touch `automl_package/models/probabilistic_regression.py`; the old "parallel: yes if it lands doc-only" made the safety of a concurrent dispatch depend on an outcome the dispatcher cannot know in advance) · deps: **P1** · tier: sonnet
high · scale: static · shape: research ·
verify: `test -f docs/flexnn/<note>.md` exits 0, and
`grep -qE "^[0-9a-f]{7,40}" docs/flexnn/<note>.md` shows the commit at
which the bar last held (from `git log -S` on the test). Then exactly one of two outcomes, both
mechanically checkable: **(a) stale bar** —
`AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m pytest tests/test_phase4_regression.py::TestPerformanceBaselines::test_prob_regression_heteroscedastic_mse -q`
passes, and the note says so; **(b) real regression** — same prove-it-fails shape as P1's template:
the fix is applied and the suite is green, THEN the fix is deleted, the same test is re-run and
shown FAILING, then the fix is restored and the file checksum is shown unchanged. "Known failure,
no fix, no acceptance note" is not a valid outcome under either branch.


<!-- from docs/plans/capacity_programme/probreg.md lines 1471-1604 -->
### P11 — the head-structure comparison

Removed 2026-10-22 (mixture framing; docs/probreg/cleanup_plan.md).
