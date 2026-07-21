# P2 — `test_prob_regression_heteroscedastic_mse`: REAL REGRESSION, candidate fix NOT applied

**Status: ADJUDICATED 2026-07-20 → VERDICT = DO NOT APPLY. The candidate fix below does NOT work.**

> ⛔ **READ THIS BEFORE ANY SECTION BELOW. The recorded premise of this note is REFUTED.**
> The four-line deletion recorded here as a working candidate fix **does not make the test pass.**
> Re-run at the root, one variable, both directions, on the same test:
>
> | state of `probabilistic_regression.py` | result |
> |---|---|
> | HEAD, unmodified | **FAILED** — `ProbReg MSE (2.8812) regressed past 2.5` |  <!-- source: `automl_package/examples/capacity_ladder_results/P2/p2_warmstart_grid.json` -->
> | four lines deleted (the candidate fix) | **FAILED** — `ProbReg MSE (2.8565) regressed past 2.5` |  <!-- source: `automl_package/examples/capacity_ladder_results/P2/p2_warmstart_grid.json` -->
>
> The deletion moves the metric by 0.025 (0.9 %); the gap to the bar is 0.39. **The "fix applied →
> 1 passed" line in the *Verification* table below could not be reproduced and must not be relied
> on** — it was transcribed from a killed worker's transcript and never re-executed until now.
> The 2.5 bar was NOT loosened and must not be.
>
> The rest of this note is retained verbatim as the recovered record, because the culprit-commit
> half of it stands (`445315e` IS the commit that broke the test — independently reconfirmed on a
> detached worktree at `445315e^`, where the test passes at MSE 2.1515). What does *not* stand is  <!-- numcheck-ignore: run at a detached worktree on 445315e^, which has no ledger cell by construction -->
> the attribution of the regression to the warm-start override and the claim that deleting it fixes
> anything. **The adjudication section at the end of this file supersedes the two sections below it
> on those points.** *(Case law, and the reason this note exists in this shape: a verification quoted
> from a transcript is not a verification. Re-execute before building on one.)*

**Provenance, stated plainly.** The worker that produced this was stopped at a context boundary
before it could write its own note, having landed nothing to disk. The diagnosis, the culprit
commit, the candidate fix and the verification runs below were **recovered from its transcript and
its scratchpad** by the root and transcribed here. Every number is quoted from that record; the
culprit commit and the current failure were independently re-verified at the root. Anything the
worker did not check is listed at the end rather than filled in.

## The failure

`tests/test_phase4_regression.py::TestPerformanceBaselines::test_prob_regression_heteroscedastic_mse`
fails with **error 2.8812 against a 2.5 bar**. It is pre-existing (verified identical to 16 digits
on a clean worktree at `600460c`) and still failing at `8dea937` — the wave-1 suite closed at
305 passed / 2 failed / 1 skipped with this as one of the two.

## Verdict: (b) REAL REGRESSION, not a stale bar

**Culprit commit: `445315e` (2026-04-20) — *"phase 10 prep: identifiability diagnostics + centroid
warm-start"*.** Re-verified at the root:

```
$ git log -1 --format='%h %ad %s' --date=short 445315e
445315e 2026-04-20 phase 10 prep: identifiability diagnostics + centroid warm-start
```

Two changes in that commit are implicated, and the worker separated them by direct experiment
rather than by inspection:

1. **`ConstantHead.__init__`** (`automl_package/models/common/regression_heads.py`, ~`:215-220`) —
   `self.mean` and `self.log_variance` moved from `nn.Parameter(torch.randn(1))` to
   `nn.Parameter(torch.zeros(1))`.
2. **A centroid warm-start override** in `ProbabilisticRegressionModel.build_model`
   (`automl_package/models/probabilistic_regression.py`, ~`:431`) that calls
   `init_middle_class_mean` when a stashed init value is present.

The worker probed (1) with a DIAGNOSTIC revert to `randn`, then **reverted that probe** — the
recovered candidate copy of `regression_heads.py` is byte-identical to HEAD. **So the change that
carries the regression is (2), the warm-start override.**

## The candidate fix (recorded, NOT applied)

Delete these four lines from `build_model` in
`automl_package/models/probabilistic_regression.py` (present at HEAD around `:431-434`):

```python
        init_val = getattr(self, "_constant_head_init_value", None)
        if init_val is not None:
            self.model.regression_module.init_middle_class_mean(init_val)

```

That is the entire diff. `regression_heads.py` is unchanged.

## Verification the worker ran (quoted from its transcript) — ⛔ ROW 1 AND ROW 2 REFUTED, see the banner

| direction | result |
|---|---|
| fix applied → the failing test | `1 passed` (12.74s; also `1 passed` in 17.83s on a re-run) |
| **fix removed → same test** | **`1 failed`** (10.05s) — prove-it-fails satisfied |
| fix applied → whole `test_phase4_regression.py` | `27 passed` in 61.82s |
| plan gates | `9 passed` |

It also recorded checksums either side of the revert/restore cycle and confirmed the work was done
against **current HEAD**, not a stale worktree.

## ⛔ Why this is NOT applied

**The fix works by deleting a feature that was added deliberately.** `445315e` introduced the
centroid warm-start as part of "identifiability diagnostics" — plausibly to fix a real problem of
its own, on which no evidence was gathered here. Making this test pass by removing it may trade
one regression for another, silently, in exactly the direction this programme keeps getting
caught: **a green suite is not the same claim as a correct model.**

Before it lands, someone must establish:
- **what the warm-start was for**, and whether anything still depends on it (grep
  `init_middle_class_mean` / `_constant_head_init_value` callers, and read `445315e` in full);
- whether the warm-start is **wrong in general** or only wrong **for the heteroscedastic case** —
  if the latter, the fix is a condition, not a deletion;
- whether removing it moves the *other* accepted failure
  (`test_probabilistic_nll_beats_constant_on_heteroscedastic`, root-caused and accepted in
  `hetero_nll_diagnosis.md`) in either direction. These two tests are on the same substrate and
  nobody has checked whether they are the same underlying defect.

**Do NOT loosen the 2.5 bar under any outcome** — that remains forbidden.

## What was NOT checked

- No trajectory evidence. The worker read pass/fail on the bar, not per-epoch behaviour, so
  "the warm-start hurts" is currently an endpoint claim (MASTER Decision 9 is not satisfied).
- Single seed. The regression was not shown stable across seeds.
- No test of the warm-start's *intended* benefit — i.e. whether deleting it breaks the
  identifiability property `445315e` was added for. **This is the load-bearing gap.**
- The 1.54 baseline cited in the bar's own comment was not tied to a specific commit.

---

# ADJUDICATION (P2 open half) — **VERDICT: DO NOT APPLY**

**The recorded premise is refuted by direct reproduction.** The candidate fix does **not** make
`test_prob_regression_heteroscedastic_mse` pass. Applied verbatim to `build_model`, the test still
fails at **MSE 2.8565** against the 2.5 bar. The recorded "fix applied → `1 passed`" could not be  <!-- source: `automl_package/examples/capacity_ladder_results/P2/p2_warmstart_grid.json` -->
reproduced and should not be relied on.

Everything below was re-derived at `capacity/wave-2`; `probabilistic_regression.py` was restored to
HEAD after every local application of the diff (`git diff --stat` on that path: empty).

## R0 — The load-bearing refutation

Real `pytest`, real test, one variable, both directions:

```
HEAD (unmodified)          FAILED — ProbReg MSE (2.8812) regressed past 2.5  <!-- source: `automl_package/examples/capacity_ladder_results/P2/p2_warmstart_grid.json` -->
4-line deletion applied    FAILED — ProbReg MSE (2.8565) regressed past 2.5  <!-- source: `automl_package/examples/capacity_ladder_results/P2/p2_warmstart_grid.json` -->
```

The deletion moves the metric by **0.025 (0.9 %)**. The gap to the bar is **0.39**. Whatever the
earlier worker observed, the diff as recorded does not clear the bar.

## R1 — What the warm-start is for, and whether anything depends on it

`git show 445315e` (full commit read). The warm-start is one of two coupled changes to the same
mechanism, both introduced there:
- `regression_heads.py`: `ConstantHead`, `ProbabilisticMiddleClassHead` and
  `SingleHeadNOutputsRegressionModule`'s middle-class parameters moved `randn → zeros`, and
  `init_mean` / `init_middle_class_mean` were added ("zero-init learnable constants (was randn,
  noisy)").
- `probabilistic_regression.py`: the 4-line call in `build_model`.

Together they replace *a random constant* for the middle-class head with *the middle bin's own
mean*. Stated purpose (commit body + `docs/probreg_identifiability_implementation_plan.md`) is the
head-class index-swap degeneracy in SEPARATE_HEADS under REGRESSION_ONLY: heads should start near
their class centroid rather than at an arbitrary random value.

**Dependents — exhaustive grep** (`init_middle_class_mean`, `_constant_head_init_value`, whole repo
minus `.venv`):

| site | kind |
|---|---|
| `automl_package/models/probabilistic_regression.py:431,433` | the call under judgement |
| `automl_package/models/probabilistic_regression.py:575,577` | producer (`constrain_middle_class` and odd `max_k` only) |
| `automl_package/models/common/regression_heads.py:351,430,494` | the three module implementations |
| `automl_package/examples/capacity_ladder_h1.py:232,234` | replicates the bookkeeping so arm (b)'s hand-rolled loop starts from the same init as arms (a)/(c) |

- **No test anywhere references either symbol** (`grep tests/` → 0 hits). The feature is entirely
  test-uncovered; nothing in the suite would notice its removal.
- The one external consumer, `capacity_ladder_h1.py`, wants *parity across its arms*. All three arms
  reach the net through `build_model`, so deleting the call keeps them in parity — but it would
  silently turn that example's documented replication into dead code.
- The identifiability property the feature exists for has **no executable check**: the plan doc's
  acceptance tests (CT1–CT5, `head(p=1.0) ≈ centroid`) were written for the *anchored-head* path,
  not for this warm-start.

**Conclusion for R1:** the warm-start is not load-bearing for any test or documented claim, but
neither is it verified to be doing its job. It is unmeasured, in both directions.

## R2 — Wrong in general, or wrong only here? **Neither — it is measurably null.**

The warm-start's entire effect on this configuration is: the middle `ConstantHead`'s scalar mean is
initialised to `mean(y | middle percentile bin)` instead of `0.0`. Measured value on the failing
config: **+0.0801**. It is a `no_grad` `fill_`, so it consumes no RNG and perturbs nothing else.  <!-- source: `automl_package/examples/capacity_ladder_results/P2/p2_warmstart_grid.json` -->

Grid, exact test configuration, 5 seeds (seed drives data generation, split, and model init
together; `seed=42` reproduces the shipped test exactly — 2.8812 to 4 dp). `warm=False` is the  <!-- source: `automl_package/examples/capacity_ladder_results/P2/p2_warmstart_grid.json` -->
candidate deletion:

| seed | warm-start value (2 dp; exact in the cell) | MSE **with** warm-start (HEAD) | MSE **without** (candidate fix) | Δ (without − with) |
|---:|---:|---:|---:|---:|
| 42 | +0.08 | **2.8812** | **2.8565** | −0.0247 |  <!-- source: `automl_package/examples/capacity_ladder_results/P2/p2_warmstart_grid.json` -->
| 0 | −0.04 | 1.8821 | 1.9144 | +0.0323 |  <!-- source: `automl_package/examples/capacity_ladder_results/P2/p2_warmstart_grid.json` -->
| 1 | +0.04 | 2.1219 | 2.1697 | +0.0478 |  <!-- source: `automl_package/examples/capacity_ladder_results/P2/p2_warmstart_grid.json` -->
| 2 | −0.17 | 2.4368 | 2.4340 | −0.0028 |  <!-- source: `automl_package/examples/capacity_ladder_results/P2/p2_warmstart_grid.json` -->
| 7 | −0.21 | 1.9273 | 1.8879 | −0.0394 |  <!-- source: `automl_package/examples/capacity_ladder_results/P2/p2_warmstart_grid.json` -->
| **mean ± sd** | | **2.2499 ± 0.4151** | **2.2525 ± 0.4008** | **+0.0026 ± 0.0374** |  <!-- numcheck-ignore: derived statistics over the cited grid cell, not leaves of it -->

Paired mean effect **+0.0026** on a per-seed spread of **±0.41**: `t ≈ 0.16`. The warm-start
**neither helps nor hurts**. It wins on 2 seeds, loses on 3, and the sign on the failing seed (the
only one the recorded diagnosis looked at) is an artifact of that seed.

**So there is no condition to write.** A conditional version of the fix would be conditioning on
noise.

## R2b — Then what actually changed at `445315e`?

The commit *is* the culprit, confirmed on a detached worktree at `445315e^` (`ccd5755`):

```
$ pytest tests/test_phase4_regression.py::TestPerformanceBaselines::test_prob_regression_heteroscedastic_mse
1 passed          # MSE = 2.1515  <!-- numcheck-ignore: run at a detached worktree on 445315e^, which has no ledger cell by construction -->
```

The 2.5 bar and the test body are unchanged since `597a416`; the test file was last touched at
`07c0b09`, before `445315e`. So this is a real crossing, not a stale bar — that part of the
existing verdict stands.

But the crossing is **not** carried by the warm-start. Full 2×2 on seed 42 (`randn` = the
pre-`445315e` `ConstantHead` parameter init):

| `ConstantHead` init | warm-start | seed-42 MSE | vs 2.5 |
|---|---|---:|---|
| `zeros` (HEAD) | on (HEAD) | 2.8812 | fail |  <!-- source: `automl_package/examples/capacity_ladder_results/P2/p2_warmstart_grid.json` -->
| `zeros` | off (candidate fix) | 2.8565 | fail |  <!-- source: `automl_package/examples/capacity_ladder_results/P2/p2_warmstart_grid.json` -->
| `randn` | on | 2.6106 | fail |  <!-- source: `automl_package/examples/capacity_ladder_results/P2/p2_warmstart_grid.json` -->
| `randn` | off (= pre-`445315e`) | 2.1732 | pass |  <!-- source: `automl_package/examples/capacity_ladder_results/P2/p2_warmstart_grid.json` -->

**Neither half of the commit alone restores the pass.** Only the conjunction does — and the
`randn → zeros` change removes RNG draws from the stream, so "reverting" it does not restore a
mechanism, it restores a *random sequence*. Across all 5 seeds the `randn` cells average 2.09 vs
2.25 for `zeros` (paired mean +0.184 ± 0.240, `t ≈ 1.7`, n=5) — suggestive at most, not established,  <!-- numcheck-ignore: derived statistics over the cited grid cell, not leaves of it -->
and pointing the wrong way for a change whose stated purpose was to reduce init noise.

The honest reading: `445315e` shifted the RNG stream, and seed 42 landed on the far side of a bar
that sits 0.25 above a distribution with sd 0.41.

## R3 — Effect on the sibling accepted failure: **essentially none**

`tests/test_phase1_probabilistic_regression.py::TestModelComparison::test_probabilistic_nll_beats_constant_on_heteroscedastic`,
real `pytest`, both directions:

| | ProbReg NLL | constant-σ NN NLL | result |
|---|---:|---:|---|
| HEAD | 1.8430 | 1.6885 | FAIL |  <!-- numcheck-ignore: pytest output for the NLL sibling test, not a ledger cell -->
| candidate fix applied | **1.8228** | 1.6885 | FAIL |  <!-- numcheck-ignore: pytest output for the NLL sibling test, not a ledger cell -->

Moves 0.020 in the right direction on a 0.155 gap — same order as the MSE effect, same conclusion:  <!-- numcheck-ignore: a gap computed from the two pytest values above -->
null. The two failures are **not** fixed by a common change here, though R4 below shows they *do*
share a root cause (single-seed brittleness), consistent with `hetero_nll_diagnosis.md`'s Outcome 2.

## R4 — Trajectory and seeds (MASTER Decision 9 now satisfied)

Per-epoch held-out **test** MSE captured via the built-in `epoch_callback` hook (no package edits).
Seed 42, both directions:

| | epochs 1–10 | last 12 epochs | best epoch |
|---|---|---|---|
| warm-start ON | 4.05, 3.29, 3.01, 2.94, 2.85, 2.83, 2.84, 2.86, 2.85, 2.83 | 3.00 … 2.87, 2.88 | **2.812** (ep 46) |  <!-- source: `automl_package/examples/capacity_ladder_results/P2/p2_warmstart_seed42_trajectories.json` -->
| warm-start OFF | 4.08, 3.30, 3.00, 2.93, 2.85, 2.83, 2.84, 2.86, 2.84, 2.83 | 2.85 … 2.85, 2.86 | **2.812** (ep 49) |  <!-- source: `automl_package/examples/capacity_ladder_results/P2/p2_warmstart_seed42_trajectories.json` -->

Two things this settles:
1. **The two trajectories are indistinguishable at every epoch**, from epoch 1 onward. There is no
   phase in training at which the warm-start matters.
2. **Seed 42 never reaches the bar in either direction.** The best held-out MSE at *any* epoch is
   2.812 — the model is plateaued, not under-trained, and 2.5 is unreachable on this split for this  <!-- source: `automl_package/examples/capacity_ladder_results/P2/p2_warmstart_seed42_trajectories.json` -->
   configuration. Reporting the endpoint was not what hid the answer; the answer is that there is no
   epoch that passes.

**Stability across seeds — stated plainly: the regression is NOT stable.** 4 of 5 seeds land under
the 2.5 bar (1.88, 1.91, 2.12, 2.43) in both directions; only seed 42 — the one the test hard-codes
— exceeds it, at 2.88, which is +1.5 sd from the 5-seed mean of 2.25. This is the same failure shape
`hetero_nll_diagnosis.md` root-caused for the NLL sibling: a brittle single-seed assertion inside
the seed-noise band.

## What would have to be true to revisit

The deletion becomes worth reconsidering only if **all** of the following are established, none of
which are today:
1. A **mechanism** for the warm-start being harmful that survives more than 5 seeds — the measured
   paired effect is `+0.003 ± 0.037`, so a real effect would need a much larger `n` to separate from
   zero, and would have to be shown on more than this one toy.
2. An **executable check for the identifiability property** `445315e` added it for (a test that
   `head(p_i → 1) ≈ c_i` for the constrained middle head, analogous to the plan doc's CT2a for
   anchored heads). Until that exists, deleting the feature trades an unmeasured property for a
   0.9 % metric move — precisely the trade this note's original ⛔ section warned against.
3. A demonstration that whatever fixes seed 42 **also holds on the other seeds**, rather than
   re-rolling the RNG stream until the shipped seed happens to pass. The `randn` column is exactly
   that failure mode and must not be adopted.

## ✅ SETTLED 2026-07-20 (user ruling) — the protocol repair is DEFERRED, not declined

The test's protocol **is** to be repaired, but **not before the strand's selection rule exists**, so
this test is patched once rather than twice. Until then this stands as a known failure and is not a
regression; the 2.5 bar does not move. The reason is not the seed: the configuration pins
`n_classes=5` with selection switched **off**, so it is none of the strand's three models and its bar
cannot separate a real regression from a badly-chosen resolution. Full ruling, the repair's shape and
its tracked dependency: `docs/plans/capacity_programme/probreg.md`, task P2.

## What this leaves open (superseded by the ruling above — retained as the reasoning that led to it)

The MSE failure is now root-caused to the same thing as its NLL sibling: `k=5` on a single
hard-coded seed, with a bar 0.25 above a distribution of sd 0.41. **The 2.5 bar was not loosened and
must not be** — but nothing in this adjudication makes the test pass, and no fix to
`probabilistic_regression.py` is warranted. Whether to accept it as a second known failure alongside
`test_probabilistic_nll_beats_constant_on_heteroscedastic`, or to change the test's *protocol*
(multi-seed, val-selected `k`) while holding the per-seed bar at 2.5, is a decision for the
programme owner, not a code change to be made under this task.

**Reproduction artifacts:**
`/tmp/claude-1000/-home-ff235-dev-MLResearch-automl/48b5c295-16f2-4366-b47d-9afc6baac146/scratchpad/p2_grid.py`
(2×2×5 grid + trajectories) and `p2_grid.json` (raw per-epoch data).
