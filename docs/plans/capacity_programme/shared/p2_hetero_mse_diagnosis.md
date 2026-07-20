# P2 — `test_prob_regression_heteroscedastic_mse`: REAL REGRESSION, candidate fix NOT applied

**Status: diagnosis COMPLETE. Candidate fix RECOVERED and recorded below. NOT APPLIED — it needs a
ruling first (see "Why this is not applied").**

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

## Verification the worker ran (quoted from its transcript)

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
