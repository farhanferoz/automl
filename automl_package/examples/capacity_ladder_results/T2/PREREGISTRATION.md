# T2 pre-registration — multi-dimensional count toys (the port de-risk)

Source: `docs/plans/perinput_selector_2026-07-10/EXECUTION_PLAN.md`, §2 WS-B, task **T2**.
Written BEFORE any real run, verbatim from the ratified plan (2026-07-10). Do not edit after a
real run starts; outcomes are read against this text, not the reverse.

## Purpose

Everything in the per-input selector program so far is 1-D. The real-model ports are gated
precisely on "neighbourhood reads may break in many dims" — T2 measures exactly that
degradation on toys with analytic ground truth.

## Toys (extend `_toy_datasets.py`)

`make_toy_d_ndim(n, dim, rotated, seed)`: x ~ U[0,1]^dim; staircase k*(s) ∈ {1,2,3} by thirds of
s, where s = x[0] (axis-aligned) or s = u·x/√dim with u a fixed known unit vector (rotated);
remaining coordinates are nuisance. Same component geometry as toy D (separation 4σ, σ=0.3,
means centred at 0). Broad twin at dim=2 (variance-matched, k*=1 everywhere). Matrix: dim=2 axis
+ broad, dim=5 axis + rotated, dim=10 axis → 5 configs × 3 seeds = 15 ladder trainings.
N_train = N_test = 2500 (K4-scale).

## Ladder

Generalize `_capacity_ladder_nested.NestedKSurrogate` to input_dim ≥ 1 (constructor arg; trunk
widens to (64,64) for dim>1). REGRESSION GUARD: with input_dim=1 the loss trajectory on toy D
seed 0 must be bit-identical to current code (selftest assert) — the 1-D certified results must
be unreproducible-risk-free.

## Reads

Neighbourhoods by kNN (n_nbr=50, Euclidean) replacing the 1-D box-car; targets = S1's
smoothing-only soft variant (arm 3) built with kNN smoothing (the tercile prior needs a
binnable scalar — in multi-D use terciles of the TRUE index s ONLY for the gold read, never for
targets: selector-visible information is x alone). Selector = S1/S2 winner recipe, input dim =
dim. G5 analog: re-read at n_nbr=25. Gold read: per-region capture via
`sample_toy_d_ndim_given_x` (analytic k*(s)).

## Pre-registered bars

(i) dim=2 axis: selector blend beats global > 2·SE on ≥ 2/3 seeds (one nuisance dim must not
kill a 4σ staircase);
(ii) the DELIVERABLE is the degradation curve — (selector − global) advantage and gold
region-capture vs dim ∈ {2,5,10}; pre-registered read: report the dim at which the advantage
drops below 2·SE (no pass/fail at 5/10);
(iii) rotated-vs-axis at dim=5: paired diff reported; prediction: kNN targets are
rotation-invariant → |diff| within 2·SE (a material gap = binning-artifact tell);
(iv) dim=2 broad twin: selector advantage ≤ 0.02 nat.

## Selftest

input_dim=1 regression guard (above) + 2-D synthetic known-answer staircase recovered.

## Script/artifacts

`automl_package/examples/capacity_ladder_t2.py` →
`capacity_ladder_results/T2/{PREREGISTRATION.md,t2_summary.json,nested_toyD_ndim_<config>_seed<seed>.pt}`.

## Run

`AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python -u
automl_package/examples/capacity_ladder_t2.py --selftest` first, then `--measure-one` (times ONE
dim=5-axis unit, extrapolates the matrix, exits — the plan's "measure ONE dim=5 ladder first,
extrapolate, report in ledger before launching the matrix"), then the full run (no flags).

## Cost

The compute bulk — expect ~1–1.5 days serialized. Measure one dim=5 unit first.

## Non-goals

dim > 10; real datasets; library changes beyond the examples-dir surrogate.

---

## CRITICAL: bar (i) risk — the prior mechanism cannot transfer to dim > 1 (stated up front, not discovered post-hoc)

S1 certified two findings on the frozen 1-D K4 tables (`capacity_ladder_results/RESULTS.md`,
"## S1" section): (1) `soft` (per-tercile PRIOR-stacked responsibilities) is the winning target
construction; (2) the prior is the LOAD-BEARING mechanism and neighbour SMOOTHING alone (S1's
arm 3, "soft_smoothed") contributes almost nothing on top of it — the factorial's 3v2 read was
+0.0001 nat (1.28·SE, ~60× below the prior's own +0.0067 nat, 3.13·SE effect).

The per-tercile prior needs a BINNABLE SCALAR to define its bins on. In `dim > 1` there is no
such scalar the selector may legitimately see: the selector's input is `x` alone (L3/L5), and
using the analytic staircase index `s` to build a prior would leak the very ground truth T2 is
built to test recovery of. So T2's target construction (`_knn_soft_targets` in
`capacity_ladder_t2.py`) can only be the kNN generalization of S1's arm 3 (smoothing-only, no
prior) — the mechanism S1 measured as **near-zero** in 1-D.

**Prediction, stated before any real run:** bar (i) may well be WEAK or FAIL even at dim=2 with
only one nuisance dimension, not because the kNN machinery is broken, but because the target
recipe available to it in multi-D is built from the mechanism S1 already found to carry almost
no signal on its own. This is a **legitimate pre-registered outcome** — report it, escalate per
EXECUTION_PLAN.md §0b (STOP the lane → fresh-context adjudicator on any failed bar) — **not** a
build defect, and not something to be tuned away (e.g. by smuggling a prior back in through the
gold index `s`, which would invalidate the whole de-risk).

"Selector = S1/S2 winner recipe, input dim = dim" (the plan's own T2 spec line) is read here as
the TRAINING SCHEME transferring — soft-label cross-entropy, router architecture/hyperparameters
(hidden (32,32), Adam lr 1e-2, 300 epochs), the blend-primary evaluation, and the honest
oracle-x bound, all generalized to `input dim = dim` — **not** the target-construction mechanism,
which structurally cannot transfer for the reason above. See
`automl_package/examples/capacity_ladder_t2.py`'s module docstring for the full reasoning.

## Implementation notes (not part of the plan text, added by the authoring worker)

- **`NestedKSurrogate` generalization resolved as a NO-OP on the library-adjacent file.**
  `_capacity_ladder_nested.NestedKSurrogate.__init__` already takes `input_dim` as a required
  first argument and is already called with arbitrary `input_dim` (K4:
  `NestedKSurrogate(input_dim=x_t.shape[1], ...)`); `train_nested_k_surrogate` already exposes
  `hidden` as a constructor-forwarded argument. `_capacity_ladder_nested.py` is therefore **not
  modified** by T2 — the ONLY new behaviour is `capacity_ladder_t2._fit_ladder` choosing
  `hidden=64` instead of the current default `32` when `dim > 1` (`_trunk_hidden`), via the
  EXISTING `hidden` arg, from the driver script. Because the underlying function is byte-for-byte
  unchanged, the REGRESSION GUARD holds structurally, not merely by construction of a matching
  test; `run_selftest`'s check (a) still asserts it explicitly (two independent calls — one
  through `_fit_ladder(dim=1, ...)`, one calling `train_nested_k_surrogate` directly with the
  literal current 1-D convention `hidden=32` — produce a bit-identical held-out score table,
  `max abs diff = 0.0`, verified 2026-07-10) so a FUTURE edit to `_fit_ladder`'s dim→hidden
  mapping that accidentally touches `dim=1` is caught, not just today's state.

- **Router and kNN-curve reader are NEW, LOCAL code in `capacity_ladder_t2.py`, not edits to
  other tasks' files.** `capacity_ladder_k6._RouterMLP`/`_train_router` hardcode a scalar input
  (`x.reshape(-1, 1)`); `_capacity_ladder.perinput_curve` is a 1-D box-car. Both are used
  (read-only, imported) by other certified tasks (K6, S1, S2, K4, K5) and are not touched. T2
  carries its own `_RouterMLP` (input_dim generalized, otherwise identical architecture/training
  scheme) and its own `knn_curve` (mirrors `perinput_curve`'s `delta`/`delta_half` dict interface
  exactly, so S1's target-construction and oracle-x code ports over with only the neighbourhood
  source function swapped).

- **The "rotated" direction `u` is a fixed isotropic unit vector, not literally re-derived from
  the plan's terse formula in a way that guarantees a uniform-shaped `s`.** `u` is drawn once per
  `dim` from a constant internal seed (`_toy_d_ndim_direction`, `_toy_datasets.py`) and normalized
  to unit L2 norm; `s = (u·x)/√dim` is bounded in `[0, 1]` by Cauchy-Schwarz for any such `u`
  regardless of sign pattern, but its realized marginal shape concentrates more tightly around its
  mean than a genuine `Uniform[0, 1]` draw (a mild CLT effect, growing with `dim`) — so the
  axis-aligned and rotated marginal distributions of `s` are not identically shaped even though
  both live in `[0, 1]`. `_staircase_k`'s hardcoded `1/3, 2/3` cutoffs are applied to `s` either
  way, per the plan's literal "by thirds of s" text — this is a KNOWN, documented caveat (see
  `_toy_datasets.toy_d_ndim_s`'s docstring), reported here rather than silently corrected: bar
  (iii)'s rotated-vs-axis paired diff at dim=5 could partly reflect this distributional-shape
  difference rather than a pure kNN-rotation-invariance effect. The plan's own bar (iii) framing
  ("a material gap = binning-artifact tell") already treats either outcome as informative, not
  pass/fail, so this caveat changes how a material gap should be INTERPRETED, not whether the bar
  passes.

- **Broad twin (`make_toy_d_ndim_broad`) variance-matching is closed-form, not empirical.** For a
  `k`-component equal-weight mixture spaced `separation·σ` apart, the component index is uniform
  over `{0, ..., k-1}` (variance `(k²-1)/12`), so
  `Var(offset) = separation² · σ² · (k²-1)/12` and total marginal variance is
  `σ²·(1 + separation²·(k²-1)/12)` (`_toy_d_ndim_total_var`). The broad twin draws a single
  Gaussian with exactly this variance at each region's `k*(s)`, never a second/third mode — the
  multi-D analog of `make_toy_c_broad`/`make_toy_e_broad`'s role.

- **Only ONE bar is gated with a fixed pass/fail threshold — bar (i).** Bars (ii)/(iii) are
  explicitly report-only per the plan's own text ("no pass/fail at 5/10"; "paired diff reported");
  bar (iv) is gated (≤ 0.02 nat, S1's convention). `capacity_ladder_t2.py`'s `_bar_ii`/`_bar_iii`
  therefore return descriptive curves/diffs with no `pass` field, matching this.

- **k_max = 8**, matching K4's toy-D setting (`KMAX["D"]=8` in `capacity_ladder_k4.py`) — the plan
  states "same component geometry as toy D," which this reads as including the same capacity
  ceiling/headroom, not just the mixture parameters (separation/σ).
