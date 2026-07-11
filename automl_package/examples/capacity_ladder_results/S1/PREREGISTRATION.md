# S1 preregistration — evaluation protocol + target-construction factorial (ProbReg)

Source: `docs/plans/perinput_selector_2026-07-10/EXECUTION_PLAN.md`, §1 WS-A, task **S1**.
Written BEFORE any real run, verbatim from the ratified plan (2026-07-10). Do not edit after
a real run starts; outcomes are read against this text, not the reverse.

## Purpose

Replace the confirmatory 3-arm comparison with a clean factorial + knob sweep, under a
corrected evaluation protocol (honest bound; blend-vs-hard axis).

## Inputs (all exist, frozen — no ladder retraining)

`automl_package/examples/capacity_ladder_results/K4/nested_toy{C,D,E}_seed{0,1,2}.pt` and the
broad-twin controls `nested_toy{C,E}_broad_seed{0,1,2}.pt` (keys: `score` (N×C float64
held-out per-example log-likelihood), `x`, `c_grid`). Reuse `capacity_ladder_k6.py` patterns
verbatim: `_RouterMLP` (32,32), 300 ep, lr 1e-2, index-parity train/eval split, targets
computed on the train half only.

## Protocol (applies to every arm, pre-registered)

- **Blend read** (primary, per L3): per-input weighted density
  `blend_i = logsumexp_c(log w_c(x_i) + score[i,c])` with the selector's own weights; report
  alongside the hard argmax-routed NLL.
- **Honest bound `oracle-x`**: split the eval half again by index parity; estimate the
  neighbour-averaged (box-car width 0.075) per-capacity advantage curve on the even eval
  points; route the odd eval points by its argmax; report their mean actual score. Keep the
  old per-point max as `oracle-noisy` (continuity label only).
- Metrics on the eval half; 3 seeds; plain bootstrap SE (B=1000) on paired diffs.

**Factorial arms** (identical router config; only the target changes):
1. soft = K6 soft (responsibilities with per-tercile stacked prior) — baseline;
2. soft-no-prior = per-row `softmax_c(score[i,:])` (isolates the prior);
3. soft-smoothed = softmax of the neighbour-averaged score rows, no prior (isolates
   smoothing);
4. hard knee labels (K6 hard);
5. raw per-row argmax (K6 pilot).

Factorial reading: 1v2 = prior effect; 2v5 = softness effect; 3v2 = smoothing effect; 1v3 =
prior-vs-smoothing form.

**Knob sweep** (winning target only; toys D + C_broad, 3 seeds): prior bins ∈ {2,3,5};
neighbour width ∈ {0.0375, 0.075, 0.15}; target temperature τ ∈ {0.5, 1, 2} (soft targets ∝
exp(log-target/τ), renormalized).

## Pre-registered bars

(i) blend ≥ hard for every arm on ≥ 8/9 structured cases;
(ii) some soft arm ≥ hard-knee arm on ≥ 7/9 (K6 replication under the new protocol);
(iii) oracle-x ≤ oracle-noisy on 9/9;
(iv) broad controls: every arm's blend advantage over global ≤ 0.02 nat on all 6 broad cases;
(v) knob robustness: winner's ordering vs arm (4) unchanged across the full knob grid — if
not, the sensitivity table IS the finding (report, don't tune).

## Selftest

Reuse K6 `_selftest_table`; add asserts: blend ≥ hard on the synthetic table; oracle-x
recovers the designed tercile peaks {1,3,6}; τ=1 reproduces the τ-free target bit-identically.

## Script/artifacts

`automl_package/examples/capacity_ladder_s1.py` →
`capacity_ladder_results/S1/{PREREGISTRATION.md,s1_summary.json}`.

## Run

`AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python -u
automl_package/examples/capacity_ladder_s1.py` (selftest first with `--selftest`).

## Cost

Minutes–1 h.

## Non-goals

No ladder retraining; no new toys; no library code.

---

## Implementation notes (not part of the plan text, added by the authoring worker)

The plan states the bars/protocol at the level of "blend ≥ hard", "oracle-x ≤ oracle-noisy",
etc. — informal shorthand that is ambiguous between raw log-score (higher-is-better) and NLL
(lower-is-better) sign conventions. The implementation (`capacity_ladder_s1.py`) resolves this
by reasoning from the underlying construction (documented in-line at each bar):

- Bars (i)/(ii): read as an NLL comparison (`nll_blend <= nll_hard`), which is the
  sign-equivalent of "blend ≥ hard" in raw log-score terms — implemented via K6's own
  established convention (`nll_soft <= nll_hard`, etc.).
- Bar (iii): read as a RAW mean-score comparison (`oracle_x_score <= oracle_noisy_score`,
  un-negated), because the plan explicitly says oracle-x reports "mean actual score" (not
  NLL) and because the honest, information-constrained bound cannot legitimately exceed the
  cheating per-point-max bound in log-score space — the opposite direction would be a bug, not
  a finding. In NLL terms this is `oracle_x_nll >= oracle_noisy_nll`.
- Bar (v) / the knob sweep's target-construction recipe: flagged as a genuine, unresolved
  design ambiguity in the S1 build reply (see the orchestrator's dispatch log) — the plan lists
  three orthogonal knobs (prior bins, neighbour width, temperature) to sweep on "the winning
  arm only", but two of the five factorial arms structurally use only ONE of the two
  mechanisms (arm 1: prior only; arm 3: smoothing only; arm 2 uses neither), so at most 2 of
  the 3 knob axes are meaningful for any single-mechanism winner. The implementation sweeps
  only the axes the winning arm's own recipe actually has (temperature always; prior-bins only
  if the winner uses a prior; width only if the winner uses smoothing), and falls back to the
  best-performing soft arm if a hard-labeled arm (4/5) wins outright (for which prior-bins/
  temperature have no defined meaning). This is a judgment call, not a plan quote — confirm or
  override before the real run if a different knob-grid interpretation is intended.
