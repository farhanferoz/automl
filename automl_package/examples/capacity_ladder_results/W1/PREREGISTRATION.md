# W1 pre-registration — per-input WIDTH dial on the ramped-sinc positive control

(`docs/plans/width_dial_2026-07-11/EXECUTION_PLAN.md` §2d; driver `sinc_width_experiment.py`, net
`nested_width_net.py`. Written BEFORE the real 3-seed run — this file records the bars as designed,
not as observed.)

## Toy and metric

Ramped sinc: `y = sin(x)/x + 0.04*x`, `x ~ Uniform(-5*pi, 5*pi)`, noise `sigma = 0.05`
(`sinc_width_experiment.ramped_sinc`/`make_data`, reused verbatim from the Step-0 probe,
`scratchpad/sinc_width_probe.py`). Region split: centre/hard = `|x| <= 2*pi` (the oscillating
part), tail/easy = `|x| > 2*pi` (the near-flat ramp) — `sinc_width_experiment.region`.

Strictly probabilistic throughout: every read is a per-example Gaussian log-likelihood
(`nested_width_net.gaussian_log_likelihood`), computed on the model's `(mean, log_var)` heads.
No MSE-only bar, no penalty/lambda, no tuned regularizer anywhere in phase 1 or phase 2.

Oracle constant (the best achievable LL if the model recovered the true mean function exactly and
scored under the true noise variance): `ORACLE_LL = -log(0.05) - 0.5*log(2*pi) - 0.5 ~= 1.5768`.

## Config

| | |
|---|---|
| `W_max` | 16 |
| seeds | 0, 1, 2 |
| depth | fixed, single hidden layer (width is the only capacity axis) |
| activation | tanh |
| `n_train` | 1500 (split 750/750 index-parity between phase 1 and phase 2, see below) |
| `n_test` | 500 |
| optimizer | Adam, lr = 1e-2 |
| phase-1 epochs | ~2500 (tuned to convergence in `--smoke`) |
| phase-2 (selector) epochs/lr | `capacity_ladder_k6.N_EPOCHS`/`LR` (300 / 1e-2), reused verbatim |
| `K_LO` | 1 |
| `K_MID` | `W_max // 2` = 8 (a judgment call — not numerically pinned by the plan; see "Judgment calls" below) |

## Two-stage procedure (no leak)

Phase 1 trains `NestedWidthNet` (single `Linear(1 -> W_max)` trunk, tanh, `(mean, log_var)`
readout heads) with the NESTED-width schedule — every training epoch, each example independently
draws its own width `k ~ Uniform{1..W_max}` and is scored only at that width; no selector exists
yet. Phase 1 trains on ONE HALF of the training set (index-parity split, `p1_idx`).

Phase 2 FREEZES the phase-1 net and distills a selector (`capacity_ladder_k6._RouterMLP`, trained
with `capacity_ladder_k6._train_router`'s SOFT objective onto `capacity_ladder_k6._soft_targets`'s
per-tercile EM-stacked responsibilities — both reused verbatim, not reimplemented) on the OTHER
half of the training set (`p2_idx`) — data the phase-1 net never trained on, so the distillation
target is never read off memorized predictions.

All three bars below are read on the untouched 500-point TEST set.

## Bar (i) — CONSTRUCTION

Centre held-out LL climbs with width (`K_LO -> K_MID`, gain `> 2*SE`, plain bootstrap) AND the
centre LL at `W_max` reaches within ~1.5x the noise floor of the oracle. Tail is comparatively
flat: no width past `K_LO` improves tail LL by `> 2*SE`.

"Within ~1.5x noise floor" is the Step-0 probe's own MSE-multiple language, translated to nats: an
effective residual variance `N` times the true noise floor gives an LL gap from `ORACLE_LL` of
EXACTLY `0.5*log(N)` (a Gaussian LL fact, not an approximation). `N = 1.5` => gap `<= 0.5*log(1.5)
~= 0.2027` nat (`sinc_width_experiment.NOISE_FLOOR_GAP_NAT`).

**Pass condition:** `construction_pass` (centre climbs AND nears the floor AND tail flat) on
`>= 2/3` seeds. Positive control: confirms the ONE nested net (trained under the per-sample width
schedule) reproduces the per-input width gradient the Step-0 probe found with dedicated
fixed-width nets.

**Known risk (carried forward from the plan, not new):** the Step-0 probe's own table shows tails
are "not perfectly flat" (`EXECUTION_PLAN.md` §1 caveat: tail MSE-multiple drops from 7.4x at
width 1 to 1.3x at width 16 — a real, if smaller, width-dependent gain, because a single shared
trunk allocates capacity across the whole domain). The `tail_flat` sub-check may legitimately fail
on the real run for this reason, independent of whether the centre signal itself is clean; the
plan's suggested fallback (tighten the region boundary) is out of scope for this pre-registration
and would be a separate, explicitly-approved change to `region()`.

## Bar (ii) — RECOVERY

The distilled selector assigns a strictly larger expected width to centre inputs than tail inputs.
Expected width per input is `sum_k k * P(k | x)` from the selector's own full width distribution
(never a collapsed argmax). Separation is read as `mean(expected_width | centre) -
mean(expected_width | tail) > 2*SE`, `SE` from an unpaired (two-sample) bootstrap over the two
region groups (`sinc_width_experiment._recovery_bar`).

**Pass condition:** separation `> 2*SE` on `>= 2/3` seeds. Confirms the selector correctly READS,
per input, how many hidden nodes that input needs — the learnable positive control the depth lane
never had.

## Bar (iii) — DEPLOY

The selector's per-input BLENDED held-out LL (`-mean_i logsumexp_k(log w_k(x_i) + log
p_k(y_i|x_i))`, mirroring `capacity_ladder_h1.py::_blended_nll`) must (a) match-or-beat the best
SINGLE global (fixed) width's held-out LL on the same test set, one-sided (`selector_NLL <=
best_global_NLL`), AND (b) land within 0.02 nat of the per-input ORACLE width's LL — the
`max_k score[i, k]` bound computed on the SAME held-out table (the standard "peek at y, pick the
best column" upper bound used throughout the capacity-ladder scripts, e.g.
`capacity_ladder_k6.py`'s `nll_oracle`).

**Pass condition:** both (a) and (b) hold on `>= 2/3` seeds. Mirrors H1's shipping bar — this is
the "does it deploy" question, not just "is the signal there."

## Verdict rule

`FOUND_LEARNABLE_WIDTH_CONTROL` if bars (i), (ii), and (iii) each pass on `>= 2/3` seeds; otherwise
the summary reports the first bar that failed and how many seeds passed it
(`sinc_width_experiment.py`'s `main()`).

## Judgment calls made while building the driver (flagged for review, not silently decided)

- `K_MID = W_max // 2` (8 at the real `W_max=16`): the plan specifies the bar structurally
  (`k_lo -> k_mid`) but does not pin `K_MID` numerically. Chosen to sit near the inflection point
  of the Step-0 probe's own width sweep (7-16 is where centre MSE-multiple flattens out).
- Phase-2 (selector) epoch count / learning rate: not specified by the plan for width; reused
  `capacity_ladder_k6.N_EPOCHS`/`LR` (300 / 1e-2) verbatim, mirroring `capacity_ladder_h1.py`'s own
  reuse of the same constants for its phase-2 gate distillation.
- Standardizing x and y for phase-1 training (then converting the net's `(mean, log_var)` back to
  the ORIGINAL y-scale via the standardization's exact affine Jacobian before scoring): not
  explicit in the plan, but the Step-0 probe itself needed this to converge (raw `x` spans
  `+-5*pi`, saturating `tanh` at typical init scales), and it is the smallest deviation that
  reproduces the probe's own result. The un-standardization is EXACT for a linear rescale, unlike
  the codebase's `symlog` uncertainty conversion which uses a linearized (approximate) Jacobian.
