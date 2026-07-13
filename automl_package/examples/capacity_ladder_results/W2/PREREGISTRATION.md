# W2 pre-registration — per-input WIDTH dial on the heterogeneous (line + sine) toy

(`docs/plans/width_dial_2026-07-11/EXECUTION_PLAN.md` §6; driver `hetero_width_experiment.py`, net
`nested_width_net.py`. Written BEFORE the real 3-seed run — this file records the bars as designed,
not as observed. A "known risk" section at the bottom documents what the required `--smoke` check
(EXECUTION_PLAN.md §Phase B verify) already showed, exactly as W1's own pre-registration carried
forward a known risk from its Step-0 probe.)

## Toy and metric

Heterogeneous toy (`nested_width_net.make_hetero`): `x ~ Uniform(-r, r)`, `r = 4*pi`. Easy region
(`x < 0`): `y = (0.5/r) * x` (a straight line). Hard region (`x >= 0`): `y = 0.5 * sin(x)` (2
native-frequency periods over `[0, r]`). Gaussian noise `sigma = 0.05` added to `y`; both branches
are `0` at `x = 0`, so the noise-free signal is continuous there. Region split: hard = `x >= 0`,
easy = `x < 0` (`make_hetero`'s own `region` return, `1` = hard, `0` = easy).

W2 replaces W1's sinc toy because the Step-0 probe for sinc showed no genuinely width-flat region
(`EXECUTION_PLAN.md` §6: W1's "tail" gained nearly as much LL from width as its "centre" did). The
W2 toy was PROBED learnable with dedicated (non-shared) fixed-width nets before this build
(`scratchpad/hetero_toy_probe_v2.py`, cited in the plan): easy flat ~1.2-2x noise floor at every
width; hard 52x(w=1) -> 14x(w=4) -> 3.8x(w=6) -> 1.8x(w=7) -> 1.3x(w=10) — a clean width gradient.

Strictly probabilistic throughout: every read is a per-example Gaussian log-likelihood
(`nested_width_net.gaussian_log_likelihood`), computed on the model's `(mean, log_var)` heads. No
MSE-only bar, no penalty/lambda, no tuned regularizer anywhere in phase 1 or phase 2.

Oracle constant (unchanged from W1 — same noise sigma): `ORACLE_LL = -log(0.05) - 0.5*log(2*pi) -
0.5 ~= 1.5768`.

## Config

| | |
|---|---|
| `W_max` | 12 (not W1's 16 — EXECUTION_PLAN.md §6 W2 toy spec) |
| seeds | 0, 1, 2 |
| depth | fixed, single hidden layer (width is the only capacity axis) |
| activation | tanh |
| `n_train` | 1500 (split 750/750 index-parity between phase 1 and phase 2, see below) |
| `n_test` | 500 |
| optimizer | Adam, lr = 1e-2 |
| phase-1 schedule | `nested_width_net.WidthSchedule.SANDWICH` (NOT W1's `NESTED`) |
| phase-1 epochs | 2500 (matches the Step-0 W2 probe's own convergence budget and W1's real run) |
| phase-2 (selector) epochs/lr | `capacity_ladder_k6.N_EPOCHS`/`LR` (300 / 1e-2), reused verbatim |
| `K_LO` | 1 |
| `K_MID` | `W_max // 2` = 6 |

## Phase-1 schedule — SANDWICH (the W2 fix over W1's NESTED draw)

W1 diagnosed a second failure mode independent of the toy's own lack of contrast: its NESTED
per-sample uniform width draw scores each training example at ONE width per epoch, so width=`w_max`
is trained on only `1/w_max` of epochs — under-fitting the hard region by 0.55 nat relative to a
dedicated net's ~0.09 nat gap at `w_max=16` (`EXECUTION_PLAN.md` §6).

SANDWICH (`nested_width_net.train_nested_width(..., schedule=WidthSchedule.SANDWICH)`) replaces
this: every training step ALWAYS full-batch-scores width=1 (min) AND width=`w_max` (max), PLUS 2
random intermediate widths drawn without replacement from `{2..w_max-1}`; the 4 per-width mean
Gaussian-NLL losses are summed and backpropagated in ONE `.backward()` + one optimizer step. This
guarantees width=`w_max` is trained on every single step, not `1/w_max` of them.

## Two-stage procedure (no leak)

Identical to W1's split shape. Phase 1 trains `NestedWidthNet` (single `Linear(1 -> W_max)` trunk,
tanh, `(mean, log_var)` readout heads) with the SANDWICH schedule on ONE HALF of the training set
(index-parity split, `p1_idx`). Phase 2 FREEZES the phase-1 net and distills a selector
(`capacity_ladder_k6._RouterMLP`, trained with `capacity_ladder_k6._train_router`'s SOFT objective
onto `capacity_ladder_k6._soft_targets`'s per-tercile EM-stacked responsibilities — both reused
verbatim via `sinc_width_experiment._fit_selector`) on the OTHER half of the training set
(`p2_idx`) — data the phase-1 net never trained on.

All three bars below are read on the untouched 500-point TEST set.

## Bar (i) — CONSTRUCTION (expectation FLIPPED vs W1)

W1's centre (oscillating region) climbed with width and its tail (flat ramp) was expected to stay
comparatively flat, but did not cleanly separate. W2's split is deliberately sharper: the EASY
region (`x < 0`) is a literal straight line — 1 node suffices, probed genuinely flat with a
dedicated net — while the HARD region (`x >= 0`) is the width-hungry sine. So for W2: **hard climbs
with width, easy stays flat** — the opposite labelling of W1's centre/tail, same bar shape.

Reusing `sinc_width_experiment._construction_bar` verbatim (region `1` -> its "centre" slot ->
here, HARD; region `0` -> its "tail" slot -> here, EASY): hard held-out LL must improve `K_LO ->
K_MID` (`1 -> 6`) by `> 2*SE` (plain bootstrap) AND the hard LL at `W_max` must reach within the
same noise-floor-multiple gap of `ORACLE_LL` as W1 used (`sinc_width_experiment.
NOISE_FLOOR_GAP_NAT`, `0.5*log(1.5) ~= 0.2027` nat — an effective residual variance 1.5x the true
noise floor). Easy held-out LL must be comparatively flat: no width past `K_LO` may beat it by `>
2*SE`.

**Pass condition:** `construction_pass` (hard climbs AND nears the floor AND easy flat) on `>= 2/3`
seeds. Positive control: confirms the ONE SANDWICH-trained nested net reproduces the per-input
width gradient the Step-0 probe found with dedicated fixed-width nets.

## Bar (ii) — RECOVERY

The distilled selector assigns a strictly larger expected width to HARD inputs than EASY inputs.
Expected width per input is `sum_k k * P(k | x)` from the selector's own full width distribution
(never a collapsed argmax). Separation is read as `mean(expected_width | hard) -
mean(expected_width | easy) > 2*SE`, `SE` from an unpaired (two-sample) bootstrap over the two
region groups (`sinc_width_experiment._recovery_bar`, reused verbatim).

**Pass condition:** separation `> 2*SE` on `>= 2/3` seeds.

## Bar (iii) — DEPLOY

The selector's per-input BLENDED held-out LL (`-mean_i logsumexp_k(log w_k(x_i) + log
p_k(y_i|x_i))`) must (a) match-or-beat the best SINGLE global (fixed) width's held-out LL on the
same test set, one-sided, AND (b) land within 0.02 nat of the per-input ORACLE width's LL
(`sinc_width_experiment._deploy_bar`, reused verbatim).

**Pass condition:** both (a) and (b) hold on `>= 2/3` seeds.

## Verdict rule

`FOUND_LEARNABLE_WIDTH_CONTROL` if bars (i), (ii), and (iii) each pass on `>= 2/3` seeds; otherwise
the summary reports the first bar that failed and how many seeds passed it
(`hetero_width_experiment.py`'s `main()`, identical shape to W1's).

## Known risk (from the required `--smoke` check — reported here, not silently patched)

The required `--smoke` run (`w_max=6`, 600 phase-1 epochs, seed 0, SANDWICH schedule) shows the
SANDWICH fix does NOT resolve W1's hard-region under-fit on this toy: hard held-out LL at `w_max=6`
was `-0.464` nat, a `2.04` nat gap to `ORACLE_LL` (`1.577`) — nowhere near the `<=0.2027` nat
noise-floor bar, and barely better than `k=1`'s `-0.550`. A deeper epoch/learning-rate sweep (not
part of this driver, kept in the session scratchpad) shows this is a genuine plateau, not slow
convergence: at `w_max=12`, hard LL is essentially flat from step ~1000 through step 20000 (~-0.39
to -0.42 nat) across three learning rates (1e-3, 3e-3, 5e-3, 1e-2), while a FIXED width-`w_max`-only
net (no other widths mixed into training) reaches `1.55` nat (`0.03` nat gap) in the same 2500
epochs — so the architecture and the Gaussian-NLL objective are not themselves the obstruction; the
obstruction is specific to jointly training width=1 (and other low widths) through the SAME shared
trunk as width=`w_max`. The smoke run's easy region is also NOT flat (contrary to its
dedicated-net-probed expectation): it gains `0.41` nat from `k=1` to `k=6`, more than the hard
region's total climb. Both are genuine per-input-width-selector-relevant findings to weigh before
committing to the real 3-seed run, not implementation bugs in this driver (`nested_width_net.py`
selftest and this driver's own selftest both pass; the fixed-width-only control converges cleanly).
