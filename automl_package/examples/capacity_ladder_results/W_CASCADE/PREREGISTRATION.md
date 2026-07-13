# W_CASCADE / W_MRL pre-registration — frozen residual cascade vs Matryoshka width heads

(`docs/plans/width_dial_2026-07-11/cascade_execution_plan_2026-07-11.md` §4.4; drivers
`cascade_width_experiment.py` (both arms), nets `cascade_width_net.py`/`matryoshka_width_net.py`.
Written BEFORE the real 3-seed run — this file records the bars as designed, not as observed. Bars
and thresholds are IDENTICAL to the W batteries (`W_CONVERGED`, `W_KDROPOUT_CONVERGED`) —
comparability across training schemes is the point.)

## Toy, metric, and data/split/seeds

`nested_width_net.make_hetero` — flat-easy line spliced to a width-hungry-but-learnable sine
(easy region `x < 0`: straight line; hard region `x >= 0`: `0.5*sin(x)`), Gaussian noise
`sigma = 0.05`. Strictly probabilistic throughout: every read is a per-example Gaussian
log-likelihood (`nested_width_net.gaussian_log_likelihood`); held-out LL is the ONLY bar — no
MSE-only bar, no penalty/lambda, no tuned regularizer anywhere.

Data/split/seeds are identical to `converged_width_experiment.py` (reused verbatim via
`import converged_width_experiment as cwe`): `make_hetero(1500, seed)` train + `make_hetero(500,
seed+500)` test; seeds (0, 1, 2); index-parity p1/p2 split of train (p1 = phase-1 fit, p2 =
selector distillation); within p1, every 5th point is the convergence-monitoring validation split;
standardize x AND y on the p1-train stats only. `w_max = K = 12`. Adam, lr = 1e-2. Depth is fixed
at a single hidden layer everywhere — width/rungs is the only capacity axis.

## The two arms

- **Cascade** (`cascade_width_net.ResidualCascadeNet` + `train_cascade`): `w_max` frozen width-1
  tanh blocks, grown one at a time. Stage b zero-inits block b's readouts (so training starts
  EXACTLY at rung b-1's output), trains only block b against the cached frozen prefix
  (multi-restart, keep best-val), applies the acceptance rule (reset to inert if the stage did not
  beat rung b-1 by `convergence.DEFAULT_MIN_DELTA`), then freezes block b forever. Guarantees
  `NLL_val(rung k)` non-increasing in k by construction (plan §2.3 Lemma 2) — the coherence
  invariant (stable identity + self-contained + importance-ordered) in its guaranteed weak form.
- **Matryoshka** (`matryoshka_width_net.MatryoshkaWidthNet` + `train_matryoshka`, the FALLBACK
  arm): shared `Linear(1 -> w_max)` trunk, tanh, but each rung k has its OWN dedicated `(mean_head_
  k, logvar_head_k)` pair instead of one shared readout. Trained JOINTLY, one optimizer, loss =
  unweighted sum of all `w_max` rungs' mean NLL every step. Removes the shared-readout
  entanglement (invariant (2) partially fixed) but the shared trunk still has no stable per-rung
  identity or guaranteed ordering (invariants (1)/(3) NOT fixed) — the comparison to Cascade
  isolates whether frozen identity + guaranteed ordering matter beyond readout disentanglement
  alone.

Frozen-net scoring, the selector, and all three bars below are `sinc_width_experiment` reused
VERBATIM (`_score_all_widths`, `_construction_bar`, `_fit_selector`, `_selector_eval`,
`_recovery_bar`, `_deploy_bar`) — both arms expose the same `w_max`/`forward_width`/
`all_widths_forward` interface as `nested_width_net.NestedWidthNet`, so the scoring pipeline is a
drop-in, identical call shape to `converged_width_experiment.run_case`.

## Bar (i) — CONSTRUCTION

Hard-region (`region==1`, `x>=0`) held-out LL climbs `k_lo=1 -> k_mid=6` by `> 2*SE` (plain
bootstrap) AND reaches near the noise floor at `k=w_max`. Easy region (`region==0`, `x<0`) is
comparatively flat: no rung past `k_lo` improves easy LL by `> 2*SE`. Verbatim
`sinc_width_experiment._construction_bar`.

**Pass condition:** `construction_pass` on `>= 2/3` seeds.

## Bar (ii) — RECOVERY

The distilled selector assigns a strictly larger expected width to hard inputs than easy inputs.
Expected width per input is `sum_k k * P(k | x)` from the selector's own full width distribution.
Separation read as `mean(expected_width | hard) - mean(expected_width | easy) > 2*SE` (unpaired
bootstrap). Verbatim `sinc_width_experiment._recovery_bar`.

**Pass condition:** separation `> 2*SE` on `>= 2/3` seeds.

## Bar (iii) — DEPLOY

The selector's blended held-out LL must (a) match-or-beat the best single global rung's held-out LL
(one-sided) AND (b) land within 0.02 nat of the per-input oracle. Verbatim
`sinc_width_experiment._deploy_bar`.

**Pass condition:** both (a) and (b) hold on `>= 2/3` seeds. **Known standing miss:** every W
battery run so far fails the 0.02-nat oracle sub-clause (T1/T3/S1/etc. and every prior W run) — the
recalibration question this raises stays OPEN, a user decision. This pre-registration reports the
number; it does not move the bar to make the cascade or Matryoshka arm pass artificially.

## ANCHOR diagnostic (informative — not a pass/fail bar)

Rung-12 test NLL compared against the dedicated w12 anchor
(`capacity_ladder_results/W_CONVERGED/w_converged_summary.json`, same seed, `per_case[i]
["per_k_nll"]["12"]` — per-seed, nested; NOT a top-level key). Reported as `anchor = arm_rung12_NLL
- dedicated_w12_NLL` (positive = cascade/Matryoshka worse than the dedicated net at full capacity;
expected, since greedy stagewise/joint-shared training is a weaker optimizer than a dedicated net
trained from scratch at that one width — plan §2.4's "where the cascade is weaker").

**Escalation trigger:** if the CASCADE arm's rung-12 HARD-region held-out LL is more than 0.10 nat
below the dedicated w12 anchor (per seed), re-run the cascade arm with `--beta-nll` (Seitzer et al.
2022, beta=0.5, plan §2.5 — the stagewise sharpening-of-heteroscedastic-pathology mitigation,
pre-registered here, not improvised mid-run). The bars above are ALWAYS evaluated with plain
held-out LL regardless of which loss trained the model — the bar never moves.

## ORDERING diagnostics (mandatory outputs — "measure, don't assume")

- Per-rung delta-LL profile (`delta_ll_per_rung`: rung k vs rung k-1, rung 0 = the standardized
  N(0,1) marginal), overall + by region. This is the strong-ordering claim (decreasing marginal
  gain per rung) that Lemma 2 does NOT guarantee — it is measured, not assumed, exactly the
  discipline the low-rank ladder skipped.
- Count and positions of inert (not-accepted) rungs — cascade only.
- Cross-seed Spearman correlation of the delta-LL-by-rung profile — reported with no threshold
  (first measurement of this quantity for this program; sets the baseline).

## Convergence gate

No conclusion from any seed x arm whose rungs are not ALL `trustworthy` (cascade: every ACCEPTED
stage's winning restart must be `trustworthy`; Matryoshka: all `w_max` per-rung trackers must be
`trustworthy`). The only valid read on an untrustworthy seed is "needs more training" — raise
`--max-epochs` and rerun that seed. `all_widths_trustworthy` in each case dict is the gate.

## Decision rule (pre-registered)

If BOTH arms pass bars (i) and (ii): the arm of record is whichever has the better DEPLOY bar
held-out LL; ties (within 2*SE) are broken toward the CASCADE (guaranteed identity/ordering, and
`~6K` params vs Matryoshka's `~(4W+2) + sum_k 2(k+1)` — plan §2.8's accounting table). If only ONE
arm passes bars (i) and (ii), it is the arm of record. If NEITHER arm passes CONSTRUCTION, the
finding is: "per-rung heads and frozen additivity do NOT rescue shared-representation nesting" — a
real (negative) answer to the middle-ground question raised by W1/W2's shared-trunk failure — and
gets folded into the report as-is, not iterated on ad hoc.

## Judgment calls made while building the drivers (flagged for review, not silently decided)

- **Anchor-path correction:** an earlier draft of this plan stated the dedicated w12 anchor lives
  at `w_converged_summary.json`'s top-level `per_k_nll["12"]`. Verified on disk: that key does not
  exist at top level — `per_k_nll` is PER-SEED, nested inside `per_case[i]`. `_load_anchor_nll`
  implements the corrected (verified) path; missing file / no matching seed / absent "12" key all
  degrade gracefully to `anchor: null` with a printed warning, never a hard failure.
- **`freeze_blocks_below` call site:** the plan's train_cascade bullet literally reads
  `freeze_blocks_below(b+1)` after stage b. Read literally (0-indexed threshold semantics, matching
  the method's own docstring "`requires_grad_(False)` on all blocks index `< b`"), that call would
  freeze block index `b` — the NEXT stage's block, not yet trained — before it starts training,
  which breaks stage b+1 (its readouts, only zeroed in place rather than reinstantiated, would stay
  permanently frozen at zero). `train_cascade` calls `freeze_blocks_below(b)` instead — freezing
  exactly the blocks trained through stage b (indices `0..b-1`), matching §2.3's own invariant
  ("block b never changes after stage b") without touching the untrained next block. Verified this
  is load-bearing by construction/selftest, not by observing a failed real run.
- **`train_cascade`/`train_matryoshka` signatures:** the plan's given signatures omit `seed` (used
  by the per-restart re-init instruction) and `lr` (used by the Adam optimizer, doctrine-pinned to
  1e-2 but not threaded through as an explicit arg in the plan text). Both added as explicit
  keyword parameters (`lr` defaulting to 1e-2), matching `nested_width_net.train_nested_width`'s
  own existing convention of taking `lr`/`seed` explicitly rather than hardcoding them.
- **`train_cascade`'s return shape:** the plan states `-> dict[int, cvg.ConvergenceResult]`, but
  the same paragraph requires recording `accepted` and `val_nll_prev` per stage alongside the
  winning `ConvergenceResult` (consumed by the driver's `accepted_rungs` field). Implemented as
  `dict[int, dict]`, each value `{"conv": ConvergenceResult, "accepted": bool, "val_nll_prev":
  float}` — the smallest extension that satisfies both the literal per-rung convergence-result
  requirement and the explicit "record accepted + val_nll_prev" instruction.
- **DEFAULT_MAX_EPOCHS = 40000** (both arms): not pinned by the plan. Chosen to match
  `converged_width_experiment.py`'s own default cap, since neither arm trains widths on only a
  FRACTION of steps the way `kdropout_converged_width_experiment.py`'s sandwich schedule does
  (cascade trains one ~6-parameter block per stage on a cached prefix; Matryoshka trains every rung
  jointly on every step) — so the slower-converging 200000-epoch cap that sandwich k-dropout needed
  should not be required here. If a real run's rungs hit the cap without converging, raise
  `--max-epochs` per the standing convergence-gate rule, same as every other W battery.
