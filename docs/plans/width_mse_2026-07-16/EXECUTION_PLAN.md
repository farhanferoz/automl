# Variable width per input, MSE-only — ⛔ FROZEN / COMPLETE. NEVER DISPATCH FROM THIS FILE.

> **FROZEN 2026-07-20. COMPLETE (WP-0..5).** Live plan of record:
> `docs/plans/capacity_programme/` — the width verdict it produced is carried by
> `docs/plans/capacity_programme/width-cert.md`, and its §5 bars are re-used by reference across
> that programme. This file is width-only and MSE-only; its one ProbReg line (§8 parked ledger) was
> **wrong and is corrected in place** — see that row.

Plan of record for the next phase of the width program. Supersedes
`docs/plans/width_dial_2026-07-11/` for forward work (those docs remain the historical record of the
NLL-based runs). Written to be orchestrated by a fresh session: every work package carries its own
input contract (files, spec, verification command, non-goals).

---

## 0. Charter — one question, definitively

**Can a single nested network serve per-input variable width?** Yes or no, with evidence that survives
adversarial reading. Nothing else is in scope.

Two scoping decisions, made 2026-07-16, that define this program:

1. **No variance fitting anywhere.** All training is plain MSE on the predicted value; all scoring is
   squared error. The variance/uncertainty question is PARKED (see §8) — including its interaction
   with variable structure, which is a later program of its own. Rationale: fitting the variance and
   fitting the capacity profile are two quantities that in-sample data cannot teach; entangling them
   contaminated the previous readouts (impossible per-point oracle; NLL training pathology in the
   cascade). MSE isolates the architecture question.
2. **One problem at a time.** Width now; depth next; width+depth together after that; variance
   simultaneously with structure last. Roadmap in §9.

"Definitive" means: the verdict comes from the decision tree in §6, whose branches were written before
the runs.

## 1. Corrected evidence base — what the prior runs actually established

⚠️ **A material record correction found 2026-07-16 while preparing this plan.** The 2026-07-13
synthesis (`docs/width_dial_synthesis_2026-07-13/per_input_width_architecture_readthrough.md`) headline
— "the cheap 1× shared net is the winner; shared-nested works when converged" — is **not supported by
the artifact it cites**. `W_KDROPOUT_CONVERGED` was produced by
`automl_package/examples/kdropout_converged_width_experiment.py`, which instantiates
**`IndependentWidthNet`** (line 164) — K disjoint sub-nets, K× parameters, **no weight sharing, no
prefix property** (class built to "deliberately break the shared-trunk prefix property",
`automl_package/examples/nested_width_net.py:160`). Its per-width best-state restoration
(`net.subnets[k-1]`) is only possible *because* the weights are disjoint. The true shared/nested
architecture (`NestedWidthNet`, 1× params) has **never been trained to convergence**.

Verified arm matrix (every row checked against source + summary JSON on 2026-07-16):

| Arm (result dir) | Driver | Architecture | Params | Budget | Hard fit @12 | Dial |
| --- | --- | --- | ---: | --- | --- | --- |
| `W2` | `hetero_width_experiment.py` | `NestedWidthNet` (true nesting) | 1× | 2.5k ep — under-trained | fails (~2 nat short) | — |
| `W_INDEP` | `independent_width_experiment.py` | `IndependentWidthNet` | K× | 2.5k ep | mixed (1.37/1.55/1.51) | 3/3 direction |
| `W_CONVERGED` | `converged_width_experiment.py` | `IndependentWidthNet` | K× | per-width, converged | floor 3/3 | 3/3, ≥2·SE |
| `W_KDROPOUT_CONVERGED` | `kdropout_converged_width_experiment.py` | **`IndependentWidthNet`** | K× | joint sandwich, converged | floor 3/3 | 3/3, ≥2·SE |
| `W_MRL` | matryoshka driver | shared trunk + per-width heads | ~1× trunk + K heads | 120k | floor 3/3 | mixed (1 clean / 1 inverted) |
| `W_CASCADE` | cascade driver | additive frozen blocks | ~1× | staged | fails (~1.7 nat short) | inverted 0/3 |

What therefore stands:

- **The per-input width signal is real and recoverable** (6/6 dial seeds across the two converged
  K×-parameter arms; separations ≥2·SE) — but only proven WITHOUT weight sharing.
- **The joint sandwich training *schedule* is not the obstruction** — with independent weights it
  converges to the same result as per-width separate training.
- **A shared trunk can reach the fit floor** when every width has its own readout head (Matryoshka),
  but its dial was seed-inconsistent (inversion diagnosed: per-width curves scrambled).
- **The staged/frozen cascade is refuted** (do not re-run).
- **OPEN — the charter question:** whether the strict nested architecture (shared trunk AND shared
  readout, 1× params, prefix truncation) can do it. `W2`'s failure conflates architecture with budget
  (2.5k epochs); at the same short budget the independent arm mostly passed, which is *suggestive* that
  architecture matters, but the converged cell is empty. **WP-2 fills it. That run — not a
  reproduction — is the definitive experiment.**
- Also unsupported until re-measured: the "payoff = compute" claim (the deployed selector evaluated so
  far is a soft blend that executes *every* width) and the per-point test-set "oracle" (selection on
  the evaluation draw; it scored 0.14–0.19 nat above the analytic ceiling, which is impossible).

## 2. Binding doctrine (applies to every WP)

1. **Data roles are disjoint.** Slice A trains networks; slice B feeds every learned readout (selector
   targets); the test set is touched exactly once, by the final scoring of a frozen pipeline. Nothing
   is ever scored on data that anything upstream of it was fit on. No per-point best-width selection
   on the test set, ever — the yardsticks are the analytic noise floor and the best single fixed width.
2. **Convergence gating, per width.** No conclusion from a fixed epoch budget. Per-width held-out-loss
   trajectories with `ConvergenceTracker` (`automl_package/examples/convergence.py`); a width that
   hasn't flattened is flagged and named in the summary; a seed whose *bar-driving* widths (1, 12) are
   unconverged is quarantined, not counted.
3. **Curve-shape gate before reading any dial.** Per seed, BEFORE the selector result counts: hard
   region's per-width MSE decreasing to a plateau; easy region flat from small k (§5 formulas). A
   scrambled curve is a training failure — quarantine the seed (the Matryoshka-inversion lesson: the
   selector faithfully reads whatever the curves say).
4. **No tuned dials.** Every constant in §5 is pre-registered here. Exactly one recalibration pass is
   allowed, after the WP-2 seed-0 pilot and before the batteries, then frozen; any later change
   invalidates the affected runs.
5. **Environment.** `AUTOML_DEVICE=cpu` on every run (XPU occupied). Heavy runs launched detached:
   `setsid nohup systemd-inhibit --what=idle:sleep:handle-lid-switch env AUTOML_DEVICE=cpu
   OMP_NUM_THREADS=<n> ~/dev/.venv/bin/python -u <script> ... > <scratchpad>/<log> 2>&1 &`.
   Monitor from the orchestrator's main thread by watching for the summary FILE (never a subagent;
   pgrep may match the systemd-inhibit wrapper). Shared 22-core box: ≤4 concurrent heavy runs.
6. **Worker briefs** carry the two standing clauses (land findings to disk immediately; do only what
   the contract names) plus the exact verification command. Workers return findings; the orchestrator
   writes shared artifacts. Verify against disk, never against a worker's claim.
7. **Commits are user-gated.** Stage nothing without an explicit go. No AI/tool provenance in any
   committed artifact.

## 3. Machinery inventory (reuse; do not reinvent)

| Piece | Where | State |
| --- | --- | --- |
| Toy generator `make_hetero` (easy-linear + hard-sine, σ=0.05) | `automl_package/examples/nested_width_net.py:216` | reuse as-is |
| `NestedWidthNet` (true nesting: shared trunk + shared heads, prefix masking) | `nested_width_net.py:85` | reuse; MSE path ignores `logvar_head` |
| `IndependentWidthNet` (K disjoint sub-nets; positive control) | `nested_width_net.py:160` | reuse as-is |
| Sandwich k-dropout convergence-gated training loop | `kdropout_converged_width_experiment.py:_train_kdropout_to_convergence` | adapt: loss flag + shared-arch checkpointing (WP-1) |
| Per-width `ConvergenceTracker` | `automl_package/examples/convergence.py` | reuse as-is |
| Split layout (A/B halves, val carve-out) | `converged_width_experiment.py:run_case` (N_TRAIN=1500, N_TEST=500) | reuse pattern |
| Scoring / bars / router | `sinc_width_experiment.py` (`_score_all_widths`, `_construction_bar`, `_recovery_bar`, `_deploy_bar`), router from `capacity_ladder_k6.py` (`_RouterMLP`, `_train_router`, `_soft_targets`) | adapt to MSE (WP-1); router net reused verbatim |

Anything not in this table that a worker thinks it needs: search first, state the search in the report.

## 4. Work packages

Dependency shape: WP-0 independent (docs only, any time). WP-1 → WP-2 → {WP-3, WP-4 in parallel} →
WP-5.

### WP-0 — Correct the written record (docs only; no code)

**Files:** `docs/width_dial_synthesis_2026-07-13/per_input_width_architecture_readthrough.md`; nothing
else (pre-run plan docs stay as history).

**Spec:** (a) Correct §3.3/§5.3/§6/§7: `W_KDROPOUT_CONVERGED` trained `IndependentWidthNet` (K×
params, no prefix property); retitle the finding to "the joint sandwich *schedule* works at
convergence (independent weights)"; mark "shared-nested works when converged" and "1× params winner"
as UNVERIFIED, superseded by this plan's WP-2; insert the §1 arm matrix. (b) Fix the oracle-bar
discussion (§1.3, §6): the per-point test-set oracle is not "too tight" — it is unreachable by
construction (selection on the evaluation draw with fitted variances; it exceeded the analytic
ceiling); its 0.02-nat sub-clause is void. (c) Note the deploy caveat: the evaluated selector was a
soft blend executing all widths; the compute claim awaits WP-2's hard-pick measurement.

**Verify:** grep the doc for "1× params — the winner" returns nothing; the matrix and both corrections
present; no other section rewritten.

**Non-goals:** do not alter per-seed numbers or the appendix; do not touch RESUME (orchestrator-owned).

### WP-1 — MSE harness (single worker; small, surgical deltas)

**Files:** `automl_package/examples/kdropout_converged_width_experiment.py` (extend),
`automl_package/examples/nested_width_net.py` (one helper), `automl_package/examples/sinc_width_experiment.py`
(MSE variants of bars — add functions, change none).

**Spec:**
1. `--arch {nested,independent}` flag: instantiate `NestedWidthNet` or `IndependentWidthNet`. Both
   already expose the same `forward_width`/`all_widths_forward` interface.
2. `--loss {nll,mse}` flag; `mse` trains on mean squared error of the mean output (standardized y);
   `logvar_head` untouched/unused. Helper `_width_mse(net, k, x, y)` beside the existing `_width_nll`.
3. **Shared-arch checkpointing:** per-width best-state restoration is impossible with shared weights
   (restoring width-3's best would clobber width-7's). For `--arch nested`: keep per-width
   `ConvergenceTracker`s for trustworthiness flags and the stop rule (all widths flat OR cap), but
   checkpoint the WHOLE net at the epoch with best MEAN per-width validation MSE, and restore that
   single state at the end. Record `best_mean_val_epoch` in the summary.
4. **MSE score table:** per point i, width k: `err2[i,k] = (y_i − μ_k(x_i))²` on slice B and on test
   (unstandardized y-units; record the standardization so the floor formula in §5 is computable).
5. **Selector targets (pre-registered primary):** cheapest-within-tolerance — per slice-B point,
   target width = smallest k with `err2[i,k] ≤ (1+δ_tie)·min_j err2[i,j]`, `δ_tie = 0.25`; train
   `_RouterMLP` with cross-entropy on these hard targets (reuse `_train_router` mechanics).
   **Sensitivity arm** (report only): soft targets `softmax_k(−err2[i,k]/(2·s²_B))` with
   `s²_B = median_i min_k err2[i,k]` — one global scale, computed, not tuned.
6. **Deploy metrics, hard-pick primary:** route each test point to its argmax width, run only that
   prefix; report `mse_hardpick`, `mean_executed_width`, `mse_best_fixed` + `best_fixed_k`, and the
   soft-blend MSE as a secondary line (labeled "executes all widths").
7. Summary JSON mirrors the existing schema (config incl. arch/loss/`n_train`/σ; per-seed convergence,
   curves, bars of §5, deploy). `--selftest` (tiny shapes, both arch × both loss) and `--smoke`
   (w_max=4, n=200, low cap) modes must pass.

**Verify (worker must run):**
`AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/kdropout_converged_width_experiment.py --selftest`
then `--smoke --arch nested --loss mse` and `--smoke --arch independent --loss mse`; sane JSONs on disk.

**Non-goals:** no new module unless the extension is genuinely infeasible (state why); no change to
NLL behavior (`--loss nll` must reproduce the old path bit-for-bit logic-wise); no Matryoshka/cascade
code; no variance anywhere.

### WP-2 — The definitive battery (constant-noise toy, N_TRAIN=1500)

**Arms:** **A = `--arch nested --loss mse`** (THE experiment — the empty cell);
**B = `--arch independent --loss mse`** (positive control, same protocol). Seeds 0/1/2 each,
`--max-epochs 300000`, convergence-gated.

**Order:** seed-0 pilot of BOTH arms first → one recalibration pass of §5 constants if needed (then
frozen) → remaining seeds, ≤4 concurrent (doctrine §2.5). Runtime is UNKNOWN for MSE (NLL runs
finished within a session on this box; assumption, not fact) — the pilot measures it; orchestrator
extrapolates before launching the batteries.

**Acceptance per arm:** fit bar + dial bar + curve gate + deploy metrics (§5) on ≥3 trustworthy seeds
(add seeds 3/4 if any seed is quarantined by the curve gate — quarantine is not failure, it just
doesn't count).

**Contingency C (run ONLY if arm A fails a bar on ≥2 trustworthy seeds while B passes):** the middle
rung — shared trunk + per-width linear heads (Matryoshka-lite readout on `NestedWidthNet.hidden`),
MSE, same protocol. Purpose: isolate WHERE strict nesting breaks (shared readout vs shared trunk).
This arm exists for the decision tree's "no — and here is why" branch; it is not a rescue.

**Non-goals:** no NLL arms; no cascade; no toy changes (that's WP-3).

### WP-3 — Discriminating control: the noisy-easy region

**Files:** `nested_width_net.py` — new generator `make_hetero3` (keep `make_hetero` untouched):
domain `[−4π, 8π)`, three equal-length regions — easy-linear (σ=0.05, as now, on `[−4π,0)`),
hard-sine (σ=0.05, as now, on `[0,4π)`), **noisy-easy-linear** on `[4π,8π)`: continuous at `4π`
(starts at `0.5·sin(4π)=0`), slope `0.5/(4π)`, **σ=0.5** (10× the quiet regions). Region labels
0/1/2. N scaled to keep per-region density equal to the 2-region toy (N_TRAIN=2250, N_TEST=750).

**Run:** the WP-2 winning arm (and arm B too if A won, as the control's control), 3 seeds, same
protocol.

**The point:** the dial must read *capacity-hunger, not raw error*. Noise level is common-mode across
widths at a fixed input, so the width curve in the noisy-easy region must be FLAT at a high level →
the honest verdict there is "stay narrow." A selector keyed to error magnitude will over-feed it.
Bars in §5 (noisy-easy clause). This is the width edition of the k-selection program's
smooth-negative-control lesson.

**Non-goals:** no variance modeling of the noisy region (we do not report calibrated uncertainty —
accepted cost of parking variance); no 4th region; no reweighting of the MSE loss.

### WP-4 — Data-size × noise ladder (where the payoff must live, if anywhere)

**Grid** on the 2-region toy, WP-2 winning arm: `n_train ∈ {200, 500, 1500, 4000}` ×
`σ ∈ {0.05, 0.15, 0.5}` × seeds {0,1,2} — 36 runs, but the σ ladder scales noise on BOTH regions
(uniform), so the floor formulas stay analytic. Convergence caps may be lowered for small n after the
pilot (state the cap per cell in the summary; the convergence rule still decides the stop).

**Readouts per cell:** dial separation (+SE), curve-gate pass rate, `mse_hardpick` vs
`mse_best_fixed` (+SE), `mean_executed_width` vs `best_fixed_k`.

**Pre-registered payoff hypothesis:** with scarce data no single global width is right everywhere
(full width overfits the easy region, small width underfits the hard one), so hard-pick routing beats
the best fixed width by ≥2·SE in at least 2 scarce cells (`n ≤ 500`). If instead the dial itself
degrades there, report the degradation frontier — "yes with a data threshold" is a legitimate branch
of §6, not a failure of the plan.

**Non-goals:** no heteroscedastic-noise cells (that's WP-3's separate job); no real datasets; no new
architectures.

### WP-5 — Verdict synthesis

Single writer (orchestrator or one worker). New doc
`docs/width_mse_2026-07-16/verdict_variable_width_mse.md`: the §6 decision-tree walk with per-seed
tables, the corrected claim ledger (what is now proven about nesting vs schedule vs readout), the
payoff statement (accuracy: from WP-4; compute: mean executed width, hard-pick), and the §9 roadmap
hand-off (what the depth program should copy). Update `RESUME.md` (verdict + pointers). Numbers only
from summary JSONs on disk; every table cites its file.

## 5. Pre-registered bars (MSE terms)

Let `s_y` = the training-half standardization scale stored in the run's `norm`; all `err2` reported in
raw y-units. Per-region analytic noise floor: `floor_R = σ_R²` (σ from the toy config; on the ladder,
the cell's σ). Region MSE at width k: `M_R(k) = mean_{i∈R} err2[i,k]` on the TEST set.

1. **Fit bar (hard region):** `M_hard(12) ≤ 1.25 · floor_hard` = pass; `≤ 1.10 ·` = strong pass.
2. **Curve-shape gate (per seed, on slice-B curves, BEFORE the dial is read):**
   hard: `M_hard(6) ≤ 0.5 · M_hard(1)` AND `M_hard(12) ≤ 1.2 · min_k M_hard(k)`;
   easy: `M_easy(2) ≤ 1.3 · M_easy(12)`. Fail → seed quarantined (training problem, not dial evidence).
3. **Dial bar:** mean expected/executed width difference `hard − easy > 0` and `> 2·SE`
   (two-sample bootstrap SE, reuse `_two_sample_boot_se`).
4. **Noisy-easy clause (WP-3 only):** `width(noisy-easy) ≤ width(easy) + 1.0` AND
   `width(hard) − width(noisy-easy) > 2·SE`.
5. **Deploy bar:** `mse_hardpick ≤ mse_best_fixed + 2·SE_paired` (accuracy preserved) AND
   `mean_executed_width < best_fixed_k` (compute saved). Report both components separately; the
   compute claim may only cite the hard-pick numbers.
6. **Trustworthiness:** widths 1 and 12 converged on every counted seed; unconverged middle rungs are
   named in the summary and may not drive any bar.

Constants {1.25, 1.10, 0.5, 1.2, 1.3, 1.0, δ_tie=0.25} are pre-registered; one recalibration allowed
after the WP-2 pilot, then frozen (doctrine §2.4).

## 6. Decision tree — the definitive answer

- **YES (nesting works):** Arm A passes fit + dial + curve gates on ≥3 trustworthy seeds, AND WP-3's
  noisy-easy clause passes, AND deploy preserves accuracy with lower executed width. State: "one 1×
  nested network serves per-input width; payoff = <accuracy claim from WP-4> + <compute numbers>."
- **YES, with a data threshold:** as above on data-rich cells, dial degrades below a frontier in WP-4
  → report the frontier as part of the affirmative verdict.
- **NO — nesting specifically fails:** A fails on ≥2 trustworthy seeds while B (independent) passes
  the identical protocol → run contingency C; report which property breaks (shared readout vs shared
  trunk). This is a clean, publishable negative: "per-input width needs per-width parameters."
- **NO — signal not recoverable under MSE:** both A and B fail the dial → contradicts the NLL-era 6/6
  result; halt and investigate the MSE readout (selector targets, δ_tie) before any verdict; escalate
  to the user with both scoring variants' numbers.
- **UNRESOLVED:** convergence caps exhausted with bar-driving widths still moving → raise caps once
  (×3) and re-run the affected seeds; if still moving, report honestly as compute-bound, no verdict.

## 7. Orchestration notes

- Waves: WP-0 ∥ WP-1 → WP-2 pilot → (recalibrate once) → WP-2 batteries → WP-3 ∥ WP-4 → WP-5.
- WP-0 and WP-1 have disjoint write sets and can be two concurrent workers (task-worker tier). WP-2..4
  are compute, not authorship: launch detached per doctrine §2.5, watch the summary files from the
  main thread. WP-5 is single-writer.
- Before re-running anything after a dead/idle worker: `pgrep -af <script>` and kill orphans first —
  two writers on one summary file has happened before.
- Budget: WP-1 is small (two flags + helpers + bars). The batteries are CPU-hours, not tokens; the
  pilot decides the wall-clock plan. State the estimate before launching each battery.

## 8. Parked ledger (deliberately NOT addressed here)

| Item | Why parked | Pointer |
| --- | --- | --- |
| Variance/uncertainty fitting, incl. simultaneously with variable structure | Charter decision 2026-07-16 — after width (and depth) | this doc §0 |
| Failing test `test_probabilistic_nll_beats_constant_on_heteroscedastic` (ProbReg 1.843 vs constant-σ 1.688) | ~~Variance-fitting bug by hypothesis (joint-μσ pathology vs under-training — undiagnosed)~~ **CORRECTED 2026-07-20: it was ALREADY DIAGNOSED the same day this line was written.** Root cause is config/protocol (a single coarse k=5 on a single seed), NOT a variance pathology — the variance head was shown well-behaved. Full diagnosis: `docs/plans/capacity_programme/shared/hetero_nll_diagnosis.md` | `tests/test_phase1_probabilistic_regression.py:243`; repro: run that test on HEAD |
| Heteroscedastic *scoring* experiments (calibration, NLL bars) | Need variance; note WP-3 keeps heteroscedastic-*data* dial tests in scope | §WP-3 |
| ProbReg / real-model ports of the dial | After the toy verdict | `docs/research_plan.md` |
| Hardcoded `/home/ff235` paths in 4 launcher scripts | Hygiene, orthogonal; do opportunistically | RESUME ### Next |

## 9. Roadmap after this program (recorded, not actioned)

1. **Depth:** transfer the validated machinery (nested training scheme + convergence gating +
   cheapest-sufficient selector + curve-shape gate) to depth. Known blocker from the closed depth
   lane: constructing a *depth-hungry but GD-learnable* toy (the tent-map target was representable but
   unlearnable at every depth) — the depth program starts with that construction, not with
   architecture.
2. **Width + depth jointly** (2-D capacity dial).
3. **Variance simultaneously with variable structure** — the parked interaction, incl. the failing
   test and the data-role chain (fit → error readout → selector → judge) designed 2026-07-16.
