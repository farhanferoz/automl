# PS — the ProbReg STRUCTURE phase (Problem 1: what IS the model)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or
> superpowers:executing-plans. This plan is EXECUTION-LEVEL: every arm, toy, k, seed, threshold,
> flag name, and decision rule is locked HERE. **An executor that finds itself making a judgment
> call is off-plan — stop and re-read the locked rule instead.**

**Status: RATIFIED for autonomous execution (user, 2026-07-21, live).** The user approved the
staged shape and instructed: (a) lock all specifics in the plan, (b) run the phase autonomously,
(c) **the only user gate is at the END (PS-4's certification memo)** — never mid-phase.

---

## 0. EXECUTION STATE — the root updates this IN PLACE after every wave

**This block is the phase's authoritative progress record.** It is updated in the same action that
completes a wave, before anything else is dispatched. `.superpowers/sdd/progress.md` mirrors it;
if the two ever disagree, `git log` settles it. A resuming session reads THIS section first and
resumes at the first wave not marked ✅ — never re-dispatches a ✅ wave.

| wave | task(s) | status | evidence |
|---|---|---|---|
| **0** | wave-1 closeout (root) | ✅ **complete 2026-07-21** | `ca18d59` test/example repair (suite 426P/2F/1S, the 2 = pre-accepted pair) · `9a80021` schedule refusal guard + 3 mislabelled `nested` cells voided (proven: differed from certified only in `provenance`+`training_schedule`) · `7ec5a58` this plan, gates 9/9 |
| **0.5** | merge `capacity/wave-1` → master | ⏳ **DEFERRED — blocked by Wave A's uncommitted tree edits.** Merge only after PS-A1+PS-A2 commit. Width WSEL-14/15 are blocked on this. | — |
| **A** | PS-A1 ∥ PS-A2 | 🔄 **PS-A1 ✅ complete** · PS-A2 authoring (driver on disk, agent live) | **PS-A1 = `25fcf9a`, review clean (spec ✅ / quality approved, 0 Critical, 0 Important).** Own tests 5/5; the two most-exposed suites 101 passed with only the pre-accepted heteroscedastic failure, independently re-verified by the reviewer at a worktree of base `91d6525` as pre-existing. **Root hardening, reviewed unprivileged:** `head_spread` lacked the string coercion every other enum kwarg in that `__init__` has, so a string would have matched none of the `is` branches and silently behaved as `PER_INPUT` — the same silent-fall-through class as the voided width cells. Now coerces and refuses non-members. 2 Minor findings logged to the ledger for the final review. |
| **B** | PS-1 trunk grid | ⬜ | — |
| **C** | PS-2 patch audit | ⬜ | — |
| **D** | PS-3 head battery | ⬜ | — |
| **E** | PS-4 certification → ⛔ user gate | ⬜ | — |

**Root-applied decisions during the autonomous run** (batched to the PS-4 memo; none irreversible):
- **D-PS-1 — §4.5 D2's metric was WRONG and is amended (not a default; a defect fix).** It named
  `heldout_nll_own`, which is not comparable across the likelihood families this battery varies by
  design, and does not exist at all for `fixed_shared` arms. The primary readout is now
  `heldout_fixed_sigma_mll` (higher is better), identically defined for every arm. Caught by the
  PS-A2 implementer, who flagged it rather than implementing a rule that would have ranked arms on
  which loss they trained. **Had this shipped, PS-1's winner would have been a metric artefact.**
- **D-PS-2 — D2 is read at k = 2 only** (truth-scale); k = 6 stays observational per §4.2. This was
  genuinely un-locked in the plan; the implementer's reading was correct and is now a hard rule.
- **D-PS-3 — `ProbabilisticRegressionModel.get_num_parameters()` fixed at source.** It called
  `self.parameters()` on a `BaseModel` wrapper that is not an `nn.Module`, so it raised
  `AttributeError` for every caller. Reproduced, then routed to the wrapped network with an
  explicit error when unbuilt. PS-3's parameter-matching depends on it.

---

**Goal:** settle what the ProbReg model structurally *is* — classifier supervision, spread
parameterisation, likelihood form, identifiability constraints, head layout — by pre-registered
experiment, and emit ONE certified configuration for the k-estimation programme to run on.

**Architecture:** one shared driver + two small model additions; four batteries (trunk → patches →
heads → certification), each decided by a mechanical rule computed by the driver itself
(`--decide`), never by an executor's judgment.

**Tech stack:** existing `automl_package` ProbReg machinery; no new frameworks.

---

## 1. THE WALL — Problem 1 vs Problem 2 (user, 2026-07-21: "clear separation")

- **Problem 1 (THIS phase):** what is the right structure for ProbReg.
- **Problem 2 (the rest of `probreg.md`, PAUSED):** given a structure, can k be estimated from
  data — one shared k or per-input — via nested training, the held-out arbiter, distillation.

**The wall, mechanically:**
1. **No cell in this phase ever chooses k from data.** k is a pinned experimental condition
   (§4.2). Any artifact produced by a k-selection code path is void here.
2. The phase exports exactly ONE artifact consumed downstream: the **pinned structure** — an
   exact constructor-kwargs dict plus the certification memo (PS-4).
3. Problem-2 tasks (P7, P8, P11, PB, PC, P3, P4, P5, P6) are ⛔ **PAUSED until PS-4's user gate**.
   P11 is **subsumed** by PS-3 (its parameter-matching method, mechanism control and halt rules
   carry over; its two *selection-mode reads* — global-k and per-input-k — move to Problem 2,
   where the selection machinery lives).
4. **Unaffected and NOT paused:** the width strand (`width.md` WSEL-14/15), wave-1 closeout
   (commit/merge + the ordering-driver schedule verification), `flexnn-package.md` bookkeeping.

**Decision-status changes this ratification makes (root applies to `MASTER.md` at execution
start, citing this file):**
- **Decision 26 (σ fixed in training) → SUSPENDED**: "fixed shared σ" becomes trunk ARM
  `fixed_shared` (§3), not a ruling. Decision 26's *selection* half (Decision 24) is untouched.
- **Decision 29's cross-entropy retirement → one arm re-admitted**: `CE_STOP_GRAD` runs as a
  labelled comparison arm through the existing escape hatch
  (`allow_retired_capacity_selection=True`, recorded per cell) — **no code change, no default
  change**. The other retired members stay retired.
- **Decision 24 (k scored on fixed-σ mixture likelihood, never a learned variance) → UNTOUCHED.**
  This phase compares structures; it never selects k, so no cell reads the forbidden metric as a
  k readout.

## 2. GLOBAL CONSTRAINTS — every cell, every task

- Interpreter: `~/dev/.venv/bin/python`. **`AUTOML_DEVICE=cpu` and `OMP_NUM_THREADS=4` on every
  run.** Grid concurrency ≤ 3 cells. Overnight grids launch with
  `setsid nohup systemd-inhibit --what=idle:sleep:handle-lid-switch ...` (GNOME idle-suspends AC
  at 900 s).
- **Workers AUTHOR; the ROOT runs every grid** backgrounded and verifies against disk. A worker
  never runs a grid, never runs `pytest tests/`, never commits.
- One JSON per cell, written the moment the cell finishes, under
  `automl_package/examples/capacity_ladder_results/PS1|PS2|PS3|PS4/`. Filenames:
  `ps<stage>_<arm_id>_<toy>_k<k>_seed<seed>.json`.
- Canonical toys only (`automl_package/examples/_toy_datasets.py`); the four legacy
  `make_datasets()` toys and the `make_v_toy*` family are 🚫 forbidden (probreg.md §0.5).
- Convergence gate: `--max-epochs 300 --check-every 10`, early-stop on held-out own-NLL with
  patience 8 checks and min-delta 1e-4; `hit_cap` recorded. Locked re-run rule: a cell with
  `hit_cap: true` is re-run ONCE at `--max-epochs 600`; if still capped it is flagged
  `converged: false` in the memo and never silently averaged in.
- Every cell records: full trajectory (§4.4), provenance, the resolved constructor kwargs
  actually used, and `allow_retired_capacity_selection` when set.
- Suite baseline: **2 known failures** (the pre-accepted heteroscedastic pair, MASTER Decision
  30). The root runs the suite once after Wave A lands; any OTHER failure blocks the phase.
- **Autonomous contract:** no mid-phase `AskUserQuestion`. Ambiguity → the locked rule; genuinely
  un-locked ambiguity → the reversible default, logged in `RESUME.md` `### Decisions`, batched
  into the PS-4 memo. A halt condition firing ⇒ end the phase early, write the memo with what was
  measured, stop. Irreversible/destructive actions: none exist in this phase by construction
  (nothing is deleted, no history rewritten, no push).

## 3. THE FACTOR LEDGER — locked inventory, admissions, deferrals

Every structural element on disk, verified this session (2026-07-21) by reading source, with its
locked disposition. <!-- numcheck-ignore: file:line anchors verified by reading, not run outputs -->

| # | element | where (verified) | default | locked disposition |
|---|---|---|---|---|
| 1 | classifier supervision (`optimization_strategy`) | `automl_package/models/probabilistic_regression.py:106` (default `REGRESSION_ONLY`), CE branch `:448-475` | `REGRESSION_ONLY`; CE modes retired `:223` | **TRUNK AXIS** `supervision ∈ {none, ce}`; `ce` = `CE_STOP_GRAD` via escape hatch |
| 2 | spread parameterisation | heads emit (mean, log-var) as a function of `p_i` — `automl_package/models/common/regression_heads.py:29` (`BaseRegressionHead`, `input_size=1` at `:337`) | per-input via head | **TRUNK AXIS** `spread ∈ {per_input, fixed_shared, all_constant}`; `all_constant` reuses `ConstantHead` `:211`. The per-class-scalar spread (the user's earlier σ-design proposal) was **REJECTED by the user at plan review ("per-class scalar goes", 2026-07-21)** — withdrawn, not an arm, recorded here so it is not re-proposed. |
| 3 | likelihood form | collapsed law-of-total-variance `regression_heads.py:511-513` + `nll_loss`; mixture `ProbRegLossType.MDN` `automl_package/enums.py:290`, branch `probabilistic_regression.py:418-433` | collapsed (`GAUSSIAN_LTV`) | **TRUNK AXIS** `likelihood ∈ {collapsed, mixture}`; fixed-σ mixture TRAINING branch is NEW (PS-A1) |
| 4 | ordering constraint | `automl_package/utils/ordering_loss.py`; auto-resolution `probabilistic_regression.py:258-264` | auto (ON for the sanctioned triple) | **PS-2 PATCH ARM** |
| 5 | anchored heads | `regression_heads.py:165` (`AnchoredHead`) | off | **PS-2 PATCH ARM** |
| 6 | monotonic head slopes | `use_monotonic_constraints`, SEPARATE_HEADS-only `probabilistic_regression.py:250-252` | off | **PS-2 PATCH ARM** (applicable only if winner spread = `per_input`; else recorded `inapplicable`, skipped) |
| 7 | middle-class constraint | `constrain_middle_class` → `ConstantHead`/`ProbabilisticMiddleClassHead` `regression_heads.py:240` | on | **PS-2 PATCH ARM** (flip-off) |
| 8 | boundary regularization | `BoundaryRegularizationMethod`, `penalties.py` | off | **PS-2 PATCH ARM** (`HARDSIGMOID` if winner spread ≠ `fixed_shared`, else `PENALTY` — the `PENALTY`×`PROBABILISTIC` pairing raises by design, `probabilistic_regression.py:162-165`) |
| 9 | head layout (`RegressionStrategy`) | three modules, `regression_heads.py:283,385,470` | `SEPARATE_HEADS` | **PS-3 AXIS** (P11's design carried over) |
| 10 | middle-class NLL penalty | `use_middle_class_nll_penalty` | off | **NOT AN ARM** — rides with #7's flip; kept off otherwise |
| 11 | β-NLL / symlog / MC-dropout / binned-residual | various | off | **OUT OF SCOPE** — orthogonal UQ conveniences, not structure; untouched |

**Admitted new elements (locked):**
- **`fixed_sigma_mixture` training loss** — required so `fixed_shared × mixture` is expressible;
  reuses `fixed_sigma_mixture_log_likelihood` (`automl_package/utils/losses.py:131`). Exact code
  in PS-A1.

**Deferred (PROPOSALS in the PS-4 memo — NOT arms; locking a novel architecture into an
autonomous run without user design review is how this becomes a mess):**
- Order-aware classifier target (cumulative-link / ordinal CE over the ordered slices; verified
  absent: `grep -rin "ordinal|cumulative.link|coral" automl_package/` → no hits).
- Head input representation (feeding features beyond `p_i` to heads).
- Real-data validation (Problem 2's P4 owns it).

## 4. READOUTS, σ POLICY, DECISION ALGORITHM — all locked

### 4.1 Toys and what each one decides
Screen set **T_SCREEN** = {`make_toy_b` (reference, known intrinsic k=2), `make_toy_a` (smooth
negative control), `make_broad_unimodal` (moment-matched twin of toy_b — THE mixture-vs-collapsed
discriminator), `make_toy_c_broad` (variance-trap: spread widens, k*(x)=1 — THE spread-axis
discriminator)}. Certification adds **T_FULL** = T_SCREEN ∪ {`make_toy_c`, `make_toy_e`,
`make_toy_d`} (fit-quality only; per-input k claims stay in Problem 2). Deviation from §0.5's
tier rule is deliberate and this paragraph is its written justification: this phase claims
nothing about the dial, so tier tables don't bind; toys are chosen per axis for what they
discriminate.

### 4.2 Pinned k
k ∈ {2, 6} in PS-1/2/3 (truth-scale and deliberate overcapacity); PS-4 adds k=4. k is a
condition, never a claim; the overcapacity cell exists to observe what surplus components DO
under each structure (annex data / stay empty / split noise), which the trajectory records.

### 4.3 σ policy (fixed-σ cells and the fixed-σ score column)
σ_toy = the toy's construction noise value, computed mechanically in the driver
(`_sigma_for_toy()`), asserted against the generator's constants in `--selftest`. For the
varying-noise toy (`make_toy_c_broad`): σ_toy = RMS over the training inputs of the generator's
true σ(x). Robustness re-score at σ_toy/2 and 2·σ_toy in PS-4 (re-scoring stored predictions —
no retraining). This implements the spec's mandatory half/double re-score, currently implemented
nowhere (carried gap, RESUME 2026-07-21).

### 4.4 Per-cell record (exact JSON keys, locked)
```
arm: {supervision, spread, likelihood, patches: {ordering, anchored, monotonic, middle_class,
      boundary}, layout}
toy, k, seed, sigma_toy, resolved_kwargs, allow_retired_recorded, provenance
trajectory[]: {epoch, train_nll_own, heldout_nll_own, per_class_sigma: {min, median, max},
              classifier_max_prob: {mean, max}, ordering_violations}
final: {heldout_nll_own, heldout_fixed_sigma_mll, rmse, coverage_1sigma, coverage_2sigma,
        min_sigma_ratio, slice_accuracy, params_count, hit_cap, converged}
```
`min_sigma_ratio` = min over classes of (median-over-batch fitted σ) / (true within-slice
residual σ of that class, computed from the generator). `slice_accuracy` = held-out fraction
where argmax class = true percentile slice. Under `fixed_shared`, `per_class_sigma` records
σ_toy for every class (the selftest asserts this).

### 4.5 The decision algorithm (mechanical; implemented in the driver's `--decide`)
- **D1 — degeneracy disqualifier.** An arm is DISQUALIFIED if, on any cell, `min_sigma_ratio`
  < 0.1 at any logged checkpoint after epoch 50 **and** the (heldout − train) own-NLL gap
  exceeds 0.5 nats with the gap monotonically increasing over the last 3 checkpoints.
- **D2 — win rule. ⚠️ AMENDED 2026-07-21 (root, during Wave A) — the original wording named the
  wrong metric and would have manufactured a false result.** It compared `heldout_nll_own` across
  arms, but this battery VARIES the likelihood family by design: a collapsed-Gaussian NLL, a
  mixture NLL and a means-only arm are not one scale, and `fixed_shared` arms have **no fitted
  predictive density at all**, so their own-NLL is undefined rather than merely different. Ranking
  on it would have scored arms partly on which loss they trained — the precise "metric artefact
  masquerading as a headline" failure `probreg.md` §0.5 exists to prevent.

  **PRIMARY READOUT = `heldout_fixed_sigma_mll`, HIGHER IS BETTER.** Arm A beats arm B iff A's
  seed-mean held-out fixed-σ mixture log-likelihood is **higher** by more than 2× the combined
  bootstrap SE (1000 point-resamples per seed of the per-point difference, RNG seed 12345;
  combined SE = `sqrt(mean(per_seed_SE**2)/n_seeds)`) on ≥ 2 toys of the battery, and is not worse
  beyond 2×SE on any toy. This metric is **identically defined for every arm** — one shared
  constant σ over the arm's own component means and weights — so no arm can win on variance
  machinery rather than structure. It is Decision 24's sanctioned readout, unchanged.
  **Comparison is read at k = 2** (the truth-scale condition); k = 6 is observational per §4.2 and
  may never be read as a win.

  **`heldout_nll_own` is a REPORTED COLUMN and a restricted tie-break, never the primary.** Fixed-σ
  scoring is deliberately blind to whether an arm's *spread* is any good, so an arm that learns a
  genuinely well-calibrated predictive density earns nothing for it under the primary. Own-NLL is
  where that can surface — but only **among arms for which it is defined** (those producing a
  proper normalised density over y; `fixed_shared` arms record `null` and are excluded from this
  comparison), and only when the primary cannot separate them. A tie-break on own-NLL must be
  recorded in the decision JSON as such, never silently folded into the primary.
- **D3 — identifiability disqualifier.** `ce` arms: DISQUALIFIED if final `slice_accuracy`
  < 1.5/k on any seed. `none` arms: DISQUALIFIED if final `ordering_violations` > 0 on any seed
  (labels have no stable meaning without it).
- **D4 — tie rule (simplicity order).** If D2 yields no winner: spread ties break toward
  `fixed_shared` > `per_input` > `all_constant` (fewer fitted objects win;
  `all_constant` last because it also freezes the mean); supervision ties break toward `ce`
  (pinned component semantics are what Problem 2's arbiter and distillation consume); likelihood
  ties break toward `mixture` (the collapsed form is its own moment-compression — user analysis,
  RESUME 2026-07-21); patch ties break toward OFF.
- The trunk winner = the arm surviving D1/D3 that D2-beats or D4-ties-over all other survivors.
  `--decide` emits the stage's decision JSON with the winner, every disqualification, and the
  D2 matrix. The root applies it verbatim.

---

## 5. TASKS

### PS-A1 — model additions (the ONLY model-code task of the phase)

**Files:**
- Modify: `automl_package/enums.py`
- Modify: `automl_package/models/common/regression_heads.py`
- Modify: `automl_package/models/architectures/probabilistic_regression_net.py`
- Modify: `automl_package/models/probabilistic_regression.py`
- Test: `tests/test_structure_phase_heads.py` (Create)

**Interfaces (Produces — PS-A2 codes against these exact names):**
- `HeadSpread` enum: `PER_INPUT = "per_input"`, `ALL_CONSTANT = "all_constant"`.
- `ProbRegLossType.FIXED_SIGMA_MIXTURE = "fixed_sigma_mixture"`.
- `ProbabilisticRegressionModel(head_spread=HeadSpread.PER_INPUT, fixed_sigma_train=None, ...)`.

**Orchestration:** parallel: yes (disjoint from PS-A2) · deps: none · tier: sonnet ·
scale: static · shape: execution · verify: `AUTOML_DEVICE=cpu OMP_NUM_THREADS=4
~/dev/.venv/bin/python -m pytest tests/test_structure_phase_heads.py -q` (all pass)

**Step 1 — enums.** Add to `automl_package/enums.py` (after `RegressionStrategy`):
```python
class HeadSpread(Enum):
    """How a SEPARATE_HEADS class's spread is parameterised (structure phase, structure.md PS-A1)."""

    PER_INPUT = "per_input"  # log-variance is a function of p_i (legacy default)
    ALL_CONSTANT = "all_constant"  # mean AND log-variance are per-class parameters (calibrated-constant baseline)
```
and extend `ProbRegLossType` with
`FIXED_SIGMA_MIXTURE = "fixed_sigma_mixture"  # mixture NLL at a required, shared, fixed sigma (structure.md)`.

**Step 2 — wiring.** `SeparateHeadsRegressionModule.__init__` gains
`head_spread: HeadSpread = HeadSpread.PER_INPUT`. Guard first:
```python
if head_spread is not HeadSpread.PER_INPUT and (use_anchored_heads or use_monotonic_constraints):
    raise ValueError("head_spread != PER_INPUT is incompatible with anchored/monotonic heads (structure.md PS-A1).")
```
In the per-class loop, BEFORE the existing branches:
```python
if head_spread is HeadSpread.ALL_CONSTANT:
    head = ConstantHead(uncertainty_method, regression_output_size)
    head.init_mean(centroid_i)
    self.heads.append(head)
    continue
```
Thread the kwarg: `ProbabilisticRegressionNet.__init__` passes `head_spread` into the
`SEPARATE_HEADS` construction (`probabilistic_regression_net.py:110-119`); model default params
gain `"head_spread": HeadSpread.PER_INPUT` and `"fixed_sigma_train": None`
(`probabilistic_regression.py:100-148` block).

**Step 3 — fixed-σ mixture training branch.** In `_calculate_loss`
(`probabilistic_regression.py`, alongside the MDN branch at `:418-433`):
```python
if self.prob_reg_loss_type == ProbRegLossType.FIXED_SIGMA_MIXTURE and per_head_outputs is not None:
    if self.fixed_sigma_train is None:
        raise ValueError("prob_reg_loss_type=FIXED_SIGMA_MIXTURE requires fixed_sigma_train (no default -- structure.md §4.3).")
    logits_for_mix = classifier_logits_out
    if self.optimization_strategy in (
        ProbabilisticRegressionOptimizationStrategy.CE_STOP_GRAD,
        ProbabilisticRegressionOptimizationStrategy.GRADIENT_STOP,
    ):
        logits_for_mix = logits_for_mix.detach()  # mirror the MDN back-door plug
    probs_for_mix = torch.softmax(logits_for_mix[:, : self.n_classes], dim=-1)
    mus = per_head_outputs[:, : self.n_classes, 0]
    regression_loss = -fixed_sigma_mixture_log_likelihood(y_true_squeezed, probs_for_mix, mus, sigma=self.fixed_sigma_train).mean()
```
(import from `automl_package.utils.losses`; validation of the `None` case ALSO at construction
time so misconfiguration fails fast, not at epoch 1).

**Step 4 — narrow the uncertainty force-override** (`probabilistic_regression.py:167-175`): the
current block silently flips `uncertainty_method → PROBABILISTIC` for any non-`REGRESSION_ONLY`
strategy; the CE branch (`:448-475`) reads logits and binned targets only and has no variance
dependence, so the flip is an over-restriction that would corrupt the `ce × fixed_shared`
trunk arms. Replace the override with the same `logger.warning` MINUS the mutation (delete the
`kwargs["uncertainty_method"] = ...` line; keep the warning text, reworded to say the combination
is now permitted for labelled comparison arms). Behavioural change is confined to escape-hatch
configurations; a test in Step 6 pins it.

**Step 5 — tests** (`tests/test_structure_phase_heads.py`, complete file; std toy tensors, no
training beyond 2 epochs; every test `AUTOML_DEVICE=cpu`):
1. `test_all_constant_ignores_input` — `ALL_CONSTANT` module: identical outputs for two
   different probability vectors.
2. `test_fixed_sigma_mixture_requires_sigma` — constructing with
   `prob_reg_loss_type=FIXED_SIGMA_MIXTURE, fixed_sigma_train=None` raises `ValueError`.
3. `test_ce_keeps_constant_uncertainty` — `CE_STOP_GRAD` (escape hatch on) +
   `uncertainty_method=CONSTANT` constructs WITHOUT the flip (asserts
   `model.uncertainty_method == UncertaintyMethod.CONSTANT`).
4. `test_head_spread_guard` — `ALL_CONSTANT` + `use_anchored_heads=True` raises.

**Non-goals:** no default behaviour changes for existing configurations (`PER_INPUT` path byte-
identical); no edits to selection/distillation code; no example scripts; no commit (root commits).

### PS-A2 — the battery driver

**Files:**
- Create: `automl_package/examples/probreg_structure_battery.py` (absorbs and REPLACES the
  uncommitted predecessor driver `probreg_variance_degeneracy_check.py` (in `automl_package/examples/`; DELETED by this task, so the path no longer resolves) — reuse its toy
  wiring, trajectory instrumentation, selftest patterns, and `_sigma_for_toy` logic, then delete
  the old file; it was never committed)

**Interfaces:**
- Consumes: PS-A1's `HeadSpread`, `ProbRegLossType.FIXED_SIGMA_MIXTURE`, `head_spread` /
  `fixed_sigma_train` kwargs (exact names above; authored in parallel against this contract).
- Produces: the per-cell JSON (§4.4 keys, verbatim), `--summarize` → a `frozen.json` per stage
  directory, `--decide --stage N` → a per-stage decision JSON in the same directory,
  implementing §4.5 verbatim.

**CLI (locked):** `--stage {1,2,3,4}` ·
`--supervision {none,ce}` · `--spread {per_input,fixed_shared,all_constant}` ·
`--likelihood {collapsed,mixture}` ·
`--ordering {auto,on,off}` `--anchored {on,off}` `--monotonic {on,off}`
`--middle-class {on,off}` `--boundary {off,hardsigmoid,penalty}` ·
`--layout {separate,single_n,single_final}` `--param-match {on,off}` ·
`--toy {toy_a,toy_b,broad_unimodal,toy_c_broad,toy_c,toy_e,toy_d}` · `--k INT` · `--seed INT` ·
`--max-epochs INT` `--check-every INT` · `--out DIR` · `--selftest` · `--summarize` · `--decide`
· `--rescore {half,double}` (PS-4 re-scoring of stored predictions; no retraining).

**Arm → constructor mapping (locked coherence matrix; the driver REFUSES anything else):**
<!-- numcheck-ignore: configuration matrix, not run output -->

| spread × likelihood | kwargs |
|---|---|
| `per_input × collapsed` | `uncertainty_method=PROBABILISTIC`, `head_spread=PER_INPUT`, `prob_reg_loss_type=GAUSSIAN_LTV` |
| `per_input × mixture` | … `prob_reg_loss_type=MDN` |
| `fixed_shared × collapsed` | `uncertainty_method=CONSTANT` (means only; loss = MSE on the mixture mean) |
| `fixed_shared × mixture` | `uncertainty_method=CONSTANT`, `prob_reg_loss_type=FIXED_SIGMA_MIXTURE`, `fixed_sigma_train=σ_toy` |
| `all_constant × mixture` | `head_spread=ALL_CONSTANT`, `MDN` (the calibrated-constant baseline) |
| `all_constant × collapsed` | 🚫 excluded (redundant with the per-class questions; keeps the grid bounded) |

`--supervision ce` adds `optimization_strategy=CE_STOP_GRAD, allow_retired_capacity_selection=True`
(recorded in the JSON). Trunk-stage patch state locked: `ordering=auto`, `middle-class=on`
(code defaults), `anchored/monotonic/boundary=off`. `RegressionStrategy=SEPARATE_HEADS` except
under `--layout`.

**Param-matching (PS-3, locked algorithm):** for each k, raise the single-head layouts'
`hidden_size` from 32 in integer steps until their parameter count ≥ 0.9× separate-heads' count
at that k; record `params_count` in every cell (realised, not intended).

**Orchestration:** parallel: yes (write set disjoint from PS-A1) · deps: interface-only on PS-A1
· tier: sonnet · scale: static · shape: execution · verify (root, after BOTH A-tasks land):
```bash
AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python -m automl_package.examples.probreg_structure_battery --selftest
~/dev/.venv/bin/python -m ruff check automl_package/examples/probreg_structure_battery.py
```
Selftest must additionally assert: σ_toy matches the generator constants for every toy; the
coherence matrix refuses `all_constant × collapsed`; `fixed_shared` cells report
`per_class_sigma == σ_toy` for every class.

**Non-goals:** never runs a grid itself; no model-file edits; no k-selection code path.

### PS-1 — the trunk battery (root-run)

**Cells:** 10 arms (5 spread×likelihood rows × 2 supervision) × T_SCREEN (4 toys) × k {2,6} ×
seeds {0,1} = **160**; then confirm: top-3 surviving trunks ∪ {the incumbent
`none × per_input × collapsed`} × T_SCREEN × k {2,6} × seed {2} = **≤ 32**.
<!-- numcheck-ignore: cell-count arithmetic, verified by the ls-count in verify -->

**Orchestration:** root only · deps: PS-A1 + PS-A2 merged · scale: static · verify:
```bash
[ "$(ls automl_package/examples/capacity_ladder_results/PS1/ps1_*.json | wc -l)" -ge 160 ] && echo CELLS-OK
AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python -m automl_package.examples.probreg_structure_battery --summarize --stage 1
AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python -m automl_package.examples.probreg_structure_battery --decide --stage 1
```
The decision JSON names the trunk. Root logs it in `RESUME.md` `### Decisions` and proceeds.
**This stage settles Decision 26's question and the cross-entropy question by measurement.**

### PS-2 — the patch audit (root-run)

**Arms (7, locked):** winner trunk W as-is · W with `ordering` flipped (off if auto-on, on if
auto-off) · W + `anchored=on` (skip+record `inapplicable` unless spread=`per_input`) ·
W + `monotonic=on` (same applicability rule) · W with `middle-class` flipped off ·
W + `boundary` (`hardsigmoid` if W's spread ≠ `fixed_shared`, else `penalty`) ·
W with ALL patches off.
**Cells:** ≤ 7 × 4 toys × k {2,6} × seeds {0,1,2} = **≤ 168**.
**Keep rule (locked):** a patch is kept iff it D2-beats W, or its removal breaks D3. The all-off
arm tying W ⇒ all patches dropped. Everything else: dropped, with its D2 matrix in the memo.
**verify:** same pattern as PS-1 with `PS2`/`--stage 2`.

### PS-3 — the head battery (root-run; subsumes P11)

**Arms:** {`separate`, `single_n`, `single_final`} at the PS-2 winner configuration,
parameter-matched; `single_final` runs its native no-component form and is labelled
**mechanism control** in every table (P11's rule). New head types: none (deferred, §3).
**Cells:** 3 × 4 toys × k {2,6} × seeds {0,1,2} = **72**.
**Halt rules (carried from P11, autonomous form):** layouts 1–2 indistinguishable by D2 → drop
`single_n` from the programme, continue. Mechanism control ties or wins at matched params → **end
the phase**, write the memo around that finding, propose nothing (it contradicts the component
story the strand rests on — a user matter, batched to the gate).
**verify:** same pattern, `PS3`/`--stage 3`.

### PS-4 — certification, memo, ⛔ USER GATE

**Cells:** final structure × T_FULL (7 toys) × k {2,4,6} × seeds {0,1,2} = **63**, plus
`--rescore half` and `--rescore double` over every fixed-σ-scored artifact (no retraining).
**Bars (locked):** no D1/D3 firing anywhere; `hit_cap: false` everywhere (after the one locked
re-run); the winner's PS-4 numbers within 2×SE of its PS-1/2/3 numbers (self-consistency).
**Memo:** `docs/plans/capacity_programme/shared/ps-certification-memo.md` — exact pinned
constructor kwargs · per-element evidence table (D2 matrices) · dropped elements with reasons ·
every logged reversible default · the deferred proposals (§3) · σ-robustness table ·
old-structure labelling rule for all prior ProbReg results.
**Then STOP.** Present the memo. Problem 2 unpauses only on the user's GO.

---

## 6. EXECUTION WAVES, WRITE SETS, COMMITS

- **Wave A:** PS-A1 ∥ PS-A2 (disjoint write sets; one dispatch message; both briefs carry the
  standing clauses, the interface block, and the verify commands). Root reviews both, runs the
  combined verify + the full suite once (expect exactly the 2 baseline failures + the new tests
  passing), commits each separately.
- **Waves B→E:** PS-1 → PS-2 → PS-3 → PS-4, strictly serial (each consumes the prior decision).
  All grids root-run, backgrounded, ≤3 concurrent, `systemd-inhibit`-wrapped overnight. Root
  commits each stage's results dir + decision JSON after its verify passes.
- **Write-set notes:** the driver file is written ONCE (PS-A2); stages differ only in flags, so
  no cross-wave file handoff exists. The session-scoped write guard is not triggered by design.
- Commit messages: plain conventional style, no tool provenance.

## 7. COST (stated before execution, per doctrine)

<!-- numcheck-ignore: pre-run estimates, not measured results -->
Authoring: 2 task-workers (Wave A) ≈ $5–10 total. Compute: ≤ 495 cells × ~2–4 min ≈ 17–33 CPU-h
at 3-way concurrency ≈ 6–11 h wall — one overnight, CPU only. Root review/decisions ≈ $10–20.
No further fan-out; no Workflow needed (grids are root-run Bash, decisions are driver-computed).

## 8. ARTIFACT ORGANISATION AND CLEANUP — binding on every task and every wave

**Why this is a section and not an afterthought (user instruction, 2026-07-21: "the plan is doing a
lot so organization is the key").** This phase creates ~495 result files, one driver, and four
result directories. The same programme has already been bitten twice by artifact disorder: six
width architectures across two directories needing a dedicated cleanup task, and three mislabelled
result cells that looked legitimate on disk. Cleanup is therefore a per-wave obligation, never a
task at the end.

**ONE driver, ONE home per stage.**
- `automl_package/examples/probreg_structure_battery.py` is the ONLY driver this phase creates.
  🚫 No per-stage drivers, no `_v2`, no forked copies. Stages differ by FLAGS only.
- Results live in exactly one place: `automl_package/examples/capacity_ladder_results/PS{1,2,3,4}/`.
  Naming is locked in §2 and is the only permitted form. A file that does not match its stage
  directory's pattern is a defect, not a variant.
- `probreg_variance_degeneracy_check.py` (the uncommitted predecessor) is **absorbed and DELETED**
  by PS-A2. Confirmed absent by the Wave A verify. It has no successor and must not reappear.

**Per-wave cleanup gate — the root runs this before marking any wave ✅ in §0:**
```bash
cd /home/ff235/dev/MLResearch/automl
git status --short          # nothing untracked except THIS stage's result JSONs
ls automl_package/examples/capacity_ladder_results/PS*/   # every file matches its locked pattern
~/dev/.venv/bin/python -m ruff check automl_package/
```
A wave is not complete while `git status --short` shows a file nobody intended. Stray scratch
files, half-written JSONs from a killed cell, and orphan `.pt` states are deleted at the gate, not
carried forward.

**Commit granularity (locked).** Each stage commits its results directory **together with** its
decision JSON, in one commit. A results set without the decision that reads it — or a decision
without its inputs — is unauditable. Every stage commit message names the cell count and the
decision outcome.

**Scratch stays out of the repo.** Briefs, worker reports, diff packages and any probe output live
under the session scratchpad. 🚫 Nothing temporary is written into the repo tree, per CLAUDE.md's
temp-file rule. Worker briefs explicitly forbid it.

**Model-side hygiene.** PS-A1 is the only task touching `automl_package/models/` or `enums.py`.
Default behaviour for every pre-existing configuration must remain byte-identical; the phase adds
capability, it never re-tunes the package. No retired mechanism is deleted by this phase.

**Handoff cleanliness (PS-4).** The certification memo names every artifact the phase produced and
its status: kept (evidence for the pinned structure), superseded (kept but labelled old-structure),
or deleted (with the reason). Nothing is left for a later reader to guess about.

## 9. NON-GOALS (phase-wide)

No k estimated from data · no real-data experiments · no deletion of any retired mechanism ·
no report writing beyond the memo · no changes to width/depth strands · no HPO (all
hyperparameters pinned by the driver's defaults, recorded per cell) · no new architecture beyond
PS-A1's additions (an enum, a loss branch, a guard fix — no new head class).
