# ProbReg — global-k sub-plan

One k for the whole dataset, chosen on held-out data: cheaply, by reading one nested network at every
k (M1), or expensively, by training one model per k (M3, the reference). Definitions:
`docs/probreg/PLAN.md` §1. Tasks moved verbatim from the former <!-- citecheck-ignore -->
`docs/plans/capacity_programme/probreg.md` on 2026-10-21 (P8, P3, P5). <!-- citecheck-ignore -->

<!-- from docs/plans/capacity_programme/probreg.md lines 1109-1236 -->
### P8 — does explicit regularisation move the selected k? — ⛔ **REOPENED 2026-07-21. RESULTS DISCARDED — SELECTED ON A FORBIDDEN METRIC (Gaussian NLL). The verdict below is VOID.**

**Why it is void.** The selection was made on **Gaussian negative log-likelihood**, a variance-based
metric: `automl_package/examples/probreg_p8.py:145` builds `sel_nll_per_sample` from the predicted
mean AND predicted standard deviation, and `:180-182` feeds exactly that as the error table into
`cheapest_within_tolerance`. The σ-scope decision removed variance from this strand and replaced the
metric set with **point prediction, primary squared error** (P0b; `docs/probreg/shared/benchmark_spec.md`
§4 — "NLL, CRPS, PIT, calibration/ECE, PICP, Winkler all assume σ… replace with a point-prediction
set"). The run already computes `report_mse` (`:150`), so the sanctioned metric was in hand and was
not the one selected on. *(User ruling 2026-07-21: results produced in violation of a constraint are
DISCARDED, not reinterpreted.)*

**Status of the artifacts.** The per-λ/seed JSONs and `frozen.json` under
`automl_package/examples/capacity_ladder_results/P8/` **stay on disk as a record and may NOT be cited
as evidence.** Deletion is user-gated and not proposed.

**Downstream consequence — the block STAYS, and the reason changes.** The earlier state was "the check
ran and the selection MOVED, therefore block". The correct state now is "**the check has not validly
run**", and MASTER Decision 21 blocks a battery read until it has. So **P4 and P6 remain blocked** —
but nobody may now cite "regularisation moves k" as an established finding, because that finding was
produced on the wrong metric. **Both the block and the claim it rested on are open questions.**

⚠️ **THE REMEDY BELOW WAS ITSELF WRONG — CORRECTED 2026-07-21 (root), and this is the second time
this strand has been about to select on a metric that cannot see its own dial.** The text originally
read *"select on the point-prediction metric (squared error)"*. **Do NOT do that.** Per MASTER
Decision 24 and the corrected §0.5, squared error is *structurally blind to k* for symmetric targets:
the mixture mean is identical whether the model resolves one component or five, so the selection
curve flattens, cheapest-within-tolerance returns k=1 at every λ, and P8 reports a confident **"the
selected k does not move"** — a verdict manufactured by the metric, not measured from the data. **On
THIS task that failure is maximally deceptive**, because "does not move" is the outcome that
*unblocks* the battery. A blind metric would have cleared the block it exists to enforce.

**The diagnosis of what voided the original run stands unchanged** — it selected on a **learned**
variance. What changes is only the remedy: the fix is a likelihood at **FIXED σ**, not a retreat to
squared error. Both errors share one root cause, which is why §0.5 now states the test as *"is σ
learned?"* rather than *"is it a likelihood?"*.

**What the re-run must change (and ONLY this):** select on the **per-sample fixed-σ mixture
log-likelihood** (MASTER Decision 24), keeping the λ grid, seeds, splits, convergence gates and the
twice-bootstrap-SE selection rule byte-identical to the spec below. **⇒ P8 therefore deps on FP-12**
(`docs/flexnn/package/plan.md`), which builds that scorer — until it exists there is nothing compliant to
select on, and re-running P8 before it lands would void the result a third time. Recording squared
error alongside as a labelled point-accuracy column is fine;
selecting on it is not.

**⚠️ Same defect, same day, different strand — this is a PATTERN, not an incident.** The width
strand's equivalent check (`docs/flexnn/width/plan.md` WSEL-11) was reopened at the same time for the same class of
violation, with the OPPOSITE verdict. Two checks whose whole purpose was to protect batteries were
themselves run outside the constraint they protect. → MASTER Decision 21 amended.

*(Recorded verdict retained, struck, as the case law.)*
~~✅ RAN 2026-07-21. VERDICT: MOVES → ⛔ STRAND-LOCAL BLOCK on P4/P6.~~

**RESULT: `selection_moved: true`** — ledger `automl_package/examples/capacity_ladder_results/P8/frozen.json`, thirteen per-cell JSONs in the same directory (λ ∈ {0, 1e-4, 1e-2} × seeds {100, 101}, plus the driver's own epoch-raise and ceiling-raise re-runs). Toy: heteroscedastic. Selection rule:
`cheapest_within_tolerance` at twice a bootstrap SE (`automl_package/utils/capacity_selection.py`).

**Consequence (MASTER Decision 21, block semantics 2026-07-21): P4 and P6 MAY NOT BE READ** until the
confound is re-derived. Strand-local: `PB`/`PC`/`P3` are unaffected, the other strands continue, and
this is **batched for end-of-run user review**, never resolved by the run.

**⚠️ READ THE PATTERN BEFORE READING THE VERDICT — the two seeds behave completely differently, and
the movement is almost certainly INSTRUMENT INSTABILITY, not a clean regularisation effect** *(root
analysis, 2026-07-21)*:
- **Seed 101 is rock stable:** selects the same k at every λ — 5, 5, 5. It does not move at all.
- **Seed 100 moves, but NOT MONOTONICALLY in λ:** 10 → 8 → 12 as λ goes 0 → 1e-4 → 1e-2. More
  regularisation selecting a LARGER k is backwards for the mechanism under test; a genuine
  "regularisation lets you get away with less capacity" effect would push k DOWN, monotonically.
- **The decisive evidence that it is the instrument: the same cell flips selection under a longer
  EPOCH budget, with λ held fixed.** The driver's own convergence escalation (`epoch_raises` in
  `frozen.json`) re-ran cells that hit the epoch cap and recorded: at λ=0 seed 100, selected k went
  **8 → 10**; at λ=1e-4 seed 100, **12 → 8**. Weight decay was CONSTANT across each of those pairs.
  **A selection that moves when only the training budget changes cannot be attributed to weight
  decay.** Seed 101's selection survived every one of those same re-runs unchanged.

⇒ **The honest statement of this result: on this toy, k-selection is UNSTABLE on one of two seeds —
to the weight-decay grid, to the epoch budget, and non-monotonically — while the other seed is
perfectly stable. Decision 21's question ("does regularisation move the selected capacity?") is NOT
cleanly answered, because the instrument moves for reasons unrelated to the treatment.** This is
closer to §3.5's HALT-2 shape (*"a study comes back incoherent rather than merely negative ... that
is a broken instrument"*) than to a clean positive. The pre-authorized strand-local block is taken
(it stops exactly what needs stopping — the battery reads), the run CONTINUES elsewhere, and the
diagnosis goes to the user.

**Do NOT, on the strength of this:** conclude that weight decay reduces the k a model needs; quote
the seed-100 λ-series as a trend (it is non-monotone); or "fix" it by widening the tolerance until
the movement disappears — that would hide the instability rather than resolve it.

**Suggested (NOT scheduled, NOT run — the user decides):** the discriminating follow-up is seed
count, not λ resolution — if k-selection varies this much across two seeds and across epoch budgets
at fixed λ, the selection-set size (`PB`'s question) and seed variance are the live suspects, and
`PB` is already the scheduled task that measures the first of them.

**Protocol note — the driver did this right, unprompted:** it detected cells hitting the 100-epoch
cap without early stopping and re-ran them at 400 epochs (MASTER Decision 9: never read a
non-converged endpoint), and re-ran the one ceiling-bound cell at the raised k_max (§3.5's "ceiling
binds" branch; λ=1e-2 seed 100 re-selected 12, so the raise did not change the answer).
`any_hit_cap: false` across the final record. Those escalations are what surfaced the instability —
had the driver read the capped endpoints, this would have been recorded as a clean "moves".

*(Original task spec follows, retained verbatim as the pre-registration this run was judged against.)*

### P8 — does explicit regularisation move the selected k? (NEW 2026-07-21, MASTER Decision 21)

**Files (write set):** `automl_package/examples/probreg_p8.py` ·
`automl_package/examples/capacity_ladder_results/P8/`
**Spec:** The programme's research training is entirely unregularised (no weight decay, dropout,
norm layers, or mini-batching — audited 2026-07-21), so cheapest-within-tolerance may partly select
small k because small OVERFITS LESS, not because small suffices — a bias identical across all three
dials and invisible to cross-strand agreement. Discriminating check, one toy: train M3's per-k
ORDINARY sweep models at AdamW `weight_decay` λ ∈ {0, 1e-4, 1e-2}, 2 seeds, unchanged convergence
gates; apply the strand's selection rule to each per-k curve; report whether the selected k moves
beyond tolerance, emitting the verdict to
`automl_package/examples/capacity_ladder_results/P8/frozen.json`. **Moves → block THIS strand's
battery reads (P4/P6 may not proceed), log prominently, continue the OTHER strands, batch for
end-of-run user review** *(pre-authorized 2026-07-21 — a strand-local block, not a whole-run halt;
the strand's numbers conflate capacity with regularisation and the battery may not be read until
re-derived)*. **Does not move → robustness note that P6's report MUST cite.** The reported numbers come from a split not used for stopping or
selection. Weight decay is framed as the Gaussian prior it is — the no-arbitrary-penalty premise
binds the SELECTION objective, not MAP training.
**Non-goals:** no change to any model definition; no tuning of λ beyond the fixed grid; no real
data.
*Orchestration:* parallel: yes (disjoint write set) · deps: none · tier: sonnet high · scale:
static · shape: execution →
verify: one JSON per (λ, seed) under `automl_package/examples/capacity_ladder_results/P8/` each
carrying `selected_k`, `held_out_trajectory`, `hit_cap: false`;
`test -f automl_package/examples/capacity_ladder_results/P8/frozen.json` exits 0 and the file
carries `selection_moved` (bool) and, if true, the per-λ selected k values.


<!-- from docs/plans/capacity_programme/probreg.md lines 1366-1408 -->
### P3 — the M1 ≈ M3 claim, both halves, on toys

⛔ **BLOCKED on the M3 candidate-set ruling (§1, raised 2026-07-21).** M3's grid starts at k=5 while
the ladder is read from the bypass up, so M1 can currently select a rung M3 is not allowed to
consider. **No M1-vs-M3 headline may be produced until that is settled** — an unqualified "M1
matches M3" off a narrow grid would flatter the cheap arm for a reason unrelated to selection.

**Files (write set):** `automl_package/examples/probreg_p3.py` (Create) ·
`automl_package/examples/capacity_ladder_results/P3/`
**Spec:** On the existing 3 toy problems: train M3 (one dedicated model per k over the frozen k
grid), score each on held-out data, record the sweep's best k. **Emit the positive control first**
— `automl_package/examples/capacity_ladder_results/P3/m3_control.json`, recording whether M3's
per-k scores reproduce §3.2's finding pattern — and check it before training or reading any M1
number (MASTER Decision 14). Train M1 (k-dropout) and record the arbiter's chosen k. Report
**(a)** quality at matched k — this re-runs §3.2's check inside the new harness so both halves come
from one artifact — and **(b)** agreement between the two chosen k's, which is the untested half.
Both across the same seeds as §3.2 so the numbers are comparable.
**Doctrine:** MASTER Decision 14 — run the known-good arm first, alone. Here that is M3 at the k
§3.2 already characterised; it must reproduce before M1 is scored. **MASTER Decisions 9 (trajectory
discipline) and 16 (exonerate optimisation before blaming architecture) both bind**: every trained
model (M1's k-dropout net and M3's per-k dedicated nets) reports its full held-out trajectory, its
convergence flag is trajectory-verified with `hit_cap=False`, and any arm that looks like it lost is
run through the escalation ladder (LR sweep → clipping → warmup → init scheme → normalization)
before being recorded as an architecture finding rather than an optimization one.
**Non-goals:** no real data (P4); no baselines (P4); do not re-tune the arbiter's neighbourhood
width.
*Orchestration:* parallel: no · deps: P1, PA, PB, P7, **FP-12** (the fixed-σ scorer AND fixed-σ
training — without it M1 and M3 both select on a learned variance) · **✅ the §1 M3 candidate-set
ruling is SETTLED (user, 2026-07-21, MASTER Decision 25 — grid spans the ladder, bypass included);
this dep is CLEARED.** *(It was previously stated only in prose above this task — a dependency-driven
dispatcher would have seen the deps clear and dispatched P3 straight past the ⛔ banner. Carried into
the deps field 2026-07-21 after an audit caught exactly that gap, and now discharged here rather than
silently dropped.)* (the arbiter must be correct, M1's
mechanism must exist, and the selection-set fraction must be frozen before M1's choices mean
anything) ·
tier: sonnet high · scale: static · shape: execution · verify:
`test -f automl_package/examples/capacity_ladder_results/P3/m3_control.json` and
`python -c "import json; d=json.load(open('automl_package/examples/capacity_ladder_results/P3/m3_control.json')); assert d['reproduces'] is True, d"`
exits 0 **before** any M1 number is read (Decision 14); then
`ls automl_package/examples/capacity_ladder_results/P3/*.json` shows one file per (problem, seed),
each containing `m1_chosen_k`, `m3_chosen_k`, `per_k_held_out_scores`, `held_out_trajectory`, and
`hit_cap: false`.


<!-- from docs/plans/capacity_programme/probreg.md lines 1441-1470 -->
### P5 — explain the middle-k coherence cost

**Files (write set):** `docs/flexnn/p5_middle_k_coherence.md` ·
`automl_package/examples/capacity_ladder_results/P5/`
**Spec:** 8 of 9 cases show a middle-k coherence failure (§2). Establish the mechanism. Candidate
to falsify first: middle k's get the least effective gradient share under uniform k-dropout while
the largest k is reinforced by every prefix. If true, the fix is a non-uniform k schedule and that
is a design decision for the user, not an improvisation.
**Prior art to falsify against FIRST (added 2026-07-21):** this candidate is width's W1 diagnosis
restated — the per-sample uniform draw starves rungs, recorded with numbers at
`docs/flexnn/width/history/width_dial_2026-07/EXECUTION_PLAN.md:119-129` and fixed there by the W2 sandwich,
with the slimmable-networks literature basis at
`docs/flexnn/width/history/width_dial_2026-07/nested_architecture_research_2026-07-11.md:104`. This strand's
plan carried no reference to that record and had budgeted P5 as fresh opus-tier discovery. **P7's
coherence re-check under the all-rungs schedule (Decision 20) is the cheap discriminating
experiment**: if the middle-k failures disappear under P7, the mechanism was the schedule and this
task reduces to recording that with the two runs side by side; P5's discovery budget is spent only
on whatever survives P7.
**Trajectory discipline (MASTER Decision 9) and optimization-first (MASTER Decision 16) both
bind**: before any middle-k pattern is attributed to architecture, show the escalation ladder (LR
sweep → clipping → warmup → init scheme → normalization) was exhausted on the middle-k arms, with
full held-out trajectories on disk, not endpoints.
**Non-goals:** the schedule itself is already ruled (Decision 20 → P7); this task explains what
survives P7, it does not redesign the schedule.
*Orchestration:* parallel: yes · deps: P3, P7 · tier: opus xhigh (discovery-shaped) · scale: static ·
shape: research · verify:
`test -f docs/flexnn/p5_middle_k_coherence.md` exits 0, and
`grep -oE "automl_package/examples/capacity_ladder_results/P5/[A-Za-z0-9_./-]+\.json" docs/flexnn/p5_middle_k_coherence.md`
returns at least one path, and each returned path resolves with `test -f`.


