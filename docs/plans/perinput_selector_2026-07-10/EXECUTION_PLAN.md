# Per-input selector program — ⛔ FROZEN / COMPLETE. NEVER DISPATCH FROM THIS FILE.

> **FROZEN 2026-07-20. COMPLETE — verified against the results ledger, not against this file.** Its
> ProbReg tasks S1, S2, T2, T3 and H1 all carry adjudicated GO verdicts in
> `automl_package/examples/capacity_ladder_results/RESULTS.md`, all completed 2026-07-10 — **even
> though this file's own inline status markers were never updated to say so.** Trust the results
> ledger, never the glyphs here.
>
> Live plan of record: `docs/plans/capacity_programme/`. **All ProbReg work is owned by
> `docs/plans/capacity_programme/probreg.md`.**

*Historical header, as ratified 2026-07-10:*

Successor program to `docs/plans/capacity_ladder_2026-07-09/EXECUTION_PLAN.md` (the
capacity-ladder program, COMPLETE + certified). That program settled the MEASUREMENT
question (which per-input capacity readouts are faithful). This program settles the
SELECTOR question: how to actually build, evaluate, and deploy the per-input capacity
selector — plus the toy-design gaps the review identified (strong-signal depth toy,
multi-dimensional inputs, moving-mode power curve) — and ends with ONE consolidated
report pass and organization cleanup.

**User ratification (2026-07-10, this session):** the user reviewed the gap analysis and
approved this exact scope, including: two-stage (frozen-fit → post-hoc selector) as the
architecture of record for selector work; the soft-weighting-vs-hard-pick axis; the
distill-into-existing-gate deployment form; deferral of X6/X9 behind this program; X8
folded into the report pass; X7 behind an evidence-based trigger gate; X10 absorbed
in trimmed form; reports updated ONCE at the end; organization cleanup included.
There is NO open scoping question. Do not re-ask any of this.

---

## 0a. Decisions locked (do not revisit, do not ask the user)

| # | decision | source |
|---|---|---|
| L1 | Two-stage is the selector architecture of record: stage 1 trains ONE model good at every capacity (per-sample random capacity draw, no selector influencing the fit); stage 2 builds the selector afterward against held-out evidence. Joint gate training is a COMPARISON ARM only (H1), never the recommended path unless it wins H1 outright. | user 2026-07-10 |
| L2 | ProbReg framing: classifier over k classes (classes + per-class heads + bypass). NEVER describe as "Gaussian mixture" in any user-facing text or report. | user feedback 2026-07-10 (memory `feedback_explain_probreg_as_classifier`) |
| L3 | Soft-vs-hard is model-specific: for ProbReg all capacities cost one forward pass (prefix renormalization), so the deliverable is a per-input WEIGHTING head (blend), argmax kept only as interpretable readout. For FlexNN a hard pick buys real compute (early exit), so hard routing is justified there iff the compute saving is measured. | user-approved refinement 1 |
| L4 | Deployment form = the EXISTING gate head (`n_classes_predictor` / `n_predictor`), retrained under a two-phase schedule. Architecture frozen; training-scheme changes only. Library code changes remain out of scope (K7/F5 still ⛔ user-gated); H1/H2 drive the real model classes from EXPERIMENT SCRIPTS only. | user-approved refinement 2 + standing ruling |
| L5 | Strictly probabilistic (THE PREMISE): every objective/criterion/target is a likelihood, a prior, or a proper-score operation on held-out data. No tuned λ, no penalties, no load-balancing auxiliaries. | standing ruling |
| L6 | X6 (NPMLE variance arm) and X9 (dendrogram global-k) DEFERRED behind this program. X8 (confidence-set readout) is a reporting upgrade executed inside R-INT, not a training task. X10 absorbed as P1 (trimmed 3-point ladder). X7 runs ONLY if the S2 trigger fires (§5, gate G-X7). | user 2026-07-10 |
| L7 | Reports updated ONCE, at the end (R-INT), via the research-report skill; the pass also absorbs the outstanding X-queue integration (X1–X4b supersede REPORT-2 §4). Certified carry-forward wordings are already adjudicator-locked in RESULTS.md "## X-queue follow-ups" — copy, do not re-derive. | user 2026-07-10 |
| L8 | Prior capacity-ladder results are CERTIFIED and are not re-litigated. No task in this program re-runs K0–K6/F0–F3/V0–V3/X1–X5 or questions their verdicts. | user 2026-07-10 |

## 0b. Standing governance — inherited VERBATIM

§0b of `docs/plans/capacity_ladder_2026-07-09/EXECUTION_PLAN.md` applies to every task
here unchanged: environment (`~/dev/.venv/bin/python`, `AUTOML_DEVICE=cpu` on EVERY run,
detached launches via `setsid nohup systemd-inhibit --what=idle:sleep:handle-lid-switch`),
strictly-probabilistic audit line, pre-registration before running (bars written first,
outcomes against them, no post-hoc reframing, 3 seeds {0,1,2} minimum), escalation on any
failed bar (STOP the lane → fresh-context adjudicator; other lanes continue), G1–G6 read
rules (bootstrap SE, abstain semantics, cap saturation, per-bin cells, locality guard at
half width — for kNN neighbourhoods the G5 analog is re-read at half `n_nbr` — and
full-vector reporting), measure-one-unit-before-matrix for every new run shape, per-task
artifact folders under `automl_package/examples/capacity_ladder_results/<TASK>/`,
`--selftest` known-answer path in every experiment script (must PASS before any real
read), local commits per completed task with plain messages, no AI provenance anywhere,
RESUME.md updated at every task completion.

Additions for this program:
- **Run ownership**: the ORCHESTRATOR MAIN THREAD owns every training run (detached
  launch + `while kill -0 <pid>` monitor). Workers author code + selftests ONLY; a
  worker never babysits a run (they idle out mid-run — known failure).
- **Shared machine**: `OMP_NUM_THREADS=4` max; before every launch check `uptime`
  1-min load < 32, else wait 15 min and re-check. Runs strictly SERIALIZED; only
  code-authoring parallelizes.
- **Ledger**: every certified task result is appended to
  `automl_package/examples/capacity_ladder_results/RESULTS.md` under a new section
  `## Per-input selector program (S/T/H/P)`, immediately on adjudication — so R-INT
  is a copy-and-compose job, not a re-derivation.

## 0c. Model/effort routing (for the orchestrating session)

| work | tier |
|---|---|
| author scripts + selftests from the locked specs below | task-worker (Sonnet/high), one per script, spec block passed verbatim |
| mechanical: json→markdown tables, ruff fixes, file moves | utility-worker (Haiku/low) |
| certify each task summary vs its prereg; any failed bar; any surprise | adjudicator (Opus/xhigh), FRESH context, never the producing session |
| run ownership, sequencing, ledger folds | orchestrator main thread |
| R-INT report editing | orchestrator main thread invoking the research-report skill (token-heavy; do not delegate the skill loop to workers) |

Every dispatch carries: exact absolute paths, the task's spec block verbatim, the
selftest command, the verification command, explicit non-goals. A worker that cannot
fill its output contract returns UNRESOLVED — it never improvises design decisions.

**The orchestrator may contact the user ONLY at three gates:**
1. **G-X7** (§5): the S2 trigger fired → present the evidence + cost (~days, new
   network), WAIT.
2. **G-COMMIT** (§7, ORG-1): one single question at the very end on commit scope.
3. **G-FORK**: a fresh-context adjudicator has examined a failed bar or surprise and
   explicitly ruled it a user-level design fork (adjudicator first, always).
Everything else: proceed. Questions outside these gates are a plan violation.

## 0d. Sequencing (runs serialized; builds parallel)

```
ORG-0 ──► build {S1, T1, T3, P1} in parallel (4 workers)
      ──► run S1 ──► build+run S2 ──► record X7-trigger verdict (don't ask yet)
      ──► run T1 ──► (T1 outcome gates H2; X10/P1 runs regardless, trimmed)
      ──► run T3 ──► run P1
      ──► build+run T2 (heaviest — last of the toys)
      ──► build+run H1 ──► H2 iff T1 PASSED both bars
      ──► if X7 trigger fired: G-X7 (user gate) ──► X7 iff approved
      ──► R-INT (single report pass) ──► ORG-1 (cleanup + G-COMMIT)
```
Dependencies: S2 needs S1's protocol + winner. H1 needs the S1/S2 winner recipe.
H2 needs T1 PASS. T2 needs S1's winner recipe (kNN-smoothed variant). R-INT needs
everything before it. Nothing else blocks anything.

**Budget (state before launch, track in ledger):** compute ≈ 3.5–5 days serialized
CPU wall-clock (T2 and H1 dominate). Tokens ≈ 4–5M total: ~1M worker builds (~8
dispatches), ~1M adjudications (~7 fresh passes), ~1.5–2M R-INT report loop, ~1M
orchestration. If projected spend exceeds 150% of this, checkpoint and reassess
scope (drop D=10 from T2 and the fine-tune arm from H1 first).

---

## 1. WS-A — selector science on frozen tables

### S1 — evaluation protocol + target-construction factorial (ProbReg)
**Purpose.** Replace the confirmatory 3-arm comparison with a clean factorial + knob
sweep, under a corrected evaluation protocol (honest bound; blend-vs-hard axis).

**Inputs (all exist, frozen — no ladder retraining):**
`automl_package/examples/capacity_ladder_results/K4/nested_toy{C,D,E}_seed{0,1,2}.pt`
and the broad-twin controls `nested_toy{C,E}_broad_seed{0,1,2}.pt` (keys: `score`
(N×C float64 held-out per-example log-likelihood), `x`, `c_grid`). Reuse
`capacity_ladder_k6.py` patterns verbatim: `_RouterMLP` (32,32), 300 ep, lr 1e-2,
index-parity train/eval split, targets computed on the train half only.

**Protocol (applies to every arm, pre-registered):**
- **Blend read** (primary, per L3): per-input weighted density
  `blend_i = logsumexp_c(log w_c(x_i) + score[i,c])` with the selector's own weights;
  report alongside the hard argmax-routed NLL.
- **Honest bound `oracle-x`**: split the eval half again by index parity; estimate the
  neighbour-averaged (box-car width 0.075) per-capacity advantage curve on the even
  eval points; route the odd eval points by its argmax; report their mean actual score.
  Keep the old per-point max as `oracle-noisy` (continuity label only).
- Metrics on the eval half; 3 seeds; plain bootstrap SE (B=1000) on paired diffs.
**Factorial arms** (identical router config; only the target changes):
  (1) soft = K6 soft (responsibilities with per-tercile stacked prior) — baseline;
  (2) soft-no-prior = per-row `softmax_c(score[i,:])` (isolates the prior);
  (3) soft-smoothed = softmax of the neighbour-averaged score rows, no prior
  (isolates smoothing); (4) hard knee labels (K6 hard); (5) raw per-row argmax (K6
  pilot). Factorial reading: 1v2 = prior effect; 2v5 = softness effect; 3v2 =
  smoothing effect; 1v3 = prior-vs-smoothing form.
**Knob sweep** (winning target only; toys D + C_broad, 3 seeds): prior bins ∈ {2,3,5};
neighbour width ∈ {0.0375, 0.075, 0.15}; target temperature τ ∈ {0.5, 1, 2}
(soft targets ∝ exp(log-target/τ), renormalized).

**Pre-registered bars:** (i) blend ≥ hard for every arm on ≥ 8/9 structured cases;
(ii) some soft arm ≥ hard-knee arm on ≥ 7/9 (K6 replication under the new protocol);
(iii) oracle-x ≤ oracle-noisy on 9/9; (iv) broad controls: every arm's blend advantage
over global ≤ 0.02 nat on all 6 broad cases; (v) knob robustness: winner's ordering vs
arm (4) unchanged across the full knob grid — if not, the sensitivity table IS the
finding (report, don't tune).
**Selftest:** reuse K6 `_selftest_table`; add asserts: blend ≥ hard on the synthetic
table; oracle-x recovers the designed tercile peaks {1,3,6}; τ=1 reproduces the τ-free
target bit-identically.
**Script/artifacts:** `automl_package/examples/capacity_ladder_s1.py` →
`capacity_ladder_results/S1/{PREREGISTRATION.md,s1_summary.json}`.
**Run:** `AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python -u
automl_package/examples/capacity_ladder_s1.py` (selftest first with `--selftest`).
**Cost:** minutes–1 h. **Non-goals:** no ladder retraining; no new toys; no library code.

### S2 — direct held-out-likelihood selector (the principled objective)
**Purpose.** Train the selector DIRECTLY on the deployment objective instead of
imitating any derived label — the arm the original study never ran.

**Design.** Same `_RouterMLP` w(x). Objective (train half only):
maximize `mean_i logsumexp_c(log softmax(w(x_i))_c + score_tr[i,c])` — per-input
stacking; an ordinary held-out log-score operation (L5-compliant, no labels, no λ).
Adam lr 1e-2, 300 ep (match S1); also record a 3000-ep arm to check under-training.
Evaluate under the FULL S1 protocol (blend + hard + oracle-x + broad controls), same
seeds/splits, head-to-head vs the S1 winner.
**Pre-registered bars:** (i) S2 blend ≥ S1-winner blend on ≥ 6/9 structured cases and
mean paired diff ≥ 0 (prediction: the direct objective wins or ties — it optimizes the
measured metric); (ii) broad controls ≤ 0.02 nat as in S1; (iii) S2 hard-read remains
≥ global on ≥ 8/9 (hard read degrades gracefully).
**X7 trigger (record verdict, do NOT ask the user unless it fires):** on toy D, gap
closure = (global − best_selector_blend)/(global − oracle-x), averaged over seeds. If
closure < 0.5 → the deployable-selector question is still open → G-X7 fires (present
at the §0d gate point). If ≥ 0.5 → X7 is SKIPPED; write one ledger line.
**Selftest:** on the K6 synthetic table the direct objective must reach blend NLL ≤
the soft-imitation arm's blend NLL + 0.005.
**Script/artifacts:** `capacity_ladder_s2.py` → `capacity_ladder_results/S2/…`.
**Cost:** minutes–1 h. **Non-goals:** no joint fine-tuning (that is H1's arm iv); no
new selector architectures (capacity of the head is the safety mechanism — (32,32) fixed).

## 2. WS-B — discriminating toys

### T1 — provably-deep-required toy (THE depth-lane discriminator)
**Purpose.** The depth lane's null was measured on toys whose per-input signal turned
out incidentally tiny. T1 builds a toy where region-B provably requires depth
(Telgarsky-style composition), so the per-input depth machinery is tested where a
large signal EXISTS by construction. Outcome semantics are pre-registered (below) —
this single task decides the depth lane's future.

**Toy (add `make_toy_t1` to `automl_package/examples/_capacity_ladder_toys.py`):**
x ~ U[0,1], N_train=1600, N_test=500 (~F2-scale — the point is signal SIZE, not N).
Region A (x<0.5): y = 1.5·x + ε. Region B (x≥0.5): y = tent^(5)(2(x−0.5)) + ε where
tent(z)=1−|2z−1| iterated 5× (2⁵=32 linear pieces), ε ~ N(0, 0.1²). **Trunk width = 8,
DELIBERATELY smaller than F2's width 24** (`capacity_ladder_f2.py:82 HIDDEN_SIZE=24`) —
the small width makes depth the binding constraint: a depth-1 width-8 net gives ~9 linear
pieces and provably cannot fit 32 oscillations, while a deeper net can (Telgarsky depth
separation: a k-fold composition needs exponential width to represent at constant depth).
The EXACT knee depth (2/3/4) is NOT pre-committed — it depends on width/optimization/noise;
T1 only needs a LARGE region-B-localized depth signal (bar (i)). Because width ≠ F2's 24,
region A (same width, same run) is the PRIMARY matched negative control; the F2 G_flat
tables are a secondary cross-reference only (width-24 mismatch noted). Also emit
`sample_toy_t1_given_x` for gold reads.
**Harness (reuse, do not reinvent):** `capacity_ladder_f2.py` (nested-depth
FlexibleHiddenLayersNN surrogate + dedicated fixed-depth sweep, BN off, depths 1..6,
3 seeds) → `.pt` depth tables; then `capacity_ladder_x3.py` `run_repeated_crossfit`
(50 splits, Nadeau–Bengio SE) + `capacity_ladder_x1.py` `hierarchical_perbin_stack`
on those tables, with bins = the two regions (A/B) and terciles within B.
**Pre-registered bars:** (i) CONSTRUCTION (toy has a large region-localized depth
signal): dedicated fixed-depth sweep, per region — region-B held-out NLL improves
by ≥ 0.3 nat TOTAL and > 2·SE from depth 1 to its knee, with the knee at depth ≥ 2
(the specific knee value is reported, not pre-committed); region-A flat (no > 2·SE
gain past d1). If (i) fails: redesign ONCE (iterate up to 7×; first try width 6 then
a deeper composition tent^6 before touching anything else), re-run; fails again →
adjudicator. (ii) PER-INPUT READ: repeated-crossfit corrected-t > 2 for the A-vs-B
contrast on ≥ 2/3 seeds AND hierarchical arm agrees (sig on the same seeds); the
matched control is REGION A within the same run (same width/N) — T1's B-vs-A per-input
pass rate must exceed region A's own null rate; the F2 G_flat tables are a secondary
cross-reference (width-24 mismatch, not the primary control).
**Outcome semantics (locked):** PASS (i)+(ii) → machinery VALIDATED; depth-lane null
reframed "toy-specific signal absence, instrument sound"; H2 UNLOCKED. PASS (i) FAIL
(ii) → genuine machinery failure on a large signal → STOP, adjudicator (this would
contradict the count-lane success — top-priority surprise). H2 stays locked.
**Selftest:** known-answer synthetic depth table (plant +0.8-nat B-region advantage)
recovered by both readers, as in X3/X1 selftests.
**Script/artifacts:** `capacity_ladder_t1.py` → `capacity_ladder_results/T1/…`.
**Cost:** ~half-day. **Non-goals:** no library changes; no N-sweeps (that's P1); no
router yet (that's H2).

### T2 — multi-dimensional count toys (the port de-risk)
**Purpose.** Everything so far is 1-D. The real-model ports are gated precisely on
"neighbourhood reads may break in many dims" — T2 measures exactly that degradation
on toys with analytic ground truth.

**Toys (extend `_toy_datasets.py`):** `make_toy_d_ndim(n, dim, rotated, seed)`:
x ~ U[0,1]^dim; staircase k*(s) ∈ {1,2,3} by thirds of s, where s = x[0]
(axis-aligned) or s = u·x/√dim with u a fixed known unit vector (rotated); remaining
coordinates are nuisance. Same component geometry as toy D (separation 4σ, σ=0.3,
means centred 0). Broad twin at dim=2 (variance-matched, k*=1 everywhere). Matrix:
dim=2 axis + broad, dim=5 axis + rotated, dim=10 axis → 5 configs × 3 seeds = 15
ladder trainings. N_train = N_test = 2500 (K4-scale).
**Ladder:** generalize `_capacity_ladder_nested.NestedKSurrogate` to input_dim ≥ 1
(constructor arg; trunk widens to (64,64) for dim>1). REGRESSION GUARD: with
input_dim=1 the loss trajectory on toy D seed 0 must be bit-identical to current code
(selftest assert) — the 1-D certified results must be unreproducible-risk-free.
**Reads:** neighbourhoods by kNN (n_nbr=50, Euclidean) replacing the 1-D box-car;
targets = S1's smoothing-only soft variant (arm 3) built with kNN smoothing (the
tercile prior needs a binnable scalar — in multi-D use terciles of the TRUE index s
ONLY for the gold read, never for targets: selector-visible information is x alone).
Selector = S1/S2 winner recipe, input dim = dim. G5 analog: re-read at n_nbr=25.
Gold read: per-region capture via `sample_toy_d_ndim_given_x` (analytic k*(s)).
**Pre-registered bars:** (i) dim=2 axis: selector blend beats global > 2·SE on ≥ 2/3
seeds (one nuisance dim must not kill a 4σ staircase); (ii) the DELIVERABLE is the
degradation curve — (selector − global) advantage and gold region-capture vs dim ∈
{2,5,10}; pre-registered read: report the dim at which the advantage drops below
2·SE (no pass/fail at 5/10); (iii) rotated-vs-axis at dim=5: paired diff reported;
prediction: kNN targets are rotation-invariant → |diff| within 2·SE (a material gap
= binning-artifact tell); (iv) dim=2 broad twin: selector advantage ≤ 0.02 nat.
**Selftest:** input_dim=1 regression guard (above) + 2-D synthetic known-answer
staircase recovered.
**Script/artifacts:** `capacity_ladder_t2.py` → `capacity_ladder_results/T2/…`.
**Cost:** the compute bulk — measure ONE dim=5 ladder first, extrapolate, report in
ledger before launching the matrix; expect ~1–1.5 days serialized. **Non-goals:**
dim > 10; real datasets; library changes beyond the examples-dir surrogate.

### T3 — moving-mode power curve (count-lane analog of X10)
**Purpose.** The moving-mode case closed as "absent at N=1000". T3 turns that into a
power statement: absent, or under-powered?
**Design.** Reuse `capacity_ladder_x4b.py` VERBATIM (non-nested June instrument,
R=8 multi-restart keep-best-by-train-MAP, leak-free) at N_train=N_test ∈ {1000, 4000,
16000}, E-type humped data + E_broad control, 3 seeds. Reads per N: gold Δ*_mid
(model-capture) and arbiter mid-region recovery, with SEs.
**Pre-registered outcome readings (locked):** hump emerges (gold Δ*_mid > 0 by 2·SE on
≥ 2/3 seeds) at some N → "recoverable at N=…" (report the crossing N). Stays ≤ 0 at
16k with control flat → "moving-mode per-input count effectively absent up to N=16k",
with the measured bound. Control must stay flat 3/3 at every N (else instrument
invalid at that N — adjudicator).
**Selftest:** X4b's existing selftest re-run unchanged (it already covers the
instrument); add an N=16000 smoke-fit (1 restart, 1 seed) for wall-time measurement.
**Script/artifacts:** `capacity_ladder_t3.py` (thin N-sweep driver around x4b
functions) → `capacity_ladder_results/T3/…`. **Cost:** hours–half-day.

## 3. WS-C — real-model head-to-head (training SCHEMES, same architecture)

### H1 — ProbReg: two-phase schedule vs shipping joint gate vs fixed-k
**Purpose.** The decisive "does our model work" comparison, on the SAME shipping
architecture (`ProbabilisticRegressionNet` driven from an experiment script): does
the program's two-phase scheme beat the current jointly-trained gate?

**Arms (toys D and E + C_broad control; 3 seeds; identical net config, k_max=6,
held-out NLL of the per-input blended density as primary metric, G6 full-vector
selector distributions logged):**
  (a) SHIPPING JOINT: `NClassesSelectionMethod.SOFT_GATING` +
      `NClassesRegularization.ELBO` (the library's best combo), trained as today.
  (b) TWO-PHASE: phase 1 — per-sample k ~ U{1..k_max} masked-prefix schedule on the
      SAME net, gate quiescent (mirror `masked_prefix_nll` / the
      `_compute_predictions_for_k` masking pattern from
      `automl_package/models/selection_strategies/base_selection_strategy.py`);
      phase 2 — freeze everything except `n_classes_predictor`, train it with the
      S1/S2-winning selector objective on a held-out-within-train split. Respect the
      head's logit layout (k∈{2..k_max} then bypass LAST,
      `probabilistic_regression_net.py:66-75`).
  (c) FIXED-k SWEEP: same net, fixed k ∈ {1..6} (reuse the pattern of
      `probreg_kselection_comparison.py`); best single k by held-out NLL.
  (d) OPTIONAL FINE-TUNE (measured, no bar): arm (b) + 100 ep joint at lr 1e-4.
      Prediction: |ΔNLL| ≤ 0.01 nat (safe no-op). Drop this arm first under budget
      pressure.
**Pre-registered bars:** (i) (b) ≥ (a) on D held-out NLL 3/3 seeds (the program's
central prediction); (ii) (b) within 0.02 nat of best (c) on C_broad (no invented
structure); (iii) parity: (b)'s phase-2 gate ≈ the standalone S-winner selector on the
same data (|ΔNLL| ≤ 0.01 — same objective, different host); (iv) E: report only (no
bar — T3 owns the moving-mode question).
**Scope guard (HARD):** experiment script `capacity_ladder_h1.py` ONLY. If phase 2
proves impossible without touching library code → STOP, adjudicator (do NOT edit the
library; K7 is user-gated).
**Selftest:** on a tiny D subsample (N=200, 50 ep) the script must run all arms
end-to-end and produce finite NLLs + full selector vectors.
**Cost:** measure one arm-(b) fit first; expect ~1 day serialized for ~30 fits.
**Non-goals:** library edits; XPU; moving-mode verdicts; K7 porting language in any
output (this is evidence FOR a future K7 decision, not the port).

### H2 — FlexNN analog (CONDITIONAL: runs iff T1 passed BOTH bars)
Same shape on `FlexibleHiddenLayersNN` with the T1 toy: (a) shipping joint depth gate
(GumbelSoftmax + DepthRegularization.ELBO), (b) two-phase (phase 1 random-depth
schedule — the F-lane scheme; phase 2 distill `n_predictor` on soft per-depth
responsibilities from held-out-within-train), (c) fixed-depth sweep. Plus the L3
compute read: `predict(inference_mode="hard")` wall-clock vs full-depth forward.
**Bars:** (i) (b) ≥ (a) on T1-toy held-out NLL ≥ 2/3 seeds; (ii) hard routing saves
≥ 25% inference wall-clock on T1 (region A exits shallow); (iii) G_flat-style control:
two-phase gate's depth distribution on flat data collapses to shallow (no invented
depth). **Script:** `capacity_ladder_h2.py`; selftest as H1. **Cost:** ~half-day.

## 4. WS-D — absorbed queue

### P1 — depth power curve (X10, trimmed)
As specced in the prior plan's X10 but 3-point: N_TEST ∈ {500, 2000, 8000} (train
scaled ∝), toy G + G_flat, reusing the X3 harness + F2 tables/readers. At each N:
the per-bin detection floor (min detectable Δ at Nadeau–Bengio-corrected 2·SE) and
the measured per-bin signal. Extend to 5 points ONLY if the curve is non-monotone.
**Bars:** the deliverable is the floor-vs-N curve + whether the signal crosses it;
pre-registered readings: crossing → "recoverable at N=…"; no crossing at 8k →
"below the floor up to N=8k" (quantified bound for the report). G_flat must stay
below its own floor at every N. **Script:** `capacity_ladder_p1.py` →
`capacity_ladder_results/P1/…`. **Cost:** hours–half-day. Runs regardless of T1's
outcome (the report wants the bound either way).

### X7 — amortized per-input evidence network (GATED)
Runs ONLY if S2's trigger fired AND the user approves at G-X7. Spec lives in the
prior plan §8.5 (Evidence-Networks loss on simulated (x,y|k); a DIAGNOSTIC net, not
a model change). Days. If skipped, one ledger line records the trigger arithmetic.

### Deferred (do not build, do not mention to user again until program end)
X6 (NPMLE variance arm), X9 (dendrogram global-k): deferred per L6; listed in the
prior plan §8.5. Re-surface them in the R-INT "future work" section only.

## 5. WS-E — R-INT: the single consolidated report pass (END of program)

Executed by the orchestrator main thread via the **research-report skill** (contract
guard + pdflatex book-chapter build + cold-read gate), AFTER every experimental task
above is certified and folded into RESULTS.md. One pass, three documents, firm scope:

1. **REPORT-2** (`docs/capacity_ladder_report_2026-07-10/capacity_ladder_report.pdf`):
   integrate (a) the outstanding X-queue corrections — X1/X3 supersede §4's
   "power-limited at N=500" → "no signal above the detectable/control band"; X2, X4/X4b
   as new results — using the adjudicator-locked wordings in RESULTS.md verbatim; and
   (b) a NEW section: the per-input selector program (S1/S2 selector science, T1–T3
   toys, H1/H2 head-to-head, P1 bound). L2 framing throughout (classes, not mixtures).
2. **ProbReg technical note** (`docs/probreg_kselection_findings.md` → rebuilt PDF):
   add the selector-construction recommendation of record (winning target/objective,
   the blend-vs-hard ruling, the two-phase schedule result, knob sensitivities).
3. **FlexNN depth note**: ONLY if H2 ran — new short note
   `docs/flexnn_depth_selector_<date>/` (else the depth story stays inside REPORT-2;
   do not create an empty shell).
Plus: **X8 as a reporting upgrade inside these documents** — wherever a knee/selector
read is Sivula-small, report the plausible-rung SET (cheap post-processing on existing
tables; a worker computes the sets; no training).

**Token-efficiency directives (binding):** batch ALL edits per document into ONE
research-report invocation each; maximum 2 cold-read rounds per document before
adjudicator triage; reuse existing figures wherever numbers are unchanged; RESULTS.md
certified wordings are the copy-source (no re-derivation); NOTE-MOE is touched only
if a cross-reference it makes is now false (check, don't rewrite). Reports are
authored as the user; no AI/tool provenance anywhere (standing rule).

## 6. WS-F — organization & cleanup

### ORG-0 (first task, before any build)
- Verify env: `~/dev/.venv/bin/python -c "import automl_package, torch"`.
- Move the two repo-root AI handover notes `REVIEW_HANDOVER_2026-07-03.md`,
  `STACKING_NOTE_2026-07-05.md` → `docs/plans/capacity_ladder_2026-07-09/handover/`
  (they are AI artifacts: keep local, never commit — add to `.git/info/exclude`).
- Confirm the K4 table inventory (§1 inputs) — done 2026-07-10, re-verify cheaply.
- No other moves now: existing experiment scripts stay where certified results
  reference them.

### ORG-1 (final task, after R-INT)
- `~/dev/.venv/bin/python -m ruff check automl_package/` clean (line length 180).
- Every new artifact in its task folder; no strays at repo root or loose in
  `examples/`; new scripts all named `capacity_ladder_<taskid>.py`.
- RESUME.md/CHANGELOG rolled via the checkpoint skill (`/checkpoint --final --tidy`
  once the user confirms the program is done).
- **G-COMMIT (single user question, the only one at program end):** present the
  full untracked inventory in three blocks — (1) June deps (`_toy_datasets.py`,
  `_variational_em_perinput.py`, `_capacity_ladder*.py`, `_kselection_metrics.py`),
  (2) experiment scripts + results (capacity_ladder_* + results folders), (3) this
  program's new files — and ask commit scope ONCE (options: all / library-relevant
  only / none). NEVER stage CLAUDE.md or any AI-instruction/handover file.

## 7. Definition of done (orchestrator checklist)

- [ ] ORG-0 done; env verified; handover notes relocated.
- [ ] S1, S2 certified + folded (selector recipe of record named in the ledger).
- [ ] X7-trigger verdict recorded (and G-X7 held if fired).
- [ ] T1 certified (+ outcome semantics applied: H2 unlocked or machinery escalation).
- [ ] T2 degradation curve certified (the port-decision evidence).
- [ ] T3 + P1 power statements certified (both lanes carry quantified bounds).
- [ ] H1 certified (± H2 if unlocked); scheme-of-record recommendation in ledger.
- [ ] R-INT: all documents rebuilt, gates clean, X-queue integration absorbed.
- [ ] ORG-1: ruff clean, foldering done, G-COMMIT answered, checkpoint written.
- [ ] Every certified block appended to RESULTS.md `## Per-input selector program`.

## 8. Resumption protocol (fresh orchestrator session)

Read, in order: this file; `capacity_ladder_results/RESULTS.md` (ledger tail =
current truth); `RESUME.md`. Task status = presence of certified blocks in the
ledger + artifact folders (an artifact folder with a summary json + a ledger block
= done; never redo). Then continue at the first unchecked §7 item, honoring §0c's
three-gate rule. The prior program's plan is REFERENCE ONLY (governance §0b and the
X7 spec); nothing there is to be executed except as imported here.
