# WSEL-22(b) — the σ-anchored per-input labelling band: derivation, adoption gate, evaluation protocol

**Scope of this document.** This is the WRITTEN SPEC for WSEL-22 part (b)
(`docs/plans/capacity_programme/width.md`, the `### WSEL-22` block) — the replacement candidate
for the per-input labelling band that turns "which capacity is acceptable for this row" from a
free tuned constant into a derivation anchored on the toy's own known noise level. Authoring only:
no code is written here, no cell is run. Adversarial read + root go follow this document, per the
standing methodology gate (same process WSEL-21's rung (i) spec used,
`shared/wsel21-escalation.md` §9).

**What this document is emphatically NOT:**
- **Not (b′), the blend.** (b′)'s user ruling (`width.md` WSEL-22 block, "THE BLEND HAS NO
  THRESHOLD") is settled and out of scope here: the blend is a probability-weighted expectation
  over ALL widths and makes no discrete per-row pick, so it needs no acceptability band at all.
  Every derivation below concerns HARD routing's discrete argmax pick only — see §8 for the exact
  boundary.
- **Not (c), the global-rule remedy.** MASTER Decision 36 (the selection-policy ruling,
  `docs/plans/capacity_programme/MASTER.md` line 654) governs the GLOBAL width-selection rule
  fielded across a whole battery (accuracy-optimal vs cheapest-within-tolerance, picked by
  deployment objective) — a different problem at a different granularity (one pick per battery,
  not one pick per row) with its own noise-aware machinery already adopted (Decision 33(i),
  WSEL-20's 2·SE bootstrap band). Decision 36 does not alter, and is not altered by, anything in
  this document. WSEL-22(c) (`width.md`, same block, "CONDITIONAL — global-rule band remedy") is a
  separate, dormant, spec-gated task that may reuse the σ-anchoring IDEA lifted to the global
  curve if WSEL-24 convicts the bootstrap band — it is not triggered by, and does not trigger,
  adoption of (b).

## 1. Why this spec exists now (grounded in the harvest, not restated from memory)

WSEL-22(a) — the mechanical sensitivity sweep — is HARVESTED and frozen at
`automl_package/examples/capacity_ladder_results/WSEL22A/frozen.json`. Its verdict, read directly
off that ledger: all 9 constant-router-control findings hold at all four swept tolerances
(`tolerances_swept: [0.05, 0.1, 0.25, 0.5]`, `baseline_tolerance: 0.25`
<!-- source: `automl_package/examples/capacity_ladder_results/WSEL22A/frozen.json` (`tolerances_swept`, `baseline_tolerance`) -->),
while 15 of the 45 tracked secondary findings differ from the τ=0.25 grading somewhere in that
sweep, concentrated at the tolerance extremes and worst in the d=2 oblique geometry
(`width.md` WSEL-22 block, "(a) HARVESTED 2026-07-23" note). This is the pre-registered trigger
firing (MASTER Decision 33(ii), `docs/plans/capacity_programme/MASTER.md` line 597: "the per-input
labelling tolerance's pre-registered trigger FIRED — sensitivity sweep + σ-anchored replacement
candidate"), and the root's own trigger-evaluation note is explicit that authoring this document is
WARRANTED by (a)'s bounded-but-real sensitivity (`width.md` WSEL-22 block, "(b) trigger evaluation
(root)").

**The constant this spec replaces, verified at the code line.** `DEFAULT_TOLERANCE = 0.25`
lives at `automl_package/models/flexnn/routing.py:77`
(`DEFAULT_TOLERANCE = 0.25  # matches sinc_width_experiment.py:333 DELTA_TIE`) — one line lower
than the `:75` the plan block's prose currently cites; a two-line drift from an intervening edit,
immaterial to the constant's identity but recorded here because "verify at the code line, never
from memory" is binding, and this is exactly the kind of small citation drift that should be
surfaced rather than silently carried forward. It is consumed by
`_cheapest_within_tolerance_labels` (`routing.py:95-113`) inside `DistilledCapacityRouter.fit()`
(`routing.py:176-218`), which `FlexibleWidthNN.fit_router` calls with `tolerance: float =
DEFAULT_TOLERANCE` as its own default (`automl_package/models/flexnn/width/model.py:358-403`,
signature at `:363`); `FlexibleHiddenLayersNN.fit_router`
(`automl_package/models/flexnn/depth/model.py:481`) shares the identical primitive for depth. This
document's derivation targets `_cheapest_within_tolerance_labels`'s acceptability test — the
function that decides, per row, which capacities are "close enough" to the row-best to route
to — not the router MLP that is trained on whatever labels that test produces.

**Noise doctrine this derivation must honor throughout (`width.md` §3.7, lines 407-484):** variance
is FIXED at the generator's true, per-point value everywhere in this strand, never learned. Every
`σ` appearing below is a KNOWN constant read from the toy generator (`HETERO_NOISE_SIGMA = 0.05`
for `make_hetero`, region-dependent `{0.05, 0.05, HETERO3_NOISY_SIGMA=0.5}` for `make_hetero3`),
never estimated from the held-out sample. This is precisely what makes a closed-form,
zero-free-parameter-except-confidence-level test possible — no variance estimation step exists to
introduce the forbidden learned-variance leak (§3.7's WD6/WD7 findings).

## 2. The derivation

### 2.1 Setup and notation

For row `i` in a held-out (report) split, with per-capacity squared error
`e_i(w) = (pred_w(x_i) - y_i)^2` for `w` in the capacity grid (widths, cheapest-first, matching
`routing.py`'s `capacity_grid` convention) — every `pred_w` is a FROZEN, already-trained net; no
retraining occurs anywhere in this derivation or its evaluation. Let

- `w*_i := argmin_w e_i(w)` — the row-best capacity (ties broken by the grid's own cheapest-first
  order, matching `_cheapest_within_tolerance_labels`'s `argmax`-on-boolean-row convention),
- `σ_i := σ_true(x_i)` — the KNOWN, fixed noise standard deviation at `x_i` (§3.7; a lookup, not an
  estimate — `HETERO_NOISE_SIGMA` for `make_hetero`, the region-indexed constant for
  `make_hetero3`),
- `y_i = f(x_i) + ε_i`, `ε_i ~ N(0, σ_i²)` — the toy's own generative assumption, Gaussian and
  known exactly on these constructions (this is a genuine input to the derivation, not an
  approximation — see §2.5 for the one place a real approximation enters).

### 2.2 The existing rule, for contrast

`_cheapest_within_tolerance_labels` (`routing.py:95-113`) accepts width `w` for row `i` iff
`e_i(w) <= (1 + τ) * e_i(w*_i)` for a single free constant `τ` (currently `0.25` everywhere, no
row- or noise-dependence at all). The acceptability budget SCALES with the row's own achieved
error, `e_i(w*_i)` — which conflates two things that should be separated: how well the BEST
capacity fits this row (a mix of true signal difficulty and one noisy realization) and how much
slack a row deserves before a wider capacity is called "meaningfully better." A row that happens to
land a noisy, unusually large `e_i(w*_i)` gets a proportionally LARGER absolute budget under the
flat rule even though the noise level at that row may be identical to every other row's.

### 2.3 The σ-anchored acceptability rule

Define the **excess** of width `w` over the row-best as
`excess_i(w) := e_i(w) - e_i(w*_i) >= 0` (non-negative by construction — `w*_i` is the row
minimum). Width `w` is **acceptable** for row `i` at confidence level `c` iff:

```
excess_i(w)  <=  σ_i² · χ²_{1,c}
```

where `χ²_{1,c}` is the `c`-quantile of the chi-square distribution with 1 degree of freedom
(e.g. `χ²_{1,0.95} = 3.8415`). <!-- numcheck-ignore: standard chi-square(1) quantile, a mathematical constant verified via `scipy.stats.chi2.ppf(0.95, df=1)`, not a ledger result -->
The row-best itself always satisfies this trivially (`excess = 0`), so the rule reduces to the
existing "first True in a cheapest-first boolean row" mechanism `_cheapest_within_tolerance_labels`
already implements — the row-best-relative acceptability BOOLEAN ARRAY and its "cheapest capacity
meeting acceptability" readout via `argmax` are unchanged; only what counts as an acceptable excess
changes, from `τ · e_i(w*_i)` (row-relative, free constant) to `σ_i² · χ²_{1,c}` (noise-relative,
one confidence-level choice). This is a mechanical drop-in for the boolean-row step inside
`_cheapest_within_tolerance_labels`, not a new labelling architecture.

### 2.4 Why this test statistic — the reasoning, in full

Consider the null hypothesis `H0(w)`: *width `w`'s prediction is exactly unbiased at row `i`*
(`pred_w(x_i) = f(x_i)`), i.e. width `w`'s entire residual is the observation noise itself,
`e_i(w) = ε_i²`. Under `H0(w)`, since `ε_i ~ N(0, σ_i²)` exactly (§3.7), the standardized quantity
`e_i(w) / σ_i² = (ε_i/σ_i)²` follows a chi-square distribution with 1 degree of freedom exactly —
this is not an approximation, it is the definition of that distribution (the square of a standard
normal variable). A one-sided upper-tail test at confidence `c` rejects `H0(w)` — declares that
width `w`'s error is NOT explainable by noise alone, i.e. width `w` carries a real capacity deficit
at this row — iff `e_i(w) > σ_i² · χ²_{1,c}`.

Because `χ²₁` is literally `Z²` for a standard normal `Z`, this one-sided χ² test at level `c` is
**exactly equivalent** to a two-sided `z`-test on the underlying signed residual at the same level
(`(ε_i/σ_i)² > q ⟺ |ε_i/σ_i| > √q`) — numerically verified: `c=0.95` gives `χ²_{1,0.95}=3.8415`, matching two-sided `z=1.9600`. <!-- numcheck-ignore: mathematical identity χ²_{1,c} = z_{(1+c)/2}², verified via scipy.stats.chi2.ppf and scipy.stats.norm.ppf, not a ledger result -->
This equivalence is the reason `c=0.95` is chosen as the pre-registered default in §3 below — it
is the SAME nominal confidence level as MASTER Decision 33(i)'s already-adopted 2·SE noise-aware
band for the GLOBAL rule (`width.md` WSEL-20 block; "2" is a rounded convention for the exact
two-sided 95% normal quantile `z≈1.96`), so the programme's two noise-aware bands — one per-row,
one per-battery — sit at the same underlying confidence level by construction, differing only in
what each is anchored to (a single row's known `σ_i` here; a bootstrap standard error over many
rows there).

Applying `H0(w*_i)` (the row-best is itself noise-only) gives `e_i(w*_i) ≈ ε_i²` — the row-best
becomes an ESTIMATE of the noise realization itself, and `excess_i(w) = e_i(w) - e_i(w*_i)`
inherits the test directly: testing whether `e_i(w)` is ALSO explainable at the noise floor,
relative to the row-best's own achieved value rather than an assumed-zero reference, is exactly
`excess_i(w) <= σ_i² · χ²_{1,c}`.

### 2.5 Where the approximation enters — stated explicitly, not glossed

**(A1) The row-best-is-unbiased approximation.** The derivation above treats `e_i(w*_i)` as if it
were a pure noise draw (`bias_{w*_i}(x_i) := pred_{w*_i}(x_i) - f(x_i) ≈ 0`). This is NOT always
true — if even the grid's densest width carries residual approximation bias at some rows (the grid
does not reach the noise floor everywhere), the band is systematically too TIGHT at those rows
(rejects widths that are genuinely fine, because the reference point itself already overstates the
achievable floor) or too LOOSE (the reverse), depending on the sign and magnitude of that residual
bias. This is not a new, unverifiable assumption specific to this document — it is exactly the
`ratio_to_noise_floor` quantity this programme ALREADY measures and gates (the calibration
artifact's `anchor_ratio_to_noise_floor`, `width_wsel19.py`'s `_load_calibration_or_refuse`), so its
plausibility is checkable per decided cell from cached calibration numbers rather than assumed
blind.

**(A2) Paired, not independent, noise realizations.** `e_i(w)` and `e_i(w*_i)` share the SAME
observed `y_i` (hence the same realized `ε_i`) — they are not two independent noise draws. A fully
rigorous treatment of `excess_i(w)`'s distribution would track the joint dependence through both
widths' predictions (algebraically, `excess_i(w) = Δ_i(w)·(bias_i(w)+bias_i(w*_i)) - 2·Δ_i(w)·ε_i`
where `Δ_i(w) := pred_w(x_i) - pred_{w*_i}(x_i)`, a mean term that depends on the unobservable true
biases plus a noise term with variance `4·Δ_i(w)²·σ_i²` — NOT `σ_i²` alone). The rule adopted in
§2.3 sidesteps this joint algebra by invoking (A1) to treat `e_i(w*_i)` as a direct noise-floor
ESTIMATE rather than tracking its own bias term, which collapses the paired problem to the single-
sample χ² test in §2.4. This is the one place a genuine approximation, rather than an exact
distributional fact, enters the derivation. It is a DEFENSIBLE approximation on these toys (the
grid is dense and the calibration gate already screens for "best-fixed error near the noise
floor"), and its quality is exactly what the oracle-agreement evaluation in §5 tests empirically,
rather than assumes.

**(A3) Gaussianity and independence across rows.** Given directly by the toy generators
(`make_hetero`/`make_hetero3` add i.i.d. `N(0, σ²)` noise by construction) — an exact fact on toys,
not an approximation. A real-data adoption could not assume this without justification; see §7.

### 2.6 The one free choice

Every other quantity in §2.3 is either data (`e_i(w)`, `e_i(w*_i)`) or a KNOWN constant (`σ_i`,
fixed at the generator's true value per §3.7). The confidence level `c` is the sole remaining
degree of freedom — a statistical convention (how much of the noise distribution's own mass counts
as "explained"), not a domain constant tuned to this problem, unlike the flat rule's `τ=0.25`
(inherited from an unrelated experiment's tie-threshold, per `routing.py`'s own module docstring
provenance note).

## 3. Confidence-level candidates and the pre-registered default

Candidate values, standard statistical conventions (verified numerically,
`scipy.stats.chi2.ppf`/`scipy.stats.norm.ppf`, `scipy` already a project dependency — no new
library):

| `c` | `χ²_{1,c}` | equivalent two-sided `z` | convention |
|---|---:|---:|---|
| 0.68 | 0.9889 <!-- numcheck-ignore: chi-square(1) quantile, mathematical constant, not a ledger result --> | 0.9945 <!-- numcheck-ignore: normal quantile, mathematical constant, not a ledger result --> | "≈1 SE" reporting convention |
| 0.6827 | 1.0000 <!-- numcheck-ignore: chi-square(1) quantile, mathematical constant, not a ledger result --> | 1.0000 <!-- numcheck-ignore: normal quantile, mathematical constant, not a ledger result --> | exact "1-sigma" mass |
| 0.90 | 2.7055 <!-- numcheck-ignore: chi-square(1) quantile, mathematical constant, not a ledger result --> | 1.6449 <!-- numcheck-ignore: normal quantile, mathematical constant, not a ledger result --> | common one-sided screening bar |
| **0.95** | **3.8415** <!-- numcheck-ignore: chi-square(1) quantile, mathematical constant, not a ledger result --> | **1.9600** <!-- numcheck-ignore: normal quantile, mathematical constant, not a ledger result --> | **PRE-REGISTERED DEFAULT** |
| 0.99 | 6.6349 <!-- numcheck-ignore: chi-square(1) quantile, mathematical constant, not a ledger result --> | 2.5758 <!-- numcheck-ignore: normal quantile, mathematical constant, not a ledger result --> | conservative/strict bar |

**Default: `c = 0.95`.** Justification (stated once, not re-derived per cell): it is the same
nominal confidence level as the already-ratified 2·SE global-rule band (MASTER Decision 33(i),
`width.md` WSEL-20 block) — the two "noise-aware" bands this programme now has sit on one shared
convention. The evaluation protocol in §5-6 sweeps all four listed `c` values against the SAME
decided cells (a) already swept at its four `τ` values, mirroring (a)'s own sweep-not-single-point
design so that confidence-level sensitivity is measured, not assumed away.

## 4. Adoption gate — literal text, and a flagged ambiguity

**Literal text (`width.md` WSEL-22 block, part (b)):** "adopted only if (a) shows verdict
sensitivity OR (b) wins on generator-true oracle agreement at the decided levels." The harvest note
restates this AFTER (a) already fired: "adoption still gated on (a)-sensitivity OR oracle-agreement
per the clause above" (`width.md` WSEL-22 block, "(b) trigger evaluation (root)").

**⚠️ OPEN QUESTION FLAGGED FOR THE ADVERSARIAL READ, not resolved unilaterally here.** Read
literally, (a) has ALREADY shown verdict sensitivity (15/45 secondary findings flip somewhere in
its own four-tolerance sweep) — so the OR's first disjunct is already true, which would make (b)'s
adoption automatic regardless of how the σ-anchored band itself performs, and would make the
oracle-agreement disjunct never load-bearing. That reading conflicts with the harvest note's own
"adoption STILL gated" phrasing, written AFTER (a)'s result was known — if the gate were already
satisfied, "still gated" would be a strange thing to say. **This document's proposed resolution**
(to be confirmed or overridden at the adversarial read, not asserted as settled): distinguish the
WARRANT to author this spec (satisfied — (a)'s sensitivity is what justifies writing this
document at all, already recorded) from the ADOPTION decision itself, which requires the σ-anchored
band to clear one of two bars measured AGAINST ITS OWN PERFORMANCE, not against (a)'s generic
sensitivity finding:

- **(i) — the (a)-style check, applied to (b) itself:** does swapping the σ-anchored labels (at
  `c=0.95`) in for the flat `τ=0.25` baseline flip any DECIDED bake-off verdict — the exact
  `verdict_stability` computation (a) already performs
  (`automl_package/examples/capacity_ladder_results/WSEL22A/frozen.json` `verdict_stability`),
  re-run with the σ-anchored labels as the comparison point instead of another flat `τ`.
- **(ii) — generator-true oracle agreement:** does the σ-anchored band's routed labels agree MORE
  CLOSELY with the noise-free generator-true oracle (§5) than the flat 0.25 band's labels do, on
  the same decided cells.

**⚖️ ROOT RESOLUTION (adversarial read, 2026-07-23) — the WARRANT-vs-ADOPTION split is
CONFIRMED, with one correction to the proposed bars.** The literal already-satisfied reading is
REJECTED: (a)'s sensitivity finding is about the INCUMBENT band's fragility and cannot certify a
REPLACEMENT it never measured — it warranted authoring this document, nothing more. The proposed
bar (i) is also REJECTED **as an adoption criterion**: flipping decided verdicts shows the
σ-anchored band CHANGES outcomes, not that it improves them — a difference is not an improvement.
(i) is retained as a mandatory REPORTED impact assessment (what adoption would reopen, per the
block's implication map), never as adoption evidence. **Adoption (as the toy-default candidate;
§7 binds regardless) rests on (ii) ALONE — the generator-true oracle-agreement win per §5.3's
amended two-metric criterion at `c=0.95` — matching the block's own operative word: "WINS on
generator-true oracle agreement." A tie is NOT adoption: changing the constant behind certified
historical labels needs positive evidence, and (a) already showed the headline findings are
band-insensitive, so a tie leaves the incumbent in place with the comparison on record.**

## 5. The generator-true oracle-agreement protocol

### 5.1 The generator-true error table — existing machinery, zero retraining

`width_wsel19.py` already builds exactly this table for its own hidden-ness falsifier (§5.4/F4):
`_hetero_h(t)` (`width_wsel19.py:1154-1160`) is `nested_width_net.make_hetero`'s own noise-free
signal formula (`y_signal = np.where(x < 0, (0.5/r)*x, 0.5*sin(x))`,
`automl_package/examples/nested_width_net.py:174`), verbatim, evaluated on the untouched scalar
coordinate `t` (for the 1-D toy, `t` IS `x` directly — `make_hetero` has no rotation; for the d=2
"rotated-box" multi-feature toy, `t` is the underlying scalar this construction embeds into `d`
dimensions, returned by `width_wsel19_toys.make_report_split(seed, d, geometry)` as its 4th element
regardless of `d`, `width_wsel19_toys.py:274`, used generically at both the d=1 calibration block
(`width_wsel19.py:1287-1291`) and every multi-feature cell (`width_wsel19.py:1415-1429`)). The
generator-true error table is `_mf_error_table(models, x_report, _hetero_h(t_report), w_max)`
(`width_wsel19.py:1143-1151`) — the SAME per-width forward-pass stacking function `(a)`'s own driver
uses for the noisy table, scored against the noise-free target instead of `y_report`. **No new
machinery, no retraining**: this reloads the SAME cached, frozen model weights (a) already reads
and performs one additional forward pass per width.

### 5.2 The generator-true oracle label

**⚖️ ROOT AMENDMENT (adversarial read, 2026-07-23) — the original single-oracle design is
SUPERSEDED as structurally biased.** The draft scored BOTH bands against
`_cheapest_within_tolerance_labels(true_error_table, tolerance=0.25)` — a truth built WITH the
incumbent's own arbitrary constant. Under that design the flat band is scored against its own
ideal behaviour while the σ-anchored band is scored against a FOREIGN target: a candidate that
routes differently BY DESIGN loses agreement points even where its decisions are better
calibrated to the truth. The replacement is a symmetric two-metric protocol; each metric is
computable from the same two tables (noisy + generator-true) already in memory:

- **M1 — noise-robustness (self-referenced, per rule):** agreement of the rule's noisy-table
  routed label with the SAME rule's label on the generator-true table (flat-0.25 vs its own
  noise-free flat-0.25 labels — exactly the draft's original computation, now applied to each
  rule symmetrically; σ-anchored at `c` vs its own noise-free σ-anchored labels at the same `c`,
  well-defined on the true table: "which widths' true approximation deficit is within what the
  known noise would mask anyway"). Measures what noise costs each rule at recovering its own
  ideal decisions. **M1 alone is GAMEABLE by insensitivity** — a rule that accepts everything
  routes the cheapest width always and self-agrees perfectly while routing terribly — hence:
- **M2 — truth-tracking (rule-free common reference):** agreement of the rule's noisy-table
  routed label with the generator-true STRICT per-row argmin width
  (`true_error_table_report.argmin(axis=1)`, the per-row label form of the machinery already
  built for the hidden-ness falsifier — `width_wsel19.py:1143-1151`). Biased toward neither band;
  catches the insensitivity failure M1 cannot. (The draft's objection — that argmin answers a
  different question, ignoring acceptability semantics — is correct, which is why M2 is one of
  two metrics rather than the sole oracle.)

### 5.3 Agreement metric, aggregation, and the win criterion

Per decided cell: `agreement(labels) := mean_i [ labels_i == oracle_labels_true_i ]`. Compute
`agreement(labels_flat_0.25)` and `agreement(labels_sigma_c)` for each candidate confidence level
in §3. **Aggregation, reusing the existing convention rather than inventing a new one:** average
`agreement` across the decided cells' 3 seeds using the SAME `_mean_se` helper (a) already uses for
every other aggregate in its ledger, reporting mean ± SE per cell group (matching
`per_group_1d`/`per_group_d2`'s existing shape in
`automl_package/examples/capacity_ladder_results/WSEL22A/frozen.json`).

**Win criterion — ⚖️ ROOT-AMENDED (adversarial read, 2026-07-23; the draft's ">1 SE" margin is
replaced — Decision 33(i) makes noise-aware 2·SE bars the strand-wide convention, and a NEW 1·SE
bar would itself be an unsourced threshold choice):** the σ-anchored band at `c=0.95` WINS iff,
with every comparison computed as the per-seed PAIRED difference vs flat-0.25 and aggregated per
decided cell group:
1. it is better by more than 2·SE on at least ONE of M1/M2 (aggregate over decided groups), AND
2. it is not worse by more than 2·SE on the OTHER metric (aggregate), AND
3. NO decided cell group shows a loss beyond 2·SE on EITHER metric — (a) already showed oblique
   geometry is the fragile split; a pooled win hiding a group-level loss must not adopt.
Report the full per-group table at every candidate `c` regardless of the win verdict (the gate
needs "wins at the decided levels", plural, and the mixed-verdict branch in §10 consumes exactly
this table).

### 5.4 Which decided cells this applies to — verified present on disk

- **1-D (hetero, `make_hetero`).** Report-split coordinates come from
  `width_wsel19._get_or_build_sweep_cache`'s `x_report` (`width_wsel19.py:492`, `x_report` IS `t`
  for this toy — no separate `t_report` field needed, §5.1). The frozen per-width nets live at
  `automl_package/examples/capacity_ladder_results/WSEL6/_cache/sweep_tier1_seed{0,1,2}_w{1..12}.pt`
  — **verified on disk: exactly 12 state dicts per seed, 3 seeds, 36 total, all present** (checked
  directly, not assumed from the code path). `_load_cached_model`
  (`automl_package/examples/width_wsel6.py:430`) loads them; no training call anywhere in this
  path.
- **d=2 (the 5 decided triples from `frozen.json`'s `decided_d2_triples`: axis/seed0/n_train4000,
  axis/seed1/n_train1500, axis/seed2/n_train4000, oblique/seed1/n_train1500,
  oblique/seed2/n_train1500
  <!-- source: `automl_package/examples/capacity_ladder_results/WSEL22A/frozen.json` (`decided_d2_triples`) -->
  ).** Report-split `t_report` comes from `width_wsel19_toys.make_report_split(seed, d, geometry)`
  (`width_wsel19_toys.py:274`). The frozen per-width nets live at
  `automl_package/examples/capacity_ladder_results/WSEL19/_mf_cache/box_d2_{geometry}_seed{seed}_ntrain{n_train}_w{1..12}.pt`
  — **verified on disk: exactly 12 state dicts per decided triple, all 5 triples present** (checked
  directly). `_mf_model_cache_paths` (`width_wsel19.py:999`) resolves these paths;
  `_verify_mf_cache_complete` (`width_wsel22a.py:509`) already asserts their presence before (a)
  reads them, the exact same guard this evaluation reuses.
- **Yacht (WSEL-9's per-input real-data cells): EXCLUDED, not silently, not partially.** The
  frozen ledger's own `yacht_survival.mechanism_caveat` states the reason precisely: the real
  W-PERINPUT router trains on the JOINTLY-TRAINED dial net's own per-width predictions
  (`_run_dial_cell`'s `model.fit_router`), and neither that net's weights nor its SELECT-split
  error table were ever cached — only the INDEPENDENTLY-TRAINED W-SWEEP dedicated nets' tables
  exist on disk, a DIFFERENT model family (2.6-7.2× apart at matched middle widths, per the
  dial-vs-sweep protocol)
  <!-- source: `automl_package/examples/capacity_ladder_results/WSEL22A/frozen.json` (`yacht_survival.mechanism_caveat`) -->
  . There is no generator-true signal formula for yacht at all (it is a real regression dataset,
  not a synthetic generator with a known noise-free `h(t)`), so a "generator-true oracle" is not
  merely uncached for yacht — it is not DEFINABLE for yacht under this protocol's construction.
  This evaluation protocol therefore runs on 1-D + the 5 d=2 triples only, exactly as (a) itself
  was scoped, and this exclusion is restated rather than assumed silently, per the standing
  instruction. The separate remedy (cache the dial net's SELECT-split error table at next
  regeneration) is recorded at WSEL-6-R and is NOT this task's to build (out of scope, cheap,
  already tracked).

### 5.5 The per-row A1 diagnostic — ⚖️ ROOT ADDITION (adversarial read, 2026-07-23; resolves §12.3)

Cell-level calibration numbers are NOT sufficient evidence for A1, because A1's failure mode is
row-local (residual bias at SOME rows while the cell aggregate looks clean). The evaluation
therefore reports, per decided cell, the **quantile-coverage of the row-best against its claimed
null**: under A1, `e_i(w*_i)/σ_i² ~ χ²₁, so the empirical fraction of rows with
`e_i(w*_i)/σ_i² ≤ χ²_{1,c}` should match the nominal `c` at every swept confidence level (§3's
table). Report `(nominal c, empirical coverage)` pairs per cell — a mechanical function of the
noisy table and the known `σ_i` already in memory, zero extra forward passes. NON-GATING: this
diagnostic feeds §10's "(A1) empirically bad" branch (systematic under-coverage at a cell =
row-best carries residual bias there = the band is mis-anchored at that cell), turning that branch
from a vague suspicion into a numbered readout.

## 6. Evaluation protocol — relabel + refit, zero per-width retraining

Mirrors (a)'s own protocol exactly (`width_wsel22a.py`'s established pattern), adding one new axis
(confidence level) alongside (a)'s existing one (flat tolerance), never removing (a)'s four flat
points from the comparison:

1. For each decided cell (1-D: 3 seeds × 3 `n_sel` values × {constant, frozen_mlp, rule_mlp,
   xgboost} backends × {hard, blend} modes, matching (a)'s own `n_cells_1d=288`
   <!-- source: `automl_package/examples/capacity_ladder_results/WSEL22A/frozen.json` (`n_cells_1d`) -->
   ; d=2: the 5 decided triples × the same backend/mode grid, matching `n_cells_d2=480`
   <!-- source: `automl_package/examples/capacity_ladder_results/WSEL22A/frozen.json` (`n_cells_d2`) -->
   ), load the CACHED per-width error tables (a) already reads — no per-width net is retrained
   anywhere in this protocol, matching (a)'s own non-goal and this task's write-set exclusion of
   `WSEL19/`/`WSEL6/`.
2. Compute labels at each of (a)'s four flat tolerances (already on disk, reused not
   recomputed) AND at the σ-anchored band for each `c` in §3's table, using the same per-row
   `σ_true(x_i)` lookup the toy generator already provides (`HETERO_NOISE_SIGMA` for 1-D;
   region-indexed for any hetero3-descended construction, though the decided cells here are all
   `make_hetero`-family per (a)'s own scope).
3. Refit the SAME four router backends per cell at each σ-anchored labelling (no new backend
   type; `_fit_frozen_mlp_at`/`_fit_rule_mlp_at`/`_fit_xgboost_at`/`_fit_constant_at`
   (`width_wsel22a.py:172-395`) already take `tolerance` as a parameter — the σ-anchored driver
   swaps the labelling FUNCTION called inside those wrappers, not the wrappers themselves).
4. Extend (a)'s `verdict_stability` table with σ-anchored columns (same 20 boolean findings,
   evaluated at each `c`) and (a)'s `per_group` aggregates with σ-anchored group keys (same shape,
   new key component instead of a `τ` value) — a strict superset of (a)'s existing ledger schema,
   not a replacement.
5. Run the §5 oracle-agreement comparison (M1 + M2, §5.2 as amended) per decided cell, at each
   `c`, plus the §5.5 A1 coverage diagnostic per cell.
6. Freeze everything to a NEW ledger (this task's own write set;
   `automl_package/examples/capacity_ladder_results/WSEL22A/frozen.json` stays untouched — (a) is
   closed and cited, not edited).

## 7. Real-data caveat — BINDING, restated verbatim per the plan block

**The σ-anchored form uses the TRUE noise level — a toy-only luxury.** Real-data adoption requires
an estimated-noise variant whose estimator passes §3.7's no-learned-variance doctrine; until one
does, the flat band remains the real-data default and (b) is a toys-only instrument (`width.md`
WSEL-22 block, "(b) real-data caveat, BINDING on its spec"). **This spec pre-registers NO
estimated-σ variant — out of scope, by the same clause.** Nothing in §2-6 constructs, tests, or
gestures at an estimator for `σ_true(x)` on non-toy data; every quantity above is computable only
because these specific toy generators hand back their own exact noise level. A production
`FlexibleWidthNN.fit_router` call on real data would need `σ_true(x)` supplied by the caller from
some source this document does not specify — that plumbing question (how a per-row known-or-
estimated noise level would even reach the labelling call) is itself a real design cost of eventual
adoption, noted here so it is not discovered as a surprise later: the flat rule needs no such input
at all, while the σ-anchored rule requires threading a per-row noise quantity through wherever
`_cheapest_within_tolerance_labels` is invoked.

## 8. Scope boundary vs (b′) — restated so the two documents cannot be conflated

The blend (`capacity_ladder_t2`-style `blend_scores`/`blend_nll`, `routing.py:314-342`) computes a
probability-weighted expectation over ALL capacities in the grid — it never makes a single
discrete pick for a row, so "is this row's pick acceptable" is not a question that applies to it.
(b′)'s user ruling (`width.md` WSEL-22 block, verbatim: "Blend should have no threshold. Blend is
purely a function of the values on the out-of-sample data for each width") is unaffected by
anything in this document, and this document's σ-anchored band is never wired into `fit_soft` or
either blend-evaluation path. **HARD routing's labels only** — the ones `_cheapest_within_tolerance_labels`
produces and `fit()` (not `fit_soft()`) trains the router MLP against
(`routing.py:176-218` vs `:220-259`).

## 9. Driver contract (built later, by a different task — not this document)

A future implementation task builds `automl_package/examples/width_wsel22b.py` <!-- citecheck-ignore: forward reference -- Create target of a later task, not yet built by this spec-authoring task --> following (a)'s own
established discipline (`width_wsel22a.py`'s module docstring: "COPIED, not imported, with
provenance cited" for anything `routing.py`/`width_wsel19.py` cannot be depended on to expose
parametrized):

- **Per-cell CLI**, one JSON written to disk immediately per cell (never held in memory across
  cells) — the same "land every finding to disk the moment it is produced" discipline this
  strand's every driver already follows, with flags at minimum: `--geometry`, `--seed`,
  `--n-train` (d=2) or nothing extra (1-D, seed-keyed), `--backend`, `--mode`, `--confidence`
  (one of §3's candidate `c` values, or a flat `τ` for continuity with (a)'s existing sweep in the
  SAME driver), `--n-sel`.
- **`--summarize`** mode aggregates all landed per-cell JSONs into the extended `verdict_stability`
  / `per_group` / oracle-agreement ledger described in §6, exactly mirroring
  `width_wsel22a.py`'s own `summarize()` (`width_wsel22a.py:893`).
- **`--selftest`** exercises the σ-anchored labelling function against a synthetic tiny fixture
  (mirroring `width_wsel22a.py:948`'s `run_selftest`), asserting at minimum: (1) the row-best always
  passes its own band trivially; (2) a larger `c` never excludes a width the same row accepted at a
  smaller `c` (monotonicity of the band in confidence level); (3) at `σ→0` the band converges to
  "only the strict row-best is acceptable" (degenerate/no-slack limit); (4) byte-identical
  agreement with `_cheapest_within_tolerance_labels`'s existing flat-tolerance path when that path
  is invoked unchanged (regression guard against accidentally altering the (a) baseline while
  adding the new axis).
- **New function, one home, not a `routing.py` edit.** Following (a)'s exact precedent for
  functions `routing.py`/`width_wsel19.py` cannot be depended on to expose in the needed shape:
  the σ-anchored acceptability function is a NEW function living in the driver itself (not an edit
  to `routing.py`, which this task's non-goals forbid touching at all — see §11), importing
  `_cheapest_within_tolerance_labels`'s existing flat-tolerance path, `DistilledCapacityRouter`, and
  every backend-fit helper verbatim from `routing.py`/`width_wsel22a.py` rather than restating any
  of them.
- **Explicit non-goal, standing clause:** this driver's author never runs the grid. Per this
  programme's fan-out doctrine, the worker who builds this driver authors it and its `--selftest`
  only; the ROOT runs every cell, backgrounded, and re-verifies against disk before the ledger is
  called frozen.

## 10. Pre-registered outcomes and failure branches — a checkable done-state either way

- **σ-anchored band wins** (clears §4(i) or §4(ii) at `c=0.95`): recorded as the toy-domain
  replacement CANDIDATE for `DEFAULT_TOLERANCE`'s role in hard routing — adoption into
  `routing.py`'s actual default still requires the owning strand's own process (FP-5.b binds; this
  spec's non-goals forbid changing `routing.py` from THIS task regardless of the evaluation's
  outcome) and remains toys-only per §7 until an estimated-σ variant exists.
- **σ-anchored band loses on both §4 criteria at every pre-registered `c`:** recorded as a
  DONE-STATE, not a failure requiring further escalation — the flat 0.25 band is retained as both
  the toy and real-data default, with the comparison numbers now on record as the reason (rather
  than the previously-inherited, never-measured status quo). This is symmetric with (a)'s own
  framing: the sweep MEASURES, it does not presuppose an outcome.
- **Split verdict** (wins at some decided cells/confidence levels, loses at others — plausible
  given (a)'s own finding that oblique geometry is the fragile one): reported cell-by-cell, exactly
  as (a) reports its own 15/45 split; no unconditional adoption or rejection is claimed from a
  mixed result.
- **(A1)'s approximation is empirically bad** (oracle-agreement comparison shows the σ-anchored
  band performing erratically at cells where the calibration ledger's `ratio_to_noise_floor` is
  far from 1): recorded as a discriminating finding about WHEN the σ-anchoring is trustworthy, not
  papered over — §2.5 already flags this as the one place a real approximation lives, so this
  outcome confirms rather than contradicts the derivation's own stated limits.

## 11. Non-goals

No change to `routing.py` (its defaults, or adding anything to it) from this task or its eventual
driver — the new function lives in the driver, imported functions are imported, nothing in
`routing.py` is edited (FP-5.b binds; adoption goes through the owning strand, not this spec). No
per-width retraining anywhere in the evaluation protocol. No estimated-noise variant (§7). No
change to the blend's evaluation path (§8). No new cells beyond the 1-D + 5 d=2 triples (a) already
decided — this spec does not reopen which levels are "decided," only how they are relabelled. No
work on WSEL-22(c) (dormant, different task, triggered only by a WSEL-24 conviction). No resolution
of the yacht per-input caching gap (tracked at WSEL-6-R, a separate task).

## 12. Open questions for the adversarial read — ✅ ALL RESOLVED (root, 2026-07-23; verdict in §13)

1. **§4's adoption-gate reading.** Is the "WARRANT-vs-ADOPTION" split this document proposes the
   right resolution of the literal-OR ambiguity, or does the plan intend the literal reading
   (adoption already unconditional once (a) fired, oracle-agreement decorative)? This changes
   whether §5-6's oracle-agreement machinery is load-bearing for the GATE or purely informative.
2. **§5.3's win-margin rule** (">1 SE of the paired per-seed difference") is this document's own
   proposed default, not sourced from the plan block (which specifies only that a win must be
   MEASURED, not how). Confirm or replace before the driver is built, since it is what the
   pre-registered outcomes in §10 key off.
3. **§2.5(A1)'s scope.** The row-best-unbiased approximation is checked only indirectly (via the
   existing `ratio_to_noise_floor` calibration gate, at the CELL level, not per-row). A per-row
   diagnostic of A1's plausibility is not proposed here — worth deciding whether the evaluation
   protocol should report it, or whether cell-level calibration numbers are sufficient evidence.
4. **Whether `τ=0.25` is the right reference tolerance for the generator-true oracle label in
   §5.2**, versus a strict argmin oracle — this document picked the tolerance-matched form to
   isolate noise-vs-no-noise cleanly; the strict-argmin alternative (already computed elsewhere as
   `oracle_true_mean`) answers a related but different question and could be reported alongside
   rather than instead.

**Resolutions (root):** (1) WARRANT-vs-ADOPTION split CONFIRMED; adoption rests on the
oracle-agreement win ALONE — the draft's bar (i) is demoted to a reported impact assessment
(difference ≠ improvement); tie → not adopted (§4 as amended). (2) REPLACED — 2·SE per Decision
33(i), plus the no-group-loss clause (§5.3 as amended). (3) RESOLVED — per-row χ²-coverage
diagnostic ADDED, non-gating (§5.5). (4) RESOLVED — the τ=0.25-referenced common oracle was
structurally biased toward the incumbent; superseded by the symmetric M1/M2 two-metric protocol
(§5.2 as amended), with the draft's original computation surviving as flat-0.25's M1 leg.

## 13. Adversarial-read verdict (root, 2026-07-23) — **GO, with the §4/§5.2/§5.3/§5.5 root amendments applied**

Load-bearing claims re-verified at source by the root before this verdict: the flat rule's
multiplicative form and `argmax`-first-True mechanism (`routing.py:95-113`), `DEFAULT_TOLERANCE`
at `routing.py:77` (the `:75` drift in `width.md`'s block prose is REAL — queued for the
plan-hygiene batch, not fixed from this spec), the generator-true machinery
(`width_wsel19.py:1143-1160` — `_mf_error_table`'s docstring itself names the noise-free-target
use), `HETERO_NOISE_SIGMA = 0.05` (`automl_package/examples/nested_width_net.py:93`), the 36
tier-1 sweep state dicts under `WSEL6/_cache/` and the d=2 cache population under
`WSEL19/_mf_cache/` (108 state dicts on disk, superset of the 5 decided triples × 12;
`_verify_mf_cache_complete` re-guards at run time), and the WSEL22A ledger fields
(`tolerances_swept`/`baseline_tolerance`/`n_cells_1d`/`n_cells_d2`/`decided_d2_triples`/
`verdict_stability`/`yacht_survival.mechanism_caveat` all present as cited). The derivation's
χ²₁ mechanics and the honestly-flagged A1/A2 approximations are sound as stated; the yacht
exclusion ("not definable, not merely uncached") is correct and well-put. The four root
amendments: adoption gate resolved (win-only, tie-retains-incumbent), 2·SE margins with a
no-group-loss clause, the symmetric M1/M2 oracle protocol replacing the incumbent-flavoured
single oracle, and the per-row A1 coverage diagnostic. Next step per the wave line: driver task
builds `automl_package/examples/width_wsel22b.py` <!-- citecheck-ignore: forward reference -- Create target of the driver task --> against §9
<!-- citecheck-ignore: forward reference -- Create target of the driver task -->; the ROOT runs
the evaluation grid backgrounded.
