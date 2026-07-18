# Joint width+depth toy — FROZEN pilot bar spec (J0 Step 2)

Frozen BEFORE any pilot runs. Every number carries a one-line arithmetic/consistency justification tied
to `docs/joint_capacity/j1_arithmetic_check.py` §5 output or the depth toy's constants
(`automl_package/examples/depth_selection_toy.py`). No bar adjusted after its pilot runs.

Adjudication basis: read `joint_toy_design.md`, `j1_arithmetic_check.py`, `depth_selection_toy.py`,
`sinc_width_experiment.py` (`_deploy_bar_mse`/`_cheapest_within_tolerance_labels`); re-ran the arithmetic
check and re-derived every bound below.

---

## 1. Constants table (paste into the toy's module constants + RESUME Decisions log)

| Constant | Value | One-line grounding |
|---|---:|---|
| `WIDTH_READ_LADDER` | `(16, 32, 48, 64)` | 4 rungs = `K_MAX=4` track levels; `64/4 = 16` state units/track; minimal ladder resolving the 4-level demand `A∈{1..4}` one-to-one (admits ρ=1.0). |
| `W_BAR_SUBPOP` | `A == 4` | The free-ride question ("can a 16-wide read hold ALL A tracks") is only diagnostic at maximal A; A=1 cannot separate free-ride from real. |
| `W_BAR_METRIC` | all-active-correct @ fixed `T=10`, held-out | At A=4 this is all-4-correct; depth held full so only width varies. |
| `W_BAR_GAP_MIN` | `0.30` | `acc(w=64,A=4) − acc(w=16,A=4)`. Real-dial gap ≈ 0.60 (floor) to 0.64; free-ride gap ≈ 0; 0.30 bisects the bimodal split, ~10× the ~0.03 noise floor. (arithmetic §2 below) |
| `W_BAR_RHO_MIN` | `0.7` | ρ_spearman(deployed-w, A); reuses depth `S3_ROUTER_SPEARMAN_MIN=0.7` verbatim; 4 rungs × 4 A-levels admit ρ=1.0. |
| `S1_FIT_ACC` | `0.90` | Reused verbatim from depth `S1_FIT_ACC`; per-**track** held-out acc at `(w=64, T=10)`, every (A,T*) cell. |
| `S2_KNEE_FRACTION` | `0.95` | Reused verbatim; depth knee, per-**track** (see FLAG-2). |
| `S2_CEILING_A5` | `0.35` | `Bayes(g=2)+0.10 = 0.25+0.10`; re-derived from A5 arithmetic (per_track g=2 = 0.2500). Single-track ⇒ per-track D-bar (FLAG-2). |
| `BAYES_G_FOR_CEILING` | `2` | Reused; T*−2 read point, g=2 unread letters. |
| `S3_ROUTER_SPEARMAN_MIN` | `0.7` | Reused verbatim; both W-bar-ρ and D-bar-ρ and X-bar router-quality. |
| `DELTA_TIE` | `0.25` | Reused verbatim (`sinc_width_experiment.DELTA_TIE`); cheapest-within-tolerance for both routers. |
| `COMPUTE_PROXY` | `w * T` | §6 "mean w·T"; per-input executed compute = routed_w × routed_T. |

Grid for the routers: `w∈{16,32,48,64} × T∈{2,4,6,8,10}` = **20 (w,T) columns**, ordered ascending by
`w·T`, **tie-break (w asc, then T asc)** (FLAG-4).

---

## 2. W-bar gap — the make-or-break arithmetic (checkable)

Measured on the **A=4 held-out subset**, all-4-correct, fixed full `T=10`:

- **At w=64 (full):** S1 forces per-track ≥ 0.90 on the A=4 cells. All-4-correct ≥ `1 − 4·(1−0.90) = 0.60`
  (union bound, dependence-free floor); independence estimate `0.90^4 = 0.656`.
- **At w=16 (real dial):** if a 16-wide read genuinely cannot hold 4 independent A5 elements, ≥1 track
  collapses to ~1/60 chance ⇒ all-4-correct ≤ `0.90^3·(1/60) ≈ 0.012`. **Robustness:** holding even 3 of 4
  tracks still zeroes all-4-correct — so ANY sub-4 packing gives ≈0, only a TRUE free-ride (holds all 4 at
  w=16) gives ≈acc64. The A=4 gap is therefore **bimodal: ≈0 (free-ride) or ≈0.60–0.64 (real)** with
  nothing between.
- **Predicted real gap ≈ 0.60 − 0.012 ≈ 0.59;** free-ride gap ≈ 0. **`W_BAR_GAP_MIN = 0.30`** sits in the
  empty middle and is ~10× the held-out noise floor (SE(all-4-correct) ≈ √(0.66·0.34/1500) ≈ 0.012 at
  ≥1500 A=4 held-out; paired across the two widths ⇒ smaller). 

**W-bar = PASS iff** `gap ≥ 0.30` **AND** `ρ_spearman(deployed-w, A) ≥ 0.7`. Fail ⇒ J-1's width dial is a
free-ride ⇒ adopt J-2 (whose block-width mask forces the collapse by construction, so it is expected to
pass its own W-bar). This is the designed J-1→J-2 fork (`joint_toy_design.md` §3, §4).

---

## 3. Width read-ladder — RATIFIED `(16, 32, 48, 64)`

- 4 rungs = `K_MAX=4`; `16 = 64/4` state units per track slot.
- The width **demand has exactly 4 levels** (`A ~ U{1..4}`), so 4 rungs is the **minimum that resolves A
  one-to-one** — a perfect dial maps A=k → w=16k, giving ρ=1.0. Fewer rungs cannot reach ρ=1 with 4
  A-levels; a **finer ladder buys nothing** (no new optimal operating points beyond the 4 distinct
  demands) and dilutes each per-width head's training data. **4 rungs is optimal, not merely sufficient**;
  ρ≥0.7 is comfortably recoverable. Ratify as proposed — do **not** add rungs.
- Note: the info-floor (A·log₂60 = 5.9/11.8/17.7/23.6 bits for A=1..4) does **not** ground the ladder — 16
  real tanh units can encode 23.6 bits, which is exactly why the width dial's reality is an *empirical*
  W-bar measurement (MOD-2), not an info-theoretic given. The ladder is grounded in **track structure**
  (4 slots), not bit capacity.

---

## 4. D-bar / S1 consistency — VALIDATED, with one required refinement (FLAG-2)

Reusing the depth toy's knee is well-posed on the joint toy **provided D-bar is measured PER-TRACK at full
width w=64**, not on the per-input all-active-correct metric:

- **Why per-track:** `S2_CEILING_A5 = Bayes(g=2)+0.10 = 0.35` is a **single-track** Bayes ceiling. The
  per-input all-active ceiling would be `Bayes(2)^A + margin = 0.25^A + 0.10` (A-dependent) — the reused
  constant would be miscalibrated. Per-track keeps the depth toy's exact ceiling valid verbatim.
- **Why full width:** at w<64 an insufficient-width read depresses accuracy even for T≥T*, confounding the
  depth knee. Holding w=64 (where S1 guarantees ≥0.90 per-track) isolates the depth axis cleanly.
- **Coherence:** all active tracks share the SAME T* by construction (the orthogonality precondition,
  `j1_arithmetic_check.py:62`), so the per-input knee is coherent — every active track flips to correct at
  T=T* together. Per-track is the clean measurement; per-input all-active-correct would only rescale it.
- Ceiling read points T*−2 ∈ {4,6,8} all lie in the T-ladder {2,4,6,8,10}; g=2 ⇒ Bayes 0.25 for every T*.
- **D-bar (frozen):** at w=64, per-track top-1 acc pooled per T* stratum: knee `acc(T=T*) ≥ 0.95·acc(T=10)`;
  ceiling `acc(T=T*−2) ≤ 0.35`; `ρ(deployed-T, T*) ≥ 0.7`. S1 unchanged: per-track ≥0.90 at (w=64,T=10),
  every (A,T*) cell.

---

## 5. X-bar well-posedness — VALIDATED, with implementation notes (FLAG-3/4/5)

§6's rule is sound and unambiguous once read fully: **acc(router) ≥ acc(best-fixed) within 2·SE AND
mean-compute(router) < compute(best-fixed)**, plus the same test pairwise against each marginal. This is
exactly `_deploy_bar_mse`'s `accuracy_preserved ∧ compute_saved`, with error = 1−acc and compute = w·T.

- **Labeling:** cheapest-within-tolerance (`_cheapest_within_tolerance_labels`, δ=0.25) over the 20 columns
  **ordered ascending by w·T** — `argmax`-first-True = cheapest-qualifying only if columns are compute-sorted.
- **Marginals are non-degenerate** (conditioned on W-bar+D-bar passing): width-only@T=10 always pays T=10
  (joint saves depth on the ~2/3 of inputs with T*∈{6,8}); depth-only@w=64 always pays w=64 (joint saves
  width on low-A inputs). The only degeneracy — a "joint" router collapsing to a depth-only marginal at
  w=16 — occurs **only if the width dial is a free-ride**, which W-bar already kills. So the **gate order
  S1 → W-bar → D-bar → X-bar must be enforced** (FLAG-7): X-bar is meaningless on a toy that failed W-bar.
- **X-bar = PASS iff** router beats the **val-selected** best-fixed (non-hindsight, FLAG-3) AND strictly
  beats **both** marginals, all on the accuracy-preserved(2·SE) ∧ strictly-less-mean-w·T basis.

---

## FLAGS (orchestrator/builder must act before/during the pilot)

1. **W-bar was underspecified in §4** ("acc(w_max)−acc(w_min)", no population). Frozen here to the **A=4
   subpopulation, all-4-correct** — the principled free-ride diagnostic, not cherry-picking (A=1 cannot
   discriminate). Paste the refined definition into the module + RESUME.
2. **D-bar MUST be per-track at w=64**, NOT per-input all-active-correct — otherwise `S2_CEILING_A5=0.35`
   (single-track) is miscalibrated (correct per-input ceiling would be `0.25^A+0.10`). Also holds width
   full so it doesn't confound the depth knee.
3. **`_deploy_bar_mse` keys compute on a 1-D level (`col+1`)** — for the 2-D joint it must be fed explicit
   `compute = w·T` per column and best-fixed over all 20 columns; use the **val-selected** best-fixed
   (`_deploy_bar_mse_valselected` pattern) as the honest comparator. Implementation adaptation, not a new
   bar.
4. **Deterministic tie-break in the w·T ordering** (ties confirmed: w·T=128 is 3-way {16·8, 32·4, 64·2};
   64/96/192/256/384 are 2-way). Secondary sort (w asc, then T asc) so cheapest-within-tolerance labels
   are reproducible across runs/seeds.
5. **Define the marginal-beat test explicitly:** joint mean-compute strictly < each marginal's mean-compute
   at accuracy no worse (within 2·SE), evaluated joint-vs-each-marginal pairwise (same shape as the
   best-fixed deploy bar).
6. **Bump pilot n:** the ~0.03 W-bar noise floor assumes ≥500 held-out/cell (⇒ total n ≳ 12000 with A×T*
   uniform; ≥1500 in the A=4 subset). The arithmetic check ran at n=4000 (min cell 303) — below this. Set
   the pilot to ~1000/cell (§4 already says "~1000/cell") ⇒ n≈12000.
7. **Enforce gate order S1 → W-bar → D-bar → X-bar** (see §5): a failed W-bar makes X-bar meaningless.

---

## Summary for the orchestrator

- **Width ladder:** `(16,32,48,64)` ratified — 4 rungs = 4 track levels = minimal one-to-one resolver of
  `A∈{1..4}`; do not add rungs.
- **W-bar gap:** `0.30` on the **A=4** all-4-correct subset at T=10. Arithmetic bound: real gap ≈ 0.60–0.64
  (S1 floor 0.60 minus ≈0.012 collapse), free-ride gap ≈ 0, split is bimodal, 0.30 bisects it at ~10× the
  noise floor. Companion `ρ(deployed-w, A) ≥ 0.7` (reuses depth S3).
- **Act-first FLAGS:** (2) D-bar per-track@w=64, (3) 2-D compute in the deploy helper, (4) deterministic
  w·T tie-break, (6) bump pilot n to ~12000. These bind the build before the J-1 pilot runs.
