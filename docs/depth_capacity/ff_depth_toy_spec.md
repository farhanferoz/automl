# Feedforward-depth pilot — design spec (F5a)

*Strand: FlexNN core (`docs/plans/capacity_programme/flexnn-core.md`, Task F5). This spec is authored
before any build (F5b is a separate, gated task). It is delivered for adversarial review; the
reviewer fills `## Review verdict (pending)` at the end — the author renders no GO/PARK call here.*

*Every architectural constant and every parameter count in this document was verified by running the
existing repo machinery (`automl_package/examples/depth_composition_toy.py`) — see the "Verified
against code" callouts. Nothing here is asserted from memory.*

---

## 0. The sharper question (why this pilot exists)

The depth strand already **certified** (`docs/depth_capacity/verdict_per_input_depth.md`, G-DEPTH =
PASS) that a **weight-shared recurrent** block serves per-input variable depth on a task where depth
is provably irreducible to width (A5 word composition), and that a distilled router selects that
depth per input. That result rests on a network that is **both** weight-tied **and** fed one letter
per step (a `RecurrentComposer`).

This pilot asks a **different, sharper** question, in two parts:

1. **Is depth non-substitutable in a PLAIN FEEDFORWARD net too** — a deep MLP on the whole flattened
   word, untied weights — not only in the weight-tied recurrent form? (This is *the user's original
   claim*.)
2. **Which ingredient is load-bearing** — weight-tying, per-step input feeding, or both? We isolate
   this with a 2×2 grid that turns each ingredient on and off independently.

Both outcomes are reportable and pre-anticipated (§8). The pilot is a **substrate** question only —
no selection/router logic (that half is already certified; §12 non-goals).

---

## 1. Generative math — A5 word composition (self-contained)

*Lifted and made explicit from `depth_composition_toy.py` (`build_group`, `word_product`,
`make_word_data`); the group axioms, generator span, and product correctness are checked by that
module's `--selftest`.*

**The group.** `A5` = the alternating group on 5 points = the even permutations of `{0,1,2,3,4}`,
under composition. `|A5| = 60`. A5 is the smallest non-solvable group; by **Barrington 1989**
(width-5 branching programs over a non-solvable group capture NC¹ exactly) width provably cannot
substitute for sequential composition depth on the A5 word problem. It is nonetheless GD-learnable
(**Liu et al. 2022**, arXiv:2210.10749), and the difficulty knob is the **number of sequential
composition steps** = the word length `L` — a natural per-input depth dial.

**The alphabet — 4 involutions.** The word alphabet is 4 fixed generators of A5, each an **order-2
involution** (a double transposition), stored in `depth_composition_toy.py::A5_GENERATORS` in
one-line notation:

| id | one-line | cycle form | order |
|---:|---|---|---:|
| g0 | (0,2,1,4,3) | (1 2)(3 4) | 2 |
| g1 | (0,3,4,1,2) | (1 3)(2 4) | 2 |
| g2 | (0,4,3,2,1) | (1 4)(2 3) | 2 |
| g3 | (1,0,2,4,3) | (0 1)(3 4) | 2 |

Each is a product of two disjoint transpositions ⇒ even ⇒ in A5, and self-inverse (`g·g = e`). Their
BFS-closure under right-multiplication is **all 60** elements (verified: selftest
`generators_span = 60/60`). Rationale for involutions rather than five-cycles is recorded in
`depth_composition_toy.py` (docstring on `A5_GENERATORS`) and the D8 verdict §10.1: involutions
restore short identity relations so early-commitment strata are constructible at `L=10`, while the
group stays non-solvable and all 60 classes remain reachable.

**Words and their product.** A word is a length-`L` sequence of generator indices
`w = (i_1, …, i_L) ∈ {0,1,2,3}^L`. Its label is the group product folded **left-to-right**:

```
P_0 = e (identity);   P_t = g_{i_t} ∘ P_{t-1}   for t = 1..L;   label(w) = P_L ∈ {0,…,59}
```

where `a ∘ b` applies `b` first then `a` (matches `word_product`: `P_t = mult[g_{i_t}, P_{t-1}]`).
Reading the word `g_{i_1} g_{i_2} … g_{i_L}` as "apply `g_{i_1}` first." The task is **60-way
classification** of `label(w)`; chance = 1/60 = **0.0167**.

**Input encoding.** Each position `i_t` becomes a one-hot vector `e_{i_t} ∈ ℝ⁴`; the word becomes the
**flattened concatenation** `x = [e_{i_1}; …; e_{i_L}] ∈ ℝ^{4L}`. At `L = 10` the input is
**40-dimensional** (`in_dim = seq_len · n_gen = 10 · 4 = 40`).

> ⚠️ **Verified against code — corrects the brief.** The F5a brief states "50-dim." The as-built
> input dimension for A5, L=10 is **40** (`make_word_data(A5, 10, ·)` returns `x` of shape
> `(n, 40)`; the D8 verdict §10.2 S4 independently records "in_dim 40"). This spec uses **40**. There
> is no configuration of this toy that yields 50 (4 generators × 10 positions = 40). The reviewer
> should treat 40 as authoritative.

**Held-out split (generalization).** The pilot fixes `L = 10`. Total possible words = `4¹⁰ =
1,048,576`, which exceeds the toy's `MAX_ENUM = 20,000` enumerate/sample cap, so `make_word_data`
**uniformly samples 20,000 words** (`rng.integers`, i.e. *with replacement* — see the accuracy note
below) and splits them `DEFAULT_TRAIN_FRAC = 0.5` into **10,000 train / 10,000 val** (verified:
`x_tr (10000,40)`, `x_val (10000,40)`). Train and val are **near-disjoint**, so **held-out accuracy
measures compositional generalization to unseen words** — a memorizer scores at chance on val; only a
net that learned the composition *algorithm* generalizes.

> ⚠️ **Verified against code — with-replacement sampling, <1% train/val overlap.** `rng.integers` does
> **not** dedupe, so the 20,000 words are not strictly distinct: measured ~178 (seed 0) / ~191 (seed 1)
> duplicate rows, of which **86 (0.86% of val, seed 0) / 96 (0.96%, seed 1)** words appear on both
> sides of the split. This leakage is **<1%, uniform across every arm** (all cells share one
> `make_word_data` pipeline), so it neither biases the *relative* grid comparison nor is material at
> the 0.90 bar. F5b **must print the observed train/val word-overlap count** into each JSON as a sanity
> check; if it ever exceeds a few percent (it will not at these sizes) the rung is flagged. Not
> deduped by design, to keep all four cells on the byte-identical data pipeline.

---

## 2. Starvation arithmetic (is failure ever about data, not architecture?)

The certified strand excludes any rung where a method cannot generalize for lack of data (D8/D5
ruling R1: at ℓ=4 for S5, `4⁴ = 256` words split 128/128 against 120 classes ≈ **1 train example per
class** — unlearnable by construction, so excluded from all bar reads).

For **this** pilot the arithmetic is comfortably clear of that floor:

- Train words = **10,000**; classes = **60** ⇒ **≈ 167 train words per class on average** (167× the
  R1 starvation floor of ~1/class). No rung is data-starved.
- **Fit feasibility is already established** for the tied/recurrent arm: the D8 anytime substrate on
  **A5, L=10** reached held-out accuracy ≥ 0.90 per stratum with ~3,000 words/stratum
  (verdict §10.2 S1). This pilot's 10,000 uniform train words meet or exceed that.
- The **open** empirical question is fit feasibility for the *untied-flat plain MLP* arm — that is
  precisely what the pilot measures, not an artifact to be assumed either way.

> **Pre-registered data sanity check (F5b must print, not assume):** the label distribution over the
> 60 A5 elements for random length-10 words over these 4 involutions is assumed near-uniform (random
> walk on a connected generating set mixes toward Haar/uniform). F5b must dump per-class train/val
> counts into each JSON; if any class is empty or < ~30 examples, that rung is flagged as
> partially-starved and read with the R1 caution, **not** silently passed. This is a check, not an
> expected failure.

---

## 3. The 2×2 architecture grid + depth ladder

All arms: hidden width `w = 64`, `n_classes = 60`, `n_gen = 4`, `L = 10`, `in_flat = 40`, Tanh
activations, full-batch Adam, convergence-gated (§7). The grid isolates **two independent factors** —
weight-tying (rows) and input schedule (columns):

|                | **flat input** (all 10 letters visible at layer 0) | **per-step input** (letter *t* fed at layer *t*) |
|---|---|---|
| **untied weights** | **Cell 1 — plain deep MLP** *(the user's original claim)* · `build_narrow_clf` · **BUILD** | **Cell 2 — untied unrolled stack** · 10 distinct blocks · **BUILD** |
| **tied weights**   | **Cell 3 — tied stack on flat input** · shared block iterated · **BUILD** | **Cell 4 = `RecurrentComposer`** · shared block ×10 · **BUILD (2-seed confirm, same single-readout protocol as Cells 1–3; anytime ≥0.90 cited as corroboration)** |

plus the **certified wide-shallow control** (depth-1 MLP, `build_wide_shallow_clf`), param-matched
per §5 — this is the width-substitution null and already exists in the repo (verdict §3, D7).

**Depth ladder.**
- **Flat-input arms (Cells 1 and 3):** depth `d ∈ {4, 7, 10}`, width 64. The ladder sweeps how many
  hidden layers (Cell 1) / shared-block iterations (Cell 3) are stacked. `d ≤ 10` caps the ladder at
  the per-step arms' intrinsic depth (`L = 10`) and at the MOD-1 GD-trainable wall (§12).
- **Per-step arms (Cells 2 and 4):** depth is **structurally pinned at `d = L = 10`** — letter `t` is
  fed at step `t`, so there are exactly `L` steps. These arms are **not** swept over `{4,7,10}`;
  feeding 10 letters through fewer than 10 per-step layers would require redesigning the input
  schedule, which the non-goals forbid (§12). *(Interpretation the author had to make; see §11.)*

> **"Depth `d`" semantics (verified against `build_narrow_clf`).** `build_narrow_clf(d, …)` produces
> `d` Tanh-activated hidden layers (1 input layer + `d−1` middle layers) followed by 1 linear
> readout — i.e. `d + 1` linear layers total. `d = 10` (flat) ≈ 10 nonlinear transformations ≈ the
> `L = 10` sequential steps of a per-step arm. The ladder and the "some `d ≤ 10`" bar (§6) are read
> in these units.

---

## 4. Concrete architecture definitions (what F5b builds)

Two cells already exist in the repo; two are new. All four are pinned here numerically so the
confound ledger (§5) is exact.

**Cell 1 — untied-flat (plain deep MLP): reuse `build_narrow_clf(d, 64, 40, 60)` verbatim.**
`Linear(40,64)→Tanh → [Linear(64,64)→Tanh]×(d−1) → Linear(64,60)`. Each layer has its **own** weights
⇒ parameter count **grows with `d`** (this is the confound §5 controls for).

**Cell 4 — tied-perstep (`RecurrentComposer`): existing module, BUILD a 2-seed confirm run.**
`state_0 = 0`; per step `state = tanh(block([state, onehot(g_{i_t})]))` with a **shared** 2-layer
`block = Linear(64+4,64)→Tanh→Linear(64,64)`; then `readout = Linear(64,60)`. `d = L = 10` by
unrolling. Corroborating prior evidence: **D8 verdict §10.2 S1**, A5 L=10 full-depth held-out acc
≥ 0.90 on both seeds (seed0 {6:0.949, 8:0.957, 10:0.946}; seed1 {6:0.938, 8:0.931, 10:0.901}).

> ✅ **Reviewer ruling (was "cite only"; now a mandatory confirm run) — see §11 item 2.** The cited A5
> L=10 result is the **anytime** substrate of `depth_selection_toy.py`: the same `RecurrentComposer`
> module but trained with a **shared readout on the running prefix product at 5 exits
> T∈{2,4,6,8,10}** — a *richer supervision signal* than the plain single-readout-on-full-word-product
> that Cells 1–3 use (verified: no plain-`RecurrentComposer`-A5-L10 JSON exists in `D_TOY_PROBES/`,
> only `depth_selection_*` files). Importing that number would confound the tied+perstep corner with a
> **training-supervision difference**, breaking the grid's architecture-only attribution (§7).
> Therefore Cell 4's grid entry MUST come from a plain single-readout `RecurrentComposer` trained
> under the **same `train_clf` protocol as Cells 1–3** — `run_pilot --net recurrent --group a5
> --seq-len 10`, 2 seeds, which the existing module produces with no new code. The anytime ≥0.90
> numbers remain as corroboration only. (Mechanical fix, author pre-authorized; bars untouched.)

**Cell 2 — untied-perstep (untied unrolled stack): NEW.** Architecturally identical to Cell 4 **but
with `L = 10` distinct blocks** (weights not shared across steps), so tying is the *only* difference
from Cell 4: `blocks = [Linear(64+4,64)→Tanh→Linear(64,64)] × 10` (distinct), `readout =
Linear(64,60)`; `state_0 = 0`, letter `t` fed at block `t`, `d = L = 10`.

**Cell 3 — tied-flat (tied stack on flat input): NEW.** Flat input enters once at layer 0 via an
input projection, then a **shared** block is iterated `d` times (tying is on; input is flat):
`inp = Linear(40,64)`; shared `block = Linear(64,64)→Tanh→Linear(64,64)` applied `d ∈ {4,7,10}`
times to the state; `readout = Linear(64,60)`. Because the block is shared, **parameter count is
constant in `d`**. *(Design choice: the block iterates on state only — the flat word is injected at
layer 0, matching "all letters visible at layer 0"; it is not re-injected each step. Flagged in §11.)*

---

## 5. Confound ledger (MANDATORY — the crux of the reviewed-spec requirement)

**The confound.** For **untied** arms (Cells 1, 2) parameter count **grows with depth** — each layer
has its own weights. A naive "deeper untied net generalizes better" finding is therefore confounded
with "more parameters." FF-CLAIM is only a genuine **depth-vs-width** test if each arm that claims
"depth beat width" is read against a **param-matched wide-shallow** control that fails. The tied arms
(Cells 3, 4) are the opposite regime — near-constant, *low* param count — so if a tied arm generalizes
where a param-matched width-net fails, the non-substitutability claim is even stronger.

**Exact parameter counts (verified by `count_params` on the actual modules):**

| Cell | Arch | `d` | Params | Grows with `d`? |
|---|---|---:|---:|---|
| 1 untied-flat | plain deep MLP | 4 | **19,004** | yes |
| 1 untied-flat | plain deep MLP | 7 | **31,484** | yes |
| 1 untied-flat | plain deep MLP | 10 | **43,964** | yes |
| 3 tied-flat | proj + shared block ×`d` | 4 / 7 / 10 | **14,844** | **no (constant)** |
| 2 untied-perstep | 10 distinct 2-layer blocks | 10 | **89,660** | n/a (fixed d=10) |
| 4 tied-perstep (`RecurrentComposer`) | shared block ×10 | 10 | **12,476** | n/a (fixed d=10) |

**Param-matched wide-shallow controls.** The width-substitution null is the certified depth-1 MLP
`build_wide_shallow_clf(W, 40, 60)`, whose parameter count is exactly `101·W + 60` (verified;
consistent with the D7 precedent's `161·101 + 120 = 16,381` arithmetic for S5). For each stack arm we
pre-register the width `W = round((P − 60) / 101)` that matches its parameter budget (all within
±0.3%, verified):

| Matches | Arm params `P` | Control width `W` | Control params | Δ |
|---|---:|---:|---:|---:|
| untied-flat `d=4`  | 19,004 | **188** | 19,048 | +0.23% |
| untied-flat `d=7`  | 31,484 | **311** | 31,471 | −0.04% |
| untied-flat `d=10` | 43,964 | **435** | 43,995 | +0.07% |
| tied-flat          | 14,844 | **146** | 14,806 | −0.26% |
| tied-perstep       | 12,476 | **123** | 12,483 | +0.06% |
| untied-perstep     | 89,660 | **887** | 89,647 | −0.01% |

**Reading rule (pre-registered).** Any cell that reaches held-out ≥ 0.90 has its "depth beat width"
sub-claim read against the wide-shallow control **param-matched to that same cell** (from the table).
The **headline FF-CLAIM** (§6) is carried specifically by untied-flat vs. its matched control(s)
{188, 311, 435}. All six control widths are frozen here **before any run**, so the width read is never
chosen post-hoc to fit the depth result (Decision 9).

**Second confound, pre-registered:** untied-perstep (Cell 2, 89,660 params) is the most
over-parameterized arm by far. If Cell 2 generalizes, that is **not** evidence for depth per se until
it beats its `W = 887` matched control — hence that control is pre-registered above and is mandatory
for any Cell-2 claim. (Cells 3 and 4 need no such caution — they are the *lean* arms.)

---

## 6. Pre-registered bars (frozen numerically; NOT adjustable after any run — Decision 9)

Metric: cross-entropy classification, read on **held-out (val) word accuracy**. Thresholds inherit
the certified graded-pilot values (`depth_composition_toy.py`: `FIT_ACC = 0.90`, `STALL_ACC = 0.60`;
G-DEPTH bars G1/G2, verdict §1).

- **FIT threshold = 0.90.** An arm "fits" iff held-out acc ≥ **0.90**.
- **STALL threshold = 0.60.** A width control "stalls" iff held-out acc ≤ **0.60**.
- **Chance = 0.0167** (1/60).

**FF-CLAIM (the headline) PASSES iff both hold:**
1. the **untied-flat** arm (Cell 1) reaches held-out **≥ 0.90 at some `d ≤ 10`**, **and**
2. the wide-shallow control **param-matched to that winning depth** (§5 table: 188 / 311 / 435 for
   `d = 4 / 7 / 10`) stays **≤ 0.60**.

**Non-substitutability falsifier (the "hidden-ness" bar).** FF-CLAIM is *falsified* if the
param-matched width control also reaches ≥ 0.90 (width substitutes ⇒ depth is not load-bearing). As
corroboration, F5b reports **train vs. val** for every width control: the certified failure mode is
**memorization** (train acc ≥ 0.90 while val ≤ 0.60), exactly as verdict §2.1 / §3 found for the S5
width-101 control (val 0.447, train ≥ 0.93). A width control that stays low on *both* train and val
is under-fit, not a clean width-substitution failure, and must be flagged rather than counted as a
stall.

**Convergence-gated (hard gate).** Every reported number must come from a **trustworthy**
(non-diverged) trajectory per `automl_package/examples/convergence.py` (full-trajectory rule +
`diverged` sub-flag, MASTER Decision 9). No number is read from an endpoint alone: F5b must record and
show each arm's val-CE trajectory, and any `trustworthy = false` / `diverged = true` cell is
quarantined, not reported. (This binds the [[feedback_check_loss_trajectory_before_concluding]] rule.)

**Seeds.** ≥ **2 seeds** per built cell; a bar is read as passed only if it holds on ≥ 2 trustworthy
seeds.

---

## 7. Attribution read — the ordered finding (not just pass/fail on one cell)

Report **which of the 4 cells generalize (held-out ≥ 0.90), ordered by held-out accuracy**, and map
the ordering onto the load-bearing ingredient. The grid is designed so the *pattern* of passes
identifies the cause:

| Observed pattern | Attribution |
|---|---|
| Cell 1 (untied-flat) passes | **Depth is non-substitutable in a plain feedforward net** — FF-CLAIM PASS (§8a). |
| Cell 1 fails, Cell 3 (tied-flat) passes | **Weight-tying is load-bearing**; flat input suffices once weights are tied. |
| Cells 1 & 3 fail, Cell 2 (untied-perstep) passes | **Per-step input feeding is load-bearing**, not tying (but check the `W=887` control — Cell 2 is the over-parameterized arm, §5). |
| Only Cell 4 passes | **Both** tying **and** per-step feeding are **jointly required** (the certified `RecurrentComposer` is minimal). |

The attribution is reported as this ordered list with each cell's `(d, held-out acc, params, matched
control acc, trustworthy)` tuple, so the reader sees the full grid, not a single verdict.

---

## 8. Both outcomes are reportable (pre-anticipated)

**(a) PASS — untied-flat wins.** Flexible depth is demonstrated on a **plain feedforward net** where
depth is provably non-substitutable by width (param-matched control stalls). This strengthens the
FlexNN depth story from "weight-tied recurrent only" to "plain deep MLP too," and directly supports
the user's original claim. Feeds F7 §3 (depth).

**(b) FAIL-but-tied-cells-pass — weight-tying is the load-bearing ingredient.** If untied-flat (and
possibly untied-perstep) fail while the tied cells pass, the finding is that **weight-tying** is what
makes depth learnable/generalizable here. This is a *positive* result for the transformer roadmap:
weight-tied = the **Universal Transformer** shape (looped shared layer + per-position processing,
**Dehghani et al. 2018**, arXiv:1807.03819, verified). Feeds F7 §3 + §7 (roadmap) and F8's staged
path (ii).

Either way the spec + review verdict are delivered for post-hoc user review, and F7 reports the
outcome whichever way it lands (no result is "failure theater").

---

## 9. Run matrix & file manifest (scope handed to F5b — not run in F5a)

New training runs (2 seeds each unless cited):

| Cell / control | configs | seeds | runs |
|---|---|---:|---:|
| Cell 1 untied-flat | `d ∈ {4,7,10}` | 2 | 6 |
| Cell 3 tied-flat | `d ∈ {4,7,10}` | 2 | 6 |
| Cell 2 untied-perstep | `d = 10` | 2 | 2 |
| Cell 4 tied-perstep | `d = 10` (plain single-readout) | 2 | 2 (MANDATORY confirm, §4 ruling; anytime ≥0.90 cited as corroboration) |
| wide-shallow controls | `W ∈ {188,311,435}` (mandatory, FF-CLAIM) | 2 | 6 |
| wide-shallow controls | `W ∈ {146,123,887}` (attribution, per §5) | 2 | 6 |

≈ **28 training runs** (22 if the three attribution-only controls are run 1-seed, or deferred until a
tied/perstep cell actually reaches ≥0.90 — reviewer may relax those to on-demand). All are full-batch
Adam on 10,000×40 inputs, 60 classes, CPU; individual runs are small but the count is non-trivial —
**F5b should validate the ~30–60 min total estimate against the actual arm count and detach the batch
per the long-run environment rule** (this spec does not certify that wall-clock).

**Output (F5b):** `automl_package/examples/capacity_ladder_results/D_TOY_PROBES/ff_depth_pilot_a5_seed{0,1}.json`
(one JSON per seed, all cells + controls + per-class counts + convergence trajectories inside), then a
results § appended to `docs/depth_capacity/verdict_per_input_depth.md`. New `NetKind` members
(`TIED_FLAT`, `UNTIED_PERSTEP`) + builders + driver flags go into `depth_composition_toy.py`, keeping
the selftest convention.

---

## 10. Non-goals (pre-registered boundaries)

- **No `L > 10`.** The A5 recurrent block hits a **GD-trainable wall (MOD-1)** at `L ≥ 12` — it fails
  to fit even the training set at length 12+ (`depth_selection_toy.py:38`; `joint_toy_design.md:38`;
  verdict §10.1). `L = 10` is the ceiling; the ladder caps at `d ≤ 10`.
- **No new groups** beyond A5 (the certified non-solvable substrate). The Z120 solvable-group clean
  control — where depth *is* width-substitutable and all arms fit — is **already certified**
  (verdict §5, G4 clean 3/3) and is **not** part of this pilot's required matrix; cite it, do not
  re-run.
- **No selection/router logic** in this pilot — substrate question only. Per-input depth *selection*
  is a separate, already-certified result (verdict §10, D8).
- **No curriculum tricks.** If the untied-flat arm fails, curriculum/annealing is **future work
  only**, noted but not built here.

---

## 11. Open interpretations & assumptions the author had to make (for the reviewer)

Making the settled design core numeric forced several concrete choices the brief left implicit. The
reviewer should confirm or overrule each:

1. **Input dimension is 40, not 50.** The brief says 50-dim; the as-built toy gives 40
   (`4 gen × 10 pos`), independently confirmed by the D8 verdict ("in_dim 40"). Spec uses 40 (§1).
   *This is a correction, not merely an interpretation.*
2. **Cell 4 — RESOLVED by reviewer: mandatory confirm run (was "cite only").** The cited ≥0.90 comes
   from the **anytime multi-exit** substrate (`depth_selection_toy.py`, shared readout on the running
   prefix product at T∈{2,4,6,8,10}) — the same module but a *richer supervision regime* than Cells
   1–3's plain single-readout-on-full-word product (verified: no plain-`RecurrentComposer`-A5-L10 JSON
   on disk). Citing it would confound the grid with a training-supervision difference, so the reviewer
   upgraded it to a **mandatory** 2-seed plain-`RecurrentComposer` confirm run under the identical
   `train_clf` protocol (`--net recurrent --group a5 --seq-len 10`); anytime numbers kept as
   corroboration (§4, §9). Mechanical, author-pre-authorized.
3. **Per-step arms (Cells 2, 4) are pinned at `d = L = 10`; only the flat arms sweep `{4,7,10}`**
   (§3). Feeding 10 letters through <10 per-step layers would be an input-schedule redesign, which
   the non-goals forbid. If the reviewer wants per-step arms at other depths, that is a design change,
   not a build detail.
4. **Cell 2 (untied-perstep) block architecture** was chosen to be *identical to Cell 4's shared
   2-layer block, untied* (10 distinct copies), so weight-tying is the sole difference between Cells 2
   and 4. Consequence: 89,660 params (the heaviest arm), controlled by the `W=887` matched null (§5).
5. **Cell 3 (tied-flat) injects the flat word once at layer 0** (via `Linear(40,64)`) and iterates a
   shared *state-only* block; it does **not** re-inject the word each step. This matches "all letters
   visible at layer 0." An alternative (re-inject the flat word each iteration, closer to a Universal
   Transformer) is defensible; the author chose the simpler reading and flags it.
6. **Attribution-only wide-shallow controls** (`W ∈ {123,146,887}`) are pre-registered but may be run
   on-demand (only if the corresponding tied/perstep cell reaches ≥0.90), to save compute, without
   weakening the pre-registration — the widths are frozen regardless of when they run (§9).
7. **Near-uniform label distribution at L=10** is assumed for the starvation arithmetic (§2); F5b must
   print per-class counts to confirm it empirically rather than rely on the mixing argument.

None of these touches the settled 2×2 core or the pre-registered bars; they are the numeric
commitments the core did not yet specify.

---

## Review verdict — SOUND-WITH-FIXES (adjudicator, 2026-07-18)

**Verdict: SOUND-WITH-FIXES → GO for F5b.** The 2×2 substrate design is correct, its confound ledger
is arithmetically exact, and its pre-registered bars are the certified strand's own (not arbitrary).
Two mechanical fixes are folded in below; neither is a principled-design fork, so F5b proceeds.

**Re-derived from primary evidence (not the author's prose):**
- **Every parameter count in §5 verifies EXACTLY** (`count_params` on the actual modules): Cell 1 =
  19,004 / 31,484 / 43,964 (d=4/7/10); Cell 3 = 14,844 (constant in d); Cell 4 = 12,476; Cell 2 =
  89,660. Control formula `101·W+60` confirmed; all six matched widths {188,311,435,146,123,887}
  reproduce within −0.26%…+0.23% (the ±0.3% claim holds). D7 precedent `161·101+120 = 16,381` checks.
- **Input dim is 40, not 50 (author correction upheld):** `make_word_data(A5,10,·)` → `x (·,40)`;
  4 gen × 10 pos = 40; no toy config yields 50.
- **Per-step arms genuinely pinned at d=L=10:** `RecurrentComposer.forward` loops over the 10 letters;
  a {4,7,10} sweep would drop letters or redesign the input schedule (forbidden §12). Real constraint.
- **Cell 3 inject-once reading is correct, not a fork:** "flat input" = all letters at layer 0;
  injecting once then iterating a shared state-only block IS that column's definition.
- **Bars 0.90/0.60 are the certified pilot's own** (`FIT_ACC/STALL_ACC`), inherited for comparability,
  frozen here per Decision 9; ±0.3% param-match makes FF-CLAIM depth-vs-width, not capacity-vs-capacity.

**Fix 1 (substantive): Cell 4 confirm run optional → MANDATORY, same single-readout protocol as Cells
1–3.** No plain-`RecurrentComposer`-A5-L10 JSON exists on disk; the cited ≥0.90 is from
`depth_selection_toy.py`'s anytime multi-exit net (shared readout on the running prefix product at 5
exits) — a richer supervision signal than Cells 1–3's plain single-readout-on-full-word product.
Citing it would confound the tied+perstep corner with a training-supervision difference. The module
already produces the correct run (`run_pilot --net recurrent --group a5 --seq-len 10`, 2 seeds) with
no new code; anytime numbers kept as corroboration. Bars untouched; run total 26 → 28.

**Fix 2 (accuracy nit): §1 "distinct/disjoint" → with-replacement, <1% overlap.** Measured 86 (0.86%) /
96 (0.96%) of val words leak into train; negligible at the 0.90 bar and uniform across arms, but the
"distinct/disjoint" wording was a factual overstatement. Corrected; F5b prints the observed overlap.

**No PI-level decision, no UNSOUND defect.** Task, group, metric, bars, and controls are continuous
with the certified G-DEPTH strand; both outcomes (§8a/b) are pre-anticipated. **F5b is GO.**
