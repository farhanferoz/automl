# Width: the mechanism, and the conditions under which it ports to a transformer

**Status: ANALYSIS, not measurement.** Every claim below is either (a) a property of the code, cited
to the line, (b) elementary algebra, or (c) an argument about a transformer we have never run. Class
(c) is marked **[ARGUMENT]** at each occurrence and may not be cited as a result. This note exists so
the account is recorded once instead of re-derived; it is not evidence for anything.

Authored 2026-07-21 (user discussion). Owner: `width.md`. The depth material in §7 is an **input to
`depth-selection.md`**, deliberately not a width claim.

---

## 1. The certified design, and why the other one fails

Three width architectures were compared (`MASTER.md` naming key). Two matter here, and both live in
`automl_package/models/architectures/nested_width_net.py`:

- **`NestedWidthNet`** (`:39-111`) — one shared hidden layer AND **one shared output layer** read at
  every width. **FAILED.** Closed; not reopened; no further compute is spent on it (user, 2026-07-21).
- **`SharedTrunkPerWidthHeadNet`** (`:164-230`) — the same shared hidden layer, but **width `k` reads
  its own output head**. **Certified: `G-WIDTH = PASS`, 2026-07-16** (`width-cert.md`).

Both mask identically: a width-`k` forward zeroes hidden units `k..w_max` before the readout
(`:212-215`). The *only* difference is whether the readout is shared.

**Why the shared readout fails — the mechanism, stated so it is not re-derived.** The training loss
is a sum over widths. With one shared output layer, the weight on hidden unit 1 is asked for two
incompatible things at once: width 1's loss term wants it tuned for the case where unit 1 carries the
whole prediction, and width `w_max`'s term wants it tuned as one contributor among `w_max`. A single
weight cannot serve both, so the widths fight over it and the network lands between the two — worse
at every width than a net dedicated to that width. Per-width output nodes end the conflict by giving
each width its own copy of the contested parameter; the *hidden* layer stays shared, so the design is
still genuinely nested and still cheap.

**Consequence for the invariant.** The nested-architecture research
(`docs/plans/width_dial_2026-07-11/nested_architecture_research_2026-07-11.md`) proposed a
three-property invariant — stable unit identity, self-contained prefixes, importance ordering. The
certified result suggests **property 2 (self-containment) is not required** when the readout is
per-width: the conflict it was meant to prevent is resolved by the separate heads instead. That is an
inference from one certified run, not a tested claim.

---

## 2. What the design actually requires

Two conditions, and everything in §4–§5 follows from them:

1. **The readout must be linear in the hidden vector.** Then zeroing the tail commutes with reading
   out, so every width's answer is a prefix sum over per-unit contributions and all widths come off
   ONE hidden-layer evaluation — implemented as a cumulative sum in
   `automl_package/models/flexnn/width/architectures.py:81-95` (`all_widths_forward`, exact and
   loop-free for the shared-readout class).
2. **Each width needs its own readout parameters** (§1).

**The one load-bearing property still UNMEASURED: importance ordering.** Hidden unit `j` receives
gradient only from widths `k >= j`, so unit 1 appears in every width's loss term and the last unit in
exactly one. The summed loss therefore *induces* a decreasing importance ordering — but nobody has
measured that it holds (is unit 1 the best single feature, unit 2 the best addition, stable across
seeds?). This is the gap the research record specifically warns not to repeat, it needs no retraining,
and it matters MORE at transformer scale, not less: with many rungs the gradient pressure on late
units thins out, which is exactly where ordering could fail silently. → `width.md` **WSEL-13**.

---

## 3. Parameter cost of per-width readouts

**As implemented (toy).** Each head is a full `Linear(w_max -> 1)` (`:196`), so the module allocates
`w_max * (w_max + 1)` parameters across `w_max` heads. But head `k` reads a vector whose tail is
zeroed, so **only the first `k` columns of head `k` can ever influence its output**: the effective
count is `1 + 2 + ... + w_max = w_max(w_max+1)/2`, roughly half the allocation, and the FLOP
accounting already charges the sliced version
(`automl_package/models/flexnn/width/architectures.py:306-320`, which bills
`_linear_macs(k, ...)`, not `w_max`). Allocated ≠ executed; report both or the cost claim is wrong.

**[ARGUMENT] At transformer scale**, with the nesting on the feed-forward hidden dimension (§4),
rungs at `{h, h/2, h/4, h/8}` and the standard `h = 4d`:

- shared up-projection: `d*h` — unchanged, one copy serves every rung;
- per-rung down-projections: `(1 + 1/2 + 1/4 + 1/8) * h * d = 1.875 * h * d` instead of `h*d`; <!-- numcheck-ignore: closed-form arithmetic on a hypothetical rung set, not a measured result -->


- so the feed-forward block grows from `2hd` to `2.875hd` — **+44% of FFN parameters**;
- against a whole block including attention (`~4d^2`) at `h = 4d`: `12d^2 -> 15.5d^2`, **~+29% block
  parameters**;
- **inference cost is unchanged** — one rung runs, and a deployment pinned to a small rung is
  *cheaper* than the baseline block, since it needs neither the other heads nor the unused hidden
  units.

Storage grows; served compute does not. State it that way or the trade reads worse than it is.

---

## 4. [ARGUMENT] Where a transformer satisfies the two conditions

| Site | Condition 1 (linear readout, no norm between) | Verdict |
|---|---|---|
| **Feed-forward hidden width** | up-projection → nonlinearity → down-projection, with the block's normalisation OUTSIDE the pair (pre-norm) | **Clean — the direct analogue.** Up-projection = our shared hidden layer, down-projection = our output head. Sharing one down-projection across rungs is our FAILED design; per-rung down-projections is our certified one. |
| **Number of attention heads** | heads are concatenated then linearly projected out | **Clean.** A prefix of heads plus a per-rung output projection is structurally the same design. |
| **Dimension inside an attention head** | scores pass through a scaling and a softmax, both nonlinear in that dimension | **Broken.** Truncation does not commute; a narrow forward is not a sub-computation of the wide one. |
| **Model dimension (residual stream)** | every block reads and writes the same stream; normalisation intervenes everywhere; the output vocabulary projection is `vocab x d` | **Expensive and global.** A rung choice becomes a whole-network decision, and per-rung vocabulary projections are large matrices, not cheap heads. This is the regime the Matryoshka work occupies, and it nests the *representation*, not the compute. |

**The portable claim, stated precisely:** what generalises is *per-rung readouts over a nested prefix,
wherever the readout is linear and no normalisation intervenes*. In a transformer that is the
feed-forward hidden width and the head count — which is where most of the parameters and most of the
compute already sit. What does NOT generalise is any claim that a narrow forward is a free by-product
of a wide one across a normalisation layer.

---

## 5. [ARGUMENT] Normalisation: the obstacle, and four repairs

**The obstacle.** Layer normalisation computes its statistics over the whole vector. Truncate the
vector and the divisor changes for every surviving unit, so (a) a narrow forward stops being a
sub-computation of the wide one and (b) a unit's meaning becomes rung-dependent — both load-bearing
properties at once.

**This is not speculation and the literature already conceded it.** Slimmable networks required a
*private normalisation per width*, and the follow-up states plainly that a naively trained model fails
to run at different widths even when its normalisation statistics are recalibrated — both quoted with
sources at `docs/plans/width_dial_2026-07-11/nested_architecture_research_2026-07-11.md:44-60`.

**Repairs, cheapest first:**

1. **Nest where no normalisation intervenes.** The feed-forward hidden layer (§4). Costs nothing,
   valid today, and it is the highest-parameter site anyway. **Preferred.**
2. **Cumulative prefix statistics.** A prefix mean and prefix mean-of-squares are cumulative sums over
   the hidden units, so *every* rung's normaliser is exactly computable from ONE pass, at the cost of
   one extra cumulative sum — the same trick `all_widths_forward` already uses for the readout. This
   fully repairs (a), the efficiency half. It does not repair (b).
3. **Per-rung normalisation parameters** (scale and shift per rung). Cheap, and this is the
   literature-supported fix (repair-1 citation above). Note the special case that makes (b) tractable:
   if the normalisation sits immediately before a **per-rung** linear readout, the rung-dependent
   divisor is a single scalar per rung and the per-rung readout absorbs it exactly. (b) only bites
   when the normalised vector then flows through further *shared nonlinear* layers — i.e. the
   model-dimension case, not the feed-forward-hidden case.
4. **A rung-independent normaliser** (fixed constant, or a normalisation-free block) makes truncation
   commute exactly but changes the model. ⚠️ **UNVERIFIED — literature not checked.** Do not build on
   this until the normalisation-free-architecture literature has been surveyed and cited. Recorded as
   an option, not a recommendation.

**Net:** the design is repairable at bounded, known cost. It does not fail to generalise; it
generalises **with conditions**, and the conditions are checkable.

---

## 6. Schedule and compute cost — what the width work must now record

The training loop currently recomputes the shared hidden layer once per sampled width and discards
all but one result (`width.md` **WSEL-12**). Two consequences that bind this strand:

- The premise "each width rung costs a real forward" (MASTER Decision 20) is an artifact of that
  defect, not a property of the architecture. Once the hidden layer is computed once, training every
  width every step costs about one forward plus cheap readout arithmetic.
- Therefore the sampling schedules (sandwich; the four-random-widths ablation; one-width-per-batch)
  can no longer be justified on compute for width. Any remaining case for them is a **quality /
  regularisation** claim and must be pre-registered as one. → `width.md` **WSEL-14**.

**Cost is a first-class recorded output from here on**, not an argument: every schedule cell reports
parameters, executed FLOPs at the deployed rung, and wall-clock training time. The accounting
primitives already exist — `automl_package/utils/capacity_accounting.py:258,276,298` — and are
reused, not re-implemented.

**Already measured, do not re-run:** four widths drawn uniformly per step (no guaranteed smallest and
largest) reached the error floor on 3 seeds, so the sandwich's guarantee is not load-bearing for the
certified architecture (`width-cert.md:210-220`; the draw count is
`automl_package/examples/kdropout_converged_width_experiment.py:80`). The untested cells are
**one width per batch** and **all widths every step**.

**Prior evidence bearing on one-width-per-batch.** The retired per-example uniform draw already
under-fit the widest width badly — and that schedule still gave every width gradient from a slice of
every batch. One width per *batch* is strictly harsher: whole widths receive nothing for entire steps.
The gradients remain *correct* under it (unselected heads leave gradients unset, and the optimiser
skips them entirely — no phantom decay or momentum drift, verified against
`automl_package/examples/nested_width_net.py:271`'s plain Adam and PyTorch 2.10's
`zero_grad(set_to_none=True)` default). The cost is variance and staleness, not correctness. ⚠️ That
benign behaviour depends on gradients being *unset* rather than *zero*: with `set_to_none=False`,
unselected heads receive zero-valued gradients, the optimiser does step them, and weight decay would
shrink heads no batch asked to change. WSEL-14 pins this in a test.

---

## 7. [ARGUMENT] Why DEPTH is structurally the better transformer target — input to `depth-selection.md`

**Not a width claim, and not actionable in this strand.** Recorded here because it was derived here;
it belongs to the depth strand when that reopens.

A transformer is *already* a stack of shape-preserving blocks, each adding a correction to a running
representation. A depth prefix — run the first `d` blocks and stop — therefore has, for free, the
three things width has to engineer:

1. **The narrow computation is literally a sub-computation of the wide one.** No vector is truncated,
   so every normalisation layer sees a full-width vector and its statistics are unchanged. The whole
   of §5 simply does not arise.
2. **Every rung comes off one forward pass**, read from the running representation after each block —
   which is why MASTER Decision 20 already assigns depth the all-rungs-every-step schedule.
3. **Shapes match, so ONE output projection serves every rung.** Width's expensive case (per-rung
   vocabulary projections, §4) has no depth analogue.

Consistent with what we measured: depth's certified arm won with a **shared** readout while width won
with **per-rung** readouts — and shared readouts are exactly what makes the depth port cheap.

**Three caveats, all real:**

- The depth substrate is not currently in a state to build on: its positive control fails, and it
  failed in the configuration already known to be wrong (shared readout, regularisation pinned off).
  MASTER Decision 14 applies — that battery measures the protocol, not the arms. Fix first.
- A representation partway up the stack is not automatically in a form the final output layer can
  read. The fix (a small per-exit normalisation) is cheap but is a real step, not nothing.
- Depth only *saves* compute if you can stop early per input — for a transformer, per token. That is
  the halting question already parked in `width-depth.md`.

**⚠️ Literature gap, blocking any novelty claim.** Early exiting / layer skipping in transformer
stacks is a crowded area. Width got a proper literature pillar
(`docs/plans/width_dial_2026-07-11/nested_architecture_research_2026-07-11.md`); depth has **no
equivalent**. A survey to that standard is required before any depth result is positioned as novel.
Nothing here should be read as a claim about what is or is not already published.

---

## 8. What this changes for the width strand

The transformer story runs through depth (§7), with the feed-forward hidden width as a genuine but
secondary port (§4). Width's remaining value is therefore the **mechanism** finding (§1) and the
**ordering property** (§2) that determines whether nesting survives many rungs at all. That is an
argument for finishing the efficiency fix, the ordering diagnostic and the schedule/cost sweep — and
against spending much beyond them on width.
