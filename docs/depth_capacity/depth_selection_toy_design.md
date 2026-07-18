# Design spec: the depth-selection toy (D8a deliverable — for review before any build)

**Charter.** Certify per-input depth *selection without an oracle*: one anytime network serving
multiple computation depths, plus a **distilled** router (MASTER Decision 13) that chooses each
input's depth from the input alone — the width protocol re-derived for depth. The certified
graded S5 toy cannot carry this: its required depth is syntactically visible (word length) and
its error-vs-depth curve is a cliff, so selection on it is degenerate. This document designs the
replacement. Per the D8a gate (`docs/plans/capacity_programme/depth.md`), nothing below is
built until this design is reviewed and approved.

Facts used below are computed, not assumed; the two learnability probes are the only open
empirical questions and are marked PENDING.

---

## 1. The central design tension

A selection toy needs per-input variation in *required* depth. Two architecture families
bracket the danger:

- **Full-input iterated nets** (state iterated T times over the whole word): computing a group
  product needs global aggregation, and a net free to learn balanced-tree reduction does it in
  ~log₂ L iterations for *every* input — required depth becomes uniform, and selection
  degenerates exactly as it did for the graded toy. Per-input variation would rest on hoping
  the net learns a *local* reduction strategy — an optimization hope, not a design guarantee.
- **Sequential-consumption nets** (the certified D1b substrate, one letter per step): required
  depth is forced to L for every input — uniform again — *unless the input itself makes the
  tail of the computation redundant.*

The preferred construction (C1′) takes the second branch and constructs the redundancy: it
plants a hidden **commitment point** after which further computation provably adds nothing.
This yields a knee in the error-vs-depth curve that is provable at the Bayes level (§3.4) —
the toy's key advantage over every alternative considered, where the knee is only a hypothesis
about what the net happens to learn.

## 2. Common setting

Alphabet = the 4 adjacent transpositions of S5 (the certified toy's generators,
`automl_package/examples/depth_composition_toy.py::build_group`). Each generator is
self-inverse. With this alphabet S5 is a Coxeter group: every element has an exact minimal
word length (its inversion count), giving exact ground truth for all difficulty bookkeeping.

Computed structural facts (script: §7 probe file; all exact):

| quantity | values |
|---|---|
| elements at Coxeter length 0..10 | 1, 4, 9, 15, 20, 22, 20, 15, 9, 4, 1 (Σ = 120) |
| classes reachable by a length-t word | t=4: 30 · t=6: 50 · t=8: 59 · t≥9: 60 (parity class) |
| TV distance to uniform-on-parity, length t | t=6: 0.47 · t=8: 0.38 · t=10: 0.31 · t=12: 0.25 |
| identity-product words of length k | k=2: 4 · k=4: 34 · k=6: 358 · k=8: 4,234 · k=10: 53,764 · k=12: 715,164 |
| Bayes accuracy with g letters unread | g=1,2: 0.250 · g=3,4: ~0.14 · g=5,6: ~0.09 · g=8: 0.065 |

Three design constraints fall straight out of the table:

1. **Parity:** identity-product words exist only at even length ⇒ the commitment-point ladder
   must step by 2, and every input's label parity equals L's parity ⇒ the label space is the
   60-element parity class, not 120.
2. **Class coverage:** commitment points below 6 are intrinsically class-starved (t=4 reaches
   only 30 of 60) — the ℓ=4 failure of the graded toy, caught here by arithmetic. Floor: t* ≥ 6.
3. **Short-tail stereotypy is intrinsic:** commitment 2 letters before the end *means* the
   last letter is doubled — surface-visible by definition, for any sampler. Hence the ladder
   keeps tails ≥ 4 (t* ≤ L − 4) for the hidden claim, and the falsifier is evaluated per
   stratum. (Revision 1's larger lesson — the identity-tail *population* is detectable at
   every tail length when sampled separately from the prefix — is what forced the
   conditioned-uniform sampler of §3.1.)

## 3. C1″ — commitment-point words, conditioned-uniform sampling (PREFERRED; revision 2)

### 3.0 Why revision 2 (the falsifier killed revision 1)

Revision 1 built words as *uniform prefix + separately-sampled identity-product tail*. The
falsifier probe rejected it decisively: a 1-hidden-layer MLP recovered the commitment point at
**79% accuracy vs 14% chance** (93–95% recall on constructed strata) — identity-product words
are a statistically special population, and a surface net detects where the "special-looking"
region begins. The same probe run surfaced the fix: **42% of purely uniform words already
commit early naturally** (their suffixes happen to multiply to identity; the rate matches the
exact walk-return probabilities: 0.25 for the last 2 letters, 0.133 for the last 4). So no
tail is manufactured at all:

### 3.1 Generative spec

Fix syntactic length **L = 14**. Realized-commitment ladder **t* ∈ {6, 8, 10, 14}** (even;
≥ 6 by constraint 2; hidden-claim strata are t* ≤ 10 = tails ≥ 4; t* = 14 is the
never-commits-early stratum). A word is drawn **uniformly from the realized stratum**
{w : t*(w) = t} by first-hit dynamic programming: choose the final product f with probability
proportional to (#length-t paths first-hitting f) × (#length-(L−t) identity-fold suffixes),
backward-sample the first-hit path under the taboo DP, backward-sample the suffix under the
unconstrained DP. Every sample is assertion-checked (realized t* recomputed and compared).
Because the whole word is uniform *given its stratum*, there is no prefix/tail distributional
seam for a surface net to find — detecting t* now requires exactly the suffix-product
computation the toy is about. **Label:** the full product (= the prefix product at t*); label
space = the 60 even-parity elements.

**Constructed vs. realized knob (a trap this design closes explicitly):** a random prefix may
*itself* contain an identity-product suffix, making the true commitment point earlier than
constructed; conversely nothing can make it later. Therefore the ground truth is **recomputed
per word**: t*(x) = the smallest t such that the prefix product at t equals the full product
(equivalently, the suffix from t+1 multiplies to the identity) — one O(L) scan with the
multiplication table, with **no contiguity requirement**: inside an identity tail the running
product wanders and only returns to the answer at the end, and *coincidental* early commitment
(a prefix hitting the final answer before the constructed split) counts. All stratification,
bars, and router labels use the realized t*(x), never the constructed one. (Probe round 1
caught an implementation of this scan that wrongly assumed contiguity — every word scored
t* = L; recorded in §7 as evidence the probe layer works.)

**Dataset arithmetic:** word space 4¹² ≈ 1.68 × 10⁷ — no global starvation at any practical
sample size. Proposed: 40,000 words per constructed stratum (160,000 total), 50/50 train/val
split of *distinct* words (the certified toy's convention). Per-stratum label skew (TV 0.25 –
0.47, §2) is real but bounded; report per-stratum accuracy and, if a stratum's class skew
biases the router, reweight classes within stratum — decided at pilot, recorded in the doc.

### 3.2 Network and dial

The **certified D1b substrate unchanged in kind**: the weight-shared recurrent block consuming
one letter per step. The dial T = number of letters consumed before readout (T ∈ {2, 4, …, 12};
compute ∝ T exactly, as executed-steps). Anytime training = the width sandwich ported to T:
each batch trains readouts at all T on the *final* label (per-position readout heads on the
shared state — the certified per-length-head machinery reused for per-*position*). No new
architecture is introduced — this is deliberate; C1 (§5) shows what happens when the
architecture is also novel.

### 3.3 Hidden-ness + falsifier

Computing t*(x) requires evaluating suffix products — group computation, not surface
statistics. Residual surface risk: tail stereotypy at small tail lengths (constraint 3), and
any unforeseen statistical tell of DP-sampled identity words. **Falsifier probe (≤15 min,
must FAIL for the design to survive):** train a 1-hidden-layer MLP on the raw one-hot word to
predict realized t*; reject the construction if per-stratum balanced accuracy exceeds
chance + 10 pp on any stratum with tail length ≥ 4. (The t*ᶜ = 10 stratum, tail length 2, is
*expected* to be partly detectable; it is retained for ladder continuity but excluded from the
hidden-ness claim, stated as such in the verdict.)

### 3.4 Gradedness — provable, then tested

By construction the tail carries zero information: for T ≥ t*(x) the final label is a
deterministic function of the consumed prefix (achievable accuracy 1.0, by the strategy
"output the prefix product"); for T < t*(x) with g = t*(x) − T informative letters unread, the
Bayes ceiling in §2's table applies (0.25 at g = 2, ~0.13 at g = 4). **Approximation stated
honestly:** those ceilings are computed for uniform random unread letters; the exact
conditional under the stratum mixture (the net could partially infer the stratum from the
prefix) differs slightly — the pilot reports the empirical curve against the table as a
sanity band, not an exact target. The knee therefore exists at the data level as a
theorem of the construction. What remains empirical — and is the entire point of the
**gradedness probe (≤15 min, PENDING)** — is whether the trained net *realizes* the knee:
accuracy(x, T) should reach ≥ 0.95 of its T = 12 value at T = t*(x), and should sit within
10 pp of the Bayes ceiling at T = t*(x) − 2 (any excess above the ceiling is a bug in the
probe, not a success).

### 3.5 Selection protocol (D8b, unchanged from the plan)

Distill the router from the slice-B per-input error table over T (cheapest-within-tolerance,
δ_tie = 0.25, sweep in battery), hard-pick deploy executing only T(x) steps, versus
best-fixed-T (val-selected). Router input: the raw word only (S4). **Scope note:** this is the
one-shot width-style router (read input → choose T). The *streaming* variant (decide per step
while consuming — the transformer halting shape) is deliberately out of D8's scope and named
as J0/M3+ work; D8 certifies the mechanism, not the streaming policy.

### 3.6 Pre-registered bars (constants frozen at pilot launch)

- **S1 substrate:** anytime net at T = 12 reaches held-out acc ≥ 0.90 on every stratum with
  t* ≥ 6.
- **S2 knee:** mean over stratum of acc(x, T = t*(x)) ≥ 0.95 × acc(x, 12); mean acc at
  T = t*(x) − 2 ≤ Bayes(g = 2) + 10 pp = 0.35.
- **S3 deploy:** distilled router's executed mean-T ≤ 0.8 × best-fixed-T at held-out accuracy
  within δ_tie of best-fixed — the width compute-payoff claim re-derived for depth.
- **S4 no-oracle:** router consumes the raw word only; features documented in the JSON.
- **Convergence discipline:** every training cell reads the D6 `diverged` flag; diverged cells
  are excluded and re-seeded, per MASTER Decision 9.

**Kill criterion:** falsifier passes (hunger visible), or the gradedness probe shows no knee
(acc at t*−2 ≫ Bayes ceiling — construction leak — or acc at t* far below its T = 12 value on
2 seeds — the net can't localize commitment). Either → C1′ dies; proceed to C2, once; then
escalate to the user (D8b escalation rule).

**Cost:** probes ≤ 15 min each (CPU); pilot (anytime training, L = 12, 4 strata) est. 45–90
min by scaling the D1b arm timings; battery 2 seeds ≈ 2–4 h detached.

## 4. Why not the alternatives

### C1 — reducible words on a full-input iterated net (DEMOTED)

The original sketch: pad a short core with cancelling material; iterate a block over the whole
word. Two design-level defects. (a) The §1 tension: nothing forces per-input iteration counts —
tree reduction serves all inputs at ~log L, and the knee-at-reducibility hypothesis is purely
an optimization hope. (b) It requires a *novel* architecture (iterated full-input block) whose
basic trainability on this task is unmeasured — compounding construction risk with
architecture risk, the exact combination that burned the four 1-D candidates. Retained only if
C1′ dies at the falsifier AND C2 dies, as raw material for a redesign discussion.

### C2 — mixed-solvability population (FALLBACK)

Same-length words from S5 vs. an abelian generator set (disjoint label blocks); solvable
inputs are shallow-computable (Liu et al. 2022, arXiv:2210.10749), S5 inputs are not
(Barrington 1989). Honest weaknesses, stated up front: the depth-hunger dichotomy is binary
(2 levels, not a ladder — a thin selection result), and the surface-shortcut risk is severe
(generator identities may betray the population; the falsifier is expected to be a close
call). C2 is a fallback, not a co-equal: it certifies *that* selection works, but on a much
coarser dial than C1′.

## 5. Theory grounding — what is licensed vs. assumed

- **Licensed:** width cannot substitute for depth on S5 word problems (Barrington 1989;
  empirically re-confirmed by D1b's G2 bar 3/3). Solvable-group shortcuts (Liu et al. 2022)
  ground C2's contrast. Coxeter word length gives exact difficulty bookkeeping. The Bayes
  knee of C1′ (§3.4) is a construction theorem, not a conjecture.
- **Assumed until probed (labeled, No-guessing gate):** (i) the anytime-trained net realizes
  the provable knee (gradedness probe); (ii) DP-sampled identity tails carry no learnable
  surface tell at tail length ≥ 4 (falsifier probe); (iii) D1b training dynamics transfer to
  mixed-t* anytime training (pilot bar S1).

## 6. THE DESIGN FORK — RESOLVED (user sign-off 2026-07-17)

**DECISION: Option A, with the narrow-deep fallback (below) documented as the escape hatch.**
Rationale (user, 2026-07-17): the charter is *learn depth as a function of the input, correctly*;
whether the required depth is easy or hard to read off the surface is **not a concern**. Concealment
is therefore dropped as a kill criterion. The certified substrate already establishes the ideal
property — depth genuinely irreducible to width (`verdict_per_input_depth.md`, G1–G4 3/3) — so the
selection toy stays on the depth-irreducible task; no retreat to a width-substitutable toy is needed
*unless the gradedness make-or-break below fails*.

Concrete consequences for D8b:
1. **Drop the falsifier as a kill criterion; keep it as a measured covariate**, and add a
   pre-registered **surface-baseline control**: the distilled router is compared head-to-head against
   a router of the same shallow architecture trained on the raw input directly. Both are reported. If
   the surface router routes as well, the verdict says so plainly — the comparison is the science, not
   a failure. This is the only guard the "no-oracle" claim needs.
2. **The make-or-break is gradedness, not concealment.** For the router to be worth anything, running
   fewer steps than an input needs must degrade *gradually* (a real accuracy-vs-compute tradeoff), not
   fall off a cliff. The commitment-point construction gives a provable Bayes knee (§3.4); the open
   empirical question is whether the anytime-trained net *realizes* it (convergence-gated gradedness
   probe — the first D8b action). Pass → ideal result. Fail → the fallback.
3. **Construction:** C1‴ as specified — A5 five-cycle generators, L = 16, ladder {6, 8, 10, 16},
   ~40k words/stratum, one-shot distilled router (streaming/halting stays J0/M3+).

**FALLBACK (documented, user-approved 2026-07-17): narrow-deep width-substitutable graded toy.**
Trigger: the ideal (depth-irreducible) selection toy fails the gradedness make-or-break on 2 seeds —
i.e. the anytime net cannot produce a graded error-vs-depth curve. Escape: build a task where depth
and width trade off *smoothly* and required depth is a graded function of the input, run a
narrow-but-deep anytime net + distilled router on it, and certify "depth learned as a function of the
input, correctly" there. This drops the irreducibility property (already certified separately by the
substrate verdict) in exchange for a guaranteed graded curve. Taken without further user ceremony if
the trigger fires; the redesign note is written into this doc.

---

### 6.1 The original fork (preserved for the record)

Three construction rounds all failed the concealment bar (§7). The structural finding: a
commitment point defined by "the suffix folds to the identity" is **intrinsically
surface-leaky** — short-to-medium identity-fold word sets are combinatorially stereotyped in
every alphabet tried (S5 adjacent transpositions: letter-count parity, 98–100%; A5 five-cycles
with no relations shorter than 5: still 59–98% detectable, decaying only slowly with suffix
gap). Options:

- **A (RECOMMENDED): retarget the bar from concealment to selection value.** The charter —
  learn depth as a function of the input with no oracle labels — never required the mapping to
  be *unlearnable from surface statistics*; the certified width toy's input→width mapping was
  also simple. Drop "hidden" as a kill criterion; keep the falsifier as a measured covariate;
  add a pre-registered **surface-baseline control**: the distilled router must be compared
  against a router trained on the same shallow architecture from raw input directly, and the
  deploy claim reports both. If the surface router routes as well as the distilled one, the
  mechanism claim is weakened and the verdict must say so — the comparison becomes the
  science, converting the leak into a control. No further construction work needed; D8b can
  start on C1‴ (A5, L = 16) as-is.
- **B: keep hard concealment and pay for it.** Falsifier scaling study over suffix gap
  (cheap), then L ≈ 20–24 with gaps ≥ 12–16 if a concealment threshold exists. Risk: it may
  not, at any feasible L; training cost grows substantially.
- **C: abandon suffix-identity commitment** — a different per-input difficulty mechanism
  entirely (new design round from scratch).

Secondary decisions if A: confirm L = 16, ladder {6, 8, 10, 16}, ~40k words/stratum for the
battery; one-shot distilled router for D8 (streaming/halting stays J0/M3+).

## 7. Probe results (three falsifier rounds, one gradedness round — all on disk)

| round | construction | falsifier (chance) | verdict |
|---|---|---:|---|
| 1 | C1′ S5 + manufactured tails | — (probe caught scan bug) | probe layer validated |
| 2 | C1′ S5 + manufactured tails | 79% (14%) | REJECTED — tail population detectable |
| v2 | C1″ S5 conditioned-uniform | 75% (25%) | REJECTED — letter-count parity of short identity suffixes (100%/98% all-even at len 4/6) |
| v3 | C1‴ A5 five-cycles, L=16 | 70% (25%); per-gap: 98/73/59% at gaps 6/8/10 | REJECTED — leak decays with gap but far above bar |

- Gradedness round 1 (S5, fixed 2,500-epoch budget): **inconclusive — undertrained** (loss
  still descending at cutoff; the probe violated the trajectory rule and was redesigned
  convergence-gated). The convergence-gated rerun was gated on a falsifier pass and therefore
  has not run; it is the first action of D8b under fork option A.
- Two implementation bugs were caught by the probe/assertion layer before any build: the
  realized-t* contiguity scan (round 1), and a group-composition side error in the suffix
  sampler that was silently masked by involution symmetry in S5 and only surfaced on A5 —
  recorded because the same masked-luck pattern could affect any future group-word tooling.
- Probe scripts + result JSONs: session scratchpad `d8a_probes/` (gen_v2/gen_v3 samplers,
  falsifier + gradedness, per-round logs). On approval they are rewritten as
  `--selftest`/`--probe` modes of `depth_selection_toy.py` (D8b), not promoted as-is.
