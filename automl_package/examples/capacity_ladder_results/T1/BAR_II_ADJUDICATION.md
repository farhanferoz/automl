# T1 bar-(ii) — fresh-context adjudication (2026-07-10)

**Adjudicated:** the orchestrator's bar-(ii) finding
(`BAR_II_FINDING_FOR_ADJUDICATION.md`), which overturns both RESUME.md's stated fix
("promote the decomposed per-region read to be the gated bar-(ii) statistic") and the
plan's premise for T1 bar (ii). Verified independently from the numbers and the code —
neither the finding nor RESUME's premise was accepted on faith.

**Bottom line:** RESUME's fix is **REFUTED**. The finding's core empirical claim and
mechanism are **CONFIRMED**. The finding's *strong* structural-tension claim ("the two
bars cannot both pass short of region A being hurt by depth, which would fail bar (i)'s
flatness") is **PARTIALLY REFUTED** — region A hurt by depth does NOT fail bar (i), so
the two bars CAN co-pass, and bar (ii) is a structural zero **only** in the
truly-indifferent-A regime. Disposition = **R1 (reframe within plan) with corrected,
regime-contingent semantics** (not the note's unconditional "structural zero"). The
construction run (bar i) **may proceed now**. Reframe = **WITHIN-PLAN**; one specific
downstream decision (unlocking H2 on a structural-zero bar (ii)) is reserved as a
**G-FORK** but only *arises after* the construction data and need not be raised now.

---

## Evidence reproduced (independent)

All three runs on this machine, `AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python`.

**1. Orchestrator probe** (`_bar_ii_probe.py`) — reproduces the note's tables to the digit:

| table | reader | crossfit/indep t | hier t |
|---|---|---:|---:|
| PLANTED flat-A / +0.8-nat-B | POOLED (bar ii as written) | −0.82 | −0.01 |
| | DECOMPOSED region_A | −0.35 (mu −0.0012) | +0.01 (mu +0.0000) |
| | DECOMPOSED region_B | **−0.84 (mu −0.0014, passfrac 0.00)** | **−0.02 (mu −0.0000, passfrac 0.02)** |
| CONFLICTING (positive control) | POOLED | +72.31 | +70.78 |
| | DECOMPOSED region_A | +13.07 (mu +0.3365) | +13.08 (mu +0.3348) |
| | DECOMPOSED region_B | +10.26 (mu +0.2733) | +10.22 (mu +0.2705) |

**2. Selftest cross-check** (`capacity_ladder_t1.py --selftest`) → PASS: (a) construction
`region_b_pass=True region_a_flat=True`; (b) positive control `crossfit_t=75.93
hier_t=76.21`; (c) diagnostic `crossfit_t=−0.70 hier_t=0.40` (near-null on flat-A/dominant-B
despite the planted +0.8-nat B step).

**3. Discriminating experiment (this adjudication)** — `_construction_bar` on the
*conflicting* table (`means_a=[0.8,0.4,0,0,0,0]`, i.e. region A HURT by depth = the same
table used as bar (ii)'s positive control):

```
region_b_pass = True   (d1->d2 beats=True, d2->d3 beats=True)
region_a_flat = True                                <-- flatness HOLDS
    region_A d1_to_d2: mean_delta=-0.4111 beats_2se=False
    region_A d1_to_d3: mean_delta=-0.8083 beats_2se=False   (all past-d1 deltas negative)
    ...
construction_pass = True
```

Because `_construction_bar` (`capacity_ladder_t1.py:145-146`) sets `region_a_flat = not
any(gain > 2·SE)` and `_region_gain` (L125-130) tests only `mean_delta > 2·SE`, a region
A that is HURT by depth (negative deltas) satisfies flatness. So on the conflicting table
BOTH bar (i) passes (`construction_pass=True`) AND bar (ii) fires (pooled t≈72) —
demonstrating the two bars CAN co-pass.

---

## Q1 — VERIFY the finding

### CONFIRMED — the empirical claim
Decomposed region-B on the planted +0.8-nat table reads **t = −0.84, mu = −0.0014** (≈ 0),
as blind as the pooled read (t = −0.82). On the conflicting positive control every reader
fires (pooled t ≈ 72; decomposed region-A ≈ +13, region-B ≈ +10). Matches the note
exactly. **RESUME's "promote the decomposed read" fix does not fix bar (ii).**

### CONFIRMED — the mechanism
Both the pooled readers (`x1.three_arm_split` L140-159; `x3.run_repeated_crossfit`) and the
decomposed reader (`t1._split_perbin_advantage` L205-217) compute a per-example
`perbin_ls − global_ls`, where `global_ls = mixture_logscore(pi_global, ·)` and `pi_global
= stack_em(score_fit)` is the pooled stack fit over **all** fit rows (`_capacity_ladder.
stack_em` L142-168, `mixture_logscore` L171-188). This is an *advantage-over-the-pooled-
optimum* = a per-input **heterogeneity** statistic: nonzero only when a region's own
optimal depth-mixture differs from the pooled optimum in a way that improves that region's
held-out score. The decomposed reader uses the **identical** `pi_global` baseline (L205,
L209), so it inherits the identical blindness — it is a per-region *reporting split*, not a
different estimator. On T1's asymmetric design region B (0.8-nat gaps) dominates the pool
so `pi_global ≈ pi_indep[B]` → region B's advantage ≈ 0 regardless of its true requirement
magnitude; a truly-flat region A has no competing preference to pull the pool away → its
advantage ≈ 0 too. Confirmed by the contrast (planted → all ≈ 0; conflicting → all fire).

### PARTIALLY REFUTED — the structural-tension claim
The finding's core is correct: **a truly-indifferent region A ⟺ bar (ii) structural zero**
(no heterogeneity for a pooled advantage-over-global reader to detect). But the finding's
*stronger* wording — "the two bars cannot both pass by construction (short of region A
being actively hurt by depth, **which would fail bar (i)'s flatness**)" — is **false**.
Bar (i)'s `region_a_flat` forbids only significant *improvement* past d1; a region A that
is *hurt* by depth (negative deltas → prefers d1) passes flatness (experiment 3 above:
`construction_pass=True` on the hurt-by-depth table) **and** makes bar (ii) fire (t≈72).
So the two bars CAN co-pass — precisely in the hurt-by-depth regime, where region A has a
real *opposing* depth preference. This correction is load-bearing: it means bar (ii) is a
structural zero **conditionally** (indifferent-A only), not unconditionally, and the
condition is exactly the empirical question the construction run resolves.

---

## Q2 — RULING ON THE FIX

**RESUME's "promote the decomposed per-region read": NO — refuted.** The decomposed reader
shares the pooled `pi_global` baseline and is equally blind (region-B decomposed t = −0.84,
mu = −0.0014 on the planted table). It does not recover the planted signal and must not be
gated as the bar-(ii) statistic.

**Disposition: R1 (reframe within plan), with corrected regime-contingent semantics.**
Not the note's unconditional "bar (ii) is a structural zero / inapplicable" — that is
right only for the indifferent-A regime and silently discards the hurt-by-depth regime in
which bar (ii) is a *valid* positive test. **Not R2** (redesigning the toy to force
region-A-hurt-by-depth): unnecessary *now* — the construction run reveals for free which
regime the real region A is in; only revisit R2 if that run shows indifferent-A AND the
user later wants a positive likelihood-bar-(ii) demonstration (see contingency below).
**Not R3.**

### Corrected bar-(ii) text (write into `PREREGISTRATION.md`, replacing bar (ii))

> **(ii) PER-INPUT READ — a heterogeneity test, regime-contingent.** The pooled "A-vs-B
> contrast" (`run_repeated_crossfit`'s `t_corrected` and `run_repeated`'s
> `hier_vs_global.t_corrected`, both `n_bins=2`, unmodified) measures each region's per-bin
> stacked-mixture advantage **over the pooled global stack**. This is a per-input
> likelihood-**heterogeneity** statistic — nonzero only when a region's own optimal
> depth-mixture differs from the pooled optimum in a way that improves that region's
> held-out score. It is **not** a test of region B's *absolute* depth requirement; bar (i)
> tests that directly and unconfounded. Verified property (selftest part c; `_bar_ii_probe.
> py`; confirmed on BOTH the pooled and the per-region-decomposed reader): on T1's
> asymmetric flat-A/dominant-B design the pooled global stack converges onto region B's own
> optimum, so region B's advantage-over-global ≈ 0 regardless of its true requirement
> magnitude, and a region A with no competing depth preference cannot pull the pool away
> from B — both regions read ≈ 0 (region-B decomposed t = −0.84, mu = −0.0014). The
> per-region-decomposed reader (`_repeated_perbin_by_region`) uses the **same** global
> baseline and is **equally blind** — it is a reporting split, not a fix. Machinery
> soundness is established **separately**, by the positive control (selftest part b: a
> genuinely conflicting two-region table reads t ≈ 76 on both readers on the identical call
> sites).
>
> Whether bar (ii) can fire on the **real** T1 toy is **contingent on region A's depth
> profile**, which the **construction run (bar i) reveals directly** via region A's
> held-out-NLL-by-depth curve:
> - **Region A INDIFFERENT** (held-out NLL ≈ flat across depths — the design intent: a
>   linear region a width-8 net fits at every depth ≥ 1): bar (ii) is a **structural zero**,
>   not a machinery failure. There is no per-input *likelihood* payoff to detect, because a
>   global "always-deep" stack loses nothing on region A. The per-input depth payoff is then
>   a **compute** saving (route region A to depth 1), measured downstream in **H2** — not a
>   likelihood advantage measurable here.
> - **Region A ACTIVELY HURT by depth** (held-out NLL *degrades* past d1 — a deep width-8
>   net overfitting the linear region; this still satisfies bar (i)'s flatness, which
>   forbids only significant *improvement* past d1): genuine per-input heterogeneity exists
>   (region A prefers d1, region B prefers ≥ d3), and bar (ii) **should fire**, in
>   proportion to region A's opposing-preference magnitude. A null bar (ii) in **this**
>   regime is a genuine concern.

### Corrected outcome-semantics (locked; replace the T1 outcome-semantics block)

> - **PASS (i) + bar (ii) FIRES** (corrected-t > 2 on both readers, ≥ 2/3 seeds, exceeding
>   the region-A / G-flat control band): machinery **VALIDATED** on a real per-input
>   heterogeneity signal; the F2 depth-lane null is reframed "toy-specific signal absence,
>   instrument sound"; **H2 UNLOCKED**.
> - **PASS (i) + bar (ii) NULL, region A INDIFFERENT** per the construction curve (NLL flat
>   across depth) with region-B decomposed advantage **near-zero-not-negative**: the
>   **documented structural zero**, NOT a machinery failure — the advantage-over-global
>   reader cannot see an absolute-only depth requirement, and the positive control (t ≈ 76)
>   confirms the reader is sound. The depth lane's per-input value is **not disproven**; it
>   is a **compute** question deferred to H2. → **STOP + fresh-context adjudicator** (as the
>   plan requires for any FAIL(ii)); with (construction region-A flat) + (region-B
>   decomposed ≈ 0) + (positive control t ≈ 76) in hand, the adjudicator rules whether to
>   **unlock H2** on the compute-payoff premise. Because that unlock overrides the plan's
>   original "PASS(i)+FAIL(ii) → H2 stays locked," the unlock decision is a **G-FORK** for
>   the user.
> - **PASS (i) + bar (ii) NULL, region A ACTIVELY HURT by depth** per the construction curve
>   (real opposing preference present) yet the readers do not fire: this is the **genuine
>   machinery-failure signature** the original bar intended → STOP + adjudicator,
>   top-priority surprise; **H2 stays locked**.
> - **FAIL (i)** → redesign per the plan's escalation (up to 7×; width unchanged — first
>   width 6, then a deeper tent^6 — before touching anything else), re-run; fails again →
>   adjudicator.

### R2 redesign spec (held in reserve — do NOT execute now)
Only if the construction run shows **indifferent-A** *and* the user later wants a positive
*likelihood* bar (ii): give region A a real, opposing depth preference of magnitude
comparable to region B's (e.g. region A = a signal whose best held-out fit is at depth 1
and which a deep net measurably overfits, so its NLL-by-depth curve *declines* by ≳ 0.3 nat
d1→d6). This converts bar (ii) into a valid positive heterogeneity test (co-passes bar (i),
per experiment 3), at the cost of coupling the depth-*requirement* story (region B) with an
*overfitting* story (region A) and weakening the clean "provably-deep-required" framing. It
is strictly unnecessary to answer the question the construction run already answers for
free, which is why it is reserve-only.

---

## Q3 — GOVERNANCE (WITHIN-PLAN vs G-FORK)

Per EXECUTION_PLAN §0c (adjudicator certifies failed bars / surprises, fresh context) and
§0d (G-FORK = "a fresh-context adjudicator has examined a failed bar or surprise and
explicitly ruled it a user-level design fork").

**WITHIN-PLAN — the orchestrator may apply and proceed:**
1. Reject RESUME's "promote the decomposed read" fix (factual refutation of a proposed
   fix).
2. Rewrite the PREREGISTRATION bar-(ii) text + outcome-semantics per R1 above. This is a
   **refinement the plan already anticipated**: the prereg's own "KNOWN READOUT PROPERTY"
   note (PREREGISTRATION.md L63-94) already documents the blind spot and explicitly
   delegates the "structural zero vs machinery failure" call to the adjudicator in any
   PASS(i)/FAIL(ii) pass. The R1 rewrite adds no work, no compute, no new toys, and no
   user-facing commitment; it makes the interpretation the prereg gestures at precise and
   corrects the one over-strong claim. Note it also corrects a latent defect in the plan's
   own selftest expectation (§2 L222 expected the planted +0.8-nat B advantage to be
   "recovered by both readers" — the readout property shows it cannot be; the orchestrator's
   construction-bar selftest already routes that check to the direct per-region NLL read,
   which is correct).
3. Run the construction battery (Q4).

**G-FORK — reserved for the user, but does NOT arise now:** the single decision to
**unlock H2 on a structural-zero bar (ii)** (i.e. treat "PASS(i) + indifferent-A + null
bar (ii)" as "machinery validated, per-input depth payoff is compute-only, proceed to H2")
overrides the plan's locked "PASS(i)+FAIL(ii) → H2 stays locked" and commits downstream
compute on a reframed premise. That is user-level. It only becomes live *after* the
construction run shows indifferent-A and bar (ii) reads null; the plan's own locked
STOP→adjudicator routing handles the intermediate step, so **nothing needs to go to the
user at this time.** If instead the construction run shows hurt-by-depth-A, bar (ii)
becomes a valid positive test and no fork is needed at all.

---

## Q4 — MAY THE CONSTRUCTION RUN (bar i) PROCEED NOW?

**YES — unconditionally, now.** Bar (i) is independent of the bar-(ii) dispute (separate
statistic: dedicated fixed-depth sweep read per region, `_construction_bar`), and its
region-A held-out-NLL-by-depth curve is precisely the **discriminating experiment** that
resolves the flat-vs-hurt-by-depth contingency on which the entire bar-(ii) disposition
turns. Running it is strictly within-plan (§0d sequences the T1 run; §2 bar (i) is
unchanged by this ruling). Proceed; read region A's NLL-by-depth curve; then apply the
corrected regime-contingent semantics above.

---

## Uncertainty (what remains unverifiable here)
- Which regime the **real** T1 toy occupies (indifferent-A vs hurt-by-depth-A) is
  unresolved until the construction run trains the actual width-8 nets — that is the point
  of Q4. The synthetic probe models both regimes but does not tell us which the trained
  toy realizes.
- Whether, in a hurt-by-depth-A regime, region A's opposing-preference magnitude is large
  enough (relative to region B's pooled dominance) to push the pooled advantage-over-global
  above t=2 is also empirical; the construction curve's region-A NLL drop d1→d6 is the
  quantitative predictor (the positive control needed a ~0.8-nat opposing swing to reach
  t≈72; a mild overfit may not clear the bar even though heterogeneity exists).

## Recommendation
**Proceed (within-plan).** (1) Reject RESUME's decomposed-read fix. (2) Apply the R1
bar-(ii) rewrite + regime-contingent outcome-semantics above to `PREREGISTRATION.md`.
(3) Run the construction battery now. (4) Interpret bar (ii) via the corrected semantics
once the region-A NLL-by-depth curve is in hand; escalate to the user (G-FORK) only if that
curve shows indifferent-A, bar (ii) reads null, and the orchestrator wants to unlock H2 on
the compute-only premise. Do not run R2.
