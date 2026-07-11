# T1 pre-registration — the provably-deep-required toy

**Task:** `docs/plans/perinput_selector_2026-07-10/EXECUTION_PLAN.md` §2 WS-B, task T1.
F2's depth lane (`capacity_ladder_f2.py`, `capacity_ladder_results/F2/`) measured a NULL
per-input depth signal on toys G/G-flat/H. Toy G's compositional region (a sine modulated by
a linear envelope) turned out to carry an incidentally tiny per-input depth requirement, so
the null could mean either "the machinery can't see per-input depth structure" or "there was
never much structure to see." T1 builds a toy where the depth requirement is **provable** by
construction (a Telgarsky-style composed tent map, not merely "harder to fit smoothly"), so
the depth-lane machinery is tested where a large signal is guaranteed to exist by design.

## Toy (`make_toy_t1`, `_capacity_ladder_toys.py`)

`x ~ U[0, 1]`. Region A (`x < 0.5`): `y = 1.5x + eps` (linear, depth-1-sufficient). Region B
(`x >= 0.5`): `y = tent^(5)(2(x - 0.5)) + eps`, where `tent(z) = 1 - |2z - 1|` — the 5-fold
composed tent map, 32 linear pieces. `eps ~ N(0, 0.1^2)` in both regions. At the F2-style
trunk width used by this driver (**8**, not F2's 24 — the provable-capacity claim, "a depth-1
net realizes at most ~hidden_size linear kinks," is pinned to width 8 by the EXECUTION_PLAN.md
T1 spec), a depth-1 net provably cannot represent 32 pieces; depth >= 3 can.

## Harness (reused, not reinvented)

Nested-depth `FlexibleHiddenLayersNN` training + dedicated fixed-depth sweep: `capacity_ladder_f2.py`'s
`RunConfig`, `_build_model`, `_nested_all_depth_log_likelihood`,
`_independent_all_depth_log_likelihood`, `_fixed_depth_log_likelihood` — depths 1..6, BN off,
3 seeds {0,1,2}, 800 epochs, lr 5e-3 (F2's values, `hidden_size` overridden to 8). Per-input
readers: `capacity_ladder_x3.run_repeated_crossfit` (50 splits, Nadeau-Bengio SE) and
`capacity_ladder_x1.run_repeated` (which calls `hierarchical_perbin_stack`), called
**unmodified**, `n_bins=2`.

## Bars (locked before running)

**(i) CONSTRUCTION** — the dedicated fixed-depth sweep, read per region (paired plain
bootstrap SE, `capacity_ladder_f2._plain_boot_se`): region B's held-out NLL must improve
`> 2*SE` at d1->d2 **and** d2->d3; region A must show no `> 2*SE` gain past d1 (flat). If (i)
fails: redesign once (iterate up to 7x, width unchanged), re-run; fails again -> adjudicator.

**(ii) PER-INPUT READ — a heterogeneity test, regime-contingent** (rewritten per
`T1/BAR_II_ADJUDICATION.md`, fresh-context Opus, 2026-07-10; supersedes the original pooled-only
wording). The pooled "A-vs-B contrast" (`run_repeated_crossfit`'s `t_corrected` and
`run_repeated`'s `hier_vs_global.t_corrected`, both `n_bins=2`, unmodified) measures each region's
per-bin stacked-mixture advantage **over the pooled global stack**. This is a per-input
likelihood-**heterogeneity** statistic — nonzero only when a region's own optimal depth-mixture
differs from the pooled optimum in a way that improves that region's held-out score. It is **not** a
test of region B's *absolute* depth requirement; bar (i) tests that directly and unconfounded.
Verified property (selftest part c; `_bar_ii_probe.py`; confirmed on BOTH the pooled AND the
per-region-decomposed reader): on T1's asymmetric flat-A/dominant-B design the pooled global stack
converges onto region B's own optimum, so region B's advantage-over-global ≈ 0 regardless of its true
requirement magnitude, and a region A with no competing depth preference cannot pull the pool away —
both regions read ≈ 0 (region-B decomposed t = −0.84, mu = −0.0014). The per-region-decomposed reader
(`_repeated_perbin_by_region`) uses the **same** global baseline and is **equally blind** — it is a
reporting split, not a fix (this REFUTES the earlier "promote the decomposed read" proposal).
Machinery soundness is established **separately**, by the positive control (selftest part b: a
genuinely conflicting two-region table reads t ≈ 76 on both readers on the identical call sites).

The gated read stays the pooled corrected-t > 2 on **both** readers, on **≥ 2/3** seeds, exceeding
the region-A / G-flat control band (F2's G-flat table
`capacity_ladder_results/F2/nested_toyG_flat_seed{0,1,2}.pt`, re-read through the identical `n_bins=2`
machinery; X3/X1's own G-flat summaries use whole-domain terciles and are not reused). But whether
bar (ii) **can** fire on the real T1 toy is **contingent on region A's depth profile**, which the
construction run (bar i) reveals directly via region A's held-out-NLL-by-depth curve:
- **Region A INDIFFERENT** (held-out NLL ≈ flat across depths — the design intent: a width-8 net
  fits the linear region at every depth ≥ 1): bar (ii) is a **structural zero**, not a machinery
  failure — there is no per-input *likelihood* payoff to detect, because a global always-deep stack
  loses nothing on region A. The per-input depth payoff is then a **compute** saving (route region A
  to depth 1), measured downstream in **H2** — not a likelihood advantage measurable here.
- **Region A ACTIVELY HURT by depth** (held-out NLL *degrades* past d1 — a deep width-8 net
  overfitting the linear region; this STILL satisfies bar (i)'s flatness, which forbids only
  significant *improvement* past d1, confirmed by `_construction_bar` on the conflicting table →
  `construction_pass=True`): genuine per-input heterogeneity exists (region A prefers d1, region B
  prefers ≥ d3), and bar (ii) **should fire**, in proportion to region A's opposing-preference
  magnitude. A null bar (ii) in **this** regime is a genuine concern.

**Secondary, reported not gated:** terciles within region B (`n_bins=3` on the `x >= 0.5`
subset only) — checks whether the region-B advantage is uniform across the 32-piece
oscillation or concentrated in part of it.

## Outcome semantics (locked)

(Rewritten per `T1/BAR_II_ADJUDICATION.md` — regime-contingent, replacing the original two-case
block. Read region A's construction NLL-by-depth curve to pick the branch.)

- **PASS (i) + bar (ii) FIRES** (corrected-t > 2 on both readers, ≥ 2/3 seeds, exceeding the
  region-A / G-flat control band): machinery **VALIDATED** on a real per-input heterogeneity signal;
  the F2 depth-lane null is reframed "toy-specific signal absence, instrument sound"; **H2 UNLOCKED**.
- **PASS (i) + bar (ii) NULL, region A INDIFFERENT** (construction curve flat across depth) with
  region-B decomposed advantage **near-zero-not-negative**: the **documented structural zero**, NOT a
  machinery failure — the advantage-over-global reader cannot see an absolute-only depth requirement,
  and the positive control (t ≈ 76) confirms the reader is sound. The depth lane's per-input value is
  **not disproven**; it is a **compute** question deferred to H2. → **STOP + fresh-context
  adjudicator** (plan requirement for any FAIL(ii)); with (construction region-A flat) + (region-B
  decomposed ≈ 0) + (positive control t ≈ 76) in hand, the adjudicator rules whether to **unlock H2**
  on the compute-payoff premise. Because that unlock overrides the plan's original "PASS(i)+FAIL(ii)
  → H2 stays locked," the unlock decision is a **G-FORK** for the user.
- **PASS (i) + bar (ii) NULL, region A ACTIVELY HURT by depth** (construction curve shows a real
  opposing preference — NLL degrades past d1) yet the readers do not fire: this is the **genuine
  machinery-failure signature** the original bar intended → STOP + adjudicator, top-priority surprise;
  **H2 stays locked**.
- **FAIL (i)** -> redesign once (iterate up to 7x, width unchanged — first width 6, then a deeper
  tent^6 — before touching anything else), re-run; fails again -> adjudicator.

## KNOWN READOUT PROPERTY (discovered while building this script's selftest)

Verified against the real, unmodified `capacity_ladder_x3.run_repeated_crossfit` and
`capacity_ladder_x1.run_repeated` — not an artifact of this driver. `stack_em`'s single shared
"global" mixture is fit by maximizing *total* log-likelihood pooled across every region, so it
converges onto whichever region has the *larger* nat-scale depth gaps. By T1's design that is
region B (the 32-piece tent map dominates a linear region in achievable log-likelihood gain).
Once the global mixture already matches region B's own optimum, region B's *own* per-bin stack
adds close to zero advantage over it — **regardless of how large region B's true depth
requirement is** — while a genuinely flat region A (the expected behaviour of a linear
function, fit equally well at any depth >= 1) cannot pull the global mixture away from region
B either, so it also reads null.

A minimal synthetic check (true-flat region A + a planted +0.8-nat region-B step, mirroring
T1's literal region shape) reads `t_corrected` in `[-0.7, +0.4]` on both readers despite the
large planted signal (`capacity_ladder_t1.run_selftest`, part (c) — printed every selftest run
as a non-gated diagnostic). A comparable-magnitude, genuinely *conflicting* two-region table
(the X3/X1 selftest style: both regions have a real, opposing preference) reads
`t_corrected ~ 76` on both readers on the same machinery — confirming the pooled statistic
works correctly when there is real tension between regions, and confirming the near-null
result above is a property of the readout given T1's asymmetric (dominant-B, flat-A) design,
not a bug.

**Practical consequence:** bar (ii) can plausibly read NULL on the real T1 toy even when bar
(i) shows a large, genuine region-B depth requirement — see the regime-contingent bar (ii) and
outcome semantics above, which now govern how such a null is interpreted.

**CORRECTION (2026-07-10, `T1/BAR_II_ADJUDICATION.md`):** an earlier draft of this note claimed
`capacity_ladder_t1.py`'s `ab_decomposed_by_seed_DIAGNOSTIC_NOT_GATED` field (`_repeated_perbin_by_region`,
region-A vs region-B read separately) could tell a structural zero apart from a machinery failure.
**That is REFUTED:** the decomposed reader uses the SAME pooled global baseline (`_split_perbin_advantage`
diff `indep_ls − global_ls`), so it is EQUALLY blind — on the planted flat-A/+0.8-nat-B table its
region-B read is t = −0.84, mu = −0.0014 (`_bar_ii_probe.py`), the same near-null as the pooled read.
The structural-zero-vs-machinery-failure discrimination is made instead by **region A's construction
NLL-by-depth curve** (indifferent ⇒ structural zero; hurt-by-depth ⇒ genuine concern), per the
regime-contingent outcome semantics above. The decomposed read remains a reported diagnostic only.

## Selftest (`--selftest`, no training)

(a) Construction-bar logic recovers a planted region-B step (flat region A, +0.8 nat region-B
advantage at d1->d3) via `_construction_bar`. (b) Per-input machinery recovers a planted
advantage on a genuinely conflicting two-region table (X3/X1 selftest style) via the literal,
unmodified `run_repeated_crossfit`/`run_repeated` call sites the real read uses. (c) Diagnostic
printout (not gated) of the same machinery on T1's literal flat-A/dominant-B region shape,
demonstrating the KNOWN READOUT PROPERTY above.

## Non-goals

No library model changes; no N-sweeps (P1's scope); no router/selector (H2's scope, gated on
this task's PASS); does not run the full real 3-seed matrix (orchestrator-owned).
