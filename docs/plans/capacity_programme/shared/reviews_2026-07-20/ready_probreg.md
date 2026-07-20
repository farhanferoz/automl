# Autonomy readiness review — docs/plans/capacity_programme/probreg.md (2026-07-20)

Reviewed every OPEN task (PA, PB, PC, P1, P2, P3, P4, P5, P6). P0/P0b DONE claims spot-checked
per rubric point 3. PT (parked) noted only. All citations below were opened and checked against
disk at the paths/lines given; none were taken from the plan's own prose.

---

## Section 1 — Verdict table

| Task | Verdict | Biggest gap |
|---|---|---|
| **PA** | **NOT-DISPATCHABLE** | M1's build target is undetermined. The only "Mechanism" probreg.md §1 names for M1 (`held_out_arbiter_advantage`) is structurally the wrong shape (per-input `(N,)` array, binary top-rung-vs-bypass only — verified below). The right-shaped primitive (`all_rung_log_likelihood`) exists, has zero callers anywhere, and is named in neither probreg.md nor benchmark_spec.md. benchmark_spec.md §2.1 — the doc PA's own header cites as having settled this — contains, in the SAME block, both "RESOLVED 2026-07-20" and, three paragraphs below it, "Do not improvise this... until it is settled, M1 is unbuildable" for the same three open questions. |
| **PB** | **DISPATCHABLE-WITH-FIX** | No `Files (write set):` line at all (every other task in this plan has one). Its own orchestration note hedges the parallel-safety claim on a condition ("disjoint from PC's write set **if** driven by separate scripts") that nothing in the task fixes. Fix: add explicit driver-script path + results-dir naming convention. |
| **PC** | **DISPATCHABLE-WITH-FIX** | Same defect as PB: no `Files (write set):` line, no named driver or output path for "a sensitivity table on disk." |
| **P1** | **DISPATCHABLE** | None material. Files named exactly, fix is copy-the-existing-pattern decision-complete, verify line requires prove-it-fails + checksum-restore. Best task in the plan. |
| **P2** | **DISPATCHABLE** | Verify line is softer than the D1/D2 precedent it's supposed to match (no prove-it-fails requirement if the conclusion is "real regression"). Minor; not blocking. |
| **P3** | **NOT-DISPATCHABLE** | Two independent problems. (a) `deps: P1` omits PA and PB, contradicting the master Order line (`P0 → P0b → P1 → P2 → PA → PB ∥ PC → P3 → ...`) and §3.6's constants-artifact discipline, which P3 needs (M1 needs a selection fraction) but which is textually binding only on P4. (b) A genuine, verified contradiction: benchmark_spec.md §2.0/§2.3 mandate M3's per-k models be k-dropout/NESTED-trained ("Training is therefore NOT a variable"), but P3's own verify line requires M3 to "reproduce §3.2" — and §3.2's reference models were "a separately trained **ordinary** model" (non-dropout), which is what the current code (`_probreg_fixed`, `NClassesSelectionMethod.NONE`) actually builds. §3.2's own headline finding is that k-dropout-read-at-k *disagrees* with the ordinary dedicated model at middle k in 8/9 cases — so if M3 is built correctly per §2.3, the "reproduces §3.2" positive control is expected to fail by the plan's own prior finding. |
| **P4** | **DISPATCHABLE-WITH-FIX** (own text is fine; inherits PA/P3) | No new defect in P4's own text — files, deps, verify are all decision-complete and internally consistent with the master Order. But it cannot run correctly until PA and P3's gaps above are closed, since it consumes both. |
| **P5** | **DISPATCHABLE** | None material in its own text; inherits P3's problems transitively (depends on P3's artifacts). |
| **P6** | **DISPATCHABLE** | None material in its own text; inherits P4/P5 transitively. |

**PT** — parked, correctly marked, not a dependency of anything; not reviewed further.
**P0, P0b** — marked DONE; verify conditions were actually re-executed by me (not just "outputs exist" — see Section 2).

---

## Section 2 — Per-task detail (everything not plain DISPATCHABLE)

### PA — NOT-DISPATCHABLE

PA's header claims `✅ decisions settled 2026-07-20 — DISPATCHABLE` and its body claims "✅ Both
blocking decisions SETTLED by the user 2026-07-20": (1) the tolerance rule, (2) deleting
`inference_mode`. Both of those really are settled and clearly specified — no complaint there.

The problem is that **"both blocking decisions" is not actually "all blocking decisions."** The
document PA cites for having specified the rule, `docs/probreg_benchmark/benchmark_spec.md` §2.1
(lines 212–248), itself lists **three** things that must be decided before M1 can be built:

> 1. What curve is read — per-rung held-out likelihood across k=1..10, or the arbiter statistic
>    generalised from top-vs-bypass to rung-vs-rung?
> 2. What rule picks k off it — argmax, or cheapest-within-tolerance? ...
> 3. Does the bypass rung k=1 compete?

PA's "both decisions settled" only clearly answers #2. Worse, benchmark_spec.md §2.1 is
internally self-contradictory about its own status: lines 212–227 open with "✅ RESOLVED
2026-07-20 (user). M1's selection procedure is now specified" and state "Rung 1 (the bypass)
competes" (answering #3) — but the SAME numbered block, retained a few lines below (lines
229–248, explicitly headed "Original finding, retained as the reason this block exists"), still
poses all three questions as open and ends: **"Do not improvise this. Escalated to the user;
until it is settled, M1 is unbuildable and the battery cannot run."** Nothing in either document
records that #1 and #3 were actually re-settled after that escalation; the "RESOLVED" banner and
the "unbuildable" conclusion currently coexist in the same section.

**The specific defect requested for verification, confirmed independently:**

- `held_out_arbiter_advantage` (`automl_package/models/probabilistic_regression.py:845-916`)
  returns a **per-input** `(N,)` array — confirmed by reading the full function body: it computes
  `ll_top - ll_bypass` per sample (`:907`), then neighbour-averages over a Euclidean box-car
  kernel (`:909-915`) and returns `out` of shape `(N,)`. Its own docstring says so explicitly:
  `"""Certified per-input capacity readout..."""` (`:852`) and `"Returns: (N,) neighbour-averaged
  advantage A(x) at every row of x"` (`:882-885`). It compares **only** the top rung (`top_k`,
  default `max_n_classes_for_probabilistic_path`) against the k=1 bypass — never any of the
  intermediate rungs `2..k_max-1`. This is not "ONE k for the dataset" in any sense; it can't
  even express "the model picked k=4."
- `all_rung_log_likelihood` (`automl_package/models/selection_strategies/n_classes_strategies.py:230-252`)
  is the primitive that *would* give the right shape — a full `(batch, n_classes)` per-rung
  likelihood table, exactly what "score every rung k=1..k_max" (benchmark_spec.md §2.1's resolved
  rule text, line 215) requires. Confirmed via `grep -rn "all_rung_log_likelihood" automl_package/
  tests/` outside its own definition: **zero matches** — zero callers anywhere in the codebase.
  Its own docstring (lines 234-235) claims *"This is the all-rung score table
  `ProbabilisticRegressionModel.held_out_arbiter_advantage` ... consume[s]"* — **this claim is
  false**: `held_out_arbiter_advantage` calls `_per_sample_log_likelihood_at_k` twice (bypass,
  top_k) and never calls `all_rung_log_likelihood` or `all_rung_outputs`. That's a second, small,
  independent defect (a stale/wrong docstring) worth a one-line fix note, separate from the main
  finding.
- `grep -n "all_rung_log_likelihood" docs/plans/capacity_programme/probreg.md
  docs/probreg_benchmark/benchmark_spec.md` → **zero hits in both files.** The one function that
  already exists and is shaped correctly for what PA needs to build is never mentioned in either
  planning document. benchmark_spec.md's own defect search (line 234: "checked by grep for
  `select_k`/`choose_k`/`best_k`/`knee`/`elbow`") did not include a search term that would have
  found it, because `all_rung_log_likelihood` isn't itself a selector — it's the scoring table a
  selector would consume.

**Net effect:** a zero-context worker dispatched on PA today, reading only PA's text plus §1 (whose
only cited "Mechanism" for M1 is `held_out_arbiter_advantage`), has a real chance of either (a)
trying to repurpose `held_out_arbiter_advantage` — the only mechanism named — which cannot produce
a multi-way k choice by construction, or (b) reinventing a per-rung scoring loop from scratch
in violation of the repo's search-before-write rule, because the existing correct primitive
(`all_rung_log_likelihood`) isn't surfaced anywhere. Given benchmark_spec.md's own "do not
improvise this" instruction sitting unresolved in the cited section, this is a genuine blocker,
not a nitpick.

**Fix that would make this DISPATCHABLE:** name `all_rung_log_likelihood` /
`all_rung_outputs` explicitly as the primitive PA builds M1's selector on top of; explicitly
restate (in probreg.md, not just implicitly in benchmark_spec.md's contradictory block) that
decision #1 = "per-rung held-out likelihood table" and decision #3 = "yes, bypass competes";
delete or reconcile benchmark_spec.md §2.1's contradictory RESOLVED/unbuildable text so only one
status survives.

### PB / PC — DISPATCHABLE-WITH-FIX

Every other task in this plan (P0, P0b, PA, P1, P2, P3, P4, P6) has an explicit `**Files (write
set):**` line. PB and PC do not — both jump straight from **Why**/**What exists** to **Spec** with
no file-level scope at all. Consequences:

- Rubric point 1 (write-set / parallel-collision check) is **not answerable from the task text**
  for either task. PB's own orchestration line hedges this itself: `parallel: yes (disjoint from
  PC's write set **if** driven by separate scripts)` — the "if" is never resolved by anything in
  either task.
- Neither task names a driver script path or a results-artifact naming convention. PB says "one
  JSON per (toy, seed, fraction, arm)" and PC says "a sensitivity table on disk" — no path, no
  filename pattern. §3.6's constants table compounds this: it says PB's fraction and PC's
  router-sensitivity results are read "from disk" by P4, but the "owning artifact" column for both
  says only "the PB sweep JSON — path fixed by PB's task, one file, named there" — i.e. the path is
  supposed to be fixed *by PB's own task text*, and it currently is not.

Fix: add a `Files (write set):` line to each naming the driver script(s) and the results directory
pattern, so (a) the parallel-safety claim becomes checkable and (b) §3.6's constants table has an
actual path to point at instead of a promise that PB's task will supply one.

### P3 — NOT-DISPATCHABLE

**Gap (a) — dependency line vs. master order and vs. §3.6.**
P3's own line reads: `deps: P1 (the arbiter must be correct before its choices mean anything)`.
The plan's master sequencing (probreg.md:227) is `P0 → P0b → P1 → P2 → PA → PB ∥ PC → P3 → P4 →
P5 → P6` — i.e. PA, PB, and PC are all upstream of P3 in the stated order, but P3's own
machine-readable `deps:` field lists only P1. Given the plan's own "Compilation note" (line
220-223) says the intent is to make this "compilable to a deterministic Workflow ... rather than
relying on dispatcher discipline," a per-task `deps:` field that omits real upstream requirements
is exactly the kind of gap that would silently produce a wrong dispatch order once this becomes
mechanical. Concretely, P3 needs PA (there is no M1 to train without it) and needs PB's frozen
selection fraction (M1's held-out cal split has to come from somewhere, and §3.6 exists
specifically so "a battery that runs at an unjustified selection fraction... produces exactly the
unattributable result this strand exists to prevent" — but §3.6's binding "Feed-forward rule" text
(line 207-209) names only **P4**, not P3, as required to fail loudly on a missing constant).

**Gap (b) — M3's training scheme is asked to be two incompatible things at once.**
This is the more serious one, and it's a genuine, previously-undocumented contradiction between
two live sections of the plan set:

1. `docs/probreg_benchmark/benchmark_spec.md` §2.0 (lines 162-167, the section explicitly headed
   "Definition of record: `docs/plans/capacity_programme/probreg.md` §1"): **"All three ProbReg
   models train identically — with k-dropout (`NClassesSelectionMethod.NESTED`, per-sample `k ~
   Uniform{1..k_max}`). Training is therefore NOT a variable."** §2.3 (lines 275-279) restates this
   for M3 specifically: **"Train a separate k-dropout model for each `k` in the sweep grid."**
2. probreg.md's P3 verify line (lines 502-503): **"the M3 positive control reproduces §3.2 before
   any M1 number is read."** §3.2 is `docs/reports/probreg_kselection/probreg_kselection.md`
   §3.2, and that section's own text (lines 282-284) describes its reference model as **"a
   separately trained *ordinary* model built to use exactly `c` classes and no others"** — i.e.
   explicitly NOT k-dropout. This matches the current code: `select_k_for_toy`
   (`automl_package/examples/report_a_benchmark.py:331`), the function §1's own table (line 34)
   and §2.3 (line 281-286) both cite as M3's mechanism to "generalise," calls `_probreg_fixed`
   (`report_a_benchmark.py:185-191`), which constructs the model with
   `n_classes_selection_method=NClassesSelectionMethod.NONE` — fixed, no dropout. Verified by
   `grep -n "n_classes_selection_method" automl_package/examples/report_a_benchmark.py`.
3. §3.2's own headline finding (probreg.md:74-80, and the report text lines 300-314) is that the
   k-dropout model read at a fixed k **measurably disagrees** with the dedicated-ordinary model at
   middle k in **8 of 9** audited cases — that disagreement is the entire subject of the coherence
   check. So if P3 builds M3 correctly per §2.0/§2.3 (k-dropout, ceiling=k), the positive control
   ("M3 reproduces §3.2") is — by the plan's own prior finding — likely to *fail* at middle k, not
   because the new driver is buggy, but because it is doing something different in kind from what
   §3.2 measured.

Neither document flags this. It is not covered by the "Anchor warning" text at probreg.md:211-218
either — that text says P3's coherence half must NOT anchor *solely* on a re-run of §3.2 through
the same harness, and instructs pairing it with "the dedicated per-k models of M3, which are
trained independently" — but "independently" there most plausibly means "as separate trained
model instances," not "using a different training scheme than §2.0/§2.3 mandate," and it does not
resolve which training scheme those dedicated M3 models actually use.

**Consequence:** a worker dispatched on P3 must invent an undocumented resolution to a real
contradiction — either building M3 as k-dropout/NESTED (per the frozen §2.0/§2.3, consistent with
§1's "training is not a variable" claim) and accepting that the stated positive-control check will
likely not literally reproduce §3.2's numbers at middle k (which nothing in the plan authorises
them to accept — MASTER Decision 14 says a failed positive control **HALTS the battery**, so this
would trigger a false HALT); or building M3 as ordinary/non-dropout (matching the current code and
literally reproducing §3.2), which reintroduces the exact "differs in two things at once" confound
that probreg.md §1.1 says invalidated the *old* spec and that this whole rewrite exists to
eliminate. Either resolution is a real design decision the plan currently makes for the worker only
by accident, in opposite directions, in two different sections.

---

## Section 3 — Plan-level findings

### 3.1 The requested defect check — confirmed, and more precisely characterised than initially reported

`held_out_arbiter_advantage` does return a per-input array, not a global value — confirmed exactly
(see PA detail above; shape `(N,)`, docstring explicitly "per-input capacity readout"). It is also
not merely "the wrong shape for a global answer" — it doesn't even do multi-way k comparison; it
is a fixed binary comparison of the top rung against the k=1 bypass, smoothed over an x-neighbourhood.
`all_rung_log_likelihood` — the primitive that would actually supply "score every rung k=1..k_max,"
which benchmark_spec.md's own (self-contradictorily) "RESOLVED" text requires — has zero callers,
confirmed by grep across `automl_package/` and `tests/`, and is named in neither planning document.
**PA's tasks do not close this gap.** PA settles the *selection rule* (cheapest-within-tolerance)
but not *what table that rule is applied to*, and the cited spec section (benchmark_spec.md §2.1)
still contains live "do not improvise / M1 is unbuildable" language for exactly that open question,
directly underneath text that claims the same question is resolved. A worker following the letter
of §1's model-definition table (whose only "Mechanism" column entry for M1 is
`held_out_arbiter_advantage`) would very plausibly build the wrong thing.

### 3.2 A second, independently-found contradiction: M3's training scheme (detailed under P3 above)

This is not the defect the review was pointed at, but it's the same failure class (an arm that
looks single-source-of-truth-defined in §1 but is actually specified two incompatible ways across
sections), and it sits directly in the load-bearing path between P3 (toy validation) and P4 (the
real-data battery) — P4 depends on P3, and P3's own positive-control gate is the mechanism meant to
catch exactly this kind of problem, but is set up in a way that will likely fire a false HALT (or
silently reintroduce a training-scheme confound) rather than catch it.

### 3.3 HALT-condition coverage gap

The four HALT conditions (probreg.md:182-187) are: (1) positive control fails, (2) a study is
incoherent, (3) a change to §1's model definitions, (4) anything irreversible/outward-facing. None
of these cleanly covers the situation this review actually found in PA: a task marked
"decisions settled — DISPATCHABLE" whose cited supporting spec explicitly says, in the same
section, "do not improvise this / escalate to the user." That's not a positive-control failure
(nothing has run yet), not "incoherent results" (there are no results yet), and arguably not "a
change to §1's model definitions" (the model definition — "one k, chosen cheaply by a held-out
arbiter" — doesn't change; only the *mechanism* that implements it turns out to be unbuilt). An
autonomous run that reaches PA and does the same reading this review did would have no clean HALT
trigger to invoke, and per the plan's own "never block on a question with a reversible default"
rule, might instead freelance a resolution — directly against benchmark_spec.md's explicit
instruction not to.

### 3.4 Constants table — one dangling consumer

§3.6's "Feed-forward rule (binding)" (lines 207-209) states its fail-loudly requirement only for
**P4**: *"if P4 runs at a value not justified by the artifact named here, its results are not
reportable."* P3 also consumes at least one of these constants (the selection-set fraction, to
build M1 for the toy battery) but is not named by this rule, and — per gap (a) above — doesn't even
list PB as a dependency. So the one binding data-provenance rule in the plan has a real consumer
(P3) that falls outside its stated scope.

### 3.5 Citation integrity

Every file:line citation checked in probreg.md and the sections of benchmark_spec.md it directly
depends on (§1, §2.0/§2.0a/§2.1/§2.3, §14.3) resolved correctly and said what the plan claimed —
no rot found, including several multi-line ranges (e.g. `probabilistic_regression_net.py:84,:96`
for the bypass-head build/None branches; `probabilistic_regression.py:535,:536` for the
REGRESSION_ONLY guard/warning; `n_classes_strategies.py:173` for `NestedStrategy`;
`distilled_router.py:57-60` for the four frozen router constants; `width-cert.md:234,:237` for the
router-invariance result; `_toy_datasets.py:174-178` for the bimodal-target symmetry claim).
D1's "3 passed / 3 failed" claim was checked against the actual test class
(`tests/test_phase3_dynamic_k.py:592-723`, `TestFitRouterSymlogSpaceAlignment`) and is exactly
right: one un-parametrized test plus one test parametrized over 2 seeds = 3 collected cases.
D3/P2's "1.54 baseline" and "2.5 bar" citations match `tests/test_phase4_regression.py:80-88`
exactly. This plan's own warning that citations rot fast was not something I could reproduce —
everything I checked was current as of this review.

### 3.6 P0 / P0b DONE-status re-verification (rubric point 3)

Both re-executed, not just outputs-exist checks:
- P0 verify (i): `grep -n "probreg.md" docs/plans/capacity_programme/MASTER.md` → 4 hits, all
  delegating (lines 22, 47, 50, 186). Confirmed.
- P0 verify (ii): benchmark_spec.md §2.1's model block sets `n_classes_selection_method=
  NClassesSelectionMethod.NESTED` (`docs/probreg_benchmark/benchmark_spec.md:197`), not NONE.
  Confirmed.
- P0 verify (iii): `grep -c "arbiter" docs/probreg_benchmark/benchmark_spec.md` → 11 (not 0).
  Confirmed.
- P0b verify: `grep -niE "crps|pit|winkler|picp|calibration|NLL"` on benchmark_spec.md returns a
  mix of struck-through/"NOT USED — removed" lines and live "NLL" hits — but the live NLL hits are
  the **new primary metric** (fixed-σ mixture NLL), which the corrected premise (documented
  candidly in P0b's own "Outcome differs from this task's original premise" section) explicitly
  keeps. The verify line as literally written predates that premise correction and would technically
  fail a strict re-run (NLL is a live requirement); the plan's own DONE note re-describes what was
  actually checked ("no removed metric or dropped model survives as a live requirement") rather than
  re-running the stale literal command. This is a real but minor process gap — not a "claimed
  without running" case; the substance was checked, but the recorded verify command no longer
  matches what was verified.

---

## Section 4 — Three changes that would most increase the chance this executes correctly unattended

1. **Resolve PA's actual build target before dispatch, in probreg.md itself.** Name
   `all_rung_log_likelihood` (`automl_package/models/selection_strategies/n_classes_strategies.py:230`)
   explicitly as the primitive M1's selector is built on; state plainly that decision #1 ("what
   curve is read") is answered — the full per-rung table, not the arbiter's top-vs-bypass
   statistic — and that decision #3 (bypass competes) is answered yes; then delete or reconcile the
   contradictory "RESOLVED" / "do not improvise, unbuildable" language sitting in the same block of
   benchmark_spec.md §2.1. Without this, PA is the task most likely to produce a plausible-looking
   but structurally wrong M1.

2. **Reconcile M3's training scheme across P3's verify line and benchmark_spec.md §2.0/§2.3 before
   P3 runs.** Either (a) state explicitly that M3's per-k models are ordinary/non-dropout (matching
   §3.2 and current code), and update §2.0/§2.3's "all three train identically" claim to exclude
   M3, or (b) state explicitly that the positive control is expected NOT to exactly reproduce §3.2
   at middle k (with a quantified tolerance) because M3 is now k-dropout-trained, and adjust MASTER
   Decision 14's HALT trigger for this one case accordingly. Whichever is chosen, say it once, in
   probreg.md, so it can't be resolved two different ways by two different workers.

3. **Add `Files (write set):` lines to PB and PC, and add PA + PB to P3's `deps:` field (and extend
   §3.6's fail-loudly constants rule to name P3, not just P4).** This closes the one mechanical gap
   (write-set omission) that's cheap to fix and blocks the "≥2 independent units → orchestrate"
   parallel dispatch of PB ∥ PC from being verifiable at all, and closes the dependency-ordering gap
   that would otherwise let a Workflow-style dispatcher start P3 before its real prerequisites land.
