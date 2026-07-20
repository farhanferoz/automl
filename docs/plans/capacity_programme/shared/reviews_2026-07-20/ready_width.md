# Autonomous-execution readiness review: docs/plans/capacity_programme/width.md

Reviewed read-only. Every citation below was opened; line numbers quoted are what the file
actually contains at review time (2026-07-20).

## Section 1 — per-task verdict table

| Task | Verdict | Biggest gap |
|---|---|---|
| WSEL-0 | DISPATCHABLE | none material |
| WSEL-1 | DISPATCHABLE | none material |
| WSEL-2 | **NOT-DISPATCHABLE** | Duplicates `flexnn-package.md` FP-3 (and `probreg.md` PA) on the same files (`enums.py`, `flexible_width_network.py`); width.md never cites FP-3; FP-3 claims "neither [width.md nor depth-selection.md] defines an API," which is false as of this task |
| WSEL-3 | **NOT-DISPATCHABLE** | The "tolerance rule" it must "import unchanged" does not exist as one rule — two different, incompatible "cheapest-within-tolerance" definitions are live in the codebase, and the task doesn't know it |
| WSEL-4 | DISPATCHABLE-WITH-FIX | Driver script has no name despite the task text claiming one is given; reproduction "tolerance" is unstated |
| WSEL-5 | DISPATCHABLE-WITH-FIX | Collides with `flexnn-package.md` FP-1, which turns `capacity_accounting.py` into a shim and relocates the real module to `automl_package/utils/`; WSEL-5 targets the pre-FP-1 location with no awareness of FP-1 |
| WSEL-6 | DISPATCHABLE-WITH-FIX | No named driver/output path; `parallel: yes` with WSEL-7 is conditional ("if driven by separate scripts"), not guaranteed; §3.6 promises "one file" this task never commits to producing |
| WSEL-7 | DISPATCHABLE-WITH-FIX | Same unnamed-path and same §3.6 "one file" gap as WSEL-6 |
| WSEL-8 | DISPATCHABLE-WITH-FIX | No named driver/output path (otherwise well specified — anchor-warning discipline correctly applied) |
| WSEL-9 | DISPATCHABLE-WITH-FIX | No named driver/output path; depends on WSEL-6/7 producing a canonical constants artifact that, per the WSEL-6/7 gap above, isn't actually specified to exist |
| WSEL-10 | DISPATCHABLE | none material |

Verify-line quality, separately from the above: WSEL-0/1/2 give literal runnable commands.
WSEL-3 through WSEL-9 all give **prose descriptions of what a test/artifact should show**, not a
command line — see Section 3.

---

## Section 2 — per-task detail (everything not plain DISPATCHABLE)

### WSEL-2 — NOT-DISPATCHABLE: duplicates another strand's task on the same files

`width.md:255` (Files, write set): `automl_package/enums.py` ·
`automl_package/models/flexible_width_network.py` · `tests/test_flexible_width_network.py` · call
sites found by grep.

`docs/plans/capacity_programme/flexnn-package.md:242-247` (Task FP-3, "the one selection API ⭐
(both other strands block on this)"), write set: `automl_package/enums.py` ·
`automl_package/models/flexible_width_network.py` · `automl_package/models/flexible_neural_network.py`
· `automl_package/models/independent_weights_flexible_neural_network.py` ·
`automl_package/models/probabilistic_regression.py` · tests · call sites found by grep.

The two tasks are not just overlapping in files — they are the same task, described almost
verbatim:

- width.md:257-260 (WSEL-2): *"`predict` loses `inference_mode` entirely (clean break — the repo
  has no external users; a shim keeps the silent-failure route alive and will eventually be used by
  accident)... Retire the dead `WidthSelectionMethod.DISTILLED` `NotImplementedError` path (WD2)."*
- flexnn-package.md:250-252, 259-260 (FP-3): *"`predict` loses `inference_mode` entirely — clean
  break, no shim (the repo has no external users; a shim keeps the silent-failure route alive and
  will eventually be used by accident)... Retire `WidthSelectionMethod.DISTILLED`'s
  `NotImplementedError` trap and the test that asserts it raises."*

Both also independently require the selection-set fraction to be "CONFIGURABLE, not a baked-in
constant" (width.md:263-264; flexnn-package.md:262-263), and both have near-identical verify lines
(`grep -rn "inference_mode" automl_package/ tests/` returns nothing).

flexnn-package.md itself states the intended relationship, and states it wrong as of this review:
*"Consumers. `width.md` and `depth-selection.md` both depend on this strand for the selection API
and the single router. **Neither of them defines an API** — that is why this file exists rather
than the API being fixed three times in three places, which would predictably produce three
different APIs"* (flexnn-package.md:17-19). width.md *does* define an API (WSEL-2), and never once
mentions `flexnn-package.md` (`grep -n "flexnn-package" docs/plans/capacity_programme/width.md`
returns nothing). File mtimes: `flexnn-package.md` 2026-07-20 15:28, `width.md` 2026-07-20 15:47 —
width.md is the newer file, so this isn't a stale FP-3 written against an old width.md; it's a live,
current contradiction on disk right now.

It is a **three-way** collision, not two. `docs/plans/capacity_programme/probreg.md:330-382` (Task
PA, marked *"✅ decisions settled 2026-07-20 — DISPATCHABLE"*) independently specifies the same
"one enum at construction, predict loses inference_mode, clean break no shim, fraction configurable"
API on `automl_package/enums.py` + `automl_package/models/probabilistic_regression.py` +
`tests/test_phase3_dynamic_k.py`, with its own `deps: P1` (no reference to FP-3 or WSEL-2 either).
WSEL-2 is the only one of the three that is even aware another one exists — *"this is the same API
shape ProbReg's PA adopts... Coordinate the enum's home with PA; if PA has already landed one,
extend it rather than adding a second"* (width.md:266-268) — but it has zero awareness of FP-3,
which is the strand that explicitly claims to be the resolution to exactly this collision (its own
doctrine line: *"fixing it three times independently produces three APIs"*,
flexnn-package.md:264-266 — a hazard that is, right now, live across these three documents). None
of the three enums (`WidthSelectionMethod`-successor, `KSelection`, or FP-3's unnamed
family-spanning enum) has landed yet — confirmed by `grep -n "KSelection\|class.*Selection"
automl_package/enums.py`, which shows only the pre-existing `LayerSelectionMethod`,
`WidthSelectionMethod`, `NClassesSelectionMethod`.

**Verdict stands NOT-DISPATCHABLE as written.** A zero-context worker given only WSEL-2's text would
build a width-only enum on `enums.py`/`flexible_width_network.py` with no way to know that
`flexnn-package.md` FP-3 plans to build a broader, superseding version of the identical thing on the
identical files, or that `probreg.md` PA is simultaneously marked dispatchable to do the same for
ProbReg. Whichever of the three lands last either silently overwrites/reverts the other two's work
or collides at the `write_set_guard` layer — neither outcome is a HALT condition this plan
recognizes (§3.5's HALT list has nothing for "another strand already owns this API").
**Fix:** before WSEL-2 is dispatchable, either (a) `width.md` deletes WSEL-2 and adds a `deps:` on
`flexnn-package.md` FP-3 landing, consuming its enum instead of building one, or (b) if width.md's
author intends WSEL-2 to supersede FP-3's width-relevant scope, that has to be negotiated and
written into *both* files in the same turn, not left implicit.

### WSEL-3 — NOT-DISPATCHABLE: the "tolerance rule" to import doesn't exist as one thing

width.md:55-58 (§1, "Selection rule, fixed for all three"): *"cheapest-within-tolerance, NOT argmax.
w is the smallest width whose held-out score is not meaningfully worse than the best width's, where
'meaningfully' = exceeding twice a bootstrap-estimated standard error (the rule published in
`docs/reports/probreg_kselection/probreg_kselection.md` §3.2)."* WSEL-3's doctrine line repeats this:
*"the tolerance rule is imported unchanged from the published one... Do **not** re-derive it"*
(width.md:285-286).

Opened `docs/reports/probreg_kselection/probreg_kselection.md` §3.2 (lines 277-316, heading
"Checking that this costs nothing important"). It does use a twice-bootstrap-SE threshold — but for
a *different* question: whether training many class-counts jointly ("budget-resampled") costs
accuracy versus a model dedicated to one count, not for *selecting* a count/width off a held-out
curve. Searched the whole document for the actual selection methodology
(`grep -n "elbow\|chosen count\|smallest.*within\|argmax"`) — the report's real selection machinery
is the "elbow" heuristic and the distilled per-input "arbiter," described in §4.1 onward; nowhere in
the document does a "smallest width/count within twice a bootstrap SE" rule get defined or applied
as a *selection* rule. §3.2 is not "the rule published" width.md's §1 claims it is.

Meanwhile the codebase already contains a mechanism literally named for this concept, and it is a
**different rule**: `automl_package/models/common/distilled_router.py:63-81`,
`_cheapest_within_tolerance_labels` — *"Smallest-index capacity with `error <= (1 + tolerance) *
row_min`, per row"* (line 64), `DEFAULT_TOLERANCE = 0.25` (line 57, a fixed 25% fractional margin
above the row minimum). This is copied from `automl_package/examples/sinc_width_experiment.py:414-423`
(same fixed-fraction definition, `DELTA_TIE`). Neither of these two existing implementations uses a
bootstrap standard error at all — they use a flat percentage.

So a worker asked to "import the tolerance rule unchanged from the published one" for WSEL-3 faces a
real, silent fork: the prose in width.md §1 describes a **statistical** rule (2×SE) that has no
corresponding code and is not actually defined at the cited location; the codebase's own
**already-named** "cheapest-within-tolerance" function implements a **fixed-fraction** rule (25%)
that has nothing to do with a bootstrap SE. These select different widths on noisy data. The most
likely failure mode is a worker finding `_cheapest_within_tolerance_labels` (it is the obvious,
already-built, identically-named thing) and wiring it in believing it satisfies "import the
published rule unchanged" — silently building the wrong selection rule for the strand's central
efficiency claim (W-SHARED ≈ W-SWEEP), and W-SHARED's numbers would then not even use the same rule
as `probreg.md`'s own PA/PB tasks reference for the identical two candidate definitions (PA cites
the SAME §3.2 for the SAME "twice a bootstrap SE" claim at `probreg.md:361-364` — so this is not a
width.md-only error, but width.md inherits it uncritically from a source that has the same defect).
**This is exactly the kind of decision §3.5's HALT condition 3 ("any change to §1's model
definitions or the selection rule") should catch — but only if the worker recognizes a decision is
being made at all**, which requires noticing the citation doesn't hold up; nothing in the task text
prompts that check.
**Fix:** resolve which rule is meant — either correct the §1 citation to point at wherever the
2×-bootstrap-SE width/count selection rule is actually specified (if it exists somewhere else), or
change §1 to name the fixed-fraction rule that's actually implemented and give its real tolerance
value, and reconcile the same ambiguity in `probreg.md` PA at the same time since it cites the
identical (non-existent-as-described) source.

### WSEL-4 — DISPATCHABLE-WITH-FIX

width.md:297 (Files, write set): *"a driver under `automl_package/examples/` (name it in the task,
not dated)"* — but the task text, as written, never actually names it; the parenthetical is an
instruction to whoever finalizes the task, not a name. A worker has to invent the filename with zero
guidance beyond "not dated," which is a minor but real write-set-completeness gap (rubric item 1)
and interacts with the two other tasks (WSEL-6, WSEL-7) that write to the same directory with the
same unresolved-name pattern — a real collision surface if two of these run in the same wave.

verify line (width.md:310-311): *"the reproduction of `W_CONVERGED` matches to a stated tolerance"*
— no tolerance value is given anywhere in the task. "Stated" implies the worker states one, which
is itself an unflagged decision (how tight must reproduction be to count as PASS before the branch
table's "does not reproduce → halt WSEL-8" fires?). Given MASTER Decision 14's stakes (a failed
positive-control here **halts** WSEL-8 downstream), the threshold for "reproduces" should not be
left to worker judgment.
**Fix:** name the driver script and its output path explicitly; state the numeric reproduction
tolerance (or the statistical test) that decides pass/fail for the positive control.

### WSEL-5 — DISPATCHABLE-WITH-FIX: collides with flexnn-package.md FP-1

width.md:315-316 (write set): `automl_package/examples/capacity_accounting.py` ·
`docs/plans/capacity_programme/shared/metrics-accounting.md`.

`docs/plans/capacity_programme/flexnn-package.md:206-224` (Task FP-1, "break the circular
dependency") turns `automl_package/examples/capacity_accounting.py` into **a re-export shim** and
moves the real accounting logic into a **new** `automl_package/utils/` module. FP-1's own non-goals
line explicitly anticipates WSEL-5's existence: *"do not add selection-cost accounting here — that
is `width.md`'s task, and it lands after this module has a stable home"* (flexnn-package.md:219-220)
— so FP-1's author knows about width.md's cost-accounting task. The reverse is not true: WSEL-5
never mentions FP-1, `flexnn-package.md`, or `automl_package/utils/`, and its write set targets the
pre-FP-1 file layout unconditionally. If FP-1 has landed by the time WSEL-5 runs (flexnn-package.md
orders FP-1 second, right after FP-0), a worker editing `capacity_accounting.py` per WSEL-5's literal
write set would be editing a shim rather than "the shared module," silently producing dead code or a
second, disconnected accounting path — the opposite of WSEL-5's own doctrine, *"Build it in the
shared module so the other strands inherit it; do NOT build a width-local copy"* (width.md:322-323).
**Fix:** WSEL-5 needs a `deps:` on FP-1 (or an explicit branch: "if FP-1 has landed, write to
`automl_package/utils/<module>`; otherwise write to `capacity_accounting.py` directly") — right now
neither the plan nor the HALT table gives the worker a way to detect which state it's in.

### WSEL-6 / WSEL-7 — DISPATCHABLE-WITH-FIX

Both (width.md:333, 349): *"Files (write set): a study driver under `automl_package/examples/` ·
its results dir"* — no concrete filename or output path for either, same gap class as WSEL-4.

`parallel: yes (disjoint from WSEL-7 if driven by separate scripts)` (width.md:343-344) — the
disjointness is conditional on a choice ("if driven by separate scripts") that the write set never
actually pins down. Nothing forces the two workers to pick different script/output names; if they
converge on similar names (plausible — both are "width selection study" variants), the
`write_set_guard` mechanism referenced in this session's own orchestration doctrine would block the
second writer mid-run, and that stall is not one of §3.5's five HALT conditions.

§3.6's constants table (width.md:195-198) claims: *"selection-set fraction — Set by WSEL-6 — the
WSEL-6 sweep JSON — path fixed by that task, **one file**, named there"* and similarly for WSEL-7's
constants, *"the WSEL-7 sensitivity JSON."* But WSEL-6's own verify line
(width.md:344-345) commits only to *"one JSON per (toy, seed, fraction, arm)"* — i.e., **many**
files, not the single canonical "frozen defaults" artifact §3.6 promises exists and can be read
mechanically at build time. WSEL-7's verify line (width.md:363-364) is the same pattern: *"a
sensitivity table on disk"* with no single named path. Neither task, as written, actually commits to
producing the one small machine-readable file §3.6 and WSEL-9's fail-loudly binding clause depend
on.
**Fix:** name the driver scripts (distinct names); add to each task's spec a requirement to emit one
small canonical "frozen defaults" JSON (fraction / router config / labelling tolerance) at a fixed,
named path, separate from the per-cell sweep results — that is the artifact §3.6 already assumes
exists.

### WSEL-8 / WSEL-9 — DISPATCHABLE-WITH-FIX

Same unnamed-driver gap (width.md:368, 386-387: *"a driver under `automl_package/examples/`... its
results dir"*, no name given). Otherwise both are well specified: WSEL-8 correctly applies the
anchor-warning doctrine (§3.6: *"pair it with W-SWEEP's dedicated per-width models"* rather than
re-deriving from the certification's own numbers), and WSEL-9 correctly applies the prove-it-fails
doctrine to its "fails loudly on a missing constant" binding clause (width.md:399: *"test this
deliberately by hiding one, per the prove-it-fails rule"*). WSEL-9's fail-loudly mechanism is only as
good as the constants actually existing at named paths, which — per the WSEL-6/WSEL-7 finding above
— they currently are not guaranteed to.

---

## Section 3 — plan-level findings

**1. Cross-plan write-set collision is the dominant risk, not anything internal to width.md.**
Two independent, concrete collisions were found against `flexnn-package.md` (WSEL-2 vs FP-3 on
`enums.py`/`flexible_width_network.py`; WSEL-5 vs FP-1 on `capacity_accounting.py`), plus a third
leg against `probreg.md` (PA vs WSEL-2/FP-3, all three "one selection API" tasks live and
uncoordinated). `width.md` never once names `flexnn-package.md` (`grep -c "flexnn-package"
width.md` = 0), despite `flexnn-package.md` explicitly discussing width.md by name and explicitly
warning against exactly this failure mode in its own text (*"fixing it three times independently
produces three APIs"*). The orchestrator's own standing doctrine (*"INDEPENDENCE = DISJOINT WRITE
SETS... the classic false-parallel"*) is stated for **workers within one dispatch**; it is not
being applied **across strand files** here, and nothing in width.md's dispatch contract reads
sibling strand files before compute begins. The task brief for this review said to read
"MASTER.md + this file... that is the whole context" is width.md's own claim (width.md:5) — that
claim is what let this collision go undetected; the whole context is provably not just those two
files.

**2. Verify-line rigor collapses after WSEL-2.** WSEL-0/1/2 give literal shell commands with
observable pass/fail (`grep`, `pytest`, `git diff --stat`). WSEL-3 through WSEL-9 — the entire
compute-heavy, expensive half of the strand — give prose ("a test constructs...", "a sensitivity
table on disk", "the chosen default justified... not by convention"). None of these are commands a
worker or an auditor can run and get a boolean answer from; all require a human or an adjudicator to
read output and judge. Given this review exists because *"'done' meant the task's output files
existed, NOT that the task's `verify:` line had been executed,"* a prose verify line is
structurally the easiest kind of line to rubber-stamp as satisfied without truly executing it — it
never had a crisp executed/not-executed state to begin with. WSEL-1 shows the plan's authors know
how to write a rigorous verify line (revert-refail-restore-checksum); the discipline just isn't
carried through the rest of the file.

**3. Unnamed write-set paths, systemically, on every task from WSEL-4 through WSEL-9.** Every
compute task's "Files (write set)" reads "a driver under `automl_package/examples/`... its results
dir" with no filename. This is not pedantry: §3.6 explicitly requires the constants table's "owning
artifact" to be "the value... read from that artifact at build time," and WSEL-9's binding clause
requires the driver to load named constants and "FAIL LOUDLY if any is missing" — neither is
mechanically checkable while the artifact paths are undecided. It also creates the collision surface
noted for WSEL-6/WSEL-7's `parallel: yes` claim.

**4. Citation integrity is otherwise strong.** Every `file:line` citation checked against the
current file resolved and said what width.md claims it says: `width-cert.md:234/237/308-318/318-328`,
`verdict_variable_width_mse.md` §2.1/§10, `flexible_width_network.py:92/192/204-205/239-298`,
`enums.py:105-109`, `kdropout_converged_width_experiment.py:273-276/387-399(via deploy_bar)/549`,
`sinc_width_experiment.py:486-525`, `moe_flexnn_comparison.py:309-336`,
`distilled_router.py:57-60`, `metrics-accounting.md` §S2, `tests/test_flexible_width_network.py`
(confirmed zero `uncertainty_method`/`predict_uncertainty` references). The one citation that does
not hold up under close reading is §1's tolerance-rule pointer to `probreg_kselection.md` §3.2
(Section 2, WSEL-3 above) — that section exists and uses a 2×SE bootstrap threshold, but for a
different check, not for width/count selection.

**5. The WD3 withdrawal is internally consistent and clean.** Checked the verdict doc's §2.1 numbers
against width.md's restatement — they match (global minima 0.0756/0.0627/0.0710 vs converged
arms ~0.024, flat 10k→120k trajectory, TEST fit ratios reproduce). Searched the rest of the `docs/`
tree for any other place that still cites the withdrawn "NestedWidthNet fails because of
non-convergence" framing (`grep -rl "WSEL\|NestedWidthNet.*fail"` across `docs/`) — the only other
hits (`flexnn-core.md`, the width-MSE `EXECUTION_PLAN.md`, an earlier readthrough doc) reference the
*retained* conclusion (nested genuinely fails) never the withdrawn *reason*, so no other document
needs correcting. One very minor inconsistency inside the verdict doc itself, not introduced by
width.md: its own prose says the global minimum "never falls below 0.063 on any seed" while its own
table shows seed 1 at 0.0627 (< 0.063) — almost certainly a rounding statement, does not change the
WD3 conclusion, not worth a task.

**6. MASTER Decision citations all check out.** Decisions 2, 9, 10, 13, 14, 15 (all cited by
width.md) exist in `MASTER.md`'s Decision register (lines 56-127) and say what width.md claims they
say, including the verbatim-adjacent match on Decision 10 (report authorship / no AI provenance) and
Decision 13 (distillation as the only primary selection mechanism).

**7. WSEL-0's premise is accurate and not yet contradicted by MASTER.md's current state.**
`grep -n "width.md" MASTER.md` currently returns nothing — `width.md` is not in the strand index
(only `width-cert.md`, row 1) and has no naming-key entry. That is exactly the gap WSEL-0 exists to
close; it is correct, not stale, that this check currently fails (it's supposed to, pre-WSEL-0).

**8. Soft ordering inconsistency (low severity).** The global line *"Order: WSEL-0 → WSEL-1 →
WSEL-2 → ... "* (width.md:217) reads as a strict chain, but several adjacent tasks have no write-set
overlap and `deps: none`/independent deps (e.g., WSEL-0 and WSEL-1 could run in the same wave by
`MASTER.md`'s own stated wave rule, "waves = `deps:` + write-set overlap"). Not a correctness bug —
running them strictly serially is a safe superset — but it's an ambiguity for a mechanical
dispatcher: follow the prose "Order:" line, or the per-task `deps:`/write-set annotations, when they
permit different parallelism than the prose implies?

**9. HALT-condition coverage.** Walked all five branch-table rows and five HALT conditions
(width.md:168-185). Two gaps not covered by either a default or a HALT: (a) a `write_set_guard`
collision from two tasks independently naming the same driver/results file (Finding 3 above); (b) a
cross-strand write-set collision discovered mid-run (Finding 1) — nothing in §3.5 tells an
autonomous dispatcher what to do if it discovers, while executing WSEL-2 or WSEL-5, that another
live strand file already claims the same files. Given HALT condition 3 already covers "any change
to §1's model definitions or the selection rule," a cross-strand API collision is arguably close
enough in spirit to route there, but it isn't named, and a zero-context worker executing WSEL-2 in
isolation (per its brief: only "that task's text plus the plan's shared sections") would never open
`flexnn-package.md` to discover the collision in the first place — the HALT condition can't fire on
information the worker was never given cause to look for.

---

## Section 4 — the three changes that would most increase the chance this plan executes correctly unattended

1. **Reconcile the "one selection API" ownership before dispatching WSEL-2 or WSEL-5.** This is the
   single highest-severity gap: three live plan documents (`width.md` WSEL-2, `flexnn-package.md`
   FP-3, `probreg.md` PA) currently claim to build the same enum on overlapping files, and a fourth
   collision (WSEL-5 vs FP-1) sits on top of it. Fix by making width.md's WSEL-2 and WSEL-5 either
   explicitly `deps:` on the relevant `flexnn-package.md` tasks landing, or by stating in both files,
   in the same edit, which one is authoritative and which is retired.

2. **Name every driver script and every artifact path that a downstream task or §3.6 needs to read
   mechanically**, for WSEL-4 through WSEL-9. Right now "a driver under `automl_package/examples/`"
   repeated six times is both a collision risk between parallel tasks and the reason §3.6's
   "single-source-of-truth, read at build time" contract can't actually be implemented as specified
   — no task commits to writing the one small canonical constants file the table promises exists.

3. **Give WSEL-3 through WSEL-9 runnable verify lines, and resolve the tolerance-rule citation.**
   Convert the prose verify lines into literal commands with an observable pass/fail (WSEL-1 is the
   template: exact pytest invocation, plus the revert/re-fail/restore sequence). Separately, and
   before WSEL-3 starts: settle what "the tolerance rule imported unchanged from the published one"
   actually refers to — `probreg_kselection.md` §3.2 does not define a width/count selection rule,
   and the codebase's own `_cheapest_within_tolerance_labels` implements a fixed 25% fractional
   margin, not the 2×-bootstrap-SE rule §1's prose describes. Building W-SHARED against the wrong
   one of these silently invalidates the strand's central W-SHARED ≈ W-SWEEP claim.
