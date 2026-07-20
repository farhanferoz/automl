# Strand: ProbReg — per-input resolution (k) selection

**Owns the whole ProbReg workstream**: model definitions, the defects, the comparison battery
(toys + real data + baselines), and the report. Read `MASTER.md` + this file — that is the whole
context. Nothing about ProbReg is decided anywhere else; if another document disagrees with this
one, **this one wins and the other is a bug to fix**.

**Why this file exists (2026-07-20).** ProbReg content was scattered across 15 files in 5 plan
directories with no owning strand, and the model definitions disagreed in three places at once
(§1). That scatter produced wrong answers to the user twice in one session. Consolidating here is
the fix; the rule that keeps it fixed is the first line of this file.

---

## 1. Model definitions — SETTLED 2026-07-20 (user, live). Supersedes every other statement.

The comparison is between three ways of choosing **k**, the number of classes the model resolves
the target into.

**M1 and M2 train identically — with k-dropout** (per-sample `k ~ Uniform{1..k_max}`,
`NClassesSelectionMethod.NESTED`,
`automl_package/models/selection_strategies/n_classes_strategies.py:173`) — and are read off the
SAME trained network. Training is therefore NOT a variable between them; they differ in exactly one
thing, **how k is chosen**, which is what makes that contrast controlled.

⚠️ **M3 DOES NOT USE k-DROPOUT (user ruling, 2026-07-20). Each of M3's per-k models is trained
ORDINARILY at its own fixed k** (`NClassesSelectionMethod.NONE`), which is what "a model dedicated
to that k" means. A per-k model trained with dropout across the whole ladder is not dedicated to
anything — it is the same network read at a different point, and it cannot serve as an independent
reference. **This is deliberate and it is not a confound**: M3 is not an arm in a controlled
contrast, it is the expensive ceiling the cheap methods are measured against, and training dedicated
models is precisely what makes it expensive. The single-difference rule binds **M1 vs M2**; it does
not bind the reference.

*(Corrected 2026-07-20 after review. The earlier text said all three trained identically with
k-dropout. That made M3 not-a-reference, and it also broke P3's positive control: the published
coherence check compares against "a separately trained **ordinary** model"
(`docs/reports/probreg_kselection/probreg_kselection.md` §3.2), and the current per-k sweep code
already builds them that way — `_probreg_fixed` sets `NClassesSelectionMethod.NONE`
(`automl_package/examples/report_a_benchmark.py:185-191`). The plan was wrong; the code was right.
**`docs/probreg_benchmark/benchmark_spec.md` §2.0 and §2.3 still carry the superseded wording and
must be corrected — see P0-b1 below.**)*

🔑 **EACH MODEL IS THE COMPLETE SYSTEM, INCLUDING ITS SELECTION MACHINERY** (user, 2026-07-20).
**M1 = ProbReg + arbiter. M2 = ProbReg + distillation. M3 = ProbReg + sweep selector.** Every one is
scored end-to-end and costed end-to-end: the selection step is *inside* the model, never a
side-analysis reported next to it. A table row for M1 is the arbiter's answer, not the network's
answer with the arbiter mentioned in a footnote. This is binding on the driver, the metrics and the
report.

| Model | = ProbReg + | How k is chosen | Cost | Mechanism |
|---|---|---|---|---|
| **M1** | **the arbiter** | ONE k for the dataset | cheap | `all_rung_log_likelihood` (`automl_package/models/selection_strategies/n_classes_strategies.py:230`) — see ⚠️ below, NOT `held_out_arbiter_advantage` |
| **M2** | **the distillation** | a k **per input** | cheap | `fit_router` + routed predict (`automl_package/models/probabilistic_regression.py:754`) |
| **M3** | **the sweep selector** | ONE k for the dataset, by training a **separate ORDINARY model per k** (no k-dropout, `NClassesSelectionMethod.NONE`) and scoring each on held-out data | **expensive — the reference** | generalise `select_k_for_toy` (`automl_package/examples/report_a_benchmark.py:331`), which already builds fixed-k models via `_probreg_fixed` (`:185-191`) |

⚠️ **M1's mechanism corrected 2026-07-20 (repair pass).** `held_out_arbiter_advantage`
(`automl_package/models/probabilistic_regression.py:845-916`) was named here originally and is
**the wrong shape for M1's job** — it returns a per-input `(N,)` array comparing ONLY the top rung
against the k=1 bypass (`:905-907`), and cannot express "chose k=4" for any middle rung, let alone
one global answer for the whole dataset. The primitive that CAN — a full `(batch, n_classes)`
per-rung held-out likelihood table, exactly what a cheapest-within-tolerance selector needs to run
over — is `all_rung_log_likelihood`
(`automl_package/models/selection_strategies/n_classes_strategies.py:230-252`), which as of this
repair has **zero callers anywhere in the repo**
(`grep -rn "all_rung_log_likelihood" automl_package/ tests/` matches only its own definition). PA
builds M1's selector on top of it, not on `held_out_arbiter_advantage`. Recorded as **D5/D6** in §3.

**What M3 is for.** M3 is the honest, expensive way to pick k. The efficiency claim of this whole
strand is that **M1 reaches M3's answer at a fraction of the cost**. M3 is therefore not optional
and not a baseline — it is the thing M1 is measured against. M1 and M3 differ only in *how the
same global k is found*; M2 is the separate question of whether k should be global at all.

**Both halves of the M1≈M3 claim must be tested, and they are different claims:**
- **(a) same quality** — read at a given k, does the k-dropout model match a model dedicated to
  that k? *(Partially established on toys — §2.)*
- **(b) same choice** — does the arbiter pick the k that the sweep would pick?
  *(NOT tested anywhere. §2.)*

**M2 runs on a DIFFERENT tolerance rule, and that is legitimate, not an oversight.** The global arms
(M1, M3) select k at **twice a bootstrap-estimated standard error** — the smallest k whose held-out
score is not meaningfully worse than the best (PA's selection rule, reusing the tolerance published
in `docs/reports/probreg_kselection/probreg_kselection.md` §3.2). M2's distilled router does not
read a curve — it labels each row independently at a flat relative margin,
`DEFAULT_TOLERANCE = 0.25` (`automl_package/models/common/distilled_router.py:57`), applied as
`error <= (1 + tolerance) * row_min` (`:80`). The two rules legitimately differ because the two
selection problems differ: a per-input labelling decision has one row's worth of evidence, and no
standard error is estimable from a single observation, whereas a global chooser reads a whole
held-out curve, over which a bootstrap standard error is exactly the right notion of noise.
**Consequence, stated so the report does not paper over it: M2's chosen k values are NOT directly
comparable to M1's or M3's on tolerance grounds** — the three models share a cost objective, not a
shared statistical selection rule; comparison lives on held-out error and cost.

### 1.1 Contradictions this replaces — all three were live simultaneously

Recorded so the same error cannot be reintroduced by reading an older document:

1. **`MASTER.md`'s naming key** defined variable-k as *"dynamic-k (ELBO + SoftGating)"* —
   **in-training** selection, which `MASTER.md` Decision 13 itself demotes to a labelled
   comparison arm. Self-contradictory. **CORRECTED 2026-07-20**: that entry now points here and
   states no definition of its own. *(No line number is cited on purpose — a line reference into a
   file this one is actively rewriting is a cache entry that rots within the hour. It did: the
   first draft of this bullet cited `:38-39`, which pointed at an unrelated entry ten minutes
   later.)*
2. **`docs/probreg_benchmark/benchmark_spec.md:66`** defines Model 1 with
   `n_classes_selection_method=NONE` — **k-dropout OFF** — and k chosen by hyperparameter tuning.
   Against §1 this arm differs from Model 2 in **two** ways at once (training scheme AND k
   choice), so no result could be attributed to either. The spec's own §3.4 claims the two arms
   *"differ in exactly one thing"*; that claim is **false as written**. **→ fix in task P0.**
3. That spec has **no M3 at all** (`grep -c "arbiter"` on it returns 0), so the efficiency claim
   had no reference to be measured against.

**Confound doctrine (MASTER Decision 15 generalised):** an arm that differs from its comparator in
more than one respect is NOT dispatchable. State the single difference in the task, or do not run
it. This is the same failure that invalidated the depth pilot; it was about to repeat here.

---

## 2. State — what is established, and what is not

**Established (toys only).** The k-dropout coherence check — one k-dropout model read at each
fixed k, against a separately trained model dedicated to that k (i.e. an M1-vs-M3 *quality* check)
— has been run on 3 toy problems × 3 repeats. Write-up and numbers:
`docs/reports/probreg_kselection/probreg_kselection.md` §3.2. Summary of what it shows: the
largest checked k **never** fails the coherence check in any of the 9 cases, but some middle k
fails in **8 of the 9** — a small, real, and currently **unexplained** cost concentrated in the
middle of the range.

**NOT established — and each is a task below:**
- **(b) same choice.** Nothing anywhere compares the arbiter's chosen k against the sweep's best
  k. §3.2 compares *fit quality at each k*; that is a different claim. → **P3**
- **Anything on real data.** All of the above is synthetic. → **P4**
- **Baselines.** No comparison against XGBoost / LightGBM / CatBoost / a standard NN has been
  run for this strand. → **P4**
- **Why the middle-k cost exists.** 8 of 9 is a pattern, not noise, and it is undiagnosed. → **P5**

**Report status.** `docs/reports/probreg_kselection/probreg_kselection.md` exists (880 lines, PDF
built) and covers the toy work. It predates §1 and therefore uses the older framing; it must not be
extended until §1 is reflected in the code and the battery. → **P6**

---

## 2.5 Where the history lives, and which records can be trusted

Surveyed 2026-07-20 (13 files, all read in full, artifact paths and a sample of line citations
checked against disk). Two disconnected generations of ProbReg planning existed, which never
referenced each other — that disconnection is what this strand ends.

**⚠ GROUND-TRUTH RULE, learned from the survey: for the first generation, only
`automl_package/examples/capacity_ladder_results/RESULTS.md` records what actually completed.** No
inline status glyph in those plan files is trustworthy — several tasks carry adjudicated GO verdicts
in the results ledger while their own plan file still shows them open, and one file's status table
contradicts its own prose from the same day. All four are now banner-frozen in place (not moved —
moving them would have broken ~30 citations including ten frozen preregistration records).

**Still-live reasoning in the frozen files — read, never dispatch:**
- `docs/plans/width_dial_2026-07-11/cascade_execution_plan_2026-07-11.md` §3.1 — the cross-k
  class-identity conflict (one component asked to be correct at several resolutions at once) **and
  its resolution**: `optimization_strategy=REGRESSION_ONLY` makes the cross-entropy branch never
  fire. Now baked into the source (`automl_package/models/probabilistic_regression.py:53-60`,
  `:531-543`). **This is why §1 freezes `REGRESSION_ONLY` for all three models** — it is not a
  stylistic choice, it is the thing that keeps the k ladder coherent.
- The same file's §4.7 — four ProbReg library fixes, **all verified landed at HEAD 2026-07-20**,
  none of which had a completion record in any plan document until that check.
- `docs/plans/capacity_programme/shared/hetero_nll_diagnosis.md` — the diagnosed, accepted
  hetero-NLL failure (§3 D3).

**Known citation rot (do not build on these):** three 2026-07-11 documents cite
`automl_package/models/probabilistic_regression.py:796-798` for a classifier-probability bug. That
file has grown ~200 lines since; the function moved to roughly `:1010` and the bug is already fixed
there. Re-verify any line citation from that generation before using it.

## 3. Known defects

**D1 — FIXED + VERIFIED 2026-07-20.** `fit_router` scored raw-space `y_val` against
`forward_at_k`'s symlog-space outputs — silently wrong error table ⇒ silently wrong selector, worst
on exactly the heavy-tailed data this strand targets. Fix at
`automl_package/models/probabilistic_regression.py:813`; contract documented in the docstring.
Verified in BOTH directions: fix present → 3 passed; fix deleted → 3 failed
(`tests/test_phase3_dynamic_k.py`, class `TestFitRouterSymlogSpaceAlignment`). Detail + the
measured blindness table: `docs/plans/capacity_programme/flexnn-core.md` (F9-fix-b block).

**D2 — OPEN, a units bug in `held_out_arbiter_advantage` (NOT M1's selection mechanism — see D5).**
The identical units mismatch in `_per_sample_log_likelihood_at_k`
(`automl_package/models/probabilistic_regression.py:835`), which passes caller-space `y` straight
to `forward_at_k`'s outputs with no transform. Its only caller is `held_out_arbiter_advantage`
(`automl_package/models/probabilistic_regression.py:845`). → **P1**.
*(Corrected 2026-07-20, repair pass: this entry originally called `held_out_arbiter_advantage`
"M1's k-selection mechanism" — D5 below shows that claim was wrong. D2 still stands as a real bug
in that function and P1 still fixes it (the neighbourhood-advantage readout is used elsewhere and
its correctness matters on its own terms), but P1's fix does NOT by itself unblock M1. **The
primitive M1 actually needs, `all_rung_log_likelihood`
(`automl_package/models/selection_strategies/n_classes_strategies.py:230`), has the SAME shape of
bug** — it passes `y_target` straight through with no symlog transform (`:251`), and has zero
callers today, so nobody has hit it yet. Whoever builds PA's M1 selector on top of it must apply
`fit_router`'s now-fixed transform pattern there too, or D2 reopens in the function that actually
matters. Not filed as a separate task here — it is PA's problem to close as part of building the
mechanism, flagged so it is not missed.)*

**D4 — OPEN, inherited, user-gated.** The cross-k class-identity conflict is only *avoided*, not
fixed: it is dodged by freezing `REGRESSION_ONLY` (§2.5). For any configuration that DOES activate
cross-entropy, the defect is live, and what shipped is a runtime warning only — the guard is at
`automl_package/models/probabilistic_regression.py:535` and the `logger.warning` it fires at `:536`
— not a behaviour change. The two candidate
fixes on record are k-stable binning, or documenting that k-dropout requires likelihood-only
training. **Not in scope for this strand's battery** (which is `REGRESSION_ONLY` throughout) but it
must not be re-discovered a fourth time — it is recorded here as the owner.

**D3 — two failing tests, both pre-existing, neither caused by current work.**
- `test_probabilistic_nll_beats_constant_on_heteroscedastic` — root-caused and deliberately
  accepted: `docs/plans/capacity_programme/shared/hetero_nll_diagnosis.md`.
- `test_prob_regression_heteroscedastic_mse` — error 2.8812 against a 2.5 bar. Verified identical
  to 16 digits on a clean worktree at commit `600460c`, so it is not new. **Undocumented anywhere**
  — the same silent-failure pattern this programme keeps paying for. → **P2**

**D5 — OPEN, the primary defect this repair pass exists to record.** `held_out_arbiter_advantage`
(`automl_package/models/probabilistic_regression.py:845-916`) is **not M1's selection mechanism**,
despite being named as such in an earlier version of §1's table. Verified by reading the full
function body: it returns a per-input `(N,)` array (`:914-916`), computed as the
neighbour-averaged advantage of ONE fixed top rung over the k=1 bypass (`:905-907`) — a binary
top-vs-bypass comparison, smoothed over an x-neighbourhood. It cannot express "chose k=4" for any
middle rung, and it answers a per-input question, not "one k for the dataset" (§1's M1
definition). → **PA must build M1's selector on `all_rung_log_likelihood` instead** (§1 ⚠️, PA's
task text).

**D6 — OPEN, a stale docstring; recorded here, not fixed (no source edits in this repair).**
`all_rung_log_likelihood`'s docstring
(`automl_package/models/selection_strategies/n_classes_strategies.py:234-235`) claims *"This is
the all-rung score table `ProbabilisticRegressionModel.held_out_arbiter_advantage` ...
consume[s]"* — **false**. `held_out_arbiter_advantage` never calls `all_rung_log_likelihood`; it
calls `_per_sample_log_likelihood_at_k` twice, once per rung
(`probabilistic_regression.py:905-906`). Confirmed by reading both functions in full. Whichever
task next touches `n_classes_strategies.py` (PA, most likely) should correct the docstring in the
same change that gives `all_rung_log_likelihood` its first real caller.

---

## 3.5 Autonomous execution contract

**This strand is scoped to run unattended (user, 2026-07-20).** The root is dispatcher + verifier.
Every foreseeable branch below has a **pre-authorised default**; take it, log it, keep going. Only
the four HALT conditions stop the run.

**Rule: never block on a question that has a reversible default.** Log every default taken to
`RESUME.md` `### Decisions` with the evidence that triggered it, and batch anything genuinely
user-only for the end of the run.

| Branch | Pre-authorised default | Log |
|---|---|---|
| **PB**: which selection fraction becomes the frozen default | the **smallest** fraction at which every arm is within its own noise band of its best (same twice-standard-error rule as everywhere else); if no fraction saturates, take the largest swept and record the study as **inconclusive, floor not found** | fraction + the curve |
| **PB**: M2 still improving at the largest fraction | freeze the largest swept, and mark M2's battery result **"router data-limited"** in the report — a loss then does NOT support "per-input k does not pay" | the mark, prominently |
| **PC**: router conclusions invariant to architecture | keep the current frozen default; record invariance as a finding | table |
| **PC**: NOT invariant | adopt the **smallest** configuration that reaches the plateau, freeze it globally, re-run PB at the new router | old → new + why |
| **P2**: the undocumented failing test | if it is a stale bar, write the acceptance note; if it is a real regression, fix it. **Never loosen the bar to make it pass.** | either way, a note on disk |
| **ceiling binds** (selected k = k_max) | re-run that cell at `k_max = 20`; report the raise | which cells |
| **Kepler feature leakage** (`benchmark_spec.md` §14.3, was PARK-1) | drop the two leaky features, and pre-register that this dataset tests conditional mean/variance, **not** a density-level multimodality claim | the pre-registration, before the run |
| a spec section contradicts §1 | **§1 wins**; fix the spec in the same turn | the correction |

**HALT and ask — these four only:**
1. A **positive control fails** (MASTER Decision 14) — the protocol is then the defect, not the arms.
2. A study comes back **incoherent rather than merely negative** (e.g. PB's curve is non-monotone
   beyond noise) — that is a broken instrument, and running the battery on it wastes the budget.
3. Any change to **§1's model definitions**. They are the user's, not the run's.
4. Anything **irreversible or outward-facing** (deleting artifacts, publishing, committing).

## 3.6 Constants the studies FREEZE, and the battery READS

The studies are not background colour — they set the parameters the comparison runs at.

🚫 **DO NOT WRITE THE VALUES INTO THIS TABLE.** The plan holds the *name of the constant* and the
*path of the artifact that owns it*; the value is read from that artifact at build time. A number
copied here is a cache entry that rots, and this programme has already paid for that twice today (a
line reference that broke within ten minutes, and a diagnosis contradicted by its own sibling
document). The driver reads these from disk; a reviewer resolves them by opening the artifact.

| Constant | Set by | Owning artifact (single source of truth) |
|---|---|---|
| selection-set fraction | **PB** | `automl_package/examples/capacity_ladder_results/PB/frozen.json` |
| M2 data-limited flag, per dataset | **PB** | same file, one boolean per (toy, arm) |
| router hidden / depth / epochs / lr | **PC** | `automl_package/examples/capacity_ladder_results/PC/frozen.json`, else the current frozen defaults at `automl_package/models/common/distilled_router.py:57-60` if that file's `invariant` field is `true` |
| labelling tolerance | **PC** | same file |
| `k_max` after any ceiling raise | **P3/P4** | the per-cell result JSON that recorded the bind |

**Feed-forward rule (binding):** if P3 or P4 runs at a value not justified by the artifact named
here, its results are **not reportable**. The point of doing the studies first is that every number
in the final table has a measured reason for the setting it was produced under.

⚠️ **Anchor warning for every `verify:` line below.** Where a task reproduces a frozen number as its
positive control, that anchor must come from something **not computed by the method under test** — a
second implementation, an invariant, or a published figure. Re-deriving an anchor with the same code
does not verify the worker; it conscripts the worker into confirming our own bug and returns it
stamped *verified*. Concretely: P3's coherence half must NOT anchor solely on
`docs/reports/probreg_kselection/probreg_kselection.md` §3.2 re-run through the same harness — pair
it with the dedicated per-k models of M3, which are trained independently and are exactly the
non-shared implementation this rule asks for.

**Compilation note.** Once PA–PC are closed and every task below is decision-complete, this strand
is compilable to a deterministic Workflow (fan out on the sweeps, serialise the writes through the
root) rather than relying on dispatcher discipline — which is the shape the unattended run should
take.

## 4. Tasks

Order is **P0 → P0b → P0-b1 → P1 → P2 → PA → PB ∥ PC → P3 → P4 → P5 → P6**. (PT is parked and gates nothing — the existing toys are retained.) P0–P2 plus PA are the "fix it properly
and completely" phase the user gated everything else behind (2026-07-20); no comparison compute runs
until they close. **PB and PC must precede P3/P4**: both fix a *parameter of the method* (how much
selection data, what router), and running the battery before they are settled would produce results
nobody could attribute — which is the failure this whole strand was created to stop.

### P0 — make the definitions single-sourced

**Files (write set):** `docs/probreg_benchmark/benchmark_spec.md` **only**
**🚫 NOT in the write set: `docs/plans/capacity_programme/MASTER.md`** — ROOT-ONLY (MASTER naming
key, "SHARED FILES ARE ROOT-ONLY"). Three sibling tasks (FP-0, WSEL-0, DSEL-0) need MASTER edits in
this same wave; concurrent writers produce contradictory text in one file. **Deliverable instead:**
emit the exact MASTER naming-key text verbatim in this task's report, for the root to apply.
**Spec:** Rewrite the `MASTER.md` naming key to point at §1 of this file instead of restating a
definition. Rewrite the benchmark spec's model section to §1's three models: all `NESTED`, M1 =
arbiter-selected global k, M2 = distilled per-input k, M3 = per-k sweep reference. Delete the
"differ in exactly one thing" claim about the old pair and re-state it truthfully for the new
arms. Record the three superseded definitions as corrections, not silent edits.
**Non-goals:** no code; no change to metrics, datasets, or the tuning protocol beyond what §1
forces.
**Status: ✅ DONE 2026-07-20, all three verify conditions executed and shown passing.**
`MASTER.md`: naming key delegates here and states no definition; strand index carries this file;
Corrections entry records the three-way contradiction and its organisational root cause.
`docs/probreg_benchmark/benchmark_spec.md`: §2 rewritten to the three-model set with a new §2.0
stating what each contrast isolates and what the old pair got wrong; M1 rebuilt on `NESTED`; M3
added; baselines renumbered 4–7; the false "differ in exactly one thing" cell corrected in place.

**Three consequences found while editing, each verified, each already folded into the spec:**
- **C3 is now RESOLVED, structurally.** The old arm could not select the bypass rung because that
  head is built only on the dynamic branch
  (`automl_package/models/architectures/probabilistic_regression_net.py:84`, `None` on the
  non-dynamic `else` at `:96`). All three arms now use `NESTED`, so all three reach rungs `1..10`.
  Standing condition added: M1's rung set must stay `1..10` or the defect returns.
- **C1 dissolved into a narrower, sharper condition.** No ProbReg arm tunes `n_classes` any more, so
  the Optuna space is identical across the three; what remains is the split each reads its k from.
  Binding: **all three select k on `cal` only** — an M1 reading `val ∪ cal` against an M2 reading
  `cal` would be rigged.
- **M3 must be EXEMPT from the matched wall-clock budget** (per-k-fit matched instead, total cost
  reported as a headline number). Matching it would starve the reference the efficiency claim is
  measured against.
*Orchestration:* parallel: no (single writer, doc set overlaps everything) · deps: none ·
tier: main loop (definitional) · scale: static · shape: design ·
verify: (i) `grep -n "probreg.md" docs/plans/capacity_programme/MASTER.md` shows the naming key
delegating here — **DONE**; (ii) the M1 code block in `docs/probreg_benchmark/benchmark_spec.md`
sets `NESTED`, not `NONE`; (iii) that spec defines three models, and `grep -c "arbiter"` on it is
no longer 0. *(Do NOT verify by grepping for the retired phrase — the Corrections entry quotes it
deliberately, so such a grep always fires. This line originally did exactly that and was
unrunnable.)*

### P0b — ✅ DONE 2026-07-20. σ-scope sweep completed at requirement level.

**Outcome differs from this task's original premise, and the difference is the point.** P0b was
written assuming σ removal meant squared-error scoring. The user corrected that: σ is **not fitted**
but scoring stays distributional with **one shared fixed σ** (`docs/probreg_benchmark/benchmark_spec.md`
§2, §4.1). The sweep was then executed on the corrected premise.

**Landed in the spec:** §2 scope block rewritten (fixed-σ mixture likelihood, the σ-value rule, the
required ×0.5/×2 ranking-stability check, and the recorded reasoning error); §2.0a naming map added
so remaining older prose cannot be misread; §4 rewritten to fixed-σ mixture NLL as primary with RMSE
retained as a point-accuracy column and a note on why RMSE can never be the k readout; §4.3 selection
score is the same likelihood at fixed σ, so the published tolerance rule transfers **unchanged**;
§6.5 search spaces collapsed to one identical ProbReg space plus the two surviving baselines, with
M3's single-search rule; §7 outcomes restated on the new arm set, thresholds carried over unchanged
because the metric family and units are unchanged; **new outcome H6** (does the cheap read match the
expensive sweep) marked NEW rather than folded into an old one; §1.1/§1.2/§8.2 build requirements for
removed metrics and dropped models struck through rather than deleted.

**Verified:** `grep` for live requirements naming a removed metric or a dropped model returns only
struck-through or explicitly-superseded text. Plan gates 6/6.

**Remaining (cosmetic, non-blocking):** ~41 occurrences of the old "shared-k"/"variable-k" wording
survive in prose. §2.0a maps them binding-ly, so the document is safe to build from; a prose rename
is a mechanical follow-up. **Do not attempt it with a blind find-and-replace** — §2.0a names the one
class of passage where the old term is superseded rather than renamed.

*Original task text follows, retained because its premise being wrong is itself the case law.*

### ~~P0b — finish the σ-scope-change sweep through the spec~~ (original, premise superseded)

**Why.** The user removed variance fitting from scope (variance is separate work). §2 of
`docs/probreg_benchmark/benchmark_spec.md` is updated and the consequences are stated there, but the
change reaches further into that document than one section, and a half-swept spec is worse than an
unswept one — the driver would implement whichever half it read first.

**Known stale regions, all in `docs/probreg_benchmark/benchmark_spec.md`:**
- **§4 (metrics) — wholly stale.** NLL, CRPS, PIT, calibration/ECE, PICP, Winkler all assume σ.
  Replace with a point-prediction set (primary: squared error; per-dataset RMSE; the existing
  matched-split and seed machinery is unaffected).
- **§6.5 search spaces** still list XGBoost and CatBoost rows; §1.2 item 5 still requires CatBoost
  probabilistic wiring. Both models are dropped.
- **§7's bars** are written on distributional metrics (H2/H3/H5 reference calibration-type claims)
  and must be restated on the point-prediction set — or dropped if they no longer mean anything
  without σ. **Do not silently re-point a bar at a different metric; a bar that changes metric is a
  new bar and is labelled as one.**
- **§5.4's `symlog` rows** stay — the transform is about target scale, not variance — but any text
  justifying it by uncertainty conversion needs re-reading.

**Doctrine:** the σ removal is a SCOPE decision, not a metric downgrade to be smuggled. Every bar
that changes must be visibly restated, and the report must carry the caveat already recorded in §2:
squared error scores the mean only, so an MSE-only benchmark **under-detects** what k buys.
**Non-goals:** do not re-add any distributional metric "just for reference" — that reopens the scope.
*Orchestration:* parallel: no (same file as P0) · deps: P0 · tier: sonnet · scale: static ·
shape: execution · verify: `grep -niE "crps|pit|winkler|picp|calibration|NLL" docs/probreg_benchmark/benchmark_spec.md`
returns only historical/superseded notes, no live requirement; and no dropped model name survives as
a live spec row.

### P0-b1 — propagate the M3 training ruling into the benchmark spec

**Files (write set):** `docs/probreg_benchmark/benchmark_spec.md`
**Why.** §1 was corrected 2026-07-20 (user ruling): M3's per-k models are trained ORDINARILY at
fixed k, not with k-dropout. The spec still carries the superseded wording as a live requirement in
at least four places, and a driver built from the spec would train M3 the wrong way — destroying the
reference the whole efficiency claim is measured against.
**Spec:** Correct §2.0's "All three ProbReg models train identically — with k-dropout" to the §1
formulation (M1 and M2 identical; M3 ordinary per k). Correct §2.3's M3 build text. Correct the
naming-map row that says the training difference "is gone". Where a code block sets `NESTED` with a
comment claiming it is the "SAME as M2/M3", fix the comment. Record each as a dated correction, not
a silent edit — the superseded claim was in force and other documents may cite it.
**Doctrine:** §1 wins over any spec (this file's opening rule). The spec is the thing a driver is
built from, so a spec that disagrees with §1 is not a documentation nit — it is the defect that
reaches the results.
**Non-goals:** no other spec change; no code.
*Orchestration:* parallel: no (same file as P0) · deps: none · tier: sonnet high · scale: static ·
shape: execution ·
verify: `grep -n "train identically" docs/probreg_benchmark/benchmark_spec.md` returns only
struck-through or explicitly-superseded text; `grep -n "SAME as M2/M3" docs/probreg_benchmark/benchmark_spec.md`
returns nothing; and the M3 code block in §2.3 sets `NClassesSelectionMethod.NONE`.

### PA — the k-selection API (✅ decisions settled 2026-07-20 — DISPATCHABLE)

**Why this is a task and not an implementation detail.** The three models must be three *options on
one model*, usable standalone and sharing every component. Today they are three different shapes,
and one is a trap. Verified 2026-07-20:
- **M2** takes three separate steps — a constructor option, a separate `fit_router()` call, then
  `inference_mode="routed"` passed to BOTH `predict` and `predict_uncertainty`
  (`automl_package/models/probabilistic_regression.py:598`, `:694`). **Forget that flag and you
  silently get the un-routed model** — the default is `"soft"`, there is no error, and nothing
  records that a router was fitted but unused. Same silent-wrong-answer class as D1 and D2.
- **`inference_mode` is a raw string**, validated by a hand-rolled membership check
  (`automl_package/models/probabilistic_regression.py:613`, `:708`), with no type anywhere. This
  breaks the repo's own closed-set rule (`CLAUDE.md` → enums, never magic strings).
- **M1** has no mechanism in code yet — the *rule* is now specified (`benchmark_spec.md` §2.1:
  cheapest-within-tolerance at twice a bootstrap standard error, bypass competing), so this is a
  build, not a design question. It is PA's main piece of work. **Build it on
  `all_rung_log_likelihood`** (`automl_package/models/selection_strategies/n_classes_strategies.py:230`)
  — the per-rung `(batch, n_classes)` held-out likelihood table — never on
  `held_out_arbiter_advantage`, which is a different, per-input readout that cannot answer a
  global "which k" question (§1 ⚠️, D5).
- **M3** exists only as a loop inside an example script
  (`automl_package/examples/report_a_benchmark.py:331`), not as a model.

**Shape (settled by this repair pass, 2026-07-20 — supersedes the earlier "propose a `KSelection`
enum" text, which collided with `flexnn-package.md` FP-3's `CapacitySelection` and is retired):**
PA **consumes** `CapacitySelection` (`automl_package/enums.py`, owned by `flexnn-package.md` FP-3)
— it does not define an enum of its own. FP-3 ships exactly two members, `FIXED` and `PER_INPUT`;
**`GLOBAL_CHEAP` and `GLOBAL_SWEEP` are added by the task that builds their mechanism** — PA adds
`GLOBAL_CHEAP` in the same change that builds M1's selector, and `GLOBAL_SWEEP` in the same change
that builds M3's, never as a member whose mechanism raises `NotImplementedError` (the
`WidthSelectionMethod.DISTILLED` trap `flexnn-package.md` FP-3.a retires; do not recreate it here
under a new name). The member is passed at construction. `fit()` performs whatever held-out
selection the chosen mode needs. **`predict` loses its `inference_mode` argument entirely** — the
model already knows how it selects k, and requiring the caller to remember is what creates the
silent failure. All three then drop into any comparison harness unchanged.

**✅ Both blocking decisions SETTLED by the user 2026-07-20:**
1. **Selection rule = cheapest-within-tolerance, NOT argmax.** k is the **smallest** rung whose
   held-out score is not meaningfully worse than the best rung's. Rationale on record: held-out
   curves are noisy, so argmax systematically overshoots (any upward wiggle at a high rung wins by
   chance) and would report more resolution than the data supports; and the question this strand
   asks is "how much resolution is needed", which is a smallest-sufficient question. **Tolerance
   rule: reuse the one already published in
   `docs/reports/probreg_kselection/probreg_kselection.md` §3.2** — a difference counts as real only
   if it exceeds twice a bootstrap-estimated standard error — so results stay comparable with the
   existing report instead of needing a caveat. **The SAME rule is applied to M3's sweep curve**, or
   M1 and M3 are not answering the same question.
2. **Clean break on `inference_mode` — delete it, do not deprecate it.** The repo has no external
   users (user, 2026-07-20), so a compatibility shim is pure cost: it keeps the silent-failure route
   alive, doubles the test surface, and will eventually be used by accident.

**Also required (user, 2026-07-20): the selection-set fraction must be CONFIGURABLE**, not a baked-in
constant — it is a parameter of the method, and PB is about to measure it.

**Files (write set):** `automl_package/models/probabilistic_regression.py` ·
`automl_package/models/selection_strategies/n_classes_strategies.py` ·
`tests/test_phase3_dynamic_k.py` · call sites found by grep, not by memory
**Non-goals:** no new selection *algorithms* — this task changes how the existing three are reached,
not what they do. No change to `DistilledCapacityRouter` internals. No change to `CapacitySelection`
itself, its name, or its cross-family contract — file a finding against `flexnn-package.md` FP-3
instead of patching it from here. Do not touch `automl_package/enums.py`.
*Orchestration:* parallel: no (touches the same file as P1) · deps: `flexnn-package.md` FP-3,
`flexnn-package.md` FP-9, P1 · tier: sonnet high · scale: static · shape: design → execution ·
verify: `grep -rn "inference_mode" automl_package/ tests/` returns nothing (clean break, no shim);
`grep -n "class KSelection" automl_package/enums.py` returns nothing (no home-grown enum);
`grep -n "CapacitySelection" automl_package/models/probabilistic_regression.py` is non-zero; a test
asserts a `CapacitySelection.PER_INPUT`-constructed, router-fitted model routes with no caller flag;
`AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m pytest tests/test_phase3_dynamic_k.py -q` exits 0.

### PB — how much data does the selection step actually need? (NEW, user 2026-07-20)

**Why.** The 15% selection fraction was chosen because it looked reasonable, **not because anything
was measured**. Searched 2026-07-20: no study of selection-set size exists anywhere in this repo for
ProbReg. The entire efficiency claim runs through a mechanism whose data requirement is unknown —
and the three arms are not equally exposed. M2 must learn a *function* from x to rung, so it should
be the hungriest; M1 and M3 need only rank rungs on average. **If M2 loses at 15%, "per-input k does
not pay" and "the router was starved" are indistinguishable** — the same class of unattributable
result that has already cost this programme twice.

**Files (write set):** `automl_package/examples/probreg_pb.py` (Create) ·
`automl_package/examples/capacity_ladder_results/PB/`
**Spec:** sweep the selection fraction (suggest `{5, 10, 15, 25, 40}%` of the training portion) on
the toys where ground truth is known, for all three arms, holding everything else fixed. Report each
arm's quality against fraction, and the fraction at which each saturates. The headline output is a
defensible default and, for M2, an explicit statement of the data floor below which its result is
not readable.
**Emit the frozen-constants artifact §3.6 promises:**
`automl_package/examples/capacity_ladder_results/PB/frozen.json`, containing exactly the two
constants this task owns per §3.6 — the selection-set fraction (the pre-authorised default per §3.5
if no fraction saturates) and the M2 data-limited flag, one boolean per (toy, arm).
**Non-goals:** no real data (that is P4's budget); no architecture changes (PC).
*Orchestration:* parallel: yes (disjoint from PC's write set — separate driver scripts and results
directories, see Files above) · deps: P1, PA · tier: sonnet high · scale: dynamic (a sweep) ·
shape: research · verify:
`test -f automl_package/examples/capacity_ladder_results/PB/frozen.json` and
`python -c "import json; d=json.load(open('automl_package/examples/capacity_ladder_results/PB/frozen.json')); assert 'fraction' in d and 'data_limited' in d, d"`
exits 0; `ls automl_package/examples/capacity_ladder_results/PB/*.json` shows one file per
(toy, seed, fraction, arm); a saturation plot file exists under the same results dir.

### PC — is the distilled router's architecture right for ProbReg? (NEW, user 2026-07-20)

**What exists.** The router is fixed at two hidden layers of 32 units, ReLU, 300 full-batch Adam
epochs, lr 1e-2, cross-entropy on cheapest-within-tolerance labels (tolerance 0.25) —
`automl_package/models/common/distilled_router.py:57-60`. Those constants were **copied from an
earlier width experiment**, not chosen for ProbReg. The only sensitivity evidence anywhere is from
the WIDTH strand: `docs/plans/capacity_programme/width-cert.md:234` ran half and double router hidden
on 3 seeds and found the deploy claims invariant (`:237`). That is reassuring but it is (a) a
different strand, (b) one dimension, (c) a does-it-break check, not a search.

**Files (write set):** `automl_package/examples/probreg_pc.py` (Create) ·
`automl_package/examples/capacity_ladder_results/PC/`
**Spec:** on the toys, vary router width/depth (at least half/double/4× hidden and 1 vs 3 layers),
epochs, and the labelling tolerance. Establish whether ProbReg's routing conclusions are invariant
the way width's were, and if not, what the router actually needs.
**Emit the frozen-constants artifact §3.6 promises:**
`automl_package/examples/capacity_ladder_results/PC/frozen.json`, containing exactly the two
constants this task owns per §3.6 — router hidden/depth/epochs/lr, and the labelling tolerance. If
this task finds invariance, the file records the current frozen defaults
(`automl_package/models/common/distilled_router.py:57-60`) rather than inventing new ones.
**Doctrine:** the router stays FROZEN and untuned inside the benchmark (§2.2) so the M1/M2 contrast
measures selection rather than search effort. **This task does not unfreeze it** — it establishes
whether the frozen choice is defensible, and any change lands as a new frozen default *before* P4
runs, never per-dataset.
**Non-goals:** no per-dataset tuning of the router, ever. No change to the labelling rule's meaning.
*Orchestration:* parallel: yes (disjoint from PB's write set — separate driver scripts and results
directories, see Files above) · deps: P1, PA · tier: sonnet high · scale: dynamic · shape: research ·
verify: `test -f automl_package/examples/capacity_ladder_results/PC/frozen.json` and
`python -c "import json; d=json.load(open('automl_package/examples/capacity_ladder_results/PC/frozen.json')); assert {'hidden','depth','epochs','lr','tolerance'} <= d.keys(), d"`
exits 0; `python -c "import json; d=json.load(open('automl_package/examples/capacity_ladder_results/PC/frozen.json')); assert 'invariant' in d, d"`
exits 0 (the invariant-or-not verdict is a field, not prose); if `d['invariant']` is `False`, the
same file's `new_default` key is non-null and cited by PB's re-run.

### P1 — fix D2 (the arbiter units mismatch), same shape as D1

**Files (write set):** `automl_package/models/probabilistic_regression.py` ·
`tests/test_phase3_dynamic_k.py`
**Spec:** Transform `y` to the fitted space inside `_per_sample_log_likelihood_at_k` when
`target_transform="symlog"`, exactly as `fit_router` now does; document the contract on
`held_out_arbiter_advantage` the way `fit_router`'s docstring does. Add a regression test that the
arbiter's advantage is identical whether given raw or pre-transformed `y`.
**Doctrine:** a regression test is not evidence until it has been shown to FAIL on the unfixed
code. Assert on the quantity the fix changes (the per-sample likelihood), never on a coarse
downstream view — that is exactly how D1's first tests came out blind.
**Non-goals:** no change to the arbiter's neighbourhood averaging or its certified width default;
no touching `fit_router` (already fixed).
*Orchestration:* parallel: no (same file as D1's fix) · deps: none · tier: sonnet high ·
scale: static · shape: execution ·
verify: `AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m pytest tests/test_phase3_dynamic_k.py -q`
green, THEN delete the transform, re-run, and show the new test FAILING, then restore and show the
file checksum unchanged.

### P2 — diagnose or accept the undocumented failing test (D3, second one)

**Files (write set):** `docs/plans/capacity_programme/shared/` (a new diagnosis note) ·
possibly `automl_package/models/probabilistic_regression.py`
**Spec:** Establish whether error 2.8812 against the 2.5 bar is a real regression or a stale bar.
The bar's own comment cites a 1.54 baseline; find when it last held (`git log -S` on the test) and
what changed. Outcome is either a fix, or a written, cited acceptance in the same shape as
`docs/plans/capacity_programme/shared/hetero_nll_diagnosis.md`. **"Known failure" with no document
is not an outcome.**
**Non-goals:** do not loosen the bar to make it pass. Do not touch the other failing test.
*Orchestration:* parallel: **no — runs AFTER P1** (both may touch `automl_package/models/probabilistic_regression.py`; the old "parallel: yes if it lands doc-only" made the safety of a concurrent dispatch depend on an outcome the dispatcher cannot know in advance) · deps: **P1** · tier: sonnet
high · scale: static · shape: research ·
verify: `test -f docs/plans/capacity_programme/shared/<note>.md` exits 0, and
`grep -qE "^[0-9a-f]{7,40}" docs/plans/capacity_programme/shared/<note>.md` shows the commit at
which the bar last held (from `git log -S` on the test). Then exactly one of two outcomes, both
mechanically checkable: **(a) stale bar** —
`AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m pytest tests/test_phase4_regression.py::TestPerformanceBaselines::test_prob_regression_heteroscedastic_mse -q`
passes, and the note says so; **(b) real regression** — same prove-it-fails shape as P1's template:
the fix is applied and the suite is green, THEN the fix is deleted, the same test is re-run and
shown FAILING, then the fix is restored and the file checksum is shown unchanged. "Known failure,
no fix, no acceptance note" is not a valid outcome under either branch.

### PT — ⏸ PARKED. Toys are RETAINED; the redesign answers a different question.

**Resolution 2026-07-20 (user).** σ is not fitted, but scoring stays **distributional** with one
shared fixed σ (`docs/probreg_benchmark/benchmark_spec.md` §2, §4.1). The three existing toys are
homoscedastic by construction — exactly what a fixed σ wants — so **they are retained unchanged and
the multimodality question they were built for survives intact.**

**What actually happened, recorded because it is the instructive part.** The user's decision was
"do not fit σ". The orchestrator inferred "therefore score on squared error", and from that
correctly derived that the toys were degenerate: their two modes are equal-weight and symmetric
about zero (`automl_package/examples/_toy_datasets.py:174-178`), so the conditional mean is
identically zero and squared error has **zero power** to detect k. The derivation was sound; **the
premise was an unrequested inference.** The user challenged it, and the third option — keep the
mixture likelihood, fix σ to a shared constant — removes the in-sample-overfitting pathology
without removing the readout. **Lesson: when a scope decision seems to invalidate an established
artifact, re-examine the inference chain before discarding the artifact.** The cost here was one
spec; had it gone the other way it would have been the whole toy suite and the question with it.

**The spec is not wasted:** `docs/probreg_benchmark/toy_design_spec.md` designs toys where
resolution helps the *conditional mean*, with an analytic floor and a falsifiable
noise-versus-resolution prediction. That is a genuinely interesting and *different* question —
"classification as a regulariser for the mean" rather than "how many components does this input
need". **Parked for a future strand**, not deleted, and explicitly not a dependency of anything
below.
*Orchestration:* PARKED — no deps, no tier, not dispatchable, blocks nothing.

### P3 — the M1 ≈ M3 claim, both halves, on toys

**Files (write set):** `automl_package/examples/probreg_p3.py` (Create) ·
`automl_package/examples/capacity_ladder_results/P3/`
**Spec:** On the existing 3 toy problems: train M3 (one dedicated model per k over the frozen k
grid), score each on held-out data, record the sweep's best k. **Emit the positive control first**
— `automl_package/examples/capacity_ladder_results/P3/m3_control.json`, recording whether M3's
per-k scores reproduce §3.2's finding pattern — and check it before training or reading any M1
number (MASTER Decision 14). Train M1 (k-dropout) and record the arbiter's chosen k. Report
**(a)** quality at matched k — this re-runs §3.2's check inside the new harness so both halves come
from one artifact — and **(b)** agreement between the two chosen k's, which is the untested half.
Both across the same seeds as §3.2 so the numbers are comparable.
**Doctrine:** MASTER Decision 14 — run the known-good arm first, alone. Here that is M3 at the k
§3.2 already characterised; it must reproduce before M1 is scored. **MASTER Decisions 9 (trajectory
discipline) and 16 (exonerate optimisation before blaming architecture) both bind**: every trained
model (M1's k-dropout net and M3's per-k dedicated nets) reports its full held-out trajectory, its
convergence flag is trajectory-verified with `hit_cap=False`, and any arm that looks like it lost is
run through the escalation ladder (LR sweep → clipping → warmup → init scheme → normalization)
before being recorded as an architecture finding rather than an optimization one.
**Non-goals:** no real data (P4); no baselines (P4); do not re-tune the arbiter's neighbourhood
width.
*Orchestration:* parallel: no · deps: P1, PA, PB (the arbiter must be correct, M1's mechanism must
exist, and the selection-set fraction must be frozen before M1's choices mean anything) ·
tier: sonnet high · scale: static · shape: execution · verify:
`test -f automl_package/examples/capacity_ladder_results/P3/m3_control.json` and
`python -c "import json; d=json.load(open('automl_package/examples/capacity_ladder_results/P3/m3_control.json')); assert d['reproduces'] is True, d"`
exits 0 **before** any M1 number is read (Decision 14); then
`ls automl_package/examples/capacity_ladder_results/P3/*.json` shows one file per (problem, seed),
each containing `m1_chosen_k`, `m3_chosen_k`, `per_k_held_out_scores`, `held_out_trajectory`, and
`hit_cap: false`.

### P4 — real data + baselines

**Files (write set):** `automl_package/examples/probreg_benchmark.py` (Create) ·
`automl_package/examples/capacity_ladder_results/P4/`
**Spec:** The battery in `docs/probreg_benchmark/benchmark_spec.md` **as corrected by P0** — M1,
M2, M3 plus the live baseline set (`MASTER.md` Decision 3 correction, 2026-07-20): **one tree model
(LightGBM), a plain single-output NN (the key control), and linear regression (the floor)**.
XGBoost and CatBoost were dropped the same day and must not appear in the driver. On the datasets
that spec freezes. The Kepler question (§14.3) is resolved by the pre-authorised default in §3.5,
pre-registered before the run.

**Binding: the driver READS §3.6's constants from their artifacts at startup and FAILS LOUDLY if any
is missing.** No default may be silently substituted — a battery that runs at an unjustified
selection fraction or router shape produces exactly the unattributable result this strand exists to
prevent. The startup read is the mechanism; a comment saying "remember to use PB's fraction" is not.
Each results JSON records the constants it ran under, so any table row can be traced to the study
that justified its settings.
**Trajectory discipline (MASTER Decision 9) and optimization-first (MASTER Decision 16) both bind**
on every model trained here, same shape as P3: full held-out trajectories, trajectory-verified
convergence flags, `hit_cap=False`, and the escalation ladder (LR sweep → clipping → warmup → init
scheme → normalization) run before any arm is called an architecture loss.
*Orchestration:* parallel: no · deps: P0, **P0-b1**, P1, P2, PA, **PB, PC**, P3 · tier: sonnet high ·
scale: dynamic · shape: execution · verify: with one §3.6 constant artifact deliberately
hidden/renamed, the driver exits non-zero (the prove-it-fails rule) — restore the artifact and
re-run; then `ls automl_package/examples/capacity_ladder_results/P4/*.csv automl_package/examples/capacity_ladder_results/P4/*.json`
shows the per-dataset CSV and per-model JSON outputs, each JSON containing a `constants` key naming
the study artifact it traces to, plus `held_out_trajectory` and `hit_cap: false`.

### P5 — explain the middle-k coherence cost

**Files (write set):** `docs/plans/capacity_programme/shared/p5_middle_k_coherence.md` ·
`automl_package/examples/capacity_ladder_results/P5/`
**Spec:** 8 of 9 cases show a middle-k coherence failure (§2). Establish the mechanism. Candidate
to falsify first: middle k's get the least effective gradient share under uniform k-dropout while
the largest k is reinforced by every prefix. If true, the fix is a non-uniform k schedule and that
is a design decision for the user, not an improvisation.
**Trajectory discipline (MASTER Decision 9) and optimization-first (MASTER Decision 16) both
bind**: before any middle-k pattern is attributed to architecture, show the escalation ladder (LR
sweep → clipping → warmup → init scheme → normalization) was exhausted on the middle-k arms, with
full held-out trajectories on disk, not endpoints.
**Non-goals:** do not change the training schedule without a user ruling.
*Orchestration:* parallel: yes · deps: P3 · tier: opus xhigh (discovery-shaped) · scale: static ·
shape: research · verify:
`test -f docs/plans/capacity_programme/shared/p5_middle_k_coherence.md` exits 0, and
`grep -oE "automl_package/examples/capacity_ladder_results/P5/[A-Za-z0-9_./-]+\.json" docs/plans/capacity_programme/shared/p5_middle_k_coherence.md`
returns at least one path, and each returned path resolves with `test -f`.

### P6 — report

**Files (write set):** `docs/reports/probreg_kselection/`
**Spec:** Extend the existing report to the three-model framing and the real-data results, via the
`research-report` skill, authored as the user, no AI/tool provenance (MASTER Decision 10). The
existing text predates §1 and its framing must be **corrected, not appended to**.

**The studies are REPORT CONTENT, not internal detail (user, 2026-07-20)** — they get their own
sections ahead of the comparison, because they are what license its settings:
- **How much data the selection step needs (PB).** The curve, the saturation point, and the chosen
  fraction. Where an arm was still improving at the largest fraction, say so — and say plainly that
  its comparison result is then a floor, not a verdict.
- **Whether the router's architecture matters (PC).** The sensitivity table and an invariance verdict.
- **Whether the cheap global read reaches the expensive sweep's answer (P3).** Both halves stated
  separately — same quality at matched k, and same chosen k — because they are different claims and
  only one of them was ever previously tested.
- **The middle-k coherence cost (P5)**, including that it was unexplained until P5 and what P5 found.

**Honesty clauses, binding:** report M3's full cost next to its accuracy (the efficiency claim is a
ratio and this is its denominator); state every constant the battery ran under and which study set
it; and carry the negative results — the data floor, any non-invariance, the middle-k cost — in the
body, not an appendix.
*Orchestration:* parallel: no · deps: P4, P5 · tier: opus xhigh (report + verdict) ·
scale: static · shape: execution · verify: the `research-report` skill's own cold-read gate
(procedural, run per the skill); then `grep -c "P[0-9A-Z]*" docs/reports/probreg_kselection/*.md`
is nonzero (studies cited by task ID, not restated from memory); then for each §3.6 constant name
(selection-set fraction, M2 data-limited flag, router hidden/depth/epochs/lr, labelling tolerance,
`k_max`), `grep -q "<constant name>" docs/reports/probreg_kselection/*.md` exits 0.

---

## 5. Non-goals for this strand

No new selection strategies. No change to `RegressionStrategy` or the loss family. No revival of
in-training k selection as a primary (MASTER Decision 13) — it may appear only as a labelled
comparison arm. No variance-programme work (MASTER Decision 2).
