# Cross-plan review — probreg.md · width.md · depth-selection.md · flexnn-package.md

Scope: the four strand files under `/home/ff235/dev/MLResearch/automl/docs/plans/capacity_programme/`,
checked against each other and against `MASTER.md`. Also read where a claim required it:
`flexnn-core.md`, `width-cert.md`, `docs/probreg_benchmark/benchmark_spec.md`, and the source files
the plans cite. Read-only; nothing in the repo was edited.

**Headline.** The four plans were written to end duplicate ownership, and between themselves they
mostly succeed on *prose*. They fail on three axes: (1) **`flexnn-core.md` was never actually
emptied** — it still holds live tasks that own the same files, the same reports and the same
benchmark as three of the four new plans, including one carrying the retired ProbReg definition;
(2) **write sets collide on nine files with no ordering**, because three plans each declare a
"selection API" task over the same modules; (3) **`MASTER.md` Decision 3 as amended forbids exactly
what `width.md` and `depth-selection.md` are built to do.**

Verification note: `~/dev/.venv/bin/python -m pytest docs/plans/capacity_programme/test_plan_gates.py -q`
→ **6 passed**. The mechanical gates do not catch any of what follows; every finding below is
invisible to them.

---

## X1 — Duplicate ownership

### X1-1 ⛔ `flexnn-core.md` still owns the ProbReg real-data benchmark that `probreg.md` P4 owns — same driver file, same spec file, and it uses the RETIRED definition

`probreg.md:1-6`:
> `# Strand: ProbReg — per-input resolution (k) selection`
> `**Owns the whole ProbReg workstream**: model definitions, the defects, the comparison battery
> (toys + real data + baselines), and the report. ... Nothing about ProbReg is decided anywhere
> else; if another document disagrees with this one, **this one wins and the other is a bug to fix**.`

`probreg.md:507-509` (P4 write set + spec):
> `**Files (write set):** `automl_package/examples/probreg_benchmark.py` (Create) · its results dir`
> `**Spec:** The battery in `docs/probreg_benchmark/benchmark_spec.md` **as corrected by P0** — M1,
> M2, M3 plus XGBoost, LightGBM, CatBoost and a standard NN`

`flexnn-core.md:569`, `:576-580`, `:589-593` (F12, still live, no superseded marker):
> `### Task F12: ProbReg benchmark — shared-k vs variable-k vs baselines, on REAL data (added 2026-07-20, user-ratified)`
> `- Create: `docs/probreg_benchmark/benchmark_spec.md` (F12-a, spec first)`
> `- Create: `automl_package/examples/probreg_benchmark.py` (driver)`
> `**Models (6).** (1) **shared-k** ProbReg — fixed `n_classes`; (2) **variable-k** ProbReg —
> per-input k via `DistilledCapacityRouter` (F9). Treated as two DISTINCT models per the MASTER
> Naming key.`

Two live tasks create the same two files. Worse, F12's model set **is** the definition
`probreg.md:51-57` records as retired ("shared-k / variable-k as two distinct models per the MASTER
Naming key") — and `MASTER.md:46-50` no longer contains that naming-key definition to be "per". F12
is a live task pointing at a deleted authority.

**Resolution:** `probreg.md` owns it. `flexnn-core.md` F12 must be struck through with a pointer to
P4, in the same edit that P0's "record the three superseded definitions as corrections" covers.
P0 is marked ✅ DONE and did not touch `flexnn-core.md` — so P0's DONE status is itself incomplete.

### X1-2 ⛔ `flexnn-core.md` F10 owns `docs/reports/probreg_kselection/`, which is `probreg.md` P6's write set

`probreg.md:537`:
> `**Files (write set):** `docs/reports/probreg_kselection/``

`flexnn-core.md:481`, `:486`:
> `### Task F10: ProbReg k-selection report — the missing report (added 2026-07-18)`
> `- Create: `docs/reports/probreg_kselection/` (own folder; md + PDF per [[reference_pdf_build]])`

The directory already exists on disk with `probreg_kselection.{md,pdf}`, so F10 has partly run.
`probreg.md:90-92` states that artifact "predates §1 and therefore uses the older framing; it must
not be extended until §1 is reflected". F10's own structure section still describes the old framing
("R2 arbiter-not-knee readout", "K6 distilled per-input router 9/9"), with no §1 correction clause.

**Resolution:** `probreg.md` P6 owns it; F10 → struck through, pointing at P6.

### X1-3 ⛔ The unified report (`flexnn-core.md` F7) still claims the width and depth report content that WSEL-10 and DSEL-12 now own

`flexnn-core.md:328`, `:346-350`:
> `### Task F7: unified research report — FlexNN: flexible-capacity networks (+ MoE comparison)`
> `- Create: `docs/reports/flexnn_unified/``
> `Structure (execution-level TOC): ... (2) Width: the shared-readout break + certified
> per-width-heads result (G-WIDTH, verdict §10); (3) Depth: why smooth toys can't show depth
> (negative note), the group-composition result (G-DEPTH D5/D8b), the feedforward 2×2 pilot outcome
> (F5) — whichever way it lands;`

`width.md:102`:
> `- **Any width report.** Width's reportable content was folded into a unified report that has never
>   run. → **WSEL-10**`

`width.md:405` / `depth-selection.md:440`:
> `**Files (write set):** `docs/reports/width_selection/``
> `**Files (write set):** `docs/reports/depth_selection/``

Neither WSEL-10 nor DSEL-12 mentions F7 or `flexnn_unified`, and F7 was not edited to give the
content up. F7 also depends on `F5b` (`flexnn-core.md:341`: `deps: F5b, F6, M0`), and `F5b` is the
block `depth-selection.md:16-18` says it is *removing* from that file.

**Resolution:** decide once, in `MASTER.md`: either F7 becomes a synthesis that *cites* the three
per-dial reports, or WSEL-10/DSEL-12 are the reports and F7 is retired. Currently both are live.

### X1-4 ⛔ THREE plans each own "the selection API" over the same modules — and `flexnn-package.md` states as fact that the other two do not

`flexnn-package.md:17-20`:
> `**Consumers.** `width.md` and `depth-selection.md` both depend on this strand for the selection API
> and the single router. **Neither of them defines an API** — that is why this file exists rather than
> the API being fixed three times in three places, which would predictably produce three different APIs.`

That sentence is **false for `width.md` and omits ProbReg entirely.**

`width.md:254-256`:
> `### WSEL-2 — the width selection API`
> `**Files (write set):** `automl_package/enums.py` · `automl_package/models/flexible_width_network.py` ·
> `tests/test_flexible_width_network.py` · call sites found by grep, not by memory`

`probreg.md:330`, `:374-375`:
> `### PA — the k-selection API (✅ decisions settled 2026-07-20 — DISPATCHABLE)`
> `**Files (write set):** `automl_package/enums.py` · `automl_package/models/probabilistic_regression.py` ·
> `tests/test_phase3_dynamic_k.py` · call sites found by grep, not by memory`

`flexnn-package.md:244-247` (FP-3 write set) covers **all of the above plus more**:
> `**Files (write set):** `automl_package/enums.py` · `automl_package/models/flexible_width_network.py` ·
> `automl_package/models/flexible_neural_network.py` ·
> `automl_package/models/independent_weights_flexible_neural_network.py` ·
> `automl_package/models/probabilistic_regression.py` · tests · call sites found by grep`

Grep evidence that the coordination does not exist: `grep -n "flexnn-package" width.md probreg.md`
returns **nothing**. `width.md` coordinates with the wrong plan —

`width.md:267-268`:
> `Coordinate the enum's home with PA; if PA has already landed one, extend it rather than adding a second.`

**Resolution:** `flexnn-package.md` FP-3 owns the enum and the API. PA and WSEL-2 must be reduced to
"consume FP-3" (the shape `depth-selection.md` DSEL-4 already has, `:326-334`) and must declare
`deps: flexnn-package.md FP-3`. This is the defect the brief already knew about, and it is a
three-way, not a two-way.

### X1-5 ⛔ Three plans each authorise changing ONE global router default

The router is genuinely shared — verified in source, ProbReg calls it too:
`automl_package/models/probabilistic_regression.py:831`: `router = DistilledCapacityRouter(device=self.device)`;
`automl_package/models/flexible_width_network.py:295`: `router = DistilledCapacityRouter(device=self.device)`.
Defaults verified at `automl_package/models/common/distilled_router.py:57-60`:
`DEFAULT_TOLERANCE = 0.25`, `DEFAULT_HIDDEN = (32, 32)`, `DEFAULT_N_EPOCHS = 300`, `DEFAULT_LR = 1e-2`.

`probreg.md:420-421` (PC):
> `any change lands as a new frozen default *before* P4 runs, never per-dataset.`

`width.md:359-360` (WSEL-7):
> `any change lands as a new frozen default *before* WSEL-9 runs, never per-dataset.`

`depth-selection.md:401-402` (DSEL-9):
> `any change lands as a new frozen default *before* DSEL-11 runs, globally, never per-dataset.`

Three independent studies, each pre-authorised (in its §3.5 branch table) to overwrite the same four
constants globally, with no ordering and no cross-strand notification. Whichever runs last wins, and
the two earlier batteries silently ran at a superseded "frozen default" — the exact unattributable
result all three files were written to prevent. `flexnn-package.md` owns
`automl_package/models/common/distilled_router.py` (FP-5 write set, `:292`) but has no rule covering
this.

**Resolution:** `flexnn-package.md` owns the defaults. The three studies MEASURE and report; only the
package strand WRITES a new default, once, after all three sensitivity tables exist. Add that to
`flexnn-package.md` §3.5 and downgrade the three branch-table rows to "report, do not write".

### X1-6 `MASTER.md` is in four write sets with no ordering

`probreg.md:235` (P0, DONE) · `width.md:225` (WSEL-0) · `depth-selection.md:180` (DSEL-0) ·
`flexnn-package.md:194` (FP-0) all list `docs/plans/capacity_programme/MASTER.md`. All three
outstanding ones declare `deps: none`. See X5.

### X1-7 `flexnn-core.md` is in three write sets across two plans, with no ordering

`depth-selection.md:181` (DSEL-0): `docs/plans/capacity_programme/flexnn-core.md` — "Move the
F5/F5b/F5c block out". `depth-selection.md:230` (DSEL-1): same file, "(halt marker only)".
`flexnn-package.md:195` (FP-0): same file — "**Move the package-refactor workstream out of
`flexnn-core.md`**". `deps: none` on both DSEL-0 and FP-0. See X5.

---

## X2 — Duplicated work

### X2-1 ⛔ WASTEFUL — the cheap global chooser is built three times, and the package strand explicitly refuses to own it

`width.md:277-283` (WSEL-3):
> `**Spec:** Implement the cheap global read: score every width on the held-out selection set, apply
> **cheapest-within-tolerance at twice a bootstrap standard error** (§1), return ONE width for the
> dataset, and store it so `predict` uses it with no caller flag.`

`depth-selection.md:354-356` (DSEL-6):
> `**Spec:** Implement the cheap global read: score every depth on the held-out selection set, apply
> **cheapest-within-tolerance at twice a bootstrap standard error** (§1b), return ONE depth for the
> dataset, and store it so `predict` uses it with no caller flag.`

`probreg.md:344-345` (PA):
> `**M1** has no mechanism in code yet — the *rule* is now specified (`benchmark_spec.md` §2.1:
> cheapest-within-tolerance at twice a bootstrap standard error, bypass competing), so this is a
> build, not a design question. It is PA's main piece of work.`

Word-for-word the same statistical rule, three times, in three modules. And the one strand that
could hold it declines:

`flexnn-package.md:267-268` (FP-3 non-goals):
> `building the cheap-global and sweep mechanisms is the consuming strands' work — this task defines
> how they are reached.`

This is aggravated by `flexnn-package.md`'s own inventory, which already flags this class of
duplication as a *known open problem* — `flexnn-package.md:100-102` (FP-7):
> `A **sixth duplicated concept is suspected but uninventoried**: bootstrap / standard-error
> statistics helpers appearing across several scripts under similar names. → **FP-7**`

So FP-7 will inventory duplicated bootstrap helpers while WSEL-3, DSEL-6 and PA add three more.

**Verdict: wasteful.** One `select_cheapest_within_tolerance(error_curve, ...)` in
`automl_package/models/common/` (next to the router that already implements the *hard-label* version
of the same idea) serves all three. Only the per-family plumbing is per-strand.

### X2-2 ⛔ WASTEFUL (implementation) / LEGITIMATE (measurement) — three selection-fraction studies with an identical grid

`probreg.md:394` (PB) · `width.md:335-336` (WSEL-6) · `depth-selection.md:382-383` (DSEL-8) all read,
near-verbatim:
> `sweep the selection fraction (suggest `{5, 10, 15, 25, 40}%` of the training portion) ... holding
> everything else fixed`

and all three declare `**Files (write set):** a study driver under `automl_package/examples/` · its
results dir`.

**Verdict: the measurement is legitimately per-dial** (different arms, different data, different
saturation points — this is the brief's "legitimate opposite" and it holds). **The three separate
drivers are not.** One parameterised study driver taking (family, arm set, ladder) produces three
result sets. Recommend: one driver, owned by `flexnn-package.md`; three result dirs, one per strand.

### X2-3 ⛔ WASTEFUL (implementation) — three router-sensitivity studies, identical spec

`probreg.md:416-417` (PC) · `width.md:353-355` (WSEL-7) · `depth-selection.md:397-398` (DSEL-9), all:
> `vary router width/depth (at least half/double/4× hidden, 1 vs 3 layers), epochs, and the labelling
> tolerance`

Same object under test (one shared `DistilledCapacityRouter` — see X1-5), same grid, three drivers.
**Verdict: wasteful.** One sensitivity harness over the shared router, run per family. See X1-5 for
the more serious problem (three writers of one default).

### X2-4 LEGITIMATE — the three expensive sweeps

`probreg.md:34` (M3, per-k models) · `width.md:299-304` (WSEL-4, per-width models) ·
`depth-selection.md:366-368` (DSEL-7, per-depth models). Each trains a genuinely different object.
No sharing is possible beyond the selection rule of X2-1. **Keep all three.**

### X2-5 LEGITIMATE (question) / WASTEFUL (three benchmark specs) — real data + baselines

`probreg.md:508` (P4) points at `docs/probreg_benchmark/benchmark_spec.md`;
`width.md:387` (WSEL-9): `a benchmark spec under `docs/width_benchmark/``;
`depth-selection.md:424` (DSEL-11): `a benchmark spec under `docs/depth_benchmark/``.

The ProbReg spec's baseline set is already frozen and verified on disk as **LightGBM, plain NN,
linear regression** (`docs/probreg_benchmark/benchmark_spec.md:300`, `:317`, `:336`; `:682`
*"(XGBoost and CatBoost rows removed — both models are dropped, §2.)"*) — which is **exactly**
WSEL-9's and DSEL-11's baseline set (`width.md:389-392`, `depth-selection.md:425-427`). The datasets,
split protocol, matched-budget rule and statistical test in that spec are family-agnostic.

**Verdict: three separate specs is waste and a drift generator.** Recommend one shared
protocol/dataset/baseline section, three thin per-family overlays. See also X3-1 — probreg.md's own
P4 already contradicts its own frozen spec on this exact point.

### X2-6 LEGITIMATE — "does the cheap read pick what the sweep picks"

`probreg.md:44-45` (P3b) · `width.md:52-54` (WSEL-8b) · `depth-selection.md:81` (DSEL-10b). One
question, three dials, three answers required. **Keep all three.**

---

## X3 — Contradictions (highest value)

### X3-1 ⛔⛔ `MASTER.md` Decision 3 as amended FORBIDS what `width.md` and `depth-selection.md` are built to do

`MASTER.md:65-70`:
> `3. **Reports are toys-only.** UCI/real-data belongs to the later Paper A/B roadmap
>    (`docs/research_plan.md`), not these reports.
>    *(**AMENDED 2026-07-20, user live.** Binds the WIDTH/DEPTH capacity reports only. The ProbReg
>    k-selection work is now explicitly IN scope for real data + baselines: ...)*`

The amendment **narrows the toys-only rule onto width and depth**. Against that:

`width.md:384-389` (WSEL-9):
> `### WSEL-9 — real data + baselines`
> `**Spec:** The three models of §1 against the baseline set — **one tree model (LightGBM), a plain
> single-output NN ..., and linear regression ...** — on real datasets frozen in the spec.`

`depth-selection.md:421-427` (DSEL-11):
> `### DSEL-11 — real data + baselines`
> `**Spec:** The three models of §1b, on both mechanisms, against **one tree model (LightGBM), a plain
> single-output NN (the key control), and linear regression (the floor)**, on real datasets frozen in
> the spec.`

`flexnn-core.md:571-572` states the amendment's scope the same way I read it:
> `**Authority:** MASTER Decision 3 as amended 2026-07-20 (user, live). This is the scope change
> from "reports are toys-only" — it binds THIS task and F10's results §, not the width/depth reports.`

Neither `width.md` nor `depth-selection.md` mentions Decision 3, an amendment to it, or any user
ruling extending real data to their strands (`grep "Decision 3"` on both → no hits). **Two whole
tasks plus two report sections are, on the current written record, out of scope.**
`width.md` §5 and `depth-selection.md` §5 non-goals do not exempt them either.

**Resolution — user ruling required, and it is the single biggest item here.** Either amend Decision 3
again in `MASTER.md` to admit width/depth real data (and say so in both strands), or delete WSEL-9 and
DSEL-11 and cut their report sections. Do NOT let both readings stay live; that is the exact failure
mode `probreg.md:8-11` was written after.

### X3-2 ⛔ `probreg.md` P4 requires four baselines; the spec it points at dropped two of them

`probreg.md:508-509` (P4):
> `**Spec:** The battery in `docs/probreg_benchmark/benchmark_spec.md` **as corrected by P0** — M1,
> M2, M3 plus XGBoost, LightGBM, CatBoost and a standard NN, on the datasets that spec freezes.`

`docs/probreg_benchmark/benchmark_spec.md:122-124`:
> `> - **CatBoost is DROPPED** — its only reason for inclusion was its probabilistic output mode ...
> > - **XGBoost is DROPPED** in favour of one tree learner (below).`

`docs/probreg_benchmark/benchmark_spec.md:682`:
> `*(XGBoost and CatBoost rows removed — both models are dropped, §2.)*`

`probreg.md` records the drop itself, in P0b — `probreg.md:313-314`:
> `- **§6.5 search spaces** still list XGBoost and CatBoost rows; §1.2 item 5 still requires CatBoost
>   probabilistic wiring. Both models are dropped.`

So within one file, P0b (✅ DONE) drops them and P4 (open) requires them. And `MASTER.md:67-68` sides
with the stale version:
> `the report must carry baseline comparisons (XGBoost / LightGBM / CatBoost / standard NN) and
> real-world datasets`

Three-way, all live. **Resolution:** the spec wins (`probreg.md:180` — *"a spec section contradicts
§1 → §1 wins"* is the wrong direction here; this is spec-vs-plan, and P0b is the later ruling). Fix
P4's model list and `MASTER.md` Decision 3's parenthetical in the same edit.

### X3-3 ⛔ "certified width selection numbers" — `flexnn-package.md` requires reproducing an artifact `width.md` says does not exist

`width.md:76-77`:
> `**This is an architecture result. It certifies none of §1's three models.**`

`width.md:11-13`:
> `Everything about *choosing* a width — which was never inside the G-WIDTH gate rule
> (`width-cert.md:308-318`, two clauses, both about the dial's behaviour, neither about selection) —
> is owned here.`

I verified `width-cert.md:308-318`: the pre-registered rule is (a) the noisy-easy clause and (b)
dial-separation + fit-at-floor. `width.md` is right about the gate.

`flexnn-package.md:62-66`:
> `- **Width.** The certified routing/selection numbers were produced by
>   `automl_package/examples/sinc_width_experiment.py`'s selector machinery, run inside
>   `kdropout_converged_width_experiment.py`. The package router has **never produced a certified
>   result**`

`flexnn-package.md:306-307` (FP-5 verify — this is a *gating condition*, not prose):
> `verify: the certified width selection numbers reproduce through the package router
> to a stated tolerance, side by side`

FP-5 cannot pass a verify line naming an artifact class the owning strand declares nonexistent.
There *are* deploy-claim result JSONs (`width-cert.md:236-239`, W6's two `RESULT:` lines), so a
referent exists — but calling them "certified" is precisely the misreading `width.md:229-230`
(WSEL-0) exists to correct.

**Resolution:** `width.md` owns the word. FP-5's verify must say "the width **deploy-claim** numbers
of `width-cert.md` W6/W10 reproduce…", naming the two `RESULT:` JSONs, not "certified selection
numbers".

### X3-4 ⛔ `MASTER.md` Decision 13 names a different "certified width mechanism" than `width.md` §1 does

`MASTER.md:97-100`:
> `13. **Selection = post-hoc DISTILLATION (user, 2026-07-17).** ... the certified width mechanism
>     (`automl_package/examples/sinc_width_experiment.py::_fit_selector_mse`).`

`width.md:39` (W-PERINPUT row):
> `| **W-PERINPUT** | the distilled router | a w **per input** | cheap | `fit_router` + routed predict
> (`automl_package/models/flexible_width_network.py:239-298`) |`

`flexnn-package.md:65-66`:
> `The package router has **never produced a certified result**; every document referencing it is a
> build spec or an unrun benchmark spec.`

`width.md` §1 defines its per-input model on the package path; `MASTER.md` Decision 13 canonises the
script path; `flexnn-package.md` says the package path is uncertified. All three are live. The
practical consequence is real: WSEL-8's and WSEL-9's W-PERINPUT arm runs code that, per
`flexnn-package.md`, has never reproduced anything.

**Resolution:** make `width.md` §1's W-PERINPUT row state both — package mechanism, certified against
the script mechanism by FP-5 — and make **FP-5 a declared dependency of WSEL-8** (it currently is
not; `width.md:379` reads `deps: WSEL-6, WSEL-7`).

### X3-5 ⛔ One name, two different tolerance rules, and no plan notices

Every plan states one rule for all arms —
`width.md:55-62`:
> `**Selection rule, fixed for all three (imported from the ProbReg decision, deliberately):**
> **cheapest-within-tolerance, NOT argmax.** ... "meaningfully" = exceeding twice a bootstrap-estimated
> standard error ... **The same rule applies to W-SWEEP's curve**, or W-SHARED and W-SWEEP are not
> answering the same question.`
(identically `depth-selection.md:50-54`; `probreg.md:357-366`.)

But the per-input arm's labels are produced by a **fixed 0.25 relative** tolerance, not a
standard-error rule. Verified in source, `automl_package/models/common/distilled_router.py:57` and
`:63-66`:
> `DEFAULT_TOLERANCE = 0.25  # matches sinc_width_experiment.py:333 DELTA_TIE`
> `def _cheapest_within_tolerance_labels(error_table, tolerance=DEFAULT_TOLERANCE) -> np.ndarray:`
> `    """Smallest-index capacity with `error <= (1 + tolerance) * row_min`, per row.`

Both plans quote that constant approvingly as the router's setting —
`width.md:350-351`: `labelling tolerance 0.25 (`automl_package/models/common/distilled_router.py:57-60`)`;
`probreg.md:407-409`: `cross-entropy on cheapest-within-tolerance labels (tolerance 0.25)`.

So the *global* arms select at "within twice a bootstrap SE" and the *per-input* arm labels at
"within 25% of the row minimum". These are different rules wearing one name. Nothing in any plan
flags it, and the §1 sentence "the same rule applies to all three" is therefore **false as written**
for the per-input arm — the same shape of error `probreg.md:62` calls out in the old benchmark spec.

**Resolution:** either state explicitly in all three §1s that the per-input labelling rule is a
*different, per-row* rule and why that is legitimate, or make the label rule use the same SE-based
criterion. Do not leave one name over two rules.

### X3-6 The three plans disagree on how many router implementations exist

`flexnn-package.md:42-44`:
> `Three independent router `nn.Module` classes exist, plus seven files that reuse one of them:`

`depth-selection.md:109-110`:
> `**Nine example scripts carry their own router implementations in total.**`

`flexnn-core.md:646-648`:
> `2. **Four router-MLP implementations now coexist:** `capacity_ladder_t2._RouterMLP`,
>    `capacity_ladder_k6._RouterMLP`, `depth_selection_toy._VectorRouterMLP`, and the package's
>    `distilled_router._CapacityRouterMLP`.`

Ground truth from disk:
`grep -rn "class .*Router.*(" --include=*.py automl_package/` → four MLP classes
(`capacity_ladder_t2.py:233`, `capacity_ladder_k6.py:75`, `distilled_router.py:84`,
`depth_selection_toy.py:607`), plus `moe_regression.py:104` `RouterOutputs` (a NamedTuple, not a router).
`grep -rln "_RouterMLP\|_VectorRouterMLP\|_CapacityRouterMLP"` → **10 files** (9 under `examples/`,
1 in `models/`).

**`flexnn-package.md` and `flexnn-core.md` are correct and agree** (3 script classes + package = 4;
10 files total). **`depth-selection.md:109-110` is wrong** — nine scripts *touch* a router, they do
not each carry their own implementation. It is a small error but it inflates the apparent size of
DD3 and would mis-scope FP-5.

**Resolution:** fix `depth-selection.md:109-110` to "nine example scripts use one of three script
router implementations", citing `flexnn-package.md` §1.2 rather than restating a count.

### X3-7 `flexnn-package.md` FP-3 miscounts the studies that depend on it

`flexnn-package.md:262-263`:
> `**The selection-set fraction must be a parameter**, not a baked-in constant — two studies are about
> to measure it.`

There are **three**: `probreg.md:384` (PB), `width.md:331` (WSEL-6), `depth-selection.md:378` (DSEL-8).
Small, but it is the tell that FP-3 was written with ProbReg out of view — the same omission as X1-4.

### X3-8 `MASTER.md` names a file that will never exist

`MASTER.md:29`:
> `Remaining split: F5 → its own `depth-ff.md`. Until that lands, `flexnn-core.md` is read WITH this warning.`

`depth-selection.md:16-18`:
> `- **This file absorbs the feed-forward depth study out of `flexnn-core.md`** (the F5/F5b/F5c block),
>   which closes the split that file has been pending.`

DSEL-0's spec (`depth-selection.md:182-187`) adds the index entry but never says to strike the
`depth-ff.md` line, and DSEL-0's verify (`:190-192`) only greps for `depth-selection.md`. The stale
sentence survives the task meant to fix it.

### X3-9 `flexnn-core.md` F7 asserts three *certified* instances of the selection law

`flexnn-core.md:343-347`:
> `(4) Selection = distillation — THREE certified instances of the same law (in-training selection
> fails; held-out distillation works): width (W-strand), depth (D8), and **k (K6 ...)**`

Against `width.md:77` (*"It certifies none of §1's three models"*) and `flexnn-package.md:65-66`
(*"The package router has never produced a certified result"*). F7 is the outward-facing deliverable;
this sentence would publish the misreading `width.md` was created to stop.

---

## X4 — Gaps: needed by someone, owned by nobody

### X4-1 ⛔⛔ Selection-cost accounting is REQUIRED by all three dials and OWNED for one — the brief's premise is half wrong

`width.md:313-324` (WSEL-5) owns it and claims the others inherit:
> `**Doctrine:** this gap is **programme-wide**, not width-specific — depth, joint and MoE draw from the
> same module. Build it in the shared module so the other strands inherit it; do NOT build a
> width-local copy.`

**Depth does NOT say it inherits it.** `grep -in "accounting" depth-selection.md` → **no hits**; the
file never cites `capacity_accounting.py`, `metrics-accounting.md`, WSEL-5, or `width.md` at all.
Yet it requires the output:
`depth-selection.md:419`: `each arm's end-to-end cost including selection`
`depth-selection.md:429-430`: `**Every arm's number includes its selection cost.**`
and `depth-selection.md`'s §3.6 constants table (`:149-156`) has **no** selection-cost row, where
`width.md:200` does:
> `| per-model selection cost | **WSEL-5** | the accounting module's own selftest artifact |`

**ProbReg is worse — entirely uncovered.** `grep -in "accounting\|selection cost" probreg.md` →
**no hits**, while:
`probreg.md:25-28`: `Every one is scored end-to-end and **costed end-to-end**: the selection step is
*inside* the model, never a side-analysis`
`probreg.md:553-554`: `report M3's full cost next to its accuracy (the efficiency claim is a ratio and
this is its denominator)`
and `probreg.md`'s §3.6 table (`:199-206`) has no cost row. Worse, WSEL-5's non-goal may exclude
ProbReg outright — `width.md:326`: `do not extend to families outside this programme's architectures.`

**Verdict: CONFIRMED gap, larger than the brief's candidate (i).** Width owns it; depth silently
assumes it without a dependency; ProbReg is neither covered nor mentioned, and one reading of
WSEL-5's non-goal excludes it.

**Resolution:** move WSEL-5 into `flexnn-package.md` (it writes a shared module and its spec doc —
that is package work, not width work), give it a ProbReg branch explicitly, and add
`deps: <that task>` to DSEL-10/DSEL-11 and to P3/P4.

### X4-2 ⛔ WSEL-5's write set targets a file FP-1 turns into a shim, with no ordering

`width.md:315-316` (WSEL-5 write set):
> `**Files (write set):** `automl_package/examples/capacity_accounting.py` ·
> `docs/plans/capacity_programme/shared/metrics-accounting.md``
with `width.md:327`: `deps: WSEL-3, WSEL-4` — **no FP-1**.

`flexnn-package.md:208-215` (FP-1):
> `**Files (write set):** `automl_package/utils/` (new accounting module) ·
> `automl_package/examples/capacity_accounting.py` (becomes a shim) ...`
> `Leave `automl_package/examples/capacity_accounting.py` as a re-export shim`

and `flexnn-package.md:219-220` knows about width but does not bind it:
> `**Non-goals:** do not add selection-cost accounting here — that is `width.md`'s task, and it lands
> after this module has a stable home.`

The dependency is asserted in one direction only. If WSEL-5 runs first it writes logic into a file
FP-1 will empty; if FP-1 runs first, WSEL-5's write set names the wrong file.

**Resolution:** add `deps: flexnn-package.md FP-1` to WSEL-5 and change its write set to the new
`automl_package/utils/` module.

### X4-3 ⛔ A shared bootstrap / standard-error helper is needed by three tasks and built by none

Needed: WSEL-3 (`width.md:281-283`), DSEL-6 (`depth-selection.md:354-356`), PA
(`probreg.md:344-345`) — all "twice a bootstrap standard error". FP-7 only *inventories* the existing
duplication (`flexnn-package.md:329-331`), explicitly: `Output is an inventory, not a cleanup`.
No task builds the shared implementation. See X2-1.

### X4-4 ⛔ Nothing owns `flexnn-core.md` after FP-0 and DSEL-0 empty it

`flexnn-package.md:198-199` (FP-0):
> `with `depth-selection.md` having taken the feed-forward study, `flexnn-core.md` retains
> only the MoE comparison and the unified report, which is the split that file has been pending.`

But `MASTER.md:24-29` says the rule is **one strand file per workstream**, and F6 (MoE) + F7 (unified
report) are two workstreams — plus F9/F10/F12/F13, which FP-0 does not mention at all and which are
still live (X1-1, X1-2, X1-3). FP-0's verify (`flexnn-package.md:203-204`) asserts a state
(`flexnn-core.md` retains only MoE + unified report) that its own spec does not produce.

**Resolution:** FP-0's spec must enumerate the disposition of **every** remaining `flexnn-core.md`
task — F6, F7, F8, F9, F10, F11, F12, F13 — not just the package block.

### X4-5 The deliverable reports: owners verified, one brief candidate REFUTED

Per-strand report owners all exist and are disjoint: P6 → `docs/reports/probreg_kselection/`
(`probreg.md:537`); WSEL-10 → `docs/reports/width_selection/` (`width.md:405`); DSEL-12 →
`docs/reports/depth_selection/` (`depth-selection.md:440`). **But each collides with `flexnn-core.md`
— see X1-2 and X1-3.**

The brief's candidate — *"an existing delivered report built on a retired definition with no successor
task"* — splits in two:
- `docs/reports/probreg_kselection/probreg_kselection.md`: **covered.** `probreg.md:90-92` names it as
  predating §1 and P6 corrects it (`probreg.md:540`: *"must be **corrected, not appended to**"*).
- `docs/reports/probreg_toys/report_a_probreg_toys.md` (MASTER strand 2, delivered): **REFUTED as a
  concern.** `grep -ci "shared-k\|variable-k\|dynamic-k"` on it returns **0** — it carries none of the
  retired framing. `flexnn-core.md:504-505` corroborates: *"report (a) covers ProbReg-vs-baselines
  ONLY — zero mentions of k-selection/arbiter/router."* No successor task needed.

### X4-6 `width.md` never states the width training schedule it depends on

`width.md:25-27` refers to `its certified joint width-dial schedule` and `width.md:307-308` forbids
changing it, but the file never says what it is. The definition lives in two *other* plans:
`flexnn-package.md:70-73`:
> `its own docstring (`:15-16`) records that it is *"specialised here to sum ALL configured widths
> every step rather than a sampled subset"* of the certified sandwich schedule`
`depth-selection.md:209-211`:
> `width uses the sandwich schedule (always the smallest and largest width, plus two random middles)
> with a head per width`

They agree with each other, which is lucky rather than designed. Under the programme's own
one-definition-one-file rule the width schedule belongs in `width.md` §1 and the other two should
point at it.

### X4-7 MASTER Decisions 9 and 16 are cited by no plan; Decision 17 by one

`grep -n "Decision 9\|Decision 16\|Decision 17"` across all four → **only**
`depth-selection.md:304` (Decision 17). Decision 9 (trajectory discipline, `hit_cap=False`, an
early-stop-OFF confirmation at ≥4× budget for load-bearing verdicts) and Decision 16 (optimisation
exonerated before architecture is blamed) bind every training battery here. See X7-2.

---

## X5 — Cross-plan write-set collisions

Union of declared write sets. **Every row below is two or more tasks in different plans writing one
path with no ordering between them.**

| Path | Tasks (plan) | Ordering declared? |
|---|---|---|
| `docs/plans/capacity_programme/MASTER.md` | WSEL-0 (width:225) · DSEL-0 (depth:180) · FP-0 (package:194) · P0 ✅done (probreg:235) | **NONE** — all three open tasks say `deps: none` |
| `docs/plans/capacity_programme/flexnn-core.md` | DSEL-0 (depth:181) · DSEL-1 (depth:230) · FP-0 (package:195) | **NONE** between DSEL-0 and FP-0 (both `deps: none`) |
| `automl_package/enums.py` | PA (probreg:374) · WSEL-2 (width:255) · FP-3 (package:244) | **NONE** — PA `deps: P1`; WSEL-2 `deps: WSEL-1`; FP-3 `deps: FP-1`. Three plans, three unrelated chains |
| `automl_package/models/probabilistic_regression.py` | P1 (probreg:429) · PA (probreg:374) · P2 "possibly" (probreg:449) · FP-3 (package:247) | **NONE** cross-plan |
| `automl_package/models/flexible_width_network.py` | WSEL-1 (width:238) · WSEL-2 (width:255) · WSEL-3 (width:279) · FP-1 (package:210) · FP-3 (package:244) · FP-4 (package:277) | **NONE** cross-plan |
| `automl_package/models/flexible_neural_network.py` | DSEL-3 (depth:309) · DSEL-6 (depth:352) · FP-1 (package:210) · FP-3 (package:245) | **NONE** cross-plan |
| `automl_package/models/independent_weights_flexible_neural_network.py` | DSEL-3 (depth:308) · FP-3 (package:246) · FP-6 (package:312) | **PARTIAL** — FP-6 defers the prior fix (package:317-318) but FP-3 does not defer anything |
| `automl_package/examples/capacity_accounting.py` | WSEL-5 (width:315) · FP-1 (package:211) | **NONE** — see X4-2 |
| `automl_package/models/common/distilled_router.py` | FP-5 (package:292) writes it; PC/WSEL-7/DSEL-9 are each authorised to change its frozen defaults (probreg:420, width:359, depth:401) | **NONE** — see X1-5 |
| `tests/test_flexible_width_network.py` | WSEL-1 (width:239) · WSEL-2 (width:256) · WSEL-3 (width:280) · FP-3 "tests" (package:247) · FP-3 retires a test in it (package:260-261) | **NONE** cross-plan |
| `automl_package/examples/probreg_benchmark.py` | P4 (probreg:507) · F12 (flexnn-core:578) | **NONE** — F12 is not marked superseded |
| `docs/probreg_benchmark/benchmark_spec.md` | P0 ✅done (probreg:236) · F12-a (flexnn-core:577) | **NONE** |
| `docs/reports/probreg_kselection/` | P6 (probreg:537) · F10 (flexnn-core:486) | **NONE** |
| `docs/plans/capacity_programme/shared/` (unnamed notes) | P2 (probreg:448) · P5 (probreg:526) · DSEL-1 (depth:229) · FP-4 (package:277) · FP-5 (package:293) · FP-7 (package:327) | filenames unspecified → low but nonzero collision risk |

`MASTER.md:144-145` states the discipline these violate:
> `- **Single writer:** workers return findings; the ORCHESTRATOR writes strand files and ledgers.
>   Fan out on compute; serialise writes. Write sets are declared per task; wave partition =
>   `deps:` + write-set overlap.`

Wave partition by `deps:` + write-set overlap **cannot be computed** from these files as written: the
overlap is real, the `deps:` are absent, so every wave computation collapses the whole programme into
one serial chain or produces a collision. This is the mechanical reason nothing can be dispatched yet.

---

## X6 — Dependency integrity across plans

### X6-1 ⛔ `width.md` declares no dependency on `flexnn-package.md`, which declares width blocked on it

`flexnn-package.md:190`:
> `FP-3 (the API) is what `width.md` and `depth-selection.md` block on; it is deliberately early.`

`grep -n "flexnn-package" width.md` → **no hits**. WSEL-2 (`width.md:271`) reads
`deps: WSEL-1`. WSEL-5 (`:327`) reads `deps: WSEL-3, WSEL-4`. Width's task order
(`width.md:217`) has no cross-plan entry at all. **The dependency exists in one file only.**

### X6-2 ⛔ `probreg.md` declares no dependency on `flexnn-package.md` either, and is not listed as its consumer

`grep -n "flexnn-package" probreg.md` → **no hits**. PA (`probreg.md:378`) reads
`deps: P1, and BOTH user decisions above`. Meanwhile FP-3's write set includes
`automl_package/models/probabilistic_regression.py` (`flexnn-package.md:247`) and
`flexnn-package.md:354-355` says:
> `No ProbReg selection-mechanism work beyond adopting the shared API (owned by `probreg.md`).`

So the package strand will edit ProbReg's model file for the API while disclaiming ProbReg
selection work, and ProbReg's own API task does not know the package strand exists.

### X6-3 `depth-selection.md` DSEL-4's cross-plan dependency is real but unnamed

`depth-selection.md:335`:
> `*Orchestration:* parallel: no · deps: `flexnn-package.md` API + router tasks; DSEL-3 ·`

FP-3 and FP-5 do deliver what DSEL-4 assumes (`flexnn-package.md:242`, `:290`). **Substance: sound.**
**Form: the only cross-plan `deps:` in the four files names no task ID**, so a wave computation
cannot resolve it. Fix to `deps: flexnn-package.md FP-3, FP-5; DSEL-3`.

### X6-4 `depth-selection.md` DSEL-5 cites an ID that does not exist in the owning plan

`depth-selection.md:343`:
> `**Files (write set):** none in this strand — **tracked here, owned by `flexnn-package.md` (DD4)**`
`depth-selection.md:347-348`:
> `verify: n/a — closed when `flexnn-package.md` closes DD4.`

`grep -n "DD4" flexnn-package.md` → **no hits.** `DD4` is `depth-selection.md`'s own defect ID. The
matching task is **FP-6** (`flexnn-package.md:310-315`), which does deliver the primitive.
**Substance: sound. Reference: dangling.** Fix to "FP-6".

### X6-5 `depth-selection.md` DSEL-1b's forward reference to FP-4 is correct

`depth-selection.md:266-267`:
> `**Default: the sandwich shape**, not summing all depths every step (which is what the shipping width
> class does and is itself an unmeasured deviation — `flexnn-package.md` FP-4).`

FP-4 exists (`flexnn-package.md:275`) and is exactly that. **Verified correct** — the only clean
cross-plan reference in the set, and the model the others should copy.

### X6-6 `depth-selection.md` DSEL-2 depends on DSEL-1 but needs DSEL-1b

`depth-selection.md:300`: `deps: DSEL-1`, while `depth-selection.md:288-289` reads:
> `**Spec:** **The primary claim of this strand.** Using DSEL-1b's nested training scheme, run the
> feed-forward ... arm`

Intra-plan, but it is on the strand's primary claim and the stated order (`:171`) does put DSEL-1b
first. Fix the `deps:` line to `DSEL-1b`.

---

## X7 — Coverage against `MASTER.md`'s own goals

### X7-1 ⛔ Decision 3 (amended) — see X3-1. A binding Decision that two plans contradict. Highest priority.

### X7-2 ⛔ Decision 9 is implemented by no plan

`MASTER.md:86-89`:
> `9. **Trajectory discipline** (binding on every training conclusion): full per-width/depth
>    held-out trajectories, convergence flags trajectory-verified, `hit_cap=False` required;
>    load-bearing verdicts get an early-stop-OFF confirmation run at ≥4× the self-terminated
>    budget. No conclusion from an endpoint.`

Cited by none of the four (`grep "Decision 9"` → no hits). Only `depth-selection.md:302-304` carries
the substance in a `verify:` line:
> `then per-(seed, depth) JSONs with full held-out trajectories and the convergence gate computed on
> the metric the bar reads (MASTER Decision 17).`

`width.md`'s load-bearing verdict tasks — WSEL-4, WSEL-8, WSEL-9 — carry **no** trajectory or
convergence condition in any `verify:` line. `probreg.md` contains **one** occurrence of any word in
{trajectory, convergence, trustworthy, hit_cap} in 567 lines, and P3/P4 train models. Decision 9 is
the rule this programme has case law for ([[feedback_check_loss_trajectory_before_concluding]]); it
is unenforced in three of four plans.

### X7-3 ⛔ Decision 16 is implemented by no plan

`MASTER.md:117-122`:
> `16. **Optimization is exonerated BEFORE architecture is blamed (2026-07-20).** No arm is recorded
>     as an architecture failure until either it is shown to fit the **training** set, or a
>     documented escalation ladder ... has been run.`

Cited by none. **Notably, `depth-selection.md` DSEL-1 SATISFIES it without citing it** —
`depth-selection.md:204` records `train 0.970/1.000, held-out 0.432/0.744`, i.e. the arm demonstrably
fits the training set, which is Decision 16's first exit. The reasoning at `:214-216` is therefore
**sound and consistent with Decision 16**, not a violation — I checked this specifically because the
task drops an escalation ladder. But every *future* negative arm (DSEL-2's feed-forward cells,
WSEL-8, P3) has no Decision-16 gate written into its `verify:`.

### X7-4 Decisions verified as adequately covered

- **Decision 13** (selection = post-hoc distillation): all three dial strands carry it as a §5
  non-goal (`probreg.md:566`, `width.md:432`, `depth-selection.md:463`). ✔ (but see X3-4 on *which*
  mechanism Decision 13 canonises).
- **Decision 14** (positive control first, alone): `probreg.md:183`/`:496`; `width.md:179`/`:305`/`:374`;
  `depth-selection.md:137`/`:293`/`:333`/`:414`. ✔ Best-covered decision in the set.
- **Decision 15** (protocol parity / one difference at a time): `probreg.md:66`, `width.md:65`,
  `depth-selection.md:56`. ✔
- **Decision 10** (report authorship, no AI provenance): `probreg.md:539`, `width.md:406`,
  `depth-selection.md:441`. ✔
- **Decision 11** (commits user-gated): in all four HALT lists. ✔
- **Decision 12** (worker tiering): every task carries a `tier:`. ✔
- **Decision 2** (MSE-only, width): `width.md` cites it as a non-goal (`:431`) but never states the
  strand's own metric anywhere. Minor gap.

### X7-5 ⛔ Gate G-JOINT is advanced by no plan, and no plan says so

`MASTER.md:19`:
> `| 4 | `width-depth.md` | Joint 2-D capacity dial + transformer halting → **G-JOINT** | ... | **J0 RAN
> 2026-07-17 — J-1/J-2 DEAD ...; G-JOINT BLOCKED, J-3 redesign ESCALATED to user** |`

All four plans list joint work as a non-goal (`width.md:431-432`, `depth-selection.md:462-463`,
`flexnn-package.md:355-356`, `probreg.md` by omission). That is defensible — G-JOINT is parked on a
user decision — but the three new plans introduce a *changed* selection rule
(cheapest-within-tolerance) and a *new* shared API that `width-depth.md` J-3 will inherit without
knowing. No task notifies it.

**Resolution:** one line in `MASTER.md` recording that `width-depth.md` is stale with respect to the
new selection rule and the FP-3 API, so J-3 is not designed against the old contract.

### X7-6 The programme's own single-definition rule is violated by the plans that assert it

`MASTER.md:187-189`:
> `**Generalised rule (user, 2026-07-20): one execution-level strand file per workstream, MASTER stays
> an index.** A definition may exist in exactly one file; everywhere else points at it.`

Definitions currently in more than one file, per findings above: the selection API (X1-4, three
files), the router defaults (X1-5, three files), the width training schedule (X4-6, stated in two
files, absent from its owner), the ProbReg benchmark model set (X1-1/X3-2, three files disagreeing),
the report ownership (X1-2/X1-3, two files each). **The rule is asserted in all four plans and
satisfied by none of them across the set.**

---

# THE FIVE THINGS TO FIX BEFORE ANY EXECUTION STARTS

### 1. Get a user ruling on Decision 3, and write it into `MASTER.md` before WSEL-9/DSEL-11 exist as tasks
**Why first:** it is the only item that can delete two whole tasks and two report sections, and it is
a user decision, not a run's. `MASTER.md:65-70` currently scopes "reports are toys-only" **onto**
width and depth; `width.md:384` and `depth-selection.md:421` do real data anyway.
**Edit:** `MASTER.md` Decision 3 — either extend the 2026-07-20 amendment to width/depth explicitly,
or mark WSEL-9/WSEL-10-realdata and DSEL-11/DSEL-12-realdata as out of scope. Then mirror the ruling
into `width.md` §5 and `depth-selection.md` §5 non-goals in the same turn.
**Lands in:** `MASTER.md`, then `width.md` and `depth-selection.md`.

### 2. Collapse the three "selection API" tasks into one, and give the other two a real `deps:` line
**Why:** `automl_package/enums.py` is in three write sets from three plans with three unrelated
dependency chains (X5), and `flexnn-package.md:17-20` asserts a division of labour that
`width.md:254` and `probreg.md:330` contradict.
**Edit:** in `flexnn-package.md`, correct `:17-20` to list all **three** consumers including ProbReg,
and correct `:262-263` ("two studies" → three). In `width.md`, replace WSEL-2's spec with "consume
FP-3" and set `deps: flexnn-package.md FP-3, WSEL-1`; delete the "coordinate with PA" line at `:267-268`.
In `probreg.md`, do the same to PA and set `deps: flexnn-package.md FP-3, P1`.
**Lands in:** `flexnn-package.md` (owner), `width.md`, `probreg.md`.

### 3. Empty `flexnn-core.md` for real — F12, F10, F13 and F7 all still own what the new plans own
**Why:** the programme's stated root cause was one thing owned in two places
(`MASTER.md:180-189`). F12 creates the same driver and spec as P4 **using the retired shared-k /
variable-k definition** (`flexnn-core.md:589-593`); F10 creates P6's report directory; F13 items 1-3
are FP-5 and FP-2; F7 writes the width and depth report content WSEL-10/DSEL-12 now own, and asserts
"THREE certified instances" against `width.md:77`.
**Edit:** rewrite FP-0's spec (`flexnn-package.md:196-199`) to require an explicit disposition line
for **every** remaining `flexnn-core.md` task — F6, F7, F8, F9, F10, F11, F12, F13 — each either
struck through with a pointer to its new owner or explicitly retained; and make FP-0's verify check
that no `flexnn-core.md` task's write set intersects any of the four new plans'.
**Lands in:** `flexnn-package.md` FP-0 (and, on execution, `flexnn-core.md`).

### 4. Make the cheap global chooser, the bootstrap SE helper, and selection-cost accounting one shared thing each
**Why:** WSEL-3, DSEL-6 and PA each build the identical selector rule (X2-1); nobody builds the
bootstrap helper they all need (X4-3); and selection-cost accounting is owned by width, silently
assumed by depth, and entirely absent for ProbReg despite `probreg.md:25-28` mandating end-to-end
costing (X4-1). FP-7 will inventory duplicated SE helpers while three tasks add three more.
**Edit:** add one task to `flexnn-package.md` — "the shared selection primitives": the
cheapest-within-tolerance-at-2×SE selector, the bootstrap SE helper, and selection-cost accounting
(moving `width.md` WSEL-5 into it, ProbReg branch included). Reduce WSEL-3/DSEL-6/PA to per-family
plumbing over it, and add that task to the `deps:` of P3, P4, WSEL-8, WSEL-9, DSEL-10, DSEL-11.
**Lands in:** `flexnn-package.md` (new task), then `width.md`, `depth-selection.md`, `probreg.md`.

### 5. Resolve the two-rules-one-name tolerance defect, and the "certified selection numbers" wording
**Why:** all three §1s claim one selection rule binds all three arms, but the per-input arm labels at
a fixed 25% relative tolerance (`distilled_router.py:57`, `:63-66`) while the global arms use twice a
bootstrap SE (X3-5) — an arm differing from its comparator in an undeclared respect, which is exactly
what `probreg.md:66-68`'s confound doctrine makes non-dispatchable. Separately, FP-5's **verify line**
(`flexnn-package.md:306-307`) requires reproducing "certified width selection numbers" that
`width.md:77` says do not exist (X3-3), so FP-5 cannot be closed as written.
**Edit:** (a) in `width.md` §1, `depth-selection.md` §1b and `probreg.md` §1/PA, add one sentence
stating that the per-input labelling rule is a different per-row criterion and why that is legitimate
— or change the label rule. (b) rewrite FP-5's verify to name the `width-cert.md` W6 deploy-claim
`RESULT:` JSONs (`width-cert.md:236-239`) instead of "certified selection numbers", and add FP-5 to
WSEL-8's `deps:`.
**Lands in:** `width.md`, `depth-selection.md`, `probreg.md` (a); `flexnn-package.md`, `width.md` (b).

---

## What I could not verify

- **Whether the user has already ruled that width/depth may use real data** (X3-1). Nothing on disk
  records such a ruling; `MASTER.md`'s written amendment says the opposite. If the ruling was verbal
  and unrecorded, that is itself the defect.
- **Whether `docs/probreg_benchmark/benchmark_spec.md`'s protocol sections are genuinely
  family-agnostic enough to be shared with width/depth** (X2-5). I read its section headers and the
  model/baseline sections; I did not read §6 in full.
- **The currency of the line citations inside `depth-selection.md` DSEL-1b** into `depth.md` and
  `docs/depth_capacity/verdict_per_input_depth.md`. I verified the plan-gate suite passes (which
  proves the lines exist), not that they say what is claimed.
- **Coverage.** This pass compared four plans against each other and against `MASTER.md`. It did not
  audit any single plan's internal completeness against the code, which is what the per-plan
  reviewers are for. Two of my findings (X1-1, X1-3) came from a file *outside* the four —
  `flexnn-core.md` — which suggests the remaining archived/frozen strand files
  (`depth.md`, `width-cert.md`, `flexnn-moe.md`, `width-depth.md`, `probreg-report.md`) should get
  the same intersection check before dispatch. I did not run it.
