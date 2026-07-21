# Open decisions — everything waiting on the user

**Written 2026-07-21 by the root, at the user's request, after a planning session and an
execution-level audit of both live strands.** One row per decision that the root cannot take alone.
Each carries a **recommendation**, so the default path is explicit and the user is confirming or
overturning rather than designing from scratch.

**How to use this file.** Answer in any order; they are independent unless a row says otherwise.
When a row is settled, the root records it in `MASTER.md`'s decision register (the authority) and
strikes the row here. **This file is a queue, never a source of truth** — a decision that lives only
here has not landed.

**Status: 6 of 7 settled 2026-07-21. Only D-4 remains open** — the user asked for a clearer
explanation before ruling it. Settled rows are retained with their reasoning rather than deleted, so
the rejected alternative stays auditable.

**Status legend:** 🔴 blocks dispatchable work now · 🟡 blocks work that is itself blocked · ⚪ no
current block, wanted for completeness.

---

## ✅ D-1 — SETTLED (user, 2026-07-21): widen the grid, bypass included. → MASTER Decision 25

**Blocks:** `probreg.md` **P3** and **P4** — every cheap-vs-expensive headline in the strand.

**The problem.** M1 and M2 read a ladder over k ∈ {1..k_max} **including the bypass rung** (rung 1 =
direct regression, "do not discretize at all"). M3 — the expensive reference — selects from a grid
that **starts at k=5** (`K_GRID = (5, 8, 10, 12)`, `automl_package/examples/report_a_benchmark.py:102`).

So on any cell where the honest answer is the bypass — **including this strand's own smooth-data
negative control, where the bypass IS correct** — M1 can select it and M3 structurally cannot. M1
then beats its own reference for a reason with nothing to do with selection quality. **Note the
direction: this flatters the cheap method**, which is the direction a reader is least likely to
suspect and the one we most need to not get wrong.

**✅ RECOMMENDATION — widen M3's grid to span the same rungs as the ladder, bypass included.**
Both sides must be able to return the same answer or the comparison is not like-for-like.

- **Cost consequence, stated honestly:** widening raises M3's price, and **M3's price is the
  denominator of every efficiency claim in this strand.** A cheaper reference makes the cheap arms
  look worse, not better — so this is a conservative choice, not a convenient one.
- **If you prefer to keep the grid narrow for cost:** then every M1-vs-M3 comparison must **exclude
  cells whose selected rung falls outside M3's grid**, and the report must say so. What is not
  acceptable is a narrow grid plus an unqualified "the cheap read matches the expensive sweep".

---

## ✅ D-2 — SETTLED (user, 2026-07-21): fix σ in TRAINING too. → MASTER Decision 26 (model-definition change; suite must be re-baselined)

**Blocks:** nothing today, but it determines whether `flexnn-package.md` **FP-12** (wave zero) is
scoped as selection-only or also touches training. **Answer this before FP-12 is dispatched.**

**The problem — a real asymmetry, not a technicality.** MASTER Decision 24 fixes the **selection**
metric at a likelihood with constant σ, and FP-12 implements that. But ProbReg's **training** loss is
each rung's NLL with the model's own **learned** per-class `log_var` — so the model still fits a
variance, and we would be training on one objective while selecting on another. Decision 2 (amended)
already fixes σ at the generator's truth for **width**; ProbReg has no equivalent ruling, because
changing it touches `probreg.md` §1's model definitions, which are yours.

**What changes if σ is fixed in training.** With σ constant, the rung NLL reduces to squared error up
to a constant — so "fixed-σ training" is plain MSE training on the mixture mean. The per-class heads
then predict **means only**, and the model's predicted spread becomes purely the **between-component**
spread.

**✅ RECOMMENDATION — YES, fix σ in training too.** Three reasons, in order of weight:

1. **It removes an escape hatch that directly undermines the dial under study.** With a learned
   per-class variance, a single component can absorb spread by widening itself — so the model can fit
   dispersed data *without* needing more components. That is exactly the confound "does this input
   need more k?" is trying to measure. Fixing σ forces dispersion to be explained by **structure**
   (more components) rather than by **width** (a fatter single component).
2. **Train/select mismatch is a confound we would have to defend.** Training to optimise a quantity
   we then do not select on invites the reviewer's obvious question, and we have no answer prepared.
3. **Parity with width.** Same principle, two dials, one ruling — and it keeps the two strands'
   numbers arguing from the same premise.

**Cost of saying yes:** it is a **model-definition change** (§1), so prior k-dropout numbers become
old-objective and must be relabelled — the same treatment the retired training schedule already gets.
FP-12 grows from a scoring change to a scoring **and** training change.

**Cost of saying no:** cheaper and faster, and defensible — but the asymmetry must be **declared in
the report** as a known limitation, and reason 1 above remains a live confound on every k result.

---

## ✅ D-3 — SETTLED (user, 2026-07-21): keep it as a labelled comparison arm. → MASTER Decision 27

**Blocks:** the exact arm list of `probreg.md` **P11** (which is itself behind FP-12/P7/P10).

**Context.** Three head layouts exist. Separate-heads is now pinned as the model of record (§1).
The no-component layout is **blocked** under nesting, because the prefix guarantee is vacuous there —
and it serves as P11's **mechanism control**. The middle layout (one head, per-class outputs) is
genuinely nested but its components are computed from the zero-padded probability vector, so
truncation leaks into every surviving component.

**✅ RECOMMENDATION — keep it, as a labelled comparison arm, never a default.** This is what the root
already applied to §1; the row exists so you can overturn it rather than inherit it silently. Keeping
it costs one arm in a battery that is already running, and it distinguishes "components help" from
"*independent* components help" — a distinction we would otherwise be guessing at.

---

## 🔴 D-4 — STILL OPEN — the user asked for a clearer explanation before ruling. The cross-k class-identity defect: fix, or continue to avoid?

**Blocks:** nothing in the current battery (which runs `REGRESSION_ONLY` throughout, where the defect
cannot fire). Recorded so it is not re-discovered a fourth time.

**The problem.** When cross-entropy is active, per-k re-binned targets redefine class identity across
k, which conflicts with the nested ladder. Today it is *avoided* by freezing the training mode, and
what shipped is a **runtime warning only** (`automl_package/models/probabilistic_regression.py:535-536`),
not a behaviour change. Two candidate fixes are on record: k-stable binning, or documenting that
k-dropout requires likelihood-only training.

**✅ RECOMMENDATION — document the constraint; do not build k-stable binning now.** The battery never
activates the path, and building a fix for an unexercised configuration spends effort on the one dial
nobody is measuring. **But** the warning should become a hard error under `NESTED` + cross-entropy, so
the trap cannot be entered silently — that is a few lines and belongs with FP-12 or P10.

---

## ✅ D-5 — SETTLED (user, 2026-07-21): PUSH. Executed the same day.

**Current state:** `master` is **47 commits ahead**, unpushed, by a standing decision from earlier in
the programme. The tree is clean and gates are 9/9.

**✅ RECOMMENDATION — push now.** The original reason to hold was churn during an unstable planning
phase; the plan is now internally consistent and gated. Forty-seven commits of unpushed work on one
machine is the largest single-point-of-failure risk in the programme right now, and it costs nothing
to remove. *(Outward-facing, so it stays yours — the root will not push without an explicit go.)*

---

## ✅ D-6 — SETTLED (user, 2026-07-21): stay parked. → MASTER Decision 28

**Current state:** parked completely at your instruction; the live programme is width + ProbReg only.
Two untried levers (per-depth output layers, regularisation) and a missing early-exit literature
survey are recorded in the strand header. **Neither lever has a written task** — writing them is the
first root action on unpark.

**✅ RECOMMENDATION — stay parked until both live strands close.** Width's architecture comparison is
expected to tell us whether the ordering/cascade ideas transfer, and the transformer-port decision
already names depth as the better port target. Unparking before those land means designing depth's
tasks on assumptions the width work is about to test.

---

## ✅ D-7 — CLOSED (user, 2026-07-21): "I think you answered all." Reopen by naming a question.

You noted several architecture questions for ProbReg that were deferred. The root cannot list them
for you. **Two are already answered** by this session's work and can be struck if they were yours:
whether ProbReg has a nested structure (yes — masked-softmax prefix, and it comes free from the
existing architecture), and whether it needs an architecture-comparison task like width's (it now has
one — P11, on head structure). **Anything else, please state and the root will scope it.**

---

## Not on this list, deliberately

These are **root-owned** and need no ruling — recorded so their absence is not mistaken for an
oversight:

- **The two voided regularisation checks** (`probreg.md` P8, `width.md` WSEL-11) — both re-run on
  their own strand's sanctioned objective. Mechanical; no decision needed. *(P8 additionally waits on
  FP-12 for a compliant metric to select on.)*
- **FP-12** (the fixed-σ scorer) — scoping is settled apart from **D-2** above.
- **The 2%-bar fix** (width) — per-input routing agreement rather than averaged error. Agreed
  earlier; unbuilt; no new decision.
- **Deletion manifests** (`probreg.md` P9 step 7, `width.md` WSEL-17) — these WILL come to you, but
  only as a manifest at execution time, under the four mechanical eligibility checks, attended. Not a
  decision to take in advance.
- **Reports** — parked behind the joint results review (Decision 23). Already settled.
