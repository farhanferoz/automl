# Strand: FlexNN package — consolidation, API, cleanup

**Owns the CODEBASE**: what lives in `automl_package/models/` versus `automl_package/examples/`, the
one selection API shared by every capacity family, the de-duplication of implementations, and the
dead-code sweep. Read `MASTER.md` + this file. If another document disagrees with this one about
where code lives or what the selection API looks like, **this one wins**.

**Mandate (user, 2026-07-20):** *"ensure all width & depth work is part of the FlexNN codebase. No
stale duplicated code, proper organisation, API, cleanup."*

**Why this strand exists.** The research was done in standalone scripts that import nothing from the
package, so the code that produced the certified results and the code that ships are, in several
places, different implementations of the same idea. This is not a tidiness problem: it means a
certification and a shipped feature can drift without anything detecting it, and it has already
produced one class that cannot participate in its own strand's experiments.

**Consumers.** `width.md`, `depth-selection.md` and `probreg.md` all depend on this strand for the
selection API, the single router, and the shared selection primitives (FP-9).

**Correction, 2026-07-20.** An earlier draft of this paragraph said "neither of them defines an API".
That was false and is the exact problem this strand exists to stop: **two sibling plans define a
selection API today** — `width.md` WSEL-2 ("One enum passed at construction selecting
`SHARED` / `PER_INPUT` / `SWEEP`") and `probreg.md` PA ("one enum `KSelection` with
`GLOBAL_ARBITER` / `PER_INPUT_ROUTER` / `GLOBAL_SWEEP`"). Two plans, two enum names, two member
vocabularies, for one concept.

**Required end-state (the root is applying this to the sibling plans; this file is the authority):**
FP-3 ships the one enum, and WSEL-2, PA and `depth-selection.md` DSEL-4 **consume** it — they migrate
their family onto `CapacitySelection` and none of them declares an enum of its own. If a sibling plan
still names its own enum when its task is dispatched, that plan is stale and this file wins (per the
precedence rule in this file's opening paragraph).

---

## 1. The inventory this plan is built on

Surveyed 2026-07-20, read-only, every claim below re-verified at the root before being written here.

### 1.1 The package and the research scripts are circularly coupled

**Verified, both directions:**
- `automl_package/examples/capacity_accounting.py:62-63` imports `FlexibleHiddenLayersNN` and
  `FlexibleWidthNN` **from the package, at module level**.
- `automl_package/models/flexible_width_network.py:290` and
  `automl_package/models/flexible_neural_network.py:492` import `executed_flops`
  **from `automl_package.examples.capacity_accounting`, inside a method body**, with the comment
  *"avoids a load-time circular import (capacity_accounting.py imports this class)"*.

Shipping package code depends on a research script, and the dependency is held together by deferred
imports. **This is the single clearest piece of evidence that the two trees are entangled rather than
merely duplicated**, and it is the first thing to fix, because every other move is constrained by it.

### 1.2 The router exists three times, and the package version is a strict subset

Three independent router `nn.Module` classes exist, plus seven files that reuse one of them:

| Implementation | Location | Input | Training objective |
|---|---|---|---|
| the original | `automl_package/examples/capacity_ladder_k6.py:75`, `:92` | scalar only | hard-label CE **or** soft-target CE |
| a vector port | `automl_package/examples/capacity_ladder_t2.py:233`, `:256` | vector | soft targets only |
| a second, independent vector port | `automl_package/examples/depth_selection_toy.py:607`, `:624` | vector | hard-label CE only |
| **the package** | `automl_package/models/common/distilled_router.py:84`, `:108` | vector | cheapest-within-tolerance hard-label CE only |

The two vector ports were written independently and do not know about each other — the same wheel
invented twice. The package version is documented as a copy-not-import synthesis of the second vector
port's architecture and `sinc_width_experiment.py`'s labelling rule.

**The package router is behaviourally a SUBSET.** It has no soft-target training, no direct-objective
training, and **no blend-likelihood evaluation path at all** — it only hard-routes and scores the
routed choice. Several script capabilities have no package equivalent. This is a real capability gap,
not a rename: "adopt the package router everywhere" would silently drop working research machinery.

### 1.3 The certified results were not produced by the package code

- **Width.** The certified routing/selection numbers were produced by
  `automl_package/examples/sinc_width_experiment.py`'s selector machinery, run inside
  `kdropout_converged_width_experiment.py`. The package router has **never produced a certified
  result**; every document referencing it is a build spec or an unrun benchmark spec.
- **Width architecture.** The architecture of record, `SharedTrunkPerWidthHeadNet`, lives in
  `automl_package/examples/nested_width_net.py:222` — outside the package. The package's port
  (`automl_package/models/flexible_width_network.py`) reuses the same prefix-masking mechanism but
  **deviates from the certified training schedule**: its own docstring (`:15-16`) records that it is
  *"specialised here to sum ALL configured widths every step rather than a sampled subset"* of the
  certified sandwich schedule. Acknowledged, not hidden — but it means the shipping class is not
  running the certified training scheme. → **FP-4**
- **Depth.** The certifying code is a group word-problem toy
  (`automl_package/examples/depth_composition_toy.py`), structurally different from the package
  class's tabular forward pass. **The certified conclusion may transfer; the certifying code does
  not.** The package depth classes carry zero cross-reference to the depth certification.

### 1.4 Eleven capacity-dial classes; four width architectures live outside the package

Seven width-dial classes (four in `nested_width_net.py`, two superseded cascade/matryoshka variants,
one package port) and four depth-dial classes (two package, two-plus research composers).

### 1.5 Dead and trap code

- **`WidthSelectionMethod.DISTILLED` is a functional trap.** `automl_package/enums.py:109` documents
  it as "not yet landed"; constructing with it raises `NotImplementedError`
  (`automl_package/models/flexible_width_network.py:91-95`), and there is a **passing test asserting
  it raises** (`tests/test_flexible_width_network.py:200`). Meanwhile the real distilled routing
  works through an entirely separate path. The enum's docstring is false: the feature IS landed, just
  not through that member.
- **Two superseded width architectures** (`ResidualCascadeNet`, `MatryoshkaWidthNet`) remain, with
  live callers, from a programme the current certification superseded.

### 1.6 What the inventory did NOT establish — do not treat these as cleared

The survey was budget-bounded and says so. Carried here so no later reader mistakes silence for a
clean bill:
- The **zero-caller sweep is incomplete** (~100 example scripts; only the router-relevant ones plus
  the architecture files were swept). → **FP-7**
- A **sixth duplicated concept is suspected but uninventoried**: bootstrap / standard-error
  statistics helpers appearing across several scripts under similar names. → **FP-7**
- ~~whether the two superseded width architectures are still under a live pre-registration~~
  **RESOLVED 2026-07-20 at the root, before dispatch.**
  `docs/plans/width_dial_2026-07-11/cascade_execution_plan_2026-07-11.md` is banner-frozen
  (`:1` — *"⛔ FROZEN. NEVER DISPATCH FROM THIS FILE"*) and its own banner states the **width content
  is superseded** by `width-cert.md`, with the live plan of record being
  `docs/plans/capacity_programme/`. The only live reasoning it retains is ProbReg's.
  **⇒ No live pre-registration commits the programme to `ResidualCascadeNet` or
  `MatryoshkaWidthNet`.** They are NOT thereby deletable — see §3's *Superseded ≠ deletable* clause for their disposition.

---

## 2. The boundary rule (the thing this strand is really for)

**Proposed rule, to be ratified by FP-0 and binding thereafter:**

> `automl_package/models/` and `automl_package/utils/` contain **library code**: reusable
> architectures, selection mechanisms, the selection API, and accounting. They **never** import from
> `automl_package/examples/`.
>
> `automl_package/examples/` contains **experiment drivers**: protocols, preregistered batteries,
> toy generators, and result production. They may import freely from the package. An experiment
> driver may hold a local implementation ONLY when it encodes an experiment-specific protocol that
> the library does not and should not express — and it must say so, in a comment, naming what
> differs.

The dependency arrow points one way. Today it points both ways (§1.1), which is the defect.

**The precedent to copy already exists in this repo.** `automl_package/examples/convergence.py` is a
thin re-export shim over `automl_package/utils/convergence.py`, kept so existing scripts' imports
keep resolving while the logic lives in the package. That is exactly the migration shape the router
and the width architectures should follow — move the logic, leave the shim, do not rewrite callers.

---

## 3. 🚫 DO NOT DELETE — the protection manifest

A careless cleanup is the main risk this strand carries. The protected set is **not prose in this
section** — prose has to be interpreted, and an interpretation is exactly what deletes working
research code. It is a machine-checkable manifest:

> **`docs/plans/capacity_programme/shared/PROTECTED.tsv`**
> Tab-separated. Columns: `path` · `symbol-or-*` · `reason` · `certified-artifact-dir`.
> 17 rows, every path and symbol confirmed on disk 2026-07-20.

**Binding rule — applies to EVERY task in this file whose write set touches code.** Before landing,
check the diff against the manifest:

```bash
git diff --name-status | awk '$1=="D"||$1=="R"{print $2}' \
  | grep -F -x -f <(grep -v '^#' docs/plans/capacity_programme/shared/PROTECTED.tsv | cut -f1 | sort -u)
```

Empty output = clear. **Any output = STOP and escalate to the user** — do not reason about whether
this particular deletion is fine.

Two clarifications the manifest encodes and a reader must not lose:

- **A re-export shim is not a deletion.** Moving logic into the package and leaving the protected
  path as a thin re-export (the `automl_package/examples/convergence.py` precedent) is the sanctioned
  migration and passes the check above, because the path still exists.
- **Superseded ≠ deletable.** `cascade_width_net.py` / `matryoshka_width_net.py` /
  `cascade_width_experiment.py` are **superseded, retained, NOT migrated, NOT deleted**
  (settled 2026-07-20). Their pre-registration is frozen and its width content explicitly superseded
  (§1.6), so nothing commits us to developing them further — but they produced results that sit on
  disk (`W_CASCADE/`, `W_MRL/`) and deleting the code orphans those artifacts from their producer.
  They stay in the experiment-driver tree and are out of scope for the library migration. No further
  ruling needed.

**Amending the manifest.** Adding a row needs no ruling. **Removing** a row needs an explicit user
ruling recorded in the amending task. Only tasks in this file amend it.

---

## 3.5 Autonomous execution contract

| Branch | Pre-authorised default | Log |
|---|---|---|
| a move would break an import in a script that produced a certified result | **move the logic, leave a re-export shim** (the `automl_package/examples/convergence.py` precedent); never rewrite the calling script | the shim + which scripts depend on it |
| a script implementation has a capability the library lacks | port the capability, or record in writing that it is out of library scope; **never silently drop it** | which, and which way |
| a behavioural difference is found between two implementations | **stop and record it**; do not pick a winner by inspection | both behaviours |
| a name collides on merge | keep the library name; alias in the shim | the mapping |
| test coverage is absent for code being moved | write the characterisation test FIRST, against current behaviour, then move | the test |

**HALT and ask:**
1. Any **deletion** of a path listed in `PROTECTED.tsv` (§3), or of anything that has produced a certified
   number.
2. ~~Whether a live pre-registration still commits to the superseded width architectures~~ —
   **RESOLVED 2026-07-20 before dispatch (§1.6, §3 *Superseded ≠ deletable*). No longer a halt.**
3. Any change that would make a **certified result unreproducible** from its own artifacts.
4. Anything **irreversible or outward-facing** (deleting artifacts, publishing, committing).

---

## 4. Tasks

Order: **FP-0 → FP-1 → FP-9 → FP-10 → FP-2 → FP-3 → FP-4 → FP-5 → FP-6 → FP-7 → FP-8.**
FP-3 (the API) is what `width.md`, `depth-selection.md` and `probreg.md` block on; it is deliberately
early. **FP-9** (the shared selection primitives) sits right after FP-1 because its cost-accounting
piece needs the accounting module to have a stable home first — and because three sibling plans are
each about to build the same three things independently. **FP-10** (added 2026-07-21, promoted from
FP-3's carried follow-up) is placed first among the not-yet-dispatched tasks, per its own text — "do
it as the first act of the next package wave."

**AS-RUN ORDER, recorded 2026-07-21 (the line above is the declared/planned order, not what actually
happened).** `FP-0` and `FP-7` ran first, in wave 1, ahead of `FP-1` — out of the declared sequence and
with no `deps:` line requiring it. This is why FP-9.b's completion note (below) can correctly
attribute its bootstrap-helper count correction to FP-7 having already run, even though FP-9 is
declared 3rd and FP-7 9th on the line above, and why FP-9's own `Orchestration: deps: FP-1` line does
not mention FP-7: the dependency was real in practice but was never written back into either the Order
line or FP-9's `deps:`. Recorded so the attribution in FP-9.b is no longer temporally impossible
against this file's own declared Order.

### FP-0 — ratify the boundary rule; record the `flexnn-core.md` dispositions — ✅ **DONE 2026-07-20 (verify re-executed 2026-07-21, all four clauses pass)**

**Files (write set):** this file · `docs/plans/capacity_programme/shared/CORE-DISPOSITIONS.tsv` (new)
**🚫 NOT in the write set: `docs/plans/capacity_programme/flexnn-core.md` or
`docs/plans/capacity_programme/MASTER.md`.** The root owns both (MASTER naming key, "SHARED FILES
ARE ROOT-ONLY"). FP-0 **records** what must change there; the root **applies** it. Editing either
here is a single-writer violation — three sibling tasks (WSEL-0, DSEL-0, P0) need MASTER edits in
this same wave, and concurrent writers produce contradictory text in one file.
**Deliverable instead:** emit the exact MASTER text — the index entry and the boundary-rule Decision,
written out verbatim, ready to paste — in this task's report to the root.

**Spec:**
1. Ratify §2's boundary rule as a programme Decision in `MASTER.md`, and add this strand to the index.
2. Produce `shared/CORE-DISPOSITIONS.tsv` — **one row for every task heading in `flexnn-core.md`**,
   tab-separated, columns `task-id`, `disposition`, `target`, `note`, where `disposition` is exactly
   one of `move` / `supersede-with-pointer` / `retain`:
   - `move` — the task belongs to one of the four strand plans; `target` names the plan **and** the
     task ID it becomes there.
   - `supersede-with-pointer` — the work is done or replaced; `target` names the document that
     replaced it. The heading stays in `flexnn-core.md` carrying that pointer.
   - `retain` — the task stays live in `flexnn-core.md`; `target` is `-`.

**⚠️ The defect this replaces — read before touching the verify.** The previous version of this task
asserted that `flexnn-core.md` "retains only the MoE comparison and the unified report". **That is
false.** As of 2026-07-20 that file carries **15** task headings (`grep -c '^### Task F\|^### F'`),
including live ProbReg work (F9 port, F10 report, F12 benchmark), live depth work (F5, F5c), roadmap
work (F8, F11) and refactor debt (F13). A worker driving the old verify to green would have deleted
live tasks belonging to other strands. **Every one of the 15 gets a disposition line. Nothing is
removed on the strength of a count.**

**Non-goals:** no code. No deletions. No edits to `flexnn-core.md` or to any sibling strand plan.

*Orchestration:* parallel: no · deps: none · tier: main loop (definitional) · scale: static ·
shape: design ·
**verify** (all four must pass):
```bash
cd /home/ff235/dev/MLResearch/automl/docs/plans/capacity_programme
# (a) MASTER carries the index entry and the boundary-rule Decision
grep -n "flexnn-package.md" MASTER.md && grep -in "boundary rule" MASTER.md
# (b) exactly one disposition row per flexnn-core.md task heading
[ "$(grep -c '^### Task F\|^### F' flexnn-core.md)" = "$(grep -vc '^#' shared/CORE-DISPOSITIONS.tsv)" ] \
  && echo COUNT-OK || echo COUNT-FAIL
# (c) every task id in flexnn-core.md appears in column 1
grep -o '^### Task F[0-9a-z]*\|^### F[0-9a-z]*' flexnn-core.md | grep -o 'F[0-9a-z]*$' | sort -u \
  | while read t; do cut -f1 shared/CORE-DISPOSITIONS.tsv | grep -qx "$t" || echo "NO-DISPOSITION $t"; done
# (d) only the three legal dispositions appear
cut -f2 shared/CORE-DISPOSITIONS.tsv | grep -v '^#' | sort -u \
  | grep -vxE 'move|supersede-with-pointer|retain|disposition' && echo DISPOSITION-FAIL || echo DISPOSITION-OK
```
**Plus the write-set intersection check (the real gate).** For every `retain` row, the task's `Files`
/ write-set paths must not appear in any of `width.md`, `depth-selection.md`, `probreg.md`,
`flexnn-package.md`. A retained task sharing a write set with a strand-plan task is two writers on one
file — that is the failure mode this whole reorganisation exists to prevent:
```bash
cd /home/ff235/dev/MLResearch/automl/docs/plans/capacity_programme
grep -oE '`automl_package/[A-Za-z0-9_./-]+\.py`|`tests/[A-Za-z0-9_./-]+\.py`' flexnn-core.md \
  | tr -d '`' | sort -u > /tmp/core_ws.txt
grep -hoE '`automl_package/[A-Za-z0-9_./-]+\.py`|`tests/[A-Za-z0-9_./-]+\.py`' \
  width.md depth-selection.md probreg.md flexnn-package.md | tr -d '`' | sort -u > /tmp/strand_ws.txt
comm -12 /tmp/core_ws.txt /tmp/strand_ws.txt
```
Every path this prints must be attributable to a `move` or `supersede-with-pointer` row — i.e. it is
leaving `flexnn-core.md`. **A path printed here that belongs to a `retain` row fails the task**, and
FP-0 records the conflict for the root rather than resolving it by editing either file.

**⚠️ CAVEAT 2026-07-21 — this extraction grep is blind to citations carrying a `:line` suffix.** The
regex above matches a clean single-backtick span (`` `automl_package/...\.py` ``) but not
`` `automl_package/examples/moe_flexnn_comparison.py:413,414` `` — a citation form this plan itself
uses pervasively (e.g. §1.1's `flexible_width_network.py:290`). A real write-set collision on
`moe_flexnn_comparison.py`, between `flexnn-core.md` F6 and this file/`width.md`, was missed by this
grep and caught only by hand (`shared/CORE-DISPOSITIONS.tsv` "CONFLICT 1"). **The gate text must strip
the `:line` suffix from each extracted path before intersecting** — until it does, this check's empty
output is a floor, not a ceiling: it proves no *unsuffixed* citation collides, not that no citation
collides.

### FP-1 — break the circular dependency — ✅ **DONE 2026-07-20** (verify re-executed at the root)

Accounting now lives at `automl_package/utils/capacity_accounting.py`, which imports nothing from
`models/` or `examples/`; `automl_package/examples/capacity_accounting.py` is a re-export shim
(verified at the root by object **identity**, so it is provably not a fork), and the four `nwn.*`
branches stayed on the examples side exactly as ruled — the package module carries no
`nested_width_net` import or dispatch branch, only comments naming it. Root-executed: verify clauses
(a), (b), (c) and the `PROTECTED.tsv` deletion check all print nothing; the accounting selftest exits
0 with all 20 known answers unchanged; `tests/test_capacity_accounting.py` (new, 13 tests) passes;
bare-name `import capacity_accounting` from `examples/` still resolves, now onto the package module.
**Full-suite clause deferred to the wave-end run** (the environment rule: the root runs `pytest tests/`
once, at wave end, not while a heavy sibling worker is training).

**⚠️ THE REGISTRATION-HOOK CONTRACT CHANGED SHAPE — FP-2 READ THIS FIRST.** This task's ruling said
the shim would register the four `nwn.*` branches "through a registration hook the package module
exposes". **No such hook exists, deliberately, and FP-2 must not look for one.** A package module
that imports the model classes in order to register them centrally **recreates the very cycle FP-1
removes**, relocated one file over: `flexible_width_network.py → utils/capacity_accounting.py →
flexible_neural_network.py → utils/capacity_accounting.py` (partially initialised) → `ImportError`.
**The actual contract is `functools.singledispatch` itself:** `executed_flops` and `param_count` are
ordinary singledispatch functions exported from the package module, and any holder of a reference —
the shim, or a model file — calls `.register(SomeType)` on that same dispatcher object. Each model
file registers its own inner class from within itself, after the class is defined. ⇒ **FP-2's move is
mechanical:** relocate the four `@executed_flops.register(...)` blocks out of the shim to wherever the
classes land, then drop the shim to a pure re-export. No new API surface. *(Deviation raised by the
worker rather than absorbed silently; adopted at the root because the literal reading provably
fails.)*

*(Original spec retained below, unchanged.)*

### FP-1 — break the circular dependency (spec)

**Files (write set):** `automl_package/utils/` (new accounting module) ·
`automl_package/examples/capacity_accounting.py` (becomes a shim) ·
`automl_package/models/flexible_width_network.py` · `automl_package/models/flexible_neural_network.py`
**Spec:** Move the cost/FLOP accounting into the package and make the two deferred imports
(`flexible_width_network.py:290`, `flexible_neural_network.py:492`) ordinary top-level imports.
Leave `automl_package/examples/capacity_accounting.py` as a re-export shim so every existing script
keeps resolving unchanged (the `automl_package/examples/convergence.py` precedent).

**⚠️ The ordering knot — SETTLED HERE, not left to the worker.** `executed_flops` dispatches on four
width classes that live in `automl_package/examples/nested_width_net.py`, imported by bare name
(`import nested_width_net as nwn`): `NestedWidthNet`, `SharedTrunkPerWidthHeadNet`,
`IndependentWidthNet`, `SharedReadoutPerWidthAffineNet`. Those classes do not move until **FP-2**,
and FP-2 depends on FP-1. Circular.

**Ruling:** the **four `nwn.*` dispatch branches stay on the examples side, in the
`capacity_accounting.py` shim, until FP-2 lands.** The shim registers them into the package module
through a registration hook the package module exposes; the package module itself must import nothing
from `examples/`. **FP-2 then moves those four branches into the package** as its last step, and the
shim drops to a pure re-export. FP-2's completion criteria include that move — it is not optional
tidy-up. Any branch that dispatches only on package classes (`FlexibleHiddenLayersNN.FlexibleNNModule`,
`FlexibleWidthNN.FlexibleWidthNNModule`, and the two shape descriptors) moves in FP-1.

**Doctrine:** this is a move, not a rewrite. Behaviour must be identical; a characterisation test
against current outputs is written **before** the move.
**Non-goals:** do not add selection-cost accounting here — that is **FP-9**, and it lands after this
module has a stable home. Do not move the four `nwn.*` branches in this task.

*Orchestration:* parallel: no (touches both model files) · deps: FP-0 · tier: sonnet high ·
scale: static · shape: execution ·
**verify** (all must pass; the bare-name check is the one that catches the real violation — an
examples-name import has no `automl_package.` prefix, so a qualified-form grep alone reports clean
while the module still imports from `examples/`):
```bash
cd /home/ff235/dev/MLResearch/automl
# (a) qualified form, in BOTH the source and the destination directory
grep -rn 'automl_package\.examples' automl_package/models/ automl_package/utils/
# (b) bare-name form: any import of any module that exists as a file in examples/
ls automl_package/examples/*.py | xargs -n1 basename | sed 's/\.py$//' > /tmp/exmods.txt
grep -rnE "^[[:space:]]*(import|from)[[:space:]]+($(paste -sd'|' /tmp/exmods.txt))\b" \
  automl_package/models/ automl_package/utils/
# (c) no sys.path manipulation smuggling examples/ onto the path
grep -rn 'sys\.path' automl_package/models/ automl_package/utils/
# (d) behaviour identical + suite green
AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m automl_package.examples.capacity_accounting --selftest
AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m pytest tests/ -q
```
(a), (b) and (c) must each print **nothing**; the selftest must exit 0; the characterisation test must
produce byte-identical numbers before and after the move.

### FP-9 — the shared selection primitives (built ONCE, here) ⭐ — ✅ **DONE 2026-07-20** (verify re-executed at the root)

**The contract every consumer imports** (WSEL-3, WSEL-5, DSEL-6, PA, FP-4 — do not reimplement):
- `bootstrap_se(values, n_boot, seed) -> float` — `automl_package/utils/numerics.py`. Covers the
  plain AND paired shapes: they are the same operation, the paired one applied to a difference
  vector. `n_boot`/`seed` are REQUIRED, no defaults.
- `two_sample_bootstrap_se(a, b, n_boot, seed) -> float` — same file.
- `cheapest_within_tolerance(error_table, n_boot=1000, seed=0) -> int` —
  `automl_package/utils/capacity_selection.py`. Compares every candidate against the **global best**
  (FP-9.a's literal wording), not §4.1's sequential-staircase framing; both agree on all three
  plan-given known answers, and the docstring records the choice.
- `router_fit_cost(in_dim, n_capacities, n_samples, n_epochs, hidden)` / `held_out_read_cost(net,
  capacity_grid, n_samples)` / `sweep_cost(net, capacity_grid, n_train_samples, n_epochs)` —
  `automl_package/utils/capacity_accounting.py`. **`hidden` is REQUIRED**, deliberately: defaulting
  it to `distilled_router.DEFAULT_HIDDEN` would silently drift when FP-5 changes that shared global.

Root-executed: the plan's six-known-answer heredoc prints `FP9-KNOWN-ANSWERS-OK`; the accounting
selftest exits 0 **having gained all three selection-cost cases** (`grep -c 'selection'` on the shim
4 → 5); `tests/test_capacity_selection.py` = 18 passed; ruff clean on all four touched files (the 49
findings under `automl_package/utils/` are pre-existing debt in files this task never opened).

**FINDING — the paired-shape duplication is a RENAME, not a drift, and this closes a live worry.**
The plan flagged the three independent paired-bootstrap implementations as "where genuine drift
between the three strands' numbers could already be hiding". Re-read at the root, all three bodies
are **character-for-character the same algorithm** (`rng.integers(0, n, size=(n_boot, n))` →
`diff[idx].mean(axis=1)` → `.std(ddof=1)`), differing only in the parameter name and whether
`n_boot`/`seed` carry defaults: `automl_package/examples/capacity_ladder_s1.py:258`,
`automl_package/examples/capacity_ladder_k4.py:76`,
`automl_package/examples/capacity_ladder_t2.py:337`. ⇒ **No cross-strand numeric drift exists or
ever existed here**, and consolidating any caller onto `bootstrap_se` is behaviour-preserving.

**Process note — FP-9.c was applied BY THE ROOT, not by the worker.** `write_set_guard` blocked the
worker from `automl_package/utils/capacity_accounting.py` (FP-1's agent wrote it earlier in the same
session; the guard is session-scoped, and it was right). The worker drafted, the root reviewed and
applied — the sanctioned handoff. The `hidden`-required change above came out of that review.

*(Original spec retained below, unchanged.)*

### FP-9 — the shared selection primitives (spec)

**Why this task exists.** Three sibling plans each independently specify the same three pieces of
machinery. Built three times, they will differ — and the moment they differ, the three strands' numbers
stop being comparable, which is the exact property `width.md` WSEL-3, `depth-selection.md` DSEL-6 and
`probreg.md` PA all explicitly promise ("the tolerance rule is imported unchanged so the numbers stay
comparable... **do not re-derive it**"). Three plans say "do not re-derive it" and none of them builds
it. FP-9 builds it.

**Files (write set):** `automl_package/utils/numerics.py` (extend) ·
`automl_package/utils/capacity_selection.py` (new) · the FP-1 accounting module ·
`tests/test_capacity_selection.py` (new)

#### FP-9.a — the cheapest-within-tolerance selector over an error CURVE

One capacity value for the whole dataset: given a held-out error **curve** over the capacity grid,
return the **smallest** capacity whose error is not meaningfully worse than the best capacity's, where
"meaningfully worse" means **the difference exceeds twice a bootstrap-estimated standard error of that
difference**.

**🚫 This is NOT `_cheapest_within_tolerance_labels`, and it must not be built by generalising it.**
Read `automl_package/models/common/distilled_router.py:63` first. That function is a **per-row**
labeller for the per-input router: it takes an `(n_samples, n_capacities)` error *table* and applies a
**fixed relative** tolerance (`error <= (1 + tolerance) * row_min`, `DEFAULT_TOLERANCE = 0.25`). It
answers "which capacity for *this input*", with a hand-set margin.

FP-9.a answers a different question — "which single capacity for *this dataset*" — with a
**noise-calibrated** margin derived from the data. Different input shape, different tolerance
semantics, different consumer. **Both exist. Neither replaces the other.** Say so in the docstring, so
the next reader does not "de-duplicate" them.

Rule of record: `docs/reports/probreg_kselection/probreg_kselection.md` §3.2 — a difference counts as
real only if it exceeds twice an estimated standard error. **Import that rule; do not re-derive it.**

#### FP-9.b — the bootstrap standard-error helper it needs

**⚠️ COUNT CORRECTED 2026-07-20 by FP-7, which re-derived it as instructed rather than trusting the
plan.** This section previously asserted **fifteen** helpers "under at least five different names".
**The re-derived count is TWELVE.** FP-7 searched three independent ways — exact-name grep, a
broadened name-pattern grep, and a *name-agnostic* `rng.integers(0` sweep across all of
`examples/` — and could not find the missing three by any of them. (The name-agnostic sweep did
surface three further files using the resample idiom; all three were read and are unrelated
data-generation code — mixture assignment, pool sampling, word-sequence generation — not SE
helpers.) **Zero exist in `automl_package/models/` or `automl_package/utils/`**, which the
re-derivation confirms.

Build **one**, in `automl_package/utils/numerics.py` (existing home for numerical utilities — reuse
the module, do not create a parallel one). It must cover the three shapes the drivers actually use.
**FP-7's inventory (`shared/zero-caller-inventory.md`) lists every site with `file:line` and
signature, grouped by these same three shapes — read it rather than re-deriving.** The three shapes
are NOT equally duplicated, and that changes how each is consolidated:

1. **plain** — SE of a 1-D vector's mean. **7 sites** (`_boot_se` ×3, `_plain_boot_se` ×4).
   **All 7 are identical 3-line wrappers around the SAME shared primitive**,
   `automl_package/examples/_capacity_ladder.py:196` `_bootstrap_col_means`. ⇒ **This is the safest
   consolidation target: a pure rename, not a behavioural merge.** There is already one
   implementation here, only seven names for it.
2. **paired** — SE of a paired difference vector's mean. **3 sites** (`_paired_bootstrap_se`,
   `paired_bootstrap_se`, `_paired_point_bootstrap_se`). ⚠️ **Unlike the plain shape, none of
   these calls the shared primitive — each independently reimplements the same 4-line resample
   loop under a different name.** This is real duplication, and it is where genuine drift between
   the three strands' numbers could already be hiding. Diff the three before replacing them.
3. **two-sample** — SE of `mean(a) - mean(b)` for independent samples. **Exactly ONE site**,
   `automl_package/examples/sinc_width_experiment.py:188` `_two_sample_boot_se` — *not* "≥5 names"
   as previously implied. It is **non-vectorised** (a Python loop) where the other two shapes are
   vectorised; match its semantics, not its performance profile.

**Also flagged by FP-7 and deliberately NOT counted in the 12** — do not sweep these up: 
`_tercile_means_se` (an *analytic* SE, a different technique, not a bootstrap), and
`gold_mid_with_se` / `_paired_bootstrap_check`, which are domain consumers that call the shape-1
primitives internally rather than being primitives themselves.

Explicit `seed` and `n_boot` parameters — a selection rule whose answer moves between runs is not a
rule. FP-9 **adds** the shared helper; it does **not** rewrite the fifteen callers (that is a follow-on
under §3's protections, and several of those files are protected).

#### FP-9.c — selection-cost accounting

The accounting module prices a network at a given capacity. It has **no notion of the cost of
*choosing* that capacity** — for any family. That missing denominator is what makes every efficiency
claim in this programme unfalsifiable, because "cheaper at inference" is worthless if selection cost
was never counted.

Add, in the FP-1 accounting module, cost accounting for the three ways a capacity gets chosen:
1. **router fit** — training the distilled router (its own forward/backward cost × epochs);
2. **cheap held-out read** — scoring every capacity once on the selection set;
3. **sweep** — training the model once per capacity value.

Analytic, matching the module's existing design (it is analytic by design — no wall-clock harness).
Family-agnostic: width, depth, ProbReg-k and MoE all draw from it.

**Consumers — these tasks CONSUME FP-9 and must not reimplement it:**

| consumer | plan | consumes |
|---|---|---|
| **WSEL-3** | `width.md` | FP-9.a + FP-9.b (the cheap global read for width) |
| **WSEL-5** | `width.md` | FP-9.c (charging the cost of selection) |
| **DSEL-6** | `depth-selection.md` | FP-9.a + FP-9.b (the cheap global read for depth) |
| **PA** | `probreg.md` | FP-9.a + FP-9.b (M1's selection rule) |
| **FP-4** | this file | FP-9.b (the paired-difference SE in its verdict bar) |

**Non-goals:** no per-input logic (that is the router); no rewriting of the fifteen existing bootstrap
helpers; no new selection *algorithms* beyond the one rule above; no wall-clock benchmarking.

*Orchestration:* parallel: no (the accounting module is single-writer) · deps: FP-1 · tier: sonnet high ·
scale: static · shape: execution ·

**verify** (runnable, with hand-computed known answers):
```bash
cd /home/ff235/dev/MLResearch/automl
AUTOML_DEVICE=cpu ~/dev/.venv/bin/python - <<'PY'
import numpy as np
from automl_package.utils.capacity_selection import cheapest_within_tolerance
from automl_package.utils.numerics import bootstrap_se

# --- KNOWN ANSWER 1: a curve flat beyond index 3 must select 3, NOT the argmin ---
# per-capacity held-out errors, 5 replicates each; capacities 0..5
rng = np.random.default_rng(0)
base = np.array([10.0, 5.0, 2.0, 1.0, 1.0, 1.0])
errs = base[None, :] + rng.normal(0, 0.01, size=(200, 6))
assert cheapest_within_tolerance(errs, seed=0) == 3, cheapest_within_tolerance(errs, seed=0)

# --- KNOWN ANSWER 2: a strictly decreasing curve well outside noise selects the LAST ---
base2 = np.array([10.0, 8.0, 6.0, 4.0, 2.0, 1.0])
errs2 = base2[None, :] + rng.normal(0, 0.01, size=(200, 6))
assert cheapest_within_tolerance(errs2, seed=0) == 5

# --- KNOWN ANSWER 3: a curve that is pure noise selects index 0 (nothing beats the cheapest) ---
errs3 = rng.normal(1.0, 0.05, size=(200, 6))
assert cheapest_within_tolerance(errs3, seed=0) == 0

# --- KNOWN ANSWER 4: bootstrap SE of the mean ~ sd/sqrt(n), within 10% at n=2000 ---
v = rng.normal(0.0, 1.0, size=2000)
se = bootstrap_se(v, n_boot=2000, seed=0)
assert abs(se - 1.0/np.sqrt(2000)) / (1.0/np.sqrt(2000)) < 0.10, se

# --- KNOWN ANSWER 5: determinism -- same seed, same answer ---
assert bootstrap_se(v, n_boot=500, seed=7) == bootstrap_se(v, n_boot=500, seed=7)

# --- KNOWN ANSWER 6: paired SE of a CONSTANT difference vector is exactly 0.0 ---
assert bootstrap_se(np.full(100, 3.0), n_boot=200, seed=0) == 0.0
print("FP9-KNOWN-ANSWERS-OK")
PY
# selection-cost accounting: every mechanism returns a finite, positive cost
AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m automl_package.examples.capacity_accounting --selftest
AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m pytest tests/test_capacity_selection.py -q
AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m pytest tests/ -q
```
The heredoc must print `FP9-KNOWN-ANSWERS-OK`; the accounting selftest must exit 0 **and must have
gained a selection-cost case** (`grep -c 'selection' automl_package/examples/capacity_accounting.py`
increases); both pytest runs green.

*(Function and parameter names above are the contract — if the task changes one, it changes it in this
plan first, because three sibling plans import them.)*

### FP-2 — bring the certified width architectures into the package

**Files (write set):** `automl_package/models/architectures/` · `automl_package/examples/nested_width_net.py`
(becomes a shim) · call sites
**Spec:** Move the four width-dial architectures — including `SharedTrunkPerWidthHeadNet`, the
architecture of record — from `automl_package/examples/nested_width_net.py` into the package's
architectures package. Leave a re-export shim; **do not rewrite the calling scripts** (§3 item 5:
their preregistration documents cite exact names).
**Doctrine:** the architecture of record living outside the shipped package is the literal form of
the user's complaint. Moving it is the mandate.
**Also required (carried from FP-1's ruling):** as its last step, FP-2 moves the four `nwn.*` dispatch
branches in `capacity_accounting.py` into the package alongside the classes, dropping the shim to a
pure re-export. This is a completion criterion, not optional tidy-up.

**Non-goals:** no behaviour change; no merging of the four classes; no touching the toy generators
(`make_hetero` and friends stay with the experiment drivers — they are protocol, not library).

*Orchestration:* parallel: no · deps: FP-1 · tier: sonnet high · scale: static · shape: execution ·

**REFERENCE NUMBERS — named, not left to the worker.**
Document of record: `docs/width_mse_2026-07-16/verdict_variable_width_mse.md` §10.2 (five-seed
headline). Cell: toy `hetero`, `n_train=1500`, `sigma=0.05`, `w_max=12`, arch `shared_trunk`,
**seeds 0–4** (five seeds). Reference JSONs, all under
`automl_package/examples/capacity_ladder_results/W_KDROPOUT_CONVERGED/`:
- seeds 0/1/2 — `w_kdropout_converged_summary_shared_trunk_mse.json`
- seed 3 — `w_kdropout_converged_summary_shared_trunk_mse_n1500_s0.05_h5s3.json`
- seed 4 — `w_kdropout_converged_summary_shared_trunk_mse_n1500_s0.05_h5s4.json`

Metric: `per_case[i].fit_bar.ratio_to_floor` (seed 0 = 1.0892838787488606; §10.2 quotes
1.089 / 1.061 / 1.077 / 1.227 / 1.061). **Read the number from the JSON, never transcribe it from the
markdown.**

**TOLERANCE: ≤ 2% relative error on `ratio_to_floor`, per seed, all five seeds.**
*(2% is a chosen default, not a derived bound — the user may retune it. It is written down so the
worker is judged against a fixed bar instead of choosing its own after seeing the result.)*

**verify** (runnable; every clause has an observable pass/fail):
```bash
cd /home/ff235/dev/MLResearch/automl
# (a0) wiring check first -- seconds, catches a broken move before any long run
AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m automl_package.examples.kdropout_converged_width_experiment --selftest
AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m automl_package.examples.kdropout_converged_width_experiment \
  --smoke --arch shared_trunk --loss mse
# (a) reproduce through the MOVED classes, 5 seeds, same cell; the driver writes a tagged JSON.
#     No --seeds flag exists: the bare run does seeds 0/1/2, `--config N` does one seed.
#     LONG RUN (convergence-gated, --max-epochs cap 300000). Run each seed separately and land
#     its JSON before starting the next -- never one 5-seed foreground call.
AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m automl_package.examples.kdropout_converged_width_experiment \
  --arch shared_trunk --loss mse --toy hetero --n-train 1500 --sigma 0.05 --tag fp2repro
for s in 3 4; do AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m automl_package.examples.kdropout_converged_width_experiment \
  --arch shared_trunk --loss mse --toy hetero --n-train 1500 --sigma 0.05 --config $s --tag fp2repro_s$s; done
# (b) mechanical comparison against the three reference JSONs -- prints PASS/FAIL per seed
#     (the task writes this comparison script under the scratchpad, not the repo)
# (c) every existing script still imports unchanged
grep -rlE '^[[:space:]]*import nested_width_net' automl_package/examples/*.py \
  | while read f; do AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -c "
import sys; sys.path.insert(0,'automl_package/examples'); import importlib
importlib.import_module('$(basename ${f%.py})')" || echo "IMPORT-BROKE $f"; done
# (d) protection check + suite
git diff --name-status | awk '$1=="D"||$1=="R"{print $2}' \
  | grep -F -x -f <(grep -v '^#' docs/plans/capacity_programme/shared/PROTECTED.tsv | cut -f1 | sort -u)
AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m pytest tests/ -q
```
(b) must print PASS for all five seeds; (c) and (d)'s grep must print nothing; the suite must be green.
**Any seed outside 2% halts the task** — record both numbers and escalate; do not widen the bar.

### FP-3 — the one selection API ⭐ — ✅ **LANDED 2026-07-20**, one follow-up carried (below)

`CapacitySelection(FIXED, PER_INPUT)` ships in `automl_package/enums.py`; `WidthSelectionMethod` is
retired **entirely** (grep-confirmed: zero references anywhere in `automl_package/` or `tests/`), and
with it the `NotImplementedError` trap this task existed to kill. `inference_mode` is gone from
`predict` and from **both** `predict_uncertainty` methods; passing it to a `predict`, a
`predict_uncertainty` or a **constructor** now raises `TypeError` instead of being swallowed into
`self.params`. Root-executed verify: clause (b) `ENUM-OK`, clause (c) `NO-TRAP-OK` for every family ×
every member, clause (d) clean (the sole surviving `DISTILLED` string is prose about Decision 13).

**Clause (a) as written in this plan is unsatisfiable and should not be "fixed" by deleting tests.**
It greps for zero occurrences of `inference_mode` — but this task's own required tests #2 and #3
assert that passing `inference_mode` RAISES, so the string must appear in the rejection code and in
the tests that prove it. Read clause (a) as *no live call site passes it*; that is what was checked.

**`independent_weights_flexible_neural_network.py` was correctly NOT touched** — root-verified: that
class has no `inference_mode`, no `fit_router`, no selection surface at all. Its parity work is FP-6.
Its absence from the FP-3 diff is not a missed file.

**⚠️ FOLLOW-UP CARRIED — a fragile workaround landed and should be replaced.** Promoted to a numbered
task, **FP-10**, 2026-07-21 (placed after FP-3's spec section, below) — see its section for the full
carried content, the 2026-07-21 runtime confirmation, and the orchestration line.

*(Original spec retained below, unchanged.)*

### FP-3 — the one selection API (spec)

**Files (write set):** `automl_package/enums.py` · `automl_package/models/flexible_width_network.py` ·
`automl_package/models/flexible_neural_network.py` ·
`automl_package/models/independent_weights_flexible_neural_network.py` ·
`automl_package/models/probabilistic_regression.py` · tests · call sites found by grep
**Everything below is SETTLED. The worker implements it; it decides none of it.**

#### FP-3.a — the enum

**Name:** `CapacitySelection`. **Home:** `automl_package/enums.py`. Owned by this task; every capacity
family imports it.

**Members shipped by FP-3 — exactly two:**

| member | meaning | mechanism today |
|---|---|---|
| `FIXED` | no selection is performed; the caller supplies the capacity value | **this is today's default behaviour in all three families** |
| `PER_INPUT` | a distilled router chooses per input | `DistilledCapacityRouter` — the only selection mechanism that exists |

**🚫 `GLOBAL_CHEAP` and `GLOBAL_SWEEP` are NOT shipped by FP-3.** Each is added by the task that
**builds its mechanism** — `GLOBAL_CHEAP` by `width.md` WSEL-3 / `depth-selection.md` DSEL-6 /
`probreg.md` PA's M1 piece, `GLOBAL_SWEEP` by WSEL-4 / DSEL-7 / PA's M3 piece — in the same change
that makes it work.

**Why this is a rule and not a preference.** Shipping an enum member whose mechanism raises
`NotImplementedError` is **exactly the trap FP-3 exists to retire**: `WidthSelectionMethod.DISTILLED`
is documented in `automl_package/enums.py:109` as "not yet landed", raises on construction
(`automl_package/models/flexible_width_network.py:91-95`), and has a passing test asserting it raises
(`tests/test_flexible_width_network.py:200`) — a named, discoverable, permanently broken option.
Recreating that shape under a new name is a task failure. **An enum member is a promise that the
mechanism works.**

#### FP-3.b — what is removed

1. **`predict` loses `inference_mode` entirely** in all three families
   (`flexible_width_network.py:192`, `flexible_neural_network.py:386`,
   `probabilistic_regression.py:598`). Clean break, no shim — the repo has no external users, and a
   shim keeps the silent-failure route alive until it is used by accident.
2. **`predict_uncertainty` loses `inference_mode` too, wherever it has it.** ⚠️ **COUNT CORRECTED
   2026-07-20 by the FP-3 worker, which re-derived it instead of trusting this plan — it is TWO
   methods, not one:** `automl_package/models/probabilistic_regression.py:704` **and**
   `automl_package/models/flexible_neural_network.py:500`. *(The previous text asserted "exactly one
   method, verified 2026-07-20" and named a line number that has since moved. Re-verified at the root
   by `grep -rn "def predict_uncertainty" automl_package/models/`: every OTHER `predict_uncertainty`
   in the repo does have the clean `(x, filter_data)` signature, so that half stood — but a worker
   driving the old text would have fixed one family and left the other's silent-failure route open.
   **Second failed re-verification of a "verified" plan claim in one session; keep re-deriving.**)*
   Missing either is the silent-failure route the strand exists to close (`probreg.md` PA: "forget
   that flag and you silently get the un-routed model").
3. **`WidthSelectionMethod.DISTILLED`'s `NotImplementedError` trap, and the test asserting it raises.**
   The feature is landed; it is reached through an entirely separate path. The enum must name the
   working mechanism, not a dead one.
4. **`FlexibleWidthNN.predict`'s `width=None` silently defaulting to the largest configured width.**
   Under `FIXED`, `width=None` must raise, not guess.

#### FP-3.c — what SURVIVES, and must not be swept up

**`FlexibleHiddenLayersNN`'s `"hard"` mode is NOT a selection mode.** Its closed set today is
`("soft", "hard", "routed")` (`automl_package/models/flexible_neural_network.py:397`). `"soft"` and
`"routed"` are ways of *choosing* a depth; `"hard"` is an **execution shortcut** — it runs only the
argmax-selected depth per sample instead of the full forward pass, for compute savings. Different
axis, different concept.

**Ruling:** `"hard"` survives as a **separate boolean constructor/predict argument** (name it in the
task; `hard_execution` or equivalent), orthogonal to `CapacitySelection`. It does not become an enum
member and it is not deleted. Live callers exist and must keep working:
`automl_package/examples/phase4_comparison.py:164`, `:168`; `tests/test_phase4_regression.py:254`,
`:280`.

#### FP-3.d — removal must FAIL LOUDLY

**`BaseModel.__init__` swallows unknown keyword arguments into `self.params`**
(`automl_package/models/base.py:45`, `:52` — `self.params = kwargs`). So a caller who passes a
removed or misspelled selection kwarg **at construction** gets no error and silently un-selected
behaviour — the same failure class this task exists to close.

**Required:** a caller passing `inference_mode=` to `predict`, to `predict_uncertainty`, or to any of
the three constructors must raise **`TypeError`**. `predict`/`predict_uncertainty` get this for free
once the parameter is deleted (no `**kwargs` in those signatures — verified). The **constructor** path
does not: FP-3 must add an explicit rejection so the swallow cannot happen. **A test per family
asserts the `TypeError`** — not just that the parameter is gone.

#### FP-3.e — the selection-set fraction

**The selection-set fraction must be a constructor parameter**, not a baked-in constant.
**Three** studies are about to measure it (an earlier draft said two): `width.md` **WSEL-6**,
`depth-selection.md` **DSEL-8**, `probreg.md` **PB**. All three sweep `{5, 10, 15, 25, 40}%` of the
training portion and all three need the same knob.

#### FP-3.f — the factual record, corrected

An earlier draft of this task said the three families have "three different closed sets and three
different defaults". The closed-set half is right; the defaults half was wrong. Verified 2026-07-20:

| class | closed set | default |
|---|---|---|
| `FlexibleWidthNN.predict` | `("fixed", "routed")` | `"fixed"` |
| `FlexibleHiddenLayersNN.predict` | `("soft", "hard", "routed")` | `"soft"` |
| `ProbabilisticRegressionModel.predict` | `("soft", "routed")` | `"soft"` |

**Three different closed sets; TWO distinct defaults** (`"fixed"`, `"soft"`, `"soft"`). There is no
`InferenceMode` type anywhere in the repo. The argument does not weaken: a caller who has learned one
class's contract still cannot safely guess another's, and the two families that share a default word
do not share a member set.

**Doctrine:** this is the repo's own closed-set rule (`CLAUDE.md`: enums, never magic strings). It is
also why this task lives here and not in the three consuming strands: fixing it three times
independently is how you get three APIs — and `width.md` WSEL-2 and `probreg.md` PA had each already
drafted their own enum before this file existed.

**Non-goals:** no new selection *algorithms*; no router internals; building the cheap-global and sweep
mechanisms is the consuming strands' work — this task defines how they are reached. Do **not** add
their enum members here.

*Orchestration:* parallel: no · deps: FP-1 · tier: **opus xhigh** (main loop; the API design half) then sonnet (mechanical
migration) · scale: static · shape: design → execution ·

**verify** (runnable; each clause has an observable pass/fail):
```bash
cd /home/ff235/dev/MLResearch/automl
# (a) the string kwarg is gone everywhere -- library, example drivers, and tests. The example
#     call sites MUST be migrated in this task or they break at runtime:
#       automl_package/examples/phase4_comparison.py:159,164,167,168  (159/167 "soft", 164/168 "hard")
#       automl_package/examples/moe_flexnn_comparison.py:413,414      ("routed")
#       tests/test_phase4_regression.py:253,254,280
grep -rn "inference_mode" automl_package/ tests/
# (b) the enum exists with EXACTLY the two shipped members
AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -c "
from automl_package.enums import CapacitySelection
got = sorted(m.name for m in CapacitySelection)
assert got == ['FIXED','PER_INPUT'], got
print('ENUM-OK', got)"
# (c) NO shipped member is a trap: every family x every member must CONSTRUCT without
#     NotImplementedError. (A textual grep is wrong here -- probabilistic_regression.py raises
#     NotImplementedError legitimately for unsupported loss/head configurations.)
AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -c "
from automl_package.enums import CapacitySelection
from automl_package.models.flexible_width_network import FlexibleWidthNN
from automl_package.models.flexible_neural_network import FlexibleHiddenLayersNN
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel
for cls in (FlexibleWidthNN, FlexibleHiddenLayersNN, ProbabilisticRegressionModel):
    for m in CapacitySelection:
        try:
            cls(input_size=1, output_size=1, capacity_selection=m)
        except NotImplementedError as e:
            raise SystemExit(f'TRAP {cls.__name__}.{m.name}: {e}')
        except TypeError:
            pass  # constructor signature differs per family; adapt the kwargs, keep the assertion
print('NO-TRAP-OK')"
# (d) the retired trap is gone from the enum and from the test suite
grep -rn "DISTILLED" automl_package/enums.py tests/
# (e) the "hard" execution shortcut still works for its live callers
AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m pytest tests/test_phase4_regression.py -q
# (f) whole suite
AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m pytest tests/ -q
```
(a) and (d) must each print **nothing**; (b) must print `ENUM-OK`; (c) must print `NO-TRAP-OK`;
(e) and (f) green.

**Tests this task must ADD** (each named here so "a test exists" is checkable, not asserted):
1. per family — a router-fitted model constructed with `CapacitySelection.PER_INPUT` routes with **no
   caller flag** on `predict`;
2. per family — `predict(x, inference_mode="routed")` raises `TypeError`;
3. per family — the **constructor** given `inference_mode=` raises `TypeError` (not swallowed into
   `self.params`);
4. `FlexibleWidthNN` under `FIXED` with `width=None` raises rather than defaulting to the largest
   configured width;
5. `FlexibleHiddenLayersNN`'s hard-execution shortcut is reachable through its new boolean argument
   and produces the same predictions it produces today.

### FP-10 — replace the residual-std workaround; close FP-3.b.4's blast radius

**Carried from FP-3's completion note (2026-07-20); promoted to a numbered task 2026-07-21.**

**Files (write set):** `automl_package/models/base_pytorch.py` ·
`automl_package/models/flexible_width_network.py` · `automl_package/models/base.py` ·
`automl_package/utils/data_handler.py` · tests

**Spec — the fragile workaround.** `PyTorchModelBase._fit_single`
(`automl_package/models/base_pytorch.py:217-219`) computes the CONSTANT-uncertainty residual std by
calling `self.predict(x_train, filter_data=False)` with **no width** — which now raises under `FIXED`
(FP-3.b.4), so every `.fit()` under the default uncertainty method would crash. That base file was
outside FP-3's write set, so the FP-3 worker worked around it inside `FlexibleWidthNN._fit_single`: it
temporarily reassigns `self.uncertainty_method`, wraps the super call in `try/finally`, and recovers
`x_train`/`y_train` via `inspect.signature(...).bind(...)`. It is documented and it works, but it is
brittle — a signature change to the base breaks it at runtime, and a method that lies about its own
configuration mid-fit is the shape of defect this strand exists to remove.

**The clean fix, verified as available at the root:** `uncertainty_method` is read exactly ONCE in
`_fit_single`, in a three-line block at the very end. Extract those three lines into an overridable
`_fit_residual_std(self, x_train, y_train)` on `PyTorchModelBase`, and have `FlexibleWidthNN` override
it to predict at `max(self.widths)`. That deletes the reassignment, the `try/finally` and the
`inspect` binding.

**⛔ AND THE SAME DEFECT IS STILL LIVE, DORMANT, IN FIVE MORE PLACES — FP-3.b.4's blast radius was
never scoped.** Found by the FP-3 worker and **re-verified at the root by grep**: generic machinery
calls `predict(x)` polymorphically, with no width, at `automl_package/models/base.py:353` and `:444`
(CV folds), `:372` (the HPO objective), `:513` (evaluation), and
`automl_package/utils/data_handler.py:102` (the log-scale check). Under `CapacitySelection.FIXED`
every one of them now **raises** for `FlexibleWidthNN`. **Dormant only because no test exercises
`FlexibleWidthNN` with `optimize_hyperparameters=True` or `cv_folds`** — the suite is green and the
breakage is real. ⇒ **`FlexibleWidthNN` currently cannot be used with HPO or cross-validation at
all.** The ruling itself stays (silently defaulting to the largest width is exactly the
silent-failure class this strand exists to remove); what is missing is an internal, non-caller-facing
prediction path for bookkeeping and scoring — the same shape as the residual-std fix above, applied
once, centrally. **Schedule with that fix; they are one task.**
*(Case law: a settled sub-decision changed a method's contract for every polymorphic caller in the
repo, and the plan scoped it as a signature change. Grep the generic callers before ruling that a
widely-called method may start raising.)*

**✅ CONFIRMED AT RUNTIME 2026-07-21.** Constructing
`FlexibleWidthNN(capacity_selection=FIXED, cv_folds=3, early_stopping_rounds=1,
validation_fraction=None).fit(x, y)` crashes with `ValueError: width must be specified under
CapacitySelection.FIXED (no implicit default to the largest configured width)`, raised from inside
`_find_optimal_iterations_with_cv`; no test exercises `cv_folds` or `optimize_hyperparameters` for
this class. This is the plan's own named fix, not a new investigation: extract the three-line
residual-std block into an overridable `_fit_residual_std(self, x_train, y_train)` on
`PyTorchModelBase`; `FlexibleWidthNN` overrides it to predict at `max(self.widths)`; add an internal
non-caller-facing prediction path for the five bookkeeping call sites (`base.py:353`, `:372`, `:444`,
`:513`; `automl_package/utils/data_handler.py:102`); delete the inspect-based workaround in
`FlexibleWidthNN._fit_single`; regression tests covering fit-under-CV and fit-under-HPO for
`FlexibleWidthNN`.

**Non-goals:** no other change to `_fit_single` or to the uncertainty methods; no change to the
`CapacitySelection.FIXED` ruling itself (silently defaulting to the largest width stays rejected).

*Orchestration:* parallel: yes · deps: none · tier: sonnet high ·
**verify:** new tests green, then revert the base-class extraction, show the CV test FAIL, restore;
`grep -n "inspect.signature" automl_package/models/flexible_width_network.py` returns nothing.

### FP-4 — resolve the package width class's schedule deviation

**Files (write set):** `automl_package/models/flexible_width_network.py` ·
`docs/plans/capacity_programme/shared/fp4-schedule-deviation.md` (new)
**Spec:** The shipping width class sums **all** configured widths every step; the certified schedule
samples a subset (`automl_package/models/flexible_width_network.py:15-16`, self-documented).
Establish whether this changes results. Either bring the class onto the certified schedule, or
measure and document that the deviation is immaterial — **with numbers, not an argument.**
**Doctrine:** "acknowledged in a docstring" is not "shown not to matter." The shipped class should
run the certified training scheme or carry evidence that it need not.
**Non-goals:** do not re-open `G-WIDTH = PASS`; this is about the package port, not the certification.

*Orchestration:* parallel: yes (disjoint from FP-5) · deps: FP-2 · tier: sonnet high · scale: static ·
shape: research ·

**THE COMPARISON — cells, seeds, metric and bar, all named here.**
- **Cell:** toy `hetero`, `n_train=1500`, `sigma=0.05`, `w_max=12`, arch `shared_trunk`.
- **Seeds:** 0, 1, 2 (**three**), the same seeds the reference JSON carries — matched, not fresh.
- **Arms:** (A) certified sandwich schedule; (B) the package class's sum-all-widths schedule.
  Same seeds, same data, same convergence gate, same epoch cap — the schedule is the ONLY difference.
- **Primary metric:** `per_case[i].fit_bar.ratio_to_floor`.
  **Secondary:** `per_case[i].deploy_bar.mean_executed_width` and `.mse_hardpick`.
- **Reference numbers:** `docs/width_mse_2026-07-16/verdict_variable_width_mse.md` §10.2, read from
  `automl_package/examples/capacity_ladder_results/W_KDROPOUT_CONVERGED/w_kdropout_converged_summary_shared_trunk_mse.json`
  (seeds 0/1/2; seed 0's `ratio_to_floor` = 1.0892838787488606). **Read the JSON, never the markdown.**
- **VERDICT BAR — decided in advance, not after seeing the numbers:** the deviation is **IMMATERIAL**
  iff arm B's `ratio_to_floor` is within **2% relative** of arm A's on **all three** seeds **and** the
  paired mean difference is smaller than twice its bootstrap standard error (**FP-9**'s helper —
  do not hand-roll one). Otherwise **MATERIAL**, and the class is brought onto the certified schedule.
  *(2% is a chosen default, not a derived bound; the user may retune it. It is fixed here so the
  worker is judged against a bar it did not pick.)*
- **Related prior evidence, for context only, not a substitute:** `width-cert.md` §10.3 records a
  `--schedule uniform` ablation (`…_schedU.json`). Uniform ≠ sum-all-widths; that run does not answer
  this question.

**verify** (runnable):
```bash
cd /home/ff235/dev/MLResearch/automl
# both arms, matched, seeds 0/1/2, tagged so nothing clobbers a canonical file
AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m automl_package.examples.kdropout_converged_width_experiment \
  --arch shared_trunk --loss mse --toy hetero --n-train 1500 --sigma 0.05 --tag fp4_sandwich
AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m automl_package.examples.kdropout_converged_width_experiment \
  --arch shared_trunk --loss mse --toy hetero --n-train 1500 --sigma 0.05 --tag fp4_sumall  # sum-all arm
# both JSONs must exist and the note must carry the verdict WORD
ls automl_package/examples/capacity_ladder_results/W_KDROPOUT_CONVERGED/*fp4_sandwich*.json \
   automl_package/examples/capacity_ladder_results/W_KDROPOUT_CONVERGED/*fp4_sumall*.json
grep -cE '^\*\*VERDICT:\*\* (MATERIAL|IMMATERIAL)$' docs/plans/capacity_programme/shared/fp4-schedule-deviation.md
```
Both JSONs must exist; the `grep -c` must print `1`. **The note must state the three per-seed
percentage differences as numbers** — a verdict without them fails the task.

### FP-5 — reconcile the routers

**Files (write set):** `automl_package/models/common/distilled_router.py` · shims under
`automl_package/examples/` · `docs/plans/capacity_programme/shared/router-capabilities.md` (new)

**Spec:** Make the package router the single implementation for what it covers, and **state in
writing what it does not cover.** Concretely: port the vector/scalar input handling so one class
serves both; add the soft-target training path; and for each capability listed in `PROTECTED.tsv` (§3)
(blend-likelihood evaluation, the direct-objective trainer, the label-construction factorial) either
port it or record explicitly that it is experiment-protocol and stays with the drivers. Leave
re-export shims for `capacity_ladder_k6`'s router so its direct importers keep resolving.

#### 🚨 FP-5.a — the importer count. RE-DERIVE IT; DO NOT TRUST THIS PLAN.

**BINDING RULE: the worker MUST re-derive the importer list by grep at execution time and MUST NOT
trust any count written in this plan, this section included.** This plan has already been wrong about
it once, and the consequence of the error is a broken paper trail on a certified result.

```bash
cd /home/ff235/dev/MLResearch/automl
grep -rn "^import capacity_ladder_k6" automl_package/examples/*.py
```

As of 2026-07-20 that returns **FIVE**, not four:

| importer | why it matters |
|---|---|
| `automl_package/examples/capacity_ladder_s2.py:55` | the direct-objective research arm |
| `automl_package/examples/capacity_ladder_s1.py:75` | the five-arm label factorial |
| `automl_package/examples/depth_selection_toy.py:88` | depth strand |
| `automl_package/examples/capacity_ladder_h1.py:68` | H1 battery |
| **`automl_package/examples/sinc_width_experiment.py:67`** | **the fifth — and the one this plan itself names as the producer of the certified width result (§1.3).** Breaking it breaks the width paper trail. |

Two further files reference `capacity_ladder_k6` in prose/docstrings without importing it
(`hetero_width_experiment.py:18`, `capacity_ladder_t2.py:27`, `:228`, `:261`) — update the text if the
names change, but they are not import-breakage risks.

#### 🚨 FP-5.b — the router constants are a shared global. MEASURE, do not overwrite.

`automl_package/models/common/distilled_router.py:57-60` holds four module-level defaults —
`DEFAULT_TOLERANCE = 0.25`, `DEFAULT_HIDDEN = (32, 32)`, `DEFAULT_N_EPOCHS = 300`, `DEFAULT_LR = 1e-2`
— shared by **every** capacity family.

**Three sibling studies are each authorised to touch this file, and each would overwrite the same four
constants:**

| study | plan | question |
|---|---|---|
| **WSEL-7** | `width.md` | is the router's architecture right for **width**? |
| **DSEL-9** | `depth-selection.md` | does the router's architecture matter for **depth**? |
| **PC** | `probreg.md` | is the distilled router's architecture right for **ProbReg**? |

**RULE: those three studies MEASURE and REPORT. They do not write a new default.** Each produces a
sensitivity table and nothing else. **Only FP-5 writes a new default, once, and only after all three
sensitivity tables exist on disk.** A family-specific optimum written into a shared constant is a
silent regression for the other two families — the last writer wins and nobody notices.

If a family genuinely needs a different value, the answer is a **per-call/per-family parameter**, not
a new global. Adding the parameter is in scope for FP-5; changing the global before all three tables
land is not.

**Doctrine:** §3's protection manifest — those scripts produced the certified width result and their
preregistration documents cite exact function names. **Move the logic, keep the names resolving,
update the documents in the same change or not at all.**

**Non-goals:** do not delete any script router; do not pick a winner between two behaviours by
inspection — if they differ, record both and escalate; do not change the four shared defaults.

*Orchestration:* parallel: yes (disjoint from FP-4) · deps: FP-3 · tier: sonnet high · scale: static ·
shape: execution ·

#### FP-5.c — what the verify reproduces, described accurately

**An earlier draft required reproducing "the certified width selection numbers". No such artifact
class exists.** `width-cert.md` records that the certification covered the **architecture**; none of
the selection *mechanisms* was certified. Requiring a worker to reproduce a nonexistent artifact
produces either a fabricated match or an unfinishable task.

**What FP-5 reproduces instead: the W6 router-capacity DEPLOY-CLAIM results** — the concrete outputs
recorded on the `RESULT`-tagged lines of `width-cert.md` Task W6 (router-capacity sensitivity). These
are deploy-claim results showing the deploy claims hold at half and double router hidden size. They are
**not** certified selection numbers, and the note FP-5 writes must describe them that way.

Reference JSONs, both under
`automl_package/examples/capacity_ladder_results/W_KDROPOUT_CONVERGED/`:
- `w_kdropout_converged_summary_shared_trunk_mse_n1500_s0.05_rhhalf.json`
- `w_kdropout_converged_summary_shared_trunk_mse_n1500_s0.05_rhx2.json`

Cell: toy `hetero`, `n_train=1500`, `sigma=0.05`, arch `shared_trunk`, **3 seeds** (0/1/2), at
`--router-hidden-mult` 0.5 and 2.0.
Metrics: `per_case[i].deploy_bar.mse_hardpick` and `per_case[i].deploy_bar.mean_executed_width`.

**TOLERANCE: ≤ 2% relative error on `mse_hardpick`, per seed, both arms.**
`mean_executed_width` must land within **±0.25 widths** absolute (the dial is coarse; a relative bar
on it is meaningless).
*(Both are chosen defaults, not derived bounds — the user may retune them. They are fixed here so the
worker cannot select its own bar after seeing the result.)*

**verify** (runnable):
```bash
cd /home/ff235/dev/MLResearch/automl
# (a) re-derive the importer list -- MUST be done, MUST match the shims that were left
grep -rn "^import capacity_ladder_k6" automl_package/examples/*.py
# (b) every importer still imports cleanly through the shim
grep -rl "^import capacity_ladder_k6" automl_package/examples/*.py | while read f; do
  AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -c "
import sys; sys.path.insert(0,'automl_package/examples'); import importlib
importlib.import_module('$(basename ${f%.py})')" || echo "IMPORT-BROKE $f"; done
# (c) the four shared defaults are UNCHANGED by this task until all three tables exist
git diff automl_package/models/common/distilled_router.py \
  | grep -E '^[-+]DEFAULT_(TOLERANCE|HIDDEN|N_EPOCHS|LR)'
# (d) reproduce the two W6 deploy-claim arms through the package router
AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m automl_package.examples.kdropout_converged_width_experiment \
  --arch shared_trunk --loss mse --toy hetero --n-train 1500 --sigma 0.05 \
  --router-hidden-mult 0.5 --tag fp5_rhhalf
AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m automl_package.examples.kdropout_converged_width_experiment \
  --arch shared_trunk --loss mse --toy hetero --n-train 1500 --sigma 0.05 \
  --router-hidden-mult 2.0 --tag fp5_rhx2
# (e) the capability note exists and classifies EVERY protected router symbol
for s in $(grep -v '^#' docs/plans/capacity_programme/shared/PROTECTED.tsv \
            | awk -F'\t' '$1 ~ /capacity_ladder_(k6|s1|s2|t2)/ {print $2}'); do
  grep -qE "^\| \`$s\` \| (ported|out-of-scope) \|" docs/plans/capacity_programme/shared/router-capabilities.md \
    || echo "UNCLASSIFIED $s"; done
# (f) protection check + suite
git diff --name-status | awk '$1=="D"||$1=="R"{print $2}' \
  | grep -F -x -f <(grep -v '^#' docs/plans/capacity_programme/shared/PROTECTED.tsv | cut -f1 | sort -u)
AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m pytest tests/ -q
```
(b), (c), (e) and (f)'s grep must each print **nothing**; (a) must print exactly the importer set the
shims cover; (d)'s two JSONs must land and match the reference JSONs within the tolerances above,
per seed, with **both numbers written into the note**; the suite must be green.

### FP-6 — give the independent-weights depth class its missing primitive

**Files (write set):** `automl_package/models/independent_weights_flexible_neural_network.py` · tests
**Spec:** That class has no per-depth forward primitive, no router, and no convergence gating — it
cannot participate in any selection mechanism. Add the primitive and wire it to the FP-3 API and the
FP-5 router, and bring convergence gating to parity with its twin.
*(The pre-registration question originally attached to this task was **resolved at the root before
dispatch** — see §1.6 and §3's *Superseded ≠ deletable* clause. No halt remains here.)*
**Non-goals:** the prefer-shallow prior bug in this class is fixed by `depth-selection.md`; do not
duplicate that fix here.
**Parity target, named.** The twin `FlexibleHiddenLayersNN` sets `self.convergence_summary_` from a
`ConvergenceTracker` (`automl_package/models/flexible_neural_network.py:382`) and surfaces
`trustworthy` from it (`:612`). Verified 2026-07-20:
`automl_package/models/independent_weights_flexible_neural_network.py` has **no** occurrence of
`convergence`, `trustworthy`, `fit_router`, or any per-depth forward primitive. That is the gap.

*Orchestration:* parallel: no · deps: FP-3, FP-5 · tier: sonnet high · scale: static · shape: execution ·

**verify** (runnable):
```bash
cd /home/ff235/dev/MLResearch/automl
# (a) parity: the class exposes the same convergence surface as its twin
AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -c "
from automl_package.models.independent_weights_flexible_neural_network import IndependentWeightsFlexibleNN
from automl_package.models.flexible_neural_network import FlexibleHiddenLayersNN
for name in ('fit_router',):
    assert hasattr(IndependentWeightsFlexibleNN, name), f'missing {name}'
print('PARITY-API-OK')"
# (b) the new tests: fits a router, routes with NO caller flag, and reports trustworthiness
AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m pytest tests/ -q -k independent_weights
# (c) suite
AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m pytest tests/ -q
```
(a) must print `PARITY-API-OK`; (b) must include a test asserting `convergence_summary_["trustworthy"]`
is populated after `fit()`, and a test asserting a `CapacitySelection.PER_INPUT` model routes with no
caller flag; (b) and (c) green.

*(The verify no longer asks for "the pre-registration question answered in writing". That clause
contradicted this task's own body, which records the question as **resolved at the root before
dispatch** — §1.6 and §3's *Superseded ≠ deletable* clause. There is nothing left to answer.)*

### FP-7 — complete the sweep the inventory could not finish — ✅ **DONE 2026-07-20 — INVENTORY STALE as of 2026-07-21**

**⚠️ STALE 2026-07-21.** The sweep covered **104** example modules; `automl_package/examples/` now
holds **105** — `depth_dsel2.py` landed after the sweep ran (untracked at sweep time; landed on
`master` at commit `e3cc52b`, a sibling strand's work). The inventory is valid for what it covered; it
must be **re-swept in the same session that dispatches FP-8** before FP-8 acts on it — see FP-8's
condition (v), added below.

**Files (write set):** `docs/plans/capacity_programme/shared/zero-caller-inventory.md` (new)

**Spec:** The duplication survey was budget-bounded (§1.6). Finish it: (i) a full zero-caller sweep
across `automl_package/`, each absence proven by the grep that shows it; (ii) inventory the suspected
bootstrap / standard-error helper duplication across the experiment drivers. Output is an inventory,
not a cleanup — deletions come after, under §3's protections.

#### 🚨 FP-7.a — the search form. A NAIVE SWEEP MANUFACTURES FALSE DEAD CODE.

**`automl_package/examples/` has NO `__init__.py`** (verify: `ls automl_package/examples/__init__.py`
→ *No such file*). Scripts there put the directory on `sys.path` and then import each other **by bare
module name**:

```python
sys.path.insert(0, _EXAMPLES_DIR)
import nested_width_net as nwn          # automl_package/examples/capacity_accounting.py:59
import capacity_ladder_k6 as ck6        # automl_package/examples/sinc_width_experiment.py:67
```

**A sweep that searches only for `automl_package.examples.M` reports every one of these live modules
as zero-caller.** FP-8 then deletes them. That is the single most dangerous failure mode in this
strand, and it is a search-syntax bug, not a judgment call.

**REQUIRED SEARCH — use exactly this, per candidate module `M`:**

```bash
cd /home/ff235/dev/MLResearch/automl
M=<module_basename_without_py>
grep -rn -E "(^[[:space:]]*(import|from)[[:space:]]+${M}\b)|(automl_package\.examples\.${M}\b)|(\b${M}\.py\b)" \
  --include='*.py' --include='*.md' --include='*.sh' --include='*.txt' \
  automl_package/ docs/ tests/ \
  README.md *.md *.sh *.toml *.txt \
  | grep -v "^automl_package/examples/${M}\.py:"
```

> ### 🚨 CORRECTED 2026-07-20 — the search form above was WRONG, and it was wrong in exactly the
> ### way this section exists to prevent.
>
> **The earlier version scoped the search to `automl_package/ docs/ tests/` and omitted repo-root
> files.** FP-7 ran it as mandated and then cross-checked out of scope, which is how this was
> caught. **8 of its 21 dead-candidates are cited in repo-root files**, most seriously
> `run_automl.py` and `noisy_data_example.py`, which are listed in **README.md's own example
> catalogue at `README.md:1173-1174`** — the user-facing document that tells a reader those
> scripts exist.
>
> **Consequence had it not been caught:** FP-8's deletion gate condition (ii) is satisfied by a
> `dead-candidate` row carrying its proving command and empty output. Both files would have
> qualified, and FP-8 would have deleted scripts the README documents — silently breaking the
> project's own front page. That is the *same* failure mode as the bare-name import hole this
> section already warned about, in a different directory.
>
> **The repo-root paths are now part of the mandated command** (the `README.md *.md *.sh *.toml
> *.txt` line). Re-run it for any candidate assessed before this correction.
>
> **General rule this establishes: a citation anywhere in the repository is a live reference,
> and "anywhere" includes the root.** A search scope is itself a claim about where references
> can live, and it must be justified rather than inherited.

It covers all four reference forms, and it must:
0. include **repo-root files** — `README.md` above all, plus `*.md`, `*.sh`, `*.toml`, `*.txt`.
   **The README is the project's own catalogue of what these scripts are for**, and a script it
   documents is live by definition regardless of what imports it;
1. include **`docs/`** — plan and report documents cite module and symbol names;
2. include **`automl_package/examples/capacity_ladder_results/*/PREREGISTRATION.md`** (reached via
   `--include='*.md'` under `automl_package/`) — **preregistration documents cite exact symbol names,
   and a citation is a live reference to the paper trail even when no code imports the module**;
3. include **`*.sh`** — sweep-launcher scripts invoke drivers by filename, not by import;
4. exclude only the module's own file (the trailing `grep -v`).

**A module is a zero-caller candidate ONLY if that command prints nothing.** Every zero-caller claim
in the inventory carries its command and its empty output. A claim without the command is not a claim.

#### FP-7.b — what the inventory must contain

For every candidate: the module path, the exact command run, the output (or `<empty>`), whether the
path appears in `PROTECTED.tsv`, and a disposition of `dead-candidate` / `live` / `protected`.
**`dead-candidate` is a nomination, not a verdict** — FP-8 decides, under FP-8's three-part gate.

Part (ii): the bootstrap / standard-error duplication. Verified starting point, 2026-07-20 —
**fifteen** such helpers exist across `automl_package/examples/` under at least five different names
(`_boot_se`, `_plain_boot_se`, `_two_sample_boot_se`, `_paired_bootstrap_se`,
`_paired_point_bootstrap_se`, `paired_bootstrap_se`, `_bootstrap_col_means`), and **zero** exist in
`automl_package/models/` or `automl_package/utils/`. Inventory them with signatures so **FP-9** can
build the one that replaces them. Re-derive the list; do not trust this count.

**⚠️ COUNT CORRECTED 2026-07-20 by FP-7 itself, re-deriving as instructed above rather than trusting
this "Verified starting point" figure — the "fifteen" above is SUPERSEDED, not deleted, so a reader
who lands here first sees the history.** Re-derivation (exact-name grep, a broadened name-pattern
grep, and a name-agnostic `rng.integers(0` sweep across all of `examples/`) found **TWELVE**, not
fifteen — confirmed by the actual on-disk deliverable, `shared/zero-caller-inventory.md:1560`
("...not one of the **12** 'helper' duplicates..."). The corrected count is also recorded at FP-9.b's
completion note (below, `flexnn-package.md:436-438`), which is the currently-authoritative status
text for this fact — read it, not this paragraph, for the number.

**Non-goals:** delete nothing in this task. Do not de-duplicate the bootstrap helpers here — that is
FP-9.

*Orchestration:* parallel: yes · deps: FP-0 · tier: sonnet high (mechanical, high volume) ·
scale: static · shape: research ·

**verify** (runnable):
```bash
cd /home/ff235/dev/MLResearch/automl
NOTE=docs/plans/capacity_programme/shared/zero-caller-inventory.md
# (a) every example module is accounted for -- none silently skipped
ls automl_package/examples/*.py | xargs -n1 basename | sed 's/\.py$//' \
  | while read m; do grep -q "\b$m\b" "$NOTE" || echo "UNSWEPT $m"; done
# (b) every dead-candidate row carries a proving command
grep -c 'dead-candidate' "$NOTE"; grep -c 'grep -rn -E' "$NOTE"
# (c) the note states its own coverage boundary
grep -n 'NOT swept' "$NOTE"
# (d) no code was touched
git status --short automl_package/ | grep -v '^??' && echo CODE-TOUCHED || echo CODE-CLEAN
```
(a) must print nothing; in (b) the second count must be **≥** the first; (c) must print a line; (d)
must print `CODE-CLEAN`.

### FP-8 — the cleanup itself

**Files (write set):** determined by FP-7's inventory, listed explicitly in the task before it runs

**Spec:** Remove what FP-7 proved dead and what FP-5 superseded, under §3's protections and the HALT
conditions.

#### 🚨 FP-8.a — THE DELETION GATE. Three conditions, all mechanical, all required.

**A path may be deleted if and only if all FIVE hold. Not three. Not "three and it looks obviously
dead".** The previous version of this task said removals must be "reviewed" — a human verb with no
observable outcome, in a task whose failure mode is silently destroying certified research code.

> ### 🚨 CONDITION (iv) ADDED 2026-07-20 — a `dead-candidate` row is NOT sufficient on its own.
>
> **(iv) the candidate's row in `shared/zero-caller-inventory.md` carries NO `CAVEAT` block.**
>
> FP-7's inventory marks **8 of its 21 dead-candidates with an inline `CAVEAT`**, because they are
> cited in **repo-root files that the search form in force at the time did not cover** — including
> `run_automl.py` and `noisy_data_example.py`, both listed in **README.md's own example catalogue
> (`README.md:1173-1174`)**. Those rows say `dead-candidate` because that is what the mandated
> command produced, and FP-7 correctly reported the mandated result unaltered rather than
> silently overriding it. **Deleting on the disposition alone would have removed scripts the
> project's front page documents.**
>
> ```bash
> # (iv) mechanically: the candidate's section must contain no CAVEAT
> awk -v p="$P" '$0 ~ "^#### .*"p {f=1} f && /^#### /&&$0 !~ p {f=0} f' \
>   docs/plans/capacity_programme/shared/zero-caller-inventory.md | grep -q 'CAVEAT' \
>   && echo "BLOCKED-BY-CAVEAT $P" || echo "no-caveat $P"
> ```
>
> **A caveated candidate is resolved, never overridden**: re-run the *corrected* FP-7.a search
> (which now includes the repo root) and let the disposition change on the evidence. If it comes
> back live, it is live.

> ### 🚨 CONDITION (v) ADDED 2026-07-21 — a stale inventory is not evidence of zero callers.
>
> **(v) the zero-caller inventory has been re-swept in THE SAME SESSION that performs the
> deletions, and every example module added since the last sweep either has a row in it or is
> excluded from deletion.** FP-7's inventory is now known to go stale between when it runs and when
> FP-8 dispatches — confirmed 2026-07-21: `depth_dsel2.py` landed after FP-7's sweep and has no row
> (FP-7's own header now carries the ✅ DONE — INVENTORY STALE marker). A `dead-candidate`
> disposition proves nothing about a file the sweep never saw.
>
> ```bash
> # (v) mechanically: every current example module has a row, re-swept in this same session
> ls automl_package/examples/*.py | xargs -n1 basename | sed 's/\.py$//' \
>   | while read m; do grep -q "\b$m\b" docs/plans/capacity_programme/shared/zero-caller-inventory.md \
>     || echo "UNSWEPT $m"; done
> ```
> Must print nothing. Any `UNSWEPT` module is excluded from this run's deletions until it is added to
> the inventory (as `dead-candidate`, `live`, or `protected`) in the same session.

| # | condition | the command that decides it |
|---|---|---|
| **(i)** | **absent from `PROTECTED.tsv`** | `grep -v '^#' docs/plans/capacity_programme/shared/PROTECTED.tsv \| cut -f1 \| grep -Fxq "$P"` → must **fail** (exit 1) |
| **(ii)** | **FP-7's inventory proves zero callers, using the FP-7.a grep** | the path appears in `zero-caller-inventory.md` as `dead-candidate` **with its proving command and empty output recorded** |
| **(iii)** | **the deletion diff clears the manifest check** | the FP-8.b command below prints nothing |

**Any candidate that fails ANY of the three is LEFT IN PLACE and LISTED**, in a
`not-deleted-and-why` section of the FP-8 completion note, with which condition it failed. **It is
never deleted on a judgment call, and the list is not optional** — a candidate that quietly vanishes
from both the deletions and the not-deleted list is a task failure.

**FP-8 has no discretion.** It does not decide what "produced a certified number" — it cannot, and
attempting to is how the code disappears. `PROTECTED.tsv` is the answer to that question; FP-8 only
executes the gate.

#### FP-8.b — the manifest check on the actual diff

```bash
cd /home/ff235/dev/MLResearch/automl
git diff --name-status | awk '$1=="D"||$1=="R"{print $2}' \
  | grep -F -x -f <(grep -v '^#' docs/plans/capacity_programme/shared/PROTECTED.tsv | cut -f1 | sort -u)
```
**Empty output = clear. Any output = STOP and escalate to the user.** Run it on the staged diff too
(`git diff --cached --name-status`) before any commit.

**Doctrine:** this task runs LAST and only against a written inventory. A cleanup that runs on
judgment rather than on a proven zero-caller list is how working research code disappears.

**Non-goals:** nothing listed in `PROTECTED.tsv` (§3), absent a user ruling. No refactoring, no
renames, no "while I'm here" edits — this task only removes.

*Orchestration:* parallel: no · deps: FP-7, FP-5, FP-6 · tier: **opus xhigh** (irreversible deletion) · **RUNS ATTENDED, never unattended** · scale: static ·
shape: execution ·

#### FP-8.c — verify (runnable)

```bash
cd /home/ff235/dev/MLResearch/automl
MAN=docs/plans/capacity_programme/shared/PROTECTED.tsv
# (i) no protected path deleted or renamed -- unstaged AND staged
git diff --name-status | awk '$1=="D"||$1=="R"{print $2}' \
  | grep -F -x -f <(grep -v '^#' $MAN | cut -f1 | sort -u)
git diff --cached --name-status | awk '$1=="D"||$1=="R"{print $2}' \
  | grep -F -x -f <(grep -v '^#' $MAN | cut -f1 | sort -u)
# (ii) every deletion appears in FP-7's inventory as a dead-candidate -- no unlisted deletion
git diff --name-status | awk '$1=="D"{print $2}' | while read p; do
  grep -q "$(basename $p)" docs/plans/capacity_programme/shared/zero-caller-inventory.md \
    || echo "UNLISTED-DELETION $p"; done
# (iii) every path in the manifest still resolves
grep -v '^#' $MAN | tail -n +2 | cut -f1 | sort -u | while read p; do
  test -e "$p" || echo "MISSING $p"; done
# (iv) every protected producer still runs its own selftest (all 11 distinct protected .py paths
#      have a --selftest flag -- verified 2026-07-20; re-derive, do not trust the count)
grep -v '^#' $MAN | tail -n +2 | cut -f1 | sort -u | grep '\.py$' | while read p; do
  m=$(basename ${p%.py}); AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m automl_package.examples.$m --selftest \
    >/dev/null 2>&1 || echo "SELFTEST-FAIL $p"; done
# (v) suite + lint
AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m pytest tests/ -q
~/dev/.venv/bin/python -m ruff check automl_package/
```
(i), (ii), (iii) and (iv) must each print **nothing**; (v) must be green and clean.

**REPRODUCTION BAR.** FP-8 deletes only proven-zero-caller paths, so it must not change any number.
- **If FP-8 deletes only** — the selftest gate (iv) plus a green suite **is** the reproduction bar.
  No expensive re-run is required, and none should be launched.
- **If FP-8 modifies any protected producer** (it should not; if it does, that is a scope breach to
  flag) — the FP-2 canonical cell must be re-run and match its reference JSON **within 2% relative on
  `per_case[i].fit_bar.ratio_to_floor`, all seeds** (same file, metric and bar as FP-2).

---

## 5. Non-goals for this strand

No experiment design, no studies, no results — this strand changes where code lives and how it is
called, never what an experiment concludes. No reopening of `G-WIDTH` or `G-DEPTH`. No ProbReg
selection-mechanism work beyond adopting the shared API (owned by `probreg.md`). No joint width+depth
work.
