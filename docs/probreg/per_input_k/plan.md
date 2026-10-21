# ProbReg — per-input-k sub-plan

A small router, distilled from held-out per-k errors, picks k for each input (M2). Definitions:
`docs/probreg/PLAN.md` §1. Tasks moved verbatim from the former <!-- citecheck-ignore -->
`docs/plans/capacity_programme/probreg.md` on 2026-10-21 (PB, PC). Earlier router work: <!-- citecheck-ignore -->
`docs/probreg/per_input_k/history/`.

<!-- from docs/plans/capacity_programme/probreg.md lines 980-1044 -->
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
directories, see Files above) · deps: P1, PA, P7 · tier: sonnet high · scale: dynamic (a sweep) ·
shape: research · verify:
`test -f automl_package/examples/capacity_ladder_results/PB/frozen.json` and
`python -c "import json; d=json.load(open('automl_package/examples/capacity_ladder_results/PB/frozen.json')); assert 'fraction' in d and 'data_limited' in d, d"`
exits 0; `ls automl_package/examples/capacity_ladder_results/PB/*.json` shows one file per
(toy, seed, fraction, arm); a saturation plot file exists under the same results dir.

### PC — is the distilled router's architecture right for ProbReg? (NEW, user 2026-07-20)

**What exists.** The router is fixed at two hidden layers of 32 units, ReLU, 300 full-batch Adam
epochs, lr 1e-2, cross-entropy on cheapest-within-tolerance labels (tolerance 0.25) —
`automl_package/models/flexnn/routing.py:57-60`. Those constants were **copied from an
earlier width experiment**, not chosen for ProbReg. The only sensitivity evidence anywhere is from
the WIDTH strand: `docs/flexnn/width/history/width-cert.md:234` ran half and double router hidden
on 3 seeds and found the deploy claims invariant (`:237`). That is reassuring but it is (a) a
different strand, (b) one dimension, (c) a does-it-break check, not a search.

**Files (write set):** `automl_package/examples/probreg_pc.py` (Create) ·
`automl_package/examples/capacity_ladder_results/PC/`
**Spec:** on the toys, vary router width/depth (at least half/double/4× hidden and 1 vs 3 layers)
and epochs. *(The labelling tolerance is STRUCK from the swept dimensions, 2026-07-21: MASTER
Decision 18 rules that sweep NOT scheduled — do not run pre-emptively; this spec contradicted the
register. The flat 0.25 stays inherited-and-accepted.)* Establish whether ProbReg's routing
conclusions are invariant the way width's were, and if not, what the router actually needs.
**Emit the frozen-constants artifact §3.6 promises:**
`automl_package/examples/capacity_ladder_results/PC/frozen.json`, containing exactly the two
constants this task owns per §3.6 — router hidden/depth/epochs/lr. If
this task finds invariance, the file records the current frozen defaults
(`automl_package/models/flexnn/routing.py:57-60`) rather than inventing new ones.
**Doctrine:** the router stays FROZEN and untuned inside the benchmark (§2.2) so the M1/M2 contrast
measures selection rather than search effort. **This task does not unfreeze it** — it establishes
whether the frozen choice is defensible, and any change lands as a new frozen default *before* P4
runs, never per-dataset.
**Non-goals:** no per-dataset tuning of the router, ever. No change to the labelling rule's meaning.
*Orchestration:* parallel: yes (disjoint from PB's write set — separate driver scripts and results
directories, see Files above) · deps: P1, PA, P7 · tier: sonnet high · scale: dynamic · shape: research ·
verify: `test -f automl_package/examples/capacity_ladder_results/PC/frozen.json` and
`python -c "import json; d=json.load(open('automl_package/examples/capacity_ladder_results/PC/frozen.json')); assert {'hidden','depth','epochs','lr'} <= d.keys(), d"`
exits 0; `python -c "import json; d=json.load(open('automl_package/examples/capacity_ladder_results/PC/frozen.json')); assert 'invariant' in d, d"`
exits 0 (the invariant-or-not verdict is a field, not prose); if `d['invariant']` is `False`, the
same file's `new_default` key is non-null and cited by PB's re-run.


