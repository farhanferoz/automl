# Width workstream audit vs ProbReg rigor bar (docs/plans/capacity_programme/probreg.md §1, §3.5, §3.6, §4)

Scope: WIDTH only (`docs/plans/capacity_programme/width-cert.md`, `automl_package/models/flexible_width_network.py`,
`automl_package/models/common/distilled_router.py`, `automl_package/examples/nested_width_net.py`,
`automl_package/examples/kdropout_converged_width_experiment.py`, `automl_package/examples/sinc_width_experiment.py`,
`automl_package/examples/converged_width_experiment.py`, RESULTS.md, and the JSON ledgers). Read-only audit.

---

## Q1 — The three ProbReg-style choices, for WIDTH

**Verdict: PARTIAL.** Per-input router = PRESENT in library code. Global cheap arbiter = PRESENT only
as an ad hoc computation inside an example-script bar, never a first-class selection mode. Expensive
independently-trained sweep = PRESENT as an artifact from an earlier phase, but ABANDONED — never used
as the reference for anything the certification work (width-cert.md) actually compares against.

**(i) cheap global choice (arbiter-equivalent).**
- Library code (`automl_package/models/flexible_width_network.py`): **ABSENT.** `FlexibleWidthNN.predict()`
  (`flexible_width_network.py:192-237`) has exactly two modes: `"fixed"` (caller supplies `width=`,
  defaults to `max(self.widths)` if omitted — `flexible_width_network.py:211-214`, no held-out
  computation at all) and `"routed"` (per-input). There is no third mode that automatically picks one
  global width from held-out data.
- Example script (`automl_package/examples/sinc_width_experiment.py`): **PARTIAL.** Two ad hoc
  functions exist purely as comparator baselines inside the `deploy_bar`/`deploy_sweep` accounting:
  - `_deploy_bar_mse` (`sinc_width_experiment.py:486-507`): `best_fixed_k = min(per_k_mse, key=per_k_mse.get)`
    (line 489) — argmin of TEST-set mean error. This is a **hindsight** selection (picks using the same
    data it is then scored on); the function's own docstring on its W7 sibling calls this out
    (`sinc_width_experiment.py:513-514`: "a hindsight choice").
  - `_deploy_bar_mse_valselected` (`sinc_width_experiment.py:510-525`, added by width-cert Task W7):
    `best_fixed_k_valselected = min(per_k_mse_p2, key=per_k_mse_p2.get)` (line 520) — argmin of
    held-out (slice-B / router-train-half) mean error, then reports that k's TEST MSE. This is the
    genuine held-out global-choice mechanism — functionally the width analog of ProbReg's arbiter.
  - **Both use plain argmin over the mean, NOT the "cheapest-within-tolerance, smallest-sufficient at
    2×SE" rule** that ProbReg's PA task explicitly ratified as the correct rule *because* argmax/argmin
    "systematically overshoots" on noisy held-out curves (`probreg.md:357-361`). Width's arbiter-analog
    uses exactly the rule ProbReg's own plan rejects.
  - Both functions live inside `sinc_width_experiment.py`'s per-arm scoring code, invoked from
    `kdropout_converged_width_experiment.py:run_case` (`sinc_width_experiment.py:370` call site) — i.e.
    they are a **side-analysis reported next to the routed model's number in the same JSON**, not an
    independently queryable selection mode on `FlexibleWidthNN`. See Q3.

**(ii) per-input choice (distilled router).**
- **PRESENT in library code.** `FlexibleWidthNN.fit_router()` (`flexible_width_network.py:239-298`) +
  `predict(x, inference_mode="routed")` (`flexible_width_network.py:223-226`), backed by the shared
  `DistilledCapacityRouter` (`automl_package/models/common/distilled_router.py`) — the same class used
  by FlexNN depth and (task F9) ProbReg's k. This is the one part of the three-choice trio that is
  genuinely wired as a library API for width.

**(iii) expensive global sweep reference (train one model per width, score each held out).**
- **PRESENT as an artifact, but structurally orphaned.** `automl_package/examples/converged_width_experiment.py`
  trains `IndependentWidthNet`'s disjoint per-width sub-nets **independently** — a separate
  `torch.optim.Adam` and its own convergence check per width (`_train_widths_to_convergence`,
  `converged_width_experiment.py:65-90`: `for k in range(1, net.w_max+1): opt = torch.optim.Adam(sub.parameters()...)`).
  This is a genuine "train w_max separate models, score each held out" sweep — the width analog of
  ProbReg's M3. Results: `automl_package/examples/capacity_ladder_results/W_CONVERGED/w_converged_summary.json`.
  - **It is never referenced by `width-cert.md`.** `grep -n "W_CONVERGED\|converged_width_experiment\.py"
    docs/plans/capacity_programme/width-cert.md` returns zero matches. Same null result against
    `docs/width_mse_2026-07-16/verdict_variable_width_mse.md`.
  - `kdropout_converged_width_experiment.py`'s own module docstring (lines 19-22) states the joint
    k-dropout summary "is directly comparable to `capacity_ladder_results/W_CONVERGED/w_converged_summary.json`"
    (same architecture family, data, scoring — "the ONLY thing that changes is the training scheme")
    — i.e. the comparison is explicitly set up to be possible, but **no task in width-cert.md executes
    it.** The "#3 positive control" used throughout W1-W10 is `IndependentWidthNet` trained via the
    SAME joint k-dropout SANDWICH schedule as every other arch (`kdropout_converged_width_experiment.py`
    `run_case`, arch dispatch at lines 285-292) — i.e. still a cheap joint run, not the expensive
    independently-trained reference.
  - **Consequence: width has never tested the ProbReg-style M1≈M3 claim at all** — neither "(a) same
    quality" (does the cheap dial match a model dedicated to each width) nor "(b) same choice" (does
    the cheap global pick match what the expensive independent sweep would pick). The honest-reference
    artifact (`W_CONVERGED`) already exists on disk from an earlier phase of work and was simply never
    brought back in when width-cert.md was written.

---

## Q2 — The two mechanisms (shared/nested vs independent weights); which produced which result

**Verdict: CONCERN CONFIRMED AS HISTORICALLY TRUE, THEN RESOLVED by width-cert.md (2026-07-16).**

Four architectures exist, all in `automl_package/examples/nested_width_net.py`:
- `NestedWidthNet` (`nested_width_net.py:97`) = "#1" — shared trunk **and** shared readout (ONE
  `mean_head`/`logvar_head` for every width).
- `IndependentWidthNet` (`nested_width_net.py:172`) = "#3" — `w_max` fully disjoint sub-nets, sharing
  nothing. Despite the name, when run under `kdropout_converged_width_experiment.py` it is trained by
  the SAME joint k-dropout schedule as every other arch (one optimizer, sampled widths per step) — see
  Q1(iii).
- `SharedTrunkPerWidthHeadNet` (`nested_width_net.py:222`) = "#2" — shared trunk, **per-width own mean
  head**. This is the architecture G-WIDTH certified as the "architecture of record"
  (`width-cert.md:3`, `width-cert.md:318-336`).
- `SharedReadoutPerWidthAffineNet` (`nested_width_net.py:291`) — a 2-parameter/width "minimum seam"
  probe, `width-cert.md` Task W3/W4.

**The prior concern is confirmed for the pre-width-cert era.** `kdropout_converged_width_experiment.py`'s
CLI default is `--arch independent` (`kdropout_converged_width_experiment.py:549`:
`default=Arch.INDEPENDENT.value`), explicitly labeled "the old default / positive control"
(`kdropout_converged_width_experiment.py:88`). The oldest surviving summary,
`capacity_ladder_results/W_KDROPOUT_CONVERGED/w_kdropout_converged_summary.json` (no `arch`/`loss` key
at all — verified by `json.load`, `cfg.get('arch') is None`), predates the `Arch` enum entirely and was
necessarily produced under that old single-architecture (`IndependentWidthNet`) path — i.e. a genuine
headline width result WAS produced with the independent-weights architecture before the
shared/nested-readout architecture (`NestedWidthNet` / `SharedTrunkPerWidthHeadNet`) had been run to
convergence at all. This matches agent-memory `project_width_kdropout_arch_error.md` exactly.

**This is now resolved, on disk, by width-cert.md's own work.** Task W4 ran all four architectures
side by side at the canonical cell and published a table (`width-cert.md:184-189`):

| arch | params/width | fit ratio (seeds 0/1/2) | converged (untrust) |
|---|---|---|---|
| #1 nested (shared readout) | 0 | 5.86 / 5.64 / 3.72 — FAILS | no ([0,1,2]) |
| affine_seam | 2 | 1.21 / 1.56 / 1.38 — intermediate | no ([0,1,2]) |
| #2 shared_trunk (per-width heads) | w_max+1 | 1.09 / 1.06 / 1.08 — at floor | **yes ([])** |
| #3 independent (disjoint nets) | — | 1.04 / 1.01 / 1.19 — at floor | **yes ([])** |

Verified against disk: `w_kdropout_converged_summary_shared_trunk_mse.json` and
`..._independent_mse.json` both have `untrustworthy_seeds: []`; `..._nested_mse.json` has
`untrustworthy_seeds: [0, 1, 2]` (checked directly via `json.load`). So: **`SharedTrunkPerWidthHeadNet`
(#2) has been run to convergence and matches #3's floor** — the concern the brief asked me to verify is
resolved for the certification artifacts. The library port (`FlexibleWidthNN`,
`flexible_width_network.py`) correctly implements the #2 shared-trunk-per-width-head structure (one
shared `trunk_linear`, per-width `nn.ModuleDict` of heads — `flexible_width_network.py:118-120`), not
#1 or #3.

**The arm that has never been run to convergence: `NestedWidthNet` (#1)**, in every recorded run
(`untrustworthy_seeds: [0,1,2]` at both the canonical cell and its `longbudget_confirm` tag — file
`w_kdropout_converged_summary_nested_mse_n1500_s0.05_longbudget_confirm.json` exists on disk, name
implies a longer-budget confirmation was attempted). Its "FAILS" verdict (5.86-3.72x above floor) is
therefore drawn from a non-converged run — width-cert.md **discloses** this in the same table rather
than hiding it (unlike the `affine_seam` arm, which is explicitly marked "INCONCLUSIVE" for the
identical reason — non-convergence — rather than "FAILS"). This is an internal inconsistency in how
the same non-convergence condition is labeled between arm #1 ("FAILS", stated as a hard verdict) and
`affine_seam` ("INCONCLUSIVE" — width-cert.md:194), worth flagging even though the #1 numbers are ~3-5x
its own floor and unlikely to converge down to it.

**Residual live risk:** the CLI default of `kdropout_converged_width_experiment.py` is still
`--arch independent` today (`kdropout_converged_width_experiment.py:549`) — the certified architecture
(`shared_trunk`) is NOT the default. Anyone running the driver without `--arch shared_trunk` silently
gets the superseded arm. Same silent-failure shape PA calls out for ProbReg's `inference_mode`
(`probreg.md:337-339`).

---

## Q3 — Lesson 1: is each arm a COMPLETE SYSTEM (scored + costed end-to-end, selection included)?

**Verdict: ABSENT for the routed/arbiter comparison; PARTIAL for the router's own cost accounting.**

- The router's OWN deployed cost is genuinely inside the JSON: `dial_bar`/`deploy_bar` report
  `mean_executed_width` alongside accuracy (`sinc_width_experiment.py:492`,
  `kdropout_converged_width_experiment.py:362`), and `DistilledCapacityRouter.mean_deployed_cost`
  (`distilled_router.py:223-231`) exists as a general-purpose costing method. So *within* the routed
  arm, compute is accounted for, not a silent footnote.
- **But the "arbiter" (global fixed-k) arm is never scored as its own row.** `_deploy_bar_mse` /
  `_deploy_bar_mse_valselected` compute `best_fixed_k` and its MSE and land the result as ONE MORE KEY
  (`deploy_bar`, `deploy_sweep.baseline_val_selected`) inside the **same JSON case dict** that also
  holds the routed model's `dial_bar`/`fit_bar`/`curve_gate` (`kdropout_converged_width_experiment.py:387-399`).
  There is no separate end-to-end "M1" object with its own cost line the way ProbReg's §1 mandates
  ("A table row for M1 is the arbiter's answer, not the network's answer with the arbiter mentioned in
  a footnote" — `probreg.md:26-28`). Width's fixed-k baseline IS exactly that kind of footnote: a
  `best_fixed_k`/`mse_best_fixed` pair nested inside the routed model's result.
- The router's OWN training cost (fitting cost of the `DistilledCapacityRouter` MLP itself — 300 epochs
  of a `(32,32)` net) is never counted into any of the reported cost numbers for width (no evidence
  found of a router-fit-FLOPs line item anywhere in `sinc_width_experiment.py` or
  `kdropout_converged_width_experiment.py`).
- No evidence of an accounted cost for the ABANDONED expensive-sweep arm either (Q1.iii) — since it is
  not used, its cost was never compared against the cheap arms' cost (the ratio the whole efficiency
  claim would need).

---

## Q4 — Lesson 2a: selection-data-size study for width

**Verdict: ABSENT.** Searched `width-cert.md`, `kdropout_converged_width_experiment.py`,
`sinc_width_experiment.py` for any fraction sweep of the selection/router-training set — none found.

The selection split is a **fixed, hardcoded 50/50 convention**, not a measured choice:
```
p1_idx = np.arange(0, n_train, 2)   # phase 1: net training + convergence monitoring
p2_idx = np.arange(1, n_train, 2)   # phase 2: router/selector training + the fixed-k baselines
```
(`kdropout_converged_width_experiment.py:273-276`). The router is trained on ALL of phase 2 (literally
every odd-indexed training point — an even/odd split of `n_train`, i.e. 50% of the training portion),
with no flag or task anywhere to vary that fraction. This is the same class of gap ProbReg's PB task
was created to close for its own 15% constant (`probreg.md:386-388`: "chosen because it looked
reasonable, not because anything was measured... no study of selection-set size exists anywhere in
this repo for ProbReg") — width's number (50%) is equally unmeasured, and no width-cert task
(W1-W10) touches it.

---

## Q5 — Lesson 2b: selector architecture sensitivity for width

**Verdict: PARTIAL — exists, but exactly as narrow as ProbReg's own plan already characterizes it.**

Task W6 ("router-capacity sensitivity", `width-cert.md:222-240`) is the only sensitivity study found.
Precisely what it did:
- **One dimension varied**: router hidden-layer size, via a new `--router-hidden-mult` flag
  (`kdropout_converged_width_experiment.py:562-563`, `run_case:309`: `router_hidden = tuple(max(1,
  round(h * router_hidden_mult)) for h in sw.ck6.HIDDEN)`), applied to BOTH of the router's two hidden
  layers uniformly (`ck6.HIDDEN = (32, 32)`, confirmed in `distilled_router.py:58` as the same
  convention).
- **Two settings tested** (plus the canonical baseline): half (`rhhalf`) and double (`rhx2`) —
  `width-cert.md:234`: "#2 canonical cell at half/double router hidden, 3 seeds each."
  Not tested: depth (number of layers, e.g. 1 vs 3 as ProbReg's PC plans), training epochs, learning
  rate, or the labeling tolerance (`DELTA_TIE`/`tolerance`) — W7's `delta_tie` sweep is a SEPARATE task
  or a different parameter, and even that only varies tie-margin, not router capacity.
- **3 seeds**, at the single canonical toy cell (hetero, n1500, σ0.05) — no cross-toy check.
- **Outcome reported as a binary pass/fail** ("router-capacity invariant — deploy claims hold at half
  and double router hidden size", `width-cert.md:237`), not a table of how the router's own accuracy or
  its output labels change with size — i.e. it checks "does the DOWNSTREAM deploy conclusion survive"
  rather than characterizing the router itself.
- This is a **does-it-break check, not a search** — exactly the characterization ProbReg's own plan
  already gives it verbatim: "That is reassuring but it is (a) a different strand, (b) one dimension,
  (c) a does-it-break check, not a search" (`probreg.md:410-413`, citing `width-cert.md:234`/`:237`
  directly). Confirmed those line numbers are correct against the file as read.

---

## Q6 — Lesson 3: API — how does a caller select between width options today?

**Verdict: PARTIAL — wired but with the identical magic-string / silent-failure pattern PA flags for
ProbReg, PLUS a second, parallel, explicitly-broken API surface.**

- `FlexibleWidthNN.predict(x, filter_data=True, width=None, inference_mode="fixed")`
  (`flexible_width_network.py:192`). `inference_mode` is validated by a hand-rolled membership check
  — `if inference_mode not in ("fixed", "routed"): raise ValueError(...)` (`flexible_width_network.py:204-205`)
  — a **raw string, not an enum or `Literal` type** at the call site. This is the exact pattern PA
  calls a defect for ProbReg ("`inference_mode` is a raw string, validated by a hand-rolled membership
  check... This breaks the repo's own closed-set rule", `probreg.md:340-342`), and CLAUDE.md's own
  house rule ("Use enums... instead of string literals for model options").
- **Silent-failure path exists, same shape as ProbReg's**: `predict()` defaults `inference_mode="fixed"`
  and, under `"fixed"`, defaults `width=max(self.widths)` when the caller omits it
  (`flexible_width_network.py:211-212`) — no error, no warning. If a caller fits a router
  (`fit_router()`) and then calls `predict(x)` without passing `inference_mode="routed"`, they silently
  get the fixed-largest-width prediction instead of the routed one, with nothing recorded that a router
  was fitted but unused. Verified: `predict()`'s only guard against this is the reverse case
  (`inference_mode="routed"` with no router fitted raises `RuntimeError`,
  `flexible_width_network.py:208-209`) — there is no guard in the other direction.
- **A second, separate, and explicitly non-functional API exists in parallel**:
  `WidthSelectionMethod` enum (`automl_package/enums.py:105-109`, values `NONE`/`DISTILLED`) is a
  constructor kwarg (`width_selection_method`), NOT the same channel as `predict`'s `inference_mode`
  string. Setting `width_selection_method=WidthSelectionMethod.DISTILLED` raises `NotImplementedError`
  at construction (`flexible_width_network.py:91-96`) — it is enum-typed but permanently broken/unwired,
  while the actually-working routing path (`inference_mode="routed"` string) is NOT enum-typed. So the
  codebase has the enum on the wrong knob: the closed-set type sits on the API that doesn't work, and
  the magic string sits on the one that does.

---

## Q7 — Lesson 4: comparison against baselines and the expensive reference

**Verdict: ABSENT for both halves.**

- **Against the expensive reference (M3-analog):** never done — see Q1(iii). No task, no JSON, no
  report text compares the cheap dial's (arbiter- or router-selected) width against what the
  independently-trained-per-width sweep (`W_CONVERGED`) would have chosen or scored, despite that
  artifact already existing on disk and the harness explicitly noting the two are "directly comparable."
- **Against external ML baselines (XGBoost/LightGBM/CatBoost/standard NN):** **no task exists for
  width at all.** `grep -rn "width" docs/plans/capacity_programme/*.md | grep -i
  "baseline\|xgboost\|lightgbm\|catboost"` returns only width-cert.md's *internal* "deploy-baseline"
  language (the fixed-width comparator, not an external model) and `flexnn-core.md:618`'s
  parenthetical "no width/depth content (F7 owns that)" inside a DIFFERENT task's non-goals list. The
  closest thing that exists is `flexnn-core.md` Task F6, "FlexNN-vs-MoE comparison battery," which
  would include `FlexibleWidthNN` (routed) as one contender against a Mixture-of-Experts baseline
  (`flexnn-core.md:294-326`) — but that is architecture-vs-architecture, not vs. XGBoost/LightGBM/etc.,
  and its own file states plainly: **"STATUS 2026-07-20: NOT RUN. ... the battery was never executed —
  `automl_package/examples/moe_comparison_results/` does not exist. A prior session carried this as
  substantially done; it is not"** (`flexnn-core.md:305-308`, verified: `ls
  automl_package/examples/moe_comparison_results` — directory absent, confirmed via the earlier
  directory listing of `automl_package/examples/` which does not show it).
- **No real-data test exists for width** — every width-cert artifact is on the synthetic `hetero`/
  `hetero3` toys (`nested_width_net.py`'s `Toy` enum has exactly two members, both synthetic).
- **Report status:** `docs/width_mse_2026-07-16/verdict_variable_width_mse.md` exists (24,376 bytes,
  §10 "Certification addendum" verified present at line 249, §10.1 the gate at line 259, verdict
  statement at line 285 matching width-cert.md's quote verbatim). This is a genuine, on-disk, closed
  report for the ARCHITECTURE question (which net structure to use). It is **not** a comparison report
  against baselines or the expensive reference — those never ran, so there is nothing for a report to
  cover on that axis.

---

## Claims in width-cert.md checked against disk artifacts

All `RESULT:` line JSON paths in `width-cert.md` (W1, W2×4, W3, W4×2, W5, W6×2, W7×2, W8×2, W9×4)
were checked against `ls automl_package/examples/capacity_ladder_results/W_KDROPOUT_CONVERGED/` —
**every referenced filename exists on disk.** Spot-checked four (`_shared_trunk_mse.json`,
`_independent_mse.json`, `_nested_mse.json`, the un-suffixed legacy file) by loading them and reading
`config.arch`/`config.loss`/`untrustworthy_seeds` directly — all matched the prose claims made about
them (arch labels, empty vs non-empty `untrustworthy_seeds`). I did not re-verify every numeric bar
value (fit ratios, SE thresholds) inside every JSON against every prose number in width-cert.md — that
would require re-deriving each bar from the JSON's raw per-case fields, which was out of the time
budget for this audit; flagging as UNVERIFIED at that level of granularity. The one prose/artifact
mismatch I did find (Q2, #1's "FAILS" label vs `affine_seam`'s "INCONCLUSIVE" label under the identical
non-convergence condition) is a labeling-consistency issue, not a fabricated result.

## Stated closure status of the width work — verbatim

`width-cert.md:318`: **"✅ G-WIDTH = PASS (2026-07-16, orchestrator/opus, re-derived from disk).
`SharedTrunkPerWidthHeadNet` (#2) is the architecture of record."** The gate's own pre-registered rule
(`width-cert.md:308-311`) is narrowly scoped to two clauses: (a) the noisy-easy dial-reads-capacity-not-
error control, and (b) dial-separation + fit-at-floor across the WP-4 corner cells. **The certification
is scoped to "which architecture" and "does the dial separate/fit," not to any of the three
ProbReg-style selection-mechanism questions (Q1) — those were simply never in the gate's scope, so
G-WIDTH=PASS should not be read as certifying an arbiter, a router-vs-sweep agreement, or a
baseline comparison for width; it certifies the network architecture only.**

---

## GAPS IF WIDTH WERE HELD TO THE PROBREG BAR (most serious first)

1. **No M1≈M3 test exists, and the honest reference already exists unused.** The independently-trained
   per-width sweep (`converged_width_experiment.py` / `W_CONVERGED/w_converged_summary.json`) sits on
   disk, explicitly documented as "directly comparable" to the k-dropout results, and was never
   compared. This is the single largest gap — it is the width analog of ProbReg's entire P3 task
   (`probreg.md:487-503`), and unlike ProbReg's case, width does not even need new compute to start:
   the reference run already exists.
2. **No baseline comparison at all** — neither against MoE (driver built, never run — Q7) nor against
   XGBoost/LightGBM/CatBoost/a standard NN (no task exists). ProbReg's P4 has no width counterpart.
3. **The arbiter-equivalent is not a first-class, end-to-end-costed arm.** It is two ad hoc argmin
   functions nested inside the routed model's own JSON (Q1.i, Q3) — no dedicated cost/accuracy row of
   its own, and it uses the argmin rule ProbReg's PA explicitly rejected in favor of
   cheapest-within-tolerance.
4. **The selection-data-size question is unmeasured** — width's 50% split is exactly as unjustified as
   ProbReg's 15% was before PB (Q4), and no sweep exists.
5. **The router-architecture sensitivity study is one dimension, two points, one toy, 3 seeds** — a
   does-it-break check (Q5), same as ProbReg's own plan already says.
6. **API has the same silent-failure and enum/magic-string mismatch ProbReg's PA is fixing** — plus a
   second, dead, enum-typed parallel API (`WidthSelectionMethod.DISTILLED`) that raises
   `NotImplementedError` (Q6).
7. **Residual footgun**: the research driver's CLI default (`--arch independent`) still does not match
   the certified architecture (`shared_trunk`) — a caller who doesn't pass `--arch shared_trunk`
   silently reproduces the superseded pre-certification result (Q2).
8. **Labeling inconsistency**: arch #1 is called "FAILS" on a non-converged run while `affine_seam` is
   called "INCONCLUSIVE" on the identically non-converged condition (Q2) — a minor but real
   methodological inconsistency in width-cert.md's own doctrine about not concluding from
   non-trustworthy results.

## UNVERIFIED

- Exact numeric reproduction of every bar value in every referenced JSON against width-cert.md's prose
  (spot-checked 4 files' `arch`/`loss`/`untrustworthy_seeds` only; did not recompute fit ratios, SE
  thresholds, or dial separation numbers from raw per-case data for all ~30 JSONs).
- Whether any width-related work exists in `docs/plans/width_dial_2026-07-11/` or
  `docs/plans/width_mse_2026-07-16/EXECUTION_PLAN.md` beyond what width-cert.md and the verdict doc
  already surface — these were out of the assigned file list and only spot-checked for existence
  (`docs/plans/width_mse_2026-07-16/EXECUTION_PLAN.md` exists, 22,904 bytes, last modified 2026-07-20;
  not read in full — orchestrator note: this file's mtime is AFTER width-cert.md's own dates, so it may
  contain post-certification updates not reflected in width-cert.md; flagging for the orchestrator to
  check if relevant).
- `tests/test_flexible_width_network.py` **exists** (confirmed via `ls`: 10,916 bytes, modified
  2026-07-18) — but its contents were not read and it was not run, so whether it passes, and whether it
  covers the silent-failure path noted in Q6, remains unverified.
