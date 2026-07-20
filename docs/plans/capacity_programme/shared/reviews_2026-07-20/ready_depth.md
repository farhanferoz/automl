# Autonomous-execution readiness review — `/home/ff235/dev/MLResearch/automl/docs/plans/capacity_programme/depth-selection.md`

Read-only review. Every claim below cites plan text or a file:line I opened.

**Headline:** the plan's *citations* are in unusually good shape (all resolve; the mechanical gate
passes), but its *inferences from those citations* contain two load-bearing errors that would send a
zero-context worker to build the wrong training scheme, and **11 of 13 dispatchable tasks have a
`verify:` line that is not a runnable command with an observable pass/fail** — the exact failure this
review exists to prevent. Only DSEL-3 carries a genuine prove-it-fails clause.

---

## Section 1 — Task table

| Task | Verdict | Single biggest gap |
|---|---|---|
| DSEL-0 | DISPATCHABLE-WITH-FIX | `verify` uses `grep -cin` (returns a *count*) to prove "returns only pointer text" — no threshold, not checkable |
| DSEL-1 | DISPATCHABLE-WITH-FIX | The note's path is never named, yet §3.6 makes it the owning artifact of the supervision constant → dangling for DSEL-11 |
| DSEL-1b | **NOT-DISPATCHABLE** | Its two binding defaults (sandwich sampling; shared readout) are both **refuted by the code the plan cites** — see §2 |
| DSEL-2 | **NOT-DISPATCHABLE** | Positive-control pass bar is never stated numerically; `deps:` omits DSEL-1b, which the spec depends on |
| DSEL-3 | DISPATCHABLE-WITH-FIX | Verify runs only `test_phase2_flexible_nn.py`; the class being changed is exercised by 3 other test files |
| DSEL-4 | **NOT-DISPATCHABLE** | Deps are unnamed tasks in another strand; "stated tolerance" is never stated; the grep in `verify` misses the local-router shapes that actually exist |
| DSEL-5 | n/a (explicitly not a task) | Appears in the §4 order line as if executable |
| DSEL-6 | DISPATCHABLE-WITH-FIX | The bootstrap-SE procedure is unspecified, and its rule contradicts the router's shipped labelling rule |
| DSEL-7 | DISPATCHABLE-WITH-FIX | "the frozen ladder" has no owning artifact; driver unnamed; verify is artifact-existence |
| DSEL-8 | **NOT-DISPATCHABLE** | Its premise ("3000 per stratum, chosen for wall-clock reasons") is contradicted by the code and unsupported by the verdict |
| DSEL-9 | DISPATCHABLE-WITH-FIX | Doctrine contradicts the spec: sweep the labelling tolerance vs "No change to the labelling rule's meaning" |
| DSEL-10 | DISPATCHABLE-WITH-FIX | "agreement" is never operationalised (metric, seed aggregation, tie handling) |
| DSEL-11 | **NOT-DISPATCHABLE** | Must "read §3.6's constants from their artifacts at startup" — 4 of 6 artifacts have no path anywhere in the plan |
| DSEL-12 | DISPATCHABLE-WITH-FIX | Verify depends on the §3.6 constants having citable artifacts (they do not) |

---

## Section 2 — Per-task detail for everything not plainly DISPATCHABLE

### DSEL-1b — NOT-DISPATCHABLE. Both binding defaults are refuted by primary evidence.

**(a) The "sandwich" default is founded on a misreading.**

Plan text (line 264-267):
> **Depth sampling — this is the actual missing piece.** The certified run trained with a
> **per-depth sandwich** (`docs/depth_capacity/verdict_per_input_depth.md:70`), the same shape width
> uses over widths: always the shallowest and the deepest, plus random middles. … **Default: the
> sandwich shape**, not summing all depths every step

What the cited line actually says (`/home/ff235/dev/MLResearch/automl/docs/depth_capacity/verdict_per_input_depth.md:70`):
> `LR 3e-3, per-depth readout sandwich, convergence-gated. Three arms: shared_readout (one readout for`

The phrase exists. The *schedule it describes* does not. Both certified depth training loops sum or
average over **every** rung, every step, with no sampling:

- `/home/ff235/dev/MLResearch/automl/automl_package/examples/depth_selection_toy.py:475` (the
  selection run, D8b):
  `loss = sum(ce(net(x_tr_t[:, : t * n_gen]), y_tr_by_t[t]) for t in t_ladder) / len(t_ladder)`
- `/home/ff235/dev/MLResearch/automl/automl_package/examples/depth_graded_toy.py:148-150` (the D1b
  substrate battery that produced the verdict §2 table — `_train_mixed`, docstring `:134`
  "summed-over-lengths CE"):
  `loss = sum(ce(fwd(tensors[ell][0], ell), tensors[ell][1]) for ell in lengths)`

The width sandwich the plan is transferring is a *different* thing —
`/home/ff235/dev/MLResearch/automl/automl_package/examples/nested_width_net.py:93`:
`SANDWICH = "sandwich"  # W2 fix: every step ALWAYS trains width=1 and width=w_max (+2 random mid).`

The plan even contradicts itself: DSEL-1's own diagnosis table (line 203) describes the certified run
as "mean cross-entropy over **every** depth in the ladder" — i.e. exactly the sum-all scheme DSEL-1b
then forbids as "an unmeasured deviation".

**Consequence, and why this is the worst defect in the plan:** DSEL-1b would build a protocol that
differs from the certified one, DSEL-2 then requires the certified recurrent arm to "reproduce its
certified bar under this task's protocol" (line 302) — under a protocol it was never certified with —
and MASTER Decision 15 (`MASTER.md:110-116`) requires exactly the diff that was not done. This is the
same shape as the F5b case law the strand was created to correct.

**(b) The shared-readout default transfers a recurrent-specific mechanism onto the feed-forward arm.**

Plan text (lines 245-251) makes `shared_readout` the default for the feed-forward build, citing
`verdict_per_input_depth.md:70-71` (param counts — **verified correct**: `:70-71` do read
`shared_readout` 16,376 params / `per_length_head` 39,776) and `:40` (**verified correct**).

But the verdict's own *mechanism* explanation, at
`/home/ff235/dev/MLResearch/automl/docs/depth_capacity/verdict_per_input_depth.md:40-43`:
> for width, one shared network **could not** serve every width (the shared readout broke); for depth,
> one shared network **can** serve every depth, **precisely because the weight-shared recurrent block
> presents every depth with the *same* state space**, so one readout suffices.

and `depth.md:29-31`:
> readout interference is WIDTH-SPECIFIC — a weight-shared recurrent block presents every depth with
> the SAME state space, so one readout serves all depths; width prefixes hand each capacity a
> DIFFERENT representation fighting over one readout.

A feed-forward net with **distinct weights per layer** (the strand's primary object, §1a line 33) is
the *prefix* case, not the weight-shared case — each depth hands the readout a different
representation. The certified evidence therefore does **not** license a shared readout for the
feed-forward arm; by the verdict's own mechanism it predicts the opposite. The task's own binding
method note (lines 275-277, "a width result is NOT evidence about depth … must be flagged as a
hypothesis and tested, never adopted as a default") is violated in the recurrent→feed-forward
direction two paragraphs above it.

Same paragraph, line 259-261: `FlexibleHiddenLayersNN` is called "the certified-winning shape,
already in the package". It is not the certified shape. The certified shape is a weight-shared
recurrent block (`depth_composition_toy.py:341` `RecurrentComposer` / `depth_graded_toy.py` heads);
`FlexibleHiddenLayersNN` is a feed-forward prefix of distinct blocks
(`/home/ff235/dev/MLResearch/automl/automl_package/models/flexible_neural_network.py:29-38` —
citation **resolves and says what the plan claims about nesting**, `:133` shared `output_layer`
**verified**). The nesting citation is fine; the "certified-winning" label conflates the two
mechanisms §1a exists to keep apart.

**(c) Mechanics.** Write set is `automl_package/examples/depth_composition_toy.py` · `tests/` —
`tests/` is a directory, not a path. DSEL-3 is declared `parallel: yes · deps: none` and writes
`tests/test_phase2_flexible_nn.py`; nothing in the plan prevents DSEL-1b from choosing that file.
`verify:` ("a test asserts every depth … produces a usable output"; "the recurrent arm retrained
under this scheme reproduces its certified bar") names no command, no test id, no bar value, no seed
count.

**Fix:** delete the sandwich default and replace it with "sum/average over all rungs, matching
`depth_graded_toy._train_mixed:148` and `depth_selection_toy.train_anytime:475`", *or* keep sandwich
and register it explicitly as a protocol deviation with a Decision-15 written justification and a
sandwich-vs-sum-all control arm. Demote `shared_readout` from default to *one of two pre-registered
arms* for the feed-forward mechanism, with the prediction written down first. Name the test file.
State the pass bar numerically.

### DSEL-2 — NOT-DISPATCHABLE

1. **Write set is self-contradicting.** Line 287: "a driver under `automl_package/examples/` (name it
   in the task, not dated)". The task does not name it. A worker must invent the path; the write-set
   guard cannot pre-clear it; DSEL-4's `verify` grep ("over this strand's drivers") has no target set.
2. **Deps are wrong.** `deps: DSEL-1` (line 300), but the spec's first sentence is "Using DSEL-1b's
   nested training scheme". DSEL-1b is the real dependency. As written, an autonomous dispatcher that
   trusts `deps:` will run DSEL-2 in parallel with DSEL-1b, on the same substrate file.
3. **The positive-control bar is never stated.** `verify:` says "reproduces ≥ its certified bar". The
   plan never gives a number and §3.6 does not own one. The repo has at least two candidate bars:
   `flexnn-core.md:235` ("PASS BAR: held-out ≥ 0.90 AND `trustworthy=true` on BOTH seeds") and the
   D1b table's ≥0.958 (`verdict_per_input_depth.md:24`). A zero-context worker cannot choose, and the
   choice decides whether the strand HALTs.
4. **Substrate parameters unfrozen**: group, word length, seeds, ladder, epochs, lr — none stated,
   none owned by §3.6. §3.6 lists "depth ladder / max depth" as owned by *DSEL-10/DSEL-11*, i.e. by
   tasks that run **after** DSEL-2 needs it.
5. `verify:` is prose ("per-(seed, depth) JSONs with full held-out trajectories") — no command, no
   path, no assertion.
6. **Note on feasibility a worker will hit:** the feed-forward net in the cited substrate is pinned:
   `/home/ff235/dev/MLResearch/automl/automl_package/examples/depth_composition_toy.py:373-374` —
   "per-step arms are structurally pinned at `d = seq_len`". A depth ladder over that arm requires
   changing the data generator, which the §3.5 "ceiling binds → extend the ladder one rung" default
   silently authorises without saying so.

### DSEL-4 — NOT-DISPATCHABLE

1. **Deps unresolvable**: "`flexnn-package.md` API + router tasks" — no task ids. `flexnn-package.md`
   has FP-0..FP-8 (`flexnn-package.md:189`); which ones gate this is left to the worker. Everything
   from DSEL-6 onward transitively depends on this, so the entire back half of the plan is gated on
   an unnamed cross-strand condition.
2. **"reproduce … to a stated tolerance"** — no tolerance is stated anywhere in the plan.
3. **The `verify:` grep is demonstrably insufficient.** It runs
   `grep -n "RouterMLP\|def fit_router"`. Local router implementations in this repo that pattern
   misses: `automl_package/examples/capacity_ladder_s2.py:86` `_train_router_direct`,
   `automl_package/examples/sinc_width_experiment.py:251,438` `_fit_selector`/`_fit_selector_mse`,
   `automl_package/examples/capacity_ladder_h1.py:386` `_standalone_router_nll`. The check can pass
   with a local router still present.
4. **Rule conflict the task must resolve but does not.** §1b (line 50-53) fixes the selection rule for
   all three models as "twice a bootstrap-estimated standard error". The certified depth selection and
   the shipping router both use a *relative* tolerance instead:
   `/home/ff235/dev/MLResearch/automl/automl_package/models/common/distilled_router.py:57`
   `DEFAULT_TOLERANCE = 0.25`, applied as `error <= (1 + tolerance) * row_min` (`:64`); and
   `depth_selection_toy.py:122` `DELTA_TIE = sw.DELTA_TIE`, `depth.md:239` "`δ_tie = 0.25`". So
   "reproduce the certified numbers through the package router" and "apply §1b's rule" cannot both be
   satisfied. Nothing in the plan says which wins.

### DSEL-8 — NOT-DISPATCHABLE

1. **The premise is unsupported and partly false.** Plan line 381: "The current selection set is 3000
   per stratum, chosen for wall-clock reasons, never measured." No citation is given. The code default
   is `/home/ff235/dev/MLResearch/automl/automl_package/examples/depth_selection_toy.py:109`
   `N_PER_STRATUM_DEFAULT = 40000`. The only "3000" I can find is
   `verdict_per_input_depth.md:290` — "n = 3000 words/stratum, 2 seeds" — which is the **total dataset
   size per stratum for the as-run D8b toy**, not a *selection* set, and the verdict attributes the
   D8a→D8b construction change to GD-trainability at L=16, not to wall-clock
   (`verdict_per_input_depth.md:292-295`).
2. **Sweep unit is ambiguous.** The task says sweep "`{5,10,15,25,40}%` of the training portion",
   while the premise is stated in absolute words/stratum. A worker cannot tell whether to vary
   `n_per_stratum`, `train_frac`, or the router's slice-B fraction — three different knobs in
   `depth_selection_toy.py` (`:389` `n_per_stratum`, `:389` `train_frac`).
3. **§3.6 promises a path this task does not provide.** §3.6 line 152: "the DSEL-8 sweep JSON, **named
   in that task**". DSEL-8 names no JSON path. DSEL-11's startup constant-read (line 428) therefore
   cannot be implemented.
4. `verify:` is artifact-existence plus a prose judgment ("the chosen default justified by the
   measured curve").
5. **`{5,10,15,25,40}%` is prefixed "suggest"** — under the autonomous contract that is an
   unauthorised open choice; §3.5 covers *which fraction becomes the default*, not *which fractions
   get swept*.

### DSEL-11 — NOT-DISPATCHABLE

The binding requirement (line 428-429) is: "the driver READS §3.6's constants from their artifacts at
startup and FAILS LOUDLY if any is missing." Walking §3.6:

| Constant | Owning artifact as written | Resolvable path? |
|---|---|---|
| supervision regime | "the DSEL-1 diff table + finding note" | **No** — DSEL-1 never names the note's filename |
| selection-set fraction | "the DSEL-8 sweep JSON, named in that task" | **No** — DSEL-8 names no JSON |
| D-PERINPUT data-limited flag | "same JSON" | **No** — same |
| router hidden/depth/epochs/lr | "the DSEL-9 sensitivity JSON, else … `distilled_router.py:57-60`" | **Half** — the fallback resolves (verified: `:57` `DEFAULT_TOLERANCE`, `:58` `DEFAULT_HIDDEN`, `:59` `DEFAULT_N_EPOCHS`, `:60` `DEFAULT_LR`); the JSON does not |
| labelling tolerance | "same" | **Half**, same |
| depth ladder / max depth | "the per-cell result JSON that recorded the bind" | **No** — and the *initial* ladder has no owner at all |

A worker cannot write a startup check against four constants with no paths. Additional gaps: the real
datasets are "frozen in the spec" that *this same task writes* (line 427) — circular, and it means the
dataset choice is an unconstrained mid-run decision covered by no §3.5 default; LightGBM/linear/NN
baseline hyperparameters are unspecified; only the "abort on a missing constant" half of `verify:` is
runnable.

### DSEL-0 — DISPATCHABLE-WITH-FIX

- `verify:` clause 2 is `grep -cin "F5c\|feedforward depth" flexnn-core.md` "returns only pointer
  text". `grep -c` returns an integer; an integer cannot show that the remaining text is a pointer.
  No expected count is given. Not checkable.
- Clause 1 (`grep -n "depth-selection.md" MASTER.md` shows index **and naming key**) — grep output
  cannot show that the naming key "states no definition of its own". Currently the file is absent from
  `MASTER.md` (verified: `grep -n "depth-selection" docs/plans/capacity_programme/MASTER.md` → no
  hits), so the task is genuinely open.
- Clause 3 (`depth.md` unmodified via `git diff --stat`) passes trivially if a worker commits.
- **Ordering collision with DSEL-1** — see §3.

### DSEL-1 — DISPATCHABLE-WITH-FIX

The diagnosis it records is **CONFIRMED against primary evidence**:
- `depth_selection_toy.train_anytime` starts at `:442`; `:475` is the per-depth mean CE — the plan's
  "nested? yes" cell is correct.
- `depth_composition_toy.train_clf` starts at `:437`; `:487` is `loss = ce(net(x_tr), y_tr)` — the
  plan's "nested? no" cell is correct.
- The quoted numbers reproduce from the artifacts: `f5c_poscontrol_a5_seed0.json` `train_acc`
  0.96970 / `val_acc` 0.43240; `seed1` 1.0 / 0.74420.
- The schema defect is **CONFIRMED**: neither JSON contains `n_train`/`n_val` (keys are `arm,
  net_kind, group, seq_len, seed, n_classes, chance, rec_state_width, params, hyperparameters,
  fit_acc, stall_acc, fit_status, train_acc, val_acc, clip_engagement_rate, trustworthy_ce, ce_gate,
  trustworthy_acc, acc_gate, val_acc_trajectory, train_acc_trajectory, train_ce_trajectory`).

Gaps:
- The note's filename is never given, yet §3.6 makes it a constant-owning artifact (see DSEL-11).
- "clear the halt marker in `flexnn-core.md`" — **which one?** `flexnn-core.md` carries at least two
  live halts: `:235` ("FAIL ⇒ **HALT**") and `:269` ("F5c-c/F5c-d remain HALTED either way").
- `verify:` clause 3 — "a fresh positive-control JSON contains `n_train`/`n_val`" — silently requires
  **re-running the positive control** (`hyperparameters.max_epochs = 40000`), which the task's own
  Non-goals forbid ("Do NOT re-run the failed control"). Direct self-contradiction, and the only
  clause of the three that is mechanically checkable.
- No prove-it-fails clause for the schema fix (a one-line assertion on a smoke-run JSON would do).

### DSEL-3 — DISPATCHABLE-WITH-FIX (the plan's best task)

Both defects are **CONFIRMED**:
- DD1: `independent_weights_flexible_neural_network.py:306` and `:312` both contain
  `torch.linspace(3.0, 1.0, self.max_hidden_layers, …)`; the sibling's removal comment is at
  `flexible_neural_network.py:300-304` and the sibling now uses `torch.zeros(...)` (`:305`). Citation
  exact.
- DD2: `flexible_neural_network.py:386` `def predict(self, x, filter_data=True, inference_mode="soft")`
  vs `:502` `def predict_uncertainty(self, x, filter_data=True)` — no mode parameter. Citation exact.
- The `verify:` line is the only genuinely runnable one in the plan and the only one carrying
  prove-it-fails ("revert each fix in turn, re-run, show the corresponding test FAILING").

Fixes needed:
- **Verify scope is too narrow.** `IndependentWeightsFlexibleNN` is imported by
  `tests/test_nested_depth_strategy.py`, `tests/test_phase1_probabilistic_regression.py`,
  `tests/test_phase4_regression.py` in addition to `tests/test_phase2_flexible_nn.py`. Changing its
  depth prior changes its training behaviour; running one file can leave a green board over three
  broken ones.
- "show both file checksums unchanged" names no baseline to compare against — state
  `sha256sum` captured before the revert cycle.
- A precedent test already exists for the sibling (`tests/test_phase2_flexible_nn.py:137`
  `test_elbo_uniform_prior_does_not_collapse`); the task should name it as the template so the worker
  does not invent a weaker assertion.

### DSEL-6 — DISPATCHABLE-WITH-FIX

- The bootstrap procedure is unspecified: what is resampled (held-out examples? seeds?), how many
  resamples, with what seed. Two workers will produce two different selectors.
- "the held-out selection set" is never defined for this codebase — no path, no split function named.
- Collides in rule with the shipped router's relative tolerance (see DSEL-4 item 4).
- Positive: the `verify:` line is the second-best in the plan — a synthetic flat curve beyond depth *d*
  is a real mechanical test — but it is still described, not written as a command/test id.
- Write-set handling is correct: the plan flags `parallel: no (same file as DSEL-3)`.

### DSEL-7 — DISPATCHABLE-WITH-FIX

- Driver unnamed. "the frozen ladder" has no owning artifact (§3.6 assigns the ladder to DSEL-10/11,
  which run later).
- Toys, seeds, budget per depth, and "same training budget as the joint net?" are all unstated — and
  budget parity is the whole validity of a reference sweep.
- `verify:` is artifact existence ("one JSON per (toy, seed, depth)") plus "total training cost …
  recorded as a headline number" — no path, no command.

### DSEL-9 — DISPATCHABLE-WITH-FIX

- **Internal contradiction**: Spec says vary "the labelling tolerance"; Doctrine says "No change to
  the labelling rule's meaning" and "This task does not unfreeze it". Whether the tolerance may be
  swept at all is genuinely unclear.
- Grid is half-specified: router width/depth have values ("half/double/4× hidden, 1 vs 3 layers"),
  epochs and tolerance have none.
- "invariant" has no threshold — the §3.5 branch table forks on invariant-or-not, so the *whole
  autonomous branch* hinges on an undefined predicate.
- Results-JSON path unnamed though §3.6 depends on it.
- `parallel: yes` with DSEL-8 is claimed "if driven by separate scripts" — unverifiable in advance
  because neither script is named.

### DSEL-10 — DISPATCHABLE-WITH-FIX

- "agreement — does D-SHARED choose the depth D-SWEEP chooses" is not operationalised: exact match?
  within-one-rung? aggregated over seeds how? reported as a rate or a per-seed boolean?
- The anchor warning (§3.6 line 161-165) is well-posed and correctly binds here — good.
- `verify:` is artifact-existence again.
- Deps (DSEL-8, DSEL-9) are correct and complete.

### DSEL-12 — DISPATCHABLE-WITH-FIX

- `verify:` = "the skill's cold-read gate, plus a check that every constant in §3.6 appears with its
  study cited". The second half is checkable only if §3.6's artifacts have paths (they mostly do not).
- Correctly cites MASTER Decision 10 (`MASTER.md:90-91` — verified: `research-report` skill, authored
  as the user, no AI/tool provenance).

### DSEL-5 — not a task

Correctly marked "not dispatchable here", but it sits in the §4 order line
(`… → DSEL-4 → DSEL-5 → DSEL-6 …`) as though a dispatcher should schedule it. An autonomous root
following the order line will try to dispatch a task with `verify: n/a`.

---

## Section 3 — Plan-level findings

### 3.1 Citation integrity — good, with two errors of *meaning*

Every `file:line` I opened resolves and, with two exceptions, says what the plan claims. The
mechanical gate confirms this: `~/dev/.venv/bin/python -m pytest
docs/plans/capacity_programme/test_plan_gates.py -q` → `6 passed in 0.06s`.

**Verified correct:** `flexible_neural_network.py:300` · `:29-38` · `:133` ·
`independent_weights_flexible_neural_network.py:306`,`:312` · `distilled_router.py:57-60` ·
`depth_selection_toy.train_anytime:475` · `depth_composition_toy.train_clf:487` ·
`verdict_per_input_depth.md:40`,`:70-71` · `depth.md:24`(-28),`:34`,`:136` ·
`probreg_kselection.md` §3.2 (line 277 heading; line 290 "twice an estimated standard error"; line 294
bootstrap) · `ff_depth_protocol_repair_spec.md` §2 (line 111, "Escalation ladder (MASTER Decision
16)") · MASTER Decisions 10, 13, 14, 17 (`MASTER.md:90, :97, :104, :123`) · DD3's "imports nothing
from `automl_package/models/`" (no such import in `depth_selection_toy.py`) · DD4 (no `fit_router`,
no `inference_mode` in the independent-weights class).

**Error 1 — MASTER Decision 15 is miscited.** Plan line 56:
> **Confound doctrine (MASTER Decision 15):** an arm differing from its comparator in more than one
> respect is NOT dispatchable.

`MASTER.md:110-116` Decision 15 is **"Protocol parity when reusing a substrate"** — "A battery
reusing a toy that produced a certified result must **diff its training loop** against the loop that
produced that result; every difference is justified in writing IN THE TASK before the run." Related,
but not the same rule, and §1 hangs the strand's stated reason for existing on it ("This strand exists
because that rule was broken"). Ironically, the actual Decision 15 is the rule DSEL-1b breaks (§2a).

**Error 2 — `verdict_per_input_depth.md:70` "sandwich".** Detailed in §2 DSEL-1b(a). The line
resolves; the schedule the plan infers from it is contradicted by both certified training loops.

**Weakly supported:** DD3's "Nine example scripts carry their own router implementations in total"
(line 110). I count 6-9 depending on what counts (`capacity_ladder_k6.py:75`, `capacity_ladder_t2.py:233`,
`capacity_ladder_s2.py:86`, `depth_selection_toy.py:607`, `capacity_ladder_h1.py:386`,
`sinc_width_experiment.py:251/438`, plus `_variational_em*.py` selectors and `moe_regression.py`'s
gating). Not falsified; not checkable as stated.

### 3.2 Branch coverage — decisions covered by neither a §3.5 default nor a HALT

Each of these is a place the unattended run stalls or improvises:

1. **The positive-control pass bar** (DSEL-1b, DSEL-2, DSEL-4, DSEL-10 all invoke "the certified
   bar"). Not a constant, not a default, not a halt.
2. **Sandwich vs sum-all** once the worker notices the contradiction in §2a. HALT rule 3 covers
   changes to §1's definitions; this is a §4 default, so no halt fires — the worker just picks one.
3. **2×SE vs relative-δ_tie selection rule** (DSEL-4/6/9). Same: silently resolved by whoever runs first.
4. **Which of `flexnn-core.md`'s halt markers DSEL-1 clears.**
5. **All unnamed driver/results paths** (DSEL-2, 7, 8, 9, 10, 11) — six write-set decisions taken by
   workers, defeating the write-set guard and the `deps:`-plus-write-set wave partition that
   `MASTER.md:144-146` requires.
6. **Real datasets for DSEL-11** and their preprocessing/split protocol.
7. **Baseline hyperparameters** (LightGBM, plain NN) for DSEL-11 — tuned or defaults? Not stated;
   this decides the headline comparison.
8. **Seeds** — no seed count is fixed anywhere in the plan. The certified work used 2-3.
9. **What "invariant" means** in DSEL-9, on which the §3.5 branch table forks.
10. **What to do if the feed-forward arm simply fails** (DSEL-2). §3.5 covers "positive control
    fails" → HALT, but a *negative primary result* on the FF arm has no branch — is that a finding to
    record and continue with (DSEL-6..DSEL-12 then measure selection machinery on an arm with no
    signal), or a stop? Given the strand's whole point is the FF claim, this is a live gap.
11. **Ceiling-raise on the FF arm** requires regenerating data, because the untied net is pinned at
    `d = seq_len` (`depth_composition_toy.py:373-374`). The §3.5 default ("extend the ladder one rung,
    re-run that cell") reads as cheap and is not.

### 3.3 HALT conditions — ambiguities

- **Rule 1 has already fired and the plan does not say so.** The f5c positive control failed
  (val 0.432/0.744 vs `flexnn-core.md:235`'s ≥0.90 bar). DSEL-1 closes it by user ruling — fine — but
  DSEL-1b and DSEL-2 re-run the same control under a new scheme. If it fails again, is that rule 1
  (HALT) or the §3.5 row "DSEL-1: other differences survive → rebuild the contrast one variable at a
  time"? Both plausibly apply.
- **Rule 2, "incoherent rather than merely negative — a broken instrument"**, has no test. It is the
  judgment a zero-context worker is least equipped to make and the one most likely to be resolved by
  continuing.
- **Rule 4 "irreversible or outward-facing (… committing)"** conflicts with `MASTER.md:142`
  ("Update the strand in the SAME TURN work lands") and `:92` ("Commits are user-gated"). Workable,
  but the run will accumulate uncommitted state across a long unattended session with no instruction
  about checkpointing.
- Rules 1 and 5 can collide: a re-run of the certified selection numbers through the package router
  (DSEL-4) that *fails* is simultaneously "positive control fails" (rule 1) and "would reopen
  `G-DEPTH = PASS`" (rule 5). Same action, but the plan should say which framing the escalation uses.

### 3.4 Constants — 4 of 6 dangle

Tabulated in §2 DSEL-11. Net effect: DSEL-11's central binding requirement ("FAILS LOUDLY if any is
missing") is unimplementable as written, and the §3.6 "Feed-forward rule (binding): if DSEL-11 runs at
a value not justified by the artifact named here, its results are **not reportable**" makes the whole
downstream report unreportable by construction. Additionally the **depth ladder's initial value** has
no owner at all — only its post-ceiling-raise value does.

### 3.5 Ordering

- **Contradiction, DSEL-0 vs DSEL-1.** DSEL-0 (line 185) moves "the F5/F5b/F5c block out of
  `flexnn-core.md` into this file's history, leaving a pointer". DSEL-1 (line 219-220, and its write
  set) then clears "the halt marker in `flexnn-core.md`". The halt markers live *inside* the F5c block
  (`flexnn-core.md:235`, `:269`) that DSEL-0 just removed. Running in the stated order, DSEL-1 edits a
  region that no longer exists.
- **DSEL-2's `deps:` omits DSEL-1b** while its spec requires it (§2 DSEL-2 item 2). This is the one
  outright dependency error, and it is on the strand's primary claim.
- **DSEL-5 in the order line** though it is explicitly not dispatchable.
- No cycles. Everything else (DSEL-6/7 → DSEL-8/9 → DSEL-10 → DSEL-11 → DSEL-12) is consistent, and
  the "DSEL-8/DSEL-9 must precede DSEL-10/DSEL-11" rationale (line 175-176) is sound.
- **Parallelism claims:** DSEL-3 `parallel: yes` vs DSEL-1/DSEL-2 — true (disjoint). DSEL-3 vs
  DSEL-1b — **unverifiable**, DSEL-1b's write set is the directory `tests/`. DSEL-8 ∥ DSEL-9 —
  unverifiable, both scripts unnamed. DSEL-6 `parallel: no (same file as DSEL-3)` — correct.

### 3.6 The "usefulness" column

Every task feeds a named consumer: DSEL-1→1b; 1b→2; 3→4/6; 4→6/7; 6/7→8/9; 8/9→10; 10→11; 11→12.
No orphaned work. This dimension is the plan's strongest.

---

## Section 4 — The three changes that would most increase the chance of correct unattended execution

**1. Fix DSEL-1b's two defaults before anything downstream runs — they are refuted by the code the
plan itself cites.** Replace "sandwich" with the sum/mean-over-all-rungs scheme that
`depth_graded_toy._train_mixed:148` and `depth_selection_toy.train_anytime:475` actually implement (or
keep sandwich and register it as a Decision-15 protocol deviation with a control arm), and demote
`shared_readout` from a default to a pre-registered arm for the feed-forward mechanism, because the
verdict's own mechanism (`verdict_per_input_depth.md:40-43`) attributes shared-readout success to
weight sharing, which the feed-forward arm does not have. Left as-is, DSEL-1b/DSEL-2 produce a result
that cannot be attributed — the precise failure mode the strand was created to fix.

**2. Replace every prose `verify:` line with a runnable command plus an expected observable, and name
every write-set path.** Concretely: name each driver and each results JSON (DSEL-2, 7, 8, 9, 10, 11),
name DSEL-1's note file, name DSEL-1b's test file, and give each `verify:` a
`python … && test -f … && python -c "assert ..."` form the way DSEL-3's does. Today only DSEL-3 is
mechanically checkable, so the programme's stated root cause ("done meant the output files existed,
not that `verify:` was executed") is structurally unfixed in 12 of 13 tasks. Naming the paths also
makes the `deps:`-plus-write-set wave partition (`MASTER.md:144-146`) and the write-set guard actually
enforceable.

**3. Freeze the numbers the run keeps asking for, in §3.6, with resolvable paths.** Minimum set: the
positive-control pass bar (value + metric + seed count), the seed count for every battery, the initial
depth ladder and substrate parameters, and the one selection rule (2×bootstrap-SE vs relative
δ_tie = 0.25 — `distilled_router.py:57` and `depth.md:239` currently say the latter, §1b says the
former). Then give the four dangling §3.6 constants real filenames so DSEL-11's startup check can
exist. Without this, DSEL-11 is unimplementable and its binding "not reportable" clause voids the
report the strand is for.

**Secondary (cheap, do them too):** fix the Decision-15 miscitation in §1; swap DSEL-0 and DSEL-1's
order (or move the halt-clearing into DSEL-0) so DSEL-1 does not edit a removed region; add DSEL-1b to
DSEL-2's `deps:`; widen DSEL-3's verify to the four test files that import
`IndependentWeightsFlexibleNN`; and add a §3.5 branch for "the feed-forward arm shows no benefit".
