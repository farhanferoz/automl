# RETIRED 2026-07-17: depth strand old Tasks D2/D3/D4 (Phase-II battery spec)

Retired by the user-ratified rescope (see `../depth.md` RESCOPE banner). Rationale: the D1b
graded battery answered or dissolved what this protocol was built to test — the transfer
prediction it would adjudicate was refuted 3/3 by D1b directly, and the S5 toy has no
depth-selection problem (per-input depth is dictated by input length), so the sandwich/selector/
deploy machinery certifies nothing here. The surviving open questions moved: LEARNED halting →
`../width-depth.md` J0 and the FlexNN strand (M3+); anytime-prefix depth nesting → J0's design
inputs. The old D4 gate rule is superseded by the rule inside `../depth.md` Task D5.

Text below is verbatim from `depth.md` as of retirement (citations in this file are frozen
history and exempt from the citation gate).

---

### Task D2: depth harness (mirror of the width WP-1)

**Files:**
- Create: `automl_package/examples/nested_depth_net.py` — `NestedDepthNet` (shared blocks +
  ONE shared output head over every depth prefix), `SharedBlocksPerDepthHeadNet` (per-depth
  heads — the #2 analog), `IndependentDepthNet` (per-depth disjoint nets — the #3 analog);
  identical `forward_depth(x, d)` / `all_depths_forward(x)` interface; MSE-only zero-`log_var`
  contract as in `nested_width_net.py:221` (`SharedTrunkPerWidthHeadNet` docstring pattern).
- Create: `automl_package/examples/kdropout_converged_depth_experiment.py` — clone the width
  driver's structure (`kdropout_converged_width_experiment.py`: sandwich-over-depths schedule,
  per-depth `ConvergenceTracker` from `automl_package/examples/convergence.py`, whole-net
  checkpointing for shared arms per its `_train_kdropout_to_convergence` docstring, `--arch/
  --toy/--tag` flags, collision-free summary names, `--selftest`/`--smoke`).

**Orchestration:** parallel: no · deps: D1 · tier: sonnet · scale: static · shape: execution ·
verify: `AUTOML_DEVICE=cpu ~/dev/.venv/bin/python automl_package/examples/kdropout_converged_depth_experiment.py --selftest` exits 0; ruff clean on both new files

- [ ] Step 1: nets + prefix selftests (the depth analogs of `nested_width_net.py` selftests
  (a)–(c): prefix invariance, block-nesting agreement, per-depth vs all-depths consistency —
  note depth blocks CHAIN through activations, so there is no cumsum shortcut; all-depths is a
  loop caching intermediate activations, the `NestedStrategy.all_depth_outputs` pattern).
- [ ] Step 2: driver clone; bars REUSED from `sinc_width_experiment.py` (`_fit_bar_mse`,
  `_curve_shape_gate_mse`, `_recovery_bar`, `_deploy_bar_mse` + W7's val-selected variant) —
  they are metric-agnostic over a `{level -> err2}` table; do not re-implement them.
- [ ] Step 3: selftest + smoke both arms; ruff.

**Non-goals:** no changes to width files (bars are imported, not edited); no FlexNN/package
model changes — this strand runs on examples-level nets; the FlexNN port happens in the
flexnn-moe strand AFTER G-DEPTH.

### Task D3: the 3-arm depth battery

**Files:**
- Create (by runs): `.../D_KDROPOUT_CONVERGED/d_kdropout_converged_summary_{nested,per_depth_head,independent}_mse.json`

**Orchestration:** parallel: no · deps: D2 · tier: sonnet · scale: static · shape: execution ·
verify: 3 summary JSONs exist; per-seed `fit_bar`/`dial_bar`/`curve_gate` present

- [ ] Step 1: seed-0 pilot, all 3 arms (measures wall-clock; one §5-constants recalibration
  allowed HERE if the depth toy's scales demand it, then frozen — record it in this file).
- [ ] Step 2: full battery, seeds 0/1/2, detached, ≤4 concurrent; trajectory-check per depth.
- [ ] Step 3: if the SHARED-readout arm fails its bars, run the 120k-epoch early-stop-OFF
  confirmation (the verdict-doc §2.1 template) before any verdict — the transfer prediction
  must not rest on an endpoint.
- [ ] Step 4: RESULT-lines → JSONs.

### Task D4: G-DEPTH verdict  ⛔ gate (orchestrator/opus)

**Files:**
- Create: `docs/depth_capacity/verdict_variable_depth_mse.md` (mirror the width verdict's
  structure: bars recap, per-seed tables, mechanism, claim ledger, file manifest)
- Modify: this file (gate row) + `MASTER.md` (strand 4 unblocked or contingency)

**Orchestration:** parallel: no · deps: D3 · tier: opus · scale: static · shape: discovery ·
verify: verdict doc exists; gate row filled with evidence pointers

**Pre-registered rule:** G-DEPTH = PASS iff the per-depth-head arm passes fit + dial on ≥2
trustworthy seeds. Transfer prediction CONFIRMED iff additionally the shared-readout arm fails
fit on ≥2 trustworthy seeds (early-stop-OFF-confirmed). Both-shared-arms-pass → prediction
refuted, record honestly (depth ≠ width mechanically — still a publishable contrast). All arms
fail → toy problem despite probes; back to D1 with a fresh candidate, once; then escalate.

**Gate decision:** *(superseded — see `../depth.md` Task D5)*
