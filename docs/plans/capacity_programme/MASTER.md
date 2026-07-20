# FlexNN programme (flexible-capacity networks)

Index + rules ONLY. Per-strand detail lives in the strand files. Read this file plus the ONE
strand you are working — that is the whole context. Fix this plan IN PLACE; never fork a dated
copy (gate: `test_plan_gates.py`).

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:subagent-driven-development`
> (recommended) or `superpowers:executing-plans`, driven from a strand file. Tasks carry
> checkbox steps and Orchestration lines; dispatch waves are computed from `deps:` + write-set
> overlap.

## Strand index & priority

| # | Strand file | Delivers | Gated on | Dispatchable now? |
|---|---|---|---|---|
| 1 | `width-cert.md` | Certification of arch #2 + seam/robustness analyses → **G-WIDTH** | — | **DONE — G-WIDTH PASS (2026-07-16)** |
| 2 | `probreg-report.md` | Report (a): ProbReg fixed-k & dynamic-k vs baselines on toys | `shared/metrics-accounting.md` T-S1..S3 | **DONE — P0-P3 done, report (a) delivered** (`docs/reports/probreg_toys/report_a_probreg_toys.{pdf,md}`) |
| 3 | `depth.md` | S5 substrate (D1 **DONE**) + selection-without-oracle (D8) → **G-DEPTH = D5 ∧ D8** | **G-WIDTH ✓** | **DONE 2026-07-17 — G-DEPTH = PASS.** Substrate D5 (3/3) ∧ selection D8b (2/2: S1–S4 pass, S5 surface covariate 100%/Option-A). Verdict `verdict_per_input_depth.md` §12; D8b toy AS-RUN rebuild (L10/involutions/shared-readout, flagged). old D2/D3 retired. → `width-depth.md` J0 |
| 4 | `width-depth.md` | Joint 2-D capacity dial + transformer halting → **G-JOINT** | **G-DEPTH** (D5 ∧ D8) | **J0 RAN 2026-07-17 — J-1/J-2 DEAD (multi-track fold substrate fails S1); G-JOINT BLOCKED, J-3 redesign ESCALATED to user** (post-mortem in strand) |
| 5 | `flexnn-moe.md` | MoE build (M0-M2 early) + reports (b), (c) | build: none; comparisons+reports: **G-JOINT** | **DONE — M0-M2 DONE 2026-07-16**; M3 rescoped as flexnn-core F6; M4/M5 superseded by F7 (user 2026-07-18) |
| 6 | `flexnn-core.md` | Package refactor + FF-depth pilot + unified report | — | **yes (unattended orchestration in progress)** |

Gates are decision points written in the owning strand (evidence-backed verdict + branch).
Priority: flexnn-core (6) waves go first. `width-depth.md` J0 is **PARKED** pending user
Option 1/3 decision. *(Strands 1, 2, 3, and 5's M0-M2 are complete; live forward order updated
2026-07-18.)*

## Naming key

- **FlexNN** = the umbrella for ALL per-input capacity work; strands are its width/depth/joint/MoE/core
  facets; the package family is `flexible_neural_network.py` + `flexible_width_network.py` (F2).
- **#1 / #2 / #3** — width architectures: #1 `NestedWidthNet` (shared trunk + shared readout,
  fails), #2 `SharedTrunkPerWidthHeadNet` (shared trunk + per-width heads, **certified winner —
  G-WIDTH PASS 2026-07-16**),
  #3 `IndependentWidthNet` (K disjoint sub-nets, positive control). All in
  `automl_package/examples/nested_width_net.py`.
- **G-WIDTH / G-DEPTH / G-JOINT** — the three programme gates.
- **shared-k / variable-k (ProbReg)** — fixed `n_classes` model vs dynamic-k
  (ELBO + SoftGating). Treated as two distinct models in all report tables.
- **Ledger** — result JSONs under
  `automl_package/examples/capacity_ladder_results/` (width/depth) and the per-strand
  results dirs named in each strand. The plan holds POINTERS to ledger files, never copied
  result numbers. A line carrying a result must be marked `RESULT:` and name its `.json`.

## Decision register (settled 2026-07-16, user-approved — do not re-ask)

1. **Charter verdicts to date** — strict nesting (#1) fails; break = shared READOUT; #2 is the
   **certified** architecture of record (**G-WIDTH PASS 2026-07-16**). Evidence:
   `docs/width_mse_2026-07-16/verdict_variable_width_mse.md` (§10 certification addendum; manifest §8).
2. **MSE-only** for all capacity strands (width/depth/joint). Variance stays parked.
   *(Amended 2026-07-17: binds the WIDTH strand only. The depth strand is CE classification per
   the `depth.md` PIVOT; the joint strand's metric follows the toy J0 adopts.)*
3. **Reports are toys-only.** UCI/real-data belongs to the later Paper A/B roadmap
   (`docs/research_plan.md`), not these reports.
   *(**AMENDED 2026-07-20, user live.** Binds the WIDTH/DEPTH capacity reports only. The ProbReg
   k-selection work is now explicitly IN scope for real data + baselines: the report must carry
   baseline comparisons (XGBoost / LightGBM / CatBoost / standard NN) and real-world datasets,
   and must report **shared-k vs variable-k as two distinct models** per the Naming key. Dataset
   candidates already sourced in `docs/research_plan.md` §6 Layer 2/3.)*
4. **Hetero-NLL default** (strand 2): minimal fix if one exists cheaply; otherwise report (a)
   documents the limitation honestly and variance stays parked. Failure there does NOT unpark
   the variance programme.
5. **Depth/joint strands are structured around #2's pattern** (shared representation +
   per-capacity readout heads), conditioned on G-WIDTH; written contingencies apply if it fails.
   *(Amended 2026-07-17: holds for WIDTH only — the depth transfer prediction was REFUTED 3/3;
   shared readout is the depth winner. Joint prior = per-width heads × depth-shared readout;
   see `depth.md` and `width-depth.md` Settled inputs.)*
6. **Depth starts with the toy, not architecture** — a depth-hungry-but-GD-learnable target
   with analytic floor; kill criterion + escalation if unconstructible.
7. **MoE config**: 8 experts, top-2 routing primary; top-1 ablation; load-balancing auxiliary
   loss; param/FLOP-matched to FlexNN/width nets. Conventions verified against Shazeer 2017 /
   Switch / Mixtral before the build brief freezes (task M1).
8. **Reports supersede** the old parked "REPORT-2 + `mathematical_guide.tex` fold-in" item.
   Math is LIFTED from `docs/mathematical_guide.tex`, not re-derived.
9. **Trajectory discipline** (binding on every training conclusion): full per-width/depth
   held-out trajectories, convergence flags trajectory-verified, `hit_cap=False` required;
   load-bearing verdicts get an early-stop-OFF confirmation run at ≥4× the self-terminated
   budget. No conclusion from an endpoint. (Case law: verdict doc §2.1.)
10. **Report authorship**: `research-report` skill; authored as the user; full mathematical
    rigor; NO AI/tool provenance anywhere committed or shared.
11. **Commits are user-gated.** Nothing staged without explicit go.
12. **Worker tiering**: haiku = mechanical/complete-spec (tables from JSONs, formatting, path
    cleanup); sonnet = default build/battery/draft against decision-complete specs;
    opus/main = discovery (root-cause, toy construction), verdicts, math review, gate decisions.
    Orchestrate on the strong model; execute on the cheap one.
13. **Selection = post-hoc DISTILLATION (user, 2026-07-17).** Per-input capacity choice (width,
    depth, joint, transformer per-token depth) is learned by distilling a router from held-out
    per-capacity error tables — the certified width mechanism
    (`automl_package/examples/sinc_width_experiment.py::_fit_selector_mse`). In-sample /
    in-training selection is never the primary (failed for width; FlexNN ELBO depth-select
    refuted, M0); it may appear only as a labeled comparison arm after the distilled primary
    passes.
14. **Positive-control gate (2026-07-20).** Any battery containing a known-good arm runs that arm
    **FIRST, alone**, and it must reproduce its certified result at the same bar on **every** seed
    before further compute is spent. A failed positive control **HALTS** the battery: the protocol
    is then the defect under investigation, not the unknown arms. *(Case law: F5b — the certified
    `RecurrentComposer` collapsed on seed 0 (val acc 0.830 → 0.097) and the battery ran to
    completion anyway, leaving all 28 runs unreadable.)*
15. **Protocol parity when reusing a substrate (2026-07-20).** A battery reusing a toy that
    produced a certified result must **diff its training loop** against the loop that produced
    that result; every difference is justified in writing IN THE TASK before the run. *(Case law:
    F5b ran A5/L=10 through `automl_package/examples/depth_composition_toy.py`, which applies no
    gradient clipping, while the certified L=10 work ran through
    `automl_package/examples/depth_selection_toy.py`, whose `GRAD_CLIP_MAX_NORM = 1.0` is
    commented "L=10 needs clipping to stay GD-trainable".)*
16. **Optimization is exonerated BEFORE architecture is blamed (2026-07-20).** No arm is recorded
    as an architecture failure until either it is shown to fit the **training** set, or a
    documented escalation ladder (LR sweep → clipping → warmup → init scheme → normalization) has
    been run. An arm low on **both** train and val is **under-fit** — an optimization finding,
    never a generalization verdict. *(Generalizes the width-control rule in
    `docs/depth_capacity/ff_depth_toy_spec.md` §6 to every arm.)*
17. **The convergence gate must be computed on the metric the bar reads (2026-07-20).** Where a
    pre-registered bar is read on metric M, the `trustworthy`/`diverged` flags are computed on M
    (or on M *and* the loss, with both reported). *(Case law: F5b gated on val cross-entropy while
    its bars read val accuracy — cells whose accuracy had cleanly plateaued were quarantined for
    CE overconfidence, and the reverse error is equally possible.)*

## Rules (cache discipline)

- **Pointers, not copies.** Result numbers live in ledger JSONs / report artifacts; strands cite
  paths. `RESULT:` lines must name their `.json` (gate-checked).
- **DONE only with evidence attached** — artifact path or ledger line, never a worker claim.
- **DONE = the task's own `verify:` line EXECUTED, with its output shown.** File existence is NOT
  evidence; a `verify:` line with three conditions needs all three checked. *(Case law 2026-07-20:
  F5b, F6 and F9 were all carried as done — battery invalid, driver never executed, tests
  failing.)*
- **No claim without an artifact.** Any result or diagnosis claim written into a strand, `RESUME`,
  `CHANGELOG` or a report cites a path that resolves on disk. Prose-only findings do not exist.
  *(Case law: a "grad-norm diagnosis / Xavier init doesn't rescue it" conclusion survives only as
  narrative — no script, log or JSON anywhere in the repo.)*
- **Update the strand in the SAME TURN work lands.** Deferred = stale.
- **Verify a task is OPEN before dispatching** (open the artifact / grep the symbol first).
- **Single writer:** workers return findings; the ORCHESTRATOR writes strand files and ledgers.
  Fan out on compute; serialise writes. Write sets are declared per task; wave partition =
  `deps:` + write-set overlap.
- **No unverified citation:** every `file:line` in a strand was opened when written; re-verify
  before building on one (they rot).
- **Environment:** `AUTOML_DEVICE=cpu` on every run (shared 22-core box; ≤4 concurrent heavy).
  Heavy runs detached:
  `setsid nohup systemd-inhibit --what=idle:sleep:handle-lid-switch env AUTOML_DEVICE=cpu OMP_NUM_THREADS=4 ~/dev/.venv/bin/python -u <script> ... > <scratchpad>/<log> 2>&1 &`
  — watch the summary FILE from the main thread, never a subagent.
- **Venv repair** (if `~/dev/.venv` loses deps again): surgical `uv pip install` of the missing
  declared deps only; `uv sync` is blocked by a mis-scoped guard hook; torch must stay XPU.

## Mechanical gates

Run from repo root:
`~/dev/.venv/bin/python -m pytest docs/plans/capacity_programme/test_plan_gates.py -q`
- **Citations**: any `path/with/slash(.py|.md|.json|.tex|.sh)` cited in MASTER/strands/shared
  must resolve on disk (`file:line` → line must exist). `archive/` exempt.
- **Dated names**: no date-shaped filename in this plan dir (`archive/` exempt).
- **Numbers**: every `RESULT:` line must contain a `.json` pointer. (This gate proves existence
  of a pointer, NOT currency of the number — the newest ledger entry wins only if you look.)
- Baseline is shrink-only: `gates_baseline.txt` lists grandfathered violations; the suite fails
  if new ones appear or the baseline grows.

## History (frozen — never dispatch from)

- `docs/plans/width_mse_2026-07-16/EXECUTION_PLAN.md` — width-MSE programme, COMPLETE
  (WP-0..5). Its §5 bars are re-used by reference throughout this programme.
- `docs/width_mse_2026-07-16/verdict_variable_width_mse.md` — the width verdict + file manifest.
- `docs/plans/width_dial_2026-07-11/`, `docs/plans/perinput_selector_2026-07-10/`,
  `docs/plans/capacity_ladder_2026-07-09/` — earlier completed programmes (NLL era).
- `docs/research_plan.md` — the two-paper roadmap (2026-04); its §1 bug ledger is PARTIALLY
  STALE (N1 fixed at HEAD; N3 appears fixed at HEAD — see strand 2 re-audit task).

## Corrections

- **2026-07-17, G-DEPTH CLOSED (both halves) = PASS.** D8b selection battery landed: 2 seeds, all
  trustworthy/`diverged=false`. **S1** full-T fit ≥0.90/stratum 2/2; **S2** make-or-break knee 2/2
  (ρ(T*,t*)=1.000/0.993, acc@t*−2=0 cliff); **S3** deploy mean-T 8.0/7.99 < best-fixed-10 with MSE
  *improving* (routing to t* beats full depth); **S4** oracle-free router (raw word only). **S5**
  surface MLP recovers t* at 100% → Option-A covariate, NOT a kill (answer f(x) not surface-computable;
  §3/D7 depth still width-irreducible). Selection verdict appended `verdict_per_input_depth.md`
  §10–§12. **D8b toy AS-RUN rebuild** (the D8a C1‴ trained to chance → root-caused): L16→10,
  five-cycle→A5-involution, per-T-head→shared-readout-on-running-product, n40k→3000, ceiling 0.2875→0.35
  — all reversible, charter intact, flagged for user veto. **G-DEPTH = D5 (3/3) ∧ D8 (2/2) = PASS →
  `width-depth.md` J0.** (Nothing committed — user-gated.)
- **2026-07-17, depth closing waves + substrate PASS + D8a sign-off.** D6 divergence guard landed
  (`convergence.py` `diverged` flag; calibration surfaced TWO genuine Z120 blow-ups, both quarantined,
  neither touches a bar). D7 param-matched wide-101 run landed (16,381 params, stalls 0.447 @ℓ10 →
  G2 holds at parity). **D5 substrate verdict = PASS 3/3** →
  `docs/depth_capacity/verdict_per_input_depth.md`. **D8a design signed off by user (Option A):**
  concealment dropped as a kill criterion (difficulty-of-detection is not the charter), surface-baseline
  control added, narrow-deep width-substitutable graded toy documented as the fallback. D8b (selection)
  now building; gradedness is the make-or-break. **G-DEPTH = substrate PASS ∧ (D8 pending).**
  *(superseded — G-DEPTH closed, see top entry)*
- **2026-07-17, depth D1b landed + rescope (user-ratified).** Graded battery: ALL pre-registered
  bars pass on 3/3 seeds with the ℓ=4 rung excluded (degenerate by design arithmetic — 128 train
  words vs 120 classes); ledger
  `automl_package/examples/capacity_ladder_results/D_TOY_PROBES/depth_graded_pilot_s5_seed0.json`
  + 5 siblings. Transfer prediction REFUTED 3/3 (Decision 5 amended); Decision 2 amended (CE for
  depth). Old D2/D3 retired → `archive/depth_d2_d3_retired.md`; verdict target renamed →
  `docs/depth_capacity/verdict_per_input_depth.md` (baseline −3/+1). Convergence gate certifies
  diverged runs as trustworthy (Z120 seed-1 shared_readout blow-up) → fix = `depth.md` D6.
  Learned halting moved to the joint strand (`width-depth.md` J0, autonomous protocol).
- **2026-07-16, depth D1 kill criterion.** All three `depth.md` §D1 candidates (gentle
  composition, hierarchical spline, 2-D multiplicative) exhausted the pre-registered probe bars
  on 2 seeds each with no full pass — see `docs/depth_capacity/depth_toy_negative_note.md` (probe
  JSONs: `automl_package/examples/capacity_ladder_results/D_TOY_PROBES/*.json`). The depth strand
  does NOT proceed to D2/D3 pending a fresh D1 candidate. D0's citations were re-verified at this
  same HEAD and still hold exactly (`output_layer` at `layer_selection_strategies.py:90,136,180`;
  `independent_weights_flexible_neural_network.py` still present) — no citation fix needed.
