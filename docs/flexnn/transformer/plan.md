# FlexNN — transformer sub-plan <!-- citecheck-ignore -->

The transformer and coding-model roadmap. Moved verbatim from the "Ultimate-aim roadmap (R-stages)" <!-- citecheck-ignore -->
section of the former `flexnn-core.md` on 2026-10-21. Analysis: `docs/flexnn/transformer/specs/`; <!-- citecheck-ignore -->
groundwork dossiers: `docs/flexnn/transformer/reports/`. Index: `docs/flexnn/PLAN.md`. <!-- citecheck-ignore -->

<!-- from docs/plans/capacity_programme/flexnn-core.md lines 182-229 -->
## Ultimate-aim roadmap (R-stages) — ratified outline 2026-07-18

Endpoint: a small Python-only instruction-level coding LM trained on DGX-Spark-class hardware
with a flexible-capacity (FlexNN-principled) architecture, benchmarked against dense + MoE
baselines at matched compute. Status flags: **SCOPED** = execution-level, dispatchable;
**PARTIAL** = key artifacts exist, a bounded design round remains; **OUTLINED** = direction +
gates known, needs its own spec/design round before any build.

- **R0 — FlexNN core (this strand, F0–F8): SCOPED.** The only fully execution-level stage.
- **R1 — G-JOINT joint dial: PARTIAL.** J-3/R1 spec + required fixes enumerated
  (`docs/flexnn/joint/specs/j3_toy_design.md` §9); decision DEFERRED to post-F5b (see waves) because
  a passing feedforward substrate may offer an entanglement-free joint toy (needs its own
  confound ledger — design round, spec-first).
- **R2 — sequence bridge: OUTLINED.** Weight-tied/looped sequence model + per-token DISTILLED
  halting on an algorithmic sequence task (UT-shaped; D8 router pattern). Needs: toy spec
  (task, bars, confound ledger) per [[feedback_toy_design_needs_reviewed_spec]].
- **R3 — transformer capacity dials at toy scale: OUTLINED.** Standard block first (baseline
  comparability); small MoD / MoE / MoDE reproductions; OUR distilled router as the comparison
  axis vs their in-training routers. Needs: M1-style convention freeze from primary sources +
  design doc.
- **R4 — probabilistically-grounded architectures: OUTLINED (research-shaped).**
  R4a deep literature review (deep-research fan-out; every anchor verified) — DELIBERATELY
  DEFERRED to this stage's gate: the adaptive-compute/MoE field moves fast, a review done now
  decays before it is consumed. R4b candidate constructions + falsifier toys — unscopable until
  R4a lands (by design, not neglect).
- **R5 — endpoint groundwork (research, CHEAP, runs EARLY — dispatchable alongside R0):**
  - **R5a dataset research: SCOPABLE NOW.** Inventory public Python instruction-level coding
    data; verified anchors to seed it (2026-07-18): CodeAlpaca-20k, Evol-CodeAlpaca-V1,
    Magicoder-OSS-Instruct-75K (arXiv:2312.02120), OpenCodeInstruct (large-scale). MUST verify:
    licenses (esp. The Stack v2 terms — NOT confirmed yet), contamination/dedup practice,
    dataset mix + token counts for a Python-only corpus. Deliverable: a sourced dataset dossier.
  - **R5b hardware envelope: SCOPABLE NOW.** DGX Spark verified 2026-07-18: GB10 (20-core Arm +
    Blackwell GPU, 6144 CUDA cores), 128 GB unified LPDDR5X, ~1 PFLOP FP4, 31 TFLOPS FP32, 4 TB
    NVMe (nvidia.com/dgx-spark; docs.nvidia.com/dgx/dgx-spark/hardware.html). Deliverable:
    trainable model-size/token-budget arithmetic (scaling-law based, sources verified), training
    precision support (NVFP4/BF16), wall-clock estimates for candidate model sizes.
  - **R5c eval harness choice: SCOPABLE NOW** (verify current standard Python coding evals +
    contamination policy; do not assert from memory).
- **R6 — the model: UNSCOPED (by design).** Pretrain + instruction-tune the small Python LM with
  the flexible-capacity architecture; benchmark vs dense + MoE at matched compute. Gated on
  R2–R5.
- **R7 — publications: PARTIAL.** Unified report = F7 (scoped); paper-form deliverables follow
  `docs/applications/research_plan.md` (amended by F8).

**Research-now-vs-later rule (ratified):** run R5a/R5b(/R5c) early — cheap, slow-decaying,
de-risks the endpoint and sizes every intermediate decision; defer R4a to its gate — expensive
and fast-decaying. R2/R3 design rounds happen at their own stages under the spec-first rule.


