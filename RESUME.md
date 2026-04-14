# Next Steps

**Status**: Phase 6 complete. 71 tests passing (all green). Research plan drafted, mathematical guide created.

**Awaiting user review of:**
1. `docs/research_plan.md` — revised plan (reviewed through §1, §2+ pending review).
2. `docs/mathematical_guide.tex` / `.pdf` — 24-page LaTeX reference. Compile: `pdflatex mathematical_guide.tex` (3 passes).

## Key documents

| File | Purpose |
|---|---|
| `docs/research_plan.md` | Two-paper publication roadmap: bugs, benchmarks, metrics, baselines, astrophysics |
| `docs/mathematical_guide.tex` | Complete math spec: all models, UQ methods, selection strategies, losses, metrics |
| `docs/benchmarks.md` | Current empirical results (Phases 1–5) |
| `docs/architecture_analysis.md` | SOTA comparison + historical bug audit |
| `ARCHIVE.md` | Per-phase change details |

## Open issues (architectural, not bugs)

- ClassReg `NN` mapper: poor results (two-stage training limitation; use ProbReg instead)
- ELBO + Gumbel for dynamic k: noisy KL gradients (use SoftGating instead)

## Two-paper structure (from research plan)

- **Paper A (ProbReg)**: classification-bottleneck probabilistic regression via law of total variance. Headline: photometric redshift on LSST DC2/SDSS.
- **Paper B (FlexibleNN)**: input-dependent depth selection with ELBO regularization. Headline: galaxy cluster mass on IllustrisTNG.
- **Paper C (future)**: FlexibleNN depth gating applied to FT-Transformer blocks.

## Phase 6 completed (this session)

- **N1**: FlexibleNN ELBO depth prior `arange` → `linspace(3,1)` (both variants)
- **N2**: STE gradient path rewritten to weighted-sum pattern in `_hard_selection_logic`
- **N3**: ECE rewritten to PIT-based formulation (Kuleshov 2018)
- **N4**: Added `log(2π)` to `beta_nll_loss` for consistency with `nll_loss`
- **N5**: Tree NLL Hessian switched to Fisher information (constant 0.5)
- **N6**: Removed double `build_model` in `base_pytorch.py`
- **N7**: `NoneStrategy` layer selection shape `(B, max+1)` → `(B, max)`
- **triton-xpu**: Runtime patch for broken triton stub (`_disable_broken_triton`)
- **Spline mapper**: Added default smoothing (`N * var(y)`) and prediction clamping
- **install.sh**: Auto-detects CUDA > XPU > CPU, installs correct torch backend

## Execution roadmap

Phase 7: Metrics + visualizations → Phase 8: Baselines → Phase 9: Benchmarks + ablations → Phase 10: Photo-z → Phase 11: Cluster mass → Phase 12A/B: Paper drafts. See `docs/research_plan.md` §8 for details.
