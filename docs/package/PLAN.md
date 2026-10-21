# Package — stream plan

**Status: maintenance.** The AutoML library itself (`automl_package/`): BaseModel, hyperparameter
search, SHAP feature selection, conformal prediction, the model zoo, the install and the docs.

Open items:
- **`README.md` is stale.** Last rewritten in April 2026, it documents an `AutoML` orchestrator class
  removed in September 2025 (commit `ed84fca`) and does not mention the FlexNN or ProbReg work.
- **`docs/package/mathematical_guide.tex` predates the July changes**: it has no fixed σ, nested
  ladder or k-dropout.
- **No PyTorch environment for this project on the DGX.** The workspace environment pins an Intel
  build with no wheels for this machine; a project-local environment (CPU-only is enough for the
  tests) is needed before the planned move of the experiment scripts can be verified here.
- Six test files are run directly rather than through repo-wide pytest collection, which their
  docstrings say an omegaconf conflict breaks.

Documents: `docs/package/mathematical_guide.tex` and its PDF; `docs/package/history/` (the April
architecture analysis, implementation plan, benchmarks and a session journal).
