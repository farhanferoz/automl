# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup & Commands

This project is part of the uv workspace at `~/dev/`. Always use the workspace venv.

```bash
# Use the workspace venv for ALL commands
~/dev/.venv/bin/python -m pytest tests/ -v --tb=short
~/dev/.venv/bin/python -m automl_package.examples.model_comparison
~/dev/.venv/bin/python -m ruff check automl_package/

# Install/sync (from workspace root, installs XPU torch on this machine)
cd ~/dev && uv sync --package automl-package

# Standalone install (auto-detects CUDA > XPU > CPU)
./install.sh                           # auto-detect best backend
./install.sh --dev                     # auto-detect + dev tools
./install.sh --cpu                     # force CPU
./install.sh --xpu                     # force Intel XPU
```

## Code Style

- Python 3.12. Line length: 180. Linter: ruff (see `ruff.toml`).
- Google docstring convention. Type hints on all new functions/methods.
- Use enums (in `enums.py`) instead of string literals for model options.
- `__init__.py` files suppress F401; `tests/` suppresses ANN, D, PLC0415.

## Device Support

Auto-detects: CUDA > XPU > CPU via `get_device()` in `utils/pytorch_utils.py`. All model code uses `self.device` from `get_device()` — no hardcoded device strings. The workspace venv on this machine has `torch+xpu` (Intel Arc). On other machines, the default PyPI torch works (CPU or CUDA).

## Architecture

### Model Hierarchy

All models inherit from **`BaseModel`** (`models/base.py`), which orchestrates fit/predict: data splitting, HPO (Optuna), CV, feature selection (SHAP), evaluation. Subclasses implement `_fit_single()` and `predict()`.

**`PyTorchModelBase`** (`models/base_pytorch.py`) extends BaseModel for neural networks, adding PyTorch training loops, optimizer setup, and learned regularization lambdas.

### Three Specialized Architectures

1. **ClassifierRegressionModel** (`models/classifier_regression.py`) — Two-stage: discretizes target into N bins (percentile-based), trains classifier (NN/XGBoost/LightGBM/CatBoost), maps probabilities back via Mapper. Best mapper: `LOOKUP_MEDIAN` (percentile bins + median lookup, weighted by class probabilities). Training is two-stage: classifier is frozen before mapper fits.

2. **ProbabilisticRegressionModel** (`models/probabilistic_regression.py`) — End-to-end: classification bottleneck + learned regression heads producing per-class (mean, log_variance), combined via Law of Total Variance. Trains jointly. Supports dynamic n_classes via `NClassesSelectionMethod` + `NClassesRegularization` (ELBO/penalty). Optional `loss_type="beta_nll"` (Seitzer 2022) and `target_transform="symlog"` for wide-range targets. Core network: `ProbabilisticRegressionNet` (`models/architectures/`).

3. **FlexibleHiddenLayersNN** (`models/flexible_neural_network.py`) — NN with learnable depth via n_predictor subnetwork. Variant: `IndependentWeightsFlexibleNN` has separate weights per depth. `predict(inference_mode="hard")` runs only the argmax-selected depth per sample for compute savings.

### Mappers (`models/mappers/`)

Transform class probabilities to regression. `LOOKUP_MEDIAN` is the most reliable non-NN mapper. `NeuralNetworkMapper` wraps regression heads but has known training issues (see RESUME.md). `nn_mapper_params` must nest head config under `"regression_head_params"` key.

### Selection Strategies (`models/selection_strategies/`)

Dynamic architecture for FlexibleNN (layer selection) and ProbabilisticRegression (n_classes). All extend `BaseSelectionStrategy`. Strategies: NoneStrategy (fixed), GumbelSoftmax, SoftGating, STE, REINFORCE — all functional.

**FlexibleNN depth control**: `DepthRegularization` enum (NONE, DEPTH_PENALTY, ELBO). ELBO uses KL vs geometric prior to prefer shallower depth on simple inputs.

**ProbReg dynamic k**: `NClassesRegularization` enum (NONE, K_PENALTY, ELBO). ELBO uses KL vs normalized prior (`linspace(3, 1, n_modes)`) — prior range is fixed regardless of n_modes to avoid steepness scaling. Best combo: **ELBO + SoftGating** (matches best fixed-k MSE, best noise correlation). ELBO + Gumbel has poor training dynamics due to noisy KL gradients. GumbelSoftmax uses deterministic softmax at eval time.

### Mixins (`models/common/`)

BoundaryLossMixin, BinnedUncertaintyMixin, RegularizationMixin, MonotonicityConfigMixin, MiddleClassPenaltyMixin.

### Conformal Prediction (`models/conformal.py`)

`ConformalWrapper` provides distribution-free split-conformal prediction intervals around any
fitted regression model. Uses finite-sample-corrected quantile (`ceil((n+1)(1-α))/n`).
Coverage holds across α values on heteroscedastic data (verified in Phase 4).

### Target transforms (`utils/transforms.py`)

`symlog(x) = sign(x) * log(1 + |x|)` and inverse `symexp`. Set `target_transform="symlog"` on
ProbReg for targets spanning multiple orders of magnitude. Symlog uncertainty conversion uses
linearized Jacobian (`exp(|μ_symlog|)`) — approximate near zero crossings.

### Supporting Modules

- **`optimizers/`** — OptunaOptimizer, Adam/HessianFree wrappers
- **`explainers/`** — SHAP-based feature importance
- **`utils/`** — metrics, data splitting, CV, losses, binning (percentile-based via `create_bins`), plotting
- **`preprocessing.py`** — OrderedTargetEncoder, OneHotEncoder
- **`enums.py`** — all enum types

## Key Dependencies

PyTorch >=2.6 (XPU), JAX, Optuna, XGBoost, LightGBM, CatBoost, SHAP, W&B.

## Session State

- **Next steps**: see `RESUME.md` (always loaded, keep concise)
- **Completed work, model history, bug audit**: see `ARCHIVE.md` (read when needed)
- **Research findings & past experiments**: see `docs/architecture_analysis.md` and `docs/implementation_plan.md` (read when needed)
- **Publication roadmap**: see `docs/research_plan.md` (two-paper structure, benchmarks, baselines, astrophysics applications)
- **Mathematical reference**: see `docs/mathematical_guide.tex` (compile with `pdflatex` × 3 passes; 24-page guide covering all models, losses, strategies, UQ methods)
