# Config-driven experiment runner: design (2026-11-01)

Serves DECISIONS.md "Experiments are config changes, not code builds", "Nothing hard-coded; everything configurable",
"Later variants slot in without re-running" and "Design chosen by looking at ONE TABLE". Card definitions:
`docs/probreg/review/review_2026-10-22.md` §B as amended in `docs/probreg/review/readback_2026-10-22.md` §7.

Searched before designing: automl has no config system (113 one-off drivers in `automl_package/examples/`);
`/home/ff235/dev/ae/automl_adapter/config.py` holds the only Hydra structured configs touching automl (LSTM only,
per-model factory listing each setting by hand); models declare every setting in a `_defaults: ClassVar[dict]`
(`base_pytorch.py:32`, `probabilistic_regression.py` `_defaults`, `neural_network.py:70`); `BaseModel.fit` takes an
explicit validation set (`base.py:244-298`); `model.convergence_` (`utils/convergence.py:37`) carries the validation
trajectory, best epoch and cap hit; `examples/_toy_datasets.py` `STUDY_TOYS` gives each study toy's generator, true
mean and floor; `ProbabilisticRegressionModel.fit_sweep_selector`, `get_classifier_predictions`,
`plot_probability_mappers`, `get_num_parameters` exist.

## 1. Shape

One package, `automl_package/experiments/`:

| File | Holds |
|---|---|
| `config.py` | Typed structured configs (dataclasses + enums), registered in Hydra's `ConfigStore` |
| `registry.py` | `ModelName` -> model class; builds a model from a `ModelConfig` (generic, no per-model code) |
| `toys.py` | The study-toy table extended for the runner (noise sd, x range, regions, true bin probabilities) |
| `cell.py` | Runs ONE cell (one variant x toy x size x seed) and writes its record |
| `study.py` | Expands a study config into cells, skips cells already done, runs the rest in a process pool |
| `report.py` | Builds the results table(s) and the probability-curve figures from whatever records exist |
| `run.py` | `@hydra.main` entry point: `python -m automl_package.experiments.run +study=probreg_5b` |
| `conf/` | YAML: `config.yaml`; groups `model/`, `toy/`, `protocol/`, `hpo/`, `study/` |

A cell is the unit: everything about it is in its resolved config, and its record is keyed by
`(study, card, variant, toy, toy_params, n_train, seed)`. Nothing about a cell depends on which other cells exist,
except the two declared dependencies in §4 (matched parameter count; per-k sweep), which read the record of the
SAME cell of the variant they name.

## 2. Configs (all fields have defaults; every model setting is reachable)

- `ModelConfig`
  - `kind: PredictorKind` = `AUTOML` | `ORACLE` | `PLANTED_OFFSET` | `PLANTED_SCALE`.
  - `name: ModelName | None` (for `AUTOML`; `ModelName` gains `LSTM_FORECASTER`).
  - `params: dict[str, Any]` = any key of the class's merged `_defaults` (walk the MRO) or of `BaseModel.__init__`'s
    named arguments. Unknown key -> error naming the valid keys. A string given where the default is an `Enum`
    member is converted to that enum (by member name, then by value; failure -> error listing the members). Nested
    parameter dicts (`base_classifier_params`, `regression_head_params`, ...) pass through as dicts.
  - `tuned: bool`; `search_space: dict[str, SearchParamConfig]` (replaces or adds a dimension, passed as
    `search_space_override`); `search_space_exclude: list[str]` (removes dimensions; new `BaseModel` argument, §6).
  - `standardise_x: bool`, `standardise_y: bool` (training-set statistics; predictions mapped back; every metric in
    original y units). Networks true, trees and linear false.
  - `match_params_to: str | None` (§4), `k_sweep: KSweepConfig | None` (§4).
- `SearchParamConfig` (moved from ae): `type: ParamType` (`INT` | `FLOAT` | `CATEGORICAL`), `low`, `high`, `step`,
  `log`, `choices`. ae's adapter switches to this one.
- `ToyConfig`: `name: StudyToy`, `params: dict` (e.g. `jump: 1.0`), `n_train`, `n_val` (default `n_train // 4`),
  `n_selection` (10,000), `n_test` (10,000; heteroscedastic 20,000 in its own YAML).
- `ProtocolConfig`: `seeds: list[int]`, `early_stopping_patience`, `max_epochs`, `batch_size` (32),
  `refit_on_train_val` (false), `calculate_feature_importance` (false), `torch_threads` (1), `device` (`cpu`).
- `HpoConfig`: `n_trials` (20), `sampler: SamplerKind` (`TPE` | `RANDOM`), `seed`.
- `OutputConfig`: `results_dir` (default `/home/ff235/dev/MLResearch/automl/results/<study>`; small JSON + NPZ,
  tracked in git per docs/RULES.md), `curves: bool`, `curve_seeds: list[int]` (default: the first seed).
- `StudyConfig`: `name`, `card`, `variants: dict[str, VariantConfig]` (variant id -> model config + optional
  overrides), `toys: list[ToyConfig]`, `sizes: list[int]`, `reference_variant: str` (the default the table compares
  against), `eval_set: EvalSet` (`SELECTION` for 5c, `TEST` for 5b and 5d), `n_workers`.

Dropout off (DECISIONS): `conf/model/probreg.yaml` and `plain_nn.yaml` exclude every `*dropout_rate` dimension and set
the rates to 0.

## 3. One cell

1. Four draws from the toy's generator with disjoint integer seeds from `np.random.SeedSequence(seed).spawn(4)`
   (train, validation, selection, test); a SHA-256 of each set's bytes goes in the record (the slot-in guarantee:
   a later variant regenerates and checks the same draws).
2. Build the predictor (§2), torch threads set, `random_seed=seed`.
3. `fit(x_train, y_train, x_val=..., y_val=...)`, one pass, early stopping on the held-out fit term, best epoch
   restored, search on the validation split when `tuned`.
4. Predict on selection and test. Metrics per set, original units: MSE; MAE; excess = MSE - floor; noise-free excess
   = mean((prediction - true mean)^2); the same per region where the toy has regions.
5. Record (JSON): key, resolved config, git commit + dirty flag, data hashes, metrics, tuned hyperparameters,
   trainable parameter count, `convergence_` (trajectory, best epoch, stop epoch, converged, hit cap), wall seconds.
6. Curves (ProbReg variants, curve seeds only; NPZ + PNG): on 400 x points across the toy's range, (a) each class
   probability p_c(x) and the true P(y in bin c | x) = Phi((hi - m)/s) - Phi((lo - m)/s), outer edges infinite;
   (b) each head's output against its own probability, with the bin edges and the true E[y | y in bin c, x]
   (truncated-normal mean) plotted against the true p_c(x); single-head layouts plot head outputs against x.
   Card 5c-4 Part 1's two scores are computed from the same arrays on the selection inputs and stored, no threshold:
   out of order = sum_c p_c p_{c+1} 1[mu_c > mu_{c+1}] / sum_c p_c p_{c+1}; out of bin = sum_c p_c 1[mu_c outside bin c];
   each averaged over x, reported with k's parity.

## 4. Declared dependencies

- `match_params_to: <variant>`: plain network, one hidden layer, width chosen so its trainable parameter count is
  closest to the named variant's count in the same cell; learning rate and regularisation still tuned; both counts
  recorded.
- `k_sweep: {from_variant, k_values, rule}`: reuses the named variant's tuned hyperparameters from the same cell,
  calls `fit_sweep_selector` on (train, validation) with `rule` (`CHEAPEST_WITHIN_TOLERANCE` | `MINIMUM`, §6), scores
  the chosen model; records every k's validation error and the chosen k.

`study.py` runs independent variants' cells first, then dependent ones.

## 5. Report (the one table)

From every record under a study's results dir: one row per variant, one column per toy x size. Cell = mean over seeds
of the paired change in noise-free excess against `reference_variant` on the same draws, as a share of the reference's
noise-free excess (negative = better); beside it the min and max over seeds; greyed when the mean is smaller than the
5a seed-to-seed noise of the reference at that toy and size (from 5a's records); no automatic pick. Written as CSV,
Markdown and a static HTML page, plus the curve figures. 5a gets its own report: (a1) oracle and planted predictors
against their expected values with the per-draw standard error; (a2) validation trajectories; (a3) excess against n
per toy and model with the seed-to-seed spread.

## 6. Library additions this needs (outside the package)

- `BaseModel(search_space_exclude=...)`: drop listed dimensions where the search space is read (`base.py:477`,
  `classifier_regression.py:602`); unknown name -> error.
- `ModelName.LSTM_FORECASTER`.
- `fit_sweep_selector(..., rule: KSelectionRule)`, `KSelectionRule` in `enums.py`; per-k validation errors kept on
  the model.
- Card 5c-1's two-stage arm: `ProbabilisticRegressionOptimizationStrategy.TWO_STAGE`: classifier trained alone on
  class labels, stopped on held-out cross-entropy (best epoch restored), frozen; then the heads trained on detached
  probabilities, stopped on the held-out regression fit term (best epoch restored).
- `toys.py` extends `STUDY_TOYS` with noise sd per x, x range and regions, and adds two toys: `OU`
  (x ~ U(-1, 1), y = 0.95x + 0.112 eps, floor 0.112^2) and `JUMP` (x ~ U(0, 1), y = sin 2 pi x + J 1[x >= 0.6] + 0.3 eps,
  floor 0.09, `J` a toy parameter; study versions J = 0 and J = 1; region |x - 0.6| <= 0.05 against the rest).

## 7. Running

- **Plan, then execute.** `run.py` (Hydra) composes a study and writes every cell's fully resolved config to
  `<results_dir>/cells_<card>.jsonl`. `python -m automl_package.experiments.execute <cells.jsonl> --shard i/N
  --workers W` runs cells from that file and needs only automl's own dependencies (no Hydra at run time), so any
  machine with the package environment can execute.
- **Two machines.** The DGX (`/home/ff235/cache/automl/venv-dgx`, aarch64) and the part-time laptop `fedora`
  (x86_64, 16 cores lent; verified 2026-11-01: `~/dev` synced, `~/dev/.venv` has torch 2.10.0+xpu, optuna 4.9.0,
  lightgbm 4.6.0, scikit-learn 1.8.0, numpy 2.4.3; no Hydra). `automl_package/experiments/run_cells.sh` picks the
  interpreter by `uname -m`. Jobs go through `submit.py --class resume --anywhere` (a rerun skips finished cells;
  `--anywhere` because `~/dev` is Syncthing, not a network mount).
- **Pairs never cross environments.** The two machines differ in optuna (5.0.0 vs 4.9.0), lightgbm, scikit-learn and
  numpy, so a search can differ for the same cell. Every record carries an environment fingerprint (machine
  architecture + those versions) and a hash of the package source. The report pairs a variant only with reference
  records of the same fingerprint; when a card runs on a machine with no matching reference record, `plan` adds the
  reference variant's cells to that card. The data draws are identical on both (legacy NumPy seeding, hash-checked).
- Later variants (nested model, router, batched-GPU reruns) are new variant entries in a study YAML: their cells
  regenerate the same draws (hash-checked) and their records join the same table.
