# Standard practice as settings: plan (2026-11-06)

Owner, 2026-11-06: "We need to stop doing a 'build' for every simple task"; target normalisation "has to be centralised in the automl
repo not just for lstm"; "y" to planning the first build. Source: the audit /home/ff235/cache/automl/build_reports/standard_practice_audit.md
(rows 1-4 and 6). Pattern: target scaling, commit c761bac (one setting on BaseModel, applied once in `fit`, undone in `predict`, family
defaults as ClassVars, refusals that name the remedy, docstring saying which space each caller passes).

## Rules for every item
- One setting per practice, a constructor parameter of BaseModel (or PyTorchModelBase for network-only items), coerced from a string, an enum
  in `enums.py` for every closed set. Family defaults as ClassVars, like `_default_target_scaling` / `_supports_target_scaling`.
- Applied ONCE in BaseModel (fit / predict), never per `_fit_single` call site (21 implementations; super() chains).
- Statistics from the TRAINING part only (the caller's training rows under an explicit split).
- Identity: with every new setting off, results are bit-identical: the ProbReg replay (probreg, toy_a, n250, s0, venv-dgx, fingerprint
  4a65fc26) max abs diff 0; the cell census over results/probreg/cells/*.jsonl unchanged; the LSTM against 95ee1aa max abs diff 0.
  The toy-study runner (experiments/registry.py) does its own x/y standardisation, so it passes every new setting OFF explicitly.
- A setting a class cannot honour raises ValueError at construction or fit naming the remedy (never silently ignored).
- Tests folded into existing files; each new guard mutation-checked; the touched test files run once; the full suite once per phase only
  where a default changes behaviour.

## Phase 1: inputs (rows 1 and 3)
1. `input_scaling: InputScaling` (NONE, STANDARD). STANDARD: each numeric input column standardised by the training part's mean and sd
   (nan-aware, `utils/data_handler.standardiser_stats`, the function target scaling uses). Columns left alone: binary 0/1 columns, columns
   named in `categorical_features`, and an `offset_feature`. Fitted in `fit` after the train/val split and before feature selection; every
   internal path (search trials, CV folds, early stopping, refit, feature selection, SHAP background) gets scaled x; `predict` and every
   public method taking x (`predict_uncertainty`, feature importance) scale it once; `training_envelope_` stays on raw x.
   Defaults: PyTorchModelBase networks and linear models STANDARD; trees NONE; the LSTM refuses it (it standardises its sequence layout
   itself, lstm_model.py `_compute_input_stats`; column 0 is a series index).
   AS BUILT: ProbReg, FlexibleWidthNN and ClassReg also refuse STANDARD (ProbReg/FlexWidth `_predict_for_scoring` bypasses `predict`;
   ClassReg's classifier applies its own settings); the held-out scores take raw x from external callers and convert once.
2. `missing_inputs: MissingInputs` (RAISE, TRAINING_MEAN, NATIVE). RAISE: a NaN in x at fit or predict raises naming the remedy;
   TRAINING_MEAN: replaced by the training part's column mean (before scaling); NATIVE: passed through (the library handles it).
   Defaults: trees NATIVE; LSTM NATIVE (it fills with the training mean itself); networks and linear RAISE.

## Phase 2: training details (rows 4 and 6)
3. `time_gap: int = 0`: the number of distinct timestamps left out between training and validation, in BOTH the TIME_ORDERED train/val split
   (`utils/data_handler.create_train_val_split`) and the time-ordered CV folds (`utils/cv.TimeSeriesSplit`, today `margin = 0` at cv.py:67).
   Counted in distinct timestamps, not rows, so a panel (many series per date) gets a gap in time. Without timestamps it counts rows.
4. On PyTorchModelBase, used by BOTH training loops (base_pytorch.py:285 and lstm_model.py:331) through one shared helper:
   `grad_clip_norm: float | None` (moved up from the LSTM, which keeps its default 1.0; others default None = off);
   `lr_schedule: LRSchedule` (NONE, REDUCE_ON_PLATEAU on the held-out fit term, COSINE over the epoch budget) with
   `lr_schedule_factor`, `lr_schedule_patience`, `lr_warmup_epochs` (0). Defaults: NONE / off (they are tuning choices; bit-identical).
   AS BUILT (f65a576): the gap also applies before the test part (a refit on train + validation is scored there); TimeSeriesSplit orders rows
   by the timestamps it now receives (it assumed sorted rows); REDUCE_ON_PLATEAU is refused with refit_on_train_val=True (the refit has no
   held-out rows); the helper is `models/common/training.TrainingSchedule`; classes with their own loops refuse the settings.

## Phase 3: sample weights (row 2 of the gap list)
5. `fit(x, y, sample_weight=None)`: weights follow every split, fold and trial; the training loss is the weighted mean; the held-out fit term
   is the weighted mean of the per-row fit term (the tuning score stays the fit term of the trained loss). Supported: PyTorchModelBase
   default loop (plain network), LSTM (per row, broadcast over horizons, NaN targets still masked), XGBoost / LightGBM / CatBoost (native
   weights, incl. their early-stopping eval sets), normal-equation and PyTorch linear regression (weighted least squares). Every other
   class raises when weights are given (ProbReg, ClassReg, FlexNN, baselines) until mapped.
   AS BUILT: weights are relative (each array rescaled to mean 1); the training loss of a batch is sum(w*l)/sum(w) (PyTorch's convention);
   also refused: BINNED_RESIDUAL_STD (its calibration reads no weights) and LightGBM PROBABILISTIC (its NLL stopping metric reads none).
   Unweighted on purpose: scaling statistics, the in-sample residual spread, evaluation reports. Found and fixed on the way (81f4920): the
   normal-equation intercept was counted twice on uncentred inputs; the 5b linear cells were rerun (82d8598), 5a3's need the laptop.

## Column roles for every model (2026-11-06)
AS BUILT (owner ruling 2026-11-06: a feed-forward network must run on the same tabular inputs as LightGBM by config alone).
`offset_feature` (a column holding each row's starting score), `categorical_features` (integer category codes, NaN = missing) and the new
`categorical_encoding` (`enums.CategoricalEncoding`: NATIVE, ONE_HOT) are `BaseModel` settings. Boosting stays on the native route, unchanged
(`ColumnRolesMixin`: the library takes `init_score` / `base_margin` / `baseline` and splits on the codes; family default NATIVE). Every other
class takes the generic route (family default ONE_HOT), `models/common/column_roles.ColumnRolesTransform`, fitted at the top of `BaseModel.fit`:
the offset column is dropped and subtracted from y (`y_inner = y - offset`, broadcast over output columns, validation part from its own offset
column), each categorical column is replaced in place by one 0/1 column per level seen in the training rows named `<column>=<level>` (sorted
levels; a missing or unseen value is all zeros; levels come from all of x when `fit` draws its own validation split, from the training rows
when the caller gives `x_val`). The rest of `fit` runs unchanged on that inner problem: feature names are the inner names, input scaling
leaves 0/1 columns alone, the target scaler is fitted on the residual, trial clones carry `offset_feature=None, categorical_features=None`.
`column_roles_`, `input_feature_names_` and `training_envelope_` (recorded on the RAW columns and targets) are installed only when training is
over. After that the outermost public call reads the raw columns (a numpy x in `input_feature_names_` order): `predict` adds the offset after
the target-scaler inverse and before the range check; the held-out scores called from outside subtract it from y; `predict_uncertainty` takes no
offset; `predict_proba` / `predict_distribution` / `predict_quantiles` raise NotImplementedError on a model with an offset.
Refusals (ValueError naming the remedy): at construction, offset on a class that cannot take it (LSTM, ProbReg, ClassReg), categorical on the
LSTM, NATIVE on a non-native class, ONE_HOT on a boosting class, offset with a non-regression task, offset also listed as categorical; at the
start of `fit`, roles with feature importance or feature selection, a numpy x on the generic route, offset with a Poisson loss on the generic
route. HPO journal names do not change at default settings (the three settings are dropped from the identity at their defaults, except the two
roles for boosting, which always had them). Caveat: ONE_HOT keeps every level, so an unregularised normal-equation fit with an intercept is
singular (the one-hot columns of a column sum to 1): give `l2_lambda` > 0 there. Report: /home/ff235/cache/automl/build_reports/column_roles_every_model.md.

Output bound for the plain network (2026-11-06, 6765520): `PyTorchNeuralNetwork` takes `output_bound=TRAINING_RANGE` (the LSTM's smooth tanh bound to
the fit-space training targets, DECISIONS "Blow-up fix" (3) for neural models), for a regression mean only (classification and PROBABILISTIC refused).
The bound is fixed once per fit by `PyTorchModelBase._set_output_bound`, shared with the LSTM. Test: extreme inputs stay inside the training range with
target scaling off (its rounding clip in `predict` would hide the network's own bound); 3/3 mutations red.

Quantile input scaling (2026-11-06; owner: "it has to be centralised transform, properly included with other transforms & controlled by config"):
`input_scaling=QUANTILE`, a third option of the one input transform (`utils/data_handler.InputTransform`, after the missing-value rule, same
left-alone columns as STANDARD), set per model like every other setting. Each scaled column goes to a standard normal through scikit-learn's
`QuantileTransformer` (normal output, all training rows, no subsampling); a value beyond the training range maps to the extreme quantile. Settings
`input_quantiles` (default 1000) and `input_quantile_noise` (default 1e-3 x each column's sd, added only while the quantiles are fitted, seeded by
the model's seed). Why the noise: scikit-learn maps every value equal to a column's minimum to the lowest quantile (-5.2), so a column with 60% zeros
at its minimum sends 60% of rows to -5.2; with the noise they land at -0.53 (Gorishniy et al., NeurIPS 2021, add N(0, 1e-3) for the same reason).
Found on the NHS tabular inputs: 24 of 47 numeric columns have values beyond 10 sd after standardising (max 131). Classes that scale their own inputs
refuse every scaling, not only STANDARD. Test: 5/5 mutations red; journal names of 7 default-configured classes identical (guard break: 7/7 change).
Not centralised yet: the LSTM's own input standardisation and ae's log1p list on its sequence route.

## Not in this build
Loss choice (Huber, quantile) beyond what exists, ordered target statistics and learned embeddings as categorical encodings, per-output weights for multi-output models,
target transforms beyond ProbReg's symlog: the next build, after this one is used. ProbReg/ClassReg/baselines supporting target scaling.

## Done when
Each phase is one local commit on `repo-reorganisation`, with its report section in /home/ff235/cache/automl/build_reports/standard_practice_build.md:
the identity checks above (all 0 / unchanged), the new tests and their mutation outcomes, ruff clean on changed lines.
