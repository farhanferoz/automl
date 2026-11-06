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

## Not in this build
Loss choice (Huber, quantile) beyond what exists, categorical encoding as a model setting, per-output weights for multi-output models,
target transforms beyond ProbReg's symlog: the next build, after this one is used. ProbReg/ClassReg/baselines supporting target scaling.

## Done when
Each phase is one local commit on `repo-reorganisation`, with its report section in /home/ff235/cache/automl/build_reports/standard_practice_build.md:
the identity checks above (all 0 / unchanged), the new tests and their mutation outcomes, ruff clean on changed lines.
