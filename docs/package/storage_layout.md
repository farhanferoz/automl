# Storage layout (approved by the owner, 2026-11-06)

Owner, 2026-11-06: "properly organized ... in a hierarchical manner", "results that we need to keep, or the ones that we
don't necessarily use on an ongoing basis, can put on to the NAS ... [Berkan] also has access to it ... We document it".

Principle: every file sits under project -> study -> experiment -> run, and within a run the kinds are kept apart
(inputs/config, raw outputs, logs, checkpoints, reports). One place per kind; nothing loose at a top level.

## 1. Machine-local working space (DGX, never synced)

```
~/cache/automl/
  envs/                       Python environments install.sh does not build (venv-dgx: reproduces the torch 2.10 ProbReg records)
  code/<study>/<run>/         frozen git worktrees that running or finished jobs execute from
  builds/gpu/                 GPU-native library builds (NCCL headers, CatBoost source build), one folder per library/version
  studies/
    probreg/
      runs/<card>/            (today: runs/C1..C4T, S0)
      review/                 (today: probreg_review/)
      smoke/                  (today: classreg_smoke/)
      archive/<date>_<what>/  (today: archive/review_2026-10-21)
  tools/<topic>/              helper and benchmark scripts (today: bench/, watchers/)
  reports/build/              build and verification reports of code changes (today: build_reports/)
  snapshots/                  rollback archives (pre_reorg_snapshot_2026-10-21.tar.zstd, ws_backup_2026-11-04)

~/cache/ae/                   NHS A&E work (repo /home/ff235/dev/ae)
  envs/                       venv, venv-lstm, venv-sf, venv-xgb3
  code/<run>/                 frozen worktrees (wt-best-effort-run, wt-round2-run, wt-lstm-noleak)
  experiments/<NN_name>/      one folder per numbered experiment, the same NN_name as its write-up and GitHub issue
    runs/<run_tag>/
      logs/  checkpoints/  hydra/
    smoke/                    smoke and timing runs that belong to this experiment
    analysis/                 scripts and outputs of the checks run on its results (e.g. outage_days.py)
  drafts/issues/              issue and write-up drafts (today: issues/)
  notes/                      working notes (covid_build_notes.md, ordered_target_notes.md, uv_setup_notes.md)
  patches/
```

Experiment folders and the runs that go in them (from the write-ups #2-#10):

| Experiment | Runs |
|---|---|
| 01_initial_setup | lightgbm_asis, lightgbm_noleak, catboost_asis/noleak (+ the GPU attempts), lstm_paper, lstm_paper_noleak, sarima_log |
| 02_post_period_features | the as-is vs no-leak pairs above (results only; same runs) |
| 03_lstm_dropout | dropout_test |
| 04_lightgbm_round2 | lgbm_r2_reference, lgbm_r2_covid_drop/phase/stringency, lgbm_r2_ordered_target |
| 06_setup_b | lightgbm_best, lstm_best |
| 07_poisson_deviance | lightgbm_best_poisson, lstm_best_poisson |
| 08_lstm_ordered_target | lstm_best_ordered |

## 2. NAS, shared with Berkan

```
<NAS share>/nhs-ae/
  README.md                   index: each folder, what produced it (repo, commit, config, command), how to load it
  experiments/<NN_name>/
    predictions/              pooled_predictions_*.parquet of each run
    metrics/                  fold and per-provider metrics CSVs
    tables/                   comparison tables and bootstrap CSVs
  archive/<date>_<what>/      finished material nobody uses day to day (old smoke runs, superseded runs)
```

Written locally first, copied to the NAS at the end (machine rule). OPEN: which NAS share Berkan can read; no share
exists for ae/automl today (exports: Container, Oasis, backup, media, tracecraft, neuroquill).

## 3. automl repo results (tracked summaries) — DONE 2026-11-06

```
results/probreg/                               output.results_root (default)
  cells/<card>.jsonl                           the plan of each card (5a1, 5a2, 5a3, 5b, 5b2, 5c1, 5c2, 5c4, 5c5)
  runs/<group>/<variant>/<toy>/n<size>/s<seed>.json   raw per-run records (+ .npz curves, .png figures)
  reports/<card label>/                        tables and pages (5c1+5c2+5c4+5c5 = the one table); reports/5a2/patience.json
```

`<group>` = `output.results_subdir` (default: the card). Cards 5b, 5b2 and 5c share the group `main`, so their reference
variant `probreg` is one set of records; 5a1, 5a2 and 5a3 keep their own groups because they reuse variant ids
(`probreg`, `plain_nn`) with different settings, which one folder would refuse (ConfigConflictError). The 5a3 group is
also the seed-noise source of every 5b/5c table (`report.noise_results_subdir: 5a3`). Code: `study.results_dir`,
`study.plan_path`, `study.layout_root`; reports are rebuilt by `python -m automl_package.experiments.report
results/probreg/cells/<card>.jsonl`. Checked at the move: every plan's cell count, settings hashes and finished/pending
counts per environment identical before and after; every rebuilt table identical apart from paths; 270 report links resolve.

Python environments cannot be moved (their scripts and pyvenv.cfg hold absolute paths): an environment outside `envs/`
moves only by being rebuilt there, e.g. `UV_PROJECT_ENVIRONMENT=<envs path> uv sync --frozen ...`.

## 4. Order of work

1. Now: items no running job uses (loose logs and notes, smoke folders, finished runs' logs and checkpoints, snapshots).
2. After LSTM jobs 1945-1947 end: their checkpoints, logs, hydra outputs, the frozen ae worktree and venv-lstm's place.
3. After approval: the repo's results/ move (code + config change) and the NAS copy for Berkan.
Every reference in RESUME, DECISIONS, DISK.md, scripts and the write-ups is updated in the same step as its move.
