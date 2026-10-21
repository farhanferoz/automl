# Research documents — index

Every research stream has one folder and one plan (`PLAN.md`) that indexes its sub-plans. Rules that
bind every stream, the decision-register index and where data lives: `docs/RULES.md`.

| Stream | What it is | Status | Plan |
|---|---|---|---|
| ProbReg | the end-to-end classification-bottleneck regressor: fixed-k, global-k, per-input-k | **open — the current focus** | `docs/probreg/PLAN.md` |
| FlexNN | flexible architectures: width, depth, joint width-depth, mixture-of-experts, transformer port, the codebase | open, on hold since 2026-10-20 | `docs/flexnn/PLAN.md` |
| ClassReg | the two-stage classifier-then-mapper regressor | closed | `docs/classreg/PLAN.md` |
| Overfitting showcase | a demonstration of overfitting across the library's models | closed | `docs/overfitting_showcase/PLAN.md` |
| Applications | real data, photometric redshift, astrophysics | parked (toys only for now) | `docs/applications/PLAN.md` |
| Package | the AutoML library itself, its install and documentation | maintenance | `docs/package/PLAN.md` |

## Layout

Inside a stream: `PLAN.md`; one folder per sub-model or sub-plan, each with its `plan.md` and, as
needed, `specs/`, `reports/` and `history/`; `shared/` for work serving several sub-models;
`history/` for finished records; `archive/` for superseded plans, kept but never worked from.

Code stays in `automl_package/` (the library) and `automl_package/examples/` (experiment scripts).
Results from June–July 2026 stay in `automl_package/examples/capacity_ladder_results/`, whose
`RESULTS.md` records what completed. Each stream plan lists its own scripts and results folders.

## Gates

`docs/test_plan_gates.py` and `docs/test_plan_numbers.py` check every live plan: cited paths and line
numbers exist, no plan file carries a date in its name, result-marked lines point at a ledger JSON, and
numbers without a ledger pointer do not grow. Run:
`AUTOML_DEVICE=cpu ~/dev/.venv/bin/python -m pytest docs/test_plan_gates.py docs/test_plan_numbers.py -q`.
Exempt: future artifacts in `docs/gates_baseline.txt`, documents never gated in
`docs/gates_frozen.txt`, and the evidence folders (`archive/`, `history/`, `reports/`, `reviews_*/`).
