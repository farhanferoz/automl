# ClassReg — stream plan

**Status: closed** (2026-10-21). Reopening it means writing its open work here.

The two-stage `ClassifierRegressionModel` (`automl_package/models/classifier_regression.py`): a
classifier over target bins is trained first and frozen, then a mapper
(`automl_package/models/mappers/`) turns its class probabilities into a regression output. It is not
one of the three ProbReg variants (`docs/probreg/PLAN.md`), which are trained end to end.

No plan or report was written for it. What exists:

| Script (`automl_package/examples/`) | Results (`automl_package/examples/`) | Last changed |
|---|---|---|
| `classifier_regression_nn_mapper_showcase.py` | — | 2026-04-18 |
| `classreg_k_sweep.py` | `classreg_k_sweep_results/` | committed 2026-07-13, with the ProbReg identifiability follow-up sweeps |
| `classreg_probability_sanity.py` | `classreg_probability_sanity_results/` | 2026-04-20 |
| `noise_robustness_benchmark.py` | `noise_robustness_results/` | 2026-04-20 |

Its place in the record: the April motivation that a three-class ClassReg beat pure regression (item
I4 in `docs/applications/research_plan.md`), which `noise_robustness_benchmark.py` was written to
check. The model code is live and tested (`tests/test_phase1_probabilistic_regression.py` imports it).
